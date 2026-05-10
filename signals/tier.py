"""
signals/tier.py
================
SignalTier — Tier Classification and TieredSignal Construction

Single responsibility:
    Accept a ScoredSignal. Assign it a tier. Build a TieredSignal
    ready for delivery. Return None if the signal does not qualify.

Architectural position:
    SignalTier sits between the scoring layer and the delivery layer.
    It is the final operational gate before a signal becomes a
    deployment decision.

    ScoredSignal (scoring layer)
        ↓
    SignalTier.classify()
        ↓
    TieredSignal (operational layer) or None

What SignalTier decides:
    - Which tier does this signal belong to?
    - What risk allocation applies?
    - What is the daily frequency cap?
    - Is the signal complete enough to deliver?

What SignalTier does NOT decide:
    - Whether the market is hostile (HostileMarketGate)
    - What the edge_score is (PatternScorer)
    - How trading levels are computed (harmonic engine)

Classification rules (from config, strictly decreasing thresholds):
    edge >= 0.70  → Tier A+  (max 1/day, 2.0% risk)
    edge >= 0.50  → Tier A   (max 3/day, 1.0% risk)
    edge >= 0.30  → Tier B   (max 5/day, 0.5% risk)
    edge >= 0.10  → Tier C   (max 99/day, 0.0% risk — paper only)
    edge <  0.10  → None     (below threshold, no signal produced)

Threshold semantics: inclusive on the lower bound.
    edge == 0.70 → Tier A+  (not Tier A)
    edge == 0.50 → Tier A   (not Tier B)

The two threshold layers:
    PatternScorer.min_edge_score (0.08):
        Whether to build a reasoning chain in ScoredSignal.
        Below this: ScoredSignal.reasoning == []

    SignalTier.tier_c_threshold (0.10):
        Whether to produce any TieredSignal at all.
        Below this: classify() returns None.

    These are different gates at different layers.
    A ScoredSignal with edge=0.09 exists (scorer produced it),
    but SignalTier returns None (no tier assigned, not delivered).

Reasoning chain extension:
    SignalTier appends two lines to scored.reasoning:
        Line N+1: tier assignment with threshold justification
        Line N+2: risk allocation and frequency cap

    This preserves the scoring reasoning chain and adds the
    operational context a subscriber sees in the Telegram message.

Daily counter (TODO — Week 2 Phase 2):
    The daily frequency cap (max_per_day) is stored in TieredSignal
    but enforcement requires a DailyCounter that does not yet exist.
    This class documents the exact injection contract.

    TODO Week 2 Phase 2: inject DailyCounter dependency here.
        DailyCounter contract:
            check(tier: str, symbol: str) -> bool
                True  = cap not yet reached, may deliver
                False = cap reached, suppress this signal
            increment(tier: str, symbol: str) -> None
                Record one delivery against the daily cap.
            reset() -> None
                Called at UTC midnight to clear all counters.
        Thread safety: DailyCounter must be thread-safe.
        Persistence: counters must survive scanner restarts
                     within the same UTC day (file or Redis).
        Injection: SignalTier.__init__(counter=None) — optional.
                   When None, cap enforcement is skipped (current behavior).
                   When provided, cap is enforced per (tier, symbol).

Never-raise contract:
    classify()  returns TieredSignal | None. Never raises.
    Any exception is caught, logged as ERROR, returns None.

Dependencies:
    scoring.score_result.ScoredSignal   — input contract
    signals.signal.TieredSignal         — output contract
    config.market_state_config.MS_CONFIG — tier rules

    MUST NOT import from:
        market_state/, patterns/, harmonic_*, delivery/, telemetry/
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from scoring.score_result import ScoredSignal
from signals.signal import TieredSignal

logger = logging.getLogger("signals.tier")


# ---------------------------------------------------------------------------
# ClassificationResult — internal intermediate
# ---------------------------------------------------------------------------

class _TierLookup:
    """
    Internal helper. Encapsulates the tier-assignment lookup.

    Not part of the public API. Used only by SignalTier._assign_tier().

    Separating the lookup from the TieredSignal construction makes
    each individually testable and keeps classify() readable.
    """

    __slots__ = ("tier", "threshold", "max_per_day", "risk_pct")

    def __init__(
        self,
        tier:        str,
        threshold:   float,
        max_per_day: int,
        risk_pct:    float,
    ) -> None:
        self.tier        = tier
        self.threshold   = threshold
        self.max_per_day = max_per_day
        self.risk_pct    = risk_pct

    def __repr__(self) -> str:
        return (
            f"_TierLookup(tier={self.tier!r} "
            f"threshold={self.threshold} "
            f"max_per_day={self.max_per_day} "
            f"risk_pct={self.risk_pct})"
        )


# ---------------------------------------------------------------------------
# SignalTier
# ---------------------------------------------------------------------------

class SignalTier:
    """
    Tier classifier and TieredSignal factory.

    Stateless after construction (ignoring the optional future
    DailyCounter). Thread-safe for concurrent classify() calls.

    Usage:
        tier   = SignalTier()
        result = tier.classify(scored_signal)

        if result is None:
            return   # below threshold — nothing to deliver

        # result is a TieredSignal — send to delivery layer

    Args:
        config: MarketStateConfig. Defaults to MS_CONFIG singleton.
                Inject a custom config for testing.
    """

    def __init__(self, config=None) -> None:
        from config.market_state_config import MS_CONFIG
        self._config = config or MS_CONFIG
        self._rules  = self._build_rules()
        self._validate_rules()
        logger.debug(
            f"SignalTier initialized | "
            f"tiers={[r.tier for r in self._rules]} | "
            f"thresholds={[r.threshold for r in self._rules]}"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def classify(self, scored: ScoredSignal) -> Optional[TieredSignal]:
        """
        Classifies a ScoredSignal and constructs a TieredSignal.

        Returns:
            TieredSignal if edge_score >= tier_c_threshold
            None         if below threshold, invalid input, or error

        Never raises. All exceptions caught and logged.
        """
        try:
            return self._run_classification(scored)
        except Exception as e:
            logger.error(
                f"SignalTier.classify() raised unexpectedly: "
                f"{type(e).__name__}: {e}. "
                f"pattern={getattr(getattr(scored, 'pattern_match', None), 'pattern_name', '?')} "
                f"edge={getattr(scored, 'edge_score', '?')}. "
                f"Returning None.",
                exc_info=True,
            )
            return None

    # ── Internal pipeline ─────────────────────────────────────────────────

    def _run_classification(
        self, scored: ScoredSignal
    ) -> Optional[TieredSignal]:
        """
        Core classification pipeline.

        Steps:
            1. Input validation
            2. Tier assignment
            3. Reasoning chain extension
            4. TieredSignal construction
        """

        # ── Step 1: Input validation ──────────────────────────────────
        if scored is None:
            logger.debug("classify() received None — returning None")
            return None

        if not isinstance(scored, ScoredSignal):
            logger.warning(
                f"classify() received unexpected type: "
                f"{type(scored).__name__}. Expected ScoredSignal. "
                f"Returning None."
            )
            return None

        # ── Step 2: Tier assignment ───────────────────────────────────
        lookup = self._assign_tier(scored.edge_score)

        if lookup is None:
            # Below tier_c_threshold — no tier, no signal
            logger.debug(
                f"edge_score={scored.edge_score:.4f} below "
                f"tier_C threshold={self._config.tier_c_threshold} — "
                f"no tier assigned"
            )
            return None

        logger.info(
            f"Tier assigned: {lookup.tier} | "
            f"edge={scored.edge_score:.4f} >= threshold={lookup.threshold} | "
            f"{scored.pattern_match.symbol} {scored.pattern_match.timeframe}"
        )

        # ── Step 3: Reasoning chain extension ────────────────────────
        # Preserve the full scoring reasoning chain and append two
        # operational lines: tier assignment + sizing/cap.
        reasoning = self._extend_reasoning(scored.reasoning, lookup)

        # ── Step 4: Construct TieredSignal ────────────────────────────
        # TieredSignal.__post_init__ enforces all invariants.
        # If construction fails (e.g. entry=0.0 from a degenerate PRZ),
        # the ValueError propagates to classify() which returns None.
        return TieredSignal(
            tier            = lookup.tier,
            max_per_day     = lookup.max_per_day,
            risk_pct        = lookup.risk_pct,
            scored          = scored,
            edge_score      = scored.edge_score,
            dominant_state  = scored.dominant_state,
            reasoning       = reasoning,
            entry           = scored.entry,
            stop            = scored.stop,
            target1         = scored.target1,
            target2         = scored.target2,
            target3         = scored.target3,
            risk_reward     = scored.risk_reward,
        )

    # ── Tier assignment ───────────────────────────────────────────────────

    def _assign_tier(self, edge_score: float) -> Optional[_TierLookup]:
        """
        Scans rules from highest to lowest threshold.
        Returns the first rule whose threshold the score meets.
        Returns None if score is below the lowest tier threshold.

        Threshold semantics: inclusive lower bound.
            edge == 0.70 → Tier A+  (>= 0.70, first rule that matches)
            edge == 0.30 → Tier B   (>= 0.30)
            edge == 0.10 → Tier C   (>= 0.10)
            edge == 0.09 → None     (< 0.10, below Tier C)

        Linear scan is O(4) — constant, negligible.
        """
        for rule in self._rules:
            if edge_score >= rule.threshold:
                return rule
        return None

    # ── Reasoning extension ───────────────────────────────────────────────

    def _extend_reasoning(
        self,
        base_reasoning: list,
        lookup:         _TierLookup,
    ) -> list:
        """
        Appends two operational lines to the scoring reasoning chain.

        Line N+1: Tier assignment with threshold justification.
        Line N+2: Risk allocation and frequency cap.

        Why two lines?
            Subscribers see: what tier and why (line N+1),
            then what it means for position size and frequency (line N+2).
            These are operationally distinct — do not merge them.

        The base_reasoning may be:
            - A full 5-line chain (above min_edge_score)
            - Empty (edge=0.0, but this path is unreachable here
              since _assign_tier would have returned None for edge=0.0
              as it falls below all tier thresholds)

        Returns a new list — never mutates base_reasoning.
        """
        paper_note = " (paper only — no real capital)" if lookup.risk_pct == 0.0 else ""

        tier_line = (
            f"Tier {lookup.tier} assigned: "
            f"edge {lookup.threshold:.0%} threshold met"
        )

        sizing_line = (
            f"Risk: {lookup.risk_pct}% of capital{paper_note} | "
            f"Frequency cap: max {lookup.max_per_day}/day"
        )

        return list(base_reasoning) + [tier_line, sizing_line]

    # ── Init helpers ──────────────────────────────────────────────────────

    def _build_rules(self) -> list:
        """
        Converts config.tier_rules() into _TierLookup objects.
        Preserves the ordering from config (highest threshold first).
        """
        return [
            _TierLookup(
                tier        = tier,
                threshold   = threshold,
                max_per_day = max_daily,
                risk_pct    = risk_pct,
            )
            for tier, threshold, max_daily, risk_pct
            in self._config.tier_rules()
        ]

    def _validate_rules(self) -> None:
        """
        Validates that rules are strictly decreasing at construction.

        The classification algorithm assumes this ordering.
        If config produces rules in the wrong order, fail loudly at
        startup — not silently during a live scan.
        """
        if not self._rules:
            raise ValueError(
                "SignalTier: tier_rules() returned empty list. "
                "At least one tier must be configured."
            )

        for i in range(len(self._rules) - 1):
            hi = self._rules[i]
            lo = self._rules[i + 1]
            if hi.threshold <= lo.threshold:
                raise ValueError(
                    f"SignalTier: tier rules not strictly decreasing. "
                    f"tier={hi.tier!r} threshold={hi.threshold} must be "
                    f"> tier={lo.tier!r} threshold={lo.threshold}. "
                    f"Check MarketStateConfig tier threshold ordering."
                )

        logger.debug(
            f"Tier rules validated: "
            f"{[(r.tier, r.threshold) for r in self._rules]}"
        )