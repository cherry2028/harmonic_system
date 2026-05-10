"""
scoring/pattern_scorer.py
==========================
PatternScorer — Axiomatic Edge Score Computation

Single responsibility:
    Given a PatternMatch and a MarketStateVector, compute a ScoredSignal
    whose edge_score satisfies the approved mathematical framework:

        edge_score = base_score × state_discount × confidence_weight

    where:
        base_score        = PatternMatch.quality_score          ∈ [0.0, 1.0]
        state_discount    = table[dominant_state][pattern_name] ∈ (0.0, 1.0]
        confidence_weight = vector.confidence                   ∈ (0.0, 1.0]
        edge_score                                              ∈ [0.0, 1.0]

    The bound is a consequence of arithmetic, not enforcement.
    No clamping. No min(). No max(). Pure multiplication.

Mathematical guarantees (enforced by ScoredSignal.__post_init__):
    1. Boundedness:  edge_score ∈ [0.0, 1.0]        (Axiom 1)
    2. Monotonicity: higher base → higher edge        (Axiom 2)
    3. State ceiling: discount ≤ 1.0 always           (Axiom 3)
    4. Chaos near-zero: chaos discount = 0.05         (Axiom 4)
    5. Product consistency: |edge - base×disc×conf| ≤ 1e-10

Failure modes and how each is handled:
    base_score > 1.0 or < 0.0:
        Log ERROR. Return None. Do not construct ScoredSignal.
        A corrupted base propagates corruption downstream — surface it.

    Missing (state, pattern) key in discount table:
        Log WARNING. Use MS_CONFIG.fallback_discount (0.50).
        Conservative but not zero — preserves multiplicative structure.

    confidence_weight == 0.0:
        Cannot happen in production (SimplexProjector floor=0.02).
        If it does: ScoredSignal construction raises ValueError.
        score() catches it, logs ERROR, returns None.

    Any other exception in score():
        Log ERROR with traceback. Return None.
        Callers handle None by skipping the pattern.

Never-raise contract:
    score()       returns ScoredSignal | None. Never raises.
    score_batch() returns List[ScoredSignal]. Never raises.

Reasoning chain (5 lines minimum, built only when edge > min_edge_score):
    Line 1: Pattern identity and base quality
    Line 2: Market state and confidence
    Line 3: State discount and interpretation
    Line 4: Confidence weight
    Line 5: Final edge score and tier estimate

    Reasoning is NOT built for below-threshold signals.
    Building 5 formatted strings for a signal that will not be
    delivered wastes ~5µs × 51,840 calls/day = 259ms/day.
    Negligible, but the principle matters: don't compute what
    you won't use.

Dependencies:
    harmonic_patterns.PatternMatch      — base_score source
    market_state.vector.MarketStateVector — confidence source
    scoring.score_result.ScoredSignal   — output contract
    config.market_state_config.MS_CONFIG — discount table + thresholds
"""

from __future__ import annotations

import logging
from typing import List, Optional

from harmonic_patterns import PatternMatch
from market_state.vector import MarketStateVector
from scoring.score_result import ScoredSignal
from config.market_state_config import MS_CONFIG

logger = logging.getLogger("scoring.pattern_scorer")


# ---------------------------------------------------------------------------
# Discount interpretation strings
# Deterministic: same discount value → same string. Always.
# Used in reasoning line 3 and Telegram delivery.
# ---------------------------------------------------------------------------

def _discount_interpretation(discount: float) -> str:
    """
    Maps a state_discount value to a human-readable quality label.

    Thresholds:
        ≥ 0.90 : "ideal conditions"      (reversal row = 1.00)
        ≥ 0.70 : "favorable"             (ranging row = 0.80)
        ≥ 0.45 : "neutral"               (compression row = 0.55)
        ≥ 0.25 : "unfavorable"           (trending row = 0.30)
        <  0.25: "hostile"               (news_chaos row = 0.05)

    Monotonic: lower discount → less favorable label. Always.
    """
    if discount >= 0.90:
        return "ideal conditions"
    if discount >= 0.70:
        return "favorable"
    if discount >= 0.45:
        return "neutral"
    if discount >= 0.25:
        return "unfavorable"
    return "hostile"


# ---------------------------------------------------------------------------
# PatternScorer
# ---------------------------------------------------------------------------

class PatternScorer:
    """
    Stateless scorer. Thread-safe for concurrent score() calls.

    Instantiate once per process. All configuration is read at
    __init__ time and cached as instance attributes.

    Args:
        config: MarketStateConfig to use. Defaults to MS_CONFIG singleton.
                Inject a custom config for testing.
    """

    def __init__(self, config=None) -> None:
        self._config = config or MS_CONFIG
        self._validate_config_on_init()
        logger.debug(
            f"PatternScorer initialized | "
            f"fallback_discount={self._config.fallback_discount} | "
            f"min_edge_score={self._config.min_edge_score}"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def score(
        self,
        match:  PatternMatch,
        vector: MarketStateVector,
    ) -> Optional[ScoredSignal]:
        """
        Computes edge_score for one (match, vector) pair.

        Returns:
            ScoredSignal if scoring succeeds and edge >= 0.0
            None         if any input is invalid or scoring fails

        Never raises. All exceptions caught and logged.
        """
        try:
            return self._compute(match, vector)
        except Exception as e:
            logger.error(
                f"PatternScorer.score() raised unexpectedly: "
                f"{type(e).__name__}: {e}. "
                f"pattern={getattr(match, 'pattern_name', '?')} "
                f"symbol={getattr(match, 'symbol', '?')}. "
                f"Returning None.",
                exc_info=True,
            )
            return None

    def score_batch(
        self,
        matches: List[PatternMatch],
        vector:  MarketStateVector,
    ) -> List[ScoredSignal]:
        """
        Scores a list of PatternMatch objects against one vector.

        Returns:
            List of successful ScoredSignals, sorted by edge_score
            descending (highest conviction first). None results are
            filtered out silently — each failure was already logged.

        Never raises.
        """
        if matches is None:
            logger.warning('score_batch received None matches list — returning []')
            return []
        results = []
        for match in matches:
            scored = self.score(match, vector)
            if scored is not None:
                results.append(scored)

        # Sort highest edge first for downstream priority
        results.sort(key=lambda s: s.edge_score, reverse=True)
        logger.debug(
            f"score_batch: {len(matches)} inputs → "
            f"{len(results)} scored | "
            f"symbol={getattr(vector, 'symbol', '?')} "
            f"tf={getattr(vector, 'timeframe', '?')}"
        )
        return results

    # ── Core computation ──────────────────────────────────────────────────

    def _compute(
        self,
        match:  PatternMatch,
        vector: MarketStateVector,
    ) -> Optional[ScoredSignal]:
        """
        Executes the five-step scoring pipeline.

        Steps:
            1. Validate and extract base_score
            2. Resolve state_discount from config table
            3. Extract confidence_weight from vector
            4. Compute edge_score as direct product
            5. Build reasoning chain (if above threshold)
            6. Construct and return ScoredSignal
        """

        # ── Step 1: base_score ────────────────────────────────────────
        base_score = self._extract_base_score(match)
        if base_score is None:
            return None     # already logged in _extract_base_score

        # ── Step 2: state_discount ────────────────────────────────────
        state_discount, discount_source = self._resolve_discount(
            vector.dominant_state,
            match.pattern_name,
        )

        # ── Step 3: confidence_weight ─────────────────────────────────
        confidence_weight = vector.confidence
        # vector.confidence is a property on a frozen dataclass.
        # It cannot be 0.0 in production (SimplexProjector floor=0.02
        # makes minimum confidence = 0.02).
        # We log a warning but do not block — ScoredSignal will
        # enforce the > 0.0 invariant and raise ValueError if violated,
        # which _compute()'s caller (score()) catches and returns None.
        if confidence_weight <= 0.0:
            logger.warning(
                f"vector.confidence={confidence_weight:.6f} is <= 0.0. "
                f"This should not happen with a properly constructed "
                f"MarketStateVector. ScoredSignal will reject this."
            )

        # ── Step 4: edge_score ────────────────────────────────────────
        # Direct product. No rounding. No clamping.
        # Axiom 1: product of three [0,1] values cannot exceed 1.0.
        edge_score = base_score * state_discount * confidence_weight

        logger.debug(
            f"Scoring | "
            f"{match.pattern_name} {match.direction} | "
            f"{match.symbol} {match.timeframe} | "
            f"base={base_score:.4f} × "
            f"discount={state_discount:.4f} ({discount_source}) × "
            f"conf={confidence_weight:.4f} = "
            f"edge={edge_score:.6f}"
        )

        # ── Step 5: Reasoning chain ───────────────────────────────────
        # Build only for above-threshold signals.
        # Below-threshold signals are not delivered — no reasoning needed.
        if edge_score >= self._config.min_edge_score:
            reasoning = self._build_reasoning(
                match           = match,
                vector          = vector,
                base_score      = base_score,
                state_discount  = state_discount,
                discount_source = discount_source,
                confidence_weight = confidence_weight,
                edge_score      = edge_score,
            )
        else:
            # Three-line minimal chain satisfies ScoredSignal contract
            # (≥3 lines when edge > 0, empty when edge == 0).
            # edge_score > 0 is possible even below min_edge_score,
            # e.g. base=0.10 × discount=0.05 × conf=0.30 = 0.0015 > 0.
            if edge_score > 0.0:
                reasoning = [
                    f"Pattern: {match.pattern_name} {match.direction}",
                    f"Market: {vector.dominant_state}",
                    f"Edge score: {edge_score:.4f} (below delivery threshold "
                    f"{self._config.min_edge_score})",
                ]
            else:
                reasoning = []

        # ── Step 6: Construct ScoredSignal ────────────────────────────
        # ScoredSignal.__post_init__ enforces all mathematical invariants.
        # If construction raises ValueError, it propagates to score()
        # which catches it and returns None.
        return ScoredSignal(
            pattern_match     = match,
            state_vector      = vector,
            base_score        = base_score,
            state_discount    = state_discount,
            confidence_weight = confidence_weight,
            edge_score        = edge_score,
            dominant_state    = vector.dominant_state,
            reasoning         = reasoning,
            entry             = match.prz.get("entry",   0.0),
            stop              = match.prz.get("stop",    0.0),
            target1           = match.prz.get("target1", 0.0),
            target2           = match.prz.get("target2"),
            target3           = match.prz.get("target3"),
            risk_reward       = match.risk_reward,
        )

    # ── Step implementations ──────────────────────────────────────────────

    def _extract_base_score(
        self, match: PatternMatch
    ) -> Optional[float]:
        """
        Extracts and validates quality_score from PatternMatch.

        Returns float in [0.0, 1.0] on success, None on failure.

        Why return None rather than a default?
            A corrupted base_score (e.g. 1.35 from a buggy scorer)
            must not silently produce a valid-looking ScoredSignal.
            Returning None surfaces the bug immediately.
        """
        base = match.quality_score

        if not isinstance(base, (int, float)):
            logger.error(
                f"quality_score={base!r} is not a number "
                f"(got {type(base).__name__}). "
                f"pattern={match.pattern_name}. Skipping."
            )
            return None

        base = float(base)

        if not (0.0 <= base <= 1.0):
            logger.error(
                f"quality_score={base:.6f} is outside [0.0, 1.0]. "
                f"pattern={match.pattern_name} symbol={match.symbol}. "
                f"This indicates a bug in the harmonic engine. "
                f"Skipping this pattern."
            )
            return None

        return base

    def _resolve_discount(
        self, dominant_state: str, pattern_name: str
    ) -> tuple[float, str]:
        """
        Resolves state_discount from the config table.

        Returns:
            (discount, source_description)
            source_description is used in reasoning and debug logs.

        Fallback behavior:
            Unknown state or pattern → fallback_discount (0.50).
            Logged at WARNING — not an error (new states/patterns may
            be added), but operator should be aware.

        Why return source_description?
            The reasoning chain must explain whether the discount
            came from the table or was a fallback. Transparency.
        """
        if not dominant_state:
            logger.warning(
                f"dominant_state is empty. "
                f"Using fallback_discount={self._config.fallback_discount}."
            )
            return self._config.fallback_discount, "fallback(empty state)"

        table_discount = self._config.harmonic_multipliers.get(
            dominant_state, {}
        ).get(pattern_name)

        if table_discount is None:
            logger.warning(
                f"No discount found for "
                f"state={dominant_state!r} pattern={pattern_name!r}. "
                f"Using fallback_discount={self._config.fallback_discount}."
            )
            return (
                self._config.fallback_discount,
                f"fallback(unknown: {dominant_state}/{pattern_name})",
            )

        return table_discount, f"table({dominant_state}/{pattern_name})"

    def _build_reasoning(
        self,
        match:             PatternMatch,
        vector:            MarketStateVector,
        base_score:        float,
        state_discount:    float,
        discount_source:   str,
        confidence_weight: float,
        edge_score:        float,
    ) -> list:
        """
        Constructs the five-line reasoning chain for Telegram delivery.

        All lines are deterministic: same inputs → same output. Always.
        No randomness, no timestamps, no external state.

        Format per line:
            Line 1: Pattern identity and Fibonacci quality
            Line 2: Market state, confidence
            Line 3: State discount value, source, interpretation
            Line 4: Confidence weight applied
            Line 5: Final edge score and tier estimate

        The reasoning chain is the transparency contract with subscribers.
        Every line must stand alone — a subscriber reading only one line
        should understand what it means.
        """
        tier_hint = self._tier_hint(edge_score)
        interpretation = _discount_interpretation(state_discount)

        return [
            # Line 1: What is this pattern?
            (
                f"Pattern: {match.pattern_name} {match.direction.upper()} "
                f"on {match.symbol} {match.timeframe} | "
                f"Fibonacci quality: {base_score:.0%}"
            ),
            # Line 2: What is the market doing?
            (
                f"Market state: {vector.dominant_state.upper()} "
                f"({confidence_weight:.0%} confident)"
            ),
            # Line 3: How does the state affect this pattern?
            (
                f"State discount: {state_discount:.2f}× — {interpretation} "
                f"for {match.pattern_name} in {vector.dominant_state} "
                f"[{discount_source}]"
            ),
            # Line 4: How much does confidence scale the state evidence?
            (
                f"Confidence weight: {confidence_weight:.2f}× "
                f"(state classification certainty)"
            ),
            # Line 5: Final score
            (
                f"Edge score: {edge_score:.0%} → {tier_hint}"
            ),
        ]

    def _tier_hint(self, edge_score: float) -> str:
        """
        Returns estimated tier label for the reasoning chain.
        Not authoritative — SignalTier is authoritative.
        Used only for display in reasoning line 5.
        """
        for tier, threshold, _, _ in self._config.tier_rules():
            if edge_score >= threshold:
                return f"Tier {tier}"
        return "Below threshold"

    # ── Init validation ───────────────────────────────────────────────────

    def _validate_config_on_init(self) -> None:
        """
        Validates the injected config satisfies the scorer's requirements.

        Runs once at __init__ — not per call.
        Catches misconfigured custom configs before they silently
        produce wrong scores.
        """
        cfg = self._config

        # All discounts must be in (0, 1] — same as MarketStateConfig validates,
        # but we re-check here because a custom config might bypass that validation.
        for state, patterns in cfg.harmonic_multipliers.items():
            for pattern, value in patterns.items():
                if not (0.0 < value <= 1.0):
                    raise ValueError(
                        f"PatternScorer: config.harmonic_multipliers"
                        f"[{state!r}][{pattern!r}] = {value} "
                        f"violates (0.0, 1.0] contract. "
                        f"Axiom 3: state discounts cannot amplify or be zero."
                    )

        # fallback_discount must be in (0, 1]
        if not (0.0 < cfg.fallback_discount <= 1.0):
            raise ValueError(
                f"PatternScorer: config.fallback_discount={cfg.fallback_discount} "
                f"must be in (0.0, 1.0]."
            )