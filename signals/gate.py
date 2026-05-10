"""
signals/gate.py
================
HostileMarketGate — Pipeline Circuit Breaker

Architectural position:
    Called at the TOP of every scan cycle, before swing detection,
    pattern matching, or scoring. When blocked, zero downstream
    compute runs. This is a circuit breaker, not a quality filter.

Single responsibility:
    Answer one question: "Is the market environment so hostile
    that running the downstream pipeline is wasteful and potentially
    harmful right now?"

Never-raise contract:
    check() NEVER raises to the caller. Three protection layers:
        Layer 1: Input validation — None/bad vector → PASS + WARNING
        Layer 2: Per-rule try/except — rule failure → skip + WARNING
        Layer 3: Outer try/except — unhandled → PASS + CRITICAL

Safe default is PASS, not BLOCK:
    A false block (blocking a real signal) harms subscribers silently.
    A false pass lets the scorer and tier system evaluate further.
    When uncertain: let the pipeline run.

The four rules (evaluated in priority order, first match wins):
    Rule 1 — NEWS_CHAOS
        vector.news_chaos >= config.gate_chaos_threshold
        News invalidates all technical analysis. Highest priority.

    Rule 2 — COMPRESSION
        vector.compression >= config.gate_compression_threshold
        Patterns form but entries never trigger. Wasteful to detect.

    Rule 3 — LOW_CONFIDENCE
        vector.confidence < config.gate_confidence_threshold
        State classifier too uncertain. State discounts unreliable.

    Rule 4 — PURE_EXPANSION
        vector.expansion >= config.gate_pure_expansion_thresh
        AND vector.reversal < 0.20
        Pure momentum breakout without reversal evidence.
        The AND is critical: expansion + reversal = Butterfly/Crab zone.
        Only fires when reversal evidence is absent.

Why the gate does NOT call vector.is_hostile() internally:
    is_hostile() is a convenience method on the vector — correct
    for quick checks. The gate re-implements the conditions explicitly
    to own its block_code assignment, reason strings, and audit trail.
    Readability and auditability take priority over DRY here.
    Someone debugging a block reads gate.py — not vector.py.

Dependencies:
    market_state.vector.MarketStateVector  — the only input
    config.market_state_config.MS_CONFIG   — threshold values

    MUST NOT import from:
        delivery/, signals/signal.py, signals/tier.py,
        scoring/, patterns/, harmonic_*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import FrozenSet, Optional

from market_state.vector import MarketStateVector

logger = logging.getLogger("signals.gate")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All valid block_code values. Fixed set — adding a rule requires
# adding a code here AND in the rule list. Intentional coupling
# to make structural additions explicit.
_VALID_BLOCK_CODES: FrozenSet[str] = frozenset({
    "NEWS_CHAOS",
    "COMPRESSION",
    "LOW_CONFIDENCE",
    "PURE_EXPANSION",
    "PASS",
})

# Reversal threshold for Rule 4.
# Below this reversal probability in a high-expansion state
# = pure momentum breakout = hostile for harmonic reversals.
# Not in config because it is a structural constant of the rule
# definition, not an operational threshold to tune.
_EXPANSION_REVERSAL_FLOOR: float = 0.20


# ---------------------------------------------------------------------------
# GateResult
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """
    Complete record of one gate evaluation.

    Carries enough information to:
        1. Drive the pipeline decision     (is_blocked)
        2. Record why in telemetry         (block_code)
        3. Explain to a human              (reason)
        4. Reproduce the decision          (vector snapshot)

    Invariants enforced in __post_init__:
        1. block_code is always a valid constant from _VALID_BLOCK_CODES
        2. is_blocked=True  → block_code != "PASS"
        3. is_blocked=False → block_code == "PASS"
        4. reason is always a non-empty string
        5. is_blocked=True  → vector is not None
           (a block without a vector is an undebugable audit trail)

    Immutability:
        Not frozen — contains MarketStateVector which is frozen,
        but GateResult itself need not be. Callers treat it as
        read-only by convention.
    """

    is_blocked:  bool
    block_code:  str
    reason:      str
    vector:      Optional[MarketStateVector]

    def __post_init__(self) -> None:
        self._validate_block_code()
        self._validate_code_blocked_consistency()
        self._validate_reason()
        self._validate_vector_on_block()

    # ── Validators ────────────────────────────────────────────────────

    def _validate_block_code(self) -> None:
        """
        block_code must be one of the five defined constants.

        Not a string equality check — membership in the frozen set.
        This means adding a new rule without updating _VALID_BLOCK_CODES
        raises ValueError immediately at construction time, not silently
        at telemetry-query time.
        """
        if self.block_code not in _VALID_BLOCK_CODES:
            raise ValueError(
                f"block_code={self.block_code!r} is not a valid code. "
                f"Must be one of {sorted(_VALID_BLOCK_CODES)}. "
                f"If a new blocking rule was added, update "
                f"_VALID_BLOCK_CODES in signals/gate.py."
            )

    def _validate_code_blocked_consistency(self) -> None:
        """
        Enforces the logical consistency between is_blocked and block_code.

        is_blocked=True with block_code="PASS" is a contradiction:
            "We blocked this signal because it passed all checks."
        is_blocked=False with block_code="NEWS_CHAOS" is a contradiction:
            "We did not block this signal because there was news chaos."

        Both are logic errors in the gate implementation.
        """
        if self.is_blocked and self.block_code == "PASS":
            raise ValueError(
                f"Contradiction: is_blocked=True but block_code='PASS'. "
                f"A blocked result must have a non-PASS block_code. "
                f"This is a bug in HostileMarketGate rule construction."
            )
        if not self.is_blocked and self.block_code != "PASS":
            raise ValueError(
                f"Contradiction: is_blocked=False but block_code={self.block_code!r}. "
                f"A passing result must have block_code='PASS'. "
                f"This is a bug in HostileMarketGate rule construction."
            )

    def _validate_reason(self) -> None:
        """
        reason must be a non-empty string.

        An empty reason string produces silent log entries:
            "Gate blocked BTCUSDT 1h: "
        That is useless for debugging. Enforce non-empty always.
        This applies to PASS results too — the reason for passing
        should be stated explicitly.
        """
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(
                f"reason={self.reason!r} must be a non-empty string. "
                f"Every GateResult must carry an explanatory reason, "
                f"including PASS results."
            )

    def _validate_vector_on_block(self) -> None:
        """
        A blocked result must carry the vector that caused the block.

        Without the vector, a block is undebugable:
        "BTCUSDT was blocked at 14:00 UTC" tells you nothing.
        "BTCUSDT was blocked: news_chaos=0.62 >= threshold=0.40" is actionable.

        PASS results may have vector=None only in the degenerate case
        where input validation failed (bad vector passed to check()).
        That case is logged at WARNING by the gate — the None is intentional.
        """
        if self.is_blocked and self.vector is None:
            raise ValueError(
                f"is_blocked=True but vector=None. "
                f"A blocked GateResult must preserve the vector "
                f"that caused the block for audit trail purposes. "
                f"This is a bug in HostileMarketGate._make_block()."
            )

    # ── Convenience properties ─────────────────────────────────────────

    @property
    def passed(self) -> bool:
        """Inverse of is_blocked. For readable pipeline conditionals."""
        return not self.is_blocked

    def summary(self) -> str:
        """
        Single-line summary for log messages.

        Examples:
            GateResult [PASS] All gate checks passed
            GateResult [BLOCK:NEWS_CHAOS] news_chaos=0.62 >= threshold 0.40
        """
        if self.is_blocked:
            label = f"BLOCK:{self.block_code}"
        else:
            label = "PASS"
        return f"GateResult [{label}] {self.reason}"

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# HostileMarketGate
# ---------------------------------------------------------------------------

class HostileMarketGate:
    """
    Pipeline circuit breaker for hostile market conditions.

    Stateless after construction. Thread-safe for concurrent check() calls
    on different (symbol, timeframe) pairs — all configuration is read-only.

    Usage:
        gate   = HostileMarketGate()
        result = gate.check(vector)

        if result.is_blocked:
            telemetry.log_gate_block(...)
            return None   # skip remainder of scan cycle

    Configuration:
        All thresholds read from MS_CONFIG at construction time.
        Storing them as instance attributes at __init__ avoids
        repeated attribute lookups on a module-level singleton
        across 576 scan cycles/day.
    """

    def __init__(self) -> None:
        # Lazy import: avoids circular imports if gate.py is imported
        # before config is fully initialized in test environments.
        from config.market_state_config import MS_CONFIG

        # Cache thresholds as instance attributes for fast per-call access
        self._chaos_threshold:       float = MS_CONFIG.gate_chaos_threshold
        self._compression_threshold: float = MS_CONFIG.gate_compression_threshold
        self._confidence_threshold:  float = MS_CONFIG.gate_confidence_threshold
        self._expansion_threshold:   float = MS_CONFIG.gate_pure_expansion_thresh

        logger.debug(
            f"HostileMarketGate initialized | "
            f"chaos>={self._chaos_threshold} | "
            f"compression>={self._compression_threshold} | "
            f"confidence<{self._confidence_threshold} | "
            f"expansion>={self._expansion_threshold}"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def check(self, vector: Optional[MarketStateVector]) -> GateResult:
        """
        Evaluates all gate rules against the vector.

        Never raises. Returns GateResult always.

        Args:
            vector: The MarketStateVector to evaluate.
                    None input → PASS with WARNING (safe default).

        Returns:
            GateResult with is_blocked=True if any rule fires,
            GateResult with is_blocked=False (PASS) otherwise.
        """
        try:
            return self._run_checks(vector)
        except Exception as e:
            # Outer catch: should never reach here — each rule has its
            # own try/except. If it does: log CRITICAL, return PASS.
            # PASS is the safe default (let downstream evaluate).
            logger.critical(
                f"HostileMarketGate.check() raised unexpectedly: "
                f"{type(e).__name__}: {e}. "
                f"Returning PASS as safe default. "
                f"This is a bug — please investigate.",
                exc_info=True,
            )
            return self._make_pass(
                reason=(
                    f"Gate evaluation failed with {type(e).__name__}. "
                    f"Returning PASS as safe default. See CRITICAL log."
                ),
                vector=vector if isinstance(vector, MarketStateVector) else None,
            )

    # ── Internal pipeline ─────────────────────────────────────────────────

    def _run_checks(
        self, vector: Optional[MarketStateVector]
    ) -> GateResult:
        """
        Runs input validation then evaluates all four rules in order.
        First matching rule returns immediately — no further evaluation.
        """
        # ── Input validation ──────────────────────────────────────────
        if vector is None:
            logger.warning(
                "HostileMarketGate.check() received None vector. "
                "Returning PASS as safe default. "
                "Check the data fetcher and state engine for errors."
            )
            return self._make_pass(
                reason="No vector provided — PASS by default (input validation failure).",
                vector=None,
            )

        if not isinstance(vector, MarketStateVector):
            logger.warning(
                f"HostileMarketGate.check() received unexpected type: "
                f"{type(vector).__name__}. Expected MarketStateVector. "
                f"Returning PASS as safe default."
            )
            return self._make_pass(
                reason=(
                    f"Unexpected input type {type(vector).__name__} — "
                    f"PASS by default (input validation failure)."
                ),
                vector=None,
            )

        # ── Rule evaluation ───────────────────────────────────────────
        # Each rule is wrapped in try/except.
        # A rule that raises is SKIPPED (not treated as a block).
        # Skipping preserves pipeline continuity — one broken rule
        # should not silence all remaining rules.

        for rule_fn in [
            self._rule_news_chaos,
            self._rule_compression,
            self._rule_low_confidence,
            self._rule_pure_expansion,
        ]:
            try:
                result = rule_fn(vector)
                if result is not None:
                    # Rule fired — log and return immediately
                    logger.info(
                        f"Gate BLOCKED | "
                        f"{vector.symbol} {vector.timeframe} | "
                        f"{result.summary()}"
                    )
                    return result
            except Exception as e:
                logger.warning(
                    f"Gate rule {rule_fn.__name__} raised "
                    f"{type(e).__name__}: {e}. "
                    f"Skipping rule — continuing evaluation."
                )
                continue

        # ── All rules passed ──────────────────────────────────────────
        logger.debug(
            f"Gate PASSED | "
            f"{vector.symbol} {vector.timeframe} | "
            f"dominant={vector.dominant_state} "
            f"conf={vector.confidence:.2f}"
        )
        return self._make_pass(
            reason=(
                f"All gate checks passed | "
                f"dominant={vector.dominant_state} "
                f"({vector.confidence:.0%} conf)"
            ),
            vector=vector,
        )

    # ── Gate rules ────────────────────────────────────────────────────────
    # Each rule returns:
    #     GateResult  — if the rule fires (block this scan)
    #     None        — if the rule does not apply (continue evaluation)
    #
    # Rules are pure functions of (self._thresholds, vector).
    # No side effects. No logging (logging happens in _run_checks).

    def _rule_news_chaos(
        self, vector: MarketStateVector
    ) -> Optional[GateResult]:
        """
        Rule 1: NEWS_CHAOS
        Fires when: vector.news_chaos >= chaos_threshold.
        Priority: HIGHEST — evaluated first.

        Rationale: News/external shocks invalidate all technical analysis.
        No pattern geometry is meaningful during chaos.
        """
        if vector.news_chaos >= self._chaos_threshold:
            return self._make_block(
                block_code="NEWS_CHAOS",
                reason=(
                    f"News/external shock detected: "
                    f"news_chaos={vector.news_chaos:.3f} "
                    f">= threshold {self._chaos_threshold} — "
                    f"all technical setups invalidated"
                ),
                vector=vector,
            )
        return None

    def _rule_compression(
        self, vector: MarketStateVector
    ) -> Optional[GateResult]:
        """
        Rule 2: COMPRESSION
        Fires when: vector.compression >= compression_threshold.
        Priority: SECOND.

        Rationale: During strong compression, harmonic patterns form
        geometrically but entries never trigger cleanly. The coil
        has not released. Running detection produces unactionable signals
        and wastes compute.
        """
        if vector.compression >= self._compression_threshold:
            return self._make_block(
                block_code="COMPRESSION",
                reason=(
                    f"Market coiling — entries unreliable: "
                    f"compression={vector.compression:.3f} "
                    f">= threshold {self._compression_threshold} — "
                    f"wait for directional breakout"
                ),
                vector=vector,
            )
        return None

    def _rule_low_confidence(
        self, vector: MarketStateVector
    ) -> Optional[GateResult]:
        """
        Rule 3: LOW_CONFIDENCE
        Fires when: vector.confidence < confidence_threshold.
        Priority: THIRD.

        Rationale: If the state classifier cannot identify the market
        regime, the state discounts applied by PatternScorer are
        unreliable. A poorly-calibrated state multiplier produces
        edge_scores that do not reflect real conditions.

        Note: confidence = max(state_probs.values()). After SimplexProjector
        normalization with PROB_FLOOR=0.02, minimum possible confidence
        is 0.02 (all states equal). Default threshold is 0.25.
        """
        if vector.confidence < self._confidence_threshold:
            return self._make_block(
                block_code="LOW_CONFIDENCE",
                reason=(
                    f"State classification too uncertain: "
                    f"confidence={vector.confidence:.3f} "
                    f"< floor {self._confidence_threshold} — "
                    f"state discounts unreliable"
                ),
                vector=vector,
            )
        return None

    def _rule_pure_expansion(
        self, vector: MarketStateVector
    ) -> Optional[GateResult]:
        """
        Rule 4: PURE_EXPANSION
        Fires when: vector.expansion >= expansion_threshold
                    AND vector.reversal < _EXPANSION_REVERSAL_FLOOR (0.20)
        Priority: FOURTH (most nuanced — evaluated last).

        Rationale: Pure volatility expansion without reversal evidence
        means momentum is dominant. Harmonic reversal entries get
        steamrolled by the momentum move.

        The AND is critical:
            expansion=0.85, reversal=0.30 → DO NOT block
                (reversal evidence present — could be Butterfly/Crab zone)
            expansion=0.85, reversal=0.05 → BLOCK
                (pure breakout, no reversal signal — hostile for harmonics)

        _EXPANSION_REVERSAL_FLOOR=0.20 is a structural constant, not
        a config threshold. It represents "meaningful reversal evidence"
        above the SimplexProjector floor.
        """
        if (
            vector.expansion >= self._expansion_threshold
            and vector.reversal < _EXPANSION_REVERSAL_FLOOR
        ):
            return self._make_block(
                block_code="PURE_EXPANSION",
                reason=(
                    f"Volatility explosion without reversal signal: "
                    f"expansion={vector.expansion:.3f} "
                    f">= threshold {self._expansion_threshold}, "
                    f"reversal={vector.reversal:.3f} "
                    f"< floor {_EXPANSION_REVERSAL_FLOOR} — "
                    f"momentum dominates, harmonic reversals likely to fail"
                ),
                vector=vector,
            )
        return None

    # ── Result factories ──────────────────────────────────────────────────
    # Centralised construction of GateResult objects.
    # All GateResult construction goes through these two methods —
    # never constructed directly in rule functions.
    # This ensures _VALID_BLOCK_CODES validation always runs.

    @staticmethod
    def _make_block(
        block_code: str,
        reason:     str,
        vector:     MarketStateVector,
    ) -> GateResult:
        """
        Constructs a blocked GateResult.
        GateResult.__post_init__ validates all invariants.
        """
        return GateResult(
            is_blocked = True,
            block_code = block_code,
            reason     = reason,
            vector     = vector,
        )

    @staticmethod
    def _make_pass(
        reason: str,
        vector: Optional[MarketStateVector],
    ) -> GateResult:
        """
        Constructs a passing GateResult.
        Used for: all-rules-passed, None input, bad-type input,
        and the outer exception handler.
        """
        return GateResult(
            is_blocked = False,
            block_code = "PASS",
            reason     = reason,
            vector     = vector,
        )