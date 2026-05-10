"""
scoring/score_result.py
========================
ScoredSignal — The Output Contract of PatternScorer

Responsibility:
    Hold all scoring inputs, the computed edge_score, and the
    human-readable reasoning chain that produced it.
    Nothing else. No computation. No I/O. No side effects.

Mathematical guarantee (enforced in __post_init__):
    edge_score = base_score × state_discount × confidence_weight

    Where:
        base_score        ∈ [0.0, 1.0]   Fibonacci quality of the pattern
        state_discount    ∈ (0.0, 1.0]   Regime alignment (from config table)
        confidence_weight ∈ (0.0, 1.0]   Equals vector.confidence directly
        edge_score        ∈ [0.0, 1.0]   Their product — always, by construction

    The bound on edge_score is NOT enforced by clamping.
    It is a CONSEQUENCE of the bounds on its three factors.
    Three numbers each ≤ 1.0, multiplied, cannot exceed 1.0.
    This is provable from arithmetic, not from a min() call.

    If __post_init__ finds edge_score > 1.0, it means one of the
    three factors was > 1.0 — which is already caught by their own
    individual bound checks. The edge_score check is therefore
    a belt-and-suspenders final assertion, not the primary guard.

Why ValueError, not AssertionError:
    AssertionError communicates "this is a bug in our own code."
    ValueError communicates "a caller passed an invalid argument."
    ScoredSignal is constructed by PatternScorer (an external caller).
    If PatternScorer passes base_score=1.35, that is a caller error.
    Callers catch ValueError to handle gracefully (skip the signal).
    AssertionError would propagate and crash the pipeline.
    The rule: use ValueError for contract violations at boundaries,
    AssertionError only for internal logic invariants within a function.

Floating-point tolerance (_PRODUCT_TOLERANCE = 1e-10):
    edge_score is computed as base × discount × confidence.
    Three IEEE 754 double multiplications accumulate at most
    ~3.3e-16 of error (3 × 0.5 ULP for values in [0,1]).
    We use 1e-10, which is ~300,000× the theoretical maximum error.
    Any discrepancy > 1e-10 is a logic error (wrong formula),
    never floating-point noise. This tolerance is deliberately
    conservative — tight enough to catch real errors,
    loose enough to never produce false positives.

Immutability contract:
    This dataclass is NOT frozen (frozen=True) for one reason:
    PatternMatch (stored as pattern_match field) is also not frozen.
    A frozen dataclass cannot reliably contain a non-frozen one
    in all Python versions without surprising behavior.
    Immutability is enforced by the absence of mutation methods.
    No setter, no __setattr__ override. The object is treated as
    read-only by convention — which is sufficient for a data carrier
    in a single-threaded scan pipeline.

Dependencies:
    harmonic_patterns.PatternMatch    — source of base_score and PRZ
    market_state.vector.MarketStateVector — source of confidence
    typing, dataclasses               — stdlib only

    This file imports NOTHING from config/, signals/, telemetry/,
    or any other module in this project except the two listed above.
    This keeps the dependency graph clean and this module independently
    testable without standing up the full pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# The only two project imports this file is allowed to make.
# If you find yourself adding a third, move the logic to the caller.
from harmonic_patterns import PatternMatch
from market_state.vector import MarketStateVector


# ---------------------------------------------------------------------------
# Tolerance constant — explained in module docstring
# ---------------------------------------------------------------------------

_PRODUCT_TOLERANCE: float = 1e-10


# ---------------------------------------------------------------------------
# ScoredSignal
# ---------------------------------------------------------------------------

@dataclass
class ScoredSignal:
    """
    Immutable-by-convention record of a scored harmonic pattern.

    Construction contract (enforced in __post_init__):
        All three factor fields must be in their valid ranges.
        edge_score must equal base × discount × confidence
        within _PRODUCT_TOLERANCE.

    Field ordering rationale:
        Required fields without defaults come first (Python rule).
        Optional fields with defaults come after.
        Fields are grouped by concern:
            Group 1: source data (pattern + state)
            Group 2: scoring factors (the three inputs)
            Group 3: computed result (edge_score)
            Group 4: derived display data (dominant_state, reasoning)
            Group 5: PRZ trading levels (forwarded from PatternMatch.prz)
    """

    # ── Group 1: Source data ──────────────────────────────────────────────

    pattern_match:     PatternMatch
    """The harmonic pattern this score was computed for."""

    state_vector:      MarketStateVector
    """The market state at the time of scoring. Snapshot — not live."""

    # ── Group 2: Scoring factors ──────────────────────────────────────────

    base_score:        float
    """
    Fibonacci ratio quality score from the harmonic engine.
    Measures how closely the pattern's ratios match ideal values.
    Range: [0.0, 1.0]
    Source: PatternMatch.quality_score
    Meaning: 1.0 = perfect Fibonacci geometry, 0.0 = no match.
    """

    state_discount:    float
    """
    Regime alignment discount from the state/pattern multiplier table.
    Measures how well the current market environment suits this pattern.
    Range: (0.0, 1.0]
    Source: MS_CONFIG.get_discount(state, pattern)
    Meaning: 1.0 = ideal environment, 0.05 = hostile environment.
    Cannot be 0.0 — preserves multiplicative structure (Axiom 4).
    """

    confidence_weight: float
    """
    State classification certainty, taken directly from vector.confidence.
    Scales the state_discount: uncertain state → smaller state effect.
    Range: (0.0, 1.0]
    Source: state_vector.confidence (which is max(state_probs.values()))
    Meaning: 1.0 = certain state classification, 0.25 = quite uncertain.
    Not a threshold — used as a continuous weight, not a gate.
    """

    # ── Group 3: Computed result ──────────────────────────────────────────

    edge_score:        float
    """
    Final conviction scalar.
    MUST equal base_score × state_discount × confidence_weight
    within _PRODUCT_TOLERANCE.
    Range: [0.0, 1.0] — guaranteed by arithmetic, not by clamping.
    Meaning: "deploy this fraction of max_risk_pct on this setup."
    Usage: maps directly to SignalTier threshold classification.
    """

    # ── Group 4: Derived display data ─────────────────────────────────────

    dominant_state:    str
    """
    The dominant market state at scoring time.
    Snapshot of state_vector.dominant_state.
    Stored flat for fast access in logging and Telegram delivery.
    """

    reasoning:         List[str] = field(default_factory=list)
    """
    Human-readable explanation chain.
    Each element is one display line in the Telegram message.
    Must contain at least 3 lines if edge_score >= 0 (enforced).
    Order: pattern description → state description →
           discount → confidence → final score.
    """

    # ── Group 5: PRZ trading levels ───────────────────────────────────────
    # Forwarded from PatternMatch.prz for flat, type-safe access.
    # Using explicit float fields instead of Dict[str, float] because:
    #   - Dict has no compile-time key guarantees
    #   - Missing keys silently return None from .get()
    #   - Explicit fields are self-documenting and IDE-completable

    entry:   float = 0.0
    """Limit entry price at the D pivot."""

    stop:    float = 0.0
    """Stop loss price. Beyond X for retracement patterns."""

    target1: float = 0.0
    """Conservative target — B level (first structural level)."""

    target2: Optional[float] = None
    """Moderate target — A level. None if pattern does not define it."""

    target3: Optional[float] = None
    """Aggressive target — Fibonacci extension. None if not defined."""

    risk_reward: Optional[float] = None
    """
    Risk:Reward ratio to target1.
    None when stop == entry (degenerate PRZ — no valid R:R computable).
    Forwarded from PatternMatch.risk_reward at construction.
    """

    # ── __post_init__ — all invariant enforcement ─────────────────────────

    def __post_init__(self) -> None:
        """
        Validates all mathematical invariants after construction.

        Raises ValueError for any violation.
        Validation order: individual factor bounds first,
        cross-field product check last.

        This ordering matters: if base_score=1.35, we report
        "base_score out of range" rather than the less informative
        "edge_score does not equal product." The first violation
        found is reported. Subsequent checks are skipped.
        """
        self._validate_base_score()
        self._validate_state_discount()
        self._validate_confidence_weight()
        self._validate_edge_score_range()
        self._validate_product_consistency()
        self._validate_dominant_state()
        self._validate_reasoning()
        self._validate_prz_fields()

    # ── Individual validators ─────────────────────────────────────────────

    def _validate_base_score(self) -> None:
        """
        base_score ∈ [0.0, 1.0]

        0.0 is allowed: a degenerate pattern (all ratios off) can
        legitimately score zero. It will produce edge_score=0.0
        and be filtered before tiering.

        > 1.0 is never valid: quality_score is a normalized closeness
        metric, not an unbounded index. If base_score > 1.0 reaches
        here, the harmonic engine has a bug — surface it immediately.
        """
        if not (0.0 <= self.base_score <= 1.0):
            raise ValueError(
                f"base_score={self.base_score!r} is outside [0.0, 1.0]. "
                f"base_score is the Fibonacci ratio quality from the harmonic "
                f"engine and must be in [0.0, 1.0]. "
                f"A value > 1.0 indicates a bug in quality_score computation. "
                f"A value < 0.0 indicates a sign error."
            )

    def _validate_state_discount(self) -> None:
        """
        state_discount ∈ (0.0, 1.0]

        > 0.0 is mandatory (strictly): zero would make edge_score = 0
        regardless of pattern quality or confidence. This violates
        Axiom 4 (near-zero floor) and destroys multiplicative structure.
        The config enforces 0.05 minimum; this check is belt-and-suspenders.

        ≤ 1.0 is mandatory: > 1.0 would amplify base_score, violating
        Axiom 3 (state context cannot exceed full credit).
        """
        if not (0.0 < self.state_discount <= 1.0):
            raise ValueError(
                f"state_discount={self.state_discount!r} is outside (0.0, 1.0]. "
                f"state_discount is the regime alignment multiplier from the "
                f"harmonic_multipliers config table. "
                f"Axiom 3: values > 1.0 would amplify beyond pattern quality ceiling. "
                f"Axiom 4: value must be > 0.0 to preserve multiplicative structure. "
                f"Check MS_CONFIG.get_discount() or the fallback_discount setting."
            )

    def _validate_confidence_weight(self) -> None:
        """
        confidence_weight ∈ (0.0, 1.0]

        This field is MarketStateVector.confidence, which is
        max(state_probs.values()). After SimplexProjector normalization,
        the minimum possible confidence is PROB_FLOOR (0.02) when all
        states are equal. It cannot be 0.0 in practice.

        We enforce > 0.0 here as belt-and-suspenders against manually
        constructed MarketStateVectors with all-zero probabilities —
        which would make edge_score = 0.0 silently for all signals.

        Cannot exceed 1.0: it is a probability.
        """
        if not (0.0 < self.confidence_weight <= 1.0):
            raise ValueError(
                f"confidence_weight={self.confidence_weight!r} is outside (0.0, 1.0]. "
                f"confidence_weight must equal MarketStateVector.confidence, "
                f"which is the maximum state probability and therefore in (0.0, 1.0]. "
                f"A value of 0.0 would suppress all signals regardless of quality. "
                f"This can occur if MarketStateVector was manually constructed with "
                f"all-zero state probabilities — use the engine, not direct construction."
            )

    def _validate_edge_score_range(self) -> None:
        """
        edge_score ∈ [0.0, 1.0]

        Primary: product of three [0,1]-bounded values cannot exceed 1.0.
        This check should never trigger if the three factor checks pass.
        It is the final guard against unforeseen floating-point scenarios.

        If this triggers AFTER the three factor checks pass, something
        deeply unexpected happened in the multiplication — log it.
        """
        if not (0.0 <= self.edge_score <= 1.0):
            raise ValueError(
                f"edge_score={self.edge_score!r} is outside [0.0, 1.0]. "
                f"This should be mathematically impossible if base_score, "
                f"state_discount, and confidence_weight are each in their "
                f"valid ranges. Their individual checks passed, which means "
                f"an unexpected floating-point overflow or underflow occurred. "
                f"Factors: base={self.base_score}, discount={self.state_discount}, "
                f"confidence={self.confidence_weight}."
            )

    def _validate_product_consistency(self) -> None:
        """
        edge_score ≈ base_score × state_discount × confidence_weight

        Tolerance: _PRODUCT_TOLERANCE = 1e-10

        Why 1e-10 and not exact equality?
            Three IEEE 754 doubles multiplied accumulate at most ~3.3e-16
            of floating-point error. Using exact equality would produce
            rare false positives on certain compilers/platforms.
            1e-10 is ~300,000× the theoretical maximum floating-point error.
            Any discrepancy > 1e-10 is unambiguously a logic error
            (caller used a different formula), not arithmetic noise.

        What this check catches:
            PatternScorer accidentally using addition instead of multiplication.
            Caller passing pre-scaled values that don't match the formula.
            Off-by-one errors in formula transcription.

        What this check does NOT catch:
            Numerically wrong but formula-correct values — that is the
            responsibility of the individual factor validators above.
        """
        expected = self.base_score * self.state_discount * self.confidence_weight
        delta    = abs(self.edge_score - expected)
        if delta > _PRODUCT_TOLERANCE:
            raise ValueError(
                f"edge_score={self.edge_score!r} does not equal "
                f"base_score × state_discount × confidence_weight = "
                f"{self.base_score!r} × {self.state_discount!r} × "
                f"{self.confidence_weight!r} = {expected!r}. "
                f"Delta={delta:.2e} exceeds tolerance={_PRODUCT_TOLERANCE:.0e}. "
                f"PatternScorer must compute edge_score as the direct product "
                f"of these three factors — no intermediate rounding, "
                f"no clamping, no alternative formula."
            )

    def _validate_dominant_state(self) -> None:
        """
        dominant_state must be a non-empty string.

        We do NOT validate against a fixed set of state names here,
        because new states may be added in future versions.
        Restricting to a hardcoded set would make ScoredSignal fragile.
        The config validator enforces valid state names at the config level.
        """
        if not isinstance(self.dominant_state, str) or not self.dominant_state:
            raise ValueError(
                f"dominant_state={self.dominant_state!r} must be a non-empty string. "
                f"Set dominant_state = state_vector.dominant_state before "
                f"constructing ScoredSignal."
            )

    def _validate_reasoning(self) -> None:
        """
        reasoning must be a list of strings.
        If edge_score > 0, it must contain at least 3 lines.

        We require 3 lines minimum because a useful reasoning chain must
        explain at minimum: (1) what the pattern is, (2) what the state is,
        (3) what the final score is. Fewer lines means the PatternScorer
        failed to build a complete chain.

        If edge_score == 0.0, reasoning may be empty: the signal will
        not be delivered, so a full reasoning chain is not required.
        """
        if not isinstance(self.reasoning, list):
            raise ValueError(
                f"reasoning must be a list, got {type(self.reasoning).__name__}."
            )
        if not all(isinstance(line, str) for line in self.reasoning):
            raise ValueError(
                "All elements of reasoning must be strings."
            )
        if self.edge_score > 0.0 and len(self.reasoning) < 3:
            raise ValueError(
                f"reasoning has {len(self.reasoning)} line(s) but must have "
                f"at least 3 when edge_score > 0.0. "
                f"A complete reasoning chain requires: pattern description, "
                f"market state description, and final edge score explanation."
            )

    def _validate_prz_fields(self) -> None:
        """
        PRZ trading levels must be non-negative.

        Zero is allowed: some pattern types may have entry=0.0 in test
        scenarios. Negative prices are never valid.

        risk_reward, target2, target3 can be None — that is a valid
        state (not all patterns define all levels, and degenerate PRZs
        produce None risk_reward from PatternMatch.risk_reward).
        """
        for name, value in [
            ("entry",   self.entry),
            ("stop",    self.stop),
            ("target1", self.target1),
        ]:
            if value < 0.0:
                raise ValueError(
                    f"{name}={value!r} is negative. "
                    f"All PRZ price levels must be non-negative."
                )

        if self.risk_reward is not None and self.risk_reward < 0.0:
            raise ValueError(
                f"risk_reward={self.risk_reward!r} is negative. "
                f"risk_reward must be non-negative or None."
            )

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def is_actionable(self) -> bool:
        """
        True when edge_score is above zero and PRZ has valid trading levels.
        Does NOT check tier thresholds — that is SignalTier's responsibility.

        Used by PatternScorer to decide whether to build a full reasoning
        chain before constructing ScoredSignal.
        """
        return (
            self.edge_score > 0.0
            and self.entry  > 0.0
            and self.stop   > 0.0
            and self.target1 > 0.0
        )

    @property
    def passes_min_threshold(self) -> bool:
        """
        True when edge_score >= min_edge_score from config.

        Importing MS_CONFIG here is the one concession to config access.
        It is a property (computed on access), not stored state, so it
        does not create a circular dependency at construction time.
        """
        from config.market_state_config import MS_CONFIG
        return self.edge_score >= MS_CONFIG.min_edge_score

    def summary(self) -> str:
        """
        Single-line human-readable summary for logging.

        Example:
            ScoredSignal [Gartley BULLISH | BTCUSDT 1h] |
            base=0.84 discount=0.80 conf=0.72 edge=0.484 → Tier A
        """
        tier_hint = self._tier_hint()
        return (
            f"ScoredSignal [{self.pattern_match.pattern_name} "
            f"{self.pattern_match.direction.upper()} | "
            f"{self.pattern_match.symbol} {self.pattern_match.timeframe}] | "
            f"base={self.base_score:.3f} "
            f"discount={self.state_discount:.3f} "
            f"conf={self.confidence_weight:.3f} "
            f"edge={self.edge_score:.4f} → {tier_hint}"
        )

    def _tier_hint(self) -> str:
        """Returns an estimated tier label for display. Not authoritative."""
        from config.market_state_config import MS_CONFIG
        for tier, threshold, _, _ in MS_CONFIG.tier_rules():
            if self.edge_score >= threshold:
                return f"Tier {tier}"
        return "Below threshold"

    def __repr__(self) -> str:
        return self.summary()