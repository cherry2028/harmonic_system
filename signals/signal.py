"""
signals/signal.py
==================
TieredSignal — An Approved Deployment Decision

Architectural position:
    This is the final data object before Telegram delivery.
    It represents a signal that has passed ALL gates:
        ✓ Market state classification (Layer 1)
        ✓ Harmonic pattern detection (Layer 2)
        ✓ Axiomatic edge scoring (PatternScorer)
        ✓ Minimum threshold filter (min_edge_score)
        ✓ Tier classification (SignalTier)

Why TieredSignal exists separately from ScoredSignal:
    ScoredSignal answers: "How good is this pattern mathematically?"
    TieredSignal answers: "Has this signal been approved for deployment,
                           and under exactly what operational constraints?"

    These are different questions at different abstraction layers.
    A ScoredSignal can have edge_score=0.0 (valid — filtered out).
    A TieredSignal with edge_score=0.0 is a contradiction — it
    would be a "deployed signal with no conviction," which is
    operationally nonsensical and structurally forbidden.

    Keeping them separate ensures:
        - Scoring changes never touch this file
        - Delivery policy changes never touch score_result.py
        - The tier/risk/cap fields can evolve independently
          of the edge_score computation

Invariants enforced in __post_init__:
    1. tier is a valid, known string ("A+", "A", "B", "C")
    2. edge_score matches scored.edge_score exactly (no divergence)
    3. edge_score >= min_edge_score (deployment threshold enforced)
    4. risk_pct in [0.0, _MAX_RISK_PCT] (institutional hard cap)
    5. max_per_day >= 1 (zero would permanently suppress delivery)
    6. entry > 0, stop > 0, target1 > 0 (live trading levels required)
    7. stop != entry (degenerate PRZ — no valid trade possible)
    8. reasoning is non-empty (transparency contract for subscribers)
    9. All reasoning lines are strings (display safety)

Why ValueError, not AssertionError (same reasoning as ScoredSignal):
    TieredSignal is constructed by SignalTier (a caller).
    Invalid arguments from callers → ValueError.
    Internal logic bugs → AssertionError (inside functions only).
    Callers catch ValueError to skip bad signals gracefully.

Dependencies:
    scoring.score_result.ScoredSignal  — the scored source signal
    typing, dataclasses                — stdlib only

    This file imports NOTHING from config/, market_state/,
    telemetry/, patterns/, or harmonic_*.
    The only config dependency is a lazy import of MS_CONFIG
    inside __post_init__ solely for min_edge_score and valid tier set.
    This lazy import prevents circular imports and keeps the
    module independently testable by injecting a mock config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional

from scoring.score_result import ScoredSignal


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Hard institutional cap on risk_pct.
# No single signal may allocate more than this % of capital.
# Defensive wall against config misconfiguration.
# If a tier ever specifies risk_pct > this, TieredSignal rejects it.
_MAX_RISK_PCT: float = 5.0

# Valid tier names. Fixed set — adding a tier requires updating this.
# Not imported from config to keep this module self-contained.
# If config defines a tier not in this set, construction fails loudly.
_VALID_TIERS: FrozenSet[str] = frozenset({"A+", "A", "B", "C"})

# Floating-point tolerance for edge_score copy comparison.
# Same value as ScoredSignal._PRODUCT_TOLERANCE — not imported,
# redefined here to keep modules independent.
_EDGE_COPY_TOLERANCE: float = 1e-10


# ---------------------------------------------------------------------------
# TieredSignal
# ---------------------------------------------------------------------------

@dataclass
class TieredSignal:
    """
    An approved harmonic signal ready for subscriber delivery.

    This object is constructed by SignalTier after a ScoredSignal
    passes all qualification gates. It is the authoritative record
    of what a subscriber will receive and under what constraints.

    Field groups:
        Group 1: Operational classification
            tier, max_per_day, risk_pct
            These are the deployment policy fields.
            They answer: "how much, how often, what priority?"

        Group 2: Source signal (preserved complete)
            scored
            The full ScoredSignal is preserved — not summarized.
            Telemetry, backtesting, and attribution analysis
            need the complete scoring audit trail.

        Group 3: Convenience copies (for delivery layer)
            edge_score, dominant_state, reasoning
            Copied from scored for fast access without indirection.
            Invariant: must equal scored.* exactly.

        Group 4: Trading levels (for delivery layer)
            entry, stop, target1, target2, target3, risk_reward
            Forwarded from scored.* for flat access.
            All required levels must be > 0 (enforced).
            Optional levels (target2, target3, risk_reward) may be None.

    Immutability:
        Not frozen (same reasoning as ScoredSignal: contains
        a non-frozen ScoredSignal). Treated as read-only.
        No mutation methods exist. Callers must not modify fields.
    """

    # ── Group 1: Operational classification ──────────────────────────────

    tier: str
    """
    Signal quality tier.
    Must be one of: "A+", "A", "B", "C"

    Semantic meaning:
        A+ : Highest conviction. Max 1/day. Deploy 2% risk.
        A  : High conviction.   Max 3/day. Deploy 1% risk.
        B  : Moderate conviction. Max 5/day. Deploy 0.5% risk.
        C  : Low conviction. Paper/educational only. 0% risk.

    This is NOT computed here. Assigned by SignalTier.
    Stored here as the authoritative tier for this signal instance.
    """

    max_per_day: int
    """
    Maximum number of signals of this tier to deliver per day.
    Stored explicitly (not derived from tier) to allow per-subscriber
    customization in Phase 2 without changing this class.
    Must be >= 1. Zero would permanently suppress delivery and is
    operationally equivalent to not generating the signal at all.
    """

    risk_pct: float
    """
    Suggested position size as a percentage of capital.
    Example: risk_pct=1.0 means "risk 1% of account on this trade."
    Range: [0.0, _MAX_RISK_PCT]

    0.0 is valid for Tier C (paper/educational — no real capital).
    Stored explicitly (not derived from tier) because:
        - Phase 2 may reduce risk after consecutive losses
        - Per-subscriber risk profiles may override tier defaults
        - The value at delivery time must be the authoritative record
    """

    # ── Group 2: Source signal ────────────────────────────────────────────

    scored: ScoredSignal
    """
    The complete scored signal this TieredSignal was derived from.
    Preserved in full for telemetry, backtesting, and attribution.
    Never summarized or truncated.
    """

    # ── Group 3: Convenience copies ───────────────────────────────────────

    edge_score: float
    """
    Copy of scored.edge_score.
    Invariant: abs(edge_score - scored.edge_score) <= _EDGE_COPY_TOLERANCE.
    Stored flat for fast delivery-layer access without indirection.
    """

    dominant_state: str
    """
    Copy of scored.dominant_state.
    Invariant: dominant_state == scored.dominant_state.
    The market state label displayed in the Telegram message.
    """

    reasoning: List[str]
    """
    The complete reasoning chain for subscriber delivery.
    May extend scored.reasoning with tier-specific lines:
        "Tier A+ assigned (edge 73% >= threshold 70%)"
        "Suggested risk: 2.0% of capital | Max signals today: 1"
    Must be non-empty. Every deployed signal must explain itself.
    """

    # ── Group 4: Trading levels ───────────────────────────────────────────

    entry: float
    """
    Limit entry price at the D pivot.
    Must be > 0. A zero entry means no valid setup — reject at construction.
    """

    stop: float
    """
    Stop loss price.
    Must be > 0 and != entry.
    stop == entry means zero risk, which is degenerate.
    """

    target1: float
    """
    Conservative target (B level — first structural level).
    Must be > 0. This is the primary target used for R:R computation
    and tier threshold qualification.
    """

    target2: Optional[float] = None
    """
    Moderate target (A level). May be None for some pattern types.
    If provided, must be > 0.
    """

    target3: Optional[float] = None
    """
    Aggressive target (Fibonacci extension). May be None.
    If provided, must be > 0.
    """

    risk_reward: Optional[float] = None
    """
    Risk:Reward ratio to target1.
    None when stop == entry (degenerate PRZ).
    Forwarded from scored.risk_reward.
    If provided, must be > 0.
    """

    # ── __post_init__ ─────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        """
        Validates all invariants after construction.

        Validation order (each check is independent; first failure reported):
            1. tier is valid and known
            2. max_per_day is >= 1
            3. risk_pct is in valid range
            4. scored is a ScoredSignal instance
            5. edge_score copy matches scored.edge_score
            6. edge_score meets deployment threshold
            7. dominant_state copy matches scored.dominant_state
            8. reasoning is non-empty list of strings
            9. entry, stop, target1 are > 0
           10. stop != entry (non-degenerate PRZ)
           11. optional levels positive if provided
        """
        self._validate_tier()
        self._validate_max_per_day()
        self._validate_risk_pct()
        self._validate_scored_type()
        self._validate_edge_score_copy()
        self._validate_edge_score_threshold()
        self._validate_dominant_state_copy()
        self._validate_reasoning()
        self._validate_required_levels()
        self._validate_stop_not_entry()
        self._validate_optional_levels()

    # ── Validators ────────────────────────────────────────────────────────

    def _validate_tier(self) -> None:
        """
        tier must be one of the four defined operational tiers.

        Using a fixed frozenset (_VALID_TIERS) rather than importing
        from config keeps this module self-contained and testable
        without standing up the full config.

        If a new tier is added to config, _VALID_TIERS must be updated
        here simultaneously. This is intentional: adding a tier is a
        structural change that should require explicit acknowledgement
        in both places.
        """
        if self.tier not in _VALID_TIERS:
            raise ValueError(
                f"tier={self.tier!r} is not a valid tier. "
                f"Must be one of {sorted(_VALID_TIERS)}. "
                f"If a new tier was added to config, update "
                f"_VALID_TIERS in signals/signal.py simultaneously."
            )

    def _validate_max_per_day(self) -> None:
        """
        max_per_day must be a positive integer (>= 1).

        Zero would mean "never deliver this tier" — operationally
        equivalent to never generating the signal. If a tier should
        not be delivered, do not construct TieredSignal for it.
        Use None return from SignalTier instead.

        Negative values are clearly wrong.
        Non-integer values (e.g. 1.5) indicate a caller type error.
        """
        if not isinstance(self.max_per_day, int):
            raise ValueError(
                f"max_per_day={self.max_per_day!r} must be an integer, "
                f"got {type(self.max_per_day).__name__}. "
                f"Frequency caps must be whole numbers."
            )
        if self.max_per_day < 1:
            raise ValueError(
                f"max_per_day={self.max_per_day!r} must be >= 1. "
                f"A cap of zero permanently suppresses delivery. "
                f"If this tier should never be delivered, return None "
                f"from SignalTier.classify() instead of constructing "
                f"a TieredSignal with max_per_day=0."
            )

    def _validate_risk_pct(self) -> None:
        """
        risk_pct must be in [0.0, _MAX_RISK_PCT].

        0.0 is valid: Tier C is paper/educational with no real capital.
        > _MAX_RISK_PCT (5.0%) is never valid: no single signal
        should ever risk more than 5% of capital regardless of tier.
        This is a hard institutional wall against config misconfiguration.

        Negative risk_pct has no valid interpretation.
        """
        if not (0.0 <= self.risk_pct <= _MAX_RISK_PCT):
            raise ValueError(
                f"risk_pct={self.risk_pct!r} is outside [0.0, {_MAX_RISK_PCT}]. "
                f"risk_pct=0.0 is valid for paper/educational tiers. "
                f"risk_pct > {_MAX_RISK_PCT} violates the institutional "
                f"maximum single-signal risk cap. "
                f"Check the tier risk configuration in MS_CONFIG."
            )

    def _validate_scored_type(self) -> None:
        """
        scored must be a ScoredSignal instance.

        Type check is necessary because Python dataclasses do not
        enforce field types at runtime. A caller could pass a dict,
        None, or a different dataclass. All three produce confusing
        AttributeError later rather than a clear failure here.
        """
        if not isinstance(self.scored, ScoredSignal):
            raise ValueError(
                f"scored must be a ScoredSignal instance, "
                f"got {type(self.scored).__name__}. "
                f"TieredSignal must be constructed from a fully "
                f"validated ScoredSignal, not from raw data."
            )

    def _validate_edge_score_copy(self) -> None:
        """
        edge_score must equal scored.edge_score within tolerance.

        This enforces that the copy is exact — no transformation,
        rounding, or formula was applied when copying the value.
        Any discrepancy > _EDGE_COPY_TOLERANCE means the caller
        used a different value than what the scorer computed.

        Why store a copy at all (given this check)?
            For delivery-layer access performance and to make
            TieredSignal self-contained for logging:
            str(tiered_signal) should not require traversing nested objects.
            The copy is valid exactly because this check enforces identity.
        """
        import math
        if not math.isclose(
            self.edge_score,
            self.scored.edge_score,
            abs_tol=_EDGE_COPY_TOLERANCE,
        ):
            raise ValueError(
                f"edge_score={self.edge_score!r} does not match "
                f"scored.edge_score={self.scored.edge_score!r}. "
                f"Delta={abs(self.edge_score - self.scored.edge_score):.2e} "
                f"exceeds tolerance={_EDGE_COPY_TOLERANCE:.0e}. "
                f"The edge_score field must be an exact copy of "
                f"scored.edge_score. Do not recompute or round it."
            )

    def _validate_edge_score_threshold(self) -> None:
        """
        edge_score must be >= MS_CONFIG.min_edge_score.

        This is the deployment gate: TieredSignal must never exist
        for a below-threshold signal. If it does, SignalTier has a bug.

        Lazy import of MS_CONFIG: prevents circular imports and allows
        test code to patch MS_CONFIG before constructing TieredSignal.
        The import cost is negligible — it returns a module-level
        singleton after the first call.
        """
        from config.market_state_config import MS_CONFIG
        if self.edge_score < MS_CONFIG.min_edge_score:
            raise ValueError(
                f"edge_score={self.edge_score!r} is below "
                f"min_edge_score={MS_CONFIG.min_edge_score!r}. "
                f"TieredSignal must only be constructed for signals "
                f"that passed the minimum threshold gate. "
                f"Signals below min_edge_score should return None from "
                f"SignalTier.classify(), not produce a TieredSignal."
            )

    def _validate_dominant_state_copy(self) -> None:
        """
        dominant_state must match scored.dominant_state exactly.

        Same reasoning as edge_score copy: the flat copy must be
        identical to the source. No transformation, no normalization.
        A mismatch means the caller used a different state label
        than what was recorded in the scoring audit trail.
        """
        if self.dominant_state != self.scored.dominant_state:
            raise ValueError(
                f"dominant_state={self.dominant_state!r} does not match "
                f"scored.dominant_state={self.scored.dominant_state!r}. "
                f"The dominant_state field must be an exact copy. "
                f"Set dominant_state = scored.dominant_state."
            )

    def _validate_reasoning(self) -> None:
        """
        reasoning must be a non-empty list of strings.

        Every deployed signal must explain itself to subscribers.
        An empty reasoning chain would deliver a signal with no context,
        which violates the transparency contract of the platform.

        We require at least one line here (stricter check: SignalTier
        should produce >= 5 lines including tier assignment and sizing).
        The minimum of 1 is a safety floor, not the target.

        Non-string elements indicate a construction error in SignalTier
        (e.g. accidentally appending a number instead of an f-string).
        """
        if not isinstance(self.reasoning, list) or len(self.reasoning) == 0:
            raise ValueError(
                f"reasoning must be a non-empty list, "
                f"got {type(self.reasoning).__name__} with "
                f"length {len(self.reasoning) if isinstance(self.reasoning, list) else 'N/A'}. "
                f"Every deployed signal must carry a non-empty reasoning "
                f"chain for subscriber transparency."
            )
        non_strings = [
            (i, type(line).__name__)
            for i, line in enumerate(self.reasoning)
            if not isinstance(line, str)
        ]
        if non_strings:
            raise ValueError(
                f"reasoning contains non-string elements at indices: "
                f"{non_strings}. All reasoning lines must be strings "
                f"for safe display in Telegram messages."
            )

    def _validate_required_levels(self) -> None:
        """
        entry, stop, target1 must all be strictly > 0.

        These are the three mandatory trading levels for any signal.
        A zero value means the PRZ was not computed (degenerate pattern)
        or a default was used accidentally. Neither is acceptable for
        a live deployment decision.

        Unlike ScoredSignal (which allows entry=0.0 for zero-edge signals),
        TieredSignal represents a live signal: zero prices are never valid.
        """
        for name, value in [
            ("entry",   self.entry),
            ("stop",    self.stop),
            ("target1", self.target1),
        ]:
            if value <= 0.0:
                raise ValueError(
                    f"{name}={value!r} must be > 0.0. "
                    f"TieredSignal represents a live deployment decision. "
                    f"A {name} of 0.0 means the PRZ was not computed or "
                    f"a placeholder was used. This signal must not be delivered."
                )

    def _validate_stop_not_entry(self) -> None:
        """
        stop must not equal entry.

        stop == entry means zero risk per share, which makes R:R
        undefined and position sizing impossible. A signal with
        this property cannot be traded.

        We use a relative tolerance of 1e-6 (0.0001%) rather than
        exact equality to handle floating-point prices where the
        difference might be epsilon rather than truly equal.
        For assets priced at $60,000 (BTC), 1e-6 tolerance = $0.06,
        well within any realistic stop distance.
        """
        relative_diff = abs(self.stop - self.entry) / max(abs(self.entry), 1e-10)
        if relative_diff < 1e-6:
            raise ValueError(
                f"stop={self.stop!r} is effectively equal to "
                f"entry={self.entry!r} (relative diff={relative_diff:.2e}). "
                f"A stop at the entry price means zero risk — "
                f"position sizing is undefined. "
                f"This is a degenerate PRZ. Do not deploy this signal."
            )

    def _validate_optional_levels(self) -> None:
        """
        target2, target3, risk_reward must be > 0 if not None.

        None means "not defined for this pattern type" — valid.
        0.0 means "someone set a default" — invalid for a live signal.
        Negative means a sign error in PRZ computation.
        """
        for name, value in [
            ("target2",     self.target2),
            ("target3",     self.target3),
            ("risk_reward", self.risk_reward),
        ]:
            if value is not None and value <= 0.0:
                raise ValueError(
                    f"{name}={value!r} must be > 0.0 if provided, or None. "
                    f"A value of 0.0 or negative indicates a PRZ computation "
                    f"error. Set to None if this level does not apply."
                )

    # ── Computed properties ───────────────────────────────────────────────

    @property
    def symbol(self) -> str:
        """Trading pair. Forwarded from scored.pattern_match.symbol."""
        return self.scored.pattern_match.symbol

    @property
    def timeframe(self) -> str:
        """Candle timeframe. Forwarded from scored.pattern_match.timeframe."""
        return self.scored.pattern_match.timeframe

    @property
    def pattern_name(self) -> str:
        """Pattern type. Forwarded from scored.pattern_match.pattern_name."""
        return self.scored.pattern_match.pattern_name

    @property
    def direction(self) -> str:
        """Trade direction. Forwarded from scored.pattern_match.direction."""
        return self.scored.pattern_match.direction

    @property
    def is_paper_only(self) -> bool:
        """True for Tier C signals — paper/educational, no real capital."""
        return self.risk_pct == 0.0

    @property
    def has_full_targets(self) -> bool:
        """True when all three targets are defined and positive."""
        return (
            self.target1 > 0.0
            and self.target2 is not None and self.target2 > 0.0
            and self.target3 is not None and self.target3 > 0.0
        )

    def summary(self) -> str:
        """
        Single-line human-readable summary for logging.

        Example:
            TieredSignal [Tier A | Gartley BULLISH | BTCUSDT 1h]
            edge=0.5376 | Entry=61500.00 SL=59800.00 T1=64000.00
            R:R=1.47 | risk=1.0% max=3/day
        """
        rr_str = f"{self.risk_reward:.2f}" if self.risk_reward else "N/A"
        return (
            f"TieredSignal [Tier {self.tier} | "
            f"{self.pattern_name} {self.direction.upper()} | "
            f"{self.symbol} {self.timeframe}] | "
            f"edge={self.edge_score:.4f} | "
            f"Entry={self.entry:.2f} "
            f"SL={self.stop:.2f} "
            f"T1={self.target1:.2f} "
            f"R:R={rr_str} | "
            f"risk={self.risk_pct}% "
            f"max={self.max_per_day}/day"
        )

    def __repr__(self) -> str:
        return self.summary()