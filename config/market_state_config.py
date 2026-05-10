"""
config/market_state_config.py
==============================
Market State Engine — Centralized Configuration

Single source of truth for all tunable parameters.
No threshold, multiplier, or limit is hardcoded anywhere else.

Axiom compliance:
    All harmonic_multiplier values are in (0.0, 1.0].
    This is enforced by __post_init__ validation — not by convention.
    Any value > 1.0 raises ValueError at import time.
    Any value <= 0.0 raises ValueError at import time.

Mathematical framework (from approved scoring philosophy):
    edge_score = base_score × state_discount × confidence_weight
    where:
        base_score        ∈ [0, 1]      pattern Fibonacci quality
        state_discount    ∈ (0, 1]      regime alignment (this table)
        confidence_weight ∈ (0, 1]      vector.confidence directly
        edge_score        ∈ [0, 1]      always, by construction

    NO clamping required. Bound is axiomatic, not enforced.

Multiplier table design rationale:
    reversal   = 1.00 for all:
        Ceiling state for harmonic patterns. Full credit.
        Axiom 3: state context cannot amplify beyond base quality.

    ranging    = 0.80/0.70/0.65:
        Good but not ideal. Retracement patterns (Gartley/Bat)
        align better with ranging than extension patterns (Butterfly/Crab).

    compression = 0.55 for all:
        Patterns form but entries are unreliable.
        Compression does not favor any harmonic type specifically.

    trending   = 0.30/0.40:
        Heavy discount — harmonics are counter-trend tools.
        Extension patterns (Butterfly/Crab) slightly higher:
        extreme overshoots can coincide with trend exhaustion.

    expansion  = 0.25/0.55/0.65:
        Pure expansion destroys Gartley/Bat.
        Butterfly/Crab are structurally aligned with volatility expansion.

    news_chaos = 0.05 for all:
        Axiom 4: near-zero, never zero.
        In practice: 1.0 × 0.05 × 1.0 = 0.05 < min_edge_score.
        All signals suppressed in chaos even if gate passes.

Changelog:
    Week 1: Initial creation (settings.py contained PatternConfig)
    Week 2: Migrated to standalone file. Removed confidence_penalty_floor
            (replaced by direct vector.confidence use). All multiplier
            values capped at 1.0 per axiomatic scoring framework.
            Added post-init validation. Added edge score constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet


# ---------------------------------------------------------------------------
# Valid state and pattern name sets — used for validation
# ---------------------------------------------------------------------------

_VALID_STATES: FrozenSet[str] = frozenset({
    "trending", "ranging", "expansion",
    "compression", "reversal", "news_chaos",
})

_VALID_PATTERNS: FrozenSet[str] = frozenset({
    "Gartley", "Bat", "Butterfly", "Crab",
})

_VALID_TIERS: FrozenSet[str] = frozenset({"A+", "A", "B", "C"})


# ---------------------------------------------------------------------------
# MarketStateConfig
# ---------------------------------------------------------------------------

@dataclass
class MarketStateConfig:
    """
    Single configuration object for the entire Market State +
    Scoring + Signal pipeline.

    Validation contract:
        __post_init__ runs on every construction.
        Any violation raises ValueError with an exact message.
        This means misconfiguration is ALWAYS caught at startup,
        never silently during a live scan.

    Thread safety:
        This object is read-only after construction.
        Do not mutate fields at runtime.
        If runtime overrides are needed, construct a new instance.
    """

    # ── Detector parameters ───────────────────────────────────────────────

    adx_period:    int   = 14
    slope_period:  int   = 20
    bb_period:     int   = 20
    atr_period:    int   = 14
    rsi_period:    int   = 14
    lookback_bars: int   = 50

    # ── Hostile gate thresholds ───────────────────────────────────────────

    gate_chaos_threshold:        float = 0.40
    gate_compression_threshold:  float = 0.65
    gate_confidence_threshold:   float = 0.25
    gate_pure_expansion_thresh:  float = 0.80

    # ── State discount table ──────────────────────────────────────────────
    #
    # Maps dominant_state → pattern_name → discount ∈ (0.0, 1.0]
    #
    # INVARIANT (enforced in __post_init__):
    #   Every value must be in (0.0, 1.0].
    #   Every valid state key must be present.
    #   Every valid pattern key must be present within each state.
    #   Violation → ValueError at construction time.
    #
    # NOTE: "reversal" row is all 1.00 (Axiom 3 — no amplification).
    #       "news_chaos" row is all 0.05 (Axiom 4 — near-zero floor).

    harmonic_multipliers: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "reversal": {
                "Gartley":   1.00,
                "Bat":       1.00,
                "Butterfly": 1.00,
                "Crab":      1.00,
            },
            "ranging": {
                "Gartley":   0.80,
                "Bat":       0.80,
                "Butterfly": 0.70,
                "Crab":      0.65,
            },
            "compression": {
                "Gartley":   0.55,
                "Bat":       0.55,
                "Butterfly": 0.55,
                "Crab":      0.55,
            },
            "trending": {
                "Gartley":   0.30,
                "Bat":       0.30,
                "Butterfly": 0.40,
                "Crab":      0.40,
            },
            "expansion": {
                "Gartley":   0.25,
                "Bat":       0.25,
                "Butterfly": 0.55,
                "Crab":      0.65,
            },
            "news_chaos": {
                "Gartley":   0.05,
                "Bat":       0.05,
                "Butterfly": 0.05,
                "Crab":      0.05,
            },
        }
    )

    # ── Edge score constants ──────────────────────────────────────────────

    # Signals below this are not tiered or delivered.
    # 0.10 means: base × discount × confidence < 0.10 → no output.
    # In chaos: 1.0 × 0.05 × 1.0 = 0.05 < 0.10 → suppressed.
    min_edge_score: float = 0.08

    # Hard ceiling. Never changes. Used in assertions only.
    # If edge_score > this, an axiom was violated — log CRITICAL.
    edge_score_ceiling: float = 1.0

    # ── Signal tier thresholds ────────────────────────────────────────────
    #
    # INVARIANT: thresholds must be strictly decreasing top to bottom.
    # Enforced in __post_init__.

    tier_ap_threshold: float = 0.70
    tier_a_threshold:  float = 0.50
    tier_b_threshold:  float = 0.30
    tier_c_threshold:  float = 0.10

    # ── Tier position sizing ──────────────────────────────────────────────

    tier_ap_risk_pct: float = 2.0
    tier_a_risk_pct:  float = 1.0
    tier_b_risk_pct:  float = 0.5
    tier_c_risk_pct:  float = 0.0   # paper only

    # ── Tier daily frequency caps ─────────────────────────────────────────

    tier_ap_max_daily: int = 1
    tier_a_max_daily:  int = 3
    tier_b_max_daily:  int = 5

    # ── Probability simplex floor ─────────────────────────────────────────

    prob_floor: float = 0.02   # minimum probability for any market state

    # ── Fallback state discount ───────────────────────────────────────────
    #
    # Used when a (state, pattern) key is missing from harmonic_multipliers.
    # 0.50 = conservative neutral — not favorable, not hostile.

    fallback_discount: float = 0.50

    # ────────────────────────────────────────────────────────────────────
    # Validation
    # ────────────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        """
        Validates all fields on construction.
        Raises ValueError with exact message on any violation.
        Runs once — never at scan time.
        """
        self._validate_multiplier_table()
        self._validate_tier_thresholds()
        self._validate_numeric_bounds()

    def _validate_multiplier_table(self) -> None:
        """
        Enforces:
            1. All six state keys present.
            2. All four pattern keys present within each state.
            3. All values in (0.0, 1.0].
            4. Reversal row: all values == 1.00 (Axiom 3).
            5. News chaos row: all values == 0.05 (Axiom 4).
        """
        table = self.harmonic_multipliers

        # Check 1: all state keys present
        missing_states = _VALID_STATES - set(table.keys())
        if missing_states:
            raise ValueError(
                f"harmonic_multipliers missing state keys: {missing_states}"
            )

        for state, pattern_dict in table.items():

            # Check 2: all pattern keys present
            missing_patterns = _VALID_PATTERNS - set(pattern_dict.keys())
            if missing_patterns:
                raise ValueError(
                    f"harmonic_multipliers['{state}'] missing pattern keys: "
                    f"{missing_patterns}"
                )

            for pattern, value in pattern_dict.items():

                # Check 3: values in (0.0, 1.0]
                if not (0.0 < value <= 1.0):
                    raise ValueError(
                        f"harmonic_multipliers['{state}']['{pattern}'] = {value} "
                        f"is outside (0.0, 1.0]. "
                        f"Axiom 3: state discounts cannot amplify (> 1.0) "
                        f"and cannot be zero (multiplicative structure)."
                    )

        # Check 4: reversal row — all must be 1.00 (Axiom 3)
        for pattern, value in table["reversal"].items():
            if value != 1.00:
                raise ValueError(
                    f"harmonic_multipliers['reversal']['{pattern}'] = {value}. "
                    f"Axiom 3 requires reversal discounts == 1.00. "
                    f"Reversal is the ceiling state — full credit, no amplification."
                )

        # Check 5: news_chaos row — all must be 0.05 (Axiom 4)
        for pattern, value in table["news_chaos"].items():
            if value != 0.05:
                raise ValueError(
                    f"harmonic_multipliers['news_chaos']['{pattern}'] = {value}. "
                    f"Axiom 4 requires news_chaos discounts == 0.05. "
                    f"Near-zero preserves multiplicative structure."
                )

    def _validate_tier_thresholds(self) -> None:
        """
        Enforces strictly decreasing threshold order.
        A+ > A > B > C > min_edge_score.
        """
        thresholds = [
            ("A+", self.tier_ap_threshold),
            ("A",  self.tier_a_threshold),
            ("B",  self.tier_b_threshold),
            ("C",  self.tier_c_threshold),
        ]
        for i in range(len(thresholds) - 1):
            name_hi, val_hi = thresholds[i]
            name_lo, val_lo = thresholds[i + 1]
            if val_hi <= val_lo:
                raise ValueError(
                    f"Tier thresholds must be strictly decreasing: "
                    f"tier_{name_hi}_threshold={val_hi} must be > "
                    f"tier_{name_lo}_threshold={val_lo}."
                )

        if self.tier_c_threshold <= self.min_edge_score:
            raise ValueError(
                f"tier_c_threshold={self.tier_c_threshold} must be > "
                f"min_edge_score={self.min_edge_score}. "
                f"Tier C is the lowest signal tier; scores below "
                f"min_edge_score produce no signal at all."
            )

    def _validate_numeric_bounds(self) -> None:
        """Validates scalar parameters are in sane ranges."""
        checks = [
            ("gate_chaos_threshold",       self.gate_chaos_threshold,       0.0, 1.0),
            ("gate_compression_threshold", self.gate_compression_threshold, 0.0, 1.0),
            ("gate_confidence_threshold",  self.gate_confidence_threshold,  0.0, 1.0),
            ("gate_pure_expansion_thresh", self.gate_pure_expansion_thresh, 0.0, 1.0),
            ("min_edge_score",             self.min_edge_score,             0.0, 1.0),
            ("edge_score_ceiling",         self.edge_score_ceiling,         1.0, 1.0),
            ("prob_floor",                 self.prob_floor,                 0.0, 0.20),
            ("fallback_discount",          self.fallback_discount,          0.0, 1.0),
            ("tier_ap_risk_pct",           self.tier_ap_risk_pct,           0.0, 5.0),
            ("tier_a_risk_pct",            self.tier_a_risk_pct,            0.0, 5.0),
            ("tier_b_risk_pct",            self.tier_b_risk_pct,            0.0, 5.0),
        ]
        for name, value, lo, hi in checks:
            if not (lo <= value <= hi):
                raise ValueError(
                    f"{name}={value} is outside valid range [{lo}, {hi}]."
                )

    # ────────────────────────────────────────────────────────────────────
    # Public helpers — used by scoring and tier modules
    # ────────────────────────────────────────────────────────────────────

    def get_discount(self, state: str, pattern: str) -> float:
        """
        Returns the state discount for a (state, pattern) pair.

        Fallback behavior:
            Unknown state:   log-worthy — returns fallback_discount.
            Unknown pattern: log-worthy — returns fallback_discount.

        This method never raises. Callers do not need try/except.
        The fallback discount (0.50) is conservative — it neither
        rewards nor heavily penalizes an unknown combination.
        """
        state_dict = self.harmonic_multipliers.get(state)
        if state_dict is None:
            return self.fallback_discount
        discount = state_dict.get(pattern)
        if discount is None:
            return self.fallback_discount
        return discount

    def tier_rules(self) -> list:
        """
        Returns tier rules as a list of (tier, threshold, max_daily, risk_pct)
        tuples, ordered from highest to lowest threshold.

        Used by SignalTier to avoid reading individual fields.
        If new tiers are added, only this method and the dataclass
        fields need updating — SignalTier code changes not required.
        """
        return [
            ("A+", self.tier_ap_threshold, self.tier_ap_max_daily, self.tier_ap_risk_pct),
            ("A",  self.tier_a_threshold,  self.tier_a_max_daily,  self.tier_a_risk_pct),
            ("B",  self.tier_b_threshold,  self.tier_b_max_daily,  self.tier_b_risk_pct),
            ("C",  self.tier_c_threshold,  99,                      self.tier_c_risk_pct),
        ]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

# Import this everywhere. Do not instantiate MarketStateConfig() in modules.
# Validation runs exactly once, at import time.
MS_CONFIG = MarketStateConfig()