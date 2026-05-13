"""
market_state/vector.py
======================
MarketStateVector — The Core Output Contract of Layer 1

This is the ONLY object that crosses the boundary between
the Market State Engine and every downstream module.

Design rules:
    1. Frozen dataclass — immutable after creation. No mutations anywhere.
    2. All probabilities are floats in [0.0, 1.0].
    3. The six state fields always sum to ~1.0 (enforced by fusion engine).
    4. All derived properties are pure functions of the stored fields.
    5. No pandas, no numpy, no external dependencies in this file.
       This file must be importable by anything without side effects.

States:
    trending     Market moving directionally with structure
    ranging      Market oscillating between defined levels
    expansion    Volatility breakout — range expanding rapidly
    compression  Volatility contraction — energy coiling
    reversal     Trend exhaustion, directional shift beginning
    news_chaos   External shock — technical analysis unreliable

Downstream consumers:
    signals/gate.py          reads is_hostile()
    scoring/pattern_scorer.py reads harmonic_edge_multiplier()
    signals/tier.py          reads dominant_state, confidence
    telemetry/logger.py      reads as_dict, summary()
    delivery/formatter.py    reads summary(), reasoning strings
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


# ---------------------------------------------------------------------------
# MarketStateVector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketStateVector:
    """
    Immutable probabilistic market state distribution.

    Frozen = thread-safe, hashable, no accidental mutation.
    Every pipeline stage receives a fresh instance — never modifies one.

    Probability contract:
        Each of the six state fields is in [0.0, 1.0].
        They sum to approximately 1.0 after fusion normalization.
        Small floating-point drift (<0.001) is acceptable.

    Metadata fields:
        symbol, timeframe, bar_index — for logging and display only.
        They do NOT affect any calculation or comparison.
    """

    # ── State probabilities ───────────────────────────────────────────
    trending:    float
    ranging:     float
    expansion:   float
    compression: float
    reversal:    float
    news_chaos:  float

    # ── Metadata (display / telemetry only) ──────────────────────────
    symbol:    str = ""
    timeframe: str = ""
    bar_index: int = 0
    def __post_init__(self) -> None:
        fields = (
            "trending",
            "ranging",
            "expansion",
            "compression",
            "reversal",
            "news_chaos",
        )

        for name in fields:
            value = getattr(self, name)

            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")

            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(
                    f"{name}={value} outside valid probability range [0.0, 1.0]"
                )

    @property
    def state_probs(self) -> Dict[str, float]:
        """
        Returns the six state probabilities as a plain dict.
        """

        return {
            "trending": self.trending,
            "ranging": self.ranging,
            "expansion": self.expansion,
            "compression": self.compression,
            "reversal": self.reversal,
            "news_chaos": self.news_chaos,
        }

    # ------------------------------------------------------------------ #
    # Core Properties                                                      #
    # ------------------------------------------------------------------ #

    @property
    def state_probs(self) -> Dict[str, float]:
        """
        Returns the six state probabilities as a plain dict.
        This is the canonical representation used by all downstream code.
        Order is fixed — important for consistent logging.
        """
        return {
            "trending":    self.trending,
            "ranging":     self.ranging,
            "expansion":   self.expansion,
            "compression": self.compression,
            "reversal":    self.reversal,
            "news_chaos":  self.news_chaos,
        }

    @property
    def dominant_state(self) -> str:
        """
        State with the highest probability.

        Tie-breaking rule: when all states are near-equal (max < 0.20),
        return 'ranging' — the most conservative fallback.
        This prevents overconfident action on uncertain data.
        """
        probs   = self.state_probs
        max_val = max(probs.values())
        if max_val < 0.20:
            return "ranging"
        # max() on dict returns key of first maximum (stable in Python 3.7+)
        return max(probs, key=probs.get)

    @property
    def confidence(self) -> float:
        """
        Probability of the dominant state.
        Represents how certain the ensemble is about the classification.

        Interpretation:
            >= 0.70 : High confidence   — use full factor weights
            0.50-0.69: Medium confidence — reduce position sizing
            0.30-0.49: Low confidence   — treat as ranging
            < 0.30  : Very low          — hostile (gate will block)
        """
        return max(self.state_probs.values())

    @property
    def is_confident(self) -> bool:
        """True when dominant state probability exceeds 60%."""
        return self.confidence >= 0.60

    # ------------------------------------------------------------------ #
    # Downstream Gate Helpers                                              #
    # ------------------------------------------------------------------ #

    def is_hostile(
        self,
        chaos_threshold:       float = 0.40,
        compression_threshold: float = 0.65,
        confidence_floor:      float = 0.25,
    ) -> bool:
        """
        Returns True when market conditions make trading inadvisable.

        Called by signals/gate.py as the primary gate check.
        Thresholds are injectable — gate.py passes values from config.

        Hostile conditions (any one triggers):
            1. news_chaos too high    → external shock active
            2. compression too high   → waiting for breakout direction
            3. confidence too low     → no reliable state classification

        Args:
            chaos_threshold:       news_chaos level that blocks trading
            compression_threshold: compression level that blocks trading
            confidence_floor:      minimum confidence to allow trading
        """
        if self.news_chaos >= chaos_threshold:
            return True
        if self.compression >= compression_threshold:
            return True
        if self.confidence < confidence_floor:
            return True
        return False

    def hostile_reason(
        self,
        chaos_threshold:       float = 0.40,
        compression_threshold: float = 0.65,
        confidence_floor:      float = 0.25,
    ) -> str:
        """
        Returns the first hostile condition found, or empty string if clean.
        Used by gate.py to populate GateResult.reason.
        """
        if self.news_chaos >= chaos_threshold:
            return (
                f"news_chaos={self.news_chaos:.3f} "
                f">= threshold {chaos_threshold}"
            )
        if self.compression >= compression_threshold:
            return (
                f"compression={self.compression:.3f} "
                f">= threshold {compression_threshold}"
            )
        if self.confidence < confidence_floor:
            return (
                f"confidence={self.confidence:.3f} "
                f"< floor {confidence_floor}"
            )
        return ""

    # ------------------------------------------------------------------ #
    # Harmonic Scoring Interface                                           #
    # ------------------------------------------------------------------ #

    def harmonic_edge_multiplier(
        self,
        multiplier_table: Dict[str, float] = None,
    ) -> float:
        """
        Returns a float multiplier for harmonic pattern edge scores.

        Called by scoring/pattern_scorer.py.
        The multiplier_table is injected from config — this method
        never hardcodes business logic.

        Default table (when None is passed):
            reversal     → 1.50  (ideal for harmonic reversals)
            ranging      → 1.20  (good — bounces between levels)
            compression  → 0.80  (pattern may form but not trigger)
            trending     → 0.50  (counter-trend risk is high)
            expansion    → 0.40  (momentum dominates reversals)
            news_chaos   → 0.10  (never trade harmonics in chaos)

        Returns a probability-weighted blend across all six states.
        This means a mixed state (e.g. 60% reversal + 40% ranging)
        gets a proportionally blended multiplier.

        Output range: [0.10, 1.50]
        """
        if multiplier_table is None:
            multiplier_table = {
                "reversal":    1.50,
                "ranging":     1.20,
                "compression": 0.80,
                "trending":    0.50,
                "expansion":   0.40,
                "news_chaos":  0.10,
            }

        probs    = self.state_probs
        weighted = sum(
            probs.get(state, 0.0) * mult
            for state, mult in multiplier_table.items()
        )

        # Clamp to valid range
        return round(max(0.10, min(1.50, weighted)), 4)

    # ------------------------------------------------------------------ #
    # Display and Logging                                                  #
    # ------------------------------------------------------------------ #

    def as_dict(self) -> Dict[str, float]:
        """
        Rounded probabilities for logging and serialization.
        4 decimal places is sufficient precision for all use cases.
        """
        return {k: round(v, 4) for k, v in self.state_probs.items()}

    def summary(self) -> str:
        """
        Single-line human-readable summary.
        Used in: log messages, Telegram headers, debug output.

        Example:
            MarketState [BTCUSDT 1h] | RANGING (72% conf) |
            T=0.05 R=0.72 E=0.04 C=0.08 Rev=0.09 N=0.02 |
            harm_mult=1.18
        """
        d = self.as_dict()
        label = f"{self.symbol} {self.timeframe}".strip()
        return (
            f"MarketState [{label}] | "
            f"{self.dominant_state.upper()} ({self.confidence:.0%} conf) | "
            f"T={d['trending']:.2f} "
            f"R={d['ranging']:.2f} "
            f"E={d['expansion']:.2f} "
            f"C={d['compression']:.2f} "
            f"Rev={d['reversal']:.2f} "
            f"N={d['news_chaos']:.2f} | "
            f"harm_mult={self.harmonic_edge_multiplier():.2f}"
        )

    def reasoning_lines(self) -> list:
        """
        Human-readable explanation list for Telegram delivery.
        Each element is one line of the signal reasoning chain.

        Example output:
            ["Market state: RANGING (72% confident)",
             "Ranging state favors harmonic reversal patterns",
             "Harmonic edge multiplier: 1.18×"]
        """
        mult  = self.harmonic_edge_multiplier()
        favor = "favors" if mult >= 1.0 else "penalizes"
        lines = [
            f"Market state: {self.dominant_state.upper()} "
            f"({self.confidence:.0%} confident)",
            f"{self.dominant_state.capitalize()} state "
            f"{favor} harmonic reversal patterns",
            f"Harmonic edge multiplier: {mult:.2f}×",
        ]
        if not self.is_confident:
            lines.append(
                f"⚠ Low confidence ({self.confidence:.0%}) — "
                f"reduced conviction applied"
            )
        return lines

    def __repr__(self) -> str:
        return self.summary()
