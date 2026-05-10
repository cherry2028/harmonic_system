"""
harmonic_patterns.py
====================
Individual Harmonic Pattern Definitions

Responsibilities:
    - One class per pattern: GartleyPattern, BatPattern,
      ButterflyPattern, CrabPattern
    - Each class owns its structural validation rules
      (beyond Fibonacci ratios — geometric rules)
    - Each class computes the PRZ (Potential Reversal Zone)
      with entry, stop, and three targets
    - Each class produces a structured PatternMatch result
    - All classes inherit from BaseHarmonicPattern (DRY, extensible)

Architecture:
    BaseHarmonicPattern
        ├── GartleyPattern
        ├── BatPattern
        ├── ButterflyPattern
        └── CrabPattern

Adding a new pattern (e.g. Shark, Cypher):
    1. Add ratio spec to HarmonicRatioLibrary (harmonic_ratios.py)
    2. Create a new class here inheriting BaseHarmonicPattern
    3. Implement _structural_check() and _compute_prz()
    4. Register in HarmonicDetector.PATTERN_REGISTRY (harmonic_detector.py)
    That is all. Zero changes to existing code.

PRZ Philosophy:
    D is the entry zone, not a single price.
    Entry = D price (or limit order near D)
    Stop  = Beyond X with a small buffer (the pattern is invalid if X breaks)
    T1    = Conservative: B level (first structural level)
    T2    = Moderate:     A level (full pattern measured move)
    T3    = Aggressive:   Fibonacci extension of XA beyond A

    For extension patterns (Butterfly, Crab) where D is beyond X:
    Stop placement is critical — must be beyond D with a buffer,
    not at X (which is already inside the trade's profit zone).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from patterns.patterns.harmonic_ratios import (
    FibonacciCalculator,
    FibonacciRatios,
    FibonacciValidator,
    HarmonicRatioLibrary,
    ValidationResult,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Swing Point (local import guard — allows standalone use)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SwingPoint:
    """
    Minimal SwingPoint definition for this module.
    In the full system, import from swing_detector.py.
    Kept here for modularity and standalone testability.
    """
    index:     int
    timestamp: pd.Timestamp
    price:     float
    kind:      str      # 'high' | 'low'

    def __repr__(self) -> str:
        return (
            f"Swing({self.kind.upper()} "
            f"@ {self.price:.6f} "
            f"| bar={self.index})"
        )


# ---------------------------------------------------------------------------
# Pattern Match Result
# ---------------------------------------------------------------------------

@dataclass
class PatternMatch:
    """
    Structured output from a successful pattern detection.

    This is the contract between the pattern layer and the signal layer.
    Phase 3 (Signal Engine) consumes only PatternMatch objects —
    it never looks inside the pattern classes themselves.

    Attributes:
        pattern_name : "Gartley" | "Bat" | "Butterfly" | "Crab"
        direction    : "bullish" | "bearish"
        symbol       : Trading pair e.g. "BTCUSDT"
        timeframe    : "15m" | "1h" | "4h"
        pivots       : Raw XABCD prices
        ratios       : Computed Fibonacci ratios
        validation   : Per-ratio validation detail
        prz          : PRZ trading levels (entry, stop, targets)
        D_index      : Bar index of D pivot in source DataFrame
        D_timestamp  : Timestamp of D pivot
        confirmed    : Confirmation candle logic (set by Signal Engine)
        quality_score: 0.0–1.0 composite quality score (for AI module)
        metadata     : Extensible dict for future fields
    """
    pattern_name:  str
    direction:     str
    symbol:        str
    timeframe:     str
    pivots:        Dict[str, float]
    ratios:        Dict[str, float]
    validation:    Dict[str, bool]
    prz:           Dict[str, float]
    D_index:       int
    D_timestamp:   pd.Timestamp
    confirmed:     bool  = False
    quality_score: float = 0.0
    metadata:      Dict  = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Computed Properties                                                  #
    # ------------------------------------------------------------------ #

    @property
    def is_bullish(self) -> bool:
        return self.direction == "bullish"

    @property
    def risk_reward(self) -> Optional[float]:
        """
        Computes R:R to Target 1 from entry and stop.
        Returns None if stop == entry (degenerate case).
        """
        entry = self.prz.get("entry", 0.0)
        stop  = self.prz.get("stop",  0.0)
        t1    = self.prz.get("target1", 0.0)

        risk   = abs(entry - stop)
        reward = abs(t1 - entry)

        if risk < 1e-10:
            return None
        return round(reward / risk, 3)

    def summary(self) -> str:
        rr = self.risk_reward
        rr_str = f"{rr:.2f}" if rr else "N/A"
        return (
            f"[{self.pattern_name} {self.direction.upper()}] "
            f"{self.symbol} {self.timeframe} | "
            f"D={self.pivots['D']:.4f} | "
            f"Entry={self.prz.get('entry', 0):.4f} | "
            f"SL={self.prz.get('stop', 0):.4f} | "
            f"T1={self.prz.get('target1', 0):.4f} | "
            f"R:R={rr_str}"
        )

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Base Pattern Class
# ---------------------------------------------------------------------------

class BaseHarmonicPattern(ABC):
    """
    Abstract base class for all harmonic pattern implementations.

    Shared responsibilities:
        - Load ratio spec from HarmonicRatioLibrary
        - Run Fibonacci validation via FibonacciValidator
        - Compute quality score from validation closeness
        - Delegate structural checks to subclass
        - Delegate PRZ computation to subclass

    Template method pattern:
        match() is the public interface.
        It calls _structural_check() and _compute_prz() on the subclass.
        Subclasses never override match() directly.

    Args:
        tolerance: Fibonacci ratio tolerance (default 0.05)
    """

    # Subclasses must define this
    PATTERN_NAME: str = ""

    def __init__(self, tolerance: float = 0.10):
        if not self.PATTERN_NAME:
            raise NotImplementedError(
                "Subclass must define PATTERN_NAME class attribute"
            )
        self.validator    = FibonacciValidator(tolerance=tolerance)
        self.calculator   = FibonacciCalculator()
        self.spec         = HarmonicRatioLibrary.get(self.PATTERN_NAME)

        if self.spec is None:
            raise ValueError(
                f"No ratio spec found for pattern '{self.PATTERN_NAME}'. "
                f"Add it to HarmonicRatioLibrary.SPECS first."
            )

        logger.debug(f"{self.PATTERN_NAME}Pattern initialized | tolerance={tolerance}")

    # ------------------------------------------------------------------ #
    # Public Interface                                                     #
    # ------------------------------------------------------------------ #

    def match(
        self,
        X:          SwingPoint,
        A:          SwingPoint,
        B:          SwingPoint,
        C:          SwingPoint,
        D:          SwingPoint,
        symbol:     str,
        timeframe:  str,
    ) -> Optional[PatternMatch]:
        """
        Runs the full pattern match pipeline.

        Steps:
            1. Compute Fibonacci ratios
            2. Validate ratios against spec
            3. Run pattern-specific structural checks
            4. Compute PRZ trading levels
            5. Build and return PatternMatch

        Returns:
            PatternMatch if pattern is detected
            None if pattern is rejected (with debug logging)
        """
        logger.debug(
            f"Matching {self.PATTERN_NAME} | "
            f"X={X.price:.4f} A={A.price:.4f} B={B.price:.4f} "
            f"C={C.price:.4f} D={D.price:.4f}"
        )

        # Step 1: Compute Fibonacci ratios
        ratios = self.calculator.compute(
            X=X.price, A=A.price,
            B=B.price, C=C.price, D=D.price,
        )

        # Step 2: Fibonacci validation
        validation = self.validator.validate(ratios, self.spec)
        if not validation.passed:
            print("FIB FAILED:", validation.failures)
            return None

        # Step 3: Structural / geometric validation (pattern-specific)
        struct_ok, struct_reason = self._structural_check(X, A, B, C, D)
        if not struct_ok:
            print("STRUCT FAILED:", struct_reason)
            return None

        # Step 4: Determine direction and compute PRZ
        direction = self._direction(X, A)
        prz       = self._compute_prz(X, A, B, C, D, direction, ratios)
        print("PRZ:", prz)

        # Step 5: Compute quality score
        score = self._quality_score(ratios, validation)

        print("BUILDING PATTERN")
        # Step 6: Build result
        result = PatternMatch(
            pattern_name  = self.PATTERN_NAME,
            direction     = direction,
            symbol        = symbol,
            timeframe     = timeframe,
            pivots        = {
                "X": X.price, "A": A.price,
                "B": B.price, "C": C.price, "D": D.price,
            },
            ratios        = ratios.as_dict(),
            validation    = validation.detail,
            prz           = prz,
            D_index       = D.index,
            D_timestamp   = D.timestamp,
            quality_score = score,
            metadata      = {
                "error_pct":   validation.error_pct,
                "leg_sizes":   ratios.legs,
                "D_bar_index": D.index,
            },
        )

        logger.info(
            f"✅ {self.PATTERN_NAME} CONFIRMED | {result.summary()}"
        )
        return result

    # ------------------------------------------------------------------ #
    # Shared Helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _direction(X: SwingPoint, A: SwingPoint) -> str:
        """
        Determines pattern direction from X→A impulse leg.

        Bullish : X is a low, A is a high (price rose in XA)
                  Pattern completes at D below A → buy zone
        Bearish : X is a high, A is a low (price fell in XA)
                  Pattern completes at D above A → sell zone
        """
        if X.kind == "low" and A.kind == "high":
            return "bullish"
        elif X.kind == "high" and A.kind == "low":
            return "bearish"
        else:
            # Defensive fallback — should be caught by pivot extractor
            logger.warning(
                f"Ambiguous direction: X.kind={X.kind}, A.kind={A.kind}"
            )
            return "unknown"

    @staticmethod
    def _quality_score(
        ratios:     FibonacciRatios,
        validation: ValidationResult,
    ) -> float:
        """
        Computes a 0.0–1.0 quality score based on how close each
        ratio is to its ideal value.

        Score = 1.0 means all ratios hit their exact ideal values.
        Score = 0.7 means average 30% deviation within tolerance.

        This score is reserved for the AI scoring module (Phase 6).
        For now it's computed and stored but not used for filtering.
        """
        if not validation.passed:
            return 0.0

        # Average error percentage across passed ratios (lower is better)
        total_error = sum(validation.error_pct.values())
        n_checked   = max(len(validation.detail), 1)
        avg_error   = total_error / n_checked

        # Convert to 0-1 score (5% max error = 0.0 quality bonus)
        score = max(0.0, 1.0 - (avg_error / 5.0))
        return round(score, 4)

    # ------------------------------------------------------------------ #
    # Abstract Methods — Implemented by Subclasses                        #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _structural_check(
        self,
        X: SwingPoint,
        A: SwingPoint,
        B: SwingPoint,
        C: SwingPoint,
        D: SwingPoint,
    ) -> tuple[bool, str]:
        """
        Pattern-specific geometric/structural validation.

        Returns:
            (True, "")            if structure is valid
            (False, "reason")     if structure is invalid

        These checks run AFTER Fibonacci validation passes —
        they are the final filter for structural integrity.
        """
        ...

    @abstractmethod
    def _compute_prz(
        self,
        X:         SwingPoint,
        A:         SwingPoint,
        B:         SwingPoint,
        C:         SwingPoint,
        D:         SwingPoint,
        direction: str,
        ratios:    FibonacciRatios,
    ) -> Dict[str, float]:
        """
        Computes the PRZ trading levels for this pattern.

        Must return a dict with at minimum:
            entry   : Limit entry price at D
            stop    : Stop loss price (beyond X for retracement patterns)
            target1 : Conservative target (B level)
            target2 : Moderate target (A level)
            target3 : Aggressive target (Fibonacci extension)
        """
        ...


# ---------------------------------------------------------------------------
# Gartley Pattern
# ---------------------------------------------------------------------------

class GartleyPattern(BaseHarmonicPattern):
    """
    Gartley 222 Pattern.

    Origin: Harold M. Gartley (1935) — "Profits in the Stock Market"
    Refined by Scott Carney with precise Fibonacci definitions.

    Fingerprint ratio: AD/XA = 0.786
    The 0.786 is derived from the square root of 0.618 (Golden Ratio).

    Structural rules (beyond Fibonacci):
        Bullish: D must be above X (it's a retracement, not an extension)
                 B must be below A (proper structure)
        Bearish: D must be below X
                 B must be above A

    Risk profile:
        Moderate — D is deep but not extreme
        Stop just below X gives room for the pattern to breathe
        R:R to T1 typically 1.5:1, to T2 typically 2.5:1
    """

    PATTERN_NAME = "Gartley"

    def _structural_check(
        self,
        X: SwingPoint, A: SwingPoint,
        B: SwingPoint, C: SwingPoint, D: SwingPoint,
    ) -> tuple[bool, str]:
        """
        For Gartley, D must remain within the XA range.
        (D is a retracement — it does not go beyond X.)
        """
        if X.kind == "low":   # Bullish
            if D.price <= X.price:
                return False, f"Bullish Gartley: D={D.price:.4f} must be > X={X.price:.4f}"
            if D.price >= A.price:
                return False, f"Bullish Gartley: D={D.price:.4f} must be < A={A.price:.4f}"
            if B.price >= A.price:
                return False, f"Bullish Gartley: B={B.price:.4f} must be < A={A.price:.4f}"
        else:                  # Bearish
            if D.price >= X.price:
                return False, f"Bearish Gartley: D={D.price:.4f} must be < X={X.price:.4f}"
            if D.price <= A.price:
                return False, f"Bearish Gartley: D={D.price:.4f} must be > A={A.price:.4f}"
            if B.price <= A.price:
                return False, f"Bearish Gartley: B={B.price:.4f} must be > A={A.price:.4f}"

        return True, ""

    def _compute_prz(
        self,
        X: SwingPoint, A: SwingPoint,
        B: SwingPoint, C: SwingPoint, D: SwingPoint,
        direction: str, ratios: FibonacciRatios,
    ) -> Dict[str, float]:
        XA_size = ratios.legs["XA"]
        entry   = D.price

        if direction == "bullish":
            stop    = round(X.price * (1 - 0.005), 6)        # 0.5% below X
            target1 = round(B.price, 6)                       # B level
            target2 = round(A.price, 6)                       # A level
            target3 = round(D.price + XA_size * 1.272, 6)    # 1.272 extension
        else:
            stop    = round(X.price * (1 + 0.005), 6)
            target1 = round(B.price, 6)
            target2 = round(A.price, 6)
            target3 = round(D.price - XA_size * 1.272, 6)

        return {
            "entry":   round(entry, 6),
            "stop":    stop,
            "target1": target1,
            "target2": target2,
            "target3": target3,
        }


# ---------------------------------------------------------------------------
# Bat Pattern
# ---------------------------------------------------------------------------

class BatPattern(BaseHarmonicPattern):
    """
    Bat Pattern.

    Discovered by Scott Carney in 2001.

    Fingerprint ratio: AD/XA = 0.886
    The 0.886 is the 4th root of the Golden Ratio (0.618^0.25).
    It represents the deepest retracement before X that still
    preserves the original trend structure.

    Key differentiator from Gartley:
        AB is shallower (0.382–0.500 vs Gartley's 0.618).
        This makes CD proportionally longer.
        D is deeper (0.886 vs 0.786) — closer to X.

    Why the Bat is valuable:
        The 0.886 is mathematically the "last chance" level.
        A break of X after a confirmed Bat D-point signals
        a high-probability trend reversal with significant momentum.
        Stop placement is extremely tight (just beyond X) giving
        excellent R:R ratios — often 3:1 or better to T2.

    Structural rules:
        Same as Gartley — D must stay within XA range.
        AB must be ≤ 0.500 of XA (differentiates from Gartley).
    """

    PATTERN_NAME = "Bat"

    def _structural_check(
        self,
        X: SwingPoint, A: SwingPoint,
        B: SwingPoint, C: SwingPoint, D: SwingPoint,
    ) -> tuple[bool, str]:
        """
        Bat-specific structural validation.
        D must be within XA range (not beyond X).
        """
        if X.kind == "low":    # Bullish Bat
            if D.price <= X.price:
                return False, f"Bullish Bat: D={D.price:.4f} must be > X={X.price:.4f}"
            if D.price >= A.price:
                return False, f"Bullish Bat: D={D.price:.4f} must be < A={A.price:.4f}"
            if C.price >= A.price:
                return False, f"Bullish Bat: C={C.price:.4f} must be < A={A.price:.4f}"
        else:                  # Bearish Bat
            if D.price >= X.price:
                return False, f"Bearish Bat: D={D.price:.4f} must be < X={X.price:.4f}"
            if D.price <= A.price:
                return False, f"Bearish Bat: D={D.price:.4f} must be > A={A.price:.4f}"
            if C.price <= A.price:
                return False, f"Bearish Bat: C={C.price:.4f} must be > A={A.price:.4f}"

        return True, ""

    def _compute_prz(
        self,
        X: SwingPoint, A: SwingPoint,
        B: SwingPoint, C: SwingPoint, D: SwingPoint,
        direction: str, ratios: FibonacciRatios,
    ) -> Dict[str, float]:
        XA_size = ratios.legs["XA"]
        CD_size = ratios.legs["CD"]
        entry   = D.price

        if direction == "bullish":
            # Stop is very tight — just beyond X. The Bat's key advantage.
            stop    = round(X.price * (1 - 0.003), 6)
            target1 = round(D.price + CD_size * 0.618, 6)   # 61.8% of CD
            target2 = round(B.price, 6)
            target3 = round(A.price, 6)
        else:
            stop    = round(X.price * (1 + 0.003), 6)
            target1 = round(D.price - CD_size * 0.618, 6)
            target2 = round(B.price, 6)
            target3 = round(A.price, 6)

        return {
            "entry":   round(entry, 6),
            "stop":    stop,
            "target1": target1,
            "target2": target2,
            "target3": target3,
        }


# ---------------------------------------------------------------------------
# Butterfly Pattern
# ---------------------------------------------------------------------------

class ButterflyPattern(BaseHarmonicPattern):
    """
    Butterfly Pattern.

    Discovered by Bryce Gilmore, refined by Scott Carney.

    Fingerprint ratio: XD/XA ≥ 1.27 (D extends BEYOND X)
    This is the defining feature — the Butterfly is an EXTENSION pattern.
    D is not a retracement of XA; it exceeds X in the direction of the move.

    Conceptual difference from Gartley/Bat:
        Gartley/Bat: XA moves up, D sits somewhere between X and A (retrace)
        Butterfly:   XA moves up, D drops BELOW X (extension beyond X)

    This means:
        Bullish Butterfly → D is BELOW X (new potential low)
        Bearish Butterfly → D is ABOVE X (new potential high)

    When to trade it:
        Butterfly completions often coincide with major support/resistance breaks
        that shake out weak hands before a violent reversal.
        The signal is rarer but tends to produce larger moves.

    Stop placement:
        Since D is beyond X, stop must be placed beyond D — not at X.
        Use a buffer of 1.0–1.5% beyond D.

    Structural rules:
        AB retraces 0.786 of XA (deep, almost full retrace)
        D goes beyond X (XD/XA ≥ 1.27)
    """

    PATTERN_NAME = "Butterfly"

    def _structural_check(
        self,
        X: SwingPoint, A: SwingPoint,
        B: SwingPoint, C: SwingPoint, D: SwingPoint,
    ) -> tuple[bool, str]:
        """
        Butterfly structural rule: D must exceed X
        (extension pattern — this is what makes it a Butterfly).
        """
        if X.kind == "low":    # Bullish Butterfly
            # Bullish: XA goes up (X=low, A=high), D drops below X
            if D.price >= X.price:
                return False, (
                    f"Bullish Butterfly: D={D.price:.4f} must be < X={X.price:.4f} "
                    f"(extension required)"
                )
            if B.price >= A.price:
                return False, f"Bullish Butterfly: B={B.price:.4f} must be < A={A.price:.4f}"

        else:                  # Bearish Butterfly
            # Bearish: XA goes down (X=high, A=low), D rises above X
            if D.price <= X.price:
                return False, (
                    f"Bearish Butterfly: D={D.price:.4f} must be > X={X.price:.4f} "
                    f"(extension required)"
                )
            if B.price <= A.price:
                return False, f"Bearish Butterfly: B={B.price:.4f} must be > A={A.price:.4f}"

        return True, ""

    def _compute_prz(
        self,
        X: SwingPoint, A: SwingPoint,
        B: SwingPoint, C: SwingPoint, D: SwingPoint,
        direction: str, ratios: FibonacciRatios,
    ) -> Dict[str, float]:
        XA_size = ratios.legs["XA"]
        entry   = D.price

        if direction == "bullish":
            # D is below X — stop goes further below D
            stop    = round(D.price * (1 - 0.008), 6)       # 0.8% below D
            target1 = round(X.price, 6)                      # Back to X
            target2 = round(B.price, 6)                      # B level
            target3 = round(A.price, 6)                      # A level (full move)
        else:
            stop    = round(D.price * (1 + 0.008), 6)
            target1 = round(X.price, 6)
            target2 = round(B.price, 6)
            target3 = round(A.price, 6)

        return {
            "entry":   round(entry, 6),
            "stop":    stop,
            "target1": target1,
            "target2": target2,
            "target3": target3,
        }


# ---------------------------------------------------------------------------
# Crab Pattern
# ---------------------------------------------------------------------------

class CrabPattern(BaseHarmonicPattern):
    """
    Crab Pattern.

    Discovered by Scott Carney in 2000. The most extreme harmonic pattern.

    Fingerprint ratio: XD/XA = 1.618 (the Golden Ratio extension)
    D is the most extreme point relative to XA of all harmonic patterns.
    This is also why the Crab has the highest accuracy when confirmed —
    price is reaching an extreme exhaustion point.

    Key characteristics:
        - AB is very shallow (0.382–0.618 of XA)
        - CD is extremely long (2.24–3.618 of BC)
        - D extends 1.618 beyond X (the Golden Ratio)
        - Requires the most precise entry timing
        - Tolerance for this pattern should be tighter (0.03–0.04)

    Entry discipline:
        Because D is at such an extreme level, any overshoot is
        likely to produce a violent reversal. Entry must be precise —
        limit orders at D, never market orders chasing.
        If D breaks without reversal signal → pattern failed → no trade.

    Risk profile:
        Highest potential R:R of all four patterns
        but also highest failed-pattern risk.
        Only trade with volume confirmation and confirmation candle.

    Structural rules:
        D extends well beyond X (XD > XA × 1.5)
        AB must be shallow (differentiates Crab from Butterfly)
    """

    PATTERN_NAME = "Crab"

    def _structural_check(
        self,
        X: SwingPoint, A: SwingPoint,
        B: SwingPoint, C: SwingPoint, D: SwingPoint,
    ) -> tuple[bool, str]:
        """
        Crab structural rule: D must extend beyond X by at least 1.5× XA.
        This is more extreme than Butterfly (1.27×).
        """
        XA_size = abs(A.price - X.price)
        XD_size = abs(D.price - X.price)

        # Minimum XD must be 1.4× XA (conservative structural floor)
        min_xd = XA_size * 1.40

        if XD_size < min_xd:
            return True, (
                f"Crab: XD={XD_size:.4f} must be ≥ {min_xd:.4f} "
                f"(1.40× XA={XA_size:.4f})"
            )

        if X.kind == "low":    # Bullish Crab
            if D.price >= X.price:
                return True, (
                    f"Bullish Crab: D={D.price:.4f} must be < X={X.price:.4f}"
                )
        else:                  # Bearish Crab
            if D.price <= X.price:
                return True, (
                    f"Bearish Crab: D={D.price:.4f} must be > X={X.price:.4f}"
                )

        return True, ""

    def _compute_prz(
        self,
        X: SwingPoint, A: SwingPoint,
        B: SwingPoint, C: SwingPoint, D: SwingPoint,
        direction: str, ratios: FibonacciRatios,
    ) -> Dict[str, float]:
        XA_size = ratios.legs["XA"]
        entry   = D.price

        if direction == "bullish":
            # Stop well below D — Crab has the most extreme D point
            stop    = round(D.price * (1 - 0.010), 6)       # 1.0% below D
            target1 = round(X.price, 6)                      # Back to X (significant move)
            target2 = round(A.price * 0.618 + D.price * 0.382, 6)  # Fibonacci blend
            target3 = round(A.price, 6)                      # Full retrace to A
        else:
            stop    = round(D.price * (1 + 0.010), 6)
            target1 = round(X.price, 6)
            target2 = round(A.price * 0.618 + D.price * 0.382, 6)
            target3 = round(A.price, 6)

        return {
            "entry":   round(entry, 6),
            "stop":    stop,
            "target1": target1,
            "target2": target2,
            "target3": target3,
        }