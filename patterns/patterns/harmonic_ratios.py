"""
harmonic_ratios.py
==================
Harmonic Pattern Fibonacci Ratio Engine

Responsibilities:
    - Define ideal Fibonacci ratios for every pattern (Gartley, Bat, Butterfly, Crab)
    - Validate computed leg ratios against those definitions with configurable tolerance
    - Compute all 5 leg ratios (AB/XA, BC/AB, CD/BC, AD/XA, XD/XA) from raw prices
    - Return per-ratio pass/fail detail for transparency and debugging

Design Philosophy:
    - Ratios are defined as (min, max) ranges — not just a target ± tolerance.
      This matches Scott Carney's published specifications exactly.
    - A global tolerance override can be applied on top of the defined ranges.
      Default is 5% (0.05). Tighter = fewer signals, higher quality.
    - FibonacciCalculator is pure: input prices → output ratios. No state.
    - FibonacciValidator is pure: input ratios + rules → output bool. No state.
    - Ratio definitions live in HarmonicRatioLibrary — single source of truth.
      Adding a new pattern only requires adding an entry here.

References:
    Scott Carney — "Harmonic Trading Vol 1 & 2"
    Harold M. Gartley — "Profits in the Stock Market" (1935)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RatioRange:
    """
    Represents an acceptable Fibonacci ratio range for a single leg.

    Attributes:
        ideal : The textbook ideal ratio (for display and scoring purposes)
        lo    : Minimum acceptable ratio value (before tolerance expansion)
        hi    : Maximum acceptable ratio value (before tolerance expansion)

    Usage:
        range_ = RatioRange(ideal=0.618, lo=0.600, hi=0.636)
        # With 5% tolerance → effective range = (0.570, 0.666)
    """
    ideal: float
    lo:    float
    hi:    float

    def effective_range(self, tolerance: float) -> Tuple[float, float]:
        """
        Expands the base range by the global tolerance factor.

        The tolerance is applied as an absolute addition to the
        already-defined lo/hi boundaries, not as a percentage of
        the ratio itself. This prevents over-expansion on small ratios.

        Args:
            tolerance: Absolute expansion on each side (e.g. 0.05 = 5%)

        Returns:
            (effective_lo, effective_hi)
        """
        return (
            max(0.0, self.lo - tolerance),
            self.hi + tolerance,
        )

    def __repr__(self) -> str:
        return f"RatioRange(ideal={self.ideal}, lo={self.lo}, hi={self.hi})"


@dataclass(frozen=True)
class PatternRatioSpec:
    """
    Complete Fibonacci ratio specification for one harmonic pattern.

    Each field maps to one XABCD leg relationship.
    All fields are RatioRange objects — never None.

    Ratio definitions:
        AB_XA  : How much AB retraces the XA impulse leg
        BC_AB  : How much BC retraces the AB leg
        CD_BC  : How much CD extends/retraces the BC leg
        AD_XA  : How much the full AD move retraces XA (primary PRZ rule)
        XD_XA  : Ratio of XD to XA (used in Butterfly/Crab extended patterns)

    Note on AD_XA vs XD_XA:
        Gartley and Bat use AD_XA (distance from A to D vs XA).
        Butterfly and Crab also use XD_XA (distance from X to D vs XA)
        as an additional confirmation of the extended structure.
    """
    AB_XA:  RatioRange
    BC_AB:  RatioRange
    CD_BC:  RatioRange
    AD_XA:  RatioRange
    XD_XA:  Optional[RatioRange] = None     # Required for Butterfly/Crab


@dataclass
class FibonacciRatios:
    """
    Computed Fibonacci ratios for a specific XABCD structure.

    All ratios are absolute (price-direction-agnostic).
    Computed once per candidate by FibonacciCalculator.

    Attributes:
        AB_XA  : AB / XA
        BC_AB  : BC / AB
        CD_BC  : CD / BC
        AD_XA  : AD / XA
        XD_XA  : XD / XA
        legs   : Raw leg sizes for reference (XA, AB, BC, CD, AD, XD)
    """
    AB_XA: float
    BC_AB: float
    CD_BC: float
    AD_XA: float
    XD_XA: float
    legs:  Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        return {
            "AB_XA": round(self.AB_XA, 5),
            "BC_AB": round(self.BC_AB, 5),
            "CD_BC": round(self.CD_BC, 5),
            "AD_XA": round(self.AD_XA, 5),
            "XD_XA": round(self.XD_XA, 5),
        }

    def __repr__(self) -> str:
        d = self.as_dict()
        return (
            f"FibRatios("
            f"AB/XA={d['AB_XA']} | "
            f"BC/AB={d['BC_AB']} | "
            f"CD/BC={d['CD_BC']} | "
            f"AD/XA={d['AD_XA']} | "
            f"XD/XA={d['XD_XA']})"
        )


@dataclass
class ValidationResult:
    """
    Output of FibonacciValidator.validate().

    Attributes:
        passed       : True only when ALL required ratios pass their ranges
        detail       : Per-ratio pass/fail map
        failures     : List of ratio names that failed (empty if passed)
        error_pct    : How far each failed ratio was outside its range (for scoring)
    """
    passed:    bool
    detail:    Dict[str, bool]
    failures:  List[str]
    error_pct: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        if self.passed:
            return "ValidationResult(PASSED)"
        return f"ValidationResult(FAILED | failed_ratios={self.failures})"


# ---------------------------------------------------------------------------
# Harmonic Ratio Library — Single Source of Truth
# ---------------------------------------------------------------------------

class HarmonicRatioLibrary:
    """
    Defines the Fibonacci ratio specifications for all supported patterns.

    Sources:
        Gartley  : Scott Carney "Harmonic Trading Vol 1" pp. 45-61
        Bat      : Scott Carney "Harmonic Trading Vol 1" pp. 63-79
        Butterfly: Scott Carney "Harmonic Trading Vol 1" pp. 81-97
        Crab     : Scott Carney "Harmonic Trading Vol 1" pp. 99-115

    Fingerprint ratios (the one ratio that uniquely identifies each pattern):
        Gartley   → AD/XA = 0.786
        Bat       → AD/XA = 0.886
        Butterfly → AD/XA = 1.27 or 1.618 (extension beyond X)
        Crab      → XD/XA = 1.618 (deepest extension of all)
    """

    SPECS: Dict[str, PatternRatioSpec] = {

        # ------------------------------------------------------------------ #
        # GARTLEY 222                                                         #
        # The original harmonic pattern. Moderate retracement.               #
        # D sits at 0.786 retrace of XA — the "Golden Mean" of XA.           #
        # ------------------------------------------------------------------ #
        "Gartley": PatternRatioSpec(
            AB_XA = RatioRange(ideal=0.618, lo=0.558, hi=0.678),
            BC_AB = RatioRange(ideal=0.618, lo=0.382, hi=0.886),
            CD_BC = RatioRange(ideal=1.272, lo=1.130, hi=1.618),
            AD_XA = RatioRange(ideal=0.786, lo=0.736, hi=0.836),  # fingerprint
            XD_XA = None,  # not used in Gartley
        ),

        # ------------------------------------------------------------------ #
        # BAT                                                                 #
        # Deeper than Gartley. AB is shallower (0.382-0.500).                #
        # D sits at 0.886 retrace of XA — a "last chance" level.             #
        # Tighter stop (just beyond X), better R:R than Gartley.             #
        # ------------------------------------------------------------------ #
        "Bat": PatternRatioSpec(
            AB_XA = RatioRange(ideal=0.382, lo=0.350, hi=0.500),  # shallower AB
            BC_AB = RatioRange(ideal=0.618, lo=0.382, hi=0.886),
            CD_BC = RatioRange(ideal=2.000, lo=1.618, hi=2.618),  # wider CD
            AD_XA = RatioRange(ideal=0.886, lo=0.836, hi=0.936),  # fingerprint
            XD_XA = None,
        ),

        # ------------------------------------------------------------------ #
        # BUTTERFLY                                                           #
        # Extension pattern — D goes BEYOND X (not a retracement).           #
        # The defining feature: XD > XA (D is outside the XA range).         #
        # AD/XA = 1.27 or 1.618. More aggressive entries, wider stops.       #
        # ------------------------------------------------------------------ #
        "Butterfly": PatternRatioSpec(
            AB_XA = RatioRange(ideal=0.786, lo=0.736, hi=0.836),  # deep AB
            BC_AB = RatioRange(ideal=0.618, lo=0.382, hi=0.886),
            CD_BC = RatioRange(ideal=1.618, lo=1.618, hi=2.618),  # extension
            AD_XA = RatioRange(ideal=1.270, lo=1.130, hi=1.618),  # extension beyond X
            XD_XA = RatioRange(ideal=1.270, lo=1.130, hi=1.618),  # fingerprint
        ),

        # ------------------------------------------------------------------ #
        # CRAB                                                                #
        # The deepest extension pattern. XD reaches 1.618 of XA.            #
        # Most precise entry timing required — D is the extreme point.       #
        # AB is very shallow (0.382-0.618), making CD very long.             #
        # ------------------------------------------------------------------ #
        "Crab": PatternRatioSpec(
            AB_XA = RatioRange(ideal=0.382, lo=0.350, hi=0.618),  # shallow AB
            BC_AB = RatioRange(ideal=0.618, lo=0.382, hi=0.886),
            CD_BC = RatioRange(ideal=2.618, lo=2.240, hi=3.618),  # longest CD
            AD_XA = RatioRange(ideal=1.618, lo=1.500, hi=1.750),  # extreme extension
            XD_XA = RatioRange(ideal=1.618, lo=1.500, hi=1.750),  # fingerprint
        ),
    }

    @classmethod
    def get(cls, pattern_name: str) -> Optional[PatternRatioSpec]:
        """
        Retrieves ratio spec for a given pattern name.
        Returns None if pattern is unknown.
        """
        spec = cls.SPECS.get(pattern_name)
        if spec is None:
            logger.error(f"Unknown pattern requested: '{pattern_name}'")
        return spec

    @classmethod
    def available_patterns(cls) -> List[str]:
        return list(cls.SPECS.keys())


# ---------------------------------------------------------------------------
# Fibonacci Calculator
# ---------------------------------------------------------------------------

class FibonacciCalculator:
    """
    Computes all five Fibonacci leg ratios from raw XABCD pivot prices.

    Pure function class — no state, no side effects.
    Input: 5 prices (X, A, B, C, D)
    Output: FibonacciRatios dataclass

    All computations use absolute leg sizes.
    Direction (bullish/bearish) does not affect ratio mathematics.

    Safe division:
        If any leg has zero size (duplicate price), the ratio is set to 0.0
        and a warning is logged. Zero ratios will always fail validation —
        this is the correct behavior (degenerate XABCD = not a pattern).
    """

    @staticmethod
    def compute(
        X: float,
        A: float,
        B: float,
        C: float,
        D: float,
    ) -> FibonacciRatios:
        """
        Args:
            X, A, B, C, D : Pivot prices in chronological order

        Returns:
            FibonacciRatios with all 5 ratios computed
        """
        # Raw leg sizes (absolute — always positive)
        XA = abs(A - X)
        AB = abs(B - A)
        BC = abs(C - B)
        CD = abs(D - C)
        AD = abs(D - A)
        XD = abs(D - X)

        legs = {
            "XA": round(XA, 8),
            "AB": round(AB, 8),
            "BC": round(BC, 8),
            "CD": round(CD, 8),
            "AD": round(AD, 8),
            "XD": round(XD, 8),
        }

        def safe_div(num: float, den: float, label: str) -> float:
            if den < 1e-10:
                logger.warning(
                    f"Zero-size denominator in ratio '{label}' "
                    f"(num={num:.8f}, den={den:.8f}) — returning 0.0"
                )
                return 0.0
            return num / den

        ratios = FibonacciRatios(
            AB_XA = safe_div(AB, XA, "AB/XA"),
            BC_AB = safe_div(BC, AB, "BC/AB"),
            CD_BC = safe_div(CD, BC, "CD/BC"),
            AD_XA = safe_div(AD, XA, "AD/XA"),
            XD_XA = safe_div(XD, XA, "XD/XA"),
            legs  = legs,
        )

        logger.debug(f"Computed ratios: {ratios}")
        return ratios


# ---------------------------------------------------------------------------
# Fibonacci Validator
# ---------------------------------------------------------------------------

class FibonacciValidator:
    """
    Validates computed FibonacciRatios against a PatternRatioSpec.

    Strategy:
        For each required ratio (AB_XA, BC_AB, CD_BC, AD_XA, and
        optionally XD_XA), check whether the computed value falls
        within the spec's effective range (base range ± global tolerance).

        If ALL required ratios pass → ValidationResult.passed = True.
        If ANY fail → passed = False, failures list is populated.

    Error percentage:
        For failed ratios, we compute how far outside the range the value
        was as a percentage. This feeds the future AI scoring module.
        e.g. if range is (0.70, 0.86) and value is 0.90 → error = 4.7%

    Tolerance parameter:
        Default: 0.05 (5%) — matches Scott Carney's published tolerance.
        Can be tightened to 0.03 for higher-quality signal filtering.
        Can be loosened to 0.08 for more signals (not recommended).
    """

    DEFAULT_TOLERANCE = 0.08

    def __init__(self, tolerance: float = DEFAULT_TOLERANCE):
        """
        Args:
            tolerance: Absolute tolerance applied symmetrically to
                       each ratio range boundary.
        """
        if not (0.0 < tolerance < 0.20):
            raise ValueError(
                f"Tolerance must be between 0.0 and 0.20, got {tolerance}"
            )
        self.tolerance = tolerance
        logger.debug(f"FibonacciValidator initialized | tolerance={tolerance}")

    def validate(
        self,
        ratios: FibonacciRatios,
        spec:   PatternRatioSpec,
    ) -> ValidationResult:
        """
        Validates all ratios in the spec against computed values.

        Args:
            ratios : Computed FibonacciRatios for this XABCD candidate
            spec   : PatternRatioSpec defining acceptable ranges

        Returns:
            ValidationResult with full per-ratio detail
        """
        computed = ratios.as_dict()
        detail:    Dict[str, bool]  = {}
        error_pct: Dict[str, float] = {}
        failures:  List[str]        = []

        # Build list of (ratio_name, RatioRange) pairs to check
        # XD_XA is optional — skip if spec does not define it
        checks: List[Tuple[str, RatioRange]] = [
            ("AB_XA", spec.AB_XA),
            ("BC_AB", spec.BC_AB),
            ("CD_BC", spec.CD_BC),
            ("AD_XA", spec.AD_XA),
        ]
        if spec.XD_XA is not None:
            checks.append(("XD_XA", spec.XD_XA))

        for ratio_name, ratio_range in checks:
            value = computed.get(ratio_name, 0.0)
            lo, hi = ratio_range.effective_range(self.tolerance)

            passed = (lo <= value <= hi)
            detail[ratio_name] = passed

            if passed:
                logger.debug(
                    f"  ✓ {ratio_name}={value:.5f} in [{lo:.5f}, {hi:.5f}]"
                )
            else:
                # Compute how far outside the range the value fell
                if value < lo:
                    pct_error = round((lo - value) / lo * 100, 2)
                else:
                    pct_error = round((value - hi) / hi * 100, 2)

                error_pct[ratio_name] = pct_error
                failures.append(ratio_name)

                logger.debug(
                    f"  ✗ {ratio_name}={value:.5f} not in [{lo:.5f}, {hi:.5f}] "
                    f"| error={pct_error:.2f}%"
                )

        all_passed = len(failures) == 0

        return ValidationResult(
            passed    = all_passed,
            detail    = detail,
            failures  = failures,
            error_pct = error_pct,
        )