"""
market_state/detectors/base.py
==============================
Detector Interface Contract

Every detector in the market_state package implements this interface.
No exceptions. This is what makes the fusion engine work cleanly —
it calls .score(df) on any detector and gets a DetectorResult back.

Design rules:
    1. Each detector is stateless after __init__.
       .score(df) is a pure function: same input → same output.
    2. Each detector owns exactly ONE concern.
       TrendDetector only detects trend. Nothing else.
    3. DetectorResult carries both the score AND the raw signals
       that produced it. This is mandatory for observability.
    4. Every detector must handle edge cases gracefully:
       - df too short → return score=0.0, do not raise
       - NaN-heavy data → return score=0.0, do not raise
       - All-zero prices → return score=0.0, do not raise
    5. All scores are in [0.0, 1.0]. Hard contract.
       Fusion engine will raise if this is violated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# DetectorResult — Output Contract
# ---------------------------------------------------------------------------

@dataclass
class DetectorResult:
    """
    Standardized output from every detector.

    Fields:
        score      : Primary output. Float in [0.0, 1.0].
                     0.0 = state definitely NOT present.
                     1.0 = state definitely IS present.
                     Intermediate values represent probability/strength.

        signals    : Dict of named intermediate values that produced
                     the score. Mandatory for observability.
                     Example: {"adx": 28.4, "adx_score": 0.65,
                                "slope_score": 0.42}
                     These appear in telemetry and debug logs.
                     Never leave this empty — it defeats the purpose.

        meta       : Optional extra context. Not used in scoring.
                     Example: {"direction": "up", "regime": "high_vol"}

        detector   : Name of the detector class. Auto-populated.
                     Used in log messages.
    """
    score:    float
    signals:  Dict[str, Any]
    meta:     Dict[str, Any]  = field(default_factory=dict)
    detector: str             = ""

    def __post_init__(self):
        # Hard enforcement of score contract
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"Detector '{self.detector}' returned score={self.score:.6f} "
                f"which is outside [0.0, 1.0]. "
                f"Clamp your output before returning DetectorResult."
            )

    def as_log_dict(self) -> Dict[str, Any]:
        """
        Flat dict for telemetry logging.
        Merges score + signals into one dict with detector prefix.
        """
        result = {f"{self.detector}.score": round(self.score, 4)}
        for k, v in self.signals.items():
            if isinstance(v, float):
                result[f"{self.detector}.{k}"] = round(v, 4)
            else:
                result[f"{self.detector}.{k}"] = v
        return result

    def __repr__(self) -> str:
        sig_str = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in self.signals.items()
        )
        return f"DetectorResult({self.detector} score={self.score:.4f} | {sig_str})"


# ---------------------------------------------------------------------------
# BaseDetector — Abstract Interface
# ---------------------------------------------------------------------------

class BaseDetector(ABC):
    """
    Abstract base class for all market state detectors.

    Subclasses must implement:
        score(df) → DetectorResult

    Subclasses must define:
        DETECTOR_NAME: str  (class-level constant)
        MIN_BARS: int       (minimum df length required)

    Subclasses must NOT:
        - Store mutable state between calls to score()
        - Raise exceptions for data quality issues (return 0.0 instead)
        - Return scores outside [0.0, 1.0]
        - Import from patterns/, signals/, or scoring/ packages
          (detectors are independent of everything except pandas/numpy)
    """

    DETECTOR_NAME: str = ""
    MIN_BARS:      int = 30

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.DETECTOR_NAME:
            raise TypeError(
                f"Class {cls.__name__} must define "
                f"DETECTOR_NAME as a non-empty class attribute."
            )

    def score(self, df: pd.DataFrame) -> DetectorResult:
        """
        Public entry point. Wraps _compute() with:
            - Input validation
            - Exception catching
            - Score range enforcement
            - Debug logging

        Do NOT override this method in subclasses.
        Override _compute() instead.
        """
        import logging
        logger = logging.getLogger(
            f"market_state.detectors.{self.DETECTOR_NAME}"
        )

        # Input validation
        if df is None or len(df) < self.MIN_BARS:
            logger.debug(
                f"{self.DETECTOR_NAME}: insufficient bars "
                f"({0 if df is None else len(df)} < {self.MIN_BARS}) "
                f"— returning 0.0"
            )
            return self._zero_result("insufficient_bars")

        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            logger.warning(
                f"{self.DETECTOR_NAME}: missing columns {missing} "
                f"— returning 0.0"
            )
            return self._zero_result("missing_columns")

        # NaN check on close prices
        nan_ratio = df["close"].isna().mean()
        if nan_ratio > 0.10:
            logger.warning(
                f"{self.DETECTOR_NAME}: {nan_ratio:.1%} NaN in close "
                f"— returning 0.0"
            )
            return self._zero_result("excessive_nan")

        # Execute detector
        try:
            result = self._compute(df)
            result.detector = self.DETECTOR_NAME

            # Clamp score to valid range (last line of defense)
            if not (0.0 <= result.score <= 1.0):
                logger.error(
                    f"{self.DETECTOR_NAME}: score={result.score:.4f} "
                    f"out of range — clamping to [0.0, 1.0]"
                )
                # Use object.__setattr__ to work around frozen — but
                # DetectorResult is NOT frozen, so direct assignment works
                result.score = max(0.0, min(1.0, result.score))

            logger.debug(f"{self.DETECTOR_NAME}: {result}")
            return result

        except Exception as e:
            logger.error(
                f"{self.DETECTOR_NAME}: exception in _compute() — {e}",
                exc_info=True,
            )
            return self._zero_result(f"exception:{type(e).__name__}")

    @abstractmethod
    def _compute(self, df: pd.DataFrame) -> DetectorResult:
        """
        Core detection logic. Implemented by every subclass.

        Guaranteed by score() before calling here:
            - df is not None
            - len(df) >= MIN_BARS
            - Required OHLCV columns exist
            - NaN ratio < 10%

        Must return DetectorResult with:
            - score in [0.0, 1.0]
            - signals dict with at least one entry
        """
        ...

    def _zero_result(self, reason: str = "") -> DetectorResult:
        """Returns a clean zero result with a reason signal."""
        return DetectorResult(
            score    = 0.0,
            signals  = {"reason": reason},
            detector = self.DETECTOR_NAME,
        )

    @staticmethod
    def _safe_clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        """Clamps a float to [lo, hi]. Use for all intermediate scores."""
        return max(lo, min(hi, float(value)))

    @staticmethod
    def _safe_div(num: float, den: float, fallback: float = 0.0) -> float:
        """Division with zero-denominator protection."""
        if abs(den) < 1e-10:
            return fallback
        return float(num / den)
