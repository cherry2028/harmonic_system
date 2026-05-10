"""
market_state/detectors/range_.py
=================================
RangeDetector — Detects oscillating / mean-reverting market state.

Score interpretation:
    High score = price bouncing predictably between levels
    Low score  = directional or chaotic movement

Components:
    Bollinger width percentile (50%): narrow bands = ranging
    Oscillation frequency     (50%): frequent direction reversals
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_state.detectors.base import BaseDetector, DetectorResult


class RangeDetector(BaseDetector):

    DETECTOR_NAME = "ranging"
    MIN_BARS      = 30

    def __init__(self, bb_period: int = 20):
        self.bb_period = bb_period

    def _compute(self, df: pd.DataFrame) -> DetectorResult:
        bb_score  = self._bollinger_width_score(df)
        osc_score = self._oscillation_score(df)

        range_score = (bb_score * 0.50) + (osc_score * 0.50)

        return DetectorResult(
            score   = self._safe_clip(range_score),
            signals = {
                "bb_score":        round(bb_score, 4),
                "oscillation":     round(osc_score, 4),
                "bb_period":       self.bb_period,
            },
        )

    def _bollinger_width_score(self, df: pd.DataFrame) -> float:
        close = df["close"]
        mid   = close.rolling(self.bb_period).mean()
        std   = close.rolling(self.bb_period).std()
        width = (2 * std * 2) / mid.replace(0, np.nan)

        valid = width.dropna()
        if len(valid) < 10:
            return 0.0

        # Low width percentile → high ranging score
        current_pctl = float((valid <= valid.iloc[-1]).mean())
        return self._safe_clip(1.0 - current_pctl)

    def _oscillation_score(self, df: pd.DataFrame) -> float:
        closes = df["close"].values[-self.bb_period:]
        if len(closes) < 5:
            return 0.0

        diffs        = np.diff(closes)
        sign_changes = float(np.sum(np.diff(np.sign(diffs)) != 0))
        max_changes  = max(len(diffs) - 1, 1)
        return self._safe_clip(sign_changes / max_changes)
