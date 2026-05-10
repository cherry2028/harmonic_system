"""
market_state/detectors/reversal.py
====================================
ReversalDetector — Detects trend exhaustion and directional shift.

This is the highest-value state for harmonic patterns.
When reversal coincides with a harmonic D-point completion,
that is the maximum-confluence setup in the entire system.

Components:
    RSI divergence (40%): price new extreme + RSI fails to confirm
    MACD divergence(35%): histogram declining at price extreme
    Volume exhaustion(25%): spike followed by rapid decline
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_state.detectors.base import BaseDetector, DetectorResult


class ReversalDetector(BaseDetector):

    DETECTOR_NAME = "reversal"
    MIN_BARS      = 40

    def __init__(self, rsi_period: int = 14, lookback: int = 30):
        self.rsi_period = rsi_period
        self.lookback   = lookback

    def _compute(self, df: pd.DataFrame) -> DetectorResult:
        rsi_div  = self._rsi_divergence(df)
        macd_div = self._macd_divergence(df)
        exhaust  = self._volume_exhaustion(df)

        reversal = (
            rsi_div  * 0.40 +
            macd_div * 0.35 +
            exhaust  * 0.25
        )

        return DetectorResult(
            score   = self._safe_clip(reversal),
            signals = {
                "rsi_divergence":      round(rsi_div, 4),
                "macd_divergence":     round(macd_div, 4),
                "volume_exhaustion":   round(exhaust, 4),
            },
        )

    def _rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_g = gain.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        avg_l = loss.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        rs    = avg_g / avg_l.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _rsi_divergence(self, df: pd.DataFrame) -> float:
        """
        Bearish: price makes higher high, RSI makes lower high.
        Bullish: price makes lower low, RSI makes higher low.
        """
        close = df["close"].iloc[-self.lookback:]
        rsi   = self._rsi(df["close"]).iloc[-self.lookback:]

        if len(close) < 10 or rsi.isna().all():
            return 0.0

        mid = len(close) // 2
        p1, p2 = close.iloc[:mid], close.iloc[mid:]
        r1, r2 = rsi.iloc[:mid],   rsi.iloc[mid:]

        p1_max, p2_max = float(p1.max()), float(p2.max())
        p1_min, p2_min = float(p1.min()), float(p2.min())
        r1_max, r2_max = float(r1.max()), float(r2.max())
        r1_min, r2_min = float(r1.min()), float(r2.min())

        bearish = (p2_max > p1_max) and (r2_max < r1_max)
        bullish = (p2_min < p1_min) and (r2_min > r1_min)

        if not (bearish or bullish):
            return 0.0

        if bearish:
            p_diff = self._safe_div(p2_max - p1_max, p1_max)
            r_diff = self._safe_div(r1_max - r2_max, 100.0)
        else:
            p_diff = self._safe_div(p1_min - p2_min, p1_min)
            r_diff = self._safe_div(r2_min - r1_min, 100.0)

        return self._safe_clip((p_diff + r_diff) * 3.0)

    def _macd_divergence(self, df: pd.DataFrame) -> float:
        """MACD histogram declining while price at extreme."""
        close  = df["close"]
        macd   = (close.ewm(span=12, adjust=False).mean()
                  - close.ewm(span=26, adjust=False).mean())
        signal = macd.ewm(span=9, adjust=False).mean()
        hist   = (macd - signal).dropna()

        if len(hist) < 5:
            return 0.0

        recent = hist.iloc[-5:]
        declining = float(recent.iloc[-1]) < float(recent.iloc[-3])
        extreme   = abs(float(recent.iloc[-1])) > abs(float(hist.mean()))

        if declining and extreme:
            return 0.65
        if declining:
            return 0.30
        return 0.0

    def _volume_exhaustion(self, df: pd.DataFrame) -> float:
        """Volume climax: spike then rapid decline."""
        vol      = df["volume"].iloc[-self.lookback:]
        avg_vol  = float(vol.mean())
        peak_vol = float(vol.max())
        recent   = float(vol.iloc[-3:].mean())

        if avg_vol < 1e-10:
            return 0.0

        spike   = self._safe_div(peak_vol, avg_vol)
        decline = self._safe_div(recent, peak_vol, fallback=1.0)

        if spike > 2.5 and decline < 0.50:
            return 0.80
        if spike > 2.0 and decline < 0.65:
            return 0.50
        if spike > 1.5:
            return 0.25
        return 0.0
