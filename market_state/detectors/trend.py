"""
market_state/detectors/trend.py
================================
TrendDetector — Detects directional trending market state.

Score interpretation:
    0.0 – 0.20 : No trend (ADX < 15, flat slope)
    0.20 – 0.45: Weak trend forming (ADX 15–25)
    0.45 – 0.70: Solid trend (ADX 25–40)
    0.70 – 1.0 : Strong trend (ADX > 40)

Components:
    ADX (60% weight): direction-agnostic trend strength
    Slope (40% weight): close price linear regression consistency

Signals emitted (all appear in telemetry):
    adx_latest, adx_score, slope_normalized, slope_score,
    plus_di, minus_di, direction
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_state.detectors.base import BaseDetector, DetectorResult


class TrendDetector(BaseDetector):

    DETECTOR_NAME = "trend"
    MIN_BARS      = 30

    def __init__(self, adx_period: int = 14, slope_period: int = 20):
        self.adx_period   = adx_period
        self.slope_period = slope_period

    def _compute(self, df: pd.DataFrame) -> DetectorResult:
        adx, plus_di, minus_di = self._adx(df)
        adx_val    = float(adx.dropna().iloc[-1])
        plus_val   = float(plus_di.dropna().iloc[-1])
        minus_val  = float(minus_di.dropna().iloc[-1])

        # ADX → score mapping (non-linear — matches real market behavior)
        adx_score = self._safe_clip(
            (adx_val - 15.0) / 35.0    # 15 → 0.0, 50 → 1.0
        )

        slope_score, slope_norm = self._slope(df)
        direction = "up" if plus_val > minus_val else "down"

        trend_score = (adx_score * 0.60) + (slope_score * 0.40)

        return DetectorResult(
            score   = self._safe_clip(trend_score),
            signals = {
                "adx_latest":      round(adx_val, 2),
                "adx_score":       round(adx_score, 4),
                "slope_score":     round(slope_score, 4),
                "slope_normalized":round(slope_norm, 4),
                "plus_di":         round(plus_val, 2),
                "minus_di":        round(minus_val, 2),
            },
            meta    = {"direction": direction},
        )

    def _adx(self, df: pd.DataFrame):
        """Wilder's ADX with +DI and -DI."""
        high, low, close = df["high"], df["low"], df["close"]
        p = self.adx_period

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)

        up_move   = high.diff()
        down_move = -(low.diff())

        plus_dm  = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=df.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=df.index,
        )

        alpha    = 1.0 / p
        atr_s    = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di  = 100 * plus_dm.ewm( alpha=alpha, adjust=False).mean() / atr_s
        minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s

        di_sum  = (plus_di + minus_di).replace(0, np.nan)
        dx      = 100 * (plus_di - minus_di).abs() / di_sum
        adx     = dx.ewm(alpha=alpha, adjust=False).mean()

        return adx, plus_di, minus_di

    def _slope(self, df: pd.DataFrame):
        """Linear regression slope on recent closes."""
        closes = df["close"].values[-self.slope_period:]
        if len(closes) < 5:
            return 0.0, 0.0

        x         = np.arange(len(closes), dtype=float)
        slope     = np.polyfit(x, closes, 1)[0]
        rng       = closes.max() - closes.min()
        if rng < 1e-10:
            return 0.0, 0.0

        normalized = abs(slope) * self.slope_period / rng
        score      = self._safe_clip(normalized)
        return score, float(normalized)
