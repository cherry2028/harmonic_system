"""
market_state/detectors/compression.py
=======================================
CompressionDetector — Detects volatility contraction / coiling state.

Compression = ATR declining BELOW historical percentile + Bollinger squeeze.
Price is coiling — energy building for eventual breakout.

DO NOT trade harmonic reversals during high compression.
The pattern forms but never triggers cleanly.
Wait for expansion signal before acting.

Components:
    ATR percentile (55%): current ATR vs recent history (low = compression)
    BB/KC squeeze  (45%): Bollinger Bands inside Keltner Channels
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_state.detectors.base import BaseDetector, DetectorResult


class CompressionDetector(BaseDetector):

    DETECTOR_NAME = "compression"
    MIN_BARS      = 35

    def __init__(self, atr_period: int = 14, lookback: int = 50):
        self.atr_period = atr_period
        self.lookback   = lookback

    def _compute(self, df: pd.DataFrame) -> DetectorResult:
        atr_series = self._atr(df)
        atr_score  = self._atr_compression_score(atr_series)
        bb_score   = self._bollinger_keltner_squeeze(df, atr_series)

        compression = (atr_score * 0.55) + (bb_score * 0.45)

        return DetectorResult(
            score   = self._safe_clip(compression),
            signals = {
                "atr_score":  round(atr_score, 4),
                "bb_kc_score": round(bb_score, 4),
            },
        )

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / self.atr_period, adjust=False).mean()

    def _atr_compression_score(self, atr: pd.Series) -> float:
        valid = atr.dropna()
        if len(valid) < 10:
            return 0.0
        lookback = valid.iloc[-self.lookback:]
        current  = float(valid.iloc[-1])
        pctl     = float((lookback <= current).mean())
        # Only score when ATR is in bottom quartile
        if pctl > 0.40:
            return 0.0
        return self._safe_clip((0.40 - pctl) / 0.40)

    def _bollinger_keltner_squeeze(
        self, df: pd.DataFrame, atr: pd.Series
    ) -> float:
        """
        John Carter squeeze: Bollinger Bands inside Keltner Channels.
        When BB is inside KC, price is compressed.
        """
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # Bollinger Bands (±2 std)
        bb_mid   = close.rolling(20).mean()
        bb_std   = close.rolling(20).std()
        bb_upper = bb_mid + 2.0 * bb_std
        bb_lower = bb_mid - 2.0 * bb_std

        # Keltner Channels (±1.5 ATR)
        kc_upper = bb_mid + 1.5 * atr
        kc_lower = bb_mid - 1.5 * atr

        # Squeeze: BB inside KC
        in_squeeze = (
            (bb_upper < kc_upper) & (bb_lower > kc_lower)
        ).dropna()

        if len(in_squeeze) == 0:
            return 0.0

        # Ratio of last 5 bars in squeeze
        recent = in_squeeze.iloc[-5:]
        return float(recent.sum() / len(recent))
