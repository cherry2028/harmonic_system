"""
market_state/detectors/expansion.py
=====================================
ExpansionDetector — Detects volatility expansion / breakout state.

Expansion = ATR spiking ABOVE historical percentile.
This is the TRANSITION from compression to expansion.
Not just "high volatility" — the rapid change matters.

Components:
    ATR percentile rank (60%): current ATR vs recent history
    Candle body ratio   (40%): recent bodies vs average
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_state.detectors.base import BaseDetector, DetectorResult


class ExpansionDetector(BaseDetector):

    DETECTOR_NAME = "expansion"
    MIN_BARS      = 30

    def __init__(self, atr_period: int = 14, lookback: int = 50):
        self.atr_period = atr_period
        self.lookback   = lookback

    def _compute(self, df: pd.DataFrame) -> DetectorResult:
        atr_series  = self._atr(df)
        atr_score   = self._atr_percentile_score(atr_series)
        body_score  = self._body_ratio_score(df)

        expansion = (atr_score * 0.60) + (body_score * 0.40)

        atr_latest = float(atr_series.dropna().iloc[-1])
        price      = float(df["close"].iloc[-1])

        return DetectorResult(
            score   = self._safe_clip(expansion),
            signals = {
                "atr_score":    round(atr_score, 4),
                "body_score":   round(body_score, 4),
                "atr_latest":   round(atr_latest, 4),
                "atr_pct":      round(
                    self._safe_div(atr_latest, price), 4
                ),
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

    def _atr_percentile_score(self, atr: pd.Series) -> float:
        valid = atr.dropna()
        if len(valid) < 10:
            return 0.0
        lookback = valid.iloc[-self.lookback:]
        current  = float(valid.iloc[-1])
        pctl     = float((lookback <= current).mean())
        # Only score high when ATR is in top quartile
        if pctl < 0.60:
            return 0.0
        return self._safe_clip((pctl - 0.60) / 0.40)

    def _body_ratio_score(self, df: pd.DataFrame) -> float:
        bodies  = (df["close"] - df["open"]).abs()
        avg     = float(bodies.rolling(20).mean().dropna().iloc[-1]) \
                  if len(bodies) >= 20 else float(bodies.mean())
        if avg < 1e-10:
            return 0.0
        ratio = float(bodies.iloc[-1]) / avg
        # Ratio > 2.0 starts scoring; ratio > 3.0 = full score
        return self._safe_clip((ratio - 1.0) / 2.0)
