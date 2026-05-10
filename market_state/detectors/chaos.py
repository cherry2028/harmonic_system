"""
market_state/detectors/chaos.py
=================================
NewsChaosDetector — Detects external shock / news-driven chaos.

Detects the EFFECT of news on price structure, not the news itself.
A calendar API integration can be added in Phase 2 as a fourth signal.

When score > 0.40:
    → Invalidate ALL technical setups
    → HostileMarketGate blocks the pipeline
    → No new entries

Components:
    Gap detection   (40%): open vs prior close abnormality
    Volume spike    (35%): current volume vs rolling average
    Extreme candle  (25%): body size vs average
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_state.detectors.base import BaseDetector, DetectorResult


class NewsChaosDetector(BaseDetector):

    DETECTOR_NAME = "news_chaos"
    MIN_BARS      = 20

    def _compute(self, df: pd.DataFrame) -> DetectorResult:
        gap_score  = self._gap_score(df)
        vol_spike  = self._volume_spike(df)
        body_score = self._extreme_candle(df)

        chaos = (
            gap_score  * 0.40 +
            vol_spike  * 0.35 +
            body_score * 0.25
        )

        return DetectorResult(
            score   = self._safe_clip(chaos),
            signals = {
                "gap_score":       round(gap_score, 4),
                "volume_spike":    round(vol_spike, 4),
                "extreme_candle":  round(body_score, 4),
            },
        )

    def _gap_score(self, df: pd.DataFrame) -> float:
        close  = df["close"]
        open_  = df["open"]
        gaps   = (open_ - close.shift(1)).abs() / close.shift(1).replace(0, np.nan)
        gaps   = gaps.dropna()
        if len(gaps) < 5:
            return 0.0
        avg_gap    = float(gaps.mean())
        recent_gap = float(gaps.iloc[-1])
        if avg_gap < 1e-10:
            return 0.0
        ratio = self._safe_div(recent_gap, avg_gap)
        return self._safe_clip((ratio - 1.0) / 5.0)

    def _volume_spike(self, df: pd.DataFrame) -> float:
        volume = df["volume"]
        avg    = float(volume.rolling(20).mean().dropna().iloc[-1]) \
                 if len(volume) >= 20 else float(volume.mean())
        latest = float(volume.iloc[-1])
        if avg < 1e-10:
            return 0.0
        ratio = self._safe_div(latest, avg)
        if ratio < 2.0:
            return 0.0
        if ratio < 3.0:
            return 0.40
        if ratio < 5.0:
            return 0.70
        return 1.0

    def _extreme_candle(self, df: pd.DataFrame) -> float:
        bodies = (df["close"] - df["open"]).abs()
        avg    = float(bodies.rolling(20).mean().dropna().iloc[-1]) \
                 if len(bodies) >= 20 else float(bodies.mean())
        latest = float(bodies.iloc[-1])
        if avg < 1e-10:
            return 0.0
        ratio = self._safe_div(latest, avg)
        return self._safe_clip((ratio - 2.0) / 4.0)
