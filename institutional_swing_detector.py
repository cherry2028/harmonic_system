"""
institutional_swing_detector.py
===============================
Production-grade adaptive swing detection and structural candidate extraction
for institutional harmonic pattern trading on cryptocurrency markets.

Optimized for: BTCUSDT 1h
Target output: 1-3 high-quality XABCD candidates per 300-bar window

Architecture:
    AdaptiveSwingDetector
        ├── ATR-based volatility calibration
        ├── Dynamic strength scaling (2-8)
        ├── Multi-pass proximity filtering
        ├── Strict alternation enforcement
        └── Structural score culling

    StructuralCandidateExtractor
        ├── 14-stage rejection hierarchy (fastest first)
        ├── Universal ratio bounds (all patterns)
        ├── Pattern-specific fuzzy scoring
        └── Quality-ranked candidate output

    HarmonicFilterPipeline (convenience wrapper)

Compatibility:
    - Drop-in replacement for swing_detector.py
    - Compatible with harmonic_detector.py CandidateExtractor interface
    - SwingPoint dataclass matches existing pipeline expectations

Author: Institutional Trading Systems
Version: 2.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum, auto

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "SwingPoint",
    "SwingConfig",
    "AdaptiveSwingDetector",
    "StructuralCandidateExtractor",
    "HarmonicFilterPipeline",
    "PatternScore",
    "detect_swings",
    "extract_structural_candidates",
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True, slots=True)
class SwingPoint:
    """
    Harmonic swing pivot. Immutable. Compatible with existing pipeline.

    Fields:
        index: Bar index in source DataFrame.
        price: Extreme price (high for peaks, low for troughs).
        kind:  "high" or "low".
    """
    index: int
    price: float
    kind: str

    @property
    def type(self) -> str:
        """Alias for kind. Supports legacy pipeline references."""
        return self.kind


@dataclass
class PatternScore:
    """
    Fuzzy match score for a candidate against ideal pattern ratios.

    Attributes:
        pattern_name: Closest matching pattern (Gartley, Bat, Butterfly, Crab).
        total_score:  0.0-1.0 composite score (1.0 = perfect match).
        ab_xa_score:  Individual ratio match quality.
        bc_ab_score:  Individual ratio match quality.
        cd_bc_score:  Individual ratio match quality.
        ad_xa_score:  Individual ratio match quality.
        is_valid:     True if all ratios within pattern tolerance.
    """
    pattern_name: str
    total_score: float
    ab_xa_score: float
    bc_ab_score: float
    cd_bc_score: float
    ad_xa_score: float
    is_valid: bool


@dataclass
class SwingConfig:
    """
    Production configuration for institutional-grade detection.

    All thresholds calibrated for BTCUSDT 1h volatility regime.
    """
    # --- ATR calibration ---
    atr_period: int = 14
    atr_mult: float = 1.2
    min_move_floor: float = 0.005

    # --- Dynamic strength ---
    base_strength: int = 3
    strength_atr_scale: bool = True
    strength_min: int = 2
    strength_max: int = 6

    # --- Proximity filtering ---
    merge_proximity_bars: int = 2
    min_bar_separation: int = 2

    # --- Output targets ---
    target_swings: Tuple[int, int] = (8, 12)
    max_iterations: int = 5

    # --- Structural pre-filter (institutional) ---
    min_leg_pct: float = 0.005
    max_leg_ratio: float = 4.0
    min_pattern_bars: int = 10
    max_pattern_bars: int = 200
    max_candidates: int = 50

    # --- Universal ratio bounds (all patterns, 5% tolerance) ---
    ab_xa_min: float = 0.332
    ab_xa_max: float = 0.836
    bc_ab_min: float = 0.332
    bc_ab_max: float = 0.936
    cd_bc_min: float = 1.222
    cd_bc_max: float = 3.718
    ad_xa_min: float = 0.736
    ad_xa_max: float = 2.668

    # --- Hard constraints (non-negotiable) ---
    hard_ab_xa_ceiling: float = 1.0
    hard_bc_ab_ceiling: float = 1.0
    hard_cd_bc_floor: float = 0.5

    # --- Pattern ideal ratios (for fuzzy scoring) ---
    pattern_ideals: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "Gartley":   {"AB_XA": 0.618, "BC_AB": 0.618, "CD_BC": 1.618, "AD_XA": 0.786},
        "Bat":       {"AB_XA": 0.500, "BC_AB": 0.618, "CD_BC": 2.000, "AD_XA": 0.886},
        "Butterfly": {"AB_XA": 0.786, "BC_AB": 0.618, "CD_BC": 2.000, "AD_XA": 1.272},
        "Crab":      {"AB_XA": 0.618, "BC_AB": 0.618, "CD_BC": 3.140, "AD_XA": 1.618},
    })
    fuzzy_tolerance: float = 0.08


# ============================================================================
# ADAPTIVE SWING DETECTOR
# ============================================================================

class AdaptiveSwingDetector:
    """
    Detects swing pivots using ATR-adaptive thresholds with institutional
    discipline. Targets 8-12 clean swings from 300 BTCUSDT 1h candles.
    """

    def __init__(self, config: Optional[SwingConfig] = None):
        self.config = config or SwingConfig()

    def detect(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Main entry: detect swings from OHLCV data.

        Returns chronologically ordered SwingPoint list.
        """
        if len(df) < 50:
            logger.warning(f"Insufficient data: {len(df)} bars")
            return []

        atr, atr_pct = self._compute_atr(df)
        adaptive_move = max(atr_pct * self.config.atr_mult, self.config.min_move_floor)
        strength = self._compute_dynamic_strength(atr_pct)

        logger.info(
            f"Adaptive config | ATR={atr_pct:.4%} move={adaptive_move:.4%} "
            f"strength={strength}"
        )

        best_swings: List[SwingPoint] = []
        best_score = float("inf")

        for iteration in range(self.config.max_iterations):
            adjustment = 1.0 + (iteration * 0.10)
            threshold = adaptive_move * adjustment

            raw = self._find_raw_pivots(df, threshold, strength)
            merged = self._merge_nearby_pivots(raw)
            swings = self._enforce_alternation(merged)

            count = len(swings)
            target_mid = sum(self.config.target_swings) / 2.0
            score = abs(count - target_mid)

            logger.debug(f"Iter {iteration}: thr={threshold:.4%} swings={count}")

            if self.config.target_swings[0] <= count <= self.config.target_swings[1]:
                logger.info(f"Target achieved: {count} swings")
                return swings

            if score < best_score:
                best_score = score
                best_swings = swings

            if count < self.config.target_swings[0]:
                continue

        if len(best_swings) > self.config.target_swings[1]:
            best_swings = self._cull_by_structural_score(best_swings, df)

        logger.info(f"Returning {len(best_swings)} swings (best effort)")
        return best_swings

    def _compute_atr(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Compute ATR14 and ATR as percentage of current price."""
        high, low, close = df["high"], df["low"], df["close"]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.config.atr_period).mean().iloc[-1]

        if pd.isna(atr):
            atr = tr.mean()

        current_price = close.iloc[-1]
        atr_pct = atr / current_price if current_price and current_price > 0 else 0.01

        return float(atr), float(atr_pct)

    def _compute_dynamic_strength(self, atr_pct: float) -> int:
        """
        Scale pivot strength inversely with volatility.
        High volatility → lower strength (catch bigger moves).
        Low volatility → higher strength (avoid chop).
        """
        if not self.config.strength_atr_scale:
            return self.config.base_strength

        if atr_pct < 0.005:
            vol_factor = 2
        elif atr_pct < 0.015:
            vol_factor = 1
        elif atr_pct < 0.03:
            vol_factor = 0
        else:
            vol_factor = -1

        strength = self.config.base_strength + vol_factor
        return int(max(self.config.strength_min, min(self.config.strength_max, strength)))

    def _find_raw_pivots(
        self, df: pd.DataFrame, threshold_pct: float, strength: int
    ) -> List[SwingPoint]:
        """
        Find local extrema where price moves > threshold from last pivot.
        Uses strict local maxima/minima with confirmation window.
        """
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        pivots: List[SwingPoint] = []
        last_pivot_price = float(closes[0])
        last_pivot_type: Optional[str] = None

        for i in range(strength, len(df) - strength):
            window_highs = highs[i - strength : i + strength + 1]
            window_lows = lows[i - strength : i + strength + 1]

            is_peak = (
                highs[i] == window_highs.max()
                and highs[i] > window_highs[0]
                and highs[i] > window_highs[-1]
            )
            is_trough = (
                lows[i] == window_lows.min()
                and lows[i] < window_lows[0]
                and lows[i] < window_lows[-1]
            )

            if not (is_peak or is_trough):
                continue

            price = float(highs[i] if is_peak else lows[i])
            move_pct = abs(price - last_pivot_price) / last_pivot_price

            if move_pct < threshold_pct:
                continue

            kind = "high" if is_peak else "low"

            if kind == last_pivot_type and pivots:
                if (kind == "high" and price > pivots[-1].price) or (
                    kind == "low" and price < pivots[-1].price
                ):
                    pivots[-1] = SwingPoint(index=i, price=price, kind=kind)
                    last_pivot_price = price
                continue

            pivots.append(SwingPoint(index=i, price=price, kind=kind))
            last_pivot_price = price
            last_pivot_type = kind

        return pivots

    def _merge_nearby_pivots(self, pivots: List[SwingPoint]) -> List[SwingPoint]:
        """Merge same-type pivots within proximity bars. Keep most extreme."""
        if not pivots or self.config.merge_proximity_bars <= 0:
            return pivots

        merged: List[SwingPoint] = [pivots[0]]

        for current in pivots[1:]:
            last = merged[-1]

            if current.kind == last.kind and (
                current.index - last.index <= self.config.merge_proximity_bars
            ):
                if current.kind == "high" and current.price > last.price:
                    merged[-1] = current
                elif current.kind == "low" and current.price < last.price:
                    merged[-1] = current
            else:
                merged.append(current)

        return merged

    def _enforce_alternation(self, pivots: List[SwingPoint]) -> List[SwingPoint]:
        """Strict alternation: high, low, high, low... or reverse."""
        if not pivots:
            return []

        result: List[SwingPoint] = [pivots[0]]
        expected = "low" if pivots[0].kind == "high" else "high"

        for p in pivots[1:]:
            if p.kind == expected:
                result.append(p)
                expected = "low" if p.kind == "high" else "high"
            elif result[-1].kind == p.kind:
                if (p.kind == "high" and p.price > result[-1].price) or (
                    p.kind == "low" and p.price < result[-1].price
                ):
                    result[-1] = p

        return result

    def _cull_by_structural_score(
        self, swings: List[SwingPoint], df: pd.DataFrame
    ) -> List[SwingPoint]:
        """
        Force swing count into target range by removing lowest-quality pivots.
        Quality = average leg size + isolation from neighbors.
        """
        if len(swings) <= self.config.target_swings[1]:
            return swings

        atr, atr_pct = self._compute_atr(df)
        scores: List[Tuple[SwingPoint, float]] = []

        for i, swing in enumerate(swings):
            score = 0.0

            if 0 < i < len(swings) - 1:
                prev_move = abs(swing.price - swings[i - 1].price) / swings[i - 1].price
                next_move = abs(swings[i + 1].price - swing.price) / swing.price
                avg_move = (prev_move + next_move) / 2.0
                score = avg_move / atr_pct if atr_pct > 0 else 1.0

            isolation = 0
            if i > 0:
                isolation += min(swing.index - swings[i - 1].index, 10)
            if i < len(swings) - 1:
                isolation += min(swings[i + 1].index - swing.index, 10)
            score += isolation * 0.005

            scores.append((swing, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        kept = [s[0] for s in scores[: self.config.target_swings[1]]]
        kept.sort(key=lambda p: p.index)
        return kept


# ============================================================================
# STRUCTURAL CANDIDATE EXTRACTOR (INSTITUTIONAL)
# ============================================================================

class StructuralCandidateExtractor:
    """
    Extracts XABCD candidates with 14-stage institutional rejection hierarchy.

    Target: 1-3 high-quality candidates from 8-12 swings.
    All candidates pass universal ratio bounds + pattern fuzzy scoring.
    """

    def __init__(self, config: Optional[SwingConfig] = None):
        self.config = config or SwingConfig()

    def extract(
        self, swings: List[SwingPoint]
    ) -> List[Tuple[SwingPoint, SwingPoint, SwingPoint, SwingPoint, SwingPoint]]:
        """
        Returns validated (X, A, B, C, D) tuples ordered by structural quality.
        """
        if len(swings) < 5:
            logger.debug(f"Insufficient swings: {len(swings)}")
            return []

        candidates: List[Tuple[SwingPoint, ...]] = []
        recent = swings[-15:] if len(swings) > 15 else swings

        for i in range(len(recent) - 4):
            window = (
                recent[i],
                recent[i + 1],
                recent[i + 2],
                recent[i + 3],
                recent[i + 4],
            )

            if not self._passes_institutional_filter(window):
                continue

            candidates.append(window)

            if len(candidates) >= self.config.max_candidates:
                logger.warning(f"Max candidates cap: {self.config.max_candidates}")
                break

        logger.info(
            f"Structural extraction: {len(candidates)} candidates from {len(recent)} swings"
        )
        return candidates

    def _passes_institutional_filter(
        self, window: Tuple[SwingPoint, SwingPoint, SwingPoint, SwingPoint, SwingPoint]
    ) -> bool:
        """
        14-stage rejection hierarchy. Fastest checks first.
        Returns True only if candidate passes ALL stages.
        """
        X, A, B, C, D = window
        cfg = self.config

        # ---- Stage 1: Chronological order ----
        if not (X.index < A.index < B.index < C.index < D.index):
            return False

        # ---- Stage 2: Strict alternation ----
        kinds = [p.kind for p in window]
        if kinds not in (
            ["low", "high", "low", "high", "low"],
            ["high", "low", "high", "low", "high"],
        ):
            return False

        # ---- Stage 3: Pattern span limits ----
        total_bars = D.index - X.index
        if not (cfg.min_pattern_bars <= total_bars <= cfg.max_pattern_bars):
            return False

        # ---- Stage 4: Leg percentage moves ----
        xa = abs(A.price - X.price)
        ab = abs(B.price - A.price)
        bc = abs(C.price - B.price)
        cd = abs(D.price - C.price)

        prices = [X.price, A.price, B.price, C.price, D.price]
        min_price = min(prices)

        xa_pct = xa / min_price
        ab_pct = ab / min_price
        bc_pct = bc / min_price
        cd_pct = cd / min_price
        legs_pct = [xa_pct, ab_pct, bc_pct, cd_pct]

        if min(legs_pct) < cfg.min_leg_pct:
            return False

        # ---- Stage 5: Hard AB/XA ceiling (retracement, not expansion) ----
        if xa == 0:
            return False
        ab_xa = ab / xa
        if ab_xa >= cfg.hard_ab_xa_ceiling:
            return False

        # ---- Stage 6: Hard BC/AB ceiling ----
        if ab == 0:
            return False
        bc_ab = bc / ab
        if bc_ab >= cfg.hard_bc_ab_ceiling:
            return False

        # ---- Stage 7: Hard CD/BC floor (must extend, not truncate) ----
        if bc == 0:
            return False
        cd_bc = cd / bc
        if cd_bc < cfg.hard_cd_bc_floor:
            return False

        # ---- Stage 8: Universal AB/XA range ----
        if not (cfg.ab_xa_min <= ab_xa <= cfg.ab_xa_max):
            return False

        # ---- Stage 9: Universal BC/AB range ----
        if not (cfg.bc_ab_min <= bc_ab <= cfg.bc_ab_max):
            return False

        # ---- Stage 10: Universal CD/BC range ----
        if not (cfg.cd_bc_min <= cd_bc <= cfg.cd_bc_max):
            return False

        # ---- Stage 11: Leg ratio sanity ----
        max_leg = max(legs_pct)
        min_leg = min(legs_pct)
        if max_leg / min_leg > cfg.max_leg_ratio:
            return False

        # ---- Stage 12: Directional geometry ----
        if X.kind == "low":  # Bullish
            if not (X.price < B.price < A.price):
                return False
            if not (B.price < C.price < A.price):
                return False
            if not (D.price < C.price):
                return False
        else:  # Bearish
            if not (A.price < B.price < X.price):
                return False
            if not (A.price < C.price < B.price):
                return False
            if not (D.price > C.price):
                return False

        # ---- Stage 13: AD/XA completion zone ----
        ad = abs(D.price - X.price)
        if min_price == 0:
            return False
        ad_xa = ad / xa
        if not (cfg.ad_xa_min <= ad_xa <= cfg.ad_xa_max):
            return False

        # ---- Stage 14: Pattern fuzzy score (quality gate) ----
        score = self._compute_pattern_score(ab_xa, bc_ab, cd_bc, ad_xa)
        if not score.is_valid:
            return False

        return True

    def _compute_pattern_score(
        self, ab_xa: float, bc_ab: float, cd_bc: float, ad_xa: float
    ) -> PatternScore:
        """
        Fuzzy match candidate ratios against all pattern ideals.
        Returns best match with validity flag.
        """
        cfg = self.config
        best_score = -1.0
        best_pattern = ""
        best_details = {}

        for name, ideals in cfg.pattern_ideals.items():
            scores = {}
            valid = True

            for ratio_name, ideal_val in ideals.items():
                actual = {"AB_XA": ab_xa, "BC_AB": bc_ab, "CD_BC": cd_bc, "AD_XA": ad_xa}[
                    ratio_name
                ]
                deviation = abs(actual - ideal_val) / ideal_val
                ratio_score = max(0.0, 1.0 - (deviation / cfg.fuzzy_tolerance))
                scores[ratio_name] = ratio_score
                if deviation > cfg.fuzzy_tolerance:
                    valid = False

            total = sum(scores.values()) / len(scores)

            if total > best_score:
                best_score = total
                best_pattern = name
                best_details = scores

        return PatternScore(
            pattern_name=best_pattern,
            total_score=best_score,
            ab_xa_score=best_details.get("AB_XA", 0.0),
            bc_ab_score=best_details.get("BC_AB", 0.0),
            cd_bc_score=best_details.get("CD_BC", 0.0),
            ad_xa_score=best_details.get("AD_XA", 0.0),
            is_valid=best_score > 0.6,
        )


# ============================================================================
# UNIFIED PIPELINE
# ============================================================================

class HarmonicFilterPipeline:
    """
    Convenience wrapper combining swing detection + structural extraction.
    Single-call interface for production deployment.
    """

    def __init__(self, config: Optional[SwingConfig] = None):
        self.config = config or SwingConfig()
        self.swing_detector = AdaptiveSwingDetector(self.config)
        self.candidate_extractor = StructuralCandidateExtractor(self.config)

    def detect(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Return clean swings."""
        return self.swing_detector.detect(df)

    def detect_and_extract(
        self, df: pd.DataFrame
    ) -> Tuple[List[SwingPoint], List[Tuple[SwingPoint, ...]]]:
        """Return (swings, structural_candidates)."""
        swings = self.swing_detector.detect(df)
        candidates = self.candidate_extractor.extract(swings)
        return swings, candidates


# ============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ============================================================================

def detect_swings(
    df: pd.DataFrame, config: Optional[SwingConfig] = None
) -> List[SwingPoint]:
    """One-shot adaptive swing detection."""
    return AdaptiveSwingDetector(config).detect(df)


def extract_structural_candidates(
    swings: List[SwingPoint], config: Optional[SwingConfig] = None
) -> List[Tuple[SwingPoint, SwingPoint, SwingPoint, SwingPoint, SwingPoint]]:
    """One-shot XABCD extraction with institutional geometric pre-filtering."""
    return StructuralCandidateExtractor(config).extract(swings)