"""
swing_detector.py
=================
Adaptive ATR-based swing detection with institutional-grade
structural pre-filtering for harmonic pattern detection.

Drop-in replacement for swing detection and candidate extraction
in the harmonic pattern pipeline.

Compatible with:
    - patterns.patterns.harmonic_patterns.SwingPoint
    - harmonic_detector.py CandidateExtractor interface
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "SwingPoint",
    "SwingConfig",
    "AdaptiveSwingDetector",
    "StructuralCandidateExtractor",
    "HarmonicSwingPipeline",
    "detect_swings",
    "extract_structural_candidates",
]


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SwingPoint:
    """
    Harmonic swing pivot. Compatible with existing harmonic_patterns.py pipeline.

    Fields:
        index: Bar index in the source DataFrame.
        price: The extreme price (high for peaks, low for troughs).
        kind:  "high" or "low".
    """

    index: int
    price: float
    kind: str

    @property
    def type(self) -> str:
        """Alias for kind. Supports pipelines referencing p.type."""
        return self.kind


@dataclass
class SwingConfig:
    """
    Production configuration for adaptive swing detection and candidate filtering.
    """

    # --- ATR adaptation ---
    atr_period: int = 14
    atr_mult: float = 0.8
    min_move_floor: float = 0.003

    # --- Dynamic strength ---
    base_strength: int = 3
    strength_atr_scale: bool = True

    # --- Proximity filtering ---
    merge_proximity_bars: int = 2
    min_bar_separation: int = 1

    # --- Output targets ---
    target_swings: Tuple[int, int] = (8, 15)
    max_iterations: int = 5

    # --- Institutional geometric pre-filter (candidate level) ---
    min_leg_pct: float = 0.003
    max_leg_ratio: float = 5.0
    min_pattern_bars: int = 5
    max_pattern_bars: int = 300
    max_candidates: int = 200


# ---------------------------------------------------------------------------
# Adaptive Swing Detector
# ---------------------------------------------------------------------------

class AdaptiveSwingDetector:
    """
    Detects swing pivots using ATR-adaptive thresholds, dynamic strength scaling,
    proximity merging, and strict alternation enforcement.
    """

    def __init__(self, config: Optional[SwingConfig] = None):
        self.config = config or SwingConfig()

    def detect(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Detect swings from OHLCV data.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close'].
                Index may be datetime or integer.

        Returns:
            Chronologically ordered list of SwingPoint objects.
        """
        if len(df) < 50:
            logger.warning(f"Insufficient data for swing detection: {len(df)} bars")
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
            loosening = 1.0 + (iteration * 0.12)
            threshold = adaptive_move * loosening

            raw = self._find_raw_pivots(df, threshold, strength)
            merged = self._merge_nearby_pivots(raw)
            swings = self._enforce_alternation(merged)

            count = len(swings)
            target_mid = (self.config.target_swings[0] + self.config.target_swings[1]) / 2.0
            score = abs(count - target_mid)

            logger.debug(
                f"Iter {iteration}: threshold={threshold:.4%} swings={count}"
            )

            if self.config.target_swings[0] <= count <= self.config.target_swings[1]:
                logger.info(f"Target swing count achieved: {count}")
                return swings

            if score < best_score:
                best_score = score
                best_swings = swings

            if count < self.config.target_swings[0]:
                logger.debug("Swing count below target; loosening threshold.")
                continue

        if len(best_swings) > self.config.target_swings[1]:
            best_swings = self._cull_by_structural_score(best_swings, df)

        logger.info(f"Returning {len(best_swings)} swings (best effort)")
        return best_swings

    def _compute_atr(self, df: pd.DataFrame) -> Tuple[float, float]:
        high = df["high"]
        low = df["low"]
        close = df["close"]

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
        if not self.config.strength_atr_scale:
            return self.config.base_strength

        if atr_pct < 0.005:
            vol_factor = 0
        elif atr_pct < 0.015:
            vol_factor = 1
        elif atr_pct < 0.03:
            vol_factor = 2
        else:
            vol_factor = 3

        strength = self.config.base_strength + vol_factor
        return int(max(2, min(8, strength)))

    def _find_raw_pivots(
        self, df: pd.DataFrame, threshold_pct: float, strength: int
    ) -> List[SwingPoint]:
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


# ---------------------------------------------------------------------------
# Institutional Candidate Extractor
# ---------------------------------------------------------------------------

class StructuralCandidateExtractor:
    """
    Extracts XABCD 5-point candidates from swings and applies
    institutional-grade geometric pre-filtering.

    Drop-in replacement for harmonic_detector.CandidateExtractor.
    """

    def __init__(self, config: Optional[SwingConfig] = None):
        self.config = config or SwingConfig()

    def extract(
        self, swings: List[SwingPoint]
    ) -> List[Tuple[SwingPoint, SwingPoint, SwingPoint, SwingPoint, SwingPoint]]:
        """
        Returns validated (X, A, B, C, D) tuples.

        All returned candidates pass:
          - Strict alternation
          - Chronological ordering
          - Pattern span limits
          - Minimum leg size
          - AB/XA < 1.0  (B does not exceed X)
          - BC/AB < 1.0  (C does not exceed A)
          - CD/BC >= 0.5 (D extends BC meaningfully)
          - Directional geometry (bullish / bearish structure)
          - Leg ratio sanity (no blow-up legs)
        """
        if len(swings) < 5:
            logger.debug(f"Insufficient swings for XABCD: {len(swings)}")
            return []

        candidates: List[Tuple[SwingPoint, SwingPoint, SwingPoint, SwingPoint, SwingPoint]] = []
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
                logger.warning(
                    f"Max candidates cap reached ({self.config.max_candidates})"
                )
                break

        logger.info(
            f"Structural extraction: {len(candidates)} candidates from {len(recent)} swings"
        )
        return candidates

    def _passes_institutional_filter(
        self, window: Tuple[SwingPoint, SwingPoint, SwingPoint, SwingPoint, SwingPoint]
    ) -> bool:
        X, A, B, C, D = window

        # ---- 1. Strict alternation ----------------------------------------
        kinds = [p.kind for p in window]
        if kinds not in (
            ["low", "high", "low", "high", "low"],
            ["high", "low", "high", "low", "high"],
        ):
            return False

        # ---- 2. Chronological order ---------------------------------------
        indices = [p.index for p in window]
        if indices != sorted(indices):
            return False

        # ---- 3. Pattern span limits ---------------------------------------
        total_bars = D.index - X.index
        if not (self.config.min_pattern_bars <= total_bars <= self.config.max_pattern_bars):
            return False

        # ---- 4. Leg percentage moves --------------------------------------
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

        if min(xa_pct, ab_pct, bc_pct, cd_pct) < self.config.min_leg_pct:
            return False

        # ---- 5. Retracement bounds (institutional) ------------------------
        # AB must be a retracement of XA: B must not exceed X
        if xa == 0:
            return False

        ab_xa = ab / xa
        if ab_xa >= 1.0 or ab_xa < 0.1:
            return False

        # BC must be a retracement of AB: C must not exceed A
        if ab == 0:
            return False

        bc_ab = bc / ab
        if bc_ab >= 1.0 or bc_ab < 0.1:
            return False

        # CD must extend BC meaningfully
        if bc == 0:
            return False

        cd_bc = cd / bc
        if cd_bc < 0.5 or cd_bc > self.config.max_leg_ratio:
            return False

        # ---- 6. Directional geometry --------------------------------------
        if X.kind == "low":  # Bullish: X low, A high, B low, C high, D low
            if not (X.price < B.price < A.price):
                return False
            if not (B.price < C.price < A.price):
                return False
            if not (D.price < C.price):
                return False
        else:  # Bearish: X high, A low, B high, C low, D high
            if not (A.price < B.price < X.price):
                return False
            if not (A.price < C.price < B.price):
                return False
            if not (D.price > C.price):
                return False

        # ---- 7. Leg ratio sanity (prevent blow-ups) -----------------------
        legs = [xa_pct, ab_pct, bc_pct, cd_pct]
        max_leg = max(legs)
        min_leg = min(legs)
        if max_leg / min_leg > self.config.max_leg_ratio:
            return False

        return True


# ---------------------------------------------------------------------------
# Unified Pipeline
# ---------------------------------------------------------------------------

class HarmonicSwingPipeline:
    """
    Convenience wrapper that combines swing detection and structural candidate
    extraction into a single callable interface.
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


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

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