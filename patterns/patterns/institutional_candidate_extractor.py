"""
institutional_candidate_extractor.py
====================================
Production-grade XABCD candidate extraction with institutional structural
filtering for harmonic pattern detection on cryptocurrency markets.

Optimizations for BTCUSDT 1h:
    - 14-stage geometric validation hierarchy
    - Volatility-normalized leg quality scoring
    - Time symmetry enforcement
    - Directional momentum consistency checks
    - Ranked candidate output (best first)

Drop-in replacement for CandidateExtractor in harmonic_detector.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "SwingPoint",
    "ExtractorConfig",
    "CandidateScore",
    "StructuralCandidateExtractor",
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True, slots=True)
class SwingPoint:
    """Harmonic swing pivot. Compatible with existing pipeline."""
    index: int
    price: float
    kind: str

    @property
    def type(self) -> str:
        return self.kind


@dataclass
class ExtractorConfig:
    """
    Institutional configuration for candidate extraction.
    Calibrated for BTCUSDT 1h volatility regime.
    """
    # --- Span constraints ---
    min_pattern_bars: int = 10
    max_pattern_bars: int = 200

    # --- Leg size constraints ---
    min_leg_pct: float = 0.008          # 0.5% absolute minimum
    min_leg_atr_mult: float = 0.3       # Leg must be >= 0.3 * ATR14
    max_leg_ratio: float = 4.0          # No leg > 4x another

    # --- Hard geometric ceilings/floors (non-negotiable) ---
    hard_ab_xa_ceiling: float = 1.0     # B must not exceed X
    hard_bc_ab_ceiling: float = 1.0     # C must not exceed A
    hard_cd_bc_floor: float = 0.618       # CD must EXTEND beyond BC (was 0.5)

    # --- Universal ratio bounds (all patterns, 5% tolerance) ---
    ab_xa_min: float = 0.382            # Raised from 0.332 (Bat lower bound)
    ab_xa_max: float = 0.836            # Butterfly upper bound
    bc_ab_min: float = 0.382            # All patterns
    bc_ab_max: float = 0.886            # All patterns
    cd_bc_min: float = 1.272            # Gartley lower bound (raised from 1.222)
    cd_bc_max: float = 3.618            # Crab upper bound
    ad_xa_min: float = 0.786            # Gartley lower bound
    ad_xa_max: float = 2.618            # Crab upper bound

    # --- Swing symmetry constraints ---
    impulse_symmetry_min: float = 0.4   # XA / BC ratio must be within [0.4, 2.5]
    impulse_symmetry_max: float = 2.5   # Prevents one impulse dwarfing the other
    retrace_extension_min: float = 1.0  # CD / AB must be >= 1.0 (extension > retrace)

    # --- Time symmetry constraints ---
    max_time_leg_ratio: float = 4.0     # No time leg > 4x another
    min_time_per_leg: int = 2           # Minimum 2 bars per leg

    # --- Volatility normalization ---
    atr_period: int = 14
    min_volatility_score: float = 0.5   # Leg / ATR must exceed this

    # --- Directional momentum ---
    max_momentum_divergence: float = 2.0  # |XA| / |BC| or |AB| / |CD| max ratio

    # --- Output control ---
    max_candidates: int = 20
    min_quality_score: float = 0.45     # Composite score gate
    top_n_candidates: int = 5           # Return only top N after ranking

    # --- Pattern ideals for fuzzy scoring ---
    pattern_ideals: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "Gartley":   {"AB_XA": 0.618, "BC_AB": 0.618, "CD_BC": 1.618, "AD_XA": 0.786},
        "Bat":       {"AB_XA": 0.500, "BC_AB": 0.618, "CD_BC": 2.000, "AD_XA": 0.886},
        "Butterfly": {"AB_XA": 0.786, "BC_AB": 0.618, "CD_BC": 2.000, "AD_XA": 1.272},
        "Crab":      {"AB_XA": 0.618, "BC_AB": 0.618, "CD_BC": 3.140, "AD_XA": 1.618},
    })
    fuzzy_tolerance: float = 0.08


@dataclass
class CandidateScore:
    """Composite quality score for a candidate XABCD structure."""
    total_score: float = 0.0
    ratio_score: float = 0.0
    time_symmetry_score: float = 0.0
    volatility_score: float = 0.0
    momentum_score: float = 0.0
    geometry_score: float = 0.0
    pattern_name: str = ""
    is_valid: bool = False


# ============================================================================
# INSTITUTIONAL CANDIDATE EXTRACTOR
# ============================================================================

class StructuralCandidateExtractor:
    """
    Extracts XABCD candidates with 14-stage institutional rejection hierarchy
    plus advanced geometric, temporal, and volatility filtering.

    Target: 1-3 high-quality candidates from 8-12 clean swings.
    """

    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.config = config or ExtractorConfig()

    def extract(
        self,
        swings: List[SwingPoint],
        df: Optional[pd.DataFrame] = None,
    ) -> List[Tuple[SwingPoint, SwingPoint, SwingPoint, SwingPoint, SwingPoint]]:
        """
        Returns validated (X, A, B, C, D) tuples ranked by structural quality.

        Args:
            swings: Chronologically ordered alternating swing points.
            df: Optional OHLCV DataFrame for ATR-based volatility scoring.
                If provided, enables volatility-normalized leg quality.

        Returns:
            Top N candidates sorted by composite quality score (best first).
        """
        if len(swings) < 5:
            logger.debug(f"Insufficient swings: {len(swings)}")
            return []

        atr_pct = self._compute_atr_pct(df) if df is not None else None
        recent = swings[-15:] if len(swings) > 15 else swings

        scored_candidates: List[Tuple[Tuple[SwingPoint, ...], CandidateScore]] = []

        for i in range(len(recent) - 4):
            window = (
                recent[i],
                recent[i + 1],
                recent[i + 2],
                recent[i + 3],
                recent[i + 4],
            )

            score = self._evaluate_candidate(window, atr_pct)
            if score.is_valid:
                scored_candidates.append((window, score))

        if not scored_candidates:
            logger.info(f"Zero candidates passed institutional filter from {len(recent)} swings")
            return []

        # Sort by total score descending
        scored_candidates.sort(key=lambda x: x[1].total_score, reverse=True)

        # Return top N
        top_n = scored_candidates[: self.config.top_n_candidates]
        result = [w for w, _ in top_n]

        logger.info(
            f"Extracted {len(result)} candidates (from {len(scored_candidates)} passing) "
            f"| best_score={top_n[0][1].total_score:.3f}"
        )
        return result

    # ------------------------------------------------------------------ #
    # Core Evaluation Pipeline                                           #
    # ------------------------------------------------------------------ #

    def _evaluate_candidate(
        self,
        window: Tuple[SwingPoint, SwingPoint, SwingPoint, SwingPoint, SwingPoint],
        atr_pct: Optional[float],
    ) -> CandidateScore:
        """
        14-stage geometric validation + composite scoring.
        Returns CandidateScore with is_valid=True only if ALL stages pass.
        """
        X, A, B, C, D = window
        cfg = self.config
        score = CandidateScore()

        # =====================================================================
        # PHASE 1: FAST REJECTION (O(1) checks — cheapest first)
        # =====================================================================

        # Stage 1: Chronological order
        if not (X.index < A.index < B.index < C.index < D.index):
            return score

        # Stage 2: Strict alternation
        kinds = [p.kind for p in window]
        is_bullish = kinds == ["low", "high", "low", "high", "low"]
        is_bearish = kinds == ["high", "low", "high", "low", "high"]
        if not (is_bullish or is_bearish):
            return score

        # Stage 3: Time span limits
        total_bars = D.index - X.index
        if not (cfg.min_pattern_bars <= total_bars <= cfg.max_pattern_bars):
            return score

        # Stage 4: Minimum time per leg
        t_xa = A.index - X.index
        t_ab = B.index - A.index
        t_bc = C.index - B.index
        t_cd = D.index - C.index
        time_legs = [t_xa, t_ab, t_bc, t_cd]
        if any(t < cfg.min_time_per_leg for t in time_legs):
            return score

        # =====================================================================
        # PHASE 2: GEOMETRIC PROGRESSION VALIDATION
        # =====================================================================

        # Compute absolute leg sizes
        xa = abs(A.price - X.price)
        ab = abs(B.price - A.price)
        bc = abs(C.price - B.price)
        cd = abs(D.price - C.price)
        ad = abs(D.price - X.price)

        prices = [X.price, A.price, B.price, C.price, D.price]
        min_price = min(prices)
        if min_price == 0:
            return score

        # Normalize to percentages
        xa_pct = xa / min_price
        ab_pct = ab / min_price
        bc_pct = bc / min_price
        cd_pct = cd / min_price
        ad_pct = ad / min_price
        legs_pct = [xa_pct, ab_pct, bc_pct, cd_pct]

        # Stage 5: Minimum leg percentage (absolute floor)
        if min(legs_pct) < cfg.min_leg_pct:
            return score

        # Stage 6: Hard geometric progression (impulse/retracement logic)
        if is_bullish:
            # XA is up impulse: X.low < A.high
            # AB is down retracement: B.low must be > X.low and < A.high
            if not (X.price < B.price < A.price):
                return score
            # BC is up impulse: C.high must be > B.low and < A.high
            if not (B.price < C.price < A.price):
                return score
            # CD is down extension: D.low must be < C.high
            if not (D.price < C.price):
                return score
        else:  # bearish
            # XA is down impulse: X.high > A.low
            # AB is up retracement: B.high must be < X.high and > A.low
            if not (A.price < B.price < X.price):
                return score
            # BC is down impulse: C.low must be > A.low and < B.high
            if not (A.price < C.price < B.price):
                return score
            # CD is up extension: D.high must be > C.low
            if not (D.price > C.price):
                return score

        # =====================================================================
        # PHASE 3: RATIO CONSTRAINTS (hard ceilings/floors)
        # =====================================================================

        if xa == 0 or ab == 0 or bc == 0:
            return score

        ab_xa = ab / xa
        bc_ab = bc / ab
        cd_bc = cd / bc
        ad_xa = ad / xa

        # Stage 7: Hard AB/XA ceiling — B must not exceed X (retracement, not expansion)
        if ab_xa >= cfg.hard_ab_xa_ceiling:
            return score

        # Stage 8: Hard BC/AB ceiling — C must not exceed A
        if bc_ab >= cfg.hard_bc_ab_ceiling:
            return score

        # Stage 9: Hard CD/BC floor — CD must EXTEND beyond BC (not truncate)
        if cd_bc < cfg.hard_cd_bc_floor:
            return score

        # Stage 10: Universal ratio bounds (all patterns)
        if not (cfg.ab_xa_min <= ab_xa <= cfg.ab_xa_max):
            return score
        if not (cfg.bc_ab_min <= bc_ab <= cfg.bc_ab_max):
            return score
        if not (cfg.cd_bc_min <= cd_bc <= cfg.cd_bc_max):
            return score
        if not (cfg.ad_xa_min <= ad_xa <= cfg.ad_xa_max):
            return score

        # =====================================================================
        # PHASE 4: ADVANCED STRUCTURAL FILTERS
        # =====================================================================

        # Stage 11: Leg ratio sanity (prevent blow-ups)
        max_leg = max(legs_pct)
        min_leg = min(legs_pct)
        if max_leg / min_leg > cfg.max_leg_ratio:
            return score

        # Stage 12: Swing symmetry — impulses XA and BC should be comparable
        if bc > 0:
            impulse_ratio = xa / bc
            if not (cfg.impulse_symmetry_min <= impulse_ratio <= cfg.impulse_symmetry_max):
                return score

        # Stage 13: Retracement vs extension proportionality
        # CD (extension) should be >= AB (retracement) in magnitude
        if ab > 0:
            retrace_extension = cd / ab
            if retrace_extension < cfg.retrace_extension_min:
                return score

        # Stage 14: Time symmetry — no leg dominates temporally
        max_time = max(time_legs)
        min_time = min(time_legs)
        if min_time > 0 and max_time / min_time > cfg.max_time_leg_ratio:
            return score

        # =====================================================================
        # PHASE 5: COMPOSITE SCORING (candidate ranking)
        # =====================================================================

        score = self._compute_composite_score(
            xa_pct, ab_pct, bc_pct, cd_pct, ad_pct,
            ab_xa, bc_ab, cd_bc, ad_xa,
            time_legs, atr_pct, is_bullish
        )

        return score

    # ------------------------------------------------------------------ #
    # Composite Scoring Engine                                           #
    # ------------------------------------------------------------------ #

    def _compute_composite_score(
        self,
        xa_pct: float, ab_pct: float, bc_pct: float, cd_pct: float, ad_pct: float,
        ab_xa: float, bc_ab: float, cd_bc: float, ad_xa: float,
        time_legs: List[int],
        atr_pct: Optional[float],
        is_bullish: bool,
    ) -> CandidateScore:
        """
        Compute multi-factor quality score for ranking candidates.
        All sub-scores are 0.0-1.0. Total is weighted average.
        """
        cfg = self.config
        score = CandidateScore()

        # --- Sub-score 1: Ratio proximity to ideal pattern ---
        best_pattern = ""
        best_ratio_score = -1.0

        for name, ideals in cfg.pattern_ideals.items():
            deviations = []
            for ratio_name, ideal_val in ideals.items():
                actual = {"AB_XA": ab_xa, "BC_AB": bc_ab, "CD_BC": cd_bc, "AD_XA": ad_xa}[ratio_name]
                deviation = abs(actual - ideal_val) / ideal_val
                deviations.append(max(0.0, 1.0 - (deviation / cfg.fuzzy_tolerance)))

            pattern_score = sum(deviations) / len(deviations)
            if pattern_score > best_ratio_score:
                best_ratio_score = pattern_score
                best_pattern = name

        score.ratio_score = best_ratio_score
        score.pattern_name = best_pattern

        # --- Sub-score 2: Time symmetry ---
        mean_time = sum(time_legs) / len(time_legs)
        time_variance = sum((t - mean_time) ** 2 for t in time_legs) / len(time_legs)
        time_std = time_variance ** 0.5
        cv = time_std / mean_time if mean_time > 0 else 1.0
        score.time_symmetry_score = max(0.0, 1.0 - cv)

        # --- Sub-score 3: Volatility-normalized leg quality ---
        if atr_pct and atr_pct > 0:
            leg_ratios = [xa_pct / atr_pct, ab_pct / atr_pct, bc_pct / atr_pct, cd_pct / atr_pct]
            min_ratio = min(leg_ratios)
            score.volatility_score = min(1.0, min_ratio / cfg.min_volatility_score)
        else:
            score.volatility_score = 1.0

        # --- Sub-score 4: Directional momentum consistency ---
        # Impulses (XA, BC) should have similar magnitude
        # Retracements (AB) and extensions (CD) should be proportional
        if bc_pct > 0 and ab_pct > 0:
            momentum_consistency = 1.0 - min(
                abs(xa_pct - bc_pct) / max(xa_pct, bc_pct, 1e-9),
                0.5
            )
            extension_quality = min(1.0, cd_pct / (ab_pct * 1.5)) if ab_pct > 0 else 0.0
            score.momentum_score = (momentum_consistency + extension_quality) / 2.0
        else:
            score.momentum_score = 0.0

        # --- Sub-score 5: Geometric purity ---
        # Penalize ratios near boundaries (prefer center of valid ranges)
        ab_xa_center = (cfg.ab_xa_min + cfg.ab_xa_max) / 2.0
        bc_ab_center = (cfg.bc_ab_min + cfg.bc_ab_max) / 2.0
        cd_bc_center = (cfg.cd_bc_min + cfg.cd_bc_max) / 2.0

        ab_xa_purity = 1.0 - abs(ab_xa - ab_xa_center) / (cfg.ab_xa_max - cfg.ab_xa_min)
        bc_ab_purity = 1.0 - abs(bc_ab - bc_ab_center) / (cfg.bc_ab_max - cfg.bc_ab_min)
        cd_bc_purity = 1.0 - abs(cd_bc - cd_bc_center) / (cfg.cd_bc_max - cfg.cd_bc_min)

        score.geometry_score = max(0.0, (ab_xa_purity + bc_ab_purity + cd_bc_purity) / 3.0)

        # --- Weighted total ---
        score.total_score = (
            score.ratio_score * 0.35 +
            score.time_symmetry_score * 0.15 +
            score.volatility_score * 0.20 +
            score.momentum_score * 0.15 +
            score.geometry_score * 0.15
        )

        score.is_valid = score.total_score >= cfg.min_quality_score
        return score

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _compute_atr_pct(self, df: Optional[pd.DataFrame]) -> Optional[float]:
        """Compute ATR as percentage of current price."""
        if df is None or len(df) < self.config.atr_period + 1:
            return None

        high, low, close = df["high"], df["low"], df["close"]
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.config.atr_period).mean().iloc[-1]

        if pd.isna(atr):
            return None

        current_price = close.iloc[-1]
        return float(atr / current_price) if current_price and current_price > 0 else None