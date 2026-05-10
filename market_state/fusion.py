"""
market_state/fusion.py
=======================
MarketStateFusion — Combines Raw Detector Scores into MarketStateVector

Responsibilities:
    1. Accept one DetectorResult per state (6 total)
    2. Apply mutual exclusivity corrections
    3. Apply probability floor (no state ever = 0.0)
    4. Normalize so all six states sum to 1.0
    5. Construct and return MarketStateVector

Design rules:
    - Fusion is deterministic: same inputs → same output. Always.
    - No ML, no learned weights, no external data.
    - All correction logic is documented with rationale.
    - Transparent: FusionResult carries the full audit trail
      (raw scores, adjusted scores, final probabilities).

Mutual exclusivity relationships:
    news_chaos ↔ all others : Chaos suppresses everything
    trending   ↔ ranging    : Strongly inversely related
    expansion  ↔ compression: Cannot be simultaneously high
    reversal   context      : Requires trend or range to make sense

Probability floor = 0.02 (2%)
    Reason: zero probability for any state is epistemically wrong.
    We are never 100% certain a state is absent.
    2% is minimal but keeps the system from being overconfident.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from market_state.detectors.base import DetectorResult
from market_state.vector import MarketStateVector
from market_state.probability_utils import SimplexProjector

logger = logging.getLogger("market_state.fusion")

# ---------------------------------------------------------------------------
# FusionResult — Full Audit Trail
# ---------------------------------------------------------------------------

@dataclass
class FusionResult:
    """
    Complete audit trail of the fusion process.

    Stored in telemetry so we can understand exactly why the
    system produced a given MarketStateVector.

    Fields:
        raw_scores      : Scores directly from detectors (before correction)
        adjusted_scores : After mutual exclusivity corrections
        final_probs     : After normalization (these go into MarketStateVector)
        corrections     : Human-readable list of adjustments made
        vector          : The final MarketStateVector output
    """
    raw_scores:      Dict[str, float]
    adjusted_scores: Dict[str, float]
    final_probs:     Dict[str, float]
    corrections:     List[str]
    vector:          MarketStateVector

    def as_log_dict(self) -> Dict:
        """Flat dict for telemetry logging."""
        result = {}
        for k, v in self.raw_scores.items():
            result[f"raw.{k}"]   = round(v, 4)
        for k, v in self.final_probs.items():
            result[f"prob.{k}"]  = round(v, 4)
        result["dominant"]    = self.vector.dominant_state
        result["confidence"]  = round(self.vector.confidence, 4)
        result["corrections"] = len(self.corrections)
        return result


# ---------------------------------------------------------------------------
# MarketStateFusion
# ---------------------------------------------------------------------------

class MarketStateFusion:
    """
    Converts 6 raw detector scores into a normalized MarketStateVector.

    Usage:
        fusion = MarketStateFusion()
        result = fusion.fuse(
            detector_results={
                "trending":    trend_result,
                "ranging":     range_result,
                "expansion":   expansion_result,
                "compression": compression_result,
                "reversal":    reversal_result,
                "news_chaos":  chaos_result,
            },
            symbol="BTCUSDT",
            timeframe="1h",
            bar_index=299,
        )
        vector = result.vector
    """

    # Probability floor — minimum probability for any state.
    # Must satisfy: PROB_FLOOR * 6 < 1.0  (currently 0.12 — feasible)
    PROB_FLOOR: float = 0.02

    # Expected detector keys — validated on every fuse() call
    EXPECTED_KEYS = frozenset([
        "trending", "ranging", "expansion",
        "compression", "reversal", "news_chaos",
    ])

    def fuse(
        self,
        detector_results: Dict[str, DetectorResult],
        symbol:           str = "",
        timeframe:        str = "",
        bar_index:        int = 0,
    ) -> FusionResult:
        """
        Main fusion entry point.

        Args:
            detector_results : One DetectorResult per state key.
                               All six EXPECTED_KEYS must be present.
            symbol           : For output vector metadata.
            timeframe        : For output vector metadata.
            bar_index        : For output vector metadata.

        Returns:
            FusionResult with full audit trail.

        Raises:
            ValueError if detector_results is missing expected keys.
        """
        # Validate inputs
        missing = self.EXPECTED_KEYS - set(detector_results.keys())
        if missing:
            raise ValueError(
                f"MarketStateFusion.fuse() missing detector results: {missing}"
            )

        # Step 1: Extract raw scores
        raw = {k: float(v.score) for k, v in detector_results.items()}
        logger.debug(f"Fusion raw scores: {self._fmt(raw)}")

        # Step 2: Apply mutual exclusivity corrections
        adjusted, corrections = self._apply_corrections(raw)
        logger.debug(f"Fusion adjusted: {self._fmt(adjusted)}")
        for c in corrections:
            logger.debug(f"  correction: {c}")

        # Steps 3+4: Project onto probability simplex with floor constraint.
        # SimplexProjector solves floor + normalization in one closed-form
        # pass — no iterative loops, no floating-point drift.
        # Guarantees: sum=1.0, all >= PROB_FLOOR, deterministic.
        projector = SimplexProjector(floor=self.PROB_FLOOR)
        final     = projector.project(adjusted)

        logger.debug(f"Fusion final probs: {self._fmt(final)}")

        # Step 5: Construct vector
        vector = MarketStateVector(
            trending    = final["trending"],
            ranging     = final["ranging"],
            expansion   = final["expansion"],
            compression = final["compression"],
            reversal    = final["reversal"],
            news_chaos  = final["news_chaos"],
            symbol      = symbol,
            timeframe   = timeframe,
            bar_index   = bar_index,
        )

        logger.info(f"Fusion complete: {vector.summary()}")

        return FusionResult(
            raw_scores      = raw,
            adjusted_scores = adjusted,
            final_probs     = final,
            corrections     = corrections,
            vector          = vector,
        )

    # ------------------------------------------------------------------ #
    # Mutual Exclusivity Corrections                                       #
    # ------------------------------------------------------------------ #

    def _apply_corrections(
        self, raw: Dict[str, float]
    ) -> tuple[Dict[str, float], List[str]]:
        """
        Applies three correction passes in fixed order.
        Each pass is independent — order matters for reproducibility.

        Pass 1: News chaos suppression
        Pass 2: Trend ↔ Range mutual suppression
        Pass 3: Expansion ↔ Compression mutual suppression

        Returns (adjusted_dict, list_of_correction_descriptions)
        """
        adj         = dict(raw)
        corrections = []

        # ── Pass 1: News chaos suppresses all others ──────────────────
        chaos = adj["news_chaos"]
        if chaos > 0.15:
            # Each unit of chaos above 0.15 suppresses others by 70%
            suppression = 1.0 - ((chaos - 0.15) * 0.70)
            suppression = max(0.10, suppression)  # Never suppress below 10%
            for k in adj:
                if k != "news_chaos":
                    before = adj[k]
                    adj[k] = adj[k] * suppression
                    if abs(before - adj[k]) > 0.01:
                        corrections.append(
                            f"chaos_suppression: {k} "
                            f"{before:.3f} → {adj[k]:.3f} "
                            f"(factor={suppression:.3f})"
                        )

        # ── Pass 2: Trend ↔ Range mutual suppression ──────────────────
        trend_dom = adj["trending"] - adj["ranging"]
        if trend_dom > 0.20:
            # Strong trend → suppress ranging
            factor    = 1.0 - (trend_dom * 0.50)
            before    = adj["ranging"]
            adj["ranging"] = adj["ranging"] * factor
            corrections.append(
                f"trend_vs_range: ranging "
                f"{before:.3f} → {adj['ranging']:.3f}"
            )
        elif trend_dom < -0.20:
            # Strong ranging → suppress trending
            factor     = 1.0 - (abs(trend_dom) * 0.50)
            before     = adj["trending"]
            adj["trending"] = adj["trending"] * factor
            corrections.append(
                f"range_vs_trend: trending "
                f"{before:.3f} → {adj['trending']:.3f}"
            )

        # ── Pass 3: Expansion ↔ Compression mutual suppression ────────
        exp_dom = adj["expansion"] - adj["compression"]
        if exp_dom > 0.20:
            factor     = 1.0 - (exp_dom * 0.60)
            before     = adj["compression"]
            adj["compression"] = adj["compression"] * max(0.0, factor)
            corrections.append(
                f"expansion_vs_compression: compression "
                f"{before:.3f} → {adj['compression']:.3f}"
            )
        elif exp_dom < -0.20:
            factor     = 1.0 - (abs(exp_dom) * 0.60)
            before     = adj["expansion"]
            adj["expansion"] = adj["expansion"] * max(0.0, factor)
            corrections.append(
                f"compression_vs_expansion: expansion "
                f"{before:.3f} → {adj['expansion']:.3f}"
            )

        # Clip all values to [0, 1] after corrections
        adj = {k: float(np.clip(v, 0.0, 1.0)) for k, v in adj.items()}
        return adj, corrections

    @staticmethod
    def _fmt(d: Dict[str, float]) -> str:
        """Compact dict display for debug logging."""
        return " | ".join(f"{k[:4]}={v:.3f}" for k, v in d.items())