"""
market_state/engine.py
=======================
MarketStateEngine — Layer 1 Orchestrator

Single public method: classify(df, symbol, timeframe) → MarketStateVector

This is the ONLY file downstream code needs to import from this package.
Everything else is internal implementation.

Internal flow:
    1. Validate input
    2. Run all 5 detectors in sequence (parallel in Phase 2 if needed)
    3. Pass all 6 DetectorResults to MarketStateFusion
    4. Log FusionResult to telemetry
    5. Return MarketStateVector

Error handling:
    Any exception in any detector → that detector returns score=0.0
    (handled inside BaseDetector.score())
    Any exception in fusion → returns safe fallback vector
    Engine never raises to the caller. Ever.

Configuration:
    All detector parameters are set in __init__.
    In production, pass a MarketStateConfig object.
    During testing, use defaults.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd

from market_state.detectors.trend       import TrendDetector
from market_state.detectors.range_      import RangeDetector
from market_state.detectors.expansion   import ExpansionDetector
from market_state.detectors.compression import CompressionDetector
from market_state.detectors.reversal    import ReversalDetector
from market_state.detectors.chaos       import NewsChaosDetector
from market_state.fusion                import MarketStateFusion, FusionResult
from market_state.vector                import MarketStateVector

logger = logging.getLogger("market_state.engine")


class MarketStateEngine:
    """
    Full Layer 1 Market Perception Engine.

    Usage:
        engine = MarketStateEngine()
        vector = engine.classify(df, symbol="BTCUSDT", timeframe="1h")

        # Downstream gate check
        if vector.is_hostile():
            return

        # Scoring multiplier
        mult = vector.harmonic_edge_multiplier()

    Production instantiation:
        Create once per process. Reuse across all scan cycles.
        All detectors are stateless — thread-safe for concurrent classify() calls.
    """

    def __init__(
        self,
        adx_period:    int = 14,
        slope_period:  int = 20,
        bb_period:     int = 20,
        atr_period:    int = 14,
        rsi_period:    int = 14,
        lookback_bars: int = 50,
    ):
        self.detectors = {
            "trending":    TrendDetector(
                               adx_period=adx_period,
                               slope_period=slope_period,
                           ),
            "ranging":     RangeDetector(bb_period=bb_period),
            "expansion":   ExpansionDetector(
                               atr_period=atr_period,
                               lookback=lookback_bars,
                           ),
            "compression": CompressionDetector(
                               atr_period=atr_period,
                               lookback=lookback_bars,
                           ),
            "reversal":    ReversalDetector(
                               rsi_period=rsi_period,
                               lookback=lookback_bars,
                           ),
            "news_chaos":  NewsChaosDetector(),
        }
        self.fusion = MarketStateFusion()
        logger.info("MarketStateEngine initialized")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def classify(
        self,
        df:        pd.DataFrame,
        symbol:    str = "",
        timeframe: str = "",
    ) -> MarketStateVector:
        """
        Classifies current market state from OHLCV data.

        Args:
            df        : OHLCV DataFrame.
                        Required columns: open, high, low, close, volume.
                        Recommended: 200–500 bars. Minimum: 35 bars.
            symbol    : Trading pair label (logging/output only).
            timeframe : Candle timeframe label (logging/output only).

        Returns:
            MarketStateVector — always. Never raises.
        """
        t0    = time.monotonic()
        label = f"{symbol} {timeframe}".strip() or "unknown"

        try:
            vector, fusion_result = self._run(df, symbol, timeframe)
        except Exception as e:
            logger.error(
                f"MarketStateEngine.classify() failed | {label} | {e}",
                exc_info=True,
            )
            vector = self._fallback(symbol, timeframe)
            fusion_result = None

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            f"classify() complete | {label} | {elapsed_ms:.1f}ms"
        )
        return vector

    def classify_with_detail(
        self,
        df:        pd.DataFrame,
        symbol:    str = "",
        timeframe: str = "",
    ) -> tuple[MarketStateVector, Optional[FusionResult]]:
        """
        Same as classify() but also returns the FusionResult.
        Use for backtesting, diagnostics, and telemetry.
        """
        try:
            return self._run(df, symbol, timeframe)
        except Exception as e:
            logger.error(
                f"classify_with_detail() failed | {symbol} {timeframe} | {e}",
                exc_info=True,
            )
            return self._fallback(symbol, timeframe), None

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _run(
        self,
        df:        pd.DataFrame,
        symbol:    str,
        timeframe: str,
    ) -> tuple[MarketStateVector, FusionResult]:
        """Runs all detectors and fusion. Raises on unexpected errors."""
        label = f"{symbol} {timeframe}".strip()
        logger.debug(f"Running detectors | {label} | {len(df)} bars")

        # Run all detectors — each handles its own errors internally
        results = {
            name: detector.score(df)
            for name, detector in self.detectors.items()
        }

        # Log individual detector scores at DEBUG level
        for name, result in results.items():
            logger.debug(
                f"  {name}: score={result.score:.4f} "
                f"signals={result.signals}"
            )

        # Fuse into probability vector
        fusion_result = self.fusion.fuse(
            detector_results = results,
            symbol           = symbol,
            timeframe        = timeframe,
            bar_index        = max(0, len(df) - 1),
        )

        return fusion_result.vector, fusion_result

    @staticmethod
    def _fallback(symbol: str, timeframe: str) -> MarketStateVector:
        """
        Safe fallback vector when engine fails completely.
        Ranging-dominant: most conservative possible classification.
        """
        logger.warning(
            f"Using fallback MarketStateVector | {symbol} {timeframe}"
        )
        return MarketStateVector(
            trending    = 0.05,
            ranging     = 0.70,
            expansion   = 0.05,
            compression = 0.08,
            reversal    = 0.07,
            news_chaos  = 0.05,
            symbol      = symbol,
            timeframe   = timeframe,
        )
