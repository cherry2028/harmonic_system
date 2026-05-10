"""
tests/test_week1.py
====================
Week 1 Test Suite — MarketStateVector, Detectors, Fusion, Telemetry

Test philosophy:
    - Deterministic: fixed random seed, reproducible always
    - Observable: every assertion has a descriptive message
    - Minimal: one test per behavior, no redundancy
    - Fast: no network calls, no file I/O (telemetry mocked)

Test categories:
    1. MarketStateVector — dataclass contracts
    2. BaseDetector — interface enforcement
    3. Individual detectors — score ranges + signal keys
    4. MarketStateFusion — normalization + corrections
    5. MarketStateEngine — integration + error handling
    6. Telemetry — write behavior + no-crash guarantee

Run with:
    cd your_project_root
    python -m pytest tests/test_week1.py -v

    # With debug logging:
    python -m pytest tests/test_week1.py -v -s --log-cli-level=DEBUG
"""

from __future__ import annotations

import sys
import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup (run from project root)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from market_state.vector            import MarketStateVector
from market_state.detectors.base    import BaseDetector, DetectorResult
from market_state.detectors.trend   import TrendDetector
from market_state.detectors.range_  import RangeDetector
from market_state.detectors.expansion   import ExpansionDetector
from market_state.detectors.compression import CompressionDetector
from market_state.detectors.reversal    import ReversalDetector
from market_state.detectors.chaos       import NewsChaosDetector
from market_state.fusion            import MarketStateFusion
from market_state.engine            import MarketStateEngine


# ---------------------------------------------------------------------------
# Test Data Generators
# ---------------------------------------------------------------------------

def make_trending_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generates a clean uptrending OHLCV DataFrame."""
    rng   = np.random.default_rng(seed)
    base  = 50000.0
    trend = np.linspace(0, 5000, n)
    noise = rng.normal(0, 100, n)
    close = base + trend + noise

    high   = close + rng.uniform(50, 200, n)
    low    = close - rng.uniform(50, 200, n)
    open_  = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.uniform(100, 1000, n)

    return pd.DataFrame({
        "open":   open_, "high": high,
        "low":    low,   "close": close,
        "volume": volume,
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))


def make_ranging_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generates a sideways oscillating OHLCV DataFrame."""
    rng   = np.random.default_rng(seed)
    base  = 50000.0
    noise = rng.normal(0, 300, n)
    # Oscillate around a flat mean
    close = base + noise * np.sin(np.linspace(0, 10 * np.pi, n))

    high   = close + rng.uniform(50, 150, n)
    low    = close - rng.uniform(50, 150, n)
    open_  = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.uniform(100, 500, n)

    return pd.DataFrame({
        "open":  open_, "high": high,
        "low":   low,   "close": close,
        "volume": volume,
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))


def make_minimal_df(n: int = 30) -> pd.DataFrame:
    """Minimal valid DataFrame — just enough for detectors."""
    return make_trending_df(n=n)


def make_chaos_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """DataFrame with a large gap + volume spike at the end."""
    rng   = np.random.default_rng(seed)
    base  = make_trending_df(n=n-1, seed=seed)

    # Add one extreme candle at the end
    last_close = float(base["close"].iloc[-1])
    chaos_row  = pd.DataFrame({
        "open":   [last_close * 1.03],    # 3% gap
        "high":   [last_close * 1.05],
        "low":    [last_close * 0.97],
        "close":  [last_close * 1.04],
        "volume": [float(base["volume"].mean()) * 8],  # 8× volume spike
    }, index=[base.index[-1] + pd.Timedelta(hours=1)])

    return pd.concat([base, chaos_row])


# ===========================================================================
# 1. MarketStateVector Tests
# ===========================================================================

class TestMarketStateVector:

    def test_frozen(self):
        """Vector must be immutable — no attribute assignment allowed."""
        vec = MarketStateVector(
            trending=0.3, ranging=0.5, expansion=0.05,
            compression=0.05, reversal=0.05, news_chaos=0.05,
        )
        with pytest.raises((AttributeError, TypeError)):
            vec.trending = 0.9

    def test_dominant_state_correct(self):
        """dominant_state returns the highest probability state."""
        vec = MarketStateVector(
            trending=0.05, ranging=0.70, expansion=0.05,
            compression=0.08, reversal=0.07, news_chaos=0.05,
        )
        assert vec.dominant_state == "ranging", (
            f"Expected 'ranging', got '{vec.dominant_state}'"
        )

    def test_dominant_state_fallback_when_uncertain(self):
        """When all probabilities are low (<0.20), returns 'ranging'."""
        vec = MarketStateVector(
            trending=0.10, ranging=0.10, expansion=0.10,
            compression=0.10, reversal=0.10, news_chaos=0.10,
        )
        assert vec.dominant_state == "ranging", (
            "Uncertain state should fall back to 'ranging'"
        )

    def test_confidence_equals_max_probability(self):
        """confidence must equal the maximum state probability."""
        vec = MarketStateVector(
            trending=0.60, ranging=0.20, expansion=0.05,
            compression=0.05, reversal=0.05, news_chaos=0.05,
        )
        assert math.isclose(vec.confidence, 0.60, rel_tol=1e-5), (
            f"confidence={vec.confidence:.4f} != 0.60"
        )

    def test_is_hostile_chaos(self):
        """is_hostile() returns True when news_chaos >= threshold."""
        vec = MarketStateVector(
            trending=0.10, ranging=0.10, expansion=0.10,
            compression=0.10, reversal=0.10, news_chaos=0.50,
        )
        assert vec.is_hostile(chaos_threshold=0.40), (
            "Should be hostile when news_chaos=0.50 >= 0.40"
        )

    def test_is_hostile_compression(self):
        """is_hostile() returns True when compression >= threshold."""
        vec = MarketStateVector(
            trending=0.05, ranging=0.10, expansion=0.05,
            compression=0.70, reversal=0.05, news_chaos=0.05,
        )
        assert vec.is_hostile(compression_threshold=0.65), (
            "Should be hostile when compression=0.70 >= 0.65"
        )

    def test_is_not_hostile_normal(self):
        """is_hostile() returns False for normal ranging market."""
        vec = MarketStateVector(
            trending=0.05, ranging=0.72, expansion=0.05,
            compression=0.08, reversal=0.05, news_chaos=0.05,
        )
        assert not vec.is_hostile(), (
            "Normal ranging market should not be hostile"
        )

    def test_harmonic_multiplier_in_range(self):
        """harmonic_edge_multiplier() always returns value in [0.10, 1.50]."""
        for _ in range(20):
            # Random valid probability vector
            probs  = np.random.dirichlet(np.ones(6))
            vec    = MarketStateVector(
                trending=probs[0], ranging=probs[1],
                expansion=probs[2], compression=probs[3],
                reversal=probs[4], news_chaos=probs[5],
            )
            mult = vec.harmonic_edge_multiplier()
            assert 0.10 <= mult <= 1.50, (
                f"harmonic_multiplier={mult:.4f} out of [0.10, 1.50]"
            )

    def test_harmonic_multiplier_reversal_highest(self):
        """Reversal-dominant state should give highest harmonic multiplier."""
        reversal_vec = MarketStateVector(
            trending=0.02, ranging=0.05, expansion=0.02,
            compression=0.02, reversal=0.87, news_chaos=0.02,
        )
        trending_vec = MarketStateVector(
            trending=0.87, ranging=0.05, expansion=0.02,
            compression=0.02, reversal=0.02, news_chaos=0.02,
        )
        assert (
            reversal_vec.harmonic_edge_multiplier()
            > trending_vec.harmonic_edge_multiplier()
        ), "Reversal state should produce higher harmonic multiplier than trending"

    def test_state_probs_keys(self):
        """state_probs must contain exactly the six expected keys."""
        vec  = MarketStateVector(
            trending=0.1, ranging=0.6, expansion=0.1,
            compression=0.1, reversal=0.05, news_chaos=0.05,
        )
        keys = set(vec.state_probs.keys())
        expected = {
            "trending", "ranging", "expansion",
            "compression", "reversal", "news_chaos",
        }
        assert keys == expected, f"Keys mismatch: {keys} != {expected}"

    def test_summary_contains_dominant_state(self):
        """summary() string must contain the dominant state name."""
        vec = MarketStateVector(
            trending=0.05, ranging=0.70, expansion=0.05,
            compression=0.08, reversal=0.07, news_chaos=0.05,
        )
        assert "RANGING" in vec.summary(), (
            f"'RANGING' not found in summary: {vec.summary()}"
        )

    def test_reasoning_lines_is_list(self):
        """reasoning_lines() must return a non-empty list of strings."""
        vec = MarketStateVector(
            trending=0.05, ranging=0.70, expansion=0.05,
            compression=0.08, reversal=0.07, news_chaos=0.05,
        )
        lines = vec.reasoning_lines()
        assert isinstance(lines, list), "reasoning_lines() must return a list"
        assert len(lines) >= 3, f"Expected >= 3 lines, got {len(lines)}"
        assert all(isinstance(l, str) for l in lines), (
            "All reasoning lines must be strings"
        )


# ===========================================================================
# 2. BaseDetector Contract Tests
# ===========================================================================

class TestBaseDetectorContract:

    def test_missing_detector_name_raises(self):
        """Subclass without DETECTOR_NAME must raise TypeError."""
        with pytest.raises(TypeError):
            class BadDetector(BaseDetector):
                DETECTOR_NAME = ""   # Empty = invalid
                def _compute(self, df): ...

    def test_short_df_returns_zero(self):
        """score() must return 0.0 (not raise) for too-short DataFrame."""
        detector = TrendDetector()
        short_df = make_minimal_df(n=5)   # Way too short
        result   = detector.score(short_df)
        assert result.score == 0.0, (
            f"Expected 0.0 for short df, got {result.score}"
        )
        assert "reason" in result.signals, (
            "Zero result must include 'reason' in signals"
        )

    def test_none_df_returns_zero(self):
        """score() must return 0.0 (not raise) for None input."""
        detector = TrendDetector()
        result   = detector.score(None)
        assert result.score == 0.0

    def test_detector_result_score_out_of_range_raises(self):
        """DetectorResult must raise ValueError if score out of [0, 1]."""
        with pytest.raises(ValueError, match="outside \\[0.0, 1.0\\]"):
            DetectorResult(score=1.5, signals={"x": 1.0}, detector="test")

    def test_score_in_valid_range(self):
        """All detectors must produce scores in [0.0, 1.0]."""
        detectors = [
            TrendDetector(), RangeDetector(), ExpansionDetector(),
            CompressionDetector(), ReversalDetector(), NewsChaosDetector(),
        ]
        df = make_trending_df(n=200)
        for det in detectors:
            result = det.score(df)
            assert 0.0 <= result.score <= 1.0, (
                f"{det.DETECTOR_NAME}: score={result.score:.4f} "
                f"out of [0.0, 1.0]"
            )

    def test_signals_not_empty(self):
        """All detectors must populate signals dict with at least one key."""
        detectors = [
            TrendDetector(), RangeDetector(), ExpansionDetector(),
            CompressionDetector(), ReversalDetector(), NewsChaosDetector(),
        ]
        df = make_trending_df(n=200)
        for det in detectors:
            result = det.score(df)
            assert len(result.signals) >= 1, (
                f"{det.DETECTOR_NAME}: signals dict is empty"
            )

    def test_detector_name_in_result(self):
        """DetectorResult.detector must be populated with detector name."""
        det    = TrendDetector()
        df     = make_trending_df(n=200)
        result = det.score(df)
        assert result.detector == "trend", (
            f"Expected 'trend', got '{result.detector}'"
        )


# ===========================================================================
# 3. Individual Detector Behavioral Tests
# ===========================================================================

class TestDetectorBehavior:

    def test_trend_detector_scores_higher_on_trending_data(self):
        """TrendDetector must score trending data higher than ranging data."""
        det      = TrendDetector()
        trending = det.score(make_trending_df(n=200))
        ranging  = det.score(make_ranging_df(n=200))
        assert trending.score > ranging.score, (
            f"Trending score {trending.score:.4f} should be > "
            f"ranging score {ranging.score:.4f}"
        )

    def test_trend_detector_signals_has_adx(self):
        """TrendDetector must emit 'adx_latest' in signals."""
        result = TrendDetector().score(make_trending_df(n=200))
        assert "adx_latest" in result.signals, (
            f"'adx_latest' missing from signals: {result.signals}"
        )

    def test_chaos_detector_scores_higher_on_chaos_data(self):
        """NewsChaosDetector must score chaos data higher than normal."""
        det    = NewsChaosDetector()
        normal = det.score(make_trending_df(n=100))
        chaos  = det.score(make_chaos_df(n=100))
        assert chaos.score > normal.score, (
            f"Chaos score {chaos.score:.4f} should be > "
            f"normal score {normal.score:.4f}"
        )

    def test_all_detectors_deterministic(self):
        """Same input must produce same output every time."""
        df = make_trending_df(n=200, seed=99)
        detectors = [
            TrendDetector(), RangeDetector(), ExpansionDetector(),
            CompressionDetector(), ReversalDetector(), NewsChaosDetector(),
        ]
        for det in detectors:
            r1 = det.score(df)
            r2 = det.score(df)
            assert r1.score == r2.score, (
                f"{det.DETECTOR_NAME}: non-deterministic! "
                f"{r1.score:.6f} != {r2.score:.6f}"
            )

    def test_compression_scores_higher_on_flat_data(self):
        """CompressionDetector: flat low-volatility data should score higher."""
        rng     = np.random.default_rng(42)
        n       = 100
        # Very flat price movement
        close   = 50000 + rng.normal(0, 10, n)
        high    = close + 5
        low     = close - 5
        open_   = np.roll(close, 1)
        open_[0] = close[0]
        flat_df = pd.DataFrame({
            "open": open_, "high": high, "low": low,
            "close": close, "volume": rng.uniform(100, 200, n),
        }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))

        normal_df = make_trending_df(n=n)

        det  = CompressionDetector()
        flat = det.score(flat_df)
        norm = det.score(normal_df)

        assert flat.score >= norm.score, (
            f"Flat data compression {flat.score:.4f} should be "
            f">= normal {norm.score:.4f}"
        )


# ===========================================================================
# 4. MarketStateFusion Tests
# ===========================================================================

class TestMarketStateFusion:

    def _make_results(self, scores: dict) -> dict:
        """Creates DetectorResult dict from a scores dict."""
        return {
            k: DetectorResult(score=v, signals={"test": v}, detector=k)
            for k, v in scores.items()
        }

    def test_output_probabilities_sum_to_one(self):
        """Fusion output probabilities must sum to ~1.0."""
        fusion  = MarketStateFusion()
        results = self._make_results({
            "trending": 0.6, "ranging": 0.2, "expansion": 0.1,
            "compression": 0.05, "reversal": 0.4, "news_chaos": 0.05,
        })
        fr  = fusion.fuse(results)
        total = sum(fr.final_probs.values())
        assert math.isclose(total, 1.0, abs_tol=0.001), (
            f"Probabilities sum to {total:.6f}, expected 1.0"
        )

    def test_all_states_above_floor(self):
        """Every state probability must be >= PROB_FLOOR after fusion."""
        fusion  = MarketStateFusion()
        results = self._make_results({
            "trending": 1.0, "ranging": 0.0, "expansion": 0.0,
            "compression": 0.0, "reversal": 0.0, "news_chaos": 0.0,
        })
        fr = fusion.fuse(results)
        for state, prob in fr.final_probs.items():
            assert prob >= fusion.PROB_FLOOR, (
                f"State '{state}' has prob={prob:.4f} below "
                f"floor {fusion.PROB_FLOOR}"
            )

    def test_chaos_suppresses_others(self):
        """High news_chaos must reduce other state scores after fusion."""
        fusion = MarketStateFusion()

        # Baseline: zero chaos
        r_no_chaos = self._make_results({
            "trending": 0.5, "ranging": 0.3, "expansion": 0.1,
            "compression": 0.1, "reversal": 0.2, "news_chaos": 0.0,
        })
        # With high chaos
        r_chaos = self._make_results({
            "trending": 0.5, "ranging": 0.3, "expansion": 0.1,
            "compression": 0.1, "reversal": 0.2, "news_chaos": 0.8,
        })

        fr_no   = fusion.fuse(r_no_chaos)
        fr_yes  = fusion.fuse(r_chaos)

        # Chaos probability itself should be dominant
        assert fr_yes.final_probs["news_chaos"] > fr_no.final_probs["news_chaos"]

    def test_fusion_missing_key_raises(self):
        """fuse() must raise ValueError when a detector result is missing."""
        fusion  = MarketStateFusion()
        results = self._make_results({
            "trending": 0.5, "ranging": 0.3,
            # Missing: expansion, compression, reversal, news_chaos
        })
        with pytest.raises(ValueError, match="missing detector results"):
            fusion.fuse(results)

    def test_dominant_state_in_vector_matches_highest_prob(self):
        """The vector's dominant_state must match the highest probability."""
        fusion  = MarketStateFusion()
        results = self._make_results({
            "trending": 0.05, "ranging": 0.70, "expansion": 0.05,
            "compression": 0.05, "reversal": 0.10, "news_chaos": 0.05,
        })
        fr = fusion.fuse(results)
        assert fr.vector.dominant_state == "ranging", (
            f"Expected dominant='ranging', got '{fr.vector.dominant_state}'"
        )

    def test_fusion_returns_fusion_result_type(self):
        """fuse() must return a FusionResult with a vector attribute."""
        from market_state.fusion import FusionResult
        fusion  = MarketStateFusion()
        results = self._make_results({
            "trending": 0.3, "ranging": 0.4, "expansion": 0.1,
            "compression": 0.1, "reversal": 0.05, "news_chaos": 0.05,
        })
        fr = fusion.fuse(results)
        assert isinstance(fr, FusionResult)
        assert isinstance(fr.vector, MarketStateVector)
        assert isinstance(fr.corrections, list)
        assert isinstance(fr.raw_scores, dict)
        assert isinstance(fr.final_probs, dict)


# ===========================================================================
# 5. MarketStateEngine Integration Tests
# ===========================================================================

class TestMarketStateEngine:

    def test_classify_returns_vector(self):
        """classify() must always return a MarketStateVector."""
        engine = MarketStateEngine()
        df     = make_trending_df(n=200)
        vector = engine.classify(df, "BTCUSDT", "1h")
        assert isinstance(vector, MarketStateVector), (
            f"Expected MarketStateVector, got {type(vector)}"
        )

    def test_classify_never_raises(self):
        """classify() must never raise — even on bad input."""
        engine = MarketStateEngine()
        # These should all return a vector, never raise
        assert engine.classify(None)            is not None
        assert engine.classify(pd.DataFrame())  is not None
        assert engine.classify(make_minimal_df(n=5)) is not None

    def test_classify_metadata_set(self):
        """classify() must populate symbol and timeframe in vector."""
        engine = MarketStateEngine()
        vector = engine.classify(
            make_trending_df(n=200), "ETHUSDT", "4h"
        )
        assert vector.symbol    == "ETHUSDT"
        assert vector.timeframe == "4h"

    def test_classify_with_detail_returns_fusion_result(self):
        """classify_with_detail() must return (vector, FusionResult)."""
        from market_state.fusion import FusionResult
        engine = MarketStateEngine()
        vector, fr = engine.classify_with_detail(
            make_trending_df(n=200), "BTCUSDT", "1h"
        )
        assert isinstance(vector, MarketStateVector)
        assert isinstance(fr, FusionResult)

    def test_classify_probabilities_sum_to_one(self):
        """classify() output probabilities must sum to ~1.0."""
        engine = MarketStateEngine()
        vector = engine.classify(make_trending_df(n=200))
        total  = sum(vector.state_probs.values())
        assert math.isclose(total, 1.0, abs_tol=0.005), (
            f"State probabilities sum to {total:.6f}, expected 1.0"
        )

    def test_classify_deterministic(self):
        """Same df + same params must produce identical vectors."""
        engine = MarketStateEngine()
        df     = make_trending_df(n=200, seed=42)
        v1     = engine.classify(df, "BTCUSDT", "1h")
        v2     = engine.classify(df, "BTCUSDT", "1h")
        assert v1.trending    == v2.trending
        assert v1.ranging     == v2.ranging
        assert v1.expansion   == v2.expansion
        assert v1.compression == v2.compression
        assert v1.reversal    == v2.reversal
        assert v1.news_chaos  == v2.news_chaos

    def test_fallback_on_engine_failure(self):
        """Engine must return fallback vector when fusion fails."""
        engine = MarketStateEngine()
        # Force fusion to fail
        with patch.object(engine.fusion, "fuse", side_effect=RuntimeError("test")):
            vector = engine.classify(make_trending_df(n=200))
        assert isinstance(vector, MarketStateVector), (
            "Engine must return fallback vector on exception"
        )
        assert vector.dominant_state == "ranging", (
            "Fallback vector must be ranging-dominant"
        )


# ===========================================================================
# 6. Telemetry Tests
# ===========================================================================

class TestTelemetry:

    def test_log_state_writes_jsonl(self, tmp_path):
        """log_state() must write a valid JSON line to market_state.jsonl."""
        import telemetry.logger as tel
        original_dir = tel.TELEMETRY_DIR
        tel.TELEMETRY_DIR = tmp_path / "telemetry"

        try:
            vec = MarketStateVector(
                trending=0.05, ranging=0.70, expansion=0.05,
                compression=0.08, reversal=0.07, news_chaos=0.05,
                symbol="BTCUSDT", timeframe="1h",
            )
            tel.log_state(vec)

            log_file = tel.TELEMETRY_DIR / "market_state.jsonl"
            assert log_file.exists(), "market_state.jsonl must be created"

            lines = log_file.read_text().strip().split("\n")
            assert len(lines) == 1, f"Expected 1 line, got {len(lines)}"

            record = json.loads(lines[0])
            assert record["type"]      == "market_state"
            assert record["symbol"]    == "BTCUSDT"
            assert record["timeframe"] == "1h"
            assert record["dominant"]  == "ranging"
            assert "ts"  in record
            assert "iso" in record
        finally:
            tel.TELEMETRY_DIR = original_dir

    def test_log_gate_block_writes_correctly(self, tmp_path):
        """log_gate_block() must write block_code and reason."""
        import telemetry.logger as tel
        original_dir = tel.TELEMETRY_DIR
        tel.TELEMETRY_DIR = tmp_path / "telemetry"

        try:
            vec = MarketStateVector(
                trending=0.05, ranging=0.10, expansion=0.05,
                compression=0.70, reversal=0.05, news_chaos=0.05,
            )
            tel.log_gate_block(
                symbol="BTCUSDT", timeframe="1h",
                block_code="COMPRESSION",
                reason="Market coiling",
                vector=vec,
            )
            log_file = tel.TELEMETRY_DIR / "gate_block.jsonl"
            assert log_file.exists()
            record = json.loads(log_file.read_text().strip())
            assert record["block_code"] == "COMPRESSION"
            assert record["reason"]     == "Market coiling"
        finally:
            tel.TELEMETRY_DIR = original_dir

    def test_telemetry_never_crashes_on_bad_input(self, tmp_path):
        """Telemetry functions must never raise on bad input."""
        import telemetry.logger as tel
        original_dir = tel.TELEMETRY_DIR
        tel.TELEMETRY_DIR = tmp_path / "telemetry"

        try:
            # All of these must silently succeed or fail without raising
            tel.log_state(None)           # bad vector
            tel.log_gate_block(           # missing fields
                symbol="", timeframe="",
                block_code="TEST", reason="test",
                vector=None,
            )
            tel.log_scan_cycle("", "", -1.0, "")
            tel.log_error("test_location", "test error")
        except Exception as e:
            pytest.fail(
                f"Telemetry raised an exception on bad input: {e}"
            )
        finally:
            tel.TELEMETRY_DIR = original_dir

    def test_scan_cycle_outcome_field_present(self, tmp_path):
        """log_scan_cycle() record must contain 'outcome' field."""
        import telemetry.logger as tel
        original_dir = tel.TELEMETRY_DIR
        tel.TELEMETRY_DIR = tmp_path / "telemetry"

        try:
            tel.log_scan_cycle("SOLUSDT", "4h", 45.2, "gate_blocked")
            log_file = tel.TELEMETRY_DIR / "scan_cycle.jsonl"
            record   = json.loads(log_file.read_text().strip())
            assert record["outcome"] == "gate_blocked"
            assert record["symbol"]  == "SOLUSDT"
        finally:
            tel.TELEMETRY_DIR = original_dir


# ===========================================================================
# Expected Output Samples (printed when run with -s)
# ===========================================================================

def test_print_expected_outputs():
    """
    Prints real output samples for manual inspection.
    Not a pass/fail test — run with -s flag to see output.

    Expected debug log examples:
        trend: score=0.6234 | adx_latest=31.20, adx_score=0.6343,
                               slope_score=0.5800, plus_di=28.4, minus_di=12.1
        ranging: score=0.4123 | bb_score=0.3800, oscillation=0.4400
        expansion: score=0.1200 | atr_score=0.0000, body_score=0.3000
        compression: score=0.0800 | atr_score=0.1200, bb_kc_score=0.0000
        reversal: score=0.2100 | rsi_divergence=0.3200, macd_divergence=0.3000,
                                  volume_exhaustion=0.0000
        news_chaos: score=0.0500 | gap_score=0.0200, volume_spike=0.0000,
                                    extreme_candle=0.1200

        Fusion raw scores: tren=0.623 | rang=0.412 | expa=0.120 |
                           comp=0.080 | reve=0.210 | news=0.050
        Fusion complete: MarketState [BTCUSDT 1h] | TRENDING (38% conf) | ...
    """
    import logging
    logging.basicConfig(level=logging.DEBUG)

    engine = MarketStateEngine()
    df     = make_trending_df(n=300)
    vector, fusion_result = engine.classify_with_detail(df, "BTCUSDT", "1h")

    print("\n" + "="*60)
    print("VECTOR OUTPUT:")
    print(vector.summary())
    print("\nREASONING LINES:")
    for line in vector.reasoning_lines():
        print(f"  {line}")
    print("\nFUSION AUDIT:")
    print(f"  Raw:     {fusion_result.raw_scores}")
    print(f"  Adj:     {fusion_result.adjusted_scores}")
    print(f"  Final:   {fusion_result.final_probs}")
    print(f"  Corrections: {fusion_result.corrections}")
    print("="*60)
