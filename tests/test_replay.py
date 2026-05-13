"""tests/test_replay.py — Phase 1 replay infrastructure contract tests.

Truly necessary: lookahead prevention, defensive copies, runner type safety.
"""
from __future__ import annotations
import json
import os
from unittest.mock import MagicMock

import pandas as pd
import pytest

from replay.bar_feeder import BarFeeder, ReplayDataFetcher
from replay.replay_runner import ReplayRecord, ReplayRunner
from replay.result_store import ResultStore
from pipeline import ScanPipeline, ScanResult
from market_state.vector import MarketStateVector
from signals.signal import TieredSignal
from scoring.score_result import ScoredSignal


# ═════════════════════════════════════════════════════════════════════════════
# BarFeeder
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """10-bar DataFrame with DatetimeIndex."""
    idx = pd.date_range("2024-01-01", periods=10, freq="1h")
    return pd.DataFrame({
        "open": range(10),
        "high": range(1, 11),
        "low": range(-1, 9),
        "close": range(2, 12),
        "volume": [100] * 10,
    }, index=idx)


class TestBarFeeder:
    """Verify forward-only, no-lookahead, defensive-copy guarantees."""

    def test_yields_correct_number_of_windows(self, sample_df):
        feeder = BarFeeder(sample_df, window_size=3)
        windows = list(feeder)
        assert len(windows) == 8  # 10 - 3 + 1
        assert len(feeder) == 8

    def test_windows_are_defensive_copies(self, sample_df):
        feeder = BarFeeder(sample_df, window_size=3)
        _, first = next(iter(feeder))
        first.iloc[0, 0] = 99999
        _, second = next(iter(feeder))
        assert second.iloc[0, 0] != 99999

    def test_no_lookahead_leakage(self, sample_df):
        """Window i must not contain the NEW bar introduced in window i+1.

        Rolling windows overlap by (window_size - 1) bars.
        The only new bar in window i+1 is its last bar, which must not
        appear in window i.
        """
        feeder = BarFeeder(sample_df, window_size=3)
        all_timestamps = []
        for ts, window in feeder:
            all_timestamps.append((ts, list(window.index)))
        for i in range(len(all_timestamps) - 1):
            _, current_idx = all_timestamps[i]
            _, next_idx = all_timestamps[i + 1]
            # The new bar in next_idx is its last element
            future_bar = next_idx[-1]
            # Current window must NOT contain that future bar
            assert future_bar not in current_idx
            # Current window's last bar must be next window's second-to-last
            assert current_idx[-1] == next_idx[-2]

    def test_temporal_order_preserved(self, sample_df):
        feeder = BarFeeder(sample_df, window_size=3)
        timestamps = [ts for ts, _ in feeder]
        assert timestamps == sorted(timestamps)
        assert len(timestamps) == len(set(timestamps))

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            BarFeeder(pd.DataFrame(), window_size=3)

    def test_window_size_too_small_raises(self, sample_df):
        with pytest.raises(ValueError, match="window_size"):
            BarFeeder(sample_df, window_size=1)

    def test_unsorted_input_gets_sorted(self, sample_df):
        shuffled = sample_df.sample(frac=1)
        feeder = BarFeeder(shuffled, window_size=3)
        timestamps = [ts for ts, _ in feeder]
        assert timestamps == sorted(timestamps)


class TestReplayDataFetcher:
    """Verify fetcher API contract."""

    def test_fetch_returns_set_window(self):
        fetcher = ReplayDataFetcher()
        df = pd.DataFrame({"close": [1, 2, 3]})
        fetcher.set_window(df)
        result = fetcher.fetch("BTCUSDT", "1h", bars=300)
        assert result is not None
        assert len(result) == 3

    def test_fetch_ignores_symbol_timeframe_bars(self):
        fetcher = ReplayDataFetcher()
        df = pd.DataFrame({"close": [1]})
        fetcher.set_window(df)
        r1 = fetcher.fetch("X", "1m", bars=1)
        r2 = fetcher.fetch("Y", "1d", bars=999)
        assert r1 is r2

    def test_fetch_returns_none_when_no_window_set(self):
        fetcher = ReplayDataFetcher()
        assert fetcher.fetch("A", "1h") is None

    def test_set_window_makes_defensive_copy(self):
        fetcher = ReplayDataFetcher()
        df = pd.DataFrame({"close": [1, 2]})
        fetcher.set_window(df)
        df.iloc[0, 0] = 999
        result = fetcher.fetch("A", "1h")
        assert result.iloc[0, 0] != 999


# ═════════════════════════════════════════════════════════════════════════════
# ReplayRunner
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_pipeline_with_replay_fetcher():
    """Return a ScanPipeline whose fetcher is a ReplayDataFetcher."""
    fetcher = ReplayDataFetcher()
    comps = {
        "data_fetcher": fetcher,
        "market_state_engine": MagicMock(),
        "swing_detector": MagicMock(),
        "harmonic_detector": MagicMock(),
        "pattern_scorer": MagicMock(),
        "hostile_gate": MagicMock(),
        "signal_tier": MagicMock(),
        "daily_counter": MagicMock(),
        "telemetry": MagicMock(),
        "signal_presentation": MagicMock(),
        "telegram_formatter": MagicMock(),
    }
    pipeline = ScanPipeline(**comps)
    return pipeline


class TestReplayRunnerConstruction:
    """Verify runner construction and type safety."""

    def test_accepts_pipeline_with_replay_fetcher(self, sample_df, mock_pipeline_with_replay_fetcher):
        feeder = BarFeeder(sample_df, window_size=3)
        runner = ReplayRunner(mock_pipeline_with_replay_fetcher, feeder)
        assert runner is not None

    def test_rejects_pipeline_without_replay_fetcher(self, sample_df):
        bad_comps = {
            "data_fetcher": MagicMock(),  # not ReplayDataFetcher
            "market_state_engine": MagicMock(),
            "swing_detector": MagicMock(),
            "harmonic_detector": MagicMock(),
            "pattern_scorer": MagicMock(),
            "hostile_gate": MagicMock(),
            "signal_tier": MagicMock(),
            "daily_counter": MagicMock(),
            "telemetry": MagicMock(),
            "signal_presentation": MagicMock(),
            "telegram_formatter": MagicMock(),
        }
        bad_pipeline = ScanPipeline(**bad_comps)
        feeder = BarFeeder(sample_df, window_size=3)
        with pytest.raises(TypeError, match="ReplayDataFetcher"):
            ReplayRunner(bad_pipeline, feeder)

    def test_preserves_symbol_and_timeframe(self, sample_df, mock_pipeline_with_replay_fetcher):
        feeder = BarFeeder(sample_df, window_size=3)
        runner = ReplayRunner(
            mock_pipeline_with_replay_fetcher, feeder,
            symbol="BTCUSDT", timeframe="1h",
        )
        assert runner._symbol == "BTCUSDT"
        assert runner._timeframe == "1h"


class TestReplayRunnerDryRun:
    """Verify runner always invokes pipeline with dry_run=True."""

    def test_scan_one_called_with_dry_run_true(self, sample_df, mock_pipeline_with_replay_fetcher):
        feeder = BarFeeder(sample_df, window_size=3)
        runner = ReplayRunner(mock_pipeline_with_replay_fetcher, feeder)
        # Mock scan_one to return empty result so iteration completes
        mock_pipeline_with_replay_fetcher.scan_one = MagicMock(return_value=ScanResult(
            symbol="REPLAY", timeframe="REPLAY", outcome="below_threshold", duration_ms=0.0
        ))
        list(runner.run())
        for call in mock_pipeline_with_replay_fetcher.scan_one.call_args_list:
            assert call.kwargs.get("dry_run") is True


class TestReplayRunnerSignalCollection:
    """Verify records are yielded only when signals are produced."""

    def test_no_records_when_no_signals(self, sample_df, mock_pipeline_with_replay_fetcher):
        feeder = BarFeeder(sample_df, window_size=3)
        mock_pipeline_with_replay_fetcher.scan_one = MagicMock(return_value=ScanResult(
            symbol="REPLAY", timeframe="REPLAY", outcome="below_threshold", duration_ms=0.0
        ))
        runner = ReplayRunner(mock_pipeline_with_replay_fetcher, feeder)
        records = list(runner.run())
        assert records == []

    def test_records_yielded_when_signals_present(self, sample_df, mock_pipeline_with_replay_fetcher):
        feeder = BarFeeder(sample_df, window_size=3)
        vector = MarketStateVector(
            trending=0.1, ranging=0.6, expansion=0.1,
            compression=0.1, reversal=0.05, news_chaos=0.05,
        )
        match = type("obj", (object,), {
            "pattern_name": "Gartley",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "direction": "bullish",
            "prz": {...},
        })()
        scored = ScoredSignal(
            pattern_match=match,
            base_score=0.8,
            state_discount=1.0,
            confidence_weight=0.9,
            edge_score=0.72,
            dominant_state="reversal",
            state_vector=vector,
            reasoning=[
                "Pattern detected: Gartley",
                "Market state: reversal",
                "Final edge score: 0.72",
            ],
        )
        tiered = TieredSignal(
            tier="A",
            scored=scored,
            max_per_day=3,
            risk_pct=1.0,

            edge_score=scored.edge_score,
            dominant_state=scored.dominant_state,
            reasoning=scored.reasoning,

            entry=50000.0,
            stop=49500.0,
            target1=51000.0,
            target2=52000.0,
            target3=53000.0,
        )
        mock_pipeline_with_replay_fetcher.scan_one = MagicMock(return_value=ScanResult(
            symbol="REPLAY", timeframe="REPLAY", outcome="signal_published", duration_ms=1.0,
            tiered_signals=[tiered],
        ))
        runner = ReplayRunner(mock_pipeline_with_replay_fetcher, feeder)
        records = list(runner.run())
        assert len(records) == 8  # one per bar, one signal each
        for rec in records:
            assert isinstance(rec, ReplayRecord)
            assert rec.produced_signal
            assert rec.tiered_signal.tier == "A"
            assert rec.scan_result.outcome == "signal_published"

    def test_run_all_returns_list(self, sample_df, mock_pipeline_with_replay_fetcher):
        feeder = BarFeeder(sample_df, window_size=3)
        mock_pipeline_with_replay_fetcher.scan_one = MagicMock(return_value=ScanResult(
            symbol="REPLAY", timeframe="REPLAY", outcome="below_threshold", duration_ms=0.0
        ))
        runner = ReplayRunner(mock_pipeline_with_replay_fetcher, feeder)
        records = runner.run_all()
        assert records == []


# ═════════════════════════════════════════════════════════════════════════════
# ResultStore
# ═════════════════════════════════════════════════════════════════════════════

class TestResultStore:
    """Verify NDJSON write contract."""

    def test_append_creates_file(self, tmp_path):
        store = ResultStore(str(tmp_path / "replay.ndjson"))
        vector = MarketStateVector(
            trending=0.1, ranging=0.6, expansion=0.1,
            compression=0.1, reversal=0.05, news_chaos=0.05,
        )
        result = ScanResult(symbol="X", timeframe="1h", outcome="test", duration_ms=0.0, vector=vector)
        record = ReplayRecord(bar_timestamp="2024-01-01T00:00:00", scan_result=result)
        store.append(record)
        assert os.path.exists(store.filepath)
        with open(store.filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["bar_timestamp"] == "2024-01-01T00:00:00"
        assert parsed["scan_result"]["outcome"] == "test"
        assert parsed["tiered_signal"] is None

    def test_extend_writes_multiple_lines(self, tmp_path):
        store = ResultStore(str(tmp_path / "replay.ndjson"))
        result = ScanResult(symbol="X", timeframe="1h", outcome="test", duration_ms=0.0)
        records = [
            ReplayRecord(bar_timestamp=f"2024-01-0{i}T00:00:00", scan_result=result)
            for i in range(1, 4)
        ]
        store.extend(records)
        with open(store.filepath, "r", encoding="utf-8") as f:
            lines = [l for l in f.readlines() if l.strip()]
        assert len(lines) == 3

    def test_clear_removes_file(self, tmp_path):
        store = ResultStore(str(tmp_path / "replay.ndjson"))
        result = ScanResult(symbol="X", timeframe="1h", outcome="test", duration_ms=0.0)
        store.append(ReplayRecord(bar_timestamp="2024-01-01T00:00:00", scan_result=result))
        assert os.path.exists(store.filepath)
        store.clear()
        assert not os.path.exists(store.filepath)

    def test_tiered_signal_serialization(self, tmp_path):
        store = ResultStore(str(tmp_path / "replay.ndjson"))
        vector = MarketStateVector(
            trending=0.1, ranging=0.6, expansion=0.1,
            compression=0.1, reversal=0.05, news_chaos=0.05,
        )
        match = type("obj", (object,), {
            "pattern_name": "Bat",
            "symbol": "ETHUSDT",
            "timeframe": "4h",
            "direction": "bullish",
            "prz": {...},
        })()
        scored = ScoredSignal(
            pattern_match=match,
            base_score=0.7,
            state_discount=0.8,
            confidence_weight=0.85,
            edge_score=0.476,
            dominant_state="ranging",
            state_vector=vector,
            reasoning=[
                "Pattern detected: Bat",
                "Market state: ranging",
                "Final edge score: 0.476",
            ],
        )
        tiered = TieredSignal(
            tier="B",
            scored=scored,
            max_per_day=5,
            risk_pct=0.5,

            edge_score=scored.edge_score,
            dominant_state=scored.dominant_state,
            reasoning=scored.reasoning,

            entry=3000.0,
            stop=2950.0,
            target1=3100.0,
            target2=3200.0,
            target3=3300.0,
        )
        result = ScanResult(symbol="ETHUSDT", timeframe="4h", outcome="signal_published", duration_ms=10.0)
        record = ReplayRecord(
            bar_timestamp="2024-06-15T12:00:00",
            scan_result=result,
            tiered_signal=tiered,
        )
        store.append(record)
        with open(store.filepath, "r", encoding="utf-8") as f:
            parsed = json.loads(f.readline())
        assert parsed["tiered_signal"]["tier"] == "B"
        assert parsed["tiered_signal"]["edge_score"] == 0.476
        assert parsed["tiered_signal"]["pattern_name"] == "Bat"
        assert parsed["tiered_signal"]["is_paper_only"] is False