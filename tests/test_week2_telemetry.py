"""
tests/test_week2_telemetry.py
==============================
Exhaustive pytest coverage for Action 8 telemetry extensions.

Tests cover:
    1.  Module structure — all 7 functions exported
    2.  log_signal() — field correctness, JSONL format, never-raise
    3.  log_daily_counter_block() — field correctness, JSONL, never-raise
    4.  Existing function regression — log_state, log_gate_block guard fix
    5.  JSONL format contract — every record has ts, iso, type
    6.  Never-raise contract — all functions with bad inputs
    7.  Append behavior — multiple calls produce multiple lines
    8.  File naming — each event type writes to its own file

All tests use isolated temp directories — no shared state.
No test modifies the global TELEMETRY_DIR permanently.
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import telemetry
import telemetry.logger as tl
from harmonic_patterns import PatternMatch
from market_state.vector import MarketStateVector
from scoring.pattern_scorer import PatternScorer
from signals.tier import SignalTier
from signals.signal import TieredSignal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_telemetry_dir(tmp_path):
    """
    Every test gets its own telemetry directory.
    Restores the original after each test.
    autouse=True — applied to all tests in this module automatically.
    """
    original = tl.TELEMETRY_DIR
    tl.TELEMETRY_DIR = tmp_path / "telemetry"
    yield tmp_path / "telemetry"
    tl.TELEMETRY_DIR = original


@pytest.fixture(scope="module")
def sample_tiered() -> TieredSignal:
    """A valid TieredSignal for use across multiple tests."""
    m = PatternMatch(
        pattern_name="Gartley", direction="bullish",
        symbol="BTCUSDT", timeframe="1h",
        pivots={"X": 60000, "A": 65000, "B": 62000, "C": 64000, "D": 61500},
        ratios={"AB_XA": 0.618, "BC_AB": 0.382, "CD_BC": 1.272,
                "AD_XA": 0.786, "XD_XA": 0.300},
        validation={"AB_XA": True, "BC_AB": True, "CD_BC": True, "AD_XA": True},
        prz={"entry": 61500.0, "stop": 59800.0,
             "target1": 64000.0, "target2": 65000.0, "target3": 66000.0},
        D_index=295,
        D_timestamp=pd.Timestamp("2024-01-15 14:00"),
        quality_score=0.84,
        metadata={},
    )
    v = MarketStateVector(
        trending=0.04, ranging=0.04, expansion=0.04,
        compression=0.04, reversal=0.80, news_chaos=0.04,
    )
    scored = PatternScorer().score(m, v)
    tiered = SignalTier().classify(scored)
    assert tiered is not None
    return tiered


def read_jsonl(path: Path) -> list:
    """Reads all JSON records from a JSONL file."""
    return [json.loads(line) for line in path.read_text().strip().split("\n")
            if line.strip()]


# ---------------------------------------------------------------------------
# Group 1: Module structure
# ---------------------------------------------------------------------------

class TestModuleStructure:

    def test_all_seven_functions_exported(self):
        expected = [
            "log_state", "log_gate_block", "log_detector_detail",
            "log_scan_cycle", "log_error",
            "log_signal", "log_daily_counter_block",
        ]
        for fn in expected:
            assert hasattr(telemetry, fn), f"telemetry.{fn} missing"
            assert callable(getattr(telemetry, fn))

    def test_set_debug_mode_exported(self):
        assert hasattr(telemetry, "set_debug_mode")
        assert callable(telemetry.set_debug_mode)

    def test_new_functions_importable_directly(self):
        from telemetry.logger import log_signal, log_daily_counter_block
        assert callable(log_signal)
        assert callable(log_daily_counter_block)

    def test_all_exports_in_all_list(self):
        for fn in ["log_signal", "log_daily_counter_block"]:
            assert fn in telemetry.__all__, f"{fn} not in __all__"


# ---------------------------------------------------------------------------
# Group 2: log_signal() field correctness
# ---------------------------------------------------------------------------

class TestLogSignal:

    def test_creates_signal_delivered_file(self, sample_tiered, isolated_telemetry_dir):
        tl.log_signal(sample_tiered)
        assert (isolated_telemetry_dir / "signal_delivered.jsonl").exists()

    def test_record_type_is_signal_delivered(self, sample_tiered, isolated_telemetry_dir):
        tl.log_signal(sample_tiered)
        records = read_jsonl(isolated_telemetry_dir / "signal_delivered.jsonl")
        assert records[0]["type"] == "signal_delivered"

    def test_signal_identity_fields(self, sample_tiered, isolated_telemetry_dir):
        tl.log_signal(sample_tiered)
        r = read_jsonl(isolated_telemetry_dir / "signal_delivered.jsonl")[0]
        assert r["symbol"]    == "BTCUSDT"
        assert r["timeframe"] == "1h"
        assert r["pattern"]   == "Gartley"
        assert r["direction"] == "bullish"

    def test_tier_and_edge_score(self, sample_tiered, isolated_telemetry_dir):
        tl.log_signal(sample_tiered)
        r = read_jsonl(isolated_telemetry_dir / "signal_delivered.jsonl")[0]
        assert r["tier"]       == sample_tiered.tier
        assert abs(r["edge_score"] - sample_tiered.edge_score) < 1e-3

    def test_market_state_field(self, sample_tiered, isolated_telemetry_dir):
        tl.log_signal(sample_tiered)
        r = read_jsonl(isolated_telemetry_dir / "signal_delivered.jsonl")[0]
        assert r["market_state"] == sample_tiered.dominant_state

    def test_trading_levels(self, sample_tiered, isolated_telemetry_dir):
        tl.log_signal(sample_tiered)
        r = read_jsonl(isolated_telemetry_dir / "signal_delivered.jsonl")[0]
        assert r["entry"]   == round(sample_tiered.entry,   4)
        assert r["stop"]    == round(sample_tiered.stop,    4)
        assert r["target1"] == round(sample_tiered.target1, 4)

    def test_risk_reward_field(self, sample_tiered, isolated_telemetry_dir):
        tl.log_signal(sample_tiered)
        r = read_jsonl(isolated_telemetry_dir / "signal_delivered.jsonl")[0]
        if sample_tiered.risk_reward is not None:
            assert r["risk_reward"] == round(sample_tiered.risk_reward, 3)
        else:
            assert r["risk_reward"] is None

    def test_operational_fields(self, sample_tiered, isolated_telemetry_dir):
        tl.log_signal(sample_tiered)
        r = read_jsonl(isolated_telemetry_dir / "signal_delivered.jsonl")[0]
        assert r["risk_pct"]    == sample_tiered.risk_pct
        assert r["max_per_day"] == sample_tiered.max_per_day

    def test_scoring_factor_fields(self, sample_tiered, isolated_telemetry_dir):
        tl.log_signal(sample_tiered)
        r = read_jsonl(isolated_telemetry_dir / "signal_delivered.jsonl")[0]
        assert r["base_score"]     is not None
        assert r["state_discount"] is not None
        assert r["conf_weight"]    is not None

    def test_reasoning_lines_count(self, sample_tiered, isolated_telemetry_dir):
        tl.log_signal(sample_tiered)
        r = read_jsonl(isolated_telemetry_dir / "signal_delivered.jsonl")[0]
        assert r["reasoning_lines"] == len(sample_tiered.reasoning)

    def test_reasoning_content_not_logged(self, sample_tiered, isolated_telemetry_dir):
        """Full reasoning chain must NOT appear in telemetry (too verbose)."""
        tl.log_signal(sample_tiered)
        r = read_jsonl(isolated_telemetry_dir / "signal_delivered.jsonl")[0]
        assert "reasoning" not in r or isinstance(r.get("reasoning"), int)

    def test_record_has_ts_and_iso(self, sample_tiered, isolated_telemetry_dir):
        tl.log_signal(sample_tiered)
        r = read_jsonl(isolated_telemetry_dir / "signal_delivered.jsonl")[0]
        assert "ts"  in r
        assert "iso" in r
        assert isinstance(r["ts"],  float)
        assert isinstance(r["iso"], str)
        assert "T" in r["iso"]    # ISO 8601 format


# ---------------------------------------------------------------------------
# Group 3: log_daily_counter_block() field correctness
# ---------------------------------------------------------------------------

class TestLogDailyCounterBlock:

    def test_creates_daily_counter_block_file(self, isolated_telemetry_dir):
        tl.log_daily_counter_block("BTCUSDT", "1h", "A+", 1, 1, 0.75)
        assert (isolated_telemetry_dir / "daily_counter_block.jsonl").exists()

    def test_record_type_is_correct(self, isolated_telemetry_dir):
        tl.log_daily_counter_block("BTCUSDT", "1h", "A+", 1, 1, 0.75)
        r = read_jsonl(isolated_telemetry_dir / "daily_counter_block.jsonl")[0]
        assert r["type"] == "daily_counter_block"

    def test_symbol_and_timeframe(self, isolated_telemetry_dir):
        tl.log_daily_counter_block("ETHUSDT", "4h", "A", 3, 3, 0.55)
        r = read_jsonl(isolated_telemetry_dir / "daily_counter_block.jsonl")[0]
        assert r["symbol"]    == "ETHUSDT"
        assert r["timeframe"] == "4h"

    def test_tier_and_counts(self, isolated_telemetry_dir):
        tl.log_daily_counter_block("BTCUSDT", "1h", "A+", 1, 1, 0.75)
        r = read_jsonl(isolated_telemetry_dir / "daily_counter_block.jsonl")[0]
        assert r["tier"]          == "A+"
        assert r["current_count"] == 1
        assert r["max_count"]     == 1

    def test_edge_score_rounded(self, isolated_telemetry_dir):
        tl.log_daily_counter_block("BTCUSDT", "1h", "A+", 1, 1, 0.75001)
        r = read_jsonl(isolated_telemetry_dir / "daily_counter_block.jsonl")[0]
        assert r["edge_score"] == round(0.75001, 4)

    def test_utc_date_field(self, isolated_telemetry_dir):
        tl.log_daily_counter_block("BTCUSDT", "1h", "A+", 1, 1, 0.75)
        r = read_jsonl(isolated_telemetry_dir / "daily_counter_block.jsonl")[0]
        expected_date = datetime.now(timezone.utc).date().isoformat()
        assert r["utc_date"] == expected_date

    def test_record_has_ts_and_iso(self, isolated_telemetry_dir):
        tl.log_daily_counter_block("BTCUSDT", "1h", "A+", 1, 1, 0.75)
        r = read_jsonl(isolated_telemetry_dir / "daily_counter_block.jsonl")[0]
        assert "ts"  in r
        assert "iso" in r


# ---------------------------------------------------------------------------
# Group 4: Existing function regression — guard position fix
# ---------------------------------------------------------------------------

class TestExistingFunctionRegression:

    def test_log_state_none_does_not_raise(self, isolated_telemetry_dir):
        """log_state(None) must not raise and must not write a file."""
        tl.log_state(None)
        assert not (isolated_telemetry_dir / "market_state.jsonl").exists()

    def test_log_state_valid_writes_file(self, isolated_telemetry_dir):
        v = MarketStateVector(
            trending=0.05, ranging=0.70, expansion=0.05,
            compression=0.08, reversal=0.07, news_chaos=0.05,
        )
        tl.log_state(v)
        assert (isolated_telemetry_dir / "market_state.jsonl").exists()

    def test_log_gate_block_none_does_not_raise(self, isolated_telemetry_dir):
        tl.log_gate_block("BTCUSDT", "1h", "NEWS_CHAOS", "reason", None)
        assert not (isolated_telemetry_dir / "gate_block.jsonl").exists()

    def test_log_gate_block_valid_writes_file(self, isolated_telemetry_dir):
        v = MarketStateVector(
            trending=0.05, ranging=0.70, expansion=0.05,
            compression=0.08, reversal=0.07, news_chaos=0.05,
        )
        tl.log_gate_block("BTCUSDT", "1h", "NEWS_CHAOS", "chaos detected", v)
        assert (isolated_telemetry_dir / "gate_block.jsonl").exists()

    def test_log_scan_cycle_writes_file(self, isolated_telemetry_dir):
        tl.log_scan_cycle("BTCUSDT", "1h", 123.4, "gate_blocked")
        assert (isolated_telemetry_dir / "scan_cycle.jsonl").exists()

    def test_log_error_writes_file(self, isolated_telemetry_dir):
        tl.log_error("test_location", "test error")
        assert (isolated_telemetry_dir / "error.jsonl").exists()


# ---------------------------------------------------------------------------
# Group 5: JSONL format contract
# ---------------------------------------------------------------------------

class TestJSONLFormatContract:

    def test_every_signal_record_has_required_base_fields(
        self, sample_tiered, isolated_telemetry_dir
    ):
        """Every JSONL record must have ts, iso, type."""
        tl.log_signal(sample_tiered)
        r = read_jsonl(isolated_telemetry_dir / "signal_delivered.jsonl")[0]
        assert "ts"   in r, "Missing 'ts'"
        assert "iso"  in r, "Missing 'iso'"
        assert "type" in r, "Missing 'type'"

    def test_every_counter_block_record_has_required_base_fields(
        self, isolated_telemetry_dir
    ):
        tl.log_daily_counter_block("BTCUSDT", "1h", "A+", 1, 1, 0.75)
        r = read_jsonl(isolated_telemetry_dir / "daily_counter_block.jsonl")[0]
        assert "ts"   in r
        assert "iso"  in r
        assert "type" in r

    def test_ts_is_unix_timestamp_float(self, sample_tiered, isolated_telemetry_dir):
        tl.log_signal(sample_tiered)
        r = read_jsonl(isolated_telemetry_dir / "signal_delivered.jsonl")[0]
        assert isinstance(r["ts"], float)
        # Should be a recent Unix timestamp (after 2020)
        assert r["ts"] > 1_580_000_000

    def test_iso_is_utc_string(self, sample_tiered, isolated_telemetry_dir):
        tl.log_signal(sample_tiered)
        r = read_jsonl(isolated_telemetry_dir / "signal_delivered.jsonl")[0]
        iso = r["iso"]
        assert isinstance(iso, str)
        assert iso.endswith("Z")
        assert "T" in iso

    def test_each_record_is_valid_json(self, sample_tiered, isolated_telemetry_dir):
        """Each line must be parseable as standalone JSON."""
        tl.log_signal(sample_tiered)
        tl.log_signal(sample_tiered)
        file_path = isolated_telemetry_dir / "signal_delivered.jsonl"
        for line in file_path.read_text().strip().split("\n"):
            obj = json.loads(line)
            assert isinstance(obj, dict)


# ---------------------------------------------------------------------------
# Group 6: Never-raise contract
# ---------------------------------------------------------------------------

class TestNeverRaiseContract:

    @pytest.mark.parametrize("bad_input", [
        None, {}, "string", 42, 3.14, [], object(),
    ])
    def test_log_signal_never_raises(self, bad_input, isolated_telemetry_dir):
        tl.log_signal(bad_input)   # must not raise

    @pytest.mark.parametrize("args", [
        (None, None, None, "bad", "bad", "bad"),
        ("BTCUSDT", "1h", "A+", "not_int", "not_int", "not_float"),
        ("BTCUSDT", "1h", "A+", -1, -1, -0.5),
    ])
    def test_log_daily_counter_block_never_raises(self, args, isolated_telemetry_dir):
        tl.log_daily_counter_block(*args)   # must not raise

    def test_log_state_none_never_raises(self, isolated_telemetry_dir):
        tl.log_state(None)

    def test_log_gate_block_none_vector_never_raises(self, isolated_telemetry_dir):
        tl.log_gate_block("X", "1h", "NEWS_CHAOS", "reason", None)

    def test_all_functions_safe_after_bad_inputs(
        self, sample_tiered, isolated_telemetry_dir
    ):
        """Good calls must work correctly even after bad calls."""
        tl.log_signal(None)
        tl.log_signal({})
        tl.log_daily_counter_block(None, None, None, "x", "x", "x")

        # Valid calls still work
        tl.log_signal(sample_tiered)
        file_path = isolated_telemetry_dir / "signal_delivered.jsonl"
        assert file_path.exists()
        records = read_jsonl(file_path)
        # At least one valid record written
        valid = [r for r in records if r.get("symbol") == "BTCUSDT"]
        assert len(valid) >= 1


# ---------------------------------------------------------------------------
# Group 7: Append behavior
# ---------------------------------------------------------------------------

class TestAppendBehavior:

    def test_multiple_signals_append_multiple_lines(
        self, sample_tiered, isolated_telemetry_dir
    ):
        for _ in range(5):
            tl.log_signal(sample_tiered)
        file_path = isolated_telemetry_dir / "signal_delivered.jsonl"
        lines = file_path.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_multiple_counter_blocks_append_multiple_lines(
        self, isolated_telemetry_dir
    ):
        tiers = ["A+", "A", "B"]
        for tier in tiers:
            tl.log_daily_counter_block("BTCUSDT", "1h", tier, 1, 1, 0.5)
        file_path = isolated_telemetry_dir / "daily_counter_block.jsonl"
        records = read_jsonl(file_path)
        assert len(records) == 3

    def test_lines_are_independent_json_objects(
        self, sample_tiered, isolated_telemetry_dir
    ):
        tl.log_signal(sample_tiered)
        tl.log_signal(sample_tiered)
        file_path = isolated_telemetry_dir / "signal_delivered.jsonl"
        records   = read_jsonl(file_path)
        # Each record must be a valid complete dict
        for r in records:
            assert isinstance(r, dict)
            assert r["type"] == "signal_delivered"


# ---------------------------------------------------------------------------
# Group 8: File naming
# ---------------------------------------------------------------------------

class TestFileNaming:

    def test_signal_writes_to_signal_delivered_file(
        self, sample_tiered, isolated_telemetry_dir
    ):
        tl.log_signal(sample_tiered)
        files = list(isolated_telemetry_dir.glob("*.jsonl"))
        names = [f.name for f in files]
        assert "signal_delivered.jsonl" in names
        # Must NOT write to other files
        assert "daily_counter_block.jsonl" not in names

    def test_counter_block_writes_to_counter_block_file(
        self, isolated_telemetry_dir
    ):
        tl.log_daily_counter_block("BTCUSDT", "1h", "A+", 1, 1, 0.75)
        files = list(isolated_telemetry_dir.glob("*.jsonl"))
        names = [f.name for f in files]
        assert "daily_counter_block.jsonl" in names
        assert "signal_delivered.jsonl"    not in names

    def test_each_event_type_has_separate_file(
        self, sample_tiered, isolated_telemetry_dir
    ):
        v = MarketStateVector(
            trending=0.05, ranging=0.70, expansion=0.05,
            compression=0.08, reversal=0.07, news_chaos=0.05,
        )
        tl.log_state(v)
        tl.log_signal(sample_tiered)
        tl.log_scan_cycle("BTCUSDT", "1h", 100.0, "signal_published")
        tl.log_daily_counter_block("BTCUSDT", "1h", "A+", 1, 1, 0.75)
        tl.log_error("test", "error")

        expected_files = [
            "market_state.jsonl",
            "signal_delivered.jsonl",
            "scan_cycle.jsonl",
            "daily_counter_block.jsonl",
            "error.jsonl",
        ]
        existing = {f.name for f in isolated_telemetry_dir.glob("*.jsonl")}
        for name in expected_files:
            assert name in existing, f"Expected file {name} not found"