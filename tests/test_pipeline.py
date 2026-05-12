"""tests/test_pipeline.py — Gate 0: ScanResult contract tests

Additive-only. No modification of existing passing tests.
"""
from __future__ import annotations
import pytest
from dataclasses import FrozenInstanceError

from pipeline import ScanResult
from market_state.vector import MarketStateVector
from signals.gate import GateResult
from signals.signal import TieredSignal
from scoring.score_result import ScoredSignal


class TestScanResultConstruction:
    """Verify ScanResult dataclass invariants and defaults."""

    def test_minimal_construction(self):
        result = ScanResult(symbol="BTCUSDT", timeframe="1h", outcome="fetch_failed", duration_ms=0.0)
        assert result.symbol == "BTCUSDT"
        assert result.timeframe == "1h"
        assert result.outcome == "fetch_failed"
        assert result.duration_ms == 0.0
        assert result.vector is None
        assert result.gate_result is None
        assert result.swings == 0
        assert result.patterns == 0
        assert result.scored_count == 0
        assert result.tiered_signals == []
        assert result.block_code is None
        assert result.error is None

    def test_full_construction(self):
        vector = MarketStateVector(
            trending=0.10, ranging=0.60, expansion=0.10,
            compression=0.10, reversal=0.05, news_chaos=0.05,
            symbol="BTCUSDT", timeframe="1h",
        )
        gate = GateResult(
            is_blocked=False, block_code="PASS",
            reason="All checks passed", vector=vector,
        )
        result = ScanResult(
            symbol="BTCUSDT", timeframe="1h",
            outcome="signal_published", duration_ms=150.5,
            vector=vector, gate_result=gate,
            swings=12, patterns=3, scored_count=2,
            tiered_signals=[],
        )
        assert result.vector is vector
        assert result.gate_result is gate
        assert result.swings == 12
        assert result.patterns == 3
        assert result.scored_count == 2

    def test_frozen_immutable(self):
        result = ScanResult(symbol="X", timeframe="1h", outcome="test", duration_ms=0.0)
        with pytest.raises(FrozenInstanceError):
            result.symbol = "Y"


class TestScanResultProperties:
    """Verify computed properties."""

    def test_published_count_empty(self):
        result = ScanResult(symbol="X", timeframe="1h", outcome="signal_published", duration_ms=0.0)
        assert result.published_count == 0

    def test_is_success_true(self):
        result = ScanResult(symbol="X", timeframe="1h", outcome="signal_published", duration_ms=0.0)
        assert result.is_success is True

    def test_is_success_false(self):
        for outcome in ["fetch_failed", "gate_blocked", "no_swings", "no_patterns", "below_threshold", "error"]:
            result = ScanResult(symbol="X", timeframe="1h", outcome=outcome, duration_ms=0.0)
            assert result.is_success is False


class TestScanResultOutcomeValues:
    """Verify all expected outcome strings are accepted (no validation enforced — string is free)."""

    @pytest.mark.parametrize("outcome", [
        "fetch_failed", "gate_blocked", "no_swings", "no_patterns",
        "below_threshold", "cap_blocked", "signal_published", "error",
    ])
    def test_outcome_accepted(self, outcome):
        result = ScanResult(symbol="X", timeframe="1h", outcome=outcome, duration_ms=0.0)
        assert result.outcome == outcome


class TestScanResultWithTieredSignals:
    """Verify ScanResult correctly holds TieredSignal list."""

    def test_tiered_signals_list(self):
        vector = MarketStateVector(
            trending=0.10, ranging=0.60, expansion=0.10,
            compression=0.10, reversal=0.05, news_chaos=0.05,
        )
        gate = GateResult(is_blocked=False, block_code="PASS", reason="ok", vector=vector)
        # Build minimal ScoredSignal + TieredSignal
        match = type("obj", (object,), {
            "pattern_name": "Gartley", "symbol": "BTCUSDT", "timeframe": "1h",
            "prz": {"entry": 50000.0, "stop": 49500.0, "target1": 51000.0, "target2": 52000.0, "target3": 53000.0},
        })()
        scored = ScoredSignal(
            pattern_match=match, base_score=0.8, state_discount=1.0,
            confidence_weight=0.9, edge_score=0.72,
            dominant_state="reversal", state_vector=vector,
            reasoning=[
                "Bullish Gartley detected",
                "Market state favors reversal",
                "Final edge score strong enough",
            ],
        )
        tiered = TieredSignal(
            tier="A",
            scored=scored,
            max_per_day=3,
            risk_pct=1.0,
            edge_score=0.72,
            dominant_state="reversal",
            reasoning=[
                "Bullish Gartley detected",
                "Market state favors reversal",
                "Final edge score strong enough",
            ],
            entry=50000.0,
            stop=49500.0,
            target1=51000.0,
        )
        result = ScanResult(
            symbol="BTCUSDT", timeframe="1h",
            outcome="signal_published", duration_ms=200.0,
            vector=vector, gate_result=gate,
            swings=8, patterns=2, scored_count=1,
            tiered_signals=[tiered],
        )
        assert result.published_count == 1
        assert result.tiered_signals[0].tier == "A"
        assert result.tiered_signals[0].edge_score == 0.72


class TestScanResultRepr:
    """Verify ScanResult has sensible repr."""

    def test_repr(self):
        result = ScanResult(symbol="BTCUSDT", timeframe="1h", outcome="fetch_failed", duration_ms=42.0)
        r = repr(result)
        assert "BTCUSDT" in r
        assert "1h" in r
        assert "fetch_failed" in r



# ─────────────────────────────────────────────────────────────────────────────
# Gate 1: ScanPipeline skeleton tests
# ─────────────────────────────────────────────────────────────────────────────

import pytest
from unittest.mock import MagicMock

from pipeline import ScanPipeline, ScanResult
from data.fetcher import DataFetcher
from market_state.engine import MarketStateEngine
from patterns.patterns.swing_detector import AdaptiveSwingDetector
from patterns.patterns.harmonic_detector import HarmonicDetector
from scoring.pattern_scorer import PatternScorer
from signals.gate import HostileMarketGate
from signals.tier import SignalTier
from signals.daily_counter import DailyCounter
from delivery.presentation import SignalPresentation
from delivery.telegram_formatter import TelegramFormatter


@pytest.fixture
def mock_components():
    """Return a dict of MagicMock instances for every pipeline dependency."""
    return {
        "data_fetcher": MagicMock(spec=DataFetcher),
        "market_state_engine": MagicMock(spec=MarketStateEngine),
        "swing_detector": MagicMock(spec=AdaptiveSwingDetector),
        "harmonic_detector": MagicMock(spec=HarmonicDetector),
        "pattern_scorer": MagicMock(spec=PatternScorer),
        "hostile_gate": MagicMock(spec=HostileMarketGate),
        "signal_tier": MagicMock(spec=SignalTier),
        "daily_counter": MagicMock(spec=DailyCounter),
        "telemetry": MagicMock(),
        "signal_presentation": MagicMock(spec=SignalPresentation),
        "telegram_formatter": MagicMock(spec=TelegramFormatter),
    }


class TestScanPipelineConstruction:
    """Verify dependency injection architecture and construction contracts."""

    def test_construct_with_all_components(self, mock_components):
        pipeline = ScanPipeline(**mock_components)
        assert pipeline is not None
        assert isinstance(pipeline, ScanPipeline)

    def test_components_property_returns_all_eleven(self, mock_components):
        pipeline = ScanPipeline(**mock_components)
        comps = pipeline.components
        expected_keys = {
            "data_fetcher", "market_state_engine", "swing_detector",
            "harmonic_detector", "pattern_scorer", "hostile_gate",
            "signal_tier", "daily_counter", "telemetry",
            "signal_presentation", "telegram_formatter",
        }
        assert set(comps.keys()) == expected_keys
        assert len(comps) == 11

    def test_components_property_returns_same_instances(self, mock_components):
        pipeline = ScanPipeline(**mock_components)
        comps = pipeline.components
        for key, expected in mock_components.items():
            assert comps[key] is expected, f"{key} is not the same instance"

    def test_positional_args_rejected(self, mock_components):
        """__init__ is keyword-only — positional args must raise TypeError."""
        with pytest.raises(TypeError):
            ScanPipeline(mock_components["data_fetcher"])

    def test_missing_component_raises_type_error(self, mock_components):
        """Omitting any required component must raise TypeError."""
        for key in mock_components:
            partial = {k: v for k, v in mock_components.items() if k != key}
            with pytest.raises(TypeError):
                ScanPipeline(**partial)

    def test_extra_kwargs_ignored_or_accepted(self, mock_components):
        """Extra kwargs should not break construction (Python ignores them by default
        unless __init__ is strict). Since we use keyword-only args, extra kwargs
        will raise TypeError — which is acceptable."""
        with pytest.raises(TypeError):
            ScanPipeline(**mock_components, extra_thing=MagicMock())


class TestScanPipelineScanOneSignature:
    """Verify scan_one() signature and pre-execution contract."""

    def test_scan_one_raises_not_implemented(self, mock_components):
        pipeline = ScanPipeline(**mock_components)
        with pytest.raises(NotImplementedError):
            pipeline.scan_one("BTCUSDT", "1h")

    def test_scan_one_accepts_dry_run(self, mock_components):
        pipeline = ScanPipeline(**mock_components)
        with pytest.raises(NotImplementedError):
            pipeline.scan_one("BTCUSDT", "1h", dry_run=True)

    def test_scan_one_returns_scanresult_type_hint(self, mock_components):
        """Verify the method exists and has correct signature via inspect."""
        import inspect
        pipeline = ScanPipeline(**mock_components)
        sig = inspect.signature(pipeline.scan_one)
        params = list(sig.parameters.keys())
        assert params == ["symbol", "timeframe", "dry_run"]
        assert sig.parameters["dry_run"].default is False

    def test_scan_one_keyword_only_dry_run(self, mock_components):
        """dry_run must be keyword-only (cannot be passed positionally)."""
        pipeline = ScanPipeline(**mock_components)
        with pytest.raises(TypeError):
            pipeline.scan_one("BTCUSDT", "1h", True)


class TestScanPipelineNoHiddenState:
    """Verify pipeline holds no per-scan mutable state."""

    def test_no_mutable_defaults_on_instance(self, mock_components):
        pipeline = ScanPipeline(**mock_components)
        # scan_one has not run; no mutable state should exist
        assert not hasattr(pipeline, "_last_result")
        assert not hasattr(pipeline, "_scan_cache")

    def test_two_pipelines_are_independent(self, mock_components):
        p1 = ScanPipeline(**mock_components)
        p2 = ScanPipeline(**{k: MagicMock() for k in mock_components})
        assert p1.components["data_fetcher"] is not p2.components["data_fetcher"]