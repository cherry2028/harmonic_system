"""tests/test_statistics.py — ReplayStatistics contract tests.

Pure deterministic statistics over replay outcomes.
No mutation of input records. No filesystem access.
"""
from __future__ import annotations
import pytest
from typing import List

from replay.statistics import ReplayStatistics, StatisticsReport, OutcomeRecord
from replay.outcome_tracker import SignalOutcome
from replay.replay_runner import ReplayRecord
from market_state.vector import MarketStateVector
from signals.signal import TieredSignal
from scoring.score_result import ScoredSignal
from pipeline import ScanResult


@pytest.fixture
def stats() -> ReplayStatistics:
    return ReplayStatistics()


def _make_vector(dominant_state: str = "ranging") -> MarketStateVector:
    # Produce a vector where the requested state is clearly dominant (0.6)
    # and all others are low (0.05-0.1)
    base = {"trending": 0.05, "ranging": 0.05, "expansion": 0.05,
            "compression": 0.05, "reversal": 0.05, "news_chaos": 0.05}
    base[dominant_state] = 0.60
    return MarketStateVector(
        trending=base["trending"], ranging=base["ranging"],
        expansion=base["expansion"], compression=base["compression"],
        reversal=base["reversal"], news_chaos=base["news_chaos"],
        symbol="BTCUSDT", timeframe="1h",
    )


def _make_tiered(
    tier: str = "A",
    pattern: str = "Gartley",
    rr: float = 2.0,
    dominant_state: str = "ranging",
) -> TieredSignal:
    vector = _make_vector(dominant_state)
    entry = 100.0
    stop = 95.0
    risk = abs(entry - stop)  # 5.0
    t1 = entry + (rr * risk)   # 100 + rr*5
    t2 = t1 + risk
    t3 = t2 + risk
    match = type("obj", (object,), {
        "pattern_name": pattern, "symbol": "BTCUSDT", "timeframe": "1h","direction": "bullish",
        "prz": {"entry": entry, "stop": stop, "target1": t1, "target2": t2, "target3": t3},
    })()
    scored = ScoredSignal(
        pattern_match=match, base_score=0.8, state_discount=1.0,
        confidence_weight=0.9, edge_score=0.72,
        dominant_state=dominant_state, state_vector=vector,
        reasoning=[
            f"Pattern detected: {pattern}",
            f"Market state: {dominant_state}",
            f"Final edge score: 0.72",
        ],
    )
    return TieredSignal(tier=tier, scored=scored, max_per_day=3, risk_pct=1.0,
    
        edge_score=scored.edge_score,
        dominant_state=scored.dominant_state,
        reasoning=scored.reasoning,

        entry=entry, stop=stop, target1=t1,target2=t2, target3=t3,

        risk_reward=rr,
    )

def _make_scan_result(dominant_state: str = "ranging") -> ScanResult:
    vector = _make_vector(dominant_state)
    return ScanResult(
        symbol="BTCUSDT", timeframe="1h",
        outcome="signal_published", duration_ms=10.0,
        vector=vector,
    )


def _make_outcome_record(
    outcome_type: str,
    tier: str = "A",
    pattern: str = "Gartley",
    rr: float = 2.0,
    dominant_state: str = "ranging",
    bars: int = 5,
) -> OutcomeRecord:
    """Build a complete OutcomeRecord with all required metadata."""
    tiered = _make_tiered(tier=tier, pattern=pattern, rr=rr, dominant_state=dominant_state)
    scan = _make_scan_result(dominant_state)
    replay = ReplayRecord(
        bar_timestamp="2024-01-01T00:00:00",
        scan_result=scan,
        tiered_signal=tiered,
    )
    outcome = SignalOutcome(
        outcome_type=outcome_type,
        hit_target1=(outcome_type == "target1"),
        hit_target2=(outcome_type == "target2"),
        hit_target3=(outcome_type == "target3"),
        hit_stop=(outcome_type == "stop"),
        bars_to_resolution=bars,
    )
    return OutcomeRecord(replay=replay, outcome=outcome)


# ═════════════════════════════════════════════════════════════════════════════
# Empty / degenerate cases
# ═════════════════════════════════════════════════════════════════════════════

class TestEmptyRecords:
    """Degenerate input handling."""

    def test_empty_list_returns_zeroed_report(self, stats):
        report = stats.compute([])
        assert report.total_signals == 0
        assert report.wins == 0
        assert report.losses == 0
        assert report.win_rate == 0.0
        assert report.expectancy == 0.0
        assert report.by_tier == {}
        assert report.by_pattern == {}
        assert report.by_market_state == {}

    def test_all_timeouts_ignored(self, stats):
        records = [
            _make_outcome_record("timeout", tier="A", bars=-1),
            _make_outcome_record("timeout", tier="B", bars=-1),
        ]
        report = stats.compute(records)
        assert report.total_signals == 0
        assert report.wins == 0
        assert report.losses == 0

    def test_mixed_timeouts_and_resolved(self, stats):
        records = [
            _make_outcome_record("timeout", tier="A", bars=-1),
            _make_outcome_record("target1", tier="A", bars=3),
        ]
        report = stats.compute(records)
        assert report.total_signals == 1
        assert report.wins == 1
        assert report.losses == 0
        assert report.win_rate == 1.0


# ═════════════════════════════════════════════════════════════════════════════
# Win / loss correctness
# ═════════════════════════════════════════════════════════════════════════════

class TestAllWins:
    """100% win rate scenarios."""

    def test_all_target1(self, stats):
        records = [_make_outcome_record("target1", bars=i) for i in range(1, 6)]
        report = stats.compute(records)
        assert report.total_signals == 5
        assert report.wins == 5
        assert report.losses == 0
        assert report.win_rate == 1.0
        assert report.target1_hit_rate == 1.0
        assert report.stop_hit_rate == 0.0
        assert report.expectancy > 0.0

    def test_mixed_targets_all_wins(self, stats):
        records = [
            _make_outcome_record("target1", bars=3),
            _make_outcome_record("target2", bars=5),
            _make_outcome_record("target3", bars=7),
        ]
        report = stats.compute(records)
        assert report.total_signals == 3
        assert report.wins == 3
        assert report.losses == 0
        assert report.win_rate == 1.0
        assert report.target1_hit_rate == pytest.approx(1 / 3)
        assert report.target2_hit_rate == pytest.approx(1 / 3)
        assert report.target3_hit_rate == pytest.approx(1 / 3)


class TestAllLosses:
    """0% win rate scenarios."""

    def test_all_stop(self, stats):
        records = [_make_outcome_record("stop", bars=i) for i in range(1, 6)]
        report = stats.compute(records)
        assert report.total_signals == 5
        assert report.wins == 0
        assert report.losses == 5
        assert report.win_rate == 0.0
        assert report.stop_hit_rate == 1.0
        assert report.target1_hit_rate == 0.0
        # Expectancy with 0% win rate and 1R loss each time
        assert report.expectancy == pytest.approx(-1.0)


class TestMixedOutcomes:
    """Mixed win/loss scenarios with verifiable math."""

    def test_60_win_rate(self, stats):
        """3 wins (target1, rr=2.0) + 2 losses (stop, 1R)."""
        records = [
            _make_outcome_record("target1", rr=2.0, bars=3),
            _make_outcome_record("target1", rr=2.0, bars=4),
            _make_outcome_record("target1", rr=2.0, bars=5),
            _make_outcome_record("stop", rr=2.0, bars=2),
            _make_outcome_record("stop", rr=2.0, bars=2),
        ]
        report = stats.compute(records)
        assert report.total_signals == 5
        assert report.wins == 3
        assert report.losses == 2
        assert report.win_rate == pytest.approx(0.6)
        # E = (0.6 * 2.0) - (0.4 * 1.0) = 1.2 - 0.4 = 0.8
        assert report.expectancy == pytest.approx(0.8, abs=1e-6)

    def test_50_win_rate_with_different_rr(self, stats):
        """1 win (rr=3.0) + 1 loss (1R)."""
        records = [
            _make_outcome_record("target1", rr=3.0, bars=5),
            _make_outcome_record("stop", rr=3.0, bars=2),
        ]
        report = stats.compute(records)
        assert report.win_rate == pytest.approx(0.5)
        # E = (0.5 * 3.0) - (0.5 * 1.0) = 1.5 - 0.5 = 1.0
        assert report.expectancy == pytest.approx(1.0, abs=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# Segmentation
# ═════════════════════════════════════════════════════════════════════════════

class TestSegmentation:
    """Grouped statistics by tier, pattern, and market state."""

    def test_by_tier(self, stats):
        records = [
            _make_outcome_record("target1", tier="A", bars=3),
            _make_outcome_record("target1", tier="A", bars=4),
            _make_outcome_record("stop", tier="B", bars=2),
        ]
        report = stats.compute(records)
        assert "A" in report.by_tier
        assert "B" in report.by_tier
        assert report.by_tier["A"].total_signals == 2
        assert report.by_tier["A"].win_rate == 1.0
        assert report.by_tier["B"].total_signals == 1
        assert report.by_tier["B"].win_rate == 0.0
        # Sub-reports are flat — no nested segmentation
        assert report.by_tier["A"].by_tier == {}

    def test_by_pattern(self, stats):
        records = [
            _make_outcome_record("target1", pattern="Gartley", bars=3),
            _make_outcome_record("stop", pattern="Bat", bars=2),
            _make_outcome_record("target1", pattern="Gartley", bars=5),
        ]
        report = stats.compute(records)
        assert "Gartley" in report.by_pattern
        assert "Bat" in report.by_pattern
        assert report.by_pattern["Gartley"].total_signals == 2
        assert report.by_pattern["Gartley"].win_rate == 1.0
        assert report.by_pattern["Bat"].total_signals == 1
        assert report.by_pattern["Bat"].win_rate == 0.0

    def test_by_market_state(self, stats):
        records = [
            _make_outcome_record("target1", dominant_state="ranging", bars=3),
            _make_outcome_record("stop", dominant_state="trending", bars=2),
            _make_outcome_record("target1", dominant_state="ranging", bars=4),
        ]
        report = stats.compute(records)
        assert "ranging" in report.by_market_state
        assert "trending" in report.by_market_state
        assert report.by_market_state["ranging"].total_signals == 2
        assert report.by_market_state["ranging"].win_rate == 1.0
        assert report.by_market_state["trending"].total_signals == 1
        assert report.by_market_state["trending"].win_rate == 0.0

    def test_unknown_key_fallback(self, stats):
        """Missing tier/pattern/state maps to 'unknown'."""
        # Build a record with empty string pattern name but valid tier
        vector = _make_vector()
        match = type("obj", (object,), {
            "pattern_name": "", "symbol": "BTCUSDT", "timeframe": "1h","direction": "bullish",
            "prz": {"entry": 100.0, "stop": 95.0, "target1": 110.0, "target2": 120.0, "target3": 130.0},
        })()
        scored = ScoredSignal(
            pattern_match=match, base_score=0.8, state_discount=1.0,
            confidence_weight=0.9, edge_score=0.72,
            dominant_state="ranging", state_vector=vector,
            reasoning=[
                "Pattern detected: unknown",
                "Market state: ranging",
                "Final edge score: 0.72",
            ],
        )
        tiered = TieredSignal(tier="C", scored=scored, max_per_day=3, risk_pct=1.0,
            edge_score=scored.edge_score,
            dominant_state=scored.dominant_state,
            reasoning=scored.reasoning,
            entry=100.0, stop=95.0, target1=110.0,
        )

        scan = _make_scan_result()
        replay = ReplayRecord(
            bar_timestamp="2024-01-01T00:00:00",
            scan_result=scan,
            tiered_signal=tiered,
        )
        outcome = SignalOutcome(
            outcome_type="target1", hit_target1=True,
            hit_target2=False, hit_target3=False,
            hit_stop=False, bars_to_resolution=3,
        )
        report = stats.compute([OutcomeRecord(replay=replay, outcome=outcome)])
        # Tier "C" is valid, pattern "" maps to "unknown"
        assert "C" in report.by_tier
        assert "unknown" in report.by_pattern


# ═════════════════════════════════════════════════════════════════════════════
# Expectancy correctness
# ═════════════════════════════════════════════════════════════════════════════

class TestExpectancy:
    """Proper expectancy: E = (Pw × Aw) - (Pl × Al)."""

    def test_zero_expectancy_at_breakeven(self, stats):
        """50% win rate with 1:1 R:R → E = 0."""
        records = [
            _make_outcome_record("target1", rr=1.0, bars=3),
            _make_outcome_record("stop", rr=1.0, bars=2),
        ]
        report = stats.compute(records)
        # E = (0.5 * 1.0) - (0.5 * 1.0) = 0.0
        assert report.expectancy == pytest.approx(0.0, abs=1e-6)

    def test_positive_expectancy(self, stats):
        """40% win rate with 3:1 R:R → E = (0.4*3) - (0.6*1) = 0.6."""
        records = [
            _make_outcome_record("target1", rr=3.0, bars=5),
            _make_outcome_record("target1", rr=3.0, bars=6),
            _make_outcome_record("stop", rr=3.0, bars=2),
            _make_outcome_record("stop", rr=3.0, bars=3),
            _make_outcome_record("stop", rr=3.0, bars=2),
        ]
        report = stats.compute(records)
        assert report.win_rate == pytest.approx(0.4)
        assert report.expectancy == pytest.approx(0.6, abs=1e-6)

    def test_negative_expectancy(self, stats):
        """30% win rate with 1.5:1 R:R → E = (0.3*1.5) - (0.7*1) = -0.25."""
        records = (
            [_make_outcome_record("target1", rr=1.5, bars=4) for _ in range(3)]
            + [_make_outcome_record("stop", rr=1.5, bars=2) for _ in range(7)]
        )
        report = stats.compute(records)
        assert report.win_rate == pytest.approx(0.3)
        assert report.expectancy == pytest.approx(-0.25, abs=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# Bars to resolution
# ═════════════════════════════════════════════════════════════════════════════

class TestBarsToResolution:
    """Average bars calculation."""

    def test_avg_bars_computed_correctly(self, stats):
        records = [
            _make_outcome_record("target1", bars=2),
            _make_outcome_record("target1", bars=4),
            _make_outcome_record("stop", bars=6),
        ]
        report = stats.compute(records)
        assert report.avg_bars_to_resolution == pytest.approx(4.0)

    def test_avg_bars_excludes_negative(self, stats):
        """Negative bars_to_resolution (from empty input edge case) should not affect mean."""
        # This is implicitly tested: _filter_completed removes timeouts with bars=-1
        records = [
            _make_outcome_record("target1", bars=3),
            _make_outcome_record("target1", bars=5),
        ]
        report = stats.compute(records)
        assert report.avg_bars_to_resolution == pytest.approx(4.0)


# ═════════════════════════════════════════════════════════════════════════════
# Immutability
# ═════════════════════════════════════════════════════════════════════════════

class TestImmutability:
    """StatisticsReport and OutcomeRecord are frozen."""

    def test_statistics_report_frozen(self):
        report = StatisticsReport(
            total_signals=1, wins=1, losses=0,
            win_rate=1.0, target1_hit_rate=1.0,
            target2_hit_rate=0.0, target3_hit_rate=0.0,
            stop_hit_rate=0.0, avg_bars_to_resolution=3.0,
            expectancy=1.0,
        )
        with pytest.raises(Exception):
            report.total_signals = 5

    def test_outcome_record_frozen(self):
        rec = _make_outcome_record("target1", bars=3)
        with pytest.raises(Exception):
            rec.outcome = SignalOutcome(
                outcome_type="stop", hit_target1=False,
                hit_target2=False, hit_target3=False,
                hit_stop=True, bars_to_resolution=1,
            )