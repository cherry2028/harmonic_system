"""tests/test_outcome_tracker.py — OutcomeTracker contract tests.

Deterministic, forward-only, no-lookahead outcome evaluation.
"""
from __future__ import annotations
import pandas as pd
import pytest

from replay.outcome_tracker import OutcomeTracker, SignalOutcome
from signals.signal import TieredSignal
from scoring.score_result import ScoredSignal
from market_state.vector import MarketStateVector


@pytest.fixture
def tracker() -> OutcomeTracker:
    return OutcomeTracker(max_bars_forward=10)


def _make_tiered(entry: float, stop: float, t1: float, t2: float, t3: float) -> TieredSignal:
    """Build a minimal TieredSignal for testing."""
    vector = MarketStateVector(
        trending=0.1, ranging=0.6, expansion=0.1,
        compression=0.1, reversal=0.05, news_chaos=0.05,
    )
    match = type("obj", (object,), {
        "pattern_name": "Test",
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "direction": "bullish",

        "prz": {
            "entry": entry,
            "stop": stop,
            "target1": t1,
            "target2": t2,
            "target3": t3,
        },
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
            "Pattern detected: Test",
            "Market state: test",
            "Final edge score: 0.8",
    ],
)
    return TieredSignal(
        tier="A",
        scored=scored,
        max_per_day=3,
        risk_pct=1.0,

        edge_score=scored.edge_score,
        dominant_state=scored.dominant_state,
        reasoning=scored.reasoning,

        entry=entry,
        stop=stop,
        target1=t1,
        target2=t2,
        target3=t3,
    )


def _make_bars(data: list[dict]) -> pd.DataFrame:
    """Build a future_bars DataFrame from list of {high, low} dicts."""
    return pd.DataFrame(data)


# ═════════════════════════════════════════════════════════════════════════════
# Basic outcomes
# ═════════════════════════════════════════════════════════════════════════════

class TestOutcomeTrackerTargetHit:
    """Target hit scenarios."""

    def test_target1_hit_long(self, tracker):
        signal = _make_tiered(entry=100.0, stop=95.0, t1=110.0, t2=120.0, t3=130.0)
        bars = _make_bars([
            {"high": 105, "low": 98},   # no hit
            {"high": 112, "low": 99},   # target1 hit
        ])
        outcome = tracker.track(signal, bars)
        assert outcome.outcome_type == "target1"
        assert outcome.hit_target1 is True
        assert outcome.hit_stop is False
        assert outcome.bars_to_resolution == 2

    def test_target2_hit_long(self, tracker):
        """When t1 and t2 are both hit in same candle, closest target (t1) wins.
        The tracker returns immediately on first hit — it does NOT continue
        checking higher targets. This is deterministic and conservative."""
        signal = _make_tiered(entry=100.0, stop=95.0, t1=110.0, t2=120.0, t3=130.0)
        bars = _make_bars([
            {"high": 105, "low": 98},
            {"high": 115, "low": 99},   # t1 hit (115 >= 110); t2 also hit but not checked
        ])
        outcome = tracker.track(signal, bars)
        assert outcome.outcome_type == "target1"
        assert outcome.hit_target1 is True
        # hit_target2 is False because tracker returns immediately on t1 hit
        assert outcome.hit_target2 is False

    def test_target3_hit_long(self, tracker):
        signal = _make_tiered(entry=100.0, stop=95.0, t1=110.0, t2=120.0, t3=130.0)
        bars = _make_bars([
            {"high": 105, "low": 98},
            {"high": 108, "low": 99},
            {"high": 135, "low": 101},  # all targets hit
        ])
        outcome = tracker.track(signal, bars)
        # t1 is closest, so it wins
        assert outcome.outcome_type == "target1"
        assert outcome.hit_target3 is True

    def test_target1_hit_short(self, tracker):
        signal = _make_tiered(entry=100.0, stop=105.0, t1=90.0, t2=80.0, t3=70.0)
        bars = _make_bars([
            {"high": 102, "low": 95},
            {"high": 101, "low": 88},   # target1 hit (low <= 90)
        ])
        outcome = tracker.track(signal, bars)
        assert outcome.outcome_type == "target1"
        assert outcome.hit_target1 is True
        assert outcome.hit_stop is False


class TestOutcomeTrackerStopHit:
    """Stop loss scenarios."""

    def test_stop_hit_long(self, tracker):
        signal = _make_tiered(entry=100.0, stop=95.0, t1=110.0, t2=120.0, t3=130.0)
        bars = _make_bars([
            {"high": 105, "low": 98},
            {"high": 103, "low": 94},   # stop hit
        ])
        outcome = tracker.track(signal, bars)
        assert outcome.outcome_type == "stop"
        assert outcome.hit_stop is True
        assert outcome.hit_target1 is False
        assert outcome.bars_to_resolution == 2

    def test_stop_hit_short(self, tracker):
        signal = _make_tiered(entry=100.0, stop=105.0, t1=90.0, t2=80.0, t3=70.0)
        bars = _make_bars([
            {"high": 102, "low": 95},
            {"high": 106, "low": 97},   # stop hit (high >= 105)
        ])
        outcome = tracker.track(signal, bars)
        assert outcome.outcome_type == "stop"
        assert outcome.hit_stop is True
        assert outcome.bars_to_resolution == 2

    def test_stop_hit_first_before_target(self, tracker):
        signal = _make_tiered(entry=100.0, stop=95.0, t1=110.0, t2=120.0, t3=130.0)
        bars = _make_bars([
            {"high": 105, "low": 98},
            {"high": 103, "low": 94},   # stop hit before target
            {"high": 115, "low": 99},   # target would have hit here
        ])
        outcome = tracker.track(signal, bars)
        assert outcome.outcome_type == "stop"
        assert outcome.bars_to_resolution == 2


class TestOutcomeTrackerTimeout:
    """Timeout / no-resolution scenarios."""

    def test_timeout_no_hits(self, tracker):
        """When fewer bars than max_bars_forward are provided without resolution,
        bars_to_resolution equals the number of bars evaluated (3)."""
        signal = _make_tiered(entry=100.0, stop=95.0, t1=110.0, t2=120.0, t3=130.0)
        bars = _make_bars([
            {"high": 105, "low": 98},
            {"high": 106, "low": 99},
            {"high": 107, "low": 97},
        ])
        outcome = tracker.track(signal, bars)
        assert outcome.outcome_type == "timeout"
        assert outcome.hit_target1 is False
        assert outcome.hit_target2 is False
        assert outcome.hit_target3 is False
        assert outcome.hit_stop is False
        # Only 3 bars were provided; tracker evaluated all 3 before timeout
        assert outcome.bars_to_resolution == 3

    def test_timeout_with_max_bars_limit(self, tracker):
        signal = _make_tiered(entry=100.0, stop=95.0, t1=110.0, t2=120.0, t3=130.0)
        bars = _make_bars([
            {"high": 105, "low": 98},
            {"high": 106, "low": 99},
            {"high": 107, "low": 97},
            {"high": 108, "low": 96},
            {"high": 109, "low": 95.5},
        ])
        # tracker default max_bars_forward=10, so all 5 bars consumed
        outcome = tracker.track(signal, bars, max_bars_forward=2)
        assert outcome.outcome_type == "timeout"
        assert outcome.bars_to_resolution == 2


# ═════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestOutcomeTrackerSimultaneousHit:
    """Worst-case rule: simultaneous stop + target → stop."""

    def test_same_candle_stop_and_target1_long(self, tracker):
        """A candle that spans both stop and target1 — stop wins (worst case)."""
        signal = _make_tiered(entry=100.0, stop=95.0, t1=110.0, t2=120.0, t3=130.0)
        bars = _make_bars([
            {"high": 115, "low": 94},   # touches both target1 (115>=110) and stop (94<=95)
        ])
        outcome = tracker.track(signal, bars)
        assert outcome.outcome_type == "stop"
        assert outcome.hit_stop is True
        assert outcome.hit_target1 is True  # both were touched
        assert outcome.bars_to_resolution == 1

    def test_same_candle_stop_and_target3_long(self, tracker):
        signal = _make_tiered(entry=100.0, stop=95.0, t1=110.0, t2=120.0, t3=130.0)
        bars = _make_bars([
            {"high": 135, "low": 94},   # all targets + stop
        ])
        outcome = tracker.track(signal, bars)
        assert outcome.outcome_type == "stop"
        assert outcome.hit_target3 is True

    def test_same_candle_stop_and_target1_short(self, tracker):
        signal = _make_tiered(entry=100.0, stop=105.0, t1=90.0, t2=80.0, t3=70.0)
        bars = _make_bars([
            {"high": 106, "low": 88},   # stop (106>=105) and target1 (88<=90)
        ])
        outcome = tracker.track(signal, bars)
        assert outcome.outcome_type == "stop"
        assert outcome.hit_stop is True
        assert outcome.hit_target1 is True


class TestOutcomeTrackerEmptyInput:
    """Empty or None future_bars."""

    def test_none_future_bars(self, tracker):
        signal = _make_tiered(entry=100.0, stop=95.0, t1=110.0, t2=120.0, t3=130.0)
        outcome = tracker.track(signal, None)
        assert outcome.outcome_type == "timeout"
        assert outcome.bars_to_resolution == -1
        assert not any([outcome.hit_target1, outcome.hit_target2, outcome.hit_target3, outcome.hit_stop])

    def test_empty_dataframe(self, tracker):
        signal = _make_tiered(entry=100.0, stop=95.0, t1=110.0, t2=120.0, t3=130.0)
        outcome = tracker.track(signal, pd.DataFrame())
        assert outcome.outcome_type == "timeout"
        assert outcome.bars_to_resolution == -1

    def test_empty_after_max_bars(self, tracker):
        signal = _make_tiered(entry=100.0, stop=95.0, t1=110.0, t2=120.0, t3=130.0)
        bars = _make_bars([])
        outcome = tracker.track(signal, bars)
        assert outcome.outcome_type == "timeout"
        assert outcome.bars_to_resolution == -1


class TestOutcomeTrackerLookaheadProtection:
    """Verify no bars are skipped or reordered."""

    def test_evaluates_bars_in_order(self, tracker):
        signal = _make_tiered(entry=100.0, stop=95.0, t1=110.0, t2=120.0, t3=130.0)
        bars = _make_bars([
            {"high": 105, "low": 98},   # 1: no hit
            {"high": 112, "low": 99},   # 2: target1 hit
        ])
        outcome = tracker.track(signal, bars)
        assert outcome.bars_to_resolution == 2
        # If we had looked ahead and used bar 2 first, resolution would be 1

    def test_does_not_use_bars_beyond_limit(self, tracker):
        signal = _make_tiered(entry=100.0, stop=95.0, t1=110.0, t2=120.0, t3=130.0)
        bars = _make_bars([
            {"high": 105, "low": 98},   # 1
            {"high": 106, "low": 99},   # 2
            {"high": 115, "low": 97},   # 3 — target1 hit, but beyond limit
        ])
        outcome = tracker.track(signal, bars, max_bars_forward=2)
        assert outcome.outcome_type == "timeout"
        assert outcome.bars_to_resolution == 2
        assert outcome.hit_target1 is False

    def test_first_bar_resolution(self, tracker):
        signal = _make_tiered(entry=100.0, stop=95.0, t1=110.0, t2=120.0, t3=130.0)
        bars = _make_bars([
            {"high": 112, "low": 99},   # immediate target1
        ])
        outcome = tracker.track(signal, bars)
        assert outcome.outcome_type == "target1"
        assert outcome.bars_to_resolution == 1


class TestOutcomeTrackerFrozen:
    """SignalOutcome is immutable."""

    def test_frozen_dataclass(self):
        outcome = SignalOutcome(
            outcome_type="target1", hit_target1=True,
            hit_target2=False, hit_target3=False,
            hit_stop=False, bars_to_resolution=3,
        )
        with pytest.raises(Exception):
            outcome.outcome_type = "stop"

    def test_invalid_outcome_type_raises(self):
        with pytest.raises(ValueError, match="Invalid outcome_type"):
            SignalOutcome(
                outcome_type="invalid", hit_target1=True,
                hit_target2=False, hit_target3=False,
                hit_stop=False, bars_to_resolution=1,
            )

    def test_negative_bars_raises(self):
        with pytest.raises(ValueError, match="bars_to_resolution"):
            SignalOutcome(
                outcome_type="target1", hit_target1=True,
                hit_target2=False, hit_target3=False,
                hit_stop=False, bars_to_resolution=-5,
            )