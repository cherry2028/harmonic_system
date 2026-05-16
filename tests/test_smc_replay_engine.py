from market_state.smc.range_state import Candle
from market_state.smc.replay_engine import (
    replay_timeframe,
)


def test_replays_candle_sequence():
    candles = [
        Candle(high=100, low=95, close=98),
        Candle(high=101, low=96, close=99),
        Candle(high=102, low=97, close=100),
        Candle(high=103, low=98, close=101),
        Candle(high=104, low=99, close=102),
    ]

    steps = replay_timeframe(
        timeframe="15m",
        candles=candles,
    )

    assert len(steps) > 0

    final_step = steps[-1]

    assert final_step.context.timeframe == "15m"