from market_state.smc.context_builder import (
    build_timeframe_context,
)
from market_state.smc.range_state import Candle


def test_builds_timeframe_context():
    candles = [
        Candle(high=110, low=100, close=108),
        Candle(high=112, low=101, close=102),
        Candle(high=111, low=99, close=100),
        Candle(high=109, low=95, close=96),
    ]

    context = build_timeframe_context(
        timeframe="15m",
        candles=candles,
    )

    assert context.timeframe == "15m"