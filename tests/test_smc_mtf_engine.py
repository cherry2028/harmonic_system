from market_state.smc.mtf_engine import (
    build_mtf_context,
)
from market_state.smc.range_state import Candle


def test_builds_mtf_context():
    candles = [
        Candle(
            high=100,
            low=99,
            close=99.5,
        ),
        Candle(
            high=101,
            low=99,
            close=100,
        ),
        Candle(
            high=102,
            low=100,
            close=101,
        ),
        Candle(
            high=103,
            low=101,
            close=102,
        ),
        Candle(
            high=104,
            low=102,
            close=103,
        ),
        Candle(
            high=105,
            low=103,
            close=104,
        ),
    ]

    context = build_mtf_context(
        htf_candles=candles,
        itf_candles=candles,
        ltf_candles=candles,
    )

    assert context.htf.timeframe == "4h"

    assert context.itf.timeframe == "15m"

    assert context.ltf.timeframe == "5m"

    assert isinstance(
        context.aligned_bearish,
        bool,
    )