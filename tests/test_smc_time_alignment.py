from market_state.smc.range_state import Candle
from market_state.smc.time_alignment import (
    find_active_candle,
)


def test_finds_active_candle():
    candles = [
        Candle(timestamp=1000),
        Candle(timestamp=2000),
        Candle(timestamp=3000),
    ]

    index = find_active_candle(
        candles=candles,
        timestamp=2500,
    )

    assert index == 2