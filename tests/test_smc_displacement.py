from market_state.smc.displacement import (
    detect_displacement,
)
from market_state.smc.range_state import Candle


def test_detects_bearish_displacement():
    candles = [
        Candle(
            high=100,
            low=99,
            close=99.5,
        ),
        Candle(
            high=101,
            low=100,
            close=100.5,
        ),
        Candle(
            high=102,
            low=101,
            close=101.5,
        ),
        Candle(
            high=102,
            low=95,
            close=96,
        ),
    ]

    result = detect_displacement(
        candles=candles,
    )

    assert (
        result.bearish_displacement
        is True
    )