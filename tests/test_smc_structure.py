from market_state.smc.range_state import Candle
from market_state.smc.structure import (
    detect_structure_shift,
)


def test_detects_bullish_shift():
    candles = [
        Candle(high=100, low=95, close=98),
        Candle(high=105, low=97, close=101),
    ]

    result = detect_structure_shift(candles)

    assert result.bullish_shift is True
    assert result.bearish_shift is False


def test_detects_bearish_shift():
    candles = [
        Candle(high=100, low=95, close=98),
        Candle(high=99, low=90, close=92),
    ]

    result = detect_structure_shift(candles)

    assert result.bearish_shift is True
    assert result.bullish_shift is False