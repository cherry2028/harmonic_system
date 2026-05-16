from market_state.smc.bias import detect_bias
from market_state.smc.range_state import Candle


def test_detects_bullish_bias():
    candles = [
        Candle(high=100, low=95, close=96),
        Candle(high=105, low=100, close=104),
    ]

    result = detect_bias(candles)

    assert result.bullish is True
    assert result.bearish is False


def test_detects_bearish_bias():
    candles = [
        Candle(high=110, low=100, close=108),
        Candle(high=105, low=95, close=96),
    ]

    result = detect_bias(candles)

    assert result.bearish is True
    assert result.bullish is False