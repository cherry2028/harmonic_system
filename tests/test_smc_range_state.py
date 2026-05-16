from market_state.smc.range_state import (
    Candle,
    detect_range,
)


def test_detects_ranging_market():
    candles = [
        Candle(high=101, low=99, close=100),
        Candle(high=102, low=100, close=101),
        Candle(high=101.5, low=99.5, close=100.5),
        Candle(high=102, low=99.8, close=101),
    ]

    state = detect_range(candles)

    assert state.is_ranging is True


def test_detects_non_ranging_market():
    candles = [
        Candle(high=100, low=90, close=95),
        Candle(high=110, low=92, close=108),
        Candle(high=120, low=100, close=118),
        Candle(high=130, low=105, close=128),
    ]

    state = detect_range(candles)

    assert state.is_ranging is False