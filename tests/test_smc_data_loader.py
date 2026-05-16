from market_state.smc.data_loader import (
    load_ohlcv,
)


def test_loads_ohlcv():
    candles = load_ohlcv(
        symbol="BTC/USDT",
        timeframe="15m",
        limit=5,
    )

    assert len(candles) == 5