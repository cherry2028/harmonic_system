from market_state.smc.csv_loader import (
    load_csv_candles,
)


def test_loads_csv_candles():
    candles = load_csv_candles(
        "data/BTCUSDT/4h/clean_btcusdt_4h.csv"
    )

    assert len(candles) > 1000

    first = candles[0]

    assert first.open > 0

    assert first.high > 0

    assert first.low > 0

    assert first.close > 0