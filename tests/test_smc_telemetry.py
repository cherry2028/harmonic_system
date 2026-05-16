from market_state.smc.data_loader import (
    load_ohlcv,
)
from market_state.smc.telemetry import (
    analyze_replay,
)


def test_generates_replay_telemetry():
    candles = load_ohlcv(
        symbol="BTC/USDT",
        timeframe="15m",
        limit=20,
    )

    telemetry = analyze_replay(
        timeframe="15m",
        candles=candles,
    )

    assert telemetry.total_steps > 0