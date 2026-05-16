from market_state.smc.mtf_replay import (
    replay_mtf,
)
from market_state.smc.range_state import Candle


def test_replays_mtf():
    candles = []

    for i in range(60):
        candles.append(
            Candle(
                high=100 + i,
                low=99 + i,
                close=99.5 + i,
            )
        )

    steps = replay_mtf(
        htf_candles=candles,
        itf_candles=candles,
        ltf_candles=candles,
    )

    assert len(steps) > 0