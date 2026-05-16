from market_state.smc.liquidity import (
    detect_liquidity_sweep,
)
from market_state.smc.range_state import Candle


def test_detects_high_sweep_rejection():
    candle = Candle(
        high=105,
        low=99,
        close=101,
    )

    result = detect_liquidity_sweep(
        candle=candle,
        range_high=103,
        range_low=98,
    )

    assert result.swept_high is True
    assert result.rejection_close is True


def test_detects_no_sweep():
    candle = Candle(
        high=102,
        low=99,
        close=101,
    )

    result = detect_liquidity_sweep(
        candle=candle,
        range_high=103,
        range_low=98,
    )

    assert result.swept_high is False