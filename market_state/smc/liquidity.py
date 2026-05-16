from dataclasses import dataclass

from market_state.smc.range_state import Candle


@dataclass(frozen=True)
class LiquiditySweep:
    swept_high: bool
    swept_low: bool
    rejection_close: bool


def detect_liquidity_sweep(
    candle: Candle,
    range_high: float,
    range_low: float,
) -> LiquiditySweep:
    swept_high = candle.high > range_high
    swept_low = candle.low < range_low

    rejection_close = (
        candle.close < range_high
        if swept_high
        else candle.close > range_low
    )

    return LiquiditySweep(
        swept_high=swept_high,
        swept_low=swept_low,
        rejection_close=rejection_close,
    )