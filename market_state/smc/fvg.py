from dataclasses import dataclass
from typing import Sequence

from market_state.smc.range_state import Candle

@dataclass(frozen=True)
class FVGState:
    bullish_fvg: bool
    bearish_fvg: bool
    gap_high: float
    gap_low: float


def detect_fvg(
    candles: Sequence[Candle],
) -> FVGState:

    if len(candles) < 3:
        return FVGState(
            bullish_fvg=False,
            bearish_fvg=False,
            gap_high=0.0,
            gap_low=0.0,
        )

    candle_1 = candles[-3]
    candle_2 = candles[-2]
    candle_3 = candles[-1]

    bullish_fvg = (
        candle_1.high < candle_3.low
    )

    bearish_fvg = (
        candle_1.low > candle_3.high
    )

    if bullish_fvg:
        return FVGState(
            bullish_fvg=True,
            bearish_fvg=False,
            gap_high=candle_3.low,
            gap_low=candle_1.high,
        )

    if bearish_fvg:
        return FVGState(
            bullish_fvg=False,
            bearish_fvg=True,
            gap_high=candle_1.low,
            gap_low=candle_3.high,
        )

    return FVGState(
        bullish_fvg=False,
        bearish_fvg=False,
        gap_high=0.0,
        gap_low=0.0,
    )