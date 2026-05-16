from dataclasses import dataclass
from typing import Sequence

from market_state.smc.range_state import Candle


@dataclass(frozen=True)
class StructureShift:
    bullish_shift: bool
    bearish_shift: bool


def detect_structure_shift(
    candles: Sequence[Candle],
) -> StructureShift:
    if len(candles) < 2:
        return StructureShift(
            bullish_shift=False,
            bearish_shift=False,
        )

    previous = candles[-2]
    current = candles[-1]

    bullish_shift = current.close > previous.high
    bearish_shift = current.close < previous.low

    return StructureShift(
        bullish_shift=bullish_shift,
        bearish_shift=bearish_shift,
    )