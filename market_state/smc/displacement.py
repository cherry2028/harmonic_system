from dataclasses import dataclass
from statistics import mean

from market_state.smc.range_state import Candle


@dataclass(frozen=True)
class DisplacementState:
    bearish_displacement: bool


def detect_displacement(
    candles: list[Candle],
    multiplier: float = 1.5,
) -> DisplacementState:
    if len(candles) < 4:
        return DisplacementState(
            bearish_displacement=False,
        )

    current = candles[-1]

    current_range = (
        current.high - current.low
    )

    previous_ranges = [
        candle.high - candle.low
        for candle in candles[-4:-1]
    ]

    average_range = mean(
        previous_ranges
    )

    bearish_displacement = (
        current.close < candles[-2].close
        and current_range
        > average_range * multiplier
    )

    return DisplacementState(
        bearish_displacement=bearish_displacement,
    )