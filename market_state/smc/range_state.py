from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Candle:
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class RangeState:
    is_ranging: bool
    range_high: float
    range_low: float
    equilibrium: float
    range_width_pct: float


def detect_range(
    candles: Sequence[Candle],
    max_range_width_pct: float = 0.03,
) -> RangeState:
    if len(candles) < 4:
        return RangeState(
            is_ranging=False,
            range_high=0.0,
            range_low=0.0,
            equilibrium=0.0,
            range_width_pct=0.0,
        )

    range_high = max(c.high for c in candles)
    range_low = min(c.low for c in candles)

    midpoint = (range_high + range_low) / 2

    if midpoint == 0:
        width_pct = 0.0
    else:
        width_pct = (range_high - range_low) / midpoint

    is_ranging = width_pct <= max_range_width_pct

    return RangeState(
        is_ranging=is_ranging,
        range_high=range_high,
        range_low=range_low,
        equilibrium=midpoint,
        range_width_pct=width_pct,
    )