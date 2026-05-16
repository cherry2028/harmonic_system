from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Candle:
    timestamp: int = 0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0


@dataclass(frozen=True)
class RangeState:
    is_ranging: bool
    range_high: float
    range_low: float
    previous_range_high:float
    previous_range_low:float
    equilibrium: float
    range_width_pct: float


def detect_range(
    candles: Sequence[Candle],
    lookback: int = 5,
    max_range_width_pct: float = 0.01,
) -> RangeState:
    if len(candles) < 4:
        return RangeState(
            is_ranging=False,
            range_high=0.0,
            range_low=0.0,
            equilibrium=0.0,
            range_width_pct=0.0,
        )
    historical_candles = candles[-(lookback + 1):-1]
    range_high = max(c.high for c in historical_candles)
    range_low = min(c.low for c in historical_candles)

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
        previous_range_high=range_high,
        previous_range_low=range_low,
    )