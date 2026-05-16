from dataclasses import dataclass
from typing import Sequence

from market_state.smc.range_state import Candle


@dataclass(frozen=True)
class BiasState:
    bullish: bool
    bearish: bool
    neutral: bool


def detect_bias(
    candles: Sequence[Candle],
) -> BiasState:
    if len(candles) < 2:
        return BiasState(
            bullish=False,
            bearish=False,
            neutral=True,
        )

    first_close = candles[0].close
    last_close = candles[-1].close

    bullish = last_close > first_close
    bearish = last_close < first_close

    neutral = not bullish and not bearish

    return BiasState(
        bullish=bullish,
        bearish=bearish,
        neutral=neutral,
    )