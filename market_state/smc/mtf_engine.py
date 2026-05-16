from dataclasses import dataclass

from market_state.smc.context_builder import (
    build_timeframe_context,
)
from market_state.smc.range_state import Candle
from market_state.smc.timeframe import (
    TimeframeContext,
)


@dataclass(frozen=True)
class MultiTimeframeContext:
    htf: TimeframeContext
    itf: TimeframeContext
    ltf: TimeframeContext

    aligned_bearish: bool


def build_mtf_context(
    htf_candles: list[Candle],
    itf_candles: list[Candle],
    ltf_candles: list[Candle],
) -> MultiTimeframeContext:
    htf = build_timeframe_context(
        timeframe="4h",
        candles=htf_candles,
    )

    itf = build_timeframe_context(
        timeframe="15m",
        candles=itf_candles,
    )

    ltf = build_timeframe_context(
        timeframe="5m",
        candles=ltf_candles,
    )

    aligned_bearish = (
        htf.bias.bearish
        and itf.setup.bearish_setup
    )

    return MultiTimeframeContext(
        htf=htf,
        itf=itf,
        ltf=ltf,

        aligned_bearish=aligned_bearish,
    )