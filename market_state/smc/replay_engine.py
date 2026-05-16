from dataclasses import dataclass
from typing import Sequence

from market_state.smc.context_builder import (
    build_timeframe_context,
)
from market_state.smc.range_state import Candle
from market_state.smc.timeframe import (
    TimeframeContext,
)


@dataclass(frozen=True)
class ReplayStep:
    index: int
    context: TimeframeContext


def replay_timeframe(
    timeframe: str,
    candles: Sequence[Candle],
    minimum_window: int = 4,
) -> list[ReplayStep]:
    steps: list[ReplayStep] = []

    for index in range(minimum_window, len(candles) + 1):
        visible_candles = candles[:index]

        context = build_timeframe_context(
            timeframe=timeframe,
            candles=visible_candles,
        )

        steps.append(
            ReplayStep(
                index=index,
                context=context,
            )
        )

    return steps