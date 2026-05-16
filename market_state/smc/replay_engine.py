from dataclasses import dataclass
from typing import Sequence

from market_state.smc.context_builder import (
    build_timeframe_context,
)
from market_state.smc.event_memory import (
    build_event_memory,
)
from market_state.smc.range_state import Candle
from market_state.smc.setup import (
    evaluate_setup,
)
from market_state.smc.timeframe import (
    TimeframeContext,
)
from market_state.smc.displacement import (
    detect_displacement,
)
from market_state.smc.premium_discount import (
    detect_premium_discount,
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

    swept_high_history: list[bool] = []

    bearish_shift_history: list[bool] = []

    for index in range(minimum_window, len(candles) + 1):
        visible_candles = candles[:index]

        context = build_timeframe_context(
            timeframe=timeframe,
            candles=visible_candles,
        )

        swept_high_history.append(
            context.liquidity.swept_high
        )

        bearish_shift_history.append(
            context.structure.bearish_shift
        )

        event_memory = build_event_memory(
            swept_high_history=swept_high_history,
            bearish_shift_history=bearish_shift_history,
        )

        updated_setup = evaluate_setup(
            bias=context.bias,
            range_state=context.range_state,
            liquidity=context.liquidity,
            structure=context.structure,
            event_memory=event_memory,
            displacement=detect_displacement(
                list(visible_candles)
            ),
            premium_discount=(
                detect_premium_discount(
                    close_price=(
                        visible_candles[-1].close
                    ),
                    range_state=(
                        context.range_state
                    ),
                )
            ),
        )

        updated_context = TimeframeContext(
            timeframe=context.timeframe,
            bias=context.bias,
            range_state=context.range_state,
            liquidity=context.liquidity,
            structure=context.structure,
            setup=updated_setup,
        )

        steps.append(
            ReplayStep(
                index=index,
                context=updated_context,
            )
        )

    return steps