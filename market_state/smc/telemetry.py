from dataclasses import dataclass

from market_state.smc.context_builder import (
    build_timeframe_context,
)
from market_state.smc.range_state import Candle
from market_state.smc.replay_engine import (
    replay_timeframe,
)


@dataclass(frozen=True)
class ReplayTelemetry:
    total_steps: int

    bearish_bias_count: int

    range_count: int

    sweep_high_count: int

    rejection_close_count: int

    bearish_shift_count: int

    bearish_setup_count: int


def analyze_replay(
    timeframe: str,
    candles: list[Candle],
) -> ReplayTelemetry:
    replay_steps = replay_timeframe(
        timeframe=timeframe,
        candles=candles,
    )

    bearish_bias_count = 0

    range_count = 0

    sweep_high_count = 0

    rejection_close_count = 0

    bearish_shift_count = 0

    bearish_setup_count = 0

    for step in replay_steps:
        context = build_timeframe_context(
            timeframe=timeframe,
            candles=candles[:step.index],
        )

        if context.bias.bearish:
            bearish_bias_count += 1

        if context.range_state.is_ranging:
            range_count += 1

        if context.liquidity.swept_high:
            sweep_high_count += 1

        if context.liquidity.rejection_close:
            rejection_close_count += 1

        if context.structure.bearish_shift:
            bearish_shift_count += 1

        if context.setup.bearish_setup:
            bearish_setup_count += 1

    return ReplayTelemetry(
        total_steps=len(replay_steps),

        bearish_bias_count=bearish_bias_count,

        range_count=range_count,

        sweep_high_count=sweep_high_count,

        rejection_close_count=rejection_close_count,

        bearish_shift_count=bearish_shift_count,

        bearish_setup_count=bearish_setup_count,
    )