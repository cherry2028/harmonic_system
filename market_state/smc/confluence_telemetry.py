from dataclasses import dataclass

from market_state.smc.replay_engine import (
    ReplayStep,
)


@dataclass(frozen=True)
class ConfluenceTelemetry:
    total_steps: int

    sweep_count: int

    bearish_shift_count: int

    bearish_setup_count: int

    aligned_bearish_count: int


def build_confluence_telemetry(
    replay_steps: list[ReplayStep],
) -> ConfluenceTelemetry:
    sweep_count = 0

    bearish_shift_count = 0

    bearish_setup_count = 0

    aligned_bearish_count = 0

    for step in replay_steps:
        context = step.context

        if context.liquidity.swept_high:
            sweep_count += 1

        if context.structure.bearish_shift:
            bearish_shift_count += 1

        if context.setup.bearish_setup:
            bearish_setup_count += 1

        if (
            context.bias.bearish
            and context.setup.bearish_setup
        ):
            aligned_bearish_count += 1

    return ConfluenceTelemetry(
        total_steps=len(replay_steps),

        sweep_count=sweep_count,

        bearish_shift_count=bearish_shift_count,

        bearish_setup_count=bearish_setup_count,

        aligned_bearish_count=aligned_bearish_count,
    )