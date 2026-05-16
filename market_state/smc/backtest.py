from dataclasses import dataclass

from market_state.smc.replay_engine import (
    ReplayStep,
)
from market_state.smc.range_state import Candle


@dataclass(frozen=True)
class BacktestResult:
    total_setups: int

    successful_setups: int

    failed_setups: int


def backtest_bearish_setups(
    replay_steps: list[ReplayStep],
    candles: list[Candle],
    lookahead: int = 3,
) -> BacktestResult:
    successful_setups = 0

    failed_setups = 0

    total_setups = 0

    for step in replay_steps:
        context = step.context

        if not context.setup.bearish_setup:
            continue

        entry_index = step.index - 1

        future_index = entry_index + lookahead

        if future_index >= len(candles):
            continue

        total_setups += 1

        entry_price = candles[
            entry_index
        ].close

        future_price = candles[
            future_index
        ].close

        if future_price < entry_price:
            successful_setups += 1
        else:
            failed_setups += 1

    return BacktestResult(
        total_setups=total_setups,

        successful_setups=successful_setups,

        failed_setups=failed_setups,
    )