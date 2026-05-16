from dataclasses import dataclass

from market_state.smc.mtf_engine import (
    MultiTimeframeContext,
    build_mtf_context,
)
from market_state.smc.range_state import Candle
from market_state.smc.time_alignment import (
    find_active_candle,
)


@dataclass(frozen=True)
class MTFReplayStep:
    step: int
    context: MultiTimeframeContext


def replay_mtf(
    htf_candles: list[Candle],
    itf_candles: list[Candle],
    ltf_candles: list[Candle],
) -> list[MTFReplayStep]:
    steps: list[MTFReplayStep] = []

    for ltf_index in range(6, len(ltf_candles) + 1):
        ltf_candle = ltf_candles[
            ltf_index - 1
        ]

        htf_index = find_active_candle(
            candles=htf_candles,
            timestamp=ltf_candle.timestamp,
        )

        itf_index = find_active_candle(
            candles=itf_candles,
            timestamp=ltf_candle.timestamp,
        )

        if htf_index < 6:
            continue

        if itf_index < 6:
            continue

        context = build_mtf_context(
            htf_candles=htf_candles[:htf_index],
            itf_candles=itf_candles[:itf_index],
            ltf_candles=ltf_candles[:ltf_index],
        )

        steps.append(
            MTFReplayStep(
                step=ltf_index,
                context=context,
            )
        )

    return steps