from dataclasses import dataclass

from market_state.smc.scoring import (
    SetupScore,
)
from market_state.smc.timeframe import (
    TimeframeContext,
)


@dataclass(frozen=True)
class SignalEvent:
    timeframe: str
    direction: str
    score: int
    valid: bool


def create_signal_event(
    context: TimeframeContext,
    score: SetupScore,
) -> SignalEvent:
    if context.bias.bearish:
        direction = "bearish"
    elif context.bias.bullish:
        direction = "bullish"
    else:
        direction = "neutral"

    return SignalEvent(
        timeframe=context.timeframe,
        direction=direction,
        score=score.score,
        valid=score.valid,
    )