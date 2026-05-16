from dataclasses import dataclass

from market_state.smc.engine import (
    MultiTimeframeContext,
)


@dataclass(frozen=True)
class SetupScore:
    score: int
    valid: bool


def score_setup(
    context: MultiTimeframeContext,
) -> SetupScore:
    score = 0

    if context.htf.bias.bearish:
        score += 30

    if context.itf.setup.bearish_setup:
        score += 40

    if context.ltf.structure.bearish_shift:
        score += 30

    valid = score >= 70

    return SetupScore(
        score=score,
        valid=valid,
    )