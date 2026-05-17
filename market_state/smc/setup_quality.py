from dataclasses import dataclass

from market_state.smc.mtf_engine import (
    MultiTimeframeContext,
)


@dataclass(frozen=True)
class SetupQuality:
    score: int
    confidence: str


def evaluate_setup_quality(
    context: MultiTimeframeContext,
) -> SetupQuality:

    score = 0

    # HTF bias
    if context.htf.bias.bearish:
        score += 25

    # ITF setup
    if context.itf.setup.bearish_setup:
        score += 35

    # LTF alignment
    if context.ltf.bias.bearish:
        score += 20

    # Regime filter
    if context.itf.regime.trending:
        score += 20

    if score >= 80:
        confidence = "A"

    elif score >= 60:
        confidence = "B"

    elif score >= 40:
        confidence = "C"

    else:
        confidence = "D"

    return SetupQuality(
        score=score,
        confidence=confidence,
    )