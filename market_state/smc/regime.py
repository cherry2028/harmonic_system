from dataclasses import dataclass

from market_state.smc.range_state import (
    RangeState,
)


@dataclass(frozen=True)
class RegimeState:
    trending: bool
    choppy: bool


def detect_regime(
    range_state: RangeState,
) -> RegimeState:
    trending = (
        range_state.range_width_pct
        > 0.015
    )

    choppy = not trending

    return RegimeState(
        trending=trending,
        choppy=choppy,
    )