from dataclasses import dataclass

from market_state.smc.bias import BiasState
from market_state.smc.displacement import (
    DisplacementState,
)
from market_state.smc.event_memory import (
    EventMemory,
)
from market_state.smc.liquidity import LiquiditySweep
from market_state.smc.premium_discount import (
    PremiumDiscountState,
)
from market_state.smc.range_state import RangeState
from market_state.smc.structure import StructureShift


@dataclass(frozen=True)
class SetupState:
    bearish_setup: bool


def evaluate_setup(
    bias: BiasState,
    range_state: RangeState,
    liquidity: LiquiditySweep,
    structure: StructureShift,
    event_memory: EventMemory,
    displacement: DisplacementState,
    premium_discount: PremiumDiscountState,
) -> SetupState:
    bearish_setup = all(
        [
            bias.bearish,

            range_state.is_ranging,

            (
                liquidity.swept_high
                or event_memory.recent_sweep_high
            ),

            (
                structure.bearish_shift
                or event_memory.recent_bearish_shift
            ),

            displacement.bearish_displacement,

            # premium_discount.in_premium,
        ]
    )

    return SetupState(
        bearish_setup=bearish_setup,
    )