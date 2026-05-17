from dataclasses import dataclass

from market_state.smc.bias import BiasState
from market_state.smc.displacement import (
    DisplacementState,
)
from market_state.smc.event_memory import (
    EventMemory,
)
from market_state.smc.fvg import FVGState
from market_state.smc.liquidity import LiquiditySweep
from market_state.smc.premium_discount import (
    PremiumDiscountState,
)
from market_state.smc.range_state import RangeState
from market_state.smc.regime import (
    RegimeState,
)
from market_state.smc.structure import StructureShift


@dataclass(frozen=True)
class SetupState:
    bearish_setup: bool
    bearish_pressure: int


def evaluate_setup(
    bias: BiasState,
    range_state: RangeState,
    liquidity: LiquiditySweep,
    structure: StructureShift,
    event_memory: EventMemory,
    displacement: DisplacementState,
    premium_discount: PremiumDiscountState,
    regime: RegimeState,
    fvg: FVGState,
) -> SetupState:

    bearish_pressure = 0

    if bias.bearish:
        bearish_pressure += 20

    if structure.bearish_shift:
        bearish_pressure += 20

    if displacement.bearish_displacement:
        bearish_pressure += 20

    if liquidity.swept_high:
        bearish_pressure += 15

    if fvg.bearish_fvg:
        bearish_pressure += 15

    if regime.trending:
        bearish_pressure += 10

    bearish_setup = (
        bearish_pressure >= 60
    )

    return SetupState(
        bearish_setup=bearish_setup,
        bearish_pressure=bearish_pressure,
    )