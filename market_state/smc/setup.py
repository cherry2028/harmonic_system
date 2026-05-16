from dataclasses import dataclass

from market_state.smc.bias import BiasState
from market_state.smc.liquidity import LiquiditySweep
from market_state.smc.memory import SetupMemory
from market_state.smc.range_state import RangeState
from market_state.smc.structure import StructureShift


@dataclass(frozen=True)
class SetupState:
    bearish_setup: bool
    bullish_setup: bool


def evaluate_setup(
    bias: BiasState,
    range_state: RangeState,
    liquidity: LiquiditySweep,
    structure: StructureShift,
    memory: SetupMemory,
) -> SetupState:
    bearish_setup = (
        bias.bearish
        and memory.recent_range_detected
        and liquidity.swept_high
        and liquidity.rejection_close
        and structure.bearish_shift
    )

    bullish_setup = (
        bias.bullish
        and memory.recent_range_detected
        and liquidity.swept_low
        and liquidity.rejection_close
        and structure.bullish_shift
    )

    return SetupState(
        bearish_setup=bearish_setup,
        bullish_setup=bullish_setup,
    )