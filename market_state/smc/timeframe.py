from dataclasses import dataclass

from market_state.smc.bias import BiasState
from market_state.smc.range_state import RangeState
from market_state.smc.liquidity import LiquiditySweep
from market_state.smc.structure import StructureShift
from market_state.smc.setup import SetupState
from market_state.smc.regime import (
    RegimeState,
)
from market_state.smc.fvg import (
    FVGState,
)


@dataclass(frozen=True)
class TimeframeContext:
    timeframe: str
    bias: BiasState
    range_state: RangeState
    liquidity: LiquiditySweep
    structure: StructureShift
    setup: SetupState
    regime: "RegimeState"
    fvg: FVGState | None = None 