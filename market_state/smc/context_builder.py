from typing import Sequence

from market_state.smc.bias import (
    BiasState,
    detect_bias,
)
from market_state.smc.liquidity import (
    LiquiditySweep,
    detect_liquidity_sweep,
)
from market_state.smc.memory import (
    SetupMemory,
)
from market_state.smc.range_state import (
    Candle,
    RangeState,
    detect_range,
)
from market_state.smc.setup import (
    SetupState,
    evaluate_setup,
)
from market_state.smc.structure import (
    StructureShift,
    detect_structure_shift,
)
from market_state.smc.timeframe import (
    TimeframeContext,
)


def build_timeframe_context(
    timeframe: str,
    candles: Sequence[Candle],
) -> TimeframeContext:
    bias: BiasState = detect_bias(candles)

    range_state: RangeState = detect_range(candles)

    latest_candle = candles[-1]

    liquidity: LiquiditySweep = detect_liquidity_sweep(
        candle=latest_candle,
        range_high=range_state.range_high,
        range_low=range_state.range_low,
    )

    structure: StructureShift = detect_structure_shift(
        candles,
    )

    memory = SetupMemory(
        recent_range_detected=range_state.is_ranging,
        recent_liquidity_sweep=(
            liquidity.swept_high
            or liquidity.swept_low
        ),
    )

    setup: SetupState = evaluate_setup(
        bias=bias,
        range_state=range_state,
        liquidity=liquidity,
        structure=structure,
        memory=memory,
    )

    return TimeframeContext(
        timeframe=timeframe,
        bias=bias,
        range_state=range_state,
        liquidity=liquidity,
        structure=structure,
        setup=setup,
    )