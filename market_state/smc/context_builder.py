from typing import Sequence

from market_state.smc.bias import (
    BiasState,
    detect_bias,
)
from market_state.smc.displacement import (
    DisplacementState,
    detect_displacement,
)
from market_state.smc.event_memory import (
    EventMemory,
)
from market_state.smc.liquidity import (
    LiquiditySweep,
    detect_liquidity_sweep,
)
from market_state.smc.premium_discount import (
    PremiumDiscountState,
    detect_premium_discount,
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
from market_state.smc.regime import (
    RegimeState,
    detect_regime,
)
from market_state.smc.fvg import (
    detect_fvg,
    FVGState,
)


def build_timeframe_context(
    timeframe: str,
    candles: Sequence[Candle],
) -> TimeframeContext:
    bias: BiasState = detect_bias(
        candles
    )

    range_state: RangeState = detect_range(
        candles
    )

    latest_candle = candles[-1]

    liquidity: LiquiditySweep = (
        detect_liquidity_sweep(
            candle=latest_candle,
            range_high=range_state.range_high,
            range_low=range_state.range_low,
        )
    )

    structure: StructureShift = (
        detect_structure_shift(
            candles,
        )
    )

    displacement: DisplacementState = (
        detect_displacement(
            list(candles),
        )
    )

    premium_discount: (
        PremiumDiscountState
    ) = detect_premium_discount(
        close_price=latest_candle.close,
        range_state=range_state,
    )
    regime: RegimeState = (
        detect_regime(
            range_state=range_state,
        )
)
    fvg: FVGState = detect_fvg(
        candles=candles,
    )

    event_memory = EventMemory(
        recent_sweep_high=(
            liquidity.swept_high
        ),

        recent_bearish_shift=(
            structure.bearish_shift
        ),
    )

    setup: SetupState = evaluate_setup(
        bias=bias,
        range_state=range_state,
        liquidity=liquidity,
        structure=structure,
        event_memory=event_memory,
        displacement=displacement,
        premium_discount=(
        premium_discount
        ),
        regime=regime,
        fvg=fvg,
    )

    return TimeframeContext(
        timeframe=timeframe,
        bias=bias,
        range_state=range_state,
        liquidity=liquidity,
        structure=structure,
        setup=setup,
        regime=regime,
        fvg=fvg,
    )