from market_state.smc.bias import BiasState
from market_state.smc.displacement import (
    DisplacementState,
)
from market_state.smc.event_memory import (
    EventMemory,
)
from market_state.smc.premium_discount import (
    PremiumDiscountState,
)
from market_state.smc.liquidity import LiquiditySweep
from market_state.smc.range_state import RangeState
from market_state.smc.setup import evaluate_setup
from market_state.smc.structure import StructureShift


def test_detects_bearish_setup():
    bias = BiasState(
        bullish=False,
        bearish=True,
        neutral=False,
    )

    range_state = RangeState(
        is_ranging=True,
        range_high=105,
        range_low=100,
        previous_range_high=105,
        previous_range_low=100,
        equilibrium=102.5,
        range_width_pct=0.02,
    )

    liquidity = LiquiditySweep(
        swept_high=True,
        swept_low=False,
        rejection_close=True,
    )

    structure = StructureShift(
        bullish_shift=False,
        bearish_shift=True,
    )

    event_memory = EventMemory(
        recent_sweep_high=True,
        recent_bearish_shift=True,
    )

    displacement = DisplacementState(
        bearish_displacement=True,
    )
    premium_discount = (
        PremiumDiscountState(
            in_premium=True,
            in_discount=False,
        )
    )

    result = evaluate_setup(
        bias=bias,
        range_state=range_state,
        liquidity=liquidity,
        structure=structure,
        event_memory=event_memory,
        displacement=displacement,
        premium_discount=premium_discount,
    )

    assert result.bearish_setup is True