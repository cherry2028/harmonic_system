from market_state.smc.bias import BiasState
from market_state.smc.engine import (
    MultiTimeframeContext,
)
from market_state.smc.liquidity import LiquiditySweep
from market_state.smc.range_state import RangeState
from market_state.smc.setup import SetupState
from market_state.smc.structure import StructureShift
from market_state.smc.timeframe import TimeframeContext
from market_state.smc.regime import (
    RegimeState,
)


def make_context(
    bearish_bias: bool,
    bearish_setup: bool,
    bearish_shift: bool,
):
    return TimeframeContext(
        timeframe="15m",
        bias=BiasState(
            bullish=not bearish_bias,
            bearish=bearish_bias,
            neutral=False,
        ),
        range_state=RangeState(
            is_ranging=True,
            range_high=105,
            range_low=100,
            equilibrium=102.5,
            range_width_pct=0.02,
            previous_range_high=105,
            previous_range_low=100,
        ),
        liquidity=LiquiditySweep(
            swept_high=True,
            swept_low=False,
            rejection_close=True,
        ),
        structure=StructureShift(
            bullish_shift=False,
            bearish_shift=bearish_shift,
        ),
        setup=SetupState(
            bearish_setup=bearish_setup,
            
        ),
        regime=RegimeState(
            trending=True,
            choppy=False,
        ),
    )


def test_detects_bearish_alignment():
    context = MultiTimeframeContext(
        htf=make_context(
            bearish_bias=True,
            bearish_setup=False,
bearish_pressure=25,
            bearish_shift=False,
        ),
        itf=make_context(
            bearish_bias=True,
            bearish_setup=true,
bearish_pressure=75,
            bearish_shift=False,
        ),
        ltf=make_context(
            bearish_bias=True,
            bearish_setup=False,
bearish_pressure=25,
            bearish_shift=True,
        ),
    )

    assert context.bearish_alignment is True