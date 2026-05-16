from market_state.smc.bias import BiasState
from market_state.smc.liquidity import LiquiditySweep
from market_state.smc.range_state import RangeState
from market_state.smc.setup import SetupState
from market_state.smc.structure import StructureShift
from market_state.smc.timeframe import TimeframeContext


def test_creates_timeframe_context():
    context = TimeframeContext(
        timeframe="15m",
        bias=BiasState(
            bullish=False,
            bearish=True,
            neutral=False,
        ),
        range_state=RangeState(
            is_ranging=True,
            range_high=105,
            range_low=100,
            previous_range_high=105,
            previous_range_low=100,
            equilibrium=102.5,
            range_width_pct=0.02,
        ),
        liquidity=LiquiditySweep(
            swept_high=True,
            swept_low=False,
            rejection_close=True,
        ),
        structure=StructureShift(
            bullish_shift=False,
            bearish_shift=True,
        ),
        setup=SetupState(
            bearish_setup=True,
            
        ),
    )

    assert context.timeframe == "15m"
    assert context.setup.bearish_setup is True