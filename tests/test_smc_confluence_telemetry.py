from market_state.smc.bias import BiasState
from market_state.smc.confluence_telemetry import (
    build_confluence_telemetry,
)
from market_state.smc.liquidity import LiquiditySweep
from market_state.smc.range_state import RangeState
from market_state.smc.replay_engine import ReplayStep
from market_state.smc.setup import SetupState
from market_state.smc.structure import StructureShift
from market_state.smc.regime import (
    RegimeState,
)
from market_state.smc.timeframe import (
    TimeframeContext,
)


def test_builds_confluence_telemetry():
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
            bearish_setup=true,
bearish_pressure=75,
        ),
        regime=RegimeState(
            trending=True,
            choppy=False,
        ),
    )

    replay_steps = [
        ReplayStep(
            index=1,
            context=context,
        )
    ]

    telemetry = build_confluence_telemetry(
        replay_steps=replay_steps,
    )

    assert telemetry.total_steps == 1

    assert telemetry.sweep_count == 1

    assert telemetry.bearish_shift_count == 1

    assert telemetry.bearish_setup_count == 1

    assert telemetry.aligned_bearish_count == 1