from market_state.smc.backtest import (
    backtest_bearish_setups,
)
from market_state.smc.bias import BiasState
from market_state.smc.liquidity import LiquiditySweep
from market_state.smc.range_state import (
    Candle,
    RangeState,
)
from market_state.smc.replay_engine import ReplayStep
from market_state.smc.setup import SetupState
from market_state.smc.structure import StructureShift
from market_state.smc.timeframe import (
    TimeframeContext,
)
from market_state.smc.regime import (
    RegimeState,
)

def test_backtests_bearish_setup():
    candles = [
        Candle(close=100),
        Candle(close=99),
        Candle(close=98),
        Candle(close=97),
    ]

    context = TimeframeContext(
        timeframe="15m",

        bias=BiasState(
            bullish=False,
            bearish=True,
            neutral=False,
        ),

        range_state=RangeState(
            is_ranging=True,
            range_high=101,
            range_low=99,
            previous_range_high=101,
            previous_range_low=99,
            equilibrium=100,
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

    result = backtest_bearish_setups(
        replay_steps=replay_steps,
        candles=candles,
    )

    assert result.total_setups == 1

    assert result.successful_setups == 1

    assert result.failed_setups == 0