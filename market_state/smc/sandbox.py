from market_state.smc.backtest import (
    backtest_bearish_setups,
)
from market_state.smc.confluence_telemetry import (
    build_confluence_telemetry,
)
from market_state.smc.csv_loader import (
    load_csv_candles,
)
from market_state.smc.mtf_replay import (
    replay_mtf,
)
from market_state.smc.replay_engine import (
    replay_timeframe,
)
from market_state.smc.setup_quality import (
    evaluate_setup_quality,
)


SYMBOL = "BTCUSDT"

HTF = "4h"

ITF = "15m"

LTF = "5m"


def run_sandbox():
    print("=" * 60)

    print("LOADING HISTORICAL CSV DATA")

    print("=" * 60)

    htf_candles = load_csv_candles(
        f"data/{SYMBOL}/{HTF}/clean_{SYMBOL.lower()}_{HTF}.csv"
    )

    itf_candles = load_csv_candles(
        f"data/{SYMBOL}/{ITF}/clean_{SYMBOL.lower()}_{ITF}.csv"
    )

    ltf_candles = load_csv_candles(
        f"data/{SYMBOL}/{LTF}/clean_{SYMBOL.lower()}_{LTF}.csv"
    )

    print(
        f"HTF Candles Loaded: "
        f"{len(htf_candles)}"
    )

    print(
        f"ITF Candles Loaded: "
        f"{len(itf_candles)}"
    )

    print(
        f"LTF Candles Loaded: "
        f"{len(ltf_candles)}"
    )

    print()

    replay_steps = replay_mtf(
        htf_candles=htf_candles[-600:],
        itf_candles=itf_candles[-9600:],
        ltf_candles=ltf_candles[-28800:],
    )

    print("=" * 60)

    print("MTF HISTORICAL REPLAY")

    print("=" * 60)

    for step in replay_steps[-5:]:
        context = step.context

        quality = evaluate_setup_quality(
            context
        )

        latest_ltf = ltf_candles[
            step.step - 1
        ]

        print("-" * 60)

        print(
            f"STEP: {step.step}"
        )

        print(
            f"LTF Timestamp: "
            f"{latest_ltf.timestamp}"
        )

        print(
            f"LTF CLOSE: "
            f"{latest_ltf.close}"
        )

        print()

        print(
            f"HTF Bearish Bias: "
            f"{context.htf.bias.bearish}"
        )

        print(
            f"Setup Quality Score: "
            f"{quality.score}"
        )

        print(
            f"Confidence Grade: "
            f"{quality.confidence}"
        )

        print(
            f"ITF Range High: "
            f"{context.itf.range_state.range_high}"
        )

        print(
            f"ITF Range Low: "
            f"{context.itf.range_state.range_low}"
        )

        print(
            f"ITF Swept High: "
            f"{context.itf.liquidity.swept_high}"
        )

        print(
            f"ITF Bearish Shift: "
            f"{context.itf.structure.bearish_shift}"
        )

        print(
            f"ITF Bearish Setup: "
            f"{context.itf.setup.bearish_setup}"
        )

        print(
            f"ITF Bullish FVG: "
            f"{context.itf.fvg.bullish_fvg}"
        )

        print(
            f"ITF Bearish FVG: "
            f"{context.itf.fvg.bearish_fvg}"
        )

        print()

        print(
            f"LTF Bearish Bias: "
            f"{context.ltf.bias.bearish}"
        )

        print(
            f"Aligned Bearish: "
            f"{context.aligned_bearish}"
        )

    print()

    print("=" * 60)

    print("CONFLUENCE TELEMETRY")

    print("=" * 60)

    replay_steps_15m = replay_timeframe(
        timeframe=ITF,
        candles=itf_candles,
    )

    telemetry = build_confluence_telemetry(
        replay_steps=replay_steps_15m,
    )

    print(
        f"Total Steps: "
        f"{telemetry.total_steps}"
    )

    print(
        f"Sweep Count: "
        f"{telemetry.sweep_count}"
    )

    print(
        f"Bearish Shift Count: "
        f"{telemetry.bearish_shift_count}"
    )

    print(
        f"Bearish Setup Count: "
        f"{telemetry.bearish_setup_count}"
    )

    print(
        f"Aligned Bearish Count: "
        f"{telemetry.aligned_bearish_count}"
    )

    print()

    print("=" * 60)

    print("BACKTEST RESULTS")

    print("=" * 60)

    backtest_result = (
        backtest_bearish_setups(
            replay_steps=replay_steps_15m,
            candles=itf_candles,
        )
    )

    print(
        f"Total Setups: "
        f"{backtest_result.total_setups}"
    )

    print(
        f"Successful Setups: "
        f"{backtest_result.successful_setups}"
    )

    print(
        f"Failed Setups: "
        f"{backtest_result.failed_setups}"
    )

    if backtest_result.total_setups > 0:
        winrate = (
            backtest_result.successful_setups
            / backtest_result.total_setups
        ) * 100

        print(
            f"Directional Winrate: "
            f"{winrate:.2f}%"
        )


if __name__ == "__main__":
    run_sandbox()