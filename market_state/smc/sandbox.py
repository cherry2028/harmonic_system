from market_state.smc.context_builder import (
    build_timeframe_context,
)
from market_state.smc.range_state import Candle
from market_state.smc.replay_engine import (
    replay_timeframe,
)


def run_sandbox():
    candles = [
        Candle(high=100, low=99, close=99.5),
        Candle(high=100.2, low=99.2, close=99.8),
        Candle(high=100.1, low=99.1, close=99.7),
        Candle(high=100.3, low=99.3, close=99.9),
        Candle(high=101.5, low=99.4, close=99.6),
        Candle(high=99.8, low=97.5, close=97.8),
    ]

    replay_steps = replay_timeframe(
        timeframe="15m",
        candles=candles,
    )

    for step in replay_steps:
        context = build_timeframe_context(
            timeframe="15m",
            candles=candles[:step.index],
        )

        print("=" * 50)

        print(
            f"STEP: {step.index}"
        )

        print(
            f"Bias bearish: "
            f"{context.bias.bearish}"
        )

        print(
            f"Range detected: "
            f"{context.range_state.is_ranging}"
        )

        print(
            f"Bearish setup: "
            f"{context.setup.bearish_setup}"
        )


if __name__ == "__main__":
    run_sandbox()