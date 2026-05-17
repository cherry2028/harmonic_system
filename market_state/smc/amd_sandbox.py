from market_state.smc.amd import (
    detect_amd,
)
from market_state.smc.csv_loader import (
    load_csv_candles,
)

SYMBOL = "BTCUSDT"

TIMEFRAME = "15m"


def run():
    candles = load_csv_candles(
        (
            f"data/{SYMBOL}/"
            f"{TIMEFRAME}/"
            f"clean_"
            f"{SYMBOL.lower()}_"
            f"{TIMEFRAME}.csv"
        )
    )

    print("=" * 50)

    print("AMD DETECTION")

    print("=" * 50)

    for i in range(
        100,
        len(candles),
    ):
        current = candles[:i]

        amd = detect_amd(
            current
        )

        if amd.accumulation:
            latest = current[-1]

            print("-" * 50)

            print(
                f"Timestamp: "
                f"{latest.timestamp}"
            )

            print(
                f"Close: "
                f"{latest.close}"
            )

            print(
                f"Accumulation: "
                f"{amd.accumulation}"
            )


if __name__ == "__main__":
    run()