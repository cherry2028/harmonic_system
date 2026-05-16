import time
from pathlib import Path

import ccxt
import pandas as pd


exchange = ccxt.binance()


def fetch_historical_dataset(
    symbol: str,
    timeframe: str,
    start_date: str,
) -> None:
    since = exchange.parse8601(
        start_date
    )

    all_candles = []

    print()

    print("=" * 60)

    print(
        f"DOWNLOADING "
        f"{symbol} "
        f"{timeframe}"
    )

    print("=" * 60)

    while True:
        candles = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since,
            limit=1000,
        )

        if len(candles) == 0:
            break

        all_candles.extend(candles)

        last_timestamp = candles[-1][0]

        print(
            f"Fetched: {len(candles)} | "
            f"Total: {len(all_candles)}"
        )

        since = last_timestamp + 1

        time.sleep(
            exchange.rateLimit / 1000
        )

    df = pd.DataFrame(
        all_candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
    )

    df = df.drop_duplicates(
        subset="timestamp"
    )

    df = df.sort_values(
        "timestamp"
    )

    df = df.reset_index(
        drop=True
    )

    symbol_folder = (
        symbol.replace("/", "")
    )

    output_dir = Path(
        f"data/{symbol_folder}/{timeframe}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"clean_{symbol_folder.lower()}_{timeframe}.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print()

    print("=" * 60)

    print("DOWNLOAD COMPLETE")

    print("=" * 60)

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Total candles: {len(df)}"
    )


if __name__ == "__main__":
    fetch_historical_dataset(
        symbol="BTC/USDT",
        timeframe="5m",
        start_date="2021-01-01T00:00:00Z",
    )