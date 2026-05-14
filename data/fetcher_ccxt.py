import time
import ccxt
import pandas as pd


exchange = ccxt.binance()

symbol = "BTC/USDT"
timeframe = "1h"

# Binance launch period
since = exchange.parse8601("2017-08-17T00:00:00Z")

all_candles = []

print("Starting historical download...")

while True:

    candles = exchange.fetch_ohlcv(
        symbol,
        timeframe=timeframe,
        since=since,
        limit=1000
    )

    if len(candles) == 0:
        break

    all_candles.extend(candles)

    last_timestamp = candles[-1][0]

    print(
        f"Fetched {len(candles)} candles | "
        f"Total: {len(all_candles)}"
    )

    # move forward
    since = last_timestamp + 1

    # rate limit safety
    time.sleep(exchange.rateLimit / 1000)

print("Converting to DataFrame...")

df = pd.DataFrame(
    all_candles,
    columns=[
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]
)

# timestamp conversion
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    unit="ms"
)

# remove duplicates safely
df = df.drop_duplicates(
    subset="timestamp"
)

# sort properly
df = df.sort_values(
    "timestamp"
)

# reset index
df = df.reset_index(drop=True)

print(df.head())

# save clean dataset
output_path = "data/BTCUSDT/1h/clean_btc_1h.csv"

df.to_csv(
    output_path,
    index=False
)

print()
print("DONE")
print(f"Saved: {output_path}")
print(f"Total candles: {len(df)}")