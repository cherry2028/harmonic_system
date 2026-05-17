import requests
import pandas as pd

url = "https://public.coindcx.com/market_data/candlesticks"

params = {
    "pair": "B-BTC_USDT",
    "resolution": "1",
    "from": 1778889600,
    "to": 1779032564,
    "pcode": "f"
}

response = requests.get(url, params=params)

data = response.json()["data"]

df = pd.DataFrame(data)

# numeric conversion

df["open"] = pd.to_numeric(df["open"])
df["high"] = pd.to_numeric(df["high"])
df["low"] = pd.to_numeric(df["low"])
df["close"] = pd.to_numeric(df["close"])
df["volume"] = pd.to_numeric(df["volume"])

# EMA

df["ema20"] = df["close"].ewm(span=20).mean()

df["ema50"] = df["close"].ewm(span=50).mean()

# RSI

delta = df["close"].diff()

gain = delta.clip(lower=0)

loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()

avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

df["rsi"] = 100 - (100 / (1 + rs))

# signal

df["long_signal"] = (
    (df["ema20"] > df["ema50"]) &
    (df["rsi"] > 55)
)

print(df[[
    "close",
    "ema20",
    "ema50",
    "rsi",
    "long_signal"
]].tail())