import pandas as pd


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

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

    # ATR
    high_low = df["high"] - df["low"]

    high_close = (df["high"] - df["close"].shift()).abs()

    low_close = (df["low"] - df["close"].shift()).abs()

    tr = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    df["atr"] = tr.rolling(14).mean()

    # Long condition
    df["long_signal"] = (
        (df["ema20"] > df["ema50"]) &
        (df["rsi"] > 55) &
        (df["close"] > df["high"].shift(1))
    )

    return df