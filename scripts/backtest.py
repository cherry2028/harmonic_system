import requests
import pandas as pd

# =========================
# SETTINGS
# =========================

INITIAL_CAPITAL = 100000

capital = INITIAL_CAPITAL

RISK_PER_TRADE = 0.01

FEES_PERCENT = 0.0005

# =========================
# FETCH DATA
# =========================

url = "https://public.coindcx.com/market_data/candlesticks"

FROM_DATE = 1776441600
TO_DATE = 1779032564

# =========================
# PARAMS
# =========================

params_5m = {
    "pair": "B-BTC_USDT",
    "resolution": "5",
    "from": FROM_DATE,
    "to": TO_DATE,
    "pcode": "f"
}

params_15m = {
    "pair": "B-BTC_USDT",
    "resolution": "15",
    "from": FROM_DATE,
    "to": TO_DATE,
    "pcode": "f"
}

params_1h = {
    "pair": "B-BTC_USDT",
    "resolution": "60",
    "from": FROM_DATE,
    "to": TO_DATE,
    "pcode": "f"
}

params_4h = {
    "pair": "B-BTC_USDT",
    "resolution": "240",
    "from": FROM_DATE,
    "to": TO_DATE,
    "pcode": "f"
}

# =========================
# API REQUESTS
# =========================

response_5m = requests.get(url, params=params_5m)

response_15m = requests.get(url, params=params_15m)

response_1h = requests.get(url, params=params_1h)

response_4h = requests.get(url, params=params_4h)

# =========================
# DATA
# =========================

data_5m = response_5m.json()["data"]

data_15m = response_15m.json()["data"]

data_1h = response_1h.json()["data"]

data_4h = response_4h.json()["data"]

# =========================
# DATAFRAMES
# =========================

df = pd.DataFrame(data_5m)

df_15m = pd.DataFrame(data_15m)

df_1h = pd.DataFrame(data_1h)

df_4h = pd.DataFrame(data_4h)

# =========================
# CLEAN DATA
# =========================

numeric_cols = [
    "open",
    "high",
    "low",
    "close",
    "volume"
]

for col in numeric_cols:

    df[col] = pd.to_numeric(df[col])

    df_15m[col] = pd.to_numeric(df_15m[col])

    df_1h[col] = pd.to_numeric(df_1h[col])

    df_4h[col] = pd.to_numeric(df_4h[col])

# =========================
# TIME
# =========================

df["datetime"] = pd.to_datetime(
    df["time"],
    unit="ms"
)

df["hour"] = df["datetime"].dt.hour

# =========================
# SESSION FILTER
# =========================

df["session_active"] = (
    ((df["hour"] >= 7) & (df["hour"] <= 10)) |
    ((df["hour"] >= 13) & (df["hour"] <= 16))
)

# =========================
# EMA
# =========================

for frame in [df, df_15m, df_1h, df_4h]:

    frame["ema5"] = (
        frame["close"]
        .ewm(span=5)
        .mean()
    )

    frame["ema9"] = (
        frame["close"]
        .ewm(span=9)
        .mean()
    )

    frame["ema21"] = (
        frame["close"]
        .ewm(span=21)
        .mean()
    )

# =========================
# RSI
# =========================

delta = df["close"].diff()

gain = delta.clip(lower=0)

loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()

avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

df["rsi"] = 100 - (100 / (1 + rs))

# =========================
# ADX
# =========================

df["previous_high"] = df["high"].shift(1)

df["previous_low"] = df["low"].shift(1)

df["previous_close"] = df["close"].shift(1)

df["tr1"] = (
    df["high"] - df["low"]
)

df["tr2"] = abs(
    df["high"] - df["previous_close"]
)

df["tr3"] = abs(
    df["low"] - df["previous_close"]
)

df["tr"] = df[
    ["tr1", "tr2", "tr3"]
].max(axis=1)

df["+dm"] = (
    df["high"] - df["previous_high"]
)

df["-dm"] = (
    df["previous_low"] - df["low"]
)

df["+dm"] = df["+dm"].where(
    (df["+dm"] > df["-dm"]) &
    (df["+dm"] > 0),
    0
)

df["-dm"] = df["-dm"].where(
    (df["-dm"] > df["+dm"]) &
    (df["-dm"] > 0),
    0
)

atr = df["tr"].rolling(14).mean()

df["+di"] = (
    100 *
    (
        df["+dm"]
        .rolling(14)
        .mean() / atr
    )
)

df["-di"] = (
    100 *
    (
        df["-dm"]
        .rolling(14)
        .mean() / atr
    )
)

df["dx"] = (
    abs(
        df["+di"] - df["-di"]
    ) /
    (
        df["+di"] + df["-di"]
    )
) * 100

df["adx"] = (
    df["dx"]
    .rolling(14)
    .mean()
)

# =========================
# RESULTS VARIABLES
# =========================

position = None

wins = 0

losses = 0

total_trades = 0

trade_logs = []

# =========================
# TRADE ENGINE
# =========================

for i in range(50, len(df)):

    row = df.iloc[i]

    current_time = row["time"]

    current_15m = df_15m[
        df_15m["time"] <= current_time
    ].iloc[-1]

    current_1h = df_1h[
        df_1h["time"] <= current_time
    ].iloc[-1]

    current_4h = df_4h[
        df_4h["time"] <= current_time
    ].iloc[-1]

    # =====================
    # MTF BIAS
    # =====================

    bullish = (
        current_4h["ema5"] >
        current_4h["ema9"] >
        current_4h["ema21"]
    ) and (
        current_1h["ema5"] >
        current_1h["ema9"] >
        current_1h["ema21"]
    ) and (
        current_15m["ema5"] >
        current_15m["ema9"] >
        current_15m["ema21"]
    )

    bearish = (
        current_4h["ema5"] <
        current_4h["ema9"] <
        current_4h["ema21"]
    ) and (
        current_1h["ema5"] <
        current_1h["ema9"] <
        current_1h["ema21"]
    ) and (
        current_15m["ema5"] <
        current_15m["ema9"] <
        current_15m["ema21"]
    )

    long_signal = (
        bullish and
        row["session_active"] and
        row["adx"] > 25 and
        row["rsi"] > 55 and
        row["ema5"] > row["ema9"]
    )

    short_signal = (
        bearish and
        row["session_active"] and
        row["adx"] > 25 and
        row["rsi"] < 45 and
        row["ema5"] < row["ema9"]
    )

    # =====================
    # ENTRY
    # =====================

    if position is None:

        if long_signal:

            entry_price = row["close"]

            stoploss = row["low"]

            risk_points = (
                entry_price -
                stoploss
            )

            if risk_points <= 0:
                continue

            risk_amount = (
                capital *
                RISK_PER_TRADE
            )

            quantity = (
                risk_amount /
                risk_points
            )

            position = "LONG"

            entry_index = i

            total_trades += 1

        elif short_signal:

            entry_price = row["close"]

            stoploss = row["high"]

            risk_points = (
                stoploss -
                entry_price
            )

            if risk_points <= 0:
                continue

            risk_amount = (
                capital *
                RISK_PER_TRADE
            )

            quantity = (
                risk_amount /
                risk_points
            )

            position = "SHORT"

            entry_index = i

            total_trades += 1

    # =====================
    # LONG MANAGEMENT
    # =====================

    elif position == "LONG":

        hold_candles = (
            i - entry_index
        )

        if row["low"] <= stoploss:

            exit_price = stoploss

            pnl_points = (
                exit_price -
                entry_price
            )

            gross_pnl = (
                pnl_points *
                quantity
            )

            fees = (
                (
                    entry_price +
                    exit_price
                ) *
                quantity *
                FEES_PERCENT
            )

            net_pnl = (
                gross_pnl -
                fees
            )

            capital += net_pnl

            losses += 1

            trade_logs.append({
                "type": "LONG",
                "entry": round(entry_price, 2),
                "exit": round(exit_price, 2),
                "qty": round(quantity, 4),
                "net_pnl": round(net_pnl, 2),
                "capital": round(capital, 2),
                "hold_candles": hold_candles,
                "reason": "STOPLOSS"
            })

            position = None

        elif row["ema5"] < row["ema9"]:

            exit_price = row["close"]

            pnl_points = (
                exit_price -
                entry_price
            )

            gross_pnl = (
                pnl_points *
                quantity
            )

            fees = (
                (
                    entry_price +
                    exit_price
                ) *
                quantity *
                FEES_PERCENT
            )

            net_pnl = (
                gross_pnl -
                fees
            )

            capital += net_pnl

            if net_pnl > 0:
                wins += 1
            else:
                losses += 1

            trade_logs.append({
                "type": "LONG",
                "entry": round(entry_price, 2),
                "exit": round(exit_price, 2),
                "qty": round(quantity, 4),
                "net_pnl": round(net_pnl, 2),
                "capital": round(capital, 2),
                "hold_candles": hold_candles,
                "reason": "EMA_EXIT"
            })

            position = None

    # =====================
    # SHORT MANAGEMENT
    # =====================

    elif position == "SHORT":

        hold_candles = (
            i - entry_index
        )

        if row["high"] >= stoploss:

            exit_price = stoploss

            pnl_points = (
                entry_price -
                exit_price
            )

            gross_pnl = (
                pnl_points *
                quantity
            )

            fees = (
                (
                    entry_price +
                    exit_price
                ) *
                quantity *
                FEES_PERCENT
            )

            net_pnl = (
                gross_pnl -
                fees
            )

            capital += net_pnl

            losses += 1

            trade_logs.append({
                "type": "SHORT",
                "entry": round(entry_price, 2),
                "exit": round(exit_price, 2),
                "qty": round(quantity, 4),
                "net_pnl": round(net_pnl, 2),
                "capital": round(capital, 2),
                "hold_candles": hold_candles,
                "reason": "STOPLOSS"
            })

            position = None

        elif row["ema5"] > row["ema9"]:

            exit_price = row["close"]

            pnl_points = (
                entry_price -
                exit_price
            )

            gross_pnl = (
                pnl_points *
                quantity
            )

            fees = (
                (
                    entry_price +
                    exit_price
                ) *
                quantity *
                FEES_PERCENT
            )

            net_pnl = (
                gross_pnl -
                fees
            )

            capital += net_pnl

            if net_pnl > 0:
                wins += 1
            else:
                losses += 1

            trade_logs.append({
                "type": "SHORT",
                "entry": round(entry_price, 2),
                "exit": round(exit_price, 2),
                "qty": round(quantity, 4),
                "net_pnl": round(net_pnl, 2),
                "capital": round(capital, 2),
                "hold_candles": hold_candles,
                "reason": "EMA_EXIT"
            })

            position = None

# =========================
# FINAL RESULTS
# =========================

winrate = 0

if total_trades > 0:

    winrate = (
        wins /
        total_trades
    ) * 100

print("\n========== REALISTIC BACKTEST ==========\n")

print(f"INITIAL CAPITAL: ₹{INITIAL_CAPITAL}")

print(f"FINAL CAPITAL: ₹{capital:.2f}")

print(f"NET PROFIT: ₹{capital - INITIAL_CAPITAL:.2f}")

print(f"TOTAL TRADES: {total_trades}")

print(f"WINS: {wins}")

print(f"LOSSES: {losses}")

print(f"WINRATE: {winrate:.2f}%")

print("\n========== LAST 20 TRADES ==========\n")

for trade in trade_logs[-20:]:

    print(trade)