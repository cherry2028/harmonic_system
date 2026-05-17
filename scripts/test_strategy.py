import numpy as np
import pandas as pd

from strategies.ema_rsi_atr.strategy import generate_signals


np.random.seed(42)

rows = 200

close = np.cumsum(np.random.randn(rows)) + 100

high = close + np.random.rand(rows)

low = close - np.random.rand(rows)

open_price = close + np.random.randn(rows) * 0.5


df = pd.DataFrame({
    "open": open_price,
    "high": high,
    "low": low,
    "close": close
})

result = generate_signals(df)

signals = result[result["long_signal"] == True]

print("\nTOTAL SIGNALS:")
print(len(signals))

print("\nLAST 5 SIGNALS:")
print(
    signals[
        ["close", "ema20", "ema50", "rsi", "atr"]
    ].tail()
)