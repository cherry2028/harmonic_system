from pathlib import Path

import pandas as pd

from market_state.smc.range_state import (
    Candle,
)


def load_csv_candles(
    path: str,
) -> list[Candle]:
    csv_path = Path(path)

    df = pd.read_csv(csv_path)

    candles: list[Candle] = []

    for row in df.itertuples():
        timestamp = int(
            pd.Timestamp(
                row.timestamp
            ).timestamp()
            * 1000
        )

        candles.append(
            Candle(
                timestamp=timestamp,

                open=float(row.open),

                high=float(row.high),

                low=float(row.low),

                close=float(row.close),
            )
        )

    return candles