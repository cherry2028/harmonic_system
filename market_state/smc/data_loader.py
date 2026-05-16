from typing import List

import ccxt

from market_state.smc.range_state import Candle


def load_ohlcv(
    symbol: str,
    timeframe: str,
    limit: int = 100,
) -> List[Candle]:
    exchange = ccxt.binance()

    ohlcv = exchange.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )

    candles: List[Candle] = []

    for row in ohlcv:
        candles.append(
            Candle(
                timestamp=row[0],
                open=row[1],
                high=row[2],
                low=row[3],
                close=row[4],
            )
        )

    return candles