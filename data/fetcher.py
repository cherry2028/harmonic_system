# data/fetcher.py  — full replacement

import pandas as pd
import requests
from typing import Optional
from utils.logger import get_logger
from config.settings import CONFIG

logger = get_logger(__name__)


class BinanceFetcher:
    """
    Fetches OHLCV from Binance public REST API.
    No API key required for market data.
    Rate limit: 1200 requests/minute — we stay well under.
    """

    BASE_URL = "https://api.binance.com/api/v3/klines"

    INTERVAL_MAP = {
        "15m": "15m",
        "1h":  "1h",
        "4h":  "4h",
        "1d":  "1d",
    }

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        bars: int = None,
    ) -> Optional[pd.DataFrame]:

        bars   = bars or CONFIG.data.lookback_candles
        interval = self.INTERVAL_MAP.get(timeframe)

        if not interval:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return None

        try:
            params = {
                "symbol":   symbol.upper(),
                "interval": interval,
                "limit":    min(bars, 1000),   # Binance hard cap = 1000
            }
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            raw = resp.json()

            df = pd.DataFrame(raw, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])

            df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            logger.info(f"Fetched {len(df)} bars | {symbol} {timeframe}")
            return df

        except requests.RequestException as e:
            logger.error(f"Binance fetch error | {symbol} {timeframe} | {e}")
            return None


# Default fetcher — swap class here to change provider globally
DataFetcher = BinanceFetcher