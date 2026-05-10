import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

class DataValidator:
    """
    Validates OHLCV data quality before pattern detection.
    Garbage in = garbage patterns. This is your first defense.
    """

    MIN_BARS = 50

    def validate(self, df: pd.DataFrame, symbol: str = "") -> bool:
        if df is None or df.empty:
            logger.warning(f"[{symbol}] DataFrame is None or empty")
            return False

        if len(df) < self.MIN_BARS:
            logger.warning(f"[{symbol}] Insufficient bars: {len(df)} < {self.MIN_BARS}")
            return False

        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            logger.error(f"[{symbol}] Missing columns: {required_cols - set(df.columns)}")
            return False

        # Check for OHLC integrity
        invalid_candles = df[df["high"] < df["low"]]
        if not invalid_candles.empty:
            logger.warning(f"[{symbol}] {len(invalid_candles)} candles with high < low")

        # Check for excessive NaN
        nan_pct = df.isnull().mean().max()
        if nan_pct > 0.05:
            logger.warning(f"[{symbol}] High NaN ratio: {nan_pct:.1%}")
            return False

        return True