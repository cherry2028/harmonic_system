"""replay/bar_feeder.py — yields rolling OHLCV windows without lookahead."""
from __future__ import annotations
from typing import Iterator, Optional, Tuple
import pandas as pd


class BarFeeder:
    """Yields strictly forward-only rolling windows from historical OHLCV data.

    Each yielded window is a defensive copy. The caller cannot mutate the
    internal state of the feeder, and the feeder never leaks future bars.
    """

    def __init__(self, df: pd.DataFrame, window_size: int = 300) -> None:
        if df is None or df.empty:
            raise ValueError("BarFeeder requires non-empty DataFrame")
        if window_size < 2:
            raise ValueError("window_size must be >= 2")

        # Defensive copy on construction; sort to guarantee temporal order
        self._df = df.copy()
        self._df.sort_index(inplace=True)
        self._window_size = window_size

    def __iter__(self) -> Iterator[Tuple[pd.Timestamp, pd.DataFrame]]:
        total = len(self._df)
        for i in range(self._window_size, total + 1):
            # Slice [i-window_size, i) — right-exclusive, so the current bar
            # is the LAST bar in the window. No future bars included.
            window = self._df.iloc[i - self._window_size : i].copy()
            timestamp = self._df.index[i - 1]
            yield timestamp, window

    def __len__(self) -> int:
        return max(0, len(self._df) - self._window_size + 1)

    @property
    def window_size(self) -> int:
        return self._window_size


class ReplayDataFetcher:
    """Drop-in DataFetcher replacement for replay mode.

    The ReplayRunner advances the feeder and calls set_window() before each
    pipeline invocation. fetch() returns that window, ignoring symbol/timeframe.
    """

    def __init__(self) -> None:
        self._current_window: Optional[pd.DataFrame] = None

    def set_window(self, df: pd.DataFrame) -> None:
        """Set the window to be returned on the next fetch() call."""
        self._current_window = df.copy() if df is not None else None

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        bars: int = 300,
    ) -> Optional[pd.DataFrame]:
        """Return the current replay window.

        Parameters are accepted for API compatibility but ignored.
        """
        return self._current_window