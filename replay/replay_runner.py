"""replay/replay_runner.py — wraps ScanPipeline for historical replay."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, List, Optional

from pipeline import ScanPipeline, ScanResult
from signals.signal import TieredSignal
from replay.bar_feeder import BarFeeder, ReplayDataFetcher


@dataclass(frozen=True)
class ReplayRecord:
    """Immutable record of one replay step that produced a signal.

    One record is emitted per TieredSignal. The full ScanResult is preserved
    for downstream context (e.g., vector, gate result, timing).
    """
    bar_timestamp: str   # ISO 8601 — JSON-serializable
    scan_result: ScanResult
    tiered_signal: Optional[TieredSignal] = None

    @property
    def produced_signal(self) -> bool:
        return self.tiered_signal is not None


class ReplayRunner:
    """Replays ScanPipeline over historical windows one bar at a time.

    The pipeline must have been constructed with a ReplayDataFetcher.
    The runner advances the feeder, injects the window into the fetcher,
    and invokes scan_one() with dry_run=True.

    Records are yielded ONLY when the pipeline produces tiered signals.
    """

    def __init__(
        self,
        pipeline: ScanPipeline,
        feeder: BarFeeder,
        *,
        symbol: str = "REPLAY",
        timeframe: str = "REPLAY",
    ) -> None:
        self._pipeline = pipeline
        self._feeder = feeder
        self._symbol = symbol
        self._timeframe = timeframe

        fetcher = pipeline.components.get("data_fetcher")
        if not isinstance(fetcher, ReplayDataFetcher):
            raise TypeError(
                "ReplayRunner requires pipeline with ReplayDataFetcher. "
                f"Got: {type(fetcher).__name__}"
            )
        self._fetcher = fetcher

    def run(self) -> Iterator[ReplayRecord]:
        """Iterate over all bars, yielding records only when signals are produced."""
        for timestamp, window in self._feeder:
            self._fetcher.set_window(window)

            result = self._pipeline.scan_one(
                symbol=self._symbol,
                timeframe=self._timeframe,
                dry_run=True,   # never emit live alerts during replay
            )

            if result.tiered_signals:
                ts_iso = str(timestamp)
                for tiered in result.tiered_signals:
                    yield ReplayRecord(
                        bar_timestamp=ts_iso,
                        scan_result=result,
                        tiered_signal=tiered,
                    )

    def run_all(self) -> List[ReplayRecord]:
        """Collect all signal-producing records into a list."""
        return list(self.run())