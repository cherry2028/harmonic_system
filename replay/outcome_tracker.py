"""replay/outcome_tracker.py — deterministic outcome evaluation for replayed signals.

Evaluates a TieredSignal against ONLY future bars (post-entry) to determine
which level (target1/2/3 or stop) was hit first. No lookahead. No mutation.

Worst-case rule for simultaneous hit:
    If a single candle touches BOTH stop and any target,
    the outcome is STOP_LOSS (worst case). This is documented,
    deterministic, and conservative.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from signals.signal import TieredSignal


@dataclass(frozen=True)
class SignalOutcome:
    """Immutable outcome of one TieredSignal evaluated against future bars."""
    outcome_type: str           # "target1", "target2", "target3", "stop", "timeout"
    hit_target1: bool
    hit_target2: bool
    hit_target3: bool
    hit_stop: bool
    bars_to_resolution: int    # -1 if empty future_bars

    def __post_init__(self):
        if self.outcome_type not in {"target1", "target2", "target3", "stop", "timeout"}:
            raise ValueError(f"Invalid outcome_type: {self.outcome_type}")
        if self.bars_to_resolution < -1:
            raise ValueError(f"bars_to_resolution must be >= -1, got {self.bars_to_resolution}")


class OutcomeTracker:
    """Deterministic outcome tracker. Forward-only. No lookahead."""

    WORST_CASE_RULE: str = (
        "If a single candle touches BOTH stop and any target level, "
        "the outcome is STOP_LOSS (worst case)."
    )

    def __init__(self, max_bars_forward: int = 50) -> None:
        self._max_bars_forward = max(max_bars_forward, 1)

    def track(
        self,
        signal: TieredSignal,
        future_bars: pd.DataFrame,
        max_bars_forward: Optional[int] = None,
    ) -> SignalOutcome:
        """Evaluate signal against future bars only.

        Parameters
        ----------
        signal : TieredSignal
            The signal to evaluate. Must have entry, stop, and targets.
        future_bars : pd.DataFrame
            OHLCV DataFrame containing ONLY bars after signal entry.
            Must have columns: high, low.
        max_bars_forward : int, optional
            Override default look-ahead limit. Bars beyond this are ignored.

        Returns
        -------
        SignalOutcome
            Immutable outcome record.
        """
        if future_bars is None or getattr(future_bars, "empty", True):
            return SignalOutcome(
                outcome_type="timeout",
                hit_target1=False,
                hit_target2=False,
                hit_target3=False,
                hit_stop=False,
                bars_to_resolution=-1,
            )

        limit = max_bars_forward if max_bars_forward is not None else self._max_bars_forward
        bars = future_bars.iloc[:limit].copy()
        if bars.empty:
            return SignalOutcome(
                outcome_type="timeout",
                hit_target1=False,
                hit_target2=False,
                hit_target3=False,
                hit_stop=False,
                bars_to_resolution=-1,
            )

        entry = signal.entry
        stop = signal.stop
        t1 = signal.target1
        t2 = signal.target2
        t3 = signal.target3

        # Determine direction from entry vs stop
        is_long = entry > stop

        for bar_idx, (_, row) in enumerate(bars.iterrows(), start=1):
            high = float(row["high"])
            low = float(row["low"])

            # Check stop hit (worst-case priority)
            stop_hit = (is_long and low <= stop) or (not is_long and high >= stop)

            # Check targets
            t1_hit = (is_long and high >= t1) or (not is_long and low <= t1)
            t2_hit = (is_long and high >= t2) or (not is_long and low <= t2)
            t3_hit = (is_long and high >= t3) or (not is_long and low <= t3)

            # Worst-case rule: simultaneous stop + target → stop
            if stop_hit:
                return SignalOutcome(
                    outcome_type="stop",
                    hit_target1=t1_hit,
                    hit_target2=t2_hit,
                    hit_target3=t3_hit,
                    hit_stop=True,
                    bars_to_resolution=bar_idx,
                )

            # Targets in order (closest first)
            if t1_hit:
                return SignalOutcome(
                    outcome_type="target1",
                    hit_target1=True,
                    hit_target2=t2_hit,
                    hit_target3=t3_hit,
                    hit_stop=False,
                    bars_to_resolution=bar_idx,
                )
            if t2_hit:
                return SignalOutcome(
                    outcome_type="target2",
                    hit_target1=t1_hit,
                    hit_target2=True,
                    hit_target3=t3_hit,
                    hit_stop=False,
                    bars_to_resolution=bar_idx,
                )
            if t3_hit:
                return SignalOutcome(
                    outcome_type="target3",
                    hit_target1=t1_hit,
                    hit_target2=t2_hit,
                    hit_target3=True,
                    hit_stop=False,
                    bars_to_resolution=bar_idx,
                )

        # No level hit within evaluated bars
        return SignalOutcome(
            outcome_type="timeout",
            hit_target1=False,
            hit_target2=False,
            hit_target3=False,
            hit_stop=False,
            bars_to_resolution=len(bars),
        )