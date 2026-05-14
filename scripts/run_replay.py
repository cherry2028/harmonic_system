#!/usr/bin/env python3
"""scripts/run_replay.py — deterministic historical replay entrypoint.

Usage
-----
    python scripts/run_replay.py --csv data/BTCUSDT_1h.csv --symbol BTCUSDT --tf 1h
    python scripts/run_replay.py --csv data/ETHUSDT_4h.csv --symbol ETHUSDT --tf 4h

CSV format expected
-------------------
    One of:
      - First column: datetime-parseable timestamp, named "timestamp" or
        "open_time" (auto-detected). Remaining columns: open, high, low,
        close, volume.
      - OR all six columns present with timestamp as a regular column
        (script will set it as the index).

    All OHLCV columns must be numeric. No other columns are required.

Replay methodology guarantees
------------------------------
    - Forward-only: BarFeeder yields windows [i-window_size, i). Bar i
      (the "current" bar) is the last bar in each window. Bars i+1..N are
      never visible to the pipeline at bar i.
    - dry_run=True: no Telegram messages, no live counter increments.
    - OutcomeTracker evaluates ONLY bars strictly after the signal bar.
      The signal bar itself is excluded from future_bars.
    - Statistics are computed twice: once excluding timeouts (resolved-only,
      an optimistic ceiling) and once treating timeouts as losses
      (all-entered, a pessimistic floor). Both are printed. The truth
      is somewhere between them.

Research integrity constraints
--------------------------------
    - No parameter tuning occurs here.
    - No thresholds are adjusted based on output.
    - This script is a read-only observer of the engine as configured.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time
from collections import Counter
from typing import List, Optional, Tuple

# Add project root to path so we can import engine modules
_ROOT = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_ROOT))

import pandas as pd

from delivery.telegram_formatter import TelegramFormatter
from market_state.engine import MarketStateEngine
from patterns.patterns.harmonic_detector import HarmonicDetector
from patterns.patterns.swing_detector import AdaptiveSwingDetector
from pipeline import ScanPipeline
from replay.bar_feeder import BarFeeder, ReplayDataFetcher
from replay.outcome_tracker import OutcomeTracker, SignalOutcome
from replay.replay_runner import ReplayRecord, ReplayRunner
from replay.statistics import OutcomeRecord, ReplayStatistics, StatisticsReport
from scoring.pattern_scorer import PatternScorer
from signals.daily_counter import DailyCounter
from signals.gate import HostileMarketGate
from signals.tier import SignalTier
from telemetry.logger import TelemetryLogger


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# How many bars forward the tracker looks for a resolution.
# 50 bars on 1h = ~2 trading days. 50 bars on 4h = ~8 trading days.
# Changing this number changes what "timeout" means — document it explicitly.
MAX_BARS_FORWARD: int = 50

# BarFeeder window: minimum context bars fed to the engine on each step.
# Must be >= engine's minimum lookback (300 is the pipeline default).
WINDOW_SIZE: int = 300

# Minimum sample size below which we print a reliability warning.
# Below 30: confidence intervals are too wide to interpret.
# Below 80: segmented statistics (per-tier, per-pattern) are unreliable.
MIN_SIGNALS_FOR_STATS: int = 30
MIN_SIGNALS_FOR_SEGMENTS: int = 80

# Wilson score confidence interval z-value (95% confidence).
_Z95: float = 1.96

# Supported symbols and timeframes (no others are accepted — prevents
# accidental testing on illiquid or non-crypto data).
SUPPORTED_SYMBOLS: frozenset = frozenset({"BTCUSDT", "ETHUSDT"})
SUPPORTED_TIMEFRAMES: frozenset = frozenset({"1h", "4h"})


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: CSV loading
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV CSV into a DatetimeIndex DataFrame.

    Handles two common column layouts:
      Layout A: timestamp (or open_time) + open/high/low/close/volume
      Layout B: all columns present, auto-detect timestamp

    Validates:
      - Required OHLCV columns present
      - Index is DatetimeIndex after loading
      - No duplicate timestamps
      - Sorted chronologically (sort is applied if not already sorted)

    Raises
    ------
    SystemExit
        On any unrecoverable loading error. We prefer a clean abort over
        a partially-loaded dataset that could produce misleading results.
    """
    p = pathlib.Path(path)
    if not p.exists():
        _abort(f"CSV file not found: {path}")
    if p.stat().st_size == 0:
        _abort(f"CSV file is empty: {path}")

    _print_step(f"Loading CSV: {p.name}")

    try:
        # Peek at headers to decide how to load
        with open(p, encoding="utf-8") as f:
            header_line = f.readline().strip()
        columns = [c.strip().lower() for c in header_line.split(",")]

        # Determine which column is the timestamp
        ts_candidates = {"timestamp", "open_time", "date", "datetime", "time"}
        ts_col = next((c for c in columns if c in ts_candidates), None)

        if ts_col is not None:
            df = pd.read_csv(
                p,
                skiprows=1,
                low_memory=False
            )

            df["Date"] = pd.to_datetime(
                df["Date"],
                format="mixed",
                dayfirst=True
            )
            df = df.sort_values("Date")

            df = df.drop_duplicates(
                subset="Date",
                keep="last",
            )
            df = df.set_index("Date")
            df.index.name = "timestamp"
        else:
            # No obvious timestamp column — try loading as-is and see if
            # the first column parses as dates
            df = pd.read_csv(
                p,
                skiprows=1,
                low_memory=False
            )

            df["Date"] = pd.to_datetime(
                df["Date"],
                format="mixed",
                dayfirst=True
            )
            df = df.sort_values("Date")
            df = df.drop_duplicates(
                subset="Date",
                keep="last"
            )
            df = df.set_index("Date")
            df.index.name = "timestamp"

    except Exception as e:
        _abort(f"Failed to read CSV: {e}")

    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    if "volume btc" in df.columns:
        df["volume"] = df["volume btc"]

    # Validate required columns
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        _abort(
            f"CSV is missing required columns: {sorted(missing)}\n"
            f"  Found columns: {sorted(df.columns)}\n"
            f"  Expected: open, high, low, close, volume"
        )

    # Keep only the columns we need — drop anything extra silently
    df = df[["open", "high", "low", "close", "volume"]].copy()

    # Validate index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        _abort(
            f"Timestamp column could not be parsed as dates.\n"
            f"  Index dtype: {df.index.dtype}\n"
            f"  First value: {df.index[0]!r}\n"
            f"  Ensure your CSV has ISO-format timestamps."
        )

    # Convert all OHLCV columns to float
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Abort on excessive NaN (> 1% of any column is a data quality problem)
    nan_pct = df.isnull().mean()
    bad_cols = nan_pct[nan_pct > 0.01]
    if not bad_cols.empty:
        _abort(
            f"NaN values exceed 1% in columns: {bad_cols.to_dict()}\n"
            f"  Repair the CSV before running replay."
        )

    # Drop any rows with NaN (protective — should be empty after above check)
    before_drop = len(df)
    df.dropna(inplace=True)
    dropped = before_drop - len(df)
    if dropped > 0:
        print(f"  ⚠  Dropped {dropped} rows with NaN values.")

    # Sort chronologically — BarFeeder requires sorted order
    df.sort_index(inplace=True)

    # Reject duplicate timestamps — these indicate malformed data
    dups = df.index.duplicated().sum()
    if dups > 0:
        _abort(
            f"CSV contains {dups} duplicate timestamps.\n"
            f"  Deduplicate before running replay."
        )

    # Sanity-check OHLC integrity: high must be >= low on every bar
    bad_candles = (df["high"] < df["low"]).sum()
    if bad_candles > 0:
        print(f"  ⚠  {bad_candles} candles have high < low. Data may be malformed.")

    _print_ok(
        f"Loaded {len(df):,} bars | "
        f"{df.index[0].date()} → {df.index[-1].date()}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Pipeline construction for replay
# ─────────────────────────────────────────────────────────────────────────────

def build_replay_pipeline() -> Tuple[ScanPipeline, ReplayDataFetcher]:
    """Construct a ScanPipeline wired for replay.

    Key differences from a live pipeline:
      - ReplayDataFetcher: ignores symbol/timeframe arguments, returns
        whatever window the ReplayRunner injected via set_window().
      - dry_run=True on TelegramFormatter: no HTTP calls, no alert delivery.
      - DailyCounter points to a temp-like location (/tmp/replay_counters)
        so it never contaminates production counter state.
      - signal_presentation=None: presentation layer is not needed for
        replay; signals are evaluated by OutcomeTracker, not displayed.

    Returns
    -------
    (pipeline, fetcher)
        The pipeline is ready for ReplayRunner. The fetcher is the same
        ReplayDataFetcher instance that the pipeline holds internally.
    """
    fetcher = ReplayDataFetcher()

    # Isolated counter directory: replay never touches production counters
    replay_counter_dir = pathlib.Path("/tmp/harmonic_replay_counters")
    replay_counter_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ScanPipeline(
        data_fetcher=fetcher,
        market_state_engine=MarketStateEngine(),
        swing_detector=AdaptiveSwingDetector(),
        harmonic_detector=HarmonicDetector(),
        pattern_scorer=PatternScorer(),
        hostile_gate=HostileMarketGate(),
        signal_tier=SignalTier(),
        daily_counter=DailyCounter(counter_dir=replay_counter_dir),
        telemetry=TelemetryLogger(),
        signal_presentation=None,         # not needed for replay
        telegram_formatter=TelegramFormatter(dry_run=True),  # no HTTP
    )
    return pipeline, fetcher


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Replay execution
# ─────────────────────────────────────────────────────────────────────────────

def run_replay(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> List[ReplayRecord]:
    """Run the replay over all windows and collect signal records.

    Forward-only guarantee: BarFeeder yields window [i-WINDOW_SIZE, i).
    The current bar is always the last bar in the window.
    Future bars are never included. This is enforced by BarFeeder's
    right-exclusive slice: df.iloc[i-window_size : i].

    Parameters
    ----------
    df : pd.DataFrame
        Full historical OHLCV data. NOT sliced here — BarFeeder does all
        window construction internally. Do not pre-slice.
    symbol : str
        Symbol label (for context in scan results, not for fetching).
    timeframe : str
        Timeframe label (for context only).

    Returns
    -------
    List[ReplayRecord]
        One record per TieredSignal produced during replay.
        Records with no signal are not stored (by ReplayRunner design).
    """
    if len(df) < WINDOW_SIZE + 1:
        _abort(
            f"CSV has only {len(df)} bars, but WINDOW_SIZE={WINDOW_SIZE} "
            f"requires at least {WINDOW_SIZE + 1} bars to produce any signals.\n"
            f"  Load a longer dataset."
        )

    pipeline, _fetcher = build_replay_pipeline()
    feeder = BarFeeder(df, window_size=WINDOW_SIZE)
    runner = ReplayRunner(pipeline, feeder, symbol=symbol, timeframe=timeframe)

    total_windows = len(feeder)
    _print_step(
        f"Running replay: {total_windows:,} windows "
        f"({WINDOW_SIZE}-bar context, 1 bar step)"
    )

    t0 = time.perf_counter()
    records = runner.run_all()
    elapsed = time.perf_counter() - t0

    _print_ok(
        f"Replay complete: {len(records)} signals in {elapsed:.1f}s "
        f"({total_windows / elapsed:.0f} windows/sec)"
    )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Outcome tracking
# ─────────────────────────────────────────────────────────────────────────────

def track_outcomes(
    records: List[ReplayRecord],
    df: pd.DataFrame,
) -> List[OutcomeRecord]:
    """Evaluate each signal against strictly future bars only.

    Lookahead protection: for a signal at bar index i, future_bars is
    df.iloc[i+1 : i+1+MAX_BARS_FORWARD]. Bar i (the signal bar) is
    excluded — it is the bar on which the D-point was confirmed.
    The first evaluation bar is bar i+1, which is the first bar where
    entry could realistically be filled (next-open execution model).

    Signals near the end of the dataset naturally have fewer future bars.
    These are tracked with partial windows — the OutcomeTracker handles
    this correctly, returning "timeout" if no level is hit within the
    available bars. This is not a bug; it is an honest representation
    of the data boundary. Timeouts from data-boundary signals and
    timeouts from genuine non-resolution are not distinguished — both
    are counted the same way in the summary.

    Parameters
    ----------
    records : List[ReplayRecord]
        Signal records from run_replay().
    df : pd.DataFrame
        The same full DataFrame used for replay. Must not be modified
        between replay and outcome tracking.

    Returns
    -------
    List[OutcomeRecord]
        Paired (ReplayRecord, SignalOutcome) for every signal.
    """
    if not records:
        return []

    _print_step(f"Tracking outcomes: {len(records)} signals")

    tracker = OutcomeTracker(max_bars_forward=MAX_BARS_FORWARD)
    outcome_records: List[OutcomeRecord] = []

    # Build a timestamp → integer index lookup once, not per signal.
    # Using get_loc() in a loop over thousands of signals is slow.
    ts_to_loc = {ts: i for i, ts in enumerate(df.index)}

    missing_ts = 0
    for rec in records:
        # Recover the bar position from the ISO timestamp stored in ReplayRecord
        signal_ts = pd.Timestamp(rec.bar_timestamp)
        bar_loc = ts_to_loc.get(signal_ts)

        if bar_loc is None:
            # This should never happen if df is unchanged between phases,
            # but guard defensively rather than crashing.
            missing_ts += 1
            continue

        # Strict future-only slice: bar_loc is the signal bar (excluded),
        # bar_loc+1 onwards is the evaluation window.
        future_bars = df.iloc[bar_loc + 1 : bar_loc + 1 + MAX_BARS_FORWARD]

        outcome = tracker.track(rec.tiered_signal, future_bars)
        outcome_records.append(OutcomeRecord(replay=rec, outcome=outcome))

    if missing_ts > 0:
        print(f"  ⚠  {missing_ts} signals had unresolvable timestamps — skipped.")

    # Count outcome types for the progress line
    type_counts = Counter(r.outcome.outcome_type for r in outcome_records)
    _print_ok(
        f"Outcomes: "
        f"target1={type_counts.get('target1', 0)}  "
        f"target2={type_counts.get('target2', 0)}  "
        f"target3={type_counts.get('target3', 0)}  "
        f"stop={type_counts.get('stop', 0)}  "
        f"timeout={type_counts.get('timeout', 0)}"
    )
    return outcome_records


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(
    outcome_records: List[OutcomeRecord],
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
) -> None:
    """Print a complete, honest research summary to stdout.

    Structure
    ---------
    1. Replay metadata (dataset bounds, bars, windows)
    2. Signal inventory (total, resolved, timeouts with percentage)
    3. Resolved-only statistics (optimistic ceiling — excludes timeouts)
    4. All-entered statistics (conservative floor — timeouts as losses)
    5. Reliability assessment (Wilson CI, sample size warnings)
    6. Breakdown: by tier, by pattern, by market state
    7. Bullish vs bearish split
    8. Research integrity notes

    Two expectancy figures are always shown: one excluding timeouts and
    one treating all timeouts as -0.5R losses (a conservative assumption
    that half the stop distance is lost on timeout exits). This brackets
    the true expectancy without requiring modeled timeout exits.
    """
    _sep()
    _header(f"REPLAY RESULTS  ·  {symbol}  {timeframe.upper()}")
    _sep()

    # ── Metadata ──────────────────────────────────────────────────────────
    date_start = df.index[0].strftime("%Y-%m-%d")
    date_end   = df.index[-1].strftime("%Y-%m-%d")
    total_bars = len(df)
    print(f"\n  Dataset    : {date_start}  →  {date_end}")
    print(f"  Bars       : {total_bars:,}  ({timeframe} candles)")
    print(f"  Windows    : {max(0, total_bars - WINDOW_SIZE):,}  "
          f"({WINDOW_SIZE}-bar context)")
    print(f"  Max forward: {MAX_BARS_FORWARD} bars per signal")

    # ── Signal inventory ──────────────────────────────────────────────────
    total_signals = len(outcome_records)
    if total_signals == 0:
        _section("SIGNAL INVENTORY")
        print("  No signals were produced during this replay.")
        print("  Possible causes:")
        print("    - Dataset too short for patterns to form")
        print("    - Gate blocked all windows (check market state distribution)")
        print("    - Pattern thresholds too strict for this asset/period")
        _sep()
        return

    timeouts = sum(1 for r in outcome_records if r.outcome.outcome_type == "timeout")
    resolved = total_signals - timeouts
    timeout_pct = (timeouts / total_signals * 100) if total_signals > 0 else 0.0

    _section("SIGNAL INVENTORY")
    print(f"  Total signals     : {total_signals:>6}")
    print(f"  Resolved          : {resolved:>6}  "
          f"(outcome determined within {MAX_BARS_FORWARD} bars)")
    print(f"  Timeouts          : {timeouts:>6}  ({timeout_pct:.1f}% of all signals)")

    if timeout_pct > 40:
        print(f"\n  ⚠  HIGH TIMEOUT RATE ({timeout_pct:.1f}%)")
        print(f"     Resolved-only statistics below are an OPTIMISTIC CEILING.")
        print(f"     The all-entered statistics give the honest lower bound.")
    elif timeout_pct > 20:
        print(f"\n  ⚠  Timeout rate {timeout_pct:.1f}% — review all-entered stats below.")

    # ── Resolved-only statistics (what ReplayStatistics computes) ─────────
    stats_engine = ReplayStatistics()
    report = stats_engine.compute(outcome_records)

    _section("RESOLVED SIGNALS  (timeouts excluded — optimistic ceiling)")

    if report.total_signals == 0:
        print("  No resolved signals to report.")
    else:
        _print_core_stats(report)

        # Wilson confidence interval on win rate
        lo, hi = _wilson_ci(report.wins, report.total_signals)
        print(f"  Win rate CI (95%) : [{lo:.1%}, {hi:.1%}]")

        if report.total_signals < MIN_SIGNALS_FOR_STATS:
            print(
                f"\n  ⚠  SAMPLE SIZE WARNING: {report.total_signals} resolved signals "
                f"(minimum {MIN_SIGNALS_FOR_STATS} recommended)."
            )
            print(f"     Confidence interval spans {hi - lo:.0%} — results are not reliable.")

    # ── All-entered statistics (conservative floor) ───────────────────────
    _section("ALL-ENTERED SIGNALS  (timeouts treated as losses — conservative floor)")

    if resolved == 0:
        print("  No resolved signals — cannot compute floor statistics.")
    else:
        # Compute floor expectancy: timeouts modeled as -0.5R
        # (half-stop loss — a conservative assumption for positions
        #  closed at market at bar MAX_BARS_FORWARD without a defined exit)
        TIMEOUT_R = -0.5
        floor_wins = report.wins
        floor_total = total_signals
        floor_losses_from_stops = report.losses
        floor_losses_from_timeouts = timeouts

        floor_win_rate = floor_wins / floor_total if floor_total > 0 else 0.0
        floor_loss_rate = (floor_losses_from_stops + floor_losses_from_timeouts) / floor_total \
                          if floor_total > 0 else 0.0

        # Floor expectancy: wins contribute realized R, resolved losses
        # contribute -1R, timeout "losses" contribute TIMEOUT_R
        if report.total_signals > 0:
            # avg realized R from resolved wins (already computed correctly by stats engine)
            avg_win_r = (report.expectancy + (1 - report.win_rate)) / report.win_rate \
                        if report.win_rate > 0 else 0.0
            # Recompute cleanly: expectancy_resolved = win_rate * avg_R - loss_rate * 1
            # → avg_R = (expectancy_resolved + loss_rate * 1) / win_rate
            if report.win_rate > 0:
                avg_win_r = (report.expectancy + report.loss_rate) / report.win_rate
            else:
                avg_win_r = 0.0

            floor_expectancy = (
                (floor_wins / floor_total) * avg_win_r
                - (floor_losses_from_stops / floor_total) * 1.0
                + (floor_losses_from_timeouts / floor_total) * TIMEOUT_R
            )
        else:
            floor_expectancy = 0.0

        floor_lo, floor_hi = _wilson_ci(floor_wins, floor_total)

        print(f"  Total signals     : {floor_total:>6}")
        print(f"  Wins              : {floor_wins:>6}  ({floor_win_rate:.1%})")
        print(f"  Stop losses       : {floor_losses_from_stops:>6}")
        print(f"  Timeouts (as −0.5R): {floor_losses_from_timeouts:>5}")
        print(f"  Win rate          : {floor_win_rate:.1%}")
        print(f"  Win rate CI (95%) : [{floor_lo:.1%}, {floor_hi:.1%}]")
        print(f"  Expectancy floor  : {floor_expectancy:+.4f}R")
        print()
        print(f"  Note: Resolved-only expectancy was {report.expectancy:+.4f}R.")
        print(f"        Truth lies between {floor_expectancy:+.4f}R and "
              f"{report.expectancy:+.4f}R depending on how timeouts resolve.")

    # ── Outcome distribution ──────────────────────────────────────────────
    _section("OUTCOME DISTRIBUTION")
    type_counts = Counter(r.outcome.outcome_type for r in outcome_records)
    for label, key in [
        ("Target 1 hit", "target1"),
        ("Target 2 hit", "target2"),
        ("Target 3 hit", "target3"),
        ("Stop hit    ", "stop"),
        ("Timeout     ", "timeout"),
    ]:
        n = type_counts.get(key, 0)
        pct = n / total_signals * 100 if total_signals > 0 else 0.0
        bar = "█" * int(pct / 2)
        print(f"  {label} : {n:>5}  ({pct:5.1f}%)  {bar}")

    if resolved > 0:
        print()
        avg_bars = _mean_bars([
            r.outcome.bars_to_resolution
            for r in outcome_records
            if r.outcome.outcome_type != "timeout"
            and r.outcome.bars_to_resolution > 0
        ])
        print(f"  Avg bars to resolution (resolved only): {avg_bars:.1f}")

    # ── By tier ───────────────────────────────────────────────────────────
    if report.by_tier:
        _section("BY TIER  (resolved signals only)")
        if report.total_signals < MIN_SIGNALS_FOR_SEGMENTS:
            print(f"  ⚠  Fewer than {MIN_SIGNALS_FOR_SEGMENTS} resolved signals — "
                  f"segment stats are unreliable.")
        _print_segment_table(report.by_tier)

    # ── By pattern ────────────────────────────────────────────────────────
    if report.by_pattern:
        _section("BY PATTERN  (resolved signals only)")
        _print_segment_table(report.by_pattern)

    # ── By market state ───────────────────────────────────────────────────
    if report.by_market_state:
        _section("BY MARKET STATE  (resolved signals only)")
        _print_segment_table(report.by_market_state)

    # ── Bullish vs bearish ────────────────────────────────────────────────
    bullish_records = [
        r for r in outcome_records
        if r.replay.tiered_signal is not None
        and r.replay.tiered_signal.direction == "bullish"
    ]
    bearish_records = [
        r for r in outcome_records
        if r.replay.tiered_signal is not None
        and r.replay.tiered_signal.direction == "bearish"
    ]

    if bullish_records or bearish_records:
        _section("BULLISH vs BEARISH SPLIT  (all signals including timeouts)")
        for label, group in [("Bullish", bullish_records), ("Bearish", bearish_records)]:
            n = len(group)
            n_wins = sum(
                1 for r in group
                if r.outcome.outcome_type in {"target1", "target2", "target3"}
            )
            n_stops = sum(1 for r in group if r.outcome.outcome_type == "stop")
            n_timeouts = sum(1 for r in group if r.outcome.outcome_type == "timeout")
            n_resolved = n_wins + n_stops
            wr = n_wins / n_resolved if n_resolved > 0 else 0.0
            print(f"  {label:8s}: {n:>4} signals  "
                  f"wins={n_wins}  stops={n_stops}  timeouts={n_timeouts}  "
                  f"resolved win rate={wr:.1%}")

    # ── Research integrity notes ──────────────────────────────────────────
    _section("RESEARCH INTEGRITY NOTES")
    print("  These results reflect the engine as configured, applied to the")
    print("  provided dataset. Interpret with the following caveats:")
    print()
    print("  1. SINGLE DATASET: Results are specific to this symbol and period.")
    print("     Validate on a separate symbol and at least one bear-market segment")
    print("     before drawing conclusions about the strategy.")
    print()
    print(f"  2. TIMEOUT MODEL: {timeouts} signals ({timeout_pct:.1f}%) timed out without")
    print(f"     resolution within {MAX_BARS_FORWARD} bars. These are NOT included in the")
    print("     resolved-only statistics. The floor expectancy above models them")
    print("     conservatively as −0.5R. Real exits may differ.")
    print()
    print("  3. FILL ASSUMPTION: Targets and stops are evaluated via candle")
    print("     high/low touch. Actual fills may be worse due to slippage,")
    print("     liquidity gaps, and wick-only touches. Expectancy has an")
    print("     uncertainty band of roughly ±0.15R from fill assumptions.")
    print()
    print("  4. NO PARAMETER TUNING: These thresholds were not adjusted to")
    print("     improve this output. If they were, these results are invalid.")
    print()
    if total_signals < MIN_SIGNALS_FOR_STATS:
        print(f"  5. ⚠  SAMPLE SIZE CRITICAL: Only {total_signals} signals total.")
        print(f"     Minimum {MIN_SIGNALS_FOR_STATS} required for any statistical interpretation.")
        print("     Extend your dataset before drawing conclusions.")
        print()
    elif total_signals < MIN_SIGNALS_FOR_SEGMENTS:
        print(f"  5. ⚠  SAMPLE SIZE: {total_signals} signals is marginal.")
        print(f"     Segment statistics require {MIN_SIGNALS_FOR_SEGMENTS}+ for reliability.")
        print()

    _sep()


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Internal printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_core_stats(report: StatisticsReport) -> None:
    """Print the central statistics block for a report."""
    print(f"  Signals           : {report.total_signals:>6}")
    print(f"  Wins              : {report.wins:>6}  ({report.win_rate:.1%})")
    print(f"  Losses            : {report.losses:>6}  ({report.loss_rate:.1%})")
    print(f"  Target 1 rate     : {report.target1_hit_rate:.1%}")
    print(f"  Target 2 rate     : {report.target2_hit_rate:.1%}")
    print(f"  Target 3 rate     : {report.target3_hit_rate:.1%}")
    print(f"  Stop rate         : {report.stop_hit_rate:.1%}")
    print(f"  Avg bars to exit  : {report.avg_bars_to_resolution:.1f}")
    print(f"  Expectancy        : {report.expectancy:+.4f}R")


def _print_segment_table(segments: dict) -> None:
    """Print a compact table for segmented statistics."""
    if not segments:
        print("  (no data)")
        return
    col_w = 14
    hdr = f"  {'Segment':<{col_w}}  {'N':>5}  {'W':>5}  {'L':>5}  " \
          f"{'Win%':>6}  {'Exp':>7}  {'CI 95%'}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for key, sub in sorted(segments.items()):
        lo, hi = _wilson_ci(sub.wins, sub.total_signals)
        warn = " ⚠" if sub.total_signals < 30 else ""
        print(
            f"  {key:<{col_w}}  "
            f"{sub.total_signals:>5}  "
            f"{sub.wins:>5}  "
            f"{sub.losses:>5}  "
            f"{sub.win_rate:>6.1%}  "
            f"{sub.expectancy:>+7.3f}R  "
            f"[{lo:.0%}, {hi:.0%}]{warn}"
        )


def _wilson_ci(wins: int, total: int, z: float = _Z95) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    More accurate than normal approximation for small samples.
    Returns (lower, upper) both in [0.0, 1.0].
    """
    if total == 0:
        return 0.0, 1.0
    p = wins / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _mean_bars(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _sep() -> None:
    print("─" * 65)


def _header(text: str) -> None:
    print(f"  {text}")


def _section(title: str) -> None:
    print()
    print(f"  ── {title} ──")
    print()


def _print_step(msg: str) -> None:
    print(f"  ▶  {msg}")


def _print_ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def _abort(msg: str) -> None:
    print(f"\n  ✗  ERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic historical replay for harmonic signal research.\n"
            "Produces an honest research summary without modifying any parameters."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        required=True,
        metavar="FILE",
        help="Path to OHLCV CSV file (timestamp, open, high, low, close, volume).",
    )
    parser.add_argument(
        "--symbol",
        required=True,
        choices=sorted(SUPPORTED_SYMBOLS),
        help="Trading pair symbol (BTCUSDT or ETHUSDT).",
    )
    parser.add_argument(
        "--tf",
        required=True,
        dest="timeframe",
        choices=sorted(SUPPORTED_TIMEFRAMES),
        help="Candle timeframe (1h or 4h).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print()
    _sep()
    _header("HARMONIC REPLAY ENGINE  ·  Research Mode")
    _header(f"Symbol: {args.symbol}   Timeframe: {args.timeframe.upper()}")
    _sep()
    print()

    # Phase 1: Load data
    df = load_csv(args.csv)

    # Phase 2: Run replay (forward-only, dry_run)
    records = run_replay(df, args.symbol, args.timeframe)

    if not records:
        print()
        print("  No signals were produced. Nothing to evaluate.")
        print("  This is a valid result — it means the engine found no qualifying")
        print("  harmonic patterns during this period under current thresholds.")
        print()
        _sep()
        return

    # Phase 3: Track outcomes (strictly future bars)
    outcome_records = track_outcomes(records, df)

    # Phase 4: Print summary
    print_summary(outcome_records, args.symbol, args.timeframe, df)


if __name__ == "__main__":
    main()