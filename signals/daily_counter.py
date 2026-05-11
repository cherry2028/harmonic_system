"""
signals/daily_counter.py
=========================
DailyCounter — Thread-Safe UTC Daily Frequency Cap Enforcer

Single responsibility:
    Track how many signals of each (tier, symbol) combination have
    been delivered today (UTC), and enforce the daily frequency cap
    defined in MarketStateConfig.

    Answers one question per call:
        "May I deliver a Tier A+ signal for BTCUSDT right now?"

Architectural position:
    Called by pipeline.py between SignalTier.classify() and
    TelegramFormatter.send(). The pipeline pattern is:

        tiered = signal_tier.classify(scored)
        if tiered is None: return

        if not counter.check(tiered.tier, symbol):
            telemetry.log_daily_counter_block(...)
            return                              ← cap hit

        formatter.send(tiered)
        counter.increment(tiered.tier, symbol)  ← commit after delivery

    check() and increment() are separate by design:
        - check() reads state without modifying it
        - increment() commits only after successful delivery
        - If formatter.send() fails, increment() is never called
          → no phantom count against the cap

Thread-safety model:
    A single threading.RLock guards all in-memory state mutations.
    The lock is held only during read-modify operations on the
    in-memory counter dict (microseconds).
    File I/O happens AFTER the lock is released, using a snapshot
    of the modified state. This minimises lock contention.

    RLock (re-entrant) rather than Lock:
        In principle check() and increment() should not be called
        re-entrantly from the same thread, but RLock is safer and
        has identical performance characteristics for non-re-entrant use.

UTC-only logic:
    All date comparisons use datetime.now(timezone.utc).date().
    Local time is never used. The counter file is named by UTC date.
    Rollover detection: every public method call checks the current
    UTC date against the stored date. If they differ, the counter
    is reset automatically. No external scheduler required.

Persistent storage:
    One JSON file per UTC date: logs/counters/YYYY-MM-DD.json
    Schema:
        {
            "_meta": {
                "utc_date": "2024-01-15",
                "created": "2024-01-15T00:00:00Z",
                "version": 1
            },
            "A+": {"BTCUSDT": 1, "ETHUSDT": 0},
            "A":  {"BTCUSDT": 3, "SOLUSDT": 1},
            "B":  {},
            "C":  {}
        }

    File is written on every increment() call. At maximum load
    (Tier A+: 1 signal/day × 3 symbols = 3 writes/day, Tier A:
    3×3 = 9, Tier B: 5×3 = 15) total writes ≈ 27/day — negligible.

Corrupt-file recovery:
    If the counter file exists but cannot be parsed (truncated,
    invalid JSON, wrong schema), DailyCounter logs WARNING and
    starts fresh with all-zero counts. This means the daily cap
    may be slightly over-delivered on the day of corruption.
    This is the correct tradeoff: availability > strict accuracy.
    A corrupt file should never prevent signal delivery.

Never-raise contract:
    check()     returns bool. Never raises.
    increment() returns None. Never raises.
    reset()     returns None. Never raises.
    Both catch all exceptions internally and log them.

Dependencies:
    threading, json, pathlib, datetime — stdlib only
    logging                           — stdlib only
    config.market_state_config        — for valid tier set (lazy import)

    NO imports from: signals/signal.py, signals/tier.py,
                     scoring/, market_state/, delivery/, pipeline.py
    DailyCounter knows NOTHING about patterns, states, or signals.
    It is a generic frequency counter.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("signals.daily_counter")

# ---------------------------------------------------------------------------
# Storage configuration
# ---------------------------------------------------------------------------

# Default storage directory.
# Overridable per-instance for testing (inject counter_dir argument).
_DEFAULT_COUNTER_DIR: Path = Path("logs/counters")

# JSON schema version — increment when schema changes incompatibly
_SCHEMA_VERSION: int = 1

# Sentinel for "no date loaded yet"
_NO_DATE = date(1970, 1, 1)


# ---------------------------------------------------------------------------
# CounterState — internal value object
# ---------------------------------------------------------------------------

class _CounterState:
    """
    Internal representation of the in-memory counter.

    Not part of the public API. Held exclusively by DailyCounter.

    Attributes:
        utc_date  : The UTC date this state applies to.
        counts    : Nested dict: tier -> symbol -> int.
                    Keys are present only when count > 0, or when
                    explicitly initialised. Missing key ≡ 0.
        dirty     : True when in-memory state differs from disk.
                    Used to skip unnecessary writes.
    """

    __slots__ = ("utc_date", "counts", "dirty")

    def __init__(self, utc_date: date) -> None:
        self.utc_date: date                     = utc_date
        self.counts:   Dict[str, Dict[str, int]] = {}
        self.dirty:    bool                      = False

    def get(self, tier: str, symbol: str) -> int:
        """Returns current count for (tier, symbol). 0 if never incremented."""
        return self.counts.get(tier, {}).get(symbol, 0)

    def increment(self, tier: str, symbol: str) -> int:
        """Increments count by 1. Returns new count."""
        tier_dict = self.counts.setdefault(tier, {})
        new_count = tier_dict.get(symbol, 0) + 1
        tier_dict[symbol] = new_count
        self.dirty = True
        return new_count

    def to_json(self) -> dict:
        """Serialises state to JSON-compatible dict."""
        return {
            "_meta": {
                "utc_date": self.utc_date.isoformat(),
                "created":  datetime.now(timezone.utc).isoformat(),
                "version":  _SCHEMA_VERSION,
            },
            **self.counts,
        }

    @classmethod
    def from_json(cls, data: dict, expected_date: date) -> "_CounterState":
        """
        Deserialises a JSON dict into a _CounterState.

        Validation:
            - _meta must exist and contain utc_date
            - utc_date in _meta must match expected_date
            - All non-_meta keys are treated as tier dicts
            - Values must be dicts of {str: int}

        Raises ValueError if validation fails — caller handles recovery.
        """
        meta = data.get("_meta", {})
        stored_date_str = meta.get("utc_date", "")

        if not stored_date_str:
            raise ValueError("Missing _meta.utc_date in counter file")

        stored_date = date.fromisoformat(stored_date_str)
        if stored_date != expected_date:
            raise ValueError(
                f"Counter file date {stored_date} != expected {expected_date}"
            )

        state = cls(utc_date=expected_date)
        for key, value in data.items():
            if key == "_meta":
                continue
            if not isinstance(value, dict):
                raise ValueError(
                    f"Tier '{key}' value is not a dict: {type(value).__name__}"
                )
            for sym, count in value.items():
                if not isinstance(count, int) or count < 0:
                    raise ValueError(
                        f"Counter[{key}][{sym}]={count!r} is not a non-negative int"
                    )
            state.counts[key] = dict(value)

        return state


# ---------------------------------------------------------------------------
# DailyCounter
# ---------------------------------------------------------------------------

class DailyCounter:
    """
    Thread-safe, persistent, UTC-based daily frequency counter.

    Usage:
        counter = DailyCounter()

        # Before delivery:
        if not counter.check(tier="A+", symbol="BTCUSDT"):
            return   # cap hit — skip delivery

        # After successful delivery:
        counter.increment(tier="A+", symbol="BTCUSDT")

    Lifecycle:
        Construct once per process. Reuse for all scan cycles.
        The counter auto-resets when UTC date changes.
        No explicit reset() call required in normal operation.

    Args:
        max_counts : Dict mapping tier name to daily cap integer.
                     If None, loaded from MS_CONFIG.tier_rules().
        counter_dir: Directory for persistent JSON files.
                     Defaults to logs/counters/.
                     Override in tests for isolation.
    """

    def __init__(
        self,
        max_counts:  Optional[Dict[str, int]] = None,
        counter_dir: Optional[Path]           = None,
    ) -> None:
        self._counter_dir = Path(counter_dir) if counter_dir else _DEFAULT_COUNTER_DIR
        self._lock        = threading.RLock()
        self._state:      Optional[_CounterState] = None
        self._max_counts: Dict[str, int] = self._resolve_max_counts(max_counts)

        logger.debug(
            f"DailyCounter initialized | "
            f"dir={self._counter_dir} | "
            f"caps={self._max_counts}"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def check(self, tier: str, symbol: str) -> bool:
        """
        Returns True if a signal of this tier may be delivered for symbol.
        Returns False if the daily cap has been reached.

        Does NOT modify state. Call increment() after successful delivery.

        Never raises. Returns True on any internal error (safe default:
        cap enforcement fails open rather than blocking valid signals).

        Args:
            tier   : Tier name — "A+", "A", "B", "C"
            symbol : Trading pair — e.g. "BTCUSDT"
        """
        try:
            return self._check(tier, symbol)
        except Exception as e:
            logger.error(
                f"DailyCounter.check() raised unexpectedly: "
                f"{type(e).__name__}: {e}. "
                f"Returning True (fail-open — do not block delivery).",
                exc_info=True,
            )
            return True   # fail-open: unexpected error → allow delivery

    def increment(self, tier: str, symbol: str) -> None:
        """
        Records one delivery for (tier, symbol) today.

        Must be called AFTER successful delivery — not before.
        If called before and delivery fails, the cap is consumed
        unnecessarily.

        Never raises. Logs ERROR on file write failure (in-memory
        state is still updated — counter is accurate in memory).

        Args:
            tier   : Tier name — "A+", "A", "B", "C"
            symbol : Trading pair — e.g. "BTCUSDT"
        """
        try:
            self._increment(tier, symbol)
        except Exception as e:
            logger.error(
                f"DailyCounter.increment() raised unexpectedly: "
                f"{type(e).__name__}: {e}. "
                f"Counter may be inconsistent.",
                exc_info=True,
            )

    def current_count(self, tier: str, symbol: str) -> int:
        """
        Returns the current delivery count for (tier, symbol) today.

        Returns 0 on any error. Never raises.
        Primarily used in tests and telemetry.
        """
        try:
            with self._lock:
                state = self._ensure_state_for_today()
                return state.get(tier, symbol)
        except Exception as e:
            logger.error(
                f"DailyCounter.current_count() raised: {e}",
                exc_info=True,
            )
            return 0

    def reset(self) -> None:
        """
        Clears all in-memory counters and forces a fresh state.

        Does NOT delete the persisted file. Call this only in tests
        or for emergency operator resets. Normal midnight rollover
        is automatic and does not require this method.

        Never raises.
        """
        try:
            with self._lock:
                self._state = None
                logger.info("DailyCounter: in-memory state reset")
        except Exception as e:
            logger.error(f"DailyCounter.reset() raised: {e}", exc_info=True)

    def cap_for(self, tier: str) -> int:
        """
        Returns the configured daily cap for a tier.
        Returns 0 if tier is unknown (effectively never deliver).
        Never raises.
        """
        return self._max_counts.get(tier, 0)

    # ── Internal implementation ───────────────────────────────────────────

    def _check(self, tier: str, symbol: str) -> bool:
        """Core check logic. May raise — caller wraps in try/except."""
        with self._lock:
            state    = self._ensure_state_for_today()
            count    = state.get(tier, symbol)
            max_cap  = self._max_counts.get(tier)

            if max_cap is None:
                # Unknown tier — log warning, allow delivery
                logger.warning(
                    f"DailyCounter.check(): unknown tier={tier!r}. "
                    f"Allowing delivery (no cap configured)."
                )
                return True

            allowed = count < max_cap
            logger.debug(
                f"check | tier={tier} symbol={symbol} | "
                f"count={count}/{max_cap} | allowed={allowed}"
            )
            return allowed

    def _increment(self, tier: str, symbol: str) -> None:
        """Core increment logic. May raise — caller wraps in try/except."""
        with self._lock:
            state    = self._ensure_state_for_today()
            new_count = state.increment(tier, symbol)
            snapshot  = state.to_json()
            file_path = self._file_path_for_date(state.utc_date)

        # File I/O outside lock — snapshot is already captured
        self._write_file(file_path, snapshot)
        logger.debug(
            f"increment | tier={tier} symbol={symbol} | "
            f"new_count={new_count}"
        )

    def _ensure_state_for_today(self) -> _CounterState:
        """
        Returns the _CounterState for today's UTC date.
        If state is None or stale (date rolled over), loads or creates
        a fresh state. Always called under self._lock.
        """
        today = datetime.now(timezone.utc).date()

        if self._state is not None and self._state.utc_date == today:
            return self._state   # fast path — common case

        # Date has changed or first call — load or create
        self._state = self._load_or_create(today)
        return self._state

    def _load_or_create(self, target_date: date) -> _CounterState:
        """
        Attempts to load counter state from disk for target_date.
        Falls back to a fresh state on any failure.
        Always called under self._lock.
        """
        file_path = self._file_path_for_date(target_date)

        if file_path.exists():
            try:
                raw  = file_path.read_text(encoding="utf-8")
                data = json.loads(raw)
                state = _CounterState.from_json(data, target_date)
                logger.debug(
                    f"Counter loaded from {file_path} | "
                    f"date={target_date} | "
                    f"counts={state.counts}"
                )
                return state
            except Exception as e:
                # Corrupt or wrong-date file — start fresh
                logger.warning(
                    f"Counter file {file_path} could not be loaded: "
                    f"{type(e).__name__}: {e}. "
                    f"Starting fresh (all counts reset to 0). "
                    f"Daily caps may be slightly over-delivered today."
                )

        # No file or corrupt — create fresh
        state = _CounterState(utc_date=target_date)
        logger.debug(f"Counter created fresh for {target_date}")
        return state

    def _write_file(self, file_path: Path, data: dict) -> None:
        """
        Atomically writes counter state to disk.

        Uses write-to-temp-then-rename pattern on POSIX systems to
        prevent partial writes. On Windows, rename() may fail if the
        target exists — we fall back to direct write in that case.

        Called OUTSIDE the lock. data is an immutable snapshot.
        """
        try:
            self._counter_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = file_path.with_suffix(".tmp")

            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()

            # Atomic rename on POSIX; best-effort on Windows
            try:
                tmp_path.rename(file_path)
            except OSError:
                # Windows: target exists — fallback to direct overwrite
                tmp_path.replace(file_path)

        except Exception as e:
            logger.error(
                f"DailyCounter: failed to write {file_path}: "
                f"{type(e).__name__}: {e}. "
                f"In-memory state is still accurate.",
                exc_info=True,
            )

    def _file_path_for_date(self, utc_date: date) -> Path:
        """Returns the canonical file path for a given UTC date."""
        return self._counter_dir / f"{utc_date.isoformat()}.json"

    # ── Init helper ───────────────────────────────────────────────────────

    @staticmethod
    def _resolve_max_counts(
        max_counts: Optional[Dict[str, int]]
    ) -> Dict[str, int]:
        """
        Resolves the max_counts dict from the argument or MS_CONFIG.

        If max_counts is provided: validates types and uses it.
        If None: loads from MS_CONFIG.tier_rules().

        Returns Dict[str, int] with tier → daily_cap mapping.
        """
        if max_counts is not None:
            # Validate caller-provided caps
            for tier, cap in max_counts.items():
                if not isinstance(tier, str):
                    raise ValueError(
                        f"max_counts key {tier!r} must be a string tier name"
                    )
                if not isinstance(cap, int) or cap < 0:
                    raise ValueError(
                        f"max_counts[{tier!r}]={cap!r} must be a non-negative int"
                    )
            return dict(max_counts)

        # Load from MS_CONFIG (lazy import prevents circular imports)
        from config.market_state_config import MS_CONFIG
        return {
            tier: max_daily
            for tier, _threshold, max_daily, _risk
            in MS_CONFIG.tier_rules()
        }