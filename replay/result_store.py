"""replay/result_store.py — newline-delimited JSON result storage."""
from __future__ import annotations
import json
import os
from typing import List

from replay.replay_runner import ReplayRecord


class ResultStore:
    """Append-only newline-delimited JSON (NDJSON) store for replay records.

    Phase 1 is write-only. Deserialization is deferred to later phases.
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────

    def append(self, record: ReplayRecord) -> None:
        """Serialize a single record as one JSON line."""
        line = self._serialize(record)
        with open(self._filepath, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def extend(self, records: List[ReplayRecord]) -> None:
        """Serialize multiple records efficiently."""
        if not records:
            return
        lines = [self._serialize(r) for r in records]
        with open(self._filepath, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def clear(self) -> None:
        """Remove the store file if it exists."""
        if os.path.exists(self._filepath):
            os.remove(self._filepath)

    @property
    def filepath(self) -> str:
        return self._filepath

    # ── Serialization ───────────────────────────────────────────────────────

    @staticmethod
    def _serialize(record: ReplayRecord) -> str:
        payload = {
            "bar_timestamp": record.bar_timestamp,
            "scan_result": {
                "symbol": record.scan_result.symbol,
                "timeframe": record.scan_result.timeframe,
                "outcome": record.scan_result.outcome,
                "duration_ms": record.scan_result.duration_ms,
                "swings": record.scan_result.swings,
                "patterns": record.scan_result.patterns,
                "scored_count": record.scan_result.scored_count,
                "published_count": record.scan_result.published_count,
            },
            "tiered_signal": None,
        }
        if record.tiered_signal is not None:
            ts = record.tiered_signal
            payload["tiered_signal"] = {
                "tier": ts.tier,
                "edge_score": ts.edge_score,
                "pattern_name": ts.pattern_name,
                "symbol": ts.symbol,
                "timeframe": ts.timeframe,
                "entry": ts.entry,
                "stop": ts.stop,
                "target1": ts.target1,
                "target2": ts.target2,
                "target3": ts.target3,
                "risk_reward": ts.risk_reward,
                "risk_pct": ts.risk_pct,
                "is_paper_only": ts.is_paper_only,
            }
        return json.dumps(payload, ensure_ascii=False, default=str)