"""
telemetry/logger.py
====================
Structured Telemetry Logger — Week 1 Implementation

Philosophy:
    Append-only JSONL files. One event per line.
    Zero external dependencies. Pure Python stdlib.
    Readable with tail -f. Queryable with jq.
    Never blocks the main pipeline. Never raises.

Event types (7 total — 5 from Week 1, 2 added in Week 2):
    market_state    : Every classification result
    gate_block      : Every hostile market gate trigger
    detector_detail : Full per-detector breakdown (debug mode only)
    scan_cycle      : Every pipeline execution summary
    error           : Any caught exception
    signal_delivered : Every successfully delivered trading signal
    daily_counter_block : Signal blocked by daily frequency cap

File layout:
    logs/telemetry/
        market_state.jsonl
        gate_block.jsonl
        detector_detail.jsonl
        scan_cycle.jsonl
        error.jsonl
        signal_delivered.jsonl
        daily_counter_block.jsonl

Each JSONL record:
    {
        "ts":    1705312800.123,   ← Unix timestamp (float)
        "type":  "market_state",   ← event type
        "iso":   "2024-01-15T14:00:00Z",  ← human-readable
        ...event-specific fields...
    }

Query examples:
    # Live state stream:
    tail -f logs/telemetry/market_state.jsonl | python3 -m json.tool

    # Count gate blocks by reason:
    cat logs/telemetry/gate_block.jsonl | jq '.block_code' | sort | uniq -c

    # All RANGING classifications with confidence > 0.6:
    cat logs/telemetry/market_state.jsonl | \\
        jq 'select(.dominant=="ranging" and .confidence > 0.6)'

    # Scan cycle outcome distribution:
    cat logs/telemetry/scan_cycle.jsonl | jq '.outcome' | sort | uniq -c
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("telemetry")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TELEMETRY_DIR  = Path("logs/telemetry")
DEBUG_MODE     = False   # Set True to enable detector_detail events

# ---------------------------------------------------------------------------
# Internal writer — the only function that touches disk
# ---------------------------------------------------------------------------

def _write(event_type: str, payload: Dict[str, Any]) -> None:
    """
    Appends one JSON line to the event-type file.
    Never raises — all exceptions are swallowed and logged to stderr.
    """
    try:
        TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
        now = time.time()
        record = {
            "ts":   now,
            "iso":  datetime.fromtimestamp(now, tz=timezone.utc)
                    .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "type": event_type,
            **payload,
        }
        log_file = TELEMETRY_DIR / f"{event_type}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        # Telemetry must NEVER break the pipeline
        logger.error(f"Telemetry write failed | {event_type} | {e}")


# ---------------------------------------------------------------------------
# Event 1: Market State Classification
# ---------------------------------------------------------------------------

def log_state(
    vector: MarketStateVector,
    fusion_result: Optional[FusionResult] = None
) -> None:
    """
    Logs every MarketStateVector produced by the engine.
    ...
    """

    if vector is None:
        return

    payload = {
        "symbol":       vector.symbol,
        "timeframe":    vector.timeframe,
        "bar_index":    vector.bar_index,
        "dominant":     vector.dominant_state,
        "confidence":   round(vector.confidence, 4),
        "is_confident": vector.is_confident,
        # State probabilities
        "trending":     round(vector.trending,    4),
        "ranging":      round(vector.ranging,     4),
        "expansion":    round(vector.expansion,   4),
        "compression":  round(vector.compression, 4),
        "reversal":     round(vector.reversal,    4),
        "news_chaos":   round(vector.news_chaos,  4),
        # Downstream multiplier
        "harmonic_mult": round(vector.harmonic_edge_multiplier(), 4),
        "is_hostile":    vector.is_hostile(),
    }

    # Add fusion audit trail if available
    if fusion_result is not None:
        payload["corrections_count"] = len(fusion_result.corrections)
        if fusion_result.corrections:
            payload["corrections"] = fusion_result.corrections

    _write("market_state", payload)

    # Also log detector breakdown in debug mode
    if DEBUG_MODE and fusion_result is not None:
        log_detector_detail(vector, fusion_result)


# ---------------------------------------------------------------------------
# Event 2: Gate Block
# ---------------------------------------------------------------------------

def log_gate_block(
    symbol:     str,
    timeframe:  str,
    block_code: str,
    reason:     str,
    vector,                          # MarketStateVector
) -> None:
    """
    Logs every time the HostileMarketGate blocks the pipeline.

    Called in: pipeline.py immediately after gate.check() returns blocked.
    This event is important for understanding how often and why
    the system is suppressing signal generation.
    """
    if vector is None:
        return
    _write("gate_block", {
        "symbol":     symbol,
        "timeframe":  timeframe,
        "block_code": block_code,
        "reason":     reason,
        "dominant":   vector.dominant_state,
        "confidence": round(vector.confidence, 4),
        "news_chaos": round(vector.news_chaos, 4),
        "compression":round(vector.compression,4),
    })


# ---------------------------------------------------------------------------
# Event 3: Detector Detail (debug mode only)
# ---------------------------------------------------------------------------

def log_detector_detail(
    vector,           # MarketStateVector
    fusion_result,    # FusionResult
) -> None:
    """
    Logs full per-detector breakdown for debugging.
    Only written when DEBUG_MODE = True.
    High volume — do not enable in production continuously.

    Called in: log_state() when DEBUG_MODE is True.
    """
    if not DEBUG_MODE:
        return

    payload = {
        "symbol":    vector.symbol,
        "timeframe": vector.timeframe,
    }
    # Raw scores before correction
    for k, v in fusion_result.raw_scores.items():
        payload[f"raw.{k}"] = round(v, 4)
    # Adjusted scores after correction
    for k, v in fusion_result.adjusted_scores.items():
        payload[f"adj.{k}"] = round(v, 4)
    # Final probabilities
    for k, v in fusion_result.final_probs.items():
        payload[f"prob.{k}"] = round(v, 4)

    _write("detector_detail", payload)


# ---------------------------------------------------------------------------
# Event 4: Scan Cycle Summary
# ---------------------------------------------------------------------------

def log_scan_cycle(
    symbol:      str,
    timeframe:   str,
    duration_ms: float,
    outcome:     str,
    bars:        int = 0,
    swings:      int = 0,
    patterns:    int = 0,
) -> None:
    """
    Logs every scan cycle execution — regardless of outcome.

    Outcomes:
        "gate_blocked"       : HostileMarketGate blocked
        "no_swings"          : Swing detector found < 5 pivots
        "no_patterns"        : Harmonic detector found nothing
        "below_threshold"    : Patterns found but all scores too low
        "signal_published"   : At least one TieredSignal sent

    Called in: pipeline.py at the end of every scan_one() call.
    """
    _write("scan_cycle", {
        "symbol":      symbol,
        "timeframe":   timeframe,
        "duration_ms": round(duration_ms, 1),
        "outcome":     outcome,
        "bars":        bars,
        "swings":      swings,
        "patterns":    patterns,
    })


# ---------------------------------------------------------------------------
# Event 5: Error
# ---------------------------------------------------------------------------

def log_error(
    location:  str,
    error:     str,
    symbol:    str = "",
    timeframe: str = "",
    extra:     Optional[Dict] = None,
) -> None:
    """
    Logs any caught exception in the pipeline.

    Called in: any try/except block in pipeline.py or scanner.py.
    Do NOT use for expected conditions (no patterns, gate blocks).
    Those have their own event types.
    """
    payload = {
        "location":  location,
        "error":     error,
        "symbol":    symbol,
        "timeframe": timeframe,
    }
    if extra:
        payload.update(extra)

    _write("error", payload)
    logger.error(f"Pipeline error | {location} | {symbol} {timeframe} | {error}")


# ---------------------------------------------------------------------------
# Utility: Enable debug mode at runtime
# ---------------------------------------------------------------------------

def set_debug_mode(enabled: bool) -> None:
    """Toggle detector_detail logging on/off at runtime."""
    global DEBUG_MODE
    DEBUG_MODE = enabled
    logger.info(f"Telemetry debug mode: {enabled}")
def log_signal(
    signal,
    risk_reward: float=0.0,
    risk_pct: float=0.0,
    max_per_day: int=0,
) -> None:
    """
    Logs every successfully delivered trading signal.

    Called in: pipeline.py immediately after Telegram delivery succeeds.
    """

    if signal is None:
        return

    try:
        _write("signal_delivered", {
            "symbol": signal.symbol,
            "timeframe": signal.timeframe,
            "pattern": signal.pattern_name,
            "direction": signal.direction,
            "tier": signal.tier,
            "edge_score": round(signal.edge_score, 4),
            "entry": signal.entry,
            "stop": signal.stop,
            "target1": signal.target1,
            "market_state": getattr(signal, "dominant_state", None),
            "risk_reward": round(float(getattr(signal, "risk_reward", 0.0)), 3),
            "risk_pct": getattr(signal, "risk_pct", 0.0),
            "base_score": getattr(signal, "base_score", signal.edge_score),
            "state_discount": getattr(signal, "state_discount", 1.0),
            "max_per_day": getattr(signal, "max_per_day", max_per_day),
            "reasoning_lines": len(signal.reasoning),
            "conf_weight": round(float(getattr(signal, "confidence_weight", 1.0)), 4),
    })
    except Exception:
        return


def log_daily_counter_block(
    symbol: str,
    timeframe: str,
    tier: str,
    current_count: int,
    max_count: int,
    edge_score: float,
) -> None:
    """
    Logs every signal blocked by the daily frequency cap.
    """
    try:
        _write("daily_counter_block", {
            "symbol": symbol,
            "timeframe": timeframe,
            "tier": tier,
            "current_count": current_count,
            "max_count": max_count,
            "edge_score": round(float(edge_score), 4),
            "utc_date": datetime.now(timezone.utc).date().isoformat(),
        })
    except Exception:
        return 
        
class TelemetryLogger:
    @staticmethod
    def log_error(*args, **kwargs):
        return log_error(*args, **kwargs)

    @staticmethod
    def log_state(*args, **kwargs):
        return log_state(*args, **kwargs)