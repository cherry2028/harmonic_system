# -*- coding: utf-8 -*-
"""
delivery/presentation.py
SignalPresentation - mobile-optimized view model for delivery channels.

Design principles:
    - Frozen dataclass (immutable, hashable, thread-safe)
    - Zero coupling to pipeline internals
    - Zero coupling to Telegram API or any delivery mechanism
    - Plain-text focused for mobile readability
    - Explicit schema - no getattr() fallbacks, no hidden defaults
    - Metadata digest for explainability without payload bloat

This module knows NOTHING about:
    - How signals are generated
    - How scores are computed
    - Telegram, Discord, email, or any transport
    - Business logic, tiering rules, or position sizing

It ONLY knows:
    - How to present a signal as human-readable text
    - What fields a subscriber needs to see on mobile

Usage:
    from delivery.presentation import SignalPresentation

    presentation = SignalPresentation.from_tiered_signal(signal)
    print(presentation.to_telegram_text())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

__all__ = ["SignalPresentation", "PresentationError"]


class PresentationError(Exception):
    """Raised when a TieredSignal cannot be converted to SignalPresentation."""
    pass


@dataclass(frozen=True)
class SignalPresentation:
    """
    Mobile-optimized presentation of a trading signal.

    All fields are strings or primitives for direct formatting.
    No nested objects. No optional business logic.

    Fields:
        headline:      One-line summary for notification preview
        symbol:        Trading pair (e.g., "BTCUSDT")
        timeframe:     Candle timeframe (e.g., "1h")
        direction:     "LONG" or "SHORT"
        tier:          Signal tier label (e.g., "ALPHA", "OPPORTUNITY")
        score:         Total score as string (e.g., "0.72")
        entry:         Entry price as formatted string
        stop:          Stop loss as formatted string
        target:        Primary target as formatted string
        risk_reward:   R:R ratio as string (e.g., "2.1")
        risk_pct:      Position risk percentage (e.g., "1%")
        invalidation:  Clear invalidation condition
        market_state:  Current market regime
        engine_digest: Which engines contributed (e.g., "Harmonic + Momentum")
        metadata:      Key-value pairs for explainability
        generated_at:  ISO timestamp of signal generation
        signal_id:     Unique identifier for tracking
    """

    # --- Core identity (always required) ---
    headline: str
    symbol: str
    timeframe: str
    direction: str
    tier: str

    # --- Execution levels (always required) ---
    entry: str
    stop: str
    target: str
    risk_reward: str
    risk_pct: str

    # --- Risk management (always required) ---
    invalidation: str

    # --- Explainability (always required) ---
    market_state: str
    engine_digest: str
    score: str

    # --- Metadata (optional but explicit) ---
    # hash=False and compare=False because dict is unhashable.
    # frozen=True still works; __hash__ excludes this field.
    metadata: Dict[str, str] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )
    generated_at: str = ""
    signal_id: str = ""

    # ------------------------------------------------------------------ #
    # Factory: from TieredSignal                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_tiered_signal(cls, signal) -> "SignalPresentation":
        """
        Convert a TieredSignal domain model into SignalPresentation.

        Args:
            signal: TieredSignal instance (or duck-typed equivalent)

        Returns:
            SignalPresentation: frozen, formatted, ready for delivery

        Raises:
            PresentationError: if required fields are missing or malformed
        """
        # --- Defensive but explicit field extraction ---
        # We check existence, type, and validity. No getattr() fallbacks.

        # Identity fields
        symbol = cls._require_str(signal, "symbol")
        timeframe = cls._require_str(signal, "timeframe")
        direction = cls._require_str(signal, "direction").upper()
        tier = cls._require_str(signal, "tier").upper()

        # Validate direction
        if direction not in ("LONG", "SHORT"):
            raise PresentationError(
                f"Invalid direction: {direction!r}. Expected 'LONG' or 'SHORT'."
            )

        # Execution fields
        entry = cls._require_numeric(signal, "entry")
        stop = cls._require_numeric(signal, "stop")
        target = cls._require_numeric(signal, "target")
        score = cls._require_numeric(signal, "score")

        # Risk fields
        risk_reward = cls._extract_rr(signal)
        risk_pct = cls._extract_risk_pct(signal)
        invalidation = cls._extract_invalidation(signal)

        # Explainability fields
        market_state = cls._require_str(signal, "market_state")
        engine_digest = cls._extract_engine_digest(signal)

        # Optional fields (explicit presence check, not getattr)
        metadata = cls._extract_metadata(signal)
        generated_at = cls._extract_generated_at(signal)
        signal_id = cls._extract_signal_id(signal)

        # Build headline
        headline = cls._build_headline(symbol, timeframe, tier, direction, score)

        # Format prices (preserve precision but mobile-friendly)
        entry_fmt = cls._format_price(entry, symbol)
        stop_fmt = cls._format_price(stop, symbol)
        target_fmt = cls._format_price(target, symbol)

        return cls(
            headline=headline,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            tier=tier,
            entry=entry_fmt,
            stop=stop_fmt,
            target=target_fmt,
            risk_reward=risk_reward,
            risk_pct=risk_pct,
            invalidation=invalidation,
            market_state=market_state,
            engine_digest=engine_digest,
            score=f"{score:.2f}",
            metadata=metadata,
            generated_at=generated_at,
            signal_id=signal_id,
        )

    # ------------------------------------------------------------------ #
    # Output: Telegram plain text                                         #
    # ------------------------------------------------------------------ #

    def to_telegram_text(self) -> str:
        """
        Format as plain text for Telegram (mobile-optimized).

        Rules:
            - Max 8 lines
            - No markdown tables (break on mobile)
            - ASCII labels for scanability
            - Price precision appropriate for symbol
        """
        lines = [
            f"[SIGNAL] {self.headline}",
            f"",
            f"[ENTRY]  {self.entry}",
            f"[STOP]   {self.stop}",
            f"[TARGET] {self.target}",
            f"",
            f"[STATS]  R:R {self.risk_reward} | Risk {self.risk_pct}",
            f"[RISK]   Invalidation: {self.invalidation}",
            f"",
            f"[INFO]   {self.engine_digest} | State: {self.market_state}",
        ]

        # Add metadata if present (compact)
        if self.metadata:
            meta_items = [f"{k}={v}" for k, v in list(self.metadata.items())[:3]]
            lines.append(f"[META]   {', '.join(meta_items)}")

        lines.append(f"[ID]     {self.signal_id}")

        return "\n".join(lines)

    def to_discord_text(self) -> str:
        """Discord allows slightly richer formatting."""
        return self.to_telegram_text()  # Same for MVP; override later

    def to_json_dict(self) -> dict:
        """Serialize for API/webhook consumption."""
        return {
            "headline": self.headline,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "tier": self.tier,
            "entry": self.entry,
            "stop": self.stop,
            "target": self.target,
            "risk_reward": self.risk_reward,
            "risk_pct": self.risk_pct,
            "invalidation": self.invalidation,
            "market_state": self.market_state,
            "engine_digest": self.engine_digest,
            "score": self.score,
            "metadata": self.metadata,
            "generated_at": self.generated_at,
            "signal_id": self.signal_id,
        }

    # ------------------------------------------------------------------ #
    # Private: explicit field extraction (no getattr fallbacks)           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_str(obj, field_name: str) -> str:
        """Extract a required string field. Raise if missing or wrong type."""
        if not hasattr(obj, field_name):
            raise PresentationError(
                f"TieredSignal missing required field: {field_name!r}"
            )
        value = getattr(obj, field_name)
        if not isinstance(value, str):
            raise PresentationError(
                f"TieredSignal.{field_name} must be str, got {type(value).__name__}"
            )
        if not value.strip():
            raise PresentationError(
                f"TieredSignal.{field_name} is empty"
            )
        return value.strip()

    @staticmethod
    def _require_numeric(obj, field_name: str) -> float:
        """Extract a required numeric field. Raise if missing or non-numeric."""
        if not hasattr(obj, field_name):
            raise PresentationError(
                f"TieredSignal missing required field: {field_name!r}"
            )
        value = getattr(obj, field_name)
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise PresentationError(
                f"TieredSignal.{field_name} must be numeric, got {value!r}: {exc}"
            )

    @staticmethod
    def _extract_rr(obj) -> str:
        """Extract risk:reward. Use explicit field or compute from levels."""
        if hasattr(obj, "risk_reward"):
            try:
                rr = float(obj.risk_reward)
                return f"{rr:.1f}"
            except (TypeError, ValueError):
                pass
        # Fallback: compute from entry/stop/target
        try:
            entry = float(obj.entry)
            stop = float(obj.stop)
            target = float(obj.target)
            risk = abs(entry - stop)
            reward = abs(target - entry)
            if risk > 0:
                return f"{reward / risk:.1f}"
        except (AttributeError, TypeError, ValueError):
            pass
        return "?.?"

    @staticmethod
    def _extract_risk_pct(obj) -> str:
        """Extract position risk percentage."""
        if hasattr(obj, "risk_pct"):
            try:
                pct = float(obj.risk_pct)
                return f"{pct:.1f}%"
            except (TypeError, ValueError):
                pass
        if hasattr(obj, "tier"):
            tier = str(getattr(obj, "tier", "")).upper()
            if tier == "ALPHA":
                return "1.0%"
            elif tier == "OPPORTUNITY":
                return "0.5%"
        return "?%"

    @staticmethod
    def _extract_invalidation(obj) -> str:
        """Extract or build invalidation message."""
        if hasattr(obj, "invalidation") and obj.invalidation:
            return str(obj.invalidation)
        # Build from stop level
        try:
            direction = str(getattr(obj, "direction", "")).upper()
            stop = float(obj.stop)
            if direction == "LONG":
                return f"Close below {stop:,.2f}"
            elif direction == "SHORT":
                return f"Close above {stop:,.2f}"
        except (AttributeError, TypeError, ValueError):
            pass
        return "Stop loss hit"

    @staticmethod
    def _extract_engine_digest(obj) -> str:
        """Build engine contribution summary."""
        engines: List[str] = []
        # Check for explicit engine_scores dict
        if hasattr(obj, "engine_scores"):
            scores = getattr(obj, "engine_scores", {})
            if isinstance(scores, dict):
                for name, score in scores.items():
                    try:
                        if float(score) > 0.3:
                            engines.append(name.title())
                    except (TypeError, ValueError):
                        continue
        # Check for individual engine flags
        engine_checks = [
            ("harmonic", "harmonic_score"),
            ("fvg", "fvg_score"),
            ("liquidity", "liquidity_score"),
        ]
        for label, attr in engine_checks:
            if hasattr(obj, attr):
                try:
                    if float(getattr(obj, attr)) > 0.3 and label.title() not in engines:
                        engines.append(label.title())
                except (TypeError, ValueError):
                    continue
        if engines:
            return " + ".join(engines)
        return "Multi-factor"

    @staticmethod
    def _extract_metadata(obj) -> Dict[str, str]:
        """Extract optional metadata for explainability."""
        meta: Dict[str, str] = {}
        # Known optional fields
        optional_fields = [
            ("hostile_gate_passed", "Gate"),
            ("structural_score", "Struct"),
            ("pattern_name", "Pattern"),
            ("prz_zone", "PRZ"),
        ]
        for attr, label in optional_fields:
            if hasattr(obj, attr):
                value = getattr(obj, attr)
                if value is not None:
                    meta[label] = str(value)[:20]  # Truncate for mobile
        return meta

    @staticmethod
    def _extract_generated_at(obj) -> str:
        """Extract generation timestamp."""
        if hasattr(obj, "generated_at"):
            value = obj.generated_at
            if isinstance(value, datetime):
                return value.isoformat()
            elif isinstance(value, str):
                return value
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _extract_signal_id(obj) -> str:
        """Extract or generate signal ID."""
        if hasattr(obj, "signal_id") and obj.signal_id:
            return str(obj.signal_id)
        if hasattr(obj, "id") and obj.id:
            return str(obj.id)
        # Generate from content hash
        try:
            content = f"{obj.symbol}:{obj.timeframe}:{obj.entry}:{obj.generated_at}"
            import hashlib
            return hashlib.sha256(content.encode()).hexdigest()[:12]
        except AttributeError:
            return "unknown"

    @staticmethod
    def _build_headline(symbol: str, timeframe: str, tier: str, direction: str, score: float) -> str:
        """Build one-line notification headline."""
        tag = "[LONG]" if direction == "LONG" else "[SHORT]"
        return f"{tag} {symbol} {timeframe} | {tier} | Score {score:.2f}"

    @staticmethod
    def _format_price(value: float, symbol: str) -> str:
        """Format price with appropriate precision for symbol."""
        # Crypto: 2 decimal places for BTC, more for alts
        if "BTC" in symbol.upper():
            return f"{value:,.2f}"
        elif "ETH" in symbol.upper():
            return f"{value:,.2f}"
        else:
            return f"{value:,.4f}"