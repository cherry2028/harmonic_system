"""
delivery/telegram_formatter.py
================================
TelegramFormatter - Signal Formatting and Delivery

Architectural position:
    Final layer before the subscriber. Called by pipeline.py after
    all gates pass, all scoring is complete, and a TieredSignal exists.

Two distinct responsibilities, intentionally separated:
    format_signal(signal) → str
        Pure function. No I/O. No side effects. Fully testable.
        Converts a TieredSignal into a human-readable Telegram message.
        Same input → identical output always.

    send(signal) → SendResult
        Calls format_signal(), then POSTs to Telegram Bot API.
        In dry_run mode: formats, logs, returns success without HTTP call.
        Never raises. Returns SendResult with success/failure detail.

Zero coupling to pipeline:
    This module does NOT import from pipeline.py.
    It does NOT import from signals/gate.py, signals/tier.py, or
    scoring/pattern_scorer.py.
    Its only project imports are:
        signals.signal.TieredSignal  - the input type
        config.market_state_config   - for tier badge lookup

Message design philosophy:
    Plain text with Unicode symbols - no HTML, no Markdown.
    Reason: Telegram HTML mode fails silently on unescaped characters.
    Plain text is always safe. Symbols provide visual structure.

    Structure (top to bottom):
        Header:    Tier badge + tier label + pattern + symbol
        Levels:    Entry, stop (with % distance), targets, R:R
        Reasoning: Full 7-line chain from TieredSignal
        Footer:    Risk allocation + frequency cap

    The reasoning chain is the product.
    A subscriber who sees "Buy at 61500" without explanation
    will not know whether to trust it next time it fires.
    A subscriber who sees the full 7-line chain understands
    exactly what the system saw and why it fired.
    Transparency builds retention.

dry_run mode:
    Controlled by the dry_run flag at construction.
    When True: format_signal() is called (tests formatting),
               the message is logged at INFO level,
               SendResult(success=True, dry_run=True) is returned.
               No HTTP call is made. No token needed.
    When False: real HTTP POST to Telegram Bot API.

Never-raise contract:
    send()         returns SendResult. Never raises.
    format_signal() returns str. Never raises (defensive formatting).

HTTP implementation:
    Uses urllib.request (stdlib). No external dependencies.
    Timeout: 10 seconds. Pipeline never waits longer.
    Retry: NOT implemented here. A single delivery attempt.
           Retry logic belongs in the pipeline layer, not the formatter.
           The formatter reports success/failure - the pipeline decides
           whether to retry (Phase 3 enhancement).

Bot token / chat_id:
    Read from environment variables:
        TELEGRAM_BOT_TOKEN  → the bot's API token
        TELEGRAM_CHANNEL_ID → the target channel or chat ID
    Overridable at construction for testing.
    Missing token in dry_run=False mode → SendResult(success=False).

Dependencies:
    urllib.request, urllib.parse - stdlib HTTP
    json, os, logging, dataclasses - stdlib
    signals.signal.TieredSignal - input contract only
    config.market_state_config  - tier badge table

    MUST NOT import from:
        pipeline.py, signals/gate.py, signals/tier.py,
        scoring/, market_state/, telemetry/
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from signals.signal import TieredSignal

logger = logging.getLogger("delivery.telegram_formatter")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Telegram Bot API endpoint template
_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

# HTTP timeout for all Telegram API calls
_HTTP_TIMEOUT: int = 10

# Visual separator line (24 Unicode box-drawing chars)
_SEP = "━" * 24

# Tier display configuration: badge, label
# Kept here (not in config) because this is pure display logic,
# not business configuration. Display changes should not require
# touching MarketStateConfig.
_TIER_DISPLAY = {
    "A+": ("🔴", "TIER A+  PREMIUM"),
    "A":  ("🔵", "TIER A   HIGH CONVICTION"),
    "B":  ("🟡", "TIER B   MODERATE"),
    "C":  ("⚪", "TIER C   EDUCATIONAL  |  Paper only"),
}

_DEFAULT_BADGE = ("⬜", "TIER ?   UNKNOWN")


# ---------------------------------------------------------------------------
# SendResult
# ---------------------------------------------------------------------------

@dataclass
class SendResult:
    """
    Complete record of one send attempt.

    Returned by TelegramFormatter.send(). Never raises.

    Fields:
        success  : True if the message was delivered (or dry_run).
                   False on any HTTP error, API error, or missing config.
        message  : The formatted message text - always populated,
                   even on failure. Enables callers to log what would
                   have been sent without re-formatting.
        error    : Human-readable error description on failure.
                   None on success.
        dry_run  : True when no real HTTP call was made.
        chat_id  : The target chat/channel ID that was used.
                   Empty string when config was missing.

    Invariants (enforced in __post_init__):
        success=True  → error is None
        success=False → error is not None and not empty
        message is always a non-empty string
    """

    success: bool
    message: str
    error:   Optional[str]
    dry_run: bool
    chat_id: str = ""

    def __post_init__(self) -> None:
        # message must always be a non-empty string
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError(
                "SendResult.message must be a non-empty string. "
                "The formatted message must always be present for audit."
            )
        # success=True → no error
        if self.success and self.error is not None:
            raise ValueError(
                f"SendResult: success=True but error={self.error!r}. "
                "A successful result must have error=None."
            )
        # success=False → error must explain why
        if not self.success and (
            not isinstance(self.error, str) or not self.error.strip()
        ):
            raise ValueError(
                "SendResult: success=False requires a non-empty error string. "
                "Callers need to know why delivery failed."
            )

    @property
    def delivered(self) -> bool:
        """Alias for success. More readable in pipeline conditionals."""
        return self.success

    def __repr__(self) -> str:
        if self.success:
            mode = "DRY_RUN" if self.dry_run else "SENT"
            return f"SendResult({mode} | chat={self.chat_id!r})"
        return f"SendResult(FAILED | {self.error!r})"


# ---------------------------------------------------------------------------
# TelegramFormatter
# ---------------------------------------------------------------------------

class TelegramFormatter:
    """
    Formats TieredSignal objects into Telegram messages and sends them.

    Stateless after construction. Thread-safe: format_signal() and send()
    can be called concurrently from multiple threads.

    Args:
        bot_token : Telegram Bot API token.
                    Defaults to TELEGRAM_BOT_TOKEN environment variable.
        chat_id   : Target channel or chat ID (e.g. "-1001234567890").
                    Defaults to TELEGRAM_CHANNEL_ID environment variable.
        dry_run   : When True, format but do not send. Default False.
                    Override to True in tests and staging environments.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id:   Optional[str] = None,
        dry_run:   bool          = False,
    ) -> None:
        self._bot_token = (
            str(bot_token).strip()
            if bot_token is not None
            else str(os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        )

        self._chat_id = (
            str(chat_id).strip()
            if chat_id is not None
            else str(os.getenv("TELEGRAM_CHANNEL_ID", "")).strip()
        )
        self._dry_run   = dry_run

        logger.debug(
            f"TelegramFormatter initialized | "
            f"dry_run={self._dry_run} | "
            f"chat_id={self._chat_id!r} | "
            f"token_set={bool(self._bot_token)}"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def format_signal(self, signal: TieredSignal) -> str:
        """
        Converts a TieredSignal into a formatted Telegram message string.

        Pure function:
            - No I/O
            - No side effects
            - Same input → identical output
            - Never raises (defensive formatting for every field)

        Args:
            signal: TieredSignal from SignalTier.classify()

        Returns:
            Formatted message string, always non-empty.
            On any error: returns a minimal safe fallback string.
        """
        try:
            return self._build_message(signal)
        except Exception as e:
            logger.error(
                f"format_signal() raised unexpectedly: "
                f"{type(e).__name__}: {e}. "
                f"Returning fallback message.",
                exc_info=True,
            )
            return self._fallback_message(signal)

    def send(self, signal: TieredSignal) -> SendResult:
        """
        Formats and sends a TieredSignal to Telegram.

        Steps:
            1. Format the message (always, even in dry_run)
            2. If dry_run: log and return success without HTTP call
            3. If no token/chat_id: return failure with clear error
            4. POST to Telegram Bot API
            5. Return SendResult with success/failure detail

        Never raises. Returns SendResult always.

        Args:
            signal: TieredSignal to deliver.

        Returns:
            SendResult describing the delivery outcome.
        """
        try:
            return self._send(signal)
        except Exception as e:
            logger.error(
                f"send() raised unexpectedly: {type(e).__name__}: {e}",
                exc_info=True,
            )
            # Build a safe fallback message for the SendResult
            try:
                msg = self.format_signal(signal)
            except Exception:
                msg = "[message formatting failed]"
            return SendResult(
                success = False,
                message = msg,
                error   = f"Unexpected error: {type(e).__name__}: {e}",
                dry_run = self._dry_run,
                chat_id = self._chat_id,
            )

    # ── Core implementation ───────────────────────────────────────────────

    def _send(self, signal: TieredSignal) -> SendResult:
        """Core send logic. May raise - caller wraps in try/except."""

        # Step 1: Format (always)
        message = self.format_signal(signal)

        # Step 2: dry_run mode - no HTTP
        if self._dry_run:
            logger.info(
                f"[DRY RUN] Signal formatted | "
                f"tier={signal.tier} "
                f"pattern={signal.pattern_name} "
                f"symbol={signal.symbol} {signal.timeframe}\n"
                f"{message}"
            )
            return SendResult(
                success = True,
                message = message,
                error   = None,
                dry_run = True,
                chat_id = self._chat_id,
            )

        # Step 3: Config check

        if not self._bot_token or str(self._bot_token).strip() == "":
            error = (
                "TELEGRAM_BOT_TOKEN is not set. "
                "Set the environment variable or pass bot_token at construction."
            )
            logger.error(error)

            return SendResult(
                success=False,
                message=message,
                error=error,
                dry_run=False,
                chat_id=self._chat_id,
            )

        if not self._chat_id or str(self._chat_id).strip() == "":
            error = (
                "TELEGRAM_CHANNEL_ID is not set. "
                "Set the environment variable or pass chat_id at construction."
            )
            logger.error(error)

            return SendResult(
                success=False,
                message=message,
                error=error,
                dry_run=False,
                chat_id=self._chat_id,
            )

        # Step 4: HTTP POST
        return self._http_post(message)

    def _http_post(self, message: str) -> SendResult:
        """
        Makes the Telegram Bot API POST request.

        Uses urllib.request (stdlib). No external dependencies.
        Timeout: _HTTP_TIMEOUT seconds.

        Returns SendResult. Does NOT raise.
        """
        url  = _API_URL.format(token=self._bot_token)
        data = urllib.parse.urlencode({
            "chat_id": self._chat_id,
            "text":    message,
        }).encode("utf-8")

        try:
            req  = urllib.request.Request(
                url,
                data    = data,
                method  = "POST",
                headers = {"Content-Type": "application/x-www-form-urlencoded"},
            )
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
                body    = resp.read().decode("utf-8")
                api_res = json.loads(body)

            if api_res.get("ok"):
                msg_id = api_res.get("result", {}).get("message_id", "?")
                logger.info(
                    f"Telegram delivery success | "
                    f"message_id={msg_id} | "
                    f"chat_id={self._chat_id!r}"
                )
                return SendResult(
                    success = True,
                    message = message,
                    error   = None,
                    dry_run = False,
                    chat_id = self._chat_id,
                )
            else:
                # Telegram returned ok=False
                error = (
                    f"Telegram API error {api_res.get('error_code', '?')}: "
                    f"{api_res.get('description', 'unknown error')}"
                )
                logger.error(f"Telegram API rejected message: {error}")
                return SendResult(
                    success = False,
                    message = message,
                    error   = error,
                    dry_run = False,
                    chat_id = self._chat_id,
                )

        except urllib.error.HTTPError as e:
            error = f"HTTP {e.code}: {e.reason}"
            logger.error(f"Telegram HTTP error: {error}")
            return SendResult(
                success = False,
                message = message,
                error   = error,
                dry_run = False,
                chat_id = self._chat_id,
            )
        except urllib.error.URLError as e:
            error = f"Network error: {e.reason}"
            logger.error(f"Telegram network error: {error}")
            return SendResult(
                success = False,
                message = message,
                error   = error,
                dry_run = False,
                chat_id = self._chat_id,
            )
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            logger.error(f"Telegram unexpected error: {error}", exc_info=True)
            return SendResult(
                success = False,
                message = message,
                error   = error,
                dry_run = False,
                chat_id = self._chat_id,
            )

    # ── Message construction ──────────────────────────────────────────────

    def _build_message(self, signal: TieredSignal) -> str:
        """
        Builds the complete formatted message string.

        Structure:
            Line 1:  Tier badge + tier label
            Line 2:  Pattern direction | Symbol Timeframe
            Line 3:  Separator
            Lines 4+: Trading levels (entry, stop, targets, R:R)
            Separator
            Lines N+: Reasoning chain
            Separator
            Last:    Operational context (risk, cap)

        Every field access is wrapped in getattr with a safe default.
        A missing field produces "N/A" in the output - never an exception.
        """
        # ── Header ────────────────────────────────────────────────────
        badge, tier_label = _TIER_DISPLAY.get(
            getattr(signal, "tier", "?"), _DEFAULT_BADGE
        )
        pattern   = getattr(signal, "pattern_name", "?")
        direction = getattr(signal, "direction",    "?").upper()
        symbol    = getattr(signal, "symbol",       "?")
        timeframe = getattr(signal, "timeframe",    "?")

        header = (
            f"{badge} {tier_label}\n"
            f"{pattern} {direction}  │  {symbol} {timeframe}"
        )

        # ── Trading levels ────────────────────────────────────────────
        entry  = getattr(signal, "entry",   None)
        stop   = getattr(signal, "stop",    None)
        t1     = getattr(signal, "target1", None)
        t2     = getattr(signal, "target2", None)
        t3     = getattr(signal, "target3", None)
        rr     = getattr(signal, "risk_reward", None)

        entry_str  = self._fmt_price(entry)
        stop_str   = self._fmt_price(stop)
        t1_str     = self._fmt_price(t1)
        rr_str     = f"{rr:.2f}" if rr is not None else "N/A"

        # Stop distance as percentage
        stop_pct_str = ""
        if entry is not None and stop is not None and entry > 0:
            stop_pct = abs((stop - entry) / entry * 100)
            direction_sym = "↓" if stop < entry else "↑"
            stop_pct_str = f" ({direction_sym}{stop_pct:.2f}%)"

        levels = (
            f"📍 Entry    {entry_str}\n"
            f"🛑 Stop     {stop_str}{stop_pct_str}\n"
            f"🎯 Target 1 {t1_str}  │  R:R {rr_str}"
        )

        # Optional additional targets
        optional = []
        if t2 is not None:
            optional.append(f"🎯 Target 2 {self._fmt_price(t2)}")
        if t3 is not None:
            optional.append(f"🎯 Target 3 {self._fmt_price(t3)}")
        if optional:
            levels += "\n" + "\n".join(optional)

        # ── Reasoning chain ───────────────────────────────────────────
        reasoning_lines = getattr(signal, "reasoning", [])
        if reasoning_lines:
            reasoning_block = "\n".join(
                f"  {line}" for line in reasoning_lines
            )
        else:
            reasoning_block = "  (no reasoning available)"

        # ── Footer ────────────────────────────────────────────────────
        risk_pct    = getattr(signal, "risk_pct",    0.0)
        max_per_day = getattr(signal, "max_per_day", 0)
        edge_score  = getattr(signal, "edge_score",  0.0)

        if risk_pct == 0.0:
            risk_str = "Paper only - no real capital"
        else:
            risk_str = f"Risk {risk_pct}% of capital"

        footer = (
            f"{risk_str}  │  Max {max_per_day}/day\n"
            f"Edge score: {edge_score:.0%}"
        )

        # ── Assemble ──────────────────────────────────────────────────
        parts = [
            header,
            _SEP,
            levels,
            _SEP,
            "📊 Why this signal:",
            reasoning_block,
            _SEP,
            footer,
        ]
        return "\n".join(parts)

    # ── Formatting helpers ────────────────────────────────────────────────

    @staticmethod
    def _fmt_price(price: Optional[float]) -> str:
        """
        Formats a price value for display.

        Rules:
            None            → "N/A"
            >= 1000         → comma-separated, 2 decimal places
            >= 1            → 4 decimal places
            < 1             → 6 decimal places (for altcoins)

        Examples:
            61500.0  → "61,500.00"
            3.14159  → "3.1416"
            0.00042  → "0.000420"
        """
        if price is None:
            return "N/A"
        try:
            price = float(price)
            if price >= 1000:
                return f"{price:,.2f}"
            elif price >= 1:
                return f"{price:.4f}"
            else:
                return f"{price:.6f}"
        except (TypeError, ValueError):
            return "N/A"

    @staticmethod
    def _fallback_message(signal) -> str:
        """
        Minimal safe fallback when _build_message() raises.
        Always returns a non-empty, deliverable string.
        """
        try:
            tier   = getattr(signal, "tier",        "?")
            symbol = getattr(signal, "symbol",      "?")
            tf     = getattr(signal, "timeframe",   "?")
            edge   = getattr(signal, "edge_score",  0.0)
            return (
                f"ALERT Signal Alert\n"
                f"Tier: {tier} | {symbol} {tf}\n"
                f"Edge: {edge:.0%}\n"
                f"[Full formatting unavailable - check logs]"
            )
        except Exception:
            return "ALERT Signal Alert\n[Formatting error - check logs]"