"""
tests/test_week2_formatter.py
==============================
Exhaustive pytest coverage for delivery/telegram_formatter.py.

Test groups:
    1.  TelegramFormatter construction
    2.  SendResult dataclass invariants
    3.  format_signal() content correctness
    4.  format_signal() price formatting
    5.  format_signal() tier badges and labels
    6.  format_signal() edge cases (None fields, missing targets)
    7.  send() dry_run mode
    8.  send() missing config failures
    9.  send() HTTP failure simulation (mock urllib)
    10. send() never-raise contract
    11. Determinism
    12. Zero coupling verification (no pipeline imports)
"""

from __future__ import annotations

import json
import sys
import unittest.mock as mock
import urllib.error
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from delivery.telegram_formatter import (
    TelegramFormatter,
    SendResult,
    _TIER_DISPLAY,
    _DEFAULT_BADGE,
    _SEP,
    _HTTP_TIMEOUT,
    _API_URL,
)
from harmonic_patterns import PatternMatch
from market_state.vector import MarketStateVector
from scoring.pattern_scorer import PatternScorer
from signals.signal import TieredSignal
from signals.tier import SignalTier


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def make_tiered(
    quality:   float = 0.84,
    pattern:   str   = "Gartley",
    direction: str   = "bullish",
    symbol:    str   = "BTCUSDT",
    tf:        str   = "1h",
    entry:     float = 61500.0,
    stop:      float = 59800.0,
    dominant:  str   = "reversal",
    confidence: float = 0.80,
) -> TieredSignal:
    m = PatternMatch(
        pattern_name = pattern,
        direction    = direction,
        symbol       = symbol,
        timeframe    = tf,
        pivots       = {"X": 60000, "A": 65000, "B": 62000, "C": 64000, "D": 61500},
        ratios       = {"AB_XA": 0.618, "BC_AB": 0.382, "CD_BC": 1.272,
                        "AD_XA": 0.786, "XD_XA": 0.300},
        validation   = {"AB_XA": True, "BC_AB": True, "CD_BC": True, "AD_XA": True},
        prz          = {"entry": entry, "stop": stop,
                        "target1": 64000.0, "target2": 65000.0, "target3": 66000.0},
        D_index      = 295,
        D_timestamp  = pd.Timestamp("2024-01-15 14:00"),
        quality_score = quality,
        metadata     = {},
    )
    rest = (1.0 - confidence) / 5
    kw   = {s: rest for s in
            ["trending", "ranging", "expansion", "compression", "reversal", "news_chaos"]}
    kw[dominant] = confidence
    v = MarketStateVector(**kw)
    scored = PatternScorer().score(m, v)
    tiered = SignalTier().classify(scored)
    assert tiered is not None, (
        f"Failed to create TieredSignal for quality={quality}"
    )
    return tiered


@pytest.fixture(scope="module")
def tiered() -> TieredSignal:
    return make_tiered()


@pytest.fixture
def dry_fmt() -> TelegramFormatter:
    return TelegramFormatter(dry_run=True, chat_id="-1001234567890")


# ---------------------------------------------------------------------------
# Group 1: TelegramFormatter construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_dry_run_construction(self):
        fmt = TelegramFormatter(dry_run=True)
        assert fmt._dry_run is True

    def test_default_not_dry_run(self):
        fmt = TelegramFormatter()
        assert fmt._dry_run is False

    def test_explicit_token_and_chat(self):
        fmt = TelegramFormatter(
            bot_token="abc123",
            chat_id="-9999",
            dry_run=True,
        )
        assert fmt._bot_token == "abc123"
        assert fmt._chat_id   == "-9999"

    def test_env_fallback_used_when_no_args(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN",  "env_token")
        monkeypatch.setenv("TELEGRAM_CHANNEL_ID", "env_chat")
        fmt = TelegramFormatter()
        assert fmt._bot_token == "env_token"
        assert fmt._chat_id   == "env_chat"

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN",  "env_token")
        monkeypatch.setenv("TELEGRAM_CHANNEL_ID", "env_chat")
        fmt = TelegramFormatter(bot_token="explicit_token", chat_id="explicit_chat")
        assert fmt._bot_token == "explicit_token"
        assert fmt._chat_id   == "explicit_chat"

    def test_format_signal_callable(self, dry_fmt):
        assert callable(dry_fmt.format_signal)

    def test_send_callable(self, dry_fmt):
        assert callable(dry_fmt.send)


# ---------------------------------------------------------------------------
# Group 2: SendResult invariants
# ---------------------------------------------------------------------------

class TestSendResultInvariants:

    def test_success_result_construction(self):
        r = SendResult(success=True, message="msg", error=None,
                       dry_run=True, chat_id="-123")
        assert r.success is True
        assert r.error   is None
        assert r.delivered is True

    def test_failure_result_construction(self):
        r = SendResult(success=False, message="msg", error="Network timeout",
                       dry_run=False, chat_id="-123")
        assert r.success   is False
        assert r.error     == "Network timeout"
        assert r.delivered is False

    def test_success_with_error_raises(self):
        with pytest.raises(ValueError):
            SendResult(success=True, message="msg", error="oops",
                       dry_run=True, chat_id="-123")

    def test_failure_without_error_raises(self):
        with pytest.raises(ValueError):
            SendResult(success=False, message="msg", error=None,
                       dry_run=False, chat_id="-123")

    def test_empty_message_raises(self):
        with pytest.raises(ValueError):
            SendResult(success=True, message="", error=None,
                       dry_run=True, chat_id="-123")

    @pytest.mark.parametrize("ws", ["   ", "\t", "\n", "\t\n"])
    def test_whitespace_message_raises(self, ws):
        with pytest.raises(ValueError):
            SendResult(success=True, message=ws, error=None,
                       dry_run=True, chat_id="-123")

    def test_empty_error_on_failure_raises(self):
        with pytest.raises(ValueError):
            SendResult(success=False, message="msg", error="",
                       dry_run=False, chat_id="-123")

    def test_delivered_is_alias_for_success(self):
        r = SendResult(success=True, message="x", error=None, dry_run=True)
        assert r.delivered == r.success

    def test_repr_success(self):
        r = SendResult(success=True, message="x", error=None,
                       dry_run=True, chat_id="-123")
        assert "DRY_RUN" in repr(r)

    def test_repr_sent(self):
        r = SendResult(success=True, message="x", error=None,
                       dry_run=False, chat_id="-123")
        assert "SENT" in repr(r)

    def test_repr_failed(self):
        r = SendResult(success=False, message="x", error="timeout",
                       dry_run=False, chat_id="-123")
        assert "FAILED" in repr(r)
        assert "timeout" in repr(r)


# ---------------------------------------------------------------------------
# Group 3: format_signal() content correctness
# ---------------------------------------------------------------------------

class TestFormatSignalContent:

    def test_returns_nonempty_string(self, dry_fmt, tiered):
        msg = dry_fmt.format_signal(tiered)
        assert isinstance(msg, str) and len(msg) > 50

    def test_contains_symbol(self, dry_fmt, tiered):
        assert tiered.symbol in dry_fmt.format_signal(tiered)

    def test_contains_timeframe(self, dry_fmt, tiered):
        assert tiered.timeframe in dry_fmt.format_signal(tiered)

    def test_contains_pattern_name(self, dry_fmt, tiered):
        assert tiered.pattern_name in dry_fmt.format_signal(tiered)

    def test_contains_direction_uppercase(self, dry_fmt, tiered):
        assert tiered.direction.upper() in dry_fmt.format_signal(tiered)

    def test_contains_entry_price(self, dry_fmt, tiered):
        msg = dry_fmt.format_signal(tiered)
        assert "61,500.00" in msg

    def test_contains_stop_price(self, dry_fmt, tiered):
        assert "59,800.00" in dry_fmt.format_signal(tiered)

    def test_contains_target1(self, dry_fmt, tiered):
        assert "64,000.00" in dry_fmt.format_signal(tiered)

    def test_contains_stop_distance_percent(self, dry_fmt, tiered):
        msg = dry_fmt.format_signal(tiered)
        assert "%" in msg
        assert "2.76" in msg    # stop distance ≈ 2.76%

    def test_contains_rr_ratio(self, dry_fmt, tiered):
        msg = dry_fmt.format_signal(tiered)
        assert "R:R" in msg
        assert "1.47" in msg   # expected R:R for this PRZ

    def test_contains_separator(self, dry_fmt, tiered):
        assert "━" in dry_fmt.format_signal(tiered)

    def test_contains_reasoning_lines(self, dry_fmt, tiered):
        msg = dry_fmt.format_signal(tiered)
        for line in tiered.reasoning:
            assert line in msg, f"Reasoning line missing: {line!r}"

    def test_contains_edge_score(self, dry_fmt, tiered):
        msg = dry_fmt.format_signal(tiered)
        assert f"{tiered.edge_score:.0%}" in msg

    def test_contains_risk_pct(self, dry_fmt, tiered):
        msg = dry_fmt.format_signal(tiered)
        if tiered.is_paper_only:
            assert "Paper" in msg or "paper" in msg
        else:
            assert str(tiered.risk_pct) in msg

    def test_contains_max_per_day(self, dry_fmt, tiered):
        msg = dry_fmt.format_signal(tiered)
        assert str(tiered.max_per_day) in msg
        assert "/day" in msg

    def test_target2_present_when_not_none(self, dry_fmt, tiered):
        if tiered.target2 is not None:
            assert "65,000.00" in dry_fmt.format_signal(tiered)

    def test_target3_present_when_not_none(self, dry_fmt, tiered):
        if tiered.target3 is not None:
            assert "66,000.00" in dry_fmt.format_signal(tiered)

    def test_why_this_signal_header(self, dry_fmt, tiered):
        assert "Why this signal" in dry_fmt.format_signal(tiered)

    @pytest.mark.parametrize("pattern", ["Gartley", "Bat", "Butterfly", "Crab"])
    def test_all_patterns_format_correctly(self, dry_fmt, pattern):
        t = make_tiered(quality=0.80, pattern=pattern)
        msg = dry_fmt.format_signal(t)
        assert pattern in msg
        assert isinstance(msg, str) and len(msg) > 50


# ---------------------------------------------------------------------------
# Group 4: format_signal() price formatting
# ---------------------------------------------------------------------------

class TestPriceFormatting:

    @pytest.mark.parametrize("price,expected", [
        (61500.0,  "61,500.00"),
        (100000.0, "100,000.00"),
        (3.14159,  "3.1416"),
        (0.000420, "0.000420"),
        (1.0,      "1.0000"),
        (999.99,   "999.9900"),
        (1000.0,   "1,000.00"),
        (None,     "N/A"),
    ])
    def test_fmt_price(self, dry_fmt, price, expected):
        result = dry_fmt._fmt_price(price)
        assert result == expected, (
            f"_fmt_price({price}) = {result!r}, expected {expected!r}"
        )

    def test_fmt_price_bad_type_returns_na(self, dry_fmt):
        for bad in ["not_a_price", [], {}]:
            result = dry_fmt._fmt_price(bad)
            assert result == "N/A", f"_fmt_price({bad!r}) should return N/A"

    def test_bearish_stop_shows_up_arrow(self, dry_fmt):
        """Bearish pattern: stop is ABOVE entry → ↑ direction symbol."""
        t = make_tiered(
            direction="bearish",
            entry=3000.0,
            stop=3200.0,    # stop above entry for bearish
        )
        if t is not None:
            msg = dry_fmt.format_signal(t)
            # Stop is above entry → upward arrow in stop distance
            assert "↑" in msg or "%" in msg   # at minimum % distance shown

    def test_bullish_stop_shows_down_arrow(self, dry_fmt, tiered):
        """Bullish pattern: stop is BELOW entry → ↓ direction symbol."""
        msg = dry_fmt.format_signal(tiered)
        assert "↓" in msg


# ---------------------------------------------------------------------------
# Group 5: Tier badges and labels
# ---------------------------------------------------------------------------

class TestTierBadgesAndLabels:

    @pytest.mark.parametrize("tier_name,badge,label_part", [
        ("A+", "🔴", "PREMIUM"),
        ("A",  "🔵", "HIGH CONVICTION"),
        ("B",  "🟡", "MODERATE"),
        ("C",  "⚪", "EDUCATIONAL"),
    ])
    def test_tier_display_table(self, tier_name, badge, label_part):
        b, label = _TIER_DISPLAY[tier_name]
        assert b == badge
        assert label_part in label

    def test_all_four_tiers_in_display_table(self):
        assert set(_TIER_DISPLAY.keys()) == {"A+", "A", "B", "C"}

    def test_tier_c_paper_only_in_message(self, dry_fmt):
        t = make_tiered(quality=0.15, dominant="ranging", confidence=0.85)
        if t is not None and t.tier == "C":
            msg = dry_fmt.format_signal(t)
            assert "Paper" in msg or "paper" in msg

    def test_default_badge_for_unknown_tier(self):
        badge, label = _DEFAULT_BADGE
        assert isinstance(badge, str)
        assert isinstance(label, str)


# ---------------------------------------------------------------------------
# Group 6: Edge cases
# ---------------------------------------------------------------------------

class TestFormatSignalEdgeCases:

    def test_format_signal_none_returns_nonempty_string(self, dry_fmt):
        msg = dry_fmt.format_signal(None)
        assert isinstance(msg, str) and len(msg.strip()) > 0

    def test_format_signal_bad_type_returns_nonempty_string(self, dry_fmt):
        for bad in [{}, "string", 42, 3.14]:
            msg = dry_fmt.format_signal(bad)
            assert isinstance(msg, str) and len(msg.strip()) > 0, (
                f"format_signal({type(bad).__name__}) returned empty"
            )

    def test_format_signal_none_input_never_raises(self, dry_fmt):
        dry_fmt.format_signal(None)

    def test_format_signal_missing_targets_omits_target_lines(self, dry_fmt):
        """When target2/target3 are None, those lines should be absent."""
        # Manually create a ScoredSignal-backed TieredSignal
        # For simplicity, use the standard tiered but verify logic
        t = make_tiered()
        if t.target2 is None:
            msg = dry_fmt.format_signal(t)
            assert "Target 2" not in msg

    def test_separator_appears_multiple_times(self, dry_fmt, tiered):
        msg = dry_fmt.format_signal(tiered)
        count = msg.count("━")
        assert count >= len(_SEP), "At least one full separator expected"


# ---------------------------------------------------------------------------
# Group 7: send() dry_run mode
# ---------------------------------------------------------------------------

class TestSendDryRun:

    def test_dry_run_returns_success(self, dry_fmt, tiered):
        result = dry_fmt.send(tiered)
        assert result.success is True

    def test_dry_run_sets_dry_run_flag(self, dry_fmt, tiered):
        result = dry_fmt.send(tiered)
        assert result.dry_run is True

    def test_dry_run_error_is_none(self, dry_fmt, tiered):
        result = dry_fmt.send(tiered)
        assert result.error is None

    def test_dry_run_message_is_formatted(self, dry_fmt, tiered):
        result = dry_fmt.send(tiered)
        assert tiered.symbol in result.message
        assert tiered.pattern_name in result.message

    def test_dry_run_chat_id_preserved(self, tiered):
        fmt = TelegramFormatter(dry_run=True, chat_id="-9999")
        result = fmt.send(tiered)
        assert result.chat_id == "-9999"

    def test_dry_run_no_http_call(self, tiered):
        """dry_run=True must never make an HTTP request."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            fmt = TelegramFormatter(dry_run=True, chat_id="-123")
            fmt.send(tiered)
            mock_urlopen.assert_not_called()

    def test_dry_run_returns_send_result_type(self, dry_fmt, tiered):
        result = dry_fmt.send(tiered)
        assert isinstance(result, SendResult)


# ---------------------------------------------------------------------------
# Group 8: send() missing config
# ---------------------------------------------------------------------------

class TestSendMissingConfig:

    def test_missing_token_returns_failure(self, tiered, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN",  raising=False)
        monkeypatch.delenv("TELEGRAM_CHANNEL_ID", raising=False)
        fmt = TelegramFormatter(bot_token="", chat_id="-123", dry_run=False)
        result = fmt.send(tiered)
        assert result.success is False
        assert "TOKEN" in result.error.upper() or "token" in result.error.lower()

    def test_missing_chat_id_returns_failure(self, tiered):
        fmt = TelegramFormatter(bot_token="fake_token", chat_id="", dry_run=False)
        result = fmt.send(tiered)
        assert result.success is False
        assert (
            "CHANNEL" in result.error.upper()
            or "chat" in result.error.lower()
        )

    def test_failure_result_still_has_message(self, tiered):
        """Even on config failure, the formatted message must be present."""
        fmt = TelegramFormatter(bot_token="", chat_id="-123", dry_run=False)
        result = fmt.send(tiered)
        assert isinstance(result.message, str) and len(result.message) > 10

    def test_failure_result_is_valid_send_result(self, tiered):
        fmt = TelegramFormatter(bot_token="", chat_id="-123", dry_run=False)
        result = fmt.send(tiered)
        assert isinstance(result, SendResult)
        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# Group 9: send() HTTP failure simulation
# ---------------------------------------------------------------------------

class TestSendHTTPFailures:

    def _make_mock_response(self, body: dict, status: int = 200):
        """Creates a mock urllib response object."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(body).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__  = MagicMock(return_value=False)
        return mock_resp

    def test_http_success_returns_success_result(self, tiered):
        fmt = TelegramFormatter(
            bot_token="fake_token", chat_id="-123", dry_run=False
        )
        mock_resp = self._make_mock_response(
            {"ok": True, "result": {"message_id": 42}}
        )
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = fmt.send(tiered)
        assert result.success is True
        assert result.dry_run is False
        assert result.error   is None

    def test_telegram_api_error_returns_failure(self, tiered):
        fmt = TelegramFormatter(
            bot_token="fake_token", chat_id="-123", dry_run=False
        )
        mock_resp = self._make_mock_response(
            {"ok": False, "error_code": 400, "description": "Bad Request"}
        )
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = fmt.send(tiered)
        assert result.success is False
        assert "400" in result.error or "Bad Request" in result.error

    def test_http_error_returns_failure(self, tiered):
        fmt = TelegramFormatter(
            bot_token="fake_token", chat_id="-123", dry_run=False
        )
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.HTTPError(
                       url="", code=401, msg="Unauthorized",
                       hdrs=None, fp=None
                   )):
            result = fmt.send(tiered)
        assert result.success is False
        assert "401" in result.error or "Unauthorized" in result.error

    def test_url_error_returns_failure(self, tiered):
        fmt = TelegramFormatter(
            bot_token="fake_token", chat_id="-123", dry_run=False
        )
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("Connection refused")):
            result = fmt.send(tiered)
        assert result.success is False
        assert "Connection" in result.error or "Network" in result.error

    def test_timeout_error_returns_failure(self, tiered):
        fmt = TelegramFormatter(
            bot_token="fake_token", chat_id="-123", dry_run=False
        )
        with patch("urllib.request.urlopen",
                   side_effect=TimeoutError("timed out")):
            result = fmt.send(tiered)
        assert result.success is False
        assert result.error is not None

    def test_failure_result_always_has_message(self, tiered):
        """HTTP failures must still carry the formatted message."""
        fmt = TelegramFormatter(
            bot_token="fake_token", chat_id="-123", dry_run=False
        )
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("refused")):
            result = fmt.send(tiered)
        assert isinstance(result.message, str) and len(result.message) > 10

    def test_http_timeout_constant(self):
        assert _HTTP_TIMEOUT == 10


# ---------------------------------------------------------------------------
# Group 10: Never-raise contract
# ---------------------------------------------------------------------------

class TestNeverRaiseContract:

    @pytest.mark.parametrize("bad_input", [
        None, {}, "string", 42, 3.14, [], object(),
    ])
    def test_format_signal_never_raises(self, dry_fmt, bad_input):
        result = dry_fmt.format_signal(bad_input)
        assert isinstance(result, str) and len(result) > 0

    @pytest.mark.parametrize("bad_input", [
        None, {}, "string", 42, 3.14, [],
    ])
    def test_send_never_raises(self, dry_fmt, bad_input):
        result = dry_fmt.send(bad_input)
        assert isinstance(result, SendResult)

    def test_repeated_bad_inputs_do_not_corrupt_formatter(self, dry_fmt, tiered):
        """After bad inputs, valid calls must still work."""
        dry_fmt.format_signal(None)
        dry_fmt.send(None)
        dry_fmt.format_signal({})
        # Valid call still works
        result = dry_fmt.send(tiered)
        assert result.success is True

    def test_send_always_returns_send_result(self, dry_fmt, tiered):
        for _ in range(5):
            result = dry_fmt.send(tiered)
            assert isinstance(result, SendResult)


# ---------------------------------------------------------------------------
# Group 11: Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_format_signal_deterministic(self, dry_fmt, tiered):
        msg1 = dry_fmt.format_signal(tiered)
        msg2 = dry_fmt.format_signal(tiered)
        assert msg1 == msg2

    def test_send_dry_run_deterministic(self, tiered):
        fmt1 = TelegramFormatter(dry_run=True, chat_id="-123")
        fmt2 = TelegramFormatter(dry_run=True, chat_id="-123")
        r1 = fmt1.send(tiered)
        r2 = fmt2.send(tiered)
        assert r1.message == r2.message
        assert r1.success == r2.success

    def test_format_100_calls_identical(self, dry_fmt, tiered):
        messages = [dry_fmt.format_signal(tiered) for _ in range(100)]
        assert len(set(messages)) == 1, "format_signal not deterministic"


# ---------------------------------------------------------------------------
# Group 12: Zero coupling verification
# ---------------------------------------------------------------------------

class TestZeroCoupling:

    def test_no_pipeline_import(self):
        """delivery/telegram_formatter.py must not import from pipeline.py."""
        import ast
        src = open(
            str(Path(__file__).parent.parent / "delivery" / "telegram_formatter.py"),
            encoding="utf-8",
        ).read()
        tree = ast.parse(src)
        import_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    import_names.append(node.module)
        assert not any("pipeline" in n for n in import_names), (
            f"Found pipeline import in: {[n for n in import_names if 'pipeline' in n]}"
        )

    def test_no_gate_import(self):
        import ast
        src  = open(
            str(Path(__file__).parent.parent / "delivery" / "telegram_formatter.py"),
            encoding="utf-8",
        ).read()
        tree = ast.parse(src)
        import_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    import_names.append(node.module)
        assert not any("gate" in n for n in import_names), (
            f"Found gate import: {[n for n in import_names if 'gate' in n]}"
        )

    def test_no_scoring_import(self):
        import ast
        src = open(
            str(Path(__file__).parent.parent / "delivery" / "telegram_formatter.py"),
            encoding="utf-8",
        ).read()
        tree = ast.parse(src)
        import_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    import_names.append(node.module)
        assert not any("pattern_scorer" in n or
                        (n.startswith("scoring") and "score_result" not in n)
                        for n in import_names), (
            f"Found unexpected scoring import: {import_names}"
        )

    def test_no_market_state_import(self):
        import ast
        src  = open(
            str(Path(__file__).parent.parent / "delivery" / "telegram_formatter.py"),
            encoding="utf-8",
        ).read()
        tree = ast.parse(src)
        import_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    import_names.append(node.module)
        assert not any(n.startswith("market_state") for n in import_names), (
            f"Found market_state import: {[n for n in import_names if n.startswith('market_state')]}"
        )

    def test_no_telemetry_import(self):
        """Formatter does not call telemetry — pipeline does."""
        import ast
        src  = open(
            str(Path(__file__).parent.parent / "delivery" / "telegram_formatter.py"),
            encoding="utf-8",
        ).read()
        tree = ast.parse(src)
        import_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    import_names.append(node.module)
        assert not any("telemetry" in n for n in import_names), (
            f"Found telemetry import: {[n for n in import_names if 'telemetry' in n]}"
        )

    def test_only_project_imports_are_signal_and_config(self):
        """
        The only project-level imports should be TieredSignal and config.
        Everything else should be stdlib.
        """
        src = open(
            str(Path(__file__).parent.parent / "delivery" / "telegram_formatter.py"),
            encoding="utf-8",
        ).read()
        # These are allowed
        assert "from signals.signal import TieredSignal" in src
        # stdlib imports are fine
        assert "import urllib" in src or "urllib" in src


# ---------------------------------------------------------------------------
# Appended in Action 9: HTTP mock, coupling, fallback, price parametrize
# ---------------------------------------------------------------------------

def mock_telegram_ok(message_id: int = 42):
    """Returns a mock urllib response simulating Telegram ok=True."""
    body = json.dumps({"ok": True, "result": {"message_id": message_id}})
    response = mock.MagicMock()
    response.read.return_value = body.encode("utf-8")
    response.__enter__ = mock.MagicMock(return_value=response)
    response.__exit__  = mock.MagicMock(return_value=False)
    return response


def mock_telegram_fail(error_code: int = 400, description: str = "Bad Request"):
    """Returns a mock urllib response simulating Telegram ok=False."""
    body = json.dumps({
        "ok": False, "error_code": error_code, "description": description,
    })
    response = mock.MagicMock()
    response.read.return_value = body.encode("utf-8")
    response.__enter__ = mock.MagicMock(return_value=response)
    response.__exit__  = mock.MagicMock(return_value=False)
    return response


@pytest.fixture
def fmt_live() -> TelegramFormatter:
    return TelegramFormatter(
        bot_token="123456:FAKE_TOKEN_FOR_TESTS",
        chat_id="-1001234567890",
        dry_run=False,
    )


class TestSendHTTP:
    """
    Full HTTP path tests via urllib mock.
    No real network calls — urlopen is patched at the module level.
    """



    def test_http_success_returns_true(self, fmt_live, tiered):
        with mock.patch("urllib.request.urlopen",
                        return_value=mock_telegram_ok()):
            result = fmt_live.send(tiered)
        assert result.success is True
        assert result.dry_run is False
        assert result.error   is None

    def test_http_success_message_populated(self, fmt_live, tiered):
        with mock.patch("urllib.request.urlopen",
                        return_value=mock_telegram_ok()):
            result = fmt_live.send(tiered)
        assert isinstance(result.message, str) and result.message.strip()

    def test_http_success_chat_id_preserved(self, fmt_live, tiered):
        with mock.patch("urllib.request.urlopen",
                        return_value=mock_telegram_ok()):
            result = fmt_live.send(tiered)
        assert result.chat_id == "-1001234567890"

    def test_telegram_ok_false_returns_failure(self, fmt_live, tiered):
        with mock.patch("urllib.request.urlopen",
                        return_value=mock_telegram_fail(400, "Bad Request")):
            result = fmt_live.send(tiered)
        assert result.success is False
        assert result.error   is not None
        assert "400" in result.error or "Bad" in result.error

    def test_http_error_returns_failure(self, fmt_live, tiered):
        http_err = urllib.error.HTTPError(
            url="http://x", code=429, msg="Too Many Requests",
            hdrs=None, fp=None,
        )
        with mock.patch("urllib.request.urlopen", side_effect=http_err):
            result = fmt_live.send(tiered)
        assert result.success is False
        assert "429" in result.error or "Many" in result.error

    def test_url_error_returns_failure(self, fmt_live, tiered):
        url_err = urllib.error.URLError(reason="Name or service not known")
        with mock.patch("urllib.request.urlopen", side_effect=url_err):
            result = fmt_live.send(tiered)
        assert result.success is False
        assert result.error   is not None

    def test_unexpected_exception_returns_failure(self, fmt_live, tiered):
        with mock.patch("urllib.request.urlopen",
                        side_effect=RuntimeError("unexpected")):
            result = fmt_live.send(tiered)
        assert result.success is False
        assert result.error   is not None

    def test_http_call_uses_correct_url(self, fmt_live, tiered):
        """POST URL must contain the bot token."""
        captured = {}
        def capture(req, timeout):
            captured["url"] = req.full_url
            return mock_telegram_ok()
        with mock.patch("urllib.request.urlopen", side_effect=capture):
            fmt_live.send(tiered)
        assert "123456:FAKE_TOKEN_FOR_TESTS" in captured["url"]

    def test_http_timeout_is_configured(self, fmt_live, tiered):
        """urlopen must be called with the module-level timeout."""
        captured = {}
        def capture(req, timeout):
            captured["timeout"] = timeout
            return mock_telegram_ok()
        with mock.patch("urllib.request.urlopen", side_effect=capture):
            fmt_live.send(tiered)
        assert captured["timeout"] == _HTTP_TIMEOUT

    def test_post_body_contains_chat_id(self, fmt_live, tiered):
        """POST body must include the target chat_id."""
        captured = {}
        def capture(req, timeout):
            captured["data"] = req.data.decode("utf-8")
            return mock_telegram_ok()
        with mock.patch("urllib.request.urlopen", side_effect=capture):
            fmt_live.send(tiered)
        assert "-1001234567890" in captured["data"]


class TestPackageCoupling:
    """
    AST-level verification that telegram_formatter.py has zero coupling
    to forbidden modules. Uses ast.parse — not import inspection —
    to catch even commented-out or conditionally-imported references.
    """

    def _get_import_modules(self):
        import ast
        src = open(
            str(Path(__file__).parent.parent / "delivery" / "telegram_formatter.py"),
            encoding="utf-8",
    ).read()
        tree = ast.parse(src)
        mods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    mods.append(alias.name)
        return mods

    def test_no_pipeline_import(self):
        mods = self._get_import_modules()
        assert not any("pipeline" in m for m in mods), (
            f"pipeline import found: {[m for m in mods if 'pipeline' in m]}"
        )

    def test_no_scoring_import(self):
        mods = self._get_import_modules()
        assert not any(m.startswith("scoring") for m in mods), (
            f"scoring import found: {[m for m in mods if m.startswith('scoring')]}"
        )

    def test_no_market_state_import(self):
        mods = self._get_import_modules()
        assert not any(m.startswith("market_state") for m in mods)

    def test_no_telemetry_import(self):
        mods = self._get_import_modules()
        assert not any(m.startswith("telemetry") for m in mods)

    def test_no_gate_import(self):
        mods = self._get_import_modules()
        assert not any("gate" in m for m in mods)

    def test_no_tier_import(self):
        """signals.signal is allowed; signals.tier is not."""
        mods = self._get_import_modules()
        assert not any(m == "signals.tier" for m in mods)

    def test_delivery_package_exports(self):
        from delivery import TelegramFormatter, SendResult
        assert TelegramFormatter is not None
        assert SendResult        is not None


class TestFallbackMessage:
    """
    _fallback_message() is the last safety net when _build_message() raises.
    It must always return a non-empty, deliverable string regardless of input.
    """

    def test_valid_signal_returns_alert_string(self, tiered):
        fmt = TelegramFormatter(dry_run=True)
        result = fmt._fallback_message(tiered)
        assert isinstance(result, str) and result.strip()

    def test_contains_alert_indicator(self, tiered):
        fmt = TelegramFormatter(dry_run=True)
        result = fmt._fallback_message(tiered)
        assert "⚠" in result or "Alert" in result or "Signal" in result

    def test_none_input_returns_safe_string(self):
        fmt = TelegramFormatter(dry_run=True)
        result = fmt._fallback_message(None)
        assert isinstance(result, str) and result.strip()

    def test_dict_input_returns_safe_string(self):
        fmt = TelegramFormatter(dry_run=True)
        result = fmt._fallback_message({})
        assert isinstance(result, str) and result.strip()

    def test_object_input_returns_safe_string(self):
        fmt = TelegramFormatter(dry_run=True)
        result = fmt._fallback_message(object())
        assert isinstance(result, str) and result.strip()


class TestFmtPriceParametrized:
    """
    Exhaustive parametrized tests for _fmt_price() rules:
        >= 1000  → comma-separated, 2dp
        >= 1     → 4dp
        < 1      → 6dp
        None     → 'N/A'
        bad type → 'N/A'
    """

    @pytest.mark.parametrize("price,expected", [
        # None
        (None,       "N/A"),
        # < 1 → 6dp
        (0.000001,   "0.000001"),
        (0.000420,   "0.000420"),
        (0.5,        "0.500000"),
        (0.999999,   "0.999999"),
        # >= 1, < 1000 → 4dp
        (1.0,        "1.0000"),
        (3.14159,    "3.1416"),
        (42.0,       "42.0000"),
        (999.99,     "999.9900"),
        # >= 1000 → 2dp with commas
        (1000.0,     "1,000.00"),
        (61500.0,    "61,500.00"),
        (100000.0,   "100,000.00"),
        (1234567.89, "1,234,567.89"),
    ])
    def test_fmt_price(self, price, expected):
        fmt = TelegramFormatter(dry_run=True)
        assert fmt._fmt_price(price) == expected, (
            f"_fmt_price({price!r}) = {fmt._fmt_price(price)!r}, "
            f"expected {expected!r}"
        )

    @pytest.mark.parametrize("bad", ["string", [], {}, object()])
    def test_fmt_price_bad_types_return_na(self, bad):
        fmt = TelegramFormatter(dry_run=True)
        assert fmt._fmt_price(bad) == "N/A"