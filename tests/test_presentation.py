"""
tests/delivery/test_presentation.py
===================================
Test suite for SignalPresentation.

Coverage:
    - Happy path: full TieredSignal -> SignalPresentation
    - Minimal TieredSignal (only required fields)
    - Missing required fields -> PresentationError
    - Wrong types -> PresentationError
    - Empty strings -> PresentationError
    - Invalid direction -> PresentationError
    - Engine digest extraction from multiple schemas
    - Telegram output formatting
    - JSON serialization
    - Immutability (frozen dataclass)
    - No getattr() fallbacks in critical paths
"""

import pytest
from dataclasses import dataclass
from datetime import datetime, timezone

from delivery.presentation import SignalPresentation, PresentationError


# ============================================================================
# FIXTURES: Minimal TieredSignal duck-types
# ============================================================================

@dataclass
class FullTieredSignal:
    """Complete TieredSignal with all optional fields."""
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    direction: str = "LONG"
    tier: str = "alpha"
    entry: float = 50000.0
    stop: float = 49000.0
    target: float = 52000.0
    score: float = 0.72
    risk_reward: float = 2.1
    risk_pct: float = 1.0
    invalidation: str = "Close below 49000"
    market_state: str = "trending"
    engine_scores: dict = None
    hostile_gate_passed: bool = True
    structural_score: float = 0.65
    pattern_name: str = "Butterfly"
    generated_at: datetime = None
    signal_id: str = "sig-001"

    def __post_init__(self):
        if self.engine_scores is None:
            self.engine_scores = {"harmonic": 0.72, "momentum": 0.65}
        if self.generated_at is None:
            self.generated_at = datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)


@dataclass
class MinimalTieredSignal:
    """Bare minimum TieredSignal -- only required fields."""
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    direction: str = "SHORT"
    tier: str = "opportunity"
    entry: float = 50000.0
    stop: float = 51000.0
    target: float = 48000.0
    score: float = 0.55
    market_state: str = "ranging"
    # No optional fields at all


@dataclass
class LegacyTieredSignal:
    """Old-style signal with individual engine scores (no engine_scores dict)."""
    symbol: str = "ETHUSDT"
    timeframe: str = "4h"
    direction: str = "LONG"
    tier: str = "alpha"
    entry: float = 3000.0
    stop: float = 2950.0
    target: float = 3100.0
    score: float = 0.68
    risk_reward: float = 2.0
    risk_pct: float = 1.0
    invalidation: str = ""
    market_state: str = "trending"
    harmonic_score: float = 0.70
    momentum_score: float = 0.45
    fvg_score: float = 0.60
    generated_at: str = "2026-05-11T12:00:00Z"
    id: str = "legacy-001"


@dataclass
class BrokenTieredSignal:
    """Signal with missing required fields for error testing."""
    symbol: str = "BTCUSDT"
    # Missing: timeframe, direction, tier, entry, stop, target, score, market_state


@dataclass
class WrongTypeTieredSignal:
    """Signal with wrong types for error testing."""
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    direction: str = "LONG"
    tier: str = "alpha"
    entry: str = "not_a_number"  # Wrong type
    stop: float = 49000.0
    target: float = 52000.0
    score: float = 0.72
    market_state: str = "trending"


# ============================================================================
# HAPPY PATH TESTS
# ============================================================================

class TestHappyPath:
    def test_full_signal_conversion(self):
        signal = FullTieredSignal()
        pres = SignalPresentation.from_tiered_signal(signal)

        assert pres.symbol == "BTCUSDT"
        assert pres.timeframe == "1h"
        assert pres.direction == "LONG"
        assert pres.tier == "ALPHA"
        assert pres.entry == "50,000.00"
        assert pres.stop == "49,000.00"
        assert pres.target == "52,000.00"
        assert pres.score == "0.72"
        assert pres.risk_reward == "2.1"
        assert pres.risk_pct == "1.0%"
        assert pres.market_state == "trending"
        assert pres.engine_digest == "Harmonic + Momentum"
        assert pres.signal_id == "sig-001"
        assert pres.invalidation == "Close below 49000"

    def test_minimal_signal_conversion(self):
        signal = MinimalTieredSignal()
        pres = SignalPresentation.from_tiered_signal(signal)

        assert pres.symbol == "BTCUSDT"
        assert pres.direction == "SHORT"
        assert pres.tier == "OPPORTUNITY"
        assert pres.risk_pct == "0.5%"  # Derived from tier
        assert pres.engine_digest == "Multi-factor"  # No engine scores
        assert pres.invalidation == "Close above 51,000.00"  # Built from stop

    def test_legacy_signal_conversion(self):
        signal = LegacyTieredSignal()
        pres = SignalPresentation.from_tiered_signal(signal)

        assert pres.symbol == "ETHUSDT"
        assert pres.direction == "LONG"
        assert pres.engine_digest == "Harmonic + Fvg"  # From individual scores
        assert pres.risk_reward == "2.0"
        assert pres.signal_id == "legacy-001"  # From id field
        assert "Pattern=Butterfly" not in pres.to_telegram_text()  # No pattern_name


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    def test_missing_required_field_raises(self):
        signal = BrokenTieredSignal()
        with pytest.raises(PresentationError) as exc_info:
            SignalPresentation.from_tiered_signal(signal)
        assert "missing required field" in str(exc_info.value)

    def test_wrong_type_raises(self):
        signal = WrongTypeTieredSignal()
        with pytest.raises(PresentationError) as exc_info:
            SignalPresentation.from_tiered_signal(signal)
        assert "must be numeric" in str(exc_info.value)
        assert "entry" in str(exc_info.value)

    def test_empty_string_raises(self):
        @dataclass
        class EmptySignal:
            symbol: str = ""
            timeframe: str = "1h"
            direction: str = "LONG"
            tier: str = "alpha"
            entry: float = 50000.0
            stop: float = 49000.0
            target: float = 52000.0
            score: float = 0.72
            market_state: str = "trending"

        with pytest.raises(PresentationError) as exc_info:
            SignalPresentation.from_tiered_signal(EmptySignal())
        assert "is empty" in str(exc_info.value)

    def test_invalid_direction_raises(self):
        @dataclass
        class BadDirection:
            symbol: str = "BTCUSDT"
            timeframe: str = "1h"
            direction: str = "UP"  # Invalid
            tier: str = "alpha"
            entry: float = 50000.0
            stop: float = 49000.0
            target: float = 52000.0
            score: float = 0.72
            market_state: str = "trending"

        with pytest.raises(PresentationError) as exc_info:
            SignalPresentation.from_tiered_signal(BadDirection())
        assert "Invalid direction" in str(exc_info.value)

    def test_short_direction_accepted(self):
        @dataclass
        class ShortSignal:
            symbol: str = "BTCUSDT"
            timeframe: str = "1h"
            direction: str = "short"  # lowercase
            tier: str = "alpha"
            entry: float = 50000.0
            stop: float = 51000.0
            target: float = 48000.0
            score: float = 0.72
            market_state: str = "trending"

        pres = SignalPresentation.from_tiered_signal(ShortSignal())
        assert pres.direction == "SHORT"


# ============================================================================
# OUTPUT FORMATTING TESTS
# ============================================================================

class TestOutputFormatting:
    def test_telegram_text_structure(self):
        signal = FullTieredSignal()
        pres = SignalPresentation.from_tiered_signal(signal)
        text = pres.to_telegram_text()

        lines = text.split("\n")
        assert len(lines) >= 8
        assert "[SIGNAL]" in lines[0]  # ASCII label
        assert "BTCUSDT" in lines[0]
        assert "ALPHA" in lines[0]
        assert "[ENTRY]" in text
        assert "[STOP]" in text
        assert "[TARGET]" in text
        assert "[STATS]" in text
        assert "[RISK]" in text  # Invalidation label
        assert "[INFO]" in text  # Engine digest label
        assert "[ID]" in text  # Signal ID label

    def test_telegram_text_mobile_length(self):
        signal = FullTieredSignal()
        pres = SignalPresentation.from_tiered_signal(signal)
        text = pres.to_telegram_text()

        # Mobile readability: max ~500 chars for single-screen view
        assert len(text) < 600, f"Telegram text too long: {len(text)} chars"

    def test_json_serialization(self):
        signal = FullTieredSignal()
        pres = SignalPresentation.from_tiered_signal(signal)
        data = pres.to_json_dict()

        assert data["symbol"] == "BTCUSDT"
        assert data["direction"] == "LONG"
        assert data["score"] == "0.72"
        assert isinstance(data["metadata"], dict)

    def test_headline_format(self):
        signal = FullTieredSignal(direction="SHORT", score=0.85)
        pres = SignalPresentation.from_tiered_signal(signal)

        assert "[SHORT]" in pres.headline
        assert "SHORT" in pres.headline
        assert "Score 0.85" in pres.headline


# ============================================================================
# IMMUTABILITY TESTS
# ============================================================================

class TestImmutability:
    def test_frozen_dataclass_cannot_mutate(self):
        signal = FullTieredSignal()
        pres = SignalPresentation.from_tiered_signal(signal)

        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            pres.entry = "99999.00"

    def test_hashable_for_deduplication(self):
        signal = FullTieredSignal()
        pres = SignalPresentation.from_tiered_signal(signal)

        # Should be usable in sets/dicts because metadata has hash=False
        s = {pres}
        assert len(s) == 1

        # Should work as dict key
        d = {pres: "value"}
        assert d[pres] == "value"


# ============================================================================
# EXPLICIT SCHEMA TESTS (no getattr fallbacks)
# ============================================================================

class TestExplicitSchema:
    def test_no_silent_defaults_for_required_fields(self):
        """Required fields must raise, not default to empty string."""
        @dataclass
        class MissingMarketState:
            symbol: str = "BTCUSDT"
            timeframe: str = "1h"
            direction: str = "LONG"
            tier: str = "alpha"
            entry: float = 50000.0
            stop: float = 49000.0
            target: float = 52000.0
            score: float = 0.72
            # Missing: market_state

        with pytest.raises(PresentationError):
            SignalPresentation.from_tiered_signal(MissingMarketState())

    def test_rr_computed_from_levels_when_missing(self):
        """RR can be computed from entry/stop/target if risk_reward missing."""
        @dataclass
        class NoRRSignal:
            symbol: str = "BTCUSDT"
            timeframe: str = "1h"
            direction: str = "LONG"
            tier: str = "alpha"
            entry: float = 50000.0
            stop: float = 49000.0
            target: float = 52000.0
            score: float = 0.72
            market_state: str = "trending"
            # No risk_reward field

        pres = SignalPresentation.from_tiered_signal(NoRRSignal())
        assert pres.risk_reward == "2.0"  # (52000-50000)/(50000-49000) = 2.0

    def test_invalidation_built_from_stop_when_missing(self):
        @dataclass
        class NoInvalidation:
            symbol: str = "BTCUSDT"
            timeframe: str = "1h"
            direction: str = "LONG"
            tier: str = "alpha"
            entry: float = 50000.0
            stop: float = 49000.0
            target: float = 52000.0
            score: float = 0.72
            market_state: str = "trending"
            # No invalidation field

        pres = SignalPresentation.from_tiered_signal(NoInvalidation())
        assert "49,000.00" in pres.invalidation
        assert "Close below" in pres.invalidation


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    def test_very_high_precision_price(self):
        @dataclass
        class ShitcoinSignal:
            symbol: str = "SHIBUSDT"
            timeframe: str = "1h"
            direction: str = "LONG"
            tier: str = "alpha"
            entry: float = 0.00001234
            stop: float = 0.00001100
            target: float = 0.00001500
            score: float = 0.72
            market_state: str = "trending"

        pres = SignalPresentation.from_tiered_signal(ShitcoinSignal())
        assert "0.0000" in pres.entry  # 4 decimal places for non-BTC/ETH

    def test_zero_score(self):
        @dataclass
        class ZeroScore:
            symbol: str = "BTCUSDT"
            timeframe: str = "1h"
            direction: str = "LONG"
            tier: str = "alpha"
            entry: float = 50000.0
            stop: float = 49000.0
            target: float = 52000.0
            score: float = 0.0
            market_state: str = "trending"

        pres = SignalPresentation.from_tiered_signal(ZeroScore())
        assert pres.score == "0.00"

    def test_none_timestamp(self):
        @dataclass
        class NoneTimestamp:
            symbol: str = "BTCUSDT"
            timeframe: str = "1h"
            direction: str = "LONG"
            tier: str = "alpha"
            entry: float = 50000.0
            stop: float = 49000.0
            target: float = 52000.0
            score: float = 0.72
            market_state: str = "trending"
            generated_at: datetime = None

        pres = SignalPresentation.from_tiered_signal(NoneTimestamp())
        assert pres.generated_at != ""  # Should default to now


# ============================================================================
# IMPORT ISOLATION TEST
# ============================================================================

class TestIsolation:
    def test_no_pipeline_imports(self):
        """Verify presentation module has zero pipeline coupling."""
        import delivery.presentation as pres_module
        import ast

        source = open(pres_module.__file__, encoding="utf-8").read()
        tree = ast.parse(source)

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)

        # Should NOT import from scoring, tiering, pipeline, etc.
        forbidden = {"scoring", "tiering", "pipeline", "position_sizing"}
        for imp in imports:
            if imp:
                for f in forbidden:
                    assert f not in imp, f"Forbidden import: {imp}"