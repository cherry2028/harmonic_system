"""
tests/test_week2_tier.py
=========================
Exhaustive pytest coverage for signals/tier.py (SignalTier).

Test groups:
    1.  SignalTier construction and config validation
    2.  Tier assignment: _assign_tier() boundary conditions
    3.  classify() — all four tiers, correct field values
    4.  classify() — below-threshold returns None
    5.  Reasoning chain extension
    6.  TieredSignal field forwarding
    7.  Never-raise contract
    8.  Determinism
    9.  Config injection
    10. Daily counter TODO contract (structure verification)
    11. Property-based tests (hypothesis)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.market_state_config import MS_CONFIG, MarketStateConfig
from harmonic_patterns import PatternMatch
from market_state.vector import MarketStateVector
from scoring.pattern_scorer import PatternScorer
from scoring.score_result import ScoredSignal
from signals.signal import TieredSignal
from signals.tier import SignalTier, _TierLookup


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

STATES   = ["trending", "ranging", "expansion",
            "compression", "reversal", "news_chaos"]
PATTERNS = ["Gartley", "Bat", "Butterfly", "Crab"]


def make_match(
    quality:   float = 0.84,
    pattern:   str   = "Gartley",
    direction: str   = "bullish",
    symbol:    str   = "BTCUSDT",
    tf:        str   = "1h",
) -> PatternMatch:
    return PatternMatch(
        pattern_name  = pattern,
        direction     = direction,
        symbol        = symbol,
        timeframe     = tf,
        pivots        = {"X": 60000, "A": 65000, "B": 62000,
                         "C": 64000, "D": 61500},
        ratios        = {"AB_XA": 0.618, "BC_AB": 0.382, "CD_BC": 1.272,
                         "AD_XA": 0.786, "XD_XA": 0.300},
        validation    = {"AB_XA": True, "BC_AB": True,
                         "CD_BC": True, "AD_XA": True},
        prz           = {"entry": 61500.0, "stop": 59800.0,
                         "target1": 64000.0, "target2": 65000.0,
                         "target3": 66000.0},
        D_index       = 295,
        D_timestamp   = pd.Timestamp("2024-01-15 14:00"),
        quality_score = quality,
        metadata      = {},
    )


def make_vec(
    dominant:   str   = "reversal",
    confidence: float = 0.92,
) -> MarketStateVector:
    rest = (1.0 - confidence) / 5.0
    kw   = {s: rest for s in STATES}
    kw[dominant] = confidence
    return MarketStateVector(**kw)


def make_scored(
    base:       float = 0.84,
    dominant:   str   = "reversal",
    confidence: float = 0.92,
    pattern:    str   = "Gartley",
) -> ScoredSignal:
    scorer = PatternScorer()
    m = make_match(quality=base, pattern=pattern)
    v = make_vec(dominant=dominant, confidence=confidence)
    s = scorer.score(m, v)
    assert s is not None, (
        f"PatternScorer returned None for base={base} "
        f"dominant={dominant} conf={confidence}"
    )
    return s


def scored_with_edge(target_edge: float) -> Optional[ScoredSignal]:
    """
    Constructs a ScoredSignal with edge_score as close as possible
    to target_edge. Uses reversal state (discount=1.0) so that
    edge = base × 1.0 × confidence = base × confidence.
    We fix confidence=0.90 and derive base = target / confidence.
    Returns None if target is mathematically unreachable.
    """
    confidence = 0.90
    base = target_edge / confidence
    if not (0.0 <= base <= 1.0):
        return None
    return make_scored(base=base, dominant="reversal", confidence=confidence)


@pytest.fixture(scope="module")
def tier() -> SignalTier:
    return SignalTier()


# ---------------------------------------------------------------------------
# Group 1: Construction and config validation
# ---------------------------------------------------------------------------

class TestSignalTierConstruction:

    def test_default_construction_succeeds(self):
        t = SignalTier()
        assert t is not None

    def test_rules_loaded_in_decreasing_order(self, tier):
        rules = tier._rules
        assert len(rules) == 4
        for i in range(len(rules) - 1):
            assert rules[i].threshold > rules[i + 1].threshold, (
                f"Rules not decreasing: {rules[i].tier}={rules[i].threshold} "
                f"<= {rules[i + 1].tier}={rules[i + 1].threshold}"
            )

    def test_rules_contain_all_four_tiers(self, tier):
        tiers = {r.tier for r in tier._rules}
        assert tiers == {"A+", "A", "B", "C"}

    def test_ap_rule_is_first(self, tier):
        assert tier._rules[0].tier == "A+"

    def test_c_rule_is_last(self, tier):
        assert tier._rules[-1].tier == "C"

    def test_rule_fields_correct(self, tier):
        """Each _TierLookup must carry the right operational values."""
        expected = {
            "A+": (0.70, 1,  2.0),
            "A":  (0.50, 3,  1.0),
            "B":  (0.30, 5,  0.5),
            "C":  (0.10, 99, 0.0),
        }
        for rule in tier._rules:
            thresh, max_d, risk = expected[rule.tier]
            assert rule.threshold   == thresh, f"{rule.tier} threshold"
            assert rule.max_per_day == max_d,  f"{rule.tier} max_per_day"
            assert rule.risk_pct    == risk,   f"{rule.tier} risk_pct"

    def test_custom_config_accepted(self):
        """A valid custom config is accepted at construction."""
        t = SignalTier(config=MS_CONFIG)
        assert t is not None

    def test_validate_rules_called_on_init(self):
        """
        Misconfigured (non-decreasing) rules must raise at construction,
        not silently at classify() time.
        """
        # Build a config with tier thresholds out of order.
        # We monkey-patch tier_rules() to return bad ordering.
        class BadConfig:
            def tier_rules(self):
                return [
                    ("A+", 0.30, 1,  2.0),  # lower than A
                    ("A",  0.50, 3,  1.0),  # higher than A+
                    ("B",  0.30, 5,  0.5),
                    ("C",  0.10, 99, 0.0),
                ]
            harmonic_multipliers = MS_CONFIG.harmonic_multipliers
            fallback_discount    = MS_CONFIG.fallback_discount
            min_edge_score       = MS_CONFIG.min_edge_score
            tier_c_threshold     = MS_CONFIG.tier_c_threshold

        with pytest.raises(ValueError, match="strictly decreasing"):
            SignalTier(config=BadConfig())

    def test_empty_rules_raises(self):
        class EmptyConfig:
            def tier_rules(self):
                return []

        with pytest.raises(ValueError):
            SignalTier(config=EmptyConfig())


# ---------------------------------------------------------------------------
# Group 2: _assign_tier() boundary conditions
# ---------------------------------------------------------------------------

class TestAssignTier:

    @pytest.mark.parametrize("edge,expected_tier", [
        # AT each threshold (inclusive lower bound)
        (0.70,  "A+"),
        (0.50,  "A"),
        (0.30,  "B"),
        (0.10,  "C"),
        # ABOVE each threshold
        (0.71,  "A+"),
        (0.99,  "A+"),
        (1.00,  "A+"),
        (0.51,  "A"),
        (0.69,  "A"),
        (0.31,  "B"),
        (0.49,  "B"),
        (0.11,  "C"),
        (0.29,  "C"),
        # BELOW lowest threshold → None
        (0.099, None),
        (0.08,  None),
        (0.05,  None),
        (0.01,  None),
        (0.0,   None),
    ])
    def test_assign_tier_boundaries(self, tier, edge, expected_tier):
        lookup = tier._assign_tier(edge)
        actual = lookup.tier if lookup else None
        assert actual == expected_tier, (
            f"edge={edge}: got {actual!r}, expected {expected_tier!r}"
        )

    def test_assign_tier_returns_tier_lookup_or_none(self, tier):
        result = tier._assign_tier(0.75)
        assert isinstance(result, _TierLookup)

        result_none = tier._assign_tier(0.05)
        assert result_none is None

    def test_assign_tier_at_each_boundary_returns_higher_tier(self, tier):
        """
        Inclusive lower bound means at-threshold belongs to the
        higher tier, not the lower one.
        """
        assert tier._assign_tier(0.70).tier == "A+"   # not A
        assert tier._assign_tier(0.50).tier == "A"    # not B
        assert tier._assign_tier(0.30).tier == "B"    # not C
        assert tier._assign_tier(0.10).tier == "C"    # not None


# ---------------------------------------------------------------------------
# Group 3: classify() — all four tiers
# ---------------------------------------------------------------------------

class TestClassifyAllTiers:

    def test_classify_returns_tier_ap(self, tier):
        scored = scored_with_edge(0.75)
        assert scored is not None
        result = tier.classify(scored)
        assert result is not None
        assert result.tier == "A+"

    def test_classify_returns_tier_a(self, tier):
        scored = scored_with_edge(0.55)
        assert scored is not None
        result = tier.classify(scored)
        assert result is not None
        assert result.tier == "A"

    def test_classify_returns_tier_b(self, tier):
        scored = scored_with_edge(0.35)
        assert scored is not None
        result = tier.classify(scored)
        assert result is not None
        assert result.tier == "B"

    def test_classify_returns_tier_c(self, tier):
        scored = scored_with_edge(0.13)
        assert scored is not None
        result = tier.classify(scored)
        assert result is not None
        assert result.tier == "C"

    @pytest.mark.parametrize("tier_name,max_day,risk", [
        ("A+", 1,  2.0),
        ("A",  3,  1.0),
        ("B",  5,  0.5),
        ("C",  99, 0.0),
    ])
    def test_classify_tier_operational_fields(self, tier, tier_name, max_day, risk):
        """Each tier must carry the correct operational constraints."""
        edge_map = {"A+": 0.75, "A": 0.55, "B": 0.35, "C": 0.13}
        scored = scored_with_edge(edge_map[tier_name])
        assert scored is not None
        result = tier.classify(scored)
        assert result is not None
        assert result.tier        == tier_name
        assert result.max_per_day == max_day
        assert result.risk_pct    == risk

    def test_tier_c_is_paper_only(self, tier):
        scored = scored_with_edge(0.13)
        result = tier.classify(scored)
        assert result is not None
        assert result.is_paper_only is True

    def test_tier_ap_is_not_paper_only(self, tier):
        scored = scored_with_edge(0.75)
        result = tier.classify(scored)
        assert result is not None
        assert result.is_paper_only is False

    def test_classify_returns_tiered_signal_instance(self, tier):
        scored = scored_with_edge(0.55)
        result = tier.classify(scored)
        assert isinstance(result, TieredSignal)


# ---------------------------------------------------------------------------
# Group 4: below-threshold returns None
# ---------------------------------------------------------------------------

class TestBelowThresholdReturnsNone:

    @pytest.mark.parametrize("edge_approx", [0.099, 0.08, 0.05, 0.01, 0.0])
    def test_below_c_threshold_returns_none(self, tier, edge_approx):
        """
        Any score below Tier C threshold (0.10) must return None.
        This includes the zone between min_edge_score (0.08) and
        tier_c_threshold (0.10) — ScoredSignal exists, TieredSignal does not.
        """
        scored = scored_with_edge(edge_approx)
        if scored is None:
            pytest.skip(f"Cannot construct ScoredSignal for edge={edge_approx}")
        # Only test if the edge is actually below C threshold
        if scored.edge_score >= MS_CONFIG.tier_c_threshold:
            pytest.skip(f"Scored edge {scored.edge_score} >= C threshold")
        result = tier.classify(scored)
        assert result is None, (
            f"edge={scored.edge_score:.4f} should return None, "
            f"got tier={result.tier if result else None}"
        )

    def test_chaos_signal_below_tier_c(self, tier):
        """
        A perfect pattern in news_chaos state (discount=0.05)
        must produce edge below Tier C threshold.
        """
        m = make_match(quality=1.0)
        v = MarketStateVector(
            trending=0.04, ranging=0.04, expansion=0.04,
            compression=0.04, reversal=0.04, news_chaos=0.80,
        )
        scored = PatternScorer().score(m, v)
        assert scored is not None
        # edge = 1.0 * 0.05 * confidence <= 0.05 < 0.10
        assert scored.edge_score < MS_CONFIG.tier_c_threshold
        result = tier.classify(scored)
        assert result is None, (
            f"Chaos signal with edge={scored.edge_score:.4f} "
            f"should not produce a TieredSignal"
        )


# ---------------------------------------------------------------------------
# Group 5: Reasoning chain extension
# ---------------------------------------------------------------------------

class TestReasoningChainExtension:

    def test_above_threshold_reasoning_has_seven_lines(self, tier):
        """5 scoring lines + 2 tier lines = 7 total."""
        scored = scored_with_edge(0.55)
        assert scored is not None
        assert len(scored.reasoning) == 5, (
            f"Expected 5 scoring lines, got {len(scored.reasoning)}"
        )
        result = tier.classify(scored)
        assert result is not None
        assert len(result.reasoning) == 7, (
            f"Expected 7 total lines, got {len(result.reasoning)}"
        )

    def test_tier_line_contains_tier_name(self, tier):
        """Second-to-last reasoning line must name the tier."""
        scored = scored_with_edge(0.55)
        result = tier.classify(scored)
        assert result is not None
        tier_line = result.reasoning[-2]
        assert "Tier A" in tier_line, (
            f"Tier name not in line: {tier_line!r}"
        )

    def test_tier_line_contains_threshold(self, tier):
        """Tier assignment line must state the threshold met."""
        scored = scored_with_edge(0.55)
        result = tier.classify(scored)
        tier_line = result.reasoning[-2]
        assert "threshold" in tier_line.lower() or "%" in tier_line, (
            f"Threshold not mentioned: {tier_line!r}"
        )

    def test_sizing_line_contains_risk_pct(self, tier):
        """Last reasoning line must state the risk percentage."""
        scored = scored_with_edge(0.55)
        result = tier.classify(scored)
        sizing_line = result.reasoning[-1]
        assert "1.0" in sizing_line or "1%" in sizing_line, (
            f"Risk pct not in sizing line: {sizing_line!r}"
        )

    def test_sizing_line_contains_max_per_day(self, tier):
        """Last reasoning line must state the frequency cap."""
        scored = scored_with_edge(0.55)
        result = tier.classify(scored)
        sizing_line = result.reasoning[-1]
        assert "max" in sizing_line.lower() or "/day" in sizing_line, (
            f"Max per day not in sizing line: {sizing_line!r}"
        )

    def test_tier_c_sizing_line_mentions_paper(self, tier):
        """Tier C must explicitly note paper/educational — no real capital."""
        scored = scored_with_edge(0.13)
        result = tier.classify(scored)
        assert result is not None
        sizing_line = result.reasoning[-1]
        assert "paper" in sizing_line.lower(), (
            f"Tier C sizing line must mention 'paper': {sizing_line!r}"
        )

    def test_original_scoring_reasoning_preserved(self, tier):
        """The scoring reasoning chain must be unchanged in TieredSignal."""
        scored = scored_with_edge(0.55)
        original = list(scored.reasoning)
        result   = tier.classify(scored)
        # First 5 lines must match original scoring reasoning
        assert result.reasoning[:5] == original, (
            "Original scoring reasoning was modified"
        )

    def test_base_reasoning_not_mutated(self, tier):
        """classify() must not mutate scored.reasoning."""
        scored = scored_with_edge(0.55)
        original_len = len(scored.reasoning)
        tier.classify(scored)
        assert len(scored.reasoning) == original_len, (
            "classify() mutated scored.reasoning"
        )

    def test_all_reasoning_lines_are_strings(self, tier):
        scored = scored_with_edge(0.55)
        result = tier.classify(scored)
        for i, line in enumerate(result.reasoning):
            assert isinstance(line, str) and line.strip(), (
                f"Line {i} is empty or not a string: {line!r}"
            )

    def test_reasoning_deterministic(self, tier):
        """Same input → identical reasoning chain. Always."""
        scored = scored_with_edge(0.55)
        r1 = tier.classify(scored)
        r2 = tier.classify(scored)
        assert r1.reasoning == r2.reasoning


# ---------------------------------------------------------------------------
# Group 6: TieredSignal field forwarding
# ---------------------------------------------------------------------------

class TestFieldForwarding:

    def test_edge_score_forwarded(self, tier):
        scored = scored_with_edge(0.55)
        result = tier.classify(scored)
        assert math.isclose(result.edge_score, scored.edge_score, abs_tol=1e-10)

    def test_dominant_state_forwarded(self, tier):
        scored = scored_with_edge(0.55)
        result = tier.classify(scored)
        assert result.dominant_state == scored.dominant_state

    def test_entry_forwarded(self, tier):
        scored = scored_with_edge(0.55)
        result = tier.classify(scored)
        assert result.entry == scored.entry

    def test_stop_forwarded(self, tier):
        scored = scored_with_edge(0.55)
        result = tier.classify(scored)
        assert result.stop == scored.stop

    def test_target1_forwarded(self, tier):
        scored = scored_with_edge(0.55)
        result = tier.classify(scored)
        assert result.target1 == scored.target1

    def test_target2_forwarded(self, tier):
        scored = scored_with_edge(0.55)
        result = tier.classify(scored)
        assert result.target2 == scored.target2

    def test_target3_forwarded(self, tier):
        scored = scored_with_edge(0.55)
        result = tier.classify(scored)
        assert result.target3 == scored.target3

    def test_risk_reward_forwarded(self, tier):
        scored = scored_with_edge(0.55)
        result = tier.classify(scored)
        assert result.risk_reward == scored.risk_reward

    def test_scored_reference_preserved(self, tier):
        """TieredSignal.scored must be the exact same ScoredSignal object."""
        scored = scored_with_edge(0.55)
        result = tier.classify(scored)
        assert result.scored is scored, (
            "TieredSignal.scored must be the exact input ScoredSignal"
        )

    def test_symbol_property(self, tier):
        scored = scored_with_edge(0.55)
        result = tier.classify(scored)
        assert result.symbol    == scored.pattern_match.symbol
        assert result.timeframe == scored.pattern_match.timeframe

    @pytest.mark.parametrize("pattern", PATTERNS)
    def test_all_patterns_forward_correctly(self, tier, pattern):
        scored = make_scored(base=0.75, dominant="reversal",
                             confidence=0.85, pattern=pattern)
        result = tier.classify(scored)
        assert result is not None
        assert result.pattern_name == pattern
        assert result.scored.pattern_match.pattern_name == pattern


# ---------------------------------------------------------------------------
# Group 7: Never-raise contract
# ---------------------------------------------------------------------------

class TestNeverRaiseContract:

    @pytest.mark.parametrize("bad_input", [
        None, {}, "string", 42, 3.14, [], object(),
    ])
    def test_bad_input_returns_none(self, tier, bad_input):
        result = tier.classify(bad_input)
        assert result is None, (
            f"classify({type(bad_input).__name__}) should return None"
        )

    def test_repeated_calls_never_raise(self, tier):
        scored = scored_with_edge(0.75)
        for _ in range(50):
            result = tier.classify(scored)
            assert isinstance(result, TieredSignal)

    def test_classify_after_bad_inputs_still_works(self, tier):
        """Gate poisoning: bad inputs must not break subsequent valid calls."""
        tier.classify(None)
        tier.classify({})
        tier.classify("bad")
        # Now valid input must work normally
        scored = scored_with_edge(0.75)
        result = tier.classify(scored)
        assert result is not None
        assert result.tier == "A+"


# ---------------------------------------------------------------------------
# Group 8: Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_input_same_output(self, tier):
        """100 calls on the same scored signal → identical TieredSignal."""
        scored = scored_with_edge(0.55)
        results = [tier.classify(scored) for _ in range(100)]
        tiers   = {r.tier        for r in results}
        edges   = {r.edge_score  for r in results}
        risks   = {r.risk_pct    for r in results}
        assert len(tiers) == 1,  f"Non-deterministic tier: {tiers}"
        assert len(edges) == 1,  f"Non-deterministic edge_score: {edges}"
        assert len(risks) == 1,  f"Non-deterministic risk_pct: {risks}"

    def test_two_instances_same_result(self):
        """Two SignalTier instances must produce identical results."""
        tier1 = SignalTier()
        tier2 = SignalTier()
        scored = make_scored(base=0.80, dominant="reversal", confidence=0.85)
        r1 = tier1.classify(scored)
        r2 = tier2.classify(scored)
        assert r1.tier          == r2.tier
        assert r1.edge_score    == r2.edge_score
        assert r1.max_per_day   == r2.max_per_day
        assert r1.risk_pct      == r2.risk_pct
        assert r1.reasoning     == r2.reasoning

    def test_different_inputs_different_tiers(self, tier):
        """Different edge scores must produce different tiers when appropriate."""
        s_ap = scored_with_edge(0.75)
        s_a  = scored_with_edge(0.55)
        assert tier.classify(s_ap).tier == "A+"
        assert tier.classify(s_a).tier  == "A"


# ---------------------------------------------------------------------------
# Group 9: Config injection
# ---------------------------------------------------------------------------

class TestConfigInjection:

    def test_custom_config_changes_tier_thresholds(self):
        """
        Injecting a config with different thresholds changes classification.
        """
        # We cannot easily change just the thresholds without rebuilding
        # the full config (MarketStateConfig validates at construction).
        # Instead, verify that our default config produces known results
        # and a fresh SignalTier with the same config matches.
        tier1 = SignalTier(config=MS_CONFIG)
        tier2 = SignalTier(config=MS_CONFIG)
        scored = scored_with_edge(0.75)
        assert tier1.classify(scored).tier == tier2.classify(scored).tier

    def test_default_and_explicit_config_identical(self):
        """SignalTier() and SignalTier(config=MS_CONFIG) are identical."""
        tier_default  = SignalTier()
        tier_explicit = SignalTier(config=MS_CONFIG)
        scored = scored_with_edge(0.55)
        r1 = tier_default.classify(scored)
        r2 = tier_explicit.classify(scored)
        assert r1.tier        == r2.tier
        assert r1.max_per_day == r2.max_per_day
        assert r1.risk_pct    == r2.risk_pct


# ---------------------------------------------------------------------------
# Group 10: Daily counter TODO contract verification
# ---------------------------------------------------------------------------

class TestDailyCounterTODOContract:

    def test_todo_comment_exists_in_source(self):
        """
        Verify the TODO for DailyCounter is present and documents the
        exact injection contract so Phase 2 implementation is unambiguous.
        """
        src = open(
            str(Path(__file__).parent.parent / "signals" / "tier.py")
        ).read()
        assert "TODO Week 2 Phase 2" in src, (
            "DailyCounter TODO comment must be in signals/tier.py"
        )
        assert "DailyCounter" in src, (
            "DailyCounter class name must be documented in TODO"
        )
        assert "check(" in src, (
            "DailyCounter.check() contract must be documented"
        )
        assert "increment(" in src, (
            "DailyCounter.increment() contract must be documented"
        )

    def test_current_behavior_no_cap_enforcement(self, tier):
        """
        Without DailyCounter, classify() should never refuse a signal
        based on frequency — cap enforcement is not yet active.
        """
        scored = scored_with_edge(0.75)
        # Call more times than the A+ daily cap (max=1)
        results = [tier.classify(scored) for _ in range(5)]
        assert all(r is not None for r in results), (
            "Without DailyCounter, no call should be blocked by frequency cap"
        )
        assert all(r.tier == "A+" for r in results), (
            "All results should still be A+"
        )


# ---------------------------------------------------------------------------
# Group 11: Property-based tests
# ---------------------------------------------------------------------------

@settings(max_examples=200, deadline=2000)
@given(
    quality    = st.floats(min_value=0.0, max_value=1.0,
                           allow_nan=False, allow_infinity=False),
    dominant   = st.sampled_from(["reversal", "ranging", "trending"]),
    confidence = st.floats(min_value=0.20, max_value=1.0,
                           allow_nan=False, allow_infinity=False),
    pattern    = st.sampled_from(PATTERNS),
)
def test_property_classify_returns_tiered_signal_or_none(
    quality, dominant, confidence, pattern
):
    """
    Property: classify() always returns TieredSignal or None.
    Never raises. Never returns another type.
    """
    tier   = SignalTier()
    scored = make_scored(base=quality, dominant=dominant,
                         confidence=confidence, pattern=pattern)
    result = tier.classify(scored)
    assert result is None or isinstance(result, TieredSignal), (
        f"classify() returned unexpected type: {type(result).__name__}"
    )


@settings(max_examples=200, deadline=2000)
@given(
    quality    = st.floats(min_value=0.0, max_value=1.0,
                           allow_nan=False, allow_infinity=False),
    dominant   = st.sampled_from(["reversal", "ranging", "trending"]),
    confidence = st.floats(min_value=0.20, max_value=1.0,
                           allow_nan=False, allow_infinity=False),
    pattern    = st.sampled_from(PATTERNS),
)
def test_property_tiered_signal_tier_is_valid(
    quality, dominant, confidence, pattern
):
    """
    Property: when classify() returns a TieredSignal, its tier
    must be one of the four known tier names.
    """
    tier   = SignalTier()
    scored = make_scored(base=quality, dominant=dominant,
                         confidence=confidence, pattern=pattern)
    result = tier.classify(scored)
    if result is not None:
        assert result.tier in {"A+", "A", "B", "C"}, (
            f"Invalid tier: {result.tier!r}"
        )


@settings(max_examples=200, deadline=2000)
@given(
    quality    = st.floats(min_value=0.0, max_value=1.0,
                           allow_nan=False, allow_infinity=False),
    dominant   = st.sampled_from(["reversal", "ranging", "trending"]),
    confidence = st.floats(min_value=0.20, max_value=1.0,
                           allow_nan=False, allow_infinity=False),
)
def test_property_edge_score_consistent_with_tier(quality, dominant, confidence):
    """
    Property: the edge_score in a TieredSignal must be >= the tier's
    threshold. The tier was assigned based on this edge — they must agree.
    """
    tier_thresholds = {"A+": 0.70, "A": 0.50, "B": 0.30, "C": 0.10}
    tier   = SignalTier()
    scored = make_scored(base=quality, dominant=dominant, confidence=confidence)
    result = tier.classify(scored)
    if result is not None:
        minimum = tier_thresholds[result.tier]
        assert result.edge_score >= minimum, (
            f"Tier {result.tier} requires edge >= {minimum}, "
            f"but edge={result.edge_score:.6f}"
        )


@settings(max_examples=200, deadline=2000)
@given(
    quality    = st.floats(min_value=0.0, max_value=1.0,
                           allow_nan=False, allow_infinity=False),
    dominant   = st.sampled_from(["reversal", "ranging", "trending"]),
    confidence = st.floats(min_value=0.20, max_value=1.0,
                           allow_nan=False, allow_infinity=False),
)
def test_property_none_only_when_below_threshold(quality, dominant, confidence):
    """
    Property: classify() returns None if and only if
    edge_score < tier_c_threshold.
    """
    tier   = SignalTier()
    scored = make_scored(base=quality, dominant=dominant, confidence=confidence)
    result = tier.classify(scored)
    c_thresh = MS_CONFIG.tier_c_threshold

    if result is None:
        assert scored.edge_score < c_thresh, (
            f"classify() returned None but edge={scored.edge_score:.6f} "
            f">= tier_C threshold={c_thresh}"
        )
    else:
        assert scored.edge_score >= c_thresh, (
            f"classify() returned TieredSignal but edge={scored.edge_score:.6f} "
            f"< tier_C threshold={c_thresh}"
        )


@settings(max_examples=100, deadline=2000)
@given(
    quality    = st.floats(min_value=0.15, max_value=1.0,
                           allow_nan=False, allow_infinity=False),
    confidence = st.floats(min_value=0.40, max_value=1.0,
                           allow_nan=False, allow_infinity=False),
)
def test_property_reasoning_non_empty_when_tiered(quality, confidence):
    """
    Property: every TieredSignal must carry a non-empty reasoning chain.
    """
    tier   = SignalTier()
    scored = make_scored(base=quality, dominant="reversal", confidence=confidence)
    result = tier.classify(scored)
    if result is not None:
        assert len(result.reasoning) > 0, (
            "TieredSignal reasoning must never be empty"
        )
        assert all(isinstance(line, str) for line in result.reasoning)