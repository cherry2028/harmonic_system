"""
tests/test_week2_scoring.py
============================
Exhaustive pytest + property-based coverage for PatternScorer.

Test groups:
    1.  PatternScorer construction and config validation
    2.  Base score extraction (_extract_base_score)
    3.  State discount resolution (_resolve_discount)
    4.  Confidence weighting
    5.  Edge score formula (product consistency)
    6.  Reasoning chain construction
    7.  score_batch behavior
    8.  Never-raise contract
    9.  Axiom compliance (deterministic cases)
    10. Property-based tests (hypothesis — full input space)
    11. ScoredSignal invariants via scorer
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.market_state_config import MS_CONFIG, MarketStateConfig
from harmonic_patterns import PatternMatch
from market_state.vector import MarketStateVector
from scoring.pattern_scorer import PatternScorer, _discount_interpretation
from scoring.score_result import ScoredSignal


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

PATTERNS  = ["Gartley", "Bat", "Butterfly", "Crab"]
STATES    = ["trending", "ranging", "expansion",
             "compression", "reversal", "news_chaos"]
DIRECTIONS = ["bullish", "bearish"]


def make_match(
    quality:   float = 0.84,
    pattern:   str   = "Gartley",
    direction: str   = "bullish",
    symbol:    str   = "BTCUSDT",
    tf:        str   = "1h",
    entry:     float = 61500.0,
    stop:      float = 59800.0,
) -> PatternMatch:
    return PatternMatch(
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


def make_vec(
    dominant:   str   = "ranging",
    confidence: float = 0.72,
) -> MarketStateVector:
    """Builds a valid MarketStateVector with the named dominant state."""
    rest = (1.0 - confidence) / 5.0
    kw = {s: rest for s in STATES}
    kw[dominant] = confidence
    return MarketStateVector(**kw)


def score_ok(
    quality:    float = 0.84,
    dominant:   str   = "ranging",
    confidence: float = 0.72,
    pattern:    str   = "Gartley",
) -> ScoredSignal:
    """Returns a valid ScoredSignal or fails the test."""
    scorer = PatternScorer()
    m = make_match(quality=quality, pattern=pattern)
    v = make_vec(dominant=dominant, confidence=confidence)
    s = scorer.score(m, v)
    assert s is not None, (
        f"score() returned None for quality={quality} "
        f"dominant={dominant} conf={confidence}"
    )
    return s


@pytest.fixture(scope="module")
def scorer() -> PatternScorer:
    return PatternScorer()


# ---------------------------------------------------------------------------
# Group 1: Construction and config validation
# ---------------------------------------------------------------------------

class TestPatternScorerConstruction:

    def test_default_construction_succeeds(self):
        scorer = PatternScorer()
        assert scorer is not None

    def test_custom_valid_config_accepted(self):
        """A properly configured custom config is accepted."""
        scorer = PatternScorer(config=MS_CONFIG)
        assert scorer is not None

    def test_config_with_amplifying_discount_rejected(self):
        """Config with discount > 1.0 must be rejected at construction."""
        bad_mults = {
            state: {p: 1.00 for p in PATTERNS}
            for state in STATES
        }
        bad_mults["reversal"] = {p: 1.50 for p in PATTERNS}  # > 1.0
        with pytest.raises((ValueError,)):
            bad_cfg = MarketStateConfig(harmonic_multipliers=bad_mults)

    def test_config_with_zero_discount_rejected(self):
        """Config with discount = 0.0 must be rejected at construction."""
        bad_mults = {state: {p: 1.00 for p in PATTERNS} for state in STATES}
        bad_mults["news_chaos"] = {p: 0.0 for p in PATTERNS}  # zero
        with pytest.raises((ValueError,)):
            MarketStateConfig(harmonic_multipliers=bad_mults)


# ---------------------------------------------------------------------------
# Group 2: Base score extraction
# ---------------------------------------------------------------------------

class TestBaseScoreExtraction:

    @pytest.mark.parametrize("quality", [0.0, 0.01, 0.5, 0.84, 0.999, 1.0])
    def test_valid_quality_scores_accepted(self, scorer, quality):
        m = make_match(quality=quality)
        v = make_vec()
        s = scorer.score(m, v)
        assert s is not None
        assert s.base_score == quality

    @pytest.mark.parametrize("bad_quality", [-0.001, -1.0, 1.001, 1.5, 99.0])
    def test_invalid_quality_scores_return_none(self, scorer, bad_quality):
        m = make_match(quality=bad_quality)
        v = make_vec()
        s = scorer.score(m, v)
        assert s is None, (
            f"quality={bad_quality} should return None, got ScoredSignal"
        )

    def test_zero_quality_produces_zero_edge(self, scorer):
        """base_score=0.0 → edge_score=0.0 regardless of state/confidence."""
        s = score_ok(quality=0.0, dominant="reversal", confidence=0.95)
        assert s.edge_score == 0.0

    def test_quality_one_with_reversal_confidence_one(self, scorer):
        """base=1.0, reversal discount=1.0, confidence=1.0 → edge=1.0."""
        m = make_match(quality=1.0)
        v = MarketStateVector(
            trending=0.0, ranging=0.0, expansion=0.0,
            compression=0.0, reversal=1.0, news_chaos=0.0,
        )
        s = scorer.score(m, v)
        assert s is not None
        assert math.isclose(s.edge_score, 1.0, abs_tol=1e-10), (
            f"edge_score={s.edge_score} should be 1.0"
        )


# ---------------------------------------------------------------------------
# Group 3: State discount resolution
# ---------------------------------------------------------------------------

class TestStateDiscountResolution:

    @pytest.mark.parametrize("pattern", PATTERNS)
    def test_reversal_discount_is_1_for_all_patterns(self, scorer, pattern):
        """Axiom 3: reversal state grants full credit — discount = 1.00."""
        s = score_ok(quality=0.80, dominant="reversal", pattern=pattern)
        assert s.state_discount == 1.00, (
            f"{pattern}/reversal discount={s.state_discount}, expected 1.00"
        )

    @pytest.mark.parametrize("pattern", PATTERNS)
    def test_news_chaos_discount_is_005_for_all_patterns(self, scorer, pattern):
        """Axiom 4: news_chaos discount = 0.05 for all patterns."""
        m = make_match(quality=0.80, pattern=pattern)
        v = MarketStateVector(trending=0.04, ranging=0.04, expansion=0.04,
                              compression=0.04, reversal=0.04, news_chaos=0.80)
        s = scorer.score(m, v)
        assert s is not None
        assert s.state_discount == 0.05, (
            f"{pattern}/news_chaos discount={s.state_discount}, expected 0.05"
        )

    def test_all_discounts_are_in_valid_range(self, scorer):
        """Every (state, pattern) combination must produce discount in (0, 1]."""
        for state in STATES:
            for pattern in PATTERNS:
                m = make_match(quality=0.80, pattern=pattern)
                v = make_vec(dominant=state, confidence=0.80)
                s = scorer.score(m, v)
                assert s is not None
                assert 0.0 < s.state_discount <= 1.0, (
                    f"{state}/{pattern}: discount={s.state_discount} "
                    f"outside (0.0, 1.0]"
                )

    def test_unknown_state_uses_fallback(self, scorer):
        """Unknown dominant_state triggers fallback discount."""
        # Construct a vector with a known dominant, then test via resolve_discount
        discount, source = scorer._resolve_discount("UNKNOWN_STATE", "Gartley")
        assert discount == MS_CONFIG.fallback_discount
        assert "fallback" in source

    def test_unknown_pattern_uses_fallback(self, scorer):
        """Unknown pattern_name triggers fallback discount."""
        discount, source = scorer._resolve_discount("ranging", "NewPattern")
        assert discount == MS_CONFIG.fallback_discount
        assert "fallback" in source

    def test_no_discount_exceeds_one(self, scorer):
        """No discount in the full table may exceed 1.0 (Axiom 3)."""
        for state, patterns in MS_CONFIG.harmonic_multipliers.items():
            for pattern, value in patterns.items():
                assert value <= 1.0, (
                    f"table[{state}][{pattern}]={value} > 1.0 — Axiom 3 violated"
                )

    def test_discount_ordering_reversal_greater_than_chaos(self, scorer):
        """reversal discount must always exceed news_chaos discount."""
        for pattern in PATTERNS:
            rev_disc = MS_CONFIG.get_discount("reversal", pattern)
            chaos_disc = MS_CONFIG.get_discount("news_chaos", pattern)
            assert rev_disc > chaos_disc, (
                f"{pattern}: reversal={rev_disc} should > chaos={chaos_disc}"
            )


# ---------------------------------------------------------------------------
# Group 4: Confidence weighting
# ---------------------------------------------------------------------------

class TestConfidenceWeighting:

    @pytest.mark.parametrize("conf", [0.10, 0.30, 0.50, 0.70, 0.90, 1.00])
    def test_confidence_equals_vector_confidence(self, scorer, conf):
        """confidence_weight must equal vector.confidence exactly."""
        m = make_match(quality=0.80)
        v = make_vec(dominant="ranging", confidence=conf)
        s = scorer.score(m, v)
        assert s is not None
        assert s.confidence_weight == v.confidence, (
            f"confidence_weight={s.confidence_weight} != "
            f"vector.confidence={v.confidence}"
        )

    def test_confidence_scales_edge_linearly(self, scorer):
        """Doubling confidence (all else equal) must double edge_score."""
        base, discount_state = 0.80, "ranging"
        conf_lo, conf_hi = 0.30, 0.60

        def make_exact_vec(conf):
            rest = (1.0 - conf) / 5.0
            return MarketStateVector(
                trending=rest, ranging=conf, expansion=rest,
                compression=rest, reversal=rest, news_chaos=rest,
            )

        m   = make_match(quality=base)
        s_lo = scorer.score(m, make_exact_vec(conf_lo))
        s_hi = scorer.score(m, make_exact_vec(conf_hi))
        assert s_lo is not None and s_hi is not None

        ratio = s_hi.edge_score / s_lo.edge_score
        assert math.isclose(ratio, conf_hi / conf_lo, rel_tol=1e-6), (
            f"Edge ratio {ratio:.6f} != conf ratio {conf_hi/conf_lo:.6f}. "
            f"Confidence must scale edge linearly."
        )

    def test_higher_confidence_produces_higher_edge(self, scorer):
        """Monotonicity: higher confidence → strictly higher edge (all else equal)."""
        m = make_match(quality=0.80)
        prev_edge = -1.0
        for conf in [0.20, 0.40, 0.60, 0.80]:
            rest = (1.0 - conf) / 5.0
            v = MarketStateVector(
                trending=rest, ranging=conf, expansion=rest,
                compression=rest, reversal=rest, news_chaos=rest,
            )
            s = scorer.score(m, v)
            assert s is not None
            assert s.edge_score > prev_edge, (
                f"edge={s.edge_score:.4f} not > prev={prev_edge:.4f} "
                f"at conf={conf}"
            )
            prev_edge = s.edge_score


# ---------------------------------------------------------------------------
# Group 5: Edge score formula
# ---------------------------------------------------------------------------

class TestEdgeScoreFormula:

    def test_edge_equals_product_of_three_factors(self, scorer):
        """edge_score == base × discount × confidence within tolerance."""
        s = score_ok(quality=0.84, dominant="ranging", confidence=0.72)
        expected = s.base_score * s.state_discount * s.confidence_weight
        assert math.isclose(s.edge_score, expected, abs_tol=1e-10), (
            f"edge={s.edge_score:.10f} != "
            f"base×disc×conf={expected:.10f}"
        )

    @pytest.mark.parametrize("base,conf", [
        (1.0, 1.0), (0.5, 0.5), (0.01, 0.99),
        (0.99, 0.01), (0.84, 0.72), (0.0, 0.80),
    ])
    def test_product_formula_across_inputs(self, scorer, base, conf):
        m = make_match(quality=base)
        v = make_vec(dominant="ranging", confidence=conf)
        s = scorer.score(m, v)
        assert s is not None
        expected = s.base_score * s.state_discount * s.confidence_weight
        assert math.isclose(s.edge_score, expected, abs_tol=1e-10)

    def test_edge_never_exceeds_base_score(self, scorer):
        """Axiom 3 consequence: discount ≤ 1.0 means edge ≤ base always."""
        for quality in [0.30, 0.60, 0.90, 1.00]:
            for state in STATES:
                s = score_ok(quality=quality, dominant=state, confidence=0.80)
                assert s.edge_score <= s.base_score + 1e-10, (
                    f"edge={s.edge_score:.6f} > base={s.base_score:.6f} "
                    f"in state={state}. Axiom 3 violated."
                )

    def test_edge_in_zero_one_range(self, scorer):
        """edge_score ∈ [0.0, 1.0] for all (state, pattern, quality) combinations."""
        for state in STATES:
            for pattern in PATTERNS:
                for quality in [0.0, 0.5, 1.0]:
                    s = score_ok(
                        quality=quality, dominant=state,
                        confidence=0.80, pattern=pattern,
                    )
                    assert 0.0 <= s.edge_score <= 1.0, (
                        f"edge={s.edge_score} out of [0,1] for "
                        f"state={state} pattern={pattern} quality={quality}"
                    )


# ---------------------------------------------------------------------------
# Group 6: Reasoning chain
# ---------------------------------------------------------------------------

class TestReasoningChain:

    def test_above_threshold_produces_five_lines(self, scorer):
        """Signals above min_edge_score must have exactly 5 reasoning lines."""
        s = score_ok(quality=0.84, dominant="reversal", confidence=0.90)
        assert s.edge_score >= MS_CONFIG.min_edge_score
        assert len(s.reasoning) == 5, (
            f"Expected 5 reasoning lines, got {len(s.reasoning)}"
        )

    def test_all_reasoning_lines_are_strings(self, scorer):
        s = score_ok(quality=0.84, dominant="ranging", confidence=0.72)
        for i, line in enumerate(s.reasoning):
            assert isinstance(line, str) and line.strip(), (
                f"Line {i} is empty or not a string: {line!r}"
            )

    def test_reasoning_line1_contains_pattern_and_quality(self, scorer):
        s = score_ok(quality=0.84, dominant="ranging", confidence=0.72, pattern="Bat")
        assert "Bat"    in s.reasoning[0]
        assert "84%"    in s.reasoning[0]

    def test_reasoning_line2_contains_state_and_confidence(self, scorer):
        s = score_ok(quality=0.84, dominant="ranging", confidence=0.72)
        assert "RANGING" in s.reasoning[1]
        assert "72%"     in s.reasoning[1]

    def test_reasoning_line3_contains_discount_and_source(self, scorer):
        s = score_ok(quality=0.84, dominant="ranging", confidence=0.72)
        assert "discount" in s.reasoning[2].lower()
        assert "table"    in s.reasoning[2].lower()

    def test_reasoning_line4_contains_confidence_weight(self, scorer):
        s = score_ok(quality=0.84, dominant="ranging", confidence=0.72)
        assert "confidence" in s.reasoning[3].lower()

    def test_reasoning_line5_contains_edge_score(self, scorer):
        s = score_ok(quality=0.84, dominant="ranging", confidence=0.72)
        assert "%" in s.reasoning[4]

    def test_below_threshold_positive_edge_gives_three_lines(self, scorer):
        """edge > 0 but below min_edge_score → 3 minimal lines."""
        m = make_match(quality=0.10, pattern="Gartley")
        v = MarketStateVector(trending=0.04, ranging=0.04, expansion=0.04,
                              compression=0.04, reversal=0.04, news_chaos=0.80)
        s = scorer.score(m, v)
        assert s is not None
        if s.edge_score > 0.0 and s.edge_score < MS_CONFIG.min_edge_score:
            assert len(s.reasoning) == 3

    def test_zero_edge_produces_empty_reasoning(self, scorer):
        """base_score=0.0 → edge=0.0 → empty reasoning list."""
        m = make_match(quality=0.0)
        v = make_vec()
        s = scorer.score(m, v)
        assert s is not None
        assert s.edge_score == 0.0
        assert s.reasoning == []

    def test_reasoning_is_deterministic(self, scorer):
        """Same inputs → identical reasoning chain. Always."""
        m = make_match(quality=0.84)
        v = make_vec(dominant="ranging", confidence=0.72)
        r1 = scorer.score(m, v).reasoning
        r2 = scorer.score(m, v).reasoning
        assert r1 == r2, "Reasoning is not deterministic"

    def test_discount_interpretation_monotonic(self):
        """_discount_interpretation maps lower discount to less favorable label."""
        pairs = [
            (1.00, "ideal conditions"),
            (0.90, "ideal conditions"),
            (0.80, "favorable"),
            (0.70, "favorable"),
            (0.55, "neutral"),
            (0.45, "neutral"),
            (0.30, "unfavorable"),
            (0.25, "unfavorable"),
            (0.10, "hostile"),
            (0.05, "hostile"),
        ]
        for val, expected in pairs:
            result = _discount_interpretation(val)
            assert result == expected, (
                f"_discount_interpretation({val}) = {result!r}, "
                f"expected {expected!r}"
            )


# ---------------------------------------------------------------------------
# Group 7: score_batch
# ---------------------------------------------------------------------------

class TestScoreBatch:

    def test_batch_returns_list(self, scorer):
        matches = [make_match(quality=q) for q in [0.90, 0.70, 0.50]]
        v = make_vec()
        results = scorer.score_batch(matches, v)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_batch_sorted_descending(self, scorer):
        """Batch results must be sorted highest edge first."""
        matches = [make_match(quality=q) for q in [0.30, 0.90, 0.60]]
        v = make_vec(dominant="ranging", confidence=0.80)
        results = scorer.score_batch(matches, v)
        for i in range(len(results) - 1):
            assert results[i].edge_score >= results[i + 1].edge_score

    def test_batch_filters_none_results(self, scorer):
        """Bad matches return None from score() — must be filtered."""
        bad  = make_match(quality=9.99)
        good = make_match(quality=0.80)
        v    = make_vec()
        results = scorer.score_batch([bad, good], v)
        assert len(results) == 1
        assert results[0].base_score == 0.80

    def test_empty_batch_returns_empty_list(self, scorer):
        assert scorer.score_batch([], make_vec()) == []

    def test_none_batch_returns_empty_list(self, scorer):
        assert scorer.score_batch(None, make_vec()) == []

    def test_all_patterns_in_batch(self, scorer):
        """Scoring all four patterns at once works correctly."""
        matches = [make_match(quality=0.80, pattern=p) for p in PATTERNS]
        v = make_vec(dominant="reversal", confidence=0.85)
        results = scorer.score_batch(matches, v)
        assert len(results) == 4
        for s in results:
            assert s.state_discount == 1.00   # reversal → full credit


# ---------------------------------------------------------------------------
# Group 8: Never-raise contract
# ---------------------------------------------------------------------------

class TestNeverRaiseContract:

    @pytest.mark.parametrize("bad_match", [None, {}, "string", 42, 3.14])
    def test_bad_match_returns_none(self, scorer, bad_match):
        result = scorer.score(bad_match, make_vec())
        assert result is None

    @pytest.mark.parametrize("bad_vec", [None, {}, "string", 42])
    def test_bad_vec_returns_none(self, scorer, bad_vec):
        result = scorer.score(make_match(), bad_vec)
        assert result is None

    def test_both_none_returns_none(self, scorer):
        assert scorer.score(None, None) is None

    def test_repeated_calls_never_raise(self, scorer):
        """50 consecutive calls with valid inputs must all succeed."""
        m = make_match()
        v = make_vec()
        for _ in range(50):
            s = scorer.score(m, v)
            assert s is not None


# ---------------------------------------------------------------------------
# Group 9: Axiom compliance (deterministic)
# ---------------------------------------------------------------------------

class TestAxiomCompliance:

    def test_axiom1_boundedness_all_state_pattern_combinations(self, scorer):
        """Axiom 1: edge_score ∈ [0.0, 1.0] for every combination."""
        for state in STATES:
            for pattern in PATTERNS:
                s = score_ok(quality=0.80, dominant=state,
                             confidence=0.75, pattern=pattern)
                assert 0.0 <= s.edge_score <= 1.0

    def test_axiom2_monotonicity_base_score(self, scorer):
        """Axiom 2: higher base → higher edge (all else equal)."""
        v = make_vec(dominant="ranging", confidence=0.72)
        scores = []
        for q in [0.10, 0.30, 0.50, 0.70, 0.90]:
            m = make_match(quality=q)
            s = scorer.score(m, v)
            assert s is not None
            scores.append(s.edge_score)
        # Strictly increasing
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], (
                f"Not monotonic at index {i}: {scores}"
            )

    def test_axiom2_monotonicity_confidence(self, scorer):
        """Axiom 2: higher confidence → higher edge (all else equal)."""
        m = make_match(quality=0.80)
        scores = []
        for conf in [0.20, 0.40, 0.60, 0.80, 1.00]:
            rest = (1.0 - conf) / 5.0
            v = MarketStateVector(
                trending=rest, ranging=conf, expansion=rest,
                compression=rest, reversal=rest, news_chaos=rest,
            )
            s = scorer.score(m, v)
            assert s is not None
            scores.append(s.edge_score)
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]

    def test_axiom3_state_ceiling_no_amplification(self, scorer):
        """Axiom 3: edge_score ≤ base_score always (discount ≤ 1.0)."""
        for state in STATES:
            for quality in [0.50, 0.80, 1.00]:
                s = score_ok(quality=quality, dominant=state, confidence=1.00)
                assert s.edge_score <= s.base_score + 1e-10, (
                    f"Axiom 3 violated: edge={s.edge_score:.6f} "
                    f"> base={s.base_score:.6f} in state={state}"
                )

    def test_axiom4_chaos_near_zero(self, scorer):
        """Axiom 4: news_chaos discount=0.05 makes all signals near-zero."""
        for pattern in PATTERNS:
            m = make_match(quality=1.0, pattern=pattern)
            v = MarketStateVector(trending=0.04, ranging=0.04, expansion=0.04,
                                  compression=0.04, reversal=0.04, news_chaos=0.80)
            s = scorer.score(m, v)
            assert s is not None
            # edge = 1.0 * 0.05 * confidence <= 0.05
            assert s.edge_score <= 0.05 + 1e-10, (
                f"{pattern}/chaos: edge={s.edge_score:.6f} > 0.05"
            )

    def test_axiom4_chaos_below_min_edge_score(self, scorer):
        """Axiom 4 consequence: chaos signals never pass min_edge_score threshold."""
        for pattern in PATTERNS:
            m = make_match(quality=1.0, pattern=pattern)
            v = MarketStateVector(trending=0.04, ranging=0.04, expansion=0.04,
                                  compression=0.04, reversal=0.04, news_chaos=0.80)
            s = scorer.score(m, v)
            assert s is not None
            assert s.edge_score < MS_CONFIG.min_edge_score, (
                f"{pattern}/chaos: edge={s.edge_score:.6f} "
                f">= min_edge_score={MS_CONFIG.min_edge_score}"
            )


# ---------------------------------------------------------------------------
# Group 10: Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------

# Hypothesis strategies for valid inputs
valid_quality    = st.floats(min_value=0.0, max_value=1.0,
                             allow_nan=False, allow_infinity=False)
valid_confidence = st.floats(min_value=0.02, max_value=1.0,
                             allow_nan=False, allow_infinity=False)
valid_state      = st.sampled_from(STATES)
valid_pattern    = st.sampled_from(PATTERNS)


@settings(max_examples=200, deadline=2000)
@given(
    quality    = valid_quality,
    dominant   = valid_state,
    confidence = valid_confidence,
    pattern    = valid_pattern,
)
def test_property_boundedness(quality, dominant, confidence, pattern):
    """
    Property 1 (Axiom 1): For ALL valid inputs, edge_score ∈ [0.0, 1.0].
    200 random combinations.
    """
    scorer = PatternScorer()
    m = make_match(quality=quality, pattern=pattern)
    rest = (1.0 - confidence) / 5.0
    kw   = {s: rest for s in STATES}
    kw[dominant] = confidence
    v = MarketStateVector(**kw)
    s = scorer.score(m, v)
    assert s is not None
    assert 0.0 <= s.edge_score <= 1.0, (
        f"Boundedness violated: edge={s.edge_score} "
        f"quality={quality} state={dominant} conf={confidence}"
    )


@settings(max_examples=200, deadline=2000)
@given(
    quality    = valid_quality,
    dominant   = valid_state,
    confidence = valid_confidence,
    pattern    = valid_pattern,
)
def test_property_edge_never_exceeds_base(quality, dominant, confidence, pattern):
    """
    Property 2 (Axiom 3): edge_score ≤ base_score always.
    State discount ≤ 1.0 guarantees this — test it exhaustively.
    """
    scorer = PatternScorer()
    m    = make_match(quality=quality, pattern=pattern)
    rest = (1.0 - confidence) / 5.0
    kw   = {s: rest for s in STATES}
    kw[dominant] = confidence
    v = MarketStateVector(**kw)
    s = scorer.score(m, v)
    assert s is not None
    assert s.edge_score <= s.base_score + 1e-10, (
        f"Axiom 3 violated: edge={s.edge_score:.8f} > base={s.base_score:.8f}"
    )


@settings(max_examples=200, deadline=2000)
@given(
    base_a     = valid_quality,
    dominant   = valid_state,
    confidence = valid_confidence,
    pattern    = valid_pattern,
)
def test_property_monotonicity_base_score(base_a, dominant, confidence, pattern):
    """
    Property 3 (Axiom 2): If base_a > base_b, then edge_a > edge_b.
    (All else equal — same state, same pattern, same confidence.)
    """
    assume(base_a > 0.01)   # ensure there is room below
    base_b = base_a * 0.5   # strictly lower base

    scorer = PatternScorer()
    rest   = (1.0 - confidence) / 5.0
    kw     = {s: rest for s in STATES}
    kw[dominant] = confidence
    v = MarketStateVector(**kw)

    ma = make_match(quality=base_a, pattern=pattern)
    mb = make_match(quality=base_b, pattern=pattern)

    sa = scorer.score(ma, v)
    sb = scorer.score(mb, v)
    assert sa is not None and sb is not None

    assert sa.edge_score > sb.edge_score, (
        f"Monotonicity violated: base_a={base_a:.4f} > base_b={base_b:.4f} "
        f"but edge_a={sa.edge_score:.6f} <= edge_b={sb.edge_score:.6f}"
    )


@settings(max_examples=200, deadline=2000)
@given(
    quality    = valid_quality,
    dominant   = valid_state,
    conf_a     = valid_confidence,
    pattern    = valid_pattern,
)
def test_property_monotonicity_confidence(quality, dominant, conf_a, pattern):
    """
    Property 4 (Axiom 2): Higher confidence → higher edge_score,
    WHEN the dominant state stays the same in both vectors.

    Hypothesis found the correct precondition: monotonicity only holds
    when the dominant state doesn't change between the two vectors.
    Changing confidence can change the dominant state, which changes
    the state_discount, which can produce a higher edge despite lower conf.

    We enforce same dominant state via assume() — this is the mathematically
    correct scope for this property: "all else equal" means same state too.
    """
    assume(conf_a > 0.05)
    conf_b = conf_a * 0.5
    assume(conf_b > 0.02)

    # Precondition check BEFORE construction:
    # For `dominant` to remain the max-probability state, conf must exceed
    # all other states' probabilities. With 6 states and uniform rest:
    # rest = (1 - conf) / 5. For dominant to win: conf > rest
    # → conf > (1 - conf)/5 → 5*conf > 1 - conf → 6*conf > 1 → conf > 1/6
    assume(conf_a > 1/6)
    assume(conf_b > 1/6)
    assume(quality > 1e-6)

    scorer = PatternScorer()
    m = make_match(quality=quality, pattern=pattern)

    def vec(conf):
        rest = (1.0 - conf) / 5.0
        kw = {s: rest for s in STATES}
        kw[dominant] = conf
        return MarketStateVector(**kw)

    va = vec(conf_a)
    vb = vec(conf_b)

    # Post-construction verification (belt and suspenders)
    assume(va.dominant_state == dominant)
    assume(vb.dominant_state == dominant)

    sa = scorer.score(m, va)
    sb = scorer.score(m, vb)
    assert sa is not None and sb is not None

    # Same dominant state → same discount → monotonicity holds
    assert sa.state_discount == sb.state_discount

    if quality == 0.0:
        assert sa.edge_score == sb.edge_score == 0.0
    else:
        assert sa.edge_score > sb.edge_score, (
            f"Monotonicity violated: conf_a={conf_a:.4f} > conf_b={conf_b:.4f} "
            f"but edge_a={sa.edge_score:.6f} <= edge_b={sb.edge_score:.6f} "
            f"(dominant={dominant}, discount={sa.state_discount:.3f})"
        )


@settings(max_examples=200, deadline=2000)
@given(
    quality    = valid_quality,
    dominant   = valid_state,
    confidence = valid_confidence,
    pattern    = valid_pattern,
)
def test_property_product_consistency(quality, dominant, confidence, pattern):
    """
    Property 5: edge_score == base × discount × confidence within tolerance.
    """
    scorer = PatternScorer()
    m    = make_match(quality=quality, pattern=pattern)
    rest = (1.0 - confidence) / 5.0
    kw   = {s: rest for s in STATES}
    kw[dominant] = confidence
    v = MarketStateVector(**kw)
    s = scorer.score(m, v)
    assert s is not None
    expected = s.base_score * s.state_discount * s.confidence_weight
    assert math.isclose(s.edge_score, expected, abs_tol=1e-10), (
        f"Product inconsistency: edge={s.edge_score:.12f} "
        f"base×disc×conf={expected:.12f}"
    )


@settings(max_examples=100, deadline=2000)
@given(
    quality    = valid_quality,
    dominant   = valid_state,
    confidence = valid_confidence,
    pattern    = valid_pattern,
)
def test_property_determinism(quality, dominant, confidence, pattern):
    """
    Property 6: Same inputs → identical ScoredSignal every time.
    """
    scorer = PatternScorer()
    m    = make_match(quality=quality, pattern=pattern)
    rest = (1.0 - confidence) / 5.0
    kw   = {s: rest for s in STATES}
    kw[dominant] = confidence
    v = MarketStateVector(**kw)
    s1 = scorer.score(m, v)
    s2 = scorer.score(m, v)
    assert s1 is not None and s2 is not None
    assert s1.edge_score       == s2.edge_score
    assert s1.base_score       == s2.base_score
    assert s1.state_discount   == s2.state_discount
    assert s1.confidence_weight == s2.confidence_weight


# ---------------------------------------------------------------------------
# Group 11: ScoredSignal invariants via scorer
# ---------------------------------------------------------------------------

class TestScoredSignalInvariantsViaScorer:

    def test_scored_signal_passes_all_invariants(self, scorer):
        """ScoredSignal construction succeeds for all valid inputs."""
        for state in STATES:
            for pattern in PATTERNS:
                s = score_ok(quality=0.80, dominant=state,
                             confidence=0.75, pattern=pattern)
                assert isinstance(s, ScoredSignal)

    def test_edge_score_copy_consistent_with_scored(self, scorer):
        """edge_score field == base × discount × confidence within tolerance."""
        s = score_ok(quality=0.84, dominant="ranging", confidence=0.72)
        assert math.isclose(
            s.edge_score,
            s.base_score * s.state_discount * s.confidence_weight,
            abs_tol=1e-10,
        )

    def test_dominant_state_matches_vector(self, scorer):
        """dominant_state field == vector.dominant_state."""
        for state in STATES:
            s = score_ok(quality=0.80, dominant=state, confidence=0.75)
            assert s.dominant_state == s.state_vector.dominant_state

    def test_prz_fields_forwarded_correctly(self, scorer):
        """PRZ trading levels must be forwarded correctly from PatternMatch.prz."""
        s = score_ok(quality=0.80)
        assert s.entry   == 61500.0
        assert s.stop    == 59800.0
        assert s.target1 == 64000.0
        assert s.target2 == 65000.0
        assert s.target3 == 66000.0

    def test_passes_min_threshold_correct(self, scorer):
        """passes_min_threshold must match edge_score vs min_edge_score."""
        s_high = score_ok(quality=0.90, dominant="reversal", confidence=0.90)
        s_low  = score_ok(quality=0.10, dominant="trending", confidence=0.30)
        assert s_high.passes_min_threshold == (
            s_high.edge_score >= MS_CONFIG.min_edge_score
        )
        assert s_low.passes_min_threshold == (
            s_low.edge_score >= MS_CONFIG.min_edge_score
        )