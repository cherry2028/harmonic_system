"""tests/test_validator.py — Exhaustive tests for data/validator.py.

Test groups
-----------
 1.  ValidationFinding — construction and to_dict()
 2.  ValidationReport  — properties, serialization, __str__
 3.  validate() backwards compatibility
 4.  Fatal: None / empty DataFrame
 5.  Fatal: Insufficient bars
 6.  Fatal: Missing columns
 7.  Fatal: Non-DatetimeIndex
 8.  Fatal: Non-monotonic timestamps
 9.  Fatal: Excessive NaN
10.  Warning: Duplicate timestamps
11.  Warning: Gaps in timestamp sequence
12.  Warning: high < low
13.  Warning: Open outside [low, high]
14.  Warning: Close outside [low, high]
15.  Warning: Negative / zero volume
16.  Warning: NaN below fatal threshold
17.  Finding codes — stable identifiers
18.  Determinism — same df → same report
19.  No mutation — source DataFrame unchanged
20.  validate_detailed() metadata correctness
21.  Combined findings — multiple issues in one DataFrame
"""
from __future__ import annotations

import json
from typing import Set

import numpy as np
import pandas as pd
import pytest

from data.validator import (
    DataValidator,
    ValidationFinding,
    ValidationReport,
    _NAN_FATAL_THRESHOLD,
    _REQUIRED_COLUMNS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _good_df(n: int = 100, freq: str = "1h") -> pd.DataFrame:
    """Minimal clean OHLCV DataFrame with DatetimeIndex."""
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    return pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100.0},
        index=idx,
    )


def _codes(report: ValidationReport) -> Set[str]:
    return {f.code for f in report.findings}


@pytest.fixture
def v() -> DataValidator:
    return DataValidator()


@pytest.fixture
def good_df() -> pd.DataFrame:
    return _good_df()


# ─────────────────────────────────────────────────────────────────────────────
# 1. ValidationFinding
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationFinding:

    def test_frozen(self):
        f = ValidationFinding(level="fatal", code="X", message="m")
        with pytest.raises(Exception):
            f.level = "warning"  # type: ignore

    def test_is_fatal_true(self):
        assert ValidationFinding(level="fatal", code="X", message="m").is_fatal()

    def test_is_fatal_false(self):
        assert not ValidationFinding(level="warning", code="X", message="m").is_fatal()

    def test_count_defaults_to_zero(self):
        f = ValidationFinding(level="warning", code="X", message="m")
        assert f.count == 0

    def test_to_dict_keys(self):
        f = ValidationFinding(level="warning", code="HIGH_LT_LOW", message="msg", count=5)
        d = f.to_dict()
        assert set(d.keys()) == {"level", "code", "message", "count"}

    def test_to_dict_values(self):
        f = ValidationFinding(level="warning", code="HIGH_LT_LOW", message="msg", count=5)
        d = f.to_dict()
        assert d["level"] == "warning"
        assert d["code"] == "HIGH_LT_LOW"
        assert d["message"] == "msg"
        assert d["count"] == 5

    def test_to_dict_all_primitives(self):
        f = ValidationFinding(level="fatal", code="X", message="m", count=3)
        for v in f.to_dict().values():
            assert isinstance(v, (str, int, float, bool))


# ─────────────────────────────────────────────────────────────────────────────
# 2. ValidationReport properties and serialization
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationReport:

    @pytest.fixture
    def clean_report(self, v, good_df) -> ValidationReport:
        return v.validate_detailed(good_df, "BTC", "1h")

    def test_frozen(self, clean_report):
        with pytest.raises(Exception):
            clean_report.passed = False  # type: ignore

    def test_passed_true_on_clean_data(self, clean_report):
        assert clean_report.passed is True

    def test_fatal_count_zero_on_clean(self, clean_report):
        assert clean_report.fatal_count == 0

    def test_warning_count_zero_on_clean(self, clean_report):
        assert clean_report.warning_count == 0

    def test_has_timestamp_issues_false_on_clean(self, clean_report):
        assert clean_report.has_timestamp_issues is False

    def test_has_ohlc_issues_false_on_clean(self, clean_report):
        assert clean_report.has_ohlc_issues is False

    def test_has_gap_issues_false_on_clean(self, clean_report):
        assert clean_report.has_gap_issues is False

    def test_ohlc_bad_count_zero_on_clean(self, clean_report):
        assert clean_report.ohlc_bad_count == 0

    def test_to_dict_keys(self, clean_report):
        d = clean_report.to_dict()
        required = {
            "symbol", "timeframe", "total_bars", "passed",
            "fatal_count", "warning_count",
            "has_timestamp_issues", "has_ohlc_issues", "has_gap_issues",
            "ohlc_bad_count", "findings",
        }
        assert required.issubset(set(d.keys()))

    def test_to_dict_all_primitives_except_findings(self, clean_report):
        d = clean_report.to_dict()
        for k, val in d.items():
            if k == "findings":
                assert isinstance(val, list)
            else:
                assert isinstance(val, (str, int, float, bool)), \
                    f"Key {k!r} has type {type(val).__name__}"

    def test_to_json_valid_json(self, clean_report):
        j = clean_report.to_json()
        assert isinstance(j, str)
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_to_json_deterministic(self, clean_report):
        assert clean_report.to_json() == clean_report.to_json()

    def test_str_contains_status(self, clean_report):
        assert "PASSED" in str(clean_report)

    def test_str_contains_symbol(self, clean_report):
        assert "BTC" in str(clean_report)

    def test_str_failed_report(self, v):
        r = v.validate_detailed(None)
        assert "FAILED" in str(r)

    def test_findings_are_tuple(self, clean_report):
        assert isinstance(clean_report.findings, tuple)

    def test_fatals_before_warnings_in_findings(self, v):
        """Fatals are always listed before warnings."""
        # Create a df that triggers both fatal (NaN>5%) and warning (dup)
        idx = pd.date_range("2024-01-01", periods=60, freq="1h")
        df = pd.DataFrame(
            {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0},
            index=idx,
        )
        # Inject >5% NaN to trigger fatal
        df.loc[df.index[:4], "close"] = float("nan")
        r = v.validate_detailed(df, "X")
        if r.findings:
            seen_warning = False
            for f in r.findings:
                if not f.is_fatal():
                    seen_warning = True
                if f.is_fatal() and seen_warning:
                    pytest.fail("Fatal finding appeared after a warning")


# ─────────────────────────────────────────────────────────────────────────────
# 3. validate() backwards compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateBackwardsCompat:

    def test_returns_bool(self, v, good_df):
        result = v.validate(good_df)
        assert isinstance(result, bool)

    def test_valid_df_returns_true(self, v, good_df):
        assert v.validate(good_df, "BTC") is True

    def test_none_returns_false(self, v):
        assert v.validate(None) is False

    def test_empty_returns_false(self, v):
        assert v.validate(pd.DataFrame()) is False

    def test_too_few_bars_returns_false(self, v):
        assert v.validate(_good_df(n=40)) is False

    def test_missing_col_returns_false(self, v):
        df = _good_df().drop(columns=["volume"])
        assert v.validate(df) is False

    def test_nan_above_threshold_returns_false(self, v):
        df = _good_df()
        df.loc[df.index[:6], "close"] = float("nan")  # 6% > 5%
        assert v.validate(df) is False

    def test_warnings_only_returns_true(self, v):
        """High < low is a warning; validate() must still return True."""
        df = _good_df()
        df.loc[df.index[0], "high"] = 0.1  # high < low
        assert v.validate(df) is True

    def test_min_bars_class_attribute_unchanged(self):
        assert DataValidator.MIN_BARS == 50

    def test_symbol_default_empty_string(self, v, good_df):
        """validate(df) without symbol must not raise."""
        assert v.validate(good_df) is True


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fatal: None / empty
# ─────────────────────────────────────────────────────────────────────────────

class TestFatalEmpty:

    def test_none_df_is_fatal(self, v):
        r = v.validate_detailed(None)
        assert r.passed is False
        assert "EMPTY_DATAFRAME" in _codes(r)

    def test_empty_df_is_fatal(self, v):
        r = v.validate_detailed(pd.DataFrame())
        assert r.passed is False
        assert "EMPTY_DATAFRAME" in _codes(r)

    def test_empty_total_bars_zero(self, v):
        r = v.validate_detailed(None)
        assert r.total_bars == 0

    def test_empty_short_circuits(self, v):
        """No further checks run after empty detection."""
        r = v.validate_detailed(None)
        assert len(r.findings) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. Fatal: Insufficient bars
# ─────────────────────────────────────────────────────────────────────────────

class TestFatalInsufficientBars:

    def test_49_bars_fails(self, v):
        r = v.validate_detailed(_good_df(n=49))
        assert r.passed is False
        assert "INSUFFICIENT_BARS" in _codes(r)

    def test_50_bars_passes(self, v):
        r = v.validate_detailed(_good_df(n=50))
        assert r.passed is True
        assert "INSUFFICIENT_BARS" not in _codes(r)

    def test_finding_count_reflects_actual_bars(self, v):
        r = v.validate_detailed(_good_df(n=30))
        finding = next(f for f in r.findings if f.code == "INSUFFICIENT_BARS")
        assert finding.count == 30

    def test_short_circuits_on_insufficient_bars(self, v):
        """Only the insufficient-bars fatal should be in findings."""
        r = v.validate_detailed(_good_df(n=10))
        assert len(r.findings) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. Fatal: Missing columns
# ─────────────────────────────────────────────────────────────────────────────

class TestFatalMissingColumns:

    @pytest.mark.parametrize("missing_col", sorted(_REQUIRED_COLUMNS))
    def test_each_required_column_individually(self, v, missing_col):
        df = _good_df().drop(columns=[missing_col])
        r = v.validate_detailed(df)
        assert r.passed is False
        assert "MISSING_COLUMNS" in _codes(r)

    def test_missing_multiple_columns(self, v):
        df = _good_df().drop(columns=["open", "volume"])
        r = v.validate_detailed(df)
        assert "MISSING_COLUMNS" in _codes(r)

    def test_missing_columns_short_circuits(self, v):
        df = _good_df().drop(columns=["high"])
        r = v.validate_detailed(df)
        assert sum(1 for f in r.findings if f.is_fatal()) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 7. Fatal: Non-DatetimeIndex
# ─────────────────────────────────────────────────────────────────────────────

class TestFatalNonDatetimeIndex:

    def test_integer_index_is_fatal(self, v):
        df = _good_df().reset_index(drop=True)
        r = v.validate_detailed(df)
        assert r.passed is False
        assert "NOT_DATETIME_INDEX" in _codes(r)

    def test_string_index_is_fatal(self, v):
        df = _good_df()
        df.index = [str(i) for i in range(len(df))]
        r = v.validate_detailed(df)
        assert "NOT_DATETIME_INDEX" in _codes(r)

    def test_non_datetime_still_runs_ohlcv_checks(self, v):
        """Even with bad index, OHLCV checks are attempted."""
        df = _good_df().reset_index(drop=True)
        # Inject a bad candle — check must still appear in findings
        df.loc[0, "high"] = 0.1   # high < low
        r = v.validate_detailed(df)
        assert "NOT_DATETIME_INDEX" in _codes(r)
        assert "HIGH_LT_LOW" in _codes(r)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Fatal: Non-monotonic timestamps
# ─────────────────────────────────────────────────────────────────────────────

class TestFatalNonMonotonic:

    def test_reversed_index_is_fatal(self, v):
        df = _good_df()
        df = df.iloc[::-1]  # reverse
        r = v.validate_detailed(df)
        assert r.passed is False
        assert "NOT_MONOTONIC" in _codes(r)

    def test_one_inversion_is_fatal(self, v):
        df = _good_df()
        idx = list(df.index)
        idx[5], idx[6] = idx[6], idx[5]   # swap two timestamps
        df.index = pd.DatetimeIndex(idx)
        r = v.validate_detailed(df)
        assert "NOT_MONOTONIC" in _codes(r)

    def test_inversion_count_recorded(self, v):
        df = _good_df()
        idx = list(df.index)
        idx[5], idx[6] = idx[6], idx[5]
        df.index = pd.DatetimeIndex(idx)
        r = v.validate_detailed(df)
        f = next(f for f in r.findings if f.code == "NOT_MONOTONIC")
        assert f.count >= 1

    def test_non_monotonic_still_runs_ohlcv_checks(self, v):
        """OHLCV checks run even when timestamps are out of order."""
        df = _good_df()
        df = df.iloc[::-1]
        df.loc[df.index[0], "high"] = 0.1   # high < low
        r = v.validate_detailed(df)
        assert "HIGH_LT_LOW" in _codes(r)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Fatal: Excessive NaN
# ─────────────────────────────────────────────────────────────────────────────

class TestFatalExcessiveNaN:

    def test_more_than_5pct_nan_is_fatal(self, v):
        df = _good_df()
        df.loc[df.index[:6], "close"] = float("nan")   # 6% > 5%
        r = v.validate_detailed(df)
        assert r.passed is False
        assert "EXCESSIVE_NAN" in _codes(r)

    def test_exactly_5pct_nan_is_not_fatal(self, v):
        df = _good_df()
        df.loc[df.index[:5], "close"] = float("nan")   # exactly 5%
        r = v.validate_detailed(df)
        assert "EXCESSIVE_NAN" not in _codes(r)
        # It should be a warning instead
        assert "NAN_PRESENT" in _codes(r)

    def test_nan_count_recorded(self, v):
        df = _good_df()
        df.loc[df.index[:6], "close"] = float("nan")
        df.loc[df.index[:6], "volume"] = float("nan")
        r = v.validate_detailed(df)
        f = next(f for f in r.findings if f.code == "EXCESSIVE_NAN")
        assert f.count == 12   # 6 close + 6 volume

    def test_nan_in_single_column_triggers_fatal(self, v):
        df = _good_df()
        df.loc[df.index[:6], "high"] = float("nan")
        r = v.validate_detailed(df)
        assert "EXCESSIVE_NAN" in _codes(r)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Warning: Duplicate timestamps
# ─────────────────────────────────────────────────────────────────────────────

class TestWarningDuplicateTimestamps:

    def test_one_duplicate_detected(self, v):
        df = _good_df()
        idx = list(df.index)
        idx[10] = idx[9]   # create one duplicate
        df.index = pd.DatetimeIndex(idx)
        # Must still be monotonically increasing or the other fatal fires
        # Sort to avoid NOT_MONOTONIC masking the duplicate
        df = df.sort_index()
        r = v.validate_detailed(df)
        assert "DUPLICATE_TIMESTAMPS" in _codes(r)

    def test_duplicate_count_correct(self, v):
        df = _good_df()
        # Build df with 3 deliberate duplicates via pd.concat
        extra = df.iloc[:3].copy()
        df2 = pd.concat([df, extra]).sort_index()
        r = v.validate_detailed(df2)
        f = next((f for f in r.findings if f.code == "DUPLICATE_TIMESTAMPS"), None)
        assert f is not None
        assert f.count == 3

    def test_duplicate_is_warning_not_fatal(self, v):
        df = _good_df()
        extra = df.iloc[:2].copy()
        df2 = pd.concat([df, extra]).sort_index()
        r = v.validate_detailed(df2)
        f = next(f for f in r.findings if f.code == "DUPLICATE_TIMESTAMPS")
        assert f.level == "warning"
        assert r.passed is True

    def test_no_duplicates_no_finding(self, v, good_df):
        r = v.validate_detailed(good_df)
        assert "DUPLICATE_TIMESTAMPS" not in _codes(r)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Warning: Gaps in timestamp sequence
# ─────────────────────────────────────────────────────────────────────────────

class TestWarningGaps:

    def test_single_gap_detected(self, v):
        df = _good_df()
        # Drop 3 bars in the middle to create a gap
        df = df.drop(df.index[50:53])
        r = v.validate_detailed(df, timeframe="1h")
        assert "GAPS_DETECTED" in _codes(r)

    def test_gap_count_reflects_locations(self, v):
        df = _good_df()
        df = df.drop(df.index[30:33])   # gap 1
        df = df.drop(df.index[60:62])   # gap 2 (indices shift, close enough)
        r = v.validate_detailed(df, timeframe="1h")
        f = next(f for f in r.findings if f.code == "GAPS_DETECTED")
        assert f.count >= 1   # at least one gap location found

    def test_gap_is_warning_not_fatal(self, v):
        df = _good_df()
        df = df.drop(df.index[50:53])
        r = v.validate_detailed(df)
        if "GAPS_DETECTED" in _codes(r):
            f = next(f for f in r.findings if f.code == "GAPS_DETECTED")
            assert f.level == "warning"
            assert r.passed is True

    def test_no_gaps_no_finding(self, v, good_df):
        r = v.validate_detailed(good_df)
        assert "GAPS_DETECTED" not in _codes(r)

    def test_two_bar_df_no_crash(self, v):
        """Edge case: only 2 timestamps → diffs has 1 entry."""
        df = _good_df(n=60)[:2]
        # Only 2 bars — won't reach gap check (< MIN_BARS), but testing _check_gaps directly
        from data.validator import DataValidator as DV
        result = DV._check_gaps(df, "X", "1h")
        assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Warning: high < low
# ─────────────────────────────────────────────────────────────────────────────

class TestWarningHighLtLow:

    def test_one_bad_candle_detected(self, v):
        df = _good_df()
        df.loc[df.index[0], "high"] = 0.1   # high=0.1 < low=0.5
        r = v.validate_detailed(df)
        assert "HIGH_LT_LOW" in _codes(r)

    def test_count_correct(self, v):
        df = _good_df()
        df.loc[df.index[0], "high"] = 0.1
        df.loc[df.index[1], "high"] = 0.2
        r = v.validate_detailed(df)
        f = next(f for f in r.findings if f.code == "HIGH_LT_LOW")
        assert f.count == 2

    def test_is_warning_not_fatal(self, v):
        df = _good_df()
        df.loc[df.index[0], "high"] = 0.1
        r = v.validate_detailed(df)
        f = next(f for f in r.findings if f.code == "HIGH_LT_LOW")
        assert f.level == "warning"
        assert r.passed is True

    def test_clean_df_no_finding(self, v, good_df):
        r = v.validate_detailed(good_df)
        assert "HIGH_LT_LOW" not in _codes(r)

    def test_has_ohlc_issues_true(self, v):
        df = _good_df()
        df.loc[df.index[0], "high"] = 0.1
        r = v.validate_detailed(df)
        assert r.has_ohlc_issues is True


# ─────────────────────────────────────────────────────────────────────────────
# 13. Warning: Open outside [low, high]
# ─────────────────────────────────────────────────────────────────────────────

class TestWarningOpenOutsideHL:

    def test_open_above_high_detected(self, v):
        df = _good_df()
        df.loc[df.index[5], "open"] = 99.0   # open=99 > high=2
        r = v.validate_detailed(df)
        assert "OPEN_OUTSIDE_HL" in _codes(r)

    def test_open_below_low_detected(self, v):
        df = _good_df()
        df.loc[df.index[5], "open"] = 0.01   # open=0.01 < low=0.5
        r = v.validate_detailed(df)
        assert "OPEN_OUTSIDE_HL" in _codes(r)

    def test_count_correct(self, v):
        df = _good_df()
        df.loc[df.index[0], "open"] = 99.0
        df.loc[df.index[1], "open"] = 0.01
        r = v.validate_detailed(df)
        f = next(f for f in r.findings if f.code == "OPEN_OUTSIDE_HL")
        assert f.count == 2

    def test_is_warning_not_fatal(self, v):
        df = _good_df()
        df.loc[df.index[0], "open"] = 99.0
        r = v.validate_detailed(df)
        f = next(f for f in r.findings if f.code == "OPEN_OUTSIDE_HL")
        assert f.level == "warning"
        assert r.passed is True

    def test_clean_df_no_finding(self, v, good_df):
        r = v.validate_detailed(good_df)
        assert "OPEN_OUTSIDE_HL" not in _codes(r)


# ─────────────────────────────────────────────────────────────────────────────
# 14. Warning: Close outside [low, high]
# ─────────────────────────────────────────────────────────────────────────────

class TestWarningCloseOutsideHL:

    def test_close_above_high_detected(self, v):
        df = _good_df()
        df.loc[df.index[5], "close"] = 99.0
        r = v.validate_detailed(df)
        assert "CLOSE_OUTSIDE_HL" in _codes(r)

    def test_close_below_low_detected(self, v):
        df = _good_df()
        df.loc[df.index[5], "close"] = 0.01
        r = v.validate_detailed(df)
        assert "CLOSE_OUTSIDE_HL" in _codes(r)

    def test_count_correct(self, v):
        df = _good_df()
        df.loc[df.index[0], "close"] = 99.0
        df.loc[df.index[1], "close"] = 0.01
        r = v.validate_detailed(df)
        f = next(f for f in r.findings if f.code == "CLOSE_OUTSIDE_HL")
        assert f.count == 2

    def test_is_warning_not_fatal(self, v):
        df = _good_df()
        df.loc[df.index[0], "close"] = 99.0
        r = v.validate_detailed(df)
        f = next(f for f in r.findings if f.code == "CLOSE_OUTSIDE_HL")
        assert f.level == "warning"
        assert r.passed is True

    def test_ohlc_bad_count_aggregates_all_ohlc_issues(self, v):
        """ohlc_bad_count = sum of counts for HIGH_LT_LOW + OPEN_OUTSIDE_HL + CLOSE_OUTSIDE_HL."""
        df = _good_df()
        df.loc[df.index[0], "high"]  = 0.1    # HIGH_LT_LOW: 1
        df.loc[df.index[1], "open"]  = 99.0   # OPEN_OUTSIDE_HL: 1
        df.loc[df.index[2], "close"] = 99.0   # CLOSE_OUTSIDE_HL: 1
        r = v.validate_detailed(df)
        assert r.ohlc_bad_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# 15. Warning: Negative / zero volume
# ─────────────────────────────────────────────────────────────────────────────

class TestWarningNegativeVolume:

    def test_zero_volume_detected(self, v):
        df = _good_df()
        df.loc[df.index[10], "volume"] = 0.0
        r = v.validate_detailed(df)
        assert "NEGATIVE_VOLUME" in _codes(r)

    def test_negative_volume_detected(self, v):
        df = _good_df()
        df.loc[df.index[10], "volume"] = -5.0
        r = v.validate_detailed(df)
        assert "NEGATIVE_VOLUME" in _codes(r)

    def test_count_correct(self, v):
        df = _good_df()
        df.loc[df.index[0], "volume"] = 0.0
        df.loc[df.index[1], "volume"] = -1.0
        r = v.validate_detailed(df)
        f = next(f for f in r.findings if f.code == "NEGATIVE_VOLUME")
        assert f.count == 2

    def test_is_warning_not_fatal(self, v):
        df = _good_df()
        df.loc[df.index[0], "volume"] = 0.0
        r = v.validate_detailed(df)
        f = next(f for f in r.findings if f.code == "NEGATIVE_VOLUME")
        assert f.level == "warning"
        assert r.passed is True

    def test_positive_volume_no_finding(self, v, good_df):
        r = v.validate_detailed(good_df)
        assert "NEGATIVE_VOLUME" not in _codes(r)


# ─────────────────────────────────────────────────────────────────────────────
# 16. Warning: NaN below fatal threshold
# ─────────────────────────────────────────────────────────────────────────────

class TestWarningNaNBelowThreshold:

    def test_3pct_nan_is_warning(self, v):
        df = _good_df()
        df.loc[df.index[:3], "close"] = float("nan")   # 3% < 5%
        r = v.validate_detailed(df)
        assert "NAN_PRESENT" in _codes(r)
        assert "EXCESSIVE_NAN" not in _codes(r)
        assert r.passed is True

    def test_nan_count_correct(self, v):
        df = _good_df()
        df.loc[df.index[:3], "close"] = float("nan")
        df.loc[df.index[:2], "volume"] = float("nan")
        r = v.validate_detailed(df)
        f = next(f for f in r.findings if f.code == "NAN_PRESENT")
        assert f.count == 5   # 3 + 2

    def test_no_nan_no_finding(self, v, good_df):
        r = v.validate_detailed(good_df)
        assert "NAN_PRESENT" not in _codes(r)
        assert "EXCESSIVE_NAN" not in _codes(r)


# ─────────────────────────────────────────────────────────────────────────────
# 17. Finding codes — stable identifiers
# ─────────────────────────────────────────────────────────────────────────────

class TestFindingCodes:
    """All codes must be exactly the documented stable strings."""

    ALL_CODES = {
        "EMPTY_DATAFRAME",
        "INSUFFICIENT_BARS",
        "MISSING_COLUMNS",
        "NOT_DATETIME_INDEX",
        "NOT_MONOTONIC",
        "EXCESSIVE_NAN",
        "DUPLICATE_TIMESTAMPS",
        "GAPS_DETECTED",
        "HIGH_LT_LOW",
        "OPEN_OUTSIDE_HL",
        "CLOSE_OUTSIDE_HL",
        "NEGATIVE_VOLUME",
        "NAN_PRESENT",
    }

    def test_no_undocumented_codes_emitted(self, v):
        """Trigger as many checks as possible; all codes must be in ALL_CODES."""
        df = _good_df()
        df.loc[df.index[0], "high"]   = 0.1    # HIGH_LT_LOW
        df.loc[df.index[1], "open"]   = 99.0   # OPEN_OUTSIDE_HL
        df.loc[df.index[2], "close"]  = 99.0   # CLOSE_OUTSIDE_HL
        df.loc[df.index[3], "volume"] = -1.0   # NEGATIVE_VOLUME
        extra = df.iloc[:2].copy()
        df = pd.concat([df, extra]).sort_index()  # DUPLICATE_TIMESTAMPS
        r = v.validate_detailed(df)
        for finding in r.findings:
            assert finding.code in self.ALL_CODES, \
                f"Undocumented code: {finding.code!r}"

    @pytest.mark.parametrize("code", [
        "EMPTY_DATAFRAME", "INSUFFICIENT_BARS", "MISSING_COLUMNS",
        "NOT_DATETIME_INDEX", "NOT_MONOTONIC", "EXCESSIVE_NAN",
        "DUPLICATE_TIMESTAMPS", "HIGH_LT_LOW", "OPEN_OUTSIDE_HL",
        "CLOSE_OUTSIDE_HL", "NEGATIVE_VOLUME", "NAN_PRESENT",
    ])
    def test_code_is_uppercase_with_underscores(self, code):
        assert code == code.upper()
        assert " " not in code


# ─────────────────────────────────────────────────────────────────────────────
# 18. Determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:

    def test_same_df_same_report_json(self, v):
        df = _good_df()
        r1 = v.validate_detailed(df, "BTC", "1h")
        r2 = v.validate_detailed(df, "BTC", "1h")
        assert r1.to_json() == r2.to_json()

    def test_same_df_same_findings(self, v):
        df = _good_df()
        df.loc[df.index[0], "high"] = 0.1
        r1 = v.validate_detailed(df)
        r2 = v.validate_detailed(df)
        assert len(r1.findings) == len(r2.findings)
        for f1, f2 in zip(r1.findings, r2.findings):
            assert f1.code  == f2.code
            assert f1.level == f2.level
            assert f1.count == f2.count

    def test_different_dfs_different_reports(self, v):
        df1 = _good_df()
        df2 = _good_df()
        df2.loc[df2.index[0], "high"] = 0.1
        r1 = v.validate_detailed(df1)
        r2 = v.validate_detailed(df2)
        assert r1.to_json() != r2.to_json()

    def test_validate_detailed_and_validate_agree(self, v):
        """validate() and validate_detailed().passed must always agree."""
        for df in [
            _good_df(),
            _good_df(n=40),
            None,
            pd.DataFrame(),
        ]:
            r = v.validate_detailed(df)
            b = v.validate(df)
            assert r.passed == b, \
                f"Disagreement: passed={r.passed} validate()={b}"


# ─────────────────────────────────────────────────────────────────────────────
# 19. No mutation of source DataFrame
# ─────────────────────────────────────────────────────────────────────────────

class TestNoMutation:

    def test_source_df_not_mutated_clean(self, v, good_df):
        original = good_df.copy(deep=True)
        v.validate_detailed(good_df, "BTC", "1h")
        pd.testing.assert_frame_equal(good_df, original)

    def test_source_df_not_mutated_with_bad_candles(self, v):
        df = _good_df()
        df.loc[df.index[0], "high"] = 0.1
        original = df.copy(deep=True)
        v.validate_detailed(df)
        pd.testing.assert_frame_equal(df, original)

    def test_source_index_not_mutated_with_duplicates(self, v):
        df = _good_df()
        extra = df.iloc[:2].copy()
        df_with_dups = pd.concat([df, extra]).sort_index()
        idx_before = list(df_with_dups.index)
        v.validate_detailed(df_with_dups)
        assert list(df_with_dups.index) == idx_before

    def test_source_values_not_mutated_with_nan(self, v):
        df = _good_df()
        df.loc[df.index[:3], "close"] = float("nan")
        original = df.copy(deep=True)
        v.validate_detailed(df)
        pd.testing.assert_frame_equal(df, original)


# ─────────────────────────────────────────────────────────────────────────────
# 20. validate_detailed() metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadata:

    def test_symbol_stored(self, v, good_df):
        r = v.validate_detailed(good_df, symbol="ETHUSDT")
        assert r.symbol == "ETHUSDT"

    def test_timeframe_stored(self, v, good_df):
        r = v.validate_detailed(good_df, timeframe="4h")
        assert r.timeframe == "4h"

    def test_symbol_default_empty(self, v, good_df):
        r = v.validate_detailed(good_df)
        assert r.symbol == ""

    def test_timeframe_default_empty(self, v, good_df):
        r = v.validate_detailed(good_df)
        assert r.timeframe == ""

    def test_total_bars_correct(self, v):
        for n in [50, 100, 500]:
            r = v.validate_detailed(_good_df(n=n))
            assert r.total_bars == n

    def test_to_dict_includes_symbol_and_timeframe(self, v, good_df):
        r = v.validate_detailed(good_df, "BTCUSDT", "1h")
        d = r.to_dict()
        assert d["symbol"] == "BTCUSDT"
        assert d["timeframe"] == "1h"


# ─────────────────────────────────────────────────────────────────────────────
# 21. Combined findings
# ─────────────────────────────────────────────────────────────────────────────

class TestCombinedFindings:

    def test_multiple_warnings_all_recorded(self, v):
        df = _good_df()
        df.loc[df.index[0], "high"]   = 0.1    # HIGH_LT_LOW
        df.loc[df.index[1], "open"]   = 99.0   # OPEN_OUTSIDE_HL
        df.loc[df.index[2], "close"]  = 99.0   # CLOSE_OUTSIDE_HL
        df.loc[df.index[3], "volume"] = -1.0   # NEGATIVE_VOLUME
        r = v.validate_detailed(df)
        codes = _codes(r)
        assert "HIGH_LT_LOW"    in codes
        assert "OPEN_OUTSIDE_HL"  in codes
        assert "CLOSE_OUTSIDE_HL" in codes
        assert "NEGATIVE_VOLUME"  in codes
        assert r.passed is True   # all warnings, no fatals

    def test_fatal_plus_warnings_fails(self, v):
        df = _good_df()
        df.loc[df.index[0], "high"] = 0.1    # warning
        df.loc[df.index[:6], "close"] = float("nan")  # fatal
        r = v.validate_detailed(df)
        assert r.passed is False
        assert r.fatal_count >= 1
        assert r.warning_count >= 0

    def test_has_timestamp_issues_set_by_gaps(self, v):
        df = _good_df()
        df = df.drop(df.index[50:53])
        r = v.validate_detailed(df)
        assert r.has_timestamp_issues is True

    def test_has_timestamp_issues_set_by_duplicates(self, v):
        df = _good_df()
        extra = df.iloc[:2].copy()
        df2 = pd.concat([df, extra]).sort_index()
        r = v.validate_detailed(df2)
        assert r.has_timestamp_issues is True

    def test_warnings_still_logged_after_fatal_nan(self, v):
        """When NaN is fatal, OHLC warnings should NOT appear
        because NaN check runs after OHLC checks — so OHLC
        checks were already completed and logged."""
        df = _good_df()
        df.loc[df.index[:6], "close"] = float("nan")  # fatal NaN
        df.loc[df.index[0], "high"] = 0.1              # OHLC warning
        r = v.validate_detailed(df)
        # OHLC ran before NaN check — both should be present
        assert "EXCESSIVE_NAN" in _codes(r)
        assert "HIGH_LT_LOW" in _codes(r)