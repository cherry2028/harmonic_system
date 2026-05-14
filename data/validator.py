from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

_NAN_FATAL_THRESHOLD = 0.05
_REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


@dataclass(frozen=True)
class ValidationFinding:
    level: str
    code: str
    message: str
    count: int = 0

    def is_fatal(self) -> bool:
        return self.level == "fatal"

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "count": self.count,
        }


@dataclass(frozen=True)
class ValidationReport:
    symbol: str
    timeframe: str
    total_bars: int
    passed: bool
    fatal_count: int
    warning_count: int
    has_timestamp_issues: bool
    has_ohlc_issues: bool
    has_gap_issues: bool
    ohlc_bad_count: int
    findings: tuple

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "total_bars": self.total_bars,
            "passed": self.passed,
            "fatal_count": self.fatal_count,
            "warning_count": self.warning_count,
            "has_timestamp_issues": self.has_timestamp_issues,
            "has_ohlc_issues": self.has_ohlc_issues,
            "has_gap_issues": self.has_gap_issues,
            "ohlc_bad_count": self.ohlc_bad_count,
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"[{self.symbol}] {status} — "
            f"{self.fatal_count} fatal, {self.warning_count} warning"
        )


class DataValidator:
    MIN_BARS = 50

    def validate(self, df: pd.DataFrame, symbol: str = "") -> bool:
        if df is None or df.empty:
            logger.warning(f"[{symbol}] DataFrame is None or empty")
            return False

        if len(df) < self.MIN_BARS:
            logger.warning(
                f"[{symbol}] Insufficient bars: {len(df)} < {self.MIN_BARS}"
            )
            return False

        required_cols = {"open", "high", "low", "close", "volume"}

        if not required_cols.issubset(df.columns):
            logger.error(
                f"[{symbol}] Missing columns: "
                f"{required_cols - set(df.columns)}"
            )
            return False

        invalid_candles = df[df["high"] < df["low"]]

        if not invalid_candles.empty:
            logger.warning(
                f"[{symbol}] {len(invalid_candles)} candles with high < low"
            )

        nan_pct = df.isnull().mean().max()

        if nan_pct > 0.05:
            logger.warning(f"[{symbol}] High NaN ratio: {nan_pct:.1%}")
            return False

        return True

    def validate_detailed(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        timeframe: str = "",
    ) -> ValidationReport:

        findings: List[ValidationFinding] = []

        total_bars = 0 if df is None else len(df)

        self._latest_ohlc_bad_count = 0

        # 1. Empty / None
        if df is None or df.empty:
            findings.append(
                ValidationFinding(
                    level="fatal",
                    code="EMPTY_DATAFRAME",
                    message="DataFrame is None or empty",
                    count=0,
                )
            )

            return self._build_report(
                symbol,
                timeframe,
                total_bars,
                findings,
            )

        # 2. Insufficient bars
        if len(df) < self.MIN_BARS:
            findings.append(
                ValidationFinding(
                    level="fatal",
                    code="INSUFFICIENT_BARS",
                    message=(
                        f"Insufficient bars: "
                        f"{len(df)} < {self.MIN_BARS}"
                    ),
                    count=len(df),
                )
            )

            return self._build_report(
                symbol,
                timeframe,
                total_bars,
                findings,
            )

        # 3. Missing columns
        missing = _REQUIRED_COLUMNS - set(df.columns)

        if missing:
            findings.append(
                ValidationFinding(
                    level="fatal",
                    code="MISSING_COLUMNS",
                    message=f"Missing columns: {sorted(missing)}",
                    count=len(missing),
                )
            )

            return self._build_report(
                symbol,
                timeframe,
                total_bars,
                findings,
            )

        # 4. Datetime index check
        is_datetime = isinstance(df.index, pd.DatetimeIndex)

        if not is_datetime:
            findings.append(
                ValidationFinding(
                    level="fatal",
                    code="NOT_DATETIME_INDEX",
                    message="Index is not a DatetimeIndex",
                    count=0,
                )
            )

        # 5. Monotonic timestamps
        if is_datetime and not df.index.is_monotonic_increasing:

            inversion_count = int(
                (df.index[1:] <= df.index[:-1]).sum()
            )

            findings.append(
                ValidationFinding(
                    level="fatal",
                    code="NOT_MONOTONIC",
                    message=(
                        "Timestamps are not "
                        "monotonically increasing"
                    ),
                    count=inversion_count,
                )
            )

        # 6. Duplicate timestamps
        if is_datetime:

            dup_count = len(df) - len(df.index.unique())

            if dup_count:
                findings.append(
                    ValidationFinding(
                        level="warning",
                        code="DUPLICATE_TIMESTAMPS",
                        message=(
                            f"Found {dup_count} "
                            f"duplicate timestamps"
                        ),
                        count=dup_count,
                    )
                )

        # 7. Gap detection
        if is_datetime:
            findings.extend(
                self._check_gaps(
                    df,
                    symbol,
                    timeframe,
                )
            )

        # 8. OHLC integrity checks

        mask_hl = df["high"] < df["low"]

        if mask_hl.any():

            count = int(mask_hl.sum())

            findings.append(
                ValidationFinding(
                    level="warning",
                    code="HIGH_LT_LOW",
                    message=f"{count} candles with high < low",
                    count=count,
                )
            )

        mask_open = (
            (df["open"] < df["low"]) |
            (df["open"] > df["high"])
        )

        if mask_open.any():

            count = int(mask_open.sum())

            findings.append(
                ValidationFinding(
                    level="warning",
                    code="OPEN_OUTSIDE_HL",
                    message=(
                        f"{count} candles with "
                        f"open outside [low, high]"
                    ),
                    count=count,
                )
            )

        mask_close = (
            (df["close"] < df["low"]) |
            (df["close"] > df["high"])
        )

        if mask_close.any():

            count = int(mask_close.sum())

            findings.append(
                ValidationFinding(
                    level="warning",
                    code="CLOSE_OUTSIDE_HL",
                    message=(
                        f"{count} candles with "
                        f"close outside [low, high]"
                    ),
                    count=count,
                )
            )

        bad_ohlc_mask = (
            mask_hl |
            mask_open |
            mask_close
        )

        self._latest_ohlc_bad_count = int(
            bad_ohlc_mask.sum()
        )

        mask_vol = df["volume"] <= 0

        if mask_vol.any():

            count = int(mask_vol.sum())

            findings.append(
                ValidationFinding(
                    level="warning",
                    code="NEGATIVE_VOLUME",
                    message=(
                        f"{count} candles with "
                        f"negative or zero volume"
                    ),
                    count=count,
                )
            )

        # 9. NaN checks

        total_nan = int(df.isnull().sum().sum())

        if total_nan > 0:

            max_nan_pct = df.isnull().mean().max()

            if max_nan_pct > _NAN_FATAL_THRESHOLD:

                findings.append(
                    ValidationFinding(
                        level="fatal",
                        code="EXCESSIVE_NAN",
                        message=(
                            f"Excessive NaN: "
                            f"{max_nan_pct:.1%}"
                        ),
                        count=total_nan,
                    )
                )

            else:

                findings.append(
                    ValidationFinding(
                        level="warning",
                        code="NAN_PRESENT",
                        message=(
                            f"NaN present: "
                            f"{max_nan_pct:.1%}"
                        ),
                        count=total_nan,
                    )
                )

        return self._build_report(
            symbol,
            timeframe,
            total_bars,
            findings,
        )

    @staticmethod
    def _check_gaps(
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> List[ValidationFinding]:

        if (
            not isinstance(df.index, pd.DatetimeIndex)
            or len(df) < 2
        ):
            return []

        try:

            if timeframe:
                expected = pd.to_timedelta(timeframe)
            else:
                expected = df.index.to_series().diff().median()

        except Exception:
            return []

        diffs = df.index[1:] - df.index[:-1]

        gap_mask = diffs > expected

        if not gap_mask.any():
            return []

        gap_count = int(gap_mask.sum())

        return [
            ValidationFinding(
                level="warning",
                code="GAPS_DETECTED",
                message=(
                    f"Found {gap_count} "
                    f"gap(s) in timestamp sequence"
                ),
                count=gap_count,
            )
        ]

    def _build_report(
        self,
        symbol: str,
        timeframe: str,
        total_bars: int,
        findings: List[ValidationFinding],
    ) -> ValidationReport:

        findings_sorted = tuple(
            sorted(
                findings,
                key=lambda f: (
                    0 if f.is_fatal() else 1,
                    f.code,
                ),
            )
        )

        fatal_count = sum(
            1 for f in findings_sorted if f.is_fatal()
        )

        warning_count = sum(
            1 for f in findings_sorted if not f.is_fatal()
        )

        timestamp_codes = {
            "DUPLICATE_TIMESTAMPS",
            "NOT_DATETIME_INDEX",
            "NOT_MONOTONIC",
            "GAPS_DETECTED",
        }

        has_timestamp_issues = any(
            f.code in timestamp_codes
            for f in findings_sorted
        )

        ohlc_codes = {
            "HIGH_LT_LOW",
            "OPEN_OUTSIDE_HL",
            "CLOSE_OUTSIDE_HL",
        }

        has_ohlc_issues = any(
            f.code in ohlc_codes
            for f in findings_sorted
        )

        ohlc_bad_count = getattr(
            self,
            "_latest_ohlc_bad_count",
            0,
        )

        has_gap_issues = any(
            f.code == "GAPS_DETECTED"
            for f in findings_sorted
        )

        passed = fatal_count == 0

        return ValidationReport(
            symbol=symbol,
            timeframe=timeframe,
            total_bars=total_bars,
            passed=passed,
            fatal_count=fatal_count,
            warning_count=warning_count,
            has_timestamp_issues=has_timestamp_issues,
            has_ohlc_issues=has_ohlc_issues,
            has_gap_issues=has_gap_issues,
            ohlc_bad_count=ohlc_bad_count,
            findings=findings_sorted,
        )