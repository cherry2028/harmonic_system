"""replay/statistics.py — pure deterministic statistics over replay outcomes.

Operates ONLY on completed outcomes (SignalOutcome with a resolution).
Ignores timeouts and unresolved records explicitly.
No mutation of input records. No filesystem access. No external deps.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

from replay.outcome_tracker import SignalOutcome
from replay.replay_runner import ReplayRecord


@dataclass(frozen=True)
class OutcomeRecord:
    """Immutable pairing of a replay record with its evaluated outcome.

    This is a thin additive wrapper — it does NOT mutate ReplayRecord.
    """
    replay: ReplayRecord
    outcome: SignalOutcome


@dataclass(frozen=True)
class StatisticsReport:
    """Immutable statistical summary of a collection of replay outcomes."""
    total_signals: int
    wins: int
    losses: int
    win_rate: float
    target1_hit_rate: float
    target2_hit_rate: float
    target3_hit_rate: float
    stop_hit_rate: float
    avg_bars_to_resolution: float
    expectancy: float
    by_tier: Dict[str, "StatisticsReport"] = field(default_factory=dict)
    by_pattern: Dict[str, "StatisticsReport"] = field(default_factory=dict)
    by_market_state: Dict[str, "StatisticsReport"] = field(default_factory=dict)

    @property
    def loss_rate(self) -> float:
        return 1.0 - self.win_rate if self.total_signals > 0 else 0.0


class ReplayStatistics:
    """Pure deterministic statistics computation.

    Input: List[OutcomeRecord] — each pairs a ReplayRecord with its SignalOutcome.
    Output: StatisticsReport — immutable, segmented, statistically correct.
    """

    # ── Public API ──────────────────────────────────────────────────────────

    def compute(self, records: List[OutcomeRecord]) -> StatisticsReport:
        """Compute statistics over completed outcomes only.

        Records with outcome_type == "timeout" are explicitly excluded.
        Records with missing tiered_signal or vector are excluded.
        """
        completed = self._filter_completed(records)
        if not completed:
            return self._empty_report()

        return self._build_report(completed)

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _filter_completed(records: List[OutcomeRecord]) -> List[OutcomeRecord]:
        """Keep only records with resolved outcomes and required metadata."""
        filtered = []
        for rec in records:
            if rec.outcome.outcome_type == "timeout":
                continue
            if rec.replay.tiered_signal is None:
                continue
            if rec.replay.scan_result.vector is None:
                continue
            filtered.append(rec)
        return filtered

    def _build_report(self, records: List[OutcomeRecord]) -> StatisticsReport:
        total = len(records)
        wins = sum(1 for r in records if r.outcome.outcome_type in {"target1", "target2", "target3"})
        losses = total - wins
        win_rate = wins / total if total > 0 else 0.0

        t1_count = sum(1 for r in records if r.outcome.outcome_type == "target1")
        t2_count = sum(1 for r in records if r.outcome.outcome_type == "target2")
        t3_count = sum(1 for r in records if r.outcome.outcome_type == "target3")
        stop_count = sum(1 for r in records if r.outcome.outcome_type == "stop")

        avg_bars = self._mean(
            [r.outcome.bars_to_resolution for r in records if r.outcome.bars_to_resolution > 0]
        )

        expectancy = self._compute_expectancy(records, win_rate, losses, total)

        return StatisticsReport(
            total_signals=total,
            wins=wins,
            losses=losses,
            win_rate=round(win_rate, 6),
            target1_hit_rate=round(t1_count / total, 6) if total > 0 else 0.0,
            target2_hit_rate=round(t2_count / total, 6) if total > 0 else 0.0,
            target3_hit_rate=round(t3_count / total, 6) if total > 0 else 0.0,
            stop_hit_rate=round(stop_count / total, 6) if total > 0 else 0.0,
            avg_bars_to_resolution=round(avg_bars, 2),
            expectancy=round(expectancy, 6),
            by_tier=self._segment(records, lambda r: r.replay.tiered_signal.tier),
            by_pattern=self._segment(records, lambda r: r.replay.tiered_signal.pattern_name),
            by_market_state=self._segment(
                records, lambda r: r.replay.scan_result.vector.dominant_state
            ),
        )

    @staticmethod
    def _compute_expectancy(
        records: List[OutcomeRecord],
        win_rate: float,
        losses: int,
        total: int,
    ) -> float:
        """Expectancy = (Pw × Aw) - (Pl × Al).

        Win amount = signal risk_reward (R-multiple).
        Loss amount = 1.0 R (standardized stop loss).
        """
        if total == 0:
            return 0.0

        win_records = [r for r in records if r.outcome.outcome_type in {"target1", "target2", "target3"}]
        avg_win_r = ReplayStatistics._mean(
            [r.replay.tiered_signal.risk_reward for r in win_records if r.replay.tiered_signal.risk_reward is not None]
        )
        if avg_win_r == 0.0 and win_records:
            # Fallback: if risk_reward is None, use 1.0 as neutral assumption
            avg_win_r = 1.0

        loss_rate = losses / total
        avg_loss_r = 1.0  # standardized 1R stop

        return (win_rate * avg_win_r) - (loss_rate * avg_loss_r)

    @staticmethod
    def _segment(
        records: List[OutcomeRecord],
        key_fn,
    ) -> Dict[str, StatisticsReport]:
        """Group records by key and compute sub-reports (flat — no nested segmentation)."""
        groups: Dict[str, List[OutcomeRecord]] = {}
        for rec in records:
            key = key_fn(rec)
            if not key:
                key = "unknown"
            groups.setdefault(key, []).append(rec)

        reports = {}
        for key, group in sorted(groups.items()):
            # Flat sub-report: same metrics, no further segmentation
            total = len(group)
            wins = sum(1 for r in group if r.outcome.outcome_type in {"target1", "target2", "target3"})
            losses = total - wins
            win_rate = wins / total if total > 0 else 0.0

            t1 = sum(1 for r in group if r.outcome.outcome_type == "target1")
            t2 = sum(1 for r in group if r.outcome.outcome_type == "target2")
            t3 = sum(1 for r in group if r.outcome.outcome_type == "target3")
            stop = sum(1 for r in group if r.outcome.outcome_type == "stop")

            avg_bars = ReplayStatistics._mean(
                [r.outcome.bars_to_resolution for r in group if r.outcome.bars_to_resolution > 0]
            )

            exp = ReplayStatistics._compute_expectancy(group, win_rate, losses, total)

            reports[key] = StatisticsReport(
                total_signals=total,
                wins=wins,
                losses=losses,
                win_rate=round(win_rate, 6),
                target1_hit_rate=round(t1 / total, 6) if total > 0 else 0.0,
                target2_hit_rate=round(t2 / total, 6) if total > 0 else 0.0,
                target3_hit_rate=round(t3 / total, 6) if total > 0 else 0.0,
                stop_hit_rate=round(stop / total, 6) if total > 0 else 0.0,
                avg_bars_to_resolution=round(avg_bars, 2),
                expectancy=round(exp, 6),
                # Sub-reports are flat — no further nesting
                by_tier={},
                by_pattern={},
                by_market_state={},
            )
        return reports

    @staticmethod
    def _mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _empty_report() -> StatisticsReport:
        return StatisticsReport(
            total_signals=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            target1_hit_rate=0.0,
            target2_hit_rate=0.0,
            target3_hit_rate=0.0,
            stop_hit_rate=0.0,
            avg_bars_to_resolution=0.0,
            expectancy=0.0,
            by_tier={},
            by_pattern={},
            by_market_state={},
        )