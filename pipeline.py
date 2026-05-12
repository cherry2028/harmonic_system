"""pipeline.py — ScanPipeline integration layer (Action 10)

Gate 0: ScanResult dataclass only.
Implementation will proceed incrementally through Gates 1–8.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from market_state.vector import MarketStateVector
from signals.gate import GateResult
from signals.signal import TieredSignal


@dataclass(frozen=True)
class ScanResult:
    """
    Immutable record of one complete scan cycle.
    Carries enough information for telemetry, testing,
    and upstream orchestration without re-running the scan.
    """
    symbol: str
    timeframe: str
    outcome: str  # from outcome enum
    duration_ms: float

    # Stage 2 output
    vector: Optional[MarketStateVector] = None

    # Stage 3 output
    gate_result: Optional[GateResult] = None

    # Stage 4-5 counts
    swings: int = 0
    patterns: int = 0

    # Stage 6-7 outputs
    scored_count: int = 0
    tiered_signals: List[TieredSignal] = field(default_factory=list)

    # Stage 3 block detail
    block_code: Optional[str] = None

    # Stage 1 / global error detail
    error: Optional[str] = None

    @property
    def published_count(self) -> int:
        return len(self.tiered_signals)

    @property
    def is_success(self) -> bool:
        return self.outcome == "signal_published"



# ─────────────────────────────────────────────────────────────────────────────
# Gate 1: ScanPipeline skeleton — dependency injection architecture
# ─────────────────────────────────────────────────────────────────────────────

from typing import Any

from data.fetcher import DataFetcher
from market_state.engine import MarketStateEngine
from patterns.patterns.swing_detector import AdaptiveSwingDetector
from patterns.patterns.harmonic_detector import HarmonicDetector
from scoring.pattern_scorer import PatternScorer
from signals.gate import HostileMarketGate
from signals.tier import SignalTier
from signals.daily_counter import DailyCounter
from delivery.presentation import SignalPresentation
from delivery.telegram_formatter import TelegramFormatter


class ScanPipeline:
    """
    Orchestrates all validated components in deterministic execution order.
    Single entrypoint: scan_one().
    All dependencies are injected at construction — zero hidden state.
    """

    def __init__(
        self,
        *,
        data_fetcher: DataFetcher,
        market_state_engine: MarketStateEngine,
        swing_detector: AdaptiveSwingDetector,
        harmonic_detector: HarmonicDetector,
        pattern_scorer: PatternScorer,
        hostile_gate: HostileMarketGate,
        signal_tier: SignalTier,
        daily_counter: DailyCounter,
        telemetry,
        signal_presentation: SignalPresentation,
        telegram_formatter: TelegramFormatter,
    ) -> None:
        """
        Initialize pipeline with all required component references.
        All arguments are keyword-only to enforce explicit wiring.
        """
        self._data_fetcher = data_fetcher
        self._market_state_engine = market_state_engine
        self._swing_detector = swing_detector
        self._harmonic_detector = harmonic_detector
        self._pattern_scorer = pattern_scorer
        self._hostile_gate = hostile_gate
        self._signal_tier = signal_tier
        self._daily_counter = daily_counter
        self._telemetry = telemetry
        self._signal_presentation = signal_presentation
        self._telegram_formatter = telegram_formatter

    # ── Public API ──────────────────────────────────────────────────────────

    def scan_one(
        self,
        symbol: str,
        timeframe: str,
        *,
        dry_run: bool = False,
    ) -> ScanResult:
        """
        Execute one complete scan cycle for a single symbol/timeframe pair.

        Parameters
        ----------
        symbol : str
            Trading pair symbol (e.g. "BTCUSDT").
        timeframe : str
            Candle timeframe (e.g. "1h", "4h").
        dry_run : bool, default False
            If True, skip HTTP delivery and counter increments.

        Returns
        -------
        ScanResult
            Immutable record of the scan outcome.
        """
        # Gate 1: signature only — execution logic in subsequent gates
        raise NotImplementedError(
            "scan_one() execution logic not yet implemented — "
            "proceed to Gate 2 for Stage 1–2 wiring"
        )

    # ── Introspection helpers (testable contract verification) ──────────────

    @property
    def components(self) -> dict[str, Any]:
        """Return mapping of component name → instance for contract verification."""
        return {
            "data_fetcher": self._data_fetcher,
            "market_state_engine": self._market_state_engine,
            "swing_detector": self._swing_detector,
            "harmonic_detector": self._harmonic_detector,
            "pattern_scorer": self._pattern_scorer,
            "hostile_gate": self._hostile_gate,
            "signal_tier": self._signal_tier,
            "daily_counter": self._daily_counter,
            "telemetry": self._telemetry,
            "signal_presentation": self._signal_presentation,
            "telegram_formatter": self._telegram_formatter,
        }