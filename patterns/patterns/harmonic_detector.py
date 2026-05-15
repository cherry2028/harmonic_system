"""
harmonic_detector.py
====================
Master Harmonic Pattern Detection Orchestrator
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

from institutional_swing_detector import *

from patterns.patterns.institutional_candidate_extractor import (
    StructuralCandidateExtractor,
    ExtractorConfig,
)

from patterns.patterns.harmonic_patterns import (
    BatPattern,
    ButterflyPattern,
    CrabPattern,
    GartleyPattern,
    PatternMatch,
    SwingPoint,
    BaseHarmonicPattern,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detector Config
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    tolerance: float = 0.05
    min_rr_ratio: float = 1.5
    min_pattern_bars: int = 5
    max_pattern_bars: int = 300
    min_leg_pct: float = 0.001
    dedup_bar_window: int = 3
    enabled_patterns: Optional[List[str]] = None
    max_candidates: int = 200


# ---------------------------------------------------------------------------
# Detection Result
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    symbol: str
    timeframe: str
    matches: List[PatternMatch]
    total_candidates: int = 0
    total_evaluated: int = 0
    detection_time_ms: float = 0.0

    @property
    def has_matches(self) -> bool:
        return len(self.matches) > 0

    @property
    def best_match(self) -> Optional[PatternMatch]:
        if not self.matches:
            return None
        return max(self.matches, key=lambda m: m.quality_score)

    def summary(self) -> str:
        return (
            f"[{self.symbol} {self.timeframe}] "
            f"{len(self.matches)} pattern(s) | "
            f"candidates={self.total_candidates} | "
            f"evaluated={self.total_evaluated} | "
            f"{self.detection_time_ms:.1f}ms"
        )

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Pattern Registry
# ---------------------------------------------------------------------------

class PatternRegistry:

    AVAILABLE_PATTERNS: Dict[str, Type[BaseHarmonicPattern]] = {
        "Gartley": GartleyPattern,
        "Bat": BatPattern,
        "Butterfly": ButterflyPattern,
        "Crab": CrabPattern,
    }

    def __init__(
        self,
        enabled: Optional[List[str]],
        tolerance: float,
    ):
        active_names = enabled if enabled else list(self.AVAILABLE_PATTERNS.keys())

        self.engines: Dict[str, BaseHarmonicPattern] = {}

        for name in active_names:
            cls = self.AVAILABLE_PATTERNS.get(name)

            if cls is None:
                logger.warning(f"Unknown pattern '{name}' — skipping")
                continue

            self.engines[name] = cls(tolerance=tolerance)
            logger.info(f"Registered pattern engine: {name}")

    def get_all(self) -> List[BaseHarmonicPattern]:
        return list(self.engines.values())


# ---------------------------------------------------------------------------
# Candidate Extractor
# ---------------------------------------------------------------------------

class CandidateExtractor:

    def __init__(self, config: DetectorConfig):
        self.config = config

    def extract(
        self,
        swings: List[SwingPoint],
    ) -> List[Tuple[SwingPoint, SwingPoint, SwingPoint, SwingPoint, SwingPoint]]:

        if len(swings) < 5:
            return []

        candidates = []

        recent_swings = swings[-15:] if len(swings) > 15 else swings

        for i in range(len(recent_swings) - 4):
            X = recent_swings[i]
            A = recent_swings[i + 1]
            B = recent_swings[i + 2]
            C = recent_swings[i + 3]
            D = recent_swings[i + 4]

            if not self._passes_pre_filter(X, A, B, C, D):
                continue

            candidates.append((X, A, B, C, D))

            if len(candidates) >= self.config.max_candidates:
                break

        logger.debug(f"Extracted {len(candidates)} XABCD candidates")

        return candidates

    def _passes_pre_filter(
        self,
        X: SwingPoint,
        A: SwingPoint,
        B: SwingPoint,
        C: SwingPoint,
        D: SwingPoint,
    ) -> bool:
        """
        True  -> valid candidate
        False -> reject candidate
        """

        # ---------------------------------------------------------
        # Check 1: Alternating structure
        # ---------------------------------------------------------

        kinds = [p.kind for p in (X, A, B, C, D)]

        bullish = ["low", "high", "low", "high", "low"]
        bearish = ["high", "low", "high", "low", "high"]

        if kinds != bullish and kinds != bearish:
            return False

        # ---------------------------------------------------------
        # Check 2: Chronological order
        # ---------------------------------------------------------

        indices = [p.index for p in (X, A, B, C, D)]

        if indices != sorted(indices):
            return False

        # ---------------------------------------------------------
        # Check 3: Pattern size
        # ---------------------------------------------------------

        total_bars = D.index - X.index

        if total_bars < self.config.min_pattern_bars:
            return False

        if total_bars > self.config.max_pattern_bars:
            return False

        # ---------------------------------------------------------
        # Check 4: Minimum leg movement
        # ---------------------------------------------------------

        prices = [p.price for p in (X, A, B, C, D)]

        legs = [
            abs(prices[1] - prices[0]) / prices[0],
            abs(prices[2] - prices[1]) / prices[1],
            abs(prices[3] - prices[2]) / prices[2],
            abs(prices[4] - prices[3]) / prices[3],
        ]

        if min(legs) < self.config.min_leg_pct:
            return False

        # ---------------------------------------------------------
        # Check 5: Basic geometry
        # ---------------------------------------------------------

        if X.kind == "low":

            if not (
                B.price > X.price and
                C.price < A.price
            ):
                return False

        else:

            if not (
                B.price < X.price and
                C.price > A.price
            ):
                return False

        return True


# ---------------------------------------------------------------------------
# Deduplication Engine
# ---------------------------------------------------------------------------

class DeduplicationEngine:

    def __init__(self, bar_window: int):
        self.bar_window = bar_window

    def deduplicate(self, matches: List[PatternMatch]) -> List[PatternMatch]:

        if len(matches) <= 1:
            return matches

        deduplicated: List[PatternMatch] = []

        for candidate in sorted(matches, key=lambda m: -m.quality_score):
            is_duplicate = False

            for existing in deduplicated:

                if (
                    candidate.pattern_name == existing.pattern_name
                    and candidate.direction == existing.direction
                    and abs(candidate.D_index - existing.D_index) <= self.bar_window
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(candidate)

        return deduplicated


# ---------------------------------------------------------------------------
# Main Harmonic Detector
# ---------------------------------------------------------------------------

class HarmonicDetector:

    def __init__(self, config: Optional[DetectorConfig] = None):

        self.config = config or DetectorConfig()

        self.registry = PatternRegistry(
            enabled=self.config.enabled_patterns,
            tolerance=self.config.tolerance,
        )

        self.extractor = StructuralCandidateExtractor(
            ExtractorConfig(
                min_leg_pct=self.config.min_leg_pct,
                max_pattern_bars=self.config.max_pattern_bars,
                min_pattern_bars=self.config.min_pattern_bars,
                max_candidates=self.config.max_candidates,
            )
        )

        self.deduplicator = DeduplicationEngine(
            self.config.dedup_bar_window
        )

    def detect(
        self,
        swings: List[SwingPoint],
        symbol: str,
        timeframe: str,
    ) -> DetectionResult:

        start = time.monotonic()

        logger.info(
            f"Starting detection | {symbol} {timeframe} | {len(swings)} swings"
        )

        candidates = self.extractor.extract(swings)
        total_candidates = len(candidates)

        if not candidates:
            return DetectionResult(
                symbol=symbol,
                timeframe=timeframe,
                matches=[],
                total_candidates=0,
                total_evaluated=0,
                detection_time_ms=(time.monotonic() - start) * 1000,
            )

        raw_matches: List[PatternMatch] = []
        engines = self.registry.get_all()

        for (X, A, B, C, D) in candidates:

            xa = abs(A.price - X.price)
            ab = abs(B.price - A.price)
            bc = abs(C.price - B.price)
            cd = abs(D.price - C.price)

            if xa == 0:
                continue

            if ab / xa > 3:
                continue

            if ab == 0:
                continue

            if bc / ab > 3:
                continue

            if bc == 0:
                continue

            if cd / bc > 3:
                continue

            logger.debug("CANDIDATE FOUND")

            for engine in engines:

                logger.debug(f"ENGINE: {engine.PATTERN_NAME}")

                match = engine.match(
                    X=X,
                    A=A,
                    B=B,
                    C=C,
                    D=D,
                    symbol=symbol,
                    timeframe=timeframe,
                )

                if match is not None:
                    raw_matches.append(match)

        total_evaluated = len(candidates)

        rr_filtered = self._apply_rr_filter(raw_matches)

        deduped = self.deduplicator.deduplicate(rr_filtered)

        final = sorted(deduped, key=lambda m: -m.quality_score)

        elapsed_ms = (time.monotonic() - start) * 1000

        result = DetectionResult(
            symbol=symbol,
            timeframe=timeframe,
            matches=final,
            total_candidates=total_candidates,
            total_evaluated=total_evaluated,
            detection_time_ms=round(elapsed_ms, 2),
        )

        return result

    def detect_batch(
        self,
        swings_map: Dict[Tuple[str, str], List[SwingPoint]],
    ) -> Dict[Tuple[str, str], DetectionResult]:

        results: Dict[Tuple[str, str], DetectionResult] = {}

        for (symbol, timeframe), swings in swings_map.items():

            try:
                results[(symbol, timeframe)] = self.detect(
                    swings,
                    symbol,
                    timeframe,
                )

            except Exception as e:
                logger.error(
                    f"Detection failed | {symbol} {timeframe} | {e}",
                    exc_info=True,
                )

                results[(symbol, timeframe)] = DetectionResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    matches=[],
                )

        return results

    def _apply_rr_filter(
        self,
        matches: List[PatternMatch],
    ) -> List[PatternMatch]:

        filtered = []

        for match in matches:

            rr = match.risk_reward

            if rr is None:
                continue

            if rr < self.config.min_rr_ratio:
                continue

            filtered.append(match)

        return filtered


# ---------------------------------------------------------------------------
# Convenience Wrapper
# ---------------------------------------------------------------------------

def detect_harmonics(
    swings: List[SwingPoint],
    symbol: str,
    timeframe: str,
    tolerance: float = 0.05,
    min_rr: float = 1.5,
    patterns: Optional[List[str]] = None,
) -> DetectionResult:

    config = DetectorConfig(
        tolerance=tolerance,
        min_rr_ratio=min_rr,
        enabled_patterns=patterns,
    )

    detector = HarmonicDetector(config)

    return detector.detect(swings, symbol, timeframe)