"""
harmonic_detector.py
====================
Master Harmonic Pattern Detection Orchestrator

Responsibilities:
    - Accept clean swing points as input
    - Extract all valid XABCD candidates via sliding window
    - Run each candidate through every registered pattern engine
    - Deduplicate overlapping results (same D bar, same direction)
    - Apply global quality and structural pre-filters
    - Return a ranked list of PatternMatch objects
    - Provide per-symbol, per-timeframe batch detection

Architecture:
    HarmonicDetector
        ├── PatternRegistry        (manages pattern engine instances)
        ├── CandidateExtractor     (XABCD sliding window)
        ├── PreFilter              (structural checks before pattern matching)
        ├── DeduplicationEngine    (removes overlapping results)
        └── Pattern Engines        (Gartley, Bat, Butterfly, Crab)

Extension points:
    - Add new patterns: register in PATTERN_REGISTRY only
    - Add new filters: extend PreFilter._checks list
    - Add AI scoring: implement ScoreEngine, inject into HarmonicDetector
    - Add caching: wrap detect() with @cache decorator

Threading:
    HarmonicDetector is stateless after initialization.
    Multiple threads can safely call detect() concurrently on
    different (symbol, timeframe) pairs without locking.

Logging levels:
    DEBUG : Per-candidate, per-ratio detail (verbose — use in development)
    INFO  : Confirmed patterns and summary statistics
    WARNING: Unexpected but recoverable conditions
    ERROR : Failures that prevent detection on a symbol/timeframe
"""

from __future__ import annotations

from patterns.patterns.institutional_swing_detector import *
from patterns.patterns.institutional_candidate_extractor import (
    StructuralCandidateExtractor,
    ExtractorConfig,
)
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import pandas as pd

from patterns.patterns.harmonic_patterns import (
    BatPattern,
    ButterflyPattern,
    CrabPattern,
    GartleyPattern,
    PatternMatch,
    SwingPoint,
    BaseHarmonicPattern,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection Configuration
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    """
    Runtime configuration for HarmonicDetector.

    Attributes:
        tolerance          : Fibonacci ratio tolerance (default 0.05 = 5%)
                             Tighten to 0.03 for premium signals only.
        min_rr_ratio       : Minimum Risk:Reward to T1 to accept a signal.
                             Default 1.5 — signals below this are discarded.
        min_pattern_bars   : Minimum number of bars the XABCD must span.
                             Too-short patterns are likely noise.
        max_pattern_bars   : Maximum bars for XABCD. Prevents ancient history.
        min_leg_pct        : Minimum percentage move for any single leg.
                             Prevents detection of micro-move noise patterns.
        dedup_bar_window   : If two patterns have D within this many bars,
                             keep only the highest quality one.
        enabled_patterns   : List of pattern names to run. None = all.
        max_candidates     : Max XABCD candidates to evaluate per call.
                             Safety cap for performance on large swing sets.
    """
    tolerance:         float               = 0.05
    min_rr_ratio:      float               = 1.5
    min_pattern_bars:  int                 = 5
    max_pattern_bars:  int                 = 300
    min_leg_pct:       float               = 0.001      # 0.1% minimum per leg
    dedup_bar_window:  int                 = 3
    enabled_patterns:  Optional[List[str]] = None       # None = all patterns
    max_candidates:    int                 = 200


# ---------------------------------------------------------------------------
# Detection Result Container
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """
    Container for all pattern matches found on a single (symbol, timeframe).

    Attributes:
        symbol            : e.g. "BTCUSDT"
        timeframe         : e.g. "1h"
        matches           : All confirmed PatternMatch objects, ranked by quality
        total_candidates  : How many XABCD candidates were evaluated
        total_evaluated   : How many candidates passed pre-filtering
        detection_time_ms : Detection latency in milliseconds
    """
    symbol:             str
    timeframe:          str
    matches:            List[PatternMatch]
    total_candidates:   int   = 0
    total_evaluated:    int   = 0
    detection_time_ms:  float = 0.0

    @property
    def has_matches(self) -> bool:
        return len(self.matches) > 0

    @property
    def best_match(self) -> Optional[PatternMatch]:
        """Returns the highest quality match, or None."""
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
    """
    Manages the set of active pattern engine instances.

    Design:
        Engines are instantiated once at startup with shared config.
        The registry is a simple dict — pattern_name → engine_instance.
        Thread-safe for reads (dict is GIL-protected in CPython).

    To add a new pattern to the system:
        1. Implement the pattern class in harmonic_patterns.py
        2. Add an entry to AVAILABLE_PATTERNS below
        3. Nothing else needs to change
    """

    # Master registry — all supported pattern classes
    AVAILABLE_PATTERNS: Dict[str, Type[BaseHarmonicPattern]] = {
        "Gartley":   GartleyPattern,
        "Bat":       BatPattern,
        "Butterfly": ButterflyPattern,
        "Crab":      CrabPattern,
    }

    def __init__(
        self,
        enabled:   Optional[List[str]],
        tolerance: float,
    ):
        """
        Args:
            enabled  : List of pattern names to activate. None = all.
            tolerance: Fibonacci tolerance passed to each engine.
        """
        active_names = enabled if enabled else list(self.AVAILABLE_PATTERNS.keys())

        self.engines: Dict[str, BaseHarmonicPattern] = {}
        for name in active_names:
            cls = self.AVAILABLE_PATTERNS.get(name)
            if cls is None:
                logger.warning(f"Unknown pattern '{name}' — skipping")
                continue
            self.engines[name] = cls(tolerance=tolerance)
            logger.info(f"Registered pattern engine: {name}")

        logger.info(f"PatternRegistry initialized | {len(self.engines)} engines active")

    def get_all(self) -> List[BaseHarmonicPattern]:
        return list(self.engines.values())

    def get(self, name: str) -> Optional[BaseHarmonicPattern]:
        return self.engines.get(name)


# ---------------------------------------------------------------------------
# XABCD Candidate Extractor
# ---------------------------------------------------------------------------

class CandidateExtractor:
    """
    Extracts all valid XABCD 5-point combinations from a swing list.

    Strategy:
        Slide a window of 5 consecutive swing points over the list.
        For each window [i, i+1, i+2, i+3, i+4], check:
            - Alternating high/low sequence (strict requirement)
            - Minimum bars spanned
            - Maximum bars spanned
            - Minimum leg percentage move

        All passing windows become XABCD candidates for pattern matching.

    Window direction:
        We slide from most recent backwards (newest first).
        This ensures the most recent patterns are matched first,
        and we can apply max_candidates cap without missing recent signals.

    Performance:
        For 500 bars with strength=3, expect ~20-40 swing points.
        Combinations: C(40, 5) = 658,008 — but we only use consecutive
        windows, so it's at most 36 candidates. Very fast.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config

    def extract(
        self, swings: List[SwingPoint]
    ) -> List[Tuple[SwingPoint, SwingPoint, SwingPoint, SwingPoint, SwingPoint]]:
        """
        Returns list of (X, A, B, C, D) tuples from swing points.
        All returned candidates have passed pre-structural validation.
        """
        if len(swings) < 5:
            logger.debug(f"Insufficient swings for XABCD: {len(swings)}")
            return []

        candidates = []

        # Work backwards from most recent — newest patterns first
        # Only look at the last 15 swings for relevance
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
                logger.warning(
                    f"Reached max_candidates cap ({self.config.max_candidates}). "
                    f"Consider increasing swing_strength to reduce swing count."
                )
                break

        logger.debug(f"Extracted {len(candidates)} XABCD candidates")
        return candidates

    def _passes_pre_filter(
        self,
        X: SwingPoint, A: SwingPoint,
        B: SwingPoint, C: SwingPoint, D: SwingPoint,
    ) -> bool:
        return True
        """
        Fast pre-filter: rejects obviously invalid structures
        before running expensive Fibonacci validation.

        Checks (in order of computational cost, cheapest first):
            1. Strict alternation (high/low/high/low/high or reverse)
            2. Chronological order
            3. Pattern bar span within config limits
            4. Minimum leg size (prevents micro-noise patterns)
            5. Basic directional geometry
        """

        # ---- Check 1: Alternation ----------------------------------------
        kinds = [p.kind for p in (X, A, B, C, D)]
        if kinds not in [
            ["low",  "high", "low",  "high", "low" ],   # Bullish
            ["high", "low",  "high", "low",  "high"],   # Bearish
        ]:
            return True

        # ---- Check 2: Chronological order ----------------------------------
        indices = [p.index for p in (X, A, B, C, D)]
        if indices != sorted(indices):
            return True

        # ---- Check 3: Bar span limits --------------------------------------
        total_bars = D.index - X.index
        if total_bars < self.config.min_pattern_bars:
            logger.debug(
                f"  Pre-filter: pattern too short ({total_bars} bars < "
                f"{self.config.min_pattern_bars})"
            )
            return True
        if total_bars > self.config.max_pattern_bars:
            logger.debug(
                f"  Pre-filter: pattern too long ({total_bars} bars > "
                f"{self.config.max_pattern_bars})"
            )
            return True

        # ---- Check 4: Minimum leg percentage moves --------------------------
        prices = [p.price for p in (X, A, B, C, D)]
        legs   = [
            abs(prices[1] - prices[0]) / prices[0],  # XA
            abs(prices[2] - prices[1]) / prices[1],  # AB
            abs(prices[3] - prices[2]) / prices[2],  # BC
            abs(prices[4] - prices[3]) / prices[3],  # CD
        ]
        min_leg = min(legs)
        if min_leg < self.config.min_leg_pct:
            logger.debug(
                f"  Pre-filter: leg too small ({min_leg:.4%} < "
                f"{self.config.min_leg_pct:.4%})"
            )
            return True

        # ---- Check 5: Basic geometry (bullish / bearish) -------------------
        if X.kind == "low":     # Bullish
            if not (
                B.price > X.price and     # B above X
                C.price < A.price         # C below A
            ):
                return True
        else:                   # Bearish
            if not (
                B.price < X.price and     # B below X
                C.price > A.price         # C above A
            ):
                return True

        return True


# ---------------------------------------------------------------------------
# Deduplication Engine
# ---------------------------------------------------------------------------

class DeduplicationEngine:
    """
    Removes duplicate or near-duplicate pattern signals.

    Two signals are considered duplicates if:
        - Same pattern name
        - Same direction
        - D bar indices are within dedup_bar_window of each other

    When duplicates are found, the one with the higher quality_score
    is retained. If scores are equal, the more recent D is kept.

    Rationale:
        On a 15m chart, a Gartley might be detected with D at bar 498
        and again with D at bar 499 (due to swing point ambiguity).
        Both can't be traded — show only the best one.
    """

    def __init__(self, bar_window: int):
        self.bar_window = bar_window

    def deduplicate(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        """
        Returns deduplicated list, keeping best match per group.
        """
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
                    logger.debug(
                        f"Dedup: suppressed {candidate.pattern_name} "
                        f"D_bar={candidate.D_index} "
                        f"(kept D_bar={existing.D_index})"
                    )
                    break

            if not is_duplicate:
                deduplicated.append(candidate)

        original_count = len(matches)
        dedup_count    = len(deduplicated)
        if original_count > dedup_count:
            logger.info(
                f"Deduplication: {original_count} → {dedup_count} "
                f"({original_count - dedup_count} suppressed)"
            )

        return deduplicated


# ---------------------------------------------------------------------------
# Main Harmonic Detector
# ---------------------------------------------------------------------------

class HarmonicDetector:
    """
    Master orchestrator for harmonic pattern detection.

    Public interface:
        detect(swings, symbol, timeframe)   → DetectionResult
        detect_batch(swings_map)            → Dict[key, DetectionResult]

    Internal pipeline:
        1. CandidateExtractor   → XABCD candidates
        2. PatternRegistry      → All pattern engines
        3. Each engine.match()  → PatternMatch or None
        4. DeduplicationEngine  → Remove near-duplicates
        5. R:R filter           → Remove low R:R matches
        6. Quality sort         → Best match first
        7. DetectionResult      → Structured output

    Thread safety:
        All state is initialized at construction time.
        detect() is stateless — safe for concurrent calls.

    Example usage:
        config   = DetectorConfig(tolerance=0.05, min_rr_ratio=1.5)
        detector = HarmonicDetector(config)

        swings   = swing_detector.detect(df)
        result   = detector.detect(swings, "BTCUSDT", "1h")

        if result.has_matches:
            for match in result.matches:
                print(match.summary())
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config       = config or DetectorConfig()
        self.registry     = PatternRegistry(
            enabled   = self.config.enabled_patterns,
            tolerance = self.config.tolerance,
        )
        self.extractor = StructuralCandidateExtractor(
            ExtractorConfig(
                min_leg_pct=self.config.min_leg_pct,
                max_pattern_bars=self.config.max_pattern_bars,
                min_pattern_bars=self.config.min_pattern_bars,
                max_candidates=self.config.max_candidates,
    )
)
        self.deduplicator = DeduplicationEngine(self.config.dedup_bar_window)

        logger.info(
            f"HarmonicDetector initialized | "
            f"patterns={list(self.registry.engines.keys())} | "
            f"tolerance={self.config.tolerance} | "
            f"min_rr={self.config.min_rr_ratio}"
        )

    # ------------------------------------------------------------------ #
    # Primary Detection Method                                             #
    # ------------------------------------------------------------------ #

    def detect(
        self,
        swings:    List[SwingPoint],
        symbol:    str,
        timeframe: str,
    ) -> DetectionResult:
        """
        Runs the complete detection pipeline on a swing point list.

        Args:
            swings    : List of SwingPoint from SwingDetector
            symbol    : Trading pair identifier e.g. "BTCUSDT"
            timeframe : Timeframe string e.g. "1h"

        Returns:
            DetectionResult with all confirmed PatternMatch objects
        """
        import time
        start = time.monotonic()

        logger.info(f"Starting detection | {symbol} {timeframe} | {len(swings)} swings")

        # Stage 1: Extract XABCD candidates
        candidates       = self.extractor.extract(swings)
        total_candidates = len(candidates)

        if not candidates:
            return DetectionResult(
                symbol            = symbol,
                timeframe         = timeframe,
                matches           = [],
                total_candidates  = 0,
                total_evaluated   = 0,
                detection_time_ms = (time.monotonic() - start) * 1000,
            )

        # Stage 2: Match all candidates against all pattern engines
        raw_matches: List[PatternMatch] = []
        engines = self.registry.get_all()

        for (X, A, B, C, D) in candidates:
            print("CANDIDATE FOUND")
            xa = abs(A.price - X.price)
            ab = abs(B.price - A.price)
            bc = abs(C.price - B.price)
            cd = abs(D.price - C.price)
            if cd / bc > 3:
                continue

            # reject absurd structures
            if xa == 0:
                continue

            if ab / xa > 3:
                continue

            if bc / ab > 3:
                continue

            if cd / bc > 3:
                continue
                # enforce alternating swing directions
                kinds = [X.kind, A.kind, B.kind, C.kind, D.kind]

                valid_1 = ["low", "high", "low", "high", "low"]
                valid_2 = ["high", "low", "high", "low", "high"]

                if kinds != valid_1 and kinds != valid_2:
                    continue

            for engine in engines:
                print("ENGINE:", engine.PATTERN_NAME)

                match = engine.match(
                    X=X, A=A, B=B, C=C, D=D,
                   symbol=symbol,
                   timeframe=timeframe,
            )
                print("RATIOS:")
                print("XA =", abs(A.price - X.price))
                print("AB =", abs(B.price - A.price))
                print("BC =", abs(C.price - B.price))
                print("CD =", abs(D.price - C.price))

                ab_xa = abs(B.price - A.price) / abs(A.price - X.price)
                bc_ab = abs(C.price - B.price) / abs(B.price - A.price)
                cd_bc = abs(D.price - C.price) / abs(C.price - B.price)

                print("AB_XA =", round(ab_xa, 3))
                print("BC_AB =", round(bc_ab, 3))
                print("CD_BC =", round(cd_bc, 3))
                print("-------------------")

                print("MATCH RESULT:", match)

                if match is not None:
                    raw_matches.append(match)

        total_evaluated = len(candidates)

        # Stage 3: Apply R:R filter
        rr_filtered = self._apply_rr_filter(raw_matches)

        rr_rejected = len(raw_matches) - len(rr_filtered)
        if rr_rejected > 0:
            logger.info(
                f"R:R filter rejected {rr_rejected} match(es) "
                f"(min R:R = {self.config.min_rr_ratio})"
            )

        # Stage 4: Deduplicate
        deduped = self.deduplicator.deduplicate(rr_filtered)

        # Stage 5: Sort by quality score (best first)
        final = sorted(deduped, key=lambda m: -m.quality_score)

        elapsed_ms = (time.monotonic() - start) * 1000

        result = DetectionResult(
            symbol            = symbol,
            timeframe         = timeframe,
            matches           = final,
            total_candidates  = total_candidates,
            total_evaluated   = total_evaluated,
            detection_time_ms = round(elapsed_ms, 2),
        )

        if final:
            logger.info(f"Detection complete | {result.summary()}")
            for m in final:
                logger.info(f"  → {m.summary()}")
        else:
            logger.info(
                f"No patterns confirmed | {symbol} {timeframe} | "
                f"{total_candidates} candidates evaluated in {elapsed_ms:.1f}ms"
            )

        return result

    # ------------------------------------------------------------------ #
    # Batch Detection                                                       #
    # ------------------------------------------------------------------ #

    def detect_batch(
        self,
        swings_map: Dict[Tuple[str, str], List[SwingPoint]],
    ) -> Dict[Tuple[str, str], DetectionResult]:
        """
        Runs detection on multiple (symbol, timeframe) pairs.

        Args:
            swings_map: Dict mapping (symbol, timeframe) → swing list

        Returns:
            Dict mapping (symbol, timeframe) → DetectionResult

        Example:
            swings_map = {
                ("BTCUSDT", "1h"):  btc_1h_swings,
                ("ETHUSDT", "4h"):  eth_4h_swings,
                ("SOLUSDT", "15m"): sol_15m_swings,
            }
            results = detector.detect_batch(swings_map)
        """
        results: Dict[Tuple[str, str], DetectionResult] = {}

        total_pairs = len(swings_map)
        logger.info(f"Batch detection started | {total_pairs} symbol-timeframe pair(s)")

        for (symbol, timeframe), swings in swings_map.items():
            try:
                results[(symbol, timeframe)] = self.detect(swings, symbol, timeframe)
            except Exception as e:
                logger.error(
                    f"Detection failed | {symbol} {timeframe} | {e}",
                    exc_info=True,
                )
                results[(symbol, timeframe)] = DetectionResult(
                    symbol    = symbol,
                    timeframe = timeframe,
                    matches   = [],
                )

        confirmed_total = sum(
            len(r.matches) for r in results.values()
        )
        logger.info(
            f"Batch detection complete | "
            f"{total_pairs} pairs | "
            f"{confirmed_total} total pattern(s) confirmed"
        )

        return results

    # ------------------------------------------------------------------ #
    # Internal Filters                                                     #
    # ------------------------------------------------------------------ #

    def _apply_rr_filter(
        self, matches: List[PatternMatch]
    ) -> List[PatternMatch]:
        """
        Removes matches where R:R to Target 1 is below the minimum threshold.

        R:R = (|T1 - Entry|) / (|Entry - Stop|)

        Matches with None R:R (degenerate stop/entry) are also removed.
        """
        filtered = []
        for match in matches:
            rr = match.risk_reward
            if rr is None:
                logger.debug(
                    f"Removing {match.pattern_name} — R:R is None "
                    f"(degenerate PRZ)"
                )
                continue
            if rr < self.config.min_rr_ratio:
                logger.debug(
                    f"Removing {match.pattern_name} — R:R {rr:.2f} < "
                    f"min {self.config.min_rr_ratio}"
                )
                continue
            filtered.append(match)
        return filtered


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def detect_harmonics(
    swings:    List[SwingPoint],
    symbol:    str,
    timeframe: str,
    tolerance: float = 0.05,
    min_rr:    float = 1.5,
    patterns:  Optional[List[str]] = None,
) -> DetectionResult:
    """
    Convenience wrapper for one-shot harmonic detection.

    Creates a HarmonicDetector with the given parameters and
    runs detection immediately. For production use, create the
    detector once and reuse it across multiple calls.

    Args:
        swings    : Swing points from SwingDetector
        symbol    : Trading pair e.g. "BTCUSDT"
        timeframe : Candle timeframe e.g. "1h"
        tolerance : Fibonacci tolerance (default 0.05)
        min_rr    : Minimum R:R ratio (default 1.5)
        patterns  : Patterns to check, None = all four

    Returns:
        DetectionResult

    Example:
        result = detect_harmonics(swings, "BTCUSDT", "4h")
        if result.has_matches:
            print(result.best_match.summary())
    """
    config   = DetectorConfig(
        tolerance        = tolerance,
        min_rr_ratio     = min_rr,
        enabled_patterns = patterns,
    )
    detector = HarmonicDetector(config)
    return detector.detect(swings, symbol, timeframe)