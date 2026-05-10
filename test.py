from data.fetcher import DataFetcher
from patterns.patterns.institutional_swing_detector import AdaptiveSwingDetector
from patterns.patterns.harmonic_detector import HarmonicDetector

# Initialize
fetcher = DataFetcher()
swing_detector = AdaptiveSwingDetector()
harmonic_detector = HarmonicDetector()


# Settings
SYMBOL = "BTCUSDT"
TIMEFRAME = "1h"


# Fetch market data
df = fetcher.fetch(
    symbol=SYMBOL,
    timeframe=TIMEFRAME,
    bars=300
)

if df is None:
    raise Exception("Failed to fetch data")


# Detect swings
result = swing_detector.detect(df)
swings = result

print(f"\nDetected Swings: {len(swings)}")


# Detect harmonic patterns
matches = harmonic_detector.detect(
    swings=swings,
    symbol=SYMBOL,
    timeframe=TIMEFRAME
)

print(f"\nDetected Harmonic Patterns: {len(matches.matches)}\n")

print(matches)


# Print matches
for match in matches.matches:
    print(match)