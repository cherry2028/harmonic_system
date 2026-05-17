from dataclasses import dataclass


@dataclass
class AMDState:
    accumulation: bool
    manipulation: bool
    distribution: bool


def detect_accumulation(candles) -> bool:
    if len(candles) < 20:
        return False

    recent = candles[-20:]

    highs = [
        candle.high
        for candle in recent
    ]

    lows = [
        candle.low
        for candle in recent
    ]

    closes = [
        candle.close
        for candle in recent
    ]

    range_high = max(highs)

    range_low = min(lows)

    range_size = (
        range_high - range_low
    )

    avg_close = (
        sum(closes)
        / len(closes)
    )

    compression = (
        range_size / avg_close
    )

    close_spread = (
        max(closes) - min(closes)
    ) / avg_close

    drift = abs(
        closes[-1] - closes[0]
    ) / avg_close

    return (
        compression < 0.008
        and close_spread < 0.004
        and drift < 0.003
    )


def detect_manipulation(candles) -> bool:
    return False


def detect_distribution(candles) -> bool:
    return False


def detect_amd(candles) -> AMDState:
    accumulation = detect_accumulation(
        candles
    )

    manipulation = detect_manipulation(
        candles
    )

    distribution = detect_distribution(
        candles
    )

    return AMDState(
        accumulation=accumulation,
        manipulation=manipulation,
        distribution=distribution,
    )