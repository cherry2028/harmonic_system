from market_state.smc.range_state import Candle


def find_active_candle(
    candles: list[Candle],
    timestamp: int,
) -> int:
    active_index = 0

    for i, candle in enumerate(candles):
        if candle.timestamp <= timestamp:
            active_index = i
        else:
            break

    return active_index + 1