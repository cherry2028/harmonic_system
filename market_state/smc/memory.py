from dataclasses import dataclass


@dataclass(frozen=True)
class SetupMemory:
    recent_range_detected: bool
    recent_liquidity_sweep: bool