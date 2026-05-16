from market_state.smc.memory import (
    SetupMemory,
)


def test_creates_setup_memory():
    memory = SetupMemory(
        recent_range_detected=True,
        recent_liquidity_sweep=False,
    )

    assert memory.recent_range_detected is True