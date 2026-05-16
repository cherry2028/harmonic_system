from market_state.smc.event_memory import (
    build_event_memory,
)


def test_detects_recent_events():
    memory = build_event_memory(
        swept_high_history=[
            False,
            False,
            True,
            False,
        ],
        bearish_shift_history=[
            False,
            False,
            False,
            True,
        ],
        memory_window=5,
    )

    assert memory.recent_sweep_high is True

    assert memory.recent_bearish_shift is True