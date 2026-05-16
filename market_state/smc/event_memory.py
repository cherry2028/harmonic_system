from dataclasses import dataclass


@dataclass(frozen=True)
class EventMemory:
    recent_sweep_high: bool

    recent_bearish_shift: bool


def build_event_memory(
    swept_high_history: list[bool],
    bearish_shift_history: list[bool],
    memory_window: int = 5,
) -> EventMemory:
    recent_sweep_high = any(
        swept_high_history[-memory_window:]
    )

    recent_bearish_shift = any(
        bearish_shift_history[-memory_window:]
    )

    return EventMemory(
        recent_sweep_high=recent_sweep_high,
        recent_bearish_shift=recent_bearish_shift,
    )