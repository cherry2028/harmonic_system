from market_state.smc.signal import (
    SignalEvent,
)
from market_state.smc.engine import (
    MultiTimeframeContext,
)


def explain_signal(
    signal: SignalEvent,
    context: MultiTimeframeContext,
) -> str:
    return (
        f"Signal: {signal.direction} | "
        f"Score: {signal.score} | "
        f"HTF Bias Bearish: {context.htf.bias.bearish} | "
        f"ITF Bearish Setup: {context.itf.setup.bearish_setup} | "
        f"LTF Bearish Shift: {context.ltf.structure.bearish_shift}"
    )