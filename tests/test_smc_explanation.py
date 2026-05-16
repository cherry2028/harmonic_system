from market_state.smc.engine import (
    MultiTimeframeContext,
)
from market_state.smc.explanation import (
    explain_signal,
)
from market_state.smc.scoring import (
    SetupScore,
    score_setup,
)
from market_state.smc.signal import (
    create_signal_event,
)
from tests.test_smc_engine import (
    make_context,
)


def test_explains_signal():
    context = MultiTimeframeContext(
        htf=make_context(
            bearish_bias=True,
            bearish_setup=False,
            bearish_shift=False,
        ),
        itf=make_context(
            bearish_bias=True,
            bearish_setup=True,
            bearish_shift=False,
        ),
        ltf=make_context(
            bearish_bias=True,
            bearish_setup=False,
            bearish_shift=True,
        ),
    )

    score = score_setup(context)

    signal = create_signal_event(
        context=context.ltf,
        score=score,
    )

    explanation = explain_signal(
        signal=signal,
        context=context,
    )

    assert "bearish" in explanation