from market_state.smc.scoring import (
    SetupScore,
)
from market_state.smc.signal import (
    create_signal_event,
)
from tests.test_smc_engine import (
    make_context,
)
from market_state.smc.regime import (
    RegimeState,
)


def test_creates_signal_event():
    context = make_context(
        bearish_bias=True,
        bearish_setup=true,
bearish_pressure=75,
        bearish_shift=True,
    )

    score = SetupScore(
        score=90,
        valid=True,
    )

    signal = create_signal_event(
        context=context,
        score=score,
    )

    assert signal.direction == "bearish"
    assert signal.valid is True