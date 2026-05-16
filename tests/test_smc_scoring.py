from market_state.smc.engine import (
    MultiTimeframeContext,
)
from market_state.smc.scoring import (
    score_setup,
)
from tests.test_smc_engine import (
    make_context,
)


def test_scores_bearish_setup():
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

    result = score_setup(context)

    assert result.score == 100
    assert result.valid is True