from market_state.smc.range_state import (
    RangeState,
)
from market_state.smc.regime import (
    detect_regime,
)


def test_detects_trending_regime():
    range_state = RangeState(
        is_ranging=False,
        range_high=110,
        range_low=90,
        previous_range_high=110,
        previous_range_low=90,
        equilibrium=100,
        range_width_pct=0.02,
    )

    regime = detect_regime(
        range_state
    )

    assert regime.trending is True

    assert regime.choppy is False