from market_state.smc.premium_discount import (
    detect_premium_discount,
)
from market_state.smc.range_state import (
    RangeState,
)


def test_detects_premium_zone():
    range_state = RangeState(
        is_ranging=True,
        range_high=110,
        range_low=90,
        previous_range_high=110,
        previous_range_low=90,
        equilibrium=100,
        range_width_pct=0.2,
    )

    result = detect_premium_discount(
        close_price=105,
        range_state=range_state,
    )

    assert result.in_premium is True

    assert result.in_discount is False