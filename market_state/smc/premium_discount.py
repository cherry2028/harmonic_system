from dataclasses import dataclass

from market_state.smc.range_state import (
    RangeState,
)


@dataclass(frozen=True)
class PremiumDiscountState:
    in_premium: bool

    in_discount: bool


def detect_premium_discount(
    close_price: float,
    range_state: RangeState,
) -> PremiumDiscountState:
    equilibrium = (
        range_state.equilibrium
    )

    in_premium = (
        close_price > equilibrium
    )

    in_discount = (
        close_price < equilibrium
    )

    return PremiumDiscountState(
        in_premium=in_premium,
        in_discount=in_discount,
    )