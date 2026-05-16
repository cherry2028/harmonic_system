from dataclasses import dataclass

from market_state.smc.timeframe import (
    TimeframeContext,
)


@dataclass(frozen=True)
class MultiTimeframeContext:
    htf: TimeframeContext
    itf: TimeframeContext
    ltf: TimeframeContext

    @property
    def bearish_alignment(self) -> bool:
        return (
            self.htf.bias.bearish
            and self.itf.setup.bearish_setup
            and self.ltf.structure.bearish_shift
        )

    @property
    def bullish_alignment(self) -> bool:
        return (
            self.htf.bias.bullish
            and self.itf.setup.bullish_setup
            and self.ltf.structure.bullish_shift
        )