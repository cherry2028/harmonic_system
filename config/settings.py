# config/settings.py — full replacement

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DataConfig:
    provider:           str       = "binance"
    timeframes:         List[str] = field(default_factory=lambda: ["15m", "1h", "4h"])
    symbols:            List[str] = field(default_factory=lambda: [
                                        "BTCUSDT", "ETHUSDT", "SOLUSDT"
                                    ])
    lookback_candles:   int       = 500
    cache_ttl_seconds:  int       = 60


@dataclass
class PatternConfig:
    # Swing detection
    swing_strength:         int   = 3       # Bars each side for pivot confirmation
    zigzag_threshold_pct:   float = 0.015   # 1.5% min move to qualify as a swing
    min_pattern_bars:       int   = 20      # Pattern must span at least 20 bars

    # Fibonacci tolerance
    fib_tolerance:          float = 0.05    # ±5% on all ratio checks

    # Signal quality gate
    min_rr_ratio:           float = 1.5
    confirmation_candle:    bool  = True    # Always wait for candle close


@dataclass
class TelegramConfig:
    bot_token:              str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    channel_id:             str = os.getenv("TELEGRAM_CHANNEL_ID", "")
    alert_cooldown_seconds: int = 300


@dataclass
class SystemConfig:
    data:     DataConfig     = field(default_factory=DataConfig)
    pattern:  PatternConfig  = field(default_factory=PatternConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    log_level: str           = "INFO"
    dry_run:   bool          = True


CONFIG = SystemConfig()