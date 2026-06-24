from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Candle:
    """Single OHLCV candle used for indicators and patterns."""

    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
