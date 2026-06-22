from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from indicators import adx, atr
from market_data import Candle


@dataclass
class RegimeSnapshot:
    regime: str
    atr_pct: float
    adx_value: float
    realized_vol: float
    trend_strength: float


class MarketRegimeDetector:
    def __init__(
        self,
        *,
        high_vol_threshold: float = 0.75,
        low_vol_threshold: float = 0.25,
        trend_threshold: float = 25.0,
    ) -> None:
        self.high_vol_threshold = float(high_vol_threshold)
        self.low_vol_threshold = float(low_vol_threshold)
        self.trend_threshold = float(trend_threshold)

    @staticmethod
    def _realized_vol(returns: Sequence[float]) -> float:
        if len(returns) < 2:
            return 0.0
        return float(np.std(np.asarray(returns, dtype=float)))

    def classify_snapshot(
        self,
        *,
        atr_pct: float,
        adx_value: float,
        realized_vol_percentile: float,
        trend_strength: float,
    ) -> str:
        if realized_vol_percentile >= self.high_vol_threshold or atr_pct >= 0.02:
            return "high_volatility"
        if adx_value >= self.trend_threshold and abs(trend_strength) > 0.0:
            return "trending"
        if realized_vol_percentile <= self.low_vol_threshold:
            return "low_volatility"
        return "ranging"

    def classify_candles(self, candles: Sequence[Candle], lookback: int = 30) -> RegimeSnapshot:
        c = list(candles or [])
        if len(c) < max(lookback, 15):
            return RegimeSnapshot("ranging", 0.0, 0.0, 0.0, 0.0)
        closes = np.asarray([float(x.close) for x in c], dtype=float)
        highs = [float(x.high) for x in c]
        lows = [float(x.low) for x in c]
        returns = np.diff(closes[-lookback:]) / np.where(closes[-lookback:-1] == 0.0, 1.0, closes[-lookback:-1])
        realized_vol = self._realized_vol(returns.tolist())
        atr_value = float(atr(highs, lows, closes.tolist(), period=14) or 0.0)
        atr_pct = atr_value / float(closes[-1]) if closes[-1] else 0.0
        adx_value = float(adx(highs, lows, closes.tolist(), period=14) or 0.0)
        trend_strength = float((closes[-1] - closes[-lookback]) / closes[-lookback]) if closes[-lookback] else 0.0

        rolling_vols = []
        for idx in range(lookback, len(closes)):
            prev = closes[idx - lookback : idx]
            rets = np.diff(prev) / np.where(prev[:-1] == 0.0, 1.0, prev[:-1])
            rolling_vols.append(self._realized_vol(rets.tolist()))
        if rolling_vols:
            realized_vol_percentile = float(np.mean(np.asarray(rolling_vols) <= realized_vol))
        else:
            realized_vol_percentile = 0.5

        regime = self.classify_snapshot(
            atr_pct=atr_pct,
            adx_value=adx_value,
            realized_vol_percentile=realized_vol_percentile,
            trend_strength=trend_strength,
        )
        return RegimeSnapshot(regime, atr_pct, adx_value, realized_vol, trend_strength)

    def performance_by_regime(self, rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        buckets: Dict[str, List[float]] = {}
        for row in rows:
            regime = str(row.get("regime") or "unknown")
            buckets.setdefault(regime, []).append(float(row.get("pnl") or 0.0))
        report: Dict[str, Dict[str, float]] = {}
        for regime, pnls in buckets.items():
            report[regime] = {
                "count": float(len(pnls)),
                "expectancy": float(sum(pnls) / len(pnls)) if pnls else 0.0,
                "win_rate": float(sum(1 for pnl in pnls if pnl > 0.0) / len(pnls)) if pnls else 0.0,
            }
        return report
