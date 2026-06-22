"""Simple statistical arbitrage primitives for pair-spread trading."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Iterable, Sequence


@dataclass(frozen=True)
class PairSpreadSignal:
    signal: str
    hedge_ratio: float
    spread: float
    spread_mean: float
    spread_std: float
    zscore: float
    confidence: float
    reason: str


def estimate_hedge_ratio(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 1.0
    x_mean = float(mean(x_values))
    y_mean = float(mean(y_values))
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    var = sum((x - x_mean) ** 2 for x in x_values)
    if abs(var) <= 1e-9:
        return 1.0
    return float(cov / var)


def evaluate_pair_spread(
    x_prices: Iterable[float],
    y_prices: Iterable[float],
    *,
    lookback: int = 30,
    entry_zscore: float = 1.5,
    exit_zscore: float = 0.5,
) -> PairSpreadSignal:
    x = [float(v) for v in x_prices if v is not None]
    y = [float(v) for v in y_prices if v is not None]
    n = min(len(x), len(y))
    if n < max(int(lookback), 5):
        return PairSpreadSignal("flat", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, "insufficient_history")

    x = x[-int(lookback) :]
    y = y[-int(lookback) :]
    hedge_ratio = estimate_hedge_ratio(x, y)
    spreads = [float(yv - hedge_ratio * xv) for xv, yv in zip(x, y)]
    spread_mean = float(mean(spreads))
    spread_std = float(pstdev(spreads)) if len(spreads) > 1 else 0.0
    spread = float(spreads[-1])
    if spread_std <= 1e-9:
        return PairSpreadSignal("flat", hedge_ratio, spread, spread_mean, spread_std, 0.0, 0.0, "zero_spread_variance")

    zscore = float((spread - spread_mean) / spread_std)
    confidence = max(0.0, min(1.0, abs(zscore) / max(float(entry_zscore), 1e-9)))
    if zscore >= float(entry_zscore):
        return PairSpreadSignal("short_spread", hedge_ratio, spread, spread_mean, spread_std, zscore, confidence, "spread_too_wide")
    if zscore <= -float(entry_zscore):
        return PairSpreadSignal("long_spread", hedge_ratio, spread, spread_mean, spread_std, zscore, confidence, "spread_too_narrow")
    if abs(zscore) <= float(exit_zscore):
        return PairSpreadSignal("exit", hedge_ratio, spread, spread_mean, spread_std, zscore, 1.0 - min(1.0, abs(zscore)), "spread_reverted")
    return PairSpreadSignal("flat", hedge_ratio, spread, spread_mean, spread_std, zscore, confidence, "inside_band")
