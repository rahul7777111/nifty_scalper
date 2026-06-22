"""Standalone trend-following sleeve primitives for feature generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TrendFollowingSignal:
    signal: str
    direction: float
    strength: float
    confidence: float
    ema_gap_pct: float
    momentum_pct: float
    breakout_score: float
    pullback_score: float
    reason: str


def _ema(values: Sequence[float], period: int) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (max(int(period), 1) + 1.0)
    value = float(values[0])
    for raw in values[1:]:
        price = float(raw)
        value = (alpha * price) + ((1.0 - alpha) * value)
    return float(value)


def evaluate_trend_following(
    closes: Iterable[float],
    *,
    fast_period: int = 8,
    slow_period: int = 21,
    momentum_lookback: int = 10,
    breakout_lookback: int = 20,
    strength_threshold: float = 0.003,
) -> TrendFollowingSignal:
    series = [float(v) for v in closes if v is not None]
    min_required = max(int(slow_period), int(momentum_lookback) + 1, int(breakout_lookback))
    if len(series) < max(min_required, 5):
        return TrendFollowingSignal("flat", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "insufficient_history")

    latest = float(series[-1])
    fast_ema = _ema(series[-int(fast_period) :], int(fast_period))
    slow_ema = _ema(series[-int(slow_period) :], int(slow_period))
    denom = abs(latest) if abs(latest) > 1e-9 else 1.0
    ema_gap_pct = float((fast_ema - slow_ema) / denom)

    anchor = float(series[-int(momentum_lookback) - 1])
    momentum_denom = abs(anchor) if abs(anchor) > 1e-9 else 1.0
    momentum_pct = float((latest - anchor) / momentum_denom)

    prior_window = series[-int(breakout_lookback) : -1]
    prior_high = max(prior_window) if prior_window else latest
    prior_low = min(prior_window) if prior_window else latest
    breakout_up = max(0.0, (latest - prior_high) / (abs(prior_high) if abs(prior_high) > 1e-9 else 1.0))
    breakout_down = max(0.0, (prior_low - latest) / (abs(prior_low) if abs(prior_low) > 1e-9 else 1.0))
    breakout_score = breakout_up if breakout_up > 0.0 else (-breakout_down if breakout_down > 0.0 else 0.0)

    pullback_score = max(0.0, 1.0 - min(1.0, abs(latest - fast_ema) / (0.01 * (abs(fast_ema) if abs(fast_ema) > 1e-9 else 1.0))))
    raw_strength = (0.45 * ema_gap_pct) + (0.35 * momentum_pct) + (0.20 * breakout_score)
    scale = max(float(strength_threshold), 1e-9)
    strength = max(-1.0, min(1.0, raw_strength / scale))
    direction = 1.0 if strength > 0.0 else (-1.0 if strength < 0.0 else 0.0)
    confidence = max(0.0, min(1.0, abs(strength) * (0.75 + (0.25 * pullback_score))))

    aligned_up = ema_gap_pct > 0.0 and momentum_pct > 0.0
    aligned_down = ema_gap_pct < 0.0 and momentum_pct < 0.0
    if strength >= 1.0 and aligned_up:
        return TrendFollowingSignal("buy_call", direction, strength, confidence, ema_gap_pct, momentum_pct, max(0.0, breakout_score), pullback_score, "uptrend_confirmation")
    if strength <= -1.0 and aligned_down:
        return TrendFollowingSignal("buy_put", direction, strength, confidence, ema_gap_pct, momentum_pct, min(0.0, breakout_score), pullback_score, "downtrend_confirmation")
    return TrendFollowingSignal("flat", direction, strength, confidence, ema_gap_pct, momentum_pct, breakout_score, pullback_score, "trend_not_confirmed")
