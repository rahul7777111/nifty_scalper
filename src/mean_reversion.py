"""Standalone mean-reversion sleeve.

This module keeps the signal generation self-contained so it can be used by the
main router, backtests, or future portfolio allocators without depending on the
full trading loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MeanReversionSignal:
    signal: str
    zscore: float
    entry_score: float
    expected_reversion: float
    half_life_bars: int
    recommended_option: str
    reason: str


def _safe_mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    try:
        return float(pstdev(values))
    except Exception:
        return 0.0


def estimate_half_life(prices: Iterable[float], *, max_bars: int = 60) -> int:
    samples = [float(v) for v in prices if v is not None]
    if len(samples) < 4:
        return 0
    reversion_strength = 0.0
    diffs = []
    levels = []
    for prev, curr in zip(samples[:-1], samples[1:]):
        diffs.append(float(curr - prev))
        levels.append(float(prev))
    level_mean = _safe_mean(levels)
    centered = [lvl - level_mean for lvl in levels]
    denom = sum(v * v for v in centered)
    if denom <= 1e-9:
        return 0
    numer = -sum(centered[i] * diffs[i] for i in range(len(diffs)))
    reversion_strength = max(0.0, float(numer / denom))
    if reversion_strength <= 1e-9:
        return 0
    half_life = int(round(0.693 / reversion_strength))
    return max(1, min(int(max_bars), half_life))


def evaluate_mean_reversion(
    closes: Iterable[float],
    *,
    lookback: int = 20,
    entry_zscore: float = 1.25,
    exit_zscore: float = 0.35,
) -> MeanReversionSignal:
    series = [float(v) for v in closes if v is not None]
    if len(series) < max(lookback, 5):
        return MeanReversionSignal(
            signal="flat",
            zscore=0.0,
            entry_score=0.0,
            expected_reversion=0.0,
            half_life_bars=0,
            recommended_option="NONE",
            reason="insufficient_history",
        )

    window = series[-int(lookback) :]
    latest = float(window[-1])
    mu = _safe_mean(window)
    sigma = _safe_stdev(window)
    if sigma <= 1e-9:
        return MeanReversionSignal(
            signal="flat",
            zscore=0.0,
            entry_score=0.0,
            expected_reversion=0.0,
            half_life_bars=0,
            recommended_option="NONE",
            reason="zero_variance_window",
        )

    zscore = float((latest - mu) / sigma)
    distance = abs(zscore)
    half_life = estimate_half_life(window)
    expected_reversion = float(mu - latest)
    if distance >= float(entry_zscore):
        if zscore < 0:
            return MeanReversionSignal(
                signal="buy_call",
                zscore=zscore,
                entry_score=min(1.0, distance / max(float(entry_zscore), 1e-9)),
                expected_reversion=expected_reversion,
                half_life_bars=half_life,
                recommended_option="CE",
                reason="price_below_mean",
            )
        return MeanReversionSignal(
            signal="buy_put",
            zscore=zscore,
            entry_score=min(1.0, distance / max(float(entry_zscore), 1e-9)),
            expected_reversion=expected_reversion,
            half_life_bars=half_life,
            recommended_option="PE",
            reason="price_above_mean",
        )
    if distance <= float(exit_zscore):
        return MeanReversionSignal(
            signal="exit",
            zscore=zscore,
            entry_score=max(0.0, 1.0 - (distance / max(float(exit_zscore), 1e-9))),
            expected_reversion=expected_reversion,
            half_life_bars=half_life,
            recommended_option="NONE",
            reason="mean_reversion_complete",
        )
    return MeanReversionSignal(
        signal="flat",
        zscore=zscore,
        entry_score=max(0.0, min(1.0, distance / max(float(entry_zscore), 1e-9))),
        expected_reversion=expected_reversion,
        half_life_bars=half_life,
        recommended_option="NONE",
        reason="inside_neutral_band",
    )
