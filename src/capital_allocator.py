"""Cross-sleeve capital allocation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass(frozen=True)
class SleeveAllocation:
    sleeve: str
    raw_score: float
    weight: float
    capital: float


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_sleeve_scores(
    *,
    regime: str,
    trend_score: float,
    mean_reversion_score: float,
    stat_arb_score: float,
) -> Dict[str, float]:
    key = str(regime or "").strip().lower()
    base = {
        "trend": max(0.0, float(trend_score)),
        "mean_reversion": max(0.0, float(mean_reversion_score)),
        "stat_arb": max(0.0, float(stat_arb_score)),
    }
    if key == "trending":
        base["trend"] *= 1.35
        base["mean_reversion"] *= 0.80
    elif key == "mean_reverting":
        base["mean_reversion"] *= 1.35
        base["trend"] *= 0.75
    elif key == "volatile":
        base["stat_arb"] *= 1.15
        base["trend"] *= 0.95
    elif key == "quiet":
        base["stat_arb"] *= 1.20
        base["mean_reversion"] *= 1.10
    return base


def allocate_capital_across_sleeves(
    *,
    account_capital: float,
    sleeve_scores: Mapping[str, float],
    max_single_sleeve_weight: float = 0.50,
    reserve_cash_weight: float = 0.10,
    drawdown_throttle: float = 1.0,
) -> Dict[str, SleeveAllocation]:
    capital = max(0.0, float(account_capital))
    reserve_weight = _clamp01(float(reserve_cash_weight))
    active_weight_budget = max(0.0, 1.0 - reserve_weight) * _clamp01(float(drawdown_throttle))
    max_weight = max(0.0, min(1.0, float(max_single_sleeve_weight)))

    positive = {str(k): max(0.0, float(v)) for k, v in dict(sleeve_scores or {}).items()}
    total = sum(positive.values())
    if total <= 1e-9 or active_weight_budget <= 0.0:
        return {
            name: SleeveAllocation(sleeve=name, raw_score=score, weight=0.0, capital=0.0)
            for name, score in positive.items()
        }

    provisional = {name: (score / total) * active_weight_budget for name, score in positive.items()}
    capped = {name: min(max_weight, weight) for name, weight in provisional.items()}
    capped_total = sum(capped.values())

    # Redistribute leftover budget across uncapped sleeves.
    leftover = max(0.0, active_weight_budget - capped_total)
    if leftover > 1e-9:
        uncapped = {name: positive[name] for name, weight in provisional.items() if weight < max_weight}
        uncapped_total = sum(uncapped.values())
        if uncapped_total > 1e-9:
            for name, score in uncapped.items():
                room = max(0.0, max_weight - capped[name])
                add = min(room, leftover * (score / uncapped_total))
                capped[name] += add

    return {
        name: SleeveAllocation(
            sleeve=name,
            raw_score=float(positive[name]),
            weight=float(capped.get(name, 0.0)),
            capital=float(capital * capped.get(name, 0.0)),
        )
        for name in positive
    }
