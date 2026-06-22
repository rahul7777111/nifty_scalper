from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from cost_model import CostModel
from market_data import Candle
REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"


@dataclass(frozen=True)
class LabelPolicySpec:
    name: str
    version: str
    description: str
    required_future_horizon: int
    parameters: Dict[str, Any]
    supports_neutral: bool = False


@dataclass
class LabelObservation:
    index: int
    timestamp: str
    label: int
    forward_return: float
    barrier_hit: str
    time_to_hit: int
    volatility_regime: str
    trend_regime: str
    session_regime: str
    noisy_zone: bool
    below_cost_threshold: bool
    meta: Dict[str, Any]


@dataclass
class LabelDataset:
    policy: LabelPolicySpec
    X: List[List[float]]
    y: List[int]
    feature_names: List[str]
    observations: List[LabelObservation]
    label_distribution: Dict[str, int]
    neutral_samples_dropped: int
    sample_indices: List[int]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if np.isnan(out) or np.isinf(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _session_regime(ts: Any) -> str:
    try:
        hour = int(getattr(ts, "hour", 0))
        minute = int(getattr(ts, "minute", 0))
    except Exception:
        return "unknown_session"
    total = hour * 60 + minute
    if total <= (10 * 60):
        return "opening_session"
    if total >= (14 * 60 + 45):
        return "closing_session"
    return "midday_lull"


def _trend_regime(snapshot_regime: str, trend_strength: float) -> str:
    if snapshot_regime == "trending":
        return "trending_up" if trend_strength >= 0.0 else "trending_down"
    return "ranging"


def get_label_policy_spec(name: str, horizon: int, cost_model: CostModel | None = None) -> LabelPolicySpec:
    cm = cost_model or CostModel()
    cost_pct = cm.assumptions.round_trip_cost_pct
    min_edge = cm.assumptions.minimum_net_edge_pct
    base: Dict[str, LabelPolicySpec] = {
        "current_triple_barrier": LabelPolicySpec(
            name="current_triple_barrier",
            version="v1",
            description="Preserves the current binary triple-barrier behavior for baseline comparison.",
            required_future_horizon=horizon,
            parameters={"profit_target_pct": 0.01, "stop_loss_pct": 0.005},
        ),
        "cost_adjusted_triple_barrier": LabelPolicySpec(
            name="cost_adjusted_triple_barrier",
            version="v1",
            description="Requires barrier wins to clear conservative round-trip friction and minimum net edge.",
            required_future_horizon=horizon,
            parameters={"profit_target_pct": 0.01, "stop_loss_pct": 0.005, "cost_pct": cost_pct, "minimum_net_edge_pct": min_edge},
        ),
        "magnitude_filtered_direction": LabelPolicySpec(
            name="magnitude_filtered_direction",
            version="v1",
            description="Maps small forward moves into a neutral noisy zone, keeping only meaningful bullish or bearish moves.",
            required_future_horizon=horizon,
            parameters={"bullish_magnitude_pct": max(cost_pct + min_edge, 0.0020), "bearish_magnitude_pct": max(cost_pct + min_edge, 0.0020)},
            supports_neutral=True,
        ),
        "trade_quality_binary": LabelPolicySpec(
            name="trade_quality_binary",
            version="v1",
            description="Predicts whether a setup has enough net edge, acceptable MAE, and timely follow-through to justify a trade.",
            required_future_horizon=horizon,
            parameters={"profit_target_pct": 0.01, "stop_loss_pct": 0.005, "cost_pct": cost_pct, "minimum_net_edge_pct": min_edge, "max_mae_pct": 0.0075, "max_duration_bars": min(horizon, 18)},
        ),
        "trade_quality_ternary": LabelPolicySpec(
            name="trade_quality_ternary",
            version="v1",
            description="Separates strong trade-quality setups from noisy no-trade zones and explicitly marks poor/bearish outcomes.",
            required_future_horizon=horizon,
            parameters={"positive_threshold_pct": max(cost_pct + min_edge, 0.0020), "negative_threshold_pct": max(cost_pct + min_edge, 0.0020), "max_mae_pct": 0.0075},
            supports_neutral=True,
        ),
        "regime_specific_trade_quality": LabelPolicySpec(
            name="regime_specific_trade_quality",
            version="v1",
            description="Applies regime- and session-specific edge thresholds without using future information for regime classification.",
            required_future_horizon=horizon,
            parameters={
                "base_profit_target_pct": 0.01,
                "base_stop_loss_pct": 0.005,
                "cost_pct": cost_pct,
                "minimum_net_edge_pct": min_edge,
                "high_vol_multiplier": 1.35,
                "opening_session_multiplier": 1.20,
                "midday_multiplier": 1.10,
                "closing_session_multiplier": 0.90,
            },
        ),
        "abstain_allowed_trade_quality": LabelPolicySpec(
            name="abstain_allowed_trade_quality",
            version="v1",
            description="Binary trade-vs-no-trade objective that rewards skipping weak edges and midday noise.",
            required_future_horizon=horizon,
            parameters={"minimum_net_edge_pct": max(cost_pct + min_edge, 0.0020), "profit_target_pct": 0.01, "skip_midday": True},
        ),
    }
    if name not in base:
        raise KeyError(f"Unknown label policy: {name}")
    return base[name]


def available_label_policies() -> List[str]:
    return [
        "current_triple_barrier",
        "cost_adjusted_triple_barrier",
        "magnitude_filtered_direction",
        "trade_quality_binary",
        "trade_quality_ternary",
        "regime_specific_trade_quality",
        "abstain_allowed_trade_quality",
    ]


def _triple_barrier_path(now_close: float, future_closes: Sequence[float], *, profit_target_pct: float, stop_loss_pct: float) -> tuple[str, int]:
    for offset, price in enumerate(future_closes, start=1):
        ret = ((float(price) - now_close) / now_close) if now_close else 0.0
        if ret >= profit_target_pct:
            return "profit_target", offset
        if ret <= -stop_loss_pct:
            return "stop_loss", offset
    return "expiry", len(future_closes)


def _regime_snapshot(candles: Sequence[Candle], idx: int, *, lookback: int = 30) -> Dict[str, Any]:
    window = candles[max(0, idx - lookback + 1) : idx + 1]
    closes = np.asarray([_safe_float(c.close, 0.0) for c in window], dtype=float)
    highs = np.asarray([_safe_float(c.high, 0.0) for c in window], dtype=float)
    lows = np.asarray([_safe_float(c.low, 0.0) for c in window], dtype=float)
    if len(closes) < 5:
        regime = "ranging"
        atr_pct = 0.0
        realized_vol = 0.0
        trend_strength = 0.0
    else:
        prev = np.where(closes[:-1] == 0.0, 1.0, closes[:-1])
        returns = np.diff(closes) / prev
        realized_vol = float(np.std(returns))
        tr_components = np.maximum(highs[1:], closes[:-1]) - np.minimum(lows[1:], closes[:-1])
        atr_pct = float(np.mean(tr_components[-min(14, len(tr_components)) :]) / closes[-1]) if closes[-1] else 0.0
        trend_strength = float((closes[-1] - closes[0]) / closes[0]) if closes[0] else 0.0
        if realized_vol >= 0.012 or atr_pct >= 0.01:
            regime = "high_volatility"
        elif abs(trend_strength) >= 0.01:
            regime = "trending"
        elif realized_vol <= 0.004:
            regime = "low_volatility"
        else:
            regime = "ranging"
    session = _session_regime(candles[idx].time if idx < len(candles) else None)
    return {
        "volatility_regime": regime,
        "trend_regime": _trend_regime(regime, trend_strength),
        "session_regime": session,
        "atr_pct": atr_pct,
        "realized_vol": realized_vol,
        "trend_strength": trend_strength,
    }


def _policy_label(
    policy: LabelPolicySpec,
    *,
    forward_return: float,
    future_closes: Sequence[float],
    now_close: float,
    max_adverse_excursion: float,
    horizon: int,
    cost_model: CostModel,
    regime: Dict[str, Any],
) -> tuple[int | None, str, int, Dict[str, Any]]:
    params = policy.parameters
    barrier_hit, time_to_hit = _triple_barrier_path(
        now_close,
        future_closes,
        profit_target_pct=float(params.get("profit_target_pct", params.get("base_profit_target_pct", 0.01))),
        stop_loss_pct=float(params.get("stop_loss_pct", params.get("base_stop_loss_pct", 0.005))),
    )
    cost_pct = float(params.get("cost_pct", cost_model.assumptions.round_trip_cost_pct))
    minimum_net_edge_pct = float(params.get("minimum_net_edge_pct", cost_model.assumptions.minimum_net_edge_pct))
    session_regime = str(regime.get("session_regime") or "midday_lull")
    vol_regime = str(regime.get("volatility_regime") or "ranging")

    if policy.name == "current_triple_barrier":
        if barrier_hit == "profit_target":
            return 1, barrier_hit, time_to_hit, {"net_edge_pct": forward_return}
        if barrier_hit == "stop_loss":
            return 0, barrier_hit, time_to_hit, {"net_edge_pct": forward_return}
        return (1 if forward_return > 0.0 else 0), barrier_hit, time_to_hit, {"net_edge_pct": forward_return}

    if policy.name == "cost_adjusted_triple_barrier":
        net_edge = cost_model.net_edge_pct(forward_return)
        if barrier_hit == "profit_target" and net_edge >= minimum_net_edge_pct:
            return 1, barrier_hit, time_to_hit, {"net_edge_pct": net_edge}
        return 0, barrier_hit, time_to_hit, {"net_edge_pct": net_edge}

    if policy.name == "magnitude_filtered_direction":
        bull_mag = float(params.get("bullish_magnitude_pct", minimum_net_edge_pct))
        bear_mag = float(params.get("bearish_magnitude_pct", minimum_net_edge_pct))
        if forward_return >= bull_mag:
            return 1, barrier_hit, time_to_hit, {"net_edge_pct": cost_model.net_edge_pct(forward_return)}
        if forward_return <= -bear_mag:
            return -1, barrier_hit, time_to_hit, {"net_edge_pct": cost_model.net_edge_pct(abs(forward_return))}
        return 0, barrier_hit, time_to_hit, {"net_edge_pct": cost_model.net_edge_pct(forward_return)}

    if policy.name == "trade_quality_binary":
        net_edge = cost_model.net_edge_pct(forward_return)
        good = barrier_hit == "profit_target" and net_edge >= minimum_net_edge_pct and max_adverse_excursion <= float(params.get("max_mae_pct", 0.0075)) and time_to_hit <= int(params.get("max_duration_bars", horizon))
        return (1 if good else 0), barrier_hit, time_to_hit, {"net_edge_pct": net_edge}

    if policy.name == "trade_quality_ternary":
        pos_thr = float(params.get("positive_threshold_pct", minimum_net_edge_pct))
        neg_thr = float(params.get("negative_threshold_pct", minimum_net_edge_pct))
        net_edge = cost_model.net_edge_pct(forward_return)
        if barrier_hit == "profit_target" and net_edge >= pos_thr and max_adverse_excursion <= float(params.get("max_mae_pct", 0.0075)):
            return 1, barrier_hit, time_to_hit, {"net_edge_pct": net_edge}
        if forward_return <= -neg_thr or barrier_hit == "stop_loss":
            return -1, barrier_hit, time_to_hit, {"net_edge_pct": net_edge}
        return 0, barrier_hit, time_to_hit, {"net_edge_pct": net_edge}

    if policy.name == "regime_specific_trade_quality":
        profit_target = float(params.get("base_profit_target_pct", 0.01))
        if vol_regime == "high_volatility":
            profit_target *= float(params.get("high_vol_multiplier", 1.35))
        if session_regime == "opening_session":
            profit_target *= float(params.get("opening_session_multiplier", 1.20))
        elif session_regime == "midday_lull":
            profit_target *= float(params.get("midday_multiplier", 1.10))
        elif session_regime == "closing_session":
            profit_target *= float(params.get("closing_session_multiplier", 0.90))
        net_edge = cost_model.net_edge_pct(forward_return)
        good = barrier_hit == "profit_target" and forward_return >= profit_target and net_edge >= minimum_net_edge_pct
        return (1 if good else 0), barrier_hit, time_to_hit, {"net_edge_pct": net_edge, "adjusted_profit_target_pct": profit_target}

    if policy.name == "abstain_allowed_trade_quality":
        skip_midday = bool(params.get("skip_midday", True))
        net_edge = cost_model.net_edge_pct(forward_return)
        if skip_midday and session_regime == "midday_lull":
            return 0, barrier_hit, time_to_hit, {"net_edge_pct": net_edge}
        good = barrier_hit == "profit_target" and net_edge >= minimum_net_edge_pct
        return (1 if good else 0), barrier_hit, time_to_hit, {"net_edge_pct": net_edge}

    raise KeyError(f"Unhandled policy: {policy.name}")


def build_label_dataset(
    candles: Sequence[Candle],
    *,
    policy_name: str,
    lookback: int,
    horizon: int,
    cost_model: CostModel | None = None,
    contexts: Optional[Sequence[Any]] = None,
    include_features: bool = True,
) -> LabelDataset:
    c = list(candles or [])
    cm = cost_model or CostModel()
    policy = get_label_policy_spec(policy_name, horizon, cm)
    if include_features:
        from ml_pipeline import build_supervised_dataset_v2

        X_full, _, feature_names = build_supervised_dataset_v2(c, lookback=lookback, horizon=horizon, contexts=contexts)
    else:
        X_full, feature_names = [], []

    observations: List[LabelObservation] = []
    y: List[int] = []
    filtered_X: List[List[float]] = []
    neutral_samples_dropped = 0
    sample_indices: List[int] = []

    for sample_idx, idx in enumerate(range(lookback, len(c) - horizon)):
        now_close = _safe_float(c[idx].close, 0.0)
        future = [_safe_float(c[pos].close, now_close) for pos in range(idx + 1, idx + horizon + 1)]
        if not future or now_close <= 0.0:
            continue
        forward_return = ((future[-1] - now_close) / now_close) if now_close else 0.0
        path_returns = [((price - now_close) / now_close) if now_close else 0.0 for price in future]
        max_adverse_excursion = abs(min(path_returns)) if path_returns else 0.0
        regime = _regime_snapshot(c, idx)
        label, barrier_hit, time_to_hit, meta = _policy_label(
            policy,
            forward_return=forward_return,
            future_closes=future,
            now_close=now_close,
            max_adverse_excursion=max_adverse_excursion,
            horizon=horizon,
            cost_model=cm,
            regime=regime,
        )
        noisy_zone = abs(forward_return) < float(policy.parameters.get("minimum_net_edge_pct", cm.assumptions.minimum_net_edge_pct))
        below_cost = abs(forward_return) < cm.estimate_cost_pct()
        observations.append(
            LabelObservation(
                index=idx,
                timestamp=c[idx].time.isoformat(),
                label=int(label) if label is not None else 0,
                forward_return=float(forward_return),
                barrier_hit=barrier_hit,
                time_to_hit=int(time_to_hit),
                volatility_regime=str(regime["volatility_regime"]),
                trend_regime=str(regime["trend_regime"]),
                session_regime=str(regime["session_regime"]),
                noisy_zone=bool(noisy_zone),
                below_cost_threshold=bool(below_cost),
                meta=meta,
            )
        )
        if label is None:
            continue
        if policy.supports_neutral and int(label) == 0 and policy.name in {"magnitude_filtered_direction", "trade_quality_ternary"}:
            neutral_samples_dropped += 1
            continue
        if include_features:
            filtered_X.append(list(map(float, X_full[sample_idx])))
        y.append(1 if int(label) > 0 else 0)
        sample_indices.append(sample_idx)

    neutral_enabled = policy.name in {"magnitude_filtered_direction", "trade_quality_ternary"}
    label_distribution = {
        "positive": sum(1 for obs in observations if obs.label > 0),
        "neutral": sum(1 for obs in observations if neutral_enabled and obs.label == 0),
        "negative": sum(1 for obs in observations if obs.label < 0 or (obs.label == 0 and not neutral_enabled)),
        "observations": len(observations),
        "training_samples": len(y),
    }
    return LabelDataset(
        policy=policy,
        X=filtered_X,
        y=y,
        feature_names=list(feature_names),
        observations=observations,
        label_distribution=label_distribution,
        neutral_samples_dropped=neutral_samples_dropped,
        sample_indices=sample_indices,
    )


def observations_to_jsonable(rows: Iterable[LabelObservation]) -> List[Dict[str, Any]]:
    return [{**asdict(row), "meta": dict(row.meta)} for row in rows]


def write_policy_distribution_report(dataset: LabelDataset, name: str) -> Dict[str, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": asdict(dataset.policy),
        "label_distribution": dataset.label_distribution,
        "neutral_samples_dropped": dataset.neutral_samples_dropped,
        "sample_preview": observations_to_jsonable(list(dataset.observations[:25])),
    }
    json_path = REPORTS_DIR / f"{name}.json"
    md_path = REPORTS_DIR / f"{name}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_lines = [
        f"# {name.replace('_', ' ').title()}",
        "",
        f"- Policy: `{dataset.policy.name}:{dataset.policy.version}`",
        f"- Description: {dataset.policy.description}",
        f"- Training samples: `{len(dataset.y)}`",
        f"- Neutral samples dropped: `{dataset.neutral_samples_dropped}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in dataset.label_distribution.items():
        md_lines.append(f"| {key} | {value} |")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {"json": json_path, "md": md_path}
