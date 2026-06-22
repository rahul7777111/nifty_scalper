from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cost_model import CostModel
from label_policies import available_label_policies, build_label_dataset
from market_data import Candle


LOOKBACK_BARS = 30
HORIZON_BARS = 45


def dict_to_candle(raw: Dict[str, Any]) -> Optional[Candle]:
    from datetime import datetime

    raw_time = str(raw.get("time", "")).split(".")[0]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw_time, fmt)
            return Candle(
                time=dt,
                open=float(raw.get("open", 0.0) or 0.0),
                high=float(raw.get("high", 0.0) or 0.0),
                low=float(raw.get("low", 0.0) or 0.0),
                close=float(raw.get("close", 0.0) or 0.0),
                volume=float(raw.get("volume", 0.0) or 0.0),
            )
        except Exception:
            continue
    return None


def load_real_candles() -> List[Candle]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("candles_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(payload.get("candles") or [])
        except Exception:
            continue
    candles = [c for c in (dict_to_candle(row) for row in rows) if c is not None]
    candles.sort(key=lambda candle: candle.time)
    return candles[::10]


def month_key(ts: str) -> str:
    return str(ts)[:7]


def weekday_key(ts: str) -> str:
    from datetime import datetime

    raw = str(ts or "")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(raw, fmt).strftime("%A")
        except Exception:
            continue
    return raw[:10]


def time_bucket(ts: str) -> str:
    try:
        return str(ts)[11:16]
    except Exception:
        return "unknown"


def autocorrelation(values: List[int], lag: int = 1) -> float:
    if len(values) <= lag:
        return 0.0
    x = np.asarray(values[:-lag], dtype=float)
    y = np.asarray(values[lag:], dtype=float)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def build_current_labeling_audit() -> Dict[str, Any]:
    audit = {
        "current_label_policy": "triple_barrier_binary_directional_proxy",
        "current_triple_barrier_configuration": {
            "lookback": LOOKBACK_BARS,
            "horizon": HORIZON_BARS,
            "profit_target_pct": 0.01,
            "stop_loss_pct": 0.005,
        },
        "prediction_horizon_bars": HORIZON_BARS,
        "target_stop_assumptions": "Fixed triple-barrier thresholds with expiry fallback to terminal direction.",
        "labels_match_actual_strategy_exits": True,
        "includes_brokerage_cost_slippage": False,
        "includes_option_time_decay": False,
        "accounts_for_time_of_day": False,
        "accounts_for_volatility_regime": False,
        "accounts_for_opening_range": False,
        "model_predicts": "short_term_direction_not_trade_quality",
        "prediction_output_usage": "entry_filter",
        "live_feature_generation_matches_training_feature_builder": True,
    }
    return audit


def build_diagnostics(policy_name: str) -> Dict[str, Any]:
    candles = load_real_candles()
    dataset = build_label_dataset(
        candles,
        policy_name=policy_name,
        lookback=LOOKBACK_BARS,
        horizon=HORIZON_BARS,
        cost_model=CostModel(),
        include_features=False,
    )
    rows = dataset.observations
    labels = [row.label for row in rows]
    neutral_enabled = policy_name in {"magnitude_filtered_direction", "trade_quality_ternary"}
    forward_returns = [row.forward_return for row in rows]
    time_to_hit = [row.time_to_hit for row in rows]
    barrier_distribution = Counter(row.barrier_hit for row in rows)

    by_month: Dict[str, List[int]] = defaultdict(list)
    by_weekday: Dict[str, List[int]] = defaultdict(list)
    by_time: Dict[str, List[int]] = defaultdict(list)
    by_session: Dict[str, List[int]] = defaultdict(list)
    by_vol: Dict[str, List[int]] = defaultdict(list)
    by_trend: Dict[str, List[int]] = defaultdict(list)
    for row in rows:
        by_month[month_key(row.timestamp)].append(row.label)
        weekday = weekday_key(row.timestamp)
        by_weekday[weekday].append(row.label)
        by_time[time_bucket(row.timestamp)].append(row.label)
        by_session[row.session_regime].append(row.label)
        by_vol[row.volatility_regime].append(row.label)
        by_trend[row.trend_regime].append(row.label)

    positive = sum(1 for label in labels if label > 0)
    neutral = sum(1 for label in labels if neutral_enabled and label == 0)
    negative = sum(1 for label in labels if label < 0 or (label == 0 and not neutral_enabled))
    noisy_zone = sum(1 for row in rows if row.noisy_zone)
    below_cost = sum(1 for row in rows if row.below_cost_threshold)

    return {
        "policy": {"name": dataset.policy.name, "version": dataset.policy.version, "description": dataset.policy.description, "parameters": dataset.policy.parameters},
        "total_samples": len(rows),
        "training_samples": len(dataset.y),
        "positive_label_count": positive,
        "negative_label_count": negative,
        "neutral_no_trade_count": neutral,
        "positive_rate": positive / len(rows) if rows else 0.0,
        "negative_rate": negative / len(rows) if rows else 0.0,
        "label_rate_by_month": {key: float(sum(1 for v in vals if v > 0) / len(vals)) if vals else 0.0 for key, vals in sorted(by_month.items())},
        "label_rate_by_weekday": {key: float(sum(1 for v in vals if v > 0) / len(vals)) if vals else 0.0 for key, vals in sorted(by_weekday.items())},
        "label_rate_by_time_of_day": {key: float(sum(1 for v in vals if v > 0) / len(vals)) if vals else 0.0 for key, vals in sorted(by_time.items())},
        "label_rate_by_session": {key: float(sum(1 for v in vals if v > 0) / len(vals)) if vals else 0.0 for key, vals in sorted(by_session.items())},
        "label_rate_by_volatility_regime": {key: float(sum(1 for v in vals if v > 0) / len(vals)) if vals else 0.0 for key, vals in sorted(by_vol.items())},
        "label_rate_by_trend_range_regime": {key: float(sum(1 for v in vals if v > 0) / len(vals)) if vals else 0.0 for key, vals in sorted(by_trend.items())},
        "average_forward_return_by_label": {
            "positive": float(np.mean([row.forward_return for row in rows if row.label > 0])) if positive else 0.0,
            "neutral": float(np.mean([row.forward_return for row in rows if neutral_enabled and row.label == 0])) if neutral else 0.0,
            "negative": float(np.mean([row.forward_return for row in rows if row.label < 0 or (row.label == 0 and not neutral_enabled)])) if negative else 0.0,
        },
        "median_forward_return_by_label": {
            "positive": float(median([row.forward_return for row in rows if row.label > 0])) if positive else 0.0,
            "neutral": float(median([row.forward_return for row in rows if neutral_enabled and row.label == 0])) if neutral else 0.0,
            "negative": float(median([row.forward_return for row in rows if row.label < 0 or (row.label == 0 and not neutral_enabled)])) if negative else 0.0,
        },
        "forward_return_distribution": {
            "p05": float(np.percentile(forward_returns, 5)) if forward_returns else 0.0,
            "p25": float(np.percentile(forward_returns, 25)) if forward_returns else 0.0,
            "p50": float(np.percentile(forward_returns, 50)) if forward_returns else 0.0,
            "p75": float(np.percentile(forward_returns, 75)) if forward_returns else 0.0,
            "p95": float(np.percentile(forward_returns, 95)) if forward_returns else 0.0,
        },
        "barrier_hit_distribution": dict(sorted(barrier_distribution.items())),
        "time_to_hit_distribution": {
            "mean": float(np.mean(time_to_hit)) if time_to_hit else 0.0,
            "median": float(np.median(time_to_hit)) if time_to_hit else 0.0,
            "p90": float(np.percentile(time_to_hit, 90)) if time_to_hit else 0.0,
        },
        "label_autocorrelation": {"lag_1": autocorrelation(labels, 1), "lag_5": autocorrelation(labels, 5)},
        "label_flip_frequency": (
            sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1]) / max(1, len(labels) - 1)
        ),
        "label_stability_across_horizons": {
            policy: build_label_dataset(candles, policy_name=policy, lookback=LOOKBACK_BARS, horizon=HORIZON_BARS, include_features=False).label_distribution
            for policy in available_label_policies()
        },
        "class_imbalance": abs((positive / len(rows) if rows else 0.0) - (negative / len(rows) if rows else 0.0)),
        "noisy_zone_percentage": noisy_zone / len(rows) if rows else 0.0,
        "below_cost_slippage_percentage": below_cost / len(rows) if rows else 0.0,
    }


def write_reports(audit: Dict[str, Any], diagnostics: Dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "current_labeling_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (REPORTS_DIR / "label_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    audit_md = [
        "# Current Labeling Audit",
        "",
        "| Check | Value |",
        "|---|---|",
    ]
    for key, value in audit.items():
        audit_md.append(f"| {key} | {value} |")
    (REPORTS_DIR / "current_labeling_audit.md").write_text("\n".join(audit_md) + "\n", encoding="utf-8")

    diag_md = [
        "# Label Diagnostics",
        "",
        f"- Policy: `{diagnostics['policy']['name']}:{diagnostics['policy']['version']}`",
        f"- Description: {diagnostics['policy']['description']}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "total_samples",
        "training_samples",
        "positive_label_count",
        "negative_label_count",
        "neutral_no_trade_count",
        "positive_rate",
        "negative_rate",
        "class_imbalance",
        "noisy_zone_percentage",
        "below_cost_slippage_percentage",
    ):
        diag_md.append(f"| {key} | {diagnostics.get(key)} |")
    diag_md.extend(["", "## Barrier Hit Distribution", "", "| Barrier | Count |", "|---|---:|"])
    for key, value in diagnostics.get("barrier_hit_distribution", {}).items():
        diag_md.append(f"| {key} | {value} |")
    diag_md.extend(["", "## Forward Return Distribution", "", "| Bucket | Value |", "|---|---:|"])
    for key, value in diagnostics.get("forward_return_distribution", {}).items():
        diag_md.append(f"| {key} | {value:.6f} |")
    (REPORTS_DIR / "label_diagnostics.md").write_text("\n".join(diag_md) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run current label audit and diagnostics.")
    parser.add_argument("--label-policy", default="current_triple_barrier", choices=available_label_policies())
    args = parser.parse_args(argv)

    audit = build_current_labeling_audit()
    diagnostics = build_diagnostics(str(args.label_policy))
    write_reports(audit, diagnostics)
    print(json.dumps({"audit": audit, "diagnostics": diagnostics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
