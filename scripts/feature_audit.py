from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
REPORTS_DIR = REPO_ROOT / "reports"
DATA_DIR = REPO_ROOT / "data"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from indicators import atr  # noqa: E402
from market_data import Candle  # noqa: E402
from ml_pipeline import TARGET_FEATURES, build_supervised_dataset_v2, walk_forward_validate_production  # noqa: E402
from ml_signals import WeightedEnsembleClassifier, build_model  # noqa: E402

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"scikit-learn is required for feature_audit.py: {exc}")


LOOKBACK_BARS = 30
HORIZON_BARS = 45
WF_SPLITS = 3
WF_GAP = 5
WF_MIN_TRAIN = 250
WF_MIN_TEST = 80
CORRELATION_THRESHOLD = 0.90
PRIORITIZED_CANDIDATES = [
    "is_opening_session",
    "is_closing_session",
    "is_midday_lull",
    "volatility_percentile_60",
    "realized_vol_percentile_60",
    "volatility_regime_classifier",
    "dist_from_opening_low_pct",
    "opening_range_breakout_strength",
    "dist_to_rolling_low_20",
    "rolling_range_position_20",
    "option_bid_ask_spread_pct",
    "theta_to_vega_ratio",
    "gamma_to_theta_ratio",
]
OPTIONS_CONTEXT_FEATURES = {
    "ctx_iv",
    "ctx_iv_change_pct",
    "ctx_iv_percentile",
    "ctx_delta",
    "ctx_gamma",
    "ctx_vega",
    "ctx_theta",
    "ctx_spot",
    "ctx_option_price",
    "ctx_adx",
    "ctx_trend_strength",
    "ctx_choppiness",
    "ctx_volume_sma",
    "ctx_time_sin",
    "ctx_time_cos",
    "ctx_dte_norm",
    "price_to_spot_pct",
    "option_to_spot_pct",
    "delta_abs",
    "greeks_imbalance",
    "option_bid_ask_spread_pct",
    "theta_to_vega_ratio",
    "gamma_to_theta_ratio",
}


@dataclass
class CandidateResult:
    feature: str
    category: str
    status: str
    reason: str
    roc_auc: float
    f1: float
    profit_factor: float
    delta_roc_auc: float
    delta_f1: float
    delta_profit_factor: float


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def feature_category(name: str) -> str:
    if name in {"is_opening_session", "is_closing_session", "is_midday_lull"} or "time" in name:
        return "time_of_day"
    if "opening" in name:
        return "opening_range"
    if "rolling" in name or "range_position" in name:
        return "market_structure"
    if "vol" in name or "atr_pct_regime" in name:
        return "volatility_regime"
    if name in {"option_bid_ask_spread_pct"}:
        return "option_liquidity"
    if "theta" in name or "gamma" in name or "delta" in name or "vega" in name or name.startswith("ctx_"):
        return "greeks_or_context"
    return "core"


def dict_to_candle(raw: dict) -> Candle:
    t_str = raw.get("time", "")
    dt = None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(t_str.split(".")[0], fmt)
            break
        except Exception:
            pass
    if dt is None:
        dt = datetime.now()
    return Candle(
        time=dt,
        open=_safe_float(raw.get("open", 0.0)),
        high=_safe_float(raw.get("high", 0.0)),
        low=_safe_float(raw.get("low", 0.0)),
        close=_safe_float(raw.get("close", 0.0)),
        volume=_safe_float(raw.get("volume", 0.0)),
    )


def load_real_candles() -> List[Candle]:
    candles: List[Candle] = []
    for path in sorted(DATA_DIR.glob("candles_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            candles.extend(dict_to_candle(row) for row in (payload.get("candles") or []))
        except Exception:
            continue
    candles = sorted(candles, key=lambda c: c.time)
    return candles[::10]


def build_dataset(candles: Sequence[Candle]) -> tuple[list[list[float]], list[int], list[str]]:
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    latest_atr = atr(highs, lows, closes, period=14) or 10.0
    latest_close = closes[-1] if closes else 100.0
    tb_profit_target_pct = (1.5 * latest_atr) / latest_close if latest_close else 0.01
    tb_stop_loss_pct = (2.0 * latest_atr) / latest_close if latest_close else 0.005
    return build_supervised_dataset_v2(
        candles,
        lookback=LOOKBACK_BARS,
        use_triple_barrier=True,
        tb_profit_target_pct=tb_profit_target_pct,
        tb_stop_loss_pct=tb_stop_loss_pct,
        horizon=HORIZON_BARS,
    )


def percentile_rank_series(values: np.ndarray, window: int = 60) -> np.ndarray:
    out = np.zeros(len(values), dtype=float)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        hist = values[start : idx + 1]
        current = values[idx]
        if len(hist) <= 1:
            out[idx] = 0.5
            continue
        out[idx] = float(np.mean(hist <= current))
    return out


def derive_candidate_features(
    X_full: Sequence[Sequence[float]],
    feature_names: Sequence[str],
) -> tuple[list[list[float]], list[str]]:
    matrix = np.asarray(X_full, dtype=float)
    names = list(feature_names)
    rows = [list(map(float, row)) for row in matrix.tolist()]
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    derived_specs = []
    if "atr_pct" in name_to_idx:
        atr_pct_vals = matrix[:, name_to_idx["atr_pct"]]
        derived_specs.append(("atr_percentile_60", percentile_rank_series(atr_pct_vals, window=60)))
    if "realized_vol_30" in name_to_idx:
        rv_vals = matrix[:, name_to_idx["realized_vol_30"]]
        derived_specs.append(("realized_vol_percentile_60", percentile_rank_series(rv_vals, window=60)))
        derived_specs.append(("volatility_percentile_60", percentile_rank_series(rv_vals, window=60)))
    if "atr_pct_regime_10" in name_to_idx and "realized_vol_30" in name_to_idx:
        rv_pct = percentile_rank_series(matrix[:, name_to_idx["realized_vol_30"]], window=60)
        atr_pct_rank = percentile_rank_series(matrix[:, name_to_idx["atr_pct"]], window=60) if "atr_pct" in name_to_idx else np.zeros(len(matrix), dtype=float)
        regime_classifier = np.where((rv_pct >= 0.7) | (atr_pct_rank >= 0.7), 1.0, 0.0)
        derived_specs.append(("volatility_regime_classifier", regime_classifier))

    for feature_name, values in derived_specs:
        if feature_name in name_to_idx:
            continue
        names.append(feature_name)
        for row, value in zip(rows, values.tolist()):
            row.append(_safe_float(value))
    return rows, names


def baseline_distribution_report(
    X_full: Sequence[Sequence[float]],
    feature_names: Sequence[str],
) -> Dict[str, object]:
    matrix = np.asarray(X_full, dtype=float)
    features: Dict[str, object] = {}
    for idx, name in enumerate(feature_names):
        col = matrix[:, idx]
        quantiles = np.quantile(col, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).tolist()
        features[name] = {
            "mean": _safe_float(np.mean(col)),
            "std": _safe_float(np.std(col)),
            "variance": _safe_float(np.var(col)),
            "quantiles": [float(v) for v in quantiles],
            "min": _safe_float(np.min(col)),
            "max": _safe_float(np.max(col)),
        }
    return {"feature_count": len(feature_names), "features": features}


def evaluate_feature_set(
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    candles: Sequence[Candle],
) -> Dict[str, float]:
    def model_factory():
        model = build_model("logistic_regression")
        return model if model is not None else WeightedEnsembleClassifier()

    result = walk_forward_validate_production(
        X,
        y,
        n_splits=WF_SPLITS,
        gap=WF_GAP,
        min_train_samples=WF_MIN_TRAIN,
        min_test_samples=WF_MIN_TEST,
        gate_roc_auc=0.55,
        gate_accuracy=0.53,
        gate_f1=0.50,
        model_factory=model_factory,
        candles=candles,
        lookback=LOOKBACK_BARS,
    )
    return dict(result.get("metrics") or {})


def matrix_for_features(
    X_full: Sequence[Sequence[float]],
    feature_names: Sequence[str],
    selected: Sequence[str],
) -> List[List[float]]:
    indices = [feature_names.index(name) for name in selected]
    return [[_safe_float(row[idx]) for idx in indices] for row in X_full]


def permutation_importance_report(
    X_full: Sequence[Sequence[float]],
    y: Sequence[int],
    feature_names: Sequence[str],
) -> Dict[str, object]:
    split_idx = max(int(len(X_full) * 0.8), WF_MIN_TRAIN)
    split_idx = min(split_idx, len(X_full) - max(50, len(X_full) // 10))
    X_train = np.asarray(X_full[:split_idx], dtype=float)
    y_train = np.asarray(y[:split_idx], dtype=int)
    X_test = np.asarray(X_full[split_idx:], dtype=float)
    y_test = np.asarray(y[split_idx:], dtype=int)

    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=1,
        random_state=42,
        scoring="roc_auc",
        n_jobs=-1,
    )
    rows = []
    for idx, name in enumerate(feature_names):
        rows.append(
            {
                "feature": name,
                "category": feature_category(name),
                "importance_mean": _safe_float(perm.importances_mean[idx]),
                "importance_std": _safe_float(perm.importances_std[idx]),
            }
        )
    rows.sort(key=lambda row: row["importance_mean"], reverse=True)
    strongest = rows[:10]
    weakest = sorted(rows, key=lambda row: row["importance_mean"])[:10]
    recommendations = []
    for row in strongest[:5]:
        recommendations.append(f"Keep `{row['feature']}` prominent; it is among the strongest predictive features.")
    for row in weakest[:5]:
        recommendations.append(f"Review `{row['feature']}` for removal or de-prioritization; importance is weak.")
    return {
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "ranking": rows,
        "strongest_features": strongest,
        "weakest_features": weakest,
        "recommendations": recommendations,
    }


def correlation_report(X_full: Sequence[Sequence[float]], feature_names: Sequence[str]) -> Dict[str, object]:
    matrix = np.asarray(X_full, dtype=float)
    constant_features = []
    near_zero_features = []
    for idx, name in enumerate(feature_names):
        col = matrix[:, idx]
        std = float(np.std(col))
        non_zero_ratio = float(np.mean(np.abs(col) > 1e-12))
        if std <= 1e-12:
            constant_features.append(name)
        elif non_zero_ratio < 0.02:
            near_zero_features.append({"feature": name, "non_zero_ratio": non_zero_ratio, "std": std})

    variable_indices = [idx for idx, name in enumerate(feature_names) if name not in constant_features]
    variable_names = [feature_names[idx] for idx in variable_indices]
    corr_pairs = []
    if variable_indices:
        corr = np.corrcoef(matrix[:, variable_indices], rowvar=False)
        for i in range(len(variable_names)):
            for j in range(i + 1, len(variable_names)):
                value = _safe_float(corr[i, j])
                if abs(value) >= CORRELATION_THRESHOLD:
                    corr_pairs.append(
                        {
                            "feature_a": variable_names[i],
                            "feature_b": variable_names[j],
                            "correlation": value,
                        }
                    )
    corr_pairs.sort(key=lambda row: abs(row["correlation"]), reverse=True)
    redundant_features = []
    for row in corr_pairs:
        redundant_features.append(
            {
                "drop_candidate": row["feature_b"],
                "keep_candidate": row["feature_a"],
                "correlation": row["correlation"],
            }
        )
    return {
        "threshold": CORRELATION_THRESHOLD,
        "constant_features": constant_features,
        "near_zero_features": near_zero_features,
        "high_correlation_pairs": corr_pairs,
        "redundant_feature_candidates": redundant_features[:20],
    }


def candidate_report(
    X_full: Sequence[Sequence[float]],
    y: Sequence[int],
    candles: Sequence[Candle],
    feature_names: Sequence[str],
) -> Dict[str, object]:
    X_augmented, feature_names_augmented = derive_candidate_features(X_full, feature_names)
    legacy_target_features = [name for name in TARGET_FEATURES if name != "is_closing_session"]
    baseline_metrics = evaluate_feature_set(matrix_for_features(X_full, feature_names, legacy_target_features), y, candles)
    series_map = {
        name: np.asarray([_safe_float(row[feature_names_augmented.index(name)]) for row in X_augmented], dtype=float)
        for name in feature_names_augmented
    }
    results: List[CandidateResult] = []
    individually_accepted: List[str] = []

    for name in PRIORITIZED_CANDIDATES:
        if name not in series_map:
            continue
        category = feature_category(name)
        series = series_map[name]
        std = float(np.std(series))
        non_zero_ratio = float(np.mean(np.abs(series) > 1e-12))
        if name in OPTIONS_CONTEXT_FEATURES and (std <= 1e-12 or non_zero_ratio < 0.02):
            results.append(
                CandidateResult(
                    feature=name,
                    category=category,
                    status="deferred",
                    reason="missing_point_in_time_options_context",
                    roc_auc=baseline_metrics.get("roc_auc", 0.0),
                    f1=baseline_metrics.get("f1", 0.0),
                    profit_factor=baseline_metrics.get("profit_factor", 0.0),
                    delta_roc_auc=0.0,
                    delta_f1=0.0,
                    delta_profit_factor=0.0,
                )
            )
            continue

        candidate_metrics = evaluate_feature_set(
            matrix_for_features(X_augmented, feature_names_augmented, legacy_target_features + [name]),
            y,
            candles,
        )
        delta_auc = candidate_metrics.get("roc_auc", 0.0) - baseline_metrics.get("roc_auc", 0.0)
        delta_f1 = candidate_metrics.get("f1", 0.0) - baseline_metrics.get("f1", 0.0)
        delta_pf = candidate_metrics.get("profit_factor", 0.0) - baseline_metrics.get("profit_factor", 0.0)
        delta_dd = candidate_metrics.get("drawdown", baseline_metrics.get("drawdown", 0.0)) - baseline_metrics.get("drawdown", 0.0)
        accepted = delta_auc > 0.0 and delta_f1 >= 0.0 and delta_pf > 0.0 and delta_dd <= 0.01
        if accepted:
            individually_accepted.append(name)
        results.append(
            CandidateResult(
                feature=name,
                category=category,
                status="accepted" if accepted else "rejected",
                reason="passes_strict_validation" if accepted else "fails_strict_validation",
                roc_auc=candidate_metrics.get("roc_auc", 0.0),
                f1=candidate_metrics.get("f1", 0.0),
                profit_factor=candidate_metrics.get("profit_factor", 0.0),
                delta_roc_auc=delta_auc,
                delta_f1=delta_f1,
                delta_profit_factor=delta_pf,
            )
        )

    greedy_selected: List[str] = []
    greedy_best = dict(baseline_metrics)
    remaining = individually_accepted[:]
    while remaining:
        improved = []
        for name in remaining:
            metrics = evaluate_feature_set(
                matrix_for_features(X_augmented, feature_names_augmented, legacy_target_features + greedy_selected + [name]),
                y,
                candles,
            )
            if (
                metrics.get("roc_auc", 0.0) > greedy_best.get("roc_auc", 0.0)
                and metrics.get("f1", 0.0) > greedy_best.get("f1", 0.0)
                and metrics.get("profit_factor", 0.0) > greedy_best.get("profit_factor", 0.0)
            ):
                improved.append((name, metrics))
        if not improved:
            break
        improved.sort(
            key=lambda item: (
                item[1].get("roc_auc", 0.0) - greedy_best.get("roc_auc", 0.0),
                item[1].get("f1", 0.0) - greedy_best.get("f1", 0.0),
                item[1].get("profit_factor", 0.0) - greedy_best.get("profit_factor", 0.0),
            ),
            reverse=True,
        )
        chosen_name, chosen_metrics = improved[0]
        greedy_selected.append(chosen_name)
        remaining.remove(chosen_name)
        greedy_best = dict(chosen_metrics)

    return {
        "legacy_target_features": legacy_target_features,
        "legacy_baseline_metrics": baseline_metrics,
        "candidate_results": [asdict(result) for result in results],
        "greedy_selected_features": greedy_selected,
        "greedy_selected_metrics": greedy_best,
        "recommendations": [
            f"Promote `{name}` only if it continues to pass leakage-safe walk-forward validation." for name in greedy_selected
        ] if greedy_selected else [
            "Do not promote additional features yet; current strict validation rejects the remaining candidates."
        ],
    }


def run_feature_audit(candles: Sequence[Candle] | None = None) -> Dict[str, object]:
    candles = list(candles or load_real_candles())
    if not candles:
        raise SystemExit("No candles found in data/. Run data collection first.")

    X_full, y, feature_names = build_dataset(candles)
    importance = permutation_importance_report(X_full, y, feature_names)
    correlation = correlation_report(X_full, feature_names)
    candidates = candidate_report(X_full, y, candles, feature_names)
    distributions = baseline_distribution_report(X_full, feature_names)

    write_json(REPORTS_DIR / "feature_importance_report.json", importance)
    write_json(REPORTS_DIR / "feature_correlation_report.json", correlation)
    write_json(REPORTS_DIR / "candidate_feature_report.json", candidates)
    write_json(REPORTS_DIR / "training_feature_distribution.json", distributions)

    write_markdown(REPORTS_DIR / "feature_importance_report.md", render_importance_markdown(importance))
    write_markdown(REPORTS_DIR / "feature_correlation_report.md", render_correlation_markdown(correlation))
    write_markdown(REPORTS_DIR / "candidate_feature_report.md", render_candidates_markdown(candidates))

    summary = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "samples": len(X_full),
        "feature_count": len(feature_names),
        "legacy_baseline_metrics": candidates["legacy_baseline_metrics"],
        "greedy_selected_features": candidates["greedy_selected_features"],
        "greedy_selected_metrics": candidates["greedy_selected_metrics"],
        "strongest_features": importance["strongest_features"],
        "weakest_features": importance["weakest_features"],
        "redundant_feature_candidates": correlation["redundant_feature_candidates"],
        "recommendations": list(importance.get("recommendations", [])) + list(candidates.get("recommendations", [])),
    }
    write_json(REPORTS_DIR / "feature_audit_summary.json", summary)
    return summary


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def render_importance_markdown(payload: Dict[str, object]) -> List[str]:
    lines = [
        "# Feature Importance Report",
        "",
        f"- Train samples: `{payload['train_samples']}`",
        f"- Test samples: `{payload['test_samples']}`",
        "",
        "| Rank | Feature | Category | Permutation ROC-AUC Drop | Std |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for rank, row in enumerate(payload["ranking"][:30], start=1):
        lines.append(
            f"| {rank} | {row['feature']} | {row['category']} | {row['importance_mean']:.6f} | {row['importance_std']:.6f} |"
        )
    lines.extend(["", "## Strongest Features", ""])
    for row in payload.get("strongest_features", [])[:10]:
        lines.append(f"- `{row['feature']}` ({row['importance_mean']:.6f})")
    lines.extend(["", "## Weakest Features", ""])
    for row in payload.get("weakest_features", [])[:10]:
        lines.append(f"- `{row['feature']}` ({row['importance_mean']:.6f})")
    lines.extend(["", "## Recommendations", ""])
    for recommendation in payload.get("recommendations", []):
        lines.append(f"- {recommendation}")
    return lines


def render_correlation_markdown(payload: Dict[str, object]) -> List[str]:
    lines = [
        "# Feature Correlation Report",
        "",
        f"- High-correlation threshold: `|rho| >= {payload['threshold']:.2f}`",
        f"- Constant features: `{len(payload['constant_features'])}`",
        f"- Near-zero features: `{len(payload['near_zero_features'])}`",
        "",
        "## Constant Features",
        "",
        ", ".join(payload["constant_features"]) if payload["constant_features"] else "None.",
        "",
        "## High-Correlation Pairs",
        "",
        "| Feature A | Feature B | Correlation |",
        "| --- | --- | ---: |",
    ]
    for row in payload["high_correlation_pairs"][:30]:
        lines.append(f"| {row['feature_a']} | {row['feature_b']} | {row['correlation']:.4f} |")
    if not payload["high_correlation_pairs"]:
        lines.append("| None | None | 0.0000 |")
    lines.extend(["", "## Redundant Feature Candidates", ""])
    for row in payload.get("redundant_feature_candidates", [])[:10]:
        lines.append(
            f"- Consider dropping `{row['drop_candidate']}` while keeping `{row['keep_candidate']}` (corr={row['correlation']:.4f})."
        )
    return lines


def render_candidates_markdown(payload: Dict[str, object]) -> List[str]:
    baseline = payload["legacy_baseline_metrics"]
    lines = [
        "# Candidate Feature Report",
        "",
        "## Legacy Baseline",
        "",
        f"- ROC-AUC: `{baseline.get('roc_auc', 0.0):.6f}`",
        f"- F1: `{baseline.get('f1', 0.0):.6f}`",
        f"- Profit Factor: `{baseline.get('profit_factor', 0.0):.6f}`",
        "",
        "## Individual Tests",
        "",
        "| Feature | Category | Status | dROC-AUC | dF1 | dProfit Factor |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in payload["candidate_results"]:
        lines.append(
            f"| {row['feature']} | {row['category']} | {row['status']} | "
            f"{row['delta_roc_auc']:.6f} | {row['delta_f1']:.6f} | {row['delta_profit_factor']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Greedy Combined Selection",
            "",
            f"- Selected: `{payload['greedy_selected_features']}`",
            f"- ROC-AUC: `{payload['greedy_selected_metrics'].get('roc_auc', 0.0):.6f}`",
            f"- F1: `{payload['greedy_selected_metrics'].get('f1', 0.0):.6f}`",
            f"- Profit Factor: `{payload['greedy_selected_metrics'].get('profit_factor', 0.0):.6f}`",
            "",
            "## Recommendations",
            "",
        ]
    )
    for recommendation in payload.get("recommendations", []):
        lines.append(f"- {recommendation}")
    return lines


def main() -> int:
    summary = run_feature_audit()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
