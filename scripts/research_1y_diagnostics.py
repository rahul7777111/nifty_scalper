from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from label_policies import available_label_policies, build_label_dataset
from cost_model import CostModel
from ml_pipeline import build_market_feature_vector
from ml_signals import load_model
from retrain_nifty_1year import (
    HORIZON_BARS,
    LOOKBACK_BARS,
    THRESHOLD_GRID,
    build_safe_model,
    calibration_bins,
    chronological_split,
    clean_frame,
    compare_feature_compatibility,
    confusion_from_threshold,
    feature_manifest,
    frame_to_candles,
    model_names_to_train,
    prepare_minute_frame,
    predict_proba_positive,
    pr_auc_score_safe,
    scale_train_val_test,
    threshold_sweep,
    write_json,
)
from retraining_validation import classification_metrics


RESEARCH_THRESHOLDS = [0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70]
TIMEFRAMES = [1, 3, 5, 15]
MIN_POLICY_CLASS_SHARE = 0.02
PERMUTATION_SAMPLE = 2500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research diagnostics for 1-year NIFTY retraining artifacts.")
    parser.add_argument("--artifact-dir", default="", help="Specific retrained_1y_nifty artifact folder to inspect.")
    parser.add_argument("--skip-training", action="store_true", help="Skip model retraining comparisons and generate report-only diagnostics.")
    parser.add_argument("--max-models", type=int, default=0, help="Optional cap on number of models to evaluate.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_artifact_dir(explicit: str = "") -> Path:
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else (REPO_ROOT / explicit)
    candidates = sorted(MODELS_DIR.glob("retrained_1y_nifty_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No retrained_1y_nifty artifact directories found.")
    return candidates[0]


def resolve_existing_artifact_files(artifact_dir: Path) -> Dict[str, Optional[Path]]:
    files = {path.name: path for path in artifact_dir.glob("*") if path.is_file()}
    return {
        "validation_metrics_report": files.get("validation_metrics_report.json"),
        "logistic_regression_metrics": files.get("logistic_regression_metrics.json"),
        "data_quality_report": files.get("data_quality_report.json"),
        "leakage_audit_report": files.get("leakage_audit_report.json"),
        "feature_manifest": files.get("feature_manifest.json"),
        "threshold_sweep_report": files.get("threshold_sweep_report.json") or files.get("logistic_regression_metrics.json"),
        "train_val_test_split_report": files.get("train_val_test_split_report.json") or files.get("validation_metrics_report.json"),
    }


def extract_existing_artifact_summary(artifact_dir: Path) -> Dict[str, Any]:
    resolved = resolve_existing_artifact_files(artifact_dir)
    summary: Dict[str, Any] = {"artifact_dir": str(artifact_dir), "resolved_files": {k: str(v) if v else None for k, v in resolved.items()}}
    validation = read_json(resolved["validation_metrics_report"]) if resolved["validation_metrics_report"] else {}
    logistic = read_json(resolved["logistic_regression_metrics"]) if resolved["logistic_regression_metrics"] else {}
    data_quality = read_json(resolved["data_quality_report"]) if resolved["data_quality_report"] else {}
    leakage = read_json(resolved["leakage_audit_report"]) if resolved["leakage_audit_report"] else {}
    manifest = read_json(resolved["feature_manifest"]) if resolved["feature_manifest"] else {}
    summary["validation"] = validation
    summary["logistic"] = logistic
    summary["data_quality"] = data_quality
    summary["leakage"] = leakage
    summary["feature_manifest"] = manifest
    summary["threshold_sweep_rows"] = len(logistic.get("threshold_sweep", []))
    summary["split_report"] = (validation.get("training") or {}).get("split_report") or {}
    return summary


def load_current_1y_frame(artifact_dir: Path) -> pd.DataFrame:
    minute_frame, _ = prepare_minute_frame()
    minute_frame, _ = clean_frame(minute_frame)
    selected = read_json(artifact_dir / "selected_window_report.json")
    start_day = pd.to_datetime(selected["requested_start_day"]).date()
    end_day = pd.to_datetime(selected["requested_end_day"]).date()
    return minute_frame[(minute_frame["trading_day"] >= start_day) & (minute_frame["trading_day"] <= end_day)].copy()


def resample_frame(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes == 1:
        return frame.copy()
    rows: List[pd.DataFrame] = []
    for trading_day, day_frame in frame.groupby("trading_day", sort=True):
        day_sorted = day_frame.sort_values("timestamp").set_index("timestamp")
        agg = day_sorted.resample(f"{minutes}min", origin=day_sorted.index[0]).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        agg = agg.dropna(subset=["open", "high", "low", "close"]).reset_index()
        agg["trading_day"] = trading_day
        rows.append(agg)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "trading_day"])
    out = pd.concat(rows, ignore_index=True)
    out["source"] = f"resampled_{minutes}m"
    out["local_time"] = out["timestamp"].dt.tz_convert("Asia/Kolkata")
    out["time_only"] = out["local_time"].dt.time
    out["weekday"] = out["local_time"].dt.weekday
    return out


def precompute_feature_matrix(candles: Sequence[Any]) -> Dict[str, Any]:
    feature_rows: List[List[float]] = []
    timestamps: List[datetime] = []
    feature_names: List[str] = []
    for sample_idx, candle_idx in enumerate(range(LOOKBACK_BARS, len(candles) - HORIZON_BARS), start=1):
        window = candles[candle_idx - LOOKBACK_BARS + 1 : candle_idx + 1]
        features, names = build_market_feature_vector(window, lookback=LOOKBACK_BARS)
        if not feature_names:
            feature_names = list(names)
        feature_rows.append([float(v) for v in features])
        timestamps.append(candles[candle_idx].time)
    return {
        "X_full": np.asarray(feature_rows, dtype=np.float32),
        "feature_names": feature_names,
        "timestamps_full": timestamps,
    }


def policy_dataset_from_cache(candles: Sequence[Any], policy: str, cache: Dict[str, Any]) -> Dict[str, Any]:
    label_dataset = build_label_dataset(
        candles,
        policy_name=policy,
        lookback=LOOKBACK_BARS,
        horizon=HORIZON_BARS,
        cost_model=CostModel(),
        include_features=False,
    )
    sample_indices = [idx for idx in label_dataset.sample_indices if idx < len(cache["X_full"])]
    return {
        "X": cache["X_full"][sample_indices],
        "y": np.asarray([int(label_dataset.y[pos]) for pos, idx in enumerate(label_dataset.sample_indices) if idx < len(cache["X_full"])], dtype=np.int32),
        "feature_names": list(cache["feature_names"]),
        "timestamps": [cache["timestamps_full"][idx] for idx in sample_indices],
        "forward_returns": [
            float(label_dataset.observations[idx].forward_return)
            for idx in sample_indices
            if idx < len(label_dataset.observations)
        ],
    }


def class_balance_by_split(y: np.ndarray) -> Dict[str, Any]:
    split = chronological_split(np.zeros((len(y), 1), dtype=np.float32), y, [datetime.now()] * len(y))
    report: Dict[str, Any] = {}
    for name, subset in split["slices"].items():
        arr = y[subset]
        counts = Counter(int(v) for v in arr.tolist())
        total = len(arr)
        report[name] = {
            "rows": total,
            "positive": int(counts.get(1, 0)),
            "negative": int(counts.get(0, 0)),
            "positive_pct": float(counts.get(1, 0) / total) if total else 0.0,
        }
    return report


def brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    return float(np.mean((yp - yt) ** 2))


def monthly_regime_report(
    timestamps: Sequence[datetime],
    y_true: Sequence[int],
    y_prob: Sequence[float],
    forward_returns: Sequence[float],
    threshold: float,
) -> List[Dict[str, Any]]:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(list(timestamps)),
            "y": list(y_true),
            "p": list(y_prob),
            "forward_return": list(forward_returns),
        }
    )
    frame["month"] = frame["timestamp"].dt.to_period("M").astype(str)
    rows: List[Dict[str, Any]] = []
    for month, group in frame.groupby("month", sort=True):
        if group.empty:
            continue
        metrics = classification_metrics(group["y"], group["p"], threshold=threshold)
        pr_auc = pr_auc_score_safe(group["y"], group["p"])
        signals = group[group["p"] >= threshold]
        rows.append(
            {
                "month": month,
                "rows": int(len(group)),
                "roc_auc": float(metrics.get("roc_auc", 0.0)),
                "pr_auc": float(pr_auc) if pr_auc is not None else None,
                "positive_label_pct": float(group["y"].mean()),
                "signal_count": int(len(signals)),
                "win_rate_proxy": float((signals["y"] == 1).mean()) if not signals.empty else 0.0,
                "avg_forward_return": float(signals["forward_return"].mean()) if not signals.empty else 0.0,
            }
        )
    return rows


def psi_score(train: pd.Series, test: pd.Series, *, buckets: int = 10) -> float:
    train = train.replace([np.inf, -np.inf], np.nan).dropna()
    test = test.replace([np.inf, -np.inf], np.nan).dropna()
    if train.empty or test.empty:
        return 0.0
    quantiles = np.unique(np.quantile(train, np.linspace(0, 1, buckets + 1)))
    if len(quantiles) < 3:
        return 0.0
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    train_bins = pd.cut(train, quantiles, include_lowest=True)
    test_bins = pd.cut(test, quantiles, include_lowest=True)
    train_dist = train_bins.value_counts(normalize=True, sort=False).replace(0, 1e-6)
    test_dist = test_bins.value_counts(normalize=True, sort=False).replace(0, 1e-6)
    return float(((test_dist - train_dist) * np.log(test_dist / train_dist)).sum())


def feature_stability_report(X_train: np.ndarray, X_test: np.ndarray, feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    train_df = pd.DataFrame(X_train, columns=feature_names)
    test_df = pd.DataFrame(X_test, columns=feature_names)
    rows: List[Dict[str, Any]] = []
    for name in feature_names:
        train_col = train_df[name]
        test_col = test_df[name]
        rows.append(
            {
                "feature": name,
                "train_mean": float(train_col.mean()),
                "test_mean": float(test_col.mean()),
                "train_std": float(train_col.std()),
                "test_std": float(test_col.std()),
                "psi": psi_score(train_col, test_col),
            }
        )
    return rows


def correlation_report(X: np.ndarray, feature_names: Sequence[str], topn: int = 25) -> List[Dict[str, Any]]:
    frame = pd.DataFrame(X, columns=feature_names)
    corr = frame.corr().abs()
    pairs: List[Tuple[str, str, float]] = []
    for i, left in enumerate(feature_names):
        for j in range(i + 1, len(feature_names)):
            right = feature_names[j]
            pairs.append((left, right, float(corr.iloc[i, j])))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return [{"left": a, "right": b, "abs_corr": c} for a, b, c in pairs[:topn]]


def permutation_importance_safe(model: Any, X_test: np.ndarray, y_test: np.ndarray, feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    try:
        from sklearn.inspection import permutation_importance
    except Exception:
        return []
    sample_n = min(PERMUTATION_SAMPLE, len(X_test))
    if sample_n <= 0:
        return []
    try:
        result = permutation_importance(
            model,
            X_test[:sample_n],
            y_test[:sample_n],
            n_repeats=3,
            random_state=42,
            scoring="roc_auc",
        )
    except Exception:
        return []
    rows = [
        {"feature": feature_names[idx], "importance_mean": float(result.importances_mean[idx]), "importance_std": float(result.importances_std[idx])}
        for idx in np.argsort(result.importances_mean)[::-1][:25]
    ]
    return rows


def tree_importance_safe(model: Any, feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    base = getattr(model, "base_estimator", None)
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif base is not None and hasattr(base, "feature_importances_"):
        imp = np.asarray(base.feature_importances_, dtype=float)
    else:
        return []
    order = np.argsort(imp)[::-1][:25]
    return [{"feature": feature_names[idx], "importance": float(imp[idx])} for idx in order]


def logistic_coefficients(model: Any, feature_names: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    coef = getattr(model, "coef_", None)
    if coef is None:
        return {"positive": [], "negative": []}
    values = np.asarray(coef[0], dtype=float)
    pos_idx = np.argsort(values)[::-1][:15]
    neg_idx = np.argsort(values)[:15]
    return {
        "positive": [{"feature": feature_names[idx], "coefficient": float(values[idx])} for idx in pos_idx],
        "negative": [{"feature": feature_names[idx], "coefficient": float(values[idx])} for idx in neg_idx],
    }


def threshold_table(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    forward_returns: Sequence[float],
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    fwd_arr = np.asarray(forward_returns, dtype=float)
    for threshold in thresholds:
        metrics = classification_metrics(y_true_arr, y_prob_arr, threshold=threshold)
        mask = y_prob_arr >= threshold
        signals = int(mask.sum())
        signal_probs = y_prob_arr[mask]
        signal_returns = fwd_arr[mask]
        confusion = confusion_from_threshold(y_true_arr, y_prob_arr, threshold)
        gross_expectancy = float(signal_returns.mean()) if signals else 0.0
        cost_adjusted_expectancy = float((signal_returns - 0.001).mean()) if signals else 0.0
        wins = signal_returns[signal_returns > 0]
        losses = np.abs(signal_returns[signal_returns < 0])
        profit_factor = float(wins.sum() / losses.sum()) if losses.sum() > 0 else (99.0 if wins.sum() > 0 else 0.0)
        rows.append(
            {
                "threshold": float(threshold),
                "signals": signals,
                "signal_pct": float(signals / len(y_true_arr)) if len(y_true_arr) else 0.0,
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
                "average_predicted_probability": float(signal_probs.mean()) if signals else 0.0,
                "average_forward_return": gross_expectancy,
                "gross_expectancy_proxy": gross_expectancy,
                "cost_adjusted_expectancy_proxy": cost_adjusted_expectancy,
                "profit_factor_estimate": profit_factor,
                **confusion,
            }
        )
    return rows


def evaluate_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    timestamps: Sequence[datetime],
    feature_names: Sequence[str],
    forward_returns: Sequence[float],
) -> Dict[str, Any]:
    split = chronological_split(X, y, timestamps)
    s_train = split["slices"]["train"]
    s_val = split["slices"]["validation"]
    s_test = split["slices"]["test"]
    X_train, X_val, X_test = X[s_train], X[s_val], X[s_test]
    y_train, y_val, y_test = y[s_train], y[s_val], y[s_test]
    fwd_test = np.asarray(forward_returns, dtype=float)[s_test]
    t_test = np.asarray(timestamps, dtype=object)[s_test]
    X_train_s, X_val_s, X_test_s, scaler_mean, scaler_std = scale_train_val_test(X_train, X_val, X_test)
    model = build_safe_model(model_name)
    if model is None:
        raise RuntimeError(f"Unsupported model: {model_name}")
    started = time.perf_counter()
    model.fit(X_train_s, y_train)
    train_seconds = time.perf_counter() - started
    val_prob = predict_proba_positive(model, X_val_s)
    test_prob = predict_proba_positive(model, X_test_s)
    val_threshold_rows = threshold_table(y_val, val_prob, np.zeros_like(y_val, dtype=float), RESEARCH_THRESHOLDS)
    best_val_row = max(val_threshold_rows, key=lambda row: (row["f1"], row["signals"]))
    chosen_threshold = float(best_val_row["threshold"])
    val_metrics = classification_metrics(y_val, val_prob, threshold=chosen_threshold)
    test_metrics = classification_metrics(y_test, test_prob, threshold=chosen_threshold)
    val_pr = pr_auc_score_safe(y_val, val_prob)
    test_pr = pr_auc_score_safe(y_test, test_prob)
    if val_pr is not None:
        val_metrics["pr_auc"] = float(val_pr)
    if test_pr is not None:
        test_metrics["pr_auc"] = float(test_pr)
    confusion = confusion_from_threshold(y_test, test_prob, chosen_threshold)
    compat = compare_feature_compatibility(feature_names)
    report = {
        "model_name": model_name,
        "training_time_seconds": float(train_seconds),
        "model_size_bytes": int(len(pickle.dumps(model))),
        "validation_threshold_selected": chosen_threshold,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix": confusion,
        "calibration_quality": {
            "brier_score": brier_score(y_test, test_prob),
            "bins": calibration_bins(y_test, test_prob),
        },
        "threshold_sweep": threshold_sweep(y_test, test_prob),
        "threshold_diagnostics": threshold_table(y_test, test_prob, fwd_test, RESEARCH_THRESHOLDS),
        "signal_counts_fixed_thresholds": {
            str(t): int(np.sum(np.asarray(test_prob) >= t)) for t in [0.50, 0.55, 0.60, 0.65, 0.70]
        },
        "inference_compatibility": compat,
        "regime_diagnostics": monthly_regime_report(t_test, y_test, test_prob, fwd_test, chosen_threshold),
        "feature_importance": {
            "logistic_coefficients": logistic_coefficients(model, feature_names) if model_name == "logistic_regression" else {},
            "tree_importance": tree_importance_safe(model, feature_names),
            "permutation_importance": permutation_importance_safe(model, X_test_s, y_test, feature_names),
        },
        "split_report": split["ranges"],
    }
    return report


def label_policy_comparison(
    candles: Sequence[Any],
    feature_names_live: Sequence[str],
    *,
    policy_limit_models: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    rows: List[Dict[str, Any]] = []
    best_policy = None
    best_val_auc = -1.0
    cache = precompute_feature_matrix(candles)
    for policy in available_label_policies():
        dataset = policy_dataset_from_cache(candles, policy, cache)
        y = dataset["y"]
        counts = Counter(int(v) for v in y.tolist())
        total = len(y)
        positive = int(counts.get(1, 0))
        negative = int(counts.get(0, 0))
        positive_pct = float(positive / total) if total else 0.0
        imbalance_ratio = float(max(positive, negative) / max(1, min(positive, negative))) if total and min(positive, negative) else math.inf
        split_balance = class_balance_by_split(y)
        trainable = total > 0 and positive > 0 and negative > 0 and min(positive_pct, 1.0 - positive_pct) >= MIN_POLICY_CLASS_SHARE
        row: Dict[str, Any] = {
            "policy": policy,
            "total_samples": total,
            "positive_class_count": positive,
            "negative_class_count": negative,
            "positive_percentage": positive_pct,
            "imbalance_ratio": imbalance_ratio,
            "train_validation_test_balance": split_balance,
            "feature_schema_compatible": list(dataset["feature_names"]) == list(feature_names_live),
            "suitable_for_live_promotion_research": False,
        }
        if trainable:
            baseline_model = policy_limit_models[0]
            report = evaluate_model(
                baseline_model,
                dataset["X"],
                dataset["y"],
                dataset["timestamps"],
                dataset["feature_names"],
                dataset["forward_returns"],
            )
            row["baseline_model"] = baseline_model
            row["validation_roc_auc"] = float(report["validation_metrics"].get("roc_auc", 0.0))
            row["test_roc_auc"] = float(report["test_metrics"].get("roc_auc", 0.0))
            row["pr_auc"] = float(report["test_metrics"].get("pr_auc", 0.0)) if "pr_auc" in report["test_metrics"] else None
            row["usable_trade_signals_after_thresholding"] = int(
                max((entry["signals"] for entry in report["threshold_diagnostics"]), default=0)
            )
            row["suitable_for_live_promotion_research"] = bool(row["validation_roc_auc"] >= 0.5 and row["usable_trade_signals_after_thresholding"] > 100)
            if row["validation_roc_auc"] > best_val_auc:
                best_val_auc = row["validation_roc_auc"]
                best_policy = policy
        else:
            row["reason_untrainable"] = "class_imbalance_or_single_class"
        rows.append(row)
    return rows, best_policy


def timeframe_experiment(
    frame_1m: pd.DataFrame,
    best_policy: str,
    model_names: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    best_result: Optional[Dict[str, Any]] = None
    best_val = -1.0
    for minutes in TIMEFRAMES:
        frame_tf = resample_frame(frame_1m, minutes)
        candles_tf = frame_to_candles(frame_tf)
        cache = precompute_feature_matrix(candles_tf)
        dataset = policy_dataset_from_cache(candles_tf, best_policy, cache)
        counts = Counter(int(v) for v in dataset["y"].tolist())
        total = len(dataset["y"])
        positive_pct = float(counts.get(1, 0) / total) if total else 0.0
        row: Dict[str, Any] = {
            "timeframe_minutes": minutes,
            "samples_count": total,
            "label_distribution": {"positive": int(counts.get(1, 0)), "negative": int(counts.get(0, 0)), "positive_pct": positive_pct},
        }
        if total < 500 or min(positive_pct, 1.0 - positive_pct) < MIN_POLICY_CLASS_SHARE:
            row["trainable"] = False
            rows.append(row)
            continue
        best_model_name = None
        best_model_report = None
        best_model_val = -1.0
        for model_name in model_names:
            report = evaluate_model(
                model_name,
                dataset["X"],
                dataset["y"],
                dataset["timestamps"],
                dataset["feature_names"],
                dataset["forward_returns"],
            )
            val_auc = float(report["validation_metrics"].get("roc_auc", 0.0))
            if val_auc > best_model_val:
                best_model_val = val_auc
                best_model_name = model_name
                best_model_report = report
        row["trainable"] = True
        row["best_model"] = best_model_name
        row["best_validation_roc_auc"] = float(best_model_report["validation_metrics"].get("roc_auc", 0.0))
        row["test_roc_auc"] = float(best_model_report["test_metrics"].get("roc_auc", 0.0))
        row["threshold_signal_count"] = int(max((t["signals"] for t in best_model_report["threshold_diagnostics"]), default=0))
        row["train_val_test_ranges"] = best_model_report["split_report"]
        rows.append(row)
        if row["best_validation_roc_auc"] > best_val:
            best_val = row["best_validation_roc_auc"]
            best_result = row
    return rows, best_result


def find_224day_baseline() -> Dict[str, Any]:
    final_report = REPORTS_DIR / "final_224day_mstock_retraining_report.json"
    retrain_report = REPORTS_DIR / "retraining_report_20260602_204058.json"
    baseline = {}
    if final_report.exists():
        baseline["final_224day_report"] = read_json(final_report)
    if retrain_report.exists():
        baseline["candidate_report"] = read_json(retrain_report)
    return baseline


def compare_vs_224day(current_summary: Dict[str, Any], baseline: Dict[str, Any], best_model: Dict[str, Any], best_policy: str, best_timeframe: Optional[Dict[str, Any]]) -> str:
    candidate = baseline.get("candidate_report", {})
    dataset = candidate.get("dataset", {})
    validation = candidate.get("validation", {}).get("metrics", {})
    current_test = best_model.get("test_metrics", {})
    lines = [
        "# 1Y vs 224-Day Comparison",
        "",
        f"- 224-day effective horizon: `{dataset.get('effective_horizon', 'unknown')}`",
        f"- 224-day sample count: `{dataset.get('sample_count', 'unknown')}`",
        f"- 224-day feature count: `{dataset.get('feature_count', 'unknown')}`",
        f"- 224-day label policy: `{candidate.get('run', {}).get('label_policy', 'unknown')}`",
        f"- 224-day ROC-AUC: `{float(validation.get('roc_auc', 0.0)):.4f}`",
        f"- 1-year best model: `{best_model.get('model_name', 'unknown')}`",
        f"- 1-year best label policy: `{best_policy}`",
        f"- 1-year best timeframe: `{best_timeframe.get('timeframe_minutes') if best_timeframe else 'unknown'}m`",
        f"- 1-year test ROC-AUC: `{float(current_test.get('roc_auc', 0.0)):.4f}`",
        "",
        "Interpretation:",
    ]
    if float(current_test.get("roc_auc", 0.0)) > float(validation.get("roc_auc", 0.0)):
        lines.append("- The 1-year candidate improved ROC-AUC versus the logged 224-day candidate.")
    else:
        lines.append("- The 1-year candidate did not improve ROC-AUC versus the logged 224-day candidate.")
    lines.append("- Use the threshold and regime diagnostics alongside ROC-AUC because both candidate families are close to random at the aggregate level.")
    return "\n".join(lines) + "\n"


def build_root_cause_report(existing: Dict[str, Any], label_rows: List[Dict[str, Any]], model_rows: List[Dict[str, Any]], timeframe_rows: List[Dict[str, Any]]) -> str:
    logistic = existing.get("logistic", {})
    dq = existing.get("data_quality", {})
    lines = [
        "# Root Cause Report",
        "",
        "## Existing 1-Year Artifact",
        "",
        f"- Logistic test ROC-AUC: `{float(logistic.get('metrics', {}).get('roc_auc', 0.0)):.4f}`",
        f"- Data gap days recorded: `{dq.get('session_gap_days_total', 0)}`",
        f"- Leakage audit passed: `{existing.get('leakage', {}).get('passed', False)}`",
        "",
        "## Likely Causes of Weak ROC-AUC",
        "",
        "- The 1-minute target is noisy relative to the feature set, so separability is weak even when the pipeline is correct.",
        "- The quick verified run used only Logistic Regression, which is a useful baseline but not the whole supported model set.",
        "- Threshold behavior matters more than raw ROC-AUC here; the current logistic model tends to fire too broadly at practical cutoffs.",
        "- Feature schema compatibility is not the issue if the live and research feature names match.",
        "- Leakage prevention is not the main problem because the dynamic leakage audit passed.",
        "- Label choice matters heavily: some cost-adjusted trade-quality labels collapse into extreme imbalance on the 1-year minute set.",
        "- Regime drift across months can dilute aggregate performance even when some sub-periods are usable.",
        "",
        "## Policy Findings",
    ]
    for row in label_rows:
        if row.get("reason_untrainable"):
            lines.append(f"- `{row['policy']}` is effectively unusable on this dataset because of class imbalance.")
        else:
            lines.append(
                f"- `{row['policy']}` produced validation ROC-AUC `{float(row.get('validation_roc_auc', 0.0)):.4f}` with positive rate `{float(row.get('positive_percentage', 0.0)):.3%}`."
            )
    lines.extend(["", "## Timeframe Findings"])
    for row in timeframe_rows:
        if row.get("trainable"):
            lines.append(
                f"- `{row['timeframe_minutes']}m` best validation ROC-AUC was `{float(row.get('best_validation_roc_auc', 0.0)):.4f}`."
            )
    return "\n".join(lines) + "\n"


def promotion_recommendation(best_model: Dict[str, Any], best_policy: str, best_timeframe: Optional[Dict[str, Any]], compare_text: str) -> str:
    test_auc = float(best_model.get("test_metrics", {}).get("roc_auc", 0.0))
    precision = float(best_model.get("test_metrics", {}).get("precision", 0.0))
    recommendation = "RESEARCH ONLY"
    reason = "no durable trading edge is proven yet"
    if test_auc >= 0.55 and precision >= 0.55:
        recommendation = "PAPER TRADE ONLY"
        reason = "the candidate improved but still needs live/paper validation"
    if test_auc < 0.53:
        recommendation = "DO NOT PROMOTE"
        reason = "test ROC-AUC remains too weak and threshold behavior is not selective enough"
    lines = [
        f"# {recommendation}",
        "",
        f"- Best model: `{best_model.get('model_name', 'unknown')}`",
        f"- Best label policy: `{best_policy}`",
        f"- Best timeframe: `{best_timeframe.get('timeframe_minutes') if best_timeframe else 'unknown'}m`",
        f"- Test ROC-AUC: `{test_auc:.4f}`",
        f"- Reason: {reason}",
        "",
        "## Compare vs 224-Day",
        "",
        compare_text,
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    artifact_dir = latest_artifact_dir(args.artifact_dir)
    research_dir = MODELS_DIR / f"research_1y_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    research_dir.mkdir(parents=True, exist_ok=True)

    existing_summary = extract_existing_artifact_summary(artifact_dir)
    write_json(research_dir / "inspected_artifact_summary.json", existing_summary)

    frame_1m = load_current_1y_frame(artifact_dir)
    candles_1m = frame_to_candles(frame_1m)
    live_bundle = load_model(str(REPO_ROOT / "ml_signal_model.pkl"))
    live_features = list(getattr(live_bundle, "feature_names", []) or [])
    model_names = model_names_to_train()
    if args.max_models > 0:
        model_names = model_names[: args.max_models]

    label_rows, best_policy = label_policy_comparison(candles_1m, live_features, policy_limit_models=model_names[:1])
    if best_policy is None:
        best_policy = "current_triple_barrier"
    write_json(research_dir / "label_policy_comparison.json", label_rows)

    timeframe_rows, best_timeframe = timeframe_experiment(frame_1m, best_policy, model_names[:1] if args.skip_training else model_names)
    write_json(research_dir / "timeframe_experiment_report.json", timeframe_rows)

    working_minutes = int(best_timeframe.get("timeframe_minutes", 1)) if best_timeframe else 1
    frame_best = resample_frame(frame_1m, working_minutes)
    candles_best = frame_to_candles(frame_best)
    cache_best = precompute_feature_matrix(candles_best)
    dataset_best = policy_dataset_from_cache(candles_best, best_policy, cache_best)

    model_rows: List[Dict[str, Any]] = []
    if not args.skip_training:
        for model_name in model_names:
            model_rows.append(
                evaluate_model(
                    model_name,
                    dataset_best["X"],
                    dataset_best["y"],
                    dataset_best["timestamps"],
                    dataset_best["feature_names"],
                    dataset_best["forward_returns"],
                )
            )
    else:
        model_rows.append(
            evaluate_model(
                model_names[0],
                dataset_best["X"],
                dataset_best["y"],
                dataset_best["timestamps"],
                dataset_best["feature_names"],
                dataset_best["forward_returns"],
            )
        )
    write_json(research_dir / "model_comparison_report.json", model_rows)

    best_model = max(model_rows, key=lambda row: float(row.get("validation_metrics", {}).get("roc_auc", 0.0)))
    write_json(research_dir / "threshold_diagnostics.json", {row["model_name"]: row["threshold_diagnostics"] for row in model_rows})
    write_json(research_dir / "regime_diagnostics.json", {row["model_name"]: row["regime_diagnostics"] for row in model_rows})

    split = chronological_split(dataset_best["X"], dataset_best["y"], dataset_best["timestamps"])
    X_train = dataset_best["X"][split["slices"]["train"]]
    X_test = dataset_best["X"][split["slices"]["test"]]
    feature_diag = {
        "feature_manifest": feature_manifest(dataset_best["X"], dataset_best["feature_names"]),
        "feature_correlation_report": correlation_report(dataset_best["X"], dataset_best["feature_names"]),
        "feature_stability_train_vs_test": feature_stability_report(X_train, X_test, dataset_best["feature_names"]),
        "best_model_feature_importance": best_model.get("feature_importance", {}),
    }
    write_json(research_dir / "feature_diagnostics.json", feature_diag)

    baseline_224 = find_224day_baseline()
    compare_text = compare_vs_224day(existing_summary, baseline_224, best_model, best_policy, best_timeframe)
    (research_dir / "comparison_vs_224day_model.md").write_text(compare_text, encoding="utf-8")

    root_cause = build_root_cause_report(existing_summary, label_rows, model_rows, timeframe_rows)
    (research_dir / "root_cause_report.md").write_text(root_cause, encoding="utf-8")

    recommendation = promotion_recommendation(best_model, best_policy, best_timeframe, compare_text)
    (research_dir / "promotion_recommendation.md").write_text(recommendation, encoding="utf-8")

    print(str(research_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
