from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
MODELS_DIR = REPO_ROOT / "models"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_1y_diagnostics import (
    brier_score,
    calibration_bins,
    class_balance_by_split,
    compare_feature_compatibility,
    confusion_from_threshold,
    extract_existing_artifact_summary,
    feature_stability_report,
    latest_artifact_dir,
    load_current_1y_frame,
    monthly_regime_report,
    policy_dataset_from_cache,
    precompute_feature_matrix,
    pr_auc_score_safe,
)
from retrain_nifty_1year import (
    build_safe_model,
    chronological_split,
    frame_to_candles,
    scale_train_val_test,
    write_json,
)
from retraining_validation import classification_metrics


BEST_MODEL = "logistic_regression"
BEST_POLICY = "trade_quality_ternary"
BEST_TIMEFRAME = 1
THRESHOLD_GRID = [0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75, 0.775, 0.80]
MIN_PRECISION = 0.30
FALLBACK_PRECISION = 0.25
MIN_SIGNALS = 50
FALLBACK_SIGNALS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research-only signal quality improvement layer.")
    parser.add_argument("--artifact-dir", default="", help="Specific retrained_1y_nifty artifact directory.")
    return parser.parse_args()


def ece_score(y_true: Sequence[int], y_prob: Sequence[float], bins: int = 10) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(yt)
    ece = 0.0
    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        mask = (yp >= left) & (yp <= right if i == bins - 1 else yp < right)
        if not np.any(mask):
            continue
        acc = float(np.mean(yt[mask]))
        conf = float(np.mean(yp[mask]))
        ece += abs(acc - conf) * (float(np.sum(mask)) / total)
    return float(ece)


def select_threshold_validation_only(
    y_val: Sequence[int],
    p_val: Sequence[float],
    thresholds: Sequence[float],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        metrics = classification_metrics(y_val, p_val, threshold=threshold)
        confusion = confusion_from_threshold(y_val, p_val, threshold)
        signals = int(sum(1 for p in p_val if float(p) >= threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "signals": signals,
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
                **confusion,
            }
        )

    def pick(min_precision: float, min_signals: int) -> Optional[Dict[str, Any]]:
        candidates = [row for row in rows if row["precision"] >= min_precision and row["signals"] >= min_signals]
        if not candidates:
            return None
        return max(candidates, key=lambda row: (row["precision"], row["f1"], row["signals"]))

    selected = pick(MIN_PRECISION, MIN_SIGNALS)
    selection_rule = "primary"
    if selected is None:
        selected = pick(FALLBACK_PRECISION, FALLBACK_SIGNALS)
        selection_rule = "fallback"
    if selected is None:
        selected = max(rows, key=lambda row: (row["f1"], row["signals"]))
        selection_rule = "f1_fallback"
    return {"selected": selected, "selection_rule": selection_rule, "grid": rows}


def apply_threshold_report(
    y_true: Sequence[int],
    p_true: Sequence[float],
    forward_returns: Sequence[float],
    threshold: float,
) -> Dict[str, Any]:
    metrics = classification_metrics(y_true, p_true, threshold=threshold)
    confusion = confusion_from_threshold(y_true, p_true, threshold)
    mask = np.asarray(p_true) >= float(threshold)
    signal_count = int(mask.sum())
    signal_probs = np.asarray(p_true)[mask]
    signal_returns = np.asarray(forward_returns, dtype=float)[mask]
    return {
        "threshold": float(threshold),
        "signal_count": signal_count,
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "f1": float(metrics.get("f1", 0.0)),
        "average_predicted_probability": float(signal_probs.mean()) if signal_count else 0.0,
        "average_forward_return": float(signal_returns.mean()) if signal_count else 0.0,
        "gross_expectancy_proxy": float(signal_returns.mean()) if signal_count else 0.0,
        "cost_adjusted_expectancy_proxy": float((signal_returns - 0.001).mean()) if signal_count else 0.0,
        **confusion,
    }


def build_base_dataset(artifact_dir: Path) -> Dict[str, Any]:
    frame = load_current_1y_frame(artifact_dir)
    candles = frame_to_candles(frame)
    cache = precompute_feature_matrix(candles)
    dataset = policy_dataset_from_cache(candles, BEST_POLICY, cache)
    return {"frame": frame, "candles": candles, "cache": cache, "dataset": dataset}


def add_context_columns(dataset: Dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    timestamps = pd.to_datetime(dataset["timestamps"])
    feature_df = pd.DataFrame(dataset["X"], columns=dataset["feature_names"])
    feature_df["timestamp"] = timestamps
    feature_df["forward_return"] = np.asarray(dataset["forward_returns"], dtype=float)
    feature_df["y"] = np.asarray(dataset["y"], dtype=int)
    feature_df["month"] = feature_df["timestamp"].dt.to_period("M").astype(str)
    feature_df["weekday"] = feature_df["timestamp"].dt.weekday
    minutes = feature_df["timestamp"].dt.hour * 60 + feature_df["timestamp"].dt.minute
    feature_df["session_bucket"] = np.where(
        minutes < (10 * 60),
        "opening",
        np.where(minutes >= (14 * 60 + 45), "closing", "mid"),
    )
    train_end = int(len(feature_df) * 0.70)
    train_slice = feature_df.iloc[:train_end]
    for col in ("atr_pct", "realized_vol_30", "ret_std"):
        if col in feature_df.columns:
            q1 = float(train_slice[col].quantile(0.33))
            q2 = float(train_slice[col].quantile(0.66))
            feature_df[f"{col}_bucket"] = pd.cut(
                feature_df[col],
                bins=[-np.inf, q1, q2, np.inf],
                labels=["low", "mid", "high"],
                include_lowest=True,
            ).astype(str)
    return feature_df


def fit_base_logistic(dataset: Dict[str, Any]) -> Dict[str, Any]:
    X = dataset["X"]
    y = dataset["y"]
    timestamps = dataset["timestamps"]
    split = chronological_split(X, y, timestamps)
    s_train = split["slices"]["train"]
    s_val = split["slices"]["validation"]
    s_test = split["slices"]["test"]
    X_train, X_val, X_test = X[s_train], X[s_val], X[s_test]
    y_train, y_val, y_test = y[s_train], y[s_val], y[s_test]
    fwd_val = np.asarray(dataset["forward_returns"], dtype=float)[s_val]
    fwd_test = np.asarray(dataset["forward_returns"], dtype=float)[s_test]
    ts_val = np.asarray(timestamps, dtype=object)[s_val]
    ts_test = np.asarray(timestamps, dtype=object)[s_test]
    X_train_s, X_val_s, X_test_s, scaler_mean, scaler_std = scale_train_val_test(X_train, X_val, X_test)
    model = build_safe_model(BEST_MODEL)
    model.fit(X_train_s, y_train)
    val_prob = np.asarray(model.predict_proba(X_val_s))[:, 1]
    test_prob = np.asarray(model.predict_proba(X_test_s))[:, 1]
    return {
        "model": model,
        "split": split,
        "X_train_s": X_train_s,
        "X_val_s": X_val_s,
        "X_test_s": X_test_s,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "forward_val": fwd_val,
        "forward_test": fwd_test,
        "ts_val": ts_val,
        "ts_test": ts_test,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
    }


def regime_filter_report(feature_df: pd.DataFrame, base: Dict[str, Any]) -> List[Dict[str, Any]]:
    val_idx = np.arange(base["split"]["slices"]["validation"].start, base["split"]["slices"]["validation"].stop)
    test_idx = np.arange(base["split"]["slices"]["test"].start, base["split"]["slices"]["test"].stop)
    rows: List[Dict[str, Any]] = []
    bucket_columns = [c for c in feature_df.columns if c.endswith("_bucket")] + ["session_bucket", "weekday", "month"]
    for col in bucket_columns:
        values = feature_df[col]
        for bucket in sorted(pd.Series(values).dropna().astype(str).unique()):
            val_mask = values.iloc[val_idx].astype(str).to_numpy() == bucket
            test_mask = values.iloc[test_idx].astype(str).to_numpy() == bucket
            val_count = int(val_mask.sum())
            test_count = int(test_mask.sum())
            if val_count < 30 or test_count < 30:
                rows.append(
                    {
                        "filter_column": col,
                        "bucket": bucket,
                        "sample_count_validation": val_count,
                        "sample_count_test": test_count,
                        "sample_size_too_small": True,
                    }
                )
                continue
            selection = select_threshold_validation_only(base["y_val"][val_mask], base["val_prob"][val_mask], THRESHOLD_GRID)
            chosen = selection["selected"]["threshold"]
            val_metrics = classification_metrics(base["y_val"][val_mask], base["val_prob"][val_mask], threshold=chosen)
            test_metrics = classification_metrics(base["y_test"][test_mask], base["test_prob"][test_mask], threshold=chosen)
            val_report = apply_threshold_report(base["y_val"][val_mask], base["val_prob"][val_mask], base["forward_val"][val_mask], chosen)
            test_report = apply_threshold_report(base["y_test"][test_mask], base["test_prob"][test_mask], base["forward_test"][test_mask], chosen)
            rows.append(
                {
                    "filter_column": col,
                    "bucket": bucket,
                    "sample_count_validation": val_count,
                    "sample_count_test": test_count,
                    "positive_label_rate_validation": float(np.mean(base["y_val"][val_mask])),
                    "positive_label_rate_test": float(np.mean(base["y_test"][test_mask])),
                    "validation_roc_auc": float(val_metrics.get("roc_auc", 0.0)),
                    "test_roc_auc": float(test_metrics.get("roc_auc", 0.0)),
                    "pr_auc_test": float(pr_auc_score_safe(base["y_test"][test_mask], base["test_prob"][test_mask]) or 0.0),
                    "selected_threshold": float(chosen),
                    "signals_at_selected_threshold": int(test_report["signal_count"]),
                    "precision_test": float(test_report["precision"]),
                    "recall_test": float(test_report["recall"]),
                    "f1_test": float(test_report["f1"]),
                    "average_predicted_probability_test": float(test_report["average_predicted_probability"]),
                    "improves_signal_quality": bool(test_report["precision"] > 0.2125),
                    "sample_size_too_small": False,
                }
            )
    return rows


def abstention_policy_report(feature_df: pd.DataFrame, base: Dict[str, Any]) -> List[Dict[str, Any]]:
    val_slice = feature_df.iloc[base["split"]["slices"]["validation"]].reset_index(drop=True)
    test_slice = feature_df.iloc[base["split"]["slices"]["test"]].reset_index(drop=True)
    train_slice = feature_df.iloc[base["split"]["slices"]["train"]]
    policies = []
    for pct in (0.8, 0.9):
        policies.append((f"skip_top_{int((1-pct)*100)}pct_realized_vol_30", val_slice["realized_vol_30"] <= float(train_slice["realized_vol_30"].quantile(pct)), test_slice["realized_vol_30"] <= float(train_slice["realized_vol_30"].quantile(pct))))
        policies.append((f"skip_top_{int((1-pct)*100)}pct_atr_pct", val_slice["atr_pct"] <= float(train_slice["atr_pct"].quantile(pct)), test_slice["atr_pct"] <= float(train_slice["atr_pct"].quantile(pct))))
    drift_score_val = (
        (val_slice["realized_vol_30"] > float(train_slice["realized_vol_30"].quantile(0.8))).astype(int)
        + (val_slice["atr_pct"] > float(train_slice["atr_pct"].quantile(0.8))).astype(int)
        + (val_slice["ret_std"] > float(train_slice["ret_std"].quantile(0.8))).astype(int)
    )
    drift_score_test = (
        (test_slice["realized_vol_30"] > float(train_slice["realized_vol_30"].quantile(0.8))).astype(int)
        + (test_slice["atr_pct"] > float(train_slice["atr_pct"].quantile(0.8))).astype(int)
        + (test_slice["ret_std"] > float(train_slice["ret_std"].quantile(0.8))).astype(int)
    )
    policies.append(("skip_multi_feature_drift_score_ge_2", drift_score_val < 2, drift_score_test < 2))
    policies.append(("skip_opening_session", val_slice["session_bucket"] != "opening", test_slice["session_bucket"] != "opening"))
    policies.append(("skip_closing_session", val_slice["session_bucket"] != "closing", test_slice["session_bucket"] != "closing"))

    rows: List[Dict[str, Any]] = []
    for name, val_mask, test_mask in policies:
        val_mask = np.asarray(val_mask, dtype=bool)
        test_mask = np.asarray(test_mask, dtype=bool)
        selection = select_threshold_validation_only(base["y_val"][val_mask], base["val_prob"][val_mask], THRESHOLD_GRID)
        chosen = selection["selected"]["threshold"]
        val_report = apply_threshold_report(base["y_val"][val_mask], base["val_prob"][val_mask], base["forward_val"][val_mask], chosen)
        test_report = apply_threshold_report(base["y_test"][test_mask], base["test_prob"][test_mask], base["forward_test"][test_mask], chosen)
        rows.append(
            {
                "policy": name,
                "retained_sample_pct_validation": float(np.mean(val_mask)),
                "removed_sample_pct_validation": float(1.0 - np.mean(val_mask)),
                "retained_sample_pct_test": float(np.mean(test_mask)),
                "removed_sample_pct_test": float(1.0 - np.mean(test_mask)),
                "selected_threshold": float(chosen),
                "validation_precision": float(val_report["precision"]),
                "validation_signal_count": int(val_report["signal_count"]),
                "test_precision": float(test_report["precision"]),
                "test_signal_count": int(test_report["signal_count"]),
                "test_roc_auc_retained": float(classification_metrics(base["y_test"][test_mask], base["test_prob"][test_mask], threshold=chosen).get("roc_auc", 0.0)),
                "pr_auc_retained": float(pr_auc_score_safe(base["y_test"][test_mask], base["test_prob"][test_mask]) or 0.0),
                "signal_quality_improved": bool(test_report["precision"] > 0.2125),
                "signal_count_too_low": bool(test_report["signal_count"] < 30),
            }
        )
    return rows


def calibration_report(base: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from sklearn.calibration import CalibratedClassifierCV
    try:
        from sklearn.frozen import FrozenEstimator
    except Exception:
        FrozenEstimator = None

    reports: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    methods = [("none", None)]
    if len(base["y_val"]) >= 100:
        methods.append(("sigmoid", "sigmoid"))
    if len(base["y_val"]) >= 400:
        methods.append(("isotonic", "isotonic"))
    for name, method in methods:
        if method is None:
            val_prob = base["val_prob"]
            test_prob = base["test_prob"]
        else:
            estimator = FrozenEstimator(base["model"]) if FrozenEstimator is not None else base["model"]
            cv_mode = None if FrozenEstimator is not None else 3
            calibrator = CalibratedClassifierCV(estimator, method=method, cv=cv_mode)
            calibrator.fit(base["X_val_s"], base["y_val"])
            val_prob = calibrator.predict_proba(base["X_val_s"])[:, 1]
            test_prob = calibrator.predict_proba(base["X_test_s"])[:, 1]
        selection = select_threshold_validation_only(base["y_val"], val_prob, THRESHOLD_GRID)
        chosen = selection["selected"]["threshold"]
        test_report = apply_threshold_report(base["y_test"], test_prob, base["forward_test"], chosen)
        row = {
            "calibration_method": name,
            "brier_score_test": float(brier_score(base["y_test"], test_prob)),
            "expected_calibration_error_test": float(ece_score(base["y_test"], test_prob)),
            "calibration_curve_bins_test": calibration_bins(base["y_test"], test_prob),
            "selected_threshold": float(chosen),
            "validation_threshold_selection": selection,
            "test_threshold_report": test_report,
        }
        reports.append(row)
        if best is None or (test_report["precision"], test_report["signal_count"]) > (best["test_threshold_report"]["precision"], best["test_threshold_report"]["signal_count"]):
            best = row
    return reports, best or reports[0]


def top_candidates(
    regime_rows: List[Dict[str, Any]],
    abstention_rows: List[Dict[str, Any]],
    calibration_best: Dict[str, Any],
    threshold_selection: Dict[str, Any],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for row in regime_rows:
        if row.get("sample_size_too_small"):
            continue
        candidates.append(
            {
                "model_name": BEST_MODEL,
                "label_policy": BEST_POLICY,
                "timeframe": BEST_TIMEFRAME,
                "calibration_method": calibration_best["calibration_method"],
                "threshold": float(row["selected_threshold"]),
                "filter_rule": f"{row['filter_column']}={row['bucket']}",
                "abstention_rule": "none",
                "validation_metrics": {"roc_auc": row["validation_roc_auc"]},
                "test_metrics": {"roc_auc": row["test_roc_auc"], "pr_auc": row["pr_auc_test"]},
                "signal_count": int(row["signals_at_selected_threshold"]),
                "precision": float(row["precision_test"]),
                "recall": float(row["recall_test"]),
                "reason_for_candidate_ranking": "regime filter improved post-threshold signal quality",
                "warning_if_sample_size_low": bool(row["sample_count_test"] < 100),
                "live_promoted": False,
            }
        )
    for row in abstention_rows:
        candidates.append(
            {
                "model_name": BEST_MODEL,
                "label_policy": BEST_POLICY,
                "timeframe": BEST_TIMEFRAME,
                "calibration_method": calibration_best["calibration_method"],
                "threshold": float(row["selected_threshold"]),
                "filter_rule": "none",
                "abstention_rule": row["policy"],
                "validation_metrics": {"precision": row["validation_precision"]},
                "test_metrics": {"roc_auc": row["test_roc_auc_retained"], "pr_auc": row["pr_auc_retained"]},
                "signal_count": int(row["test_signal_count"]),
                "precision": float(row["test_precision"]),
                "recall": 0.0,
                "reason_for_candidate_ranking": "abstention policy improved retained-region precision",
                "warning_if_sample_size_low": bool(row["test_signal_count"] < 50),
                "live_promoted": False,
            }
        )
    candidates.append(
        {
            "model_name": BEST_MODEL,
            "label_policy": BEST_POLICY,
            "timeframe": BEST_TIMEFRAME,
            "calibration_method": calibration_best["calibration_method"],
            "threshold": float(calibration_best["selected_threshold"]),
            "filter_rule": "none",
            "abstention_rule": "none",
            "validation_metrics": threshold_selection["selected"],
            "test_metrics": calibration_best["test_threshold_report"],
            "signal_count": int(calibration_best["test_threshold_report"]["signal_count"]),
            "precision": float(calibration_best["test_threshold_report"]["precision"]),
            "recall": float(calibration_best["test_threshold_report"]["recall"]),
            "reason_for_candidate_ranking": "best calibrated threshold-only candidate",
            "warning_if_sample_size_low": bool(calibration_best["test_threshold_report"]["signal_count"] < 50),
            "live_promoted": False,
        }
    )
    ranked = sorted(candidates, key=lambda row: (row["precision"], row["signal_count"], row["test_metrics"].get("roc_auc", 0.0)), reverse=True)
    return ranked[:3]


def final_recommendation(candidates: List[Dict[str, Any]], calibration_best: Dict[str, Any]) -> str:
    best = candidates[0] if candidates else None
    if best is None:
        return "# DO NOT PROMOTE\n\nNo candidate passed the research filters.\n"
    precision = float(best["precision"])
    signals = int(best["signal_count"])
    if precision < 0.30:
        label = "DO NOT PROMOTE" if signals >= 30 else "PAPER WATCHLIST ONLY"
        reason = "test precision remains below 0.30 even after filtering/calibration"
    elif signals < 50:
        label = "PAPER WATCHLIST ONLY"
        reason = "precision improved but signal count is too low for practical use"
    else:
        label = "PAPER TRADE ONLY"
        reason = "candidate clears precision and signal-count sanity checks but still needs paper validation"
    return "\n".join(
        [
            f"# {label}",
            "",
            f"- Best candidate filter: `{best['filter_rule']}`",
            f"- Best abstention rule: `{best['abstention_rule']}`",
            f"- Best calibration: `{best['calibration_method']}`",
            f"- Threshold: `{best['threshold']}`",
            f"- Test precision: `{precision:.4f}`",
            f"- Test signal count: `{signals}`",
            f"- Reason: {reason}",
            "",
            f"- Baseline unfiltered threshold-0.65 precision was `0.2125` on `80` signals.",
            f"- Best calibrated method was `{calibration_best['calibration_method']}`.",
        ]
    ) + "\n"


def main() -> int:
    args = parse_args()
    artifact_dir = latest_artifact_dir(args.artifact_dir)
    output_dir = MODELS_DIR / f"research_signal_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = extract_existing_artifact_summary(artifact_dir)
    base_dataset = build_base_dataset(artifact_dir)
    feature_df = add_context_columns(base_dataset["dataset"], base_dataset["frame"])
    base = fit_base_logistic(base_dataset["dataset"])

    threshold_selection = select_threshold_validation_only(base["y_val"], base["val_prob"], THRESHOLD_GRID)
    threshold_test_report = apply_threshold_report(base["y_test"], base["test_prob"], base["forward_test"], threshold_selection["selected"]["threshold"])
    threshold_payload = {
        "selected_threshold": threshold_selection["selected"]["threshold"],
        "selection_rule": threshold_selection["selection_rule"],
        "validation_selected_row": threshold_selection["selected"],
        "validation_grid": threshold_selection["grid"],
        "test_report": threshold_test_report,
    }

    regime_rows = regime_filter_report(feature_df, base)
    abstention_rows = abstention_policy_report(feature_df, base)
    calibration_rows, calibration_best = calibration_report(base)

    candidates = top_candidates(regime_rows, abstention_rows, calibration_best, threshold_selection)
    for idx, candidate in enumerate(candidates, start=1):
        write_json(output_dir / f"candidate_signal_filter_{idx}.json", candidate)

    write_json(output_dir / "regime_filter_report.json", regime_rows)
    write_json(output_dir / "abstention_policy_report.json", abstention_rows)
    write_json(output_dir / "calibration_report.json", calibration_rows)
    write_json(output_dir / "validation_threshold_selection.json", threshold_payload)

    root_md = "\n".join(
        [
            "# Signal Quality Root Cause",
            "",
            f"- Base artifact: `{artifact_dir}`",
            f"- Best current setup from prior diagnostics: `{BEST_MODEL}` + `{BEST_POLICY}` + `{BEST_TIMEFRAME}m`.",
            f"- Baseline validation-selected threshold: `{threshold_selection['selected']['threshold']}`.",
            f"- Baseline test precision after validation-only selection: `{threshold_test_report['precision']:.4f}` on `{threshold_test_report['signal_count']}` signals.",
            "- This layer tests regime filters, abstention rules, and calibration without changing live trading.",
        ]
    ) + "\n"
    (output_dir / "signal_quality_root_cause.md").write_text(root_md, encoding="utf-8")

    recommendation = final_recommendation(candidates, calibration_best)
    (output_dir / "final_signal_quality_recommendation.md").write_text(recommendation, encoding="utf-8")

    print(str(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
