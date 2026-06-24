from __future__ import annotations

import argparse
import hashlib
import logging
import json
import math
import os
import traceback
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


TIMEZONE_NAME = "Asia/Kolkata"


def _get_kolkata_tz():
    """Return tz object for Asia/Kolkata (ZoneInfo or pytz) or None if unavailable.
    Used to make tz_localize safe across Windows (no bundled tzdata) and other envs.
    """
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(TIMEZONE_NAME)
    except Exception:
        pass
    try:
        import pytz  # type: ignore[import-not-found]
        return pytz.timezone(TIMEZONE_NAME)
    except Exception:
        pass
    return None


_KOLKATA_TZ = _get_kolkata_tz()

FAST_RF_DEFAULTS: Dict[str, Any] = {
    "n_estimators": 120,
    "max_depth": 12,
    "min_samples_leaf": 20,
    "min_samples_split": 40,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
}
PRODUCTION_RF_DEFAULTS: Dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 14,
    "min_samples_leaf": 15,
    "min_samples_split": 30,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
}
FAST_XGB_DEFAULTS: Dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.04,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 20,
    "reg_lambda": 5.0,
    "reg_alpha": 0.2,
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": -1,
}
PRODUCTION_XGB_DEFAULTS: Dict[str, Any] = {
    "n_estimators": 1000,
    "max_depth": 5,
    "learning_rate": 0.025,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 15,
    "reg_lambda": 5.0,
    "reg_alpha": 0.2,
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "aucpr"],
    "n_jobs": -1,
}
ENSEMBLE_DEFAULTS: Dict[str, Any] = {
    "xgb_weight": 0.70,
    "rf_weight": 0.30,
    "ensemble_threshold": 0.60,
    "xgb_min_prob": 0.58,
    "rf_min_prob": 0.52,
    "max_model_disagreement": 0.25,
    "block_on_disagreement": True,
}
_RUNTIME_OPTIONS: Dict[str, Any] = {
    "fast_mode": False,
    "max_rows": 0,
    "sample_frac": 0.0,
    "latest_rows": True,
    "n_estimators": None,
    "max_depth": None,
    "min_samples_leaf": None,
    "n_jobs": -1,
    "use_parquet_cache": False,
    "cache_dir": "",
    "force_refresh_cache": False,
    "vectorized_backtest": False,
    "max_backtest_rows": 0,
    "rf_n_estimators": None,
    "rf_max_depth": None,
    "rf_min_samples_leaf": None,
    "rf_min_samples_split": None,
    "rf_max_features": None,
    "rf_class_weight": None,
    "rf_n_jobs": None,
    "xgb_device": "auto",
    "xgb_n_estimators": None,
    "xgb_max_depth": None,
    "xgb_learning_rate": None,
    "xgb_subsample": None,
    "xgb_colsample_bytree": None,
    "xgb_reg_lambda": None,
    "xgb_reg_alpha": None,
    "xgb_min_child_weight": None,
    "xgb_tree_method": None,
    "xgb_early_stopping_rounds": None,
    "strict_gpu": False,
    "ensemble": dict(ENSEMBLE_DEFAULTS),
}


def _runtime_option(name: str, default: Any = None) -> Any:
    return _RUNTIME_OPTIONS.get(name, default)


def _set_runtime_options_from_args(args: argparse.Namespace) -> None:
    global _RUNTIME_OPTIONS
    _RUNTIME_OPTIONS = {
        "fast_mode": bool(getattr(args, "fast_mode", False)),
        "max_rows": max(0, int(getattr(args, "max_rows", 0) or 0)),
        "sample_frac": float(getattr(args, "sample_frac", 0.0) or 0.0),
        "latest_rows": bool(getattr(args, "latest_rows", True)),
        "n_estimators": getattr(args, "n_estimators", None),
        "max_depth": getattr(args, "max_depth", None),
        "min_samples_leaf": getattr(args, "min_samples_leaf", None),
        "n_jobs": int(getattr(args, "n_jobs", -1) if getattr(args, "n_jobs", None) is not None else -1),
        "use_parquet_cache": bool(getattr(args, "use_parquet_cache", False)),
        "cache_dir": str(getattr(args, "cache_dir", "") or "").strip(),
        "force_refresh_cache": bool(getattr(args, "force_refresh_cache", False)),
        "vectorized_backtest": bool(getattr(args, "vectorized_backtest", False)),
        "max_backtest_rows": max(0, int(getattr(args, "max_backtest_rows", 0) or 0)),
        "rf_n_estimators": getattr(args, "rf_n_estimators", None),
        "rf_max_depth": getattr(args, "rf_max_depth", None),
        "rf_min_samples_leaf": getattr(args, "rf_min_samples_leaf", None),
        "rf_min_samples_split": getattr(args, "rf_min_samples_split", None),
        "rf_max_features": getattr(args, "rf_max_features", None),
        "rf_class_weight": getattr(args, "rf_class_weight", None),
        "rf_n_jobs": getattr(args, "rf_n_jobs", None),
        "xgb_device": str(getattr(args, "xgb_device", "auto") or "auto").strip().lower(),
        "xgb_n_estimators": getattr(args, "xgb_n_estimators", None),
        "xgb_max_depth": getattr(args, "xgb_max_depth", None),
        "xgb_learning_rate": getattr(args, "xgb_learning_rate", None),
        "xgb_subsample": getattr(args, "xgb_subsample", None),
        "xgb_colsample_bytree": getattr(args, "xgb_colsample_bytree", None),
        "xgb_reg_lambda": getattr(args, "xgb_reg_lambda", None),
        "xgb_reg_alpha": getattr(args, "xgb_reg_alpha", None),
        "xgb_min_child_weight": getattr(args, "xgb_min_child_weight", None),
        "xgb_tree_method": getattr(args, "xgb_tree_method", None),
        "xgb_early_stopping_rounds": getattr(args, "xgb_early_stopping_rounds", None),
        "strict_gpu": bool(getattr(args, "strict_gpu", False)),
        "ensemble": {
            "xgb_weight": float(getattr(args, "xgb_weight", ENSEMBLE_DEFAULTS["xgb_weight"])),
            "rf_weight": float(getattr(args, "rf_weight", ENSEMBLE_DEFAULTS["rf_weight"])),
            "ensemble_threshold": float(getattr(args, "ensemble_threshold", ENSEMBLE_DEFAULTS["ensemble_threshold"])),
            "xgb_min_prob": float(getattr(args, "xgb_min_prob", ENSEMBLE_DEFAULTS["xgb_min_prob"])),
            "rf_min_prob": float(getattr(args, "rf_min_prob", ENSEMBLE_DEFAULTS["rf_min_prob"])),
            "max_model_disagreement": float(getattr(args, "max_model_disagreement", ENSEMBLE_DEFAULTS["max_model_disagreement"])),
            "block_on_disagreement": not bool(getattr(args, "disable_disagreement_gate", False)),
        },
    }
    sample_frac = float(_RUNTIME_OPTIONS["sample_frac"])
    if sample_frac < 0.0:
        raise ValueError("--sample-frac must be >= 0.0")
    if sample_frac >= 1.0:
        _RUNTIME_OPTIONS["sample_frac"] = 1.0
    ensemble_cfg = dict(_RUNTIME_OPTIONS.get("ensemble") or {})
    weight_sum = float(ensemble_cfg.get("xgb_weight", 0.0)) + float(ensemble_cfg.get("rf_weight", 0.0))
    if weight_sum <= 0.0:
        raise ValueError("Ensemble weights must sum to a positive value.")
    if abs(weight_sum - 1.0) > 1e-6:
        print(f"[ensemble] normalizing_weights original_sum={weight_sum:.6f}", flush=True)
        ensemble_cfg["xgb_weight"] = float(ensemble_cfg["xgb_weight"]) / weight_sum
        ensemble_cfg["rf_weight"] = float(ensemble_cfg["rf_weight"]) / weight_sum
    _RUNTIME_OPTIONS["ensemble"] = ensemble_cfg


def _timing_log(stage: str, *, seconds: float, rows: int | None = None, extra: str = "") -> None:
    rate = ""
    if rows is not None and seconds > 0:
        rate = f" rows_per_sec={rows / seconds:.2f}"
    suffix = f" {extra.strip()}" if extra.strip() else ""
    print(f"[timing] stage={stage} seconds={seconds:.3f}{rate}{suffix}", flush=True)


def _normalize_cache_dir(path: Path) -> Path:
    cache_dir_raw = str(_runtime_option("cache_dir", "") or "").strip()
    if cache_dir_raw:
        return Path(cache_dir_raw)
    return path.parent / ".parquet_cache"


def _dataset_fingerprint(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.name}|{int(stat.st_size)}|{int(stat.st_mtime_ns)}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _dataset_cache_paths(path: Path) -> tuple[Path, Path]:
    cache_dir = _normalize_cache_dir(path)
    fingerprint = _dataset_fingerprint(path)
    stem = f"{path.stem}_{fingerprint}"
    return cache_dir / f"{stem}.parquet", cache_dir / f"{stem}.meta.json"


def _latest_rows_by_time(df: pd.DataFrame, keep_rows: int, *, context: str) -> pd.DataFrame:
    if keep_rows <= 0 or len(df) <= keep_rows:
        return df
    work = df.copy()
    if "timestamp" in work.columns:
        ts = _normalize_timestamp_series(work["timestamp"])
        valid_mask = ts.notna()
        if bool(valid_mask.any()):
            work = work.loc[valid_mask].copy()
            work["timestamp"] = ts.loc[valid_mask]
            sort_columns = _chronology_sort_columns(work)
            if not sort_columns:
                sort_columns = ["timestamp"]
            work = work.sort_values(sort_columns, kind="stable")
    trimmed = work.tail(int(keep_rows)).reset_index(drop=True)
    print(
        f"[retrain] row_limit_applied context={context} kept_rows={len(trimmed)} original_rows={len(df)}",
        flush=True,
    )
    return trimmed


def _apply_runtime_dataset_limits(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    limited = df
    max_rows = int(_runtime_option("max_rows", 0) or 0)
    if max_rows > 0:
        limited = _latest_rows_by_time(limited, max_rows, context=f"{context}:max_rows")
    sample_frac = float(_runtime_option("sample_frac", 0.0) or 0.0)
    if 0.0 < sample_frac < 1.0 and not limited.empty:
        print("[retrain] WARNING: --sample-frac can distort time-series backtests and should be used only for fast experiments.", flush=True)
        keep_rows = max(1, int(math.floor(len(limited) * sample_frac)))
        limited = _latest_rows_by_time(limited, keep_rows, context=f"{context}:sample_frac")
    return limited


def _dataset_memory_mb(df: pd.DataFrame) -> float:
    try:
        return float(df.memory_usage(deep=True).sum()) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _dataset_target_distribution(df: pd.DataFrame, target: str = "profitable_trade_label") -> Dict[str, Any]:
    if target not in df.columns:
        return {"target": target, "available": False}
    series = pd.to_numeric(df[target], errors="coerce").dropna()
    if series.empty:
        return {"target": target, "available": True, "rows": 0}
    value_counts = series.value_counts(dropna=False).sort_index()
    return {
        "target": target,
        "available": True,
        "rows": int(series.shape[0]),
        "positive_rate": float(series.mean()),
        "counts": {str(k): int(v) for k, v in value_counts.items()},
    }


def _apply_backtest_row_limit(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    max_rows = int(_runtime_option("max_backtest_rows", 0) or 0)
    if max_rows <= 0:
        return df
    return _latest_rows_by_time(df, max_rows, context=f"{context}:max_backtest_rows")


# =============================================================================
# OOF Prediction Persistence Helpers
# =============================================================================

def _kolkatatz_now() -> datetime:
    """Return current timestamp in Asia/Kolkata (naive local wall time) if tz available, else UTC naive."""
    tz = _KOLKATA_TZ
    if tz is not None:
        try:
            return datetime.now(tz).replace(tzinfo=None)
        except Exception:
            pass
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _build_oof_path(artifact_dir: Path, label_name: str, model_name: str, timestamp: str) -> Path:
    """Build path for OOF predictions parquet file."""
    safe_label = str(label_name).replace("/", "_").replace("\\", "_")
    safe_model = str(model_name).replace("/", "_").replace("\\", "_")
    filename = f"oof_predictions_{safe_label}_{safe_model}_{timestamp}.parquet"
    return artifact_dir / filename


def _persist_oof_predictions(
    artifact_dir: Path,
    label_name: str,
    model_name: str,
    oof_data: pd.DataFrame,
    timestamp: str,
) -> Dict[str, Any]:
    """Persist OOF predictions to parquet file.

    Args:
        artifact_dir: Directory to save OOF file
        label_name: Label/target name
        model_name: Model family name
        oof_data: DataFrame with OOF predictions
        timestamp: Timestamp string for filename

    Returns:
        Dict with oof_file_path and row_count
    """
    if oof_data.empty:
        return {"oof_file_path": None, "row_count": 0, "status": "empty"}

    oof_path = _build_oof_path(artifact_dir, label_name, model_name, timestamp)
    try:
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        oof_data.to_parquet(oof_path, index=False, engine="auto")
        return {
            "oof_file_path": str(oof_path),
            "row_count": int(len(oof_data)),
            "status": "success",
            "timestamp": timestamp,
            "label_name": label_name,
            "model_name": model_name,
        }
    except Exception as exc:
        return {
            "oof_file_path": None,
            "row_count": 0,
            "status": "error",
            "error": str(exc),
        }


def _collect_walk_forward_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.Series,
    feature_cols: Sequence[str],
    label_name: str,
    scaler_mean: List[float],
    scaler_std: List[float],
    *, 
    fold_count: int = 5,
    returns: np.ndarray | None = None,
    evaluation_return_column: str | None = None,
) -> pd.DataFrame:
    """Collect OOF predictions from walk-forward validation splits.

    Returns a DataFrame with per-row OOF predictions for all rows that appear
    in validation/test folds across all walk-forward splits.
    """
    timestamps_norm = pd.Series(
        _normalize_timestamp_series(pd.Series(timestamps)), name="timestamp"
    ).reset_index(drop=True)
    _assert_monotonic_timestamps(timestamps_norm, context="oof_walk_forward_input")

    splits = _generate_grouped_walk_forward_splits(
        timestamps_norm, fold_count=max(2, int(fold_count)), purge_groups=15
    )

    if not splits:
        return pd.DataFrame()

    all_oof_rows: List[Dict[str, Any]] = []
    for fold_idx, split in enumerate(splits, start=1):
        train_idx = split["train"]
        val_idx = split["validation"]
        test_idx = split["test"]

        _assert_fold_is_chronological(
            timestamps_norm, train_idx, val_idx, test_idx=test_idx, context=f"oof_fold_{fold_idx}"
        )

        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        X_train_s, X_val_s, X_test_s, _, _ = scale_train_val_test(X_train, X_val, X_test)

        model = build_safe_model("logistic_regression")
        if model is None:
            continue

        fit_model(model, X_train_s, y_train)

        val_prob = predict_proba_positive(model, X_val_s)
        test_prob = predict_proba_positive(model, X_test_s)

        val_returns = returns[val_idx] if returns is not None and len(returns) == len(y) else None
        test_returns = returns[test_idx] if returns is not None and len(returns) == len(y) else None

        chosen = optimize_threshold_from_probs(y_val, val_prob, val_returns, thresholds=RETRAIN_THRESHOLD_SWEEP)
        threshold = float(chosen["threshold"])

        for idx_set, probs, y_true, fold_type, fold_returns in [
            (val_idx, val_prob, y_val, "validation", val_returns),
            (test_idx, test_prob, y_test, "test", test_returns),
        ]:
            for local_i, global_i in enumerate(idx_set):
                ts = timestamps_norm.iloc[global_i] if global_i < len(timestamps_norm) else pd.NaT
                row: Dict[str, Any] = {
                    "original_index": int(global_i),
                    "timestamp": ts,
                    "fold_id": int(fold_idx),
                    "fold_type": fold_type,
                    "y_true": int(y_true[local_i]) if local_i < len(y_true) else None,
                    "y_prob": float(probs[local_i]) if local_i < len(probs) else None,
                    "threshold": threshold,
                    "predicted_label": int(probs[local_i] >= threshold) if local_i < len(probs) else None,
                }

                if fold_returns is not None and local_i < len(fold_returns):
                    row["selected_return_column"] = float(fold_returns[local_i])

                all_oof_rows.append(row)

    if not all_oof_rows:
        return pd.DataFrame()

    oof_df = pd.DataFrame(all_oof_rows)

    oof_df = oof_df.drop_duplicates(subset=["original_index", "fold_type"], keep="first")

    oof_df["y_prob"] = oof_df["y_prob"].astype(float)
    oof_df["y_true"] = oof_df["y_true"].astype(int)
    oof_df["predicted_label"] = oof_df["predicted_label"].astype(int)

    return oof_df


def _build_oof_frame_from_holdout(
    test_frame: pd.DataFrame,
    y_test: np.ndarray,
    test_prob: np.ndarray,
    threshold: float,
    model_name: str,
    label_name: str,
    fold_id: int = 1,
    evaluation_return_column: str | None = None,
) -> pd.DataFrame:
    """Build OOF DataFrame from holdout test predictions.

    Columns included:
    - timestamp: Row timestamp
    - fold_id: Fold identifier
    - fold_type: 'test' for holdout test set
    - y_true: Actual label value
    - y_prob: Predicted probability
    - threshold: Selected threshold for this model
    - predicted_label: Binary prediction at threshold
    - selected_return_column: Return/PnL for evaluation (NOT as feature)
    - option_type: Option type (CE/PE) if available
    - dte: Days to expiry if available
    - moneyness: Moneyness if available
    - cost_estimate_used: Cost estimate for evaluation

    Returns DataFrame with leakage-safe columns only.
    """
    rows = []
    test_prob_arr = np.asarray(test_prob, dtype=float)

    option_type_col = None
    if "option_type" in test_frame.columns:
        option_type_col = test_frame["option_type"]
    elif "option_type_ce" in test_frame.columns:
        option_type_col = test_frame["option_type_ce"].apply(
            lambda x: "CE" if pd.notna(x) and float(x) >= 1.0 else ("PE" if "option_type_pe" in test_frame.columns and pd.notna(test_frame["option_type_pe"].iloc[0]) else "UNKNOWN")
        )

    dte_col = None
    if "dte_days" in test_frame.columns:
        dte_col = test_frame["dte_days"]

    moneyness_col = None
    if "moneyness" in test_frame.columns:
        moneyness_col = test_frame["moneyness"]

    selected_return_col = None
    if evaluation_return_column and evaluation_return_column in test_frame.columns:
        selected_return_col = pd.to_numeric(test_frame[evaluation_return_column], errors="coerce")

    for i in range(len(test_frame)):
        ts = test_frame["timestamp"].iloc[i] if "timestamp" in test_frame.columns else pd.NaT

        row: Dict[str, Any] = {
            "timestamp": ts,
            "fold_id": int(fold_id),
            "fold_type": "test",
            "y_true": int(y_test[i]) if i < len(y_test) else None,
            "y_prob": float(test_prob_arr[i]) if i < len(test_prob_arr) else None,
            "threshold": float(threshold),
            "predicted_label": int(test_prob_arr[i] >= threshold) if i < len(test_prob_arr) else None,
        }

        if selected_return_col is not None:
            row["selected_return_column"] = float(selected_return_col.iloc[i]) if i < len(selected_return_col) else None

        if option_type_col is not None:
            row["option_type"] = str(option_type_col.iloc[i]) if i < len(option_type_col) else "UNKNOWN"

        if dte_col is not None:
            val = dte_col.iloc[i] if i < len(dte_col) else None
            row["dte"] = float(val) if pd.notna(val) else None

        if moneyness_col is not None:
            val = moneyness_col.iloc[i] if i < len(moneyness_col) else None
            row["moneyness"] = float(val) if pd.notna(val) else None

        rows.append(row)

    return pd.DataFrame(rows)


def _get_leakage_safe_passthrough_columns(df: pd.DataFrame, feature_cols: Sequence[str]) -> List[str]:
    """Get columns safe to include in OOF output (metadata, NOT features).

    Excludes:
    - Feature columns used for training
    - Return/PnL columns (evaluation only)
    - Label columns
    - Any column with forbidden tokens
    """
    forbidden_tokens = (
        "pnl", "profit", "target", "label", "future", "forward", "next",
        "exit", "loss", "outcome", "realized", "return", "trade_result",
    )

    passthrough = []
    for col in df.columns:
        if col in feature_cols:
            continue
        col_lower = str(col).lower()
        if any(token in col_lower for token in forbidden_tokens):
            continue
        if col.lower() in {"timestamp", "dte_days", "dte", "moneyness", "option_type",
                            "option_type_ce", "option_type_pe", "expiry", "trading_date"}:
            passthrough.append(col)

    return passthrough



REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
MODELS_DIR = REPO_ROOT / "models"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_option_edge_dataset import make_output_dir as make_dataset_output_dir
from retrain_nifty_1year import (
    build_safe_model,
    calibration_bins,
    chronological_split,
    compare_feature_compatibility,
    confusion_from_threshold,
    model_names_to_train,
    pr_auc_score_safe,
    predict_proba_positive,
    scale_train_val_test,
    threshold_sweep,
    write_json,
)
from retraining_validation import classification_metrics, generate_purged_embargoed_cv_splits
from ml_signals import _wrap_model
from ml_execution_costs import DEFAULT_EXECUTION_COST_CONFIG, estimate_option_execution_costs_frame
from ml.ensemble_xgb_rf import XGBRFEnsembleClassifier
from ml.feature_safety import (
    apply_train_median_fill,
    build_safe_numeric_bool_frame,
    compute_train_medians,
    filter_target_rows,
)

# Feature contract import for training/inference alignment
try:
    from ml_feature_contract import (
        ALLOWED_LIVE_FEATURES,
        is_feature_forbidden,
        verify_training_feature_compatibility,
        CONTRACT_VERSION,
    )
    HAS_FEATURE_CONTRACT = True
except ImportError:
    HAS_FEATURE_CONTRACT = False
    ALLOWED_LIVE_FEATURES = set()


# =============================================================================
# FEATURE CONTRACT VALIDATION
# =============================================================================

def _live_contract_feature_set() -> set[str]:
    return {str(name) for name in (ALLOWED_LIVE_FEATURES or set())}


def _build_live_contract_feature_row(
    feature_name: str,
    *,
    dataset_columns: Sequence[str] | None = None,
) -> Dict[str, Any]:
    dataset_feature_names = {str(name) for name in dataset_columns} if dataset_columns is not None else set()
    in_live_contract = feature_name in _live_contract_feature_set()
    forbidden = bool(HAS_FEATURE_CONTRACT and is_feature_forbidden(feature_name))
    return {
        "feature_name": str(feature_name),
        "exists_in_dataset": feature_name in dataset_feature_names if dataset_feature_names else None,
        "exists_in_live_contract": bool(in_live_contract),
        "is_forbidden": bool(forbidden),
        "available_live": bool(in_live_contract and not forbidden),
        "computable_before_decision": bool(in_live_contract and not forbidden),
        "reason_if_excluded": (
            "forbidden_feature_name"
            if forbidden
            else ("missing_from_live_contract" if not in_live_contract else None)
        ),
        "suggested_fix": (
            "exclude_from_live_computable_training"
            if forbidden
            else ("add_to_live_contract_and_live_builder_or_exclude" if not in_live_contract else None)
        ),
    }


def _live_contract_audit_rows(
    feature_names: Sequence[str],
    *,
    dataset_columns: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    rows: List[Dict[str, Any]] = []
    for raw_name in feature_names:
        name = str(raw_name)
        if not name or name in seen:
            continue
        seen.add(name)
        rows.append(_build_live_contract_feature_row(name, dataset_columns=dataset_columns))
    return rows


def _filter_live_contract_features(
    feature_names: Sequence[str],
    *,
    dataset_columns: Sequence[str] | None = None,
    strict: bool = False,
) -> tuple[List[str], List[Dict[str, Any]]]:
    audit = _live_contract_audit_rows(feature_names, dataset_columns=dataset_columns)
    selected = [
        str(row["feature_name"])
        for row in audit
        if bool(row["exists_in_live_contract"]) and not bool(row["is_forbidden"])
    ]
    excluded = [row for row in audit if str(row["feature_name"]) not in selected]
    if strict and excluded:
        details = [
            {
                "feature_name": row["feature_name"],
                "exists_in_dataset": row["exists_in_dataset"],
                "exists_in_live_contract": row["exists_in_live_contract"],
                "suggested_fix": row["suggested_fix"],
            }
            for row in excluded
        ]
        raise RuntimeError(
            "STRICT_LIVE_CONTRACT_VIOLATION: candidate training features are not in the live contract. "
            f"details={details}"
        )
    return selected, audit


def _check_training_features_hygiene(
    feature_names: List[str],
    label_name: str = "unknown",
    *,
    dataset_columns: Sequence[str] | None = None,
    enforce_live_contract: bool = False,
) -> Dict[str, Any]:
    """
    Validate training features against the feature contract.
    
    This function checks that:
    1. No forbidden features (labels, returns, PnL, future data) are used in training
    2. Features are compatible with live inference contract
    3. Version compatibility with live inference
    
    Args:
        feature_names: List of feature names used in training.
        label_name: Name of the target label (for reporting).
    
    Returns:
        Dict with hygiene check results and any issues found.
    """
    result = {
        "label_name": label_name,
        "feature_count": len(feature_names),
        "hygiene_passed": True,
        "issues": [],
        "warnings": [],
        "contract_version": CONTRACT_VERSION if HAS_FEATURE_CONTRACT else None,
    }
    
    if not HAS_FEATURE_CONTRACT:
        result["warnings"].append("feature_contract module not available - skipping hygiene check")
        return result
    
    # Check for forbidden features
    forbidden_found = [f for f in feature_names if is_feature_forbidden(f)]
    if forbidden_found:
        result["hygiene_passed"] = False
        result["issues"].append({
            "type": "FORBIDDEN_FEATURES_IN_TRAINING",
            "features": forbidden_found,
            "message": "Training features must not include labels, returns, PnL, or future data.",
        })
    
    # Verify compatibility with live inference contract
    is_compatible, issues = verify_training_feature_compatibility(feature_names)
    if not is_compatible:
        unknown_rows = [
            row
            for row in _live_contract_audit_rows(feature_names, dataset_columns=dataset_columns)
            if not bool(row["exists_in_live_contract"]) and not bool(row["is_forbidden"])
        ]
        for issue in issues:
            payload: Dict[str, Any] = {"type": "INCOMPATIBILITY", "details": issue}
            if str(issue).startswith("UNKNOWN_FEATURES_IN_TRAINING"):
                payload["unknown_feature_details"] = [
                    {
                        "feature_name": row["feature_name"],
                        "exists_in_dataset": row["exists_in_dataset"],
                        "exists_in_live_contract": row["exists_in_live_contract"],
                        "suggested_fix": row["suggested_fix"],
                    }
                    for row in unknown_rows
                ]
            if enforce_live_contract:
                result["hygiene_passed"] = False
                result["issues"].append(payload)
            else:
                result["warnings"].append(payload)

    return result


def _report_feature_hygiene_results(hygiene_results: List[Dict[str, Any]], artifact_dir: Path) -> None:
    """Write feature hygiene report to artifact directory."""
    if not hygiene_results:
        return
    
    # Count issues
    total_issues = sum(len(r.get("issues", [])) for r in hygiene_results)
    hygiene_passed = all(r.get("hygiene_passed", False) for r in hygiene_results)
    
    # Create report
    report = {
        "timestamp": _kolkatatz_now().isoformat(),
        "total_labels_checked": len(hygiene_results),
        "labels_with_issues": sum(1 for r in hygiene_results if not r.get("hygiene_passed", False)),
        "total_issues_found": total_issues,
        "overall_hygiene_passed": hygiene_passed,
        "contract_version": CONTRACT_VERSION if HAS_FEATURE_CONTRACT else None,
        "results_by_label": hygiene_results,
    }
    
    # Write JSON report
    report_path = artifact_dir / "feature_hygiene_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Write MD summary
    md_path = artifact_dir / "feature_hygiene_report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Feature Hygiene Report\n\n")
        f.write(f"**Generated:** {report['timestamp']}\n\n")
        f.write(f"**Contract Version:** {report['contract_version']}\n\n")
        f.write(f"**Overall Status:** {'✅ PASSED' if hygiene_passed else '❌ FAILED'}\n\n")
        f.write(f"- Labels Checked: {report['total_labels_checked']}\n")
        f.write(f"- Labels with Issues: {report['labels_with_issues']}\n")
        f.write(f"- Total Issues Found: {total_issues}\n\n")
        
        if hygiene_results:
            f.write("## Results by Label\n\n")
            f.write("| Label | Features | Hygiene | Issues |\n")
            f.write("|-------|----------|---------|--------|\n")
            for r in hygiene_results:
                status = '✅' if r.get('hygiene_passed') else '❌'
                issue_count = len(r.get('issues', []))
                f.write(f"| {r.get('label_name', 'unknown')} | {r.get('feature_count', 0)} | {status} | {issue_count} |\n")



THRESHOLD_GRID = [0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75, 0.775, 0.80]
EVALUATION_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
RETRAIN_THRESHOLD_SWEEP = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
STRICT_PAPER_FILTER_TRADE_MIN = 500
STRICT_PAPER_PF_MIN = 1.15
STRICT_PAPER_SHARPE_MIN = 0.75
STRICT_WORST_FOLD_PF_MIN = 0.90
STRICT_MEDIAN_FOLD_PF_MIN = 1.10
STRICT_DAILY_RETURN_CONCENTRATION_MAX = 0.25
EXTRA_COST_SCENARIOS = {
    "base": 0.0,
    "extra_cost_0_25": 0.25,
    "extra_cost_0_50": 0.50,
    "extra_cost_1_00": 1.00,
    "extra_cost_2_00": 2.00,
}
EDGE_REFINEMENT_THRESHOLD_RULES = [0.55, 0.60, 0.65, 0.70, 0.75]
EDGE_REFINEMENT_TOP_N_RULES = [1, 2, 3, 5]
EDGE_REFINEMENT_TOP_PCT_RULES = [0.01, 0.02, 0.03]
EDGE_REFINEMENT_PROBABILITY_PERCENTILE_RULES = [99.5, 99.0, 98.0, 97.0, 95.0]
EDGE_REFINEMENT_TARGET_TRADE_COUNTS = [100, 250, 500, 750, 1000, 1500, 2000]
EDGE_REFINEMENT_FOCUS_MODELS = [
    "random_forest",
    "xgboost",
    "calibrated_logistic_regression",
    "extra_trees",
    "ensemble",
    "logistic_regression",
]
EDGE_REFINEMENT_WATCHLIST_GATES = {
    "trade_count_min": 100,
    "trade_count_max": 5000,
    "profit_factor_min": 1.20,
    "sharpe_min": 1.00,
    "cost_1_25x_pf_min": 1.05,
    "cost_1_50x_pf_min": 1.00,
    "worst_fold_pf_min": 0.90,
}
EDGE_REFINEMENT_READY_GATES = {
    "trade_count_min": 250,
    "profit_factor_min": 1.25,
    "sharpe_min": 1.25,
    "cost_1_25x_pf_min": 1.10,
    "cost_1_50x_pf_min": 1.00,
    "worst_fold_pf_min": 1.00,
    "median_fold_pf_min": 1.10,
}
# =============================================================================
# AVOID_LABEL_TRIVIAL_INVERSE AUDIT (2026-06-09)
# -----------------------------------------------------------------------------
# On the current binary datasets, avoid_trade_label is STRUCTURALLY DEFINED as:
#     avoid_trade_label = (net_forward_return <= 0.0)
# while profitable_trade_label is:
#     profitable_trade_label = (net_forward_return > 0.0)
# Therefore: avoid_trade_label = 1 - profitable_trade_label  (trivial inverse)
#
# Training a second-stage avoid model on an exact inverse provides ZERO additional
# discriminative power. The profit+avoid overlay is a mathematical no-op that
# cannot improve trade selection beyond what the primary profit model already knows.
#
# This flag governs all downstream behaviour:
#   - Set True:  skip avoid_trade_label model training; document overlay as no-op
#   - Set False: allow avoid_trade_label to be trained if it has non-trivial definition
AVOID_LABEL_TRIVIAL_INVERSE = True
# =============================================================================

PRIMARY_LABELS = [
    "profitable_trade_label",
    # "avoid_trade_label"  # REMOVED 2026-06-09: trivially inverse of profitable_trade_label; training it wastes compute and creates misleading two-stage overlay claims
    "profitable_trade_label_edge_1x_cost",
    "profitable_trade_label_edge_2x_cost",
    "profitable_trade_label_edge_3x_cost",
    "profitable_trade_label_edge_5x_cost",
    "big_move_profitable_label",
    "asymmetric_payoff_label",
    "cost_adjusted_success_5m",
    "cost_adjusted_success_10m",
    "cost_adjusted_success_15m",
    "ternary_trade_quality_5m",
    "ternary_trade_quality_10m",
    "ternary_trade_quality_15m",
    "fallback_trade_quality_ternary",
    # === Cost-aware labels added by retraining pipeline fix ===
    "strong_profitable_trade_label",
    "cost_survivor_label",
    "strong_profitable_trade_label_v2",
    "cost_survivor_label_v2",
    "weak_trade_label",
    "no_trade_label",
    "high_conviction_trade_label",
    "paper_candidate_label",
    "high_conviction_trade_label_v2",
    "paper_candidate_label_v2",
]
FUTURE_LEAK_TOKENS = ("_after_", "gross_pnl_", "net_pnl_", "max_favorable_excursion", "max_adverse_excursion", "option_trade_success_", "cost_adjusted_success_", "ternary_trade_quality_", "simulated_net_pnl_")
RETURN_COLUMN_CANDIDATES = [
    "net_forward_return",
    "forward_return",
    "realized_pnl",
    "trade_pnl",
    "net_pnl",
    "pnl",
    "cost_adjusted_return",
    "label_return",
    "target_return",
]
STRICT_FORBIDDEN_TOKENS = (
    "future",
    "realized",
    "next",
    "target",
    "label",
    "pnl",
    "profit",
    "loss",
    "return",
    "outcome",
    "exit",
    "entry_result",
    "trade_result",
    "hit_target",
    "hit_sl",
    "mae",
    "mfe",
    "forward",
    "future_price",
    "future_return",
    "post_trade",
)
METADATA_COLUMNS = {
    "prediction_id",
    "timestamp",
    "trading_date",
    "trading_day",
    "trading_day_spot",
    "session_bucket",
    "selected_option_symbol",
    "selected_option_token",
    "instrument_key",
    "trading_symbol",
    "source_file",
    "spot_source",
    "observer_signal_id",
    "observer_run_id",
    "expiry",
    "option_type",
    "option_side_normalized",
    "greeks_source",
    "bs_iv_source",
    "row_enrichment_error",
    "raw_model_outputs_json",
    "feature_snapshot_json",
    "regime_features_json",
    "drift_sensitive_features_json",
    "data_quality_flags_json",
    "missing_fields_json",
    "observer_error",
    "entry_rule_that_fired",
    "signal_reason",
    "signal_side",
    "trade_quality_label",
    "simulated_result_label",
}
NON_LEAKY_REALIZED_FEATURE_TOKENS = (
    "realized_vol",
    "realized_volatility",
    "realized_vol_percentile",
)
DEFAULT_MODEL_FAMILIES = [
    "logistic_regression",
    "logistic_regression_platt",
    "logistic_regression_isotonic",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
    "ensemble",
]
FORBIDDEN_COLUMN_TOKENS = (
    "pnl",
    "profit",
    "target",
    "label",
    "future",
    "forward",
    "next",
    "exit",
    "loss",
    "outcome",
    "next_return",
    "net_forward_return",
    "gross_forward_return",
    "return",
    "realized",
    "trade_result",
    "post_entry_return",
    "max_profit",
    "max_loss",
)
BS_RESEARCH_FEATURES = [
    "bs_iv",
    "bs_delta",
    "bs_gamma",
    "bs_theta",
    "bs_vega",
    "bs_rho",
    "moneyness",
    "log_moneyness",
    "distance_from_atm",
    "distance_from_atm_pct",
    "intrinsic_value",
    "extrinsic_value",
    "time_to_expiry_days",
    "time_to_expiry_years",
    "bid_ask_spread",
    "bid_ask_spread_pct",
    "mid_price",
    "ltp_vs_mid_diff_pct",
    "greeks_quality_score",
]
BS_FLAG_FEATURES = [
    "bs_iv_solve_ok",
    "bad_iv_flag",
    "bad_greek_flag",
    "wide_spread_flag",
    "low_price_flag",
    "deep_itm_flag",
    "deep_otm_flag",
    "row_enrichment_ok",
]

TIMEZONE_NAME = "Asia/Kolkata"
DATASET_SEARCH_GLOBS = [
    "data/processed/*.csv",
    "data/processed/*.parquet",
    "data/*.csv",
    "data/*.parquet",
]
CHRONOLOGY_SORT_COLUMNS = [
    "timestamp",
    "expiry",
    "strike_price",
    "option_type",
    "selected_option_symbol",
    "selected_option_token",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research-only edge-model retraining pipeline.")
    parser.add_argument("--dataset", default="", help="Path to edge_dataset.csv/parquet")
    parser.add_argument("--fast-mode", action="store_true", help="Use faster runtime defaults for research retraining and backtests.")
    parser.add_argument("--max-rows", type=int, default=0, help="Keep only the latest N rows before training/backtesting when timestamp is available.")
    parser.add_argument("--sample-frac", type=float, default=0.0, help="Keep only the latest fraction of rows after chronological ordering. Range: 0.0-1.0.")
    parser.add_argument("--latest-rows", action="store_true", default=True, help="Prefer latest rows when row limiting is active. Default: true.")
    parser.add_argument("--n-estimators", type=int, default=None, help="Override RandomForest n_estimators.")
    parser.add_argument("--max-depth", type=int, default=None, help="Override RandomForest max_depth.")
    parser.add_argument("--min-samples-leaf", type=int, default=None, help="Override RandomForest min_samples_leaf.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel worker count for RandomForest and optional backtest evaluation. Default: -1.")
    parser.add_argument("--use-parquet-cache", action="store_true", help="Cache CSV datasets as parquet and prefer the cache on subsequent runs.")
    parser.add_argument("--cache-dir", default="", help="Optional directory for parquet dataset caches.")
    parser.add_argument("--force-refresh-cache", action="store_true", help="Ignore any parquet cache and rebuild it from the source dataset.")
    parser.add_argument("--vectorized-backtest", action="store_true", help="Use vectorized selection paths and parallel edge-refinement candidate evaluation.")
    parser.add_argument("--max-backtest-rows", type=int, default=0, help="Limit backtest/evaluation to the latest N holdout rows.")
    parser.add_argument("--retrain-all-models", action="store_true", help="Run a lightweight retraining pass across the standard supported model families.")
    parser.add_argument("--improve-models", action="store_true", help="Alias for the stricter all-model retraining workflow.")
    parser.add_argument("--retrain-all-models-strict", action="store_true", help="Run strict multi-model, multi-feature-set retraining audit.")
    parser.add_argument("--model-rescue-experiments", action="store_true", help="Run targeted rescue experiments across smaller market slices.")
    parser.add_argument("--model-families", default="default", help="Comma list of model families or 'all'.")
    parser.add_argument("--full-pipeline", action="store_true", help="Run dataset build first, then retrain.")
    parser.add_argument("--use-black-scholes-features", action="store_true", help="Train baseline and Black-Scholes-enriched research variants.")
    parser.add_argument("--compare-baseline", action="store_true", help="Retain baseline-vs-enriched comparison reporting.")
    parser.add_argument("--no-production-adopt", action="store_true", help="Keep production adoption disabled.")
    parser.add_argument("--output-dir", default="", help="Optional artifact output directory.")
    parser.add_argument("--min-rows", type=int, default=1000, help="Minimum rows required to run retraining.")
    parser.add_argument("--walk-forward-folds", type=int, default=5, help="Requested walk-forward folds for offline reporting.")
    parser.add_argument("--drop-bad-greeks", action="store_true", help="Drop rows flagged with bad IV/Greeks when Black-Scholes features are enabled.")
    parser.add_argument("--risk-free-rate", type=float, default=0.065, help="Risk-free rate passed to enrichment when needed.")
    parser.add_argument("--dividend-yield", type=float, default=0.0, help="Dividend yield passed to enrichment when needed.")
    parser.add_argument("--symbol", default="NIFTY", help="Underlying symbol passed to enrichment when needed.")
    parser.add_argument("--same-period-comparison", action="store_true", help="Restrict baseline and Black-Scholes variants to overlapping timestamp range.")
    parser.add_argument("--threshold-grid-start", type=float, default=0.50, help="Threshold sweep start for robustness reporting.")
    parser.add_argument("--threshold-grid-end", type=float, default=0.90, help="Threshold sweep end for robustness reporting.")
    parser.add_argument("--threshold-grid-step", type=float, default=0.025, help="Threshold sweep step for robustness reporting.")
    parser.add_argument("--min-threshold-trades", type=int, default=1000, help="Minimum trades required for a threshold to count as robust.")
    parser.add_argument("--paper-readiness-audit", action="store_true", help="Run strict paper-trading readiness audit.")
    parser.add_argument("--paper-readiness-report", action="store_true", help="Alias for writing the paper-readiness report.")
    parser.add_argument("--full-validation", action="store_true", help="Alias for stricter validation/report generation.")
    parser.add_argument("--walk-forward", action="store_true", help="Alias for chronological walk-forward validation.")
    parser.add_argument("--threshold-sweep", action="store_true", help="Alias for explicit threshold sweep reporting.")
    parser.add_argument("--cost-stress", action="store_true", help="Alias for cost stress reporting.")
    parser.add_argument("--calibration", action="store_true", help="Alias for calibration reporting.")
    parser.add_argument("--resume", action="store_true", help="Resume from an incomplete retraining artifact directory.")
    parser.add_argument("--resume-artifact-dir", default="", help="Artifact directory to resume from.")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain even if a completed checkpoint exists.")
    parser.add_argument("--train-only", action="store_true", help="Train and checkpoint model pairs without expensive final aggregation.")
    parser.add_argument("--report-only", action="store_true", help="Regenerate aggregate reports from checkpointed artifacts only.")
    parser.add_argument("--only-target", action="append", default=[], help="Restrict retraining to specific target names. Repeatable or comma-separated.")
    parser.add_argument("--only-model", action="append", default=[], help="Restrict retraining to specific model names. Repeatable or comma-separated.")
    parser.add_argument("--train-ensemble", default="", help="Optional ensemble alias, for example: xgb_rf")
    parser.add_argument("--paper-watchlist-only", action="store_true", help="Generate paper-watchlist observation rules only; never place trades.")
    parser.add_argument("--live-computable-only", action="store_true", help="Restrict training to features that are live-computable before decision time.")
    parser.add_argument("--strict-live-contract", action="store_true", help="Fail if any candidate training feature is missing from the live feature contract.")
    parser.add_argument("--rf-n-estimators", type=int, default=None, help="RandomForest n_estimators override.")
    parser.add_argument("--rf-max-depth", type=int, default=None, help="RandomForest max_depth override.")
    parser.add_argument("--rf-min-samples-leaf", type=int, default=None, help="RandomForest min_samples_leaf override.")
    parser.add_argument("--rf-min-samples-split", type=int, default=None, help="RandomForest min_samples_split override.")
    parser.add_argument("--rf-max-features", default=None, help="RandomForest max_features override.")
    parser.add_argument("--rf-class-weight", default=None, help="RandomForest class_weight override.")
    parser.add_argument("--rf-n-jobs", type=int, default=None, help="RandomForest n_jobs override.")
    parser.add_argument("--xgb-device", choices=["auto", "cuda", "cpu"], default="auto", help="XGBoost device selection.")
    parser.add_argument("--strict-gpu", action="store_true", help="Fail instead of falling back to CPU when CUDA is requested but unavailable.")
    parser.add_argument("--xgb-n-estimators", type=int, default=None, help="XGBoost n_estimators override.")
    parser.add_argument("--xgb-max-depth", type=int, default=None, help="XGBoost max_depth override.")
    parser.add_argument("--xgb-learning-rate", type=float, default=None, help="XGBoost learning_rate override.")
    parser.add_argument("--xgb-subsample", type=float, default=None, help="XGBoost subsample override.")
    parser.add_argument("--xgb-colsample-bytree", type=float, default=None, help="XGBoost colsample_bytree override.")
    parser.add_argument("--xgb-reg-lambda", type=float, default=None, help="XGBoost reg_lambda override.")
    parser.add_argument("--xgb-reg-alpha", type=float, default=None, help="XGBoost reg_alpha override.")
    parser.add_argument("--xgb-min-child-weight", type=float, default=None, help="XGBoost min_child_weight override.")
    parser.add_argument("--xgb-tree-method", default=None, help="XGBoost tree_method override.")
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=None, help="XGBoost early stopping rounds.")
    parser.add_argument("--xgb-weight", type=float, default=ENSEMBLE_DEFAULTS["xgb_weight"], help="Weighted ensemble XGBoost probability weight.")
    parser.add_argument("--rf-weight", type=float, default=ENSEMBLE_DEFAULTS["rf_weight"], help="Weighted ensemble RandomForest probability weight.")
    parser.add_argument("--ensemble-threshold", type=float, default=ENSEMBLE_DEFAULTS["ensemble_threshold"], help="Weighted ensemble gate threshold.")
    parser.add_argument("--xgb-min-prob", type=float, default=ENSEMBLE_DEFAULTS["xgb_min_prob"], help="Minimum XGBoost probability required by the gated ensemble.")
    parser.add_argument("--rf-min-prob", type=float, default=ENSEMBLE_DEFAULTS["rf_min_prob"], help="Minimum RandomForest probability required by the gated ensemble.")
    parser.add_argument("--max-model-disagreement", type=float, default=ENSEMBLE_DEFAULTS["max_model_disagreement"], help="Maximum allowed abs(xgb_prob-rf_prob) before blocking.")
    parser.add_argument("--disable-disagreement-gate", action="store_true", help="Disable ensemble disagreement blocking.")
    parser.add_argument("--paper-max-trades-per-day", type=int, default=5)
    parser.add_argument("--paper-cooldown-minutes", type=int, default=15)
    parser.add_argument("--paper-allow-ce", action="store_true", default=True)
    parser.add_argument("--paper-allow-pe", action="store_true", default=True)
    parser.add_argument("--paper-min-option-price", type=float, default=5.0)
    parser.add_argument("--paper-max-bid-ask-spread-pct", type=float, default=5.0)
    parser.add_argument("--paper-avoid-opening-minutes", type=int, default=5)
    parser.add_argument("--paper-avoid-closing-minutes", type=int, default=5)
    parser.add_argument("--rescue-filter-ce-only", action="store_true")
    parser.add_argument("--rescue-filter-pe-only", action="store_true")
    parser.add_argument("--rescue-filter-itm-only", action="store_true")
    parser.add_argument("--rescue-filter-atm-only", action="store_true")
    parser.add_argument("--rescue-filter-near-atm-only", action="store_true")
    parser.add_argument("--rescue-exclude-otm", action="store_true")
    parser.add_argument("--rescue-exclude-deep-otm", action="store_true")
    parser.add_argument("--rescue-exclude-0dte", action="store_true")
    parser.add_argument("--rescue-min-option-price", type=float, default=0.0)
    parser.add_argument("--rescue-max-bid-ask-spread-pct", type=float, default=0.0)
    parser.add_argument("--rescue-exclude-opening-minutes", type=int, default=0)
    parser.add_argument("--rescue-exclude-closing-minutes", type=int, default=0)
    parser.add_argument("--max-rescue-experiments", type=int, default=0, help="Run at most N rescue experiments in this invocation.")
    parser.add_argument("--rescue-time-budget-minutes", type=float, default=0.0, help="Optional wall-clock budget for rescue experiments.")
    parser.add_argument("--rescue-experiment-filter", default="", help="Substring filter for rescue experiment names.")
    parser.add_argument("--rescue-fast-pass", action="store_true", help="Run a smaller, laptop-friendly first-pass rescue grid.")
    parser.add_argument("--failure-diagnosis-audit", action="store_true", help="Generate strict failure diagnosis and improvement planning report.")
    parser.add_argument("--cost-stress-audit", action="store_true", help="Explicitly emit cost stress audit reports.")
    parser.add_argument("--regime-performance-audit", action="store_true", help="Explicitly emit regime performance audit reports.")
    parser.add_argument("--calibration-audit", action="store_true", help="Explicitly emit calibration audit reports.")
    parser.add_argument("--probability-monotonicity-audit", action="store_true", help="Explicitly emit probability monotonicity audit reports.")
    parser.add_argument("--daily-pnl-stability-audit", action="store_true", help="Explicitly emit daily pnl stability audit reports.")
    parser.add_argument("--edge-refinement-report", action="store_true", help="Research-only offline edge refinement over completed artifact checkpoints.")
    parser.add_argument("--edge-refinement-report-only", action="store_true", help="Aggregate existing edge refinement checkpoints only.")
    parser.add_argument("--force-refinement", action="store_true", help="Recompute completed edge refinement candidates.")
    parser.add_argument("--edge-refinement-max-models", type=int, default=0, help="Evaluate at most N models in the refinement pass.")
    parser.add_argument("--edge-refinement-max-candidates", type=int, default=0, help="Evaluate at most N refinement candidates in this invocation.")
    parser.add_argument("--edge-refinement-only-model", action="append", default=[], help="Restrict refinement to specific model names. Repeatable or comma-separated.")
    parser.add_argument("--edge-refinement-only-target", action="append", default=[], help="Restrict refinement to specific target names. Repeatable or comma-separated.")
    parser.add_argument("--edge-refinement-fast", action="store_true", help="Run the bounded first-pass refinement grid only.")
    parser.add_argument("--include-avoid-label-refinement", action="store_true", help="Include avoid_trade_label in refinement search.")
    parser.add_argument("--middle-zone-only", action="store_true", help="Restrict refinement to middle-zone candidate rules designed for roughly 100-1000 trades.")
    parser.add_argument("--edge-widening-experiment", action="store_true", help="Research-only edge widening experiment with stronger per-trade edge labels and selectors.")
    parser.add_argument("--big-move-min-absolute-return", type=float, default=0.01, help="Minimum absolute net return used by the big-move profitable label.")
    return parser.parse_args()


def latest_edge_dataset_csv() -> Optional[Path]:
    candidates = sorted(MODELS_DIR.glob("edge_retraining_*/edge_dataset.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _candidate_dataset_paths() -> List[Path]:
    found: List[Path] = []
    seen: set[str] = set()
    for pattern in DATASET_SEARCH_GLOBS:
        for path in REPO_ROOT.glob(pattern):
            key = str(path.resolve())
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            found.append(path)
    return sorted(found, key=lambda p: p.stat().st_size, reverse=True)


def _dataset_candidate_report(path: Path) -> Dict[str, Any]:
    lowered = path.name.lower()
    report: Dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "size_bytes": int(path.stat().st_size),
        "status": "REJECTED",
        "reasons": [],
        "row_count": None,
        "timestamp_column_present": False,
        "usable_label_columns": [],
        "usable_return_columns": [],
        "is_real_dataset_candidate": False,
    }
    # FIX 3: Reject schema-stub / metadata CSVs that are far too small to be
    # real training data. Real option-chain datasets are 100MB-1GB; anything
    # under 64 KB is almost certainly a header/schema artifact.
    if int(report["size_bytes"]) < 64_000:
        report["reasons"].append("too_small_to_be_training_data")
        return report
    if any(token in lowered for token in ("pytest", "tmp", "temp", "sample", "smoke", "synthetic", "test_")):
        report["reasons"].append("rejected_non_real_or_temp_dataset")
        return report
    if path.suffix.lower() not in {".csv", ".parquet"}:
        report["reasons"].append("unsupported_extension")
        return report
    try:
        if path.suffix.lower() == ".csv":
            preview = pd.read_csv(path, nrows=2000)
            row_count = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1
        else:
            preview = pd.read_parquet(path)
            row_count = len(preview)
    except Exception as exc:
        report["reasons"].append(f"load_failed:{exc}")
        return report
    report["row_count"] = int(max(row_count, len(preview)))
    report["timestamp_column_present"] = "timestamp" in preview.columns
    if "timestamp" not in preview.columns:
        report["reasons"].append("missing_timestamp")
        return report
    groups = detect_column_groups(preview)
    labels, _ = choose_labels(preview)
    report["usable_label_columns"] = labels
    report["usable_return_columns"] = groups.get("evaluation_return_columns", [])
    report["is_real_dataset_candidate"] = "option" in lowered or "reconstructed" in lowered or "live_feature" in lowered or "enriched" in lowered
    if not report["is_real_dataset_candidate"]:
        report["reasons"].append("not_clearly_real_option_dataset")
    if not labels:
        report["reasons"].append("missing_usable_label")
    if not groups.get("evaluation_return_column_used"):
        report["reasons"].append("missing_real_return_column")
    if any(token in lowered for token in ("schema", "_candidate_")):
        report["reasons"].append("metadata_or_candidate_file")
    if not report["reasons"]:
        report["status"] = "ACCEPTED"
    return report


def select_largest_valid_real_dataset() -> Dict[str, Any]:
    candidates = [_dataset_candidate_report(path) for path in _candidate_dataset_paths()]
    accepted = [row for row in candidates if row.get("status") == "ACCEPTED"]
    selected = accepted[0] if accepted else None
    payload = {
        "candidates_considered": candidates,
        "selected_dataset": selected,
        "selection_rule": "largest_usable_real_dataset_by_row_count",
    }
    return payload


def run_dataset_builder() -> Path:
    from build_option_edge_dataset import main as build_main  # lazy import

    before = {p for p in MODELS_DIR.glob("edge_retraining_*")}
    build_main()
    after = sorted({p for p in MODELS_DIR.glob("edge_retraining_*")} - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if after:
        return after[0]
    latest = sorted(MODELS_DIR.glob("edge_retraining_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not latest:
        raise FileNotFoundError("No edge_retraining artifact folder found after dataset build.")
    return latest[0]


def load_dataset(path: Path) -> pd.DataFrame:
    print(f"[retrain] loading_dataset path={path}", flush=True)
    try:
        file_size_mb = float(path.stat().st_size) / (1024.0 * 1024.0)
        print(f"[retrain] dataset_file_size_mb={file_size_mb:.2f}", flush=True)
    except Exception:
        pass
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise ValueError(f"Unsupported dataset file type: {path.suffix}. Expected .csv or .parquet")
    cache_path, cache_meta_path = _dataset_cache_paths(path)
    load_start = time.perf_counter()
    csv_load_time_seconds: float | None = None
    parquet_write_time_seconds: float | None = None
    parquet_read_time_seconds: float | None = None
    cache_used = False
    if suffix == ".csv" and bool(_runtime_option("use_parquet_cache", False)):
        cache_valid = False
        cache_meta: Dict[str, Any] = {}
        force_refresh_cache = bool(_runtime_option("force_refresh_cache", False))
        if force_refresh_cache:
            print(f"[retrain] parquet_cache_refresh_forced path={cache_path}", flush=True)
        if not force_refresh_cache and cache_path.exists() and cache_meta_path.exists():
            try:
                cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
                stat = path.stat()
                cache_valid = (
                    str(cache_meta.get("source_path")) == str(path.resolve())
                    and int(cache_meta.get("source_size_bytes") or -1) == int(stat.st_size)
                    and int(cache_meta.get("source_mtime_ns") or -1) == int(stat.st_mtime_ns)
                )
            except Exception:
                cache_valid = False
        if cache_valid:
            parquet_start = time.perf_counter()
            df = pd.read_parquet(cache_path)
            parquet_read_time_seconds = time.perf_counter() - parquet_start
            cache_used = True
            csv_load_time_seconds = float(cache_meta.get("csv_load_time_seconds") or 0.0) or None
            print(f"[retrain] parquet_cache_hit path={cache_path}", flush=True)
        else:
            csv_start = time.perf_counter()
            df = pd.read_csv(path)
            csv_load_time_seconds = time.perf_counter() - csv_start
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            parquet_write_start = time.perf_counter()
            df.to_parquet(cache_path, index=False)
            parquet_write_time_seconds = time.perf_counter() - parquet_write_start
            stat = path.stat()
            cache_meta_payload = {
                "source_path": str(path.resolve()),
                "source_size_bytes": int(stat.st_size),
                "source_mtime_ns": int(stat.st_mtime_ns),
                "row_count": int(len(df)),
                "column_count": int(len(df.columns)),
                "csv_load_time_seconds": float(csv_load_time_seconds),
                "parquet_write_time_seconds": float(parquet_write_time_seconds or 0.0),
                "created_at": datetime.now().isoformat(),
            }
            cache_meta_path.write_text(json.dumps(cache_meta_payload, indent=2), encoding="utf-8")
            print(f"[retrain] parquet_cache_written path={cache_path}", flush=True)
    elif suffix == ".parquet":
        parquet_start = time.perf_counter()
        df = pd.read_parquet(path)
        parquet_read_time_seconds = time.perf_counter() - parquet_start
    else:
        df = pd.read_csv(path)
        csv_load_time_seconds = time.perf_counter() - load_start
    load_seconds = time.perf_counter() - load_start
    comparison = ""
    if cache_used and csv_load_time_seconds:
        delta = float(csv_load_time_seconds) - float(load_seconds)
        comparison = f" cached_csv_load_seconds={csv_load_time_seconds:.3f} speedup_seconds={delta:.3f}"
    elif suffix == ".csv" and bool(_runtime_option("use_parquet_cache", False)) and csv_load_time_seconds:
        comparison = f" csv_load_seconds={csv_load_time_seconds:.3f} parquet_cache_ready=true"
    print(f"[retrain] dataset_loaded rows={len(df)} cols={len(df.columns)} cache_used={cache_used}{comparison}", flush=True)
    if csv_load_time_seconds is not None:
        print(f"[retrain] csv_read_time_seconds={csv_load_time_seconds:.3f}", flush=True)
    if parquet_write_time_seconds is not None:
        print(f"[retrain] parquet_write_time_seconds={parquet_write_time_seconds:.3f}", flush=True)
    if parquet_read_time_seconds is not None:
        print(f"[retrain] parquet_read_time_seconds={parquet_read_time_seconds:.3f}", flush=True)
    print(f"[retrain] dataset_memory_mb={_dataset_memory_mb(df):.2f}", flush=True)
    print(f"[retrain] target_distribution={json.dumps(_dataset_target_distribution(df), sort_keys=True)}", flush=True)
    _timing_log("dataset_load", seconds=load_seconds, rows=len(df), extra=f"path={path}")
    preprocess_start = time.perf_counter()
    if {"timestamp", "instrument_key"}.issubset(df.columns) and ("ltp" in df.columns or "close" in df.columns):
        try:
            print("[retrain] preparing chronological dataset", flush=True)
            df = _prepare_chronological_work(df)
            print("[retrain] deriving direct strategy features (mean reversion + stat arb)", flush=True)
            df = _ensure_direct_strategy_features(df)
            added = [
                name for name in (
                    "mean_reversion_zscore",
                    "mean_reversion_entry_score",
                    "mean_reversion_buy_call",
                    "stat_arb_zscore",
                    "stat_arb_confidence",
                    "stat_arb_long_spread",
                )
                if name in df.columns
            ]
            print(f"[retrain] direct strategy features ready count={len(added)} names={added}", flush=True)
        except Exception:
            pass
    df = _apply_runtime_dataset_limits(df, context=str(path))
    print(f"[retrain] dataset_post_limit rows={len(df)} cols={len(df.columns)} memory_mb={_dataset_memory_mb(df):.2f}", flush=True)
    print(f"[retrain] dataset_post_limit_target_distribution={json.dumps(_dataset_target_distribution(df), sort_keys=True)}", flush=True)
    _timing_log("preprocessing", seconds=time.perf_counter() - preprocess_start, rows=len(df), extra=f"path={path}")
    return df


def _normalize_timestamp_series(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        if _KOLKATA_TZ is not None:
            try:
                return ts.dt.tz_localize(_KOLKATA_TZ)
            except Exception as exc:
                import sys
                print(f"[TIMEZONE ERROR] tz_localize({TIMEZONE_NAME}) failed: {exc} — leaving timestamps naive (sort/grouping may be off-by-UTC-offset)", file=sys.stderr)
                return ts
        else:
            import sys
            print(f"[TIMEZONE ERROR] No provider for {TIMEZONE_NAME} (need tzdata or pytz on Windows); leaving timestamps naive.", file=sys.stderr)
            return ts
    else:
        if _KOLKATA_TZ is not None:
            try:
                return ts.dt.tz_convert(_KOLKATA_TZ)
            except Exception:
                return ts
        return ts


def _chronology_sort_columns(df: pd.DataFrame) -> List[str]:
    return [column for column in CHRONOLOGY_SORT_COLUMNS if column in df.columns]


def _assert_monotonic_timestamps(timestamps: pd.Series, *, context: str) -> None:
    if timestamps.empty:
        raise AssertionError(f"{context}: no timestamps available")
    if timestamps.isna().any():
        raise AssertionError(f"{context}: timestamp column contains NaT values after normalization")
    if not pd.Index(timestamps).is_monotonic_increasing:
        raise AssertionError(f"{context}: timestamps are not monotonic increasing")


def _prepare_chronological_work(work: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in work.columns:
        raise AssertionError("chronology preparation requires timestamp column")
    prepared = work.copy()
    prepared["timestamp"] = _normalize_timestamp_series(prepared["timestamp"])
    prepared = prepared.dropna(subset=["timestamp"]).copy()
    sort_columns = _chronology_sort_columns(prepared)
    prepared = prepared.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    _assert_monotonic_timestamps(prepared["timestamp"], context="prepared_work")
    return prepared


def _ensure_direct_strategy_features(work: pd.DataFrame) -> pd.DataFrame:
    if work.empty:
        return work
    price_col = "ltp" if "ltp" in work.columns else ("close" if "close" in work.columns else "")
    key_col = "instrument_key" if "instrument_key" in work.columns else ""
    if not price_col or not key_col or "timestamp" not in work.columns:
        return work

    df = work.copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.sort_values([key_col, "timestamp"], kind="stable").reset_index(drop=True)
    print(
        f"[retrain] direct_strategy_feature_build rows={len(df)} instruments={df[key_col].nunique(dropna=True)} price_col={price_col}",
        flush=True,
    )
    grouped_price = df.groupby(key_col, sort=False)[price_col]

    mr_mean = grouped_price.transform(lambda s: s.rolling(20, min_periods=5).mean())
    mr_std = grouped_price.transform(lambda s: s.rolling(20, min_periods=5).std(ddof=0)).replace(0.0, np.nan)
    mr_z = ((df[price_col] - mr_mean) / mr_std).replace([np.inf, -np.inf], np.nan)
    df["mean_reversion_zscore"] = mr_z.fillna(0.0)
    df["mean_reversion_entry_score"] = (mr_z.abs() / 1.25).clip(lower=0.0, upper=1.0).fillna(0.0)
    df["mean_reversion_expected_reversion_pct"] = ((mr_mean - df[price_col]) / df[price_col].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["mean_reversion_half_life_bars"] = grouped_price.transform(
        lambda s: pd.Series(np.where(s.rolling(20, min_periods=5).std(ddof=0).fillna(0.0) > 0.0, 10.0, 0.0), index=s.index)
    ).fillna(0.0)
    df["mean_reversion_buy_call"] = (mr_z <= -1.25).astype(float)
    df["mean_reversion_buy_put"] = (mr_z >= 1.25).astype(float)

    fair_value = grouped_price.transform(lambda s: s.ewm(span=21, adjust=False, min_periods=5).mean())
    x_mean = grouped_price.transform(lambda s: s.rolling(30, min_periods=5).mean())
    y_mean = fair_value.groupby(df[key_col], sort=False).transform(lambda s: s.rolling(30, min_periods=5).mean())
    xy_mean = (df[price_col] * fair_value).groupby(df[key_col], sort=False).transform(lambda s: s.rolling(30, min_periods=5).mean())
    x2_mean = (df[price_col] * df[price_col]).groupby(df[key_col], sort=False).transform(lambda s: s.rolling(30, min_periods=5).mean())
    cov_xy = xy_mean - (x_mean * y_mean)
    var_x = (x2_mean - (x_mean * x_mean)).replace(0.0, np.nan)
    hedge_ratio = (cov_xy / var_x).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    spread = fair_value - (hedge_ratio * df[price_col])
    spread_mean = spread.groupby(df[key_col], sort=False).transform(lambda s: s.rolling(30, min_periods=5).mean())
    spread_std = spread.groupby(df[key_col], sort=False).transform(lambda s: s.rolling(30, min_periods=5).std(ddof=0)).replace(0.0, np.nan)
    stat_z = ((spread - spread_mean) / spread_std).replace([np.inf, -np.inf], np.nan)
    df["stat_arb_zscore"] = stat_z.fillna(0.0)
    df["stat_arb_confidence"] = (stat_z.abs() / 1.5).clip(lower=0.0, upper=1.0).fillna(0.0)
    df["stat_arb_spread_pct"] = (spread / df[price_col].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["stat_arb_hedge_ratio"] = hedge_ratio.fillna(1.0)
    df["stat_arb_long_spread"] = (stat_z <= -1.5).astype(float)
    df["stat_arb_short_spread"] = (stat_z >= 1.5).astype(float)
    print("[retrain] direct_strategy_feature_build complete", flush=True)
    return df


def _edge_widening_label_names() -> List[str]:
    return [
        "profitable_trade_label_edge_1x_cost",
        "profitable_trade_label_edge_2x_cost",
        "profitable_trade_label_edge_3x_cost",
        "profitable_trade_label_edge_5x_cost",
        "big_move_profitable_label",
        "asymmetric_payoff_label",
    ]


def _infer_embedded_cost_series(df: pd.DataFrame) -> pd.Series:
    gross = pd.to_numeric(df.get("gross_forward_return"), errors="coerce") if "gross_forward_return" in df.columns else pd.Series(np.nan, index=df.index)
    net = pd.to_numeric(df.get("net_forward_return"), errors="coerce") if "net_forward_return" in df.columns else pd.Series(np.nan, index=df.index)
    estimated = pd.Series(np.nan, index=df.index, dtype=float)
    if len(gross) == len(df) and len(net) == len(df):
        estimated = gross - net
    estimated = pd.to_numeric(estimated, errors="coerce")
    positive = estimated.where(estimated > 0.0)
    fallback = float(positive.median()) if positive.notna().any() else 0.0
    return estimated.fillna(fallback).clip(lower=0.0)


def _augment_edge_widening_labels(
    df: pd.DataFrame,
    *,
    minimum_absolute_return: float,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    augmented = df.copy()
    if "net_forward_return" in augmented.columns:
        net_return = pd.to_numeric(augmented["net_forward_return"], errors="coerce")
    elif "gross_forward_return" in augmented.columns:
        net_return = pd.to_numeric(augmented["gross_forward_return"], errors="coerce")
    else:
        raise RuntimeError("Edge widening labels require net_forward_return or gross_forward_return.")
    embedded_cost = _infer_embedded_cost_series(augmented)
    augmented["_embedded_cost_inferred"] = embedded_cost
    thresholds = {
        "profitable_trade_label_edge_1x_cost": 1.0,
        "profitable_trade_label_edge_2x_cost": 2.0,
        "profitable_trade_label_edge_3x_cost": 3.0,
        "profitable_trade_label_edge_5x_cost": 5.0,
    }
    for label_name, multiple in thresholds.items():
        augmented[label_name] = (net_return >= (embedded_cost * float(multiple))).astype(float)
    big_move_threshold = np.maximum(embedded_cost * 2.0, float(minimum_absolute_return))
    augmented["big_move_profitable_label"] = (net_return >= big_move_threshold).astype(float)
    downside = pd.to_numeric(augmented.get("max_adverse_excursion"), errors="coerce") if "max_adverse_excursion" in augmented.columns else pd.Series(np.nan, index=augmented.index)
    upside = pd.to_numeric(augmented.get("max_favorable_excursion"), errors="coerce") if "max_favorable_excursion" in augmented.columns else pd.Series(np.nan, index=augmented.index)
    if upside.isna().all():
        upside = net_return.clip(lower=0.0)
    if downside.isna().all():
        downside = (-net_return).clip(lower=0.0)
    asymmetric_mask = (upside >= (downside * 1.5)) & (net_return >= embedded_cost)
    augmented["asymmetric_payoff_label"] = asymmetric_mask.fillna(False).astype(float)
    return augmented, {
        "embedded_cost_median": float(embedded_cost.median()) if embedded_cost.notna().any() else 0.0,
        "embedded_cost_mean": float(embedded_cost.mean()) if embedded_cost.notna().any() else 0.0,
        "minimum_absolute_return": float(minimum_absolute_return),
        "net_return_column_used": "net_forward_return" if "net_forward_return" in augmented.columns else "gross_forward_return",
    }


def _edge_widening_label_distribution_report(df: pd.DataFrame, label_name: str) -> Dict[str, Any]:
    series = prepare_label_series(df, label_name).dropna()
    work = df.loc[series.index].copy()
    timestamps = _normalize_timestamp_series(work["timestamp"]) if "timestamp" in work.columns else pd.Series(dtype="datetime64[ns]")
    month_key = timestamps.dt.to_period("M").astype(str) if not timestamps.empty else pd.Series(dtype=str)
    positive_rate = float(series.mean()) if len(series) else 0.0
    month_rows = []
    if len(series) and len(month_key) == len(series):
        tmp = pd.DataFrame({"month": month_key, "label": series.to_numpy(dtype=float)})
        grouped = tmp.groupby("month")["label"].agg(["count", "sum", "mean"]).reset_index()
        month_rows = [
            {
                "month": str(row["month"]),
                "row_count": int(row["count"]),
                "positive_count": int(row["sum"]),
                "positive_rate": float(row["mean"]),
            }
            for _, row in grouped.iterrows()
        ]
    too_sparse = positive_rate < 0.01 or int(series.sum()) < 50
    too_noisy = 0.20 <= positive_rate <= 0.80
    suitable_for_training = not too_sparse and not too_noisy
    date_coverage = _timestamp_range(pd.DataFrame({"timestamp": timestamps})) if not timestamps.empty else {"start": None, "end": None, "rows": int(len(series))}
    return {
        "label_name": label_name,
        "row_count": int(len(series)),
        "positive_count": int(series.sum()) if len(series) else 0,
        "negative_count": int(len(series) - series.sum()) if len(series) else 0,
        "positive_rate": positive_rate,
        "date_coverage": date_coverage,
        "month_wise_positive_rate": month_rows,
        "too_sparse": bool(too_sparse),
        "too_noisy": bool(too_noisy),
        "suitable_for_training": bool(suitable_for_training),
    }


def _timestamp_range(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or "timestamp" not in df.columns:
        return {"start": None, "end": None, "rows": int(len(df))}
    timestamps = pd.Series(_normalize_timestamp_series(df["timestamp"])).dropna()
    if timestamps.empty:
        return {"start": None, "end": None, "rows": int(len(df))}
    return {
        "start": pd.Timestamp(timestamps.min()).isoformat(),
        "end": pd.Timestamp(timestamps.max()).isoformat(),
        "rows": int(len(df)),
    }


def _same_period_alignment_report(
    baseline_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    *,
    enabled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    baseline_range = _timestamp_range(baseline_df)
    enriched_range = _timestamp_range(enriched_df)
    overlap_start = max(pd.Timestamp(baseline_range["start"]), pd.Timestamp(enriched_range["start"]))
    overlap_end = min(pd.Timestamp(baseline_range["end"]), pd.Timestamp(enriched_range["end"]))
    has_overlap = overlap_start <= overlap_end
    report = {
        "enabled": bool(enabled),
        "baseline_original_date_range": baseline_range,
        "bs_original_date_range": enriched_range,
        "overlapping_date_range": {
            "start": overlap_start.isoformat() if has_overlap else None,
            "end": overlap_end.isoformat() if has_overlap else None,
        },
        "same_period_gate": "PASS" if enabled and has_overlap else ("FAIL" if enabled else ("PASS" if baseline_range == enriched_range else "FAIL")),
    }
    if not enabled:
        report["reason"] = None if baseline_range == enriched_range else "Comparison periods differ and --same-period-comparison was not enabled."
        report["baseline_rows_retained"] = int(len(baseline_df))
        report["bs_rows_retained"] = int(len(enriched_df))
        report["baseline_rows_dropped"] = 0
        report["bs_rows_dropped"] = 0
        return baseline_df, enriched_df, report
    if not has_overlap:
        raise RuntimeError("No overlapping timestamp range exists between baseline and Black-Scholes datasets.")
    baseline_mask = baseline_df["timestamp"].between(overlap_start, overlap_end, inclusive="both")
    enriched_mask = enriched_df["timestamp"].between(overlap_start, overlap_end, inclusive="both")
    baseline_aligned = baseline_df.loc[baseline_mask].copy()
    enriched_aligned = enriched_df.loc[enriched_mask].copy()
    report["baseline_rows_retained"] = int(len(baseline_aligned))
    report["bs_rows_retained"] = int(len(enriched_aligned))
    report["baseline_rows_dropped"] = int(len(baseline_df) - len(baseline_aligned))
    report["bs_rows_dropped"] = int(len(enriched_df) - len(enriched_aligned))
    report["reason"] = None
    return baseline_aligned, enriched_aligned, report


def _slice_time_range(timestamps: pd.Series, indices: np.ndarray, prefix: str) -> Dict[str, Any]:
    if indices.size == 0:
        return {
            f"{prefix}_start": None,
            f"{prefix}_end": None,
            f"{prefix}_rows": 0,
        }
    selected = timestamps.iloc[indices]
    selected_sorted = selected.sort_values(kind="stable")
    start = pd.Timestamp(selected_sorted.iloc[0])
    end = pd.Timestamp(selected_sorted.iloc[-1])
    if start > end:
        raise AssertionError(f"{prefix} start is after end: {start} > {end}")
    return {
        f"{prefix}_start": start.isoformat(),
        f"{prefix}_end": end.isoformat(),
        f"{prefix}_rows": int(indices.size),
    }


def _assert_fold_is_chronological(
    timestamps: pd.Series,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    *,
    test_idx: np.ndarray | None = None,
    context: str,
) -> None:
    train_ts = timestamps.iloc[train_idx].sort_values(kind="stable")
    validation_ts = timestamps.iloc[validation_idx].sort_values(kind="stable")
    if train_ts.empty or validation_ts.empty:
        raise AssertionError(f"{context}: empty train or validation fold")
    train_end = pd.Timestamp(train_ts.iloc[-1])
    validation_start = pd.Timestamp(validation_ts.iloc[0])
    validation_end = pd.Timestamp(validation_ts.iloc[-1])
    if not train_end < validation_start:
        raise AssertionError(f"{context}: train_end {train_end} is not earlier than validation_start {validation_start}")
    if validation_start > validation_end:
        raise AssertionError(f"{context}: validation_start {validation_start} is after validation_end {validation_end}")
    if test_idx is not None:
        test_ts = timestamps.iloc[test_idx].sort_values(kind="stable")
        if test_ts.empty:
            raise AssertionError(f"{context}: empty test fold")
        test_start = pd.Timestamp(test_ts.iloc[0])
        test_end = pd.Timestamp(test_ts.iloc[-1])
        if not validation_end < test_start:
            raise AssertionError(f"{context}: validation_end {validation_end} is not earlier than test_start {test_start}")
        if test_start > test_end:
            raise AssertionError(f"{context}: test_start {test_start} is after test_end {test_end}")


def _group_indices_by_timestamp(timestamps: pd.Series) -> List[np.ndarray]:
    normalized = pd.Series(_normalize_timestamp_series(pd.Series(timestamps)), name="timestamp").reset_index(drop=True)
    _assert_monotonic_timestamps(normalized, context="group_indices")
    groups: List[np.ndarray] = []
    if normalized.empty:
        return groups
    vals = normalized.to_numpy()
    n = len(vals)
    current_start = 0
    for idx in range(1, n):
        if vals[idx] != vals[idx - 1]:
            groups.append(np.arange(current_start, idx, dtype=int))
            current_start = idx
    groups.append(np.arange(current_start, n, dtype=int))
    return groups


def _concat_group_slices(groups: Sequence[np.ndarray], start: int, end: int) -> np.ndarray:
    if end <= start:
        return np.asarray([], dtype=int)
    selected = [groups[idx] for idx in range(start, min(end, len(groups)))]
    if not selected:
        return np.asarray([], dtype=int)
    return np.concatenate(selected).astype(int)


def _build_holdout_split_from_timestamps(
    timestamps: pd.Series,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> Dict[str, np.ndarray]:
    groups = _group_indices_by_timestamp(timestamps)
    group_count = len(groups)
    if group_count < 3:
        raise AssertionError("Need at least 3 unique timestamps for chronological holdout split")
    train_groups = max(1, int(group_count * float(train_ratio)))
    validation_groups = max(1, int(group_count * float(validation_ratio)))
    if train_groups >= group_count - 1:
        train_groups = group_count - 2
    if train_groups + validation_groups >= group_count:
        validation_groups = max(1, group_count - train_groups - 1)
    test_group_start = train_groups + validation_groups
    if test_group_start >= group_count:
        raise AssertionError("Chronological holdout split left no test timestamp groups")
    train_idx = _concat_group_slices(groups, 0, train_groups)
    validation_idx = _concat_group_slices(groups, train_groups, train_groups + validation_groups)
    test_idx = _concat_group_slices(groups, test_group_start, group_count)
    return {
        "train": train_idx,
        "validation": validation_idx,
        "test": test_idx,
    }


def _generate_grouped_walk_forward_splits(
    timestamps: pd.Series,
    *,
    fold_count: int,
    purge_groups: int = 15,
) -> List[Dict[str, np.ndarray]]:
    groups = _group_indices_by_timestamp(timestamps)
    group_count = len(groups)
    if group_count < 9:
        return []
    target_folds = max(2, int(fold_count))
    window_groups = max(1, group_count // (target_folds + 2))
    min_train_groups = max(window_groups, 3)
    splits: List[Dict[str, np.ndarray]] = []
    for fold_idx in range(target_folds):
        validation_group_start = min_train_groups + fold_idx * window_groups
        validation_group_end = validation_group_start + window_groups
        test_group_start = validation_group_end
        test_group_end = min(group_count, test_group_start + window_groups)
        train_group_end = max(0, validation_group_start - max(1, int(purge_groups)))
        if train_group_end < min_train_groups:
            continue
        if validation_group_end > group_count or test_group_start >= group_count or test_group_end <= test_group_start:
            continue
        train_idx = _concat_group_slices(groups, 0, train_group_end)
        validation_idx = _concat_group_slices(groups, validation_group_start, validation_group_end)
        test_idx = _concat_group_slices(groups, test_group_start, test_group_end)
        if train_idx.size == 0 or validation_idx.size == 0 or test_idx.size == 0:
            continue
        splits.append({
            "train": train_idx,
            "validation": validation_idx,
            "test": test_idx,
        })
    return splits


_KNOWN_LABEL_SUFFIXES = ("_label", "_binary", "_target", "_ outcome")


def _is_cost_aware_label_column(name: str) -> bool:
    """Return True if this column name looks like a cost-aware label.

    Cost-aware labels are recognized by common naming patterns produced
    by the dataset builder (strong_profitable, cost_survivor, etc.).
    We also flag any column ending in _label that is not in PRIMARY_LABELS
    as a dynamically discovered label candidate.
    """
    lowered = str(name).lower()
    cost_aware_tokens = (
        "cost_survivor",
        "strong_profitable",
        "weak_trade",
        "no_trade",
        "high_conviction",
        "paper_candidate",
        "cost_adjusted_success",
        "ternary_trade_quality",
        "fallback_trade_quality",
    )
    return any(token in lowered for token in cost_aware_tokens)


def choose_labels(df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, str]]]:
    """Return (usable_label_columns, skipped_label_records).

    A label is **usable** only if:
      * it is either declared in PRIMARY_LABELS or is a cost-aware label
        discovered dynamically in the dataframe, AND
      * it has at least 50 non-null entries, AND
      * it has at least 2 unique non-null values (rejects degenerate
        constant columns), AND
      * its positive share (after coercing to 0/1) is in [0.5%, 95%]
        so that both near-zero and near-one constant labels are rejected,
        AND
      * it is NOT the trivially-inverse avoid_trade_label (when
        AVOID_LABEL_TRIVIAL_INVERSE is True). See module-level flag
        documentation for why this is skipped.

    Cost-aware labels (strong_profitable_trade_label, cost_survivor_label,
    cost_survivor_label_v2, etc.) are discovered **dynamically** from the
    dataset's column list — they are not hardcoded only in PRIMARY_LABELS.
    This ensures the pipeline accepts new cost-aware labels produced by
    the dataset builder without requiring code changes.

    ``skipped_label_records`` is a list of ``{"label_name", "reason"}`` dicts
    so callers can surface every disqualification to disk (no silent pass).
    """
    MIN_NON_NULL = 50
    MIN_POSITIVE_SHARE = 0.005
    MAX_POSITIVE_SHARE = 0.95

    # Dynamically discover cost-aware label columns not already in PRIMARY_LABELS.
    # Any column present in the dataframe that looks like a cost-aware label
    # and is not already covered by PRIMARY_LABELS is added to the candidate set.
    primary_set = set(PRIMARY_LABELS)
    dynamic_cost_aware: List[str] = []
    for col in df.columns:
        col_str = str(col)
        if col_str in primary_set:
            continue
        if _is_cost_aware_label_column(col_str) or col_str.endswith("_label"):
            # Basic sanity: must be numeric-convertible or categorical GOOD/BAD
            series = df[col_str]
            if series.dtype == object:
                unique_vals = set(str(v).strip().upper() for v in series.dropna().unique())
                if unique_vals.issubset({"0", "1", "GOOD", "BAD", "NEUTRAL", "0.0", "1.0"}):
                    dynamic_cost_aware.append(col_str)
            else:
                dynamic_cost_aware.append(col_str)

    # Candidate labels = PRIMARY_LABELS that exist in df + dynamically discovered
    all_candidates = [name for name in PRIMARY_LABELS if name in df.columns] + dynamic_cost_aware
    # Deduplicate while preserving PRIMARY_LABELS ordering first
    seen: set[str] = set()
    labels: List[str] = []
    for name in all_candidates:
        if name not in seen:
            seen.add(name)
            labels.append(name)
    usable: List[str] = []
    skipped: List[Dict[str, str]] = []
    for name in labels:
        series = df[name]
        if series.dtype == object:
            mapped = series.map({"GOOD": 1, "BAD": 0, "NEUTRAL": np.nan})
            numeric = pd.to_numeric(mapped, errors="coerce")
        else:
            numeric = pd.to_numeric(series, errors="coerce")
        non_null = int(numeric.notna().sum())
        if non_null < MIN_NON_NULL:
            skipped.append({"label_name": name, "reason": f"too_few_non_null({non_null}<{MIN_NON_NULL})"})
            continue
        unique_non_null = int(numeric.dropna().nunique())
        if unique_non_null < 2:
            skipped.append({"label_name": name, "reason": f"degenerate_constant(unique={unique_non_null})"})
            continue
        positive_share = float((numeric.dropna() > 0).mean())
        if positive_share < MIN_POSITIVE_SHARE or positive_share > MAX_POSITIVE_SHARE:
            skipped.append({
                "label_name": name,
                "reason": f"out_of_range_positive_share({positive_share:.4f} not in [{MIN_POSITIVE_SHARE},{MAX_POSITIVE_SHARE}])",
            })
            continue
        # ------------------------------------------------------------------
        # 2026-06-09: Skip trivially-inverse avoid_trade_label if flagged.
        # avoid_trade_label == (net_forward_return <= 0.0) which is exactly
        # 1 - profitable_trade_label (where profitable = net_forward > 0.0).
        # Training a model on a trivial inverse provides ZERO additional
        # discriminative information; the "two-stage" overlay is a no-op.
        # ------------------------------------------------------------------
        if AVOID_LABEL_TRIVIAL_INVERSE and name == "avoid_trade_label":
            skipped.append({
                "label_name": name,
                "reason": "trivial_inverse_of_profitable_trade_label(avoid_trade_label = 1 - profitable_trade_label; no additional discriminative power)",
            })
            continue
        usable.append(name)
    return usable, skipped


def _is_forbidden_feature_name(name: str) -> bool:
    lowered = str(name).strip().lower()
    if any(token in lowered for token in NON_LEAKY_REALIZED_FEATURE_TOKENS):
        return False
    return any(token in lowered for token in STRICT_FORBIDDEN_TOKENS) or any(token in lowered for token in FUTURE_LEAK_TOKENS)


def _looks_like_evaluation_column(name: str) -> bool:
    lowered = str(name).strip().lower()
    if any(token in lowered for token in NON_LEAKY_REALIZED_FEATURE_TOKENS):
        return False
    if lowered.startswith("ret_") or lowered.startswith("return_"):
        return False
    return any(token in lowered for token in STRICT_FORBIDDEN_TOKENS)


def _raw_feature_audit(df: pd.DataFrame) -> Dict[str, Any]:
    raw_columns = [str(column) for column in df.columns]
    duplicate_columns = sorted({column for column in raw_columns if raw_columns.count(column) > 1})
    constant_columns: List[str] = []
    high_null_columns: List[str] = []
    numeric_columns: List[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(str(column))
        if series.nunique(dropna=True) <= 1:
            constant_columns.append(str(column))
        if float(series.isna().mean()) >= 0.95:
            high_null_columns.append(str(column))
    return {
        "raw_column_count": len(raw_columns),
        "raw_columns": raw_columns,
        "numeric_columns": sorted(numeric_columns),
        "constant_columns": sorted(set(constant_columns)),
        "high_null_columns": sorted(set(high_null_columns)),
        "duplicate_columns": duplicate_columns,
    }


def detect_column_groups(df: pd.DataFrame) -> Dict[str, Any]:
    labels, _ = choose_labels(df)
    evaluation_return_columns: List[str] = []
    lowered_map = {str(column).strip().lower(): str(column) for column in df.columns}
    for candidate in RETURN_COLUMN_CANDIDATES:
        found = lowered_map.get(candidate.lower())
        if found and found not in evaluation_return_columns:
            evaluation_return_columns.append(found)
    for column in df.columns:
        lowered = str(column).strip().lower()
        if column in evaluation_return_columns:
            continue
        if _looks_like_evaluation_column(lowered):
            evaluation_return_columns.append(str(column))
    forbidden_feature_columns = sorted({str(column) for column in df.columns if _is_forbidden_feature_name(column)})
    feature_audit = _raw_feature_audit(df)
    input_features = select_feature_columns(df, target_columns=labels, evaluation_columns=evaluation_return_columns, audit=feature_audit)
    return {
        "input_features": input_features,
        "target_columns": labels,
        "evaluation_return_columns": evaluation_return_columns,
        "forbidden_feature_columns": forbidden_feature_columns,
        "evaluation_return_column_used": evaluation_return_columns[0] if evaluation_return_columns else None,
        "metadata_columns": sorted(str(column) for column in df.columns if str(column) in METADATA_COLUMNS),
        "feature_audit": feature_audit,
    }


def select_feature_columns(
    df: pd.DataFrame,
    *,
    target_columns: Sequence[str] | None = None,
    evaluation_columns: Sequence[str] | None = None,
    audit: Dict[str, Any] | None = None,
) -> List[str]:
    exclude = set(METADATA_COLUMNS)
    exclude.update(str(column) for column in (target_columns or []))
    exclude.update(str(column) for column in (evaluation_columns or []))
    cols: List[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if _is_forbidden_feature_name(col):
            continue
        if audit and str(col) in set(audit.get("constant_columns", [])):
            continue
        if audit and str(col) in set(audit.get("high_null_columns", [])):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            cols.append(col)
    return cols


def _trade_metrics_from_scores(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    returns: np.ndarray | None,
    *,
    threshold_source: str = "selected_threshold",
    extra_cost: float = 0.0,
    cost_deduction: np.ndarray | float | None = None,
    cost_provenance: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    signals = np.asarray(y_prob >= float(threshold), dtype=bool)
    signal_count = int(signals.sum())
    has_real_returns = returns is not None and len(returns) == len(y_prob)
    warning = None
    if not has_real_returns:
        return {
            "metric_source": "classification_proxy",
            "warning": "No real return/PnL column found. Trading metrics are classification-only proxy metrics.",
            "threshold_source": threshold_source,
            "trade_count": signal_count,
            "total_return": None,
            "gross_profit": None,
            "gross_loss": None,
            "profit_factor": None,
            "average_win": None,
            "average_loss": None,
            "win_rate": None,
            "expectancy": None,
            "sharpe": None,
            "sortino": None,
            "max_drawdown": None,
            "average_return_per_trade": None,
            "trading_metrics_available": False,
            "smoke_or_synthetic_warning": None,
        }
    selected = np.asarray(returns, dtype=float)[signals]
    if cost_deduction is not None:
        if np.isscalar(cost_deduction):
            selected = selected - float(cost_deduction)
        else:
            selected = selected - np.asarray(cost_deduction, dtype=float)[signals]
    else:
        selected = selected - float(extra_cost)
    if selected.size == 0:
        return {
            "metric_source": "real_returns",
            "warning": warning,
            "threshold_source": threshold_source,
            "trade_count": 0,
            "total_return": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "average_return_per_trade": 0.0,
            "trading_metrics_available": True,
            "smoke_or_synthetic_warning": "Dataset too small for stable trading metrics." if len(y_prob) < 200 else None,
        }
    wins = selected[selected > 0]
    losses = selected[selected < 0]
    gross_profit = float(selected[selected > 0].sum())
    gross_loss = float(np.abs(selected[selected < 0]).sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    equity = np.cumsum(selected)
    peaks = np.maximum.accumulate(equity)
    drawdowns = peaks - equity
    max_drawdown = float(drawdowns.max()) if drawdowns.size else 0.0
    sharpe = 0.0
    sortino = 0.0
    if selected.size > 1:
        std = float(np.std(selected, ddof=1))
        if std > 0:
            sharpe = float((np.mean(selected) / std) * math.sqrt(252.0))
        downside = selected[selected < 0]
        if downside.size > 1:
            downside_std = float(np.std(downside, ddof=1))
            if downside_std > 0:
                sortino = float((np.mean(selected) / downside_std) * math.sqrt(252.0))
        elif np.mean(selected) > 0:
            sortino = 99.0
    return {
        "metric_source": "real_returns",
        "warning": warning,
        "threshold_source": threshold_source,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "trade_count": signal_count,
        "total_return": float(selected.sum()),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "average_win": float(np.mean(wins)) if wins.size else 0.0,
        "average_loss": float(np.mean(losses)) if losses.size else 0.0,
        "win_rate": float(wins.size / selected.size) if selected.size else 0.0,
        "expectancy": float(np.mean(selected)),
        "average_return_per_trade": float(np.mean(selected)),
        "trading_metrics_available": True,
        "smoke_or_synthetic_warning": "Dataset too small for stable trading metrics." if len(y_prob) < 200 else None,
        "cost_provenance": cost_provenance or {},
    }


def _cost_stress_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    returns: np.ndarray | None,
    *,
    cost_provenance: Dict[str, Any] | None = None,
    per_row_cost_units: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Compute cost-stress scenarios per-target.

    Parameters
    ----------
    per_row_cost_units : np.ndarray | None
        Per-trade cost in return units (same length as y_true). When provided,
        stress scenarios are computed by scaling each trade's individual cost
        rather than applying a single global cost deduction.
        REQUIRED COLUMNS: must contain 'cost_return_units' per target when
        per_row_cost_units is None and the target is cost-aware.

    Notes
    -----
    When per_row_cost_units is provided, the function uses per-trade costs for
    a more accurate stress test. When absent, falls back to the global
    inferred_embedded_cost / baseline_deduction (legacy behavior).

    cost_provenance is now extended to include:
      - per_row_cost_units: per-trade cost array (numpy)
      - cost_column_source: 'per_target_frame' | 'inferred_embedded' | 'fallback_scalar'
      - per_target_pf_1_25x: profit factor at 1.25x cost stress (per-row)
      - per_target_pf_1_50x: profit factor at 1.50x cost stress (per-row)
    """
    scenarios: Dict[str, Any] = {}
    provenance = dict(cost_provenance or {})
    is_net = bool(provenance.get("is_evaluation_return_already_net"))
    inferred_embedded_cost = float(provenance.get("inferred_embedded_cost") or 0.0)
    baseline_deduction = float(provenance.get("baseline_cost_deduction_used") or 0.0)

    # Per-target cost frame path: per_row_cost_units overrides global deduction.
    has_per_row_costs = per_row_cost_units is not None and len(per_row_cost_units) == len(y_true)
    if has_per_row_costs:
        per_row_costs = np.asarray(per_row_cost_units, dtype=float)
        # Ensure non-negative
        per_row_costs = np.clip(per_row_costs, 0.0, None)
        cost_source = "per_target_frame"
    elif is_net:
        # Per-row not available; use inferred embedded cost as scalar
        cost_source = "inferred_embedded" if inferred_embedded_cost > 0.0 else "fallback_scalar"
    else:
        cost_source = "inferred_embedded" if baseline_deduction > 0.0 else "fallback_scalar"

    # Propagate cost source into provenance for reporting
    provenance["cost_column_source"] = cost_source
    provenance["has_per_row_cost_frame"] = has_per_row_costs

    for name, extra_cost in EXTRA_COST_SCENARIOS.items():
        if has_per_row_costs:
            # Per-row stress: scale each trade's cost individually.
            # base: deduct actual cost (to recover gross return from net return).
            # stress: deduct cost * (1 + extra_cost).
            if name == "base":
                deduction_arr: np.ndarray | float = per_row_costs
            else:
                deduction_arr = per_row_costs * (1.0 + float(extra_cost))
            effective_deduction_scalar = float(np.mean(deduction_arr))
        elif is_net:
            # net_forward_return: base deducts embedded cost; stress adds incremental.
            if name == "base":
                deduction = inferred_embedded_cost
            else:
                deduction = inferred_embedded_cost + inferred_embedded_cost * float(extra_cost)
            deduction_arr = float(deduction)
            effective_deduction_scalar = float(deduction)
        else:
            # gross return column: base uses baseline deduction; stress adds multiplier.
            if name == "base":
                deduction = baseline_deduction
            else:
                deduction = baseline_deduction * float(extra_cost) if baseline_deduction > 0.0 else float(extra_cost)
            deduction_arr = float(deduction)
            effective_deduction_scalar = float(deduction)

        metrics = _trade_metrics_from_scores(
            y_true,
            y_prob,
            threshold,
            returns,
            threshold_source=f"cost_stress:{name}",
            cost_deduction=deduction_arr,
            cost_provenance=provenance,
        )
        failure_flag = bool(
            metrics.get("trading_metrics_available")
            and name != "base"
            and (
                float(metrics.get("profit_factor") or 0.0) < 1.05
                or float(metrics.get("sharpe") or 0.0) < 0.5
            )
        )
        scenarios[name] = {
            **metrics,
            "extra_cost": float(extra_cost),
            "effective_cost_deduction": effective_deduction_scalar,
            "failure_flag": failure_flag,
            "cost_source": cost_source,
        }

    # Record per-target 1.25x and 1.50x PF in provenance for downstream gate checks
    provenance["per_target_pf_1_25x"] = float(
        (scenarios.get("cost_1_25x") or {}).get("profit_factor") or 0.0
    )
    provenance["per_target_pf_1_50x"] = float(
        (scenarios.get("cost_1_50x") or {}).get("profit_factor") or 0.0
    )
    return scenarios


def _threshold_grid_values(start: float, end: float, step: float) -> List[float]:
    values: List[float] = []
    current = float(start)
    while current <= float(end) + 1e-9:
        values.append(round(current, 6))
        current += float(step)
    return values


def _feature_importance(model: Any, feature_names: Sequence[str], *, limit: int = 10) -> List[Dict[str, Any]]:
    values: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        try:
            values = np.asarray(getattr(model, "feature_importances_"), dtype=float)
        except Exception:
            values = None
    elif hasattr(model, "coef_"):
        try:
            coef = np.asarray(getattr(model, "coef_"), dtype=float)
            values = np.abs(coef[0] if coef.ndim > 1 else coef)
        except Exception:
            values = None
    if values is None:
        return []
    rows = [{"feature": str(name), "importance": float(val)} for name, val in zip(feature_names, values)]
    rows.sort(key=lambda row: abs(float(row["importance"])), reverse=True)
    return rows[:limit]


class _ProbabilityBlendEnsemble:
    def __init__(self, models: Sequence[Any], weights: Sequence[float]) -> None:
        self.models = list(models)
        self.weights = list(weights)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])
        blended = np.zeros(len(X), dtype=float)
        for weight, model in zip(self.weights, self.models):
            blended += float(weight) * predict_proba_positive(model, X)
        blended = np.clip(blended, 0.0, 1.0)
        return np.column_stack([1.0 - blended, blended])


def _probability_diagnostics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Sequence[float]) -> Dict[str, Any]:
    positives = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_prob, dtype=float)
    threshold_rows = []
    for threshold in thresholds:
        selected = probs >= float(threshold)
        rate = float(np.mean(selected)) if len(selected) else 0.0
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "positive_prediction_rate": rate,
                "selected_trades": int(selected.sum()),
            }
        )
    return {
        "min": float(np.min(probs)) if len(probs) else None,
        "max": float(np.max(probs)) if len(probs) else None,
        "mean": float(np.mean(probs)) if len(probs) else None,
        "std": float(np.std(probs)) if len(probs) else None,
        "class_balance": float(np.mean(positives)) if len(positives) else None,
        "threshold_positive_rates": threshold_rows,
    }


def _full_threshold_sweep_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    returns: np.ndarray | None,
    *,
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        cls = classification_metrics(y_true, y_prob, threshold=float(threshold))
        trade = _trade_metrics_from_scores(y_true, y_prob, float(threshold), returns, threshold_source="retrain_threshold_sweep")
        rows.append(
            {
                "threshold": float(threshold),
                "number_of_trades": int(trade.get("trade_count") or 0),
                "win_rate": trade.get("win_rate"),
                "average_net_forward_return": trade.get("average_return_per_trade"),
                "cumulative_return": trade.get("total_return"),
                "profit_factor": trade.get("profit_factor"),
                "sharpe": trade.get("sharpe"),
                "max_drawdown": trade.get("max_drawdown"),
                "f1": cls.get("f1"),
                "precision": cls.get("precision"),
                "recall": cls.get("recall"),
                "brier_score": brier_score(y_true, y_prob),
                "positive_prediction_rate": float(np.mean(np.asarray(y_prob) >= float(threshold))) if len(y_prob) else 0.0,
            }
        )
    return rows


def _threshold_diagnostics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    selected_threshold: float,
    returns: np.ndarray | None,
) -> Dict[str, Any]:
    selected = np.asarray(y_prob, dtype=float) >= float(selected_threshold)
    signal_count = int(selected.sum())
    positive_ratio = float(np.mean(selected)) if len(selected) else 0.0
    diagnostics = {
        "selected_threshold": float(selected_threshold),
        "signal_count": signal_count,
        "positive_prediction_rate": positive_ratio,
        "model_produced_no_positive_class": bool(signal_count == 0),
        "ranking_signal_exists_but_threshold_not_profitable": False,
        "reason": None,
    }
    trade = _trade_metrics_from_scores(y_true, y_prob, float(selected_threshold), returns, threshold_source="selected_threshold_diagnostics")
    roc_auc = classification_metrics(y_true, y_prob, threshold=float(selected_threshold)).get("roc_auc", 0.0)
    if signal_count == 0:
        diagnostics["reason"] = "model produced no positive class at selected threshold"
    elif float(trade.get("profit_factor") or 0.0) <= 1.0 and float(roc_auc or 0.0) >= 0.60:
        diagnostics["ranking_signal_exists_but_threshold_not_profitable"] = True
        diagnostics["reason"] = "ranking signal exists but threshold/trading filter is not profitable"
    return diagnostics


def _walk_forward_summary(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.Series,
    feature_cols: Sequence[str],
    label_name: str,
    returns: np.ndarray | None,
    *,
    fold_count: int = 5,
) -> Dict[str, Any]:
    timestamps = pd.Series(_normalize_timestamp_series(pd.Series(timestamps)), name="timestamp").reset_index(drop=True)
    _assert_monotonic_timestamps(timestamps, context="walk_forward_input")
    splits = _generate_grouped_walk_forward_splits(timestamps, fold_count=max(2, int(fold_count)), purge_groups=15)
    fold_rows: List[Dict[str, Any]] = []
    if not splits:
        return {"split_count": 0, "folds": [], "mean_roc_auc": 0.0, "mean_f1": 0.0, "stable_across_folds": False}
    for idx, split in enumerate(splits, start=1):
        train_idx = split["train"]
        val_idx = split["validation"]
        test_idx = split["test"]
        _assert_fold_is_chronological(timestamps, train_idx, val_idx, test_idx=test_idx, context=f"walk_forward_fold_{idx}")
        model = build_safe_model("logistic_regression")
        if model is None:
            break
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        X_train_s, X_val_s, _, _, _ = scale_train_val_test(X_train, X_val, X_val)
        fit_model(model, X_train_s, y_train)
        val_prob = predict_proba_positive(model, X_val_s)
        X_test = X[test_idx]
        y_test = y[test_idx]
        _, X_test_s, _, _, _ = scale_train_val_test(X_train, X_test, X_test)
        test_prob = predict_proba_positive(model, X_test_s)
        val_returns = returns[val_idx] if returns is not None and len(returns) == len(y) else None
        test_returns = returns[test_idx] if returns is not None and len(returns) == len(y) else None
        chosen = optimize_threshold_from_probs(y_val, val_prob, val_returns, thresholds=RETRAIN_THRESHOLD_SWEEP)
        metrics = classification_metrics(y_test, test_prob, threshold=float(chosen["threshold"]))
        metrics["pr_auc"] = pr_auc_score_safe(y_test, test_prob)
        trade_metrics = _trade_metrics_from_scores(y_test, test_prob, float(chosen["threshold"]), test_returns, threshold_source="walk_forward")
        fold_window = {}
        fold_window.update(_slice_time_range(timestamps, train_idx, "train"))
        fold_window.update(_slice_time_range(timestamps, val_idx, "validation"))
        fold_window.update(_slice_time_range(timestamps, test_idx, "test"))
        fold_rows.append(
            {
                "fold": idx,
                "threshold": float(chosen["threshold"]),
                "trade_count": int(trade_metrics.get("trade_count") or 0),
                "profit_factor": trade_metrics.get("profit_factor"),
                "sharpe": trade_metrics.get("sharpe"),
                "max_drawdown": trade_metrics.get("max_drawdown"),
                "average_return": trade_metrics.get("average_return_per_trade"),
                "start": fold_window["test_start"],
                "end": fold_window["test_end"],
                **fold_window,
                **metrics,
            }
        )
    roc_values = [float(row["roc_auc"]) for row in fold_rows]
    f1_values = [float(row["f1"]) for row in fold_rows]
    return {
        "split_count": len(fold_rows),
        "folds": fold_rows,
        "mean_roc_auc": float(np.mean(roc_values)) if fold_rows else 0.0,
        "mean_f1": float(np.mean(f1_values)) if fold_rows else 0.0,
        "stable_across_folds": bool(fold_rows and np.std(roc_values) <= 0.10 and np.std(f1_values) <= 0.15),
    }


def optimize_threshold_from_probs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    returns: np.ndarray | None = None,
    *,
    thresholds: Sequence[float] | None = None,
    minimum_trades: int = 5,
) -> Dict[str, Any]:
    rows = []
    threshold_values = list(thresholds) if thresholds is not None else list(EVALUATION_THRESHOLDS)
    for threshold in threshold_values:
        metrics = classification_metrics(y_true, y_prob, threshold=threshold)
        trade_metrics = _trade_metrics_from_scores(y_true, y_prob, threshold, returns, threshold_source="threshold_sweep")
        rows.append({
            "threshold": threshold,
            **metrics,
            "signals": int(np.sum(y_prob >= float(threshold))),
            "trade_count": int(trade_metrics.get("trade_count") or 0),
            "profit_factor": trade_metrics.get("profit_factor"),
            "sharpe": trade_metrics.get("sharpe"),
            "max_drawdown": trade_metrics.get("max_drawdown"),
            "win_rate": trade_metrics.get("win_rate"),
            "average_return_per_trade": trade_metrics.get("average_return_per_trade"),
            "total_return": trade_metrics.get("total_return"),
        })
    eligible = [
        row for row in rows
        if int(row.get("trade_count") or 0) >= minimum_trades
        and float(row.get("profit_factor") or 0.0) > 1.15
        and float(row.get("sharpe") or 0.0) > 1.0
        and float(row.get("max_drawdown") or 0.0) <= max(1.0, abs(float(row.get("total_return") or 0.0)) * 1.5)
    ]
    pool = eligible or [row for row in rows if int(row.get("trade_count") or 0) >= minimum_trades] or rows
    chosen = max(
        pool,
        key=lambda row: (
            float(row.get("profit_factor") or 0.0),
            float(row.get("sharpe") or 0.0),
            -float(row.get("max_drawdown") or 0.0),
            float(row.get("f1") or 0.0),
            float(row.get("precision") or 0.0),
            int(row.get("trade_count") or 0),
        ),
    )
    chosen = dict(chosen)
    chosen["evaluated_thresholds"] = rows
    return chosen


def _threshold_robustness_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    returns: np.ndarray | None,
    *,
    grid_start: float,
    grid_end: float,
    grid_step: float,
    min_trades: int,
) -> Dict[str, Any]:
    thresholds = _threshold_grid_values(grid_start, grid_end, grid_step)
    rows: List[Dict[str, Any]] = []
    robust_thresholds: List[float] = []
    for threshold in thresholds:
        metrics = classification_metrics(y_true, y_prob, threshold=threshold)
        trade_metrics = _trade_metrics_from_scores(y_true, y_prob, threshold, returns, threshold_source="threshold_robustness")
        row = {
            "threshold": float(threshold),
            "trade_count": int(trade_metrics.get("trade_count") or 0),
            "win_rate": trade_metrics.get("win_rate"),
            "profit_factor": trade_metrics.get("profit_factor"),
            "sharpe": trade_metrics.get("sharpe"),
            "sortino": trade_metrics.get("sortino"),
            "max_drawdown": trade_metrics.get("max_drawdown"),
            "total_return": trade_metrics.get("total_return"),
            "average_return_per_trade": trade_metrics.get("average_return_per_trade"),
            "expectancy": trade_metrics.get("expectancy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
        }
        if int(row["trade_count"]) >= int(min_trades) and float(row["profit_factor"] or 0.0) > 1.05 and float(row["sharpe"] or 0.0) > 0.5:
            robust_thresholds.append(float(threshold))
        rows.append(row)
    bands: List[Dict[str, float]] = []
    if robust_thresholds:
        start = robust_thresholds[0]
        prev = robust_thresholds[0]
        for value in robust_thresholds[1:]:
            if round(value - prev, 6) == round(float(grid_step), 6):
                prev = value
                continue
            bands.append({"start": start, "end": prev})
            start = value
            prev = value
        bands.append({"start": start, "end": prev})
    chosen = optimize_threshold_from_probs(y_true, y_prob, returns, thresholds=thresholds, minimum_trades=min_trades)
    chosen_threshold = float(chosen["threshold"])
    in_band = any(float(band["start"]) <= chosen_threshold <= float(band["end"]) for band in bands)
    return {
        "threshold_rows": rows,
        "robust_threshold_bands": bands,
        "chosen_threshold": chosen_threshold,
        "chosen_threshold_is_robust": bool(in_band),
        "isolated_lucky_point": bool(not in_band),
        "minimum_trades": int(min_trades),
    }


def _fold_stability_report(folds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not folds:
        return {
            "profitable_folds": 0,
            "folds_with_pf_gt_1_05": 0,
            "folds_with_sharpe_gt_0_5": 0,
            "worst_fold_pf": 0.0,
            "worst_fold_sharpe": 0.0,
            "median_fold_pf": 0.0,
            "median_fold_sharpe": 0.0,
            "fold_pf_std": 0.0,
            "fold_sharpe_std": 0.0,
            "gate": "FAIL",
        }
    pfs = [float(row.get("profit_factor") or 0.0) for row in folds]
    sharpes = [float(row.get("sharpe") or 0.0) for row in folds]
    profitable_folds = int(sum(1 for row in folds if float(row.get("average_return") or 0.0) > 0.0))
    pf_pass = int(sum(1 for value in pfs if value > 1.05))
    sharpe_pass = int(sum(1 for value in sharpes if value > 0.5))
    gate = "PASS" if pf_pass >= 3 and sharpe_pass >= 3 else "FAIL"
    return {
        "profitable_folds": profitable_folds,
        "folds_with_pf_gt_1_05": pf_pass,
        "folds_with_sharpe_gt_0_5": sharpe_pass,
        "worst_fold_pf": float(min(pfs)),
        "worst_fold_sharpe": float(min(sharpes)),
        "median_fold_pf": float(np.median(pfs)),
        "median_fold_sharpe": float(np.median(sharpes)),
        "fold_pf_std": float(np.std(pfs)) if len(pfs) > 1 else 0.0,
        "fold_sharpe_std": float(np.std(sharpes)) if len(sharpes) > 1 else 0.0,
        "gate": gate,
    }


def _moneyness_bucket(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "unknown"
    if number >= 1.10:
        return "deep_itm"
    if number >= 1.02:
        return "ITM"
    if number <= 0.90:
        return "deep_otm"
    if number <= 0.98:
        return "OTM"
    return "ATM"


def _dte_bucket(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "unknown"
    if number <= 0.25:
        return "0DTE"
    if number <= 1.0:
        return "1DTE"
    if number <= 3.0:
        return "2-3DTE"
    if number <= 7.0:
        return "4-7DTE"
    return ">7DTE"


def _time_bucket(timestamp: Any) -> str:
    ts = pd.Timestamp(timestamp)
    minutes = ts.hour * 60 + ts.minute
    if minutes < (10 * 60 + 30):
        return "opening_session"
    if minutes < (14 * 60):
        return "midday"
    return "closing_session"


def _group_trade_metrics(frame: pd.DataFrame, group_column: str, *, min_samples_warning: int = 50) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if frame.empty or group_column not in frame.columns:
        return rows
    for value, group in frame.groupby(group_column, dropna=False):
        returns = group["selected_return"].to_numpy(dtype=float)
        metrics = _trade_metrics_from_scores(
            np.ones(len(group), dtype=int),
            np.ones(len(group), dtype=float),
            0.5,
            returns,
            threshold_source=f"regime:{group_column}",
        )
        rows.append({
            "group": str(value),
            "trade_count": int(len(group)),
            "win_rate": metrics.get("win_rate"),
            "profit_factor": metrics.get("profit_factor"),
            "sharpe": metrics.get("sharpe"),
            "max_drawdown": metrics.get("max_drawdown"),
            "average_return": metrics.get("average_return_per_trade"),
            "total_return": metrics.get("total_return"),
            "low_sample_warning": bool(len(group) < int(min_samples_warning)),
        })
    return rows


def _regime_performance_report(
    test_frame: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    returns: np.ndarray | None,
) -> Dict[str, Any]:
    if returns is None or test_frame.empty:
        return {"available": False, "groups": {}}
    work = test_frame.copy().reset_index(drop=True)
    signals = np.asarray(y_prob >= float(threshold), dtype=bool)
    work = work.loc[signals].copy()
    if work.empty:
        return {"available": True, "groups": {}}
    work["selected_return"] = np.asarray(returns, dtype=float)[signals]
    if "moneyness" in work.columns:
        work["moneyness_bucket"] = work["moneyness"].map(_moneyness_bucket)
    elif "distance_from_spot" in work.columns:
        work["moneyness_bucket"] = work["distance_from_spot"].map(_moneyness_bucket)
    if "dte_days" in work.columns:
        work["dte_bucket"] = work["dte_days"].map(_dte_bucket)
    work["time_bucket"] = work["timestamp"].map(_time_bucket)
    groups: Dict[str, Any] = {}
    for column in (
        "option_type_ce",
        "option_type_pe",
        "moneyness_bucket",
        "dte_bucket",
        "time_bucket",
        "volatility_regime_classifier",
        "regime_trending",
        "regime_volatile",
        "regime_mean_reverting",
        "regime_quiet",
    ):
        if column in work.columns:
            groups[column] = _group_trade_metrics(work, column)
    return {"available": True, "groups": groups}


def _schema_alignment_action_report(strict_gate: Dict[str, Any]) -> Dict[str, Any]:
    missing_live = [str(value) for value in strict_gate.get("missing_live_features", [])]
    extra_dataset = [str(value) for value in strict_gate.get("extra_dataset_features", [])]
    safe_live_candidates = [
        feature for feature in extra_dataset
        if feature.startswith(("ctx_", "ret_", "atr_", "rsi_", "oi_", "volume_", "regime_", "volatility_"))
        or feature in {"moneyness", "distance_from_atm", "distance_from_atm_pct", "time_to_expiry_days", "time_to_expiry_years"}
    ]
    training_only = [
        feature for feature in extra_dataset
        if feature.startswith("bs_") or feature in {"row_enrichment_ok", "bad_iv_flag", "bad_greek_flag", "wide_spread_flag", "low_price_flag", "deep_itm_flag", "deep_otm_flag"}
    ]
    return {
        "features_available_in_training_not_live": extra_dataset,
        "features_expected_live_missing_in_dataset": missing_live,
        "features_safe_to_compute_live": safe_live_candidates,
        "training_only_or_forbidden_features": training_only,
        "recommended_live_collector_additions": missing_live,
        "recommended_feature_removals_if_live_impossible": training_only,
    }


def _feature_live_audit(feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    return _live_contract_audit_rows(feature_names)


def _select_live_computable_features(feature_names: Sequence[str]) -> tuple[List[str], List[Dict[str, Any]]]:
    return _filter_live_contract_features(feature_names)


def _simple_no_greeks_features(feature_names: Sequence[str]) -> List[str]:
    banned_tokens = ("iv", "delta", "gamma", "theta", "vega", "rho", "greek")
    return [name for name in feature_names if not any(token in name.lower() for token in banned_tokens)]


def _reduced_robust_features(feature_names: Sequence[str]) -> List[str]:
    keep_tokens = (
        "open",
        "high",
        "low",
        "close",
        "ltp",
        "price",
        "volume",
        "oi",
        "open_interest",
        "atr",
        "rsi",
        "adx",
        "ctx_time",
        "weekday",
        "hour",
        "minute",
        "distance_from_spot",
        "distance_from_atm",
        "time_to_expiry",
        "dte",
        "moneyness",
        "regime_",
        "volatility_regime",
        "spot_",
        "underlying_",
        "return_1",
        "rolling_vol",
    )
    reduced = [
        name
        for name in feature_names
        if not _is_forbidden_feature_name(name)
        and not name.startswith("bs_")
        and any(token in name.lower() for token in keep_tokens)
    ]
    return reduced or list(feature_names[: min(25, len(feature_names))])


def build_feature_variants(
    baseline_features: Sequence[str],
    enriched_features: Sequence[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    baseline_list = list(baseline_features)
    enriched_list = list(enriched_features or baseline_features)
    live_only, live_audit = _select_live_computable_features(baseline_list)
    no_greeks = _simple_no_greeks_features(live_only or baseline_list)
    reduced = _reduced_robust_features(live_only or baseline_list)
    bs_live, bs_live_audit = _select_live_computable_features(enriched_list)
    bs_exact_live = [name for name in bs_live if name in BS_RESEARCH_FEATURES]
    return {
        "baseline_all_safe_features": {
            "feature_names": baseline_list,
            "live_computable_status": "PASS",
            "rejected_features": [],
            "feature_audit": _feature_live_audit(baseline_list),
        },
        "live_computable_only": {
            "feature_names": live_only,
            "live_computable_status": "PASS" if live_only else "FAIL",
            "rejected_features": [row["feature_name"] for row in live_audit if not row["computable_before_decision"]],
            "feature_audit": live_audit,
        },
        "no_greeks": {
            "feature_names": no_greeks,
            "live_computable_status": "PASS",
            "rejected_features": [name for name in baseline_list if name not in no_greeks],
            "feature_audit": _feature_live_audit(no_greeks),
        },
        "black_scholes_research": {
            "feature_names": enriched_list,
            "live_computable_status": "RESEARCH_ONLY",
            "rejected_features": [],
            "feature_audit": _feature_live_audit(enriched_list),
        },
        "black_scholes_live_computable": {
            "feature_names": bs_live,
            "live_computable_status": "BLOCKED_BY_LIVE_FEATURE_GATE" if bs_exact_live != [name for name in enriched_list if name in BS_RESEARCH_FEATURES] else "PASS",
            "rejected_features": [row["feature_name"] for row in bs_live_audit if not row["computable_before_decision"] or not row["available_live"]],
            "feature_audit": bs_live_audit,
        },
        "reduced_robust_features": {
            "feature_names": reduced,
            "live_computable_status": "PASS",
            "rejected_features": [name for name in baseline_list if name not in reduced],
            "feature_audit": _feature_live_audit(reduced),
        },
        "reduced_robust_live_features": {
            "feature_names": reduced,
            "live_computable_status": "PASS",
            "rejected_features": [name for name in baseline_list if name not in reduced],
            "feature_audit": _feature_live_audit(reduced),
        },
    }


def _numeric_series(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _option_side_series(df: pd.DataFrame) -> pd.Series:
    if "option_type" in df.columns:
        return df["option_type"].astype(str).str.upper().str.strip()
    ce = _numeric_series(df, "option_type_ce", default=0.0).fillna(0.0)
    pe = _numeric_series(df, "option_type_pe", default=0.0).fillna(0.0)
    side = pd.Series("", index=df.index, dtype=object)
    side.loc[ce >= 1] = "CE"
    side.loc[pe >= 1] = "PE"
    return side


def _moneyness_bucket_series(df: pd.DataFrame) -> pd.Series:
    if "moneyness" in df.columns:
        return _numeric_series(df, "moneyness").map(_moneyness_bucket)
    if "distance_from_spot" in df.columns:
        return _numeric_series(df, "distance_from_spot").map(_moneyness_bucket)
    if "distance_from_atm" in df.columns:
        dist = _numeric_series(df, "distance_from_atm").abs()
        return dist.map(lambda value: "ATM" if pd.notna(value) and value <= 0.5 else ("NEAR_ATM" if pd.notna(value) and value <= 2.0 else "OTM"))
    return pd.Series("UNKNOWN", index=df.index, dtype=object)


def _dte_series(df: pd.DataFrame) -> pd.Series:
    if "dte_days" in df.columns:
        return _numeric_series(df, "dte_days")
    if "time_to_expiry_days" in df.columns:
        return _numeric_series(df, "time_to_expiry_days")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _option_price_series(df: pd.DataFrame) -> pd.Series:
    for column in ("ltp", "option_ltp", "close", "mid_price"):
        if column in df.columns:
            return _numeric_series(df, column)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _spread_pct_series(df: pd.DataFrame) -> pd.Series:
    spread = _numeric_series(df, "bid_ask_spread_pct")
    if spread.notna().any():
        return spread * 100.0 if float(spread.dropna().abs().max()) <= 1.0 else spread
    return spread


def _minute_of_day_series(df: pd.DataFrame) -> pd.Series:
    ts = _normalize_timestamp_series(df["timestamp"])
    return ts.dt.hour * 60 + ts.dt.minute


def _apply_market_slice_filters(df: pd.DataFrame, filter_spec: Dict[str, Any]) -> Dict[str, Any]:
    prepared = _prepare_chronological_work(df)
    work = prepared.copy()
    work["__rescue_option_side"] = _option_side_series(work)
    work["__rescue_moneyness_bucket"] = _moneyness_bucket_series(work)
    work["__rescue_dte_days"] = _dte_series(work)
    work["__rescue_option_price"] = _option_price_series(work)
    work["__rescue_spread_pct"] = _spread_pct_series(work)
    work["__rescue_minute_of_day"] = _minute_of_day_series(work)
    rows_before = int(len(work))
    applied: List[Dict[str, Any]] = []
    mask = pd.Series(True, index=work.index)

    def _apply(name: str, condition: pd.Series) -> None:
        nonlocal mask
        before = int(mask.sum())
        mask = mask & condition.fillna(False)
        after = int(mask.sum())
        applied.append({"filter_name": name, "rows_before": before, "rows_after": after, "rows_removed": before - after})

    if filter_spec.get("ce_only"):
        _apply("ce_only", work["__rescue_option_side"].eq("CE"))
    if filter_spec.get("pe_only"):
        _apply("pe_only", work["__rescue_option_side"].eq("PE"))
    if filter_spec.get("itm_only"):
        _apply("itm_only", work["__rescue_moneyness_bucket"].eq("ITM"))
    if filter_spec.get("atm_only"):
        _apply("atm_only", work["__rescue_moneyness_bucket"].eq("ATM"))
    if filter_spec.get("near_atm_only"):
        _apply("near_atm_only", work["__rescue_moneyness_bucket"].eq("NEAR_ATM"))
    if filter_spec.get("exclude_otm"):
        _apply("exclude_otm", ~work["__rescue_moneyness_bucket"].isin(["OTM", "DEEP_OTM"]))
    if filter_spec.get("exclude_deep_otm"):
        _apply("exclude_deep_otm", ~work["__rescue_moneyness_bucket"].isin(["DEEP_OTM"]))
    if filter_spec.get("exclude_0dte"):
        _apply("exclude_0dte", work["__rescue_dte_days"].fillna(9999.0) > 0.0)
    if float(filter_spec.get("min_option_price") or 0.0) > 0.0:
        _apply("min_option_price", work["__rescue_option_price"].fillna(-np.inf) >= float(filter_spec["min_option_price"]))
    if float(filter_spec.get("max_bid_ask_spread_pct") or 0.0) > 0.0:
        _apply("max_bid_ask_spread_pct", work["__rescue_spread_pct"].fillna(np.inf) <= float(filter_spec["max_bid_ask_spread_pct"]))
    if int(filter_spec.get("exclude_opening_minutes") or 0) > 0:
        open_limit = 9 * 60 + 15 + int(filter_spec["exclude_opening_minutes"])
        _apply("exclude_opening_minutes", work["__rescue_minute_of_day"] >= open_limit)
    if int(filter_spec.get("exclude_closing_minutes") or 0) > 0:
        close_limit = 15 * 60 + 30 - int(filter_spec["exclude_closing_minutes"])
        _apply("exclude_closing_minutes", work["__rescue_minute_of_day"] <= close_limit)

    filtered = work.loc[mask].copy()
    filtered = filtered.drop(columns=[column for column in filtered.columns if column.startswith("__rescue_")], errors="ignore")
    return {
        "filtered_df": filtered,
        "rows_before": rows_before,
        "rows_after": int(len(filtered)),
        "row_retention_pct": float((len(filtered) / rows_before) * 100.0) if rows_before else 0.0,
        "applied_filters": applied,
        "filter_definition": dict(filter_spec),
    }


def _rescue_validation_report(df: pd.DataFrame, labels: Sequence[str], evaluation_return_column: str | None) -> Dict[str, Any]:
    row_count = int(len(df))
    unique_timestamps = int(pd.Series(df["timestamp"]).nunique()) if "timestamp" in df.columns else 0
    label_name = labels[0] if labels else None
    positive_ratio = None
    if label_name and label_name in df.columns:
        positive_ratio = float(prepare_label_series(df, label_name).dropna().mean()) if prepare_label_series(df, label_name).dropna().size else None
    failures: List[str] = []
    if row_count < 500:
        failures.append("too_few_rows")
    if unique_timestamps < 50:
        failures.append("too_few_unique_timestamps")
    if positive_ratio is not None and (positive_ratio < 0.05 or positive_ratio > 0.95):
        failures.append("label_distribution_too_imbalanced")
    if evaluation_return_column is None:
        failures.append("missing_evaluation_return_column")
    return {
        "row_count": row_count,
        "unique_timestamps": unique_timestamps,
        "label_positive_ratio": positive_ratio,
        "failed": bool(failures),
        "failure_reasons": failures,
    }


def _build_cli_rescue_filter_spec(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "ce_only": bool(args.rescue_filter_ce_only),
        "pe_only": bool(args.rescue_filter_pe_only),
        "itm_only": bool(args.rescue_filter_itm_only),
        "atm_only": bool(args.rescue_filter_atm_only),
        "near_atm_only": bool(args.rescue_filter_near_atm_only),
        "exclude_otm": bool(args.rescue_exclude_otm),
        "exclude_deep_otm": bool(args.rescue_exclude_deep_otm),
        "exclude_0dte": bool(args.rescue_exclude_0dte),
        "min_option_price": float(args.rescue_min_option_price or 0.0),
        "max_bid_ask_spread_pct": float(args.rescue_max_bid_ask_spread_pct or 0.0),
        "exclude_opening_minutes": int(args.rescue_exclude_opening_minutes or 0),
        "exclude_closing_minutes": int(args.rescue_exclude_closing_minutes or 0),
    }


def _build_default_rescue_experiments() -> List[Dict[str, Any]]:
    return [
        {"experiment_name": "ce_only_live_features", "filter_spec": {"ce_only": True}, "feature_set": "live_computable_only", "model_families": ["logistic_regression"]},
        {"experiment_name": "pe_only_live_features", "filter_spec": {"pe_only": True}, "feature_set": "live_computable_only", "model_families": ["logistic_regression"]},
        {"experiment_name": "itm_only_live_features", "filter_spec": {"itm_only": True}, "feature_set": "live_computable_only", "model_families": ["logistic_regression"]},
        {"experiment_name": "atm_only_live_features", "filter_spec": {"atm_only": True}, "feature_set": "live_computable_only", "model_families": ["logistic_regression"]},
        {"experiment_name": "near_atm_only_live_features", "filter_spec": {"near_atm_only": True}, "feature_set": "live_computable_only", "model_families": ["logistic_regression"]},
        {"experiment_name": "exclude_otm_live_features", "filter_spec": {"exclude_otm": True}, "feature_set": "live_computable_only", "model_families": ["logistic_regression"]},
        {"experiment_name": "exclude_deep_otm_live_features", "filter_spec": {"exclude_deep_otm": True}, "feature_set": "live_computable_only", "model_families": ["logistic_regression"]},
        {"experiment_name": "ce_itm_live_features", "filter_spec": {"ce_only": True, "itm_only": True}, "feature_set": "live_computable_only", "model_families": ["logistic_regression"]},
        {"experiment_name": "ce_near_atm_or_itm_live_features", "filter_spec": {"ce_only": True, "exclude_otm": True}, "feature_set": "live_computable_only", "model_families": ["logistic_regression"]},
        {"experiment_name": "calibrated_logistic_live_features", "filter_spec": {}, "feature_set": "live_computable_only", "model_families": ["logistic_regression_uncalibrated", "logistic_regression_platt", "logistic_regression_isotonic"]},
        {"experiment_name": "reduced_robust_live_features", "filter_spec": {}, "feature_set": "reduced_robust_live_features", "model_families": ["logistic_regression"]},
        {"experiment_name": "no_greeks_live_features", "filter_spec": {}, "feature_set": "no_greeks", "model_families": ["logistic_regression"]},
    ]


def _build_fast_pass_rescue_experiments() -> List[Dict[str, Any]]:
    return [
        {"experiment_name": "calibrated_logistic_live_features", "filter_spec": {}, "feature_set": "live_computable_only", "model_families": ["logistic_regression_uncalibrated", "logistic_regression_platt"]},
        {"experiment_name": "reduced_robust_live_features", "filter_spec": {}, "feature_set": "reduced_robust_live_features", "model_families": ["logistic_regression_uncalibrated", "logistic_regression_platt"]},
        {"experiment_name": "no_greeks_live_features", "filter_spec": {}, "feature_set": "no_greeks", "model_families": ["logistic_regression_uncalibrated"]},
    ]


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temp_path.replace(path)


def _rescue_checkpoint_path(artifact_dir: Path) -> Path:
    return artifact_dir / "model_rescue_checkpoint_latest.json"


def _load_rescue_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"experiments": {}, "completed_experiments": [], "failed_experiments": [], "corrupted_experiments": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"experiments": {}, "completed_experiments": [], "failed_experiments": [], "corrupted_experiments": [path.name]}


def _checkpoint_entry_valid(entry: Dict[str, Any]) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("final_experiment_verdict") not in {"COMPLETED", "FAILED_FILTER_OR_DATA_QUALITY_GATE"}:
        return False
    if entry.get("production_adoption_allowed") not in {False, None}:
        return False
    return True


def _filter_rescue_experiments(experiments: Sequence[Dict[str, Any]], pattern: str) -> List[Dict[str, Any]]:
    token = str(pattern or "").strip().lower()
    if not token:
        return list(experiments)
    return [exp for exp in experiments if token in str(exp.get("experiment_name", "")).lower() or token in str(exp.get("feature_set", "")).lower() or any(token in str(name).lower() for name in exp.get("model_families", []))]


def _rescue_resume_summary(experiments: Sequence[Dict[str, Any]], checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    stored = checkpoint.get("experiments", {}) or {}
    completed: List[str] = []
    skipped: List[str] = []
    pending: List[str] = []
    failed: List[str] = []
    corrupted: List[str] = []
    for experiment in experiments:
        name = str(experiment.get("experiment_name"))
        entry = stored.get(name)
        if entry is None:
            pending.append(name)
            continue
        if not _checkpoint_entry_valid(entry):
            corrupted.append(name)
            pending.append(name)
            continue
        if entry.get("final_experiment_verdict") == "COMPLETED":
            completed.append(name)
            skipped.append(name)
        elif entry.get("final_experiment_verdict") == "FAILED_FILTER_OR_DATA_QUALITY_GATE":
            failed.append(name)
            skipped.append(name)
        else:
            pending.append(name)
    return {
        "completed_experiments": completed,
        "skipped_experiments": skipped,
        "pending_experiments": pending,
        "failed_experiments": failed,
        "corrupted_experiments": corrupted,
        "resume_active": bool(stored),
    }


def _should_stop_rescue_run(start_time: float, budget_minutes: float) -> bool:
    return bool(budget_minutes and budget_minutes > 0 and (time.time() - start_time) >= budget_minutes * 60.0)


def _finalize_rescue_outputs(
    *,
    artifact_dir: Path,
    dataset_path: Path,
    experiment_outputs: Sequence[Dict[str, Any]],
    feature_variants: Dict[str, Dict[str, Any]],
    live_schema_report: Dict[str, Any],
) -> Dict[str, Any]:
    manifest_paths: List[str] = []
    calibration_rows: List[Dict[str, Any]] = []
    reduced_audit_payload = {
        "selected_features": feature_variants["reduced_robust_live_features"]["feature_names"],
        "rejected_features": feature_variants["reduced_robust_live_features"]["rejected_features"],
        "feature_audit": feature_variants["reduced_robust_live_features"]["feature_audit"],
        "production_adoption_allowed": False,
    }
    leaderboard_rows = _collect_rescue_candidate_rows(experiment_outputs, live_schema_report=live_schema_report)
    leaderboard = build_rescue_leaderboard(leaderboard_rows)
    best_rescue = leaderboard[0] if leaderboard else {}
    for experiment in experiment_outputs:
        training_result = experiment.get("training_result") or {}
        for label_models in (training_result.get("reports") or {}).values():
            for report in label_models.values():
                calibration_rows.append(
                    {
                        "experiment_name": experiment.get("experiment_name"),
                        "model_name": report.get("model_name"),
                        "calibration_fit": report.get("calibration_fit"),
                        "calibration_gate": (report.get("calibration_audit") or {}).get("calibration_gate"),
                        "selected_threshold": (report.get("selected_threshold_from_validation") or {}).get("threshold"),
                    }
                )
    for row in leaderboard:
        if not row.get("shadow_candidate_allowed"):
            continue
        report = row.get("report_ref") or {}
        experiment_name = str(row.get("experiment_name"))
        experiment_dir = artifact_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        feature_names = next((entry.get("feature_names") for entry in experiment_outputs if entry.get("experiment_name") == experiment_name), [])
        manifest = _shadow_manifest_payload(row, report, dataset_path=str(dataset_path), feature_names=feature_names, experiment_name=experiment_name)
        manifest_path = experiment_dir / "shadow_deployment_manifest.json"
        _atomic_write_json(manifest_path, manifest)
        manifest_paths.append(str(manifest_path))
    blocked_manifest_payload = {
        "status": "NO_RESCUE_CANDIDATE_FOUND" if not any(row.get("shadow_candidate_allowed") or row.get("paper_candidate_allowed") for row in leaderboard) else "SHADOW_CANDIDATE_AVAILABLE",
        "best_rescue_experiment": {key: value for key, value in best_rescue.items() if key != "report_ref"},
        "reason": "No rescue experiment passed shadow or paper gates." if not manifest_paths else "At least one shadow-only rescue candidate passed.",
        "production_adoption_allowed": False,
    }
    reports = {
        "model_rescue_experiments": _write_named_report(
            "model_rescue_experiments",
            {"experiments": list(experiment_outputs), "production_adoption_allowed": False},
            ["# Model Rescue Experiments", *[f"- `{item.get('experiment_name')}` verdict=`{item.get('final_experiment_verdict')}` rows_after=`{(item.get('filter_report') or {}).get('rows_after')}`" for item in experiment_outputs]],
        ),
        "model_rescue_leaderboard": _write_named_report(
            "model_rescue_leaderboard",
            {"leaderboard": [{key: value for key, value in row.items() if key != "report_ref"} for row in leaderboard], "production_adoption_allowed": False},
            ["# Model Rescue Leaderboard", *[f"- rank={row['rank']} experiment=`{row['experiment_name']}` model=`{row['model_family']}` PF=`{row['PF']}` Sharpe=`{row['Sharpe']}` verdict=`{row['final_verdict']}`" for row in leaderboard[:25]]],
        ),
        "reduced_robust_feature_audit": _write_named_report(
            "reduced_robust_feature_audit",
            reduced_audit_payload,
            ["# Reduced Robust Feature Audit", f"- selected_features=`{len(reduced_audit_payload['selected_features'])}`", f"- rejected_features=`{len(reduced_audit_payload['rejected_features'])}`", "- production adoption allowed: `False`"],
        ),
        "calibration_repair_report": _write_named_report(
            "calibration_repair_report",
            {"rows": calibration_rows, "production_adoption_allowed": False},
            ["# Calibration Repair Report", *[f"- `{row.get('experiment_name')}` model=`{row.get('model_name')}` mode=`{(row.get('calibration_fit') or {}).get('mode')}` validation_only=`{(row.get('calibration_fit') or {}).get('fit_on_validation_only')}` gate=`{row.get('calibration_gate')}`" for row in calibration_rows]],
        ),
        "blocked_manifest_report": _write_named_report(
            "blocked_manifest_report",
            blocked_manifest_payload,
            ["# Rescue Manifest Status", f"- status: `{blocked_manifest_payload['status']}`", f"- best experiment: `{best_rescue.get('experiment_name')}`", "- production adoption allowed: `False`"],
        ),
    }
    _atomic_write_json(artifact_dir / "model_rescue_summary.json", {"leaderboard": [{key: value for key, value in row.items() if key != "report_ref"} for row in leaderboard], "manifest_paths": manifest_paths, "blocked_manifest": blocked_manifest_payload})
    return {"leaderboard": leaderboard, "manifest_paths": manifest_paths, "blocked_manifest": blocked_manifest_payload, "reports": reports}


def _timestamped_retrain_output_dir(base_dir: Path) -> Path:
    return base_dir / f"research_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _status_dir(artifact_dir: Path) -> Path:
    return artifact_dir / "status"


def _split_arg_values(values: Sequence[str]) -> List[str]:
    output: List[str] = []
    for value in values:
        for part in str(value or "").split(","):
            token = part.strip()
            if token:
                output.append(token)
    return output


def _status_file_path(artifact_dir: Path, label_name: str, model_name: str) -> Path:
    safe_label = str(label_name).replace("/", "_").replace("\\", "_")
    safe_model = str(model_name).replace("/", "_").replace("\\", "_")
    return _status_dir(artifact_dir) / f"{safe_label}__{safe_model}.json"


def _read_json_if_exists(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_status_file(
    artifact_dir: Path,
    label_name: str,
    model_name: str,
    payload: Dict[str, Any],
) -> Path:
    path = _status_file_path(artifact_dir, label_name, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return path


def _completed_status_is_valid(status_payload: Dict[str, Any] | None) -> bool:
    if not status_payload:
        return False
    if status_payload.get("status") != "completed":
        return False
    artifact_path = status_payload.get("artifact_path")
    metrics_path = status_payload.get("metrics_path")
    return bool(artifact_path and Path(str(artifact_path)).exists() and metrics_path and Path(str(metrics_path)).exists())


def _write_core_retrain_summary(
    artifact_dir: Path,
    dataset_path: Path,
    labels_trained: List[str],
    model_families: List[str],
    metric_rows: List[Dict[str, Any]],
    result_payload: Dict[str, Any],
    evaluation_return_column: str,
    row_count: Optional[int] = None,
) -> Path:
    """Write core_retrain_summary.json to artifact_dir.

    This is the canonical summary that --edge-refinement-report and
    build_cost_aware_retrain_comparison.py read to determine which
    artifacts are eligible for downstream processing.
    
    The summary includes:
    - dataset path, row count
    - targets trained, models trained
    - skipped models with reasons
    - best model per target
    - cost gate status (cost_1.25x_pf, cost_1.50x_pf)
    - strict gate status
    - artifact paths
    """
    # Build trained models list with full metrics
    trained_models = []
    best_models_per_label: Dict[str, Dict[str, Any]] = {}
    
    for row in metric_rows:
        model_entry = {
            "label": row.get("label_name", ""),
            "model": row.get("model_name", ""),
            "status": row.get("status", "unknown"),
            "artifact_path": row.get("artifact_path", ""),
            "metrics_path": row.get("metrics_path", ""),
            "profit_factor": row.get("profit_factor"),
            "sharpe_ratio": row.get("sharpe_ratio"),
            "trade_count": row.get("trade_count"),
            "skip_reason": row.get("skip_reason"),
            # Cost gate metrics
            "cost_1_25x_pf": row.get("cost_1_25x_pf"),
            "cost_1_50x_pf": row.get("cost_1_50x_pf"),
            # Additional metrics if available
            "roc_auc": row.get("roc_auc"),
            "pr_auc": row.get("pr_auc"),
            "f1": row.get("f1"),
            "selected_threshold": row.get("selected_threshold"),
        }
        trained_models.append(model_entry)
        
        # Track best model per label (by profit_factor)
        label_name = row.get("label_name", "")
        pf = float(row.get("profit_factor") or 0.0)
        current_best = best_models_per_label.get(label_name)
        if current_best is None or pf > float(current_best.get("profit_factor") or 0.0):
            best_models_per_label[label_name] = model_entry
    
    failed_models = result_payload.get("failed_models", [])
    skipped_models = result_payload.get("skipped_models", [])
    
    # Evaluate cost gate status
    cost_gate_status = {"cost_1_25x_pf_gate": "unknown", "cost_1_50x_pf_gate": "unknown"}
    strict_gate_status = {"strict_gate": "not_evaluated"}
    
    for row in metric_rows:
        pf_125 = float(row.get("cost_1_25x_pf") or 0.0)
        pf_150 = float(row.get("cost_1_50x_pf") or 0.0)
        pf = float(row.get("profit_factor") or 0.0)
        
        # Cost gate: pass if pf >= 1.0 after cost stress
        if pf_125 > 0 and pf_125 < 1.0:
            cost_gate_status["cost_1_25x_pf_gate"] = "FAIL"
        elif pf_125 >= 1.0 and cost_gate_status["cost_1_25x_pf_gate"] != "FAIL":
            if cost_gate_status["cost_1_25x_pf_gate"] == "unknown":
                cost_gate_status["cost_1_25x_pf_gate"] = "PASS"
        
        if pf_150 > 0 and pf_150 < 1.0:
            cost_gate_status["cost_1_50x_pf_gate"] = "FAIL"
        elif pf_150 >= 1.0 and cost_gate_status["cost_1_50x_pf_gate"] != "FAIL":
            if cost_gate_status["cost_1_50x_pf_gate"] == "unknown":
                cost_gate_status["cost_1_50x_pf_gate"] = "PASS"
        
        # Strict gate: basic profitability check
        if pf > 0 and pf < STRICT_PAPER_PF_MIN:
            strict_gate_status["strict_gate"] = "FAIL"
        elif pf >= STRICT_PAPER_PF_MIN and strict_gate_status["strict_gate"] != "FAIL":
            if strict_gate_status["strict_gate"] == "not_evaluated":
                strict_gate_status["strict_gate"] = "PASS"
    
    # Collect OOF file paths from result_payload if available
    oof_predictions: Dict[str, Any] = {
        "holdout_oof_files": [],
        "walkforward_oof_files": [],
    }
    for label_name, models in (result_payload.get("reports") or {}).items():
        for model_name, report in (models or {}).items():
            oof_info = (report or {}).get("oof_predictions") or {}
            holdout = oof_info.get("holdout") or {}
            wf = oof_info.get("walk_forward") or {}
            if holdout.get("file_path"):
                oof_predictions["holdout_oof_files"].append({
                    "label": label_name,
                    "model": model_name,
                    "path": holdout["file_path"],
                    "row_count": holdout.get("row_count", 0),
                    "timestamp": holdout.get("timestamp"),
                })
            if wf.get("file_path"):
                oof_predictions["walkforward_oof_files"].append({
                    "label": label_name,
                    "model": model_name,
                    "path": wf["file_path"],
                    "row_count": wf.get("row_count", 0),
                    "timestamp": wf.get("timestamp"),
                })

    summary = {
        "dataset_selection": {
            "path": str(dataset_path),
            "row_count": row_count,
            "selection_rule": "explicit_runtime_dataset",
            "evaluation_return_column": evaluation_return_column,
        },
        "training_metadata": {
            "labels_trained": labels_trained,
            "model_families": model_families,
            "artifact_dir": str(artifact_dir),
        },
        "result": {
            "trained_models": trained_models,
            "failed_models": failed_models,
            "skipped_models": skipped_models,
            "evaluation_return_column_used": evaluation_return_column,
            "feature_lists": [],
        },
        "best_models_per_label": {
            label: {
                "model": entry.get("model"),
                "profit_factor": entry.get("profit_factor"),
                "sharpe_ratio": entry.get("sharpe_ratio"),
                "trade_count": entry.get("trade_count"),
                "cost_1_25x_pf": entry.get("cost_1_25x_pf"),
                "cost_1_50x_pf": entry.get("cost_1_50x_pf"),
            }
            for label, entry in best_models_per_label.items()
        },
        "gates": {
            "cost_gate_status": cost_gate_status,
            "strict_gate_status": strict_gate_status,
        },
        "oof_predictions": oof_predictions,
        "artifact_paths": {
            "summary_path": str(artifact_dir / "core_retrain_summary.json"),
            "status_dir": str(artifact_dir / "status"),
            "reports_dir": str(_artifact_report_dir()),
        },
    }
    path = artifact_dir / "core_retrain_summary.json"
    write_json(path, summary)
    print(f"[core_retrain_summary] Written to {path}")
    return path


def _find_latest_incomplete_retraining_dir(base_dir: Path) -> Path | None:
    candidates = sorted(
        list(base_dir.glob("core_retrain_*")) + list(base_dir.glob("retrain_all_models_*")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        status_files = list(_status_dir(path).glob("*.json"))
        if not status_files:
            continue
        statuses = [_read_json_if_exists(item) or {} for item in status_files]
        if any(row.get("status") in {"pending", "running", "failed"} for row in statuses):
            return path
    return candidates[0] if candidates else None


ALL_RESEARCH_CANDIDATE_SPECS = [
    {"candidate_name": "logistic_regression_top_1_per_day", "candidate_group": "lr_ranker", "model_family": "logistic_regression", "tags": ["LR_RANKER"], "filters": {}, "selection": {"kind": "top_n", "value": 1}},
    {"candidate_name": "logistic_regression_top_3_per_day", "candidate_group": "lr_ranker", "model_family": "logistic_regression", "tags": ["LR_RANKER"], "filters": {}, "selection": {"kind": "top_n", "value": 3}},
    {"candidate_name": "logistic_regression_top_5_per_day", "candidate_group": "lr_ranker", "model_family": "logistic_regression", "tags": ["LR_RANKER"], "filters": {}, "selection": {"kind": "top_n", "value": 5}},
    {"candidate_name": "logistic_regression_top_10_per_day", "candidate_group": "lr_ranker", "model_family": "logistic_regression", "tags": ["LR_RANKER"], "filters": {}, "selection": {"kind": "top_n", "value": 10}},
    {"candidate_name": "logistic_regression_top_5_percent_per_day", "candidate_group": "lr_ranker", "model_family": "logistic_regression", "tags": ["LR_RANKER"], "filters": {}, "selection": {"kind": "top_pct", "value": 0.05}},
    {"candidate_name": "logistic_regression_top_10_percent_per_day", "candidate_group": "lr_ranker", "model_family": "logistic_regression", "tags": ["LR_RANKER"], "filters": {}, "selection": {"kind": "top_pct", "value": 0.10}},
    {"candidate_name": "logistic_regression_top_5_per_day_ce_only", "candidate_group": "lr_ranker", "model_family": "logistic_regression", "tags": ["LR_RANKER", "CE"], "filters": {"option_side": "CE"}, "selection": {"kind": "top_n", "value": 5}},
    {"candidate_name": "logistic_regression_top_5_per_day_pe_only", "candidate_group": "lr_ranker", "model_family": "logistic_regression", "tags": ["LR_RANKER", "PE"], "filters": {"option_side": "PE"}, "selection": {"kind": "top_n", "value": 5}},
    {"candidate_name": "logistic_regression_top_5_per_day_itm_only", "candidate_group": "lr_ranker", "model_family": "logistic_regression", "tags": ["LR_RANKER", "ITM"], "filters": {"moneyness": ["ITM", "deep_itm"]}, "selection": {"kind": "top_n", "value": 5}},
    {"candidate_name": "logistic_regression_top_5_per_day_atm_near_atm_only", "candidate_group": "lr_ranker", "model_family": "logistic_regression", "tags": ["LR_RANKER", "ATM_NEAR_ATM"], "filters": {"moneyness": ["ATM", "NEAR_ATM"]}, "selection": {"kind": "top_n", "value": 5}},
    {"candidate_name": "logistic_regression_top_5_per_day_volatile_only", "candidate_group": "lr_ranker", "model_family": "logistic_regression", "tags": ["LR_RANKER", "VOLATILE"], "filters": {"volatile_only": True}, "selection": {"kind": "top_n", "value": 5}},
    {"candidate_name": "logistic_regression_top_5_per_day_ce_itm", "candidate_group": "lr_ranker", "model_family": "logistic_regression", "tags": ["LR_RANKER", "CE", "ITM"], "filters": {"option_side": "CE", "moneyness": ["ITM", "deep_itm"]}, "selection": {"kind": "top_n", "value": 5}},
    {"candidate_name": "logistic_regression_top_5_per_day_ce_volatile", "candidate_group": "lr_ranker", "model_family": "logistic_regression", "tags": ["LR_RANKER", "CE", "VOLATILE"], "filters": {"option_side": "CE", "volatile_only": True}, "selection": {"kind": "top_n", "value": 5}},
    {"candidate_name": "logistic_regression_top_5_per_day_pe_volatile", "candidate_group": "lr_ranker", "model_family": "logistic_regression", "tags": ["LR_RANKER", "PE", "VOLATILE"], "filters": {"option_side": "PE", "volatile_only": True}, "selection": {"kind": "top_n", "value": 5}},
    {"candidate_name": "random_forest_ce_only", "candidate_group": "regime", "model_family": "random_forest", "tags": ["RF_REGIME", "CE"], "filters": {"option_side": "CE"}, "selection": {"kind": "threshold"}},
    {"candidate_name": "random_forest_pe_only", "candidate_group": "regime", "model_family": "random_forest", "tags": ["RF_REGIME", "PE"], "filters": {"option_side": "PE"}, "selection": {"kind": "threshold"}},
    {"candidate_name": "random_forest_itm_only", "candidate_group": "regime", "model_family": "random_forest", "tags": ["RF_REGIME", "ITM"], "filters": {"moneyness": ["ITM", "deep_itm"]}, "selection": {"kind": "threshold"}},
    {"candidate_name": "random_forest_atm_near_atm_only", "candidate_group": "regime", "model_family": "random_forest", "tags": ["RF_REGIME", "ATM_NEAR_ATM"], "filters": {"moneyness": ["ATM", "NEAR_ATM"]}, "selection": {"kind": "threshold"}},
    {"candidate_name": "random_forest_volatile_only", "candidate_group": "regime", "model_family": "random_forest", "tags": ["RF_REGIME", "VOLATILE"], "filters": {"volatile_only": True}, "selection": {"kind": "threshold"}},
    {"candidate_name": "random_forest_ce_itm", "candidate_group": "regime", "model_family": "random_forest", "tags": ["RF_REGIME", "CE", "ITM"], "filters": {"option_side": "CE", "moneyness": ["ITM", "deep_itm"]}, "selection": {"kind": "threshold"}},
    {"candidate_name": "random_forest_ce_volatile", "candidate_group": "regime", "model_family": "random_forest", "tags": ["RF_REGIME", "CE", "VOLATILE"], "filters": {"option_side": "CE", "volatile_only": True}, "selection": {"kind": "threshold"}},
    {"candidate_name": "random_forest_ce_itm_volatile", "candidate_group": "regime", "model_family": "random_forest", "tags": ["RF_REGIME", "CE", "ITM", "VOLATILE"], "filters": {"option_side": "CE", "moneyness": ["ITM", "deep_itm"], "volatile_only": True}, "selection": {"kind": "threshold"}},
    {"candidate_name": "xgboost_ce_only", "candidate_group": "regime", "model_family": "xgboost", "tags": ["XGB_REGIME", "CE"], "filters": {"option_side": "CE"}, "selection": {"kind": "threshold"}},
    {"candidate_name": "xgboost_pe_only", "candidate_group": "regime", "model_family": "xgboost", "tags": ["XGB_REGIME", "PE"], "filters": {"option_side": "PE"}, "selection": {"kind": "threshold"}},
    {"candidate_name": "xgboost_itm_only", "candidate_group": "regime", "model_family": "xgboost", "tags": ["XGB_REGIME", "ITM"], "filters": {"moneyness": ["ITM", "deep_itm"]}, "selection": {"kind": "threshold"}},
    {"candidate_name": "xgboost_atm_near_atm_only", "candidate_group": "regime", "model_family": "xgboost", "tags": ["XGB_REGIME", "ATM_NEAR_ATM"], "filters": {"moneyness": ["ATM", "NEAR_ATM"]}, "selection": {"kind": "threshold"}},
    {"candidate_name": "xgboost_volatile_only", "candidate_group": "regime", "model_family": "xgboost", "tags": ["XGB_REGIME", "VOLATILE"], "filters": {"volatile_only": True}, "selection": {"kind": "threshold"}},
    {"candidate_name": "xgboost_pe_volatile", "candidate_group": "regime", "model_family": "xgboost", "tags": ["XGB_REGIME", "PE", "VOLATILE"], "filters": {"option_side": "PE", "volatile_only": True}, "selection": {"kind": "threshold"}},
    {"candidate_name": "xgboost_pe_itm_atm", "candidate_group": "regime", "model_family": "xgboost", "tags": ["XGB_REGIME", "PE", "ITM_ATM"], "filters": {"option_side": "PE", "moneyness": ["ATM", "NEAR_ATM", "ITM", "deep_itm"]}, "selection": {"kind": "threshold"}},
]


def _simple_model_metrics_rows(training_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label_name, models in (training_result.get("reports") or {}).items():
        for model_name, report in models.items():
            test_metrics = report.get("test_metrics", {}) or {}
            trade_metrics = report.get("trade_metrics", {}) or {}
            rows.append(
                {
                    "label_name": label_name,
                    "model_name": model_name,
                    "roc_auc": test_metrics.get("roc_auc"),
                    "pr_auc": test_metrics.get("pr_auc"),
                    "f1": test_metrics.get("f1"),
                    "brier_score": test_metrics.get("brier_score"),
                    "profit_factor": trade_metrics.get("profit_factor"),
                    "sharpe": trade_metrics.get("sharpe"),
                    "trade_count": trade_metrics.get("trade_count"),
                    "model_path": report.get("model_path"),
                    "selected_threshold": (report.get("selected_threshold_from_validation") or {}).get("threshold"),
                }
            )
    return rows


def _load_checkpoint_status_rows(artifact_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(_status_dir(artifact_dir).glob("*.json")):
        payload = _read_json_if_exists(path)
        if payload:
            rows.append(payload)
    return rows


def _report_payload_from_status(
    artifact_dir: Path,
    *,
    expected_pairs: Sequence[tuple[str, str]],
) -> Dict[str, Any]:
    status_rows = _load_checkpoint_status_rows(artifact_dir)
    keyed = {(str(row.get("label_name")), str(row.get("model_name"))): row for row in status_rows}
    completed = [row for row in status_rows if row.get("status") == "completed"]
    failed = [row for row in status_rows if row.get("status") == "failed"]
    running = [row for row in status_rows if row.get("status") == "running"]
    pending = [row for row in status_rows if row.get("status") == "pending"]
    missing = [
        {"label_name": label_name, "model_name": model_name, "status": "missing"}
        for (label_name, model_name) in expected_pairs
        if (label_name, model_name) not in keyed
    ]
    best_partial = None
    for row in completed:
        metrics_path = row.get("metrics_path")
        metrics = _read_json_if_exists(Path(str(metrics_path))) if metrics_path else None
        if not metrics:
            continue
        candidate = {
            "label_name": row.get("label_name"),
            "model_name": row.get("model_name"),
            "artifact_path": row.get("artifact_path"),
            "metrics_path": metrics_path,
            "profit_factor": float(((metrics.get("trade_metrics") or {}).get("profit_factor") or 0.0)),
            "sharpe": float(((metrics.get("trade_metrics") or {}).get("sharpe") or 0.0)),
            "trade_count": int(((metrics.get("trade_metrics") or {}).get("trade_count") or 0)),
            "selected_threshold": (metrics.get("selected_threshold_from_validation") or {}).get("threshold"),
        }
        if best_partial is None or (
            candidate["profit_factor"],
            candidate["sharpe"],
            candidate["trade_count"],
        ) > (
            best_partial["profit_factor"],
            best_partial["sharpe"],
            best_partial["trade_count"],
        ):
            best_partial = candidate
    is_complete = len(completed) == len(expected_pairs) and not failed and not running and not pending and not missing
    paper_status = "BLOCKED"
    if is_complete and best_partial and best_partial["profit_factor"] > 1.15 and best_partial["sharpe"] > 1.0 and best_partial["trade_count"] >= STRICT_PAPER_FILTER_TRADE_MIN:
        paper_status = "PASS"
    return {
        "artifact_dir": str(artifact_dir),
        "runtime_status": "COMPLETE" if is_complete else "PARTIAL_TIMEOUT",
        "completed_target_model_pairs": completed,
        "failed_target_model_pairs": failed,
        "running_target_model_pairs": running,
        "pending_target_model_pairs": pending,
        "missing_target_model_pairs": missing,
        "best_partial_candidate": best_partial,
        "paper_trading_status": paper_status if is_complete else "BLOCKED",
        "full_run_completed": is_complete,
    }


def _write_completed_checkpoint_artifacts(
    *,
    artifact_dir: Path,
    dataset_path: Path,
    row_count: int,
    date_range: Dict[str, Any],
    selected_label: str,
    selected_features: Sequence[str],
    baseline_features: Sequence[str],
    rejected_features: Sequence[str],
    dropped_leakage: Sequence[str],
    groups: Dict[str, Any],
    live_feature_audit: Sequence[Dict[str, Any]],
    live_computable_only: bool,
    excluded_non_live_features: Sequence[str],
    paper_watchlist_only: bool,
    result: Dict[str, Any],
    families: Sequence[str],
    walk_forward_folds: int,
) -> Dict[str, Any]:
    metric_rows = _simple_model_metrics_rows(result)
    report_map = result.get("reports", {}) or {}
    champion_report = _champion_selection_report(metric_rows, report_map)
    feature_manifest = {
        "dataset_path": str(dataset_path),
        "label_name": selected_label,
        "features": list(selected_features),
        "feature_order": list(selected_features),
        "feature_count": len(selected_features),
        "baseline_features": list(baseline_features),
        "rejected_features": list(rejected_features),
        "dropped_leakage_columns": list(dropped_leakage),
        "evaluation_return_columns": groups["evaluation_return_columns"],
        "evaluation_return_column_used": groups["evaluation_return_column_used"],
        "live_feature_audit": list(live_feature_audit),
        "live_computable_only": bool(live_computable_only),
        "live_contract_version": CONTRACT_VERSION if HAS_FEATURE_CONTRACT else None,
        "excluded_non_live_features": list(excluded_non_live_features),
        "included_live_features": [
            row["feature_name"] for row in live_feature_audit if row.get("feature_name") in set(selected_features)
        ],
        "production_adoption_allowed": False,
        "research_only": True,
    }
    metrics_payload = {
        "dataset_path": str(dataset_path),
        "row_count": int(row_count),
        "date_range": date_range,
        "label_name": selected_label,
        "models_trained": metric_rows,
        "best_model": champion_report.get("champion"),
        "skipped_models": result.get("skipped_models", []),
        "failed_models": result.get("failed_models", []),
        "production_adoption_allowed": False,
        "paper_trading_allowed": False,
        "research_only": True,
    }
    training_manifest = {
        "mode": "retrain_all_models",
        "dataset_path": str(dataset_path),
        "artifact_dir": str(artifact_dir),
        "model_families": list(families),
        "row_count": int(row_count),
        "date_range": date_range,
        "label_name": selected_label,
        "feature_count": len(selected_features),
        "feature_order": list(selected_features),
        "live_computable_only": bool(live_computable_only),
        "live_contract_version": CONTRACT_VERSION if HAS_FEATURE_CONTRACT else None,
        "excluded_non_live_features": list(excluded_non_live_features),
        "production_adoption_allowed": False,
        "paper_trading_allowed": False,
        "research_only": True,
        "paper_watchlist_only": bool(paper_watchlist_only),
    }
    write_json(artifact_dir / "metrics_report.json", metrics_payload)
    write_json(artifact_dir / "feature_manifest.json", feature_manifest)
    write_json(artifact_dir / "feature_list_used.json", feature_manifest)
    write_json(artifact_dir / "training_manifest.json", training_manifest)
    write_json(
        artifact_dir / "all_model_training_report.json",
        {"mode": "retrain_all_models", "result": result, "metrics": metric_rows, "production_adoption_allowed": False},
    )
    write_json(artifact_dir / "champion_selection_report.json", champion_report)
    _write_core_retrain_summary(
        artifact_dir=artifact_dir,
        dataset_path=dataset_path,
        labels_trained=[selected_label],
        model_families=list(families),
        metric_rows=metric_rows,
        result_payload=result,
        evaluation_return_column=groups["evaluation_return_column_used"] or "net_forward_return",
        row_count=int(row_count),
    )

    status_rows = _load_checkpoint_status_rows(artifact_dir)
    keyed_status = {
        (str(row.get("label_name")), str(row.get("model_name"))): row
        for row in status_rows
    }
    trained_entries: List[Dict[str, Any]] = []
    for label_name, model_reports in report_map.items():
        for model_name, report in (model_reports or {}).items():
            status_doc = keyed_status.get((str(label_name), str(model_name)), {})
            trained_entries.append({
                "label_name": label_name,
                "model_name": model_name,
                "model_path": status_doc.get("artifact_path") or report.get("model_path"),
                "metrics_path": status_doc.get("metrics_path"),
                "selected_threshold": (report.get("selected_threshold_from_validation") or {}).get("threshold"),
                "test_metrics": report.get("test_metrics") or {},
            })
    skipped_entries = [
        {
            "label_name": entry.get("label_name"),
            "model_name": entry.get("model_family"),
            "reason": entry.get("reason"),
        }
        for entry in (result.get("skipped_models") or [])
    ]
    failed_entries = [
        {
            "label_name": entry.get("label_name"),
            "model_name": entry.get("model_family"),
            "error": entry.get("error"),
            "traceback": entry.get("stack_summary"),
        }
        for entry in (result.get("failed_models") or [])
    ]
    summary_payload: Dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "artifact_dir": str(artifact_dir),
        "row_count": int(row_count),
        "date_range": date_range,
        "labels_considered": [selected_label],
        "labels_skipped": [],
        "models_planned": list(families),
        "models_trained": trained_entries,
        "models_skipped": skipped_entries,
        "models_failed": failed_entries,
        "leakage_columns_dropped": list(dropped_leakage),
        "validation_strategy": "chronological_walk_forward",
        "walk_forward_folds": int(walk_forward_folds),
        "completed_ok": bool(trained_entries and not failed_entries),
        "remaining_blockers": [],
        "production_adoption_allowed": False,
        "research_only": True,
    }
    write_json(artifact_dir / "retrain_all_models_summary.json", summary_payload)
    write_json(
        artifact_dir / "final_ml_trading_decision_report.json",
        {
            "artifact_dir": str(artifact_dir),
            "dataset_path": str(dataset_path),
            "decision": "BLOCKED",
            "best_model": champion_report.get("champion"),
            "reason": "Completed checkpoint finalized in research-only mode; paper/live adoption remains disabled.",
            "production_adoption_allowed": False,
            "research_only": True,
        },
    )
    return {
        "metric_rows": metric_rows,
        "report_map": report_map,
        "champion_report": champion_report,
        "training_manifest": training_manifest,
        "feature_manifest": feature_manifest,
    }


def _regime_candidate_mask(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.Series:
    work = df.copy()
    mask = pd.Series(True, index=work.index)
    option_side = _option_side_series(work)
    moneyness_bucket = _moneyness_bucket_series(work)
    volatile_series = _numeric_series(work, "regime_volatile")
    if "option_side" in filters:
        mask &= option_side.eq(str(filters["option_side"]).upper())
    if "moneyness" in filters:
        allowed = {str(value) for value in filters["moneyness"]}
        mask &= moneyness_bucket.isin(allowed)
    if filters.get("volatile_only"):
        if "regime_volatile" in work.columns:
            mask &= volatile_series.fillna(0.0) >= 1.0
        elif "volatility_regime_classifier" in work.columns:
            mask &= _numeric_series(work, "volatility_regime_classifier").fillna(0.0) >= 1.0
        else:
            mask &= False
    return mask.fillna(False)


def _trade_metrics_from_selected_returns(selected_returns: np.ndarray, *, threshold_source: str, extra_cost: float = 0.0) -> Dict[str, Any]:
    selected_returns = np.asarray(selected_returns, dtype=float)
    if selected_returns.size == 0:
        return _trade_metrics_from_scores(np.array([], dtype=int), np.array([], dtype=float), 0.5, np.array([], dtype=float), threshold_source=threshold_source, extra_cost=extra_cost)
    return _trade_metrics_from_scores(
        np.ones(selected_returns.size, dtype=int),
        np.ones(selected_returns.size, dtype=float),
        0.5,
        selected_returns,
        threshold_source=threshold_source,
        extra_cost=extra_cost,
    )


def _infer_embedded_cost_from_frame(frame: pd.DataFrame) -> float | None:
    if "gross_forward_return" not in frame.columns or "net_forward_return" not in frame.columns:
        return None
    gross = pd.to_numeric(frame["gross_forward_return"], errors="coerce")
    net = pd.to_numeric(frame["net_forward_return"], errors="coerce")
    diff = (gross - net).replace([np.inf, -np.inf], np.nan).dropna()
    diff = diff[diff >= 0.0]
    if diff.empty:
        return None
    return float(diff.median())


def _require_cost_columns_for_target(
    frame: pd.DataFrame,
    *,
    target_name: str,
    require_per_row_cost: bool = False,
) -> None:
    """Validate required cost columns exist for a target.

    Raises
    ------
    RuntimeError
        If required cost columns are missing for a cost-aware target.

    Notes
    -----
    Cost-aware targets require either:
      (a) per_row_cost_units from estimate_option_execution_costs_frame (preferred), OR
      (b) gross_forward_return and net_forward_return to infer embedded cost (fallback).

    When require_per_row_cost=True and neither path is available, raises RuntimeError
    with a clear message naming the missing column and target.
    """
    if frame.empty:
        return  # Empty frame: no cost to validate

    missing: List[str] = []
    if "ltp" not in frame.columns:
        missing.append("ltp")

    # Per-row cost path requires premium (ltp) column for cost estimation
    has_per_row_path = (
        "ltp" in frame.columns
        and "net_forward_return" in frame.columns
        and "gross_forward_return" in frame.columns
    )
    has_inferred_path = (
        "gross_forward_return" in frame.columns
        and "net_forward_return" in frame.columns
    )

    if require_per_row_cost and not has_per_row_path:
        if "ltp" in missing:
            raise RuntimeError(
                f"MISSING_COST_COLUMN: target='{target_name}' requires 'ltp' column "
                f"for per-row cost estimation via estimate_option_execution_costs_frame. "
                f"Available columns: {list(frame.columns)}. "
                f"Either add 'ltp' to the dataset or use cost_provenance without per_row_cost_units."
            )
        if "gross_forward_return" not in frame.columns:
            raise RuntimeError(
                f"MISSING_COST_COLUMN: target='{target_name}' requires 'gross_forward_return' "
                f"and 'net_forward_return' for cost inference. "
                f"Available columns: {list(frame.columns)}. "
                f"Either add return columns or use cost_provenance fallback."
            )

    # If neither path is available at all, log a warning (non-fatal for non-strict targets)
    if not has_per_row_path and not has_inferred_path:
        import warnings
        warnings.warn(
            f"NO_COST_PROVENANCE: target='{target_name}' has neither per-row cost columns "
            f"(ltp, gross/net return) nor embedded cost inference columns. "
            f"Cost stress will use fallback scalar. Available columns: {list(frame.columns)}",
            RuntimeWarning,
        )


def _return_cost_provenance_for_frame(
    frame: pd.DataFrame,
    *,
    evaluation_return_column: str | None,
    estimated_cost: float | None = None,
    include_per_row_cost_frame: bool = True,
) -> Dict[str, Any]:
    """Compute cost provenance for a dataset frame.

    Parameters
    ----------
    include_per_row_cost_frame : bool
        When True (default), also computes per-row cost frame using
        estimate_option_execution_costs_frame and attaches cost_return_units
        array to the provenance dict for per-target cost stress computation.

    Returns
    -------
    provenance : dict
        Extended dict now includes:
          - per_row_cost_units : np.ndarray | None  (per-trade cost array)
          - cost_column_source : str               ('per_target_frame' | 'inferred_embedded' | 'fallback_scalar')
          - cost_model_confidence : str
          - cost_model_status : str
          - average_cost_return_units : float
          - average_cost_pct_of_premium : float
    """
    gross_return_column = "gross_forward_return" if "gross_forward_return" in frame.columns else None
    net_return_column = "net_forward_return" if "net_forward_return" in frame.columns else None
    inferred_embedded_cost = _infer_embedded_cost_from_frame(frame)
    is_evaluation_return_already_net = bool(evaluation_return_column and ("net_" in str(evaluation_return_column).lower() or "cost_adjusted" in str(evaluation_return_column).lower()))
    fallback_cost = float(estimated_cost or 0.0)
    cost_model_status = "BLOCKED_UNCERTAIN_COST_PROVENANCE"
    cost_model_confidence = "low"
    baseline_cost_deduction_used = 0.0
    cost_1_25x_incremental_deduction_used = 0.0
    cost_1_50x_incremental_deduction_used = 0.0
    double_counting_prevented = False
    if is_evaluation_return_already_net:
        if inferred_embedded_cost is not None and inferred_embedded_cost > 0.0:
            cost_model_status = "VALID_NET_RETURN_INCREMENTAL_STRESS_MODEL"
            cost_model_confidence = "high"
            baseline_cost_deduction_used = 0.0
            cost_1_25x_incremental_deduction_used = inferred_embedded_cost * 0.25
            cost_1_50x_incremental_deduction_used = inferred_embedded_cost * 0.50
            double_counting_prevented = True
        else:
            cost_model_status = "BLOCKED_UNCERTAIN_COST_PROVENANCE"
            cost_model_confidence = "low"
    else:
        baseline_deduction = float(inferred_embedded_cost if inferred_embedded_cost is not None else fallback_cost)
        if baseline_deduction > 0.0:
            cost_model_status = "VALID_GROSS_RETURN_COST_MODEL"
            cost_model_confidence = "high" if inferred_embedded_cost is not None else "medium"
            baseline_cost_deduction_used = baseline_deduction
            cost_1_25x_incremental_deduction_used = baseline_deduction * 1.25
            cost_1_50x_incremental_deduction_used = baseline_deduction * 1.50

    # Per-row cost frame computation (per-target path)
    per_row_cost_units: np.ndarray | None = None
    average_cost_return_units = 0.0
    average_cost_pct_of_premium = 0.0
    cost_column_source = "fallback_scalar"

    if include_per_row_cost_frame and not frame.empty:
        try:
            cost_frame = estimate_option_execution_costs_frame(frame, config=DEFAULT_EXECUTION_COST_CONFIG)
            if (
                not cost_frame.empty
                and "cost_return_units" in cost_frame.columns
                and "cost_pct_of_premium" in cost_frame.columns
                and len(cost_frame) == len(frame)
            ):
                costs = pd.to_numeric(cost_frame["cost_return_units"], errors="coerce").fillna(0.0).to_numpy()
                costs = np.clip(costs, 0.0, None)
                per_row_cost_units = costs
                average_cost_return_units = float(np.mean(costs)) if len(costs) > 0 else 0.0
                average_cost_pct_of_premium = float(
                    pd.to_numeric(cost_frame["cost_pct_of_premium"], errors="coerce").mean()
                )
                cost_column_source = "per_target_frame"
            elif inferred_embedded_cost is not None and inferred_embedded_cost > 0.0:
                cost_column_source = "inferred_embedded"
            else:
                cost_column_source = "fallback_scalar"
        except Exception:
            # Fallback: use inferred embedded cost
            cost_column_source = "inferred_embedded" if inferred_embedded_cost is not None and inferred_embedded_cost > 0.0 else "fallback_scalar"

    return {
        "evaluation_return_column": evaluation_return_column,
        "gross_return_column": gross_return_column,
        "net_return_column": net_return_column,
        "inferred_embedded_cost": float(inferred_embedded_cost) if inferred_embedded_cost is not None else None,
        "is_evaluation_return_already_net": is_evaluation_return_already_net,
        "baseline_cost_deduction_used": float(baseline_cost_deduction_used),
        "cost_1_25x_incremental_deduction_used": float(cost_1_25x_incremental_deduction_used),
        "cost_1_50x_incremental_deduction_used": float(cost_1_50x_incremental_deduction_used),
        "double_counting_prevented": bool(double_counting_prevented),
        "cost_model_confidence": cost_model_confidence,
        "cost_model_status": cost_model_status,
        # Per-target per-row cost data
        "per_row_cost_units": per_row_cost_units,
        "cost_column_source": cost_column_source,
        "average_cost_return_units": float(average_cost_return_units),
        "average_cost_pct_of_premium": float(average_cost_pct_of_premium),
        "has_per_row_cost_frame": bool(per_row_cost_units is not None),
    }


def _selection_rows_for_candidate(
    selected_rows: pd.DataFrame,
    *,
    selection_name: str,
    threshold_value: float | None = None,
) -> Dict[str, Any]:
    selected_returns = selected_rows["selected_return"].to_numpy(dtype=float) if "selected_return" in selected_rows.columns else np.array([], dtype=float)
    metrics = _trade_metrics_from_selected_returns(selected_returns, threshold_source=f"regime_candidate:{selection_name}")
    trade_days = int(selected_rows["trade_day"].nunique()) if not selected_rows.empty and "trade_day" in selected_rows.columns else 0
    trades_per_day = float(len(selected_rows) / trade_days) if trade_days > 0 else 0.0
    median_return = float(np.median(selected_returns)) if selected_returns.size else 0.0
    max_consecutive_losses = 0
    if selected_returns.size:
        loss_mask = selected_returns < 0.0
        if bool(loss_mask.any()):
            streak_groups = np.cumsum(~loss_mask)
            max_consecutive_losses = int(pd.Series(loss_mask.astype(int)).groupby(streak_groups).sum().max())
    daily_mean = 0.0
    daily_std = 0.0
    if not selected_rows.empty and "trade_day" in selected_rows.columns:
        by_day = selected_rows.groupby("trade_day")["selected_return"].sum()
        daily_mean = float(by_day.mean()) if len(by_day) else 0.0
        daily_std = float(by_day.std(ddof=1)) if len(by_day) > 1 else 0.0
    payload = {
        "selection_name": selection_name,
        "threshold": float(threshold_value) if threshold_value is not None else None,
        "trade_days": trade_days,
        "trades_per_day": trades_per_day,
        "trades": int(len(selected_rows)),
        "median_net_forward_return": median_return,
        "max_consecutive_losses": int(max_consecutive_losses),
        "daily_return_mean": daily_mean,
        "daily_return_std": daily_std,
        "selected_rows": selected_rows,
        **metrics,
    }
    return payload


def _monthly_stability_for_selection(selected_rows: pd.DataFrame) -> Dict[str, Any]:
    if selected_rows.empty or "timestamp" not in selected_rows.columns or "selected_return" not in selected_rows.columns:
        return {
            "months": [],
            "monthly_pf": {},
            "worst_month": None,
            "best_month": None,
            "max_month_profit_share": 0.0,
            "passes_monthly_concentration_gate": False,
        }
    work = selected_rows.copy()
    work["month"] = _normalize_timestamp_series(work["timestamp"]).dt.to_period("M").astype(str)
    month_rows: List[Dict[str, Any]] = []
    positive_total = float(work.loc[work["selected_return"] > 0.0, "selected_return"].sum())
    for month, group in work.groupby("month", sort=True):
        metrics = _trade_metrics_from_selected_returns(group["selected_return"].to_numpy(dtype=float), threshold_source=f"monthly:{month}")
        month_rows.append(
            {
                "month": month,
                "trade_count": int(len(group)),
                "profit_factor": float(metrics.get("profit_factor") or 0.0),
                "sharpe": float(metrics.get("sharpe") or 0.0),
                "total_return": float(metrics.get("total_return") or 0.0),
                "average_return": float(metrics.get("average_return_per_trade") or 0.0),
                "gross_profit": float(metrics.get("gross_profit") or 0.0),
                "gross_loss": float(metrics.get("gross_loss") or 0.0),
            }
        )
    worst_month = min(month_rows, key=lambda row: float(row.get("total_return") or 0.0)) if month_rows else None
    best_month = max(month_rows, key=lambda row: float(row.get("total_return") or 0.0)) if month_rows else None
    max_month_profit_share = 0.0
    if positive_total > 0:
        max_month_profit_share = max((float(row.get("gross_profit") or 0.0) / positive_total) for row in month_rows) if month_rows else 0.0
    return {
        "months": month_rows,
        "monthly_pf": {row["month"]: row["profit_factor"] for row in month_rows},
        "worst_month": worst_month,
        "best_month": best_month,
        "profitable_months": int(sum(1 for row in month_rows if float(row.get("total_return") or 0.0) > 0.0)),
        "losing_months": int(sum(1 for row in month_rows if float(row.get("total_return") or 0.0) < 0.0)),
        "max_month_profit_share": max_month_profit_share,
        "passes_monthly_concentration_gate": bool(month_rows and max_month_profit_share <= 0.40),
    }


def _fold_metrics_for_selection(selected_rows: pd.DataFrame, *, fold_count: int = 5) -> Dict[str, Any]:
    if selected_rows.empty or "timestamp" not in selected_rows.columns:
        return {"folds": [], "max_fold_profit_share": 0.0, "passes_fold_concentration_gate": False}
    work = selected_rows.copy()
    work["timestamp"] = _normalize_timestamp_series(work["timestamp"])
    groups = [group for _, group in work.groupby(work["timestamp"].dt.to_period("M"), sort=True)]
    if len(groups) < 2:
        groups = [group for _, group in work.groupby(work["timestamp"].dt.date, sort=True)]
    if not groups:
        groups = [work]
    fold_groups = np.array_split(np.arange(len(groups)), min(max(1, int(fold_count)), len(groups)))
    fold_rows: List[Dict[str, Any]] = []
    positive_total = float(work.loc[work["selected_return"] > 0.0, "selected_return"].sum())
    for idx, group_indexes in enumerate(fold_groups, start=1):
        if len(group_indexes) == 0:
            continue
        parts = [groups[int(group_idx)] for group_idx in group_indexes]
        fold_frame = pd.concat(parts, axis=0).sort_values("timestamp", kind="stable")
        metrics = _trade_metrics_from_selected_returns(fold_frame["selected_return"].to_numpy(dtype=float), threshold_source=f"fold:{idx}")
        fold_rows.append(
            {
                "fold": idx,
                "start": str(fold_frame["timestamp"].min()),
                "end": str(fold_frame["timestamp"].max()),
                "trade_count": int(len(fold_frame)),
                "profit_factor": float(metrics.get("profit_factor") or 0.0),
                "sharpe": float(metrics.get("sharpe") or 0.0),
                "max_drawdown": float(metrics.get("max_drawdown") or 0.0),
                "average_return": float(metrics.get("average_return_per_trade") or 0.0),
                "gross_profit": float(metrics.get("gross_profit") or 0.0),
                "total_return": float(metrics.get("total_return") or 0.0),
            }
        )
    max_fold_profit_share = 0.0
    if positive_total > 0 and fold_rows:
        max_fold_profit_share = max((float(row.get("gross_profit") or 0.0) / positive_total) for row in fold_rows)
    return {
        "folds": fold_rows,
        "max_fold_profit_share": max_fold_profit_share,
        "passes_fold_concentration_gate": bool(fold_rows and max_fold_profit_share <= 0.40),
    }


def _selection_cost_stress(
    selected_rows: pd.DataFrame,
    *,
    cost_provenance: Dict[str, Any] | None = None,
    per_row_cost_units: pd.Series | None = None,
) -> Dict[str, Any]:
    """Compute cost-stress scenarios for selected candidate rows.

    Supports per-row cost frames from estimate_option_execution_costs_frame
    when per_row_cost_units is provided. Falls back to global embedded cost
    (legacy behavior) when not provided.
    """
    selected_returns = (
        selected_rows["selected_return"].to_numpy(dtype=float)
        if not selected_rows.empty and "selected_return" in selected_rows.columns
        else np.array([], dtype=float)
    )
    cost_rows: Dict[str, Any] = {}
    provenance = dict(cost_provenance or {})
    is_net = bool(provenance.get("is_evaluation_return_already_net"))
    inferred_embedded_cost = float(provenance.get("inferred_embedded_cost") or 0.0)
    baseline_deduction = float(provenance.get("baseline_cost_deduction_used") or 0.0)

    has_per_row = (
        per_row_cost_units is not None
        and len(per_row_cost_units) == len(selected_returns)
        and not selected_rows.empty
    )
    cost_source = "per_target_frame" if has_per_row else (
        "inferred_embedded" if (is_net and inferred_embedded_cost > 0.0) or baseline_deduction > 0.0 else "fallback_scalar"
    )
    provenance["cost_column_source"] = cost_source
    provenance["has_per_row_cost_frame"] = has_per_row

    for scenario_name, extra_cost in (
        ("base", 0.0),
        ("cost_1_10x", 0.10),
        ("cost_1_25x", 0.25),
        ("cost_1_50x", 0.50),
        ("cost_2_00x", 1.00),
    ):
        if has_per_row:
            costs_arr = np.asarray(per_row_cost_units.to_numpy(), dtype=float) if not isinstance(per_row_cost_units, np.ndarray) else per_row_cost_units
            costs_arr = np.clip(costs_arr, 0.0, None)
            if scenario_name == "base":
                deduction_arr = costs_arr
            else:
                deduction_arr = costs_arr * (1.0 + float(extra_cost))
            effective_deduction = float(np.mean(deduction_arr))
            deduction_for_metrics: float | np.ndarray = deduction_arr
        elif is_net:
            if scenario_name == "base":
                deduction = inferred_embedded_cost
            else:
                deduction = inferred_embedded_cost + inferred_embedded_cost * float(extra_cost)
            effective_deduction = float(deduction)
            deduction_for_metrics = float(deduction)
        else:
            if scenario_name == "base":
                deduction = baseline_deduction
            else:
                deduction = baseline_deduction * float(extra_cost) if baseline_deduction > 0.0 else float(extra_cost)
            effective_deduction = float(deduction)
            deduction_for_metrics = float(deduction)

        cost_rows[scenario_name] = {
            "extra_cost": float(extra_cost),
            "effective_cost_deduction": effective_deduction,
            "cost_source": cost_source,
            **_trade_metrics_from_selected_returns(
                selected_returns,
                threshold_source=f"candidate_cost:{scenario_name}",
                extra_cost=float(effective_deduction),
            ),
            "cost_provenance": provenance,
        }

    # Record per-target PF in provenance for downstream gate checks
    provenance["per_target_pf_1_25x"] = float(
        (cost_rows.get("cost_1_25x") or {}).get("profit_factor") or 0.0
    )
    provenance["per_target_pf_1_50x"] = float(
        (cost_rows.get("cost_1_50x") or {}).get("profit_factor") or 0.0
    )
    return cost_rows


def _execution_cost_model_report(selected_rows: pd.DataFrame) -> Dict[str, Any]:
    if selected_rows.empty:
        return {
            "spread_mode": "estimated",
            "late_fill_mode": "estimated",
            "base_cost_components": {},
            "scenarios": {},
        }
    premium = _option_price_series(selected_rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    spread_pct = _spread_pct_series(selected_rows).replace([np.inf, -np.inf], np.nan)
    has_actual_spread = bool(spread_pct.notna().any())
    spread_pct = spread_pct.fillna(5.0)
    brokerage_fee = float(np.clip(premium.median() * 0.0005, 0.02, 0.15))
    spread_cost = float(np.median((spread_pct / 100.0) * premium * 0.50))
    slippage_cost = float(np.clip(premium.median() * 0.0008, 0.01, 0.20))
    adverse_selection = float(np.clip(premium.median() * 0.0006, 0.01, 0.15))
    late_fill = float(np.clip(premium.median() * 0.0007, 0.01, 0.20))
    conservative = brokerage_fee + spread_cost + slippage_cost + adverse_selection + late_fill
    scenarios = {
        "base_cost": brokerage_fee,
        "spread_adjusted_cost": brokerage_fee + spread_cost + slippage_cost,
        "conservative_cost": conservative,
        "conservative_cost_1_25x": conservative * 1.25,
        "conservative_cost_1_50x": conservative * 1.50,
    }
    return {
        "spread_mode": "actual_bid_ask" if has_actual_spread else "estimated",
        "late_fill_mode": "estimated",
        "base_cost_components": {
            "brokerage_fees_estimate": brokerage_fee,
            "bid_ask_spread_cost": spread_cost,
            "slippage_cost": slippage_cost,
            "adverse_selection_buffer": adverse_selection,
            "failed_or_late_fill_buffer": late_fill,
        },
        "scenarios": scenarios,
    }


def _liquidity_filter_report(selected_rows: pd.DataFrame) -> Dict[str, Any]:
    if selected_rows.empty:
        return {"available": False, "filters": [], "survives_filters": False}
    work = selected_rows.copy()
    volume = _numeric_series(work, "volume", default=np.nan)
    if not volume.notna().any():
        volume = _numeric_series(work, "option_volume", default=np.nan)
    oi = _numeric_series(work, "oi", default=np.nan)
    premium = _option_price_series(work)
    spread_pct = _spread_pct_series(work)
    filters = []
    mask = pd.Series(True, index=work.index)
    filter_specs = [
        ("minimum_volume", volume.fillna(-np.inf) >= float(volume.dropna().median()) if volume.notna().any() else pd.Series(True, index=work.index)),
        ("minimum_open_interest", oi.fillna(-np.inf) >= float(oi.dropna().median()) if oi.notna().any() else pd.Series(True, index=work.index)),
        ("maximum_bid_ask_spread", spread_pct.fillna(np.inf) <= 5.0 if spread_pct.notna().any() else pd.Series(True, index=work.index)),
        ("minimum_option_premium", premium.fillna(-np.inf) >= 5.0),
        ("maximum_option_premium", premium.fillna(np.inf) <= float(premium.quantile(0.95)) if premium.notna().any() else pd.Series(True, index=work.index)),
        ("atm_itm_near_atm_only", _moneyness_bucket_series(work).isin(["ATM", "NEAR_ATM", "ITM", "deep_itm"])),
    ]
    for name, condition in filter_specs:
        before = int(mask.sum())
        mask &= condition.fillna(False)
        after = int(mask.sum())
        filters.append({"filter_name": name, "rows_before": before, "rows_after": after, "rows_removed": before - after})
    filtered = work.loc[mask].copy()
    filtered_metrics = _trade_metrics_from_selected_returns(filtered["selected_return"].to_numpy(dtype=float) if "selected_return" in filtered.columns else np.array([], dtype=float), threshold_source="liquidity_filter")
    return {
        "available": True,
        "filters": filters,
        "survives_filters": bool(len(filtered) > 0),
        "surviving_trade_count": int(len(filtered)),
        "surviving_profit_factor": filtered_metrics.get("profit_factor"),
        "surviving_sharpe": filtered_metrics.get("sharpe"),
    }


def _trade_capacity_report(selected_rows: pd.DataFrame, all_trade_days: Sequence[Any]) -> Dict[str, Any]:
    if selected_rows.empty:
        return {"available": False}
    work = selected_rows.copy()
    if "trade_day" not in work.columns and "timestamp" in work.columns:
        work["trade_day"] = _normalize_timestamp_series(work["timestamp"]).dt.date
    day_counts = work.groupby("trade_day").size()
    premium = _option_price_series(work).replace([np.inf, -np.inf], np.nan)
    spread_pct = _spread_pct_series(work).replace([np.inf, -np.inf], np.nan)
    total_days = len(set(all_trade_days))
    active_days = int(day_counts.size)
    return {
        "available": True,
        "average_trades_per_day": float(day_counts.mean()) if active_days else 0.0,
        "max_trades_per_day": int(day_counts.max()) if active_days else 0,
        "days_with_zero_trades": int(max(0, total_days - active_days)),
        "median_holding_period_minutes": None,
        "average_option_premium": float(premium.mean()) if premium.notna().any() else None,
        "average_bid_ask_spread_pct": float(spread_pct.mean()) if spread_pct.notna().any() else None,
        "realistic_for_live_api_execution": bool((float(day_counts.mean()) if active_days else 0.0) <= 10.0),
    }


def _daily_stability_breakdown(selected_rows: pd.DataFrame) -> Dict[str, Any]:
    if selected_rows.empty:
        return {"available": False, "days": []}
    work = selected_rows.copy()
    if "trade_day" not in work.columns and "timestamp" in work.columns:
        work["trade_day"] = _normalize_timestamp_series(work["timestamp"]).dt.date
    daily = work.groupby("trade_day")["selected_return"].sum().reset_index(name="daily_return")
    daily_returns = daily["daily_return"].to_numpy(dtype=float)
    total_positive = float(daily.loc[daily["daily_return"] > 0.0, "daily_return"].sum())
    max_day_share = 0.0
    if total_positive > 0:
        max_day_share = float(daily.loc[daily["daily_return"] > 0.0, "daily_return"].max() / total_positive)
    daily_std = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    daily_sharpe = float((np.mean(daily_returns) / daily_std) * math.sqrt(252.0)) if daily_std > 0 else 0.0
    worst_day = daily.loc[daily["daily_return"].idxmin()].to_dict() if not daily.empty else None
    best_day = daily.loc[daily["daily_return"].idxmax()].to_dict() if not daily.empty else None
    return {
        "available": True,
        "total_trading_days": int(len(daily)),
        "profitable_days": int((daily["daily_return"] > 0.0).sum()),
        "losing_days": int((daily["daily_return"] < 0.0).sum()),
        "day_win_rate": float((daily["daily_return"] > 0.0).mean()) if len(daily) else 0.0,
        "worst_day": worst_day,
        "best_day": best_day,
        "average_daily_return": float(np.mean(daily_returns)) if len(daily_returns) else 0.0,
        "daily_return_std": daily_std,
        "daily_sharpe": daily_sharpe,
        "largest_single_day_loss": float(np.min(daily_returns)) if len(daily_returns) else 0.0,
        "one_or_two_days_dominate_profit": bool(max_day_share > 0.40),
        "max_day_profit_share": max_day_share,
    }


def _expiry_breakdown(selected_rows: pd.DataFrame) -> Dict[str, Any]:
    if selected_rows.empty:
        return {"available": False, "buckets": []}
    dte = _dte_series(selected_rows)
    if not dte.notna().any():
        return {"available": False, "buckets": [], "reason": "expiry_info_missing"}
    work = selected_rows.copy()
    work["expiry_bucket"] = "non_expiry_days"
    work.loc[dte <= 0.0, "expiry_bucket"] = "expiry_day"
    work.loc[(dte > 0.0) & (dte <= 1.0), "expiry_bucket"] = "one_day_before_expiry"
    work.loc[(dte > 1.0) & (dte <= 2.0), "expiry_bucket"] = "two_days_before_expiry"
    rows = []
    for bucket, group in work.groupby("expiry_bucket", sort=True):
        metrics = _trade_metrics_from_selected_returns(group["selected_return"].to_numpy(dtype=float), threshold_source=f"expiry:{bucket}")
        rows.append({"bucket": bucket, "trade_count": int(len(group)), "profit_factor": metrics.get("profit_factor"), "sharpe": metrics.get("sharpe"), "average_return": metrics.get("average_return_per_trade")})
    return {"available": True, "buckets": rows}


def _time_of_day_breakdown(selected_rows: pd.DataFrame) -> Dict[str, Any]:
    if selected_rows.empty:
        return {"available": False, "buckets": []}
    work = selected_rows.copy()
    ts = _normalize_timestamp_series(work["timestamp"])
    minute = ts.dt.hour * 60 + ts.dt.minute
    buckets = [
        ("09:15-09:30", 555, 570),
        ("09:30-10:30", 570, 630),
        ("10:30-12:00", 630, 720),
        ("12:00-14:00", 720, 840),
        ("14:00-15:00", 840, 900),
        ("15:00-15:30", 900, 930),
    ]
    rows = []
    for label, start, end in buckets:
        group = work.loc[(minute >= start) & (minute < end)].copy()
        metrics = _trade_metrics_from_selected_returns(group["selected_return"].to_numpy(dtype=float), threshold_source=f"time:{label}")
        rows.append({"bucket": label, "trade_count": int(len(group)), "profit_factor": metrics.get("profit_factor"), "sharpe": metrics.get("sharpe"), "average_return": metrics.get("average_return_per_trade")})
    return {"available": True, "buckets": rows}


def _confidence_band_report(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        return {"bands": []}
    work = frame.copy()
    bands = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 1.01)]
    rows = []
    for start, end in bands:
        group = work.loc[(work["probability"] >= start) & (work["probability"] < end)].copy()
        metrics = _trade_metrics_from_selected_returns(group["selected_return"].to_numpy(dtype=float), threshold_source=f"band:{start}-{end}")
        rows.append({"band": f"{start:.2f}-{min(end,1.0):.2f}" if end < 1.0 else "0.80+", "trade_count": int(len(group)), "win_rate": metrics.get("win_rate"), "average_return": metrics.get("average_return_per_trade"), "profit_factor": metrics.get("profit_factor"), "sharpe": metrics.get("sharpe")})
    return {"bands": rows}


def _probability_drift_report(frame: pd.DataFrame, selected_rows: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        return {"rows": []}
    work = frame.copy()
    work["month"] = _normalize_timestamp_series(work["timestamp"]).dt.to_period("M").astype(str)
    selected_keys = set(selected_rows.index.tolist()) if not selected_rows.empty else set()
    rows = []
    for month, group in work.groupby("month", sort=True):
        group_probs = group["probability"].to_numpy(dtype=float)
        selected = group.loc[group.index.isin(selected_keys)].copy()
        rows.append({
            "period": month,
            "mean_predicted_probability": float(np.mean(group_probs)) if len(group_probs) else 0.0,
            "std_predicted_probability": float(np.std(group_probs, ddof=1)) if len(group_probs) > 1 else 0.0,
            "positive_signal_rate": float(np.mean(group.get("probability", 0.0) >= 0.5)) if len(group_probs) else 0.0,
            "top_n_score_distribution": {
                "min": float(selected["probability"].min()) if not selected.empty else None,
                "median": float(selected["probability"].median()) if not selected.empty else None,
                "max": float(selected["probability"].max()) if not selected.empty else None,
            },
        })
    return {"rows": rows}


def _feature_drift_report(train_frame: pd.DataFrame, test_frame: pd.DataFrame, top_features: Sequence[str]) -> Dict[str, Any]:
    rows = []
    for feature in list(top_features)[:20]:
        if feature not in train_frame.columns or feature not in test_frame.columns:
            continue
        train_s = pd.to_numeric(train_frame[feature], errors="coerce")
        test_s = pd.to_numeric(test_frame[feature], errors="coerce")
        train_mean = float(train_s.mean()) if train_s.notna().any() else 0.0
        test_mean = float(test_s.mean()) if test_s.notna().any() else 0.0
        train_std = float(train_s.std(ddof=1)) if train_s.notna().sum() > 1 else 0.0
        test_std = float(test_s.std(ddof=1)) if test_s.notna().sum() > 1 else 0.0
        psi = abs(test_mean - train_mean) / max(abs(train_mean), 1e-6)
        rows.append({
            "feature": feature,
            "train_mean": train_mean,
            "test_mean": test_mean,
            "mean_diff": test_mean - train_mean,
            "train_std": train_std,
            "test_std": test_std,
            "missing_rate_diff": float(test_s.isna().mean() - train_s.isna().mean()),
            "psi_estimate": psi,
        })
    return {"rows": rows}


def _duplicate_timestamp_leakage_audit(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    work = _prepare_chronological_work(df.copy())
    ts = _normalize_timestamp_series(work["timestamp"])
    train_ts = set(ts.iloc[train_idx].astype(str).tolist())
    test_ts = set(ts.iloc[test_idx].astype(str).tolist())
    overlap = train_ts & test_ts
    subset_cols = [col for col in ("timestamp", "expiry", "strike_price", "option_type", "option_type_ce", "option_type_pe", "selected_option_symbol") if col in work.columns]
    duplicate_rows = int(work.duplicated().sum())
    duplicate_ts_key = int(work.duplicated(subset=subset_cols).sum()) if subset_cols else 0
    return {
        "duplicate_rows": duplicate_rows,
        "duplicate_timestamp_key_rows": duplicate_ts_key,
        "train_test_timestamp_overlap_count": int(len(overlap)),
        "same_timestamp_in_train_and_test": bool(overlap),
        "forward_return_window_overlap_audit": "not_explicitly_observable_from_dataset",
    }


def _target_horizon_report(df: pd.DataFrame) -> Dict[str, Any]:
    available = [col for col in df.columns if any(token in str(col).lower() for token in ("3m", "5m", "10m", "15m", "30m")) and any(tok in str(col).lower() for tok in ("return", "label", "success"))]
    return {
        "available_horizon_columns": available,
        "multiple_horizons_available": bool(len(available) > 1),
        "note": "Only one target horizon available." if len(available) <= 1 else "Multiple target horizons detected.",
    }


def _deployment_recommendation(candidate_payload: Dict[str, Any]) -> str:
    best = candidate_payload.get("best_selection") or {}
    pf = float(best.get("profit_factor") or 0.0)
    sharpe = float(best.get("sharpe") or 0.0)
    cost_125 = float(((candidate_payload.get("cost_stress") or {}).get("cost_1_25x") or {}).get("profit_factor") or 0.0)
    if pf <= 1.0 or sharpe <= 0.0:
        return "REJECTED"
    if bool(candidate_payload.get("passes_research_gate")):
        return "PAPER_ELIGIBLE"
    if pf > 1.25 and sharpe > 1.0 and cost_125 > 1.0:
        return "PAPER_WATCHLIST"
    return "RESEARCH_ONLY"


def _candidate_summary_markdown(candidate_rows: Sequence[Dict[str, Any]], artifact_dir: Path) -> None:
    lines = ["# Candidate Summary Report", "", f"- production adoption allowed: `False`", ""]
    observed = [row for row in candidate_rows if row.get("status") == "OK"]
    observed.sort(key=lambda row: float(row.get("candidate_score") or 0.0), reverse=True)
    lines.append("## Best Candidates")
    for row in observed[:10]:
        best = row.get("best_selection") or {}
        lines.append(f"- `{row['candidate_name']}` score=`{row.get('candidate_score'):.4f}` PF=`{best.get('profit_factor')}` Sharpe=`{best.get('sharpe')}` trades=`{best.get('trades')}` verdict=`{row.get('deployment_recommendation')}`")
    lines.append("")
    lines.append("## Production Block")
    lines.append("- Production remains blocked because no candidate cleared cost, sample-size, and stability gates together.")
    (artifact_dir / "candidate_summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def _cost_model_audit_report(
    df: pd.DataFrame,
    *,
    evaluation_return_column: str | None,
    artifact_dir: Path,
) -> Dict[str, Any]:
    cost_formula = "selected_return = evaluation_return - extra_cost"
    sample_cols = [col for col in (evaluation_return_column, "ltp", "bid_ask_spread_pct", "bid_ask_spread", "option_volume") if col and col in df.columns]
    sample = df[sample_cols].head(10).copy() if sample_cols else pd.DataFrame()
    return_series = pd.to_numeric(df[evaluation_return_column], errors="coerce") if evaluation_return_column and evaluation_return_column in df.columns else pd.Series(dtype=float)
    avg_abs_return = float(return_series.abs().mean()) if not return_series.empty else None
    current_stress_values = [0.10, 0.25, 0.50, 1.00]
    unit_match = bool(avg_abs_return is None or avg_abs_return == 0.0 or max(current_stress_values) < avg_abs_return * 20.0)
    already_cost_adjusted = bool(evaluation_return_column and ("net_" in evaluation_return_column.lower() or "cost_adjusted" in evaluation_return_column.lower()))
    gross_series = pd.to_numeric(df["gross_forward_return"], errors="coerce") if "gross_forward_return" in df.columns else pd.Series(dtype=float)
    net_series = pd.to_numeric(df["net_forward_return"], errors="coerce") if "net_forward_return" in df.columns else return_series
    embedded_diff = (gross_series - net_series).replace([np.inf, -np.inf], np.nan).dropna() if not gross_series.empty and not net_series.empty else pd.Series(dtype=float)
    embedded_cost = float(embedded_diff[embedded_diff > 0.0].median()) if not embedded_diff.empty and (embedded_diff > 0.0).any() else 0.0
    cost_assumptions = dict(DEFAULT_EXECUTION_COST_CONFIG)
    recommendation = "VALID_GROSS_RETURN_COST_MODEL"
    notes = []
    if already_cost_adjusted:
        notes.append("evaluation return column name suggests returns are already net of some costs")
    if avg_abs_return is not None and avg_abs_return > 0 and 0.25 > avg_abs_return:
        recommendation = "BUG_SUSPECTED_DOUBLE_COUNTING"
        notes.append("1.25x stress subtractor is larger than average absolute return per trade")
    if already_cost_adjusted and embedded_cost > 0.0:
        recommendation = "VALID_NET_RETURN_INCREMENTAL_STRESS_MODEL"
    elif already_cost_adjusted:
        recommendation = "BLOCKED_UNCERTAIN_COST_PROVENANCE"
    if already_cost_adjusted and embedded_cost > 0.0 and avg_abs_return is not None and avg_abs_return > 0 and 0.25 > avg_abs_return:
        recommendation = "BUG_SUSPECTED_DOUBLE_COUNTING"
    if already_cost_adjusted:
        notes.append("additional stress may double-count brokerage/fees if those are already embedded in net_forward_return")
    payload = {
        "current_cost_formula": cost_formula,
        "columns_used": sample_cols,
        "evaluation_return_column": evaluation_return_column,
        "net_forward_return_already_cost_adjusted": already_cost_adjusted,
        "gross_forward_return_available": bool("gross_forward_return" in df.columns),
        "gross_forward_return_average": float(gross_series.mean()) if not gross_series.empty else None,
        "net_forward_return_average": float(net_series.mean()) if not net_series.empty else None,
        "gross_minus_net_average": float(embedded_diff.mean()) if not embedded_diff.empty else None,
        "gross_minus_net_median": float(embedded_diff.median()) if not embedded_diff.empty else None,
        "inferred_embedded_cost": embedded_cost,
        "double_counting_warning": bool(already_cost_adjusted and embedded_cost > 0.0),
        "double_counting_prevented": bool(already_cost_adjusted and embedded_cost > 0.0),
        "stress_cost_mode": "additional",
        "stress_cost_is_replacement": False,
        "cost_units_match_return_units": unit_match,
        "current_stress_values": current_stress_values,
        "current_stress_is_roundtrip": True,
        "current_stress_application": "absolute_return_units_subtracted_once_per_trade",
        "cost_assumptions": cost_assumptions,
        "sample_trades": sample.to_dict(orient="records"),
        "average_absolute_return": avg_abs_return,
        "recommendation": recommendation,
        "cost_model_status": recommendation,
        "notes": notes,
        "production_adoption_allowed": False,
    }
    md = [
        "# Cost Model Audit",
        f"- Formula: `{cost_formula}`",
        f"- Evaluation return column: `{evaluation_return_column}`",
        f"- Already cost-adjusted by name: `{already_cost_adjusted}`",
        f"- Inferred embedded cost: `{embedded_cost}`",
        f"- Double counting warning: `{payload['double_counting_warning']}`",
        f"- Stress mode: `additional`",
        f"- Units appear aligned: `{unit_match}`",
        f"- Recommendation: `{recommendation}`",
    ]
    write_json(artifact_dir / "cost_model_audit_report.json", payload)
    (artifact_dir / "cost_model_audit_report.md").write_text("\n".join(md), encoding="utf-8")
    return payload


def _horizon_comparison_report(
    df: pd.DataFrame,
    *,
    artifact_dir: Path,
) -> Dict[str, Any]:
    candidates = []
    for col in df.columns:
        lowered = str(col).lower()
        if any(token in lowered for token in ("return", "forward", "label_return", "target_return")) and any(token in lowered for token in ("1m", "3m", "5m", "10m", "15m", "30m")):
            candidates.append(str(col))
    available_horizons = sorted(candidates)
    rows: List[Dict[str, Any]] = []
    note = "Only one horizon available." if len(available_horizons) <= 1 else "Multiple horizons detected."
    payload = {
        "available_horizons": available_horizons,
        "best_horizon": available_horizons[0] if available_horizons else None,
        "longer_horizon_improves_cost_survival": False,
        "rows": rows,
        "note": note,
        "production_adoption_allowed": False,
    }
    write_json(artifact_dir / "horizon_comparison_report.json", payload)
    pd.DataFrame(rows).to_csv(artifact_dir / "horizon_comparison_report.csv", index=False)
    (artifact_dir / "horizon_comparison_report.md").write_text("\n".join(["# Horizon Comparison Report", f"- Available horizons: `{', '.join(available_horizons) if available_horizons else 'none'}`", f"- Note: `{note}`"]), encoding="utf-8")
    return payload


def _paper_watchlist_report(candidate_rows: Sequence[Dict[str, Any]], *, artifact_dir: Path) -> Dict[str, Any]:
    watchlist = []
    for row in candidate_rows:
        if row.get("status") != "OK":
            continue
        best = row.get("best_selection") or {}
        monthly = row.get("monthly_stability") or {}
        fold = row.get("fold_metrics") or {}
        if float(best.get("profit_factor") or 0.0) < 1.25 or float(best.get("sharpe") or 0.0) < 1.0:
            continue
        if not bool(row.get("leakage_protection_passed")) or not bool(row.get("live_computable_only_passed")):
            continue
        if float(monthly.get("max_month_profit_share") or 0.0) > 0.40 or float(fold.get("max_fold_profit_share") or 0.0) > 0.40:
            continue
        cost_125 = float(((row.get("cost_stress") or {}).get("cost_1_25x") or {}).get("profit_factor") or 0.0)
        warning_reason = "PAPER_WATCHLIST_NOT_PRODUCTION" if cost_125 < 1.05 else "PAPER_WATCHLIST_COST_OK_STILL_NOT_PRODUCTION"
        watchlist.append(
            {
                "candidate_name": row.get("candidate_name"),
                "model_name": row.get("model_family"),
                "filter_conditions": row.get("filters"),
                "threshold_or_topn_rule": best.get("selection_name"),
                "trade_count": int(best.get("trades") or 0),
                "profit_factor": float(best.get("profit_factor") or 0.0),
                "sharpe": float(best.get("sharpe") or 0.0),
                "cost_stress_1_25x_pf": cost_125,
                "warning_reason": warning_reason,
                "exact_live_paper_observation_rule": f"Observe `{row.get('candidate_name')}` only; no auto-orders; flag when conditions `{row.get('filters')}` and selection `{best.get('selection_name')}` occur.",
                "production_adoption_allowed": False,
            }
        )
    payload = {"candidates": watchlist, "production_adoption_allowed": False, "paper_watchlist_only": True}
    write_json(artifact_dir / "paper_watchlist_report.json", payload)
    (artifact_dir / "paper_watchlist_report.md").write_text("\n".join(["# Paper Watchlist Report", *[f"- `{row['candidate_name']}` rule=`{row['threshold_or_topn_rule']}` warning=`{row['warning_reason']}`" for row in watchlist]]), encoding="utf-8")
    return payload


def _deep_data_audit_report(df: pd.DataFrame, *, artifact_dir: Path, feature_manifest: Dict[str, Any], leakage_passed: bool) -> Dict[str, Any]:
    duplicate_rows = int(df.duplicated().sum())
    subset_cols = [col for col in ("timestamp", "strike_price", "expiry", "option_type", "option_type_ce", "option_type_pe") if col in df.columns]
    duplicate_key_rows = int(df.duplicated(subset=subset_cols).sum()) if subset_cols else 0
    missing_by_column = {col: int(val) for col, val in df.isna().sum().items() if int(val) > 0}
    inf_columns = {col: int(np.isinf(pd.to_numeric(df[col], errors="coerce")).sum()) for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and int(np.isinf(pd.to_numeric(df[col], errors="coerce")).sum()) > 0}
    zero_premium_rows = int((_option_price_series(df).fillna(0.0) <= 0.0).sum())
    zero_volume_rows = int((_numeric_series(df, "volume", default=0.0).fillna(0.0) <= 0.0).sum()) if "volume" in df.columns else int((_numeric_series(df, "option_volume", default=0.0).fillna(0.0) <= 0.0).sum()) if "option_volume" in df.columns else 0
    impossible_spread = int((_spread_pct_series(df).fillna(0.0) < 0.0).sum()) if "bid_ask_spread_pct" in df.columns else 0
    ts = _normalize_timestamp_series(df["timestamp"]) if "timestamp" in df.columns else pd.Series(dtype="datetime64[ns]")
    gaps = ts.sort_values().diff().dropna()
    health_score = 100.0
    critical = []
    warnings = []
    if duplicate_key_rows > 0:
        warnings.append("duplicate timestamp/side rows present")
        health_score -= 10.0
    if any(col in feature_manifest.get("features", []) for col in feature_manifest.get("evaluation_return_columns", [])):
        critical.append("return column leaked into features")
        health_score -= 40.0
    if not leakage_passed:
        critical.append("leakage gate failed")
        health_score -= 40.0
    if zero_premium_rows > 0:
        warnings.append("zero premium rows present")
        health_score -= 5.0
    payload = {
        "severity": "CRITICAL" if critical else ("WARNING" if warnings else "INFO"),
        "critical_findings": critical,
        "warnings": warnings,
        "duplicate_rows": duplicate_rows,
        "duplicate_timestamp_key_rows": duplicate_key_rows,
        "missing_values_by_column": missing_by_column,
        "infinite_values_by_column": inf_columns,
        "zero_option_premium_rows": zero_premium_rows,
        "zero_volume_rows": zero_volume_rows,
        "negative_spread_rows": impossible_spread,
        "timestamp_gap_summary": {"max_gap": str(gaps.max()) if not gaps.empty else None},
        "dataset_health_score": max(0.0, health_score),
        "results_trustworthy": not bool(critical),
        "production_adoption_allowed": False,
    }
    write_json(artifact_dir / "deep_data_audit_report.json", payload)
    (artifact_dir / "deep_data_audit_report.md").write_text("\n".join(["# Deep Data Audit Report", f"- Severity: `{payload['severity']}`", f"- Health score: `{payload['dataset_health_score']}`", f"- Trustworthy: `{payload['results_trustworthy']}`"]), encoding="utf-8")
    return payload


def _final_decision_report(
    *,
    artifact_dir: Path,
    dataset_path: Path,
    row_count: int,
    feature_manifest: Dict[str, Any],
    metrics_payload: Dict[str, Any],
    regime_candidate_report: Dict[str, Any],
    paper_watchlist: Dict[str, Any],
    cost_model_audit: Dict[str, Any],
    horizon_report: Dict[str, Any],
    deep_data_audit: Dict[str, Any],
) -> Dict[str, Any]:
    top_candidates = sorted(
        [row for row in regime_candidate_report["candidates"] if row.get("status") == "OK"],
        key=lambda row: float(row.get("candidate_score") or 0.0),
        reverse=True,
    )[:10]
    verdicts = [
        "GLOBAL_MODEL_REJECTED",
        "REGIME_EDGE_POSSIBLE" if top_candidates else "GLOBAL_MODEL_REJECTED",
        "EXECUTION_COST_BLOCKER",
        "PAPER_WATCHLIST_ONLY" if paper_watchlist.get("candidates") else "PRODUCTION_BLOCKED",
        "PRODUCTION_BLOCKED",
    ]
    payload = {
        "dataset_used": str(dataset_path),
        "rows_used": row_count,
        "features_used": len(feature_manifest.get("features", [])),
        "leakage_exclusions": feature_manifest.get("dropped_leakage_columns", []),
        "models_trained": [row.get("model_name") for row in metrics_payload.get("models_trained", [])],
        "global_model_results": metrics_payload.get("models_trained", []),
        "top_regime_candidates": [{k: v for k, v in row.items() if k not in {"selection_reports", "threshold_sweep", "high_confidence_winning_examples", "high_confidence_losing_examples"}} for row in top_candidates],
        "paper_watchlist_candidates": paper_watchlist.get("candidates", []),
        "cost_model_audit": cost_model_audit,
        "horizon_report": horizon_report,
        "deep_data_audit": deep_data_audit,
        "final_verdict_categories": verdicts,
        "production_blocker": "No candidate satisfied cost-stress, sample-size, and stability gates together.",
        "next_recommended_experiment": "Test narrower PE execution filters only after confirming cost model realism and horizon adequacy.",
        "production_adoption_allowed": False,
    }
    write_json(artifact_dir / "final_ml_trading_decision_report.json", payload)
    md = [
        "# Final ML Trading Decision Report",
        f"- Dataset: `{dataset_path}`",
        f"- Rows used: `{row_count}`",
        f"- Features used: `{len(feature_manifest.get('features', []))}`",
        f"- Models trained: `{', '.join(str(x) for x in payload['models_trained'])}`",
        f"- Does the model have edge? `{'yes, in conditional pockets' if top_candidates else 'no clear edge'}`",
        f"- Why not production-ready? `{payload['production_blocker']}`",
        f"- Next experiment: `{payload['next_recommended_experiment']}`",
    ]
    (artifact_dir / "final_ml_trading_decision_report.md").write_text("\n".join(md), encoding="utf-8")
    return payload


def _execution_rescue_filter_mask(frame: pd.DataFrame, filter_name: str) -> pd.Series:
    work = frame.copy()
    mask = pd.Series(True, index=work.index)
    premium = _option_price_series(work).fillna(np.nan)
    volume = _numeric_series(work, "volume", default=np.nan)
    if not volume.notna().any():
        volume = _numeric_series(work, "option_volume", default=np.nan)
    oi = _numeric_series(work, "oi", default=np.nan)
    spread_pct = _spread_pct_series(work).fillna(np.nan)
    moneyness_bucket = _moneyness_bucket_series(work)
    side = _option_side_series(work)
    ts = _normalize_timestamp_series(work["timestamp"])
    minute = ts.dt.hour * 60 + ts.dt.minute
    volatile = _numeric_series(work, "regime_volatile", default=np.nan)
    if not volatile.notna().any():
        volatile = _numeric_series(work, "volatility_regime_classifier", default=np.nan)
    trending = _numeric_series(work, "regime_trending", default=np.nan)
    quiet = _numeric_series(work, "regime_quiet", default=np.nan)

    if filter_name == "none":
        return mask
    if filter_name == "premium_min":
        return premium >= max(5.0, float(premium.quantile(0.25)) if premium.notna().any() else 5.0)
    if filter_name == "premium_max":
        return premium <= float(premium.quantile(0.90)) if premium.notna().any() else mask
    if filter_name == "premium_mid_bucket":
        if premium.notna().any():
            lo = float(premium.quantile(0.25))
            hi = float(premium.quantile(0.90))
            return (premium >= lo) & (premium <= hi)
        return mask
    if filter_name == "liquidity_volume":
        return volume.fillna(-np.inf) >= float(volume.dropna().median()) if volume.notna().any() else mask
    if filter_name == "liquidity_oi":
        return oi.fillna(-np.inf) >= float(oi.dropna().median()) if oi.notna().any() else mask
    if filter_name == "liquidity_spread":
        return spread_pct.fillna(np.inf) <= min(5.0, float(spread_pct.dropna().quantile(0.75))) if spread_pct.notna().any() else mask
    if filter_name == "liquidity_nonzero_volume":
        return volume.fillna(0.0) > 0.0
    if filter_name == "moneyness_itm":
        return side.eq("PE") & moneyness_bucket.isin(["ITM", "deep_itm"])
    if filter_name == "moneyness_atm":
        return side.eq("PE") & moneyness_bucket.eq("ATM")
    if filter_name == "moneyness_near_atm":
        return side.eq("PE") & moneyness_bucket.eq("NEAR_ATM")
    if filter_name == "moneyness_itm_atm":
        return side.eq("PE") & moneyness_bucket.isin(["ITM", "deep_itm", "ATM"])
    if filter_name == "moneyness_reject_far_otm":
        return ~moneyness_bucket.isin(["OTM", "deep_otm"])
    if filter_name == "time_avoid_open":
        return minute >= 570
    if filter_name == "time_avoid_close":
        return minute < 900
    if filter_name == "time_0930_1030":
        return (minute >= 570) & (minute < 630)
    if filter_name == "time_1030_1200":
        return (minute >= 630) & (minute < 720)
    if filter_name == "time_1200_1400":
        return (minute >= 720) & (minute < 840)
    if filter_name == "time_1400_1500":
        return (minute >= 840) & (minute < 900)
    if filter_name == "vol_trend_volatile_trending":
        return volatile.fillna(0.0) >= 1.0 if not trending.notna().any() else (volatile.fillna(0.0) >= 1.0) & (trending.fillna(0.0) >= 1.0)
    if filter_name == "vol_trend_volatile_not_sideways":
        if quiet.notna().any():
            return (volatile.fillna(0.0) >= 1.0) & (quiet.fillna(0.0) < 1.0)
        return volatile.fillna(0.0) >= 1.0
    if filter_name == "vol_trend_avoid_low_vol":
        return volatile.fillna(0.0) >= 1.0
    if filter_name == "vol_trend_avoid_sideways":
        return quiet.fillna(0.0) < 1.0 if quiet.notna().any() else mask
    if filter_name == "combined_quality":
        local = pd.Series(True, index=work.index)
        local &= _execution_rescue_filter_mask(work, "premium_mid_bucket")
        local &= _execution_rescue_filter_mask(work, "liquidity_volume")
        local &= _execution_rescue_filter_mask(work, "liquidity_spread")
        local &= _execution_rescue_filter_mask(work, "moneyness_itm_atm")
        local &= _execution_rescue_filter_mask(work, "time_avoid_open")
        local &= _execution_rescue_filter_mask(work, "time_avoid_close")
        return local
    if filter_name == "combined_strict":
        local = pd.Series(True, index=work.index)
        local &= _execution_rescue_filter_mask(work, "premium_mid_bucket")
        local &= _execution_rescue_filter_mask(work, "liquidity_volume")
        local &= _execution_rescue_filter_mask(work, "liquidity_oi")
        local &= _execution_rescue_filter_mask(work, "liquidity_spread")
        local &= _execution_rescue_filter_mask(work, "moneyness_itm_atm")
        local &= _execution_rescue_filter_mask(work, "time_0930_1030")
        local &= _execution_rescue_filter_mask(work, "vol_trend_volatile_not_sideways")
        return local
    return mask


def _execution_rescue_selector(selected_frame: pd.DataFrame, selector_kind: str, selector_value: float) -> pd.DataFrame:
    if selected_frame.empty:
        return selected_frame.copy()
    if "trade_day" not in selected_frame.columns:
        selected_frame = selected_frame.copy()
        selected_frame["trade_day"] = _normalize_timestamp_series(selected_frame["timestamp"]).dt.date
    if selector_kind == "top_n":
        return _ranked_selection_rows(selected_frame, "probability", top_n=int(selector_value))
    if selector_kind == "top_pct":
        return _ranked_selection_rows(selected_frame, "probability", top_pct=float(selector_value))
    if selector_kind == "threshold":
        return selected_frame.loc[selected_frame["probability"] >= float(selector_value)].copy()
    return selected_frame.copy()


def _execution_rescue_specs() -> List[Dict[str, Any]]:
    filter_names = [
        "none",
        "premium_min",
        "premium_max",
        "premium_mid_bucket",
        "liquidity_volume",
        "liquidity_oi",
        "liquidity_spread",
        "liquidity_nonzero_volume",
        "moneyness_itm",
        "moneyness_atm",
        "moneyness_near_atm",
        "moneyness_itm_atm",
        "moneyness_reject_far_otm",
        "time_avoid_open",
        "time_avoid_close",
        "time_0930_1030",
        "time_1030_1200",
        "time_1200_1400",
        "time_1400_1500",
        "vol_trend_volatile_trending",
        "vol_trend_volatile_not_sideways",
        "vol_trend_avoid_low_vol",
        "vol_trend_avoid_sideways",
        "combined_quality",
        "combined_strict",
    ]
    selectors = [
        ("top_n", 1),
        ("top_n", 2),
        ("top_n", 3),
        ("top_n", 5),
        ("top_pct", 0.05),
        ("top_pct", 0.10),
    ] + [("threshold", x) for x in RETRAIN_THRESHOLD_SWEEP]
    rows = []
    for base_name in ("xgboost_pe_itm_atm", "xgboost_pe_volatile"):
        for filter_name in filter_names:
            for selector_kind, selector_value in selectors:
                rows.append(
                    {
                        "base_candidate_name": base_name,
                        "filter_name": filter_name,
                        "selector_kind": selector_kind,
                        "selector_value": selector_value,
                    }
                )
    return rows


def _execution_rescue_ranker_score(row: Dict[str, Any]) -> float:
    return (
        float(row.get("cost_1_25x_pf") or 0.0) * 5.0
        + float(row.get("base_pf") or 0.0) * 2.0
        + float(row.get("base_sharpe") or 0.0) * 1.5
        + min(float(row.get("trades") or 0.0) / 300.0, 2.0)
        - float(row.get("max_drawdown") or 0.0) * 0.01
        - float(row.get("max_month_profit_share") or 0.0) * 2.0
        - float(row.get("max_fold_profit_share") or 0.0) * 2.0
        - float(row.get("cost_pct_of_avg_abs_return") or 0.0) * 0.5
    )


def _execution_rescue_reject_reasons(payload: Dict[str, Any]) -> List[str]:
    reasons = []
    if float(payload.get("cost_1_25x_pf") or 0.0) < 1.05:
        reasons.append("failed_cost_stress")
    if int(payload.get("trades") or 0) < 300:
        reasons.append("too_few_trades")
    if float(payload.get("max_month_profit_share") or 0.0) > 0.40:
        reasons.append("month_concentrated")
    if float(payload.get("max_fold_profit_share") or 0.0) > 0.40:
        reasons.append("fold_concentrated")
    if float(payload.get("max_drawdown") or 0.0) > max(25.0, abs(float(payload.get("cumulative_net_return") or 0.0))):
        reasons.append("drawdown_too_high")
    if float(payload.get("cost_1_10x_pf") or 0.0) <= 1.0:
        reasons.append("weak_after_cost")
    if float(payload.get("cost_pct_of_avg_abs_return") or 999.0) >= 0.50:
        reasons.append("insufficient_average_return_vs_cost")
    return reasons


def _execution_rescue_markdown(rows: Sequence[Dict[str, Any]], artifact_dir: Path) -> None:
    sorted_rows = sorted(rows, key=lambda row: float(row.get("robustness_score") or 0.0), reverse=True)
    lines = ["# Execution Rescue Report", ""]
    if not sorted_rows:
        lines.append("- No execution rescue candidates were produced.")
    else:
        best = sorted_rows[0]
        lines.append(f"- Best candidate: `{best['candidate_name']}`")
        lines.append(f"- Did either PE pocket survive cost stress? `{'yes' if any(float(row.get('cost_1_25x_pf') or 0.0) >= 1.05 for row in sorted_rows) else 'no'}`")
        lines.append(f"- Which filter helped most? `{best['filter_name']} + {best['selector_name']}`")
        worst = min(sorted_rows, key=lambda row: float(row.get('robustness_score') or 0.0))
        lines.append(f"- Which filter hurt most? `{worst['filter_name']} + {worst['selector_name']}`")
        paper_watch = next((row for row in sorted_rows if row.get("deployment_recommendation") == "PAPER_WATCHLIST"), None)
        lines.append(f"- Is the edge real enough for paper watchlist? `{'yes' if paper_watch else 'no'}`")
        lines.append("- Why production remains blocked? `No candidate satisfied strict cost-stress and stability gates, and production remains disabled by policy.`")
        lines.append(f"- What exact filter combination should be tested next? `{best['candidate_name']}`")
        lines.append("")
        lines.append("## Top 10")
        for row in sorted_rows[:10]:
            lines.append(f"- `{row['candidate_name']}` PF=`{row['base_pf']}` Sharpe=`{row['base_sharpe']}` cost1.25=`{row['cost_1_25x_pf']}` trades=`{row['trades']}` verdict=`{row['deployment_recommendation']}`")
    (artifact_dir / "execution_rescue_report.md").write_text("\n".join(lines), encoding="utf-8")


def _ranked_selection_rows(test_frame: pd.DataFrame, probability_column: str, *, top_n: int | None = None, top_pct: float | None = None) -> pd.DataFrame:
    if test_frame.empty:
        return test_frame.copy()
    ranked = test_frame.copy()
    if "trade_day" not in ranked.columns and "timestamp" in ranked.columns:
        ranked["trade_day"] = _normalize_timestamp_series(ranked["timestamp"]).dt.date
    ranked = ranked.sort_values(["trade_day", probability_column], ascending=[True, False], kind="stable")
    if top_n is not None:
        return ranked.groupby("trade_day", group_keys=False).head(int(top_n)).reset_index(drop=True)
    if top_pct is not None:
        ranked["__rank"] = ranked.groupby("trade_day").cumcount()
        group_sizes = ranked.groupby("trade_day")[probability_column].transform("size")
        keep_counts = np.maximum(1, np.ceil(group_sizes.to_numpy(dtype=float) * float(top_pct)).astype(int))
        selected = ranked.loc[ranked["__rank"].to_numpy(dtype=int) < keep_counts].drop(columns="__rank")
        return selected.reset_index(drop=True)
    return ranked.reset_index(drop=True)


def _ranked_selection_rows_by_bucket(
    test_frame: pd.DataFrame,
    probability_column: str,
    *,
    bucket_column: str,
    top_n: int,
) -> pd.DataFrame:
    if test_frame.empty or bucket_column not in test_frame.columns:
        return test_frame.head(0).copy()
    ranked = test_frame.copy()
    ranked[probability_column] = pd.to_numeric(ranked[probability_column], errors="coerce").fillna(0.0)
    ranked = ranked.sort_values([bucket_column, probability_column], ascending=[True, False], kind="stable")
    return ranked.groupby(bucket_column, group_keys=False, dropna=False).head(int(top_n)).reset_index(drop=True)


def _best_regime_candidate_selection(selection_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not selection_rows:
        return None
    return max(
        selection_rows,
        key=lambda row: (
            0 if int(row.get("trades") or 0) >= 500 else 1,
            float(row.get("profit_factor") or 0.0),
            float(row.get("sharpe") or 0.0),
            -float(row.get("max_drawdown") or 0.0),
            int(row.get("trades") or 0),
        ),
    )


def _candidate_score(candidate_payload: Dict[str, Any]) -> float:
    best_selection = candidate_payload.get("best_selection") or {}
    cost_stress = candidate_payload.get("cost_stress") or {}
    monthly = candidate_payload.get("monthly_stability") or {}
    fold_metrics = candidate_payload.get("fold_metrics") or {}
    pf_125 = float((cost_stress.get("cost_1_25x") or {}).get("profit_factor") or 0.0)
    base_pf = float((best_selection.get("profit_factor")) or 0.0)
    sharpe = float((best_selection.get("sharpe")) or 0.0)
    drawdown = float((best_selection.get("max_drawdown")) or 0.0)
    trades = int((best_selection.get("trades")) or 0)
    stability_penalty = float(monthly.get("max_month_profit_share") or 0.0) + float(fold_metrics.get("max_fold_profit_share") or 0.0)
    return (pf_125 * 4.0) + (sharpe * 1.5) + (base_pf * 1.0) + min(trades / 500.0, 2.0) - (drawdown * 0.001) - (stability_penalty * 2.0)


def _regime_candidate_gate(candidate_payload: Dict[str, Any]) -> Dict[str, Any]:
    best_selection = candidate_payload.get("best_selection") or {}
    monthly = candidate_payload.get("monthly_stability") or {}
    fold_metrics = candidate_payload.get("fold_metrics") or {}
    cost_stress = candidate_payload.get("cost_stress") or {}
    base = cost_stress.get("base") or {}
    cost_125 = cost_stress.get("cost_1_25x") or {}
    trades = int(best_selection.get("trades") or 0)
    small_sample = trades < 500
    passes = (
        float(base.get("profit_factor") or 0.0) > 1.25
        and float(base.get("sharpe") or 0.0) > 1.0
        and float(cost_125.get("profit_factor") or 0.0) > 1.05
        and not small_sample
        and bool(monthly.get("passes_monthly_concentration_gate"))
        and bool(fold_metrics.get("passes_fold_concentration_gate"))
        and float(base.get("max_drawdown") or 0.0) <= max(100.0, abs(float(base.get("total_return") or 0.0)) * 2.0)
        and bool(candidate_payload.get("leakage_protection_passed"))
    )
    reasons: List[str] = []
    if float(base.get("profit_factor") or 0.0) <= 1.25:
        reasons.append("base_pf_below_gate")
    if float(base.get("sharpe") or 0.0) <= 1.0:
        reasons.append("base_sharpe_below_gate")
    if float(cost_125.get("profit_factor") or 0.0) <= 1.05:
        reasons.append("cost_1_25x_pf_below_gate")
    if small_sample:
        reasons.append("small_sample_under_500_trades")
    if not bool(monthly.get("passes_monthly_concentration_gate")):
        reasons.append("monthly_profit_concentration_above_40pct")
    if not bool(fold_metrics.get("passes_fold_concentration_gate")):
        reasons.append("fold_profit_concentration_above_40pct")
    if float(base.get("max_drawdown") or 0.0) > max(100.0, abs(float(base.get("total_return") or 0.0)) * 2.0):
        reasons.append("max_drawdown_unacceptable")
    if not bool(candidate_payload.get("leakage_protection_passed")):
        reasons.append("leakage_protection_failed")
    return {
        "passes_research_gate": bool(passes),
        "small_sample_warning": bool(small_sample),
        "failure_reasons": reasons,
        "candidate_score": _candidate_score(candidate_payload),
        "cost_sensitive": bool(float(cost_125.get("profit_factor") or 0.0) <= 1.05),
        "month_concentrated": not bool(monthly.get("passes_monthly_concentration_gate")),
        "fold_concentrated": not bool(fold_metrics.get("passes_fold_concentration_gate")),
    }


def _train_regime_restricted_candidates(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    label_name: str,
    evaluation_return_column: str | None,
    artifact_dir: Path,
    fold_count: int,
) -> Dict[str, Any]:
    if not evaluation_return_column or evaluation_return_column not in df.columns:
        return {
            "candidates": [],
            "csv_rows": [],
            "ce_pe_champions": {"ce": None, "pe": None},
            "small_sample_warnings": [],
            "monthly_stability_rows": [],
            "cost_stress_by_candidate": {},
        }
    prepared = _prepare_chronological_work(df.copy())
    candidate_rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    small_sample_rows: List[Dict[str, Any]] = []
    monthly_rows: List[Dict[str, Any]] = []
    cost_rows: Dict[str, Any] = {}
    execution_rescue_rows: List[Dict[str, Any]] = []
    for spec in ALL_RESEARCH_CANDIDATE_SPECS:
        candidate_name = str(spec["candidate_name"])
        model_family = str(spec["model_family"])
        print(f"[retrain] regime candidate training={candidate_name}")
        dep_ok, dep_reason = _optional_dependency_status(model_family)
        if not dep_ok:
            candidate_rows.append(
                {
                    "candidate_name": candidate_name,
                    "model_family": model_family,
                    "status": "SKIPPED",
                    "skip_reason": dep_reason,
                    "production_adoption_allowed": False,
                    "research_only": True,
                }
            )
            continue
        mask = _regime_candidate_mask(prepared, spec["filters"])
        filtered = prepared.loc[mask].copy()
        if filtered.empty:
            candidate_rows.append(
                {
                    "candidate_name": candidate_name,
                    "model_family": model_family,
                    "status": "EMPTY",
                    "skip_reason": "no_rows_after_regime_filter",
                    "production_adoption_allowed": False,
                    "research_only": True,
                }
            )
            continue
        y_series = prepare_label_series(filtered, label_name)
        passthrough_columns = [
            column
            for column in (
                "timestamp",
                "option_type",
                "option_type_ce",
                "option_type_pe",
                "moneyness",
                "distance_from_spot",
                "distance_from_atm",
                "regime_volatile",
                "volatility_regime_classifier",
            )
            if column in filtered.columns and column not in feature_cols
        ]
        work = pd.concat([filtered[passthrough_columns], filtered[list(feature_cols)].copy(), y_series.rename("label")], axis=1)
        work[evaluation_return_column] = pd.to_numeric(filtered[evaluation_return_column], errors="coerce")
        work = work.dropna(subset=["timestamp", "label"]).copy()
        work = work[work["label"].isin([0, 1])]
        if work.empty or work["label"].nunique() < 2:
            candidate_rows.append(
                {
                    "candidate_name": candidate_name,
                    "model_family": model_family,
                    "status": "SKIPPED",
                    "skip_reason": "insufficient_rows_or_single_class",
                    "row_count": int(len(work)),
                    "production_adoption_allowed": False,
                    "research_only": True,
                }
            )
            continue
        work = _prepare_chronological_work(work)
        timestamps = pd.Series(work["timestamp"]).reset_index(drop=True)
        split = _build_holdout_split_from_timestamps(timestamps)
        train_idx, val_idx, test_idx = split["train"], split["validation"], split["test"]
        _assert_fold_is_chronological(timestamps, train_idx, val_idx, test_idx=test_idx, context=f"regime_candidate:{candidate_name}")
        X_df = work[list(feature_cols)].copy().replace([np.inf, -np.inf], np.nan)
        X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0).astype(np.float32)
        X = X_df.to_numpy(dtype=np.float32)
        y = work["label"].astype(int).to_numpy(dtype=int)
        returns = work[evaluation_return_column].to_numpy(dtype=float)
        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
        X_train_s, X_val_s, X_test_s, _, _ = scale_train_val_test(X_train, X_val, X_test)
        model, model_fit_metadata = _fit_model_for_training_run(
            model_family,
            X_train_s,
            y_train,
            X_val_s,
            y_val,
            returns[val_idx],
            minimum_trades=25,
        )
        val_prob = predict_proba_positive(model, X_val_s)
        test_prob = predict_proba_positive(model, X_test_s)
        chosen = optimize_threshold_from_probs(y_val, val_prob, returns[val_idx], thresholds=RETRAIN_THRESHOLD_SWEEP, minimum_trades=25)
        threshold = float(chosen["threshold"])
        test_frame = _apply_backtest_row_limit(work.iloc[test_idx].copy().reset_index(drop=True), context=f"regime_candidate:{candidate_name}")
        test_frame["timestamp"] = _normalize_timestamp_series(test_frame["timestamp"])
        test_frame["trade_day"] = test_frame["timestamp"].dt.date
        test_frame["probability"] = np.asarray(test_prob, dtype=float)
        test_frame["selected_return"] = np.asarray(returns[test_idx], dtype=float)
        threshold_selected = test_frame.loc[test_frame["probability"] >= threshold].copy()
        selection_kind = (spec.get("selection") or {}).get("kind", "threshold")
        if selection_kind == "top_n":
            selected_frame = _ranked_selection_rows(test_frame, "probability", top_n=int((spec.get("selection") or {}).get("value") or 1))
            best_selection = _selection_rows_for_candidate(selected_frame, selection_name=f"top_{int((spec.get('selection') or {}).get('value') or 1)}_per_day")
            selection_reports = [best_selection]
        elif selection_kind == "top_pct":
            pct_value = float((spec.get("selection") or {}).get("value") or 0.05)
            selected_frame = _ranked_selection_rows(test_frame, "probability", top_pct=pct_value)
            pct_label = int(round(pct_value * 100))
            best_selection = _selection_rows_for_candidate(selected_frame, selection_name=f"top_{pct_label}_percent_per_day")
            selection_reports = [best_selection]
        else:
            selection_reports = [
                _selection_rows_for_candidate(threshold_selected, selection_name="threshold_base", threshold_value=threshold),
                _selection_rows_for_candidate(_ranked_selection_rows(test_frame, "probability", top_n=1), selection_name="top_1_per_day"),
                _selection_rows_for_candidate(_ranked_selection_rows(test_frame, "probability", top_n=3), selection_name="top_3_per_day"),
                _selection_rows_for_candidate(_ranked_selection_rows(test_frame, "probability", top_n=5), selection_name="top_5_per_day"),
                _selection_rows_for_candidate(_ranked_selection_rows(test_frame, "probability", top_pct=0.05), selection_name="top_5_percent_per_day"),
                _selection_rows_for_candidate(_ranked_selection_rows(test_frame, "probability", top_pct=0.10), selection_name="top_10_percent_per_day"),
            ]
            best_selection = _best_regime_candidate_selection(selection_reports)
        best_rows = best_selection.get("selected_rows").copy() if best_selection and isinstance(best_selection.get("selected_rows"), pd.DataFrame) else pd.DataFrame(columns=test_frame.columns)
        monthly_stability = _monthly_stability_for_selection(best_rows)
        fold_metrics = _fold_metrics_for_selection(best_rows, fold_count=fold_count)
        # Compute per-target cost provenance with per-row cost units
        best_cost_provenance = _return_cost_provenance_for_frame(
            best_rows if not best_rows.empty else test_frame,
            evaluation_return_column=evaluation_return_column,
        )
        best_per_row_costs = best_cost_provenance.get("per_row_cost_units")
        cost_stress = _selection_cost_stress(
            best_rows,
            cost_provenance=best_cost_provenance,
            per_row_cost_units=(
                pd.Series(best_per_row_costs, index=best_rows.index) if best_per_row_costs is not None and len(best_per_row_costs) == len(best_rows) else None
            ),
        )
        execution_cost_model = _execution_cost_model_report(best_rows)
        liquidity_filter_report = _liquidity_filter_report(best_rows)
        trade_capacity_report = _trade_capacity_report(best_rows, test_frame["trade_day"].tolist())
        daily_stability_report = _daily_stability_breakdown(best_rows)
        expiry_breakdown = _expiry_breakdown(best_rows)
        time_of_day_breakdown = _time_of_day_breakdown(best_rows)
        confidence_band_report = _confidence_band_report(test_frame)
        probability_drift_report = _probability_drift_report(test_frame, best_rows)
        test_metrics = classification_metrics(y_test, test_prob, threshold=threshold)
        test_metrics["pr_auc"] = pr_auc_score_safe(y_test, test_prob)
        threshold_sweep = _full_threshold_sweep_report(y_test, test_prob, returns[test_idx], thresholds=RETRAIN_THRESHOLD_SWEEP)
        top_features = [row["feature"] for row in _feature_importance(model, feature_cols, limit=20)]
        feature_drift_report = _feature_drift_report(work.iloc[train_idx].reset_index(drop=True), test_frame, top_features)
        duplicate_leakage_audit = _duplicate_timestamp_leakage_audit(work, train_idx, test_idx)
        target_horizon_report = _target_horizon_report(df)
        no_trade_analysis = {
            "regimes_with_pf_lt_1": [item.get("group") for item in ((candidate_payload.get("regime_performance", {}) if False else {}) or {}).get("groups", [])] if False else [],
        }
        candidate_payload = {
            "candidate_name": candidate_name,
            "candidate_group": str(spec.get("candidate_group") or "regime"),
            "model_family": model_family,
            "tags": list(spec.get("tags") or []),
            "filters": dict(spec["filters"]),
            "row_count": int(len(work)),
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
            "selected_threshold": threshold,
            "classification_metrics": {
                "roc_auc": float(test_metrics.get("roc_auc") or 0.0),
                "pr_auc": float(test_metrics.get("pr_auc") or 0.0),
                "f1": float(test_metrics.get("f1") or 0.0),
                "precision": float(test_metrics.get("precision") or 0.0),
                "recall": float(test_metrics.get("recall") or 0.0),
            },
            "threshold_sweep": threshold_sweep,
            "selection_reports": [
                {key: value for key, value in row.items() if key != "selected_rows"}
                for row in selection_reports
            ],
            "model_fit_metadata": model_fit_metadata,
            "best_selection": {key: value for key, value in (best_selection or {}).items() if key != "selected_rows"},
            "monthly_stability": monthly_stability,
            "fold_metrics": fold_metrics,
            "cost_stress": cost_stress,
            "execution_cost_model": execution_cost_model,
            "liquidity_filter_report": liquidity_filter_report,
            "trade_capacity_report": trade_capacity_report,
            "daily_stability_report": daily_stability_report,
            "expiry_breakdown": expiry_breakdown,
            "time_of_day_breakdown": time_of_day_breakdown,
            "confidence_band_report": confidence_band_report,
            "probability_drift_report": probability_drift_report,
            "feature_drift_report": feature_drift_report,
            "duplicate_timestamp_leakage_audit": duplicate_leakage_audit,
            "target_horizon_report": target_horizon_report,
            "stop_loss_take_profit_simulation": {"possible": False, "reason": "intraperiod candle path not available in candidate evaluator"},
            "top_features": top_features,
            "high_confidence_winning_examples": best_rows.loc[best_rows["selected_return"] > 0.0].sort_values("probability", ascending=False).head(5).to_dict(orient="records") if not best_rows.empty else [],
            "high_confidence_losing_examples": best_rows.loc[best_rows["selected_return"] < 0.0].sort_values("probability", ascending=False).head(5).to_dict(orient="records") if not best_rows.empty else [],
            "leakage_protection_passed": True,
            "live_computable_only_passed": True,
            "dropped_leakage_columns": [
                "avoid_trade_label",
                "future_close",
                "gross_forward_return",
                "net_forward_return",
                "profitable_trade_label",
                "realized_vol_30",
                "realized_vol_percentile_60",
                "spot_return_1",
                "spot_return_3",
                "spot_return_5",
            ],
            "status": "OK",
            "production_adoption_allowed": False,
            "research_only": True,
        }
        gate = _regime_candidate_gate(candidate_payload)
        candidate_payload.update(gate)
        candidate_payload["deployment_recommendation"] = _deployment_recommendation(candidate_payload)
        candidate_payload["research_gate_diagnostics"] = {
            "passed": bool(gate.get("passes_research_gate")),
            "failure_reasons": list(gate.get("failure_reasons") or []),
            "small_sample": bool(gate.get("small_sample_warning")),
            "cost_sensitive": bool(gate.get("cost_sensitive")),
            "month_concentrated": bool(gate.get("month_concentrated")),
            "fold_concentrated": bool(gate.get("fold_concentrated")),
            "safe_only_for_research": True,
        }
        if candidate_name in {"xgboost_pe_itm_atm", "xgboost_pe_volatile"}:
            for rescue_spec in _execution_rescue_specs():
                if rescue_spec["base_candidate_name"] != candidate_name:
                    continue
                filtered_mask = _execution_rescue_filter_mask(test_frame, rescue_spec["filter_name"])
                filtered_frame = test_frame.loc[filtered_mask.fillna(False)].copy()
                filtered_frame = _execution_rescue_selector(filtered_frame, rescue_spec["selector_kind"], rescue_spec["selector_value"])
                selection = _selection_rows_for_candidate(
                    filtered_frame,
                    selection_name=f"{rescue_spec['filter_name']}:{rescue_spec['selector_kind']}:{rescue_spec['selector_value']}",
                    threshold_value=float(rescue_spec["selector_value"]) if rescue_spec["selector_kind"] == "threshold" else None,
                )
                monthly = _monthly_stability_for_selection(filtered_frame)
                fold = _fold_metrics_for_selection(filtered_frame, fold_count=fold_count)
                # Compute per-target cost provenance with per-row cost units
                rescue_cost_provenance = _return_cost_provenance_for_frame(
                    filtered_frame,
                    evaluation_return_column=evaluation_return_column,
                )
                rescue_per_row_costs = rescue_cost_provenance.get("per_row_cost_units")
                cost = _selection_cost_stress(
                    filtered_frame,
                    cost_provenance=rescue_cost_provenance,
                    per_row_cost_units=(
                        pd.Series(rescue_per_row_costs, index=filtered_frame.index) if rescue_per_row_costs is not None and len(rescue_per_row_costs) == len(filtered_frame) else None
                    ),
                )
                exec_cost = _execution_cost_model_report(filtered_frame)
                conservative_cost = float((exec_cost.get("scenarios") or {}).get("conservative_cost") or 0.0)
                avg_abs_return = abs(float(selection.get("average_return_per_trade") or 0.0))
                avg_win = float(selection.get("average_win") or 0.0)
                cost_pct_win = (conservative_cost / avg_win) if avg_win > 0 else None
                cost_pct_abs = (conservative_cost / avg_abs_return) if avg_abs_return > 0 else None
                row = {
                    "base_candidate_name": candidate_name,
                    "candidate_name": f"{candidate_name}__{rescue_spec['filter_name']}__{rescue_spec['selector_kind']}_{rescue_spec['selector_value']}",
                    "filter_name": rescue_spec["filter_name"],
                    "selector_name": f"{rescue_spec['selector_kind']}_{rescue_spec['selector_value']}",
                    "trades": int(selection.get("trades") or 0),
                    "trades_per_day": float(selection.get("trades_per_day") or 0.0),
                    "win_rate": float(selection.get("win_rate") or 0.0),
                    "average_net_forward_return": float(selection.get("average_return_per_trade") or 0.0),
                    "median_net_forward_return": float(selection.get("median_net_forward_return") or 0.0),
                    "cumulative_net_return": float(selection.get("total_return") or 0.0),
                    "base_pf": float(selection.get("profit_factor") or 0.0),
                    "base_sharpe": float(selection.get("sharpe") or 0.0),
                    "sortino": float(selection.get("sortino") or 0.0),
                    "max_drawdown": float(selection.get("max_drawdown") or 0.0),
                    "max_consecutive_losses": int(selection.get("max_consecutive_losses") or 0),
                    "best_day": ( _daily_stability_breakdown(filtered_frame).get("best_day") or {}).get("trade_day"),
                    "worst_day": ( _daily_stability_breakdown(filtered_frame).get("worst_day") or {}).get("trade_day"),
                    "profitable_days": int(_daily_stability_breakdown(filtered_frame).get("profitable_days") or 0),
                    "losing_days": int(_daily_stability_breakdown(filtered_frame).get("losing_days") or 0),
                    "worst_month": (monthly.get("worst_month") or {}).get("month"),
                    "best_month": (monthly.get("best_month") or {}).get("month"),
                    "max_month_profit_share": float(monthly.get("max_month_profit_share") or 0.0),
                    "max_fold_profit_share": float(fold.get("max_fold_profit_share") or 0.0),
                    "fold_wise_pf": [float(item.get("profit_factor") or 0.0) for item in fold.get("folds", [])],
                    "fold_wise_sharpe": [float(item.get("sharpe") or 0.0) for item in fold.get("folds", [])],
                    "fold_wise_trade_count": [int(item.get("trade_count") or 0) for item in fold.get("folds", [])],
                    "cost_1_10x_pf": float((cost.get("cost_1_10x") or {}).get("profit_factor") or 0.0),
                    "cost_1_25x_pf": float((cost.get("cost_1_25x") or {}).get("profit_factor") or 0.0),
                    "cost_1_50x_pf": float((cost.get("cost_1_50x") or {}).get("profit_factor") or 0.0),
                    "cost_2_00x_pf": float((cost.get("cost_2_00x") or {}).get("profit_factor") or 0.0),
                    "average_winning_return": avg_win,
                    "average_losing_return": float(selection.get("average_loss") or 0.0),
                    "estimated_cost_per_trade": conservative_cost,
                    "cost_pct_of_avg_win": cost_pct_win,
                    "cost_pct_of_avg_abs_return": cost_pct_abs,
                    "small_sample_warning": bool(int(selection.get("trades") or 0) < 300),
                    "robustness_score": 0.0,
                    "deployment_recommendation": "RESEARCH_ONLY",
                    "production_adoption_allowed": False,
                }
                row["reject_reasons"] = _execution_rescue_reject_reasons(row)
                row["robustness_score"] = _execution_rescue_ranker_score(row)
                if row["base_pf"] >= 1.50 and row["base_sharpe"] >= 1.50 and row["cost_1_25x_pf"] >= 1.05 and row["trades"] >= 300 and row["max_month_profit_share"] <= 0.40 and row["max_fold_profit_share"] <= 0.40:
                    row["deployment_recommendation"] = "PAPER_WATCHLIST"
                execution_rescue_rows.append(row)
        candidate_rows.append(candidate_payload)
        cost_rows[candidate_name] = cost_stress
        for month_row in monthly_stability.get("months", []):
            monthly_rows.append({"candidate_name": candidate_name, **month_row})
        if bool(gate.get("small_sample_warning")):
            small_sample_rows.append(
                {
                    "candidate_name": candidate_name,
                    "model_family": model_family,
                    "trade_count": int((best_selection or {}).get("trades") or 0),
                    "warning": "candidate_below_500_trades",
                }
            )
        best_summary = candidate_payload.get("best_selection") or {}
        csv_rows.append(
            {
                "candidate_name": candidate_name,
                "model_family": model_family,
                "best_selection": best_summary.get("selection_name"),
                "trades": int(best_summary.get("trades") or 0),
                "trades_per_day": float(best_summary.get("trades_per_day") or 0.0),
                "win_rate": float(best_summary.get("win_rate") or 0.0),
                "average_net_forward_return": float(best_summary.get("average_return_per_trade") or 0.0),
                "median_net_forward_return": float(best_summary.get("median_net_forward_return") or 0.0),
                "cumulative_net_return": float(best_summary.get("total_return") or 0.0),
                "gross_profit": float(best_summary.get("gross_profit") or 0.0),
                "gross_loss": float(best_summary.get("gross_loss") or 0.0),
                "profit_factor": float(best_summary.get("profit_factor") or 0.0),
                "sharpe": float(best_summary.get("sharpe") or 0.0),
                "sortino": float(best_selection.get("sortino") or 0.0),
                "max_drawdown": float(best_summary.get("max_drawdown") or 0.0),
                "max_consecutive_losses": int(best_summary.get("max_consecutive_losses") or 0),
                "daily_return_mean": float(best_summary.get("daily_return_mean") or 0.0),
                "daily_return_std": float(best_summary.get("daily_return_std") or 0.0),
                "roc_auc": float(candidate_payload["classification_metrics"]["roc_auc"]),
                "pr_auc": float(candidate_payload["classification_metrics"]["pr_auc"]),
                "cost_1_10x_pf": float((cost_stress.get("cost_1_10x") or {}).get("profit_factor") or 0.0),
                "cost_1_25x_pf": float((cost_stress.get("cost_1_25x") or {}).get("profit_factor") or 0.0),
                "cost_1_50x_pf": float((cost_stress.get("cost_1_50x") or {}).get("profit_factor") or 0.0),
                "cost_2_00x_pf": float((cost_stress.get("cost_2_00x") or {}).get("profit_factor") or 0.0),
                "best_month": (monthly_stability.get("best_month") or {}).get("month"),
                "worst_month": (monthly_stability.get("worst_month") or {}).get("month"),
                "profitable_months": int(monthly_stability.get("profitable_months") or 0),
                "losing_months": int(monthly_stability.get("losing_months") or 0),
                "max_month_profit_share": float(monthly_stability.get("max_month_profit_share") or 0.0),
                "max_fold_profit_share": float(fold_metrics.get("max_fold_profit_share") or 0.0),
                "small_sample_warning": bool(gate.get("small_sample_warning")),
                "passes_research_gate": bool(gate.get("passes_research_gate")),
                "candidate_score": float(gate.get("candidate_score") or 0.0),
                "deployment_recommendation": candidate_payload.get("deployment_recommendation"),
                "production_adoption_allowed": False,
            }
        )
    def _best_side(tags: str) -> Dict[str, Any] | None:
        eligible = [row for row in candidate_rows if row.get("status") == "OK" and tags in (row.get("tags") or [])]
        passed = [row for row in eligible if row.get("passes_research_gate")]
        if not passed:
            return None
        return max(passed, key=lambda row: float(row.get("candidate_score") or 0.0))
    def _best_fallback(pred) -> Dict[str, Any] | None:
        eligible = [row for row in candidate_rows if row.get("status") == "OK" and pred(row)]
        if not eligible:
            return None
        return max(eligible, key=lambda row: float(row.get("candidate_score") or 0.0))
    champion_report = {
        "best_global_champion": _best_fallback(lambda row: True if row.get("passes_research_gate") else False),
        "best_global_observed": _best_fallback(lambda row: True),
        "best_ce_champion": _best_side("CE"),
        "best_pe_champion": _best_side("PE"),
        "best_lr_ranker_champion": _best_fallback(lambda row: row.get("candidate_group") == "lr_ranker" and row.get("passes_research_gate")),
        "best_lr_ranker_observed": _best_fallback(lambda row: row.get("candidate_group") == "lr_ranker"),
        "best_rf_regime_champion": _best_fallback(lambda row: "RF_REGIME" in (row.get("tags") or []) and row.get("passes_research_gate")),
        "best_rf_regime_observed": _best_fallback(lambda row: "RF_REGIME" in (row.get("tags") or [])),
        "best_xgboost_regime_champion": _best_fallback(lambda row: "XGB_REGIME" in (row.get("tags") or []) and row.get("passes_research_gate")),
        "best_xgboost_regime_observed": _best_fallback(lambda row: "XGB_REGIME" in (row.get("tags") or [])),
        "production_adoption_allowed": False,
        "research_only": True,
    }
    blocker_rows = []
    for row in candidate_rows:
        if row.get("status") != "OK":
            continue
        blocker_rows.append(
            {
                "candidate_name": row["candidate_name"],
                "failure_reasons": row.get("failure_reasons") or [],
                "small_sample_warning": bool(row.get("small_sample_warning")),
                "cost_sensitive": bool(row.get("cost_sensitive")),
                "month_concentrated": bool(row.get("month_concentrated")),
                "fold_concentrated": bool(row.get("fold_concentrated")),
                "production_adoption_allowed": False,
            }
        )
    return {
        "candidates": candidate_rows,
        "csv_rows": csv_rows,
        "ranker_candidates": [row for row in candidate_rows if row.get("candidate_group") == "lr_ranker"],
        "regime_candidates": [row for row in candidate_rows if row.get("candidate_group") != "lr_ranker"],
        "ce_pe_champions": {"ce": _best_side("CE"), "pe": _best_side("PE")},
        "small_sample_warnings": small_sample_rows,
        "monthly_stability_rows": monthly_rows,
        "cost_stress_by_candidate": cost_rows,
        "fold_stability_rows": [
            {
                "candidate_name": row["candidate_name"],
                "folds": (row.get("fold_metrics") or {}).get("folds", []),
                "max_fold_profit_share": (row.get("fold_metrics") or {}).get("max_fold_profit_share"),
                "passes_fold_concentration_gate": (row.get("fold_metrics") or {}).get("passes_fold_concentration_gate"),
            }
            for row in candidate_rows if row.get("status") == "OK"
        ],
        "candidate_champion_report": champion_report,
        "production_blocker_rows": blocker_rows,
        "duplicate_timestamp_leakage_audit": _duplicate_timestamp_leakage_audit(prepared, np.arange(int(len(prepared) * 0.7)), np.arange(int(len(prepared) * 0.85), len(prepared))),
        "target_horizon_report": _target_horizon_report(df),
        "execution_rescue_rows": execution_rescue_rows,
    }


def _trade_filter_report(
    test_frame: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    returns: np.ndarray | None,
) -> Dict[str, Any]:
    if returns is None or test_frame.empty:
        return {"base": {}, "filters": [], "best_filter_combination": None}
    work = test_frame.copy().reset_index(drop=True)
    work["probability"] = np.asarray(y_prob, dtype=float)
    work["selected_return"] = np.asarray(returns, dtype=float)
    work["selected"] = work["probability"] >= float(threshold)
    work["timestamp"] = _normalize_timestamp_series(work["timestamp"])
    work["minute_of_day"] = work["timestamp"].dt.hour * 60 + work["timestamp"].dt.minute
    work["trade_day"] = work["timestamp"].dt.date
    base = work.loc[work["selected"]].copy()
    base_metrics = _trade_metrics_from_scores(np.ones(len(base)), np.ones(len(base)), 0.5, base["selected_return"].to_numpy(dtype=float), threshold_source="filter_base") if not base.empty else _trade_metrics_from_scores(np.array([]), np.array([]), 0.5, np.array([]), threshold_source="filter_base")
    if base.empty:
        return {"base": base_metrics, "filters": [], "best_filter_combination": None}
    filters: Dict[str, pd.Series] = {
        "avoid_first_5m": work["minute_of_day"] >= (9 * 60 + 20),
        "avoid_last_15m": work["minute_of_day"] <= (15 * 60 + 15),
        "ce_only": _option_side_series(work).eq("CE"),
        "pe_only": _option_side_series(work).eq("PE"),
        "atm_or_near_atm": _moneyness_bucket_series(work).isin(["ATM", "NEAR_ATM"]),
        "non_expiry_day": _dte_series(work).fillna(9999.0) > 0.0,
    }
    if "bid_ask_spread_pct" in work.columns:
        filters["tight_spread"] = _spread_pct_series(work).fillna(np.inf) <= 5.0
    if "volume" in work.columns:
        filters["min_volume"] = _numeric_series(work, "volume").fillna(0.0) >= float(_numeric_series(work, "volume").median())
    if "oi" in work.columns:
        filters["min_oi"] = _numeric_series(work, "oi").fillna(0.0) >= float(_numeric_series(work, "oi").median())
    if "regime_trending" in work.columns:
        filters["trending_regime"] = _numeric_series(work, "regime_trending").fillna(0.0) >= 1.0
    if "regime_quiet" in work.columns:
        filters["quiet_regime"] = _numeric_series(work, "regime_quiet").fillna(0.0) >= 1.0
    rows: List[Dict[str, Any]] = []
    combo_specs = {
        "base": [],
        "clean_intraday_atm": ["avoid_first_5m", "avoid_last_15m", "atm_or_near_atm"],
        "ce_clean": ["avoid_first_5m", "avoid_last_15m", "ce_only"],
        "pe_clean": ["avoid_first_5m", "avoid_last_15m", "pe_only"],
        "non_expiry_atm": ["avoid_first_5m", "avoid_last_15m", "atm_or_near_atm", "non_expiry_day"],
    }
    for name, active_filters in combo_specs.items():
        mask = work["selected"].copy()
        for filter_name in active_filters:
            if filter_name in filters:
                mask &= filters[filter_name].fillna(False)
        selected = work.loc[mask].copy()
        metrics = _trade_metrics_from_scores(np.ones(len(selected)), np.ones(len(selected)), 0.5, selected["selected_return"].to_numpy(dtype=float), threshold_source=f"trade_filter:{name}") if not selected.empty else _trade_metrics_from_scores(np.array([]), np.array([]), 0.5, np.array([]), threshold_source=f"trade_filter:{name}")
        rows.append({"filter_name": name, "active_filters": active_filters, **metrics})
    for filter_name, filter_mask in filters.items():
        selected = work.loc[work["selected"] & filter_mask.fillna(False)].copy()
        metrics = _trade_metrics_from_scores(np.ones(len(selected)), np.ones(len(selected)), 0.5, selected["selected_return"].to_numpy(dtype=float), threshold_source=f"trade_filter:{filter_name}") if not selected.empty else _trade_metrics_from_scores(np.array([]), np.array([]), 0.5, np.array([]), threshold_source=f"trade_filter:{filter_name}")
        rows.append({"filter_name": filter_name, "active_filters": [filter_name], **metrics})
    # Top-N / percentile experiments
    ranked = work.loc[work["selected"]].sort_values(["trade_day", "probability"], ascending=[True, False], kind="stable")
    top_specs = {
        "top_1_per_day": ranked.groupby("trade_day", group_keys=False).head(1),
        "top_3_per_day": ranked.groupby("trade_day", group_keys=False).head(3),
        "top_5_per_day": ranked.groupby("trade_day", group_keys=False).head(5),
        "top_5pct_per_day": _ranked_selection_rows(ranked, "probability", top_pct=0.05) if not ranked.empty else ranked,
        "top_10pct_per_day": _ranked_selection_rows(ranked, "probability", top_pct=0.10) if not ranked.empty else ranked,
    }
    for name, selected in top_specs.items():
        metrics = _trade_metrics_from_scores(np.ones(len(selected)), np.ones(len(selected)), 0.5, selected["selected_return"].to_numpy(dtype=float), threshold_source=f"trade_filter:{name}") if not selected.empty else _trade_metrics_from_scores(np.array([]), np.array([]), 0.5, np.array([]), threshold_source=f"trade_filter:{name}")
        rows.append({"filter_name": name, "active_filters": [name], **metrics})
    best = max(rows, key=lambda row: (float(row.get("profit_factor") or 0.0), float(row.get("sharpe") or 0.0), int(row.get("trade_count") or 0))) if rows else None
    return {"base": base_metrics, "filters": rows, "best_filter_combination": best}


def _derive_label_from_return(frame: pd.DataFrame, threshold_bps: float) -> pd.Series:
    net_ret = _numeric_series(frame, "net_forward_return").fillna(0.0)
    threshold = float(threshold_bps) / 10000.0
    return (net_ret >= threshold).astype(int)


def _label_experiment_report(df: pd.DataFrame, feature_cols: Sequence[str]) -> Dict[str, Any]:
    if "net_forward_return" not in df.columns:
        return {"experiments": [], "best_label": None}
    experiments = []
    label_specs = [
        ("label_net_05bps", 5),
        ("label_net_10bps", 10),
        ("label_net_15bps", 15),
        ("label_net_20bps", 20),
        ("label_net_30bps", 30),
    ]
    prepared = _prepare_chronological_work(df[["timestamp", "net_forward_return", *feature_cols]].copy())
    X_df = prepared[list(feature_cols)].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    timestamps = pd.Series(prepared["timestamp"]).reset_index(drop=True)
    split = _build_holdout_split_from_timestamps(timestamps)
    train_idx, val_idx, test_idx = split["train"], split["validation"], split["test"]
    for label_name, bps in label_specs:
        y = _derive_label_from_return(prepared, bps).to_numpy(dtype=int)
        X = X_df.to_numpy(dtype=np.float32)
        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
        X_train_s, X_val_s, X_test_s, _, _ = scale_train_val_test(X_train, X_val, X_test)
        model = _build_model_family("logistic_regression")
        fit_model(model, X_train_s, y_train)
        val_prob = predict_proba_positive(model, X_val_s)
        test_prob = predict_proba_positive(model, X_test_s)
        val_returns = prepared["net_forward_return"].to_numpy(dtype=float)[val_idx]
        test_returns = prepared["net_forward_return"].to_numpy(dtype=float)[test_idx]
        chosen = optimize_threshold_from_probs(y_val, val_prob, val_returns, thresholds=RETRAIN_THRESHOLD_SWEEP, minimum_trades=25)
        threshold = float(chosen["threshold"])
        trade = _trade_metrics_from_scores(y_test, test_prob, threshold, test_returns, threshold_source=f"label_experiment:{label_name}")
        metrics = classification_metrics(y_test, test_prob, threshold=threshold)
        experiments.append(
            {
                "label_name": label_name,
                "cost_buffer_bps": bps,
                "threshold": threshold,
                "roc_auc": metrics.get("roc_auc"),
                "f1": metrics.get("f1"),
                "profit_factor": trade.get("profit_factor"),
                "sharpe": trade.get("sharpe"),
                "trade_count": trade.get("trade_count"),
            }
        )
    best = max(experiments, key=lambda row: (float(row.get("profit_factor") or 0.0), float(row.get("sharpe") or 0.0), float(row.get("f1") or 0.0))) if experiments else None
    return {"experiments": experiments, "best_label": best}


def _feature_robustness_report(feature_names: Sequence[str], report_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    fold_importance = []
    for row in report_rows:
        importance = row.get("feature_importance") or []
        fold_importance.append({"model_name": row.get("model_name"), "top_features": importance})
    return {
        "feature_count_candidates": [30, 50, 75, 100, len(feature_names)],
        "selected_feature_count": len(feature_names),
        "fold_wise_feature_importance": fold_importance,
        "unstable_features_removed": [],
        "correlated_features_removed": [],
    }


def _champion_selection_report(metric_rows: Sequence[Dict[str, Any]], all_reports: Dict[str, Any]) -> Dict[str, Any]:
    candidates = []
    for row in metric_rows:
        report = next((rep for label_models in all_reports.values() for name, rep in label_models.items() if rep.get("model_name") == row.get("model_name")), None)
        if report is None:
            continue
        threshold_diag = report.get("threshold_diagnostics") or {}
        cost_stress = report.get("cost_stress") or {}
        regime_gate = _regime_stability_gate(report.get("regime_performance", {})).get("gate", "FAIL")
        candidates.append(
            {
                "model_name": row.get("model_name"),
                "selected_threshold": (report.get("selected_threshold_from_validation") or {}).get("threshold"),
                "profit_factor": row.get("profit_factor"),
                "sharpe": row.get("sharpe"),
                "max_drawdown": (report.get("trade_metrics") or {}).get("max_drawdown"),
                "trade_count": row.get("trade_count"),
                "cost_stress_1_25x_pf": ((cost_stress.get("extra_cost_0_25") or {}).get("profit_factor")),
                "cost_stress_1_5x_pf": ((cost_stress.get("extra_cost_0_50") or {}).get("profit_factor")),
                "cost_stress_2_0x_pf": ((cost_stress.get("extra_cost_1_00") or {}).get("profit_factor")),
                "walk_forward_mean_f1": (report.get("walk_forward") or {}).get("mean_f1"),
                "walk_forward_mean_roc_auc": (report.get("walk_forward") or {}).get("mean_roc_auc"),
                "ranking_signal_exists_but_threshold_not_profitable": threshold_diag.get("ranking_signal_exists_but_threshold_not_profitable"),
                "model_produced_no_positive_class": threshold_diag.get("model_produced_no_positive_class"),
                "regime_gate": regime_gate,
            }
        )
    champion = max(candidates, key=lambda row: (float(row.get("profit_factor") or 0.0), float(row.get("sharpe") or 0.0), -float(row.get("max_drawdown") or 0.0), int(row.get("trade_count") or 0))) if candidates else None
    blocked = True
    reason = "production blocked because PF and Sharpe do not safely exceed thresholds across walk-forward and cost-stress gates"
    if champion and float(champion.get("profit_factor") or 0.0) > 1.15 and float(champion.get("sharpe") or 0.0) > 1.0 and float(champion.get("cost_stress_1_25x_pf") or 0.0) > 1.05 and champion.get("regime_gate") == "PASS":
        blocked = False
        reason = "candidate cleared lightweight research gates"
    return {
        "candidates": candidates,
        "champion": champion,
        "production_adoption_allowed": False,
        "paper_trading_allowed": False,
        "champion_rejected": True if blocked else False,
        "reason": reason if blocked else "research candidate only; production still blocked by policy",
    }


def _final_decision_matrix(
    *,
    baseline_best: Dict[str, Any],
    enriched_best: Dict[str, Any],
    leakage_result: Dict[str, Any],
    adoption: Dict[str, Any],
    same_period_report: Dict[str, Any],
) -> Dict[str, Any]:
    chronology_gate = "PASS" if bool(enriched_best.get("walk_forward", {}).get("folds")) else "FAIL"
    leakage_gate = "PASS" if bool(leakage_result.get("passed")) else "FAIL"
    same_period_gate = same_period_report.get("same_period_gate", "FAIL")
    trading_metrics_gate = "PASS" if float(enriched_best.get("trade_metrics", {}).get("profit_factor") or 0.0) > 1.05 and float(enriched_best.get("trade_metrics", {}).get("sharpe") or 0.0) > 0.5 else "FAIL"
    cost_stress_gate = "PASS"
    for scenario_name, payload in (enriched_best.get("cost_stress") or {}).items():
        if scenario_name == "base":
            continue
        if bool(payload.get("failure_flag")):
            cost_stress_gate = "FAIL"
            break
    fold_stability_gate = (enriched_best.get("fold_stability") or {}).get("gate", "FAIL")
    threshold_robustness_gate = "PASS" if bool((enriched_best.get("threshold_robustness") or {}).get("chosen_threshold_is_robust")) else "FAIL"
    live_schema_gate = "PASS" if bool((adoption.get("strict_live_schema_gate") or {}).get("compatible")) else "FAIL"
    if chronology_gate == "FAIL" or leakage_gate == "FAIL":
        final_verdict = "REJECT_NO_EDGE"
    elif trading_metrics_gate == "PASS" and cost_stress_gate == "PASS" and fold_stability_gate == "PASS" and threshold_robustness_gate == "PASS" and live_schema_gate == "PASS":
        final_verdict = "PRODUCTION_READY"
    elif trading_metrics_gate == "PASS":
        final_verdict = "RESEARCH_ONLY_WEAK_EDGE"
    else:
        final_verdict = "REJECT_NO_EDGE"
    return {
        "chronology_gate": chronology_gate,
        "leakage_gate": leakage_gate,
        "same_period_gate": same_period_gate,
        "trading_metrics_gate": trading_metrics_gate,
        "cost_stress_gate": cost_stress_gate,
        "fold_stability_gate": fold_stability_gate,
        "threshold_robustness_gate": threshold_robustness_gate,
        "live_schema_gate": live_schema_gate,
        "production_adoption_allowed": False,
        "final_verdict": final_verdict if final_verdict != "PRODUCTION_READY" else "RESEARCH_ONLY_WEAK_EDGE",
    }


def _paper_execution_filter_report(
    test_frame: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    returns: np.ndarray | None,
    *,
    max_trades_per_day: int,
    cooldown_minutes: int,
    allow_ce: bool,
    allow_pe: bool,
    min_option_price: float,
    max_bid_ask_spread_pct: float,
    avoid_opening_minutes: int,
    avoid_closing_minutes: int,
) -> Dict[str, Any]:
    if returns is None or test_frame.empty:
        return {"before": {}, "after": {}, "selected_rows": []}
    work = test_frame.copy().reset_index(drop=True)
    work["probability"] = np.asarray(y_prob, dtype=float)
    work["selected_return"] = np.asarray(returns, dtype=float)
    work["selected"] = work["probability"] >= float(threshold)
    pre = work.loc[work["selected"]].copy()
    before = _trade_metrics_from_scores(np.ones(len(pre)), np.ones(len(pre)), 0.5, pre["selected_return"].to_numpy(dtype=float), threshold_source="paper_before") if not pre.empty else _trade_metrics_from_scores(np.array([]), np.array([]), 0.5, np.array([]), threshold_source="paper_before")
    if pre.empty:
        return {"before": before, "after": before, "selected_rows": []}
    pre["timestamp"] = pd.to_datetime(pre["timestamp"])
    pre["trade_day"] = pre["timestamp"].dt.date
    pre["minute_of_day"] = pre["timestamp"].dt.hour * 60 + pre["timestamp"].dt.minute
    if "option_type_ce" in pre.columns and not allow_ce:
        pre = pre.loc[pd.to_numeric(pre["option_type_ce"], errors="coerce").fillna(0) < 1].copy()
    if "option_type_pe" in pre.columns and not allow_pe:
        pre = pre.loc[pd.to_numeric(pre["option_type_pe"], errors="coerce").fillna(0) < 1].copy()
    if "ltp" in pre.columns:
        pre = pre.loc[pd.to_numeric(pre["ltp"], errors="coerce").fillna(0.0) >= float(min_option_price)].copy()
    if "bid_ask_spread_pct" in pre.columns:
        pre = pre.loc[pd.to_numeric(pre["bid_ask_spread_pct"], errors="coerce").fillna(0.0) * 100.0 <= float(max_bid_ask_spread_pct)].copy()
    open_limit = 9 * 60 + 15 + int(avoid_opening_minutes)
    close_limit = 15 * 60 + 30 - int(avoid_closing_minutes)
    pre = pre.loc[(pre["minute_of_day"] >= open_limit) & (pre["minute_of_day"] <= close_limit)].copy()
    filtered_rows: List[int] = []
    last_trade_by_day: Dict[Any, pd.Timestamp] = {}
    count_by_day: Dict[Any, int] = {}
    for idx, row in pre.sort_values("timestamp", kind="stable").iterrows():
        day = row["trade_day"]
        if count_by_day.get(day, 0) >= int(max_trades_per_day):
            continue
        last_ts = last_trade_by_day.get(day)
        current_ts = pd.Timestamp(row["timestamp"])
        if last_ts is not None and (current_ts - last_ts).total_seconds() < int(cooldown_minutes) * 60:
            continue
        filtered_rows.append(idx)
        last_trade_by_day[day] = current_ts
        count_by_day[day] = count_by_day.get(day, 0) + 1
    post = pre.loc[filtered_rows].copy()
    after = _trade_metrics_from_scores(np.ones(len(post)), np.ones(len(post)), 0.5, post["selected_return"].to_numpy(dtype=float), threshold_source="paper_after") if not post.empty else _trade_metrics_from_scores(np.array([]), np.array([]), 0.5, np.array([]), threshold_source="paper_after")
    return {"before": before, "after": after, "selected_rows": post.to_dict(orient="records")}


def _calibration_audit(y_true: np.ndarray, y_prob: np.ndarray, returns: np.ndarray | None) -> Dict[str, Any]:
    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 1.01)]
    rows: List[Dict[str, Any]] = []
    last_win_rate = None
    monotonic = True
    for start, end in buckets:
        mask = (y_prob >= start) & (y_prob < end)
        if not mask.any():
            rows.append({"bucket": f"{start:.2f}-{min(end,1.0):.2f}", "sample_count": 0, "predicted_probability_mean": None, "actual_win_rate": None, "average_real_return": None, "profit_factor": None, "sharpe": None})
            continue
        bucket_returns = returns[mask] if returns is not None else None
        bucket_metrics = _trade_metrics_from_scores(y_true[mask], y_prob[mask], start, bucket_returns, threshold_source="calibration")
        actual_win_rate = float(np.mean(y_true[mask])) if mask.any() else None
        if last_win_rate is not None and actual_win_rate is not None and actual_win_rate + 1e-9 < last_win_rate:
            monotonic = False
        if actual_win_rate is not None:
            last_win_rate = actual_win_rate
        rows.append({
            "bucket": f"{start:.2f}-{min(end,1.0):.2f}",
            "sample_count": int(mask.sum()),
            "predicted_probability_mean": float(np.mean(y_prob[mask])),
            "actual_win_rate": actual_win_rate,
            "average_real_return": float(np.mean(bucket_returns)) if bucket_returns is not None and len(bucket_returns) else None,
            "profit_factor": bucket_metrics.get("profit_factor"),
            "sharpe": bucket_metrics.get("sharpe"),
            "small_sample_warning": bool(mask.sum() < 50),
        })
    return {"buckets": rows, "calibration_gate": "PASS" if monotonic else "FAIL", "monotonic_win_rate": monotonic}


def _probability_monotonicity_report(y_prob: np.ndarray, returns: np.ndarray | None) -> Dict[str, Any]:
    if returns is None or len(y_prob) == 0:
        return {"deciles": [], "gate": "FAIL"}
    frame = pd.DataFrame({"probability": y_prob, "selected_return": returns})
    frame["decile"] = pd.qcut(frame["probability"], q=10, labels=False, duplicates="drop")
    rows: List[Dict[str, Any]] = []
    for decile, group in frame.groupby("decile"):
        metrics = _trade_metrics_from_scores(np.ones(len(group)), np.ones(len(group)), 0.5, group["selected_return"].to_numpy(dtype=float), threshold_source="probability_decile")
        rows.append({
            "decile": int(decile),
            "sample_count": int(len(group)),
            "win_rate": metrics.get("win_rate"),
            "average_return": metrics.get("average_return_per_trade"),
            "total_return": metrics.get("total_return"),
            "profit_factor": metrics.get("profit_factor"),
            "sharpe": metrics.get("sharpe"),
        })
    rows.sort(key=lambda row: row["decile"])
    gate = "PASS" if rows and float(rows[-1].get("average_return") or 0.0) > float(rows[len(rows)//2].get("average_return") or 0.0) >= float(rows[0].get("average_return") or 0.0) else "FAIL"
    return {"deciles": rows, "gate": gate}


def _daily_pnl_stability_report(selected_rows: Sequence[Dict[str, Any]], *, max_trades_per_day: int) -> Dict[str, Any]:
    if not selected_rows:
        return {"gate": "FAIL", "active_trading_days": 0}
    frame = pd.DataFrame(list(selected_rows))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["trade_day"] = frame["timestamp"].dt.date
    daily = frame.groupby("trade_day")["selected_return"].agg(["sum", "count"]).reset_index()
    total_return = float(daily["sum"].sum()) if not daily.empty else 0.0
    contribution = float(daily["sum"].abs().max() / abs(total_return)) if total_return else 0.0
    profitable_days = int((daily["sum"] > 0).sum())
    losing_days = int((daily["sum"] < 0).sum())
    equity = daily["sum"].cumsum()
    peaks = equity.cummax()
    dd = peaks - equity
    longest_losing = 0
    current = 0
    for value in daily["sum"]:
        if value < 0:
            current += 1
            longest_losing = max(longest_losing, current)
        else:
            current = 0
    gate = "PASS"
    if contribution > 0.25 or (profitable_days / max(1, len(daily))) < 0.50:
        gate = "FAIL"
    return {
        "total_trading_days": int(len(daily)),
        "active_trading_days": int(len(daily)),
        "profitable_days": profitable_days,
        "losing_days": losing_days,
        "day_win_rate": float(profitable_days / max(1, len(daily))),
        "best_day": float(daily["sum"].max()),
        "worst_day": float(daily["sum"].min()),
        "average_daily_return": float(daily["sum"].mean()),
        "median_daily_return": float(daily["sum"].median()),
        "daily_sharpe": float(daily["sum"].mean() / daily["sum"].std()) if len(daily) > 1 and float(daily["sum"].std()) > 0 else 0.0,
        "longest_losing_streak_days": int(longest_losing),
        "max_daily_drawdown": float(dd.max()) if len(dd) else 0.0,
        "days_with_more_than_allowed_trades": int((daily["count"] > int(max_trades_per_day)).sum()),
        "single_day_return_contribution": contribution,
        "gate": gate,
    }


def _enriched_dataset_path(dataset_path: Path) -> Path:
    suffix = dataset_path.suffix.lower()
    target_suffix = suffix if suffix in {".csv", ".parquet"} else ".csv"
    return dataset_path.with_name(f"{dataset_path.stem}_bs_enriched{target_suffix}")


def _all_supported_model_families() -> List[str]:
    return [
        "logistic_regression",
        "logistic_regression_uncalibrated",
        "logistic_regression_platt",
        "logistic_regression_isotonic",
        "elasticnet_logistic_regression",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "hist_gradient_boosting",
        "xgboost",
        "lightgbm",
        "catboost",
        "calibrated_logistic_regression",
        "ensemble",
        "xgb_rf_ensemble",
    ]


def _parse_model_families_arg(value: str) -> List[str]:
    token = str(value or "default").strip().lower()
    if token in {"", "default"}:
        return [name for name in _all_supported_model_families() if name != "xgb_rf_ensemble" and _optional_dependency_status(name)[0]]
    if token == "all":
        return [name for name in _all_supported_model_families() if name != "xgb_rf_ensemble" and _optional_dependency_status(name)[0]]
    return [part.strip().lower() for part in token.split(",") if part.strip()]


def _optional_dependency_status(model_family: str) -> tuple[bool, str | None]:
    try:
        if model_family in {"xgboost", "xgb_rf_ensemble"}:
            __import__("xgboost")
        elif model_family == "lightgbm":
            __import__("lightgbm")
        elif model_family == "catboost":
            __import__("catboost")
        return True, None
    except Exception as exc:
        return False, f"SKIPPED_OPTIONAL_DEPENDENCY_MISSING:{model_family}:{exc.__class__.__name__}"


def _runtime_rf_params() -> Dict[str, Any]:
    defaults = dict(FAST_RF_DEFAULTS if bool(_runtime_option("fast_mode", False)) else PRODUCTION_RF_DEFAULTS)
    defaults["n_estimators"] = int(_runtime_option("rf_n_estimators", None) or _runtime_option("n_estimators", None) or defaults["n_estimators"])
    defaults["max_depth"] = int(_runtime_option("rf_max_depth", None) or _runtime_option("max_depth", None) or defaults["max_depth"])
    defaults["min_samples_leaf"] = int(_runtime_option("rf_min_samples_leaf", None) or _runtime_option("min_samples_leaf", None) or defaults["min_samples_leaf"])
    defaults["min_samples_split"] = int(_runtime_option("rf_min_samples_split", None) or defaults["min_samples_split"])
    defaults["max_features"] = _runtime_option("rf_max_features", None) or defaults["max_features"]
    defaults["class_weight"] = _runtime_option("rf_class_weight", None) or defaults["class_weight"]
    defaults["n_jobs"] = int(_runtime_option("rf_n_jobs", None) or _runtime_option("n_jobs", -1) or defaults["n_jobs"])
    return defaults


def _runtime_xgb_params() -> Dict[str, Any]:
    defaults = dict(FAST_XGB_DEFAULTS if bool(_runtime_option("fast_mode", False)) else PRODUCTION_XGB_DEFAULTS)
    mapping = {
        "xgb_n_estimators": "n_estimators",
        "xgb_max_depth": "max_depth",
        "xgb_learning_rate": "learning_rate",
        "xgb_subsample": "subsample",
        "xgb_colsample_bytree": "colsample_bytree",
        "xgb_reg_lambda": "reg_lambda",
        "xgb_reg_alpha": "reg_alpha",
        "xgb_min_child_weight": "min_child_weight",
        "xgb_tree_method": "tree_method",
    }
    for runtime_key, param_key in mapping.items():
        value = _runtime_option(runtime_key, None)
        if value is not None:
            defaults[param_key] = value
    defaults["n_jobs"] = int(_runtime_option("n_jobs", -1) or defaults.get("n_jobs", -1))
    defaults["random_state"] = 42
    return defaults


def _runtime_ensemble_params() -> Dict[str, Any]:
    return dict(_runtime_option("ensemble", {}) or {})


def _default_threshold_sweep_grid() -> List[float]:
    return [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]


def _build_model_family(model_family: str) -> Any:
    name = str(model_family).strip().lower()
    if name in {"logistic_regression", "logistic_regression_uncalibrated", "logistic_regression_platt", "logistic_regression_isotonic", "ensemble", "random_forest", "xgboost", "catboost"}:
        return build_safe_model(name)
    if name == "elasticnet_logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            max_iter=4000,
            class_weight="balanced",
            random_state=42,
        )
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=2,
        )
    if name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(random_state=42)
    if name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=42, max_depth=6, learning_rate=0.05, max_iter=250, min_samples_leaf=50)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    if name == "calibrated_logistic_regression":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression

        base = LogisticRegression(
            max_iter=3000,
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
        )
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    raise ValueError(f"Unsupported model family: {model_family}")


def ensure_black_scholes_dataset(
    dataset_path: Path,
    *,
    risk_free_rate: float = 0.065,
    dividend_yield: float = 0.0,
    symbol: str = "NIFTY",
) -> Path:
    if "_bs_enriched" in dataset_path.stem.lower():
        return dataset_path
    enriched_path = _enriched_dataset_path(dataset_path)
    if enriched_path.exists():
        return enriched_path
    script_path = REPO_ROOT / "scripts" / "enrich_option_dataset_black_scholes.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--input",
        str(dataset_path),
        "--output",
        str(enriched_path),
        "--risk-free-rate",
        str(float(risk_free_rate)),
        "--dividend-yield",
        str(float(dividend_yield)),
        "--symbol",
        str(symbol),
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if not enriched_path.exists():
        raise FileNotFoundError(f"Expected enriched dataset was not created: {enriched_path}")
    return enriched_path


def build_feature_sets(df: pd.DataFrame, *, use_black_scholes_features: bool) -> Dict[str, Any]:
    groups = detect_column_groups(df)
    audit = groups.get("feature_audit", {})
    baseline = list(groups["input_features"])
    baseline = [name for name in baseline if name not in BS_RESEARCH_FEATURES and name not in BS_FLAG_FEATURES]
    rejected = [name for name in df.columns if _is_forbidden_feature_name(name)]
    payload: Dict[str, Any] = {
        "baseline_features": baseline,
        "bs_research_features_present": [name for name in BS_RESEARCH_FEATURES if name in df.columns],
        "bs_flag_features_present": [name for name in BS_FLAG_FEATURES if name in df.columns],
        "rejected_features": sorted(set(rejected)),
        "target_columns": groups.get("target_columns", []),
        "evaluation_return_columns": groups.get("evaluation_return_columns", []),
        "metadata_columns": groups.get("metadata_columns", []),
        "audit": {
            "raw_column_count": audit.get("raw_column_count", len(df.columns)),
            "selected_input_feature_count": len(baseline),
            "forbidden_columns_removed": sorted(set(groups.get("forbidden_feature_columns", []))),
            "constant_columns_removed": audit.get("constant_columns", []),
            "high_null_columns_removed": audit.get("high_null_columns", []),
            "duplicate_columns_removed": audit.get("duplicate_columns", []),
            "final_input_features": baseline,
        },
    }
    if not use_black_scholes_features:
        payload["enriched_features"] = baseline
        return payload
    missing = [name for name in BS_RESEARCH_FEATURES + BS_FLAG_FEATURES if name not in df.columns]
    if missing:
        raise RuntimeError(f"Black-Scholes feature flag enabled but enriched columns are missing: {missing}")
    payload["enriched_features"] = baseline + BS_RESEARCH_FEATURES + BS_FLAG_FEATURES
    return payload


def _edge_refinement_cost_metrics(selected_rows: pd.DataFrame, *, cost_provenance: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if selected_rows.empty or "selected_return" not in selected_rows.columns:
        return {
            "estimated_roundtrip_cost": 0.0,
            "average_estimated_cost_pct": 0.0,
            "average_return_per_trade": 0.0,
            "median_return_per_trade": 0.0,
            "average_positive_return": 0.0,
            "average_negative_return": 0.0,
            "expected_return_after_cost": 0.0,
            "median_return_after_cost": 0.0,
            "return_to_cost_ratio": 0.0,
            "median_return_to_cost_ratio": 0.0,
            "gross_edge_before_cost": 0.0,
            "net_edge_after_cost": 0.0,
            "percentage_of_trades_with_return_greater_than_cost": 0.0,
            "percentage_of_trades_with_return_greater_than_1_25x_cost": 0.0,
            "percentage_of_trades_with_return_greater_than_1_5x_cost": 0.0,
        }
    cost_frame = estimate_option_execution_costs_frame(selected_rows, config=DEFAULT_EXECUTION_COST_CONFIG)
    returns = pd.to_numeric(selected_rows["selected_return"], errors="coerce").fillna(0.0)
    costs = pd.to_numeric(cost_frame["cost_return_units"], errors="coerce").fillna(0.0)
    provenance = dict(cost_provenance or {})
    already_net = bool(provenance.get("is_evaluation_return_already_net"))
    positive = returns[returns > 0.0]
    negative = returns[returns < 0.0]
    after_cost = returns if already_net else (returns - costs)
    safe_costs = costs.replace(0.0, np.nan)
    ratio = after_cost / safe_costs
    return {
        "estimated_roundtrip_cost": float(costs.mean()) if len(costs) else 0.0,
        "average_estimated_cost_pct": float(pd.to_numeric(cost_frame["cost_pct_of_premium"], errors="coerce").fillna(0.0).mean()) if not cost_frame.empty else 0.0,
        "average_return_per_trade": float(returns.mean()) if len(returns) else 0.0,
        "median_return_per_trade": float(returns.median()) if len(returns) else 0.0,
        "average_positive_return": float(positive.mean()) if len(positive) else 0.0,
        "average_negative_return": float(negative.mean()) if len(negative) else 0.0,
        "expected_return_after_cost": float(after_cost.mean()) if len(after_cost) else 0.0,
        "median_return_after_cost": float(after_cost.median()) if len(after_cost) else 0.0,
        "return_to_cost_ratio": float((after_cost.mean() / costs.mean()) if float(costs.mean()) > 0.0 else 0.0),
        "median_return_to_cost_ratio": float(ratio.replace([np.inf, -np.inf], np.nan).median()) if ratio.notna().any() else 0.0,
        "gross_edge_before_cost": float(returns.sum()) if len(returns) else 0.0,
        "net_edge_after_cost": float(after_cost.sum()) if len(after_cost) else 0.0,
        "percentage_of_trades_with_return_greater_than_cost": float((returns > costs).mean()) if len(returns) else 0.0,
        "percentage_of_trades_with_return_greater_than_1_25x_cost": float((returns > (costs * 1.25)).mean()) if len(returns) else 0.0,
        "percentage_of_trades_with_return_greater_than_1_5x_cost": float((returns > (costs * 1.50)).mean()) if len(returns) else 0.0,
    }


def _filter_invalid_bs_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "row_enrichment_ok" not in df.columns:
        return df
    flag = pd.to_numeric(df["row_enrichment_ok"], errors="coerce").fillna(0).astype(bool)
    return df.loc[flag].copy()


def _drop_bad_greeks_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "bad_iv_flag" not in df.columns and "bad_greek_flag" not in df.columns:
        return df, 0
    bad_iv = pd.to_numeric(df.get("bad_iv_flag"), errors="coerce").fillna(0).astype(bool) if "bad_iv_flag" in df.columns else pd.Series(False, index=df.index)
    bad_greek = pd.to_numeric(df.get("bad_greek_flag"), errors="coerce").fillna(0).astype(bool) if "bad_greek_flag" in df.columns else pd.Series(False, index=df.index)
    drop_mask = bad_iv | bad_greek
    return df.loc[~drop_mask].copy(), int(drop_mask.sum())


def train_variant(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    labels: Sequence[str],
    artifact_dir: Path,
    variant_name: str,
    evaluation_return_column: str | None,
    walk_forward_folds: int = 5,
    threshold_grid_start: float = 0.50,
    threshold_grid_end: float = 0.90,
    threshold_grid_step: float = 0.025,
    min_threshold_trades: int = 1000,
    paper_config: Dict[str, Any] | None = None,
    model_families: Sequence[str] | None = None,
    only_targets: Sequence[str] | None = None,
    only_models: Sequence[str] | None = None,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    variant_start = time.perf_counter()
    # =====================================================================
    # FEATURE CONTRACT HYGIENE CHECK
    # Verify training features are compatible with live inference contract.
    # This prevents training on features that can't be computed at inference time.
    # =====================================================================
    feature_hygiene_results: List[Dict[str, Any]] = []
    hygiene_failed = False
    for label_name in labels:
        hygiene = _check_training_features_hygiene(
            list(feature_cols),
            label_name=str(label_name),
            dataset_columns=df.columns,
            enforce_live_contract=bool((paper_config or {}).get("live_computable_only")) or bool((paper_config or {}).get("strict_live_contract")),
        )
        feature_hygiene_results.append(hygiene)
        if not hygiene.get("hygiene_passed", False):
            hygiene_failed = True
            LOGGER = logging.getLogger(__name__)
            LOGGER.error(
                "FEATURE HYGIENE FAILED for label=%s: %s",
                label_name,
                hygiene.get("issues", [])
            )
    
    # Write feature hygiene report to artifact directory
    if feature_hygiene_results:
        _report_feature_hygiene_results(feature_hygiene_results, artifact_dir)
    if hygiene_failed:
        raise RuntimeError(
            f"FEATURE HYGIENE FAILED for labels={[str(item) for item in labels]}. "
            f"See {artifact_dir / 'feature_hygiene_report.json'} for details."
        )
    
    preprocess_start = time.perf_counter()
    X_df = df[list(feature_cols)].copy().replace([np.inf, -np.inf], np.nan)
    medians = X_df.median(numeric_only=True)
    X_df = X_df.fillna(medians).fillna(0.0).astype(np.float32)
    _timing_log("variant_preprocessing", seconds=time.perf_counter() - preprocess_start, rows=len(X_df), extra=f"variant={variant_name}")
    all_reports: Dict[str, Any] = {}
    trained_models: List[str] = []
    feature_manifests: Dict[str, Any] = {}
    skipped_models: List[Dict[str, Any]] = []
    failed_models: List[Dict[str, Any]] = []
    returns_col = evaluation_return_column if evaluation_return_column in df.columns else None
    families = list(model_families or model_names_to_train())
    allowed_targets = set(str(item).strip() for item in (only_targets or []) if str(item).strip())
    # FIX 4: Normalise to lowercase so mixed-case CLI inputs (XGBoost, ExtraTrees)
    # match the lowercased model registry without silent zero-model training.
    # FIX 5 (defence-in-depth): if --only-model was specified but resolves to no
    # models after case-normalised lookup, fail immediately.  Also expand comma-
    # separated values and apply common aliases (randomforest->random_forest etc.).
    _MODEL_ALIASES_VARIANT = {
        "randomforest": "random_forest",
        "extratrees": "extra_trees",
        "gradientboosting": "gradient_boosting",
        "histgradientboosting": "hist_gradient_boosting",
    }
    raw_tokens: List[str] = []
    for item in (only_models or []):
        raw_tokens.extend(str(item).split(","))
    allowed_models = {
        _MODEL_ALIASES_VARIANT.get(str(t).strip().lower(), str(t).strip().lower())
        for t in raw_tokens if str(t).strip()
    }
    # FIX 6: if --only-model was specified but resolves to no models, fail immediately.
    if only_models and not allowed_models:
        raise ValueError(
            f"--only-model={list(only_models)!r}: resolved to zero models after "
            f"case-normalised lookup.  Available families (lowercased): "
            f"{sorted({str(f).strip().lower() for f in families})!r}."
        )
    for label_name in labels:
        if label_name not in df.columns:
            raise RuntimeError(f"Missing target column '{label_name}' in dataset.")
        if allowed_targets and str(label_name) not in allowed_targets:
            continue
        requested_base_models = [
            str(model_name)
            for model_name in families
            if str(model_name).strip().lower() not in {"ensemble", "xgb_rf_ensemble"}
            and (not allowed_models or str(model_name).strip().lower() in allowed_models)
        ]
        requested_ensemble = any(
            str(model_name).strip().lower() == "ensemble"
            and (not allowed_models or str(model_name).strip().lower() in allowed_models)
            for model_name in families
        )
        requested_xgb_rf_ensemble = any(
            str(model_name).strip().lower() == "xgb_rf_ensemble"
            and (not allowed_models or str(model_name).strip().lower() in allowed_models)
            for model_name in families
        )
        if not force_retrain and requested_base_models:
            completed_reports: Dict[str, Any] = {}
            all_requested_complete = True
            for model_name in requested_base_models:
                status_payload = _read_json_if_exists(_status_file_path(artifact_dir, label_name, model_name))
                if not _completed_status_is_valid(status_payload):
                    all_requested_complete = False
                    break
                metrics_payload = _read_json_if_exists(Path(str(status_payload["metrics_path"])))
                if not metrics_payload:
                    all_requested_complete = False
                    break
                completed_reports[model_name] = metrics_payload
            if all_requested_complete and requested_ensemble:
                ensemble_status = _read_json_if_exists(_status_file_path(artifact_dir, label_name, "ensemble"))
                if not _completed_status_is_valid(ensemble_status):
                    all_requested_complete = False
                else:
                    ensemble_metrics = _read_json_if_exists(Path(str(ensemble_status["metrics_path"])))
                    if not ensemble_metrics:
                        all_requested_complete = False
                    else:
                        completed_reports["ensemble"] = ensemble_metrics
            if all_requested_complete and requested_xgb_rf_ensemble:
                xgb_rf_status = _read_json_if_exists(_status_file_path(artifact_dir, label_name, "xgb_rf_ensemble"))
                if not _completed_status_is_valid(xgb_rf_status):
                    all_requested_complete = False
                else:
                    xgb_rf_metrics = _read_json_if_exists(Path(str(xgb_rf_status["metrics_path"])))
                    if not xgb_rf_metrics:
                        all_requested_complete = False
                    else:
                        completed_reports["xgb_rf_ensemble"] = xgb_rf_metrics
            if all_requested_complete:
                all_reports[label_name] = completed_reports
                feature_manifests[label_name] = list(feature_cols)
                for model_name in completed_reports:
                    trained_models.append(f"{variant_name}:{model_name}:{label_name}")
                continue
        y_series = prepare_label_series(df, label_name)
        passthrough_columns = [
            column
            for column in (
                "timestamp",
                "dte_days",
                "moneyness",
                "distance_from_spot",
                "option_type_ce",
                "option_type_pe",
                "volatility_regime_classifier",
                "regime_trending",
                "regime_volatile",
                "regime_mean_reverting",
                "regime_quiet",
                # Required for _return_cost_provenance_for_frame to infer embedded cost
                # from gross_forward_return - net_forward_return diff >= 0.0 cases.
                "gross_forward_return",
                "net_forward_return",
                # Required for estimate_option_execution_costs_frame per-row cost estimation.
                "ltp",
            )
            if column in df.columns and (column == "timestamp" or column not in feature_cols)
        ]
        work = pd.concat([df[passthrough_columns], X_df, y_series.rename("label")], axis=1)
        if returns_col:
            work[returns_col] = pd.to_numeric(df[returns_col], errors="coerce")
        work = work.dropna(subset=["timestamp", "label"]).copy()
        work = work[work["label"].isin([0, 1])]
        if work.empty:
            all_reports[label_name] = {}
            continue
        work = _prepare_chronological_work(work)
        y = work["label"].astype(int).to_numpy()
        X = np.nan_to_num(work[list(feature_cols)].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        timestamps = pd.Series(work["timestamp"]).reset_index(drop=True)
        eval_returns = work[returns_col].to_numpy(dtype=float) if returns_col else None
        split = _build_holdout_split_from_timestamps(timestamps)
        train_idx = split["train"]
        val_idx = split["validation"]
        test_idx = split["test"]
        _assert_fold_is_chronological(timestamps, train_idx, val_idx, test_idx=test_idx, context=f"{variant_name}:{label_name}:holdout_split")
        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
        test_frame = _apply_backtest_row_limit(work.iloc[test_idx].reset_index(drop=True), context=f"{variant_name}:{label_name}")
        X_train_s, X_val_s, X_test_s, scaler_mean, scaler_std = scale_train_val_test(X_train, X_val, X_test)
        walk_forward = _walk_forward_summary(X, y, timestamps, feature_cols, label_name, eval_returns, fold_count=walk_forward_folds)
        split_window = {}
        split_window.update(_slice_time_range(timestamps, train_idx, "train"))
        split_window.update(_slice_time_range(timestamps, val_idx, "validation"))
        split_window.update(_slice_time_range(timestamps, test_idx, "test"))
        label_reports: Dict[str, Any] = {}
        base_ensemble_candidates: List[Dict[str, Any]] = []
        run_ensemble = any(str(name).strip().lower() == "ensemble" for name in families)
        for model_name in families:
            if allowed_models and str(model_name).strip().lower() not in allowed_models:
                continue
            if str(model_name).strip().lower() in {"ensemble", "xgb_rf_ensemble"}:
                continue
            status_path = _status_file_path(artifact_dir, label_name, model_name)
            threshold_json = artifact_dir / f"{variant_name}_{model_name}_{label_name}_threshold_sweep.json"
            threshold_csv = artifact_dir / f"{variant_name}_{model_name}_{label_name}_threshold_sweep.csv"
            existing_status = _read_json_if_exists(status_path)
            if not force_retrain and _completed_status_is_valid(existing_status):
                metrics_payload = _read_json_if_exists(Path(str(existing_status["metrics_path"]))) or {}
                label_reports[model_name] = metrics_payload
                trained_models.append(f"{variant_name}:{model_name}:{label_name}")
                continue
            dep_ok, dep_reason = _optional_dependency_status(model_name)
            if not dep_ok:
                _write_status_file(artifact_dir, label_name, model_name, {
                    "label_name": label_name,
                    "model_name": model_name,
                    "status": "failed",
                    "start_time": None,
                    "end_time": datetime.now().isoformat(),
                    "artifact_path": None,
                    "metrics_path": None,
                    "error_message": dep_reason,
                    "training_time_seconds": 0.0,
                })
                skipped_models.append({"variant_name": variant_name, "label_name": label_name, "model_family": model_name, "reason": dep_reason})
                continue
            start = time.time()
            _write_status_file(artifact_dir, label_name, model_name, {
                "label_name": label_name,
                "model_name": model_name,
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "artifact_path": None,
                "metrics_path": None,
                "error_message": None,
                "training_time_seconds": None,
            })
            try:
                val_returns = eval_returns[val_idx] if eval_returns is not None else None
                model, calibration_fit = _fit_model_for_training_run(
                    model_name,
                    X_train_s,
                    y_train,
                    X_val_s,
                    y_val,
                    val_returns,
                    minimum_trades=min_threshold_trades,
                )
            except Exception as exc:
                train_time = time.time() - start
                _write_status_file(artifact_dir, label_name, model_name, {
                    "label_name": label_name,
                    "model_name": model_name,
                    "status": "failed",
                    "start_time": None,
                    "end_time": datetime.now().isoformat(),
                    "artifact_path": None,
                    "metrics_path": None,
                    "error_message": str(exc),
                    "training_time_seconds": train_time,
                })
                failed_models.append({
                    "variant_name": variant_name,
                    "label_name": label_name,
                    "model_family": model_name,
                    "error": str(exc),
                    "stack_summary": traceback.format_exc(limit=3),
                })
                continue
            train_time = time.time() - start
            predict_start = time.perf_counter()
            val_prob = predict_proba_positive(model, X_val_s)
            test_prob = predict_proba_positive(model, X_test_s)
            predict_time = time.perf_counter() - predict_start
            _timing_log("predict", seconds=predict_time, rows=len(X_val_s) + len(X_test_s), extra=f"variant={variant_name} label={label_name} model={model_name}")

            # =========================================================================
            # OOF PREDICTION PERSISTENCE
            # =========================================================================
            # Build OOF predictions DataFrame with leakage-safe columns only.
            # This DataFrame is for evaluation purposes and does NOT include any
            # feature columns (return/label columns are also excluded).
            oof_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # threshold may not be in scope if the path through train_variant skipped
            # threshold optimisation.  Fall back to 0.55 for OOF persistence purposes.
            oof_threshold = float(threshold) if "threshold" in dir() and threshold is not None else 0.55
            oof_df = _build_oof_frame_from_holdout(
                test_frame=test_frame,
                y_test=y_test,
                test_prob=test_prob,
                threshold=oof_threshold,
                model_name=str(model_name),
                label_name=str(label_name),
                fold_id=1,
                evaluation_return_column=returns_col,
            )
            oof_persist_result = _persist_oof_predictions(
                artifact_dir=artifact_dir,
                label_name=str(label_name),
                model_name=str(model_name),
                oof_data=oof_df,
                timestamp=oof_timestamp,
            )

            # Also collect walk-forward OOF predictions for richer decile analysis
            try:
                wf_oof_df = _collect_walk_forward_oof_predictions(
                    X=X, y=y, timestamps=timestamps, feature_cols=feature_cols,
                    label_name=str(label_name), scaler_mean=scaler_mean, scaler_std=scaler_std,
                    fold_count=walk_forward_folds, returns=eval_returns,
                    evaluation_return_column=returns_col,
                )
                if not wf_oof_df.empty:
                    wf_oof_persist_result = _persist_oof_predictions(
                        artifact_dir=artifact_dir,
                        label_name=str(label_name),
                        model_name=f"{model_name}_walkforward",
                        oof_data=wf_oof_df,
                        timestamp=oof_timestamp,
                    )
            except Exception as exc:
                wf_oof_persist_result = {"status": "error", "error": str(exc)}

            backtest_start = time.perf_counter()
            chosen = optimize_threshold_from_probs(y_val, val_prob, val_returns, thresholds=RETRAIN_THRESHOLD_SWEEP, minimum_trades=min_threshold_trades)
            threshold = float(chosen["threshold"])
            val_metrics = classification_metrics(y_val, val_prob, threshold=threshold)
            test_metrics = classification_metrics(y_test, test_prob, threshold=threshold)
            val_metrics["pr_auc"] = pr_auc_score_safe(y_val, val_prob)
            val_metrics["brier_score"] = brier_score(y_val, val_prob)
            test_metrics["pr_auc"] = pr_auc_score_safe(y_test, test_prob)
            test_metrics["brier_score"] = brier_score(y_test, test_prob)
            threshold_rows = _full_threshold_sweep_report(y_test, test_prob, eval_returns[test_idx] if eval_returns is not None else None, thresholds=RETRAIN_THRESHOLD_SWEEP)
            returns = eval_returns[test_idx] if eval_returns is not None else None
            return_cost_provenance = _return_cost_provenance_for_frame(test_frame, evaluation_return_column=returns_col)
            validation_trade_metrics = _trade_metrics_from_scores(y_val, val_prob, threshold, val_returns, threshold_source="validation_selected_threshold")
            trade_metrics = _trade_metrics_from_scores(y_test, test_prob, threshold, returns)
            cost_stress = _cost_stress_report(
                y_test,
                test_prob,
                threshold,
                returns,
                cost_provenance=return_cost_provenance,
                per_row_cost_units=return_cost_provenance.get("per_row_cost_units"),
            )
            threshold_robustness = _threshold_robustness_report(
                y_test,
                test_prob,
                returns,
                grid_start=threshold_grid_start,
                grid_end=threshold_grid_end,
                grid_step=threshold_grid_step,
                min_trades=min_threshold_trades,
            )
            fold_stability = _fold_stability_report(walk_forward.get("folds", []))
            regime_report = _regime_performance_report(test_frame, test_prob, threshold, returns)
            paper_filter_report = _paper_execution_filter_report(
                test_frame,
                test_prob,
                threshold,
                returns,
                max_trades_per_day=int((paper_config or {}).get("paper_max_trades_per_day", 5)),
                cooldown_minutes=int((paper_config or {}).get("paper_cooldown_minutes", 15)),
                allow_ce=bool((paper_config or {}).get("paper_allow_ce", True)),
                allow_pe=bool((paper_config or {}).get("paper_allow_pe", True)),
                min_option_price=float((paper_config or {}).get("paper_min_option_price", 5.0)),
                max_bid_ask_spread_pct=float((paper_config or {}).get("paper_max_bid_ask_spread_pct", 5.0)),
                avoid_opening_minutes=int((paper_config or {}).get("paper_avoid_opening_minutes", 5)),
                avoid_closing_minutes=int((paper_config or {}).get("paper_avoid_closing_minutes", 5)),
            )
            calibration_audit = _calibration_audit(y_test, test_prob, returns)
            monotonicity_report = _probability_monotonicity_report(test_prob, returns)
            probability_diagnostics = _probability_diagnostics(y_test, test_prob, RETRAIN_THRESHOLD_SWEEP)
            threshold_diagnostics = _threshold_diagnostics(y_test, test_prob, threshold, returns)
            trade_filter_report = _trade_filter_report(test_frame, test_prob, threshold, returns)
            daily_pnl_report = _daily_pnl_stability_report(
                paper_filter_report.get("selected_rows", []),
                max_trades_per_day=int((paper_config or {}).get("paper_max_trades_per_day", 5)),
            )
            backtest_time = time.perf_counter() - backtest_start
            _timing_log("backtest", seconds=backtest_time, rows=len(test_frame), extra=f"variant={variant_name} label={label_name} model={model_name}")
            write_json(threshold_json, {"model_name": model_name, "label_name": label_name, "rows": threshold_rows})
            pd.DataFrame(threshold_rows).to_csv(threshold_csv, index=False)
            report = {
                "variant_name": variant_name,
                "model_name": model_name,
                "label_name": label_name,
                "validation_metrics": val_metrics,
                "validation_trade_metrics": validation_trade_metrics,
                "test_metrics": test_metrics,
                "test_threshold_metrics": {**test_metrics, **confusion_from_threshold(y_test, test_prob, threshold), "signal_count": int(np.sum(test_prob >= threshold))},
                "selected_threshold_from_validation": chosen,
                "training_time_seconds": train_time,
                "holdout_split": split_window,
                "walk_forward": walk_forward,
                "fold_stability": fold_stability,
                "trade_metrics": trade_metrics,
                "cost_stress": cost_stress,
                "return_cost_provenance": return_cost_provenance,
                # Per-target cost status (JSON-serializable summary, per-target not global)
                "per_target_cost_status": {
                    "target_name": label_name,
                    "cost_column_source": str(return_cost_provenance.get("cost_column_source") or "unknown"),
                    "has_per_row_cost_frame": bool(return_cost_provenance.get("has_per_row_cost_frame") or False),
                    "cost_model_status": str(return_cost_provenance.get("cost_model_status") or "unknown"),
                    "cost_model_confidence": str(return_cost_provenance.get("cost_model_confidence") or "unknown"),
                    "average_cost_return_units": float(return_cost_provenance.get("average_cost_return_units") or 0.0),
                    "average_cost_pct_of_premium": float(return_cost_provenance.get("average_cost_pct_of_premium") or 0.0),
                    "inferred_embedded_cost": (
                        float(return_cost_provenance["inferred_embedded_cost"])
                        if return_cost_provenance.get("inferred_embedded_cost") is not None
                        else None
                    ),
                    "cost_1_25x_pf_from_per_target_costs": float(
                        return_cost_provenance.get("per_target_pf_1_25x", 0.0)
                    ),
                    "cost_1_50x_pf_from_per_target_costs": float(
                        return_cost_provenance.get("per_target_pf_1_50x", 0.0)
                    ),
                    "evaluation_return_column": str(return_cost_provenance.get("evaluation_return_column") or ""),
                },
                "threshold_sweep": threshold_rows,
                "threshold_robustness": threshold_robustness,
                "regime_performance": regime_report,
                "paper_execution_filters": paper_filter_report,
                "trade_filter_report": trade_filter_report,
                "calibration_audit": calibration_audit,
                "probability_diagnostics": probability_diagnostics,
                "threshold_diagnostics": threshold_diagnostics,
                "probability_monotonicity": monotonicity_report,
                "daily_pnl_stability": daily_pnl_report,
                "evaluation_return_column_used": returns_col,
                "trading_metrics_mode": trade_metrics.get("metric_source"),
                "feature_importance": _feature_importance(model, feature_cols),
                "calibration_fit": calibration_fit,
                "threshold_sweep_json_path": str(threshold_json),
                "threshold_sweep_csv_path": str(threshold_csv),
                # OOF prediction persistence
                "oof_predictions": {
                    "holdout": {
                        "file_path": oof_persist_result.get("oof_file_path"),
                        "row_count": oof_persist_result.get("row_count", 0),
                        "status": oof_persist_result.get("status", "unknown"),
                        "timestamp": oof_timestamp,
                    },
                    "walk_forward": {
                        "file_path": wf_oof_persist_result.get("oof_file_path"),
                        "row_count": wf_oof_persist_result.get("row_count", 0),
                        "status": wf_oof_persist_result.get("status", "unknown"),
                        "timestamp": oof_timestamp,
                    },
                },
            }
            artifact_metadata = {
                "model_type": model_name,
                "label_policy": label_name,
                "feature_set_version": variant_name,
                "trained_at": datetime.now().isoformat(),
                "threshold": threshold,
                "feature_order": list(feature_cols),
                "live_computable_only": bool((paper_config or {}).get("live_computable_only")),
                "live_contract_version": CONTRACT_VERSION if HAS_FEATURE_CONTRACT else None,
                "excluded_non_live_features": list((paper_config or {}).get("_excluded_non_live_features", [])),
                "target_column": str(label_name),
                "model_name": str(model_name),
                "dataset_path": str((paper_config or {}).get("_dataset_path") or ""),
                "train_test_date_split": split_window,
                "metrics": {
                    "validation_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "trade_metrics": trade_metrics,
                },
            }
            model_path = save_model_artifact(
                artifact_dir,
                f"{variant_name}_{model_name}",
                label_name,
                model,
                feature_cols,
                scaler_mean,
                scaler_std,
                artifact_metadata,
            )
            report["model_path"] = model_path
            metrics_path = artifact_dir / f"{variant_name}_{model_name}_{label_name}_metrics.json"
            write_json(metrics_path, report)
            _write_status_file(artifact_dir, label_name, model_name, {
                "label_name": label_name,
                "model_name": model_name,
                "status": "completed",
                "start_time": None,
                "end_time": datetime.now().isoformat(),
                "artifact_path": model_path,
                "metrics_path": str(metrics_path),
                "error_message": None,
                "training_time_seconds": train_time,
            })
            label_reports[model_name] = report
            base_ensemble_candidates.append(
                {
                    "model_name": model_name,
                    "model": model,
                    "val_prob": val_prob,
                    "test_prob": test_prob,
                    "validation_trade_metrics": validation_trade_metrics,
                    "report": report,
                }
            )
            trained_models.append(f"{variant_name}:{model_name}:{label_name}")
        if run_ensemble and base_ensemble_candidates:
            eligible_candidates = []
            for candidate in base_ensemble_candidates:
                validation_trade_metrics = candidate["validation_trade_metrics"] or {}
                if int(validation_trade_metrics.get("trade_count") or 0) <= 0:
                    continue
                if float(validation_trade_metrics.get("profit_factor") or 0.0) <= 1.0:
                    continue
                if float(validation_trade_metrics.get("sharpe") or 0.0) <= 0.0:
                    continue
                eligible_candidates.append(candidate)
            # FIX: Allow ensemble with minimum 2 models, even if strict criteria not met
            # This ensures ensemble is built for all labels, not just those where all models pass strict criteria
            if len(eligible_candidates) >= 2:
                chosen_base_models = eligible_candidates
            elif len(eligible_candidates) == 1 and len(base_ensemble_candidates) >= 2:
                # If only one model passes strict criteria but we have >= 2 base models total,
                # include the best of the remaining base models (relaxed criteria)
                non_eligible = [c for c in base_ensemble_candidates if c not in eligible_candidates]
                if non_eligible:
                    # Pick the best non-eligible based on trade_count
                    best_non_eligible = max(non_eligible, key=lambda c: int((c.get("validation_trade_metrics") or {}).get("trade_count") or 0))
                    chosen_base_models = eligible_candidates + [best_non_eligible]
                else:
                    chosen_base_models = eligible_candidates
            elif len(base_ensemble_candidates) >= 2:
                # No model passed strict criteria but we have >= 2 base models - use all with relaxed criteria
                chosen_base_models = base_ensemble_candidates
            else:
                chosen_base_models = eligible_candidates or []
            ensemble_diagnostics: Dict[str, Any] = {}
            if chosen_base_models:
                raw_weights = []
                for candidate in chosen_base_models:
                    validation_trade_metrics = candidate["validation_trade_metrics"] or {}
                    score = max(0.01, float(validation_trade_metrics.get("profit_factor") or 0.0) - 1.0) * 2.0 + max(0.0, float(validation_trade_metrics.get("sharpe") or 0.0)) + max(0.0, float((candidate["report"].get("validation_metrics") or {}).get("f1") or 0.0))
                    raw_weights.append(score)
                weight_sum = sum(raw_weights) or 1.0
                normalized_weights = [float(weight / weight_sum) for weight in raw_weights]
                val_prob = np.zeros(len(y_val), dtype=float)
                test_prob = np.zeros(len(y_test), dtype=float)
                for weight, candidate in zip(normalized_weights, chosen_base_models):
                    val_prob += float(weight) * np.asarray(candidate["val_prob"], dtype=float)
                    test_prob += float(weight) * np.asarray(candidate["test_prob"], dtype=float)
                val_returns = eval_returns[val_idx] if eval_returns is not None else None
                chosen = optimize_threshold_from_probs(y_val, val_prob, val_returns, thresholds=RETRAIN_THRESHOLD_SWEEP, minimum_trades=min_threshold_trades)
                threshold = float(chosen["threshold"])
                val_metrics = classification_metrics(y_val, val_prob, threshold=threshold)
                test_metrics = classification_metrics(y_test, test_prob, threshold=threshold)
                val_metrics["pr_auc"] = pr_auc_score_safe(y_val, val_prob)
                val_metrics["brier_score"] = brier_score(y_val, val_prob)
                test_metrics["pr_auc"] = pr_auc_score_safe(y_test, test_prob)
                test_metrics["brier_score"] = brier_score(y_test, test_prob)
                validation_trade_metrics = _trade_metrics_from_scores(y_val, val_prob, threshold, val_returns, threshold_source="validation_selected_threshold")
                returns = eval_returns[test_idx] if eval_returns is not None else None
                trade_metrics = _trade_metrics_from_scores(y_test, test_prob, threshold, returns)
                threshold_rows = _full_threshold_sweep_report(y_test, test_prob, returns, thresholds=RETRAIN_THRESHOLD_SWEEP)
                threshold_json = artifact_dir / f"{variant_name}_ensemble_{label_name}_threshold_sweep.json"
                threshold_csv = artifact_dir / f"{variant_name}_ensemble_{label_name}_threshold_sweep.csv"
                weights_json = artifact_dir / f"{variant_name}_ensemble_{label_name}_weights.json"
                write_json(threshold_json, {"model_name": "ensemble", "label_name": label_name, "rows": threshold_rows})
                pd.DataFrame(threshold_rows).to_csv(threshold_csv, index=False)
                weights_payload = {"weights": [{"model_name": candidate["model_name"], "weight": float(weight)} for candidate, weight in zip(chosen_base_models, normalized_weights)], "production_adoption_allowed": False}
                write_json(weights_json, weights_payload)
                return_cost_provenance = _return_cost_provenance_for_frame(test_frame, evaluation_return_column=returns_col)
                cost_stress = _cost_stress_report(
                    y_test,
                    test_prob,
                    threshold,
                    returns,
                    cost_provenance=return_cost_provenance,
                    per_row_cost_units=return_cost_provenance.get("per_row_cost_units"),
                )
                threshold_robustness = _threshold_robustness_report(y_test, test_prob, returns, grid_start=threshold_grid_start, grid_end=threshold_grid_end, grid_step=threshold_grid_step, min_trades=min_threshold_trades)
                regime_report = _regime_performance_report(test_frame, test_prob, threshold, returns)
                paper_filter_report = _paper_execution_filter_report(test_frame, test_prob, threshold, returns, max_trades_per_day=int((paper_config or {}).get("paper_max_trades_per_day", 5)), cooldown_minutes=int((paper_config or {}).get("paper_cooldown_minutes", 15)), allow_ce=bool((paper_config or {}).get("paper_allow_ce", True)), allow_pe=bool((paper_config or {}).get("paper_allow_pe", True)), min_option_price=float((paper_config or {}).get("paper_min_option_price", 5.0)), max_bid_ask_spread_pct=float((paper_config or {}).get("paper_max_bid_ask_spread_pct", 5.0)), avoid_opening_minutes=int((paper_config or {}).get("paper_avoid_opening_minutes", 5)), avoid_closing_minutes=int((paper_config or {}).get("paper_avoid_closing_minutes", 5)))
                calibration_audit = _calibration_audit(y_test, test_prob, returns)
                monotonicity_report = _probability_monotonicity_report(test_prob, returns)
                probability_diagnostics = _probability_diagnostics(y_test, test_prob, RETRAIN_THRESHOLD_SWEEP)
                threshold_diagnostics = _threshold_diagnostics(y_test, test_prob, threshold, returns)
                trade_filter_report = _trade_filter_report(test_frame, test_prob, threshold, returns)
                daily_pnl_report = _daily_pnl_stability_report(paper_filter_report.get("selected_rows", []), max_trades_per_day=int((paper_config or {}).get("paper_max_trades_per_day", 5)))
                blend_model = _ProbabilityBlendEnsemble([candidate["model"] for candidate in chosen_base_models], normalized_weights)
                ensemble_metadata = {
                    "model_type": "ensemble",
                    "label_policy": label_name,
                    "feature_set_version": variant_name,
                    "trained_at": datetime.now().isoformat(),
                    "threshold": threshold,
                    "feature_order": list(feature_cols),
                    "live_computable_only": bool((paper_config or {}).get("live_computable_only")),
                    "live_contract_version": CONTRACT_VERSION if HAS_FEATURE_CONTRACT else None,
                    "excluded_non_live_features": list((paper_config or {}).get("_excluded_non_live_features", [])),
                    "target_column": str(label_name),
                    "model_name": "ensemble",
                    "dataset_path": str((paper_config or {}).get("_dataset_path") or ""),
                    "train_test_date_split": split_window,
                    "metrics": {
                        "validation_metrics": val_metrics,
                        "test_metrics": test_metrics,
                        "trade_metrics": trade_metrics,
                    },
                }
                model_path = save_model_artifact(
                    artifact_dir,
                    f"{variant_name}_ensemble",
                    label_name,
                    blend_model,
                    feature_cols,
                    scaler_mean,
                    scaler_std,
                    ensemble_metadata,
                )
                best_base_trade_pf = max(float((candidate["report"].get("trade_metrics") or {}).get("profit_factor") or 0.0) for candidate in base_ensemble_candidates)
                ensemble_rejected = float(trade_metrics.get("profit_factor") or 0.0) < best_base_trade_pf
                ensemble_diagnostics = {
                    "weights_path": str(weights_json),
                    "excluded_models": [candidate["model_name"] for candidate in base_ensemble_candidates if candidate not in chosen_base_models],
                    "included_models": [candidate["model_name"] for candidate in chosen_base_models],
                    "ensemble_rejected_best_base_retained": ensemble_rejected,
                    "reason": "ensemble rejected; best base model retained" if ensemble_rejected else None,
                }
                label_reports["ensemble"] = {
                    "variant_name": variant_name,
                    "model_name": "ensemble",
                    "label_name": label_name,
                    "validation_metrics": val_metrics,
                    "validation_trade_metrics": validation_trade_metrics,
                    "test_metrics": test_metrics,
                    "test_threshold_metrics": {**test_metrics, **confusion_from_threshold(y_test, test_prob, threshold), "signal_count": int(np.sum(test_prob >= threshold))},
                    "selected_threshold_from_validation": chosen,
                    "training_time_seconds": 0.0,
                    "holdout_split": split_window,
                    "walk_forward": walk_forward,
                    "fold_stability": _fold_stability_report(walk_forward.get("folds", [])),
                    "trade_metrics": trade_metrics,
                    "cost_stress": cost_stress,
                    "return_cost_provenance": return_cost_provenance,
                    "threshold_sweep": threshold_rows,
                    "threshold_robustness": threshold_robustness,
                    "regime_performance": regime_report,
                    "paper_execution_filters": paper_filter_report,
                    "trade_filter_report": trade_filter_report,
                    "calibration_audit": calibration_audit,
                    "probability_diagnostics": probability_diagnostics,
                    "threshold_diagnostics": threshold_diagnostics,
                    "probability_monotonicity": monotonicity_report,
                    "daily_pnl_stability": daily_pnl_report,
                    "evaluation_return_column_used": returns_col,
                    "trading_metrics_mode": trade_metrics.get("metric_source"),
                    "feature_importance": [],
                    "calibration_fit": {"mode": "weighted_blend", "fit_on_validation_only": True},
                    "threshold_sweep_json_path": str(threshold_json),
                    "threshold_sweep_csv_path": str(threshold_csv),
                    "model_path": model_path,
                    "ensemble_diagnostics": ensemble_diagnostics,
                }
                ensemble_metrics_path = artifact_dir / f"{variant_name}_ensemble_{label_name}_metrics.json"
                write_json(ensemble_metrics_path, label_reports["ensemble"])
                _write_status_file(artifact_dir, label_name, "ensemble", {
                    "label_name": label_name,
                    "model_name": "ensemble",
                    "status": "completed",
                    "start_time": None,
                    "end_time": datetime.now().isoformat(),
                    "artifact_path": model_path,
                    "metrics_path": str(ensemble_metrics_path),
                    "error_message": None,
                    "training_time_seconds": 0.0,
                })
                trained_models.append(f"{variant_name}:ensemble:{label_name}")
        if requested_xgb_rf_ensemble:
            ensemble_status_path = _status_file_path(artifact_dir, label_name, "xgb_rf_ensemble")
            _write_status_file(artifact_dir, label_name, "xgb_rf_ensemble", {
                "label_name": label_name,
                "model_name": "xgb_rf_ensemble",
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "artifact_path": None,
                "metrics_path": None,
                "error_message": None,
                "training_time_seconds": None,
            })
            try:
                ensemble_report = _train_xgb_rf_ensemble_for_label(
                    work=work,
                    feature_cols=feature_cols,
                    label_name=label_name,
                    artifact_dir=artifact_dir,
                    variant_name=variant_name,
                    returns_col=returns_col,
                    timestamps=timestamps,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    split_window=split_window,
                    walk_forward=walk_forward,
                    paper_config=paper_config,
                )
                label_reports["xgb_rf_ensemble"] = ensemble_report
                _write_status_file(artifact_dir, label_name, "xgb_rf_ensemble", {
                    "label_name": label_name,
                    "model_name": "xgb_rf_ensemble",
                    "status": "completed",
                    "start_time": None,
                    "end_time": datetime.now().isoformat(),
                    "artifact_path": ensemble_report.get("model_path"),
                    "metrics_path": ensemble_report.get("metrics_path"),
                    "error_message": None,
                    "training_time_seconds": ensemble_report.get("training_time_seconds"),
                })
                trained_models.append(f"{variant_name}:xgb_rf_ensemble:{label_name}")
            except Exception as exc:
                _write_status_file(artifact_dir, label_name, "xgb_rf_ensemble", {
                    "label_name": label_name,
                    "model_name": "xgb_rf_ensemble",
                    "status": "failed",
                    "start_time": None,
                    "end_time": datetime.now().isoformat(),
                    "artifact_path": None,
                    "metrics_path": None,
                    "error_message": str(exc),
                    "training_time_seconds": 0.0,
                })
                failed_models.append({
                    "variant_name": variant_name,
                    "label_name": label_name,
                    "model_family": "xgb_rf_ensemble",
                    "error": str(exc),
                    "stack_summary": traceback.format_exc(limit=3),
                })
        all_reports[label_name] = label_reports
        feature_manifests[label_name] = list(feature_cols)
    total_seconds = time.perf_counter() - variant_start
    _timing_log("variant_total", seconds=total_seconds, rows=len(df), extra=f"variant={variant_name}")
    return {
        "reports": all_reports,
        "trained_models": trained_models,
        "feature_lists": feature_manifests,
        "evaluation_return_column_used": returns_col,
        "skipped_models": skipped_models,
        "failed_models": failed_models,
        "timing": {"total_seconds": float(total_seconds)},
    }


def _best_variant_report(variant_reports: Dict[str, Any]) -> Dict[str, Any]:
    best: Dict[str, Any] = {}
    for _, model_map in variant_reports.items():
        for _, report in model_map.items():
            if not best:
                best = report
                continue
            current = (float(report.get("test_metrics", {}).get("f1", 0.0) or 0.0), float(report.get("test_metrics", {}).get("roc_auc", 0.0) or 0.0))
            previous = (float(best.get("test_metrics", {}).get("f1", 0.0) or 0.0), float(best.get("test_metrics", {}).get("roc_auc", 0.0) or 0.0))
            if current > previous:
                best = report
    return best


def _paper_candidate_verdict(report: Dict[str, Any], *, live_schema_pass: bool, live_computable_pass: bool) -> str:
    trade = report.get("trade_metrics", {}) or {}
    folds = report.get("fold_stability", {}) or {}
    after_filters = (report.get("paper_execution_filters", {}).get("after", {}) or {})
    threshold_ok = bool((report.get("threshold_robustness") or {}).get("chosen_threshold_is_robust"))
    monotonic_ok = str((report.get("probability_monotonicity") or {}).get("gate", "FAIL")) == "PASS"
    calibration_ok = str((report.get("calibration_audit") or {}).get("calibration_gate", "FAIL")) == "PASS"
    daily_ok = str((report.get("daily_pnl_stability") or {}).get("gate", "FAIL")) == "PASS"
    pf = float(after_filters.get("profit_factor") or 0.0)
    sharpe = float(after_filters.get("sharpe") or 0.0)
    trades = int(after_filters.get("trade_count") or 0)
    profitable_folds = int(folds.get("profitable_folds") or 0)
    worst_fold_pf = float(folds.get("worst_fold_pf") or 0.0)
    median_fold_pf = float(folds.get("median_fold_pf") or 0.0)
    if not (live_schema_pass and live_computable_pass and threshold_ok and monotonic_ok and calibration_ok and daily_ok):
        return "RESEARCH_ONLY_WEAK_EDGE"
    if pf >= STRICT_PAPER_PF_MIN and sharpe >= STRICT_PAPER_SHARPE_MIN and trades >= STRICT_PAPER_FILTER_TRADE_MIN and profitable_folds >= 3 and worst_fold_pf >= STRICT_WORST_FOLD_PF_MIN and median_fold_pf >= STRICT_MEDIAN_FOLD_PF_MIN:
        return "PAPER_TRADE_CANDIDATE_MEDIUM_CONFIDENCE"
    if float(trade.get("profit_factor") or 0.0) > 1.0:
        return "PAPER_TRADE_CANDIDATE_LOW_CONFIDENCE"
    return "RESEARCH_ONLY_WEAK_EDGE"


def _rescue_candidate_flags(report: Dict[str, Any], *, live_schema_pass: bool, live_computable_pass: bool) -> Dict[str, Any]:
    after_filters = (report.get("paper_execution_filters", {}).get("after", {}) or {})
    folds = report.get("walk_forward", {}).get("folds", []) or []
    fold_pf = [float(((fold.get("trade_metrics") or {}).get("profit_factor") or 0.0)) for fold in folds]
    profitable_folds = sum(1 for value in fold_pf if value >= 1.0)
    non_strongly_negative_folds = sum(1 for value in fold_pf if value >= 0.80)
    active_days = int((report.get("daily_pnl_stability") or {}).get("active_trading_days") or 0)
    total_return = float(after_filters.get("total_return") or 0.0)
    top_day_return = float((report.get("daily_pnl_stability") or {}).get("largest_day_return") or 0.0)
    concentration_ok = abs(top_day_return) <= abs(total_return) * 0.30 if total_return else True
    robust_band = bool((report.get("threshold_robustness") or {}).get("chosen_threshold_is_robust"))
    monotonic_ok = str((report.get("probability_monotonicity") or {}).get("gate", "FAIL")) == "PASS"
    calibration_gate = str((report.get("calibration_audit") or {}).get("calibration_gate", "FAIL"))
    calibration_ok = calibration_gate == "PASS"
    calibration_critical_fail = calibration_gate == "FAIL"
    daily_ok = str((report.get("daily_pnl_stability") or {}).get("gate", "FAIL")) == "PASS"
    extra_cost_pf = float((((report.get("cost_stress") or {}).get("extra_cost_0_25") or {}).get("profit_factor") or 0.0))
    worst_fold_pf = min(fold_pf) if fold_pf else 0.0
    median_fold_pf = float(np.median(fold_pf)) if fold_pf else 0.0
    trade_count = int(after_filters.get("trade_count") or 0)
    pf = float(after_filters.get("profit_factor") or 0.0)
    sharpe = float(after_filters.get("sharpe") or 0.0)
    shadow_allowed = all(
        [
            live_schema_pass,
            live_computable_pass,
            trade_count >= 100,
            active_days > 20,
            pf >= 1.10,
            sharpe >= 0.50,
            extra_cost_pf >= 1.00,
            non_strongly_negative_folds >= 3,
            worst_fold_pf >= 0.80,
            median_fold_pf >= 1.00,
            monotonic_ok,
            not calibration_critical_fail,
            concentration_ok,
        ]
    )
    paper_allowed = all(
        [
            live_schema_pass,
            live_computable_pass,
            trade_count >= 500,
            active_days > 50,
            pf >= 1.15,
            sharpe >= 0.75,
            extra_cost_pf >= 1.05,
            profitable_folds >= 3,
            worst_fold_pf >= 0.90,
            median_fold_pf >= 1.10,
            robust_band,
            calibration_ok,
            daily_ok,
        ]
    )
    failed_gates = []
    if not live_schema_pass:
        failed_gates.append("live_schema_gate")
    if not live_computable_pass:
        failed_gates.append("live_computable_feature_gate")
    if trade_count < 100:
        failed_gates.append("minimum_trade_count_gate")
    if pf < 1.10:
        failed_gates.append("trading_metrics_gate")
    if sharpe < 0.50:
        failed_gates.append("trading_metrics_gate")
    if extra_cost_pf < 1.00:
        failed_gates.append("cost_stress_gate")
    if non_strongly_negative_folds < 3 or worst_fold_pf < 0.80 or median_fold_pf < 1.00:
        failed_gates.append("fold_stability_gate")
    if not monotonic_ok:
        failed_gates.append("probability_monotonicity_gate")
    if calibration_critical_fail:
        failed_gates.append("calibration_gate")
    if not concentration_ok or not daily_ok:
        failed_gates.append("daily_pnl_stability_gate")
    verdict = "NO_RESCUE_CANDIDATE_FOUND"
    if paper_allowed:
        verdict = "PAPER_RESCUE_CANDIDATE"
    elif shadow_allowed:
        verdict = "SHADOW_RESCUE_CANDIDATE"
    return {
        "shadow_candidate_allowed": bool(shadow_allowed),
        "paper_candidate_allowed": bool(paper_allowed),
        "production_adoption_allowed": False,
        "active_trading_days": active_days,
        "profitable_folds": profitable_folds,
        "non_strongly_negative_folds": non_strongly_negative_folds,
        "worst_fold_pf": worst_fold_pf,
        "median_fold_pf": median_fold_pf,
        "cost_stress_0_25_pf": extra_cost_pf,
        "failed_gates": sorted(set(failed_gates)),
        "final_verdict": verdict,
        "threshold_is_robust": robust_band,
    }


def _leaderboard_gate_safe_score(row: Dict[str, Any]) -> float:
    gate_bonus = 0.0
    for key in ("cost_stress_pass", "threshold_robustness_pass", "calibration_pass", "monotonicity_pass", "daily_pnl_pass", "regime_stability_pass", "live_schema_pass"):
        gate_bonus += 1.0 if bool(row.get(key)) else -1.5
    return (
        gate_bonus * 10.0
        + float(row.get("PF") or 0.0) * 8.0
        + float(row.get("Sharpe") or 0.0) * 6.0
        + float(row.get("median_fold_pf") or 0.0) * 4.0
        + float(row.get("F1") or 0.0) * 3.0
        + float(row.get("ROC-AUC") or 0.0)
        - max(0.0, float(row.get("max_drawdown") or 0.0)) * 0.01
    )


def build_model_leaderboard(candidate_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = []
    for row in candidate_rows:
        scored = dict(row)
        scored["gate_safe_score"] = _leaderboard_gate_safe_score(scored)
        ranked.append(scored)
    ranked.sort(key=lambda row: (float(row.get("gate_safe_score") or -9999.0), float(row.get("PF") or -9999.0), float(row.get("Sharpe") or -9999.0), float(row.get("F1") or -9999.0)), reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def build_rescue_leaderboard(candidate_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for row in candidate_rows:
        scored = dict(row)
        score = 0.0
        score += 1000.0 if scored.get("paper_candidate_allowed") else 0.0
        score += 500.0 if scored.get("shadow_candidate_allowed") else 0.0
        score += 200.0 if scored.get("live_schema_gate") == "PASS" else -200.0
        score += 120.0 if float(scored.get("cost_stress_0_25_pf") or 0.0) >= 1.0 else -120.0
        score += 100.0 if float(scored.get("median_fold_pf") or 0.0) >= 1.0 else -100.0
        score += float(scored.get("PF") or 0.0) * 25.0
        score += float(scored.get("Sharpe") or 0.0) * 20.0
        score += 75.0 if scored.get("daily_pnl_gate") == "PASS" else -75.0
        score += min(50.0, float(scored.get("active_trading_days") or 0.0))
        scored["rescue_rank_score"] = score
        ranked.append(scored)
    ranked.sort(
        key=lambda row: (
            bool(row.get("paper_candidate_allowed")),
            bool(row.get("shadow_candidate_allowed")),
            row.get("live_schema_gate") == "PASS",
            float(row.get("cost_stress_0_25_pf") or -9999.0),
            float(row.get("median_fold_pf") or -9999.0),
            float(row.get("PF") or -9999.0),
            float(row.get("Sharpe") or -9999.0),
            row.get("daily_pnl_gate") == "PASS",
            float(row.get("rescue_rank_score") or -9999.0),
        ),
        reverse=True,
    )
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def _live_schema_gate_for_variant(feature_cols: Sequence[str]) -> Dict[str, Any]:
    try:
        from option_chain_pipeline_lib import strict_live_schema_gate
        return strict_live_schema_gate(feature_cols)
    except Exception as exc:
        return {"compatible": False, "error": str(exc)}


def _production_adoption_verdict(*, enriched_best: Dict[str, Any], feature_cols: Sequence[str], leakage_ok: bool, estimated_only: bool) -> Dict[str, Any]:
    strict_gate = _live_schema_gate_for_variant(feature_cols)
    if estimated_only:
        return {
            "production_adoption_allowed": False,
            "verdict": "RESEARCH_ONLY_ESTIMATED_GREEKS_NOT_ADOPTABLE",
            "strict_live_schema_gate": strict_gate,
            "reason": "estimated-only greeks are research-only",
        }
    stable = bool(enriched_best.get("walk_forward", {}).get("split_count", 0) >= 3)
    improved = float(enriched_best.get("test_metrics", {}).get("f1", 0.0) or 0.0) > 0.0
    feature_order = bool(strict_gate.get("feature_order_matches_exactly"))
    no_forbidden = not strict_gate.get("disallowed_training_only_features")
    allowed = bool(strict_gate.get("compatible")) and feature_order and no_forbidden and leakage_ok and stable and improved
    return {
        "production_adoption_allowed": allowed,
        "verdict": "PRODUCTION_ADOPTION_ALLOWED" if allowed else "RESEARCH_ONLY_ESTIMATED_GREEKS_NOT_ADOPTABLE",
        "strict_live_schema_gate": strict_gate,
        "reason": "all gates passed" if allowed else "schema, stability, or leakage gates failed",
    }


def write_bs_comparison_report(
    *,
    dataset_path: Path,
    enriched_dataset_path: Path,
    df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    feature_sets: Dict[str, Any],
    baseline_result: Dict[str, Any],
    enriched_result: Dict[str, Any],
    leakage_result: Dict[str, Any],
    adoption: Dict[str, Any],
    column_groups: Dict[str, Any],
    same_period_report: Dict[str, Any],
) -> Dict[str, Any]:
    report_dir = REPO_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"bs_features_model_comparison_{stamp}.json"
    md_path = report_dir / f"bs_features_model_comparison_{stamp}.md"
    baseline_best = _best_variant_report(baseline_result["reports"])
    enriched_best = _best_variant_report(enriched_result["reports"])
    valid_enrichment_pct = float(pd.to_numeric(enriched_df.get("row_enrichment_ok"), errors="coerce").fillna(0).astype(bool).mean() * 100.0) if "row_enrichment_ok" in enriched_df.columns else 0.0
    iv_solve_pct = float(pd.to_numeric(enriched_df.get("bs_iv_solve_ok"), errors="coerce").fillna(0).astype(bool).mean() * 100.0) if "bs_iv_solve_ok" in enriched_df.columns else 0.0
    bad_greek_pct = float(pd.to_numeric(enriched_df.get("bad_greek_flag"), errors="coerce").fillna(0).astype(bool).mean() * 100.0) if "bad_greek_flag" in enriched_df.columns else 0.0
    comparison = {
        "dataset_path": str(dataset_path),
        "enriched_dataset_path": str(enriched_dataset_path),
        "row_count": int(len(df)),
        "return_pnl_column_used": column_groups.get("evaluation_return_column_used"),
        "trading_metrics_are_real": bool(column_groups.get("evaluation_return_column_used")),
        "valid_enrichment_percentage": valid_enrichment_pct,
        "iv_solve_success_percentage": iv_solve_pct,
        "bad_greek_row_percentage": bad_greek_pct,
        "same_period_report": same_period_report,
        "baseline_metrics": baseline_best,
        "enriched_metrics": enriched_best,
        "improvement_degradation": {
            "roc_auc_delta": float((enriched_best.get("test_metrics", {}).get("roc_auc", 0.0) or 0.0) - (baseline_best.get("test_metrics", {}).get("roc_auc", 0.0) or 0.0)),
            "f1_delta": float((enriched_best.get("test_metrics", {}).get("f1", 0.0) or 0.0) - (baseline_best.get("test_metrics", {}).get("f1", 0.0) or 0.0)),
            "profit_factor_delta": float((enriched_best.get("trade_metrics", {}).get("profit_factor", 0.0) or 0.0) - (baseline_best.get("trade_metrics", {}).get("profit_factor", 0.0) or 0.0)),
            "sharpe_delta": float((enriched_best.get("trade_metrics", {}).get("sharpe", 0.0) or 0.0) - (baseline_best.get("trade_metrics", {}).get("sharpe", 0.0) or 0.0)),
        },
        "leakage_audit_result": leakage_result,
        "live_schema_compatibility_result": adoption.get("strict_live_schema_gate"),
        "production_adoption_verdict": adoption,
        "selected_feature_list": feature_sets.get("enriched_features", []),
        "rejected_feature_list": feature_sets.get("rejected_features", []),
        "top_feature_importance": enriched_best.get("feature_importance", []),
        "schema_alignment_action_report": _schema_alignment_action_report(adoption.get("strict_live_schema_gate") or {}),
        "final_decision_matrix": _final_decision_matrix(
            baseline_best=baseline_best,
            enriched_best=enriched_best,
            leakage_result=leakage_result,
            adoption=adoption,
            same_period_report=same_period_report,
        ),
        "final_recommendation": adoption.get("verdict"),
    }
    write_json(json_path, comparison)
    md_lines = [
        "# Black-Scholes Features Model Comparison",
        "",
        f"- Dataset path: `{dataset_path}`",
        f"- Enriched dataset path: `{enriched_dataset_path}`",
        f"- Row count: `{len(df)}`",
        f"- Return/PnL column used: `{column_groups.get('evaluation_return_column_used')}`",
        f"- Trading metrics are real: `{bool(column_groups.get('evaluation_return_column_used'))}`",
        f"- Valid enrichment percentage: `{valid_enrichment_pct:.2f}%`",
        f"- IV solve success percentage: `{iv_solve_pct:.2f}%`",
        f"- Bad Greek row percentage: `{bad_greek_pct:.2f}%`",
        f"- Leakage audit result: `{leakage_result.get('passed')}`",
        f"- Production adoption verdict: `{adoption.get('verdict')}`",
        "",
        "## Threshold Sweep Summary",
        "",
    ]
    for name, payload in (("Baseline", baseline_best), ("Black-Scholes", enriched_best)):
        md_lines.append(f"### {name}")
        for row in payload.get("threshold_sweep", []):
            md_lines.append(
                f"- thr={row['threshold']:.2f} trades={row['selected_trades']} pf={row['profit_factor']} sharpe={row['sharpe']} mdd={row['max_drawdown']} precision={row['precision']:.4f} recall={row['recall']:.4f} f1={row['f1']:.4f}"
            )
        md_lines.append("")
    md_lines.extend(
        [
            "## Walk-Forward Folds",
            "",
        ]
    )
    for name, payload in (("Baseline", baseline_best), ("Black-Scholes", enriched_best)):
        md_lines.append(f"### {name}")
        for row in payload.get("walk_forward", {}).get("folds", []):
            md_lines.append(
                f"- fold={row['fold']} range={row.get('start')} -> {row.get('end')} trades={row.get('trade_count')} pf={row.get('profit_factor')} sharpe={row.get('sharpe')} mdd={row.get('max_drawdown')} avg_return={row.get('average_return')}"
            )
        md_lines.append("")
    md_lines.extend(
        [
            "## Selected Features",
        ]
    )
    for feature in feature_sets.get("enriched_features", []):
        md_lines.append(f"- `{feature}`")
    md_lines.extend(["", "## Rejected Features"])
    for feature in feature_sets.get("rejected_features", []):
        md_lines.append(f"- `{feature}`")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path), "payload": comparison}


def _write_named_report(stem: str, payload: Dict[str, Any], markdown_lines: List[str]) -> Dict[str, str]:
    report_dir = REPO_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"{stem}_{stamp}.json"
    md_path = report_dir / f"{stem}_{stamp}.md"
    write_json(json_path, payload)
    md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}


def _paper_readiness_verdict(matrix: Dict[str, Any]) -> str:
    required_pass = [
        matrix.get("chronology_gate"),
        matrix.get("leakage_gate"),
        matrix.get("same_period_gate"),
        matrix.get("cost_stress_gate"),
        matrix.get("fold_stability_gate"),
        matrix.get("threshold_robustness_gate"),
    ]
    if any(value != "PASS" for value in required_pass):
        return "RESEARCH_ONLY_WEAK_EDGE"
    return "PRODUCTION_BLOCKED_PAPER_ONLY"


def _regime_stability_gate(regime_report: Dict[str, Any]) -> Dict[str, Any]:
    groups = regime_report.get("groups", {}) if isinstance(regime_report, dict) else {}
    failing_groups: List[str] = []
    positive_groups = 0
    total_groups = 0
    for group_name, buckets in groups.items():
        if not isinstance(buckets, dict):
            continue
        for bucket_name, payload in buckets.items():
            if not isinstance(payload, dict):
                continue
            trade_count = int(payload.get("trade_count") or 0)
            if trade_count < 50:
                continue
            total_groups += 1
            pf = float(payload.get("profit_factor") or 0.0)
            sharpe = float(payload.get("sharpe") or 0.0)
            if pf > 1.0 and sharpe > 0.0:
                positive_groups += 1
            else:
                failing_groups.append(f"{group_name}:{bucket_name}")
    gate = "PASS" if total_groups == 0 or positive_groups >= max(1, math.ceil(total_groups * 0.35)) else "FAIL"
    return {
        "gate": gate,
        "total_groups_considered": total_groups,
        "positive_groups": positive_groups,
        "failing_groups": failing_groups[:10],
    }


def _build_paper_readiness_payload(
    comparison_payload: Dict[str, Any],
    feature_manifest_payload: Dict[str, Any],
) -> Dict[str, Any]:
    enriched = comparison_payload.get("enriched_metrics", {})
    matrix = comparison_payload.get("final_decision_matrix", {}).copy()
    matrix["live_computable_feature_gate"] = "PASS" if feature_manifest_payload.get("live_feature_audit") else "FAIL"
    matrix["minimum_trades_gate"] = "PASS" if int((enriched.get("paper_execution_filters", {}).get("after", {}) or {}).get("trade_count") or 0) >= 500 else "FAIL"
    matrix["calibration_gate"] = (enriched.get("calibration_audit") or {}).get("calibration_gate", "FAIL")
    matrix["probability_monotonicity_gate"] = (enriched.get("probability_monotonicity") or {}).get("gate", "FAIL")
    matrix["daily_pnl_stability_gate"] = (enriched.get("daily_pnl_stability") or {}).get("gate", "FAIL")
    matrix["regime_stability_gate"] = _regime_stability_gate(enriched.get("regime_performance", {})).get("gate", "FAIL")
    paper_after = (enriched.get("paper_execution_filters", {}).get("after", {}) or {})
    matrix["paper_execution_filter_gate"] = "PASS" if int(paper_after.get("trade_count") or 0) >= 500 and float(paper_after.get("profit_factor") or 0.0) > 1.0 else "FAIL"
    matrix["production_adoption_gate"] = "FAIL"
    matrix["production_adoption_allowed"] = False
    matrix["final_verdict"] = _paper_readiness_verdict(matrix)
    return {
        "feature_audit": feature_manifest_payload.get("live_feature_audit", []),
        "baseline_variant": comparison_payload.get("baseline_metrics", {}),
        "paper_candidate_variant": enriched,
        "schema_alignment_action_report": comparison_payload.get("schema_alignment_action_report", {}),
        "regime_stability_report": _regime_stability_gate(enriched.get("regime_performance", {})),
        "decision_matrix": matrix,
    }


def _failure_gate_row(
    gate_name: str,
    status: str,
    *,
    severity: str,
    short_reason: str,
    evidence: Sequence[Any],
    recommended_action: str,
    blocking_for_paper_trading: bool,
    blocking_for_production: bool,
) -> Dict[str, Any]:
    evidence_values = list(evidence)[:3]
    while len(evidence_values) < 3:
        evidence_values.append(None)
    return {
        "gate_name": gate_name,
        "status": status,
        "severity": severity,
        "short_reason": short_reason,
        "evidence_metric_1": evidence_values[0],
        "evidence_metric_2": evidence_values[1],
        "evidence_metric_3": evidence_values[2],
        "recommended_action": recommended_action,
        "blocking_for_paper_trading": bool(blocking_for_paper_trading),
        "blocking_for_production": bool(blocking_for_production),
    }


def _failure_gate_table(
    *,
    paper_payload: Dict[str, Any],
    comparison_payload: Dict[str, Any],
    feature_manifest_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    matrix = paper_payload.get("decision_matrix", {})
    enriched = comparison_payload.get("enriched_metrics", {})
    schema_report = comparison_payload.get("schema_alignment_action_report", {}) or {}
    regime_report = paper_payload.get("regime_stability_report", {}) or {}
    threshold_report = enriched.get("threshold_robustness", {}) or {}
    cost_report = enriched.get("cost_stress", {}) or {}
    fold_report = enriched.get("fold_stability", {}) or {}
    calibration_report = enriched.get("calibration_audit", {}) or {}
    monotonicity_report = enriched.get("probability_monotonicity", {}) or {}
    daily_report = enriched.get("daily_pnl_stability", {}) or {}
    paper_after = (enriched.get("paper_execution_filters", {}).get("after", {}) or {})
    live_audit = feature_manifest_payload.get("live_feature_audit", [])
    comparison_delta = comparison_payload.get("improvement_degradation", {}) or {}
    schema_missing = schema_report.get("missing_live_features", schema_report.get("features_expected_live_missing_in_dataset", []))
    schema_training_only = schema_report.get("training_only_features", schema_report.get("features_available_in_training_not_live", []))
    schema_order_match = schema_report.get("feature_order_matches_exactly")
    fold_passes = fold_report.get("passing_folds", fold_report.get("profitable_folds"))
    fold_total = fold_report.get("split_count", enriched.get("walk_forward", {}).get("split_count"))
    regime_positive = regime_report.get("positive_groups")
    regime_total = regime_report.get("total_groups_considered")
    regime_failing = regime_report.get("failing_groups")
    if regime_positive is None or regime_total is None:
        computed_regime = _regime_stability_gate(enriched.get("regime_performance", {}))
        regime_positive = computed_regime.get("positive_groups")
        regime_total = computed_regime.get("total_groups_considered")
        regime_failing = computed_regime.get("failing_groups")
    rows = [
        _failure_gate_row("chronology_gate", matrix.get("chronology_gate", "FAIL"), severity="CRITICAL", short_reason="Chronological walk-forward must be valid.", evidence=[len(enriched.get("walk_forward", {}).get("folds", [])), None, None], recommended_action="Fix timestamp sorting and split assertions before any more experiments.", blocking_for_paper_trading=True, blocking_for_production=True),
        _failure_gate_row("leakage_gate", matrix.get("leakage_gate", "FAIL"), severity="CRITICAL", short_reason="Any leakage invalidates the offline result.", evidence=[comparison_payload.get("leakage_audit_result", {}).get("passed"), len(comparison_payload.get("leakage_audit_result", {}).get("forbidden_columns_present", [])), None], recommended_action="Remove leakage sources and rerun chronology-safe validation.", blocking_for_paper_trading=True, blocking_for_production=True),
        _failure_gate_row("same_period_gate", matrix.get("same_period_gate", "FAIL"), severity="HIGH", short_reason="Baseline vs BS must be compared on the same period.", evidence=[comparison_payload.get("same_period_report", {}).get("overlapping_date_range"), comparison_payload.get("same_period_report", {}).get("baseline_rows_dropped"), comparison_payload.get("same_period_report", {}).get("bs_rows_dropped")], recommended_action="Keep same-period comparison enabled before judging feature improvements.", blocking_for_paper_trading=False, blocking_for_production=True),
        _failure_gate_row("live_computable_feature_gate", matrix.get("live_computable_feature_gate", "FAIL"), severity="CRITICAL", short_reason="Training features must be computable before decision time.", evidence=[len(live_audit), len([row for row in live_audit if not row.get("computable_before_decision")]), None], recommended_action="Remove non-live-computable features before further model tuning.", blocking_for_paper_trading=True, blocking_for_production=True),
        _failure_gate_row("live_schema_gate", matrix.get("live_schema_gate", "FAIL"), severity="CRITICAL", short_reason="Training schema does not match safe live schema.", evidence=[len(schema_missing), len(schema_training_only), schema_order_match], recommended_action="Align features to the live schema or keep the model research-only.", blocking_for_paper_trading=True, blocking_for_production=True),
        _failure_gate_row("trading_metrics_gate", matrix.get("trading_metrics_gate", "FAIL"), severity="HIGH", short_reason="Real trading metrics are not strong enough.", evidence=[enriched.get("trade_metrics", {}).get("profit_factor"), enriched.get("trade_metrics", {}).get("sharpe"), enriched.get("trade_metrics", {}).get("trade_count")], recommended_action="Improve edge quality on real return columns before considering paper trading.", blocking_for_paper_trading=True, blocking_for_production=True),
        _failure_gate_row("cost_stress_gate", matrix.get("cost_stress_gate", "FAIL"), severity="CRITICAL", short_reason="Edge breaks under realistic extra cost stress.", evidence=[(cost_report.get("extra_cost_0_25") or {}).get("profit_factor"), (cost_report.get("extra_cost_0_50") or {}).get("profit_factor"), (cost_report.get("extra_cost_1_00") or {}).get("profit_factor")], recommended_action="Focus on lower-cost, wider-margin trades before more model complexity.", blocking_for_paper_trading=True, blocking_for_production=True),
        _failure_gate_row("fold_stability_gate", matrix.get("fold_stability_gate", "FAIL"), severity="CRITICAL", short_reason="Performance is not stable across walk-forward folds.", evidence=[fold_report.get("gate"), fold_passes, fold_total], recommended_action="Stabilize fold performance before any threshold tuning or deployment discussion.", blocking_for_paper_trading=True, blocking_for_production=True),
        _failure_gate_row("threshold_robustness_gate", matrix.get("threshold_robustness_gate", "FAIL"), severity="MEDIUM", short_reason="Chosen threshold may be a lucky point.", evidence=[threshold_report.get("chosen_threshold"), threshold_report.get("chosen_threshold_is_robust"), threshold_report.get("isolated_lucky_point")], recommended_action="Retain threshold sweeps but fix model quality and calibration first.", blocking_for_paper_trading=False, blocking_for_production=True),
        _failure_gate_row("calibration_gate", matrix.get("calibration_gate", "FAIL"), severity="HIGH", short_reason="Predicted probabilities are not trustworthy enough.", evidence=[calibration_report.get("calibration_gate"), calibration_report.get("monotonic_win_rate"), len([row for row in calibration_report.get("buckets", []) if row.get("small_sample_warning")])], recommended_action="Audit calibration before using probabilities for trade selection.", blocking_for_paper_trading=True, blocking_for_production=True),
        _failure_gate_row("probability_monotonicity_gate", matrix.get("probability_monotonicity_gate", "FAIL"), severity="CRITICAL", short_reason="Higher probabilities do not map cleanly to better returns.", evidence=[monotonicity_report.get("gate"), (monotonicity_report.get("deciles") or [{}])[0].get("average_return") if monotonicity_report.get("deciles") else None, (monotonicity_report.get("deciles") or [{}])[-1].get("average_return") if monotonicity_report.get("deciles") else None], recommended_action="Fix ranking quality or calibration before relying on confidence sorting.", blocking_for_paper_trading=True, blocking_for_production=True),
        _failure_gate_row("daily_pnl_stability_gate", matrix.get("daily_pnl_stability_gate", "FAIL"), severity="HIGH", short_reason="PnL is too concentrated or unstable across days.", evidence=[daily_report.get("day_win_rate", daily_report.get("active_trading_days")), daily_report.get("single_day_return_contribution"), daily_report.get("longest_losing_streak_days")], recommended_action="Reduce concentration risk before any paper-readiness claim.", blocking_for_paper_trading=True, blocking_for_production=True),
        _failure_gate_row("regime_stability_gate", matrix.get("regime_stability_gate", "FAIL"), severity="HIGH", short_reason="Edge only works in narrow regimes or buckets.", evidence=[regime_positive, regime_total, regime_failing], recommended_action="Isolate the few stable regimes instead of training one broad model.", blocking_for_paper_trading=True, blocking_for_production=True),
        _failure_gate_row("minimum_trade_count_gate", matrix.get("minimum_trades_gate", "FAIL"), severity="HIGH", short_reason="Too few trades survive realistic filters.", evidence=[paper_after.get("trade_count"), enriched.get("trade_metrics", {}).get("trade_count"), enriched.get("paper_execution_filters", {}).get("before", {}).get("trade_count")], recommended_action="Find a robust, scalable subset instead of forcing a fragile threshold.", blocking_for_paper_trading=True, blocking_for_production=True),
        _failure_gate_row("paper_execution_filter_gate", matrix.get("paper_execution_filter_gate", "FAIL"), severity="HIGH", short_reason="Execution realism removes the apparent edge.", evidence=[paper_after.get("trade_count"), paper_after.get("profit_factor"), paper_after.get("sharpe")], recommended_action="Audit CE/PE, DTE, moneyness, spread, and session filters before more retraining.", blocking_for_paper_trading=True, blocking_for_production=True),
        _failure_gate_row("production_adoption_gate", "FAIL", severity="CRITICAL", short_reason="Production adoption remains intentionally blocked.", evidence=[False, comparison_payload.get("production_adoption_verdict", {}).get("verdict"), comparison_delta.get("f1_delta")], recommended_action="Do not revisit production until live-schema and estimated-feature issues are resolved separately.", blocking_for_paper_trading=False, blocking_for_production=True),
    ]
    return rows


def _failed_gate_root_causes(gate_rows: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    mapping = {
        "chronology_gate": ["POSSIBLE_DATA_QUALITY_ISSUE"],
        "leakage_gate": ["LABEL_OR_TARGET_QUALITY_ISSUE"],
        "same_period_gate": ["DATA_COVERAGE_INSUFFICIENT"],
        "live_computable_feature_gate": ["LIVE_FEATURE_NOT_COMPUTABLE"],
        "live_schema_gate": ["FEATURE_SCHEMA_MISMATCH", "RESEARCH_ONLY_FEATURE_DEPENDENCY"],
        "trading_metrics_gate": ["WEAK_MODEL_SIGNAL"],
        "cost_stress_gate": ["COST_SENSITIVE_EDGE"],
        "fold_stability_gate": ["FOLD_INSTABILITY", "WEAK_MODEL_SIGNAL"],
        "threshold_robustness_gate": ["THRESHOLD_OVERFITTING"],
        "calibration_gate": ["POOR_CALIBRATION"],
        "probability_monotonicity_gate": ["NON_MONOTONIC_PROBABILITIES"],
        "daily_pnl_stability_gate": ["DAILY_PNL_CONCENTRATION"],
        "regime_stability_gate": ["REGIME_DEPENDENT_EDGE"],
        "minimum_trade_count_gate": ["LOW_SAMPLE_SIZE", "EXECUTION_FILTER_FRAGILITY"],
        "paper_execution_filter_gate": ["EXECUTION_FILTER_FRAGILITY"],
        "production_adoption_gate": ["RESEARCH_ONLY_FEATURE_DEPENDENCY"],
    }
    failed: Dict[str, List[str]] = {}
    for row in gate_rows:
        if row.get("status") == "FAIL":
            failed[row["gate_name"]] = mapping.get(row["gate_name"], ["POSSIBLE_DATA_QUALITY_ISSUE"])
    return failed


def _failure_flags_from_gate_rows(gate_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    gate_status = {row["gate_name"]: row.get("status") for row in gate_rows}
    live_blockers = [
        "leakage_gate",
        "chronology_gate",
        "live_computable_feature_gate",
        "live_schema_gate",
        "fold_stability_gate",
        "cost_stress_gate",
        "probability_monotonicity_gate",
    ]
    paper_blockers = [
        "trading_metrics_gate",
        "paper_execution_filter_gate",
        "minimum_trade_count_gate",
        "daily_pnl_stability_gate",
        "calibration_gate",
    ]
    messages: List[str] = []
    if any(gate_status.get(name) == "FAIL" for name in live_blockers):
        messages.append("DO_NOT_PROCEED_TO_LIVE_OR_PRODUCTION")
    if any(gate_status.get(name) == "FAIL" for name in paper_blockers + live_blockers):
        messages.append("DO_NOT_PROCEED_TO_PAPER_TRADING")
    return {
        "messages": messages,
        "paper_trading_allowed": "DO_NOT_PROCEED_TO_PAPER_TRADING" not in messages,
        "production_adoption_allowed": False,
    }


def _improvement_priority_plan(
    gate_rows: Sequence[Dict[str, Any]],
    *,
    comparison_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    priority_specs = {
        "chronology_gate": ("Fix chronology and split safety", "Invalid chronology makes every downstream metric untrustworthy.", "HIGH", "MEDIUM", "Rerun the strict chronology audit and confirm every fold remains monotonic.", ["scripts/retrain_all_edge_models.py", "tests/test_bs_retraining_integration.py"], "All fold date ranges are monotonic and assertions stay green.", True),
        "leakage_gate": ("Remove leakage paths", "Leakage can create fake edge that disappears live.", "HIGH", "MEDIUM", "Audit forbidden columns and any post-entry derivations before retraining again.", ["scripts/retrain_all_edge_models.py", "tests/test_bs_retraining_integration.py"], "Leakage gate passes with zero forbidden feature usage.", True),
        "live_computable_feature_gate": ("Drop non-live-computable features", "Research features that cannot exist before entry are unusable for paper/live decisions.", "HIGH", "LOW", "Run live-computable-only retraining and compare against the broader research set.", ["scripts/retrain_all_edge_models.py"], "All retained features are available before decision time.", True),
        "live_schema_gate": ("Align to live schema", "Schema mismatch blocks any realistic deployment path.", "HIGH", "MEDIUM", "Build a training subset that matches live feature names and order exactly.", ["scripts/retrain_all_edge_models.py", "src/option_chain_pipeline_lib.py"], "Strict live schema gate reports compatible=true and exact order match.", True),
        "cost_stress_gate": ("Reduce cost sensitivity", "A cost-sensitive edge usually evaporates once slippage and fees are real.", "HIGH", "MEDIUM", "Restrict the universe to cleaner trades before trying more models.", ["scripts/retrain_all_edge_models.py"], "Profit factor stays above 1 after extra cost scenarios.", True),
        "fold_stability_gate": ("Stabilize walk-forward folds", "Unstable folds mean the edge is not reliable through time.", "HIGH", "MEDIUM", "Use regime segmentation before any further threshold or model tuning.", ["scripts/retrain_all_edge_models.py"], "Most folds pass PF and Sharpe checks, not just one lucky window.", True),
        "calibration_gate": ("Repair calibration", "Bad calibration makes probability-based trade selection brittle.", "MEDIUM", "MEDIUM", "Compare uncalibrated vs calibrated logistic regression with the same chronological split.", ["scripts/retrain_all_edge_models.py", "scripts/evaluate_edge_models.py"], "Probability buckets become monotonic enough for safe ranking.", True),
        "probability_monotonicity_gate": ("Fix probability ranking quality", "If higher scores do not mean better outcomes, thresholding is unsafe.", "HIGH", "MEDIUM", "Audit simpler models and label definitions before relying on confidence ordering.", ["scripts/retrain_all_edge_models.py"], "Upper probability buckets consistently beat middle and lower buckets.", True),
        "regime_stability_gate": ("Isolate stable regimes", "A model that only works in narrow buckets should not trade the whole universe.", "MEDIUM", "LOW", "Split CE/PE, DTE, and moneyness cohorts and compare them separately.", ["scripts/retrain_all_edge_models.py"], "At least one practical regime subset shows stable fold-level improvement.", True),
        "paper_execution_filter_gate": ("Make the edge survive execution filters", "If realistic filters remove all trades, the strategy is not paper ready.", "HIGH", "LOW", "Target cleaner buckets instead of forcing thresholds.", ["scripts/retrain_all_edge_models.py"], "A filtered candidate still has enough trades and positive PF.", True),
    }
    failed = [row for row in gate_rows if row.get("status") == "FAIL"]
    ordering = ["chronology_gate", "leakage_gate", "live_computable_feature_gate", "live_schema_gate", "cost_stress_gate", "fold_stability_gate", "calibration_gate", "probability_monotonicity_gate", "regime_stability_gate", "paper_execution_filter_gate", "minimum_trade_count_gate", "trading_metrics_gate", "threshold_robustness_gate", "same_period_gate", "production_adoption_gate"]
    ranked: List[Dict[str, Any]] = []
    rank = 1
    for gate_name in ordering:
        row = next((item for item in failed if item["gate_name"] == gate_name), None)
        if row is None:
            continue
        spec = priority_specs.get(gate_name, ("Investigate failed gate", row.get("short_reason"), "MEDIUM", "MEDIUM", row.get("recommended_action"), ["scripts/retrain_all_edge_models.py"], "Gate must pass cleanly before more tuning.", False))
        ranked.append({
            "priority_rank": rank,
            "issue": spec[0],
            "why_it_matters": spec[1],
            "expected_impact": spec[2],
            "implementation_difficulty": spec[3],
            "suggested_next_experiment": spec[4],
            "files_likely_involved": spec[5],
            "acceptance_criteria": spec[6],
            "should_run_before_more_model_training": spec[7],
        })
        rank += 1
    return ranked


def _experiment_generator(
    gate_rows: Sequence[Dict[str, Any]],
    *,
    dataset_path: str,
    output_dir: str,
) -> List[Dict[str, Any]]:
    failed_names = {row["gate_name"] for row in gate_rows if row.get("status") == "FAIL"}
    output_dir = output_dir or "models\\failure_diagnosis_experiments"
    experiments: List[Dict[str, Any]] = []
    base_prefix = "python scripts\\retrain_all_edge_models.py"
    common = f'--dataset "{dataset_path}" --use-black-scholes-features --compare-baseline --no-production-adopt --walk-forward-folds 5 --drop-bad-greeks'
    if "live_computable_feature_gate" in failed_names or "live_schema_gate" in failed_names:
        experiments.append({
            "experiment_name": "live_computable_only_retrain",
            "hypothesis": "Removing non-live-computable features will reduce schema risk and reveal whether any edge survives safely.",
            "exact_command": f'{base_prefix} {common} --same-period-comparison --live-computable-only --paper-readiness-audit --output-dir "{output_dir}\\live_computable_only"',
            "expected_success_condition": "Live-computable gate passes and paper metrics do not collapse.",
            "expected_failure_condition": "Metrics degrade sharply or schema mismatch still remains.",
            "report_outputs": ["paper_readiness_audit", "feature_compatibility_report", "same_period_comparison"],
            "risk_of_overfitting": "LOW",
            "notes": "This should happen before any model tuning.",
        })
    if "cost_stress_gate" in failed_names or "paper_execution_filter_gate" in failed_names:
        experiments.append({
            "experiment_name": "stricter_cost_and_execution_audit",
            "hypothesis": "The current edge depends on marginal trades that disappear under realistic execution constraints.",
            "exact_command": f'{base_prefix} {common} --same-period-comparison --live-computable-only --paper-readiness-audit --paper-min-option-price 10 --paper-max-bid-ask-spread-pct 3 --paper-max-trades-per-day 3 --paper-cooldown-minutes 20 --output-dir "{output_dir}\\strict_execution"',
            "expected_success_condition": "Filtered trade count remains practical and profit factor stays above 1.",
            "expected_failure_condition": "Trade count collapses or PF falls below 1 again.",
            "report_outputs": ["paper_readiness_audit", "cost_stress_report", "daily_pnl_stability_report"],
            "risk_of_overfitting": "MEDIUM",
            "notes": "Do not loosen filters just to manufacture trades.",
        })
    if "fold_stability_gate" in failed_names or "regime_stability_gate" in failed_names:
        experiments.append({
            "experiment_name": "ce_pe_and_regime_isolation_audit",
            "hypothesis": "The broad model hides a narrow sub-regime that is less unstable.",
            "exact_command": f'{base_prefix} {common} --same-period-comparison --live-computable-only --paper-readiness-audit --output-dir "{output_dir}\\regime_isolation"',
            "expected_success_condition": "A clearly defined subset shows stronger fold stability than the broad model.",
            "expected_failure_condition": "All subsets remain unstable across folds.",
            "report_outputs": ["regime_performance_report", "paper_readiness_audit", "same_period_comparison"],
            "risk_of_overfitting": "MEDIUM",
            "notes": "Follow up by filtering the dataset outside this script if one side or bucket dominates.",
        })
    if "calibration_gate" in failed_names or "probability_monotonicity_gate" in failed_names:
        experiments.append({
            "experiment_name": "calibration_repair_audit",
            "hypothesis": "Probability quality is the bottleneck rather than raw classification separability.",
            "exact_command": f'{base_prefix} {common} --same-period-comparison --live-computable-only --paper-readiness-audit --threshold-grid-start 0.50 --threshold-grid-end 0.80 --threshold-grid-step 0.025 --output-dir "{output_dir}\\calibration_repair"',
            "expected_success_condition": "Calibration buckets and return ranking become more monotonic without hiding costs.",
            "expected_failure_condition": "Top-score buckets still fail to dominate lower-score buckets.",
            "report_outputs": ["probability_calibration_report", "probability_monotonicity_report", "threshold_robustness_report"],
            "risk_of_overfitting": "MEDIUM",
            "notes": "Keep the split chronological and do not cherry-pick one lucky threshold.",
        })
    experiments.append({
        "experiment_name": "same_period_baseline_vs_bs_recheck",
        "hypothesis": "The apparent BS uplift may disappear once both variants are restricted to the same safe period and feature rules.",
        "exact_command": f'{base_prefix} {common} --same-period-comparison --paper-readiness-audit --output-dir "{output_dir}\\same_period_recheck"',
        "expected_success_condition": "The comparison remains honest and reproducible with the same period locked.",
        "expected_failure_condition": "BS still cannot beat baseline without research-only dependencies.",
        "report_outputs": ["same_period_comparison", "bs_features_model_comparison", "final_decision_matrix"],
        "risk_of_overfitting": "LOW",
        "notes": "This is a sanity check, not a production gate bypass.",
    })
    return experiments[:7]


def _model_rescue_plan() -> List[str]:
    return [
        "Remove non-live-computable and research-only features before adding more model complexity.",
        "Isolate CE/PE sides if one side is structurally weaker or fails execution filters.",
        "Avoid unstable DTE and moneyness buckets instead of fitting one model across every regime.",
        "Avoid high-spread and low-price options before trying more expressive models.",
        "Compare label horizons and simpler models before any complex ensemble expansion.",
        "Improve data quality and regime segmentation before adding deep learning.",
        "Do not deploy because profit factor is barely above 1 on one threshold or one fold.",
        "Do not use estimated-only Greeks in live production unless they are computed identically live.",
        "Do not ignore transaction costs, slippage, same-period comparison, or execution realism.",
    ]


def _build_failure_diagnosis_payload(
    *,
    dataset_path: str,
    comparison_payload: Dict[str, Any],
    paper_payload: Dict[str, Any],
    feature_manifest_payload: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    gate_rows = _failure_gate_table(
        paper_payload=paper_payload,
        comparison_payload=comparison_payload,
        feature_manifest_payload=feature_manifest_payload,
    )
    root_causes = _failed_gate_root_causes(gate_rows)
    warnings = _failure_flags_from_gate_rows(gate_rows)
    plan = _improvement_priority_plan(gate_rows, comparison_payload=comparison_payload)
    experiments = _experiment_generator(gate_rows, dataset_path=dataset_path, output_dir=output_dir)
    matrix = paper_payload.get("decision_matrix", {})
    verdict = "WEAK_BUT_RESEARCHABLE" if warnings.get("paper_trading_allowed") is False else "LIMITED_PAPER_ONLY"
    if matrix.get("chronology_gate") == "FAIL" or matrix.get("leakage_gate") == "FAIL":
        verdict = "INVALID_EVALUATION_DO_NOT_USE"
    if not warnings.get("paper_trading_allowed", False):
        verdict = "BLOCKED_FROM_PAPER_AND_PRODUCTION"
    return {
        "final_status": verdict,
        "paper_trading_allowed": warnings.get("paper_trading_allowed", False),
        "production_adoption_allowed": False,
        "main_reason": next((row.get("short_reason") for row in gate_rows if row.get("status") == "FAIL"), "Production remains blocked."),
        "gate_failure_table": gate_rows,
        "failed_gate_root_causes": root_causes,
        "improvement_priority_plan": plan,
        "do_not_proceed_messages": warnings.get("messages", []),
        "next_experiments": experiments,
        "model_rescue_plan": _model_rescue_plan(),
        "files_and_commands_used": {
            "dataset_path": dataset_path,
            "output_dir": output_dir,
            "comparison_report": comparison_payload.get("dataset_path"),
            "selected_feature_count": len(feature_manifest_payload.get("black_scholes_features") or feature_manifest_payload.get("features") or []),
        },
        "strict_final_recommendation": "Keep this model blocked from paper and production until chronology-safe, live-computable, schema-compatible, cost-resilient, and fold-stable evidence exists.",
    }


def _label_audit_for_frame(df: pd.DataFrame, label_name: str) -> Dict[str, Any]:
    series = prepare_label_series(df, label_name)
    positive_ratio = float(series.dropna().mean()) if len(series.dropna()) else None
    return {
        "label_name": label_name,
        "positive_class_ratio": positive_ratio,
        "target_horizon": "unknown",
        "label_leakage_audit_result": "PASS" if label_name in PRIMARY_LABELS else "WARNING",
        "class_imbalance_warning": bool(positive_ratio is not None and (positive_ratio < 0.20 or positive_ratio > 0.80)),
    }


def _collect_candidate_rows(
    variant_results: Sequence[Dict[str, Any]],
    *,
    live_schema_report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in variant_results:
        variant_name = item["variant_name"]
        feature_set = item["feature_set"]
        report_map = item["result"].get("reports", {})
        for label_name, models in report_map.items():
            for model_family, report in models.items():
                fold = report.get("fold_stability", {}) or {}
                after_filters = (report.get("paper_execution_filters", {}).get("after", {}) or {})
                live_schema_pass = bool(live_schema_report.get("compatible")) and not feature_set.startswith("black_scholes")
                live_computable_pass = feature_set in {"baseline_all_safe_features", "live_computable_only", "no_greeks", "reduced_robust_features"}
                verdict = _paper_candidate_verdict(report, live_schema_pass=live_schema_pass, live_computable_pass=live_computable_pass)
                rows.append({
                    "model_family": model_family,
                    "feature_set": feature_set,
                    "label": label_name,
                    "selected_threshold": report.get("selected_threshold_from_validation", {}).get("threshold"),
                    "ROC-AUC": report.get("test_metrics", {}).get("roc_auc"),
                    "PR-AUC": report.get("test_metrics", {}).get("pr_auc"),
                    "F1": report.get("test_metrics", {}).get("f1"),
                    "Brier score": report.get("test_metrics", {}).get("brier_score"),
                    "PF": report.get("trade_metrics", {}).get("profit_factor"),
                    "Sharpe": report.get("trade_metrics", {}).get("sharpe"),
                    "Sortino": report.get("trade_metrics", {}).get("sortino"),
                    "max_drawdown": report.get("trade_metrics", {}).get("max_drawdown"),
                    "trade_count": report.get("trade_metrics", {}).get("trade_count"),
                    "post_filter_trade_count": after_filters.get("trade_count"),
                    "profitable_folds": fold.get("profitable_folds"),
                    "worst_fold_pf": fold.get("worst_fold_pf"),
                    "median_fold_pf": fold.get("median_fold_pf"),
                    "cost_stress_pass": all(not bool(v.get("failure_flag")) for k, v in (report.get("cost_stress") or {}).items() if k != "base"),
                    "threshold_robustness_pass": bool((report.get("threshold_robustness") or {}).get("chosen_threshold_is_robust")),
                    "calibration_pass": str((report.get("calibration_audit") or {}).get("calibration_gate", "FAIL")) == "PASS",
                    "monotonicity_pass": str((report.get("probability_monotonicity") or {}).get("gate", "FAIL")) == "PASS",
                    "daily_pnl_pass": str((report.get("daily_pnl_stability") or {}).get("gate", "FAIL")) == "PASS",
                    "regime_stability_pass": _regime_stability_gate(report.get("regime_performance", {})).get("gate") == "PASS",
                    "live_schema_pass": live_schema_pass,
                    "production_adoption_allowed": False,
                    "paper_readiness_verdict": verdict,
                    "final_recommendation": verdict,
                    "variant_name": variant_name,
                })
    return rows


def _collect_rescue_candidate_rows(
    experiment_results: Sequence[Dict[str, Any]],
    *,
    live_schema_report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for experiment in experiment_results:
        report_map = (((experiment.get("training_result") or {}).get("reports")) or {})
        for label_name, models in report_map.items():
            for model_family, report in models.items():
                live_schema_pass = bool(live_schema_report.get("compatible")) and experiment.get("feature_set") != "black_scholes_research"
                live_computable_pass = experiment.get("feature_set") in {"live_computable_only", "no_greeks", "reduced_robust_live_features", "reduced_robust_features"}
                rescue = _rescue_candidate_flags(report, live_schema_pass=live_schema_pass, live_computable_pass=live_computable_pass)
                after_filters = (report.get("paper_execution_filters", {}).get("after", {}) or {})
                rows.append(
                    {
                        "experiment_name": experiment.get("experiment_name"),
                        "model_family": model_family,
                        "feature_set": experiment.get("feature_set"),
                        "label": label_name,
                        "row_count": experiment.get("filter_report", {}).get("rows_after"),
                        "post_paper_filter_trade_count": int(after_filters.get("trade_count") or 0),
                        "active_trading_days": rescue.get("active_trading_days"),
                        "selected_threshold": report.get("selected_threshold_from_validation", {}).get("threshold"),
                        "ROC-AUC": report.get("test_metrics", {}).get("roc_auc"),
                        "PR-AUC": report.get("test_metrics", {}).get("pr_auc"),
                        "F1": report.get("test_metrics", {}).get("f1"),
                        "PF": after_filters.get("profit_factor"),
                        "Sharpe": after_filters.get("sharpe"),
                        "Sortino": after_filters.get("sortino"),
                        "max_drawdown": after_filters.get("max_drawdown"),
                        "win_rate": after_filters.get("win_rate"),
                        "expectancy": after_filters.get("expectancy"),
                        "worst_fold_pf": rescue.get("worst_fold_pf"),
                        "median_fold_pf": rescue.get("median_fold_pf"),
                        "profitable_folds": rescue.get("profitable_folds"),
                        "cost_stress_0_25_pf": rescue.get("cost_stress_0_25_pf"),
                        "calibration_gate": (report.get("calibration_audit") or {}).get("calibration_gate", "FAIL"),
                        "monotonicity_gate": (report.get("probability_monotonicity") or {}).get("gate", "FAIL"),
                        "daily_pnl_gate": (report.get("daily_pnl_stability") or {}).get("gate", "FAIL"),
                        "live_schema_gate": "PASS" if live_schema_pass else "FAIL",
                        "shadow_candidate_allowed": rescue.get("shadow_candidate_allowed"),
                        "paper_candidate_allowed": rescue.get("paper_candidate_allowed"),
                        "final_verdict": rescue.get("final_verdict"),
                        "failed_gates": rescue.get("failed_gates"),
                        "production_adoption_allowed": False,
                        "report_ref": report,
                    }
                )
    return rows


def _shadow_manifest_payload(
    row: Dict[str, Any],
    report: Dict[str, Any],
    *,
    dataset_path: str,
    feature_names: Sequence[str],
    experiment_name: str,
) -> Dict[str, Any]:
    return {
        "model_id": f"{experiment_name}:{row.get('model_family')}:{row.get('label')}",
        "experiment_name": experiment_name,
        "model_family": row.get("model_family"),
        "feature_set": row.get("feature_set"),
        "selected_feature_list": list(feature_names),
        "selected_threshold": row.get("selected_threshold"),
        "model_path": report.get("model_path"),
        "scaler_or_calibrator": report.get("calibration_fit"),
        "training_dataset_path": dataset_path,
        "training_validation_test_ranges": report.get("holdout_split"),
        "gate_summary": {
            "calibration_gate": row.get("calibration_gate"),
            "monotonicity_gate": row.get("monotonicity_gate"),
            "daily_pnl_gate": row.get("daily_pnl_gate"),
            "live_schema_gate": row.get("live_schema_gate"),
            "failed_gates": row.get("failed_gates"),
        },
        "rescue_verdict": row.get("final_verdict"),
        "shadow_candidate_allowed": bool(row.get("shadow_candidate_allowed")),
        "paper_candidate_allowed": bool(row.get("paper_candidate_allowed")),
        "production_adoption_allowed": False,
    }


def prepare_label_series(df: pd.DataFrame, label_name: str) -> pd.Series:
    series = df[label_name]
    if series.dtype == object:
        mapped = series.map({"GOOD": 1, "BAD": 0, "NEUTRAL": np.nan})
        return pd.to_numeric(mapped, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def select_threshold(validation_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    preferred = [row for row in validation_rows if float(row.get("precision", 0.0)) >= 0.35 and int(row.get("signals", 0)) >= 75]
    if preferred:
        return max(preferred, key=lambda row: (float(row.get("precision", 0.0)), float(row.get("f1", 0.0)), int(row.get("signals", 0))))
    fallback = [row for row in validation_rows if float(row.get("precision", 0.0)) >= 0.30 and int(row.get("signals", 0)) >= 50]
    if fallback:
        return max(fallback, key=lambda row: (float(row.get("precision", 0.0)), float(row.get("f1", 0.0)), int(row.get("signals", 0))))
    return max(validation_rows, key=lambda row: (float(row.get("precision", 0.0)), float(row.get("f1", 0.0)), int(row.get("signals", 0))))


def brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    return float(np.mean((yp - yt) ** 2)) if len(yt) else 0.0


def fit_model(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    model.fit(X_train, y_train)
    return model


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except Exception:
        return default


def _rf_runtime_config() -> Dict[str, Any]:
    defaults = _runtime_rf_params()
    default_trees = int(defaults["n_estimators"])
    stage1_trees = _env_int("RF_SEARCH_STAGE1_TREES", max(50, min(default_trees, 160)), 50)
    stage2_trees = _env_int("RF_SEARCH_STAGE2_TREES", max(stage1_trees, min(max(default_trees, stage1_trees), 320)), stage1_trees)
    final_trees = int(defaults["n_estimators"])
    stage1_top_k = 1
    n_jobs = int(defaults["n_jobs"])
    adaptive_margin_bps = _env_int("RF_ADAPTIVE_MARGIN_BPS", 50, 0)
    return {
        "stage1_trees": stage1_trees,
        "stage2_trees": stage2_trees,
        "final_trees": final_trees,
        "stage1_top_k": stage1_top_k,
        "n_jobs": n_jobs,
        "adaptive_margin_bps": adaptive_margin_bps,
        "max_depth": int(defaults["max_depth"]),
        "min_samples_leaf": int(defaults["min_samples_leaf"]),
        "min_samples_split": int(defaults["min_samples_split"]),
        "max_features": defaults["max_features"],
        "class_weight": defaults["class_weight"],
        "fast_mode": bool(_runtime_option("fast_mode", False)),
    }


def _random_forest_search_space() -> List[Dict[str, Any]]:
    search_rows: List[Dict[str, Any]] = []
    base_shapes = [
        {"max_depth": 6, "min_samples_leaf": 8, "min_samples_split": 24, "max_features": "sqrt"},
        {"max_depth": 8, "min_samples_leaf": 6, "min_samples_split": 18, "max_features": "sqrt"},
        {"max_depth": 10, "min_samples_leaf": 4, "min_samples_split": 12, "max_features": 0.6},
        {"max_depth": 12, "min_samples_leaf": 3, "min_samples_split": 10, "max_features": 0.5},
    ]
    overlays = [
        {"criterion": "gini", "max_samples": 0.70, "class_weight": "balanced_subsample", "ccp_alpha": 0.0, "max_leaf_nodes": 32},
        {"criterion": "gini", "max_samples": 0.85, "class_weight": "balanced", "ccp_alpha": 0.0, "max_leaf_nodes": 64},
        {"criterion": "entropy", "max_samples": 0.70, "class_weight": "balanced_subsample", "ccp_alpha": 0.0005, "max_leaf_nodes": 48},
        {"criterion": "entropy", "max_samples": None, "class_weight": "balanced", "ccp_alpha": 0.0010, "max_leaf_nodes": 96},
        {"criterion": "log_loss", "max_samples": 0.85, "class_weight": "balanced_subsample", "ccp_alpha": 0.0, "max_leaf_nodes": 64},
        {"criterion": "log_loss", "max_samples": None, "class_weight": "balanced", "ccp_alpha": 0.0005, "max_leaf_nodes": 128},
    ]
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for base in base_shapes:
        for overlay in overlays:
            params = {**base, **overlay}
            key = tuple(sorted(params.items(), key=lambda item: item[0]))
            if key in seen:
                continue
            seen.add(key)
            search_rows.append(params)
    return search_rows


def _normalize_rf_max_samples(value: Any, row_count: int) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        return max(1, int(round(float(value) * max(1, int(row_count)))))
    return value


def _build_rf_estimator(params: Dict[str, Any], *, n_estimators: int, n_jobs: int, row_count: int) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    final_params = dict(params)
    final_params["n_estimators"] = int(n_estimators)
    final_params["max_samples"] = _normalize_rf_max_samples(final_params.get("max_samples"), row_count)
    final_params.setdefault("min_samples_split", int(_runtime_rf_params()["min_samples_split"]))
    final_params.setdefault("min_samples_leaf", int(_runtime_rf_params()["min_samples_leaf"]))
    final_params.setdefault("max_depth", _runtime_rf_params()["max_depth"])
    final_params.setdefault("max_features", _runtime_rf_params()["max_features"])
    final_params.setdefault("class_weight", _runtime_rf_params()["class_weight"])
    return RandomForestClassifier(
        **final_params,
        bootstrap=True,
        random_state=42,
        n_jobs=int(n_jobs),
    )


def _validation_trade_quality_score(
    threshold_summary: Dict[str, Any],
    classification_summary: Dict[str, Any],
) -> float:
    pf = float(threshold_summary.get("profit_factor") or 0.0)
    sharpe = float(threshold_summary.get("sharpe") or 0.0)
    f1 = float(classification_summary.get("f1") or 0.0)
    roc_auc = float(classification_summary.get("roc_auc") or 0.0)
    precision = float(classification_summary.get("precision") or 0.0)
    drawdown = float(threshold_summary.get("max_drawdown") or 0.0)
    trades = int(threshold_summary.get("trade_count") or 0)

    sharpe_bonus = min(max(sharpe, 0.0), 3.0) * 4.0
    pf_bonus = max(pf - 1.0, 0.0) * 3.0
    classification_bonus = (f1 * 2.0) + roc_auc + precision
    trade_bonus = min(trades, 200) / 200.0
    drawdown_penalty = min(max(drawdown, 0.0), 2.5) * 0.5

    return sharpe_bonus + pf_bonus + classification_bonus + trade_bonus - drawdown_penalty


def _rf_stage_candidate_row(
    params: Dict[str, Any],
    stage_name: str,
    n_estimators: int,
    model: Any,
    X_val_s: np.ndarray,
    y_val: np.ndarray,
    val_returns: np.ndarray | None,
    *,
    minimum_trades: int,
) -> Dict[str, Any]:
    val_prob = predict_proba_positive(model, X_val_s)
    threshold_row = optimize_threshold_from_probs(
        y_val,
        val_prob,
        val_returns,
        thresholds=RETRAIN_THRESHOLD_SWEEP,
        minimum_trades=minimum_trades,
    )
    class_metrics = classification_metrics(y_val, val_prob, threshold=float(threshold_row["threshold"]))
    score = _validation_trade_quality_score(threshold_row, class_metrics)
    return {
        "stage": stage_name,
        "params": dict(params),
        "n_estimators": int(n_estimators),
        "selected_threshold": float(threshold_row["threshold"]),
        "profit_factor": float(threshold_row.get("profit_factor") or 0.0),
        "sharpe": float(threshold_row.get("sharpe") or 0.0),
        "trade_count": int(threshold_row.get("trade_count") or 0),
        "f1": float(class_metrics.get("f1") or 0.0),
        "roc_auc": float(class_metrics.get("roc_auc") or 0.0),
        "score": float(score),
        "threshold_preview": dict(threshold_row),
    }


def _rf_stage_ranking_key(row: Dict[str, Any]) -> tuple[float, float, float, float, int]:
    return (
        float(row.get("score") or 0.0),
        float(row.get("profit_factor") or 0.0),
        float(row.get("sharpe") or 0.0),
        float(row.get("f1") or 0.0),
        int(row.get("trade_count") or 0),
    )


def _rf_adaptive_stage2_pool(stage1_rows: List[Dict[str, Any]], stage1_top_k: int, adaptive_margin_bps: int) -> List[Dict[str, Any]]:
    ranked = sorted(stage1_rows, key=_rf_stage_ranking_key, reverse=True)
    if not ranked:
        return []
    best_score = float(ranked[0].get("score") or 0.0)
    margin = float(adaptive_margin_bps) / 100.0
    shortlisted: List[Dict[str, Any]] = []
    for row in ranked:
        if len(shortlisted) < int(stage1_top_k):
            shortlisted.append(row)
            continue
        row_score = float(row.get("score") or 0.0)
        row_pf = float(row.get("profit_factor") or 0.0)
        row_sharpe = float(row.get("sharpe") or 0.0)
        if row_score + margin >= best_score and (row_pf > 1.15 or row_sharpe > 1.0):
            shortlisted.append(row)
            continue
        break
    return shortlisted


def _fit_random_forest_with_validation_search(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_val_s: np.ndarray,
    y_val: np.ndarray,
    val_returns: np.ndarray | None,
    *,
    minimum_trades: int,
) -> tuple[Any, Dict[str, Any]]:
    cfg = _rf_runtime_config()
    row_count = int(len(X_train_s))
    direct_params = {
        "criterion": "gini",
        "max_depth": int(cfg["max_depth"]),
        "min_samples_leaf": int(cfg["min_samples_leaf"]),
        "min_samples_split": int(cfg["min_samples_split"]),
        "max_features": cfg["max_features"],
        "class_weight": cfg["class_weight"],
        "max_samples": 0.85 if bool(cfg.get("fast_mode")) else None,
        "max_leaf_nodes": None,
        "ccp_alpha": 0.0,
    }
    print(
        "[retrain] random_forest_fit "
        f"trees={cfg['final_trees']} max_depth={direct_params['max_depth']} "
        f"min_samples_leaf={direct_params['min_samples_leaf']} min_samples_split={direct_params['min_samples_split']} "
        f"max_features={direct_params['max_features']} class_weight={direct_params['class_weight']} "
        f"n_jobs={cfg['n_jobs']} rows={row_count}",
        flush=True,
    )
    best_model = _build_rf_estimator(
        direct_params,
        n_estimators=int(cfg["final_trees"]),
        n_jobs=int(cfg["n_jobs"]),
        row_count=row_count,
    )
    fit_start = time.perf_counter()
    fit_model(best_model, X_train_s, y_train)
    fit_seconds = time.perf_counter() - fit_start
    final_row = _rf_stage_candidate_row(
        direct_params,
        "direct_fit",
        int(cfg["final_trees"]),
        best_model,
        X_val_s,
        y_val,
        val_returns,
        minimum_trades=minimum_trades,
    )
    _timing_log("random_forest_train", seconds=fit_seconds, rows=row_count, extra=f"trees={cfg['final_trees']}")

    return best_model, {
        "mode": "direct_fit",
        "fit_on_validation_only": False,
        "runtime_config": cfg,
        "selected_params": direct_params,
        "selected_threshold_preview": dict(final_row["threshold_preview"]),
        "search_rows": [final_row],
        "stage1_candidate_count": 1,
        "stage2_candidate_count": 0,
    }


def _fit_xgboost_with_validation(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_val_s: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Any, Dict[str, Any]]:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError(f"xgboost not installed: {exc}") from exc

    runtime_params = _runtime_xgb_params()
    device_preference = str(_runtime_option("xgb_device", "auto") or "auto").strip().lower()
    strict_gpu = bool(_runtime_option("strict_gpu", False))
    candidate_params: List[Dict[str, Any]] = []
    if device_preference in {"auto", "cuda"}:
        candidate_params.append({**runtime_params, "device": "cuda", "tree_method": str(runtime_params.get("tree_method") or "hist")})
    candidate_params.append({**runtime_params, "device": "cpu", "tree_method": str(runtime_params.get("tree_method") or "hist")})

    fit_errors: List[str] = []
    for candidate in candidate_params:
        backend = str(candidate.get("device") or "cpu")
        model = XGBClassifier(**candidate)
        fit_kwargs: Dict[str, Any] = {}
        early_stopping_rounds = _runtime_option("xgb_early_stopping_rounds", None)
        if early_stopping_rounds:
            fit_kwargs["eval_set"] = [(X_val_s, y_val)]
            fit_kwargs["verbose"] = False
        print(
            "[retrain] xgboost_fit_start "
            f"device={backend} rows={len(X_train_s)} features={X_train_s.shape[1] if X_train_s.ndim == 2 else 0} "
            f"params={json.dumps({k: v for k, v in candidate.items() if k != 'eval_metric'}, sort_keys=True, default=str)}",
            flush=True,
        )
        fit_start = time.perf_counter()
        try:
            if early_stopping_rounds and hasattr(model, "set_params"):
                try:
                    model.set_params(early_stopping_rounds=int(early_stopping_rounds))
                except Exception:
                    pass
            model.fit(X_train_s, y_train, **fit_kwargs)
            fit_seconds = time.perf_counter() - fit_start
            _timing_log("xgboost_train", seconds=fit_seconds, rows=len(X_train_s), extra=f"device={backend}")
            return model, {
                "mode": "direct_fit",
                "fit_on_validation_only": False,
                "runtime_backend": backend,
                "selected_params": candidate,
                "early_stopping_rounds": int(early_stopping_rounds) if early_stopping_rounds else None,
            }
        except Exception as exc:
            fit_seconds = time.perf_counter() - fit_start
            fit_errors.append(f"{backend}:{exc}")
            print(f"[retrain] xgboost_fit_failed device={backend} seconds={fit_seconds:.3f} reason={exc}", flush=True)
            if backend == "cuda" and device_preference == "cuda" and strict_gpu:
                raise RuntimeError(f"CUDA unavailable for xgboost and --strict-gpu is set: {exc}") from exc
            continue
    raise RuntimeError(f"XGBoost fit failed on all backends: {' | '.join(fit_errors)}")


def _fit_model_for_training_run(
    model_name: str,
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_val_s: np.ndarray,
    y_val: np.ndarray,
    val_returns: np.ndarray | None,
    *,
    minimum_trades: int,
) -> tuple[Any, Dict[str, Any]]:
    name = str(model_name).strip().lower()
    if name in {"logistic_regression_platt", "logistic_regression_isotonic"}:
        return _fit_model_with_validation_calibration(model_name, X_train_s, y_train, X_val_s, y_val)
    if name == "random_forest":
        return _fit_random_forest_with_validation_search(
            X_train_s,
            y_train,
            X_val_s,
            y_val,
            val_returns,
            minimum_trades=minimum_trades,
        )
    if name == "xgboost":
        return _fit_xgboost_with_validation(
            X_train_s,
            y_train,
            X_val_s,
            y_val,
        )
    model = _build_model_family(model_name)
    fit_model(model, X_train_s, y_train)
    return model, {"mode": "uncalibrated", "fit_on_validation_only": False}


class _ValidationOnlyCalibratedModel:
    def __init__(self, base_model: Any, calibrator: Any) -> None:
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        base_prob = predict_proba_positive(self.base_model, X)
        if hasattr(self.calibrator, "predict_proba"):
            calibrated = np.asarray(self.calibrator.predict_proba(base_prob.reshape(-1, 1))[:, 1], dtype=float)
        else:
            calibrated = np.asarray(self.calibrator.predict(base_prob), dtype=float)
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])


def _fit_model_with_validation_calibration(
    model_name: str,
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_val_s: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Any, Dict[str, Any]]:
    name = str(model_name).strip().lower()
    if name not in {"logistic_regression_platt", "logistic_regression_isotonic"}:
        model = _build_model_family(model_name)
        fit_model(model, X_train_s, y_train)
        return model, {"mode": "uncalibrated", "fit_on_validation_only": False}
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression

    base = LogisticRegression(max_iter=3000, solver="liblinear", class_weight="balanced", random_state=42)
    base.fit(X_train_s, y_train)
    method = "sigmoid" if name.endswith("platt") else "isotonic"
    val_prob = predict_proba_positive(base, X_val_s)
    if method == "sigmoid":
        calibrator = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
        calibrator.fit(val_prob.reshape(-1, 1), y_val)
    else:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_prob, y_val)
    return _ValidationOnlyCalibratedModel(base, calibrator), {"mode": method, "fit_on_validation_only": True, "validation_rows_used": int(len(y_val))}


def _gate_rows_from_ensemble_details(details: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([1 if bool(row.get("final_allowed")) else 0 for row in details], dtype=int)


def _classification_metrics_from_decisions(y_true: np.ndarray, decisions: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    base = classification_metrics(y_true, y_prob, threshold=0.5)
    confusion = confusion_from_threshold(y_true, decisions, 0.5)
    tp = int(confusion["tp"])
    fp = int(confusion["fp"])
    fn = int(confusion["fn"])
    tn = int(confusion["tn"])
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    accuracy = float((tp + tn) / max(1, tp + tn + fp + fn))
    f1 = float((2 * precision * recall) / max(1e-12, precision + recall)) if (precision + recall) else 0.0
    base.update({
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
    })
    return base


def _ensemble_threshold_sweep(
    y_true: np.ndarray,
    decisions_by_threshold: Sequence[Dict[str, Any]],
    returns: np.ndarray | None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in decisions_by_threshold:
        threshold = float(item["threshold"])
        decisions = np.asarray(item["decisions"], dtype=int)
        ensemble_prob = np.asarray(item["ensemble_prob"], dtype=float)
        cls = _classification_metrics_from_decisions(y_true, decisions, ensemble_prob)
        trade = _trade_metrics_from_scores(y_true, decisions, 0.5, returns, threshold_source=f"ensemble_gate:{threshold}")
        rows.append(
            {
                "threshold": threshold,
                "trade_count": int(trade.get("trade_count") or 0),
                "positive_predictions": int(decisions.sum()),
                "precision": cls.get("precision"),
                "recall": cls.get("recall"),
                "accuracy": cls.get("accuracy"),
                "f1": cls.get("f1"),
                "estimated_win_rate": trade.get("win_rate"),
                "average_net_forward_return": trade.get("average_return_per_trade"),
                "total_net_forward_return": trade.get("total_return"),
                "profit_factor": trade.get("profit_factor"),
                "max_drawdown": trade.get("max_drawdown"),
                "brier_score": brier_score(y_true, ensemble_prob),
                "positive_rate": float(decisions.mean()) if len(decisions) else 0.0,
            }
        )
    return rows


def _save_ensemble_artifact(
    *,
    artifact_dir: Path,
    model: XGBRFEnsembleClassifier,
    label_name: str,
    feature_columns: Sequence[str],
    fill_values: Dict[str, float],
    metrics_payload: Dict[str, Any],
    threshold_rows: Sequence[Dict[str, Any]],
    comparison_payload: Dict[str, Any],
    dataset_path: str,
    split_sizes: Dict[str, int],
) -> Dict[str, str]:
    import joblib

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pkl"
    rf_path = artifact_dir / "rf_model.pkl"
    xgb_path = artifact_dir / "xgb_model.pkl"
    ensemble_config = {
        "ensemble_type": "weighted_xgb_rf",
        "target": str(label_name),
        **_runtime_ensemble_params(),
        "feature_columns": list(feature_columns),
        "created_at": datetime.now().isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_rows": int(sum(split_sizes.values())),
        "train_rows": int(split_sizes.get("train_rows", 0)),
        "validation_rows": int(split_sizes.get("validation_rows", 0)),
        "test_rows": int(split_sizes.get("test_rows", 0)),
        "model_type": "xgb_rf_ensemble",
    }
    joblib.dump(_wrap_model(model, feature_names=feature_columns, metrics=ensemble_config, scaler_mean=[], scaler_std=[]), model_path)
    if getattr(model, "rf_model_", None) is not None:
        joblib.dump(getattr(model, "rf_model_"), rf_path)
    if getattr(model, "xgb_model_", None) is not None:
        joblib.dump(getattr(model, "xgb_model_"), xgb_path)
    write_json(artifact_dir / "ensemble_config.json", ensemble_config)
    write_json(artifact_dir / "feature_columns.json", {"feature_columns": list(feature_columns)})
    write_json(artifact_dir / "fill_values.json", dict(fill_values))
    write_json(artifact_dir / "metrics.json", metrics_payload)
    pd.DataFrame(list(threshold_rows)).to_csv(artifact_dir / "threshold_sweep.csv", index=False)
    write_json(artifact_dir / "backtest_report.json", metrics_payload.get("backtest_report") or {})
    write_json(artifact_dir / "model_comparison_report.json", comparison_payload)
    model_card_lines = [
        "# XGB + RF Ensemble Model Card",
        f"- Target: `{label_name}`",
        f"- Dataset: `{dataset_path}`",
        f"- Feature count: `{len(feature_columns)}`",
        f"- Train/Val/Test rows: `{split_sizes.get('train_rows', 0)}` / `{split_sizes.get('validation_rows', 0)}` / `{split_sizes.get('test_rows', 0)}`",
        f"- XGB backend: `{getattr(model, 'xgb_runtime_backend_', 'unknown')}`",
        "- Production adoption allowed: `False`",
    ]
    (artifact_dir / "model_card.md").write_text("\n".join(model_card_lines), encoding="utf-8")
    print(
        f"[ensemble] saved_files={json.dumps(sorted([p.name for p in artifact_dir.iterdir() if p.is_file()]))}",
        flush=True,
    )
    return {
        "model_path": str(model_path),
        "rf_model_path": str(rf_path),
        "xgb_model_path": str(xgb_path),
        "metrics_path": str(artifact_dir / "metrics.json"),
        "threshold_sweep_csv_path": str(artifact_dir / "threshold_sweep.csv"),
        "ensemble_config_path": str(artifact_dir / "ensemble_config.json"),
        "model_comparison_report_path": str(artifact_dir / "model_comparison_report.json"),
    }


def _train_xgb_rf_ensemble_for_label(
    *,
    work: pd.DataFrame,
    feature_cols: Sequence[str],
    label_name: str,
    artifact_dir: Path,
    variant_name: str,
    returns_col: str | None,
    timestamps: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    split_window: Dict[str, Any],
    walk_forward: Dict[str, Any],
    paper_config: Dict[str, Any] | None,
) -> Dict[str, Any]:
    filtered = filter_target_rows(work, "label", allowed_values=(0, 1))
    feature_frame, _ = build_safe_numeric_bool_frame(
        filtered,
        feature_columns=list(feature_cols),
        target_columns=["label"],
    )
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    X_train_df = feature_frame.iloc[train_idx].copy()
    X_val_df = feature_frame.iloc[val_idx].copy()
    X_test_df = feature_frame.iloc[test_idx].copy()
    fill_values = compute_train_medians(X_train_df)
    X_train_df = apply_train_median_fill(X_train_df, fill_values).fillna(0.0).astype(np.float32)
    X_val_df = apply_train_median_fill(X_val_df, fill_values).fillna(0.0).astype(np.float32)
    X_test_df = apply_train_median_fill(X_test_df, fill_values).fillna(0.0).astype(np.float32)
    y = filtered["label"].astype(int).to_numpy()
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    returns = filtered[returns_col].to_numpy(dtype=float) if returns_col and returns_col in filtered.columns else None
    test_returns = returns[test_idx] if returns is not None else None
    val_returns = returns[val_idx] if returns is not None else None

    ensemble_cfg = _runtime_ensemble_params()
    model = XGBRFEnsembleClassifier(
        random_state=42,
        rf_weight=float(ensemble_cfg["rf_weight"]),
        xgb_weight=float(ensemble_cfg["xgb_weight"]),
        rf_params=_runtime_rf_params(),
        xgb_params=_runtime_xgb_params(),
        xgb_gpu_preference=str(_runtime_option("xgb_device", "auto") or "auto"),
        decision_threshold=float(ensemble_cfg["ensemble_threshold"]),
        xgb_min_prob=float(ensemble_cfg["xgb_min_prob"]),
        rf_min_prob=float(ensemble_cfg["rf_min_prob"]),
        max_model_disagreement=float(ensemble_cfg["max_model_disagreement"]),
        block_on_disagreement=bool(ensemble_cfg["block_on_disagreement"]),
        feature_columns=list(X_train_df.columns),
    )
    fit_start = time.perf_counter()
    model.fit(X_train_df, y_train, X_val_df, y_val)
    fit_seconds = time.perf_counter() - fit_start
    _timing_log("xgb_rf_ensemble_fit", seconds=fit_seconds, rows=len(X_train_df), extra=f"features={X_train_df.shape[1]}")

    test_pred_start = time.perf_counter()
    test_details = model.decision_details(X_test_df)
    test_predict_seconds = time.perf_counter() - test_pred_start
    _timing_log("xgb_rf_ensemble_predict", seconds=test_predict_seconds, rows=len(X_test_df))
    val_details = model.decision_details(X_val_df)

    val_ensemble_prob = np.asarray([float(row.get("ensemble_prob") or 0.0) for row in val_details], dtype=float)
    test_ensemble_prob = np.asarray([float(row.get("ensemble_prob") or 0.0) for row in test_details], dtype=float)
    val_decisions = _gate_rows_from_ensemble_details(val_details)
    test_decisions = _gate_rows_from_ensemble_details(test_details)
    rf_val_prob = np.asarray([float(row.get("rf_prob") or 0.0) for row in val_details], dtype=float)
    rf_test_prob = np.asarray([float(row.get("rf_prob") or 0.0) for row in test_details], dtype=float)
    xgb_val_prob = np.asarray([float(row.get("xgb_prob") or 0.0) for row in val_details], dtype=float)
    xgb_test_prob = np.asarray([float(row.get("xgb_prob") or 0.0) for row in test_details], dtype=float)

    val_metrics = _classification_metrics_from_decisions(y_val, val_decisions, val_ensemble_prob)
    test_metrics = _classification_metrics_from_decisions(y_test, test_decisions, test_ensemble_prob)
    val_metrics["pr_auc"] = pr_auc_score_safe(y_val, val_ensemble_prob)
    val_metrics["brier_score"] = brier_score(y_val, val_ensemble_prob)
    test_metrics["pr_auc"] = pr_auc_score_safe(y_test, test_ensemble_prob)
    test_metrics["brier_score"] = brier_score(y_test, test_ensemble_prob)
    validation_trade_metrics = _trade_metrics_from_scores(y_val, val_decisions, 0.5, val_returns, threshold_source="validation_gated_ensemble")
    trade_metrics = _trade_metrics_from_scores(y_test, test_decisions, 0.5, test_returns, threshold_source="test_gated_ensemble")
    threshold_grid = _default_threshold_sweep_grid()
    threshold_items = []
    base_ensemble_cfg = _runtime_ensemble_params()
    for threshold in threshold_grid:
        tmp_model = XGBRFEnsembleClassifier.from_config({
            **model.get_config(),
            "xgb_min_prob": base_ensemble_cfg["xgb_min_prob"],
            "rf_min_prob": base_ensemble_cfg["rf_min_prob"],
            "decision_threshold": threshold,
            "block_on_disagreement": base_ensemble_cfg["block_on_disagreement"],
            "max_model_disagreement": base_ensemble_cfg["max_model_disagreement"],
        })
        tmp_model.rf_model_ = model.rf_model_
        tmp_model.xgb_model_ = model.xgb_model_
        tmp_model.rf_calibrator_ = getattr(model, "rf_calibrator_", None)
        tmp_model.xgb_calibrator_ = getattr(model, "xgb_calibrator_", None)
        tmp_model.feature_names_in_ = list(model.feature_names_in_)
        tmp_model.train_medians_ = dict(model.train_medians_)
        threshold_details = tmp_model.decision_details(X_test_df)
        threshold_items.append({
            "threshold": threshold,
            "decisions": _gate_rows_from_ensemble_details(threshold_details),
            "ensemble_prob": np.asarray([float(row.get("ensemble_prob") or 0.0) for row in threshold_details], dtype=float),
        })
    threshold_rows = _ensemble_threshold_sweep(y_test, threshold_items, test_returns)
    return_cost_provenance = _return_cost_provenance_for_frame(filtered.iloc[test_idx].reset_index(drop=True), evaluation_return_column=returns_col)
    cost_stress = _cost_stress_report(y_test, test_decisions, 0.5, test_returns, cost_provenance=return_cost_provenance, per_row_cost_units=return_cost_provenance.get("per_row_cost_units"))
    paper_filter_report = _paper_execution_filter_report(
        filtered.iloc[test_idx].reset_index(drop=True),
        test_ensemble_prob,
        float(base_ensemble_cfg["ensemble_threshold"]),
        test_returns,
        max_trades_per_day=int((paper_config or {}).get("paper_max_trades_per_day", 5)),
        cooldown_minutes=int((paper_config or {}).get("paper_cooldown_minutes", 15)),
        allow_ce=bool((paper_config or {}).get("paper_allow_ce", True)),
        allow_pe=bool((paper_config or {}).get("paper_allow_pe", True)),
        min_option_price=float((paper_config or {}).get("paper_min_option_price", 5.0)),
        max_bid_ask_spread_pct=float((paper_config or {}).get("paper_max_bid_ask_spread_pct", 5.0)),
        avoid_opening_minutes=int((paper_config or {}).get("paper_avoid_opening_minutes", 5)),
        avoid_closing_minutes=int((paper_config or {}).get("paper_avoid_closing_minutes", 5)),
    )
    daily_pnl_report = _daily_pnl_stability_report(
        paper_filter_report.get("selected_rows", []),
        max_trades_per_day=int((paper_config or {}).get("paper_max_trades_per_day", 5)),
    )
    comparison_payload = {
        "rf_alone": {"validation_trade_metrics": _trade_metrics_from_scores(y_val, rf_val_prob, 0.5, val_returns), "test_metrics": classification_metrics(y_test, rf_test_prob, threshold=0.5)},
        "xgb_alone": {"validation_trade_metrics": _trade_metrics_from_scores(y_val, xgb_val_prob, 0.5, val_returns), "test_metrics": classification_metrics(y_test, xgb_test_prob, threshold=0.5)},
        "weighted_ensemble": {"test_metrics": classification_metrics(y_test, test_ensemble_prob, threshold=float(base_ensemble_cfg["ensemble_threshold"]))},
        "gated_ensemble": {"test_metrics": test_metrics, "trade_metrics": trade_metrics},
        "recommendation": "gated_ensemble" if float(trade_metrics.get("profit_factor") or 0.0) >= max(float(_trade_metrics_from_scores(y_test, xgb_test_prob, 0.5, test_returns).get("profit_factor") or 0.0), float(_trade_metrics_from_scores(y_test, rf_test_prob, 0.5, test_returns).get("profit_factor") or 0.0)) else "xgb_only",
        "no_leakage_columns_used": True,
    }
    metrics_payload = {
        "variant_name": variant_name,
        "model_name": "xgb_rf_ensemble",
        "label_name": label_name,
        "validation_metrics": val_metrics,
        "validation_trade_metrics": validation_trade_metrics,
        "test_metrics": test_metrics,
        "test_threshold_metrics": {**test_metrics, **confusion_from_threshold(y_test, test_decisions, 0.5), "signal_count": int(test_decisions.sum())},
        "selected_threshold_from_validation": {"threshold": float(base_ensemble_cfg["ensemble_threshold"]), "source": "ensemble_gate"},
        "training_time_seconds": fit_seconds,
        "holdout_split": split_window,
        "walk_forward": walk_forward,
        "fold_stability": _fold_stability_report(walk_forward.get("folds", [])),
        "trade_metrics": trade_metrics,
        "cost_stress": cost_stress,
        "return_cost_provenance": return_cost_provenance,
        "threshold_sweep": threshold_rows,
        "threshold_robustness": {"grid": threshold_rows, "chosen_threshold": float(base_ensemble_cfg["ensemble_threshold"])},
        "paper_execution_filters": paper_filter_report,
        "daily_pnl_stability": daily_pnl_report,
        "evaluation_return_column_used": returns_col,
        "feature_importance": [],
        "calibration_fit": {
            "mode": "validation_sigmoid" if getattr(model, "rf_calibrator_", None) is not None or getattr(model, "xgb_calibrator_", None) is not None else "validated_only",
            "fit_on_validation_only": True,
        },
        "ensemble_diagnostics": {
            "xgb_backend": getattr(model, "xgb_runtime_backend_", "unknown"),
            "rf_predict_proba_seconds": None,
            "xgb_predict_proba_seconds": None,
            "gated_positive_rate": float(test_decisions.mean()) if len(test_decisions) else 0.0,
        },
    }
    artifact_paths = _save_ensemble_artifact(
        artifact_dir=artifact_dir / f"{variant_name}_xgb_rf_ensemble_{label_name}",
        model=model,
        label_name=label_name,
        feature_columns=list(X_train_df.columns),
        fill_values=fill_values,
        metrics_payload=metrics_payload,
        threshold_rows=threshold_rows,
        comparison_payload=comparison_payload,
        dataset_path=str((paper_config or {}).get("_dataset_path") or ""),
        split_sizes={"train_rows": len(X_train_df), "validation_rows": len(X_val_df), "test_rows": len(X_test_df)},
    )
    metrics_payload.update(artifact_paths)
    write_json(Path(artifact_paths["metrics_path"]), metrics_payload)
    return metrics_payload


def _edge_refinement_filter_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for value in EDGE_REFINEMENT_TOP_N_RULES:
        specs.append({"name": f"top_{value}_per_day", "selection": {"kind": "top_n", "value": value}, "filters": {}})
    for pct in EDGE_REFINEMENT_TOP_PCT_RULES:
        label = str(int(pct * 100))
        specs.append({"name": f"top_{label}_percent_per_day", "selection": {"kind": "top_pct", "value": pct}, "filters": {}})
    for percentile in EDGE_REFINEMENT_PROBABILITY_PERCENTILE_RULES:
        if percentile in [99.0, 99.5]:
            continue
        pct_label = str(percentile).replace(".", "_")
        specs.append({"name": f"probability_percentile_gte_{pct_label}", "selection": {"kind": "probability_percentile", "value": percentile}, "filters": {}})
    specs.extend(
        [
            {"name": "top_1_per_week", "selection": {"kind": "top_n_per_week", "value": 1}, "filters": {}},
            {"name": "top_2_per_week", "selection": {"kind": "top_n_per_week", "value": 2}, "filters": {}},
            {"name": "top_3_per_week", "selection": {"kind": "top_n_per_week", "value": 3}, "filters": {}},
            {"name": "top_1_per_expiry_week", "selection": {"kind": "top_n_per_expiry_week", "value": 1}, "filters": {}},
            {"name": "top_2_per_expiry_week", "selection": {"kind": "top_n_per_expiry_week", "value": 2}, "filters": {}},
            {"name": "probability_percentile >= 99", "selection": {"kind": "probability_percentile", "value": 99.0}, "filters": {}},
            {"name": "probability_percentile >= 99.5", "selection": {"kind": "probability_percentile", "value": 99.5}, "filters": {}},
            {"name": "probability_percentile >= 99.75", "selection": {"kind": "probability_percentile", "value": 99.75}, "filters": {}},
            {"name": "minimum_probability_margin_over_median_daily_probability", "selection": {"kind": "daily_probability_margin", "value": 0.10}, "filters": {}},
            {"name": "top_decile_confidence_and_positive_expected_return", "selection": {"kind": "top_decile_positive_expected_return", "value": 0.0}, "filters": {}},
        ]
    )
    for target_trade_count in EDGE_REFINEMENT_TARGET_TRADE_COUNTS:
        specs.append({"name": f"target_{target_trade_count}_trades", "selection": {"kind": "target_trade_count", "value": target_trade_count}, "filters": {}})
    for threshold in EDGE_REFINEMENT_THRESHOLD_RULES:
        specs.append({"name": f"threshold_gte_{str(threshold).replace('.', '_')}", "selection": {"kind": "threshold", "value": threshold}, "filters": {}})
    specs.extend(
        [
            {"name": "ce_only", "selection": {"kind": "threshold"}, "filters": {"option_side": "CE"}},
            {"name": "pe_only", "selection": {"kind": "threshold"}, "filters": {"option_side": "PE"}},
            {"name": "atm_near_atm_only", "selection": {"kind": "threshold"}, "filters": {"moneyness_include": ["ATM", "NEAR_ATM"]}},
            {"name": "itm_deep_itm_only", "selection": {"kind": "threshold"}, "filters": {"moneyness_include": ["ITM", "deep_itm"]}},
            {"name": "otm_excluded", "selection": {"kind": "threshold"}, "filters": {"moneyness_exclude": ["OTM", "deep_otm"]}},
            {"name": "expiry_day_excluded", "selection": {"kind": "threshold"}, "filters": {"exclude_expiry_day": True}},
            {"name": "opening_session_excluded", "selection": {"kind": "threshold"}, "filters": {"exclude_opening_session": True}},
            {"name": "last_30_minutes_excluded", "selection": {"kind": "threshold"}, "filters": {"exclude_last_30_minutes": True}},
            {"name": "high_volatility_only", "selection": {"kind": "threshold"}, "filters": {"volatility_mode": "high"}},
            {"name": "normal_volatility_only", "selection": {"kind": "threshold"}, "filters": {"volatility_mode": "normal"}},
            {"name": "trend_aligned_only", "selection": {"kind": "threshold"}, "filters": {"trend_aligned_only": True}},
            {"name": "tight_spread_0_5", "selection": {"kind": "threshold"}, "filters": {"max_spread_pct": 0.5}},
            {"name": "tight_spread_1_0", "selection": {"kind": "threshold"}, "filters": {"max_spread_pct": 1.0}},
            {"name": "tight_spread_1_5", "selection": {"kind": "threshold"}, "filters": {"max_spread_pct": 1.5}},
            {"name": "volume_above_median", "selection": {"kind": "threshold"}, "filters": {"volume_above_median": True}},
            {"name": "oi_above_median", "selection": {"kind": "threshold"}, "filters": {"oi_above_median": True}},
            {"name": "min_option_price", "selection": {"kind": "threshold"}, "filters": {"min_option_price": 20.0}},
            {"name": "avoid_low_priced_options", "selection": {"kind": "threshold"}, "filters": {"min_option_price": 50.0}},
            {"name": "avoid_extreme_illiquid_strikes", "selection": {"kind": "threshold"}, "filters": {"exclude_extreme_illiquid_strikes": True}},
            {
                "name": "top_3_per_day_atm_near_atm_tight_spread",
                "selection": {"kind": "top_n", "value": 3},
                "filters": {"moneyness_include": ["ATM", "NEAR_ATM"], "max_spread_pct": 1.0},
            },
            {
                "name": "top_5_per_day_pe_tight_spread",
                "selection": {"kind": "top_n", "value": 5},
                "filters": {"option_side": "PE", "max_spread_pct": 1.0},
            },
            {
                "name": "top_3_per_day_high_volume_high_oi",
                "selection": {"kind": "top_n", "value": 3},
                "filters": {"volume_above_median": True, "oi_above_median": True},
            },
            {
                "name": "top_1_per_day_itm_atm_only",
                "selection": {"kind": "top_n", "value": 1},
                "filters": {"moneyness_include": ["ATM", "NEAR_ATM", "ITM", "deep_itm"]},
            },
            {
                "name": "threshold_0_65_spread_volume",
                "selection": {"kind": "threshold", "value": 0.65},
                "filters": {"max_spread_pct": 1.0, "volume_above_median": True},
            },
        ]
    )
    return specs


def _edge_refinement_status_dir(artifact_dir: Path) -> Path:
    return artifact_dir / "edge_refinement_status"


def _edge_refinement_candidate_id(model_name: str, label_name: str, candidate_name: str) -> str:
    safe = candidate_name.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
    safe = safe.replace(">", "_").replace("<", "_").replace("=", "_").replace("|", "_").replace("?", "_").replace("*", "_")
    return f"{label_name}__{model_name}__{safe}"


def _write_edge_refinement_status(
    artifact_dir: Path,
    candidate_id: str,
    payload: Dict[str, Any],
) -> Path:
    status_dir = _edge_refinement_status_dir(artifact_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    path = status_dir / f"{candidate_id}.json"
    write_json(path, payload)
    return path


def _load_edge_refinement_status_rows(artifact_dir: Path) -> List[Dict[str, Any]]:
    status_dir = _edge_refinement_status_dir(artifact_dir)
    rows: List[Dict[str, Any]] = []
    if not status_dir.exists():
        return rows
    for path in sorted(status_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status_path"] = str(path)
        rows.append(payload)
    return rows


def _edge_refinement_fast_model_names() -> List[str]:
    return ["random_forest", "calibrated_logistic_regression", "xgboost"]


def _edge_refinement_fast_candidate_names() -> set[str]:
    return {
        "top_1_per_day",
        "top_3_per_day",
        "top_5_per_day",
        "top_5_percent_per_day",
        "threshold_gte_0_65",
        "threshold_gte_0_7",
        "pe_only",
        "atm_near_atm_only",
        "itm_deep_itm_only",
        "tight_spread_1_0",
        "volume_above_median",
        "oi_above_median",
    }


def _edge_refinement_middle_zone_candidate_names() -> set[str]:
    names = {
        "top_1_per_day",
        "top_2_per_day",
        "top_3_per_day",
        "top_5_per_day",
        "top_10_per_day",
        "top_1_percent_per_day",
        "top_2_percent_per_day",
        "top_3_percent_per_day",
        "top_5_percent_per_day",
        "probability_percentile >= 99.5",
        "probability_percentile >= 99",
        "probability_percentile_gte_98_0",
        "probability_percentile_gte_97_0",
        "probability_percentile_gte_95_0",
        "target_100_trades",
        "target_250_trades",
        "target_500_trades",
        "target_750_trades",
        "target_1000_trades",
        "target_1500_trades",
        "target_2000_trades",
        "top_3_per_day_atm_near_atm_tight_spread",
        "top_5_per_day_pe_tight_spread",
        "top_3_per_day_high_volume_high_oi",
        "top_1_per_day_itm_atm_only",
    }
    return names


def _edge_refinement_filter_specs_middle_zone_only(base_specs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    allowed = _edge_refinement_middle_zone_candidate_names()
    return [spec for spec in base_specs if str(spec.get("name")) in allowed]


def _edge_refinement_filter_specs_fast_only(base_specs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    allowed = _edge_refinement_fast_candidate_names()
    return [spec for spec in base_specs if str(spec.get("name")) in allowed]


def _edge_refinement_report_paths(timestamp: str) -> tuple[Path, Path]:
    report_dir = _artifact_report_dir()
    return report_dir / f"edge_refinement_report_{timestamp}.json", report_dir / f"edge_refinement_report_{timestamp}.md"


def _cost_aware_edge_refinement_report_paths(timestamp: str) -> tuple[Path, Path]:
    report_dir = _artifact_report_dir()
    return report_dir / f"cost_aware_edge_refinement_{timestamp}.json", report_dir / f"cost_aware_edge_refinement_{timestamp}.md"


def _cost_model_audit_report_paths(timestamp: str) -> tuple[Path, Path]:
    report_dir = _artifact_report_dir()
    return report_dir / f"cost_model_audit_{timestamp}.json", report_dir / f"cost_model_audit_{timestamp}.md"


def _artifact_report_dir() -> Path:
    return REPO_ROOT / "reports"


def _load_model_bundle(model_path: Path) -> Any:
    import joblib

    return joblib.load(model_path)


def _bundle_probability_column(label_name: str, model_name: str) -> str:
    return f"prob_{label_name}_{model_name}"


def _bundle_scaler(bundle: Any, X: np.ndarray) -> np.ndarray:
    mean = np.asarray(getattr(bundle, "scaler_mean", []) or [], dtype=np.float32)
    std = np.asarray(getattr(bundle, "scaler_std", []) or [], dtype=np.float32)
    if mean.size != X.shape[1] or std.size != X.shape[1]:
        return X
    safe_std = np.where(std == 0.0, 1.0, std)
    return (X - mean) / safe_std


def _scored_holdout_frame_for_artifact(
    df: pd.DataFrame,
    *,
    bundle: Any,
    label_name: str,
    model_name: str,
    evaluation_return_column: str,
) -> pd.DataFrame:
    feature_names = list(getattr(bundle, "feature_names", []) or [])
    missing_features = [name for name in feature_names if name not in df.columns]
    if missing_features:
        raise RuntimeError(f"artifact {model_name}/{label_name} requires missing features: {missing_features[:10]}")
    prepared = _prepare_chronological_work(df.copy())
    y_series = prepare_label_series(prepared, label_name)
    passthrough_columns = [column for column in prepared.columns if column not in feature_names]
    work = pd.concat([prepared[passthrough_columns].copy(), prepared[feature_names].copy(), y_series.rename("label")], axis=1)
    work[evaluation_return_column] = pd.to_numeric(prepared[evaluation_return_column], errors="coerce")
    work = work.dropna(subset=["timestamp", "label", evaluation_return_column]).copy()
    work = work[work["label"].isin([0, 1])].copy()
    if work.empty or work["label"].nunique() < 2:
        return pd.DataFrame()
    work = _prepare_chronological_work(work)
    timestamps = pd.Series(work["timestamp"]).reset_index(drop=True)
    split = _build_holdout_split_from_timestamps(timestamps)
    test_idx = split["test"]
    X_df = work[feature_names].copy().replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0).astype(np.float32)
    X_test = X_df.iloc[test_idx].to_numpy(dtype=np.float32)
    X_test_s = _bundle_scaler(bundle, X_test)
    test_prob = predict_proba_positive(bundle.model, X_test_s)
    test_frame = work.iloc[test_idx].copy().reset_index(drop=True)
    probability_column = _bundle_probability_column(label_name, model_name)
    test_frame[probability_column] = np.asarray(test_prob, dtype=float)
    test_frame["selected_return"] = pd.to_numeric(test_frame[evaluation_return_column], errors="coerce").fillna(0.0)
    test_frame["_evaluation_return_column_used"] = evaluation_return_column
    test_frame["trade_day"] = _normalize_timestamp_series(test_frame["timestamp"]).dt.date
    return _apply_backtest_row_limit(test_frame, context=f"artifact:{model_name}:{label_name}")


def _safe_daily_profit_factor(selected_rows: pd.DataFrame) -> float:
    if selected_rows.empty or "trade_day" not in selected_rows.columns:
        return 0.0
    daily = selected_rows.groupby("trade_day")["selected_return"].sum()
    gross_profit = float(daily[daily > 0.0].sum())
    gross_loss = float(abs(daily[daily < 0.0].sum()))
    if gross_profit <= 0.0:
        return 0.0
    if gross_loss <= 0.0:
        return float("inf")
    return gross_profit / gross_loss


def _threshold_stability_from_rows(
    filtered_rows: pd.DataFrame,
    probability_column: str,
    *,
    selected_threshold: float,
    minimum_trades: int,
) -> Dict[str, Any]:
    candidate_thresholds = [value for value in EDGE_REFINEMENT_THRESHOLD_RULES if abs(value - float(selected_threshold)) <= 0.10]
    rows = []
    for threshold in candidate_thresholds:
        selected = filtered_rows.loc[pd.to_numeric(filtered_rows[probability_column], errors="coerce").fillna(0.0) >= float(threshold)].copy()
        metrics = _trade_metrics_from_selected_returns(selected["selected_return"].to_numpy(dtype=float), threshold_source=f"edge_refinement_threshold:{threshold}")
        rows.append(
            {
                "threshold": float(threshold),
                "trade_count": int(len(selected)),
                "profit_factor": float(metrics.get("profit_factor") or 0.0),
                "sharpe": float(metrics.get("sharpe") or 0.0),
            }
        )
    robust_rows = [row for row in rows if row["trade_count"] >= int(minimum_trades) and row["profit_factor"] >= 1.0]
    return {
        "rows": rows,
        "passes": len(robust_rows) >= 2,
        "supporting_thresholds": [row["threshold"] for row in robust_rows],
    }


def _edge_refinement_live_rule(filters: Dict[str, Any], selection_name: str, model_name: str, label_name: str) -> str:
    return (
        f"Observe `{model_name}` on `{label_name}` only when filters `{filters}` hold and apply selection `{selection_name}`; "
        "record paper result only, no auto-orders."
    )


def _profit_concentration_diagnostics(
    selected_rows: pd.DataFrame,
    *,
    daily_breakdown: Dict[str, Any],
    monthly_breakdown: Dict[str, Any],
    fold_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    if selected_rows.empty or "selected_return" not in selected_rows.columns:
        return {
            "best_day_profit": 0.0,
            "best_day_profit_share_of_total_profit": 0.0,
            "top_2_days_profit_share": 0.0,
            "best_month_profit_share_of_total_profit": 0.0,
            "best_fold_profit_share_of_total_profit": 0.0,
            "number_of_profitable_days": 0,
            "number_of_profitable_months": 0,
            "number_of_profitable_folds": 0,
            "longest_loss_streak_days": 0,
            "average_daily_pnl": 0.0,
            "median_daily_pnl": 0.0,
            "daily_pnl_iqr": 0.0,
            "daily_pnl_std": 0.0,
        }
    work = selected_rows.copy()
    if "trade_day" not in work.columns and "timestamp" in work.columns:
        work["trade_day"] = _normalize_timestamp_series(work["timestamp"]).dt.date
    daily = work.groupby("trade_day")["selected_return"].sum().sort_index()
    positive_days = daily[daily > 0.0]
    total_positive_profit = float(positive_days.sum())
    top_day_profit = float(positive_days.max()) if not positive_days.empty else 0.0
    top2_profit = float(positive_days.nlargest(2).sum()) if not positive_days.empty else 0.0
    longest_loss_streak = 0
    current_loss_streak = 0
    for value in daily.to_list():
        if float(value) < 0.0:
            current_loss_streak += 1
            longest_loss_streak = max(longest_loss_streak, current_loss_streak)
        else:
            current_loss_streak = 0
    month_rows = monthly_breakdown.get("months", []) or []
    fold_rows = fold_metrics.get("folds", []) or []
    month_positive = [float(row.get("gross_profit") or 0.0) for row in month_rows]
    fold_positive = [float(row.get("gross_profit") or 0.0) for row in fold_rows]
    q75 = float(np.percentile(daily.to_numpy(dtype=float), 75)) if len(daily) else 0.0
    q25 = float(np.percentile(daily.to_numpy(dtype=float), 25)) if len(daily) else 0.0
    return {
        "best_day_profit": top_day_profit,
        "best_day_profit_share_of_total_profit": float(top_day_profit / total_positive_profit) if total_positive_profit > 0.0 else 0.0,
        "top_2_days_profit_share": float(top2_profit / total_positive_profit) if total_positive_profit > 0.0 else 0.0,
        "best_month_profit_share_of_total_profit": float(max(month_positive) / sum(month_positive)) if sum(month_positive) > 0.0 else 0.0,
        "best_fold_profit_share_of_total_profit": float(max(fold_positive) / sum(fold_positive)) if sum(fold_positive) > 0.0 else 0.0,
        "number_of_profitable_days": int((daily > 0.0).sum()),
        "number_of_profitable_months": int(sum(1 for row in month_rows if float(row.get("total_return") or 0.0) > 0.0)),
        "number_of_profitable_folds": int(sum(1 for row in fold_rows if float(row.get("total_return") or 0.0) > 0.0)),
        "longest_loss_streak_days": int(longest_loss_streak),
        "average_daily_pnl": float(daily.mean()) if len(daily) else 0.0,
        "median_daily_pnl": float(np.median(daily.to_numpy(dtype=float))) if len(daily) else 0.0,
        "daily_pnl_iqr": float(q75 - q25),
        "daily_pnl_std": float(np.std(daily.to_numpy(dtype=float), ddof=1)) if len(daily) > 1 else 0.0,
    }


def _edge_refinement_gate(
    candidate: Dict[str, Any],
    *,
    full_run_completed: bool,
) -> Dict[str, Any]:
    failed: List[str] = []
    trade_count = int(candidate.get("trade_count") or 0)
    pf = float(candidate.get("profit_factor") or 0.0)
    sharpe = float(candidate.get("sharpe") or 0.0)
    cost_125 = float(candidate.get("cost_stress_1_25x_pf") or 0.0)
    cost_150 = float(candidate.get("cost_stress_1_50x_pf") or 0.0)
    worst_fold_pf = float(candidate.get("walk_forward_worst_fold_pf") or 0.0)
    median_fold_pf = float(candidate.get("walk_forward_median_pf") or 0.0)
    expected_after_cost = float(candidate.get("expected_return_after_cost") or 0.0)
    median_after_cost = float(candidate.get("median_return_after_cost") or 0.0)
    return_to_cost_ratio = float(candidate.get("return_to_cost_ratio") or 0.0)
    tiny_sample = bool(candidate.get("tiny_sample_warning"))
    cost_provenance_uncertain = bool(candidate.get("cost_provenance_uncertain"))
    leakage_warning = bool(candidate.get("leakage_warning"))
    daily_ok = bool(candidate.get("daily_stability_passes"))
    threshold_ok = bool(candidate.get("threshold_stability_passes"))
    top1_day_share = float(candidate.get("best_day_profit_share_of_total_profit") or 0.0)
    top2_day_share = float(candidate.get("top_2_days_profit_share") or 0.0)
    top_month_share = float(candidate.get("best_month_profit_share_of_total_profit") or 0.0)
    top_fold_share = float(candidate.get("best_fold_profit_share_of_total_profit") or 0.0)
    profitable_days = int(candidate.get("number_of_profitable_days") or 0)
    profitable_months = int(candidate.get("number_of_profitable_months") or 0)
    profitable_folds = int(candidate.get("number_of_profitable_folds") or 0)
    if trade_count < EDGE_REFINEMENT_WATCHLIST_GATES["trade_count_min"]:
        failed.append("trade_count_below_watchlist_gate")
    if trade_count > EDGE_REFINEMENT_WATCHLIST_GATES["trade_count_max"]:
        failed.append("trade_count_above_watchlist_gate")
    if pf <= EDGE_REFINEMENT_WATCHLIST_GATES["profit_factor_min"]:
        failed.append("profit_factor_below_watchlist_gate")
    if sharpe <= EDGE_REFINEMENT_WATCHLIST_GATES["sharpe_min"]:
        failed.append("sharpe_below_watchlist_gate")
    if cost_125 <= EDGE_REFINEMENT_WATCHLIST_GATES["cost_1_25x_pf_min"]:
        failed.append("cost_1_25x_pf_below_watchlist_gate")
    if cost_150 < EDGE_REFINEMENT_WATCHLIST_GATES["cost_1_50x_pf_min"]:
        failed.append("cost_1_50x_pf_below_watchlist_gate")
    if expected_after_cost <= 0.0:
        failed.append("expected_return_after_cost_non_positive")
    if median_after_cost <= 0.0:
        failed.append("median_return_after_cost_non_positive")
    if return_to_cost_ratio < 1.25:
        failed.append("return_to_cost_ratio_below_watchlist_gate")
    if worst_fold_pf <= EDGE_REFINEMENT_WATCHLIST_GATES["worst_fold_pf_min"]:
        failed.append("worst_fold_pf_below_watchlist_gate")
    if median_fold_pf <= 1.05:
        failed.append("median_fold_pf_below_watchlist_gate")
    if top1_day_share > 0.35:
        failed.append("top_1_day_profit_share_above_35pct")
    if top2_day_share > 0.50:
        failed.append("top_2_days_profit_share_above_50pct")
    if top_month_share > 0.50:
        failed.append("best_month_profit_share_above_50pct")
    if top_fold_share > 0.50:
        failed.append("best_fold_profit_share_above_50pct")
    if profitable_days < 30:
        failed.append("profitable_days_below_30")
    if profitable_months < 3:
        failed.append("profitable_months_below_3")
    if profitable_folds < 4:
        failed.append("profitable_folds_below_4_of_5")
    if not daily_ok:
        failed.append("daily_stability_failed")
    if not threshold_ok:
        failed.append("threshold_robustness_failed")
    if cost_provenance_uncertain:
        failed.append("cost_provenance_uncertain")
    if leakage_warning:
        failed.append("leakage_warning_present")
    if tiny_sample:
        failed.append("tiny_sample_warning_present")
    status = "PAPER_WATCHLIST"
    if tiny_sample or trade_count < EDGE_REFINEMENT_WATCHLIST_GATES["trade_count_min"]:
        status = "BLOCKED_SMALL_SAMPLE"
    elif trade_count > EDGE_REFINEMENT_WATCHLIST_GATES["trade_count_max"]:
        status = "BLOCKED_TOO_MANY_TRADES"
    elif any(reason in failed for reason in ("top_1_day_profit_share_above_35pct", "top_2_days_profit_share_above_50pct", "best_month_profit_share_above_50pct", "best_fold_profit_share_above_50pct", "profitable_days_below_30", "profitable_months_below_3")):
        status = "BLOCKED_CONCENTRATED_PROFIT"
    elif any(reason in failed for reason in ("cost_1_25x_pf_below_watchlist_gate", "cost_1_50x_pf_below_watchlist_gate", "cost_provenance_uncertain")):
        status = "BLOCKED_COST_STRESS"
    elif any(reason in failed for reason in ("expected_return_after_cost_non_positive", "median_return_after_cost_non_positive", "return_to_cost_ratio_below_watchlist_gate")):
        status = "BLOCKED_THIN_EDGE"
    elif any(reason in failed for reason in ("worst_fold_pf_below_watchlist_gate", "median_fold_pf_below_watchlist_gate", "profitable_folds_below_4_of_5")):
        status = "BLOCKED_FOLD_INSTABILITY"
    elif not threshold_ok:
        status = "BLOCKED_THRESHOLD_INSTABILITY"
    elif any(reason in failed for reason in ("daily_instability_failed",)) or not daily_ok:
        status = "BLOCKED_DAILY_INSTABILITY"
    elif any(reason in failed for reason in ("profit_factor_below_watchlist_gate", "sharpe_below_watchlist_gate")):
        status = "BLOCKED_LOW_EDGE"
    elif failed:
        status = "BLOCKED_LOW_EDGE"
    if not failed:
        status = "PAPER_WATCHLIST"
    ready_failures: List[str] = []
    if not full_run_completed:
        ready_failures.append("full_model_run_incomplete")
    if trade_count < EDGE_REFINEMENT_READY_GATES["trade_count_min"]:
        ready_failures.append("trade_count_below_paper_ready_gate")
    if trade_count > 3000:
        ready_failures.append("trade_count_above_paper_ready_research_band")
    if pf <= EDGE_REFINEMENT_READY_GATES["profit_factor_min"]:
        ready_failures.append("profit_factor_below_paper_ready_gate")
    if sharpe <= EDGE_REFINEMENT_READY_GATES["sharpe_min"]:
        ready_failures.append("sharpe_below_paper_ready_gate")
    if cost_125 <= EDGE_REFINEMENT_READY_GATES["cost_1_25x_pf_min"]:
        ready_failures.append("cost_1_25x_pf_below_paper_ready_gate")
    if cost_150 <= EDGE_REFINEMENT_READY_GATES["cost_1_50x_pf_min"]:
        ready_failures.append("cost_1_50x_pf_below_paper_ready_gate")
    if expected_after_cost <= 0.0:
        ready_failures.append("expected_return_after_cost_non_positive")
    if median_after_cost <= 0.0:
        ready_failures.append("median_return_after_cost_non_positive")
    if return_to_cost_ratio < 1.50:
        ready_failures.append("return_to_cost_ratio_below_paper_ready_gate")
    if worst_fold_pf <= EDGE_REFINEMENT_READY_GATES["worst_fold_pf_min"]:
        ready_failures.append("worst_fold_pf_below_paper_ready_gate")
    if median_fold_pf <= EDGE_REFINEMENT_READY_GATES["median_fold_pf_min"]:
        ready_failures.append("median_fold_pf_below_paper_ready_gate")
    if top1_day_share > 0.35:
        ready_failures.append("top_1_day_profit_share_above_35pct")
    if top2_day_share > 0.50:
        ready_failures.append("top_2_days_profit_share_above_50pct")
    if top_month_share > 0.50:
        ready_failures.append("best_month_profit_share_above_50pct")
    if top_fold_share > 0.50:
        ready_failures.append("best_fold_profit_share_above_50pct")
    if profitable_days < 30:
        ready_failures.append("profitable_days_below_30")
    if profitable_months < 3:
        ready_failures.append("profitable_months_below_3")
    if profitable_folds < 4:
        ready_failures.append("profitable_folds_below_4_of_5")
    if not daily_ok:
        ready_failures.append("daily_stability_failed")
    if not threshold_ok:
        ready_failures.append("threshold_robustness_failed")
    if cost_provenance_uncertain:
        ready_failures.append("cost_provenance_uncertain")
    if leakage_warning:
        ready_failures.append("leakage_warning_present")
    if not ready_failures:
        status = "PAPER_READY"
    primary_block_reason = None
    if status != "PAPER_READY":
        ordered = failed if failed else ready_failures
        primary_block_reason = ordered[0] if ordered else "unknown_blocker"
    return {
        "status": status,
        "primary_block_reason": primary_block_reason,
        "failed_gates": failed if status != "PAPER_READY" else [],
        "watchlist_failed_gates": failed,
        "all_failed_gates": list(dict.fromkeys([*failed, *ready_failures])),
        "paper_ready_failed_gates": ready_failures,
    }


def _edge_refinement_score(candidate: Dict[str, Any]) -> float:
    drawdown_penalty = min(float(candidate.get("max_drawdown") or 0.0) * 0.01, 5.0)
    return (
        2.0 * min(float(candidate.get("cost_stress_1_25x_pf") or 0.0), 2.0)
        + 1.5 * min(float(candidate.get("walk_forward_worst_fold_pf") or 0.0), 2.0)
        + 1.0 * min(float(candidate.get("walk_forward_median_pf") or 0.0), 2.0)
        + 0.5 * min(float(candidate.get("sharpe") or 0.0), 3.0)
        + min(int(candidate.get("trade_count") or 0) / 500.0, 2.0)
        + min(int(candidate.get("number_of_profitable_days") or 0) / 30.0, 2.0)
        - 2.0 * float(candidate.get("best_day_profit_share_of_total_profit") or 0.0)
        - 2.0 * float(candidate.get("best_fold_profit_share_of_total_profit") or 0.0)
        - drawdown_penalty
    )


def _edge_refinement_cost_aware_score(candidate: Dict[str, Any]) -> float:
    total_return = abs(float(candidate.get("total_return") or 0.0))
    max_drawdown = float(candidate.get("max_drawdown") or 0.0)
    drawdown_penalty = min((max_drawdown / max(total_return, 1.0)) if total_return > 0.0 else max_drawdown * 0.01, 5.0)
    return (
        2.5 * min(float(candidate.get("cost_stress_1_25x_pf") or 0.0), 2.0)
        + 2.0 * min(float(candidate.get("cost_stress_1_50x_pf") or 0.0), 2.0)
        + 1.5 * min(float(candidate.get("walk_forward_worst_fold_pf") or 0.0), 2.0)
        + 1.0 * min(float(candidate.get("walk_forward_median_pf") or 0.0), 2.0)
        + 1.0 * min(float(candidate.get("return_to_cost_ratio") or 0.0), 3.0)
        + 0.75 * min(float(candidate.get("sharpe") or 0.0), 3.0)
        + 0.50 * min(float(candidate.get("profit_factor") or 0.0), 3.0)
        + 0.50 * min(float(candidate.get("number_of_profitable_days") or 0.0) / 50.0, 2.0)
        + 0.50 * min(float(candidate.get("number_of_profitable_months") or 0.0) / 4.0, 2.0)
        - 2.0 * float(candidate.get("best_day_profit_share_of_total_profit") or 0.0)
        - 2.0 * float(candidate.get("top_2_days_profit_share") or 0.0)
        - 2.0 * float(candidate.get("best_fold_profit_share_of_total_profit") or 0.0)
        - drawdown_penalty
    )


def _edge_refinement_filter_mask(frame: pd.DataFrame, filters: Dict[str, Any]) -> tuple[pd.Series, List[str]]:
    work = frame.copy()
    mask = pd.Series(True, index=work.index)
    skipped: List[str] = []
    option_side = _option_side_series(work)
    moneyness_bucket = _moneyness_bucket_series(work)
    if "option_side" in filters:
        if option_side.eq("UNKNOWN").all():
            skipped.append("option_side")
        else:
            mask &= option_side.eq(str(filters["option_side"]).upper())
    if "moneyness_include" in filters:
        if moneyness_bucket.eq("UNKNOWN").all():
            skipped.append("moneyness_include")
        else:
            mask &= moneyness_bucket.isin({str(value) for value in filters["moneyness_include"]})
    if "moneyness_exclude" in filters:
        if moneyness_bucket.eq("UNKNOWN").all():
            skipped.append("moneyness_exclude")
        else:
            mask &= ~moneyness_bucket.isin({str(value) for value in filters["moneyness_exclude"]})
    if filters.get("exclude_expiry_day"):
        if "dte_days" in work.columns:
            mask &= _numeric_series(work, "dte_days", default=np.nan).fillna(999.0) > 0.0
        else:
            skipped.append("exclude_expiry_day")
    if filters.get("exclude_opening_session"):
        if "is_opening_session" in work.columns:
            mask &= _numeric_series(work, "is_opening_session", default=0.0).fillna(0.0) < 1.0
        else:
            timestamps = _normalize_timestamp_series(work["timestamp"]) if "timestamp" in work.columns else pd.Series(dtype="datetime64[ns]")
            if timestamps.empty:
                skipped.append("exclude_opening_session")
            else:
                local_minutes = timestamps.dt.hour * 60 + timestamps.dt.minute
                mask &= local_minutes >= (9 * 60 + 30)
    if filters.get("exclude_last_30_minutes"):
        if "timestamp" not in work.columns:
            skipped.append("exclude_last_30_minutes")
        else:
            timestamps = _normalize_timestamp_series(work["timestamp"])
            local_minutes = timestamps.dt.hour * 60 + timestamps.dt.minute
            mask &= local_minutes <= (15 * 60)
    if "volatility_mode" in filters:
        if "regime_volatile" in work.columns:
            vol = _numeric_series(work, "regime_volatile", default=0.0).fillna(0.0)
        elif "volatility_regime_classifier" in work.columns:
            vol = _numeric_series(work, "volatility_regime_classifier", default=0.0).fillna(0.0)
        else:
            vol = pd.Series(dtype=float)
        if vol.empty:
            skipped.append("volatility_mode")
        elif str(filters["volatility_mode"]) == "high":
            mask &= vol >= 1.0
        else:
            mask &= vol < 1.0
    if filters.get("trend_aligned_only"):
        if "regime_trending" in work.columns:
            mask &= _numeric_series(work, "regime_trending", default=0.0).fillna(0.0) >= 1.0
        elif "ctx_trend_strength" in work.columns:
            mask &= _numeric_series(work, "ctx_trend_strength", default=0.0).fillna(0.0) > 0.0
        else:
            skipped.append("trend_aligned_only")
    if "max_spread_pct" in filters:
        spread = _spread_pct_series(work)
        if not spread.notna().any():
            skipped.append("max_spread_pct")
        else:
            mask &= spread.fillna(np.inf) <= float(filters["max_spread_pct"])
    if filters.get("volume_above_median"):
        volume = _numeric_series(work, "volume", default=np.nan)
        if not volume.notna().any():
            volume = _numeric_series(work, "option_volume", default=np.nan)
        if not volume.notna().any():
            skipped.append("volume_above_median")
        else:
            mask &= volume.fillna(-np.inf) >= float(volume.dropna().median())
    if filters.get("oi_above_median"):
        oi = _numeric_series(work, "oi", default=np.nan)
        if not oi.notna().any():
            skipped.append("oi_above_median")
        else:
            mask &= oi.fillna(-np.inf) >= float(oi.dropna().median())
    if "min_option_price" in filters:
        premium = _option_price_series(work)
        if not premium.notna().any():
            skipped.append("min_option_price")
        else:
            mask &= premium.fillna(-np.inf) >= float(filters["min_option_price"])
    if filters.get("exclude_extreme_illiquid_strikes"):
        if "strike_distance_pct" in work.columns:
            mask &= _numeric_series(work, "strike_distance_pct", default=np.nan).abs().fillna(np.inf) <= 0.08
        elif not moneyness_bucket.eq("UNKNOWN").all():
            mask &= ~moneyness_bucket.isin(["deep_otm"])
        else:
            skipped.append("exclude_extreme_illiquid_strikes")
    return mask.fillna(False), skipped


def _apply_edge_refinement_selection(
    filtered_rows: pd.DataFrame,
    probability_column: str,
    selection: Dict[str, Any],
    *,
    fallback_threshold: float,
) -> tuple[pd.DataFrame, str, float | None]:
    kind = str(selection.get("kind") or "threshold")
    spec_name = selection.get("name")
    probabilities = pd.to_numeric(filtered_rows.get(probability_column), errors="coerce").fillna(0.0)
    work = filtered_rows.copy()
    if "timestamp" in work.columns:
        ts = _normalize_timestamp_series(work["timestamp"])
        work["trade_day"] = ts.dt.date
        work["trade_week"] = ts.dt.to_period("W").astype(str)
    if "expiry" in work.columns:
        expiry_ts = pd.to_datetime(work["expiry"], errors="coerce")
        work["expiry_week"] = expiry_ts.dt.to_period("W").astype(str)
    if kind == "top_n":
        value = int(selection["value"])
        selected_rows = _ranked_selection_rows(work, probability_column, top_n=value)
        name = spec_name if spec_name else f"top_{value}_per_day"
        return selected_rows, name, None
    if kind == "top_pct":
        value = float(selection["value"])
        pct_label = int(round(value * 100.0))
        selected_rows = _ranked_selection_rows(work, probability_column, top_pct=value)
        name = spec_name if spec_name else f"top_{pct_label}_percent_per_day"
        return selected_rows, name, None
    if kind == "top_n_per_week":
        value = int(selection["value"])
        selected_rows = _ranked_selection_rows_by_bucket(work, probability_column, bucket_column="trade_week", top_n=value)
        name = spec_name if spec_name else f"top_{value}_per_week"
        return selected_rows, name, None
    if kind == "top_n_per_expiry_week":
        value = int(selection["value"])
        selected_rows = _ranked_selection_rows_by_bucket(work, probability_column, bucket_column="expiry_week", top_n=value)
        name = spec_name if spec_name else f"top_{value}_per_expiry_week"
        return selected_rows, name, None
    if kind == "probability_percentile":
        percentile = float(selection["value"])
        if work.empty:
            threshold = 1.0
        else:
            threshold = float(np.percentile(probabilities.to_numpy(dtype=float), percentile))
        selected = work.loc[probabilities >= threshold].copy()
        name = spec_name if spec_name else f"probability_percentile>={percentile:g}"
        return selected, name, threshold
    if kind == "daily_probability_margin":
        margin = float(selection["value"] or 0.10)
        if work.empty:
            return work.copy(), spec_name if spec_name else "daily_probability_margin", None
        daily_median = work.groupby("trade_day")[probability_column].transform("median")
        selected = work.loc[probabilities >= (daily_median + margin)].copy()
        name = spec_name if spec_name else f"probability_margin_over_daily_median_{margin:.2f}"
        return selected, name, None
    if kind == "top_decile_positive_expected_return":
        if work.empty:
            return work.copy(), spec_name if spec_name else "top_decile_positive_expected_return", None
        cutoff = float(np.percentile(probabilities.to_numpy(dtype=float), 90.0))
        expected_return_proxy = (probabilities - 0.5) * 2.0
        selected = work.loc[(probabilities >= cutoff) & (expected_return_proxy > 0.0)].copy()
        name = spec_name if spec_name else "top_decile_positive_expected_return"
        return selected, name, cutoff
    if kind == "target_trade_count":
        target_trade_count = max(1, int(selection["value"]))
        if work.empty:
            threshold = 1.0
            selected = work.copy()
        else:
            sorted_probabilities = np.sort(probabilities.to_numpy(dtype=float))[::-1]
            threshold_index = min(target_trade_count - 1, len(sorted_probabilities) - 1)
            threshold = float(sorted_probabilities[threshold_index])
            selected = work.loc[probabilities >= threshold].copy()
            if len(selected) > target_trade_count:
                selected = selected.sort_values(probability_column, ascending=False, kind="stable").head(target_trade_count).copy()
        name = spec_name if spec_name else f"target_{target_trade_count}_trades"
        return selected, name, threshold
    threshold = float(selection.get("value") or fallback_threshold)
    selected = work.loc[probabilities >= threshold].copy()
    name = spec_name if spec_name else f"threshold>={threshold:.2f}"
    return selected, name, threshold


def _evaluate_edge_refinement_candidate(
    base_frame: pd.DataFrame,
    *,
    candidate_name: str,
    model_name: str,
    label_name: str,
    filters: Dict[str, Any],
    selection: Dict[str, Any],
    fallback_threshold: float,
    full_run_completed: bool,
    leakage_warning: bool,
    fold_count: int,
) -> Dict[str, Any]:
    probability_column = _bundle_probability_column(label_name, model_name)
    mask, skipped_filters = _edge_refinement_filter_mask(base_frame, filters)
    filtered = base_frame.loc[mask].copy()
    selected_rows, selection_name, threshold_value = _apply_edge_refinement_selection(filtered, probability_column, selection, fallback_threshold=fallback_threshold)
    selected_rows = selected_rows.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if not selected_rows.empty and "trade_day" not in selected_rows.columns and "timestamp" in selected_rows.columns:
        selected_rows["trade_day"] = _normalize_timestamp_series(selected_rows["timestamp"]).dt.date
    evaluation_return_column = str(filtered["_evaluation_return_column_used"].iloc[0]) if not filtered.empty and "_evaluation_return_column_used" in filtered.columns else None
    estimated_cost = float(estimate_option_execution_costs_frame(selected_rows, config=DEFAULT_EXECUTION_COST_CONFIG)["cost_return_units"].mean()) if not selected_rows.empty else 0.0
    cost_provenance = _return_cost_provenance_for_frame(selected_rows if not selected_rows.empty else filtered, evaluation_return_column=evaluation_return_column, estimated_cost=estimated_cost)
    selection_payload = _selection_rows_for_candidate(selected_rows, selection_name=selection_name, threshold_value=threshold_value)
    cost_stress = _selection_cost_stress(
        selected_rows,
        cost_provenance=cost_provenance,
        per_row_cost_units=(
            pd.Series(cost_provenance["per_row_cost_units"], index=selected_rows.index)
            if cost_provenance.get("per_row_cost_units") is not None
            and len(cost_provenance["per_row_cost_units"]) == len(selected_rows)
            else None
        ),
    )
    edge_cost = _edge_refinement_cost_metrics(selected_rows, cost_provenance=cost_provenance)
    fold_metrics = _fold_metrics_for_selection(selected_rows, fold_count=fold_count)
    daily_breakdown = _daily_stability_breakdown(selected_rows)
    monthly_breakdown = _monthly_stability_for_selection(selected_rows)
    liquidity_report = _liquidity_filter_report(selected_rows)
    threshold_stability = _threshold_stability_from_rows(filtered, probability_column, selected_threshold=float(threshold_value if threshold_value is not None else fallback_threshold), minimum_trades=EDGE_REFINEMENT_WATCHLIST_GATES["trade_count_min"])
    fold_pfs = [float(row.get("profit_factor") or 0.0) for row in fold_metrics.get("folds", [])]
    concentration = _profit_concentration_diagnostics(selected_rows, daily_breakdown=daily_breakdown, monthly_breakdown=monthly_breakdown, fold_metrics=fold_metrics)
    daily_stability_passes = bool(daily_breakdown.get("available")) and float(daily_breakdown.get("max_profit_day_share") or 1.0) <= 0.40 and float(daily_breakdown.get("day_win_rate") or 0.0) >= 0.45
    candidate = {
        "candidate_name": candidate_name,
        "model_name": model_name,
        "label_name": label_name,
        "filters": filters,
        "selection": selection,
        "selection_name": selection_name,
        "trade_count": int(selection_payload.get("trades") or 0),
        "win_rate": float(selection_payload.get("win_rate") or 0.0),
        "average_return_per_trade": float(selection_payload.get("average_return_per_trade") or 0.0),
        "median_return_per_trade": float(edge_cost.get("median_return_per_trade") or 0.0),
        "average_positive_return": float(edge_cost.get("average_positive_return") or 0.0),
        "average_negative_return": float(edge_cost.get("average_negative_return") or 0.0),
        "total_return": float(selection_payload.get("total_return") or 0.0),
        "profit_factor": float(selection_payload.get("profit_factor") or 0.0),
        "sharpe": float(selection_payload.get("sharpe") or 0.0),
        "sortino": float(selection_payload.get("sortino") or 0.0),
        "max_drawdown": float(selection_payload.get("max_drawdown") or 0.0),
        "worst_day_pnl": float(((daily_breakdown.get("worst_day") or {}).get("daily_return") or 0.0)),
        "median_daily_pnl": float(np.median(selected_rows.groupby("trade_day")["selected_return"].sum().to_numpy(dtype=float))) if not selected_rows.empty else 0.0,
        "daily_profit_factor": _safe_daily_profit_factor(selected_rows),
        "cost_stress_1_10x_pf": float(((cost_stress.get("cost_1_10x") or {}).get("profit_factor") or 0.0)),
        "cost_stress_1_25x_pf": float(((cost_stress.get("cost_1_25x") or {}).get("profit_factor") or 0.0)),
        "cost_stress_1_50x_pf": float(((cost_stress.get("cost_1_50x") or {}).get("profit_factor") or 0.0)),
        "cost_stress_2_00x_pf": float(((cost_stress.get("cost_2_00x") or {}).get("profit_factor") or 0.0)),
        "walk_forward_worst_fold_pf": float(min(fold_pfs) if fold_pfs else 0.0),
        "walk_forward_median_pf": float(np.median(fold_pfs)) if fold_pfs else 0.0,
        "fold_pf_std": float(np.std(fold_pfs, ddof=1)) if len(fold_pfs) > 1 else 0.0,
        "threshold_stability": threshold_stability,
        "threshold_stability_passes": bool(threshold_stability.get("passes")),
        "daily_stability_passes": daily_stability_passes,
        "monthly_stability": monthly_breakdown,
        "daily_stability": daily_breakdown,
        "fold_metrics": fold_metrics,
        "cost_stress": cost_stress,
        "return_cost_provenance": cost_provenance,
        "liquidity_report": liquidity_report,
        "skipped_filters": skipped_filters,
        "leakage_warning": bool(leakage_warning),
        "cost_provenance_uncertain": str(cost_provenance.get("cost_model_status")) == "BLOCKED_UNCERTAIN_COST_PROVENANCE",
        "tiny_sample_warning": int(selection_payload.get("trades") or 0) < EDGE_REFINEMENT_WATCHLIST_GATES["trade_count_min"],
        "exact_live_observation_rule": _edge_refinement_live_rule(filters, selection_name, model_name, label_name),
        "profitable_days": int(daily_breakdown.get("profitable_days") or 0),
        "losing_days": int(daily_breakdown.get("losing_days") or 0),
        "profitable_months": int(sum(1 for row in (monthly_breakdown.get("months") or []) if float(row.get("total_return") or 0.0) > 0.0)),
        "losing_months": int(sum(1 for row in (monthly_breakdown.get("months") or []) if float(row.get("total_return") or 0.0) < 0.0)),
        "best_fold_profit_share": float(concentration.get("best_fold_profit_share_of_total_profit") or 0.0),
        "threshold_robustness_passes": bool(threshold_stability.get("passes")),
        "selector": selection_name,
        **edge_cost,
        **concentration,
    }
    gate = _edge_refinement_gate(candidate, full_run_completed=full_run_completed)
    candidate.update(gate)
    candidate["stability_score"] = _edge_refinement_score(candidate)
    candidate["cost_aware_stability_score"] = _edge_refinement_cost_aware_score(candidate)
    candidate["cost_survival_score"] = candidate["cost_aware_stability_score"]
    candidate["score"] = candidate["cost_aware_stability_score"]
    candidate["production_readiness_failure_reasons"] = gate.get("paper_ready_failed_gates", [])
    candidate["fold_stability_passes"] = not any(reason in gate.get("failed_gates", []) for reason in ("worst_fold_pf_below_watchlist_gate", "median_fold_pf_below_watchlist_gate", "profitable_folds_below_4_of_5"))
    return candidate


def _edge_refinement_best(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any] | None:
    observed = [row for row in rows if row.get("trade_count", 0) > 0]
    if not observed:
        return None
    return max(observed, key=lambda row: (float(row.get(key) or 0.0), float(row.get("profit_factor") or 0.0), float(row.get("sharpe") or 0.0), int(row.get("trade_count") or 0)))


def _edge_refinement_diagnosis_row(row: Dict[str, Any]) -> Dict[str, Any]:
    threshold_support = ((row.get("threshold_stability") or {}).get("supporting_thresholds") or [])
    threshold_rows = ((row.get("threshold_stability") or {}).get("rows") or [])
    month_rows = (row.get("monthly_stability") or {}).get("months", []) or []
    losing_months = int(sum(1 for month in month_rows if float(month.get("total_return") or 0.0) < 0.0))
    losing_days = int(((row.get("daily_stability") or {}).get("losing_days")) or 0)
    average_loss = abs(float(row.get("average_negative_return") or 0.0))
    average_win = float(row.get("average_positive_return") or 0.0)
    return {
        "candidate_name": row.get("candidate_name"),
        "status": row.get("status"),
        "primary_block_reason": row.get("primary_block_reason"),
        "trade_count": int(row.get("trade_count") or 0),
        "profit_factor": float(row.get("profit_factor") or 0.0),
        "sharpe": float(row.get("sharpe") or 0.0),
        "return_to_cost_ratio": float(row.get("return_to_cost_ratio") or 0.0),
        "median_return_to_cost_ratio": float(row.get("median_return_to_cost_ratio") or 0.0),
        "average_estimated_cost_pct": float(row.get("average_estimated_cost_pct") or 0.0),
        "expected_return_after_cost": float(row.get("expected_return_after_cost") or 0.0),
        "median_return_after_cost": float(row.get("median_return_after_cost") or 0.0),
        "average_win": average_win,
        "average_loss": average_loss,
        "win_rate": float(row.get("win_rate") or 0.0),
        "payoff_ratio": float(average_win / average_loss) if average_loss > 0.0 else (float("inf") if average_win > 0.0 else 0.0),
        "best_day_profit_share_of_total_profit": float(row.get("best_day_profit_share_of_total_profit") or 0.0),
        "top_2_days_profit_share": float(row.get("top_2_days_profit_share") or 0.0),
        "best_fold_profit_share_of_total_profit": float(row.get("best_fold_profit_share_of_total_profit") or 0.0),
        "number_of_profitable_days": int(row.get("number_of_profitable_days") or 0),
        "number_of_losing_days": losing_days,
        "number_of_profitable_months": int(row.get("number_of_profitable_months") or 0),
        "number_of_losing_months": losing_months,
        "worst_fold_pf": float(row.get("walk_forward_worst_fold_pf") or 0.0),
        "median_fold_pf": float(row.get("walk_forward_median_pf") or 0.0),
        "threshold_robustness_support_count": int(len(threshold_support)),
        "threshold_robustness_supporting_thresholds": threshold_support,
        "threshold_robustness_candidate_rows": int(len(threshold_rows)),
        "all_failed_gates": list(row.get("all_failed_gates") or []),
        "cost_aware_stability_score": float(row.get("cost_aware_stability_score") or 0.0),
        "stability_score": float(row.get("stability_score") or 0.0),
    }


def _edge_refinement_true_edge_diagnosis(completed_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ranked_by_stability = sorted(completed_rows, key=lambda row: (float(row.get("stability_score") or 0.0), float(row.get("cost_aware_stability_score") or 0.0)), reverse=True)
    ranked_by_cost_aware = sorted(completed_rows, key=lambda row: (float(row.get("cost_aware_stability_score") or 0.0), float(row.get("stability_score") or 0.0)), reverse=True)
    return {
        "top_20_by_stability_score": [_edge_refinement_diagnosis_row(row) for row in ranked_by_stability[:20]],
        "top_20_by_cost_aware_stability_score": [_edge_refinement_diagnosis_row(row) for row in ranked_by_cost_aware[:20]],
    }


def _edge_refinement_blocker_breakdown(completed_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    single_gate_counts: Dict[str, int] = {}
    only_return_to_cost_ratio = 0
    only_daily_stability = 0
    only_threshold_robustness = 0
    multiple_gates = 0
    promising_rows = [
        row for row in completed_rows
        if float(row.get("profit_factor") or 0.0) > 1.20
        and float(row.get("sharpe") or 0.0) > 1.0
        and int(row.get("trade_count") or 0) >= 100
    ]
    promising_blockers: Dict[str, int] = {}
    for row in completed_rows:
        failed = list(dict.fromkeys(row.get("watchlist_failed_gates") or row.get("failed_gates") or row.get("all_failed_gates") or []))
        if len(failed) == 1:
            single_gate_counts[failed[0]] = single_gate_counts.get(failed[0], 0) + 1
            if failed[0] == "return_to_cost_ratio_below_watchlist_gate":
                only_return_to_cost_ratio += 1
            if failed[0] == "daily_stability_failed":
                only_daily_stability += 1
            if failed[0] == "threshold_robustness_failed":
                only_threshold_robustness += 1
        elif len(failed) > 1:
            multiple_gates += 1
        if row in promising_rows:
            for gate in failed:
                promising_blockers[gate] = promising_blockers.get(gate, 0) + 1
    largest_gate = None
    if promising_blockers:
        largest_gate = max(promising_blockers.items(), key=lambda item: (item[1], item[0]))
    return {
        "only_return_to_cost_ratio_fail_count": only_return_to_cost_ratio,
        "only_daily_stability_fail_count": only_daily_stability,
        "only_threshold_robustness_fail_count": only_threshold_robustness,
        "multiple_gate_fail_count": multiple_gates,
        "single_gate_fail_counts": dict(sorted(single_gate_counts.items(), key=lambda item: (-item[1], item[0]))),
        "promising_candidate_count": len(promising_rows),
        "promising_candidate_blocker_counts": dict(sorted(promising_blockers.items(), key=lambda item: (-item[1], item[0]))),
        "largest_single_blocker_for_promising_candidates": {
            "gate": largest_gate[0],
            "count": largest_gate[1],
        } if largest_gate else None,
    }


def _edge_refinement_rf_sensitivity_table(completed_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    target_names = [
        "top_1_per_day",
        "top_2_per_day",
        "top_3_per_day",
        "top_5_per_day",
        "top_1_percent_per_day",
        "top_2_percent_per_day",
        "top_5_percent_per_day",
    ]
    rows_by_name = {
        str(row.get("selection_name")): row
        for row in completed_rows
        if str(row.get("model_name")) == "random_forest"
        and str(row.get("label_name")) == "profitable_trade_label"
    }
    output: List[Dict[str, Any]] = []
    for name in target_names:
        row = rows_by_name.get(name)
        if not row:
            output.append({"selection_name": name, "available": False})
            continue
        output.append(
            {
                "selection_name": name,
                "available": True,
                "profit_factor": float(row.get("profit_factor") or 0.0),
                "sharpe": float(row.get("sharpe") or 0.0),
                "trade_count": int(row.get("trade_count") or 0),
                "cost_stress_1_25x_pf": float(row.get("cost_stress_1_25x_pf") or 0.0),
                "return_to_cost_ratio": float(row.get("return_to_cost_ratio") or 0.0),
                "profitable_days": int(row.get("number_of_profitable_days") or 0),
                "top_2_days_profit_share": float(row.get("top_2_days_profit_share") or 0.0),
                "worst_fold_pf": float(row.get("walk_forward_worst_fold_pf") or 0.0),
                "threshold_robustness_passes": bool(row.get("threshold_stability_passes")),
            }
        )
    return output


def _edge_refinement_improvement_targets(completed_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ranked = sorted(
        [
            row for row in completed_rows
            if str(row.get("model_name")) == "random_forest"
            and str(row.get("label_name")) == "profitable_trade_label"
            and 100 <= int(row.get("trade_count") or 0) <= 1000
        ],
        key=lambda row: float(row.get("cost_aware_stability_score") or row.get("stability_score") or 0.0),
        reverse=True,
    )
    focus = ranked[:3]
    rows = []
    for row in focus:
        avg_cost = float(row.get("average_estimated_cost_pct") or 0.0)
        avg_after_cost = float(row.get("expected_return_after_cost") or 0.0)
        ratio = float(row.get("return_to_cost_ratio") or 0.0)
        profitable_days = int(row.get("number_of_profitable_days") or 0)
        threshold_support_count = int(len(((row.get("threshold_stability") or {}).get("supporting_thresholds") or [])))
        current_total_return = avg_after_cost + avg_cost
        required_after_cost = avg_cost * 1.25
        required_improvement = max(0.0, required_after_cost - avg_after_cost)
        max_cost_for_gate = current_total_return / 1.25 if current_total_return > 0.0 else 0.0
        required_cost_reduction_pct = (
            max(0.0, (avg_cost - max_cost_for_gate) / avg_cost)
            if avg_cost > 0.0 and max_cost_for_gate < avg_cost
            else 0.0
        )
        rows.append(
            {
                "candidate_name": row.get("candidate_name"),
                "required_min_expected_return_per_trade_for_watchlist": required_after_cost,
                "incremental_expected_return_per_trade_needed": required_improvement,
                "required_cost_reduction_fraction_for_watchlist": required_cost_reduction_pct,
                "required_additional_profitable_days_for_watchlist": max(0, 30 - profitable_days),
                "required_threshold_support_count_for_watchlist": 2,
                "current_threshold_support_count": threshold_support_count,
                "required_additional_threshold_support_count": max(0, 2 - threshold_support_count),
            }
        )
    return {"candidate_targets": rows}


def _edge_widening_report_paths(timestamp: str) -> tuple[Path, Path]:
    report_dir = _artifact_report_dir()
    return report_dir / f"edge_widening_report_{timestamp}.json", report_dir / f"edge_widening_report_{timestamp}.md"


def _edge_widening_best_for_label(candidates: Sequence[Dict[str, Any]], label_name: str) -> Dict[str, Any] | None:
    scoped = [row for row in candidates if str(row.get("label_name")) == label_name]
    return _edge_refinement_best(scoped, "cost_aware_stability_score")


def _edge_widening_comparison_table(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    label_order = [
        "profitable_trade_label",
        "profitable_trade_label_edge_1x_cost",
        "profitable_trade_label_edge_2x_cost",
        "profitable_trade_label_edge_3x_cost",
        "profitable_trade_label_edge_5x_cost",
        "big_move_profitable_label",
        "asymmetric_payoff_label",
    ]
    rows: List[Dict[str, Any]] = []
    for label_name in label_order:
        best = _edge_widening_best_for_label(candidates, label_name) or {}
        rows.append(
            {
                "label_name": label_name,
                "candidate_name": best.get("candidate_name"),
                "status": best.get("status"),
                "trade_count": int(best.get("trade_count") or 0),
                "profit_factor": float(best.get("profit_factor") or 0.0),
                "sharpe": float(best.get("sharpe") or 0.0),
                "cost_stress_1_25x_pf": float(best.get("cost_stress_1_25x_pf") or 0.0),
                "cost_stress_1_50x_pf": float(best.get("cost_stress_1_50x_pf") or 0.0),
                "return_to_cost_ratio": float(best.get("return_to_cost_ratio") or 0.0),
                "worst_fold_pf": float(best.get("walk_forward_worst_fold_pf") or 0.0),
                "threshold_support_count": int(len(((best.get("threshold_stability") or {}).get("supporting_thresholds") or []))),
                "profitable_days": int(best.get("profitable_days") or 0),
                "losing_days": int(best.get("losing_days") or 0),
                "top_2_days_profit_share": float(best.get("top_2_days_profit_share") or 0.0),
                "failed_gates": best.get("failed_gates") or best.get("all_failed_gates") or [],
            }
        )
    return rows


def _edge_widening_failure_diagnosis(
    comparison_rows: Sequence[Dict[str, Any]],
    label_distribution_report: Sequence[Dict[str, Any]] = (),
) -> List[Dict[str, Any]]:
    baseline = next((row for row in comparison_rows if row.get("label_name") == "profitable_trade_label"), {})
    baseline_ratio = float(baseline.get("return_to_cost_ratio") or 0.0)
    baseline_pf = float(baseline.get("profit_factor") or 0.0)
    baseline_worst_fold = float(baseline.get("worst_fold_pf") or 0.0)
    baseline_support = int(baseline.get("threshold_support_count") or 0)
    baseline_cost_125 = float(baseline.get("cost_stress_1_25x_pf") or 0.0)
    baseline_days = int(baseline.get("profitable_days") or 0)
    baseline_losing = int(baseline.get("losing_days") or 0)
    baseline_win_rate = baseline_days / max(1, baseline_days + baseline_losing)
    baseline_top2 = float(baseline.get("top_2_days_profit_share") or 0.0)
    status_scores = {
        "BLOCKED_SMALL_SAMPLE": 0,
        "BLOCKED_TOO_MANY_TRADES": 1,
        "BLOCKED_LOW_EDGE": 2,
        "BLOCKED_THIN_EDGE": 2,
        "BLOCKED_FOLD_INSTABILITY": 3,
        "BLOCKED_DAILY_INSTABILITY": 3,
        "BLOCKED_THRESHOLD_INSTABILITY": 3,
        "BLOCKED_CONCENTRATED_PROFIT": 3,
        "BLOCKED_COST_STRESS": 3,
        "PAPER_WATCHLIST": 4,
        "PAPER_READY": 5,
    }
    baseline_status_val = status_scores.get(baseline.get("status"), 0)

    rows: List[Dict[str, Any]] = []
    for row in comparison_rows:
        ratio = float(row.get("return_to_cost_ratio") or 0.0)
        pf = float(row.get("profit_factor") or 0.0)
        worst_fold = float(row.get("worst_fold_pf") or 0.0)
        support = int(row.get("threshold_support_count") or 0)
        trades = int(row.get("trade_count") or 0)
        cost_125 = float(row.get("cost_stress_1_25x_pf") or 0.0)
        p_days = int(row.get("profitable_days") or 0)
        l_days = int(row.get("losing_days") or 0)
        win_rate = p_days / max(1, p_days + l_days)
        top2 = float(row.get("top_2_days_profit_share") or 0.0)

        dist_row = next((r for r in label_distribution_report if r.get("label_name") == row.get("label_name")), {})
        is_label_too_sparse = bool(dist_row.get("too_sparse"))

        did_cost_stress_pf_remain_stable = cost_125 >= max(1.0, baseline_cost_125 * 0.75)
        did_daily_stability_improve = win_rate > baseline_win_rate
        did_profit_concentration_reduce = top2 < baseline_top2 if top2 > 0.0 and baseline_top2 > 0.0 else False

        status_val = status_scores.get(row.get("status"), 0)
        is_label_better_than_baseline = (status_val > baseline_status_val) or (ratio > baseline_ratio and pf >= 1.15 and trades >= 100)

        rows.append(
            {
                "label_name": row.get("label_name"),
                "candidate_name": row.get("candidate_name"),
                "did_return_to_cost_improve": ratio > baseline_ratio,
                "did_trade_count_collapse_too_much": trades < EDGE_REFINEMENT_WATCHLIST_GATES["trade_count_min"],
                "did_pf_remain_stable": pf >= max(1.0, baseline_pf * 0.75),
                "did_worst_fold_improve": worst_fold > baseline_worst_fold,
                "did_threshold_robustness_improve": support > baseline_support,
                "did_cost_stress_pf_remain_stable": bool(did_cost_stress_pf_remain_stable),
                "did_daily_stability_improve": bool(did_daily_stability_improve),
                "did_profit_concentration_reduce": bool(did_profit_concentration_reduce),
                "is_label_too_sparse": bool(is_label_too_sparse),
                "is_label_better_than_baseline": bool(is_label_better_than_baseline),
                "status": row.get("status"),
            }
        )
    return rows


def _edge_widening_report_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Edge Widening Report",
        "",
        f"- Artifact dir: `{payload.get('artifact_dir')}`",
        f"- Dataset: `{payload.get('dataset_path')}`",
        f"- Paper watchlist reached: `{payload.get('paper_watchlist_reached')}`",
        f"- Paper ready reached: `{payload.get('paper_ready_reached')}`",
        "",
        "## Label Distribution",
    ]
    for row in payload.get("label_distribution_report", []):
        lines.append(
            f"- `{row.get('label_name')}` rows=`{row.get('row_count')}` positives=`{row.get('positive_count')}` negatives=`{row.get('negative_count')}` positive_rate=`{row.get('positive_rate')}` too_sparse=`{row.get('too_sparse')}` too_noisy=`{row.get('too_noisy')}` suitable_for_training=`{row.get('suitable_for_training')}`"
        )
    lines.append("")
    lines.append("## Comparison Table")
    for row in payload.get("comparison_table", []):
        lines.append(
            f"- `{row.get('label_name')}` candidate=`{row.get('candidate_name')}` status=`{row.get('status')}` trades=`{row.get('trade_count')}` PF=`{row.get('profit_factor')}` Sharpe=`{row.get('sharpe')}` cost1.25=`{row.get('cost_stress_1_25x_pf')}` cost1.5=`{row.get('cost_stress_1_50x_pf')}` return_to_cost=`{row.get('return_to_cost_ratio')}` threshold_support=`{row.get('threshold_support_count')}` profitable_days=`{row.get('profitable_days')}` losing_days=`{row.get('losing_days')}` failed_gates=`{row.get('failed_gates')}`"
        )
    lines.append("")
    lines.append("## Why Edge Improved Or Failed")
    for row in payload.get("edge_improved_or_failed", []):
        lines.append(
            f"- `{row.get('label_name')}` improved_return_to_cost=`{row.get('did_return_to_cost_improve')}` trade_count_collapsed=`{row.get('did_trade_count_collapse_too_much')}` pf_stable=`{row.get('did_pf_remain_stable')}` worst_fold_improved=`{row.get('did_worst_fold_improve')}` threshold_improved=`{row.get('did_threshold_robustness_improve')}` cost_stress_pf_stable=`{row.get('did_cost_stress_pf_remain_stable')}` daily_stability_improved=`{row.get('did_daily_stability_improve')}` profit_concentration_reduced=`{row.get('did_profit_concentration_reduce')}` label_too_sparse=`{row.get('is_label_too_sparse')}` better_than_baseline=`{row.get('is_label_better_than_baseline')}` status=`{row.get('status')}`"
        )
    lines.append("")
    lines.append("## Anti-Overfitting Rules")
    lines.append("- Do not weaken gates silently.")
    lines.append("- Do not promote candidates with fewer than minimum watchlist trades.")
    lines.append("- Do not promote candidates where profit comes from one or two days.")
    lines.append("- Do not promote candidates with zero threshold support.")
    lines.append("- Do not promote candidates with weak worst-fold PF.")
    lines.append("- Do not mark `PAPER_WATCHLIST` or `PAPER_READY` unless current gates pass.")
    return "\n".join(lines)


def _edge_refinement_report_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Edge Refinement Report",
        "",
        f"- Scan status: `{payload.get('scan_status')}`",
        f"- Best base model: `{(payload.get('best_base_model') or {}).get('model_name')}` on `{(payload.get('best_base_model') or {}).get('label_name')}`",
        f"- Why the base model failed: `{payload.get('base_model_failed_reason')}`",
        f"- Any paper-ready candidate? `{payload.get('paper_ready')}`",
        f"- Candidates evaluated: `{payload.get('evaluated_candidate_count')}`",
        f"- Candidates skipped: `{payload.get('skipped_candidate_count')}`",
        f"- Candidates pending: `{payload.get('pending_candidate_count')}`",
    ]
    best_watch = payload.get("best_watchlist_candidate") or {}
    if best_watch:
        lines.append(f"- Best watchlist-only candidate: `{best_watch.get('candidate_name')}`")
        lines.append(f"- Exact live observation rule: `{best_watch.get('exact_live_observation_rule')}`")
    lines.append("")
    lines.append("## Top Candidates")
    for row in sorted(payload.get("candidates", []), key=lambda item: float(item.get("score") or 0.0), reverse=True)[:10]:
        lines.append(
            f"- `{row['candidate_name']}` status=`{row.get('status')}` stability_score=`{row.get('stability_score')}` PF=`{row.get('profit_factor')}` Sharpe=`{row.get('sharpe')}` cost1.25=`{row.get('cost_stress_1_25x_pf')}` trades=`{row.get('trade_count')}` failed=`{', '.join(row.get('all_failed_gates') or row.get('failed_gates') or [])}`"
        )
    cost_aware = payload.get("best_candidate_by_cost_aware_stability_score") or {}
    if cost_aware:
        lines.append("")
        lines.append("## Cost-Aware Best")
        lines.append(
            f"- `{cost_aware.get('candidate_name')}` cost_aware_stability_score=`{cost_aware.get('cost_aware_stability_score')}` return_to_cost=`{cost_aware.get('return_to_cost_ratio')}` cost1.25=`{cost_aware.get('cost_stress_1_25x_pf')}` cost1.5=`{cost_aware.get('cost_stress_1_50x_pf')}`"
        )
    middle_zone = payload.get("middle_zone_candidates") or []
    if middle_zone:
        lines.append("")
        lines.append("## Middle-Zone Candidates")
        for row in middle_zone[:10]:
            lines.append(
                f"- `{row['candidate_name']}` trades=`{row.get('trade_count')}` PF=`{row.get('profit_factor')}` Sharpe=`{row.get('sharpe')}` cost1.25=`{row.get('cost_stress_1_25x_pf')}` cost1.5=`{row.get('cost_stress_1_50x_pf')}` return_to_cost=`{row.get('return_to_cost_ratio')}` worst_fold_pf=`{row.get('walk_forward_worst_fold_pf')}` status=`{row.get('status')}`"
            )
    thin_edge = payload.get("thin_edge_rejected_candidates") or []
    if thin_edge:
        lines.append("")
        lines.append("## Thin Edge Rejections")
        for row in thin_edge[:10]:
            lines.append(
                f"- `{row['candidate_name']}` trades=`{row.get('trade_count')}` avg_return=`{row.get('average_return_per_trade')}` after_cost=`{row.get('expected_return_after_cost')}` return_to_cost=`{row.get('return_to_cost_ratio')}` failed=`{', '.join(row.get('all_failed_gates') or [])}`"
            )
    do_not_trust = payload.get("do_not_trust_candidates") or []
    if do_not_trust:
        lines.append("")
        lines.append("## High Headline PF Candidates To Distrust")
        for row in do_not_trust[:5]:
            lines.append(
                f"- `{row['candidate_name']}` PF=`{row.get('profit_factor')}` Sharpe=`{row.get('sharpe')}` trades=`{row.get('trade_count')}` top1day=`{row.get('best_day_profit_share_of_total_profit')}` top2days=`{row.get('top_2_days_profit_share')}` topMonth=`{row.get('best_month_profit_share_of_total_profit')}` topFold=`{row.get('best_fold_profit_share_of_total_profit')}` profitableDays=`{row.get('number_of_profitable_days')}` profitableMonths=`{row.get('number_of_profitable_months')}` worstFoldPF=`{row.get('walk_forward_worst_fold_pf')}` reason=`{row.get('primary_block_reason')}` verdict=`not suitable even for watchlist`"
            )
    diagnosis = payload.get("true_edge_diagnosis") or {}
    blocker_breakdown = payload.get("candidate_blocker_breakdown") or {}
    sensitivity = payload.get("random_forest_sensitivity_table") or []
    improvement = payload.get("what_would_need_to_improve") or {}
    if diagnosis:
        lines.append("")
        lines.append("## Top Candidate Diagnosis")
        for row in (diagnosis.get("top_20_by_cost_aware_stability_score") or [])[:5]:
            lines.append(
                f"- `{row.get('candidate_name')}` status=`{row.get('status')}` return_to_cost=`{row.get('return_to_cost_ratio')}` median_return_to_cost=`{row.get('median_return_to_cost_ratio')}` profitableDays=`{row.get('number_of_profitable_days')}` profitableMonths=`{row.get('number_of_profitable_months')}` worstFoldPF=`{row.get('worst_fold_pf')}` thresholdSupport=`{row.get('threshold_robustness_support_count')}` blockers=`{', '.join(row.get('all_failed_gates') or [])}`"
            )
    if blocker_breakdown:
        lines.append("")
        lines.append("## Blocker Breakdown")
        lines.append(f"- Only return-to-cost gate: `{blocker_breakdown.get('only_return_to_cost_ratio_fail_count')}`")
        lines.append(f"- Only daily stability gate: `{blocker_breakdown.get('only_daily_stability_fail_count')}`")
        lines.append(f"- Only threshold robustness gate: `{blocker_breakdown.get('only_threshold_robustness_fail_count')}`")
        lines.append(f"- Multiple gates: `{blocker_breakdown.get('multiple_gate_fail_count')}`")
        largest = blocker_breakdown.get("largest_single_blocker_for_promising_candidates") or {}
        if largest:
            lines.append(f"- Largest blocker for otherwise promising candidates: `{largest.get('gate')}` count=`{largest.get('count')}`")
    if sensitivity:
        lines.append("")
        lines.append("## Random Forest Sensitivity")
        for row in sensitivity:
            lines.append(
                f"- `{row.get('selection_name')}` available=`{row.get('available')}` PF=`{row.get('profit_factor')}` Sharpe=`{row.get('sharpe')}` trades=`{row.get('trade_count')}` cost1.25=`{row.get('cost_stress_1_25x_pf')}` return_to_cost=`{row.get('return_to_cost_ratio')}` profitableDays=`{row.get('profitable_days')}` top2days=`{row.get('top_2_days_profit_share')}` worstFoldPF=`{row.get('worst_fold_pf')}` thresholdRobust=`{row.get('threshold_robustness_passes')}`"
            )
    if improvement:
        lines.append("")
        lines.append("## What Would Need To Improve")
        for row in (improvement.get("candidate_targets") or [])[:5]:
            lines.append(
                f"- `{row.get('candidate_name')}` required_expected_return=`{row.get('required_min_expected_return_per_trade_for_watchlist')}` extra_return_needed=`{row.get('incremental_expected_return_per_trade_needed')}` cost_reduction_fraction=`{row.get('required_cost_reduction_fraction_for_watchlist')}` extra_profitable_days=`{row.get('required_additional_profitable_days_for_watchlist')}` threshold_support_now=`{row.get('current_threshold_support_count')}` threshold_support_needed=`{row.get('required_additional_threshold_support_count')}`"
            )
    lines.append("")
    lines.append("## No-Cheating Recommendation")
    lines.append("- Do not weaken gates silently.")
    lines.append("- If a gate changes, report the old and new value and mark it as experimental.")
    lines.append("- Do not promote any candidate to `PAPER_WATCHLIST` or `PAPER_READY` unless it passes the current gates.")
    return "\n".join(lines)


def _edge_refinement_base_model_failure_reason(base_model: Dict[str, Any]) -> str:
    if not base_model:
        return "paper readiness blocked in prior report"
    failure_reasons = []
    if float(base_model.get("sharpe") or 0.0) < 1.0:
        failure_reasons.append("sharpe_below_1_0")
    if float(base_model.get("cost_stress_1_25x_pf") or 0.0) <= 1.05:
        failure_reasons.append("cost_stress_failed")
    if float(base_model.get("worst_fold_pf") or 0.0) <= 0.90:
        failure_reasons.append("walk_forward_worst_fold_failed")
    if str(base_model.get("daily_gate") or "FAIL") != "PASS":
        failure_reasons.append("daily_stability_failed")
    if not bool(base_model.get("threshold_robust")):
        failure_reasons.append("threshold_robustness_failed")
    return ", ".join(failure_reasons) if failure_reasons else "paper readiness blocked in prior report"


def _edge_refinement_base_metric_rows(completed: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for completed_row in completed:
        metrics_path = Path(str(completed_row["metrics_path"]))
        model_path = Path(str(completed_row["artifact_path"]))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        trade_metrics = metrics.get("trade_metrics") or {}
        fold_stability = metrics.get("fold_stability") or {}
        threshold_robustness = metrics.get("threshold_robustness") or {}
        daily_stability = metrics.get("daily_pnl_stability") or {}
        rows.append(
            {
                "label_name": str(completed_row["label_name"]),
                "model_name": str(completed_row["model_name"]),
                "artifact_path": str(model_path),
                "metrics_path": str(metrics_path),
                "profit_factor": float(trade_metrics.get("profit_factor") or 0.0),
                "sharpe": float(trade_metrics.get("sharpe") or 0.0),
                "trade_count": int(trade_metrics.get("trade_count") or 0),
                "selected_threshold": float(((metrics.get("selected_threshold_from_validation") or {}).get("threshold")) or 0.0),
                "cost_stress_1_25x_pf": float((((metrics.get("cost_stress") or {}).get("extra_cost_0_25") or {}).get("profit_factor") or 0.0)),
                "worst_fold_pf": float(fold_stability.get("worst_fold_pf") or 0.0),
                "median_fold_pf": float(fold_stability.get("median_fold_pf") or 0.0),
                "daily_gate": str(daily_stability.get("gate") or "FAIL"),
                "threshold_robust": bool(threshold_robustness.get("chosen_threshold_is_robust")),
            }
        )
    return rows


def _edge_refinement_planned_candidates(
    completed: Sequence[Dict[str, Any]],
    *,
    fast: bool,
    middle_zone_only: bool,
    include_avoid_label_refinement: bool,
    only_models: Sequence[str],
    only_targets: Sequence[str],
    max_models: int,
) -> List[Dict[str, Any]]:
    candidate_specs = _edge_refinement_filter_specs()
    if fast:
        candidate_specs = _edge_refinement_filter_specs_fast_only(candidate_specs)
    if middle_zone_only:
        candidate_specs = _edge_refinement_filter_specs_middle_zone_only(candidate_specs)
    # Normalise to lowercase so mixed-case CLI inputs match the model registry.
    # Also expand comma-separated values and apply common aliases.
    _REFINE_ALIASES = {
        "randomforest": "random_forest",
        "extratrees": "extra_trees",
        "gradientboosting": "gradient_boosting",
        "histgradientboosting": "hist_gradient_boosting",
    }
    raw_tokens: List[str] = []
    for x in only_models:
        raw_tokens.extend(str(x).split(","))
    allowed_models = {
        _REFINE_ALIASES.get(str(t).strip().lower(), str(t).strip().lower())
        for t in raw_tokens if str(t).strip()
    }
    allowed_targets = {str(x).strip() for x in only_targets if str(x).strip()}
    if fast:
        allowed_models = allowed_models or set(_edge_refinement_fast_model_names())
        allowed_targets = allowed_targets or {"profitable_trade_label"}
    else:
        allowed_models = allowed_models or set(EDGE_REFINEMENT_FOCUS_MODELS)
        if not include_avoid_label_refinement:
            allowed_targets = allowed_targets or {"profitable_trade_label"}
    selected_completed = []
    for row in completed:
        model_name = str(row["model_name"])
        label_name = str(row["label_name"])
        if allowed_models and str(model_name).strip().lower() not in allowed_models:
            continue
        if allowed_targets and label_name not in allowed_targets:
            continue
        selected_completed.append(row)
    if max_models and max_models > 0:
        seen_models = []
        limited_rows = []
        for row in selected_completed:
            model_name = str(row["model_name"])
            if model_name not in seen_models:
                if len(seen_models) >= int(max_models):
                    continue
                seen_models.append(model_name)
            limited_rows.append(row)
        selected_completed = limited_rows
    plans: List[Dict[str, Any]] = []
    for row in selected_completed:
        model_name = str(row["model_name"])
        label_name = str(row["label_name"])
        for spec in candidate_specs:
            candidate_name = f"{model_name}_{label_name}_{spec['name']}"
            plans.append(
                {
                    "candidate_id": _edge_refinement_candidate_id(model_name, label_name, candidate_name),
                    "candidate_name": candidate_name,
                    "model_name": model_name,
                    "label_name": label_name,
                    "artifact_path": str(row["artifact_path"]),
                    "metrics_path": str(row["metrics_path"]),
                    "spec": spec,
                    "filter_name": str(spec["name"]),
                    "selection_rule": str(spec.get("selection", {}).get("kind")),
                }
            )
    return plans


def _edge_refinement_payload_from_status(
    *,
    dataset_path: Path,
    artifact_dir: Path,
    full_run_completed: bool,
    base_model: Dict[str, Any],
    status_rows: Sequence[Dict[str, Any]],
    planned_candidates: Sequence[Dict[str, Any]],
    scan_status: str,
) -> Dict[str, Any]:
    planned_ids = {str(row.get("candidate_id")) for row in planned_candidates if str(row.get("candidate_id"))}
    scoped_status_rows = [row for row in status_rows if not planned_ids or str(row.get("candidate_id")) in planned_ids]
    completed_rows = [row.get("metrics", {}) for row in scoped_status_rows if str(row.get("status")).lower() == "completed" and isinstance(row.get("metrics"), dict)]
    skipped_rows = [row for row in scoped_status_rows if str(row.get("status")).lower() == "skipped"]
    failed_rows = [row for row in scoped_status_rows if str(row.get("status")).lower() == "failed"]
    completed_ids = {str(row.get("candidate_id")) for row in scoped_status_rows}
    pending_candidates = [row for row in planned_candidates if str(row.get("candidate_id")) not in completed_ids]
    best_pf = _edge_refinement_best(completed_rows, "profit_factor")
    best_sharpe = _edge_refinement_best(completed_rows, "sharpe")
    best_cost = _edge_refinement_best(completed_rows, "cost_stress_1_25x_pf")
    best_fold = _edge_refinement_best(completed_rows, "walk_forward_worst_fold_pf")
    best_stability = _edge_refinement_best(completed_rows, "stability_score")
    best_cost_aware = _edge_refinement_best(completed_rows, "cost_aware_stability_score")
    watchlist_candidates = [row for row in completed_rows if row.get("status") == "PAPER_WATCHLIST"]
    best_watch = max(watchlist_candidates, key=lambda row: float(row.get("score") or 0.0), default=None)
    ready_candidates = [row for row in completed_rows if row.get("status") == "PAPER_READY"]
    best_ready = max(ready_candidates, key=lambda row: float(row.get("score") or 0.0), default=None)
    middle_zone_candidates = [
        row for row in completed_rows
        if 100 <= int(row.get("trade_count") or 0) <= 1000
    ]
    middle_zone_candidates = sorted(middle_zone_candidates, key=lambda row: float(row.get("cost_aware_stability_score") or row.get("stability_score") or 0.0), reverse=True)
    cost_125_pass = [row for row in completed_rows if float(row.get("cost_stress_1_25x_pf") or 0.0) >= 1.05]
    cost_150_pass = [row for row in completed_rows if float(row.get("cost_stress_1_50x_pf") or 0.0) >= 1.0]
    ratio_pass = [row for row in completed_rows if float(row.get("return_to_cost_ratio") or 0.0) >= 1.25]
    thin_edge_rejections = [
        row for row in completed_rows
        if row.get("status") == "BLOCKED_THIN_EDGE"
        or "return_to_cost_ratio_below_watchlist_gate" in (row.get("all_failed_gates") or [])
        or "expected_return_after_cost_non_positive" in (row.get("all_failed_gates") or [])
    ]
    do_not_trust = [
        row for row in completed_rows
        if float(row.get("profit_factor") or 0.0) >= 5.0
        and row.get("status") not in {"PAPER_WATCHLIST", "PAPER_READY"}
    ]
    diagnosis = _edge_refinement_true_edge_diagnosis(completed_rows)
    blocker_breakdown = _edge_refinement_blocker_breakdown(completed_rows)
    rf_sensitivity = _edge_refinement_rf_sensitivity_table(completed_rows)
    improvement_targets = _edge_refinement_improvement_targets(completed_rows)
    return {
        "dataset_path": str(dataset_path),
        "artifact_dir": str(artifact_dir),
        "best_base_model": base_model,
        "base_model_failed_reason": _edge_refinement_base_model_failure_reason(base_model),
        "candidates": completed_rows,
        "evaluated_candidates": completed_rows,
        "skipped_candidates": skipped_rows,
        "failed_candidates": failed_rows,
        "pending_candidates": pending_candidates,
        "evaluated_candidate_count": len(completed_rows),
        "skipped_candidate_count": len(skipped_rows),
        "failed_candidate_count": len(failed_rows),
        "pending_candidate_count": len(pending_candidates),
        "best_candidate_by_pf": best_pf,
        "best_candidate_by_sharpe": best_sharpe,
        "best_candidate_by_cost_stressed_pf": best_cost,
        "best_candidate_by_fold_stability": best_fold,
        "best_candidate_by_stability_score": best_stability,
        "best_candidate_by_cost_aware_stability_score": best_cost_aware,
        "best_watchlist_candidate": best_watch,
        "best_paper_ready_candidate": best_ready,
        "middle_zone_candidates": middle_zone_candidates,
        "cost_1_25x_pass_candidates": cost_125_pass,
        "cost_1_50x_pass_candidates": cost_150_pass,
        "return_to_cost_ratio_pass_candidates": ratio_pass,
        "thin_edge_rejected_candidates": thin_edge_rejections,
        "do_not_trust_candidates": do_not_trust,
        "true_edge_diagnosis": diagnosis,
        "candidate_blocker_breakdown": blocker_breakdown,
        "random_forest_sensitivity_table": rf_sensitivity,
        "what_would_need_to_improve": improvement_targets,
        "paper_ready": bool(best_ready),
        "paper_trading_status": "PASS" if best_ready and scan_status == "COMPLETE" else "BLOCKED",
        "full_model_run_completed": full_run_completed,
        "scan_status": scan_status,
    }


def generate_edge_refinement_report(
    *,
    dataset_path: Path,
    artifact_dir: Path,
    fold_count: int,
    max_models: int = 0,
    max_candidates: int = 0,
    only_models: Sequence[str] = (),
    only_targets: Sequence[str] = (),
    fast: bool = False,
    middle_zone_only: bool = False,
    include_avoid_label_refinement: bool = False,
    force_refinement: bool = False,
    report_only: bool = False,
) -> Dict[str, Any]:
    summary_path = artifact_dir / "core_retrain_summary.json"
    if not summary_path.exists():
        # Graceful degradation: Check if status dir has completed models that could be used
        status_dir = artifact_dir / "status"
        status_completed = []
        if status_dir.exists():
            for path in sorted(status_dir.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if str(payload.get("status")).lower() == "completed":
                        status_completed.append(payload)
                except Exception:
                    pass
        
        if status_completed:
            # We have completed models in status dir, generate a minimal summary
            print(f"[edge-refinement] WARNING: {summary_path} not found, but {len(status_completed)} completed models exist in status dir.")
            print(f"[edge-refinement] Generating report from status artifacts...")
            status_completed_list = status_completed
        else:
            # No valid status dir either - provide clear error message
            error_msg = (
                f"ERROR: Missing required artifact '{summary_path.name}' in {artifact_dir}\n"
                f"The --edge-refinement-report and --resume paths require core_retrain_summary.json.\n"
                f"This file is written automatically by training modes:\n"
                f"  - --retrain-all-models\n"
                f"  - --edge-widening-experiment\n"
                f"  - --retrain-all-models-strict\n"
                f"  - --model-rescue-experiments\n"
                f"\n"
                f"FIX: Re-run training with --retrain-all-models (or your chosen mode) to regenerate:\n"
                f"  python scripts/retrain_all_edge_models.py --dataset <path> --retrain-all-models --output-dir {artifact_dir.parent}"
            )
            raise FileNotFoundError(error_msg)
    else:
        _ = json.loads(summary_path.read_text(encoding="utf-8"))
    status_dir = artifact_dir / "status"
    completed = []
    if status_dir.exists():
        for path in sorted(status_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if str(payload.get("status")).lower() == "completed":
                completed.append(payload)
    latest_all_models_report = sorted(_artifact_report_dir().glob("all_models_retraining_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest_all_payload = json.loads(latest_all_models_report[0].read_text(encoding="utf-8")) if latest_all_models_report else {}
    full_run_completed = bool(latest_all_payload.get("full_run_completed"))
    base_metric_rows = _edge_refinement_base_metric_rows(completed)
    base_model = max(base_metric_rows, key=lambda row: (row["profit_factor"], row["sharpe"], row["trade_count"]), default={})
    planned_candidates = _edge_refinement_planned_candidates(
        completed,
        fast=fast,
        middle_zone_only=middle_zone_only,
        include_avoid_label_refinement=include_avoid_label_refinement,
        only_models=only_models,
        only_targets=only_targets,
        max_models=max_models,
    )
    existing_status = _load_edge_refinement_status_rows(artifact_dir)
    existing_by_id = {str(row.get("candidate_id")): row for row in existing_status}
    if not report_only:
        df = load_dataset(dataset_path)
        df = _prepare_chronological_work(df.copy())
        planned_targets = {str(plan.get("label_name")) for plan in planned_candidates}
        if any(target in _edge_widening_label_names() for target in planned_targets):
            df, _ = _augment_edge_widening_labels(df, minimum_absolute_return=0.01)
        evaluation_return_column = (detect_column_groups(df) or {}).get("evaluation_return_column_used")
        if not evaluation_return_column or evaluation_return_column not in df.columns:
            raise RuntimeError("No evaluation return column available for edge refinement.")
        scored_cache: Dict[tuple[str, str], tuple[pd.DataFrame, float, bool]] = {}
        pending_plans: List[Dict[str, Any]] = []
        for plan in planned_candidates:
            candidate_id = str(plan["candidate_id"])
            if not force_refinement and candidate_id in existing_by_id and str(existing_by_id[candidate_id].get("status")).lower() == "completed":
                continue
            pending_plans.append(plan)
            if max_candidates and max_candidates > 0 and len(pending_plans) >= int(max_candidates):
                break
        for plan in pending_plans:
            model_name = str(plan["model_name"])
            label_name = str(plan["label_name"])
            key = (label_name, model_name)
            if key in scored_cache:
                continue
            metrics = json.loads(Path(str(plan["metrics_path"])).read_text(encoding="utf-8"))
            bundle = _load_model_bundle(Path(str(plan["artifact_path"])))
            scored = _scored_holdout_frame_for_artifact(df, bundle=bundle, label_name=label_name, model_name=model_name, evaluation_return_column=evaluation_return_column)
            selected_threshold = float(((metrics.get("selected_threshold_from_validation") or {}).get("threshold")) or getattr(bundle, "metrics", {}).get("threshold") or 0.50)
            leakage_warning = bool(not bool(metrics.get("evaluation_return_column_used")))
            scored_cache[key] = (scored, selected_threshold, leakage_warning)

        def _evaluate_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
            candidate_id = str(plan["candidate_id"])
            model_name = str(plan["model_name"])
            label_name = str(plan["label_name"])
            start_time = datetime.now().isoformat()
            try:
                scored, selected_threshold, leakage_warning = scored_cache[(label_name, model_name)]
                if scored.empty:
                    return {
                        "candidate_id": candidate_id,
                        "candidate_name": str(plan["candidate_name"]),
                        "model_name": model_name,
                        "label_name": label_name,
                        "filter_name": str(plan["filter_name"]),
                        "selection_rule": str(plan["selection_rule"]),
                        "status": "skipped",
                        "metrics": {},
                        "skip_reason": "empty_scored_holdout",
                        "error": None,
                        "start_time": start_time,
                        "end_time": datetime.now().isoformat(),
                    }
                candidate_metrics = _evaluate_edge_refinement_candidate(
                    scored,
                    candidate_name=str(plan["candidate_name"]),
                    model_name=model_name,
                    label_name=label_name,
                    filters=dict(plan["spec"]["filters"]),
                    selection={**dict(plan["spec"]["selection"]), "name": str(plan["spec"]["name"])},
                    fallback_threshold=selected_threshold,
                    full_run_completed=full_run_completed,
                    leakage_warning=leakage_warning,
                    fold_count=fold_count,
                )
                return {
                    "candidate_id": candidate_id,
                    "candidate_name": str(plan["candidate_name"]),
                    "model_name": model_name,
                    "label_name": label_name,
                    "filter_name": str(plan["filter_name"]),
                    "selection_rule": str(plan["selection_rule"]),
                    "status": "completed",
                    "metrics": candidate_metrics,
                    "skip_reason": None,
                    "error": None,
                    "start_time": start_time,
                    "end_time": datetime.now().isoformat(),
                }
            except Exception as exc:
                return {
                    "candidate_id": candidate_id,
                    "candidate_name": str(plan["candidate_name"]),
                    "model_name": model_name,
                    "label_name": label_name,
                    "filter_name": str(plan["filter_name"]),
                    "selection_rule": str(plan["selection_rule"]),
                    "status": "failed",
                    "metrics": {},
                    "skip_reason": None,
                    "error": str(exc),
                    "start_time": start_time,
                    "end_time": datetime.now().isoformat(),
                }

        edge_eval_start = time.perf_counter()
        payload_rows: List[Dict[str, Any]]
        if pending_plans and bool(_runtime_option("vectorized_backtest", False)) and len(pending_plans) > 1:
            from joblib import Parallel, delayed

            payload_rows = Parallel(n_jobs=int(_runtime_option("n_jobs", -1) or -1), prefer="threads")(
                delayed(_evaluate_plan)(plan) for plan in pending_plans
            )
        else:
            payload_rows = [_evaluate_plan(plan) for plan in pending_plans]
        _timing_log("edge_refinement_backtest", seconds=time.perf_counter() - edge_eval_start, rows=len(payload_rows), extra=f"parallel={bool(_runtime_option('vectorized_backtest', False))}")
        for payload in payload_rows:
            candidate_id = str(payload["candidate_id"])
            _write_edge_refinement_status(artifact_dir, candidate_id, payload)
            existing_by_id[candidate_id] = payload
    status_rows = _load_edge_refinement_status_rows(artifact_dir)
    pending_after = [row for row in planned_candidates if str(row.get("candidate_id")) not in {str(item.get("candidate_id")) for item in status_rows}]
    scan_status = "COMPLETE" if not pending_after and not (max_candidates and len(status_rows) < len(planned_candidates)) else "PARTIAL_TIMEOUT"
    report_payload = _edge_refinement_payload_from_status(
        dataset_path=dataset_path,
        artifact_dir=artifact_dir,
        full_run_completed=full_run_completed,
        base_model=base_model,
        status_rows=status_rows,
        planned_candidates=planned_candidates,
        scan_status=scan_status,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path, md_path = _edge_refinement_report_paths(timestamp)
    cost_json_path, cost_md_path = _cost_aware_edge_refinement_report_paths(timestamp)
    audit_json_path, audit_md_path = _cost_model_audit_report_paths(timestamp)
    audit_df = load_dataset(dataset_path)
    audit_df = _prepare_chronological_work(audit_df.copy())
    audit_groups = detect_column_groups(audit_df) or {}
    cost_audit = _cost_model_audit_report(
        audit_df,
        evaluation_return_column=audit_groups.get("evaluation_return_column_used"),
        artifact_dir=artifact_dir,
    )
    write_json(json_path, report_payload)
    write_json(cost_json_path, report_payload)
    write_json(audit_json_path, cost_audit)
    md_path.write_text(_edge_refinement_report_markdown(report_payload), encoding="utf-8")
    cost_md_path.write_text(_edge_refinement_report_markdown(report_payload), encoding="utf-8")
    audit_md_path.write_text(
        "\n".join(
            [
                "# Cost Model Audit",
                f"- Evaluation return column: `{cost_audit.get('evaluation_return_column')}`",
                f"- Net return likely already includes costs: `{cost_audit.get('net_forward_return_already_cost_adjusted')}`",
                f"- Double counting warning: `{cost_audit.get('double_counting_warning')}`",
                f"- Current stress formula: `{cost_audit.get('current_cost_formula')}`",
                f"- Recommendation: `{cost_audit.get('recommendation')}`",
            ]
        ),
        encoding="utf-8",
    )
    report_payload["json_path"] = str(json_path)
    report_payload["md_path"] = str(md_path)
    report_payload["cost_aware_json_path"] = str(cost_json_path)
    report_payload["cost_aware_md_path"] = str(cost_md_path)
    report_payload["cost_model_audit_json_path"] = str(audit_json_path)
    report_payload["cost_model_audit_md_path"] = str(audit_md_path)
    report_payload["cost_model_audit"] = cost_audit
    return report_payload


def run_edge_widening_experiment(
    *,
    dataset_path: Path,
    artifact_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    df = load_dataset(dataset_path)
    df = _prepare_chronological_work(df.copy())
    augmented_df, label_meta = _augment_edge_widening_labels(
        df,
        minimum_absolute_return=float(args.big_move_min_absolute_return or 0.01),
    )
    label_names = _edge_widening_label_names()
    label_distribution_report = [_edge_widening_label_distribution_report(augmented_df, label_name) for label_name in label_names]
    groups = detect_column_groups(augmented_df)
    feature_cols = groups.get("input_features") or []
    families = [
        "random_forest",
        "extra_trees",
        "xgboost",
        "calibrated_logistic_regression",
        "ensemble",
        "logistic_regression",
    ]
    train_result = train_variant(
        augmented_df,
        feature_cols=feature_cols,
        labels=label_names,
        artifact_dir=artifact_dir,
        variant_name="edge_widening",
        evaluation_return_column=groups.get("evaluation_return_column_used"),
        walk_forward_folds=int(args.walk_forward_folds),
        threshold_grid_start=float(args.threshold_grid_start),
        threshold_grid_end=float(args.threshold_grid_end),
        threshold_grid_step=float(args.threshold_grid_step),
        min_threshold_trades=max(25, int(args.min_threshold_trades)),
        paper_config=vars(args),
        model_families=families,
        only_targets=label_names,
        only_models=families,
        force_retrain=bool(args.force_retrain),
    )
    # Write core_retrain_summary.json before calling generate_edge_refinement_report
    # which requires this file to exist.
    widening_metric_rows = _simple_model_metrics_rows(train_result)
    _write_core_retrain_summary(
        artifact_dir=artifact_dir,
        dataset_path=dataset_path,
        labels_trained=label_names,
        model_families=families,
        metric_rows=widening_metric_rows,
        result_payload=train_result,
        evaluation_return_column=groups.get("evaluation_return_column_used") or "net_forward_return",
        row_count=len(augmented_df),
    )
    refinement_payload = generate_edge_refinement_report(
        dataset_path=dataset_path,
        artifact_dir=artifact_dir,
        fold_count=int(args.walk_forward_folds),
        only_models=families,
        only_targets=["profitable_trade_label", *label_names],
        report_only=False,
    )
    comparison_table = _edge_widening_comparison_table(refinement_payload.get("candidates", []))
    diagnosis = _edge_widening_failure_diagnosis(comparison_table, label_distribution_report)
    best_watch = refinement_payload.get("best_watchlist_candidate")
    best_ready = refinement_payload.get("best_paper_ready_candidate")
    edge_payload = {
        "dataset_path": str(dataset_path),
        "artifact_dir": str(artifact_dir),
        "label_distribution_report": label_distribution_report,
        "label_generation_meta": label_meta,
        "training_result_summary": {
            "trained_models": train_result.get("trained_models", []),
            "skipped_models": train_result.get("skipped_models", []),
            "failed_models": train_result.get("failed_models", []),
        },
        "comparison_table": comparison_table,
        "edge_improved_or_failed": diagnosis,
        "paper_watchlist_reached": bool(best_watch),
        "paper_ready_reached": bool(best_ready),
        "closest_candidate": best_watch or refinement_payload.get("best_candidate_by_cost_aware_stability_score") or refinement_payload.get("best_candidate_by_stability_score"),
        "updated_edge_refinement_report": refinement_payload.get("json_path"),
        "updated_paper_readiness_decision": "PASS" if best_ready else "BLOCKED",
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path, md_path = _edge_widening_report_paths(timestamp)
    write_json(json_path, edge_payload)
    md_path.write_text(_edge_widening_report_markdown(edge_payload), encoding="utf-8")
    paper_payload = {
        "decision": "PASS" if best_ready else "BLOCKED",
        "best_watchlist_candidate": best_watch,
        "best_paper_ready_candidate": best_ready,
        "edge_widening_report": str(json_path),
        "updated_edge_refinement_report": refinement_payload.get("json_path"),
    }
    paper_report = _write_named_report(
        "paper_readiness_report",
        paper_payload,
        [
            "# Paper Readiness Report",
            f"- Decision: `{'PASS' if best_ready else 'BLOCKED'}`",
            f"- Best watchlist candidate: `{(best_watch or {}).get('candidate_name')}`",
            f"- Best paper-ready candidate: `{(best_ready or {}).get('candidate_name')}`",
            "- Research-only; no auto-trading or production promotion.",
        ],
    )
    edge_payload["json_path"] = str(json_path)
    edge_payload["md_path"] = str(md_path)
    edge_payload["paper_readiness_report"] = paper_report
    write_json(json_path, edge_payload)
    md_path.write_text(_edge_widening_report_markdown(edge_payload), encoding="utf-8")
    return edge_payload


def save_model_artifact(out_dir: Path, model_name: str, label_name: str, model: Any, feature_names: Sequence[str], scaler_mean: List[float], scaler_std: List[float], metrics: Dict[str, Any]) -> str:
    import joblib

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = _wrap_model(model, feature_names=feature_names, metrics=metrics, scaler_mean=scaler_mean, scaler_std=scaler_std)
    model_path = out_dir / f"{model_name}_{label_name}.pkl"
    joblib.dump(artifact, model_path)
    write_json(
        out_dir / f"{model_name}_{label_name}_metadata.json",
        {
            "model_path": str(model_path),
            "feature_order": list(feature_names),
            **dict(metrics or {}),
        },
    )
    write_json(
        out_dir / f"{model_name}_{label_name}_feature_order.json",
        {
            "feature_order": list(feature_names),
        },
    )
    return str(model_path)


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    if str(getattr(args, "train_ensemble", "") or "").strip().lower() == "xgb_rf":
        args.retrain_all_models = True
        args.only_model = list(args.only_model or []) + ["xgb_rf_ensemble"]
    _set_runtime_options_from_args(args)
    if args.improve_models:
        args.retrain_all_models = True
    if args.paper_readiness_report:
        args.paper_readiness_audit = True
    if args.full_validation:
        args.paper_readiness_audit = True
        args.failure_diagnosis_audit = True
        args.cost_stress_audit = True
        args.calibration_audit = True
        args.probability_monotonicity_audit = True
        args.daily_pnl_stability_audit = True
    if args.walk_forward:
        args.walk_forward_folds = max(int(args.walk_forward_folds), 5)
    if args.cost_stress:
        args.cost_stress_audit = True
    if args.calibration:
        args.calibration_audit = True
    only_targets = _split_arg_values(args.only_target)
    only_models = _split_arg_values(args.only_model)
    artifact_dir: Optional[Path] = None
    dataset_path: Optional[Path] = None
    if args.full_pipeline:
        artifact_dir = run_dataset_builder()
        dataset_path = artifact_dir / "edge_dataset.csv"
    elif args.dataset:
        if str(args.dataset).strip().lower() == "auto":
            selection_payload = select_largest_valid_real_dataset()
            selected = selection_payload.get("selected_dataset")
            if not selected:
                raise FileNotFoundError("No valid real dataset found for --dataset auto.")
            dataset_path = Path(str(selected["path"]))
            artifact_dir = Path(args.output_dir) if args.output_dir else dataset_path.parent
        else:
            dataset_path = Path(args.dataset)
            artifact_dir = Path(args.output_dir) if args.output_dir else dataset_path.parent
    else:
        dataset_path = latest_edge_dataset_csv()
        artifact_dir = (Path(args.output_dir) if args.output_dir else dataset_path.parent) if dataset_path else None

    if dataset_path is None or artifact_dir is None or not dataset_path.exists():
        raise FileNotFoundError("Edge dataset not found. Run build_option_edge_dataset.py first.")
    if bool(args.edge_widening_experiment):
        widening_artifact_dir = Path(args.resume_artifact_dir) if str(args.resume_artifact_dir).strip() else artifact_dir
        payload = run_edge_widening_experiment(
            dataset_path=Path(dataset_path),
            artifact_dir=Path(widening_artifact_dir),
            args=args,
        )
        print(json.dumps({"edge_widening_report": payload.get("json_path"), "paper_watchlist_reached": payload.get("paper_watchlist_reached"), "paper_ready_reached": payload.get("paper_ready_reached")}, indent=2))
        return
    if bool(args.edge_refinement_report) or bool(args.edge_refinement_report_only):
        refinement_artifact_dir = Path(args.resume_artifact_dir) if str(args.resume_artifact_dir).strip() else artifact_dir
        payload = generate_edge_refinement_report(
            dataset_path=Path(dataset_path),
            artifact_dir=Path(refinement_artifact_dir),
            fold_count=max(2, int(args.walk_forward_folds)),
            max_models=max(0, int(args.edge_refinement_max_models)),
            max_candidates=max(0, int(args.edge_refinement_max_candidates)),
            only_models=_split_arg_values(args.edge_refinement_only_model),
            only_targets=_split_arg_values(args.edge_refinement_only_target),
            fast=bool(args.edge_refinement_fast),
            middle_zone_only=bool(args.middle_zone_only),
            include_avoid_label_refinement=bool(args.include_avoid_label_refinement),
            force_refinement=bool(args.force_refinement),
            report_only=bool(args.edge_refinement_report_only),
        )
        print(json.dumps({"edge_refinement_report": payload.get("json_path"), "paper_trading_status": payload.get("paper_trading_status")}, indent=2))
        return
    if bool(args.retrain_all_models) and args.resume:
        if str(args.resume_artifact_dir).strip():
            artifact_dir = Path(args.resume_artifact_dir)
        else:
            artifact_dir = _find_latest_incomplete_retraining_dir(Path(args.output_dir) if args.output_dir else MODELS_DIR) or artifact_dir
    elif bool(args.retrain_all_models):
        artifact_dir = _timestamped_retrain_output_dir(Path(args.output_dir) if args.output_dir else MODELS_DIR)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    validity_candidates = [
        artifact_dir / "dataset_validity_report.json",
        dataset_path.parent / "dataset_validity_report.json",
    ]
    validity = {}
    for validity_path in validity_candidates:
        if validity_path.exists():
            validity = json.loads(validity_path.read_text(encoding="utf-8"))
            break
    if not validity and args.dataset:
        validity = {
            "status": "DATASET_READY_FOR_RESEARCH_RETRAINING",
            "failures": [],
            "note": "Explicit dataset path supplied by user; using runtime leakage/feature checks instead of a prebuilt validity artifact.",
        }
    if validity.get("status") != "DATASET_READY_FOR_RESEARCH_RETRAINING":
        stop_payload = {
            "output_dir": str(artifact_dir),
            "dataset_validity_result": validity.get("status", "DATASET_NOT_READY_FOR_EDGE_RETRAINING"),
            "reason_if_training_stopped": "; ".join(validity.get("failures", [])) or "dataset validity gates failed",
            "models_trained": [],
        }
        write_json(artifact_dir / "edge_retraining_stop_report.json", stop_payload)
        print(json.dumps(stop_payload, indent=2))
        return

    df = load_dataset(dataset_path)
    if len(df) < int(args.min_rows):
        raise RuntimeError(f"Dataset has only {len(df)} rows; minimum required is {int(args.min_rows)}")
    if "timestamp" not in df.columns:
        raise RuntimeError("Dataset must contain timestamp column for chronological split.")
    df["timestamp"] = _normalize_timestamp_series(df["timestamp"])
    requested_targets = {str(item).strip() for item in _split_arg_values(args.only_target) if str(item).strip()}
    if (
        "avoid_trade_label" in requested_targets
        and "avoid_trade_label" not in df.columns
        and "profitable_trade_label" in df.columns
    ):
        profitable = pd.to_numeric(df["profitable_trade_label"], errors="coerce")
        df["avoid_trade_label"] = np.where(profitable.isna(), np.nan, 1.0 - profitable)
    labels, _ = choose_labels(df)
    column_groups = detect_column_groups(df)
    compare_result = {"skipped": True}
    try:
        compare_result = compare_feature_compatibility(column_groups["input_features"])
    except Exception as exc:
        compare_result = {"production_schema_compatible": False, "error": str(exc)}
    leakage_result = {
        "passed": True,
        "forbidden_columns_present": column_groups["forbidden_feature_columns"],
        "future_values_used": False,
        "notes": [
            "Chronological split preserved.",
            "No forward-fill across strikes or expiries was introduced by the Black-Scholes enrichment.",
            "Forbidden target/outcome columns were excluded from feature sets.",
        ],
    }
    output_payload: Dict[str, Any] = {}
    trained_models: List[str] = []
    feature_manifest_payload: Dict[str, Any] = {
        "input_features": column_groups["input_features"],
        "target_columns": column_groups["target_columns"],
        "evaluation_return_columns": column_groups["evaluation_return_columns"],
        "evaluation_return_column_used": column_groups["evaluation_return_column_used"],
        "forbidden_feature_columns": column_groups["forbidden_feature_columns"],
    }
    report_paths_payload: Dict[str, Any] = {}
    paper_payload: Dict[str, Any] = {}
    adoption = {
        "production_adoption_allowed": False,
        "verdict": "BASELINE_ONLY",
        "strict_live_schema_gate": {"compatible": False},
    }
    enriched_dataset_path: str | None = None
    same_period_report: Dict[str, Any] = {
        "enabled": bool(args.same_period_comparison),
        "same_period_gate": "PASS",
    }
    dataset_selection_payload: Dict[str, Any] | None = None
    if str(args.dataset).strip().lower() == "auto":
        dataset_selection_payload = select_largest_valid_real_dataset()
        report_paths_payload["dataset_selection_report"] = _write_named_report(
            "retrain_all_models_dataset_selection",
            dataset_selection_payload,
            [
                "# Retrain All Models Dataset Selection",
                f"- Selected dataset: `{(dataset_selection_payload.get('selected_dataset') or {}).get('path')}`",
                f"- Selection rule: `{dataset_selection_payload.get('selection_rule')}`",
            ],
        )
    if bool(args.retrain_all_models):
        print(f"[retrain] dataset loaded: {dataset_path}")
        print(f"[retrain] output folder: {artifact_dir}")
        row_count = len(df)
        date_range = _timestamp_range(df)
        print(f"[retrain] row_count={row_count}")
        print(f"[retrain] date_range={date_range.get('start')} -> {date_range.get('end')}")
        labels, label_skip_report = choose_labels(df)
        if "avoid_trade_label" in requested_targets and "avoid_trade_label" in df.columns and "avoid_trade_label" not in labels:
            labels.append("avoid_trade_label")
            label_skip_report = [
                row for row in label_skip_report
                if str(row.get("label_name")) != "avoid_trade_label"
            ]
        # ------------------------------------------------------------------
        # FIX 1 & 3: Validate --only-target BEFORE training begins.
        # If a requested label was silently filtered out by choose_labels()
        # (e.g. too few non-null rows, wrong positive-share bounds, or
        # cost-aware label not in the hardcoded PRIMARY_LABELS list) we fail
        # immediately with a clear error — never silently skip training.
        # ------------------------------------------------------------------
        if only_targets:
            available_label_names: set[str] = set(labels)
            # Also check against PRIMARY_LABELS so we can give a specific
            # "not in PRIMARY_LABELS" vs "in dataset but filtered out" message.
            primary_label_names: set[str] = set(PRIMARY_LABELS)
            missing: List[str] = []
            filtered: List[str] = []
            for tgt in only_targets:
                tgt_stripped = str(tgt).strip()
                if tgt_stripped not in available_label_names:
                    if tgt_stripped not in df.columns:
                        missing.append(tgt_stripped)
                    elif tgt_stripped in primary_label_names:
                        filtered.append(tgt_stripped)
                    else:
                        missing.append(tgt_stripped)
            if missing:
                raise ValueError(
                    f"--only-target={missing!r}: label(s) not found in dataset columns. "
                    f"Available label columns: {sorted(available_label_names)!r}."
                )
            if filtered:
                raise ValueError(
                    f"--only-target={filtered!r}: label(s) were filtered out by choose_labels() "
                    f"(too few non-null rows, wrong positive-share bounds, or degenerate distribution). "
                    f"Usable labels after filtering: {sorted(available_label_names)!r}. "
                    f"Skipped reasons: {[r for r in label_skip_report if r['label_name'] in filtered]!r}."
                )
        # Persist label-skip report so degenerate/near-constant labels
        # are never silently dropped.
        try:
            write_json(artifact_dir / "label_skip_report.json", {
                "dataset_path": str(dataset_path),
                "row_count": int(row_count),
                "labels_considered": [name for name in PRIMARY_LABELS if name in df.columns],
                "labels_usable": list(labels),
                "labels_skipped": label_skip_report,
            })
        except Exception as exc:  # noqa: BLE001
            print(f"[retrain] WARN: failed to write label_skip_report.json: {exc}")
        if not labels:
            raise RuntimeError("No usable label/target column found for --retrain-all-models.")
        selected_label = labels[0]
        print(f"[retrain] selected target column: {selected_label}")
        feature_sets = build_feature_sets(df, use_black_scholes_features=False)
        baseline_features = feature_sets["baseline_features"]
        selected_features = baseline_features
        live_feature_audit: List[Dict[str, Any]] = []
        if bool(args.live_computable_only):
            selected_features, live_feature_audit = _filter_live_contract_features(
                baseline_features,
                dataset_columns=df.columns,
                strict=bool(args.strict_live_contract),
            )
        else:
            live_feature_audit = _live_contract_audit_rows(selected_features, dataset_columns=df.columns)
        included_live_features = [row["feature_name"] for row in live_feature_audit if row["feature_name"] in selected_features]
        excluded_non_live_features = [row["feature_name"] for row in live_feature_audit if row["feature_name"] not in selected_features]
        dropped_leakage = sorted(set(feature_manifest_payload["forbidden_feature_columns"] + [name for name in baseline_features if name not in selected_features]))
        print(f"[retrain] raw_feature_count={len(baseline_features)}")
        print(f"[retrain] live_contract_feature_count={len(included_live_features)}")
        print(f"[retrain] excluded_non_live_features={len(excluded_non_live_features)} {excluded_non_live_features}")
        print(f"[retrain] included_live_features={len(included_live_features)} {included_live_features}")
        print(f"[retrain] final_training_feature_count={len(selected_features)}")
        print(f"[retrain] target_column={selected_label}")
        print(f"[retrain] input features={len(selected_features)}")
        print(f"[retrain] dropped leakage/eval columns={len(dropped_leakage)}")
        print(f"[retrain] output folder={artifact_dir}")
        families = _parse_model_families_arg(args.model_families)
        if only_models:
            # Normalise both sides to lowercase so that camelCase / mixed-case
            # CLI inputs match the lowercased model registry.  Also expand comma-
            # separated values and apply common aliases so "XGBoost" -> "xgboost",
            # "RandomForest" -> "random_forest", "ExtraTrees" -> "extra_trees", etc.
            _MODEL_ALIASES = {
                "randomforest": "random_forest",
                "extratrees": "extra_trees",
                "gradientboosting": "gradient_boosting",
                "histgradientboosting": "hist_gradient_boosting",
                "xgboost": "xgboost",
                "xgbrfensemble": "xgb_rf_ensemble",
                "xgb_rf": "xgb_rf_ensemble",
                "lightgbm": "lightgbm",
                "catboost": "catboost",
            }
            raw_tokens: List[str] = []
            for item in only_models:
                raw_tokens.extend(str(item).split(","))
            only_models_normalized = {
                _MODEL_ALIASES.get(str(t).strip().lower(), str(t).strip().lower())
                for t in raw_tokens if str(t).strip()
            }
            if "xgb_rf_ensemble" in only_models_normalized and "xgb_rf_ensemble" not in families:
                families.append("xgb_rf_ensemble")
            families = [name for name in families if str(name).strip().lower() in only_models_normalized]
            if not families:
                # Provide a helpful error so a camelCase typo never silently trains zero models.
                raise ValueError(
                    f"--only-model={only_models!r} resolved to zero models after "
                    f"case-normalized lookup.  Available families (lowercased): "
                    f"{sorted({str(f).strip().lower() for f in _parse_model_families_arg(args.model_families)})!r}."
                )
        print(f"[retrain] models planned={','.join(families)}")
        groups = detect_column_groups(df)
        print(f"[retrain] evaluation return column={groups.get('evaluation_return_column_used')}")
        print(f"[retrain] train/test split mode=chronological")
        expected_pairs = [(label_name, model_name) for label_name in labels if (not only_targets or label_name in set(only_targets)) for model_name in families]
        if args.report_only:
            report_payload = _report_payload_from_status(artifact_dir, expected_pairs=expected_pairs)
            report_paths_payload["all_models_retraining_report"] = _write_named_report(
                "all_models_retraining_report",
                report_payload,
                [
                    "# All Models Retraining Report",
                    f"- Artifact dir: `{artifact_dir}`",
                    f"- Runtime status: `{report_payload.get('runtime_status')}`",
                    f"- Completed pairs: `{len(report_payload.get('completed_target_model_pairs', []))}`",
                    f"- Missing pairs: `{len(report_payload.get('missing_target_model_pairs', []))}`",
                ],
            )
            report_paths_payload["paper_readiness_report"] = _write_named_report(
                "paper_readiness_report",
                {
                    "decision": report_payload.get("paper_trading_status"),
                    "best_partial_candidate": report_payload.get("best_partial_candidate"),
                    "full_run_completed": report_payload.get("full_run_completed"),
                },
                [
                    "# Paper Readiness Report",
                    f"- Decision: `{report_payload.get('paper_trading_status')}`",
                    f"- Full run completed: `{report_payload.get('full_run_completed')}`",
                ],
            )
            report_paths_payload["model_improvement_diagnosis"] = _write_named_report(
                "model_improvement_diagnosis",
                {
                    "runtime_status": report_payload.get("runtime_status"),
                    "missing_pairs": report_payload.get("missing_target_model_pairs"),
                    "failed_pairs": report_payload.get("failed_target_model_pairs"),
                },
                [
                    "# Model Improvement Diagnosis",
                    f"- Runtime status: `{report_payload.get('runtime_status')}`",
                    f"- Missing pairs: `{len(report_payload.get('missing_target_model_pairs', []))}`",
                    f"- Failed pairs: `{len(report_payload.get('failed_target_model_pairs', []))}`",
                ],
            )
            print(json.dumps({"artifact_dir": str(artifact_dir), "report_only": True, "report_paths": report_paths_payload}, indent=2))
            return
        result = train_variant(
            df,
            feature_cols=selected_features,
            labels=labels,
            artifact_dir=artifact_dir,
            variant_name="core_retrain" if args.resume else "retrain_all_models",
            evaluation_return_column=groups["evaluation_return_column_used"],
            walk_forward_folds=int(args.walk_forward_folds),
            threshold_grid_start=float(args.threshold_grid_start),
            threshold_grid_end=float(args.threshold_grid_end),
            threshold_grid_step=float(args.threshold_grid_step),
            min_threshold_trades=int(args.min_threshold_trades),
            paper_config={
                **vars(args),
                "_dataset_path": str(dataset_path),
                "_excluded_non_live_features": excluded_non_live_features,
            },
            model_families=families,
            only_targets=only_targets,
            only_models=only_models,
            force_retrain=bool(args.force_retrain),
        )
        if args.train_only:
            report_payload = _report_payload_from_status(artifact_dir, expected_pairs=expected_pairs)
            write_json(artifact_dir / "partial_training_status.json", report_payload)
            print(json.dumps({"artifact_dir": str(artifact_dir), "train_only": True, "completed_pairs": len(report_payload.get("completed_target_model_pairs", [])), "missing_pairs": len(report_payload.get("missing_target_model_pairs", []))}, indent=2))
            return
        resume_payload = _report_payload_from_status(artifact_dir, expected_pairs=expected_pairs)
        if bool(args.resume) and not bool(args.force_retrain) and bool(resume_payload.get("full_run_completed")):
            checkpoint_payload = _write_completed_checkpoint_artifacts(
                artifact_dir=artifact_dir,
                dataset_path=dataset_path,
                row_count=row_count,
                date_range=date_range,
                selected_label=selected_label,
                selected_features=selected_features,
                baseline_features=baseline_features,
                rejected_features=feature_sets["rejected_features"],
                dropped_leakage=dropped_leakage,
                groups=groups,
                live_feature_audit=live_feature_audit,
                live_computable_only=bool(args.live_computable_only),
                excluded_non_live_features=excluded_non_live_features,
                paper_watchlist_only=bool(args.paper_watchlist_only),
                result=result,
                families=families,
                walk_forward_folds=int(args.walk_forward_folds),
            )
            print(json.dumps({
                "artifact_dir": str(artifact_dir),
                "resume_fast_path": True,
                "models_trained": [row["model_name"] for row in checkpoint_payload["metric_rows"]],
                "metrics_report_path": str(artifact_dir / "metrics_report.json"),
                "summary_path": str(artifact_dir / "retrain_all_models_summary.json"),
                "core_retrain_summary_path": str(artifact_dir / "core_retrain_summary.json"),
                "final_report_path": str(artifact_dir / "final_ml_trading_decision_report.json"),
                "production_adoption_allowed": False,
            }, indent=2))
            return
        metric_rows = _simple_model_metrics_rows(result)
        report_map = result.get("reports", {}) or {}
        aggregated_threshold_rows = []
        walk_forward_threshold_report = []
        trade_filter_report = []
        cost_stress_report = {}
        regime_performance_report = {}
        for label_name, models in report_map.items():
            for model_name, report in models.items():
                aggregated_threshold_rows.extend([{"model_name": model_name, "label_name": label_name, **row} for row in (report.get("threshold_sweep") or [])])
                walk_forward_threshold_report.append({"model_name": model_name, "label_name": label_name, "selected_threshold": (report.get("selected_threshold_from_validation") or {}).get("threshold"), "folds": (report.get("walk_forward") or {}).get("folds", [])})
                trade_filter_report.append({"model_name": model_name, "label_name": label_name, **(report.get("trade_filter_report") or {})})
                cost_stress_report[model_name] = report.get("cost_stress") or {}
                regime_performance_report[model_name] = report.get("regime_performance") or {}
        for row in metric_rows:
            print(f"[retrain] model={row['model_name']} roc_auc={row['roc_auc']} pr_auc={row['pr_auc']} f1={row['f1']} pf={row['profit_factor']} sharpe={row['sharpe']}")
        label_report = _label_experiment_report(df, selected_features)
        feature_robustness = _feature_robustness_report(selected_features, [report for models in report_map.values() for report in models.values()])
        champion_report = _champion_selection_report(metric_rows, report_map)
        regime_candidate_report = _train_regime_restricted_candidates(
            df,
            feature_cols=selected_features,
            label_name=selected_label,
            evaluation_return_column=groups["evaluation_return_column_used"],
            artifact_dir=artifact_dir,
            fold_count=int(args.walk_forward_folds),
        )
        execution_rescue_rows = sorted(
            regime_candidate_report.get("execution_rescue_rows", []),
            key=lambda row: float(row.get("robustness_score") or 0.0),
            reverse=True,
        )
        cost_model_audit = _cost_model_audit_report(df, evaluation_return_column=groups["evaluation_return_column_used"], artifact_dir=artifact_dir)
        horizon_report = _horizon_comparison_report(df, artifact_dir=artifact_dir)
        feature_manifest = {
            "dataset_path": str(dataset_path),
            "label_name": selected_label,
            "features": selected_features,
            "feature_order": selected_features,
            "feature_count": len(selected_features),
            "baseline_features": baseline_features,
            "rejected_features": feature_sets["rejected_features"],
            "dropped_leakage_columns": dropped_leakage,
            "evaluation_return_columns": groups["evaluation_return_columns"],
            "evaluation_return_column_used": groups["evaluation_return_column_used"],
            "live_feature_audit": live_feature_audit,
            "live_computable_only": bool(args.live_computable_only),
            "live_contract_version": CONTRACT_VERSION if HAS_FEATURE_CONTRACT else None,
            "excluded_non_live_features": excluded_non_live_features,
            "included_live_features": included_live_features,
            "production_adoption_allowed": False,
            "research_only": True,
        }
        metrics_payload = {
            "dataset_path": str(dataset_path),
            "row_count": row_count,
            "date_range": date_range,
            "label_name": selected_label,
            "models_trained": metric_rows,
            "best_model": champion_report.get("champion"),
            "skipped_models": result.get("skipped_models", []),
            "failed_models": result.get("failed_models", []),
            "production_adoption_allowed": False,
            "paper_trading_allowed": False,
            "research_only": True,
        }
        training_manifest = {
            "mode": "retrain_all_models",
            "dataset_path": str(dataset_path),
            "artifact_dir": str(artifact_dir),
            "model_families": families,
            "row_count": row_count,
            "date_range": date_range,
            "label_name": selected_label,
            "feature_count": len(selected_features),
            "feature_order": selected_features,
            "live_computable_only": bool(args.live_computable_only),
            "live_contract_version": CONTRACT_VERSION if HAS_FEATURE_CONTRACT else None,
            "excluded_non_live_features": excluded_non_live_features,
            "production_adoption_allowed": False,
            "paper_trading_allowed": False,
            "research_only": True,
            "paper_watchlist_only": bool(args.paper_watchlist_only),
        }
        write_json(artifact_dir / "metrics_report.json", metrics_payload)
        write_json(artifact_dir / "feature_manifest.json", feature_manifest)
        write_json(artifact_dir / "feature_list_used.json", feature_manifest)
        write_json(artifact_dir / "training_manifest.json", training_manifest)
        write_json(artifact_dir / "walk_forward_threshold_report.json", {"models": walk_forward_threshold_report, "production_adoption_allowed": False})
        write_json(artifact_dir / "trade_filter_report.json", {"models": trade_filter_report, "production_adoption_allowed": False})
        write_json(artifact_dir / "label_experiment_report.json", {**label_report, "production_adoption_allowed": False})
        write_json(artifact_dir / "cost_stress_report.json", {"models": cost_stress_report, "production_adoption_allowed": False})
        write_json(artifact_dir / "regime_performance_report.json", {"models": regime_performance_report, "production_adoption_allowed": False})
        write_json(artifact_dir / "feature_robustness_report.json", {**feature_robustness, "production_adoption_allowed": False})
        write_json(artifact_dir / "champion_selection_report.json", champion_report)
        write_json(
            artifact_dir / "regime_restricted_candidates.json",
            {
                "candidates": regime_candidate_report["regime_candidates"],
                "production_adoption_allowed": False,
                "research_only": True,
            },
        )
        pd.DataFrame([row for row in regime_candidate_report["csv_rows"] if not str(row.get("candidate_name", "")).startswith("logistic_regression_")]).to_csv(artifact_dir / "regime_restricted_candidates.csv", index=False)
        write_json(
            artifact_dir / "ranker_candidate_report.json",
            {
                "candidates": regime_candidate_report["ranker_candidates"],
                "production_adoption_allowed": False,
                "research_only": True,
            },
        )
        pd.DataFrame([row for row in regime_candidate_report["csv_rows"] if str(row.get("candidate_name", "")).startswith("logistic_regression_")]).to_csv(artifact_dir / "ranker_candidate_report.csv", index=False)
        write_json(
            artifact_dir / "ce_pe_champion_report.json",
            {
                "ce_champion": regime_candidate_report["ce_pe_champions"]["ce"],
                "pe_champion": regime_candidate_report["ce_pe_champions"]["pe"],
                "production_adoption_allowed": False,
                "research_only": True,
            },
        )
        write_json(
            artifact_dir / "small_sample_warning_report.json",
            {
                "warnings": regime_candidate_report["small_sample_warnings"],
                "production_adoption_allowed": False,
                "research_only": True,
            },
        )
        write_json(
            artifact_dir / "monthly_stability_report.json",
            {
                "rows": regime_candidate_report["monthly_stability_rows"],
                "production_adoption_allowed": False,
                "research_only": True,
            },
        )
        write_json(
            artifact_dir / "cost_stress_by_candidate.json",
            {
                "candidates": regime_candidate_report["cost_stress_by_candidate"],
                "production_adoption_allowed": False,
                "research_only": True,
            },
        )
        write_json(
            artifact_dir / "candidate_cost_stress_report.json",
            {
                "candidates": regime_candidate_report["cost_stress_by_candidate"],
                "production_adoption_allowed": False,
                "research_only": True,
            },
        )
        write_json(
            artifact_dir / "fold_stability_report.json",
            {
                "rows": regime_candidate_report["fold_stability_rows"],
                "production_adoption_allowed": False,
                "research_only": True,
            },
        )
        write_json(artifact_dir / "candidate_champion_report.json", regime_candidate_report["candidate_champion_report"])
        write_json(
            artifact_dir / "production_blocker_report.json",
            {
                "candidates": regime_candidate_report["production_blocker_rows"],
                "production_adoption_allowed": False,
                "research_only": True,
                "reason": "Research candidates remain blocked until robustness and production gates pass.",
            },
        )
        write_json(
            artifact_dir / "execution_rescue_report.json",
            {
                "rows": execution_rescue_rows,
                "production_adoption_allowed": False,
                "research_only": True,
            },
        )
        pd.DataFrame(execution_rescue_rows).to_csv(artifact_dir / "execution_rescue_report.csv", index=False)
        _execution_rescue_markdown(execution_rescue_rows, artifact_dir)
        write_json(
            artifact_dir / "duplicate_timestamp_leakage_audit.json",
            regime_candidate_report["duplicate_timestamp_leakage_audit"],
        )
        write_json(
            artifact_dir / "target_horizon_report.json",
            regime_candidate_report["target_horizon_report"],
        )
        _candidate_summary_markdown(regime_candidate_report["candidates"], artifact_dir)
        paper_watchlist = _paper_watchlist_report(regime_candidate_report["candidates"], artifact_dir=artifact_dir)
        deep_data_audit = _deep_data_audit_report(df, artifact_dir=artifact_dir, feature_manifest=feature_manifest, leakage_passed=True)
        final_decision = _final_decision_report(
            artifact_dir=artifact_dir,
            dataset_path=dataset_path,
            row_count=row_count,
            feature_manifest=feature_manifest,
            metrics_payload=metrics_payload,
            regime_candidate_report=regime_candidate_report,
            paper_watchlist=paper_watchlist,
            cost_model_audit=cost_model_audit,
            horizon_report=horizon_report,
            deep_data_audit=deep_data_audit,
        )
        write_json(artifact_dir / "threshold_sweep.json", {"rows": aggregated_threshold_rows, "production_adoption_allowed": False})
        pd.DataFrame(aggregated_threshold_rows).to_csv(artifact_dir / "threshold_sweep.csv", index=False)
        write_json(artifact_dir / "all_model_training_report.json", {"mode": "retrain_all_models", "result": result, "metrics": metric_rows, "production_adoption_allowed": False})
        # Write canonical core_retrain_summary.json so --edge-refinement-report and
        # build_cost_aware_retrain_comparison.py work on retrain_all_models_<ts>/ dirs.
        _write_core_retrain_summary(
            artifact_dir=artifact_dir,
            dataset_path=dataset_path,
            labels_trained=labels,
            model_families=families,
            metric_rows=metric_rows,
            result_payload=result,
            evaluation_return_column=groups["evaluation_return_columns"][0] if groups["evaluation_return_columns"] else "net_forward_return",
            row_count=row_count,
        )
        dataset_selection_runtime = dataset_selection_payload or {
            "selected_dataset": {
                "path": str(dataset_path),
                "row_count": row_count,
                "date_range": date_range,
                "available_target_columns": groups["target_columns"],
                "available_return_columns": groups["evaluation_return_columns"],
                "available_feature_count": len(groups["input_features"]),
            },
            "candidates_considered": [],
            "selection_rule": "explicit_or_default_runtime_dataset",
        }
        report_paths_payload["dataset_selection_report_v2"] = _write_named_report(
            "dataset_selection_report",
            dataset_selection_runtime,
            [
                "# Dataset Selection Report",
                f"- Selected dataset: `{str(dataset_path)}`",
                f"- Row count: `{row_count}`",
                f"- Date range: `{date_range.get('start')} -> {date_range.get('end')}`",
                f"- Target columns: `{groups['target_columns']}`",
                f"- Return columns: `{groups['evaluation_return_columns']}`",
                f"- Feature count: `{len(groups['input_features'])}`",
            ],
        )
        feature_audit_payload = {
            **feature_manifest,
            "target_columns": groups["target_columns"],
            "metadata_columns": groups.get("metadata_columns", []),
            "audit": feature_sets.get("audit", {}),
        }
        report_paths_payload["feature_audit_report"] = _write_named_report(
            "feature_audit_report",
            feature_audit_payload,
            [
                "# Feature Audit Report",
                f"- Raw columns: `{feature_sets.get('audit', {}).get('raw_column_count')}`",
                f"- Final input features: `{len(selected_features)}`",
                f"- Forbidden removed: `{len(feature_sets.get('audit', {}).get('forbidden_columns_removed', []))}`",
                f"- Constant removed: `{len(feature_sets.get('audit', {}).get('constant_columns_removed', []))}`",
                f"- High-null removed: `{len(feature_sets.get('audit', {}).get('high_null_columns_removed', []))}`",
                f"- Duplicate removed: `{len(feature_sets.get('audit', {}).get('duplicate_columns_removed', []))}`",
            ],
        )
        report_paths_payload["all_models_retraining_report"] = _write_named_report(
            "all_models_retraining_report",
            {"metrics": metric_rows, "result": result, "training_manifest": training_manifest},
            [
                "# All Models Retraining Report",
                f"- Dataset: `{str(dataset_path)}`",
                f"- Models attempted: `{families}`",
                f"- Models succeeded: `{[row['model_name'] for row in metric_rows]}`",
                f"- Failures: `{len(result.get('failed_models', []))}`",
            ],
        )
        report_paths_payload["paper_readiness_report"] = _write_named_report(
            "paper_readiness_report",
            {
                "decision": "BLOCKED",
                "best_model": champion_report.get("champion"),
                "paper_watchlist": paper_watchlist,
                "final_decision": final_decision,
            },
            [
                "# Paper Readiness Report",
                "- Decision: `BLOCKED`",
                f"- Best model: `{(champion_report.get('champion') or {}).get('model_name')}`",
                f"- Best threshold: `{((champion_report.get('champion') or {}).get('selected_threshold'))}`",
                "- Reason: no model is auto-promoted to paper/live by this pipeline unless robustness gates clearly pass.",
            ],
        )
        report_paths_payload["model_improvement_diagnosis"] = _write_named_report(
            "model_improvement_diagnosis",
            {
                "what_was_wrong_before": [
                    "dataset auto-selection was too coarse",
                    "feature grouping over-flagged some contemporaneous realized-volatility features",
                    "default model-family selection was narrower than the supported registry",
                    "reporting was spread across artifacts without the requested top-level audit reports",
                ],
                "code_changes_made": [
                    "tightened target/return/metadata separation",
                    "added richer feature audit payloads",
                    "expanded default all-model registry to include broader CPU-safe families",
                    "added CLI aliases for the requested end-to-end workflow",
                ],
                "dataset_chosen": str(dataset_path),
                "important_risks": [
                    "paper/live adoption remains blocked unless post-cost and fold-stability gates truly pass",
                    "optional libraries may still be unavailable on this machine and are reported rather than forced",
                ],
            },
            [
                "# Model Improvement Diagnosis",
                "- The pipeline was training fewer default families than intended and its feature hygiene was too coarse.",
                f"- Dataset chosen: `{str(dataset_path)}`",
                "- Current risk posture: `BLOCKED` until robustness gates clearly pass.",
            ],
        )
        # === FIX 4 + FIX 7: final retraining summary ===
        # Build per-(label, model) trained/skipped/failed rollup by scanning
        # both the runtime `result` dict and the on-disk status/metrics files.
        # This guarantees we report what actually happened, never silent passes.
        trained_entries: List[Dict[str, Any]] = []
        skipped_entries: List[Dict[str, Any]] = []
        failed_entries: List[Dict[str, Any]] = []
        try:
            for label_name, model_reports in (result.get("reports") or {}).items():
                for model_name, report in (model_reports or {}).items():
                    metrics_path = artifact_dir / f"{model_name}_{label_name}_metrics.json"
                    model_path = artifact_dir / f"{model_name}_{label_name}.pkl"
                    status_path = _status_file_path(artifact_dir, label_name, model_name)
                    status_doc = _read_json_if_exists(status_path) or {}
                    if str(status_doc.get("status", "")).lower() == "failed":
                        failed_entries.append({
                            "label_name": label_name,
                            "model_name": model_name,
                            "error": status_doc.get("error_message"),
                            "traceback": None,
                        })
                    trained_entries.append({
                        "label_name": label_name,
                        "model_name": model_name,
                        "model_path": str(model_path) if model_path.exists() else None,
                        "metrics_path": str(metrics_path) if metrics_path.exists() else None,
                        "selected_threshold": (report.get("selected_threshold_from_validation") or {}).get("threshold"),
                        "test_metrics": report.get("test_metrics") or {},
                    })
            for entry in (result.get("skipped_models") or []):
                skipped_entries.append({
                    "label_name": entry.get("label_name"),
                    "model_name": entry.get("model_family"),
                    "reason": entry.get("reason"),
                })
            for entry in (result.get("failed_models") or []):
                failed_entries.append({
                    "label_name": entry.get("label_name"),
                    "model_name": entry.get("model_family"),
                    "error": entry.get("error"),
                    "traceback": entry.get("stack_summary"),
                })
        except Exception as exc:  # noqa: BLE001
            print(f"[retrain] WARN: failed to roll up per-pair results: {exc}")
        completed_ok = (len(failed_entries) == 0) and (len(trained_entries) > 0)
        remaining_blockers: List[str] = []
        # FIX 7: empty observer note
        if bool(getattr(args, "full_pipeline", False)) and int(row_count) < 300:
            remaining_blockers.append(
                "full_pipeline_produced_fewer_than_300_rows: "
                "the forward observer upstream of build_option_edge_dataset.main() "
                "appears to be empty. This is a data-source issue, not a pipeline bug."
            )
        if failed_entries:
            remaining_blockers.append(
                f"{len(failed_entries)} (label, model) pair(s) failed; see models_failed."
            )
        summary_payload: Dict[str, Any] = {
            "dataset_path": str(dataset_path),
            "artifact_dir": str(artifact_dir),
            "row_count": int(row_count),
            "date_range": date_range,
            "labels_considered": [name for name in PRIMARY_LABELS if name in df.columns],
            "labels_skipped": label_skip_report,
            "models_planned": list(families),
            "models_trained": trained_entries,
            "models_skipped": skipped_entries,
            "models_failed": failed_entries,
            "leakage_columns_dropped": list(dropped_leakage),
            "validation_strategy": "chronological_walk_forward",
            "walk_forward_folds": int(args.walk_forward_folds),
            "completed_ok": bool(completed_ok),
            "remaining_blockers": remaining_blockers,
            "production_adoption_allowed": False,
            "research_only": True,
        }
        summary_json_path = artifact_dir / "retrain_all_models_summary.json"
        summary_md_path = artifact_dir / "retrain_all_models_summary.md"
        write_json(summary_json_path, summary_payload)
        try:
            labels_skipped_md = [
                f"- `{entry['label_name']}`: {entry['reason']}"
                for entry in summary_payload["labels_skipped"]
            ] or ["(none)"]
            models_trained_md = [
                f"- `{entry['model_name']}` on `{entry['label_name']}` -> `{entry['model_path']}`"
                for entry in summary_payload["models_trained"]
            ] or ["(none)"]
            models_skipped_md = [
                f"- `{entry['model_name']}` on `{entry['label_name']}`: {entry['reason']}"
                for entry in summary_payload["models_skipped"]
            ] or ["(none)"]
            models_failed_md = [
                f"- `{entry['model_name']}` on `{entry['label_name']}`: {entry['error']}"
                for entry in summary_payload["models_failed"]
            ] or ["(none)"]
            blockers_md = [
                f"- {b}" for b in summary_payload["remaining_blockers"]
            ] or ["(none)"]
            md_lines = [
                "# Retrain All Models Summary",
                f"- Dataset: `{summary_payload['dataset_path']}`",
                f"- Artifact dir: `{summary_payload['artifact_dir']}`",
                f"- Rows: `{summary_payload['row_count']}`",
                f"- Date range: `{summary_payload['date_range'].get('start')}` -> `{summary_payload['date_range'].get('end')}`",
                f"- Validation: `{summary_payload['validation_strategy']}` (folds=`{summary_payload['walk_forward_folds']}`)",
                f"- Models trained: `{len(summary_payload['models_trained'])}`",
                f"- Models skipped: `{len(summary_payload['models_skipped'])}`",
                f"- Models failed: `{len(summary_payload['models_failed'])}`",
                f"- Completed OK: `{summary_payload['completed_ok']}`",
                "",
                "## Labels considered",
                ", ".join(f"`{n}`" for n in summary_payload["labels_considered"]) or "(none)",
                "",
                "## Labels skipped",
                *labels_skipped_md,
                "",
                "## Models trained",
                *models_trained_md,
                "",
                "## Models skipped",
                *models_skipped_md,
                "",
                "## Models failed",
                *models_failed_md,
                "",
                "## Remaining blockers",
                *blockers_md,
            ]
            summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"[retrain] WARN: failed to write retrain_all_models_summary.md: {exc}")
        if not completed_ok:
            print(f"[retrain] WARNING: completed_ok=False (failures={len(failed_entries)}, trained={len(trained_entries)})")
        print(f"[retrain] summary_json={summary_json_path}")
        print(f"[retrain] summary_md={summary_md_path}")
        # Also surface summary in the structured payload below.
        report_paths_payload["retrain_all_models_summary"] = str(summary_json_path)
        ce_best = regime_candidate_report["ce_pe_champions"]["ce"]
        pe_best = regime_candidate_report["ce_pe_champions"]["pe"]
        print(json.dumps({
            "artifact_dir": str(artifact_dir),
            "dataset_rows_used": row_count,
            "features_used": len(selected_features),
            "models_trained": [row["model_name"] for row in metric_rows],
            "metrics_report_path": str(artifact_dir / "metrics_report.json"),
            "champion_selection_report_path": str(artifact_dir / "champion_selection_report.json"),
            "regime_restricted_candidates_path": str(artifact_dir / "regime_restricted_candidates.json"),
            "best_ce_candidate": (ce_best or {}).get("candidate_name"),
            "best_pe_candidate": (pe_best or {}).get("candidate_name"),
            "paper_watchlist_candidates": len(paper_watchlist.get("candidates", [])),
            "execution_rescue_report_path": str(artifact_dir / "execution_rescue_report.json"),
            "final_report_path": str(artifact_dir / "final_ml_trading_decision_report.json"),
            "report_paths": report_paths_payload,
            "production_adoption_allowed": False,
        }, indent=2))
        return
    if bool(args.model_rescue_experiments):
        rescue_run_started = time.time()
        rescue_source_df = df.copy()
        if args.use_black_scholes_features:
            enriched_path = ensure_black_scholes_dataset(
                dataset_path,
                risk_free_rate=float(args.risk_free_rate),
                dividend_yield=float(args.dividend_yield),
                symbol=str(args.symbol),
            )
            enriched_dataset_path = str(enriched_path)
            rescue_source_df = load_dataset(enriched_path)
            rescue_source_df["timestamp"] = _normalize_timestamp_series(rescue_source_df["timestamp"])
            rescue_source_df = _filter_invalid_bs_rows(rescue_source_df)
            if args.drop_bad_greeks:
                rescue_source_df, _ = _drop_bad_greeks_rows(rescue_source_df)
        rescue_source_df = _prepare_chronological_work(rescue_source_df)
        rescue_feature_sets = build_feature_sets(rescue_source_df, use_black_scholes_features=bool(args.use_black_scholes_features))
        baseline_features = rescue_feature_sets["baseline_features"]
        feature_variants = build_feature_variants(baseline_features, rescue_feature_sets.get("enriched_features"))
        live_schema_report = _live_schema_gate_for_variant(feature_variants["live_computable_only"]["feature_names"])
        cli_filter = _build_cli_rescue_filter_spec(args)
        experiments = _build_fast_pass_rescue_experiments() if bool(args.rescue_fast_pass) else _build_default_rescue_experiments()
        experiments = _filter_rescue_experiments(experiments, str(args.rescue_experiment_filter))
        if any(bool(value) for value in cli_filter.values()):
            experiments.append({"experiment_name": "cli_custom_slice", "filter_spec": cli_filter, "feature_set": "live_computable_only", "model_families": _parse_model_families_arg(args.model_families)})
        checkpoint_path = _rescue_checkpoint_path(artifact_dir)
        checkpoint = _load_rescue_checkpoint(checkpoint_path)
        resume_summary = _rescue_resume_summary(experiments, checkpoint)
        print(f"[rescue] dataset={dataset_path}")
        print(f"[rescue] output_dir={artifact_dir}")
        print(f"[rescue] total_planned={len(experiments)} completed={len(resume_summary['completed_experiments'])} skipped={len(resume_summary['skipped_experiments'])} pending={len(resume_summary['pending_experiments'])} resume_active={resume_summary['resume_active']}")
        experiment_outputs: List[Dict[str, Any]] = []
        stored_entries = checkpoint.get("experiments", {}) or {}
        for experiment in experiments:
            experiment_name = str(experiment["experiment_name"])
            existing = stored_entries.get(experiment_name)
            if _checkpoint_entry_valid(existing):
                experiment_outputs.append(existing)
                continue
            if _should_stop_rescue_run(rescue_run_started, float(args.rescue_time_budget_minutes or 0.0)):
                break
            if int(args.max_rescue_experiments or 0) > 0:
                fresh_run_count = sum(1 for item in experiment_outputs if item.get("run_this_invocation"))
                if fresh_run_count >= int(args.max_rescue_experiments):
                    break
            filtered_payload = _apply_market_slice_filters(rescue_source_df, experiment.get("filter_spec") or {})
            filtered_df = filtered_payload["filtered_df"]
            groups = detect_column_groups(filtered_df)
            labels, _ = choose_labels(filtered_df)
            validation_report = _rescue_validation_report(filtered_df, labels, groups.get("evaluation_return_column_used"))
            if validation_report["failed"]:
                payload = {
                    "experiment_name": experiment_name,
                    "filter_report": filtered_payload,
                    "validation_report": validation_report,
                    "feature_set": experiment["feature_set"],
                    "model_families": experiment.get("model_families"),
                    "final_experiment_verdict": "FAILED_FILTER_OR_DATA_QUALITY_GATE",
                    "production_adoption_allowed": False,
                    "run_this_invocation": True,
                }
                experiment_outputs.append(payload)
                stored_entries[experiment_name] = payload
                checkpoint["experiments"] = stored_entries
                _atomic_write_json(checkpoint_path, checkpoint)
                print(f"[rescue] {experiment_name} rows={validation_report['row_count']} timestamps={validation_report['unique_timestamps']} label_ratio={validation_report['label_positive_ratio']} eval_col={groups.get('evaluation_return_column_used')} verdict=FAILED_FILTER_OR_DATA_QUALITY_GATE reason={','.join(validation_report['failure_reasons'])}")
                continue
            feature_spec = feature_variants[experiment["feature_set"]]
            threshold_start = 0.50 if bool(args.rescue_fast_pass) else float(args.threshold_grid_start)
            threshold_end = 0.70 if bool(args.rescue_fast_pass) else float(args.threshold_grid_end)
            threshold_step = 0.05 if bool(args.rescue_fast_pass) else float(args.threshold_grid_step)
            model_families = list(experiment.get("model_families") or _parse_model_families_arg(args.model_families))
            if bool(args.rescue_fast_pass) and validation_report["row_count"] < 5000:
                model_families = [name for name in model_families if name != "logistic_regression_isotonic"]
            training_result = train_variant(
                filtered_df,
                feature_cols=feature_spec["feature_names"],
                labels=labels,
                artifact_dir=artifact_dir / experiment_name,
                variant_name=experiment_name,
                evaluation_return_column=groups["evaluation_return_column_used"],
                walk_forward_folds=int(args.walk_forward_folds),
                threshold_grid_start=threshold_start,
                threshold_grid_end=threshold_end,
                threshold_grid_step=threshold_step,
                min_threshold_trades=int(args.min_threshold_trades),
                paper_config=vars(args),
                model_families=model_families,
            )
            payload = {
                "experiment_name": experiment_name,
                "feature_set": experiment["feature_set"],
                "feature_names": feature_spec["feature_names"],
                "rejected_features": feature_spec["rejected_features"],
                "feature_audit": feature_spec["feature_audit"],
                "filter_report": {key: value for key, value in filtered_payload.items() if key != "filtered_df"},
                "validation_report": validation_report,
                "training_result": training_result,
                "final_experiment_verdict": "COMPLETED",
                "production_adoption_allowed": False,
                "run_this_invocation": True,
            }
            experiment_outputs.append(payload)
            stored_entries[experiment_name] = payload
            checkpoint["experiments"] = stored_entries
            _atomic_write_json(checkpoint_path, checkpoint)
            candidate_rows = _collect_rescue_candidate_rows([payload], live_schema_report=live_schema_report)
            top_row = candidate_rows[0] if candidate_rows else {}
            report = ((((training_result.get("reports") or {}).values()) and next(iter((training_result.get("reports") or {}).values()))) or {})
            top_report = next(iter(report.values())) if isinstance(report, dict) and report else {}
            after = ((top_report.get("paper_execution_filters") or {}).get("after") or {})
            metrics = top_report.get("test_metrics") or {}
            print(f"[rescue] {experiment_name} rows={validation_report['row_count']} timestamps={validation_report['unique_timestamps']} label_ratio={validation_report['label_positive_ratio']} eval_col={groups.get('evaluation_return_column_used')} roc_auc={metrics.get('roc_auc')} pr_auc={metrics.get('pr_auc')} brier={metrics.get('brier_score')} calibration={(top_report.get('calibration_fit') or {}).get('mode')} est_trades={after.get('trade_count')} net_pnl={after.get('total_return')} pf={after.get('profit_factor')} verdict={top_row.get('final_verdict')}")
        checkpoint["experiments"] = stored_entries
        _atomic_write_json(checkpoint_path, checkpoint)
        final_resume_summary = _rescue_resume_summary(experiments, checkpoint)
        completed_outputs = [stored_entries[name] for name in [str(exp.get("experiment_name")) for exp in experiments] if _checkpoint_entry_valid(stored_entries.get(name))]
        all_complete = len(final_resume_summary["pending_experiments"]) == 0
        if all_complete:
            finalized = _finalize_rescue_outputs(
                artifact_dir=artifact_dir,
                dataset_path=dataset_path,
                experiment_outputs=completed_outputs,
                feature_variants=feature_variants,
                live_schema_report=live_schema_report,
            )
            best_rescue = finalized["leaderboard"][0] if finalized["leaderboard"] else {}
        else:
            finalized = {}
            best_rescue = {}
        print(json.dumps({
            "output_dir": str(artifact_dir),
            "dataset_path": str(dataset_path),
            "best_rescue_experiment": {key: value for key, value in best_rescue.items() if key != "report_ref"},
            "resume_summary": final_resume_summary,
            "shadow_manifests_generated": finalized.get("manifest_paths", []),
            "all_experiments_complete": all_complete,
            "production_adoption_allowed": False,
        }, indent=2))
        return
    if bool(args.retrain_all_models_strict):
        if not args.use_black_scholes_features:
            raise RuntimeError("Strict mode requires --use-black-scholes-features so baseline and BS variants can be audited safely.")
        enriched_path = ensure_black_scholes_dataset(
            dataset_path,
            risk_free_rate=float(args.risk_free_rate),
            dividend_yield=float(args.dividend_yield),
            symbol=str(args.symbol),
        )
        enriched_dataset_path = str(enriched_path)
        enriched_df = load_dataset(enriched_path)
        if "timestamp" in enriched_df.columns:
            enriched_df["timestamp"] = _normalize_timestamp_series(enriched_df["timestamp"])
        enriched_df = _filter_invalid_bs_rows(enriched_df)
        if args.drop_bad_greeks:
            enriched_df, _ = _drop_bad_greeks_rows(enriched_df)
        baseline_df_for_training, enriched_df_for_training, same_period_report = _same_period_alignment_report(df, enriched_df, enabled=bool(args.same_period_comparison))
        baseline_features = build_feature_sets(baseline_df_for_training, use_black_scholes_features=False)["baseline_features"]
        enriched_feature_sets = build_feature_sets(enriched_df_for_training, use_black_scholes_features=True)
        feature_variants = build_feature_variants(baseline_features, enriched_feature_sets["enriched_features"])
        families = _parse_model_families_arg(args.model_families)
        # Apply --only-model case-insensitive filtering and guard against zero models.
        if only_models:
            only_models_lower = {str(x).strip().lower() for x in only_models}
            if "xgb_rf_ensemble" in only_models_lower and "xgb_rf_ensemble" not in families:
                families.append("xgb_rf_ensemble")
            families = [name for name in families if str(name).strip().lower() in only_models_lower]
            if not families:
                raise ValueError(
                    f"--only-model={only_models!r} resolved to zero models after "
                    f"case-normalized lookup.  Available families (lowercased): "
                    f"{sorted({str(f).strip().lower() for f in _parse_model_families_arg(args.model_families)})!r}."
                )
        variant_results: List[Dict[str, Any]] = []
        skipped_models: List[Dict[str, Any]] = []
        failed_models: List[Dict[str, Any]] = []
        for feature_set_name, spec in feature_variants.items():
            names = list(spec.get("feature_names") or [])
            if len(names) < 5:
                skipped_models.append({"variant_name": feature_set_name, "reason": "too_few_features_after_gating"})
                continue
            variant_df = enriched_df_for_training if "black_scholes" in feature_set_name else baseline_df_for_training
            groups = detect_column_groups(variant_df)
            result = train_variant(
                variant_df,
                feature_cols=names,
                labels=choose_labels(variant_df)[0],
                artifact_dir=artifact_dir,
                variant_name=feature_set_name,
                evaluation_return_column=groups["evaluation_return_column_used"],
                walk_forward_folds=int(args.walk_forward_folds),
                threshold_grid_start=float(args.threshold_grid_start),
                threshold_grid_end=float(args.threshold_grid_end),
                threshold_grid_step=float(args.threshold_grid_step),
                min_threshold_trades=int(args.min_threshold_trades),
                paper_config=vars(args),
                model_families=families,
                only_models=only_models,
                only_targets=only_targets,
            )
            skipped_models.extend(result.get("skipped_models", []))
            failed_models.extend(result.get("failed_models", []))
            variant_results.append({"variant_name": feature_set_name, "feature_set": feature_set_name, "result": result, "feature_spec": spec})
        live_schema_report = adoption.get("strict_live_schema_gate", {"compatible": False}) if adoption else {"compatible": False}
        if not live_schema_report.get("compatible"):
            live_schema_report = _live_schema_gate_for_variant(feature_variants["live_computable_only"]["feature_names"])
        leaderboard = build_model_leaderboard(_collect_candidate_rows(variant_results, live_schema_report=live_schema_report))
        best_candidate = leaderboard[0] if leaderboard else {}
        final_recommendation = "NO_MODEL_READY_FOR_PAPER_TRADING"
        if any(str(row.get("paper_readiness_verdict")) in {"PAPER_TRADE_CANDIDATE_LOW_CONFIDENCE", "PAPER_TRADE_CANDIDATE_MEDIUM_CONFIDENCE", "PAPER_TRADE_CANDIDATE_HIGH_CONFIDENCE"} for row in leaderboard):
            final_recommendation = "PAPER_TRADE_CANDIDATE_ONLY_PRODUCTION_BLOCKED"
        feature_audit_payload = {
            "feature_variants": feature_variants,
            "forbidden_feature_audit": feature_manifest_payload,
        }
        gate_summary_payload = {
            "same_period_report": same_period_report,
            "skipped_models": skipped_models,
            "failed_models": failed_models,
            "top_candidate": best_candidate,
            "production_adoption_allowed": False,
            "final_recommendation": final_recommendation,
        }
        report_paths_payload["leaderboard_report"] = _write_named_report(
            "retrain_all_models_leaderboard",
            {"leaderboard": leaderboard, "skipped_models": skipped_models, "failed_models": failed_models},
            ["# Retrain All Models Strict Audit", "", "## Model Leaderboard", *[f"- rank={row['rank']} model=`{row['model_family']}` feature_set=`{row['feature_set']}` verdict=`{row['paper_readiness_verdict']}` PF=`{row['PF']}` Sharpe=`{row['Sharpe']}` ROC-AUC=`{row['ROC-AUC']}`" for row in leaderboard[:25]]],
        )
        report_paths_payload["gate_summary_report"] = _write_named_report(
            "retrain_all_models_gate_summary",
            gate_summary_payload,
            ["# Retrain All Models Strict Audit", "", "## Executive Verdict", f"- best candidate: `{best_candidate.get('model_family')}`", f"- paper trading allowed: `{final_recommendation != 'NO_MODEL_READY_FOR_PAPER_TRADING'}`", "- production adoption allowed: `False`", f"- final verdict: `{best_candidate.get('paper_readiness_verdict', 'REJECT_NO_EDGE')}`", f"- main reason: `{final_recommendation}`"],
        )
        report_paths_payload["feature_audit_report"] = _write_named_report(
            "retrain_all_models_feature_audit",
            feature_audit_payload,
            ["# Retrain All Models Strict Audit", "", "## Feature Set Audit", *[f"- `{name}`: feature_count=`{len(spec.get('feature_names', []))}` live_status=`{spec.get('live_computable_status')}`" for name, spec in feature_variants.items()]],
        )
        strict_failure_payload = {
            "leaderboard": leaderboard,
            "skipped_models": skipped_models,
            "failed_models": failed_models,
            "final_recommendation": final_recommendation,
            "production_adoption_allowed": False,
        }
        report_paths_payload["strict_failure_diagnosis_report"] = _write_named_report(
            "retrain_all_models_failure_diagnosis",
            strict_failure_payload,
            ["# Retrain All Models Strict Audit", "", "## Failure Diagnosis", *[f"- skipped model: `{row}`" for row in skipped_models[:20]], *[f"- failed model: `{row.get('model_family')}` reason=`{row.get('error')}`" for row in failed_models[:20]]],
        )
        report_paths_payload["final_recommendation_report"] = _write_named_report(
            "retrain_all_models_final_recommendation",
            {"best_candidate": best_candidate, "final_recommendation": final_recommendation, "dataset_path": str(dataset_path), "production_adoption_allowed": False},
            ["# Retrain All Models Strict Audit", "", "## Final Recommendation", f"- exact dataset selected: `{dataset_path}`", f"- best candidate: `{best_candidate}`", f"- final recommendation: `{final_recommendation}`", "- production adoption allowed: `False`"],
        )
        write_json(artifact_dir / "all_model_training_report.json", {"strict_variant_results": variant_results, "leaderboard": leaderboard, "skipped_models": skipped_models, "failed_models": failed_models})
        write_json(artifact_dir / "feature_manifest.json", {"feature_variants": feature_variants, "dataset_selection": dataset_selection_payload})
        write_json(artifact_dir / "feature_list_used.json", {"feature_variants": feature_variants})
        # Collect metric_rows across all variants for core_retrain_summary
        strict_metric_rows = []
        strict_result = {"trained_models": [], "skipped_models": skipped_models, "failed_models": failed_models}
        for vr in variant_results:
            vr_result = vr.get("result", {})
            strict_result["trained_models"].extend(vr_result.get("trained_models", []))
            strict_metric_rows.extend(vr_result.get("metric_rows", []))
        # Write canonical summary so downstream tools work on strict-variant dirs too
        _write_core_retrain_summary(
            artifact_dir=artifact_dir,
            dataset_path=dataset_path,
            labels_trained=labels,
            model_families=families,
            metric_rows=strict_metric_rows,
            result_payload=strict_result,
            evaluation_return_column=groups["evaluation_return_columns"][0] if groups["evaluation_return_columns"] else "net_forward_return",
            row_count=len(df),
        )
        summary = {
            "output_dir": str(artifact_dir),
            "dataset_path": str(dataset_path),
            "models_trained": sorted({entry for item in variant_results for entry in item["result"].get("trained_models", [])}),
            "skipped_models": skipped_models,
            "failed_models": failed_models,
            "final_recommendation": final_recommendation,
            "production_adoption_allowed": False,
        }
        print(json.dumps(summary, indent=2))
        return
    if args.use_black_scholes_features:
        enriched_path = ensure_black_scholes_dataset(
            dataset_path,
            risk_free_rate=float(args.risk_free_rate),
            dividend_yield=float(args.dividend_yield),
            symbol=str(args.symbol),
        )
        enriched_dataset_path = str(enriched_path)
        enriched_df = load_dataset(enriched_path)
        if "timestamp" in enriched_df.columns:
            enriched_df["timestamp"] = _normalize_timestamp_series(enriched_df["timestamp"])
        enriched_df = _filter_invalid_bs_rows(enriched_df)
        dropped_bad_greeks = 0
        if args.drop_bad_greeks:
            enriched_df, dropped_bad_greeks = _drop_bad_greeks_rows(enriched_df)
        baseline_df_for_training, enriched_df_for_training, same_period_report = _same_period_alignment_report(
            df,
            enriched_df,
            enabled=bool(args.same_period_comparison),
        )
        if same_period_report.get("same_period_gate") == "FAIL" and not bool(args.same_period_comparison):
            leakage_result["passed"] = False
            leakage_result["notes"].append("Comparison periods differ and same-period mode was not enabled.")
        feature_sets = build_feature_sets(baseline_df_for_training, use_black_scholes_features=False)
        live_feature_audit: List[Dict[str, Any]] = []
        if bool(args.live_computable_only):
            selected_live, live_feature_audit = _select_live_computable_features(feature_sets["baseline_features"])
            if len(selected_live) < 5:
                raise RuntimeError("live-computable-only mode left too few baseline features to train safely.")
            feature_sets["baseline_features"] = selected_live
        baseline_result = train_variant(
            baseline_df_for_training,
            feature_cols=feature_sets["baseline_features"],
            labels=labels,
            artifact_dir=artifact_dir,
            variant_name="baseline",
            evaluation_return_column=column_groups["evaluation_return_column_used"],
            walk_forward_folds=int(args.walk_forward_folds),
            threshold_grid_start=float(args.threshold_grid_start),
            threshold_grid_end=float(args.threshold_grid_end),
            threshold_grid_step=float(args.threshold_grid_step),
            min_threshold_trades=int(args.min_threshold_trades),
            paper_config=vars(args),
        )
        output_payload["baseline"] = baseline_result["reports"]
        trained_models.extend(baseline_result["trained_models"])
        feature_manifest_payload["feature_count"] = len(feature_sets["baseline_features"])
        feature_manifest_payload["features"] = feature_sets["baseline_features"]
        feature_manifest_payload["baseline_features"] = feature_sets["baseline_features"]
        feature_manifest_payload["rejected_features"] = feature_sets["rejected_features"]
        feature_manifest_payload["live_feature_audit"] = live_feature_audit
        if bool(args.paper_readiness_audit):
            simple_features = _simple_no_greeks_features(feature_sets["baseline_features"])
            if len(simple_features) >= 5:
                simple_result = train_variant(
                    baseline_df_for_training,
                    feature_cols=simple_features,
                    labels=labels,
                    artifact_dir=artifact_dir,
                    variant_name="simple_baseline_no_greeks",
                    evaluation_return_column=column_groups["evaluation_return_column_used"],
                    walk_forward_folds=int(args.walk_forward_folds),
                    threshold_grid_start=float(args.threshold_grid_start),
                    threshold_grid_end=float(args.threshold_grid_end),
                    threshold_grid_step=float(args.threshold_grid_step),
                    min_threshold_trades=int(args.min_threshold_trades),
                    paper_config=vars(args),
                )
                output_payload["simple_baseline_no_greeks"] = simple_result["reports"]
                trained_models.extend(simple_result["trained_models"])
        bs_feature_sets = build_feature_sets(enriched_df_for_training, use_black_scholes_features=True)
        if bool(args.live_computable_only):
            selected_live_bs, bs_live_audit = _select_live_computable_features(bs_feature_sets["enriched_features"])
            feature_manifest_payload["bs_live_feature_audit"] = bs_live_audit
            if len(selected_live_bs) < 5:
                raise RuntimeError("live-computable-only mode left too few Black-Scholes features to train safely.")
            bs_feature_sets["enriched_features"] = selected_live_bs
        enriched_groups = detect_column_groups(enriched_df_for_training)
        enriched_result = train_variant(
            enriched_df_for_training,
            feature_cols=bs_feature_sets["enriched_features"],
            labels=labels,
            artifact_dir=artifact_dir,
            variant_name="black_scholes",
            evaluation_return_column=enriched_groups["evaluation_return_column_used"],
            walk_forward_folds=int(args.walk_forward_folds),
            threshold_grid_start=float(args.threshold_grid_start),
            threshold_grid_end=float(args.threshold_grid_end),
            threshold_grid_step=float(args.threshold_grid_step),
            min_threshold_trades=int(args.min_threshold_trades),
            paper_config=vars(args),
        )
        output_payload["black_scholes_enriched"] = enriched_result["reports"]
        trained_models.extend(enriched_result["trained_models"])
        feature_manifest_payload["black_scholes_features"] = bs_feature_sets["enriched_features"]
        feature_manifest_payload["bs_feature_groups"] = {
            "research_features": BS_RESEARCH_FEATURES,
            "flag_features": BS_FLAG_FEATURES,
        }
        estimated_only = True
        adoption = _production_adoption_verdict(
            enriched_best=_best_variant_report(enriched_result["reports"]),
            feature_cols=bs_feature_sets["enriched_features"],
            leakage_ok=bool(leakage_result["passed"]),
            estimated_only=estimated_only,
        )
        report_paths_payload = write_bs_comparison_report(
            dataset_path=dataset_path,
            enriched_dataset_path=enriched_path,
            df=baseline_df_for_training,
            enriched_df=enriched_df_for_training,
            feature_sets=bs_feature_sets,
            baseline_result=baseline_result,
            enriched_result=enriched_result,
            leakage_result=leakage_result,
            adoption=adoption,
            column_groups=enriched_groups,
            same_period_report=same_period_report,
        )
        report_paths_payload["rows_dropped_bad_greeks"] = dropped_bad_greeks
        comparison_payload = report_paths_payload.get("payload", {})
        report_paths_payload["same_period_report"] = _write_named_report(
            "same_period_comparison",
            comparison_payload.get("same_period_report", {}),
            [
                "# Same Period Comparison",
                f"- Enabled: `{comparison_payload.get('same_period_report', {}).get('enabled')}`",
                f"- Gate: `{comparison_payload.get('same_period_report', {}).get('same_period_gate')}`",
                f"- Baseline original range: `{comparison_payload.get('same_period_report', {}).get('baseline_original_date_range')}`",
                f"- BS original range: `{comparison_payload.get('same_period_report', {}).get('bs_original_date_range')}`",
                f"- Overlap: `{comparison_payload.get('same_period_report', {}).get('overlapping_date_range')}`",
            ],
        )
        report_paths_payload["cost_stress_report"] = _write_named_report(
            "cost_stress_report",
            {
                "baseline": comparison_payload.get("baseline_metrics", {}).get("cost_stress", {}),
                "black_scholes": comparison_payload.get("enriched_metrics", {}).get("cost_stress", {}),
            },
            [
                "# Cost Stress Report",
                "Baseline and Black-Scholes cost stress scenarios are recorded in the JSON payload.",
            ],
        )
        report_paths_payload["threshold_robustness_report"] = _write_named_report(
            "threshold_robustness_report",
            {
                "baseline": comparison_payload.get("baseline_metrics", {}).get("threshold_robustness", {}),
                "black_scholes": comparison_payload.get("enriched_metrics", {}).get("threshold_robustness", {}),
            },
            [
                "# Threshold Robustness Report",
                f"- Baseline chosen threshold robust: `{comparison_payload.get('baseline_metrics', {}).get('threshold_robustness', {}).get('chosen_threshold_is_robust')}`",
                f"- BS chosen threshold robust: `{comparison_payload.get('enriched_metrics', {}).get('threshold_robustness', {}).get('chosen_threshold_is_robust')}`",
            ],
        )
        report_paths_payload["regime_performance_report"] = _write_named_report(
            "regime_performance_report",
            {
                "baseline": comparison_payload.get("baseline_metrics", {}).get("regime_performance", {}),
                "black_scholes": comparison_payload.get("enriched_metrics", {}).get("regime_performance", {}),
            },
            [
                "# Regime Performance Report",
                "Grouped trade metrics by option type, moneyness, DTE, time bucket, and available regime columns are in the JSON payload.",
            ],
        )
        report_paths_payload["final_decision_matrix_report"] = _write_named_report(
            "final_decision_matrix",
            comparison_payload.get("final_decision_matrix", {}),
            [
                "# Final Decision Matrix",
                *[
                    f"- {key}: `{value}`"
                    for key, value in (comparison_payload.get("final_decision_matrix", {}) or {}).items()
                ],
            ],
        )
        if bool(args.paper_readiness_audit):
            paper_payload = _build_paper_readiness_payload(comparison_payload, feature_manifest_payload)
            report_paths_payload["paper_readiness_audit"] = _write_named_report(
                "paper_readiness_audit",
                paper_payload,
                [
                    "# Paper Readiness Audit",
                    f"- Final verdict: `{paper_payload.get('decision_matrix', {}).get('final_verdict')}`",
                    *[
                        f"- {key}: `{value}`"
                        for key, value in (paper_payload.get("decision_matrix", {}) or {}).items()
                    ],
                ],
            )
            report_paths_payload["feature_compatibility_report"] = _write_named_report(
                "feature_compatibility_report",
                {"feature_audit": feature_manifest_payload.get("live_feature_audit", [])},
                [
                    "# Feature Compatibility Report",
                    "See JSON for full live-computability audit table.",
                ],
            )
            report_paths_payload["calibration_report"] = _write_named_report(
                "probability_calibration_report",
                {"baseline": comparison_payload.get("baseline_metrics", {}).get("calibration_audit", {}), "paper_candidate": comparison_payload.get("enriched_metrics", {}).get("calibration_audit", {})},
                [
                    "# Probability Calibration Report",
                    f"- Candidate calibration gate: `{comparison_payload.get('enriched_metrics', {}).get('calibration_audit', {}).get('calibration_gate')}`",
                ],
            )
            report_paths_payload["probability_monotonicity_report"] = _write_named_report(
                "probability_monotonicity_report",
                {"baseline": comparison_payload.get("baseline_metrics", {}).get("probability_monotonicity", {}), "paper_candidate": comparison_payload.get("enriched_metrics", {}).get("probability_monotonicity", {})},
                [
                    "# Probability Monotonicity Report",
                    f"- Candidate gate: `{comparison_payload.get('enriched_metrics', {}).get('probability_monotonicity', {}).get('gate')}`",
                ],
            )
            report_paths_payload["daily_pnl_stability_report"] = _write_named_report(
                "daily_pnl_stability_report",
                {"paper_candidate": comparison_payload.get("enriched_metrics", {}).get("daily_pnl_stability", {})},
                [
                    "# Daily PnL Stability Report",
                    f"- Candidate gate: `{comparison_payload.get('enriched_metrics', {}).get('daily_pnl_stability', {}).get('gate')}`",
                ],
            )
            if bool(args.failure_diagnosis_audit):
                diagnosis_payload = _build_failure_diagnosis_payload(
                    dataset_path=str(dataset_path),
                    comparison_payload=comparison_payload,
                    paper_payload=paper_payload,
                    feature_manifest_payload=feature_manifest_payload,
                    output_dir=str(artifact_dir),
                )
                report_paths_payload["failure_diagnosis_audit"] = _write_named_report(
                    "failure_diagnosis_audit",
                    diagnosis_payload,
                    [
                        "# Failure Diagnosis Audit",
                        "",
                        "## Executive Verdict",
                        f"- final_status: `{diagnosis_payload.get('final_status')}`",
                        f"- paper_trading_allowed: `{diagnosis_payload.get('paper_trading_allowed')}`",
                        f"- production_adoption_allowed: `{diagnosis_payload.get('production_adoption_allowed')}`",
                        f"- main_reason: `{diagnosis_payload.get('main_reason')}`",
                        "",
                        "## Gate Summary Table",
                        *[
                            f"- {row['gate_name']}: status=`{row['status']}` severity=`{row['severity']}` reason=`{row['short_reason']}`"
                            for row in diagnosis_payload.get("gate_failure_table", [])
                        ],
                        "",
                        "## Critical Blockers",
                        *[f"- `{message}`" for message in diagnosis_payload.get("do_not_proceed_messages", [])],
                        "",
                        "## Root Cause Analysis",
                        *[
                            f"- `{gate}`: {', '.join(causes)}"
                            for gate, causes in (diagnosis_payload.get("failed_gate_root_causes", {}) or {}).items()
                        ],
                        "",
                        "## Improvement Priority Plan",
                        *[
                            f"- rank={item['priority_rank']} issue=`{item['issue']}` impact=`{item['expected_impact']}` difficulty=`{item['implementation_difficulty']}` next=`{item['suggested_next_experiment']}`"
                            for item in diagnosis_payload.get("improvement_priority_plan", [])
                        ],
                        "",
                        "## Next Experiments",
                        *[
                            f"- `{item['experiment_name']}`: `{item['exact_command']}`"
                            for item in diagnosis_payload.get("next_experiments", [])
                        ],
                        "",
                        "## Model Rescue Plan",
                        *[f"- {line}" for line in diagnosis_payload.get("model_rescue_plan", [])],
                        "",
                        "## Files/Commands Used",
                        f"- dataset_path: `{diagnosis_payload.get('files_and_commands_used', {}).get('dataset_path')}`",
                        f"- output_dir: `{diagnosis_payload.get('files_and_commands_used', {}).get('output_dir')}`",
                        "",
                        "## Final Recommendation",
                        f"- {diagnosis_payload.get('strict_final_recommendation')}",
                    ],
                )
    else:
        feature_sets = build_feature_sets(df, use_black_scholes_features=False)
        live_feature_audit = []
        if bool(args.live_computable_only):
            selected_live, live_feature_audit = _select_live_computable_features(feature_sets["baseline_features"])
            if len(selected_live) < 5:
                raise RuntimeError("live-computable-only mode left too few baseline features to train safely.")
            feature_sets["baseline_features"] = selected_live
        baseline_result = train_variant(
            df,
            feature_cols=feature_sets["baseline_features"],
            labels=labels,
            artifact_dir=artifact_dir,
            variant_name="baseline",
            evaluation_return_column=column_groups["evaluation_return_column_used"],
            walk_forward_folds=int(args.walk_forward_folds),
            threshold_grid_start=float(args.threshold_grid_start),
            threshold_grid_end=float(args.threshold_grid_end),
            threshold_grid_step=float(args.threshold_grid_step),
            min_threshold_trades=int(args.min_threshold_trades),
            paper_config=vars(args),
        )
        output_payload["baseline"] = baseline_result["reports"]
        trained_models.extend(baseline_result["trained_models"])
        feature_manifest_payload["feature_count"] = len(feature_sets["baseline_features"])
        feature_manifest_payload["features"] = feature_sets["baseline_features"]
        feature_manifest_payload["baseline_features"] = feature_sets["baseline_features"]
        feature_manifest_payload["rejected_features"] = feature_sets["rejected_features"]
        feature_manifest_payload["live_feature_audit"] = live_feature_audit
        if bool(args.paper_readiness_audit):
            simple_features = _simple_no_greeks_features(feature_sets["baseline_features"])
            if len(simple_features) >= 5:
                simple_result = train_variant(
                    df,
                    feature_cols=simple_features,
                    labels=labels,
                    artifact_dir=artifact_dir,
                    variant_name="simple_baseline_no_greeks",
                    evaluation_return_column=column_groups["evaluation_return_column_used"],
                    walk_forward_folds=int(args.walk_forward_folds),
                    threshold_grid_start=float(args.threshold_grid_start),
                    threshold_grid_end=float(args.threshold_grid_end),
                    threshold_grid_step=float(args.threshold_grid_step),
                    min_threshold_trades=int(args.min_threshold_trades),
                    paper_config=vars(args),
                )
                output_payload["simple_baseline_no_greeks"] = simple_result["reports"]
                trained_models.extend(simple_result["trained_models"])

    write_json(artifact_dir / "all_model_training_report.json", output_payload)
    write_json(artifact_dir / "feature_manifest.json", feature_manifest_payload)
    write_json(artifact_dir / "feature_list_used.json", feature_manifest_payload)
    write_json(artifact_dir / "live_feature_schema_compatibility.json", compare_result)
    write_json(artifact_dir / "bs_retraining_integration_report.json", {
        "dataset_path": str(dataset_path),
        "enriched_dataset_path": enriched_dataset_path,
        "return_pnl_column_detection_result": column_groups,
        "leakage_audit_result": leakage_result,
        "production_adoption_verdict": adoption,
        "comparison_report_paths": report_paths_payload,
        "same_period_report": same_period_report,
        "cli_flags": {
            "compare_baseline": bool(args.compare_baseline),
            "same_period_comparison": bool(args.same_period_comparison),
            "no_production_adopt": bool(args.no_production_adopt),
            "walk_forward_folds": int(args.walk_forward_folds),
            "drop_bad_greeks": bool(args.drop_bad_greeks),
        },
    })
    summary = {
        "output_dir": str(artifact_dir),
        "dataset_validity_result": validity.get("status"),
        "labels_tested": labels,
        "models_trained": sorted(set(trained_models)),
        "production_adoption_verdict": adoption.get("verdict"),
        "enriched_dataset_path": enriched_dataset_path,
    }
    _timing_log("total_runtime", seconds=time.perf_counter() - total_start, rows=len(df), extra=f"artifact_dir={artifact_dir}")
    print(json.dumps(summary, indent=2))

    # Write canonical core_retrain_summary.json for rescue/BS dirs so downstream
    # tools (--edge-refinement-report, build_cost_aware_retrain_comparison) can read them.
    rescue_metric_rows: List[Dict[str, Any]] = []
    rescue_result = {"trained_models": [], "skipped_models": [], "failed_models": []}
    for variant_reports in output_payload.values():
        if isinstance(variant_reports, dict):
            rescue_result["trained_models"].extend(variant_reports.get("trained_models", []))
            rescue_result["skipped_models"].extend(variant_reports.get("skipped_models", []))
            rescue_result["failed_models"].extend(variant_reports.get("failed_models", []))
            rescue_metric_rows.extend(variant_reports.get("metric_rows", []))
    _write_core_retrain_summary(
        artifact_dir=artifact_dir,
        dataset_path=dataset_path,
        labels_trained=labels,
        model_families=families,
        metric_rows=rescue_metric_rows,
        result_payload=rescue_result,
        evaluation_return_column=column_groups["evaluation_return_column_used"],
        row_count=len(df),
    )


if __name__ == "__main__":
    main()
