from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cost_model import CostModel
from label_policies import build_label_dataset
from market_data import Candle
from ml_pipeline import TARGET_FEATURES, build_supervised_dataset_v2, verify_no_lookahead_leakage
from retraining_validation import (
    aggregate_trade_metrics,
    classification_metrics,
    generate_purged_embargoed_cv_splits,
    optimize_threshold_for_economic_edge,
)

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("scikit-learn is required for the benchmark runner") from exc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOG = logging.getLogger("train_window_benchmark")


LOOKBACK_BARS = 30
HORIZON_BARS = 45
WF_SPLITS = 5
WF_MIN_TRAIN = 250
WF_MIN_TEST = 50
TIMEZONE = "Asia/Kolkata"
POLICIES = ["trade_quality_binary", "abstain_allowed_trade_quality"]
FEATURE_VARIANTS = ["C_regime_features", "F_final_selected_features"]
THRESHOLD_MIN_TRADES = 20
PROFIT_TARGET = 0.01
STOP_LOSS = 0.005
BASE_DATASET_CACHE: Dict[str, Any] = {}
CHECKPOINT_PATH = REPORTS_DIR / ".benchmark_checkpoint.json"
THRESHOLD_GRID = [round(step / 100.0, 2) for step in range(1, 100)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def checkpoint_key(policy: str, variant: str, fold_idx: int, threshold: float) -> str:
    return f"{policy}|{variant}|{int(fold_idx)}|{float(threshold):.2f}"


def fold_cache_key(policy: str, variant: str, fold_idx: int) -> str:
    return f"{policy}|{variant}|{int(fold_idx)}"


def load_checkpoint() -> Dict[str, Any]:
    if not CHECKPOINT_PATH.exists():
        return {
            "schema_version": 1,
            "completed": {},
            "fold_predictions": {},
        }
    payload = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    payload.setdefault("schema_version", 1)
    payload.setdefault("completed", {})
    payload.setdefault("fold_predictions", {})
    return payload


def save_checkpoint(checkpoint: Dict[str, Any]) -> None:
    atomic_write_json(CHECKPOINT_PATH, checkpoint)


def _parse_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"Invalid timestamp: {value!r}")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(TIMEZONE)
    else:
        timestamp = timestamp.tz_convert(TIMEZONE)
    return timestamp


def _frame_to_candles(frame: pd.DataFrame) -> List[Candle]:
    candles: List[Candle] = []
    for row in frame.itertuples():
        candles.append(
            Candle(
                time=row.timestamp.to_pydatetime(),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(getattr(row, "volume", 0.0) or 0.0),
            )
        )
    return candles


def load_market_frame() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("candles_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(payload.get("candles") or [])
        except Exception as exc:
            LOG.warning("Failed to load %s: %s", path.name, exc)
    if rows:
        frame = pd.DataFrame(rows)
        frame["timestamp"] = frame["time"].map(_parse_timestamp)
        frame["volume"] = pd.to_numeric(frame.get("volume", 0.0), errors="coerce").fillna(0.0)
        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
        frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        LOG.info("Loaded benchmark frame from NIFTY JSON candle cache (%s rows)", len(frame))
        return frame[["timestamp", "open", "high", "low", "close", "volume"]]

    parquet_candidates = sorted(
        [path for path in DATA_DIR.rglob("*.parquet") if "NIFTY" in path.name.upper() and "BANKNIFTY" not in path.name.upper()]
    )
    for path in parquet_candidates:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        timestamp_col = next((col for col in ("timestamp", "time", "datetime", "ts") if col in frame.columns), None)
        price_cols = {"open", "high", "low", "close"}
        if timestamp_col is None or not price_cols.issubset(set(frame.columns)):
            continue
        working = frame.copy()
        working["timestamp"] = pd.to_datetime(working[timestamp_col], errors="coerce")
        working = working.dropna(subset=["timestamp", "open", "high", "low", "close"])
        if working["timestamp"].dt.tz is None:
            working["timestamp"] = working["timestamp"].dt.tz_localize(TIMEZONE)
        else:
            working["timestamp"] = working["timestamp"].dt.tz_convert(TIMEZONE)
        working = working.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        LOG.info("Loaded benchmark frame from parquet: %s (%s rows)", path.name, len(working))
        return working[["timestamp", "open", "high", "low", "close", "volume"] if "volume" in working.columns else ["timestamp", "open", "high", "low", "close"]]
    raise RuntimeError("No NIFTY benchmark dataset found in data/ as candles_*.json or NIFTY parquet")


def compute_labeling_params(candles: Sequence[Candle]) -> Dict[str, float]:
    highs = [float(c.high) for c in candles]
    lows = [float(c.low) for c in candles]
    closes = [float(c.close) for c in candles]
    if not closes:
        return {"tb_profit_target_pct": PROFIT_TARGET, "tb_stop_loss_pct": STOP_LOSS}
    rolling_true_ranges = [
        max(high - low, abs(high - prev_close), abs(low - prev_close))
        for high, low, prev_close in zip(highs[1:], lows[1:], closes[:-1])
    ]
    atr_proxy = float(np.mean(rolling_true_ranges[-14:])) if rolling_true_ranges else closes[-1] * 0.01
    latest_close = max(float(closes[-1]), 1e-7)
    return {
        "tb_profit_target_pct": max(PROFIT_TARGET, (1.5 * atr_proxy) / latest_close),
        "tb_stop_loss_pct": max(STOP_LOSS, (2.0 * atr_proxy) / latest_close),
    }


def build_dataset(candles: Sequence[Candle], *, label_policy: str) -> tuple[list[list[float]], list[int], list[str], dict[str, Any]]:
    cache_key = f"{len(candles)}:{candles[0].time.isoformat() if candles else 'na'}:{candles[-1].time.isoformat() if candles else 'na'}"
    cached = BASE_DATASET_CACHE.get(cache_key)
    if cached is None:
        params = compute_labeling_params(candles)
        X_full, _, feature_names = build_supervised_dataset_v2(
            candles,
            lookback=LOOKBACK_BARS,
            horizon=HORIZON_BARS,
            use_triple_barrier=True,
            **params,
        )
        cached = {"X_full": X_full, "feature_names": feature_names}
        BASE_DATASET_CACHE[cache_key] = cached
    else:
        X_full = cached["X_full"]
        feature_names = cached["feature_names"]
    dataset = build_label_dataset(
        candles,
        policy_name=label_policy,
        lookback=LOOKBACK_BARS,
        horizon=HORIZON_BARS,
        cost_model=CostModel(),
        include_features=False,
    )
    X = [list(map(float, X_full[idx])) for idx in dataset.sample_indices if idx < len(X_full)]
    meta = {
        "label_policy": dataset.policy.name,
        "label_policy_version": dataset.policy.version,
        "neutral_samples_dropped": int(dataset.neutral_samples_dropped or 0),
        "observations": dataset.observations,
        "label_distribution": dataset.label_distribution,
        "sample_indices": list(dataset.sample_indices),
    }
    return X, dataset.y, list(feature_names), meta


def load_feature_governance() -> Dict[str, Any]:
    summary_path = REPORTS_DIR / "feature_audit_summary.json"
    corr_path = REPORTS_DIR / "feature_correlation_report.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    correlation = json.loads(corr_path.read_text(encoding="utf-8")) if corr_path.exists() else {}
    return {"summary": summary, "correlation": correlation}


def feature_variants(feature_names: Sequence[str], governance: Dict[str, Any]) -> Dict[str, List[str]]:
    base = [name for name in TARGET_FEATURES if name in feature_names]
    corr = governance.get("correlation") or {}
    redundant = {row.get("drop_candidate") for row in (corr.get("redundant_feature_candidates") or [])}
    constant = set(corr.get("constant_features") or [])
    near_zero = {row.get("feature") for row in (corr.get("near_zero_features") or [])}
    pruned = [name for name in base if name not in redundant and name not in constant and name not in near_zero]

    regime_focus = [
        name
        for name in (
            "volatility_regime_classifier",
            "realized_vol_percentile_60",
            "atr_percentile_60",
            "rolling_range_position_20",
            "dist_to_rolling_low_20",
            "opening_range_breakout_strength",
            "is_closing_session",
        )
        if name in feature_names and name not in pruned
    ]
    final_focus = [
        name
        for name in (
            "opening_range_breakout_strength",
            "dist_from_opening_low_pct",
            "dist_to_rolling_low_20",
            "rolling_range_position_20",
            "realized_vol_percentile_60",
            "volatility_regime_classifier",
            "is_closing_session",
        )
        if name in feature_names and name not in pruned
    ]
    return {
        "C_regime_features": pruned + regime_focus,
        "F_final_selected_features": pruned + final_focus,
    }


def compress_matrix(X_full: Sequence[Sequence[float]], feature_names: Sequence[str], selected_features: Sequence[str]) -> List[List[float]]:
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    indices = [feature_index[name] for name in selected_features if name in feature_index]
    return [[float(row[idx]) for idx in indices] for row in X_full]


def build_split_frame(frame: pd.DataFrame, sample_indices: Sequence[int]) -> pd.DataFrame:
    sample_times = [
        pd.to_datetime(frame["timestamp"].iloc[LOOKBACK_BARS + idx])
        for idx in sample_indices
        if LOOKBACK_BARS + idx < len(frame)
    ]
    index = pd.DatetimeIndex(sample_times)
    if index.tz is None:
        index = index.tz_localize(TIMEZONE)
    else:
        index = index.tz_convert(TIMEZONE)
    return pd.DataFrame(index=index)


def make_model() -> CalibratedClassifierCV:
    base_estimator = RandomForestClassifier(
        n_estimators=40,
        random_state=42,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    return CalibratedClassifierCV(base_estimator, method="sigmoid", cv=3)


def simulate_trade_metrics(y_true: Sequence[int], probabilities: Sequence[float], threshold: float, sample_times: Sequence[pd.Timestamp]) -> Dict[str, float]:
    pnls: List[float] = []
    trade_days: List[Any] = []
    for idx, probability in enumerate(probabilities):
        if float(probability) < float(threshold):
            continue
        pnl = PROFIT_TARGET if int(y_true[idx]) == 1 else -STOP_LOSS
        pnls.append(pnl)
        trade_days.append(pd.to_datetime(sample_times[idx]).date())
    return aggregate_trade_metrics(pnls, trade_days)


def summarize_threshold_payload(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
    sample_times: Sequence[pd.Timestamp],
) -> Dict[str, float]:
    metrics = classification_metrics(y_true, probabilities, threshold=threshold)
    trade_metrics = simulate_trade_metrics(y_true, probabilities, threshold, sample_times)
    metrics.update(trade_metrics)
    metrics["threshold"] = float(threshold)
    metrics["max_drawdown"] = float(metrics.get("drawdown", 0.0))
    return metrics


def evaluate_configuration(
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    split_frame: pd.DataFrame,
    candles: Sequence[Candle],
    *,
    label_policy: str,
    label_meta: Dict[str, Any],
    variant_name: str,
    selected_features: Sequence[str],
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    folds: List[Dict[str, Any]] = []

    split_pairs = generate_purged_embargoed_cv_splits(
        split_frame,
        label_horizon_bars=HORIZON_BARS,
        embargo_pct=0.01,
        n_splits=WF_SPLITS,
    )
    split_pairs = [(train_idx, val_idx) for train_idx, val_idx in split_pairs if len(train_idx) >= WF_MIN_TRAIN and len(val_idx) >= WF_MIN_TEST]
    if not split_pairs:
        return {"ok": False, "reason": "no_valid_purged_splits"}

    started = time.perf_counter()
    X_matrix = np.asarray(X, dtype=float)
    y_array = np.asarray(y, dtype=int)
    split_index = split_frame.index

    for fold_number, (train_idx, val_idx) in enumerate(split_pairs, start=1):
        cache_id = fold_cache_key(label_policy, variant_name, fold_number)
        fold_cache = checkpoint["fold_predictions"].get(cache_id)
        if fold_cache is None:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_matrix[train_idx])
            X_val = scaler.transform(X_matrix[val_idx])
            y_train = y_array[train_idx]
            y_val = y_array[val_idx]
            if len(np.unique(y_train)) < 2:
                continue

            model = make_model()
            model.fit(X_train, y_train)
            probs = [float(row[1]) for row in model.predict_proba(X_val)]
            fold_cache = {
                "fold": fold_number,
                "train_start": pd.to_datetime(split_index[train_idx[0]]).isoformat(),
                "train_end": pd.to_datetime(split_index[val_idx[0] - 1] if len(val_idx) > 0 and val_idx[0] > 0 else split_index[train_idx[-1]]).isoformat(),
                "validation_start": pd.to_datetime(split_index[val_idx[0]]).isoformat(),
                "validation_end": pd.to_datetime(split_index[val_idx[-1]]).isoformat(),
                "train_samples": int(len(train_idx)),
                "validation_samples": int(len(val_idx)),
                "positive_rate": float(np.mean(y_val)),
                "negative_rate": float(1.0 - np.mean(y_val)),
                "y_true": [int(value) for value in y_val.tolist()],
                "probabilities": [float(value) for value in probs],
                "sample_times": [pd.to_datetime(ts).isoformat() for ts in split_index[val_idx]],
            }
            checkpoint["fold_predictions"][cache_id] = fold_cache
            save_checkpoint(checkpoint)

        y_true_fold = [int(value) for value in fold_cache["y_true"]]
        probs_fold = [float(value) for value in fold_cache["probabilities"]]
        sample_times_fold = [pd.to_datetime(value) for value in fold_cache["sample_times"]]
        for threshold in THRESHOLD_GRID:
            completed_id = checkpoint_key(label_policy, variant_name, fold_number, threshold)
            if completed_id in checkpoint["completed"]:
                continue
            threshold_metrics = summarize_threshold_payload(
                y_true_fold,
                probs_fold,
                threshold,
                sample_times_fold,
            )
            checkpoint["completed"][completed_id] = {
                "policy": str(label_policy),
                "variant": str(variant_name),
                "fold_idx": int(fold_number),
                "threshold": float(threshold),
                "metrics": threshold_metrics,
            }
            save_checkpoint(checkpoint)

        base_metrics = summarize_threshold_payload(
            y_true_fold,
            probs_fold,
            0.5,
            sample_times_fold,
        )
        folds.append(
            {
                "fold": int(fold_cache["fold"]),
                "train_start": str(fold_cache["train_start"]),
                "train_end": str(fold_cache["train_end"]),
                "validation_start": str(fold_cache["validation_start"]),
                "validation_end": str(fold_cache["validation_end"]),
                "train_samples": int(fold_cache["train_samples"]),
                "validation_samples": int(fold_cache["validation_samples"]),
                "positive_rate": float(fold_cache["positive_rate"]),
                "negative_rate": float(fold_cache["negative_rate"]),
                **base_metrics,
            }
        )

    fold_caches = [
        checkpoint["fold_predictions"][fold_cache_key(label_policy, variant_name, fold_number)]
        for fold_number in range(1, len(split_pairs) + 1)
        if fold_cache_key(label_policy, variant_name, fold_number) in checkpoint["fold_predictions"]
    ]
    if not fold_caches:
        return {"ok": False, "reason": "no_fold_outputs"}

    aggregate_true = [int(value) for cache in fold_caches for value in cache["y_true"]]
    aggregate_prob = [float(value) for cache in fold_caches for value in cache["probabilities"]]
    aggregate_times = [pd.to_datetime(value) for cache in fold_caches for value in cache["sample_times"]]
    threshold_result = optimize_threshold_for_economic_edge(aggregate_true, aggregate_prob, min_trades=THRESHOLD_MIN_TRADES)
    threshold = float(threshold_result["threshold"])
    aggregate_metrics = summarize_threshold_payload(
        aggregate_true,
        aggregate_prob,
        threshold,
        aggregate_times,
    )
    aggregate_metrics["expectancy_positive"] = bool(float(aggregate_metrics.get("expectancy", 0.0)) > 0.0)
    aggregate_metrics["stable_threshold_valid"] = bool(float(aggregate_metrics.get("trades_count", 0.0)) >= THRESHOLD_MIN_TRADES)

    roc_values = [float(fold["roc_auc"]) for fold in folds]
    f1_values = [float(fold["f1"]) for fold in folds]
    elapsed = time.perf_counter() - started
    return {
        "ok": True,
        "label_policy": label_policy,
        "label_policy_version": label_meta.get("label_policy_version"),
        "neutral_samples_dropped": int(label_meta.get("neutral_samples_dropped") or 0),
        "model": "random_forest_calibrated",
        "variant": variant_name,
        "feature_count": len(selected_features),
        "features": list(selected_features),
        "sample_count": len(y),
        "training_time_seconds": round(elapsed, 4),
        "metrics": aggregate_metrics,
        "stability": {
            "roc_auc_std": float(np.std(roc_values)) if len(roc_values) > 1 else 0.0,
            "f1_std": float(np.std(f1_values)) if len(f1_values) > 1 else 0.0,
        },
        "folds": folds,
    }


def choose_best_row(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid_rows = [row for row in rows if row.get("ok")]
    if not valid_rows:
        return None
    survivors = [
        row for row in valid_rows
        if bool(row["metrics"].get("stable_threshold_valid"))
        and bool(row["metrics"].get("expectancy_positive"))
    ]
    pool = survivors or valid_rows
    ranked = sorted(
        pool,
        key=lambda row: (
            -float(row["metrics"].get("f1", 0.0)),
            -float(row["metrics"].get("expectancy", 0.0)),
            -float(row["metrics"].get("profit_factor", 0.0)),
            -float(row["metrics"].get("roc_auc", 0.0)),
            float(row["stability"].get("f1_std", 0.0)),
        ),
    )
    return ranked[0]


def render_markdown(report: Dict[str, Any]) -> str:
    rows = report.get("benchmark_matrix", {}).get("rows", [])
    lines = [
        "# Final Label Redesign And Model Adaptation Report",
        "",
        "## Fresh Walk-Forward Benchmark",
        "",
        "| Label Policy | Variant | Model | Threshold | ROC-AUC | Accuracy | Precision | Recall | F1 | Trades | Profit Factor | Expectancy | Sharpe | Max Drawdown | ROC-AUC Std | F1 Std |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row.get("metrics", {})
        stability = row.get("stability", {})
        lines.append(
            f"| {row.get('label_policy')} | {row.get('variant')} | {row.get('model')} | {metrics.get('threshold', 0.5):.2f} | "
            f"{metrics.get('roc_auc', 0.0):.4f} | {metrics.get('accuracy', 0.0):.4f} | {metrics.get('precision', 0.0):.4f} | "
            f"{metrics.get('recall', 0.0):.4f} | {metrics.get('f1', 0.0):.4f} | {int(metrics.get('trades_count', 0.0))} | "
            f"{metrics.get('profit_factor', 0.0):.4f} | {metrics.get('expectancy', 0.0):.4f} | {metrics.get('sharpe', 0.0):.4f} | "
            f"{metrics.get('drawdown', 0.0):.4f} | {stability.get('roc_auc_std', 0.0):.4f} | {stability.get('f1_std', 0.0):.4f} |"
        )
    best = report.get("benchmark_matrix", {}).get("best")
    if best:
        metrics = best.get("metrics", {})
        lines.extend(
            [
                "",
                "## Preferred Architecture",
                "",
                f"- Label policy: `{best.get('label_policy')}`",
                f"- Feature variant: `{best.get('variant')}`",
                f"- Threshold: `{metrics.get('threshold', 0.5):.2f}`",
                f"- ROC-AUC: `{metrics.get('roc_auc', 0.0):.4f}`",
                f"- F1: `{metrics.get('f1', 0.0):.4f}`",
                f"- Trades: `{int(metrics.get('trades_count', 0.0))}`",
                f"- Expectancy: `{metrics.get('expectancy', 0.0):.4f}`",
            ]
        )
    audit = report.get("label_policy_safety_audit") or {}
    lines.extend(
        [
            "",
            "## Audit Hook",
            "",
            f"- Passed: `{audit.get('passed')}`",
            f"- Purged/embargoed splits available: `{all(bool(row.get('purged_embargo_splits_available')) for row in audit.get('policies', [])) if audit.get('policies') else False}`",
        ]
    )
    return "\n".join(lines)


def run_safety_audit() -> Dict[str, Any]:
    command = [sys.executable, str(REPO_ROOT / "tools" / "audit_label_policy_safety.py")]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"audit_label_policy_safety.py failed: {completed.stderr or completed.stdout}")
    report_path = REPORTS_DIR / "label_policy_safety_audit.json"
    return json.loads(report_path.read_text(encoding="utf-8"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run fresh purged walk-forward benchmark matrix on clean NIFTY dataset.")
    parser.add_argument("--no-audit", action="store_true", help="Skip trailing audit hook")
    args = parser.parse_args(argv)

    frame = load_market_frame()
    frame["timestamp"] = frame["timestamp"].map(_parse_timestamp)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    candles = _frame_to_candles(frame)
    checkpoint = load_checkpoint()
    completed_count = len(checkpoint.get("completed", {}))
    cached_folds = len(checkpoint.get("fold_predictions", {}))
    if CHECKPOINT_PATH.exists():
        LOG.info(
            "Resuming benchmark from checkpoint: completed_threshold_rows=%s cached_folds=%s file=%s",
            completed_count,
            cached_folds,
            CHECKPOINT_PATH,
        )
    else:
        LOG.info("Starting benchmark with a clean checkpoint state")

    leakage = verify_no_lookahead_leakage(
        candles[-min(len(candles), 1500):],
        build_supervised_dataset_v2,
        lookback=LOOKBACK_BARS,
        horizon=HORIZON_BARS,
    )
    if not leakage.get("passed"):
        raise RuntimeError(f"Leakage audit failed before benchmark: {leakage}")

    governance = load_feature_governance()
    rows: List[Dict[str, Any]] = []
    for label_policy in POLICIES:
        X_full, y, feature_names, label_meta = build_dataset(candles, label_policy=label_policy)
        variants = feature_variants(feature_names, governance)
        for variant_name in FEATURE_VARIANTS:
            selected_features = variants.get(variant_name) or []
            if not selected_features:
                LOG.warning("Skipping empty feature variant: %s", variant_name)
                continue
            X_variant = compress_matrix(X_full, feature_names, selected_features)
            split_frame = build_split_frame(frame, label_meta.get("sample_indices") or [])
            row = evaluate_configuration(
                X_variant,
                y,
                split_frame,
                candles,
                label_policy=label_policy,
                label_meta=label_meta,
                variant_name=variant_name,
                selected_features=selected_features,
                checkpoint=checkpoint,
            )
            if row.get("ok"):
                rows.append(row)

    best = choose_best_row(rows)
    final_report_path = REPORTS_DIR / "final_label_redesign_and_model_adaptation_report.json"
    existing_report = json.loads(final_report_path.read_text(encoding="utf-8")) if final_report_path.exists() else {}
    benchmark_report = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "dataset": {
            "rows": int(len(frame)),
            "start": pd.to_datetime(frame["timestamp"].iloc[0]).isoformat(),
            "end": pd.to_datetime(frame["timestamp"].iloc[-1]).isoformat(),
            "timezone": str(pd.DatetimeIndex(frame["timestamp"]).tz),
        },
        "models": ["random_forest_calibrated"],
        "label_policies": POLICIES,
        "feature_variants": FEATURE_VARIANTS,
        "rows": rows,
        "best": best,
    }
    existing_report["benchmark_matrix"] = benchmark_report

    if not args.no_audit:
        existing_report["label_policy_safety_audit"] = run_safety_audit()

    write_json(final_report_path, existing_report)
    write_markdown(REPORTS_DIR / "final_label_redesign_and_model_adaptation_report.md", render_markdown(existing_report))
    write_json(REPORTS_DIR / "train_window_benchmark.json", benchmark_report)
    write_markdown(REPORTS_DIR / "train_window_benchmark.md", render_markdown({"benchmark_matrix": benchmark_report, "label_policy_safety_audit": existing_report.get("label_policy_safety_audit", {})}))
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    LOG.info("Saved benchmark outputs to reports/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
