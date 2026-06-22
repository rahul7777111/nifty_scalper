from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(REPO_ROOT / ".scalper.env")
except Exception:
    pass

from cost_model import CostModel
from label_policies import available_label_policies, build_label_dataset
from market_data import Candle
from ml_pipeline import TARGET_FEATURES, build_market_feature_vector, verify_no_lookahead_leakage
from ml_signals import (
    CalibratedClassifierCV,
    CatBoostClassifier,
    LogisticRegression,
    MLModelBundle,
    RandomForestClassifier,
    WeightedEnsembleClassifier,
    XGBClassifier,
    load_model,
)
from retraining_validation import classification_metrics, generate_purged_embargoed_cv_splits, optimize_threshold_for_economic_edge


TIMEZONE = "Asia/Kolkata"
SESSION_START = time(9, 15)
SESSION_END = time(15, 29)
EXPECTED_BARS_PER_SESSION = 375
LOOKBACK_BARS = 30
HORIZON_BARS = 45
LABEL_POLICY = os.getenv("MSTOCK_RETRAIN_LABEL_POLICY", "current_triple_barrier").strip() or "current_triple_barrier"
MIN_PER_DAY_FETCH = 200
MIN_CLASS_SHARE = 0.02
THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
LEAKAGE_AUDIT_MAX_CANDLES = 1500


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOG = logging.getLogger("retrain_nifty_1year")


@dataclass
class SourceSummary:
    path: str
    rows: int
    earliest: Optional[str]
    latest: Optional[str]
    interval_guess: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely retrain NIFTY ML models on 1 year of minute data.")
    parser.add_argument("--label-policy", default=LABEL_POLICY, choices=available_label_policies())
    parser.add_argument("--skip-broker-backfill", action="store_true", help="Do not fetch missing local minute data from m.Stock.")
    parser.add_argument("--skip-training", action="store_true", help="Only prepare data and diagnostics.")
    parser.add_argument("--max-models", type=int, default=0, help="Optional cap on how many model types to train.")
    return parser.parse_args()


def ensure_json_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): ensure_json_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [ensure_json_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [ensure_json_serializable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()  # .tolist() returns Python native types
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ensure_json_serializable(payload), indent=2), encoding="utf-8")


def load_json_cache_rows() -> Tuple[pd.DataFrame, List[SourceSummary]]:
    rows: List[Dict[str, Any]] = []
    summaries: List[SourceSummary] = []
    for path in sorted(DATA_DIR.glob("candles_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            candles = payload.get("candles") or []
        except Exception as exc:
            LOG.warning("Failed to load %s: %s", path.name, exc)
            continue
        for item in candles:
            rows.append(
                {
                    "timestamp": item.get("time"),
                    "open": item.get("open"),
                    "high": item.get("high"),
                    "low": item.get("low"),
                    "close": item.get("close"),
                    "volume": item.get("volume"),
                    "source": path.name,
                }
            )
        summaries.append(SourceSummary(path=str(path), rows=len(candles), earliest=None, latest=None, interval_guess="1m_json"))
    frame = pd.DataFrame(rows)
    return frame, summaries


def load_tabular_sources() -> Tuple[pd.DataFrame, List[SourceSummary]]:
    frames: List[pd.DataFrame] = []
    summaries: List[SourceSummary] = []
    for path in sorted((DATA_DIR / "historical").glob("NIFTY_*.*")):
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".parquet"}:
            continue
        try:
            frame = pd.read_csv(path) if suffix == ".csv" else pd.read_parquet(path)
        except Exception as exc:
            LOG.warning("Failed to load %s: %s", path.name, exc)
            continue
        ts_col = next((col for col in ("timestamp", "time", "datetime", "date", "Date") if col in frame.columns), None)
        if ts_col is None or not {"open", "high", "low", "close"}.issubset(frame.columns):
            continue
        working = frame.copy()
        working["timestamp"] = working[ts_col]
        if "volume" not in working.columns:
            working["volume"] = 0.0
        working["source"] = path.name
        working = working[["timestamp", "open", "high", "low", "close", "volume", "source"]]
        interval_guess = "daily" if "daily" in path.stem.lower() else "1m"
        summaries.append(
            SourceSummary(
                path=str(path),
                rows=len(working),
                earliest=str(pd.to_datetime(working["timestamp"], errors="coerce").min()),
                latest=str(pd.to_datetime(working["timestamp"], errors="coerce").max()),
                interval_guess=interval_guess,
            )
        )
        frames.append(working)
    if not frames:
        return pd.DataFrame(), summaries
    return pd.concat(frames, ignore_index=True), summaries


def normalize_timestamp_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is None:
        return parsed.dt.tz_localize(TIMEZONE, nonexistent="NaT", ambiguous="NaT")
    return parsed.dt.tz_convert(TIMEZONE)


def interval_guess(frame: pd.DataFrame) -> str:
    if frame.empty or "timestamp" not in frame.columns:
        return "unknown"
    diffs = frame["timestamp"].sort_values().diff().dropna()
    if diffs.empty:
        return "unknown"
    median_seconds = diffs.median().total_seconds()
    if median_seconds <= 60:
        return "1m"
    if median_seconds <= 300:
        return "5m"
    if median_seconds <= 3600:
        return "intraday"
    return "daily"


def prepare_minute_frame() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    json_frame, json_sources = load_json_cache_rows()
    tabular_frame, tabular_sources = load_tabular_sources()
    combined = pd.concat([json_frame, tabular_frame], ignore_index=True) if not json_frame.empty or not tabular_frame.empty else pd.DataFrame()
    if combined.empty:
        raise RuntimeError("No local NIFTY sources found under data/.")

    combined["timestamp"] = normalize_timestamp_series(combined["timestamp"])
    for column in ("open", "high", "low", "close", "volume"):
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    combined = combined.dropna(subset=["timestamp", "open", "high", "low", "close"])
    combined["local_time"] = combined["timestamp"].dt.tz_convert(TIMEZONE)
    combined["time_only"] = combined["local_time"].dt.time
    combined["weekday"] = combined["local_time"].dt.weekday

    minute_candidates = combined[
        combined["source"].str.contains("candles_|1m", case=False, na=False)
    ].copy()
    minute_candidates = minute_candidates[
        (minute_candidates["weekday"] < 5)
        & (minute_candidates["time_only"] >= SESSION_START)
        & (minute_candidates["time_only"] <= SESSION_END)
    ]
    minute_candidates = minute_candidates.sort_values("timestamp", kind="stable")
    duplicates_before = int(minute_candidates.duplicated(subset=["timestamp"]).sum())
    minute_candidates = minute_candidates.drop_duplicates(subset=["timestamp"], keep="last")
    minute_candidates["trading_day"] = minute_candidates["timestamp"].dt.tz_convert(TIMEZONE).dt.date

    discovery = {
        "sources": [asdict(s) for s in (json_sources + tabular_sources)],
        "minute_source_rows": int(len(minute_candidates)),
        "minute_interval_detected": interval_guess(minute_candidates),
        "duplicates_dropped_on_load": duplicates_before,
    }
    return minute_candidates.reset_index(drop=True), discovery


def trading_days_between(start_day: date, end_day: date) -> List[date]:
    days: List[date] = []
    cursor = start_day
    while cursor <= end_day:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def backfill_missing_days(days: Sequence[date]) -> Dict[str, Any]:
    if not days:
        return {"fetched_days": [], "empty_days": [], "failed_days": []}
    from collect_training_data import build_client, fetch_day_candles, save_day_candles

    client = build_client()
    fetched_days: List[str] = []
    empty_days: List[str] = []
    failed_days: List[Dict[str, str]] = []
    for idx, trading_day in enumerate(days, start=1):
        LOG.info("Broker backfill [%s/%s] %s", idx, len(days), trading_day)
        try:
            candles = fetch_day_candles(client, trading_day)
        except Exception as exc:
            failed_days.append({"day": str(trading_day), "error": str(exc)})
            continue
        if len(candles) >= MIN_PER_DAY_FETCH:
            save_day_candles(candles, trading_day)
            fetched_days.append(str(trading_day))
        else:
            empty_days.append(str(trading_day))
    return {"fetched_days": fetched_days, "empty_days": empty_days, "failed_days": failed_days}


def clean_frame(frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    working = frame.copy()
    invalid_mask = (
        (working["high"] < working["low"])
        | (working["open"] <= 0)
        | (working["high"] <= 0)
        | (working["low"] <= 0)
        | (working["close"] <= 0)
        | (working["close"] > working["high"])
        | (working["close"] < working["low"])
        | (working["open"] > working["high"])
        | (working["open"] < working["low"])
    )
    invalid_rows = int(invalid_mask.sum())
    working = working.loc[~invalid_mask].copy()
    working = working.sort_values("timestamp", kind="stable").drop_duplicates(subset=["timestamp"], keep="last")
    working["trading_day"] = working["timestamp"].dt.tz_convert(TIMEZONE).dt.date
    daily_counts = working.groupby("trading_day").size().sort_index()

    session_gaps: List[Dict[str, Any]] = []
    missing_bar_total = 0
    for trading_day, day_frame in working.groupby("trading_day", sort=True):
        ts = day_frame["timestamp"].sort_values()
        diffs = ts.diff().dropna()
        gap_points = diffs[diffs > pd.Timedelta(minutes=1)]
        gap_minutes = int(sum(int(delta.total_seconds() // 60) - 1 for delta in gap_points))
        missing_bar_total += max(0, gap_minutes)
        if not gap_points.empty:
            session_gaps.append(
                {
                    "day": str(trading_day),
                    "gaps_detected": int(len(gap_points)),
                    "missing_bars_estimate": int(gap_minutes),
                    "first_timestamp": str(ts.iloc[0]),
                    "last_timestamp": str(ts.iloc[-1]),
                }
            )

    quality = {
        "earliest_timestamp": str(working["timestamp"].min()),
        "latest_timestamp": str(working["timestamp"].max()),
        "total_candles": int(len(working)),
        "trading_days_count": int(daily_counts.shape[0]),
        "interval_detected": interval_guess(working),
        "duplicate_timestamps_count": 0,
        "missing_invalid_ohlc_rows_count": invalid_rows,
        "daily_counts_summary": {
            "min": int(daily_counts.min()) if not daily_counts.empty else 0,
            "median": int(daily_counts.median()) if not daily_counts.empty else 0,
            "max": int(daily_counts.max()) if not daily_counts.empty else 0,
        },
        "session_gap_days": session_gaps[:100],
        "session_gap_days_total": int(len(session_gaps)),
        "session_missing_bar_estimate": int(missing_bar_total),
    }
    return working.reset_index(drop=True), quality


def select_one_year_window(frame: pd.DataFrame, *, allow_backfill: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if frame.empty:
        raise RuntimeError("No minute candles available after source loading.")

    latest_ts = frame["timestamp"].max()
    latest_day = latest_ts.tz_convert(TIMEZONE).date()
    start_day = (latest_ts.tz_convert(TIMEZONE) - pd.DateOffset(years=1) + pd.Timedelta(days=1)).date()
    existing_days = set(frame["trading_day"].tolist())
    target_weekdays = trading_days_between(start_day, latest_day)
    missing_weekdays = [day for day in target_weekdays if day not in existing_days]

    backfill_report: Dict[str, Any] = {"required": bool(missing_weekdays), "attempted": False}
    broker_empty_days: set[date] = set()
    if missing_weekdays and allow_backfill:
        LOG.info("Missing %s weekday sessions in target window. Starting broker backfill.", len(missing_weekdays))
        backfill_report["attempted"] = True
        backfill_report["result"] = backfill_missing_days(missing_weekdays)
        broker_empty_days = {
            datetime.strptime(day_str, "%Y-%m-%d").date()
            for day_str in (backfill_report["result"].get("empty_days") or [])
        }
        refreshed, _ = prepare_minute_frame()
        refreshed, _ = clean_frame(refreshed)
        frame = refreshed
        existing_days = set(frame["trading_day"].tolist())
        missing_weekdays = [day for day in target_weekdays if day not in existing_days]
    else:
        backfill_report["result"] = {"fetched_days": [], "empty_days": [], "failed_days": []}

    selected = frame[(frame["trading_day"] >= start_day) & (frame["trading_day"] <= latest_day)].copy()
    calendar_days = int((latest_day - start_day).days + 1)
    missing_after_holiday_filter = [day for day in missing_weekdays if day not in broker_empty_days]
    target_complete = len(missing_after_holiday_filter) == 0
    report = {
        "requested_start_day": str(start_day),
        "requested_end_day": str(latest_day),
        "calendar_days": calendar_days,
        "target_weekday_count": int(len(target_weekdays)),
        "available_trading_days": int(selected["trading_day"].nunique()),
        "missing_weekdays_after_backfill": [str(day) for day in missing_weekdays],
        "broker_confirmed_non_trading_days": [str(day) for day in sorted(broker_empty_days)],
        "missing_data_days_after_holiday_filter": [str(day) for day in missing_after_holiday_filter],
        "target_complete": target_complete,
        "backfill": backfill_report,
    }
    if not target_complete:
        max_start = frame["trading_day"].min()
        report["max_available_range"] = {
            "start_day": str(max_start),
            "end_day": str(latest_day),
            "calendar_days": int((latest_day - max_start).days + 1),
            "trading_days": int(frame["trading_day"].nunique()),
        }
        raise RuntimeError(json.dumps(report, indent=2))
    return selected.reset_index(drop=True), report


def frame_to_candles(frame: pd.DataFrame) -> List[Candle]:
    candles: List[Candle] = []
    for row in frame.itertuples(index=False):
        ts = row.timestamp.tz_convert(TIMEZONE).tz_localize(None).to_pydatetime()
        candles.append(
            Candle(
                time=ts,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume or 0.0),
            )
        )
    return candles


def build_dataset_payload(candles: Sequence[Candle], *, label_policy: str) -> Dict[str, Any]:
    label_dataset = build_label_dataset(
        candles,
        policy_name=label_policy,
        lookback=LOOKBACK_BARS,
        horizon=HORIZON_BARS,
        cost_model=CostModel(),
        include_features=False,
    )
    sample_indices = list(label_dataset.sample_indices)
    feature_rows: List[List[float]] = []
    feature_names: List[str] = []
    for pos, sample_idx in enumerate(sample_indices, start=1):
        candle_idx = sample_idx + LOOKBACK_BARS
        window = candles[candle_idx - LOOKBACK_BARS + 1 : candle_idx + 1]
        features, names = build_market_feature_vector(window, lookback=LOOKBACK_BARS)
        if not feature_names:
            feature_names = list(names)
        feature_rows.append([float(value) for value in features])
        if pos % 10000 == 0:
            LOG.info("Feature generation progress: %s / %s samples", pos, len(sample_indices))
    X = np.asarray(feature_rows, dtype=np.float32)
    y = np.asarray([int(v) for v in label_dataset.y], dtype=np.int32)
    timestamps = [candles[idx + LOOKBACK_BARS].time for idx in sample_indices if (idx + LOOKBACK_BARS) < len(candles)]
    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "timestamps": timestamps,
        "forward_returns": [
            float(label_dataset.observations[idx].forward_return)
            for idx in sample_indices
            if idx < len(label_dataset.observations)
        ],
        "label_distribution": label_dataset.label_distribution,
        "neutral_samples_dropped": int(label_dataset.neutral_samples_dropped or 0),
        "observations": label_dataset.observations,
        "sample_indices": sample_indices,
    }


def feature_manifest(X: np.ndarray, feature_names: Sequence[str]) -> Dict[str, Any]:
    frame = pd.DataFrame(X, columns=list(feature_names))
    manifest = []
    for name in frame.columns:
        col = frame[name]
        manifest.append(
            {
                "feature": str(name),
                "dtype": str(col.dtype),
                "missing_pct": float(col.isna().mean() * 100.0),
            }
        )
    return {"feature_count": int(len(feature_names)), "features": manifest}


def compare_feature_compatibility(feature_names: Sequence[str]) -> Dict[str, Any]:
    current_bundle = load_model(str(REPO_ROOT / "ml_signal_model.pkl"))
    current_features = list(getattr(current_bundle, "feature_names", []) or [])
    target_features = list(TARGET_FEATURES)
    compatibility = {
        "production_model_present": bool(current_bundle is not None),
        "production_feature_count": int(len(current_features)),
        "dataset_feature_count": int(len(feature_names)),
        "target_feature_count": int(len(target_features)),
        "matches_production": current_features == list(feature_names) if current_features else False,
        "missing_vs_production": [name for name in current_features if name not in feature_names][:25],
        "extra_vs_production": [name for name in feature_names if name not in current_features][:25],
        "missing_vs_target": [name for name in target_features if name not in feature_names][:25],
        "extra_vs_target": [name for name in feature_names if name not in target_features][:25],
    }
    if current_features and current_features != list(feature_names):
        raise RuntimeError(json.dumps({"feature_compatibility_failed": compatibility}, indent=2))
    return compatibility


def build_label_report(y: np.ndarray) -> Dict[str, Any]:
    unique, counts = np.unique(y, return_counts=True)
    total = int(counts.sum()) if counts.size else 0
    distribution = {str(int(label)): int(count) for label, count in zip(unique, counts)}
    percentages = {label: (count / total) * 100.0 if total else 0.0 for label, count in distribution.items()}
    minority = min(distribution.values()) if distribution else 0
    majority = max(distribution.values()) if distribution else 0
    imbalance_ratio = float(majority / minority) if minority else math.inf
    report = {
        "class_counts": distribution,
        "class_percentages": percentages,
        "imbalance_ratio": imbalance_ratio,
    }
    if len(distribution) < 2:
        raise RuntimeError(json.dumps({"label_quality_failed": {**report, "reason": "single_class_labels"}}, indent=2))
    min_share = min(float(v) / total for v in distribution.values())
    if min_share < MIN_CLASS_SHARE:
        raise RuntimeError(json.dumps({"label_quality_failed": {**report, "reason": "minority_class_too_small"}}, indent=2))
    return report


def chronological_split(X: np.ndarray, y: np.ndarray, timestamps: Sequence[datetime]) -> Dict[str, Any]:
    n = len(X)
    train_end = max(1, int(n * 0.70))
    val_end = max(train_end + 1, int(n * 0.85))
    val_end = min(val_end, n - 1)
    splits = {
        "train": slice(0, train_end),
        "validation": slice(train_end, val_end),
        "test": slice(val_end, n),
    }
    ranges = {}
    for name, subset in splits.items():
        subset_times = timestamps[subset]
        ranges[name] = {
            "rows": int(len(subset_times)),
            "start": str(subset_times[0]) if subset_times else None,
            "end": str(subset_times[-1]) if subset_times else None,
        }
    return {"slices": splits, "ranges": ranges}


def scale_train_val_test(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float], List[float]]:
    mean = X_train.mean(axis=0).astype(np.float32)
    std = X_train.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-9, 1e-9, std)
    return (
        ((X_train - mean) / std).astype(np.float32),
        ((X_val - mean) / std).astype(np.float32),
        ((X_test - mean) / std).astype(np.float32),
        mean.tolist(),
        std.tolist(),
    )


def build_safe_model(model_name: str) -> Any:
    key = str(model_name).strip().lower()
    if key == "logistic_regression" and LogisticRegression is not None:
        return LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=42)
    if key == "random_forest" and RandomForestClassifier is not None:
        base = RandomForestClassifier(
            n_estimators=120,
            max_depth=6,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        if CalibratedClassifierCV is not None:
            try:
                return CalibratedClassifierCV(base, method="sigmoid", cv=3)
            except Exception:
                return base
        return base
    if key == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
    if key == "catboost" and CatBoostClassifier is not None:
        return CatBoostClassifier(
            iterations=120,
            depth=4,
            learning_rate=0.05,
            random_seed=42,
            verbose=False,
            loss_function="Logloss",
        )
    if key == "ensemble":
        enabled = [name for name in ("logistic_regression", "random_forest", "xgboost", "catboost") if build_safe_model(name) is not None]
        return WeightedEnsembleClassifier(enabled_models=enabled, random_state=42, n_estimators=120)
    return None


def model_names_to_train() -> List[str]:
    names = [name for name in ("logistic_regression", "random_forest", "xgboost", "catboost", "ensemble") if build_safe_model(name) is not None]
    return names


def predict_proba_positive(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        return np.asarray([float(row[1]) if len(row) > 1 else float(row[0]) for row in probs], dtype=np.float32)
    preds = model.predict(X)
    return np.asarray([float(v) for v in preds], dtype=np.float32)


def confusion_from_threshold(y_true: Sequence[int], y_prob: Sequence[float], threshold: float) -> Dict[str, int]:
    tp = tn = fp = fn = 0
    for truth, prob in zip(y_true, y_prob):
        pred = 1 if float(prob) >= float(threshold) else 0
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 0:
            tn += 1
        elif truth == 0 and pred == 1:
            fp += 1
        else:
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def pr_auc_score_safe(y_true: Sequence[int], y_prob: Sequence[float]) -> Optional[float]:
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return None


def calibration_bins(y_true: Sequence[int], y_prob: Sequence[float], *, bins: int = 10) -> List[Dict[str, Any]]:
    frame = pd.DataFrame({"y": list(y_true), "p": list(y_prob)})
    frame["bin"] = pd.cut(frame["p"], bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True)
    rows: List[Dict[str, Any]] = []
    for bucket, group in frame.groupby("bin", observed=False):
        if group.empty:
            continue
        rows.append(
            {
                "bucket": str(bucket),
                "count": int(len(group)),
                "avg_predicted_prob": float(group["p"].mean()),
                "actual_positive_rate": float(group["y"].mean()),
            }
        )
    return rows


def threshold_sweep(y_true: Sequence[int], y_prob: Sequence[float]) -> List[Dict[str, Any]]:
    report: List[Dict[str, Any]] = []
    for threshold in THRESHOLD_GRID:
        metrics = classification_metrics(y_true, y_prob, threshold=threshold)
        confusion = confusion_from_threshold(y_true, y_prob, threshold)
        signals = int(sum(1 for p in y_prob if float(p) >= threshold))
        win_rate_proxy = float(confusion["tp"] / signals) if signals else 0.0
        avg_signal_prob = float(np.mean([p for p in y_prob if float(p) >= threshold])) if signals else 0.0
        report.append(
            {
                "threshold": float(threshold),
                "signals": signals,
                "win_rate_proxy": win_rate_proxy,
                "avg_signal_probability": avg_signal_prob,
                "false_positives": int(confusion["fp"]),
                **metrics,
                **confusion,
            }
        )
    return report


def walk_forward_report(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    df = pd.DataFrame(X)
    df.index = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=len(df), freq="min", tz=TIMEZONE))
    splits = generate_purged_embargoed_cv_splits(df, label_horizon_bars=HORIZON_BARS, n_splits=5)
    return {
        "purged_embargo_splits_available": bool(splits),
        "purged_embargo_split_count": int(len(splits)),
    }


def train_and_evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: Sequence[datetime],
    feature_names: Sequence[str],
    artifact_dir: Path,
    *,
    max_models: int = 0,
) -> Dict[str, Any]:
    split = chronological_split(X, y, timestamps)
    train_slice = split["slices"]["train"]
    val_slice = split["slices"]["validation"]
    test_slice = split["slices"]["test"]

    X_train, X_val, X_test = X[train_slice], X[val_slice], X[test_slice]
    y_train, y_val, y_test = y[train_slice], y[val_slice], y[test_slice]
    X_train_s, X_val_s, X_test_s, scaler_mean, scaler_std = scale_train_val_test(X_train, X_val, X_test)

    model_reports: Dict[str, Any] = {}
    names = model_names_to_train()
    if max_models > 0:
        names = names[:max_models]

    for idx, model_name in enumerate(names, start=1):
        LOG.info("Training model [%s/%s]: %s", idx, len(names), model_name)
        model = build_safe_model(model_name)
        if model is None:
            continue
        model.fit(X_train_s, y_train)
        val_prob = predict_proba_positive(model, X_val_s)
        best = optimize_threshold_for_economic_edge(y_val, val_prob, min_trades=20)
        chosen_threshold = float(best.get("threshold", 0.5))
        test_prob = predict_proba_positive(model, X_test_s)
        metrics = classification_metrics(y_test, test_prob, threshold=chosen_threshold)
        pr_auc = pr_auc_score_safe(y_test, test_prob)
        if pr_auc is not None:
            metrics["pr_auc"] = pr_auc
        confusion = confusion_from_threshold(y_test, test_prob, chosen_threshold)
        sweep = threshold_sweep(y_test, test_prob)
        signal_count = int(sum(1 for p in test_prob if float(p) >= chosen_threshold))
        positive_probs = [float(p) for p in test_prob if float(p) >= chosen_threshold]
        false_positive_behavior = {
            "signals": signal_count,
            "false_positives": int(confusion["fp"]),
            "false_positive_rate_among_signals": float(confusion["fp"] / signal_count) if signal_count else 0.0,
        }
        confidence_distribution = {
            "p10": float(np.quantile(test_prob, 0.10)),
            "p50": float(np.quantile(test_prob, 0.50)),
            "p90": float(np.quantile(test_prob, 0.90)),
            "mean": float(np.mean(test_prob)),
        }
        report = {
            "model_name": model_name,
            "threshold_selected_on_validation": chosen_threshold,
            "validation_threshold_pick": best,
            "metrics": metrics,
            "confusion_matrix": confusion,
            "calibration_bins": calibration_bins(y_test, test_prob),
            "threshold_sweep": sweep,
            "trading_validation": {
                "number_of_signals": signal_count,
                "win_rate_proxy": float(confusion["tp"] / signal_count) if signal_count else 0.0,
                "average_predicted_probability": float(np.mean(positive_probs)) if positive_probs else 0.0,
                "false_positive_behavior": false_positive_behavior,
                "model_confidence_distribution": confidence_distribution,
            },
        }
        bundle = MLModelBundle(
            model=model,
            feature_names=list(feature_names),
            metrics={
                **metrics,
                "threshold": chosen_threshold,
                "label_policy": LABEL_POLICY,
                "train_samples": int(len(X_train)),
                "validation_samples": int(len(X_val)),
                "test_samples": int(len(X_test)),
            },
            trained_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
        )
        model_path = artifact_dir / f"{model_name}.pkl"
        import joblib

        joblib.dump(bundle, model_path)
        report["artifact"] = str(model_path)
        model_reports[model_name] = report
        write_json(artifact_dir / f"{model_name}_metrics.json", report)

    return {"split_report": split["ranges"], "models": model_reports}


def promotion_checklist(payload: Dict[str, Any]) -> Dict[str, Any]:
    models = payload.get("training", {}).get("models", {})
    best_roc = max((float(m.get("metrics", {}).get("roc_auc", 0.0)) for m in models.values()), default=0.0)
    feature_ok = payload.get("feature_compatibility", {}).get("matches_production", False)
    leakage_ok = payload.get("leakage_audit", {}).get("passed", False)
    data_ok = payload.get("data_quality", {}).get("session_missing_bar_estimate", 0) == 0
    return {
        "data_quality_passed": bool(data_ok),
        "no_leakage_detected": bool(leakage_ok),
        "feature_compatibility_passed": bool(feature_ok),
        "test_roc_auc_acceptable": bool(best_roc >= 0.55),
        "threshold_behavior_acceptable": bool(models),
        "paper_live_gate_passed": False,
    }


def main() -> int:
    args = parse_args()
    started = datetime.now()
    artifact_dir = MODELS_DIR / f"retrained_1y_nifty_{started.strftime('%Y%m%d_%H%M%S')}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading local NIFTY data sources")
    minute_frame, discovery = prepare_minute_frame()
    minute_frame, base_quality = clean_frame(minute_frame)
    write_json(artifact_dir / "data_discovery.json", discovery)
    write_json(artifact_dir / "base_data_quality.json", base_quality)

    LOG.info("Selecting most recent continuous 1-year window")
    selected_frame, window_report = select_one_year_window(minute_frame, allow_backfill=not args.skip_broker_backfill)
    selected_frame, selected_quality = clean_frame(selected_frame)
    write_json(artifact_dir / "selected_window_report.json", window_report)
    write_json(artifact_dir / "data_quality_report.json", selected_quality)

    LOG.info("Building leakage-safe training dataset")
    candles = frame_to_candles(selected_frame)
    leakage_candles = candles[-LEAKAGE_AUDIT_MAX_CANDLES:] if len(candles) > LEAKAGE_AUDIT_MAX_CANDLES else candles
    leakage = verify_no_lookahead_leakage(leakage_candles, lookback=LOOKBACK_BARS, horizon=HORIZON_BARS)
    leakage["audit_candles_used"] = int(len(leakage_candles))
    write_json(artifact_dir / "leakage_audit_report.json", leakage)
    if not leakage.get("passed", False):
        raise RuntimeError(json.dumps({"leakage_audit_failed": leakage}, indent=2))

    dataset = build_dataset_payload(candles, label_policy=args.label_policy)
    feature_compat = compare_feature_compatibility(dataset["feature_names"])
    label_report = build_label_report(dataset["y"])
    manifest = feature_manifest(dataset["X"], dataset["feature_names"])
    walk_forward = walk_forward_report(dataset["X"], dataset["y"])

    write_json(artifact_dir / "feature_manifest.json", manifest)
    write_json(artifact_dir / "feature_compatibility.json", feature_compat)
    write_json(artifact_dir / "label_report.json", label_report)
    write_json(artifact_dir / "walk_forward_report.json", walk_forward)
    write_json(
        artifact_dir / "training_config.json",
        {
            "label_policy": args.label_policy,
            "lookback_bars": LOOKBACK_BARS,
            "horizon_bars": HORIZON_BARS,
            "timezone": TIMEZONE,
            "session_start": SESSION_START.isoformat(),
            "session_end": SESSION_END.isoformat(),
        },
    )

    final_report: Dict[str, Any] = {
        "run_started_at": started.isoformat(),
        "artifact_dir": str(artifact_dir),
        "data_discovery": discovery,
        "selected_window": window_report,
        "data_quality": selected_quality,
        "leakage_audit": leakage,
        "feature_compatibility": feature_compat,
        "feature_manifest_summary": {"feature_count": manifest["feature_count"]},
        "label_report": label_report,
        "walk_forward": walk_forward,
    }

    if args.skip_training:
        final_report["training"] = {"skipped": True}
        final_report["promotion_checklist"] = promotion_checklist(final_report)
        write_json(artifact_dir / "validation_metrics_report.json", final_report)
        LOG.info("Diagnostics completed. Outputs saved to %s", artifact_dir)
        return 0

    LOG.info("Training models on chronological train/validation/test split")
    training = train_and_evaluate_models(
        dataset["X"],
        dataset["y"],
        dataset["timestamps"],
        dataset["feature_names"],
        artifact_dir,
        max_models=int(args.max_models or 0),
    )
    final_report["training"] = training
    final_report["promotion_checklist"] = promotion_checklist(final_report)
    write_json(artifact_dir / "validation_metrics_report.json", final_report)

    LOG.info("Saved retraining artifacts to %s", artifact_dir)
    best_model = None
    best_auc = -1.0
    for name, payload in training.get("models", {}).items():
        roc_auc = float(payload.get("metrics", {}).get("roc_auc", 0.0))
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model = name
    LOG.info("Best test ROC-AUC: %.4f (%s)", best_auc, best_model or "n/a")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOG.error("Retraining pipeline failed: %s", exc)
        raise
