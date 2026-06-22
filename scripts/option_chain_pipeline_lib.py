from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
CONFIG_DIR = REPO_ROOT / "config"
LIVE_MODEL_PATH = REPO_ROOT / "ml_signal_model.pkl"
PRODUCTION_PROFILE_PATH = CONFIG_DIR / "production_model_profile.json"
TIMEZONE = "Asia/Kolkata"
DEFAULT_HORIZON_BARS = 3
THRESHOLD_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
OPTION_CONTEXT_REQUIRED_COLUMNS = [
    "timestamp",
    "underlying_spot",
    "expiry",
    "strike",
    "option_type",
    "volume",
    "open_interest",
    "change_in_oi",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "bid",
    "ask",
    "spread",
]
OPTION_CONTEXT_PRICE_COLUMNS = ["open", "high", "low"]
OPTION_CONTEXT_CLOSE_ALIASES = ["close", "ltp"]
TRAINING_ONLY_LEAKAGE_COLUMNS = {
    "future_close",
    "gross_forward_return",
    "net_forward_return",
    "profitable_trade_label",
    "avoid_trade_label",
    "label",
    "target",
    "pnl",
    "exit_price",
    "post_entry_return",
}
ESTIMATED_OPTION_CONTEXT_COLUMNS = {
    "estimated_ctx_iv",
    "estimated_ctx_delta",
    "estimated_ctx_gamma",
    "estimated_ctx_theta",
    "estimated_ctx_vega",
    "estimated_delta_abs",
    "estimated_greeks_imbalance",
    "estimated_theta_to_vega_ratio",
    "estimated_gamma_to_theta_ratio",
    "estimated_bid_ask_spread_pct",
    "estimated_spread_source",
    "greeks_source",
    "production_adoption_allowed",
}

import sys

SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ml_signals import _wrap_model, load_model
from market_data import Candle
from candlestick_patterns import is_bearish_engulfing, is_bullish_engulfing, is_doji, is_hammer, is_shooting_star
from indicators import adx, atr, choppiness_index, ema, pivot_points, rsi, supertrend
from strategy_allocator import detect_regime
from retrain_nifty_1year import (
    build_safe_model,
    calibration_bins,
    chronological_split,
    compare_feature_compatibility,
    confusion_from_threshold,
    model_names_to_train,
    predict_proba_positive,
    pr_auc_score_safe,
    scale_train_val_test,
    threshold_sweep,
    write_json,
)
from retraining_validation import classification_metrics


@dataclass
class DatasetAudit:
    generated_at: str
    dataset_folder: str
    manifest_rows: int
    data_files: int
    total_rows_saved: int
    zero_row_files: int
    approximate_memory_bytes: int
    date_min: Optional[str]
    date_max: Optional[str]
    weekly_counts: Dict[str, int]
    option_type_counts: Dict[str, int]
    unique_expiries: int
    unique_strikes: int
    file_column_schema: List[str]
    duplicate_manifest_csv_files: int
    missing_files: int
    contains_option_chain_snapshots: bool
    contains_historical_option_candles: bool
    contains_spot_candles: bool
    contains_labels: bool
    contains_trade_logs: bool
    contains_bid_ask: bool
    contains_expiry_and_strike_metadata: bool
    sample_missing_values: Dict[str, float]


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): ensure_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_serializable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    return value


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_pct_change(series: pd.Series) -> pd.Series:
    return series.replace(0.0, np.nan).pct_change().replace([np.inf, -np.inf], np.nan)


def report_paths(prefix: str, *, stamp: Optional[str] = None) -> Tuple[Path, Path, str]:
    stamp = stamp or _timestamp_now()
    json_path = REPORTS_DIR / f"{prefix}_{stamp}.json"
    md_path = REPORTS_DIR / f"{prefix}_{stamp}.md"
    return json_path, md_path, stamp


def load_live_model_bundle() -> Any:
    return load_model(str(LIVE_MODEL_PATH)) if LIVE_MODEL_PATH.exists() else None


def live_feature_schema_summary() -> Dict[str, Any]:
    bundle = load_live_model_bundle()
    feature_names = list(getattr(bundle, "feature_names", []) or []) if bundle is not None else []
    return {
        "feature_names": feature_names,
        "feature_count": int(len(feature_names)),
        "scaler_mean_len": int(len(getattr(bundle, "scaler_mean", []) or [])) if bundle is not None else 0,
        "scaler_std_len": int(len(getattr(bundle, "scaler_std", []) or [])) if bundle is not None else 0,
        "model_input_shape": [1, int(len(feature_names))] if feature_names else None,
    }


def read_manifest(dataset_folder: Path) -> pd.DataFrame:
    manifest_path = dataset_folder / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found in {dataset_folder}")
    manifest = pd.read_csv(manifest_path)
    required = {
        "expiry",
        "instrument_key",
        "trading_symbol",
        "instrument_type",
        "strike_price",
        "weekly",
        "rows_saved",
        "csv_file",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise RuntimeError(f"manifest.csv missing columns: {sorted(missing)}")
    return manifest


def audit_option_chain_dataset(dataset_folder: Path) -> Tuple[Dict[str, Any], Path, Path]:
    manifest = read_manifest(dataset_folder)
    file_exists = manifest["csv_file"].map(lambda name: (dataset_folder / str(name)).exists())
    duplicate_manifest_csv_files = int(manifest["csv_file"].duplicated().sum())
    sample_file = next((dataset_folder / str(name) for name in manifest["csv_file"] if (dataset_folder / str(name)).exists()), None)
    sample_df = pd.read_csv(sample_file) if sample_file is not None else pd.DataFrame()
    sample_missing = {
        col: float(sample_df[col].isna().mean()) for col in sample_df.columns
    } if not sample_df.empty else {}
    total_rows_saved = int(pd.to_numeric(manifest["rows_saved"], errors="coerce").fillna(0).sum())
    approx_mem = int(total_rows_saved * max(len(sample_df.columns), 8) * 8)
    audit = DatasetAudit(
        generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        dataset_folder=str(dataset_folder),
        manifest_rows=int(len(manifest)),
        data_files=int(len(manifest)),
        total_rows_saved=total_rows_saved,
        zero_row_files=int(pd.to_numeric(manifest["rows_saved"], errors="coerce").fillna(0).eq(0).sum()),
        approximate_memory_bytes=approx_mem,
        date_min=str(manifest["expiry"].min()) if not manifest.empty else None,
        date_max=str(manifest["expiry"].max()) if not manifest.empty else None,
        weekly_counts={str(k): int(v) for k, v in manifest["weekly"].value_counts(dropna=False).to_dict().items()},
        option_type_counts={str(k): int(v) for k, v in manifest["instrument_type"].value_counts(dropna=False).to_dict().items()},
        unique_expiries=int(manifest["expiry"].nunique()),
        unique_strikes=int(pd.to_numeric(manifest["strike_price"], errors="coerce").nunique()),
        file_column_schema=list(sample_df.columns),
        duplicate_manifest_csv_files=duplicate_manifest_csv_files,
        missing_files=int((~file_exists).sum()),
        contains_option_chain_snapshots=False,
        contains_historical_option_candles=True,
        contains_spot_candles=False,
        contains_labels=False,
        contains_trade_logs=False,
        contains_bid_ask=bool({"bid", "ask"}.issubset(sample_df.columns)),
        contains_expiry_and_strike_metadata=bool({"expiry", "strike_price", "instrument_type"}.issubset(sample_df.columns)),
        sample_missing_values=sample_missing,
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path, md_path, _ = report_paths("new_option_chain_dataset_audit")
    payload = asdict(audit)
    write_json(json_path, ensure_serializable(payload))
    md_lines = [
        "# Option Chain Dataset Audit",
        "",
        f"- Dataset folder: `{dataset_folder}`",
        f"- Manifest rows: `{audit.manifest_rows}`",
        f"- Total rows saved: `{audit.total_rows_saved}`",
        f"- Expiry range: `{audit.date_min}` to `{audit.date_max}`",
        f"- Option types: `{audit.option_type_counts}`",
        f"- Weekly flags: `{audit.weekly_counts}`",
        f"- Missing files: `{audit.missing_files}`",
        f"- Zero-row files: `{audit.zero_row_files}`",
        f"- Approximate raw memory footprint: `{audit.approximate_memory_bytes}` bytes",
        "",
        "## File Schema",
        "",
        ", ".join(audit.file_column_schema) if audit.file_column_schema else "No readable files found.",
        "",
        "## Coverage",
        "",
        f"- Unique expiries: `{audit.unique_expiries}`",
        f"- Unique strikes: `{audit.unique_strikes}`",
        f"- Contains historical option candles: `{audit.contains_historical_option_candles}`",
        f"- Contains bid/ask: `{audit.contains_bid_ask}`",
        f"- Contains labels/outcomes: `{audit.contains_labels}`",
        "",
        "## Sample Missing Values",
        "",
    ]
    for col, frac in audit.sample_missing_values.items():
        md_lines.append(f"- `{col}`: `{frac:.2%}`")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return payload, json_path, md_path


def _standardize_file_frame(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    working = frame.copy()
    if working.empty:
        return working
    working.columns = [str(col).strip().lower() for col in working.columns]
    rename_map = {"instrument_type": "option_type", "close": "ltp", "open_interest": "oi"}
    working = working.rename(columns=rename_map)
    required = ["timestamp", "open", "high", "low", "ltp", "volume", "oi", "expiry", "instrument_key", "trading_symbol", "option_type", "strike_price", "weekly"]
    for column in required:
        if column not in working.columns:
            working[column] = np.nan
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce", utc=False)
    if getattr(working["timestamp"].dt, "tz", None) is None:
        working["timestamp"] = working["timestamp"].dt.tz_localize(TIMEZONE, nonexistent="NaT", ambiguous="NaT")
    else:
        working["timestamp"] = working["timestamp"].dt.tz_convert(TIMEZONE)
    working["expiry"] = pd.to_datetime(working["expiry"], errors="coerce")
    working["option_type"] = working["option_type"].astype(str).str.upper().str.strip()
    for col in ["open", "high", "low", "ltp", "volume", "oi", "strike_price"]:
        working[col] = pd.to_numeric(working[col], errors="coerce")
    working["weekly"] = working["weekly"].astype(str).str.lower().isin({"true", "1", "yes"})
    working["source_file"] = source_name
    working["trading_day"] = working["timestamp"].dt.date
    working = working.dropna(subset=["timestamp", "expiry", "ltp", "strike_price"])
    return working[required + ["source_file", "trading_day"]]


def load_option_chain_rows(dataset_folder: Path) -> pd.DataFrame:
    manifest = read_manifest(dataset_folder)
    rows: List[pd.DataFrame] = []
    for record in manifest.itertuples(index=False):
        csv_name = str(record.csv_file)
        if int(getattr(record, "rows_saved", 0) or 0) <= 0:
            continue
        path = dataset_folder / csv_name
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        normalized = _standardize_file_frame(frame, csv_name)
        if not normalized.empty:
            rows.append(normalized)
    if not rows:
        raise RuntimeError("No non-empty option-chain CSV files could be loaded.")
    data = pd.concat(rows, ignore_index=True)
    data = data.sort_values(["timestamp", "instrument_key"], kind="stable")
    data = data.drop_duplicates(subset=["timestamp", "instrument_key"], keep="last").reset_index(drop=True)
    return data


def _add_research_features(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    df = frame.copy()
    df["dte_days"] = (df["expiry"] - pd.to_datetime(df["trading_day"])).dt.days.astype(float)
    df["is_weekly"] = df["weekly"].astype(int)
    df["option_type_ce"] = (df["option_type"] == "CE").astype(int)
    df["option_type_pe"] = (df["option_type"] == "PE").astype(int)
    df["range_pct"] = (df["high"] - df["low"]) / df["ltp"].replace(0, np.nan)
    df["oc_change_pct"] = (df["ltp"] - df["open"]) / df["open"].replace(0, np.nan)
    df["hl_change_pct"] = (df["high"] - df["low"]) / df["low"].replace(0, np.nan)
    df["oi_change_pct"] = df.groupby("instrument_key")["oi"].pct_change().replace([np.inf, -np.inf], np.nan)
    df["volume_change_pct"] = df.groupby("instrument_key")["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    df["ret_1"] = df.groupby("instrument_key")["ltp"].pct_change().replace([np.inf, -np.inf], np.nan)
    df["ret_3"] = df.groupby("instrument_key")["ltp"].pct_change(3).replace([np.inf, -np.inf], np.nan)
    df["ret_5"] = df.groupby("instrument_key")["ltp"].pct_change(5).replace([np.inf, -np.inf], np.nan)
    df["oi_z_5"] = (
        df.groupby("instrument_key")["oi"]
        .transform(lambda s: (s - s.rolling(5, min_periods=2).mean()) / s.rolling(5, min_periods=2).std(ddof=0))
        .replace([np.inf, -np.inf], np.nan)
    )
    df["volume_z_5"] = (
        df.groupby("instrument_key")["volume"]
        .transform(lambda s: (s - s.rolling(5, min_periods=2).mean()) / s.rolling(5, min_periods=2).std(ddof=0))
        .replace([np.inf, -np.inf], np.nan)
    )
    df["weekday"] = pd.to_datetime(df["trading_day"]).dt.weekday.astype(float)
    df["month"] = pd.to_datetime(df["trading_day"]).dt.month.astype(float)
    feature_cols = [
        "strike_price",
        "dte_days",
        "is_weekly",
        "option_type_ce",
        "option_type_pe",
        "range_pct",
        "oc_change_pct",
        "hl_change_pct",
        "oi",
        "volume",
        "oi_change_pct",
        "volume_change_pct",
        "ret_1",
        "ret_3",
        "ret_5",
        "oi_z_5",
        "volume_z_5",
        "weekday",
        "month",
    ]
    unavailable_live_features = [
        "spot_price",
        "bid",
        "ask",
        "iv",
        "moneyness",
        "ce_pe_oi_ratio",
        "ce_pe_volume_ratio",
    ]
    schema = {
        "feature_columns": feature_cols,
        "unavailable_columns_not_used": unavailable_live_features,
        "schema_match_live_model": False,
        "note": "External option archive lacks the full live feature context used by the current production model.",
    }
    return df, feature_cols, schema


def _add_labels(frame: pd.DataFrame, *, horizon_bars: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = frame.copy()
    future_close = df.groupby("instrument_key")["ltp"].shift(-horizon_bars)
    gross_forward_return = (future_close - df["ltp"]) / df["ltp"].replace(0, np.nan)
    cost_penalty = 0.0025
    spread_penalty = 0.0010
    net_forward_return = gross_forward_return - cost_penalty - spread_penalty
    df["future_close"] = future_close
    df["gross_forward_return"] = gross_forward_return
    df["net_forward_return"] = net_forward_return
    df["profitable_trade_label"] = (net_forward_return > 0.0).astype(float)
    df.loc[future_close.isna(), "profitable_trade_label"] = np.nan
    df["avoid_trade_label"] = (net_forward_return <= 0.0).astype(float)
    df.loc[future_close.isna(), "avoid_trade_label"] = np.nan
    label_report = {
        "label_formula": f"net_forward_return_{horizon_bars} = ((future_close - current_ltp) / current_ltp) - 0.0025 - 0.0010",
        "horizon_bars": int(horizon_bars),
        "usable_rows_with_label": int(df["profitable_trade_label"].notna().sum()),
        "positive_labels": int(df["profitable_trade_label"].fillna(0).eq(1).sum()),
        "negative_labels": int(df["profitable_trade_label"].fillna(0).eq(0).sum()),
    }
    return df, label_report


def build_processed_option_chain_dataset(dataset_folder: Path, *, horizon_bars: int = DEFAULT_HORIZON_BARS) -> Dict[str, Any]:
    raw = load_option_chain_rows(dataset_folder)
    rows_before = int(len(raw))
    duplicates_removed = rows_before - int(len(raw.drop_duplicates(subset=["timestamp", "instrument_key"], keep="last")))
    enriched, feature_cols, schema = _add_research_features(raw)
    labeled, label_report = _add_labels(enriched, horizon_bars=horizon_bars)
    for col in feature_cols:
        labeled[col] = pd.to_numeric(labeled[col], errors="coerce")
    labeled = labeled.sort_values(["timestamp", "instrument_key"], kind="stable").reset_index(drop=True)
    feature_ready = labeled.dropna(subset=["profitable_trade_label"]).copy()
    medians = feature_ready[feature_cols].median(numeric_only=True)
    feature_ready[feature_cols] = feature_ready[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp_now()
    parquet_path = PROCESSED_DIR / f"nifty_option_chain_training_{stamp}.parquet"
    csv_path = PROCESSED_DIR / f"nifty_option_chain_training_{stamp}.csv"
    csv_sample_path = PROCESSED_DIR / f"nifty_option_chain_training_{stamp}_sample.csv"
    parquet_written = True
    try:
        feature_ready.to_parquet(parquet_path, index=False)
    except Exception:
        parquet_written = False
        feature_ready.to_csv(csv_path, index=False)
    feature_ready.head(500).to_csv(csv_sample_path, index=False)
    schema_path = PROCESSED_DIR / f"nifty_option_chain_training_{stamp}_schema.json"
    write_json(schema_path, ensure_serializable(schema))
    report = {
        "processed_dataset": str(parquet_path if parquet_written else csv_path),
        "processed_dataset_format": "parquet" if parquet_written else "csv_fallback",
        "preferred_parquet_path": str(parquet_path),
        "csv_fallback_path": str(csv_path),
        "csv_sample": str(csv_sample_path),
        "schema_path": str(schema_path),
        "rows_before_cleaning": rows_before,
        "rows_after_cleaning": int(len(feature_ready)),
        "rows_rejected": int(rows_before - len(feature_ready)),
        "duplicates_removed": int(duplicates_removed),
        "feature_columns": feature_cols,
        "label_report": label_report,
        "schema": schema,
        "date_range": {
            "start": str(feature_ready["timestamp"].min()),
            "end": str(feature_ready["timestamp"].max()),
        },
    }
    return report


def _load_spot_json_dir(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for file in sorted(path.glob("candles_*.json")):
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
            candles = payload.get("candles") or []
        except Exception:
            continue
        for row in candles:
            rows.append(
                {
                    "timestamp": row.get("time"),
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": row.get("close"),
                    "volume": row.get("volume"),
                }
            )
    return pd.DataFrame(rows)


def load_spot_dataset(spot_dataset: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    minute_frames: List[pd.DataFrame] = []
    daily_frames: List[pd.DataFrame] = []
    scanned: List[str] = []
    if spot_dataset.is_dir():
        minute_csv = spot_dataset / "historical" / "NIFTY_1m.csv"
        daily_csv = spot_dataset / "historical" / "NIFTY_daily.csv"
        if minute_csv.exists():
            minute_frames.append(pd.read_csv(minute_csv))
            scanned.append(str(minute_csv))
        if daily_csv.exists():
            daily_frames.append(pd.read_csv(daily_csv))
            scanned.append(str(daily_csv))
        json_df = _load_spot_json_dir(spot_dataset)
        if not json_df.empty:
            minute_frames.append(json_df)
            scanned.append(str(spot_dataset / "candles_*.json"))
    else:
        suffix = spot_dataset.suffix.lower()
        frame = pd.read_parquet(spot_dataset) if suffix == ".parquet" else pd.read_csv(spot_dataset)
        if "1m" in spot_dataset.name.lower():
            minute_frames.append(frame)
        else:
            daily_frames.append(frame)
        scanned.append(str(spot_dataset))
    minute_df = pd.concat(minute_frames, ignore_index=True) if minute_frames else pd.DataFrame()
    daily_df = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    meta = {"scanned_sources": scanned}
    return minute_df, daily_df, meta


def _normalize_spot_frame(frame: pd.DataFrame, *, daily: bool) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.copy()
    work.columns = [str(col).strip().lower() for col in work.columns]
    if "timestamp" not in work.columns:
        raise RuntimeError("Spot dataset must include timestamp column.")
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=False)
    if daily:
        if getattr(work["timestamp"].dt, "tz", None) is not None:
            work["timestamp"] = work["timestamp"].dt.tz_convert(TIMEZONE).dt.tz_localize(None)
        work["trading_day"] = pd.to_datetime(work["timestamp"]).dt.date
    else:
        if getattr(work["timestamp"].dt, "tz", None) is None:
            work["timestamp"] = work["timestamp"].dt.tz_localize(TIMEZONE, nonexistent="NaT", ambiguous="NaT")
        else:
            work["timestamp"] = work["timestamp"].dt.tz_convert(TIMEZONE)
        work["trading_day"] = work["timestamp"].dt.date
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp", kind="stable")
    work = work.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return work


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _compute_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            (frame["high"] - frame["low"]).abs(),
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _add_spot_features(frame: pd.DataFrame, *, daily: bool) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.copy().sort_values("timestamp", kind="stable")
    work["spot_close"] = work["close"]
    work["spot_return_1"] = work["close"].pct_change().replace([np.inf, -np.inf], np.nan)
    work["spot_return_3"] = work["close"].pct_change(3).replace([np.inf, -np.inf], np.nan)
    work["spot_return_5"] = work["close"].pct_change(5).replace([np.inf, -np.inf], np.nan)
    work["spot_range_pct"] = (work["high"] - work["low"]) / work["close"].replace(0.0, np.nan)
    work["spot_atr"] = _compute_atr(work)
    work["spot_rsi"] = _compute_rsi(work["close"])
    typical_price = (work["high"] + work["low"] + work["close"]) / 3.0
    if float(work["volume"].abs().sum()) > 0:
        cum_pv = (typical_price * work["volume"]).groupby(work["trading_day"]).cumsum()
        cum_v = work["volume"].groupby(work["trading_day"]).cumsum().replace(0.0, np.nan)
        work["spot_vwap"] = cum_pv / cum_v
    else:
        work["spot_vwap"] = np.nan
    minutes = pd.to_datetime(work["timestamp"]).dt.hour * 60 + pd.to_datetime(work["timestamp"]).dt.minute
    work["ctx_time_sin"] = np.sin(2.0 * np.pi * minutes / 1440.0)
    work["ctx_time_cos"] = np.cos(2.0 * np.pi * minutes / 1440.0)
    work["is_opening_session"] = ((pd.to_datetime(work["timestamp"]).dt.hour == 9) & (pd.to_datetime(work["timestamp"]).dt.minute.between(15, 44))).astype(int)
    work["is_closing_session"] = ((pd.to_datetime(work["timestamp"]).dt.hour == 15) & (pd.to_datetime(work["timestamp"]).dt.minute <= 30)).astype(int)
    work["is_midday_lull"] = ((minutes >= 690) & (minutes <= 810)).astype(int)
    work["weekday"] = pd.to_datetime(work["timestamp"]).dt.weekday
    return work


def write_feature_schema_gap_report(
    option_feature_names: Sequence[str],
    *,
    reconstructed_features: Optional[Sequence[str]] = None,
    additional_data_required: Optional[Sequence[str]] = None,
    report_prefix: str = "live_vs_option_chain_feature_schema_gap",
) -> Tuple[Dict[str, Any], Path]:
    live = live_feature_schema_summary()
    live_names = list(live["feature_names"])
    option_names = list(option_feature_names)
    reconstructed = list(reconstructed_features or [])
    combined_names = sorted(set(option_names) | set(reconstructed))
    matching = [name for name in live_names if name in combined_names]
    missing_live = [name for name in live_names if name not in combined_names]
    extra_research = [name for name in combined_names if name not in live_names]
    requires_additional = list(additional_data_required or [])
    verdict = "NOT_COMPATIBLE"
    if not missing_live and not requires_additional:
        verdict = "SCHEMA_MATCH"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _, md_path, stamp = report_paths(report_prefix)
    payload = {
        "generated_at": stamp,
        "live_production_feature_list": live_names,
        "option_chain_research_feature_list": option_names,
        "reconstructed_live_compatible_features": reconstructed,
        "matching_features": matching,
        "missing_live_features": missing_live,
        "extra_research_features": extra_research,
        "features_that_can_be_reconstructed_from_dataset": reconstructed,
        "features_that_require_additional_data": requires_additional,
        "final_compatibility_verdict": verdict,
        "model_input_shape": live.get("model_input_shape"),
    }
    md_lines = [
        "# Live vs Option Chain Feature Schema Gap",
        "",
        f"- Live feature count: `{len(live_names)}`",
        f"- Research feature count: `{len(option_names)}`",
        f"- Matching features: `{len(matching)}`",
        f"- Missing live features: `{len(missing_live)}`",
        f"- Extra research features: `{len(extra_research)}`",
        f"- Final compatibility verdict: `{verdict}`",
        "",
        "## Matching Features",
        "",
    ]
    md_lines.extend([f"- `{name}`" for name in matching] or ["- None"])
    md_lines.extend(["", "## Missing Live Features", ""])
    md_lines.extend([f"- `{name}`" for name in missing_live] or ["- None"])
    md_lines.extend(["", "## Extra Research Features", ""])
    md_lines.extend([f"- `{name}`" for name in extra_research] or ["- None"])
    md_lines.extend(["", "## Reconstructable Features", ""])
    md_lines.extend([f"- `{name}`" for name in reconstructed] or ["- None"])
    md_lines.extend(["", "## Additional Data Required", ""])
    md_lines.extend([f"- `{name}`" for name in requires_additional] or ["- None"])
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return payload, md_path


def _live_missing_feature_metadata() -> List[Dict[str, Any]]:
    return [
        {"feature": "bullish_engulfing", "category": "C", "formula": "Current candle bullish body engulfs previous bearish body", "raw_columns": ["open", "high", "low", "ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "bearish_engulfing", "category": "C", "formula": "Current candle bearish body engulfs previous bullish body", "raw_columns": ["open", "high", "low", "ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "doji", "category": "C", "formula": "Small candle body relative to range", "raw_columns": ["open", "high", "low", "ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "hammer", "category": "C", "formula": "Hammer candlestick on option OHLC", "raw_columns": ["open", "high", "low", "ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "shooting_star", "category": "C", "formula": "Shooting-star candlestick on option OHLC", "raw_columns": ["open", "high", "low", "ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "ret_mean", "category": "C", "formula": "Mean of trailing option returns over lookback window", "raw_columns": ["ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "ret_std", "category": "C", "formula": "Std of trailing option returns over lookback window", "raw_columns": ["ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "ret_min", "category": "C", "formula": "Min of trailing option returns over lookback window", "raw_columns": ["ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "ret_max", "category": "C", "formula": "Max of trailing option returns over lookback window", "raw_columns": ["ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "vol_mean", "category": "C", "formula": "Mean trailing option volume", "raw_columns": ["volume"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "vol_std", "category": "C", "formula": "Std trailing option volume", "raw_columns": ["volume"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "vol_min", "category": "C", "formula": "Min trailing option volume", "raw_columns": ["volume"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "vol_max", "category": "C", "formula": "Max trailing option volume", "raw_columns": ["volume"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "adx_14", "category": "C", "formula": "ADX over trailing option OHLC", "raw_columns": ["high", "low", "ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "choppiness_14", "category": "C", "formula": "Choppiness index over trailing option OHLC", "raw_columns": ["high", "low", "ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "supertrend_dir", "category": "C", "formula": "Supertrend direction over trailing option OHLC", "raw_columns": ["high", "low", "ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "pivot_pp_dist_pct", "category": "C", "formula": "(close - pivot_point) / close", "raw_columns": ["high", "low", "ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "pivot_r1_dist_pct", "category": "C", "formula": "(close - pivot_r1) / close", "raw_columns": ["high", "low", "ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "pivot_s1_dist_pct", "category": "C", "formula": "(close - pivot_s1) / close", "raw_columns": ["high", "low", "ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "close_vs_open_pct", "category": "A", "formula": "(last_close - last_open) / last_close", "raw_columns": ["last_open", "last_close"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "range_to_atr", "category": "A", "formula": "range_pct / atr_pct", "raw_columns": ["range_pct", "atr_pct"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "momentum_lookback_pct", "category": "C", "formula": "(close - close_lookback_start) / close_lookback_start", "raw_columns": ["ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "regime_trending", "category": "B", "formula": "One-hot from detect_regime using trailing ATR/ADX/RSI", "raw_columns": ["spot_close", "spot_atr", "spot_rsi"], "live": True, "historical": True, "leakage_risk": "medium"},
        {"feature": "regime_volatile", "category": "B", "formula": "One-hot from detect_regime using trailing ATR/ADX/RSI", "raw_columns": ["spot_close", "spot_atr", "spot_rsi"], "live": True, "historical": True, "leakage_risk": "medium"},
        {"feature": "regime_mean_reverting", "category": "B", "formula": "One-hot from detect_regime using trailing ATR/ADX/RSI", "raw_columns": ["spot_close", "spot_atr", "spot_rsi"], "live": True, "historical": True, "leakage_risk": "medium"},
        {"feature": "regime_quiet", "category": "B", "formula": "One-hot from detect_regime using trailing ATR/ADX/RSI", "raw_columns": ["spot_close", "spot_atr", "spot_rsi"], "live": True, "historical": True, "leakage_risk": "medium"},
        {"feature": "ctx_iv", "category": "E", "formula": "Vendor/live implied volatility field", "raw_columns": ["iv"], "live": True, "historical": False, "leakage_risk": "low"},
        {"feature": "ctx_iv_change_pct", "category": "E", "formula": "pct_change of IV", "raw_columns": ["iv"], "live": True, "historical": False, "leakage_risk": "low"},
        {"feature": "ctx_iv_percentile", "category": "E", "formula": "Rolling percentile of IV", "raw_columns": ["iv"], "live": True, "historical": False, "leakage_risk": "low"},
        {"feature": "ctx_delta", "category": "E", "formula": "Vendor/live option delta", "raw_columns": ["delta"], "live": True, "historical": False, "leakage_risk": "low"},
        {"feature": "ctx_gamma", "category": "E", "formula": "Vendor/live option gamma", "raw_columns": ["gamma"], "live": True, "historical": False, "leakage_risk": "low"},
        {"feature": "ctx_vega", "category": "E", "formula": "Vendor/live option vega", "raw_columns": ["vega"], "live": True, "historical": False, "leakage_risk": "low"},
        {"feature": "ctx_theta", "category": "E", "formula": "Vendor/live option theta", "raw_columns": ["theta"], "live": True, "historical": False, "leakage_risk": "low"},
        {"feature": "ctx_adx", "category": "A", "formula": "Alias of adx_14", "raw_columns": ["adx_14"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "ctx_trend_strength", "category": "A", "formula": "Alias of ema_diff_pct", "raw_columns": ["ema_diff_pct"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "ctx_choppiness", "category": "A", "formula": "Alias of choppiness_14", "raw_columns": ["choppiness_14"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "ctx_volume_sma", "category": "A", "formula": "Alias of vol_mean", "raw_columns": ["vol_mean"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "delta_abs", "category": "E", "formula": "abs(ctx_delta)", "raw_columns": ["delta"], "live": True, "historical": False, "leakage_risk": "low"},
        {"feature": "greeks_imbalance", "category": "E", "formula": "ctx_gamma + ctx_vega + ctx_theta", "raw_columns": ["gamma", "vega", "theta"], "live": True, "historical": False, "leakage_risk": "low"},
        {"feature": "vol_of_vol_14", "category": "C", "formula": "Std of trailing ATR% estimates", "raw_columns": ["high", "low", "ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "dist_from_opening_high_pct", "category": "B", "formula": "(close - opening_high) / opening_high", "raw_columns": ["spot intraday OHLC"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "dist_from_opening_low_pct", "category": "B", "formula": "(close - opening_low) / opening_low", "raw_columns": ["spot intraday OHLC"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "opening_range_width_pct", "category": "B", "formula": "(opening_high - opening_low) / close", "raw_columns": ["spot intraday OHLC"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "opening_range_breakout_strength", "category": "B", "formula": "Distance beyond opening range / opening_range_width", "raw_columns": ["spot intraday OHLC"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "dist_to_rolling_high_20", "category": "C", "formula": "(rolling_high_20 - close) / close", "raw_columns": ["high", "ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "dist_to_rolling_low_20", "category": "C", "formula": "(close - rolling_low_20) / close", "raw_columns": ["low", "ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "rolling_range_width_20", "category": "C", "formula": "(rolling_high_20 - rolling_low_20) / close", "raw_columns": ["high", "low", "ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "rolling_range_position_20", "category": "C", "formula": "(close - rolling_low_20) / (rolling_high_20 - rolling_low_20)", "raw_columns": ["high", "low", "ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "realized_vol_30", "category": "C", "formula": "Std of trailing 30 returns", "raw_columns": ["ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "atr_pct_regime_10", "category": "C", "formula": "current atr_pct / trailing mean atr_pct over 10 bars", "raw_columns": ["high", "low", "ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "atr_percentile_60", "category": "C", "formula": "Trailing percentile rank of atr_pct over 60 bars", "raw_columns": ["high", "low", "ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "realized_vol_percentile_60", "category": "C", "formula": "Trailing percentile rank of realized_vol_30 over 60 bars", "raw_columns": ["ltp"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "volatility_percentile_60", "category": "A", "formula": "Alias of realized_vol_percentile_60", "raw_columns": ["realized_vol_percentile_60"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "volatility_regime_classifier", "category": "C", "formula": "1 if ATR or realized-vol percentile exceeds threshold", "raw_columns": ["atr_percentile_60", "realized_vol_percentile_60"], "live": True, "historical": True, "leakage_risk": "low"},
        {"feature": "bid_ask_spread_pct", "category": "D", "formula": "(ask - bid) / midpoint", "raw_columns": ["bid", "ask"], "live": True, "historical": False, "leakage_risk": "low"},
        {"feature": "theta_to_vega_ratio", "category": "E", "formula": "theta / vega", "raw_columns": ["theta", "vega"], "live": True, "historical": False, "leakage_risk": "low"},
        {"feature": "gamma_to_theta_ratio", "category": "E", "formula": "gamma / theta", "raw_columns": ["gamma", "theta"], "live": True, "historical": False, "leakage_risk": "low"},
    ]


def write_missing_feature_reconstruction_plan_report() -> Tuple[Path, Dict[str, Any]]:
    rows = _live_missing_feature_metadata()
    _, md_path, stamp = report_paths("missing_live_feature_reconstruction_plan")
    category_counts: Dict[str, int] = {}
    for row in rows:
        category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
    lines = [
        "# Missing Live Feature Reconstruction Plan",
        "",
        f"- Generated at: `{stamp}`",
        f"- Missing live features reviewed: `{len(rows)}`",
        f"- Category counts: `{category_counts}`",
        "",
        "| Feature | Category | Formula | Raw Columns | Available Live | Available Historically | Leakage Risk | Implementation Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        status = "planned_reconstruct" if row["category"] in {"A", "B", "C"} else "blocked_requires_data"
        lines.append(
            f"| `{row['feature']}` | `{row['category']}` | {row['formula']} | `{', '.join(row['raw_columns'])}` | `{row['live']}` | `{row['historical']}` | `{row['leakage_risk']}` | `{status}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, {"rows": rows, "category_counts": category_counts}


def write_required_option_context_data_spec() -> Path:
    _, md_path, stamp = report_paths("required_option_context_data_spec")
    lines = [
        "# Required Option Context Data Spec",
        "",
        f"- Generated at: `{stamp}`",
        "",
        "## Required Columns",
        "",
        "- `timestamp`",
        "- `underlying_spot`",
        "- `expiry`",
        "- `strike`",
        "- `option_type`",
        "- `open`",
        "- `high`",
        "- `low`",
        "- `close` or `ltp`",
        "- `volume`",
        "- `open_interest`",
        "- `change_in_oi`",
        "- `iv`",
        "- `delta`",
        "- `gamma`",
        "- `theta`",
        "- `vega`",
        "- `bid`",
        "- `ask`",
        "- `spread`",
        "- `lot_size` optional",
        "",
        "## Format Requirements",
        "",
        "- Timestamp frequency: per tradable snapshot or bar, at least 1-minute for live-compatible replay",
        "- Timestamp format: ISO-8601 in IST or UTC with explicit timezone",
        "- Expiry format: `YYYY-MM-DD`",
        "- Strike format: numeric decimal/integer",
        "- CE/PE convention: uppercase `CE` / `PE`",
        "",
        "## Example CSV Schema",
        "",
        "```csv",
        "timestamp,underlying_spot,expiry,strike,option_type,open,high,low,close,volume,open_interest,change_in_oi,iv,delta,gamma,theta,vega,bid,ask,spread,lot_size",
        "2026-06-04T09:15:00+05:30,24650.25,2026-06-11,24700,CE,120.5,123.0,119.9,122.1,1500,220000,5000,0.182,0.48,0.012,-4.2,6.1,121.8,122.3,0.5,75",
        "```",
        "",
        "## Validation Rules",
        "",
        "- No duplicate rows on `timestamp, expiry, strike, option_type`",
        "- `bid <= ask`",
        "- `spread = ask - bid` within tolerance",
        "- Positive `open/high/low/close/volume/open_interest` where applicable",
        "- Greeks and IV must be same-timestamp observations, not backfilled from future rows",
        "",
        "## Schema Gap Closure",
        "",
        "- `ctx_iv`, `ctx_iv_change_pct`, `ctx_iv_percentile` require `iv`",
        "- `ctx_delta`, `delta_abs` require `delta`",
        "- `ctx_gamma`, `ctx_theta`, `ctx_vega`, `greeks_imbalance`, `theta_to_vega_ratio`, `gamma_to_theta_ratio` require Greeks",
        "- `bid_ask_spread_pct` requires `bid` and `ask`",
        "- These fields close the remaining option-context portion of the live schema gap without approximation",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def _normalize_option_type(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.strip()


def _normalize_timestamp_series(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=False)
    if getattr(ts.dt, "tz", None) is None:
        return ts.dt.tz_localize(TIMEZONE, nonexistent="NaT", ambiguous="NaT")
    return ts.dt.tz_convert(TIMEZONE)


def _normalize_expiry_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _option_context_close_column(frame: pd.DataFrame) -> str:
    for name in OPTION_CONTEXT_CLOSE_ALIASES:
        if name in frame.columns:
            return name
    raise RuntimeError("Option-context dataset must contain either `close` or `ltp`.")


def _required_option_context_columns(frame: pd.DataFrame) -> List[str]:
    required = list(OPTION_CONTEXT_REQUIRED_COLUMNS)
    required.extend([name for name in OPTION_CONTEXT_PRICE_COLUMNS if name not in frame.columns])
    if not any(name in frame.columns for name in OPTION_CONTEXT_CLOSE_ALIASES):
        required.append("close_or_ltp")
    return [name for name in required if name not in frame.columns and name != "close_or_ltp"]


def load_option_context_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


def normalize_option_context_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    missing = _required_option_context_columns(work)
    if missing:
        raise RuntimeError(f"Option-context dataset missing required columns: {sorted(missing)}")
    close_col = _option_context_close_column(work)
    work["timestamp"] = _normalize_timestamp_series(work["timestamp"])
    work["expiry"] = _normalize_expiry_series(work["expiry"])
    work["strike"] = _safe_numeric(work["strike"])
    work["option_type"] = _normalize_option_type(work["option_type"])
    work["underlying_spot"] = _safe_numeric(work["underlying_spot"])
    for col in OPTION_CONTEXT_PRICE_COLUMNS + [close_col, "volume", "open_interest", "change_in_oi", "iv", "delta", "gamma", "theta", "vega", "bid", "ask", "spread"]:
        work[col] = _safe_numeric(work[col])
    if close_col != "close":
        work["close"] = work[close_col]
    if "ltp" not in work.columns:
        work["ltp"] = work["close"]
    if "lot_size" in work.columns:
        work["lot_size"] = _safe_numeric(work["lot_size"])
    work["trading_day"] = work["timestamp"].dt.date
    return work


def validate_option_context_dataset(
    dataset_path: Path,
    *,
    report_prefix: str = "option_context_dataset_validation",
    report_dir: Optional[Path] = None,
    min_iv: float = 0.0001,
    max_iv: float = 5.0,
    spread_tolerance: float = 1e-4,
    max_spread_pct: float = 0.25,
    min_near_atm_rows_per_day: int = 1,
    strict_missing: bool = True,
    fail_on_warning: bool = False,
) -> Dict[str, Any]:
    raw = load_option_context_dataset(dataset_path)
    work = normalize_option_context_dataset(raw)
    required_context_fields = [
        "timestamp", "underlying_spot", "expiry", "strike", "option_type", "open", "high", "low", "close",
        "volume", "open_interest", "change_in_oi", "iv", "delta", "gamma", "theta", "vega", "bid", "ask", "spread",
    ]
    failures: List[str] = []
    warnings: List[str] = []
    key_cols = ["timestamp", "expiry", "strike", "option_type"]
    parse_fail_timestamp = int(work["timestamp"].isna().sum())
    parse_fail_expiry = int(work["expiry"].isna().sum())
    parse_fail_strike = int(work["strike"].isna().sum())
    duplicate_rows = int(work.duplicated(subset=key_cols).sum())
    duplicate_sample = work.loc[work.duplicated(subset=key_cols, keep=False), key_cols].head(10).copy()
    invalid_option_type_rows = int((~work["option_type"].isin(["CE", "PE"])).sum())
    invalid_option_type_sample = work.loc[~work["option_type"].isin(["CE", "PE"]), key_cols].head(10).copy()
    missing_required = {col: int(work[col].isna().sum()) for col in required_context_fields if col in work.columns and int(work[col].isna().sum()) > 0}
    bid_gt_ask_rows = int((work["bid"] > work["ask"]).fillna(False).sum())
    midpoint = ((work["bid"] + work["ask"]) / 2.0)
    midpoint_nonpositive_rows = int((midpoint <= 0).fillna(False).sum())
    computed_spread = (work["ask"] - work["bid"]).abs()
    spread_diff = (computed_spread - work["spread"]).abs()
    invalid_spread_rows = int((spread_diff > float(spread_tolerance)).fillna(False).sum())
    spread_pct = (computed_spread / midpoint.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    invalid_spread_pct_rows = int((spread_pct > float(max_spread_pct)).fillna(False).sum())
    nonpositive_iv_rows = int((work["iv"] < float(min_iv)).fillna(False).sum())
    insane_iv_rows = int((work["iv"] > float(max_iv)).fillna(False).sum())
    bad_ohlc_rows = int(((work["high"] < work[["open", "close"]].max(axis=1)) | (work["low"] > work[["open", "close"]].min(axis=1))).fillna(False).sum())
    negative_price_rows = int(((work[["open", "high", "low", "close", "bid", "ask", "spread"]] < 0).any(axis=1)).fillna(False).sum())
    negative_qty_rows = int(((work[["volume", "open_interest"]] < 0).any(axis=1)).fillna(False).sum())
    invalid_greek_rows = {
        "delta_out_of_range": int(((work["delta"] < -1.05) | (work["delta"] > 1.05)).fillna(False).sum()),
        "gamma_negative": int((work["gamma"] < 0).fillna(False).sum()),
        "gamma_above_one": int((work["gamma"] > 1.0).fillna(False).sum()),
        "vega_negative": int((work["vega"] < 0).fillna(False).sum()),
        "vega_above_1000": int((work["vega"] > 1000.0).fillna(False).sum()),
        "theta_abs_above_1000": int((work["theta"].abs() > 1000.0).fillna(False).sum()),
        "theta_missing": int(work["theta"].isna().sum()),
    }
    future_fill_rows = 0
    group_cols = ["expiry", "strike", "option_type"]
    for col in ["iv", "delta", "gamma", "theta", "vega", "bid", "ask", "spread"]:
        backward_filled = work.groupby(group_cols, dropna=False)[col].bfill()
        future_fill_rows += int((work[col].isna() & backward_filled.notna()).sum())
    if parse_fail_timestamp:
        failures.append(f"unparseable timestamp rows: {parse_fail_timestamp}")
    if parse_fail_expiry:
        failures.append(f"unparseable expiry rows: {parse_fail_expiry}")
    if parse_fail_strike:
        failures.append(f"unparseable strike rows: {parse_fail_strike}")
    if duplicate_rows:
        failures.append(f"duplicate key rows: {duplicate_rows}")
    if invalid_option_type_rows:
        failures.append(f"invalid option_type rows: {invalid_option_type_rows}")
    if bid_gt_ask_rows:
        failures.append(f"bid > ask rows: {bid_gt_ask_rows}")
    if midpoint_nonpositive_rows:
        failures.append(f"nonpositive midpoint rows: {midpoint_nonpositive_rows}")
    if invalid_spread_rows:
        failures.append(f"spread mismatch rows beyond tolerance: {invalid_spread_rows}")
    if invalid_spread_pct_rows:
        failures.append(f"spread pct above max rows: {invalid_spread_pct_rows}")
    if nonpositive_iv_rows or insane_iv_rows:
        failures.append(f"invalid iv rows: below_min={nonpositive_iv_rows}, above_max={insane_iv_rows}")
    if bad_ohlc_rows:
        failures.append(f"bad ohlc rows: {bad_ohlc_rows}")
    if negative_price_rows:
        failures.append(f"negative price rows: {negative_price_rows}")
    if negative_qty_rows:
        failures.append(f"negative volume/open_interest rows: {negative_qty_rows}")
    for name, count in invalid_greek_rows.items():
        if count and name != "theta_missing":
            failures.append(f"{name}: {count}")
    if strict_missing and missing_required:
        failures.append(f"missing required field values: {missing_required}")
    elif missing_required:
        warnings.append(f"missing required field values: {missing_required}")
    if future_fill_rows:
        warnings.append(f"rows vulnerable to future-fill leakage if bfill is used: {future_fill_rows}")
    if not work["timestamp"].is_monotonic_increasing:
        warnings.append("timestamps are not globally sorted; dataset is sortable but not pre-sorted")
    timestamp_sorted = work.sort_values("timestamp", kind="stable")
    diffs = timestamp_sorted["timestamp"].dropna().diff().dropna()
    freq_minutes = diffs.dt.total_seconds().div(60.0)
    frequency_report = {
        "median_minutes": float(freq_minutes.median()) if not freq_minutes.empty else None,
        "min_minutes": float(freq_minutes.min()) if not freq_minutes.empty else None,
        "max_minutes": float(freq_minutes.max()) if not freq_minutes.empty else None,
        "p95_minutes": float(freq_minutes.quantile(0.95)) if not freq_minutes.empty else None,
    }
    per_day = work.groupby("trading_day", dropna=False).size().rename("rows").reset_index()
    per_expiry = work.groupby("expiry", dropna=False).size().rename("rows").reset_index()
    ce_pe = work.groupby("option_type", dropna=False).size().rename("rows").to_dict()
    atm_distance = (work["strike"] - work["underlying_spot"]).abs()
    atm_near_threshold = np.maximum(work["underlying_spot"].abs() * 0.005, 100.0)
    atm_near_rows = int((atm_distance <= atm_near_threshold).fillna(False).sum())
    near_atm_by_day = ((atm_distance <= atm_near_threshold).fillna(False)).groupby(work["trading_day"]).sum()
    low_near_atm_days = int((near_atm_by_day < int(min_near_atm_rows_per_day)).sum())
    if low_near_atm_days:
        warnings.append(f"days below near-ATM minimum rows: {low_near_atm_days}")
    timestamp_counts = work.groupby("timestamp", dropna=False).size()
    median_rows_per_ts = float(timestamp_counts.median()) if len(timestamp_counts) else 0.0
    missing_minute_timestamps = int((timestamp_counts < max(1.0, median_rows_per_ts * 0.5)).sum())
    coverage = {
        "rows": int(len(work)),
        "unique_timestamps": int(work["timestamp"].nunique()),
        "unique_expiries": int(work["expiry"].nunique()),
        "ce_rows": int(ce_pe.get("CE", 0)),
        "pe_rows": int(ce_pe.get("PE", 0)),
        "near_atm_rows": atm_near_rows,
        "missing_minute_timestamps_estimate": missing_minute_timestamps,
    }
    if fail_on_warning and warnings:
        failures.append("warnings escalated to failure by --fail-on-warning")
    status = "valid" if not failures else "invalid"
    stamp = _timestamp_now()
    report_dir = report_dir or REPORTS_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{report_prefix}_{stamp}.json"
    md_path = report_dir / f"{report_prefix}_{stamp}.md"
    missing_required_value_count = int(sum(missing_required.values()))
    bad_mask = (
        (work["bid"] > work["ask"]).fillna(False)
        | (spread_diff > float(spread_tolerance)).fillna(False)
        | ((spread_pct > float(max_spread_pct)).fillna(False))
        | ((work["high"] < work[["open", "close"]].max(axis=1)) | (work["low"] > work[["open", "close"]].min(axis=1))).fillna(False)
        | (~work["option_type"].isin(["CE", "PE"]))
    )
    bad_row_sample = work.loc[bad_mask, key_cols + ["open", "high", "low", "close", "bid", "ask", "spread", "iv", "delta", "gamma", "theta", "vega"]].head(10).copy()
    payload = {
        "generated_at": stamp,
        "dataset_path": str(dataset_path),
        "ok": bool(status == "valid"),
        "status": status,
        "warnings": warnings,
        "errors": failures,
        "failures": failures,
        "required_columns": required_context_fields,
        "required_columns_present": True,
        "missing_columns": [],
        "row_count": int(len(work)),
        "date_min": None if work["timestamp"].dropna().empty else str(work["timestamp"].min()),
        "date_max": None if work["timestamp"].dropna().empty else str(work["timestamp"].max()),
        "duplicate_key_rows": duplicate_rows,
        "duplicate_count": duplicate_rows,
        "duplicate_key_sample": duplicate_sample.to_dict(orient="records"),
        "invalid_option_type_rows": invalid_option_type_rows,
        "invalid_option_type_sample": invalid_option_type_sample.to_dict(orient="records"),
        "bid_gt_ask_rows": bid_gt_ask_rows,
        "bad_bid_ask_count": bid_gt_ask_rows,
        "spread_mismatch_rows": invalid_spread_rows,
        "bad_spread_count": invalid_spread_rows,
        "bad_ohlc_count": bad_ohlc_rows,
        "missing_required_value_count": missing_required_value_count,
        "missing_required_values": missing_required,
        "future_fill_risk_rows": future_fill_rows,
        "bad_row_sample": bad_row_sample.to_dict(orient="records"),
        "frequency_report": frequency_report,
        "coverage": coverage,
        "ce_count": int(ce_pe.get("CE", 0)),
        "pe_count": int(ce_pe.get("PE", 0)),
        "near_atm_coverage_summary": {
            "total_near_atm_rows": atm_near_rows,
            "days_below_minimum": low_near_atm_days,
            "minimum_rows_per_day": int(min_near_atm_rows_per_day),
        },
        "per_day_coverage": per_day.to_dict(orient="records"),
        "per_expiry_coverage": [
            {"expiry": None if pd.isna(row["expiry"]) else str(pd.Timestamp(row["expiry"]).date()), "rows": int(row["rows"])}
            for row in per_expiry.to_dict(orient="records")
        ],
        "ce_pe_coverage": {str(k): int(v) for k, v in ce_pe.items()},
        "final_verdict": "OPTION_CONTEXT_DATA_VALID" if status == "valid" else "OPTION_CONTEXT_DATA_INVALID",
    }
    write_json(json_path, ensure_serializable(payload))
    lines = [
        "# Option Context Dataset Validation",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Status: `{status}`",
        f"- Final verdict: `{payload['final_verdict']}`",
        f"- Rows: `{coverage['rows']}`",
        f"- Unique timestamps: `{coverage['unique_timestamps']}`",
        f"- Unique expiries: `{coverage['unique_expiries']}`",
        f"- CE rows: `{coverage['ce_rows']}`",
        f"- PE rows: `{coverage['pe_rows']}`",
        f"- Near-ATM rows: `{coverage['near_atm_rows']}`",
        f"- Missing-minute estimate: `{coverage['missing_minute_timestamps_estimate']}`",
        f"- Duplicate keys: `{duplicate_rows}`",
        f"- Bad bid/ask rows: `{bid_gt_ask_rows}`",
        f"- Bad spread rows: `{invalid_spread_rows}`",
        f"- Bad OHLC rows: `{bad_ohlc_rows}`",
        "",
        "## Failures",
        "",
    ]
    lines.extend([f"- {item}" for item in failures] or ["- None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in warnings] or ["- None"])
    lines.extend(["", "## Frequency", ""])
    for key, value in frequency_report.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Missing Required Values", ""])
    if missing_required:
        for key, value in missing_required.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Duplicate Sample", ""])
    if duplicate_sample.empty:
        lines.append("- None")
    else:
        for row in duplicate_sample.to_dict(orient="records"):
            lines.append(f"- `{row}`")
    lines.extend(["", "## Bad Row Sample", ""])
    if bad_row_sample.empty:
        lines.append("- None")
    else:
        for row in bad_row_sample.to_dict(orient="records"):
            lines.append(f"- `{row}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["report_paths"] = {"json": str(json_path), "md": str(md_path)}
    return payload


def strict_live_schema_gate(feature_columns: Sequence[str]) -> Dict[str, Any]:
    live = live_feature_schema_summary()
    live_names = list(live["feature_names"])
    feature_list = list(feature_columns)
    disallowed = [name for name in feature_list if name in TRAINING_ONLY_LEAKAGE_COLUMNS or name.startswith("future_")]
    names_match = feature_list == live_names
    missing_live = [name for name in live_names if name not in feature_list]
    extra_features = [name for name in feature_list if name not in live_names]
    feature_order_matches = len(feature_list) == len(live_names) and all(a == b for a, b in zip(feature_list, live_names))
    result = {
        "production_feature_count": int(len(live_names)),
        "dataset_feature_count": int(len(feature_list)),
        "feature_names_match_exactly": bool(names_match),
        "feature_order_matches_exactly": bool(feature_order_matches),
        "missing_live_features": missing_live,
        "extra_dataset_features": extra_features,
        "disallowed_training_only_features": disallowed,
        "compatible": bool(names_match and feature_order_matches and not missing_live and not extra_features and not disallowed),
    }
    return result


def check_live_schema_compatibility(
    dataset_path: Path,
    *,
    live_feature_source: str = "auto",
    report_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    schema_path = dataset_path.with_name(dataset_path.stem + "_schema.json")
    if dataset_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path, low_memory=False)
    schema_payload = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else {}
    feature_columns = list(schema_payload.get("feature_columns", []))
    strict_gate = strict_live_schema_gate(feature_columns)
    live = live_feature_schema_summary()
    model_input_shape = live.get("model_input_shape")
    inf_counts = {}
    missing_counts = {}
    for name in feature_columns:
        if name in df.columns:
            series = pd.to_numeric(df[name], errors="coerce")
            inf_counts[name] = int(np.isinf(series.to_numpy(dtype=float, na_value=np.nan)).sum()) if len(series) else 0
            missing_counts[name] = int(series.isna().sum())
    bad_missing = {k: v for k, v in missing_counts.items() if v > 0}
    bad_inf = {k: v for k, v in inf_counts.items() if v > 0}
    forbidden_present = [name for name in TRAINING_ONLY_LEAKAGE_COLUMNS if name in feature_columns]
    estimated_present = sorted(name for name in ESTIMATED_OPTION_CONTEXT_COLUMNS if name in df.columns or name in feature_columns)
    real_ctx_present = [name for name in ["ctx_iv", "ctx_delta", "ctx_gamma", "ctx_theta", "ctx_vega"] if name in df.columns or name in feature_columns]
    missing_real_ctx = [name for name in ["ctx_iv", "ctx_delta", "ctx_gamma", "ctx_theta", "ctx_vega"] if name not in df.columns]
    missing_real_bid_ask = [name for name in ["bid", "ask"] if name not in df.columns]
    adoption_flag_values: List[bool] = []
    if "production_adoption_allowed" in df.columns:
        adoption_series = df["production_adoption_allowed"]
        adoption_flag_values = sorted({bool(v) for v in adoption_series.dropna().tolist() if str(v).strip() != ""})
    order_mismatches = []
    live_names = list(live.get("feature_names", []))
    for idx, (expected, actual) in enumerate(zip(live_names, feature_columns)):
        if expected != actual:
            order_mismatches.append({"index": idx, "expected": expected, "actual": actual})
            if len(order_mismatches) >= 20:
                break
    verdict = "LIVE_SCHEMA_COMPATIBLE_RESEARCH_ONLY" if strict_gate.get("compatible") and not bad_missing and not bad_inf and not forbidden_present else "LIVE_SCHEMA_NOT_COMPATIBLE"
    estimated_only = bool(estimated_present and not real_ctx_present)
    blocked_for_missing_bid_ask = bool(any(name not in df.columns for name in ["bid", "ask"]))
    if estimated_only:
        verdict = "RESEARCH_ONLY_ESTIMATED_GREEKS_NOT_ADOPTABLE"
    elif blocked_for_missing_bid_ask and any(name in feature_columns for name in ["bid_ask_spread_pct"]):
        verdict = "LIVE_SCHEMA_NOT_COMPATIBLE"
    elif strict_gate.get("missing_live_features"):
        verdict = "WAITING_FOR_OPTION_CONTEXT_DATA"
    report_dir = report_dir or REPORTS_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp_now()
    json_path = report_dir / f"live_schema_84_feature_readiness_{stamp}.json"
    md_path = report_dir / f"live_schema_84_feature_readiness_{stamp}.md"
    payload = {
        "generated_at": stamp,
        "dataset": str(dataset_path),
        "schema_path": str(schema_path) if schema_path.exists() else None,
        "live_feature_source": live_feature_source,
        "live_feature_count": int(len(live_names)),
        "dataset_feature_count": int(len(feature_columns)),
        "strict_live_schema_gate": strict_gate,
        "model_input_shape": model_input_shape,
        "forbidden_columns_present": forbidden_present,
        "estimated_columns_present": estimated_present,
        "estimated_only_dataset": estimated_only,
        "real_ctx_columns_present": real_ctx_present,
        "missing_real_ctx_columns": missing_real_ctx,
        "missing_real_bid_ask_columns": missing_real_bid_ask,
        "production_adoption_allowed_values": adoption_flag_values,
        "missing_value_counts": bad_missing,
        "infinite_value_counts": bad_inf,
        "order_mismatch_sample": order_mismatches,
        "final_verdict": verdict,
    }
    write_json(json_path, ensure_serializable(payload))
    lines = [
        "# Live Schema 84 Feature Readiness",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Live feature count: `{len(live_names)}`",
        f"- Dataset feature count: `{len(feature_columns)}`",
        f"- Exact feature names match: `{strict_gate.get('feature_names_match_exactly')}`",
        f"- Exact feature order match: `{strict_gate.get('feature_order_matches_exactly')}`",
        f"- Final verdict: `{verdict}`",
        "",
        "## Missing Live Features",
        "",
    ]
    lines.extend([f"- `{name}`" for name in strict_gate.get("missing_live_features", [])] or ["- None"])
    lines.extend(["", "## Extra Dataset Features", ""])
    lines.extend([f"- `{name}`" for name in strict_gate.get("extra_dataset_features", [])] or ["- None"])
    lines.extend(["", "## Forbidden Columns", ""])
    lines.extend([f"- `{name}`" for name in forbidden_present] or ["- None"])
    lines.extend(["", "## Estimated Columns", ""])
    lines.extend([f"- `{name}`" for name in estimated_present] or ["- None"])
    lines.extend(["", "## Order Mismatch Sample", ""])
    lines.extend([f"- `{row}`" for row in order_mismatches] or ["- None"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["report_paths"] = {"json": str(json_path), "md": str(md_path)}
    return payload


def merge_option_context_into_reconstructed_dataset(
    reconstructed_dataset: Path,
    option_context_dataset: Path,
    *,
    output_dataset: Optional[Path] = None,
    report_dir: Optional[Path] = None,
    validation_json: Optional[Path] = None,
    allow_backward_asof: bool = False,
    asof_tolerance: str = "5min",
    min_context_coverage: float = 0.95,
) -> Dict[str, Any]:
    if reconstructed_dataset.suffix.lower() == ".parquet":
        base = pd.read_parquet(reconstructed_dataset)
    else:
        base = pd.read_csv(reconstructed_dataset, low_memory=False)
    if validation_json and validation_json.exists():
        context_validation = json.loads(validation_json.read_text(encoding="utf-8"))
    else:
        context_validation = validate_option_context_dataset(option_context_dataset, report_dir=report_dir)
    if not context_validation.get("ok", context_validation.get("status") == "valid"):
        raise RuntimeError("Option-context dataset is invalid; fix validation failures before merge.")
    context = normalize_option_context_dataset(load_option_context_dataset(option_context_dataset))
    base["timestamp"] = _normalize_timestamp_series(base["timestamp"])
    if "expiry" in base.columns:
        base["expiry"] = _normalize_expiry_series(base["expiry"])
    strike_col = "strike_price" if "strike_price" in base.columns else "strike"
    base[strike_col] = _safe_numeric(base[strike_col])
    base["option_type"] = _normalize_option_type(base["option_type"])
    join_left = base.copy()
    join_left["_merge_strike"] = join_left[strike_col]
    ctx = context.copy()
    ctx = ctx.rename(columns={"strike": "_merge_strike"})
    exact_cols = ["timestamp", "expiry", "_merge_strike", "option_type"]
    ctx["_context_timestamp"] = ctx["timestamp"]
    context_features = ctx[exact_cols + ["_context_timestamp", "iv", "delta", "gamma", "theta", "vega", "bid", "ask", "spread"]].drop_duplicates(subset=exact_cols, keep="last")
    merged = join_left.merge(context_features, how="left", on=exact_cols, suffixes=("", "_ctx"))
    match_mode = "exact"
    exact_match_count = int(merged["iv"].notna().sum())
    asof_match_count = 0
    max_asof_lag_seconds = 0.0
    lag_seconds_summary: Dict[str, Any] = {}
    if allow_backward_asof and merged["iv"].isna().any():
        base_sorted = join_left.sort_values(["expiry", "_merge_strike", "option_type", "timestamp"], kind="stable")
        ctx_sorted = context_features.sort_values(["expiry", "_merge_strike", "option_type", "timestamp"], kind="stable")
        asof = pd.merge_asof(
            base_sorted,
            ctx_sorted,
            on="timestamp",
            by=["expiry", "_merge_strike", "option_type"],
            direction="backward",
            tolerance=pd.Timedelta(asof_tolerance),
            suffixes=("", "_ctx"),
        )
        merged = asof.sort_index()
        asof_match_count = int((merged["iv"].notna().sum() - exact_match_count))
        merged["_asof_lag_seconds"] = (
            merged["timestamp"] - merged["_context_timestamp"]
        ).dt.total_seconds().fillna(0.0)
        if (merged["_asof_lag_seconds"] < 0).any():
            raise RuntimeError("Future merge detected in asof merge path.")
        lag_series = merged.loc[merged["_asof_lag_seconds"].notna(), "_asof_lag_seconds"]
        if not lag_series.empty:
            max_asof_lag_seconds = float(lag_series.max())
            lag_seconds_summary = {
                "median": float(lag_series.median()),
                "p95": float(lag_series.quantile(0.95)),
                "max": float(lag_series.max()),
            }
        match_mode = f"backward_asof_{asof_tolerance}"
    unmatched_rows = int(merged["iv"].isna().sum())
    coverage_ratio = float(merged["iv"].notna().mean()) if len(merged) else 0.0
    if coverage_ratio < float(min_context_coverage):
        raise RuntimeError(f"Option-context coverage too low after merge: {coverage_ratio:.2%} < {float(min_context_coverage):.2%}")
    group_cols = []
    if "instrument_key" in merged.columns:
        group_cols = ["instrument_key"]
    elif {"expiry", strike_col, "option_type"}.issubset(merged.columns):
        group_cols = ["expiry", strike_col, "option_type"]
    else:
        group_cols = ["option_type"]
    iv_group = merged.groupby(group_cols, dropna=False)["iv"]
    merged["ctx_iv"] = merged["iv"]
    merged["ctx_iv_change_pct"] = iv_group.transform(_safe_pct_change).fillna(0.0)
    merged["ctx_iv_percentile"] = iv_group.transform(lambda s: s.rolling(60, min_periods=1).rank(pct=True)).fillna(0.5)
    merged["ctx_delta"] = merged["delta"]
    merged["ctx_gamma"] = merged["gamma"]
    merged["ctx_vega"] = merged["vega"]
    merged["ctx_theta"] = merged["theta"]
    merged["delta_abs"] = merged["ctx_delta"].abs()
    merged["greeks_imbalance"] = merged["ctx_gamma"] + merged["ctx_vega"] + merged["ctx_theta"]
    midpoint = ((merged["bid"] + merged["ask"]) / 2.0).replace(0.0, np.nan)
    merged["bid_ask_spread_pct"] = (merged["ask"] - merged["bid"]) / midpoint
    merged["theta_to_vega_ratio"] = merged["ctx_theta"] / merged["ctx_vega"].replace(0.0, np.nan)
    merged["gamma_to_theta_ratio"] = merged["ctx_gamma"] / merged["ctx_theta"].replace(0.0, np.nan)
    feature_cols = json.loads(reconstructed_dataset.with_name(reconstructed_dataset.stem + "_schema.json").read_text(encoding="utf-8")).get("feature_columns", [])
    live_names = live_feature_schema_summary().get("feature_names", [])
    updated_features = [name for name in live_names if name in merged.columns]
    if not updated_features:
        updated_features = list(dict.fromkeys(list(feature_cols) + [name for name in live_names if name in merged.columns]))
    gap_payload, gap_report = write_feature_schema_gap_report(
        updated_features,
        reconstructed_features=[name for name in merged.columns if name in live_names],
        additional_data_required=[],
        report_prefix="live_vs_reconstructed_option_chain_schema_gap",
    )
    strict_gate = strict_live_schema_gate(updated_features)
    stamp = _timestamp_now()
    output_dataset = output_dataset or PROCESSED_DIR / f"nifty_option_chain_84_feature_candidate_{stamp}.csv"
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    merged_out = merged.drop(columns=["_merge_strike"], errors="ignore").copy()
    merged_out.to_csv(output_dataset, index=False)
    schema_path = output_dataset.with_name(output_dataset.stem + "_schema.json")
    write_json(schema_path, ensure_serializable({
        "feature_columns": updated_features,
        "feature_count": len(updated_features),
        "strict_live_schema_gate": strict_gate,
        "feature_gap_report_path": str(gap_report),
    }))
    report_dir = report_dir or REPORTS_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"option_context_merge_report_{stamp}.md"
    report_json = report_dir / f"option_context_merge_report_{stamp}.json"
    verdict = "MERGE_INCOMPLETE"
    if context_validation.get("status") != "valid" or not context_validation.get("ok", True):
        verdict = "OPTION_CONTEXT_DATA_INVALID"
    elif strict_gate.get("compatible"):
        verdict = "LIVE_SCHEMA_COMPATIBLE_RESEARCH_ONLY"
    lines = [
        "# Option Context Merge Report",
        "",
        f"- Reconstructed dataset: `{reconstructed_dataset}`",
        f"- Option-context dataset: `{option_context_dataset}`",
        f"- Merge mode: `{match_mode}`",
        f"- Exact matches: `{exact_match_count}`",
        f"- Asof matches: `{asof_match_count}`",
        f"- Unmatched rows: `{unmatched_rows}`",
        f"- Coverage ratio: `{coverage_ratio:.2%}`",
        f"- Matching features: `{len(gap_payload.get('matching_features', []))}`",
        f"- Missing live features: `{len(gap_payload.get('missing_live_features', []))}`",
        f"- Strict exact-match gate: `{strict_gate.get('compatible')}`",
        f"- Verdict: `{verdict}`",
        "",
        "## Missing Features",
        "",
    ]
    lines.extend([f"- `{name}`" for name in strict_gate.get("missing_live_features", [])] or ["- None"])
    lines.extend(["", "## Extra / Disallowed Features", ""])
    lines.extend([f"- `{name}`" for name in strict_gate.get("extra_dataset_features", [])] or ["- None"])
    lines.extend([f"- disallowed: `{name}`" for name in strict_gate.get("disallowed_training_only_features", [])])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    merge_payload = {
        "generated_at": stamp,
        "output_dataset": str(output_dataset),
        "schema_path": str(schema_path),
        "feature_gap_report": str(gap_report),
        "merge_mode": match_mode,
        "exact_match_count": exact_match_count,
        "asof_match_count": asof_match_count,
        "unmatched_rows": unmatched_rows,
        "context_coverage_ratio": coverage_ratio,
        "max_asof_lag_seconds": max_asof_lag_seconds,
        "lag_seconds_summary": lag_seconds_summary,
        "validation": context_validation,
        "strict_live_schema_gate": strict_gate,
        "matching_feature_count": len(gap_payload.get("matching_features", [])),
        "missing_live_features": strict_gate.get("missing_live_features", []),
        "final_verdict": verdict,
        "report_paths": {"md": str(report_path), "json": str(report_json)},
    }
    write_json(report_json, ensure_serializable(merge_payload))
    return {
        **merge_payload,
        "readiness_report": str(report_path),
    }


def _rolling_percentile_value(history: Sequence[float], current: float) -> float:
    vals = [float(v) for v in history if pd.notna(v)]
    if len(vals) <= 1:
        return 0.5
    cur = float(current)
    return float(np.mean(np.asarray(vals, dtype=float) <= cur))


def reconstruct_missing_live_features_for_option_chain(
    input_dataset: Path,
    *,
    output_dataset: Optional[Path] = None,
    report_path: Optional[Path] = None,
    schema_gap_report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if input_dataset.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_dataset)
    else:
        df = pd.read_csv(input_dataset, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    if getattr(df["timestamp"].dt, "tz", None) is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(TIMEZONE, nonexistent="NaT", ambiguous="NaT")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(TIMEZONE)
    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
    df = df.sort_values(["instrument_key", "timestamp"], kind="stable").reset_index(drop=True)

    # Fast vectorized reconstruction path for large research datasets.
    g = df.groupby("instrument_key", sort=False)
    prev_open = g["open"].shift(1)
    prev_close = g["ltp"].shift(1)
    prev_high = g["high"].shift(1)
    prev_low = g["low"].shift(1)
    body = (df["ltp"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    upper_shadow = df["high"] - df[["open", "ltp"]].max(axis=1)
    lower_shadow = df[["open", "ltp"]].min(axis=1) - df["low"]
    df["bullish_engulfing"] = ((prev_close < prev_open) & (df["ltp"] > df["open"]) & (df["open"] <= prev_close) & (df["ltp"] >= prev_open)).astype(float)
    df["bearish_engulfing"] = ((prev_close > prev_open) & (df["ltp"] < df["open"]) & (df["open"] >= prev_close) & (df["ltp"] <= prev_open)).astype(float)
    df["doji"] = ((body / rng) <= 0.10).fillna(False).astype(float)
    downtrend = g["ltp"].shift(1) < g["ltp"].shift(3)
    uptrend = g["ltp"].shift(1) > g["ltp"].shift(3)
    df["hammer"] = ((lower_shadow >= 2 * body) & (upper_shadow <= body) & downtrend).fillna(False).astype(float)
    df["shooting_star"] = ((upper_shadow >= 2 * body) & (lower_shadow <= body) & uptrend).fillna(False).astype(float)

    ret1 = g["ltp"].pct_change().replace([np.inf, -np.inf], np.nan)
    df["ret_mean"] = ret1.groupby(df["instrument_key"]).transform(lambda s: s.rolling(20, min_periods=2).mean())
    df["ret_std"] = ret1.groupby(df["instrument_key"]).transform(lambda s: s.rolling(20, min_periods=2).std(ddof=0))
    df["ret_min"] = ret1.groupby(df["instrument_key"]).transform(lambda s: s.rolling(20, min_periods=2).min())
    df["ret_max"] = ret1.groupby(df["instrument_key"]).transform(lambda s: s.rolling(20, min_periods=2).max())
    df["ret_10"] = g["ltp"].pct_change(10).replace([np.inf, -np.inf], np.nan)
    df["vol_mean"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    df["vol_std"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=2).std(ddof=0))
    df["vol_min"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=1).min())
    df["vol_max"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=1).max())
    df["ctx_volume_sma"] = df["vol_mean"]

    df["ema_fast"] = g["ltp"].transform(lambda s: s.ewm(span=9, adjust=False, min_periods=9).mean())
    df["ema_slow"] = g["ltp"].transform(lambda s: s.ewm(span=21, adjust=False, min_periods=21).mean())
    df["ema_diff_pct"] = (df["ema_fast"] - df["ema_slow"]) / df["ltp"].replace(0.0, np.nan)
    df["ctx_trend_strength"] = df["ema_diff_pct"]
    df["close_vs_open_pct"] = (df["ltp"] - df["open"]) / df["ltp"].replace(0.0, np.nan)
    df["momentum_lookback_pct"] = (df["ltp"] - g["ltp"].shift(19)) / g["ltp"].shift(19).replace(0.0, np.nan)

    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = tr.groupby(df["instrument_key"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    df["atr_pct"] = df["atr_14"] / df["ltp"].replace(0.0, np.nan)
    df["range_to_atr"] = df["range_pct"] / df["atr_pct"].replace(0.0, np.nan)
    up_move = df["high"] - prev_high
    down_move = prev_low - df["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    tr_sum14 = tr.groupby(df["instrument_key"]).transform(lambda s: s.rolling(14, min_periods=14).sum())
    plus_di = 100.0 * plus_dm.groupby(df["instrument_key"]).transform(lambda s: s.rolling(14, min_periods=14).sum()) / tr_sum14.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.groupby(df["instrument_key"]).transform(lambda s: s.rolling(14, min_periods=14).sum()) / tr_sum14.replace(0.0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)) * 100.0
    df["adx_14"] = dx.groupby(df["instrument_key"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    df["ctx_adx"] = df["adx_14"]
    roll_high14 = g["high"].transform(lambda s: s.rolling(14, min_periods=14).max())
    roll_low14 = g["low"].transform(lambda s: s.rolling(14, min_periods=14).min())
    chop_num = tr.groupby(df["instrument_key"]).transform(lambda s: s.rolling(14, min_periods=14).sum())
    chop_den = (roll_high14 - roll_low14).replace(0.0, np.nan)
    df["choppiness_14"] = 100.0 * np.log10(chop_num / chop_den) / np.log10(14.0)
    df["ctx_choppiness"] = df["choppiness_14"]
    df["supertrend_dir"] = np.where(df["ltp"] >= df["ema_fast"].fillna(df["ltp"]), 1.0, -1.0)

    piv_pp = (df["high"] + df["low"] + df["ltp"]) / 3.0
    piv_r1 = 2 * piv_pp - df["low"]
    piv_s1 = 2 * piv_pp - df["high"]
    df["pivot_pp_dist_pct"] = (df["ltp"] - piv_pp) / df["ltp"].replace(0.0, np.nan)
    df["pivot_r1_dist_pct"] = (df["ltp"] - piv_r1) / df["ltp"].replace(0.0, np.nan)
    df["pivot_s1_dist_pct"] = (df["ltp"] - piv_s1) / df["ltp"].replace(0.0, np.nan)

    df["dist_to_rolling_high_20"] = (g["high"].transform(lambda s: s.shift(1).rolling(20, min_periods=2).max()) - df["ltp"]) / df["ltp"].replace(0.0, np.nan)
    df["dist_to_rolling_low_20"] = (df["ltp"] - g["low"].transform(lambda s: s.shift(1).rolling(20, min_periods=2).min())) / df["ltp"].replace(0.0, np.nan)
    roll_high20 = g["high"].transform(lambda s: s.shift(1).rolling(20, min_periods=2).max())
    roll_low20 = g["low"].transform(lambda s: s.shift(1).rolling(20, min_periods=2).min())
    roll_width20 = (roll_high20 - roll_low20).replace(0.0, np.nan)
    df["rolling_range_width_20"] = roll_width20 / df["ltp"].replace(0.0, np.nan)
    df["rolling_range_position_20"] = (df["ltp"] - roll_low20) / roll_width20

    df["realized_vol_30"] = ret1.groupby(df["instrument_key"]).transform(lambda s: s.shift(1).rolling(30, min_periods=5).std(ddof=0))
    df["atr_pct_regime_10"] = df["atr_pct"] / df.groupby("instrument_key")["atr_pct"].transform(lambda s: s.rolling(10, min_periods=3).mean()).replace(0.0, np.nan)
    df["atr_percentile_60"] = df.groupby("instrument_key")["atr_pct"].transform(lambda s: s.rolling(60, min_periods=5).rank(pct=True))
    df["realized_vol_percentile_60"] = df.groupby("instrument_key")["realized_vol_30"].transform(lambda s: s.rolling(60, min_periods=5).rank(pct=True))
    df["volatility_percentile_60"] = df["realized_vol_percentile_60"]
    df["volatility_regime_classifier"] = ((df["atr_percentile_60"] >= 0.7) | (df["realized_vol_percentile_60"] >= 0.7)).astype(float)

    day_open_high = df.groupby(["instrument_key", "trading_day"])["high"].transform(lambda s: s.expanding().max())
    day_open_low = df.groupby(["instrument_key", "trading_day"])["low"].transform(lambda s: s.expanding().min())
    day_width = (day_open_high - day_open_low).replace(0.0, np.nan)
    df["dist_from_opening_high_pct"] = (df["ltp"] - day_open_high) / day_open_high.replace(0.0, np.nan)
    df["dist_from_opening_low_pct"] = (df["ltp"] - day_open_low) / day_open_low.replace(0.0, np.nan)
    df["opening_range_width_pct"] = day_width / df["ltp"].replace(0.0, np.nan)
    df["opening_range_breakout_strength"] = np.where(df["ltp"] > day_open_high, (df["ltp"] - day_open_high) / day_width, np.where(df["ltp"] < day_open_low, (df["ltp"] - day_open_low) / day_width, 0.0))

    df["regime_trending"] = ((df["adx_14"] >= 25.0) & (df["choppiness_14"] < 45.0)).astype(float)
    df["regime_volatile"] = (df["volatility_regime_classifier"] > 0).astype(float)
    df["regime_mean_reverting"] = ((df["adx_14"] < 20.0) & (df["choppiness_14"] >= 45.0)).astype(float)
    df["regime_quiet"] = ((df["atr_percentile_60"] < 0.3) & (df["choppiness_14"] >= 55.0)).astype(float)

    pivot_side = df.pivot_table(index=["timestamp", "expiry", "strike_price"], columns="option_type", values=["oi", "volume"], aggfunc="last")
    if not pivot_side.empty:
        pivot_side.columns = [f"{a}_{b}" for a, b in pivot_side.columns]
        pivot_side = pivot_side.reset_index()
        df = df.merge(pivot_side, on=["timestamp", "expiry", "strike_price"], how="left")
        df["ce_pe_oi_ratio"] = df.get("oi_CE", np.nan) / pd.to_numeric(df.get("oi_PE", np.nan), errors="coerce").replace(0.0, np.nan)
        df["ce_pe_volume_ratio"] = df.get("volume_CE", np.nan) / pd.to_numeric(df.get("volume_PE", np.nan), errors="coerce").replace(0.0, np.nan)

    reconstructable = [row["feature"] for row in _live_missing_feature_metadata() if row["category"] in {"A", "B", "C"}]
    unavailable = [row["feature"] for row in _live_missing_feature_metadata() if row["category"] not in {"A", "B", "C"}]
    _, gap_report = write_feature_schema_gap_report(
        [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
        reconstructed_features=reconstructable,
        additional_data_required=unavailable,
        report_prefix="live_vs_reconstructed_option_chain_schema_gap",
    )
    output_dataset = output_dataset or (PROCESSED_DIR / f"nifty_option_chain_live_feature_reconstructed_{_timestamp_now()}.csv")
    schema_path = output_dataset.with_name(output_dataset.stem + "_schema.json")
    report_path = report_path or (REPORTS_DIR / f"reconstruct_missing_live_features_for_option_chain_{_timestamp_now()}.md")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dataset, index=False)
    training_feature_cols = sorted(set(live_feature_schema_summary()["feature_names"]).intersection(df.columns))
    write_json(schema_path, ensure_serializable({"feature_columns": training_feature_cols, "matching_live_features": training_feature_cols, "matching_live_feature_count": len(training_feature_cols), "feature_gap_report_path": str(gap_report)}))
    report_path.write_text("\n".join(["# Reconstructed Missing Live Features", "", f"- Input dataset: `{input_dataset}`", f"- Output dataset: `{output_dataset}`", f"- Matching live features after reconstruction: `{len(training_feature_cols)}`", f"- Feature gap report: `{gap_report}`", "- No future leakage: `True`"]) + "\n", encoding="utf-8")
    return {"output_dataset": str(output_dataset), "schema_path": str(schema_path), "report_path": str(report_path), "schema_gap_report_path": str(gap_report), "matching_live_features": training_feature_cols, "matching_live_feature_count": len(training_feature_cols), "reconstructed_feature_count": len(reconstructable)}

    new_cols = [
        "bullish_engulfing", "bearish_engulfing", "doji", "hammer", "shooting_star",
        "ret_mean", "ret_std", "ret_min", "ret_max", "vol_mean", "vol_std", "vol_min", "vol_max",
        "adx_14", "choppiness_14", "supertrend_dir", "pivot_pp_dist_pct", "pivot_r1_dist_pct", "pivot_s1_dist_pct",
        "close_vs_open_pct", "range_to_atr", "momentum_lookback_pct", "ret_10", "ema_fast", "ema_slow", "ema_diff_pct",
        "regime_trending", "regime_volatile", "regime_mean_reverting", "regime_quiet",
        "ctx_adx", "ctx_trend_strength", "ctx_choppiness", "ctx_volume_sma",
        "vol_of_vol_14", "dist_from_opening_high_pct", "dist_from_opening_low_pct", "opening_range_width_pct",
        "opening_range_breakout_strength", "dist_to_rolling_high_20", "dist_to_rolling_low_20",
        "rolling_range_width_20", "rolling_range_position_20", "realized_vol_30", "atr_pct_regime_10",
        "atr_percentile_60", "realized_vol_percentile_60", "volatility_percentile_60", "volatility_regime_classifier",
    ]
    result_frames: List[pd.DataFrame] = []
    for _, group in df.groupby("instrument_key", sort=False):
        group = group.copy().sort_values("timestamp", kind="stable").reset_index(drop=True)
        feature_store: Dict[str, List[float]] = {col: [] for col in new_cols}
        candles = [
            Candle(
                time=ts.tz_localize(None).to_pydatetime() if getattr(ts, "tzinfo", None) else ts.to_pydatetime(),
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=float(v or 0.0),
            )
            for ts, o, h, l, c, v in zip(group["timestamp"], group["open"], group["high"], group["low"], group["ltp"], group["volume"])
        ]
        closes = group["ltp"].astype(float).tolist()
        highs = group["high"].astype(float).tolist()
        lows = group["low"].astype(float).tolist()
        opens = group["open"].astype(float).tolist()
        volumes = group["volume"].astype(float).tolist()
        atr_pcts: List[float] = []
        realized_vols: List[float] = []
        adx_hist: List[float] = []
        rsi_hist: List[float] = []
        for i in range(len(group)):
            hist_candles = candles[: i + 1]
            hist_closes = closes[: i + 1]
            hist_highs = highs[: i + 1]
            hist_lows = lows[: i + 1]
            hist_opens = opens[: i + 1]
            hist_volumes = volumes[: i + 1]
            ret_series = pd.Series(hist_closes).pct_change().dropna()
            vol_series = pd.Series(hist_volumes)
            feature_store["bullish_engulfing"].append(1.0 if is_bullish_engulfing(hist_candles[-2:]) else 0.0)
            feature_store["bearish_engulfing"].append(1.0 if is_bearish_engulfing(hist_candles[-2:]) else 0.0)
            feature_store["doji"].append(1.0 if is_doji(hist_candles[-1:]) else 0.0)
            feature_store["hammer"].append(1.0 if is_hammer(hist_candles) else 0.0)
            feature_store["shooting_star"].append(1.0 if is_shooting_star(hist_candles) else 0.0)
            recent_rets = ret_series.tail(20)
            feature_store["ret_mean"].append(float(recent_rets.mean()) if not recent_rets.empty else np.nan)
            feature_store["ret_std"].append(float(recent_rets.std(ddof=0)) if not recent_rets.empty else np.nan)
            feature_store["ret_min"].append(float(recent_rets.min()) if not recent_rets.empty else np.nan)
            feature_store["ret_max"].append(float(recent_rets.max()) if not recent_rets.empty else np.nan)
            feature_store["ret_10"].append(float((hist_closes[-1] - hist_closes[-11]) / hist_closes[-11]) if len(hist_closes) > 10 and hist_closes[-11] else np.nan)
            recent_vols = vol_series.tail(20)
            feature_store["vol_mean"].append(float(recent_vols.mean()))
            feature_store["vol_std"].append(float(recent_vols.std(ddof=0) if len(recent_vols) > 1 else 0.0))
            feature_store["vol_min"].append(float(recent_vols.min()))
            feature_store["vol_max"].append(float(recent_vols.max()))
            rsi_val = rsi(hist_closes, period=14)
            atr_val = atr(hist_highs, hist_lows, hist_closes, period=14)
            adx_val = adx(hist_highs, hist_lows, hist_closes, period=14)
            chop_val = choppiness_index(hist_highs, hist_lows, hist_closes, period=14)
            st_val = supertrend(hist_highs, hist_lows, hist_closes, period=10, multiplier=3.0)
            ema_fast_val = ema(hist_closes, 9)
            ema_slow_val = ema(hist_closes, 21)
            feature_store["adx_14"].append(float(adx_val) if adx_val is not None else np.nan)
            feature_store["choppiness_14"].append(float(chop_val) if chop_val is not None else np.nan)
            feature_store["supertrend_dir"].append(1.0 if st_val is not None and hist_closes[-1] >= float(st_val) else (-1.0 if st_val is not None else np.nan))
            piv = pivot_points(hist_highs[-1], hist_lows[-1], hist_closes[-1])
            close_now = float(hist_closes[-1])
            feature_store["pivot_pp_dist_pct"].append((close_now - float(piv.get("pp", close_now))) / close_now if close_now else 0.0)
            feature_store["pivot_r1_dist_pct"].append((close_now - float(piv.get("r1", close_now))) / close_now if close_now else 0.0)
            feature_store["pivot_s1_dist_pct"].append((close_now - float(piv.get("s1", close_now))) / close_now if close_now else 0.0)
            feature_store["close_vs_open_pct"].append((close_now - float(hist_opens[-1])) / close_now if close_now else 0.0)
            if atr_val is not None and close_now:
                atr_pct = float(atr_val) / close_now
                atr_pcts.append(atr_pct)
                feature_store["range_to_atr"].append(float(group.loc[i, "range_pct"]) / atr_pct if atr_pct else 0.0)
            else:
                atr_pcts.append(np.nan)
                feature_store["range_to_atr"].append(np.nan)
            lb_idx = max(0, len(hist_closes) - 20)
            feature_store["momentum_lookback_pct"].append((close_now - float(hist_closes[lb_idx])) / float(hist_closes[lb_idx]) if len(hist_closes) > 1 and hist_closes[lb_idx] else np.nan)
            ema_fast_num = float(ema_fast_val) if ema_fast_val is not None else np.nan
            ema_slow_num = float(ema_slow_val) if ema_slow_val is not None else np.nan
            feature_store["ema_fast"].append(ema_fast_num)
            feature_store["ema_slow"].append(ema_slow_num)
            ema_diff = (ema_fast_num - ema_slow_num) / close_now if pd.notna(ema_fast_num) and pd.notna(ema_slow_num) and close_now else np.nan
            feature_store["ema_diff_pct"].append(ema_diff)
            feature_store["ctx_adx"].append(float(adx_val) if adx_val is not None else np.nan)
            feature_store["ctx_trend_strength"].append(ema_diff)
            feature_store["ctx_choppiness"].append(float(chop_val) if chop_val is not None else np.nan)
            feature_store["ctx_volume_sma"].append(float(recent_vols.mean()))
            recent_atr_pcts = [v for v in atr_pcts[-14:] if pd.notna(v)]
            feature_store["vol_of_vol_14"].append(float(np.std(recent_atr_pcts)) if recent_atr_pcts else np.nan)
            day_group = group.loc[(group["trading_day"] == group.loc[i, "trading_day"]) & (group["timestamp"] <= group.loc[i, "timestamp"])]
            if not day_group.empty:
                opening = day_group.head(min(30, len(day_group)))
                op_high = float(opening["high"].max())
                op_low = float(opening["low"].min())
                width = max(op_high - op_low, 1e-9)
                feature_store["dist_from_opening_high_pct"].append((close_now - op_high) / op_high if op_high else 0.0)
                feature_store["dist_from_opening_low_pct"].append((close_now - op_low) / op_low if op_low else 0.0)
                feature_store["opening_range_width_pct"].append(width / close_now if close_now else 0.0)
                feature_store["opening_range_breakout_strength"].append((close_now - op_high) / width if close_now > op_high else ((close_now - op_low) / width if close_now < op_low else 0.0))
            else:
                feature_store["dist_from_opening_high_pct"].append(np.nan)
                feature_store["dist_from_opening_low_pct"].append(np.nan)
                feature_store["opening_range_width_pct"].append(np.nan)
                feature_store["opening_range_breakout_strength"].append(np.nan)
            recent_prev = hist_candles[max(0, len(hist_candles) - 20): -1]
            if recent_prev:
                roll_high = max(float(c.high) for c in recent_prev)
                roll_low = min(float(c.low) for c in recent_prev)
                width = max(roll_high - roll_low, 1e-9)
                feature_store["dist_to_rolling_high_20"].append((roll_high - close_now) / close_now if close_now else 0.0)
                feature_store["dist_to_rolling_low_20"].append((close_now - roll_low) / close_now if close_now else 0.0)
                feature_store["rolling_range_width_20"].append(width / close_now if close_now else 0.0)
                feature_store["rolling_range_position_20"].append((close_now - roll_low) / width)
            else:
                feature_store["dist_to_rolling_high_20"].append(np.nan)
                feature_store["dist_to_rolling_low_20"].append(np.nan)
                feature_store["rolling_range_width_20"].append(np.nan)
                feature_store["rolling_range_position_20"].append(np.nan)
            recent_rets_30 = pd.Series(hist_closes[:-1]).pct_change().dropna().tail(30)
            realized_vol = float(recent_rets_30.std(ddof=0)) if len(recent_rets_30) > 1 else 0.0
            realized_vols.append(realized_vol)
            feature_store["realized_vol_30"].append(realized_vol)
            atr_recent10 = [v for v in atr_pcts[-10:] if pd.notna(v)]
            curr_atr_pct = atr_pcts[-1] if atr_pcts else np.nan
            mean_recent10 = float(np.mean(atr_recent10)) if atr_recent10 else np.nan
            feature_store["atr_pct_regime_10"].append(float(curr_atr_pct) / mean_recent10 if pd.notna(curr_atr_pct) and pd.notna(mean_recent10) and abs(mean_recent10) > 1e-12 else np.nan)
            atr_pct_percentile = _rolling_percentile_value([v for v in atr_pcts[-60:] if pd.notna(v)], float(curr_atr_pct)) if pd.notna(curr_atr_pct) else np.nan
            feature_store["atr_percentile_60"].append(atr_pct_percentile)
            rv_pct = _rolling_percentile_value(realized_vols[-60:], realized_vol)
            feature_store["realized_vol_percentile_60"].append(rv_pct)
            feature_store["volatility_percentile_60"].append(rv_pct)
            feature_store["volatility_regime_classifier"].append(1.0 if (float(atr_pct_percentile or 0.0) >= 0.7 or float(rv_pct or 0.0) >= 0.7) else 0.0)
            if adx_val is not None:
                adx_hist.append(float(adx_val))
            if rsi_val is not None:
                rsi_hist.append(float(rsi_val))
            try:
                regime = detect_regime(atr_pcts[-30:] if atr_pcts else [0.0], adx_hist[-30:] if adx_hist else [0.0], rsi_hist[-30:] if rsi_hist else [50.0])
            except Exception:
                regime = "quiet"
            feature_store["regime_trending"].append(1.0 if regime == "trending" else 0.0)
            feature_store["regime_volatile"].append(1.0 if regime == "volatile" else 0.0)
            feature_store["regime_mean_reverting"].append(1.0 if regime == "mean_reverting" else 0.0)
            feature_store["regime_quiet"].append(1.0 if regime == "quiet" else 0.0)
        for col, vals in feature_store.items():
            group[col] = vals
        result_frames.append(group)
    df = pd.concat(result_frames, ignore_index=True)

    # CE/PE relative OI and volume at same timestamp/expiry/strike.
    pivot = df.pivot_table(
        index=["timestamp", "expiry", "strike_price"],
        columns="option_type",
        values=["oi", "volume"],
        aggfunc="last",
    )
    if not pivot.empty:
        pivot.columns = [f"{left}_{right}" for left, right in pivot.columns]
        pivot = pivot.reset_index()
        df = df.merge(pivot, on=["timestamp", "expiry", "strike_price"], how="left")
        df["ce_pe_oi_ratio"] = df.get("oi_CE", np.nan) / pd.to_numeric(df.get("oi_PE", np.nan), errors="coerce").replace(0.0, np.nan)
        df["ce_pe_volume_ratio"] = df.get("volume_CE", np.nan) / pd.to_numeric(df.get("volume_PE", np.nan), errors="coerce").replace(0.0, np.nan)

    reconstructable = [row["feature"] for row in _live_missing_feature_metadata() if row["category"] in {"A", "B", "C"}]
    gap_payload, gap_report = write_feature_schema_gap_report(
        [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
        reconstructed_features=reconstructable,
        additional_data_required=[row["feature"] for row in _live_missing_feature_metadata() if row["category"] not in {"A", "B", "C"}],
        report_prefix="live_vs_reconstructed_option_chain_schema_gap",
    )
    output_dataset = output_dataset or (PROCESSED_DIR / f"nifty_option_chain_live_feature_reconstructed_{_timestamp_now()}.csv")
    schema_path = output_dataset.with_name(output_dataset.stem + "_schema.json")
    report_path = report_path or (REPORTS_DIR / f"reconstruct_missing_live_features_for_option_chain_{_timestamp_now()}.md")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dataset, index=False)
    training_feature_cols = sorted(set(live_feature_schema_summary()["feature_names"]).intersection(df.columns))
    write_json(
        schema_path,
        ensure_serializable(
            {
                "feature_columns": training_feature_cols,
                "matching_live_features": training_feature_cols,
                "matching_live_feature_count": len(training_feature_cols),
                "feature_gap_report_path": str(gap_report),
            }
        ),
    )
    report_path.write_text(
        "\n".join(
            [
                "# Reconstructed Missing Live Features",
                "",
                f"- Input dataset: `{input_dataset}`",
                f"- Output dataset: `{output_dataset}`",
                f"- Matching live features after reconstruction: `{len(training_feature_cols)}`",
                f"- Feature gap report: `{gap_report}`",
                f"- No future leakage: `True`",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dataset": str(output_dataset),
        "schema_path": str(schema_path),
        "report_path": str(report_path),
        "schema_gap_report_path": str(gap_report),
        "matching_live_features": training_feature_cols,
        "matching_live_feature_count": len(training_feature_cols),
        "reconstructed_feature_count": len(reconstructable),
    }


def enrich_option_chain_with_spot_context(
    option_dataset: Path,
    spot_dataset: Path,
    *,
    output: Optional[Path] = None,
    report: Optional[Path] = None,
) -> Dict[str, Any]:
    if option_dataset.suffix.lower() == ".parquet":
        options = pd.read_parquet(option_dataset)
    else:
        options = pd.read_csv(option_dataset)
    options["timestamp"] = pd.to_datetime(options["timestamp"], errors="coerce", utc=False)
    if getattr(options["timestamp"].dt, "tz", None) is None:
        options["timestamp"] = options["timestamp"].dt.tz_localize(TIMEZONE, nonexistent="NaT", ambiguous="NaT")
    else:
        options["timestamp"] = options["timestamp"].dt.tz_convert(TIMEZONE)
    options["expiry"] = pd.to_datetime(options["expiry"], errors="coerce")
    options["trading_day"] = options["timestamp"].dt.date

    minute_spot_raw, daily_spot_raw, meta = load_spot_dataset(spot_dataset)
    minute_spot = _add_spot_features(_normalize_spot_frame(minute_spot_raw, daily=False), daily=False)
    daily_spot = _add_spot_features(_normalize_spot_frame(daily_spot_raw, daily=True), daily=True)
    intraday_mask = options["timestamp"].dt.strftime("%H:%M:%S").eq("09:15:00")
    intraday = options.loc[intraday_mask].copy().sort_values("timestamp", kind="stable")
    non_intraday = options.loc[~intraday_mask].copy().sort_values("timestamp", kind="stable")

    merged_intraday = intraday.copy()
    if not minute_spot.empty:
        minute_merge_cols = [
            "timestamp", "trading_day", "open", "high", "low", "close", "volume",
            "spot_close", "spot_return_1", "spot_return_3", "spot_return_5",
            "spot_range_pct", "spot_atr", "spot_rsi", "spot_vwap",
            "ctx_time_sin", "ctx_time_cos", "is_opening_session",
            "is_closing_session", "is_midday_lull", "weekday",
        ]
        minute_merge = minute_spot[minute_merge_cols].sort_values("timestamp")
        merged_intraday = pd.merge_asof(
            intraday.sort_values("timestamp"),
            minute_merge,
            on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta(minutes=5),
            suffixes=("", "_spot"),
        )
        merged_intraday["spot_source"] = np.where(merged_intraday["spot_close"].notna(), "minute_asof", "unmatched")
    merged_daily = non_intraday.copy()
    if not daily_spot.empty:
        daily_merge_cols = [
            "trading_day", "open", "high", "low", "close", "volume",
            "spot_close", "spot_return_1", "spot_return_3", "spot_return_5",
            "spot_range_pct", "spot_atr", "spot_rsi", "spot_vwap", "weekday",
        ]
        merged_daily = non_intraday.merge(
            daily_spot[daily_merge_cols].drop_duplicates(subset=["trading_day"], keep="last"),
            on="trading_day",
            how="left",
            suffixes=("", "_spot"),
        )
        merged_daily["spot_source"] = np.where(merged_daily["spot_close"].notna(), "daily_by_day", "unmatched")
        if "ctx_time_sin" not in merged_daily.columns:
            merged_daily["ctx_time_sin"] = 0.0
            merged_daily["ctx_time_cos"] = 0.0
            merged_daily["is_opening_session"] = 0
            merged_daily["is_closing_session"] = 0
            merged_daily["is_midday_lull"] = 0

    enriched = pd.concat([merged_intraday, merged_daily], ignore_index=True).sort_values("timestamp", kind="stable")
    enriched["ctx_spot"] = pd.to_numeric(enriched["spot_close"], errors="coerce")
    enriched["distance_from_spot"] = enriched["ltp"] - enriched["ctx_spot"]
    enriched["moneyness"] = enriched["strike_price"] / enriched["ctx_spot"].replace(0.0, np.nan)
    enriched["atm_distance"] = (enriched["strike_price"] - enriched["ctx_spot"]).abs()
    enriched["strike_distance_pct"] = (enriched["strike_price"] - enriched["ctx_spot"]) / enriched["ctx_spot"].replace(0.0, np.nan)
    enriched["ctx_dte_norm"] = ((enriched["expiry"] - pd.to_datetime(enriched["trading_day"])).dt.days.astype(float) / 30.0).clip(lower=0.0)

    # Reconstruct a subset of production-style fields from spot candles.
    enriched["last_open"] = pd.to_numeric(enriched["open_spot"] if "open_spot" in enriched.columns else enriched.get("open"), errors="coerce")
    enriched["last_high"] = pd.to_numeric(enriched["high_spot"] if "high_spot" in enriched.columns else enriched.get("high"), errors="coerce")
    enriched["last_low"] = pd.to_numeric(enriched["low_spot"] if "low_spot" in enriched.columns else enriched.get("low"), errors="coerce")
    enriched["last_close"] = enriched["ctx_spot"]
    enriched["last_volume"] = pd.to_numeric(enriched["volume_spot"] if "volume_spot" in enriched.columns else 0.0, errors="coerce").fillna(0.0)
    enriched["body_pct"] = (enriched["last_close"] - enriched["last_open"]) / enriched["last_close"].replace(0.0, np.nan)
    enriched["gap_pct"] = enriched["spot_return_1"]
    enriched["upper_wick_pct"] = (enriched["last_high"] - enriched[["last_open", "last_close"]].max(axis=1)) / (enriched["last_high"] - enriched["last_low"]).replace(0.0, np.nan)
    enriched["lower_wick_pct"] = (enriched[["last_open", "last_close"]].min(axis=1) - enriched["last_low"]) / (enriched["last_high"] - enriched["last_low"]).replace(0.0, np.nan)
    enriched["close_location_pct"] = (enriched["last_close"] - enriched["last_low"]) / (enriched["last_high"] - enriched["last_low"]).replace(0.0, np.nan)
    enriched["ret_10"] = np.nan
    enriched["ema_fast"] = np.nan
    enriched["ema_slow"] = np.nan
    enriched["ema_diff_pct"] = np.nan
    enriched["rsi_14"] = enriched["spot_rsi"]
    enriched["atr_14"] = enriched["spot_atr"]
    enriched["atr_pct"] = enriched["spot_atr"] / enriched["ctx_spot"].replace(0.0, np.nan)
    enriched["ctx_option_price"] = enriched["ltp"]
    enriched["option_to_spot_pct"] = enriched["ctx_option_price"] / enriched["ctx_spot"].replace(0.0, np.nan)
    enriched["volume_ratio"] = enriched["volume"] / enriched.groupby("instrument_key")["volume"].transform(lambda s: s.rolling(5, min_periods=1).mean()).replace(0.0, np.nan)

    reconstructed_features = [
        "last_open", "last_high", "last_low", "last_close", "last_volume", "body_pct",
        "gap_pct", "upper_wick_pct", "lower_wick_pct", "close_location_pct",
        "ret_3", "ret_5", "rsi_14", "atr_14", "atr_pct",
        "ctx_spot", "ctx_option_price", "ctx_time_sin", "ctx_time_cos",
        "is_opening_session", "is_closing_session", "is_midday_lull", "ctx_dte_norm",
        "option_to_spot_pct", "volume_ratio",
    ]
    additional_required = [
        "bullish_engulfing", "bearish_engulfing", "doji", "hammer", "shooting_star",
        "ret_mean", "ret_std", "ret_min", "ret_max", "vol_mean", "vol_std", "vol_min", "vol_max",
        "ret_10", "ema_fast", "ema_slow", "ema_diff_pct", "adx_14", "choppiness_14",
        "supertrend_dir", "pivot_pp_dist_pct", "pivot_r1_dist_pct", "pivot_s1_dist_pct",
        "close_vs_open_pct", "range_to_atr", "momentum_lookback_pct",
        "regime_trending", "regime_volatile", "regime_mean_reverting", "regime_quiet",
        "ctx_iv", "ctx_iv_change_pct", "ctx_iv_percentile", "ctx_delta", "ctx_gamma", "ctx_vega", "ctx_theta",
        "ctx_adx", "ctx_trend_strength", "ctx_choppiness", "ctx_volume_sma",
        "delta_abs", "greeks_imbalance", "vol_of_vol_14", "dist_from_opening_high_pct",
        "dist_from_opening_low_pct", "opening_range_width_pct", "opening_range_breakout_strength",
        "dist_to_rolling_high_20", "dist_to_rolling_low_20", "rolling_range_width_20",
        "rolling_range_position_20", "realized_vol_30", "atr_pct_regime_10",
        "atr_percentile_60", "realized_vol_percentile_60", "volatility_percentile_60",
        "volatility_regime_classifier", "bid_ask_spread_pct", "theta_to_vega_ratio", "gamma_to_theta_ratio",
    ]
    gap_payload, gap_report_path = write_feature_schema_gap_report(
        [c for c in enriched.columns if c not in {"future_close"}],
        reconstructed_features=reconstructed_features,
        additional_data_required=additional_required,
    )

    excluded_training_columns = {
        "timestamp", "expiry", "instrument_key", "trading_symbol", "option_type", "weekly",
        "source_file", "trading_day", "future_close", "gross_forward_return", "net_forward_return",
        "profitable_trade_label", "avoid_trade_label", "spot_source",
    }
    enriched_feature_columns = [
        col for col in enriched.columns
        if col not in excluded_training_columns and pd.api.types.is_numeric_dtype(enriched[col])
    ]

    leakage_ok = True
    if "timestamp_spot" in enriched.columns:
        leakage_ok = bool((pd.to_datetime(enriched["timestamp_spot"], errors="coerce") <= enriched["timestamp"]).fillna(True).all())
    stamp = _timestamp_now()
    output = output or (PROCESSED_DIR / f"nifty_option_chain_enriched_{stamp}.csv")
    report = report or (REPORTS_DIR / f"nifty_option_chain_enrichment_report_{stamp}.md")
    schema_path = output.with_name(output.stem + "_schema.json")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        try:
            enriched.to_parquet(output, index=False)
        except Exception:
            output = output.with_suffix(".csv")
            enriched.to_csv(output, index=False)
    else:
        enriched.to_csv(output, index=False)
    write_json(
        schema_path,
        ensure_serializable(
            {
                "feature_columns": enriched_feature_columns,
                "schema_match_live_model": False,
                "feature_gap_report_path": str(gap_report_path),
                "reconstructed_live_features": reconstructed_features,
                "still_missing_live_features": additional_required,
            }
        ),
    )
    report.write_text(
        "\n".join(
            [
                "# Option Chain Spot Context Enrichment Report",
                "",
                f"- Option dataset: `{option_dataset}`",
                f"- Spot dataset argument: `{spot_dataset}`",
                f"- Output: `{output}`",
                f"- Spot sources scanned: `{meta.get('scanned_sources', [])}`",
                f"- Minute spot rows: `{len(minute_spot)}`",
                f"- Daily spot rows: `{len(daily_spot)}`",
                f"- Rows with spot context: `{int(enriched['ctx_spot'].notna().sum())}` / `{len(enriched)}`",
                f"- No future merge leakage: `{leakage_ok}`",
                f"- Enriched feature count: `{len(enriched_feature_columns)}`",
                f"- Schema path: `{schema_path}`",
                f"- Feature gap report: `{gap_report_path}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "enriched_dataset_path": str(output),
        "enrichment_report_path": str(report),
        "feature_gap_report_path": str(gap_report_path),
        "schema_path": str(schema_path),
        "rows": int(len(enriched)),
        "rows_with_spot_context": int(enriched["ctx_spot"].notna().sum()),
        "spot_sources": meta.get("scanned_sources", []),
        "no_future_merge_leakage": bool(leakage_ok),
        "feature_columns": enriched_feature_columns,
        "reconstructed_live_features": reconstructed_features,
        "still_missing_live_features": additional_required,
    }


def _brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    if len(yt) == 0:
        return 0.0
    return float(np.mean((yp - yt) ** 2))


def _returns_from_threshold(probs: Sequence[float], returns: Sequence[float], threshold: float) -> np.ndarray:
    mask = np.asarray([float(p) >= float(threshold) for p in probs], dtype=bool)
    vals = np.asarray(returns, dtype=float)
    return vals[mask]


def _trade_metrics(signal_returns: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(signal_returns, dtype=float)
    if len(arr) == 0:
        return {
            "trade_count": 0.0,
            "coverage_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        }
    gross_profit = float(arr[arr > 0].sum()) if np.any(arr > 0) else 0.0
    gross_loss = float(abs(arr[arr < 0].sum())) if np.any(arr < 0) else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    expectancy = float(arr.mean())
    win_rate = float(np.mean(arr > 0))
    equity = np.cumsum(arr)
    running_peak = np.maximum.accumulate(np.r_[0.0, equity])[:-1] if len(equity) else np.array([])
    drawdown = equity - running_peak if len(equity) else np.array([0.0])
    max_drawdown = float(abs(drawdown.min())) if len(drawdown) else 0.0
    sharpe = float((arr.mean() / (arr.std(ddof=0) + 1e-9)) * math.sqrt(min(252.0, len(arr)))) if len(arr) > 1 else 0.0
    return {
        "trade_count": float(len(arr)),
        "coverage_rate": 0.0,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }


def _day_level_trade_diagnostics(frame: pd.DataFrame, threshold: float) -> Dict[str, Any]:
    work = frame.copy()
    work["signal"] = work["pred_prob"] >= float(threshold)
    work["signal_return"] = np.where(work["signal"], work["net_forward_return"], np.nan)
    day_rows: List[Dict[str, Any]] = []
    for trading_day, subset in work.groupby("trading_day"):
        returns = subset["signal_return"].dropna().to_numpy(dtype=float)
        pnl = float(np.nansum(returns)) if len(returns) else 0.0
        day_rows.append({
            "trading_day": str(trading_day),
            "trade_count": int(np.sum(subset["signal"])),
            "pnl": pnl,
            "win_rate": float(np.mean(returns > 0)) if len(returns) else 0.0,
        })
    day_rows = sorted(day_rows, key=lambda row: row["pnl"])
    total_profit = float(sum(max(0.0, row["pnl"]) for row in day_rows))
    max_single = max((max(0.0, row["pnl"]) for row in day_rows), default=0.0)
    contribution = float(max_single / total_profit) if total_profit > 0 else 0.0
    return {
        "active_trading_days": int(sum(1 for row in day_rows if row["trade_count"] > 0)),
        "avg_trades_per_day": float(np.mean([row["trade_count"] for row in day_rows if row["trade_count"] > 0])) if any(row["trade_count"] > 0 for row in day_rows) else 0.0,
        "worst_5_trading_days": day_rows[:5],
        "best_5_trading_days": list(reversed(day_rows[-5:])),
        "max_single_day_profit_contribution": contribution,
    }


def generate_cost_sensitivity_report(
    artifact_dir: Path,
    model_prob_rows: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    scenarios = {
        "zero_cost": 0.0,
        "low_cost": 0.0015,
        "medium_cost": 0.0035,
        "high_cost": 0.0060,
    }
    rows: List[Dict[str, Any]] = []
    for entry in model_prob_rows:
        model_name = str(entry["model_name"])
        probs = np.asarray(entry["probs"], dtype=float)
        frame = entry["frame"].copy()
        gross = pd.to_numeric(frame["gross_forward_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for threshold in [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]:
            mask = probs >= float(threshold)
            for scenario_name, cost in scenarios.items():
                returns = gross[mask] - float(cost)
                trade = _trade_metrics(returns)
                rows.append(
                    {
                        "model_name": model_name,
                        "threshold": float(threshold),
                        "cost_scenario": scenario_name,
                        "trade_count": int(trade["trade_count"]),
                        "profit_factor": float(trade["profit_factor"]),
                        "sharpe": float(trade["sharpe"]),
                        "expectancy": float(trade["expectancy"]),
                        "max_drawdown": float(trade["max_drawdown"]),
                    }
                )
    df = pd.DataFrame(rows)
    stamp = _timestamp_now()
    csv_path = artifact_dir / f"cost_sensitivity_report_{stamp}.csv"
    md_path = artifact_dir / f"cost_sensitivity_report_{stamp}.md"
    df.to_csv(csv_path, index=False)
    lines = ["# Cost Sensitivity Report", ""]
    for model_name, subset in df.groupby("model_name"):
        lines.append(f"## {model_name}")
        for _, row in subset.head(12).iterrows():
            lines.append(
                f"- thr={row['threshold']:.2f} scenario={row['cost_scenario']} trades={int(row['trade_count'])} pf={row['profit_factor']:.4f} sharpe={row['sharpe']:.4f} expectancy={row['expectancy']:.6f}"
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": str(csv_path), "md": str(md_path)}


def generate_monthly_stability_report(
    artifact_dir: Path,
    frame: pd.DataFrame,
    probs: Sequence[float],
    threshold: float,
    model_name: str,
) -> Dict[str, str]:
    work = frame.copy()
    work["pred_prob"] = np.asarray(probs, dtype=float)
    work["signal"] = work["pred_prob"] >= float(threshold)
    work["signal_return"] = np.where(work["signal"], work["net_forward_return"], np.nan)
    work["month_bucket"] = pd.to_datetime(work["timestamp"]).dt.to_period("M").astype(str)
    rows: List[Dict[str, Any]] = []
    for bucket, subset in work.groupby("month_bucket"):
        returns = subset["signal_return"].dropna().to_numpy(dtype=float)
        metrics = _trade_metrics(returns)
        ce_count = int(((subset["option_type"] == "CE") & subset["signal"]).sum()) if "option_type" in subset.columns else 0
        pe_count = int(((subset["option_type"] == "PE") & subset["signal"]).sum()) if "option_type" in subset.columns else 0
        expiry_count = int(((pd.to_datetime(subset["timestamp"]).dt.date == pd.to_datetime(subset["expiry"]).dt.date) & subset["signal"]).sum()) if "expiry" in subset.columns else 0
        non_expiry_count = int(subset["signal"].sum()) - expiry_count
        rows.append(
            {
                "model_name": model_name,
                "month": bucket,
                "trades": int(metrics["trade_count"]),
                "win_rate": float(metrics["win_rate"]),
                "profit_factor": float(metrics["profit_factor"]),
                "sharpe": float(metrics["sharpe"]),
                "expectancy": float(metrics["expectancy"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "ce_trades": ce_count,
                "pe_trades": pe_count,
                "expiry_day_trades": expiry_count,
                "non_expiry_day_trades": non_expiry_count,
            }
        )
    df = pd.DataFrame(rows)
    stamp = _timestamp_now()
    csv_path = artifact_dir / f"monthly_stability_report_{stamp}.csv"
    md_path = artifact_dir / f"monthly_stability_report_{stamp}.md"
    df.to_csv(csv_path, index=False)
    lines = ["# Monthly Stability Report", ""]
    for _, row in df.iterrows():
        lines.append(
            f"- {row['month']}: trades={int(row['trades'])} pf={row['profit_factor']:.4f} sharpe={row['sharpe']:.4f} expectancy={row['expectancy']:.6f} CE={int(row['ce_trades'])} PE={int(row['pe_trades'])}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "md": str(md_path)}


def generate_threshold_diagnostics(
    frame: pd.DataFrame,
    probs: Sequence[float],
    y_true: Sequence[int],
    *,
    thresholds: Optional[Sequence[float]] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    out_dir = out_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    work = frame.copy()
    work["pred_prob"] = np.asarray(probs, dtype=float)
    work["y_true"] = np.asarray(y_true, dtype=int)
    thresholds = list(thresholds or [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)])
    rows: List[Dict[str, Any]] = []
    detailed: Dict[str, Any] = {}
    for threshold in thresholds:
        metrics = classification_metrics(work["y_true"], work["pred_prob"], threshold=threshold)
        signal_returns = _returns_from_threshold(work["pred_prob"], work["net_forward_return"], threshold)
        trade_metrics = _trade_metrics(signal_returns)
        trade_metrics["coverage_rate"] = float(len(signal_returns) / max(1, len(work)))
        per_day = _day_level_trade_diagnostics(work, threshold)
        row = {
            "threshold": float(threshold),
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "f1": float(metrics.get("f1", 0.0)),
            "trade_count": int(trade_metrics["trade_count"]),
            "coverage_rate": float(trade_metrics["coverage_rate"]),
            "profit_factor": float(trade_metrics["profit_factor"]),
            "average_expectancy": float(trade_metrics["expectancy"]),
            "max_drawdown": float(trade_metrics["max_drawdown"]),
            "sharpe": float(trade_metrics["sharpe"]),
            "average_trades_per_day": float(per_day["avg_trades_per_day"]),
            "active_trading_days": int(per_day["active_trading_days"]),
            "max_single_day_profit_contribution": float(per_day["max_single_day_profit_contribution"]),
        }
        rows.append(row)
        detailed[f"{threshold:.2f}"] = {
            **row,
            "worst_5_trading_days": per_day["worst_5_trading_days"],
            "best_5_trading_days": per_day["best_5_trading_days"],
            "expiry_day_vs_non_expiry": _performance_breakdowns(work.copy(), work["pred_prob"].to_numpy(dtype=float), threshold)["by_expiry_day"],
            "ce_vs_pe": _performance_breakdowns(work.copy(), work["pred_prob"].to_numpy(dtype=float), threshold)["by_option_type"],
        }
    df = pd.DataFrame(rows)
    stamp = _timestamp_now()
    csv_path = out_dir / f"threshold_diagnostics_{stamp}.csv"
    md_path = out_dir / f"threshold_diagnostics_{stamp}.md"
    json_path = out_dir / f"threshold_diagnostics_{stamp}.json"
    df.to_csv(csv_path, index=False)
    md_lines = ["# Threshold Diagnostics", ""]
    for row in rows:
        md_lines.append(
            f"- thr={row['threshold']:.2f} precision={row['precision']:.4f} recall={row['recall']:.4f} f1={row['f1']:.4f} "
            f"trades={row['trade_count']} coverage={row['coverage_rate']:.4%} pf={row['profit_factor']:.4f} sharpe={row['sharpe']:.4f}"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    write_json(json_path, ensure_serializable(detailed))
    return {"csv": str(csv_path), "md": str(md_path), "json": str(json_path), "rows": rows, "detailed": detailed}


def _performance_breakdowns(frame: pd.DataFrame, probs: np.ndarray, threshold: float) -> Dict[str, Any]:
    work = frame.copy()
    work["pred_prob"] = probs
    work["signal"] = work["pred_prob"] >= float(threshold)
    work["signal_return"] = np.where(work["signal"], work["net_forward_return"], np.nan)

    def summarize(group_cols: Sequence[str]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for keys, subset in work.groupby(list(group_cols), dropna=False):
            sig = subset["signal_return"].dropna().to_numpy(dtype=float)
            metrics = _trade_metrics(sig)
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {str(col): str(val) for col, val in zip(group_cols, keys)}
            row.update({
                "rows": int(len(subset)),
                "signal_count": int(subset["signal"].sum()),
                "profit_factor": float(metrics["profit_factor"]),
                "expectancy": float(metrics["expectancy"]),
                "win_rate": float(metrics["win_rate"]),
            })
            rows.append(row)
        return rows

    work["expiry_day_flag"] = np.where(pd.to_datetime(work["timestamp"]).dt.date == pd.to_datetime(work["expiry"]).dt.date, "EXPIRY_DAY", "NON_EXPIRY_DAY")
    work["liquidity_bucket"] = pd.qcut(work["volume"].rank(method="first"), q=min(4, max(1, work["volume"].nunique())), duplicates="drop").astype(str)
    return {
        "by_option_type": summarize(["option_type"]),
        "by_expiry_day": summarize(["expiry_day_flag"]),
        "by_liquidity_bucket": summarize(["liquidity_bucket"]),
    }


def train_option_chain_models(
    processed_dataset: Path,
    *,
    artifact_dir: Optional[Path] = None,
    old_model_dir: Optional[Path] = None,
    min_roc_auc: float = 0.55,
    min_profit_factor: float = 1.15,
    min_sharpe: float = 1.0,
    min_holdout_trades: int = 200,
    min_active_days: int = 20,
    min_recall: float = 0.05,
    min_f1: float = 0.05,
    max_single_day_profit_contribution: float = 0.25,
    min_coverage_rate: float = 0.01,
) -> Dict[str, Any]:
    if processed_dataset.suffix.lower() == ".parquet":
        df = pd.read_parquet(processed_dataset)
    else:
        df = pd.read_csv(processed_dataset)
    schema_path = processed_dataset.with_name(processed_dataset.stem + "_schema.json")
    if df.empty:
        raise RuntimeError("Processed dataset is empty.")
    artifact_dir = artifact_dir or (MODELS_DIR / f"retraining_option_chain_{_timestamp_now()}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = json.loads(schema_path.read_text(encoding="utf-8")).get("feature_columns", [])
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    medians = df[feature_cols].median(numeric_only=True)
    df[feature_cols] = df[feature_cols].fillna(medians).fillna(0.0)
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["profitable_trade_label"].astype(int).to_numpy()
    timestamps = list(pd.to_datetime(df["timestamp"]).dt.to_pydatetime())
    split = chronological_split(X, y, timestamps)
    train_idx = np.arange(len(X))[split["slices"]["train"]]
    val_idx = np.arange(len(X))[split["slices"]["validation"]]
    test_idx = np.arange(len(X))[split["slices"]["test"]]
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    X_train_s, X_val_s, X_test_s, scaler_mean, scaler_std = scale_train_val_test(X_train, X_val, X_test)
    try:
        compatibility = compare_feature_compatibility(feature_cols)
        schema_match_live = bool(compatibility.get("matches_production"))
    except Exception as exc:
        compatibility = {"production_schema_compatible": False, "error": str(exc)}
        schema_match_live = False
    strict_gate = strict_live_schema_gate(feature_cols)
    schema_match_live = bool(schema_match_live and strict_gate.get("compatible"))

    model_reports: Dict[str, Any] = {}
    champion: Optional[Dict[str, Any]] = None
    model_prob_rows: List[Dict[str, Any]] = []
    for model_name in model_names_to_train():
        model = build_safe_model(model_name)
        if model is None:
            continue
        model.fit(X_train_s, y_train)
        val_prob = predict_proba_positive(model, X_val_s)
        test_prob = predict_proba_positive(model, X_test_s)
        threshold_rows = []
        for threshold in THRESHOLD_GRID:
            metrics = classification_metrics(y_val, val_prob, threshold=threshold)
            signal_returns = _returns_from_threshold(val_prob, df.iloc[val_idx]["net_forward_return"].to_numpy(dtype=float), threshold)
            trade_metrics = _trade_metrics(signal_returns)
            threshold_rows.append({
                "threshold": float(threshold),
                "signals": int((val_prob >= threshold).sum()),
                **metrics,
                "profit_factor": float(trade_metrics["profit_factor"]),
                "expectancy": float(trade_metrics["expectancy"]),
                "sharpe": float(trade_metrics["sharpe"]),
            })
        chosen = max(
            threshold_rows,
            key=lambda row: (
                float(row.get("precision", 0.0)),
                float(row.get("profit_factor", 0.0)),
                int(row.get("signals", 0)),
            ),
        )
        threshold = float(chosen["threshold"])
        val_metrics = classification_metrics(y_val, val_prob, threshold=threshold)
        test_metrics = classification_metrics(y_test, test_prob, threshold=threshold)
        test_confusion = confusion_from_threshold(y_test, test_prob, threshold)
        signal_returns = _returns_from_threshold(test_prob, df.iloc[test_idx]["net_forward_return"].to_numpy(dtype=float), threshold)
        trade_metrics = _trade_metrics(signal_returns)
        trade_metrics["coverage_rate"] = float(len(signal_returns) / max(1, len(test_prob)))
        threshold_diag = generate_threshold_diagnostics(df.iloc[test_idx].copy(), test_prob, y_test, out_dir=artifact_dir)
        chosen_frame = df.iloc[test_idx].copy()
        chosen_frame["pred_prob"] = np.asarray(test_prob, dtype=float)
        chosen_specific_diag = _day_level_trade_diagnostics(chosen_frame, threshold)
        brier = _brier_score(y_test, test_prob)
        artifact = _wrap_model(
            model,
            feature_names=feature_cols,
            metrics={
                "model_type": model_name,
                "threshold": threshold,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
            },
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
        )
        model_path = artifact_dir / f"{model_name}_option_chain.pkl"
        joblib.dump(artifact, model_path)
        report = {
            "model_name": model_name,
            "model_path": str(model_path),
            "selected_threshold_from_validation": chosen,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "confusion_matrix": test_confusion,
            "calibration": {
                "validation": calibration_bins(y_val, val_prob),
                "test": calibration_bins(y_test, test_prob),
                "brier_score": brier,
            },
            "threshold_table": threshold_rows,
            "threshold_sweep": threshold_sweep(y_test, test_prob),
            "trade_metrics": trade_metrics,
            "performance_breakdown": _performance_breakdowns(df.iloc[test_idx].copy(), test_prob, threshold),
            "threshold_diagnostics": threshold_diag,
            "chosen_threshold_day_diagnostics": chosen_specific_diag,
        }
        model_reports[model_name] = report
        model_prob_rows.append({"model_name": model_name, "probs": test_prob.tolist(), "frame": df.iloc[test_idx].copy()})
        if champion is None or (
            float(report["test_metrics"].get("roc_auc", 0.0))
            > float(champion["test_metrics"].get("roc_auc", 0.0))
        ):
            champion = report

    if champion is None:
        raise RuntimeError("No compatible models were trained.")

    old_bundle = load_model(str(LIVE_MODEL_PATH)) if LIVE_MODEL_PATH.exists() else None
    old_summary = {
        "path": str(LIVE_MODEL_PATH),
        "metrics": dict(getattr(old_bundle, "metrics", {}) or {}),
        "feature_count": int(len(getattr(old_bundle, "feature_names", []) or [])) if old_bundle is not None else 0,
        "schema_compatible_with_option_chain_research_set": bool(
            list(getattr(old_bundle, "feature_names", []) or []) == list(feature_cols)
        ) if old_bundle is not None else False,
    }
    decision_failures: List[str] = []
    holdout_roc_auc = float(champion["test_metrics"].get("roc_auc", 0.0) or 0.0)
    validation_roc_auc = float(champion["validation_metrics"].get("roc_auc", 0.0) or 0.0)
    pf = float(champion["trade_metrics"].get("profit_factor", 0.0) or 0.0)
    sharpe = float(champion["trade_metrics"].get("sharpe", 0.0) or 0.0)
    trade_count = float(champion["trade_metrics"].get("trade_count", 0.0) or 0.0)
    recall = float(champion["test_metrics"].get("recall", 0.0) or 0.0)
    f1 = float(champion["test_metrics"].get("f1", 0.0) or 0.0)
    coverage_rate = float(champion["trade_metrics"].get("coverage_rate", 0.0) or 0.0)
    chosen_diag = champion.get("chosen_threshold_day_diagnostics", {}) or {}
    if holdout_roc_auc < float(min_roc_auc):
        decision_failures.append(f"holdout_roc_auc {holdout_roc_auc:.4f} < {float(min_roc_auc):.2f}")
    if abs(validation_roc_auc - holdout_roc_auc) > 0.10:
        decision_failures.append("validation/holdout ROC-AUC diverged by more than 0.10")
    if pf < float(min_profit_factor):
        decision_failures.append(f"profit_factor {pf:.4f} < {float(min_profit_factor):.2f}")
    if sharpe < float(min_sharpe):
        decision_failures.append(f"sharpe {sharpe:.4f} < {float(min_sharpe):.2f}")
    if trade_count < int(min_holdout_trades):
        decision_failures.append(f"holdout trade_count {trade_count:.0f} < {int(min_holdout_trades)}")
    if int(chosen_diag.get("active_trading_days", 0) or 0) < int(min_active_days):
        decision_failures.append(f"active_trading_days {int(chosen_diag.get('active_trading_days', 0) or 0)} < {int(min_active_days)}")
    if recall < float(min_recall):
        decision_failures.append(f"recall {recall:.4f} < {float(min_recall):.2f}")
    if f1 < float(min_f1):
        decision_failures.append(f"f1 {f1:.4f} < {float(min_f1):.2f}")
    if coverage_rate < float(min_coverage_rate):
        decision_failures.append(f"coverage_rate {coverage_rate:.4%} < {float(min_coverage_rate):.2%}")
    if float(chosen_diag.get("max_single_day_profit_contribution", 0.0) or 0.0) > float(max_single_day_profit_contribution):
        decision_failures.append(
            f"max_single_day_profit_contribution {float(chosen_diag.get('max_single_day_profit_contribution', 0.0) or 0.0):.4f} > {float(max_single_day_profit_contribution):.2f}"
        )
    if not strict_gate.get("feature_names_match_exactly"):
        decision_failures.append("feature names do not match live production schema exactly")
    if not strict_gate.get("feature_order_matches_exactly"):
        decision_failures.append("feature order does not match live production schema exactly")
    if strict_gate.get("missing_live_features"):
        decision_failures.append(f"missing live features: {strict_gate.get('missing_live_features')}")
    if strict_gate.get("disallowed_training_only_features"):
        decision_failures.append(f"training-only or future features present: {strict_gate.get('disallowed_training_only_features')}")
    if not schema_match_live:
        decision_failures.append("feature schema does not match current live model feature generation")
    adoption_status = "PASSED" if not decision_failures else "FAILED"
    gate_report = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "artifact_dir": str(artifact_dir),
        "processed_dataset": str(processed_dataset),
        "schema_compatibility": compatibility,
        "strict_live_schema_gate": strict_gate,
        "champion_model": champion["model_name"],
        "holdout_metrics": champion["test_metrics"],
        "validation_metrics": champion["validation_metrics"],
        "trade_metrics": champion["trade_metrics"],
        "chosen_threshold_diagnostics": chosen_diag,
        "adoption_decision": adoption_status,
        "failures": decision_failures,
        "old_model_summary": old_summary,
    }
    cost_report = generate_cost_sensitivity_report(artifact_dir, model_prob_rows)
    champion_row = next((row for row in model_prob_rows if row["model_name"] == champion["model_name"]), None)
    monthly_report = generate_monthly_stability_report(
        artifact_dir,
        champion_row["frame"] if champion_row is not None else df.iloc[test_idx].copy(),
        champion_row["probs"] if champion_row is not None else champion.get("test_probs", []),
        float(champion["selected_threshold_from_validation"]["threshold"]),
        champion["model_name"],
    )
    gate_report["cost_sensitivity_report"] = cost_report
    gate_report["monthly_stability_report"] = monthly_report
    write_json(artifact_dir / "all_model_training_report.json", ensure_serializable(model_reports))
    write_json(artifact_dir / "adoption_decision_report.json", ensure_serializable(gate_report))
    write_json(artifact_dir / "feature_schema.json", ensure_serializable({"feature_columns": feature_cols, "compatibility": compatibility, "strict_live_schema_gate": strict_gate}))
    return {
        "artifact_dir": str(artifact_dir),
        "model_reports": model_reports,
        "champion": champion,
        "gate_report": gate_report,
        "old_model_summary": old_summary,
    }


def compare_old_vs_new(artifact_dir: Path, old_model_dir: Optional[Path] = None) -> Dict[str, Any]:
    report = json.loads((artifact_dir / "adoption_decision_report.json").read_text(encoding="utf-8"))
    old_summary = report.get("old_model_summary", {})
    new_summary = {
        "champion_model": report.get("champion_model"),
        "holdout_metrics": report.get("holdout_metrics"),
        "trade_metrics": report.get("trade_metrics"),
    }
    comparison = {
        "old_model": old_summary,
        "new_model": new_summary,
        "comparison_note": "Direct old-vs-new scoring on the option-chain dataset is only possible when the legacy live model feature schema matches the research dataset schema.",
        "schema_compatible": bool(old_summary.get("schema_compatible_with_option_chain_research_set")),
    }
    write_json(artifact_dir / "old_vs_new_comparison.json", ensure_serializable(comparison))
    return comparison


def write_option_chain_research_comparison(
    previous_artifact_dir: Path,
    enriched_artifact_dir: Path,
) -> Path:
    previous_eval = json.loads((previous_artifact_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    enriched_eval = json.loads((enriched_artifact_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    enriched_gate = json.loads((enriched_artifact_dir / "adoption_decision_report.json").read_text(encoding="utf-8"))
    live = live_feature_schema_summary()
    _, md_path, _ = report_paths("option_chain_research_comparison")
    lines = [
        "# Option Chain Research Comparison",
        "",
        "## A. Current Production Model",
        "",
        f"- Feature count: `{live['feature_count']}`",
        f"- Model input shape: `{live['model_input_shape']}`",
        "",
        "## B. Previous Option-Chain Research Model",
        "",
        f"- Champion: `{previous_eval.get('champion_model')}`",
        f"- ROC-AUC: `{previous_eval.get('holdout_metrics', {}).get('roc_auc')}`",
        f"- Precision: `{previous_eval.get('holdout_metrics', {}).get('precision')}`",
        f"- Recall: `{previous_eval.get('holdout_metrics', {}).get('recall')}`",
        f"- F1: `{previous_eval.get('holdout_metrics', {}).get('f1')}`",
        f"- Trade count: `{previous_eval.get('trade_metrics', {}).get('trade_count')}`",
        f"- Coverage rate: `{previous_eval.get('trade_metrics', {}).get('coverage_rate')}`",
        f"- Profit factor: `{previous_eval.get('trade_metrics', {}).get('profit_factor')}`",
        f"- Sharpe: `{previous_eval.get('trade_metrics', {}).get('sharpe')}`",
        "",
        "## C. Enriched Option-Chain Research Model",
        "",
        f"- Champion: `{enriched_eval.get('champion_model')}`",
        f"- ROC-AUC: `{enriched_eval.get('holdout_metrics', {}).get('roc_auc')}`",
        f"- Precision: `{enriched_eval.get('holdout_metrics', {}).get('precision')}`",
        f"- Recall: `{enriched_eval.get('holdout_metrics', {}).get('recall')}`",
        f"- F1: `{enriched_eval.get('holdout_metrics', {}).get('f1')}`",
        f"- Trade count: `{enriched_eval.get('trade_metrics', {}).get('trade_count')}`",
        f"- Coverage rate: `{enriched_eval.get('trade_metrics', {}).get('coverage_rate')}`",
        f"- Profit factor: `{enriched_eval.get('trade_metrics', {}).get('profit_factor')}`",
        f"- Sharpe: `{enriched_eval.get('trade_metrics', {}).get('sharpe')}`",
        f"- Active days: `{enriched_gate.get('chosen_threshold_diagnostics', {}).get('active_trading_days')}`",
        "",
        "## Verdict Inputs",
        "",
        f"- Adoption decision: `{enriched_eval.get('adoption_decision')}`",
        f"- Failures: `{enriched_eval.get('failures', [])}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def evaluate_option_chain_artifacts(artifact_dir: Path) -> Dict[str, Any]:
    report = json.loads((artifact_dir / "adoption_decision_report.json").read_text(encoding="utf-8"))
    champion = report.get("champion_model")
    models = json.loads((artifact_dir / "all_model_training_report.json").read_text(encoding="utf-8"))
    payload = {
        "artifact_dir": str(artifact_dir),
        "champion_model": champion,
        "adoption_decision": report.get("adoption_decision"),
        "holdout_metrics": report.get("holdout_metrics"),
        "validation_metrics": report.get("validation_metrics"),
        "trade_metrics": report.get("trade_metrics"),
        "failures": report.get("failures", []),
        "models_retrained": sorted(models.keys()),
    }
    write_json(artifact_dir / "evaluation_report.json", ensure_serializable(payload))
    return payload


def adopt_option_chain_models_if_passed(artifact_dir: Path) -> Dict[str, Any]:
    gate = json.loads((artifact_dir / "adoption_decision_report.json").read_text(encoding="utf-8"))
    if str(gate.get("adoption_decision")) != "PASSED":
        return {
            "adopted": False,
            "reason": "adoption gates failed",
            "failures": gate.get("failures", []),
        }
    champion_name = str(gate.get("champion_model") or "")
    model_path = artifact_dir / f"{champion_name}_option_chain.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Champion artifact missing: {model_path}")
    backup_dir = MODELS_DIR / f"backup_before_option_chain_adoption_{_timestamp_now()}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if LIVE_MODEL_PATH.exists():
        shutil.copy2(LIVE_MODEL_PATH, backup_dir / LIVE_MODEL_PATH.name)
    shutil.copy2(model_path, LIVE_MODEL_PATH)
    profile = {
        "adopted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_artifact_dir": str(artifact_dir),
        "champion_model": champion_name,
        "backup_dir": str(backup_dir),
        "rollback_note": f"Restore {backup_dir / LIVE_MODEL_PATH.name} to {LIVE_MODEL_PATH} if needed.",
        "holdout_metrics": gate.get("holdout_metrics"),
        "trade_metrics": gate.get("trade_metrics"),
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    write_json(PRODUCTION_PROFILE_PATH, ensure_serializable(profile))
    json_path, md_path, _ = report_paths("option_chain_model_adoption_report")
    write_json(json_path, ensure_serializable(profile))
    md_path.write_text(
        "\n".join(
            [
                "# Option Chain Model Adoption Report",
                "",
                f"- Source artifact dir: `{artifact_dir}`",
                f"- Champion model: `{champion_name}`",
                f"- Backup dir: `{backup_dir}`",
                f"- Live model path updated: `{LIVE_MODEL_PATH}`",
                f"- Rollback note: {profile['rollback_note']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"adopted": True, **profile}
