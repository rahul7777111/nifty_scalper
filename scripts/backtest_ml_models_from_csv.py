#!/usr/bin/env python
"""
scripts/backtest_ml_models_from_csv.py

Offline historical ML-driven backtester for NIFTY options chain CSV data.

- Tolerant column mapping for common option data fields (timestamp, ltp/close, strike, option_type, trading_symbol, etc.)
- Loads ML candidate config (paper_forward_candidates.json style) if provided; loads model bundles and scores rows.
- Falls back to pre-existing model_score / probability / proba columns in the CSV if no model/config.
- Applies simple filters (spread, premium, DTE proxy via expiry if parsable).
- Simulates entries on score >= threshold (with daily cap).
- Exits: target_pct, stoploss_pct, max_hold_bars, EOD (date change).
- Reuses scripts/ml_execution_costs.py for net cost adjustment when available.
- Fully offline: no broker, no websocket, no live state mutation.
- Thread-safe friendly: accepts optional stop_event (threading.Event), log_fn(str), and progress_fn(fraction, message) callbacks.
- Writes:
    backtest_trades_<ts>.csv
    backtest_summary_<ts>.json
    backtest_report_<ts>.md
- Returns summary dict + paths for GUI integration.

Intended to be called from GUI (Backtest Runner tab) and CLI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import sys
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup so we can import repo modules when run as script or from GUI
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
_SRC = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Reusable pieces (best effort; fallbacks provided)
try:
    from ml_signals import load_model, predict as ml_predict  # type: ignore
    HAS_ML_SIGNALS = True
except Exception:
    HAS_ML_SIGNALS = False
    load_model = None  # type: ignore
    ml_predict = None  # type: ignore

try:
    from scripts.ml_execution_costs import (  # type: ignore
        estimate_option_execution_cost,
        estimate_option_execution_costs_frame,
    )
    HAS_COST_MODEL = True
except Exception:
    HAS_COST_MODEL = False

    def estimate_option_execution_cost(row: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        prem = 0.0
        try:
            prem = float(getattr(row, "ltp", None) or row.get("ltp") or row.get("close") or row.get("premium") or 0.0)
        except Exception:
            prem = 0.0
        # Very rough fallback cost ~ 0.5% of premium roundtrip if no model
        cost = max(0.5, prem * 0.005)
        return {
            "premium": prem,
            "total_estimated_roundtrip_cost": cost,
            "cost_return_units": cost,
            "cost_pct_of_premium": 0.005 if prem > 0 else 0.0,
        }

    def estimate_option_execution_costs_frame(frame: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        recs = [estimate_option_execution_cost(r, config=config) for r in frame.to_dict("records")]
        return pd.DataFrame.from_records(recs, index=frame.index)


# ---------------------------------------------------------------------------
# Constants / defaults (UI can override via params; structured for easy extension)
# ---------------------------------------------------------------------------
LOT_SIZE = 75  # NIFTY options; adjust if your data uses different (e.g. 25)
DEFAULT_THRESHOLD = 0.60
DEFAULT_TARGET_PCT = 0.20
DEFAULT_STOPLOSS_PCT = 0.10
DEFAULT_MAX_HOLD_BARS = 5
DEFAULT_MAX_TRADES_PER_DAY = 3

# Column tolerant aliases (order = preference)
TIMESTAMP_ALIASES = ["timestamp", "datetime", "time", "date", "ts", "event_time"]
SYMBOL_ALIASES = ["trading_symbol", "tradingsymbol", "symbol", "instrument", "instrument_key"]
OPTION_TYPE_ALIASES = ["option_type", "right", "ce_pe", "type", "call_put"]
STRIKE_ALIASES = ["strike", "strike_price", "strikeprice"]
EXPIRY_ALIASES = ["expiry", "expiration", "exp", "expiry_date"]
PRICE_ALIASES = ["ltp", "close", "price", "premium", "last_price", "mid"]
SPOT_ALIASES = ["spot", "underlying", "nifty_spot", "index_price", "close_spot", "trading_day_spot"]
SPREAD_ALIASES = ["spread", "bid_ask_spread", "bid_ask_spread_pct", "spread_pct"]
BID_ALIASES = ["bid"]
ASK_ALIASES = ["ask"]
SCORE_ALIASES = [
    "pred_proba",
    "prediction_proba",
    "proba",
    "proba_1",
    "probability",
    "class_1_probability",
    "model_score",
    "ml_score",
    "signal_score",
    "edge_score",
    "expected_edge",
    "expected_return",
    "confidence",
    "candidate_score",
    "entry_score",
    "buy_probability",
    "p_up",
    "p_win",
    "score",
    "prediction",
    "signal",
    "edge",
    "ml",
]
SCORE_DIAGNOSTIC_TOKENS = (
    "score",
    "proba",
    "prob",
    "prediction",
    "signal",
    "edge",
    "expected",
    "confidence",
    "ml",
)
SCORE_PRIORITY_TOKENS = (
    "pred_proba",
    "prediction_proba",
    "proba_1",
    "proba",
    "probability",
    "class_1_probability",
    "model_score",
    "confidence",
    "edge_score",
    "expected_edge",
    "expected_return",
    "candidate_score",
    "entry_score",
    "buy_probability",
    "p_win",
    "p_up",
    "score",
)
VOLUME_ALIASES = ["volume", "vol"]
OI_ALIASES = ["oi", "open_interest"]

# Filters we understand from candidate config (simple top level or under "filters")
SUPPORTED_FILTER_KEYS = {"spread_limit", "max_spread_pct", "premium_min", "premium_max", "dte_min", "dte_max", "option_types", "entry_cutoff", "liquidity_min_oi", "min_oi"}

ARTIFACT_FIELD_KEYS = (
    "artifact_path",
    "model_path",
    "bundle_path",
    "artifacts_dir",
    "artifact_dir",
    "candidate_dir",
    "metadata_path",
    "pickle_path",
    "pkl_path",
    "model_dir",
    "path",
    "model_pkl",
    "pkl",
)
ESTIMATOR_KEYS = ("model", "estimator", "pipeline", "clf", "classifier", "calibrator")
FEATURE_KEYS = (
    "feature_cols",
    "feature_order",
    "feature_names",
    "required_features",
    "features",
    "feature_list",
    "model_features",
    "input_features",
    "columns",
    "selected_features",
    "selected_feature_list",
    "live_computable_features",
)
THRESHOLD_KEYS = ("threshold", "selected_threshold", "decision_threshold", "min_probability", "min_confidence")
SIDECAR_FEATURE_FILES = (
    "feature_order.json",
    "features.json",
    "feature_schema.json",
    "metadata.json",
    "model_metadata.json",
    "training_metadata.json",
    "candidate_profile.json",
    "manifest.json",
)
METADATA_CONTAINERS = (
    "metadata",
    "model_metadata",
    "training_metadata",
    "preprocessing",
    "candidate_profile",
    "manifest",
    "model_bundle",
    "bundle",
    "artifact",
    "payload",
    "ml_signals",
)


@dataclass
class LoadedCandidateModel:
    candidate_id: str
    artifact_path: Path
    estimator: Any
    feature_cols: List[str]
    threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    feature_source: str = ""
    model_type: str = ""
    option_type_filter: Optional[List[str]] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    preset: Dict[str, Any] = field(default_factory=dict)
    max_trades_per_day: Optional[int] = None
    score_column: Optional[str] = None
    rejected_reason: Optional[str] = None


@dataclass
class ArtifactLoadFailure:
    candidate_id: str
    path: str
    reason: str
    traceback: str = ""


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "pyproject.toml").exists() or ((p / "src").exists() and (p / "config").exists()):
            return p
    return _REPO_ROOT


def _resolve_path(value: Any, root: Optional[Path] = None, base: Optional[Path] = None) -> Optional[Path]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute():
        return p
    root = root or _project_root()
    candidates = []
    if base is not None:
        candidates.append(base / p)
    candidates.append(root / p)
    candidates.append(Path.cwd() / p)
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return (root / p).resolve()


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower()).strip("_")


def _parse_threshold_from_candidate_id(candidate_id: str) -> Optional[float]:
    text = str(candidate_id or "")
    m = re.search(r"(?:^|_)t(\d{2,3})(?:_|$)", text)
    if not m:
        return None
    val = int(m.group(1))
    if 0 < val <= 100:
        return val / 100.0
    return None


def _candidate_threshold(cand: Dict[str, Any], fallback: float, *, use_candidate_thresholds: bool = True) -> float:
    if not use_candidate_thresholds:
        return float(fallback)
    for key in THRESHOLD_KEYS:
        if key in cand and cand.get(key) is not None:
            val = _safe_float(cand.get(key), float("nan"))
            if math.isfinite(val):
                return val / 100.0 if val > 1.0 else val
    for parent in ("metadata", "thresholds", "preset", "filter_info", "filters"):
        obj = cand.get(parent)
        if isinstance(obj, dict):
            for key in THRESHOLD_KEYS:
                if key in obj and obj.get(key) is not None:
                    val = _safe_float(obj.get(key), float("nan"))
                    if math.isfinite(val):
                        return val / 100.0 if val > 1.0 else val
    parsed = _parse_threshold_from_candidate_id(str(cand.get("candidate_id") or cand.get("id") or ""))
    return float(parsed if parsed is not None else fallback)


def _candidate_option_types(cand: Dict[str, Any]) -> Optional[List[str]]:
    raw = (
        cand.get("option_types")
        or cand.get("option_type")
        or cand.get("side_policy")
        or cand.get("side")
        or (cand.get("filters") or {}).get("option_types")
        or (cand.get("entry_filters") or {}).get("option_types")
    )
    if not raw:
        cid = str(cand.get("candidate_id") or "")
        if "_PE_" in cid or "PE_only" in cid:
            return ["PE"]
        if "_CE_" in cid or "CE_only" in cid:
            return ["CE"]
        return None
    if isinstance(raw, str):
        up = raw.upper()
        if "PE_ONLY" in up or up == "PE":
            return ["PE"]
        if "CE_ONLY" in up or up == "CE":
            return ["CE"]
        return None
    vals = [str(x).upper()[:2] for x in raw if str(x).strip()]
    vals = [v for v in vals if v in {"CE", "PE"}]
    return vals or None


def _candidate_max_trades_per_day(cand: Dict[str, Any]) -> Optional[int]:
    if cand.get("max_trades_per_day") is not None:
        try:
            val = int(cand.get("max_trades_per_day"))
            if val > 0:
                return val
        except Exception:
            pass
    preset = cand.get("preset")
    if isinstance(preset, dict):
        for key in ("max_trades_per_day", "top_n_confidence_per_day"):
            if preset.get(key) is not None:
                try:
                    val = int(preset.get(key))
                    if val > 0:
                        return val
                except Exception:
                    pass
    return None


def _log(log_fn: Optional[Callable[[str], None]], msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    if log_fn:
        try:
            log_fn(line)
        except Exception:
            pass
    else:
        print(line)


def _progress(
    progress_fn: Optional[Callable[[float, str], None]],
    fraction: float,
    message: str,
) -> None:
    if not progress_fn:
        return
    try:
        progress_fn(max(0.0, min(1.0, fraction)), message)
    except Exception:
        pass


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return default


def _parse_timestamp(val: Any) -> Optional[pd.Timestamp]:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return None
    try:
        ts = pd.to_datetime(val, errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map many possible column names to canonical internal names. Does not drop original cols."""
    col_map: Dict[str, str] = {}
    lower_to_orig = {c.lower().strip(): c for c in df.columns}

    def pick(aliases: List[str]) -> Optional[str]:
        for a in aliases:
            if a in lower_to_orig:
                return lower_to_orig[a]
        return None

    tcol = pick(TIMESTAMP_ALIASES)
    if tcol:
        col_map[tcol] = "timestamp"
    scol = pick(SYMBOL_ALIASES)
    if scol:
        col_map[scol] = "symbol"
    ocol = pick(OPTION_TYPE_ALIASES)
    if ocol:
        col_map[ocol] = "option_type"
    kcol = pick(STRIKE_ALIASES)
    if kcol:
        col_map[kcol] = "strike"
    ecol = pick(EXPIRY_ALIASES)
    if ecol:
        col_map[ecol] = "expiry"
    pcol = pick(PRICE_ALIASES)
    if pcol:
        col_map[pcol] = "ltp"
    spcol = pick(SPOT_ALIASES)
    if spcol:
        col_map[spcol] = "spot"
    spreadcol = pick(SPREAD_ALIASES)
    if spreadcol:
        col_map[spreadcol] = "spread"
    bidcol = pick(BID_ALIASES)
    if bidcol:
        col_map[bidcol] = "bid"
    askcol = pick(ASK_ALIASES)
    if askcol:
        col_map[askcol] = "ask"
    volcol = pick(VOLUME_ALIASES)
    if volcol:
        col_map[volcol] = "volume"
    oicol = pick(OI_ALIASES)
    if oicol:
        col_map[oicol] = "oi"
    if col_map:
        df = df.rename(columns=col_map)
    # Preserve model-facing aliases after canonical trade-column normalization.
    if "strike" in df.columns and "strike_price" not in df.columns:
        df["strike_price"] = df["strike"]
    if "spot" in df.columns and "ctx_spot" not in df.columns:
        df["ctx_spot"] = df["spot"]
    if "ltp" in df.columns and "ctx_option_price" not in df.columns:
        df["ctx_option_price"] = df["ltp"]
    return df


def _read_csv_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        return next(reader)


def _preferred_data_path(path: Path, log_fn: Optional[Callable[[str], None]] = None) -> Path:
    if path.suffix.lower() == ".csv":
        pq = path.with_suffix(".parquet")
        if pq.exists():
            _log(log_fn, f"Using Parquet companion instead of CSV: {pq}")
            return pq
    return path


def _read_data_header(path: Path) -> List[str]:
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore

            return [str(name) for name in pq.ParquetFile(path).schema.names]
        except Exception:
            return list(pd.read_parquet(path).head(0).columns)
    return _read_csv_header(path)


def _resolve_header_columns(header_cols: Iterable[str], wanted: Iterable[str]) -> List[str]:
    lower_to_orig = {str(col).strip().lower(): str(col) for col in header_cols}
    resolved: List[str] = []
    seen = set()
    for name in wanted:
        key = str(name).strip().lower()
        orig = lower_to_orig.get(key)
        if orig and orig not in seen:
            resolved.append(orig)
            seen.add(orig)
    return resolved


def _planned_csv_columns(
    header_cols: List[str],
    loaded_models: List[LoadedCandidateModel],
    *,
    include_score_columns: bool = True,
) -> List[str]:
    wanted: List[str] = []
    alias_groups = [
        TIMESTAMP_ALIASES,
        SYMBOL_ALIASES,
        OPTION_TYPE_ALIASES,
        STRIKE_ALIASES,
        EXPIRY_ALIASES,
        PRICE_ALIASES,
        SPOT_ALIASES,
        SPREAD_ALIASES,
        BID_ALIASES,
        ASK_ALIASES,
        VOLUME_ALIASES,
        OI_ALIASES,
    ]
    for aliases in alias_groups:
        wanted.extend(aliases)
    if include_score_columns:
        wanted.extend(str(col) for col in header_cols if any(tok in _normalize_name(col) for tok in SCORE_DIAGNOSTIC_TOKENS))
    for model in loaded_models:
        wanted.extend(str(col) for col in (model.feature_cols or []))
    resolved = _resolve_header_columns(header_cols, wanted)
    return resolved or list(header_cols)


def _read_backtest_csv(
    csv_path: Path,
    *,
    usecols: Optional[List[str]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    attempts: List[Dict[str, Any]] = [
        {"low_memory": False, "usecols": usecols},
        {"low_memory": True, "usecols": usecols},
        {"engine": "python", "usecols": usecols},
    ]
    last_exc: Optional[BaseException] = None
    for idx, kwargs in enumerate(attempts, start=1):
        try:
            desc = ", ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
            _log(log_fn, f"CSV read attempt {idx}: pandas.read_csv({desc or 'default'})")
            return pd.read_csv(csv_path, **kwargs)
        except Exception as exc:
            last_exc = exc
            _log(log_fn, f"CSV read attempt {idx} failed: {exc}")
    raise RuntimeError(f"Failed to read CSV: {last_exc}")


def _read_backtest_frame(
    data_path: Path,
    *,
    usecols: Optional[List[str]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    if data_path.suffix.lower() == ".parquet":
        try:
            _log(log_fn, f"Parquet read: pandas.read_parquet(columns={len(usecols) if usecols else 'all'})")
            return pd.read_parquet(data_path, columns=usecols)
        except Exception as exc:
            _log(log_fn, f"Parquet column-pruned read failed: {exc}; retrying full read")
            return pd.read_parquet(data_path)
    return _read_backtest_csv(data_path, usecols=usecols, log_fn=log_fn)


def _filter_candidates_by_ids(cands: List[Dict[str, Any]], selected_ids: Optional[Iterable[str]]) -> List[Dict[str, Any]]:
    wanted = {_normalize_name(x) for x in (selected_ids or []) if str(x).strip()}
    if not wanted:
        return cands
    out: List[Dict[str, Any]] = []
    for c in cands:
        tokens = [
            c.get("candidate_id"),
            c.get("id"),
            c.get("model_name"),
            c.get("name"),
        ]
        if any(_normalize_name(str(tok or "")) in wanted for tok in tokens):
            out.append(c)
    return out


def _market_time_mask(ts: pd.Series) -> pd.Series:
    local = ts.dt.tz_convert("Asia/Kolkata") if getattr(ts.dt, "tz", None) is not None else ts
    minutes = local.dt.hour * 60 + local.dt.minute
    return (minutes >= (9 * 60 + 15)) & (minutes <= (15 * 60 + 30))


def _entry_prefilter_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if "ts" in df.columns:
        mask &= _market_time_mask(df["ts"])
    if "option_type" in df.columns:
        mask &= df["option_type"].astype(str).str.upper().str[:2].isin(["CE", "PE"])
    if "ltp" in df.columns:
        mask &= pd.to_numeric(df["ltp"], errors="coerce").notna()
        mask &= pd.to_numeric(df["ltp"], errors="coerce") > 0
    if "spread" in df.columns:
        spread = pd.to_numeric(df["spread"], errors="coerce")
        mask &= spread.isna() | (spread >= 0)
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        mask &= vol.isna() | (vol >= 0)
    elif "oi" in df.columns:
        oi = pd.to_numeric(df["oi"], errors="coerce")
        mask &= oi.isna() | (oi >= 0)
    return mask.fillna(False)


def _score_like_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        norm = _normalize_name(col)
        if any(tok in norm for tok in SCORE_DIAGNOSTIC_TOKENS):
            cols.append(str(col))
    return cols


def _score_column_priority(col: str) -> Tuple[int, int, str]:
    norm = _normalize_name(col)
    for i, token in enumerate(SCORE_PRIORITY_TOKENS):
        if token in norm:
            return (i, len(norm), norm)
    return (len(SCORE_PRIORITY_TOKENS), len(norm), norm)


def _select_score_column(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    if preferred:
        for col in df.columns:
            if str(col) == preferred or _normalize_name(str(col)) == _normalize_name(preferred):
                return str(col)
    score_cols = _score_like_columns(df)
    numeric_cols: List[str] = []
    for col in score_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() > 0:
            numeric_cols.append(col)
    if not numeric_cols:
        return None
    return sorted(numeric_cols, key=_score_column_priority)[0]


def _score_column_stats(df: pd.DataFrame, score_cols: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for col in score_cols:
        try:
            vals = pd.to_numeric(df[col], errors="coerce")
            non_null = int(vals.notna().sum())
            stats[str(col)] = {
                "non_null": non_null,
                "min": float(vals.min()) if non_null else None,
                "max": float(vals.max()) if non_null else None,
                "mean": float(vals.mean()) if non_null else None,
            }
        except Exception as exc:
            stats[str(col)] = {"error": str(exc)}
    return stats


def _log_score_diagnostics(
    df: pd.DataFrame,
    *,
    selected_score_col: Optional[str],
    threshold: float,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    score_cols = _score_like_columns(df)
    stats = _score_column_stats(df, score_cols)
    _log(log_fn, f"Score-like columns detected ({len(score_cols)}): {score_cols[:80]}")
    if not selected_score_col:
        _log(log_fn, f"No embedded score column detected. Available score-like columns: {score_cols[:80]}")
    else:
        _log(log_fn, f"Selected embedded score column: {selected_score_col}")
    for col, st in stats.items():
        if "error" in st:
            _log(log_fn, f"  score column {col}: error={st['error']}")
        else:
            _log(
                log_fn,
                "  score column "
                f"{col}: non_null={st.get('non_null')} min={st.get('min')} max={st.get('max')} mean={st.get('mean')}",
            )
    rows_above = 0
    if selected_score_col and selected_score_col in df.columns:
        vals = pd.to_numeric(df[selected_score_col], errors="coerce")
        rows_above = int((vals >= threshold).sum())
    _log(log_fn, f"Rows >= fallback threshold {threshold:.4f} on selected score column: {rows_above}")
    return {
        "score_like_columns": score_cols,
        "score_column_stats": stats,
        "selected_score_column": selected_score_col,
        "rows_above_fallback_threshold": rows_above,
    }


def _ensure_contract_key(row: pd.Series) -> str:
    """Build a stable contract identifier for tracking the same option over time."""
    sym = str(row.get("symbol") or row.get("trading_symbol") or "NIFTY").strip().upper()
    strike = _safe_float(row.get("strike") or row.get("strike_price"), 0.0)
    otype = str(row.get("option_type") or row.get("ce_pe") or "XX").strip().upper()[:2]
    if otype in ("CE", "CALL", "C"):
        otype = "CE"
    elif otype in ("PE", "PUT", "P"):
        otype = "PE"
    exp = str(row.get("expiry") or "").strip()
    # Prefer trading_symbol if it looks complete
    tsym = str(row.get("symbol") or "").strip()
    if tsym and any(c.isdigit() for c in tsym) and ("CE" in tsym.upper() or "PE" in tsym.upper()):
        return tsym.upper()
    if exp:
        return f"{sym}|{exp}|{strike:g}|{otype}"
    return f"{sym}|{strike:g}|{otype}"


def _get_price(row: pd.Series) -> float:
    for c in ("ltp", "close", "price", "premium"):
        if c in row and pd.notna(row[c]):
            return _safe_float(row[c], 0.0)
    return 0.0


def _get_spread_pct(row: pd.Series) -> float:
    for c in ("spread", "bid_ask_spread_pct"):
        if c in row and pd.notna(row[c]):
            val = _safe_float(row[c], 0.0)
            # If looks like absolute (e.g. 5.0 on a 100 prem), convert heuristically; else treat as pct
            if val > 1.0 and val < 100.0:
                # could be rupees or pct; for safety if >1 treat as pct points already? keep as-is
                return val
            return val
    # derive
    bid = _safe_float(row.get("bid"), 0.0)
    ask = _safe_float(row.get("ask"), 0.0)
    mid = _get_price(row)
    if mid > 0 and ask > bid > 0:
        return ((ask - bid) / mid) * 100.0
    return 0.0


def _passes_filters(row: pd.Series, filt: Dict[str, Any], log_fn: Optional[Callable[[str], None]] = None) -> Tuple[bool, List[str]]:
    """Return (ok, failed_reasons). Very defensive; unknown filters ignored."""
    reasons: List[str] = []
    prem = _get_price(row)
    spread = _get_spread_pct(row)

    # spread
    spread_lim = filt.get("spread_limit") or filt.get("max_spread_pct") or filt.get("spread_limit_pct")
    if spread_lim is not None:
        try:
            if spread > float(spread_lim):
                reasons.append(f"spread>{spread_lim}")
        except Exception:
            pass

    # premium band
    pmin = filt.get("premium_min")
    pmax = filt.get("premium_max")
    if pmin is not None and prem < float(pmin):
        reasons.append(f"prem<{pmin}")
    if pmax is not None and prem > float(pmax):
        reasons.append(f"prem>{pmax}")

    # option type whitelist
    otypes = filt.get("option_types") or filt.get("allowed_types")
    if otypes:
        ot = str(row.get("option_type", "")).upper()[:2]
        allowed = [str(x).upper()[:2] for x in (otypes if isinstance(otypes, (list, tuple)) else [otypes])]
        if allowed and ot not in allowed:
            reasons.append("otype_blocked")

    # liquidity (oi/volume)
    liq = filt.get("liquidity_min_oi") or filt.get("min_oi")
    if liq is not None:
        oi = _safe_float(row.get("oi"), 0.0)
        if oi < float(liq):
            reasons.append("oi_low")

    liquidity_min = filt.get("liquidity_min")
    if liquidity_min is not None:
        volume = _safe_float(row.get("volume"), 0.0)
        oi = _safe_float(row.get("oi"), 0.0)
        liquidity_proxy = max(volume, oi)
        try:
            if liquidity_proxy < float(liquidity_min):
                reasons.append("liquidity_low")
        except Exception:
            pass

    ok = len(reasons) == 0
    return ok, reasons


def _candidate_from_model_path(path: Path) -> Dict[str, Any]:
    inferred: Dict[str, Any] = {}
    for loader in (
        _infer_direct_artifact_from_core_summary,
        _infer_direct_artifact_from_retrain_summary,
        _infer_direct_artifact_from_champion_report,
    ):
        inferred = loader(path)
        if inferred:
            break
    cand = {
        "candidate_id": path.stem,
        "enabled": True,
        "model_path": str(path),
        "artifact_path": str(path),
        "model_name": path.stem,
        "direct_model_artifact": True,
    }
    cand.update(inferred)
    return cand


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    try:
        if path.exists() and path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
    except Exception:
        return {}
    return {}


def _artifact_stem_tokens(path: Path) -> set[str]:
    return {tok for tok in re.split(r"[_\W]+", path.stem.lower()) if tok}


def _infer_direct_artifact_from_core_summary(path: Path) -> Dict[str, Any]:
    payload = _load_json_if_exists(path.parent / "core_retrain_summary.json")
    trained = (((payload.get("result") or {}).get("trained_models")) or [])
    if len(trained) != 1 or not isinstance(trained[0], dict):
        return {}
    row = trained[0]
    out: Dict[str, Any] = {}
    if row.get("selected_threshold") is not None:
        out["selected_threshold"] = row.get("selected_threshold")
    if row.get("model"):
        out["model_name"] = row.get("model")
    if row.get("label"):
        out["label_name"] = row.get("label")
    return out


def _infer_direct_artifact_from_retrain_summary(path: Path) -> Dict[str, Any]:
    payload = _load_json_if_exists(path.parent / "retrain_all_models_summary.json")
    models = payload.get("models_trained") or []
    if not isinstance(models, list):
        return {}
    stem_tokens = _artifact_stem_tokens(path)
    for row in models:
        if not isinstance(row, dict):
            continue
        model_name = str(row.get("model_name") or "").lower()
        label_name = str(row.get("label_name") or "").lower()
        label_tokens = {tok for tok in re.split(r"[_\W]+", label_name) if tok}
        if model_name and model_name not in stem_tokens:
            continue
        if label_tokens and not label_tokens.issubset(stem_tokens):
            continue
        out: Dict[str, Any] = {}
        if row.get("selected_threshold") is not None:
            out["selected_threshold"] = row.get("selected_threshold")
        if row.get("model_name"):
            out["model_name"] = row.get("model_name")
        if row.get("label_name"):
            out["label_name"] = row.get("label_name")
        if out:
            return out
    return {}


def _infer_direct_artifact_from_champion_report(path: Path) -> Dict[str, Any]:
    payload = _load_json_if_exists(path.parent / "champion_selection_report.json")
    champion = payload.get("champion") or {}
    if not isinstance(champion, dict):
        return {}
    model_name = str(champion.get("model_name") or "").lower()
    stem_tokens = _artifact_stem_tokens(path)
    if model_name and model_name not in stem_tokens:
        return {}
    out: Dict[str, Any] = {}
    if champion.get("selected_threshold") is not None:
        out["selected_threshold"] = champion.get("selected_threshold")
    if champion.get("model_name"):
        out["model_name"] = champion.get("model_name")
    return out


def _load_candidate_config(path: Optional[str], log_fn: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    if not path:
        return {"candidates": []}
    p = Path(path)
    if not p.exists():
        _log(log_fn, f"WARNING: candidate config not found: {path}")
        return {"candidates": []}
    if p.is_file() and p.suffix.lower() in {".pkl", ".joblib"}:
        _log(log_fn, f"Using direct model artifact as candidate: {path}")
        return {"candidates": [_candidate_from_model_path(p)]}
    if p.is_dir():
        model_files = sorted(
            [*p.rglob("*.pkl"), *p.rglob("*.joblib")],
            key=lambda x: x.stat().st_mtime if x.exists() else 0,
            reverse=True,
        )
        candidates = [_candidate_from_model_path(x) for x in model_files[:20]]
        _log(log_fn, f"Discovered {len(candidates)} model artifact(s) under directory: {path}")
        return {"candidates": candidates}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            cands = data.get("candidates", data.get("selected_candidates", data.get("models", []))) or []
            _log(log_fn, f"Loaded candidate config with {len(cands)} candidate(s) from {path}")
            if "candidates" not in data:
                data["candidates"] = cands
            return data
    except Exception as e:
        _log(log_fn, f"ERROR loading candidate config {path}: {e}")
    return {"candidates": []}


def _find_model_pkl(artifact_dir: Path, log_fn: Optional[Callable[[str], None]] = None) -> Optional[Path]:
    if not artifact_dir or not artifact_dir.exists():
        return None
    # Prefer common retrain patterns
    for pat in [
        "*_profitable_trade_label.pkl",
        "*_cost_survivor_label.pkl",
        "*_high_conviction*.pkl",
        "*_strong*.pkl",
        "*.pkl",
    ]:
        cands = sorted(artifact_dir.glob(pat))
        for c in cands:
            name = c.name.lower()
            if any(x in name for x in ("_metrics", "_threshold", "ensemble_weights", ".json")):
                continue
            return c
    return None


def _iter_candidate_path_values(cand: Dict[str, Any]) -> Iterable[Any]:
    for key in ARTIFACT_FIELD_KEYS:
        if cand.get(key):
            yield cand.get(key)
    paths = cand.get("artifact_paths")
    if isinstance(paths, dict):
        for key in ARTIFACT_FIELD_KEYS + ("model", "bundle", "manifest", "feature_list_path"):
            if paths.get(key):
                yield paths.get(key)
    elif isinstance(paths, (list, tuple)):
        for p in paths:
            yield p


def _artifact_candidate_tokens(cand: Dict[str, Any]) -> List[str]:
    cid = str(cand.get("candidate_id") or cand.get("id") or "").strip()
    model_name = str(cand.get("model_name") or "").strip()
    preset = str(cand.get("preset_family") or cand.get("preset") or "").strip()
    tokens = [cid, model_name, preset]
    base = cid
    for sep in ("_t30_", "_t35_", "_t40_", "_t45_", "_t50_", "_t60_", "_t70_"):
        if sep in cid:
            base = cid.split(sep)[0]
            tokens.append(base)
            break
    return [_normalize_name(t) for t in tokens if _normalize_name(t)]


def _looks_like_model_artifact(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() not in {".pkl", ".joblib"}:
        return False
    blocked = ("metric", "threshold", "ensemble_weight", "oof", "report", "summary")
    return not any(tok in name for tok in blocked)


def _load_pickle_or_joblib(path: Path) -> Any:
    last_exc: Optional[BaseException] = None
    try:
        import joblib

        return joblib.load(path)
    except Exception as exc:
        last_exc = exc
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        if last_exc is not None:
            raise RuntimeError(f"joblib failed with {last_exc!r}; pickle failed with {exc!r}") from exc
        raise


def _feature_list_from_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        for key in FEATURE_KEYS:
            found = _feature_list_from_value(value.get(key))
            if found:
                return found
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text:
            return [x.strip() for x in text.split(",") if x.strip()]
        return [text]
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, (str, int, float)):
                text = str(item).strip()
                if text:
                    out.append(text)
        return out
    try:
        if hasattr(value, "tolist"):
            return _feature_list_from_value(value.tolist())
    except Exception:
        pass
    return []


def _model_is_predictable(obj: Any) -> bool:
    return obj is not None and any(hasattr(obj, attr) for attr in ("predict_proba", "decision_function", "predict"))


class _XgbRfEnsembleAdapter:
    def __init__(
        self,
        *,
        xgb_model: Any,
        rf_model: Any,
        xgb_weight: float = 0.5,
        rf_weight: float = 0.5,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.xgb_model = xgb_model
        self.rf_model = rf_model
        self.xgb_weight = float(xgb_weight)
        self.rf_weight = float(rf_weight)
        self.feature_names_in_ = list(feature_names or [])
        self.classes_ = [0, 1]

    def _predict_positive_prob(self, model: Any, X: Any) -> Any:
        import numpy as np

        X_input = X
        if hasattr(model, "feature_names_in_") and not hasattr(X, "columns"):
            try:
                import pandas as pd

                X_input = pd.DataFrame(X, columns=self.feature_names_in_ or list(getattr(model, "feature_names_in_", [])))
            except Exception:
                X_input = X
        if hasattr(model, "predict_proba"):
            probs = np.asarray(model.predict_proba(X_input), dtype=float)
            if probs.ndim != 2:
                raise RuntimeError("predict_proba returned invalid shape")
            classes = list(getattr(model, "classes_", []))
            if 1 in classes:
                idx = classes.index(1)
            else:
                idx = 1 if probs.shape[1] > 1 else 0
            return probs[:, idx]
        if hasattr(model, "decision_function"):
            raw = np.asarray(model.decision_function(X_input), dtype=float).reshape(-1)
            return 1.0 / (1.0 + np.exp(-raw))
        if hasattr(model, "predict"):
            raw = np.asarray(model.predict(X_input), dtype=float).reshape(-1)
            return np.clip(raw, 0.0, 1.0)
        raise RuntimeError("ensemble component has no predict method")

    def predict_proba(self, X: Any) -> Any:
        import numpy as np

        X_input = X
        if hasattr(X, "loc"):
            X_input = X.loc[:, self.feature_names_in_] if self.feature_names_in_ else X
        total_weight = self.xgb_weight + self.rf_weight
        if total_weight <= 0:
            total_weight = 1.0
            xgb_weight = 0.5
            rf_weight = 0.5
        else:
            xgb_weight = self.xgb_weight
            rf_weight = self.rf_weight
        xgb_probs = self._predict_positive_prob(self.xgb_model, X_input)
        rf_probs = self._predict_positive_prob(self.rf_model, X_input)
        positive = ((xgb_probs * xgb_weight) + (rf_probs * rf_weight)) / total_weight
        positive = np.clip(np.asarray(positive, dtype=float).reshape(-1), 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: Any) -> Any:
        import numpy as np

        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def decision_function(self, X: Any) -> Any:
        return self.predict_proba(X)[:, 1] - 0.5


def _extract_named_component(container: Any, keys: Iterable[str]) -> Any:
    if container is None:
        return None
    if isinstance(container, dict):
        for key in keys:
            if container.get(key) is not None:
                return container.get(key)
    for key in keys:
        if hasattr(container, key):
            try:
                value = getattr(container, key)
            except Exception:
                continue
            if value is not None:
                return value
    return None


def _extract_component_weight(container: Any, family: str) -> float | None:
    family_key = "rf" if family == "rf" else "xgb"
    direct_keys = (
        f"{family_key}_weight",
        f"{family_key}_model_weight",
        f"{family_key}_probability_weight",
    )
    weights_obj = None
    if isinstance(container, dict):
        weights_obj = container.get("weights") or container.get("model_weights")
        for key in direct_keys:
            if container.get(key) not in (None, ""):
                try:
                    return float(container.get(key))
                except Exception:
                    return None
    else:
        for attr in ("weights", "model_weights"):
            if hasattr(container, attr):
                try:
                    weights_obj = getattr(container, attr)
                    break
                except Exception:
                    pass
        for key in direct_keys:
            if hasattr(container, key):
                try:
                    return float(getattr(container, key))
                except Exception:
                    return None
    if isinstance(weights_obj, dict):
        aliases = [family_key, family, "random_forest" if family_key == "rf" else "xgboost"]
        for key in aliases:
            if weights_obj.get(key) not in (None, ""):
                try:
                    return float(weights_obj.get(key))
                except Exception:
                    return None
    return None


def _extract_xgb_rf_ensemble_estimator(obj: Any, feature_cols: Optional[List[str]] = None) -> Any:
    xgb_model = _extract_named_component(obj, ("xgb_model", "xgb_model_", "xgboost_model", "xgboost_model_"))
    rf_model = _extract_named_component(obj, ("rf_model", "rf_model_", "random_forest_model", "random_forest_model_"))
    if xgb_model is None or rf_model is None:
        return None
    xgb_weight = _extract_component_weight(obj, "xgb")
    rf_weight = _extract_component_weight(obj, "rf")
    if xgb_weight is None and rf_weight is None:
        xgb_weight = 0.5
        rf_weight = 0.5
    elif xgb_weight is None:
        rf_weight = float(rf_weight)
        xgb_weight = max(0.0, 1.0 - rf_weight)
    elif rf_weight is None:
        xgb_weight = float(xgb_weight)
        rf_weight = max(0.0, 1.0 - xgb_weight)
    return _XgbRfEnsembleAdapter(
        xgb_model=xgb_model,
        rf_model=rf_model,
        xgb_weight=float(xgb_weight),
        rf_weight=float(rf_weight),
        feature_names=list(feature_cols or []),
    )


def _find_predictable_model(obj: Any, depth: int = 0) -> Any:
    if obj is None or depth > 5:
        return None
    if _model_is_predictable(obj):
        return obj
    if isinstance(obj, dict):
        for key in ESTIMATOR_KEYS + ("model", "wrapped_model"):
            found = _find_predictable_model(obj.get(key), depth + 1)
            if found is not None:
                return found
        for key in METADATA_CONTAINERS:
            found = _find_predictable_model(obj.get(key), depth + 1)
            if found is not None:
                return found
        return None
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _find_predictable_model(item, depth + 1)
            if found is not None:
                return found
        return None
    for key in ESTIMATOR_KEYS + ("model", "estimator", "pipeline", "wrapped_model", "model_bundle"):
        if hasattr(obj, key):
            try:
                found = _find_predictable_model(getattr(obj, key), depth + 1)
                if found is not None:
                    return found
            except Exception:
                pass
    return None


def _find_feature_order_in_obj(obj: Any, source: str = "artifact", depth: int = 0) -> Tuple[List[str], str]:
    if obj is None or depth > 6:
        return [], ""
    if isinstance(obj, dict):
        for key in FEATURE_KEYS:
            features = _feature_list_from_value(obj.get(key))
            if features:
                return features, f"{source}.{key}"
        for key in METADATA_CONTAINERS:
            if key in obj:
                features, found_source = _find_feature_order_in_obj(obj.get(key), f"{source}.{key}", depth + 1)
                if features:
                    return features, found_source
        for key, value in obj.items():
            if isinstance(value, (dict, list, tuple)):
                features, found_source = _find_feature_order_in_obj(value, f"{source}.{key}", depth + 1)
                if features:
                    return features, found_source
        return [], ""
    if isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            features, found_source = _find_feature_order_in_obj(item, f"{source}[{idx}]", depth + 1)
            if features:
                return features, found_source
        return [], ""
    for key in FEATURE_KEYS + ("feature_names_in_",):
        if hasattr(obj, key):
            try:
                features = _feature_list_from_value(getattr(obj, key))
                if features:
                    return features, f"{source}.{key}"
            except Exception:
                pass
    for key in ("metadata", "model_metadata", "training_metadata", "preprocessing", "model", "estimator", "pipeline", "model_bundle"):
        if hasattr(obj, key):
            try:
                features, found_source = _find_feature_order_in_obj(getattr(obj, key), f"{source}.{key}", depth + 1)
                if features:
                    return features, found_source
            except Exception:
                pass
    return [], ""


def _metadata_from_obj(obj: Any, depth: int = 0) -> Dict[str, Any]:
    if obj is None or depth > 4:
        return {}
    meta: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in METADATA_CONTAINERS or key in ("metrics", "preset", "filters"):
                if isinstance(value, dict):
                    meta[key] = value
            elif key in ("candidate_id", "model_name", "label_name", "feature_set_name", "selected_threshold", "threshold"):
                meta[key] = value
        return meta
    for key in ("metadata", "model_metadata", "training_metadata", "metrics", "candidate_id", "model_name"):
        if hasattr(obj, key):
            try:
                value = getattr(obj, key)
                if isinstance(value, dict):
                    meta[key] = value
                elif value is not None:
                    meta[key] = value
            except Exception:
                pass
    return meta


def _sidecar_feature_order(artifact_path: Path) -> Tuple[List[str], str, List[str], Dict[str, Any]]:
    folder = artifact_path.parent if artifact_path else Path(".")
    found_files: List[str] = []
    metadata: Dict[str, Any] = {}
    for fname in SIDECAR_FEATURE_FILES:
        fp = folder / fname
        if not fp.exists():
            continue
        found_files.append(str(fp))
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            metadata[fname] = payload
        features, source = _find_feature_order_in_obj(payload, f"sidecar:{fname}")
        if features:
            return features, source, found_files, metadata
        if isinstance(payload, list):
            features = _feature_list_from_value(payload)
            if features:
                return features, f"sidecar:{fname}", found_files, metadata
    return [], "", found_files, metadata


def _candidate_feature_sidecar_paths(
    cand: Dict[str, Any],
    *,
    root: Path,
    config_dir: Optional[Path],
    artifact_path: Optional[Path] = None,
) -> List[Path]:
    paths: List[Path] = []

    for raw in (
        cand.get("feature_order_source"),
        cand.get("feature_schema_path"),
    ):
        resolved = _resolve_path(raw, root=root, base=config_dir)
        if resolved:
            paths.append(resolved)

    artifact_paths = cand.get("artifact_paths")
    if isinstance(artifact_paths, dict):
        for raw in (
            artifact_paths.get("feature_list_path"),
            artifact_paths.get("feature_schema"),
            artifact_paths.get("feature_schema_path"),
            artifact_paths.get("feature_order_source"),
        ):
            resolved = _resolve_path(raw, root=root, base=config_dir)
            if resolved:
                paths.append(resolved)

    resolved_artifact_dir = _resolve_path(cand.get("artifact_dir") or cand.get("model_dir"), root=root, base=config_dir)
    candidate_dirs: List[Path] = []
    for maybe_dir in (resolved_artifact_dir, artifact_path.parent if artifact_path else None):
        if maybe_dir and maybe_dir.exists() and maybe_dir.is_dir():
            candidate_dirs.append(maybe_dir)

    for folder in candidate_dirs:
        for name in SIDECAR_FEATURE_FILES:
            paths.append(folder / name)
        try:
            paths.extend(sorted(folder.glob("*_metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True))
        except Exception:
            pass

    deduped: List[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _candidate_declared_feature_order(
    cand: Dict[str, Any],
    *,
    root: Path,
    config_dir: Optional[Path],
    artifact_path: Optional[Path] = None,
) -> Tuple[List[str], str]:
    for path in _candidate_feature_sidecar_paths(
        cand,
        root=root,
        config_dir=config_dir,
        artifact_path=artifact_path,
    ):
        if not path.exists() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        features, source = _find_feature_order_in_obj(payload, f"candidate_sidecar:{path.name}")
        if not features and isinstance(payload, list):
            features = _feature_list_from_value(payload)
            if features:
                source = f"candidate_sidecar:{path.name}"
        if features:
            return [str(x) for x in features if str(x).strip()], source
    return [], ""


def extract_model_feature_order(loaded_artifact: Any, artifact_path: Path | str) -> Dict[str, Any]:
    """Return the predictable model, feature order, source, and metadata for repo model bundles."""
    path = Path(artifact_path)
    model = _find_predictable_model(loaded_artifact)
    features, feature_source = _find_feature_order_in_obj(loaded_artifact, "artifact")
    metadata = _metadata_from_obj(loaded_artifact)
    sidecar_files: List[str] = []
    if not features:
        side_features, side_source, sidecar_files, side_meta = _sidecar_feature_order(path)
        if side_features:
            features = side_features
            feature_source = side_source
        if side_meta:
            metadata.setdefault("sidecars", {}).update(side_meta)
    if not features and model is not None and hasattr(model, "feature_names_in_"):
        features = _feature_list_from_value(getattr(model, "feature_names_in_"))
        if features:
            feature_source = "model.feature_names_in_"
    return {
        "model": model if model is not None else loaded_artifact,
        "feature_order": [str(x) for x in features if str(x).strip()],
        "feature_source": feature_source,
        "metadata": metadata,
        "sidecar_files_found": sidecar_files,
        "artifact_type": type(loaded_artifact).__name__,
        "model_type": type(model if model is not None else loaded_artifact).__name__,
        "top_level_keys": list(loaded_artifact.keys()) if isinstance(loaded_artifact, dict) else [],
    }


def _log_model_features_missing(
    *,
    artifact_path: Path,
    artifact_obj: Any,
    introspection: Dict[str, Any],
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    _log(log_fn, "MODEL_FEATURES_MISSING")
    _log(log_fn, f"  artifact_path={artifact_path}")
    _log(log_fn, f"  artifact_type={introspection.get('artifact_type') or type(artifact_obj).__name__}")
    _log(log_fn, f"  top_level_keys={introspection.get('top_level_keys') or []}")
    _log(log_fn, f"  model_type={introspection.get('model_type')}")
    _log(log_fn, f"  sidecar_files_found={introspection.get('sidecar_files_found') or []}")
    _log(log_fn, "  suggestion=write feature_order/feature_names into the model bundle or a sidecar in the artifact directory")


def _extract_first(obj: Any, keys: Iterable[str], depth: int = 0) -> Any:
    if depth > 4 or obj is None:
        return None
    if isinstance(obj, dict):
        for key in keys:
            if key in obj and obj[key] is not None:
                return obj[key]
        for key in ("metadata", "bundle", "artifact", "model_bundle", "payload", "candidate"):
            if isinstance(obj.get(key), dict):
                found = _extract_first(obj[key], keys, depth + 1)
                if found is not None:
                    return found
    for key in keys:
        if hasattr(obj, key):
            try:
                val = getattr(obj, key)
                if val is not None:
                    return val
            except Exception:
                pass
    for key in ("metadata", "bundle", "artifact", "model_bundle", "payload", "candidate"):
        if hasattr(obj, key):
            try:
                found = _extract_first(getattr(obj, key), keys, depth + 1)
                if found is not None:
                    return found
            except Exception:
                pass
    return None


def _extract_metadata(obj: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for key in ("metadata", "metrics", "candidate_profile", "manifest", "preset", "filters"):
            if isinstance(obj.get(key), dict):
                meta[key] = obj.get(key)
        for key in ("candidate_id", "model_name", "label_name", "feature_set_name"):
            if key in obj:
                meta[key] = obj.get(key)
    else:
        for key in ("metadata", "metrics", "candidate_id", "model_name"):
            if hasattr(obj, key):
                try:
                    meta[key] = getattr(obj, key)
                except Exception:
                    pass
    return meta


def _normalize_loaded_artifact(
    cand: Dict[str, Any],
    artifact_path: Path,
    obj: Any,
    *,
    fallback_threshold: float,
    use_candidate_thresholds: bool,
) -> LoadedCandidateModel:
    introspection = extract_model_feature_order(obj, artifact_path)
    feature_cols = [str(x) for x in (introspection.get("feature_order") or [])]
    estimator = introspection.get("model")
    if estimator is None or not _model_is_predictable(estimator):
        ensemble_estimator = _extract_xgb_rf_ensemble_estimator(obj, feature_cols=feature_cols)
        if ensemble_estimator is not None:
            estimator = ensemble_estimator
    if not _model_is_predictable(estimator):
        estimator = _extract_first(obj, ESTIMATOR_KEYS)
    if estimator is not None and not _model_is_predictable(estimator):
        ensemble_estimator = _extract_xgb_rf_ensemble_estimator(estimator, feature_cols=feature_cols)
        if ensemble_estimator is not None:
            estimator = ensemble_estimator
    if estimator is None and _model_is_predictable(obj):
        estimator = obj
    if estimator is None:
        raise RuntimeError("no estimator found in artifact")
    if isinstance(obj, dict) and obj.get("scaler") is not None and obj.get("model") is estimator:
        scaler = obj.get("scaler")
        if hasattr(scaler, "transform"):
            try:
                from sklearn.pipeline import Pipeline

                estimator = Pipeline([("scaler", scaler), ("model", estimator)])
            except Exception:
                pass
    threshold_val = _extract_first(obj, THRESHOLD_KEYS)
    if threshold_val is not None and use_candidate_thresholds:
        th = _safe_float(threshold_val, _candidate_threshold(cand, fallback_threshold, use_candidate_thresholds=True))
        if th > 1:
            th /= 100.0
    else:
        th = _candidate_threshold(cand, fallback_threshold, use_candidate_thresholds=use_candidate_thresholds)
    resolved_model_type = str(introspection.get("model_type") or "")
    if isinstance(estimator, _XgbRfEnsembleAdapter):
        resolved_model_type = "xgb_rf_ensemble"
    elif not resolved_model_type or resolved_model_type == "dict":
        resolved_model_type = type(estimator).__name__
    return LoadedCandidateModel(
        candidate_id=str(cand.get("candidate_id") or cand.get("id") or artifact_path.stem),
        artifact_path=artifact_path,
        estimator=estimator,
        feature_cols=feature_cols,
        threshold=float(th),
        metadata={**_extract_metadata(obj), **(introspection.get("metadata") or {})},
        feature_source=str(introspection.get("feature_source") or ""),
        model_type=resolved_model_type,
        option_type_filter=_candidate_option_types(cand),
        filters=cand.get("filters") or cand.get("entry_filters") or {},
        preset=cand.get("preset") if isinstance(cand.get("preset"), dict) else {},
        max_trades_per_day=_candidate_max_trades_per_day(cand),
    )


def _load_candidate_model(
    cand: Dict[str, Any],
    *,
    root: Path,
    config_dir: Optional[Path],
    fallback_threshold: float,
    use_candidate_thresholds: bool,
    log_fn: Optional[Callable[[str], None]] = None,
    strict_artifacts: bool = True,
    allow_artifact_fallback: bool = False,
    debug: bool = False,
) -> Tuple[Optional[LoadedCandidateModel], List[ArtifactLoadFailure]]:
    from candidate_artifact_resolver import log_artifact_resolution, resolve_candidate_artifact

    cid = str(cand.get("candidate_id") or cand.get("id") or cand.get("model_name") or "candidate")
    failures: List[ArtifactLoadFailure] = []
    direct_artifact = bool(cand.get("direct_model_artifact"))
    direct_path = _resolve_path(cand.get("model_path") or cand.get("artifact_path"), root=root, base=config_dir)
    if direct_artifact and direct_path and direct_path.is_file() and direct_path.suffix.lower() in {".pkl", ".joblib"}:
        artifact = direct_path
        try:
            obj = _load_pickle_or_joblib(artifact)
            model = _normalize_loaded_artifact(
                cand,
                artifact,
                obj,
                fallback_threshold=fallback_threshold,
                use_candidate_thresholds=use_candidate_thresholds,
            )
            if not model.feature_cols:
                feature_cols, feature_source = _candidate_declared_feature_order(
                    cand,
                    root=root,
                    config_dir=config_dir,
                    artifact_path=artifact,
                )
                if feature_cols:
                    model.feature_cols = feature_cols
                    model.feature_source = feature_source or model.feature_source
            if debug and not model.feature_cols:
                _log_model_features_missing(
                    artifact_path=artifact,
                    artifact_obj=obj,
                    introspection=extract_model_feature_order(obj, artifact),
                    log_fn=log_fn,
                )
            _log(log_fn, f"Candidate {cid}: selected direct artifact path {artifact} threshold={model.threshold:.4f}")
            return model, failures
        except Exception as exc:
            tb = traceback.format_exc()
            _log(log_fn, f"Candidate {cid}: ERROR loading direct model {artifact}: {exc}")
            _log(log_fn, tb)
            failures.append(ArtifactLoadFailure(cid, str(artifact), str(exc), tb))
            return None, failures

    resolution = resolve_candidate_artifact(
        cand,
        root,
        strict=strict_artifacts,
        allow_fallback=allow_artifact_fallback,
    )
    log_artifact_resolution(resolution, prefix="[BACKTEST-ARTIFACT]")
    if resolution.artifact_identity_status == "MISMATCH":
        failures.append(
            ArtifactLoadFailure(
                cid,
                resolution.configured_artifact_path or resolution.expected_artifact_path,
                "ARTIFACT_IDENTITY_MISMATCH",
            )
        )
        if resolution.mismatch_fields:
            failures.append(ArtifactLoadFailure(cid, resolution.expected_artifact_path, ",".join(resolution.mismatch_fields)))
        return None, failures
    if not resolution.loaded or not resolution.model_file:
        failures.append(
            ArtifactLoadFailure(
                cid,
                resolution.expected_artifact_path,
                resolution.disable_reason or "ARTIFACT_NOT_FOUND",
            )
        )
        return None, failures

    artifact = Path(resolution.model_file)
    try:
        if HAS_ML_SIGNALS and callable(load_model):
            try:
                bundle = load_model(str(artifact))  # type: ignore[misc]
                if bundle is not None:
                    model = _normalize_loaded_artifact(
                        cand,
                        artifact,
                        bundle,
                        fallback_threshold=fallback_threshold,
                        use_candidate_thresholds=use_candidate_thresholds,
                    )
                    if debug and not model.feature_cols:
                        _log_model_features_missing(
                            artifact_path=artifact,
                            artifact_obj=bundle,
                            introspection=extract_model_feature_order(bundle, artifact),
                            log_fn=log_fn,
                        )
                    if not model.feature_cols:
                        _log(log_fn, f"Candidate {cid}: ml_signals.load_model returned no feature order; retrying raw artifact introspection")
                    else:
                        _log(log_fn, f"Candidate {cid}: selected artifact path {artifact} via ml_signals.load_model threshold={model.threshold:.4f}")
                        return model, failures
            except Exception:
                pass
        obj = _load_pickle_or_joblib(artifact)
        model = _normalize_loaded_artifact(
            cand,
            artifact,
            obj,
            fallback_threshold=fallback_threshold,
            use_candidate_thresholds=use_candidate_thresholds,
        )
        if not model.feature_cols:
            feature_cols, feature_source = _candidate_declared_feature_order(
                cand,
                root=root,
                config_dir=config_dir,
                artifact_path=artifact,
            )
            if feature_cols:
                model.feature_cols = feature_cols
                model.feature_source = feature_source or model.feature_source
        if debug and not model.feature_cols:
            _log_model_features_missing(
                artifact_path=artifact,
                artifact_obj=obj,
                introspection=extract_model_feature_order(obj, artifact),
                log_fn=log_fn,
            )
        _log(log_fn, f"Candidate {cid}: selected artifact path {artifact} threshold={model.threshold:.4f}")
        return model, failures
    except Exception as exc:
        tb = traceback.format_exc()
        _log(log_fn, f"Candidate {cid}: ERROR loading model {artifact}: {exc}")
        _log(log_fn, tb)
        failures.append(ArtifactLoadFailure(cid, str(artifact), str(exc), tb))
        return None, failures


def _load_bundle_for_candidate(cand: Dict[str, Any], log_fn: Optional[Callable[[str], None]] = None) -> Optional[Any]:
    """Return a bundle usable with ml_signals.predict or a simple wrapper."""
    art = cand.get("artifact_dir") or cand.get("artifact_path") or cand.get("model_dir") or cand.get("path")
    if not art and isinstance(cand.get("artifact_paths"), dict):
        paths = cand.get("artifact_paths") or {}
        art = paths.get("artifact_dir") or paths.get("model_dir") or paths.get("model_path")
    if not art:
        # maybe direct model path in config
        mp = cand.get("model_path") or cand.get("pkl")
        if mp:
            art = mp
    if not art:
        return None
    p = Path(art)
    if p.is_file() and p.suffix == ".pkl":
        model_path = p
    else:
        model_path = _find_model_pkl(p, log_fn)
    if not model_path or not model_path.exists():
        _log(log_fn, f"  candidate {cand.get('candidate_id', cand.get('model_name','?'))}: no .pkl found under {art}")
        return None
    try:
        import joblib
        loaded = joblib.load(model_path)
        # Normalize to something predict() understands (MLModelBundle or dict with model+feature_names)
        if HAS_ML_SIGNALS:
            try:
                b = load_model(str(model_path))  # type: ignore
                if b:
                    _log(log_fn, f"  Loaded model for {cand.get('candidate_id','?')} via ml_signals: {model_path.name}")
                    return b
            except Exception:
                pass
        # Fallback wrapper
        if isinstance(loaded, dict) and "model" in loaded:
            bundle = {
                "model": loaded["model"],
                "feature_names": list(loaded.get("feature_names") or []),
                "scaler_mean": loaded.get("scaler_mean"),
                "scaler_std": loaded.get("scaler_std"),
            }
        else:
            bundle = {"model": loaded, "feature_names": [], "scaler_mean": None, "scaler_std": None}
        _log(log_fn, f"  Loaded model for {cand.get('candidate_id','?')}: {model_path.name} (fallback)")
        return bundle
    except Exception as e:
        _log(log_fn, f"  ERROR loading model {model_path}: {e}")
        return None


def _get_feature_names(bundle: Any) -> List[str]:
    if bundle is None:
        return []
    if hasattr(bundle, "feature_names"):
        return list(getattr(bundle, "feature_names") or [])
    if isinstance(bundle, dict):
        names = list(bundle.get("feature_names") or [])
        if names:
            return names
        model = bundle.get("model")
        if hasattr(model, "feature_names_in_"):
            return [str(x) for x in getattr(model, "feature_names_in_")]
    return []


def _predict_score(bundle: Any, row: pd.Series, feature_names: List[str], log_fn: Optional[Callable[[str], None]] = None) -> float:
    """Build vector from row using feature_names; call predict; return proba for positive class."""
    if bundle is None or not feature_names:
        return 0.0
    vec: List[float] = []
    for fn in feature_names:
        # Never use future/leak columns even if present in CSV
        if any(bad in fn.lower() for bad in ("future", "label", "net_forward", "gross_forward", "return_to_cost", "cost_return_units")):
            vec.append(0.0)
            continue
        v = row.get(fn, row.get(fn.replace("_", "")))
        vec.append(_safe_float(v, 0.0))
    try:
        if HAS_ML_SIGNALS and callable(ml_predict):
            probs = ml_predict(bundle, [vec])  # type: ignore
            return float(probs[0]) if probs else 0.0
        # manual fallback (copied minimal logic)
        model = bundle.get("model") if isinstance(bundle, dict) else getattr(bundle, "model", None)
        sm = bundle.get("scaler_mean") if isinstance(bundle, dict) else getattr(bundle, "scaler_mean", None)
        ss = bundle.get("scaler_std") if isinstance(bundle, dict) else getattr(bundle, "scaler_std", None)
        if model is None:
            return 0.0
        X = [vec]
        if sm and ss and len(vec) == len(sm):
            X = [[(vec[j] - sm[j]) / ss[j] if ss[j] else vec[j] for j in range(len(vec))]]
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            return float(p[0][1]) if len(p[0]) > 1 else float(p[0][0])
        if hasattr(model, "predict"):
            pr = model.predict(X)
            return float(pr[0])
        return 0.0
    except Exception as e:
        _log(log_fn, f"  predict error: {e}")
        return 0.0


def _feature_alignment_diagnostics(
    model: LoadedCandidateModel,
    df: pd.DataFrame,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    required = list(model.feature_cols or [])
    if not required and hasattr(model.estimator, "feature_names_in_"):
        required = [str(x) for x in getattr(model.estimator, "feature_names_in_")]
        model.feature_cols = required
    available = set(str(c) for c in df.columns)
    missing = [f for f in required if f not in available]
    diag = {
        "candidate_id": model.candidate_id,
        "artifact_path": str(model.artifact_path),
        "model_type": model.model_type or type(model.estimator).__name__,
        "feature_source": model.feature_source,
        "required_feature_count": len(required),
        "available_feature_count": len([f for f in required if f in available]),
        "missing_feature_count": len(missing),
        "first_20_required_features": required[:20],
        "first_20_missing_features": missing[:20],
        "missing_features_first_50": missing[:50],
        "model_ready": len(required) > 0 and len(missing) == 0,
    }
    _log(
        log_fn,
        f"MODEL-SCORING candidate_id={model.candidate_id} artifact_path={model.artifact_path} "
        f"model_type={diag['model_type']} feature_source={diag['feature_source']} "
        f"required_feature_count={diag['required_feature_count']} "
        f"available_feature_count={diag['available_feature_count']} "
        f"missing_feature_count={diag['missing_feature_count']} "
        f"first_20_required_features={diag['first_20_required_features']} "
        f"first_20_missing_features={diag['first_20_missing_features']} "
        f"model_ready={diag['model_ready']}",
    )
    if missing:
        _log(log_fn, f"  missing first 50: {missing[:50]}")
    return diag


def _numeric_col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _first_numeric_col(df: pd.DataFrame, names: Iterable[str], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _materialize_offline_model_features(
    df: pd.DataFrame,
    required_features: Iterable[str],
    *,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Create deterministic offline equivalents for live-derived model features."""
    required = set(str(f) for f in required_features)
    added: List[str] = []

    def add(name: str, value: Any) -> None:
        if name in required and name not in df.columns:
            df[name] = value
            added.append(name)

    if "strike" in df.columns and "strike_price" not in df.columns:
        add("strike_price", df["strike"])
    if "strike_price" in df.columns and "strike" not in df.columns:
        df["strike"] = df["strike_price"]

    strike = _first_numeric_col(df, ("strike_price", "strike"), 0.0)
    spot = _first_numeric_col(df, ("ctx_spot", "trading_day_spot", "spot_close", "spot", "underlying", "close_spot"), 0.0)
    ltp = _first_numeric_col(df, ("ltp", "ctx_option_price", "close", "price"), 0.0)
    dte = _first_numeric_col(df, ("dte_days", "time_to_expiry_days"), 0.0)
    safe_spot = spot.where(spot.abs() > 1e-9)
    safe_strike = strike.where(strike.abs() > 1e-9)

    add("time_to_expiry_days", dte)
    add("time_to_expiry_years", (dte / 365.0).clip(lower=0.0))
    add("distance_from_atm", (strike - spot).abs())
    add("distance_from_atm_pct", ((strike - spot).abs() / safe_spot.abs()).fillna(0.0))
    add("log_moneyness", (safe_strike / safe_spot).apply(lambda x: math.log(float(x)) if pd.notna(x) and x > 0 else 0.0))

    opt = df.get("option_type")
    if opt is not None:
        opt_s = opt.astype(str).str.upper()
        intrinsic_ce = (spot - strike).clip(lower=0.0)
        intrinsic_pe = (strike - spot).clip(lower=0.0)
        intrinsic = intrinsic_ce.where(opt_s.str.startswith("C"), intrinsic_pe)
    else:
        intrinsic = pd.Series(0.0, index=df.index)
    add("intrinsic_value", intrinsic)
    add("extrinsic_value", (ltp - intrinsic).clip(lower=0.0))

    vol_proxy = _first_numeric_col(df, ("final_iv", "bs_iv", "realized_vol_30", "recent_vol_proxy", "atr_pct", "ret_std"), 0.20)
    vol_proxy = vol_proxy.where(vol_proxy > 0, 0.20).fillna(0.20)
    add("bs_iv", vol_proxy)
    add("final_iv", vol_proxy)
    mny = _first_numeric_col(df, ("moneyness",), 1.0).fillna(1.0)
    delta_base = (0.5 + (mny - 1.0) * 2.0).clip(lower=0.05, upper=0.95)
    if opt is not None:
        opt_s = opt.astype(str).str.upper()
        delta = delta_base.where(opt_s.str.startswith("C"), -delta_base)
    else:
        delta = delta_base
    add("bs_delta", delta)
    add("bs_gamma", pd.Series(0.0, index=df.index))
    add("bs_theta", pd.Series(0.0, index=df.index))
    add("bs_vega", vol_proxy * 0.01)
    add("bs_rho", pd.Series(0.0, index=df.index))
    add("greeks_quality_score", pd.Series(0.0, index=df.index))

    spread = _first_numeric_col(df, ("bid_ask_spread", "spread", "bid_ask_spread_abs"), 0.0)
    spread_pct = _first_numeric_col(df, ("bid_ask_spread_pct", "spread_pct"), 0.0)
    if (spread.fillna(0.0) == 0.0).all():
        spread = (ltp.abs() * 0.01).fillna(0.0)
    if (spread_pct.fillna(0.0) == 0.0).all():
        spread_pct = (spread / ltp.where(ltp.abs() > 1e-9).abs()).fillna(0.0)
    add("bid_ask_spread", spread)
    add("bid_ask_spread_pct", spread_pct)
    add("mid_price", ltp)
    add("ltp_vs_mid_diff", pd.Series(0.0, index=df.index))
    add("ltp_vs_mid_diff_pct", pd.Series(0.0, index=df.index))

    cost_pct = _first_numeric_col(df, ("cost_pct_of_premium",), 0.0)
    if (cost_pct.fillna(0.0) == 0.0).all():
        cost_pct = (spread_pct.fillna(0.0) + 0.005).clip(lower=0.0)
    add("cost_pct_of_premium", cost_pct)
    add("estimated_cost_bps", cost_pct * 10000.0)
    add("spread_cost_component", spread_pct.fillna(0.0) * 10000.0)
    add("slippage_cost_component", pd.Series(5.0, index=df.index))
    add("brokerage_or_fee_component", pd.Series(5.0, index=df.index))

    date_key = df["date"] if "date" in df.columns else (df["ts"].dt.date if "ts" in df.columns else pd.Series(0, index=df.index))

    def pct_rank(series: pd.Series) -> pd.Series:
        vals = pd.to_numeric(series, errors="coerce").fillna(0.0)
        try:
            return vals.groupby(date_key).rank(pct=True).fillna(0.5)
        except Exception:
            return pd.Series(0.5, index=df.index)

    add("spread_pctile_day", pct_rank(spread_pct))
    add("volume_pctile_day", pct_rank(_numeric_col(df, "volume", 0.0)))
    add("oi_pctile_day", pct_rank(_numeric_col(df, "oi", 0.0)))
    add("premium_pctile_expiry", pct_rank(ltp))
    add("recent_vol_proxy", vol_proxy)
    ce_vol = _first_numeric_col(df, ("volume_CE", "ce_vol_day"), 0.0)
    pe_vol = _first_numeric_col(df, ("volume_PE", "pe_vol_day"), 0.0)
    add("ce_vol_day", ce_vol)
    add("pe_vol_day", pe_vol)
    add("ce_pe_vol_imbalance", ((ce_vol - pe_vol) / (ce_vol + pe_vol).where((ce_vol + pe_vol).abs() > 1e-9)).fillna(0.0))
    add("atm_distance_rank_day", pct_rank((strike - spot).abs()))
    vol_rank = pct_rank(_numeric_col(df, "volume", 0.0))
    oi_rank = pct_rank(_numeric_col(df, "oi", 0.0))
    add("liq_regime_day", ((vol_rank > 0.5).astype(float) + (oi_rank > 0.5).astype(float)))
    add("spread_regime_day", (pct_rank(spread_pct) > 0.5).astype(float))
    ce_oi = _first_numeric_col(df, ("oi_CE",), 0.0)
    pe_oi = _first_numeric_col(df, ("oi_PE",), 0.0)
    rel_from_vol = ((ce_vol - pe_vol) / (ce_vol + pe_vol).where((ce_vol + pe_vol).abs() > 1e-9)).fillna(0.0)
    rel_from_oi = ((ce_oi - pe_oi) / (ce_oi + pe_oi).where((ce_oi + pe_oi).abs() > 1e-9)).fillna(0.0)
    add("ce_pe_rel_strength", rel_from_vol.where((ce_vol + pe_vol).abs() > 1e-9, rel_from_oi))

    direct_strategy_features = {
        "mean_reversion_zscore",
        "mean_reversion_entry_score",
        "mean_reversion_expected_reversion_pct",
        "mean_reversion_half_life_bars",
        "mean_reversion_buy_call",
        "mean_reversion_buy_put",
        "stat_arb_zscore",
        "stat_arb_confidence",
        "stat_arb_spread_pct",
        "stat_arb_hedge_ratio",
        "stat_arb_long_spread",
        "stat_arb_short_spread",
    }
    if required.intersection(direct_strategy_features):
        group_key_col = ""
        for candidate_key in ("instrument_key", "_contract", "symbol"):
            if candidate_key in df.columns:
                group_key_col = candidate_key
                break
        if group_key_col and ("ts" in df.columns or "timestamp" in df.columns):
            work = df
            if "ts" in work.columns:
                work = work.sort_values([group_key_col, "ts"], kind="stable")
            elif "timestamp" in work.columns:
                work = work.sort_values([group_key_col, "timestamp"], kind="stable")
            grouped_price = work.groupby(group_key_col, sort=False)[ltp.name if hasattr(ltp, "name") and ltp.name else "ltp"]

            mr_mean = grouped_price.transform(lambda s: s.rolling(20, min_periods=5).mean())
            mr_std = grouped_price.transform(lambda s: s.rolling(20, min_periods=5).std(ddof=0)).replace(0.0, pd.NA)
            mr_z = ((work["ltp"] - mr_mean) / mr_std).replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
            mr_expected = ((mr_mean - work["ltp"]) / work["ltp"].replace(0.0, pd.NA)).replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
            mr_half_life = grouped_price.transform(
                lambda s: s.rolling(20, min_periods=5).std(ddof=0).fillna(0.0).gt(0.0).astype(float) * 10.0
            ).fillna(0.0)

            fair_value = grouped_price.transform(lambda s: s.ewm(span=21, adjust=False, min_periods=5).mean())
            x_mean = grouped_price.transform(lambda s: s.rolling(30, min_periods=5).mean())
            y_mean = fair_value.groupby(work[group_key_col], sort=False).transform(lambda s: s.rolling(30, min_periods=5).mean())
            xy_mean = (work["ltp"] * fair_value).groupby(work[group_key_col], sort=False).transform(lambda s: s.rolling(30, min_periods=5).mean())
            x2_mean = (work["ltp"] * work["ltp"]).groupby(work[group_key_col], sort=False).transform(lambda s: s.rolling(30, min_periods=5).mean())
            cov_xy = xy_mean - (x_mean * y_mean)
            var_x = (x2_mean - (x_mean * x_mean)).replace(0.0, pd.NA)
            hedge_ratio = (cov_xy / var_x).replace([float("inf"), float("-inf")], pd.NA).fillna(1.0)
            spread_series = fair_value - (hedge_ratio * work["ltp"])
            spread_mean = spread_series.groupby(work[group_key_col], sort=False).transform(lambda s: s.rolling(30, min_periods=5).mean())
            spread_std = spread_series.groupby(work[group_key_col], sort=False).transform(lambda s: s.rolling(30, min_periods=5).std(ddof=0)).replace(0.0, pd.NA)
            stat_z = ((spread_series - spread_mean) / spread_std).replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
            stat_spread_pct = (spread_series / work["ltp"].replace(0.0, pd.NA)).replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)

            feature_frame = pd.DataFrame(
                {
                    "mean_reversion_zscore": mr_z,
                    "mean_reversion_entry_score": (mr_z.abs() / 1.25).clip(lower=0.0, upper=1.0).fillna(0.0),
                    "mean_reversion_expected_reversion_pct": mr_expected,
                    "mean_reversion_half_life_bars": mr_half_life,
                    "mean_reversion_buy_call": mr_z.le(-1.25).astype(float),
                    "mean_reversion_buy_put": mr_z.ge(1.25).astype(float),
                    "stat_arb_zscore": stat_z,
                    "stat_arb_confidence": (stat_z.abs() / 1.5).clip(lower=0.0, upper=1.0).fillna(0.0),
                    "stat_arb_spread_pct": stat_spread_pct,
                    "stat_arb_hedge_ratio": hedge_ratio.fillna(1.0),
                    "stat_arb_long_spread": stat_z.le(-1.5).astype(float),
                    "stat_arb_short_spread": stat_z.ge(1.5).astype(float),
                },
                index=work.index,
            ).reindex(df.index).fillna(0.0)

            for feature_name in direct_strategy_features:
                add(feature_name, feature_frame[feature_name])

    if added:
        _log(log_fn, f"Offline model feature materialization added {len(added)} column(s): {added[:50]}")
    return {"added_features": added, "added_count": len(added)}


def _model_can_predict(model: LoadedCandidateModel, df: pd.DataFrame, *, log_fn: Optional[Callable[[str], None]] = None) -> bool:
    diag = _feature_alignment_diagnostics(model, df, log_fn=log_fn)
    required = int(diag["required_feature_count"])
    missing = int(diag["missing_feature_count"])
    if required == 0:
        _log(log_fn, f"Candidate {model.candidate_id}: no feature list found; skipping model artifact")
        model.rejected_reason = "no_feature_list"
        return False
    if missing == 0:
        return True
    # Safe default fill is only acceptable for a small minority of features.
    if missing <= 3 or missing / max(required, 1) <= 0.05:
        _log(log_fn, f"Candidate {model.candidate_id}: filling {missing} missing feature(s) with 0.0 for offline diagnostics")
        return True
    model.rejected_reason = f"too_many_missing_features:{missing}/{required}"
    _log(log_fn, f"Candidate {model.candidate_id}: skipped, too many missing features ({missing}/{required})")
    return False


def _build_feature_frame(model: LoadedCandidateModel, row: pd.Series) -> pd.DataFrame:
    vals: Dict[str, float] = {}
    for fn in model.feature_cols:
        low = fn.lower()
        if any(bad in low for bad in ("future", "label", "net_forward", "gross_forward", "return_to_cost", "cost_return_units")):
            vals[fn] = 0.0
            continue
        vals[fn] = _safe_float(row.get(fn, 0.0), 0.0)
    return pd.DataFrame([vals], columns=model.feature_cols)


def _safe_score_column_name(candidate_id: str) -> str:
    return "ml_score_" + re.sub(r"[^A-Za-z0-9_]+", "_", str(candidate_id or "candidate")).strip("_")[:120]


def _build_model_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    data: Dict[str, Any] = {}
    for fn in feature_cols:
        if fn in df.columns:
            data[fn] = pd.to_numeric(df[fn], errors="coerce")
        else:
            data[fn] = 0.0
    X = pd.DataFrame(data, index=df.index, columns=feature_cols)
    # Avoid DataFrame.replace(..., pd.NA) here: some recent pandas builds can
    # crash with an internal block-manager IndexError on large mixed frames.
    for col in X.columns:
        series = pd.to_numeric(X[col], errors="coerce")
        series = series.mask(series == float("inf"), 0.0)
        series = series.mask(series == float("-inf"), 0.0)
        X[col] = series.fillna(0.0)
    return X


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _predict_loaded_candidate_score(
    model: LoadedCandidateModel,
    row: pd.Series,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
) -> float:
    try:
        X = _build_feature_frame(model, row)
        est = model.estimator
        X_input = X if hasattr(est, "feature_names_in_") else X.to_numpy(dtype="float32")
        if hasattr(est, "predict_proba"):
            probs = est.predict_proba(X_input)
            try:
                classes = list(getattr(est, "classes_", []))
                idx = classes.index(1) if 1 in classes else 1 if len(probs[0]) > 1 else 0
            except Exception:
                idx = 1 if len(probs[0]) > 1 else 0
            return _safe_float(probs[0][idx], 0.0)
        if hasattr(est, "decision_function"):
            raw = est.decision_function(X_input)
            val = raw[0] if hasattr(raw, "__len__") else raw
            if hasattr(val, "__len__"):
                val = val[-1]
            return _sigmoid(float(val))
        if hasattr(est, "predict"):
            preds = est.predict(X_input)
            _log(log_fn, f"Candidate {model.candidate_id}: using predict() fallback; score may be class label, not probability")
            return max(0.0, min(1.0, _safe_float(preds[0], 0.0)))
    except Exception as exc:
        _log(log_fn, f"Candidate {model.candidate_id}: predict error: {exc}")
    return 0.0


def _score_loaded_candidate_frame(
    model: LoadedCandidateModel,
    df: pd.DataFrame,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    row_mask: Optional[pd.Series] = None,
) -> str:
    score_col = _safe_score_column_name(model.candidate_id)
    target_index = df.index if row_mask is None else df.index[pd.Series(row_mask, index=df.index).fillna(False)]
    df[score_col] = 0.0
    if len(target_index) == 0:
        model.score_column = score_col
        _log(log_fn, f"Candidate {model.candidate_id}: wrote empty score column {score_col} (0 eligible rows)")
        return score_col
    X = _build_model_feature_matrix(df.loc[target_index], model.feature_cols)
    est = model.estimator
    X_input = X if hasattr(est, "feature_names_in_") else X.to_numpy(dtype="float32")
    try:
        if hasattr(est, "predict_proba"):
            probs = est.predict_proba(X_input)
            classes = list(getattr(est, "classes_", []))
            idx = 1
            if classes:
                idx = classes.index(1) if 1 in classes else min(1, len(classes) - 1)
            vals = [float(row[idx] if hasattr(row, "__len__") else row) for row in probs]
        elif hasattr(est, "decision_function"):
            raw = est.decision_function(X_input)
            vals = []
            for item in raw:
                val = item[-1] if hasattr(item, "__len__") and not isinstance(item, (str, bytes)) else item
                vals.append(_sigmoid(float(val)))
        elif hasattr(est, "predict"):
            preds = est.predict(X_input)
            vals = [max(0.0, min(1.0, _safe_float(v, 0.0))) for v in preds]
            _log(log_fn, f"Candidate {model.candidate_id}: using predict() fallback for frame scoring")
        else:
            raise RuntimeError("model has no predict_proba, decision_function, or predict")
        df.loc[target_index, score_col] = vals
        model.score_column = score_col
        _log(log_fn, f"Candidate {model.candidate_id}: wrote score column {score_col} rows={len(target_index):,}/{len(df):,}")
        return score_col
    except Exception as exc:
        model.rejected_reason = f"score_failed:{exc}"
        _log(log_fn, f"Candidate {model.candidate_id}: frame scoring failed: {exc}")
        return ""


def _score_candidate_worker(args: Tuple[LoadedCandidateModel, pd.DataFrame, Optional[List[Any]]]) -> Tuple[str, List[Any], List[float], str]:
    model, frame, index_values = args
    score_col = _safe_score_column_name(model.candidate_id)
    try:
        local = frame.copy()
        _score_loaded_candidate_frame(model, local, row_mask=None, log_fn=None)
        vals = [float(x) for x in local[score_col].tolist()]
        return model.candidate_id, list(index_values or local.index.tolist()), vals, ""
    except Exception as exc:
        return model.candidate_id, list(index_values or frame.index.tolist()), [], str(exc)


def _get_row_score(row: pd.Series, bundles: List[Tuple[str, Any]], feature_name_lists: List[List[str]], log_fn: Optional[Callable[[str], None]] = None) -> Tuple[float, str]:
    """Return (best_score, best_candidate_id). Uses CSV model_score if present and no bundles, else model(s)."""
    # If we have precomputed score in data and no bundles loaded, trust it (common for pre-scored exports)
    if not bundles:
        if "model_score" in row and pd.notna(row["model_score"]):
            return _safe_float(row["model_score"]), "csv_score"
        for alt in ("probability", "proba", "pred_proba", "score"):
            if alt in row and pd.notna(row[alt]):
                return _safe_float(row[alt]), "csv_score"
        return 0.0, "no_score"

    # Score with each candidate's model; pick highest (or could do per-cand separate runs; here we take best signal)
    best = 0.0
    best_id = "none"
    for (cid, bundle), fnames in zip(bundles, feature_name_lists):
        sc = _predict_score(bundle, row, fnames, log_fn)
        if sc > best:
            best = sc
            best_id = cid or "model"
    return best, best_id


def _compute_pnl_and_cost(entry_price: float, exit_price: float, premium_for_cost: float, config: Optional[Dict[str, Any]] = None) -> Tuple[float, float, float]:
    """gross_pnl (rupee per lot), cost_units, net_pnl."""
    gross = (exit_price - entry_price) * LOT_SIZE
    try:
        cost_info = estimate_option_execution_cost({"ltp": premium_for_cost or entry_price}, config=config)
        cost = float(cost_info.get("cost_return_units") or cost_info.get("total_estimated_roundtrip_cost") or 0.0)
    except Exception:
        cost = max(1.0, (premium_for_cost or entry_price) * 0.005 * 2)  # rough
    net = gross - cost
    return gross, cost, net


def _build_equity_curve(trades: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    """Return (net_pnl, max_dd_abs, max_dd_pct) from trade list (in order)."""
    if not trades:
        return 0.0, 0.0, 0.0
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cum += _safe_float(t.get("net_pnl", 0.0))
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    final_net = cum
    # pct dd rough vs peak or vs starting 0
    max_dd_pct = (max_dd / peak) if peak > 0 else 0.0
    return final_net, max_dd, max_dd_pct


def _profit_factor(trades: List[Dict[str, Any]]) -> float:
    wins = sum(_safe_float(t.get("net_pnl")) for t in trades if _safe_float(t.get("net_pnl")) > 0)
    losses = abs(sum(_safe_float(t.get("net_pnl")) for t in trades if _safe_float(t.get("net_pnl")) < 0))
    if losses < 1e-9:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def _sharpe_like(trades: List[Dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    pnls = [_safe_float(t.get("net_pnl")) for t in trades]
    import numpy as np  # optional; fallback if absent
    try:
        mu = float(np.mean(pnls))
        sd = float(np.std(pnls))
        if sd < 1e-12:
            return 0.0
        return (mu / sd) * math.sqrt(max(1, len(pnls)))
    except Exception:
        # pure python
        n = len(pnls)
        if n < 2:
            return 0.0
        mu = sum(pnls) / n
        var = sum((p - mu) ** 2 for p in pnls) / (n - 1)
        sd = math.sqrt(var) if var > 0 else 0.0
        return (mu / sd) * math.sqrt(n) if sd > 0 else 0.0


@dataclass
class BacktestResult:
    summary: Dict[str, Any]
    trades: List[Dict[str, Any]]
    output_paths: Dict[str, str]
    stopped: bool = False


class _Timer:
    def __init__(self) -> None:
        self.sections: Dict[str, float] = defaultdict(float)

    def track(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.start = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc, tb):
                timer.sections[name] += time.perf_counter() - self.start

        return _Ctx()

    def summary(self) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in self.sections.items()}


def run_historical_ml_backtest(
    csv_path: str,
    candidate_config_path: Optional[str] = None,
    output_dir: str = "reports/backtests",
    threshold: float = DEFAULT_THRESHOLD,
    target_pct: float = DEFAULT_TARGET_PCT,
    stoploss_pct: float = DEFAULT_STOPLOSS_PCT,
    max_hold_bars: int = DEFAULT_MAX_HOLD_BARS,
    max_trades_per_day: int = DEFAULT_MAX_TRADES_PER_DAY,
    stop_event: Optional[threading.Event] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_fn: Optional[Callable[[float, str], None]] = None,
    debug: bool = False,
    use_candidate_thresholds: bool = True,
    strict_artifacts: bool = True,
    allow_artifact_fallback: bool = False,
    ml_artifact_scoring: bool = True,
    allow_embedded_score_fallback: Optional[bool] = None,
    full_csv_read_fallback: bool = False,
    selected_candidate_ids: Optional[Iterable[str]] = None,
    fast_mode: bool = False,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    max_rows: Optional[int] = None,
    max_candidates: Optional[int] = None,
    skip_verbose_logs: bool = False,
    multiprocessing: bool = False,
    progress_every: int = 5000,
) -> BacktestResult:
    """
    Main entry. Reads CSV, (optionally) loads candidates/models, runs threshold backtest, writes artefacts, returns result.
    """
    stopped = False
    timer = _Timer()
    if skip_verbose_logs and log_fn is not None:
        real_log_fn = log_fn

        def quiet_log(line: str) -> None:
            text = str(line)
            important = (
                "Starting " in text
                or text.startswith("Loaded ")
                or text.startswith("Using ")
                or text.startswith("Wrote ")
                or text.startswith("COMPLETE")
                or text.startswith("FAILED")
                or "ERROR" in text
                or "Timing" in text
            )
            if important:
                real_log_fn(text)

        log_fn = quiet_log
    root = _project_root()
    csv_abs = _resolve_path(csv_path, root=root) or Path(csv_path).resolve()
    data_abs = _preferred_data_path(csv_abs, log_fn)
    config_abs = _resolve_path(candidate_config_path, root=root) if candidate_config_path else None
    out_dir = _resolve_path(output_dir, root=root) or (root / output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _now_stamp()
    trades_out = out_dir / f"backtest_trades_{stamp}.csv"
    summary_out = out_dir / f"backtest_summary_{stamp}.json"
    report_out = out_dir / f"backtest_report_{stamp}.md"

    _progress(progress_fn, 0.0, "Starting backtest...")
    _log(log_fn, f"Starting historical ML backtest")
    _log(log_fn, f"  Project root: {root}")
    _log(log_fn, f"  CSV: {csv_abs}")
    if data_abs != csv_abs:
        _log(log_fn, f"  Data source: {data_abs}")
    _log(log_fn, f"  Config: {config_abs or '(none)'}")
    _log(log_fn, f"  Output: {out_dir.resolve()}")
    if allow_embedded_score_fallback is None:
        allow_embedded_score_fallback = candidate_config_path is None
    _log(log_fn, f"  Params: fallback_threshold={threshold:.2f} use_candidate_thresholds={use_candidate_thresholds} strict_artifacts={strict_artifacts} allow_artifact_fallback={allow_artifact_fallback} ml_artifact_scoring={ml_artifact_scoring} allow_embedded_score_fallback={allow_embedded_score_fallback} full_csv_read_fallback={full_csv_read_fallback} fast_mode={fast_mode} max_rows={max_rows} max_candidates={max_candidates} multiprocessing={multiprocessing} target={target_pct*100:.1f}% sl={stoploss_pct*100:.1f}% hold={max_hold_bars} max/day={max_trades_per_day} debug={debug}")

    # --- Load candidates / models
    _progress(progress_fn, 0.02, "Loading ML candidates...")
    cfg = _load_candidate_config(str(config_abs) if config_abs else None, log_fn)
    raw_cands = cfg.get("candidates", []) or []
    enabled_cands = [c for c in raw_cands if c.get("enabled", True)]
    enabled_cands = _filter_candidates_by_ids(enabled_cands, selected_candidate_ids)
    if max_candidates is not None and int(max_candidates) > 0:
        enabled_cands = enabled_cands[: int(max_candidates)]
    loaded_models: List[LoadedCandidateModel] = []
    artifact_failures: List[ArtifactLoadFailure] = []
    config_dir = config_abs.parent if config_abs and config_abs.exists() and config_abs.is_file() else None
    scoring_cands = enabled_cands[:20] if ml_artifact_scoring else []
    artifact_cache: Dict[str, Optional[LoadedCandidateModel]] = {}
    for idx, c in enumerate(scoring_cands):  # safety cap
        _progress(
            progress_fn,
            0.02 + 0.06 * ((idx + 1) / max(len(scoring_cands), 1)),
            f"Loading candidate {idx + 1}/{max(len(scoring_cands), 1)}...",
        )
        cid = str(c.get("candidate_id") or c.get("model_name") or c.get("id") or "cand")
        _log(log_fn, f"Candidate {cid}: final threshold={_candidate_threshold(c, threshold, use_candidate_thresholds=use_candidate_thresholds):.4f}")
        cache_key = json.dumps(c, sort_keys=True, default=str)
        if cache_key in artifact_cache:
            model = artifact_cache[cache_key]
            failures = []
        else:
            with timer.track("artifact_load_time"):
                model, failures = _load_candidate_model(
                    c,
                    root=root,
                    config_dir=config_dir,
                    fallback_threshold=threshold,
                    use_candidate_thresholds=use_candidate_thresholds,
                    log_fn=log_fn,
                    strict_artifacts=strict_artifacts,
                    allow_artifact_fallback=allow_artifact_fallback,
                    debug=debug,
                )
            artifact_cache[cache_key] = model
            artifact_failures.extend(failures)
        if model:
            loaded_models.append(model)
    if loaded_models:
        _log(log_fn, f"Loaded {len(loaded_models)} ML model artifact(s); validating feature alignment after CSV load")
    else:
        if allow_embedded_score_fallback:
            _log(log_fn, "No loadable ML models; will rely on CSV-embedded score columns if present.")
        else:
            _log(log_fn, "No loadable ML models and embedded score fallback is disabled.")

    # --- Load & normalize CSV
    _progress(progress_fn, 0.10, "Reading dataset header...")
    try:
        with timer.track("csv_load_time"):
            header_cols = _read_data_header(data_abs)
    except Exception as exc:
        raise RuntimeError(f"Failed to read data header: {exc}")
    if loaded_models and not any(m.feature_cols for m in loaded_models) and full_csv_read_fallback:
        planned_cols = list(header_cols)
        _log(log_fn, "Full CSV read fallback enabled because no loaded artifact exposed feature_order")
    else:
        planned_cols = _planned_csv_columns(
            header_cols,
            loaded_models,
            include_score_columns=bool(allow_embedded_score_fallback),
        )
    _log(log_fn, f"Planned CSV columns: reading {len(planned_cols)} of {len(header_cols)} total columns")
    if debug:
        _log(log_fn, f"CSV columns requested (first 80): {planned_cols[:80]}")

    _progress(progress_fn, 0.12, "Loading dataset rows...")
    with timer.track("csv_load_time"):
        df = _read_backtest_frame(data_abs, usecols=planned_cols, log_fn=log_fn)
    original_rows = len(df)
    _progress(progress_fn, 0.18, f"Loaded {original_rows:,} rows. Preparing timestamps...")
    _log(log_fn, f"Loaded {len(df)} rows, {len(df.columns)} columns")
    df = _normalize_columns(df)
    selected_score_col = _select_score_column(df)
    score_diagnostics = _log_score_diagnostics(df, selected_score_col=selected_score_col, threshold=threshold, log_fn=log_fn)

    missing: List[str] = []
    if "timestamp" not in df.columns:
        missing.append("timestamp (or datetime/time/date)")
    if "ltp" not in df.columns:
        missing.append("ltp (or close/price/premium)")
    if not ("symbol" in df.columns or "strike" in df.columns):
        missing.append("symbol/trading_symbol or strike+option_type+expiry (for contract tracking)")

    if missing:
        avail = ", ".join(sorted(df.columns)[:30])
        raise RuntimeError(f"Missing required columns: {missing}. Available (first 30): {avail}")

    # Parse times
    _progress(progress_fn, 0.19, "Parsing timestamps and sorting rows...")
    with timer.track("feature_build_time"):
        df["ts"] = df["timestamp"].apply(_parse_timestamp)
    valid_timestamp_rows = int(df["ts"].notna().sum())
    df = df[df["ts"].notna()].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No valid timestamps after parsing.")

    if fast_mode:
        if date_from:
            start = pd.to_datetime(date_from, errors="coerce", utc=True)
            if pd.notna(start):
                df = df[df["ts"] >= start].copy()
        if date_to:
            end = pd.to_datetime(date_to, errors="coerce", utc=True)
            if pd.notna(end):
                df = df[df["ts"] <= end].copy()
        if max_rows is not None and int(max_rows) > 0 and len(df) > int(max_rows):
            df = df.head(int(max_rows)).copy()
        df = df.reset_index(drop=True)
        _log(log_fn, f"Fast mode data slice rows={len(df):,} date_from={date_from or '-'} date_to={date_to or '-'} max_rows={max_rows or '-'}")
        if df.empty:
            raise RuntimeError("No rows after fast-mode date/max-row filters.")

    _progress(progress_fn, 0.20, "Building contract keys and validating prices...")
    with timer.track("feature_build_time"):
        df["date"] = df["ts"].dt.date
        df["_contract"] = df.apply(_ensure_contract_key, axis=1)
    price_series = pd.to_numeric(df["ltp"], errors="coerce") if "ltp" in df.columns else pd.Series([], dtype=float)
    valid_price_rows = int((price_series > 0).sum()) if len(price_series) else 0
    _progress(progress_fn, 0.22, "Materializing offline model features. This can take a while on large datasets...")
    required_feature_union = sorted({f for model in loaded_models for f in (model.feature_cols or [])})
    with timer.track("feature_build_time"):
        feature_materialization = _materialize_offline_model_features(df, required_feature_union, log_fn=log_fn)
        _progress(progress_fn, 0.24, "Applying entry prefilter and preparing model input frame...")
        prediction_prefilter_mask = _entry_prefilter_mask(df)
    _log(log_fn, f"Prediction prefilter eligible rows={int(prediction_prefilter_mask.sum()):,}/{len(df):,}")

    validated_models: List[LoadedCandidateModel] = []
    feature_alignment_payload: List[Dict[str, Any]] = []
    for model in loaded_models:
        diag = _feature_alignment_diagnostics(model, df, log_fn=log_fn)
        feature_alignment_payload.append(diag)
        required = int(diag["required_feature_count"])
        missing_count = int(diag["missing_feature_count"])
        if required == 0:
            model.rejected_reason = "MODEL_FEATURES_MISSING"
            if debug:
                _log(log_fn, "MODEL_FEATURES_MISSING")
                _log(log_fn, f"  artifact_path={model.artifact_path}")
                _log(log_fn, f"  artifact_type={model.metadata.get('artifact_type', '')}")
                _log(log_fn, f"  top_level_keys={model.metadata.get('top_level_keys', [])}")
                _log(log_fn, f"  model_type={model.model_type or type(model.estimator).__name__}")
                _log(log_fn, f"  sidecar_files_found={model.metadata.get('sidecar_files_found', [])}")
                _log(log_fn, "  suggestion=write feature_order/feature_names into the model bundle or a sidecar in the artifact directory")
            artifact_failures.append(ArtifactLoadFailure(model.candidate_id, str(model.artifact_path), model.rejected_reason))
            continue
        if missing_count == 0:
            validated_models.append(model)
        else:
            if missing_count <= 3 or missing_count / max(required, 1) <= 0.05:
                _log(log_fn, f"Candidate {model.candidate_id}: filling {missing_count} missing feature(s) with 0.0 for offline diagnostics")
                validated_models.append(model)
                continue
            model.rejected_reason = f"too_many_missing_features:{missing_count}/{required}"
            artifact_failures.append(
                ArtifactLoadFailure(
                    model.candidate_id,
                    str(model.artifact_path),
                    model.rejected_reason or "feature alignment failed",
                )
            )
    loaded_models = validated_models
    if loaded_models:
        _log(log_fn, f"Using {len(loaded_models)} loadable ML model(s) for scoring")
    else:
        _log(log_fn, "No feature-aligned ML models after CSV load.")

    scored_models: List[LoadedCandidateModel] = []
    eligible_idx = list(df.index[prediction_prefilter_mask])
    used_parallel_scoring = False
    if multiprocessing and len(loaded_models) > 1 and eligible_idx:
        try:
            workers = min(len(loaded_models), max(1, (os.cpu_count() or 2) - 1))
            _log(log_fn, f"Scoring {len(loaded_models)} candidate(s) with multiprocessing workers={workers}")
            scored_by_id: Dict[str, Tuple[List[Any], List[float], str]] = {}
            with timer.track("prediction_time"):
                _progress(progress_fn, 0.25, f"Scoring {len(loaded_models)} model(s) across {len(eligible_idx):,} eligible rows...")
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(_score_candidate_worker, (model, df.loc[eligible_idx, :], eligible_idx)): model
                        for model in loaded_models
                    }
                    done = 0
                    for fut in as_completed(futures):
                        done += 1
                        model = futures[fut]
                        _progress(progress_fn, 0.25 + 0.15 * (done / max(len(loaded_models), 1)), f"Scored candidate {done}/{len(loaded_models)}...")
                        cid, idx_values, vals, err = fut.result()
                        scored_by_id[cid] = (idx_values, vals, err)
            for model in loaded_models:
                idx_values, vals, err = scored_by_id.get(model.candidate_id, ([], [], "missing worker result"))
                score_col = _safe_score_column_name(model.candidate_id)
                df[score_col] = 0.0
                if err:
                    model.rejected_reason = f"score_failed:{err}"
                    artifact_failures.append(ArtifactLoadFailure(model.candidate_id, str(model.artifact_path), model.rejected_reason))
                    continue
                df.loc[idx_values, score_col] = vals
                model.score_column = score_col
                scored_models.append(model)
            used_parallel_scoring = True
        except Exception as exc:
            _log(log_fn, f"Multiprocessing candidate scoring failed; falling back to serial scoring: {exc}")

    if not used_parallel_scoring:
        for idx, model in enumerate(loaded_models):
            _progress(
                progress_fn,
                0.25 + 0.15 * ((idx + 1) / max(len(loaded_models), 1)),
                f"Scoring {len(eligible_idx):,} eligible rows with {model.candidate_id}...",
            )
            with timer.track("prediction_time"):
                score_col = _score_loaded_candidate_frame(model, df, log_fn=log_fn, row_mask=prediction_prefilter_mask)
            if score_col:
                scored_models.append(model)
            else:
                artifact_failures.append(ArtifactLoadFailure(model.candidate_id, str(model.artifact_path), model.rejected_reason or "score_failed"))
    loaded_models = scored_models

    if candidate_config_path and not loaded_models and not allow_embedded_score_fallback:
        status = "FAILED - 0 MODEL-READY CANDIDATES"
        _log(log_fn, status)
        artifact_failure_payload = [asdict(f) for f in artifact_failures]
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "csv_path": str(csv_abs),
            "candidate_config_path": str(config_abs) if config_abs else None,
            "parameters": {
                "fallback_threshold": threshold,
                "use_candidate_thresholds": use_candidate_thresholds,
                "target_pct": target_pct,
                "stoploss_pct": stoploss_pct,
                "max_hold_bars": max_hold_bars,
                "max_trades_per_day": max_trades_per_day,
                "lot_size": LOT_SIZE,
                "ml_artifact_scoring": ml_artifact_scoring,
                "allow_embedded_score_fallback": allow_embedded_score_fallback,
                "full_csv_read_fallback": full_csv_read_fallback,
                "fast_mode": fast_mode,
                "date_from": date_from,
                "date_to": date_to,
                "max_rows": max_rows,
                "max_candidates": max_candidates,
                "selected_candidate_ids": list(selected_candidate_ids or []),
                "skip_verbose_logs": skip_verbose_logs,
                "multiprocessing": multiprocessing,
                "progress_every": progress_every,
            },
            "total_trades": 0,
            "loaded_model_count": 0,
            "model_ready_count": 0,
            "feature_alignment": feature_alignment_payload,
            "feature_materialization": feature_materialization,
            "artifact_loading_failures": artifact_failure_payload,
            "selected_score_column": None,
            "rows_above_threshold": 0,
            "rejection_counts": {},
            "rows_processed": 0,
            "timing_seconds": timer.summary(),
        }
        with open(trades_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["candidate_id", "entry_ts", "net_pnl"])
            w.writeheader()
        summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        report_out.write_text(f"# ML Historical Backtest Report\n\n{status}\n", encoding="utf-8")
        _progress(progress_fn, 1.0, status)
        return BacktestResult(
            summary=summary,
            trades=[],
            output_paths={"trades_csv": str(trades_out), "summary_json": str(summary_out), "report_md": str(report_out)},
            stopped=False,
        )

    score_candidates: List[LoadedCandidateModel] = []
    if not loaded_models and allow_embedded_score_fallback:
        score_cols = _score_like_columns(df)
        numeric_score_cols = [c for c in score_cols if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
        ordered_score_cols = sorted(numeric_score_cols, key=_score_column_priority)
        for score_col in ordered_score_cols:
            score_candidates.append(
                LoadedCandidateModel(
                    candidate_id=f"csv_score::{score_col}",
                    artifact_path=csv_abs,
                    estimator=None,
                    feature_cols=[],
                    threshold=float(threshold),
                    metadata={"source": "embedded_csv_score"},
                    filters=cfg.get("global_filters") or {},
                    score_column=score_col,
                )
            )
        if score_candidates:
            _log(log_fn, f"Using CSV score-column candidate(s): {[c.candidate_id for c in score_candidates]}")
        elif selected_score_col:
            _log(log_fn, f"Selected score column {selected_score_col} was not numeric enough to backtest")

    # --- Simulation state
    open_trades: Dict[str, Dict[str, Any]] = {}  # key -> trade dict
    closed_trades: List[Dict[str, Any]] = []
    daily_count: Dict[Any, int] = defaultdict(int)
    last_date: Any = None
    bars_processed = 0
    rejection_counts: Dict[str, int] = defaultdict(int)
    first_rejections: List[Dict[str, Any]] = []
    top_scores: List[Dict[str, Any]] = []
    rows_above_threshold = 0
    selected_signal_sources = [m.candidate_id for m in loaded_models] or [m.candidate_id for m in score_candidates]

    # Precompute cost config
    cost_cfg = {"roundtrip": True, "lot_size": LOT_SIZE}

    def record_rejection(reason: str, row: pd.Series, score: float = 0.0, cid: str = "") -> None:
        rejection_counts[reason] += 1
        if len(first_rejections) < 20:
            first_rejections.append(
                {
                    "reason": reason,
                    "timestamp": str(row.get("ts") or row.get("timestamp") or ""),
                    "symbol": row.get("symbol"),
                    "option_type": row.get("option_type"),
                    "strike": row.get("strike"),
                    "expiry": row.get("expiry"),
                    "price": _get_price(row),
                    "score": round(float(score), 6) if math.isfinite(float(score or 0.0)) else score,
                    "candidate_id": cid,
                }
            )

    def remember_top_score(row: pd.Series, score: float, cid: str, threshold_used: float) -> None:
        if not math.isfinite(float(score or 0.0)):
            return
        top_scores.append(
            {
                "score": float(score),
                "threshold": float(threshold_used),
                "candidate_id": cid,
                "timestamp": str(row.get("ts") or row.get("timestamp") or ""),
                "symbol": row.get("symbol"),
                "option_type": row.get("option_type"),
                "strike": row.get("strike"),
                "expiry": row.get("expiry"),
                "price": _get_price(row),
            }
        )
        if len(top_scores) > 200:
            top_scores.sort(key=lambda x: _safe_float(x.get("score")), reverse=True)
            del top_scores[20:]

    def best_signal_for_row(row: pd.Series) -> Tuple[float, str, float, Dict[str, Any], Optional[List[str]], Optional[int]]:
        best_score = -1.0
        best_cid = "no_score"
        best_threshold = float(threshold)
        best_filters: Dict[str, Any] = {}
        best_otypes: Optional[List[str]] = None
        best_max_trades: Optional[int] = None
        candidates = loaded_models or score_candidates
        for model in candidates:
            if model.score_column:
                score = _safe_float(row.get(model.score_column), 0.0)
            else:
                score = _predict_loaded_candidate_score(model, row, log_fn=log_fn)
            remember_top_score(row, score, model.candidate_id, model.threshold)
            ratio = score / model.threshold if model.threshold > 0 else score
            best_ratio = best_score / best_threshold if best_threshold > 0 and best_score >= 0 else -1.0
            if ratio > best_ratio:
                best_score = score
                best_cid = model.candidate_id
                best_threshold = model.threshold
                best_filters = model.filters or {}
                best_otypes = model.option_type_filter
                best_max_trades = model.max_trades_per_day
        if not candidates:
            return 0.0, "no_score", float(threshold), {}, None, None
        return max(0.0, best_score), best_cid, best_threshold, best_filters, best_otypes, best_max_trades

    # Iterate rows chronologically
    total_rows = len(df)
    progress_interval = max(1, int(progress_every or 0) or (total_rows // 100) or 1)
    _progress(progress_fn, 0.40, f"Model scoring complete. Simulating trades (0/{total_rows:,} rows)...")
    with timer.track("simulation_time"):
        for idx, row in df.iterrows():
            if stop_event is not None and stop_event.is_set():
                stopped = True
                _log(log_fn, "STOP requested - finishing current bar and saving partial results")
                break

            bars_processed += 1
            if (
                bars_processed == 1
                or bars_processed == total_rows
                or bars_processed % progress_interval == 0
            ):
                sim_frac = 0.40 + 0.55 * (bars_processed / max(total_rows, 1))
                _progress(
                    progress_fn,
                    sim_frac,
                    f"Simulating row {bars_processed:,}/{total_rows:,} "
                    f"(open={len(open_trades)}, closed={len(closed_trades)})",
                )
            if not skip_verbose_logs and bars_processed % max(1, int(progress_every or 5000)) == 0:
                _log(log_fn, f"  processed {bars_processed}/{total_rows} rows, open={len(open_trades)}, closed={len(closed_trades)}")

            cur_ts = row["ts"]
            cur_date = row["date"]
            price = _get_price(row)
            if price <= 0:
                record_rejection("price_valid", row, 0.0, "")
                continue

            contract = row["_contract"]

            # 1) Manage exits for matching open trades
            to_close = []
            for tkey, tr in list(open_trades.items()):
                if tr["contract"] != contract:
                    continue
                if cur_ts <= tr["entry_ts"]:
                    continue
                exit_price = price
                held = tr.get("bars_held", 0) + 1
                tr["bars_held"] = held
                tr["last_price"] = exit_price

                ret = (exit_price / tr["entry_price"] - 1.0) if tr["entry_price"] > 0 else 0.0
                reason = None
                if ret >= target_pct:
                    reason = "target"
                elif ret <= -stoploss_pct:
                    reason = "stoploss"
                elif held >= max_hold_bars:
                    reason = "max_hold"
                elif cur_date != tr["entry_date"]:
                    reason = "eod"

                if reason:
                    g, cst, n = _compute_pnl_and_cost(tr["entry_price"], exit_price, tr["entry_price"], cost_cfg)
                    trade_rec = {
                        "candidate_id": tr["candidate_id"],
                        "entry_ts": str(tr["entry_ts"]),
                        "exit_ts": str(cur_ts),
                        "symbol": tr.get("symbol"),
                        "option_type": tr.get("option_type"),
                        "strike": tr.get("strike"),
                        "expiry": tr.get("expiry"),
                        "entry_price": round(tr["entry_price"], 4),
                        "exit_price": round(exit_price, 4),
                        "gross_pnl": round(g, 2),
                        "cost": round(cst, 2),
                        "net_pnl": round(n, 2),
                        "exit_reason": reason,
                        "model_score": round(tr["model_score"], 4),
                        "bars_held": held,
                        "filters_passed": tr.get("filters_passed", ""),
                    }
                    closed_trades.append(trade_rec)
                    to_close.append(tkey)
                    daily_count[cur_date] = daily_count.get(cur_date, 0)  # already counted at entry

            for k in to_close:
                open_trades.pop(k, None)

            # 2) Consider new entry from this row (after processing exits at this price level)
            if price <= 0:
                continue

            score, cid, row_threshold, filt, option_type_filter, row_max_trades = best_signal_for_row(row)
            if not selected_signal_sources:
                record_rejection("no_model_or_score", row, score, cid)
                continue
            if score < row_threshold:
                record_rejection("score_threshold", row, score, cid)
                continue
            rows_above_threshold += 1

            if option_type_filter:
                ot = str(row.get("option_type", "")).upper()[:2]
                if ot not in option_type_filter:
                    record_rejection("option_type_filter", row, score, cid)
                    continue

            filt_ok, bad = _passes_filters(row, filt, log_fn)
            if not filt_ok:
                for reason in bad:
                    if reason.startswith("spread"):
                        record_rejection("spread_valid", row, score, cid)
                    elif reason.startswith("prem"):
                        record_rejection("premium_valid", row, score, cid)
                    elif reason == "otype_blocked":
                        record_rejection("option_type_valid", row, score, cid)
                    else:
                        record_rejection(f"filter:{reason}", row, score, cid)
                continue

            effective_max_trades = max_trades_per_day
            if row_max_trades is not None and row_max_trades > 0:
                effective_max_trades = min(effective_max_trades, int(row_max_trades))
            if daily_count[cur_date] >= effective_max_trades:
                record_rejection("max_trades_per_day", row, score, cid)
                continue

            # open new
            tkey = f"{contract}|{cur_ts.isoformat()}"
            open_trades[tkey] = {
                "contract": contract,
                "candidate_id": cid,
                "entry_ts": cur_ts,
                "entry_date": cur_date,
                "entry_price": price,
                "model_score": score,
                "threshold": row_threshold,
                "symbol": row.get("symbol"),
                "option_type": row.get("option_type"),
                "strike": _safe_float(row.get("strike"), 0),
                "expiry": row.get("expiry"),
                "bars_held": 0,
                "filters_passed": "ok" if not bad else ",".join(bad),
            }
            daily_count[cur_date] += 1
            if not skip_verbose_logs and (len(closed_trades) < 50 or (bars_processed % 2000 == 0)):
                _log(log_fn, f"  ENTRY {cid} score={score:.3f} threshold={row_threshold:.3f} @ {price:.2f} {contract} (day trades={daily_count[cur_date]})")

    if stopped:
        _log(log_fn, "Backtest halted by stop flag.")

    # Close any remaining at last known price (conservative)
    if open_trades:
        _log(log_fn, f"Closing {len(open_trades)} open positions at end-of-data (EOD forced)")
        last_price_map: Dict[str, float] = {}
        for _, r in df.tail(200).iterrows():  # rough last prices
            last_price_map[r["_contract"]] = _get_price(r)
        for tkey, tr in list(open_trades.items()):
            ep = last_price_map.get(tr["contract"], tr["entry_price"])
            g, cst, n = _compute_pnl_and_cost(tr["entry_price"], ep, tr["entry_price"], cost_cfg)
            rec = {
                "candidate_id": tr["candidate_id"],
                "entry_ts": str(tr["entry_ts"]),
                "exit_ts": "EOD",
                "symbol": tr.get("symbol"),
                "option_type": tr.get("option_type"),
                "strike": tr.get("strike"),
                "expiry": tr.get("expiry"),
                "entry_price": round(tr["entry_price"], 4),
                "exit_price": round(ep, 4),
                "gross_pnl": round(g, 2),
                "cost": round(cst, 2),
                "net_pnl": round(n, 2),
                "exit_reason": "eod_forced",
                "model_score": round(tr["model_score"], 4),
                "bars_held": tr.get("bars_held", 0),
                "filters_passed": tr.get("filters_passed", ""),
            }
            closed_trades.append(rec)
        open_trades.clear()

    # --- Metrics
    total = len(closed_trades)
    wins = sum(1 for t in closed_trades if _safe_float(t.get("net_pnl")) > 0)
    losses = sum(1 for t in closed_trades if _safe_float(t.get("net_pnl")) < 0)
    win_rate = (wins / total) if total > 0 else 0.0
    gross_sum = sum(_safe_float(t.get("gross_pnl")) for t in closed_trades)
    net_sum = sum(_safe_float(t.get("net_pnl")) for t in closed_trades)
    pf = _profit_factor(closed_trades)
    _, max_dd_abs, max_dd_pct = _build_equity_curve(closed_trades)
    avg_trade = (net_sum / total) if total > 0 else 0.0
    sharpe = _sharpe_like(closed_trades)

    # best candidate by net contrib
    by_cand: Dict[str, float] = defaultdict(float)
    for t in closed_trades:
        by_cand[t.get("candidate_id", "unknown")] += _safe_float(t.get("net_pnl"))
    best_cand = max(by_cand, key=by_cand.get) if by_cand else "n/a"
    top_scores.sort(key=lambda x: _safe_float(x.get("score")), reverse=True)
    top_20_scores = top_scores[:20]
    artifact_failure_payload = [asdict(f) for f in artifact_failures]
    selected_score_label = selected_score_col or ("artifact_model_scores" if loaded_models else None)
    zero_trade_diagnostics = {
        "total_rows": original_rows,
        "valid_timestamp_rows": valid_timestamp_rows,
        "valid_price_rows": valid_price_rows,
        "selected_price_column": "ltp" if "ltp" in df.columns else None,
        "selected_timestamp_column": "timestamp" if "timestamp" in df.columns else None,
        "selected_score_column": selected_score_label,
        "rows_above_threshold": rows_above_threshold,
        "top_20_scores": top_20_scores,
        "filter_rejection_counts": dict(rejection_counts),
        "first_20_rejections": first_rejections,
        "candidate_artifact_loading_failures": artifact_failure_payload,
        "recommendations": [
            "lower threshold or enable candidate-specific thresholds",
            "select the correct artifact directory or repair candidate config paths",
            "run the retrainer/export step to materialize model.pkl/joblib artifacts",
            "use a CSV with embedded score/probability columns",
            "repair config/paper_forward_candidates.json artifact_path/model_path fields",
        ],
    }

    timing_before_write = timer.summary()
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(csv_abs),
        "data_path": str(data_abs),
        "candidate_config_path": str(config_abs) if config_abs else None,
        "parameters": {
            "fallback_threshold": threshold,
            "use_candidate_thresholds": use_candidate_thresholds,
            "target_pct": target_pct,
            "stoploss_pct": stoploss_pct,
            "max_hold_bars": max_hold_bars,
            "max_trades_per_day": max_trades_per_day,
            "lot_size": LOT_SIZE,
            "ml_artifact_scoring": ml_artifact_scoring,
            "allow_embedded_score_fallback": allow_embedded_score_fallback,
            "full_csv_read_fallback": full_csv_read_fallback,
            "fast_mode": fast_mode,
            "date_from": date_from,
            "date_to": date_to,
            "max_rows": max_rows,
            "max_candidates": max_candidates,
            "selected_candidate_ids": list(selected_candidate_ids or []),
            "skip_verbose_logs": skip_verbose_logs,
            "multiprocessing": multiprocessing,
            "progress_every": progress_every,
        },
        "status": "COMPLETE" if total > 0 else "COMPLETE - 0 TRADES",
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "gross_pnl": round(gross_sum, 2),
        "net_pnl": round(net_sum, 2),
        "profit_factor": round(pf, 4) if math.isfinite(pf) else None,
        "max_drawdown": round(max_dd_abs, 2),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "avg_trade": round(avg_trade, 2),
        "sharpe_like": round(sharpe, 4),
        "best_candidate": best_cand,
        "candidates_used": list(by_cand.keys()),
        "signal_sources_considered": selected_signal_sources,
        "loaded_model_count": len(loaded_models),
        "model_ready_count": len(loaded_models),
        "feature_alignment": feature_alignment_payload,
        "feature_materialization": feature_materialization,
        "csv_score_candidate_count": len(score_candidates),
        "selected_score_column": selected_score_label,
        "rows_above_threshold": rows_above_threshold,
        "rejected_by_filters": int(sum(v for k, v in rejection_counts.items() if k != "score_threshold")),
        "rejection_counts": dict(rejection_counts),
        "score_diagnostics": score_diagnostics,
        "artifact_loading_failures": artifact_failure_payload,
        "zero_trade_diagnostics": zero_trade_diagnostics if total == 0 else {},
        "stopped_early": stopped,
        "rows_processed": bars_processed,
        "prediction_prefilter_rows": int(prediction_prefilter_mask.sum()),
        "timing_seconds": timing_before_write,
    }

    # --- Persist artefacts
    _progress(progress_fn, 0.95, "Writing reports...")
    with timer.track("report_write_time"):
        if closed_trades:
            pd.DataFrame(closed_trades).to_csv(trades_out, index=False)
            _log(log_fn, f"Wrote trades: {trades_out}")
        else:
            # write empty
            with open(trades_out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["candidate_id", "entry_ts", "net_pnl"])
                w.writeheader()
            _log(log_fn, "No trades generated; wrote empty trades csv")

        summary["timing_seconds"] = timer.summary()
        summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _log(log_fn, f"Wrote summary: {summary_out}")

    # Human report
    lines = [
        f"# ML Historical Backtest Report — {stamp}",
        "",
        f"**Data file:** `{summary['csv_path']}`",
        f"**Loaded source:** `{summary.get('data_path') or summary['csv_path']}`",
        f"**Candidate config:** `{summary['candidate_config_path'] or 'N/A (used CSV scores)'}`",
        f"**Generated:** {summary['generated_at']}",
        "",
        "## Parameters",
        f"- fallback_threshold: {threshold}",
        f"- use_candidate_thresholds: {use_candidate_thresholds}",
        f"- target_pct: {target_pct}",
        f"- stoploss_pct: {stoploss_pct}",
        f"- max_hold_bars: {max_hold_bars}",
        f"- max_trades_per_day: {max_trades_per_day}",
        f"- lot_size: {LOT_SIZE}",
        "",
        "## Performance Summary",
        f"- Total trades: {total}",
        f"- Win rate: {win_rate:.1%} ({wins}W / {losses}L)",
        f"- Gross PnL: {gross_sum:.2f}",
        f"- Net PnL: {net_sum:.2f}",
        f"- Profit factor: {pf if math.isfinite(pf) else 'inf'}",
        f"- Max drawdown: {max_dd_abs:.2f} ({max_dd_pct:.1%})",
        f"- Avg trade (net): {avg_trade:.2f}",
        f"- Sharpe-like (trade pnls): {sharpe:.3f}",
        f"- Best candidate by net: {best_cand}",
        f"- Selected score column: {selected_score_label or 'none'}",
        f"- Rows above threshold: {rows_above_threshold}",
        f"- Rejected by filters/caps: {summary['rejected_by_filters']}",
        "",
        "## Timing",
        f"- CSV/Parquet load time: {summary['timing_seconds'].get('csv_load_time', 0.0):.4f}s",
        f"- Artifact load time: {summary['timing_seconds'].get('artifact_load_time', 0.0):.4f}s",
        f"- Feature build time: {summary['timing_seconds'].get('feature_build_time', 0.0):.4f}s",
        f"- Prediction time: {summary['timing_seconds'].get('prediction_time', 0.0):.4f}s",
        f"- Simulation time: {summary['timing_seconds'].get('simulation_time', 0.0):.4f}s",
        f"- Report write time: {summary['timing_seconds'].get('report_write_time', 0.0):.4f}s",
        "",
        "## Output files",
        f"- Trades: `{trades_out.name}`",
        f"- Summary: `{summary_out.name}`",
        f"- This report: `{report_out.name}`",
        "",
        "## Warnings / Notes",
    ]
    if total == 0:
        lines.append("- No trades produced. Diagnostic details follow.")
        lines.extend([
            "",
            "## 0-Trade Diagnostics",
            f"- total_rows: {original_rows}",
            f"- valid_timestamp_rows: {valid_timestamp_rows}",
            f"- valid_price_rows: {valid_price_rows}",
            f"- selected_price_column: {'ltp' if 'ltp' in df.columns else 'none'}",
            f"- selected_timestamp_column: {'timestamp' if 'timestamp' in df.columns else 'none'}",
            f"- selected_score_column: {selected_score_label or 'none'}",
            f"- rows_above_threshold: {rows_above_threshold}",
            f"- rejection_counts: `{dict(rejection_counts)}`",
            "",
            "### Top 20 Scores",
        ])
        if top_20_scores:
            lines.append("| score | threshold | candidate_id | timestamp | symbol | option_type | strike | expiry | price |")
            lines.append("|---:|---:|---|---|---|---|---:|---|---:|")
            for item in top_20_scores:
                lines.append(
                    f"| {item.get('score', 0):.6f} | {item.get('threshold', 0):.4f} | {item.get('candidate_id')} | "
                    f"{item.get('timestamp')} | {item.get('symbol')} | {item.get('option_type')} | "
                    f"{item.get('strike')} | {item.get('expiry')} | {item.get('price')} |"
                )
        else:
            lines.append("- No scores were available.")
        lines.extend(["", "### First Rejections"])
        for rej in first_rejections:
            lines.append(f"- `{rej}`")
        lines.extend(["", "### Artifact Loading Failures"])
        for fail in artifact_failure_payload[:40]:
            lines.append(f"- candidate={fail.get('candidate_id')} path={fail.get('path') or '-'} reason={fail.get('reason')}")
        lines.extend([
            "",
            "### Recommendations",
            "- Lower threshold or enable candidate-specific thresholds.",
            "- Select the correct artifact directory or repair candidate config paths.",
            "- Run retrainer/export step to materialize model artifacts.",
            "- Use a CSV with embedded score/probability columns.",
            "- Repair candidate config paths in `config/paper_forward_candidates.json`.",
        ])
    if stopped:
        lines.append("- Backtest was stopped early; results are partial.")
    if not loaded_models and not score_candidates:
        lines.append("- No ML models were loaded and no model_score column was found in CSV. Scores were 0.")
    lines.append("- This is an offline research backtester. Does not reflect live slippage, queue priority, or regime changes.")
    lines.append("- Next checks: inspect per-candidate equity curves, cost sensitivity, and threshold robustness on the produced trades CSV.")
    lines.append("")

    with timer.track("report_write_time"):
        summary["timing_seconds"] = timer.summary()
        report_out.write_text("\n".join(lines), encoding="utf-8")
        summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(log_fn, f"Wrote report: {report_out}")

    result = BacktestResult(
        summary=summary,
        trades=closed_trades,
        output_paths={
            "trades_csv": str(trades_out),
            "summary_json": str(summary_out),
            "report_md": str(report_out),
            "output_dir": str(out_dir),
        },
        stopped=stopped,
    )
    if total == 0:
        _log(log_fn, f"COMPLETE - 0 TRADES: rows_above_threshold={rows_above_threshold} rejection_counts={dict(rejection_counts)} selected_score_column={selected_score_label or 'none'}")
        _progress(progress_fn, 1.0, "Complete — 0 trades")
    else:
        _log(log_fn, f"COMPLETE: {total} trades, net_pnl={net_sum:.2f}")
        _progress(progress_fn, 1.0, f"Complete — {total} trades")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run ML score threshold backtest over historical options CSV.")
    p.add_argument("--csv", required=True, help="Path to historical options CSV (processed or raw with tolerant cols)")
    p.add_argument("--config", default=None, help="Path to ML candidate config JSON (e.g. config/paper_forward_candidates.json)")
    p.add_argument("--output-dir", "--output", dest="output_dir", default="reports/backtests", help="Where to write artefacts")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--target-pct", type=float, default=DEFAULT_TARGET_PCT)
    p.add_argument("--stoploss-pct", type=float, default=DEFAULT_STOPLOSS_PCT)
    p.add_argument("--max-hold-bars", type=int, default=DEFAULT_MAX_HOLD_BARS)
    p.add_argument("--max-trades-per-day", type=int, default=DEFAULT_MAX_TRADES_PER_DAY)
    p.add_argument("--debug", action="store_true", help="Print/write detailed diagnostics, especially useful for 0-trade runs")
    p.add_argument("--no-candidate-thresholds", action="store_true", help="Use --threshold for every candidate instead of config/candidate-id thresholds")
    p.add_argument("--strict-artifacts", dest="strict_artifacts", action="store_true", default=True, help="Require exact artifacts/candidates/<candidate_id>/model.pkl (default: true)")
    p.add_argument("--no-strict-artifacts", dest="strict_artifacts", action="store_false", help="Disable strict artifact identity resolution")
    p.add_argument("--allow-artifact-fallback", action="store_true", default=False, help="Allow configured fallback artifact paths when exact artifact is missing (default: false)")
    p.add_argument("--ml-artifact-scoring", dest="ml_artifact_scoring", action="store_true", default=True, help="Score with loaded ML artifacts (default: true)")
    p.add_argument("--no-ml-artifact-scoring", dest="ml_artifact_scoring", action="store_false", help="Disable ML artifact scoring")
    p.add_argument("--allow-embedded-score-fallback", action="store_true", default=None, help="Allow CSV embedded score fallback when a config is provided")
    p.add_argument("--full-csv-read-fallback", action="store_true", default=False, help="Read full CSV for diagnostics if artifacts expose no feature metadata")
    p.add_argument("--selected-candidate-id", action="append", default=None, help="Run only the named candidate id/model name; repeatable")
    p.add_argument("--fast-mode", action="store_true", default=False, help="Enable date/max-row/max-candidate slicing and quiet progress defaults")
    p.add_argument("--date-from", default=None, help="Fast mode start timestamp/date")
    p.add_argument("--date-to", default=None, help="Fast mode end timestamp/date")
    p.add_argument("--max-rows", type=int, default=None, help="Fast mode maximum rows after sorting/date filtering")
    p.add_argument("--max-candidates", type=int, default=None, help="Maximum enabled candidates to load/score")
    p.add_argument("--skip-verbose-logs", action="store_true", default=False, help="Suppress row-level/entry logs")
    p.add_argument("--multiprocessing", action="store_true", default=False, help="Score candidates in separate processes when possible")
    p.add_argument("--progress-every", type=int, default=5000, help="Progress update/log interval in rows")
    args = p.parse_args(argv)

    def _print_log(s: str):
        print(s)

    try:
        res = run_historical_ml_backtest(
            csv_path=args.csv,
            candidate_config_path=args.config,
            output_dir=args.output_dir,
            threshold=args.threshold,
            target_pct=args.target_pct,
            stoploss_pct=args.stoploss_pct,
            max_hold_bars=args.max_hold_bars,
            max_trades_per_day=args.max_trades_per_day,
            log_fn=_print_log,
            debug=args.debug,
            use_candidate_thresholds=not args.no_candidate_thresholds,
            strict_artifacts=args.strict_artifacts,
            allow_artifact_fallback=args.allow_artifact_fallback,
            ml_artifact_scoring=args.ml_artifact_scoring,
            allow_embedded_score_fallback=args.allow_embedded_score_fallback,
            full_csv_read_fallback=args.full_csv_read_fallback,
            selected_candidate_ids=args.selected_candidate_id,
            fast_mode=args.fast_mode,
            date_from=args.date_from,
            date_to=args.date_to,
            max_rows=args.max_rows,
            max_candidates=args.max_candidates,
            skip_verbose_logs=args.skip_verbose_logs,
            multiprocessing=args.multiprocessing,
            progress_every=args.progress_every,
        )
        print(json.dumps(res.summary, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
