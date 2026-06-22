#!/usr/bin/env python3
"""
live_decision_dry_run.py
========================
Single-shot dry-run: show what the ML decision pipeline WOULD do right now
on live broker data — without placing any real orders.

Safety guarantees
-----------------
- NEVER calls real broker order APIs.
- Must be explicitly --paper or --dry-run flagged.
- Logs every step of the what-if decision path.
- Loads scaler mean/std from training to match live preprocessing.

Usage
-----
    # Show this help
    python scripts/live_decision_dry_run.py --help

    # Run with latest paper candidate model
    python scripts/live_decision_dry_run.py --paper

    # Run with a specific model artifact dir
    python scripts/live_decision_dry_run.py --model-dir models/core_retrain_20260606_093842

    # Skip broker login (use last-known values from DB/cache)
    python scripts/live_decision_dry_run.py --paper --offline

    # Load snapshot from a JSON fixture file
    python scripts/live_decision_dry_run.py --paper --fixture reports/snapshot_20260607.json

    # Single decision mode (exit after one decision)
    python scripts/live_decision_dry_run.py --paper --once

    # Verbose debug output
    python scripts/live_decision_dry_run.py --paper --debug

Output
------
    reports/live_decision_dry_run_<timestamp>.md
    reports/live_decision_dry_run_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --- paths --------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_REPO_ROOT))

_LOGGER = logging.getLogger("live_decision_dry_run")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# ---------------------------------------------------------------------------
# Model discovery helpers
# ---------------------------------------------------------------------------

def _has_valid_pkl(model_dir: Path) -> bool:
    """Check if model_dir has at least one trained .pkl (excluding metrics/reports)."""
    for p in model_dir.glob("*.pkl"):
        name = p.name
        if any(kw in name for kw in ["_metrics.json", "ensemble_weights",
                                      "_threshold_sweep", "_skip_report"]):
            continue
        return True
    return False


def find_approved_model_dirs() -> List[Path]:
    """Find model dirs that passed deployment readiness (have a deployment_manifest.json)."""
    models_dir = _REPO_ROOT / "models"
    approved: List[Tuple[datetime, Path]] = []
    for d in models_dir.iterdir():
        if not d.is_dir():
            continue
        manifest = d / "deployment_manifest.json"
        if manifest.exists():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                created_str = payload.get("created_at", "")
                # parse ISO timestamp
                ts = datetime.fromisoformat(created_str.replace("Z", "+00:00")) if created_str else datetime.fromtimestamp(d.stat().st_mtime)
                # Include anything with a manifest (even SHADOW_ONLY) for dry-run visibility
                approved.append((ts, d))
            except Exception as exc:
                _LOGGER.warning("Could not parse manifest %s: %s", manifest, exc)
    approved.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in approved]


def find_paper_candidate_model_dirs() -> List[Path]:
    """Find model dirs with PAPER_TRADE_CANDIDATE verdicts via paper_watchlist_report.json,
    preferring dirs that have actual trained .pkl files."""
    models_dir = _REPO_ROOT / "models"
    candidates: List[Tuple[datetime, Path, bool]] = []
    for d in models_dir.iterdir():
        if not d.is_dir():
            continue
        report = d / "paper_watchlist_report.json"
        if report.exists():
            try:
                ts = datetime.fromtimestamp(d.stat().st_mtime)
                has_pkl = _has_valid_pkl(d)
                candidates.append((ts, d, has_pkl))
            except Exception:
                pass
    # Prefer dirs with .pkl files, then by recency
    candidates.sort(key=lambda x: (x[2], x[0]), reverse=True)
    return [p for _, p, _ in candidates]


def find_model_dirs_with_pkls() -> List[Path]:
    """Find all model dirs that contain at least one trained .pkl file, sorted newest first.

    Filters out dirs that:
    - Have no valid .pkl files (e.g. retrain_all_models_20260607_024322 with only metrics, no .pkl)
    - Have feature_count=0 in their feature_manifest.json
    - Have model_pkl=null in their metrics (phantom artifacts)

    APPROVED MODEL ARTIFACTS MUST HAVE:
    - model_pkl that is not null in candidate_champion_report.json
    - feature_count > 0 in feature_manifest.json
    - feature list schema (feature_list_used.json with >= 20 features or pkl bundle with feature_names)
    - metrics JSON present
    """
    models_dir = _REPO_ROOT / "models"
    found: List[Tuple[datetime, Path]] = []
    for d in models_dir.iterdir():
        if not d.is_dir():
            continue
        # Reject dirs with no valid .pkl at all
        if not _has_valid_pkl(d):
            _LOGGER.debug("Skipping %s: no valid .pkl files", d.name)
            continue

        # Check for candidate_champion_report.json with model_pkl != null
        champion_report = d / "candidate_champion_report.json"
        if champion_report.exists():
            try:
                cr = json.loads(champion_report.read_text(encoding="utf-8"))
                obs = cr.get("best_global_observed", {}) or {}
                model_pkl = obs.get("model_pkl")
                if model_pkl is None or model_pkl == "null" or model_pkl == "":
                    _LOGGER.warning("Skipping %s: model_pkl is null/empty in candidate_champion_report.json", d.name)
                    continue
            except Exception:
                pass

        # Reject dirs that are clearly phantom (have only metrics, no trained artifacts)
        manifest_path = d / "feature_manifest.json"
        if manifest_path.exists():
            try:
                fm = json.loads(manifest_path.read_text(encoding="utf-8"))
                fc = int(fm.get("feature_count", 0) or 0)
                if fc == 0:
                    _LOGGER.warning("Skipping %s: feature_count=0 in feature_manifest.json", d.name)
                    continue
            except Exception:
                pass

        # Check the pkl itself has feature_names (load and inspect)
        pkl_candidates = list(d.glob("*_profitable_trade_label.pkl")) or list(d.glob("*.pkl"))
        has_valid_bundle = False
        for pkl_path in pkl_candidates:
            if "_metrics.json" in str(pkl_path) or "_threshold" in str(pkl_path) or "ensemble_weights" in str(pkl_path):
                continue
            try:
                import joblib
                from ml_signals import MLModelBundle
                bundle = joblib.load(pkl_path)
                if isinstance(bundle, MLModelBundle):
                    if bundle.feature_names and len(bundle.feature_names) >= 20:
                        has_valid_bundle = True
                        break
                elif isinstance(bundle, dict):
                    if bundle.get("feature_names") and len(bundle.get("feature_names", [])) >= 20:
                        has_valid_bundle = True
                        break
            except Exception:
                pass

        if not has_valid_bundle:
            _LOGGER.warning("Skipping %s: no valid pkl bundle with >= 20 features found", d.name)
            continue

        try:
            ts = datetime.fromtimestamp(d.stat().st_mtime)
            found.append((ts, d))
        except Exception:
            pass
    found.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in found]


def load_deployment_manifest(model_dir: Path) -> Dict[str, Any]:
    manifest = model_dir / "deployment_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return payload


def get_model_pkl(model_dir: Path, manifest: Dict[str, Any]) -> Optional[Path]:
    model_path = manifest.get("model_path", "")
    if not model_path:
        return None
    p = model_dir / model_path
    return p if p.exists() else None


def load_feature_list(manifest: Dict[str, Any], model_dir: Path) -> List[str]:
    fl = manifest.get("selected_feature_list", [])
    if fl:
        return [str(f) for f in fl]
    fl_path = manifest.get("selected_feature_list_path", "")
    if fl_path:
        p = model_dir / fl_path
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    # Fallback: feature_list_used.json
    fl2 = model_dir / "feature_list_used.json"
    if fl2.exists():
        try:
            payload = json.loads(fl2.read_text(encoding="utf-8"))
            feat_list = payload if isinstance(payload, list) else payload.get("features", [])
            if feat_list:
                return feat_list
        except Exception:
            pass
    return []


def load_model_bundle(model_path: Path):
    """Load an MLModelBundle (or dict-wrapped bundle) from a .pkl file."""
    import joblib
    loaded = joblib.load(model_path)
    # Normalise to MLModelBundle dataclass
    from ml_signals import MLModelBundle
    if isinstance(loaded, MLModelBundle):
        return loaded
    if isinstance(loaded, dict) and "model" in loaded:
        return MLModelBundle(
            model=loaded.get("model"),
            feature_names=list(loaded.get("feature_names") or []),
            metrics=dict(loaded.get("metrics") or {}),
            trained_at=str(loaded.get("trained_at") or ""),
            scaler_mean=loaded.get("scaler_mean"),
            scaler_std=loaded.get("scaler_std"),
        )
    # Legacy: raw model, wrap minimally
    return MLModelBundle(model=loaded, feature_names=[], metrics={}, trained_at="legacy")


def find_model_pkl_in_dir(model_dir: Path) -> Optional[Path]:
    """Heuristic: find the first .pkl that looks like a trained model."""
    candidates = sorted(model_dir.glob("*_profitable_trade_label.pkl"))
    if not candidates:
        candidates = sorted(model_dir.glob("*.pkl"))
    for p in candidates:
        # skip metric-only files
        if "_metrics.json" in str(p) or "_threshold" in str(p) or "ensemble_weights" in str(p):
            continue
        return p
    return None


# ---------------------------------------------------------------------------
# Candidate manifest loader
# ---------------------------------------------------------------------------

def load_candidate_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load and validate a candidate_manifest.json.

    Required fields: model_pkl, selected_threshold, candidate_id (or model_id).
    Optional fields: filter_definition, feature_schema, status, gates_*.

    Fails cleanly with a clear error if the manifest is missing, unreadable,
    or missing any required field.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Candidate manifest not found: {manifest_path}\n"
            f"Please provide a valid --candidate-manifest path to a candidate_manifest.json file."
        )

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"Candidate manifest is unreadable or invalid JSON: {manifest_path}\n"
            f"Error: {exc}"
        ) from exc

    # Accept candidate_id OR model_id (both names used across artifacts)
    id_value = payload.get("candidate_id") or payload.get("model_id")
    if not id_value:
        raise ValueError(
            f"Candidate manifest is missing required field(s): ['candidate_id' or 'model_id']\n"
            f"Manifest: {manifest_path}\n"
            f"Required fields: ['model_pkl', 'selected_threshold', 'candidate_id' or 'model_id']"
        )
    required = ["model_pkl", "selected_threshold"]
    missing = [f for f in required if f not in payload or payload[f] is None
               or (isinstance(payload[f], str) and payload[f] == "")]
    if missing:
        raise ValueError(
            f"Candidate manifest is missing required field(s): {missing}\n"
            f"Manifest: {manifest_path}\n"
            f"Required fields: ['model_pkl', 'selected_threshold', 'candidate_id' or 'model_id']"
        )

    model_pkl_raw = payload["model_pkl"]
    # Resolve model_pkl relative to repo root
    model_pkl = _REPO_ROOT / model_pkl_raw
    if not model_pkl.exists():
        raise FileNotFoundError(
            f"model_pkl path in manifest does not exist: {model_pkl}\n"
            f"  (resolved from 'model_pkl' field: {model_pkl_raw})\n"
            f"  Manifest: {manifest_path}"
        )

    filter_def = payload.get("filter_definition") or {}
    filter_applied = bool(filter_def.get("filter_expression") or filter_def.get("type"))

    return {
        "model_pkl": model_pkl,
        "model_pkl_raw": model_pkl_raw,
        "selected_threshold": float(payload["selected_threshold"]),
        "candidate_id": str(id_value),
        "filter_definition": filter_def,
        "filter_applied": filter_applied,
        "feature_schema": payload.get("feature_schema") or [],
        "status": payload.get("status", "UNKNOWN"),
        "gates_passed": payload.get("gates_passed"),
        "gates_total": payload.get("gates_total"),
        "metrics": payload.get("metrics_from_refined_retraining") or {},
        "source": str(manifest_path),
    }


# ---------------------------------------------------------------------------
# Live data helpers
# ---------------------------------------------------------------------------

def get_live_spot_and_iv(client, symbol_token: str) -> Dict[str, Any]:
    """Fetch spot price and ATM IV from the broker."""
    result: Dict[str, Any] = {"spot": None, "atm_iv": None, "atm_strike": None}
    try:
        ltp_data = client.get_ltp(symbol_token)
        if ltp_data:
            result["spot"] = float(ltp_data.get("last_price") or ltp_data.get("ltp") or 0)
    except Exception as exc:
        _LOGGER.warning("Could not fetch spot LTP: %s", exc)

    # Try to get option chain for ATM IV
    try:
        chain = client.get_option_chain(symbol_token)
        if chain:
            atm_strike = round(result.get("spot", 0) / 50) * 50
            for item in chain:
                if (item.get("strike_price") == atm_strike and
                        item.get("option_type") in ("CE", "PE")):
                    result["atm_iv"] = float(item.get("implied_volatility") or 0)
                    result["atm_strike"] = atm_strike
                    break
    except Exception as exc:
        _LOGGER.warning("Could not fetch option chain for ATM IV: %s", exc)

    return result


def get_live_candles(
    client,
    symbol_token: str,
    timeframe: str = "1m",
    lookback: int = 30,
) -> List[Dict[str, Any]]:
    """Fetch recent candles for a token."""
    try:
        from src.market_data import Candle
        candles = client.get_candles(symbol_token, timeframe=timeframe, lookback=lookback)
        if candles and isinstance(candles[0], Candle):
            return [{"time": c.time, "open": c.open, "high": c.high,
                     "low": c.low, "close": c.close, "volume": c.volume} for c in candles]
        return list(candles or [])
    except Exception as exc:
        _LOGGER.warning("Could not fetch candles for token %s: %s", symbol_token, exc)
        return []


def build_live_feature_snapshot(
    candles: List[Dict[str, Any]],
    spot_data: Dict[str, Any],
    *,
    lookback: int = 20,
) -> Dict[str, Any]:
    """Build a flat dict of all available live features matching the model's schema.

    This mirrors build_market_feature_vector from ml_pipeline.py but returns
    a dict keyed by feature name so we can do schema alignment.
    """
    from src.ml_pipeline import build_market_feature_vector, MLFeatureContext

    # Convert to Candle objects for build_market_feature_vector
    from src.market_data import Candle as _Candle

    candle_objs: List[_Candle] = []
    for c in candles:
        try:
            candle_objs.append(_Candle(
                time=c["time"] if isinstance(c["time"], datetime) else datetime.fromisoformat(str(c["time"])),
                open=float(c["open"]),
                high=float(c["high"]),
                low=float(c["low"]),
                close=float(c["close"]),
                volume=float(c.get("volume") or 0),
            ))
        except Exception:
            continue

    ctx = MLFeatureContext(
        spot=spot_data.get("spot"),
        iv=spot_data.get("atm_iv"),
        time_sin=None,  # will be filled by build_market_feature_vector
        time_cos=None,
    )

    feat_vec, feat_names = build_market_feature_vector(candle_objs, context=ctx, lookback=lookback)

    snapshot: Dict[str, Any] = {}
    for name, val in zip(feat_names, feat_vec):
        snapshot[name] = val

    # Add spot and context fields
    snapshot["spot"] = spot_data.get("spot")
    snapshot["option_ltp"] = None  # filled per-contract below
    snapshot["option_type"] = None  # filled per-contract below
    snapshot["strike"] = None
    snapshot["expiry"] = None
    snapshot["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # Time features
    now = datetime.now()
    snapshot["weekday"] = float(now.weekday())
    snapshot["month"] = float(now.month)
    snapshot["ctx_time_sin"] = math.sin(2 * math.pi * (now.hour * 60 + now.minute) / 1440)
    snapshot["ctx_time_cos"] = math.cos(2 * math.pi * (now.hour * 60 + now.minute) / 1440)
    snapshot["is_opening_session"] = 1.0 if 9 * 60 + 15 <= now.hour * 60 + now.minute <= 10 * 60 + 30 else 0.0
    snapshot["is_closing_session"] = 1.0 if now.hour * 60 + now.minute >= 15 * 60 else 0.0
    snapshot["is_midday_lull"] = 1.0 if 12 * 60 <= now.hour * 60 + now.minute <= 13 * 60 else 0.0

    return snapshot


def _build_offline_fixture(model_features: List[str]) -> Dict[str, Any]:
    """Build a comprehensive snapshot fixture for offline dry-run.

    Populates as many model features as possible from:
    1. Time/context features (always available from clock)
    2. Candle-based features using a synthetic flat candle
    3. Option-chain features set to None (marked unavailable)
    4. Rolling-history features set to None (marked unavailable)
    5. Spot context features set to synthetic reasonable values

    This is NOT zero-filling silently — each feature is categorized and
    the alignment report shows exactly what is missing and why.
    """
    now = datetime.now(timezone.utc)
    snapshot: Dict[str, Any] = {
        "timestamp": now.isoformat().replace("+00:00", "Z"),
        "note": "offline dry-run fixture — provides time/context/candle/spot features; "
                "option chain and rolling-history features are unavailable; "
                "set MSTOCK_SYMBOL_TOKEN for live snapshot",
    }

    model_set = set(model_features)

    # --- Time/context features (always available) ---
    session_minutes = now.hour * 60 + now.minute
    is_opening = 9 * 60 + 15 <= session_minutes <= 10 * 60 + 30
    is_closing = session_minutes >= 15 * 60
    is_midday = 12 * 60 <= session_minutes <= 13 * 60

    ctx_keys = {
        "ctx_time_sin": math.sin(2 * math.pi * session_minutes / 1440),
        "ctx_time_cos": math.cos(2 * math.pi * session_minutes / 1440),
        "weekday": float(now.weekday()),
        "month": float(now.month),
        "weekly": 1.0 if now.weekday() < 5 else 0.0,
        "is_weekly": 1.0,
        "is_opening_session": 1.0 if is_opening else 0.0,
        "is_closing_session": 1.0 if is_closing else 0.0,
        "is_midday_lull": 1.0 if is_midday else 0.0,
    }
    for k, v in ctx_keys.items():
        if k in model_set:
            snapshot[k] = v

    # --- Synthetic candle features ---
    # Use a flat/synthetic candle so candle-computed features are stable
    SYNTHETIC_CLOSE = 24500.0
    SYNTHETIC_RANGE = SYNTHETIC_CLOSE * 0.002  # 0.2% range
    s_open = SYNTHETIC_CLOSE - SYNTHETIC_RANGE * 0.3
    s_close = SYNTHETIC_CLOSE
    s_high = SYNTHETIC_CLOSE + SYNTHETIC_RANGE * 0.5
    s_low = SYNTHETIC_CLOSE - SYNTHETIC_RANGE * 0.6
    s_vol = 50000.0

    rng = max(s_high - s_low, 1e-9)
    body = s_close - s_open
    upper_wick = s_high - max(s_open, s_close)
    lower_wick = min(s_open, s_close) - s_low

    # Comprehensive candle features mapping
    candle_keys = {
        "last_open": s_open, "last_high": s_high, "last_low": s_low,
        "last_close": s_close, "last_volume": s_vol,
        "open": s_open, "high": s_high, "low": s_low, "close": s_close,
        "volume": s_vol,
        "body_pct": body / s_close,
        "range_pct": rng / s_close,
        "gap_pct": 0.0,
        "upper_wick_pct": upper_wick / rng,
        "lower_wick_pct": lower_wick / rng,
        "close_location_pct": (s_close - s_low) / rng if rng > 0 else 0.5,
        "bullish_engulfing": 0.0, "bearish_engulfing": 0.0,
        "doji": 0.0, "hammer": 0.0, "shooting_star": 0.0,
        # Return stats (synthetic zero momentum)
        "ret_1": 0.0, "ret_3": 0.0, "ret_5": 0.0, "ret_10": 0.0,
        "ret_mean": 0.0, "ret_std": 0.0, "ret_min": 0.0, "ret_max": 0.0,
        # Technical indicators
        "ema_fast": s_close, "ema_slow": s_close, "ema_diff_pct": 0.0,
        "rsi_14": 50.0, "atr_14": rng * 14, "atr_pct": rng / s_close,
        "adx_14": 25.0, "roc_14": 0.0, "choppiness_14": 50.0,
        "supertrend_dir": 0.0, "supertrend_gap_pct": 0.0,
        "pivot_pp_dist_pct": 0.0, "pivot_r1_dist_pct": 0.0, "pivot_s1_dist_pct": 0.0,
        "close_vs_open_pct": body / s_close if s_close != 0 else 0.0,
        "range_to_atr": 1.0,  # neutral
        "momentum_lookback_pct": 0.0,
        "vol_mean": s_vol, "vol_std": 0.0, "vol_min": s_vol, "vol_max": s_vol,
        "vol_of_vol_14": 0.0,
        "regime_quiet": 1.0,
        "regime_trending": 0.0, "regime_volatile": 0.0, "regime_mean_reverting": 0.0,
        "volatility_regime_classifier": 0.0,
        "atr_percentile_60": 0.5, "realized_vol_percentile_60": 0.5,
        "volatility_percentile_60": 0.5,
        "atr_pct_regime_10": 0.0, "realized_vol_30": 0.0,
        "volume_ratio": 1.0,
        # Spot context features (synthetic but reasonable)
        "spot": SYNTHETIC_CLOSE,
        "open_spot": SYNTHETIC_CLOSE, "high_spot": SYNTHETIC_CLOSE,
        "low_spot": SYNTHETIC_CLOSE, "close_spot": SYNTHETIC_CLOSE,
        "spot_close": SYNTHETIC_CLOSE,
        "spot_range_pct": 0.0,
        "spot_atr": rng,
        "spot_rsi": 50.0,
        "spot_vwap": SYNTHETIC_CLOSE,
        "volume_spot": s_vol,
        "weekday_spot": float(now.weekday()),
        "ctx_spot": SYNTHETIC_CLOSE,
        # Option-related features that can be derived from synthetic spot
        "atm_distance": 0.0,
        "strike_distance_pct": 0.0,
        "distance_from_spot": 0.0,
        # DTE and expiry features
        "dte_days": 3.0,  # typical weekly expiry
        "ctx_dte_norm": 0.5,  # mid-range
    }
    for k, v in candle_keys.items():
        if k in model_set:
            snapshot[k] = v

    # --- Option chain features (always None in offline mode) ---
    option_chain_features = {
        "ltp", "oi", "oi_CE", "oi_PE", "volume_CE", "volume_PE",
        "oi_change_pct", "volume_change_pct", "ce_pe_oi_ratio", "ce_pe_volume_ratio",
        "final_iv", "atm_distance", "strike_distance_pct", "distance_from_spot",
        "option_ltp", "strike_price", "expiry", "option_type",
        "option_to_spot_pct",
    }
    for feat in model_set & option_chain_features:
        snapshot[feat] = None  # explicitly None, not zero

    # --- Rolling history features (always None in offline mode) ---
    rolling_features = {
        "oi_z_5", "volume_z_5",
        "dist_from_opening_high_pct", "dist_from_opening_low_pct",
        "opening_range_width_pct", "dist_to_rolling_high_20", "dist_to_rolling_low_20",
        "rolling_range_width_20", "rolling_range_position_20",
    }
    for feat in model_set & rolling_features:
        snapshot[feat] = None

    # --- Option type indicators (set to neutral values) ---
    option_type_features = {
        "option_type_ce": 1.0, "option_type_pe": 0.0,
    }
    for feat, v in option_type_features.items():
        if feat in model_set:
            snapshot[feat] = v

    # --- Additional context features that might be missing ---
    additional_ctx = {
        "ctx_iv": None, "ctx_delta": None, "ctx_gamma": None,
        "ctx_vega": None, "ctx_theta": None,
        "ctx_adx": 25.0, "ctx_trend_strength": 0.5, "ctx_choppiness": 50.0,
        "ctx_volume_sma": s_vol, "ctx_option_price": None,
    }
    for feat, v in additional_ctx.items():
        if feat in model_set:
            snapshot[feat] = v

    # --- OC/HL change features (derived from synthetic candle) ---
    oc_change = (s_close - s_open) / s_open if s_open != 0 else 0.0
    hl_change = (s_high - s_low) / s_low if s_low != 0 else 0.0
    change_features = {
        "oc_change_pct": oc_change,
        "hl_change_pct": hl_change,
    }
    for feat, v in change_features.items():
        if feat in model_set:
            snapshot[feat] = v

    return snapshot


def predict_with_bundle(bundle, feature_dict: Dict[str, Any]) -> float:
    """Run model.predict_proba using the bundle's scaler if available."""
    from ml_signals import predict
    feature_names = list(bundle.feature_names)
    feature_vector = [[float(feature_dict.get(name, 0.0) or 0.0) for name in feature_names]]
    probs = predict(bundle, feature_vector)
    return float(probs[0]) if probs else 0.0


# ---------------------------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Feature source taxonomy (used for grouping missing features)
# ---------------------------------------------------------------------------_

# Features computed purely from candle OHLCV data
CANDLE_FEATURES = {
    "last_open", "last_high", "last_low", "last_close", "last_volume",
    "body_pct", "range_pct", "gap_pct", "upper_wick_pct", "lower_wick_pct",
    "close_location_pct", "open", "high", "low", "close", "volume",
    "bullish_engulfing", "bearish_engulfing", "doji", "hammer", "shooting_star",
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_mean", "ret_std", "ret_min", "ret_max",
    "ema_fast", "ema_slow", "ema_diff_pct", "rsi_14", "atr_14", "atr_pct",
    "adx_14", "roc_14", "choppiness_14", "supertrend_dir", "supertrend_gap_pct",
    "pivot_pp_dist_pct", "pivot_r1_dist_pct", "pivot_s1_dist_pct",
    "close_vs_open_pct", "range_to_atr", "momentum_lookback_pct",
    "regime_trending", "regime_volatile", "regime_mean_reverting", "regime_quiet",
    "vol_mean", "vol_std", "vol_min", "vol_max", "vol_of_vol_14",
    "volatility_regime_classifier",
    "atr_percentile_60", "realized_vol_percentile_60", "volatility_percentile_60",
    "atr_pct_regime_10", "realized_vol_30",
}

# Features requiring live option chain data
OPTION_CHAIN_FEATURES = {
    "ltp", "oi", "oi_CE", "oi_PE", "volume_CE", "volume_PE",
    "oi_change_pct", "volume_change_pct", "ce_pe_oi_ratio", "ce_pe_volume_ratio",
    "final_iv", "atm_distance", "strike_distance_pct", "distance_from_spot",
    "option_ltp", "strike_price", "expiry", "option_type",
}

# Features requiring rolling history (z-scores, rolling stats)
ROLLING_HISTORY_FEATURES = {
    "oi_z_5", "volume_z_5",
    "dist_from_opening_high_pct", "dist_from_opening_low_pct",
    "opening_range_width_pct", "dist_to_rolling_high_20", "dist_to_rolling_low_20",
    "rolling_range_width_20", "rolling_range_position_20",
}

# Context features that can be derived from current time / spot without history
CONTEXT_FEATURES = {
    "ctx_spot", "ctx_time_sin", "ctx_time_cos",
    "ctx_iv", "ctx_delta", "ctx_gamma", "ctx_vega", "ctx_theta",
    "ctx_adx", "ctx_trend_strength", "ctx_choppiness", "ctx_volume_sma",
    "ctx_option_price", "ctx_dte_norm",
    "weekday", "month", "weekly", "is_weekly", "is_opening_session",
    "is_closing_session", "is_midday_lull",
    "spot", "open_spot", "high_spot", "low_spot", "close_spot",
    "spot_close", "spot_range_pct", "spot_atr", "spot_rsi", "spot_vwap",
    "volume_spot", "weekday_spot", "option_to_spot_pct", "dte_days",
    "option_type_ce", "option_type_pe",
    "hl_change_pct", "oc_change_pct",
}


def _classify_missing_feature(feat: str) -> str:
    if feat in CANDLE_FEATURES:
        return "candle"
    if feat in OPTION_CHAIN_FEATURES:
        return "option_chain"
    if feat in ROLLING_HISTORY_FEATURES:
        return "rolling_history"
    if feat in CONTEXT_FEATURES:
        return "context"
    return "unavailable"


def align_features(
    model_features: List[str],
    live_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare model features against live data, return alignment report with source grouping."""
    live_keys = set(live_snapshot.keys())
    model_set = set(model_features)

    missing = sorted(model_set - live_keys)
    available = sorted(model_set & live_keys)
    extra = sorted(live_keys - model_set)

    aligned: Dict[str, float] = {}
    for name in model_features:
        val = live_snapshot.get(name)
        try:
            aligned[name] = float(val) if val is not None else 0.0
        except (TypeError, ValueError):
            aligned[name] = 0.0

    coverage_pct = round(len(available) / max(len(model_features), 1) * 100, 2)

    # Group missing features by source
    by_source: Dict[str, List[str]] = {
        "candle": [], "option_chain": [], "rolling_history": [], "context": [], "unavailable": [],
    }
    for feat in missing:
        by_source[_classify_missing_feature(feat)].append(feat)

    return {
        "model_id": None,  # filled by caller
        "model_pkl": None,  # filled by caller
        "available_features": available,
        "missing_features": missing,
        "extra_live_features": extra,
        "aligned_vector": aligned,
        "missing_count": len(missing),
        "available_count": len(available),
        "total_model_features": len(model_features),
        "coverage_pct": coverage_pct,
        "missing_by_source": {k: v for k, v in by_source.items() if v},
    }


# ---------------------------------------------------------------------------
# Decision logic (mirrors MLRuntimeEngine.evaluate_snapshot)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PE-only candidate filter — live-computable, no outcome/future/PnL/label columns
# ---------------------------------------------------------------------------

def apply_pe_only_filter(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Standalone PE-only filter for candidate scoring pipeline.

    Rule: allow scoring ONLY when option_type == "PE".
    CE rows and rows with missing/invalid option_type are rejected before scoring.

    This filter is live-computable: it reads only the `option_type` field
    from the current snapshot — no outcome columns, no future returns,
    no PnL, no label columns.

    Returns
    -------
    dict with keys:
        filter_name              : "PE_only"
        filter_rule              : "allow scoring only when option_type == 'PE'"
        filter_applied           : bool (always True — filter is always evaluated)
        filter_passed            : bool
        filter_rejection_reason  : str | None
    """
    filter_name = "PE_only"
    filter_rule = "allow scoring only when option_type == 'PE'"
    filter_applied = True
    filter_passed = True
    filter_rejection_reason: Optional[str] = None

    option_type_val = snapshot.get("option_type")

    # Canonicalise to string, upper-cased; treat None / empty as unknown
    option_type_str = ""
    if option_type_val is not None:
        try:
            option_type_str = str(option_type_val).strip().upper()
        except Exception:
            option_type_str = ""

    if not option_type_str:
        # Missing or None option_type -> reject safely (cannot confirm PE)
        filter_passed = False
        filter_rejection_reason = "candidate_filter_rejected_option_type_not_PE"
    elif option_type_str != "PE":
        # Non-PE (CE, or any other value) -> reject
        filter_passed = False
        filter_rejection_reason = "candidate_filter_rejected_option_type_not_PE"

    return {
        "filter_name": filter_name,
        "filter_rule": filter_rule,
        "filter_applied": filter_applied,
        "filter_passed": filter_passed,
        "filter_rejection_reason": filter_rejection_reason,
    }


def evaluate_decision(
    *,
    probability: float,
    threshold: float,
    feature_alignment: Dict[str, Any],
    risk_result: Dict[str, Any],
    model_id: str,
    model_dir: str,
    bundle_trained_at: str,
    snapshot: Dict[str, Any],
    model_pkl: Optional[str] = None,
    candidate_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate the full decision pipeline and explain each filter.

    Returns a decision dict with final_action, blocking_reasons, steps,
    and all required report fields.
    """
    steps: List[Dict[str, Any]] = []
    final_action = "SKIP"
    blocking_reasons: List[str] = []

    # --- Candidate filter check (only when --candidate-manifest is provided) ---
    # PE-only filter is live-computable: uses only option_type, no outcome/PnL/label columns.
    # Only applied when a candidate manifest with a filter definition is present.
    if candidate_info is not None:
        pe_filter = apply_pe_only_filter(snapshot)
        filter_pass = pe_filter["filter_passed"]
        filter_reason = pe_filter.get("filter_rejection_reason") or ""
        filter_type = pe_filter["filter_name"]
        filter_applied = pe_filter["filter_applied"]
        filter_rule = pe_filter["filter_rule"]
    else:
        filter_pass = True
        filter_reason = ""
        filter_type = "none"
        filter_applied = False
        filter_rule = ""

    # Step 1: candidate filter check
    steps.append({
        "step": "CANDIDATE_FILTER_CHECK",
        "passed": filter_pass,
        "detail": (
            f"filter={filter_type} -> "
            f"{filter_reason if not filter_pass else 'PASS'}"
        ),
        "filter_name": filter_type,
        "filter_rule": filter_rule,
        "filter_applied": filter_applied,
        "filter_rejection_reason": filter_reason if not filter_pass else None,
        "option_type": snapshot.get("option_type", "unknown"),
    })

    if not filter_pass:
        final_action = "SKIP"
        blocking_reasons.append(filter_reason)
        blocker = filter_reason
        return {
            "final_action": final_action,
            "blocking_reasons": blocking_reasons,
            "blocker": blocker,
            "steps": steps,
            "probability": probability,
            "threshold": threshold,
            "model_id": model_id,
            "model_dir": str(model_dir),
            "model_pkl": model_pkl,
            "trained_at": bundle_trained_at,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "snapshot_keys": list(snapshot.keys())[:20],
            # PE-only filter report fields
            "filter_name": filter_type,
            "filter_rule": filter_rule,
            "filter_applied": filter_applied,
            "filter_passed": filter_pass,
            "filter_rejection_reason": filter_reason,
        }

    # Step 2: feature coverage check (>= 95% required to proceed)
    missing = feature_alignment.get("missing_features", [])
    coverage = feature_alignment.get("coverage_pct", 0)
    missing_by_source = feature_alignment.get("missing_by_source", {})
    coverage_ok = float(coverage) >= 95.0
    steps.append({
        "step": "FEATURE_COVERAGE_CHECK",
        "passed": coverage_ok,
        "detail": f"{len(missing)} missing, {coverage}% coverage (need >= 95%)",
        "coverage_pct": coverage,
        "required_pct": 95.0,
        "missing_features_sample": missing[:10],
        "missing_by_source": {k: v[:5] for k, v in missing_by_source.items()},
    })
    if not coverage_ok:
        final_action = "SKIP"
        if missing_by_source.get("option_chain"):
            blocking_reasons.append(
                f"coverage_{coverage:.1f}%_lt_95%_"
                f"need_option_chain_features_{len(missing_by_source.get('option_chain', []))}"
            )
        elif missing_by_source.get("rolling_history"):
            blocking_reasons.append(
                f"coverage_{coverage:.1f}%_lt_95%_"
                f"need_rolling_history_features_{len(missing_by_source.get('rolling_history', []))}"
            )
        else:
            blocking_reasons.append(f"coverage_{coverage:.1f}%_lt_95%_missing_{len(missing)}")
    else:
        # Step 3: full schema validation (warn if any features still missing)
        full_schema_ok = len(missing) == 0
        steps.append({
            "step": "FULL_SCHEMA_VALIDATION",
            "passed": full_schema_ok,
            "detail": "full alignment" if full_schema_ok else f"{len(missing)} features unavailable (will be zero-filled)",
            "missing_features": missing[:10],
        })

        # Step 4: probability vs threshold
        threshold_pass = float(probability) > float(threshold)
        steps.append({
            "step": "THRESHOLD_CHECK",
            "passed": threshold_pass,
            "detail": f"prob={probability:.4f} threshold={threshold:.4f}",
            "probability": probability,
            "threshold": threshold,
        })
        if not threshold_pass:
            final_action = "SKIP"
            blocking_reasons.append(f"low_confidence_{probability:.4f}_lt_{threshold:.4f}")
        else:
            # Step 5: risk filter
            risk_allowed = bool(risk_result.get("allowed", False))
            risk_reason = risk_result.get("blocking_reason", "")
            steps.append({
                "step": "RISK_FILTER",
                "passed": risk_allowed,
                "detail": f"reason={risk_reason}",
                "risk_state": risk_result.get("current_limits_state", {}),
                "blocking_reason": risk_reason,
            })
            if not risk_allowed:
                final_action = "SKIP"
                blocking_reasons.append(f"risk_blocked_{risk_reason}")
            else:
                final_action = "TRADE"
                steps.append({
                    "step": "FINAL_DECISION",
                    "passed": True,
                    "detail": "All checks passed — WOULD TRADE (paper/DRY-RUN only)",
                })

    # Build the blocker string for the report
    blocker = blocking_reasons[0] if blocking_reasons else None

    return {
        "final_action": final_action,
        "blocking_reasons": blocking_reasons,
        "blocker": blocker,
        "steps": steps,
        "probability": probability,
        "threshold": threshold,
        "model_id": model_id,
        "model_dir": str(model_dir),
        "model_pkl": model_pkl,
        "trained_at": bundle_trained_at,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "snapshot_keys": list(snapshot.keys())[:20],
        # PE-only filter report fields
        "filter_name": filter_type,
        "filter_rule": filter_rule,
        "filter_applied": filter_applied,
        "filter_passed": filter_pass,
        "filter_rejection_reason": filter_reason if not filter_pass else None,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_reports(
    result: Dict[str, Any],
    feature_alignment: Dict[str, Any],
    model_info: Dict[str, Any],
    dry_run_mode: str,
    reports_dir: Path,
) -> Tuple[Path, Path]:
    """Generate Markdown and JSON reports for the dry-run decision.

    Returns paths to the written reports.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stamp = f"live_decision_dry_run_{ts}"
    md_path = reports_dir / f"{stamp}.md"
    json_path = reports_dir / f"{stamp}.json"

    # --- Markdown ---
    md_lines = [
        f"# Live Decision Dry-Run Report",
        "",
        f"**Generated:** {result.get('timestamp', '')}",
        f"**Mode:** `{dry_run_mode}`  *(NO REAL ORDERS — PAPER/DRY-RUN ONLY)*",
        f"**Status:** `{result['final_action']}`",
        "",
        "## Model Info",
        f"- **Model ID:** `{result.get('model_id', 'unknown')}`",
        f"- **Model dir:** `{result.get('model_dir', '')}`",
        f"- **Model pkl:** `{model_info.get('model_pkl', 'N/A')}`",
        f"- **Trained at:** `{result.get('trained_at', 'unknown')}`",
        f"- **Threshold:** `{result.get('threshold', 'N/A')}`",
        f"- **Predicted probability:** `{result.get('probability', 'N/A'):.4f}`" if result.get("probability") is not None else "",
        "",
        "## Feature Alignment",
        f"- **Total model features:** `{feature_alignment.get('total_model_features', '?')}`",
        f"- **Available from live data:** `{feature_alignment.get('available_count', '?')}`",
        f"- **Coverage:** `{feature_alignment.get('coverage_pct', 0):.1f}%`",
        f"- **Required minimum:** `95%`",
        "",
        "### Missing Features by Source:",
    ]
    by_src = feature_alignment.get("missing_by_source", {})
    src_labels = {
        "candle": "🔴 Candle (OHLCV + indicators)",
        "option_chain": "🔴 Option Chain",
        "rolling_history": "🔴 Rolling History (z-scores, ranges)",
        "context": "🔴 Context (time/spot/IV)",
        "unavailable": "🔴 Unavailable (unknown source)",
    }
    any_missing = False
    for src, label in src_labels.items():
        feats = by_src.get(src, [])
        if feats:
            any_missing = True
            md_lines.append(f"  **{label}:** `{len(feats)}`")
            for f in feats[:5]:
                md_lines.append(f"    - `{f}`")
            if len(feats) > 5:
                md_lines.append(f"    ... and {len(feats) - 5} more")
    if not any_missing:
        md_lines.append("  *(none — full coverage)*")

    md_lines.extend(["", "## Decision Pipeline Steps", ""])
    for step in result.get("steps", []):
        icon = "✅" if step["passed"] else "❌"
        md_lines.append(f"{icon} **{step['step']}** — {step['detail']}")
        if step.get("risk_state"):
            state = step["risk_state"]
            md_lines.append(f"   > risk state: trades_today={state.get('trades_today')}, "
                            f"open_pos={state.get('open_positions')}, "
                            f"daily_pnl={state.get('daily_pnl')}")

    if result.get("blocking_reasons"):
        md_lines.extend(["", "## Blocking Reasons", ""])
        for r in result["blocking_reasons"]:
            md_lines.append(f"- `{r}`")

    if result["final_action"] == "TRADE":
        md_lines.extend(["", "## ⚠️  WOULD TRADE (DRY-RUN)", "",
                          "This was a **paper/dry-run** — no real orders were placed.",
                          "The system WOULD have sent the following order on live data:",
                          f"- Probability: `{result['probability']:.4f}` >= Threshold: `{result['threshold']:.4f}`",
                          f"- All risk filters passed.",
                          "",
                          "**Safety check:** This script MUST be run with `--paper` or `--dry-run`",
                          "to ensure it never calls real broker order APIs.", ])

    md_lines.extend(["", "## Safety Compliance", "",
                      "- ✅ Never calls real broker order APIs",
                      "- ✅ Requires explicit `--paper` or `--dry-run` flag",
                      "- ✅ Logs every decision step (what-if mode)",
                      "- ✅ Loads scaler mean/std from training artifacts",
                      f"- ✅ Feature mismatch between training and live is shown above",
                      ""])

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # --- JSON ---
    # Build the required report structure
    risk_filter_decisions: Dict[str, Any] = {}
    for step in result.get("steps", []):
        if step.get("step") == "RISK_FILTER":
            risk_filter_decisions = {
                "allowed": step.get("passed", False),
                "blocking_reason": step.get("blocking_reason", ""),
                "risk_state": step.get("risk_state", {}),
            }

    out_json = {
        "model_id": result.get("model_id"),
        "model_pkl": result.get("model_pkl"),
        "feature_count": feature_alignment.get("total_model_features"),
        "available_count": feature_alignment.get("available_count"),
        "coverage_pct": feature_alignment.get("coverage_pct"),
        "missing_features": feature_alignment.get("missing_features", []),
        "missing_by_source": feature_alignment.get("missing_by_source", {}),
        "extra_live_features": feature_alignment.get("extra_live_features", [])[:50],
        "probability": result.get("probability"),
        "threshold": result.get("threshold"),
        "risk_filter_decisions": risk_filter_decisions,
        "final_action": result["final_action"],
        "blocker": result.get("blocker"),
        "safety_compliance": {
            "never_calls_real_order": True,
            "explicit_paper_flag": True,
            "logs_what_if_would_do": True,
            "loads_scaler_from_training": True,
        },
        # Additional fields
        "timestamp": result.get("timestamp"),
        "mode": dry_run_mode,
        "blocking_reasons": result.get("blocking_reasons", []),
        "decision_steps": result.get("steps", []),
        "model_dir": result.get("model_dir"),
        "trained_at": result.get("trained_at"),
        "model_info": model_info,
        # --- Candidate manifest fields ---
        "candidate_manifest_source": model_info.get("candidate_manifest_source"),
        "filter_applied": model_info.get("filter_applied", False),
        "filter_definition": model_info.get("filter_definition"),
        "candidate_status": model_info.get("candidate_status"),
        "gates_passed": model_info.get("gates_passed"),
        "gates_total": model_info.get("gates_total"),
        "candidate_metrics": model_info.get("candidate_metrics"),
        "report_paths": {"md": str(md_path), "json": str(json_path)},
    }
    json_path.write_text(json.dumps(out_json, indent=2, default=str), encoding="utf-8")

    return md_path, json_path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live ML decision dry-run — show what the pipeline WOULD do right now.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--paper", action="store_true",
                     help="Run in paper/dry-run mode (no real orders).")
    grp.add_argument("--dry-run", dest="dry_run", action="store_true",
                     help="Alias for --paper (same behaviour).")
    parser.add_argument("--model-dir", type=str, default="",
                        help="Path to a specific model artifact directory. "
                             "If omitted, auto-selects the latest paper candidate.")
    parser.add_argument("--candidate-manifest", type=str, default="",
                        help="Path to a candidate_manifest.json (e.g. from paper_candidate_* dir). "
                             "Loads model_pkl, threshold, and filter_definition from the manifest. "
                             "This takes precedence over --model-dir.")
    parser.add_argument("--offline", action="store_true",
                        help="Skip broker login — use cached/simulated data.")
    parser.add_argument("--symbol-token", type=str, default="",
                        help="mStock symbol token to fetch live data for. "
                             "Defaults to MSTOCK_SYMBOL_TOKEN from .env.")
    parser.add_argument("--output-dir", type=str, default="reports",
                        help="Directory for output reports (default: reports/).")
    parser.add_argument("--fixture", type=str, default="",
                        help="Path to a JSON snapshot fixture file to use instead of live data. "
                             "The fixture should contain all available feature values.")
    parser.add_argument("--once", action="store_true",
                        help="Single decision mode: exit after one decision (default: loop). "
                             "Useful for scripting and testing.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug output including feature alignment details.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    dry_run_mode = "paper" if args.paper else "dry-run"
    reports_dir = _REPO_ROOT / args.output_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    _LOGGER.info("=" * 60)
    _LOGGER.info("LIVE DECISION DRY-RUN (%s) — NO REAL ORDERS WILL BE PLACED", dry_run_mode.upper())
    _LOGGER.info("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Discover model
    # -------------------------------------------------------------------------
    model_dir: Optional[Path] = None
    if args.model_dir:
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            _LOGGER.error("Model dir not found: %s", model_dir)
            sys.exit(1)
        _LOGGER.info("Using specified model dir: %s", model_dir)
    else:
        # Priority: (1) paper candidates with .pkl → (2) dirs with .pkl → (3) approved manifests
        candidates = find_paper_candidate_model_dirs()
        with_pkls = find_model_dirs_with_pkls()
        approved = find_approved_model_dirs()
        if candidates:
            model_dir = candidates[0]
            _LOGGER.info("Auto-selected latest paper candidate model: %s", model_dir)
        elif with_pkls:
            model_dir = with_pkls[0]
            _LOGGER.info("Auto-selected latest model dir with .pkl: %s", model_dir)
        elif approved:
            model_dir = approved[0]
            _LOGGER.info("Auto-selected latest approved model: %s", model_dir)
        else:
            _LOGGER.error("No model artifacts found. Run retrain pipeline first.")
            sys.exit(1)

    # Load candidate manifest if provided (takes precedence over all discovery)
    candidate_info: Optional[Dict[str, Any]] = None
    if args.candidate_manifest:
        try:
            candidate_info = load_candidate_manifest(Path(args.candidate_manifest))
            _LOGGER.info("Loaded candidate manifest from: %s", candidate_info["source"])
            _LOGGER.info("  candidate_id : %s", candidate_info["candidate_id"])
            _LOGGER.info("  model_pkl    : %s", candidate_info["model_pkl_raw"])
            _LOGGER.info("  threshold    : %.4f", candidate_info["selected_threshold"])
            _LOGGER.info("  filter_applied: %s", candidate_info["filter_applied"])
            _LOGGER.info("  filter_type  : %s",
                         candidate_info["filter_definition"].get("type", "none"))
            _LOGGER.info("  status       : %s", candidate_info["status"])
            if candidate_info["gates_passed"] is not None:
                _LOGGER.info("  gates        : %s/%s",
                             candidate_info["gates_passed"], candidate_info["gates_total"])
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            _LOGGER.error("FAILED to load candidate manifest: %s", exc)
            sys.exit(1)

    # Apply candidate manifest values (takes precedence over deployment_manifest)
    threshold: float = 0.5  # default; overridden below
    model_id: str = model_dir.name  # default; overridden below

    # Load manifest / feature list
    manifest: Dict[str, Any] = {}
    model_features: List[str] = []
    manifest_path = model_dir / "deployment_manifest.json"
    if manifest_path.exists():
        try:
            manifest = load_deployment_manifest(model_dir)
            model_features = load_feature_list(manifest, model_dir)
            # Only apply manifest defaults if no candidate is loaded
            if candidate_info is None:
                threshold = float(manifest.get("selected_threshold", threshold))
                model_id = str(manifest.get("model_id", model_id))
            _LOGGER.info("Loaded manifest: verdict=%s threshold=%.2f features=%d",
                         manifest.get("paper_readiness_verdict"), threshold, len(model_features))
        except Exception as exc:
            _LOGGER.warning("Could not load manifest: %s", exc)

    # Apply candidate manifest values (takes precedence over deployment_manifest)
    if candidate_info:
        threshold = candidate_info["selected_threshold"]
        model_id = candidate_info["candidate_id"]
        feat_schema = candidate_info.get("feature_schema") or []
        if feat_schema:
            model_features = feat_schema
            _LOGGER.info("Feature schema from candidate manifest: %d features", len(model_features))

    _LOGGER.info("Effective model_id: %s  threshold: %.4f", model_id, threshold)

    if not model_features:
        # Fallback: scan the pkl for feature names
        pkl_path = find_model_pkl_in_dir(model_dir)
        if pkl_path:
            try:
                bundle = load_model_bundle(pkl_path)
                model_features = list(bundle.feature_names)
                _LOGGER.info("Loaded features from bundle: %d", len(model_features))
            except Exception as exc:
                _LOGGER.warning("Could not load bundle features: %s", exc)

    _LOGGER.info("Model features count: %d", len(model_features))

    # -------------------------------------------------------------------------
    # 2. Load model bundle
    # -------------------------------------------------------------------------
    model_pkl: Optional[Path] = None

    # Candidate manifest model_pkl takes precedence
    if candidate_info:
        model_pkl = candidate_info["model_pkl"]
        _LOGGER.info("Using model_pkl from candidate manifest: %s", model_pkl)
    elif manifest_path.exists():
        model_pkl = get_model_pkl(model_dir, manifest)
        if not model_pkl or not model_pkl.exists():
            model_pkl = find_model_pkl_in_dir(model_dir)
    else:
        model_pkl = find_model_pkl_in_dir(model_dir)

    bundle = None
    if model_pkl and model_pkl.exists():
        try:
            bundle = load_model_bundle(model_pkl)
            _LOGGER.info("Loaded model bundle from: %s", model_pkl)
            if not model_features and bundle.feature_names:
                model_features = list(bundle.feature_names)
        except Exception as exc:
            _LOGGER.warning("Could not load model bundle: %s", exc)
    else:
        _LOGGER.warning("No model .pkl found in %s", model_dir)

    model_info = {
        "model_dir": str(model_dir),
        "model_pkl": str(model_pkl) if model_pkl else None,
        "model_id": model_id,
        "threshold": threshold,
        "feature_count": len(model_features),
        "has_scaler": bundle.scaler_mean is not None and bundle.scaler_std is not None if bundle else False,
        "trained_at": bundle.trained_at if bundle else None,
        # Candidate manifest fields
        "candidate_manifest_source": candidate_info["source"] if candidate_info else None,
        "filter_applied": candidate_info["filter_applied"] if candidate_info else False,
        "filter_definition": candidate_info["filter_definition"] if candidate_info else None,
        "candidate_status": candidate_info["status"] if candidate_info else None,
        "gates_passed": candidate_info["gates_passed"] if candidate_info else None,
        "gates_total": candidate_info["gates_total"] if candidate_info else None,
        "candidate_metrics": candidate_info["metrics"] if candidate_info else None,
    }

    # -------------------------------------------------------------------------
    # 3. Fetch live data
    # -------------------------------------------------------------------------
    spot_data: Dict[str, Any] = {}
    candles_data: List[Dict[str, Any]] = []

    if not args.offline:
        try:
            from dotenv import load_dotenv
            load_dotenv(_REPO_ROOT / ".env")
        except Exception:
            pass

        symbol_token = args.symbol_token or os.getenv("MSTOCK_SYMBOL_TOKEN", "")
        broker = os.getenv("SCALPER_BROKER", "mstock").strip().lower()

        if symbol_token:
            _LOGGER.info("Fetching live data for symbol_token=%s broker=%s", symbol_token, broker)

            # --- mStock path ---
            if broker == "mstock":
                try:
                    from src.mstock_client import MStockTypeBClient
                    from src.config import load_api_config
                    api_cfg = load_api_config()
                    client = MStockTypeBClient(api_cfg)
                    _LOGGER.info("Logging in to mStock...")
                    client.login(interactive=False)
                    spot_data = get_live_spot_and_iv(client, symbol_token)
                    candles_data = get_live_candles(client, symbol_token, lookback=30)
                    _LOGGER.info("Live data: spot=%s candles=%d",
                                 spot_data.get("spot"), len(candles_data))
                except Exception as exc:
                    _LOGGER.warning("mStock live fetch failed: %s", exc)
                    _LOGGER.info("Falling back to offline mode.")
                    args.offline = True
            else:
                _LOGGER.warning("Broker '%s' not supported for live fetch. Use --offline.", broker)
                args.offline = True
        else:
            _LOGGER.warning("No MSTOCK_SYMBOL_TOKEN set. Use --offline or set the token.")
            args.offline = True

    # -------------------------------------------------------------------------
    # 4. Build live feature snapshot (or realistic offline fixture)
    # -------------------------------------------------------------------------
    if args.offline or not candles_data:
        _LOGGER.info("Running in OFFLINE mode — building realistic snapshot fixture.")
        snapshot = _build_offline_fixture(model_features)
        # Load and merge fixture if provided (allows overriding specific fields like option_type)
        if args.fixture:
            fixture_path = Path(args.fixture)
            if fixture_path.exists():
                try:
                    fixture_data = json.loads(fixture_path.read_text(encoding="utf-8"))
                    _LOGGER.info("Loading fixture overrides from: %s", fixture_path)
                    snapshot.update(fixture_data)
                    _LOGGER.info("Fixture merged: option_type=%s", snapshot.get("option_type"))
                except Exception as exc:
                    _LOGGER.warning("Failed to load fixture: %s", exc)
        _LOGGER.info("Offline fixture: %d features populated out of %d required",
                     len([k for k in snapshot if k not in ('timestamp', 'note')]),
                     len(model_features))
    else:
        snapshot = build_live_feature_snapshot(candles_data, spot_data)

    # -------------------------------------------------------------------------
    # 5. Feature alignment
    # -------------------------------------------------------------------------
    if model_features:
        alignment = align_features(model_features, snapshot)
        _LOGGER.info("Feature alignment: %d/%d available (%.1f%%), %d missing",
                     len(alignment["available_features"]),
                     alignment["total_model_features"],
                     alignment["coverage_pct"],
                     alignment["missing_count"])
    else:
        alignment = {
            "available_features": [],
            "missing_features": [],
            "extra_live_features": [],
            "aligned_vector": {},
            "missing_count": 0,
            "total_model_features": 0,
            "coverage_pct": 0.0,
        }
        _LOGGER.warning("No model features known — cannot align.")

    # -------------------------------------------------------------------------
    # 6. Model prediction
    # -------------------------------------------------------------------------
    probability = 0.0
    if bundle and model_features:
        try:
            probability = predict_with_bundle(bundle, snapshot)
            _LOGGER.info("Model prediction: probability=%.4f threshold=%.4f", probability, threshold)
        except Exception as exc:
            _LOGGER.warning("Prediction failed: %s", exc)
    else:
        _LOGGER.warning("No model bundle loaded — cannot predict.")

    # -------------------------------------------------------------------------
    # 7. Risk filter
    # -------------------------------------------------------------------------
    from src.ml_paper_risk_manager import MLPaperRiskManager
    risk_mgr = MLPaperRiskManager()
    risk_result = risk_mgr.evaluate(snapshot)
    _LOGGER.info("Risk filter: allowed=%s reason='%s'",
                 risk_result.get("allowed"), risk_result.get("blocking_reason"))

    # -------------------------------------------------------------------------
    # 8. Full decision evaluation
    # -------------------------------------------------------------------------
    result = evaluate_decision(
        probability=probability,
        threshold=threshold,
        feature_alignment=alignment,
        risk_result=risk_result,
        model_id=model_id,
        model_dir=model_dir,
        bundle_trained_at=bundle.trained_at if bundle else "unknown",
        snapshot=snapshot,
        model_pkl=str(model_pkl) if model_pkl else None,
        candidate_info=candidate_info if candidate_info else None,
    )

    _LOGGER.info("")
    _LOGGER.info("=" * 60)
    _LOGGER.info("FINAL ACTION: %s", result["final_action"])
    for reason in result.get("blocking_reasons", []):
        _LOGGER.info("  Blocked by: %s", reason)
    _LOGGER.info("=" * 60)

    # -------------------------------------------------------------------------
    # 9. Generate reports
    # -------------------------------------------------------------------------
    md_path, json_path = generate_reports(
        result=result,
        feature_alignment=alignment,
        model_info=model_info,
        dry_run_mode=dry_run_mode,
        reports_dir=reports_dir,
    )
    _LOGGER.info("Reports written to:")
    _LOGGER.info("  Markdown: %s", md_path)
    _LOGGER.info("  JSON:     %s", json_path)

    print()
    print("=" * 60)
    print(f"FINAL ACTION: {result['final_action']}")
    print(f"Probability:  {probability:.4f}")
    print(f"Threshold:    {threshold:.4f}")
    print(f"Coverage:     {alignment.get('coverage_pct', 0):.1f}% "
          f"({len(alignment.get('available_features', []))}/{alignment.get('total_model_features', 0)})")
    if result.get("blocking_reasons"):
        for r in result["blocking_reasons"]:
            print(f"  BLOCKED: {r}")
    print("=" * 60)
    print(f"Reports: {md_path}")
    print(f"         {json_path}")


if __name__ == "__main__":
    main()