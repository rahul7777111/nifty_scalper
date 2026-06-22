"""Paper Forward readiness, cost-quality classification, and LTP helpers."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

XGBOOST_INSTALL_HINT = r".\.venv\Scripts\pip.exe install xgboost"

LTP_SOURCE_ALIASES = {
    "BROKER_LIVE_LTP": "BROKER_LTP",
    "REAL_CHAIN_LTP": "CHAIN_LTP",
    "CHAIN_OPTION_LTP": "CHAIN_LTP",
    "CHAIN_MARK": "CHAIN_LTP",
    "STALE_SYNTHETIC": "SYNTHETIC",
    "SYNTHETIC_MARK": "SYNTHETIC",
    "SYNTHETIC_FALLBACK": "SYNTHETIC",
}


def normalize_ltp_source(source: str) -> str:
    src = str(source or "").strip().upper()
    return LTP_SOURCE_ALIASES.get(src, src or "UNKNOWN")


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def is_xgboost_missing_error(message: str) -> bool:
    msg = str(message or "").lower()
    return "no module named 'xgboost'" in msg or "no module named xgboost" in msg


def artifact_expects_xgboost(
    cand_meta: Optional[Dict[str, Any]] = None,
    *,
    model_type: str = "",
    artifact_dir: Optional[str] = None,
) -> bool:
    meta = cand_meta or {}
    tokens = [
        str(meta.get("model_name") or ""),
        str(meta.get("model_family") or ""),
        str(model_type or ""),
    ]
    if artifact_dir:
        tokens.append(str(artifact_dir))
    blob = " ".join(tokens).lower()
    return "xgboost" in blob or "xgb" in blob


def classify_load_error(
    exc: BaseException,
    *,
    cand_meta: Optional[Dict[str, Any]] = None,
    model_type: str = "",
    artifact_dir: Optional[str] = None,
) -> str:
    msg = str(exc)
    if is_xgboost_missing_error(msg):
        return "XGBOOST_NOT_INSTALLED"
    if artifact_expects_xgboost(cand_meta, model_type=model_type, artifact_dir=artifact_dir) and not xgboost_available():
        return "XGBOOST_NOT_INSTALLED"
    return f"pickle_load_failed:{exc}"


def _row_is_synthetic(row: Optional[Dict[str, Any]]) -> bool:
    if not row:
        return False
    return bool(row.get("synthetic")) or str(row.get("chain_source") or "") == "BLACK_SCHOLES_SYNTHETIC"


def _has_broker_numeric(row: Optional[Dict[str, Any]], *keys: str) -> bool:
    if not row or _row_is_synthetic(row):
        return False
    for key in keys:
        val = row.get(key)
        if val is None or val == "":
            continue
        try:
            if float(val) != 0.0 or key in ("volume", "oi", "open_interest", "traded_volume"):
                return True
        except Exception:
            return True
    return False


def compute_cost_quality(
    *,
    row: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, Any]] = None,
    used_default_fee: bool = False,
    used_default_spread: bool = False,
    used_chain_pctile: bool = False,
) -> Tuple[str, List[str]]:
    """Return (REAL|APPROX, approx_flags). REAL only when broker bid/ask/volume/OI drive costs."""
    features = features or {}
    row = row or {}
    approx_flags: List[str] = []

    bid = row.get("bid") or row.get("bid_price") or features.get("bid") or features.get("bid_price")
    ask = row.get("ask") or row.get("ask_price") or features.get("ask") or features.get("ask_price")
    vol = row.get("volume") or row.get("traded_volume") or features.get("volume") or features.get("traded_volume")
    oi = row.get("oi") or row.get("open_interest") or features.get("oi") or features.get("open_interest")

    has_broker_spread = _has_broker_numeric(row, "bid", "bid_price", "ask", "ask_price") or (
        bid not in (None, "") and ask not in (None, "")
    )
    has_broker_vol = _has_broker_numeric(row, "volume", "traded_volume") or vol not in (None, "")
    has_broker_oi = _has_broker_numeric(row, "oi", "open_interest") or oi not in (None, "")

    if used_default_fee:
        approx_flags.append("default_fee_bps")
    if used_default_spread:
        approx_flags.append("default_spread_bps")
    if used_chain_pctile:
        approx_flags.append("live_chain_pctile_approx")

    broker_cost_inputs = has_broker_spread and has_broker_vol and has_broker_oi and not _row_is_synthetic(row)
    if broker_cost_inputs and not approx_flags:
        return "REAL", []
    if broker_cost_inputs and approx_flags:
        return "APPROX", approx_flags
    if approx_flags:
        return "APPROX", approx_flags
    if has_broker_spread:
        return "APPROX", ["partial_broker_inputs"]
    return "APPROX", ["missing_broker_quote_inputs"]


def is_stale_last_good_mark(rec: Dict[str, Any], ttl_sec: float) -> bool:
    if not rec:
        return True
    age = time.time() - float(rec.get("timestamp") or 0.0)
    return age > float(ttl_sec)


def last_good_mark_allowed(rec: Optional[Dict[str, Any]], ttl_sec: float) -> bool:
    if not rec:
        return False
    try:
        px = float(rec.get("price") or 0.0)
    except Exception:
        return False
    if px <= 0:
        return False
    return not is_stale_last_good_mark(rec, ttl_sec)


def ltp_fallback_priority() -> List[str]:
    return ["BROKER_LTP", "CHAIN_LTP", "MID_BID_ASK", "LAST_GOOD_MARK", "SYNTHETIC"]


def classify_readiness_reason(no_trade_reason: str, pred_error: str = "") -> str:
    reason = str(no_trade_reason or pred_error or "").upper()
    if "XGBOOST_NOT_INSTALLED" in reason:
        return "blocked_dependency"
    if reason in ("FEATURE_VECTOR_INVALID", "FEATURES_MISSING") or "FEATURE_VECTOR" in reason:
        return "blocked_feature"
    if reason.startswith("LOW_CONFIDENCE") or "LOW_CONFIDENCE" in reason:
        return "low_confidence"
    if str(no_trade_reason or "").startswith("BUY_"):
        return "signal"
    if pred_error in ("", None) and no_trade_reason in ("", None):
        return "predictable"
    if "MODEL_OUTPUT_INVALID" in reason or "PREDICT_EXCEPTION" in reason:
        return "blocked_dependency" if "XGBOOST" in reason else "blocked_other"
    return "other"


def summarize_readiness(
    *,
    candidates_total: int,
    results: List[Dict[str, Any]],
    cost_qualities: List[str],
    ltp_sources: List[str],
    data_quality: str = "",
) -> Dict[str, Any]:
    blocked_feature = 0
    blocked_dependency = 0
    low_confidence = 0
    signals = 0
    predictable = 0

    for dec in results:
        ntr = str(dec.get("no_trade_reason") or "")
        pred_err = str((dec.get("debug") or {}).get("predict", {}).get("error") or "")
        if dec.get("predict_attempted") and isinstance(dec.get("confidence"), (int, float)):
            predictable += 1
        bucket = classify_readiness_reason(ntr, pred_err)
        if bucket == "blocked_feature":
            blocked_feature += 1
        elif bucket == "blocked_dependency":
            blocked_dependency += 1
        elif bucket == "low_confidence":
            low_confidence += 1
        elif bucket == "signal" or str(dec.get("final_signal", "")).startswith("BUY_"):
            signals += 1

    norm_ltp = [normalize_ltp_source(s) for s in ltp_sources if s]
    return {
        "candidates_total": candidates_total,
        "candidates_predictable": predictable,
        "candidates_blocked_feature": blocked_feature,
        "candidates_blocked_dependency": blocked_dependency,
        "candidates_low_confidence": low_confidence,
        "candidates_signal": signals,
        "cost_real": sum(1 for q in cost_qualities if str(q).upper() == "REAL"),
        "cost_approx": sum(1 for q in cost_qualities if str(q).upper() != "REAL"),
        "ltp_broker_ok": sum(1 for s in norm_ltp if s == "BROKER_LTP"),
        "ltp_fallback": sum(1 for s in norm_ltp if s not in ("", "BROKER_LTP", "UNKNOWN")),
        "data_quality": data_quality or "",
    }


def format_readiness_log(summary: Dict[str, Any]) -> str:
    return (
        f"[PF-READINESS] candidates_total={summary.get('candidates_total', 0)} "
        f"candidates_predictable={summary.get('candidates_predictable', 0)} "
        f"candidates_blocked_feature={summary.get('candidates_blocked_feature', 0)} "
        f"candidates_blocked_dependency={summary.get('candidates_blocked_dependency', 0)} "
        f"candidates_low_confidence={summary.get('candidates_low_confidence', 0)} "
        f"candidates_signal={summary.get('candidates_signal', 0)} "
        f"cost_real={summary.get('cost_real', 0)} "
        f"cost_approx={summary.get('cost_approx', 0)} "
        f"ltp_broker_ok={summary.get('ltp_broker_ok', 0)} "
        f"ltp_fallback={summary.get('ltp_fallback', 0)} "
        f"data_quality={summary.get('data_quality') or '-'}"
    )
