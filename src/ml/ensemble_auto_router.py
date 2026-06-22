"""Optional ensemble auto router for direct model-family voting.

The router is deliberately independent of the existing candidate router. It
loads four configured model-family artifacts, scores CE and PE views of the
same snapshot, and returns one paper-mode decision with explicit failure states.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import pickle
import statistics
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as dt_time, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "ensemble_auto_router.json"

DECISION_BUY_CE = "BUY_CE"
DECISION_BUY_PE = "BUY_PE"
DECISION_NO_TRADE = "NO_TRADE"

STATUS_OK = "OK"
STATUS_FEATURES_MISSING = "FEATURES_MISSING"
STATUS_ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"
STATUS_MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
STATUS_PREDICT_FAILED = "PREDICT_FAILED"
STATUS_FEATURE_VECTOR_INVALID = "FEATURE_VECTOR_INVALID"
STATUS_UNKNOWN_LABEL_MAPPING = "UNKNOWN_LABEL_MAPPING"

FEATURE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "ltp": ("last_price", "lastPrice", "option_ltp", "close", "option_price", "traded_price", "last_traded_price"),
    "oi": ("open_interest", "openInterest"),
    "strike": ("strike_price", "strikePrice"),
    "ce_pe": ("option_type", "type", "right", "optionType"),
    "symbol": ("trading_symbol", "tradingsymbol", "tradingSymbol", "tsym"),
    "expiry": ("expiry_date", "expiryDate", "expDate"),
    "ts": ("timestamp", "datetime", "time", "date"),
    "bid": ("best_bid", "best_bid_price", "bid_price", "bidPrice"),
    "ask": ("best_ask", "best_ask_price", "ask_price", "askPrice", "offer"),
}

SAFE_ZERO_FEATURES = {
    "volume",
    "oi",
    "oi_change_pct",
    "iv_change_pct",
    "spread_pct",
}

PRICE_FEATURE_NAMES = {
    "open",
    "high",
    "low",
    "close",
    "ltp",
    "last_price",
    "bid",
    "ask",
    "spot",
    "spot_close",
    "ctx_spot",
    "underlying_price",
    "strike",
    "strike_price",
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "paper_mode_enabled": True,
    "config_version": 1,
    "artifact_root": "artifacts",
    "artifact_search_paths": [
        "artifacts",
        "artifacts/models",
        "artifacts/candidates",
        "artifacts/ml_signals",
    ],
    "ensemble_min_confidence": 0.65,
    "ensemble_min_direction_edge": 0.20,
    "ensemble_max_model_disagreement": 0.35,
    "ensemble_min_valid_models": 2,
    "debug_allow_low_edge_paper_trade": False,
    "models": {
        "elasticnet": {
            "enabled": True,
            "label": "ElasticNet Logistic Regression",
            "artifact_path": "artifacts/candidates/elasticnet_BOTH_directional_auto_BOTH_dir_auto_t30_20260610_130953",
            "weight": 0.30,
            "side_policy": "AUTO_DIRECTIONAL",
        },
        "logistic_regression": {
            "enabled": True,
            "label": "Logistic Regression",
            "artifact_path": "artifacts/candidates/logistic_regression_BOTH_directional_auto_BOTH_dir_auto_t30_20260610_130953",
            "weight": 0.20,
            "side_policy": "AUTO_DIRECTIONAL",
        },
        "calibrated_logistic_regression": {
            "enabled": True,
            "label": "Calibrated Logistic Regression",
            "artifact_path": "artifacts/candidates/calibrated_logistic_regression_BOTH_symmetric_BOTH_sym_t30_20260610_130953",
            "weight": 0.25,
            "side_policy": "BOTH_SYMMETRIC",
        },
        "xgboost": {
            "enabled": True,
            "label": "XGBoost",
            "artifact_path": "artifacts/candidates/xgboost_volatility_breakout_vol_breakout_t30_20260610_130953",
            "weight": 0.25,
            "side_policy": "BOTH",
        },
    },
    "thresholds": {
        "ensemble_min_confidence": 0.65,
        "ensemble_min_direction_edge": 0.20,
        "ensemble_max_model_disagreement": 0.35,
        "ensemble_min_valid_models": 2,
    },
    "liquidity": {
        "max_spread_pct": 0.12,
        "min_volume": 0,
        "min_oi": 0,
        "min_ltp": 0.05,
    },
    "duplicate_trade_rules": {
        "max_trades_per_day": 3,
        "one_position_per_side": True,
    },
    "paper_trading": {
        "lot_size": 65,
        "brokerage_per_order": 5.0,
        "slippage_pct": 0.001,
        "cost_bps": 1.5,
        "trade_log_path": "reports/ensemble_auto_router_trades.csv",
        "summary_path": "reports/ensemble_auto_router_summary.json",
    },
    "exits": {
        "stop_loss_pct": 0.10,
        "target_pct": 0.20,
        "max_hold_bars": 5,
        "force_exit_time": "15:20",
        "entry_cutoff_time": "15:00",
        "daily_loss_limit": -2500.0,
    },
    "logging": {
        "debug": True,
        "log_model_details": True,
    },
}

EXPECTED_MODEL_ORDER = (
    "elasticnet",
    "logistic_regression",
    "calibrated_logistic_regression",
    "xgboost",
)


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    cfg_path = Path(path)
    data: Dict[str, Any] = {}
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("[ENSEMBLE-ROUTER] config_load_failed path=%s error=%s", cfg_path, exc)
    return normalize_config(_deep_merge(DEFAULT_CONFIG, data))


def normalize_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = _deep_merge(DEFAULT_CONFIG, dict(config or {}))
    thresholds = dict(cfg.get("thresholds") or {})
    for key in (
        "ensemble_min_confidence",
        "ensemble_min_direction_edge",
        "ensemble_max_model_disagreement",
        "ensemble_min_valid_models",
    ):
        if key in cfg:
            thresholds[key] = cfg[key]
    cfg["thresholds"] = thresholds
    return cfg


def write_default_config(path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    cfg_path = Path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")


def _resolve_path(value: object, *, base: Path = REPO_ROOT) -> Path:
    raw = str(value or "").strip()
    p = Path(raw)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            obj = json.loads(path.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}
    return {}


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _finite_float(value: object) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping.get(key) not in (None, ""):
            return mapping.get(key)
    return None


def _latest_mapping(value: object) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in reversed(value):
            if isinstance(item, Mapping):
                return dict(item)
    return {}


def _extract_latest_candle(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    for key in ("candle", "latest_candle", "spot_candle", "last_candle"):
        row = _latest_mapping(snapshot.get(key))
        if row:
            return row
    for key in ("candles", "spot_candles", "ohlc", "bars"):
        row = _latest_mapping(snapshot.get(key))
        if row:
            return row
    return {}


def _parse_date_like(value: object) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raw = str(value or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d%b%Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(raw[: max(len(raw), len(fmt))], fmt).date()
        except Exception:
            pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except Exception:
        return None


def _expiry_days(expiry: object, ts: object = None) -> Optional[float]:
    exp = _parse_date_like(expiry)
    if exp is None:
        return None
    base = _parse_date_like(ts) or datetime.now().date()
    return float(max(0, (exp - base).days))


def _parse_datetime_like(value: object) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            pass
    return None


def _is_weekly_expiry(expiry: object) -> float:
    exp = _parse_date_like(expiry)
    if exp is None:
        return 0.0
    # Monthly NIFTY expiry is normally the last Thursday; anything earlier is weekly.
    if exp.weekday() != 3:
        return 1.0
    next_week = exp.toordinal() + 7
    try:
        return 0.0 if date.fromordinal(next_week).month != exp.month else 1.0
    except Exception:
        return 1.0


def _normalise_live_snapshot(snapshot: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    out = dict(snapshot)
    alias_sources: Dict[str, str] = {}
    for canonical, aliases in FEATURE_ALIASES.items():
        if canonical not in out or out.get(canonical) in (None, ""):
            for alias in aliases:
                if alias in out and out.get(alias) not in (None, ""):
                    out[canonical] = out.get(alias)
                    alias_sources[canonical] = alias
                    break
    return out, alias_sources


def _positive_float_from(mapping: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = _finite_float(mapping.get(key))
        if value is not None and value > 0:
            return value
    return None


def _derive_option_price_fields(data: Mapping[str, Any]) -> Dict[str, float]:
    out = dict(data)
    price = _positive_float_from(
        out,
        ("ltp", "last_price", "lastPrice", "close", "option_price", "traded_price", "last_traded_price", "ctx_option_price"),
    )
    bid = _positive_float_from(out, ("bid", "best_bid", "best_bid_price", "bid_price", "bidPrice"))
    ask = _positive_float_from(out, ("ask", "best_ask", "best_ask_price", "ask_price", "askPrice", "offer"))
    if price is None and bid is not None and ask is not None:
        price = (bid + ask) / 2.0

    spot = _positive_float_from(out, ("spot", "spot_close", "ctx_spot", "underlying_price", "underlyingValue"))
    strike = _positive_float_from(out, ("strike", "strike_price", "strikePrice"))
    side = _normalise_side(out.get("option_type") or out.get("ce_pe") or out.get("right"))
    intrinsic = 0.0
    if spot is not None and strike is not None:
        intrinsic = max(spot - strike, 0.0) if side == "CE" else max(strike - spot, 0.0)

    raw_extrinsic = _finite_float(out.get("extrinsic_value"))
    if price is None and raw_extrinsic is not None:
        price = intrinsic + max(raw_extrinsic, 0.0)
    if price is None or price <= 0:
        return {}

    if bid is None:
        bid = price
    if ask is None:
        ask = price
    mid = (bid + ask) / 2.0
    spread = max(0.0, ask - bid)
    spread_pct = spread / mid if mid > 0 else 0.0
    extrinsic = max(price - intrinsic, 0.0)
    ltp_vs_mid_diff = price - mid
    ltp_vs_mid_diff_pct = ltp_vs_mid_diff / mid if mid > 0 else 0.0
    moneyness = (strike / spot) if spot and strike is not None else 0.0
    distance_from_atm = abs(strike - spot) if spot is not None and strike is not None else 0.0
    return {
        "ltp": price,
        "ctx_option_price": price,
        "bid": bid,
        "ask": ask,
        "mid_price": mid,
        "bid_ask_spread": spread,
        "spread_pct": spread_pct,
        "intrinsic_value": intrinsic,
        "extrinsic_value": extrinsic,
        "ltp_vs_mid_diff": ltp_vs_mid_diff,
        "ltp_vs_mid_diff_pct": ltp_vs_mid_diff_pct,
        "moneyness": moneyness,
        "option_moneyness": moneyness,
        "distance_from_atm": distance_from_atm,
    }


def _log_feature_row(side: str, row: Mapping[str, Any]) -> None:
    derived = _derive_option_price_fields(row)
    if not derived:
        return
    line = (
        "[ENSEMBLE-FEATURE] "
        f"side={_normalise_side(side)} "
        f"symbol={row.get('symbol') or row.get('trading_symbol') or row.get('tradingsymbol') or '-'} "
        f"strike={row.get('strike') or row.get('strike_price') or row.get('strikePrice') or '-'} "
        f"ltp={derived.get('ltp')} bid={derived.get('bid')} ask={derived.get('ask')} "
        f"mid={derived.get('mid_price')} intrinsic={derived.get('intrinsic_value')} "
        f"extrinsic={derived.get('extrinsic_value')}"
    )
    print(line, flush=True)
    LOGGER.info(line)


def _side_numeric(snapshot: Mapping[str, Any]) -> float:
    side = _normalise_side(snapshot.get("option_type") or snapshot.get("ce_pe") or snapshot.get("right"))
    if side == "CE":
        return 1.0
    if side == "PE":
        return -1.0
    return 0.0


def _safe_default_allowed(feature: str) -> bool:
    name = feature.lower()
    if name in SAFE_ZERO_FEATURES:
        return True
    if name.startswith(("is_", "has_", "flag_")):
        return True
    if name.startswith(("ret_", "vol_", "dist_", "ctx_", "regime_", "pivot_", "spread_", "volume_", "premium_", "liq_")):
        return True
    if name.endswith(("_pct", "_ratio", "_change", "_change_pct", "_chg", "_delta", "_zscore", "_score")):
        return True
    if any(token in name for token in ("momentum", "trend", "slope", "rolling", "ema", "sma", "rsi", "atr", "macd", "vwap", "theta", "vega", "gamma", "delta", "rho", "iv", "adx", "choppiness", "supertrend", "engulfing", "doji", "hammer", "star", "percentile", "pctile", "rank", "quality", "cost", "slippage", "brokerage", "fee", "strength", "classifier", "imbalance")):
        return True
    if any(token in name for token in ("count", "volume", "vol_", "oi_", "depth", "imbalance")):
        return True
    return False


def _is_price_feature(feature: str) -> bool:
    name = feature.lower()
    if name in PRICE_FEATURE_NAMES:
        return True
    return any(token in name for token in ("price", "premium")) and not name.endswith(("_change_pct", "_pct", "_ratio"))


def _feature_value(feature: str, data: Mapping[str, Any], candle: Mapping[str, Any]) -> Tuple[Optional[float], str]:
    name = feature.lower()
    spot = _finite_float(_first_present(data, ("spot", "spot_close", "ctx_spot", "underlying_price", "underlyingValue")))
    strike_for_price = _finite_float(_first_present(data, ("strike", "strike_price", "strikePrice")))
    side_for_price = _normalise_side(data.get("option_type") or data.get("ce_pe") or data.get("right"))
    intrinsic_for_price: Optional[float] = None
    if strike_for_price is not None and spot is not None:
        intrinsic_for_price = max(0.0, spot - strike_for_price) if side_for_price == "CE" else max(0.0, strike_for_price - spot)
    ltp = _finite_float(
        _first_present(
            data,
            ("ltp", "last_price", "lastPrice", "option_ltp", "option_price", "traded_price", "last_traded_price", "close"),
        )
    )
    bid = _finite_float(_first_present(data, ("bid", "best_bid", "best_bid_price", "bid_price", "bidPrice")))
    ask = _finite_float(_first_present(data, ("ask", "best_ask", "best_ask_price", "ask_price", "askPrice", "offer")))
    if ltp is None and bid is not None and ask is not None:
        ltp = (bid + ask) / 2.0
    if ltp is None and intrinsic_for_price is not None:
        raw_extrinsic = _finite_float(data.get("extrinsic_value"))
        if raw_extrinsic is not None:
            ltp = intrinsic_for_price + raw_extrinsic
    if ltp is not None:
        if bid is None:
            bid = ltp
        if ask is None:
            ask = ltp
    base_price = ltp if ltp is not None and ltp > 0 else spot

    raw = _finite_float(data.get(feature))
    if raw is not None:
        return raw, "raw"
    if name in FEATURE_ALIASES:
        alias_val = _finite_float(_first_present(data, FEATURE_ALIASES[name]))
        if alias_val is not None:
            return alias_val, "alias"
    if name == "open":
        return _finite_float(candle.get("open")) or _finite_float(data.get("open")) or base_price, "derived"
    if name == "high":
        return _finite_float(candle.get("high")) or _finite_float(data.get("high")) or base_price, "derived"
    if name == "low":
        return _finite_float(candle.get("low")) or _finite_float(data.get("low")) or base_price, "derived"
    if name == "close":
        return _finite_float(candle.get("close")) or spot or ltp, "derived"
    if name in {"open_spot", "spot_open"}:
        return _finite_float(candle.get("open")) or spot, "derived"
    if name in {"high_spot", "spot_high"}:
        return _finite_float(candle.get("high")) or spot, "derived"
    if name in {"low_spot", "spot_low"}:
        return _finite_float(candle.get("low")) or spot, "derived"
    if name in {"volume_spot", "spot_volume", "last_volume"}:
        return _finite_float(candle.get("volume")) or _finite_float(data.get("volume")) or 0.0, "default"
    if name in {"last_open"}:
        return _finite_float(candle.get("open")) or _finite_float(data.get("open")) or base_price, "derived"
    if name in {"last_high"}:
        return _finite_float(candle.get("high")) or _finite_float(data.get("high")) or base_price, "derived"
    if name in {"last_low"}:
        return _finite_float(candle.get("low")) or _finite_float(data.get("low")) or base_price, "derived"
    if name in {"last_close"}:
        return _finite_float(candle.get("close")) or _finite_float(data.get("close")) or base_price, "derived"
    if name in {"ltp", "last_price", "option_ltp", "option_price", "traded_price"}:
        return ltp, "derived"
    if name in {"bid", "best_bid", "bid_price"}:
        return bid if bid is not None else ltp, "derived"
    if name in {"ask", "best_ask", "ask_price"}:
        return ask if ask is not None else ltp, "derived"
    if name in {"spot", "spot_close", "ctx_spot", "underlying_price"}:
        return spot if spot is not None else ltp, "derived"
    if name in {"strike", "strike_price"}:
        return _finite_float(_first_present(data, ("strike", "strike_price", "strikePrice"))), "alias"
    if name in {"volume", "vol"}:
        return _finite_float(_first_present(data, ("volume", "traded_volume", "totalTradedVolume"))) or 0.0, "default"
    if name in {"oi", "open_interest"}:
        return _finite_float(_first_present(data, ("oi", "open_interest", "openInterest"))) or 0.0, "default"
    if name == "dte_days":
        return _expiry_days(data.get("expiry") or data.get("expiry_date"), data.get("ts") or data.get("timestamp")), "derived"
    if name == "is_weekly":
        return _is_weekly_expiry(data.get("expiry") or data.get("expiry_date")), "derived"
    if name in {"time_to_expiry_days"}:
        return _expiry_days(data.get("expiry") or data.get("expiry_date"), data.get("ts") or data.get("timestamp")), "derived"
    if name in {"time_to_expiry_years"}:
        dte = _expiry_days(data.get("expiry") or data.get("expiry_date"), data.get("ts") or data.get("timestamp"))
        return (dte / 365.0) if dte is not None else None, "derived"
    if name in {"ctx_dte_norm"}:
        dte = _expiry_days(data.get("expiry") or data.get("expiry_date"), data.get("ts") or data.get("timestamp"))
        return (dte / 365.0) if dte is not None else 0.0, "derived"
    if name in {"weekday", "weekday_spot", "month"}:
        dt = _parse_datetime_like(data.get("ts") or data.get("timestamp") or data.get("datetime")) or datetime.now()
        return float(dt.weekday() if name != "month" else dt.month), "derived"
    if name in {"ctx_time_sin", "ctx_time_cos"}:
        dt = _parse_datetime_like(data.get("ts") or data.get("timestamp") or data.get("datetime")) or datetime.now()
        minutes = dt.hour * 60 + dt.minute
        angle = 2.0 * math.pi * (minutes / 1440.0)
        return (math.sin(angle) if name.endswith("_sin") else math.cos(angle)), "derived"
    if name in {"ce_pe", "option_type"}:
        return _side_numeric(data), "alias"
    if name == "option_type_ce":
        return 1.0 if _normalise_side(data.get("option_type")) == "CE" else 0.0, "derived"
    if name == "option_type_pe":
        return 1.0 if _normalise_side(data.get("option_type")) == "PE" else 0.0, "derived"

    open_v = _finite_float(candle.get("open")) or _finite_float(data.get("open")) or base_price
    high_v = _finite_float(candle.get("high")) or _finite_float(data.get("high")) or base_price
    low_v = _finite_float(candle.get("low")) or _finite_float(data.get("low")) or base_price
    close_v = _finite_float(candle.get("close")) or _finite_float(data.get("close")) or spot or ltp
    strike_v = _finite_float(_first_present(data, ("strike", "strike_price", "strikePrice")))
    oi_v = _finite_float(_first_present(data, ("oi", "open_interest", "openInterest"))) or 0.0
    prev_oi = _finite_float(_first_present(data, ("prev_oi", "previous_oi", "oi_prev", "prev_open_interest")))
    if name == "range_pct" and high_v is not None and low_v is not None and close_v:
        return (high_v - low_v) / close_v, "derived"
    if name == "oc_change_pct" and open_v:
        return ((close_v or open_v) - open_v) / open_v, "derived"
    if name == "hl_change_pct" and low_v:
        return ((high_v or low_v) - low_v) / low_v, "derived"
    if name == "oi_change_pct":
        if prev_oi and prev_oi > 0:
            return (oi_v - prev_oi) / prev_oi, "derived"
        return 0.0, "default"
    if name == "iv_change_pct":
        return 0.0, "default"
    if name == "spread_pct":
        spread = _spread_pct(bid, ask, ltp)
        return (spread if spread is not None else _finite_float(data.get("spread_pct")) or 0.0), "default"
    if name == "spot_range_pct" and high_v is not None and low_v is not None and spot:
        return (high_v - low_v) / spot, "derived"
    if name == "body_pct" and open_v and close_v is not None:
        return abs(close_v - open_v) / open_v, "derived"
    if name == "gap_pct":
        return 0.0, "default"
    if name == "upper_wick_pct" and high_v is not None and open_v is not None and close_v is not None and close_v:
        return (high_v - max(open_v, close_v)) / close_v, "derived"
    if name == "lower_wick_pct" and low_v is not None and open_v is not None and close_v is not None and close_v:
        return (min(open_v, close_v) - low_v) / close_v, "derived"
    if name == "close_location_pct" and high_v is not None and low_v is not None and close_v is not None:
        denom = high_v - low_v
        return ((close_v - low_v) / denom) if denom else 0.5, "derived"
    if name in {"close_vs_open_pct", "momentum_lookback_pct"} and open_v:
        return ((close_v or open_v) - open_v) / open_v, "derived"
    if name in {"distance_from_spot"} and strike_v is not None and spot is not None:
        return strike_v - spot, "derived"
    if name in {"distance_from_atm"} and strike_v is not None and spot is not None:
        return abs(strike_v - spot), "derived"
    if name in {"atm_distance"} and strike_v is not None and spot is not None:
        return abs(strike_v - spot), "derived"
    if name in {"strike_distance_pct", "distance_from_atm_pct"} and strike_v is not None and spot:
        return (strike_v - spot) / spot, "derived"
    if name in {"moneyness", "option_moneyness"} and strike_v is not None and spot:
        return strike_v / spot, "derived"
    if name == "log_moneyness" and strike_v is not None and spot and strike_v > 0:
        return math.log(spot / strike_v), "derived"
    if name == "option_to_spot_pct" and ltp is not None and spot:
        return ltp / spot, "derived"
    if name == "ctx_option_price":
        if ltp is not None:
            return ltp, "derived"
        close_fallback = _finite_float(data.get("close")) or _finite_float(candle.get("close"))
        if close_fallback is not None:
            return close_fallback, "derived"
        if intrinsic_for_price is not None:
            raw_extrinsic = _finite_float(data.get("extrinsic_value")) or 0.0
            return intrinsic_for_price + raw_extrinsic, "derived"
        return None, "missing"
    if name == "intrinsic_value" and strike_v is not None and spot is not None:
        side = _normalise_side(data.get("option_type"))
        return max(0.0, spot - strike_v) if side == "CE" else max(0.0, strike_v - spot), "derived"
    if name == "extrinsic_value" and ltp is not None and strike_v is not None and spot is not None:
        side = _normalise_side(data.get("option_type"))
        intrinsic = max(0.0, spot - strike_v) if side == "CE" else max(0.0, strike_v - spot)
        return max(0.0, ltp - intrinsic), "derived"
    if name == "bid_ask_spread":
        if bid is not None and ask is not None:
            return ask - bid, "derived"
        return 0.0, "default"
    if name == "bid_ask_spread_pct":
        spread = _spread_pct(bid, ask, ltp)
        return spread if spread is not None else 0.0, "derived"
    if name == "mid_price":
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0, "derived"
        return ltp, "derived"
    if name == "ltp_vs_mid_diff":
        if ltp is None:
            return None, "missing"
        mid = ((bid + ask) / 2.0) if bid is not None and ask is not None else ltp
        return ltp - mid, "derived"
    if name == "ltp_vs_mid_diff_pct":
        if not ltp:
            return None, "missing"
        mid = ((bid + ask) / 2.0) if bid is not None and ask is not None else ltp
        return (ltp - mid) / ltp, "derived"
    if name in {"oi_ce", "oi_pe", "volume_ce", "volume_pe", "ce_vol_day", "pe_vol_day"}:
        target = "CE" if "_ce" in name or name.startswith("ce_") else "PE"
        rows = _option_chain_rows(data)
        total = 0.0
        for row in rows:
            if _normalise_side(row.get("option_type") or row.get("type")) == target:
                if "volume" in name or "vol" in name:
                    total += _finite_float(row.get("volume")) or 0.0
                else:
                    total += _finite_float(row.get("oi") or row.get("open_interest")) or 0.0
        if not rows and _normalise_side(data.get("option_type")) == target:
            return (_finite_float(data.get("volume")) or 0.0) if ("volume" in name or "vol" in name) else oi_v, "default"
        return total, "default"
    if name in {"ce_pe_oi_ratio", "ce_pe_volume_ratio", "ce_pe_vol_imbalance", "ce_pe_rel_strength"}:
        return 0.0, "default"

    if _is_price_feature(name):
        return base_price, "derived"
    if _safe_default_allowed(name):
        return 0.0, "default"
    return None, "missing"


def build_ensemble_feature_frame(
    snapshot: Mapping[str, Any],
    feature_order: Sequence[str],
    *,
    return_diagnostics: bool = False,
) -> Any:
    import numpy as np
    import pandas as pd

    data, alias_sources = _normalise_live_snapshot(snapshot)
    derived_option_fields = _derive_option_price_fields(data)
    if derived_option_fields:
        for key, value in derived_option_fields.items():
            data[key] = value
    candle = _extract_latest_candle(snapshot)
    values: Dict[str, float] = {}
    missing: List[str] = []
    raw_missing = 0
    alias_count = 0
    default_count = 0
    for feature in [str(f) for f in feature_order]:
        if feature not in snapshot or snapshot.get(feature) in (None, ""):
            raw_missing += 1
        value, source = _feature_value(feature, data, candle)
        if source == "alias" or feature in alias_sources:
            alias_count += 1
        if source == "default":
            default_count += 1
        if value is None:
            missing.append(feature)
            continue
        values[feature] = float(value)
    diagnostics = {
        "missing_raw_count": raw_missing,
        "filled_by_alias_count": alias_count,
        "filled_by_default_count": default_count,
        "truly_missing_count": len(missing),
        "truly_missing_features": missing,
    }
    if missing:
        df = pd.DataFrame([{feature: values.get(feature, np.nan) for feature in feature_order}], columns=list(feature_order))
        df.attrs["diagnostics"] = diagnostics
        return (df, diagnostics) if return_diagnostics else df
    df = pd.DataFrame([{feature: values[feature] for feature in feature_order}], columns=list(feature_order))
    arr = df.to_numpy(dtype=float)
    if arr.shape != (1, len(feature_order)) or not np.all(np.isfinite(arr)):
        diagnostics["truly_missing_count"] = max(1, diagnostics["truly_missing_count"])
        diagnostics.setdefault("truly_missing_features", []).append("non_finite_or_shape_invalid")
    df.attrs["diagnostics"] = diagnostics
    return (df, diagnostics) if return_diagnostics else df


def _normalise_side(value: object) -> str:
    side = str(value or "").strip().upper()
    if side in {"CALL", "C"}:
        return "CE"
    if side in {"PUT", "P"}:
        return "PE"
    return side


def _parse_hhmm(value: object, default: dt_time) -> dt_time:
    try:
        raw = str(value or "").strip()
        if not raw:
            return default
        parts = raw.split(":")
        return dt_time(int(parts[0]), int(parts[1]))
    except Exception:
        return default


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_log(message: str) -> None:
    line = f"[ENSEMBLE-LOAD] {message}"
    print(line, flush=True)
    LOGGER.info(line)


def _decision_log(message: str) -> str:
    line = f"[ENSEMBLE-DECISION] {message}"
    print(line, flush=True)
    LOGGER.info(line)
    return line


def _short_json(value: Any, *, limit: int = 220) -> str:
    try:
        text = json.dumps(value, sort_keys=True)
    except Exception:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class ModelOutput:
    model_key: str
    label: str
    status: str
    loaded: bool = False
    valid: bool = False
    probability_ce: Optional[float] = None
    probability_pe: Optional[float] = None
    confidence: Optional[float] = None
    direction_score: Optional[float] = None
    vote: str = "NO_VOTE"
    weight: float = 0.0
    normalized_weight: float = 0.0
    error: str = ""
    missing_features: List[str] = field(default_factory=list)
    artifact_path: str = ""
    feature_count: int = 0
    predict_method: str = ""
    probability_ce_raw: str = ""
    probability_pe_raw: str = ""
    raw_output: str = ""
    label_mapping: str = ""


@dataclass
class DecisionResult:
    decision: str
    final_direction: float = 0.0
    final_confidence: float = 0.0
    valid_model_count: int = 0
    block_reason: str = ""
    model_outputs: List[ModelOutput] = field(default_factory=list)
    selected_option: Dict[str, Any] = field(default_factory=dict)
    spread_pct: Optional[float] = None
    timestamp: str = field(default_factory=_now_utc)
    skipped_reason_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class PaperPosition:
    side: str
    symbol: str
    strike: Optional[float]
    expiry: str
    entry_price: float
    qty: int
    entry_time: str
    entry_bar: int
    entry_spread_pct: float
    entry_cost: float
    model_confidence: float
    direction_score: float


class ModelArtifact:
    def __init__(self, key: str, spec: Mapping[str, Any], config: Mapping[str, Any]):
        self.key = str(key)
        self.label = str(spec.get("label") or key)
        self.enabled = bool(spec.get("enabled", True))
        self.weight = float(spec.get("weight", 0.0) or 0.0)
        self.side_policy = str(spec.get("side_policy") or "").upper()
        self.config = dict(config)
        self.spec = dict(spec)
        self.artifact_path = _resolve_path(spec.get("artifact_path") or "")
        self.model_file: Optional[Path] = None
        self.model: Any = None
        self.scaler: Any = None
        self.preprocessor: Any = None
        self.calibrator: Any = None
        self.feature_order: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.status = STATUS_ARTIFACT_NOT_FOUND
        self.error = ""
        self.loaded = False
        self.load_events: List[str] = []
        self.searched_paths: List[str] = []

    def _log_load(self, message: str) -> None:
        line = f"[ENSEMBLE-LOAD] {message}"
        self.load_events.append(line)
        print(line, flush=True)
        LOGGER.info(line)

    def load(self) -> ModelOutput:
        if not self.enabled:
            self.status = "DISABLED"
            self.error = "model disabled in config"
            return self._output(valid=False)
        try:
            self.load_events = []
            self.searched_paths = []
            self.model_file = self._find_model_file()
            if self.model_file is None:
                self.status = STATUS_ARTIFACT_NOT_FOUND
                self.error = f"no matching artifact found; searched paths: {', '.join(self.searched_paths) or '-'}"
                self._log_load(f"model={self.label} status={self.status} error={self.error}")
                return self._output(valid=False)
            self._log_load(f"model={self.label} path={self.model_file}")
            self.metadata = self._load_metadata()
            loaded = self._load_pickle_or_joblib(self.model_file)
            self._unwrap_bundle(loaded)
            self.feature_order = self._detect_feature_order()
            self.loaded = self.model is not None
            self.status = STATUS_OK if self.loaded else STATUS_MODEL_LOAD_FAILED
            self._log_load(f"model={self.label} status={'LOADED' if self.loaded else self.status} features={len(self.feature_order)}")
            LOGGER.info(
                "[ENSEMBLE-MODEL] key=%s status=%s model_file=%s features=%s",
                self.key,
                self.status,
                self.model_file,
                len(self.feature_order),
            )
            return self._output(valid=False)
        except Exception as exc:
            self.loaded = False
            self.status = STATUS_MODEL_LOAD_FAILED
            self.error = f"{type(exc).__name__}: {exc}"
            self._log_load(f"model={self.label} status={self.status} error={self.error}")
            LOGGER.exception("[ENSEMBLE-MODEL] key=%s status=%s error=%s", self.key, self.status, self.error)
            return self._output(valid=False)

    def _find_model_file(self) -> Optional[Path]:
        path = self.artifact_path
        self.searched_paths.append(str(path))
        self._log_load(f"searching artifacts path={path}")
        direct = self._find_model_file_in_path(path)
        if direct is not None:
            self._log_load("found files count=1")
            return direct
        self._log_load("found files count=0")
        if "artifact_search_paths" in self.config:
            roots = self.config.get("artifact_search_paths") or []
        else:
            roots = ["artifacts", "artifacts/models", "artifacts/candidates", "artifacts/ml_signals"]
        candidates: List[Path] = []
        for raw_root in roots:
            root = _resolve_path(raw_root)
            self.searched_paths.append(str(root))
            self._log_load(f"searching artifacts path={root}")
            if not root.exists():
                self._log_load("found files count=0")
                continue
            files: List[Path] = []
            for pattern in ("*.pkl", "*.joblib", "*.pickle"):
                files.extend(root.rglob(pattern))
            files = sorted({f.resolve() for f in files if f.is_file()})
            self._log_load(f"found files count={len(files)}")
            for file_path in files:
                if self._matches_model_family(file_path):
                    candidates.append(file_path)
        if not candidates:
            return None
        return sorted(candidates, key=lambda p: (self._candidate_rank(p), len(str(p)), str(p).lower()))[0]

    def _find_model_file_in_path(self, path: Path) -> Optional[Path]:
        if path.is_file() and path.suffix.lower() in {".pkl", ".joblib", ".pickle"}:
            return path
        if not path.exists():
            return None
        for name in ("model.pkl", "model.joblib", "model.pickle", "bundle.pkl", "bundle.joblib"):
            cand = path / name
            if cand.exists():
                return cand
        for pattern in ("*.pkl", "*.joblib", "*.pickle"):
            found = sorted(path.glob(pattern))
            if found:
                return found[0]
        return None

    def _matches_model_family(self, path: Path) -> bool:
        text = str(path).replace("\\", "/").lower()
        key = self.key.lower()
        if key == "elasticnet":
            return "elasticnet" in text or "elastic_net" in text or "elastic-net" in text
        if key == "logistic_regression":
            if any(x in text for x in ("calibrated", "platt", "isotonic", "elasticnet", "elastic_net", "xgboost", "xgb")):
                return False
            return "logistic_regression" in text or "logistic-regression" in text or "logistic" in text or "logreg" in text
        if key == "calibrated_logistic_regression":
            return any(x in text for x in ("calibrated", "calibrated_logistic", "platt", "isotonic"))
        if key == "xgboost":
            return "xgboost" in text or "xgb" in text
        return key in text

    def _candidate_rank(self, path: Path) -> int:
        text = str(path).replace("\\", "/").lower()
        rank = 100
        if "/candidates/" in text:
            rank -= 30
        if path.name.lower() in {"model.pkl", "model.joblib", "model.pickle"}:
            rank -= 20
        if self.artifact_path and str(self.artifact_path).lower() in text:
            rank -= 40
        return rank

    def _load_pickle_or_joblib(self, path: Path) -> Any:
        suffix = path.suffix.lower()
        if suffix == ".joblib":
            try:
                import joblib  # type: ignore

                return joblib.load(path)
            except Exception:
                pass
        with path.open("rb") as fh:
            return pickle.load(fh)

    def _load_metadata(self) -> Dict[str, Any]:
        source = self.model_file or self.artifact_path
        path = source.parent if source.is_file() else source
        merged: Dict[str, Any] = {}
        for name in (
            "candidate_profile.json",
            "candidate_manifest.json",
            "feature_schema.json",
            "training_metadata.json",
            "preprocessing_metadata.json",
            "metadata.json",
            "model_card.json",
        ):
            data = _read_json(path / name)
            if name == "feature_schema.json" and data:
                merged.setdefault("feature_schema", data)
            merged = _deep_merge(merged, data)
        return merged

    def _unwrap_bundle(self, bundle: Any) -> None:
        self.model = bundle
        if isinstance(bundle, dict):
            self.model = (
                bundle.get("model")
                or bundle.get("estimator")
                or bundle.get("classifier")
                or bundle.get("pipeline")
                or bundle
            )
            self.scaler = bundle.get("scaler")
            self.preprocessor = bundle.get("preprocessor") or bundle.get("transformer")
            self.calibrator = bundle.get("calibrator") or bundle.get("calibration_model")
            nested_meta = bundle.get("metadata")
            if isinstance(nested_meta, Mapping):
                self.metadata = _deep_merge(self.metadata, nested_meta)
            for key in (
                "feature_order",
                "required_features",
                "feature_names",
                "features",
                "side",
                "target_name",
                "label_mapping",
                "candidate_side",
                "option_type",
                "positive_class",
                "positive_label",
                "class_mapping",
                "target_mapping",
            ):
                val = bundle.get(key)
                if val not in (None, "", []):
                    self.metadata.setdefault(key, val)
        for attr in ("scaler", "preprocessor", "calibrator"):
            if getattr(self, attr) is None and hasattr(self.model, attr):
                setattr(self, attr, getattr(self.model, attr))

    def _detect_feature_order(self) -> List[str]:
        sources: List[Any] = []
        for key in ("feature_order", "required_features", "feature_names", "features", "live_computable_features"):
            sources.append(self.metadata.get(key))
        fs = self.metadata.get("feature_schema")
        if isinstance(fs, dict):
            sources.extend([fs.get("features"), fs.get("feature_order"), fs.get("required_features")])
        for attr in ("feature_names_in_", "feature_names", "feature_order"):
            sources.append(getattr(self.model, attr, None))
        for src in sources:
            if src is None:
                continue
            try:
                vals = [str(x) for x in list(src) if str(x or "").strip()]
            except Exception:
                vals = []
            if vals:
                return vals
        return []

    def _output(self, *, valid: bool) -> ModelOutput:
        return ModelOutput(
            model_key=self.key,
            label=self.label,
            status=self.status,
            loaded=self.loaded,
            valid=valid,
            weight=self.weight,
            error=self.error,
            artifact_path=str(self.model_file or self.artifact_path),
            feature_count=len(self.feature_order),
            label_mapping=self._label_mapping_summary() if self.loaded else "",
        )

    def score(self, snapshot: Mapping[str, Any]) -> ModelOutput:
        if not self.loaded or self.model is None:
            return self._output(valid=False)
        if not self.feature_order:
            self.status = STATUS_FEATURE_VECTOR_INVALID
            self.error = "feature_order not detected"
            return self._output(valid=False)
        ce_snapshot = _snapshot_for_side(snapshot, "CE")
        pe_snapshot = _snapshot_for_side(snapshot, "PE")
        ce = self._predict_side(ce_snapshot)
        pe = self._predict_side(pe_snapshot)
        if ce.status != STATUS_OK:
            return ce
        if pe.status != STATUS_OK:
            return pe
        if ce.probability_ce is None or pe.probability_pe is None:
            out = self._output(valid=False)
            out.status = STATUS_PREDICT_FAILED
            out.error = "prediction returned no mapped CE/PE probability"
            out.probability_ce_raw = ce.probability_ce_raw
            out.probability_pe_raw = pe.probability_pe_raw
            out.raw_output = " | ".join(x for x in (ce.raw_output, pe.raw_output) if x)
            out.label_mapping = ce.label_mapping or pe.label_mapping
            return out
        prob_ce = float(ce.probability_ce)
        prob_pe = float(pe.probability_pe)
        direction = prob_ce - prob_pe
        confidence = max(prob_ce, prob_pe)
        thresholds = self.config.get("thresholds") or {}
        edge = float(thresholds.get("ensemble_min_direction_edge", 0.20) or 0.20)
        vote = DECISION_BUY_CE if direction > edge else DECISION_BUY_PE if direction < -edge else "NO_VOTE"
        raw_output = " | ".join(x for x in (ce.raw_output, pe.raw_output) if x)
        label_mapping = ce.label_mapping or pe.label_mapping
        print(
            "[ENSEMBLE-SCORE] "
            f"model={self.label} ce_prob={prob_ce:.6f} pe_prob={prob_pe:.6f} "
            f"direction={direction:.6f} confidence={confidence:.6f}",
            flush=True,
        )
        return ModelOutput(
            model_key=self.key,
            label=self.label,
            status=STATUS_OK,
            loaded=True,
            valid=True,
            probability_ce=prob_ce,
            probability_pe=prob_pe,
            confidence=float(max(0.0, min(1.0, confidence))),
            direction_score=float(max(-1.0, min(1.0, direction))),
            vote=vote,
            weight=self.weight,
            artifact_path=str(self.model_file or self.artifact_path),
            feature_count=len(self.feature_order),
            predict_method=ce.predict_method or pe.predict_method,
            probability_ce_raw=ce.probability_ce_raw,
            probability_pe_raw=pe.probability_pe_raw,
            raw_output=raw_output,
            label_mapping=label_mapping,
        )

    def _predict_side(self, snapshot: Mapping[str, Any]) -> ModelOutput:
        try:
            frame, diagnostics = build_ensemble_feature_frame(snapshot, self.feature_order, return_diagnostics=True)
        except Exception as exc:
            out = self._output(valid=False)
            out.status = STATUS_FEATURE_VECTOR_INVALID
            out.error = f"feature mapping failed: {type(exc).__name__}: {exc}"
            return out
        missing = list(diagnostics.get("truly_missing_features") or [])
        if diagnostics.get("truly_missing_count", 0):
            out = self._output(valid=False)
            out.status = STATUS_FEATURES_MISSING
            preview = ", ".join(missing[:20])
            out.error = (
                f"missing_raw_count={diagnostics.get('missing_raw_count', 0)} "
                f"filled_by_alias_count={diagnostics.get('filled_by_alias_count', 0)} "
                f"filled_by_default_count={diagnostics.get('filled_by_default_count', 0)} "
                f"truly_missing_count={diagnostics.get('truly_missing_count', 0)} "
                f"missing={preview}"
            )
            out.missing_features = missing
            return out
        try:
            import numpy as np

            X: Any = frame.to_numpy(dtype=float)
            if X.shape != (1, len(self.feature_order)) or not np.all(np.isfinite(X)):
                out = self._output(valid=False)
                out.status = STATUS_FEATURE_VECTOR_INVALID
                out.error = (
                    "feature vector invalid "
                    f"shape={getattr(X, 'shape', None)} expected={(1, len(self.feature_order))}"
                )
                return out
            for transformer in (self.preprocessor, self.scaler):
                if transformer is not None and hasattr(transformer, "transform"):
                    X = transformer.transform(X)
            side = _normalise_side(snapshot.get("option_type"))
            prob, method, raw_output, label_mapping = self._predict_probability(X)
            if prob is None:
                out = self._output(valid=False)
                out.status = STATUS_UNKNOWN_LABEL_MAPPING
                out.error = "UNKNOWN_LABEL_MAPPING: binary predict_proba output has no artifact label metadata"
                out.predict_method = method
                out.raw_output = raw_output
                out.label_mapping = label_mapping
                return out
            print(
                "[ENSEMBLE-SCORE] "
                f"model={self.label} side={side or '-'} raw={raw_output} prob={prob:.6f}",
                flush=True,
            )
            out = self._output(valid=True)
            out.status = STATUS_OK
            out.valid = True
            out.predict_method = method
            out.raw_output = f"{side or '-'}={raw_output}"
            out.label_mapping = label_mapping
            if side == "PE":
                out.probability_pe = prob
                out.probability_pe_raw = raw_output
            else:
                out.probability_ce = prob
                out.probability_ce_raw = raw_output
            return out
        except Exception as exc:
            out = self._output(valid=False)
            out.status = STATUS_PREDICT_FAILED
            out.error = f"{type(exc).__name__}: {exc}"
            LOGGER.info("[ENSEMBLE-MODEL] key=%s status=%s error=%s", self.key, out.status, out.error)
            return out

    def _metadata_value(self, *keys: str) -> Any:
        containers: List[Any] = [self.metadata, self.spec]
        for name in ("metadata", "candidate_metadata", "training_metadata"):
            value = self.metadata.get(name)
            if isinstance(value, Mapping):
                containers.append(value)
        for container in containers:
            if not isinstance(container, Mapping):
                continue
            for key in keys:
                if key in container and container.get(key) not in (None, "", []):
                    return container.get(key)
        return None

    def _label_mapping_summary(self) -> str:
        parts: List[str] = []
        for key in (
            "label_mapping",
            "class_mapping",
            "target_mapping",
            "positive_class",
            "positive_label",
            "target_name",
            "side",
            "candidate_side",
            "option_type",
            "side_policy",
        ):
            value = self._metadata_value(key)
            if value not in (None, "", []):
                try:
                    rendered = json.dumps(value, sort_keys=True)
                except Exception:
                    rendered = str(value)
                parts.append(f"{key}={rendered}")
        return "; ".join(parts) if parts else "UNKNOWN_LABEL_MAPPING"

    def _has_binary_label_metadata(self) -> bool:
        return bool(
            self._metadata_value(
                "label_mapping",
                "class_mapping",
                "target_mapping",
                "positive_class",
                "positive_label",
                "target_name",
                "side",
                "candidate_side",
                "option_type",
                "side_policy",
            )
            not in (None, "", [])
        )

    @staticmethod
    def _same_label(a: Any, b: Any) -> bool:
        if a == b:
            return True
        return str(a).strip().lower() == str(b).strip().lower()

    @staticmethod
    def _favorable_label_text(value: Any) -> bool:
        text = str(value).strip().lower()
        if text in {"1", "true", "yes"}:
            return True
        positives = ("favorable", "positive", "profit", "win", "survivor", "buy", "trade", "entry")
        negatives = ("no_trade", "not_", "false", "negative", "loss", "reject", "skip")
        return any(token in text for token in positives) and not any(token in text for token in negatives)

    def _positive_class_index(self, estimator: Any) -> Tuple[Optional[int], str]:
        classes = getattr(estimator, "classes_", None)
        if classes is None:
            return 0, self._label_mapping_summary() or "no classes_; single probability output"
        try:
            vals = list(classes)
        except Exception:
            return None, "UNKNOWN_LABEL_MAPPING: classes_ unreadable"

        summary = self._label_mapping_summary()
        positive = self._metadata_value("positive_class", "positive_label", "favorable_class", "favorable_label")
        if positive is not None:
            for idx, cls in enumerate(vals):
                if self._same_label(cls, positive):
                    return idx, summary

        for mapping_key in ("label_mapping", "class_mapping", "target_mapping"):
            mapping = self._metadata_value(mapping_key)
            if not isinstance(mapping, Mapping):
                continue
            for key, value in mapping.items():
                if self._favorable_label_text(value):
                    for idx, cls in enumerate(vals):
                        if self._same_label(cls, key):
                            return idx, summary
                if self._favorable_label_text(key):
                    for idx, cls in enumerate(vals):
                        if self._same_label(cls, value):
                            return idx, summary

        if not self._has_binary_label_metadata():
            return None, "UNKNOWN_LABEL_MAPPING"
        for candidate in (1, True, "1", "true", "positive"):
            for idx, cls in enumerate(vals):
                if self._same_label(cls, candidate):
                    return idx, summary
        if len(vals) == 1:
            return 0, summary
        return None, f"UNKNOWN_LABEL_MAPPING: classes={vals} metadata={summary}"

    def _predict_probability(self, X: Any) -> Tuple[Optional[float], str, str, str]:
        import numpy as np

        estimator = self.calibrator if self.calibrator is not None else self.model
        if hasattr(estimator, "predict_proba"):
            proba = np.asarray(estimator.predict_proba(X), dtype=float)
            raw_output = _short_json(proba.tolist())
            idx, label_mapping = self._positive_class_index(estimator)
            if idx is None:
                print(
                    "[ENSEMBLE-SCORE] "
                    f"model={self.label} raw={raw_output} prob=UNKNOWN_LABEL_MAPPING",
                    flush=True,
                )
                return None, "predict_proba", raw_output, label_mapping
            if proba.ndim == 1:
                prob = float(proba[idx])
            else:
                if idx >= proba.shape[1]:
                    return None, "predict_proba", raw_output, f"UNKNOWN_LABEL_MAPPING: class_index={idx} raw_shape={proba.shape}"
                prob = float(proba[0][idx])
            return float(max(0.0, min(1.0, prob))), "predict_proba", raw_output, label_mapping
        if hasattr(estimator, "decision_function"):
            raw = float(np.asarray(estimator.decision_function(X), dtype=float).ravel()[0])
            prob = _sigmoid(raw)
            return prob, "decision_function_sigmoid", _short_json(raw), self._label_mapping_summary()
        if hasattr(estimator, "predict"):
            raw = float(np.asarray(estimator.predict(X), dtype=float).ravel()[0])
            prob = raw if 0.0 <= raw <= 1.0 else _sigmoid(raw)
            return float(max(0.0, min(1.0, prob))), "predict", _short_json(raw), self._label_mapping_summary()
        raise RuntimeError("no_supported_predict_method")


def _snapshot_for_side(snapshot: Mapping[str, Any], side: str) -> Dict[str, Any]:
    out = dict(snapshot)
    selected = _select_atm_option_for_side(snapshot, side)
    if selected:
        out.update(selected)
    side = _normalise_side(side)
    out["option_type"] = side
    out["option_type_ce"] = 1.0 if side == "CE" else 0.0
    out["option_type_pe"] = 1.0 if side == "PE" else 0.0
    if selected:
        _log_feature_row(side, out)
    return out


def _option_chain_rows(snapshot: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in ("option_chain", "chain", "options", "contracts"):
        value = snapshot.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, dict)):
            rows.extend([dict(r) for r in value if isinstance(r, Mapping)])
    return rows


def _select_atm_option_for_side(snapshot: Mapping[str, Any], side: str) -> Dict[str, Any]:
    rows = _option_chain_rows(snapshot)
    if not rows:
        return {}
    side = _normalise_side(side)
    spot = _finite_float(_first_present(snapshot, ("spot", "spot_close", "ctx_spot", "underlying_price", "underlyingValue")))
    if spot is None:
        spot = _finite_float(snapshot.get("close")) or _finite_float(snapshot.get("ltp"))
    expiries = [(_parse_date_like(r.get("expiry") or r.get("expiry_date")), r.get("expiry") or r.get("expiry_date")) for r in rows]
    valid_expiries = sorted({item for parsed, item in expiries if parsed is not None}, key=lambda x: _parse_date_like(x) or date.max)
    current_expiry = valid_expiries[0] if valid_expiries else None
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        if _normalise_side(row.get("option_type") or row.get("type") or row.get("right")) != side:
            continue
        if current_expiry is not None and (row.get("expiry") or row.get("expiry_date")) != current_expiry:
            continue
        ltp = _positive_float_from(
            row,
            ("ltp", "last_price", "lastPrice", "option_ltp", "close", "option_price", "traded_price", "last_traded_price", "ctx_option_price"),
        )
        bid = _positive_float_from(row, ("bid", "best_bid", "best_bid_price", "bid_price", "bidPrice"))
        ask = _positive_float_from(row, ("ask", "best_ask", "best_ask_price", "ask_price", "askPrice", "offer"))
        if ltp is None and bid is not None and ask is not None:
            ltp = (bid + ask) / 2.0
        if ltp is None or ltp <= 0:
            continue
        spread = _spread_pct(bid if bid is not None else ltp, ask if ask is not None else ltp, ltp)
        if spread is not None and spread < 0:
            continue
        row = dict(row)
        row["option_type"] = side
        row.setdefault("ltp", ltp)
        row.setdefault("bid", bid if bid is not None else ltp)
        row.setdefault("ask", ask if ask is not None else ltp)
        candidates.append(row)
    if not candidates:
        return {}
    if spot is not None:
        candidates.sort(key=lambda r: abs(float(_finite_float(r.get("strike") or r.get("strike_price")) or spot) - spot))
    return candidates[0]


class EnsembleAutoRouter:
    def __init__(self, config: Optional[Mapping[str, Any]] = None, *, config_path: str | Path = DEFAULT_CONFIG_PATH):
        self.config_path = Path(config_path)
        self.config = normalize_config(dict(config or load_config(self.config_path)))
        self.models: Dict[str, ModelArtifact] = {}
        self.last_decision: Optional[DecisionResult] = None
        self.skipped_reason_counts: Dict[str, int] = {}
        self.load_events: List[str] = []
        self.scanned_artifacts: List[Path] = []
        self.last_decision_log = ""
        self.load_models()

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    def set_runtime_enabled(self, enabled: bool = True) -> None:
        self.config["enabled"] = bool(enabled)

    def reload(self) -> None:
        self.config = load_config(self.config_path)
        self.load_models()

    def _log_load(self, message: str) -> None:
        line = f"[ENSEMBLE-LOAD] {message}"
        self.load_events.append(line)
        print(line, flush=True)
        LOGGER.info(line)

    def scan_artifacts(self) -> List[Path]:
        roots = self.config.get("artifact_search_paths") or [
            "artifacts",
            "artifacts/models",
            "artifacts/candidates",
            "artifacts/ml_signals",
        ]
        files: List[Path] = []
        artifact_root = _resolve_path(self.config.get("artifact_root") or "artifacts")
        self._log_load(f"artifacts_root={artifact_root}")
        for raw_root in roots:
            root = _resolve_path(raw_root)
            self._log_load(f"searching artifacts path={root}")
            if not root.exists():
                self._log_load("found files count=0")
                continue
            found: List[Path] = []
            for pattern in ("*.pkl", "*.pickle", "*.joblib"):
                found.extend(root.rglob(pattern))
            found = sorted({p.resolve() for p in found if p.is_file()})
            self._log_load(f"found files count={len(found)}")
            files.extend(found)
        self.scanned_artifacts = sorted({p.resolve() for p in files})
        self._log_load(f"files_found={len(self.scanned_artifacts)}")
        return list(self.scanned_artifacts)

    def load_models(self) -> List[ModelOutput]:
        outputs: List[ModelOutput] = []
        self.load_events = []
        self.scan_artifacts()
        self.models = {}
        specs = self.config.get("models") or {}
        ordered_keys = [k for k in EXPECTED_MODEL_ORDER if k in specs]
        ordered_keys.extend([k for k in specs.keys() if k not in ordered_keys])
        for key in ordered_keys:
            spec = specs.get(key)
            if not isinstance(spec, Mapping):
                continue
            artifact = ModelArtifact(str(key), spec, self.config)
            self.models[str(key)] = artifact
            output = artifact.load()
            self.load_events.extend(artifact.load_events)
            self._log_load(f"{artifact.label} => {output.artifact_path}/{output.status}/{output.error or '-'}")
            outputs.append(output)
        return outputs

    def model_status(self) -> List[ModelOutput]:
        rows = []
        specs = self.config.get("models") or {}
        for key in EXPECTED_MODEL_ORDER:
            model = self.models.get(key)
            if model is not None:
                rows.append(model._output(valid=False))
                continue
            spec = specs.get(key) if isinstance(specs, Mapping) else None
            if not isinstance(spec, Mapping):
                spec = DEFAULT_CONFIG["models"].get(key, {})
            row = ModelOutput(
                model_key=key,
                label=str(spec.get("label") or key),
                status=STATUS_ARTIFACT_NOT_FOUND,
                loaded=False,
                valid=False,
                weight=float(spec.get("weight", 0.0) or 0.0),
                error="model spec missing or not loaded",
                artifact_path=str(_resolve_path(spec.get("artifact_path") or "")),
            )
            rows.append(row)
        for key, model in self.models.items():
            if key not in EXPECTED_MODEL_ORDER:
                rows.append(model._output(valid=False))
        return rows

    def model_vote_rows(self, result: Optional[DecisionResult] = None) -> List[Dict[str, Any]]:
        by_key: Dict[str, ModelOutput] = {}
        if result is not None:
            by_key.update({row.model_key: row for row in result.model_outputs})
        for row in self.model_status():
            by_key.setdefault(row.model_key, row)
        rows: List[Dict[str, Any]] = []
        for key in EXPECTED_MODEL_ORDER:
            row = by_key.get(key)
            if row is None:
                continue
            error = row.error or ""
            if row.missing_features:
                preview = ", ".join(row.missing_features[:20])
                error = f"missing {len(row.missing_features)} feature(s): {preview}"
            rows.append(
                {
                    "model_key": row.model_key,
                    "model": row.label,
                    "loaded": bool(row.loaded),
                    "status": row.status,
                    "confidence": None if row.confidence is None else float(row.confidence),
                    "ce_prob": None if row.probability_ce is None else float(row.probability_ce),
                    "pe_prob": None if row.probability_pe is None else float(row.probability_pe),
                    "raw_output": row.raw_output or " | ".join(
                        x for x in (row.probability_ce_raw, row.probability_pe_raw) if x
                    ),
                    "label_mapping": row.label_mapping,
                    "direction": None if row.direction_score is None else float(row.direction_score),
                    "vote": row.vote or "NO_VOTE",
                    "weight": float(row.normalized_weight or row.weight or 0.0),
                    "error": error,
                    "artifact_path": row.artifact_path,
                }
            )
        return rows

    def debug_lines(self, result: Optional[DecisionResult] = None) -> List[str]:
        lines = list(self.load_events[-80:])
        active_result = result or self.last_decision
        for row in self.model_vote_rows(active_result):
            lines.append(
                "[ENSEMBLE-LOAD] "
                f"{row['model']} => {row.get('artifact_path') or '-'}/{row['status']}/{row.get('error') or '-'}"
            )
        if active_result is not None:
            thresholds = self.config.get("thresholds") or {}
            min_valid = int(thresholds.get("ensemble_min_valid_models", 2) or 2)
            lines.append(
                "[ENSEMBLE-DECISION] "
                f"valid_models={active_result.valid_model_count} min_required={min_valid} "
                f"block={active_result.block_reason or '-'}"
            )
        elif self.last_decision_log:
            lines.append(self.last_decision_log)
        return lines

    def decide(
        self,
        snapshot: Mapping[str, Any],
        *,
        open_positions: Optional[Sequence[Mapping[str, Any]]] = None,
        paper_mode: bool = False,
    ) -> DecisionResult:
        if not self.enabled:
            return self._blocked("ROUTER_DISABLED", self.model_status())
        if not isinstance(snapshot, Mapping) or not snapshot:
            return self._blocked("MARKET_SNAPSHOT_INVALID", self.model_status())
        model_outputs = [model.score(snapshot) for model in self.models.values() if model.enabled]
        valid = [m for m in model_outputs if m.valid and m.direction_score is not None and m.confidence is not None]
        thresholds = self.config.get("thresholds") or {}
        min_valid = int(thresholds.get("ensemble_min_valid_models", 2) or 2)
        self.last_decision_log = _decision_log(f"valid_models={len(valid)} min_required={min_valid}")
        if len(valid) < min_valid:
            return self._blocked("FEWER_THAN_MIN_VALID_MODELS", model_outputs)
        weight_sum = sum(max(0.0, float(m.weight or 0.0)) for m in valid)
        if weight_sum <= 0:
            return self._blocked("NO_POSITIVE_MODEL_WEIGHT", model_outputs)
        final_direction = 0.0
        final_confidence = 0.0
        for m in valid:
            m.normalized_weight = max(0.0, float(m.weight or 0.0)) / weight_sum
            final_direction += float(m.direction_score or 0.0) * m.normalized_weight
            final_confidence += float(m.confidence or 0.0) * m.normalized_weight
        disagreements = _pairwise_disagreement(valid)
        max_disagreement = float(thresholds.get("ensemble_max_model_disagreement", 0.35) or 0.35)
        if disagreements > max_disagreement:
            return self._blocked("MODELS_DISAGREE_STRONGLY", model_outputs, final_direction, final_confidence)
        min_edge = float(thresholds.get("ensemble_min_direction_edge", 0.20) or 0.20)
        min_conf = float(thresholds.get("ensemble_min_confidence", 0.65) or 0.65)
        if paper_mode and bool(self.config.get("debug_allow_low_edge_paper_trade", False)):
            min_edge = 0.03
            min_conf = 0.50
            _decision_log("debug_allow_low_edge_paper_trade=true paper_mode=true min_confidence=0.50 min_direction_edge=0.03")
        if abs(final_direction) <= min_edge:
            return self._blocked("DIRECTION_EDGE_TOO_LOW", model_outputs, final_direction, final_confidence)
        if final_confidence < min_conf:
            return self._blocked("CONFIDENCE_TOO_LOW", model_outputs, final_direction, final_confidence)
        side = "CE" if final_direction > 0 else "PE"
        duplicate_reason = self._duplicate_block_reason(side, open_positions)
        if duplicate_reason:
            return self._blocked(duplicate_reason, model_outputs, final_direction, final_confidence)
        option, spread_pct, liquidity_reason = self._select_option(snapshot, side)
        if liquidity_reason:
            return self._blocked(liquidity_reason, model_outputs, final_direction, final_confidence, option, spread_pct)
        decision = DECISION_BUY_CE if side == "CE" else DECISION_BUY_PE
        result = DecisionResult(
            decision=decision,
            final_direction=float(final_direction),
            final_confidence=float(final_confidence),
            valid_model_count=len(valid),
            block_reason="",
            model_outputs=model_outputs,
            selected_option=option,
            spread_pct=spread_pct,
            skipped_reason_counts=dict(self.skipped_reason_counts),
        )
        self.last_decision = result
        LOGGER.info(
            "[ENSEMBLE-DECISION] decision=%s direction=%.4f confidence=%.4f valid_models=%s symbol=%s",
            decision,
            final_direction,
            final_confidence,
            len(valid),
            option.get("symbol", ""),
        )
        return result

    def _blocked(
        self,
        reason: str,
        model_outputs: Sequence[ModelOutput],
        final_direction: float = 0.0,
        final_confidence: float = 0.0,
        selected_option: Optional[Dict[str, Any]] = None,
        spread_pct: Optional[float] = None,
    ) -> DecisionResult:
        self.skipped_reason_counts[reason] = self.skipped_reason_counts.get(reason, 0) + 1
        self.last_decision_log = _decision_log(f"block_reason={reason}")
        result = DecisionResult(
            decision=DECISION_NO_TRADE,
            final_direction=float(final_direction),
            final_confidence=float(final_confidence),
            valid_model_count=sum(1 for m in model_outputs if m.valid),
            block_reason=reason,
            model_outputs=list(model_outputs),
            selected_option=selected_option or {},
            spread_pct=spread_pct,
            skipped_reason_counts=dict(self.skipped_reason_counts),
        )
        self.last_decision = result
        LOGGER.info(
            "[ENSEMBLE-DECISION] decision=NO_TRADE reason=%s direction=%.4f confidence=%.4f valid_models=%s",
            reason,
            final_direction,
            final_confidence,
            result.valid_model_count,
        )
        return result

    def _duplicate_block_reason(self, side: str, open_positions: Optional[Sequence[Mapping[str, Any]]]) -> str:
        rules = self.config.get("duplicate_trade_rules") or {}
        if not open_positions:
            return ""
        if bool(rules.get("one_position_per_side", True)):
            for pos in open_positions:
                pos_side = _normalise_side(pos.get("side") or pos.get("option_type") or pos.get("decision"))
                if pos_side == side and str(pos.get("status") or "OPEN").upper() in {"", "OPEN"}:
                    return "DUPLICATE_POSITION_OPEN"
        return ""

    def _select_option(self, snapshot: Mapping[str, Any], side: str) -> Tuple[Dict[str, Any], Optional[float], str]:
        rows = []
        for key in ("option_chain", "chain", "options", "contracts"):
            value = snapshot.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, dict)):
                rows.extend([r for r in value if isinstance(r, Mapping)])
        if not rows:
            rows = [snapshot]
        side = _normalise_side(side)
        candidates = [dict(r) for r in rows if _normalise_side(r.get("option_type") or r.get("side")) in {"", side}]
        if not candidates:
            return {}, None, "NO_OPTION_FOR_SIDE"
        candidates.sort(key=lambda r: abs(float(_finite_float(r.get("atm_distance") or r.get("distance_from_spot") or 0.0) or 0.0)))
        rules = self.config.get("liquidity") or {}
        max_spread = float(rules.get("max_spread_pct", 0.12) or 0.12)
        min_volume = float(rules.get("min_volume", 0) or 0)
        min_oi = float(rules.get("min_oi", 0) or 0)
        min_ltp = float(rules.get("min_ltp", 0.05) or 0.05)
        for row in candidates:
            bid = _finite_float(row.get("bid") or row.get("best_bid") or row.get("bid_price"))
            ask = _finite_float(row.get("ask") or row.get("best_ask") or row.get("ask_price"))
            ltp = _finite_float(row.get("ltp") or row.get("last_price") or row.get("close") or row.get("option_price"))
            if ltp is None and bid is not None and ask is not None:
                ltp = (bid + ask) / 2.0
            if ltp is None or ltp < min_ltp:
                continue
            spread_pct = _spread_pct(bid, ask, ltp)
            if spread_pct is None:
                spread_pct = float(_finite_float(row.get("spread_pct")) or 0.0)
            if spread_pct > max_spread:
                return row, spread_pct, "SPREAD_TOO_WIDE"
            if float(_finite_float(row.get("volume")) or 0.0) < min_volume:
                return row, spread_pct, "VOLUME_TOO_LOW"
            if float(_finite_float(row.get("oi") or row.get("open_interest")) or 0.0) < min_oi:
                return row, spread_pct, "OI_TOO_LOW"
            row["option_type"] = side
            row["ltp"] = ltp
            row["bid"] = bid if bid is not None else ltp
            row["ask"] = ask if ask is not None else ltp
            return row, spread_pct, ""
        return {}, None, "LIQUIDITY_FILTER_FAILED"


def _spread_pct(bid: Optional[float], ask: Optional[float], ltp: Optional[float]) -> Optional[float]:
    if bid is None or ask is None or ltp is None or ltp <= 0 or ask < bid:
        return None
    return float((ask - bid) / max(ltp, 1e-9))


def _pairwise_disagreement(outputs: Sequence[ModelOutput]) -> float:
    vals = [float(m.direction_score or 0.0) for m in outputs if m.direction_score is not None]
    if len(vals) < 2:
        return 0.0
    return max(abs(a - b) for i, a in enumerate(vals) for b in vals[i + 1 :])


class EnsemblePaperTrader:
    def __init__(self, router: EnsembleAutoRouter):
        self.router = router
        self.config = router.config
        self.open_position: Optional[PaperPosition] = None
        self.trades: List[Dict[str, Any]] = []
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.paper_running = False
        self.bar_index = 0
        self.trades_today = 0

    def start(self) -> None:
        self.paper_running = True
        LOGGER.info("[ENSEMBLE-PAPER] action=start")

    def stop(self) -> None:
        self.paper_running = False
        LOGGER.info("[ENSEMBLE-PAPER] action=stop")

    def on_snapshot(self, snapshot: Mapping[str, Any]) -> DecisionResult:
        self.bar_index += 1
        exited_this_bar = self._update_open_position(snapshot)
        if not self.paper_running:
            return self.router._blocked("PAPER_MODE_STOPPED", [])
        if exited_this_bar:
            return self.router._blocked("EXITED_THIS_BAR", [])
        open_positions = [asdict(self.open_position)] if self.open_position else []
        decision = self.router.decide(snapshot, open_positions=open_positions, paper_mode=True)
        risk_reason = self._new_entry_risk_reason()
        if decision.decision in {DECISION_BUY_CE, DECISION_BUY_PE} and risk_reason:
            return self.router._blocked(
                risk_reason,
                decision.model_outputs,
                decision.final_direction,
                decision.final_confidence,
                decision.selected_option,
                decision.spread_pct,
            )
        if decision.decision in {DECISION_BUY_CE, DECISION_BUY_PE}:
            self._open_from_decision(decision)
        return decision

    def _risk_blocks_new_entry(self) -> bool:
        return bool(self._new_entry_risk_reason())

    def _new_entry_risk_reason(self) -> str:
        rules = self.config.get("duplicate_trade_rules") or {}
        exits = self.config.get("exits") or {}
        if self.trades_today >= int(rules.get("max_trades_per_day", 3) or 3):
            return "MAX_TRADES_DAY_REACHED"
        if self.realized_pnl <= float(exits.get("daily_loss_limit", -2500.0) or -2500.0):
            return "DAILY_LOSS_LIMIT_REACHED"
        if self.open_position is not None:
            return "DUPLICATE_POSITION_OPEN"
        now_t = datetime.now().time()
        cutoff = _parse_hhmm(exits.get("entry_cutoff_time", "15:00"), dt_time(15, 0))
        return "ENTRY_CUTOFF_TIME" if now_t >= cutoff else ""

    def _open_from_decision(self, decision: DecisionResult) -> None:
        if self.open_position is not None:
            return
        opt = decision.selected_option or {}
        side = "CE" if decision.decision == DECISION_BUY_CE else "PE"
        paper_cfg = self.config.get("paper_trading") or {}
        qty = int(paper_cfg.get("lot_size", 65) if paper_cfg.get("lot_size", None) is not None else 65)
        ltp = float(_finite_float(opt.get("ask") or opt.get("ltp")) or 0.0)
        slippage_pct = float(paper_cfg.get("slippage_pct", 0.001) if paper_cfg.get("slippage_pct", None) is not None else 0.001)
        entry_price = ltp * (1.0 + slippage_pct)
        cost = self._roundtrip_cost(entry_price, entry_price, qty) / 2.0
        self.open_position = PaperPosition(
            side=side,
            symbol=str(opt.get("symbol") or opt.get("trading_symbol") or f"NIFTY-{side}"),
            strike=_finite_float(opt.get("strike") or opt.get("strike_price")),
            expiry=str(opt.get("expiry") or opt.get("expiry_date") or ""),
            entry_price=entry_price,
            qty=qty,
            entry_time=_now_utc(),
            entry_bar=self.bar_index,
            entry_spread_pct=float(decision.spread_pct or 0.0),
            entry_cost=cost,
            model_confidence=float(decision.final_confidence),
            direction_score=float(decision.final_direction),
        )
        self.trades_today += 1
        LOGGER.info("[ENSEMBLE-PAPER] action=open side=%s symbol=%s price=%.4f qty=%s", side, self.open_position.symbol, entry_price, qty)

    def _update_open_position(self, snapshot: Mapping[str, Any]) -> bool:
        if self.open_position is None:
            self.unrealized_pnl = 0.0
            return False
        ltp = self._price_for_position(snapshot)
        if ltp is None:
            return False
        pos = self.open_position
        gross = (ltp - pos.entry_price) * pos.qty
        cost = self._roundtrip_cost(pos.entry_price, ltp, pos.qty)
        self.unrealized_pnl = gross - cost
        exit_reason = self._exit_reason(ltp)
        if exit_reason:
            self._close_position(ltp, exit_reason)
            return True
        return False

    def _price_for_position(self, snapshot: Mapping[str, Any]) -> Optional[float]:
        if self.open_position is None:
            return None
        side = self.open_position.side
        for key in ("option_chain", "chain", "options", "contracts"):
            rows = snapshot.get(key)
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, dict)):
                for row in rows:
                    if isinstance(row, Mapping) and _normalise_side(row.get("option_type")) == side:
                        if str(row.get("symbol") or row.get("trading_symbol") or "") == self.open_position.symbol:
                            return _finite_float(row.get("bid") or row.get("ltp") or row.get("close"))
        return _finite_float(snapshot.get("bid") or snapshot.get("ltp") or snapshot.get("close") or snapshot.get("option_price"))

    def _exit_reason(self, current_price: float) -> str:
        pos = self.open_position
        if pos is None or pos.entry_price <= 0:
            return ""
        exits = self.config.get("exits") or {}
        ret = (current_price - pos.entry_price) / pos.entry_price
        if ret <= -float(exits.get("stop_loss_pct", 0.10) or 0.10):
            return "STOP_LOSS"
        if ret >= float(exits.get("target_pct", 0.20) or 0.20):
            return "TARGET"
        if self.bar_index - pos.entry_bar >= int(exits.get("max_hold_bars", 5) or 5):
            return "MAX_HOLD_BARS"
        force_t = _parse_hhmm(exits.get("force_exit_time", "15:20"), dt_time(15, 20))
        if datetime.now().time() >= force_t:
            return "FORCE_EXIT_TIME"
        return ""

    def _close_position(self, exit_price: float, reason: str) -> None:
        pos = self.open_position
        if pos is None:
            return
        cost = self._roundtrip_cost(pos.entry_price, exit_price, pos.qty)
        gross = (exit_price - pos.entry_price) * pos.qty
        pnl = gross - cost
        trade = {
            "entry_time": pos.entry_time,
            "exit_time": _now_utc(),
            "side": pos.side,
            "symbol": pos.symbol,
            "strike": pos.strike,
            "expiry": pos.expiry,
            "qty": pos.qty,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "gross_pnl": gross,
            "costs": cost,
            "net_pnl": pnl,
            "exit_reason": reason,
            "confidence": pos.model_confidence,
            "direction_score": pos.direction_score,
        }
        self.trades.append(trade)
        self.realized_pnl += pnl
        self.unrealized_pnl = 0.0
        self.open_position = None
        LOGGER.info("[ENSEMBLE-PAPER] action=close side=%s symbol=%s pnl=%.2f reason=%s", pos.side, pos.symbol, pnl, reason)

    def _roundtrip_cost(self, buy_price: float, sell_price: float, qty: int) -> float:
        paper_cfg = self.config.get("paper_trading") or {}
        brokerage = float(paper_cfg.get("brokerage_per_order", 5.0) if paper_cfg.get("brokerage_per_order", None) is not None else 5.0) * 2.0
        cost_bps = float(paper_cfg.get("cost_bps", 1.5) if paper_cfg.get("cost_bps", None) is not None else 1.5) / 10000.0
        turnover = max(0.0, (float(buy_price) + float(sell_price)) * int(qty))
        return brokerage + turnover * cost_bps

    def export(self, *, trades_path: str | Path | None = None, summary_path: str | Path | None = None) -> Tuple[Path, Path]:
        paper_cfg = self.config.get("paper_trading") or {}
        tpath = _resolve_path(trades_path or paper_cfg.get("trade_log_path") or "reports/ensemble_auto_router_trades.csv")
        spath = _resolve_path(summary_path or paper_cfg.get("summary_path") or "reports/ensemble_auto_router_summary.json")
        tpath.parent.mkdir(parents=True, exist_ok=True)
        spath.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "entry_time", "exit_time", "side", "symbol", "strike", "expiry", "qty",
            "entry_price", "exit_price", "gross_pnl", "costs", "net_pnl", "exit_reason",
            "confidence", "direction_score",
        ]
        with tpath.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for trade in self.trades:
                writer.writerow({k: trade.get(k, "") for k in fieldnames})
        summary = summarize_trades(self.trades, self.router.skipped_reason_counts)
        summary.update(
            {
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": self.unrealized_pnl,
                "open_position": asdict(self.open_position) if self.open_position else None,
                "exported_at": _now_utc(),
            }
        )
        spath.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        LOGGER.info("[ENSEMBLE-PAPER] action=export trades=%s summary=%s", tpath, spath)
        return tpath, spath


def summarize_trades(trades: Sequence[Mapping[str, Any]], skipped_reason_counts: Optional[Mapping[str, int]] = None, model_agreements: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    pnls = [float(_finite_float(t.get("net_pnl")) or 0.0) for t in trades]
    gross = [float(_finite_float(t.get("gross_pnl")) or 0.0) for t in trades]
    costs = [float(_finite_float(t.get("costs")) or 0.0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    agreements = list(model_agreements or [])
    return {
        "total_trades": len(trades),
        "win_rate": (len(wins) / len(trades)) if trades else 0.0,
        "net_pnl": sum(pnls),
        "gross_pnl": sum(gross),
        "costs": sum(costs),
        "profit_factor": (sum(wins) / abs(sum(losses))) if losses else (99.0 if wins else 0.0),
        "max_drawdown": max_dd,
        "average_trade": statistics.fmean(pnls) if pnls else 0.0,
        "ce_trades": sum(1 for t in trades if _normalise_side(t.get("side")) == "CE"),
        "pe_trades": sum(1 for t in trades if _normalise_side(t.get("side")) == "PE"),
        "skipped_reason_counts": dict(skipped_reason_counts or {}),
        "model_agreement_stats": {
            "count": len(agreements),
            "avg_disagreement": statistics.fmean(agreements) if agreements else 0.0,
            "max_disagreement": max(agreements) if agreements else 0.0,
        },
    }


def selected_option_state_row(
    *,
    selected_option: Optional[Mapping[str, Any]] = None,
    spread_pct: Optional[float] = None,
    open_position: Optional[Mapping[str, Any] | PaperPosition] = None,
    realized_pnl: float = 0.0,
    unrealized_pnl: float = 0.0,
) -> Tuple[str, object, object, object, object, object, object, str, str]:
    if open_position is not None:
        if isinstance(open_position, PaperPosition):
            symbol = open_position.symbol
            strike = "-" if open_position.strike is None else open_position.strike
            expiry = open_position.expiry or "-"
            ltp = f"{open_position.entry_price:.2f}"
            spread = f"{open_position.entry_spread_pct * 100:.2f}"
        else:
            symbol = str(open_position.get("symbol") or "Open paper position")
            strike = open_position.get("strike") or "-"
            expiry = open_position.get("expiry") or "-"
            ltp = open_position.get("entry_price") or open_position.get("ltp") or "-"
            spread = open_position.get("entry_spread_pct") or open_position.get("spread_pct") or "-"
        return (symbol, strike, expiry, "-", "-", ltp, spread, f"{realized_pnl:.2f}", f"{unrealized_pnl:.2f}")
    opt = dict(selected_option or {})
    if opt:
        return (
            str(opt.get("symbol") or opt.get("trading_symbol") or "-"),
            opt.get("strike") or opt.get("strike_price") or "-",
            opt.get("expiry") or opt.get("expiry_date") or "-",
            opt.get("bid") or "-",
            opt.get("ask") or "-",
            opt.get("ltp") or opt.get("close") or "-",
            "-" if spread_pct is None else f"{float(spread_pct) * 100:.2f}",
            f"{realized_pnl:.2f}",
            f"{unrealized_pnl:.2f}",
        )
    return ("No open paper position", "-", "-", "-", "-", "-", "-", f"{realized_pnl:.2f}", f"{unrealized_pnl:.2f}")


def decision_to_dict(result: DecisionResult) -> Dict[str, Any]:
    out = asdict(result)
    out["model_outputs"] = [asdict(m) for m in result.model_outputs]
    return out
