#!/usr/bin/env python3
"""
src/synthetic_option_chain.py

Black-Scholes synthetic NIFTY option chain for Paper Forward Monitor only.
Never used for real broker order placement.
"""

from __future__ import annotations

import math
import os
from datetime import date, datetime, time, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

CHAIN_SOURCE = "BLACK_SCHOLES_SYNTHETIC"
QUALITY = "SYNTHETIC_CHAIN"
MIN_PRICE = 0.05

PAPER_FORWARD_REQUIRED_COLUMNS = (
    "trading_symbol",
    "symbol",
    "strike_price",
    "option_type",
    "expiry",
    "ltp",
    "theoretical_price",
    "bid",
    "ask",
    "mid",
    "spread",
    "spread_pct",
    "volume",
    "oi",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "dte_days",
    "underlying_ltp",
    "spot",
    "moneyness",
    "abs_moneyness",
    "synthetic",
    "chain_source",
    "quality",
)


def _bool_env(name: str, default: bool = False) -> bool:
    return str(os.getenv(name, str(default))).strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def load_bs_config() -> Dict[str, Any]:
    return {
        "use_black_scholes_chain": _bool_env("PF_USE_BLACK_SCHOLES_CHAIN", True),
        "strike_step": _int_env("PF_BS_STRIKE_STEP", 50),
        "strikes_each_side": _int_env("PF_BS_STRIKES_EACH_SIDE", 20),
        "default_iv": _float_env("PF_BS_DEFAULT_IV", 0.18),
        "risk_free_rate": _float_env("PF_BS_RISK_FREE_RATE", 0.06),
        "dividend_yield": _float_env("PF_BS_DIVIDEND_YIELD", 0.00),
        "allow_candle_fallback": _bool_env("PF_ALLOW_SYNTHETIC_CANDLE_FALLBACK", True),
        "min_candles_for_prediction": _int_env("PF_MIN_CANDLES_FOR_PREDICTION", 20),
        "synthetic_candle_count": _int_env("PF_SYNTHETIC_CANDLE_COUNT", 100),
        "iv_min": 0.08,
        "iv_max": 0.45,
    }


def load_paper_mark_config() -> Dict[str, Any]:
    """Paper-forward position marking: prefer live broker LTP when available."""
    return {
        "use_broker_option_prices": _bool_env("PF_USE_BROKER_OPTION_PRICES", True),
        "broker_price_fallback_synthetic": _bool_env("PF_BROKER_PRICE_FALLBACK_SYNTHETIC", True),
    }


def load_paper_sim_config() -> Dict[str, Any]:
    """Paper-forward simulation guards and exit rules (synthetic mode)."""
    return {
        "max_open_per_candidate": _int_env("PF_MAX_OPEN_POSITIONS_PER_CANDIDATE", 1),
        "synthetic_min_entry_gap_sec": _float_env("PF_SYNTHETIC_MIN_ENTRY_GAP_SECONDS", 300.0),
        "synthetic_min_exit_gap_sec": _float_env("PF_SYNTHETIC_MIN_EXIT_GAP_SECONDS", 300.0),
        "paper_target_pct": _float_env("PF_PAPER_TARGET_PCT", 0.20),
        "paper_stop_pct": _float_env("PF_PAPER_STOPLOSS_PCT", 0.10),
        "paper_max_hold_sec": _float_env("PF_PAPER_MAX_HOLD_SECONDS", 900.0),
        "exit_on_opposite_signal": _bool_env("PF_EXIT_ON_OPPOSITE_SIGNAL", True),
    }


def is_synthetic_chain_enabled(broker: str = "mstock") -> bool:
    cfg = load_bs_config()
    if not cfg["use_black_scholes_chain"]:
        return False
    return str(broker or "").strip().lower() in {"mstock", "m.stock", "m_stock", "mstocks"}


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _safe_t_years(dte_days: float) -> float:
    if dte_days <= 0:
        return 1.0 / (365.0 * 24.0 * 60.0)  # ~1 minute horizon at expiry
    return max(dte_days, 1.0 / (365.0 * 24.0)) / 365.0


def bs_price(
    spot: float,
    strike: float,
    t_years: float,
    rate: float,
    q: float,
    iv: float,
    option_type: str,
) -> float:
    spot = max(float(spot), 1e-9)
    strike = max(float(strike), 1e-9)
    iv = max(float(iv), 1e-9)
    t_years = max(float(t_years), 1e-12)
    opt = str(option_type or "").upper()
    if opt in ("CALL", "C"):
        opt = "CE"
    if opt in ("PUT", "P"):
        opt = "PE"
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate - q + 0.5 * iv * iv) * t_years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    if opt == "CE":
        price = spot * math.exp(-q * t_years) * norm_cdf(d1) - strike * math.exp(-rate * t_years) * norm_cdf(d2)
    else:
        price = strike * math.exp(-rate * t_years) * norm_cdf(-d2) - spot * math.exp(-q * t_years) * norm_cdf(-d1)
    return max(float(price), MIN_PRICE)


def bs_greeks(
    spot: float,
    strike: float,
    t_years: float,
    rate: float,
    q: float,
    iv: float,
    option_type: str,
) -> Dict[str, float]:
    spot = max(float(spot), 1e-9)
    strike = max(float(strike), 1e-9)
    iv = max(float(iv), 1e-9)
    t_years = max(float(t_years), 1e-12)
    opt = str(option_type or "").upper()
    if opt in ("CALL", "C"):
        opt = "CE"
    if opt in ("PUT", "P"):
        opt = "PE"
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate - q + 0.5 * iv * iv) * t_years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    pdf = norm_pdf(d1)
    sign = 1.0 if opt == "CE" else -1.0
    delta = math.exp(-q * t_years) * norm_cdf(d1) if opt == "CE" else math.exp(-q * t_years) * (norm_cdf(d1) - 1.0)
    gamma = math.exp(-q * t_years) * pdf / (spot * iv * sqrt_t)
    vega = spot * math.exp(-q * t_years) * pdf * sqrt_t
    theta_yearly = (
        -(spot * pdf * iv * math.exp(-q * t_years)) / (2.0 * sqrt_t)
        - sign * rate * strike * math.exp(-rate * t_years) * norm_cdf(sign * d2)
        + sign * q * spot * math.exp(-q * t_years) * norm_cdf(sign * d1)
    )
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta_yearly / 365.0,
        "vega": vega / 100.0,
        "iv": iv,
    }


def _candle_close(candle: Any) -> Optional[float]:
    if candle is None:
        return None
    if isinstance(candle, dict):
        for k in ("close", "last", "price", "ltp"):
            v = candle.get(k)
            if v not in (None, ""):
                try:
                    return float(v)
                except Exception:
                    pass
        return None
    for attr in ("close", "last", "price"):
        v = getattr(candle, attr, None)
        if v not in (None, ""):
            try:
                return float(v)
            except Exception:
                pass
    return None


def estimate_iv_from_candles(
    candles: Sequence[Any],
    *,
    default_iv: float = 0.18,
    iv_min: float = 0.08,
    iv_max: float = 0.45,
    bars_per_day: float = 375.0,
) -> Tuple[float, str, str]:
    """Return (iv, source, reason)."""
    closes: List[float] = []
    for c in candles or []:
        px = _candle_close(c)
        if px is not None and px > 0:
            closes.append(px)
    if len(closes) < 2:
        print(f"[BS-IV] source=default iv={default_iv:.4f} reason=no_candles")
        return default_iv, "default", "no_candles"
    start = max(0, len(closes) - 100)
    window = closes[start:]
    if len(window) < 2:
        print(f"[BS-IV] source=default iv={default_iv:.4f} reason=insufficient_closes")
        return default_iv, "default", "insufficient_closes"
    log_rets: List[float] = []
    for i in range(1, len(window)):
        if window[i - 1] > 0 and window[i] > 0:
            log_rets.append(math.log(window[i] / window[i - 1]))
    if len(log_rets) < 2:
        print(f"[BS-IV] source=default iv={default_iv:.4f} reason=no_log_returns")
        return default_iv, "default", "no_log_returns"
    mean = sum(log_rets) / len(log_rets)
    var = sum((r - mean) ** 2 for r in log_rets) / max(len(log_rets) - 1, 1)
    std = math.sqrt(max(var, 0.0))
    annualized = std * math.sqrt(max(bars_per_day, 1.0) * 252.0)
    iv = min(iv_max, max(iv_min, annualized if annualized > 0 else default_iv))
    print(f"[BS-IV] source=candles rows={len(window)} iv={iv:.4f}")
    return iv, "candles", ""


def _parse_expiry(expiry: Any, now: Optional[datetime] = None) -> Tuple[Optional[date], float]:
    now = now or datetime.now(timezone.utc)
    if expiry is None or expiry == "":
        return None, 0.0
    if isinstance(expiry, datetime):
        exp_dt = expiry
    elif isinstance(expiry, date):
        exp_dt = datetime.combine(expiry, time(15, 30))
    else:
        text = str(expiry).strip()
        exp_dt = None
        for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d-%b-%Y", "%d/%m/%Y"):
            try:
                exp_dt = datetime.strptime(text, fmt)
                break
            except Exception:
                continue
        if exp_dt is None:
            try:
                exp_dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except Exception:
                return None, 0.0
    if now.tzinfo is not None and exp_dt.tzinfo is None:
        exp_dt = exp_dt.replace(tzinfo=now.tzinfo)
    elif now.tzinfo is None and exp_dt.tzinfo is not None:
        now = now.replace(tzinfo=exp_dt.tzinfo)
    elif exp_dt.tzinfo is None:
        exp_dt = exp_dt.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
    exp_eod = datetime.combine(exp_dt.date(), time(15, 30), tzinfo=exp_dt.tzinfo)
    dte_days = max(0.0, (exp_eod - now).total_seconds() / 86400.0)
    return exp_dt.date(), dte_days


def atm_strike(spot: float, step: int = 50) -> float:
    step = max(int(step), 1)
    return round(float(spot) / step) * step


def _liquidity_for_strike(
    strike: float,
    atm: float,
    spot: float,
    mid: float,
    step: int,
) -> Tuple[int, int, float]:
    dist_strikes = abs(strike - atm) / max(step, 1)
    dist_pct = abs(strike - spot) / max(spot, 1e-9)
    atm_weight = max(0.05, 1.0 - 0.08 * dist_strikes - 0.5 * dist_pct)
    premium_penalty = max(0.3, min(1.0, mid / max(spot * 0.01, 1.0)))
    volume = int(5000 * atm_weight * premium_penalty)
    oi = int(25000 * atm_weight * premium_penalty)
    spread_pct = 0.002 + 0.0015 * dist_strikes + max(0.0, 0.01 - mid / max(spot, 1.0))
    spread_pct = min(spread_pct, 0.08)
    return max(volume, 1), max(oi, 1), spread_pct


def _format_expiry_tag(expiry_date: date) -> str:
    return expiry_date.strftime("%d%b%y").upper()


def _build_row(
    *,
    spot: float,
    strike: float,
    option_type: str,
    expiry_date: date,
    dte_days: float,
    iv: float,
    rate: float,
    q: float,
    step: int,
    underlying: str = "NIFTY",
) -> Dict[str, Any]:
    t_years = _safe_t_years(dte_days)
    theo = bs_price(spot, strike, t_years, rate, q, iv, option_type)
    greeks = bs_greeks(spot, strike, t_years, rate, q, iv, option_type)
    volume, oi, spread_pct = _liquidity_for_strike(strike, atm_strike(spot, step), spot, theo, step)
    spread = max(theo * spread_pct, MIN_PRICE * 0.2)
    mid = max(theo, MIN_PRICE)
    bid = max(mid - spread / 2.0, MIN_PRICE)
    ask = max(bid, mid + spread / 2.0)
    if ask < bid:
        ask = bid
    moneyness = (strike - spot) / max(spot, 1e-9)
    exp_tag = _format_expiry_tag(expiry_date)
    sym = f"{underlying}{exp_tag}{int(strike)}{option_type}"
    return {
        "trading_symbol": sym,
        "symbol": sym,
        "tradingsymbol": sym,
        "strike_price": float(strike),
        "strike": float(strike),
        "option_type": option_type,
        "expiry": expiry_date.isoformat(),
        "expiry_date": expiry_date.isoformat(),
        "ltp": mid,
        "theoretical_price": theo,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": ask - bid,
        "spread_pct": (ask - bid) / max(mid, MIN_PRICE),
        "volume": volume,
        "oi": oi,
        "iv": greeks["iv"],
        "delta": greeks["delta"],
        "gamma": greeks["gamma"],
        "theta": greeks["theta"],
        "vega": greeks["vega"],
        "dte_days": dte_days,
        "underlying_ltp": spot,
        "spot": spot,
        "underlying": spot,
        "underlying_price": spot,
        "moneyness": moneyness,
        "abs_moneyness": abs(moneyness),
        "synthetic": True,
        "chain_source": CHAIN_SOURCE,
        "quality": QUALITY,
        "symbol_root": underlying,
        "exchange": os.getenv("MSTOCK_OPTION_EXCHANGE_ID") or os.getenv("MSTOCK_SCRIPMASTER_EXCH") or "NFO",
    }


def generate_synthetic_option_chain(
    *,
    spot: float,
    expiry: Union[str, date, datetime],
    timestamp: Optional[datetime] = None,
    candles: Optional[Sequence[Any]] = None,
    underlying: str = "NIFTY",
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate CE/PE rows for Paper Forward. Returns (rows, meta)."""
    cfg = dict(load_bs_config())
    if config:
        cfg.update(config)
    now = timestamp or datetime.now(timezone.utc)
    spot_f = float(spot)
    if spot_f <= 0:
        raise ValueError("spot must be positive")
    expiry_date, dte_days = _parse_expiry(expiry, now)
    if expiry_date is None:
        raise ValueError("invalid expiry")
    iv, iv_source, iv_reason = estimate_iv_from_candles(
        candles or [],
        default_iv=cfg["default_iv"],
        iv_min=cfg["iv_min"],
        iv_max=cfg["iv_max"],
    )
    step = int(cfg["strike_step"])
    each_side = int(cfg["strikes_each_side"])
    atm = atm_strike(spot_f, step)
    strikes = [atm + (i - each_side) * step for i in range(2 * each_side + 1)]
    rows: List[Dict[str, Any]] = []
    for strike in strikes:
        if strike <= 0:
            continue
        for opt in ("CE", "PE"):
            rows.append(
                _build_row(
                    spot=spot_f,
                    strike=float(strike),
                    option_type=opt,
                    expiry_date=expiry_date,
                    dte_days=dte_days,
                    iv=iv,
                    rate=cfg["risk_free_rate"],
                    q=cfg["dividend_yield"],
                    step=step,
                    underlying=underlying,
                )
            )
    ce = sum(1 for r in rows if r.get("option_type") == "CE")
    pe = sum(1 for r in rows if r.get("option_type") == "PE")
    max_vol = max((r.get("volume") or 0) for r in rows) if rows else 0
    max_oi = max((r.get("oi") or 0) for r in rows) if rows else 0
    print(
        f"[BS-CHAIN-GENERATE] broker=mstock spot={spot_f:.2f} expiry={expiry_date} "
        f"dte_days={dte_days:.4f} atm={atm} strikes={len(strikes)} rows={len(rows)}"
    )
    print(f"[BS-LIQUIDITY] rows={len(rows)} atm_strike={atm} max_volume={max_vol} max_oi={max_oi}")
    meta = {
        "chain_source": CHAIN_SOURCE,
        "quality": QUALITY,
        "synthetic": True,
        "spot": spot_f,
        "expiry": expiry_date.isoformat(),
        "dte_days": dte_days,
        "atm_strike": atm,
        "iv": iv,
        "iv_source": iv_source,
        "iv_reason": iv_reason,
        "rows": len(rows),
        "ce_rows": ce,
        "pe_rows": pe,
        "strikes": len(strikes),
    }
    return rows, meta


def infer_paper_forward_data_quality(
    *,
    option_rows: int,
    candle_count: int,
    spot: Any,
    chain_source: str = "",
    synthetic: bool = False,
    candle_source: str = "",
    synthetic_candles: bool = False,
    broker_ltp_used: bool = False,
    chain_spot_valid: bool = True,
) -> str:
    has_spot = spot not in (None, "", 0, 0.0)
    is_synth = synthetic or chain_source == CHAIN_SOURCE
    cfg = load_bs_config()
    min_candles = int(cfg.get("min_candles_for_prediction", 20))
    synth_target = int(cfg.get("synthetic_candle_count", 100))
    csrc = str(candle_source or "").upper()
    is_synth_candles = bool(
        synthetic_candles
        or csrc in ("SYNTHETIC_SPOT_FALLBACK", "SYNTHETIC_SPOT_FALLBACK".upper())
        or ("SYNTHETIC" in csrc and "FALLBACK" in csrc)
    )
    is_real_candles = bool(
        candle_count >= min_candles
        and not is_synth_candles
        and (
            "HISTORICAL" in csrc
            or csrc.endswith("_HISTORICAL")
            or csrc in ("MSTOCK_HISTORICAL", "DHAN_HISTORICAL", "LIVE_CHART_CACHE", "LIVE_CHART_RAW_CACHE", "LIVE_CHART_RENDER_CACHE", "CHART_CANDLES_CACHE", "SCALPER_CANDLES", "SCALPER_SPOT_CANDLES")
        )
    )
    if not chain_spot_valid:
        return "INVALID_CHAIN_SPOT"
    if not has_spot:
        return "SPOT_MISSING"
    if option_rows <= 0:
        return "OPTION_CHAIN_EMPTY"
    if not is_synth:
        if candle_count >= min_candles:
            if broker_ltp_used:
                return "REAL_CANDLES_REAL_CHAIN_BROKER_LTP"
            return "REAL_CANDLES_REAL_CHAIN"
        return "WAITING_FOR_CANDLES"
    if is_synth_candles and candle_count >= synth_target:
        return "SYNTHETIC_CHAIN_WITH_SYNTHETIC_CANDLES"
    if candle_count >= min_candles:
        if is_real_candles:
            if broker_ltp_used:
                return "SYNTHETIC_CHAIN_WITH_BROKER_MARK"
            return "REAL_CANDLES_SYNTHETIC_CHAIN"
        return "SYNTHETIC_CHAIN_OK"
    if cfg["allow_candle_fallback"] and candle_count > 0:
        if is_real_candles:
            if broker_ltp_used:
                return "SYNTHETIC_CHAIN_WITH_BROKER_MARK"
            return "REAL_CANDLES_SYNTHETIC_CHAIN"
        return "SYNTHETIC_CHAIN_WITH_SPOT_CANDLE_FALLBACK"
    if cfg["allow_candle_fallback"]:
        return "SYNTHETIC_CHAIN_WITH_SPOT_CANDLE_FALLBACK"
    if candle_count < min_candles:
        return "WAITING_FOR_CANDLES"
    return "SYNTHETIC_CHAIN_READY_WAITING_FOR_CANDLES"


def _deterministic_noise(index: int, spot: float) -> float:
    """Tiny deterministic intraday wiggle seeded from spot (not random per refresh)."""
    seed = int(round(float(spot) * 100.0)) & 0xFFFFFFFF
    x = (seed ^ (index * 2654435761)) & 0xFFFFFFFF
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = (x >> 16) ^ x
    unit = (x & 0xFFFF) / 65535.0 - 0.5
    return unit * max(float(spot) * 0.00008, 0.02)


def build_spot_candle_fallback(
    spot: float,
    timestamp: Optional[datetime] = None,
    *,
    count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate deterministic synthetic spot candles for paper-forward feature fallback."""
    cfg = load_bs_config()
    n = int(count or cfg.get("synthetic_candle_count", 100))
    n = max(n, int(cfg.get("min_candles_for_prediction", 20)))
    now = timestamp or datetime.now(timezone.utc)
    spot_f = float(spot)
    if spot_f <= 0:
        return []
    closes: List[float] = []
    price = spot_f
    for i in range(n):
        price = max(1.0, price + _deterministic_noise(i, spot_f))
        closes.append(price)
    # Anchor last close to current spot for realism
    if closes:
        drift = spot_f - closes[-1]
        closes = [c + drift for c in closes]
    rows: List[Dict[str, Any]] = []
    for i, close_px in enumerate(closes):
        bar_idx = n - 1 - i
        ts = now.replace(microsecond=0) if hasattr(now, "replace") else now
        try:
            from datetime import timedelta
            bar_ts = ts - timedelta(minutes=bar_idx)
        except Exception:
            bar_ts = ts
        wiggle = max(spot_f * 0.00005, 0.01)
        open_px = close_px - _deterministic_noise(i + 1000, spot_f) * 0.5
        high_px = max(open_px, close_px) + wiggle
        low_px = min(open_px, close_px) - wiggle
        vol = int(5000 + abs(_deterministic_noise(i + 2000, spot_f)) * 20000)
        rows.append(
            {
                "time": bar_ts.isoformat(),
                "timestamp": bar_ts.isoformat(),
                "open": round(open_px, 2),
                "high": round(high_px, 2),
                "low": round(low_px, 2),
                "close": round(close_px, 2),
                "volume": vol,
                "candle_source": "SYNTHETIC_SPOT_FALLBACK",
                "synthetic_candles": True,
            }
        )
    first = rows[0]["close"] if rows else None
    last = rows[-1]["close"] if rows else None
    print(
        f"[PF-SYNTH-CANDLES] rows={len(rows)} spot={spot_f:.2f} first={first} last={last} "
        f"source=SYNTHETIC_SPOT_FALLBACK"
    )
    return rows


def map_paper_forward_reason(raw: str) -> Tuple[str, str]:
    """Map internal reason strings to clean GUI labels. Returns (display_label, raw_reason)."""
    r = str(raw or "").strip()
    rl = r.lower()
    exact = {
        "TRADE_OPENED_PAPER": "TRADE_OPENED_PAPER",
        "HOLD_EXISTING_POSITION": "HOLD_EXISTING_POSITION",
        "ENTRY_COOLDOWN": "ENTRY_COOLDOWN",
        "SIGNAL_DEDUPE": "SIGNAL_DEDUPE",
        "PAPER_EXIT_TARGET": "PAPER_EXIT_TARGET",
        "PAPER_EXIT_STOP": "PAPER_EXIT_STOP",
        "PAPER_EXIT_MAX_HOLD": "PAPER_EXIT_MAX_HOLD",
        "PAPER_EXIT_OPPOSITE": "PAPER_EXIT_OPPOSITE",
        "PAPER_EXIT_EOD": "PAPER_EXIT_EOD",
        "LOW_CONFIDENCE": "LOW_CONFIDENCE",
        "NO_TRADE": "NO_TRADE",
    }
    if r in exact:
        return exact[r], r
    if not r or rl == "ok":
        return "TRADE_OPENED_PAPER", r
    if rl.startswith("low_confidence"):
        return r, r
    if rl in ("no_trade",):
        return "NO_TRADE", r
    if "hold_existing" in rl:
        return "HOLD_EXISTING_POSITION", r
    if "entry_cooldown" in rl:
        return "ENTRY_COOLDOWN", r
    if "signal_dedupe" in rl or "repeated_signal" in rl:
        return "SIGNAL_DEDUPE", r
    if "paper_exit_target" in rl or rl == "target":
        return "PAPER_EXIT_TARGET", r
    if "paper_exit_stop" in rl or rl == "stop":
        return "PAPER_EXIT_STOP", r
    if "paper_exit_max_hold" in rl or "max_hold" in rl:
        return "PAPER_EXIT_MAX_HOLD", r
    if "paper_exit_opposite" in rl or rl == "opposite":
        return "PAPER_EXIT_OPPOSITE", r
    if "paper_exit_eod" in rl or rl == "eod":
        return "PAPER_EXIT_EOD", r
    if "auto_directional_mixed_regime_no_trade" in rl:
        return "MIXED_REGIME_NO_TRADE", r
    if "cost_edge" in rl or "cost_proxy" in rl or "cost_gate" in rl:
        return "COST_GATE", r
    if "spread_" in rl and "_gt_" in rl:
        return "LIQUIDITY_SPREAD_GATE", r
    if "side_policy" in rl or "no_side" in rl or "requires_explicit_side" in rl:
        return "SIDE_POLICY_BLOCK", r
    if "below_threshold" in rl:
        return "BELOW_THRESHOLD", r
    if "synthetic_chain_ready" in rl:
        return "SYNTHETIC_CHAIN_READY", r
    if "waiting_for_candles" in rl or rl == "candles_empty":
        return "WAITING_FOR_CANDLES", r
    if "artifact_not_found" in rl or "missing_artifact" in rl or "candidate_missing_artifact" in rl:
        return "ARTIFACT_NOT_FOUND", r
    if "mixed_regime" in rl:
        return "MIXED_REGIME_NO_TRADE", r
    if "model_route_exception" in rl:
        return "PREDICT_ERROR", r
    if "missing_features" in rl:
        return "MISSING_FEATURES", r
    if len(r) > 32:
        return r[:32], r
    return r, r


def verify_paper_forward_columns(rows: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    if not rows:
        return False, list(PAPER_FORWARD_REQUIRED_COLUMNS)
    sample = rows[0]
    missing = [c for c in PAPER_FORWARD_REQUIRED_COLUMNS if c not in sample]
    return len(missing) == 0, missing
