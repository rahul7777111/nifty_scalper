"""
Live Chart data contract — NiftyScalper.

Provides:
  - Candle, SignalMarker, ShadowMetrics, OptionChainSummary,
    DataHealth, LiveChartSnapshot dataclasses
  - build_live_chart_snapshot()          — assembles all data into snapshot
  - option_chain_to_summary()            — parses raw chain payload
  - compute_shadow_metrics()             — derives metrics from DB rows
  - assess_data_health()                 — computes health status

All functions are read-only, work without broker credentials,
and degrade gracefully when data is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class SignalMarker:
    timestamp: datetime
    price: float
    side: str          # BUY / SELL / EXIT / BLOCKED
    instrument: str    # CE / PE
    strike: Optional[float] = None
    probability: float = 0.0
    confidence: float = 0.0   # 0.0–1.0
    model_name: str = "N/A"
    threshold: float = 0.0
    expected_edge: Optional[float] = None
    reason: str = ""
    status: str = "shadow"    # shadow / executed / blocked
    trade_id: Optional[str] = None
    pnl: Optional[float] = None


@dataclass
class ShadowMetrics:
    predictions_today: int = 0
    accepted_signals: int = 0
    blocked_signals: int = 0
    resolved_trades: int = 0
    pending_trades: int = 0
    win_rate: float = 0.0      # 0.0–100.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0    # per trade
    max_drawdown: float = 0.0
    daily_pnl: float = 0.0
    avg_holding_minutes: float = 0.0
    calibration_error: Optional[float] = None
    drift_warning: bool = False
    last_prediction_time: Optional[datetime] = None


@dataclass
class OptionChainSummary:
    atm_strike: Optional[float] = None
    ce_ltp: Optional[float] = None
    pe_ltp: Optional[float] = None
    ce_iv: Optional[float] = None
    pe_iv: Optional[float] = None
    ce_oi: Optional[float] = None
    pe_oi: Optional[float] = None
    pcr: Optional[float] = None
    ce_bid_ask_spread: Optional[float] = None
    pe_bid_ask_spread: Optional[float] = None
    liquidity_score: str = "N/A"   # OK / WARNING / DANGER / N/A
    is_stale: bool = False
    wide_spread_warning: bool = False
    iv_spike_warning: bool = False


@dataclass
class DataHealth:
    last_candle_time: Optional[datetime] = None
    last_tick_time: Optional[datetime] = None
    stale_data_seconds: int = -1    # -1 = never received
    missing_candles: int = 0
    broker_connected: bool = False
    model_loaded: bool = False
    shadow_logger_ok: bool = False
    db_write_ok: bool = False
    api_error_count_today: int = 0
    status: str = "UNKNOWN"         # OK / WARNING / CRITICAL
    status_reason: str = ""


@dataclass
class LiveChartSnapshot:
    timestamp: datetime = field(default_factory=datetime.now)
    spot_price: Optional[float] = None
    futures_price: Optional[float] = None
    atm_strike: Optional[float] = None
    candles: list = field(default_factory=list)
    volume: list = field(default_factory=list)
    vwap: Optional[float] = None
    ema_9: Optional[float] = None
    ema_21: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None
    previous_day_high: Optional[float] = None
    previous_day_low: Optional[float] = None
    session_high: Optional[float] = None
    session_low: Optional[float] = None
    opening_range_high: Optional[float] = None
    opening_range_low: Optional[float] = None
    current_signal: Optional[str] = None   # BUY / SELL / EXIT / NO_TRADE
    latest_prediction: Optional[SignalMarker] = None
    option_chain_summary: OptionChainSummary = field(default_factory=OptionChainSummary)
    shadow_metrics: ShadowMetrics = field(default_factory=ShadowMetrics)
    data_health: DataHealth = field(default_factory=DataHealth)
    market_status: str = "CLOSED"
    broker_connection_status: str = "DISCONNECTED"
    model_status: str = "NOT_LOADED"
    shadow_mode_active: bool = True
    regime_label: str = "N/A"


# ---------------------------------------------------------------------------
# Candle normalizer
# ---------------------------------------------------------------------------

def normalize_live_chart_candles(
    rows: Any,
) -> list[Candle]:
    """Normalize heterogeneous candle inputs into live_chart_snapshot.Candle objects.

    Accepts:
      - Objects with ``time`` or ``timestamp`` attribute
        (e.g. market_data.Candle, live_chart_snapshot.Candle)
      - Dicts with time/timestamp/ts/date keys
      - List/tuple rows from broker API (6 elements: time, open, high, low, close, volume)

    Returns a list of Candle objects with timestamp=, open=, high=, low=, close=,
    volume= attributes. Logs rejected count and reason when rows are dropped.

    Does NOT silently return an empty list when input has data.
    """
    if not rows:
        return []

    if not isinstance(rows, (list, tuple)):
        rows = [rows]

    output: list[Candle] = []
    rejected: dict[str, int] = {}

    for idx, row in enumerate(rows):
        try:
            normalized = _normalize_single_candle(row, idx)
            if normalized is not None:
                output.append(normalized)
            else:
                key = "invalid_candle_structure"
                rejected[key] = rejected.get(key, 0) + 1
        except Exception as exc:  # pragma: no cover — defensive
            key = f"exception:{type(exc).__name__}"
            rejected[key] = rejected.get(key, 0) + 1

    total = len(rows)
    kept = len(output)
    dropped = total - kept

    if dropped > 0:
        reasons = "; ".join(f"{k}={v}" for k, v in rejected.items())
        logger.warning(
            "normalize_live_chart_candles: rejected %d/%d rows — %s",
            dropped, total, reasons,
        )
    elif total > 0:
        logger.debug(
            "normalize_live_chart_candles: accepted %d/%d rows", kept, total
        )

    return output


def _normalize_single_candle(row: Any, idx: int) -> Optional[Candle]:
    """Normalize a single row into a Candle or return None on failure."""
    # Case 1: dict-like
    if isinstance(row, dict):
        return _normalize_dict_candle(row)

    # Case 2: object with time/timestamp attribute
    if hasattr(row, "time") or hasattr(row, "timestamp"):
        return _normalize_object_candle(row)

    # Case 3: list/tuple from broker API (positional: time, open, high, low, close, volume)
    if isinstance(row, (list, tuple)) and len(row) >= 6:
        return _normalize_sequence_candle(row)

    # Unrecognized
    return None


def _normalize_dict_candle(d: dict[str, Any]) -> Optional[Candle]:
    """Normalize a dict with time/timestamp/ts/date keys."""
    # Find the time value using multiple possible keys
    t_raw = d.get("time") or d.get("timestamp") or d.get("ts") or d.get("date")
    if t_raw is None:
        return None

    dt = _parse_datetime(t_raw)
    if dt is None:
        return None

    def _float(v, default=0.0):
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    return Candle(
        timestamp=dt,
        open=_float(d.get("open")),
        high=_float(d.get("high")),
        low=_float(d.get("low")),
        close=_float(d.get("close")),
        volume=_float(d.get("volume")),
    )


def _normalize_object_candle(obj: Any) -> Optional[Candle]:
    """Normalize an object with time or timestamp attribute."""
    t_raw = getattr(obj, "time", None) or getattr(obj, "timestamp", None)
    if t_raw is None:
        return None

    dt = _parse_datetime(t_raw)
    if dt is None:
        return None

    def _float(v, default=0.0):
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    return Candle(
        timestamp=dt,
        open=_float(getattr(obj, "open", None)),
        high=_float(getattr(obj, "high", None)),
        low=_float(getattr(obj, "low", None)),
        close=_float(getattr(obj, "close", None)),
        volume=_float(getattr(obj, "volume", None)),
    )


def _normalize_sequence_candle(seq: Any) -> Optional[Candle]:
    """Normalize a list/tuple: (time, open, high, low, close, volume)."""
    if len(seq) < 6:
        return None

    t_raw = seq[0]
    dt = _parse_datetime(t_raw)
    if dt is None:
        return None

    def _float(v, default=0.0):
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    return Candle(
        timestamp=dt,
        open=_float(seq[1]),
        high=_float(seq[2]),
        low=_float(seq[3]),
        close=_float(seq[4]),
        volume=_float(seq[5]),
    )


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Parse a datetime from various timestamp representations."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromtimestamp(float(value))
    except (TypeError, ValueError, OSError):
        pass
    try:
        # Handle ISO strings
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Option chain parser
# ---------------------------------------------------------------------------

# Known broker key variants for option chain fields.
_STRIKE_KEYS = ("strike", "strike_price", "strikePrice")
_BID_KEYS    = ("bid", "bid_price", "best_bid", "bestBidPrice", "best_bid_price")
_ASK_KEYS    = ("ask", "ask_price", "best_ask", "bestAskPrice", "best_ask_price")
_LTP_KEYS    = ("ltp", "last_price", "lastPrice", "lastTradedPrice", "LastTradedPrice", "last_traded_price", "LTP")
_IV_KEYS     = ("iv", "implied_volatility", "impliedVolatility", "IV", "ImpliedVolatility")
_OI_KEYS     = ("oi", "open_interest", "openInterest", "OpenInterest")


def _multi_get(row: dict, keys: tuple[str, ...], default=None):
    """Try each key in order and return the first non-None value."""
    for k in keys:
        v = row.get(k)
        if v is not None:
            return v
    return default


def _strike_from_row(row: dict) -> float:
    v = _multi_get(row, _STRIKE_KEYS, 0.0)
    try:
        return float(v) if v is not None else 0.0
    except Exception:
        return 0.0


def option_chain_to_summary(
    payload: Optional[dict[str, Any]],
    spot: Optional[float] = None,
) -> OptionChainSummary:
    """Parse a raw option chain payload into OptionChainSummary.

    Handles the following broker payload shapes:
      - Flat list under ``payload["chain"]``
      - Nested ``payload["call_options"]`` / ``payload["put_options"]``
      - Nested ``payload["call"]`` / ``payload["put"]`` (contract model style)
      - Rows where each entry has nested ``{"CE": {...}, "PE": {...}}``

    All field names are normalised across broker variants:
      strike / strike_price / strikePrice
      bid / bid_price / best_bid / bestBidPrice / best_bid_price
      ask / ask_price / best_ask / bestAskPrice / best_ask_price
      ltp / last_price / lastPrice / lastTradedPrice / last_traded_price / LTP
      iv / implied_volatility / impliedVolatility / IV / ImpliedVolatility
      oi / open_interest / openInterest / OpenInterest
    """
    summary = OptionChainSummary()

    if payload is None:
        return summary

    try:
        # ATM strike
        if spot is not None:
            summary.atm_strike = round(float(spot) / 50) * 50

        # -----------------------------------------------------------------
        # Normalise the chain to a flat list of row dicts.
        # -----------------------------------------------------------------
        chain: list[dict[str, Any]] = []

        if isinstance(payload, list):
            chain = payload
        elif isinstance(payload, dict):
            # Nested call_options / put_options  (broker variant)
            call_opts = payload.get("call_options") or payload.get("call") or []
            put_opts  = payload.get("put_options")  or payload.get("put")  or []

            if isinstance(call_opts, list) and call_opts and isinstance(put_opts, list) and put_opts:
                for row in call_opts:
                    if isinstance(row, dict):
                        chain.append({**row, "option_type": "CE"})
                for row in put_opts:
                    if isinstance(row, dict):
                        chain.append({**row, "option_type": "PE"})
                if not chain:
                    chain = payload.get("chain", [])
            else:
                # Flat chain list, possibly inside payload
                chain = payload.get("chain", [])

                # Handle rows that are dicts with CE/PE sub-dicts (broker variant)
                if chain and all(isinstance(r, dict) and ("CE" in r or "PE" in r) for r in chain):
                    normalised = []
                    for r in chain:
                        strike_val = _strike_from_row(r)
                        for opt_type in ("CE", "PE"):
                            leg = r.get(opt_type)
                            if isinstance(leg, dict):
                                normalised.append({**leg, "option_type": opt_type, "strike": strike_val})
                    chain = normalised

        if not isinstance(chain, list):
            return summary

        ce_contracts: list[dict] = []
        pe_contracts: list[dict] = []
        for row in chain:
            if not isinstance(row, dict):
                continue
            ot = str(_multi_get(row, ("option_type", "instrument", "right", "type")) or "").upper()
            if ot in ("CE", "CALL"):
                ce_contracts.append(row)
            elif ot in ("PE", "PUT"):
                pe_contracts.append(row)

        def _strike_key(c: dict) -> float:
            return _strike_from_row(c)

        ce_contracts.sort(key=_strike_key)
        pe_contracts.sort(key=_strike_key)

        atm = summary.atm_strike
        if atm is not None:
            ce_nearest = min(ce_contracts, key=lambda c: abs(_strike_key(c) - atm), default=None) if ce_contracts else None
            pe_nearest = min(pe_contracts, key=lambda c: abs(_strike_key(c) - atm), default=None) if pe_contracts else None

            def _float(v, default=None):
                try:
                    return float(v) if v is not None else default
                except Exception:
                    return default

            def _int(v, default=None):
                try:
                    return int(v) if v is not None else default
                except Exception:
                    return default

            if ce_nearest:
                summary.ce_ltp = _float(_multi_get(ce_nearest, _LTP_KEYS))
                summary.ce_iv  = _float(_multi_get(ce_nearest, _IV_KEYS))
                summary.ce_oi  = _int(_multi_get(ce_nearest, _OI_KEYS))
                ce_bid = _float(_multi_get(ce_nearest, _BID_KEYS))
                ce_ask = _float(_multi_get(ce_nearest, _ASK_KEYS))
                if ce_bid is not None and ce_ask is not None:
                    summary.ce_bid_ask_spread = ce_ask - ce_bid

            if pe_nearest:
                summary.pe_ltp = _float(_multi_get(pe_nearest, _LTP_KEYS))
                summary.pe_iv  = _float(_multi_get(pe_nearest, _IV_KEYS))
                summary.pe_oi  = _int(_multi_get(pe_nearest, _OI_KEYS))
                pe_bid = _float(_multi_get(pe_nearest, _BID_KEYS))
                pe_ask = _float(_multi_get(pe_nearest, _ASK_KEYS))
                if pe_bid is not None and pe_ask is not None:
                    summary.pe_bid_ask_spread = pe_ask - pe_bid

        # -----------------------------------------------------------------
        # PCR — sum OI across all strikes using all key variants
        # -----------------------------------------------------------------
        ce_oi_total = 0
        pe_oi_total = 0
        for c in chain:
            if not isinstance(c, dict):
                continue
            ot = str(_multi_get(c, ("option_type", "instrument", "right", "type")) or "").upper()
            oi_raw = _multi_get(c, _OI_KEYS)
            if oi_raw is None:
                continue
            try:
                oi_val = int(oi_raw)
            except Exception:
                oi_val = 0
            if ot in ("CE", "CALL"):
                ce_oi_total += oi_val
            elif ot in ("PE", "PUT"):
                pe_oi_total += oi_val

        if ce_oi_total > 0:
            summary.pcr = pe_oi_total / ce_oi_total

        # Staleness
        ts = _multi_get(payload, ("timestamp", "ts")) if isinstance(payload, dict) else None
        if ts is not None:
            try:
                if isinstance(ts, (int, float)):
                    age = datetime.now() - datetime.fromtimestamp(float(ts))
                    summary.is_stale = age.total_seconds() > 60
            except Exception:
                pass

        # Warnings
        ce_spread = summary.ce_bid_ask_spread
        pe_spread = summary.pe_bid_ask_spread
        if (ce_spread is not None and ce_spread > 2.0) or (pe_spread is not None and pe_spread > 2.0):
            summary.wide_spread_warning = True

        ce_iv = summary.ce_iv
        pe_iv = summary.pe_iv
        if (ce_iv is not None and ce_iv > 30) or (pe_iv is not None and pe_iv > 30):
            summary.iv_spike_warning = True

        # Liquidity score
        wide  = summary.wide_spread_warning
        spike = summary.iv_spike_warning
        stale = summary.is_stale
        if stale:
            summary.liquidity_score = "DANGER"
        elif wide and spike:
            summary.liquidity_score = "DANGER"
        elif wide or spike:
            summary.liquidity_score = "WARNING"
        else:
            summary.liquidity_score = "OK"

    except Exception:
        pass

    return summary


# ---------------------------------------------------------------------------
# Shadow metrics computation
# ---------------------------------------------------------------------------

def compute_shadow_metrics(
    prediction_rows: Optional[list[dict[str, Any]]],
    trade_events: Optional[list[dict[str, Any]]],
) -> ShadowMetrics:
    """Derive ShadowMetrics from DB prediction/trade rows."""
    metrics = ShadowMetrics()

    if prediction_rows:
        try:
            now = datetime.now()
            today_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
            if now < today_start:
                today_start = today_start - timedelta(days=1)  # pre-market: use yesterday
            today_start_ts = today_start.timestamp()

            # Filter today's predictions
            today_rows = [
                r for r in prediction_rows
                if _float_val(r.get("ts"), 0) >= today_start_ts
            ]
            metrics.predictions_today = len(today_rows)

            # Accepted vs blocked
            for r in today_rows:
                taken = r.get("trade_taken") in (True, 1, "1", "true", "True")
                if taken:
                    metrics.accepted_signals += 1
                else:
                    metrics.blocked_signals += 1

            # Win rate (prob > 0.5 = would-be win among accepted)
            accepted = [r for r in today_rows if r.get("trade_taken") in (True, 1, "1", "true", "True")]
            if accepted:
                wins = sum(1 for r in accepted if _float_val(r.get("probability"), 0) > 0.5)
                metrics.win_rate = (wins / len(accepted)) * 100.0

            # Last prediction time
            if today_rows:
                latest_ts = max(_float_val(r.get("ts"), 0) for r in today_rows)
                if latest_ts > 0:
                    metrics.last_prediction_time = datetime.fromtimestamp(latest_ts)

            # Drift check
            if len(today_rows) >= 10:
                recent_probs = [_float_val(r.get("probability"), 0.5) for r in today_rows[:10]]
                mean_p = sum(recent_probs) / len(recent_probs)
                variance = sum((p - mean_p) ** 2 for p in recent_probs) / len(recent_probs)
                if variance > 0.1:
                    metrics.drift_warning = True

        except Exception:
            pass

    if not trade_events:
        return metrics

    try:
        realized: list[float] = []
        holding_times: list[float] = []
        gross_profit = 0.0
        gross_loss = 0.0
        peak = 0.0
        running = 0.0

        for e in trade_events:
            rlz = _float_val(e.get("realized"), None)
            if rlz is None:
                continue
            realized.append(rlz)
            if rlz > 0:
                gross_profit += rlz
            else:
                gross_loss += abs(rlz)

            # Holding time (entry_ts / exit_ts)
            entry_ts = _float_val(e.get("entry_ts") or e.get("ts"), 0)
            exit_ts  = _float_val(e.get("exit_ts") or e.get("close_ts"), 0)
            if exit_ts > entry_ts > 0:
                holding_times.append((exit_ts - entry_ts) / 60.0)

            # Running PnL for MDD
            running += rlz
            if running > peak:
                peak = running
            dd = running - peak
            if dd < metrics.max_drawdown:
                metrics.max_drawdown = dd

        # Profit factor
        if gross_loss > 0:
            metrics.profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            metrics.profit_factor = gross_profit

        # Expectancy
        if realized:
            metrics.expectancy = sum(realized) / len(realized)
            metrics.avg_win = gross_profit / max(sum(1 for r in realized if r > 0), 1)
            metrics.avg_loss = gross_loss / max(sum(1 for r in realized if r < 0), 1)

        # Daily PnL
        now = datetime.now()
        day_start = now.replace(hour=9, minute=15, second=0, microsecond=0).timestamp()
        metrics.daily_pnl = sum(r for r in realized if r > 0)  # simplified

        # Average holding time
        if holding_times:
            metrics.avg_holding_minutes = sum(holding_times) / len(holding_times)

        metrics.resolved_trades = len(realized)
        metrics.pending_trades = max(0, metrics.accepted_signals - metrics.resolved_trades)

    except Exception:
        pass

    return metrics


def _float_val(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Data health assessment
# ---------------------------------------------------------------------------

def assess_data_health(
    last_candle_ts: Optional[float],
    last_tick_ts: Optional[float],
    broker_connected: bool,
    model_loaded: bool,
    shadow_logger_ok: bool,
    db_write_ok: bool,
    missing_candles: int,
    api_errors_today: int,
) -> DataHealth:
    """Return DataHealth with status OK / WARNING / CRITICAL."""
    health = DataHealth(
        broker_connected=broker_connected,
        model_loaded=model_loaded,
        shadow_logger_ok=shadow_logger_ok,
        db_write_ok=db_write_ok,
        missing_candles=missing_candles,
        api_error_count_today=api_errors_today,
    )

    now_ts = datetime.now().timestamp()

    if last_candle_ts:
        try:
            health.last_candle_time = datetime.fromtimestamp(last_candle_ts)
            health.stale_data_seconds = int(now_ts - last_candle_ts)
        except Exception:
            pass

    if last_tick_ts:
        try:
            health.last_tick_time = datetime.fromtimestamp(last_tick_ts)
        except Exception:
            pass

    # Determine status
    reasons: list[str] = []

    if not broker_connected:
        reasons.append("broker disconnected")
    if not model_loaded:
        reasons.append("model not loaded")
    if health.stale_data_seconds >= 120:
        reasons.append(f"data stale {health.stale_data_seconds}s")
    elif health.stale_data_seconds >= 30:
        reasons.append(f"data age {health.stale_data_seconds}s")
    if missing_candles > 0:
        reasons.append(f"{missing_candles} missing candles")
    if api_errors_today > 10:
        reasons.append(f"{api_errors_today} API errors today")
    if not shadow_logger_ok:
        reasons.append("shadow logger inactive")

    if not broker_connected or not model_loaded or health.stale_data_seconds >= 120:
        health.status = "CRITICAL"
    elif len(reasons) > 0:
        health.status = "WARNING"
    else:
        health.status = "OK"

    health.status_reason = "; ".join(reasons) if reasons else "all systems nominal"

    return health


# ---------------------------------------------------------------------------
# Main snapshot builder
# ---------------------------------------------------------------------------

def build_live_chart_snapshot(
    *,
    timestamp: Optional[datetime] = None,
    spot_price: Optional[float] = None,
    futures_price: Optional[float] = None,
    atm_strike: Optional[float] = None,
    candles: Optional[list[Any]] = None,
    vwap: Optional[float] = None,
    ema_9: Optional[float] = None,
    ema_21: Optional[float] = None,
    ema_50: Optional[float] = None,
    ema_200: Optional[float] = None,
    previous_day_high: Optional[float] = None,
    previous_day_low: Optional[float] = None,
    session_high: Optional[float] = None,
    session_low: Optional[float] = None,
    opening_range_high: Optional[float] = None,
    opening_range_low: Optional[float] = None,
    latest_prediction: Optional[SignalMarker] = None,
    option_chain_payload: Optional[dict[str, Any]] = None,
    shadow_rows: Optional[list[dict[str, Any]]] = None,
    trade_events: Optional[list[dict[str, Any]]] = None,
    last_candle_ts: Optional[float] = None,
    last_tick_ts: Optional[float] = None,
    broker_connected: bool = False,
    model_loaded: bool = False,
    shadow_logger_ok: bool = False,
    db_write_ok: bool = False,
    missing_candles: int = 0,
    api_errors_today: int = 0,
    market_status: str = "CLOSED",
    broker_connection_status: str = "DISCONNECTED",
    model_status: str = "NOT_LOADED",
    shadow_mode_active: bool = True,
    regime_label: str = "N/A",
) -> LiveChartSnapshot:
    """Assemble a LiveChartSnapshot from all available data sources.

    All parameters are optional. The function never crashes and always
    returns a valid snapshot (with N/A defaults for any missing field).
    """
    now = timestamp or datetime.now()

    # Parse candles into Candle objects
    parsed_candles: list[Candle] = []
    volumes: list[float] = []
    if candles:
        for c in candles:
            try:
                # Support both market_data.Candle (time=) and live_chart_snapshot.Candle (timestamp=).
                t = getattr(c, "time", None) or getattr(c, "timestamp", None) if hasattr(c, "time") or hasattr(c, "timestamp") else None
                if t is None:
                    continue
                dt = t if isinstance(t, datetime) else datetime.fromtimestamp(float(t))
                parsed_candles.append(Candle(
                    timestamp=dt,
                    open=float(getattr(c, "open", c.get("open") if isinstance(c, dict) else 0)),
                    high=float(getattr(c, "high", c.get("high") if isinstance(c, dict) else 0)),
                    low=float(getattr(c, "low", c.get("low") if isinstance(c, dict) else 0)),
                    close=float(getattr(c, "close", c.get("close") if isinstance(c, dict) else 0)),
                    volume=float(getattr(c, "volume", c.get("volume") if isinstance(c, dict) else 0)),
                ))
                volumes.append(parsed_candles[-1].volume)
            except Exception:
                continue

    # Derive current signal from latest prediction
    current_signal = None
    if latest_prediction:
        current_signal = latest_prediction.side

    # Build sub-structures
    chain_summary    = option_chain_to_summary(option_chain_payload, spot_price)
    shadow_metrics   = compute_shadow_metrics(shadow_rows, trade_events)
    data_health      = assess_data_health(
        last_candle_ts, last_tick_ts,
        broker_connected, model_loaded, shadow_logger_ok, db_write_ok,
        missing_candles, api_errors_today,
    )

    return LiveChartSnapshot(
        timestamp=now,
        spot_price=spot_price,
        futures_price=futures_price,
        atm_strike=atm_strike,
        candles=parsed_candles,
        volume=volumes,
        vwap=vwap,
        ema_9=ema_9,
        ema_21=ema_21,
        ema_50=ema_50,
        ema_200=ema_200,
        previous_day_high=previous_day_high,
        previous_day_low=previous_day_low,
        session_high=session_high,
        session_low=session_low,
        opening_range_high=opening_range_high,
        opening_range_low=opening_range_low,
        current_signal=current_signal,
        latest_prediction=latest_prediction,
        option_chain_summary=chain_summary,
        shadow_metrics=shadow_metrics,
        data_health=data_health,
        market_status=market_status,
        broker_connection_status=broker_connection_status,
        model_status=model_status,
        shadow_mode_active=shadow_mode_active,
        regime_label=regime_label,
    )