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

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional


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
# Option chain parser
# ---------------------------------------------------------------------------

def option_chain_to_summary(
    payload: Optional[dict[str, Any]],
    spot: Optional[float] = None,
) -> OptionChainSummary:
    """Parse a raw option chain payload into OptionChainSummary."""
    summary = OptionChainSummary()

    if payload is None:
        return summary

    try:
        # ATM strike
        if spot is not None:
            summary.atm_strike = round(float(spot) / 50) * 50

        # Find CE and PE nearest to ATM
        chain = payload if isinstance(payload, list) else payload.get("chain", [])
        if not isinstance(chain, list):
            return summary

        ce_contracts = []
        pe_contracts = []
        for row in chain:
            if not isinstance(row, dict):
                continue
            ot = str(row.get("option_type", "") or row.get("instrument", "")).upper()
            if ot in ("CE", "CALL"):
                ce_contracts.append(row)
            elif ot in ("PE", "PUT"):
                pe_contracts.append(row)

        def _strike_key(c: dict) -> float:
            try:
                return float(c.get("strike", 0))
            except Exception:
                return 0.0

        ce_contracts.sort(key=_strike_key)
        pe_contracts.sort(key=_strike_key)

        atm = summary.atm_strike
        if atm is not None:
            # Nearest ATM CE and PE
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
                summary.ce_ltp    = _float(ce_nearest.get("ltp") or ce_nearest.get("last_price"))
                summary.ce_iv     = _float(ce_nearest.get("iv") or ce_nearest.get("implied_volatility"))
                summary.ce_oi     = _int(ce_nearest.get("oi") or ce_nearest.get("open_interest"))
                ce_bid = _float(ce_nearest.get("bid"))
                ce_ask = _float(ce_nearest.get("ask"))
                if ce_bid is not None and ce_ask is not None:
                    summary.ce_bid_ask_spread = ce_ask - ce_bid

            if pe_nearest:
                summary.pe_ltp    = _float(pe_nearest.get("ltp") or pe_nearest.get("last_price"))
                summary.pe_iv     = _float(pe_nearest.get("iv") or pe_nearest.get("implied_volatility"))
                summary.pe_oi     = _int(pe_nearest.get("oi") or pe_nearest.get("open_interest"))
                pe_bid = _float(pe_nearest.get("bid"))
                pe_ask = _float(pe_nearest.get("ask"))
                if pe_bid is not None and pe_ask is not None:
                    summary.pe_bid_ask_spread = pe_ask - pe_bid

        # PCR
        ce_oi_total = sum(
            int(c.get("oi", 0) or 0) for c in ce_contracts
            if c.get("oi") is not None
        )
        pe_oi_total = sum(
            int(c.get("oi", 0) or 0) for c in pe_contracts
            if c.get("oi") is not None
        )
        if ce_oi_total > 0:
            summary.pcr = pe_oi_total / ce_oi_total

        # Staleness
        ts = payload.get("timestamp") or payload.get("ts")
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
        wide = summary.wide_spread_warning
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