from __future__ import annotations

import calendar
import hashlib
import json
import logging
import os
import re
import time
import math
from datetime import date
from datetime import datetime as dt_datetime, time as dt_time, timedelta
from dataclasses import dataclass, replace
from threading import Event, Lock
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import pytz  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    pytz = None

try:
    from zoneinfo import ZoneInfo
except Exception:  # noqa: BLE001
    ZoneInfo = None  # type: ignore[assignment]


IST: Optional[object] = None
_TZ_AVAILABLE: bool = False

# Set _TZ_AVAILABLE at module load time — BEFORE any other imports that might
# fail. This ensures the flag reflects timezone-availability state accurately
# regardless of whether later imports (candlestick_patterns, etc.) succeed.
try:
    if ZoneInfo is not None:
        IST = ZoneInfo("Asia/Kolkata")
        _TZ_AVAILABLE = True
    elif pytz is not None:
        IST = pytz.timezone("Asia/Kolkata")
        _TZ_AVAILABLE = True
    else:
        _TZ_AVAILABLE = False
except Exception as exc:
    # If timezone loading fails, log explicitly and leave IST = None.
    # _TZ_AVAILABLE stays False so is_market_open() returns False (fail-closed).
    import sys
    print(f"[TIMEZONE ERROR] Failed to load Asia/Kolkata timezone: {exc}", file=sys.stderr)
    _TZ_AVAILABLE = False

# Guard: if we set IST successfully but the flag is somehow still False, correct it.
if IST is not None:
    _TZ_AVAILABLE = True

# Guard: if neither zoneinfo nor pytz is available at all, log and ensure fail-closed.
if ZoneInfo is None and pytz is None:
    _TZ_AVAILABLE = False
    IST = None


def is_market_open() -> bool:
    """Return True only during IST 09:15–15:30; fail-closed if timezone unavailable.

    Safety rules:
    - Never use local (wall-clock) time as a proxy for IST.
    - If neither ZoneInfo nor pytz can resolve Asia/Kolkata, return False
      so the system does not trade outside intended hours.
    """
    if not _TZ_AVAILABLE or IST is None:
        # Fail-closed: block trading when we cannot determine IST correctly.
        # Do NOT fall back to local wall-clock time — that would incorrectly
        # treat e.g. midnight IST-equivalent hours as "market open".
        import sys
        print(
            "[TIMEZONE SAFETY] Cannot determine IST; is_market_open() returning False "
            "(fail-closed). _TZ_AVAILABLE=False. Restore timezone support to enable "
            "market-hours gating.",
            file=sys.stderr,
        )
        return False
    now = dt_datetime.now(IST).time()
    return dt_time(9, 15) <= now <= dt_time(15, 30)

from candlestick_patterns import (
    is_bearish_engulfing,
    is_bullish_engulfing,
    is_doji,
    is_hammer,
    is_shooting_star,
)
from config import StrategyConfig
from cost_model import CostModel, evaluate_live_execution_friction
from greeks import OptionType, delta, gamma, implied_volatility, theta as bs_theta, vega
from indicators import adx, atr, ema, rsi, sma, supertrend, roc, choppiness_index, pivot_points
from market_data import Candle
from mean_reversion import evaluate_mean_reversion
from mstock_client import MStockTypeBClient, Order
from stat_arb import evaluate_pair_spread
from volatility import forecast_volatility
from strategy_allocator import allocate_sleeves_for_regime, detect_regime, get_regime_tuning, select_strategy_for_regime
from risk_cvar import compute_cvar
from greeks_manager import GreeksManager
from exit_optimizer import (
    ExitOptimizer,
    TripleBarrierState,
    evaluate_triple_barrier_state,
    initialize_triple_barrier_state,
)
from label_policies import get_label_policy_spec
from ml_signals import evaluate_ml_gating_before_execution, feature_vector_from_candles, load_model, predict

try:
    from ml.ensemble_artifact_loader import try_load_ensemble_dir
except Exception:  # noqa: BLE001
    try:
        from ensemble_artifact_loader import try_load_ensemble_dir  # type: ignore
    except Exception:
        try_load_ensemble_dir = None  # type: ignore

# Production candidate router (must be called for every entry decision path)
try:
    from candidate_router import route_candidate_decision, get_last_router_decision
except Exception:  # noqa: BLE001
    route_candidate_decision = None  # type: ignore
    get_last_router_decision = None  # type: ignore
from position_sizing import calculate_position_size, volatility_target_size, kelly_fraction
from trailing import atr_trailing_stop
from backtest_harness import simulate_simple


LOGGER = logging.getLogger(__name__)


def calc_partial_exit_qty(total_qty: int, partial_pct: float, lot_size: int) -> int:
    """Return a valid partial-exit quantity.

    - Rounds down to the nearest `lot_size` multiple.
    - Ensures we don't fully close the position (must leave at least one lot).
    - Returns 0 when a partial exit is not possible.
    """

    try:
        qty = int(total_qty)
    except Exception:
        return 0
    if qty <= 0:
        return 0

    try:
        pct = float(partial_pct)
    except Exception:
        return 0
    if pct <= 0:
        return 0
    if pct > 1.0:
        pct = 1.0

    try:
        step = int(lot_size)
    except Exception:
        step = 1
    if step <= 0:
        step = 1

    # Need at least 2 lots to partially exit while leaving >=1 lot.
    if qty < step * 2:
        return 0

    steps_total = qty // step
    if steps_total < 2:
        return 0

    desired_steps = int(steps_total * pct)
    if desired_steps < 1:
        desired_steps = 1
    if desired_steps >= steps_total:
        desired_steps = steps_total - 1

    exit_qty = desired_steps * step
    if exit_qty <= 0 or exit_qty >= qty:
        return 0
    return exit_qty


def submit_paper_trade_order(
    strategy_context: Dict[str, Any],
    prediction_id: str,
    probability: float,
) -> Dict[str, Any]:
    """Build a trade payload with explicit ML telemetry linkage."""

    payload = dict(strategy_context or {})
    metadata = dict(payload.get("metadata") or {})
    metadata["prediction_id"] = str(prediction_id or "")
    metadata["probability"] = float(probability)
    payload["metadata"] = metadata
    return payload


@dataclass
class TradeState:
    open_orders: List[Order]
    open_multi: List[Dict[str, object]]
    open_directional: List[Dict[str, object]]
    trades_today: int = 0
    last_entry_ts: float = 0.0
    last_exit_ts: float = 0.0
    last_stopout_ts: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    stopouts_today: int = 0
    consecutive_stopouts: int = 0
    consecutive_wins: int = 0
    risk_counters_day: Optional[date] = None
    delta_hedge_orders_today: int = 0
    delta_hedge_orders_day: Optional[date] = None


@dataclass(frozen=True)
class TradeLogEvent:
    ts: float
    event: str  # "OPEN" | "UPDATE" | "CLOSE" | "PARTIAL_CLOSE" | "SKIP"
    trade_id: str
    position_type: str  # "multi" | "directional"
    name: str
    legs: List[Dict[str, object]]
    mtm: Optional[float] = None
    realized: Optional[float] = None
    reason: Optional[str] = None
    margin_required: Optional[float] = None
    # ---- paper-journal validation fields ----
    timestamp: Optional[str] = None  # ISO timestamp of event
    symbol: Optional[str] = None  # e.g. "NIFTY" (root, derived from legs)
    strike: Optional[float] = None  # ATM strike at entry
    option_type: Optional[str] = None  # "CE" | "PE"
    side: Optional[str] = None  # "BUY" | "SELL"
    quantity: Optional[int] = None  # lots × multiplier
    bid: Optional[float] = None  # best bid at entry/exit
    ask: Optional[float] = None  # best ask at entry/exit
    ltp: Optional[float] = None  # last traded price at entry/exit
    execution_price: Optional[float] = None  # actual price used
    execution_price_source: Optional[str] = None  # "ask" | "bid" | "ltp_fallback" | "missing_bid_ask"
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    gross_pnl: Optional[float] = None  # before costs
    net_pnl: Optional[float] = None  # after costs (spread + slippage + brokerage)
    spread_cost: Optional[float] = None  # spread cost in rupees
    slippage_cost: Optional[float] = None  # slippage cost in rupees
    brokerage_cost: Optional[float] = None  # brokerage + taxes + fees in rupees
    exit_reason: Optional[str] = None  # stop_loss | target_hit | time_exit | eod_squareoff | gpt_override
    risk_filter_decisions: Optional[Dict[str, Any]] = None  # dict of what filters passed/failed
    realized_slippage_pct: Optional[float] = None  # slippage / notional_value
    paper_mode: bool = True  # True for simulated fills; False for live
    # ---- additional required journal fields ----
    skip_reason: Optional[str] = None  # why trade was skipped (for SKIP events)
    spread_pct_at_entry: Optional[float] = None  # spread / mid at entry
    spread_pct_at_exit: Optional[float] = None  # spread / mid at exit
    filter_premium_ok: Optional[bool] = None
    filter_spread_ok: Optional[bool] = None
    filter_bid_ask_ok: Optional[bool] = None


@dataclass(frozen=True)
class OptionContract:
    exchange: str
    tradingsymbol: str
    instrument_token: str
    expiry: object
    strike: float
    option_type: str


class NiftyScalper:
    """Very simple Nifty options scalper template.

    This does NOT implement a profitable strategy; it only shows a typical
    control loop structure: read prices, apply rules, send orders, enforce
    risk limits.
    """

    def __init__(
        self,
        client: MStockTypeBClient,
        cfg: StrategyConfig,
        *,
        event_sink: Optional[Callable[[TradeLogEvent], None]] = None,
        on_tick: Optional[Callable[[List[Candle]], None]] = None,
    ) -> None:
        self.client = client
        self.cfg = cfg
        self.paper_mode = not bool(getattr(cfg, "enable_live_trading", False))
        self.state = TradeState(open_orders=[], open_multi=[], open_directional=[])
        self._event_sink = event_sink
        self._on_tick = on_tick
        
        # Integrations
        self.db_manager = None
        # Lazy import to avoid breaking if not present
        try:
            from db import DatabaseManager
            self.db_manager = DatabaseManager()
        except ImportError:
            pass
        self.reconciliation_engine = None
        try:
            from reconciliation import ReconciliationEngine
            self.reconciliation_engine = ReconciliationEngine()
        except ImportError:
            pass
            
        self.notifier = None
        try:
            from alerts import TelegramNotifier
            if cfg.telegram_bot_token and cfg.telegram_chat_id:
                self.notifier = TelegramNotifier(cfg.telegram_bot_token, cfg.telegram_chat_id)
        except ImportError:
            pass
        self._trade_seq = 0
        self._last_candle_err: str = ""
        self._last_candle_err_ts: float = 0.0
        self._dir_style_locked: Optional[str] = None
        self._dir_style_locked_ts: float = 0.0
        self._last_trade_type_key: Optional[str] = None
        self._last_trade_type_streak: int = 0

        # Track strategy-name normalization announcements (avoid log spam).
        self._last_strategy_name_raw: Optional[str] = None
        self._last_strategy_name_effective: Optional[str] = None

        # Auto-mode strategy lock (reduces churn between directional/premium branches).
        self._auto_strat_locked: Optional[str] = None
        self._auto_strat_locked_ts: float = 0.0

        # ---- Reroute cooldown and max-attempts tracking ----
        self._reroute_cooldown_seconds: float = 30.0
        self._last_reroute_ts: float = 0.0
        self._reroute_count_this_cycle: int = 0
        self._cycle_reroute_reset_ts: float = 0.0
        self._max_reroutes_per_cycle: int = 3

        # When intraday candles are enabled but the intraday endpoint returns empty
        # (common for some tokens/accounts), we can synthesize candles from live LTP
        # samples to avoid being stuck in warmup. This does NOT use historical endpoints.
        self._synthetic_candles: Dict[str, List[Candle]] = {}
        self._synthetic_current: Dict[str, Candle] = {}
        self._last_synth_log_ts: float = 0.0

        # Track LTP freshness to avoid trading on stale quotes.
        # key -> {"ltp": float, "seen_ts": float, "change_ts": float}
        self._ltp_watch: Dict[str, Dict[str, float]] = {}
        # Track IV trend to avoid selling into expansion / buying into collapse.
        self._iv_watch: Dict[str, object] = {"value": None, "ts": 0.0}
        # ATR history for IV percentile position sizing.
        self._iv_pct_hist: List[float] = []
        # Track recent option bid/ask spread by symbol/token to avoid entering
        # during sudden liquidity shocks.
        self._entry_spread_watch: Dict[str, List[float]] = {}

        # Cache the latest ATR value for dynamic pyramiding.
        self._cached_atr: Optional[float] = None

        # Win-rate tracker state
        self._strategy_winrate: Dict[str, Dict[str, object]] = {}
        self._last_prediction_id: Optional[str] = None
        self._last_prediction_record: Optional[Dict[str, object]] = None
        self._ml_model_checksum: Optional[str] = None
        self._entries_paused: bool = False
        self._disabled_strategies: Dict[str, float] = {}
        
        # Throttle entry evaluation to candle boundaries to avoid repeated
        # decisions on the same candle (especially when polling every second).
        self._last_entry_bucket_start_ts: Optional[float] = None

        # ---- Equity holdings trading (optional) ----
        # symbol -> position dict {side, qty, entry, stop, target, entry_ts, last_action_ts}
        self._equity_positions: Dict[str, Dict[str, object]] = {}
        self._last_equity_bucket_start_ts: Optional[float] = None
        self._gpt_equity_watchlist_cache: Dict[str, object] = {}
        # Standalone portfolio hedge trade (beta-only) state.
        self._portfolio_hedge_trade: Optional[Dict[str, object]] = None

        # Runtime diagnostics / observability.

        self._entry_block_counts: Dict[str, int] = {}
        self._last_entry_block: Dict[str, object] = {"code": "", "reason": "", "ts": 0.0}
        self._strategy_considered_counts: Dict[str, int] = {}
        self._strategy_decision_counts: Dict[str, int] = {}
        self._strategy_selected_counts: Dict[str, int] = {}
        self._last_router_snapshot: Dict[str, object] = {}
        self._last_portfolio_risk: Dict[str, object] = {}
        self._last_auto_fallback_note: str = ""
        self._config_warnings: List[str] = list(getattr(self.cfg, "validation_warnings", []) or [])
        self._last_gpt_preset_request: Dict[str, object] = {}
        self._auto_delta_hedge_symbol_cache: Dict[str, Tuple[str, str]] = {}
        self._gpt_leg_require_ui_visible: bool = False
        self._gpt_leg_ui_visible_trades: Dict[str, float] = {}
        self._auto_gpt_strategy_parameters: Dict[str, Dict[str, float]] = {}
        self._auto_gpt_directional_steps: Optional[int] = None
        self._auto_gpt_strike_context: Dict[str, object] = {}
        self._last_ml_feature_names: List[str] = []
        self._last_ml_features: List[float] = []
        self._last_regime_profile: Dict[str, object] = {}
        self._last_ml_bet_multiplier: float = 0.0
        self._last_baseline_vol_lots: int = 0
        self._last_final_allocated_lots: int = 0
        self._peak_equity_rupees: float = max(float(getattr(self.cfg, "account_capital", 100000.0) or 100000.0), 1.0)

        # ---- P1: GPT Exit Management state ----
        self._gpt_exit_mgmt_last_run: float = 0.0

        # Enhancements integration
        try:
            self.greeks_mgr = GreeksManager()
        except Exception:
            self.greeks_mgr = None
        try:
            self.exit_optimizer = ExitOptimizer(getattr(cfg, "exit_optimizer", None))
        except Exception:
            self.exit_optimizer = None

        # ML model
        self.ml_model = None
        try:
            if getattr(cfg, "enable_ml_signals", False):
                ensemble_dir = str(getattr(cfg, "ensemble_artifact_dir", "") or "").strip()
                if ensemble_dir and try_load_ensemble_dir is not None:
                    try:
                        from pathlib import Path
                        ad = Path(ensemble_dir)
                        if ad.exists():
                            loader = try_load_ensemble_dir(ad)
                            if loader and getattr(loader, "status", "") == "OK":
                                self.ml_model = loader
                    except Exception:
                        pass
                if self.ml_model is None:
                    self.ml_model = load_model()
        except Exception:
            self.ml_model = None

        # last ML preds cache
        self._last_ml_pred: float = 0.0
        self._last_ml_threshold: float = 0.0

        # ---- P4: GPT Regime Monitor state ----
        self._gpt_regime_last_run: float = 0.0
        self._gpt_regime_actions: Dict[str, object] = {}

        # ---- P6: GPT What-If state ----
        self._gpt_whatif_last_run: float = 0.0

        # GPT telemetry counters (for diagnostics)
        self._gpt_attempts: int = 0
        self._gpt_recommendations: int = 0
        self._gpt_gate_take: int = 0
        self._gpt_gate_skip: int = 0
        self._gpt_gate_unknown: int = 0
        self._gpt_fallbacks: int = 0
        self._gpt_skipped_required: int = 0
        self._last_gpt_error: str = ""

    def run_enhancements_diagnostics(self) -> Dict[str, object]:
        """Run quick diagnostics for the added enhancement modules.

        This is a lightweight, import-safe check that returns information
        the UI or operator can inspect without running live trading.
        """
        out: Dict[str, object] = {}
        try:
            # Volatility forecast (uses fallback EWMA when arch not installed)
            iv_hist = list(self._iv_pct_hist or [])
            if not iv_hist:
                # synthesize small returns for a smoke-test
                iv_hist = [0.01] * 60
            vol = 0.0
            try:
                vol = float(forecast_volatility(iv_hist, horizon=1, span=20))
            except Exception:
                vol = 0.0
            out["volatility_forecast"] = vol

            # Regime detection and strategy suggestion
            atr_vals = [float(self._cached_atr) if self._cached_atr is not None else 0.01] * 30
            adx_vals = [20.0] * 30
            rsi_vals = [50.0] * 30
            regime = detect_regime(atr_vals, adx_vals, rsi_vals)
            out["regime"] = regime
            out["suggested_strategy"] = select_strategy_for_regime(regime)

            # CVaR smoke test
            out["cvar_sample"] = float(compute_cvar([0.1, -0.05, 0.02, -0.2], alpha=0.95))

            # Greeks exposure
            if self.greeks_mgr is not None:
                out["greeks"] = dict(self.greeks_mgr.portfolio_exposure().__dict__)
            else:
                out["greeks"] = None

            # Exit suggestion
            if self.exit_optimizer is not None:
                ex = self.exit_optimizer.suggest_exit({
                    "unrealized_pnl": float(getattr(self.state, "unrealized_pnl", 0.0) or 0.0),
                    "stop_loss": float(getattr(self.cfg, "stop_loss", -1e9) or -1e9),
                    "profit_target": float(getattr(self.cfg, "profit_target", 1e9) or 1e9),
                })
                out["exit_suggestion"] = ex
            else:
                out["exit_suggestion"] = None

            # ML prediction (smoke)
            try:
                if getattr(self.cfg, "enable_ml_signals", False) and self.ml_model is not None:
                    # synthesize a single feature vector: [rsi, ema_diff, atr]
                    feat = [[50.0, 0.0, float(self._cached_atr or 0.01)]]
                    probs = predict(self.ml_model, feat)
                    out["ml_signal_prob"] = float(probs[0] if probs else 0.0)
                else:
                    out["ml_signal_prob"] = None
            except Exception:
                out["ml_signal_prob"] = None

            # Walk-forward smoke test using a deterministic synthetic candle set.
            try:
                from ml_pipeline import build_supervised_dataset, walk_forward_backtest

                base = 100.0
                candles: List[Candle] = []
                now = dt_datetime.now()
                for idx in range(72):
                    drift = 0.18 * float(idx)
                    swing = math.sin(float(idx) / 4.0) * 1.5
                    close = base + drift + swing
                    candles.append(
                        Candle(
                            time=now + timedelta(minutes=idx),
                            open=close - 0.35,
                            high=close + 0.65,
                            low=close - 0.75,
                            close=close,
                            volume=1000.0 + float(idx) * 5.0,
                        )
                    )

                X, y, _feature_names = build_supervised_dataset(candles, lookback=12, horizon=1)
                if X and y:
                    wf = walk_forward_backtest(X, y, n_splits=4, threshold=0.5)
                    out["walk_forward_smoke"] = {
                        "ok": bool(wf.get("ok")),
                        "fold_count": int(float(wf.get("aggregate", {}).get("fold_count", 0.0) or 0.0)),
                        "aggregate": dict(wf.get("aggregate") or {}),
                    }
                else:
                    out["walk_forward_smoke"] = {"ok": False, "reason": "insufficient_synthetic_data"}
            except Exception as exc:
                out["walk_forward_smoke"] = {"ok": False, "error": str(exc)}
        except Exception as exc:  # pragma: no cover - diagnostics should never crash
            out["diagnostics_error"] = str(exc)
        return out

    def evaluate_entry_signals(self, candles: List[Candle]) -> Dict[str, object]:
        """Evaluate entry signals combining indicators, ML, and regime.

        Returns a dict with keys: `take` (bool), `reason`, `size` (int), `ml_prob`.
        """
        try:
            if not candles:
                LOGGER.debug("Entry evaluation skipped: no candles available")
                return {"take": False, "reason": "no_data", "size": 0, "ml_prob": 0.0}
            latest = candles[-1]
            # basic technicals
            r = rsi([c.close for c in candles], period=self.cfg.atr_period)
            atr_val = atr([c.high for c in candles], [c.low for c in candles], [c.close for c in candles], period=self.cfg.atr_period)
            self._cached_atr = float(atr_val or 0.0)
            closes = [float(c.close) for c in candles]
            fast_ema_live = float(ema(closes, period=self.cfg.ema_fast) or 0.0)
            slow_ema_live = float(ema(closes, period=self.cfg.ema_slow) or 0.0)
            trend_strength_live = float(self._trend_strength(fast_ema_live, slow_ema_live, float(latest.close or 0.0)))
            # regime
            regime = detect_regime([self._cached_atr] * 30, [20.0] * 30, [r or 50.0] * 30)
            regime_profile = get_regime_tuning(regime, self.cfg)
            self._last_regime_profile = dict(regime_profile)
            mr_signal = evaluate_mean_reversion(
                closes,
                lookback=int(getattr(self.cfg, "mean_reversion_lookback", 20) or 20),
                entry_zscore=float(getattr(self.cfg, "mean_reversion_entry_zscore", 1.25) or 1.25),
                exit_zscore=float(getattr(self.cfg, "mean_reversion_exit_zscore", 0.35) or 0.35),
            )
            fair_value_series = [float(ema(closes[: idx + 1], period=self.cfg.ema_slow) or closes[idx]) for idx in range(len(closes))]
            stat_arb_signal = evaluate_pair_spread(
                closes,
                fair_value_series,
                lookback=int(getattr(self.cfg, "stat_arb_lookback", 30) or 30),
                entry_zscore=float(getattr(self.cfg, "stat_arb_entry_zscore", 1.5) or 1.5),
                exit_zscore=float(getattr(self.cfg, "stat_arb_exit_zscore", 0.5) or 0.5),
            )
            sleeve_plan = allocate_sleeves_for_regime(
                regime,
                account_capital=float(getattr(self.cfg, "account_capital", 100000.0) or 100000.0),
                trend_score=max(0.0, trend_strength_live * 100.0),
                mean_reversion_score=float(mr_signal.entry_score),
                stat_arb_score=float(stat_arb_signal.confidence),
                max_single_sleeve_weight=float(getattr(self.cfg, "sleeve_max_single_weight", 0.50) or 0.50),
                reserve_cash_weight=float(getattr(self.cfg, "sleeve_reserve_cash_weight", 0.10) or 0.10),
                drawdown_throttle=max(0.25, 1.0 - min(1.0, float(self._current_drawdown_pct()) / 0.05)),
            )
            self._last_router_snapshot = dict(getattr(self, "_last_router_snapshot", {}) or {})
            self._last_router_snapshot["sleeve_plan"] = sleeve_plan
            suggested = str(sleeve_plan.get("selected_strategy") or select_strategy_for_regime(regime, self.cfg))
            ctx: Dict[str, Any] = {}
            ml_prob = 0.0
            if getattr(self.cfg, "enable_ml_signals", False) and self.ml_model is not None:
                LOGGER.debug("Entry evaluation starting ML scoring on %d candles", len(candles))
                current_iv = None
                try:
                    iv_watch = self._iv_watch if isinstance(getattr(self, "_iv_watch", None), dict) else {}
                    current_iv = float(iv_watch.get("value")) if iv_watch.get("value") is not None else None
                except Exception:
                    current_iv = None
                greeks = None
                try:
                    if self.greeks_mgr is not None:
                        greeks = self.greeks_mgr.portfolio_exposure()
                except Exception:
                    greeks = None
                ctx = {
                    "regime": regime,
                    "iv": current_iv,
                    "delta": float(getattr(greeks, "delta", 0.0) or 0.0) if greeks is not None else 0.0,
                    "gamma": float(getattr(greeks, "gamma", 0.0) or 0.0) if greeks is not None else 0.0,
                    "vega": float(getattr(greeks, "vega", 0.0) or 0.0) if greeks is not None else 0.0,
                    "theta": float(getattr(greeks, "theta", 0.0) or 0.0) if greeks is not None else 0.0,
                    "spot": float(latest.close or 0.0),
                    "trend_strength": float(trend_strength_live),
                    "mean_reversion_zscore": float(mr_signal.zscore),
                    "stat_arb_zscore": float(stat_arb_signal.zscore),
                }
                feat_vec, feat_names = feature_vector_from_candles(candles, context=ctx)
                self._last_ml_features = list(feat_vec)
                self._last_ml_feature_names = list(feat_names)
                live_row = {
                    name: feat_vec[idx]
                    for idx, name in enumerate(feat_names)
                    if idx < len(feat_vec)
                }
                prediction_id, ml_prob = evaluate_ml_gating_before_execution(
                    live_row,
                    self.ml_model,
                    feat_names,
                )
                LOGGER.debug(
                    "ML scoring completed prediction_id=%s prob=%.4f features=%d paper_mode=%s",
                    prediction_id,
                    float(ml_prob),
                    len(feat_names),
                    bool(getattr(self.cfg, "ml_paper_mode_enabled", False)),
                )
                self._last_prediction_id = prediction_id
                self._last_ml_pred = ml_prob
                try:
                    self._record_ml_prediction(
                        prediction_id=str(prediction_id or ""),
                        candles=candles,
                        probability=float(ml_prob),
                        threshold=0.5,
                        take=False,
                        reason="ml_scored_pending",
                        regime=str(regime or ""),
                        suggested_strategy=None,
                        feature_names=self._last_ml_feature_names,
                        feature_values=self._last_ml_features,
                        ctx=ctx,
                    )
                except Exception:
                    pass
            else:
                LOGGER.debug(
                    "Entry evaluation bypassed ML scoring enable_ml_signals=%s model_loaded=%s",
                    bool(getattr(self.cfg, "enable_ml_signals", False)),
                    self.ml_model is not None,
                )
                self._last_ml_features = []
                self._last_ml_feature_names = []

            try:
                ml_threshold = float(regime_profile.get("ml_threshold") if isinstance(regime_profile, dict) else 0.55)
            except Exception:
                ml_threshold = 0.55
            self._last_ml_threshold = float(ml_threshold)

            # Regime-specific gating: use per-regime ML thresholds before falling back.
            take = False
            reason = ""
            if getattr(self.cfg, "strategy_name", "directional") == "auto":
                # If GPT auto-selection is enabled, consult it for strategy choice.
                try:
                    if bool(getattr(self.cfg, "gpt_enable", False)) and bool(getattr(self.cfg, "gpt_auto_select", True)):
                        try:
                            gpt_rec = self._gpt_auto_select_strategy(
                                chain=[],
                                spot=float(latest.close or 0.0),
                                atr_val=float(self._cached_atr or 0.0),
                                rsi_val=float(r) if r is not None else None,
                                trend_strength=float(self._trend_strength(float(ema([c.close for c in candles], period=self.cfg.ema_fast) or 0.0), float(ema([c.close for c in candles], period=self.cfg.ema_slow) or 0.0), float(latest.close or 0.0))),
                                vwap_val=None,
                                call_atm=None,
                                put_atm=None,
                            )
                        except Exception:
                            gpt_rec = None
                        if gpt_rec:
                            suggested = str(self._normalize_strategy_name(str(gpt_rec)))
                            # GPT has supplied an explicit auto-mode choice, so treat it
                            # as a valid trade signal instead of only a branch label.
                            take = True
                            reason = "gpt_auto_select"
                        else:
                            # If GPT recommendation is required, block entry when missing.
                            if bool(getattr(self.cfg, "gpt_require_recommendation", False)):
                                return {"take": False, "reason": "gpt_missing", "size": 0, "ml_prob": float(ml_prob), "suggested_strategy": None}
                except Exception:
                    pass

                # Use ML gating as before (GPT only chooses the strategy). Fall back
                # to ML/technical checks for entry approval.
                if ml_prob >= ml_threshold:
                    take = True
                    reason = "ml_gate"
                else:
                    # fall back to technical rule: rsi oversold/overbought
                    if (r or 50.0) < 30 or (r or 50.0) > 70:
                        take = True
                        reason = "technical_fallback"
                LOGGER.debug(
                    "Auto entry gating prediction_id=%s prob=%.4f threshold=%.4f take=%s reason=%s",
                    str(getattr(self, "_last_prediction_id", "") or ""),
                    float(ml_prob),
                    float(ml_threshold),
                    bool(take),
                    str(reason or ""),
                )

                # If GPT is enabled, optionally ask GPT to approve/deny the proposed trade.
                # GPT_MARKET_COMMENTARY_ENABLED (default False) gates all GPT commentary calls.
                # When False, GPT 402 errors cannot block ML trading since no GPT calls are made.
                try:
                    if (bool(getattr(self.cfg, "gpt_enable", False))
                        and bool(getattr(self.cfg, "gpt_auto_select", True))
                        and bool(getattr(self.cfg, "gpt_market_commentary_enabled", False))):
                        # AUTO mode should still give GPT a chance to originate a trade.
                        # Otherwise GPT can only veto trades that a local gate already accepted.
                        consult_gpt = bool(take) or bool(getattr(self.cfg, "gpt_require_recommendation", False))
                        if str(getattr(self.cfg, "strategy_name", "directional") or "").strip().lower() == "auto":
                            consult_gpt = True
                        if consult_gpt:
                            try:
                                import gpt_advisor as _ga  # type: ignore
                                advice_model = os.getenv("MSTOCK_GPT_MODEL", "") or "gpt-4o-mini"
                                advice_key = (os.getenv("MSTOCK_GPT_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
                                # Enrich proposal with contextual market data to help GPT make a better judgment.
                                proposal = {
                                    "strategy": suggested,
                                    "ml_prob": float(ml_prob),
                                    "size": int(size or 0),
                                    "reason": reason,
                                    "latest_close": float(latest.close or 0.0),
                                    "timestamp": int(time.time()),
                                }
                                try:
                                    # Recent candles (last 5) simplified as dicts
                                    recent = []
                                    for c in (candles[-5:] if len(candles) >= 1 else candles):
                                        recent.append({
                                            "time": int(getattr(c, "time", time.time())),
                                            "open": float(getattr(c, "open", 0.0) or 0.0),
                                            "high": float(getattr(c, "high", 0.0) or 0.0),
                                            "low": float(getattr(c, "low", 0.0) or 0.0),
                                            "close": float(getattr(c, "close", 0.0) or 0.0),
                                        })
                                    proposal["recent_candles"] = recent
                                except Exception:
                                    pass
                                try:
                                    proposal["atr"] = float(self._cached_atr or 0.0)
                                except Exception:
                                    proposal["atr"] = None
                                try:
                                    iv_val = None
                                    if isinstance(getattr(self, "_iv_watch", None), dict):
                                        iv_val = self._iv_watch.get("value")
                                    proposal["iv"] = float(iv_val) if iv_val is not None else None
                                except Exception:
                                    proposal["iv"] = None
                                try:
                                    if getattr(self, "greeks_mgr", None) is not None:
                                        g = self.greeks_mgr.portfolio_exposure()
                                        proposal["greeks"] = {
                                            "delta": float(getattr(g, "delta", 0.0) or 0.0),
                                            "gamma": float(getattr(g, "gamma", 0.0) or 0.0),
                                            "vega": float(getattr(g, "vega", 0.0) or 0.0),
                                            "theta": float(getattr(g, "theta", 0.0) or 0.0),
                                        }
                                except Exception:
                                    proposal["greeks"] = None
                                try:
                                    # Small option chain sample near ATM (best-effort, limited size)
                                    chain_sample = []
                                    chain = []
                                    try:
                                        chain = list(self.client.get_option_chain(self.cfg.underlying) or [])
                                    except Exception:
                                        chain = []
                                    if chain:
                                        # Attempt to pick nearest strikes around ATM by strike if present.
                                        # Normalise to list of dicts with symbol/strike/option_type/iv
                                        simplified = []
                                        for row in chain:
                                            try:
                                                simplified.append({
                                                    "symbol": str(row.get("symbol") or ""),
                                                    "strike": float(row.get("strike") or 0.0),
                                                    "option_type": str(row.get("option_type") or ""),
                                                    "iv": float(row.get("iv") or 0.0) if row.get("iv") not in (None, "") else None,
                                                })
                                            except Exception:
                                                continue
                                        # Sort by strike and pick the middle 8 if possible
                                        try:
                                            simplified.sort(key=lambda x: float(x.get("strike") or 0.0))
                                            mid = len(simplified) // 2
                                            start = max(0, mid - 4)
                                            chain_sample = simplified[start : start + 8]
                                        except Exception:
                                            chain_sample = simplified[:8]
                                    proposal["option_chain_sample"] = chain_sample
                                except Exception:
                                    proposal["option_chain_sample"] = []
                                parsed = _ga.advise_trade(proposal=proposal, model=advice_model, api_key=advice_key)
                            except Exception:
                                parsed = None
                            # parsed is a GPTAdvice dataclass-like with .decision in {TAKE, SKIP, UNKNOWN}
                            if parsed is not None:
                                try:
                                    decision = str(getattr(parsed, "decision", "UNKNOWN") or "UNKNOWN").upper()
                                except Exception:
                                    decision = "UNKNOWN"
                                # Record recent GPT advice for UI enrichment of the next emitted event.
                                try:
                                    self._pending_gpt_advice = {
                                        "ts": time.time(),
                                        "decision": decision,
                                        "reason": str(getattr(parsed, "reason", "") or ""),
                                        "confidence": float(getattr(parsed, "confidence", 0.0) or 0.0),
                                    }
                                except Exception:
                                    try:
                                        self._pending_gpt_advice = {"ts": time.time(), "decision": decision}
                                    except Exception:
                                        self._pending_gpt_advice = None
                                if decision == "SKIP":
                                    take = False
                                    reason = f"gpt_skip:{getattr(parsed, 'reason', '') or ''}"
                                elif decision == "TAKE":
                                    take = True
                                    reason = f"gpt_take:{getattr(parsed, 'reason', '') or ''}"
                                else:
                                    # UNKNOWN: if GPT recommendation is required, block; else keep prior decision.
                                    # HTTP 402 from GPT provider MUST NOT block ML trading. Treat as non-blocking
                                    # (log warning, preserve prior take/reason decision).
                                    gpt_reason = str(getattr(parsed, "reason", "") or "").strip().lower()
                                    if gpt_reason.startswith("http 402:"):
                                        try:
                                            print(f"[GPT ADVISOR] HTTP 402 received — GPT commentary unavailable, continuing without it")
                                        except Exception:
                                            pass
                                        # HTTP 402 must not set prediction probability to 0 or block shadow/paper.
                                        # Keep the prior `take` and `reason` (ml_gate / technical_fallback) intact.
                                    elif bool(getattr(self.cfg, "gpt_require_recommendation", False)):
                                        take = False
                                        reason = f"gpt_unknown:{getattr(parsed, 'reason', '') or ''}"
                            else:
                                # No parsed advice (API failure): if GPT required, block; else continue
                                if bool(getattr(self.cfg, "gpt_require_recommendation", False)):
                                    take = False
                                    reason = "gpt_missing_api"
                        # end consult_gpt
                except Exception:
                    pass
            else:
                # non-auto: use ml as supplement
                if ml_prob >= 0.6:
                    take = True
                    reason = "ml_supplement"

            # position sizing using volatility targeting
            size = 0
            baseline_vol_lots = 0
            ml_bet_multiplier = 0.0
            final_allocated_lots = 0
            if take:
                if getattr(self.cfg, "risk_per_trade_percentage", 0.0) > 0:
                    cash = float(getattr(self.cfg, "account_capital", 100000.0) or 100000.0)
                    vol_target = float(getattr(self.cfg, "risk_per_trade_percentage", 0.0) or 0.0)
                    baseline_vol_lots = volatility_target_size(cash, vol_target, float(self._cached_atr or 0.01), float(latest.close or 0.0))
                else:
                    # default small size
                    baseline_vol_lots = max(1, int(self.cfg.lot_size // 1))

                sizing = calculate_position_size(
                    baseline_lots=baseline_vol_lots,
                    calibrated_prob=float(ml_prob),
                    optimal_threshold=float(ml_threshold),
                    standard_error=0.10,
                    current_drawdown_pct=self._current_drawdown_pct(),
                    max_allowable_drawdown=0.05,
                    enforce_min_lot=True,
                )
                ml_bet_multiplier = float(sizing.get("ml_bet_multiplier", 0.0) or 0.0)
                final_allocated_lots = int(sizing.get("final_allocated_lots", 0) or 0)
                try:
                    router_size_mult = max(0.25, min(1.5, float(self._router_size_multiplier())))
                except Exception:
                    router_size_mult = 1.0
                if final_allocated_lots > 0 and abs(router_size_mult - 1.0) > 1e-9:
                    step_qty = int(getattr(self.cfg, "lot_size", 1) or 1)
                    if step_qty <= 0:
                        step_qty = 1
                    scaled_lots = int(round(float(final_allocated_lots) * float(router_size_mult)))
                    scaled_lots = max(step_qty, int(round(float(scaled_lots) / float(step_qty))) * step_qty)
                    final_allocated_lots = int(scaled_lots)
                size = final_allocated_lots

            self._last_ml_bet_multiplier = float(ml_bet_multiplier)
            self._last_baseline_vol_lots = int(baseline_vol_lots)
            self._last_final_allocated_lots = int(final_allocated_lots)

            if getattr(self.cfg, "enable_ml_signals", False) and self.ml_model is not None and self._last_ml_feature_names:
                try:
                    sizing_ctx = dict(ctx or {})
                    sizing_ctx.update(
                        {
                            "ml_bet_multiplier": float(self._last_ml_bet_multiplier),
                            "baseline_vol_lots": int(self._last_baseline_vol_lots),
                            "final_allocated_lots": int(self._last_final_allocated_lots),
                            "current_drawdown_pct": float(self._current_drawdown_pct()),
                        }
                    )
                    self._record_ml_prediction(
                        prediction_id=str(getattr(self, "_last_prediction_id", "") or ""),
                        candles=candles,
                        probability=float(ml_prob),
                        threshold=float(ml_threshold),
                        take=bool(take),
                        reason=str(reason or ""),
                        regime=str(regime or ""),
                        suggested_strategy=str(suggested or ""),
                        feature_names=self._last_ml_feature_names,
                        feature_values=self._last_ml_features,
                        ctx=sizing_ctx,
                    )
                except Exception:
                    pass

            LOGGER.debug(
                "Entry evaluation finalized prediction_id=%s take=%s size=%d ml_prob=%.4f multiplier=%.4f baseline_lots=%d final_lots=%d reason=%s",
                str(getattr(self, "_last_prediction_id", "") or ""),
                bool(take),
                int(size),
                float(ml_prob),
                float(self._last_ml_bet_multiplier),
                int(self._last_baseline_vol_lots),
                int(self._last_final_allocated_lots),
                str(reason or ""),
            )

            return {
                "take": take,
                "reason": reason,
                "size": int(size),
                "ml_prob": float(ml_prob),
                "ml_bet_multiplier": float(self._last_ml_bet_multiplier),
                "baseline_vol_lots": int(self._last_baseline_vol_lots),
                "final_allocated_lots": int(self._last_final_allocated_lots),
                "suggested_strategy": suggested,
                "prediction_id": getattr(self, "_last_prediction_id", None),
                "mean_reversion_signal": mr_signal.signal,
                "stat_arb_signal": stat_arb_signal.signal,
                "sleeve_plan": sleeve_plan,
            }
        except Exception as exc:
            return {"take": False, "reason": f"error:{exc}", "size": 0, "ml_prob": 0.0}

    def _simulate_or_place_basic_directional(self, *, take: bool, size: int, reason: str, candles: List[Candle]) -> Optional[Dict[str, object]]:
        """Simulate (paper) or place a very basic directional entry using latest close.

        This is a conservative integration point for ML-backed signals. It does
        not attempt to compute option strikes or build multi-leg structures.
        Instead it creates a lightweight directional trade record for the
        UI/backtester and — when `enable_live_trading` is True — places a
        market order for the underlying/hedge symbol if the client supports it.
        """
        try:
            if not take or size <= 0:
                return None
            latest = candles[-1] if candles else None
            spot = float(latest.close) if latest is not None else 0.0
            if spot <= 0:
                return None

            # Calculate ATM strike for Nifty options (50 point interval)
            strike_interval = 50
            atm_strike = round(spot / strike_interval) * strike_interval

            # Get expiry string (format: DDMMM, e.g., 22MAY)
            expiry = getattr(self.cfg, "expiry", None)
            if not expiry:
                # Try MSTOCK_TARGET_EXPIRY first
                target_expiry_raw = str(getattr(self.cfg, "target_expiry", "") or "").strip()
                if target_expiry_raw:
                    try:
                        parsed_date = self._parse_any_date(target_expiry_raw)
                        if parsed_date:
                            expiry = parsed_date.strftime("%d%b").upper()
                    except Exception:
                        pass
            if not expiry:
                # Try getting it from nearest weekly expiry in the option chain
                try:
                    chain = self.get_option_chain(self.cfg.underlying)
                    nearest = self._filter_nearest_weekly_expiry(chain)
                    if nearest:
                        exp_date = self._row_expiry_date(nearest[0])
                        if exp_date:
                            expiry = exp_date.strftime("%d%b").upper()
                except Exception:
                    pass
            if not expiry:
                from datetime import datetime
                now = datetime.now()
                expiry = now.strftime("%d%b").upper()

            # Determine option type (CE for bullish, PE for bearish)
            option_type = "CE"  # Default to Call for bullish directional
            try:
                closes = [c.close for c in candles]
                ema_fast_val = ema(closes, period=int(getattr(self.cfg, "ema_fast", 9)))
                ema_slow_val = ema(closes, period=int(getattr(self.cfg, "ema_slow", 21)))
                if ema_fast_val is not None and ema_slow_val is not None:
                    if ema_fast_val < ema_slow_val:
                        option_type = "PE"
            except Exception:
                pass

            # Override via reason search if specified
            r_lower = str(reason or "").lower()
            if "bear" in r_lower or "put" in r_lower or "short" in r_lower or "sell" in r_lower:
                option_type = "PE"
            elif "bull" in r_lower or "call" in r_lower or "long" in r_lower or "buy" in r_lower:
                option_type = "CE"

            option_symbol = f"{self.cfg.underlying}{expiry}{atm_strike}{option_type}"
            contract = self._option_contract_from_row(
                {
                    "symbol": option_symbol,
                    "exchange": "NFO",
                    "token": "",
                    "expiry": expiry,
                    "strike": atm_strike,
                    "option_type": option_type,
                }
            )
            if contract is None or not str(contract.instrument_token or "").isdigit():
                self._entry_block_for_missing_contract_token(row={"symbol": option_symbol}, context="auto_ml_directional")
                return None
            resolved_symbol = contract.tradingsymbol
            resolved_token = contract.instrument_token
            resolved_exchange = contract.exchange
            print(
                f"[CONTRACT] selected tradingsymbol={resolved_symbol} "
                f"token={resolved_token} exchange={resolved_exchange}"
            )

            # Parse resolved expiry string back to datetime.date object for greeks/portfolio risk snapshot
            parsed_exp = None
            try:
                from datetime import datetime
                # expiry is a string in DDMMM format like '25MAY'
                # Use current year for parsing
                parsed_exp = datetime.strptime(f"{expiry}{datetime.now().year}", "%d%b%Y").date()
            except Exception:
                try:
                    # fallback
                    parsed_exp = self._parse_any_date(expiry)
                except Exception:
                    from datetime import date
                    parsed_exp = date.today()

            # Fetch option contract LTP
            try:
                option_ltp = self.client.get_ltp(f"{resolved_exchange}:{resolved_token}")
                if option_ltp is None:
                    print(f"[ERROR] No LTP available for option {resolved_symbol}")
                    return None
                price = float(option_ltp)
            except Exception as e:
                print(f"[ERROR] Failed to fetch LTP for {resolved_symbol}: {e}")
                return None
            name = "auto_ml_directional"
            target_symbol = resolved_symbol if not bool(getattr(self.cfg, "enable_live_trading", False)) else str(getattr(self.cfg, "delta_hedge_symbol", "") or self.cfg.underlying)
            friction_preview = evaluate_live_execution_friction(
                {
                    "client": self.client,
                    "symbol": resolved_symbol,
                    "exchange": resolved_exchange,
                    "cost_model": CostModel(),
                    "ltp": price,
                    "enforce_cost_boundary": False,
                }
            )
            if str(friction_preview.get("status") or "") == "REJECT_LIQUIDITY_CEILING":
                print(f"[ENTRY BLOCKED] {friction_preview.get('status')}: {friction_preview.get('reason')}")
                return None

            # Pyramiding: check for existing open position first
            existing = None
            max_pyr = 0
            try:
                max_pyr = self._dynamic_pyramid_max_level()
            except Exception:
                max_pyr = 0

            if max_pyr > 0:
                for t in self.state.open_directional:
                    if t.get("name") == name:
                        t_sym = str(t.get("symbol") or "").strip().upper()
                        new_sym = str(target_symbol).strip().upper()
                        if t_sym == new_sym:
                            existing = t
                            break
                if existing is not None:
                    try:
                        lev = int(existing.get("pyramid_level", 0) or 0)
                    except Exception:
                        lev = 0
                    if lev >= max_pyr:
                        print(f"[PYRAMID] Max pyramiding reached for {name} (level={lev}, max={max_pyr}); skipping add")
                        return existing

            if existing is not None:
                lev = int(existing.get("pyramid_level", 0))
                existing["pyramid_level"] = lev + 1
                existing["quantity"] = int(existing.get("quantity", 0)) + int(size)
                legs_existing = existing.get("legs")
                if isinstance(legs_existing, list) and legs_existing and isinstance(legs_existing[0], dict):
                    legs_existing[0]["quantity"] = int(existing.get("quantity", 0))

                # Count this add as a trade for daily limits.
                self.state.trades_today += 1
                self.state.last_entry_ts = time.time()
                try:
                    self._note_opened_trade_type(position_type="directional", name=str(name))
                except Exception:
                    pass

                # If live trading, place the order
                if bool(getattr(self.cfg, "enable_live_trading", False)):
                    try:
                        hedge_sym = str(getattr(self.cfg, "delta_hedge_symbol", "") or self.cfg.underlying)
                        print(f"[LIVE][AUTO][PYRAMID] placing market BUY {hedge_sym} x{size} as underlying proxy for {name}")
                        self.client.place_order(symbol=hedge_sym, side="BUY", quantity=int(size), order_type="MARKET")
                    except Exception as exc:
                        print(f"[LIVE][AUTO][PYRAMID] failed to place underlying order: {exc}")
                        # Rollback qty on failure
                        existing["quantity"] = int(existing.get("quantity", 0)) - int(size)
                        if isinstance(legs_existing, list) and legs_existing and isinstance(legs_existing[0], dict):
                            legs_existing[0]["quantity"] = int(existing.get("quantity", 0))
                        return None
                else:
                    print(f"[PAPER][AUTO][PYRAMID] {name}: added size={size} @ price={price}")

                if self._has_event_sink():
                    try:
                        trade_id_existing = str(existing.get("trade_id") or "(unknown)")
                        name_existing = str(existing.get("name") or name)
                        legs_evt = [dict(l) for l in legs_existing if isinstance(l, dict)] if isinstance(legs_existing, list) else []
                        self._emit(
                            TradeLogEvent(
                                ts=time.time(),
                                event="UPDATE",
                                trade_id=trade_id_existing,
                                position_type="directional",
                                name=name_existing,
                                legs=legs_evt,
                                mtm=self._compute_legs_mtm(legs_evt),
                            )
                        )
                    except Exception:
                        pass

                print(f"[PYRAMID] Incremented {name} level to {lev + 1}")
                return existing

            # Paper mode: just print and append a minimal trade record
            if not bool(getattr(self.cfg, "enable_live_trading", False)):
                # ----------------------------------------------------------------
                # Paper BUY entry: use ask price from broker, not LTP.
                # Block if bid/ask is unavailable (no silent LTP fallback).
                # ----------------------------------------------------------------
                bid_raw, ask_raw, _ = self.client.get_bid_ask(
                    resolved_symbol,
                    exchange_hint=resolved_exchange,
                )
                if ask_raw is None:
                    print(f"[PAPER][BLOCKED] {name}: ask price unavailable for {resolved_symbol} — bid/ask required for paper realism")
                    if self._has_event_sink():
                        try:
                            self._emit(TradeLogEvent(
                                ts=time.time(),
                                event="SKIP",
                                trade_id=self._new_trade_id("D"),
                                position_type="directional",
                                name=str(name),
                                legs=[],
                                skip_reason="ask_price_unavailable",
                                symbol=self.cfg.underlying,
                                strike=float(atm_strike),
                                option_type=option_type,
                                side="BUY",
                                quantity=int(size),
                                bid=float(bid_raw) if bid_raw is not None else None,
                                ask=None,
                                ltp=float(price),
                                execution_price=None,
                                execution_price_source="missing_bid_ask",
                                filter_bid_ask_ok=False,
                                paper_mode=True,
                            ))
                        except Exception:
                            pass
                    return None
                paper_entry_price = float(ask_raw)
                paper_spread_cost = 0.0
                try:
                    bid_f = float(bid_raw) if bid_raw is not None else None
                    if bid_f is not None and paper_entry_price > bid_f:
                        paper_spread_cost = float((paper_entry_price - bid_f) * float(size))
                except Exception:
                    pass
                # Model slippage + brokerage costs using the paper cost model.
                # Use cfg.paper_slippage_pct (default 0.1%) as the primary slippage input.
                # The spread_cost is logged but NOT double-deducted — it is embedded in the ask price.
                try:
                    from cost_model import estimate_paper_execution_costs
                    _entry_costs = estimate_paper_execution_costs(
                        execution_price=paper_entry_price,
                        quantity=int(size),
                        slippage_pct=float(getattr(self.cfg, "paper_slippage_pct", 0.001) or 0.001),
                        extra_market_impact_pct=float(getattr(self.cfg, "paper_extra_market_impact_pct", 0.0) or 0.0),
                        apply_brokerage=bool(getattr(self.cfg, "paper_apply_brokerage_costs", True)),
                        cost_model_source=str(getattr(self.cfg, "paper_cost_model_source", "cost_model_assumptions") or "cost_model_assumptions"),
                        side="BUY",
                    )
                    _entry_slippage_cost = float(_entry_costs.get("slippage_cost", 0.0))
                    _entry_brokerage_cost = float(_entry_costs.get("total_brokerage_charges", 0.0))
                    _entry_total_cost = float(_entry_costs.get("total_cost", 0.0))
                    _entry_slippage_pct = float(getattr(self.cfg, "paper_slippage_pct", 0.001) or 0.001)
                except Exception:
                    _entry_costs = {}
                    _entry_slippage_cost = 0.0
                    _entry_brokerage_cost = 0.0
                    _entry_total_cost = 0.0
                    _entry_slippage_pct = 0.0

                # Enforce premium and liquidity filters before accepting paper entry.
                paper_legs = [
                    {
                        "symbol": resolved_symbol,
                        "token": resolved_token,
                        "exchange": resolved_exchange,
                        "side": "BUY",
                        "quantity": int(size),
                        "entry_price": paper_entry_price,
                    }
                ]
                # Compute spread_pct_at_entry from bid/ask mid
                _spread_pct_entry = 0.0
                try:
                    _mid = (float(bid_raw) + paper_entry_price) / 2.0 if bid_raw is not None and float(bid_raw) > 0 else paper_entry_price
                    if _mid > 0:
                        _spread_pct_entry = abs(paper_entry_price - float(bid_raw)) / _mid * 100.0
                except Exception:
                    pass

                ok, filter_reason = self._check_entry_leg_premiums(paper_legs)
                if not ok:
                    print(f"[PAPER][BLOCKED] {name}: premium filter rejected — {filter_reason}")
                    if self._has_event_sink():
                        try:
                            self._emit(TradeLogEvent(
                                ts=time.time(),
                                event="SKIP",
                                trade_id=self._new_trade_id("D"),
                                position_type="directional",
                                name=str(name),
                                legs=[],
                                skip_reason=str(filter_reason or "premium_filter_rejected"),
                                symbol=self.cfg.underlying,
                                strike=float(atm_strike),
                                option_type=option_type,
                                side="BUY",
                                quantity=int(size),
                                bid=float(bid_raw) if bid_raw is not None else None,
                                ask=float(ask_raw) if ask_raw is not None else None,
                                ltp=float(price),
                                execution_price=float(paper_entry_price),
                                execution_price_source="ask",
                                filter_premium_ok=False,
                                paper_mode=True,
                            ))
                        except Exception:
                            pass
                    return None

                print(f"[PAPER][AUTO] {name}: side=BUY size={size} price={paper_entry_price} (ask) reason={reason}")
                prediction_id = str(getattr(self, "_last_prediction_id", "") or "")
                meta: Dict[str, Any] = {
                    "simulated": True,
                    "reason": reason,
                    "prediction_id": prediction_id,
                    "ml_probability": float(getattr(self, "_last_ml_pred", 0.0) or 0.0),
                    "ml_threshold": float(getattr(self, "_last_ml_threshold", 0.0) or 0.0),
                    "ml_bet_multiplier": float(getattr(self, "_last_ml_bet_multiplier", 0.0) or 0.0),
                    "baseline_vol_lots": int(getattr(self, "_last_baseline_vol_lots", 0) or 0),
                    "final_allocated_lots": int(getattr(self, "_last_final_allocated_lots", size) or size),
                    "live_bid_ask_spread_pct": float(friction_preview.get("relative_spread_pct") or 0.0),
                    "order_routing_style": "midpoint_pegged_limit",
                    "spread_cost_rupees": float(paper_spread_cost),  # logged (embedded in ask price; NOT double-deducted)
                    "slippage_cost_rupees": float(_entry_slippage_cost),
                    "brokerage_cost_rupees": float(_entry_brokerage_cost),
                    "total_entry_cost": float(_entry_total_cost),
                    "realized_slippage_pct": float(_entry_slippage_pct),
                    "execution_price_source": "ask",
                    "spread_pct_at_entry": float(_spread_pct_entry),
                    "execution_friction_status": str(friction_preview.get("status") or ""),
                }
                meta.update(self._trade_risk_meta_overrides())
                if prediction_id:
                    try:
                        meta["triple_barrier"] = self._build_triple_barrier_metadata(
                            fill_price=paper_entry_price,
                            prediction_id=prediction_id,
                            bar_timestamp=getattr(latest, "time", None),
                        )
                    except Exception:
                        pass
                tr = submit_paper_trade_order(
                    {
                    "trade_id": self._new_trade_id("D"),
                    "name": name,
                    "symbol": resolved_symbol,
                    "token": resolved_token,
                    "exchange": resolved_exchange,
                    "side": "BUY",
                    "quantity": int(size),
                    "entry_spot": float(spot),
                    "entry_price": float(paper_entry_price),
                    "atr": float(self._cached_atr or 0.0),
                    "legs": [
                        {
                            "symbol": resolved_symbol,
                            "token": resolved_token,
                            "exchange": resolved_exchange,
                            "quantity": int(size),
                            "side": "BUY",
                            "entry_price": float(paper_entry_price),
                            "strike": float(atm_strike),
                            "option_type": option_type,
                            "expiry": parsed_exp,
                        }
                    ],
                    "meta": meta,
                    "opened_ts": time.time(),
                    "entry_time": dt_datetime.now(IST) if IST else dt_datetime.now(),
                    },
                    prediction_id,
                    float(getattr(self, "_last_ml_pred", 0.0) or 0.0),
                )
                if prediction_id and self.db_manager is not None:
                    try:
                        self.db_manager.link_prediction_to_trade(prediction_id, str(tr["trade_id"]))
                    except Exception:
                        pass
                self.state.open_directional.append(tr)
                self.state.trades_today += 1
                self.state.last_entry_ts = time.time()
                return tr

            # Live trading: attempt to place a market order for an available hedge/underlying symbol.
            # This is intentionally conservative: it places an underlying instrument order
            # rather than complex options legs. Users should enable live only after review.
            try:
                hedge_sym = str(getattr(self.cfg, "delta_hedge_symbol", "") or self.cfg.underlying)
                print(f"[LIVE][AUTO] placing market BUY {hedge_sym} x{size} as underlying proxy for {name}")
                route_meta: Dict[str, Any] = {}
                route_result = self._route_midpoint_pegged_limit_order(
                    symbol=resolved_symbol,
                    side="BUY",
                    quantity=int(size),
                    exchange=resolved_exchange or None,
                    symbol_token=resolved_token or None,
                    meta=route_meta,
                )
                if not bool(route_result.get("ok")):
                    print(f"[LIVE][AUTO] midpoint route blocked: {route_result.get('status')} {route_result.get('reason')}")
                    return None
            except Exception as exc:
                print(f"[LIVE][AUTO] failed to place underlying order: {exc}")
                return None

            prediction_id = str(getattr(self, "_last_prediction_id", "") or "")
            meta_live: Dict[str, Any] = {
                "simulated": False,
                "reason": reason,
                "prediction_id": prediction_id,
                "ml_probability": float(getattr(self, "_last_ml_pred", 0.0) or 0.0),
                "ml_threshold": float(getattr(self, "_last_ml_threshold", 0.0) or 0.0),
                "ml_bet_multiplier": float(getattr(self, "_last_ml_bet_multiplier", 0.0) or 0.0),
                "baseline_vol_lots": int(getattr(self, "_last_baseline_vol_lots", 0) or 0),
                "final_allocated_lots": int(getattr(self, "_last_final_allocated_lots", size) or size),
                "live_bid_ask_spread_pct": float(route_meta.get("live_bid_ask_spread_pct") or 0.0),
                "order_routing_style": "midpoint_pegged_limit",
                "realized_slippage_pct": float(route_meta.get("realized_slippage_pct") or 0.0),
                "execution_friction_status": str(route_meta.get("execution_friction_status") or ""),
            }
            meta_live.update(self._trade_risk_meta_overrides())
            if prediction_id and price:
                try:
                    meta_live["triple_barrier"] = self._build_triple_barrier_metadata(
                        fill_price=float(route_result.get("fill_price") or price),
                        prediction_id=prediction_id,
                        bar_timestamp=getattr(latest, "time", None),
                    )
                except Exception:
                    pass
            tr_live = submit_paper_trade_order(
                {
                "trade_id": self._new_trade_id("D"),
                "name": name,
                "symbol": hedge_sym,
                "side": "BUY",
                "quantity": int(size),
                "entry_spot": float(price),
                "entry_price": float(route_result.get("fill_price") or price),
                "atr": float(self._cached_atr or 0.0),
                "legs": [
                    {
                        "symbol": hedge_sym,
                        "quantity": int(size),
                        "side": "BUY",
                        "entry_price": float(route_result.get("fill_price") or price),
                        "strike": float(atm_strike),
                        "option_type": option_type,
                        "expiry": parsed_exp,
                    }
                ],
                "meta": meta_live,
                "opened_ts": time.time(),
                "entry_time": dt_datetime.now(IST) if IST else dt_datetime.now(),
                },
                prediction_id,
                float(getattr(self, "_last_ml_pred", 0.0) or 0.0),
            )
            if prediction_id and self.db_manager is not None:
                try:
                    self.db_manager.link_prediction_to_trade(prediction_id, str(tr_live["trade_id"]))
                except Exception:
                    pass
            self.state.open_directional.append(tr_live)
            self.state.trades_today += 1
            self.state.last_entry_ts = time.time()
            return tr_live
        except Exception:
            return None

    def _normalize_strategy_name(self, name: str) -> str:
        """Map UI-facing strategy names to implemented strategy branches.

        This lets the UI expose common naming variants without breaking entries.
        """

        s = str(name or "").strip().lower()
        if not s:
            return "directional"

        supported = {
            # top-level modes
            "auto",
            "directional",
            # premium / multi-leg
            "short_straddle",
            "short_strangle",
            "bull_call_spread",
            "bull_put_spread",
            "call_ratio_backspread",
            "put_ratio_backspread",
            "iron_condor",
            "iron_fly",
            "iron_butterfly",
            "long_straddle",
            "long_strangle",
            "delta_hedged_short_straddle",
            "delta_hedged_long_straddle",
            # directional explicit variants
            "long_call",
            "long_put",
            "short_call",
            "short_put",
        }

        aliases = {
            # Volatility strategy naming
            # NOTE: short_premium/long_premium are deprecated umbrella labels.
            # Map them (and common synonyms) to explicit implemented structures.
            "short_premium": "short_straddle",
            "long_premium": "long_strangle",
            "short_volatility_strategy": "short_strangle",
            "long_volatility_strategy": "long_strangle",
            "short_volatility": "short_strangle",
            "long_volatility": "long_strangle",
            "sell_vol": "short_strangle",
            "buy_vol": "long_strangle",

            # Premium structure naming variants
            "short_straddle_weekly": "short_straddle",
            "short_strangle_weekly": "short_strangle",
            "long_straddle_weekly": "long_straddle",
            "long_strangle_weekly": "long_strangle",
            # Condor variants
            "condor_call": "iron_condor",
            "condor_put": "iron_condor",
            # Butterfly / fly naming
            "iron_fly": "iron_fly",
            "iron_butterfly": "iron_butterfly",
            "butterfly_spread_call": "iron_butterfly",
            "butterfly_spread_put": "iron_butterfly",

            # Bull spreads / ratio backspreads
            "bull_call": "bull_call_spread",
            "bull_put": "bull_put_spread",
            "bull_call_spread": "bull_call_spread",
            "bull_put_spread": "bull_put_spread",
            "call_ratio_backspread": "call_ratio_backspread",
            "put_ratio_backspread": "put_ratio_backspread",
            "call_backspread": "call_ratio_backspread",
            "put_backspread": "put_ratio_backspread",

            # Calendar/diagonal naming (mapped to long premium as a safe default)
            "calendar_spread_call": "long_strangle",
            "calendar_spread_put": "long_strangle",
            "double_calendar_spread": "long_strangle",
            "diagonal_spread": "long_strangle",

            # Simple directional naming
            "covered_call": "short_call",
            "protective_put": "long_put",
            "long_call": "long_call",
            "long_put": "long_put",
            "short_call": "short_call",
            "short_put": "short_put",
        }

        mapped = str(aliases.get(s, s))
        if mapped in supported:
            return mapped

        # Heuristic fallback: ensure *every* strategy string leads to an executable branch.
        # This intentionally prefers safe, already-implemented templates.
        t = mapped
        try:
            # Normalize separators
            t = t.replace("-", "_").replace(" ", "_")
        except Exception:
            pass

        # Long vol / debit-ish structures
        if any(k in t for k in ("calendar", "diagonal", "backspread", "ratio_backspread", "long_vol", "debit")):
            return "long_strangle"
        # Short vol / credit-ish structures
        if any(k in t for k in ("condor", "fly", "butterfly", "lizard", "guts", "ladder", "short_vol", "credit")):
            return "short_strangle"
        # Directional / single-leg hints
        if any(k in t for k in ("call", "put", "directional")):
            return "directional"
        # Default conservative fallback
        return "directional"

    def _ltp_key(self, symbol: str) -> str:
        s = str(symbol or "").strip().upper()
        if not s:
            return ""
        if ":" in s:
            s = s.split(":", 1)[1].strip()
        return s

    def _note_ltp_seen(self, symbol_key: str, ltp: float) -> None:
        k = str(symbol_key or "").strip().upper()
        if not k:
            return
        now_ts = float(time.time())
        prev = self._ltp_watch.get(k) or {}
        prev_ltp = prev.get("ltp")
        change_ts = float(prev.get("change_ts") or now_ts)
        try:
            if prev_ltp is None or abs(float(ltp) - float(prev_ltp)) > 1e-9:
                change_ts = now_ts
        except Exception:
            change_ts = now_ts
        self._ltp_watch[k] = {
            "ltp": float(ltp),
            "seen_ts": now_ts,
            "change_ts": change_ts,
        }

    def _seconds_since_ltp_change(self, symbol: str) -> Optional[float]:
        k = self._ltp_key(symbol)
        if not k:
            return None
        row = self._ltp_watch.get(k)
        if not row:
            return None
        try:
            # Prefer "seen_ts" for freshness; fall back to "change_ts" if missing.
            ts = row.get("seen_ts") or row.get("change_ts") or 0.0
            return float(time.time() - float(ts))
        except Exception:
            return None

    def _bool_env(self, name: str, default: bool) -> bool:
        v = os.getenv(name)
        if v is None:
            return bool(default)
        return str(v).strip().lower() in {"1", "true", "yes", "y"}

    def _kill_switch_active(self) -> bool:
        """Check external kill-switch env vars at runtime.

        Returns True if either MSTOCK_KILL_SWITCH or SCALPER_KILL_SWITCH
        is set to a truthy value, which blocks all new entries and live orders.
        """
        if self._bool_env("MSTOCK_KILL_SWITCH", False):
            LOGGER.warning("[KILL_SWITCH] MSTOCK_KILL_SWITCH is active — blocking all entries and live orders")
            return True
        if self._bool_env("SCALPER_KILL_SWITCH", False):
            LOGGER.warning("[KILL_SWITCH] SCALPER_KILL_SWITCH is active — blocking all entries and live orders")
            return True
        return False

    def _debug_delta_hedge(self) -> bool:
        return self._bool_env("MSTOCK_DEBUG_DELTA_HEDGE", False)

    def _compute_beta(self, asset_returns: List[float], index_returns: List[float]) -> Optional[float]:
        """Compute beta = cov(asset, index) / var(index)."""

        if not asset_returns or not index_returns:
            return None
        n = min(len(asset_returns), len(index_returns))
        if n < 10:
            return None
        a = asset_returns[-n:]
        m = index_returns[-n:]
        mean_a = sum(a) / float(n)
        mean_m = sum(m) / float(n)
        var_m = sum((x - mean_m) ** 2 for x in m)
        if var_m <= 1e-18:
            return None
        cov = sum((a[i] - mean_a) * (m[i] - mean_m) for i in range(n))
        beta = cov / var_m
        # Clamp extreme betas to keep hedging stable.
        if beta > 5.0:
            beta = 5.0
        if beta < -5.0:
            beta = -5.0
        return float(beta)

    def _daily_returns_from_candles(self, candles: List[Candle]) -> List[float]:
        if not candles or len(candles) < 2:
            return []
        closes: List[float] = []
        for c in candles:
            try:
                closes.append(float(c.close))
            except Exception:
                continue
        if len(closes) < 2:
            return []
        out: List[float] = []
        prev = closes[0]
        for v in closes[1:]:
            if prev and prev > 0:
                out.append((float(v) / float(prev)) - 1.0)
            prev = v
        return out

    def _normalize_equity_symbol(self, sym: str) -> str:
        s = str(sym or "").strip().upper()
        if not s:
            return ""
        # Drop exchange prefix if present.
        if ":" in s:
            s = s.split(":", 1)[-1].strip().upper()
        # Normalize common cash-market suffix.
        if not s.endswith("-EQ"):
            s = f"{s}-EQ"
        return s

    def _equity_symbol_candidates(self, sym: str) -> List[str]:
        s = str(sym or "").strip().upper()
        if not s:
            return []
        if ":" in s:
            s = s.split(":", 1)[-1].strip().upper()
        out = [s]
        if s.endswith("-EQ"):
            out.append(s[: -len("-EQ")])
        else:
            out.append(f"{s}-EQ")
        # Dedup, preserve order.
        seen = set()
        uniq: List[str] = []
        for x in out:
            if x and x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    def _equity_signal_from_indicators(self, candles: List[Candle]) -> Optional[str]:
        """Return 'BUY', 'SELL', or None for no signal."""
        if not candles or len(candles) < 30:
            return None
        try:
            candles = sorted(candles, key=lambda c: c.time)
        except Exception:
            pass

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]

        try:
            e_f = int(getattr(self.cfg, "ema_fast", 9) or 9)
            e_s = int(getattr(self.cfg, "ema_slow", 21) or 21)
        except Exception:
            e_f, e_s = (9, 21)

        try:
            rsi_period = int(getattr(self.cfg, "rsi_period", 14) or 14)
        except Exception:
            rsi_period = 14

        fast = ema(closes, period=int(e_f))
        slow = ema(closes, period=int(e_s))
        r = rsi(closes, period=int(rsi_period))

        st_dir = None
        try:
            st = supertrend(highs, lows, closes, period=int(getattr(self.cfg, "supertrend_period", 10) or 10), multiplier=float(getattr(self.cfg, "supertrend_multiplier", 3.0) or 3.0))
            # supertrend() in this repo returns either a tuple or a dict depending on implementation.
            if isinstance(st, dict):
                st_dir = st.get("direction")
            elif isinstance(st, (list, tuple)) and st:
                # Common pattern: (trend, direction)
                if len(st) >= 2:
                    st_dir = st[1]
        except Exception:
            st_dir = None

        if fast is None or slow is None or r is None:
            return None

        # Simple regime: trend + momentum filter.
        want_buy = (float(fast) > float(slow)) and (float(r) >= 50.0)
        want_sell = (float(fast) < float(slow)) and (float(r) <= 50.0)

        # If supertrend direction is available, use it as confirmation.
        try:
            if st_dir is not None:
                # st_dir expected as 1/-1 or 'up'/'down'.
                if str(st_dir).strip().lower() in {"-1", "down", "bear", "bearish"}:
                    want_buy = False
                if str(st_dir).strip().lower() in {"1", "up", "bull", "bullish"}:
                    want_sell = False
        except Exception:
            pass

        if want_buy:
            return "BUY"
        if want_sell:
            return "SELL"
        return None

    def _portfolio_beta_delta_units(self, *, index_symbol: str, spot: float) -> Optional[float]:
        """Estimate equity holdings beta exposure and return equivalent index units.

        Returns Q where portfolio change ~ Q * dS for index move dS.
        Q = sum(beta_i * value_i) / spot
        """

        try:
            enabled = bool(getattr(self.cfg, "delta_hedge_include_equity_portfolio_beta", False))
        except Exception:
            enabled = False
        if not enabled:
            return None

        try:
            spot_f = float(spot)
        except Exception:
            return None
        if spot_f <= 0:
            return None

        # Throttle portfolio beta computation (it can be API-heavy).
        now_ts = float(time.time())
        cache = getattr(self, "_beta_portfolio_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_beta_portfolio_cache", cache)

        try:
            refresh_sec = float(getattr(self.cfg, "delta_hedge_beta_refresh_sec", 300.0) or 300.0)
        except Exception:
            refresh_sec = 300.0
        last_ts = float(cache.get("ts") or 0.0)
        if refresh_sec > 0 and (now_ts - last_ts) < refresh_sec:
            try:
                v = cache.get("delta_units")
                return float(v) if v is not None else None
            except Exception:
                return None

        # Fetch holdings.
        try:
            holdings = self.client.get_holdings()
        except Exception:
            holdings = []
        if not isinstance(holdings, list) or not holdings:
            cache["ts"] = now_ts
            cache["delta_units"] = 0.0
            return 0.0

        # Resolve index token for daily candles.
        try:
            idx_exch, idx_tok = self.client.resolve_exchange_token(index_symbol, exchange_hint="NSE")
        except Exception:
            idx_exch, idx_tok = ("NSE", "26000")

        try:
            lookback = int(getattr(self.cfg, "delta_hedge_beta_lookback_days", 90) or 90)
        except Exception:
            lookback = 90
        lookback = max(30, min(365, int(lookback)))

        idx_candles, _tf = self.client.fetch_index_candles(
            str(idx_tok), exchange=str(idx_exch or "NSE"), timeframe="ONE_DAY", limit=int(lookback) + 5, force_historical_only=True
        )
        if not idx_candles:
            cache["ts"] = now_ts
            cache["delta_units"] = 0.0
            return 0.0
        idx_rets = self._daily_returns_from_candles(list(idx_candles))
        if len(idx_rets) < 10:
            cache["ts"] = now_ts
            cache["delta_units"] = 0.0
            return 0.0

        def _get_any(row: dict, keys: tuple[str, ...]) -> str:
            for k in keys:
                if k in row and row.get(k) is not None:
                    s = str(row.get(k)).strip()
                    if s:
                        return s
            return ""

        def _get_int(row: dict, keys: tuple[str, ...]) -> Optional[int]:
            s = _get_any(row, keys)
            if not s:
                return None
            try:
                return int(float(s))
            except Exception:
                return None

        # Build candidate equity positions.
        rows: List[tuple[str, int]] = []
        for r in holdings:
            if not isinstance(r, dict):
                continue
            ts = _get_any(r, ("tradingsymbol", "tradingSymbol", "symbol", "name")).strip().upper()
            if not ts:
                continue
            q = _get_int(r, ("quantity", "qty", "netqty", "netQty"))
            if q is None or int(q) <= 0:
                continue
            # Skip ETFs if user is already hedging with an ETF symbol separately; they can use include_holdings for that.
            rows.append((ts, int(q)))

        if not rows:
            cache["ts"] = now_ts
            cache["delta_units"] = 0.0
            return 0.0

        try:
            max_syms = int(getattr(self.cfg, "delta_hedge_beta_max_symbols", 12) or 12)
        except Exception:
            max_syms = 12
        max_syms = max(1, min(50, int(max_syms)))

        # Also include bot-managed equity positions (intraday) so the hedge
        # covers newly opened equity trades as well.
        try:
            for sym, pos in (self._equity_positions or {}).items():
                if not isinstance(pos, dict):
                    continue
                try:
                    q = int(pos.get("qty") or 0)
                except Exception:
                    q = 0
                if q == 0:
                    continue
                ts = str(sym or "").strip().upper()
                if ts:
                    rows.append((ts, int(abs(q))))
        except Exception:
            pass

        # Value holdings and pick top-N by notional.
        valued: List[tuple[float, str, int, float]] = []  # (value, ts, qty, ltp)
        for ts, qty in rows:
            ltp = None
            # Holdings may come as "TCS" or "TCS-EQ" depending on source.
            candidates = [ts]
            if ts.endswith("-EQ"):
                candidates.append(ts[: -len("-EQ")])
            else:
                candidates.append(f"{ts}-EQ")

            for c in candidates:
                try:
                    ltp = float(self.client.get_ltp(f"NSE:{c}"))
                    if ltp > 0:
                        break
                except Exception:
                    ltp = None
            if ltp is None:
                continue
            if ltp <= 0:
                continue
            valued.append((float(qty) * float(ltp), ts, qty, ltp))

        if not valued:
            cache["ts"] = now_ts
            cache["delta_units"] = 0.0
            return 0.0

        valued.sort(key=lambda t: t[0], reverse=True)
        valued = valued[:max_syms]

        # Estimate beta for each holding vs index.
        beta_cache = cache.get("betas")
        if not isinstance(beta_cache, dict):
            beta_cache = {}
            cache["betas"] = beta_cache

        beta_notional = 0.0
        used = 0
        for value, ts, qty, ltp in valued:
            # Cache betas for the day.
            b_row = beta_cache.get(ts) if isinstance(beta_cache, dict) else None
            beta_val = None
            try:
                if isinstance(b_row, dict) and (now_ts - float(b_row.get("ts") or 0.0)) < 24 * 3600:
                    beta_val = float(b_row.get("beta"))
            except Exception:
                beta_val = None

            if beta_val is None:
                try:
                    sym_candidates = [ts]
                    if ts.endswith("-EQ"):
                        sym_candidates.append(ts[: -len("-EQ")])
                    else:
                        sym_candidates.append(f"{ts}-EQ")

                    ex, tok = ("", "")
                    for sym_try in sym_candidates:
                        try:
                            ex, tok = self.client.resolve_exchange_token(f"NSE:{sym_try}", exchange_hint="NSE")
                            if ex and tok:
                                break
                        except Exception:
                            ex, tok = ("", "")
                    if not (ex and tok):
                        continue
                except Exception:
                    continue
                cnd, _ = self.client.fetch_index_candles(
                    str(tok), exchange=str(ex or "NSE"), timeframe="ONE_DAY", limit=int(lookback) + 5, force_historical_only=True
                )
                if not cnd:
                    continue
                rets = self._daily_returns_from_candles(list(cnd))
                beta_val = self._compute_beta(rets, idx_rets)
                if beta_val is None:
                    continue
                try:
                    beta_cache[ts] = {"beta": float(beta_val), "ts": now_ts}
                except Exception:
                    pass

            beta_notional += float(beta_val) * float(value)
            used += 1

        delta_units = float(beta_notional) / float(spot_f)
        cache["ts"] = now_ts
        cache["delta_units"] = float(delta_units)
        cache["used"] = int(used)
        return float(delta_units)

    def _equity_fetch_candles(self, symbol: str, *, timeframe: str, limit: int) -> List[Candle]:
        # Use the existing client candle fetcher. It accepts symbols like "NSE:ACC".
        sym_base = str(symbol or "").strip().upper()
        if not sym_base:
            return []
        # Try both raw and -EQ.
        for c in self._equity_symbol_candidates(sym_base):
            try:
                return self.client.get_candles(symbol=f"NSE:{c}", timeframe=str(timeframe), limit=int(limit))
            except Exception:
                continue
        return []

    def _equity_ltp(self, symbol: str) -> Optional[float]:
        sym_base = str(symbol or "").strip().upper()
        if not sym_base:
            return None
        for c in self._equity_symbol_candidates(sym_base):
            try:
                v = float(self.client.get_ltp(f"NSE:{c}"))
                if v > 0:
                    return float(v)
            except Exception:
                continue
        return None

    def _equity_watchlist_symbols(self) -> List[str]:
        raw = str(getattr(self.cfg, "equity_trade_symbols", "") or "")
        if not raw.strip():
            return []
        parts = re.split(r"[\n,;|]+", raw)
        seen: set[str] = set()
        out: List[str] = []
        for item in parts:
            sym = self._normalize_equity_symbol(str(item or "").strip())
            if not sym or sym in seen:
                continue
            seen.add(sym)
            out.append(sym)
        return out

    def _equity_horizon_mode(self, value: Optional[str] = None) -> str:
        raw = value if value is not None else getattr(self.cfg, "equity_trade_horizon", "INTRADAY")
        mode = str(raw or "INTRADAY").strip().upper() or "INTRADAY"
        if mode not in {"INTRADAY", "LONGTERM", "BOTH"}:
            return "INTRADAY"
        return mode

    def _equity_position_horizon(self, pos: Optional[Dict[str, object]] = None) -> str:
        if isinstance(pos, dict):
            h = self._equity_horizon_mode(str(pos.get("horizon") or ""))
            if h != "INTRADAY" or str(pos.get("horizon") or "").strip():
                return h
        mode = self._equity_horizon_mode()
        if mode == "BOTH":
            return "INTRADAY"
        return mode

    def _equity_product_type_for_horizon(self, horizon: str) -> str:
        h = self._equity_horizon_mode(horizon)
        if h == "LONGTERM":
            return str(getattr(self.cfg, "equity_trade_longterm_product_type", "CNC") or "CNC").strip().upper() or "CNC"
        return str(getattr(self.cfg, "equity_trade_product_type", "INTRADAY") or "INTRADAY").strip().upper() or "INTRADAY"

    def _equity_gpt_watchlist_enabled(self, *, is_paper: bool) -> bool:
        if not self._gpt_should_apply_now(is_paper=bool(is_paper)):
            return False
        try:
            if str(getattr(self.cfg, "equity_trade_engine", "INDICATORS") or "INDICATORS").strip().upper() != "GPT":
                return False
        except Exception:
            return False
        try:
            return bool(getattr(self.cfg, "equity_trade_gpt_watchlist_enable", False))
        except Exception:
            return False

    def _equity_candidate_universe(self) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []

        def add_symbol(sym: object) -> None:
            norm = self._normalize_equity_symbol(str(sym or "").strip())
            if not norm or norm in seen:
                return
            seen.add(norm)
            out.append(norm)

        for sym in self._equity_watchlist_symbols():
            add_symbol(sym)

        try:
            holdings = self.client.get_holdings() or []
        except Exception:
            holdings = []
        if isinstance(holdings, list):
            for row in holdings:
                if not isinstance(row, dict):
                    continue
                add_symbol(row.get("tradingsymbol") or row.get("tradingSymbol") or row.get("symbol") or row.get("name"))

        try:
            sm_getter = getattr(self.client, "_get_scripmaster", None)
            sm = sm_getter() if callable(sm_getter) else None
        except Exception:
            sm = None
        if sm is not None:
            try:
                rows = sm._load()  # type: ignore[attr-defined]
            except Exception:
                rows = []
            for row in rows or []:
                try:
                    exch = str(getattr(row, "exch", "") or "").strip().upper()
                    exch_type = str(getattr(row, "exch_type", "") or "").strip().upper()
                    ts = str(getattr(row, "tradingsymbol", "") or "").strip().upper()
                    opt_type = str(getattr(row, "opt_type", "") or "").strip().upper()
                    expiry = getattr(row, "expiry", None)
                except Exception:
                    continue
                if exch != "NSE":
                    continue
                if opt_type in {"CE", "PE"} or expiry is not None:
                    continue
                if exch_type not in {"EQ", "CM", "CASH", "NSECM", "EQUITY"} and not ts.endswith("-EQ"):
                    continue
                add_symbol(ts)

        try:
            limit = int(getattr(self.cfg, "equity_trade_candidate_limit", 20) or 20)
        except Exception:
            limit = 20
        limit = max(1, min(200, int(limit)))
        return out[:limit]

    def _equity_candidate_snapshot(self, symbol: str, *, timeframe: str) -> Optional[Dict[str, object]]:
        candles = self._equity_fetch_candles(symbol, timeframe=timeframe, limit=80)
        if not candles or len(candles) < 20:
            return None
        try:
            candles = sorted(candles, key=lambda c: c.time)
        except Exception:
            pass

        closes = [float(c.close) for c in candles]
        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]
        ltp = closes[-1] if closes else None

        try:
            ema_fast_v = ema(closes, period=int(getattr(self.cfg, "ema_fast", 9) or 9))
        except Exception:
            ema_fast_v = None
        try:
            ema_slow_v = ema(closes, period=int(getattr(self.cfg, "ema_slow", 21) or 21))
        except Exception:
            ema_slow_v = None
        try:
            rsi_v = rsi(closes, period=int(getattr(self.cfg, "rsi_period", 14) or 14))
        except Exception:
            rsi_v = None
        try:
            atr_v = atr(highs, lows, closes, period=int(getattr(self.cfg, "equity_trade_atr_period", 14) or 14))
        except Exception:
            atr_v = None
        try:
            roc_v = roc(closes, period=10)
        except Exception:
            roc_v = None

        return {
            "symbol": str(symbol),
            "ltp": float(ltp) if ltp is not None else None,
            "ema_fast": float(ema_fast_v) if ema_fast_v is not None else None,
            "ema_slow": float(ema_slow_v) if ema_slow_v is not None else None,
            "rsi": float(rsi_v) if rsi_v is not None else None,
            "atr": float(atr_v) if atr_v is not None else None,
            "atr_pct": (float(atr_v) / float(ltp) * 100.0) if (atr_v is not None and ltp and float(ltp) > 0) else None,
            "roc": float(roc_v) if roc_v is not None else None,
        }

    def _gpt_generate_equity_watchlist(self, *, timeframe: str, max_symbols: int, is_paper: bool) -> List[Dict[str, object]]:
        if not self._equity_gpt_watchlist_enabled(is_paper=bool(is_paper)):
            return []
        try:
            if str(getattr(self.cfg, "strategy_name", "") or "").strip().lower() == "auto":
                return []
        except Exception:
            return []

        cache = self._gpt_equity_watchlist_cache if isinstance(self._gpt_equity_watchlist_cache, dict) else {}
        try:
            refresh_sec = float(getattr(self.cfg, "equity_trade_watchlist_refresh_sec", 900.0) or 900.0)
        except Exception:
            refresh_sec = 900.0
        now_ts = float(time.time())
        cached_items = cache.get("items") if isinstance(cache.get("items"), list) else None
        if cached_items is not None:
            try:
                last_ts = float(cache.get("ts") or 0.0)
            except Exception:
                last_ts = 0.0
            if refresh_sec <= 0 or (now_ts - last_ts) < refresh_sec:
                return [dict(x) for x in cached_items if isinstance(x, dict)]

        symbols = self._equity_candidate_universe()
        if not symbols:
            return []

        snapshots: List[Dict[str, object]] = []
        for sym in symbols:
            snap = self._equity_candidate_snapshot(sym, timeframe=timeframe)
            if isinstance(snap, dict):
                snapshots.append(snap)
        if not snapshots:
            return []

        mode = self._equity_horizon_mode()
        proposal = {
            "instrument": "EQUITY",
            "mode": "WATCHLIST",
            "horizon_mode": mode,
            "timeframe": str(timeframe),
            "max_symbols": int(max_symbols),
            "candidates": snapshots,
            "constraints": {
                "allow_short": bool(getattr(self.cfg, "equity_trade_allow_short", False)),
                "max_symbols": int(max_symbols),
            },
        }
        sys_prompt = (
            "You are building a stock watchlist for an automated equity trading bot. "
            "Respond with ONLY valid JSON (no markdown). "
            "Choose the strongest symbols from the provided candidates for automated trading now. "
            "If horizon_mode is BOTH, assign each chosen symbol a horizon of INTRADAY or LONGTERM. Otherwise keep the requested horizon. "
            "Return at most max_symbols items. "
            "Schema: {\"decision\":\"TAKE\"|\"SKIP\",\"reason\":string,\"confidence\":number,\"watchlist\":[{\"symbol\":string,\"horizon\":\"INTRADAY\"|\"LONGTERM\",\"score\":number,\"note\":string}]}"
        )
        adv = self._gpt_control_advise(proposal=proposal, system_prompt=sys_prompt, max_tokens=420)
        extras = getattr(adv, "extras", {}) if adv is not None else {}
        rows = extras.get("watchlist") if isinstance(extras, dict) else None
        if not isinstance(rows, list):
            return []

        allowed = set(symbols)
        out: List[Dict[str, object]] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            sym = self._normalize_equity_symbol(str(row.get("symbol") or "").strip())
            if not sym or sym in seen or sym not in allowed:
                continue
            h = self._equity_horizon_mode(str(row.get("horizon") or mode))
            if mode != "BOTH":
                h = "LONGTERM" if mode == "LONGTERM" else "INTRADAY"
            out.append(
                {
                    "symbol": sym,
                    "horizon": h,
                    "score": row.get("score"),
                    "note": str(row.get("note") or "").strip(),
                }
            )
            seen.add(sym)
            if len(out) >= int(max_symbols):
                break

        self._gpt_equity_watchlist_cache = {"ts": now_ts, "items": [dict(x) for x in out]}
        return out

    def _equity_open_notional(self) -> float:
        total = 0.0
        for sym, pos in list((self._equity_positions or {}).items()):
            if not isinstance(pos, dict):
                continue
            try:
                qty = abs(int(pos.get("qty") or 0))
            except Exception:
                qty = 0
            if qty <= 0:
                continue
            ltp = self._equity_ltp(str(sym))
            if ltp is None:
                try:
                    ltp = float(pos.get("entry")) if pos.get("entry") is not None else None
                except Exception:
                    ltp = None
            if ltp is None or float(ltp) <= 0:
                continue
            total += abs(float(ltp) * float(qty))
        return float(total)

    def _equity_available_capital(self) -> Optional[float]:
        try:
            cap = float(getattr(self.cfg, "equity_trade_capital_rupees", 0.0) or 0.0)
        except Exception:
            cap = 0.0
        if cap <= 0:
            return None
        return max(0.0, float(cap) - float(self._equity_open_notional()))

    def _equity_daily_gpt_manage_enabled(self, *, is_paper: bool) -> bool:
        if not self._gpt_should_apply_now(is_paper=bool(is_paper)):
            return False
        try:
            engine = str(getattr(self.cfg, "equity_trade_engine", "INDICATORS") or "INDICATORS").strip().upper()
        except Exception:
            engine = "INDICATORS"
        if engine != "GPT":
            return False
        try:
            return bool(getattr(self.cfg, "equity_trade_gpt_manage_daily", True))
        except Exception:
            return True

    def _equity_apply_daily_gpt_management(
        self,
        *,
        symbol: str,
        pos: Dict[str, object],
        ltp: float,
        horizon: str,
        candles: List[Candle],
    ) -> Optional[Tuple[str, int]]:
        today_key = self._now_ist_date().isoformat()
        if str(pos.get("last_gpt_manage_day") or "") == today_key:
            return None

        try:
            last_n = int(min(40, max(10, len(candles))))
        except Exception:
            last_n = 30
        c_payload: List[Dict[str, object]] = []
        for candle in candles[-int(last_n) :]:
            try:
                c_payload.append(
                    {
                        "t": getattr(candle, "time", None).isoformat() if getattr(candle, "time", None) is not None else None,
                        "o": float(candle.open),
                        "h": float(candle.high),
                        "l": float(candle.low),
                        "c": float(candle.close),
                    }
                )
            except Exception:
                continue

        side_now = str(pos.get("side") or "").strip().upper()
        try:
            qty_now = abs(int(pos.get("qty") or 0))
        except Exception:
            qty_now = 0
        try:
            entry_now = float(pos.get("entry")) if pos.get("entry") is not None else None
        except Exception:
            entry_now = None
        try:
            stop_now = float(pos.get("stop")) if pos.get("stop") is not None else None
        except Exception:
            stop_now = None
        try:
            target_now = float(pos.get("target")) if pos.get("target") is not None else None
        except Exception:
            target_now = None

        unrealized = None
        if entry_now is not None and qty_now > 0:
            try:
                sign = 1.0 if side_now == "BUY" else -1.0
                unrealized = (float(ltp) - float(entry_now)) * sign * float(qty_now)
            except Exception:
                unrealized = None

        proposal = {
            "instrument": "EQUITY",
            "mode": "MANAGE_DAILY",
            "horizon": str(horizon or "INTRADAY").strip().upper(),
            "symbol": str(symbol),
            "ltp": float(ltp),
            "position": {
                "side": side_now,
                "qty": int(qty_now),
                "entry": entry_now,
                "stop": stop_now,
                "target": target_now,
                "entry_ts": pos.get("entry_ts"),
                "unrealized_pnl": unrealized,
            },
            "candles": c_payload,
            "today": today_key,
        }

        sys_prompt = (
            "You are a daily equity position manager for an automated trading bot. "
            "Respond with ONLY valid JSON (no markdown). "
            "Review the open position once for the current day and choose one action: EXIT, UPDATE, or HOLD. "
            "EXIT means close part/all of the position now. UPDATE means keep holding but optionally change stop/target. HOLD means no change. "
            "Use decision=TAKE only when action=EXIT. Otherwise use decision=SKIP. "
            "If exiting, include side (BUY/SELL) and qty. If updating, stop/target are optional positive prices. "
            "Schema: {\"decision\":\"TAKE\"|\"SKIP\",\"reason\":string,\"confidence\":number,\"action\":\"EXIT\"|\"UPDATE\"|\"HOLD\",\"side\":\"BUY\"|\"SELL\",\"qty\":number,\"stop\":number,\"target\":number}"
        )

        adv = self._gpt_control_advise(proposal=proposal, system_prompt=sys_prompt, max_tokens=260)
        extras = getattr(adv, "extras", {}) if adv is not None else {}
        action = str((extras or {}).get("action") or "HOLD").strip().upper() or "HOLD"

        try:
            stop_new = float((extras or {}).get("stop")) if (extras or {}).get("stop") is not None else None
        except Exception:
            stop_new = None
        try:
            target_new = float((extras or {}).get("target")) if (extras or {}).get("target") is not None else None
        except Exception:
            target_new = None

        if stop_new is not None and float(stop_new) > 0:
            pos["stop"] = float(stop_new)
        if target_new is not None and float(target_new) > 0:
            pos["target"] = float(target_new)
        pos["last_gpt_manage_day"] = today_key
        pos["last_gpt_manage_ts"] = float(time.time())

        if action != "EXIT" and str(getattr(adv, "decision", "") or "").strip().upper() != "TAKE":
            return None

        try:
            side_exit = str((extras or {}).get("side") or "").strip().upper()
        except Exception:
            side_exit = ""
        if side_exit not in {"BUY", "SELL"}:
            side_exit = "SELL" if side_now == "BUY" else "BUY"

        try:
            qty_exit = int(float((extras or {}).get("qty") or 0))
        except Exception:
            qty_exit = 0
        if qty_exit <= 0:
            qty_exit = int(qty_now)
        qty_exit = int(min(max(0, qty_exit), max(0, qty_now)))
        if qty_exit <= 0:
            return None
        return side_exit, qty_exit

    def _equity_place_order(self, symbol: str, side: str, qty: int, *, horizon: Optional[str] = None) -> None:
        if self._kill_switch_active():
            LOGGER.warning("[KILL_SWITCH] Equity order blocked: %s %s x%d", side, symbol, qty)
            return
        if qty <= 0:
            return
        sym_norm = self._normalize_equity_symbol(symbol)
        # Resolve token (try both with and without -EQ).
        exch, tok = ("", "")
        for c in self._equity_symbol_candidates(sym_norm):
            try:
                exch, tok = self.client.resolve_exchange_token(f"NSE:{c}", exchange_hint="NSE")
                if exch and tok:
                    sym_norm = self._normalize_equity_symbol(c)
                    break
            except Exception:
                continue
        if not (exch and tok):
            raise RuntimeError(f"Unable to resolve token for equity symbol {symbol!r}")

        order_type = str(getattr(self.cfg, "equity_trade_order_type", "MARKET") or "MARKET").strip().upper() or "MARKET"
        product_type = self._equity_product_type_for_horizon(str(horizon or ""))

        if not bool(getattr(self.cfg, "enable_live_trading", False)):
            print(f"[PAPER][EQUITY] {side} {sym_norm} x{qty} ({product_type})")
            return

        self.client.place_order(
            symbol=sym_norm,
            side=str(side).strip().upper(),
            quantity=int(qty),
            order_type=order_type,
            exchange=str(exch or "NSE"),
            symbol_token=str(tok),
            product_type=product_type,
            ordertag="nifty-scalper-bot-equity",
        )

    def _manage_equity_positions(self) -> None:
        if not bool(getattr(self.cfg, "equity_trade_enable", False)):
            return

        try:
            eq_engine = str(getattr(self.cfg, "equity_trade_engine", "INDICATORS") or "INDICATORS").strip().upper()
        except Exception:
            eq_engine = "INDICATORS"
        eq_horizon = self._equity_horizon_mode()
        churn_enable = bool(getattr(self.cfg, "equity_trade_churn_enable", False))

        # Long-term target for a quarter (default 90d) when enabled.
        lt_target_pct = 0.0
        lt_days = 90
        if eq_horizon == "LONGTERM":
            try:
                lt_target_pct = float(getattr(self.cfg, "equity_longterm_target_pct", 0.0) or 0.0)
            except Exception:
                lt_target_pct = 0.0
            try:
                lt_days = int(getattr(self.cfg, "equity_longterm_horizon_days", 90) or 90)
            except Exception:
                lt_days = 90
            lt_days = max(1, int(lt_days))

        # Exit logic: stop/target and opposite signal.
        to_close: List[Tuple[str, str, int]] = []
        for sym, pos in list((self._equity_positions or {}).items()):
            if not isinstance(pos, dict):
                continue
            try:
                qty = int(pos.get("qty") or 0)
            except Exception:
                qty = 0
            if qty == 0:
                continue
            pos_horizon = self._equity_position_horizon(pos)
            side = str(pos.get("side") or "").strip().upper()
            ltp = self._equity_ltp(sym)
            if ltp is None:
                continue

            try:
                is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
            except Exception:
                is_paper = True
            if self._equity_daily_gpt_manage_enabled(is_paper=bool(is_paper)):
                try:
                    tf = str(getattr(self.cfg, "equity_trade_timeframe", "5m") or "5m").strip().lower()
                except Exception:
                    tf = "5m"
                try:
                    atr_p = int(getattr(self.cfg, "equity_trade_atr_period", 14) or 14)
                except Exception:
                    atr_p = 14
                manage_candles = self._equity_fetch_candles(sym, timeframe=tf, limit=max(80, atr_p * 4))
                if manage_candles:
                    try:
                        manage_candles = sorted(manage_candles, key=lambda c: c.time)
                    except Exception:
                        pass
                    exit_req = self._equity_apply_daily_gpt_management(
                        symbol=str(sym),
                        pos=pos,
                        ltp=float(ltp),
                        horizon=pos_horizon,
                        candles=manage_candles,
                    )
                    if exit_req is not None:
                        to_close.append((str(sym), str(exit_req[0]), int(exit_req[1])))
                        continue
            entry = None
            try:
                entry = float(pos.get("entry")) if pos.get("entry") is not None else None
            except Exception:
                entry = None
            stop = pos.get("stop")
            target = pos.get("target")
            try:
                stop_f = float(stop) if stop is not None else None
            except Exception:
                stop_f = None
            try:
                target_f = float(target) if target is not None else None
            except Exception:
                target_f = None

            # Stop/target checks.
            if side == "BUY":
                if stop_f is not None and float(ltp) <= float(stop_f):
                    to_close.append((sym, "SELL", int(abs(qty))))
                    continue
                if target_f is not None and float(ltp) >= float(target_f):
                    to_close.append((sym, "SELL", int(abs(qty))))
                    continue
            elif side == "SELL":
                if stop_f is not None and float(ltp) >= float(stop_f):
                    to_close.append((sym, "BUY", int(abs(qty))))
                    continue
                if target_f is not None and float(ltp) <= float(target_f):
                    to_close.append((sym, "BUY", int(abs(qty))))
                    continue

            # Long-term % target for a quarter.
            if pos_horizon == "LONGTERM" and entry is not None and float(entry) > 0 and float(lt_target_pct) > 0:
                try:
                    pct = float(lt_target_pct) / 100.0
                except Exception:
                    pct = 0.0
                if pct > 0:
                    if side == "BUY" and float(ltp) >= float(entry) * (1.0 + pct):
                        to_close.append((sym, "SELL", int(abs(qty))))
                        continue
                    if side == "SELL" and float(ltp) <= float(entry) * (1.0 - pct):
                        to_close.append((sym, "BUY", int(abs(qty))))
                        continue

            # Long-term horizon max hold (quarter) when enabled.
            if pos_horizon == "LONGTERM":
                try:
                    ets = float(pos.get("entry_ts") or 0.0)
                except Exception:
                    ets = 0.0
                if ets > 0:
                    try:
                        held_days = (dt_datetime.fromtimestamp(float(time.time())).date() - dt_datetime.fromtimestamp(float(ets)).date()).days
                    except Exception:
                        held_days = int(max(0.0, (float(time.time()) - float(ets)) / 86400.0))
                    if int(held_days) >= int(lt_days):
                        to_close.append((sym, "SELL" if side == "BUY" else "BUY", int(abs(qty))))
                        continue

            # Optional GPT churn: ask whether to keep/exit.
            if eq_engine == "GPT" and churn_enable:
                # Only run GPT churn when GPT gating is enabled for this run.
                try:
                    is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
                except Exception:
                    is_paper = True
                if self._gpt_should_apply_now(is_paper=bool(is_paper)):
                    proposal = {
                        "instrument": "EQUITY",
                        "mode": "CHURN",
                        "horizon": eq_horizon,
                        "symbol": str(sym),
                        "ltp": float(ltp),
                        "position": {
                            "side": side,
                            "qty": int(qty),
                            "entry": entry,
                            "stop": stop_f,
                            "target": target_f,
                            "entry_ts": pos.get("entry_ts"),
                        },
                        "longterm": {
                            "target_pct": float(lt_target_pct),
                            "horizon_days": int(lt_days),
                        }
                        if eq_horizon == "LONGTERM"
                        else None,
                    }

                    sys_prompt = (
                        "You are an equity portfolio manager for an automated bot. "
                        "Respond with ONLY valid JSON. "
                        "Decide whether to TAKE (exit now) or SKIP (keep holding). "
                        "If TAKE, include side as the exit action to place now: BUY or SELL, and qty (positive integer). "
                        "If information is missing/ambiguous, SKIP. "
                        "Schema: {\"decision\":\"TAKE\"|\"SKIP\",\"reason\":string,\"confidence\":number,\"side\":\"BUY\"|\"SELL\",\"qty\":number}"
                    )

                    adv = self._gpt_control_advise(proposal=proposal, system_prompt=sys_prompt, max_tokens=250)
                    try:
                        dec = str(getattr(adv, "decision", "UNKNOWN") or "UNKNOWN").strip().upper()
                    except Exception:
                        dec = "UNKNOWN"
                    if dec == "TAKE":
                        try:
                            side2 = str(getattr(adv, "extras", {}).get("side") or "").strip().upper()
                        except Exception:
                            side2 = ""
                        try:
                            q2 = int(float(getattr(adv, "extras", {}).get("qty") or 0))
                        except Exception:
                            q2 = 0
                        if side2 in {"BUY", "SELL"} and int(q2) > 0:
                            to_close.append((sym, side2, int(min(abs(qty), abs(q2)))))
                            continue

        for sym, close_side, qty in to_close:
            try:
                horizon = self._equity_position_horizon(self._equity_positions.get(sym) if isinstance(self._equity_positions, dict) else None)
                self._equity_place_order(sym, close_side, qty, horizon=horizon)
                # Remove from internal state.
                self._equity_positions.pop(sym, None)
                self.state.last_exit_ts = float(time.time())
            except Exception as exc:
                print(f"[EQUITY] Close failed for {sym}: {exc}")

    def _decide_equity_entries(self) -> None:
        if not bool(getattr(self.cfg, "equity_trade_enable", False)):
            return
        if self.state.trades_today >= int(self.cfg.max_trades_per_day):
            return

        try:
            eq_engine = str(getattr(self.cfg, "equity_trade_engine", "INDICATORS") or "INDICATORS").strip().upper()
        except Exception:
            eq_engine = "INDICATORS"
        eq_horizon = self._equity_horizon_mode()
        churn_enable = bool(getattr(self.cfg, "equity_trade_churn_enable", False))

        try:
            max_syms = int(getattr(self.cfg, "equity_trade_max_symbols", 10) or 10)
        except Exception:
            max_syms = 10
        max_syms = max(1, min(50, int(max_syms)))

        try:
            tf = str(getattr(self.cfg, "equity_trade_timeframe", "5m") or "5m").strip().lower()
        except Exception:
            tf = "5m"

        # Bucket guard to avoid repeated decisions within the same candle.
        try:
            bucket_sec = int(self._tf_bucket_seconds(tf))
        except Exception:
            bucket_sec = 300
        if bucket_sec <= 0:
            bucket_sec = 300
        now_ts = float(time.time())
        bucket_start_ts = now_ts - (now_ts % float(bucket_sec))
        if self._last_equity_bucket_start_ts is not None and float(self._last_equity_bucket_start_ts) == float(bucket_start_ts):
            return
        self._last_equity_bucket_start_ts = float(bucket_start_ts)

        # Rebalance interval.
        try:
            reb = float(getattr(self.cfg, "equity_trade_rebalance_interval_sec", 60.0) or 60.0)
        except Exception:
            reb = 60.0
        if reb > 0:
            # Use last entry ts as a global throttle for equity entries.
            try:
                last = float(getattr(self, "_last_equity_decide_ts", 0.0) or 0.0)
            except Exception:
                last = 0.0
            if (now_ts - last) < reb:
                return
            setattr(self, "_last_equity_decide_ts", now_ts)

        # Pull holdings and/or a user-provided watchlist.
        try:
            holdings = self.client.get_holdings() or []
        except Exception:
            holdings = []
        if not isinstance(holdings, list):
            holdings = []

        def _get_any(row: dict, keys: tuple[str, ...]) -> str:
            for k in keys:
                if k in row and row.get(k) is not None:
                    s = str(row.get(k)).strip()
                    if s:
                        return s
            return ""

        def _get_int(row: dict, keys: tuple[str, ...]) -> Optional[int]:
            s = _get_any(row, keys)
            if not s:
                return None
            try:
                return int(float(s))
            except Exception:
                return None

        holding_qty_by_sym: Dict[str, int] = {}
        for r in holdings:
            if not isinstance(r, dict):
                continue
            sym = self._normalize_equity_symbol(_get_any(r, ("tradingsymbol", "tradingSymbol", "symbol", "name")).strip().upper())
            qty = _get_int(r, ("quantity", "qty", "netqty", "netQty"))
            if not sym or qty is None or int(qty) <= 0:
                continue
            holding_qty_by_sym[str(sym)] = int(qty)

        try:
            is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
        except Exception:
            is_paper = True
        watchlist_plan = self._gpt_generate_equity_watchlist(timeframe=tf, max_symbols=max_syms, is_paper=bool(is_paper))
        horizon_by_symbol: Dict[str, str] = {}
        symbols: List[str] = []
        if watchlist_plan:
            for row in watchlist_plan:
                if not isinstance(row, dict):
                    continue
                sym = self._normalize_equity_symbol(str(row.get("symbol") or "").strip())
                if not sym:
                    continue
                symbols.append(sym)
                horizon_by_symbol[sym] = self._equity_horizon_mode(str(row.get("horizon") or eq_horizon))
        else:
            watchlist_symbols = self._equity_watchlist_symbols()
            if watchlist_symbols:
                symbols = list(watchlist_symbols[:max_syms])
                for sym in symbols:
                    horizon_by_symbol[sym] = "LONGTERM" if eq_horizon == "LONGTERM" else "INTRADAY"
            else:
                candidates: List[Tuple[float, str]] = []
                for sym, qty in list(holding_qty_by_sym.items()):
                    ltp = self._equity_ltp(sym)
                    if ltp is None:
                        continue
                    candidates.append((float(ltp) * float(qty), sym))
                if not candidates:
                    return
                candidates.sort(key=lambda t: t[0], reverse=True)
                symbols = [s for _v, s in candidates[:max_syms]]
                for sym in symbols:
                    horizon_by_symbol[sym] = "LONGTERM" if eq_horizon == "LONGTERM" else "INTRADAY"
        if not symbols:
            return

        allow_short = bool(getattr(self.cfg, "equity_trade_allow_short", False))

        # Equity cooldown.
        try:
            cd = float(getattr(self.cfg, "equity_trade_cooldown_sec", 300.0) or 300.0)
        except Exception:
            cd = 300.0

        for sym in symbols:
            # Skip if already in a position.
            if sym in (self._equity_positions or {}):
                continue

            # Cooldown per symbol.
            try:
                last_act = float(getattr(self, "_equity_last_action", {}).get(sym) or 0.0)
            except Exception:
                last_act = 0.0
            if cd > 0 and last_act and (now_ts - last_act) < cd:
                continue

            # Fetch candles.
            atr_p = int(getattr(self.cfg, "equity_trade_atr_period", 14) or 14)
            limit = max(120, atr_p * 3)
            candles = self._equity_fetch_candles(sym, timeframe=tf, limit=int(limit))
            if not candles:
                continue
            try:
                candles = sorted(candles, key=lambda c: c.time)
            except Exception:
                pass
            # Decide signal.
            signal = None
            gpt_qty = None
            gpt_stop = None
            gpt_target = None
            symbol_horizon = self._equity_horizon_mode(horizon_by_symbol.get(sym) or eq_horizon)
            if symbol_horizon == "BOTH":
                symbol_horizon = "INTRADAY"

            if eq_engine == "GPT":
                # GPT-only equity decision: ask for BUY/SELL/HOLD.
                if not self._gpt_should_apply_now(is_paper=bool(is_paper)):
                    # If GPT engine is selected but GPT gating isn't enabled, be safe and do nothing.
                    continue

                # Keep prompt payload small: include only the last N candles.
                try:
                    last_n = int(min(40, max(10, len(candles))))
                except Exception:
                    last_n = 30
                c_slice = candles[-int(last_n) :]
                c_payload = []
                for c in c_slice:
                    try:
                        c_payload.append(
                            {
                                "t": getattr(c, "time", None).isoformat() if getattr(c, "time", None) is not None else None,
                                "o": float(c.open),
                                "h": float(c.high),
                                "l": float(c.low),
                                "c": float(c.close),
                            }
                        )
                    except Exception:
                        continue

                # Long-term target constraints.
                lt_target_pct = 0.0
                lt_days = 90
                if symbol_horizon == "LONGTERM":
                    try:
                        lt_target_pct = float(getattr(self.cfg, "equity_longterm_target_pct", 0.0) or 0.0)
                    except Exception:
                        lt_target_pct = 0.0
                    try:
                        lt_days = int(getattr(self.cfg, "equity_longterm_horizon_days", 90) or 90)
                    except Exception:
                        lt_days = 90
                    lt_days = max(1, int(lt_days))

                proposal = {
                    "instrument": "EQUITY",
                    "mode": "ENTRY",
                    "horizon": symbol_horizon,
                    "horizon_mode": eq_horizon,
                    "symbol": str(sym),
                    "timeframe": str(tf),
                    "ltp": float(self._equity_ltp(sym) or 0.0),
                    "candles": c_payload,
                    "holding_qty": int(holding_qty_by_sym.get(str(sym), 0) or 0),
                    "constraints": {
                        "allow_short": bool(allow_short),
                        "max_notional": float(getattr(self.cfg, "equity_trade_max_notional", 20000.0) or 20000.0),
                        "product_type": self._equity_product_type_for_horizon(symbol_horizon),
                        "allowed_horizons": [symbol_horizon] if eq_horizon != "BOTH" else ["INTRADAY", "LONGTERM"],
                    },
                    "longterm": {
                        "target_pct": float(lt_target_pct),
                        "horizon_days": int(lt_days),
                    }
                    if symbol_horizon == "LONGTERM"
                    else None,
                }

                sys_prompt = (
                    "You are an equity trading decision engine for an automated bot. "
                    "Respond with ONLY valid JSON (no markdown). "
                    "Decide whether to TAKE or SKIP the proposed opportunity. "
                    "If TAKE, include side=BUY or SELL, qty (positive integer), and optional stop/target prices. "
                    "If multiple horizons are allowed, also include horizon as INTRADAY or LONGTERM. "
                    "If SKIP, do not include side/qty. "
                    "Do NOT suggest short SELL unless constraints.allow_short is true. "
                    "Prefer conservative decisions; if uncertain, SKIP. "
                    "Schema: {\"decision\":\"TAKE\"|\"SKIP\",\"reason\":string,\"confidence\":number,\"side\":\"BUY\"|\"SELL\",\"qty\":number,\"stop\":number,\"target\":number,\"horizon\":\"INTRADAY\"|\"LONGTERM\"}"
                )

                adv = self._gpt_control_advise(proposal=proposal, system_prompt=sys_prompt, max_tokens=280)
                try:
                    dec = str(getattr(adv, "decision", "UNKNOWN") or "UNKNOWN").strip().upper()
                except Exception:
                    dec = "UNKNOWN"
                if dec != "TAKE":
                    continue

                extras = getattr(adv, "extras", {}) if adv is not None else {}
                try:
                    signal = str((extras or {}).get("side") or "").strip().upper()
                except Exception:
                    signal = None
                try:
                    gpt_qty = int(float((extras or {}).get("qty") or 0))
                except Exception:
                    gpt_qty = None
                try:
                    gpt_stop = float((extras or {}).get("stop")) if (extras or {}).get("stop") is not None else None
                except Exception:
                    gpt_stop = None
                try:
                    gpt_target = float((extras or {}).get("target")) if (extras or {}).get("target") is not None else None
                except Exception:
                    gpt_target = None
                try:
                    gpt_horizon = self._equity_horizon_mode(str((extras or {}).get("horizon") or symbol_horizon))
                except Exception:
                    gpt_horizon = symbol_horizon
                if eq_horizon == "BOTH":
                    symbol_horizon = gpt_horizon

                if signal not in {"BUY", "SELL"}:
                    continue
                if signal == "SELL" and (not allow_short):
                    # In intraday mode, SELL implies shorting; respect allow_short.
                    # (Long-term churn of holdings is intentionally not auto-enabled here.)
                    continue
                if signal == "SELL" and symbol_horizon == "LONGTERM":
                    continue
            else:
                signal = self._equity_signal_from_indicators(candles)
                if signal is None:
                    continue
                if signal == "SELL" and not allow_short:
                    continue

            ltp = self._equity_ltp(sym)
            if ltp is None:
                continue

            # ATR sizing (fallback if GPT doesn't provide qty).
            atr_val = None
            try:
                highs = [c.high for c in candles]
                lows = [c.low for c in candles]
                closes = [c.close for c in candles]
                atr_val = atr(highs, lows, closes, period=int(atr_p))
            except Exception:
                atr_val = None
            if atr_val is not None and float(atr_val) > 0:
                self._cached_atr = float(atr_val)
            if atr_val is None or float(atr_val) <= 0:
                continue

            try:
                risk_rupees = float(getattr(self.cfg, "equity_trade_risk_rupees", 500.0) or 500.0)
            except Exception:
                risk_rupees = 500.0
            try:
                sl_mult = float(getattr(self.cfg, "equity_trade_sl_atr_mult", 2.0) or 2.0)
            except Exception:
                sl_mult = 2.0
            try:
                tp_mult = float(getattr(self.cfg, "equity_trade_tp_atr_mult", 3.0) or 3.0)
            except Exception:
                tp_mult = 3.0
            try:
                max_notional = float(getattr(self.cfg, "equity_trade_max_notional", 20000.0) or 20000.0)
            except Exception:
                max_notional = 20000.0
            capital_remaining = self._equity_available_capital()
            if capital_remaining is not None and float(capital_remaining) <= 0:
                break

            stop_dist = float(atr_val) * float(sl_mult)
            if stop_dist <= 0:
                continue
            qty = int(max(1.0, float(risk_rupees) / float(stop_dist)))
            entry_notional_cap: Optional[float] = None
            if max_notional > 0:
                entry_notional_cap = float(max_notional)
            if capital_remaining is not None:
                entry_notional_cap = float(capital_remaining) if entry_notional_cap is None else min(float(entry_notional_cap), float(capital_remaining))
            if entry_notional_cap is not None:
                if float(entry_notional_cap) < float(ltp):
                    continue
                qty = min(int(qty), int(float(entry_notional_cap) / float(ltp)))
            if qty <= 0:
                continue

            # If GPT suggested a qty, clamp it.
            if eq_engine == "GPT" and gpt_qty is not None and int(gpt_qty) > 0:
                qty = int(max(1, int(gpt_qty)))
                if entry_notional_cap is not None:
                    if float(entry_notional_cap) < float(ltp):
                        continue
                    qty = min(int(qty), int(float(entry_notional_cap) / float(ltp)))
                if qty <= 0:
                    continue

            if signal == "BUY":
                stop = float(ltp) - float(stop_dist)
                target = float(ltp) + float(atr_val) * float(tp_mult)
            else:
                stop = float(ltp) + float(stop_dist)
                target = float(ltp) - float(atr_val) * float(tp_mult)

            # If GPT suggested stop/target, accept them (with minimal sanity checks).
            if eq_engine == "GPT":
                try:
                    if gpt_stop is not None and float(gpt_stop) > 0:
                        stop = float(gpt_stop)
                except Exception:
                    pass
                try:
                    if gpt_target is not None and float(gpt_target) > 0:
                        target = float(gpt_target)
                except Exception:
                    pass

            try:
                self._equity_place_order(sym, signal, int(qty), horizon=symbol_horizon)
            except Exception as exc:
                print(f"[EQUITY] Entry failed for {sym}: {exc}")
                continue

            self._equity_positions[sym] = {
                "symbol": str(sym),
                "horizon": str(symbol_horizon),
                "side": signal,
                "qty": int(qty) if signal == "BUY" else -int(qty),
                "entry": float(ltp),
                "stop": float(stop),
                "target": float(target),
                "entry_ts": float(now_ts),
                "last_action_ts": float(now_ts),
                "last_gpt_manage_day": "",
            }
            try:
                last = getattr(self, "_equity_last_action", None)
                if not isinstance(last, dict):
                    last = {}
                    setattr(self, "_equity_last_action", last)
                last[sym] = now_ts
            except Exception:
                pass

            self.state.trades_today += 1
            self.state.last_entry_ts = float(now_ts)

    def _ensure_portfolio_hedge_trade(self) -> Dict[str, object]:
        if isinstance(self._portfolio_hedge_trade, dict):
            return self._portfolio_hedge_trade

        trade: Dict[str, object] = {
            "trade_id": "P-BETA",
            "name": "Portfolio Beta Hedge",
            "position_type": "multi",
            "legs": [],
            "meta": {
                "delta_hedge": True,
                # Allow delta hedge to run even without option legs.
                "delta_hedge_portfolio_only": True,
                # Use global hedge symbol selection (NIFTY/BANKNIFTY fields).
                "delta_hedge_symbol": str(getattr(self.cfg, "delta_hedge_symbol", "") or "").strip() or None,
            },
        }
        self._portfolio_hedge_trade = trade
        return trade

    def _rebalance_portfolio_beta_hedge(self) -> None:
        # Must be explicitly enabled.
        if not bool(getattr(self.cfg, "delta_hedge_portfolio_hedge_enable", False)):
            return
        if not bool(getattr(self.cfg, "delta_hedge_include_equity_portfolio_beta", False)):
            return

        # Benchmark spot.
        bench = str(getattr(self.cfg, "equity_trade_benchmark", "NIFTY") or "NIFTY").strip().upper() or "NIFTY"
        spot = None
        try:
            # Try common m.Stock LTP resolution.
            spot = float(self.client.get_ltp(f"NSE:{bench}"))
        except Exception:
            spot = None
        if spot is None or float(spot) <= 0:
            return

        trade = self._ensure_portfolio_hedge_trade()
        meta = trade.get("meta") if isinstance(trade.get("meta"), dict) else {}
        if isinstance(meta, dict):
            meta["delta_hedge_underlying"] = bench
            trade["meta"] = meta
        self._rebalance_delta_hedge(trade, spot=float(spot), force=False)

    def _intraday_only_enabled(self) -> bool:
        """Match mstock_client intraday-only semantics.

        If MSTOCK_USE_INTRADAY_CHART is enabled, we default to intraday-only unless
        MSTOCK_INTRADAY_ONLY is explicitly set.
        """

        use_intraday = self._bool_env("MSTOCK_USE_INTRADAY_CHART", False)
        intraday_only_env = os.getenv("MSTOCK_INTRADAY_ONLY")
        if intraday_only_env is None:
            return bool(use_intraday)
        return bool(use_intraday) and self._bool_env("MSTOCK_INTRADAY_ONLY", False)

    def _prefer_synthetic_warmup(self) -> bool:
        if self._intraday_only_enabled():
            return True
        try:
            client_name = type(self.client).__name__.strip().lower()
        except Exception:
            client_name = ""
        return client_name == "dhanclient"

    def _tf_bucket_seconds(self, timeframe: str) -> int:
        tf = str(timeframe or "").strip().lower()
        if tf.endswith("m"):
            try:
                return max(60, int(tf[:-1]) * 60)
            except Exception:
                return 60
        if tf.endswith("h"):
            try:
                return max(3600, int(tf[:-1]) * 3600)
            except Exception:
                return 3600
        if tf.endswith("d"):
            return 86400
        # Default: treat as 1m.
        return 60

    def _synth_candles_from_ltp(self, *, timeframe: str, limit: int) -> List[Candle]:
        """Build candles from live underlying LTP samples.

        - Uses only quote/LTP (no historical endpoints).
        - Aggregates into timeframe buckets.
        """

        tf_key = str(timeframe or "").strip().lower() or "1m"
        bucket_sec = int(self._tf_bucket_seconds(tf_key))
        now_ts = float(time.time())
        bucket_start_ts = now_ts - (now_ts % float(bucket_sec))
        bucket_dt = dt_datetime.fromtimestamp(bucket_start_ts)

        spot = self._try_get_ltp(self.cfg.underlying)
        if spot is None:
            return list(self._synthetic_candles.get(tf_key, []))

        cur = self._synthetic_current.get(tf_key)
        if cur is None or cur.time != bucket_dt:
            # Roll the previous candle into history.
            if cur is not None:
                hist = self._synthetic_candles.setdefault(tf_key, [])
                hist.append(cur)
                # Cap memory.
                if len(hist) > max(200, int(limit) * 2):
                    del hist[: max(0, len(hist) - max(200, int(limit) * 2))]

            cur = Candle(time=bucket_dt, open=spot, high=spot, low=spot, close=spot, volume=None)
            self._synthetic_current[tf_key] = cur

            # One-time bootstrap: if we have zero history for this timeframe,
            # backfill a small flat series so indicator warmup can proceed.
            hist = self._synthetic_candles.setdefault(tf_key, [])
            if not hist:
                try:
                    backfill_n = max(0, min(60, int(limit) - 1))
                except Exception:
                    backfill_n = 30
                if backfill_n > 0:
                    for i in range(backfill_n, 0, -1):
                        ts = bucket_start_ts - float(i * bucket_sec)
                        t = dt_datetime.fromtimestamp(ts)
                        hist.append(Candle(time=t, open=spot, high=spot, low=spot, close=spot, volume=None))
        else:
            # Update current candle.
            try:
                cur.high = float(max(float(cur.high), spot))
            except Exception:
                cur.high = spot
            try:
                cur.low = float(min(float(cur.low), spot))
            except Exception:
                cur.low = spot
            cur.close = float(spot)

        hist = list(self._synthetic_candles.get(tf_key, []))
        out = hist + [cur]
        if int(limit) > 0 and len(out) > int(limit):
            out = out[-int(limit) :]
        return out

    def _has_event_sink(self) -> bool:
        return self._event_sink is not None

    def _normalize_root(self, s: str) -> str:
        """Map common underlying synonyms to their standard option ticker roots."""
        val = str(s or "").strip().upper().replace(" ", "")
        if val in {"NIFTY50", "NIFTY-50", "CNXNIFTY", "NIFTYINDEX"}:
            return "NIFTY"
        if val in {"BANKNIFTY", "NIFTYBANK", "BANK-NIFTY", "NSEBANK", "BANKNIFTYINDEX"}:
            return "BANKNIFTY"
        if val in {"FINNIFTY", "NIFTYFINSERVICE", "FIN-NIFTY", "FINNIFTYINDEX"}:
            return "FINNIFTY"
        if val in {"MIDCPNIFTY", "NIFTYMIDSELECT", "MIDCP-NIFTY", "MIDCPNIFTYINDEX"}:
            return "MIDCPNIFTY"
        return val

    def _infer_underlying_root_from_legs(self, legs: List[dict]) -> Optional[str]:
        """Best-effort inference of underlying root from option leg symbols."""

        if not isinstance(legs, list):
            return None

        for leg in legs:
            if not isinstance(leg, dict):
                continue
            if bool(leg.get("is_hedge")):
                continue

            # Some option-chain sources omit strike/expiry/option_type.
            # Derive them from the symbol so delta hedging can still work.
            self._ensure_option_leg_fields(leg)

            opt_type = str(leg.get("option_type") or "").strip().upper()
            if opt_type not in {"CE", "PE", "CALL", "PUT", "C", "P"}:
                # Likely not an option leg.
                continue

            sym = str(leg.get("symbol") or "").strip().upper()
            if not sym:
                continue

            # Common index roots appear in the option trading symbol.
            if "BANKNIFTY" in sym:
                return "BANKNIFTY"
            if "NIFTY" in sym:
                return "NIFTY"

            # Fallback: attempt prefix match before first digit.
            m = re.match(r"^([A-Z]+)", sym)
            if m:
                root = self._normalize_root(m.group(1))
                if root in {"NIFTY", "BANKNIFTY"}:
                    return root

        return None

    def _pick_delta_hedge_symbol_for_trade(self, trade: Dict[str, object]) -> Tuple[str, str, Optional[str]]:
        """Return (hedge_symbol_raw, exchange_hint, inferred_root)."""

        meta = trade.get("meta") if isinstance(trade.get("meta"), dict) else {}
        if not isinstance(meta, dict):
            meta = {}

        # 1) Explicit override on the trade.
        sym = str(meta.get("delta_hedge_symbol") or "").strip()
        exch = str(meta.get("delta_hedge_exchange") or "").strip()
        if sym:
            return sym, exch, None

        legs = trade.get("legs")
        root = None
        if isinstance(legs, list):
            root = self._infer_underlying_root_from_legs(legs)
        if not root:
            try:
                root = self._normalize_root(str(getattr(self.cfg, "underlying", "") or ""))
            except Exception:
                root = None

        # 2) Per-underlying config (if root inferred).
        if root == "NIFTY":
            sym_n = str(getattr(self.cfg, "delta_hedge_symbol_nifty", "") or "").strip()
            exch_n = str(getattr(self.cfg, "delta_hedge_exchange_nifty", "") or "").strip()
            if sym_n:
                return sym_n, exch_n, root
        if root == "BANKNIFTY":
            sym_b = str(getattr(self.cfg, "delta_hedge_symbol_banknifty", "") or "").strip()
            exch_b = str(getattr(self.cfg, "delta_hedge_exchange_banknifty", "") or "").strip()
            if sym_b:
                return sym_b, exch_b, root

        # 3) Fallback: global hedge symbol.
        sym_g = str(getattr(self.cfg, "delta_hedge_symbol", "") or "").strip()
        exch_g = str(getattr(self.cfg, "delta_hedge_exchange", "") or "").strip()
        if sym_g:
            return sym_g, exch_g, root

        # 4) Zero-config fallback: try a sensible broker-resolvable hedge symbol.
        auto_sym, auto_exch = self._auto_delta_hedge_symbol_for_root(root)
        if auto_sym:
            return auto_sym, auto_exch, root
        return "", "", root

    def _auto_delta_hedge_symbol_for_root(self, root: Optional[str]) -> Tuple[str, str]:
        root_u = self._normalize_root(str(root or "").strip().upper())
        if root_u not in {"NIFTY", "BANKNIFTY"}:
            return "", ""

        cache = self._auto_delta_hedge_symbol_cache if isinstance(self._auto_delta_hedge_symbol_cache, dict) else {}
        cached = cache.get(root_u)
        if isinstance(cached, tuple) and len(cached) == 2 and str(cached[0] or "").strip():
            return str(cached[0]), str(cached[1])

        candidates: List[Tuple[str, str]] = []
        if root_u == "NIFTY":
            candidates.extend([
                ("NSE:NIFTYBEES", "NSE"),
                ("NFO:NIFTY", "NFO"),
            ])
        elif root_u == "BANKNIFTY":
            candidates.extend([
                ("NSE:BANKBEES", "NSE"),
                ("NFO:BANKNIFTY", "NFO"),
            ])

        try:
            sm_getter = getattr(self.client, "_get_scripmaster", None)
            sm = sm_getter() if callable(sm_getter) else None
        except Exception:
            sm = None
        if sm is not None:
            try:
                fut_ts = sm.nearest_future_tradingsymbol(
                    root_u,
                    exch=(os.getenv("MSTOCK_SCRIPMASTER_EXCH", "NFO") or "NFO"),
                    asof=self._now_ist_date(),
                )
            except Exception:
                fut_ts = None
            if fut_ts:
                candidates.append((f"NFO:{str(fut_ts).strip()}", "NFO"))

        seen: set[str] = set()
        for sym_raw, exch_hint in candidates:
            key = f"{str(exch_hint).strip().upper()}::{str(sym_raw).strip().upper()}"
            if not sym_raw or key in seen:
                continue
            seen.add(key)
            try:
                resolved_exch, _resolved_tok = self.client.resolve_exchange_token(sym_raw, exchange_hint=exch_hint or None)
            except Exception:
                continue
            out = (str(sym_raw).strip(), str(resolved_exch or exch_hint).strip().upper())
            self._auto_delta_hedge_symbol_cache[root_u] = out
            return out

        return "", ""

    def _trade_type_key(self, *, position_type: str, name: str) -> str:
        return f"{str(position_type).strip().lower()}:{str(name).strip().lower()}"

    def _base_trade_name(self, name: str) -> str:
        s = str(name or "").strip().lower()
        if s.endswith("(weekly)"):
            s = s[: -len("(weekly)")].strip()
        return s

    def _supported_multi_trade_names(self) -> List[str]:
        return [
            "directional",
            "short_straddle",
            "short_strangle",
            "bull_call_spread",
            "bull_put_spread",
            "call_ratio_backspread",
            "put_ratio_backspread",
            "iron_condor",
            "iron_fly",
            "iron_butterfly",
            "long_straddle",
            "long_strangle",
            "delta_hedged_short_straddle",
            "delta_hedged_long_straddle",
        ]

    def _same_trade_type_alternatives(self, *, position_type: str, name: str) -> List[str]:
        position_key = str(position_type or "").strip().lower()
        base_name = self._base_trade_name(name)
        out: List[str] = []

        def push(candidate: object) -> None:
            cand_raw = str(candidate or "").strip().lower()
            if not cand_raw:
                return
            cand = self._normalize_strategy_name(cand_raw)
            if cand in {"", "auto", "directional"}:
                return
            if cand == base_name:
                return
            if position_key == "multi" and cand not in self._supported_multi_trade_names():
                return
            if cand not in out:
                out.append(cand)

        try:
            router_candidates = (self._last_router_snapshot or {}).get("candidates")
            if isinstance(router_candidates, list):
                for candidate in router_candidates:
                    push(candidate)
        except Exception:
            pass

        if position_key == "multi":
            defaults_by_trade = {
                "iron_condor": ["iron_fly", "short_strangle", "short_straddle", "iron_butterfly", "long_strangle"],
                "iron_fly": ["iron_condor", "short_straddle", "short_strangle", "iron_butterfly", "long_strangle"],
                "iron_butterfly": ["iron_fly", "iron_condor", "short_straddle", "short_strangle", "long_strangle"],
                "short_straddle": ["short_strangle", "iron_fly", "iron_condor", "iron_butterfly", "long_strangle"],
                "short_strangle": ["iron_condor", "iron_fly", "short_straddle", "iron_butterfly", "long_strangle"],
                "bull_put_spread": ["bull_call_spread", "short_strangle", "iron_condor", "put_ratio_backspread"],
                "bull_call_spread": ["call_ratio_backspread", "long_strangle", "bull_put_spread", "long_straddle"],
                "call_ratio_backspread": ["bull_call_spread", "long_strangle", "long_straddle", "put_ratio_backspread"],
                "put_ratio_backspread": ["bull_put_spread", "long_strangle", "long_straddle", "call_ratio_backspread"],
                "long_straddle": ["long_strangle", "call_ratio_backspread", "put_ratio_backspread", "bull_call_spread"],
                "long_strangle": ["long_straddle", "call_ratio_backspread", "put_ratio_backspread", "iron_condor"],
                "delta_hedged_short_straddle": ["iron_condor", "short_strangle", "iron_fly", "delta_hedged_long_straddle"],
                "delta_hedged_long_straddle": ["long_strangle", "long_straddle", "call_ratio_backspread", "put_ratio_backspread"],
            }
            for candidate in defaults_by_trade.get(base_name, self._supported_multi_trade_names()):
                push(candidate)

        return out

    def _suggest_alternative_trade_type(self, *, position_type: str, name: str) -> Optional[Dict[str, object]]:
        candidates = self._same_trade_type_alternatives(position_type=position_type, name=name)
        if not candidates:
            return None

        chosen = str(candidates[0])
        source = "policy"
        gpt_reason = ""

        try:
            gpt_enabled = bool(getattr(self.cfg, "gpt_enable", False))
        except Exception:
            gpt_enabled = False

        if gpt_enabled:
            try:
                advice = self._gpt_control_advise(
                    proposal={
                        "request_type": "alternative_trade_due_to_same_trade_type_block",
                        "position_type": str(position_type or ""),
                        "blocked_trade_name": str(name or ""),
                        "strategy_mode": str(getattr(self.cfg, "strategy_name", "") or ""),
                        "timeframe": str(getattr(self.cfg, "timeframe", "") or ""),
                        "candidate_strategies": list(candidates),
                        "open_positions": self._gpt_open_positions_mtm_snapshot(max_trades=6),
                    },
                    system_prompt=(
                        "You are helping an automated options strategy choose an alternative trade after a safety block. "
                        "The original trade was blocked only because it would exceed the max consecutive same trade-type streak. "
                        "You MUST respond with ONLY valid JSON. Pick the best alternative from candidate_strategies and include it as alternative_trade. "
                        "Use TAKE when an alternative should be attempted, or SKIP only if none of the candidates should be taken now. "
                        "Output schema: {\"decision\":\"TAKE\"|\"SKIP\",\"alternative_trade\":string,\"reason\":string,\"confidence\":number}."
                    ),
                    max_tokens=180,
                )
            except Exception:
                advice = None

            try:
                gpt_reason = str(getattr(advice, "reason", "") or "").strip()
            except Exception:
                gpt_reason = ""

            extras = {}
            try:
                extras = getattr(advice, "extras", {}) or {}
            except Exception:
                extras = {}

            alt_raw = ""
            try:
                alt_raw = str(
                    extras.get("alternative_trade")
                    or extras.get("recommended_strategy")
                    or extras.get("trade_name")
                    or ""
                ).strip()
            except Exception:
                alt_raw = ""
            alt_name = self._normalize_strategy_name(alt_raw) if alt_raw else ""
            if alt_name in candidates:
                chosen = str(alt_name)
                source = "gpt"

        return {
            "trade_name": str(chosen),
            "candidates": list(candidates),
            "source": str(source),
            "reason": str(gpt_reason),
        }

    def _resolve_auto_strategy_candidate(self, candidate: object) -> str:
        raw = str(candidate or "").strip().lower().replace("-", "_").replace(" ", "_")
        if raw in {"long_call", "long_put", "short_call", "short_put"}:
            return raw
        if not raw:
            return ""
        return str(self._normalize_strategy_name(raw) or "")

    def _alternative_strategy_candidates_for_block(
        self,
        *,
        current_strategy: str,
        blocked_reason: str,
        attempted: Optional[set[str]] = None,
    ) -> List[str]:
        current_key = self._resolve_auto_strategy_candidate(current_strategy)
        attempted_keys = {
            self._resolve_auto_strategy_candidate(name)
            for name in set(attempted or set())
            if str(name or "").strip()
        }
        blocked_reason_l = str(blocked_reason or "").strip().lower()
        out: List[str] = []

        try:
            allowed = set(self._gpt_allowed_strategies_for_auto() or [])
        except Exception:
            allowed = set(self._supported_multi_trade_names()) | {"long_call", "long_put", "short_call", "short_put"}

        short_premium = {
            "short_straddle",
            "short_strangle",
            "bull_put_spread",
            "iron_condor",
            "iron_fly",
            "iron_butterfly",
            "delta_hedged_short_straddle",
        }
        long_vol = {
            "long_straddle",
            "long_strangle",
            "bull_call_spread",
            "call_ratio_backspread",
            "put_ratio_backspread",
            "delta_hedged_long_straddle",
        }
        directional = {"directional", "long_call", "long_put", "short_call", "short_put"}

        def push(candidate: object) -> None:
            cand = self._resolve_auto_strategy_candidate(candidate)
            if cand in {"", "auto", "directional"}:
                return
            if cand == current_key or cand in attempted_keys:
                return
            if allowed and cand not in allowed:
                return
            if cand not in out:
                out.append(cand)

        try:
            router_candidates = (self._last_router_snapshot or {}).get("candidates")
            if isinstance(router_candidates, list):
                for candidate in router_candidates:
                    push(candidate)
        except Exception:
            pass

        defaults: List[str]
        if current_key in short_premium or any(
            token in blocked_reason_l
            for token in ("premium", "rsi", "trend strength", "vwap deviation", "iv expanding")
        ):
            defaults = [
                "long_strangle",
                "long_straddle",
                "bull_call_spread",
                "call_ratio_backspread",
                "put_ratio_backspread",
                "delta_hedged_long_straddle",
                "bull_put_spread",
                "short_strangle",
                "iron_fly",
                "iron_condor",
            ]
        elif current_key in long_vol or "iv collapsing" in blocked_reason_l:
            defaults = [
                "iron_condor",
                "short_strangle",
                "iron_fly",
                "short_straddle",
                "bull_put_spread",
                "delta_hedged_short_straddle",
                "bull_call_spread",
                "call_ratio_backspread",
            ]
        elif current_key in directional or any(
            token in blocked_reason_l
            for token in (
                "directional",
                "adx",
                "volume",
                "supertrend",
                "bullish signal",
                "bearish signal",
                "ema+rsi",
                "atr ",
            )
        ):
            defaults = [
                "bull_call_spread",
                "bull_put_spread",
                "long_strangle",
                "long_straddle",
                "short_strangle",
                "iron_condor",
                "iron_fly",
                "call_ratio_backspread",
                "put_ratio_backspread",
            ]
        else:
            defaults = [str(name) for name in self._gpt_allowed_strategies_for_auto()]

        for candidate in defaults:
            push(candidate)

        return out

    def _suggest_alternative_strategy_for_block(
        self,
        *,
        blocked_strategy: str,
        blocked_reason: str,
        candidate_strategies: List[str],
    ) -> Optional[Dict[str, object]]:
        candidates = [
            self._resolve_auto_strategy_candidate(candidate)
            for candidate in list(candidate_strategies or [])
            if self._resolve_auto_strategy_candidate(candidate)
        ]
        if not candidates:
            return None

        chosen = str(candidates[0])
        source = "policy"
        gpt_reason = ""

        try:
            gpt_enabled = bool(getattr(self.cfg, "gpt_enable", False))
        except Exception:
            gpt_enabled = False

        if gpt_enabled:
            try:
                advice = self._gpt_control_advise(
                    proposal={
                        "request_type": "alternative_strategy_due_to_filter_block",
                        "blocked_trade_name": str(blocked_strategy or ""),
                        "blocked_reason": str(blocked_reason or ""),
                        "strategy_mode": str(getattr(self.cfg, "strategy_name", "") or ""),
                        "timeframe": str(getattr(self.cfg, "timeframe", "") or ""),
                        "candidate_strategies": list(candidates),
                        "open_positions": self._gpt_open_positions_mtm_snapshot(max_trades=6),
                    },
                    system_prompt=(
                        "You are helping an automated options strategy recover from a strategy-specific filter block. "
                        "The original strategy was valid in auto mode but failed a runtime entry filter. "
                        "You MUST respond with ONLY valid JSON. Pick the best executable alternative from candidate_strategies and include it as alternative_trade. "
                        "Use TAKE when an alternative should be attempted, or SKIP only if none of the candidates should be taken now. "
                        "Never repeat the blocked strategy. Output schema: {\"decision\":\"TAKE\"|\"SKIP\",\"alternative_trade\":string,\"reason\":string,\"confidence\":number}."
                    ),
                    max_tokens=220,
                )
            except Exception:
                advice = None

            try:
                gpt_reason = str(getattr(advice, "reason", "") or "").strip()
            except Exception:
                gpt_reason = ""

            decision = ""
            try:
                decision = str(getattr(advice, "decision", "") or "").strip().upper()
            except Exception:
                decision = ""

            extras = {}
            try:
                extras = getattr(advice, "extras", {}) or {}
            except Exception:
                extras = {}

            alt_raw = ""
            try:
                alt_raw = str(
                    extras.get("alternative_trade")
                    or extras.get("recommended_strategy")
                    or extras.get("trade_name")
                    or ""
                ).strip()
            except Exception:
                alt_raw = ""
            alt_name = self._resolve_auto_strategy_candidate(alt_raw) if alt_raw else ""
            if alt_name in candidates:
                chosen = str(alt_name)
                source = "gpt"
            elif decision == "SKIP":
                return None

        return {
            "trade_name": str(chosen),
            "candidates": list(candidates),
            "source": str(source),
            "reason": str(gpt_reason),
        }

    def _blocked_auto_strategy_reroute(
        self,
        *,
        blocked_strategy: str,
        blocked_reason: str,
        attempted_strategies: Optional[set[str]] = None,
    ) -> Optional[Dict[str, object]]:
        # Fail-fast: option-chain unavailable reasons do not benefit from rerouting.
        reason_lower = str(blocked_reason or "").lower()
        option_chain_fail_keywords = (
            "option chain empty",
            "option chain parameters are not configured",
            "missing_config",
            "ip_mismatch",
            "ip mismatch",
        )
        if any(kw in reason_lower for kw in option_chain_fail_keywords):
            print("[AUTO][REROUTE] blocked: option chain unavailable, skipping all fallback strategies")
            return None

        # Cooldown guard: suppress reroutes within the cooldown window.
        now_ts = float(time.time())
        if self._last_reroute_ts > 0.0 and (now_ts - self._last_reroute_ts) < self._reroute_cooldown_seconds:
            return None

        # Max-attempts-per-cycle guard.
        if self._reroute_count_this_cycle >= self._max_reroutes_per_cycle:
            print("[AUTO][REROUTE] stopped: max attempts per cycle reached")
            return None

        blocked_key = self._resolve_auto_strategy_candidate(blocked_strategy)
        candidates = self._alternative_strategy_candidates_for_block(
            current_strategy=str(blocked_key or blocked_strategy),
            blocked_reason=str(blocked_reason or ""),
            attempted=attempted_strategies,
        )
        if not candidates:
            return None

        suggestion = self._suggest_alternative_strategy_for_block(
            blocked_strategy=str(blocked_strategy or blocked_key),
            blocked_reason=str(blocked_reason or ""),
            candidate_strategies=list(candidates),
        )
        if not isinstance(suggestion, dict):
            return None

        alt_name = self._resolve_auto_strategy_candidate(suggestion.get("trade_name"))
        attempted_keys = {
            self._resolve_auto_strategy_candidate(name)
            for name in set(attempted_strategies or set())
            if str(name or "").strip()
        }
        if not alt_name or alt_name == blocked_key or alt_name in attempted_keys:
            return None

        # Commit the reroute: bump counter and timestamp.
        self._reroute_count_this_cycle += 1
        self._last_reroute_ts = now_ts

        return {
            "trade_name": str(alt_name),
            "candidates": list(candidates),
            "source": str(suggestion.get("source") or "policy"),
            "reason": str(suggestion.get("reason") or ""),
        }

    def _get_pivot_points(self) -> Optional[Dict[str, float]]:
        """Calculate Pivot Points using previous day's HLC."""
        try:
            # Fetch 1-day candles (limit 2 to get previous day)
            # timeframe "ONE_DAY" or similar might be needed. 
            # Check mstock_client.py for supported daily timeframe string.
            # Usually "ONE_DAY" or "DAILY".
            candles, _tf = self.client.fetch_index_candles(
                self.cfg.underlying_token or "26000", 
                exchange=getattr(self.cfg, "underlying_exchange", "NSE") or "NSE", 
                timeframe="ONE_DAY", 
                limit=2
            )
            if candles and len(candles) >= 1:
                # candles[0] is previous day if limit=2 and we are currently in a new day
                # or if we fetch exactly the last completed daily candle.
                prev = candles[-1]
                return pivot_points(prev.high, prev.low, prev.close)
        except Exception as exc:
            print(f"[PIVOTS] Failed: {exc}")
        return None

    def _pivot_target_level(
        self,
        pivots: Dict[str, float],
        *,
        direction: str,
        entry_spot: float,
        mtm_val: Optional[float],
    ) -> Optional[Tuple[float, str]]:
        """Return pivot-based target level and key.

        Fixes immediate-close behavior by choosing the *next* pivot level beyond
        the entry spot:
        - Long: nearest of R1/R2/R3 that is strictly above entry spot
        - Short: nearest of S1/S2/S3 that is strictly below entry spot

        Also makes pivot target less aggressive during MTM drawdown by skipping
        pivot-target exits when MTM is negative.
        """

        if not pivots:
            return None

        if mtm_val is not None:
            try:
                if float(mtm_val) < 0:
                    return None
            except Exception:
                pass

        try:
            entry_spot_f = float(entry_spot)
        except Exception:
            return None

        d = str(direction or "").strip().lower()

        if d in {"long", "bull", "buy"}:
            levels = [("r1", pivots.get("r1")), ("r2", pivots.get("r2")), ("r3", pivots.get("r3"))]
            candidates: List[Tuple[float, str]] = []
            for k, v in levels:
                if v is None:
                    continue
                try:
                    vf = float(v)
                except Exception:
                    continue
                if vf > entry_spot_f:
                    candidates.append((vf, k))
            if not candidates:
                return None
            vf, k = min(candidates, key=lambda x: x[0])
            return float(vf), str(k)

        if d in {"short", "bear", "sell"}:
            levels = [("s1", pivots.get("s1")), ("s2", pivots.get("s2")), ("s3", pivots.get("s3"))]
            candidates = []
            for k, v in levels:
                if v is None:
                    continue
                try:
                    vf = float(v)
                except Exception:
                    continue
                if vf < entry_spot_f:
                    candidates.append((vf, k))
            if not candidates:
                return None
            vf, k = max(candidates, key=lambda x: x[0])
            return float(vf), str(k)

        return None

    def _can_open_trade_type(self, *, position_type: str, name: str) -> bool:
        try:
            max_streak = int(getattr(self.cfg, "max_consecutive_same_trade_type", 0) or 0)
        except Exception:
            max_streak = 0
        if max_streak <= 0:
            return True

        key = self._trade_type_key(position_type=position_type, name=name)
        if self._last_trade_type_key == key and int(self._last_trade_type_streak) >= int(max_streak):
            reason = (
                f"[SAFETY] Blocked entry: would exceed max consecutive same trade type "
                f"({max_streak}) for {key}"
            )
            extras: Dict[str, object] = {
                "blocked_trade_key": str(key),
                "blocked_trade_name": str(name),
                "max_same_trade_type_streak": int(max_streak),
            }
            suggestion = self._suggest_alternative_trade_type(position_type=position_type, name=name)
            if isinstance(suggestion, dict):
                alt_name = str(suggestion.get("trade_name") or "").strip()
                if alt_name:
                    extras["suggested_trade"] = str(alt_name)
                    extras["suggestion_source"] = str(suggestion.get("source") or "policy")
                candidates = suggestion.get("candidates")
                if isinstance(candidates, list):
                    extras["alternative_candidates"] = list(candidates)
                alt_reason = str(suggestion.get("reason") or "").strip()
                if alt_reason:
                    extras["suggestion_reason"] = str(alt_reason)
            self._note_entry_blocked(reason, extras=extras)
            return False
        return True

    def _dispatch_multi_strategy_entry(
        self,
        *,
        strategy_name: str,
        chain: List[dict],
        spot: float,
        atr_val: Optional[float],
        allow_alternative_retry: bool,
    ) -> bool:
        strategy_key = self._normalize_strategy_name(strategy_name)
        handlers = {
            "short_straddle": self._enter_short_straddle,
            "short_strangle": self._enter_short_strangle,
            "bull_call_spread": self._enter_bull_call_spread,
            "bull_put_spread": self._enter_bull_put_spread,
            "call_ratio_backspread": self._enter_call_ratio_backspread,
            "put_ratio_backspread": self._enter_put_ratio_backspread,
            "iron_condor": self._enter_iron_condor,
            "iron_fly": self._enter_iron_fly,
            "iron_butterfly": self._enter_iron_butterfly,
            "long_straddle": self._enter_long_straddle,
            "long_strangle": self._enter_long_strangle,
            "delta_hedged_short_straddle": self._enter_delta_hedged_short_straddle,
            "delta_hedged_long_straddle": self._enter_delta_hedged_long_straddle,
        }
        handler = handlers.get(strategy_key)
        if handler is None:
            return False
        handler(chain, spot, atr_val=atr_val, allow_alternative_retry=allow_alternative_retry)
        return True

    def _can_open_or_reroute_multi_trade(
        self,
        *,
        base_name: str,
        trade_name: str,
        chain: List[dict],
        spot: float,
        atr_val: Optional[float],
        allow_alternative_retry: bool,
    ) -> bool:
        if self._can_open_trade_type(position_type="multi", name=str(trade_name)):
            return True
        if not allow_alternative_retry:
            return False

        try:
            strategy_mode = self._normalize_strategy_name(str(getattr(self.cfg, "strategy_name", "") or ""))
        except Exception:
            strategy_mode = ""
        if strategy_mode != "auto":
            return False

        suggestion = self._suggest_alternative_trade_type(position_type="multi", name=str(trade_name))
        if not isinstance(suggestion, dict):
            return False
        alt_name = self._normalize_strategy_name(str(suggestion.get("trade_name") or "").strip())
        if not alt_name or alt_name == self._base_trade_name(base_name):
            return False

        source = str(suggestion.get("source") or "policy").strip() or "policy"
        why = str(suggestion.get("reason") or "").strip()
        suffix = f" reason={why}" if why else ""
        print(f"[SAFETY][ALT] {trade_name} blocked; trying {alt_name} ({source}){suffix}")
        return self._dispatch_multi_strategy_entry(
            strategy_name=str(alt_name),
            chain=chain,
            spot=float(spot),
            atr_val=atr_val,
            allow_alternative_retry=False,
        )

    def _note_opened_trade_type(self, *, position_type: str, name: str) -> None:
        key = self._trade_type_key(position_type=position_type, name=name)
        if self._last_trade_type_key == key:
            self._last_trade_type_streak = int(self._last_trade_type_streak) + 1
        else:
            self._last_trade_type_key = key
            self._last_trade_type_streak = 1

        try:
            nm = str(name or "").strip().lower()
            if nm:
                self._strategy_selected_counts[nm] = int(self._strategy_selected_counts.get(nm, 0)) + 1
        except Exception:
            pass

    def _infer_block_code(self, reason: str) -> str:
        s = str(reason or "").strip().lower()
        if not s:
            return "unknown"
        rules = [
            ("max trades per day", "max_trades_day"),
            ("max consecutive same trade type", "same_trade_type_streak"),
            ("same-type streak", "same_trade_type_streak"),
            ("cooldown after stopout", "cooldown_stopout"),
            ("cooldown", "cooldown"),
            ("max open positions", "max_open_positions"),
            ("entry cutoff", "entry_cutoff"),
            ("warming up", "warmup"),
            ("stale", "data_stale"),
            ("spot ltp unavailable", "spot_unavailable"),
            ("candle fetch failed", "candle_fetch_failed"),
            ("atr", "atr_filter"),
            ("rsi", "rsi_filter"),
            ("vwap", "vwap_filter"),
            ("trend strength", "trend_filter"),
            ("gap too large", "gap_filter"),
            ("range too large", "range_filter"),
            ("option chain empty", "chain_empty"),
            ("premium entries disabled", "premium_cutoff"),
            ("liquidity", "liquidity"),
            ("spread too wide", "liquidity"),
            ("gpt", "gpt_gate"),
            ("risk", "risk_limit"),
            ("portfolio", "portfolio_limit"),
        ]
        for token, code in rules:
            if token in s:
                return code
        return "unknown"

    def _note_entry_blocked(self, reason: str, extras: Optional[Dict[str, object]] = None) -> None:
        code = self._infer_block_code(reason)
        self._entry_block_counts[code] = int(self._entry_block_counts.get(code, 0)) + 1
        payload: Dict[str, object] = {"code": str(code), "reason": str(reason), "ts": float(time.time())}
        if isinstance(extras, dict):
            try:
                payload.update({str(k): v for k, v in extras.items()})
            except Exception:
                pass
        self._last_entry_block = payload
        print(f"[ENTRY BLOCKED] {reason}")

    def _note_strategy_considered(self, names: List[str]) -> None:
        for n in names:
            try:
                key = str(n or "").strip().lower()
                if not key:
                    continue
                self._strategy_considered_counts[key] = int(self._strategy_considered_counts.get(key, 0)) + 1
            except Exception:
                continue

    def _note_strategy_decision(self, name: str) -> None:
        try:
            key = str(name or "").strip().lower()
            if not key:
                return
            self._strategy_decision_counts[key] = int(self._strategy_decision_counts.get(key, 0)) + 1
        except Exception:
            return

    def _place_order_with_retry(self, **kwargs: object) -> Order:
        if self._kill_switch_active():
            raise RuntimeError("[KILL_SWITCH] Active — live orders blocked")

        # =====================================================================
        # PHASE 6: Router + live enablement gate at the actual order choke point.
        # No bypass allowed. Checked on every place_order attempt.
        # =====================================================================
        router_ok = bool(getattr(self, "_last_router_allowed", True))
        router_dec = getattr(self, "_last_router_decision", None) or {}
        live_orders_env_ok = str(os.getenv("MSTOCK_ENABLE_LIVE_ORDERS", "")).strip().lower() in {"1", "true", "yes"}
        cfg_live_ok = bool(getattr(self.cfg, "enable_live_trading", False))
        if router_dec and router_dec.get("final_signal", "NO_TRADE") == "NO_TRADE":
            router_ok = False

        if (not router_ok) or (not (live_orders_env_ok or cfg_live_ok)):
            LOGGER.info(
                "[LIVE-GATE] candidate_id=%s preset=%s side=%s final_signal=%s shadow_ready=%s "
                "live_orders_enabled=%s order_allowed=%s reason=%s",
                router_dec.get("candidate_id", "") if router_dec else "",
                router_dec.get("selected_preset", "") if router_dec else "",
                router_dec.get("side_decision", "") if router_dec else "",
                router_dec.get("final_signal", "NO_TRADE") if router_dec else "NO_TRADE",
                router_dec.get("shadow_ready", False) if router_dec else False,
                live_orders_env_ok,
                False,
                "router_or_mstock_enable_live_orders_block",
            )
            if not (live_orders_env_ok or cfg_live_ok):
                try:
                    self.cfg.enable_live_trading = False
                except Exception:
                    pass
            raise RuntimeError("ORDER_BLOCKED_BY_ROUTER_OR_LIVE_GATE")

        # BLOCKER 6: explicit paper forward guard before any broker order
        if router_dec and (router_dec.get("paper_forward_only") or not router_dec.get("live_orders_enabled", False) or not router_dec.get("broker_orders_enabled", False)):
            LOGGER.info(
                "[ORDER-GUARD] paper_forward_only=true live_orders_enabled=false broker_orders_enabled=false action=paper_state_only"
            )
            raise RuntimeError("ORDER_BLOCKED_PAPER_FORWARD_ONLY")

        attempts = 1
        delay = 0.0
        try:
            attempts = max(1, int(getattr(self.cfg, "order_retry_attempts", 2) or 2))
        except Exception:
            attempts = 1
        try:
            delay = max(0.0, float(getattr(self.cfg, "order_retry_delay_sec", 0.25) or 0.25))
        except Exception:
            delay = 0.0

        last_exc: Optional[Exception] = None
        for i in range(attempts):
            try:
                return self.client.place_order(**kwargs)
            except Exception as exc:
                last_exc = exc
                if i >= attempts - 1:
                    break
                if delay > 0:
                    time.sleep(delay)
        raise RuntimeError(f"place_order failed after {attempts} attempts: {last_exc}")

    @staticmethod
    def _round_limit_price(price: Optional[float]) -> Optional[float]:
        try:
            px = float(price) if price is not None else None
        except Exception:
            px = None
        if px is None or px <= 0.0:
            return None
        return round(px * 20.0) / 20.0

    def _route_midpoint_pegged_limit_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: int,
        exchange: Optional[str],
        symbol_token: Optional[str],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        routing_meta = meta if isinstance(meta, dict) else {}
        fsm = None
        if getattr(self, "reconciliation_engine", None) is not None:
            try:
                fsm = self.reconciliation_engine.register_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    metadata={
                        "symbol_token": symbol_token,
                        "exchange": exchange,
                        "order_routing_style": "midpoint_pegged_limit",
                    },
                )
            except Exception:
                fsm = None
        friction = evaluate_live_execution_friction(
            {
                "client": self.client,
                "symbol": symbol,
                "exchange": exchange,
                "cost_model": CostModel(),
                "enforce_cost_boundary": bool(getattr(self.cfg, "enforce_cost_boundary", False)),
            }
        )
        routing_meta["live_bid_ask_spread_pct"] = float(friction.get("relative_spread_pct") or 0.0)
        routing_meta["order_routing_style"] = "midpoint_pegged_limit"
        routing_meta["execution_friction_status"] = str(friction.get("status") or "")
        routing_meta["execution_friction_reason"] = str(friction.get("reason") or "")
        routing_meta["initial_midpoint_at_signal"] = friction.get("midpoint")
        routing_meta["unfilled_ticks"] = 0
        routing_meta["realized_slippage_pct"] = 0.0

        if str(friction.get("status") or "") == "REJECT_LIQUIDITY_CEILING":
            return {
                "ok": False,
                "status": "REJECT_LIQUIDITY_CEILING",
                "reason": str(friction.get("reason") or "live_friction_rejected"),
                "meta": routing_meta,
                "fsm": fsm,
            }

        limit_price = self._round_limit_price(friction.get("midpoint"))
        if limit_price is None:
            return {
                "ok": False,
                "status": "REJECT_LIQUIDITY_CEILING",
                "reason": "invalid_midpoint_price",
                "meta": routing_meta,
                "fsm": fsm,
            }

        try:
            chase_reset_ticks = max(1, int(getattr(self.cfg, "midpoint_router_chase_reset_ticks", 5) or 5))
        except Exception:
            chase_reset_ticks = 5
        try:
            max_unfilled_ticks = max(chase_reset_ticks, int(getattr(self.cfg, "midpoint_router_timeout_ticks", 15) or 15))
        except Exception:
            max_unfilled_ticks = 15
        try:
            tick_sleep_sec = max(0.0, float(getattr(self.cfg, "midpoint_router_tick_sleep_sec", 0.25) or 0.25))
        except Exception:
            tick_sleep_sec = 0.25

        order: Optional[Order] = None
        latest_friction = dict(friction)
        for unfilled_ticks in range(max_unfilled_ticks + 1):
            routing_meta["unfilled_ticks"] = int(unfilled_ticks)
            if order is None or unfilled_ticks == 0 or (unfilled_ticks % chase_reset_ticks == 0 and unfilled_ticks < max_unfilled_ticks):
                if order is not None and getattr(order, "order_id", ""):
                    if fsm is not None:
                        try:
                            self.reconciliation_engine.mark_cancel_requested(fsm)
                        except Exception:
                            pass
                    try:
                        self.client.cancel_order(str(order.order_id))
                        if fsm is not None:
                            try:
                                self.reconciliation_engine.resolve_cancel_result(fsm=fsm, cancel_succeeded=True)
                            except Exception:
                                pass
                    except Exception:
                        if fsm is not None:
                            try:
                                self.reconciliation_engine.resolve_cancel_result(
                                    fsm=fsm,
                                    cancel_succeeded=False,
                                    fill_qty=int(quantity),
                                    fill_price=float(limit_price or 0.0),
                                    unsolicited=True,
                                )
                            except Exception:
                                pass
                        routing_meta["cancel_replace_latency_ms"] = fsm.cancel_replace_latency_ms if fsm is not None else None
                        routing_meta["unsolicited_fill_occurred"] = True
                        routing_meta["reconciliation_status_code"] = "LATE_FILL_ABORT"
                        routing_meta["realized_slippage_pct"] = abs(float(limit_price or 0.0) - float(routing_meta.get("initial_midpoint_at_signal") or limit_price or 1.0)) / max(float(routing_meta.get("initial_midpoint_at_signal") or limit_price or 1.0), 1e-9)
                        return {
                            "ok": True,
                            "status": "late_fill_abort",
                            "order": order,
                            "fill_price": float(limit_price or 0.0),
                            "meta": routing_meta,
                            "fsm": fsm,
                        }
                latest_friction = evaluate_live_execution_friction(
                    {
                        "client": self.client,
                        "symbol": symbol,
                        "exchange": exchange,
                        "cost_model": CostModel(),
                        "enforce_cost_boundary": bool(getattr(self.cfg, "enforce_cost_boundary", False)),
                    }
                )
                routing_meta["live_bid_ask_spread_pct"] = float(latest_friction.get("relative_spread_pct") or 0.0)
                routing_meta["execution_friction_status"] = str(latest_friction.get("status") or "")
                routing_meta["execution_friction_reason"] = str(latest_friction.get("reason") or "")
                if str(latest_friction.get("status") or "") == "REJECT_LIQUIDITY_CEILING":
                    return {
                        "ok": False,
                        "status": "REJECT_LIQUIDITY_CEILING",
                        "reason": str(latest_friction.get("reason") or "spread_widened_during_chase"),
                        "meta": routing_meta,
                    }
                limit_price = self._round_limit_price(latest_friction.get("midpoint"))
                if limit_price is None:
                    return {
                        "ok": False,
                        "status": "REJECT_LIQUIDITY_CEILING",
                        "reason": "invalid_midpoint_during_chase",
                        "meta": routing_meta,
                    }
                order = self._place_order_with_retry(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type="LIMIT",
                    price=limit_price,
                    exchange=exchange,
                    symbol_token=symbol_token,
                )
                if fsm is not None:
                    try:
                        self.reconciliation_engine.bind_order_id(fsm, str(getattr(order, "order_id", "") or ""))
                    except Exception:
                        pass
                # Broker fill polling is not yet exposed, so treat the newly routed limit as filled at midpoint.
                fill_price = float(limit_price)
                initial_mid = float(routing_meta.get("initial_midpoint_at_signal") or limit_price)
                routing_meta["realized_slippage_pct"] = abs(fill_price - initial_mid) / max(initial_mid, 1e-9)
                routing_meta["final_limit_price"] = float(limit_price)
                routing_meta["broker_order_id"] = str(getattr(order, "order_id", "") or "")
                if fsm is not None:
                    try:
                        self.reconciliation_engine.resolve_cancel_result(
                            fsm=fsm,
                            cancel_succeeded=False,
                            fill_qty=int(quantity),
                            fill_price=float(fill_price),
                            unsolicited=False,
                        )
                    except Exception:
                        pass
                    routing_meta["cancel_replace_latency_ms"] = fsm.cancel_replace_latency_ms
                    routing_meta["unsolicited_fill_occurred"] = bool(fsm.unsolicited_fill_occurred)
                    routing_meta["reconciliation_status_code"] = str(fsm.reconciliation_status_code or "")
                return {
                    "ok": True,
                    "status": "filled",
                    "order": order,
                    "fill_price": float(fill_price),
                    "meta": routing_meta,
                    "fsm": fsm,
                }
            if tick_sleep_sec > 0.0:
                time.sleep(tick_sleep_sec)

        if order is not None and getattr(order, "order_id", ""):
            late_fill = False
            late_fill_price = None
            if fsm is not None:
                try:
                    self.reconciliation_engine.mark_cancel_requested(fsm)
                except Exception:
                    pass
            try:
                self.client.cancel_order(str(order.order_id))
                if fsm is not None:
                    try:
                        self.reconciliation_engine.resolve_cancel_result(fsm=fsm, cancel_succeeded=True)
                    except Exception:
                        pass
            except Exception:
                late_fill = True
                late_fill_price = limit_price
                if fsm is not None:
                    try:
                        self.reconciliation_engine.resolve_cancel_result(
                            fsm=fsm,
                            cancel_succeeded=False,
                            fill_qty=int(quantity),
                            fill_price=float(limit_price or 0.0),
                            unsolicited=True,
                        )
                    except Exception:
                        pass
            if late_fill:
                routing_meta["cancel_replace_latency_ms"] = fsm.cancel_replace_latency_ms if fsm is not None else None
                routing_meta["unsolicited_fill_occurred"] = True
                routing_meta["reconciliation_status_code"] = "LATE_FILL_ABORT"
                routing_meta["realized_slippage_pct"] = abs(float(late_fill_price or 0.0) - float(routing_meta.get("initial_midpoint_at_signal") or late_fill_price or 1.0)) / max(float(routing_meta.get("initial_midpoint_at_signal") or late_fill_price or 1.0), 1e-9)
                return {
                    "ok": True,
                    "status": "late_fill_abort",
                    "order": order,
                    "fill_price": float(late_fill_price or 0.0),
                    "meta": routing_meta,
                    "fsm": fsm,
                }
        return {
            "ok": False,
            "status": "CANCEL_UNFILLED_TIMEOUT",
            "reason": "midpoint_pegged_limit_timeout",
            "meta": routing_meta,
            "fsm": fsm,
        }

    def _build_bracket_levels(
        self,
        *,
        entry_price: Optional[float],
        side: str,
        quantity: int = 0,
        risk_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, object]:
        try:
            px = float(entry_price) if entry_price is not None else None
        except Exception:
            px = None
        if px is None or px <= 0:
            return {}

        side_u = str(side or "").strip().upper()
        if side_u not in {"BUY", "SELL"}:
            return {}

        override = risk_overrides if isinstance(risk_overrides, dict) else {}

        try:
            stop_pct = float(
                override.get("stop_loss_pct")
                if override.get("stop_loss_pct") is not None
                else getattr(self.cfg, "premium_mtm_stop_pct", 0.30) or 0.30
            )
        except Exception:
            stop_pct = 0.30
        try:
            target_pct = float(
                override.get("target_pct")
                if override.get("target_pct") is not None
                else getattr(self.cfg, "premium_mtm_target_pct", 0.18) or 0.18
            )
        except Exception:
            target_pct = 0.18
        try:
            trail_start_pct = float(getattr(self.cfg, "premium_mtm_trail_start_pct", 0.05) or 0.05)
        except Exception:
            trail_start_pct = 0.05
        try:
            trail_stop_pct = float(
                override.get("trailing_sl_pct")
                if override.get("trailing_sl_pct") is not None
                else getattr(self.cfg, "premium_mtm_trail_stop_pct", 0.05) or 0.05
            )
        except Exception:
            trail_stop_pct = 0.05

        if side_u == "BUY":
            prem_stop = max(0.0, float(px) * (1.0 - float(stop_pct)))
            prem_target = max(0.0, float(px) * (1.0 + float(target_pct)))
        else:
            prem_stop = max(0.0, float(px) * (1.0 + float(stop_pct)))
            prem_target = max(0.0, float(px) * (1.0 - float(target_pct)))

        return {
            "prem_stop": float(prem_stop),
            "prem_target": float(prem_target),
            "trail_start": float(trail_start_pct),
            "trail_stop": float(trail_stop_pct),
            "stoploss": float(prem_stop),
            "targetPrice": float(prem_target),
            "trailingStopLoss": float(trail_stop_pct),
            "bracket_enabled": True,
            "entry_price": float(px),
            "quantity": int(quantity),
        }

    def _annotate_leg_brackets(self, leg: Dict[str, object], risk_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, object]:
        leg_copy = dict(leg)
        bracket = self._build_bracket_levels(
            entry_price=leg_copy.get("entry_price"),
            side=str(leg_copy.get("side") or ""),
            quantity=int(leg_copy.get("quantity") or 0),
            risk_overrides=risk_overrides,
        )
        if bracket:
            for key in ("prem_stop", "prem_target", "trail_start", "trail_stop", "stoploss", "targetPrice", "trailingStopLoss"):
                if key in bracket:
                    leg_copy[key] = bracket[key]
            leg_copy["bracket"] = {k: v for k, v in bracket.items() if k != "quantity"}
        return leg_copy

    def _portfolio_risk_snapshot(self, spot: float) -> Dict[str, object]:
        """Approximate portfolio risk from open option legs."""
        out: Dict[str, object] = {
            "option_notional_abs": 0.0,
            "delta_abs": 0.0,
            "delta_net": 0.0,
            "gamma_net": 0.0,
            "vega_net": 0.0,
            "legs_count": 0,
        }
        if spot <= 0:
            return out

        legs: List[dict] = []
        try:
            for tr in list(self.state.open_multi or []):
                if isinstance(tr, dict) and isinstance(tr.get("legs"), list):
                    legs.extend([l for l in tr.get("legs", []) if isinstance(l, dict)])
            for tr in list(self.state.open_directional or []):
                if isinstance(tr, dict) and isinstance(tr.get("legs"), list):
                    legs.extend([l for l in tr.get("legs", []) if isinstance(l, dict)])
        except Exception:
            pass

        today = date.today()
        for lg in legs:
            if bool(lg.get("is_hedge")):
                continue
            side = str(lg.get("side") or "").strip().upper()
            if side not in {"BUY", "SELL"}:
                continue
            try:
                qty = int(lg.get("quantity") or 0)
            except Exception:
                qty = 0
            if qty <= 0:
                continue
            sign = 1.0 if side == "BUY" else -1.0

            ltp = None
            try:
                v = lg.get("ltp")
                if v is not None:
                    ltp = float(v)
            except Exception:
                ltp = None
            if ltp is None:
                try:
                    ep = lg.get("entry_price")
                    if ep is not None:
                        ltp = float(ep)
                except Exception:
                    ltp = None
            if ltp is not None and ltp > 0:
                out["option_notional_abs"] = float(out["option_notional_abs"]) + abs(float(ltp) * float(qty))

            try:
                strike = float(lg.get("strike") or 0.0)
            except Exception:
                strike = 0.0
            if strike <= 0:
                continue
            ot = str(lg.get("option_type") or "").strip().upper()
            if ot in {"CALL", "C"}:
                ot = "CE"
            elif ot in {"PUT", "P"}:
                ot = "PE"
            if ot not in {"CE", "PE"}:
                continue

            iv_val = None
            try:
                iv_raw = lg.get("iv")
                if iv_raw is not None:
                    iv_f = float(iv_raw)
                    iv_val = (iv_f / 100.0) if iv_f > 1.5 else iv_f
            except Exception:
                iv_val = None
            if iv_val is None or iv_val <= 0:
                iv_val = 0.20

            t_years = 1.0 / 252.0
            try:
                exp_raw = lg.get("expiry")
                exp_d = self._parse_any_date(str(exp_raw)) if exp_raw else None
                if exp_d is not None:
                    dte = max(1, int((exp_d - today).days))
                    t_years = float(dte) / 365.0
            except Exception:
                t_years = 1.0 / 252.0

            try:
                d = float(delta(float(spot), strike, float(t_years), 0.06, float(iv_val), ot)) * float(sign) * float(qty)
                g = float(gamma(float(spot), strike, float(t_years), 0.06, float(iv_val))) * float(sign) * float(qty)
                v = float(vega(float(spot), strike, float(t_years), 0.06, float(iv_val))) * float(sign) * float(qty)
            except Exception:
                continue

            out["delta_net"] = float(out["delta_net"]) + d
            out["delta_abs"] = float(out["delta_abs"]) + abs(d)
            out["gamma_net"] = float(out["gamma_net"]) + g
            out["vega_net"] = float(out["vega_net"]) + v
            out["legs_count"] = int(out["legs_count"]) + 1

        return out

    def _projected_portfolio_risk_snapshot(self, spot: float, new_legs: List[dict]) -> Dict[str, object]:
        """Return the portfolio risk snapshot after hypothetically adding legs.

        This is used as a pre-trade gate so the bot can block orders that would
        breach configured portfolio caps before the broker sees them.
        """

        base = self._portfolio_risk_snapshot(spot)
        projected = dict(base)
        if spot <= 0:
            return projected

        extra_notional = 0.0
        extra_delta_abs = 0.0
        extra_delta_net = 0.0
        extra_gamma_net = 0.0
        extra_vega_net = 0.0
        legs_count = int(projected.get("legs_count") or 0)

        today = date.today()
        for lg in list(new_legs or []):
            if not isinstance(lg, dict) or bool(lg.get("is_hedge")):
                continue

            side = str(lg.get("side") or "").strip().upper()
            if side not in {"BUY", "SELL"}:
                continue

            try:
                qty = int(lg.get("quantity") or 0)
            except Exception:
                qty = 0
            if qty <= 0:
                continue

            sign = 1.0 if side == "BUY" else -1.0

            try:
                ltp = float(lg.get("ltp") or lg.get("entry_price") or 0.0)
            except Exception:
                ltp = 0.0
            if ltp > 0:
                extra_notional += abs(float(ltp) * float(qty))

            try:
                strike = float(lg.get("strike") or 0.0)
            except Exception:
                strike = 0.0
            if strike <= 0:
                continue

            ot = str(lg.get("option_type") or "").strip().upper()
            if ot in {"CALL", "C"}:
                ot = "CE"
            elif ot in {"PUT", "P"}:
                ot = "PE"
            if ot not in {"CE", "PE"}:
                continue

            iv_val = None
            try:
                iv_raw = lg.get("iv")
                if iv_raw is not None:
                    iv_f = float(iv_raw)
                    iv_val = (iv_f / 100.0) if iv_f > 1.5 else iv_f
            except Exception:
                iv_val = None
            if iv_val is None or iv_val <= 0:
                iv_val = 0.20

            t_years = 1.0 / 252.0
            try:
                exp_raw = lg.get("expiry")
                exp_d = self._parse_any_date(str(exp_raw)) if exp_raw else None
                if exp_d is not None:
                    dte = max(1, int((exp_d - today).days))
                    t_years = float(dte) / 365.0
            except Exception:
                t_years = 1.0 / 252.0

            try:
                d = float(delta(float(spot), strike, float(t_years), 0.06, float(iv_val), ot)) * float(sign) * float(qty)
                g = float(gamma(float(spot), strike, float(t_years), 0.06, float(iv_val))) * float(sign) * float(qty)
                v = float(vega(float(spot), strike, float(t_years), 0.06, float(iv_val))) * float(sign) * float(qty)
            except Exception:
                continue

            extra_delta_net += d
            extra_delta_abs += abs(d)
            extra_gamma_net += g
            extra_vega_net += v
            legs_count += 1

        projected["option_notional_abs"] = float(projected.get("option_notional_abs") or 0.0) + float(extra_notional)
        projected["delta_abs"] = float(projected.get("delta_abs") or 0.0) + float(extra_delta_abs)
        projected["delta_net"] = float(projected.get("delta_net") or 0.0) + float(extra_delta_net)
        projected["gamma_net"] = float(projected.get("gamma_net") or 0.0) + float(extra_gamma_net)
        projected["vega_net"] = float(projected.get("vega_net") or 0.0) + float(extra_vega_net)
        projected["legs_count"] = int(legs_count)
        return projected

    def _portfolio_caps_allow_legs(self, spot: float, legs: List[dict], *, context: str = "") -> bool:
        """Block trades that would breach portfolio caps after adding `legs`."""

        try:
            max_notional = float(getattr(self.cfg, "max_portfolio_option_notional", 0.0) or 0.0)
        except Exception:
            max_notional = 0.0
        try:
            max_delta_abs = float(getattr(self.cfg, "max_portfolio_delta_abs", 0.0) or 0.0)
        except Exception:
            max_delta_abs = 0.0
        if max_notional <= 0 and max_delta_abs <= 0:
            return True

        projected = self._projected_portfolio_risk_snapshot(float(spot), list(legs or []))
        notional_abs = float(projected.get("option_notional_abs") or 0.0)
        delta_abs_now = float(projected.get("delta_abs") or 0.0)

        if max_notional > 0 and notional_abs >= max_notional:
            print(
                f"[RISK] Portfolio option notional cap would be exceeded after {context or 'trade'} "
                f"({notional_abs:.0f} >= {max_notional:.0f})"
            )
            return False
        if max_delta_abs > 0 and delta_abs_now >= max_delta_abs:
            print(
                f"[RISK] Portfolio abs-delta cap would be exceeded after {context or 'trade'} "
                f"({delta_abs_now:.1f} >= {max_delta_abs:.1f})"
            )
            return False
        return True

    def _strategy_router(
        self,
        *,
        chain_available: bool,
        range_bound: bool,
        trend_strength: float,
        ts_limit: float,
        atr_val: float,
        atr_threshold: float,
        fast_ema: Optional[float],
        slow_ema: Optional[float],
        regime: Optional[str] = None,
        regime_profile: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Deterministic policy router used before/alongside GPT."""
        mode = str(getattr(self.cfg, "strategy_router_mode", "balanced") or "balanced").strip().lower()
        if mode not in {"conservative", "balanced", "aggressive"}:
            mode = "balanced"

        chosen = self._auto_select_effective_strategy(
            chain_available=bool(chain_available),
            range_bound=bool(range_bound),
            trend_strength=float(trend_strength),
            ts_limit=float(ts_limit),
            atr_val=float(atr_val),
            atr_threshold=float(atr_threshold),
            fast_ema=fast_ema,
            slow_ema=slow_ema,
        )
        candidates: List[str] = [str(chosen)]
        thr = float(atr_threshold or 0.0)

        if bool(chain_available) and bool(range_bound):
            candidates = ["short_straddle", "short_strangle", "iron_fly", "iron_condor", "long_strangle"]
            if thr > 0 and float(atr_val) <= 0.9 * thr:
                candidates = ["iron_fly", "short_straddle", "short_strangle", "iron_condor", "long_strangle"]
            elif thr > 0 and float(atr_val) >= 1.35 * thr:
                candidates = ["long_strangle", "iron_condor", "short_strangle", "short_straddle", "iron_fly"]
            if mode == "conservative":
                candidates = [s for s in candidates if s in {"iron_fly", "iron_condor", "short_strangle", "short_straddle"}]
            if mode == "aggressive" and "long_straddle" not in candidates:
                candidates.append("long_straddle")
        elif bool(chain_available) and float(trend_strength) > float(ts_limit):
            bullish = (fast_ema is not None and slow_ema is not None and float(fast_ema) >= float(slow_ema))
            if bullish:
                candidates = ["bull_call_spread", "call_ratio_backspread", "long_call", "short_put"]
            else:
                candidates = ["put_ratio_backspread", "long_put", "short_call", "bull_put_spread"]
        else:
            candidates = ["long_call", "long_put", "short_call", "short_put"]

        if str(chosen) not in candidates:
            candidates.insert(0, str(chosen))

        try:
            if regime_profile and isinstance(regime_profile, dict):
                preferred = [str(x) for x in list(regime_profile.get("preferred") or []) if str(x).strip()]
                if preferred:
                    ordered_pref = [s for s in preferred if s in candidates]
                    ordered_rest = [s for s in candidates if s not in ordered_pref]
                    candidates = ordered_pref + ordered_rest
        except Exception:
            pass
        # Keep deterministic and compact.
        ordered: List[str] = []
        for s in candidates:
            if s not in ordered:
                ordered.append(s)
        chosen = ordered[0] if ordered else "directional"
        return {"selected": str(chosen), "candidates": ordered, "mode": mode}

    def get_runtime_diagnostics(self) -> Dict[str, object]:
        """Small snapshot for UI/debugging."""
        top_blocks = sorted(
            ((k, int(v)) for k, v in self._entry_block_counts.items()),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        top_selected = sorted(
            ((k, int(v)) for k, v in self._strategy_selected_counts.items()),
            key=lambda x: x[1],
            reverse=True,
        )[:6]
        top_decisions = sorted(
            ((k, int(v)) for k, v in self._strategy_decision_counts.items()),
            key=lambda x: x[1],
            reverse=True,
        )[:6]
        top_considered = sorted(
            ((k, int(v)) for k, v in self._strategy_considered_counts.items()),
            key=lambda x: x[1],
            reverse=True,
        )[:6]
        return {
            "ts": float(time.time()),
            "last_block": dict(self._last_entry_block or {}),
            "top_block_codes": top_blocks,
            "top_selected": top_selected,
            "top_decisions": top_decisions,
            "top_considered": top_considered,
            "router": dict(self._last_router_snapshot or {}),
            "portfolio_risk": dict(self._last_portfolio_risk or {}),
            "gpt_preset_request": dict(self._last_gpt_preset_request or {}),
            "config_warnings": list(self._config_warnings or []),
            "gpt_metrics": {
                "attempts": int(getattr(self, "_gpt_attempts", 0) or 0),
                "recommendations": int(getattr(self, "_gpt_recommendations", 0) or 0),
                "gate_take": int(getattr(self, "_gpt_gate_take", 0) or 0),
                "gate_skip": int(getattr(self, "_gpt_gate_skip", 0) or 0),
                "gate_unknown": int(getattr(self, "_gpt_gate_unknown", 0) or 0),
                "fallbacks": int(getattr(self, "_gpt_fallbacks", 0) or 0),
                "skipped_required": int(getattr(self, "_gpt_skipped_required", 0) or 0),
            },
        }

    def _resolve_directional_trade_style(self, *, vol_decreasing: bool) -> str:
        """Return "long" or "short" for directional entries.

        Goal: minimize losses from frequent switching by allowing a style lock.
        """

        raw = str(getattr(self.cfg, "directional_trade_style", "auto") or "auto").strip().lower()
        if raw in {"long", "long_only", "buy", "buy_only"}:
            return "long"
        if raw in {"short", "short_only", "sell", "sell_only"}:
            return "short"

        desired = "short" if bool(vol_decreasing) else "long"

        # Never flip style while a directional position is open.
        if bool(self.state.open_directional) and self._dir_style_locked in {"long", "short"}:
            return str(self._dir_style_locked)

        try:
            lock_min = int(getattr(self.cfg, "directional_style_lock_minutes", 0) or 0)
        except Exception:
            lock_min = 0
        if lock_min <= 0:
            self._dir_style_locked = desired
            self._dir_style_locked_ts = time.time()
            return desired

        now = time.time()
        if self._dir_style_locked in {"long", "short"}:
            if (now - float(self._dir_style_locked_ts)) < float(lock_min) * 60.0:
                return str(self._dir_style_locked)

        self._dir_style_locked = desired
        self._dir_style_locked_ts = now
        return desired

    def _log_throttled(self, msg: str, *, key: str, interval_sec: float = 15.0) -> None:
        now = time.time()
        if key == self._last_candle_err and (now - self._last_candle_err_ts) < float(interval_sec):
            return
        self._last_candle_err = key
        self._last_candle_err_ts = now
        print(msg)

    def _current_model_artifact_path(self) -> str:
        try:
            return os.path.abspath("ml_signal_model.pkl")
        except Exception:
            return "ml_signal_model.pkl"

    def _current_model_checksum(self) -> str:
        if self._ml_model_checksum is not None:
            return str(self._ml_model_checksum)
        try:
            with open(self._current_model_artifact_path(), "rb") as fh:
                self._ml_model_checksum = hashlib.sha1(fh.read()).hexdigest()
        except Exception:
            self._ml_model_checksum = ""
        return str(self._ml_model_checksum or "")

    def _confidence_bucket(self, probability: float) -> str:
        try:
            lower = math.floor(float(probability) * 10.0) / 10.0
        except Exception:
            lower = 0.0
        upper = min(1.0, lower + 0.1)
        return f"{lower:.1f}-{upper:.1f}"

    def _current_drawdown_pct(self) -> float:
        try:
            account_capital = max(float(getattr(self.cfg, "account_capital", 100000.0) or 100000.0), 1.0)
        except Exception:
            account_capital = 100000.0
        current_equity = account_capital + float(getattr(self.state, "realized_pnl", 0.0) or 0.0) + float(getattr(self.state, "unrealized_pnl", 0.0) or 0.0)
        self._peak_equity_rupees = max(float(getattr(self, "_peak_equity_rupees", account_capital) or account_capital), current_equity)
        peak_equity = max(float(self._peak_equity_rupees), 1.0)
        drawdown = max(0.0, peak_equity - current_equity)
        return float(drawdown / peak_equity)

    def _active_label_policy_config(self) -> Dict[str, Any]:
        model_bundle = getattr(self, "ml_model", None)
        try:
            metrics = dict(getattr(model_bundle, "metrics", {}) or {})
        except Exception:
            metrics = {}
        raw_name = str(metrics.get("label_policy") or "trade_quality_binary").strip()
        policy_name = raw_name.split(":", 1)[0] or "trade_quality_binary"
        horizon = max(1, int(metrics.get("horizon_bars") or 45))
        try:
            spec = get_label_policy_spec(policy_name, horizon)
        except Exception:
            spec = get_label_policy_spec("trade_quality_binary", horizon)
        params = dict(spec.parameters or {})
        return {
            "policy_name": str(spec.name or policy_name),
            "label_policy_version": f"{spec.name}:{spec.version}",
            "profit_target_pct": float(params.get("profit_target_pct", 0.01) or 0.01),
            "stop_loss_pct": float(params.get("stop_loss_pct", 0.005) or 0.005),
            "max_duration_bars": max(1, int(params.get("max_duration_bars", min(horizon, 18)) or min(horizon, 18))),
        }

    def _build_triple_barrier_metadata(
        self,
        *,
        fill_price: float,
        prediction_id: str,
        bar_timestamp: Optional[dt_datetime],
    ) -> Dict[str, Any]:
        cfg = self._active_label_policy_config()
        state = initialize_triple_barrier_state(
            fill_price=float(fill_price),
            prediction_id=str(prediction_id or ""),
            policy_name=str(cfg["policy_name"]),
            profit_target_pct=float(cfg["profit_target_pct"]),
            stop_loss_pct=float(cfg["stop_loss_pct"]),
            max_duration_bars=int(cfg["max_duration_bars"]),
            bar_timestamp=bar_timestamp,
        )
        payload = state.to_dict()
        payload["enabled"] = True
        payload["label_policy_version"] = str(cfg["label_policy_version"])
        return payload

    def _extract_triple_barrier_state(self, trade: Dict[str, Any]) -> Optional[TripleBarrierState]:
        meta = trade.get("meta")
        if not isinstance(meta, dict):
            return None
        payload = meta.get("triple_barrier")
        return TripleBarrierState.from_dict(payload if isinstance(payload, dict) else None)

    def _write_triple_barrier_state(self, trade: Dict[str, Any], state: TripleBarrierState) -> None:
        meta = trade.setdefault("meta", {})
        if not isinstance(meta, dict):
            meta = {}
            trade["meta"] = meta
        existing = meta.get("triple_barrier")
        payload = state.to_dict()
        if isinstance(existing, dict):
            for key, value in existing.items():
                if key not in payload:
                    payload[key] = value
        payload["enabled"] = True
        meta["triple_barrier"] = payload

    def _triple_barrier_price(self, trade: Dict[str, Any], leg: Dict[str, Any]) -> Optional[float]:
        symbol = str(leg.get("symbol") or trade.get("symbol") or "")
        exchange = str(leg.get("exchange") or trade.get("exchange") or "").strip() or None
        try:
            bid, ask, ltp = self.client.get_bid_ask(symbol, exchange_hint=exchange)
            if bid is not None and ask is not None:
                return float((float(bid) + float(ask)) / 2.0)
            if ltp is not None:
                return float(ltp)
        except Exception:
            pass
        return self._try_get_ltp_for_leg(leg)

    def _persist_directional_trade_outcome(
        self,
        *,
        trade: Dict[str, Any],
        exit_reason: str,
        realized_pnl: Optional[float],
        estimated_slippage_pct: float,
    ) -> None:
        if self.db_manager is None:
            return
        meta = trade.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        tb = meta.get("triple_barrier")
        tb = tb if isinstance(tb, dict) else {}
        prediction_id = str(meta.get("prediction_id") or tb.get("prediction_id") or "")
        exit_time = dt_datetime.now(IST) if IST else dt_datetime.now()
        try:
            duration_minutes = max(0.0, (time.time() - float(trade.get("opened_ts") or time.time())) / 60.0)
        except Exception:
            duration_minutes = None
        try:
            self.db_manager.insert_trade_outcome(
                {
                    "trade_id": str(trade.get("trade_id") or ""),
                    "prediction_id": prediction_id,
                    "entry_time": str(trade.get("entry_time") or ""),
                    "exit_time": str(exit_time),
                    "duration_minutes": duration_minutes,
                    "strategy": str(trade.get("name") or ""),
                    "regime": str(meta.get("regime") or ""),
                    "strategy_signal": str(meta.get("strategy_signal") or ""),
                    "entry_reason": str(meta.get("reason") or ""),
                    "exit_reason": str(exit_reason or ""),
                    "probability": meta.get("ml_probability"),
                    "threshold": meta.get("ml_threshold"),
                    "gross_pnl": realized_pnl,
                    "pnl": realized_pnl,
                    "estimated_slippage": float(estimated_slippage_pct),
                }
            )
        except Exception:
            pass

    def _close_directional_trade_via_triple_barrier(
        self,
        *,
        trade: Dict[str, Any],
        trigger_source: str,
        reference_price: float,
    ) -> bool:
        trade_lock = trade.get("_triple_barrier_lock")
        if not hasattr(trade_lock, "acquire"):
            trade_lock = Lock()
            trade["_triple_barrier_lock"] = trade_lock
        if not trade_lock.acquire(blocking=False):
            return False
        try:
            state = self._extract_triple_barrier_state(trade)
            if state is None or state.exit_lock:
                return False
            state.exit_lock = True
            state.exit_lock_ts = float(time.time())
            state.exit_trigger_source = str(trigger_source or "")
            if not state.final_execution_duration_bars:
                state.final_execution_duration_bars = int(state.elapsed_bars)
            self._write_triple_barrier_state(trade, state)

            meta = trade.setdefault("meta", {})
            if not isinstance(meta, dict):
                meta = {}
                trade["meta"] = meta

            total_realized = 0.0
            realized_any = False
            exit_prices: List[float] = []
            legs = trade.get("legs")
            if not isinstance(legs, list) or not legs:
                legs = [
                    {
                        "symbol": trade.get("symbol"),
                        "token": trade.get("token"),
                        "exchange": trade.get("exchange"),
                        "side": trade.get("side"),
                        "quantity": trade.get("quantity"),
                        "entry_price": trade.get("entry_price"),
                        "strike": trade.get("strike"),
                        "option_type": trade.get("option_type"),
                        "expiry": trade.get("expiry"),
                    }
                ]
            emitted_legs: List[Dict[str, Any]] = []
            for leg in legs:
                if not isinstance(leg, dict) or int(leg.get("quantity") or 0) <= 0:
                    continue
                realized = self._close_single_leg(
                    trade=trade,
                    leg=leg,
                    position_type="directional",
                    reason=str(trigger_source or ""),
                )
                total_realized += float(realized)
                realized_any = True
                try:
                    exit_prices.append(float(leg.get("exit_price")))
                except Exception:
                    pass
                emitted_legs.append(dict(leg))

            final_exit_price = float(sum(exit_prices) / len(exit_prices)) if exit_prices else float(reference_price)
            ref_price = max(float(reference_price or 0.0), 1e-9)
            realized_slippage_pct = abs(final_exit_price - ref_price) / ref_price if ref_price > 0.0 else 0.0
            meta["exit_trigger_source"] = str(trigger_source or "")
            meta["final_execution_duration_bars"] = int(state.final_execution_duration_bars)
            meta["realized_slippage_pct"] = float(realized_slippage_pct)
            self._write_triple_barrier_state(trade, state)

            self.state.last_exit_ts = time.time()
            if trigger_source == "stop_loss":
                self._mark_stopout()
            else:
                self._mark_non_stop_exit()

            res_for_note = float(total_realized) if realized_any else None
            self._note_trade_result(res_for_note)
            if res_for_note is not None:
                self._strategy_winrate_record_result(str(trade.get("name") or ""), float(res_for_note) >= 0)

            if emitted_legs:
                try:
                    self._emit(
                        TradeLogEvent(
                            ts=time.time(),
                            event="CLOSE",
                            trade_id=str(trade.get("trade_id") or "(unknown)"),
                            position_type="directional",
                            name=str(trade.get("name") or ""),
                            legs=emitted_legs,
                            realized=res_for_note,
                            reason=str(trigger_source or ""),
                        )
                    )
                except Exception:
                    pass

            self._persist_directional_trade_outcome(
                trade=trade,
                exit_reason=str(trigger_source or ""),
                realized_pnl=res_for_note,
                estimated_slippage_pct=float(realized_slippage_pct),
            )
            return True
        finally:
            trade_lock.release()

    def _record_ml_prediction(
        self,
        *,
        prediction_id: str,
        candles: List[Candle],
        probability: float,
        threshold: float,
        take: bool,
        reason: str,
        regime: str,
        suggested_strategy: Optional[str],
        feature_names: Sequence[str],
        feature_values: Sequence[float],
        ctx: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        if self.db_manager is None:
            LOGGER.debug("Shadow/ML prediction skipped because db_manager is unavailable")
            return None
        latest = candles[-1] if candles else None
        ts = float(getattr(latest.time, "timestamp", lambda: time.time())()) if latest is not None else float(time.time())
        feature_snapshot = {
            str(name): float(feature_values[idx])
            for idx, name in enumerate(feature_names)
            if idx < len(feature_values)
        }
        try:
            feature_vector_checksum = hashlib.sha1(json.dumps(feature_snapshot, sort_keys=True).encode("utf-8")).hexdigest()
        except Exception:
            feature_vector_checksum = ""

        model_bundle = getattr(self, "ml_model", None)
        metrics: Dict[str, Any] = {}
        trained_at = ""
        try:
            metrics = dict(getattr(model_bundle, "metrics", {}) or {})
            trained_at = str(getattr(model_bundle, "trained_at", "") or "")
        except Exception:
            pass

        probability = float(probability)
        threshold = float(threshold)
        reason_text = str(reason or "")
        paper_mode = bool(getattr(self.cfg, "ml_paper_mode_enabled", False))
        if paper_mode:
            if "blocked" in reason_text or "missing" in reason_text:
                source = "PAPER_BLOCKED"
            elif take:
                source = "PAPER"
            else:
                source = "PAPER_FILTERED"
        else:
            source = "ACTIVE"
        record = {
            "prediction_id": str(prediction_id or ""),
            "ts": ts,
            "symbol": str(getattr(self.cfg, "underlying", "") or ""),
            "exchange": str(getattr(self.cfg, "exchange", "") or ""),
            "token": "",
            "option_symbol": "",
            "underlying_price": float(getattr(latest, "close", 0.0) or 0.0) if latest is not None else 0.0,
            "direction": "bull" if probability >= threshold else "flat",
            "regime": str(regime or ""),
            "model_version": str(metrics.get("model_type") or trained_at or "ml_signal_model"),
            "model_artifact_path": self._current_model_artifact_path(),
            "model_checksum": self._current_model_checksum(),
            "label_policy_version": str(metrics.get("label_policy") or "current_triple_barrier:v1"),
            "feature_set_version": str(metrics.get("feature_set_version") or "target_features_v1"),
            "threshold": threshold,
            "probability": probability,
            "prediction": probability,
            "predicted_class": 1 if probability >= threshold else 0,
            "confidence": abs(probability - 0.5) * 2.0,
            "confidence_bucket": self._confidence_bucket(probability),
            "strategy_context": json.dumps(ctx or {}, default=str),
            "strategy_signal": str(suggested_strategy or ""),
            "source": source,
            "reason": str(reason or ""),
            "trade_candidate": bool(take),
            "trade_taken": False,
            "no_trade_reason": "" if take else str(reason or "blocked"),
            "feature_snapshot": feature_snapshot,
            "features": feature_snapshot,
            "feature_vector_checksum": feature_vector_checksum,
            "future_label_status": "pending",
            "horizon_bars": 45,
        }
        try:
            LOGGER.debug(
                "Recording ML prediction id=%s source=%s prob=%.4f threshold=%.4f take=%s reason=%s",
                record["prediction_id"],
                source,
                probability,
                threshold,
                bool(take),
                reason_text,
            )
            prediction_id = self.db_manager.insert_prediction(record)
            self._last_prediction_id = prediction_id
            self._last_prediction_record = record
            LOGGER.debug("Recorded ML prediction id=%s successfully", prediction_id)
            return prediction_id
        except Exception as exc:
            LOGGER.warning("Failed to record ML prediction id=%s: %s", record["prediction_id"], exc)
            return None

    def _emit(self, evt: TradeLogEvent) -> None:
        if self.notifier and str(evt.event).upper() in {"OPEN", "CLOSE"}:
            try:
                msg = f"<b>[{evt.event}]</b> {evt.name} - {evt.position_type}\n"
                if evt.realized is not None:
                    msg += f"PnL: {evt.realized:.2f}\n"
                if evt.reason:
                    msg += f"Reason: {evt.reason}\n"
                self.notifier.send_message(msg)
            except Exception:
                pass
                
        if not self._event_sink:
            return
        try:
            # Best-effort: compute margin requirement for OPEN/UPDATE snapshots.
            # Never block trading/UI on margin calculation failures.
            if evt.margin_required is None and str(evt.event).upper() in {"OPEN", "UPDATE"}:
                try:
                    mr = self._try_calc_margin_required(evt.legs)
                    if mr is not None:
                        evt = replace(evt, margin_required=float(mr))
                except Exception:
                    pass

            # Best-effort: attach stop/target fields to leg snapshots so the UI
            # can render live risk thresholds.
            try:
                evt = self._enrich_evt_for_ui(evt)
            except Exception:
                pass
            # If we have a recent GPT advice recorded, attach it to the next OPEN event
            try:
                pending = getattr(self, "_pending_gpt_advice", None)
                if pending and isinstance(pending, dict) and (time.time() - float(pending.get("ts", 0.0) or 0.0)) < 6.0:
                    if str(evt.event).upper() == "OPEN":
                        try:
                            note = f"gpt:{str(pending.get('decision') or '')}:{str(pending.get('reason') or '')}"
                            evt = replace(evt, reason=note)
                        except Exception:
                            pass
                        try:
                            # Clear after attaching once.
                            self._pending_gpt_advice = None
                        except Exception:
                            pass
            except Exception:
                pass
            self._event_sink(evt)
        except Exception:
            # Never let UI reporting crash the strategy loop.
            return

    def _find_open_trade(self, *, trade_id: str, position_type: str) -> Optional[Dict[str, object]]:
        tid = str(trade_id or "").strip()
        p = str(position_type or "").strip().lower()
        if not tid:
            return None

        src: List[Dict[str, object]]
        if p == "multi":
            src = self.state.open_multi
        elif p == "directional":
            src = self.state.open_directional
        else:
            return None

        for tr in src or []:
            if not isinstance(tr, dict):
                continue
            if str(tr.get("trade_id") or "").strip() == tid:
                return tr
        return None

    def _enrich_evt_for_ui(self, evt: TradeLogEvent) -> TradeLogEvent:
        if not isinstance(evt.legs, list) or not evt.legs:
            return evt

        p = str(evt.position_type or "").strip().lower()
        tr = self._find_open_trade(trade_id=str(evt.trade_id or ""), position_type=p)

        mtm_stop_val: Optional[float] = None
        mtm_target_val: Optional[float] = None
        spot_stop_val: Optional[float] = None
        spot_target_val: Optional[float] = None

        if p == "multi" and isinstance(tr, dict):
            meta = tr.get("meta") if isinstance(tr.get("meta"), dict) else {}

            entry_abs = None
            try:
                if isinstance(meta, dict) and meta.get("entry_premium_abs") is not None:
                    entry_abs = float(meta.get("entry_premium_abs"))
                elif isinstance(meta, dict) and meta.get("entry_premium") is not None:
                    entry_abs = abs(float(meta.get("entry_premium")))
            except Exception:
                entry_abs = None

            if entry_abs is not None and entry_abs > 0:
                try:
                    stop_pct_eff = (
                        float(meta.get("gpt_mtm_stop_pct"))
                        if isinstance(meta, dict) and meta.get("gpt_mtm_stop_pct") is not None
                        else float(
                            meta.get("ml_dynamic_stop_loss_pct")
                            if isinstance(meta, dict) and meta.get("ml_dynamic_stop_loss_pct") is not None
                            else getattr(self.cfg, "premium_mtm_stop_pct", 0.30) or 0.30
                        )
                    )
                except Exception:
                    stop_pct_eff = 0.30
                try:
                    target_pct_eff = (
                        float(meta.get("gpt_mtm_target_pct"))
                        if isinstance(meta, dict) and meta.get("gpt_mtm_target_pct") is not None
                        else float(
                            meta.get("ml_dynamic_target_pct")
                            if isinstance(meta, dict) and meta.get("ml_dynamic_target_pct") is not None
                            else getattr(self.cfg, "premium_mtm_target_pct", 0.18)
                        )
                    )
                except Exception:
                    target_pct_eff = 0.18

                mtm_stop_val = -float(stop_pct_eff) * float(entry_abs)
                mtm_target_val = float(target_pct_eff) * float(entry_abs) if float(target_pct_eff) > 0 else None

        if p == "directional" and isinstance(tr, dict):
            # Directional trades are typically single-leg option positions.
            # For UI display, expose premium-based MTM stop/target too so the
            # UI can render stop/target in option premium terms.
            meta = tr.get("meta") if isinstance(tr.get("meta"), dict) else {}

            entry_abs = None
            try:
                if isinstance(meta, dict) and meta.get("entry_premium_abs") is not None:
                    entry_abs = float(meta.get("entry_premium_abs"))
                elif isinstance(meta, dict) and meta.get("entry_premium") is not None:
                    entry_abs = abs(float(meta.get("entry_premium")))
            except Exception:
                entry_abs = None

            # Fallback: compute from the event legs snapshot.
            if entry_abs is None:
                try:
                    entry_premium = 0.0
                    any_price = False
                    for lg in evt.legs:
                        if not isinstance(lg, dict) or bool(lg.get("is_hedge")):
                            continue
                        p0 = lg.get("entry_price")
                        try:
                            pf = float(p0) if p0 is not None else None
                        except Exception:
                            pf = None
                        if pf is None:
                            continue
                        q = float(lg.get("quantity") or 0)
                        side = str(lg.get("side") or "").upper()
                        if q <= 0 or side not in {"BUY", "SELL"}:
                            continue
                        any_price = True
                        if side == "SELL":
                            entry_premium += pf * q
                        else:
                            entry_premium -= pf * q
                    if any_price:
                        entry_abs = float(abs(entry_premium))
                except Exception:
                    entry_abs = None

            if entry_abs is not None and entry_abs > 0:
                try:
                    stop_pct_eff = (
                        float(meta.get("gpt_mtm_stop_pct"))
                        if isinstance(meta, dict) and meta.get("gpt_mtm_stop_pct") is not None
                        else float(
                            meta.get("ml_dynamic_stop_loss_pct")
                            if isinstance(meta, dict) and meta.get("ml_dynamic_stop_loss_pct") is not None
                            else getattr(self.cfg, "premium_mtm_stop_pct", 0.30) or 0.30
                        )
                    )
                except Exception:
                    stop_pct_eff = 0.30
                try:
                    target_pct_eff = (
                        float(meta.get("gpt_mtm_target_pct"))
                        if isinstance(meta, dict) and meta.get("gpt_mtm_target_pct") is not None
                        else float(
                            meta.get("ml_dynamic_target_pct")
                            if isinstance(meta, dict) and meta.get("ml_dynamic_target_pct") is not None
                            else getattr(self.cfg, "premium_mtm_target_pct", 0.18)
                        )
                    )
                except Exception:
                    target_pct_eff = 0.18

                mtm_stop_val = -float(stop_pct_eff) * float(entry_abs)
                mtm_target_val = float(target_pct_eff) * float(entry_abs) if float(target_pct_eff) > 0 else None

            try:
                if tr.get("spot_stop") is not None:
                    spot_stop_val = float(tr.get("spot_stop"))
            except Exception:
                spot_stop_val = None
            try:
                if tr.get("spot_target") is not None:
                    spot_target_val = float(tr.get("spot_target"))
            except Exception:
                spot_target_val = None

        # If nothing to enrich, return as-is.
        if (
            mtm_stop_val is None
            and mtm_target_val is None
            and spot_stop_val is None
            and spot_target_val is None
        ):
            return evt

        legs_out: List[Dict[str, object]] = []
        for leg in evt.legs:
            if not isinstance(leg, dict):
                continue
            d = dict(leg)
            if bool(d.get("is_hedge")):
                legs_out.append(d)
                continue
            # Always overwrite derived thresholds so the UI stays accurate when
            # trailing stops / dynamic exit thresholds change over time.
            if mtm_stop_val is not None:
                d["mtm_stop"] = float(mtm_stop_val)
            if mtm_target_val is not None:
                d["mtm_target"] = float(mtm_target_val)
            if spot_stop_val is not None:
                d["spot_stop"] = float(spot_stop_val)
            if spot_target_val is not None:
                d["spot_target"] = float(spot_target_val)
            legs_out.append(d)

        if not legs_out:
            return evt
        return replace(evt, legs=legs_out)

    def _try_calc_margin_required(self, legs: List[Dict[str, object]]) -> Optional[float]:
        fn = getattr(self.client, "calculate_order_margin_required", None)
        if not callable(fn):
            return None

        if not isinstance(legs, list) or not legs:
            return None

        total = 0.0
        any_ok = False
        for leg in legs:
            if not isinstance(leg, dict):
                continue

            try:
                token = str(leg.get("token") or "").strip()
                exchange_raw = str(leg.get("exchange") or "").strip().upper()
                symbol = str(leg.get("symbol") or "").strip() or None
                side = str(leg.get("side") or "").strip().upper() or None
                quantity = int(leg.get("quantity") or 0)
            except Exception:
                continue

            # Infer exchange for option legs when missing.
            if not exchange_raw:
                try:
                    opt_type = str(leg.get("option_type") or "").strip().upper()
                except Exception:
                    opt_type = ""
                sym_u = str(symbol or "").strip().upper()
                if opt_type in {"CE", "PE", "CALL", "PUT", "C", "P"} or sym_u.endswith("CE") or sym_u.endswith("PE"):
                    exchange_raw = "NFO"

            # Normalize common aliases.
            if exchange_raw in {"NSEFO", "NFO"}:
                exchange_raw = "NFO"
            exchange = exchange_raw.strip() or None

            if not token or not exchange or not symbol or not side or quantity <= 0:
                continue

            price = None
            try:
                p = leg.get("entry_price")
                if p is None:
                    p = leg.get("exit_price")
                if p is not None:
                    price = float(p)
            except Exception:
                price = None

            try:
                mr = fn(
                    exchange=exchange,
                    symbol=symbol,
                    token=token,
                    side=side,
                    quantity=quantity,
                    price=price,
                )
            except Exception:
                mr = None

            if mr is None:
                continue
            try:
                total += float(mr)
                any_ok = True
            except Exception:
                continue

        if not any_ok:
            return None
        return float(total)

    def _now_ist_time(self) -> dt_time:
        if IST is None:
            return dt_datetime.now().time()
        return dt_datetime.now(IST).time()

    def _now_ist_date(self) -> date:
        if IST is None:
            return date.today()
        try:
            return dt_datetime.now(IST).date()
        except Exception:
            return date.today()

    def _ensure_hedge_orders_day(self) -> None:
        today = self._now_ist_date()
        if self.state.delta_hedge_orders_day != today:
            self.state.delta_hedge_orders_day = today
            self.state.delta_hedge_orders_today = 0

    def _ensure_daily_risk_counters_day(self) -> None:
        today = self._now_ist_date()
        if self.state.risk_counters_day == today:
            return
        self.state.risk_counters_day = today
        self.state.trades_today = 0
        self.state.stopouts_today = 0
        self.state.consecutive_stopouts = 0
        self.state.consecutive_wins = 0
        self.state.realized_pnl = 0.0
        self.state.unrealized_pnl = 0.0

    def _candle_age_sec(self, candle: Candle) -> Optional[float]:
        try:
            ts = candle.time
        except Exception:
            return None
        if not isinstance(ts, dt_datetime):
            return None
        try:
            if ts.tzinfo is not None:
                now_dt = dt_datetime.now(ts.tzinfo)
            elif IST is not None:
                now_dt = dt_datetime.now(IST)
            else:
                now_dt = dt_datetime.now()
            age = (now_dt - ts).total_seconds()
            return max(0.0, float(age))
        except Exception:
            return None

    def _entry_price_bounds_ok(self, price: Optional[float]) -> tuple[bool, str]:
        try:
            min_p = float(getattr(self.cfg, "entry_min_option_premium", 0.0) or 0.0)
        except Exception:
            min_p = 0.0
        try:
            max_p = float(getattr(self.cfg, "entry_max_option_premium", 0.0) or 0.0)
        except Exception:
            max_p = 0.0

        if min_p <= 0 and max_p <= 0:
            return True, ""

        if price is None:
            return False, "Option premium unavailable for bounds check"

        try:
            p = float(price)
        except Exception:
            return False, "Option premium could not be parsed"

        if min_p > 0 and p < min_p:
            return False, f"Option premium {p:.2f} below min {min_p:.2f}"
        if max_p > 0 and p > max_p:
            return False, f"Option premium {p:.2f} above max {max_p:.2f}"
        return True, ""

    def _check_entry_leg_premiums(self, legs: List[Dict[str, object]]) -> tuple[bool, str]:
        try:
            min_p = float(getattr(self.cfg, "entry_min_option_premium", 0.0) or 0.0)
        except Exception:
            min_p = 0.0
        try:
            max_p = float(getattr(self.cfg, "entry_max_option_premium", 0.0) or 0.0)
        except Exception:
            max_p = 0.0
        try:
            min_total = float(getattr(self.cfg, "entry_min_total_premium", 0.0) or 0.0)
        except Exception:
            min_total = 0.0
        try:
            max_total = float(getattr(self.cfg, "entry_max_total_premium", 0.0) or 0.0)
        except Exception:
            max_total = 0.0

        if min_p <= 0 and max_p <= 0 and min_total <= 0 and max_total <= 0:
            ok, reason = self._check_entry_liquidity(legs)
            if not ok:
                return False, reason
            return True, ""

        # Fall back to min_option_premium when entry_min_option_premium is disabled.
        # This ensures the global min_option_premium safeguard is not bypassed.
        try:
            min_option_fallback = float(getattr(self.cfg, "min_option_premium", 0.0) or 0.0)
        except Exception:
            min_option_fallback = 0.0
        if min_p <= 0 and min_option_fallback > 0:
            min_p = min_option_fallback

        total_abs = 0.0
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            price = leg.get("entry_price")
            if price is None:
                price = self._try_get_ltp_for_leg(leg)
            if price is None:
                return False, "Option premium unavailable for bounds check"
            try:
                p = float(price)
            except Exception:
                return False, "Option premium could not be parsed"
            total_abs += abs(p)
            if min_p > 0 and p < min_p:
                return False, f"Leg premium {p:.2f} below min {min_p:.2f}"
            if max_p > 0 and p > max_p:
                return False, f"Leg premium {p:.2f} above max {max_p:.2f}"

        if min_total > 0 and total_abs < min_total:
            return False, f"Total premium {total_abs:.2f} below min {min_total:.2f}"
        if max_total > 0 and total_abs > max_total:
            return False, f"Total premium {total_abs:.2f} above max {max_total:.2f}"
        ok, reason = self._check_entry_liquidity(legs)
        if not ok:
            return False, reason
        return True, ""

    def _check_entry_liquidity(self, legs: List[Dict[str, object]]) -> tuple[bool, str]:
        try:
            require_ba = bool(getattr(self.cfg, "entry_require_bid_ask", False))
        except Exception:
            require_ba = False
        try:
            max_pct = float(getattr(self.cfg, "entry_max_bid_ask_spread_pct", 0.0) or 0.0)
        except Exception:
            max_pct = 0.0
        try:
            max_abs = float(getattr(self.cfg, "entry_max_bid_ask_spread_abs", 0.0) or 0.0)
        except Exception:
            max_abs = 0.0
        try:
            shock_mult = float(getattr(self.cfg, "entry_spread_shock_mult", 0.0) or 0.0)
        except Exception:
            shock_mult = 0.0

        # Fast path: if no effective liquidity checks are configured, skip
        # bid/ask probing entirely to avoid unnecessary broker calls.
        if max_pct <= 0 and max_abs <= 0 and shock_mult <= 0:
            return True, ""

        def _spread_key(leg: Dict[str, object], fallback_symbol: str) -> str:
            token = str(leg.get("token") or "").strip()
            exch = str(leg.get("exchange") or "").strip().upper()
            if token:
                return f"{exch}:{token}" if exch else token
            return str(fallback_symbol or "").strip().upper()

        def _shock_ok(key: str, spread: float) -> tuple[bool, str]:
            if shock_mult <= 0:
                return True, ""
            try:
                lookback = int(getattr(self.cfg, "entry_spread_shock_lookback", 20) or 20)
            except Exception:
                lookback = 20
            try:
                min_samples = int(getattr(self.cfg, "entry_spread_shock_min_samples", 5) or 5)
            except Exception:
                min_samples = 5
            lookback = max(2, min(200, int(lookback)))
            min_samples = max(1, min(int(lookback), int(min_samples)))

            hist = list(self._entry_spread_watch.get(key) or [])
            ok = True
            reason = ""
            if len(hist) >= int(min_samples):
                ordered = sorted(float(x) for x in hist if float(x) >= 0)
                if ordered:
                    mid_i = len(ordered) // 2
                    if len(ordered) % 2:
                        baseline = float(ordered[mid_i])
                    else:
                        baseline = (float(ordered[mid_i - 1]) + float(ordered[mid_i])) / 2.0
                    if baseline > 0 and float(spread) > float(baseline) * float(shock_mult):
                        ok = False
                        reason = (
                            f"Spread shock for {key} "
                            f"({float(spread):.2f} > {float(shock_mult):.2f}x median {float(baseline):.2f})"
                        )

            hist.append(float(spread))
            if len(hist) > int(lookback):
                hist = hist[-int(lookback) :]
            self._entry_spread_watch[key] = hist
            return ok, reason

        for leg in legs:
            if not isinstance(leg, dict):
                continue
            symbol = str(leg.get("symbol") or "")
            if not symbol:
                continue
            exchange = str(leg.get("exchange") or "") or None
            # Prefer numeric token when available to avoid symbol->token resolution
            # failures for option contracts.
            tok = str(leg.get("token") or "").strip()
            quote_key = tok if (tok.isdigit() and int(tok) > 0) else symbol
            bid, ask, _ltp = self.client.get_bid_ask(quote_key, exchange_hint=exchange)
            if require_ba and (bid is None or ask is None):
                # TypeB market quote API does not guarantee depth/bid/ask for all instruments.
                # If we at least have LTP, proceed using LTP-only to avoid blocking entries.
                if _ltp is None:
                    try:
                        _ltp = self._try_get_ltp_for_leg(leg)
                    except Exception:
                        _ltp = None
                if _ltp is not None:
                    if bool(getattr(self.cfg, "debug_log_no_signal", False)):
                        try:
                            self._log_throttled(
                                f"[LIQUIDITY][DEBUG] Bid/ask unavailable for {symbol}; proceeding using LTP only",
                                key=f"liquidity:ba-missing:{symbol}",
                                interval_sec=60.0,
                            )
                        except Exception:
                            pass
                    continue
                return False, f"Bid/ask unavailable for {symbol}"
            if bid is None or ask is None:
                continue
            try:
                spread = float(ask) - float(bid)
            except Exception:
                continue
            if spread < 0:
                spread = abs(spread)
            shock_key = _spread_key(leg, symbol)
            ok, reason = _shock_ok(shock_key, float(spread))
            if not ok:
                return False, reason
            if max_abs > 0 and spread > max_abs:
                return False, f"Spread too wide for {symbol} ({spread:.2f} > {max_abs:.2f})"
            if max_pct > 0:
                mid = (float(ask) + float(bid)) / 2.0
                if mid > 0:
                    pct = spread / mid  # fraction, e.g. 0.005 for 0.5%
                    if pct > max_pct:
                        return False, f"Spread too wide for {symbol} ({pct*100:.2f}% > {max_pct*100:.2f}%)"

        return True, ""

    def _check_exit_liquidity(self, legs: List[Dict[str, object]]) -> tuple[bool, str]:
        """Check exit-time liquidity filters (bid/ask spread guard for closing trades).

        Wider spreads are tolerated at exit vs. entry because you must close the
        position.  Defaults: require_bid_ask=True, max_spread_pct=3.0%, max_abs=₹8.
        """
        try:
            require_ba = bool(getattr(self.cfg, "exit_require_bid_ask", True))
        except Exception:
            require_ba = True
        try:
            max_pct = float(getattr(self.cfg, "exit_max_bid_ask_spread_pct", 3.0) or 3.0)
        except Exception:
            max_pct = 3.0
        try:
            max_abs = float(getattr(self.cfg, "exit_max_bid_ask_spread_abs", 8.0) or 8.0)
        except Exception:
            max_abs = 8.0

        # If all guards are disabled, skip broker calls
        if max_pct <= 0 and max_abs <= 0 and not require_ba:
            return True, ""

        for leg in legs:
            if not isinstance(leg, dict):
                continue
            symbol = str(leg.get("symbol") or "")
            if not symbol:
                continue
            exchange = str(leg.get("exchange") or "") or None
            tok = str(leg.get("token") or "").strip()
            quote_key = tok if (tok.isdigit() and int(tok) > 0) else symbol
            bid, ask, _ = self.client.get_bid_ask(quote_key, exchange_hint=exchange)
            if require_ba and (bid is None or ask is None):
                return False, f"Bid/ask unavailable for {symbol}"
            if bid is None or ask is None:
                continue
            try:
                spread = abs(float(ask) - float(bid))
            except Exception:
                continue
            if max_abs > 0 and spread > max_abs:
                return False, f"Exit spread too wide for {symbol} ({spread:.2f} > {max_abs:.2f})"
            if max_pct > 0:
                mid = (float(ask) + float(bid)) / 2.0
                if mid > 0:
                    pct = (spread / mid) * 100.0
                    if pct > max_pct:
                        return False, f"Exit spread too wide for {symbol} ({pct:.2f}% > {max_pct:.2f}%)"
        return True, ""

    def _session_bucket(self, now_t: dt_time) -> str:
        open_start = self._parse_hhmm(getattr(self.cfg, "session_open_start_hhmm", "09:15"), default=dt_time(9, 15))
        open_end = self._parse_hhmm(getattr(self.cfg, "session_open_end_hhmm", "10:00"), default=dt_time(10, 0))
        close_start = self._parse_hhmm(getattr(self.cfg, "session_close_start_hhmm", "14:30"), default=dt_time(14, 30))
        close_end = self._parse_hhmm(getattr(self.cfg, "session_close_end_hhmm", "15:30"), default=dt_time(15, 30))

        if open_start <= now_t <= open_end:
            return "open"
        if close_start <= now_t <= close_end:
            return "close"
        return "mid"

    def _session_overrides(self, now_t: dt_time) -> tuple[float, float, float, float, str]:
        try:
            min_atr = float(self.cfg.min_atr)
        except Exception:
            min_atr = 0.0
        try:
            max_atr = float(self.cfg.max_atr)
        except Exception:
            max_atr = 1e9
        try:
            rsi_low = float(self.cfg.premium_rsi_low)
        except Exception:
            rsi_low = 45.0
        try:
            rsi_high = float(self.cfg.premium_rsi_high)
        except Exception:
            rsi_high = 55.0

        bucket = self._session_bucket(now_t)
        if bucket == "open":
            if getattr(self.cfg, "session_open_min_atr", None) is not None:
                min_atr = float(getattr(self.cfg, "session_open_min_atr"))
            if getattr(self.cfg, "session_open_max_atr", None) is not None:
                max_atr = float(getattr(self.cfg, "session_open_max_atr"))
            if getattr(self.cfg, "session_open_premium_rsi_low", None) is not None:
                rsi_low = float(getattr(self.cfg, "session_open_premium_rsi_low"))
            if getattr(self.cfg, "session_open_premium_rsi_high", None) is not None:
                rsi_high = float(getattr(self.cfg, "session_open_premium_rsi_high"))
        elif bucket == "close":
            if getattr(self.cfg, "session_close_min_atr", None) is not None:
                min_atr = float(getattr(self.cfg, "session_close_min_atr"))
            if getattr(self.cfg, "session_close_max_atr", None) is not None:
                max_atr = float(getattr(self.cfg, "session_close_max_atr"))
            if getattr(self.cfg, "session_close_premium_rsi_low", None) is not None:
                rsi_low = float(getattr(self.cfg, "session_close_premium_rsi_low"))
            if getattr(self.cfg, "session_close_premium_rsi_high", None) is not None:
                rsi_high = float(getattr(self.cfg, "session_close_premium_rsi_high"))
        else:
            if getattr(self.cfg, "session_mid_min_atr", None) is not None:
                min_atr = float(getattr(self.cfg, "session_mid_min_atr"))
            if getattr(self.cfg, "session_mid_max_atr", None) is not None:
                max_atr = float(getattr(self.cfg, "session_mid_max_atr"))
            if getattr(self.cfg, "session_mid_premium_rsi_low", None) is not None:
                rsi_low = float(getattr(self.cfg, "session_mid_premium_rsi_low"))
            if getattr(self.cfg, "session_mid_premium_rsi_high", None) is not None:
                rsi_high = float(getattr(self.cfg, "session_mid_premium_rsi_high"))

        return min_atr, max_atr, rsi_low, rsi_high, bucket

    def _note_trade_result(self, realized: Optional[float]) -> None:
        if realized is None:
            return
        try:
            r = float(realized)
        except Exception:
            return
        if r > 0:
            self.state.consecutive_wins += 1
        else:
            self.state.consecutive_wins = 0

    def _entry_qty(self, atr_val: Optional[float]) -> int:
        try:
            base_qty = int(self.cfg.lot_size)
        except Exception:
            base_qty = 1
        qty = base_qty

        # 1. Dynamic Position Sizing Override
        try:
            risk_pct = float(getattr(self.cfg, "risk_per_trade_percentage", 0.0) or 0.0)
            capital = float(getattr(self.cfg, "account_capital", 100000.0) or 100000.0)
        except Exception:
            risk_pct = 0.0
            capital = 100000.0

        if risk_pct > 0.0 and capital > 0.0 and atr_val is not None and float(atr_val) > 0:
            # Risk = (Capital * Risk%) / 100
            # Sizing based on 1 ATR stop loss (assuming 1 delta for simplicity or basic stop distance)
            risk_amount = (capital * risk_pct) / 100.0
            raw_qty = risk_amount / float(atr_val)
            # Round down to nearest lot size
            qty = max(base_qty, int((raw_qty // base_qty) * base_qty))

        # 2. Previous Risk Scaling features (apply on top of whatever qty we have)
        if not bool(getattr(self.cfg, "risk_scale_enabled", False)):
            return qty

        factor = 1.0
        try:
            stop_factor = float(getattr(self.cfg, "risk_scale_stopout_factor", 1.0))
        except Exception:
            stop_factor = 1.0
        try:
            recovery_wins = int(getattr(self.cfg, "risk_scale_recovery_wins", 0) or 0)
        except Exception:
            recovery_wins = 0
        if int(self.state.consecutive_stopouts) > 0:
            if recovery_wins <= 0 or int(self.state.consecutive_wins) < recovery_wins:
                factor *= float(stop_factor) ** int(self.state.consecutive_stopouts)

        # IV Percentile Sizing Factor
        iv_factor = self._entry_iv_percentile_factor()
        factor *= float(iv_factor)

        try:
            atr_high = float(getattr(self.cfg, "risk_scale_atr_high", 0.0) or 0.0)
        except Exception:
            atr_high = 0.0
        try:
            atr_factor = float(getattr(self.cfg, "risk_scale_atr_high_factor", 1.0) or 1.0)
        except Exception:
            atr_factor = 1.0
        if atr_val is not None and atr_high > 0 and float(atr_val) >= atr_high:
            factor *= float(atr_factor)

        try:
            qty = int(round(float(qty) * float(factor)))
        except Exception:
            qty = max(base_qty, qty)

        try:
            step = int(getattr(self.cfg, "risk_scale_step_qty", 0) or 0)
        except Exception:
            step = 0
        if step <= 0:
            step = base_qty if base_qty > 0 else 1
        if step > 0:
            qty = int(round(float(qty) / float(step))) * int(step)
            if qty <= 0:
                qty = step

        try:
            min_qty = int(getattr(self.cfg, "risk_scale_min_qty", 0) or 0)
        except Exception:
            min_qty = 0
        try:
            max_qty = int(getattr(self.cfg, "risk_scale_max_qty", 0) or 0)
        except Exception:
            max_qty = 0
        if min_qty <= 0:
            min_qty = step
        if max_qty <= 0:
            max_qty = base_qty
        if min_qty > 0:
            qty = max(min_qty, qty)
        if max_qty > 0 and max_qty < min_qty:
            max_qty = min_qty
        if max_qty > 0:
            qty = min(max_qty, qty)

        try:
            size_mult = float(self._router_size_multiplier())
        except Exception:
            size_mult = 1.0
        size_mult = max(0.25, min(1.5, float(size_mult)))
        if abs(size_mult - 1.0) > 1e-9:
            qty = max(step, int(round(float(qty) * float(size_mult)) / float(step)) * int(step))
            if max_qty > 0:
                qty = min(max_qty, qty)

        # P3: Apply GPT position sizing factor
        try:
            if getattr(self.cfg, "gpt_position_sizing_enabled", False):
                sizing_factor = self._gpt_sizing_factor()
                qt_f = int(qty) * float(sizing_factor)
                qt_f = max(1, int(qt_f))
                lot = int(getattr(self.cfg, "lot_size", 65) or 65)
                if lot > 0:
                    qt_f = (qt_f // lot) * lot
                    if qt_f < 1:
                        qt_f = lot
                qty = int(qt_f)
        except Exception:
            pass
        return int(qty)

    def _router_risk_overrides(self) -> Dict[str, Any]:
        router_dec = getattr(self, "_last_router_decision", None) or {}
        risk = router_dec.get("risk") if isinstance(router_dec, dict) else {}
        if not isinstance(risk, dict):
            return {}
        return dict(risk)

    def _router_size_multiplier(self) -> float:
        risk = self._router_risk_overrides()
        try:
            return float(risk.get("size_multiplier") or 1.0)
        except Exception:
            return 1.0

    def _trade_risk_meta_overrides(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        risk = self._router_risk_overrides()
        out: Dict[str, Any] = {}
        for key in (
            "stop_loss_pct",
            "target_pct",
            "trailing_sl_pct",
            "size_multiplier",
            "dir_sl_atr_mult",
            "dir_tp_atr_mult",
            "dir_trail_atr_mult",
            "cooldown_minutes",
            "max_trades_per_day",
        ):
            value = risk.get(key)
            if value is not None:
                out[f"ml_dynamic_{key}"] = value
        router_dec = getattr(self, "_last_router_decision", None) or {}
        if isinstance(router_dec, dict):
            selected_preset = str(router_dec.get("selected_preset") or "").strip()
            if selected_preset:
                out["ml_dynamic_preset"] = selected_preset
        if extra:
            out.update(extra)
        return out

    def _current_atm_iv(self, call: Optional[dict], put: Optional[dict]) -> Optional[float]:
        vals: List[float] = []
        for opt in (call, put):
            if not isinstance(opt, dict):
                continue
            iv = opt.get("iv")
            try:
                if iv is not None:
                    vals.append(float(iv))
            except Exception:
                continue
        if not vals:
            return None
        return sum(vals) / float(len(vals))

    def _note_iv_change_pct(self, iv: Optional[float]) -> Optional[float]:
        if iv is None:
            return None
        try:
            iv_f = float(iv)
        except Exception:
            return None
        if iv_f <= 0:
            return None
        last_val = self._iv_watch.get("value")
        try:
            last_iv = float(last_val) if last_val is not None else None
        except Exception:
            last_iv = None
        self._iv_watch["value"] = iv_f
        self._iv_watch["ts"] = float(time.time())
        if last_iv is None or last_iv <= 0:
            return None
        return (iv_f - last_iv) / last_iv * 100.0

    def _directional_exposure_qty(self) -> int:
        net = 0
        for tr in list(self.state.open_directional):
            if not isinstance(tr, dict):
                continue
            if bool(tr.get("is_hedge")):
                continue
            name = str(tr.get("name") or "").lower().strip()
            qty = int(tr.get("quantity") or 0)
            if qty <= 0:
                continue
            if name in {"long_call", "short_put"}:
                net += qty
            elif name in {"long_put", "short_call"}:
                net -= qty
        return int(net)

    def _parse_hhmm(self, value: object, *, default: dt_time) -> dt_time:
        s = str(value or "").strip()
        if not s:
            return default
        try:
            parts = s.split(":")
            if len(parts) != 2:
                return default
            hh = int(parts[0])
            mm = int(parts[1])
            if not (0 <= hh <= 23 and 0 <= mm <= 59):
                return default
            return dt_time(hh, mm)
        except Exception:
            return default

    def _vwap(self, candles: List[Candle]) -> Optional[float]:
        pv = 0.0
        vol = 0.0
        any_vol = False
        for c in candles:
            v = c.volume
            if v is None:
                continue
            try:
                vf = float(v)
            except Exception:
                continue
            if vf <= 0:
                continue
            any_vol = True
            pv += float(c.close) * vf
            vol += vf
        if (not any_vol) or vol <= 0:
            return None
        return pv / vol

    def _gpt_should_apply_now(self, *, is_paper: bool) -> bool:
        # gpt_market_commentary_enabled (default False) must gate all GPT commentary calls.
        # When False, no GPT calls are made and therefore HTTP 402 cannot block ML trading.
        try:
            if not bool(getattr(self.cfg, "gpt_market_commentary_enabled", False)):
                return False
            if not bool(getattr(self.cfg, "gpt_enabled", getattr(self.cfg, "gpt_enable", False))):
                return False
        except Exception:
            return False
        try:
            import gpt_advisor as _ga  # type: ignore
            if bool(getattr(_ga, "is_gpt_disabled_for_session", lambda: False)()):
                return False
        except Exception:
            pass

        if is_paper:
            try:
                return bool(getattr(self.cfg, "gpt_apply_paper", True))
            except Exception:
                return True

        return True

    def _gpt_leg_manage_enabled(self, *, is_paper: bool) -> bool:
        """Return True if GPT per-leg management is enabled for this run."""

        if not self._gpt_should_apply_now(is_paper=bool(is_paper)):
            return False

        try:
            return bool(getattr(self.cfg, "gpt_leg_manage", False))
        except Exception:
            return False

    def enable_gpt_leg_ui_visibility_gate(self, enabled: bool = True) -> None:
        self._gpt_leg_require_ui_visible = bool(enabled)

    def mark_trade_ui_visible(self, trade_id: str, *, ts: Optional[float] = None) -> None:
        tid = str(trade_id or "").strip()
        if not tid:
            return
        self._gpt_leg_ui_visible_trades[tid] = float(ts if ts is not None else time.time())

    def _gpt_leg_management_ready(self, *, trade: Dict[str, object], legs: List[dict]) -> bool:
        """Return True when a trade is mature enough for GPT leg management.

        Guard GPT from closing fresh positions before the UI has had time to
        show leg LTP/MTM to the user.
        """

        try:
            raw_min_age = getattr(self.cfg, "gpt_leg_manage_min_age_sec", 90.0)
            min_age = 90.0 if raw_min_age is None else float(raw_min_age)
        except Exception:
            min_age = 90.0
        if min_age < 0:
            min_age = 0.0

        try:
            opened_ts = float(trade.get("opened_ts") or 0.0)
        except Exception:
            opened_ts = 0.0
        if opened_ts > 0 and (float(time.time()) - float(opened_ts)) < float(min_age):
            return False

        if bool(getattr(self, "_gpt_leg_require_ui_visible", False)):
            try:
                trade_id = str(trade.get("trade_id") or "").strip()
            except Exception:
                trade_id = ""
            if not trade_id:
                return False
            try:
                seen_ts = float((self._gpt_leg_ui_visible_trades or {}).get(trade_id) or 0.0)
            except Exception:
                seen_ts = 0.0
            if seen_ts <= 0.0:
                return False

        any_main_leg = False
        for leg in legs:
            if not isinstance(leg, dict) or bool(leg.get("is_hedge")):
                continue
            try:
                qty = int(leg.get("quantity") or 0)
            except Exception:
                qty = 0
            if qty <= 0:
                continue
            any_main_leg = True
            try:
                ltp = leg.get("ltp")
                if ltp is None or float(ltp) <= 0:
                    return False
            except Exception:
                return False

        return any_main_leg

    def _gpt_non_tightening_leg_stop(self, *, leg: dict, proposed_stop: Optional[float]) -> Optional[float]:
        if proposed_stop is None:
            return None

        try:
            entry = float(leg.get("entry_price")) if leg.get("entry_price") is not None else None
        except Exception:
            entry = None
        if entry is None or entry <= 0:
            return float(proposed_stop) if float(proposed_stop) > 0 else None

        try:
            side = str(leg.get("side") or "").strip().upper()
        except Exception:
            side = ""
        if side not in {"BUY", "SELL"}:
            return float(proposed_stop) if float(proposed_stop) > 0 else None

        try:
            stop_pct_default = float(getattr(self.cfg, "premium_mtm_stop_pct", 0.30) or 0.30)
        except Exception:
            stop_pct_default = 0.30
        if stop_pct_default <= 0:
            stop_pct_default = 0.30

        default_stop = float(entry) * (1.0 - float(stop_pct_default)) if side == "BUY" else float(entry) * (1.0 + float(stop_pct_default))
        try:
            existing_stop = float(leg.get("prem_stop")) if leg.get("prem_stop") is not None else None
        except Exception:
            existing_stop = None

        baseline_stop = default_stop
        if existing_stop is not None and existing_stop > 0:
            if side == "BUY":
                baseline_stop = min(float(existing_stop), float(default_stop))
            else:
                baseline_stop = max(float(existing_stop), float(default_stop))

        try:
            proposed = float(proposed_stop)
        except Exception:
            return None
        if proposed <= 0:
            return None

        if side == "BUY" and proposed > float(baseline_stop):
            return float(baseline_stop)
        if side == "SELL" and proposed < float(baseline_stop):
            return float(baseline_stop)
        return float(proposed)

    def _gpt_leg_levels_with_min_distance(
        self,
        *,
        leg: dict,
        stop_price: Optional[float],
        target_price: Optional[float],
    ) -> tuple[Optional[float], Optional[float]]:
        """Keep GPT stop/target far enough from current LTP to avoid instant hits."""

        try:
            ltp = float(leg.get("ltp")) if leg.get("ltp") is not None else None
        except Exception:
            ltp = None
        if ltp is None or ltp <= 0:
            return stop_price, target_price

        try:
            side = str(leg.get("side") or "").strip().upper()
        except Exception:
            side = ""
        if side not in {"BUY", "SELL"}:
            return stop_price, target_price

        try:
            stop_dist = float(getattr(self.cfg, "gpt_leg_manage_min_stop_distance_pct", 0.08) or 0.08)
        except Exception:
            stop_dist = 0.08
        try:
            target_dist = float(getattr(self.cfg, "gpt_leg_manage_min_target_distance_pct", 0.10) or 0.10)
        except Exception:
            target_dist = 0.10
        stop_dist = max(0.0, stop_dist)
        target_dist = max(0.0, target_dist)

        if side == "BUY":
            if stop_price is not None:
                max_stop = float(ltp) * (1.0 - float(stop_dist))
                stop_price = min(float(stop_price), float(max_stop))
            if target_price is not None:
                min_target = float(ltp) * (1.0 + float(target_dist))
                target_price = max(float(target_price), float(min_target))
        else:
            if stop_price is not None:
                min_stop = float(ltp) * (1.0 + float(stop_dist))
                stop_price = max(float(stop_price), float(min_stop))
            if target_price is not None:
                max_target = float(ltp) * (1.0 - float(target_dist))
                if max_target > 0:
                    target_price = min(float(target_price), float(max_target))

        if stop_price is not None and stop_price <= 0:
            stop_price = None
        if target_price is not None and target_price <= 0:
            target_price = None

        return stop_price, target_price

    def _gpt_leg_exits_armed(self, *, trade: Dict[str, object]) -> bool:
        meta = trade.get("meta") if isinstance(trade.get("meta"), dict) else {}
        if not isinstance(meta, dict):
            return True
        try:
            armed_ts = float(meta.get("gpt_leg_exits_armed_ts") or 0.0)
        except Exception:
            armed_ts = 0.0
        if armed_ts <= 0.0:
            return True
        return float(time.time()) >= float(armed_ts)

    def _apply_gpt_leg_exits(self, *, trade: Dict[str, object], position_type: str, legs: List[dict]) -> None:
        """Best-effort: ask GPT for per-leg premium stop/target levels.

        GPT should return extras.leg_exits = list of objects matching legs by token/symbol.
        Each item can provide:
        - stop_price / target_price (premium levels)
        - OR stop_pct / target_pct (0-1) relative to entry_price
        """

        if not isinstance(trade, dict) or not isinstance(legs, list) or not legs:
            return

        if not self._gpt_leg_management_ready(trade=trade, legs=legs):
            return

        meta = trade.get("meta") if isinstance(trade.get("meta"), dict) else {}
        if not isinstance(meta, dict):
            meta = {}

        # Throttle GPT calls per trade.
        now = float(time.time())
        last_ts = 0.0
        try:
            last_ts = float(meta.get("gpt_leg_exits_ts") or 0.0)
        except Exception:
            last_ts = 0.0
        try:
            refresh = float(getattr(self.cfg, "gpt_leg_manage_refresh_sec", 45.0) or 45.0)
        except Exception:
            refresh = 45.0
        if refresh < 5.0:
            refresh = 5.0
        if now - last_ts < refresh:
            return

        # Build a compact, stable payload for the model.
        legs_payload: List[Dict[str, object]] = []
        for lg in legs:
            if not isinstance(lg, dict) or bool(lg.get("is_hedge")):
                continue
            try:
                legs_payload.append(
                    {
                        "symbol": str(lg.get("symbol") or "").strip(),
                        "exchange": str(lg.get("exchange") or "").strip().upper() or None,
                        "token": str(lg.get("token") or "").strip() or None,
                        "side": str(lg.get("side") or "").strip().upper(),
                        "quantity": int(lg.get("quantity") or 0),
                        "entry_price": float(lg.get("entry_price")) if lg.get("entry_price") is not None else None,
                        "ltp": float(lg.get("ltp")) if lg.get("ltp") is not None else None,
                    }
                )
            except Exception:
                continue

        if not legs_payload:
            return

        system_prompt = (
            "You are a trading risk manager for open options positions. "
            "You MUST respond with ONLY valid JSON (no markdown). "
            "Return per-leg stop/target premium levels for the provided legs. "
            "Output schema: {\"decision\":\"HOLD\"|\"EXIT\",\"reason\":string,\"confidence\":number," \
            "\"leg_exits\":[{\"token\"?:string,\"symbol\"?:string,\"exchange\"?:string," \
            "\"stop_price\"?:number,\"target_price\"?:number,\"stop_pct\"?:number,\"target_pct\"?:number}]}. "
            "- stop_price/target_price are option premium levels (same units as LTP). "
            "- stop_pct/target_pct are fractions (e.g. 0.25 = 25%) relative to entry_price. "
            "- For SELL legs, stop should be above entry, target below entry. "
            "If unsure, omit a field rather than guessing."
        )

        proposal = {
            "position_type": str(position_type or "").strip().lower(),
            "trade_id": str(trade.get("trade_id") or "").strip(),
            "name": str(trade.get("name") or trade.get("strategy") or "").strip(),
            "legs": legs_payload,
        }

        advice = self._gpt_control_advise(proposal=proposal, system_prompt=system_prompt, max_tokens=450)
        extras = getattr(advice, "extras", None)
        if not isinstance(extras, dict):
            return
        items = extras.get("leg_exits")
        if not isinstance(items, list) or not items:
            return

        def _as_float(v: object) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        def _match_item(lg: dict, it: dict) -> bool:
            try:
                tok_l = str(lg.get("token") or "").strip()
            except Exception:
                tok_l = ""
            try:
                tok_i = str(it.get("token") or "").strip()
            except Exception:
                tok_i = ""
            if tok_l and tok_i and tok_l == tok_i:
                return True
            try:
                sym_l = str(lg.get("symbol") or "").strip().upper()
            except Exception:
                sym_l = ""
            try:
                sym_i = str(it.get("symbol") or "").strip().upper()
            except Exception:
                sym_i = ""
            if sym_l and sym_i and sym_l == sym_i:
                return True
            return False

        updated_any = False
        for lg in legs:
            if not isinstance(lg, dict) or bool(lg.get("is_hedge")):
                continue
            entry = _as_float(lg.get("entry_price"))
            if entry is None or entry <= 0:
                continue
            side = str(lg.get("side") or "").strip().upper()
            if side not in {"BUY", "SELL"}:
                continue

            chosen: Optional[dict] = None
            for it in items:
                if isinstance(it, dict) and _match_item(lg, it):
                    chosen = it
                    break
            if chosen is None:
                continue

            stop_price = _as_float(chosen.get("stop_price"))
            target_price = _as_float(chosen.get("target_price"))
            stop_pct = _as_float(chosen.get("stop_pct"))
            target_pct = _as_float(chosen.get("target_pct"))

            existing_stop = _as_float(lg.get("prem_stop"))
            ltp_now = _as_float(lg.get("ltp"))
            preserve_hit_stop: Optional[float] = None
            if existing_stop is not None and existing_stop > 0 and ltp_now is not None and ltp_now > 0:
                try:
                    if side == "BUY" and float(ltp_now) <= float(existing_stop):
                        preserve_hit_stop = float(existing_stop)
                    elif side == "SELL" and float(ltp_now) >= float(existing_stop):
                        preserve_hit_stop = float(existing_stop)
                except Exception:
                    preserve_hit_stop = None

            # If pct provided, convert to price levels.
            if (stop_price is None) and (stop_pct is not None) and stop_pct > 0:
                try:
                    if side == "BUY":
                        stop_price = float(entry) * (1.0 - float(stop_pct))
                    else:
                        stop_price = float(entry) * (1.0 + float(stop_pct))
                except Exception:
                    stop_price = None
            if (target_price is None) and (target_pct is not None) and target_pct > 0:
                try:
                    if side == "BUY":
                        target_price = float(entry) * (1.0 + float(target_pct))
                    else:
                        target_price = float(entry) * (1.0 - float(target_pct))
                except Exception:
                    target_price = None

            # Sanity: don't allow negative/zero prices.
            if stop_price is not None and stop_price <= 0:
                stop_price = None
            if target_price is not None and target_price <= 0:
                target_price = None

            stop_price, target_price = self._gpt_leg_levels_with_min_distance(
                leg=lg,
                stop_price=stop_price,
                target_price=target_price,
            )

            if preserve_hit_stop is not None and preserve_hit_stop > 0:
                stop_price = float(preserve_hit_stop)

            if stop_price is None and target_price is None:
                continue

            if stop_price is not None:
                lg["prem_stop"] = float(stop_price)
                updated_any = True
            if target_price is not None:
                lg["prem_target"] = float(target_price)
                updated_any = True

        if updated_any:
            meta["gpt_leg_exits_ts"] = float(now)
            try:
                grace = float(getattr(self.cfg, "gpt_leg_manage_post_update_grace_sec", 20.0) or 20.0)
            except Exception:
                grace = 20.0
            grace = max(0.0, float(grace))
            if grace > 0.0:
                meta["gpt_leg_exits_armed_ts"] = float(now + grace)
            else:
                meta.pop("gpt_leg_exits_armed_ts", None)
            trade["meta"] = meta

    def _leg_exit_hit(self, *, leg: dict) -> Optional[str]:
        """Return reason string if per-leg premium stop/target is hit."""
        if not isinstance(leg, dict) or bool(leg.get("is_hedge")):
            return None

        try:
            side = str(leg.get("side") or "").strip().upper()
        except Exception:
            side = ""
        if side not in {"BUY", "SELL"}:
            return None

        try:
            ltp = float(leg.get("ltp")) if leg.get("ltp") is not None else None
        except Exception:
            ltp = None
        if ltp is None:
            return None

        try:
            stop_p = float(leg.get("prem_stop")) if leg.get("prem_stop") is not None else None
        except Exception:
            stop_p = None
        try:
            tgt_p = float(leg.get("prem_target")) if leg.get("prem_target") is not None else None
        except Exception:
            tgt_p = None

        if side == "BUY":
            if stop_p is not None and float(ltp) <= float(stop_p):
                return "leg_stop"
            if tgt_p is not None and float(ltp) >= float(tgt_p):
                return "leg_target"
        else:
            if stop_p is not None and float(ltp) >= float(stop_p):
                return "leg_stop"
            if tgt_p is not None and float(ltp) <= float(tgt_p):
                return "leg_target"
        return None

    def _get_paper_exit_price(
        self,
        symbol: str,
        exchange: Optional[str],
        leg_side: str,
        qty: int,
        *,
        is_entry_side_buy: bool,
    ) -> Tuple[Optional[float], float]:
        """Return (paper_exit_price, spread_deduct) for a paper-mode leg exit.

        Realistic paper pricing:
        - Closing a BUY (long) leg → we SELL → use bid (buyer's price)
        - Closing a SELL (short) leg → we BUY → use ask (seller's price)

        Uses cfg.paper_allow_ltp_fallback when bid/ask is unavailable.

        Returns (None, 0.0) when bid/ask is missing and LTP fallback is disabled.
        """
        spread_deduct: float = 0.0
        use_bid_ask = bool(getattr(self.cfg, "paper_use_bid_ask_execution", True))
        allow_ltp_fallback = bool(getattr(self.cfg, "paper_allow_ltp_fallback", False))
        ltp_fallback_penalty = float(getattr(self.cfg, "paper_ltp_fallback_spread_pct", 0.05) or 0.05)

        if not use_bid_ask:
            # Fall back to LTP
            ltp_val = self._try_get_ltp_for_leg({"symbol": symbol, "exchange": exchange})
            return (ltp_val, 0.0) if ltp_val else (None, 0.0)

        bid_raw, ask_raw, ltp_raw = self.client.get_bid_ask(symbol, exchange_hint=exchange)

        if is_entry_side_buy:
            # Closing BUY position = we SELL → receive bid
            if bid_raw is None:
                if allow_ltp_fallback and ltp_raw is not None:
                    return (float(ltp_raw) * (1.0 - ltp_fallback_penalty), 0.0)
                return (None, 0.0)
            exit_price = float(bid_raw)
            if ask_raw is not None:
                spread_deduct = float((float(ask_raw) - float(bid_raw)) * float(qty))
        else:
            # Closing SELL position = we BUY → pay ask
            if ask_raw is None:
                if allow_ltp_fallback and ltp_raw is not None:
                    return (float(ltp_raw) * (1.0 + ltp_fallback_penalty), 0.0)
                return (None, 0.0)
            exit_price = float(ask_raw)
            if bid_raw is not None:
                spread_deduct = float((float(ask_raw) - float(bid_raw)) * float(qty))

        return (exit_price, spread_deduct)

    def _close_single_leg(self, *, trade: Dict[str, object], leg: Dict[str, object], position_type: str, reason: str) -> float:
        """Close exactly one leg (best-effort). Returns realized P&L for that leg."""
        symbol = str(leg.get("symbol") or "")
        token = str(leg.get("token") or "").strip() or None
        exchange = str(leg.get("exchange") or "").strip() or None
        side = str(leg.get("side") or "").strip().upper()
        qty = int(leg.get("quantity") or 0)
        if not symbol or side not in {"BUY", "SELL"} or qty <= 0:
            return 0.0

        close_side = "BUY" if side == "SELL" else "SELL"

        # For paper trades, use bid (for closing BUY) or ask (for closing SELL).
        # Block if bid/ask is unavailable — do NOT silently fall back to LTP.
        paper_exit_price: Optional[float] = None
        paper_spread_deduct: float = 0.0
        price_source: str = "bid" if side == "BUY" else "ask"
        if not self.cfg.enable_live_trading:
            bid_raw, ask_raw, _ = self.client.get_bid_ask(symbol, exchange_hint=exchange)
            if side == "BUY":
                # Closing a BUY position = selling → use bid (what we receive).
                if bid_raw is None:
                    print(f"[PAPER][BLOCKED] Close leg {symbol}: bid price unavailable — bid/ask required")
                    return 0.0
                paper_exit_price = float(bid_raw)
                if ask_raw is not None and paper_exit_price is not None:
                    paper_spread_deduct = float((float(ask_raw) - float(bid_raw)) * float(qty))
            else:
                # Closing a SELL position = buying back → use ask (what we pay).
                if ask_raw is None:
                    print(f"[PAPER][BLOCKED] Close leg {symbol}: ask price unavailable — bid/ask required")
                    return 0.0
                paper_exit_price = float(ask_raw)
                if bid_raw is not None:
                    paper_spread_deduct = float((float(ask_raw) - float(bid_raw)) * float(qty))
            print(f"[PAPER] Close leg: {close_side} {symbol} x{qty} @ {paper_exit_price:.2f} ({price_source}) ({reason})")
        else:
            try:
                self.client.place_order(
                    symbol=symbol,
                    side=close_side,
                    quantity=qty,
                    exchange=exchange,
                    symbol_token=token,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to close leg {symbol}: {exc}")
                return 0.0

        entry = leg.get("entry_price")
        try:
            entry_f = float(entry) if entry is not None else None
        except Exception:
            entry_f = None

        if not self.cfg.enable_live_trading:
            cur = paper_exit_price
        else:
            cur = self._try_get_ltp_for_leg(leg)
        try:
            leg["exit_price"] = cur
        except Exception:
            pass
        try:
            leg["execution_price_source"] = price_source if not self.cfg.enable_live_trading else "ltp"
        except Exception:
            pass

        realized = 0.0
        # Track gross P&L before costs for reporting.
        gross_pnl = 0.0
        total_slippage = 0.0
        total_brokerage = 0.0
        if entry_f is not None and cur is not None:
            sign = 1.0 if side == "BUY" else -1.0
            gross_pnl = (float(cur) - float(entry_f)) * sign * float(qty)
            realized = gross_pnl
            # Spread cost: already embedded in ask/bid execution prices, so NOT deducted again.
            # Only deduct slippage + brokerage here (paper cost model).
            try:
                from cost_model import estimate_paper_execution_costs
                _exit_costs = estimate_paper_execution_costs(
                    execution_price=float(cur),
                    quantity=int(qty),
                    slippage_pct=float(getattr(self.cfg, "paper_slippage_pct", 0.001) or 0.001),
                    extra_market_impact_pct=float(getattr(self.cfg, "paper_extra_market_impact_pct", 0.0) or 0.0),
                    apply_brokerage=bool(getattr(self.cfg, "paper_apply_brokerage_costs", True)),
                    cost_model_source=str(getattr(self.cfg, "paper_cost_model_source", "cost_model_assumptions") or "cost_model_assumptions"),
                    side=close_side,
                )
                total_slippage = float(_exit_costs.get("slippage_cost", 0.0))
                total_brokerage = float(_exit_costs.get("total_brokerage_charges", 0.0))
                realized -= total_slippage + total_brokerage
            except Exception:
                pass

        # Mark leg as closed in-place.
        try:
            leg["quantity"] = 0
        except Exception:
            pass

        if self._has_event_sink():
            trade_id = str(trade.get("trade_id") or "")
            name = str(trade.get("name") or "")
            out_leg = dict(leg)
            out_leg["quantity"] = int(qty)
            if cur is not None:
                out_leg["exit_price"] = float(cur)
            self._emit(
                TradeLogEvent(
                    ts=time.time(),
                    event="PARTIAL_CLOSE",
                    trade_id=trade_id or "(unknown)",
                    position_type=str(position_type or ""),
                    name=name,
                    legs=[out_leg],
                    realized=float(realized),
                    reason=str(reason or ""),
                    gross_pnl=float(gross_pnl),
                    net_pnl=float(realized),
                    execution_price_source=price_source if not self.cfg.enable_live_trading else "ltp",
                    spread_cost=paper_spread_deduct,  # logged for audit (embedded, not double-deducted)
                    slippage_cost=total_slippage,
                    brokerage_cost=total_brokerage,
                )
            )

        # Track realized P&L in state for risk limits.
        try:
            self.state.realized_pnl += float(realized)
        except Exception:
            pass
        return float(realized)


    # ---------------------------------------------------------------------------
    # P1: GPT-Powered Dynamic Exit Management
    # ---------------------------------------------------------------------------
    def _gpt_exit_management(self, *, trade: Dict[str, object], candles: List[Candle], spot: float) -> None:
        """Review one open directional trade + market snapshot and apply GPT exit advice.

        Stored actions:
          - EXIT_NOW       -> immediately close via _close_single_leg
          - TRAIL_TIGHTER  -> set tr["_gpt_trail_override"] for tighter trailing
          - PARTIAL_EXIT   -> reduce position quantity
          - HOLD           -> no action
        """
        try:
            enabled = bool(getattr(self.cfg, "gpt_exit_management", False))
        except Exception:
            enabled = False
        if not enabled:
            return
        try:
            refresh = float(getattr(self.cfg, "gpt_exit_management_refresh_sec", 60.0) or 60.0)
        except Exception:
            refresh = 60.0
        now = time.time()
        if refresh > 0 and (now - self._gpt_exit_mgmt_last_run) < refresh:
            return
        self._gpt_exit_mgmt_last_run = now

        legs = trade.get("legs")
        if not isinstance(legs, list) or not legs:
            return

        # Build a lightweight snapshot for GPT
        leg_snapshots = []
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            leg_snapshots.append({
                "symbol": str(leg.get("symbol", "")),
                "side": str(leg.get("side", "")),
                "qty": int(leg.get("quantity", 0) or 0),
                "entry_price": float(leg.get("entry_price", 0) or 0),
                "ltp": float(leg.get("ltp", 0) or 0),
                "option_type": str(leg.get("option_type", "") or "").upper(),
                "strike": float(leg.get("strike", 0) or 0),
            })

        mtm = self._compute_legs_mtm_cached(legs)
        try:
            atr_val = float(self._cached_atr or 0.0)
        except Exception:
            atr_val = 0.0

        proposal = {
            "mode": "EXIT_MANAGEMENT",
            "trade_id": str(trade.get("trade_id", "")),
            "trade_name": str(trade.get("name", "")),
            "position_type": "directional",
            "spot": float(spot),
            "atr": float(atr_val),
            "legs": leg_snapshots,
            "mtm": float(mtm) if mtm is not None else None,
            "elapsed_sec": float(now - float(trade.get("opened_ts", now))),
        }

        sys_prompt = (
            "You are an options exit manager. Respond with ONLY valid JSON. "
            "Review the open trade and choose ONE action: "
            "HOLD (no action), TRAIL_TIGHTER (tighten trailing stop), "
            "EXIT_NOW (close entire position immediately), or "
            "PARTIAL_EXIT (reduce position by qty). "
            "If PARTIAL_EXIT, include qty (positive integer, leave at least 1 lot). "
            "If TRAIL_TIGHTER, include trail_mult (float, lower than current trail ATR multiple). "
            "Be conservative: prefer HOLD when uncertain. "
            "Schema: {\"decision\":\"TAKE\"|\"SKIP\",\"action\":\"HOLD\"|\"TRAIL_TIGHTER\"|\"EXIT_NOW\"|\"PARTIAL_EXIT\","
            "\"qty\":number,\"trail_mult\":number,\"reason\":string,\"confidence\":number}"
        )

        try:
            import gpt_advisor as ga
        except ImportError:
            try:
                from src import gpt_advisor as ga  # type: ignore[no-redef,import-not-found]
            except ImportError:
                return

        api_key = os.getenv("MSTOCK_GPT_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or ""
        base_url = os.getenv("MSTOCK_GPT_API_BASE_URL", "") or None
        model = os.getenv("MSTOCK_GPT_MODEL", "") or "gpt-4o-mini"

        adv = ga.advise_trade(
            proposal=proposal,
            model=model,
            api_key=api_key,
            base_url=base_url,
            system_prompt=sys_prompt,
            timeout_sec=float(getattr(self.cfg, "gpt_timeout_sec", 45.0) or 45.0),
            max_tokens=200,
        )
        if adv is None or str(getattr(adv, "decision", "")).strip().upper() != "TAKE":
            return

        extras = getattr(adv, "extras", {}) if isinstance(getattr(adv, "extras", {}), dict) else {}
        action = str(extras.get("action") or "HOLD").strip().upper() or "HOLD"

        if action == "EXIT_NOW":
            # Close all legs
            for leg in legs:
                if not isinstance(leg, dict):
                    continue
                try:
                    self._close_single_leg(
                        trade=trade, leg=leg, position_type="directional",
                        reason=f"gpt_exit:{extras.get('reason','')}"
                    )
                except Exception:
                    pass
        elif action == "TRAIL_TIGHTER":
            try:
                trail_val = float(extras.get("trail_mult") or 0.0)
                if trail_val > 0:
                    trade["_gpt_trail_override"] = float(trail_val)
                    trade["_gpt_trail_override_ts"] = time.time()
            except Exception:
                pass
        elif action == "PARTIAL_EXIT":
            try:
                partial_qty = abs(int(float(extras.get("qty") or 0)))
            except Exception:
                partial_qty = 0
            if partial_qty <= 0:
                return
            for leg in legs:
                if not isinstance(leg, dict):
                    continue
                try:
                    cur_qty = abs(int(leg.get("quantity") or 0))
                except Exception:
                    cur_qty = 0
                if cur_qty <= 0:
                    continue
                exit_qty = min(partial_qty, cur_qty)
                if exit_qty <= 0 or exit_qty >= cur_qty:
                    continue
                # We close exit_qty of this leg via _close_single_leg
                # Store original side in case we need to invert
                try:
                    self._close_single_leg(
                        trade=trade, leg=leg, position_type="directional",
                        reason=f"gpt_partial:{extras.get('reason','')}"
                    )
                except Exception:
                    pass
                # Reduce the remaining quantity
                try:
                    leg["quantity"] = int(cur_qty - exit_qty)
                except Exception:
                    pass

    # ---------------------------------------------------------------------------
    # P2: Multi-Call Strategy Voting
    # ---------------------------------------------------------------------------
    def _gpt_multi_call_vote(self, *, candidates: List[str], snapshot: Dict[str, object]) -> Optional[str]:
        """Make multiple GPT calls with slight temperature variations, pick consensus."""
        try:
            enabled = bool(getattr(self.cfg, "gpt_strategy_voting_enabled", False))
        except Exception:
            enabled = False
        if not enabled or not candidates:
            return None
        try:
            num_calls = int(getattr(self.cfg, "gpt_strategy_voting_calls", 3) or 3)
        except Exception:
            num_calls = 3
        num_calls = max(2, min(5, int(num_calls)))
        try:
            base_temp = float(getattr(self.cfg, "gpt_strategy_voting_temperature", 0.3) or 0.3)
        except Exception:
            base_temp = 0.3

        try:
            import gpt_advisor as ga
        except ImportError:
            try:
                from src import gpt_advisor as ga  # type: ignore[no-redef,import-not-found]
            except ImportError:
                return None

        api_key = os.getenv("MSTOCK_GPT_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or ""
        base_url = os.getenv("MSTOCK_GPT_API_BASE_URL", "") or None
        model = os.getenv("MSTOCK_GPT_MODEL", "") or "gpt-4o-mini"
        timeout = float(getattr(self.cfg, "gpt_timeout_sec", 45.0) or 45.0)

        sys_prompt = (
            "You are a strategy selection expert. Respond with ONLY valid JSON. "
            "Pick the best strategy from the candidate list for current market conditions. "
            "Return the chosen strategy name exactly as it appears in candidates. "
            "Schema: {\"decision\":\"TAKE\"|\"SKIP\",\"strategy\":string,\"reason\":string,\"confidence\":number}"
        )

        votes: Dict[str, int] = {}
        for i in range(int(num_calls)):
            temp = float(base_temp) + (float(i) * 0.05)
            try:
                adv = ga.advise_trade(
                    proposal={
                        "mode": "STRATEGY_VOTE",
                        "candidates": list(candidates),
                        "snapshot": dict(snapshot) if isinstance(snapshot, dict) else {},
                        "vote_id": int(i),
                    },
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    system_prompt=sys_prompt,
                    timeout_sec=float(timeout),
                    temperature=float(min(temp, 1.0)),
                    max_tokens=150,
                )
            except Exception:
                continue
            if adv is None:
                continue
            dec = str(getattr(adv, "decision", "")).strip().upper()
            extras = getattr(adv, "extras", {}) if isinstance(getattr(adv, "extras", {}), dict) else {}
            chosen = str(extras.get("strategy") or "").strip()
            if dec == "TAKE" and chosen and chosen in candidates:
                votes[chosen] = int(votes.get(chosen, 0)) + 1

        if not votes:
            return None

        # Return the strategy with most votes
        best = max(votes, key=lambda k: (int(votes[k]), k))
        return str(best)

    # ---------------------------------------------------------------------------
    # P3: GPT-Based Position Sizing Factor
    # ---------------------------------------------------------------------------
    def _gpt_sizing_factor(self) -> float:
        """Return a sizing multiplier (0.5-2.0) based on GPT's market assessment.

        Cached for 60 seconds to avoid excessive API calls.
        Also consumed by regime sizing_mult when set.
        """
        cache: Dict[str, object] = getattr(self, "_gpt_sizing_factor_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_gpt_sizing_factor_cache", cache)

        now = time.time()
        try:
            cached_val = float(cache.get("value") or 0.0)
            cached_ts = float(cache.get("ts") or 0.0)
            if cached_val > 0 and (now - cached_ts) < 60.0:
                # Apply regime sizing_mult on top of cached GPT factor
                regime = self._gpt_regime_actions if isinstance(self._gpt_regime_actions, dict) else {}
                try:
                    sizing_mult = float(regime.get("sizing_mult") or 1.0)
                except Exception:
                    sizing_mult = 1.0
                return float(cached_val) * float(sizing_mult)
        except Exception:
            pass

        # Check if feature is enabled
        try:
            enabled = bool(getattr(self.cfg, "gpt_position_sizing_enabled", False))
        except Exception:
            enabled = False
        if not enabled:
            return 1.0

        try:
            import gpt_advisor as ga
        except ImportError:
            try:
                from src import gpt_advisor as ga  # type: ignore[no-redef,import-not-found]
            except ImportError:
                return 1.0

        api_key = os.getenv("MSTOCK_GPT_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or ""
        base_url = os.getenv("MSTOCK_GPT_API_BASE_URL", "") or None
        model = os.getenv("MSTOCK_GPT_MODEL", "") or "gpt-4o-mini"
        timeout = float(getattr(self.cfg, "gpt_timeout_sec", 45.0) or 45.0)

        # Build a minimal context snapshot for GPT
        snapshot = {
            "mode": "SIZING",
            "winrate": {},
            "streak": {},
        }
        try:
            total = sum(self._strategy_winrate.get(s, {}).get("wins", 0) for s in self._strategy_winrate) + \
                    sum(self._strategy_winrate.get(s, {}).get("losses", 0) for s in self._strategy_winrate)
            wins = sum(self._strategy_winrate.get(s, {}).get("wins", 0) for s in self._strategy_winrate)
            if total > 0:
                snapshot["winrate"] = {"wins": int(wins), "total": int(total), "pct": float(wins / total * 100)}
        except Exception:
            pass
        try:
            snapshot["streak"] = {
                "consecutive_wins": int(self.state.consecutive_wins),
                "consecutive_stopouts": int(self.state.consecutive_stopouts),
            }
        except Exception:
            pass

        sys_prompt = (
            "You are a position sizing advisor. Respond with ONLY valid JSON. "
            "Return a sizing_factor (float, 0.5 to 2.0) based on current conditions. "
            "1.0 = normal, <1.0 = reduce size, >1.0 = increase size (capped at 2.0). "
            "Be conservative: factor 1.0 when uncertain. "
            "Schema: {\"decision\":\"TAKE\"|\"SKIP\",\"sizing_factor\":number,\"reason\":string,\"confidence\":number}"
        )

        try:
            adv = ga.advise_trade(
                proposal=snapshot,
                model=model,
                api_key=api_key,
                base_url=base_url,
                system_prompt=sys_prompt,
                timeout_sec=float(timeout),
                max_tokens=120,
            )
        except Exception:
            adv = None

        factor = 1.0
        if adv is not None and str(getattr(adv, "decision", "")).strip().upper() == "TAKE":
            extras = getattr(adv, "extras", {}) if isinstance(getattr(adv, "extras", {}), dict) else {}
            try:
                raw = float(extras.get("sizing_factor") or 1.0)
                factor = max(0.5, min(2.0, float(raw)))
            except Exception:
                factor = 1.0

        # Cache
        cache["value"] = float(factor)
        cache["ts"] = time.time()

        # Apply regime sizing_mult
        try:
            regime = self._gpt_regime_actions if isinstance(self._gpt_regime_actions, dict) else {}
            sizing_mult = float(regime.get("sizing_mult") or 1.0)
        except Exception:
            sizing_mult = 1.0
        return float(factor) * float(sizing_mult)

    # ---------------------------------------------------------------------------
    # P4: Regime Change Detection / Proactive Risk Actions
    # ---------------------------------------------------------------------------
    def _gpt_regime_monitor(self) -> None:
        """Detect market regime changes via GPT and apply proactive risk actions.

        May set:
          - self._entries_paused = True  (pause new entries)
          - self._gpt_regime_actions["stops_mult"] (float, e.g. 0.7)
          - self._gpt_regime_actions["sizing_mult"] (float, e.g. 0.8)
        Resume when GPT returns risk_level=normal.
        """
        try:
            if str(getattr(self.cfg, "strategy_name", "") or "").strip().lower() == "auto":
                return
        except Exception:
            return
        try:
            enabled = bool(getattr(self.cfg, "gpt_regime_monitor_enabled", False))
        except Exception:
            enabled = False
        if not enabled:
            return
        try:
            interval = float(getattr(self.cfg, "gpt_regime_monitor_interval_sec", 300.0) or 300.0)
        except Exception:
            interval = 300.0
        now = time.time()
        if interval > 0 and (now - self._gpt_regime_last_run) < interval:
            return
        self._gpt_regime_last_run = now

        try:
            import gpt_advisor as ga
        except ImportError:
            try:
                from src import gpt_advisor as ga  # type: ignore[no-redef,import-not-found]
            except ImportError:
                return

        api_key = os.getenv("MSTOCK_GPT_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or ""
        base_url = os.getenv("MSTOCK_GPT_API_BASE_URL", "") or None
        model = os.getenv("MSTOCK_GPT_MODEL", "") or "gpt-4o-mini"
        timeout = float(getattr(self.cfg, "gpt_timeout_sec", 45.0) or 45.0)

        # Build regime snapshot
        snapshot = {
            "mode": "REGIME_MONITOR",
            "open_trades": len(self.state.open_directional) + len(self.state.open_multi),
            "today_pnl": float(self.state.realized_pnl),
            "today_trades": int(self.state.trades_today),
            "stopouts_today": int(self.state.stopouts_today),
        }

        sys_prompt = (
            "You are a market regime monitor. Respond with ONLY valid JSON. "
            "Assess current conditions and return a risk_level: normal, cautious, or paused. "
            "If paused, set entries_paused=true. "
            "Optionally include stops_mult (0.5-1.5) and sizing_mult (0.5-1.5). "
            "Be conservative: default to normal when uncertain. "
            "Schema: {\"decision\":\"TAKE\"|\"SKIP\",\"risk_level\":\"normal\"|\"cautious\"|\"paused\","
            "\"entries_paused\":bool,\"stops_mult\":number,\"sizing_mult\":number,"
            "\"reason\":string,\"confidence\":number}"
        )

        try:
            adv = ga.advise_trade(
                proposal=snapshot,
                model=model,
                api_key=api_key,
                base_url=base_url,
                system_prompt=sys_prompt,
                timeout_sec=float(timeout),
                max_tokens=180,
            )
        except Exception:
            adv = None

        if adv is None or str(getattr(adv, "decision", "")).strip().upper() != "TAKE":
            return

        extras = getattr(adv, "extras", {}) if isinstance(getattr(adv, "extras", {}), dict) else {}
        risk_level = str(extras.get("risk_level") or "normal").strip().lower() or "normal"

        if risk_level == "paused":
            self._entries_paused = True
            self._note_entry_blocked(f"regime_paused:{extras.get('reason','')}")
        elif risk_level == "cautious":
            self._entries_paused = False
        else:
            self._entries_paused = False

        actions: Dict[str, object] = {}
        try:
            stops_mult = float(extras.get("stops_mult") or 1.0)
            actions["stops_mult"] = max(0.5, min(1.5, float(stops_mult)))
        except Exception:
            pass
        try:
            sizing_mult = float(extras.get("sizing_mult") or 1.0)
            actions["sizing_mult"] = max(0.5, min(1.5, float(sizing_mult)))
        except Exception:
            pass
        actions["risk_level"] = str(risk_level)
        actions["reason"] = str(extras.get("reason") or "")
        actions["ts"] = float(time.time())
        self._gpt_regime_actions = actions

    # ---------------------------------------------------------------------------
    # P5: Enhanced Strike Selection with Full Greeks Context
    # ---------------------------------------------------------------------------
    def _gpt_strike_selection(self, *, chain: List[dict], spot: float, strategy: str) -> Optional[Dict[str, object]]:
        """Pass live IV/greeks data to GPT for strike selection and wing width advice."""
        try:
            enabled = bool(getattr(self.cfg, "gpt_strike_selection_enabled", False))
        except Exception:
            enabled = False
        if not enabled or not chain:
            return None

        try:
            import gpt_advisor as ga
        except ImportError:
            try:
                from src import gpt_advisor as ga  # type: ignore[no-redef,import-not-found]
            except ImportError:
                return None

        api_key = os.getenv("MSTOCK_GPT_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or ""
        base_url = os.getenv("MSTOCK_GPT_API_BASE_URL", "") or None
        model = os.getenv("MSTOCK_GPT_MODEL", "") or "gpt-4o-mini"
        timeout = float(getattr(self.cfg, "gpt_timeout_sec", 45.0) or 45.0)

        # Build a compact chain preview with greeks
        chain_preview = []
        for row in chain[:30]:  # limit to 30 rows
            if not isinstance(row, dict):
                continue
            try:
                iv = float(row.get("iv") or 0.0)
                theta_val = self._estimate_option_theta_from_chain_row(row, float(spot))
            except Exception:
                iv = 0.0
                theta_val = None
            chain_preview.append({
                "symbol": str(row.get("symbol", "")),
                "option_type": str(row.get("option_type", "") or "").upper(),
                "strike": float(row.get("strike", 0) or 0),
                "ltp": float(row.get("ltp", 0) or 0),
                "iv": float(iv),
                "theta": float(theta_val) if theta_val is not None else None,
                "delta": float(row.get("delta", 0) or 0) if row.get("delta") is not None else None,
            })

        proposal = {
            "mode": "STRIKE_SELECTION",
            "strategy": str(strategy),
            "spot": float(spot),
            "chain": chain_preview,
        }

        sys_prompt = (
            "You are an options strike selector. Respond with ONLY valid JSON. "
            "Given the option chain and strategy, suggest optimal strikes. "
            "Include suggested_strike (float), wing_width (float, if applicable), "
            "and reason. "
            "Return TAKE if you can suggest strikes, SKIP if uncertain. "
            "Schema: {\"decision\":\"TAKE\"|\"SKIP\",\"suggested_strike\":number,"
            "\"wing_width\":number,\"reason\":string,\"confidence\":number}"
        )

        try:
            adv = ga.advise_trade(
                proposal=proposal,
                model=model,
                api_key=api_key,
                base_url=base_url,
                system_prompt=sys_prompt,
                timeout_sec=float(timeout),
                max_tokens=200,
            )
        except Exception:
            return None

        if adv is None or str(getattr(adv, "decision", "")).strip().upper() != "TAKE":
            return None

        extras = getattr(adv, "extras", {}) if isinstance(getattr(adv, "extras", {}), dict) else {}
        result: Dict[str, object] = {}
        try:
            suggested = float(extras.get("suggested_strike") or 0.0)
            if suggested > 0:
                result["suggested_strike"] = float(suggested)
        except Exception:
            pass
        try:
            wing = float(extras.get("wing_width") or 0.0)
            if wing > 0:
                result["wing_width"] = float(wing)
        except Exception:
            pass
        result["reason"] = str(extras.get("reason") or "")
        result["gpt_selected"] = True

        return result if result.get("suggested_strike") or result.get("wing_width") else None

    # ---------------------------------------------------------------------------
    # P6: What-If Scenario Analysis
    # ---------------------------------------------------------------------------
    def _gpt_whatif_analysis(self) -> None:
        """Periodic GPT call: analyze risk of +/-1%, +/-2% moves on open positions."""
        try:
            if str(getattr(self.cfg, "strategy_name", "") or "").strip().lower() == "auto":
                return
        except Exception:
            return
        try:
            enabled = bool(getattr(self.cfg, "gpt_whatif_enabled", False))
        except Exception:
            enabled = False
        if not enabled:
            return
        try:
            interval = float(getattr(self.cfg, "gpt_whatif_interval_sec", 120.0) or 120.0)
        except Exception:
            interval = 120.0
        now = time.time()
        if interval > 0 and (now - self._gpt_whatif_last_run) < interval:
            return
        self._gpt_whatif_last_run = now

        # Gather open trades
        open_legs = []
        all_mtm = 0.0
        for trade in list(self.state.open_directional) + list(self.state.open_multi):
            if not isinstance(trade, dict):
                continue
            legs = trade.get("legs")
            if not isinstance(legs, list):
                continue
            for leg in legs:
                if not isinstance(leg, dict):
                    continue
                try:
                    qty = abs(int(leg.get("quantity") or 0))
                except Exception:
                    qty = 0
                if qty <= 0:
                    continue
                open_legs.append({
                    "symbol": str(leg.get("symbol", "")),
                    "side": str(leg.get("side", "")),
                    "qty": int(qty),
                    "entry": float(leg.get("entry_price", 0) or 0),
                    "ltp": float(leg.get("ltp", 0) or 0),
                    "strike": float(leg.get("strike", 0) or 0),
                    "option_type": str(leg.get("option_type", "") or "").upper(),
                })
                try:
                    all_mtm += float(leg.get("ltp", 0) or 0)
                except Exception:
                    pass

        if not open_legs:
            return

        try:
            import gpt_advisor as ga
        except ImportError:
            try:
                from src import gpt_advisor as ga  # type: ignore[no-redef,import-not-found]
            except ImportError:
                return

        api_key = os.getenv("MSTOCK_GPT_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or ""
        base_url = os.getenv("MSTOCK_GPT_API_BASE_URL", "") or None
        model = os.getenv("MSTOCK_GPT_MODEL", "") or "gpt-4o-mini"
        timeout = float(getattr(self.cfg, "gpt_timeout_sec", 45.0) or 45.0)

        # Estimate delta for each leg (rough, using ATM IV approximation)
        for leg in open_legs:
            try:
                if "delta" not in leg or leg["delta"] is None:
                    ot = 1 if str(leg.get("option_type", "")).upper() == "CE" else -1
                    # Rough approximation: delta ≈ N(d1) ≈ 0.5 for ATM
                    spot_approx = float(leg.get("strike") or 0)
                    if spot_approx > 0:
                        moneyness = float(leg.get("ltp") or 0) / spot_approx
                        if moneyness > 0.02:
                            leg["delta"] = 0.5 * (1.0 if ot == 1 else -1.0)
                        elif moneyness > 0.01:
                            leg["delta"] = 0.3 * (1.0 if ot == 1 else -1.0)
                        else:
                            leg["delta"] = 0.15 * (1.0 if ot == 1 else -1.0)
            except Exception:
                pass

        proposal = {
            "mode": "WHATIF_ANALYSIS",
            "open_legs": open_legs,
            "estimated_total_exposure": float(all_mtm),
        }

        sys_prompt = (
            "You are a portfolio risk analyst. Respond with ONLY valid JSON. "
            "Analyze the open options positions for +/-1% and +/-2% spot moves. "
            "Identify the most at-risk positions. "
            "Optionally recommend action: EXIT or HEDGE for specific positions. "
            "Schema: {\"decision\":\"TAKE\"|\"SKIP\",\"risk_assessment\":string,"
            "\"recommended_action\":\"HOLD\"|\"EXIT\"|\"HEDGE\","
            "\"affected_symbols\":[string],\"reason\":string,\"confidence\":number}"
        )

        try:
            adv = ga.advise_trade(
                proposal=proposal,
                model=model,
                api_key=api_key,
                base_url=base_url,
                system_prompt=sys_prompt,
                timeout_sec=float(timeout),
                max_tokens=250,
            )
        except Exception:
            return

        if adv is None:
            return

        dec = str(getattr(adv, "decision", "")).strip().upper()
        extras = getattr(adv, "extras", {}) if isinstance(getattr(adv, "extras", {}), dict) else {}
        action = str(extras.get("recommended_action") or "HOLD").strip().upper() or "HOLD"

        if dec == "TAKE" and action == "EXIT":
            # GPT recommends closing specific symbols - log for now
            affected = extras.get("affected_symbols")
            if isinstance(affected, list):
                reason = f"gpt_whatif:{extras.get('reason','')}"
                for trade in list(self.state.open_directional) + list(self.state.open_multi):
                    if not isinstance(trade, dict):
                        continue
                    legs = trade.get("legs")
                    if not isinstance(legs, list):
                        continue
                    for leg in legs:
                        if not isinstance(leg, dict):
                            continue
                        sym = str(leg.get("symbol", "")).strip().upper()
                        if any(sym.endswith(a.upper()) for a in affected if isinstance(a, str)):
                            try:
                                self._close_single_leg(
                                    trade=trade, leg=leg,
                                    position_type=str(trade.get("position_type", "directional")),
                                    reason=reason,
                                )
                            except Exception:
                                pass
        elif dec == "TAKE" and action == "HEDGE":
            # Log the recommendation but don't auto-hedge (too risky)
            print(f"[GPT][WHATIF] Hedge recommended: {extras.get('reason','')}")
    def _gpt_control_advise(
        self,
        *,
        proposal: Dict[str, Any],
        system_prompt: Optional[str] = None,
        max_tokens: int = 300,
    ):
        """Best-effort GPT call wrapper used by both entry and trade-control logic.

        Returns a GPTAdvice-like object. On errors, returns UNKNOWN.
        """

        gpt_advisor = None
        try:
            import gpt_advisor as _ga  # type: ignore

            gpt_advisor = _ga
        except Exception:
            try:
                from src import gpt_advisor as _ga  # type: ignore

                gpt_advisor = _ga
            except Exception:
                gpt_advisor = None

        if gpt_advisor is None:
            try:
                return None
            except Exception:
                return None
        try:
            if bool(getattr(gpt_advisor, "is_gpt_disabled_for_session", lambda: False)()):
                return None
        except Exception:
            pass

        try:
            api_key = (os.getenv("MSTOCK_GPT_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        except Exception:
            api_key = ""
        try:
            base_url = (os.getenv("MSTOCK_GPT_API_BASE_URL") or "").strip() or None
        except Exception:
            base_url = None
        try:
            model = (os.getenv("MSTOCK_GPT_MODEL") or "").strip() or "gpt-4o-mini"
        except Exception:
            model = "gpt-4o-mini"
        try:
            timeout_sec = float(getattr(self.cfg, "gpt_timeout_sec", 45.0) or 45.0)
        except Exception:
            timeout_sec = 45.0

        try:
            adv = gpt_advisor.advise_trade(
                proposal=dict(proposal or {}),
                model=str(model),
                api_key=str(api_key),
                base_url=base_url,
                timeout_sec=float(timeout_sec),
                max_tokens=int(max_tokens),
                system_prompt=system_prompt,
            )
            try:
                if isinstance(adv, dict):
                    decision = str(adv.get("decision", "") or "").strip().upper()
                    reason = str(adv.get("reason", "") or "").strip().lower()
                else:
                    decision = str(getattr(adv, "decision", "") or "").strip().upper()
                    reason = str(getattr(adv, "reason", "") or "").strip().lower()
            except Exception:
                decision = ""
                reason = ""
            if decision == "UNKNOWN" and (
                reason.startswith("http 504:")
                or "timed out" in reason
                or "connection timed out" in reason
                or reason.startswith("http 402:")
            ):
                # HTTP 402 (Payment Required) from GPT provider must not block
                # ML trading. Treated identically to timeout: return None so the
                # calling gate allows the trade to proceed.
                try:
                    print(f"[GPT CONTROL] HTTP 402 / timeout in advise_trade reason={reason[:120]}")
                except Exception:
                    pass
                return None
            return adv
        except Exception:
            try:
                return gpt_advisor.GPTAdvice(decision="UNKNOWN", reason="Advisor call failed")
            except Exception:
                return None

    def _gpt_entry_gate(self, *, proposal: Dict[str, Any], is_paper: bool) -> bool:
        """Return True to allow an entry, False to block.

        - MSTOCK_GPT_ENABLE=1 enables
        - MSTOCK_GPT_MODE=gate blocks unless TAKE
        - MSTOCK_GPT_MODE=advice logs but allows
        - MSTOCK_GPT_APPLY_PAPER=1 applies in paper too (default enabled)
        """

        if not self._gpt_should_apply_now(is_paper=bool(is_paper)):
            return True

        try:
            mode = str(getattr(self.cfg, "gpt_mode", "gate") or "gate").strip().lower()
        except Exception:
            mode = "gate"
        if mode not in {"gate", "advice"}:
            mode = "gate"

        preset = ""
        try:
            preset = str(getattr(self.cfg, "preset", "") or "").strip().lower()
        except Exception:
            preset = ""

        # Provide a small, stable set of preset-derived constraints so the advisor can
        # respect the user's risk profile after legs are already selected.
        constraints: Dict[str, Any] = {
            "preset": preset,
            "strategy_mode": str(getattr(self.cfg, "strategy_name", "") or ""),
            "timeframe": str(getattr(self.cfg, "timeframe", "") or ""),
        }
        for k in (
            "max_pyramid_levels",
            "cooldown_sec",
            "cooldown_after_stopout_sec",
            "max_hold_minutes",
            "max_hold_days",
            "risk_scale_enabled",
            "entry_max_same_direction_qty",
        ):
            try:
                constraints[k] = getattr(self.cfg, k)
            except Exception:
                continue
        try:
            mhd = int(getattr(self.cfg, "max_hold_days", 0) or 0)
        except Exception:
            mhd = 0
        constraints["holding_policy"] = (
            "intraday_only_same_day_exit" if mhd <= 0 else f"carry_allowed_up_to_{mhd}_day(s)"
        )

        preset_guidance = (
            "Use a balanced risk posture. Prefer clear reward-to-risk edges; if ambiguous, SKIP."
        )
        if preset == "conservative":
            preset_guidance = (
                "Use a conservative risk posture, but do not reject valid multi-leg structures solely because the preset is conservative. "
                "Respect the configured max hold and runtime constraints as authoritative."
            )
        elif preset == "aggressive":
            preset_guidance = (
                "Use an aggressive but disciplined posture. Prefer setups with stronger expected reward and profit potential when runtime constraints allow them. "
                "Do not SKIP solely because a trade is moderately risky if its reward-to-risk is favorable."
            )

        system = (
            "You are a trading risk manager for an automated options strategy. "
            "You MUST respond with ONLY valid JSON (no markdown). "
            "Decide TAKE or SKIP for the proposed entry. "
            "The `preset` represents the user's risk profile and influences how strict you are about risk. "
            "It MUST NOT be used to ban specific strategy types or multi-leg structures by itself. "
            "Use the provided config_constraints as the authoritative runtime settings (timeframe/cooldown/pyramiding/etc). "
            "IMPORTANT: max_hold_days=0 means intraday mode (force same-day exit), and does NOT mean 'do not open trades'. "
            f"{preset_guidance} "
            "If the proposal conflicts with config_constraints, respond SKIP. "
            "Output schema: {\"decision\":\"TAKE\"|\"SKIP\",\"reason\":string,\"confidence\":number}."
        )

        payload = dict(proposal or {})
        payload.setdefault("preset", preset)
        payload.setdefault("paper", bool(is_paper))
        try:
            payload.setdefault("filters", getattr(self, "_last_filters_snapshot", {}))
        except Exception:
            pass
        payload.setdefault("config_constraints", constraints)

        advice = self._gpt_control_advise(proposal=payload, system_prompt=system, max_tokens=300)
        if advice is None:
            try:
                self._gpt_gate_unknown = int(getattr(self, "_gpt_gate_unknown", 0)) + 1
            except Exception:
                pass
            try:
                setattr(self, "_last_gpt_gate_decision", "UNKNOWN")
            except Exception:
                pass
            return True

        decision = "UNKNOWN"
        reason = ""
        try:
            if isinstance(advice, dict):
                decision = str(advice.get("decision", "UNKNOWN") or "UNKNOWN").strip().upper()
                reason = str(advice.get("reason", "") or "").strip()
            else:
                decision = str(getattr(advice, "decision", "UNKNOWN") or "UNKNOWN").strip().upper()
                reason = str(getattr(advice, "reason", "") or "").strip()
        except Exception:
            decision = "UNKNOWN"
            reason = ""
        try:
            setattr(self, "_last_gpt_gate_decision", str(decision))
        except Exception:
            pass

        if mode == "advice":
            print(f"[GPT ADVICE] decision={decision} reason={reason[:160]}")
            try:
                # Advice mode does not gate; count as unknown for telemetry.
                if decision == "TAKE":
                    self._gpt_gate_take = int(getattr(self, "_gpt_gate_take", 0)) + 1
                elif decision == "SKIP":
                    self._gpt_gate_skip = int(getattr(self, "_gpt_gate_skip", 0)) + 1
                else:
                    self._gpt_gate_unknown = int(getattr(self, "_gpt_gate_unknown", 0)) + 1
            except Exception:
                pass
            return True

        # gate mode
        if decision == "TAKE":
            print(f"[GPT GATE] TAKE {reason[:160]}")
            try:
                self._gpt_gate_take = int(getattr(self, "_gpt_gate_take", 0)) + 1
            except Exception:
                pass
            return True

        if decision == "SKIP":
            # Guard against a common false-negative interpretation:
            # max_hold_days=0 => intraday-only, not a ban on new entries.
            try:
                mhd_now = int(getattr(self.cfg, "max_hold_days", 0) or 0)
            except Exception:
                mhd_now = 0
            reason_l = str(reason or "").strip().lower()
            if mhd_now <= 0 and "max_hold_days" in reason_l and any(
                t in reason_l for t in ("no positions can be held overnight", "cannot be held overnight", "held overnight")
            ):
                print(f"[GPT GATE] OVERRIDE TAKE (max_hold_days=0 intraday semantics) {reason[:160]}")
                return True

            try:
                proposal_name = str(
                    (proposal or {}).get("trade_name")
                    or (proposal or {}).get("name")
                    or (proposal or {}).get("recommended_strategy")
                    or ""
                ).strip().lower()
            except Exception:
                proposal_name = ""
            multi_leg_names = {
                "short_straddle",
                "short_strangle",
                "bull_call_spread",
                "bull_put_spread",
                "call_ratio_backspread",
                "put_ratio_backspread",
                "iron_condor",
                "iron_fly",
                "iron_butterfly",
                "long_straddle",
                "long_strangle",
                "delta_hedged_short_straddle",
                "delta_hedged_long_straddle",
            }
            if proposal_name in multi_leg_names and any(
                txt in reason_l
                for txt in (
                    "conservative preset",
                    "not suitable for a conservative preset",
                    "too aggressive for a conservative preset",
                    "not aligned with the conservative preset",
                )
            ):
                print(f"[GPT GATE] OVERRIDE TAKE (preset-only multileg veto) {reason[:160]}")
                return True
            print(f"[GPT GATE] SKIP {reason[:160]}")
            try:
                self._gpt_gate_skip = int(getattr(self, "_gpt_gate_skip", 0)) + 1
            except Exception:
                pass
            return False

        # UNKNOWN => block (safer default)
        print(f"[GPT GATE] UNKNOWN {reason[:160]}")
        try:
            self._gpt_gate_unknown = int(getattr(self, "_gpt_gate_unknown", 0)) + 1
        except Exception:
            pass
        return False

    def _should_gpt_override_entry_checks(self, proposal_dict: Optional[Dict[str, Any]] = None) -> bool:
        """Check if GPT override is enabled and gate returns TAKE for multi-leg entry.
        
        When gpt_override_checks=True and GPT gate explicitly returns TAKE, this
        allows multi-leg entries to bypass premium and liquidity checks.
        
        Args:
            proposal_dict: Optional proposal to pass to _gpt_entry_gate. If None,
                          a generic proposal is constructed.
        
        Returns:
            True if override should be applied; False otherwise.
        """
        try:
            if not bool(getattr(self.cfg, "gpt_override_checks", False)):
                return False
        except Exception:
            return False
        
        # Call GPT gate early to check for explicit TAKE
        try:
            is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
        except Exception:
            is_paper = True
        
        if proposal_dict is None:
            proposal_dict = {"position_type": "multi"}
        
        try:
            gpt_ok = self._gpt_entry_gate(proposal=proposal_dict, is_paper=bool(is_paper))
        except Exception:
            gpt_ok = True
        
        # Only override if advisor explicitly TOOK
        if bool(gpt_ok) and str(getattr(self, "_last_gpt_gate_decision", "") or "").strip().upper() == "TAKE":
            return True
        
        return False

    def _gpt_allowed_strategies_for_auto(self) -> List[str]:
        # Full set of supported entries the strategy engine can execute.
        # (auto itself is excluded because this is used to pick an effective strategy.)
        return [
            "directional",
            "short_straddle",
            "short_strangle",
            "bull_call_spread",
            "bull_put_spread",
            "call_ratio_backspread",
            "put_ratio_backspread",
            "long_straddle",
            "long_strangle",
            "iron_condor",
            "iron_fly",
            "iron_butterfly",
            "delta_hedged_long_straddle",
            "delta_hedged_short_straddle",
            # Directional explicit variants
            "long_call",
            "long_put",
            "short_call",
            "short_put",
        ]

    def _auto_select_effective_strategy(
        self,
        *,
        chain_available: bool,
        range_bound: bool,
        trend_strength: float,
        ts_limit: float,
        atr_val: float,
        atr_threshold: float,
        fast_ema: Optional[float],
        slow_ema: Optional[float],
    ) -> str:
        """Heuristic picker for strategy_name='auto'.

        The strategy engine supports many multi-leg structures, but auto mode
        must pick a *single* explicit branch each cycle.

        Rules of thumb:
        - If no option chain: directional
        - Range-bound: prefer neutral structures (iron fly/condor/straddle/strangle)
        - Trending: optionally use directional multi-leg (spreads/backspreads)
        """

        if not bool(chain_available):
            return "directional"

        thr = float(atr_threshold or 0.0)

        if bool(range_bound):
            # Preserve legacy behavior when the threshold is disabled.
            if thr <= 0.0:
                return "short_straddle"

            # Use a few ATR buckets so auto mode can place other supported
            # multi-leg structures (not only straddle/strangle).
            if float(atr_val) <= 0.90 * thr:
                return "iron_fly"
            if float(atr_val) <= 1.05 * thr:
                return "short_straddle"
            if float(atr_val) <= 1.35 * thr:
                return "iron_condor"

            # Higher ATR (even if trend is low) -> prefer long-vol structure.
            return "long_strangle"

        # If we are only out of the RSI range-band (but trend is still mild),
        # do not force a multi-leg structure.
        if float(trend_strength) <= float(ts_limit):
            return "directional"

        # Trending regime: allow directional multi-leg entries.
        if fast_ema is None or slow_ema is None:
            return "directional"

        if float(fast_ema) >= float(slow_ema):
            # Bullish: low/moderate ATR -> bull call spread, high ATR -> call backspread.
            if thr > 0.0 and float(atr_val) >= 1.25 * thr:
                return "call_ratio_backspread"
            return "bull_call_spread"

        # Bearish: only pick the convex long-vol structure when volatility is elevated.
        if thr > 0.0 and float(atr_val) >= 1.25 * thr:
            return "put_ratio_backspread"
        return "directional"

    def _gpt_auto_select_strategy(
        self,
        *,
        chain: List[dict],
        spot: float,
        atr_val: float,
        rsi_val: Optional[float],
        trend_strength: float,
        vwap_val: Optional[float],
        call_atm: Optional[dict],
        put_atm: Optional[dict],
        adx_val: Optional[float] = None,
        vol_sma: Optional[float] = None,
        current_vol: Optional[float] = None,
    ) -> Optional[str]:
        """Optional GPT override for auto-mode strategy selection.

        Returns a normalized strategy name or None (keep heuristic selection).
        Never blocks trading by itself; entry gating handles approvals.
        """

        self._auto_gpt_strategy_parameters = {}
        self._auto_gpt_directional_steps = None

        try:
            # gpt_market_commentary_enabled (default False) gates GPT market commentary.
            if not bool(getattr(self.cfg, "gpt_market_commentary_enabled", False)):
                return None
            if not bool(getattr(self.cfg, "gpt_enabled", getattr(self.cfg, "gpt_enable", False))):
                return None
            if not bool(getattr(self.cfg, "use_gpt_market_analysis", False)):
                return None
            if not bool(getattr(self.cfg, "gpt_auto_select", True)):
                return None
        except Exception:
            return None

        try:
            api_key = (os.getenv("MSTOCK_GPT_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        except Exception:
            api_key = ""
        if not api_key:
            return None

        try:
            import gpt_advisor as _ga  # type: ignore

            gpt_advisor = _ga
        except Exception:
            try:
                from src import gpt_advisor as _ga  # type: ignore

                gpt_advisor = _ga
            except Exception:
                return None
        try:
            if bool(getattr(gpt_advisor, "is_gpt_disabled_for_session", lambda: False)()):
                return None
        except Exception:
            pass

        allowed = self._gpt_allowed_strategies_for_auto()
        preset = ""
        try:
            preset = str(getattr(self.cfg, "preset", "") or "").strip().lower()
        except Exception:
            preset = ""

        snap: Dict[str, Any] = {
            "symbol": str(getattr(self.cfg, "symbol", "") or ""),
            "timeframe": str(getattr(self.cfg, "timeframe", "") or ""),
            "spot": float(spot),
            "atr": float(atr_val),
            "rsi": float(rsi_val) if rsi_val is not None else None,
            "trend_strength": float(trend_strength),
            "vwap": float(vwap_val) if vwap_val is not None else None,
            "allowed_strategies": list(allowed),
            "preset": preset,
            "preset_note": "Preset is risk-profile guidance. Do not exclude multi-leg strategies solely because of the preset.",
            "chain_available": bool(chain),
            "atm": {
                "call": {
                    "symbol": str((call_atm or {}).get("symbol") or ""),
                    "strike": (call_atm or {}).get("strike"),
                    "ltp": (call_atm or {}).get("ltp"),
                    "theta": (self._estimate_option_theta_from_chain_row(call_atm, float(spot)) if isinstance(call_atm, dict) else None),
                    "iv": (call_atm or {}).get("iv"),
                },
                "put": {
                    "symbol": str((put_atm or {}).get("symbol") or ""),
                    "strike": (put_atm or {}).get("strike"),
                    "ltp": (put_atm or {}).get("ltp"),
                    "theta": (self._estimate_option_theta_from_chain_row(put_atm, float(spot)) if isinstance(put_atm, dict) else None),
                    "iv": (put_atm or {}).get("iv"),
                },
            },
            "chain_preview": [],
            "open_positions": self._gpt_open_positions_mtm_snapshot(max_trades=6),
            "filters": {
                "enable_adx": bool(getattr(self.cfg, "enable_adx_filter", False)),
                "adx": (float(adx_val) if adx_val is not None else None),
                "enable_volume": bool(getattr(self.cfg, "enable_volume_filter", False)),
                "vol_sma": (float(vol_sma) if vol_sma is not None else None),
                "current_vol": (float(current_vol) if current_vol is not None else None),
                "enable_supertrend": bool(getattr(self.cfg, "enable_supertrend_filter", False)),
            },
        }

        try:
            preview_rows: List[dict] = []
            for row in (chain or []):
                if not isinstance(row, dict):
                    continue
                try:
                    k = float(row.get("strike"))
                except Exception:
                    continue
                if k <= 0:
                    continue
                preview_rows.append(row)
            preview_rows = sorted(preview_rows, key=lambda r: abs(float(r.get("strike")) - float(spot)))[:16]
            chain_preview_dicts = []
            for r in preview_rows:
                theta_val = self._estimate_option_theta_from_chain_row(r, float(spot))
                chain_preview_dicts.append({
                    "symbol": str(r.get("symbol") or ""),
                    "option_type": str(r.get("option_type") or ""),
                    "strike": r.get("strike"),
                    "ltp": r.get("ltp"),
                    "theta": theta_val,
                    "iv": r.get("iv"),
                    "expiry": str(self._row_expiry_date(r) or ""),
                })
            
            snap["chain_preview"] = chain_preview_dicts
        except Exception:
            pass

        try:
            base_url = (os.getenv("MSTOCK_GPT_API_BASE_URL") or "").strip() or None
        except Exception:
            base_url = None
        try:
            model = (os.getenv("MSTOCK_GPT_MODEL") or "").strip() or "gpt-4o-mini"
        except Exception:
            model = "gpt-4o-mini"
        try:
            timeout_sec = float(getattr(self.cfg, "gpt_timeout_sec", 45.0) or 45.0)
        except Exception:
            timeout_sec = 45.0

        # Telemetry: we are about to call the advisor.
        try:
            self._gpt_attempts = int(getattr(self, "_gpt_attempts", 0)) + 1
        except Exception:
            pass

        try:
            analysis = gpt_advisor.analyze_market(
                snapshot=snap,
                model=str(model),
                api_key=str(api_key),
                base_url=base_url,
                timeout_sec=float(timeout_sec),
            )
        except Exception as exc:
            try:
                err = str(exc or "")
            except Exception:
                err = "unknown"
            try:
                self._last_gpt_error = str(err)
            except Exception:
                pass
            try:
                print(f"[AUTO][GPT] analyze_market exception: {err}")
            except Exception:
                pass
            return None

        # Clear last-gpt-error when we have a valid analysis object.
        try:
            self._last_gpt_error = ""
        except Exception:
            pass

        # Optional: GPT can request a preset switch based on market conditions.
        try:
            p_req = str(getattr(analysis, "preset_request", "") or "").strip().lower()
        except Exception:
            p_req = ""
        if p_req in {"aggressive", "conservative"}:
            try:
                p_reason = str(getattr(analysis, "preset_reason", "") or getattr(analysis, "reason", "") or "").strip()
            except Exception:
                p_reason = ""
            try:
                p_conf = float(getattr(analysis, "confidence", 0.0) or 0.0)
            except Exception:
                p_conf = 0.0
            cur_preset = ""
            try:
                cur_preset = str(getattr(self.cfg, "preset", "") or "").strip().lower()
            except Exception:
                cur_preset = ""
            self._last_gpt_preset_request = {
                "preset_request": str(p_req),
                "preset_reason": str(p_reason),
                "confidence": float(p_conf),
                "current_preset": str(cur_preset),
                "ts": float(time.time()),
            }
            try:
                key_now = f"{p_req}:{p_reason[:80]}"
                if getattr(self, "_last_gpt_preset_req_key", None) != key_now:
                    print(f"[GPT][PRESET] request={p_req} conf={p_conf:.2f} reason={p_reason[:160]}")
                    setattr(self, "_last_gpt_preset_req_key", key_now)
            except Exception:
                pass

        rec = ""
        try:
            # Record analysis for diagnostics (support both object and dict results)
            try:
                if isinstance(analysis, dict):
                    self._last_gpt_analysis = {
                        "raw": analysis,
                        "reason": str(analysis.get("reason") or analysis.get("error") or "").strip(),
                        "confidence": float(analysis.get("confidence") or 0.0),
                    }
                else:
                    self._last_gpt_analysis = {
                        "raw": analysis,
                        "reason": str(getattr(analysis, "reason", "") or "").strip(),
                        "confidence": float(getattr(analysis, "confidence", 0.0) or 0.0),
                    }
            except Exception:
                self._last_gpt_analysis = {"raw": analysis}

            # Robust extraction of recommended strategy from dict or object
            if isinstance(analysis, dict):
                rec = str(analysis.get("recommended_strategy") or analysis.get("selected") or analysis.get("strategy") or "").strip()
                bias = str(analysis.get("ce_pe_bias") or analysis.get("bias") or "").strip().upper()
            else:
                rec = str(getattr(analysis, "recommended_strategy", "") or getattr(analysis, "selected", "") or "").strip()
                bias = str(getattr(analysis, "ce_pe_bias", "") or "").strip().upper()

            if bias in {"CE", "PE", "NEUTRAL"}:
                self._last_gpt_bias = bias
            else:
                self._last_gpt_bias = None

            if rec:
                try:
                    self._gpt_recommendations = int(getattr(self, "_gpt_recommendations", 0)) + 1
                except Exception:
                    pass
        except Exception:
            rec = ""

        try:
            self._auto_gpt_strategy_parameters = {
                str(k).strip().lower().replace("-", "_").replace(" ", "_"): {
                    str(pk): float(pv)
                    for pk, pv in (v or {}).items()
                }
                for k, v in dict(getattr(analysis, "strategy_parameters", {}) or {}).items()
                if isinstance(v, dict)
            }
        except Exception:
            self._auto_gpt_strategy_parameters = {}

        try:
            raw_steps = getattr(analysis, "directional_strike_offset_steps", None)
            if raw_steps is None:
                self._auto_gpt_directional_steps = None
            else:
                self._auto_gpt_directional_steps = max(0, int(float(raw_steps)))
        except Exception:
            self._auto_gpt_directional_steps = None

        if not rec:
            # Tolerant fallback: scan raw analysis or reason text for a known strategy keyword
            try:
                raw = getattr(self, "_last_gpt_analysis", {}) or {}
                raw_text = ""
                if isinstance(raw, dict):
                    # join string values
                    parts = []
                    for v in raw.values():
                        try:
                            if isinstance(v, str):
                                parts.append(v)
                        except Exception:
                            continue
                    raw_text = " ".join(parts)
                else:
                    try:
                        raw_text = str(raw)
                    except Exception:
                        raw_text = ""
                raw_text = (raw_text or "") + " " + (str(getattr(analysis, "reason", "") or "") if not isinstance(analysis, dict) else str((analysis or {}).get("reason") or ""))
                raw_text = raw_text.lower()
                # Look for allowed strategy keywords in the text.
                for cand in allowed:
                    if cand and str(cand).lower() in raw_text:
                        rec = str(cand)
                        try:
                            self._gpt_recommendations = int(getattr(self, "_gpt_recommendations", 0)) + 1
                        except Exception:
                            pass
                        break
            except Exception:
                rec = ""
        if not rec:
            return None

        allowed_set = set(allowed)
        rec_raw = str(rec).strip().lower().replace("-", "_").replace(" ", "_")
        banned_aliases = {
            "short_premium",
            "long_premium",
            "short_volatility",
            "long_volatility",
            "short_volatility_strategy",
            "long_volatility_strategy",
        }
        if rec_raw in banned_aliases:
            # Require explicit, executable strategy names so GPT doesn't collapse
            # auto mode into umbrella aliases.
            return None

        rec_norm = self._normalize_strategy_name(str(rec_raw))
        if rec_norm == "auto":
            return None

        # If GPT recommends a multi-leg structure but we have no chain, ignore.
        if not bool(chain) and rec_norm not in {"directional", "long_call", "long_put", "short_call", "short_put"}:
            return None

        if rec_norm not in allowed_set:
            # Allow directional variants which map to directional.
            if rec_norm in {"directional", "long_call", "long_put", "short_call", "short_put"}:
                return rec_norm
            return None

        # Log once per change for transparency.
        try:
            if getattr(self, "_last_gpt_auto_strat", None) != rec_norm:
                print(f"[AUTO][GPT] recommended_strategy={rec_norm}")
                setattr(self, "_last_gpt_auto_strat", rec_norm)
        except Exception:
            pass

        return str(rec_norm)

    def _open_directional_from_option(
        self,
        opt: dict,
        *,
        name: str,
        spot: float,
        atr_val: float,
        append_state: bool = True,
    ) -> Optional[Dict[str, object]]:
        if not self._can_open_trade_type(position_type="directional", name=str(name)):
            return None

        contract = self._option_contract_from_row(opt)
        if contract is None or not contract.tradingsymbol:
            return None
        if not str(contract.instrument_token or "").isdigit():
            self._entry_block_for_missing_contract_token(row=opt, context=str(name))
            return None
        symbol = str(contract.tradingsymbol)

        entry_side = "SELL" if str(name or "").strip().lower().startswith("short_") else "BUY"
        qty = self._entry_qty(atr_val)
        if qty <= 0:
            return None

        # Pyramiding: enforce max level BEFORE placing any add-on order.
        existing = None
        max_pyr = 0
        if append_state:
            try:
                max_pyr = self._dynamic_pyramid_max_level()
            except Exception:
                max_pyr = 0
            if max_pyr > 0:
                for t in self.state.open_directional:
                    if t.get("name") == name:
                        t_sym = str(t.get("symbol") or "").strip().upper()
                        new_sym = str(opt.get("symbol") or "").strip().upper()
                        if t_sym == new_sym:
                            existing = t
                            break
                if existing is not None:
                    try:
                        lev = int(existing.get("pyramid_level", 0) or 0)
                    except Exception:
                        lev = 0
                    if lev >= max_pyr:
                        print(f"[PYRAMID] Max pyramiding reached for {name} (level={lev}, max={max_pyr}); skipping add")
                        return existing

        exchange = str(contract.exchange or "").strip() or None
        token = str(contract.instrument_token or "").strip()
        strike = contract.strike
        opt_type = contract.option_type
        expiry = contract.expiry
        print(f"[CONTRACT] selected tradingsymbol={symbol} token={token} exchange={exchange}")

        # ----------------------------------------------------------------
        # Paper execution: use bid/ask for realistic price discovery.
        # - BUY entry  (going long): pay the ask  (seller's price)
        # - SELL entry (going short): receive bid (buyer's bid)
        # Block if bid/ask is unavailable unless LTP fallback is explicitly
        # enabled via paper_allow_ltp_fallback.
        # ----------------------------------------------------------------
        use_bid_ask = bool(getattr(self.cfg, "paper_use_bid_ask_execution", True))
        allow_ltp_fallback = bool(getattr(self.cfg, "paper_allow_ltp_fallback", False))
        ltp_fallback_penalty = float(getattr(self.cfg, "paper_ltp_fallback_spread_pct", 0.05) or 0.05)

        entry_price = None
        price_source = "ltp"
        if use_bid_ask and not self.cfg.enable_live_trading:
            # First attempt: fetch FULL quote directly with token to get depth/bid-ask.
            # This avoids symbol->token resolution failures that LTP-mode has.
            quote_data = self.client.fetch_option_quote_for_paper(exchange, token, symbol)
            bid_raw = quote_data.get("bid_price")
            ask_raw = quote_data.get("ask_price")

            # Fallback to get_bid_ask if FULL quote returned no depth
            if bid_raw is None and ask_raw is None:
                bid_raw, ask_raw, _ = self.client.get_bid_ask(symbol, exchange_hint=exchange)

            if entry_side == "BUY":
                # Long entry: pay ask (you are the buyer paying seller's ask)
                if ask_raw is None:
                    if allow_ltp_fallback:
                        print(f"[PAPER][LTP FALLBACK] {symbol}: ask unavailable, using LTP -{ltp_fallback_penalty*100:.1f}%")
                        raw_ltp = self._try_get_ltp_for_leg({"symbol": symbol, "exchange": exchange, "token": token, "option_type": opt_type})
                        entry_price = float(raw_ltp) * (1.0 - ltp_fallback_penalty) if raw_ltp is not None else None
                        price_source = "ltp_fallback"
                    else:
                        print(f"[PAPER][BLOCKED] {name}: ask price unavailable for {symbol} — bid/ask required for paper realism")
                        return None
                else:
                    entry_price = float(ask_raw)
                    price_source = "ask"
            else:  # SELL entry (short)
                # Short entry: receive bid (you are the seller receiving buyer's bid)
                if bid_raw is None:
                    if allow_ltp_fallback:
                        print(f"[PAPER][LTP FALLBACK] {symbol}: bid unavailable, using LTP +{ltp_fallback_penalty*100:.1f}%")
                        raw_ltp = self._try_get_ltp_for_leg({"symbol": symbol, "exchange": exchange, "token": token, "option_type": opt_type})
                        entry_price = float(raw_ltp) * (1.0 + ltp_fallback_penalty) if raw_ltp is not None else None
                        price_source = "ltp_fallback"
                    else:
                        print(f"[PAPER][BLOCKED] {name}: bid price unavailable for {symbol} — bid/ask required for paper realism")
                        return None
                else:
                    entry_price = float(bid_raw)
                    price_source = "bid"
        else:
            entry_price = self._try_get_ltp_for_leg({"symbol": symbol, "exchange": exchange, "token": token, "option_type": opt_type})
            price_source = "ltp"
        projected_legs = [
            {
                "symbol": symbol,
                "token": token,
                "exchange": exchange,
                "side": entry_side,
                "quantity": int(qty),
                "entry_price": entry_price,
                "strike": strike,
                "option_type": opt_type,
                "expiry": expiry,
                "iv": opt.get("iv"),
            }
        ]
        projected_legs = [self._annotate_leg_brackets(lg, risk_overrides=self._trade_risk_meta_overrides()) for lg in projected_legs]
        if not self._portfolio_caps_allow_legs(float(spot), projected_legs, context=f"directional {name}"):
            return None
        # Optional: allow GPT gate TAKE to override price / liquidity checks
        gpt_forced_take = False
        try:
            if bool(getattr(self.cfg, "gpt_override_checks", False)):
                try:
                    is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
                except Exception:
                    is_paper = True
                try:
                    gpt_ok = self._gpt_entry_gate(
                        proposal={
                            "position_type": "directional",
                            "trade_name": str(name),
                            "symbol": str(symbol),
                            "exchange": str(exchange or ""),
                            "side": str(entry_side),
                            "quantity": int(qty),
                            "spot": float(spot),
                            "atr": float(atr_val),
                            "entry_price": float(entry_price) if entry_price is not None else None,
                            "order_type": str("MARKET"),
                            "limit_price": None,
                            "pyramiding": bool(existing is not None),
                            "pyramid_level": int(existing.get("pyramid_level", 0)) if existing is not None else 0,
                            "strategy_mode": str(getattr(self.cfg, "strategy_name", "") or ""),
                            "timeframe": str(getattr(self.cfg, "timeframe", "") or ""),
                        },
                        is_paper=bool(is_paper),
                    )
                except Exception:
                    gpt_ok = True
                # If advisor explicitly TAKES, bypass later safety checks.
                if bool(gpt_ok) and str(getattr(self, "_last_gpt_gate_decision", "") or "").strip().upper() == "TAKE":
                    gpt_forced_take = True

        except Exception:
            gpt_forced_take = False

        # Standard price bounds + liquidity checks (skip when GPT forced TAKE)
        if not gpt_forced_take:
            ok, reason = self._entry_price_bounds_ok(entry_price)
            if not ok:
                print(f"[ENTRY BLOCKED] {reason}")
                return None
            ok, reason = self._check_entry_liquidity([{"symbol": symbol, "exchange": exchange, "token": token}])
            if not ok:
                print(f"[ENTRY BLOCKED] {reason}")
                return None

        # Execution Enhancements: midpoint-pegged limit routing
        order_type = "LIMIT"
        limit_price = self._round_limit_price(entry_price)

        # Optional GPT entry gate (applies before placing any orders).
        try:
            is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
        except Exception:
            is_paper = True
        try:
            gpt_ok = self._gpt_entry_gate(
                proposal={
                    "position_type": "directional",
                    "trade_name": str(name),
                    "symbol": str(symbol),
                    "exchange": str(exchange or ""),
                    "side": str(entry_side),
                    "quantity": int(qty),
                    "spot": float(spot),
                    "atr": float(atr_val),
                    "entry_price": float(entry_price) if entry_price is not None else None,
                    "order_type": str(order_type),
                    "limit_price": float(limit_price) if limit_price is not None else None,
                    "pyramiding": bool(existing is not None),
                    "pyramid_level": int(existing.get("pyramid_level", 0)) if existing is not None else 0,
                    "strategy_mode": str(getattr(self.cfg, "strategy_name", "") or ""),
                    "timeframe": str(getattr(self.cfg, "timeframe", "") or ""),
                },
                is_paper=bool(is_paper),
            )
        except Exception:
            gpt_ok = True
        if not bool(gpt_ok):
            # Block new entry or pyramid add-on.
            return existing

        friction_preview: Dict[str, Any] = {}
        live_meta: Dict[str, Any] = {}
        if not self.cfg.enable_live_trading:
            friction_preview = evaluate_live_execution_friction(
                {
                    "client": self.client,
                    "symbol": symbol,
                    "exchange": exchange,
                    "cost_model": CostModel(),
                    "enforce_cost_boundary": False,
                }
            )
            p_price = f" @ {limit_price:.2f}" if limit_price else ""
            print(f"[PAPER] {entry_side} {symbol} x{qty} {order_type}{p_price}")
        else:
            route_result = self._route_midpoint_pegged_limit_order(
                symbol=symbol,
                side=entry_side,
                quantity=qty,
                exchange=exchange,
                symbol_token=str(token or "") or None,
                meta=live_meta,
            )
            if not bool(route_result.get("ok")):
                print(f"[ENTRY BLOCKED] {route_result.get('status')}: {route_result.get('reason')}")
                return existing
            limit_price = float(route_result.get("fill_price") or limit_price or entry_price or 0.0)

        # IMPORTANT: Handle pyramiding BEFORE allocating a new trade_id.
        # The pyramiding path increases qty on an existing position; it should not
        # consume a new sequence number, and it must emit an UPDATE so the UI log
        # reflects the added quantity.
        if append_state and existing:
            lev = int(existing.get("pyramid_level", 0))
            existing["pyramid_level"] = lev + 1
            existing["quantity"] = int(existing.get("quantity", 0)) + int(qty)
            legs_existing = existing.get("legs")
            if isinstance(legs_existing, list) and legs_existing and isinstance(legs_existing[0], dict):
                legs_existing[0]["quantity"] = int(existing.get("quantity", 0))

            # Count this add as a trade for daily limits.
            self.state.trades_today += 1
            self.state.last_entry_ts = time.time()
            self._note_opened_trade_type(position_type="directional", name=str(name))

            if self._has_event_sink():
                try:
                    trade_id_existing = str(existing.get("trade_id") or "(unknown)")
                    name_existing = str(existing.get("name") or name)
                    legs_evt = [dict(l) for l in legs_existing if isinstance(l, dict)] if isinstance(legs_existing, list) else []
                    self._emit(
                        TradeLogEvent(
                            ts=time.time(),
                            event="UPDATE",
                            trade_id=trade_id_existing,
                            position_type="directional",
                            name=name_existing,
                            legs=legs_evt,
                            mtm=self._compute_legs_mtm(legs_evt),
                        )
                    )
                except Exception:
                    pass

            print(f"[PYRAMID] Incremented {name} level to {lev + 1}")
            return existing

        trade_id = self._new_trade_id("D")
        legs_for_trade: List[Dict[str, object]] = [dict(lg) for lg in projected_legs]
        prediction_id = str(getattr(self, "_last_prediction_id", "") or "")
        meta: Dict[str, Any] = {
            "reason": "directional_entry",
            "prediction_id": prediction_id,
            "ml_probability": float(getattr(self, "_last_ml_pred", 0.0) or 0.0),
            "ml_threshold": float(getattr(self, "_last_ml_threshold", 0.0) or 0.0),
            "ml_bet_multiplier": float(getattr(self, "_last_ml_bet_multiplier", 0.0) or 0.0),
            "baseline_vol_lots": int(getattr(self, "_last_baseline_vol_lots", 0) or 0),
            "final_allocated_lots": int(getattr(self, "_last_final_allocated_lots", qty) or qty),
            "order_routing_style": "midpoint_pegged_limit",
        }
        meta.update(self._trade_risk_meta_overrides())
        try:
            if not self.cfg.enable_live_trading:
                meta["live_bid_ask_spread_pct"] = float(friction_preview.get("relative_spread_pct") or 0.0)
                meta["realized_slippage_pct"] = float(getattr(self.cfg, "paper_slippage_pct", 0.001) or 0.001)
                meta["execution_friction_status"] = str(friction_preview.get("status") or "")
                meta["execution_price_source"] = str(price_source)
            else:
                meta.update(dict(live_meta or {}))
                meta["execution_price_source"] = "live"
        except Exception:
            pass
        if prediction_id and entry_price:
            try:
                meta["triple_barrier"] = self._build_triple_barrier_metadata(
                    fill_price=float(limit_price or entry_price),
                    prediction_id=prediction_id,
                    bar_timestamp=dt_datetime.now(IST) if IST else dt_datetime.now(),
                )
            except Exception:
                pass

        tr: Dict[str, object] = {
            "trade_id": trade_id,
            "name": name,
            "symbol": symbol,
            "token": token,
            "exchange": exchange,
            "side": entry_side,
            "quantity": int(qty),
            "entry_spot": float(spot),
            "entry_price": float(limit_price or entry_price) if (limit_price or entry_price) is not None else entry_price,
            "atr": float(atr_val),
            "strike": strike,
            "option_type": opt_type,
            "expiry": expiry,
            "legs": legs_for_trade,
            "meta": meta,
            "opened_ts": time.time(),
            "entry_time": dt_datetime.now(IST) if IST else dt_datetime.now(),
            "pyramid_level": 0,
        }
        if append_state:
            if existing is None:
                try:
                    open_count_now = len(self.state.open_directional) + len(self.state.open_multi)
                    max_open_positions = int(getattr(self.cfg, "max_open_positions", 0) or 0)
                except Exception:
                    open_count_now = 0
                    max_open_positions = 0
                if max_open_positions > 0 and open_count_now >= max_open_positions:
                    print(
                        f"[RISK] Max open positions reached ({open_count_now}/{max_open_positions}); "
                        f"blocking new directional entry {name}"
                    )
                    return None
            self.state.open_directional.append(tr)
        self.state.trades_today += 1
        self.state.last_entry_ts = time.time()

        self._note_opened_trade_type(position_type="directional", name=str(name))

        if self._has_event_sink():
            legs = [self._annotate_leg_brackets({
                "symbol": symbol,
                "token": token,
                "exchange": exchange,
                "side": entry_side,
                "quantity": int(qty),
                "entry_price": entry_price,
                "strike": strike,
                "option_type": opt_type,
                "expiry": expiry,
            }, risk_overrides=self._trade_risk_meta_overrides())]
            self._emit(
                TradeLogEvent(
                    ts=time.time(),
                    event="OPEN",
                    trade_id=trade_id,
                    position_type="directional",
                    name=name,
                    legs=legs,
                    mtm=self._compute_legs_mtm(legs),
                    timestamp=dt_datetime.now(IST).isoformat() if IST else dt_datetime.now().isoformat(),
                    symbol=self.cfg.underlying,
                    strike=float(atm_strike),
                    option_type=option_type,
                    side="BUY",
                    quantity=int(size),
                    bid=float(bid_raw) if bid_raw is not None else None,
                    ask=float(ask_raw) if ask_raw is not None else None,
                    ltp=float(price),
                    execution_price=float(paper_entry_price),
                    execution_price_source="ask",
                    entry_price=float(paper_entry_price),
                    spread_cost=float(paper_spread_cost),
                    slippage_cost=float(_entry_slippage_cost),
                    brokerage_cost=float(_entry_brokerage_cost),
                    realized_slippage_pct=float(_entry_slippage_pct),
                    spread_pct_at_entry=float(_spread_pct_entry),
                    filter_premium_ok=True,
                    filter_bid_ask_ok=True,
                    paper_mode=True,
                )
            )

        return tr

    def _is_atr_decreasing(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        *,
        period: int,
        steps: int = 1,
    ) -> bool:
        """Return True when ATR has been decreasing monotonically for the last `steps` samples.

        This is used as a simple, data-available proxy for "volatility decreasing in order".
        """

        try:
            p = int(period)
        except Exception:
            p = 14
        if p <= 0:
            return False

        try:
            s = int(steps)
        except Exception:
            s = 3
        if s < 2:
            s = 2

        if not (len(highs) == len(lows) == len(closes)):
            return False
        if len(closes) < p + 1 + (s - 1):
            return False

        vals: List[float] = []
        for end in range(len(closes) - (s - 1), len(closes) + 1):
            v = atr(highs[:end], lows[:end], closes[:end], period=p)
            if v is None:
                return False
            try:
                vals.append(float(v))
            except Exception:
                return False

        eps = 1e-12
        return all((a - b) > eps for a, b in zip(vals, vals[1:]))

    def _is_atr_increasing(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        *,
        period: int,
        steps: int = 3,
    ) -> bool:
        """Return True when ATR has been increasing monotonically for the last `steps` samples."""

        try:
            p = int(period)
        except Exception:
            p = 14
        if p <= 0:
            return False

        try:
            s = int(steps)
        except Exception:
            s = 3
        if s < 2:
            s = 2

        if not (len(highs) == len(lows) == len(closes)):
            return False
        if len(closes) < p + 1 + (s - 1):
            return False

        vals: List[float] = []
        for end in range(len(closes) - (s - 1), len(closes) + 1):
            v = atr(highs[:end], lows[:end], closes[:end], period=p)
            if v is None:
                return False
            try:
                vals.append(float(v))
            except Exception:
                return False

        eps = 1e-12
        return all((b - a) > eps for a, b in zip(vals, vals[1:]))

    def _new_trade_id(self, prefix: str) -> str:
        self._trade_seq += 1
        return f"{prefix}{self._trade_seq}"  # e.g. M1, D2

    def _try_get_ltp(self, symbol: str, exchange: Optional[str] = None) -> Optional[float]:
        sym = str(symbol or "").strip()
        if not sym:
            return None

        candidates: List[str] = []
        if ":" in sym:
            candidates.append(sym)
        exch = str(exchange or "").strip()
        if exch and ":" not in sym:
            candidates.append(f"{exch}:{sym}")
        candidates.append(sym)

        for s in candidates:
            try:
                v = float(self.client.get_ltp(s))
                # Track freshness against the logical symbol key (not the candidate string).
                try:
                    self._note_ltp_seen(self._ltp_key(sym), v)
                except Exception:
                    pass
            except Exception:
                continue
            return v
        return None

    def _try_get_ltp_for_leg(self, leg: dict) -> Optional[float]:
        """Best-effort LTP lookup for an option leg.

        Prefer the contract token (with exchange) so we don't
        accidentally use the index/underlying LTP.
        """

        if not isinstance(leg, dict):
            return None

        token = leg.get("token")
        exchange_raw = str(leg.get("exchange") or "").strip().upper()
        symbol_raw = str(leg.get("symbol") or "")

        # Normalize "EXCH:TRADINGSYMBOL" into raw trading symbol.
        symbol = symbol_raw
        if ":" in symbol:
            try:
                maybe_exch, maybe_sym = symbol.split(":", 1)
                if maybe_exch.strip().upper() in {"NSE", "BSE", "NFO", "NSEFO", "NSECM", "BSECM"} and maybe_sym.strip():
                    symbol = maybe_sym.strip()
                    if not exchange_raw:
                        exchange_raw = maybe_exch.strip().upper()
            except Exception:
                symbol = symbol_raw

        # Many call sites store option legs without an explicit exchange.
        # For options, default to NFO so token-only quotes don't accidentally
        # resolve on NSE and return None/stale results.
        if not exchange_raw:
            try:
                opt_type = str(leg.get("option_type") or "").strip().upper()
            except Exception:
                opt_type = ""
            sym_u = str(symbol or "").strip().upper()
            if opt_type in {"CE", "PE", "CALL", "PUT", "C", "P"} or sym_u.endswith("CE") or sym_u.endswith("PE"):
                exchange_raw = "NFO"

        # Normalize common aliases.
        if exchange_raw in {"NSEFO", "NFO"}:
            exchange_raw = "NFO"
        exchange = exchange_raw.strip() or None

        # If we have a numeric token, resolve LTP using it directly.
        try:
            tok_str = str(token).strip()
        except Exception:
            tok_str = ""

        is_option_contract = False
        try:
            opt_type = str(leg.get("option_type") or "").strip().upper()
            is_option_contract = opt_type in {"CE", "PE", "CALL", "PUT", "C", "P"} or str(symbol).strip().upper().endswith(("CE", "PE"))
        except Exception:
            is_option_contract = False
        # Best-effort token resolution is fine for non-option symbols. For
        # option entries we require a tokenized contract before quoting.
        if (not tok_str or not tok_str.isdigit()) and symbol and (not is_option_contract):
            try:
                resolved_exch, resolved_tok = self.client.resolve_exchange_token(symbol, exchange_hint=exchange)
                if resolved_tok is not None and str(resolved_tok).strip():
                    tok_str = str(resolved_tok).strip()
                    try:
                        leg["token"] = tok_str
                    except Exception:
                        pass
                if resolved_exch is not None and str(resolved_exch).strip() and not exchange:
                    exchange = str(resolved_exch).strip().upper() or exchange
                    try:
                        leg["exchange"] = exchange
                    except Exception:
                        pass
                if exchange in {"NSEFO", "NFO"}:
                    exchange = "NFO"
            except Exception:
                pass

        candidates: List[str] = []
        if tok_str and tok_str.isdigit():
            # IMPORTANT: never fall back to a bare token when we have an exchange.
            # Token namespaces can collide across exchanges; a bare token may return
            # a valid-but-wrong instrument's LTP.
            if exchange:
                candidates.append(f"{exchange}:{tok_str}")
            else:
                candidates.append(tok_str)

        if symbol:
            # Fallback to symbol-based resolution.
            if exchange and ":" not in symbol:
                candidates.append(f"{exchange}:{symbol}")
            candidates.append(symbol)

        for s in candidates:
            try:
                return float(self.client.get_ltp(s))
            except Exception:
                continue
        return None

    def _option_contract_from_row(self, row: Dict[str, Any]) -> Optional[OptionContract]:
        if not isinstance(row, dict):
            return None
        try:
            symbol = str(row.get("symbol") or "").strip()
            exchange = str(row.get("exchange") or "NFO").strip().upper() or "NFO"
            if exchange in {"NSEFO", "NFO"}:
                exchange = "NFO"
            token = str(row.get("token") or "").strip()
            if (not token or not token.isdigit()) and symbol and hasattr(self.client, "resolve_exchange_token_symbol"):
                try:
                    res_exch, res_tok, res_sym = self.client.resolve_exchange_token_symbol(symbol, exchange_hint=exchange)
                    if res_sym:
                        symbol = str(res_sym).strip()
                    if res_tok:
                        token = str(res_tok).strip()
                    if res_exch:
                        exchange = str(res_exch).strip().upper() or exchange
                except Exception:
                    pass
            strike = float(row.get("strike"))
            option_type = str(row.get("option_type") or "").strip().upper()
            if option_type in {"CALL", "C"}:
                option_type = "CE"
            elif option_type in {"PUT", "P"}:
                option_type = "PE"
            return OptionContract(
                exchange=exchange,
                tradingsymbol=symbol,
                instrument_token=token,
                expiry=row.get("expiry"),
                strike=strike,
                option_type=option_type,
            )
        except Exception:
            return None

    def _entry_block_for_missing_contract_token(self, *, row: Dict[str, Any], context: str) -> None:
        symbol = str(row.get("symbol") or "").strip()
        print(f"[ENTRY BLOCKED] CONTRACT_TOKEN_MISSING context={context} symbol={symbol}")

    def _normalize_close_reason(self, raw_reason: str, realized_pnl: Optional[float], exit_is_stop: bool) -> str:
        reason = str(raw_reason or "").strip().lower()
        try:
            pnl = float(realized_pnl) if realized_pnl is not None else None
        except Exception:
            pnl = None
        if reason in {"trail_stop", "trailing_stop", "gpt_leg_stop"}:
            return "TRAILING_STOP"
        if reason.startswith("pivot_target") or reason in {"target", "spot_target", "mtm_target", "profit_target", "partial_target"}:
            return "TARGET_PROFIT"
        if reason in {"time_exit", "max_hold", "max_hold_days"}:
            return "TIME_EXIT"
        if reason in {"manual_exit"}:
            return "MANUAL_EXIT"
        if reason in {"stop", "stop_loss", "mtm_stop", "stop_loss_hit"}:
            if pnl is not None and pnl > 0:
                return "TRAILING_STOP" if exit_is_stop else "SIGNAL_EXIT"
            return "STOP_LOSS"
        if exit_is_stop:
            return "STOP_LOSS"
        return "SIGNAL_EXIT"

    def _log_directional_close_audit(
        self,
        *,
        symbol: str,
        side: str,
        qty: int,
        entry_price: Optional[float],
        exit_price: Optional[float],
        realized_pnl: Optional[float],
        reason: str,
    ) -> None:
        try:
            print(
                "[CLOSE AUDIT] "
                f"symbol={symbol} side={side} qty={int(qty)} "
                f"entry={'' if entry_price is None else f'{float(entry_price):.2f}'} "
                f"exit={'' if exit_price is None else f'{float(exit_price):.2f}'} "
                f"realized={'' if realized_pnl is None else f'{float(realized_pnl):.2f}'} "
                f"reason={reason}"
            )
        except Exception:
            pass

    def _annotate_legs_with_entry_price(self, legs: List[dict]) -> None:
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            ltp = self._try_get_ltp_for_leg(leg)
            if ltp is not None:
                leg["entry_price"] = float(ltp)

    def _time_to_expiry_years(self, expiry_raw: object) -> Optional[float]:
        exp = None
        if isinstance(expiry_raw, dt_datetime):
            exp = expiry_raw.date()
        elif isinstance(expiry_raw, date):
            exp = expiry_raw
        else:
            exp = self._parse_any_date(expiry_raw)
        if exp is None:
            return None

        # Assume expiry at 15:30 IST on expiry date.
        try:
            exp_dt = dt_datetime.combine(exp, dt_time(15, 30))
            if IST is not None:
                try:
                    exp_dt = exp_dt.replace(tzinfo=IST)  # type: ignore[arg-type]
                except Exception:
                    pass
        except Exception:
            return None

        try:
            now_dt = dt_datetime.now(IST) if IST is not None else dt_datetime.now()
        except Exception:
            now_dt = dt_datetime.now()

        try:
            seconds = (exp_dt - now_dt).total_seconds()
        except Exception:
            return None
        if seconds <= 0:
            return None
        return float(seconds) / (365.0 * 24.0 * 3600.0)

    def _trade_net_option_delta(self, legs: List[dict], *, spot: float) -> Optional[float]:
        """Return net delta in underlying units for option legs only."""

        rate = float(getattr(self.cfg, "delta_hedge_rate", 0.06) or 0.06)
        assumed_iv = float(getattr(self.cfg, "delta_hedge_assumed_iv", 0.20) or 0.20)

        net = 0.0
        any_leg = False

        for leg in legs:
            if not isinstance(leg, dict):
                continue
            if bool(leg.get("is_hedge")):
                continue

            # Some entry paths / chain sources omit strike/expiry/option_type.
            # Derive them from the symbol so delta hedging can still work.
            self._ensure_option_leg_fields(leg)

            opt_type = str(leg.get("option_type") or "").strip().upper()
            if opt_type in {"CALL", "C"}:
                opt_type = "CE"
            elif opt_type in {"PUT", "P"}:
                opt_type = "PE"
            if opt_type not in {"CE", "PE"}:
                continue

            try:
                strike = float(leg.get("strike") or 0.0)
            except Exception:
                continue
            if strike <= 0:
                continue

            t = self._time_to_expiry_years(leg.get("expiry"))
            if t is None or t <= 0:
                continue

            iv_raw = leg.get("iv")
            try:
                iv = float(iv_raw) if iv_raw is not None else assumed_iv
            except Exception:
                iv = assumed_iv
            if iv <= 0:
                iv = assumed_iv

            side = str(leg.get("side") or "").strip().upper()
            if side not in {"BUY", "SELL"}:
                continue
            sign = 1.0 if side == "BUY" else -1.0

            try:
                qty = float(leg.get("quantity") or 0.0)
            except Exception:
                qty = 0.0
            if qty <= 0:
                continue

            try:
                d = float(delta(float(spot), float(strike), float(t), float(rate), float(iv), opt_type))
            except Exception:
                continue

            any_leg = True
            net += sign * d * qty

        return float(net) if any_leg else None

    def _current_hedge_qty_from_legs(self, legs: List[dict]) -> float:
        qty = 0.0
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            if not bool(leg.get("is_hedge")):
                continue

            # Delta-hedge quantity is for underlying hedges only.
            # Do not include option hedge legs (CE/PE) in this net.
            try:
                opt_type = str(leg.get("option_type") or "").strip().upper()
            except Exception:
                opt_type = ""
            try:
                sym_u = str(leg.get("symbol") or "").strip().upper()
            except Exception:
                sym_u = ""
            if opt_type in {"CE", "PE"} or sym_u.endswith("CE") or sym_u.endswith("PE"):
                continue

            side = str(leg.get("side") or "").strip().upper()
            if side not in {"BUY", "SELL"}:
                continue
            sign = 1.0 if side == "BUY" else -1.0
            try:
                q = float(leg.get("quantity") or 0.0)
            except Exception:
                q = 0.0
            if q <= 0:
                continue
            qty += sign * q
        return float(qty)

    def _upsert_underlying_hedge_leg(
        self,
        legs: List[dict],
        *,
        symbol: str,
        token: Optional[str],
        exchange: Optional[str],
        new_signed_qty: float,
        fill_price: Optional[float],
        prev_signed_qty: float,
    ) -> None:
        """Ensure exactly one underlying hedge leg exists for `symbol`.

        `new_signed_qty` is positive for BUY, negative for SELL.
        Updates quantity/side in-place and removes any duplicates.
        """

        sym = str(symbol or "").strip()
        if not sym:
            return

        # Remove all existing underlying hedge legs for this symbol.
        existing: List[dict] = []
        for leg in list(legs):
            if not isinstance(leg, dict):
                continue
            if not bool(leg.get("is_hedge")):
                continue
            try:
                s = str(leg.get("symbol") or "").strip()
            except Exception:
                s = ""
            if s != sym:
                continue

            # Skip option hedge legs.
            try:
                ot = str(leg.get("option_type") or "").strip().upper()
            except Exception:
                ot = ""
            s_u = s.upper()
            if ot in {"CE", "PE"} or s_u.endswith("CE") or s_u.endswith("PE"):
                continue

            existing.append(leg)

        for leg in existing:
            try:
                legs.remove(leg)
            except Exception:
                pass

        # If net qty is ~0, hedge is flat.
        try:
            signed = float(new_signed_qty)
        except Exception:
            signed = 0.0
        if abs(signed) < 1e-9:
            return

        side = "BUY" if signed > 0 else "SELL"
        qty = int(abs(signed))
        if qty <= 0:
            return

        # Compute a reasonable entry price for the remaining net position.
        entry_price = None
        prev_signed = float(prev_signed_qty)
        prev_abs = abs(prev_signed)
        fill_abs = abs(float(signed) - float(prev_signed))

        # If we crossed through zero, treat as a fresh position at fill price.
        if prev_signed == 0.0 or (prev_signed > 0 and signed < 0) or (prev_signed < 0 and signed > 0):
            entry_price = float(fill_price) if fill_price is not None else None
        else:
            # Same direction. If we increased exposure, update avg.
            if abs(signed) > prev_abs and fill_price is not None and fill_abs > 0:
                # Old avg is unknown if we removed duplicates; best-effort use the first leg's entry.
                old_entry = None
                if existing:
                    try:
                        old_entry = float(existing[0].get("entry_price")) if existing[0].get("entry_price") is not None else None
                    except Exception:
                        old_entry = None
                if old_entry is not None:
                    entry_price = (old_entry * prev_abs + float(fill_price) * fill_abs) / float(abs(signed))
                else:
                    entry_price = float(fill_price)
            else:
                # Reduced exposure: keep previous entry if available.
                if existing:
                    try:
                        entry_price = float(existing[0].get("entry_price")) if existing[0].get("entry_price") is not None else None
                    except Exception:
                        entry_price = None
                if entry_price is None and fill_price is not None:
                    entry_price = float(fill_price)

        legs.append(
            {
                "symbol": sym,
                "token": str(token or "").strip() or None,
                "exchange": str(exchange or "").strip() or None,
                "side": side,
                "quantity": int(qty),
                "entry_price": float(entry_price) if entry_price is not None else None,
                "is_hedge": True,
            }
        )

    def _pick_step_rounded_qty(self, target_qty: float, *, step: int) -> float:
        if step <= 0:
            return float(target_qty)
        import math

        steps = float(target_qty) / float(step)
        lo = math.floor(steps) * float(step)
        hi = math.ceil(steps) * float(step)
        # Choose the candidate closer to target (ties -> smaller magnitude).
        if abs(target_qty - lo) < abs(target_qty - hi):
            return float(lo)
        if abs(target_qty - hi) < abs(target_qty - lo):
            return float(hi)
        # Tie: round half-steps away from zero (slightly more aggressive hedging).
        return float(hi) if float(target_qty) >= 0 else float(lo)

    def _rebalance_delta_hedge(self, trade: Dict[str, object], *, spot: float, force: bool = False) -> None:
        meta = trade.get("meta") if isinstance(trade.get("meta"), dict) else {}
        if not isinstance(meta, dict) or not bool(meta.get("delta_hedge")):
            return

        legs = trade.get("legs")
        if not isinstance(legs, list):
            return

        self._ensure_hedge_orders_day()

        try:
            dh_hours_only = bool(getattr(self.cfg, "delta_hedge_market_hours_only", False))
        except Exception:
            dh_hours_only = False
        if dh_hours_only and bool(getattr(self.cfg, "enable_live_trading", False)) and not is_market_open():
            return

        hedge_sym_raw, hedge_exch_hint, inferred_root = self._pick_delta_hedge_symbol_for_trade(trade)
        hedge_sym_raw = str(hedge_sym_raw or "").strip()
        hedge_exch_hint = str(hedge_exch_hint or "").strip()
        if inferred_root:
            meta["delta_hedge_underlying"] = str(inferred_root)
        if not hedge_sym_raw:
            if not bool(meta.get("delta_hedge_missing_warned")):
                root_hint = str(inferred_root or meta.get("delta_hedge_underlying") or "").strip().upper()
                if root_hint in {"NIFTY", "BANKNIFTY"}:
                    print(
                        f"[DELTA HEDGE] Using default hedge fallback for {root_hint}; set MSTOCK_DELTA_HEDGE_SYMBOL_{root_hint} or MSTOCK_DELTA_HEDGE_SYMBOL to override."
                    )
                else:
                    print(
                        "[DELTA HEDGE] No hedge symbol configured; delta hedge skipped because the underlying could not be inferred."
                    )
                meta["delta_hedge_missing_warned"] = True
                trade["meta"] = meta
            return

        # Respect rebalance cadence.
        now_ts = time.time()
        if not force:
            try:
                interval = float(getattr(self.cfg, "delta_hedge_rebalance_interval_sec", 30.0) or 30.0)
            except Exception:
                interval = 30.0
            try:
                last_ts = float(meta.get("delta_hedge_last_ts") or 0.0)
            except Exception:
                last_ts = 0.0
            if interval > 0 and (now_ts - last_ts) < interval:
                return

        net_delta = self._trade_net_option_delta(legs, spot=float(spot))
        if net_delta is None:
            # Portfolio-only hedging mode: allow delta-hedge to run even when
            # there are no option legs (net option delta = 0) so that beta-based
            # equity exposure can still be hedged.
            if bool(meta.get("delta_hedge_portfolio_only")):
                net_delta = 0.0
            else:
                if self._debug_delta_hedge():
                    try:
                        tid = str(trade.get("trade_id") or "")
                    except Exception:
                        tid = ""
                    # Often caused by missing strike/expiry/option_type/side/qty in leg snapshots.
                    self._log_throttled(
                        f"[DELTA HEDGE] Skip: unable to compute net option delta (missing/invalid leg fields). trade_id={tid}",
                        key=f"deltahedge-netdelta-none:{tid}",
                        interval_sec=30.0,
                    )
                return

        # Option delta + optional equity-portfolio beta delta.
        beta_units = None
        try:
            # Use the inferred root (when available) to pick the benchmark index.
            bench = "NIFTY"
            try:
                bench = str(meta.get("delta_hedge_underlying") or "NIFTY").strip().upper() or "NIFTY"
            except Exception:
                bench = "NIFTY"
            beta_units = self._portfolio_beta_delta_units(index_symbol=f"NSE:{bench}", spot=float(spot))
        except Exception:
            beta_units = None

        if beta_units is not None:
            meta["delta_hedge_portfolio_beta_units"] = float(beta_units)
            net_delta = float(net_delta) + float(beta_units)
        else:
            meta.pop("delta_hedge_portfolio_beta_units", None)

        meta["delta_hedge_net_delta"] = float(net_delta)

        try:
            base_tol = float(getattr(self.cfg, "delta_hedge_delta_tolerance", 10.0) or 10.0)
        except Exception:
            base_tol = 10.0
        try:
            entry_tol = float(getattr(self.cfg, "delta_hedge_entry_tolerance", 0.0) or 0.0)
        except Exception:
            entry_tol = 0.0
        try:
            exit_tol = float(getattr(self.cfg, "delta_hedge_exit_tolerance", 0.0) or 0.0)
        except Exception:
            exit_tol = 0.0
        
        # ---- #9: Volatility-Weighted Delta Hedging Tolerance ----
        try:
            if bool(getattr(self.cfg, "delta_hedge_vol_adjust_enabled", False)):
                iv_hist = list(getattr(self, "_iv_pct_hist", []) or [])
                valid_iv = [float(x) for x in iv_hist if float(x) > 0]
                if len(valid_iv) >= 5:
                    current_iv = valid_iv[-1]
                    sorted_iv = sorted(valid_iv)
                    rank = sum(1 for x in sorted_iv if x <= current_iv)
                    iv_pctile = float(rank) / float(len(sorted_iv)) * 100.0
                    low_factor = float(getattr(self.cfg, "delta_hedge_vol_low_tol_factor", 0.7) or 0.7)
                    high_factor = float(getattr(self.cfg, "delta_hedge_vol_high_tol_factor", 1.5) or 1.5)
                    # Low IV (< 30th percentile) => tighter tolerance (more hedging)
                    # High IV (> 70th percentile) => wider tolerance (less hedging)
                    if iv_pctile <= 30.0:
                        vol_adj = float(low_factor)
                    elif iv_pctile >= 70.0:
                        vol_adj = float(high_factor)
                    else:
                        # Linear interpolation between 30th-70th percentile
                        normalized = (iv_pctile - 30.0) / 40.0
                        vol_adj = float(low_factor) + normalized * (float(high_factor) - float(low_factor))
                    base_tol = float(base_tol) * float(vol_adj)
                    entry_tol = float(entry_tol) * float(vol_adj) if entry_tol > 0 else 0.0
                    exit_tol = float(exit_tol) * float(vol_adj) if exit_tol > 0 else 0.0
                    try:
                        trade["_vol_adj_dh_tolerance"] = float(base_tol)
                    except Exception:
                        pass
        except Exception:
            pass
        if entry_tol <= 0:
            entry_tol = float(base_tol)
        if exit_tol <= 0:
            exit_tol = float(base_tol)
        if entry_tol < exit_tol:
            entry_tol = float(exit_tol)

        def _row_get_any(row: dict, keys: tuple[str, ...]) -> str:
            for k in keys:
                if k in row and row.get(k) is not None:
                    s = str(row.get(k)).strip()
                    if s:
                        return s
            return ""

        def _row_get_float(row: dict, keys: tuple[str, ...]) -> Optional[float]:
            s = _row_get_any(row, keys)
            if not s:
                return None
            try:
                return float(s)
            except Exception:
                return None

        def _row_get_int(row: dict, keys: tuple[str, ...]) -> Optional[int]:
            v = _row_get_float(row, keys)
            if v is None:
                return None
            try:
                return int(v)
            except Exception:
                return None

        def _extract_net_qty(row: dict) -> Optional[int]:
            # Common explicit net qty keys.
            for keys in (
                ("netqty", "netQty", "net_quantity", "netQuantity"),
                ("net", "net_position", "netPosition"),
            ):
                q = _row_get_int(row, keys)
                if q is not None:
                    return int(q)

            # Common buy/sell qty keys.
            buy = _row_get_int(row, ("buyqty", "buyQty", "buy_quantity", "buyQuantity", "buyQtyTotal"))
            sell = _row_get_int(row, ("sellqty", "sellQty", "sell_quantity", "sellQuantity", "sellQtyTotal"))
            if buy is not None or sell is not None:
                return int(buy or 0) - int(sell or 0)

            # Fallback: quantity + side.
            qty = _row_get_int(row, ("quantity", "qty", "netQty"))
            if qty is None:
                return None
            side = _row_get_any(row, ("transactiontype", "transactionType", "side")).upper()
            if side in {"SELL", "S"}:
                return -abs(int(qty))
            if side in {"BUY", "B"}:
                return abs(int(qty))
            return int(qty)

        # Track current hedge qty from our trade legs.
        internal_qty = self._current_hedge_qty_from_legs(legs)
        meta["delta_hedge_qty_internal"] = float(internal_qty)

        effective_qty = float(internal_qty)
        meta.pop("delta_hedge_qty_external", None)
        meta.pop("delta_hedge_qty_holdings", None)

        include_pos = bool(getattr(self.cfg, "delta_hedge_include_broker_positions", False))
        include_hold = bool(getattr(self.cfg, "delta_hedge_include_holdings", False))

        # If requested, use broker-reported net qty for the hedge instrument.
        # This helps hedge against existing portfolio positions and avoids double-hedging.
        exch_resolved = None
        tok_resolved = None
        if include_pos or include_hold:
            try:
                exch_resolved, tok_resolved = self.client.resolve_exchange_token(
                    hedge_sym_raw, exchange_hint=hedge_exch_hint or None
                )
            except Exception:
                exch_resolved, tok_resolved = None, None

        broker_qty_val: Optional[int] = None
        if include_pos and tok_resolved:
            try:
                rows = self.client.get_open_positions()
            except Exception:
                rows = []
            total = 0
            any_ok = False
            for r in rows:
                if not isinstance(r, dict):
                    continue
                tok = _row_get_any(r, ("symboltoken", "symbolToken", "token", "instrumentToken", "instrument_token", "symbol_token"))
                if tok and str(tok).strip() == str(tok_resolved).strip():
                    q = _extract_net_qty(r)
                    if q is None:
                        continue
                    total += int(q)
                    any_ok = True
            if any_ok:
                broker_qty_val = int(total)
                meta["delta_hedge_qty_external"] = float(broker_qty_val)
                effective_qty = float(broker_qty_val)

        # Include holdings for ETF-style hedges (mostly NSE).
        if include_hold and exch_resolved and str(exch_resolved).strip().upper() in {"NSE", "BSE"}:
            hold_qty = 0
            any_hold = False
            try:
                hrows = self.client.get_holdings()
            except Exception:
                hrows = []
            hedge_ts = hedge_sym_raw.split(":", 1)[-1].strip().upper()
            for r in hrows:
                if not isinstance(r, dict):
                    continue
                ts = _row_get_any(r, ("tradingsymbol", "tradingSymbol", "symbol", "name")).strip().upper()
                if ts and hedge_ts and ts != hedge_ts:
                    continue
                q = _row_get_int(r, ("quantity", "qty", "netqty", "netQty"))
                if q is None:
                    continue
                hold_qty += int(q)
                any_hold = True
            if any_hold:
                meta["delta_hedge_qty_holdings"] = float(hold_qty)
                # Holdings are additive. If broker positions are used as authoritative,
                # add holdings on top (delivery holdings may not appear in positions).
                effective_qty = float(effective_qty) + float(hold_qty)

        # Downstream code expects `current_qty`.
        current_qty = float(effective_qty)

        # Persist for UI/debug.
        meta["delta_hedge_qty"] = float(effective_qty)

        # If we're already within tolerance, don't churn.
        try:
            residual = float(net_delta) + float(effective_qty)
        except Exception:
            residual = float(net_delta)
        if abs(residual) <= float(exit_tol):
            meta["delta_hedge_last_ts"] = float(now_ts)
            trade["meta"] = meta
            if self._debug_delta_hedge():
                try:
                    tid = str(trade.get("trade_id") or "")
                except Exception:
                    tid = ""
                self._log_throttled(
                    f"[DELTA HEDGE] No-op: residual within tolerance (residual={residual:.1f}, tol={exit_tol:.1f}). trade_id={tid}",
                    key=f"deltahedge-within-tol:{tid}",
                    interval_sec=30.0,
                )
            return
        if abs(residual) < float(entry_tol):
            meta["delta_hedge_last_ts"] = float(now_ts)
            trade["meta"] = meta
            if self._debug_delta_hedge():
                try:
                    tid = str(trade.get("trade_id") or "")
                except Exception:
                    tid = ""
                self._log_throttled(
                    f"[DELTA HEDGE] No-op: residual below entry tolerance (residual={residual:.1f}, entry_tol={entry_tol:.1f}). trade_id={tid}",
                    key=f"deltahedge-below-entry-tol:{tid}",
                    interval_sec=30.0,
                )
            return

        if not force:
            try:
                min_move = float(getattr(self.cfg, "delta_hedge_min_spot_move", 0.0) or 0.0)
            except Exception:
                min_move = 0.0
            try:
                min_move_mult = float(
                    getattr(self.cfg, "delta_hedge_min_spot_move_atr_mult", 0.0) or 0.0
                )
            except Exception:
                min_move_mult = 0.0
            atr_for_move = None
            try:
                if isinstance(meta, dict):
                    atr_for_move = meta.get("entry_atr") or meta.get("atr")
            except Exception:
                atr_for_move = None
            if atr_for_move is None:
                atr_for_move = trade.get("atr")
            try:
                if min_move_mult > 0 and atr_for_move is not None:
                    min_move = max(float(min_move), float(atr_for_move) * float(min_move_mult))
            except Exception:
                pass
            try:
                last_spot = float(meta.get("delta_hedge_last_spot") or 0.0)
            except Exception:
                last_spot = 0.0
            if min_move > 0 and last_spot > 0:
                if abs(float(spot) - float(last_spot)) < float(min_move):
                    meta["delta_hedge_last_ts"] = float(now_ts)
                    trade["meta"] = meta
                    return

        try:
            max_orders = int(getattr(self.cfg, "delta_hedge_max_orders_per_day", 0) or 0)
        except Exception:
            max_orders = 0
        if max_orders > 0 and int(self.state.delta_hedge_orders_today) >= max_orders:
            meta["delta_hedge_last_ts"] = float(now_ts)
            trade["meta"] = meta
            return

        # Compute desired hedge qty to bring total delta ~ 0.
        try:
            adj = float(getattr(self.cfg, "delta_hedge_adjustment_factor", 1.0) or 1.0)
        except Exception:
            adj = 1.0
        if adj <= 0:
            adj = 1.0
        if adj > 1.0:
            adj = 1.0
        if adj < 1.0:
            target_qty = float(current_qty) - float(residual) * float(adj)
        else:
            target_qty = -float(net_delta)
        try:
            step = int(getattr(self.cfg, "delta_hedge_step_qty", 0) or 0)
        except Exception:
            step = 0
        if step <= 0:
            try:
                step = int(getattr(self.cfg, "lot_size", 1) or 1)
            except Exception:
                step = 1
        if step <= 0:
            step = 1

        desired_qty = self._pick_step_rounded_qty(target_qty, step=step)

        try:
            max_abs = int(getattr(self.cfg, "delta_hedge_max_abs_qty", 0) or 0)
        except Exception:
            max_abs = 0
        if max_abs <= 0:
            max_abs = int(step) * 10
        if desired_qty > float(max_abs):
            desired_qty = float(max_abs)
        if desired_qty < -float(max_abs):
            desired_qty = -float(max_abs)

        try:
            max_adj = float(getattr(self.cfg, "delta_hedge_max_adjust_abs_qty", 0.0) or 0.0)
        except Exception:
            max_adj = 0.0

        diff = float(desired_qty) - float(current_qty)
        if max_adj > 0 and abs(diff) > float(max_adj):
            diff = float(max_adj) if diff > 0 else -float(max_adj)
            desired_qty = self._pick_step_rounded_qty(float(current_qty) + float(diff), step=step)
            diff = float(desired_qty) - float(current_qty)
        if abs(diff) < float(step):
            meta["delta_hedge_last_ts"] = float(now_ts)
            trade["meta"] = meta
            if self._debug_delta_hedge():
                try:
                    tid = str(trade.get("trade_id") or "")
                except Exception:
                    tid = ""
                self._log_throttled(
                    f"[DELTA HEDGE] No-op: required adjustment smaller than step (diff={diff:.1f}, step={step}). trade_id={tid}",
                    key=f"deltahedge-below-step:{tid}",
                    interval_sec=30.0,
                )
            return

        # Normalize symbol/exchange for placing orders.
        sym_for_order = hedge_sym_raw.split(":", 1)[-1].strip()
        try:
            exch, tok = self.client.resolve_exchange_token(hedge_sym_raw, exchange_hint=hedge_exch_hint or None)
        except Exception as exc:
            if not bool(meta.get("delta_hedge_resolve_warned")):
                print(f"[DELTA HEDGE] Unable to resolve token for hedge symbol {hedge_sym_raw!r}: {exc}")
                meta["delta_hedge_resolve_warned"] = True
                trade["meta"] = meta
            return

        side = "BUY" if diff > 0 else "SELL"
        qty = int(abs(diff))
        if qty <= 0:
            meta["delta_hedge_last_ts"] = float(now_ts)
            trade["meta"] = meta
            return

        try:
            require_ba = bool(getattr(self.cfg, "delta_hedge_require_bid_ask", False))
        except Exception:
            require_ba = False
        try:
            max_spread_pct = float(getattr(self.cfg, "delta_hedge_max_spread_pct", 0.0) or 0.0)
        except Exception:
            max_spread_pct = 0.0
        try:
            max_spread_abs = float(getattr(self.cfg, "delta_hedge_max_spread_abs", 0.0) or 0.0)
        except Exception:
            max_spread_abs = 0.0
        if require_ba or max_spread_pct > 0 or max_spread_abs > 0:
            bid, ask, _ltp = self.client.get_bid_ask(sym_for_order, exchange_hint=str(exch or "").strip() or None)
            if require_ba and (bid is None or ask is None):
                meta["delta_hedge_last_ts"] = float(now_ts)
                trade["meta"] = meta
                return
            if bid is not None and ask is not None:
                try:
                    spread = float(ask) - float(bid)
                except Exception:
                    spread = None
                if spread is not None:
                    if spread < 0:
                        spread = abs(spread)
                    if max_spread_abs > 0 and float(spread) > float(max_spread_abs):
                        meta["delta_hedge_last_ts"] = float(now_ts)
                        trade["meta"] = meta
                        return
                    if max_spread_pct > 0:
                        mid = (float(ask) + float(bid)) / 2.0
                        if mid > 0:
                            pct = (float(spread) / float(mid)) * 100.0
                            if pct > float(max_spread_pct) * 100.0:
                                meta["delta_hedge_last_ts"] = float(now_ts)
                                trade["meta"] = meta
                                return

        ltp = self._try_get_ltp(sym_for_order, exchange=str(exch or "").strip() or None)

        if not self.cfg.enable_live_trading:
            print(
                f"[PAPER][DELTA HEDGE] {side} {sym_for_order} x{qty} to hedge Δ={net_delta:.1f} (residual={residual:.1f})"
            )
        else:
            try:
                self.client.place_order(
                    symbol=sym_for_order,
                    side=side,
                    quantity=qty,
                    exchange=str(exch or "").strip() or None,
                    symbol_token=str(tok or "").strip() or None,
                )
            except Exception as exc:
                print(f"[DELTA HEDGE] Hedge order failed: {exc}")
                return

        # Update hedge leg as a single net position (avoid appending duplicates).
        # After executing the order, the new net hedge qty should be desired_qty.
        self._upsert_underlying_hedge_leg(
            legs,
            symbol=sym_for_order,
            token=str(tok or "").strip() or None,
            exchange=str(exch or "").strip() or None,
            new_signed_qty=float(internal_qty) + float(diff),
            fill_price=float(ltp) if ltp is not None else None,
            prev_signed_qty=float(internal_qty),
        )

        meta["delta_hedge_qty"] = float(desired_qty)
        meta["delta_hedge_last_spot"] = float(spot)
        self.state.delta_hedge_orders_today += 1

        meta["delta_hedge_last_ts"] = float(now_ts)
        trade["meta"] = meta

        # Emit an UPDATE so the UI can show the NET Δ-HEDGE row immediately.
        if self._has_event_sink():
            try:
                trade_id = str(trade.get("trade_id") or "(unknown)")
                name = str(trade.get("name") or "")
                pt = str(trade.get("position_type") or "").strip().lower()
                if pt not in {"multi", "directional"}:
                    if trade_id.startswith("M"):
                        pt = "multi"
                    elif trade_id.startswith("D"):
                        pt = "directional"
                    else:
                        pt = "multi"

                leg_dicts: List[Dict[str, object]] = []
                for l in legs:
                    if not isinstance(l, dict):
                        continue
                    leg_copy = dict(l)
                    leg_copy["ltp"] = self._try_get_ltp_for_leg(leg_copy)
                    leg_dicts.append(leg_copy)

                self._emit(
                    TradeLogEvent(
                        ts=time.time(),
                        event="UPDATE",
                        trade_id=trade_id,
                        position_type=pt,
                        name=name,
                        legs=leg_dicts,
                        mtm=self._compute_legs_mtm(leg_dicts),
                    )
                )
            except Exception:
                pass

    def _capture_entry_premium_meta(
        self,
        legs: List[dict],
        meta: Dict[str, object],
        *,
        atr_val: Optional[float] = None,
    ) -> None:
        """Capture entry premium (absolute) so MTM % exits work for any structure.

        The signed premium uses a simple convention:
        - SELL legs add premium (credit)
        - BUY legs subtract premium (debit)
        """

        self._annotate_legs_with_entry_price(legs)
        entry_premium = 0.0
        any_price = False

        for leg in legs:
            if not isinstance(leg, dict):
                continue
            p = leg.get("entry_price")
            try:
                pf = float(p) if p is not None else None
            except Exception:
                pf = None
            if pf is None:
                continue

            q = float(leg.get("quantity") or 0)
            side = str(leg.get("side") or "").upper()
            if q <= 0 or side not in {"BUY", "SELL"}:
                continue

            any_price = True
            if side == "SELL":
                entry_premium += pf * q
            else:
                entry_premium -= pf * q

        meta["entry_premium"] = float(entry_premium) if any_price else None
        meta["entry_premium_abs"] = float(abs(entry_premium)) if any_price else None
        meta["entry_atr"] = float(atr_val) if atr_val is not None else meta.get("entry_atr")

    def _compute_legs_mtm(self, legs: List[dict]) -> Optional[float]:
        mtm = 0.0
        any_price = False
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            side = str(leg.get("side") or "").upper()
            qty = int(leg.get("quantity") or 0)
            symbol = str(leg.get("symbol") or "")
            exchange = str(leg.get("exchange") or "") or None
            entry = leg.get("entry_price")
            try:
                entry_f = float(entry) if entry is not None else None
            except Exception:
                entry_f = None

            if entry_f is None or qty <= 0 or not symbol or side not in {"BUY", "SELL"}:
                continue

            cur = self._try_get_ltp_for_leg(leg)
            if cur is None:
                continue

            # Cache on the leg dict so callers can reuse without extra quote calls.
            # This is best-effort and may be overwritten on the next refresh.
            try:
                leg["ltp"] = float(cur)
            except Exception:
                pass

            any_price = True
            sign = 1.0 if side == "BUY" else -1.0
            mtm += (float(cur) - float(entry_f)) * sign * float(qty)

        return float(mtm) if any_price else None

    def _compute_legs_mtm_cached(self, legs: List[dict]) -> Optional[float]:
        """Compute MTM using cached LTP when present.

        This is a lighter-weight variant used for GPT snapshots and UI refreshes.
        If a leg already contains `ltp`, we avoid an extra quote call.
        """

        mtm = 0.0
        any_price = False
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            side = str(leg.get("side") or "").upper()
            qty = int(leg.get("quantity") or 0)
            symbol = str(leg.get("symbol") or "")
            entry = leg.get("entry_price")
            try:
                entry_f = float(entry) if entry is not None else None
            except Exception:
                entry_f = None

            if entry_f is None or qty <= 0 or not symbol or side not in {"BUY", "SELL"}:
                continue

            cur: Optional[float]
            try:
                ltp_cached = leg.get("ltp")
                cur = float(ltp_cached) if ltp_cached is not None else None
            except Exception:
                cur = None

            if cur is None:
                cur = self._try_get_ltp_for_leg(leg)
            if cur is None:
                continue

            any_price = True
            sign = 1.0 if side == "BUY" else -1.0
            mtm += (float(cur) - float(entry_f)) * sign * float(qty)

        return float(mtm) if any_price else None

    def _refresh_legs_ltp_and_mtm(self, legs: List[dict]) -> Optional[float]:
        """Refresh leg LTPs once and compute MTM from the refreshed snapshot."""

        refreshed: List[dict] = []
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            cur = self._try_get_ltp_for_leg(leg)
            if cur is not None:
                try:
                    leg["ltp"] = float(cur)
                except Exception:
                    pass
            refreshed.append(leg)
        return self._compute_legs_mtm_cached(refreshed)

    def _compute_total_unrealized_mtm(self) -> float:
        """Compute aggregate unrealized MTM across all open positions.

        Iterates every leg in open_directional and open_multi, fetches current
        LTP, and returns sum((ltp - entry_price) * qty * sign) for all legs.

        This is used by _risk_status() so that the max_daily_loss guard fires
        even when a large adverse move happens on positions that haven't been
        closed yet (only realized P&L was counted before this fix).

        Returns 0.0 if no open positions or if LTP cannot be fetched for any.
        """
        total = 0.0
        all_trades = list(self.state.open_multi or []) + list(self.state.open_directional or [])
        for tr in all_trades:
            if not isinstance(tr, dict):
                continue
            legs = tr.get("legs")
            if not isinstance(legs, list):
                continue
            mtm = self._compute_legs_mtm_cached(list(legs))
            if mtm is not None:
                total += float(mtm)
        return float(total)

    def _entry_iv_percentile_factor(self) -> float:
        """Return a sizing multiplier based on the IV percentile of the underlying.

        When current IV is in a high percentile, we size down (avoid overpaying).
        When IV is in a low percentile, we size normally or slightly up.

        Reads config keys:
          iv_sizing_enabled (bool) - master switch
          iv_sizing_lookback_days (int) - days of IV history to track (default 20)
          iv_sizing_high_percentile (float) - above this, reduce size (default 75.0)
          iv_sizing_low_percentile (float) - below this, normal size (default 25.0)
          iv_sizing_high_factor (float) - multiplier when IV is high (default 0.5)
          iv_sizing_max_factor (float) - max factor allowed (default 1.0)

        Returns 1.0 when disabled or on error (no impact).
        """
        try:
            if not bool(getattr(self.cfg, "iv_sizing_enabled", False)):
                return 1.0
        except Exception:
            return 1.0

        try:
            lookback = int(getattr(self.cfg, "iv_sizing_lookback_days", 20) or 20)
        except Exception:
            lookback = 20
        try:
            high_pct = float(getattr(self.cfg, "iv_sizing_high_percentile", 75.0) or 75.0)
        except Exception:
            high_pct = 75.0
        try:
            low_pct = float(getattr(self.cfg, "iv_sizing_low_percentile", 25.0) or 25.0)
        except Exception:
            low_pct = 25.0
        try:
            high_factor = float(getattr(self.cfg, "iv_sizing_high_factor", 0.5) or 0.5)
        except Exception:
            high_factor = 0.5
        try:
            max_factor = float(getattr(self.cfg, "iv_sizing_max_factor", 1.0) or 1.0)
        except Exception:
            max_factor = 1.0

        high_pct = max(1.0, min(99.0, float(high_pct)))
        low_pct = max(0.0, min(float(high_pct) - 1.0, float(low_pct)))
        lookback = max(2, min(100, int(lookback)))
        high_factor = max(0.1, min(1.0, float(high_factor)))
        max_factor = max(float(high_factor), min(2.0, float(max_factor)))

        # Build IV/ATR history from tracking attribute
        iv_hist = list(getattr(self, "_iv_pct_hist", []) or [])
        if len(iv_hist) < int(lookback):
            return 1.0

        recent = list(iv_hist)
        ordered = sorted(float(x) for x in recent if isinstance(x, (int, float)))
        if len(ordered) < 2:
            return 1.0

        current = float(ordered[-1])
        rank = sum(1 for x in ordered if x <= current)
        percentile = (float(rank) / float(len(ordered))) * 100.0

        if float(percentile) >= float(high_pct):
            pct_above = (float(percentile) - float(high_pct)) / (100.0 - float(high_pct) + 1e-9)
            pct_above = max(0.0, min(1.0, float(pct_above)))
            factor = float(high_factor) * (1.0 + float(pct_above)) / 2.0
            return max(float(high_factor) * 0.5, min(float(max_factor), float(factor)))
        elif float(percentile) <= float(low_pct):
            return min(float(max_factor), 1.0)
        else:
            return 1.0



    def _iv_percentile_strategy_bias(self) -> Optional[str]:
        """Return a strategy bias based on current IV percentile.
        
        Returns:
          "short_premium" - IV is high, favor selling premium (collect high premiums)
          "long_premium"  - IV is low, favor buying premium (cheap options)
          None            - No clear bias / feature disabled / insufficient data
        """
        try:
            if not bool(getattr(self.cfg, "iv_filter_enabled", False)):
                return None
        except Exception:
            return None
        
        try:
            high_pct = float(getattr(self.cfg, "iv_filter_high_percentile", 70.0) or 70.0)
        except Exception:
            high_pct = 70.0
        try:
            low_pct = float(getattr(self.cfg, "iv_filter_low_percentile", 30.0) or 30.0)
        except Exception:
            low_pct = 30.0
        
        iv_hist = list(getattr(self, "_iv_pct_hist", []) or [])
        valid = [float(x) for x in iv_hist if float(x) > 0]
        if len(valid) < 5:
            return None
        
        current_iv = valid[-1]
        sorted_vals = sorted(valid)
        rank = sum(1 for x in sorted_vals if x <= current_iv)
        pctile = float(rank) / float(len(sorted_vals)) * 100.0
        
        if pctile >= float(high_pct):
            return "short_premium"
        elif pctile <= float(low_pct):
            return "long_premium"
        return None

    def _dynamic_pyramid_max_level(self) -> int:
        """Compute max pyramid levels dynamically based on market conditions.

        Factors considered (when enable_dynamic_pyramiding is True):
          1. IV percentile - more levels when IV is low (cheap), fewer when high
          2. Winrate - more levels when recent winrate is above threshold
          3. ATR regime - standard levels in moderate ATR, reduced in extreme ATR
          4. Session time - tighter pyramid in opening/closing sessions

        When disabled or on error, returns the static max_pyramid_levels.
        """
        try:
            if not bool(getattr(self.cfg, "enable_dynamic_pyramiding", False)):
                return int(getattr(self.cfg, "max_pyramid_levels", 1) or 1)
        except Exception:
            return int(getattr(self.cfg, "max_pyramid_levels", 1) or 1)

        try:
            static_max = int(getattr(self.cfg, "max_pyramid_levels", 1) or 1)
        except Exception:
            static_max = 1
        try:
            iv_max = int(getattr(self.cfg, "pyramid_iv_max_levels", 0) or 0)
        except Exception:
            iv_max = 0
        try:
            atr_max = int(getattr(self.cfg, "pyramid_atr_max_levels", 0) or 0)
        except Exception:
            atr_max = 0
        try:
            win_min = float(getattr(self.cfg, "pyramid_winrate_min_for_extra", 50.0) or 50.0)
        except Exception:
            win_min = 50.0
        try:
            session_tighten = float(getattr(self.cfg, "pyramid_session_tighten_factor", 0.5) or 0.5)
        except Exception:
            session_tighten = 0.5
        try:
            cap = int(getattr(self.cfg, "pyramid_max_total_levels", 5) or 5)
        except Exception:
            cap = 5

        cap = max(1, int(cap))
        session_tighten = max(0.1, min(1.0, float(session_tighten)))

        # Start with static max as baseline
        dynamic_max = int(static_max)

        # Factor 1: IV percentile - when IV is in low percentile, allow more levels
        if int(iv_max) > 0:
            try:
                iv_factor = self._entry_iv_percentile_factor()
                # iv_factor near 1.0 means low/normal IV -> allow more levels
                # iv_factor near 0.5 means high IV -> fewer levels
                iv_bonus = int(round(float(iv_max) * float(iv_factor)))
                dynamic_max = max(int(dynamic_max), int(iv_bonus))
            except Exception:
                pass

        # Factor 2: ATR regime - in moderate ATR, allow standard; in extreme, reduce
        if int(atr_max) > 0:
            try:
                cached_atr = getattr(self, "_cached_atr", None)
                if cached_atr is not None and float(cached_atr) > 0:
                    try:
                        thresh_low = float(getattr(self.cfg, "strike_atr_threshold_low", 30.0) or 30.0)
                        thresh_high = float(getattr(self.cfg, "strike_atr_threshold_high", 60.0) or 60.0)
                    except Exception:
                        thresh_low, thresh_high = 30.0, 60.0
                    if float(thresh_low) <= float(cached_atr) <= float(thresh_high):
                        dynamic_max = max(int(dynamic_max), int(atr_max))
                    else:
                        dynamic_max = max(int(static_max), (int(static_max) + int(atr_max)) // 2)
            except Exception:
                pass

        # Factor 3: Winrate - if recent winrate exceeds threshold, allow more
        try:
            name = str(getattr(self.cfg, "strategy_name", "") or "").strip().lower() or "directional"
            wr = self._strategy_winrate.get(name) if isinstance(self._strategy_winrate, dict) else None
            if isinstance(wr, dict):
                wins = int(wr.get("wins", 0) or 0)
                losses = int(wr.get("losses", 0) or 0)
                total = int(wins) + int(losses)
                if int(total) > 0:
                    rate = (float(wins) / float(total)) * 100.0
                    if float(rate) >= float(win_min):
                        dynamic_max = int(dynamic_max) + 1
        except Exception:
            pass

        # Factor 4: Session time - tighten pyramid in opening/closing sessions
        try:
            session_adj = self._session_exit_adjustments()
            if isinstance(session_adj, dict):
                sl_mult = float(session_adj.get("sl_mult", 1.0) or 1.0)
                if float(sl_mult) < 1.0:
                    tight_levels = max(1, int(round(int(dynamic_max) * float(session_tighten))))
                    dynamic_max = min(int(dynamic_max), int(tight_levels))
        except Exception:
            pass

        # Clamp to cap and ensure at least 1
        dynamic_max = max(1, min(int(cap), int(dynamic_max)))
        return int(dynamic_max)


    def _held_days_since_open(self, opened_ts: float) -> int:
        """Return whole days held since `opened_ts` (best-effort).

        Uses calendar-day difference in IST when available, falling back to
        elapsed seconds / 86400.
        """

        opened = float(opened_ts or 0.0)
        if opened <= 0:
            return 0
        try:
            if IST is not None:
                opened_dt = dt_datetime.fromtimestamp(opened, IST)
                now_dt = dt_datetime.now(IST)
            else:
                opened_dt = dt_datetime.fromtimestamp(opened)
                now_dt = dt_datetime.now()
            return int((now_dt.date() - opened_dt.date()).days)
        except Exception:
            try:
                return int((time.time() - opened) // 86400)
            except Exception:
                return 0

    def _gpt_open_positions_mtm_snapshot(self, *, max_trades: int = 6) -> Dict[str, object]:
        """Compact MTM snapshot of current open positions for GPT.

        Keep this small and JSON-serializable.
        """

        try:
            max_n = int(max_trades)
        except Exception:
            max_n = 6
        if max_n < 0:
            max_n = 0

        positions: List[Dict[str, object]] = []
        mtm_multi_total = 0.0
        mtm_dir_total = 0.0
        entry_multi_abs_total = 0.0
        entry_dir_abs_total = 0.0
        mtm_multi_any = False
        mtm_dir_any = False
        entry_multi_any = False
        entry_dir_any = False

        def _entry_premium_abs_from_legs(legs: List[Dict[str, object]]) -> Optional[float]:
            entry_premium = 0.0
            any_price = False
            for leg in legs:
                if not isinstance(leg, dict):
                    continue
                side = str(leg.get("side") or "").upper()
                if side not in {"BUY", "SELL"}:
                    continue
                try:
                    qty = float(leg.get("quantity") or 0)
                except Exception:
                    qty = 0.0
                if qty <= 0:
                    continue
                p = leg.get("entry_price")
                try:
                    pf = float(p) if p is not None else None
                except Exception:
                    pf = None
                if pf is None:
                    continue

                any_price = True
                if side == "SELL":
                    entry_premium += pf * qty
                else:
                    entry_premium -= pf * qty
            if not any_price:
                return None
            return float(abs(entry_premium))

        def _summarize_trade(tr: Dict[str, object], pt: str) -> None:
            nonlocal mtm_multi_total, mtm_dir_total
            nonlocal entry_multi_abs_total, entry_dir_abs_total
            nonlocal mtm_multi_any, mtm_dir_any
            nonlocal entry_multi_any, entry_dir_any

            if not isinstance(tr, dict):
                return
            legs = tr.get("legs")
            if not isinstance(legs, list) or not legs:
                return

            # Work on copies to avoid mutating open trade state.
            legs_copy: List[Dict[str, object]] = [dict(l) for l in legs if isinstance(l, dict)]
            mtm_val = self._compute_legs_mtm_cached(legs_copy)
            entry_abs = _entry_premium_abs_from_legs(legs_copy)

            mtm_pct_of_entry: Optional[float] = None
            if mtm_val is not None and entry_abs is not None and float(entry_abs) > 0:
                mtm_pct_of_entry = float(mtm_val) / float(entry_abs) * 100.0

            if mtm_val is not None:
                if pt == "multi":
                    mtm_multi_total += float(mtm_val)
                    mtm_multi_any = True
                else:
                    mtm_dir_total += float(mtm_val)
                    mtm_dir_any = True

            if entry_abs is not None and float(entry_abs) > 0:
                if pt == "multi":
                    entry_multi_abs_total += float(entry_abs)
                    entry_multi_any = True
                else:
                    entry_dir_abs_total += float(entry_abs)
                    entry_dir_any = True

            try:
                opened_ts = float(tr.get("opened_ts") or 0.0)
            except Exception:
                opened_ts = 0.0

            leg_summ: List[Dict[str, object]] = []
            for l in legs_copy[:6]:
                leg_summ.append(
                    {
                        "symbol": str(l.get("symbol") or ""),
                        "side": str(l.get("side") or ""),
                        "qty": int(l.get("quantity") or 0),
                        "entry": l.get("entry_price"),
                        "ltp": l.get("ltp"),
                        "option_type": str(l.get("option_type") or ""),
                        "strike": l.get("strike"),
                    }
                )

            positions.append(
                {
                    "trade_id": str(tr.get("trade_id") or ""),
                    "position_type": str(pt),
                    "name": str(tr.get("name") or ""),
                    "mtm": float(mtm_val) if mtm_val is not None else None,
                    "entry_premium_abs": float(entry_abs) if entry_abs is not None else None,
                    "mtm_pct_of_entry": float(mtm_pct_of_entry) if mtm_pct_of_entry is not None else None,
                    "opened_ts": float(opened_ts) if opened_ts else None,
                    "legs": leg_summ,
                }
            )

        # Newest-first helps GPT focus on the most recent/most relevant exposure.
        try:
            multi_trades = list(self.state.open_multi or [])
        except Exception:
            multi_trades = []
        try:
            dir_trades = list(self.state.open_directional or [])
        except Exception:
            dir_trades = []

        def _sort_key(t: object) -> float:
            if not isinstance(t, dict):
                return 0.0
            try:
                return float(t.get("opened_ts") or 0.0)
            except Exception:
                return 0.0

        multi_trades.sort(key=_sort_key, reverse=True)
        dir_trades.sort(key=_sort_key, reverse=True)

        for tr in multi_trades:
            if len(positions) >= max_n:
                break
            _summarize_trade(tr if isinstance(tr, dict) else {}, "multi")

        for tr in dir_trades:
            if len(positions) >= max_n:
                break
            _summarize_trade(tr if isinstance(tr, dict) else {}, "directional")

        # Weighted MTM % (total MTM divided by total entry premium abs).
        dir_mtm_pct = None
        if mtm_dir_any and entry_dir_any and float(entry_dir_abs_total) > 0:
            dir_mtm_pct = float(mtm_dir_total) / float(entry_dir_abs_total) * 100.0
        multi_mtm_pct = None
        if mtm_multi_any and entry_multi_any and float(entry_multi_abs_total) > 0:
            multi_mtm_pct = float(mtm_multi_total) / float(entry_multi_abs_total) * 100.0
        total_mtm_pct = None
        total_entry_abs = float(entry_dir_abs_total + entry_multi_abs_total)
        total_mtm = float(mtm_dir_total + mtm_multi_total)
        if (mtm_dir_any or mtm_multi_any) and (entry_dir_any or entry_multi_any) and total_entry_abs > 0:
            total_mtm_pct = total_mtm / total_entry_abs * 100.0

        return {
            "counts": {
                "directional": int(len(getattr(self.state, "open_directional", []) or [])),
                "multi": int(len(getattr(self.state, "open_multi", []) or [])),
            },
            "mtm": {
                "directional": float(mtm_dir_total) if mtm_dir_any else None,
                "multi": float(mtm_multi_total) if mtm_multi_any else None,
                "total": float(mtm_dir_total + mtm_multi_total) if (mtm_dir_any or mtm_multi_any) else None,
            },
            "mtm_pct_of_entry": {
                "directional": float(dir_mtm_pct) if dir_mtm_pct is not None else None,
                "multi": float(multi_mtm_pct) if multi_mtm_pct is not None else None,
                "total": float(total_mtm_pct) if total_mtm_pct is not None else None,
            },
            "positions": positions,
        }

    # ---- Multi-leg helpers ----

    def _trend_strength(self, fast_ema: float, slow_ema: float, close: float) -> float:
        if close == 0:
            return 0.0
        return abs(fast_ema - slow_ema) / abs(close)

    def _parse_expiry_from_option_symbol(self, symbol: str) -> Optional[date]:
        """Best-effort parse NSE-style expiry from option trading symbols.

        Common formats include: NIFTY09JAN26..., BANKNIFTY16JAN26..., etc.
        We look for DDMMMYY anywhere in the string.
        """

        s = str(symbol or "").upper().strip()
        if not s:
            return None

        # Support both 2-digit (YY) and 4-digit (20YY) years.
        # Use (20\d{2}|\d{2}) to ensure we don't greedily eat into the strike price
        # if the year is 2 digits but followed by more digits (e.g. JAN2640000 -> 26, not 2640).
        m = re.search(r"(\d{2})([A-Z]{3})(20\d{2}|\d{2})", s)
        if not m:
            return None

        dd_s, mon_s, yy_s = m.group(1), m.group(2), m.group(3)
        try:
            dd = int(dd_s)
            if len(yy_s) == 4:
                year = int(yy_s)
            else:
                # NSE symbols use 2-digit year.
                year = 2000 + int(yy_s)
        except Exception:
            return None

        mon_map = {
            "JAN": 1,
            "FEB": 2,
            "MAR": 3,
            "APR": 4,
            "MAY": 5,
            "JUN": 6,
            "JUL": 7,
            "AUG": 8,
            "SEP": 9,
            "OCT": 10,
            "NOV": 11,
            "DEC": 12,
        }
        mm = mon_map.get(mon_s)
        if not mm:
            return None

        try:
            return date(year, mm, dd)
        except Exception:
            return None

    def _parse_strike_from_option_symbol(self, symbol: str) -> Optional[float]:
        """Best-effort parse strike from an NSE-style option trading symbol.

        We take the last contiguous digit group immediately before CE/PE.
        Example: NIFTY06FEB2619600CE -> 19600
        """

        s = str(symbol or "").upper().strip()
        if not s:
            return None

        sym = s.split(":", 1)[-1].strip()
        m = re.search(r"(\d+)(CE|PE)$", sym)
        if not m:
            return None
        try:
            strike = float(m.group(1))
        except Exception:
            return None
        if strike <= 0:
            return None
        return float(strike)

    def _ensure_option_leg_fields(self, leg: dict) -> None:
        """Fill missing option leg fields (strike/expiry/option_type) from symbol."""

        if not isinstance(leg, dict):
            return

        sym_raw = str(leg.get("symbol") or "").strip()
        sym = sym_raw.split(":", 1)[-1].strip().upper()
        if not sym:
            return

        # option_type
        opt_type = str(leg.get("option_type") or "").strip().upper()
        if opt_type in {"CALL", "C"}:
            opt_type = "CE"
        elif opt_type in {"PUT", "P"}:
            opt_type = "PE"
        if opt_type not in {"CE", "PE"}:
            if sym.endswith("CE"):
                opt_type = "CE"
            elif sym.endswith("PE"):
                opt_type = "PE"
        if opt_type in {"CE", "PE"}:
            leg["option_type"] = opt_type

        # strike
        strike_ok = False
        try:
            v = leg.get("strike")
            if v is not None and float(v) > 0:
                strike_ok = True
        except Exception:
            strike_ok = False
        if not strike_ok:
            strike_parsed = self._parse_strike_from_option_symbol(sym)
            if strike_parsed is not None:
                leg["strike"] = float(strike_parsed)

        # expiry
        exp_raw = leg.get("expiry")
        exp = None
        if isinstance(exp_raw, dt_datetime):
            exp = exp_raw.date()
        elif isinstance(exp_raw, date):
            exp = exp_raw
        elif exp_raw is not None:
            exp = self._parse_any_date(exp_raw)
        if exp is None:
            exp = self._parse_expiry_from_option_symbol(sym)
        if exp is not None:
            leg["expiry"] = exp

    def _is_last_thursday_of_month(self, d: date) -> bool:
        # Back-compat wrapper (originally hard-coded Thursday).
        return self._is_last_weekday_of_month(d)

    def _is_last_weekday_of_month(self, d: date) -> bool:
        wd = d.weekday()
        last_day = calendar.monthrange(d.year, d.month)[1]
        last = date(d.year, d.month, last_day)
        while last.weekday() != wd:
            last = last.replace(day=last.day - 1)
        return d == last

    def _parse_any_date(self, raw: object) -> Optional[date]:
        s = str(raw or "").strip()
        if not s:
            return None

        # Normalize common ISO variants.
        iso = s.replace("Z", "+00:00")
        try:
            parsed = dt_datetime.fromisoformat(iso)
            return parsed.date()
        except Exception:
            pass

        # Common user-input formats.
        for fmt in (
            "%d-%m-%Y",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%d-%b-%Y",
            "%d-%B-%Y",
            "%d %b %Y",
            "%d %B %Y",
            "%Y%m%d",
            "%d-%m-%y",
            "%d/%m/%y",
        ):
            try:
                return dt_datetime.strptime(s, fmt).date()
            except Exception:
                continue

        # Accept DDMMMYY / DDMMMYYYY (e.g. 06FEB26, 06FEB2026).
        up = s.upper().replace("-", "").replace("/", "").replace(" ", "")
        m = re.fullmatch(r"(\d{2})([A-Z]{3})(\d{2}|\d{4})", up)
        if not m:
            return None

        dd_s, mon_s, yy_s = m.group(1), m.group(2), m.group(3)
        mon_map = {
            "JAN": 1,
            "FEB": 2,
            "MAR": 3,
            "APR": 4,
            "MAY": 5,
            "JUN": 6,
            "JUL": 7,
            "AUG": 8,
            "SEP": 9,
            "OCT": 10,
            "NOV": 11,
            "DEC": 12,
        }
        mm = mon_map.get(mon_s)
        if not mm:
            return None

        try:
            dd = int(dd_s)
            if len(yy_s) == 2:
                year = 2000 + int(yy_s)
            else:
                year = int(yy_s)
            return date(year, mm, dd)
        except Exception:
            return None

    def _row_expiry_date(self, row: dict) -> Optional[date]:
        exp_raw = row.get("expiry")
        if isinstance(exp_raw, dt_datetime):
            return exp_raw.date()
        if isinstance(exp_raw, date):
            return exp_raw
        if exp_raw is not None:
            exp = self._parse_any_date(exp_raw)
            if exp is not None:
                return exp
        sym_raw = str(row.get("symbol") or "").upper()
        sym = sym_raw.split(":", 1)[-1]
        return self._parse_expiry_from_option_symbol(sym)

    def _filter_weekly_only(self, chain: List[dict]) -> List[dict]:
        """Keep only weekly contracts for the nearest weekly expiry of the current symbol."""

        if not chain:
            return []

        today = date.today()

        # Optional hard override: use a specific expiry date if configured.
        override_exp: Optional[date] = None
        raw_override = str(getattr(self.cfg, "target_expiry", "") or "").strip()
        if raw_override:
            override_exp = self._parse_any_date(raw_override)

        if override_exp is not None:
            selected: List[dict] = []
            for row in chain:
                if not isinstance(row, dict):
                    continue
                sym_root = self._normalize_root(self.cfg.symbol)
                under_root = self._normalize_root(self.cfg.underlying)
                valid_roots = {r for r in (sym_root, under_root) if r}
                root = self._normalize_root(str(row.get("symbol_root") or ""))
                if not root:
                    sym_raw = str(row.get("symbol") or "").upper()
                    sym = sym_raw.split(":", 1)[-1]
                    if sym.startswith(sym_root):
                        root = sym_root
                    elif under_root and sym.startswith(under_root):
                        root = under_root
                    else:
                        root = ""
                if root not in valid_roots:
                    continue

                exp = self._row_expiry_date(row)
                if exp is None:
                    continue
                if exp == override_exp:
                    selected.append(row)

            if selected:
                # Log usage only once here as well (though it might have been logged in _decide_entries).
                return selected

            print(
                f"[ENTRY] MSTOCK_TARGET_EXPIRY={raw_override!r} did not match any {self.cfg.symbol.upper()} contracts "
                "in weekly filter; falling back to nearest weekly."
            )

        # Determine the weekly expiry weekday from the nearest available expiry.
        expiries: List[date] = []
        for row in chain:
            if not isinstance(row, dict):
                continue
            sym_root = self.cfg.symbol.upper()
            under_root = str(getattr(self.cfg, "underlying", "") or "").strip().upper()
            valid_roots = {r for r in (sym_root, under_root) if r}
            root = str(row.get("symbol_root") or "").strip().upper()
            if not root:
                sym_raw = str(row.get("symbol") or "").upper()
                sym = sym_raw.split(":", 1)[-1]
                if sym.startswith(sym_root):
                    root = sym_root
                elif under_root and sym.startswith(under_root):
                    root = under_root
                else:
                    root = ""
            if root not in valid_roots:
                continue

            exp = self._row_expiry_date(row)
            if exp is None or exp < today:
                continue
            expiries.append(exp)

        if not expiries:
            return []

        weekly_wd = min(expiries).weekday()

        buckets: dict[date, List[dict]] = {}
        for row in chain:
            if not isinstance(row, dict):
                continue
            sym_root = self.cfg.symbol.upper()
            under_root = str(getattr(self.cfg, "underlying", "") or "").strip().upper()
            valid_roots = {r for r in (sym_root, under_root) if r}
            # Prefer explicit symbol root (CSV path), fallback to symbol prefix.
            root = str(row.get("symbol_root") or "").strip().upper()
            if not root:
                sym_raw = str(row.get("symbol") or "").upper()
                sym = sym_raw.split(":", 1)[-1]
                if sym.startswith(sym_root):
                    root = sym_root
                elif under_root and sym.startswith(under_root):
                    root = under_root
                else:
                    root = ""
            # Avoid other indices.
            if root not in valid_roots:
                continue

            exp = self._row_expiry_date(row)
            if exp is None:
                continue
            if exp < today:
                continue
            # Weekly options: infer weekday from the nearest expiry in this chain.
            if exp.weekday() != weekly_wd:
                continue
            # Weekly only: exclude monthly expiry (last occurrence of that weekday in the month).
            if self._is_last_weekday_of_month(exp):
                continue
            buckets.setdefault(exp, []).append(row)

        if not buckets:
            return []

        nearest = min(buckets.keys())
        return buckets.get(nearest, [])

    def _pick_nearest_strike(
        self,
        chain: List[dict],
        option_type: str,
        target_strike: float,
    ) -> Optional[dict]:
        def norm_type(v: object) -> str:
            s = str(v or "").strip().upper()
            if s in {"CALL", "C"}:
                return "CE"
            if s in {"PUT", "P"}:
                return "PE"
            return s

        want = norm_type(option_type)
        candidates = [
            opt
            for opt in chain
            if norm_type(opt.get("option_type", "")) == want
            and opt.get("strike") is not None
        ]
        if not candidates:
            return None

        # Filter out negative/extreme theta if enabled
        if getattr(self.cfg, "enable_theta_decay_filter", False):
            try:
                from greeks import bs_theta
                # Using a generic 1-day (1/365) time to expiry for rough theta if not available
                filtered = []
                for c in candidates:
                    iv = float(c.get("iv") or 0.2)
                    strike = float(c.get("strike"))
                    # If we don't have exact spot, we approximate spot as target_strike for the theta calc
                    # It's an approximation to avoid plumbing spot through every signature
                    t = bs_theta(S=target_strike, K=strike, T=1.0/365.0, r=0.06, sigma=iv, option_type=want.lower())
                    if t > -10.0:  # arbitrary daily threshold, but we'll accept
                        filtered.append(c)
                if filtered:
                    candidates = filtered
            except Exception:
                pass
                
        # Delta Strike Selection override
        if getattr(self.cfg, "enable_delta_strike_selection", False):
            try:
                from greeks import delta
                target_del = float(getattr(self.cfg, "target_delta", 0.40))
                best_c = None
                min_diff = 999.0
                for c in candidates:
                    iv = float(c.get("iv") or 0.2)
                    strike = float(c.get("strike"))
                    # approximate spot as target_strike since it's the closest reference point in this context
                    d = delta(S=target_strike, K=strike, T=1.0/365.0, r=0.06, sigma=iv, option_type=want.lower())
                    diff = abs(abs(d) - target_del)
                    if diff < min_diff:
                        min_diff = diff
                        best_c = c
                if best_c is not None:
                    return best_c
            except Exception:
                pass

        ctx = self._auto_gpt_strike_context if isinstance(self._auto_gpt_strike_context, dict) else {}
        if bool(ctx.get("active")):
            theta_target = ctx.get("theta_target")
            spot = ctx.get("spot")
            try:
                theta_target_f = float(theta_target) if theta_target is not None else None
            except Exception:
                theta_target_f = None
            try:
                spot_f = float(spot) if spot is not None else None
            except Exception:
                spot_f = None
            if theta_target_f is not None and spot_f is not None and spot_f > 0:
                try:
                    step = float(ctx.get("strike_step") or 50.0)
                except Exception:
                    step = 50.0
                if step <= 0:
                    step = 50.0

                def _score(row: dict) -> float:
                    try:
                        strike_gap = abs(float(row.get("strike")) - float(target_strike))
                    except Exception:
                        return float("inf")
                    theta_val = self._estimate_option_theta_from_chain_row(row, float(spot_f))
                    if theta_val is None:
                        return strike_gap + (3.0 * step)
                    return float(strike_gap + abs(float(theta_val) - float(theta_target_f)) * step)

                try:
                    return min(candidates, key=_score)
                except Exception:
                    pass

        return min(candidates, key=lambda o: abs(float(o["strike"]) - target_strike))

    def _clear_auto_gpt_strike_context(self) -> None:
        self._auto_gpt_strike_context = {}

    def _infer_chain_strike_step(self, chain: List[dict]) -> float:
        strikes: List[float] = []
        for row in chain or []:
            if not isinstance(row, dict):
                continue
            try:
                k = float(row.get("strike"))
            except Exception:
                continue
            if k > 0:
                strikes.append(k)

        uniq = sorted(set(strikes))
        if len(uniq) < 2:
            return 50.0
        diffs = [b - a for a, b in zip(uniq, uniq[1:]) if (b - a) > 0]
        if not diffs:
            return 50.0
        return float(min(diffs))

    def _resolve_auto_gpt_params_for_strategy(self, strategy_name: str) -> Dict[str, float]:
        params_map = self._auto_gpt_strategy_parameters if isinstance(self._auto_gpt_strategy_parameters, dict) else {}
        key = str(strategy_name or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = [key, self._normalize_strategy_name(key)]
        if key in {"directional", "long_call", "long_put", "short_call", "short_put"}:
            aliases.append("directional")

        for alias in aliases:
            row = params_map.get(alias)
            if not isinstance(row, dict):
                continue
            out: Dict[str, float] = {}
            for k, v in row.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    continue
            if out:
                return out
        return {}

    def _set_auto_gpt_strike_context(
        self,
        *,
        selected_strategy: str,
        chain: List[dict],
        spot: float,
    ) -> None:
        params = self._resolve_auto_gpt_params_for_strategy(str(selected_strategy))
        strike_step = self._infer_chain_strike_step(chain)

        distance_override: Optional[float] = None
        try:
            raw_d = params.get("distance")
            if raw_d is not None:
                d = float(raw_d)
                if d > 0:
                    distance_override = d
        except Exception:
            distance_override = None

        theta_target: Optional[float] = None
        try:
            raw_t = params.get("theta")
            if raw_t is not None:
                theta_target = float(raw_t)
        except Exception:
            theta_target = None

        directional_offset_override: Optional[float] = None
        if self._auto_gpt_directional_steps is not None:
            try:
                st = max(0, int(self._auto_gpt_directional_steps))
                if st > 0:
                    directional_offset_override = float(st) * float(strike_step)
            except Exception:
                directional_offset_override = None
        if directional_offset_override is None and distance_override is not None:
            directional_offset_override = float(distance_override)

        # P5: GPT strike selection enhancement
        gpt_strike_result = self._gpt_strike_selection(
            chain=list(chain),
            spot=float(spot),
            strategy=str(selected_strategy),
        )
        if isinstance(gpt_strike_result, dict):
            gpt_suggested = gpt_strike_result.get("suggested_strike")
            gpt_wing = gpt_strike_result.get("wing_width")
            gpt_selected = bool(gpt_strike_result.get("gpt_selected", False))
        else:
            gpt_suggested = None
            gpt_wing = None
            gpt_selected = False

        self._auto_gpt_strike_context = {
            "active": True,
            "strategy": str(selected_strategy),
            "distance_override": distance_override,
            "directional_offset_override": directional_offset_override,
            "theta_target": theta_target,
            "strike_step": float(strike_step),
            "spot": float(spot),
            "gpt_selected": bool(gpt_selected),
            "gpt_suggested_strike": float(gpt_suggested) if gpt_suggested is not None else None,
            "gpt_wing_width": float(gpt_wing) if gpt_wing is not None else None,
        }

    def _estimate_option_theta_from_chain_row(self, row: dict, spot: float) -> Optional[float]:
        if not isinstance(row, dict):
            return None
        try:
            s = float(spot)
        except Exception:
            return None
        if s <= 0:
            return None

        try:
            strike = float(row.get("strike"))
        except Exception:
            return None
        if strike <= 0:
            return None

        ot = str(row.get("option_type") or "").strip().upper()
        if ot in {"CALL", "C"}:
            ot = "CE"
        elif ot in {"PUT", "P"}:
            ot = "PE"
        if ot not in {"CE", "PE"}:
            return None

        exp = self._row_expiry_date(row)
        if exp is None:
            return None
        try:
            exp_dt = dt_datetime(int(exp.year), int(exp.month), int(exp.day), 15, 30, 0)
            t_sec = float((exp_dt - dt_datetime.now()).total_seconds())
        except Exception:
            return None
        if t_sec <= 60:
            return None
        t_years = t_sec / float(365.0 * 24.0 * 3600.0)

        iv_val: Optional[float] = None
        try:
            iv_raw = row.get("iv")
            if iv_raw is not None:
                iv_guess = float(iv_raw)
                if iv_guess > 3.0:
                    iv_guess = iv_guess / 100.0
                if 0.0001 < iv_guess < 10.0:
                    iv_val = float(iv_guess)
        except Exception:
            iv_val = None

        if iv_val is None:
            try:
                px_raw = row.get("ltp")
                if px_raw is None:
                    px_raw = row.get("price")
                px = float(px_raw) if px_raw is not None else None
                if px is not None and px > 0:
                    if px > (1.5 * s) and px > 1000.0:
                        px = px / 100.0
                    iv_sol = implied_volatility(px, s, strike, t_years, 0.0, ot)  # type: ignore[arg-type]
                    if iv_sol is not None and float(iv_sol) > 0:
                        iv_val = float(iv_sol)
                        row["iv"] = iv_val
            except Exception:
                iv_val = None

        if iv_val is None or iv_val <= 0:
            return None

        try:
            return float(bs_theta(s, strike, t_years, 0.0, float(iv_val), ot) / 365.0)  # type: ignore[arg-type]
        except Exception:
            return None

    def _leg_order(
        self,
        opt: dict,
        side: str,
        quantity: int,
    ) -> Order:
        symbol = str(opt.get("symbol") or "")
        token = str(opt.get("token") or "").strip() or None
        exchange = str(opt.get("exchange") or "").strip() or None
        if not exchange and str(symbol).strip().upper().endswith(("CE", "PE")):
            exchange = "NFO"
        elif str(exchange or "").strip().upper() in {"NSEFO", "NFO"}:
            exchange = "NFO"
        if not symbol:
            raise RuntimeError("Option leg missing symbol; adjust option chain mapping.")
        bracket = self._build_bracket_levels(
            entry_price=opt.get("entry_price"),
            side=side,
            quantity=quantity,
        )
        return self._place_order_with_retry(
            symbol=symbol,
            side=side,
            quantity=quantity,
            exchange=exchange,
            symbol_token=token,
            stoploss=bracket.get("stoploss"),
            target_price=bracket.get("targetPrice"),
            trailing_stop_loss=bracket.get("trailingStopLoss"),
        )

    def _place_multi_leg_transactional(
        self,
        legs_to_place: List[Tuple[dict, str, int]],
        trade_name: str,
        raw_legs: List[dict],
        meta: Dict[str, Any],
    ) -> None:
        if not self.cfg.enable_live_trading:
            print(f"[PAPER] Enter {trade_name}: " + ", ".join([f"{side} {leg.get('symbol')}" for leg, side, _ in legs_to_place]))
            self._record_multi_trade_with_meta(trade_name, raw_legs, meta)
            return

        filled = []
        try:
            for leg_dict, side, qty in legs_to_place:
                self._leg_order(leg_dict, side, qty)
                filled.append((leg_dict, side, qty))
            self._record_multi_trade_with_meta(trade_name, raw_legs, meta)
        except Exception as exc:
            print(f"[FATAL] Multi-leg placement failed for {trade_name} due to: {exc}")
            print("[FAIL-SAFE] Triggering rollback execution to flatten filled legs...")
            for leg_dict, original_side, qty in reversed(filled):
                rollback_side = "SELL" if original_side == "BUY" else "BUY"
                try:
                    symbol = str(leg_dict.get("symbol") or "")
                    exchange = str(leg_dict.get("exchange") or "").strip() or None
                    if not exchange and str(symbol).strip().upper().endswith(("CE", "PE")):
                        exchange = "NFO"
                    elif str(exchange or "").strip().upper() in {"NSEFO", "NFO"}:
                        exchange = "NFO"
                    token = str(leg_dict.get("token") or "").strip() or None
                    print(f"[FAIL-SAFE] EMERGENCY UNWIND: placing {rollback_side} market order for {symbol}")
                    self.client.place_order(
                        symbol=symbol,
                        side=rollback_side,
                        quantity=qty,
                        exchange=exchange,
                        symbol_token=token,
                        order_type="MARKET",
                    )
                except Exception as unwind_exc:
                    print(f"[FATAL WARNING] Failed to unwind {leg_dict.get('symbol')} during rollback: {unwind_exc}")
            raise RuntimeError(f"Multi-leg transaction failed: {exc}")

    def _apply_delta_hedge_scope(self, trade: Dict[str, object], *, position_type: str) -> Dict[str, object]:
        """Enable/prepare delta-hedging meta based on cfg.delta_hedge_scope.

        This allows turning on dynamic delta hedging for strategies like `auto`
        without requiring every entry path to explicitly set meta["delta_hedge"].

        Respects an explicit opt-out: if meta contains a falsey `delta_hedge`,
        do not auto-enable it.
        """

        try:
            scope = str(getattr(self.cfg, "delta_hedge_scope", "strategy_only") or "strategy_only").strip().lower()
        except Exception:
            scope = "strategy_only"

        if scope not in {"multi_only", "all_options"}:
            return trade.get("meta") if isinstance(trade.get("meta"), dict) else {}

        pt = str(position_type or "").strip().lower()
        if scope == "multi_only" and pt != "multi":
            return trade.get("meta") if isinstance(trade.get("meta"), dict) else {}

        if pt not in {"multi", "directional"}:
            return trade.get("meta") if isinstance(trade.get("meta"), dict) else {}

        meta = trade.get("meta") if isinstance(trade.get("meta"), dict) else {}
        if not isinstance(meta, dict):
            meta = {}

        # Respect explicit disable.
        if "delta_hedge" in meta and not bool(meta.get("delta_hedge")):
            trade["meta"] = meta
            return meta

        meta["delta_hedge"] = True

        legs = trade.get("legs") if isinstance(trade.get("legs"), list) else []
        inferred_root = self._infer_underlying_root_from_legs(legs) if isinstance(legs, list) else None
        if inferred_root and "delta_hedge_underlying" not in meta:
            meta["delta_hedge_underlying"] = str(inferred_root)

        # Provide resolved symbol/exchange defaults, but do not override per-trade config.
        resolved_sym = ""
        resolved_exch = ""
        if "delta_hedge_symbol" not in meta or "delta_hedge_exchange" not in meta:
            resolved_sym, resolved_exch, _root = self._pick_delta_hedge_symbol_for_trade({"legs": legs, "meta": meta})
        if "delta_hedge_symbol" not in meta:
            sym = str(resolved_sym or "").strip()
            if sym:
                meta["delta_hedge_symbol"] = sym
        if "delta_hedge_exchange" not in meta:
            exch = str(resolved_exch or "").strip()
            if exch:
                meta["delta_hedge_exchange"] = exch

        trade["meta"] = meta
        return meta

    def _record_multi_trade(self, name: str, legs: List[dict]) -> None:
        self._record_multi_trade_with_meta(name=name, legs=legs, meta=None)

    def _record_multi_trade_with_meta(self, name: str, legs: List[dict], meta: Optional[Dict[str, object]]) -> None:
        trade_id = self._new_trade_id("M")
        effective_meta: Dict[str, object] = dict(meta or {})
        effective_meta.update(self._trade_risk_meta_overrides())
        legs = [
            self._annotate_leg_brackets(
                dict(lg),
                risk_overrides={
                    "stop_loss_pct": effective_meta.get("ml_dynamic_stop_loss_pct"),
                    "target_pct": effective_meta.get("ml_dynamic_target_pct"),
                    "trailing_sl_pct": effective_meta.get("ml_dynamic_trailing_sl_pct"),
                },
            )
            for lg in legs
            if isinstance(lg, dict)
        ]
        payload: Dict[str, object] = {
            "trade_id": trade_id,
            "name": name,
            "legs": legs,
            "opened_ts": time.time(),
        }

        # Ensure delta-hedge scope can apply to multi trades (e.g. strategy=`auto`).
        if effective_meta:
            payload["meta"] = effective_meta
        self._apply_delta_hedge_scope(payload, position_type="multi")

        # ---- Pyramiding for multi-leg trades ----
        # If pyramiding is enabled and an open multi-leg trade with the same
        # name exists, increment leg quantities instead of creating a new trade.
        appended = False
        if int(getattr(self.cfg, "max_pyramid_levels", 0) or 0) > 0:
            pyramid_max = self._dynamic_pyramid_max_level()
            existing = None
            for t in self.state.open_multi:
                if t.get("name") == name:
                    t_legs = t.get("legs") or []
                    if isinstance(t_legs, list) and len(t_legs) == len(legs):
                        t_syms = sorted([str(lg.get("symbol") or "").strip().upper() for lg in t_legs if isinstance(lg, dict)])
                        new_syms = sorted([str(lg.get("symbol") or "").strip().upper() for lg in legs if isinstance(lg, dict)])
                        if t_syms == new_syms:
                            existing = t
                            break
            if existing is not None:
                try:
                    lev = int(existing.get("pyramid_level", 0) or 0)
                except Exception:
                    lev = 0
                if lev < pyramid_max:
                    # Increment quantities on matching legs by symbol.
                    existing_legs = existing.get("legs")
                    if isinstance(existing_legs, list):
                        for new_leg in legs:
                            if not isinstance(new_leg, dict):
                                continue
                            new_symbol = str(new_leg.get("symbol") or "").strip().upper()
                            new_qty = 0
                            try:
                                new_qty = abs(int(new_leg.get("quantity") or 0))
                            except Exception:
                                pass
                            if new_qty <= 0 or not new_symbol:
                                continue
                            # Match by trading symbol (most reliable for multi-leg).
                            matched = False
                            for e_leg in existing_legs:
                                if not isinstance(e_leg, dict):
                                    continue
                                e_sym = str(e_leg.get("symbol") or "").strip().upper()
                                if e_sym == new_symbol:
                                    try:
                                        old_qty = int(abs(float(e_leg.get("quantity") or 0)))
                                    except Exception:
                                        old_qty = 0
                                    e_leg["quantity"] = old_qty + new_qty
                                    matched = True
                                    break
                            if not matched:
                                # Leg with this symbol doesn't exist yet; add it.
                                existing_legs.append(dict(new_leg))
                    else:
                        existing["legs"] = [dict(l) for l in legs if isinstance(l, dict)]

                    lev += 1
                    existing["pyramid_level"] = lev
                    appended = True

                    if self._has_event_sink():
                        self._emit(
                            TradeLogEvent(
                                ts=time.time(),
                                event="UPDATE",
                                trade_id=str(existing.get("trade_id") or trade_id),
                                position_type="multi",
                                name=name,
                                legs=[dict(l) for l in existing.get("legs") or [] if isinstance(l, dict)],
                                mtm=self._compute_legs_mtm(existing.get("legs") or []),
                            )
                        )

        if not appended:
            try:
                open_count_now = len(self.state.open_directional) + len(self.state.open_multi)
                max_open_positions = int(getattr(self.cfg, "max_open_positions", 0) or 0)
            except Exception:
                open_count_now = 0
                max_open_positions = 0
            if max_open_positions > 0 and open_count_now >= max_open_positions:
                print(
                    f"[RISK] Max open positions reached ({open_count_now}/{max_open_positions}); "
                    f"blocking new multi entry {name}"
                )
                return None
            self.state.open_multi.append(payload)
            self._note_opened_trade_type(position_type="multi", name=str(name))

        if not appended and self._has_event_sink():
            self._emit(
                TradeLogEvent(
                    ts=time.time(),
                    event="OPEN",
                    trade_id=trade_id,
                    position_type="multi",
                    name=name,
                    legs=[dict(l) for l in legs if isinstance(l, dict)],
                    mtm=self._compute_legs_mtm(legs),
                )
            )

    def _safe_exit_all_positions(self, *, reason: str = "KILL_SWITCH") -> None:
        """Exit every open position (directional + multi-leg) immediately.

        Used by the SCALPER_KILL_SWITCH env-var to achieve a clean, safe
        stopout of all positions when the operator needs an instant halt.
        Does NOT send live orders in paper mode.
        """
        try:
            self._ensure_daily_risk_counters_day()
        except Exception:
            pass

        # Close multi-leg trades first.
        for tr in list(self.state.open_multi or []):
            if not isinstance(tr, dict):
                continue
            try:
                self._close_multi_trade(tr, reason=reason)
            except Exception as exc:
                print(f"[_safe_exit_all] error closing multi trade {tr.get('trade_id')}: {exc}")

        # Close directional trades.
        for tr in list(self.state.open_directional or []):
            if not isinstance(tr, dict):
                continue
            try:
                self._close_directional_trade(tr, reason=reason)
            except Exception as exc:
                print(f"[_safe_exit_all] error closing directional trade {tr.get('trade_id')}: {exc}")

        self.state.open_multi = []
        self.state.open_directional = []
        self.state.open_orders = []
        print(f"[_safe_exit_all] All positions exited. reason={reason}")

    def _mark_stopout(self) -> None:
        self._ensure_daily_risk_counters_day()
        self.state.stopouts_today += 1
        self.state.consecutive_stopouts += 1
        self.state.last_stopout_ts = time.time()
        self.state.consecutive_wins = 0

    def _mark_non_stop_exit(self) -> None:
        self._ensure_daily_risk_counters_day()
        # Reset only the consecutive counter; daily stopouts remains.
        self.state.consecutive_stopouts = 0

    def _close_multi_trade(self, trade: Dict[str, object], *, reason: str = "") -> float:
        legs = trade.get("legs")
        if not isinstance(legs, list):
            return 0.0

        # Apply exit liquidity filter before closing any leg.
        # Block the entire exit if any leg fails the spread/bid-ask guard.
        ok, filter_reason = self._check_exit_liquidity(legs)
        if not ok:
            print(f"[PAPER][BLOCKED] Exit {trade.get('name', '?')}: liquidity filter rejected — {filter_reason}")
            return 0.0

        realized = 0.0

        def opposite(side: str) -> str:
            return "BUY" if side.upper() == "SELL" else "SELL"

        for leg in legs:
            if not isinstance(leg, dict):
                continue
            symbol = str(leg.get("symbol") or "")
            token = str(leg.get("token") or "").strip() or None
            exchange = str(leg.get("exchange") or "").strip() or None
            side = str(leg.get("side") or "")
            qty = int(leg.get("quantity") or 0)
            if not symbol or not side or qty <= 0:
                continue

            if not self.cfg.enable_live_trading:
                # Paper mode: use bid/ask for realistic exit pricing.
                # - Closing BUY leg (we SELL) → use bid (what buyer pays)
                # - Closing SELL leg (we BUY)  → use ask (what seller asks)
                paper_exit_price: Optional[float] = None
                paper_spread_deduct: float = 0.0
                leg_side = side.upper()
                close_action = opposite(side)  # our closing action
                if close_action == "SELL":
                    # Closing a BUY (long) position → we sell → receive bid
                    paper_exit_price, paper_spread_deduct = self._get_paper_exit_price(
                        symbol, exchange, leg_side, qty, is_entry_side_buy=True
                    )
                else:
                    # Closing a SELL (short) position → we buy → pay ask
                    paper_exit_price, paper_spread_deduct = self._get_paper_exit_price(
                        symbol, exchange, leg_side, qty, is_entry_side_buy=False
                    )
                if paper_exit_price is None or paper_exit_price <= 0.0:
                    print(f"[PAPER][BLOCKED] Close leg {symbol}: exit price unavailable — bid/ask required")
                    continue
                print(f"[PAPER] Close leg: {close_action} {symbol} x{qty} @ {paper_exit_price:.2f} ({'bid' if close_action == 'SELL' else 'ask'})")
                entry = leg.get("entry_price")
                try:
                    entry_f = float(entry) if entry is not None else None
                except Exception:
                    entry_f = None
                cur = paper_exit_price
                # Store exit price for UI display.
                leg["exit_price"] = cur
                leg["execution_price_source"] = "bid" if close_action == "SELL" else "ask"
                if entry_f is not None and cur is not None and side.upper() in {"BUY", "SELL"}:
                    sign = 1.0 if side.upper() == "BUY" else -1.0
                    gross_pnl = (float(cur) - float(entry_f)) * sign * float(qty)
                    # Spread cost already embedded in bid/ask execution price.
                    # Deduct only slippage + brokerage via paper cost model.
                    try:
                        from cost_model import estimate_paper_execution_costs
                        _exit_costs = estimate_paper_execution_costs(
                            execution_price=float(cur),
                            quantity=int(qty),
                            slippage_pct=float(getattr(self.cfg, "paper_slippage_pct", 0.001) or 0.001),
                            extra_market_impact_pct=float(getattr(self.cfg, "paper_extra_market_impact_pct", 0.0) or 0.0),
                            apply_brokerage=bool(getattr(self.cfg, "paper_apply_brokerage_costs", True)),
                            cost_model_source=str(getattr(self.cfg, "paper_cost_model_source", "cost_model_assumptions") or "cost_model_assumptions"),
                            side=close_action,
                        )
                        realized += gross_pnl - float(_exit_costs.get("slippage_cost", 0.0)) - float(_exit_costs.get("total_brokerage_charges", 0.0))
                    except Exception:
                        realized += gross_pnl
                continue

            try:
                self.client.place_order(
                    symbol=symbol,
                    side=opposite(side),
                    quantity=qty,
                    exchange=exchange,
                    symbol_token=token,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to close leg {symbol}: {exc}")
            # Track P&L in live mode too (for max_daily_loss enforcement)
            entry = leg.get("entry_price")
            try:
                entry_f = float(entry) if entry is not None else None
            except Exception:
                entry_f = None
            cur = self._try_get_ltp_for_leg(leg)
            # Store exit price for UI display.
            # Note: In live mode, this `leg` dict is a copy, so this won't update the stored trade.
            # The UI will fetch live prices for MTM.
            leg["exit_price"] = cur
            if entry_f is not None and cur is not None and side.upper() in {"BUY", "SELL"}:
                sign = 1.0 if side.upper() == "BUY" else -1.0
                realized += (float(cur) - float(entry_f)) * sign * float(qty)

        # Track realized P&L for BOTH paper and live modes (max_daily_loss enforcement)
        self.state.realized_pnl += float(realized)
        self.state.last_exit_ts = time.time()
        self._note_trade_result(realized)
        self._strategy_winrate_record_result(str(trade.get("name") or ""), float(realized) >= 0)

        if self._has_event_sink():
            trade_id = str(trade.get("trade_id") or "")
            name = str(trade.get("name") or "")
            self._emit(
                TradeLogEvent(
                    ts=time.time(),
                    event="CLOSE",
                    trade_id=trade_id or "(unknown)",
                    position_type="multi",
                    name=name,
                    legs=[dict(l) for l in legs if isinstance(l, dict)],
                    realized=float(realized),
                    mtm=None,
                    reason=reason or None,
                )
            )

        return float(realized)

    def _enter_long_straddle(
        self,
        chain: List[dict],
        spot: float,
        atr_val: Optional[float] = None,
        allow_alternative_retry: bool = True,
    ) -> None:
        call = self._pick_nearest_strike(chain, "CE", spot)
        put = self._pick_nearest_strike(chain, "PE", spot)
        if not call or not put:
            print("Long straddle: could not find ATM CE/PE")
            return
        qty = self._entry_qty(atr_val)
        if qty <= 0:
            return

        def f(v: object) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        call_strike = f(call.get("strike"))
        put_strike = f(put.get("strike"))

        legs = [
            {
                "symbol": call.get("symbol"),
                "token": call.get("token"),
                "exchange": call.get("exchange"),
                "strike": call_strike,
                "option_type": "CE",
                "expiry": call.get("expiry"),
                "iv": call.get("iv"),
                "side": "BUY",
                "quantity": qty,
            },
            {
                "symbol": put.get("symbol"),
                "token": put.get("token"),
                "exchange": put.get("exchange"),
                "strike": put_strike,
                "option_type": "PE",
                "expiry": put.get("expiry"),
                "iv": put.get("iv"),
                "side": "BUY",
                "quantity": qty,
            },
        ]

        # Skip premium/liquidity checks if GPT gate forces TAKE
        if not self._should_gpt_override_entry_checks({"position_type": "multi"}):
            ok, reason = self._check_entry_leg_premiums(legs)
            if not ok:
                print(f"[ENTRY BLOCKED] {reason}")
                return

        meta: Dict[str, object] = {
            "entry_spot": float(spot),
            "atr": float(atr_val) if atr_val is not None else None,
            "long_call_strike": call_strike,
            "long_put_strike": put_strike,
        }

        # Capture entry premium so MTM % exits work for long-premium too.
        self._capture_entry_premium_meta(legs, meta, atr_val=atr_val)

        base_name = "long_straddle"
        trade_name = (
            f"{base_name} (weekly)" if bool(getattr(self.cfg, "nifty_weekly_only", False)) else base_name
        )

        if not self._can_open_or_reroute_multi_trade(
            base_name=base_name,
            trade_name=str(trade_name),
            chain=chain,
            spot=float(spot),
            atr_val=atr_val,
            allow_alternative_retry=allow_alternative_retry,
        ):
            return

        # Optional GPT entry gate (before placing any orders / recording paper trades).
        try:
            is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
        except Exception:
            is_paper = True
        if not self._gpt_entry_gate(
            proposal={
                "position_type": "multi",
                "trade_name": str(trade_name),
                "strategy_mode": str(getattr(self.cfg, "strategy_name", "") or ""),
                "timeframe": str(getattr(self.cfg, "timeframe", "") or ""),
                "spot": float(spot),
                "atr": float(atr_val) if atr_val is not None else None,
                "legs": [dict(l) for l in legs if isinstance(l, dict)],
                "meta": dict(meta),
            },
            is_paper=bool(is_paper),
        ):
            return

        self._place_multi_leg_transactional([(call, "BUY", qty), (put, "BUY", qty)], trade_name, legs, meta)

    def _get_dynamic_strike_distance(self, atr_val: Optional[float]) -> float:
        """Calculate dynamic strike distance based on ATR and configured multipliers.
        
        Returns base distance if dynamic strikes disabled or ATR unavailable.
        """
        base_distance = float(self.cfg.short_strike_distance)

        ctx = self._auto_gpt_strike_context if isinstance(self._auto_gpt_strike_context, dict) else {}
        if bool(ctx.get("active")):
            try:
                d_ovr = ctx.get("distance_override")
                if d_ovr is not None and float(d_ovr) > 0:
                    return float(d_ovr)
            except Exception:
                pass
            # AUTO mode: do not use ATR-based strike-distance adaptation.
            return float(base_distance)
        
        if not bool(getattr(self.cfg, "enable_dynamic_strikes", True)):
            return base_distance
        
        if atr_val is None or atr_val <= 0:
            return base_distance
        
        atr = float(atr_val)
        threshold_low = float(getattr(self.cfg, "strike_atr_threshold_low", 30.0))
        threshold_high = float(getattr(self.cfg, "strike_atr_threshold_high", 60.0))
        
        # Determine multiplier based on ATR
        if atr < threshold_low:
            multiplier = float(getattr(self.cfg, "strike_atr_multiplier_low", 0.8))
        elif atr <= threshold_high:
            multiplier = float(getattr(self.cfg, "strike_atr_multiplier_mid", 1.0))
        else:
            multiplier = float(getattr(self.cfg, "strike_atr_multiplier_high", 1.5))
        
        dynamic_distance = base_distance * multiplier
        
        # Log for debugging
        if bool(getattr(self.cfg, "debug_log_no_signal", False)):
            print(f"[DYNAMIC STRIKES] ATR={atr:.2f} Base={base_distance:.0f} Mult={multiplier:.2f} Distance={dynamic_distance:.0f}")
        
        return dynamic_distance

    def _get_directional_strike_offset(self, atr_val: Optional[float]) -> float:
        """Calculate dynamic strike offset for directional trades based on ATR.
        
        Returns 0 (ATM) if dynamic strikes disabled or ATR unavailable.
        
        For calls (bullish trades):
        - Positive offset = OTM calls (strike > spot)
        - Negative offset = ITM calls (strike < spot)
        
        For puts (bearish trades):
        - Positive offset = ITM puts (strike > spot) 
        - Negative offset = OTM puts (strike < spot)
        
        ITM options are used in extreme volatility conditions (ATR > threshold or ATR < threshold)
        to get higher delta exposure at potentially lower cost.
        """
        ctx = self._auto_gpt_strike_context if isinstance(self._auto_gpt_strike_context, dict) else {}
        if bool(ctx.get("active")):
            try:
                off_ovr = ctx.get("directional_offset_override")
                if off_ovr is not None and float(off_ovr) > 0:
                    return float(off_ovr)
            except Exception:
                pass
            try:
                d_ovr = ctx.get("distance_override")
                if d_ovr is not None and float(d_ovr) > 0:
                    return float(d_ovr)
            except Exception:
                pass
            return float(getattr(self.cfg, "short_strike_distance", 50.0) or 50.0)

        if not bool(getattr(self.cfg, "enable_directional_dynamic_strikes", True)):
            return 0.0
        
        if atr_val is None or atr_val <= 0:
            return 0.0
        
        atr = float(atr_val)
        threshold_low = float(getattr(self.cfg, "dir_strike_atr_threshold_low", 30.0))
        threshold_high = float(getattr(self.cfg, "dir_strike_atr_threshold_high", 60.0))
        use_itm_above = float(getattr(self.cfg, "dir_use_itm_above_atr", 80.0))
        use_itm_below = float(getattr(self.cfg, "dir_use_itm_below_atr", 20.0))
        
        # Check if ITM options should be used
        use_itm = False
        if atr > use_itm_above or atr < use_itm_below:
            use_itm = True
        
        if use_itm:
            # Use ITM multipliers (negative values)
            if atr < threshold_low:
                multiplier = float(getattr(self.cfg, "dir_strike_itm_multiplier_low", -0.5))
            elif atr <= threshold_high:
                multiplier = float(getattr(self.cfg, "dir_strike_itm_multiplier_mid", -1.0))
            else:
                multiplier = float(getattr(self.cfg, "dir_strike_itm_multiplier_high", -1.5))
        else:
            # Use OTM multipliers (positive values)
            if atr < threshold_low:
                multiplier = float(getattr(self.cfg, "dir_strike_atr_multiplier_low", 0.5))
            elif atr <= threshold_high:
                multiplier = float(getattr(self.cfg, "dir_strike_atr_multiplier_mid", 1.0))
            else:
                multiplier = float(getattr(self.cfg, "dir_strike_atr_multiplier_high", 1.5))
        
        # Use ATR as the base for strike offset
        offset = atr * multiplier
        
        # Log for debugging
        if bool(getattr(self.cfg, "debug_log_no_signal", False)):
            option_type = "ITM" if use_itm else "OTM"
            print(f"[DIRECTIONAL DYNAMIC STRIKES] ATR={atr:.2f} Type={option_type} Mult={multiplier:.2f} Offset={offset:.0f}")
        
        return offset

    def _enter_short_straddle(
        self,
        chain: List[dict],
        spot: float,
        atr_val: Optional[float] = None,
        allow_alternative_retry: bool = True,
    ) -> None:
        call = self._pick_nearest_strike(chain, "CE", spot)
        put = self._pick_nearest_strike(chain, "PE", spot)
        if not call or not put:
            print("Straddle: could not find ATM CE/PE")
            return
        qty = self._entry_qty(atr_val)
        if qty <= 0:
            return

        def f(v: object) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        call_strike = f(call.get("strike"))
        put_strike = f(put.get("strike"))

        legs = [
            {
                "symbol": call.get("symbol"),
                "token": call.get("token"),
                "exchange": call.get("exchange"),
                "strike": call_strike,
                "option_type": "CE",
                "expiry": call.get("expiry"),
                "iv": call.get("iv"),
                "side": "SELL",
                "quantity": qty,
            },
            {
                "symbol": put.get("symbol"),
                "token": put.get("token"),
                "exchange": put.get("exchange"),
                "strike": put_strike,
                "option_type": "PE",
                "expiry": put.get("expiry"),
                "iv": put.get("iv"),
                "side": "SELL",
                "quantity": qty,
            },
        ]

        # Skip premium/liquidity checks if GPT gate forces TAKE
        if not self._should_gpt_override_entry_checks({"position_type": "multi"}):
            ok, reason = self._check_entry_leg_premiums(legs)
            if not ok:
                print(f"[ENTRY BLOCKED] {reason}")
                return

        meta: Dict[str, object] = {
            "entry_spot": float(spot),
            "atr": float(atr_val) if atr_val is not None else None,
            "short_call_strike": call_strike,
            "short_put_strike": put_strike,
        }

        # Capture entry premium so MTM % exits work reliably.
        self._capture_entry_premium_meta(legs, meta, atr_val=atr_val)
        meta["premium_selling"] = True

        # Label weekly structures clearly in the UI when configured.
        base_name = "short_straddle"
        trade_name = (
            f"{base_name} (weekly)" if bool(getattr(self.cfg, "nifty_weekly_only", False)) else base_name
        )

        if not self._can_open_or_reroute_multi_trade(
            base_name=base_name,
            trade_name=str(trade_name),
            chain=chain,
            spot=float(spot),
            atr_val=atr_val,
            allow_alternative_retry=allow_alternative_retry,
        ):
            return

        # Optional GPT entry gate (before placing any orders / recording paper trades).
        try:
            is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
        except Exception:
            is_paper = True
        if not self._gpt_entry_gate(
            proposal={
                "position_type": "multi",
                "trade_name": str(trade_name),
                "strategy_mode": str(getattr(self.cfg, "strategy_name", "") or ""),
                "timeframe": str(getattr(self.cfg, "timeframe", "") or ""),
                "spot": float(spot),
                "atr": float(atr_val) if atr_val is not None else None,
                "legs": [dict(l) for l in legs if isinstance(l, dict)],
                "meta": dict(meta),
            },
            is_paper=bool(is_paper),
        ):
            return

        self._place_multi_leg_transactional([(call, "SELL", qty), (put, "SELL", qty)], trade_name, legs, meta)

    def _enter_long_strangle(
        self,
        chain: List[dict],
        spot: float,
        atr_val: Optional[float] = None,
        allow_alternative_retry: bool = True,
    ) -> None:
        d = self._get_dynamic_strike_distance(atr_val)
        call = self._pick_nearest_strike(chain, "CE", spot + d)
        put = self._pick_nearest_strike(chain, "PE", spot - d)
        if not call or not put:
            print("Long strangle: could not find OTM CE/PE")
            return
        qty = self._entry_qty(atr_val)
        if qty <= 0:
            return

        def f(v: object) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        call_strike = f(call.get("strike"))
        put_strike = f(put.get("strike"))

        legs = [
            {
                "symbol": call.get("symbol"),
                "token": call.get("token"),
                "exchange": call.get("exchange"),
                "strike": call_strike,
                "option_type": "CE",
                "expiry": call.get("expiry"),
                "iv": call.get("iv"),
                "side": "BUY",
                "quantity": qty,
            },
            {
                "symbol": put.get("symbol"),
                "token": put.get("token"),
                "exchange": put.get("exchange"),
                "strike": put_strike,
                "option_type": "PE",
                "expiry": put.get("expiry"),
                "iv": put.get("iv"),
                "side": "BUY",
                "quantity": qty,
            },
        ]

        # Skip premium/liquidity checks if GPT gate forces TAKE
        if not self._should_gpt_override_entry_checks({"position_type": "multi"}):
            ok, reason = self._check_entry_leg_premiums(legs)
            if not ok:
                print(f"[ENTRY BLOCKED] {reason}")
                return

        meta: Dict[str, object] = {
            "entry_spot": float(spot),
            "atr": float(atr_val) if atr_val is not None else None,
            "long_call_strike": call_strike,
            "long_put_strike": put_strike,
            "d": float(d),
        }

        self._capture_entry_premium_meta(legs, meta, atr_val=atr_val)

        base_name = "long_strangle"
        trade_name = (
            f"{base_name} (weekly)" if bool(getattr(self.cfg, "nifty_weekly_only", False)) else base_name
        )

        if not self._can_open_or_reroute_multi_trade(
            base_name=base_name,
            trade_name=str(trade_name),
            chain=chain,
            spot=float(spot),
            atr_val=atr_val,
            allow_alternative_retry=allow_alternative_retry,
        ):
            return

        # Optional GPT entry gate (before placing any orders / recording paper trades).
        try:
            is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
        except Exception:
            is_paper = True
        if not self._gpt_entry_gate(
            proposal={
                "position_type": "multi",
                "trade_name": str(trade_name),
                "strategy_mode": str(getattr(self.cfg, "strategy_name", "") or ""),
                "timeframe": str(getattr(self.cfg, "timeframe", "") or ""),
                "spot": float(spot),
                "atr": float(atr_val) if atr_val is not None else None,
                "legs": [dict(l) for l in legs if isinstance(l, dict)],
                "meta": dict(meta),
            },
            is_paper=bool(is_paper),
        ):
            return

        self._place_multi_leg_transactional([(call, "BUY", qty), (put, "BUY", qty)], trade_name, legs, meta)

    def _enter_bull_call_spread(
        self,
        chain: List[dict],
        spot: float,
        atr_val: Optional[float] = None,
        allow_alternative_retry: bool = True,
    ) -> None:
        # Buy ATM call, sell higher-strike call.
        long_call = self._pick_nearest_strike(chain, "CE", spot)
        if not long_call:
            print("Bull call spread: could not find long call")
            return

        def f(v: object) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        lc_strike = f(long_call.get("strike"))
        if lc_strike is None:
            print("Bull call spread: long call strike missing")
            return

        w = float(getattr(self.cfg, "wing_width", 0.0) or 0.0)
        if w <= 0:
            w = float(self._get_dynamic_strike_distance(atr_val))
        short_call = self._pick_nearest_strike(chain, "CE", float(lc_strike) + float(w))
        if not short_call:
            print("Bull call spread: could not find short call")
            return

        sc_strike = f(short_call.get("strike"))
        qty = self._entry_qty(atr_val)
        if qty <= 0:
            return

        legs = [
            {
                "symbol": long_call.get("symbol"),
                "token": long_call.get("token"),
                "exchange": long_call.get("exchange"),
                "strike": lc_strike,
                "option_type": "CE",
                "expiry": long_call.get("expiry"),
                "iv": long_call.get("iv"),
                "side": "BUY",
                "quantity": qty,
            },
            {
                "symbol": short_call.get("symbol"),
                "token": short_call.get("token"),
                "exchange": short_call.get("exchange"),
                "strike": sc_strike,
                "option_type": "CE",
                "expiry": short_call.get("expiry"),
                "iv": short_call.get("iv"),
                "side": "SELL",
                "quantity": qty,
            },
        ]

        # Skip premium/liquidity checks if GPT gate forces TAKE
        if not self._should_gpt_override_entry_checks({"position_type": "multi"}):
            ok, reason = self._check_entry_leg_premiums(legs)
            if not ok:
                print(f"[ENTRY BLOCKED] {reason}")
                return

        meta: Dict[str, object] = {
            "entry_spot": float(spot),
            "atr": float(atr_val) if atr_val is not None else None,
            "long_call_strike": lc_strike,
            "short_call_strike": sc_strike,
            "spread_width": float(w),
        }
        self._capture_entry_premium_meta(legs, meta, atr_val=atr_val)

        base_name = "bull_call_spread"
        trade_name = f"{base_name} (weekly)" if bool(getattr(self.cfg, "nifty_weekly_only", False)) else base_name

        if not self._can_open_or_reroute_multi_trade(
            base_name=base_name,
            trade_name=str(trade_name),
            chain=chain,
            spot=float(spot),
            atr_val=atr_val,
            allow_alternative_retry=allow_alternative_retry,
        ):
            return

        try:
            is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
        except Exception:
            is_paper = True
        if not self._gpt_entry_gate(
            proposal={
                "position_type": "multi",
                "trade_name": str(trade_name),
                "strategy_mode": str(getattr(self.cfg, "strategy_name", "") or ""),
                "timeframe": str(getattr(self.cfg, "timeframe", "") or ""),
                "spot": float(spot),
                "atr": float(atr_val) if atr_val is not None else None,
                "legs": [dict(l) for l in legs if isinstance(l, dict)],
                "meta": dict(meta),
            },
            is_paper=bool(is_paper),
        ):
            return

        self._place_multi_leg_transactional([(long_call, "BUY", qty), (short_call, "SELL", qty)], trade_name, legs, meta)

    def _enter_bull_put_spread(
        self,
        chain: List[dict],
        spot: float,
        atr_val: Optional[float] = None,
        allow_alternative_retry: bool = True,
    ) -> None:
        # Sell OTM put, buy lower-strike put (credit spread).
        d = self._get_dynamic_strike_distance(atr_val)
        w = float(getattr(self.cfg, "wing_width", 0.0) or 0.0)
        if w <= 0:
            w = float(d)

        short_put = self._pick_nearest_strike(chain, "PE", spot - float(d))
        if not short_put:
            print("Bull put spread: could not find short put")
            return

        def f(v: object) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        sp_strike = f(short_put.get("strike"))
        if sp_strike is None:
            print("Bull put spread: short put strike missing")
            return

        long_put = self._pick_nearest_strike(chain, "PE", float(sp_strike) - float(w))
        if not long_put:
            print("Bull put spread: could not find long put")
            return
        lp_strike = f(long_put.get("strike"))

        qty = self._entry_qty(atr_val)
        if qty <= 0:
            return

        legs = [
            {
                "symbol": long_put.get("symbol"),
                "token": long_put.get("token"),
                "exchange": long_put.get("exchange"),
                "strike": lp_strike,
                "option_type": "PE",
                "expiry": long_put.get("expiry"),
                "iv": long_put.get("iv"),
                "side": "BUY",
                "quantity": qty,
            },
            {
                "symbol": short_put.get("symbol"),
                "token": short_put.get("token"),
                "exchange": short_put.get("exchange"),
                "strike": sp_strike,
                "option_type": "PE",
                "expiry": short_put.get("expiry"),
                "iv": short_put.get("iv"),
                "side": "SELL",
                "quantity": qty,
            },
        ]

        # Skip premium/liquidity checks if GPT gate forces TAKE
        if not self._should_gpt_override_entry_checks({"position_type": "multi"}):
            ok, reason = self._check_entry_leg_premiums(legs)
            if not ok:
                print(f"[ENTRY BLOCKED] {reason}")
                return

        meta: Dict[str, object] = {
            "entry_spot": float(spot),
            "atr": float(atr_val) if atr_val is not None else None,
            "short_put_strike": sp_strike,
            "long_put_strike": lp_strike,
            "d": float(d),
            "spread_width": float(w),
            "premium_selling": True,
        }
        self._capture_entry_premium_meta(legs, meta, atr_val=atr_val)

        base_name = "bull_put_spread"
        trade_name = f"{base_name} (weekly)" if bool(getattr(self.cfg, "nifty_weekly_only", False)) else base_name

        if not self._can_open_or_reroute_multi_trade(
            base_name=base_name,
            trade_name=str(trade_name),
            chain=chain,
            spot=float(spot),
            atr_val=atr_val,
            allow_alternative_retry=allow_alternative_retry,
        ):
            return

        try:
            is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
        except Exception:
            is_paper = True
        if not self._gpt_entry_gate(
            proposal={
                "position_type": "multi",
                "trade_name": str(trade_name),
                "strategy_mode": str(getattr(self.cfg, "strategy_name", "") or ""),
                "timeframe": str(getattr(self.cfg, "timeframe", "") or ""),
                "spot": float(spot),
                "atr": float(atr_val) if atr_val is not None else None,
                "legs": [dict(l) for l in legs if isinstance(l, dict)],
                "meta": dict(meta),
            },
            is_paper=bool(is_paper),
        ):
            return

        self._place_multi_leg_transactional([(long_put, "BUY", qty), (short_put, "SELL", qty)], trade_name, legs, meta)

    def _enter_call_ratio_backspread(
        self,
        chain: List[dict],
        spot: float,
        atr_val: Optional[float] = None,
        allow_alternative_retry: bool = True,
    ) -> None:
        # Sell 1x ATM call, buy 2x higher-strike calls.
        d = self._get_dynamic_strike_distance(atr_val)
        short_call = self._pick_nearest_strike(chain, "CE", spot)
        if not short_call:
            print("Call ratio backspread: could not find short call")
            return

        long_call = self._pick_nearest_strike(chain, "CE", spot + float(d))
        if not long_call:
            print("Call ratio backspread: could not find long calls")
            return

        def f(v: object) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        sc_strike = f(short_call.get("strike"))
        lc_strike = f(long_call.get("strike"))

        qty = self._entry_qty(atr_val)
        if qty <= 0:
            return
        qty2 = int(qty) * 2

        legs = [
            {
                "symbol": long_call.get("symbol"),
                "token": long_call.get("token"),
                "exchange": long_call.get("exchange"),
                "strike": lc_strike,
                "option_type": "CE",
                "expiry": long_call.get("expiry"),
                "iv": long_call.get("iv"),
                "side": "BUY",
                "quantity": qty2,
            },
            {
                "symbol": short_call.get("symbol"),
                "token": short_call.get("token"),
                "exchange": short_call.get("exchange"),
                "strike": sc_strike,
                "option_type": "CE",
                "expiry": short_call.get("expiry"),
                "iv": short_call.get("iv"),
                "side": "SELL",
                "quantity": qty,
            },
        ]

        # Skip premium/liquidity checks if GPT gate forces TAKE
        if not self._should_gpt_override_entry_checks({"position_type": "multi"}):
            ok, reason = self._check_entry_leg_premiums(legs)
            if not ok:
                print(f"[ENTRY BLOCKED] {reason}")
                return

        meta: Dict[str, object] = {
            "entry_spot": float(spot),
            "atr": float(atr_val) if atr_val is not None else None,
            "short_call_strike": sc_strike,
            "long_call_strike": lc_strike,
            "d": float(d),
            "ratio_long": 2,
            "ratio_short": 1,
        }
        self._capture_entry_premium_meta(legs, meta, atr_val=atr_val)

        base_name = "call_ratio_backspread"
        trade_name = f"{base_name} (weekly)" if bool(getattr(self.cfg, "nifty_weekly_only", False)) else base_name

        if not self._can_open_or_reroute_multi_trade(
            base_name=base_name,
            trade_name=str(trade_name),
            chain=chain,
            spot=float(spot),
            atr_val=atr_val,
            allow_alternative_retry=allow_alternative_retry,
        ):
            return

        try:
            is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
        except Exception:
            is_paper = True
        if not self._gpt_entry_gate(
            proposal={
                "position_type": "multi",
                "trade_name": str(trade_name),
                "strategy_mode": str(getattr(self.cfg, "strategy_name", "") or ""),
                "timeframe": str(getattr(self.cfg, "timeframe", "") or ""),
                "spot": float(spot),
                "atr": float(atr_val) if atr_val is not None else None,
                "legs": [dict(l) for l in legs if isinstance(l, dict)],
                "meta": dict(meta),
            },
            is_paper=bool(is_paper),
        ):
            return

        self._place_multi_leg_transactional([(long_call, "BUY", qty2), (short_call, "SELL", qty)], trade_name, legs, meta)

    def _enter_put_ratio_backspread(
        self,
        chain: List[dict],
        spot: float,
        atr_val: Optional[float] = None,
        allow_alternative_retry: bool = True,
    ) -> None:
        # Sell 1x ATM put, buy 2x lower-strike puts.
        d = self._get_dynamic_strike_distance(atr_val)
        short_put = self._pick_nearest_strike(chain, "PE", spot)
        if not short_put:
            print("Put ratio backspread: could not find short put")
            return

        long_put = self._pick_nearest_strike(chain, "PE", spot - float(d))
        if not long_put:
            print("Put ratio backspread: could not find long puts")
            return

        def f(v: object) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        sp_strike = f(short_put.get("strike"))
        lp_strike = f(long_put.get("strike"))

        qty = self._entry_qty(atr_val)
        if qty <= 0:
            return
        qty2 = int(qty) * 2

        legs = [
            {
                "symbol": long_put.get("symbol"),
                "token": long_put.get("token"),
                "exchange": long_put.get("exchange"),
                "strike": lp_strike,
                "option_type": "PE",
                "expiry": long_put.get("expiry"),
                "iv": long_put.get("iv"),
                "side": "BUY",
                "quantity": qty2,
            },
            {
                "symbol": short_put.get("symbol"),
                "token": short_put.get("token"),
                "exchange": short_put.get("exchange"),
                "strike": sp_strike,
                "option_type": "PE",
                "expiry": short_put.get("expiry"),
                "iv": short_put.get("iv"),
                "side": "SELL",
                "quantity": qty,
            },
        ]

        # Skip premium/liquidity checks if GPT gate forces TAKE
        if not self._should_gpt_override_entry_checks({"position_type": "multi"}):
            ok, reason = self._check_entry_leg_premiums(legs)
            if not ok:
                print(f"[ENTRY BLOCKED] {reason}")
                return

        meta: Dict[str, object] = {
            "entry_spot": float(spot),
            "atr": float(atr_val) if atr_val is not None else None,
            "short_put_strike": sp_strike,
            "long_put_strike": lp_strike,
            "d": float(d),
            "ratio_long": 2,
            "ratio_short": 1,
        }
        self._capture_entry_premium_meta(legs, meta, atr_val=atr_val)

        base_name = "put_ratio_backspread"
        trade_name = f"{base_name} (weekly)" if bool(getattr(self.cfg, "nifty_weekly_only", False)) else base_name

        if not self._can_open_or_reroute_multi_trade(
            base_name=base_name,
            trade_name=str(trade_name),
            chain=chain,
            spot=float(spot),
            atr_val=atr_val,
            allow_alternative_retry=allow_alternative_retry,
        ):
            return

        try:
            is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
        except Exception:
            is_paper = True
        if not self._gpt_entry_gate(
            proposal={
                "position_type": "multi",
                "trade_name": str(trade_name),
                "strategy_mode": str(getattr(self.cfg, "strategy_name", "") or ""),
                "timeframe": str(getattr(self.cfg, "timeframe", "") or ""),
                "spot": float(spot),
                "atr": float(atr_val) if atr_val is not None else None,
                "legs": [dict(l) for l in legs if isinstance(l, dict)],
                "meta": dict(meta),
            },
            is_paper=bool(is_paper),
        ):
            return

        self._place_multi_leg_transactional([(long_put, "BUY", qty2), (short_put, "SELL", qty)], trade_name, legs, meta)

    def _enter_short_strangle(
        self,
        chain: List[dict],
        spot: float,
        atr_val: Optional[float] = None,
        allow_alternative_retry: bool = True,
    ) -> None:
        d = self._get_dynamic_strike_distance(atr_val)
        call = self._pick_nearest_strike(chain, "CE", spot + d)
        put = self._pick_nearest_strike(chain, "PE", spot - d)
        if not call or not put:
            print("Strangle: could not find OTM CE/PE")
            return
        qty = self._entry_qty(atr_val)
        if qty <= 0:
            return

        def f(v: object) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        call_strike = f(call.get("strike"))
        put_strike = f(put.get("strike"))

        legs = [
            {
                "symbol": call.get("symbol"),
                "token": call.get("token"),
                "exchange": call.get("exchange"),
                "strike": call_strike,
                "option_type": "CE",
                "expiry": call.get("expiry"),
                "iv": call.get("iv"),
                "side": "SELL",
                "quantity": qty,
            },
            {
                "symbol": put.get("symbol"),
                "token": put.get("token"),
                "exchange": put.get("exchange"),
                "strike": put_strike,
                "option_type": "PE",
                "expiry": put.get("expiry"),
                "iv": put.get("iv"),
                "side": "SELL",
                "quantity": qty,
            },
        ]

        # Skip premium/liquidity checks if GPT gate forces TAKE
        if not self._should_gpt_override_entry_checks({"position_type": "multi"}):
            ok, reason = self._check_entry_leg_premiums(legs)
            if not ok:
                print(f"[ENTRY BLOCKED] {reason}")
                return

        meta: Dict[str, object] = {
            "entry_spot": float(spot),
            "atr": float(atr_val) if atr_val is not None else None,
            "short_call_strike": call_strike,
            "short_put_strike": put_strike,
            "d": float(d),
        }

        self._capture_entry_premium_meta(legs, meta, atr_val=atr_val)
        meta["premium_selling"] = True

        base_name = "short_strangle"
        trade_name = (
            f"{base_name} (weekly)" if bool(getattr(self.cfg, "nifty_weekly_only", False)) else base_name
        )

        if not self._can_open_or_reroute_multi_trade(
            base_name=base_name,
            trade_name=str(trade_name),
            chain=chain,
            spot=float(spot),
            atr_val=atr_val,
            allow_alternative_retry=allow_alternative_retry,
        ):
            return

        # Optional GPT entry gate (before placing any orders / recording paper trades).
        try:
            is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
        except Exception:
            is_paper = True
        if not self._gpt_entry_gate(
            proposal={
                "position_type": "multi",
                "trade_name": str(trade_name),
                "strategy_mode": str(getattr(self.cfg, "strategy_name", "") or ""),
                "timeframe": str(getattr(self.cfg, "timeframe", "") or ""),
                "spot": float(spot),
                "atr": float(atr_val) if atr_val is not None else None,
                "legs": [dict(l) for l in legs if isinstance(l, dict)],
                "meta": dict(meta),
            },
            is_paper=bool(is_paper),
        ):
            return

        self._place_multi_leg_transactional([(call, "SELL", qty), (put, "SELL", qty)], trade_name, legs, meta)

    def _enter_iron_condor(
        self,
        chain: List[dict],
        spot: float,
        atr_val: Optional[float] = None,
        allow_alternative_retry: bool = True,
    ) -> None:
        d = self._get_dynamic_strike_distance(atr_val)
        self._enter_iron_structure(
            chain,
            spot,
            atr_val=atr_val,
            d=float(d),
            base_name="iron_condor",
            allow_alternative_retry=allow_alternative_retry,
        )

    def _enter_iron_fly(
        self,
        chain: List[dict],
        spot: float,
        atr_val: Optional[float] = None,
        allow_alternative_retry: bool = True,
    ) -> None:
        # Iron fly = ATM short straddle with protective wings.
        self._enter_iron_structure(
            chain,
            spot,
            atr_val=atr_val,
            d=0.0,
            base_name="iron_fly",
            allow_alternative_retry=allow_alternative_retry,
        )

    def _enter_iron_butterfly(
        self,
        chain: List[dict],
        spot: float,
        atr_val: Optional[float] = None,
        allow_alternative_retry: bool = True,
    ) -> None:
        # Iron butterfly is effectively the same structure as an iron fly in this template.
        self._enter_iron_structure(
            chain,
            spot,
            atr_val=atr_val,
            d=0.0,
            base_name="iron_butterfly",
            allow_alternative_retry=allow_alternative_retry,
        )

    def _enter_iron_structure(
        self,
        chain: List[dict],
        spot: float,
        *,
        atr_val: Optional[float],
        d: float,
        base_name: str,
        allow_alternative_retry: bool = True,
    ) -> None:
        """Generic 4-leg iron structure.

        - `iron_condor`: d > 0
        - `iron_fly` / `iron_butterfly`: d == 0
        """

        w = float(self.cfg.wing_width)

        short_call = self._pick_nearest_strike(chain, "CE", spot + d)
        short_put = self._pick_nearest_strike(chain, "PE", spot - d)
        long_call = self._pick_nearest_strike(chain, "CE", spot + d + w)
        long_put = self._pick_nearest_strike(chain, "PE", spot - d - w)

        if not short_call or not short_put or not long_call or not long_put:
            print(f"{base_name}: could not find required strikes")
            return
        qty = self._entry_qty(atr_val)
        if qty <= 0:
            return

        def f(v: object) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        sc = f(short_call.get("strike"))
        sp = f(short_put.get("strike"))
        lc = f(long_call.get("strike"))
        lp = f(long_put.get("strike"))

        legs = [
            {
                "symbol": short_call.get("symbol"),
                "token": short_call.get("token"),
                "exchange": short_call.get("exchange"),
                "strike": sc,
                "option_type": "CE",
                "expiry": short_call.get("expiry"),
                "iv": short_call.get("iv"),
                "side": "SELL",
                "quantity": qty,
            },
            {
                "symbol": short_put.get("symbol"),
                "token": short_put.get("token"),
                "exchange": short_put.get("exchange"),
                "strike": sp,
                "option_type": "PE",
                "expiry": short_put.get("expiry"),
                "iv": short_put.get("iv"),
                "side": "SELL",
                "quantity": qty,
            },
            {
                "symbol": long_call.get("symbol"),
                "token": long_call.get("token"),
                "exchange": long_call.get("exchange"),
                "strike": lc,
                "option_type": "CE",
                "expiry": long_call.get("expiry"),
                "iv": long_call.get("iv"),
                "side": "BUY",
                "quantity": qty,
            },
            {
                "symbol": long_put.get("symbol"),
                "token": long_put.get("token"),
                "exchange": long_put.get("exchange"),
                "strike": lp,
                "option_type": "PE",
                "expiry": long_put.get("expiry"),
                "iv": long_put.get("iv"),
                "side": "BUY",
                "quantity": qty,
            },
        ]

        # Skip premium/liquidity checks if GPT gate forces TAKE
        if not self._should_gpt_override_entry_checks({"position_type": "multi"}):
            ok, reason = self._check_entry_leg_premiums(legs)
            if not ok:
                print(f"[ENTRY BLOCKED] {reason}")
                return

        meta: Dict[str, object] = {
            "entry_spot": float(spot),
            "atr": float(atr_val) if atr_val is not None else None,
            "short_call_strike": sc,
            "short_put_strike": sp,
            "long_call_strike": lc,
            "long_put_strike": lp,
            "d": float(d),
            "w": float(w),
        }

        self._capture_entry_premium_meta(legs, meta, atr_val=atr_val)
        meta["premium_selling"] = True

        trade_name = f"{base_name} (weekly)" if bool(getattr(self.cfg, "nifty_weekly_only", False)) else base_name

        if not self._can_open_or_reroute_multi_trade(
            base_name=base_name,
            trade_name=str(trade_name),
            chain=chain,
            spot=float(spot),
            atr_val=atr_val,
            allow_alternative_retry=allow_alternative_retry,
        ):
            return

        # Optional GPT entry gate (before placing any orders / recording paper trades).
        try:
            is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
        except Exception:
            is_paper = True
        if not self._gpt_entry_gate(
            proposal={
                "position_type": "multi",
                "trade_name": str(trade_name),
                "strategy_mode": str(getattr(self.cfg, "strategy_name", "") or ""),
                "timeframe": str(getattr(self.cfg, "timeframe", "") or ""),
                "spot": float(spot),
                "atr": float(atr_val) if atr_val is not None else None,
                "legs": [dict(l) for l in legs if isinstance(l, dict)],
                "meta": dict(meta),
            },
            is_paper=bool(is_paper),
        ):
            return

        self._place_multi_leg_transactional([
            (long_call, "BUY", qty),
            (long_put, "BUY", qty),
            (short_call, "SELL", qty),
            (short_put, "SELL", qty)
        ], trade_name, legs, meta)

    def _enter_delta_hedged_long_straddle(
        self,
        chain: List[dict],
        spot: float,
        atr_val: Optional[float] = None,
        allow_alternative_retry: bool = True,
    ) -> None:
        call = self._pick_nearest_strike(chain, "CE", spot)
        put = self._pick_nearest_strike(chain, "PE", spot)
        if not call or not put:
            print("Delta-hedged long straddle: could not find ATM CE/PE")
            return
        qty = self._entry_qty(atr_val)
        if qty <= 0:
            return

        def f(v: object) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        call_strike = f(call.get("strike"))
        put_strike = f(put.get("strike"))

        legs = [
            {
                "symbol": call.get("symbol"),
                "token": call.get("token"),
                "exchange": call.get("exchange"),
                "strike": call_strike,
                "option_type": "CE",
                "expiry": call.get("expiry"),
                "iv": call.get("iv"),
                "side": "BUY",
                "quantity": qty,
            },
            {
                "symbol": put.get("symbol"),
                "token": put.get("token"),
                "exchange": put.get("exchange"),
                "strike": put_strike,
                "option_type": "PE",
                "expiry": put.get("expiry"),
                "iv": put.get("iv"),
                "side": "BUY",
                "quantity": qty,
            },
        ]

        # Skip premium/liquidity checks if GPT gate forces TAKE
        if not self._should_gpt_override_entry_checks({"position_type": "multi"}):
            ok, reason = self._check_entry_leg_premiums(legs)
            if not ok:
                print(f"[ENTRY BLOCKED] {reason}")
                return

        meta: Dict[str, object] = {
            "entry_spot": float(spot),
            "atr": float(atr_val) if atr_val is not None else None,
            "long_call_strike": call_strike,
            "long_put_strike": put_strike,
            "delta_hedge": True,
            "delta_hedge_symbol": str(getattr(self.cfg, "delta_hedge_symbol", "") or "").strip() or None,
            "delta_hedge_exchange": str(getattr(self.cfg, "delta_hedge_exchange", "") or "").strip() or None,
        }

        self._capture_entry_premium_meta(legs, meta, atr_val=atr_val)

        base_name = "delta_hedged_long_straddle"
        trade_name = f"{base_name} (weekly)" if bool(getattr(self.cfg, "nifty_weekly_only", False)) else base_name

        if not self._can_open_or_reroute_multi_trade(
            base_name=base_name,
            trade_name=str(trade_name),
            chain=chain,
            spot=float(spot),
            atr_val=atr_val,
            allow_alternative_retry=allow_alternative_retry,
        ):
            return

        # Optional GPT entry gate (before placing any orders / recording paper trades).
        try:
            is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
        except Exception:
            is_paper = True
        if not self._gpt_entry_gate(
            proposal={
                "position_type": "multi",
                "trade_name": str(trade_name),
                "strategy_mode": str(getattr(self.cfg, "strategy_name", "") or ""),
                "timeframe": str(getattr(self.cfg, "timeframe", "") or ""),
                "spot": float(spot),
                "atr": float(atr_val) if atr_val is not None else None,
                "legs": [dict(l) for l in legs if isinstance(l, dict)],
                "meta": dict(meta),
            },
            is_paper=bool(is_paper),
        ):
            return

        self._place_multi_leg_transactional([(call, "BUY", qty), (put, "BUY", qty)], trade_name, legs, meta)
        try:
            tr = self.state.open_multi[-1]
            self._rebalance_delta_hedge(tr, spot=float(spot), force=True)
        except Exception:
            pass

    def _enter_delta_hedged_short_straddle(
        self,
        chain: List[dict],
        spot: float,
        atr_val: Optional[float] = None,
        allow_alternative_retry: bool = True,
    ) -> None:
        call = self._pick_nearest_strike(chain, "CE", spot)
        put = self._pick_nearest_strike(chain, "PE", spot)
        if not call or not put:
            print("Delta-hedged short straddle: could not find ATM CE/PE")
            return
        qty = self._entry_qty(atr_val)
        if qty <= 0:
            return

        def f(v: object) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        call_strike = f(call.get("strike"))
        put_strike = f(put.get("strike"))

        legs = [
            {
                "symbol": call.get("symbol"),
                "token": call.get("token"),
                "exchange": call.get("exchange"),
                "strike": call_strike,
                "option_type": "CE",
                "expiry": call.get("expiry"),
                "iv": call.get("iv"),
                "side": "SELL",
                "quantity": qty,
            },
            {
                "symbol": put.get("symbol"),
                "token": put.get("token"),
                "exchange": put.get("exchange"),
                "strike": put_strike,
                "option_type": "PE",
                "expiry": put.get("expiry"),
                "iv": put.get("iv"),
                "side": "SELL",
                "quantity": qty,
            },
        ]

        # Skip premium/liquidity checks if GPT gate forces TAKE
        if not self._should_gpt_override_entry_checks({"position_type": "multi"}):
            ok, reason = self._check_entry_leg_premiums(legs)
            if not ok:
                print(f"[ENTRY BLOCKED] {reason}")
                return

        meta: Dict[str, object] = {
            "entry_spot": float(spot),
            "atr": float(atr_val) if atr_val is not None else None,
            "short_call_strike": call_strike,
            "short_put_strike": put_strike,
            "premium_selling": True,
            "delta_hedge": True,
            "delta_hedge_symbol": str(getattr(self.cfg, "delta_hedge_symbol", "") or "").strip() or None,
            "delta_hedge_exchange": str(getattr(self.cfg, "delta_hedge_exchange", "") or "").strip() or None,
        }

        self._capture_entry_premium_meta(legs, meta, atr_val=atr_val)

        base_name = "delta_hedged_short_straddle"
        trade_name = f"{base_name} (weekly)" if bool(getattr(self.cfg, "nifty_weekly_only", False)) else base_name

        if not self._can_open_or_reroute_multi_trade(
            base_name=base_name,
            trade_name=str(trade_name),
            chain=chain,
            spot=float(spot),
            atr_val=atr_val,
            allow_alternative_retry=allow_alternative_retry,
        ):
            return

        # Optional GPT entry gate (before placing any orders / recording paper trades).
        try:
            is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
        except Exception:
            is_paper = True
        if not self._gpt_entry_gate(
            proposal={
                "position_type": "multi",
                "trade_name": str(trade_name),
                "strategy_mode": str(getattr(self.cfg, "strategy_name", "") or ""),
                "timeframe": str(getattr(self.cfg, "timeframe", "") or ""),
                "spot": float(spot),
                "atr": float(atr_val) if atr_val is not None else None,
                "legs": [dict(l) for l in legs if isinstance(l, dict)],
                "meta": dict(meta),
            },
            is_paper=bool(is_paper),
        ):
            return

        self._place_multi_leg_transactional([(call, "SELL", qty), (put, "SELL", qty)], trade_name, legs, meta)
        try:
            tr = self.state.open_multi[-1]
            self._rebalance_delta_hedge(tr, spot=float(spot), force=True)
        except Exception:
            pass

    def _risk_status(self) -> tuple[bool, str]:
        self._ensure_daily_risk_counters_day()
        # P&L-based daily stops need a reliable positions/LTP schema from the broker.
        # Until then, enforce broker-schema-independent kill-switch rules.
        try:
            max_stopouts = int(self.cfg.max_stopouts_per_day)
        except Exception:
            max_stopouts = 0
        if max_stopouts > 0 and int(self.state.stopouts_today) >= max_stopouts:
            return (
                False,
                f"max_stopouts_per_day reached ({int(self.state.stopouts_today)}/{max_stopouts})",
            )

        try:
            max_consecutive = int(self.cfg.max_consecutive_stopouts)
        except Exception:
            max_consecutive = 0
        if max_consecutive > 0 and int(self.state.consecutive_stopouts) >= max_consecutive:
            return (
                False,
                f"max_consecutive_stopouts reached ({int(self.state.consecutive_stopouts)}/{max_consecutive})",
            )

        # Only enforce daily loss guard when it's positive; a value
        # of 0 or negative disables this kill switch.
        try:
            max_dd = float(self.cfg.max_daily_loss)
        except Exception:
            max_dd = 0.0

        try:
            max_prof = float(getattr(self.cfg, "max_daily_profit", 0.0) or 0.0)
        except Exception:
            max_prof = 0.0

        current_pnl = float(self.state.realized_pnl)

        # If SQLite is enabled, pull the db sum instead of trusting just RAM
        try:
            if hasattr(self, "db_manager") and self.db_manager:
                db_pnl = self.db_manager.get_todays_realized_pnl()
                current_pnl = db_pnl
        except Exception:
            pass

        # Risk gap fix: include unrealized MTM on open positions in total loss.
        # total_loss = realized_pnl + sum(ltp - entry_price for each open leg)
        unrealized = self._compute_total_unrealized_mtm()
        total_pnl = current_pnl + unrealized

        if max_dd > 0 and total_pnl <= -max_dd:
            return (
                False,
                f"max_daily_loss hit (total_pnl={total_pnl:.2f} <= -{max_dd:.2f}, "
                f"realized={current_pnl:.2f} + unrealized={unrealized:.2f})",
            )

        if max_prof > 0 and total_pnl >= max_prof:
            return (
                False,
                f"max_daily_profit hit (total_pnl={total_pnl:.2f} >= {max_prof:.2f}, "
                f"realized={current_pnl:.2f} + unrealized={unrealized:.2f})",
            )

        return True, ""

    def _within_risk_limits(self) -> bool:
        ok, _reason = self._risk_status()
        return ok

    def _select_atm_option(
        self,
        option_chain: List[dict],
        spot: float,
        option_type: OptionType,
    ) -> Optional[dict]:
        """Pick ATM option from chain based on closest strike to spot.

        Expects each element in `option_chain` to contain at least
        `strike`, `option_type`, and `symbol` keys, and ideally an
        `iv` field for greeks. You may need to adapt the keys to match
        your m.Stock response.
        """
        def norm_type(v: object) -> str:
            s = str(v or "").strip().upper()
            if s in {"CALL", "C"}:
                return "CE"
            if s in {"PUT", "P"}:
                return "PE"
            return s

        want = norm_type(option_type)
        candidates = [
            opt
            for opt in option_chain
            if norm_type(opt.get("option_type", "")) == want and opt.get("strike") is not None
        ]
        if not candidates:
            return None
        try:
            return min(candidates, key=lambda o: abs(float(o["strike"]) - spot))
        except Exception:
            # If some rows have non-numeric strikes, skip them.
            safe: List[dict] = []
            for o in candidates:
                try:
                    float(o.get("strike"))
                except Exception:
                    continue
                safe.append(o)
            if not safe:
                return None
            return min(safe, key=lambda o: abs(float(o["strike"]) - spot))

    def _strategy_winrate_check(self, strategy_name: str) -> bool:
        """Check if a strategy should be allowed based on its recent win-rate.

        Returns True if allowed, False if blocked.

        Reads config keys:
          winrate_tracker_enabled (bool) - master switch
          winrate_lookback (int) - how many recent trades to track (default 10)
          winrate_min_wins (int) - minimum wins before filter activates (default 3)
          winrate_max_loss_streak (int) - consecutive losses before disable (default 3)
          winrate_min_win_rate (float) - min win rate % to stay enabled (default 30.0)
          winrate_cooldown_sec (int) - seconds before retry (default 300)
        """
        try:
            if not bool(getattr(self.cfg, "winrate_tracker_enabled", False)):
                return True
        except Exception:
            return True

        name = str(strategy_name or "").strip().lower()
        if not name or name in {"auto", "directional"}:
            return True

        try:
            lookback = int(getattr(self.cfg, "winrate_lookback", 10) or 10)
        except Exception:
            lookback = 10
        try:
            min_wins = int(getattr(self.cfg, "winrate_min_wins", 3) or 3)
        except Exception:
            min_wins = 3
        try:
            max_loss_streak = int(getattr(self.cfg, "winrate_max_loss_streak", 3) or 3)
        except Exception:
            max_loss_streak = 3
        try:
            min_win_rate = float(getattr(self.cfg, "winrate_min_win_rate", 30.0) or 30.0)
        except Exception:
            min_win_rate = 30.0
        try:
            cooldown = int(getattr(self.cfg, "winrate_cooldown_sec", 300) or 300)
        except Exception:
            cooldown = 300

        lookback = max(2, min(50, int(lookback)))
        min_wins = max(1, min(int(lookback) - 1, int(min_wins)))
        max_loss_streak = max(1, min(int(lookback), int(max_loss_streak)))
        min_win_rate = max(1.0, min(99.0, float(min_win_rate)))

        now_ts = float(time.time())

        # Check if currently disabled
        disabled_until = float(self._disabled_strategies.get(name, 0.0) or 0.0)
        if disabled_until > 0:
            if now_ts < disabled_until:
                return False
            else:
                self._disabled_strategies.pop(name, None)
                return True

        # Get or create tracker entry
        tracker = self._strategy_winrate.setdefault(name, {"results": [], "wins": 0, "losses": 0})
        try:
            results = list(tracker.get("results") or [])
        except Exception:
            results = []

        if len(results) < int(min_wins):
            return True

        recent = list(results)[-int(lookback):]
        wins = sum(1 for r in recent if r is True)
        losses = sum(1 for r in recent if r is False)

        # Consecutive loss check
        consec_losses = 0
        for r in reversed(recent):
            if r is False:
                consec_losses += 1
            else:
                break

        if int(consec_losses) >= int(max_loss_streak):
            self._disabled_strategies[name] = float(now_ts) + float(cooldown)
            print(f"[WINRATE] {name}: {consec_losses} consecutive losses - disabled for {cooldown}s")
            return False

        # Win rate check
        if wins + losses > 0:
            win_rate = (float(wins) / float(wins + losses)) * 100.0
            if float(win_rate) < float(min_win_rate):
                self._disabled_strategies[name] = float(now_ts) + float(cooldown)
                print(f"[WINRATE] {name}: {win_rate:.1f}% win rate (< {min_win_rate}%) - disabled for {cooldown}s")
                return False

        return True

    def _strategy_winrate_record_result(self, strategy_name: str, won: bool) -> None:
        """Record a win or loss for a strategy."""
        if not bool(getattr(self.cfg, "winrate_tracker_enabled", False)):
            return

        name = str(strategy_name or "").strip().lower()
        if not name or name in {"auto", "directional"}:
            return

        try:
            lookback = int(getattr(self.cfg, "winrate_lookback", 10) or 10)
        except Exception:
            lookback = 10
        lookback = max(2, min(50, int(lookback)))

        tracker = self._strategy_winrate.setdefault(name, {"results": [], "wins": 0, "losses": 0})
        try:
            results = list(tracker.get("results") or [])
        except Exception:
            results = []

        results.append(bool(won))
        if len(results) > int(lookback) * 2:
            results = results[-int(lookback) * 2:]

        wins = sum(1 for r in results if r is True)
        losses = sum(1 for r in results if r is False)

        tracker["results"] = results
        tracker["wins"] = int(wins)
        tracker["losses"] = int(losses)

    def _decide_entries(
        self,
        *,
        strategy_override: Optional[str] = None,
        attempted_auto_strategies: Optional[set[str]] = None,
    ) -> None:
        """Decide whether to open new scalping trades using TA + greeks.

        This is an example template combining:
        - EMAs and RSI on the underlying
        - Candlestick patterns on the underlying
        - Delta of ATM options from the option chain
        """
        def block(reason: str) -> None:
            self._note_entry_blocked(reason)

        # P4: Check if entries are paused by regime monitor
        if getattr(self, "_entries_paused", False):
            block("entries_paused_by_gpt_regime")
            return

        # External kill-switch: block all new entries when active.
        if self._kill_switch_active():
            block("kill_switch_active")
            return

        # ---- Fail-fast: if option chain is unavailable, skip all strategies this cycle ----
        try:
            client = getattr(self, 'client', None)
            if client is not None and hasattr(client, '_option_chain_status'):
                chain_status = str(getattr(client, '_option_chain_status', 'UNKNOWN') or 'UNKNOWN')
                if chain_status in ('MISSING_CONFIG', 'FETCH_FAILED', 'EMPTY'):
                    self._note_entry_blocked(f"option_chain_{chain_status.lower()}")
                    # [OPTION-CHAIN] Throttled logging to prevent log spam
                    self._log_throttled(
                        f"[ENTRY] Option chain unavailable ({chain_status}) — skipping all strategies",
                        key=f"option_chain_{chain_status}",
                        interval_sec=30.0
                    )
                    return
        except Exception:
            pass

        # =====================================================================
        # PHASE 6: PRODUCTION WIRING — route_candidate_decision MUST be called
        # before any entry decision that can lead to order placement.
        # Router result drives final_signal / gates. Legacy ML is passed as fallback.
        # =====================================================================
        router_decision = None
        router_final = "NO_TRADE"
        router_allowed = False
        router_no_trade_reason = "router_not_called"
        try:
            if route_candidate_decision is not None:
                # Build minimal live-computable snapshot from current state (no future/return/label)
                latest_candle = candles[-1] if candles else None
                spot = float(getattr(latest_candle, "close", 0.0) or 0.0)
                # Pull last ML prob computed in evaluate_entry_signals (or 0)
                last_ml_p = float(getattr(self, "_last_ml_pred", 0.0) or 0.0)
                last_ml_thr = float(getattr(self, "_last_ml_threshold", 0.5) or 0.5)
                legacy_sig = {"ml_prob": last_ml_p, "threshold": last_ml_thr, "reason": "from_evaluate_entry_signals"}

                # Active candidate wiring (env-driven, safe defaults)
                active_cid = (os.getenv("MSTOCK_ACTIVE_CANDIDATE_ID") or "").strip() or None
                # Candidate dir search order (no hard-coded secrets)
                cdir = None
                for cand in ("artifacts/candidates", "models/candidates", "data/candidates"):
                    if os.path.isdir(cand):
                        cdir = cand
                        break
                if not cdir:
                    cdir = "models/candidates"

                mode_hint = "live" if bool(getattr(self.cfg, "enable_live_trading", False)) else "shadow"
                router_decision = route_candidate_decision(
                    market_snapshot={
                        "spot": spot,
                        "last_close": spot,
                        "regime": str(getattr(self, "_last_regime_profile", {}).get("regime", "unknown")),
                        "market_regime": str(getattr(self, "_last_regime_profile", {}).get("regime", "unknown")),
                        "atr_pct": float(self._cached_atr or 0.01),
                        "option_type": None,  # explicit side comes from preset/policy
                    },
                    option_chain_snapshot=None,
                    legacy_signal=legacy_sig,
                    mode=mode_hint,
                    active_candidate_id=active_cid,
                    candidate_dir=cdir,
                    force_eval=False,  # never auto-force in live strategy path
                )
                router_final = str(router_decision.get("final_signal", "NO_TRADE"))
                # All gates must be true for entry (except in pure shadow observation)
                router_allowed = (
                    router_final.startswith("BUY_")
                    and bool(router_decision.get("allowed_by_model"))
                    and bool(router_decision.get("allowed_by_preset"))
                    and bool(router_decision.get("allowed_by_side_policy"))
                    and bool(router_decision.get("allowed_by_liquidity"))
                    and bool(router_decision.get("allowed_by_cost"))
                    and bool(router_decision.get("allowed_by_risk"))
                )
                router_no_trade_reason = router_decision.get("no_trade_reason") or ""
                risk_cfg = (router_decision or {}).get("risk") if isinstance(router_decision, dict) else {}
                if isinstance(risk_cfg, dict):
                    try:
                        router_max_trades = int(risk_cfg.get("max_trades_per_day") or 0)
                    except Exception:
                        router_max_trades = 0
                    if router_max_trades > 0 and int(self.state.trades_today) >= router_max_trades:
                        router_allowed = False
                        router_no_trade_reason = f"router_max_trades_{self.state.trades_today}_gte_{router_max_trades}"
                    try:
                        cooldown_minutes = int(risk_cfg.get("cooldown_minutes") or 0)
                    except Exception:
                        cooldown_minutes = 0
                    if cooldown_minutes > 0:
                        last_entry_ts = float(getattr(self.state, "last_entry_ts", 0.0) or 0.0)
                        if last_entry_ts > 0 and (time.time() - last_entry_ts) < (cooldown_minutes * 60):
                            router_allowed = False
                            router_no_trade_reason = f"router_cooldown_{cooldown_minutes}m_active"
                # Stash for GUI + later placement guards
                self._last_router_decision = router_decision
                self._last_router_allowed = router_allowed
            else:
                # Router module not importable — fail closed for new candidate path
                router_allowed = False
                router_no_trade_reason = "candidate_router_unavailable"
                self._last_router_decision = None
                self._last_router_allowed = False
        except Exception as _exc:
            router_allowed = False
            router_no_trade_reason = f"router_exception:{type(_exc).__name__}"
            self._last_router_decision = {"error": str(_exc)}
            self._last_router_allowed = False

        # If router says NO or any gate failed, block new entries from candidate path.
        # (Legacy non-candidate branches may still run for compatibility but will also be gated at placement.)
        if not router_allowed and router_decision is not None:
            # Log the gate decision
            live_enabled_flag = str(os.getenv("MSTOCK_ENABLE_LIVE_ORDERS", "")).strip().lower() in {"1", "true", "yes"}
            LOGGER.info(
                "[LIVE-GATE] candidate_id=%s preset=%s side=%s final_signal=%s shadow_ready=%s "
                "live_orders_enabled=%s order_allowed=%s reason=%s",
                (router_decision or {}).get("candidate_id", ""),
                (router_decision or {}).get("selected_preset", ""),
                (router_decision or {}).get("side_decision", ""),
                router_final,
                (router_decision or {}).get("shadow_ready", False),
                live_enabled_flag,
                False,
                router_no_trade_reason or "router_block",
            )
            # In live mode we are extra strict; in shadow/paper we still respect router final for candidate flow.
            if bool(getattr(self.cfg, "enable_live_trading", False)) or router_final == "NO_TRADE":
                block(f"router_{router_no_trade_reason or 'no_trade'}")
                # Do not return here for all paths — some legacy sleeves still exist — but we set a hard flag.
                self._router_forced_block = True
            else:
                self._router_forced_block = False
        else:
            self._router_forced_block = False

        attempted_auto_strategies_set = {
            self._resolve_auto_strategy_candidate(name)
            for name in set(attempted_auto_strategies or set())
            if str(name or "").strip()
        }

        def maybe_reroute_entry_block(reason: str, *, current_strategy: str) -> bool:
            extras = {
                "blocked_strategy": str(current_strategy or ""),
                "blocked_reason": str(reason or ""),
            }

            # ---- Early-exit for option-chain failures: do NOT attempt reroute ----
            reason_lower = str(reason or "").lower()
            option_chain_fail_keywords = (
                "option chain empty",
                "option chain parameters are not configured",
                "missing_config",
                "ip_mismatch",
                "ip mismatch",
            )
            if any(kw in reason_lower for kw in option_chain_fail_keywords):
                self._note_entry_blocked(reason, extras=extras)
                print("[AUTO][REROUTE] blocked: option chain unavailable, skipping all fallback strategies")
                return False

            try:
                strategy_mode = self._normalize_strategy_name(str(getattr(self.cfg, "strategy_name", "") or ""))
            except Exception:
                strategy_mode = ""

            current_key = self._resolve_auto_strategy_candidate(current_strategy)
            if strategy_mode != "auto" or not current_key:
                self._note_entry_blocked(reason, extras=extras)
                return False

            suggestion = self._blocked_auto_strategy_reroute(
                blocked_strategy=str(current_key),
                blocked_reason=str(reason or ""),
                attempted_strategies=set(attempted_auto_strategies_set),
            )
            if not isinstance(suggestion, dict):
                self._note_entry_blocked(reason, extras=extras)
                return False

            alt_name = self._resolve_auto_strategy_candidate(suggestion.get("trade_name"))
            if not alt_name:
                self._note_entry_blocked(reason, extras=extras)
                return False

            extras.update(
                {
                    "suggested_trade": str(alt_name),
                    "suggestion_source": str(suggestion.get("source") or "policy"),
                    "alternative_candidates": list(suggestion.get("candidates") or []),
                    "suggestion_reason": str(suggestion.get("reason") or ""),
                }
            )
            self._note_entry_blocked(reason, extras=extras)

            why = str(suggestion.get("reason") or "").strip()
            suffix = f" reason={why}" if why else ""
            print(
                f"[AUTO][REROUTE] {current_key} blocked by {reason}; "
                f"trying {alt_name} ({extras['suggestion_source']}){suffix}"
            )

            next_attempted = set(attempted_auto_strategies_set)
            next_attempted.add(str(current_key))
            self._decide_entries(
                strategy_override=str(alt_name),
                attempted_auto_strategies=next_attempted,
            )
            return True

        print("[CHECK] Time:", self._now_ist_time())

        open_count = len(self.state.open_directional) + len(self.state.open_multi)
        last_action_ts = 0.0
        try:
            last_action_ts = max(float(self.state.last_entry_ts or 0.0), float(self.state.last_exit_ts or 0.0))
        except Exception:
            last_action_ts = float(self.state.last_entry_ts or 0.0)

        cooldown_s = None
        try:
            if last_action_ts:
                cooldown_s = float(time.time() - float(last_action_ts))
        except Exception:
            cooldown_s = None
        print(
            f"[STATE] trades_today={self.state.trades_today}, "
            f"open_positions={open_count}, "
            f"cooldown={'N/A' if cooldown_s is None else f'{cooldown_s:.1f}s'}"
        )

        ok_risk, reason_risk = self._risk_status()
        if not bool(ok_risk):
            block(f"Risk limit active: {reason_risk}")
            return

        # Cooldown + max trades/day guardrails.
        if self.state.trades_today >= int(self.cfg.max_trades_per_day):
            block("Max trades per day reached")
            return
        if last_action_ts and (time.time() - float(last_action_ts)) < float(self.cfg.cooldown_sec):
            block("Cooldown active")
            return
        try:
            stop_cd = float(getattr(self.cfg, "cooldown_after_stopout_sec", 0.0) or 0.0)
        except Exception:
            stop_cd = 0.0
        if stop_cd > 0 and float(self.state.last_stopout_ts or 0.0) > 0:
            since_stop = float(time.time() - float(self.state.last_stopout_ts))
            if since_stop < stop_cd:
                block(f"Cooldown after stopout ({since_stop:.0f}s/{stop_cd:.0f}s)")
                return

        # Cap total "positions" across directional and multi-leg.
        # If pyramiding is enabled (max_pyramid_levels > 0), allow entering
        # any strategy type as it will increment existing trade quantity.
        if open_count >= self.cfg.max_open_positions:
            if int(getattr(self.cfg, "max_pyramid_levels", 0) or 0) <= 0:
                block("Max open positions reached")
                return

        now_t = self._now_ist_time()
        sess_min_atr, sess_max_atr, sess_rsi_low, sess_rsi_high, sess_bucket = self._session_overrides(now_t)

        # Global entry cutoff: when configured, do not open ANY new trades after this time.
        # (Exits are still managed by the exit logic elsewhere.)
        entry_cutoff_raw = str(getattr(self.cfg, "premium_entry_cutoff_hhmm", "") or "").strip()
        if entry_cutoff_raw:
            entry_cutoff_t = self._parse_hhmm(entry_cutoff_raw, default=dt_time(23, 59))
            if now_t >= entry_cutoff_t:
                block(f"Entry cutoff active ({entry_cutoff_raw}); no new entries")
                return

        try:
            opening_minutes = int(getattr(self.cfg, "opening_filter_minutes", 45) or 45)
        except Exception:
            opening_minutes = 45

        # If we are early in the session, fetch enough candles to cover the opening filter.
        # Ensure sufficient history for ATR/EMA.
        atr_p = int(self.cfg.atr_period) if self.cfg.atr_period else 14
        
        # Timeframe awareness: adjust candle limit for higher timeframes.
        tf_str = self.cfg.timeframe
        tf_min = 1
        if tf_str.endswith("m"):
            try:
                tf_min = int(tf_str[:-1])
            except Exception:
                tf_min = 1
        elif tf_str.endswith("h"):
            try:
                tf_min = int(tf_str[:-1]) * 60
            except Exception:
                tf_min = 60
        elif tf_str.endswith("d"):
            tf_min = 1440
        
        # Calculate how many candles we need for the 'opening' window
        opening_candles = (opening_minutes + tf_min - 1) // tf_min
        candle_limit = max(100, opening_candles + 5, atr_p * 3)

        using_synth_candles = False

        try:
            candles: List[Candle] = self.client.get_candles(
                symbol=self.cfg.underlying,
                timeframe=tf_str,
                limit=int(candle_limit),
            )
        except Exception as exc:  # noqa: BLE001
            candles = []
            # Log and skip if data fetching fails.
            self._log_throttled(
                f"Failed to fetch candles: {exc}",
                key=str(exc),
                interval_sec=max(10.0, float(self.cfg.polling_interval_sec) * 10.0),
            )
            block("Candle fetch failed")
            return

        # Intraday-only warmup fallback: if the intraday candle endpoint returns empty,
        # synthesize candles from live LTP samples (no historical endpoints).
        if (not candles) and self._prefer_synthetic_warmup():
            candles = self._synth_candles_from_ltp(timeframe=tf_str, limit=int(candle_limit))
            using_synth_candles = bool(candles)
            if candles and self._bool_env("MSTOCK_DEBUG_INTRADAY", False):
                now_ts = float(time.time())
                if (now_ts - float(self._last_synth_log_ts)) > 30.0:
                    print(
                        f"[CANDLES] Intraday empty; using synthetic LTP candles for warmup ({tf_str}, n={len(candles)})"
                    )
                    self._last_synth_log_ts = now_ts
                    
        # Expose the candles to the UI layer for charting
        if candles and self._on_tick:
            try:
                self._on_tick(candles)
            except Exception as exc:
                pass

        # ---- MTF Confirmation ----
        mtf_ema_up = False
        mtf_ema_down = False
        if bool(getattr(self.cfg, "enable_mtf_confirmation", False)):
            mtf_tf = getattr(self.cfg, "mtf_timeframe", "5m")
            try:
                # Reuse ema periods from config
                e_f = int(getattr(self.cfg, "ema_fast", 9))
                e_s = int(getattr(self.cfg, "ema_slow", 21))
                mtf_limit = max(50, e_s + 5)
                m_candles: List[Candle] = self.client.get_candles(
                    symbol=self.cfg.underlying,
                    timeframe=mtf_tf,
                    limit=mtf_limit,
                )
                if (not m_candles) and self._prefer_synthetic_warmup():
                    m_candles = self._synth_candles_from_ltp(timeframe=mtf_tf, limit=int(mtf_limit))
                if m_candles and len(m_candles) >= e_s:
                    m_candles = sorted(m_candles, key=lambda c: c.time)
                    m_closes = [c.close for c in m_candles]
                    m_fast = ema(m_closes, period=e_f)
                    m_slow = ema(m_closes, period=e_s)
                    if m_fast is not None and m_slow is not None:
                        mtf_ema_up = m_fast > m_slow
                        mtf_ema_down = m_fast < m_slow
            except Exception as exc:  # noqa: BLE001
                print(f"[MTF] Failed to fetch {mtf_tf} candles: {exc}")

        # ---- #2: MTF Hard Filter ----
        # When enabled, block entries that go against the MTF trend.
        try:
            mtf_hard = bool(getattr(self.cfg, "mtf_hard_filter", False))
        except Exception:
            mtf_hard = False
        if mtf_hard and (mtf_ema_up or mtf_ema_down):
            try:
                desired_dir_local = desired_dir
            except NameError:
                desired_dir_local = None
            if desired_dir_local is not None:
                if desired_dir_local == "bull" and mtf_ema_down and not mtf_ema_up:
                    reason = "MTF trend is bearish (conflicts with bull entry)"
                    if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                        return
                elif desired_dir_local == "bear" and mtf_ema_up and not mtf_ema_down:
                    reason = "MTF trend is bullish (conflicts with bear entry)"
                    if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                        return

        deferred_eval_res = None

        # Some sources return candles newest-first. Indicators assume oldest->newest.
        # Sort defensively so EMA/RSI direction is correct.
        try:
            candles = sorted(candles, key=lambda c: c.time)
        except Exception:
            pass

        # Determine minimum candles required for all indicators used.
        # ATR(N) requires N+1 candles, RSI(M) requires M+1, EMA(K) requires K.
        # We also want a safety margin for stability.
        ema_f_p = int(getattr(self.cfg, "ema_fast", 9))
        ema_s_p = int(getattr(self.cfg, "ema_slow", 21))
        rsi_p = 14  # currently hardcoded in indicators.py
        atr_p = int(self.cfg.atr_period) if self.cfg.atr_period else 14

        # The absolute minimum for any indicator to return a value:
        # RSI(14) needs 15 closes. ATR(14) needs 15 candles. EMA(21) needs 21.
        MIN_CANDLES = max(atr_p + 1, rsi_p + 1, ema_s_p, ema_f_p)

        if (not candles) or len(candles) < MIN_CANDLES:
            block(f"Warming up ({len(candles)}/{MIN_CANDLES} candles)")
            time.sleep(10)  # slightly faster retry than 30s
            return

        try:
            max_age = float(getattr(self.cfg, "entry_candle_max_age_sec", 0.0) or 0.0)
        except Exception:
            max_age = 0.0
        if max_age > 0:
            age = self._candle_age_sec(candles[-1])
            if age is not None and age > max_age:
                block(f"Latest candle stale ({age:.0f}s > {max_age:.0f}s)")
                return

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]

        # Opening impulse filter (first N minutes only): skip strong trend days early.
        try:
            open_move_limit = float(getattr(self.cfg, "opening_max_move_pct", 0.0) or 0.0)
        except Exception:
            open_move_limit = 0.0
        if bool(getattr(self.cfg, "enable_opening_filter", True)) and open_move_limit > 0:
            # Apply only during the early window.
            try:
                market_open = dt_time(9, 15)
                end_h = (9 * 60 + 15 + int(opening_minutes)) // 60
                end_m = (9 * 60 + 15 + int(opening_minutes)) % 60
                end_t = dt_time(end_h, end_m)
            except Exception:
                end_t = dt_time(10, 0)

            if market_open <= now_t <= end_t and candles:
                day_open = float(candles[0].open)
                spot_now = float(closes[-1])
                if day_open > 0:
                    open_move_pct = abs(spot_now - day_open) / day_open * 100.0
                    if open_move_pct > open_move_limit:
                        block("Strong opening trend day")
                        return
        fast_ema = ema(closes, period=ema_f_p)
        slow_ema = ema(closes, period=ema_s_p)
        rsi_val = rsi(closes, period=rsi_p)

        atr_val = atr(highs, lows, closes, period=atr_p)
        # Cache for dynamic pyramiding and other adaptive features.
        if atr_val is not None and float(atr_val) > 0:
            self._cached_atr = float(atr_val)



        # Feed ATR into IV percentile history for position sizing.
        if atr_val is not None and float(atr_val) > 0:
            self._iv_pct_hist.append(float(atr_val))
            if len(self._iv_pct_hist) > 200:
                self._iv_pct_hist = self._iv_pct_hist[-200:]

        # Profit Enhancements (ADX & Volume)
        adx_val = None
        if bool(getattr(self.cfg, "enable_adx_filter", True)):
            adx_p = int(getattr(self.cfg, "adx_period", 14))
            adx_val = adx(highs, lows, closes, period=adx_p)

        vol_sma = None
        current_vol = 0.0
        if bool(getattr(self.cfg, "enable_volume_filter", True)):
            try:
                volumes = [float(c.volume or 0.0) for c in candles]
                if volumes:
                    current_vol = volumes[-1]
                    vol_p = int(getattr(self.cfg, "volume_sma_period", 20))
                    vol_sma = sma(volumes, period=vol_p)
            except Exception:
                pass
        if atr_val is None:
            block("ATR could not be computed")
            return
        
        
        # Validate ATR against min/max thresholds (session-aware).
        if not (float(sess_min_atr) <= float(atr_val) <= float(sess_max_atr)):
            block(
                f"ATR {float(atr_val):.2f} out of range for {tf_str} "
                f"(session={sess_bucket}, min={sess_min_atr:.2f}, max={sess_max_atr:.2f})"
            )
            return

        try:
            cd_mult = float(getattr(self.cfg, "cooldown_atr_mult", 0.0) or 0.0)
        except Exception:
            cd_mult = 0.0
        if cd_mult > 0 and last_action_ts:
            try:
                base_cd = float(self.cfg.cooldown_sec)
            except Exception:
                base_cd = 0.0
            dyn_cd = base_cd + (float(atr_val) * cd_mult)
            if (time.time() - float(last_action_ts)) < float(dyn_cd):
                block(f"Adaptive cooldown active ({dyn_cd:.1f}s)")
                return

        try:
            max_range_mult = float(getattr(self.cfg, "entry_max_candle_range_atr_mult", 0.0) or 0.0)
        except Exception:
            max_range_mult = 0.0
        if max_range_mult > 0:
            try:
                last_range = float(highs[-1]) - float(lows[-1])
            except Exception:
                last_range = None
            if last_range is not None and float(atr_val) > 0 and last_range > (float(atr_val) * max_range_mult):
                block(f"Last candle range too large ({last_range:.2f} > {max_range_mult:.2f} * ATR)")
                return

        try:
            gap_mult = float(getattr(self.cfg, "entry_gap_atr_mult", 0.0) or 0.0)
        except Exception:
            gap_mult = 0.0
        if gap_mult > 0 and len(candles) >= 2:
            try:
                gap = abs(float(candles[-1].open) - float(candles[-2].close))
            except Exception:
                gap = None
            if gap is not None and float(atr_val) > 0 and gap > (float(atr_val) * gap_mult):
                block(f"Gap too large ({gap:.2f} > {gap_mult:.2f} * ATR)")
                return

        if fast_ema is None:
            block(f"Fast EMA not ready (p={ema_f_p}, n={len(closes)})")
            return
        if slow_ema is None:
            block(f"Slow EMA not ready (p={ema_s_p}, n={len(closes)})")
            return
        if rsi_val is None:
            block(f"RSI not ready (p={rsi_p}, n={len(closes)})")
            return

        # ---- ROC & Choppiness Filters ----
        roc_val = roc(closes, period=int(getattr(self.cfg, "roc_period", 14)))
        chop_val = choppiness_index(highs, lows, closes, period=int(getattr(self.cfg, "choppiness_period", 14)))

        enable_roc = bool(getattr(self.cfg, "enable_roc_filter", False))
        if str(getattr(self.cfg, "strategy_name", "")).strip().lower() == "auto" and bool(getattr(self.cfg, "gpt_enable", False)):
            enable_roc = False
        if enable_roc and roc_val is not None:
            if using_synth_candles:
                # Synthetic candles start with a flat backfill, which makes ROC ~0 and
                # incorrectly blocks entries when intraday endpoint is empty.
                self._log_throttled(
                    "Skipping ROC filter (synthetic candles in intraday-only mode)",
                    key="skip_roc_synth",
                    interval_sec=30.0,
                )
            else:
                roc_min = float(getattr(self.cfg, "roc_min_threshold", 0.05))
                if abs(roc_val) < roc_min:
                    block(f"ROC too low ({roc_val:.3f} < {roc_min})")
                    return

        enable_chop = bool(getattr(self.cfg, "enable_choppiness_filter", False))
        if str(getattr(self.cfg, "strategy_name", "")).strip().lower() == "auto" and bool(getattr(self.cfg, "gpt_enable", False)):
            enable_chop = False
        if enable_chop and chop_val is not None:
            chop_max = float(getattr(self.cfg, "choppiness_threshold", 61.8))
            if chop_val > chop_max:
                block(f"Choppy market ({chop_val:.1f} > {chop_max})")
                return

        # NOTE: Trend-strength is a range-bound filter intended for premium-selling entries.
        # Do not globally block all strategies here; enforce it in the relevant strategy branch.

        bullish_candle = is_bullish_engulfing(candles) or is_hammer(candles)
        bearish_candle = is_bearish_engulfing(candles) or is_shooting_star(candles)
        indecision = is_doji(candles)

        # Fetch spot and option chain independently: a failure in one should not wipe the other.
        # IMPORTANT: do not place new entries when spot LTP is unavailable or appears stale.
        try:
            spot_val = self._try_get_ltp(self.cfg.underlying)
            if spot_val is None:
                raise RuntimeError("Spot LTP returned None")
            spot = float(spot_val)
        except Exception as exc:  # noqa: BLE001
            print(f"[DATA] Spot LTP failed: {exc} | skipping entries")
            block("Spot LTP unavailable")
            return

        try:
            max_stale = float(getattr(self.cfg, "max_stale_ltp_sec", 0.0) or 0.0)
        except Exception:
            max_stale = 0.0
        if max_stale > 0:
            stale_for = self._seconds_since_ltp_change(self.cfg.underlying)
            if stale_for is not None and stale_for >= float(max_stale):
                print(f"[DATA] Spot LTP not refreshed for {stale_for:.1f}s (threshold={max_stale:.1f}s) | skipping entries")
                block("Spot LTP stale")
                return

        # Portfolio-level risk snapshot/caps (applies before opening any new trade).
        try:
            self._last_portfolio_risk = self._portfolio_risk_snapshot(float(spot))
        except Exception:
            self._last_portfolio_risk = {}
        try:
            max_notional = float(getattr(self.cfg, "max_portfolio_option_notional", 0.0) or 0.0)
        except Exception:
            max_notional = 0.0
        try:
            max_delta_abs = float(getattr(self.cfg, "max_portfolio_delta_abs", 0.0) or 0.0)
        except Exception:
            max_delta_abs = 0.0
        try:
            notional_abs = float((self._last_portfolio_risk or {}).get("option_notional_abs") or 0.0)
        except Exception:
            notional_abs = 0.0
        try:
            delta_abs_now = float((self._last_portfolio_risk or {}).get("delta_abs") or 0.0)
        except Exception:
            delta_abs_now = 0.0
        if max_notional > 0 and notional_abs >= max_notional:
            block(f"Portfolio option notional cap reached ({notional_abs:.0f} >= {max_notional:.0f})")
            return
        if max_delta_abs > 0 and delta_abs_now >= max_delta_abs:
            block(f"Portfolio abs-delta cap reached ({delta_abs_now:.1f} >= {max_delta_abs:.1f})")
            return

        try:
            chain = self.client.get_option_chain(self.cfg.underlying)
        except Exception as exc:  # noqa: BLE001
            print(f"[DATA] Option chain fetch failed: {exc}")
            chain = []

        # Target expiry (if configured) should restrict the chain before strike selection.
        # This applies even when weekly-only filtering is disabled.
        raw_override = str(getattr(self.cfg, "target_expiry", "") or "").strip()
        if raw_override and chain:
            override_exp = self._parse_any_date(raw_override)
            today = date.today()
            if override_exp is None:
                print(
                    f"[ENTRY] MSTOCK_TARGET_EXPIRY={raw_override!r} could not be parsed; "
                    "use DD-MM-YYYY (or ISO / DD-MMM-YYYY / DD-MM-YY). Ignoring override."
                )
            elif override_exp < today:
                print(
                    f"[ENTRY] MSTOCK_TARGET_EXPIRY={raw_override!r} is in the past; "
                    "ignoring override."
                )
            else:
                sym_root = self._normalize_root(self.cfg.symbol)
                under_root = self._normalize_root(self.cfg.underlying)
                valid_roots = {r for r in (sym_root, under_root) if r}
                
                # Debug logging of roots
                if bool(getattr(self.cfg, "debug_log_no_signal", False)):
                    print(f"[DEBUG] Target Expiry Filter: roots={valid_roots}, chain_len={len(chain)}")
                
                selected: List[dict] = []
                for row in chain:
                    if not isinstance(row, dict):
                        continue
                    root = self._normalize_root(str(row.get("symbol_root") or ""))
                    if not root:
                        sym_raw = str(row.get("symbol") or "").upper()
                        sym = sym_raw.split(":", 1)[-1]
                        if sym.startswith(sym_root):
                            root = sym_root
                        elif under_root and sym.startswith(under_root):
                            root = under_root
                        else:
                            root = ""
                    
                    if root not in valid_roots:
                        continue
                    
                    exp = self._row_expiry_date(row)
                    if exp is None:
                        continue
                    if exp == override_exp:
                        selected.append(row)

                if selected:
                    print(f"[ENTRY] Using target expiry: {override_exp} ({len(selected)} contracts)")
                    chain = selected
                else:
                    # Log more details about why it failed to match.
                    all_roots = sorted({self._normalize_root(str(r.get('symbol_root') or '')) for r in chain})
                    all_expiries = sorted({str(self._row_expiry_date(r) or 'N/A') for r in chain})
                    print(
                        f"[ENTRY] MSTOCK_TARGET_EXPIRY={raw_override!r} did not match any contracts. "
                        f"Target roots={valid_roots}. Found in chain: roots={all_roots}, expiries={all_expiries}. "
                        "Using auto expiry selection."
                    )

        # Premium-selling entry cutoff (no new premium after configured time).
        # If the cutoff is empty/unset, do not block premium entries.
        cutoff_raw = str(getattr(self.cfg, "premium_entry_cutoff_hhmm", "") or "").strip()
        if cutoff_raw:
            # If the user entered an invalid value, default to late-in-day (effectively disabled)
            # rather than accidentally blocking all premium entries.
            cutoff_t = self._parse_hhmm(cutoff_raw, default=dt_time(23, 59))
            # Do not block directional mode; only block premium-selling entries.
            # We handle this below by returning early when premium strategies are selected.
            premium_entry_blocked = now_t >= cutoff_t
        else:
            premium_entry_blocked = False

        # VWAP (used to filter short straddle entries).
        vwap_val = self._vwap(candles)

        # Global VWAP deviation filter (applies to all strategies when enabled).
        try:
            max_dev_all = float(getattr(self.cfg, "vwap_max_dev_pct", 0.0) or 0.0)
        except Exception:
            max_dev_all = 0.0
        if (
            bool(getattr(self.cfg, "enable_vwap_filter", True))
            and max_dev_all > 0
            and vwap_val is not None
            and float(vwap_val) != 0
        ):
            dev_pct = abs(float(spot) - float(vwap_val)) / abs(float(vwap_val)) * 100.0
            if dev_pct > max_dev_all:
                block("VWAP deviation too high")
                return

        if bool(getattr(self.cfg, "nifty_weekly_only", True)) and not bool(getattr(self.cfg, "bypass_weekly_filter", False)):
            filtered = self._filter_weekly_only(chain)
            if chain and not filtered:
                # If symbols don't match (e.g. underlying is NIFTY but symbol is BANKNIFTY),
                # the filter will naturally remove everything. Log this clearly.
                block(f"Weekly filter removed all contracts for {self.cfg.symbol.upper()}; "
                      f"check if MSTOCK_UNDERLYING ({self.cfg.underlying}) matches.")
            elif filtered:
                # If the weekly filter accidentally drops only one side (CE/PE)
                # due to missing expiry metadata on some rows, fall back to the
                # unfiltered chain so PE trades aren't artificially blocked.
                def _norm_ot(v: object) -> str:
                    s = str(v or "").strip().upper()
                    if s in {"CALL", "C"}:
                        return "CE"
                    if s in {"PUT", "P"}:
                        return "PE"
                    return s

                try:
                    orig_ce = sum(1 for o in chain if _norm_ot(o.get("option_type", "")) == "CE")
                    orig_pe = sum(1 for o in chain if _norm_ot(o.get("option_type", "")) == "PE")
                    flt_ce = sum(1 for o in filtered if _norm_ot(o.get("option_type", "")) == "CE")
                    flt_pe = sum(1 for o in filtered if _norm_ot(o.get("option_type", "")) == "PE")
                except Exception:
                    orig_ce = orig_pe = flt_ce = flt_pe = 0

                if (orig_ce > 0 and orig_pe > 0) and (flt_ce == 0 or flt_pe == 0):
                    block("Weekly filter dropped CE/PE side; using unfiltered chain")
                else:
                    chain = filtered

        call = self._select_atm_option(chain, spot, "CE") if chain else None
        put = self._select_atm_option(chain, spot, "PE") if chain else None
        iv_change_pct = self._note_iv_change_pct(self._current_atm_iv(call, put)) if chain else None

        if bool(getattr(self.cfg, "debug_log_no_signal", False)) and chain and (call is None or put is None):
            def _norm_ot(v: object) -> str:
                s = str(v or "").strip().upper()
                if s in {"CALL", "C"}:
                    return "CE"
                if s in {"PUT", "P"}:
                    return "PE"
                return s

            try:
                ce_n = sum(1 for o in chain if _norm_ot(o.get("option_type", "")) == "CE")
                pe_n = sum(1 for o in chain if _norm_ot(o.get("option_type", "")) == "PE")
                if call is None and put is not None:
                    print(f"[CHAIN] ATM CE missing; chain_counts(norm): CE={ce_n} PE={pe_n}")
                elif put is None and call is not None:
                    print(f"[CHAIN] ATM PE missing; chain_counts(norm): CE={ce_n} PE={pe_n}")
                elif call is None and put is None:
                    print(f"[CHAIN] ATM CE+PE missing; chain_counts(norm): CE={ce_n} PE={pe_n}")
            except Exception:
                pass

        # Decide which strategy branch to run.
        # - Explicit modes: directional / short_straddle / short_strangle / iron_condor
        #   plus long_straddle / long_strangle for long volatility structures
        # - Auto mode: choose between directional vs a range-bound multi-leg strategy
        #   based on regime (trend/RSI) and ATR.
        strat_raw = (self.cfg.strategy_name or "directional")
        strat = self._normalize_strategy_name(str(strat_raw))
        effective_strat = strat
        self._clear_auto_gpt_strike_context()

        # Announce alias mapping at most once per change.
        try:
            raw_s = str(strat_raw).strip().lower()
            eff_s = str(effective_strat).strip().lower()
            if raw_s and eff_s and raw_s != eff_s:
                if self._last_strategy_name_raw != raw_s or self._last_strategy_name_effective != eff_s:
                    if raw_s in {"short_premium", "long_premium"}:
                        print(f"[STRAT] Using deprecated alias -> {eff_s}")
                    else:
                        print(f"[STRAT] Using alias: {raw_s} -> {eff_s}")
                    self._last_strategy_name_raw = raw_s
                    self._last_strategy_name_effective = eff_s
        except Exception:
            pass
        auto_ts_limit = None
        if strat == "auto":
            trend_strength = self._trend_strength(float(fast_ema), float(slow_ema), float(closes[-1]))
            rsi_low = float(sess_rsi_low)
            rsi_high = float(sess_rsi_high)
            try:
                auto_mult = float(getattr(self.cfg, "auto_range_trend_mult", 2.0) or 2.0)
            except Exception:
                auto_mult = 2.0
            ts_limit = float(self.cfg.max_trend_strength) * max(1.0, float(auto_mult))
            auto_ts_limit = float(ts_limit)

            # Auto regime selection should respect the premium RSI filter toggle.
            # If premium RSI filtering is disabled, do not force a narrow RSI band
            # for choosing premium strategies.
            use_rsi_for_range = bool(getattr(self.cfg, "enable_premium_rsi_filter", True))
            if use_rsi_for_range:
                range_bound = (
                    trend_strength <= float(ts_limit)
                    and (rsi_val is not None)
                    and (rsi_low <= float(rsi_val) <= rsi_high)
                )
            else:
                range_bound = trend_strength <= float(ts_limit)

            try:
                if trend_strength > float(ts_limit):
                    auto_regime = "trending"
                elif float(atr_val) >= float(getattr(self.cfg, "straddle_atr_threshold", 0.0) or 0.0) and float(getattr(self.cfg, "straddle_atr_threshold", 0.0) or 0.0) > 0:
                    auto_regime = "volatile"
                elif rsi_val is not None and abs(float(rsi_val) - 50.0) < 8:
                    auto_regime = "mean_reverting"
                else:
                    auto_regime = "quiet"
            except Exception:
                auto_regime = "quiet"
            regime = auto_regime
            regime_profile = get_regime_tuning(regime, self.cfg)

            router = self._strategy_router(
                chain_available=bool(chain),
                range_bound=bool(range_bound),
                trend_strength=float(trend_strength),
                ts_limit=float(ts_limit) * float(regime_profile.get("trend_mult", 1.0) if isinstance(regime_profile, dict) else 1.0),
                atr_val=float(atr_val),
                atr_threshold=float(getattr(self.cfg, "straddle_atr_threshold", 0.0) or 0.0),
                fast_ema=float(fast_ema) if fast_ema is not None else None,
                slow_ema=float(slow_ema) if slow_ema is not None else None,
                regime=regime,
                regime_profile=regime_profile,
            )
            try:
                self._note_strategy_considered([str(x) for x in list(router.get("candidates") or [])])
            except Exception:
                pass

            # AUTO mode prefers GPT strategy selection if active.
            try:
                gpt_rec = self._gpt_auto_select_strategy(
                    chain=list(chain or []),
                    spot=float(spot),
                    atr_val=float(atr_val),
                    rsi_val=float(rsi_val) if rsi_val is not None else None,
                    trend_strength=float(trend_strength),
                    vwap_val=float(vwap_val) if vwap_val is not None else None,
                    call_atm=call,
                    put_atm=put,
                )
            except Exception:
                gpt_rec = None

            if gpt_rec:
                try:
                    effective_strat = str(self._normalize_strategy_name(str(gpt_rec)))
                except Exception:
                    effective_strat = str(gpt_rec)
                try:
                    self._last_auto_fallback_note = ""
                except Exception:
                    pass
                self._last_router_snapshot = {
                    "source": "gpt",
                    "range_bound": bool(range_bound),
                    "trend_strength": float(trend_strength),
                    "ts_limit": float(ts_limit),
                    "atr": float(atr_val),
                    "selected": str(effective_strat),
                    "candidates": list(router.get("candidates") or []),
                }
            else:
                # No GPT recommendation.
                # If we require a GPT recommendation, block entry here!
                if bool(getattr(self.cfg, "gpt_enable", False)) and bool(getattr(self.cfg, "gpt_auto_select", True)) and bool(getattr(self.cfg, "gpt_require_recommendation", False)):
                    try:
                        note = "gpt_rec_required_but_missing"
                        if note != str(getattr(self, "_last_auto_fallback_note", "") or ""):
                            print("[AUTO] GPT recommendation required but missing; skipping entry")
                            self._last_auto_fallback_note = str(note)
                    except Exception:
                        pass
                    self._last_router_snapshot = {
                        "source": "gpt_missing",
                        "range_bound": bool(range_bound),
                        "trend_strength": float(trend_strength),
                        "ts_limit": float(ts_limit),
                        "atr": float(atr_val),
                        "selected": None,
                        "candidates": list(router.get("candidates") or []),
                    }
                    return

                try:
                    fallback = router.get("selected") or (list(router.get("candidates") or [])[:1] or [None])[0]
                except Exception:
                    fallback = None

                if not fallback:
                    try:
                        note = "router_missing"
                        if note != str(getattr(self, "_last_auto_fallback_note", "") or ""):
                            print("[AUTO] No active trade signals found (router: no signal); skipping entry")
                            self._last_auto_fallback_note = str(note)
                    except Exception:
                        pass
                    self._last_router_snapshot = {
                        "source": "router_missing",
                        "range_bound": bool(range_bound),
                        "trend_strength": float(trend_strength),
                        "ts_limit": float(ts_limit),
                        "atr": float(atr_val),
                        "selected": None,
                        "candidates": list(router.get("candidates") or []),
                    }
                    return

                try:
                    effective_strat = str(self._normalize_strategy_name(str(fallback)))
                except Exception:
                    effective_strat = str(fallback)
                try:
                    self._last_auto_fallback_note = ""
                except Exception:
                    pass
                self._last_router_snapshot = {
                    "source": "router",
                    "range_bound": bool(range_bound),
                    "trend_strength": float(trend_strength),
                    "ts_limit": float(ts_limit),
                    "atr": float(atr_val),
                    "selected": str(effective_strat),
                    "candidates": list(router.get("candidates") or []),
                }

            # Optional hysteresis: lock the auto-selected strategy for N minutes,
            # and never change it while positions are open.
            try:
                lock_min = int(getattr(self.cfg, "auto_strategy_lock_minutes", 0) or 0)
            except Exception:
                lock_min = 0
            if lock_min > 0:
                now_ts = float(time.time())
                open_count = len(self.state.open_directional) + len(self.state.open_multi)

                # When positions are open, stick to the last locked strategy.
                if open_count > 0 and str(getattr(self, "_auto_strat_locked", "") or ""):
                    effective_strat = str(self._auto_strat_locked)
                else:
                    if str(getattr(self, "_auto_strat_locked", "") or ""):
                        if (now_ts - float(self._auto_strat_locked_ts)) < float(lock_min) * 60.0:
                            effective_strat = str(self._auto_strat_locked)
                        else:
                            self._auto_strat_locked = str(effective_strat)
                            self._auto_strat_locked_ts = now_ts
                    else:
                        self._auto_strat_locked = str(effective_strat)
                        self._auto_strat_locked_ts = now_ts

            if bool(getattr(self.cfg, "debug_log_no_signal", False)):
                rsi_dbg = f"{float(rsi_val):.1f}" if rsi_val is not None else "N/A"
                print(
                    f"[AUTO] selected={effective_strat} range_bound={'yes' if bool(range_bound) else 'no'} "
                    f"trend_strength={float(trend_strength):.6f} ts_limit={float(ts_limit):.6f} "
                    f"rsi={rsi_dbg} rsi_band={float(rsi_low):.1f}-{float(rsi_high):.1f} "
                    f"use_rsi={'yes' if bool(use_rsi_for_range) else 'no'} atr={float(atr_val):.2f} "
                    f"chain={'yes' if bool(chain) else 'no'}"
                )
        else:
            try:
                self._note_strategy_considered([str(effective_strat)])
                self._last_router_snapshot = {
                    "source": "explicit",
                    "selected": str(effective_strat),
                    "candidates": [str(effective_strat)],
                }
            except Exception:
                pass

        override_name = self._resolve_auto_strategy_candidate(strategy_override)
        if override_name:
            effective_strat = str(override_name)
            try:
                current_candidates = list((self._last_router_snapshot or {}).get("candidates") or [])
            except Exception:
                current_candidates = []
            if override_name not in current_candidates:
                current_candidates.insert(0, str(override_name))
            self._last_router_snapshot = {
                "source": "reroute_override",
                "selected": str(effective_strat),
                "candidates": current_candidates or [str(effective_strat)],
            }

        # Mark whether the upcoming entry was selected by GPT so entry gates
        # can behave accordingly. This is a transient flag cleared on next cycle.
        try:
            src = str((getattr(self, "_last_router_snapshot", {}) or {}).get("source") or "")
            setattr(self, "_current_entry_recommended_by_gpt", True if src == "gpt" else False)
        except Exception:
            try:
                setattr(self, "_current_entry_recommended_by_gpt", False)
            except Exception:
                pass

        # Explicit directional variants: allow selecting the exact option style
        # while still using the directional signal engine to decide bull vs bear.
        self._note_strategy_decision(str(effective_strat))
        forced_directional_name: Optional[str] = None
        if effective_strat in {"long_call", "long_put", "short_call", "short_put"}:
            forced_directional_name = str(effective_strat)
            effective_strat = "directional"

        if strat == "auto":
            try:
                selected_for_ctx = str(forced_directional_name or effective_strat or "").strip().lower()
                if selected_for_ctx:
                    self._set_auto_gpt_strike_context(
                        selected_strategy=selected_for_ctx,
                        chain=list(chain or []),
                        spot=float(spot),
                    )
            except Exception:
                self._clear_auto_gpt_strike_context()

        if iv_change_pct is not None:
            try:
                iv_expand = float(getattr(self.cfg, "iv_expand_threshold_pct", 0.0) or 0.0)
            except Exception:
                iv_expand = 0.0
            try:
                iv_contract = float(getattr(self.cfg, "iv_contract_threshold_pct", 0.0) or 0.0)
            except Exception:
                iv_contract = 0.0
            short_iv_strats = {
                "short_straddle",
                "short_strangle",
                "bull_put_spread",
                "iron_condor",
                "iron_fly",
                "iron_butterfly",
                "delta_hedged_short_straddle",
            }
            long_iv_strats = {
                "long_straddle",
                "long_strangle",
                "bull_call_spread",
                "call_ratio_backspread",
                "put_ratio_backspread",
                "delta_hedged_long_straddle",
            }
            if iv_expand > 0 and effective_strat in short_iv_strats and float(iv_change_pct) >= iv_expand:
                reason = f"IV expanding too fast ({iv_change_pct:.2f}% >= {iv_expand:.2f}%)"
                if maybe_reroute_entry_block(reason, current_strategy=str(effective_strat)):
                    return
                return
            if iv_contract > 0 and effective_strat in long_iv_strats and float(iv_change_pct) <= -iv_contract:
                reason = f"IV collapsing too fast ({iv_change_pct:.2f}% <= -{iv_contract:.2f}%)"
                if maybe_reroute_entry_block(reason, current_strategy=str(effective_strat)):
                    return
                return

        # ---- Multi-leg strategies ----
        if effective_strat in {
            "short_straddle",
            "short_strangle",
            "bull_call_spread",
            "bull_put_spread",
            "call_ratio_backspread",
            "put_ratio_backspread",
            "iron_condor",
            "iron_fly",
            "iron_butterfly",
            "long_straddle",
            "long_strangle",
            "delta_hedged_short_straddle",
            "delta_hedged_long_straddle",
        }:
            if premium_entry_blocked and effective_strat in {
                "short_straddle",
                "short_strangle",
                "bull_put_spread",
                "iron_condor",
                "iron_fly",
                "iron_butterfly",
                "delta_hedged_short_straddle",
            }:
                reason = "Premium entries disabled after cutoff time"
                if maybe_reroute_entry_block(reason, current_strategy=str(effective_strat)):
                    return
                return

            if not chain:
                reason = "Option chain empty"
                if maybe_reroute_entry_block(reason, current_strategy=str(effective_strat)):
                    return
                return

            if indecision:
                # doji is fine for range-bound strategies, don't early-return
                pass

            if fast_ema is None or slow_ema is None:
                reason = "EMA not ready"
                if maybe_reroute_entry_block(reason, current_strategy=str(effective_strat)):
                    return
                return

            trend_strength = self._trend_strength(float(fast_ema), float(slow_ema), float(closes[-1]))
            if effective_strat in {
                "short_straddle",
                "short_strangle",
                "bull_put_spread",
                "iron_condor",
                "iron_fly",
                "iron_butterfly",
            }:
                ts_limit = float(auto_ts_limit) if (strat == "auto" and auto_ts_limit is not None) else float(self.cfg.max_trend_strength)
                enable_tf = bool(getattr(self.cfg, "enable_trend_filter", True))
                if getattr(self, "_current_entry_recommended_by_gpt", False):
                    enable_tf = False
                if enable_tf and trend_strength > float(ts_limit):
                    reason = "Trend strength too high"
                    if maybe_reroute_entry_block(reason, current_strategy=str(effective_strat)):
                        return
                    return

            # Require RSI near the middle (basic range-bound filter).
            # NOTE: This is a *premium-selling* guard; long-volatility structures shouldn't be blocked by it.
            if effective_strat in {
                "short_straddle",
                "short_strangle",
                "bull_put_spread",
                "iron_condor",
                "iron_fly",
                "iron_butterfly",
                "delta_hedged_short_straddle",
            }:
                rsi_low = float(sess_rsi_low)
                rsi_high = float(sess_rsi_high)
                enable_pr = bool(getattr(self.cfg, "enable_premium_rsi_filter", True))
                if getattr(self, "_current_entry_recommended_by_gpt", False):
                    enable_pr = False
                if enable_pr:
                    if rsi_val is None or not (rsi_low <= float(rsi_val) <= rsi_high):
                        # The premium RSI filter only blocks in auto mode where rerouting can find
                        # an alternative strategy. In explicit mode the filter is advisory - the
                        # user chose this strategy deliberately, so let the trade proceed.
                        if str(getattr(self.cfg, "strategy_name", "") or "").strip().lower() == "auto":
                            reason = f"RSI {float(rsi_val) if rsi_val is not None else 'N/A'} out of range"
                            if maybe_reroute_entry_block(reason, current_strategy=str(effective_strat)):
                                return

            if effective_strat == "short_straddle":
                try:
                    max_dev = float(getattr(self.cfg, "vwap_max_dev_pct", 0.0) or 0.0)
                except Exception:
                    max_dev = 0.0
                enable_vwap = bool(getattr(self.cfg, "enable_vwap_filter", True))
                if getattr(self, "_current_entry_recommended_by_gpt", False):
                    enable_vwap = False
                if (
                    enable_vwap
                    and max_dev > 0
                    and vwap_val is not None
                    and float(vwap_val) != 0
                ):
                    dev_pct = abs(float(spot) - float(vwap_val)) / abs(float(vwap_val)) * 100.0
                    if dev_pct > max_dev:
                        reason = "VWAP deviation too high"
                        if maybe_reroute_entry_block(reason, current_strategy=str(effective_strat)):
                            return
                        return
                self._enter_short_straddle(chain, spot, atr_val=float(atr_val))
                return
            if effective_strat == "delta_hedged_short_straddle":
                try:
                    max_dev = float(getattr(self.cfg, "vwap_max_dev_pct", 0.0) or 0.0)
                except Exception:
                    max_dev = 0.0
                enable_vwap = bool(getattr(self.cfg, "enable_vwap_filter", True))
                if getattr(self, "_current_entry_recommended_by_gpt", False):
                    enable_vwap = False
                if (
                    enable_vwap
                    and max_dev > 0
                    and vwap_val is not None
                    and float(vwap_val) != 0
                ):
                    dev_pct = abs(float(spot) - float(vwap_val)) / abs(float(vwap_val)) * 100.0
                    if dev_pct > max_dev:
                        reason = "VWAP deviation too high"
                        if maybe_reroute_entry_block(reason, current_strategy=str(effective_strat)):
                            return
                        return
                self._enter_delta_hedged_short_straddle(chain, spot, atr_val=float(atr_val))
                return
            if effective_strat == "short_strangle":
                self._enter_short_strangle(chain, spot, atr_val=float(atr_val))
                return
            if effective_strat == "bull_call_spread":
                self._enter_bull_call_spread(chain, spot, atr_val=float(atr_val))
                return
            if effective_strat == "bull_put_spread":
                self._enter_bull_put_spread(chain, spot, atr_val=float(atr_val))
                return
            if effective_strat == "call_ratio_backspread":
                self._enter_call_ratio_backspread(chain, spot, atr_val=float(atr_val))
                return
            if effective_strat == "put_ratio_backspread":
                self._enter_put_ratio_backspread(chain, spot, atr_val=float(atr_val))
                return
            if effective_strat == "iron_condor":
                self._enter_iron_condor(chain, spot, atr_val=float(atr_val))
                return
            if effective_strat == "iron_fly":
                self._enter_iron_fly(chain, spot, atr_val=float(atr_val))
                return
            if effective_strat == "iron_butterfly":
                self._enter_iron_butterfly(chain, spot, atr_val=float(atr_val))
                return
            if effective_strat == "long_straddle":
                self._enter_long_straddle(chain, spot, atr_val=float(atr_val))
                return
            if effective_strat == "delta_hedged_long_straddle":
                self._enter_delta_hedged_long_straddle(chain, spot, atr_val=float(atr_val))
                return
            if effective_strat == "long_strangle":
                self._enter_long_strangle(chain, spot, atr_val=float(atr_val))
                return

        # Example: assume 1 day to expiry and 6% risk-free; adapt as needed.
        time_to_expiry = 1.0 / 252.0
        rate = 0.06

        # Auto-mode directional-lite entry: if auto chose "directional", use EMA+RSI to actually trade.
        # Also handles explicit "directional" strategy mode.
        if effective_strat == "directional":
            # Directional entry quality knobs (reduce churn)
            try:
                min_score = int(getattr(self.cfg, "dir_min_confirmations", 4) or 4)
            except Exception:
                min_score = 4
            if min_score < 1:
                min_score = 1
            try:
                min_diff = int(getattr(self.cfg, "dir_min_score_diff", 1) or 1)
            except Exception:
                min_diff = 1
            if min_diff < 0:
                min_diff = 0
            allow_tie = bool(getattr(self.cfg, "dir_allow_tie_break_entries", False))
            try:
                mom_mult = float(getattr(self.cfg, "dir_momentum_atr_mult", 0.10) or 0.10)
            except Exception:
                mom_mult = 0.10
            if mom_mult < 0:
                mom_mult = 0.0

            if not chain:
                reason = "Option chain empty (directional requires ATM option)"
                if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                    return
                return
            if call is None and put is None:
                reason = "ATM options missing from chain"
                if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                    return
                return
            # ADX Filter
            enable_adx = bool(getattr(self.cfg, "enable_adx_filter", True))
            if getattr(self, "_current_entry_recommended_by_gpt", False):
                enable_adx = False
            if enable_adx:
                if using_synth_candles:
                    # Synthetic candles can be flat/backfilled; ADX becomes meaningless and
                    # can falsely block all entries.
                    if bool(getattr(self.cfg, "debug_log_no_signal", False)):
                        print("[DATA] Skipping ADX filter (synthetic candles in intraday-only mode)")
                else:
                    min_adx = float(getattr(self.cfg, "adx_min_strength", 25.0) or 25.0)
                    if adx_val is not None and float(adx_val) < min_adx:
                        reason = f"ADX {float(adx_val):.2f} < {min_adx} (Choppy Market)"
                        if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                            return
                        return

            # Volume Filter
            enable_vol = bool(getattr(self.cfg, "enable_volume_filter", True))
            if getattr(self, "_current_entry_recommended_by_gpt", False):
                enable_vol = False
            if enable_vol:
                if vol_sma is not None and current_vol > 0:
                    if current_vol < float(vol_sma):
                        reason = f"Volume {int(current_vol)} < SMA {int(vol_sma)} (Low Activity)"
                        if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                            return
                        return

            try:
                deferred_eval_res = self.evaluate_entry_signals(candles)
                if isinstance(deferred_eval_res, dict) and bool(deferred_eval_res.get("take")):
                    tr = self._simulate_or_place_basic_directional(
                        take=True,
                        size=int(deferred_eval_res.get("size") or 0),
                        reason=str(deferred_eval_res.get("reason") or "ml_take"),
                        candles=candles,
                    )
                    if tr is not None:
                        return
            except Exception:
                pass

            # Supertrend Filter
            st_val = None
            if bool(getattr(self.cfg, "enable_supertrend_filter", True)):
                st_period = int(getattr(self.cfg, "supertrend_period", 10))
                st_mult = float(getattr(self.cfg, "supertrend_multiplier", 3.0))
                st_val = supertrend(highs, lows, closes, period=st_period, multiplier=st_mult)
                if st_val is not None:
                    curr_price = float(closes[-1])
                    # Log for debugging
                    if bool(getattr(self.cfg, "debug_log_no_signal", False)):
                        trend_dir = "UP" if curr_price > st_val else "DOWN"
                        print(f"[SUPERTREND] Price={curr_price:.2f} ST={st_val:.2f} Trend={trend_dir}")

            # Volatility regime for directional entries (STRICT volatility-based selection):
            # - If ATR is increasing in order => prefer LONG options (long_call/long_put)
            # - If ATR is decreasing in order => prefer SHORT options (short_put/short_call)
            # - Otherwise (no clean monotonic trend): fall back to ATR level
            #   (low ATR => short, high ATR => long)
            try:
                atr_period = int(getattr(self.cfg, "atr_period", 14) or 14)
            except Exception:
                atr_period = 14
            try:
                vol_steps = int(getattr(self.cfg, "dir_vol_decreasing_steps", 3) or 3)
            except Exception:
                vol_steps = 3
            vol_decreasing = self._is_atr_decreasing(highs, lows, closes, period=atr_period, steps=vol_steps)
            vol_increasing = self._is_atr_increasing(highs, lows, closes, period=atr_period, steps=vol_steps)

            # Directional long-vs-short style.
            # NOTE: This uses _resolve_directional_trade_style() so that:
            # - explicit cfg.directional_trade_style is honored
            # - cfg.directional_style_lock_minutes is honored
            # We still preserve the earlier ATR-level fallback for "prefer short" when ATR is low.
            try:
                atr_level_thr = float(getattr(self.cfg, "straddle_atr_threshold", 0.0) or 0.0)
            except Exception:
                atr_level_thr = 0.0
            prefer_short = bool(vol_decreasing) or (
                (not bool(vol_increasing))
                and float(atr_level_thr) > 0.0
                and float(atr_val) <= float(atr_level_thr)
            )
            dir_style = self._resolve_directional_trade_style(vol_decreasing=bool(prefer_short))

            # Normal scoring/decision continues.

            try:
                atr_level_thr_dbg = float(getattr(self.cfg, "straddle_atr_threshold", 0.0) or 0.0)
            except Exception:
                atr_level_thr_dbg = 0.0
            if bool(getattr(self.cfg, "debug_log_no_signal", False)):
                print(
                    f"[DIR STYLE] style={dir_style} trade_style_cfg={str(getattr(self.cfg, 'directional_trade_style', 'auto') or 'auto')} "
                    f"atr_inc={'yes' if bool(vol_increasing) else 'no'} atr_dec={'yes' if bool(vol_decreasing) else 'no'} "
                    f"(steps={vol_steps}, period={atr_period}, atr_now={float(atr_val):.2f}, level_thr={atr_level_thr_dbg:.2f}, prefer_short={'yes' if bool(prefer_short) else 'no'})"
                )

            try:
                rsi_buy = float(getattr(self.cfg, "auto_dir_rsi_buy", 52.0) or 52.0)
            except Exception:
                rsi_buy = 52.0
            try:
                rsi_sell = float(getattr(self.cfg, "auto_dir_rsi_sell", 48.0) or 48.0)
            except Exception:
                rsi_sell = 48.0
            try:
                rsi_slop = float(getattr(self.cfg, "auto_dir_rsi_slop", 1.0) or 1.0)
            except Exception:
                rsi_slop = 1.0

            # Combined signals: EMA trend + spot move + RSI band + candle pattern.
            # We decide CE vs PE based on which side has the stronger confirmation.
            bull_score = 0
            bear_score = 0

            ema_up = bool(fast_ema > slow_ema)
            ema_down = bool(fast_ema < slow_ema)

            # EMA slope vote (directional): requires meaningful slope vs ATR.
            try:
                slope_mult = float(getattr(self.cfg, "dir_ema_slope_atr_mult", 0.0) or 0.0)
            except Exception:
                slope_mult = 0.0
            ema_slope = None
            if slope_mult > 0:
                try:
                    ema_prev = ema(closes[:-1], period=ema_f_p) if len(closes) > ema_f_p else None
                    if ema_prev is not None:
                        ema_slope = float(fast_ema) - float(ema_prev)
                except Exception:
                    ema_slope = None
            if ema_slope is not None and atr_val is not None and float(atr_val) > 0 and slope_mult > 0:
                slope_thr = float(atr_val) * slope_mult
                if ema_slope > slope_thr:
                    bull_score += 1
                elif ema_slope < -slope_thr:
                    bear_score += 1
                if bool(getattr(self.cfg, "debug_log_no_signal", False)):
                    print(f"[EMA SLOPE] slope={ema_slope:.3f} thr={slope_thr:.3f} votes=({bull_score},{bear_score})")

            # 1) Trend direction
            if ema_up:
                bull_score += 1
            elif ema_down:
                bear_score += 1

            # 2) Candle pattern confirmation
            if bullish_candle:
                bull_score += 1
            if bearish_candle:
                bear_score += 1

            # 3) Spot move over last few minutes (filter noise with ATR if available)
            delta_spot = None
            try:
                lookback = 6
                if len(closes) >= lookback:
                    delta_spot = float(closes[-1]) - float(closes[-lookback])
                    thr = 0.0
                    try:
                        if atr_val is not None:
                            thr = float(mom_mult) * float(atr_val)
                    except Exception:
                        thr = 0.0
                    if delta_spot >= thr:
                        bull_score += 1
                    elif delta_spot <= -thr:
                        bear_score += 1
            except Exception:
                pass

            # 4) RSI confirmation - use EITHER boundary OR midline, not both
            try:
                r = float(rsi_val) if rsi_val is not None else None
            except Exception:
                r = None
            if r is not None:
                # Strong directional signals take priority
                rsi_buy_line = float(rsi_buy) - float(rsi_slop)
                rsi_sell_line = float(rsi_sell) + float(rsi_slop)
                
                if r >= rsi_buy:
                    # Very strong bullish RSI
                    bull_score += 2
                elif r >= rsi_buy_line:
                    # Strong bullish RSI
                    bull_score += 1
                elif r <= rsi_sell:
                    # Very strong bearish RSI
                    bear_score += 2
                elif r <= rsi_sell_line:
                    # Strong bearish RSI
                    bear_score += 1
                else:
                    # Weak signal - use midline position for subtle bias only
                    midline_threshold = 2.0  # require stronger deviation from 50
                    if r > (50.0 + midline_threshold):
                        bull_score += 1
                    elif r < (50.0 - midline_threshold):
                        bear_score += 1
                    # Otherwise neutral (no points for either side)

            # Decision:
            # - Prefer EMA trend agreement, but allow counter-trend trades with strong confirmation
            # - Allow ties when trend + spot-move agree (common in smooth downtrends/uptrends).
            ema_neutral = not ema_up and not ema_down
            
            # Debug logging
            if bool(getattr(self.cfg, "debug_log_no_signal", False)):
                rsi_dbg = "N/A" if rsi_val is None else f"{float(rsi_val):.1f}"
                ema_diff = float(fast_ema) - float(slow_ema)
                print(f"[DIRECTIONAL SCORES] EMA9={float(fast_ema):.2f} EMA21={float(slow_ema):.2f} Diff={ema_diff:.2f} "
                      f"RSI={rsi_dbg} Bull={bull_score} Bear={bear_score} EMA_Up={ema_up} EMA_Down={ema_down} EMA_Neutral={ema_neutral}")

            # Supertrend alignment for directional entries.
            # NOTE: This is configured as COUNTER-TREND gating per user preference:
            # - Supertrend UP  (price > ST)  => allow only BEARISH directional entries (block bullish)
            # - Supertrend DOWN(price < ST)  => allow only BULLISH directional entries (block bearish)
            st_dir = None
            curr_price = None
            if st_val is not None and bool(getattr(self.cfg, "enable_supertrend_filter", True)):
                try:
                    curr_price = float(closes[-1])
                    st_dir = "UP" if curr_price > float(st_val) else "DOWN"
                except Exception:
                    st_dir = None
            if bool(getattr(self.cfg, "debug_log_no_signal", False)) and st_dir is not None and curr_price is not None:
                print(f"[SUPERTREND] Dir={st_dir} Price={curr_price:.2f} ST={float(st_val):.2f}")
            
            # Check for strong bearish signal.
            # Allow counter-trend trades only when the score advantage is decisive.
            bearish_signal = bear_score >= int(min_score) and (
                bear_score > bull_score
                or (bear_score == bull_score and (delta_spot is not None and float(delta_spot) < 0))
            )
            bearish_ok = bool(ema_down) or bool(bear_score >= (bull_score + 2))

            # Check for strong bullish signal.
            # Allow counter-trend trades only when the score advantage is decisive.
            bullish_signal = bull_score >= int(min_score) and (
                bull_score > bear_score
                or (bull_score == bear_score and (delta_spot is not None and float(delta_spot) > 0))
            )

            # 5) Supertrend Scoring (Voting)
            # Incorporate Supertrend as a vote rather than a hard gate.
            if st_val is not None:
                # Treat Supertrend as a *tiebreaker* only: apply its vote only when
                # the current directional scores are close.
                try:
                    score_diff = abs(int(bull_score) - int(bear_score))
                except Exception:
                    score_diff = 999
                if score_diff > 1:
                    if bool(getattr(self.cfg, "debug_log_no_signal", False)):
                        print(f"[SUPERTREND VOTE] Skipped (score_diff={score_diff} > 1)")
                else:
                    st_mode = str(getattr(self.cfg, "supertrend_mode", "counter") or "counter").strip().lower()
                    # Determine which direction Supertrend supports
                    st_bullish_signal = False
                    st_bearish_signal = False

                    if st_dir == "UP":
                        # Price > ST
                        if st_mode == "trend":
                            st_bullish_signal = True
                        else:
                            st_bearish_signal = True
                    elif st_dir == "DOWN":
                        # Price < ST
                        if st_mode == "trend":
                            st_bearish_signal = True
                        else:
                            st_bullish_signal = True

                    # Get the vote weight for Supertrend (how many votes it contributes)
                    st_vote_weight = int(getattr(self.cfg, "supertrend_vote_weight", 1) or 1)

                    if st_bullish_signal:
                        bull_score += st_vote_weight
                    if st_bearish_signal:
                        bear_score += st_vote_weight

                    if bool(getattr(self.cfg, "debug_log_no_signal", False)):
                        print(f"[SUPERTREND VOTE] Mode={st_mode} ST={st_dir} Weight={st_vote_weight} -> Bull+={int(st_bullish_signal)*st_vote_weight} Bear+={int(st_bearish_signal)*st_vote_weight}")


            # 4.1) MTF Confirmation (reuse the MTF snapshot computed earlier in this cycle)
            if bool(getattr(self.cfg, "enable_mtf_confirmation", False)):
                if mtf_ema_up:
                    bull_score += 1
                if mtf_ema_down:
                    bear_score += 1

            # 4.2) ROC Filter
            if bool(getattr(self.cfg, "enable_roc_filter", False)):
                roc_p = int(getattr(self.cfg, "roc_period", 14))
                roc_thr = float(getattr(self.cfg, "roc_min_threshold", 0.0))
                r_val = roc(closes, period=roc_p)
                if r_val is not None:
                    if r_val > roc_thr: bull_score += 1
                    elif r_val < -roc_thr: bear_score += 1

            # 4.3) Choppiness Filter
            enable_chop = bool(getattr(self.cfg, "enable_choppiness_filter", False))
            if getattr(self, "_current_entry_recommended_by_gpt", False):
                enable_chop = False
            if enable_chop:
                chop_p = int(getattr(self.cfg, "choppiness_period", 14))
                chop_thr = float(getattr(self.cfg, "choppiness_threshold", 61.8))
                c_val = choppiness_index(highs, lows, closes, period=chop_p)
                if c_val is not None and c_val > chop_thr:
                    if getattr(self.cfg, "choppiness_hard_filter", False):
                        reason = f"Choppiness {c_val:.2f} > {chop_thr} (Hard Filter)"
                        if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                            return
                        return
                    else:
                        # Choppy -> penalize both scores
                        bull_score = max(0, bull_score - 1)
                        bear_score = max(0, bear_score - 1)

            # Apply IV percentile strategy bias to scores
            try:
                iv_biased = bool(getattr(self.cfg, "iv_filter_enabled", False))
            except Exception:
                iv_biased = False
            if iv_biased:
                iv_strat_bias = self._iv_percentile_strategy_bias()
                if iv_strat_bias == "short_premium":
                    # High IV favors premium selling - reduce directional aggression
                    bull_score = max(0, bull_score - 1)
                    bear_score = max(0, bear_score - 1)
                elif iv_strat_bias == "long_premium":
                    # Low IV favors premium buying - slight directional boost
                    bull_score = bull_score + 1
                    bear_score = bear_score + 1

            # Re-evaluate signals with new scores
            # Win-rate tracker check for strategy
            if getattr(self.cfg, "winrate_tracker_enabled", False):
                try:
                    strat_check = str(forced_directional_name or effective_strat or "").strip().lower()
                    if strat_check and strat_check not in {"auto", "directional"}:
                        if not self._strategy_winrate_check(strat_check):
                            if maybe_reroute_entry_block(f"Winrate filter blocked {strat_check}", current_strategy=str(forced_directional_name or effective_strat)):
                                return
                            return
                except Exception:
                    pass

            # Handle tied scores with comprehensive tiebreaker logic
            if bull_score == bear_score and bull_score >= int(min_score):
                # Tiebreaker hierarchy when scores are equal
                if not bool(allow_tie):
                    bullish_signal = False
                    bearish_signal = False
                    bullish_ok = False
                    bearish_ok = False
                elif delta_spot is not None and abs(float(delta_spot)) > 0.01:
                    # Use price momentum as primary tiebreaker
                    bullish_signal = float(delta_spot) > 0
                    bearish_signal = float(delta_spot) < 0
                    if bool(getattr(self.cfg, "debug_log_no_signal", False)):
                        print(f"[TIEBREAKER] Using delta_spot={float(delta_spot):.2f} -> {'BULL' if bullish_signal else 'BEAR'}")
                elif ema_up or ema_down:
                    # Use EMA trend as secondary tiebreaker
                    bullish_signal = ema_up
                    bearish_signal = ema_down
                    if bool(getattr(self.cfg, "debug_log_no_signal", False)):
                        print(f"[TIEBREAKER] Using EMA trend -> {'BULL' if ema_up else 'BEAR'}")
                elif rsi_val is not None:
                    # Final fallback: RSI midline
                    bullish_signal = float(rsi_val) >= 50.0
                    bearish_signal = float(rsi_val) < 50.0
                    if bool(getattr(self.cfg, "debug_log_no_signal", False)):
                        print(f"[TIEBREAKER] Using RSI={float(rsi_val):.1f} -> {'BULL' if bullish_signal else 'BEAR'}")
                else:
                    # Should never happen, but default to bullish
                    bullish_signal = True
                    bearish_signal = False
                    if bool(getattr(self.cfg, "debug_log_no_signal", False)):
                        print("[TIEBREAKER] Using default -> BULL")
                # Allow the trade regardless of EMA direction when tied
                bullish_ok = bullish_signal
                bearish_ok = bearish_signal
            else:
                # Normal scoring logic when scores are different
                bearish_signal = bear_score >= int(min_score) and bear_score > bull_score
                bearish_ok = bool(ema_down) or bool(bear_score >= (bull_score + 2))
                
                bullish_signal = bull_score >= int(min_score) and bull_score > bear_score
                bullish_ok = bool(ema_up) or bool(bull_score >= (bear_score + 2))

            desired_dir = None  # "bull" | "bear"

            # Decide bull vs bear purely from scores.
            # Supertrend gating is applied ONLY after we have selected a specific strike/trade to place.
            if (bearish_signal and bearish_ok) and (bullish_signal and bullish_ok):
                if bull_score > bear_score:
                    desired_dir = "bull"
                elif bear_score > bull_score:
                    desired_dir = "bear"
                else:
                    if delta_spot is not None and float(delta_spot) != 0:
                        desired_dir = "bull" if float(delta_spot) > 0 else "bear"
                    else:
                        desired_dir = "bull" if bool(ema_up) else "bear"
            elif bearish_signal and bearish_ok:
                desired_dir = "bear"
            elif bullish_signal and bullish_ok:
                desired_dir = "bull"

            # If the trade was recommended by GPT, override the desired direction directly
            # and bypass the indicators / rsi confluence / min score diff.
            if getattr(self, "_current_entry_recommended_by_gpt", False):
                if forced_directional_name in {"long_call", "short_put"}:
                    desired_dir = "bull"
                elif forced_directional_name in {"long_put", "short_call"}:
                    desired_dir = "bear"
                elif getattr(self, "_last_gpt_bias", None) == "CE":
                    desired_dir = "bull"
                elif getattr(self, "_last_gpt_bias", None) == "PE":
                    desired_dir = "bear"
                else:
                    reason = "GPT recommended directional strategy but bias is neutral/unknown"
                    if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                        return
                    return
            else:
                # Enforce minimum score advantage to reduce churn.
                if desired_dir is not None:
                    try:
                        if abs(int(bull_score) - int(bear_score)) < int(min_diff):
                            desired_dir = None
                    except Exception:
                        pass

            if desired_dir is not None:
                enable_rc = getattr(self.cfg, "enable_rsi_confluence", False)
                if getattr(self, "_current_entry_recommended_by_gpt", False):
                    enable_rc = False
                if enable_rc and rsi_val is not None:
                    if desired_dir == "bull" and float(rsi_val) < 50.0:
                        reason = f"RSI {float(rsi_val):.2f} < 50 (No Confluence)"
                        if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                            return
                        desired_dir = None
                    elif desired_dir == "bear" and float(rsi_val) >= 50.0:
                        reason = f"RSI {float(rsi_val):.2f} >= 50 (No Confluence)"
                        if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                            return
                        desired_dir = None

            if desired_dir is not None:
                try:
                    max_same_qty = int(getattr(self.cfg, "entry_max_same_direction_qty", 0) or 0)
                except Exception:
                    max_same_qty = 0
                if max_same_qty > 0:
                    bypass_qty = False
                    if getattr(self, "_current_entry_recommended_by_gpt", False) and bool(getattr(self.cfg, "gpt_override_checks", False)):
                        bypass_qty = True
                    if not bypass_qty:
                        net_qty = self._directional_exposure_qty()
                        if desired_dir == "bull" and int(net_qty) >= int(max_same_qty):
                            reason = f"Directional exposure too high (bull qty={net_qty} >= {max_same_qty})"
                            if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                                return
                            return
                        if desired_dir == "bear" and int(net_qty) <= -int(max_same_qty):
                            reason = f"Directional exposure too high (bear qty={net_qty} <= -{max_same_qty})"
                            if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                                return
                            return



            if desired_dir == "bear":
                # Forced-directional: only enter when the desired direction matches.
                if forced_directional_name in {"long_call", "short_put"}:
                    reason = f"Forced {forced_directional_name}: waiting for bullish signal"
                    if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                        return
                    return

                if forced_directional_name in {"long_put", "short_call"}:
                    # Execute forced bearish style directly.
                    if not self._can_open_trade_type(position_type="directional", name=str(forced_directional_name)):
                        return
                    strike_offset = self._get_directional_strike_offset(atr_val)
                    if forced_directional_name == "short_call":
                        target_call_strike = float(spot) + strike_offset
                        dynamic_call = self._pick_nearest_strike(chain, "CE", target_call_strike)
                        if dynamic_call is None:
                            reason = f"Dynamic CE missing (target strike: {target_call_strike:.0f})"
                            if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                                return
                            return
                        self._open_directional_from_option(dynamic_call, name="short_call", spot=float(spot), atr_val=float(atr_val))
                    else:
                        target_put_strike = float(spot) - strike_offset
                        dynamic_put = self._pick_nearest_strike(chain, "PE", target_put_strike)
                        if dynamic_put is None:
                            reason = f"Dynamic PE missing (target strike: {target_put_strike:.0f})"
                            if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                                return
                            return
                        self._open_directional_from_option(dynamic_put, name="long_put", spot=float(spot), atr_val=float(atr_val))
                    return

                strike_offset = self._get_directional_strike_offset(atr_val)
                # Same-direction fallback to avoid getting stuck:
                # - Bearish can be expressed as long_put (debit) OR short_call (credit).
                primary = "short_call" if dir_style == "short" else "long_put"
                fallback = "long_put" if primary == "short_call" else "short_call"

                chosen = None
                if self._can_open_trade_type(position_type="directional", name=primary):
                    chosen = primary
                elif self._can_open_trade_type(position_type="directional", name=fallback):
                    print(f"[SAFETY] Fallback entry: {primary} blocked, trying {fallback}")
                    chosen = fallback
                else:
                    reason = f"Safety: blocked {primary} and {fallback} (max same-type streak)"
                    if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                        return
                    return

                if chosen == "short_call":
                    target_call_strike = float(spot) + strike_offset
                    dynamic_call = self._pick_nearest_strike(chain, "CE", target_call_strike)
                    if dynamic_call is None:
                        reason = f"Dynamic CE missing (target strike: {target_call_strike:.0f})"
                        if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                            return
                        return
                    # Supertrend voting is now upstream in scoring.
                    self._open_directional_from_option(dynamic_call, name="short_call", spot=float(spot), atr_val=float(atr_val))
                else:
                    target_put_strike = float(spot) - strike_offset
                    dynamic_put = self._pick_nearest_strike(chain, "PE", target_put_strike)
                    if dynamic_put is None:
                        reason = f"Dynamic PE missing (target strike: {target_put_strike:.0f})"
                        if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                            return
                        return
                    # Supertrend voting is now upstream in scoring.
                    self._open_directional_from_option(dynamic_put, name="long_put", spot=float(spot), atr_val=float(atr_val))
                return

            if desired_dir == "bull":
                # Forced-directional: only enter when the desired direction matches.
                if forced_directional_name in {"long_put", "short_call"}:
                    reason = f"Forced {forced_directional_name}: waiting for bearish signal"
                    if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                        return
                    return

                if forced_directional_name in {"long_call", "short_put"}:
                    # Execute forced bullish style directly.
                    if not self._can_open_trade_type(position_type="directional", name=str(forced_directional_name)):
                        return
                    strike_offset = self._get_directional_strike_offset(atr_val)
                    if forced_directional_name == "short_put":
                        target_put_strike = float(spot) - strike_offset
                        dynamic_put = self._pick_nearest_strike(chain, "PE", target_put_strike)
                        if dynamic_put is None:
                            reason = f"Dynamic PE missing (target strike: {target_put_strike:.0f})"
                            if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                                return
                            return
                        self._open_directional_from_option(dynamic_put, name="short_put", spot=float(spot), atr_val=float(atr_val))
                    else:
                        target_call_strike = float(spot) + strike_offset
                        dynamic_call = self._pick_nearest_strike(chain, "CE", target_call_strike)
                        if dynamic_call is None:
                            reason = f"Dynamic CE missing (target strike: {target_call_strike:.0f})"
                            if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                                return
                            return
                        self._open_directional_from_option(dynamic_call, name="long_call", spot=float(spot), atr_val=float(atr_val))
                    return

                strike_offset = self._get_directional_strike_offset(atr_val)
                # Same-direction fallback to avoid getting stuck:
                # - Bullish can be expressed as long_call (debit) OR short_put (credit).
                primary = "short_put" if dir_style == "short" else "long_call"
                fallback = "long_call" if primary == "short_put" else "short_put"

                chosen = None
                if self._can_open_trade_type(position_type="directional", name=primary):
                    chosen = primary
                elif self._can_open_trade_type(position_type="directional", name=fallback):
                    print(f"[SAFETY] Fallback entry: {primary} blocked, trying {fallback}")
                    chosen = fallback
                else:
                    reason = f"Safety: blocked {primary} and {fallback} (max same-type streak)"
                    if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                        return
                    return

                if chosen == "short_put":
                    target_put_strike = float(spot) - strike_offset
                    dynamic_put = self._pick_nearest_strike(chain, "PE", target_put_strike)
                    if dynamic_put is None:
                        reason = f"Dynamic PE missing (target strike: {target_put_strike:.0f})"
                        if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                            return
                        return
                    # Supertrend voting is now upstream in scoring.
                    self._open_directional_from_option(dynamic_put, name="short_put", spot=float(spot), atr_val=float(atr_val))
                else:
                    target_call_strike = float(spot) + strike_offset
                    dynamic_call = self._pick_nearest_strike(chain, "CE", target_call_strike)
                    if dynamic_call is None:
                        reason = f"Dynamic CE missing (target strike: {target_call_strike:.0f})"
                        if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                            return
                        return
                    # Supertrend voting is now upstream in scoring.
                    self._open_directional_from_option(dynamic_call, name="long_call", spot=float(spot), atr_val=float(atr_val))
                return

            rsi_dbg = "N/A" if rsi_val is None else f"{float(rsi_val):.1f}"
            reason = (
                "Auto directional: no EMA+RSI setup "
                f"(EMA9={float(fast_ema):.2f}, EMA21={float(slow_ema):.2f}, RSI={rsi_dbg}; "
                f"scores: bull={int(bull_score)} bear={int(bear_score)}; "
                f"need >={int(min_score)} confirmations and score diff >= {int(min_diff)} to trade)"
            )
            if maybe_reroute_entry_block(reason, current_strategy=str(forced_directional_name or effective_strat)):
                return
            return

        call_delta = None
        if call and call.get("iv"):
            try:
                call_delta = delta(
                    spot=spot,
                    strike=float(call["strike"]),
                    time_to_expiry=time_to_expiry,
                    rate=rate,
                    iv=float(call["iv"]),
                    option_type="CE",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to compute call delta: {exc}")

        put_delta = None
        if put and put.get("iv"):
            try:
                put_delta = delta(
                    spot=spot,
                    strike=float(put["strike"]),
                    time_to_expiry=time_to_expiry,
                    rate=rate,
                    iv=float(put["iv"]),
                    option_type="PE",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to compute put delta: {exc}")

        # ---- Example signal logic ----

        uptrend = fast_ema > slow_ema
        downtrend = fast_ema < slow_ema

        if indecision:
            block("Indecision candle")
            return

        # Block directional entries when ATR is below minimum threshold (applies to explicit directional mode)
        try:
            dir_min_atr = float(getattr(self.cfg, "dir_min_atr", 20.0) or 20.0)
        except Exception:
            dir_min_atr = 20.0
        if atr_val < dir_min_atr:
            block(f"ATR {atr_val:.2f} below minimum {dir_min_atr:.2f} for directional trades")
            return

        if bool(getattr(self.cfg, "debug_log_no_signal", False)):
            block("No directional signal")

        bullish_ok = uptrend and not indecision
        # Allow either a clear bullish pattern OR strong RSI; don't require both.
        if bullish_candle or (rsi_val is not None and 55.0 <= rsi_val <= 70.0):
            bullish_ok = bullish_ok and True
        else:
            bullish_ok = False

        delta_ok = True
        if call_delta is not None:
            # Be robust to unexpected delta sign conventions.
            delta_ok = 0.3 <= abs(float(call_delta)) <= 0.8

        qty = self._entry_qty(atr_val)
        if qty <= 0:
            return

        if bullish_ok and delta_ok:
            rsi_str = f"{rsi_val:.1f}" if rsi_val is not None else "N/A"
            call_delta_str = f"{call_delta:.2f}" if call_delta is not None else "N/A"
            spot_str = f"{spot:.1f}" if spot is not None else "N/A"
            msg = (
                f"BULLISH signal: EMA9>{'EMA21'}; RSI={rsi_str}; "
                f"call Δ≈{call_delta_str}; spot={spot_str}"
            )
            print(msg)
            if not self.cfg.enable_live_trading:
                print("Paper mode enabled; simulating order.")

            # Example live order (enable with care and adapt symbol fields).
            if call is not None:
                symbol = str(call["symbol"])
                try:
                    exchange = str(call.get("exchange") or "") or None
                    token = call.get("token")
                    entry_price = self._try_get_ltp(symbol, exchange=exchange)
                    ok, reason = self._check_entry_liquidity([{"symbol": symbol, "exchange": exchange, "token": token}])
                    if not ok:
                        print(f"[ENTRY BLOCKED] {reason}")
                        return

                    if not self.cfg.enable_live_trading:
                        print(f"[PAPER] BUY {symbol} x{qty}")
                    else:
                        self.client.place_order(
                            symbol=symbol,
                            side="BUY",
                            quantity=qty,
                            exchange=exchange,
                            symbol_token=str(token or "") or None,
                        )

                    trade_id = self._new_trade_id("D")

                    strike = call.get("strike")
                    opt_type = str(call.get("option_type") or "CE").strip().upper()
                    if opt_type in {"CALL", "C"}:
                        opt_type = "CE"
                    elif opt_type in {"PUT", "P"}:
                        opt_type = "PE"
                    expiry = call.get("expiry")

                    legs_for_trade = [
                        {
                            "symbol": symbol,
                            "token": token,
                            "exchange": exchange,
                            "side": "BUY",
                            "quantity": int(qty),
                            "entry_price": entry_price,
                            "strike": strike,
                            "option_type": opt_type,
                            "expiry": expiry,
                            "iv": call.get("iv"),
                        }
                    ]

                    self.state.open_directional.append(
                        {
                            "trade_id": trade_id,
                            "name": "long_call",
                            "symbol": symbol,
                            "token": token,
                            "exchange": exchange,
                            "side": "BUY",
                            "quantity": int(qty),
                            "entry_spot": float(spot),
                            "entry_price": entry_price,
                            "atr": float(atr_val),
                            "strike": strike,
                            "option_type": opt_type,
                            "expiry": expiry,
                            "iv": call.get("iv"),
                            "legs": legs_for_trade,
                            "meta": {},
                            "opened_ts": time.time(),
                        }
                    )
                    self.state.trades_today += 1
                    self.state.last_entry_ts = time.time()

                    if self._has_event_sink():
                        legs = [dict(l) for l in legs_for_trade]
                        self._emit(
                            TradeLogEvent(
                                ts=time.time(),
                                event="OPEN",
                                trade_id=trade_id,
                                position_type="directional",
                                name="long_call",
                                legs=legs,
                                mtm=self._compute_legs_mtm(legs),
                            )
                        )
                except Exception as exc:  # noqa: BLE001
                    print(f"Failed to place bullish order: {exc}")
            return

        bearish_ok = downtrend and not indecision
        if bearish_candle or (rsi_val is not None and 30.0 <= rsi_val <= 45.0):
            bearish_ok = bearish_ok and True
        else:
            bearish_ok = False

        delta_ok_put = True
        if put_delta is not None:
            # Some implementations return negative deltas for puts, others return positive.
            # Use magnitude so PE entries aren't unfairly blocked.
            delta_ok_put = 0.3 <= abs(float(put_delta)) <= 0.8

        if bearish_ok and delta_ok_put:
            rsi_str = f"{rsi_val:.1f}" if rsi_val is not None else "N/A"
            put_delta_str = f"{put_delta:.2f}" if put_delta is not None else "N/A"
            spot_str = f"{spot:.1f}" if spot is not None else "N/A"
            msg = (
                f"BEARISH signal: EMA9<{'EMA21'}; RSI={rsi_str}; "
                f"put Δ≈{put_delta_str}; spot={spot_str}"
            )
            print(msg)
            if not self.cfg.enable_live_trading:
                print("Paper mode enabled; simulating order.")

            if put is not None:
                symbol = str(put["symbol"])
                try:
                    exchange = str(put.get("exchange") or "") or None
                    token = put.get("token")
                    entry_price = self._try_get_ltp(symbol, exchange=exchange)
                    ok, reason = self._check_entry_liquidity([{"symbol": symbol, "exchange": exchange, "token": token}])
                    if not ok:
                        print(f"[ENTRY BLOCKED] {reason}")
                        return

                    if not self.cfg.enable_live_trading:
                        print(f"[PAPER] BUY {symbol} x{qty}")
                    else:
                        self.client.place_order(
                            symbol=symbol,
                            side="BUY",
                            quantity=qty,
                            exchange=exchange,
                            symbol_token=str(token or "") or None,
                        )

                    trade_id = self._new_trade_id("D")

                    strike = put.get("strike")
                    opt_type = str(put.get("option_type") or "PE").strip().upper()
                    if opt_type in {"CALL", "C"}:
                        opt_type = "CE"
                    elif opt_type in {"PUT", "P"}:
                        opt_type = "PE"
                    expiry = put.get("expiry")

                    legs_for_trade = [
                        {
                            "symbol": symbol,
                            "token": token,
                            "exchange": exchange,
                            "side": "BUY",
                            "quantity": int(qty),
                            "entry_price": entry_price,
                            "strike": strike,
                            "option_type": opt_type,
                            "expiry": expiry,
                            "iv": put.get("iv"),
                        }
                    ]

                    self.state.open_directional.append(
                        {
                            "trade_id": trade_id,
                            "name": "long_put",
                            "symbol": symbol,
                            "token": token,
                            "exchange": exchange,
                            "side": "BUY",
                            "quantity": int(qty),
                            "entry_spot": float(spot),
                            "entry_price": entry_price,
                            "atr": float(atr_val),
                            "strike": strike,
                            "option_type": opt_type,
                            "expiry": expiry,
                            "iv": put.get("iv"),
                            "legs": legs_for_trade,
                            "meta": {},
                            "opened_ts": time.time(),
                        }
                    )
                    self.state.trades_today += 1
                    self.state.last_entry_ts = time.time()

                    if self._has_event_sink():
                        legs = [dict(l) for l in legs_for_trade]
                        self._emit(
                            TradeLogEvent(
                                ts=time.time(),
                                event="OPEN",
                                trade_id=trade_id,
                                position_type="directional",
                                name="long_put",
                                legs=legs,
                                mtm=self._compute_legs_mtm(legs),
                            )
                        )
                except Exception as exc:  # noqa: BLE001
                    print(f"Failed to place bearish order: {exc}")
            return

    def _emit_paper_mtm_updates(self) -> None:
        # Historically this only ran in paper mode.
        # We now allow MTM/leg snapshots in live mode too when the UI attaches
        # an event sink; without a sink, skip to avoid extra quote calls.
        if not self._has_event_sink():
            return

        for tr in list(self.state.open_directional):
            if not isinstance(tr, dict):
                continue
            trade_id = str(tr.get("trade_id") or "")
            name = str(tr.get("name") or "")
            legs_src = tr.get("legs")
            if isinstance(legs_src, list) and legs_src:
                legs = []
                for l in legs_src:
                    if not isinstance(l, dict):
                        continue
                    leg_copy = dict(l)
                    legs.append(leg_copy)
                mtm_val = self._refresh_legs_ltp_and_mtm(legs)
            else:
                symbol = str(tr.get("symbol") or "")
                exchange = str(tr.get("exchange") or "") or None
                token = tr.get("token")
                qty = int(tr.get("quantity") or 0)
                entry_price = tr.get("entry_price")
                cur_price = self._try_get_ltp_for_leg({
                    "symbol": symbol,
                    "token": token,
                    "exchange": exchange,
                })
                legs = [
                    {
                        "symbol": symbol,
                        "token": token,
                        "exchange": exchange,
                        "side": str(tr.get("side") or ""),
                        "quantity": qty,
                        "entry_price": entry_price,
                        "strike": tr.get("strike"),
                        "option_type": tr.get("option_type"),
                        "expiry": tr.get("expiry"),
                        "ltp": cur_price,
                    }
                ]
                mtm_val = self._compute_legs_mtm_cached(legs)
            self._emit(
                TradeLogEvent(
                    ts=time.time(),
                    event="UPDATE",
                    trade_id=trade_id or "(unknown)",
                    position_type="directional",
                    name=name,
                    legs=legs,
                    mtm=mtm_val,
                )
            )

        for trade in list(self.state.open_multi):
            if not isinstance(trade, dict):
                continue
            trade_id = str(trade.get("trade_id") or "")
            name = str(trade.get("name") or "")
            legs = trade.get("legs")
            if not isinstance(legs, list):
                continue
            leg_dicts = []
            for l in legs:
                if not isinstance(l, dict):
                    continue
                leg_copy = dict(l)
                leg_dicts.append(leg_copy)
            mtm_val = self._refresh_legs_ltp_and_mtm(leg_dicts)
            self._emit(
                TradeLogEvent(
                    ts=time.time(),
                    event="UPDATE",
                    trade_id=trade_id or "(unknown)",
                    position_type="multi",
                    name=name,
                    legs=leg_dicts,
                    mtm=mtm_val,
                )
            )

    def _session_exit_adjustments(self) -> Dict[str, float]:
        """Return exit parameter multipliers based on current trading session time.

        Returns a dict with keys: sl_mult, tp_mult, trail_mult.

        Reads config keys:
          session_exit_enabled (bool) - master switch
          session_opening_hhmm (str) - end of opening session (default "10:00")
          session_lunch_start_hhmm (str) - start of lunch session (default "12:00")
          session_lunch_end_hhmm (str) - end of lunch session (default "13:30")
          session_close_hhmm (str) - start of closing session (default "14:45")
          session_opening_sl_mult (float) - tighter stop in opening (default 0.7)
          session_opening_tp_mult (float) - tighter target in opening (default 0.8)
          session_lunch_sl_mult (float) - wider stop in lunch (default 1.3)
          session_lunch_tp_mult (float) - wider target in lunch (default 1.2)
          session_close_sl_mult (float) - tighter stop in closing (default 0.6)
          session_close_tp_mult (float) - tighter target in closing (default 0.7)

        Returns {sl: 1.0, tp: 1.0, trail: 1.0} when disabled or on error.
        """
        default = {"sl_mult": 1.0, "tp_mult": 1.0, "trail_mult": 1.0}

        try:
            if not bool(getattr(self.cfg, "session_exit_enabled", False)):
                return dict(default)
        except Exception:
            return dict(default)

        try:
            opening_end = str(getattr(self.cfg, "session_opening_hhmm", "10:00") or "10:00").strip()
            lunch_start = str(getattr(self.cfg, "session_lunch_start_hhmm", "12:00") or "12:00").strip()
            lunch_end = str(getattr(self.cfg, "session_lunch_end_hhmm", "13:30") or "13:30").strip()
            close_start = str(getattr(self.cfg, "session_close_hhmm", "14:45") or "14:45").strip()
            opening_sl = float(getattr(self.cfg, "session_opening_sl_mult", 0.7) or 0.7)
            opening_tp = float(getattr(self.cfg, "session_opening_tp_mult", 0.8) or 0.8)
            lunch_sl = float(getattr(self.cfg, "session_lunch_sl_mult", 1.3) or 1.3)
            lunch_tp = float(getattr(self.cfg, "session_lunch_tp_mult", 1.2) or 1.2)
            close_sl = float(getattr(self.cfg, "session_close_sl_mult", 0.6) or 0.6)
            close_tp = float(getattr(self.cfg, "session_close_tp_mult", 0.7) or 0.7)
        except Exception:
            return dict(default)

        def _parse_hhmm(s: str) -> int:
            try:
                parts = str(s or "").strip().split(":")
                return int(parts[0]) * 60 + int(parts[1])
            except Exception:
                return 0

        now = dt_datetime.now(IST).time() if IST is not None else dt_datetime.now().time()
        now_min = int(now.hour) * 60 + int(now.minute)

        open_min = 9 * 60 + 15  # 09:15
        opening_end_min = _parse_hhmm(opening_end) or (10 * 60)
        lunch_start_min = _parse_hhmm(lunch_start) or (12 * 60)
        lunch_end_min = _parse_hhmm(lunch_end) or (13 * 60 + 30)
        close_start_min = _parse_hhmm(close_start) or (14 * 60 + 45)
        market_close_min = 15 * 60 + 30  # 15:30

        if now_min < open_min or now_min > market_close_min:
            return dict(default)

        sl_adj = 1.0
        tp_adj = 1.0

        if now_min <= opening_end_min:
            sl_adj = float(opening_sl)
            tp_adj = float(opening_tp)
        elif now_min <= lunch_start_min:
            sl_adj = 1.0
            tp_adj = 1.0
        elif now_min <= lunch_end_min:
            sl_adj = float(lunch_sl)
            tp_adj = float(lunch_tp)
        elif now_min <= close_start_min:
            sl_adj = 1.0
            tp_adj = 1.0
        else:
            sl_adj = float(close_sl)
            tp_adj = float(close_tp)

        sl_adj = max(0.3, min(3.0, float(sl_adj)))
        tp_adj = max(0.3, min(3.0, float(tp_adj)))
        return {"sl_mult": float(sl_adj), "tp_mult": float(tp_adj), "trail_mult": float(sl_adj)}

    def _record_trade_outcome(self, trade_id: str, realized_pnl: float, strategy_name: str) -> None:
        """Record a trade outcome and auto-disable losing strategies."""
        if not bool(getattr(self.cfg, "winrate_tracker_enabled", False)):
            return
        try:
            key = str(strategy_name or "unknown").strip().lower()
            if not key:
                return
            win = float(realized_pnl or 0.0) > 0.0
            
            # Get or create tracker for this strategy
            tracker = dict(self._strategy_winrate.get(key, {}) or {})
            streak = int(tracker.get("streak", 0) or 0)
            wins = int(tracker.get("wins", 0) or 0)
            losses = int(tracker.get("losses", 0) or 0)
            total = int(tracker.get("total", 0) or 0)
            
            if win:
                streak = streak + 1 if streak > 0 else 1
                wins += 1
            else:
                streak = streak - 1 if streak < 0 else -1
                losses += 1
            total += 1
            
            tracker["streak"] = streak
            tracker["wins"] = wins
            tracker["losses"] = losses
            tracker["total"] = total
            tracker["last_ts"] = float(time.time())
            self._strategy_winrate[key] = tracker
            
            # Auto-disable on consecutive losses
            max_losses = int(getattr(self.cfg, "winrate_tracker_max_losses", 3) or 3)
            if not win and abs(streak) >= max_losses:
                self._disabled_strategies[key] = float(time.time())
                print(f"[WINRATE] Disabled {key}: {abs(streak)} consecutive losses")
            
            # Re-enable on recovery wins
            recovery_wins = int(getattr(self.cfg, "winrate_tracker_recovery_wins", 1) or 1)
            if win and streak >= recovery_wins:
                disabled = dict(self._disabled_strategies)
                if key in disabled:
                    del self._disabled_strategies[key]
                    print(f"[WINRATE] Re-enabled {key}: {streak} consecutive wins")
        except Exception:
            pass
    
    def _is_strategy_disabled(self, strategy_name: str) -> bool:
        """Check if a strategy is currently disabled by the win-rate tracker."""
        if not bool(getattr(self.cfg, "winrate_tracker_enabled", False)):
            return False
        try:
            key = str(strategy_name or "unknown").strip().lower()
            if not key:
                return False
            disabled = dict(getattr(self, "_disabled_strategies", {}) or {})
            if key in disabled:
                elapsed = float(time.time()) - float(disabled.get(key, 0.0) or 0.0)
                # Auto-re-enable after 1 hour as a safety measure
                if elapsed > 3600.0:
                    del self._disabled_strategies[key]
                    return False
                return True
            return False
        except Exception:
            return False

    def _manage_open_trades(self) -> None:
        """Manage exits for open positions based on P&L or price levels."""
        now_t = self._now_ist_time()
        force_exit_t = self._parse_hhmm(
            getattr(self.cfg, "premium_force_exit_hhmm", ""),
            default=dt_time(14, 20),
        )

        # Shared market-state snapshot for kill-switch checks.
        tf_str = self.cfg.timeframe
        try:
            candles: List[Candle] = self.client.get_candles(
                symbol=self.cfg.underlying,
                timeframe=tf_str,
                limit=100, # sufficient for standard indicators
            )
            closes = [c.close for c in candles]
            highs = [c.high for c in candles]
            lows = [c.low for c in candles]
            ema_f_p = int(getattr(self.cfg, "ema_fast", 9))
            ema_s_p = int(getattr(self.cfg, "ema_slow", 21))
            rsi_p = 14
            atr_p = int(self.cfg.atr_period) if self.cfg.atr_period else 14
            min_c = max(atr_p + 1, rsi_p + 1, ema_s_p, ema_f_p)

            if len(candles) < min_c:
                trend_strength = 0.0
                atr_now = None
            else:
                fast_ema = ema(closes, period=ema_f_p)
                slow_ema = ema(closes, period=ema_s_p)
                close = closes[-1] if closes else 0.0
                trend_strength = (
                    self._trend_strength(float(fast_ema), float(slow_ema), float(close))
                    if fast_ema is not None and slow_ema is not None
                    else 0.0
                )
                atr_now = atr(highs, lows, closes, period=atr_p)
        except Exception:
            trend_strength = 0.0
            atr_now = None

        try:
            stop_pct = float(getattr(self.cfg, "premium_mtm_stop_pct", 0.30) or 0.30)
        except Exception:
            stop_pct = 0.30
        try:
            target_pct = float(getattr(self.cfg, "premium_mtm_target_pct", 0.18))
        except Exception:
            target_pct = 0.18
        try:
            spike_mult = float(getattr(self.cfg, "atr_spike_mult", 2.0) or 2.0)
        except Exception:
            spike_mult = 2.0

        # ---- Manage directional exits (underlying-based) ----
        if self.state.open_directional:
            latest_bar_ts = getattr(candles[-1], "time", None) if candles else None
            try:
                spot = float(self.client.get_ltp(self.cfg.underlying))
            except Exception:
                spot = None
            try:
                be_mult = float(getattr(self.cfg, "dir_breakeven_atr_mult", 0.6) or 0.6)
            except Exception:
                be_mult = 0.6
            try:
                router_risk = self._router_risk_overrides()
                trail_mult = float(router_risk.get("dir_trail_atr_mult") or getattr(self.cfg, "dir_trail_atr_mult", 1.0) or 1.0)
            except Exception:
                trail_mult = 1.0
            try:
                partial_mult = float(getattr(self.cfg, "dir_partial_target_mult", 0.0) or 0.0)
            except Exception:
                partial_mult = 0.0
            try:
                partial_pct = float(getattr(self.cfg, "dir_partial_qty_pct", 0.0) or 0.0)
            except Exception:
                partial_pct = 0.0

            # Enhancement Exits Config
            stag_mins = int(getattr(self.cfg, "stagnation_exit_minutes", 0) or 0)
            stag_pnl = float(getattr(self.cfg, "stagnation_pnl_threshold", 100.0) or 100.0)
            chan_enabled = bool(getattr(self.cfg, "enable_chandelier_exit", False))
            chan_mult = float(getattr(self.cfg, "chandelier_multiplier", 3.0) or 3.0)
            piv_enabled = bool(getattr(self.cfg, "enable_pivot_targets", False))
            pivots = self._get_pivot_points() if piv_enabled else None

            flip_enabled = bool(getattr(self.cfg, "dir_flip_long_to_short_on_stop", False))
            flip_chain: Optional[List[dict]] = None
            new_dir_trades: List[Dict[str, object]] = []

            remaining_dir: List[Dict[str, object]] = []
            for tr in self.state.open_directional:
                if tr.get("close_failed") is True:
                    remaining_dir.append(tr)
                    continue
                if tr.get("close_attempts", 0) > 0:
                    last_attempt = float(tr.get("last_close_attempt_ts") or 0.0)
                    if time.time() - last_attempt < 30.0:
                        remaining_dir.append(tr)
                        continue
                # Keep current risk levels on the trade so UI snapshots can display them.
                # These are recomputed each management cycle and represent the *current* thresholds.
                try:
                    tr["spot_stop"] = None
                    tr["spot_target"] = None
                except Exception:
                    pass
                entry_spot = float(tr.get("entry_spot") or 0.0)
                atr_val = float(tr.get("atr") or 0.0)
                name = str(tr.get("name") or "")
                symbol_for_log = str(tr.get("symbol") or "")
                qty = int(tr.get("quantity") or 0)
                symbol = str(tr.get("symbol") or "")
                token = tr.get("token")
                exchange = str(tr.get("exchange") or "") or None
                entry_price = tr.get("entry_price")
                meta = tr.get("meta") if isinstance(tr.get("meta"), dict) else {}
                tb_state = self._extract_triple_barrier_state(tr)
                if tb_state is not None:
                    tb_legs = tr.get("legs")
                    tb_leg = None
                    if isinstance(tb_legs, list):
                        for candidate_leg in tb_legs:
                            if isinstance(candidate_leg, dict) and not bool(candidate_leg.get("is_hedge")) and int(candidate_leg.get("quantity") or 0) > 0:
                                tb_leg = candidate_leg
                                break
                    if tb_leg is None:
                        tb_leg = {
                            "symbol": symbol,
                            "token": token,
                            "exchange": exchange,
                            "side": tr.get("side"),
                            "quantity": tr.get("quantity"),
                            "entry_price": entry_price,
                            "strike": tr.get("strike"),
                            "option_type": tr.get("option_type"),
                            "expiry": tr.get("expiry"),
                        }
                    current_price = self._triple_barrier_price(tr, tb_leg)
                    if current_price is not None:
                        evaluation = evaluate_triple_barrier_state(
                            tb_state,
                            current_price=float(current_price),
                            bar_timestamp=latest_bar_ts,
                            count_bar_close=True,
                        )
                        self._write_triple_barrier_state(tr, tb_state)
                        if evaluation.trigger_source:
                            if self._close_directional_trade_via_triple_barrier(
                                trade=tr,
                                trigger_source=str(evaluation.trigger_source),
                                reference_price=float(evaluation.current_price),
                            ):
                                continue
                    remaining_dir.append(tr)
                    continue

                be_mult_local = float(be_mult)
                trail_mult_local = float(trail_mult)
                if bool(tr.get("partial_exit_done")):
                    try:
                        post_be = float(getattr(self.cfg, "dir_post_partial_be_atr_mult", 0.0) or 0.0)
                    except Exception:
                        post_be = 0.0
                    try:
                        post_trail = float(getattr(self.cfg, "dir_post_partial_trail_atr_mult", 0.0) or 0.0)
                    except Exception:
                        post_trail = 0.0
                    if post_be > 0:
                        be_mult_local = post_be
                    if post_trail > 0:
                        trail_mult_local = post_trail
                # Session-specific exit adjustments (overrides)
                if bool(getattr(self.cfg, "session_exit_enabled", False)):
                    try:
                        session_adj = self._session_exit_adjustments()
                        if isinstance(session_adj, dict):
                            sa_sl = float(session_adj.get("sl_mult", 1.0) or 1.0)
                            sa_trail = float(session_adj.get("trail_mult", 1.0) or 1.0)
                            be_mult_local *= sa_sl
                            trail_mult_local *= sa_trail
                    except Exception:
                        pass
                entry_side = str(tr.get("side") or "").upper()
                close_side = "SELL" if entry_side == "BUY" else "BUY"

                should_exit = False
                reason = ""

                mtm_val: Optional[float] = None
                entry_abs: Optional[float] = None
                cur_premium_val: Optional[float] = None

                # Compute MTM for trailing stop check
                legs_src = tr.get("legs")
                if isinstance(legs_src, list) and legs_src:
                    legs = legs_src
                else:
                    legs = [
                        {
                            "symbol": symbol,
                            "token": token,
                            "exchange": exchange,
                            "side": str(tr.get("side") or ""),
                            "quantity": qty,
                            "entry_price": entry_price,
                            "strike": tr.get("strike"),
                            "option_type": tr.get("option_type"),
                            "expiry": tr.get("expiry"),
                            "iv": tr.get("iv"),
                        }
                    ]
                    tr["legs"] = legs

                mtm_val = self._compute_legs_mtm(legs)

                # Optional: GPT per-leg premium stop/target management.
                try:
                    is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
                except Exception:
                    is_paper = True
                if self._gpt_leg_manage_enabled(is_paper=bool(is_paper)):
                    try:
                        self._apply_gpt_leg_exits(trade=tr, position_type="directional", legs=legs)
                    except Exception:
                        pass

                if not should_exit:
                    exits_armed = self._gpt_leg_exits_armed(trade=tr)
                    for lg in legs:
                        hit = self._leg_exit_hit(leg=lg)
                        if not hit:
                            continue
                        # Hard safety rule: once stop is breached, exit immediately
                        # even during post-update grace. Target exits still respect arming.
                        if hit == "leg_stop" or exits_armed:
                            should_exit = True
                            reason = f"gpt_{hit}"
                            break

                # Current premium value (absolute rupees) for premium trailing stop.
                # Uses cached leg["ltp"] populated by _compute_legs_mtm.
                try:
                    prem_val = 0.0
                    for lg in legs:
                        if not isinstance(lg, dict):
                            continue
                        if bool(lg.get("is_hedge")):
                            continue
                        q = int(lg.get("quantity") or 0)
                        if q <= 0:
                            continue
                        p = lg.get("ltp")
                        if p is None:
                            continue
                        prem_val += abs(float(p)) * abs(float(q))
                    if prem_val > 0:
                        cur_premium_val = float(prem_val)
                except Exception:
                    cur_premium_val = None

                # Optional delta hedge rebalance for directional trades.
                scope = str(getattr(self.cfg, "delta_hedge_scope", "strategy_only") or "strategy_only").strip().lower()
                if scope == "all_options" and spot is not None:
                    meta = tr.get("meta") if isinstance(tr.get("meta"), dict) else {}
                    if not isinstance(meta, dict):
                        meta = {}
                    meta["delta_hedge"] = True
                    legs = tr.get("legs") if isinstance(tr.get("legs"), list) else []
                    sym, exch, inferred_root = self._pick_delta_hedge_symbol_for_trade({"legs": legs, "meta": meta})
                    if inferred_root and "delta_hedge_underlying" not in meta:
                        meta["delta_hedge_underlying"] = inferred_root
                    if "delta_hedge_symbol" not in meta and sym:
                        meta["delta_hedge_symbol"] = sym
                    if "delta_hedge_exchange" not in meta and exch:
                        meta["delta_hedge_exchange"] = exch
                    tr["meta"] = meta
                    try:
                        self._rebalance_delta_hedge(tr, spot=float(spot))
                    except Exception as exc:
                        if not bool(meta.get("delta_hedge_err_warned")):
                            print(f"[DELTA HEDGE] Directional rebalance failed: {exc}")
                            meta["delta_hedge_err_warned"] = True
                            tr["meta"] = meta

                # 4) Stagnation Exit
                if stag_mins > 0 and not should_exit:
                    entry_dt_raw = tr.get("entry_time")
                    if entry_dt_raw:
                        try:
                            if isinstance(entry_dt_raw, str):
                                entry_dt = dt_datetime.fromisoformat(entry_dt_raw)
                            else:
                                entry_dt = entry_dt_raw
                            
                            if IST:
                                now_dt = dt_datetime.now(IST)
                            else:
                                now_dt = dt_datetime.now()
                            
                            elapsed_mins = (now_dt - entry_dt).total_seconds() / 60.0
                            if elapsed_mins >= stag_mins and mtm_val is not None and abs(mtm_val) < stag_pnl:
                                should_exit = True
                                reason = "stagnation"
                        except Exception as exc:
                            print(f"[EXIT] Stagnation check failed: {exc}")

                # 2) MTM stop/target removed for directional trades

                # 3) Volatility spike kill-switch removed for directional trades

                if spot is not None and entry_spot > 0:
                    if name in {"long_call", "short_put"}:
                        # Track best favorable spot (highest).
                        best = float(tr.get("best_spot") or entry_spot)
                        if spot > best:
                            best = float(spot)
                            tr["best_spot"] = best
                        
                        # Chandelier Exit (Long): Highest High - Multiplier * ATR
                        if chan_enabled and atr_val > 0:
                            chan_stop = best - (chan_mult * atr_val)
                            if spot <= chan_stop:
                                should_exit = True
                                reason = "chandelier_stop"
                        
                        # Pivot Target (Long)
                        if pivots and not should_exit:
                            tgt = self._pivot_target_level(pivots, direction="long", entry_spot=entry_spot, mtm_val=mtm_val)
                            if tgt is not None:
                                tgt_price, tgt_key = tgt
                                try:
                                    tr["spot_target"] = float(tgt_price)
                                except Exception:
                                    pass
                                if spot >= tgt_price:
                                    should_exit = True
                                    reason = f"pivot_target_{tgt_key}"
                        
                        # P1: GPT exit management override for this trade
                        if getattr(self.cfg, "gpt_exit_management", False):
                            try:
                                self._gpt_exit_management(trade=tr, candles=candles, spot=spot)
                            except Exception:
                                pass

                        # Target Exit (Final TP)
                        tp_mult = float(
                            meta.get("ml_dynamic_dir_tp_atr_mult")
                            if isinstance(meta, dict) and meta.get("ml_dynamic_dir_tp_atr_mult") is not None
                            else getattr(self.cfg, "dir_tp_atr_mult", 0.0)
                        )
                        if tp_mult > 0:
                            tp_price = entry_spot + (tp_mult * atr_val)
                            try:
                                tr["spot_target"] = float(tp_price)
                            except Exception:
                                pass
                            if spot >= tp_price:
                                should_exit = True
                                reason = "spot_target"

                        # Partial Booking Logic (Single)
                        if partial_mult > 0 and partial_pct > 0 and not bool(tr.get("partial_exit_done")):
                            partial_target_price = entry_spot + (partial_mult * atr_val)
                        # ---- #8: Smart Exit Ladder ----
                        ladder_done = False
                        try:
                            if bool(getattr(self.cfg, "exit_ladder_enabled", False)):
                                ladder_steps_raw = str(getattr(self.cfg, "exit_ladder_steps", "0.25:1.0,0.25:1.5,0.25:2.0") or "")
                                steps_done = tr.get("exit_ladder_done")
                                if not isinstance(steps_done, list):
                                    steps_done = []
                                steps_parts = [s.strip() for s in ladder_steps_raw.split(",") if s.strip()]
                                for step_idx, step_str in enumerate(steps_parts):
                                    if step_idx in steps_done:
                                        continue
                                    parts = step_str.split(":")
                                    if len(parts) != 2:
                                        continue
                                    try:
                                        step_pct = float(parts[0])
                                        step_mult = float(parts[1])
                                    except Exception:
                                        continue
                                    if step_pct <= 0 or step_mult <= 0:
                                        continue
                                    step_target = entry_spot + (step_mult * atr_val)
                                    if spot >= step_target:
                                        exit_qty = calc_partial_exit_qty(total_qty=qty, partial_pct=step_pct, lot_size=int(lot_size))
                                        if exit_qty > 0:
                                            # Execute partial exit
                                            try:
                                                if not bool(self.cfg.enable_live_trading):
                                                    print(f"[PAPER][LADDER] Partial exit {step_idx+1}/{len(steps_parts)}: {step_pct*100:.0f}% at {step_target:.2f} ({step_mult}x ATR)")
                                                else:
                                                    # Use first leg's symbol to place the order
                                                    leg0_exit = self._close_single_leg(tr, exit_qty)
                                                    if leg0_exit is not None:
                                                        pass  # already handled
                                            except Exception as exc:
                                                print(f"[LADDER] Exit step {step_idx+1} failed: {exc}")
                                                continue
                                            # Update state
                                            try:
                                                realized_partial = exit_qty * (spot - entry_spot)
                                                tr["realized_pnl"] = float(tr.get("realized_pnl") or 0.0) + float(realized_partial)
                                            except Exception:
                                                pass
                                            steps_done.append(step_idx)
                                            tr["exit_ladder_done"] = steps_done
                                            tr["exit_ladder_last_ts"] = time.time()
                                            qty -= exit_qty
                                            if qty <= lot_size:
                                                break
                                if steps_done:
                                    ladder_done = True
                        except Exception:
                            pass
                            try:
                                if tr.get("spot_target") is None:
                                    tr["spot_target"] = float(partial_target_price)
                            except Exception:
                                pass
                            
                            if spot >= partial_target_price:
                                # Execute Partial Exit
                                exit_qty = calc_partial_exit_qty(qty, partial_pct, int(getattr(self.cfg, "lot_size", 1) or 1))
                                if exit_qty > 0:
                                    if not self.cfg.enable_live_trading:
                                        print(f"[PAPER] PARTIAL EXIT {symbol} {close_side} x{exit_qty} @ spot={spot:.2f} (Target 1 hit)")
                                    else:
                                        try:
                                            self.client.place_order(symbol=symbol, side=close_side, quantity=exit_qty, exchange=exchange, symbol_token=str(token or ""))
                                        except Exception as e:
                                            print(f"Partial exit failed: {e}")
                                            
                                    cur_p = self._try_get_ltp_for_leg({
                                        "symbol": symbol,
                                        "token": token,
                                        "exchange": exchange,
                                    })
                                    if cur_p is not None and entry_price is not None:
                                        sign = 1.0 if entry_side == "BUY" else -1.0
                                        partial_realized = (float(cur_p) - float(entry_price)) * sign * float(exit_qty)
                                        self.state.realized_pnl += partial_realized
                                        
                                        self._emit(
                                            TradeLogEvent(
                                                ts=time.time(),
                                                event="PARTIAL_CLOSE",
                                                trade_id=tr.get("trade_id") or "(unknown)",
                                                position_type="directional",
                                                name=name,
                                                legs=[{
                                                    "symbol": symbol,
                                                    "side": entry_side,
                                                    "quantity": exit_qty,
                                                    "entry_price": entry_price,
                                                    "exit_price": cur_p,
                                                }],
                                                realized=partial_realized,
                                                reason="partial_target",
                                            )
                                        )

                                    # Update Trade State
                                    remaining_qty = int(qty - exit_qty)
                                    tr["quantity"] = remaining_qty
                                    # Keep stored legs in sync so UI/state reflect the true remaining position.
                                    legs_update = tr.get("legs")
                                    if isinstance(legs_update, list):
                                        for leg in legs_update:
                                            if not isinstance(leg, dict):
                                                continue
                                            if bool(leg.get("is_hedge")):
                                                continue
                                            if str(leg.get("symbol") or "") == str(symbol or ""):
                                                leg["quantity"] = remaining_qty
                                                break
                                    tr["partial_exit_done"] = True
                                    print(f"Partial exit done. Remaining: {remaining_qty}")
                                    qty = remaining_qty  # Update local var for subsequent logic


                        if atr_val > 0:
                            # Directional trades ignore ATR stop/target.
                            # BE/trail uses underlying move, independent of MTM sign.
                            # If BE < 0, treat it as an initial stop-loss offset (in ATRs).
                            abs_trail = abs(float(trail_mult_local))
                            if abs_trail <= 0:
                                abs_trail = 1.0

                            be_trigger_mult = max(float(be_mult_local), 0.0)
                            
                            # Explicit ATR Stop Loss
                            sl_mult = float(
                                meta.get("ml_dynamic_dir_sl_atr_mult")
                                if isinstance(meta, dict) and meta.get("ml_dynamic_dir_sl_atr_mult") is not None
                                else getattr(self.cfg, "dir_sl_atr_mult", 1.5)
                            )
                            # ---- #6: Theta-Adjusted Stop Distance ----
                            try:
                                if bool(getattr(self.cfg, "theta_stop_widen_enabled", False)):
                                    theta_thr = float(getattr(self.cfg, "theta_stop_widen_threshold", -5.0) or -5.0)
                                    theta_max_mult = float(getattr(self.cfg, "theta_stop_widen_max_mult", 2.0) or 2.0)
                                    # Find theta for the option legs of this trade
                                    legs = tr.get("legs") if isinstance(tr.get("legs"), list) else []
                                    worst_theta = 0.0
                                    for leg in legs:
                                        if not isinstance(leg, dict):
                                            continue
                                        try:
                                            leg_theta = float(leg.get("theta") or 0.0)
                                        except Exception:
                                            leg_theta = 0.0
                                        if leg_theta < worst_theta:
                                            worst_theta = leg_theta
                                    if worst_theta < float(theta_thr):
                                        # Widen stop based on theta acceleration
                                        theta_ratio = min(float(theta_max_mult), max(1.0, abs(float(worst_theta)) / abs(float(theta_thr))))
                                        sl_mult = float(sl_mult) * float(theta_ratio)
                                        try:
                                            tr["_theta_adjusted_sl_mult"] = float(sl_mult)
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                            loss_mult = -sl_mult
                            if float(be_mult_local) < 0:
                                loss_mult = float(be_mult_local)

                            eff_stop = float(entry_spot) + (loss_mult * float(atr_val))

                            if (best - entry_spot) >= (be_trigger_mult * atr_val):
                                baseline = float(entry_spot)
                                if float(be_mult_local) < 0:
                                    baseline = float(entry_spot) + (float(be_mult_local) * float(atr_val))
                                eff_stop = max(
                                    baseline,
                                    float(best) - (abs_trail * float(atr_val)),
                                )

                            try:
                                tr["spot_stop"] = float(eff_stop)
                            except Exception:
                                pass

                            if spot <= eff_stop:
                                should_exit = True
                                reason = "stop" if (loss_mult < 0 and eff_stop < entry_spot) else "trail_stop"
                    elif name in {"long_put", "short_call"}:
                        # Track best favorable spot (lowest).
                        best = float(tr.get("best_spot") or entry_spot)
                        if spot < best:
                            best = float(spot)
                            tr["best_spot"] = best

                        # Chandelier Exit (Short): Lowest Low + Multiplier * ATR
                        if chan_enabled and atr_val > 0:
                            chan_stop = best + (chan_mult * atr_val)
                            if spot >= chan_stop:
                                should_exit = True
                                reason = "chandelier_stop"

                        # Pivot Target (Short)
                        if pivots and not should_exit:
                            tgt = self._pivot_target_level(pivots, direction="short", entry_spot=entry_spot, mtm_val=mtm_val)
                            if tgt is not None:
                                tgt_price, tgt_key = tgt
                                try:
                                    tr["spot_target"] = float(tgt_price)
                                except Exception:
                                    pass
                                if spot <= tgt_price:
                                    should_exit = True
                                    reason = f"pivot_target_{tgt_key}"

                        # Target Exit (Final TP)
                        tp_mult = float(
                            meta.get("ml_dynamic_dir_tp_atr_mult")
                            if isinstance(meta, dict) and meta.get("ml_dynamic_dir_tp_atr_mult") is not None
                            else getattr(self.cfg, "dir_tp_atr_mult", 0.0)
                        )
                        if tp_mult > 0:
                            tp_price = entry_spot - (tp_mult * atr_val)
                            try:
                                tr["spot_target"] = float(tp_price)
                            except Exception:
                                pass
                            if spot <= tp_price:
                                should_exit = True
                                reason = "spot_target"

                        # Partial Booking Logic for Puts (Single)
                        if partial_mult > 0 and partial_pct > 0 and not bool(tr.get("partial_exit_done")):
                            partial_target_price = entry_spot - (partial_mult * atr_val)
                        # ---- #8: Smart Exit Ladder (Short) ----
                        try:
                            if bool(getattr(self.cfg, "exit_ladder_enabled", False)):
                                ladder_steps_raw = str(getattr(self.cfg, "exit_ladder_steps", "0.25:1.0,0.25:1.5,0.25:2.0") or "")
                                steps_done = tr.get("exit_ladder_done")
                                if not isinstance(steps_done, list):
                                    steps_done = []
                                steps_parts = [s.strip() for s in ladder_steps_raw.split(",") if s.strip()]
                                for step_idx, step_str in enumerate(steps_parts):
                                    if step_idx in steps_done:
                                        continue
                                    parts = step_str.split(":")
                                    if len(parts) != 2:
                                        continue
                                    try:
                                        step_pct = float(parts[0])
                                        step_mult = float(parts[1])
                                    except Exception:
                                        continue
                                    if step_pct <= 0 or step_mult <= 0:
                                        continue
                                    step_target = entry_spot - (step_mult * atr_val)
                                    if spot <= step_target:
                                        exit_qty = calc_partial_exit_qty(total_qty=qty, partial_pct=step_pct, lot_size=int(lot_size))
                                        if exit_qty > 0:
                                            try:
                                                if not bool(self.cfg.enable_live_trading):
                                                    print(f"[PAPER][LADDER] Short partial exit {step_idx+1}/{len(steps_parts)}: {step_pct*100:.0f}% at {step_target:.2f} ({step_mult}x ATR)")
                                                else:
                                                    leg0_exit = self._close_single_leg(tr, exit_qty)
                                                    if leg0_exit is not None:
                                                        pass
                                            except Exception as exc:
                                                print(f"[LADDER] Short exit step {step_idx+1} failed: {exc}")
                                                continue
                                            try:
                                                realized_partial = exit_qty * (entry_spot - spot)
                                                tr["realized_pnl"] = float(tr.get("realized_pnl") or 0.0) + float(realized_partial)
                                            except Exception:
                                                pass
                                            steps_done.append(step_idx)
                                            tr["exit_ladder_done"] = steps_done
                                            tr["exit_ladder_last_ts"] = time.time()
                                            qty -= exit_qty
                                            if qty <= lot_size:
                                                break
                        except Exception:
                            pass
                            try:
                                if tr.get("spot_target") is None:
                                    tr["spot_target"] = float(partial_target_price)
                            except Exception:
                                pass
                            
                            if spot <= partial_target_price:
                                # Execute Partial Exit
                                exit_qty = calc_partial_exit_qty(qty, partial_pct, int(getattr(self.cfg, "lot_size", 1) or 1))
                                if exit_qty > 0:
                                    if not self.cfg.enable_live_trading:
                                        print(f"[PAPER] PARTIAL EXIT {symbol} {close_side} x{exit_qty} @ spot={spot:.2f} (Target 1 hit)")
                                    else:
                                        try:
                                            self.client.place_order(symbol=symbol, side=close_side, quantity=exit_qty, exchange=exchange, symbol_token=str(token or ""))
                                        except Exception as e:
                                            print(f"Partial exit failed: {e}")
                                    
                                    cur_p = self._try_get_ltp_for_leg({
                                        "symbol": symbol,
                                        "token": token,
                                        "exchange": exchange,
                                    })
                                    if cur_p is not None and entry_price is not None:
                                        sign = 1.0 if entry_side == "BUY" else -1.0
                                        partial_realized = (float(cur_p) - float(entry_price)) * sign * float(exit_qty)
                                        self.state.realized_pnl += partial_realized
                                        
                                        self._emit(
                                            TradeLogEvent(
                                                ts=time.time(),
                                                event="PARTIAL_CLOSE",
                                                trade_id=tr.get("trade_id") or "(unknown)",
                                                position_type="directional",
                                                name=name,
                                                legs=[{
                                                    "symbol": symbol,
                                                    "side": entry_side,
                                                    "quantity": exit_qty,
                                                    "entry_price": entry_price,
                                                    "exit_price": cur_p,
                                                }],
                                                realized=partial_realized,
                                                reason="partial_target",
                                            )
                                        )

                                    remaining_qty = int(qty - exit_qty)
                                    tr["quantity"] = remaining_qty
                                    legs_update = tr.get("legs")
                                    if isinstance(legs_update, list):
                                        for leg in legs_update:
                                            if not isinstance(leg, dict):
                                                continue
                                            if bool(leg.get("is_hedge")):
                                                continue
                                            if str(leg.get("symbol") or "") == str(symbol or ""):
                                                leg["quantity"] = remaining_qty
                                                break
                                    tr["partial_exit_done"] = True
                                    print(f"Partial exit done. Remaining: {remaining_qty}")
                                    qty = remaining_qty


                        if atr_val > 0:
                            abs_trail = abs(float(trail_mult_local))
                            if abs_trail <= 0:
                                abs_trail = 1.0

                            be_trigger_mult = max(float(be_mult_local), 0.0)
                            
                            # Explicit ATR Stop Loss
                            sl_mult = float(
                                meta.get("ml_dynamic_dir_sl_atr_mult")
                                if isinstance(meta, dict) and meta.get("ml_dynamic_dir_sl_atr_mult") is not None
                                else getattr(self.cfg, "dir_sl_atr_mult", 1.5)
                            )
                            # ---- #6: Theta-Adjusted Stop Distance ----
                            try:
                                if bool(getattr(self.cfg, "theta_stop_widen_enabled", False)):
                                    theta_thr = float(getattr(self.cfg, "theta_stop_widen_threshold", -5.0) or -5.0)
                                    theta_max_mult = float(getattr(self.cfg, "theta_stop_widen_max_mult", 2.0) or 2.0)
                                    # Find theta for the option legs of this trade
                                    legs = tr.get("legs") if isinstance(tr.get("legs"), list) else []
                                    worst_theta = 0.0
                                    for leg in legs:
                                        if not isinstance(leg, dict):
                                            continue
                                        try:
                                            leg_theta = float(leg.get("theta") or 0.0)
                                        except Exception:
                                            leg_theta = 0.0
                                        if leg_theta < worst_theta:
                                            worst_theta = leg_theta
                                    if worst_theta < float(theta_thr):
                                        # Widen stop based on theta acceleration
                                        theta_ratio = min(float(theta_max_mult), max(1.0, abs(float(worst_theta)) / abs(float(theta_thr))))
                                        sl_mult = float(sl_mult) * float(theta_ratio)
                                        try:
                                            tr["_theta_adjusted_sl_mult"] = float(sl_mult)
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                            loss_mult = -sl_mult
                            if float(be_mult_local) < 0:
                                loss_mult = float(be_mult_local)

                            eff_stop = float(entry_spot) - (loss_mult * float(atr_val))

                            if (entry_spot - best) >= (be_trigger_mult * atr_val):
                                baseline = float(entry_spot)
                                if float(be_mult_local) < 0:
                                    baseline = float(entry_spot) - (float(be_mult_local) * float(atr_val))
                                eff_stop = min(
                                    baseline,
                                    float(best) + (abs_trail * float(atr_val)),
                                )

                            try:
                                tr["spot_stop"] = float(eff_stop)
                            except Exception:
                                pass

                            if spot >= eff_stop:
                                should_exit = True
                                reason = "stop" if (loss_mult < 0 and eff_stop > entry_spot) else "trail_stop"

                # 4) Premium Trailing Stop (Theta protection)
                # If enabled, exit if option premium drops by X% from peak.
                # ONLY active when MTM is positive.
                if not should_exit and cur_premium_val is not None and cur_premium_val > 0 and mtm_val is not None and float(mtm_val) > 0:
                    try:
                        prem_trail_pct = float(getattr(self.cfg, "dir_premium_trail_pct", 0.0) or 0.0)
                        if prem_trail_pct > 0:
                            # Tracking high-water mark of premium value (absolute rupees)
                            best_prem = float(tr.get("best_premium_val") or 0.0)
                            if cur_premium_val > best_prem:
                                best_prem = cur_premium_val
                                tr["best_premium_val"] = best_prem
                            
                            # Check trailing stop condition
                            # Stop level = Best * (1 - pct)
                            stop_level = best_prem * (1.0 - prem_trail_pct)
                            if cur_premium_val < stop_level:
                                should_exit = True
                                reason = f"premium_trail (dropped {prem_trail_pct*100:.1f}%)"
                    except Exception:
                        pass

                # Time-based exits (minutes and/or days).
                max_hold = int(self.cfg.max_hold_minutes)
                try:
                    max_hold_days = int(getattr(self.cfg, "max_hold_days", 0) or 0)
                except Exception:
                    max_hold_days = 0
                opened = float(tr.get("opened_ts") or 0.0)
                held_min = (time.time() - opened) / 60.0 if opened else 0.0
                held_days = self._held_days_since_open(opened)

                # Daily force-exit should respect max_hold_days:
                # - max_hold_days=0 => intraday behavior (always force-exit)
                # - max_hold_days>0 => only force-exit on/after the max-hold day
                force_exit_due = bool(now_t >= force_exit_t) and (max_hold_days <= 0 or held_days >= max_hold_days)
                if force_exit_due:
                    should_exit = True
                    reason = reason or "time_exit"

                if max_hold_days > 0 and held_days >= max_hold_days:
                    should_exit = True
                    reason = reason or "max_hold_days"
                elif max_hold > 0 and held_min >= max_hold:
                    should_exit = True
                    reason = reason or "time"

                if should_exit:
                    # Optional: flip long option to equivalent short structure on stop.
                    # This is intentionally conservative: only for long_call/long_put and only on stop/trailing-stop.
                    do_flip = (
                        flip_enabled
                        and spot is not None
                        and str(tr.get("name") or "") in {"long_call", "long_put"}
                        and str(reason or "") in {"stop", "trail_stop"}
                    )
                    flip_opt = None
                    flip_name = ""
                    if do_flip:
                        if flip_chain is None:
                            try:
                                flip_chain = self.client.get_option_chain(self.cfg.underlying)
                            except Exception:
                                flip_chain = []

                        # Use current ATR if available; fall back to entry ATR.
                        atr_for_flip = None
                        try:
                            if atr_now is not None:
                                atr_for_flip = float(atr_now)
                        except Exception:
                            atr_for_flip = None
                        if atr_for_flip is None:
                            try:
                                atr_for_flip = float(atr_val)
                            except Exception:
                                atr_for_flip = 0.0

                        if flip_chain:
                            strike_offset = self._get_directional_strike_offset(atr_for_flip)
                            if str(tr.get("name") or "") == "long_put":
                                # Bearish long put stopped out -> flip to bearish short call
                                flip_name = "short_call"
                                target_call_strike = float(spot) + float(strike_offset)
                                flip_opt = self._pick_nearest_strike(flip_chain, "CE", target_call_strike)
                            elif str(tr.get("name") or "") == "long_call":
                                # Bullish long call stopped out -> flip to bullish short put
                                flip_name = "short_put"
                                target_put_strike = float(spot) - float(strike_offset)
                                flip_opt = self._pick_nearest_strike(flip_chain, "PE", target_put_strike)

                    symbol = str(tr.get("symbol") or "")
                    qty = int(tr.get("quantity") or 0)
                    token = str(tr.get("token") or "").strip() or None
                    exchange = str(tr.get("exchange") or "").strip() or None
                    trade_id = str(tr.get("trade_id") or "")
                    exit_is_stop = reason in {"stop", "mtm_stop", "gpt_leg_stop"}
                    print(f"Closing directional trade {symbol} due to {reason}.")
                    if not self.cfg.enable_live_trading:
                        print(f"[PAPER] {close_side} {symbol} x{qty}")

                        if exit_is_stop:
                            self._mark_stopout()
                        else:
                            self._mark_non_stop_exit()

                        self.state.last_exit_ts = time.time()

                        entry = tr.get("entry_price")
                        try:
                            entry_f = float(entry) if entry is not None else None
                        except Exception:
                            entry_f = None
                        cur = self._try_get_ltp_for_leg({
                            "symbol": symbol,
                            "token": token,
                            "exchange": exchange,
                        })
                        realized = None
                        if entry_f is not None and cur is not None:
                            side = str(tr.get("side") or "").upper()
                            sign = 1.0 if side == "BUY" else -1.0
                            realized = (float(cur) - float(entry_f)) * sign * float(qty)

                        # Close any hedge legs attached to this directional trade (if enabled).
                        hedge_realized = 0.0
                        hedge_legs_out: List[Dict[str, object]] = []
                        legs_src2 = tr.get("legs")
                        if isinstance(legs_src2, list):
                            for hl in legs_src2:
                                if not isinstance(hl, dict):
                                    continue
                                if not bool(hl.get("is_hedge")):
                                    continue
                                hs = str(hl.get("symbol") or "")
                                ht = str(hl.get("token") or "").strip() or None
                                he = str(hl.get("exchange") or "").strip() or None
                                hside = str(hl.get("side") or "").upper()
                                hqty = int(hl.get("quantity") or 0)
                                if not hs or hside not in {"BUY", "SELL"} or hqty <= 0:
                                    continue
                                hclose = "SELL" if hside == "BUY" else "BUY"
                                print(f"[PAPER] Close hedge: {hclose} {hs} x{hqty}")
                                h_entry = hl.get("entry_price")
                                try:
                                    h_entry_f = float(h_entry) if h_entry is not None else None
                                except Exception:
                                    h_entry_f = None
                                h_cur = self._try_get_ltp_for_leg({"symbol": hs, "token": ht, "exchange": he})
                                if h_entry_f is not None and h_cur is not None:
                                    h_sign = 1.0 if hside == "BUY" else -1.0
                                    hedge_realized += (float(h_cur) - float(h_entry_f)) * h_sign * float(hqty)
                                hedge_legs_out.append({
                                    "symbol": hs,
                                    "token": ht,
                                    "exchange": he,
                                    "side": hside,
                                    "quantity": hqty,
                                    "entry_price": h_entry_f,
                                    "exit_price": h_cur,
                                    "is_hedge": True,
                                })

                        total_realized = None
                        if realized is not None:
                            total_realized = float(realized) + float(hedge_realized)
                            self.state.realized_pnl += float(total_realized)
                        else:
                            # If we couldn't compute main realized, still track hedge if any.
                            if hedge_realized != 0.0:
                                self.state.realized_pnl += float(hedge_realized)
                        res_for_note = total_realized if total_realized is not None else realized
                        if res_for_note is None and hedge_realized != 0.0:
                            res_for_note = hedge_realized
                        close_reason = self._normalize_close_reason(str(reason or ""), res_for_note, bool(exit_is_stop))
                        self._log_directional_close_audit(
                            symbol=str(symbol),
                            side=str(tr.get("side") or ""),
                            qty=int(qty),
                            entry_price=entry_f,
                            exit_price=cur,
                            realized_pnl=res_for_note,
                            reason=close_reason,
                        )
                        self._note_trade_result(res_for_note)
                        if res_for_note is not None:
                            self._strategy_winrate_record_result(str(tr.get("name") or ""), float(res_for_note) >= 0)

                        legs = [
                            {
                                "symbol": symbol,
                                "token": token,
                                "exchange": exchange,
                                "side": str(tr.get("side") or ""),
                                "quantity": qty,
                                "entry_price": entry_f,
                                "exit_price": cur,
                                "strike": tr.get("strike"),
                                "option_type": tr.get("option_type"),
                                "expiry": tr.get("expiry"),
                            }
                        ] + hedge_legs_out
                        self._emit(
                            TradeLogEvent(
                                ts=time.time(),
                                event="CLOSE",
                                trade_id=trade_id or "(unknown)",
                                position_type="directional",
                                name=str(tr.get("name") or ""),
                                legs=legs,
                                realized=float(total_realized) if total_realized is not None else (float(realized) if realized is not None else None),
                                reason=close_reason,
                            )
                        )

                        # Flip in paper mode after closing.
                    if do_flip and flip_opt is not None and flip_name and spot is not None:
                        try:
                            atr_for_flip_val = float(atr_now) if atr_now is not None else float(atr_val)
                        except Exception:
                            atr_for_flip_val = float(atr_val)
                        new_tr = self._open_directional_from_option(
                            flip_opt,
                            name=flip_name,
                            spot=float(spot),
                            atr_val=float(atr_for_flip_val),
                            append_state=False,
                        )
                        if isinstance(new_tr, dict):
                            new_tr["flipped_from"] = trade_id
                            new_dir_trades.append(new_tr)
                    else:
                        try:
                            # Close hedge legs first (if any).
                            legs_src2 = tr.get("legs")
                            if isinstance(legs_src2, list):
                                for hl in legs_src2:
                                    if not isinstance(hl, dict):
                                        continue
                                    if not bool(hl.get("is_hedge")):
                                        continue
                                    hs = str(hl.get("symbol") or "")
                                    ht = str(hl.get("token") or "").strip() or None
                                    he = str(hl.get("exchange") or "").strip() or None
                                    hside = str(hl.get("side") or "").upper()
                                    hqty = int(hl.get("quantity") or 0)
                                    if not hs or hside not in {"BUY", "SELL"} or hqty <= 0:
                                        continue
                                    hclose = "SELL" if hside == "BUY" else "BUY"
                                    self.client.place_order(
                                        symbol=hs,
                                        side=hclose,
                                        quantity=hqty,
                                        exchange=he,
                                        symbol_token=ht,
                                    )

                            self.client.place_order(
                                symbol=symbol,
                                side=close_side,
                                quantity=qty,
                                exchange=exchange,
                                symbol_token=token,
                            )

                            if exit_is_stop:
                                self._mark_stopout()
                            else:
                                self._mark_non_stop_exit()

                            self.state.last_exit_ts = time.time()
                            # Track P&L in live mode too (for max_daily_loss enforcement)
                            entry = tr.get("entry_price")
                            try:
                                entry_f = float(entry) if entry is not None else None
                            except Exception:
                                entry_f = None
                            cur = self._try_get_ltp_for_leg({
                                "symbol": symbol,
                                "token": token,
                                "exchange": exchange,
                            })
                            if entry_f is not None and cur is not None:
                                side = str(tr.get("side") or "").upper()
                                sign = 1.0 if side == "BUY" else -1.0
                                realized = (float(cur) - float(entry_f)) * sign * float(qty)
                                self.state.realized_pnl += float(realized)
                                close_reason = self._normalize_close_reason(str(reason or ""), realized, bool(exit_is_stop))
                                self._log_directional_close_audit(
                                    symbol=str(symbol),
                                    side=str(tr.get("side") or ""),
                                    qty=int(qty),
                                    entry_price=entry_f,
                                    exit_price=cur,
                                    realized_pnl=realized,
                                    reason=close_reason,
                                )
                                self._note_trade_result(realized)
                                self._strategy_winrate_record_result(str(tr.get("name") or ""), float(realized) >= 0)

                            # Flip in live mode after the close succeeds.
                            if do_flip and flip_opt is not None and flip_name and spot is not None:
                                try:
                                    atr_for_flip_val = float(atr_now) if atr_now is not None else float(atr_val)
                                except Exception:
                                    atr_for_flip_val = float(atr_val)
                                new_tr = self._open_directional_from_option(
                                    flip_opt,
                                    name=flip_name,
                                    spot=float(spot),
                                    atr_val=float(atr_for_flip_val),
                                    append_state=False,
                                )
                                if isinstance(new_tr, dict):
                                    new_tr["flipped_from"] = trade_id
                                    new_dir_trades.append(new_tr)
                        except Exception as exc:  # noqa: BLE001
                            print(f"Failed to close {symbol}: {exc}")
                            tr["close_attempts"] = int(tr.get("close_attempts", 0)) + 1
                            tr["last_close_attempt_ts"] = time.time()
                            if tr["close_attempts"] >= 3:
                                tr["close_failed"] = True
                            remaining_dir.append(tr)
                    continue

                remaining_dir.append(tr)

            self.state.open_directional = remaining_dir + new_dir_trades

        # ---- Manage multi-leg exits ----
        if not self.state.open_multi:
            return

        # Spot snapshot for delta hedging.
        try:
            spot_for_hedge = float(self.client.get_ltp(self.cfg.underlying))
        except Exception:
            spot_for_hedge = None

        max_hold = int(self.cfg.max_hold_minutes)
        try:
            max_hold_days = int(getattr(self.cfg, "max_hold_days", 0) or 0)
        except Exception:
            max_hold_days = 0
        now = time.time()

        remaining: List[Dict[str, object]] = []
        for trade in self.state.open_multi:
            opened = float(trade.get("opened_ts") or 0.0)
            held_min = (now - opened) / 60.0 if opened else 0.0

            held_days = self._held_days_since_open(opened)

            days_exit = max_hold_days > 0 and held_days >= max_hold_days
            time_exit = days_exit or (max_hold > 0 and held_min >= max_hold)
            breakout_exit = trend_strength > float(self.cfg.max_trend_strength) * 2.0

            meta = trade.get("meta") if isinstance(trade.get("meta"), dict) else {}
            name = str(trade.get("name") or "")
            premium_selling = False
            if isinstance(meta, dict) and bool(meta.get("premium_selling")):
                premium_selling = True
            else:
                premium_selling = name.startswith("short_") or name == "iron_condor"
            legs = trade.get("legs")
            if not isinstance(legs, list):
                remaining.append(trade)
                continue

            # Apply delta-hedge scope for multi-leg trades (multi_only/all_options).
            meta = self._apply_delta_hedge_scope(trade, position_type="multi")

            # Delta hedging: rebalance before evaluating exits (unless we're about to force/time exit).
            try:
                if (
                    isinstance(meta, dict)
                    and bool(meta.get("delta_hedge"))
                    and spot_for_hedge is not None
                    and (now_t < force_exit_t)
                    and (not bool(time_exit))
                ):
                    self._rebalance_delta_hedge(trade, spot=float(spot_for_hedge))
            except Exception:
                pass

            # 1) MTM stop/target based on entry premium (absolute)
            mtm_val = self._compute_legs_mtm(legs)

            # Optional: GPT per-leg premium stop/target management.
            try:
                is_paper = not bool(getattr(self.cfg, "enable_live_trading", False))
            except Exception:
                is_paper = True
            if self._gpt_leg_manage_enabled(is_paper=bool(is_paper)):
                try:
                    self._apply_gpt_leg_exits(trade=trade, position_type="multi", legs=legs)
                except Exception:
                    pass

            # Enforce per-leg exits: close only the leg that hit.
            try:
                any_leg_closed = False
                any_leg_stop = False
                any_leg_non_stop = False
                exits_armed = self._gpt_leg_exits_armed(trade=trade)
                for lg in legs:
                    if not isinstance(lg, dict) or bool(lg.get("is_hedge")):
                        continue
                    hit = self._leg_exit_hit(leg=lg)
                    if not hit:
                        continue
                    # Hard safety rule: breached stop must close now; targets remain gated.
                    if hit != "leg_stop" and not exits_armed:
                        continue
                    try:
                        qty_before = int(lg.get("quantity") or 0)
                    except Exception:
                        qty_before = 0
                    reason = f"gpt_{hit}"
                    self._close_single_leg(trade=trade, leg=lg, position_type="multi", reason=reason)
                    try:
                        qty_after = int(lg.get("quantity") or 0)
                    except Exception:
                        qty_after = qty_before
                    if qty_before > 0 and qty_after <= 0:
                        any_leg_closed = True
                        if hit == "leg_stop":
                            any_leg_stop = True
                        else:
                            any_leg_non_stop = True
                if any_leg_closed:
                    if any_leg_stop:
                        self._mark_stopout()
                    elif any_leg_non_stop:
                        self._mark_non_stop_exit()
                    # If all main legs are closed, remove the trade.
                    any_open = False
                    for lg in legs:
                        if not isinstance(lg, dict) or bool(lg.get("is_hedge")):
                            continue
                        try:
                            if int(lg.get("quantity") or 0) > 0:
                                any_open = True
                                break
                        except Exception:
                            continue
                    if not any_open:
                        # Emit a CLOSE snapshot for UI consistency.
                        try:
                            self._emit(
                                TradeLogEvent(
                                    ts=time.time(),
                                    event="CLOSE",
                                    trade_id=str(trade.get("trade_id") or "") or "(unknown)",
                                    position_type="multi",
                                    name=str(trade.get("name") or ""),
                                    legs=[dict(l) for l in legs if isinstance(l, dict)],
                                    realized=None,
                                    mtm=None,
                                    reason="all_legs_closed",
                                )
                            )
                        except Exception:
                            pass
                        continue
            except Exception:
                pass
            entry_abs = None
            entry_atr = None
            if isinstance(meta, dict):
                try:
                    if meta.get("entry_premium_abs") is not None:
                        entry_abs = float(meta.get("entry_premium_abs"))
                    elif meta.get("entry_premium") is not None:
                        entry_abs = abs(float(meta.get("entry_premium")))
                except Exception:
                    entry_abs = None
                try:
                    if meta.get("entry_atr") is not None:
                        entry_atr = float(meta.get("entry_atr"))
                    elif meta.get("atr") is not None:
                        entry_atr = float(meta.get("atr"))
                except Exception:
                    entry_atr = None

            # Optional: allow GPT to dynamically adjust MTM stop/target for live trades.
            # This is deliberately best-effort and never blocks exits.
            stop_pct_eff = float(stop_pct)
            target_pct_eff = float(target_pct)
            try:
                gpt_on = bool(getattr(self.cfg, "gpt_enable", False))
            except Exception:
                gpt_on = False
            if gpt_on and bool(getattr(self.cfg, "enable_live_trading", False)):
                advise_fn = getattr(self, "_gpt_control_advise", None)
                if callable(advise_fn) and isinstance(meta, dict) and mtm_val is not None and entry_abs is not None and entry_abs > 0:
                    try:
                        proposal = {
                            "position_type": "multi",
                            "trade_id": str(trade.get("trade_id") or ""),
                            "name": str(trade.get("name") or ""),
                            "mtm": float(mtm_val),
                            "entry_premium_abs": float(entry_abs),
                            "default_mtm_stop_pct": float(stop_pct_eff),
                            "default_mtm_target_pct": float(target_pct_eff),
                        }
                        system_prompt = (
                            "You are a trading risk manager. If you want to adjust the MTM exit thresholds for this open trade, "
                            "return them in `extras` as numeric fields: mtm_target_pct, mtm_stop_pct."
                        )
                        advice = advise_fn(proposal=proposal, system_prompt=system_prompt, max_tokens=200)
                        extras = getattr(advice, "extras", None)
                        if isinstance(extras, dict):
                            if extras.get("mtm_target_pct") is not None:
                                try:
                                    tgt = float(extras.get("mtm_target_pct"))
                                    if tgt > 0:
                                        target_pct_eff = float(tgt)
                                        meta["gpt_mtm_target_pct"] = float(tgt)
                                except Exception:
                                    pass
                            if extras.get("mtm_stop_pct") is not None:
                                try:
                                    stp = float(extras.get("mtm_stop_pct"))
                                    if stp > 0 and stp >= float(stop_pct_eff):
                                        stop_pct_eff = float(stp)
                                        meta["gpt_mtm_stop_pct"] = float(stp)
                                except Exception:
                                    pass

                        # Persist back onto trade so external references (tests/UI) can see it.
                        trade["meta"] = meta
                    except Exception:
                        pass

            if mtm_val is not None and entry_abs is not None and entry_abs > 0:
                if float(mtm_val) <= -stop_pct_eff * float(entry_abs):
                    print("[MTM STOP]")
                    self._mark_stopout()
                    self._close_multi_trade(trade, reason="mtm_stop")
                    continue
                if float(target_pct_eff) > 0 and float(mtm_val) >= target_pct_eff * float(entry_abs):
                    print("[MTM TARGET]")
                    self._mark_non_stop_exit()
                    self._close_multi_trade(trade, reason="mtm_target")
                    continue
            

            # 2) Hard time exit (non-negotiable) - REMOVED DUPLICATE, see below at step 5

            # 3) Volatility spike kill-switch
            # Do not let volatility spikes override MTM-based exits.
            # Only use vol_spike as a fallback safety exit when MTM/entry premium
            # are not available (e.g. missing quotes).
            if (mtm_val is None or entry_abs is None or entry_abs <= 0) and atr_now is not None and entry_atr is not None and entry_atr > 0 and spike_mult > 0:
                spike = float(atr_now) > spike_mult * float(entry_atr)
                if spike:
                    held_sec = (now - opened) if opened else 0.0

                    # Avoid knee-jerk exits right after opening.
                    min_hold_sec = 30.0
                    confirm_hits = 3

                    if held_sec >= min_hold_sec:
                        hits = int(trade.get("vol_spike_hits") or 0) + 1
                        trade["vol_spike_hits"] = hits
                        if hits >= confirm_hits:
                            print("[VOL SPIKE EXIT]")
                            self._mark_non_stop_exit()
                            self._close_multi_trade(trade, reason="vol_spike")
                            continue
                    else:
                        trade["vol_spike_hits"] = 0
                else:
                    trade["vol_spike_hits"] = 0

            # 4) Premium MTM Trailing Stop (Protect Profits)
            # Active when MTM > X% (start_pct) of entry premium.
            # Stop if MTM drops by Y% (stop_pct) of entry premium from HWM.
            if mtm_val is not None and entry_abs is not None and entry_abs > 0:
                try:
                    trail_start_pct = float(getattr(self.cfg, "premium_mtm_trail_start_pct", 0.05))
                    trail_stop_pct = float(getattr(self.cfg, "premium_mtm_trail_stop_pct", 0.05))

                    current_mtm = float(mtm_val)
                    
                    # Initialize HWM if not present
                    # Start tracking HWM only if MTM is above the start threshold? 
                    # Or always track HWM but only ACTIVATE trailing stop if it was above threshold?
                    # Better: Always track HWM. Activate stop if HWM > entry * start_pct.
                    
                    best_mtm = float(trade.get("best_mtm_val") or -9e9)
                    if current_mtm > best_mtm:
                        best_mtm = current_mtm
                        trade["best_mtm_val"] = best_mtm
                        
                    # Check activation
                    activation_thresh = entry_abs * trail_start_pct
                    is_active = best_mtm >= activation_thresh
                    
                    if is_active and trail_stop_pct > 0:
                         # Stop level = Best MTM * (1 - trail_pct)
                         # This means we allow MTM to drop by trail_pct % from the Peak.
                         mtm_sl_val = best_mtm * (1.0 - trail_stop_pct)
                         
                         if current_mtm < mtm_sl_val:
                             reason_str = f"mtm_trail (peak={best_mtm:.1f}, drop={trail_stop_pct*100:.1f}%)"
                             print(f"[MTM TRAIL] {reason_str} Cur:{current_mtm:.1f} SL:{mtm_sl_val:.1f}")
                             self._mark_non_stop_exit()
                             self._close_multi_trade(trade, reason=reason_str)
                             continue

                except Exception:
                    pass

            # 5) Time-based exits (consolidates both force_exit and max_hold)
            force_exit_due = bool(now_t >= force_exit_t) and (max_hold_days <= 0 or held_days >= max_hold_days)
            if force_exit_due:
                print("[FORCE TIME EXIT]")
                self._mark_non_stop_exit()
                self._close_multi_trade(trade, reason="time_exit")
                continue
            if time_exit:
                print("[MAX HOLD EXIT]")
                self._mark_non_stop_exit()
                self._close_multi_trade(trade, reason="max_hold_days" if days_exit else "max_hold")
                continue

            # Non-premium multi-leg: keep legacy trend-breakout exit.
            # (Time exits are handled above for all trades.)
            if (not premium_selling) and breakout_exit and not bool(isinstance(meta, dict) and meta.get("delta_hedge")):
                reason = "trend breakout"
                print(f"Closing multi-leg trade ({trade.get('name')}) due to {reason}.")
                self._mark_non_stop_exit()
                self._close_multi_trade(trade, reason=reason)
                continue

            remaining.append(trade)

        self.state.open_multi = remaining

    def run_forever(self, stop_event: Optional[Event] = None) -> None:
        """Main blocking loop.

        Call this from your entrypoint after `client.login()`.
        """
        while True:
            if stop_event is not None and stop_event.is_set():
                print("Stop requested. Exiting scalper loop.")
                break

            # External env-var kill switch: MSTOCK_KILL_SWITCH or SCALPER_KILL_SWITCH
            # instantly halts all new entries and safely exits all open positions.
            try:
                if self._kill_switch_active():
                    ks_reason = "MSTOCK_KILL_SWITCH" if self._bool_env("MSTOCK_KILL_SWITCH", False) else "SCALPER_KILL_SWITCH"
                    print(
                        f"[KILL_SWITCH] {ks_reason} active — "
                        "blocking all entries and exiting open positions."
                    )
                    try:
                        self._safe_exit_all_positions(reason=ks_reason)
                    except Exception as exc_kill:
                        print(f"[KILL_SWITCH] Error during safe exit: {exc_kill}")
                    break
            except Exception:
                pass

            # External env-var paper-force: SCALPER_FORCE_PAPER=true forces paper mode
            # regardless of cfg.enable_live_trading. Checked every iteration so that
            # even a mid-session toggle takes effect without a restart.
            try:
                force_paper_raw = os.getenv("SCALPER_FORCE_PAPER", "").strip().lower()
                if force_paper_raw in {"1", "true", "yes"}:
                    if self.cfg.enable_live_trading:
                        print(
                            "[SCALPER_FORCE_PAPER] SCALPER_FORCE_PAPER=true detected — "
                            "overriding enable_live_trading=True and forcing paper mode."
                        )
                    self.cfg.enable_live_trading = False
            except Exception:
                pass

            ok, reason = self._risk_status()
            if not ok:
                print(f"Risk limit hit ({reason}). Stopping scalper loop.")
                break

            # Market-hours guard: block live trading outside hours, but allow
            # paper trading so users can test the strategy after hours.
            if not is_market_open():
                if bool(self.cfg.enable_live_trading):
                    print(f"Market closed ({self._now_ist_time()} IST) — waiting...")
                    if stop_event is not None and stop_event.wait(120):
                        print("Stop requested. Exiting scalper loop.")
                        break
                    continue
                else:
                    # In paper mode, we still run the loop so that demo
                    # trades can be generated even when exchange is closed.
                    print("Market closed (paper mode) — generating demo signals anyway.")

            try:
                # Only evaluate entries once per timeframe bucket to avoid aggressive
                # repeated decisions on the same candle.
                try:
                    bucket_sec = int(self._tf_bucket_seconds(str(getattr(self.cfg, "timeframe", "1m"))))
                except Exception:
                    bucket_sec = 60
                if bucket_sec <= 0:
                    bucket_sec = 60

                now_ts = float(time.time())
                bucket_start_ts = now_ts - (now_ts % float(bucket_sec))
                # ---- Reset reroute cycle counter every hour ----
                if self._cycle_reroute_reset_ts == 0.0 or (now_ts - self._cycle_reroute_reset_ts) >= 3600:
                    self._reroute_count_this_cycle = 0
                    self._cycle_reroute_reset_ts = now_ts
                # P4: Periodic regime monitor
                if getattr(self.cfg, "gpt_regime_monitor_enabled", False):
                    try:
                        self._gpt_regime_monitor()
                    except Exception:
                        pass

                # P6: Periodic what-if analysis
                if getattr(self.cfg, "gpt_whatif_enabled", False):
                    try:
                        self._gpt_whatif_analysis()
                    except Exception:
                        pass

                if self._last_entry_bucket_start_ts is None or float(self._last_entry_bucket_start_ts) != float(bucket_start_ts):
                    self._last_entry_bucket_start_ts = float(bucket_start_ts)
                    self._decide_entries()

                # Equity holdings trading (optional)
                self._manage_equity_positions()
                self._decide_equity_entries()

                # Standalone portfolio beta hedge (optional)
                self._rebalance_portfolio_beta_hedge()

                self._manage_open_trades()
                self._emit_paper_mtm_updates()

                try:
                    import json
                    with open(".engine_diagnostics.json", "w") as f:
                        json.dump(self.get_runtime_diagnostics(), f)
                except Exception:
                    pass
            except Exception as exc:  # noqa: BLE001
                print(f"Error in strategy loop: {exc}")

            try:
                sleep_sec = float(self.cfg.polling_interval_sec)
            except Exception:
                sleep_sec = 1.0
            if stop_event is not None and stop_event.wait(max(0.0, sleep_sec)):
                print("Stop requested. Exiting scalper loop.")
                break
