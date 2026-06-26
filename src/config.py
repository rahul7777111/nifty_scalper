import json
import os
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple


@dataclass
class APIConfig:
    base_url: str
    api_key: str
    api_secret: str
    client_id: str
    # Add any other fields required by m.Stock Type B (e.g. vendor code, app id)
    vendor_code: Optional[str] = None


@dataclass
class StrategyConfig:
    symbol: str = "NIFTY"
    underlying: str = "NIFTY"
    underlying_token: str = ""
    underlying_exchange: str = "NSE"
    timeframe: str = "1m"
    # Default NIFTY options lot size (can be overridden via MSTOCK_LOT_SIZE).
    lot_size: int = 65
    max_open_positions: int = 6
    max_daily_loss: float = 5000.0
    max_daily_profit: float = 10000.0
    polling_interval_sec: float = 1.0
    enable_live_trading: bool = False
    ml_paper_mode_enabled: bool = False
    ml_deployment_manifest_path: str = ""
    ml_min_confidence_threshold: float = 0.0
    ml_max_predictions_per_day: int = 1000
    ml_log_feature_vector: bool = True
    ml_log_prediction_reason: bool = True
    ml_fail_closed_on_schema_mismatch: bool = True
    ml_disable_all: bool = False
    ml_max_daily_paper_loss: float = 5000.0
    ml_max_consecutive_paper_losses: int = 3

    # ---- Micro-live (live probe) mode — extreme safety defaults ----
    enable_micro_live: bool = False                        # DISABLED by default
    micro_live_max_trades_per_day: int = 1                 # 1 trade per day max
    micro_live_max_lots: int = 1                           # 1 lot max
    micro_live_min_probability: float = 0.70              # 70% confidence minimum
    micro_live_max_daily_loss: float = 500.0               # Rs 500 max daily loss
    micro_live_force_exit_time: str = "15:10"              # auto-squareoff by 3:10 PM
    micro_live_require_spread_pct: float = 0.01           # max 1% spread
    micro_live_require_min_premium: float = 10.0          # min Rs 10 premium
    micro_live_strict_kill_switch: bool = True            # kill switch MUST be False

    # ---- Paper Forward-Test Runner ----
    paper_forward_test_interval_sec: int = 300       # 5 min between checks
    paper_forward_test_max_trades_per_day: int = 5   # stop after N trades
    paper_forward_test_output_dir: str = "reports/paper_forward_test"
    paper_forward_test_require_market_hours: bool = True
    paper_forward_test_min_runtime_minutes: int = 30  # run at least 30 min

    # ---- Notifier Settings ----
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    
    # ---- Dynamic Position Sizing ----
    account_capital: float = 100000.0
    risk_per_trade_percentage: float = 0.0  # 0 disables dynamic sizing

    # Data-quality guardrail: block new entries if spot LTP appears stale.
    # 0 disables. Default is conservative for index trading.
    max_stale_ltp_sec: float = 20.0
    # Entry guardrails based on candle freshness / volatility.
    # 0 disables each guard.
    entry_candle_max_age_sec: float = 120.0
    # Block entries if the latest candle's range (high-low) exceeds ATR * X.
    entry_max_candle_range_atr_mult: float = 2.5
    # Block entries if the gap between last candle open and prior close exceeds ATR * X.
    entry_gap_atr_mult: float = 1.5
    # Session windows (IST, HH:MM) and per-session overrides.
    session_open_start_hhmm: str = "09:15"
    session_open_end_hhmm: str = "10:00"
    session_close_start_hhmm: str = "14:30"
    session_close_end_hhmm: str = "15:30"
    session_open_min_atr: Optional[float] = None
    session_open_max_atr: Optional[float] = None
    session_open_premium_rsi_low: Optional[float] = None
    session_open_premium_rsi_high: Optional[float] = None
    session_mid_min_atr: Optional[float] = None
    session_mid_max_atr: Optional[float] = None
    session_mid_premium_rsi_low: Optional[float] = None
    session_mid_premium_rsi_high: Optional[float] = None
    session_close_min_atr: Optional[float] = None
    session_close_max_atr: Optional[float] = None
    session_close_premium_rsi_low: Optional[float] = None
    session_close_premium_rsi_high: Optional[float] = None

    # Strategy selection
    # - "directional" (existing EMA/RSI/candle + delta template)
    # - "short_straddle"
    # - "short_strangle"
    # - "bull_call_spread" (buy ATM call + sell higher-strike call)
    # - "bull_put_spread" (sell OTM put + buy lower-strike put)
    # - "call_ratio_backspread" (sell 1x ATM call + buy 2x higher-strike calls)
    # - "put_ratio_backspread" (sell 1x ATM put + buy 2x lower-strike puts)
    # - "iron_condor"
    # - "long_straddle" (ATM CE+PE buy)
    # - "long_strangle" (OTM CE+PE buy)
    # - "auto" (choose between directional vs a range-bound multi-leg strategy based on regime)
    strategy_name: str = "directional"

    # ---- Delta-hedging (dynamic) ----
    # Used by strategies like "delta_hedged_short_straddle" / "delta_hedged_long_straddle".
    # Hedge instrument must be a tradable symbol that the broker can place orders for.
    # Examples: an index future, an ETF (e.g. NIFTYBEES), or any liquid proxy.
    # Scope:
    # - "strategy_only" (default): hedge only trades created with meta["delta_hedge"]=True
    # - "multi_only": hedge all multi-leg option trades
    # - "all_options": hedge both multi-leg and directional option trades
    delta_hedge_scope: str = "multi_only"
    # Optional per-underlying hedge symbols. If set, these take precedence over
    # the global `delta_hedge_symbol` when the trade underlying can be inferred.
    # Examples: NFO futures or NSE ETFs.
    delta_hedge_symbol_nifty: str = ""  # e.g. "NFO:NIFTY26FEBFUT" or "NSE:NIFTYBEES"
    delta_hedge_symbol_banknifty: str = ""  # e.g. "NFO:BANKNIFTY26FEBFUT"
    delta_hedge_exchange_nifty: str = ""
    delta_hedge_exchange_banknifty: str = ""
    delta_hedge_symbol: str = ""  # e.g. "NSE:NIFTYBEES" or "NFO:NIFTY26JANFUT"
    delta_hedge_exchange: str = ""  # optional hint; can be embedded in delta_hedge_symbol
    # Minimum absolute delta (in underlying units) to trigger a hedge rebalance.
    delta_hedge_delta_tolerance: float = 5.0
    # Hysteresis band for delta hedging.
    # If set (>0), entry tolerance is used to trigger hedging and exit tolerance
    # is used to stop hedging when residual delta shrinks.
    delta_hedge_entry_tolerance: float = 0.0
    delta_hedge_exit_tolerance: float = 0.0
    # Adjust only a fraction of residual delta each rebalance (0-1].
    delta_hedge_adjustment_factor: float = 1.0
    # Minimum spot move required between hedge rebalances.
    delta_hedge_min_spot_move: float = 0.0
    delta_hedge_min_spot_move_atr_mult: float = 0.0
    # Maximum absolute change in hedge quantity per rebalance (0 disables).
    delta_hedge_max_adjust_abs_qty: float = 0.0
    # Hard cap on number of hedge orders per day (0 disables).
    delta_hedge_max_orders_per_day: int = 0
    # Liquidity guardrails for hedge instrument.
    delta_hedge_require_bid_ask: bool = False
    delta_hedge_max_spread_pct: float = 0.0
    delta_hedge_max_spread_abs: float = 0.0
    # Optional: include existing broker portfolio positions when hedging.
    # When enabled, the bot will use broker-reported net quantity of the hedge
    # instrument (FUT/ETF) as the effective hedge position to avoid double-hedging
    # and to hedge against pre-existing positions.
    delta_hedge_include_broker_positions: bool = False
    # Optional: also include delivery holdings (equity portfolio) for the hedge symbol.
    # This mainly helps if you hedge with ETFs like NIFTYBEES/BANKBEES.
    delta_hedge_include_holdings: bool = False

    # Optional: beta-based equity portfolio hedge.
    # When enabled, the strategy estimates each holding's beta vs the index and
    # converts your equity holdings into an equivalent index delta (units of index).
    # That delta is added to net option delta before hedging.
    delta_hedge_include_equity_portfolio_beta: bool = False
    delta_hedge_beta_lookback_days: int = 90
    delta_hedge_beta_max_symbols: int = 12
    delta_hedge_beta_refresh_sec: float = 300.0

    # Optional: run a standalone portfolio beta hedge even when there are no option
    # trades with delta-hedging enabled. This is useful when you want to hedge an
    # equity basket/holdings against the index using a single FUT/ETF hedge.
    delta_hedge_portfolio_hedge_enable: bool = False
    # If true, skip hedging outside market hours in live mode.
    delta_hedge_market_hours_only: bool = False
    # Minimum seconds between hedge rebalances.
    delta_hedge_rebalance_interval_sec: float = 15.0
    # Round hedge order quantities to this step (in broker order quantity units).
    # For many NSE derivatives, this is the contract lot size.
    delta_hedge_step_qty: int = 0  # 0 => defaults to cfg.lot_size at runtime
    # Safety clamp to avoid runaway hedging.
    delta_hedge_max_abs_qty: int = 0  # 0 => defaults to 10 * cfg.lot_size at runtime
    # Greeks inputs
    delta_hedge_assumed_iv: float = 0.20
    delta_hedge_rate: float = 0.06

    # ---- Equity holdings trading (optional) ----
    # When enabled, the bot will trade NSE equities from your holdings using
    # simple indicator signals (EMA/RSI/Supertrend) and risk controls.
    # OFF by default.
    equity_trade_enable: bool = False
    # Decision engine for equity trades:
    # - INDICATORS: legacy local-signal engine
    # - GPT: model decides BUY/SELL/HOLD (still subject to risk clamps)
    equity_trade_engine: str = "INDICATORS"
    # Trading horizon. Primarily informational for the GPT engine and for
    # applying long-term target logic.
    equity_trade_horizon: str = "INTRADAY"  # INTRADAY|LONGTERM
    # If enabled, allow the equity engine to actively rotate (churn) positions
    # rather than only entering new ones.
    equity_trade_churn_enable: bool = False
    # If enabled and the equity engine is GPT, the bot asks GPT to build the
    # active equity watchlist from the available universe instead of relying on
    # a fixed user-entered watchlist.
    equity_trade_gpt_watchlist_enable: bool = False
    # Long-term target for a quarter (default 90 days). Used when
    # equity_trade_horizon=LONGTERM.
    equity_longterm_target_pct: float = 0.0
    equity_longterm_horizon_days: int = 90
    equity_trade_allow_short: bool = False
    equity_trade_benchmark: str = "NIFTY"  # Used for beta hedge benchmark
    # Optional watchlist for bot-managed equity entries. If empty, the bot
    # falls back to scanning existing holdings only.

    # GPT behavior: when True, AUTO mode will *require* a GPT recommendation
    # to place trades; if GPT returns no recommendation, AUTO will skip and
    # not fall back to the router heuristic. Default False to prioritize
    # recommendations and fallback to heuristic when needed.
    gpt_require_recommendation: bool = False
    # If True, allow GPT gate TAKE decisions to override some safety checks
    # (price bounds, liquidity) and permit the entry. Use with caution.
    gpt_override_checks: bool = False
    equity_trade_symbols: str = ""
    equity_trade_timeframe: str = "5m"
    equity_trade_max_symbols: int = 10
    equity_trade_rebalance_interval_sec: float = 60.0
    equity_trade_cooldown_sec: float = 300.0
    equity_trade_watchlist_refresh_sec: float = 900.0
    equity_trade_candidate_limit: int = 20
    equity_trade_atr_period: int = 14
    equity_trade_risk_rupees: float = 500.0
    equity_trade_sl_atr_mult: float = 2.0
    equity_trade_tp_atr_mult: float = 3.0
    equity_trade_max_notional: float = 20000.0
    # Total capital bucket reserved for bot-managed equity trades. 0 disables
    # the total-capital clamp and leaves max_notional as the only size cap.
    equity_trade_capital_rupees: float = 0.0
    # If enabled, GPT reviews each open equity position at most once per IST
    # day and may update stop/target levels or request an exit.
    equity_trade_gpt_manage_daily: bool = True
    equity_trade_product_type: str = "INTRADAY"
    equity_trade_longterm_product_type: str = "CNC"
    equity_trade_order_type: str = "MARKET"

    # Multi-leg parameters (points)
    # For strangle/condor: distance from spot to sell strikes.
    short_strike_distance: float = 50.0
    # If ATR (in points) is <= this threshold, prefer short straddle; else short strangle.
    straddle_atr_threshold: float = 40.0
    # For iron condor: additional distance for hedge wings.
    wing_width: float = 50.0
    # Exit safeguard: close after this many minutes (0 disables).
    max_hold_minutes: int = 30
    # Exit safeguard: close after this many days since opening (0 disables).
    # This is evaluated using the local/IST calendar date boundary when possible.
    max_hold_days: int = 0

    # Range-bound filter for premium-selling strategies.
    # If abs(EMA9-EMA21)/close is above this, skip entries.
    max_trend_strength: float = 0.0015

    # Directional internal indicators
    ema_fast: int = 9
    ema_slow: int = 21

    # Directional scalping enhancements
    cooldown_sec: float = 30.0
    # Extra cooldown after a stopout to avoid revenge trades.
    cooldown_after_stopout_sec: float = 180.0
    # Additive cooldown based on ATR (seconds = base + ATR * mult). 0 disables.
    cooldown_atr_mult: float = 0.25
    # IMPORTANT (premium selling): keep trade count small.
    max_trades_per_day: int = 2
    atr_period: int = 14

    min_atr: float = 0.0
    max_atr: float = 1e9

    # Risk controller (broker-schema independent)
    # Premium-selling exits based on underlying movement.
    premium_exit_on_short_strike_touch: bool = True
    premium_exit_on_wing_touch: bool = True

    # Kill-switch style controls (work even without P&L feeds)
    # IMPORTANT: stop trading after the first stop-out by default.
    max_stopouts_per_day: int = 1
    max_consecutive_stopouts: int = 1

    # Guardrail: avoid taking only one trade type repeatedly.
    # 0 disables. If set to N>0, blocks opening the (N+1)th consecutive trade
    # of the same type (type = position_type + strategy/name).
    max_consecutive_same_trade_type: int = 2

    # ---- Premium-selling (short option) professional-grade guards ----
    # MTM-based exits are expressed as fractions of premium collected (credit).
    # Example: stop at -0.30 => lose 30% of entry credit.
    premium_mtm_stop_pct: float = 0.30
    premium_mtm_target_pct: float = 0.18
    # Entry premium guardrails (per-leg and total, in absolute premium points).
    # 0 disables each guard.
    entry_min_option_premium: float = 5.0  # ₹5 minimum — blocks illiquid deep-OTM contracts
    entry_max_option_premium: float = 0.0
    entry_min_total_premium: float = 0.0
    entry_max_total_premium: float = 0.0
    # Liquidity guardrails.
    entry_require_bid_ask: bool = True
    entry_max_bid_ask_spread_pct: float = 0.02   # 2% — block if spread exceeds 2%
    entry_max_bid_ask_spread_abs: float = 5.0    # ₹5 absolute spread — block if wider
    # Minimum option premium filter (₹). 0 disables.
    # Low-premium options have disproportionately high brokerages and wider effective spreads.
    min_option_premium: float = 5.0              # enable by default: block trades < ₹5 premium
    # Dynamic liquidity guard: block entries when current bid/ask spread is
    # much wider than the recent spread history for that exact option.
    # 0 disables. Example 2.5 => block if spread > 2.5x recent median.
    entry_spread_shock_mult: float = 1.0   # 1.0 = normal mode, no shock — safe paper default
    entry_spread_shock_lookback: int = 20
    entry_spread_shock_min_samples: int = 5

    # Trailing MTM exits (HWM-based)
    # Start trailing when MTM >= trail_start_pct * entry_premium.
    # Stop if MTM drops by trail_stop_pct from Peak MTM.
    # Percentages of entry premium for start, of peak MTM for stop.
    premium_mtm_trail_start_pct: float = 0.05
    premium_mtm_trail_stop_pct: float = 0.05

    # Hard time-based exits / entry cutoffs (HH:MM, 24h IST).
    premium_force_exit_hhmm: str = "14:20"  # exit all premium-selling trades
    # If empty, premium entry cutoff is disabled.
    premium_entry_cutoff_hhmm: str = "14:00"  #"  # empty = disabled

    # VWAP filter (percentage points, e.g. 0.15 => 0.15%).
    # 0.15% is extremely tight (~35 pts). Relaxed to 0.75% (~180 pts).
    vwap_max_dev_pct: float = 0.75
    enable_vwap_filter: bool = True

    # RSI bounds for range-bound premium-selling entries.
    # Default widened to avoid suppressing entries too aggressively.
    premium_rsi_low: float = 40.0
    premium_rsi_high: float = 60.0

    # Opening impulse filter (percentage points, e.g. 0.40 => 0.40%).
    opening_max_move_pct: float = 0.40
    opening_filter_minutes: int = 45
    enable_opening_filter: bool = True

    # Trend-strength filter for range-bound premium-selling strategies.
    enable_trend_filter: bool = True

    # RSI filter for premium-selling entries.
    enable_premium_rsi_filter: bool = True

    # Volatility spike kill-switch.
    atr_spike_mult: float = 2.0

    # Profit Enhancements (ADX & Volume)
    enable_adx_filter: bool = True
    adx_period: int = 14
    adx_min_strength: float = 25.0
    enable_volume_filter: bool = True
    volume_sma_period: int = 20

    # ---- Directional (long_call/long_put) profit tuning ----
    # If true, do not open new trades when underlying spot LTP fetch fails.
    # (Prevents entries on stale last-close data during quote timeouts.)
    require_spot_ltp_for_entry: bool = True
    # Once trade moves favorably by this many ATRs, move stop to breakeven and
    # start trailing.
    dir_breakeven_atr_mult: float = 0.4
    # Trail distance from best favorable spot, in ATRs.
    dir_trail_atr_mult: float = 0.8
    # Premium-based trailing stop (percentage).
    # If > 0, exit if option premium drops by this % from its highest point.
    # e.g. 0.10 means exit if premium drops 10% from the peak.
    dir_premium_trail_pct: float = 0.10
    
    # Partial booking settings
    # ATR multiple for first target exit (e.g. 1.0 ATR).
    dir_partial_target_mult: float = 1.0
    # Percentage of quantity to book at partial target (e.g. 0.5 = 50%).
    dir_partial_qty_pct: float = 0.4
    # After partial exit, optionally tighten BE/trail for profit lock.
    dir_post_partial_be_atr_mult: float = 0.3
    dir_post_partial_trail_atr_mult: float = 0.6

    # ---- Directional style selection (minimize switching losses) ----
    # Controls whether directional entries use long options or short options.
    # - "auto": choose based on volatility regime (ATR decreasing => short, else long)
    # - "long_only": always use long_call / long_put
    # - "short_only": always use short_put / short_call
    directional_trade_style: str = "auto"
    # Optional lock (in minutes) to avoid flip-flopping between long/short styles.
    # 0 disables the lock.
    directional_style_lock_minutes: int = 0
    # How many consecutive ATR samples must be decreasing to qualify as "vol decreasing in order".
    dir_vol_decreasing_steps: int = 3
    # Optional: when a LONG directional trade hits its stop/trailing-stop, flip into the
    # equivalent SHORT structure instead of simply stopping out.
    # - long_call -> short_put
    # - long_put  -> short_call
    # Off by default (higher risk; uses option chain to re-enter).
    dir_flip_long_to_short_on_stop: bool = True

    # ---- Directional entry quality (reduce churn) ----
    # Minimum score required to take a directional entry.
    # Higher => fewer trades (less churn), typically higher average quality.
    dir_min_confirmations: int = 4
    # Minimum score advantage required (winner - loser). 0 allows near-ties.
    dir_min_score_diff: int = 1
    # Minimum spot momentum (as ATR multiple) to award the momentum vote.
    # Example: 0.10 => require move >= 0.10 * ATR over the lookback window.
    dir_momentum_atr_mult: float = 0.10
    # If true, allow tie-break entries when bull_score == bear_score.
    # Default false to avoid flip-flopping in noisy regimes.
    dir_allow_tie_break_entries: bool = True
    # EMA slope confirmation (directional). Expressed as ATR multiple.
    # Example: 0.05 => require EMA slope > 0.05 * ATR to add a vote.
    dir_ema_slope_atr_mult: float = 0.05
    # Directional exposure guard (approx, based on open directional legs).
    # 0 disables.
    entry_max_same_direction_qty: int = 0

    # ---- IV regime filter ----
    # Skip short premium entries when IV is expanding fast.
    iv_expand_threshold_pct: float = 3.0
    # Skip long premium entries when IV is collapsing fast.
    iv_contract_threshold_pct: float = 3.0

    # ---- Dynamic risk scaling ----
    risk_scale_enabled: bool = True
    risk_scale_stopout_factor: float = 0.7
    risk_scale_recovery_wins: int = 2
    risk_scale_atr_high: float = 70.0
    risk_scale_atr_high_factor: float = 0.7

    # ---- Enhancement toggles ----
    enable_volatility_forecast: bool = True
    enable_ml_signals: bool = True
    ensemble_artifact_dir: str = "models/all_combined_parquet_rf_xgb_ensemble"
    enable_cvar_sizing: bool = False
    cvar_target: float = 0.05
    enable_exit_optimizer: bool = True
    risk_scale_min_qty: int = 0
    risk_scale_max_qty: int = 0
    risk_scale_step_qty: int = 0

    # Supertrend filter for directional trades
    # If enabled, blocks long entries when price < Supertrend and short entries when price > Supertrend.
    enable_supertrend_filter: bool = True
    # Supertrend mode:
    # - "trend": trade WITH Supertrend direction
    # - "counter": trade AGAINST Supertrend direction
    # (Directional entries will be aligned accordingly when Supertrend is available.)
    supertrend_mode: str = "counter"
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    supertrend_vote_weight: int = 1  # How many votes Supertrend gets in directional scoring

    # Dynamic strike selection for directional trades
    # Adapts strike selection based on ATR (volatility) - allows OTM/ITM options for better risk/reward
    enable_directional_dynamic_strikes: bool = True
    # OTM multipliers (positive = OTM options)
    dir_strike_atr_multiplier_low: float = 0.5    # ATR < threshold_low (closer to ATM)
    dir_strike_atr_multiplier_mid: float = 1.0    # threshold_low <= ATR <= threshold_high (ATM)
    dir_strike_atr_multiplier_high: float = 1.5   # ATR > threshold_high (OTM for better premium)
    dir_strike_atr_threshold_low: float = 30.0
    dir_strike_atr_threshold_high: float = 60.0
    # ITM multipliers (negative = ITM options - cheaper but higher delta)
    dir_strike_itm_multiplier_low: float = -0.5   # ATR < threshold_low (slight ITM)
    dir_strike_itm_multiplier_mid: float = -1.0   # threshold_low <= ATR <= threshold_high (ITM)
    dir_strike_itm_multiplier_high: float = -1.5  # ATR > threshold_high (deep ITM)
    # When to prefer ITM over OTM options (extreme volatility or low volatility conditions)
    dir_use_itm_above_atr: float = 80.0          # Use ITM options when ATR > this value (high vol)
    dir_use_itm_below_atr: float = 20.0          # Use ITM options when ATR < this value (low vol)

    # Options Greeks Selection
    enable_delta_strike_selection: bool = True
    target_delta: float = 0.40
    enable_theta_decay_filter: bool = True

    # Exit Logic enhancements
    dir_sl_atr_mult: float = 1.5
    dir_tp_atr_mult: float = 2.5
    choppiness_hard_filter: bool = True
    enable_rsi_confluence: bool = True

    # Dynamic strike selection for premium strategies
    # Adapts strike distance based on ATR (volatility)
    enable_dynamic_strikes: bool = True
    strike_atr_multiplier_low: float = 0.8    # ATR < threshold_low
    strike_atr_multiplier_mid: float = 1.0    # threshold_low <= ATR <= threshold_high
    strike_atr_multiplier_high: float = 1.5   # ATR > threshold_high
    strike_atr_threshold_low: float = 30.0
    strike_atr_threshold_high: float = 60.0

    # Contract selection
    # If true, only trade NIFTY weekly options (exclude monthly expiry = last Thursday).
    nifty_weekly_only: bool = True
    # Optional: force all option selection to a specific expiry date
    # (e.g. "06-01-2026" or "2026-01-06"). If empty, the nearest
    # weekly expiry is used.
    target_expiry: str = ""

    # Debug helpers
    bypass_weekly_filter: bool = False
    debug_log_no_signal: bool = False

    # Preset (optional): e.g. "debug" for relaxed entry constraints.
    preset: str = ""

    # ---- GPT advisor (optional) ----
    # Enable and configure the GPT-based advisor for entry gating and (optionally)
    # auto-mode strategy selection. Keys are read from env at runtime.
    gpt_enable: bool = False
    gpt_enabled: bool = False
    use_gpt_market_analysis: bool = False
    # When True, GPT is queried for market commentary (strategy suggestions, trade
    # approval). When False (default), GPT calls are skipped entirely and ML trading
    # proceeds without GPT commentary. HTTP 402 errors from GPT MUST NOT block
    # shadow mode, paper-forward, or ML signal generation regardless of this flag.
    gpt_market_commentary_enabled: bool = False
    # "gate" blocks entries unless GPT returns TAKE; "advice" only logs.
    gpt_mode: str = "gate"
    # Apply GPT gating in paper mode as well (README: default enabled).
    gpt_apply_paper: bool = True
    # Request timeout passed to the advisor client.
    gpt_timeout_sec: float = 45.0
    # When strategy_name=auto, allow GPT to override the auto-selected branch.
    gpt_auto_select: bool = True

    # ---- GPT dynamic position management (optional) ----
    # If enabled, GPT may return per-leg stop/target premium levels to manage
    # open option legs. This is applied only when GPT is enabled.
    gpt_leg_manage: bool = False
    # Minimum seconds between GPT refreshes for leg exits per trade.
    gpt_leg_manage_refresh_sec: float = 45.0
    # Minimum age of an open option trade before GPT may manage its legs.
    # This avoids immediate GPT exits before the UI has shown leg LTP/MTM.
    gpt_leg_manage_min_age_sec: float = 90.0
    # After GPT updates leg stop/target levels, wait this many seconds before
    # enforcing those exits. This avoids immediate close on stale/tick-latency.
    gpt_leg_manage_post_update_grace_sec: float = 20.0
    # Minimum distance from current option LTP when accepting GPT stop/target
    # levels. Example 0.08 => at least 8% away from current LTP.
    gpt_leg_manage_min_stop_distance_pct: float = 0.08
    gpt_leg_manage_min_target_distance_pct: float = 0.10

    # ---- P1: GPT-Powered Dynamic Exit Management ----
    # Periodically reviews open positions + market snapshot and recommends
    # HOLD, TRAIL_TIGHTER, EXIT_NOW, or PARTIAL_EXIT with quantity.
    gpt_exit_management: bool = False
    gpt_exit_management_refresh_sec: float = 60.0

    # ---- P2: Multi-Call Strategy Voting ----
    # Makes multiple parallel GPT calls with slightly different temperature
    # and picks the strategy with highest consensus.
    gpt_strategy_voting_enabled: bool = False
    gpt_strategy_voting_calls: int = 3
    gpt_strategy_voting_temperature: float = 0.3

    # ---- P3: GPT-Based Position Sizing & Risk Adjustment ----
    # GPT scales position size based on IV percentile, win/loss streak,
    # strategy type, and correlation between concurrent open trades.
    gpt_position_sizing_enabled: bool = False

    # ---- P4: Regime Change Detection / Proactive Risk Actions ----
    # GPT monitors market regime and may tighten stops, reduce sizing,
    # pause entries, or switch strategy families based on IV/trend shifts.
    gpt_regime_monitor_enabled: bool = False
    gpt_regime_monitor_interval_sec: float = 300.0

    # ---- P5: Enhanced Strike Selection with Full Greeks Context ----
    # Passes live IV surface / greeks data so GPT can suggest wing widths,
    # strike distances, and multi-leg structure optimizations.
    gpt_strike_selection_enabled: bool = False

    # ---- P6: What-If Scenario Analysis ----
    # Periodic GPT call: "If NIFTY moves ±1%/±2%, which open positions
    # are most at risk? Should I hedge or exit any?"
    gpt_whatif_enabled: bool = False
    gpt_whatif_interval_sec: float = 120.0

    # ---- Auto-mode tuning ----
    # Expand the range-bound detector beyond max_trend_strength by a multiplier.
    # Example: 2.0 means allow up to 2x the base trend strength before rejecting range-bound.
    auto_range_trend_mult: float = 2.0

    # Reduce churn in auto-mode by locking the selected strategy for N minutes.
    # 0 disables.
    auto_strategy_lock_minutes: int = 0

    # When auto selects directional, use a simple EMA+RSI trigger so the bot actually trades.
    # (Values are conservative defaults; adjust via env.)
    auto_dir_rsi_buy: float = 52.0
    auto_dir_rsi_sell: float = 48.0
    # RSI tolerance for auto-directional triggers. Example: slop=1.0 means
    # allow call when RSI >= (buy-1.0) and put when RSI <= (sell+1.0).
    auto_dir_rsi_slop: float = 1.0

    # ---- Signal Quality Enhancement Settings ----
    enable_mtf_confirmation: bool = True
    mtf_timeframe: str = "5m"
    enable_roc_filter: bool = True
    roc_period: int = 14
    roc_min_threshold: float = 0.05
    enable_choppiness_filter: bool = True
    choppiness_period: int = 14
    choppiness_threshold: float = 61.8

    # ---- Risk Management Enhancement Settings ----
    stagnation_exit_minutes: int = 15  # Exit trades that stall for 15+ min
    stagnation_pnl_threshold: float = 100.0
    enable_chandelier_exit: bool = True
    chandelier_period: int = 22
    chandelier_multiplier: float = 3.0
    enable_pivot_targets: bool = True

    # ---- Execution Enhancement Settings ----
    enable_limit_orders: bool = True
    limit_price_buffer_pct: float = 0.05
    max_pyramid_levels: int = 1  # Add to winners once

    # ---- Paper Execution Cost Model ----
    # Deduct realistic execution costs from paper P&L to prevent optimistic bias.
    # No double-counting: if execution_price uses ask/bid, spread is already embedded
    # and only the "extra" slippage / brokerage / taxes are deducted separately.

    # Slippage cost applied ON TOP of the embedded ask/bid execution price.
    # Represented as a fraction of the execution price per leg.
    # Default 0.001 = 0.1% slippage per leg (one-way).
    paper_slippage_pct: float = 0.001
    # Additional market-impact cost (fraction of execution price). 0 disables.
    paper_extra_market_impact_pct: float = 0.0
    # If True, deduct brokerage + taxes (STT/GST/exchange/sebi/stamp) per leg.
    paper_apply_brokerage_costs: bool = True
    # Source for brokerage/tax estimates:
    # - "cost_model_assumptions"  : uses CostModelAssumptions (per-side %)
    # - "ml_execution_costs"      : uses ml_execution_costs DEFAULT_EXECUTION_COST_CONFIG
    paper_cost_model_source: str = "cost_model_assumptions"

    # ---- Dynamic Pyramiding ----
    # When enabled, max_pyramid_levels is computed at runtime based on
    # current market conditions (IV percentile, winrate, ATR regime, session time).
    # Static max_pyramid_levels is used as the floor/upper bound.
    enable_dynamic_pyramiding: bool = False
    # Max pyramid levels when IV is in a favorable (low) percentile.
    pyramid_iv_max_levels: int = 0  # 0 = use static max_pyramid_levels
    # Max pyramid levels when ATR regime is moderate.
    pyramid_atr_max_levels: int = 0
    # Minimum winrate (%) to allow extra pyramid levels.
    pyramid_winrate_min_for_extra: float = 50.0
    # Fractional multiplier applied during opening/closing sessions (e.g. 0.5 = half).
    pyramid_session_tighten_factor: float = 0.5
    # Absolute ceiling on dynamic pyramid levels.
    pyramid_max_total_levels: int = 5
    order_retry_attempts: int = 2
    order_retry_delay_sec: float = 0.25

    # ---- Paper Execution Realism ----
    # When True (default), paper trades use real bid/ask prices:
    #   - BUY entry: ask price (you pay what the seller asks)
    #   - SELL exit (close BUY): bid price (you receive what buyer bids)
    #   - SELL entry (short): bid price (you receive what buyer bids)
    #   - BUY exit (close SELL): ask price (you pay what seller asks)
    # When False, paper trades use LTP (last traded price) which is unrealistic.
    paper_use_bid_ask_execution: bool = True
    # When True and bid/ask is unavailable, fallback to LTP with spread penalty.
    # When False (default), block the trade if bid/ask is missing.
    paper_allow_ltp_fallback: bool = False
    # Spread penalty (percentage) applied to LTP when fallback is triggered.
    # Example: 0.05 = 5% worse than LTP to model realistic execution cost.
    paper_ltp_fallback_spread_pct: float = 0.05

    # ---- Strategy Router / Diagnostics ----
    strategy_router_mode: str = "balanced"  # conservative | balanced | aggressive
    enable_mean_reversion_engine: bool = True
    mean_reversion_lookback: int = 20
    mean_reversion_entry_zscore: float = 1.25
    mean_reversion_exit_zscore: float = 0.35
    enable_stat_arb_module: bool = True
    stat_arb_lookback: int = 30
    stat_arb_entry_zscore: float = 1.5
    stat_arb_exit_zscore: float = 0.5
    sleeve_reserve_cash_weight: float = 0.10
    sleeve_max_single_weight: float = 0.50
    diagnostics_enabled: bool = True

    # ---- Regime-specific tuning ----
    # Used by auto-mode and the ML gate.
    regime_trending_trend_mult: float = 1.20
    regime_trending_ml_threshold: float = 0.58
    regime_trending_preferred_strategy: str = "bull_call_spread"

    regime_volatile_trend_mult: float = 0.90
    regime_volatile_ml_threshold: float = 0.62
    regime_volatile_preferred_strategy: str = "long_straddle"

    regime_mean_reverting_trend_mult: float = 0.75
    regime_mean_reverting_ml_threshold: float = 0.55
    regime_mean_reverting_preferred_strategy: str = "iron_condor"

    regime_quiet_trend_mult: float = 0.65
    regime_quiet_ml_threshold: float = 0.57
    regime_quiet_preferred_strategy: str = "short_strangle"

    # ---- Portfolio-level Risk Caps ----
    # 0 disables each cap.
    max_portfolio_option_notional: float = 500000.0
    max_portfolio_delta_abs: float = 500.0

    # ---- IV Percentile Position Sizing ----
    iv_sizing_enabled: bool = False
    iv_sizing_lookback_days: int = 20
    iv_sizing_high_percentile: float = 75.0
    iv_sizing_low_percentile: float = 25.0
    iv_sizing_high_factor: float = 0.5
    iv_sizing_max_factor: float = 1.0

    # ---- Session-Specific Exit Adjustments ----
    session_exit_enabled: bool = False
    session_opening_hhmm: str = "10:00"
    session_lunch_start_hhmm: str = "12:00"
    session_lunch_end_hhmm: str = "13:30"
    session_close_hhmm: str = "14:45"
    session_opening_sl_mult: float = 0.7
    session_opening_tp_mult: float = 0.8
    session_lunch_sl_mult: float = 1.3
    session_lunch_tp_mult: float = 1.2
    session_close_sl_mult: float = 0.6
    session_close_tp_mult: float = 0.7


    # ---- #1: IV Rank/Percentile Entry Filter ----
    # When enabled, biases strategy selection based on IV percentile.
    # High IV percentile (> threshold) favors premium-selling strategies.
    # Low IV percentile (< threshold) favors premium-buying strategies.
    iv_filter_enabled: bool = False
    iv_filter_high_percentile: float = 70.0
    iv_filter_low_percentile: float = 30.0

    # ---- #2: Multi-Timeframe Trend Filter (hard filter) ----
    # When enabled, blocks directional entries that conflict with the MTF trend.
    # When disabled (default), MTF only adds/subtracts score votes.
    mtf_hard_filter: bool = False

    # ---- #6: Time-Decay Adjusted Stop Distance ----
    # Widen stops for options as theta accelerates near expiry.
    theta_stop_widen_enabled: bool = False
    # Maximum multiplier applied to stop distance (e.g. 2.0 = at most 2x normal stop).
    theta_stop_widen_max_mult: float = 2.0
    # Theta (daily) below this threshold triggers widening (more negative = faster decay).
    # A typical weekly ATM option might have theta of -5 to -20 near expiry.
    theta_stop_widen_threshold: float = -5.0

    # ---- #8: Smart Partial Exit Ladder ----
    # Multi-level partial exits instead of a single partial target.
    # Format: comma-separated "qty_pct:atr_mult" pairs.
    # Example: "0.25:1.0,0.25:1.5,0.25:2.0" closes 25% at 1.0 ATR, 25% at 1.5 ATR,
    # 25% at 2.0 ATR, leaving 25% for the final target.
    exit_ladder_enabled: bool = False
    exit_ladder_steps: str = "0.25:1.0,0.25:1.5,0.25:2.0"

    # ---- #9: Volatility-Weighted Delta Hedging Threshold ----
    # Scale delta hedge tolerance by IV percentile.
    # Low IV => tighter tolerance (hedge more aggressively).
    # High IV => wider tolerance (avoid over-hedging during vol spikes).
    delta_hedge_vol_adjust_enabled: bool = False
    delta_hedge_vol_low_tol_factor: float = 0.7
    delta_hedge_vol_high_tol_factor: float = 1.5

    # ---- Strategy Win-Rate Tracker ----
    winrate_tracker_enabled: bool = True
    winrate_lookback: int = 10
    winrate_min_wins: int = 3
    winrate_max_loss_streak: int = 3
    winrate_min_win_rate: float = 30.0
    winrate_cooldown_sec: int = 300


def _default_credential_path() -> Path:
    """Location where the UI optionally saves credentials.

    Kept outside the repo by default.
    """

    base = os.getenv("APPDATA")
    if base:
        return Path(base) / "scalper" / "credentials.json"
    return Path.home() / ".scalper" / "credentials.json"


def load_saved_credentials() -> Dict[str, Any]:
    """Load locally saved credentials (if present).

    This is an optional convenience for CLI/headless runs to reuse the same
    file the UI writes. The returned mapping may include: api_key, username,
    password, etc.
    """

    p = _default_credential_path()
    try:
        if not p.exists():
            return {}
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _sdk_default_api_key() -> str:
    cfg = _load_local_typeb_config()
    if cfg is not None:
        key = str(getattr(cfg, "API_KEY", "") or "").strip()
        if key and "ENTER_YOUR_API_KEY_HERE" not in key:
            return key

    try:
        from tradingapi_b import __config__

        key = str(getattr(__config__, "API_KEY", "") or "").strip()
        if key and "ENTER_YOUR_API_KEY_HERE" not in key:
            return key
    except Exception:
        pass
    return ""


def _sdk_default_base_url() -> str:
    cfg = _load_local_typeb_config()
    if cfg is not None:
        url = str(getattr(cfg, "default_root_uri", "") or "").strip()
        if url:
            return url

    try:
        from tradingapi_b import __config__

        url = str(getattr(__config__, "default_root_uri", "") or "").strip()
        return url
    except Exception:
        return ""


def _repo_root() -> Path:
    # src/config.py -> repo_root/src/config.py
    return Path(__file__).resolve().parent.parent


_PERSISTED_ENV_LOADED = False


def _settings_env_path() -> Path:
    return _repo_root() / ".scalper.env"


def _dotenv_env_path() -> Path:
    return _repo_root() / ".env"


def _decode_env_value(raw: str) -> str:
    v = (raw or "").strip()
    if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
        v = v[1:-1]
        v = v.replace("\\n", "\n").replace("\\\\", "\\").replace('\\"', '"').replace("\\'", "'")
    return v


def _encode_env_value(value: str) -> str:
    v = str(value if value is not None else "")
    v = v.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
    return f'"{v}"'


def load_persisted_env(*, override_existing: bool = False, path: Optional[Path] = None) -> Dict[str, str]:
    """Load key=value pairs from repo-root .scalper.env into os.environ.

    By default, does NOT override values already present in os.environ.
    """

    p = path or _settings_env_path()
    try:
        if not p.exists():
            return {}
        raw = p.read_text(encoding="utf-8")
    except Exception:
        return {}

    loaded: Dict[str, str] = {}
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().startswith("export "):
            s = s[7:].strip()
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = k.strip()
        if not key:
            continue
        val = _decode_env_value(v)
        loaded[key] = val
        if (not override_existing) and (key in os.environ):
            continue
        os.environ[key] = val
    return loaded


def ensure_persisted_env_loaded() -> None:
    global _PERSISTED_ENV_LOADED
    if _PERSISTED_ENV_LOADED:
        return
    # Load repo-local .env without requiring python-dotenv. This carries secrets
    # such as MSTOCK_ACCESS_TOKEN for CLI/headless runs and VS Code terminals.
    load_persisted_env(override_existing=False, path=_dotenv_env_path())
    # UI-persisted settings should win over .env when both define a setting.
    load_persisted_env(override_existing=True, path=_settings_env_path())
    _PERSISTED_ENV_LOADED = True


def persist_settings_env(
    set_values: Dict[str, str],
    unset_keys: Optional[list[str]] = None,
    *,
    path: Optional[Path] = None,
) -> Path:
    """Persist settings to repo-root .scalper.env.

    Intended for non-secret MSTOCK_* settings edited via the UI.
    """

    p = path or _settings_env_path()
    existing = load_persisted_env(override_existing=False, path=p)

    merged: Dict[str, str] = dict(existing)
    for k, v in (set_values or {}).items():
        if not k:
            continue
        merged[str(k)] = str(v)
    for k in (unset_keys or []):
        merged.pop(str(k), None)

    try:
        lines = ["# Auto-generated by Scalper UI Settings", "# Format: KEY=VALUE (dotenv)"]
        for key in sorted(merged.keys()):
            lines.append(f"{key}={_encode_env_value(merged[key])}")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        # Best-effort persistence; don't crash the bot if disk write fails.
        return p
    return p


def _local_typeb_config_path() -> Path:
    return _repo_root() / "pytradingapi-typeB-main" / "tradingapi_b" / "__config__.py"


def _load_local_typeb_config():
    """Load the vendored Type-B SDK config module, if present.

    This avoids requiring the SDK to be installed to read its defaults.
    """

    p = _local_typeb_config_path()
    try:
        if not p.exists():
            return None
        spec = importlib.util.spec_from_file_location("_local_tradingapi_b_config", str(p))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


def get_sdk_default_api_key() -> str:
    """Public helper: best-effort SDK API key default (no env)."""

    return _sdk_default_api_key()


def get_sdk_default_base_url() -> str:
    """Public helper: best-effort SDK base URL default (no env)."""

    return _sdk_default_base_url()


def load_api_config() -> APIConfig:
    """Load API config from environment variables.

    Replace placeholders based on your actual m.Stock Type B naming.
    """
    ensure_persisted_env_loaded()
    saved = load_saved_credentials()

    base_url = os.getenv("MSTOCK_BASE_URL", "").strip() or _sdk_default_base_url()
    api_key = os.getenv("MSTOCK_API_KEY", "").strip() or str(saved.get("api_key") or "").strip() or _sdk_default_api_key()
    api_secret = os.getenv("MSTOCK_API_SECRET", "").strip() or str(saved.get("api_secret") or "").strip()
    client_id = os.getenv("MSTOCK_CLIENT_ID", "").strip() or str(saved.get("client_id") or "").strip()
    vendor_code = (os.getenv("MSTOCK_VENDOR_CODE") or str(saved.get("vendor_code") or "")).strip() or None

    return APIConfig(
        base_url=base_url,
        api_key=api_key,
        api_secret=api_secret,
        client_id=client_id,
        vendor_code=vendor_code,
    )


def _present_secret(value: str) -> str:
    v = str(value or "").strip()
    if not v:
        return "missing"
    if "ENTER_YOUR_API_KEY_HERE" in v:
        return "placeholder"
    return f"set (len={len(v)})"


def print_config_diagnostics() -> None:
    """Print a safe, non-secret config report."""

    saved = load_saved_credentials()
    cfg = load_api_config()

    env_api_key = os.getenv("MSTOCK_API_KEY", "")
    env_access_token = os.getenv("MSTOCK_ACCESS_TOKEN", "")
    env_user = os.getenv("MSTOCK_USERNAME", "")
    env_pass = os.getenv("MSTOCK_PASSWORD", "")

    sdk_key = _sdk_default_api_key()
    cred_path = _default_credential_path()

    print("m.Stock Type B credential check (no secrets printed):\n")
    print(f"- MSTOCK_API_KEY env:      {_present_secret(env_api_key)}")
    print(f"- Saved credentials file:  {'present' if bool(saved) else 'missing'} ({cred_path})")
    if saved:
        print(f"  - saved.api_key:         {_present_secret(str(saved.get('api_key') or ''))}")
        print(f"  - saved.username:        {_present_secret(str(saved.get('username') or ''))}")
        print(f"  - saved.password:        {_present_secret(str(saved.get('password') or ''))}")
    print(f"- tradingapi_b API_KEY:    {_present_secret(sdk_key)}")
    print(f"- Effective api_key used:  {_present_secret(cfg.api_key)}")
    print(f"- MSTOCK_ACCESS_TOKEN env: {_present_secret(env_access_token)}")
    print(f"- MSTOCK_USERNAME env:     {_present_secret(env_user)}")
    print(f"- MSTOCK_PASSWORD env:     {_present_secret(env_pass)}")
    print("\nNotes:")
    print("- Bot runtime needs MSTOCK_API_KEY + MSTOCK_ACCESS_TOKEN.")
    print("- If access token is missing, run: python -m src.auth (OTP flow) or use the UI.")


def main() -> None:
    print_config_diagnostics()


if __name__ == "__main__":
    main()


def load_strategy_config() -> StrategyConfig:
    ensure_persisted_env_loaded()
    cfg = StrategyConfig()

    def _normalize_timeframe(value: object, default: str = "1m") -> str:
        raw = str(value or "").strip().lower()
        aliases = {
            "1m": "1m",
            "m1": "1m",
            "one_minute": "1m",
            "3m": "3m",
            "m3": "3m",
            "three_minute": "3m",
            "5m": "5m",
            "m5": "5m",
            "five_minute": "5m",
            "10m": "10m",
            "m10": "10m",
            "ten_minute": "10m",
            "15m": "15m",
            "m15": "15m",
            "fifteen_minute": "15m",
            "30m": "30m",
            "m30": "30m",
            "thirty_minute": "30m",
            "1h": "1h",
            "60m": "1h",
            "one_hour": "1h",
            "1d": "1d",
            "d1": "1d",
            "one_day": "1d",
        }
        return aliases.get(raw, str(default or "1m"))

    cfg.preset = os.getenv("MSTOCK_PRESET", cfg.preset).strip().lower()

    cfg.symbol = os.getenv("MSTOCK_SYMBOL", cfg.symbol)
    cfg.underlying = os.getenv("MSTOCK_UNDERLYING", cfg.underlying)
    cfg.underlying_token = os.getenv("MSTOCK_UNDERLYING_TOKEN", cfg.underlying_token)
    cfg.underlying_exchange = os.getenv("MSTOCK_UNDERLYING_EXCHANGE", cfg.underlying_exchange)

    cfg.target_expiry = os.getenv("MSTOCK_TARGET_EXPIRY", cfg.target_expiry)
    cfg.timeframe = _normalize_timeframe(os.getenv("MSTOCK_TIMEFRAME", cfg.timeframe), str(cfg.timeframe or "1m"))

    try:
        cfg.max_open_positions = int(os.getenv("MSTOCK_MAX_OPEN_POSITIONS", str(cfg.max_open_positions)))
    except Exception:
        pass

    try:
        cfg.max_daily_loss = float(os.getenv("MSTOCK_MAX_DAILY_LOSS", str(cfg.max_daily_loss)))
    except Exception:
        pass

    try:
        cfg.max_daily_profit = float(os.getenv("MSTOCK_MAX_DAILY_PROFIT", str(cfg.max_daily_profit)))
    except Exception:
        pass

    cfg.telegram_bot_token = os.getenv("MSTOCK_TELEGRAM_BOT_TOKEN", cfg.telegram_bot_token)
    cfg.telegram_chat_id = os.getenv("MSTOCK_TELEGRAM_CHAT_ID", cfg.telegram_chat_id)

    try:
        cfg.account_capital = float(os.getenv("MSTOCK_ACCOUNT_CAPITAL", str(cfg.account_capital)))
    except Exception:
        pass

    try:
        cfg.risk_per_trade_percentage = float(os.getenv("MSTOCK_RISK_PER_TRADE_PERCENTAGE", str(cfg.risk_per_trade_percentage)))
    except Exception:
        pass

    try:
        cfg.lot_size = int(os.getenv("MSTOCK_LOT_SIZE", str(cfg.lot_size)))
    except Exception:
        pass

    cfg.strategy_name = os.getenv("MSTOCK_STRATEGY", cfg.strategy_name).strip().lower()

    # ---- GPT advisor (optional) ----
    try:
        cfg.gpt_enable = str(os.getenv("MSTOCK_GPT_ENABLE", "") or "").strip().lower() in {"1", "true", "yes", "y"}
    except Exception:
        pass
    try:
        cfg.gpt_enabled = str(
            os.getenv("GPT_ENABLED", os.getenv("MSTOCK_GPT_ENABLED", str(cfg.gpt_enable))) or str(cfg.gpt_enable)
        ).strip().lower() in {"1", "true", "yes", "y"}
    except Exception:
        cfg.gpt_enabled = bool(cfg.gpt_enable)
    try:
        cfg.use_gpt_market_analysis = str(
            os.getenv(
                "USE_GPT_MARKET_ANALYSIS",
                os.getenv("MSTOCK_USE_GPT_MARKET_ANALYSIS", str(cfg.gpt_enabled)),
            ) or str(cfg.gpt_enabled)
        ).strip().lower() in {"1", "true", "yes", "y"}
    except Exception:
        cfg.use_gpt_market_analysis = bool(cfg.gpt_enabled)
    cfg.gpt_enable = bool(cfg.gpt_enabled)
    try:
        cfg.gpt_mode = str(os.getenv("MSTOCK_GPT_MODE", cfg.gpt_mode) or cfg.gpt_mode).strip().lower()
    except Exception:
        pass
    if cfg.gpt_mode not in {"gate", "advice"}:
        cfg.gpt_mode = "gate"
    v_apply = os.getenv("MSTOCK_GPT_APPLY_PAPER")
    if v_apply is not None:
        try:
            cfg.gpt_apply_paper = str(v_apply).strip().lower() in {"1", "true", "yes", "y"}
        except Exception:
            pass
    try:
        cfg.gpt_timeout_sec = float(os.getenv("MSTOCK_GPT_TIMEOUT_SEC", str(cfg.gpt_timeout_sec)))
    except Exception:
        pass
    try:
        cfg.gpt_market_commentary_enabled = str(
            os.getenv(
                "GPT_MARKET_COMMENTARY_ENABLED",
                os.getenv("MSTOCK_GPT_MARKET_COMMENTARY_ENABLED", "false"),
            )
            or "false"
        ).strip().lower() in {"1", "true", "yes", "y"}
    except Exception:
        cfg.gpt_market_commentary_enabled = False
    v_auto = os.getenv("MSTOCK_GPT_AUTO_SELECT")
    if v_auto is not None:
        try:
            cfg.gpt_auto_select = str(v_auto).strip().lower() in {"1", "true", "yes", "y"}
        except Exception:
            pass

    v_req_rec = os.getenv("MSTOCK_GPT_REQUIRE_RECOMMENDATION")
    if v_req_rec is not None:
        try:
            cfg.gpt_require_recommendation = str(v_req_rec).strip().lower() in {"1", "true", "yes", "y"}
        except Exception:
            pass

    v_leg = os.getenv("MSTOCK_GPT_LEG_MANAGE")
    if v_leg is not None:
        try:
            cfg.gpt_leg_manage = str(v_leg).strip().lower() in {"1", "true", "yes", "y"}
        except Exception:
            pass

    # P1: GPT Exit Management
    v_gpt_exit = os.getenv("MSTOCK_GPT_EXIT_MANAGEMENT")
    if v_gpt_exit is not None:
        try:
            cfg.gpt_exit_management = str(v_gpt_exit).strip().lower() in {"1", "true", "yes", "y"}
        except Exception:
            pass
    try:
        cfg.gpt_exit_management_refresh_sec = float(
            os.getenv("MSTOCK_GPT_EXIT_MANAGEMENT_REFRESH_SEC", str(cfg.gpt_exit_management_refresh_sec))
        )
    except Exception:
        pass

    # P2: Multi-Call Strategy Voting
    v_gpt_vote = os.getenv("MSTOCK_GPT_STRATEGY_VOTING")
    if v_gpt_vote is not None:
        try:
            cfg.gpt_strategy_voting_enabled = str(v_gpt_vote).strip().lower() in {"1", "true", "yes", "y"}
        except Exception:
            pass
    try:
        cfg.gpt_strategy_voting_calls = int(
            float(os.getenv("MSTOCK_GPT_STRATEGY_VOTING_CALLS", str(cfg.gpt_strategy_voting_calls)))
        )
    except Exception:
        pass
    try:
        cfg.gpt_strategy_voting_temperature = float(
            os.getenv("MSTOCK_GPT_STRATEGY_VOTING_TEMPERATURE", str(cfg.gpt_strategy_voting_temperature))
        )
    except Exception:
        pass

    # P3: GPT Position Sizing
    v_gpt_sizing = os.getenv("MSTOCK_GPT_POSITION_SIZING")
    if v_gpt_sizing is not None:
        try:
            cfg.gpt_position_sizing_enabled = str(v_gpt_sizing).strip().lower() in {"1", "true", "yes", "y"}
        except Exception:
            pass

    # P4: Regime Monitor
    v_gpt_regime = os.getenv("MSTOCK_GPT_REGIME_MONITOR")
    if v_gpt_regime is not None:
        try:
            cfg.gpt_regime_monitor_enabled = str(v_gpt_regime).strip().lower() in {"1", "true", "yes", "y"}
        except Exception:
            pass
    try:
        cfg.gpt_regime_monitor_interval_sec = float(
            os.getenv("MSTOCK_GPT_REGIME_MONITOR_INTERVAL_SEC", str(cfg.gpt_regime_monitor_interval_sec))
        )
    except Exception:
        pass

    # P5: Strike Selection
    v_gpt_strike = os.getenv("MSTOCK_GPT_STRIKE_SELECTION")
    if v_gpt_strike is not None:
        try:
            cfg.gpt_strike_selection_enabled = str(v_gpt_strike).strip().lower() in {"1", "true", "yes", "y"}
        except Exception:
            pass

    # P6: What-If Analysis
    v_gpt_whatif = os.getenv("MSTOCK_GPT_WHATIF")
    if v_gpt_whatif is not None:
        try:
            cfg.gpt_whatif_enabled = str(v_gpt_whatif).strip().lower() in {"1", "true", "yes", "y"}
        except Exception:
            pass
    try:
        cfg.gpt_whatif_interval_sec = float(
            os.getenv("MSTOCK_GPT_WHATIF_INTERVAL_SEC", str(cfg.gpt_whatif_interval_sec))
        )
    except Exception:
        pass
    try:
        cfg.gpt_leg_manage_refresh_sec = float(
            os.getenv("MSTOCK_GPT_LEG_MANAGE_REFRESH_SEC", str(cfg.gpt_leg_manage_refresh_sec))
        )
    except Exception:
        pass
    try:
        cfg.gpt_leg_manage_min_age_sec = float(
            os.getenv("MSTOCK_GPT_LEG_MANAGE_MIN_AGE_SEC", str(cfg.gpt_leg_manage_min_age_sec))
        )
    except Exception:
        pass
    try:
        cfg.gpt_leg_manage_post_update_grace_sec = float(
            os.getenv(
                "MSTOCK_GPT_LEG_MANAGE_POST_UPDATE_GRACE_SEC",
                str(cfg.gpt_leg_manage_post_update_grace_sec),
            )
        )
    except Exception:
        pass
    try:
        cfg.gpt_leg_manage_min_stop_distance_pct = float(
            os.getenv(
                "MSTOCK_GPT_LEG_MANAGE_MIN_STOP_DISTANCE_PCT",
                str(cfg.gpt_leg_manage_min_stop_distance_pct),
            )
        )
    except Exception:
        pass
    try:
        cfg.gpt_leg_manage_min_target_distance_pct = float(
            os.getenv(
                "MSTOCK_GPT_LEG_MANAGE_MIN_TARGET_DISTANCE_PCT",
                str(cfg.gpt_leg_manage_min_target_distance_pct),
            )
        )
    except Exception:
        pass

    # Delta hedging (dynamic)
    cfg.delta_hedge_scope = os.getenv("MSTOCK_DELTA_HEDGE_SCOPE", cfg.delta_hedge_scope).strip().lower()
    cfg.delta_hedge_symbol_nifty = os.getenv("MSTOCK_DELTA_HEDGE_SYMBOL_NIFTY", cfg.delta_hedge_symbol_nifty).strip()
    cfg.delta_hedge_symbol_banknifty = os.getenv(
        "MSTOCK_DELTA_HEDGE_SYMBOL_BANKNIFTY",
        cfg.delta_hedge_symbol_banknifty,
    ).strip()
    cfg.delta_hedge_exchange_nifty = os.getenv(
        "MSTOCK_DELTA_HEDGE_EXCHANGE_NIFTY",
        cfg.delta_hedge_exchange_nifty,
    ).strip()
    cfg.delta_hedge_exchange_banknifty = os.getenv(
        "MSTOCK_DELTA_HEDGE_EXCHANGE_BANKNIFTY",
        cfg.delta_hedge_exchange_banknifty,
    ).strip()
    cfg.delta_hedge_symbol = os.getenv("MSTOCK_DELTA_HEDGE_SYMBOL", cfg.delta_hedge_symbol).strip()
    cfg.delta_hedge_exchange = os.getenv("MSTOCK_DELTA_HEDGE_EXCHANGE", cfg.delta_hedge_exchange).strip()
    try:
        cfg.delta_hedge_delta_tolerance = float(
            os.getenv("MSTOCK_DELTA_HEDGE_TOLERANCE", str(cfg.delta_hedge_delta_tolerance))
        )
    except Exception:
        pass
    try:
        cfg.delta_hedge_entry_tolerance = float(
            os.getenv("MSTOCK_DELTA_HEDGE_ENTRY_TOLERANCE", str(cfg.delta_hedge_entry_tolerance))
        )
    except Exception:
        pass
    try:
        cfg.delta_hedge_exit_tolerance = float(
            os.getenv("MSTOCK_DELTA_HEDGE_EXIT_TOLERANCE", str(cfg.delta_hedge_exit_tolerance))
        )
    except Exception:
        pass
    try:
        cfg.delta_hedge_adjustment_factor = float(
            os.getenv("MSTOCK_DELTA_HEDGE_ADJUSTMENT_FACTOR", str(cfg.delta_hedge_adjustment_factor))
        )
    except Exception:
        pass
    try:
        cfg.delta_hedge_min_spot_move = float(
            os.getenv("MSTOCK_DELTA_HEDGE_MIN_SPOT_MOVE", str(cfg.delta_hedge_min_spot_move))
        )
    except Exception:
        pass
    try:
        cfg.delta_hedge_min_spot_move_atr_mult = float(
            os.getenv(
                "MSTOCK_DELTA_HEDGE_MIN_SPOT_MOVE_ATR_MULT",
                str(cfg.delta_hedge_min_spot_move_atr_mult),
            )
        )
    except Exception:
        pass
    try:
        cfg.delta_hedge_max_adjust_abs_qty = float(
            os.getenv("MSTOCK_DELTA_HEDGE_MAX_ADJUST_ABS_QTY", str(cfg.delta_hedge_max_adjust_abs_qty))
        )
    except Exception:
        pass
    try:
        cfg.delta_hedge_max_orders_per_day = int(
            os.getenv("MSTOCK_DELTA_HEDGE_MAX_ORDERS_PER_DAY", str(cfg.delta_hedge_max_orders_per_day))
        )
    except Exception:
        pass
    dh_require = os.getenv("MSTOCK_DELTA_HEDGE_REQUIRE_BID_ASK")
    if dh_require is not None:
        cfg.delta_hedge_require_bid_ask = str(dh_require).strip().lower() in {"1", "true", "yes", "y"}
    try:
        cfg.delta_hedge_max_spread_pct = float(
            os.getenv("MSTOCK_DELTA_HEDGE_MAX_SPREAD_PCT", str(cfg.delta_hedge_max_spread_pct))
        )
    except Exception:
        pass
    try:
        cfg.delta_hedge_max_spread_abs = float(
            os.getenv("MSTOCK_DELTA_HEDGE_MAX_SPREAD_ABS", str(cfg.delta_hedge_max_spread_abs))
        )
    except Exception:
        pass
    dh_hours = os.getenv("MSTOCK_DELTA_HEDGE_MARKET_HOURS_ONLY")
    if dh_hours is not None:
        cfg.delta_hedge_market_hours_only = str(dh_hours).strip().lower() in {"1", "true", "yes", "y"}

    dh_inc_pos = os.getenv("MSTOCK_DELTA_HEDGE_INCLUDE_POSITIONS")
    if dh_inc_pos is not None:
        cfg.delta_hedge_include_broker_positions = str(dh_inc_pos).strip().lower() in {"1", "true", "yes", "y"}
    dh_inc_hold = os.getenv("MSTOCK_DELTA_HEDGE_INCLUDE_HOLDINGS")
    if dh_inc_hold is not None:
        cfg.delta_hedge_include_holdings = str(dh_inc_hold).strip().lower() in {"1", "true", "yes", "y"}

    dh_beta = os.getenv("MSTOCK_DELTA_HEDGE_INCLUDE_EQUITY_BETA")
    if dh_beta is not None:
        cfg.delta_hedge_include_equity_portfolio_beta = str(dh_beta).strip().lower() in {"1", "true", "yes", "y"}
    dh_port = os.getenv("MSTOCK_DELTA_HEDGE_PORTFOLIO_HEDGE_ENABLE")
    if dh_port is not None:
        cfg.delta_hedge_portfolio_hedge_enable = str(dh_port).strip().lower() in {"1", "true", "yes", "y"}
    try:
        cfg.delta_hedge_beta_lookback_days = int(
            os.getenv("MSTOCK_DELTA_HEDGE_BETA_LOOKBACK_DAYS", str(cfg.delta_hedge_beta_lookback_days))
        )
    except Exception:
        pass
    try:
        cfg.delta_hedge_beta_max_symbols = int(
            os.getenv("MSTOCK_DELTA_HEDGE_BETA_MAX_SYMBOLS", str(cfg.delta_hedge_beta_max_symbols))
        )
    except Exception:
        pass
    try:
        cfg.delta_hedge_beta_refresh_sec = float(
            os.getenv("MSTOCK_DELTA_HEDGE_BETA_REFRESH_SEC", str(cfg.delta_hedge_beta_refresh_sec))
        )
    except Exception:
        pass
    try:
        cfg.delta_hedge_rebalance_interval_sec = float(
            os.getenv(
                "MSTOCK_DELTA_HEDGE_REBALANCE_SEC",
                str(cfg.delta_hedge_rebalance_interval_sec),
            )
        )
    except Exception:
        pass
    try:
        cfg.delta_hedge_step_qty = int(os.getenv("MSTOCK_DELTA_HEDGE_STEP_QTY", str(cfg.delta_hedge_step_qty)))
    except Exception:
        pass
    try:
        cfg.delta_hedge_max_abs_qty = int(
            os.getenv("MSTOCK_DELTA_HEDGE_MAX_ABS_QTY", str(cfg.delta_hedge_max_abs_qty))
        )
    except Exception:
        pass
    try:
        cfg.delta_hedge_assumed_iv = float(
            os.getenv("MSTOCK_DELTA_HEDGE_ASSUMED_IV", str(cfg.delta_hedge_assumed_iv))
        )
    except Exception:
        pass
    try:
        cfg.delta_hedge_rate = float(os.getenv("MSTOCK_DELTA_HEDGE_RATE", str(cfg.delta_hedge_rate)))
    except Exception:
        pass

    eq_en = os.getenv("MSTOCK_EQUITY_TRADE_ENABLE")
    if eq_en is not None:
        cfg.equity_trade_enable = str(eq_en).strip().lower() in {"1", "true", "yes", "y"}

    eq_engine = os.getenv("MSTOCK_EQUITY_TRADE_ENGINE")
    if eq_engine is not None and str(eq_engine).strip():
        cfg.equity_trade_engine = str(eq_engine).strip().upper()

    eq_h = os.getenv("MSTOCK_EQUITY_TRADE_HORIZON")
    if eq_h is not None and str(eq_h).strip():
        cfg.equity_trade_horizon = str(eq_h).strip().upper()

    eq_churn = os.getenv("MSTOCK_EQUITY_TRADE_CHURN_ENABLE")
    if eq_churn is not None:
        cfg.equity_trade_churn_enable = str(eq_churn).strip().lower() in {"1", "true", "yes", "y"}
    eq_gpt_watch = os.getenv("MSTOCK_EQUITY_TRADE_GPT_WATCHLIST_ENABLE")
    if eq_gpt_watch is not None:
        cfg.equity_trade_gpt_watchlist_enable = str(eq_gpt_watch).strip().lower() in {"1", "true", "yes", "y"}

    eq_lt_t = os.getenv("MSTOCK_EQUITY_LONGTERM_TARGET_PCT")
    if eq_lt_t is not None and str(eq_lt_t).strip():
        try:
            cfg.equity_longterm_target_pct = float(str(eq_lt_t).strip())
        except Exception:
            pass

    eq_lt_days = os.getenv("MSTOCK_EQUITY_LONGTERM_HORIZON_DAYS")
    if eq_lt_days is not None and str(eq_lt_days).strip():
        try:
            cfg.equity_longterm_horizon_days = int(float(str(eq_lt_days).strip()))
        except Exception:
            pass
    eq_short = os.getenv("MSTOCK_EQUITY_TRADE_ALLOW_SHORT")
    if eq_short is not None:
        cfg.equity_trade_allow_short = str(eq_short).strip().lower() in {"1", "true", "yes", "y"}
    eq_bench = os.getenv("MSTOCK_EQUITY_TRADE_BENCHMARK")
    if eq_bench is not None and str(eq_bench).strip():
        cfg.equity_trade_benchmark = str(eq_bench).strip().upper()
    eq_symbols = os.getenv("MSTOCK_EQUITY_TRADE_SYMBOLS")
    if eq_symbols is not None:
        cfg.equity_trade_symbols = str(eq_symbols)
    eq_tf = os.getenv("MSTOCK_EQUITY_TRADE_TIMEFRAME")
    if eq_tf is not None and str(eq_tf).strip():
        cfg.equity_trade_timeframe = str(eq_tf).strip().lower()
    try:
        cfg.equity_trade_max_symbols = int(
            os.getenv("MSTOCK_EQUITY_TRADE_MAX_SYMBOLS", str(cfg.equity_trade_max_symbols))
        )
    except Exception:
        pass
    try:
        cfg.equity_trade_rebalance_interval_sec = float(
            os.getenv("MSTOCK_EQUITY_TRADE_REBALANCE_SEC", str(cfg.equity_trade_rebalance_interval_sec))
        )
    except Exception:
        pass
    try:
        cfg.equity_trade_cooldown_sec = float(
            os.getenv("MSTOCK_EQUITY_TRADE_COOLDOWN_SEC", str(cfg.equity_trade_cooldown_sec))
        )
    except Exception:
        pass
    try:
        cfg.equity_trade_watchlist_refresh_sec = float(
            os.getenv("MSTOCK_EQUITY_TRADE_WATCHLIST_REFRESH_SEC", str(cfg.equity_trade_watchlist_refresh_sec))
        )
    except Exception:
        pass
    try:
        cfg.equity_trade_candidate_limit = int(
            os.getenv("MSTOCK_EQUITY_TRADE_CANDIDATE_LIMIT", str(cfg.equity_trade_candidate_limit))
        )
    except Exception:
        pass
    try:
        cfg.equity_trade_atr_period = int(
            os.getenv("MSTOCK_EQUITY_TRADE_ATR_PERIOD", str(cfg.equity_trade_atr_period))
        )
    except Exception:
        pass
    try:
        cfg.equity_trade_risk_rupees = float(
            os.getenv("MSTOCK_EQUITY_TRADE_RISK_RUPEES", str(cfg.equity_trade_risk_rupees))
        )
    except Exception:
        pass
    try:
        cfg.equity_trade_sl_atr_mult = float(
            os.getenv("MSTOCK_EQUITY_TRADE_SL_ATR_MULT", str(cfg.equity_trade_sl_atr_mult))
        )
    except Exception:
        pass
    try:
        cfg.equity_trade_tp_atr_mult = float(
            os.getenv("MSTOCK_EQUITY_TRADE_TP_ATR_MULT", str(cfg.equity_trade_tp_atr_mult))
        )
    except Exception:
        pass
    try:
        cfg.equity_trade_max_notional = float(
            os.getenv("MSTOCK_EQUITY_TRADE_MAX_NOTIONAL", str(cfg.equity_trade_max_notional))
        )
    except Exception:
        pass
    try:
        cfg.equity_trade_capital_rupees = float(
            os.getenv("MSTOCK_EQUITY_TRADE_CAPITAL_RUPEES", str(cfg.equity_trade_capital_rupees))
        )
    except Exception:
        pass
    eq_daily = os.getenv("MSTOCK_EQUITY_TRADE_GPT_MANAGE_DAILY")
    if eq_daily is not None:
        cfg.equity_trade_gpt_manage_daily = str(eq_daily).strip().lower() in {"1", "true", "yes", "y"}
    eq_prod = os.getenv("MSTOCK_EQUITY_TRADE_PRODUCT_TYPE")
    if eq_prod is not None and str(eq_prod).strip():
        cfg.equity_trade_product_type = str(eq_prod).strip().upper()
    eq_prod_lt = os.getenv("MSTOCK_EQUITY_TRADE_LONGTERM_PRODUCT_TYPE")
    if eq_prod_lt is not None and str(eq_prod_lt).strip():
        cfg.equity_trade_longterm_product_type = str(eq_prod_lt).strip().upper()
    eq_ot = os.getenv("MSTOCK_EQUITY_TRADE_ORDER_TYPE")
    if eq_ot is not None and str(eq_ot).strip():
        cfg.equity_trade_order_type = str(eq_ot).strip().upper()

    try:
        cfg.short_strike_distance = float(os.getenv("MSTOCK_SHORT_STRIKE_DISTANCE", str(cfg.short_strike_distance)))
    except Exception:
        pass
    try:
        cfg.straddle_atr_threshold = float(
            os.getenv("MSTOCK_STRADDLE_ATR_THRESHOLD", str(cfg.straddle_atr_threshold))
        )
    except Exception:
        pass
    try:
        cfg.wing_width = float(os.getenv("MSTOCK_WING_WIDTH", str(cfg.wing_width)))
    except Exception:
        pass
    try:
        cfg.max_hold_minutes = int(os.getenv("MSTOCK_MAX_HOLD_MINUTES", str(cfg.max_hold_minutes)))
    except Exception:
        pass
    try:
        cfg.max_hold_days = int(os.getenv("MSTOCK_MAX_HOLD_DAYS", str(getattr(cfg, "max_hold_days", 0) or 0)))
    except Exception:
        pass
    try:
        cfg.max_trend_strength = float(os.getenv("MSTOCK_MAX_TREND_STRENGTH", str(cfg.max_trend_strength)))
    except Exception:
        pass

    try:
        cfg.cooldown_sec = float(os.getenv("MSTOCK_COOLDOWN_SEC", str(cfg.cooldown_sec)))
    except Exception:
        pass
    try:
        cfg.cooldown_after_stopout_sec = float(
            os.getenv("MSTOCK_COOLDOWN_AFTER_STOPOUT_SEC", str(cfg.cooldown_after_stopout_sec))
        )
    except Exception:
        pass
    try:
        cfg.cooldown_atr_mult = float(
            os.getenv("MSTOCK_COOLDOWN_ATR_MULT", str(cfg.cooldown_atr_mult))
        )
    except Exception:
        pass

    try:
        cfg.max_stale_ltp_sec = float(os.getenv("MSTOCK_MAX_STALE_LTP_SEC", str(cfg.max_stale_ltp_sec)))
    except Exception:
        pass
    try:
        cfg.entry_candle_max_age_sec = float(
            os.getenv("MSTOCK_ENTRY_CANDLE_MAX_AGE_SEC", str(cfg.entry_candle_max_age_sec))
        )
    except Exception:
        pass
    try:
        cfg.entry_max_candle_range_atr_mult = float(
            os.getenv(
                "MSTOCK_ENTRY_MAX_CANDLE_RANGE_ATR_MULT",
                str(cfg.entry_max_candle_range_atr_mult),
            )
        )
    except Exception:
        pass
    try:
        cfg.entry_gap_atr_mult = float(
            os.getenv("MSTOCK_ENTRY_GAP_ATR_MULT", str(cfg.entry_gap_atr_mult))
        )
    except Exception:
        pass
    v = os.getenv("MSTOCK_SESSION_OPEN_START")
    if v is not None and str(v).strip():
        cfg.session_open_start_hhmm = str(v).strip()
    v = os.getenv("MSTOCK_SESSION_OPEN_END")
    if v is not None and str(v).strip():
        cfg.session_open_end_hhmm = str(v).strip()
    v = os.getenv("MSTOCK_SESSION_CLOSE_START")
    if v is not None and str(v).strip():
        cfg.session_close_start_hhmm = str(v).strip()
    v = os.getenv("MSTOCK_SESSION_CLOSE_END")
    if v is not None and str(v).strip():
        cfg.session_close_end_hhmm = str(v).strip()
    v = os.getenv("MSTOCK_SESSION_OPEN_MIN_ATR")
    if v is not None and str(v).strip():
        try:
            cfg.session_open_min_atr = float(v)
        except Exception:
            pass
    v = os.getenv("MSTOCK_SESSION_OPEN_MAX_ATR")
    if v is not None and str(v).strip():
        try:
            cfg.session_open_max_atr = float(v)
        except Exception:
            pass
    v = os.getenv("MSTOCK_SESSION_OPEN_PREMIUM_RSI_LOW")
    if v is not None and str(v).strip():
        try:
            cfg.session_open_premium_rsi_low = float(v)
        except Exception:
            pass
    v = os.getenv("MSTOCK_SESSION_OPEN_PREMIUM_RSI_HIGH")
    if v is not None and str(v).strip():
        try:
            cfg.session_open_premium_rsi_high = float(v)
        except Exception:
            pass
    v = os.getenv("MSTOCK_SESSION_MID_MIN_ATR")
    if v is not None and str(v).strip():
        try:
            cfg.session_mid_min_atr = float(v)
        except Exception:
            pass
    v = os.getenv("MSTOCK_SESSION_MID_MAX_ATR")
    if v is not None and str(v).strip():
        try:
            cfg.session_mid_max_atr = float(v)
        except Exception:
            pass
    v = os.getenv("MSTOCK_SESSION_MID_PREMIUM_RSI_LOW")
    if v is not None and str(v).strip():
        try:
            cfg.session_mid_premium_rsi_low = float(v)
        except Exception:
            pass
    v = os.getenv("MSTOCK_SESSION_MID_PREMIUM_RSI_HIGH")
    if v is not None and str(v).strip():
        try:
            cfg.session_mid_premium_rsi_high = float(v)
        except Exception:
            pass
    v = os.getenv("MSTOCK_SESSION_CLOSE_MIN_ATR")
    if v is not None and str(v).strip():
        try:
            cfg.session_close_min_atr = float(v)
        except Exception:
            pass
    v = os.getenv("MSTOCK_SESSION_CLOSE_MAX_ATR")
    if v is not None and str(v).strip():
        try:
            cfg.session_close_max_atr = float(v)
        except Exception:
            pass
    v = os.getenv("MSTOCK_SESSION_CLOSE_PREMIUM_RSI_LOW")
    if v is not None and str(v).strip():
        try:
            cfg.session_close_premium_rsi_low = float(v)
        except Exception:
            pass
    v = os.getenv("MSTOCK_SESSION_CLOSE_PREMIUM_RSI_HIGH")
    if v is not None and str(v).strip():
        try:
            cfg.session_close_premium_rsi_high = float(v)
        except Exception:
            pass
    try:
        cfg.max_trades_per_day = int(os.getenv("MSTOCK_MAX_TRADES_PER_DAY", str(cfg.max_trades_per_day)))
    except Exception:
        pass

    # Directional entry quality (reduce churn)
    try:
        cfg.dir_min_confirmations = int(
            os.getenv("MSTOCK_DIR_MIN_CONFIRMATIONS", str(getattr(cfg, "dir_min_confirmations", 4)))
        )
    except Exception:
        pass
    try:
        cfg.dir_min_score_diff = int(
            os.getenv("MSTOCK_DIR_MIN_SCORE_DIFF", str(getattr(cfg, "dir_min_score_diff", 1)))
        )
    except Exception:
        pass
    try:
        cfg.dir_momentum_atr_mult = float(
            os.getenv("MSTOCK_DIR_MOMENTUM_ATR_MULT", str(getattr(cfg, "dir_momentum_atr_mult", 0.10)))
        )
    except Exception:
        pass
    dir_allow_tie = os.getenv("MSTOCK_DIR_ALLOW_TIEBREAK")
    if dir_allow_tie is not None:
        cfg.dir_allow_tie_break_entries = str(dir_allow_tie).strip().lower() in {"1", "true", "yes", "y"}
    try:
        cfg.dir_ema_slope_atr_mult = float(
            os.getenv("MSTOCK_DIR_EMA_SLOPE_ATR_MULT", str(cfg.dir_ema_slope_atr_mult))
        )
    except Exception:
        pass
    try:
        cfg.entry_max_same_direction_qty = int(
            os.getenv("MSTOCK_ENTRY_MAX_SAME_DIRECTION_QTY", str(cfg.entry_max_same_direction_qty))
        )
    except Exception:
        pass
    try:
        cfg.iv_expand_threshold_pct = float(
            os.getenv("MSTOCK_IV_EXPAND_THRESHOLD_PCT", str(cfg.iv_expand_threshold_pct))
        )
    except Exception:
        pass
    try:
        cfg.iv_contract_threshold_pct = float(
            os.getenv("MSTOCK_IV_CONTRACT_THRESHOLD_PCT", str(cfg.iv_contract_threshold_pct))
        )
    except Exception:
        pass
    risk_enabled = os.getenv("MSTOCK_RISK_SCALE_ENABLED")
    if risk_enabled is not None:
        cfg.risk_scale_enabled = str(risk_enabled).strip().lower() in {"1", "true", "yes", "y"}
    try:
        cfg.risk_scale_stopout_factor = float(
            os.getenv("MSTOCK_RISK_SCALE_STOPOUT_FACTOR", str(cfg.risk_scale_stopout_factor))
        )
    except Exception:
        pass
    try:
        cfg.risk_scale_recovery_wins = int(
            os.getenv("MSTOCK_RISK_SCALE_RECOVERY_WINS", str(cfg.risk_scale_recovery_wins))
        )
    except Exception:
        pass
    try:
        cfg.risk_scale_atr_high = float(
            os.getenv("MSTOCK_RISK_SCALE_ATR_HIGH", str(cfg.risk_scale_atr_high))
        )
    except Exception:
        pass
    try:
        cfg.risk_scale_atr_high_factor = float(
            os.getenv("MSTOCK_RISK_SCALE_ATR_HIGH_FACTOR", str(cfg.risk_scale_atr_high_factor))
        )
    except Exception:
        pass
    try:
        cfg.risk_scale_min_qty = int(
            os.getenv("MSTOCK_RISK_SCALE_MIN_QTY", str(cfg.risk_scale_min_qty))
        )
    except Exception:
        pass
    try:
        cfg.risk_scale_max_qty = int(
            os.getenv("MSTOCK_RISK_SCALE_MAX_QTY", str(cfg.risk_scale_max_qty))
        )
    except Exception:
        pass
    try:
        cfg.risk_scale_step_qty = int(
            os.getenv("MSTOCK_RISK_SCALE_STEP_QTY", str(cfg.risk_scale_step_qty))
        )
    except Exception:
        pass

    # Premium-selling professional controls
    try:
        cfg.premium_mtm_stop_pct = float(os.getenv("MSTOCK_PREMIUM_MTM_STOP_PCT", str(cfg.premium_mtm_stop_pct)))
    except Exception:
        pass
    try:
        cfg.premium_mtm_target_pct = float(
            os.getenv("MSTOCK_PREMIUM_MTM_TARGET_PCT", str(cfg.premium_mtm_target_pct))
        )
    except Exception:
        pass
    try:
        cfg.premium_mtm_trail_start_pct = float(
            os.getenv("MSTOCK_PREMIUM_MTM_TRAIL_START_PCT", str(cfg.premium_mtm_trail_start_pct))
        )
    except Exception:
        pass
    try:
        cfg.premium_mtm_trail_stop_pct = float(
            os.getenv("MSTOCK_PREMIUM_MTM_TRAIL_STOP_PCT", str(cfg.premium_mtm_trail_stop_pct))
        )
    except Exception:
        pass
    try:
        cfg.entry_min_option_premium = float(
            os.getenv("MSTOCK_ENTRY_MIN_OPTION_PREMIUM", str(cfg.entry_min_option_premium))
        )
    except Exception:
        pass
    try:
        cfg.entry_max_option_premium = float(
            os.getenv("MSTOCK_ENTRY_MAX_OPTION_PREMIUM", str(cfg.entry_max_option_premium))
        )
    except Exception:
        pass
    try:
        cfg.entry_min_total_premium = float(
            os.getenv("MSTOCK_ENTRY_MIN_TOTAL_PREMIUM", str(cfg.entry_min_total_premium))
        )
    except Exception:
        pass
    try:
        cfg.entry_max_total_premium = float(
            os.getenv("MSTOCK_ENTRY_MAX_TOTAL_PREMIUM", str(cfg.entry_max_total_premium))
        )
    except Exception:
        pass
    bidask_req = os.getenv("MSTOCK_ENTRY_REQUIRE_BID_ASK")
    if bidask_req is not None:
        cfg.entry_require_bid_ask = str(bidask_req).strip().lower() in {"1", "true", "yes", "y"}
    try:
        cfg.entry_max_bid_ask_spread_pct = float(
            os.getenv("MSTOCK_ENTRY_MAX_BID_ASK_SPREAD_PCT", str(cfg.entry_max_bid_ask_spread_pct))
        )
    except Exception:
        pass
    try:
        cfg.entry_max_bid_ask_spread_abs = float(
            os.getenv("MSTOCK_ENTRY_MAX_BID_ASK_SPREAD_ABS", str(cfg.entry_max_bid_ask_spread_abs))
        )
    except Exception:
        pass
    try:
        cfg.min_option_premium = float(
            os.getenv("MSTOCK_MIN_OPTION_PREMIUM", str(cfg.min_option_premium))
        )
    except Exception:
        pass
    try:
        cfg.entry_spread_shock_mult = float(
            os.getenv("MSTOCK_ENTRY_SPREAD_SHOCK_MULT", str(cfg.entry_spread_shock_mult))
        )
    except Exception:
        pass
    try:
        cfg.entry_spread_shock_lookback = int(
            float(os.getenv("MSTOCK_ENTRY_SPREAD_SHOCK_LOOKBACK", str(cfg.entry_spread_shock_lookback)))
        )
    except Exception:
        pass
    try:
        cfg.entry_spread_shock_min_samples = int(
            float(os.getenv("MSTOCK_ENTRY_SPREAD_SHOCK_MIN_SAMPLES", str(cfg.entry_spread_shock_min_samples)))
        )
    except Exception:
        pass

    cfg.premium_force_exit_hhmm = os.getenv("MSTOCK_PREMIUM_FORCE_EXIT", cfg.premium_force_exit_hhmm).strip()
    cfg.premium_entry_cutoff_hhmm = os.getenv(
        "MSTOCK_PREMIUM_ENTRY_CUTOFF", cfg.premium_entry_cutoff_hhmm
    ).strip()

    try:
        cfg.vwap_max_dev_pct = float(os.getenv("MSTOCK_VWAP_MAX_DEV_PCT", str(cfg.vwap_max_dev_pct)))
    except Exception:
        pass

    vwap_enabled = os.getenv("MSTOCK_ENABLE_VWAP_FILTER")
    if vwap_enabled is not None:
        cfg.enable_vwap_filter = str(vwap_enabled).strip().lower() in {"1", "true", "yes", "y"}

    try:
        cfg.premium_rsi_low = float(os.getenv("MSTOCK_PREMIUM_RSI_LOW", str(cfg.premium_rsi_low)))
    except Exception:
        pass
    try:
        cfg.premium_rsi_high = float(os.getenv("MSTOCK_PREMIUM_RSI_HIGH", str(cfg.premium_rsi_high)))
    except Exception:
        pass
    try:
        cfg.opening_max_move_pct = float(os.getenv("MSTOCK_MAX_OPEN_MOVE_PCT", str(cfg.opening_max_move_pct)))
    except Exception:
        pass
    try:
        cfg.opening_filter_minutes = int(
            os.getenv("MSTOCK_OPENING_FILTER_MINUTES", str(cfg.opening_filter_minutes))
        )
    except Exception:
        pass

    opening_enabled = os.getenv("MSTOCK_ENABLE_OPENING_FILTER")
    if opening_enabled is not None:
        cfg.enable_opening_filter = str(opening_enabled).strip().lower() in {"1", "true", "yes", "y"}

    trend_enabled = os.getenv("MSTOCK_ENABLE_TREND_FILTER")
    if trend_enabled is not None:
        cfg.enable_trend_filter = str(trend_enabled).strip().lower() in {"1", "true", "yes", "y"}

    rsi_enabled = os.getenv("MSTOCK_ENABLE_PREMIUM_RSI_FILTER")
    if rsi_enabled is not None:
        cfg.enable_premium_rsi_filter = str(rsi_enabled).strip().lower() in {"1", "true", "yes", "y"}

    no_signal = os.getenv("MSTOCK_DEBUG_NO_SIGNAL")
    if no_signal is not None:
        cfg.debug_log_no_signal = str(no_signal).strip().lower() in {"1", "true", "yes", "y"}
    try:
        cfg.atr_spike_mult = float(os.getenv("MSTOCK_ATR_SPIKE_MULT", str(cfg.atr_spike_mult)))
    except Exception:
        pass
    try:
        cfg.atr_period = int(os.getenv("MSTOCK_ATR_PERIOD", str(cfg.atr_period)))
    except Exception:
        pass

    try:
        cfg.min_atr = float(os.getenv("MSTOCK_MIN_ATR", str(cfg.min_atr)))
    except Exception:
        pass

    # ADX & Volume Settings
    adx_enabled = os.getenv("MSTOCK_ENABLE_ADX_FILTER")
    if adx_enabled is not None:
        cfg.enable_adx_filter = str(adx_enabled).strip().lower() in {"1", "true", "yes", "y"}
    try:
        cfg.adx_period = int(os.getenv("MSTOCK_ADX_PERIOD", str(cfg.adx_period)))
    except Exception:
        pass
    try:
        cfg.adx_min_strength = float(os.getenv("MSTOCK_ADX_MIN_STRENGTH", str(cfg.adx_min_strength)))
    except Exception:
        pass

    vol_enabled = os.getenv("MSTOCK_ENABLE_VOLUME_FILTER")
    if vol_enabled is not None:
        cfg.enable_volume_filter = str(vol_enabled).strip().lower() in {"1", "true", "yes", "y"}
    try:
        cfg.volume_sma_period = int(os.getenv("MSTOCK_VOLUME_SMA_PERIOD", str(cfg.volume_sma_period)))
    except Exception:
        pass
    try:
        cfg.max_atr = float(os.getenv("MSTOCK_MAX_ATR", str(cfg.max_atr)))
    except Exception:
        pass

    # Supertrend Filter Settings
    st_enabled = os.getenv("MSTOCK_ENABLE_SUPERTREND_FILTER")
    if st_enabled is not None:
        cfg.enable_supertrend_filter = str(st_enabled).strip().lower() in {"1", "true", "yes", "y"}

    st_mode = os.getenv("MSTOCK_SUPERTREND_MODE")
    if st_mode is not None:
        m = str(st_mode).strip().lower()
        if m in {"trend", "trend_follow", "trend_following", "follow"}:
            cfg.supertrend_mode = "trend"
        elif m in {"counter", "counter_trend", "countertrend", "fade"}:
            cfg.supertrend_mode = "counter"
        else:
            # Ignore invalid values and keep default.
            pass
    try:
        cfg.supertrend_period = int(os.getenv("MSTOCK_SUPERTREND_PERIOD", str(cfg.supertrend_period)))
    except Exception:
        pass
    try:
        cfg.supertrend_multiplier = float(os.getenv("MSTOCK_SUPERTREND_MULTIPLIER", str(cfg.supertrend_multiplier)))
    except Exception:
        pass

    try:
        cfg.supertrend_vote_weight = int(os.getenv("MSTOCK_SUPERTREND_VOTE_WEIGHT", str(cfg.supertrend_vote_weight)))
    except Exception:
        pass

    # Directional Dynamic Strike Selection Settings
    dds_enabled = os.getenv("MSTOCK_ENABLE_DIRECTIONAL_DYNAMIC_STRIKES")
    if dds_enabled is not None:
        cfg.enable_directional_dynamic_strikes = str(dds_enabled).strip().lower() in {"1", "true", "yes", "y"}
    try:
        cfg.dir_strike_atr_multiplier_low = float(os.getenv("MSTOCK_DIR_STRIKE_ATR_MULT_LOW", str(cfg.dir_strike_atr_multiplier_low)))
    except Exception:
        pass
    try:
        cfg.dir_strike_atr_multiplier_mid = float(os.getenv("MSTOCK_DIR_STRIKE_ATR_MULT_MID", str(cfg.dir_strike_atr_multiplier_mid)))
    except Exception:
        pass
    try:
        cfg.dir_strike_atr_multiplier_high = float(os.getenv("MSTOCK_DIR_STRIKE_ATR_MULT_HIGH", str(cfg.dir_strike_atr_multiplier_high)))
    except Exception:
        pass
    try:
        cfg.dir_strike_atr_threshold_low = float(os.getenv("MSTOCK_DIR_STRIKE_ATR_THRESHOLD_LOW", str(cfg.dir_strike_atr_threshold_low)))
    except Exception:
        pass
    try:
        cfg.dir_strike_atr_threshold_high = float(os.getenv("MSTOCK_DIR_STRIKE_ATR_THRESHOLD_HIGH", str(cfg.dir_strike_atr_threshold_high)))
    except Exception:
        pass
    try:
        cfg.dir_strike_itm_multiplier_low = float(os.getenv("MSTOCK_DIR_STRIKE_ITM_MULT_LOW", str(cfg.dir_strike_itm_multiplier_low)))
    except Exception:
        pass
    try:
        cfg.dir_strike_itm_multiplier_mid = float(os.getenv("MSTOCK_DIR_STRIKE_ITM_MULT_MID", str(cfg.dir_strike_itm_multiplier_mid)))
    except Exception:
        pass
    try:
        cfg.dir_strike_itm_multiplier_high = float(os.getenv("MSTOCK_DIR_STRIKE_ITM_MULT_HIGH", str(cfg.dir_strike_itm_multiplier_high)))
    except Exception:
        pass
    try:
        cfg.dir_use_itm_above_atr = float(os.getenv("MSTOCK_DIR_USE_ITM_ABOVE_ATR", str(cfg.dir_use_itm_above_atr)))
    except Exception:
        pass
    try:
        cfg.dir_use_itm_below_atr = float(os.getenv("MSTOCK_DIR_USE_ITM_BELOW_ATR", str(cfg.dir_use_itm_below_atr)))
    except Exception:
        pass

    # Dynamic Strike Selection Settings
    ds_enabled = os.getenv("MSTOCK_ENABLE_DYNAMIC_STRIKES")
    if ds_enabled is not None:
        cfg.enable_dynamic_strikes = str(ds_enabled).strip().lower() in {"1", "true", "yes", "y"}
    try:
        cfg.strike_atr_multiplier_low = float(os.getenv("MSTOCK_STRIKE_ATR_MULT_LOW", str(cfg.strike_atr_multiplier_low)))
    except Exception:
        pass
    try:
        cfg.strike_atr_multiplier_mid = float(os.getenv("MSTOCK_STRIKE_ATR_MULT_MID", str(cfg.strike_atr_multiplier_mid)))
    except Exception:
        pass
    try:
        cfg.strike_atr_multiplier_high = float(os.getenv("MSTOCK_STRIKE_ATR_MULT_HIGH", str(cfg.strike_atr_multiplier_high)))
    except Exception:
        pass
    try:
        cfg.strike_atr_threshold_low = float(os.getenv("MSTOCK_STRIKE_ATR_THRESHOLD_LOW", str(cfg.strike_atr_threshold_low)))
    except Exception:
        pass
    try:
        cfg.strike_atr_threshold_high = float(os.getenv("MSTOCK_STRIKE_ATR_THRESHOLD_HIGH", str(cfg.strike_atr_threshold_high)))
    except Exception:
        pass


    exit_touch = os.getenv("MSTOCK_PREMIUM_EXIT_ON_SHORT_TOUCH")
    if exit_touch is not None:
        cfg.premium_exit_on_short_strike_touch = str(exit_touch).strip().lower() in {"1", "true", "yes", "y"}

    exit_wing = os.getenv("MSTOCK_PREMIUM_EXIT_ON_WING_TOUCH")
    if exit_wing is not None:
        cfg.premium_exit_on_wing_touch = str(exit_wing).strip().lower() in {"1", "true", "yes", "y"}

    try:
        cfg.max_stopouts_per_day = int(os.getenv("MSTOCK_MAX_STOPOUTS_PER_DAY", str(cfg.max_stopouts_per_day)))
    except Exception:
        pass

    try:
        cfg.max_consecutive_stopouts = int(
            os.getenv("MSTOCK_MAX_CONSECUTIVE_STOPOUTS", str(cfg.max_consecutive_stopouts))
        )
    except Exception:
        pass

    try:
        cfg.max_consecutive_same_trade_type = int(
            os.getenv(
                "MSTOCK_MAX_CONSECUTIVE_SAME_TRADE_TYPE",
                str(getattr(cfg, "max_consecutive_same_trade_type", 0)),
            )
        )
    except Exception:
        pass

    req_spot = os.getenv("MSTOCK_REQUIRE_SPOT_LTP_FOR_ENTRY")
    if req_spot is not None:
        cfg.require_spot_ltp_for_entry = str(req_spot).strip().lower() in {"1", "true", "yes", "y"}

    try:
        cfg.dir_breakeven_atr_mult = float(
            os.getenv("MSTOCK_DIR_BREAKEVEN_ATR_MULT", str(cfg.dir_breakeven_atr_mult))
        )
    except Exception:
        pass
    try:
        cfg.dir_partial_target_mult = float(os.getenv("MSTOCK_DIR_PARTIAL_TARGET_MULT", str(cfg.dir_partial_target_mult)))
    except Exception:
        pass
    try:
        cfg.dir_partial_qty_pct = float(os.getenv("MSTOCK_DIR_PARTIAL_QTY_PCT", str(cfg.dir_partial_qty_pct)))
    except Exception:
        pass
    try:
        cfg.dir_post_partial_be_atr_mult = float(
            os.getenv("MSTOCK_DIR_POST_PARTIAL_BE_ATR_MULT", str(cfg.dir_post_partial_be_atr_mult))
        )
    except Exception:
        pass
    try:
        cfg.dir_post_partial_trail_atr_mult = float(
            os.getenv("MSTOCK_DIR_POST_PARTIAL_TRAIL_ATR_MULT", str(cfg.dir_post_partial_trail_atr_mult))
        )
    except Exception:
        pass
    try:
        cfg.dir_trail_atr_mult = float(os.getenv("MSTOCK_DIR_TRAIL_ATR_MULT", str(cfg.dir_trail_atr_mult)))
    except Exception:
        pass
    try:
        cfg.dir_premium_trail_pct = float(
            os.getenv("MSTOCK_DIR_PREMIUM_TRAIL_PCT", str(cfg.dir_premium_trail_pct))
        )
    except Exception:
        pass

    # Directional style selection
    cfg.directional_trade_style = os.getenv(
        "MSTOCK_DIR_TRADE_STYLE",
        os.getenv("MSTOCK_DIRECTIONAL_TRADE_STYLE", cfg.directional_trade_style),
    ).strip().lower()
    try:
        cfg.directional_style_lock_minutes = int(
            os.getenv(
                "MSTOCK_DIR_STYLE_LOCK_MINUTES",
                os.getenv("MSTOCK_DIRECTIONAL_STYLE_LOCK_MINUTES", str(cfg.directional_style_lock_minutes)),
            )
        )
    except Exception:
        pass
    try:
        cfg.dir_vol_decreasing_steps = int(
            os.getenv(
                "MSTOCK_DIR_VOL_DECREASING_STEPS",
                os.getenv("MSTOCK_DIRECTIONAL_VOL_DECREASING_STEPS", str(cfg.dir_vol_decreasing_steps)),
            )
        )
    except Exception:
        pass

    flip_long = os.getenv("MSTOCK_DIR_FLIP_LONG_TO_SHORT_ON_STOP")
    if flip_long is not None:
        cfg.dir_flip_long_to_short_on_stop = str(flip_long).strip().lower() in {"1", "true", "yes", "y"}

    try:
        cfg.ema_fast = int(os.getenv("MSTOCK_EMA_FAST", str(cfg.ema_fast)))
    except Exception:
        pass
    try:
        cfg.ema_slow = int(os.getenv("MSTOCK_EMA_SLOW", str(cfg.ema_slow)))
    except Exception:
        pass

    try:
        cfg.enable_delta_strike_selection = str(os.getenv("MSTOCK_ENABLE_DELTA_STRIKE_SELECTION", str(cfg.enable_delta_strike_selection))).lower() in {"1", "true", "yes", "y"}
        cfg.target_delta = float(os.getenv("MSTOCK_TARGET_DELTA", str(cfg.target_delta)))
        cfg.enable_theta_decay_filter = str(os.getenv("MSTOCK_ENABLE_THETA_DECAY_FILTER", str(cfg.enable_theta_decay_filter))).lower() in {"1", "true", "yes", "y"}
        
        cfg.dir_sl_atr_mult = float(os.getenv("MSTOCK_DIR_SL_ATR_MULT", str(cfg.dir_sl_atr_mult)))
        cfg.dir_tp_atr_mult = float(os.getenv("MSTOCK_DIR_TP_ATR_MULT", str(cfg.dir_tp_atr_mult)))
        cfg.choppiness_hard_filter = str(os.getenv("MSTOCK_CHOPPINESS_HARD_FILTER", str(cfg.choppiness_hard_filter))).lower() in {"1", "true", "yes", "y"}
        cfg.enable_rsi_confluence = str(os.getenv("MSTOCK_ENABLE_RSI_CONFLUENCE", str(cfg.enable_rsi_confluence))).lower() in {"1", "true", "yes", "y"}
    except Exception:
        pass

    live = os.getenv("MSTOCK_ENABLE_LIVE_TRADING")
    if live is not None:
        cfg.enable_live_trading = str(live).strip().lower() in {"1", "true", "yes", "y"}

    paper_mode = os.getenv("ML_PAPER_MODE_ENABLED")
    if paper_mode is not None:
        cfg.ml_paper_mode_enabled = str(paper_mode).strip().lower() in {"1", "true", "yes", "y"}
    manifest_path = os.getenv("ML_DEPLOYMENT_MANIFEST_PATH")
    if manifest_path is not None:
        cfg.ml_deployment_manifest_path = str(manifest_path).strip()
    try:
        cfg.ml_min_confidence_threshold = float(os.getenv("ML_MIN_CONFIDENCE_THRESHOLD", str(cfg.ml_min_confidence_threshold)))
    except Exception:
        pass
    try:
        cfg.ml_max_predictions_per_day = int(os.getenv("ML_MAX_PREDICTIONS_PER_DAY", str(cfg.ml_max_predictions_per_day)))
    except Exception:
        pass
    feature_log = os.getenv("ML_LOG_FEATURE_VECTOR")
    if feature_log is not None:
        cfg.ml_log_feature_vector = str(feature_log).strip().lower() in {"1", "true", "yes", "y"}
    reason_log = os.getenv("ML_LOG_PREDICTION_REASON")
    if reason_log is not None:
        cfg.ml_log_prediction_reason = str(reason_log).strip().lower() in {"1", "true", "yes", "y"}
    fail_closed = os.getenv("ML_FAIL_CLOSED_ON_SCHEMA_MISMATCH")
    if fail_closed is not None:
        cfg.ml_fail_closed_on_schema_mismatch = str(fail_closed).strip().lower() in {"1", "true", "yes", "y"}
    kill_switch = os.getenv("ML_DISABLE_ALL")
    if kill_switch is not None:
        cfg.ml_disable_all = str(kill_switch).strip().lower() in {"1", "true", "yes", "y"}
    try:
        cfg.ml_max_daily_paper_loss = float(os.getenv("ML_MAX_DAILY_PAPER_LOSS", str(cfg.ml_max_daily_paper_loss)))
    except Exception:
        pass
    try:
        cfg.ml_max_consecutive_paper_losses = int(os.getenv("ML_MAX_CONSECUTIVE_PAPER_LOSSES", str(cfg.ml_max_consecutive_paper_losses)))
    except Exception:
        pass

    # ---- Paper Forward-Test Runner ----
    try:
        cfg.paper_forward_test_interval_sec = int(
            os.getenv("SCALPER_PAPER_FWD_INTERVAL_SEC", str(cfg.paper_forward_test_interval_sec))
        )
    except Exception:
        pass
    try:
        cfg.paper_forward_test_max_trades_per_day = int(
            os.getenv("SCALPER_PAPER_FWD_MAX_TRADES", str(cfg.paper_forward_test_max_trades_per_day))
        )
    except Exception:
        pass
    cfg.paper_forward_test_output_dir = os.getenv(
        "SCALPER_PAPER_FWD_OUTPUT_DIR", cfg.paper_forward_test_output_dir
    ).strip()
    try:
        cfg.paper_forward_test_require_market_hours = str(
            os.getenv("SCALPER_PAPER_FWD_REQUIRE_MARKET_HOURS", str(cfg.paper_forward_test_require_market_hours))
        ).strip().lower() in {"1", "true", "yes", "y"}
    except Exception:
        pass
    try:
        cfg.paper_forward_test_min_runtime_minutes = int(
            os.getenv("SCALPER_PAPER_FWD_MIN_RUNTIME_MINUTES", str(cfg.paper_forward_test_min_runtime_minutes))
        )
    except Exception:
        pass

    try:
        cfg.enable_micro_live = str(os.getenv("SCALPER_ENABLE_MICRO_LIVE", "")).strip().lower() in {"1", "true", "yes", "y"}
        cfg.micro_live_max_trades_per_day = int(os.getenv("SCALPER_MICRO_LIVE_MAX_TRADES", str(cfg.micro_live_max_trades_per_day)))
        cfg.micro_live_max_lots = int(os.getenv("SCALPER_MICRO_LIVE_MAX_LOTS", str(cfg.micro_live_max_lots)))
        cfg.micro_live_min_probability = float(os.getenv("SCALPER_MICRO_LIVE_MIN_PROB", str(cfg.micro_live_min_probability)))
        cfg.micro_live_max_daily_loss = float(os.getenv("SCALPER_MICRO_LIVE_MAX_DAILY_LOSS", str(cfg.micro_live_max_daily_loss)))
        cfg.micro_live_force_exit_time = str(os.getenv("SCALPER_MICRO_LIVE_FORCE_EXIT", str(cfg.micro_live_force_exit_time)))
        cfg.micro_live_require_spread_pct = float(os.getenv("SCALPER_MICRO_LIVE_MAX_SPREAD", str(cfg.micro_live_require_spread_pct)))
        cfg.micro_live_require_min_premium = float(os.getenv("SCALPER_MICRO_LIVE_MIN_PREMIUM", str(cfg.micro_live_require_min_premium)))
    except Exception:
        pass

    weekly_only = os.getenv("MSTOCK_NIFTY_WEEKLY_ONLY")
    if weekly_only is not None:
        cfg.nifty_weekly_only = str(weekly_only).strip().lower() in {"1", "true", "yes", "y"}

    bypass_weekly = os.getenv("MSTOCK_BYPASS_WEEKLY_FILTER")
    if bypass_weekly is not None:
        cfg.bypass_weekly_filter = str(bypass_weekly).strip().lower() in {"1", "true", "yes", "y"}

    # Presets apply last, but should not clobber explicit env vars.
    def _env_present(key: str) -> bool:
        v = os.getenv(key)
        return v is not None and str(v).strip() != ""

    if cfg.preset == "aggresive":
        cfg.preset = "aggressive"

    if cfg.preset in {"aggressive", "conservative"}:
        if cfg.preset == "aggressive":
            if not _env_present("MSTOCK_MAX_OPEN_POSITIONS"):
                cfg.max_open_positions = 6
            if not _env_present("MSTOCK_STRATEGY"):
                cfg.strategy_name = "auto"
            if not _env_present("MSTOCK_TIMEFRAME"):
                cfg.timeframe = "1m"

            if not _env_present("MSTOCK_COOLDOWN_SEC"):
                cfg.cooldown_sec = 20.0
            if not _env_present("MSTOCK_COOLDOWN_AFTER_STOPOUT_SEC"):
                cfg.cooldown_after_stopout_sec = 75.0
            if not _env_present("MSTOCK_COOLDOWN_ATR_MULT"):
                cfg.cooldown_atr_mult = 0.25
            if not _env_present("MSTOCK_MAX_HOLD_MINUTES"):
                cfg.max_hold_minutes = 35
            if not _env_present("MSTOCK_MAX_CONSECUTIVE_SAME_TRADE_TYPE"):
                cfg.max_consecutive_same_trade_type = 0
            if not _env_present("MSTOCK_ENTRY_MAX_SAME_DIRECTION_QTY"):
                cfg.entry_max_same_direction_qty = 600
            if not _env_present("MSTOCK_MAX_PYRAMID_LEVELS"):
                cfg.max_pyramid_levels = 3
            if not _env_present("MSTOCK_DIR_PREMIUM_TRAIL_PCT"):
                cfg.dir_premium_trail_pct = 0.07
            if not _env_present("MSTOCK_DIR_PARTIAL_TARGET_MULT"):
                cfg.dir_partial_target_mult = 1.2
            if not _env_present("MSTOCK_DIR_PARTIAL_QTY_PCT"):
                cfg.dir_partial_qty_pct = 0.35
            if not _env_present("MSTOCK_RISK_SCALE_ENABLED"):
                cfg.risk_scale_enabled = True
            if not _env_present("MSTOCK_RISK_SCALE_STOPOUT_FACTOR"):
                cfg.risk_scale_stopout_factor = 0.90
            if not _env_present("MSTOCK_RISK_SCALE_RECOVERY_WINS"):
                cfg.risk_scale_recovery_wins = 1
            if not _env_present("MSTOCK_RISK_SCALE_ATR_HIGH"):
                cfg.risk_scale_atr_high = 85.0
            if not _env_present("MSTOCK_RISK_SCALE_ATR_HIGH_FACTOR"):
                cfg.risk_scale_atr_high_factor = 0.90
            if not _env_present("MSTOCK_RISK_SCALE_MIN_QTY"):
                cfg.risk_scale_min_qty = int(getattr(cfg, "lot_size", 65) or 65)
            if not _env_present("MSTOCK_RISK_SCALE_MAX_QTY"):
                cfg.risk_scale_max_qty = 600
            if not _env_present("MSTOCK_RISK_SCALE_STEP_QTY"):
                cfg.risk_scale_step_qty = int(getattr(cfg, "lot_size", 65) or 65)
            if not _env_present("MSTOCK_STRATEGY_ROUTER_MODE"):
                cfg.strategy_router_mode = "aggressive"
            if not _env_present("MSTOCK_IV_FILTER_ENABLED"):
                cfg.iv_filter_enabled = True
            if not _env_present("MSTOCK_MTF_HARD_FILTER"):
                cfg.mtf_hard_filter = True
            if not _env_present("MSTOCK_THETA_STOP_WIDEN_ENABLED"):
                cfg.theta_stop_widen_enabled = True
            if not _env_present("MSTOCK_THETA_STOP_WIDEN_MAX_MULT"):
                cfg.theta_stop_widen_max_mult = 2.5
            if not _env_present("MSTOCK_THETA_STOP_WIDEN_THRESHOLD"):
                cfg.theta_stop_widen_threshold = -8.0
            if not _env_present("MSTOCK_EXIT_LADDER_ENABLED"):
                cfg.exit_ladder_enabled = True
            if not _env_present("MSTOCK_EXIT_LADDER_STEPS"):
                cfg.exit_ladder_steps = "0.25:0.8,0.25:1.2,0.30:1.8"
            if not _env_present("MSTOCK_DELTA_HEDGE_VOL_ADJUST_ENABLED"):
                cfg.delta_hedge_vol_adjust_enabled = True
            if not _env_present("MSTOCK_DELTA_HEDGE_VOL_LOW_TOL_FACTOR"):
                cfg.delta_hedge_vol_low_tol_factor = 0.6
            if not _env_present("MSTOCK_DELTA_HEDGE_VOL_HIGH_TOL_FACTOR"):
                cfg.delta_hedge_vol_high_tol_factor = 1.6

        else:
            if not _env_present("MSTOCK_MAX_OPEN_POSITIONS"):
                cfg.max_open_positions = 6
            if not _env_present("MSTOCK_STRATEGY"):
                cfg.strategy_name = "auto"
            if not _env_present("MSTOCK_TIMEFRAME"):
                cfg.timeframe = "3m"

            if not _env_present("MSTOCK_COOLDOWN_SEC"):
                cfg.cooldown_sec = 45.0
            if not _env_present("MSTOCK_COOLDOWN_AFTER_STOPOUT_SEC"):
                cfg.cooldown_after_stopout_sec = 180.0
            if not _env_present("MSTOCK_COOLDOWN_ATR_MULT"):
                cfg.cooldown_atr_mult = 0.45
            if not _env_present("MSTOCK_MAX_HOLD_MINUTES"):
                cfg.max_hold_minutes = 60
            if not _env_present("MSTOCK_MAX_CONSECUTIVE_SAME_TRADE_TYPE"):
                cfg.max_consecutive_same_trade_type = 0
            if not _env_present("MSTOCK_ENTRY_MAX_SAME_DIRECTION_QTY"):
                cfg.entry_max_same_direction_qty = 150
            if not _env_present("MSTOCK_MAX_PYRAMID_LEVELS"):
                cfg.max_pyramid_levels = 1
            if not _env_present("MSTOCK_DIR_PREMIUM_TRAIL_PCT"):
                cfg.dir_premium_trail_pct = 0.10
            if not _env_present("MSTOCK_DIR_PARTIAL_TARGET_MULT"):
                cfg.dir_partial_target_mult = 1.0
            if not _env_present("MSTOCK_DIR_PARTIAL_QTY_PCT"):
                cfg.dir_partial_qty_pct = 0.25
            if not _env_present("MSTOCK_RISK_SCALE_ENABLED"):
                cfg.risk_scale_enabled = True
            if not _env_present("MSTOCK_RISK_SCALE_STOPOUT_FACTOR"):
                cfg.risk_scale_stopout_factor = 0.75
            if not _env_present("MSTOCK_RISK_SCALE_RECOVERY_WINS"):
                cfg.risk_scale_recovery_wins = 2
            if not _env_present("MSTOCK_RISK_SCALE_ATR_HIGH"):
                cfg.risk_scale_atr_high = 70.0
            if not _env_present("MSTOCK_RISK_SCALE_ATR_HIGH_FACTOR"):
                cfg.risk_scale_atr_high_factor = 0.80
            if not _env_present("MSTOCK_RISK_SCALE_MIN_QTY"):
                cfg.risk_scale_min_qty = int(getattr(cfg, "lot_size", 65) or 65)
            if not _env_present("MSTOCK_RISK_SCALE_MAX_QTY"):
                cfg.risk_scale_max_qty = 300
            if not _env_present("MSTOCK_ENABLE_DYNAMIC_PYRAMIDING"):
                cfg.enable_dynamic_pyramiding = False
            if not _env_present("MSTOCK_IV_FILTER_ENABLED"):
                cfg.iv_filter_enabled = True
            if not _env_present("MSTOCK_MTF_HARD_FILTER"):
                cfg.mtf_hard_filter = False
            if not _env_present("MSTOCK_THETA_STOP_WIDEN_ENABLED"):
                cfg.theta_stop_widen_enabled = True
            if not _env_present("MSTOCK_THETA_STOP_WIDEN_MAX_MULT"):
                cfg.theta_stop_widen_max_mult = 1.8
            if not _env_present("MSTOCK_THETA_STOP_WIDEN_THRESHOLD"):
                cfg.theta_stop_widen_threshold = -4.0
            if not _env_present("MSTOCK_EXIT_LADDER_ENABLED"):
                cfg.exit_ladder_enabled = True
            if not _env_present("MSTOCK_EXIT_LADDER_STEPS"):
                cfg.exit_ladder_steps = "0.20:1.0,0.20:1.5,0.20:2.0"
            if not _env_present("MSTOCK_DELTA_HEDGE_VOL_ADJUST_ENABLED"):
                cfg.delta_hedge_vol_adjust_enabled = True
            if not _env_present("MSTOCK_DELTA_HEDGE_VOL_LOW_TOL_FACTOR"):
                cfg.delta_hedge_vol_low_tol_factor = 0.8
            if not _env_present("MSTOCK_DELTA_HEDGE_VOL_HIGH_TOL_FACTOR"):
                cfg.delta_hedge_vol_high_tol_factor = 1.3

            if not _env_present("MSTOCK_RISK_SCALE_STEP_QTY"):
                cfg.risk_scale_step_qty = int(getattr(cfg, "lot_size", 65) or 65)
            if not _env_present("MSTOCK_STRATEGY_ROUTER_MODE"):
                cfg.strategy_router_mode = "balanced"

    # Auto-mode knobs
    try:
        cfg.auto_range_trend_mult = float(os.getenv("MSTOCK_AUTO_RANGE_TREND_MULT", str(cfg.auto_range_trend_mult)))
    except Exception:
        pass
    try:
        cfg.auto_strategy_lock_minutes = int(os.getenv("MSTOCK_AUTO_STRATEGY_LOCK_MINUTES", str(cfg.auto_strategy_lock_minutes)))
    except Exception:
        pass
    try:
        cfg.auto_dir_rsi_buy = float(os.getenv("MSTOCK_AUTO_DIR_RSI_BUY", str(cfg.auto_dir_rsi_buy)))
    except Exception:
        pass
    try:
        cfg.auto_dir_rsi_sell = float(os.getenv("MSTOCK_AUTO_DIR_RSI_SELL", str(cfg.auto_dir_rsi_sell)))
    except Exception:
        pass
    try:
        cfg.auto_dir_rsi_slop = float(os.getenv("MSTOCK_AUTO_DIR_RSI_SLOP", str(cfg.auto_dir_rsi_slop)))
    except Exception:
        pass

    # Regime tuning overrides (optional).
    for regime_key in ("TRENDING", "VOLATILE", "MEAN_REVERTING", "QUIET"):
        for suffix, caster in (("TREND_MULT", float), ("ML_THRESHOLD", float), ("PREFERRED_STRATEGY", str)):
            env_name = f"MSTOCK_REGIME_{regime_key}_{suffix}"
            v = os.getenv(env_name)
            if v is None:
                continue
            attr = f"regime_{regime_key.lower()}_{suffix.lower()}"
            try:
                if caster is float:
                    setattr(cfg, attr, float(v))
                else:
                    setattr(cfg, attr, str(v).strip())
            except Exception:
                pass

    # Signal Quality
    mtf_enabled = os.getenv("MSTOCK_ENABLE_MTF_CONFIRMATION")
    if mtf_enabled is not None:
        cfg.enable_mtf_confirmation = str(mtf_enabled).strip().lower() in {"1", "true", "yes", "y"}
    cfg.mtf_timeframe = os.getenv("MSTOCK_MTF_TIMEFRAME", cfg.mtf_timeframe)
    
    roc_enabled = os.getenv("MSTOCK_ENABLE_ROC_FILTER")
    if roc_enabled is not None:
        cfg.enable_roc_filter = str(roc_enabled).strip().lower() in {"1", "true", "yes", "y"}
    try:
        cfg.roc_period = int(os.getenv("MSTOCK_ROC_PERIOD", str(cfg.roc_period)))
    except Exception:
        pass
    try:
        cfg.roc_min_threshold = float(os.getenv("MSTOCK_ROC_MIN_THRESHOLD", str(cfg.roc_min_threshold)))
    except Exception:
        pass

    chop_enabled = os.getenv("MSTOCK_ENABLE_CHOPPINESS_FILTER")
    if chop_enabled is not None:
        cfg.enable_choppiness_filter = str(chop_enabled).strip().lower() in {"1", "true", "yes", "y"}
    try:
        cfg.choppiness_period = int(os.getenv("MSTOCK_CHOPPINESS_PERIOD", str(cfg.choppiness_period)))
    except Exception:
        pass
    try:
        cfg.choppiness_threshold = float(os.getenv("MSTOCK_CHOPPINESS_THRESHOLD", str(cfg.choppiness_threshold)))
    except Exception:
        pass

    # Risk Management
    try:
        cfg.stagnation_exit_minutes = int(os.getenv("MSTOCK_STAGNATION_EXIT_MINUTES", str(cfg.stagnation_exit_minutes)))
    except Exception:
        pass
    try:
        cfg.stagnation_pnl_threshold = float(os.getenv("MSTOCK_STAGNATION_PNL_THRESHOLD", str(cfg.stagnation_pnl_threshold)))
    except Exception:
        pass

    chandelier_enabled = os.getenv("MSTOCK_ENABLE_CHANDELIER_EXIT")
    if chandelier_enabled is not None:
        cfg.enable_chandelier_exit = str(chandelier_enabled).strip().lower() in {"1", "true", "yes", "y"}
    try:
        cfg.chandelier_period = int(os.getenv("MSTOCK_CHANDELIER_PERIOD", str(cfg.chandelier_period)))
    except Exception:
        pass
    try:
        cfg.chandelier_multiplier = float(os.getenv("MSTOCK_CHANDELIER_MULTIPLIER", str(cfg.chandelier_multiplier)))
    except Exception:
        pass

    pivot_enabled = os.getenv("MSTOCK_ENABLE_PIVOT_TARGETS")
    if pivot_enabled is not None:
        cfg.enable_pivot_targets = str(pivot_enabled).strip().lower() in {"1", "true", "yes", "y"}

    # Execution
    limit_enabled = os.getenv("MSTOCK_ENABLE_LIMIT_ORDERS")
    if limit_enabled is not None:
        cfg.enable_limit_orders = str(limit_enabled).strip().lower() in {"1", "true", "yes", "y"}
    try:
        cfg.paper_slippage_pct = float(os.getenv("MSTOCK_PAPER_SLIPPAGE_PCT", str(cfg.paper_slippage_pct)))
    except Exception:
        pass
    try:
        cfg.paper_extra_market_impact_pct = float(os.getenv("MSTOCK_PAPER_EXTRA_MARKET_IMPACT_PCT", str(cfg.paper_extra_market_impact_pct)))
    except Exception:
        pass
    try:
        cfg.paper_apply_brokerage_costs = str(
            os.getenv("MSTOCK_PAPER_APPLY_BROKERAGE_COSTS", str(cfg.paper_apply_brokerage_costs))
        ).strip().lower() in {"1", "true", "yes", "y"}
    except Exception:
        pass
    try:
        v_paper_src = str(os.getenv("MSTOCK_PAPER_COST_MODEL_SOURCE", str(cfg.paper_cost_model_source)) or "cost_model_assumptions").strip().lower()
        if v_paper_src in {"cost_model_assumptions", "ml_execution_costs"}:
            cfg.paper_cost_model_source = v_paper_src
    except Exception:
        pass

    try:
        cfg.limit_price_buffer_pct = float(os.getenv("MSTOCK_LIMIT_PRICE_BUFFER_PCT", str(cfg.limit_price_buffer_pct)))
    except Exception:
        pass
    try:
        cfg.max_pyramid_levels = int(os.getenv("MSTOCK_MAX_PYRAMID_LEVELS", str(cfg.max_pyramid_levels)))
    except Exception:
        pass

    try:
        cfg.order_retry_attempts = int(os.getenv("MSTOCK_ORDER_RETRY_ATTEMPTS", str(cfg.order_retry_attempts)))
    except Exception:
        pass
    try:
        cfg.order_retry_delay_sec = float(os.getenv("MSTOCK_ORDER_RETRY_DELAY_SEC", str(cfg.order_retry_delay_sec)))
    except Exception:
        pass

    try:
        cfg.paper_use_bid_ask_execution = str(
            os.getenv("MSTOCK_PAPER_USE_BID_ASK_EXECUTION", str(cfg.paper_use_bid_ask_execution))
        ).strip().lower() in {"1", "true", "yes", "y"}
    except Exception:
        pass
    try:
        cfg.paper_allow_ltp_fallback = str(
            os.getenv("MSTOCK_PAPER_ALLOW_LTP_FALLBACK", str(cfg.paper_allow_ltp_fallback))
        ).strip().lower() in {"1", "true", "yes", "y"}
    except Exception:
        pass
    try:
        cfg.paper_ltp_fallback_spread_pct = float(
            os.getenv("MSTOCK_PAPER_LTP_FALLBACK_SPREAD_PCT", str(cfg.paper_ltp_fallback_spread_pct))
        )
    except Exception:
        pass

    v = os.getenv("MSTOCK_STRATEGY_ROUTER_MODE")
    if v is not None:
        try:
            cfg.strategy_router_mode = str(v).strip().lower() or cfg.strategy_router_mode
        except Exception:
            pass

    v = os.getenv("MSTOCK_DIAGNOSTICS_ENABLED")
    if v is not None:
        try:
            cfg.diagnostics_enabled = str(v).strip().lower() in {"1", "true", "yes", "y"}
        except Exception:
            pass

    try:
        cfg.max_portfolio_option_notional = float(
            os.getenv("MSTOCK_MAX_PORTFOLIO_OPTION_NOTIONAL", str(cfg.max_portfolio_option_notional))
        )
    except Exception:
        pass
    try:
        cfg.max_portfolio_delta_abs = float(
            os.getenv("MSTOCK_MAX_PORTFOLIO_DELTA_ABS", str(cfg.max_portfolio_delta_abs))
        )
    except Exception:
        pass

    warnings = validate_strategy_config(cfg)
    setattr(cfg, "validation_warnings", warnings)
    for w in warnings:
        try:
            print(f"[CONFIG][WARN] {w}")
        except Exception:
            pass

    return cfg


def validate_strategy_config(cfg: StrategyConfig) -> list[str]:
    """Normalize/clamp risky values and return warning messages."""
    warns: list[str] = []

    try:
        if int(cfg.max_open_positions) < 1:
            cfg.max_open_positions = 1
            warns.append("max_open_positions was <1; clamped to 1.")
    except Exception:
        pass
    try:
        if int(cfg.max_trades_per_day) < 1:
            cfg.max_trades_per_day = 1
            warns.append("max_trades_per_day was <1; clamped to 1.")
    except Exception:
        pass
    try:
        if float(cfg.cooldown_sec) < 0:
            cfg.cooldown_sec = 0.0
            warns.append("cooldown_sec was negative; clamped to 0.")
    except Exception:
        pass
    try:
        if int(cfg.order_retry_attempts) < 1:
            cfg.order_retry_attempts = 1
            warns.append("order_retry_attempts was <1; clamped to 1.")
    except Exception:
        pass
    try:
        if float(cfg.order_retry_delay_sec) < 0:
            cfg.order_retry_delay_sec = 0.0
            warns.append("order_retry_delay_sec was negative; clamped to 0.")
    except Exception:
        pass
    try:
        mode = str(getattr(cfg, "strategy_router_mode", "balanced") or "balanced").strip().lower()
        if mode not in {"conservative", "balanced", "aggressive"}:
            cfg.strategy_router_mode = "balanced"
            warns.append("strategy_router_mode invalid; set to 'balanced'.")
    except Exception:
        pass
    try:
        if float(cfg.max_portfolio_option_notional) < 0:
            cfg.max_portfolio_option_notional = 0.0
            warns.append("max_portfolio_option_notional was negative; disabled.")
    except Exception:
        pass
    try:
        if float(cfg.max_portfolio_delta_abs) < 0:
            cfg.max_portfolio_delta_abs = 0.0
            warns.append("max_portfolio_delta_abs was negative; disabled.")
    except Exception:
        pass
    try:
        if int(cfg.equity_trade_max_symbols) < 1:
            cfg.equity_trade_max_symbols = 1
            warns.append("equity_trade_max_symbols was <1; clamped to 1.")
    except Exception:
        pass
    try:
        if float(cfg.equity_trade_rebalance_interval_sec) < 0:
            cfg.equity_trade_rebalance_interval_sec = 0.0
            warns.append("equity_trade_rebalance_interval_sec was negative; clamped to 0.")
    except Exception:
        pass
    try:
        if float(cfg.equity_trade_cooldown_sec) < 0:
            cfg.equity_trade_cooldown_sec = 0.0
            warns.append("equity_trade_cooldown_sec was negative; clamped to 0.")
    except Exception:
        pass
    try:
        if float(cfg.equity_trade_watchlist_refresh_sec) < 0:
            cfg.equity_trade_watchlist_refresh_sec = 0.0
            warns.append("equity_trade_watchlist_refresh_sec was negative; clamped to 0.")
    except Exception:
        pass
    try:
        if int(cfg.equity_trade_candidate_limit) < 1:
            cfg.equity_trade_candidate_limit = 1
            warns.append("equity_trade_candidate_limit was <1; clamped to 1.")
    except Exception:
        pass
    try:
        if float(cfg.equity_trade_max_notional) < 0:
            cfg.equity_trade_max_notional = 0.0
            warns.append("equity_trade_max_notional was negative; disabled.")
    except Exception:
        pass
    try:
        if float(getattr(cfg, "equity_trade_capital_rupees", 0.0) or 0.0) < 0:
            cfg.equity_trade_capital_rupees = 0.0
            warns.append("equity_trade_capital_rupees was negative; disabled.")
    except Exception:
        pass
    try:
        h = str(getattr(cfg, "equity_trade_horizon", "INTRADAY") or "INTRADAY").strip().upper()
        if h not in {"INTRADAY", "LONGTERM", "BOTH"}:
            cfg.equity_trade_horizon = "INTRADAY"
            warns.append("equity_trade_horizon invalid; set to 'INTRADAY'.")
    except Exception:
        pass
    try:
        prod = str(getattr(cfg, "equity_trade_longterm_product_type", "CNC") or "CNC").strip().upper()
        if prod not in {"CNC", "DELIVERY", "INTRADAY"}:
            cfg.equity_trade_longterm_product_type = "CNC"
            warns.append("equity_trade_longterm_product_type invalid; set to 'CNC'.")
    except Exception:
        pass
    try:
        eq_enabled = bool(getattr(cfg, "equity_trade_enable", False))
        eq_engine = str(getattr(cfg, "equity_trade_engine", "INDICATORS") or "INDICATORS").strip().upper()
        gpt_enabled = bool(getattr(cfg, "gpt_enable", False))
        eq_gpt_watch = bool(getattr(cfg, "equity_trade_gpt_watchlist_enable", False))
        eq_gpt_manage = bool(getattr(cfg, "equity_trade_gpt_manage_daily", True))

        if eq_enabled and eq_engine == "GPT" and not gpt_enabled:
            warns.append("equity_trade_engine is GPT but gpt_enable is off; GPT equity entries will never run.")
        if eq_enabled and eq_engine != "GPT" and (eq_gpt_watch or eq_gpt_manage):
            warns.append("GPT equity options are enabled but equity_trade_engine is not GPT; equity entries will use indicators instead.")
    except Exception:
        pass
    return warns


# =============================================================================
# RealTradingGate — comprehensive real-trading safety gate
# =============================================================================


@dataclass
class RealTradingGate:
    """All conditions that must pass before real trading is allowed.

    Every field is a safety check. The system is LIVE only when ALL of them
    are satisfied (or explicitly overridden with safe values).
    """

    # ── Master switches ──────────────────────────────────────────────────────
    enable_live_trading: bool = False
    scalper_allow_live_orders: bool = False   # SCALPER_ALLOW_LIVE_ORDERS=true
    scalper_real_trading_ack: bool = False    # SCALPER_REAL_TRADING_ACK=true

    # ── Broker credential health ─────────────────────────────────────────────
    broker_token_valid: bool = True
    broker_token_expiring: bool = True        # True = invalid/expiring (blocks)

    # ── Order infrastructure ─────────────────────────────────────────────────
    order_polling_available: bool = False

    # ── Kill switches ────────────────────────────────────────────────────────
    kill_switch_active: bool = True           # True = kill switch ON (blocks)

    # ── Readiness from prior audit reports ──────────────────────────────────
    paper_readiness_pass: bool = False
    broker_safety_pass: bool = False

    # ── Feature coverage ─────────────────────────────────────────────────────
    dry_run_coverage_pct: float = 0.0

    # ── Model readiness ──────────────────────────────────────────────────────
    model_pkl_exists: bool = False
    current_probability: float = 0.0
    probability_threshold: float = 0.5

    # ── Entry quality ────────────────────────────────────────────────────────
    spread_pct: float = 999.0
    max_spread_pct: float = 0.02
    premium: float = 0.0
    min_premium: float = 5.0

    # ── Risk clamps ──────────────────────────────────────────────────────────
    max_daily_loss_breached: bool = False
    max_trades_per_day_breached: bool = False

    # ── Market state ─────────────────────────────────────────────────────────
    market_hours_valid: bool = True
    open_stale_position: bool = False


def real_trading_allowed(gate: RealTradingGate) -> tuple[bool, list[str]]:
    """Central gate: returns (allowed, list_of_blockers).

    ALL conditions must pass for real trading to be allowed.
    Add a call to this at every real-order entry point.
    """
    blockers: list[str] = []

    if not gate.enable_live_trading:
        blockers.append("enable_live_trading is False")
    if not gate.scalper_allow_live_orders:
        blockers.append("SCALPER_ALLOW_LIVE_ORDERS must be set to true")
    if not gate.scalper_real_trading_ack:
        blockers.append("SCALPER_REAL_TRADING_ACK must be set to true")
    if gate.broker_token_expiring:
        blockers.append("broker token is missing, expiring, or expired")
    if not gate.broker_token_valid:
        blockers.append("broker token is not valid")
    if not gate.order_polling_available:
        blockers.append("order status polling not available")
    if gate.kill_switch_active:
        blockers.append("kill switch is active")
    if not gate.paper_readiness_pass:
        blockers.append("paper readiness not PASS")
    if not gate.broker_safety_pass:
        blockers.append("broker safety checks have not passed")
    if gate.dry_run_coverage_pct < 95.0:
        blockers.append(f"dry-run coverage {gate.dry_run_coverage_pct:.1f}% < 95%")
    if not gate.model_pkl_exists:
        blockers.append("model_pkl is null or missing")
    if gate.current_probability < gate.probability_threshold:
        blockers.append(
            f"probability {gate.current_probability:.4f} < threshold {gate.probability_threshold:.4f}"
        )
    if gate.spread_pct > gate.max_spread_pct:
        blockers.append(
            f"spread {gate.spread_pct*100:.2f}% > max {gate.max_spread_pct*100:.2f}%"
        )
    if gate.premium < gate.min_premium:
        blockers.append(f"premium {gate.premium:.2f} < min {gate.min_premium:.2f}")
    if gate.max_daily_loss_breached:
        blockers.append("max daily loss breach detected")
    if gate.max_trades_per_day_breached:
        blockers.append("max trades per day exceeded")
    if not gate.market_hours_valid:
        blockers.append("outside valid market hours")
    if gate.open_stale_position:
        blockers.append("open stale position from previous session")

    return (len(blockers) == 0, blockers)


# =============================================================================
# micro_live_allowed — lightweight live trading for tiny positions
# =============================================================================


def micro_live_allowed(
    cfg: Any,
    *,
    probability: float,
    spread_pct: float,
    premium: float,
    trades_today: int,
    daily_pnl: float,
    kill_switch: bool,
) -> tuple[bool, list[str]]:
    """Evaluate micro-live mode safety gate.

    Micro-live is a restricted subset of full live trading with tighter
    constraints: 1 trade/day, 1 lot, 70%+ probability, spread <= 1%, premium >= Rs10.
    """
    blockers: list[str] = []

    enable_micro_live = bool(getattr(cfg, "enable_micro_live", False))
    if not enable_micro_live:
        blockers.append("enable_micro_live is False")

    strict_kill = bool(getattr(cfg, "micro_live_strict_kill_switch", True))
    if kill_switch and strict_kill:
        blockers.append("kill switch is active")

    min_prob = float(getattr(cfg, "micro_live_min_probability", 0.70))
    if probability < min_prob:
        blockers.append(f"probability {probability:.4f} < micro_live min {min_prob:.4f}")

    max_spread = float(getattr(cfg, "micro_live_require_spread_pct", 0.01))
    if spread_pct > max_spread:
        blockers.append(f"spread {spread_pct*100:.2f}% > micro_live max {max_spread*100:.2f}%")

    min_premium = float(getattr(cfg, "micro_live_require_min_premium", 10.0))
    if premium < min_premium:
        blockers.append(f"premium {premium:.2f} < micro_live min {min_premium:.2f}")

    max_trades = int(getattr(cfg, "micro_live_max_trades_per_day", 1))
    if trades_today >= max_trades:
        blockers.append(f"micro_live trades today {trades_today} >= max {max_trades}")

    max_loss = float(getattr(cfg, "micro_live_max_daily_loss", 500.0))
    if daily_pnl <= -max_loss:
        blockers.append(f"daily P&L Rs{daily_pnl:.2f} exceeds micro_live max loss Rs{max_loss:.2f}")

    return (len(blockers) == 0, blockers)


# =============================================================================
# LiveTradeGate — the 8+ explicit gates required before any real broker order
# =============================================================================

from dataclasses import dataclass as _dc


@_dc
class LiveTradeGate:
    """All conditions that must be simultaneously true for a real 1-lot (or scaled) live order."""

    # Master env combination (new names per task spec)
    live_mode: bool = False                    # LIVE_MODE=true
    order_placement_enabled: bool = False      # ORDER_PLACEMENT_ENABLED=true
    live_order_dry_run: bool = True            # LIVE_ORDER_DRY_RUN must be false for real orders
    kill_switch_active: bool = True            # SCALPER_KILL_SWITCH != 1

    # Candidate lifecycle / whitelist
    candidate_live_whitelisted: bool = False
    candidate_status: str = "DISABLED"         # must be LIVE_1_LOT or LIVE_SCALED

    # Broker / session
    broker_session_valid: bool = False

    # Market data freshness (caller supplies)
    option_chain_fresh: bool = False
    selected_expiry_valid: bool = False
    current_time_le_entry_cutoff: bool = False

    # Quote quality
    spread_pct: float = 999.0
    max_spread_pct: float = 0.02
    premium: float = 0.0
    min_premium: float = 5.0
    max_premium: float = 1e9

    # Trade construction rules
    stop_loss_exists: bool = False
    exit_rule_exists: bool = False

    # Risk / position limits (caller supplies current state)
    open_positions: int = 0
    max_open_positions: int = 6
    trades_today: int = 0
    max_trades_per_day: int = 2
    daily_loss: float = 0.0
    max_daily_loss: float = 1000.0
    duplicate_open_position: bool = False

    # First-live rule
    order_type_is_limit: bool = True           # LIMIT required for first live (1-lot) phase


def live_trade_allowed(gate: LiveTradeGate) -> Tuple[bool, List[str]]:
    """
    Central live-trade gate. Returns (allowed, list_of_blockers).

    This is the explicit matrix required by the promotion pipeline spec.
    It is intended to be called from mstock_client.place_order (and any other
    real-order entry point) in addition to the existing RealTradingGate.
    """
    blockers: List[str] = []

    if not gate.live_mode:
        blockers.append("LIVE_MODE is not true")
    if not gate.order_placement_enabled:
        blockers.append("ORDER_PLACEMENT_ENABLED is not true")
    if gate.live_order_dry_run:
        blockers.append("LIVE_ORDER_DRY_RUN must be false for real orders")
    if gate.kill_switch_active:
        blockers.append("SCALPER_KILL_SWITCH is active (or set to 1/true)")

    if not gate.candidate_live_whitelisted:
        blockers.append("candidate.live_whitelisted is false")
    if gate.candidate_status not in ("LIVE_1_LOT", "LIVE_SCALED"):
        blockers.append(f"candidate.status={gate.candidate_status} not in [LIVE_1_LOT, LIVE_SCALED]")

    if not gate.broker_session_valid:
        blockers.append("broker_session_valid is false")

    if not gate.option_chain_fresh:
        blockers.append("option_chain_fresh is false (stale chain)")
    if not gate.selected_expiry_valid:
        blockers.append("selected_expiry_valid is false")
    if not gate.current_time_le_entry_cutoff:
        blockers.append("current_time after entry cutoff")

    if gate.spread_pct > gate.max_spread_pct:
        blockers.append(f"spread_pct {gate.spread_pct:.4f} > max {gate.max_spread_pct:.4f}")
    if gate.premium < gate.min_premium:
        blockers.append(f"premium {gate.premium:.2f} < min {gate.min_premium:.2f}")
    if gate.premium > gate.max_premium:
        blockers.append(f"premium {gate.premium:.2f} > max {gate.max_premium:.2f}")

    if not gate.stop_loss_exists:
        blockers.append("stop_loss rule missing")
    if not gate.exit_rule_exists:
        blockers.append("exit_rule missing")

    if gate.open_positions >= gate.max_open_positions:
        blockers.append(f"open_positions {gate.open_positions} >= max {gate.max_open_positions}")
    if gate.trades_today >= gate.max_trades_per_day:
        blockers.append(f"trades_today {gate.trades_today} >= max {gate.max_trades_per_day}")
    if gate.daily_loss <= -gate.max_daily_loss:
        blockers.append(f"daily_loss {gate.daily_loss:.2f} <= -max_daily_loss {gate.max_daily_loss:.2f}")
    if gate.duplicate_open_position:
        blockers.append("duplicate open position for same leg")

    # First-live rule
    if not gate.order_type_is_limit:
        blockers.append("order_type must be LIMIT for first live (1-lot) phase")

    return (len(blockers) == 0, blockers)


def load_live_trade_gate_from_env(
    *,
    candidate_whitelisted: bool,
    candidate_status: str,
    broker_session_ok: bool,
    chain_fresh: bool,
    expiry_ok: bool,
    before_cutoff: bool,
    spread: float,
    premium_val: float,
    has_sl: bool,
    has_exit: bool,
    open_pos: int,
    trades_today: int,
    daily_pnl: float,
    is_duplicate: bool,
    order_type_limit: bool = True,
    max_open: int = 6,
    max_trades: int = 2,
    max_loss: float = 1000.0,
    max_spread: float = 0.02,
    min_prem: float = 5.0,
) -> LiveTradeGate:
    """Convenience: build a LiveTradeGate from the four new env vars + runtime state."""
    from .candidate_lifecycle import (  # local import to avoid cycles at module load
        get_live_mode_env,
        get_order_placement_enabled_env,
        get_live_order_dry_run_env,
        get_kill_switch_active,
    )

    return LiveTradeGate(
        live_mode=get_live_mode_env(),
        order_placement_enabled=get_order_placement_enabled_env(),
        live_order_dry_run=get_live_order_dry_run_env(),
        kill_switch_active=get_kill_switch_active(),
        candidate_live_whitelisted=candidate_whitelisted,
        candidate_status=candidate_status,
        broker_session_valid=broker_session_ok,
        option_chain_fresh=chain_fresh,
        selected_expiry_valid=expiry_ok,
        current_time_le_entry_cutoff=before_cutoff,
        spread_pct=spread,
        max_spread_pct=max_spread,
        premium=premium_val,
        min_premium=min_prem,
        stop_loss_exists=has_sl,
        exit_rule_exists=has_exit,
        open_positions=open_pos,
        max_open_positions=max_open,
        trades_today=trades_today,
        max_trades_per_day=max_trades,
        daily_loss=daily_pnl,
        max_daily_loss=max_loss,
        duplicate_open_position=is_duplicate,
        order_type_is_limit=order_type_limit,
    )

