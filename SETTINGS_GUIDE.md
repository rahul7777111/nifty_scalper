# Scalper Bot Settings Guide

This document explains each configuration option available in the Scalper UI **Settings** dialog. These settings control the trading logic, risk management, and filtering criteria of the bot.

---

## 1. General Settings

| Setting | Env Variable | Description |
| :--- | :--- | :--- |
| **Strategy** | `MSTOCK_STRATEGY` | The core trading logic to use. Options include:<br>• `auto`: Dynamically switches between directional and range-bound.<br>• `directional`: Uses EMA/RSI/Candle patterns for buying calls/puts.<br>• `short_straddle`, `short_strangle`, `iron_condor`: Specific multi-leg selling strategies.<br>• `bull_call_spread`, `bull_put_spread`: Vertical spreads (bullish).<br>• `call_ratio_backspread`, `put_ratio_backspread`: Ratio backspreads (convex).<br>• `long_straddle`, `long_strangle`: Specific multi-leg buying strategies.<br>• `delta_hedged_short_straddle`, `delta_hedged_long_straddle`: Straddles with dynamic delta hedging using a configured hedge instrument. |
| **Preset** | `MSTOCK_PRESET` | Optional bundle of defaults (does not guarantee profits). Supported values:<br>• `conservative`: fewer trades, longer cooldowns, tighter guards<br>• `aggressive`: more trades, shorter cooldowns, allows some pyramiding |
| **Symbol** | `MSTOCK_SYMBOL` | The trading symbol root used for contract selection (e.g., `NIFTY`). |
| **Underlying** | `MSTOCK_UNDERLYING` | The underlying index/stock name used for candle data fetching (e.g., `NIFTY`). |
| **Underlying token (candles)** | `MSTOCK_UNDERLYING_TOKEN` | Numeric token used by m.Stock chart endpoints for the underlying. If blank, the bot may auto-resolve it via the instruments master (first run can be slower). |
| **Underlying exchange** | `MSTOCK_UNDERLYING_EXCHANGE` | Exchange for underlying token resolution / chart calls (defaults to `NSE`). |
| **Timeframe** | `MSTOCK_TIMEFRAME` | The candle interval for technical indicators (e.g., `1m`, `5m`). |
| **Lot Size** | `MSTOCK_LOT_SIZE` | The number of units per contract (e.g., 65 for NIFTY). |

---

## 2. Risk Management

| Setting | Env Variable | Description |
| :--- | :--- | :--- |
| **Max daily loss** | `MSTOCK_MAX_DAILY_LOSS` | The bot stops trading for the day if the total realized loss exceeds this value. |
| **Max open positions** | `MSTOCK_MAX_OPEN_POSITIONS` | Maximum number of active trades allowed at any given time. |
| **Max trades/day** | `MSTOCK_MAX_TRADES_PER_DAY` | Total number of entry events allowed per day. |
| **Max stopouts/day** | `MSTOCK_MAX_STOPOUTS_PER_DAY` | Stop trading if you hit this many losing trades in a day. |
| **Max consecutive stopouts** | `MSTOCK_MAX_CONSECUTIVE_STOPOUTS` | Stop trading if you hit this many losing trades in a row. |
| **Cooldown (sec)** | `MSTOCK_COOLDOWN_SEC` | Minimum wait time between closing one trade and opening the next. |
| **Cooldown after stopout (sec)** | `MSTOCK_COOLDOWN_AFTER_STOPOUT_SEC` | Extra cooldown applied after a stopout to avoid immediate re-entry. |
| **Adaptive cooldown (ATR mult)** | `MSTOCK_COOLDOWN_ATR_MULT` | Adds `ATR * mult` seconds to the base cooldown. |
| **Max hold (min)** | `MSTOCK_MAX_HOLD_MINUTES` | Force close any trade if it stays open longer than this duration. |

---

## 2.1 Dynamic Risk Scaling

| Setting | Env Variable | Description |
| :--- | :--- | :--- |
| **Risk scaling enabled** | `MSTOCK_RISK_SCALE_ENABLED` | Enable dynamic quantity scaling after stopouts or ATR spikes. |
| **Stopout factor** | `MSTOCK_RISK_SCALE_STOPOUT_FACTOR` | Multiplier applied per consecutive stopout (e.g., 0.7). |
| **Recovery wins** | `MSTOCK_RISK_SCALE_RECOVERY_WINS` | Number of consecutive wins required to restore full size. |
| **ATR high** | `MSTOCK_RISK_SCALE_ATR_HIGH` | If ATR >= this value, apply the ATR scaling factor. |
| **ATR high factor** | `MSTOCK_RISK_SCALE_ATR_HIGH_FACTOR` | Multiplier applied when ATR is above the high threshold. |
| **Min qty** | `MSTOCK_RISK_SCALE_MIN_QTY` | Minimum order quantity after scaling (0 uses lot step). |
| **Max qty** | `MSTOCK_RISK_SCALE_MAX_QTY` | Maximum order quantity after scaling (0 uses base lot size). |
| **Step qty** | `MSTOCK_RISK_SCALE_STEP_QTY` | Round scaled quantities to this step (0 uses lot size). |

## 3. Technical Indicators (ATR / EMA)

| Setting | Env Variable | Description |
| :--- | :--- | :--- |
| **ATR Period** | `MSTOCK_ATR_PERIOD` | Lookback period for Average True Range calculation (default 14). |
| **Min / Max ATR** | `MSTOCK_MIN_ATR` / `MSTOCK_MAX_ATR` | Only allow trades if the current volatility (ATR) is within this range. |
| **ATR stop x** | `MSTOCK_ATR_STOP_MULT` | Stop loss distance in multiples of ATR (e.g., 1.0 means stop is at 1.0 * ATR from entry). |
| **ATR target x** | `MSTOCK_ATR_TARGET_MULT` | Profit target distance in multiples of ATR. |
| **EMA Fast / Slow** | `MSTOCK_EMA_FAST` / `MSTOCK_EMA_SLOW` | Periods for the trend-following EMAs (default 9 and 21). |
| **EMA slope (ATR mult)** | `MSTOCK_DIR_EMA_SLOPE_ATR_MULT` | Adds a directional vote only when EMA slope exceeds this ATR multiple. |
| **Max stale spot LTP (sec)** | `MSTOCK_MAX_STALE_LTP_SEC` | Blocks *new entries* when the spot LTP feed looks stale (age in seconds). Set `0` to disable. |
| **Candle max age (sec)** | `MSTOCK_ENTRY_CANDLE_MAX_AGE_SEC` | Skip entries if the latest candle is older than this threshold. |
| **Max candle range (ATR mult)** | `MSTOCK_ENTRY_MAX_CANDLE_RANGE_ATR_MULT` | Skip entries when the latest candle range exceeds this ATR multiple. |
| **Gap filter (ATR mult)** | `MSTOCK_ENTRY_GAP_ATR_MULT` | Skip entries when the last open vs prior close gap exceeds this ATR multiple. |

---

## 3.1 Session Regimes

| Setting | Env Variable | Description |
| :--- | :--- | :--- |
| **Open window** | `MSTOCK_SESSION_OPEN_START` / `MSTOCK_SESSION_OPEN_END` | Session window for market open (HH:MM, IST). |
| **Close window** | `MSTOCK_SESSION_CLOSE_START` / `MSTOCK_SESSION_CLOSE_END` | Session window for market close (HH:MM, IST). |
| **Open ATR min/max** | `MSTOCK_SESSION_OPEN_MIN_ATR` / `MSTOCK_SESSION_OPEN_MAX_ATR` | Override ATR bounds during open session. |
| **Mid ATR min/max** | `MSTOCK_SESSION_MID_MIN_ATR` / `MSTOCK_SESSION_MID_MAX_ATR` | Override ATR bounds during mid session. |
| **Close ATR min/max** | `MSTOCK_SESSION_CLOSE_MIN_ATR` / `MSTOCK_SESSION_CLOSE_MAX_ATR` | Override ATR bounds during close session. |
| **Open RSI band** | `MSTOCK_SESSION_OPEN_PREMIUM_RSI_LOW` / `MSTOCK_SESSION_OPEN_PREMIUM_RSI_HIGH` | Override premium RSI band during open session. |
| **Mid RSI band** | `MSTOCK_SESSION_MID_PREMIUM_RSI_LOW` / `MSTOCK_SESSION_MID_PREMIUM_RSI_HIGH` | Override premium RSI band during mid session. |
| **Close RSI band** | `MSTOCK_SESSION_CLOSE_PREMIUM_RSI_LOW` / `MSTOCK_SESSION_CLOSE_PREMIUM_RSI_HIGH` | Override premium RSI band during close session. |

## 4. Multi-Leg / Premium Selling

| Setting | Env Variable | Description |
| :--- | :--- | :--- |
| **Short strike distance** | `MSTOCK_SHORT_STRIKE_DISTANCE` | Points away from spot to select OTM strikes for strangles/condors. |
| **Straddle ATR threshold** | `MSTOCK_STRADDLE_ATR_THRESHOLD` | If ATR is below this, the bot prefers straddles; otherwise, strangles. |
| **Wing width** | `MSTOCK_WING_WIDTH` | Distance for hedge legs in Iron Condor structures. |
| **MTM stop %** | `MSTOCK_PREMIUM_MTM_STOP_PCT` | Exit if the total premium of a short position increases by this % (loss). |
| **MTM target %** | `MSTOCK_PREMIUM_MTM_TARGET_PCT` | Exit if the total premium decays by this % (profit). |
| **Exit on short/wing touch** | `MSTOCK_PREMIUM_EXIT...` | Stop out immediately if the spot price touches the sold strike or hedge wing. |
| **Min/Max option premium** | `MSTOCK_ENTRY_MIN_OPTION_PREMIUM` / `MSTOCK_ENTRY_MAX_OPTION_PREMIUM` | Reject entries when any leg premium is outside these bounds. |
| **Min/Max total premium** | `MSTOCK_ENTRY_MIN_TOTAL_PREMIUM` / `MSTOCK_ENTRY_MAX_TOTAL_PREMIUM` | Reject entries when total absolute premium across legs is outside bounds. |

---

## 4.1 Delta Hedging

These settings are used by delta-hedged strategies (e.g. `delta_hedged_short_straddle`). You must configure a tradable hedge symbol.

You can also enable **dynamic delta hedging for all option trades** (including single-leg directional option trades) via `MSTOCK_DELTA_HEDGE_SCOPE`.

| Setting | Env Variable | Description |
| :--- | :--- | :--- |
| **Hedging scope** | `MSTOCK_DELTA_HEDGE_SCOPE` | Controls which trades are delta-hedged:<br>• `strategy_only` (default): only strategies that explicitly enable hedging (delta-hedged straddles).<br>• `multi_only`: hedge all multi-leg option trades (straddles/strangles/condors).<br>• `all_options`: hedge both multi-leg and directional option trades. |
| **Hedge symbol (NIFTY)** | `MSTOCK_DELTA_HEDGE_SYMBOL_NIFTY` | Optional per-underlying hedge symbol used for NIFTY option trades (takes precedence over `MSTOCK_DELTA_HEDGE_SYMBOL` when the trade underlying is inferred). |
| **Hedge symbol (BANKNIFTY)** | `MSTOCK_DELTA_HEDGE_SYMBOL_BANKNIFTY` | Optional per-underlying hedge symbol used for BANKNIFTY option trades (takes precedence over `MSTOCK_DELTA_HEDGE_SYMBOL` when the trade underlying is inferred). |
| **Hedge symbol** | `MSTOCK_DELTA_HEDGE_SYMBOL` | Tradable hedge instrument. Accepts `EXCH:TRADINGSYMBOL` (preferred) or plain symbol if resolvable via ScripMaster/instruments. |
| **Hedge exchange (hint)** | `MSTOCK_DELTA_HEDGE_EXCHANGE` | Optional exchange hint when hedge symbol is plain (e.g. `NSE`, `NFO`). |
| **Use options for hedging** | `MSTOCK_DELTA_HEDGE_USE_OPTIONS` | If true, hedge with options (CE/PE) instead of the underlying instrument. |
| **Option hedge strike offset** | `MSTOCK_DELTA_HEDGE_OPTION_STRIKE_OFFSET` | Strike offset from spot for option hedges (0 = ATM). Calls use `spot + offset`, puts use `spot - offset`. |
| **Option hedge sell (short)** | `MSTOCK_DELTA_HEDGE_OPTION_SELL` | If true, hedge using short options instead of long-only. |
| **Option hedge auto-unwind** | `MSTOCK_DELTA_HEDGE_OPTION_AUTO_UNWIND` | If true, closes hedge option legs that don’t match the current hedge side/type. |
| **Delta tolerance** | `MSTOCK_DELTA_HEDGE_TOLERANCE` | Minimum absolute residual delta (underlying units) before rebalancing. |
| **Entry tolerance** | `MSTOCK_DELTA_HEDGE_ENTRY_TOLERANCE` | Hysteresis entry threshold (rebalancing starts above this residual delta). |
| **Exit tolerance** | `MSTOCK_DELTA_HEDGE_EXIT_TOLERANCE` | Hysteresis exit threshold (stop rebalancing below this residual delta). |
| **Rebalance interval (sec)** | `MSTOCK_DELTA_HEDGE_REBALANCE_SEC` | Minimum seconds between hedge adjustments. |
| **Adjustment factor** | `MSTOCK_DELTA_HEDGE_ADJUSTMENT_FACTOR` | 0–1.0 fraction of residual delta to hedge per rebalance (1.0 = full). |
| **Include broker positions** | `MSTOCK_DELTA_HEDGE_INCLUDE_POSITIONS` | If true, includes broker-reported net position in the hedge symbol to avoid double-hedging. |
| **Include holdings** | `MSTOCK_DELTA_HEDGE_INCLUDE_HOLDINGS` | If true, includes delivery holdings for the hedge symbol (useful for ETF hedges). |
| **Include equity beta** | `MSTOCK_DELTA_HEDGE_INCLUDE_EQUITY_BETA` | If true, estimates holdings beta vs the benchmark and adds an equity-portfolio delta to the hedge calculation. |
| **Portfolio hedge enable** | `MSTOCK_DELTA_HEDGE_PORTFOLIO_HEDGE_ENABLE` | If true, runs a portfolio hedge loop even when there are no option trades with hedging enabled. |
| **Min spot move** | `MSTOCK_DELTA_HEDGE_MIN_SPOT_MOVE` | Minimum absolute spot move required before re-hedging. |
| **Min spot move (ATR mult)** | `MSTOCK_DELTA_HEDGE_MIN_SPOT_MOVE_ATR_MULT` | Spot move threshold as ATR multiple (max with absolute threshold). |
| **Qty step** | `MSTOCK_DELTA_HEDGE_STEP_QTY` | Round hedge order quantities to this step (often the contract lot size). `0` defaults to `MSTOCK_LOT_SIZE`. |
| **Max abs qty** | `MSTOCK_DELTA_HEDGE_MAX_ABS_QTY` | Safety clamp for hedge position size in order-quantity units. `0` defaults to `10 * MSTOCK_DELTA_HEDGE_STEP_QTY`. |
| **Max adjust qty** | `MSTOCK_DELTA_HEDGE_MAX_ADJUST_ABS_QTY` | Maximum change in hedge qty per rebalance (limits large jumps). |
| **Max hedge orders/day** | `MSTOCK_DELTA_HEDGE_MAX_ORDERS_PER_DAY` | Hard cap on number of hedge orders per day. |
| **Require bid/ask** | `MSTOCK_DELTA_HEDGE_REQUIRE_BID_ASK` | If true, skip hedging when bid/ask is unavailable. |
| **Max spread %** | `MSTOCK_DELTA_HEDGE_MAX_SPREAD_PCT` | Skip hedging if bid/ask spread exceeds this percent of mid. |
| **Max spread abs** | `MSTOCK_DELTA_HEDGE_MAX_SPREAD_ABS` | Skip hedging if bid/ask spread exceeds this absolute value. |
| **Market hours only** | `MSTOCK_DELTA_HEDGE_MARKET_HOURS_ONLY` | Skip hedging outside market hours in live mode. |
| **Assumed IV** | `MSTOCK_DELTA_HEDGE_ASSUMED_IV` | Used for delta estimation when IV is not available in the chain (common with CSV chains). |
| **Rate** | `MSTOCK_DELTA_HEDGE_RATE` | Risk-free rate used for Black–Scholes delta estimation. |

### How it is implemented

- Each option trade is represented as one or more **legs** (CE/PE) with `strike`, `expiry`, `side`, and `quantity`.
- On each management cycle, the bot computes the **net option delta** by summing per-leg Black–Scholes deltas (from the existing Greeks module) and multiplying by signed quantity (BUY = +, SELL = −).
- If the absolute residual delta is above `MSTOCK_DELTA_HEDGE_TOLERANCE` and at least `MSTOCK_DELTA_HEDGE_REBALANCE_SEC` seconds have elapsed, the bot places a hedge order in the configured hedge instrument (`MSTOCK_DELTA_HEDGE_SYMBOL`).
- Hedge quantity is **rounded** to `MSTOCK_DELTA_HEDGE_STEP_QTY` and **clamped** by `MSTOCK_DELTA_HEDGE_MAX_ABS_QTY` to reduce churn and prevent runaway sizing.
- When `MSTOCK_DELTA_HEDGE_USE_OPTIONS=true`, the bot uses CE/PE options to offset residual delta, using the nearest strike to `spot ± MSTOCK_DELTA_HEDGE_OPTION_STRIKE_OFFSET`. If `MSTOCK_DELTA_HEDGE_OPTION_SELL=true`, it uses short options instead of long-only.
- Hedge orders are tracked as additional legs with `is_hedge=true`, so MTM and close logic can include them.

### Advantages of implementing it here

- **Reduces directional exposure** for non-directional premium strategies (short straddles/strangles/condors), helping stabilize MTM during trends.
- **Improves risk control** by automatically re-centering delta as spot moves (especially useful around fast intraday moves).
- **Modular integration**: rebalancing runs inside the existing “manage open trades” loop, so it works consistently across strategies without adding a separate scheduler.
- **Practical safeguards**: tolerance + time cadence + step rounding + max clamp reduce over-trading and slippage.

---

## 4.2 GPT (Optional)

These settings enable an LLM to participate in decisions. The bot still enforces its own safety clamps (step size, max qty, max orders/day, etc.).

| Setting | Env Variable | Description |
| :--- | :--- | :--- |
| **Enable GPT** | `MSTOCK_GPT_ENABLE` | Master toggle for GPT features. |
| **Apply in paper mode** | `MSTOCK_GPT_APPLY_PAPER` | If `true`, GPT is used even when `MSTOCK_ENABLE_LIVE_TRADING=false` (recommended for dry-runs). |
| **GPT controls delta hedging** | `MSTOCK_GPT_CONTROL_DELTA_HEDGE` | If `true`, GPT can SKIP or approve delta-hedge rebalances and optionally override hedge sizing (still clamped). |
| **GPT controls equity entries** | `MSTOCK_GPT_CONTROL_EQUITY` | If `true`, GPT can SKIP/approve equity entries on holdings and choose `BUY`/`SELL` direction (shorts still respect `MSTOCK_EQUITY_TRADE_ALLOW_SHORT`). |
| **API key (not persisted)** | `MSTOCK_GPT_API_KEY` / `OPENAI_API_KEY` | API key is read from env only; it is not saved by the UI. |

---

## 4.3 Equity Trading (Holdings)

These settings control the optional **equity trading on your holdings universe**. You can run it as a simple indicator-based engine or switch to a GPT-driven decision engine.

| Setting | Env Variable | Description |
| :--- | :--- | :--- |
| **Enable equity trading** | `MSTOCK_EQUITY_TRADE_ENABLE` | Master toggle for equity trading on holdings. |
| **Engine** | `MSTOCK_EQUITY_TRADE_ENGINE` | `INDICATORS` (legacy) or `GPT` (model decides BUY/SELL/HOLD). |
| **Horizon** | `MSTOCK_EQUITY_TRADE_HORIZON` | `INTRADAY` or `LONGTERM`. LONGTERM enables quarter-style target logic. |
| **Portfolio churn** | `MSTOCK_EQUITY_TRADE_CHURN_ENABLE` | If true (and engine is GPT), the bot may ask GPT whether to exit existing equity positions. |
| **Longterm target %** | `MSTOCK_EQUITY_LONGTERM_TARGET_PCT` | Profit target for LONGTERM mode, expressed as a percent (e.g. `8` for +8%). |
| **Longterm horizon days** | `MSTOCK_EQUITY_LONGTERM_HORIZON_DAYS` | Holding horizon for LONGTERM mode (default 90 days). |
| **Product type** | `MSTOCK_EQUITY_TRADE_PRODUCT_TYPE` | Broker product type (commonly `INTRADAY`, `CNC`, or `DELIVERY`). |
| **Allow short** | `MSTOCK_EQUITY_TRADE_ALLOW_SHORT` | If true, allow short-selling in INTRADAY equity trades (GPT engine respects this). |

## 5. Entry Filters

| Setting | Env Variable | Description |
| :--- | :--- | :--- |
| **Max trend strength** | `MSTOCK_MAX_TREND_STRENGTH` | Prevents range-bound entries if the EMA spread is too wide (trending market). |
| **Premium RSI Low / High** | `MSTOCK_PREMIUM_RSI_...` | Only allow premium-selling entries if RSI is within this range (overbought/oversold filter). |
| **VWAP max dev %** | `MSTOCK_VWAP_MAX_DEV_PCT` | Distance from VWAP allowed for entries. |
| **Opening max move %** | `MSTOCK_MAX_OPEN_MOVE_PCT` | Prevents trading if the market opened with a massive gap or spike. |
| **Opening window (min)** | `MSTOCK_OPENING_FILTER_MINUTES` | The duration after market open during which the opening filter is active. |
| **Require bid/ask** | `MSTOCK_ENTRY_REQUIRE_BID_ASK` | If true, reject entries when best bid/ask is unavailable. |
| **Max bid/ask spread %** | `MSTOCK_ENTRY_MAX_BID_ASK_SPREAD_PCT` | Reject entries if bid/ask spread exceeds this percent of mid price. |
| **Max bid/ask spread abs** | `MSTOCK_ENTRY_MAX_BID_ASK_SPREAD_ABS` | Reject entries if bid/ask spread exceeds this absolute value. |
| **IV expand threshold %** | `MSTOCK_IV_EXPAND_THRESHOLD_PCT` | Block short premium entries when IV is expanding faster than this percent. |
| **IV contract threshold %** | `MSTOCK_IV_CONTRACT_THRESHOLD_PCT` | Block long premium entries when IV is collapsing faster than this percent. |
| **Max same-direction qty** | `MSTOCK_ENTRY_MAX_SAME_DIRECTION_QTY` | Block directional entries if existing directional exposure exceeds this qty. |

---

## 6. Directional (Long) Tweaks

| Setting | Env Variable | Description |
| :--- | :--- | :--- |
| **Dir BE x ATR** | `MSTOCK_DIR_BREAKEVEN_ATR_MULT` | Move stop loss to breakeven once price moves favorably by this much. |
| **Dir Trail x ATR** | `MSTOCK_DIR_TRAIL_ATR_MULT` | Trail the stop loss at this distance from the highest favorable spot. |
| **Dir Prem Trail %** | `MSTOCK_DIR_PREMIUM_TRAIL_PCT` | Exit if the option premium drops by this % from its peak. |
| **Dir Partial Target / Qty** | `MSTOCK_DIR_PARTIAL_...` | Book a percentage of quantity at a specific ATR target. |
| **Post-partial BE x ATR** | `MSTOCK_DIR_POST_PARTIAL_BE_ATR_MULT` | After a partial exit, tighten breakeven to this ATR multiple. |
| **Post-partial Trail x ATR** | `MSTOCK_DIR_POST_PARTIAL_TRAIL_ATR_MULT` | After a partial exit, tighten trailing stop to this ATR multiple. |
| **Spot LTP Req** | `MSTOCK_REQUIRE_SPOT_LTP_FOR_ENTRY`| If true, prevents entries if the spot price feed is stale/down. |
| **Flip long->short on stop** | `MSTOCK_DIR_FLIP_LONG_TO_SHORT_ON_STOP` | If enabled: when a `long_call/long_put` hits stop/trail stop, it closes and re-enters as `short_put/short_call` (higher risk). |
| **Dir Min Confirmations** | `MSTOCK_DIR_MIN_CONFIRMATIONS` | Minimum directional score required to enter (higher = fewer, higher-quality trades). |
| **Dir Min Score Diff** | `MSTOCK_DIR_MIN_SCORE_DIFF` | Minimum score advantage required (winner − loser). Helps avoid noisy flip-flops. |
| **Dir Momentum xATR** | `MSTOCK_DIR_MOMENTUM_ATR_MULT` | Threshold (as ATR multiple) to count the momentum vote in directional scoring. |
| **Dir Allow Tiebreak** | `MSTOCK_DIR_ALLOW_TIEBREAK` | If true, allows tie-break entries when bull and bear scores are equal (more trades, more churn). |

### 6.2 Directional Entry Quality (Reduce Turnover)

These settings directly control **how selective** the directional entry logic is. If your turnover is high compared to profits, increase selectivity.

| Setting | Env Variable | Description |
| :--- | :--- | :--- |
| **Min confirmations** | `MSTOCK_DIR_MIN_CONFIRMATIONS` | Minimum directional score required to enter. Higher = fewer trades. Default: 4. |
| **Min score diff** | `MSTOCK_DIR_MIN_SCORE_DIFF` | Minimum score advantage required (winner − loser). Default: 1 (blocks ties). |
| **Momentum ATR mult** | `MSTOCK_DIR_MOMENTUM_ATR_MULT` | Controls the spot-momentum vote threshold: requires move ≥ `mult * ATR` over lookback. Default: 0.10. |
| **Allow tiebreak** | `MSTOCK_DIR_ALLOW_TIEBREAK` | If true, allows tie-break entries when bull and bear scores are equal. Default: false. |

---

## 6.1 Supertrend Mode

| Setting | Env Variable | Description |
| :--- | :--- | :--- |
| **Supertrend mode** | `MSTOCK_SUPERTREND_MODE` | `trend` = take trades with Supertrend direction, `counter` = take trades opposite Supertrend direction. |

---

## 7. Auto & Expiry Settings

| Setting | Env Variable | Description |
| :--- | :--- | :--- |
| **Auto range trend x** | `MSTOCK_AUTO_RANGE_TREND_MULT`| Multiplier to relax trend filter when in `auto` mode. |
| **Auto RSI B/S/slop** | `MSTOCK_AUTO_DIR_RSI_...` | Triggers for directional buys in `auto` mode (Buy RSI, Sell RSI, Tolerance). |
| **Hard exit / Cutoff** | `MSTOCK_PREMIUM_FORCE...` | Time-based cutoffs for entries and exits (IST). |
| **NIFTY weekly only** | `MSTOCK_NIFTY_WEEKLY_ONLY` | If checked, avoids monthly expiry contracts. |
| **Target expiry** | `MSTOCK_TARGET_EXPIRY` | Manually force a specific expiry date (DD-MM-YYYY). |

---

## 8. Debug & Safety

| Setting | Env Variable | Description |
| :--- | :--- | :--- |
| **Max same trade-type streak** | `MSTOCK_MAX_CONSECUTIVE_SAME_TRADE_TYPE` | Blocks opening more than N consecutive trades of the same type (type = `multi/directional` + strategy/name). Set `0` to disable. |
| **ATR spike x** | `MSTOCK_ATR_SPIKE_MULT` | Kill-switch: Close all trades if volatility suddenly spikes (flash crash protection). |
| **Bypass weekly filter**| `MSTOCK_BYPASS_WEEKLY_FILTER`| Allows trading monthly or far-dated expiries for debugging. |
| **No signal log** | `MSTOCK_DEBUG_NO_SIGNAL` | If enabled, logs why a signal was *not* generated every polling interval. |

---

## 9. Testing Commands

| Command | Description |
| :--- | :--- |
| `pytest -q -m real_dataset tests/test_real_dataset_retrain_integration.py` | Run real-dataset integration test (requires ~719MB dataset, 405k rows). NON-default test - skipped in normal runs. |
| `python -m pytest -q` | Run full test suite |
| `pytest -q -m real_dataset tests/test_real_dataset_retrain_integration.py::test_real_dataset_core_retrain_summary_written` | Run specific real-dataset test |
