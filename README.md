# NIFTY-OPTIONS-SCALPER

**Overview**

NIFTY-OPTIONS-SCALPER is a Python-based template and UI for experimenting with intraday options scalping and multi-leg option strategies on the Indian NIFTY index. It combines a strategy engine, a thin broker client wrapper for Mirae's m.Stock Type B API, a rich Tkinter GUI for live monitoring and control, optional GPT-based advisory features, and small utilities for persistence and alerts.

**Key Features**

- **Strategy Engine:** Implements multiple strategy templates (directional scalps, premium-selling structures such as short straddle/strangle, spreads, condors, ratio backspreads, long volatility entries and simple delta-hedged variants). Strategy rules, risk limits, and guardrails are configurable via a StrategyConfig.
- **Broker Adapter:** `MStockTypeBClient` provides a simplified interface to the official Type B SDK (mconnect/MConnectB), candle fetching (intraday + historical fallback), LTP, and order primitives.
- **Graphical UI:** A single-window Tkinter app with tabs for Live Dashboard, Trade Logs, Signals/Greeks, GPT Advisor, and Settings. The UI shows portfolio snapshots, per-leg MTM, P&L breakdown, greeks, and risk diagnostics.
- **GPT Advisor (optional):** Best-effort GPT integration to provide trade-level advice and preset suggestions. Works with OpenAI-compatible endpoints and includes connectivity/health checks and safe parsing.
- **Persistence & Analytics:** Optional SQLite persistence for trades and a pluggable notifier (Telegram) for alerts.
- **Safety & Guardrails:** Hard limits (max daily loss, max stopouts), entry freshness checks, liquidity/spread guards, ATR / VWAP / RSI / ADX filters, session windows, and kill-switch checks.
- **Extensible:** Modular code organization makes it easy to extend strategies, connectors, or add new UI elements.

**Repository Layout**

- `src/main.py` — CLI entry to run the scalper loop headlessly via `NiftyScalper`.
- `src/ui.py` — Full Tkinter GUI, tabbed views, PDF export helpers, and settings persistence.
- `src/strategy.py` — `NiftyScalper` strategy engine, trade state, rules, risk controls, and integration hooks (event sink, on_tick).
- `src/mstock_client.py` — `MStockTypeBClient` wrapper around the Mirae Type B SDK with candle fetching, intraday fallback, and order abstractions.
- `src/config.py` — `APIConfig` and `StrategyConfig` dataclasses and helpers to load/persist configuration.
- `src/gpt_advisor.py` — GPT call helpers, model normalization, and advice parsing.
- `src/db.py` — Optional `DatabaseManager` for SQLite persistence.
- `src/alerts.py` — Notifiers (Telegram) and async message helpers.
- `pytradingapi-typeB-main/` — Vendored vendor SDK examples used when tradingapi_b is not installed system-wide.

**Configuration**

The application uses a mix of environment variables and config files. Important settings:

- `MSTOCK_ACCESS_TOKEN` — Broker access token used by the SDK (if not supplied elsewhere).
- `MSTOCK_API_KEY` / `MSTOCK_GPT_API_KEY` / `OPENAI_API_KEY` — API keys for broker and optional GPT features.
- `MSTOCK_GPT_API_BASE_URL` / `MSTOCK_GPT_MODEL` — GPT endpoint and model selection for advisor features.
- Various `MSTOCK_*` env vars control UI theme, refresh intervals, and debugging toggles. See `src/ui.py` for defaults and available environment flags.

Strategy-specific options are available in `StrategyConfig` (defaults in `src/config.py`). Key knobs include `lot_size`, `max_stopouts_per_day`, `max_hold_minutes`, `premium_mtm_stop_pct`, delta-hedge parameters, and volatility/ATR guardrails.

**Running the App**

- GUI: Run `python -m src.ui` (or `python src/ui.py`) to launch the Tkinter UI. The UI supports interactive login helpers and persistent settings.
- Headless scalper loop: `python -m src.main` will load API and strategy configs, login the `MStockTypeBClient`, instantiate `NiftyScalper`, and call `run_forever()` to start the bot loop.

Ensure the virtual environment includes dependencies from `requirements.txt` (GUI uses `tkinter` from the standard library; optional extras include `reportlab` for PDF export and `requests` for web calls).

**UI Highlights**

- **Live Dashboard:** Per-leg option rows, MTM breakdowns (base vs hedge), portfolio risk summary, traded quantities and amounts.
- **Trade Logs:** Persisted and live-updateable trade events (OPEN/UPDATE/CLOSE) with sortable columns, export to PDF from the UI.
- **Signals/Greeks:** Live greeks table, implied-vol calculations, and common candlestick/indicator signals (EMA, ATR, RSI, ADX, Supertrend, Pivot Points).
- **GPT Advisor Tab:** Configure GPT keys and ask for per-trade advice. Connectivity checks included to surface authentication or endpoint errors.
- **Settings:** Scrollable form for strategy knobs, credential management, and opt-ins for advanced features like GPT and delta-hedging.

**How The Bot Trades**

The NiftyScalper bot operates as an event-driven, rules-based trading system with optional AI assistance. Below is a detailed breakdown of its trading workflow.

### Main Trading Loop (`run_forever`)

The bot runs a continuous loop with the following sequence:

1. **Risk Check**: Before each iteration, verifies that risk limits (daily loss, stopouts, max trades) are not breached.
2. **Market Hours Guard**: Live trading is blocked outside market hours; paper trading continues for testing.
3. **Entry Evaluation**: Once per timeframe bucket (e.g., every 1-minute or 5-minute candle), the bot evaluates entry signals.
4. **Position Management**: Manages open positions (exits, trailing stops, MTM updates).
5. **Equity Trading** (optional): Manages equity positions separately if enabled.
6. **Portfolio Hedging** (optional): Rebalances delta/beta hedges.
7. **Sleep**: Waits for the configured polling interval before the next iteration.

### Entry Logic (`_decide_entries`)

The bot uses a multi-layered approach to decide when to enter trades:

#### 1. Pre-Entry Guards
- **Entry Pause**: GPT regime monitor can pause entries during unfavorable market conditions.
- **Risk Limits**: Checks max trades per day, cooldown periods, and stopout cooldowns.
- **Time Filters**: Respects entry cutoff times and opening filter (first N minutes of market).
- **Market State**: Verifies sufficient candle history is available.

#### 2. Technical Indicators
The bot computes multiple indicators on the underlying (NIFTY) candles:
- **EMA (Exponential Moving Average)**: Fast (default 9) and Slow (default 21) for trend direction.
- **RSI (Relative Strength Index)**: Default period 14, used for momentum confirmation.
- **ATR (Average True Range)**: Measures volatility for position sizing and filters.
- **ADX (Average Directional Index)**: Optional filter to ensure sufficient trend strength.
- **VWAP**: Volume-weighted average price for intraday bias.
- **Supertrend**: Optional trend confirmation.
- **Pivot Points**: Support/resistance levels.

#### 3. Multi-Timeframe (MTF) Confirmation
When enabled, the bot fetches candles from a higher timeframe (e.g., 5m) and confirms the trend direction matches the entry signal.

#### 4. ML/AI Signal Integration
- **ML Ensemble**: Optional ML model (`src/ml_signals.py`) provides a combined signal score.
- **GPT Entry Gate**: When configured, GPT can approve/reject entries based on market context.

#### 5. Strategy Selection
The bot supports multiple strategy templates, selected via:
- **Explicit Config**: Set `strategy_name` to a specific strategy.
- **Auto Mode**: GPT or regime-based allocator selects the best strategy for current market conditions.
- **Rerouting**: If an entry is blocked (e.g., poor win rate), the bot can automatically try alternative strategies.

### Supported Strategy Templates

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `directional` | Simple bull/bear scalps on underlying | Trending markets |
| `long_straddle` | Buy ATM call + put | High volatility expected |
| `short_straddle` | Sell ATM call + put | Low volatility, range-bound |
| `long_strangle` | Buy OTM call + put | High volatility expected (cheaper) |
| `short_strangle` | Sell OTM call + put | Range-bound market |
| `bull_call_spread` | Buy lower strike call, sell higher | Moderate bullish |
| `bull_put_spread` | Sell higher strike put, buy lower | Moderate bullish |
| `call_ratio_backspread` | Sell 1 ATM call, buy 2 OTM | Strong bullish with hedge |
| `put_ratio_backspread` | Sell 1 ATM put, buy 2 OTM | Strong bearish with hedge |
| `iron_condor` | Combine bull put + bear call spreads | Low volatility range |
| `iron_fly` | Straddle + strangle combo | Neutral with theta decay |
| `iron_butterfly` | Precise neutral strategy | Pin-point neutral |
| `delta_hedged_long_straddle` | Long straddle + delta hedge | Long vol with delta neutral |
| `delta_hedged_short_straddle` | Short straddle + delta hedge | Short vol with delta neutral |

### Exit Management (`_manage_open_trades`)

The bot employs multiple exit mechanisms:

#### 1. Stop Loss & Target
- **Stop Loss**: Based on ATR, fixed percentage, or premium MTM percentage.
- **Target**: Fixed profit target or risk:reward ratio.
- **Bracket Orders**: Optional bracket legs with stop/target prices.

#### 2. MTM-Based Exits
- **Premium MTM Stop**: Exit when unrealized loss exceeds a percentage of entry premium.
- **Trailing MTM**: Trailing stop based on maximum unrealized profit.

#### 3. Time-Based Exits
- **Max Hold Time**: Automatically exit trades after a configured duration.
- **Session Exit**: Option to close all positions before market close.

#### 4. Technical Exits
- **Signal Reversal**: Exit when the opposite signal triggers (e.g., EMA crossover in opposite direction).
- **Candlestick Patterns**: Exit on reversal patterns (engulfing, doji, etc.).

#### 5. GPT-Managed Exits
When enabled, GPT can:
- Suggest exits based on market context.
- Adjust stop/target levels dynamically.
- Recommend partial exits.

#### 6. Partial Exits
- **Pyramiding**: Add to winning positions (up to `max_pyramid_levels`).
- **Partial Profit Booking**: Exit a percentage of the position at profit targets.

### Position Sizing

The bot calculates position size using:
- **ATR-Based**: Size = (Risk Amount) / (ATR × Multiplier)
- **Volatility Targeting**: Adjust size based on forecast vs. target volatility.
- **Kelly Criterion**: Optional Kelly fraction for optimal sizing.
- **Lot Size**: Respects exchange lot sizes with step rounding.

### Detailed Risk Management

Risk management in NiftyScalper is handled in multiple layers to prevent catastrophic losses and manage drawdowns effectively. The bot applies these controls automatically.

#### 1. Daily & Portfolio Limits (The Kill-Switches)
- **Max Daily Loss (`max_daily_loss`)**: A hard cap on the total realized + unrealized loss for the day. If breached, the bot enters a "lockdown" state and ceases all new entries for the remainder of the session.
- **Max Trades Per Day (`max_trades_per_day`)**: Prevents overtrading and excessive brokerage by capping the maximum number of executed trades.
- **Max Stopouts Per Day (`max_stopouts_per_day`)**: Limits the number of consecutive stop-loss hits. If breached, it enforces an extended cooldown or stops trading for the day to avoid revenge trading.
- **Max Open Positions (`max_open_positions`)**: Restricts the bot from overleveraging by capping concurrent open strategies.

#### 2. Pre-Trade Guardrails
Before entering *any* trade, the bot checks these safety conditions:
- **Cooldown Periods**: Enforces a `cooldown_sec` between trades, and a longer `cooldown_after_stopout_sec` following a loss, letting volatile moves settle.
- **Stale Data Rejection (`entry_candle_max_age_sec`)**: Rejects signals if the latest candle data is too old, preventing entries on delayed or stuck data feeds.
- **Premium Constraints (`premium_entry_max`)**: Skips entries if the options premium has spiked too high above historical norms, preventing buying into IV crushes.
- **Spread Shock Protection**: Monitors the real-time bid-ask spread of the option legs. If liquidity dries up and spreads widen aggressively, the entry is aborted.

#### 3. In-Trade Management (Exits & Stops)
Once a trade is live, it is managed strictly via:
- **ATR-Based Stop Loss**: Dynamic stop loss derived from the Average True Range to account for current market volatility.
- **Premium MTM Stop (`premium_mtm_stop_pct`)**: For option-selling strategies (like straddles/strangles), it cuts the trade if the unrealized MTM loss exceeds a specific percentage of the total premium collected.
- **Trailing MTM Stop**: Locks in profits by trailing the maximum unrealized profit. If the P&L pulls back by a set amount, the trade is closed.
- **Max Hold Time (`max_hold_minutes`)**: Time-based exit to prevent holding decaying assets (especially long options) for too long.

#### 4. System-Level Safety
- **Market Hours Guard**: Automatically halts logic outside of standard NSE market hours (09:15 to 15:30 IST) to prevent erratic out-of-hours pricing behavior.
- **Portfolio CVaR (Conditional Value at Risk)**: Advanced risk metric calculated to estimate the expected shortfall in worst-case scenarios, ensuring the total portfolio exposure stays within bounds.

### Delta Hedging (Optional)

For strategies that need delta neutrality:
- **Auto Hedge**: Automatically opens a hedge leg (usually underlying futures/equity).
- **Rebalance**: Periodically adjusts hedge quantity to maintain delta near zero.
- **Tolerance**: Only rebalance when delta drift exceeds threshold.
- **Per-Day Cap**: Limits hedge turnover to control costs.

### Portfolio Beta Hedge (Optional)

Hedges the equity portfolio beta:
- Fetches holdings and calculates portfolio beta vs. NIFTY.
- Opens an offsetting position in NIFTY futures/ETF.
- Refreshes periodically based on configuration.

### Equity Trading Module (Optional)

When `equity_trade_enable=True`:
- **Engine**: Choose between indicator-based or GPT-based entries.
- **Watchlist**: Trade from a configured list or GPT-generated watchlist.
- **Horizon**: Intraday or long-term (CNC) positions.
- **Management**: GPT can manage exits, adjust stops/targets daily.

### GPT Integration (Optional)

The bot integrates with OpenAI-compatible endpoints for:

| Feature | Config Flag | Description |
|---------|-------------|-------------|
| Market Regime Monitor | `gpt_regime_monitor_enabled` | Pauses entries during unfavorable regimes |
| Entry Gate | `gpt_entry_gate_enabled` | GPT approves/rejects each entry |
| Auto Strategy Selection | `strategy_name=auto` | GPT selects best strategy |
| What-If Analysis | `gpt_whatif_enabled` | Analyzes potential trade outcomes |
| Per-Trade Advice | Per-trade UI button | On-demand advice for open positions |
| Equity Watchlist | `equity_trade_gpt_watchlist_enable` | GPT generates equity watchlist |
| Equity Management | `equity_trade_gpt_manage_daily` | GPT manages equity exits daily |

### How GPT Works in Auto Mode

When `strategy_name` is set to `"auto"` and `gpt_auto_select=True`, the bot relies on a Large Language Model (e.g., `gpt-4o-mini`, `o1`) to act as an intelligent options trading assistant. Here is exactly how this process works behind the scenes:

1. **Market Snapshot Generation**
   Before making any requests to the API, the bot compiles a rich JSON snapshot of the current market context. This includes:
   - **Market State:** Underlying spot price, timeframe, and recently detected candlestick patterns (e.g., Bullish Engulfing).
   - **Technical Indicators:** Values for ATR (volatility), RSI (momentum), VWAP, ADX, and moving averages.
   - **Options Data:** Live ATM Call and Put strikes, LTP, implied volatility (IV), and estimated Theta decay. It also samples the options chain near the ATM to evaluate liquidity and spread width.
   - **Portfolio Context:** The current open positions (to maintain directional neutrality if needed) and the allowed strategies based on your configuration.

2. **GPT Analysis & Prompting**
   The bot sends this snapshot to the AI with a strict system prompt instructing it to respond *only* with valid JSON. The AI evaluates the inputs and is asked to provide:
   - `ce_pe_bias`: Its directional bias (`CE`, `PE`, or `NEUTRAL`).
   - `recommended_strategy`: The exact strategy template to deploy (e.g., `bull_call_spread`, `short_strangle`). It is instructed to prefer multi-leg neutral structures when the market is sideways and IV is appropriate.
   - `directional_strike_offset_steps`: How many steps Out-Of-The-Money (OTM) to choose for strikes.
   - `strategy_parameters`: Optional runtime tweaks to parameters like wing distance, delta targets, or theta ratios.

3. **Parsing & Execution**
   - The bot receives and validates the JSON response. 
   - It cross-checks the AI's `recommended_strategy` against the safe list of allowed templates.
   - The trade is then formulated using the AI's parameter tweaks. 
   - **Crucially:** The formulated trade *must still pass* the bot's hard Risk Management limits (such as `max_daily_loss` and margin checks) before actual order execution.

### Advanced Modeling, Machine Learning, and Optimization

When enabled, the bot employs several advanced machine learning models, statistical methods, and convex optimization solvers to enhance entry selection, optimize exits, size positions, and model volatility.

#### 1. Directional ML Signals (Supervised Classification)
When `enable_ml_signals=True`, the bot runs an embedded Machine Learning model ([src/ml_signals.py](file:///c:/Users/rahul/Downloads/NiftyScalper/src/ml_signals.py) and [src/ml_pipeline.py](file:///c:/Users/rahul/Downloads/NiftyScalper/src/ml_pipeline.py)) to generate a probability score for the market's direction:
- **Feature Engineering:** Computes a dense 67-dimensional feature vector from candlestick data, containing:
  - *Candlestick Context:* Body/wick sizes, gap percentages, and pattern detections (e.g., engulfing, hammer, doji).
  - *Rolling Statistics:* Trailing mean, std, min, and max for returns and volume over a lookback window.
  - *Technical Indicators:* Fast/slow EMA differences, RSI, ATR, ADX, ROC, Choppiness Index, Supertrend, and Pivot Points.
  - *Market & Greek Context:* Live delta, gamma, vega, theta, IV, spot price, and one-hot encoded market regime.
- **Model Architecture:** Uses a `RandomForestClassifier` from `scikit-learn` with 100 decision trees. If `scikit-learn` is not installed, it falls back to a custom, pure-Python `FallbackClassifier` utilizing mean-difference profiles and a sigmoid logistic mapping.
- **Exponential Time-Decay:** Integrates sample weighting ($w = e^{-\lambda \cdot (N-1-i)}$) during fitting to prioritize recent market structures.
- **Concept Drift & Online Adaptation:** An `OnlineMLSignal` wrapper monitors the rolling prediction accuracy over a 50-step window. If accuracy drops below `retrain_threshold` (default 60%), it automatically triggers background retraining on the accumulated real-world data, constrained by a 50-sample cooldown to prevent resource churn.
- **Walk-Forward Validation:** Utilizes chronological time-series cross-validation (`TimeSeriesSplit` or chronological block splitting) to validate model performance (Precision, Recall, F1, and ROC AUC) without lookahead bias.

#### 2. Reinforcement Learning (RL) Exit Agent
Implemented in [src/exit_optimizer.py](file:///c:/Users/rahul/Downloads/NiftyScalper/src/exit_optimizer.py), the bot features a hybrid exit optimizer that leverages a reinforcement learning agent when `use_rl=True`:
- **Model Architecture:** Trains a **PPO (Proximal Policy Optimization)** agent with an `MlpPolicy` using `stable-baselines3`.
- **Custom Environment:** Defines `ExitEnvironment` (matching standard OpenAI Gym/Gymnasium design patterns) that constructs a 20-dimensional trade observation vector (unrealized PnL, PnL percentage, trade duration, trailing drawdown, Greeks, IV metrics, volatility/momentum indicators, and regimes).
- **Actions:** Maps discrete agent actions to: `0` (Hold), `1` (Full Exit), `2` (25% Partial Exit), `3` (50% Partial Exit), and `4` (75% Partial Exit).
- **Reward Function:** Custom-designed to reward high PnL captures, incentivize taking timely partial profits, penalize holding losing positions too long, and punish large drawdowns.
- **Rule-Based Fallback:** Seamlessly falls back to a rules-based exit optimizer tracking dynamic trailing stops, theta decay limits, delta explosions, and IV collapse.

#### 3. Volatility Forecasting & Surface Modeling (GARCH + SVI)
Located in [src/volatility.py](file:///c:/Users/rahul/Downloads/NiftyScalper/src/volatility.py), the bot utilizes advanced statistical models for forecasting asset volatility and smoothing implied volatility curves:
- **GARCH(1,1) Forecasting:** Fits a zero-mean normal **GARCH(p=1, q=1)** model to historical return series using the `arch` library to forecast short-term volatility.
- **EWMA Volatility Fallback:** In minimal environments or when asset return history is short (< 50 points), the engine falls back to an **Exponentially Weighted Moving Average (EWMA)** volatility estimator.
- **Stochastic Volatility Inspired (SVI) Surface:** Fits the raw SVI model to option chain log-moneyness ($x = \log(K/S)$) and total variance ($w = \sigma^2 \cdot T$). It uses `scipy.optimize.minimize` (L-BFGS-B method) to solve for the five parameters ($a$, $b$, $\rho$, $m$, $\sigma$) of the SVI formulation:
  $$w(\theta) = a + b \left( \rho(\theta - m) + \sqrt{(\theta - m)^2 + \sigma^2} \right)$$
  This constructs an arbitrage-free **implied volatility surface**, enabling the extraction of ATM IV, 25-delta call/put IVs, and option skew.

#### 4. Convex Portfolio CVaR Optimization
Located in [src/risk_cvar.py](file:///c:/Users/rahul/Downloads/NiftyScalper/src/risk_cvar.py), the position sizing module can optimize multi-asset strategy allocation using risk budgeting:
- **Empirical CVaR:** Computes Conditional Value at Risk (Expected Shortfall) at a given alpha confidence level (default 95%) to estimate worst-case tail loss.
- **Linearized CVaR Programming:** Using the `cvxpy` convex optimization library with the `SCS` solver, the bot solves a linearized expected return maximization problem constrained by portfolio budget limits and a strict maximum target CVaR:
  $$\max_w \mathbb{E}[R^T w] \quad \text{s.t.} \quad w \ge 0, \quad \sum w \le \text{budget}, \quad \text{CVaR}_{\alpha}(w) \le \text{target\_cvar}$$
- **Fallback Heuristic:** If `cvxpy` is not present, it defaults to an equal-weight heuristic, iteratively scaling weights down until the estimated historical CVaR matches the target risk tolerance.

### Detailed Trading Flow

The following sequence diagram outlines the exact decision process the bot goes through during each iteration of its main loop (`run_forever`):

```mermaid
flowchart TD
    Start([Bot Loop Start]) --> RiskCheck{Daily Risk Limits OK?}
    RiskCheck -- No (Max Loss/Stopouts) --> Sleep[Sleep/Wait]
    RiskCheck -- Yes --> MarketHours{Is Market Open?}
    
    MarketHours -- No --> Sleep
    MarketHours -- Yes --> FetchData[Fetch Latest Candles & LTP]
    
    FetchData --> PreGuards{Pre-Entry Guards Pass?}
    PreGuards -- No (Stale data/Liquidity) --> MngTrades
    PreGuards -- Yes --> CalcInd[Compute Tech Indicators\n(EMA, RSI, ATR, VWAP)]
    
    CalcInd --> MLRegime[Detect Market Regime &\nCalculate ML Signal]
    
    MLRegime --> StratSelect[Strategy Selection\n(Auto-allocation or Fixed)]
    
    StratSelect --> GateCheck{GPT Entry Gate / \nML Threshold Met?}
    GateCheck -- No --> MngTrades
    GateCheck -- Yes --> PosSizing[Position Sizing\n(Volatility Target / ATR / Lot)]
    
    PosSizing --> PlaceOrders[Execute Orders via Broker API]
    PlaceOrders --> RecordState[Update Internal Trade State & DB]
    
    RecordState --> MngTrades[Manage Open Trades]
    MngTrades --> CheckExits{Evaluate Exit Conditions}
    
    CheckExits -- Stop Loss / Target Hit --> CloseTrade[Close Positions]
    CheckExits -- MTM Trailing Stop Hit --> CloseTrade
    CheckExits -- Technical Reversal --> CloseTrade
    CheckExits -- Time-based Exit --> CloseTrade
    CheckExits -- Hold Position --> Sleep
    
    CloseTrade --> UpdatePNL[Update Daily P&L and Stopout Counters]
    UpdatePNL --> Sleep
    
    Sleep --> Start
```

1. **Initialization & Risk Check:** At the start of each iteration, the bot checks if global daily risk limits (e.g., max daily loss, max stopouts) have been breached. If so, it stops trading for the day.
2. **Data Ingestion:** The bot fetches the latest candles (1m, 5m, etc.) and current Last Traded Price (LTP) from the broker API.
3. **Indicator & ML Processing:** Technical indicators (RSI, EMA, ATR, etc.) and regime detection profiles are computed. If enabled, the Machine Learning model generates a probability score for market direction.
4. **Strategy Formulation:** Based on the config or auto-regime detection, a strategy template (like directional scalp, short straddle) is selected.
5. **Entry Gating:** The proposed trade must pass the ML threshold and/or GPT Entry Gate (if enabled). Pre-entry guardrails (e.g., bid-ask spread limits, stale data checks) are also verified.
6. **Execution & Management:** Once approved, orders are placed, and the trade state is persisted. In subsequent loops, the bot evaluates the open positions against trailing stops, fixed targets, and time-based exits.

### Strategy Details (Legacy Summary)

- Directional scalps: EMA/RSI/price-action signals with configurable ATR-based trailing/stops and partial booking.
- Premium-selling templates: Short straddle/strangle/condor/fly with MTM stop/target, entry premium guards, spread shock protections, and trailing MTM logic.
- Delta hedging: Optional automatic hedge via a specified hedge instrument, with tolerances, rebalance intervals, step rounding, and per-day caps.
- Equity mode: Optional module to trade equities from holdings using local indicators or GPT watchlist generation.

**Safety, Limitations, and Intended Use**

- This project is a template and demonstration; it is NOT a ready-to-run profitable system. The strategy logic is illustrative and requires thorough backtesting and dry-run validation before real trading.
- Use conservative default configs and test in a simulated or paper environment. Ensure API keys and account permissions are set appropriately.
- The code includes many guardrails (stale-quote blocking, spread checks, kill-switches) but users must validate behavior with their broker and market conditions.

**Development & Testing**

- Unit tests in the repository (tests prefixed with `test_*.py`) exercise strategy components and utilities. Run tests with `pytest -q` after activating the virtual environment.
- Logs written by the UI are persisted to `.scalper.ui.log` in the repo root for troubleshooting.

**Extending the Project**

- Add new strategy templates in `src/strategy.py` and expose configuration knobs via `src/config.py` and the UI settings form in `src/ui.py`.
- To add a new broker connector, implement the minimal client interface used by `NiftyScalper` (get candle/LTP, place/cancel orders, list positions).

**Files of Interest**

- [src/main.py](src/main.py)
- [src/ui.py](src/ui.py)
- [src/strategy.py](src/strategy.py)
- [src/mstock_client.py](src/mstock_client.py)
- [src/config.py](src/config.py)
- [src/gpt_advisor.py](src/gpt_advisor.py)
- [docs/CHANGELOG.md](docs/CHANGELOG.md)

**Contact & Contribution**

If you'd like help configuring the bot for paper trading, adding a connector, or writing a custom strategy, open an issue or start a discussion in the repository. Contributions are welcome — please follow standard Python packaging and testing practices.

---
Generated on: May 19, 2026

**Quickstart**

1. Create and activate a Python virtual environment (Windows example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Configure API keys and environment variables (example):

```powershell
setx MSTOCK_API_KEY "<YOUR_MSTOCK_API_KEY>"
setx MSTOCK_ACCESS_TOKEN "<YOUR_MSTOCK_ACCESS_TOKEN>"
setx MSTOCK_GPT_API_KEY "<YOUR_GPT_KEY>"   # optional
```

3. Launch the UI:

```powershell
python src/ui.py
```

4. Run the headless scalper loop (for unattended execution):

```powershell
python src/main.py
```

**Example Configuration Snippets**

Sample `StrategyConfig` overrides (set via environment or a config loader):

- Enable live trading (careful): `MSTOCK_ENABLE_LIVE_TRADING=1`
- Change lot size: `MSTOCK_LOT_SIZE=75`
- Short strangle defaults: `MSTOCK_STRATEGY_NAME=short_strangle`

A minimal JSON-style override file (for reference):

```json
{
	"strategy_name": "short_strangle",
	"lot_size": 65,
	"max_stopouts_per_day": 1,
	"premium_mtm_stop_pct": 0.30,
	"enable_live_trading": false
}
```

**Architecture (high-level)**

The codebase follows a modular pattern:

- UI layer: [src/ui.py](src/ui.py) — Tkinter front-end, tabbed dashboard, export helpers.
- Strategy layer: [src/strategy.py](src/strategy.py) — `NiftyScalper` engine, risk controls, and trade lifecycle.
- Broker adapter: [src/mstock_client.py](src/mstock_client.py) — wrapper over Mirae Type B SDK for candles, LTP, and orders.
- Config & persistence: [src/config.py](src/config.py), [src/db.py](src/db.py) — settings, SQLite persistence.
- Integrations: [src/gpt_advisor.py](src/gpt_advisor.py), [src/alerts.py](src/alerts.py)

Mermaid diagram (rendering supported in some Markdown viewers):

```mermaid
graph TD
	UI[Scalper UI (Tkinter)] -->|control/events| Strategy[NiftyScalper]
	Strategy -->|calls| Broker[MStockTypeBClient]
	Broker -->|candles/ltp/orders| Exchange[Exchange / TypeB API]
	Strategy -->|save| DB[SQLite Database]
	UI -->|GPT requests| GPT[GPT Advisor]
	Strategy -->|alerts| Telegram[Telegram Notifier]
```

**Troubleshooting & Notes**

- If intraday candle calls fail frequently, the client falls back to historical candles (see `src/mstock_client.py`).
- For GUI theme issues on Windows, toggle `MSTOCK_UI_USE_SV_TTK` or use native themes (see `src/ui.py`).
- Enable debug logs via `MSTOCK_DEBUG_INTRADAY=1` and inspect `.scalper.ui.log` in the repo root.

**Running Tests**

Run unit tests with `pytest` after activating the virtualenv:

```powershell
pip install -r requirements.txt
pytest -q
```

**Next steps**

- I can add a sample `.env` file, generate a more detailed architecture diagram (SVG), or produce a quickstart screencast script. Which would you like next?

**Recent Enhancements (added)**

- Volatility forecasting (GARCH + EWMA fallback) in `src/volatility.py`.
- Regime detection and dynamic strategy selection in `src/strategy_allocator.py`.
- Portfolio CVaR utilities and optimizer stub in `src/risk_cvar.py`.
- ML ensemble signal scaffolding in `src/ml_signals.py` (sklearn-based).
- Position sizing helpers in `src/position_sizing.py` (volatility targeting + Kelly).
- Trailing stop helper in `src/trailing.py` and a minimal backtest harness in `src/backtest_harness.py`.
- Greeks manager and exit optimizer scaffolds in `src/greeks_manager.py` and `src/exit_optimizer.py`.

To run a quick diagnostics check (headless):

```powershell
python -m src.main
```

This will print a short diagnostics report for the enhancements and perform a GPT health check if GPT keys are configured.

**Implemented Features**

Below is a concise list of features that are implemented in the repository (as of this commit). These map to code in the `src/` package and the vendored SDK folder.

- Core strategy engine (`src/strategy.py`):
	- `NiftyScalper` control loop with event sink and on-tick hooks.
	- Multiple strategy templates (directional, short_straddle, short_strangle, spreads, condors, backspreads, long_straddle/strangle, delta-hedged variants).
	- Trade lifecycle events (OPEN / UPDATE / PARTIAL_CLOSE / CLOSE) and per-trade state tracking.
	- Risk controls: `max_stopouts_per_day`, `max_daily_loss`, `max_trades_per_day`, consecutive-stopout and consecutive-win counters.
	- Entry/exit guardrails: stale-LTP checks, ATR/VWAP/Rsi/ADX filters, opening-impulse filters, spread-shock protection, per-leg and trade-level MTM stops/targets, trailing MTM logic.
	- Partial exits calculation and quantity rounding (`calc_partial_exit_qty`).
	- Synthetic candle support when intraday data is missing.

- Broker client (`src/mstock_client.py`):
	- `MStockTypeBClient` thin wrapper over Mirae Type B SDK (`tradingapi_b.MConnectB`).
	- Candle fetching with intraday endpoint plus historical fallback and throttled logging.
	- LTP fetch, instrument/scripmaster resolution, and order abstractions.
	- Exchange/segment mapping and known-index-token helpers.

- UI (`src/ui.py`):
	- Tkinter-based single-window app with Notebook tabs: Live Dashboard, Trade Logs, Signals/Greeks, GPT Advisor, Settings.
	- Live portfolio render: per-leg rows, MTM/realized breakdown, hedge contributions, P&L summary, traded quantities.
	- Sortable Trade Logs table, export to PDF support (ReportLab integration when available).
	- Signals tab with indicators and greeks rendering; greeks computed using `src/greeks.py`.
	- Config form with persisted environment loading/saving and credential sync.
	- UI-level logging, file fallback `.scalper.ui.log`, and safe theme handling.

- Integrations & analytics:
	- GPT Advisor helpers (`src/gpt_advisor.py`) with model normalization, health checks, and advice parsing.
	- Candlestick pattern detectors and common indicators (`src/candlestick_patterns.py`, `src/indicators.py`).
	- Greeks computations and IV solver (`src/greeks.py`).
	- Optional `TelegramNotifier` for async alerts (`src/alerts.py`).

- Persistence & tooling:
	- Lightweight SQLite `DatabaseManager` for trade persistence and daily summaries (`src/db.py`).
	- Unit tests scaffold present (run with `pytest -q`).

- Safety & developer conveniences:
	- Environment-driven configuration (`src/config.py` dataclasses) and many `MSTOCK_*` env toggles.
	- Debug logging toggles and intraday fallback handling.
	- PDF report export and simple CSV-like table rendering for prints/exports.

If you'd like, I can mark this task completed and/or expand any feature entry with file/line links to where it's implemented.