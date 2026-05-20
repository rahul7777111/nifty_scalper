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

**Strategy Details**

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