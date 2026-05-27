## Changelog

2026-05-20 — Major enhancements
- Added ML signal ensemble scaffolding and demo model (`src/ml_signals.py`).
- Integrated volatility forecasting with EWMA/GARCH fallback (`src/volatility.py`).
- Added CVaR sizing utilities and integration (`src/risk_cvar.py`, `src/position_sizing.py`).
- Greeks manager and one-click simulated hedges (`src/greeks_manager.py`, UI enhancements).
- Exit optimizer scaffold and RL harness stub (`src/exit_agent.py`).
- GPT Advisor improvements and UI transparency (mini-log, health) (`src/gpt_advisor.py`, UI).
- Trade Builder UI with net premium & margin preview (`src/ui.py`).
- Backtest Runner UI and minimal backtest harness (`src/backtest_harness.py`, UI).
- Tools: ML training runner `tools/train_ml.py`.
