from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ui


class DummyVar:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def set(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


def test_calculate_intraday_charges_returns_nonzero_for_options() -> None:
    charges = ui.calculate_intraday_charges(100.0, 110.0, 50, is_options=True)
    assert charges["total"] > 0.0
    assert charges["brokerage"] >= 40.0


def test_estimate_dashboard_charges_aggregates_rows() -> None:
    app = object.__new__(ui.ScalperUI)
    snapshot = {
        "equity_rows": [{"entry": 100.0, "ltp": 101.0, "qty": 10}],
        "option_rows": [{"trade_id": "T1", "symbol": "NIFTY24JUN25000CE", "entry": 120.0, "ltp": 130.0, "qty": 50}],
    }
    total = ui.ScalperUI._estimate_dashboard_charges(app, snapshot)
    assert total > 0.0


def test_render_dashboard_portfolio_updates_charge_vars() -> None:
    app = object.__new__(ui.ScalperUI)
    app._dash_gross_pnl_var = DummyVar()
    app._dash_charges_var = DummyVar()
    app._dash_net_pnl_var = DummyVar()
    app._dash_portfolio_summary_var = DummyVar()
    app.pnl_series_history = {"Realized P&L": [], "Unrealized MTM": [], "Live P&L": []}
    app._pnl_ledger = [100.0]
    app.pnl_chart_plugin = None
    snapshot = {
        "equity_rows": [{"entry": 100.0, "ltp": 101.0, "qty": 10}],
        "option_rows": [{"trade_id": "T1", "symbol": "NIFTY24JUN25000CE", "entry": 120.0, "ltp": 130.0, "qty": 50}],
        "equity_exposure": 1010.0,
        "option_exposure": 6500.0,
        "equity_ratio": 0.13,
        "option_ratio": 0.87,
        "equity_count": 1,
        "option_trade_count": 1,
        "option_leg_count": 1,
        "total_unrealized": 25.0,
        "ts": 1.0,
    }

    ui.ScalperUI._render_dashboard_portfolio(app, snapshot)

    assert app._dash_gross_pnl_var.get().startswith("₹")
    assert app._dash_charges_var.get().startswith("₹")
    assert app._dash_net_pnl_var.get().startswith("₹")
