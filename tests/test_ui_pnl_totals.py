from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ui


class _DummyVar:
    def __init__(self) -> None:
        self.value = None

    def set(self, value: str) -> None:
        self.value = value


def test_refresh_pnl_totals_uses_parent_realized_once_per_trade():
    app = object.__new__(ui.ScalperUI)
    app._trade_state = {
        "T1": {"status": "OPEN", "realized": 100.0},
        "T1-H": {"status": "OPEN", "realized": 25.0},
        "T2": {"status": "CLOSED", "realized": -40.0},
    }
    app._pnl_ledger = [125.0, -40.0]
    app._pnl_profit_var = _DummyVar()
    app._pnl_loss_var = _DummyVar()
    app._avg_entry_minus_exit_var = _DummyVar()
    app._qty_traded_today_var = _DummyVar()
    app._hedge_qty_traded_today_var = _DummyVar()
    app._margin_required_var = _DummyVar()
    app._qty_traded_total = 0
    app._hedge_qty_traded_total = 0
    app._margin_required_live_total = None
    app._margin_required_live_ts = 0.0
    app._margin_live_stale_sec = 0.0
    app._avg_entry_minus_exit_today = lambda: (None, 0)
    ui.ScalperUI._refresh_pnl_totals(app)

    assert app._pnl_profit_var.value == "₹125.00"
    assert app._pnl_loss_var.value == "₹40.00"
