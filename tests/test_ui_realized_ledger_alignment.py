from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ui
from strategy import TradeLogEvent


class _DummyVar:
    def __init__(self) -> None:
        self.value = None

    def set(self, value: str) -> None:
        self.value = value


class _DummyQueue:
    def __init__(self, items):
        self._items = list(items)

    def get_nowait(self):
        if not self._items:
            raise ui.queue.Empty
        return self._items.pop(0)


def _make_app(events):
    app = object.__new__(ui.ScalperUI)
    app._trade_q = _DummyQueue(events)
    app._trade_state = {}
    app._trade_rows = {}
    app._pnl_ledger = []
    app._db_manager = None
    app.live_chart_plugin = None
    app._note_closed_option_legs = lambda evt: None
    app._note_traded_qty = lambda evt: None
    app._note_hedge_traded_qty = lambda evt: None
    app._merge_trade_legs = lambda existing, new: list(new or existing or [])
    app._refresh_pnl_totals = lambda: None
    app._sync_trade_log_row = lambda *args, **kwargs: None
    app._update_trade_log_empty_state = lambda: None
    app._is_closed_trade_state = lambda trade_id, state: str(state.get("status") or "").startswith("CLOSED")
    app._closed_trade_display_value = lambda state: float(state.get("realized") or 0.0)
    app._compute_parent_display_pnl = lambda trade_id, state: (state.get("mtm"), state.get("realized"))
    app._split_legs_for_display = lambda legs: (list(legs or []), [], [])
    app._first_display_option_context_from_legs = lambda legs: None
    app._format_trade_legs_for_state = lambda trade_id, state: ""
    app._safe_after = lambda *args, **kwargs: None
    app.after = lambda *args, **kwargs: None
    app.trade_tree = None
    return app


def test_partial_and_close_both_flow_into_realized_ledger():
    events = [
        TradeLogEvent(
            ts=1.0,
            event="OPEN",
            trade_id="T1",
            position_type="directional",
            name="long_call",
            legs=[{"symbol": "ABC", "side": "BUY", "quantity": 10, "entry_price": 100.0}],
            mtm=0.0,
        ),
        TradeLogEvent(
            ts=2.0,
            event="PARTIAL_CLOSE",
            trade_id="T1",
            position_type="directional",
            name="long_call",
            legs=[{"symbol": "ABC", "side": "BUY", "quantity": 5, "entry_price": 100.0, "exit_price": 110.0}],
            realized=50.0,
            reason="partial_target",
        ),
        TradeLogEvent(
            ts=3.0,
            event="CLOSE",
            trade_id="T1",
            position_type="directional",
            name="long_call",
            legs=[{"symbol": "ABC", "side": "BUY", "quantity": 5, "entry_price": 100.0, "exit_price": 120.0}],
            realized=100.0,
            reason="target",
        ),
    ]

    app = _make_app(events)
    ui.ScalperUI._pump_trades(app)

    assert app._pnl_ledger == [50.0, 100.0]


def test_refresh_pnl_totals_uses_realized_ledger():
    app = object.__new__(ui.ScalperUI)
    app._pnl_ledger = [125.0, -40.0, 25.0]
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
    app._trade_state = {}
    app._avg_entry_minus_exit_today = lambda: (None, 0)

    ui.ScalperUI._refresh_pnl_totals(app)

    assert app._pnl_profit_var.value == "₹150.00"
    assert app._pnl_loss_var.value == "₹40.00"
