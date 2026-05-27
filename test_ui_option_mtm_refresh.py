from __future__ import annotations

import queue
from types import SimpleNamespace

from src.ui import ScalperUI
from src.strategy import TradeLogEvent


def test_compute_cached_trade_mtm_uses_cached_leg_ltps() -> None:
    dummy = SimpleNamespace()

    mtm = ScalperUI._compute_cached_trade_mtm(
        dummy,
        [
            {"side": "SELL", "quantity": 50, "entry_price": 100.0, "ltp": 90.0},
            {"side": "BUY", "quantity": 50, "entry_price": 20.0, "ltp": 24.0},
        ],
    )

    assert mtm == 700.0


def test_build_live_option_update_event_refreshes_option_leg_prices() -> None:
    client = SimpleNamespace(get_ltp=lambda key: 110.0 if str(key).endswith(":123") else 95.0)

    dummy = SimpleNamespace()
    dummy._is_closed_trade_state = lambda trade_id, state: False
    dummy._is_option_leg = lambda leg: str(leg.get("symbol") or "").endswith(("CE", "PE"))
    dummy._try_get_live_ltp_for_leg = lambda client, leg: ScalperUI._try_get_live_ltp_for_leg(dummy, client, leg)
    dummy._compute_cached_trade_mtm = lambda legs: ScalperUI._compute_cached_trade_mtm(dummy, legs)

    state = {
        "pos_type": "multi",
        "strategy": "iron_condor",
        "status": "OPEN",
        "legs": [
            {"symbol": "NIFTY2631723950CE", "token": "123", "exchange": "NFO", "side": "SELL", "quantity": 50, "entry_price": 100.0},
            {"symbol": "NIFTY2631723750PE", "token": "124", "exchange": "NFO", "side": "BUY", "quantity": 50, "entry_price": 90.0},
        ],
    }

    evt = ScalperUI._build_live_option_update_event(dummy, client, "T-1", state)

    assert evt is not None
    assert evt.event == "UPDATE"
    assert evt.trade_id == "T-1"
    assert evt.mtm == -250.0
    assert [float(leg.get("ltp")) for leg in evt.legs] == [110.0, 95.0]


def test_build_live_option_update_event_ignores_zero_qty_legs() -> None:
    client = SimpleNamespace(get_ltp=lambda key: 111.0)

    dummy = SimpleNamespace()
    dummy._is_closed_trade_state = lambda trade_id, state: False
    dummy._is_option_leg = lambda leg: str(leg.get("symbol") or "").endswith(("CE", "PE"))
    dummy._try_get_live_ltp_for_leg = lambda client, leg: ScalperUI._try_get_live_ltp_for_leg(dummy, client, leg)
    dummy._compute_cached_trade_mtm = lambda legs: ScalperUI._compute_cached_trade_mtm(dummy, legs)

    state = {
        "pos_type": "multi",
        "strategy": "long_straddle",
        "status": "PARTIAL",
        "legs": [
            {
                "symbol": "NIFTY2631723950CE",
                "token": "123",
                "exchange": "NFO",
                "side": "BUY",
                "quantity": 0,
                "entry_price": 100.0,
            }
        ],
    }

    evt = ScalperUI._build_live_option_update_event(dummy, client, "M-closedish", state)
    assert evt is None


def test_build_live_option_update_event_refreshes_hedge_rows_when_requested() -> None:
    client = SimpleNamespace(get_ltp=lambda key: 242.0 if str(key).endswith("NIFTYBEES") else 0.0)

    dummy = SimpleNamespace()
    dummy._is_closed_trade_state = lambda trade_id, state: False
    dummy._is_option_leg = lambda leg: str(leg.get("symbol") or "").endswith(("CE", "PE"))
    dummy._try_get_live_ltp_for_leg = lambda client, leg: ScalperUI._try_get_live_ltp_for_leg(dummy, client, leg)
    dummy._compute_cached_trade_mtm = lambda legs: ScalperUI._compute_cached_trade_mtm(dummy, legs)

    state = {
        "pos_type": "hedge",
        "strategy": "delta_hedge",
        "status": "OPEN",
        "legs": [
            {"symbol": "NSE:NIFTYBEES", "exchange": "NSE", "side": "BUY", "quantity": 50, "entry_price": 240.0},
            {"symbol": "NSE:NIFTYBEES", "exchange": "NSE", "side": "SELL", "quantity": 20, "entry_price": 241.0},
        ],
    }

    evt = ScalperUI._build_live_option_update_event(
        dummy,
        client,
        "T-hedge-H",
        state,
        refresh_all_legs=True,
    )

    assert evt is not None
    assert evt.trade_id == "T-hedge-H"
    assert [float(leg.get("ltp")) for leg in evt.legs] == [242.0, 242.0]
    assert evt.mtm == 80.0


def test_closed_trade_display_uses_realized_not_live_mtm() -> None:
    dummy = SimpleNamespace()
    dummy._trade_state = {}
    dummy._get_leg_qty = lambda leg: ScalperUI._get_leg_qty(dummy, leg)
    dummy._split_legs_for_display = lambda legs: ScalperUI._split_legs_for_display(dummy, legs)
    dummy._compute_cached_trade_mtm = lambda legs: ScalperUI._compute_cached_trade_mtm(dummy, legs)
    dummy._compute_realized_from_closed_legs = lambda legs: ScalperUI._compute_realized_from_closed_legs(dummy, legs)
    dummy._closed_trade_display_value = lambda state: ScalperUI._closed_trade_display_value(dummy, state)
    dummy._is_closed_trade_state = lambda trade_id, state: ScalperUI._is_closed_trade_state(dummy, trade_id, state)

    state = {
        "status": "CLOSED (target)",
        "realized": -250.25,
        "mtm": 999.0,
        "legs": [
            {
                "symbol": "NIFTY26MAY23950PE",
                "side": "BUY",
                "quantity": 50,
                "entry_price": 80.0,
                "ltp": 120.0,
                "exit_price": 75.0,
            }
        ],
    }

    breakdown = ScalperUI._compute_parent_display_pnl_breakdown(dummy, "D1", state)

    assert breakdown["parent_mtm"] == -250.25
    assert breakdown["parent_realized"] == -250.25


def test_pump_trades_ignores_late_update_for_closed_trade() -> None:
    dummy = SimpleNamespace()
    dummy._trade_q = queue.Queue()
    dummy._trade_state = {
        "D1": {
            "opened_ts": 1.0,
            "pos_type": "directional",
            "strategy": "auto_ml_directional",
            "status": "CLOSED (target)",
            "realized": -250.25,
            "mtm": -250.25,
            "_closed_display_pnl": -250.25,
            "_close_realized_applied": True,
            "legs": [
                {
                    "symbol": "NIFTY26MAY23950PE",
                    "side": "BUY",
                    "quantity": 50,
                    "entry_price": 80.0,
                    "ltp": 75.0,
                    "exit_price": 75.0,
                }
            ],
        }
    }
    dummy._trade_rows = {"D1": "row-D1"}
    dummy._db_manager = None
    dummy._scalper = None
    dummy._bot_thread = None
    dummy.status_var = SimpleNamespace(get=lambda: "running")
    dummy.after = lambda *_args, **_kwargs: None
    dummy._pump_trades = lambda: None
    dummy._refresh_pnl_totals = lambda: None
    dummy._pump_dashboard_portfolio = lambda: None
    dummy._render_signals_and_greeks = lambda force=False: None
    dummy._prune_stale_open_rows_when_idle = lambda: None
    dummy._refresh_broker_health = lambda: None
    dummy._is_closed_trade_state = lambda trade_id, state: ScalperUI._is_closed_trade_state(dummy, trade_id, state)

    class MockTree:
        def item(self, *_args, **_kwargs):
            raise AssertionError("closed row should not be re-rendered by late UPDATE")

        def insert(self, *_args, **_kwargs):
            raise AssertionError("closed row should not create a new row")

    dummy.trade_tree = MockTree()
    dummy._trade_q.put(
        TradeLogEvent(
            ts=2.0,
            event="UPDATE",
            trade_id="D1",
            position_type="directional",
            name="auto_ml_directional",
            legs=[
                {
                    "symbol": "NIFTY26MAY23950PE",
                    "side": "BUY",
                    "quantity": 50,
                    "entry_price": 80.0,
                    "ltp": 110.0,
                }
            ],
            mtm=1500.0,
        )
    )
    dummy._trade_q.put(
        TradeLogEvent(
            ts=3.0,
            event="CLOSE",
            trade_id="D1",
            position_type="directional",
            name="auto_ml_directional",
            legs=[],
            realized=500.0,
            reason="duplicate",
        )
    )

    ScalperUI._pump_trades(dummy)

    assert dummy._trade_state["D1"]["mtm"] == -250.25
    assert dummy._trade_state["D1"]["realized"] == -250.25


def test_is_leg_stop_hit_for_display_buy_and_sell() -> None:
    dummy = SimpleNamespace()

    assert ScalperUI._is_leg_stop_hit_for_display(dummy, side="BUY", ltp=74.2, stop_price=75.59) is True
    assert ScalperUI._is_leg_stop_hit_for_display(dummy, side="BUY", ltp=76.0, stop_price=75.59) is False
    assert ScalperUI._is_leg_stop_hit_for_display(dummy, side="SELL", ltp=102.0, stop_price=101.0) is True
    assert ScalperUI._is_leg_stop_hit_for_display(dummy, side="SELL", ltp=99.0, stop_price=101.0) is False
