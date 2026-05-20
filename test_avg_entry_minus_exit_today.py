from __future__ import annotations

from datetime import datetime
from queue import Queue

from src.strategy import TradeLogEvent
from src.ui import ScalperUI


def test_avg_entry_minus_exit_today_signed_weighted() -> None:
    ui = ScalperUI.__new__(ScalperUI)
    today = datetime.now().date()
    ui._closed_option_legs_day = today

    ui._closed_option_legs = [
        {
            "ts": 0.0,
            "symbol": "X",
            "side": "BUY",
            "quantity": 10,
            "entry_price": 100.0,
            "exit_price": 110.0,
        },
        {
            "ts": 0.0,
            "symbol": "Y",
            "side": "BUY",
            "quantity": 5,
            "entry_price": 200.0,
            "exit_price": 190.0,
        },
    ]

    avg, qty = ScalperUI._avg_entry_minus_exit_today(ui)
    assert qty == 15
    assert avg is not None
    assert abs(float(avg) - ((10 * 10 + (-10) * 5) / 15.0)) < 1e-9


def test_avg_entry_minus_exit_today_sell_leg_profit_positive() -> None:
    ui = ScalperUI.__new__(ScalperUI)
    today = datetime.now().date()
    ui._closed_option_legs_day = today

    ui._closed_option_legs = [
        {
            "ts": 0.0,
            "symbol": "Z",
            "side": "SELL",
            "quantity": 10,
            "entry_price": 100.0,
            "exit_price": 90.0,
        }
    ]

    avg, qty = ScalperUI._avg_entry_minus_exit_today(ui)
    assert qty == 10
    assert avg == 10.0


def test_close_event_with_zero_qty_keeps_leg_history_and_ltp_exit() -> None:
    ui = ScalperUI.__new__(ScalperUI)
    today = datetime.now().date()
    now_ts = datetime.now().timestamp()

    ui._closed_option_legs_day = today
    ui._closed_option_legs = []
    ui._qty_traded_day = today
    ui._qty_traded_total = 0
    ui._open_qty_by_trade_symbol = {}
    ui._recent_qty_reduction_by_track_key = {}
    ui._open_hedge_qty_by_trade_symbol = {}
    ui._trade_state = {
        "T1": {
            "opened_ts": now_ts,
            "pos_type": "multi",
            "strategy": "iron_condor",
            "legs": [
                {
                    "symbol": "NIFTY27MAR2522400CE",
                    "side": "SELL",
                    "quantity": 50,
                    "strike": 22400,
                    "option_type": "CE",
                    "entry_price": 101.25,
                }
            ],
            "mtm": 0.0,
            "realized": 0.0,
            "status": "OPEN",
        }
    }
    ui._trade_q = Queue()
    ui.after = lambda *args, **kwargs: None  # type: ignore[assignment]

    ui._trade_q.put(
        TradeLogEvent(
            ts=now_ts,
            event="CLOSE",
            trade_id="T1",
            position_type="multi",
            name="iron_condor",
            legs=[
                {
                    "symbol": "NIFTY27MAR2522400CE",
                    "side": "SELL",
                    "quantity": 0,
                    "strike": 22400,
                    "option_type": "CE",
                    "ltp": 95.0,
                }
            ],
            realized=250.0,
        )
    )

    ui._pump_trades()

    assert ui._trade_state["T1"]["status"].startswith("CLOSED")
    assert ui._trade_state["T1"]["legs"][0]["quantity"] == 50
    assert ui._trade_state["T1"]["legs"][0]["ltp"] == 95.0
    assert len(ui._closed_option_legs) == 1
    assert ui._closed_option_legs[0]["exit_price"] == 95.0


def test_completed_trade_event_is_persisted_to_db() -> None:
    class DummyDb:
        def __init__(self) -> None:
            self.rows = []

        def insert_trade(self, **kwargs) -> None:
            self.rows.append(kwargs)

    ui = ScalperUI.__new__(ScalperUI)
    db = DummyDb()
    ui._db_manager = db
    ui._persisted_trade_events = set()

    evt = TradeLogEvent(
        ts=1234.0,
        event="CLOSE",
        trade_id="T-db",
        position_type="multi",
        name="iron_condor",
        legs=[
            {
                "symbol": "NIFTY27MAR2522400CE",
                "side": "SELL",
                "quantity": 50,
                "entry_price": 100.0,
                "exit_price": 90.0,
            }
        ],
        realized=500.0,
    )

    ScalperUI._persist_completed_trade_event(ui, evt)
    ScalperUI._persist_completed_trade_event(ui, evt)

    assert len(db.rows) == 1
    assert db.rows[0]["trade_id"] == "T-db"
    assert db.rows[0]["symbol"] == "NIFTY27MAR2522400CE"
    assert db.rows[0]["quantity"] == 50
    assert db.rows[0]["realized_pnl"] == 500.0
