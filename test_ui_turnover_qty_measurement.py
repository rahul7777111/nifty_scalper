from __future__ import annotations

from datetime import datetime

from src.strategy import TradeLogEvent
from src.ui import ScalperUI


def _evt(ts: float, event: str, qty: int) -> TradeLogEvent:
    return TradeLogEvent(
        ts=ts,
        event=event,
        trade_id="T-1",
        position_type="multi",
        name="iron_condor",
        legs=[
            {
                "symbol": "NIFTY26APR25000CE",
                "token": "111",
                "side": "SELL",
                "quantity": qty,
                "option_type": "CE",
                "is_hedge": False,
            }
        ],
    )


def _fresh_ui() -> ScalperUI:
    ui = ScalperUI.__new__(ScalperUI)
    ui._qty_traded_day = datetime.now().date()
    ui._qty_traded_total = 0
    ui._open_qty_by_trade_symbol = {}
    ui._recent_qty_reduction_by_track_key = {}
    return ui


def test_turnover_not_double_counted_when_update_precedes_partial_close() -> None:
    ui = _fresh_ui()

    ScalperUI._note_traded_qty(ui, _evt(1000.0, "OPEN", 100))
    ScalperUI._note_traded_qty(ui, _evt(1001.0, "UPDATE", 60))
    ScalperUI._note_traded_qty(ui, _evt(1002.0, "PARTIAL_CLOSE", 40))

    # OPEN=100 and one 40-lot exit should be 140, not 180.
    assert ui._qty_traded_total == 140


def test_turnover_still_counts_partial_when_partial_precedes_update() -> None:
    ui = _fresh_ui()

    ScalperUI._note_traded_qty(ui, _evt(2000.0, "OPEN", 100))
    ScalperUI._note_traded_qty(ui, _evt(2001.0, "PARTIAL_CLOSE", 40))
    ScalperUI._note_traded_qty(ui, _evt(2002.0, "UPDATE", 60))

    # Same logical exit as above, event order reversed.
    assert ui._qty_traded_total == 140
