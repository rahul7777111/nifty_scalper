from __future__ import annotations

from datetime import datetime

import src.ui as ui_module
from src.ui import ScalperUI


class FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        current = cls(2025, 3, 10, 14, 30, 0)
        if tz is not None:
            return tz.fromutc(current.replace(tzinfo=tz))
        return current


def test_collect_today_trade_log_pdf_rows_filters_and_formats(monkeypatch) -> None:
    monkeypatch.setattr(ui_module, "datetime", FrozenDateTime)

    ui = ScalperUI.__new__(ScalperUI)
    today_open = FrozenDateTime(2025, 3, 10, 9, 20, 15).timestamp()
    today_hedge = FrozenDateTime(2025, 3, 10, 9, 45, 0).timestamp()
    old_open = FrozenDateTime(2025, 3, 9, 15, 20, 0).timestamp()
    ui._trade_state = {
        "T1": {
            "opened_ts": today_open,
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
                },
                {
                    "symbol": "NIFTY27MAR2522600CE",
                    "side": "BUY",
                    "quantity": 50,
                    "strike": 22600,
                    "option_type": "CE",
                    "entry_price": 41.5,
                    "is_hedge": True,
                },
            ],
            "mtm": 1250.5,
            "realized": 300.0,
            "status": "OPEN",
        },
        "T1-H": {
            "opened_ts": today_hedge,
            "pos_type": "hedge",
            "strategy": "delta_hedge",
            "legs": [
                {
                    "symbol": "NIFTY-I",
                    "side": "BUY",
                    "quantity": 50,
                    "is_hedge": True,
                }
            ],
            "mtm": -10.0,
            "realized": 0.0,
            "status": "OPEN",
        },
        "OLD1": {
            "opened_ts": old_open,
            "pos_type": "directional",
            "strategy": "long_call",
            "legs": [],
            "mtm": 999.0,
            "realized": 500.0,
            "status": "CLOSED",
        },
    }

    rows = ScalperUI._collect_today_trade_log_pdf_rows(ui)

    assert [row[1] for row in rows] == ["T1", "T1-H"]
    assert rows[0][0] == "09:20:15"
    assert rows[0][2] == "multi"
    assert rows[0][3] == "iron_condor"
    assert "SELL 22400 CE x50 (entry 101.25)" in rows[0][4]
    assert "BUY 22600 CE x50 (entry 41.50)" in rows[0][4]
    assert "HEDGE:" not in rows[0][4]
    assert rows[0][5] == "1240.50 (base 1250.50 + hedge -10.00)"
    assert rows[0][6] == "300.00"
    assert rows[1][4] == "BUY 22400 CE x50"
    assert rows[1][5] == "-10.00"
