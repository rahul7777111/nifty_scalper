import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI


class _Client:
    def __init__(self, rows):
        self.rows = rows

    def get_option_chain(self, underlying):
        return self.rows


def _row(i):
    return {
        "strikePrice": 23000 + i * 50,
        "right": "CALL" if i % 2 == 0 else "PUT",
        "lastPrice": 100 + i,
        "traded_volume": 1000 + i,
        "openInterest": 5000 + i,
        "expiry": "2026-06-25",
    }


def test_option_chain_empty_reports_exact_reason():
    ui = object.__new__(ScalperUI)
    rows, status, error, expiry = ui._pf_fetch_full_option_chain_readonly(_Client([]), spot=23172.5)

    assert rows == []
    assert status == "option_chain_empty"
    assert "option_chain_empty_response" in error
    assert expiry == ""


def test_full_option_chain_normalizes_rows(monkeypatch):
    monkeypatch.setenv("PAPER_FORWARD_MIN_OPTION_CHAIN_ROWS", "20")
    ui = object.__new__(ScalperUI)
    rows, status, error, expiry = ui._pf_fetch_full_option_chain_readonly(_Client([_row(i) for i in range(24)]), spot=23172.5)

    assert len(rows) == 24
    assert status == "DATA_OK"
    assert error == ""
    assert expiry == "2026-06-25"
    assert rows[0]["strike"] == 23000
    assert rows[0]["ltp"] == 100
    assert rows[0]["volume"] == 1000
    assert rows[0]["oi"] == 5000
