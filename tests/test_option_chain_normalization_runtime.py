from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI


def test_option_chain_normalizes_ltp_bid_ask_aliases() -> None:
    app = object()
    rows = ScalperUI._normalize_option_chain_rows(
        app,
        [
            {
                "tradingSymbol": "NIFTY2661624000CE",
                "strikePrice": 24000,
                "optionType": "CE",
                "last_traded_price": 12.3,
                "best_bid": 12.2,
                "best_ask": 12.4,
                "securityId": "50615",
                "underlyingValue": 23982.9,
            }
        ],
        resolved_spot=23982.9,
    )
    assert rows[0]["option_ltp"] == 12.3
    assert rows[0]["ltp"] == 12.3
    assert rows[0]["bid"] == 12.2
    assert rows[0]["ask"] == 12.4
    assert rows[0]["security_id"] == "50615"


def test_contract_master_only_row_is_not_markable_as_price_chain() -> None:
    app = object()
    rows = ScalperUI._normalize_option_chain_rows(
        app,
        [{"symbol": "NIFTY2661624000CE", "strike": 24000, "option_type": "CE", "token": "50615"}],
        resolved_spot=23982.9,
    )
    assert rows[0]["chain_source"] == "CONTRACT_MASTER_ONLY"
    assert rows[0]["marking_allowed"] is False
    assert rows[0].get("option_ltp") in (None, "")
