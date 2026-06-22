from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mstock_client import MStockTypeBClient
from paper_forward_engine import _fetch_live_broker_option_ltp
from ui import ScalperUI


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _Raw:
    def __init__(self, payload):
        self.payload = payload

    def get_market_quote(self, mode, exchange_tokens):
        return _Resp(self.payload)


def _client(payload):
    c = MStockTypeBClient.__new__(MStockTypeBClient)
    c._raw = _Raw(payload)
    c._broker_ip_mismatch = False
    c._option_chain_status = "UNKNOWN"
    c._last_option_chain_error = ""
    c._broker_endpoint_failures = {}
    c._broker_endpoint_paused_until = {}
    c._broker_endpoint_last_error = {}
    c._raw_quote_failure_logged = set()
    c._log_throttle = {}
    c._ensure_valid_token = lambda *a, **k: None
    c._resolve_token_for_quote = lambda symbol, exchange_hint=None: ("NSE", "26000")
    c.get_bid_ask = lambda *a, **k: (None, None, None)
    return c


def test_get_ltp_with_ia403_response_returns_none_and_sets_status():
    payload = {
        "status": False,
        "errorcode": "IA403",
        "message": "Primary and Secondary IP Address are not matching with current IP address.",
    }
    c = _client(payload)
    assert c.get_ltp("NSE:26000") is None
    st = c.get_broker_auth_status()
    assert st["broker_ip_mismatch"] is True
    assert st["broker_data_status"] == "BROKER_IP_MISMATCH"


def test_get_ltp_with_data_dict():
    c = _client({"status": True, "data": {"NSE": {"26000": {"ltp": 23999.5, "token": "26000"}}}})
    assert c.get_ltp("NSE:26000") == 23999.5


def test_get_ltp_with_data_list():
    c = _client({"status": True, "data": [{"instrumentToken": "26000", "lastTradedPrice": "24001.25"}]})
    assert c.get_ltp("NSE:26000") == 24001.25


def test_get_ltp_missing_ltp_returns_none():
    c = _client({"status": True, "data": [{"instrumentToken": "26000", "foo": "bar"}]})
    assert c.get_ltp("NSE:26000") is None


def test_paper_forward_ltp_none_does_not_crash():
    class C:
        def get_ltp(self, symbol):
            return None

    price, source = _fetch_live_broker_option_ltp(
        C(),
        symbol="NIFTY2661624000CE",
        token="50615",
        exchange="NFO",
        contract={"symbol": "NIFTY2661624000CE", "token": "50615", "exchange": "NFO"},
    )
    assert price is None
    assert source == ""


def test_gui_status_receives_broker_ip_mismatch():
    class Var:
        def __init__(self):
            self.value = ""

        def set(self, value):
            self.value = value

    class Client:
        def get_broker_auth_status(self):
            return {
                "broker_ip_mismatch": True,
                "broker_data_status": "BROKER_IP_MISMATCH",
                "broker_status_message": "m.Stock IP mismatch: current public IP is not whitelisted in m.Stock API settings",
            }

    ui = ScalperUI.__new__(ScalperUI)
    ui._selected_broker = lambda: "mstock"
    ui.status_var = Var()
    ui._app_status_var = Var()
    ui._option_chain_status_var = Var()
    ScalperUI._refresh_broker_status(ui, Client())
    assert "m.Stock IP mismatch" in ui.status_var.value
    assert ui._app_status_var.value == "BROKER_IP_MISMATCH"
    assert ui._option_chain_status_var.value == "BROKER_IP_MISMATCH"
