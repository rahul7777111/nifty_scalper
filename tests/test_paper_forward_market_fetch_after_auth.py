import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI


class Client:
    def __init__(self):
        self.chain_calls = 0
        self.candle_calls = 0

    def get_option_chain(self, underlying):
        self.chain_calls += 1
        return [{"strike": 23000 + i * 50, "option_type": "CE", "ltp": 10, "volume": 1, "oi": 1} for i in range(24)]

    def resolve_exchange_token(self, symbol, exchange_hint=None):
        return "NSE", "26000"

    def fetch_index_candles(self, token, exchange, limit, timeframe, force_historical_only):
        self.candle_calls += 1
        return ([{"open": 1, "high": 2, "low": 1, "close": 2, "volume": 100}] * limit, timeframe)


def test_market_fetch_helpers_run_after_auth_ok(monkeypatch):
    monkeypatch.setenv("PAPER_FORWARD_MIN_OPTION_CHAIN_ROWS", "20")
    client = Client()
    ui = object.__new__(ScalperUI)

    chain, chain_status, _chain_error, _expiry = ui._pf_fetch_full_option_chain_readonly(client, spot=23257.95)
    candles, candle_source, _candle_error = ui._pf_fetch_candles_readonly(client)

    assert chain_status == "DATA_OK"
    assert len(chain) == 24
    assert candle_source == "mstock_historical"
    assert len(candles) == 100
    assert client.chain_calls == 1
    assert client.candle_calls == 1


def test_auth_failed_status_indicates_fetch_should_be_skipped():
    ui = object.__new__(ScalperUI)

    label = ui._pf_status_label_from_data_status({
        "broker_auth": "SESSION_EXPIRED:IA401",
        "auth_validation_attempted": True,
        "data_quality_status": "SESSION_EXPIRED",
    })

    assert label == "BROKER_SESSION_EXPIRED"
