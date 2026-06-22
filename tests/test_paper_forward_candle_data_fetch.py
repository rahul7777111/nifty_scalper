import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI


class _Client:
    def resolve_exchange_token(self, symbol, exchange_hint=None):
        return "NSE", "26000"

    def fetch_index_candles(self, token, exchange, limit, timeframe, force_historical_only):
        return ([{"open": i, "high": i + 1, "low": i - 1, "close": i, "volume": 1000 + i} for i in range(limit)], timeframe)


def test_historical_candles_fetch_sets_source(monkeypatch):
    monkeypatch.delenv("MSTOCK_UNDERLYING_TOKEN", raising=False)
    ui = object.__new__(ScalperUI)

    candles, source, error = ui._pf_fetch_candles_readonly(_Client())

    assert len(candles) == 100
    assert source == "mstock_historical"
    assert error == ""
    assert {"open", "high", "low", "close", "volume"} <= set(candles[-1])
