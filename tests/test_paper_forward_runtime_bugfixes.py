from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_forward_engine import (
    _compute_option_pnl,
    _fetch_live_broker_option_ltp,
    _last_good_mark,
    _LAST_GOOD_MARKS,
    _LTP_FAIL_CACHE,
    _record_last_good_mark,
)


def test_paper_forward_pnl_qty_65_low_ltp_cases() -> None:
    assert _compute_option_pnl(6.35, 0.10, 65, side="BUY") == -406.25
    assert abs(_compute_option_pnl(4.95, 0.10, 65, side="BUY") - -315.25) < 1e-9


def test_last_good_mark_survives_after_broker_ltp() -> None:
    _LAST_GOOD_MARKS.clear()
    _record_last_good_mark(
        symbol="NIFTY2661624000CE",
        token="50615",
        exchange="NFO",
        price=0.10,
        source="BROKER_LTP",
    )
    px, source, stale, rec = _last_good_mark("NIFTY2661624000CE", "50615", "NFO")
    assert px == 0.10
    assert source == "BROKER_LTP"
    assert stale is False
    assert rec["symbol"] == "NIFTY2661624000CE"


class _FallbackLtpClient:
    def fetch_option_quote_for_paper(self, exchange, token, symbol):
        raise RuntimeError("token quote down")

    def get_ltp(self, key):
        if key == "NFO:50615" or key == "50615":
            raise RuntimeError("token get_ltp down")
        if key == "NFO:NIFTY2661624000CE":
            return 0.10
        raise RuntimeError(f"unexpected key {key}")


def test_ltp_token_failure_does_not_block_exchange_symbol_fallback() -> None:
    _LTP_FAIL_CACHE.clear()
    px, source = _fetch_live_broker_option_ltp(
        _FallbackLtpClient(),
        symbol="NIFTY2661624000CE",
        token="50615",
        exchange="NFO",
        contract={"trading_symbol": "NIFTY2661624000CE", "token": "50615", "exchange": "NFO"},
    )
    assert px == 0.10
    assert source == "BROKER_LTP"


def test_ltp_failure_cache_is_short_lived() -> None:
    _LTP_FAIL_CACHE.clear()
    _LTP_FAIL_CACHE["token_get_ltp:NFO:50615"] = time.time() - 10.0
    px, source = _fetch_live_broker_option_ltp(
        _FallbackLtpClient(),
        symbol="NIFTY2661624000CE",
        token="50615",
        exchange="NFO",
        contract={"trading_symbol": "NIFTY2661624000CE", "token": "50615", "exchange": "NFO"},
    )
    assert px == 0.10
    assert source == "BROKER_LTP"
