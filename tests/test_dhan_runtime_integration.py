from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import dhan_client
from dhan_client import DhanClient, get_dhan_underlying_security_id


def _client() -> DhanClient:
    client = DhanClient.__new__(DhanClient)
    client.cfg = SimpleNamespace(underlying="NIFTY")
    client.client_id = "cid"
    client.access_token = "token123456"
    client.underlying_security_id = "13"
    client._raw = SimpleNamespace(option_chain=lambda **kwargs: {}, intraday_minute_data=lambda **kwargs: {})
    client._context = object()
    client._sdk_diagnostics = {"import_ok": True, "sdk_source": "test"}
    client._last_api_error = ""
    client._last_success_ts = None
    client._validated_live_security_ids = set()
    client._security_df = None
    client._health_passed_current_session = False
    client._live_readiness = {}
    client._ensure_ready = lambda: None  # type: ignore[assignment]
    client._record_success = lambda: None  # type: ignore[assignment]
    client._record_error = lambda endpoint, exc: None  # type: ignore[assignment]
    return client


def test_dhan_underlying_security_id_supports_canonical_and_legacy(monkeypatch):
    monkeypatch.setenv("DHAN_UNDERLYING_SECURITY_ID", "13")
    monkeypatch.delenv("DHAN_UNDER_SECURITY_ID", raising=False)
    assert get_dhan_underlying_security_id("NIFTY") == "13"
    assert dhan_client.os.getenv("DHAN_UNDER_SECURITY_ID") == "13"


def test_dhan_option_chain_normalizes_nested_oc_payload(monkeypatch):
    client = _client()
    payload = {
        "data": {
            "last_price": 24025.5,
            "oc": {
                "24000": {
                    "ce": {"securityId": "101", "tradingSymbol": "NIFTY24000CE", "lastPrice": 120.0, "bid": 119, "ask": 121, "oi": 1000},
                    "pe": {"securityId": "102", "tradingSymbol": "NIFTY24000PE", "lastPrice": 95.0, "bid": 94, "ask": 96, "oi": 900},
                }
            },
        }
    }
    client._resolve_expiry = lambda underlying: "2026-06-25"  # type: ignore[assignment]
    client._call_api_with_retry = lambda *args, **kwargs: payload  # type: ignore[assignment]
    monkeypatch.setenv("DHAN_UNDERLYING_SECURITY_ID", "13")

    rows = DhanClient.get_option_chain(client, "NIFTY")

    assert len(rows) == 2
    assert {row["option_type"] for row in rows} == {"CE", "PE"}
    assert rows[0]["strike_price"] == 24000.0
    assert rows[0]["trading_symbol"].startswith("NIFTY")
    assert rows[0]["ltp"] is not None


def test_dhan_candle_normalizes_parallel_array_payload(monkeypatch):
    client = _client()
    client.resolve_exchange_token = lambda symbol: ("IDX_I", "13")  # type: ignore[assignment]
    client._call_api_with_retry = lambda *args, **kwargs: {
        "data": {
            "timestamp": ["2026-06-15 09:15:00", "2026-06-15 09:16:00"],
            "open": [1, 2],
            "high": [2, 3],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [100, 200],
        }
    }  # type: ignore[assignment]

    candles = DhanClient.get_candles(client, "13", timeframe="1m", limit=100)

    assert len(candles) == 2
    assert candles[-1].close == 2.5
    assert candles[-1].volume == 200


def test_dhan_candle_normalizes_list_of_dict_payload(monkeypatch):
    client = _client()
    client.resolve_exchange_token = lambda symbol: ("IDX_I", "13")  # type: ignore[assignment]
    client._call_api_with_retry = lambda *args, **kwargs: {
        "data": [
            {
                "start_Time": "2026-06-15 09:15:00",
                "open": 10,
                "high": 12,
                "low": 9,
                "close": 11,
                "volume": 500,
            },
            {
                "timestamp": "2026-06-15 09:16:00",
                "open": 11,
                "high": 13,
                "low": 10,
                "close": 12,
                "volume": 600,
            },
        ]
    }  # type: ignore[assignment]

    candles = DhanClient.get_candles(client, "13", timeframe="1m", limit=100)

    assert len(candles) == 2
    assert candles[0].open == 10
    assert candles[-1].close == 12


def test_dhan_empty_candle_response_is_structured_empty_response(monkeypatch):
    client = _client()
    client.resolve_exchange_token = lambda symbol: ("IDX_I", "13")  # type: ignore[assignment]
    client._call_api_with_retry = lambda *args, **kwargs: {"status": "ok", "remarks": "", "data": {}}  # type: ignore[assignment]

    result = DhanClient.get_candles_result(client, "13", timeframe="1m", limit=100)

    assert result.ok is False
    assert result.error_code == "EMPTY_RESPONSE"
    assert result.data == []
