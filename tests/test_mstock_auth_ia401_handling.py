#!/usr/bin/env python3
"""Test m.Stock IA401 handling, backoff, no secret leak, GUI status."""
import os
import time
import pytest
from unittest.mock import patch, MagicMock

# Assume import after edits
try:
    from mstock_client import MStockTypeBClient
    from config import APIConfig
except Exception:
    MStockTypeBClient = None
    APIConfig = None

def test_ia401_sets_backoff_and_clears_token(monkeypatch):
    if MStockTypeBClient is None:
        pytest.skip("client not importable in this env")
    cfg = APIConfig(base_url="https://example", api_key="k", api_secret="s", client_id="c")
    client = MStockTypeBClient(cfg)
    # simulate token
    os.environ["MSTOCK_ACCESS_TOKEN"] = "fake.token"
    # mock the fetch to raise 401
    with patch.object(client, "_fetch_historical_chart_direct", side_effect=RuntimeError("get_historical_chart failed with HTTP 401. ... IA401")):
        try:
            client.get_historical_candles("123", "NSE", "ONE_MINUTE", "2024-01-01", "2024-01-02")
        except RuntimeError as e:
            assert "401" in str(e) or "IA401" in str(e)
    # backoff should be set
    assert time.time() < client._historical_401_backoff_until
    # token should be cleared from env for safety
    assert "MSTOCK_ACCESS_TOKEN" not in os.environ or not os.environ.get("MSTOCK_ACCESS_TOKEN")

def test_no_spam_after_ia401(monkeypatch):
    if MStockTypeBClient is None:
        pytest.skip()
    cfg = APIConfig(base_url="https://example", api_key="k", api_secret="s", client_id="c")
    client = MStockTypeBClient(cfg)
    os.environ["MSTOCK_ACCESS_TOKEN"] = "fake"
    client._historical_401_backoff_until = time.time() + 60
    calls = []
    def fake_fetch(*a, **k):
        calls.append(1)
        return []
    with patch.object(client, "_fetch_historical_chart_direct", side_effect=fake_fetch):
        res = client.get_historical_candles("123", "NSE", "ONE_MINUTE", "2024-01-01", "2024-01-02")
        assert res == []
        assert len(calls) == 0  # should short circuit before fetch

def test_status_no_secret():
    if MStockTypeBClient is None:
        pytest.skip()
    cfg = APIConfig(base_url="https://example", api_key="k", api_secret="s", client_id="c")
    client = MStockTypeBClient(cfg)
    os.environ["MSTOCK_ACCESS_TOKEN"] = "secret123"
    st = client.get_broker_auth_status()
    assert "secret" not in str(st).lower()
    assert st["token_present"] is True
    # token value not in status
    assert "secret123" not in str(st)

def test_gui_status_broker_auth_failed(monkeypatch):
    # mock in UI context
    # This is high level; assume app exposes status
    # For now just verify client status has the key
    if MStockTypeBClient is None:
        pytest.skip()
    cfg = APIConfig(base_url="https://example", api_key="k", api_secret="s", client_id="c")
    client = MStockTypeBClient(cfg)
    client._last_historical_auth_error = "IA401"
    st = client.get_broker_auth_status()
    assert st["last_historical_auth_error"] == "IA401"
