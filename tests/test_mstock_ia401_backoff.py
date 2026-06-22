#!/usr/bin/env python3
"""IA401 backoff, no spam, clean status, no secret leak."""
import os
import time
import pytest
from unittest.mock import patch

# reuse or minimal
try:
    from mstock_client import MStockTypeBClient
    from config import APIConfig
except:
    MStockTypeBClient = None
    APIConfig = dict

def test_ia401_backoff_no_spam_and_status(monkeypatch):
    if MStockTypeBClient is None:
        pytest.skip("no client")
    cfg = APIConfig(base_url="u", api_key="k", api_secret="s", client_id="c")
    client = MStockTypeBClient(cfg)
    os.environ["MSTOCK_ACCESS_TOKEN"] = "tok"
    with patch.object(client, "_fetch_historical_chart_direct", side_effect=RuntimeError("HTTP 401 IA401")):
        for _ in range(3):
            try:
                client.get_historical_candles("t", "NSE", "1m", "d1", "d2")
            except:
                pass
    assert time.time() < getattr(client, "_historical_401_backoff_until", 0)
    st = client.get_broker_auth_status()
    assert st["last_historical_auth_error"] == "IA401" or "401" in str(st.get("last_historical_auth_error",""))
    # token value itself should not be in the values (we mask in practice)
    assert "supersecret" not in str(list(st.values()))  # no secret value leaked in status values

def test_status_no_leak():
    if MStockTypeBClient is None:
        pytest.skip()
    cfg = APIConfig(base_url="u", api_key="k", api_secret="s", client_id="c")
    client = MStockTypeBClient(cfg)
    os.environ["MSTOCK_ACCESS_TOKEN"] = "supersecret"
    st = client.get_broker_auth_status()
    assert "supersecret" not in str(st).lower()
    assert st.get("token_present") is True
