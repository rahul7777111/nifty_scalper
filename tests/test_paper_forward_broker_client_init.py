import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ui


def test_token_set_creates_data_only_client(monkeypatch):
    created = {}

    class DummyClient:
        def __init__(self, cfg):
            created["cfg"] = cfg
            self._raw = SimpleNamespace(set_access_token=lambda token: created.setdefault("token", token))

    app = ui.ScalperUI.__new__(ui.ScalperUI)
    app._client = None
    app._scalper = None
    monkeypatch.setenv("MSTOCK_ACCESS_TOKEN", "token-123")
    monkeypatch.setattr(ui, "load_api_config", lambda: SimpleNamespace(api_key="k"))
    monkeypatch.setattr(ui, "MStockTypeBClient", DummyClient)

    client, status, error = app._pf_get_or_create_data_client()

    assert client is not None
    assert status == "created_mstock_data_client"
    assert error == ""
    assert created["cfg"].api_key == "k"
    assert created["token"] == "token-123"


def test_token_set_does_not_report_no_client_forever(monkeypatch):
    app = ui.ScalperUI.__new__(ui.ScalperUI)
    client = SimpleNamespace(get_ltp=lambda symbol: 24150.0)
    monkeypatch.setenv("MSTOCK_ACCESS_TOKEN", "token-123")

    auth, err, spot = app._pf_validate_broker_auth_readonly(client)

    assert auth == "AUTH_OK"
    assert err == ""
    assert spot == 24150.0
