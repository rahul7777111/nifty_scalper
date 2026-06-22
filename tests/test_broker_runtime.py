from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from scripts import collect_live_option_context as collector
from scripts.validate_broker_runtime import validate_dhan_runtime
from src import dhan_client
from src.dhan_client import DhanClient, get_dhan_sdk_diagnostics, mask_token, validate_dhan_live_readiness
from src.ui import merge_saved_broker_credentials


def test_broker_selection_persists_correctly() -> None:
    existing = {"broker": "mstock", "api_key": "mk", "username": "u", "password": "p"}
    merged = merge_saved_broker_credentials(existing, "dhan", {"dhan_client_id": "cid", "dhan_access_token": "token", "dhan_under_security_id": "13"})
    assert merged["broker"] == "dhan"
    assert merged["api_key"] == "mk"
    assert merged["dhan_client_id"] == "cid"


def test_dhan_credentials_load_save_does_not_delete_mstock() -> None:
    existing = {"api_key": "mk", "username": "u", "password": "p"}
    merged = merge_saved_broker_credentials(existing, "dhan", {"dhan_client_id": "cid", "dhan_access_token": "tok"})
    assert merged["api_key"] == "mk"
    assert merged["username"] == "u"
    assert merged["dhan_access_token"] == "tok"


def test_mstock_fields_unaffected() -> None:
    existing = {"dhan_client_id": "cid", "dhan_access_token": "tok", "dhan_under_security_id": "13"}
    merged = merge_saved_broker_credentials(existing, "mstock", {"api_key": "mk", "username": "u", "password": "p"})
    assert merged["dhan_client_id"] == "cid"
    assert merged["api_key"] == "mk"


def test_dhan_health_check_handles_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DHAN_CLIENT_ID", "client")
    monkeypatch.delenv("DHAN_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("DHAN_UNDER_SECURITY_ID", "13")
    payload = validate_dhan_runtime("NIFTY", dry_run=True)
    assert payload["final_verdict"] == "BROKER_RUNTIME_INVALID"
    assert payload["credentials"]["access_token_present"] is False


def test_dhan_health_check_handles_missing_underlying_security_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DHAN_CLIENT_ID", "client")
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "token123456")
    monkeypatch.delenv("DHAN_UNDERLYING_SECURITY_ID", raising=False)
    monkeypatch.delenv("DHAN_UNDER_SECURITY_ID", raising=False)
    payload = validate_dhan_runtime("NIFTY", dry_run=True)
    assert payload["final_verdict"] == "BROKER_RUNTIME_INVALID"
    assert payload["credentials"]["underlying_security_id_present"] is False


def test_dhan_live_order_blocked_when_allow_live_orders_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DhanClient.__new__(DhanClient)
    client.access_token = "token123456"
    client.client_id = "client"
    client._sdk_diagnostics = {"import_ok": True, "sdk_source": "local_path"}
    client._validated_live_security_ids = {"123"}
    client._health_passed_current_session = True
    client._last_success_ts = None
    client._last_api_error = ""
    client._live_readiness = {
        "sdk_import_ok": True,
        "sdk_source": "local_path",
        "client_id_present": True,
        "access_token_present": True,
        "underlying_security_id_present": True,
        "option_chain_ok": True,
        "atm_detected": True,
        "selected_contract_from_live_chain": True,
        "selected_security_id": "123",
        "selected_symbol": "NIFTYCE",
        "ltp_ok": True,
        "bid_ask_ok": True,
        "positions_ok": True,
        "holdings_ok": True,
        "last_success_timestamp": None,
        "last_error": "",
        "live_order_allowed": True,
    }
    client.resolve_exchange_token_symbol = lambda symbol, exchange_hint=None: ("NSE_FNO", "123", symbol)  # type: ignore[assignment]
    client._call_api = lambda *args, **kwargs: {}  # type: ignore[assignment]
    monkeypatch.setenv("SCALPER_BROKER", "dhan")
    monkeypatch.delenv("SCALPER_ALLOW_LIVE_ORDERS", raising=False)
    with pytest.raises(RuntimeError, match="SCALPER_ALLOW_LIVE_ORDERS=true is required"):
        client.place_order("NIFTY", "BUY", 1, order_mode="live")


def test_dhan_live_order_blocked_when_sdk_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DhanClient.__new__(DhanClient)
    client.access_token = "token123456"
    client.client_id = "client"
    client._sdk_diagnostics = {"import_ok": False, "sdk_source": "not_found"}
    client._validated_live_security_ids = {"123"}
    client._health_passed_current_session = False
    client._last_success_ts = None
    client._last_api_error = "sdk missing"
    client._live_readiness = {
        "sdk_import_ok": False,
        "sdk_source": "not_found",
        "client_id_present": True,
        "access_token_present": True,
        "underlying_security_id_present": True,
        "option_chain_ok": False,
        "atm_detected": False,
        "selected_contract_from_live_chain": False,
        "selected_security_id": "",
        "selected_symbol": "",
        "ltp_ok": False,
        "bid_ask_ok": False,
        "positions_ok": False,
        "holdings_ok": False,
        "last_success_timestamp": None,
        "last_error": "sdk missing",
        "live_order_allowed": False,
    }
    client.resolve_exchange_token_symbol = lambda symbol, exchange_hint=None: ("NSE_FNO", "123", symbol)  # type: ignore[assignment]
    monkeypatch.setenv("SCALPER_BROKER", "dhan")
    monkeypatch.setenv("SCALPER_ALLOW_LIVE_ORDERS", "true")
    with pytest.raises(RuntimeError, match="SDK import is not healthy"):
        client.place_order("NIFTY", "BUY", 1, order_mode="live")


def test_dhan_live_order_blocked_when_option_chain_validation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DhanClient.__new__(DhanClient)
    client.access_token = "token123456"
    client.client_id = "client"
    client._sdk_diagnostics = {"import_ok": True, "sdk_source": "local_path"}
    client._validated_live_security_ids = {"123"}
    client._health_passed_current_session = False
    client._last_success_ts = None
    client._last_api_error = "option chain failed"
    client._live_readiness = {
        "sdk_import_ok": True,
        "sdk_source": "local_path",
        "client_id_present": True,
        "access_token_present": True,
        "underlying_security_id_present": True,
        "option_chain_ok": False,
        "atm_detected": False,
        "selected_contract_from_live_chain": False,
        "selected_security_id": "",
        "selected_symbol": "",
        "ltp_ok": False,
        "bid_ask_ok": False,
        "positions_ok": False,
        "holdings_ok": False,
        "last_success_timestamp": None,
        "last_error": "option chain failed",
        "live_order_allowed": False,
    }
    client.resolve_exchange_token_symbol = lambda symbol, exchange_hint=None: ("NSE_FNO", "123", symbol)  # type: ignore[assignment]
    monkeypatch.setenv("SCALPER_BROKER", "dhan")
    monkeypatch.setenv("SCALPER_ALLOW_LIVE_ORDERS", "true")
    with pytest.raises(RuntimeError, match="live option chain validation has not passed"):
        client.place_order("NIFTY", "BUY", 1, order_mode="live")


def test_dhan_live_order_blocked_when_security_id_not_from_live_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DhanClient.__new__(DhanClient)
    client.access_token = "token123456"
    client.client_id = "client"
    client._sdk_diagnostics = {"import_ok": True, "sdk_source": "local_path"}
    client._validated_live_security_ids = {"999"}
    client._health_passed_current_session = True
    client._last_success_ts = None
    client._last_api_error = ""
    client._live_readiness = {
        "sdk_import_ok": True,
        "sdk_source": "local_path",
        "client_id_present": True,
        "access_token_present": True,
        "underlying_security_id_present": True,
        "option_chain_ok": True,
        "atm_detected": True,
        "selected_contract_from_live_chain": True,
        "selected_security_id": "999",
        "selected_symbol": "NIFTYCE",
        "ltp_ok": True,
        "bid_ask_ok": True,
        "positions_ok": True,
        "holdings_ok": True,
        "last_success_timestamp": None,
        "last_error": "",
        "live_order_allowed": True,
    }
    client.resolve_exchange_token_symbol = lambda symbol, exchange_hint=None: ("NSE_FNO", "123", symbol)  # type: ignore[assignment]
    monkeypatch.setenv("SCALPER_BROKER", "dhan")
    monkeypatch.setenv("SCALPER_ALLOW_LIVE_ORDERS", "true")
    with pytest.raises(RuntimeError, match="security_id was not validated"):
        client.place_order("NIFTY", "BUY", 1, order_mode="live")


def test_dhan_live_order_allowed_only_when_all_readiness_checks_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DhanClient.__new__(DhanClient)
    client.access_token = "token123456"
    client.client_id = "client"
    client._sdk_diagnostics = {"import_ok": True, "sdk_source": "local_path"}
    client._validated_live_security_ids = {"123"}
    client._health_passed_current_session = True
    client._last_success_ts = None
    client._last_api_error = ""
    client._live_readiness = {
        "sdk_import_ok": True,
        "sdk_source": "local_path",
        "client_id_present": True,
        "access_token_present": True,
        "underlying_security_id_present": True,
        "option_chain_ok": True,
        "atm_detected": True,
        "selected_contract_from_live_chain": True,
        "selected_security_id": "123",
        "selected_symbol": "NIFTYCE",
        "ltp_ok": True,
        "bid_ask_ok": True,
        "positions_ok": True,
        "holdings_ok": True,
        "last_success_timestamp": None,
        "last_error": "",
        "live_order_allowed": True,
    }
    client.resolve_exchange_token_symbol = lambda symbol, exchange_hint=None: ("NSE_FNO", "123", symbol)  # type: ignore[assignment]
    client._raw = SimpleNamespace(place_order=lambda **kwargs: {"orderId": "OID1", "status": "ok"})
    client._call_api = lambda *args, **kwargs: {"orderId": "OID1", "status": "ok"}  # type: ignore[assignment]
    monkeypatch.setenv("SCALPER_BROKER", "dhan")
    monkeypatch.setenv("SCALPER_ALLOW_LIVE_ORDERS", "true")
    order = client.place_order("NIFTY", "BUY", 1, order_mode="live")
    assert order.order_id == "OID1"


def test_dhan_token_is_masked_in_logs(capsys: pytest.CaptureFixture[str]) -> None:
    client = DhanClient.__new__(DhanClient)
    client.access_token = "abcdefghijklmnop"
    client._last_api_error = ""
    client._last_success_ts = None
    client._record_success = lambda: None  # type: ignore[assignment]
    client._record_error = lambda endpoint, exc: None  # type: ignore[assignment]
    out = client._call_api("test", lambda: {"status": "ok"})
    captured = capsys.readouterr().out
    assert "abcd...mnop" in captured
    assert "abcdefghijklmnop" not in captured
    assert out["status"] == "ok"


def test_mask_token_helper() -> None:
    assert mask_token("abcdefghijklmnop") == "abcd...mnop"


def test_dry_run_validation_never_places_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        client_id = "client"
        access_token = "token123456"
        _validated_live_security_ids = {"123"}

        def __init__(self):
            self.placed = False

        def sdk_diagnostics(self):
            return {"sdk_source": "local_path", "sdk_path": "C:\\sdk", "import_ok": True, "error": ""}

        def get_option_chain(self, symbol):
            return [
                {"symbol": "NIFTYCE", "token": "123", "option_type": "CE", "strike": 22500.0, "exchange": "NSE_FNO", "raw": {"underlying_spot_price": 22510.0}},
                {"symbol": "NIFTYPE", "token": "124", "option_type": "PE", "strike": 22500.0, "exchange": "NSE_FNO", "raw": {"underlying_spot_price": 22510.0}},
            ]

        def get_ltp(self, symbol):
            return 100.0

        def get_bid_ask(self, symbol, exchange_hint=None):
            return 99.0, 101.0, 100.0

        def get_open_positions(self):
            return []

        def get_holdings(self):
            return []

        def update_live_readiness(self, readiness):
            self._live_readiness = dict(readiness)
            self._live_readiness["live_order_allowed"] = True
            return self._live_readiness

        def live_readiness(self):
            return self._live_readiness

        def mark_health_check_passed(self, passed):
            self.passed = passed

        def last_api_error(self):
            return ""

        def last_success_timestamp(self):
            return None

        def _resolve_expiry(self, symbol):
            return "2026-06-25"

        def place_order(self, *args, **kwargs):
            self.placed = True
            raise AssertionError("place_order should not be called in dry-run validation")

    fake = FakeClient()
    monkeypatch.setenv("DHAN_CLIENT_ID", "client")
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "token123456")
    monkeypatch.setenv("DHAN_UNDER_SECURITY_ID", "13")
    monkeypatch.setattr("scripts.validate_broker_runtime.DhanClient", lambda cfg=None: fake)
    monkeypatch.setattr("scripts.validate_broker_runtime.validate_dhan_live_readiness", lambda symbol, client=None: fake.update_live_readiness({
        "sdk_import_ok": True,
        "sdk_source": "local_path",
        "client_id_present": True,
        "access_token_present": True,
        "underlying_security_id_present": True,
        "option_chain_ok": True,
        "atm_detected": True,
        "selected_contract_from_live_chain": True,
        "selected_security_id": "123",
        "selected_symbol": "NIFTYCE",
        "ltp_ok": True,
        "bid_ask_ok": True,
        "positions_ok": True,
        "holdings_ok": True,
        "last_success_timestamp": None,
        "last_error": "",
    }))
    payload = validate_dhan_runtime("NIFTY", dry_run=True)
    assert payload["paper_order_payload_construction"]["detail"].startswith("dry-run validation is read-only")
    assert fake.placed is False


def test_validate_dhan_live_readiness_returns_masked_safe_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        client_id = "client"
        access_token = "token123456"
        _validated_live_security_ids = {"123"}
        _last_success_ts = None
        _last_api_error = ""

        def get_option_chain(self, symbol):
            return [
                {"symbol": "NIFTYCE", "token": "123", "option_type": "CE", "strike": 22500.0, "exchange": "NSE_FNO", "raw": {"underlying_spot_price": 22510.0}},
                {"symbol": "NIFTYPE", "token": "124", "option_type": "PE", "strike": 22500.0, "exchange": "NSE_FNO", "raw": {"underlying_spot_price": 22510.0}},
            ]

        def get_bid_ask(self, symbol, exchange_hint=None):
            return 99.0, 101.0, 100.0

        def get_ltp(self, symbol):
            return 100.0

        def get_open_positions(self):
            return []

        def get_holdings(self):
            return []

        def update_live_readiness(self, readiness):
            self._live_readiness = dict(readiness)
            self._live_readiness["live_order_allowed"] = True
            return self._live_readiness

        def live_readiness(self):
            return self._live_readiness

    monkeypatch.setenv("DHAN_CLIENT_ID", "client")
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "token123456")
    monkeypatch.setenv("DHAN_UNDER_SECURITY_ID", "13")
    readiness = validate_dhan_live_readiness("NIFTY", client=FakeClient())
    serialized = str(readiness)
    assert "token123456" not in serialized


def test_dhan_runtime_report_and_exception_do_not_expose_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DHAN_CLIENT_ID", "client")
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "token123456")
    monkeypatch.delenv("DHAN_UNDER_SECURITY_ID", raising=False)
    payload = validate_dhan_runtime("NIFTY", dry_run=True)
    assert "token123456" not in str(payload)

    client = DhanClient.__new__(DhanClient)
    client.access_token = "token123456"
    client.client_id = "client"
    client._sdk_diagnostics = {"import_ok": False, "sdk_source": "not_found"}
    client._validated_live_security_ids = set()
    client._health_passed_current_session = False
    client._last_success_ts = None
    client._last_api_error = ""
    client._live_readiness = {
        "sdk_import_ok": False,
        "sdk_source": "not_found",
        "client_id_present": True,
        "access_token_present": True,
        "underlying_security_id_present": True,
        "option_chain_ok": False,
        "atm_detected": False,
        "selected_contract_from_live_chain": False,
        "selected_security_id": "",
        "selected_symbol": "",
        "ltp_ok": False,
        "bid_ask_ok": False,
        "positions_ok": False,
        "holdings_ok": False,
        "last_success_timestamp": None,
        "last_error": "sdk missing",
        "live_order_allowed": False,
    }
    client.resolve_exchange_token_symbol = lambda symbol, exchange_hint=None: ("NSE_FNO", "123", symbol)  # type: ignore[assignment]
    monkeypatch.setenv("SCALPER_BROKER", "dhan")
    monkeypatch.setenv("SCALPER_ALLOW_LIVE_ORDERS", "true")
    with pytest.raises(RuntimeError) as excinfo:
        client.place_order("NIFTY", "BUY", 1, order_mode="live")
    assert "token123456" not in str(excinfo.value)


def test_dhan_sdk_local_path_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSpec:
        origin = str(dhan_client._LOCAL_DHAN_SDK / "dhanhq" / "__init__.py")
        submodule_search_locations = [str(dhan_client._LOCAL_DHAN_SDK / "dhanhq")]

    fake_module = SimpleNamespace(__file__=str(dhan_client._LOCAL_DHAN_SDK / "dhanhq" / "__init__.py"))
    monkeypatch.setattr(dhan_client.importlib_util, "find_spec", lambda name: FakeSpec())
    monkeypatch.setitem(__import__("sys").modules, "dhanhq", fake_module)
    diagnostics = get_dhan_sdk_diagnostics()
    assert diagnostics["import_ok"] is True
    assert diagnostics["sdk_source"] == "local_path"
    assert str(dhan_client._LOCAL_DHAN_SDK) in diagnostics["sdk_path"]


def test_dhan_sdk_pip_package_fallback_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSpec:
        origin = r"C:\Python\Lib\site-packages\dhanhq\__init__.py"
        submodule_search_locations = [r"C:\Python\Lib\site-packages\dhanhq"]

    fake_module = SimpleNamespace(__file__=r"C:\Python\Lib\site-packages\dhanhq\__init__.py")
    monkeypatch.setattr(dhan_client.importlib_util, "find_spec", lambda name: FakeSpec())
    monkeypatch.setitem(__import__("sys").modules, "dhanhq", fake_module)
    diagnostics = get_dhan_sdk_diagnostics()
    assert diagnostics["import_ok"] is True
    assert diagnostics["sdk_source"] == "pip_package"


def test_dhan_sdk_missing_package_handled_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dhan_client.importlib_util, "find_spec", lambda name: None)
    monkeypatch.delitem(__import__("sys").modules, "dhanhq", raising=False)
    diagnostics = get_dhan_sdk_diagnostics()
    assert diagnostics["import_ok"] is False
    assert diagnostics["sdk_source"] == "not_found"
    assert diagnostics["error"]


def test_dhan_sdk_diagnostics_never_include_token_or_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "supersecrettoken1234")
    monkeypatch.setenv("DHAN_CLIENT_ID", "client-secret-like-value")
    diagnostics = get_dhan_sdk_diagnostics()
    serialized = str(diagnostics)
    assert "supersecrettoken1234" not in serialized
    assert "client-secret-like-value" not in serialized


def test_dhan_security_df_reads_with_string_dtype(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "dhan_security.csv"
    csv_path.write_text("SEM_SMST_SECURITY_ID,SEM_TRADING_SYMBOL\n13,NIFTY\n", encoding="utf-8")
    client = DhanClient.__new__(DhanClient)
    client._security_df = None
    monkeypatch.setattr(dhan_client, "Security", object())
    monkeypatch.setenv("DHAN_SECURITY_LIST_PATH", str(csv_path))
    df = DhanClient._ensure_security_df(client)
    assert str(df.iloc[0]["SEM_SMST_SECURITY_ID"]) == "13"


def test_dhan_login_renews_expiring_token(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DhanClient.__new__(DhanClient)
    client.client_id = "client"
    client.access_token = "old-token"
    client._last_success_ts = None
    client._last_api_error = ""
    client._context = SimpleNamespace(
        dhan_login=SimpleNamespace(
            renew_token=lambda token: {"accessToken": "renewed-token"},
            user_profile=lambda token: {"status": "ok"},
        )
    )
    client._raw = object()
    monkeypatch.setattr(dhan_client, "DhanContext", lambda client_id, access_token: SimpleNamespace(dhan_login=client._context.dhan_login))
    monkeypatch.setattr(dhan_client, "DhanHQClient", lambda context: object())
    monkeypatch.setattr(dhan_client, "is_dhan_token_refresh_due", lambda token, within_seconds=900: True)
    monkeypatch.setattr(dhan_client, "persist_dhan_access_token", lambda token: 0)
    client._record_success = lambda: None  # type: ignore[assignment]

    DhanClient.login(client, interactive=False)

    assert client.access_token == "renewed-token"


def test_dhan_login_regenerates_when_profile_check_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DhanClient.__new__(DhanClient)
    client.client_id = "client"
    client.access_token = "old-token"
    client._last_success_ts = None
    client._last_api_error = ""
    client._context = SimpleNamespace(
        dhan_login=SimpleNamespace(
            renew_token=lambda token: {"accessToken": "renewed-token"},
            user_profile=lambda token: (_ for _ in ()).throw(RuntimeError("expired")),
        )
    )
    client._raw = object()
    monkeypatch.setattr(dhan_client, "DhanContext", lambda client_id, access_token: SimpleNamespace(dhan_login=client._context.dhan_login))
    monkeypatch.setattr(dhan_client, "DhanHQClient", lambda context: object())
    monkeypatch.setattr(dhan_client, "is_dhan_token_refresh_due", lambda token, within_seconds=900: False)
    monkeypatch.setattr(dhan_client, "persist_dhan_access_token", lambda token: 0)
    monkeypatch.setattr(DhanClient, "_generate_token_from_local_creds", lambda self: "regenerated-token")
    client._record_success = lambda: None  # type: ignore[assignment]

    DhanClient.login(client, interactive=False)

    assert client.access_token == "regenerated-token"


def test_dhan_login_refresh_uses_shared_totp_secret_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DhanClient.__new__(DhanClient)
    client.client_id = "client"
    monkeypatch.setenv("DHAN_PIN", "1234")
    monkeypatch.delenv("DHAN_TOTP_SECRET", raising=False)
    monkeypatch.setenv("MSTOCK_TOTP_SECRET", "BASE32SECRET")
    captured = {}
    monkeypatch.setattr(
        dhan_client,
        "generate_dhan_access_token",
        lambda client_id, pin, totp_secret="", totp_code="": captured.update(
            {"client_id": client_id, "pin": pin, "totp_secret": totp_secret, "totp_code": totp_code}
        ) or "fresh-token",
    )

    token = DhanClient._generate_token_from_local_creds(client)

    assert token == "fresh-token"
    assert captured["totp_secret"] == "BASE32SECRET"


def test_dhan_login_refresh_prefers_secret_over_stale_code(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DhanClient.__new__(DhanClient)
    client.client_id = "client"
    monkeypatch.setenv("DHAN_PIN", "1234")
    monkeypatch.setenv("DHAN_TOTP_SECRET", "BASE32SECRET")
    monkeypatch.setenv("DHAN_TOTP_CODE", "111111")
    captured = {}
    monkeypatch.setattr(
        dhan_client,
        "generate_dhan_access_token",
        lambda client_id, pin, totp_secret="", totp_code="": captured.update(
            {"client_id": client_id, "pin": pin, "totp_secret": totp_secret, "totp_code": totp_code}
        ) or "fresh-token",
    )

    token = DhanClient._generate_token_from_local_creds(client)

    assert token == "fresh-token"
    assert captured["totp_secret"] == "BASE32SECRET"
    assert captured["totp_code"] == ""


def test_collector_dry_run_prints_dhan_sdk_source(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("DHAN_CLIENT_ID", "client")
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "token123456")
    monkeypatch.setenv("DHAN_UNDER_SECURITY_ID", "13")
    monkeypatch.setattr(
        collector,
        "get_dhan_sdk_diagnostics",
        lambda: {"sdk_source": "local_path", "sdk_path": r"C:\sdk\dhanhq\__init__.py", "import_ok": True, "error": ""},
    )
    monkeypatch.setattr(collector, "resolve_dhan_expiry", lambda *args, **kwargs: "2026-06-25")
    class FakeDhanClient:
        def option_chain(self, **kwargs):
            return [{"strike": 22500, "optionType": "CE", "securityId": "111", "ltp": 100.0}]

    monkeypatch.setattr(collector, "build_dhan_client", lambda: FakeDhanClient())
    args = SimpleNamespace(
        broker="dhan",
        symbol="NIFTY",
        expiry="",
        dry_run=True,
        debug_raw_payload=False,
        max_strikes_around_atm=20,
        output_dir="data/option_context_live",
        report_dir="reports",
    )
    caplog.set_level("INFO")
    payload = collector.collect_once(args, collector.logging.getLogger("test_collect"))
    assert payload["sdk_diagnostics"]["sdk_source"] == "local_path"
    assert any("Dhan SDK diagnostics" in rec.message and "local_path" in rec.message for rec in caplog.records)
