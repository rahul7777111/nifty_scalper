from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ui
import dhan_auth


class DummyVar:
    def __init__(self, value=None) -> None:
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


class DummyButton:
    def __init__(self) -> None:
        self.state = {}

    def configure(self, **kwargs) -> None:
        self.state.update(kwargs)


class DummyLabel:
    def __init__(self) -> None:
        self.state = {}

    def configure(self, **kwargs) -> None:
        self.state.update(kwargs)


class ImmediateThread:
    def __init__(self, target=None, daemon=None, **kwargs) -> None:
        self._target = target
        self._alive = False

    def start(self) -> None:
        self._alive = True
        try:
            if self._target is not None:
                self._target()
        finally:
            self._alive = False

    def is_alive(self) -> bool:
        return self._alive


def _build_app(tmp_path: Path) -> ui.ScalperUI:
    app = object.__new__(ui.ScalperUI)
    app.tk = SimpleNamespace()
    app._cred_path = tmp_path / "credentials.json"
    app.broker_var = DummyVar("mstock")
    app.api_key_var = DummyVar("api")
    app.username_var = DummyVar("user")
    app.password_var = DummyVar("pass")
    app.access_token_var = DummyVar("mstock-token")
    app.totp_secret_var = DummyVar("")
    app.totp_code_var = DummyVar("")
    app.dhan_client_id_var = DummyVar("")
    app.dhan_access_token_var = DummyVar("")
    app.dhan_underlying_security_id_var = DummyVar("")
    app.dhan_pin_var = DummyVar("")
    app.dhan_api_key_var = DummyVar("")
    app.dhan_api_secret_var = DummyVar("")
    app.strategy_var = DummyVar("directional")
    app.timeframe_var = DummyVar("1m")
    app.distance_var = DummyVar("50")
    app.live_var = DummyVar(False)
    app.remember_var = DummyVar(True)
    app.edit_creds_var = DummyVar(False)
    app.status_var = DummyVar("Idle")
    app._app_status_var = DummyVar("Ready.")
    app.btn_login_totp = DummyButton()
    app.totp_secret_label = DummyLabel()
    app.totp_code_label = DummyLabel()
    app.totp_code_hint_label = DummyLabel()
    app.btn_start = DummyButton()
    app.btn_stop = DummyButton()
    app._bot_thread = None
    app._bot_stop = SimpleNamespace(clear=lambda: None, set=lambda: None)
    app._bot_last_error = ""
    app._client = None
    app._scalper = None
    app.after = lambda delay, fn=None: fn() if fn else None
    app.ui_call = lambda fn, *args, **kwargs: fn(*args, **kwargs)
    app._reset_trade_log = lambda: None
    app._prune_stale_open_rows_when_idle = lambda force=False: None
    app._sync_trade_log_visibility = lambda: None
    return app


def test_ui_imports_successfully() -> None:
    assert hasattr(ui, "ScalperUI")


def test_default_broker_is_mstock(tmp_path: Path) -> None:
    app = _build_app(tmp_path)
    assert ui.ScalperUI._selected_broker(app) == "mstock"


def test_selected_broker_uses_env_dhan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = _build_app(tmp_path)
    del app.broker_var
    monkeypatch.setenv("SCALPER_BROKER", "dhan")
    assert ui.ScalperUI._selected_broker(app) == "dhan"


def test_broker_selector_values_are_supported(tmp_path: Path) -> None:
    app = _build_app(tmp_path)
    app.broker_var.set("dhan")
    assert ui.ScalperUI._selected_broker(app) == "dhan"
    app.broker_var.set("mstock")
    assert ui.ScalperUI._selected_broker(app) == "mstock"


def test_mstock_and_dhan_credentials_are_preserved_on_save(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = _build_app(tmp_path)
    app._cred_path.write_text(
        json.dumps(
            {
                "broker": "mstock",
                "api_key": "old-api",
                "username": "old-user",
                "password": "old-pass",
                "dhan_client_id": "existing-client",
                "dhan_access_token": "existing-token",
                "dhan_pin": "1111",
            }
        ),
        encoding="utf-8",
    )
    persisted = {}
    monkeypatch.setattr(ui, "persist_settings_env", lambda values, unset_keys: persisted.update(values) or tmp_path / ".scalper.env")
    app.broker_var.set("dhan")
    app.api_key_var.set("")
    app.username_var.set("")
    app.password_var.set("")
    app.dhan_client_id_var.set("new-client")
    app.dhan_access_token_var.set("new-token")
    app.dhan_underlying_security_id_var.set("13")
    app.dhan_pin_var.set("2222")

    ui.ScalperUI._on_save_credentials(app)

    saved = json.loads(app._cred_path.read_text(encoding="utf-8"))
    assert saved["api_key"] == "old-api"
    assert saved["username"] == "old-user"
    assert saved["password"] == "old-pass"
    assert saved["dhan_client_id"] == "new-client"
    assert saved["dhan_access_token"] == "new-token"
    assert saved["dhan_pin"] == "2222"
    assert persisted["SCALPER_BROKER"] == "dhan"
    assert persisted["DHAN_UNDERLYING_SECURITY_ID"] == "13"


def test_load_prefilled_credentials_reads_dhan_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = _build_app(tmp_path)
    app._cred_path.write_text(
        json.dumps(
            {
                "broker": "dhan",
                "api_key": "api",
                "username": "user",
                "password": "pass",
                    "dhan_client_id": "cid",
                    "dhan_access_token": "secret-token",
                    "dhan_underlying_security_id": "13",
                    "dhan_pin": "4321",
                }
            ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ui, "persist_settings_env", lambda *args, **kwargs: tmp_path / ".scalper.env")

    ui.ScalperUI._load_prefilled_credentials(app)

    assert app.broker_var.get() == "dhan"
    assert app.dhan_client_id_var.get() == "cid"
    assert app.dhan_access_token_var.get() == "secret-token"
    assert app.dhan_underlying_security_id_var.get() == "13"
    assert app.dhan_pin_var.get() == "4321"


def test_dhan_login_path_does_not_call_mstock_totp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = _build_app(tmp_path)
    app.broker_var.set("dhan")
    app.dhan_client_id_var.set("cid")
    app.dhan_access_token_var.set("abcdefghijklmnop")
    called = {"totp": 0}
    monkeypatch.setattr(ui, "login_with_totp", lambda *args, **kwargs: called.__setitem__("totp", called["totp"] + 1))

    ui.ScalperUI._on_login_totp(app)

    assert called["totp"] == 0
    assert "abcd...mnop" in app.status_var.get()
    assert "abcdefghijklmnop" not in app.status_var.get()


def test_dhan_login_can_auto_generate_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = _build_app(tmp_path)
    app.broker_var.set("dhan")
    app.dhan_client_id_var.set("cid")
    app.dhan_pin_var.set("1234")
    app.totp_code_var.set("123456")
    app.dhan_access_token_var.set("")
    monkeypatch.setattr(ui, "generate_dhan_access_token", lambda client_id, pin, totp_secret="", totp_code="": "generated-token-1234")
    monkeypatch.setattr(ui, "persist_dhan_access_token", lambda token: 0)

    ui.ScalperUI._on_login_totp(app)

    assert app.dhan_access_token_var.get() == "generated-token-1234"
    assert "gene...1234" in app.status_var.get()


def test_ensure_dhan_session_ready_refreshes_expiring_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = _build_app(tmp_path)
    app.broker_var.set("dhan")
    app.dhan_client_id_var.set("cid")
    app.dhan_pin_var.set("1234")
    app.totp_code_var.set("123456")
    app.dhan_access_token_var.set("expiring-token")
    monkeypatch.setattr(ui, "is_dhan_token_refresh_due", lambda token, within_seconds=900: True)
    monkeypatch.setattr(ui, "generate_dhan_access_token", lambda client_id, pin, totp_secret="", totp_code="": "fresh-token-9999")
    monkeypatch.setattr(ui, "persist_dhan_access_token", lambda token: 0)

    token = ui.ScalperUI._ensure_dhan_session_ready(app)

    assert token == "fresh-token-9999"
    assert app.dhan_access_token_var.get() == "fresh-token-9999"


def test_ensure_dhan_session_ready_rotates_legacy_token_when_creds_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_app(tmp_path)
    app.broker_var.set("dhan")
    app.dhan_client_id_var.set("cid")
    app.dhan_pin_var.set("1234")
    app.totp_code_var.set("123456")
    app.dhan_access_token_var.set("legacy-token")
    monkeypatch.delenv("DHAN_ACCESS_TOKEN_GENERATED_AT", raising=False)
    monkeypatch.setattr(ui, "is_dhan_token_refresh_due", lambda token, within_seconds=900: False)
    monkeypatch.setattr(ui, "generate_dhan_access_token", lambda client_id, pin, totp_secret="", totp_code="": "fresh-token-legacy")
    monkeypatch.setattr(ui, "persist_dhan_access_token", lambda token: 0)

    token = ui.ScalperUI._ensure_dhan_session_ready(app)

    assert token == "fresh-token-legacy"
    assert app.dhan_access_token_var.get() == "fresh-token-legacy"


def test_sync_broker_ui_updates_button_text(tmp_path: Path) -> None:
    app = _build_app(tmp_path)
    app.broker_var.set("dhan")
    ui.ScalperUI._sync_broker_ui(app)
    assert app.btn_login_totp.state["text"] == "Generate/Use Dhan Token"
    assert "m.Stock or Dhan" in app.totp_secret_label.state["text"]
    assert "m.Stock or Dhan" in app.totp_code_label.state["text"]
    app.broker_var.set("mstock")
    ui.ScalperUI._sync_broker_ui(app)
    assert app.btn_login_totp.state["text"] == "Login TOTP"
    assert app.totp_secret_label.state["text"] == "TOTP Secret"


def test_on_start_uses_mstock_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = _build_app(tmp_path)
    created = {}
    monkeypatch.setattr(ui.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(ui, "load_api_config", lambda: "api-cfg")
    monkeypatch.setattr(ui, "load_strategy_config", lambda: SimpleNamespace(underlying="NIFTY", timeframe="1m", enable_live_trading=False))
    monkeypatch.setattr(ui, "NiftyScalper", lambda client, cfg, event_sink=None, on_tick=None: SimpleNamespace(run_forever=lambda stop_event=None: None))
    monkeypatch.setattr(ui, "MStockTypeBClient", lambda cfg: created.setdefault("client", SimpleNamespace(login=lambda: created.setdefault("login", "mstock"), _resolve_token_from_instruments=lambda *args: "26000")))
    monkeypatch.setattr(ui.messagebox, "showerror", lambda *args, **kwargs: pytest.fail("showerror should not be called"))

    ui.ScalperUI._on_start(app)

    assert created["login"] == "mstock"


def test_on_start_uses_dhan_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = _build_app(tmp_path)
    app.broker_var.set("dhan")
    app.access_token_var.set("")
    app.dhan_client_id_var.set("cid")
    app.dhan_access_token_var.set("abcdefghijklmnop")
    built = {}
    monkeypatch.setattr(ui.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(ui, "load_api_config", lambda: "api-cfg")
    monkeypatch.setattr(ui, "load_strategy_config", lambda: SimpleNamespace(underlying="NIFTY", timeframe="1m", enable_live_trading=False))
    monkeypatch.setattr(ui, "NiftyScalper", lambda client, cfg, event_sink=None, on_tick=None: SimpleNamespace(run_forever=lambda stop_event=None: None))
    monkeypatch.setattr(ui.ScalperUI, "_build_dhan_client", lambda self, strat_cfg=None: built.setdefault("client", SimpleNamespace(access_token="abcdefghijklmnop", login=lambda interactive=False: built.setdefault("login", interactive))))
    monkeypatch.setattr(ui.messagebox, "showerror", lambda *args, **kwargs: pytest.fail("showerror should not be called"))

    ui.ScalperUI._on_start(app)

    assert built["login"] is False
    assert app._selected_broker() == "dhan"


def test_missing_dhan_sdk_does_not_crash_ui_import() -> None:
    assert hasattr(ui, "ScalperUI")


def test_mask_secret_never_exposes_full_token(tmp_path: Path) -> None:
    app = _build_app(tmp_path)
    masked = ui.ScalperUI._mask_secret(app, "abcdefghijklmnop")
    assert masked == "abcd...mnop"
    assert "abcdefghijklmnop" not in masked


def test_generate_dhan_access_token_uses_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLogin:
        def __init__(self, client_id) -> None:
            self.client_id = client_id

        def generate_token(self, pin, totp):
            assert self.client_id == "cid"
            assert pin == "1234"
            assert totp == "654321"
            return {"accessToken": "sdk-token"}

    monkeypatch.setattr(dhan_auth, "DhanLogin", FakeLogin)
    token = dhan_auth.generate_dhan_access_token("cid", "1234", totp_code="654321")
    assert token == "sdk-token"


def test_generate_dhan_access_token_normalizes_dhan_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeLogin:
        def __init__(self, client_id) -> None:
            self.client_id = client_id

        def generate_token(self, pin, totp):
            captured["pin"] = pin
            return {"accessToken": "sdk-token"}

    monkeypatch.setattr(dhan_auth, "DhanLogin", FakeLogin)
    token = dhan_auth.generate_dhan_access_token("cid", "12 34-56", totp_code="654321")

    assert token == "sdk-token"
    assert captured["pin"] == "123456"


def test_generate_dhan_access_token_rejects_non_digit_pin() -> None:
    with pytest.raises(RuntimeError, match="digits only"):
        dhan_auth.generate_dhan_access_token("cid", "api-secret", totp_code="654321")


def test_generate_dhan_access_token_surfaces_invalid_pin_action(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLogin:
        def __init__(self, client_id) -> None:
            self.client_id = client_id

        def generate_token(self, pin, totp):
            return {"status": "failure", "message": "Invalid PIN"}

    monkeypatch.setattr(dhan_auth, "DhanLogin", FakeLogin)
    with pytest.raises(RuntimeError, match="Dhan rejected the PIN"):
        dhan_auth.generate_dhan_access_token("cid", "123456", totp_code="654321")


def test_generate_dhan_access_token_accepts_message_jwt(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLogin:
        def __init__(self, client_id) -> None:
            self.client_id = client_id

        def generate_token(self, pin, totp):
            return {"status": "success", "message": "aaa.bbb.ccc"}

    monkeypatch.setattr(dhan_auth, "DhanLogin", FakeLogin)
    token = dhan_auth.generate_dhan_access_token("cid", "1234", totp_code="654321")
    assert token == "aaa.bbb.ccc"


def test_generate_dhan_access_token_surfaces_broker_message(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLogin:
        def __init__(self, client_id) -> None:
            self.client_id = client_id

        def generate_token(self, pin, totp):
            return {"status": "failure", "message": "Invalid TOTP"}

    monkeypatch.setattr(dhan_auth, "DhanLogin", FakeLogin)
    try:
        dhan_auth.generate_dhan_access_token("cid", "1234", totp_code="654321")
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Invalid TOTP" in str(exc)


def test_generate_dhan_access_token_retries_invalid_manual_code_with_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class FakeLogin:
        def __init__(self, client_id) -> None:
            self.client_id = client_id

        def generate_token(self, pin, totp):
            calls.append(totp)
            if len(calls) == 1:
                return {"status": "failure", "message": "Invalid TOTP"}
            return {"accessToken": "fresh-token"}

    monkeypatch.setattr(dhan_auth, "DhanLogin", FakeLogin)
    monkeypatch.setattr(dhan_auth, "get_totp_from_secret", lambda secret: "222222")
    monkeypatch.setattr(dhan_auth.time, "time", lambda: 100)

    token = dhan_auth.generate_dhan_access_token(
        "cid",
        "1234",
        totp_secret="BASE32SECRET",
        totp_code="111111",
    )

    assert token == "fresh-token"
    assert calls == ["111111", "222222"]


def test_decode_dhan_jwt_and_expiry_helpers() -> None:
    token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJleHAiOjE5OTk5OTk5OTksImRoYW5DbGllbnRJZCI6IjExMSJ9.signature"
    payload = dhan_auth.decode_dhan_jwt(token)
    assert payload["exp"] == 1999999999
    assert payload["dhanClientId"] == "111"
    assert dhan_auth.dhan_token_expiry_epoch(token) == 1999999999


def test_dhan_refresh_due_for_opaque_token_after_24h(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DHAN_ACCESS_TOKEN_GENERATED_AT", "1000")
    monkeypatch.setattr(dhan_auth.time, "time", lambda: 1000 + 24 * 60 * 60)

    assert dhan_auth.is_dhan_token_refresh_due("opaque-token", within_seconds=0)


def test_dhan_live_chart_initial_candles_do_not_call_mstock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = _build_app(tmp_path)
    app.broker_var.set("dhan")
    app._dash_candles_var = DummyVar("")
    app._dash_last_tick_var = DummyVar("")
    app._publish_market_snapshot_to_paper_forward = lambda: None
    app.live_chart_plugin = SimpleNamespace(pushed=[])
    app.live_chart_plugin.push_candles = lambda rows: app.live_chart_plugin.pushed.append(rows)
    fake = SimpleNamespace(
        fetch_index_candles=lambda token, exchange="IDX_I", limit=100, timeframe="1m": (
            [
                {
                    "timestamp": "2026-06-15 09:15:00",
                    "open": 1,
                    "high": 2,
                    "low": 0.5,
                    "close": 1.5,
                    "volume": 100,
                }
            ],
            "1m",
        )
    )
    monkeypatch.setattr(ui.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(ui.ScalperUI, "_build_dhan_client", lambda self, strat_cfg=None: fake)
    monkeypatch.setattr(ui, "MStockTypeBClient", lambda cfg: pytest.fail("m.Stock client must not be used for Dhan chart candles"))

    ui.ScalperUI._fetch_and_push_initial_candles(app)

    assert app._latest_candles
    assert app._chart_candles == app._latest_candles
    assert app.candles == app._latest_candles
    assert app.live_chart_plugin.pushed


def test_load_prefilled_credentials_keeps_non_empty_gui_values_over_blank_saved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_app(tmp_path)
    app._cred_path.write_text(
        json.dumps(
            {
                "broker": "dhan",
                "dhan_client_id": "",
                "dhan_access_token": "",
                "dhan_underlying_security_id": "",
            }
        ),
        encoding="utf-8",
    )
    app.dhan_client_id_var.set("gui-client")
    app.dhan_access_token_var.set("gui-token")
    app.dhan_underlying_security_id_var.set("13")
    monkeypatch.setenv("DHAN_CLIENT_ID", "env-client")
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "env-token")
    monkeypatch.setattr(ui, "persist_settings_env", lambda *args, **kwargs: tmp_path / ".scalper.env")

    ui.ScalperUI._load_prefilled_credentials(app)

    assert app.dhan_client_id_var.get() == "gui-client"
    assert app.dhan_access_token_var.get() == "gui-token"
    assert app.dhan_underlying_security_id_var.get() == "13"


def test_paper_forward_config_detects_active_candidate(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    cfg = tmp_path / "paper_forward_candidates.json"
    cfg.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "candidate-1",
                        "enabled": True,
                        "active": True,
                        "artifact_dir": str(artifact_dir),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = ui.diagnose_paper_forward_candidate_config(cfg)

    assert result["active"] == 1
    assert result["enabled"] == 1
    assert result["active_candidate_id"] == "candidate-1"


def test_paper_forward_config_reports_artifact_not_found(tmp_path: Path) -> None:
    cfg = tmp_path / "paper_forward_candidates.json"
    cfg.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "candidate-1",
                        "enabled": False,
                        "disabled_reason": "ARTIFACT_NOT_FOUND",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = ui.diagnose_paper_forward_candidate_config(cfg)

    assert result["active"] == 0
    assert result["reason"] == "ARTIFACT_NOT_FOUND"
