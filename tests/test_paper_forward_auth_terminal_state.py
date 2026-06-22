import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI


class _OK:
    def get_ltp(self, symbol):
        return 23172.5


class _Expired:
    def get_ltp(self, symbol):
        raise RuntimeError("IA401 token expired")


def test_token_present_unchecked_does_not_persist_after_success(monkeypatch):
    monkeypatch.setenv("MSTOCK_ACCESS_TOKEN", "abc")
    ui = object.__new__(ScalperUI)
    ui._client = _OK()

    status, error, spot = ui._pf_ensure_auth_terminal()

    assert status == "AUTH_OK"
    assert error == ""
    assert spot == 23172.5
    assert ui._pf_auth_validation_attempted is True
    assert ui._pf_last_auth_status == "AUTH_OK"


def test_token_present_unchecked_does_not_persist_after_expired(monkeypatch):
    monkeypatch.setenv("MSTOCK_ACCESS_TOKEN", "abc")
    ui = object.__new__(ScalperUI)
    ui._client = _Expired()

    status, error, _spot = ui._pf_ensure_auth_terminal()

    assert status.startswith("SESSION_EXPIRED:")
    assert "IA401" in error
    assert ui._pf_auth_validation_attempted is True
    assert ui._pf_last_auth_status.startswith("SESSION_EXPIRED:")


def test_token_change_resets_validation(monkeypatch):
    ui = object.__new__(ScalperUI)
    monkeypatch.setenv("MSTOCK_ACCESS_TOKEN", "abc")
    ui._pf_reset_auth_if_token_changed()
    ui._pf_auth_validation_attempted = True
    ui._pf_last_auth_status = "AUTH_OK"

    monkeypatch.setenv("MSTOCK_ACCESS_TOKEN", "def")
    ui._pf_reset_auth_if_token_changed()

    assert ui._pf_auth_validation_attempted is False
    assert ui._pf_last_auth_status == "TOKEN_PRESENT_UNCHECKED"
