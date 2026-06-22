import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI


class _ClientOK:
    def get_ltp(self, symbol):
        return 23172.5


class _ClientFail:
    def get_ltp(self, symbol):
        raise RuntimeError("IA401 token expired")


def test_token_set_not_verified_does_not_persist_after_success(monkeypatch):
    monkeypatch.setenv("MSTOCK_ACCESS_TOKEN", "token")
    ui = object.__new__(ScalperUI)

    status, error, spot = ui._pf_validate_broker_auth_readonly(_ClientOK())

    assert status == "AUTH_OK"
    assert error == ""
    assert spot == 23172.5
    assert ui._pf_auth_validation_attempted is True
    assert ui._pf_last_auth_status == "AUTH_OK"
    assert ui._pf_last_auth_endpoint.startswith("get_ltp:")


def test_auth_failure_sets_exact_error(monkeypatch):
    monkeypatch.setenv("MSTOCK_ACCESS_TOKEN", "token")
    ui = object.__new__(ScalperUI)

    status, error, spot = ui._pf_validate_broker_auth_readonly(_ClientFail())

    assert status.startswith("SESSION_EXPIRED:")
    assert "IA401 token expired" in status
    assert "IA401 token expired" in error
    assert spot is None
    assert ui._pf_auth_validation_attempted is True
