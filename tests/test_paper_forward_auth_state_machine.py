import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_forward_engine import PaperForwardDataStatus
from ui import ScalperUI


def test_token_present_unchecked_transitions_to_auth_ok(monkeypatch):
    monkeypatch.setenv("MSTOCK_ACCESS_TOKEN", "token")

    class Client:
        def get_ltp(self, symbol):
            return 23257.95

    ui = object.__new__(ScalperUI)
    status, error, spot = ui._pf_validate_broker_auth_readonly(Client())

    assert status == "AUTH_OK"
    assert error == ""
    assert spot == 23257.95
    assert ui._pf_auth_validation_attempted is True
    assert ui._pf_auth_validation_in_progress is False


def test_token_present_unchecked_transitions_to_session_expired(monkeypatch):
    monkeypatch.setenv("MSTOCK_ACCESS_TOKEN", "token")

    class Client:
        def get_ltp(self, symbol):
            raise RuntimeError("IA401 unauthorized expired token")

    ui = object.__new__(ScalperUI)
    status, error, _spot = ui._pf_validate_broker_auth_readonly(Client())

    assert status.startswith("SESSION_EXPIRED:")
    assert "IA401" in error
    assert ui._pf_auth_failure_count == 1


def test_verifying_state_requires_in_progress():
    ui = object.__new__(ScalperUI)
    assert ui._pf_status_label_from_data_status({
        "broker_auth": "VERIFYING_BROKER_TOKEN",
        "auth_validation_in_progress": True,
        "auth_validation_attempted": False,
        "data_quality_status": "TOKEN_NOT_VERIFIED",
    }) == "VERIFYING_BROKER_TOKEN"

    assert ui._pf_status_label_from_data_status({
        "broker_auth": "TOKEN_PRESENT_UNCHECKED",
        "auth_validation_in_progress": False,
        "auth_validation_attempted": True,
        "data_quality_status": "TOKEN_NOT_VERIFIED",
    }) == "BROKER_AUTH_FAILED"


def test_data_status_tracks_auth_counters():
    ds = PaperForwardDataStatus(
        broker_auth="AUTH_OK",
        auth_validation_attempted=True,
        auth_validation_endpoint="get_ltp:NIFTY",
        auth_success_count=2,
        auth_failure_count=1,
        spot=23257.95,
        option_rows=24,
        candle_count=100,
        candle_source="mstock_historical",
    ).as_dict()

    assert ds["data_quality_status"] == "DATA_OK"
    assert ds["auth_success_count"] == 2
    assert ds["auth_failure_count"] == 1
