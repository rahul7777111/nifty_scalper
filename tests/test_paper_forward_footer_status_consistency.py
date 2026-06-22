import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_forward_engine import PaperForwardDataStatus
from paper_forward_engine import PaperForwardAuthState
from ui import ScalperUI


def test_footer_cannot_show_none_when_candle_count_positive():
    status = PaperForwardDataStatus.from_snapshot(
        {"price": 24150, "broker_auth": "TOKEN_SET_NOT_VERIFIED", "source": "none", "candles": [{} for _ in range(100)]},
        [],
    )
    data = status.as_dict()
    assert data["candle_count"] == 100
    assert data["candle_source"] != "none"
    assert "candles=100" in status.footer_text()


def test_footer_cannot_show_unknown_when_token_set(monkeypatch):
    monkeypatch.setenv("MSTOCK_ACCESS_TOKEN", "dummy-token")
    status = PaperForwardDataStatus.from_snapshot({"price": 24150, "broker_auth": "UNKNOWN", "candles": []}, [])
    assert status.as_dict()["broker_auth"] in {"TOKEN_SET_NOT_VERIFIED", "TOKEN_PRESENT_UNCHECKED"}
    assert "UNKNOWN" not in status.footer_text()


def test_footer_auth_fields_preserve_auth_ok_runtime_state(monkeypatch):
    app = ScalperUI.__new__(ScalperUI)
    st = PaperForwardAuthState()
    st.status = "AUTH_OK"
    st.token_hash = "abc12345"
    st.client_created = True
    st.validation_attempted = True
    app._pf_auth_state = st
    app._pf_last_auth_status = "TOKEN_MISSING"
    app._pf_last_valid_auth_state = {"status": "AUTH_OK", "token_hash": "abc12345", "client_created": True}
    app._pf_poll_cycle_id = 7
    app._pf_runtime = object()
    app._client = object()
    monkeypatch.setattr(app, "_pf_enforce_auth_timeout", lambda: None)
    fields = app._pf_auth_status_fields()
    assert fields["broker_auth"] == "AUTH_OK"
    assert fields["token_present"] is True
    assert fields["client_created"] is True
