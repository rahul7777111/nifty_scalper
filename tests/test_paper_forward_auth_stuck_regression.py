import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI
from paper_forward_engine import PaperForwardEngine, PaperForwardDataStatus


class _OKClient:
    def get_ltp(self, symbol):
        return 23172.5


class _ExpiredClient:
    def get_ltp(self, symbol):
        raise RuntimeError("IA401 token expired")


class _ExplodingClient:
    def get_ltp(self, symbol):
        raise RuntimeError("connection refused - broker init fail")


def test_auth_stuck_regression_verification_called_once_then_terminal(monkeypatch):
    monkeypatch.setenv("MSTOCK_ACCESS_TOKEN", "tok123")
    ui = object.__new__(ScalperUI)
    ui._client = _ExpiredClient()
    # ensure clean state
    for k in list(ui.__dict__.keys()):
        if k.startswith("_pf_"):
            delattr(ui, k)
    ui._pf_poll_cycle_id = 0
    ui._pf_auth_validation_attempted = False
    ui._pf_auth_validation_in_progress = False
    ui._pf_last_auth_status = ""
    ui._pf_last_auth_error = ""
    ui._pf_auth_token_hash_prefix = ""

    # track calls to the actual verify path
    call_count = {"verify": 0, "validate": 0}
    orig_validate = ui._pf_validate_broker_auth_readonly

    def counting_validate(client):
        call_count["validate"] += 1
        # simulate the side effect of real verify by setting status like expired
        ui._pf_last_auth_status = "SESSION_EXPIRED:IA401"
        ui._pf_last_auth_error = "IA401 token expired"
        ui._pf_last_auth_endpoint = "get_ltp"
        ui._pf_auth_validation_attempted = True
        ui._pf_auth_validation_in_progress = False
        return "SESSION_EXPIRED:IA401", "IA401 token expired", None

    monkeypatch.setattr(ui, "_pf_validate_broker_auth_readonly", counting_validate)

    # also ensure get_or_create returns our client
    def fake_get_client():
        return ui._client, "test_client", ""

    monkeypatch.setattr(ui, "_pf_get_or_create_data_client", fake_get_client)

    # Simulate 3 poll cycles (each would call ensure)
    s1, e1, _ = ui._pf_ensure_auth_terminal(force=False)
    s2, e2, _ = ui._pf_ensure_auth_terminal(force=False)
    s3, e3, _ = ui._pf_ensure_auth_terminal(force=False)

    assert call_count["validate"] == 1, f"verification (validate) must be called exactly once, was {call_count['validate']}"
    assert s1.startswith("SESSION_EXPIRED")
    assert s2.startswith("SESSION_EXPIRED")
    assert s3.startswith("SESSION_EXPIRED")
    assert ui._pf_last_auth_status.startswith("SESSION_EXPIRED:")
    assert ui._pf_auth_validation_attempted is True
    assert ui._pf_auth_validation_in_progress is False
    assert ui._pf_last_auth_status != "TOKEN_PRESENT_UNCHECKED"
    assert ui._pf_last_auth_status != "VERIFYING_BROKER_TOKEN"

    # Now test that evals do not increase: use a dummy engine, call publish 3 times with pending (but after terminal)
    # after the above, status is terminal fail, so publish should not call on_market
    class DummyEngine:
        def __init__(self):
            self.calls = 0
            self._evaluations_count = 0
            self._auth_wait_cycles = 0
        def on_market_snapshot(self, snap, chain):
            self.calls += 1
            self._evaluations_count += 16  # would be 16 cands
            return []
        def get_status_table(self):
            return []
        def write_jsonl(self, d): pass
        def write_summary(self): pass

    ui.pf_engine = DummyEngine()
    ui._pf_refresh_table = lambda r: None
    ui._pf_pending_auth_snapshot_published = False
    ui._pf_last_auth_snapshot_key = None

    snap_bad = {
        "broker_auth": ui._pf_last_auth_status,
        "auth_error": ui._pf_last_auth_error,
        "auth_validation_attempted": True,
        "token_hash_prefix": ui._pf_token_hash_prefix(),
        "spot": 23172.5,
    }
    ui._publish_market_snapshot_to_paper_forward(dict(snap_bad), [])
    ui._publish_market_snapshot_to_paper_forward(dict(snap_bad), [])
    ui._publish_market_snapshot_to_paper_forward(dict(snap_bad), [])

    # because attempted and not AUTH_OK, should not have called on_market more than 0 times (or at most 1 before guard)
    # our guard skips for auth_not_ok + attempted
    assert ui.pf_engine.calls <= 1, f"on_market should not be called repeatedly for terminal fail auth; calls={ui.pf_engine.calls}"
    # evals should not have grown by 16*3
    assert ui.pf_engine._evaluations_count <= 16, f"evals must not climb 16 per cycle while auth terminal fail; got {ui.pf_engine._evaluations_count}"


def test_auth_stuck_regression_exception_becomes_terminal_not_stuck(monkeypatch):
    monkeypatch.setenv("MSTOCK_ACCESS_TOKEN", "tokboom")
    ui = object.__new__(ScalperUI)
    ui._client = _ExplodingClient()
    for k in list(ui.__dict__.keys()):
        if k.startswith("_pf_"):
            delattr(ui, k)
    ui._pf_poll_cycle_id = 0
    ui._pf_auth_validation_attempted = False
    ui._pf_auth_validation_in_progress = False
    ui._pf_last_auth_status = ""
    ui._pf_last_auth_error = ""

    def exploding_get_client():
        return ui._client, "boom", ""

    monkeypatch.setattr(ui, "_pf_get_or_create_data_client", exploding_get_client)

    # patch validate to raise to simulate exception path inside ensure
    def raise_in_validate(c):
        raise RuntimeError("verify blew up")

    monkeypatch.setattr(ui, "_pf_validate_broker_auth_readonly", raise_in_validate)

    s, e, _ = ui._pf_ensure_auth_terminal(force=False)

    assert ui._pf_auth_validation_attempted is True
    assert ui._pf_auth_validation_in_progress is False
    assert ui._pf_last_auth_status.startswith("AUTH_FAILED") or ui._pf_last_auth_status == "BROKER_CLIENT_INIT_FAILED"
    assert "AUTH_OK" not in ui._pf_last_auth_status
    assert ui._pf_last_auth_status != "VERIFYING_BROKER_TOKEN"
    assert ui._pf_last_auth_status != "TOKEN_PRESENT_UNCHECKED"


def test_auth_stuck_regression_status_refresh_does_not_overwrite_terminal(monkeypatch):
    monkeypatch.setenv("MSTOCK_ACCESS_TOKEN", "tokkeep")
    ui = object.__new__(ScalperUI)
    for k in list(ui.__dict__.keys()):
        if k.startswith("_pf_"):
            delattr(ui, k)
    ui._pf_poll_cycle_id = 0
    ui._pf_auth_validation_attempted = True
    ui._pf_auth_validation_in_progress = False
    ui._pf_last_auth_status = "AUTH_FAILED:simulated"
    ui._pf_last_auth_error = "sim fail"
    ui._pf_auth_token_hash_prefix = ui._pf_token_hash_prefix("tokkeep")

    # simulate footer / schedule / apply paths that might recompute
    # call _pf_auth_status_fields (used by footer/status)
    f = ui._pf_auth_status_fields()
    # call auth terminal status (light)
    term = ui._pf_auth_terminal_status()
    # simulate a status refresh path without full _build (which requires tk vars and can recurse on bare UI object)
    # directly exercise ensure (with force=False, same token + attempted should early return terminal)
    s_ref, e_ref, _ = ui._pf_ensure_auth_terminal(force=False)
    # direct from status
    ds = PaperForwardDataStatus(
        broker_auth=ui._pf_last_auth_status,
        auth_error=ui._pf_last_auth_error,
        auth_validation_attempted=True,
    ).as_dict()

    assert ui._pf_last_auth_status == "AUTH_FAILED:simulated"
    assert term == "AUTH_FAILED:simulated" or term.startswith("AUTH_FAILED")
    assert s_ref == "AUTH_FAILED:simulated" or s_ref.startswith("AUTH_FAILED")
    assert f.get("auth_validation_attempted") is True
    # crucially, the ds and state must not have forced it back to UNCHECKED
    assert ds.get("broker_auth") != "TOKEN_PRESENT_UNCHECKED"
    assert ui._pf_last_auth_status != "TOKEN_PRESENT_UNCHECKED"
    assert ui._pf_last_auth_status != "VERIFYING_BROKER_TOKEN"
