import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI


def test_market_fetch_skipped_until_auth_ok(monkeypatch):
    ui = object.__new__(ScalperUI)
    monkeypatch.setattr(ui, "_pf_ensure_auth_terminal", lambda force=False: ("SESSION_EXPIRED:IA401", "IA401", None))
    monkeypatch.setattr(ui, "_pf_auth_status_fields", lambda: {
        "auth_validation_attempted": True,
        "auth_validation_in_progress": False,
        "token_hash_prefix": "abc",
    })
    called = {"chain": 0, "candles": 0, "published": None}
    monkeypatch.setattr(ui, "_pf_fetch_full_option_chain_readonly", lambda *a, **k: called.__setitem__("chain", called["chain"] + 1))
    monkeypatch.setattr(ui, "_pf_fetch_candles_readonly", lambda *a, **k: called.__setitem__("candles", called["candles"] + 1))
    monkeypatch.setattr(ui, "_pf_apply_data_status_to_ui", lambda status: None)
    monkeypatch.setattr(ui, "_publish_market_snapshot_to_paper_forward", lambda snap, chain: called.__setitem__("published", snap))

    ui._paper_forward_data_poll_once()

    assert called["chain"] == 0
    assert called["candles"] == 0
    assert called["published"]["option_chain_status"] == "not_fetched_auth_not_ok"
    assert called["published"]["chain_fetch_attempted"] is False
    assert "auth_status=SESSION_EXPIRED" in called["published"]["option_chain_error"]


def test_market_fetch_runs_after_auth_ok(monkeypatch):
    ui = object.__new__(ScalperUI)
    ui._client = object()
    monkeypatch.setattr(ui, "_pf_ensure_auth_terminal", lambda force=False: ("AUTH_OK", "", 23172.5))
    monkeypatch.setattr(ui, "_pf_auth_status_fields", lambda: {"auth_validation_attempted": True})
    monkeypatch.setattr(ui, "_pf_fetch_full_option_chain_readonly", lambda *a, **k: ([{"strike": 1}] * 24, "DATA_OK", "", ""))
    monkeypatch.setattr(ui, "_pf_fetch_candles_readonly", lambda *a, **k: ([{"close": 1}] * 100, "mstock_historical", ""))
    monkeypatch.setattr(ui, "_pf_apply_data_status_to_ui", lambda status: None)
    captured = {}
    monkeypatch.setattr(ui, "_publish_market_snapshot_to_paper_forward", lambda snap, chain: captured.update(snap=snap, chain=chain))

    ui._paper_forward_data_poll_once()

    assert captured["snap"]["chain_fetch_attempted"] is True
    assert captured["snap"]["candle_fetch_attempted"] is True
    assert captured["snap"]["option_chain_status"] == "DATA_OK"
