from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gpt_advisor
import strategy


class _Cfg:
    gpt_enabled = True
    gpt_enable = True
    gpt_apply_paper = True
    gpt_mode = "gate"


def _reset_gpt_session() -> None:
    gpt_advisor._gpt_session_disabled = False
    gpt_advisor._gpt_session_disable_reason = ""
    gpt_advisor.circuit_breaker.failure_count = 0
    gpt_advisor.circuit_breaker.disabled_until = 0.0


def test_http_402_disables_gpt_without_retry():
    _reset_gpt_session()
    calls = {"n": 0}

    def http_post(url, headers, payload, timeout):
        calls["n"] += 1
        return 402, "payment required"

    res = gpt_advisor.advise_trade(
        proposal={"symbol": "NIFTY"},
        model="gpt-4o-mini",
        api_key="x",
        timeout_sec=0.5,
        http_post=http_post,
    )
    assert res.decision == "SKIP"
    assert "402" in res.reason.lower()
    assert calls["n"] == 1
    assert gpt_advisor.is_gpt_disabled_for_session() is True

    res2 = gpt_advisor.advise_trade(
        proposal={"symbol": "NIFTY"},
        model="gpt-4o-mini",
        api_key="x",
        timeout_sec=0.5,
        http_post=http_post,
    )
    assert res2.decision == "SKIP"
    assert calls["n"] == 1


def test_gpt_fallback_does_not_block_strategy():
    app = object.__new__(strategy.NiftyScalper)
    app.cfg = _Cfg()
    app._gpt_should_apply_now = lambda is_paper: True
    app._gpt_control_advise = lambda **kwargs: None
    assert strategy.NiftyScalper._gpt_entry_gate(app, proposal={"trade_name": "x"}, is_paper=True) is True
