import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import paper_forward_engine as pfe
from paper_forward_engine import LOAD_OK, PaperForwardEngine


def _chain(rows=20):
    return [{"strike": 24000 + i * 50, "option_type": "CE" if i % 2 == 0 else "PE", "ltp": 10, "spot": 24150} for i in range(rows)]


def _engine_with_candidate(feature_list=None):
    art = Path("tests") / "_tmp_paper_forward_route" / "c1"
    art.mkdir(parents=True, exist_ok=True)
    eng = PaperForwardEngine(broker_safe_mode=True)
    cand = {
        "candidate_id": "c1",
        "enabled": True,
        "artifact_dir": str(art),
        "_load_status": LOAD_OK,
        "_feature_list": feature_list or [],
        "model_name": "unit_model",
        "preset_family": "unit_preset",
        "side_policy": "BOTH",
    }
    eng.candidates = [cand]
    eng._state = {"c1": {"last_no_trade_reason": "", "confidence": None, "predict_attempted": False}}
    return eng


def test_missing_features_do_not_call_router_or_count_route_error(monkeypatch):
    eng = _engine_with_candidate(["feature_that_is_not_live"])
    called = {"router": 0}

    def bad_router(**kwargs):
        called["router"] += 1
        raise AssertionError("router should not be called")

    monkeypatch.setattr(pfe, "route_candidate_decision", bad_router)
    decisions = eng.on_market_snapshot({"price": 24150, "timestamp": "t"}, _chain())

    assert called["router"] == 0
    assert decisions[0]["no_trade_reason"] == "feature_vector_missing_columns"
    assert decisions[0]["predict_attempted"] is False
    assert decisions[0]["route_error"] is False
    diag = eng.get_diagnostics()
    assert diag["evaluations_count"] == 1
    assert diag["route_errors"] == 0


def test_route_exception_is_logged_structured_and_exposed(monkeypatch):
    eng = _engine_with_candidate(["price"])

    def bad_router(**kwargs):
        raise NameError("volatility_state")

    monkeypatch.setattr(pfe, "route_candidate_decision", bad_router)
    decisions = eng.on_market_snapshot({"price": 24150, "timestamp": "t"}, _chain())

    assert decisions[0]["no_trade_reason"] == "model_route_exception"
    assert decisions[0]["route_error"] is True
    diag = eng.get_diagnostics()
    assert diag["route_errors"] == 1
    assert diag["route_exception_type_counts"]["NameError"] == 1
    assert "volatility_state" in diag["route_exception_tracebacks"][0]["exception_message"]


def test_router_receives_enriched_feature_snapshot(monkeypatch):
    eng = _engine_with_candidate(["volume", "oi", "ctx_time_sin", "dte_days"])
    captured = {}

    def router(**kwargs):
        snap = kwargs["market_snapshot"]
        captured.update({k: snap.get(k) for k in ("volume", "oi", "ctx_time_sin", "dte_days", "feature_coverage_pct")})
        return {
            "candidate_id": "c1",
            "final_signal": "NO_TRADE",
            "confidence": 0.0,
            "threshold": 0.35,
            "no_trade_reason": "low_confidence_0.0000_lt_0.3500",
        }

    monkeypatch.setattr(pfe, "route_candidate_decision", router)
    snapshot = {
        "price": 24150,
        "broker_auth": "AUTH_OK",
        "timestamp": "2026-06-11T09:30:00+00:00",
        "candles": [
            {"open": 24100, "high": 24150, "low": 24090, "close": 24120, "volume": 1000},
            {"open": 24120, "high": 24190, "low": 24110, "close": 24150, "volume": 2000},
        ],
    }
    chain = [
        {
            "strike": 24150 + i * 50,
            "option_type": "CE" if i % 2 == 0 else "PE",
            "ltp": 10 + i,
            "traded_volume": 123,
            "openInterest": 456,
            "expiry": "2026-06-25",
            "spot": 24150,
        }
        for i in range(20)
    ]

    decisions = eng.on_market_snapshot(snapshot, chain)

    assert decisions[0]["predict_attempted"] is True
    assert captured["volume"] == 123
    assert captured["oi"] == 456
    assert captured["ctx_time_sin"] is not None
    assert captured["dte_days"] > 0
    assert captured["feature_coverage_pct"] == 100.0


def test_paper_forward_engine_never_calls_broker_place_order():
    source = Path("src/paper_forward_engine.py").read_text(encoding="utf-8")
    assert ".place_order(" not in source
