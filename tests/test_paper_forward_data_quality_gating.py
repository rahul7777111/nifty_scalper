import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import paper_forward_engine as pfe
from paper_forward_engine import LOAD_OK, PaperForwardEngine


def test_thin_option_chain_blocks_prediction_and_entries(monkeypatch):
    art = Path("tests") / "_tmp_paper_forward_quality" / "c1"
    art.mkdir(parents=True, exist_ok=True)
    eng = PaperForwardEngine(broker_safe_mode=True)
    eng.candidates = [{
        "candidate_id": "c1",
        "enabled": True,
        "artifact_dir": str(art),
        "_load_status": LOAD_OK,
        "_feature_list": [],
        "model_name": "unit_model",
        "preset_family": "unit_preset",
    }]
    eng._state = {"c1": {"last_no_trade_reason": "", "confidence": None, "predict_attempted": False}}

    def route(**kwargs):
        raise AssertionError("thin option chain must not route/predict")

    monkeypatch.setattr(pfe, "route_candidate_decision", route)
    decisions = eng.on_market_snapshot(
        {"price": 24150, "timestamp": "t", "broker_auth": "AUTH_OK", "candles": [{"close": 1}] * 100},
        [{"strike": 24150, "option_type": "CE", "ltp": 10, "spot": 24150}],
    )

    assert decisions[0]["final_signal"] == "NO_TRADE"
    assert decisions[0]["no_trade_reason"] == "thin_option_chain"
    assert decisions[0]["predict_attempted"] is False
    assert decisions[0]["route_error"] is False
    assert eng.get_diagnostics()["route_errors"] == 0


def test_no_broker_place_order_call_from_paper_forward_source():
    source = Path("src/paper_forward_engine.py").read_text(encoding="utf-8")
    assert ".place_order(" not in source
