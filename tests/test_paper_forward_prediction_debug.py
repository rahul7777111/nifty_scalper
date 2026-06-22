import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import paper_forward_engine as pfe
from paper_forward_engine import LOAD_OK, PaperForwardEngine


def test_low_confidence_logs_raw_prediction_debug(monkeypatch, capsys):
    art = Path("tests") / "_tmp_paper_forward_predict" / "c1"
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
        return {"final_signal": "NO_TRADE", "confidence": 0.0, "threshold": 0.35, "no_trade_reason": "low_confidence_0.0000_lt_0.3500"}

    monkeypatch.setattr(pfe, "route_candidate_decision", route)
    chain = [{"strike": 23000 + i * 50, "option_type": "CE" if i % 2 == 0 else "PE", "ltp": 10, "spot": 24150} for i in range(24)]
    eng.on_market_snapshot({"price": 24150, "timestamp": "t", "broker_auth": "AUTH_OK", "candles": [{"open": 1, "high": 2, "low": 1, "close": 2}] * 20}, chain)

    out = capsys.readouterr().out
    assert "[PAPER-FWD-PREDICT]" in out
    assert "raw_output=0.0" in out
    assert "confidence=0.0" in out


def test_prediction_debug_contract_mentions_scaler_and_calibrator():
    source = Path("src/paper_forward_engine.py").read_text(encoding="utf-8")
    assert "scaler_applied=" in source
    assert "calibrator_applied=" in source
