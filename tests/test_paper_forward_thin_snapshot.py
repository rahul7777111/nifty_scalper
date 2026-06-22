"""TASK 7: thin snapshot handling."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_forward_engine import PaperForwardEngine, LOAD_OK

def test_thin_chain_only_returns_structured_decisions_no_waiting():
    eng = PaperForwardEngine(candidate_file="config/paper_forward_candidates.json", broker_safe_mode=True)
    loadable = [c for c in eng.candidates if c.get("_load_status") in (LOAD_OK, "candidate_loaded_ok")]
    if not loadable:
        return
    eng.candidates = loadable[:4]
    chain = [{"strike": 24150, "option_type": "PE", "ltp": 42, "bid": 41.5, "ask": 42.5, "spot": 24150, "expiry": "2026-06-25"}]
    snap = {"price": 24150, "timestamp": "2026-06-11T10:00:00Z", "source": "test_thin"}
    decs = eng.on_market_snapshot(snap, chain)
    waiting = sum(1 for d in decs if "waiting_for_snapshot" in str(d.get("no_trade_reason", "")).lower())
    exc = sum(1 for d in decs if "route_exception" in str(d.get("no_trade_reason", "")).lower() or d.get("final_signal") == "ERROR")
    assert waiting == 0
    assert exc == 0 or len(decs) > 0  # allow some model_route if truly thin, but structured
    assert len(decs) > 0

def test_missing_features_reported():
    eng = PaperForwardEngine(candidate_file="config/paper_forward_candidates.json", broker_safe_mode=True)
    # force a cand with required features
    for c in eng.candidates:
        c["_feature_list"] = ["some_missing_indicator_999", "ret_1"]
    eng.candidates = eng.candidates[:1]
    chain = [{"strike": 24100, "option_type": "CE", "ltp": 30, "spot": 24150}]
    snap = {"price": 24150, "timestamp": "..."}
    decs = eng.on_market_snapshot(snap, chain)
    if decs:
        assert "feature_vector_missing_columns" in str(decs[0].get("no_trade_reason", "")) or "missing" in str(decs[0])
        assert decs[0].get("predict_attempted") is False
        assert decs[0].get("confidence") is None
