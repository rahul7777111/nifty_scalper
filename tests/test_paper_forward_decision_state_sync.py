import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_forward_engine import PaperForwardEngine, PaperForwardCandidateRuntimeState

def test_apply_decision_updates_state():
    eng = PaperForwardEngine.__new__(PaperForwardEngine)
    eng._state = {}
    eng._candidate_states = {}
    eng._reason_counts = {}
    eng._decisions = []
    eng._evaluations_count = 0
    eng._predictions_attempted = 0
    eng._predictions_success = 0
    eng._skipped_missing_features = 0
    eng._skipped_market_data = 0
    eng.candidates = [{"candidate_id": "c1", "enabled": True, "model_name": "rf", "preset_family": "p1", "side_policy": "CE"}]
    # init like in code
    for c in eng.candidates:
        cid = c["candidate_id"]
        cs = PaperForwardCandidateRuntimeState(cid)
        cs.enabled = c.get("enabled", True)
        cs.model = c.get("model_name", "")
        cs.preset = c.get("preset_family", "")
        cs.side = c.get("side_policy", "")
        eng._candidate_states[cid] = cs
        eng._state[cid] = {"open_position": False, "last_no_trade_reason": "", "predict_attempted": False}

    dec = {"candidate_id": "c1", "final_signal": "NO_TRADE", "confidence": 0.1, "predict_attempted": True, "reason_code": "low_confidence_0.0000_lt_0.3500", "snapshot_id": "s1", "timestamp": "t1", "data_quality_status": "DATA_OK"}
    eng._record_decision("c1", dec, "t1", {"price": 23172.5})  # will call apply

    cs = eng._candidate_states["c1"]
    assert cs.final_signal == "NO_TRADE"
    assert cs.predict_attempted is True
    assert cs.reason_code == "low_confidence_0.0000_lt_0.3500"
    assert cs.last_snapshot_id == "s1"
    assert any("low_confidence" in k for k in eng._reason_counts)
    print("[TEST] decision state sync ok")