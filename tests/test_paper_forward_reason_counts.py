import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_forward_engine import PaperForwardEngine, PaperForwardCandidateRuntimeState

def test_reason_counts_non_empty():
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
    eng.candidates = [{"candidate_id": "c1"}]
    cs = PaperForwardCandidateRuntimeState("c1")
    eng._candidate_states["c1"] = cs
    eng._state["c1"] = {}
    dec = {"candidate_id": "c1", "reason_code": "feature_vector_missing_columns", "predict_attempted": False}
    eng._record_decision("c1", dec, "t", {})
    assert eng._reason_counts.get("feature_vector_missing_columns", 0) > 0
    print("[TEST] reason_counts populated")