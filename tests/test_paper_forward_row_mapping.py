import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_forward_engine import PaperForwardEngine

def test_row_mapping_uses_full_cid():
    eng = PaperForwardEngine.__new__(PaperForwardEngine)
    eng.candidates = [{"candidate_id": "full_candidate_id_123"}]
    eng._state = {"full_candidate_id_123": {}}
    eng._candidate_states = {}
    eng._decisions = []
    eng._evaluations_count = 0
    eng._reason_counts = {}
    rows = eng.get_status_table()
    assert rows[0]["candidate_id"] == "full_candidate_id_123"
    # simulate decision with full id
    dec = {"candidate_id": "full_candidate_id_123", "final_signal": "NO_TRADE", "reason_code": "x"}
    eng._record_decision("full_candidate_id_123", dec, "t", {})
    # no miss log in test, but mapping worked
    print("[TEST] row mapping full cid ok")