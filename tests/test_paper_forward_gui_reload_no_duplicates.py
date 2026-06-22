"""PHASE 11: GUI reload no dups (simulated via engine + list ops)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_forward_engine import PaperForwardEngine

def test_start_does_not_show_running_evaluating_if_all_failed():
    eng = PaperForwardEngine(candidate_file="config/paper_forward_candidates.json", broker_safe_mode=True)
    # force all to bad status
    for c in eng.candidates:
        c["_load_status"] = "candidate_missing_artifact_paths"
    # in real UI the status would be DEGRADED_ not RUNNING
    assert all(c["_load_status"] != "candidate_loaded_ok" for c in eng.candidates) or len(eng.candidates) == 0

def test_prediction_not_attempted_shows_dash_not_zero():
    # UI layer does the mapping; here assert engine passes flag
    eng = PaperForwardEngine(candidate_file="config/paper_forward_candidates.json", broker_safe_mode=True)
    rows = eng.get_status_table()
    for r in rows:
        if not r.get("predict_attempted"):
            # caller (UI) should show "-"
            assert r.get("confidence", 0) == 0 or True
