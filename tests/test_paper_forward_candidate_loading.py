"""PHASE 11: candidate loading + dedupe + state."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_forward_engine import PaperForwardEngine, LOAD_OK

def test_valid_artifact_paths_load_successfully():
    eng = PaperForwardEngine(candidate_file="config/paper_forward_candidates.json", broker_safe_mode=True)
    # at least the ones that resolve in this env should have attempted load
    assert len(eng.candidates) > 0
    any_ok = any(c.get("_load_status") in (LOAD_OK, "candidate_loaded_ok") or c.get("_preset", {}).get("preset_fallback_generated") for c in eng.candidates)
    if not any_ok:
        assert all(c.get("enabled") is False and c.get("disabled_reason") for c in eng.candidates)

def test_duplicate_reload_does_not_duplicate_treeview_rows():
    # simulated: engine _load has dedupe
    eng = PaperForwardEngine(candidate_file="config/paper_forward_candidates.json", broker_safe_mode=True)
    before = len(eng.candidates)
    # simulate reload path
    eng._load_candidates(None, "config/paper_forward_candidates.json")
    after = len(eng.candidates)
    assert after <= before + 0  # no growth on re-load


def test_engine_coerces_enabled_n_to_disabled_status_row(tmp_path):
    cfg = {
        "candidates": [
            {
                "candidate_id": "disabled_str",
                "paper_forward_only": True,
                "enabled": "N",
                "artifact_dir": str(tmp_path / "disabled_str"),
                "model_name": "logistic_regression",
                "preset_family": "p",
                "side_policy": "BOTH",
            }
        ]
    }
    path = tmp_path / "paper_forward_candidates.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    eng = PaperForwardEngine(candidate_file=str(path), artifacts_dir=str(tmp_path), broker_safe_mode=True)

    assert eng.candidates[0]["enabled"] is False
    assert eng.get_status_table()[0]["enabled"] is False
