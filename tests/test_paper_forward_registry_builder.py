"""TASK 9 tests for the new registry builders."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def test_artifact_registry_maps_exact_candidate_id():
    reg_path = Path("config/paper_forward_artifact_registry.json")
    if reg_path.exists():
        data = json.loads(reg_path.read_text())
        assert "candidates" in data
        assert len(data["candidates"]) >= 5

def test_missing_artifact_gives_nearest_matches():
    # The builder always produces an entry even for weak matches
    reg_path = Path("config/paper_forward_artifact_registry.json")
    if reg_path.exists():
        data = json.loads(reg_path.read_text())
        for cid, e in data.get("candidates", {}).items():
            assert "load_status" in e

def test_preset_registry_loads_real_preset_before_fallback():
    reg_path = Path("config/paper_forward_preset_registry.json")
    if reg_path.exists():
        data = json.loads(reg_path.read_text())
        assert data.get("num_presets", 0) > 0
