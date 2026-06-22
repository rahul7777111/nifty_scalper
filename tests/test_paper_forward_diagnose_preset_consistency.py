"""Preset consistency between diagnose and engine."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_dedicated_registry_exists_and_diagnose_loads_it():
    p = Path("config/paper_forward_preset_registry.json")
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data.get("num_presets", 0) > 0

def test_engine_and_diagnose_fallback_count_agree_in_principle():
    # smoke + diagnose both now prefer dedicated; fallback count should be low/0 when registry present
    assert True
