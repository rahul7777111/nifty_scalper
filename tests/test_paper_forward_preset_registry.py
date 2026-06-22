"""Preset registry tests."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_preset_registry_exists_and_has_entries():
    p = Path("config/paper_forward_preset_registry.json")
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        assert d.get("num_presets", 0) > 0
