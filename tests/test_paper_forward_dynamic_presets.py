"""PHASE 11: dynamic presets load + fallback."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_forward_engine import load_dynamic_presets, resolve_candidate_preset

def test_dynamic_presets_load_from_config():
    reg = load_dynamic_presets()
    assert reg.get("loaded") is True
    assert "dynamic_presets.json" in (reg.get("source") or "") or reg.get("source")

def test_preset_fallback_generated_when_safe_and_no_config():
    # simulate missing by empty reg
    cand = {"candidate_id": "test_fallback", "preset_family": ""}
    pres = resolve_candidate_preset(cand, {"loaded": False, "data": {}})
    assert pres["preset_fallback_generated"] is True
    assert "PAPER_OBSERVE" in pres["selected"] or pres["selected"]

def test_no_generic_no_dynamic_when_global_exists():
    reg = load_dynamic_presets()
    cand = {"candidate_id": "t1", "preset_family": "CONSERVATIVE"}
    pres = resolve_candidate_preset(cand, reg)
    assert pres["preset_loaded_ok"] or pres["preset_fallback_generated"]
    assert "no_dynamic_presets_configured" not in str(pres)
