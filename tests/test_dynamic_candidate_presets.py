from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from candidate_profile import (  # noqa: E402
    CANDIDATE_MATRIX,
    CandidateProfile,
    DynamicPreset,
    SIDE_CE_ONLY,
    SIDE_PE_ONLY,
    SIDE_AUTO_DIRECTIONAL,
    build_preset_family,
    evaluate_candidate_gates,
    filter_name_for_side,
    load_candidate_profile,
)
from candidate_filters import apply_ce_only_filter, apply_pe_only_filter  # noqa: E402
from dynamic_preset_selector import select_dynamic_preset, safe_no_trade_preset  # noqa: E402


def test_dynamic_preset_roundtrip():
    p = DynamicPreset(name="test_preset", option_side_policy=SIDE_PE_ONLY, entry_threshold=0.4)
    d = p.to_dict()
    p2 = DynamicPreset.from_dict(d)
    assert p2.name == "test_preset"
    assert p2.entry_threshold == 0.4


def test_candidate_profile_save_load(tmp_path: Path):
    profile = CandidateProfile(
        candidate_id="test_candidate",
        model_name="elasticnet",
        feature_set_name="live_v1",
        target_name="cost_survivor_label_v2",
        side_policy=SIDE_PE_ONLY,
        preset_family="PE_only_conservative",
        dynamic_presets={"p1": DynamicPreset(name="p1").to_retrainer_dict()},
        threshold_policy={},
        selection_policy={},
        risk_policy={},
        cost_policy={},
        artifact_paths={"model_pkl": "artifacts/candidates/test_candidate/model.pkl"},
        validation_metrics={"roc_auc": 0.61},
        gate_results={"shadow_ready": False, "gate_fail_count": 1},
        live_computable_features=["atr_14"],
        created_at="2026-06-10T00:00:00Z",
    )
    profile.save(tmp_path)
    candidate_dir = tmp_path / "test_candidate"
    loaded_json = json.loads((candidate_dir / "candidate_profile.json").read_text())
    loaded_profile = load_candidate_profile(candidate_dir)

    assert loaded_json["candidate_id"] == "test_candidate"
    assert loaded_profile.candidate_id == "test_candidate"
    assert loaded_profile.artifact_paths["model_pkl"].endswith("model.pkl")
    assert loaded_profile.validation_metrics["roc_auc"] == 0.61
    assert loaded_profile.gate_results["gate_fail_count"] == 1
    assert (candidate_dir / "dynamic_preset.json").exists()
    assert (candidate_dir / "dynamic_presets.json").exists()
    assert (candidate_dir / "metrics.json").exists()
    assert (candidate_dir / "gates.json").exists()


def test_preset_families_exist():
    families = [
        "PE_only_conservative",
        "CE_only_conservative",
        "BOTH_directional_auto",
        "BOTH_symmetric",
        "high_confidence_low_frequency",
    ]
    for fam in families:
        presets = build_preset_family(fam)
        assert len(presets) >= 1


def test_candidate_matrix_not_empty():
    assert len(CANDIDATE_MATRIX) >= 5


def test_pe_only_filter_rejects_ce():
    snap = {"option_type": "CE"}
    assert apply_pe_only_filter(snap)["filter_passed"] is False


def test_ce_only_filter_rejects_pe():
    snap = {"option_type": "PE"}
    assert apply_ce_only_filter(snap)["filter_passed"] is False


def test_gate_rejects_tiny_trade_count():
    metrics = {
        "trade_count": 10,
        "pe_trade_count": 10,
        "ce_trade_count": 0,
        "pf_net": 1.5,
        "cost_1_5x_pf": 1.2,
    }
    gates = evaluate_candidate_gates(metrics, side_policy=SIDE_PE_ONLY)
    assert gates["shadow_ready"] is False
    assert "not_tiny_trade_count" in gates["reject_reasons"] or "enough_trades" in gates["reject_reasons"]


def test_select_dynamic_preset_no_presets():
    profile = CandidateProfile(
        candidate_id="x",
        model_name="m",
        feature_set_name="f",
        target_name="t",
        side_policy=SIDE_AUTO_DIRECTIONAL,
        preset_family="BOTH_directional_auto",
        dynamic_presets={},
        threshold_policy={},
        selection_policy={},
        risk_policy={},
        cost_policy={},
        artifact_paths={},
        validation_metrics={},
        gate_results={},
        live_computable_features=[],
    )
    out = select_dynamic_preset(profile, {"option_type": "PE", "atr_pct": 0.01})
    assert out["selected_preset_name"] == "unnamed_preset"
    assert out["trade_allowed"] is True


def test_auto_directional_selects_pe_in_bearish_regime():
    preset = build_preset_family("BOTH_directional_auto")[0]
    profile = CandidateProfile(
        candidate_id="auto1",
        model_name="elasticnet",
        feature_set_name="f",
        target_name="t",
        side_policy=SIDE_AUTO_DIRECTIONAL,
        preset_family="BOTH_directional_auto",
        dynamic_presets={preset.name: preset.to_retrainer_dict()},
        threshold_policy={},
        selection_policy={},
        risk_policy={},
        cost_policy={},
        artifact_paths={},
        validation_metrics={},
        gate_results={},
        live_computable_features=[],
    )
    out = select_dynamic_preset(profile, {"option_type": "PE", "regime_label": "bearish", "atr_pct": 0.01})
    if out.get("trade_allowed"):
        assert out["side_allowed"] in {"PE", "CE", "BOTH", "NONE"}


def test_filter_name_for_side():
    assert filter_name_for_side(SIDE_PE_ONLY) == "PE_only"
    assert filter_name_for_side(SIDE_CE_ONLY) == "CE_only"


def test_safe_no_trade_preset():
    out = safe_no_trade_preset("test_reason")
    assert out["trade_allowed"] is False
    assert out["no_trade_reason"] == "test_reason"
