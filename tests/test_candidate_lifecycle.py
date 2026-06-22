from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import candidate_lifecycle as cl


def make_artifact(tmp_path: Path, cid: str, *, features: list[str] | None = None) -> Path:
    artifact = tmp_path / cid
    artifact.mkdir()
    (artifact / "model.pkl").write_bytes(b"dummy")
    (artifact / "candidate_profile.json").write_text(
        json.dumps(
            {
                "candidate_id": cid,
                "model_name": "logistic_regression",
                "preset_family": "BOTH_directional_auto",
                "feature_order": features if features is not None else ["spot", "iv"],
            }
        ),
        encoding="utf-8",
    )
    return artifact


@pytest.fixture(autouse=True)
def skip_real_model_load(monkeypatch):
    monkeypatch.setattr(cl, "_dummy_prediction_check", lambda model_path, feature_order: (True, "prediction smoke passed"))


def test_artifact_missing_candidate_becomes_artifact_missing(tmp_path: Path):
    row = {"candidate_id": "missing", "enabled": True, "artifact_dir": str(tmp_path / "missing"), "model_name": "x", "preset_family": "p"}
    status = cl.evaluate_candidate_lifecycle(row, {}, {}, cl.build_lifecycle_context([row]))
    assert status.current_stage == cl.STAGE_ARTIFACT_MISSING
    assert status.block_reason == "ARTIFACT_DIR_MISSING"


def test_required_features_zero_candidate_blocked(tmp_path: Path):
    cid = "zero_features"
    artifact = make_artifact(tmp_path, cid, features=[])
    row = {"candidate_id": cid, "enabled": True, "artifact_dir": str(artifact), "model_path": str(artifact / "model.pkl"), "model_name": "logistic_regression", "preset_family": "BOTH_directional_auto", "required_features": 0}
    status = cl.evaluate_candidate_lifecycle(row, {}, {}, cl.build_lifecycle_context([row]))
    assert status.current_stage == cl.STAGE_ARTIFACT_MISSING
    assert status.block_reason == "FEATURE_ORDER_EMPTY"


def test_valid_candidate_appears_in_qualified_view(tmp_path: Path):
    cid = "valid"
    artifact = make_artifact(tmp_path, cid)
    row = {"candidate_id": cid, "enabled": True, "artifact_dir": str(artifact), "model_path": str(artifact / "model.pkl"), "model_name": "logistic_regression", "preset_family": "BOTH_directional_auto"}
    status = cl.evaluate_candidate_lifecycle(row, {}, {}, cl.build_lifecycle_context([row]))
    assert status.current_stage == cl.STAGE_PAPER_FORWARD_ELIGIBLE
    assert cl.lifecycle_status_visible(status, "Show Qualified Only")


def test_feature_schema_file_counts_as_feature_order(tmp_path: Path):
    cid = "schema_features"
    artifact = tmp_path / cid
    artifact.mkdir()
    (artifact / "model.pkl").write_bytes(b"dummy")
    (artifact / "candidate_profile.json").write_text(
        json.dumps({"candidate_id": cid, "model_name": "logistic_regression", "preset_family": "BOTH_directional_auto"}),
        encoding="utf-8",
    )
    (artifact / "feature_schema.json").write_text(json.dumps({"features": ["spot", "ltp"]}), encoding="utf-8")
    row = {"candidate_id": cid, "enabled": True, "artifact_dir": str(artifact), "model_path": str(artifact / "model.pkl"), "model_name": "logistic_regression", "preset_family": "BOTH_directional_auto"}
    status = cl.evaluate_candidate_lifecycle(row, {}, {}, cl.build_lifecycle_context([row]))
    assert status.current_stage == cl.STAGE_PAPER_FORWARD_ELIGIBLE


def test_paper_forward_only_without_promotion_path_hidden_from_default(tmp_path: Path):
    cid = "paper_only"
    artifact = make_artifact(tmp_path, cid)
    row = {
        "candidate_id": cid,
        "enabled": True,
        "artifact_dir": str(artifact),
        "model_path": str(artifact / "model.pkl"),
        "model_name": "logistic_regression",
        "preset_family": "BOTH_directional_auto",
        "classification": "paper_forward_only",
        "paper_forward_only": True,
        "shadow_ready": False,
        "notes": "paper observation only",
    }
    status = cl.evaluate_candidate_lifecycle(row, {}, {}, cl.build_lifecycle_context([row]))
    assert status.current_stage == cl.STAGE_PAPER_FORWARD_ELIGIBLE
    assert status.block_reason == "PAPER_OBSERVATION_ONLY_NOT_LIVE_PATH"
    assert cl.lifecycle_status_visible(status, "Show Qualified Only") is False
    assert cl.lifecycle_status_visible(status, "Show Paper Forward Eligible") is True


def test_paper_forward_passed_candidate_becomes_shadow_eligible(tmp_path: Path):
    cid = "paper_pass"
    artifact = make_artifact(tmp_path, cid)
    row = {"candidate_id": cid, "enabled": True, "artifact_dir": str(artifact), "model_path": str(artifact / "model.pkl"), "model_name": "logistic_regression", "preset_family": "BOTH_directional_auto", "paper_forward_passed": True}
    status = cl.evaluate_candidate_lifecycle(row, {}, {}, cl.build_lifecycle_context([row]))
    assert status.current_stage == cl.STAGE_SHADOW_ELIGIBLE


def test_shadow_passed_candidate_becomes_live_dryrun_eligible(tmp_path: Path):
    cid = "shadow_pass"
    artifact = make_artifact(tmp_path, cid)
    row = {"candidate_id": cid, "enabled": True, "artifact_dir": str(artifact), "model_path": str(artifact / "model.pkl"), "model_name": "logistic_regression", "preset_family": "BOTH_directional_auto", "shadow_passed": True}
    status = cl.evaluate_candidate_lifecycle(row, {}, {}, cl.build_lifecycle_context([row]))
    assert status.current_stage == cl.STAGE_LIVE_DRY_RUN_ELIGIBLE


def test_live_dryrun_passed_candidate_becomes_live_1lot_eligible(tmp_path: Path, monkeypatch):
    cid = "dry_pass"
    artifact = make_artifact(tmp_path, cid)
    row = {"candidate_id": cid, "enabled": True, "artifact_dir": str(artifact), "model_path": str(artifact / "model.pkl"), "model_name": "logistic_regression", "preset_family": "BOTH_directional_auto", "dryrun_passed": True}
    monkeypatch.setenv("REQUIRE_MANUAL_CONFIRM", "true")
    monkeypatch.setenv("ORDER_PLACEMENT_ENABLED", "false")
    context = cl.build_lifecycle_context([row], live_whitelist=[cid])
    status = cl.evaluate_candidate_lifecycle(row, {}, {}, context)
    assert status.current_stage == cl.STAGE_LIVE_1_LOT_ELIGIBLE
    assert status.can_be_live_deployed is True


def test_live_1lot_requires_exactly_one_whitelist_candidate(tmp_path: Path):
    cid = "dry_pass"
    artifact = make_artifact(tmp_path, cid)
    row = {"candidate_id": cid, "enabled": True, "artifact_dir": str(artifact), "model_path": str(artifact / "model.pkl"), "model_name": "logistic_regression", "preset_family": "BOTH_directional_auto", "dryrun_passed": True}
    status = cl.evaluate_candidate_lifecycle(row, {}, {}, cl.build_lifecycle_context([row], live_whitelist=[]))
    assert status.current_stage == cl.STAGE_LIVE_1_LOT_ELIGIBLE
    assert status.can_be_live_deployed is False
    assert "EXACTLY_ONE" in status.block_reason


def test_nifty_one_lot_quantity_is_65():
    assert cl.LIVE_ONE_LOT_QTY == 65


def test_confidence_none_is_not_converted_to_zero(tmp_path: Path):
    cid = "none_conf"
    artifact = make_artifact(tmp_path, cid)
    row = {"candidate_id": cid, "enabled": True, "artifact_dir": str(artifact), "model_path": str(artifact / "model.pkl"), "model_name": "logistic_regression", "preset_family": "BOTH_directional_auto"}
    status = cl.evaluate_candidate_lifecycle(row, {}, {cid: {"predict_attempted": True, "confidence": None}}, cl.build_lifecycle_context([row]))
    assert status.block_reason == "PREDICT_CONFIDENCE_NONE"


def test_load_paper_forward_candidate_rows_coerces_enabled_n_to_false(tmp_path: Path):
    config_path = tmp_path / "paper_forward_candidates.json"
    config_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {"candidate_id": "disabled_str", "enabled": "N"},
                    {"candidate_id": "enabled_str", "enabled": "Y"},
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = cl.load_paper_forward_candidate_rows(config_path)

    assert rows[0]["enabled"] is False
    assert rows[1]["enabled"] is True
