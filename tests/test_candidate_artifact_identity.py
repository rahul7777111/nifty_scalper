"""Strict candidate artifact identity resolver tests."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from candidate_artifact_resolver import resolve_candidate_artifact  # noqa: E402
from paper_forward_engine import resolve_candidate_artifacts  # noqa: E402


def _write_candidate_artifact(
    root: Path,
    candidate_id: str,
    *,
    model_name: str,
    preset_family: str,
    side_policy: str,
) -> Path:
    folder = root / "artifacts" / "candidates" / candidate_id
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / "model.pkl"
    with model_path.open("wb") as fh:
        pickle.dump({"model": object(), "features": ["f1"], "selected_threshold": 0.3}, fh)
    (folder / "feature_schema.json").write_text(json.dumps({"features": ["f1"]}), encoding="utf-8")
    manifest = {
        "candidate_id": candidate_id,
        "model_name": model_name,
        "preset_family": preset_family,
        "side_policy": side_policy,
        "selected_threshold": 0.3,
    }
    (folder / "candidate_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return model_path


def test_exact_model_pkl_loads_successfully(tmp_path: Path):
    cid = "elasticnet_BOTH_directional_auto_BOTH_dir_auto_t30_20260610_130953"
    _write_candidate_artifact(
        tmp_path,
        cid,
        model_name="elasticnet",
        preset_family="BOTH_directional_auto",
        side_policy="AUTO_DIRECTIONAL",
    )
    candidate = {
        "candidate_id": cid,
        "model_name": "elasticnet",
        "preset_family": "BOTH_directional_auto",
        "side_policy": "AUTO_DIRECTIONAL",
    }
    resolution = resolve_candidate_artifact(candidate, tmp_path)
    assert resolution.artifact_identity_status == "EXACT_MATCH"
    assert resolution.loaded is True
    assert resolution.selected_artifact_path.endswith("model.pkl")


def test_t30_must_not_load_t35(tmp_path: Path):
    t30 = "elasticnet_BOTH_directional_auto_BOTH_dir_auto_t30_20260610_130953"
    t35 = "elasticnet_BOTH_directional_auto_BOTH_dir_auto_t35_20260610_130953"
    _write_candidate_artifact(tmp_path, t35, model_name="elasticnet", preset_family="BOTH_directional_auto", side_policy="AUTO_DIRECTIONAL")
    candidate = {
        "candidate_id": t30,
        "model_name": "elasticnet",
        "preset_family": "BOTH_directional_auto",
        "side_policy": "AUTO_DIRECTIONAL",
    }
    resolution = resolve_candidate_artifact(candidate, tmp_path)
    assert resolution.loaded is False
    assert resolution.artifact_identity_status == "MISSING"
    assert resolution.disable_reason == "ARTIFACT_NOT_FOUND"


def test_directional_auto_must_not_load_auto_balanced(tmp_path: Path):
    directional = "elasticnet_BOTH_directional_auto_BOTH_dir_auto_t30_20260610_130953"
    balanced = "elasticnet_BOTH_auto_balanced_BOTH_auto_bal_t35_20260610_130953"
    _write_candidate_artifact(tmp_path, balanced, model_name="elasticnet", preset_family="BOTH_auto_balanced", side_policy="BOTH")
    candidate = {
        "candidate_id": directional,
        "model_name": "elasticnet",
        "preset_family": "BOTH_directional_auto",
        "side_policy": "AUTO_DIRECTIONAL",
        "artifact_dir": f"artifacts/candidates/{balanced}",
        "model_path": f"artifacts/candidates/{balanced}/model.pkl",
    }
    resolution = resolve_candidate_artifact(candidate, tmp_path)
    assert resolution.loaded is False
    assert resolution.artifact_identity_status in {"MISSING", "MISMATCH"}


def test_stale_configured_path_rejected_even_when_exact_exists(tmp_path: Path):
    directional = "elasticnet_BOTH_directional_auto_BOTH_dir_auto_t30_20260610_130953"
    balanced = "elasticnet_BOTH_auto_balanced_BOTH_auto_bal_t30_20260610_130953"
    _write_candidate_artifact(tmp_path, directional, model_name="elasticnet", preset_family="BOTH_directional_auto", side_policy="AUTO_DIRECTIONAL")
    _write_candidate_artifact(tmp_path, balanced, model_name="elasticnet", preset_family="BOTH_auto_balanced", side_policy="BOTH")
    candidate = {
        "candidate_id": directional,
        "model_name": "elasticnet",
        "preset_family": "BOTH_directional_auto",
        "side_policy": "AUTO_DIRECTIONAL",
        "artifact_dir": f"artifacts/candidates/{balanced}",
        "model_path": f"artifacts/candidates/{balanced}/model.pkl",
    }
    resolution = resolve_candidate_artifact(candidate, tmp_path)
    assert resolution.loaded is False
    assert resolution.artifact_identity_status == "MISMATCH"
    assert resolution.disable_reason == "ARTIFACT_IDENTITY_MISMATCH"
    assert "candidate_id" in resolution.mismatch_fields


def test_volatility_breakout_must_not_load_cost_survivor_balanced(tmp_path: Path):
    breakout = "xgboost_volatility_breakout_vol_breakout_t30_20260610_130953"
    survivor = "xgboost_cost_survivor_balanced_cost_surv_bal_t30_20260610_130953"
    _write_candidate_artifact(tmp_path, survivor, model_name="xgboost", preset_family="cost_survivor_balanced", side_policy="COST_SURVIVOR")
    candidate = {
        "candidate_id": breakout,
        "model_name": "xgboost",
        "preset_family": "volatility_breakout",
        "side_policy": "VOLATILITY_BREAKOUT",
        "artifact_dir": f"artifacts/candidates/{survivor}",
        "model_path": f"artifacts/candidates/{survivor}/model.pkl",
    }
    resolution = resolve_candidate_artifact(candidate, tmp_path)
    assert resolution.loaded is False
    assert resolution.artifact_identity_status in {"MISSING", "MISMATCH"}


def test_pe_only_must_not_load_both(tmp_path: Path):
    pe = "elasticnet_PE_only_balanced_PE_only_balanced_t30_20260610_130953"
    both = "elasticnet_BOTH_auto_balanced_BOTH_auto_bal_t30_20260610_130953"
    _write_candidate_artifact(tmp_path, both, model_name="elasticnet", preset_family="BOTH_auto_balanced", side_policy="BOTH")
    candidate = {
        "candidate_id": pe,
        "model_name": "elasticnet",
        "preset_family": "PE_only_balanced",
        "side_policy": "PE_ONLY",
        "artifact_dir": f"artifacts/candidates/{both}",
        "model_path": f"artifacts/candidates/{both}/model.pkl",
    }
    resolution = resolve_candidate_artifact(candidate, tmp_path)
    assert resolution.loaded is False
    assert resolution.artifact_identity_status in {"MISSING", "MISMATCH"}


def test_missing_artifact_disables_candidate(tmp_path: Path):
    candidate = {
        "candidate_id": "missing_candidate_t30_20260610_130953",
        "model_name": "elasticnet",
        "preset_family": "BOTH_directional_auto",
        "side_policy": "AUTO_DIRECTIONAL",
        "enabled": True,
    }
    resolution = resolve_candidate_artifact(candidate, tmp_path)
    assert resolution.loaded is False
    assert resolution.disable_reason == "ARTIFACT_NOT_FOUND"


def test_strict_resolver_used_by_paper_forward_engine(tmp_path: Path):
    cid = "elasticnet_BOTH_directional_auto_BOTH_dir_auto_t30_20260610_130953"
    _write_candidate_artifact(tmp_path, cid, model_name="elasticnet", preset_family="BOTH_directional_auto", side_policy="AUTO_DIRECTIONAL")
    candidate = {
        "candidate_id": cid,
        "model_name": "elasticnet",
        "preset_family": "BOTH_directional_auto",
        "side_policy": "AUTO_DIRECTIONAL",
    }
    result = resolve_candidate_artifacts(candidate, project_root=tmp_path)
    assert result["artifact_identity_status"] == "EXACT_MATCH"
    assert result["loaded"] is True
    assert result["resolved_dir"].endswith(cid)


def test_strict_resolver_used_by_backtest_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from scripts import backtest_ml_models_from_csv as bt

    cid = "elasticnet_BOTH_directional_auto_BOTH_dir_auto_t30_20260610_130953"
    _write_candidate_artifact(tmp_path, cid, model_name="elasticnet", preset_family="BOTH_directional_auto", side_policy="AUTO_DIRECTIONAL")
    candidate = {
        "candidate_id": cid,
        "model_name": "elasticnet",
        "preset_family": "BOTH_directional_auto",
        "side_policy": "AUTO_DIRECTIONAL",
        "enabled": True,
    }
    model, failures = bt._load_candidate_model(
        candidate,
        root=tmp_path,
        config_dir=None,
        fallback_threshold=0.3,
        use_candidate_thresholds=True,
        log_fn=None,
        strict_artifacts=True,
        allow_artifact_fallback=False,
    )
    assert model is not None
    assert model.candidate_id == cid
    assert str(model.artifact_path).endswith("model.pkl")
    assert failures == []
