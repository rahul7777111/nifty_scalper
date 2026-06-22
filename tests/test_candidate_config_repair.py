from __future__ import annotations

import json
from pathlib import Path

from scripts import repair_candidate_lifecycle_config as repair
from src import candidate_lifecycle as cl


def make_artifact(tmp_path: Path, cid: str) -> Path:
    artifact = tmp_path / cid
    artifact.mkdir()
    (artifact / "model.pkl").write_bytes(b"dummy")
    (artifact / "candidate_profile.json").write_text(
        json.dumps(
            {
                "candidate_id": cid,
                "model_name": "logistic_regression",
                "preset_family": "BOTH_directional_auto",
                "feature_order": ["spot"],
            }
        ),
        encoding="utf-8",
    )
    return artifact


def test_duplicate_candidate_archived(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cl, "_dummy_prediction_check", lambda model_path, feature_order: (True, "prediction smoke passed"))
    monkeypatch.setattr(repair.cl if hasattr(repair, "cl") else cl, "_dummy_prediction_check", lambda model_path, feature_order: (True, "prediction smoke passed"), raising=False)
    artifact = make_artifact(tmp_path, "dup")
    config = tmp_path / "paper_forward_candidates.json"
    archive = tmp_path / "archived_paper_forward_candidates.json"
    backups = tmp_path / "backups"
    rows = [
        {"candidate_id": "dup", "enabled": True, "artifact_dir": str(artifact), "model_path": str(artifact / "model.pkl"), "model_name": "logistic_regression", "preset_family": "BOTH_directional_auto"},
        {"candidate_id": "dup", "enabled": True, "artifact_dir": str(artifact), "model_path": str(artifact / "model.pkl"), "model_name": "logistic_regression", "preset_family": "BOTH_directional_auto"},
    ]
    config.write_text(json.dumps({"candidates": rows}), encoding="utf-8")
    monkeypatch.setattr(repair, "CONFIG_PATH", config)
    monkeypatch.setattr(repair, "ARCHIVE_PATH", archive)
    monkeypatch.setattr(repair, "BACKUP_DIR", backups)

    summary = repair.repair_config(apply=True)

    assert summary["archived"] == 2
    active = json.loads(config.read_text(encoding="utf-8"))["candidates"]
    archived = json.loads(archive.read_text(encoding="utf-8"))["candidates"]
    assert active == []
    assert len(archived) == 2
    assert list(backups.glob("paper_forward_candidates_*.json"))


def test_dry_run_does_not_modify_config(tmp_path: Path, monkeypatch):
    artifact = make_artifact(tmp_path, "valid")
    config = tmp_path / "paper_forward_candidates.json"
    original = {
        "candidates": [
            {"candidate_id": "valid", "enabled": True, "artifact_dir": str(artifact), "model_path": str(artifact / "model.pkl"), "model_name": "logistic_regression", "preset_family": "BOTH_directional_auto"}
        ]
    }
    config.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(repair, "CONFIG_PATH", config)
    monkeypatch.setattr(repair, "ARCHIVE_PATH", tmp_path / "archive.json")
    monkeypatch.setattr(repair, "BACKUP_DIR", tmp_path / "backups")
    monkeypatch.setattr(cl, "_dummy_prediction_check", lambda model_path, feature_order: (True, "prediction smoke passed"))

    repair.repair_config(apply=False)

    assert json.loads(config.read_text(encoding="utf-8")) == original
