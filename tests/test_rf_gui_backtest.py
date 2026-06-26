from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rf_gui_backtest import (  # noqa: E402
    discover_ensemble_test_target,
    find_latest_ensemble_artifact,
    load_ensemble_feature_names,
    load_ensemble_selected_threshold,
    materialize_ensemble_dynamic_candidate,
    resolve_dataset_path,
)


def test_load_ensemble_feature_names_from_feature_order_json(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "research_retrain_20260619_151158"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "retrain_all_models_xgb_rf_ensemble_profitable_trade_label_feature_order.json").write_text(
        json.dumps({"feature_order": ["open", "ltp", "volume"]}),
        encoding="utf-8",
    )

    assert load_ensemble_feature_names(artifact_dir) == ["open", "ltp", "volume"]


def test_find_latest_ensemble_artifact_under_gui_retrain_root() -> None:
    root = REPO_ROOT / "models" / "ensemble_gui_retrain_fixed"
    if not root.exists():
        pytest.skip("ensemble fixture not present")
    artifact_path, artifact_dir = find_latest_ensemble_artifact(root)
    assert artifact_path is not None
    assert artifact_dir is not None
    assert "xgb_rf_ensemble" in artifact_path.name


def test_materialize_ensemble_dynamic_candidate_without_feature_manifest() -> None:
    root = REPO_ROOT / "models" / "ensemble_gui_retrain_fixed"
    if not root.exists():
        pytest.skip("ensemble fixture not present")
    artifact_path, artifact_dir = find_latest_ensemble_artifact(root)
    assert artifact_path is not None and artifact_dir is not None

    wrapper_dir, config_path = materialize_ensemble_dynamic_candidate(REPO_ROOT, artifact_path, artifact_dir)
    assert wrapper_dir is not None
    assert config_path is not None
    assert config_path.exists()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["candidates"][0]["model_name"] == "xgb_rf_ensemble"
    assert payload["candidates"][0]["model_path"]


def test_discover_ensemble_test_target_prefers_static_config() -> None:
    static_config = REPO_ROOT / "config" / "ensemble_xgb_rf_dynamic_candidate.json"
    if not static_config.exists():
        return

    target = discover_ensemble_test_target(REPO_ROOT)
    assert "error" not in target
    if target["source"] == "static_config":
        assert target["config_path"] == str(static_config)
        assert target["candidate_id"]
    else:
        assert target["source"] == "latest_gui_retrain"
        assert target["artifact_path"]


def test_load_ensemble_selected_threshold_without_full_metrics_parse() -> None:
    artifact_dir = (
        REPO_ROOT
        / "models"
        / "ensemble_gui_retrain_fixed"
        / "research_retrain_20260619_151158"
    )
    if not artifact_dir.exists():
        pytest.skip("ensemble fixture not present")
    threshold = load_ensemble_selected_threshold(artifact_dir)
    assert threshold == 0.4


def test_resolve_dataset_path_supports_repo_relative_paths() -> None:
    rel = "data/processed/nifty_option_chain_cost_aware_edge_dataset_20260610_121730.csv"
    resolved = resolve_dataset_path(REPO_ROOT, rel)
    if Path(REPO_ROOT / rel).is_file():
        assert resolved == str((REPO_ROOT / rel).resolve())


def test_discover_ensemble_test_target_honors_session_artifact(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "research_retrain_test"
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "retrain_all_models_xgb_rf_ensemble_profitable_trade_label.pkl"
    artifact.write_bytes(b"stub")
    (artifact_dir / "retrain_all_models_xgb_rf_ensemble_profitable_trade_label_feature_order.json").write_text(
        json.dumps({"feature_order": ["open", "ltp"]}),
        encoding="utf-8",
    )
    (artifact_dir / "retrain_all_models_xgb_rf_ensemble_profitable_trade_label_metrics.json").write_text(
        json.dumps({"selected_threshold_from_validation": {"threshold": 0.4}}),
        encoding="utf-8",
    )

    target = discover_ensemble_test_target(REPO_ROOT, last_artifact=str(artifact))
    assert target["artifact_path"] == str(artifact)
    assert target["selected_threshold"] == 0.4


def test_discover_ensemble_test_target_prefers_newer_gui_retrain_over_stale_static_config(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "config").mkdir(parents=True)
    static_artifact_dir = repo_root / "models" / "ensemble_gui_retrain_fixed" / "research_retrain_20260619_100000"
    static_artifact_dir.mkdir(parents=True)
    static_artifact = static_artifact_dir / "retrain_all_models_xgb_rf_ensemble_profitable_trade_label.pkl"
    static_artifact.write_bytes(b"old")
    static_config = repo_root / "config" / "ensemble_xgb_rf_dynamic_candidate.json"
    static_config.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "old_ensemble",
                        "enabled": True,
                        "model_name": "xgb_rf_ensemble",
                        "model_path": str(static_artifact.relative_to(repo_root)).replace("/", "\\"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    latest_dir = repo_root / "models" / "ensemble_gui_retrain_20260619_200000" / "research_retrain_20260619_200001"
    latest_dir.mkdir(parents=True)
    latest_artifact = latest_dir / "retrain_all_models_xgb_rf_ensemble_profitable_trade_label.pkl"
    latest_artifact.write_bytes(b"new")
    (latest_dir / "retrain_all_models_xgb_rf_ensemble_profitable_trade_label_metrics.json").write_text(
        json.dumps({"selected_threshold_from_validation": {"threshold": 0.4}}),
        encoding="utf-8",
    )

    target = discover_ensemble_test_target(repo_root)
    assert target["source"] == "latest_gui_retrain"
    assert Path(target["artifact_path"]) == latest_artifact


def test_discover_ensemble_test_target_reads_dataset_hint_from_metadata(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifact_dir = repo_root / "models" / "ensemble_gui_retrain_20260620_100000" / "research_retrain_20260620_100001"
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "retrain_all_models_xgb_rf_ensemble_profitable_trade_label.pkl"
    artifact.write_bytes(b"ensemble")
    dataset = repo_root / "data" / "processed" / "ensemble_training.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("timestamp,ltp,symbol\n", encoding="utf-8")
    (artifact_dir / "retrain_all_models_xgb_rf_ensemble_profitable_trade_label_metadata.json").write_text(
        json.dumps({"dataset_path": str(dataset)}),
        encoding="utf-8",
    )

    target = discover_ensemble_test_target(repo_root, last_artifact=str(artifact))
    assert target["dataset_hint"] == str(dataset)


def test_discover_ensemble_test_target_reads_dataset_hint_through_wrapper_config(tmp_path: Path) -> None:
    repo_root = tmp_path
    wrapper_dir = repo_root / "models" / "candidates" / "ensemble_wrapper"
    source_dir = repo_root / "models" / "ensemble_gui_retrain_20260620_100000" / "research_retrain_20260620_100001"
    wrapper_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)

    dataset = repo_root / "data" / "processed" / "ensemble_training.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("timestamp,ltp,symbol\n", encoding="utf-8")
    artifact = source_dir / "retrain_all_models_xgb_rf_ensemble_profitable_trade_label.pkl"
    artifact.write_bytes(b"ensemble")
    (source_dir / "retrain_all_models_xgb_rf_ensemble_profitable_trade_label_metadata.json").write_text(
        json.dumps({"dataset_path": str(dataset)}),
        encoding="utf-8",
    )
    (wrapper_dir / "preprocessing_metadata.json").write_text(
        json.dumps({"source_artifact_dir": str(source_dir.relative_to(repo_root)).replace("/", "\\")}),
        encoding="utf-8",
    )

    config = repo_root / "config" / "ensemble_xgb_rf_dynamic_candidate.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "ensemble_wrapper",
                        "enabled": True,
                        "model_name": "xgb_rf_ensemble",
                        "artifact_dir": str(wrapper_dir.relative_to(repo_root)).replace("/", "\\"),
                        "model_path": str(artifact.relative_to(repo_root)).replace("/", "\\"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    target = discover_ensemble_test_target(repo_root, last_config=str(config))
    assert target["dataset_hint"] == str(dataset)
