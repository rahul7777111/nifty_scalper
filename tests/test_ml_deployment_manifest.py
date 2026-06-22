from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_model_registry import load_deployment_manifest


class DummyModel:
    def predict_proba(self, X):
        return [[0.2, 0.8] for _ in range(len(X))]


class DummyBundle:
    def __init__(self):
        self.model = DummyModel()
        self.scaler_mean = [0.0, 0.0]
        self.scaler_std = [1.0, 1.0]


def _manifest(tmp_path: Path, verdict: str = "PAPER_TRADE_CANDIDATE_LOW_CONFIDENCE") -> Path:
    model_path = tmp_path / "model.pkl"
    joblib.dump(DummyBundle(), model_path)
    manifest = {
        "model_id": "demo-1",
        "model_family": "logistic_regression",
        "feature_set_name": "live_computable_only",
        "label_name": "profitable_trade_label",
        "selected_threshold": 0.65,
        "model_path": "model.pkl",
        "selected_feature_list": ["feature_a", "feature_b"],
        "rejected_feature_list": [],
        "training_dataset_path": "data.csv",
        "training_date_range": {},
        "validation_date_range": {},
        "test_date_range": {},
        "walk_forward_summary": {},
        "gate_summary": {},
        "paper_readiness_verdict": verdict,
        "production_adoption_allowed": False,
        "created_at": "2026-06-05T00:00:00+05:30",
        "report_paths": {},
    }
    path = tmp_path / "deployment_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_valid_manifest_loads_successfully(tmp_path: Path) -> None:
    manifest = load_deployment_manifest(_manifest(tmp_path))
    assert manifest.model_id == "demo-1"


def test_invalid_manifest_rejected_when_model_missing(tmp_path: Path) -> None:
    path = tmp_path / "deployment_manifest.json"
    path.write_text(json.dumps({"model_id": "x", "model_family": "y", "feature_set_name": "z", "label_name": "a", "selected_threshold": 0.5, "model_path": "missing.pkl", "selected_feature_list": ["f"], "paper_readiness_verdict": "PAPER_TRADE_CANDIDATE_LOW_CONFIDENCE", "production_adoption_allowed": False}), encoding="utf-8")
    try:
        load_deployment_manifest(path)
        assert False, "expected failure"
    except FileNotFoundError:
        pass

