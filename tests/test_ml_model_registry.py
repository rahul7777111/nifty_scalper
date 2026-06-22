from __future__ import annotations

import sys
from pathlib import Path

from test_ml_deployment_manifest import _manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_model_registry import MLModelRegistry


def test_missing_feature_blocks_prediction(tmp_path: Path) -> None:
    registry = MLModelRegistry(_manifest(tmp_path))
    ok, missing, invalid, forbidden = registry.validate_runtime_features({"feature_a": 1.0})
    assert not ok
    assert "feature_b" in missing


def test_nan_feature_blocks_prediction(tmp_path: Path) -> None:
    registry = MLModelRegistry(_manifest(tmp_path))
    ok, missing, invalid, forbidden = registry.validate_runtime_features({"feature_a": 1.0, "feature_b": float("nan")})
    assert not ok
    assert "feature_b" in invalid


def test_valid_prediction_runs(tmp_path: Path) -> None:
    registry = MLModelRegistry(_manifest(tmp_path))
    prob = registry.predict_proba({"feature_a": 1.0, "feature_b": 2.0})
    assert 0.0 <= prob <= 1.0

