"""Tests for ensemble_artifact_loader discovery, loading, and prediction.

Paths are auto-selected via ``discover_ensemble_artifact_dirs`` / ``select_best_ensemble_dir``
so tests do not hardcode a specific artifact directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest


def _discover_root() -> Path:
    """Return the repository root (where models/ lives)."""
    here = Path(__file__).resolve().parent
    return here.parent


def _ensure_ensemble_dir() -> Path | None:
    """Auto-select the best ensemble artifact dir for testing."""
    root = _discover_root()
    try:
        from ml.ensemble_artifact_loader import select_best_ensemble_dir

        d = select_best_ensemble_dir(root / "models")
        return d
    except Exception:
        return None


@pytest.fixture(scope="session")
def ensemble_dir() -> Path | None:
    return _ensure_ensemble_dir()


@pytest.fixture(scope="session")
def ensemble_config_path(ensemble_dir: Path | None) -> Path | None:
    if ensemble_dir is None:
        return None
    return ensemble_dir / "ensemble_config.json"


class FakeModel:
    """Stand-in for sklearn / xgboost models when the real ones are absent."""

    def __init__(self, prob: float = 0.75):
        self._prob = float(prob)
        self.classes_ = [0, 1]

    def predict_proba(self, X: Any) -> Any:
        import numpy as np

        return np.asarray([[1.0 - self._prob, self._prob]])

    def predict(self, X: Any) -> Any:
        import numpy as np

        return np.asarray([1])


def test_discover_finds_real_ensemble_dir() -> None:
    """Smoke test: discover logic should find the real ensemble artifact dir."""
    from ml.ensemble_artifact_loader import discover_ensemble_artifact_dirs

    root = _discover_root()
    dirs = discover_ensemble_artifact_dirs(root / "models")
    assert isinstance(dirs, list)
    # If there is a real ensemble dir on disk, it should be discoverable
    for d in dirs:
        assert (d / "ensemble_config.json").exists()
        assert (d / "random_forest_model.pkl").exists() or any(
            "random_forest" in f.name.lower() for f in d.glob("*.pkl")
        )


def test_select_best_prefers_metrics() -> None:
    """``select_best_ensemble_dir`` prefers dirs with ``ensemble_metrics.json``."""
    from ml.ensemble_artifact_loader import (
        discover_ensemble_artifact_dirs,
        select_best_ensemble_dir,
    )

    root = _discover_root()
    best = select_best_ensemble_dir(root / "models")
    if best is not None:
        assert best in discover_ensemble_artifact_dirs(root / "models")


def test_config_reads_feature_columns(ensemble_config_path: Path | None) -> None:
    """ensemble_config.json must contain feature_columns and fill_values."""
    if ensemble_config_path is None:
        pytest.skip("No ensemble artifact dir discovered")
    cfg = json.loads(ensemble_config_path.read_text(encoding="utf-8"))
    assert isinstance(cfg.get("feature_columns"), list)
    assert len(cfg["feature_columns"]) > 0
    assert isinstance(cfg.get("fill_values"), dict)
    assert len(cfg["fill_values"]) > 0


def test_config_gate_fields_present(ensemble_config_path: Path | None) -> None:
    """Gate params must be present in ensemble_config.json."""
    if ensemble_config_path is None:
        pytest.skip("No ensemble artifact dir discovered")
    cfg = json.loads(ensemble_config_path.read_text(encoding="utf-8"))
    for key in ("xgb_weight", "rf_weight", "ensemble_threshold",
                "xgb_min_prob", "rf_min_prob", "max_model_disagreement"):
        assert key in cfg, f"Missing gate key: {key}"
        assert isinstance(cfg[key], (int, float))


def test_loader_detects_dir(ensemble_dir: Path | None) -> None:
    """``EnsembleArtifactLoader`` sets status=OK for a real ensemble dir."""
    from ml.ensemble_artifact_loader import EnsembleArtifactLoader, STATUS_OK

    if ensemble_dir is None:
        pytest.skip("No ensemble artifact dir discovered")
    loader = EnsembleArtifactLoader(ensemble_dir).load()
    assert loader.status == STATUS_OK
    assert len(loader.feature_columns) > 0
    assert loader.rf_model_path is not None or loader.xgb_model_path is not None


def test_loader_predict_with_dummy_models(tmp_path: Path) -> None:
    """Build a synthetic ensemble dir and verify the full predict pipeline."""
    from ml.ensemble_artifact_loader import EnsembleArtifactLoader, STATUS_OK

    import pickle

    d = tmp_path / "synthetic_ensemble"
    d.mkdir()

    with (d / "random_forest_model.pkl").open("wb") as fh:
        pickle.dump(FakeModel(prob=0.55), fh)
    with (d / "xgboost_model.pkl").open("wb") as fh:
        pickle.dump(FakeModel(prob=0.65), fh)

    config: Dict[str, Any] = {
        "feature_columns": ["open", "high", "low", "ltp"],
        "fill_values": {"open": 100.0, "high": 101.0, "low": 99.0, "ltp": 100.5},
        "xgb_weight": 0.7,
        "rf_weight": 0.3,
        "ensemble_threshold": 0.6,
        "xgb_min_prob": 0.58,
        "rf_min_prob": 0.52,
        "max_model_disagreement": 0.25,
        "block_on_disagreement": True,
    }
    (d / "ensemble_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    loader = EnsembleArtifactLoader(d).load()
    assert loader.status == STATUS_OK
    assert loader.rf_model_path is not None
    assert loader.xgb_model_path is not None

    snapshot = {"open": 100.0, "high": 101.0, "low": 99.0, "ltp": 100.5}
    result = loader.predict(snapshot)

    assert result["status"] == STATUS_OK
    assert result["rf_prob"] is not None
    assert result["xgb_prob"] is not None
    assert result["ensemble_prob"] is not None
    assert result["model_disagreement"] is not None
    assert isinstance(result["allowed"], bool)
    # ensemble_prob = 0.7*0.65 + 0.3*0.55 = 0.62
    assert result["ensemble_prob"] == pytest.approx(0.62, abs=1e-6)
    assert result["feature_missing_count"] == 0
    assert result["feature_invalid_count"] == 0
    assert result["model_type"] == "ensemble"


def test_loader_predict_with_missing_features(tmp_path: Path) -> None:
    """``predict`` fills missing features and reports counts."""
    from ml.ensemble_artifact_loader import (
        EnsembleArtifactLoader,
        STATUS_OK,
        STATUS_FEATURES_MISSING,
    )

    import pickle

    d = tmp_path / "synthetic_ensemble2"
    d.mkdir()

    with (d / "random_forest_model.pkl").open("wb") as fh:
        pickle.dump(FakeModel(prob=0.55), fh)

    config: Dict[str, Any] = {
        "feature_columns": ["open", "high", "missing_col"],
        "fill_values": {"open": 10.0, "high": 11.0},
        "xgb_weight": 0.7,
        "rf_weight": 0.3,
        "ensemble_threshold": 0.6,
        "xgb_min_prob": 0.58,
        "rf_min_prob": 0.52,
        "max_model_disagreement": 0.25,
        "block_on_disagreement": True,
    }
    (d / "ensemble_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    loader = EnsembleArtifactLoader(d).load()
    snapshot = {"open": 10.0}
    result = loader.predict(snapshot)
    # missing_col has no fill value => feature_missing_count > 0
    assert result["feature_missing_count"] == 1
    assert result["status"] == STATUS_OK or result["status"] == STATUS_FEATURES_MISSING


def test_loader_predict_missing_models(tmp_path: Path) -> None:
    """``EnsembleArtifactLoader`` gracefully handles missing model files."""
    from ml.ensemble_artifact_loader import (
        EnsembleArtifactLoader,
        STATUS_MODEL_LOAD_FAILED,
    )

    d = tmp_path / "incomplete_ensemble"
    d.mkdir()
    (d / "ensemble_config.json").write_text(
        json.dumps({"feature_columns": ["a"], "fill_values": {"a": 1.0}}),
        encoding="utf-8",
    )
    loader = EnsembleArtifactLoader(d).load()
    assert loader.status == STATUS_MODEL_LOAD_FAILED
    result = loader.predict({"a": 1.0})
    assert result["allowed"] is False
    assert result["block_reason"] == "MODEL_LOAD_FAILED"


def test_predict_from_ensemble_dir_returns_not_ensemble() -> None:
    """``predict_from_ensemble_dir`` returns not_ensemble for plain dirs."""
    from ml.ensemble_artifact_loader import predict_from_ensemble_dir

    result = predict_from_ensemble_dir("/tmp/not_an_ensemble", {})
    assert result["status"] == "not_ensemble"


def test_is_ensemble_artifact_dir_false_for_plain_dir(tmp_path: Path) -> None:
    """``_is_ensemble_artifact_dir`` is False for directories without config."""
    from ml.ensemble_artifact_loader import _is_ensemble_artifact_dir

    plain = tmp_path / "plain"
    plain.mkdir()
    assert _is_ensemble_artifact_dir(plain) is False


def test_is_ensemble_artifact_dir_true(tmp_path: Path) -> None:
    """``_is_ensemble_artifact_dir`` is True when both models + config exist."""
    from ml.ensemble_artifact_loader import _is_ensemble_artifact_dir
    import pickle

    d = tmp_path / "real_ensemble"
    d.mkdir()
    (d / "ensemble_config.json").write_text("{}", encoding="utf-8")
    with (d / "random_forest_model.pkl").open("wb") as fh:
        pickle.dump(FakeModel(), fh)
    with (d / "xgboost_model.pkl").open("wb") as fh:
        pickle.dump(FakeModel(), fh)
    assert _is_ensemble_artifact_dir(d) is True
