"""Tests for predict_proba binary/multiclass handling in MLModelRegistry.

Safety: paper-only / shadow-only / research. No real orders placed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_model_registry import MLModelRegistry


# ---------------------------------------------------------------------------
# Model / bundle helpers
# ---------------------------------------------------------------------------


class DummyModelClasses01:
    """Binary model with classes_=[0, 1]; positive at index 1."""

    def predict_proba(self, X):
        # Return 2-column: [P(class=0), P(class=1)]
        return [[0.3, 0.7] for _ in range(len(X))]


class DummyModelClasses10:
    """Binary model with classes_=[1, 0]; positive (1) at index 0 — edge case."""

    def __init__(self):
        self.classes_ = np.array([1, 0])  # intentionally reversed column order

    def predict_proba(self, X):
        return [[0.7, 0.3] for _ in range(len(X))]


class DummyModelNoClasses:
    """Binary model without classes_ attribute."""

    def predict_proba(self, X):
        return [[0.2, 0.8] for _ in range(len(X))]


class DummyModelMulticlass:
    """Multi-class model (3 classes); must fail closed."""

    def __init__(self):
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, X):
        return [[0.1, 0.6, 0.3] for _ in range(len(X))]


class DummyModelMulticlassNoClasses:
    """Multi-class model returning 4 columns but no classes_ — must fail closed."""

    def predict_proba(self, X):
        return [[0.1, 0.4, 0.3, 0.2] for _ in range(len(X))]


class DummyModelPredictOnly:
    """Model with only predict(), no predict_proba — wrapped by fallback."""

    def predict(self, X):
        return [1 for _ in range(len(X))]


class DummyModelPredictContinuous:
    """Model with only predict() returning raw scores — clamped to [0, 1]."""

    def predict(self, X):
        return [1.5 for _ in range(len(X))]


class DummyModelSingleClass:
    """Model trained on a single class only."""

    def __init__(self):
        self.classes_ = np.array([0])

    def predict_proba(self, X):
        return [[1.0] for _ in range(len(X))]


class OneColModel:
    """Binary model returning a single probability column — rare edge case."""

    def predict_proba(self, X):
        return [[0.5] for _ in range(len(X))]


class DummyBundle:
    def __init__(self, model):
        self.model = model
        self.scaler_mean = None
        self.scaler_std = None


def _registry(tmp_path: Path, model) -> MLModelRegistry:
    model_path = tmp_path / "model.pkl"
    joblib.dump(DummyBundle(model), model_path)
    manifest = {
        "model_id": "test-binary-1",
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
        "paper_readiness_verdict": "PAPER_TRADE_CANDIDATE_LOW_CONFIDENCE",
        "production_adoption_allowed": False,
        "created_at": "2026-06-05T00:00:00+05:30",
        "report_paths": {},
    }
    path = tmp_path / "deployment_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return MLModelRegistry(path)


FEATURES = {"feature_a": 1.0, "feature_b": 2.0}


# ---------------------------------------------------------------------------
# Binary: standard [0, 1] column order
# ---------------------------------------------------------------------------


def test_binary_standard_classes_returns_positive_prob(tmp_path: Path) -> None:
    """classes_=[0,1] → index 1 is positive; probs[0][1] should be 0.7."""
    registry = _registry(tmp_path, DummyModelClasses01())
    prob = registry.predict_proba(FEATURES)
    assert 0.0 <= prob <= 1.0
    # The DummyModelClasses01 returns P(class=1)=0.7
    assert abs(prob - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# Binary: reversed column order
# ---------------------------------------------------------------------------


def test_binary_reversed_classes_returns_positive_prob(tmp_path: Path) -> None:
    """classes_=[1,0] → index 0 is positive; probs[0][0] should be 0.7."""
    registry = _registry(tmp_path, DummyModelClasses10())
    prob = registry.predict_proba(FEATURES)
    assert 0.0 <= prob <= 1.0
    # DummyModelClasses10 returns col0=0.7 (positive), col1=0.3
    assert abs(prob - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# Binary: no classes_ attribute
# ---------------------------------------------------------------------------


def test_binary_no_classes_falls_back_to_second_column(tmp_path: Path) -> None:
    """No classes_ → fallback to index 1 for binary (legacy behaviour)."""
    registry = _registry(tmp_path, DummyModelNoClasses())
    prob = registry.predict_proba(FEATURES)
    assert 0.0 <= prob <= 1.0
    assert abs(prob - 0.8) < 1e-6  # DummyModelNoClasses returns P(class=1)=0.8


# ---------------------------------------------------------------------------
# Binary: no classes_, single-column output
# ---------------------------------------------------------------------------


def test_binary_no_classes_single_column_returns_first_column(tmp_path: Path) -> None:
    """Binary fallback when only 1 column returned → use column 0."""
    registry = _registry(tmp_path, OneColModel())
    prob = registry.predict_proba(FEATURES)
    assert abs(prob - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Binary: single-class model
# ---------------------------------------------------------------------------


def test_binary_single_class_returns_without_error(tmp_path: Path) -> None:
    """Single-class model → _find_positive_class_index returns None → fallback."""
    registry = _registry(tmp_path, DummyModelSingleClass())
    prob = registry.predict_proba(FEATURES)
    # Single class: fallback path; should return column 0 = 1.0
    assert abs(prob - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Multiclass: with classes_ attribute
# ---------------------------------------------------------------------------


def test_multiclass_with_classes_fails_closed_with_clear_error(tmp_path: Path) -> None:
    """3-class model with classes_ should raise RuntimeError (fail closed)."""
    registry = _registry(tmp_path, DummyModelMulticlass())
    try:
        registry.predict_proba(FEATURES)
        assert False, "expected RuntimeError for multi-class (>2 labels)"
    except RuntimeError as exc:
        assert "multi-class model detected (3 classes)" in str(exc)


# ---------------------------------------------------------------------------
# Multiclass: no classes_ attribute, >2 columns
# ---------------------------------------------------------------------------


def test_multiclass_no_classes_fails_closed_with_clear_error(tmp_path: Path) -> None:
    """4-column output without classes_ should raise RuntimeError."""
    registry = _registry(tmp_path, DummyModelMulticlassNoClasses())
    try:
        registry.predict_proba(FEATURES)
        assert False, "expected RuntimeError for multi-class without classes_"
    except RuntimeError as exc:
        assert "multi-class output" in str(exc).lower() or "positive class index" in str(exc).lower()


# ---------------------------------------------------------------------------
# Fallback: model with only predict(), no predict_proba
# ---------------------------------------------------------------------------


def test_predict_only_model_returns_clamped_prediction(tmp_path: Path) -> None:
    """Model with only predict() (returns 0/1) should return clamped value."""
    registry = _registry(tmp_path, DummyModelPredictOnly())
    prob = registry.predict_proba(FEATURES)
    # predict() returns [1]; clamped to [0,1] → 1.0
    assert abs(prob - 1.0) < 1e-6


def test_predict_only_model_continuous_score_clamped(tmp_path: Path) -> None:
    """predict() returning continuous score > 1.0 is clamped to 1.0."""
    registry = _registry(tmp_path, DummyModelPredictContinuous())
    prob = registry.predict_proba(FEATURES)
    # predict() returns 1.5, clamped → 1.0
    assert abs(prob - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Edge: healthy check
# ---------------------------------------------------------------------------


def test_predict_proba_raises_when_registry_unhealthy(tmp_path: Path) -> None:
    """Unhealthy registry (no manifest) should raise RuntimeError."""
    registry = MLModelRegistry(None)
    try:
        registry.predict_proba(FEATURES)
        assert False, "expected RuntimeError for unhealthy registry"
    except RuntimeError as exc:
        assert "not healthy" in str(exc)


# ---------------------------------------------------------------------------
# Edge: feature validation failure
# ---------------------------------------------------------------------------


def test_predict_proba_raises_on_missing_features(tmp_path: Path) -> None:
    """Missing runtime features should raise RuntimeError."""
    registry = _registry(tmp_path, DummyModelClasses01())
    try:
        registry.predict_proba({"feature_a": 1.0})  # feature_b missing
        assert False, "expected RuntimeError for missing features"
    except RuntimeError as exc:
        assert "runtime feature validation failed" in str(exc)


# ---------------------------------------------------------------------------
# Sanity: returned probability is always in [0, 1]
# ---------------------------------------------------------------------------


def test_returned_probability_is_bounded(tmp_path: Path) -> None:
    """All valid binary paths should return probability in [0, 1]."""
    registry = _registry(tmp_path, DummyModelClasses01())
    for _ in range(10):
        prob = registry.predict_proba(FEATURES)
        assert 0.0 <= prob <= 1.0, f"probability {prob} out of bounds"