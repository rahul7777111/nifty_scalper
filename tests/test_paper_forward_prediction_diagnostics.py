"""Tests for paper-forward prediction diagnostics and probability extraction."""

from __future__ import annotations

import json
import math
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from candidate_router import route_candidate_decision, _predict_confidence_from_artifact
from paper_forward_predict import (
    build_aligned_feature_row,
    predict_confidence_from_artifact,
    validate_feature_vector_quality,
    _extract_probability,
    _find_positive_class_index,
)


def _mk_artifact(
    tmp: Path,
    *,
    features: List[str],
    classes: np.ndarray,
    coef: np.ndarray | None = None,
    candidate_id: str = "test_cand",
) -> Path:
    art = tmp / candidate_id
    art.mkdir(parents=True, exist_ok=True)
    X = np.random.randn(40, len(features))
    y = (X[:, 0] > 0).astype(int)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = LogisticRegression(max_iter=200)
    model.classes_ = classes
    if coef is not None:
        model.coef_ = coef
        model.intercept_ = np.array([0.0])
        model.n_features_in_ = len(features)
    else:
        model.fit(Xs, y)
    bundle = {"model": model, "scaler": scaler, "model_type": "logistic", "n_features": len(features)}
    with (art / "model.pkl").open("wb") as fh:
        pickle.dump(bundle, fh)
    (art / "feature_schema.json").write_text(json.dumps({"features": features}), encoding="utf-8")
    (art / "candidate_profile.json").write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "model_name": "logistic_regression",
                "preset_family": "TEST",
                "side_policy": "PE_ONLY",
                "feature_set_name": "live",
                "target_name": "edge",
                "dynamic_presets": {},
                "selection_policy": {},
                "risk_policy": {},
                "cost_policy": {},
                "validation_metrics": {},
                "gate_results": {},
                "live_computable_features": features,
                "threshold_policy": {"entry_threshold": 0.35},
                "artifact_paths": {"model_pkl": str(art / "model.pkl")},
            }
        ),
        encoding="utf-8",
    )
    return art


def _snap_for_features(features: List[str], value: float = 1.5) -> Dict[str, Any]:
    snap: Dict[str, Any] = {"spot": 23500.0, "option_type": "PE"}
    for f in features:
        snap[f] = value + (hash(f) % 7) * 0.01
    return snap


def test_predict_proba_class_order_01():
    features = ["f0", "f1", "f2"]
    model = LogisticRegression()
    model.classes_ = np.array([0, 1])
    model.coef_ = np.array([[2.0, 0.0, 0.0]])
    model.intercept_ = np.array([0.0])
    model.n_features_in_ = 3
    X = np.array([[1.0, 0.0, 0.0]])
    raw, prob, method = _extract_probability(model, X)
    assert method == "predict_proba"
    assert prob > 0.5
    assert _find_positive_class_index(model) == 1


def test_predict_proba_class_order_10():
    class _FakeModel:
        classes_ = np.array([1, 0])

        def predict_proba(self, X):
            return np.array([[0.25, 0.75]])

    model = _FakeModel()
    assert _find_positive_class_index(model) == 0
    _, prob, method = _extract_probability(model, np.zeros((1, 2)))
    assert method == "predict_proba"
    assert abs(prob - 0.25) < 1e-9


def test_decision_function_conversion():
    from sklearn.svm import LinearSVC

    model = LinearSVC()
    X = np.array([[-1.0, 0.0], [1.0, 0.0]])
    y = np.array([0, 1])
    model.fit(X, y)
    raw, prob, method = _extract_probability(model, np.array([[1.0, 0.0]]))
    assert method == "decision_function_sigmoid"
    assert prob > 0.5


def test_missing_feature_blocks_prediction(tmp_path):
    features = ["a", "b", "c"]
    art = _mk_artifact(tmp_path, features=features, classes=np.array([0, 1]))
    snap = {"a": 1.0, "b": 2.0}
    res = predict_confidence_from_artifact(
        str(art / "model.pkl"),
        snap,
        feature_order=features,
        artifact_dir=str(art),
        cand_meta={"candidate_id": art.name},
    )
    assert res.get("error") == "FEATURE_VECTOR_INVALID"
    assert res.get("confidence") is None
    assert "c" in (res.get("missing_features") or [])


def test_all_zero_feature_vector_blocks_prediction(tmp_path):
    features = ["a", "b", "c"]
    art = _mk_artifact(tmp_path, features=features, classes=np.array([0, 1]))
    snap = {"a": 0.0, "b": 0.0, "c": 0.0}
    fd: Dict[str, Any] = {}
    build_aligned_feature_row(features, snap, fd)
    ok, reason, _ = validate_feature_vector_quality(features, fd)
    assert not ok
    assert reason == "FEATURE_VECTOR_INVALID"


def test_artifact_identity_mismatch_blocks_prediction(tmp_path):
    features = ["a", "b", "c"]
    art = _mk_artifact(tmp_path, features=features, classes=np.array([0, 1]), candidate_id="real_id")
    snap = _snap_for_features(features)
    res = predict_confidence_from_artifact(
        str(art / "model.pkl"),
        snap,
        feature_order=features,
        artifact_dir=str(art),
        cand_meta={
            "candidate_id": "wrong_id",
            "model_name": "xgboost",
            "preset_family": "OTHER",
            "side_policy": "CE_ONLY",
            "required_features": 3,
            "threshold": 0.99,
        },
        threshold=0.35,
    )
    assert res.get("error") == "ARTIFACT_IDENTITY_MISMATCH"
    assert res.get("confidence") is None


def test_valid_mock_model_returns_nonzero_confidence(tmp_path):
    features = [f"f{i}" for i in range(8)]
    art = _mk_artifact(tmp_path, features=features, classes=np.array([0, 1]), candidate_id="good_cand")
    snap = _snap_for_features(features, value=2.0)
    res = predict_confidence_from_artifact(
        str(art / "model.pkl"),
        snap,
        feature_order=features,
        artifact_dir=str(art),
        cand_meta={"candidate_id": "good_cand", "model_name": "logistic_regression", "preset_family": "TEST", "side_policy": "PE_ONLY"},
    )
    assert res.get("error") in (None, "")
    assert isinstance(res.get("confidence"), float)
    assert res["confidence"] > 0.0
    assert res.get("scaler_applied") is True


def test_xgboost_missing_returns_install_hint_not_model_output_invalid(tmp_path, monkeypatch):
    features = ["a", "b", "c"]
    art = _mk_artifact(tmp_path, features=features, classes=np.array([0, 1]), candidate_id="xgb_router")
    prof = json.loads((art / "candidate_profile.json").read_text(encoding="utf-8"))
    prof["model_name"] = "xgboost"
    (art / "candidate_profile.json").write_text(json.dumps(prof), encoding="utf-8")
    monkeypatch.setattr("paper_forward_predict.xgboost_available", lambda: False)
    snap = _snap_for_features(features)
    res = predict_confidence_from_artifact(
        str(art / "model.pkl"),
        snap,
        feature_order=features,
        artifact_dir=str(art),
        cand_meta={"candidate_id": "xgb_router", "model_name": "xgboost", "model_family": "xgboost", "preset_family": "TEST", "side_policy": "PE_ONLY"},
    )
    assert res.get("error") == "XGBOOST_NOT_INSTALLED"
    assert res.get("install_hint")
    assert "xgboost" in res.get("install_hint", "").lower()

    dec = route_candidate_decision(snap, None, None, "paper", "xgb_router", str(art), True)
    assert dec.get("no_trade_reason") == "XGBOOST_NOT_INSTALLED"
    assert dec.get("confidence") is None


def test_router_surfaces_feature_vector_invalid_not_zero_conf(tmp_path):
    features = ["x", "y", "z"]
    art = _mk_artifact(tmp_path, features=features, classes=np.array([0, 1]), candidate_id="router_cand")
    snap = {"x": 1.0, "y": 2.0, "option_type": "PE"}
    dec = route_candidate_decision(snap, None, None, "paper", "router_cand", str(art), True)
    assert dec.get("no_trade_reason") == "FEATURE_VECTOR_INVALID"
    assert dec.get("confidence") is None