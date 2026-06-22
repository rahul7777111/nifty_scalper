#!/usr/bin/env python3
"""Verify XGBoost inference uses correct input types per model class."""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from candidate_router import _PF_MODEL_CACHE, _predict_confidence_from_artifact  # noqa: E402


class _FakeBooster:
    __module__ = "xgboost.core"

    def predict(self, dmatrix):
        assert type(dmatrix).__name__ == "DMatrix", "Booster must receive DMatrix"
        return np.array([0.42])


class _FakeXGBClassifier:
    __module__ = "xgboost.sklearn"

    def predict_proba(self, x):
        assert type(x).__name__ != "DMatrix", "XGBClassifier must not receive DMatrix"
        return np.array([[0.3, 0.7]])


class _BrokenClassifier:
    __module__ = "xgboost.sklearn"

    def predict_proba(self, x):
        raise TypeError("Not supported type for data.<class 'xgboost.core.DMatrix'>")


class DMatrix:
    created = 0

    def __init__(self, *args, **kwargs):
        type(self).created += 1


def _predict_with_model(inner_model, *, candidate_id: str = "test_xgb") -> dict:
    from unittest.mock import mock_open

    _PF_MODEL_CACHE.clear()
    snap = {f"f{i}": float(i) for i in range(3)}
    fake_path = REPO / "artifacts" / "_verify_xgb_tmp" / "model.pkl"
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": inner_model}

    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=b"fake")):
            with patch("pickle.load", return_value=bundle):
                return _predict_confidence_from_artifact(
                    str(fake_path),
                    snap,
                    feature_order=["f0", "f1", "f2"],
                    cand_meta={"candidate_id": candidate_id},
                )


def main() -> int:
    errors: list[str] = []

    fake_xgb = type(sys)("xgboost")
    fake_xgb.Booster = _FakeBooster
    fake_xgb.DMatrix = DMatrix

    with patch.dict(sys.modules, {"xgboost": fake_xgb}):
        DMatrix.created = 0
        res_booster = _predict_with_model(_FakeBooster())
        if DMatrix.created < 1:
            errors.append("Booster path should create DMatrix")
        if res_booster.get("error"):
            errors.append(f"Booster path failed: {res_booster.get('error')}")

        DMatrix.created = 0
        res_clf = _predict_with_model(_FakeXGBClassifier())
        if DMatrix.created != 0:
            errors.append("XGBClassifier path must not create DMatrix")
        if res_clf.get("error"):
            errors.append(f"XGBClassifier path failed: {res_clf.get('error')}")
        if abs(float(res_clf.get("confidence") or 0) - 0.7) > 1e-6:
            errors.append(f"XGBClassifier confidence unexpected: {res_clf}")

        res_exc = _predict_with_model(_BrokenClassifier(), candidate_id="broken_xgb")
        if res_exc.get("error") != "PREDICT_EXCEPTION":
            errors.append(f"expected PREDICT_EXCEPTION, got {res_exc.get('error')}")
        if not res_exc.get("route_error"):
            errors.append("route_error should be True on predict exception")

    if errors:
        print("verify_xgboost_inference_handling FAILED")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("verify_xgboost_inference_handling PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())