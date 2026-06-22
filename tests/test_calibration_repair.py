from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def test_platt_and_isotonic_calibration_fit_only_on_validation() -> None:
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y_train = np.array([0, 0, 1, 1], dtype=int)
    X_val = np.array([[0.5], [2.5], [1.5], [3.5]], dtype=float)
    y_val = np.array([0, 1, 0, 1], dtype=int)
    for model_name in ("logistic_regression_platt", "logistic_regression_isotonic"):
        _, info = retrain._fit_model_with_validation_calibration(model_name, X_train, y_train, X_val, y_val)
        assert info["fit_on_validation_only"] is True
        assert info["validation_rows_used"] == len(y_val)
