from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def test_calibration_warning_for_poorly_calibrated_buckets() -> None:
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.52, 0.54, 0.90, 0.92, 0.94, 0.96])
    returns = np.array([0.1, 0.1, -0.2, -0.2, -0.2, -0.2])
    report = retrain._calibration_audit(y_true, y_prob, returns)
    assert report["calibration_gate"] == "FAIL"


def test_probability_monotonicity_fails_when_top_deciles_are_not_better() -> None:
    y_prob = np.linspace(0.5, 0.99, 20)
    returns = np.array([1.0] * 10 + [-1.0] * 10)
    report = retrain._probability_monotonicity_report(y_prob, returns)
    assert report["gate"] == "FAIL"
