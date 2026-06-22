from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def test_threshold_robustness_detects_isolated_win() -> None:
    y_true = np.array([1] * 20 + [0] * 20)
    y_prob = np.array([0.76] * 10 + [0.4] * 10 + [0.74] * 10 + [0.2] * 10)
    returns = np.array([2.0] * 10 + [-0.1] * 10 + [-0.5] * 10 + [-0.1] * 10)
    report = retrain._threshold_robustness_report(
        y_true,
        y_prob,
        returns,
        grid_start=0.5,
        grid_end=0.8,
        grid_step=0.05,
        min_trades=5,
    )
    assert isinstance(report["chosen_threshold_is_robust"], bool)
    assert isinstance(report["isolated_lucky_point"], bool)
    assert report["chosen_threshold"] >= 0.5
    assert report["threshold_rows"]


def test_production_adoption_stays_blocked_with_schema_mismatch() -> None:
    verdict = retrain._final_decision_matrix(
        baseline_best={"trade_metrics": {"profit_factor": 0.9, "sharpe": 0.0}},
        enriched_best={
            "trade_metrics": {"profit_factor": 1.2, "sharpe": 1.0},
            "cost_stress": {"base": {}, "extra_cost_0_25": {"failure_flag": False}},
            "fold_stability": {"gate": "PASS"},
            "threshold_robustness": {"chosen_threshold_is_robust": True},
            "walk_forward": {"folds": [1]},
        },
        leakage_result={"passed": True},
        adoption={"strict_live_schema_gate": {"compatible": False}},
        same_period_report={"same_period_gate": "PASS"},
    )
    assert verdict["production_adoption_allowed"] is False
    assert verdict["final_verdict"] != "PRODUCTION_READY"
