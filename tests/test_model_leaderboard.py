from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def test_leaderboard_ranks_by_gate_safe_score_not_roc_auc_alone() -> None:
    rows = [
        {"model_family": "a", "ROC-AUC": 0.90, "PF": 0.8, "Sharpe": -0.2, "F1": 0.5, "max_drawdown": 10, "median_fold_pf": 0.8, "cost_stress_pass": False, "threshold_robustness_pass": True, "calibration_pass": True, "monotonicity_pass": True, "daily_pnl_pass": True, "regime_stability_pass": True, "live_schema_pass": True},
        {"model_family": "b", "ROC-AUC": 0.70, "PF": 1.3, "Sharpe": 1.0, "F1": 0.4, "max_drawdown": 5, "median_fold_pf": 1.2, "cost_stress_pass": True, "threshold_robustness_pass": True, "calibration_pass": True, "monotonicity_pass": True, "daily_pnl_pass": True, "regime_stability_pass": True, "live_schema_pass": True},
    ]
    leaderboard = retrain.build_model_leaderboard(rows)
    assert leaderboard[0]["model_family"] == "b"


def test_high_roc_auc_failed_cost_stress_is_not_top_ranked() -> None:
    rows = [
        {"model_family": "weak", "ROC-AUC": 0.95, "PF": 1.0, "Sharpe": 0.4, "F1": 0.6, "max_drawdown": 4, "median_fold_pf": 1.0, "cost_stress_pass": False, "threshold_robustness_pass": True, "calibration_pass": True, "monotonicity_pass": True, "daily_pnl_pass": True, "regime_stability_pass": True, "live_schema_pass": True},
        {"model_family": "stronger", "ROC-AUC": 0.75, "PF": 1.2, "Sharpe": 0.9, "F1": 0.45, "max_drawdown": 3, "median_fold_pf": 1.15, "cost_stress_pass": True, "threshold_robustness_pass": True, "calibration_pass": True, "monotonicity_pass": True, "daily_pnl_pass": True, "regime_stability_pass": True, "live_schema_pass": True},
    ]
    leaderboard = retrain.build_model_leaderboard(rows)
    assert leaderboard[0]["model_family"] == "stronger"

