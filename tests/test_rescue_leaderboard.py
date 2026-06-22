from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def test_rescue_leaderboard_not_ranked_by_roc_auc_alone() -> None:
    ranked = retrain.build_rescue_leaderboard(
        [
            {
                "experiment_name": "roc_only",
                "model_family": "logistic_regression",
                "feature_set": "live_computable_only",
                "ROC-AUC": 0.99,
                "PF": 0.7,
                "Sharpe": -0.2,
                "median_fold_pf": 0.7,
                "cost_stress_0_25_pf": 0.7,
                "daily_pnl_gate": "FAIL",
                "live_schema_gate": "FAIL",
                "paper_candidate_allowed": False,
                "shadow_candidate_allowed": False,
            },
            {
                "experiment_name": "stable_candidate",
                "model_family": "logistic_regression",
                "feature_set": "live_computable_only",
                "ROC-AUC": 0.70,
                "PF": 1.2,
                "Sharpe": 0.8,
                "median_fold_pf": 1.1,
                "cost_stress_0_25_pf": 1.05,
                "daily_pnl_gate": "PASS",
                "live_schema_gate": "PASS",
                "paper_candidate_allowed": False,
                "shadow_candidate_allowed": True,
            },
        ]
    )
    assert ranked[0]["experiment_name"] == "stable_candidate"
