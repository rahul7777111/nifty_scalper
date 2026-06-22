from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def test_optional_model_dependency_is_skipped_cleanly() -> None:
    ok, reason = retrain._optional_dependency_status("lightgbm")
    assert ok or str(reason).startswith("SKIPPED_OPTIONAL_DEPENDENCY_MISSING")


def test_probability_monotonicity_fail_should_not_rank_as_safe_candidate() -> None:
    verdict = retrain._paper_candidate_verdict(
        {
            "trade_metrics": {"profit_factor": 1.4, "sharpe": 1.1},
            "fold_stability": {"profitable_folds": 4, "worst_fold_pf": 1.0, "median_fold_pf": 1.2},
            "paper_execution_filters": {"after": {"trade_count": 900, "profit_factor": 1.3, "sharpe": 0.9}},
            "threshold_robustness": {"chosen_threshold_is_robust": True},
            "probability_monotonicity": {"gate": "FAIL"},
            "calibration_audit": {"calibration_gate": "PASS"},
            "daily_pnl_stability": {"gate": "PASS"},
        },
        live_schema_pass=True,
        live_computable_pass=True,
    )
    assert verdict == "RESEARCH_ONLY_WEAK_EDGE"
