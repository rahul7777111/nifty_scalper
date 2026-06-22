from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def test_dataset_candidate_rejects_temp_and_missing_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "pytest_temp.csv"
    pd.DataFrame({"a": [1], "profitable_trade_label": [1], "net_forward_return": [0.1]}).to_csv(path, index=False)
    report = retrain._dataset_candidate_report(path)
    assert report["status"] == "REJECTED"


def test_strict_mode_never_allows_production_ready_verdict() -> None:
    verdict = retrain._paper_candidate_verdict(
        {
            "trade_metrics": {"profit_factor": 2.0, "sharpe": 2.0},
            "fold_stability": {"profitable_folds": 4, "worst_fold_pf": 1.2, "median_fold_pf": 1.3},
            "paper_execution_filters": {"after": {"trade_count": 800, "profit_factor": 1.5, "sharpe": 1.0}},
            "threshold_robustness": {"chosen_threshold_is_robust": True},
            "probability_monotonicity": {"gate": "PASS"},
            "calibration_audit": {"calibration_gate": "PASS"},
            "daily_pnl_stability": {"gate": "PASS"},
        },
        live_schema_pass=True,
        live_computable_pass=True,
    )
    assert verdict != "PRODUCTION_READY"

