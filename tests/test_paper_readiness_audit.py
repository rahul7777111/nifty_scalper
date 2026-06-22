from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def test_paper_readiness_fails_if_schema_mismatch_exists() -> None:
    payload = retrain._build_paper_readiness_payload(
        {
            "baseline_metrics": {},
            "enriched_metrics": {
                "paper_execution_filters": {"after": {"trade_count": 600}},
                "calibration_audit": {"calibration_gate": "PASS"},
                "probability_monotonicity": {"gate": "PASS"},
                "daily_pnl_stability": {"gate": "PASS"},
            },
            "final_decision_matrix": {
                "chronology_gate": "PASS",
                "leakage_gate": "PASS",
                "same_period_gate": "PASS",
                "cost_stress_gate": "PASS",
                "fold_stability_gate": "PASS",
                "threshold_robustness_gate": "PASS",
            },
        },
        {"live_feature_audit": []},
    )
    assert payload["decision_matrix"]["final_verdict"] != "PRODUCTION_READY"
