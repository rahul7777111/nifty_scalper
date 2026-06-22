from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def _rows() -> list[dict]:
    return [
        {"gate_name": "live_schema_gate", "status": "FAIL", "short_reason": "schema mismatch", "recommended_action": "align schema"},
        {"gate_name": "cost_stress_gate", "status": "FAIL", "short_reason": "cost sensitive", "recommended_action": "stress test"},
        {"gate_name": "fold_stability_gate", "status": "FAIL", "short_reason": "unstable folds", "recommended_action": "stabilize folds"},
        {"gate_name": "threshold_robustness_gate", "status": "FAIL", "short_reason": "lucky threshold", "recommended_action": "reduce threshold fragility"},
    ]


def test_live_schema_ranks_above_model_tuning() -> None:
    plan = retrain._improvement_priority_plan(_rows(), comparison_payload={})
    assert plan[0]["issue"] == "Align to live schema"
    assert plan[0]["should_run_before_more_model_training"] is True


def test_fold_stability_priority_is_before_threshold_tuning() -> None:
    plan = retrain._improvement_priority_plan(_rows(), comparison_payload={})
    issues = [item["issue"] for item in plan]
    assert issues.index("Stabilize walk-forward folds") < issues.index("Investigate failed gate")
