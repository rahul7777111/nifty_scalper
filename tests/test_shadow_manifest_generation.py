from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def test_shadow_manifest_payload_always_blocks_production() -> None:
    payload = retrain._shadow_manifest_payload(
        {
            "model_family": "logistic_regression",
            "feature_set": "live_computable_only",
            "label": "profitable_trade_label",
            "selected_threshold": 0.6,
            "shadow_candidate_allowed": True,
            "paper_candidate_allowed": False,
            "final_verdict": "SHADOW_RESCUE_CANDIDATE",
            "calibration_gate": "PASS",
            "monotonicity_gate": "PASS",
            "daily_pnl_gate": "PASS",
            "live_schema_gate": "PASS",
            "failed_gates": [],
        },
        {"model_path": "model.pkl", "holdout_split": {}, "calibration_fit": {"mode": "sigmoid"}},
        dataset_path="dataset.csv",
        feature_names=["feature_a"],
        experiment_name="demo",
    )
    assert payload["shadow_candidate_allowed"] is True
    assert payload["paper_candidate_allowed"] is False
    assert payload["production_adoption_allowed"] is False
