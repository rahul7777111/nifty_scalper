from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def test_experiment_generator_creates_relevant_experiments() -> None:
    rows = [
        {"gate_name": "live_computable_feature_gate", "status": "FAIL"},
        {"gate_name": "cost_stress_gate", "status": "FAIL"},
        {"gate_name": "calibration_gate", "status": "FAIL"},
    ]
    experiments = retrain._experiment_generator(rows, dataset_path="data/processed/test.csv", output_dir="models/out")
    names = {item["experiment_name"] for item in experiments}
    assert "live_computable_only_retrain" in names
    assert "stricter_cost_and_execution_audit" in names
    assert "calibration_repair_audit" in names

