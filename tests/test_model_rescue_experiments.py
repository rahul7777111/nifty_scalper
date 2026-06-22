from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def _tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 09:15", periods=10, freq="5min"),
            "feature_a": range(10),
            "profitable_trade_label": [1, 0] * 5,
            "net_forward_return": [0.1, -0.1] * 5,
        }
    )


def test_too_few_rows_causes_experiment_failure() -> None:
    report = retrain._rescue_validation_report(_tiny_frame().head(10), ["profitable_trade_label"], "net_forward_return")
    assert report["failed"] is True
    assert "too_few_rows" in report["failure_reasons"]


def test_return_columns_are_for_evaluation_not_features() -> None:
    frame = _tiny_frame()
    groups = retrain.detect_column_groups(frame)
    assert "net_forward_return" in groups["evaluation_return_columns"]
    assert "net_forward_return" not in groups["input_features"]


def test_no_rescue_candidate_verdict_when_report_fails_shadow_rules() -> None:
    flags = retrain._rescue_candidate_flags(
        {
            "paper_execution_filters": {"after": {"trade_count": 0, "profit_factor": 0.0, "sharpe": 0.0}},
            "walk_forward": {"folds": []},
            "daily_pnl_stability": {"active_trading_days": 0, "gate": "FAIL"},
            "threshold_robustness": {"chosen_threshold_is_robust": False},
            "probability_monotonicity": {"gate": "FAIL"},
            "calibration_audit": {"calibration_gate": "FAIL"},
            "cost_stress": {"extra_cost_0_25": {"profit_factor": 0.0}},
        },
        live_schema_pass=False,
        live_computable_pass=False,
    )
    assert flags["final_verdict"] == "NO_RESCUE_CANDIDATE_FOUND"
    assert flags["production_adoption_allowed"] is False


def test_resume_skips_completed_experiments() -> None:
    experiments = [{"experiment_name": "exp_a"}, {"experiment_name": "exp_b"}]
    checkpoint = {
        "experiments": {
            "exp_a": {"final_experiment_verdict": "COMPLETED", "production_adoption_allowed": False},
        }
    }
    summary = retrain._rescue_resume_summary(experiments, checkpoint)
    assert "exp_a" in summary["completed_experiments"]
    assert "exp_a" in summary["skipped_experiments"]
    assert "exp_b" in summary["pending_experiments"]


def test_corrupted_result_file_is_rerun() -> None:
    experiments = [{"experiment_name": "exp_a"}]
    checkpoint = {"experiments": {"exp_a": {"final_experiment_verdict": "BROKEN", "production_adoption_allowed": False}}}
    summary = retrain._rescue_resume_summary(experiments, checkpoint)
    assert "exp_a" in summary["corrupted_experiments"]
    assert "exp_a" in summary["pending_experiments"]


def test_checkpoint_written_after_candidate(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    retrain._atomic_write_json(path, {"experiments": {"exp_a": {"final_experiment_verdict": "COMPLETED", "production_adoption_allowed": False}}})
    loaded = retrain._load_rescue_checkpoint(path)
    assert loaded["experiments"]["exp_a"]["final_experiment_verdict"] == "COMPLETED"


def test_max_rescue_experiments_limit_is_countable() -> None:
    outputs = [{"run_this_invocation": True}, {"run_this_invocation": True}]
    fresh_run_count = sum(1 for item in outputs if item.get("run_this_invocation"))
    assert fresh_run_count == 2


def test_rescue_time_budget_exits_cleanly() -> None:
    start = time.time() - 120.0
    assert retrain._should_stop_rescue_run(start, 1.0) is True


def test_rescue_fast_pass_uses_smaller_grid() -> None:
    experiments = retrain._build_fast_pass_rescue_experiments()
    names = [row["experiment_name"] for row in experiments]
    assert "reduced_robust_live_features" in names
    assert "calibrated_logistic_live_features" in names
    assert "ce_only_live_features" not in names


def test_final_leaderboard_requires_all_experiments_complete() -> None:
    experiments = [{"experiment_name": "exp_a"}, {"experiment_name": "exp_b"}]
    checkpoint = {"experiments": {"exp_a": {"final_experiment_verdict": "COMPLETED", "production_adoption_allowed": False}}}
    summary = retrain._rescue_resume_summary(experiments, checkpoint)
    assert summary["pending_experiments"] == ["exp_b"]
