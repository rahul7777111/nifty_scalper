from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def _comparison_payload() -> dict:
    return {
        "dataset_path": "data/processed/test.csv",
        "improvement_degradation": {"f1_delta": -0.01},
        "baseline_metrics": {},
        "enriched_metrics": {
            "walk_forward": {"folds": [1, 2, 3]},
            "trade_metrics": {"profit_factor": 0.88, "sharpe": -0.6, "trade_count": 7219},
            "cost_stress": {
                "base": {"profit_factor": 0.88},
                "extra_cost_0_25": {"profit_factor": 0.72, "failure_flag": True},
                "extra_cost_0_50": {"profit_factor": 0.63, "failure_flag": True},
                "extra_cost_1_00": {"profit_factor": 0.41, "failure_flag": True},
            },
            "fold_stability": {"gate": "FAIL", "passing_folds": 1, "split_count": 4},
            "threshold_robustness": {"chosen_threshold": 0.7, "chosen_threshold_is_robust": False, "isolated_lucky_point": True},
            "calibration_audit": {"calibration_gate": "FAIL", "monotonic_win_rate": False, "buckets": [{"small_sample_warning": True}]},
            "probability_monotonicity": {"gate": "FAIL", "deciles": [{"average_return": -0.05}, {"average_return": -0.02}]},
            "daily_pnl_stability": {"gate": "FAIL", "day_win_rate": 0.4, "single_day_return_contribution": 0.51, "longest_losing_streak_days": 6},
            "regime_performance": {"groups": {"moneyness_bucket": {"ATM": {"trade_count": 800, "profit_factor": 0.7, "sharpe": -0.3}}}},
            "paper_execution_filters": {
                "before": {"trade_count": 7219},
                "after": {"trade_count": 0, "profit_factor": 0.0, "sharpe": 0.0},
            },
        },
        "same_period_report": {"same_period_gate": "PASS", "baseline_rows_dropped": 32168, "bs_rows_dropped": 0, "overlapping_date_range": {"start": "2024-12-03", "end": "2026-05-27"}},
        "leakage_audit_result": {"passed": True, "forbidden_columns_present": []},
        "schema_alignment_action_report": {
            "missing_live_features": ["ctx_iv", "ctx_delta"],
            "training_only_features": ["bs_iv"],
            "feature_order_matches_exactly": False,
        },
        "production_adoption_verdict": {"verdict": "RESEARCH_ONLY_ESTIMATED_GREEKS_NOT_ADOPTABLE"},
    }


def _paper_payload() -> dict:
    return {
        "decision_matrix": {
            "chronology_gate": "PASS",
            "leakage_gate": "PASS",
            "same_period_gate": "PASS",
            "live_computable_feature_gate": "FAIL",
            "live_schema_gate": "FAIL",
            "trading_metrics_gate": "FAIL",
            "cost_stress_gate": "FAIL",
            "fold_stability_gate": "FAIL",
            "threshold_robustness_gate": "FAIL",
            "calibration_gate": "FAIL",
            "probability_monotonicity_gate": "FAIL",
            "daily_pnl_stability_gate": "FAIL",
            "regime_stability_gate": "FAIL",
            "minimum_trades_gate": "FAIL",
            "paper_execution_filter_gate": "FAIL",
            "production_adoption_gate": "FAIL",
        },
        "regime_stability_report": {"gate": "FAIL", "positive_groups": 0, "total_groups_considered": 1, "failing_groups": ["moneyness_bucket:ATM"]},
    }


def _feature_payload() -> dict:
    return {"live_feature_audit": [{"feature_name": "feature_a", "computable_before_decision": True, "available_live": True}]}


def test_failed_fold_stability_creates_root_cause() -> None:
    table = retrain._failure_gate_table(paper_payload=_paper_payload(), comparison_payload=_comparison_payload(), feature_manifest_payload=_feature_payload())
    causes = retrain._failed_gate_root_causes(table)
    assert "FOLD_INSTABILITY" in causes["fold_stability_gate"]


def test_failed_live_schema_creates_feature_schema_mismatch_root_cause() -> None:
    table = retrain._failure_gate_table(paper_payload=_paper_payload(), comparison_payload=_comparison_payload(), feature_manifest_payload=_feature_payload())
    causes = retrain._failed_gate_root_causes(table)
    assert "FEATURE_SCHEMA_MISMATCH" in causes["live_schema_gate"]


def test_failed_cost_stress_creates_cost_sensitive_edge_root_cause() -> None:
    table = retrain._failure_gate_table(paper_payload=_paper_payload(), comparison_payload=_comparison_payload(), feature_manifest_payload=_feature_payload())
    causes = retrain._failed_gate_root_causes(table)
    assert "COST_SENSITIVE_EDGE" in causes["cost_stress_gate"]


def test_failed_calibration_creates_poor_calibration_root_cause() -> None:
    table = retrain._failure_gate_table(paper_payload=_paper_payload(), comparison_payload=_comparison_payload(), feature_manifest_payload=_feature_payload())
    causes = retrain._failed_gate_root_causes(table)
    assert "POOR_CALIBRATION" in causes["calibration_gate"]


def test_failed_probability_monotonicity_creates_non_monotonic_root_cause() -> None:
    table = retrain._failure_gate_table(paper_payload=_paper_payload(), comparison_payload=_comparison_payload(), feature_manifest_payload=_feature_payload())
    causes = retrain._failed_gate_root_causes(table)
    assert "NON_MONOTONIC_PROBABILITIES" in causes["probability_monotonicity_gate"]


def test_critical_failures_emit_do_not_proceed_messages() -> None:
    table = retrain._failure_gate_table(paper_payload=_paper_payload(), comparison_payload=_comparison_payload(), feature_manifest_payload=_feature_payload())
    flags = retrain._failure_flags_from_gate_rows(table)
    assert "DO_NOT_PROCEED_TO_LIVE_OR_PRODUCTION" in flags["messages"]
    assert "DO_NOT_PROCEED_TO_PAPER_TRADING" in flags["messages"]


def test_failure_diagnosis_mode_keeps_production_blocked() -> None:
    payload = retrain._build_failure_diagnosis_payload(
        dataset_path="data/processed/test.csv",
        comparison_payload=_comparison_payload(),
        paper_payload=_paper_payload(),
        feature_manifest_payload=_feature_payload(),
        output_dir="models/failure_diagnosis_test",
    )
    assert payload["production_adoption_allowed"] is False

