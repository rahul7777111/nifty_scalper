from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain

SCRIPT_PATH = REPO_ROOT / "scripts" / "retrain_all_edge_models.py"


def _dataset() -> pd.DataFrame:
    rows = []
    base_ts = pd.Timestamp("2026-06-01 09:15:00+05:30")
    for idx in range(240):
        side_is_ce = idx % 2 == 0
        volatile = 1 if idx % 5 in (0, 1) else 0
        if idx % 6 in (0, 1):
            moneyness = 1.04
        elif idx % 6 in (2, 3):
            moneyness = 1.0
        else:
            moneyness = 0.95
        is_positive = idx % 4 in (0, 1)
        rows.append(
            {
                "timestamp": base_ts + pd.Timedelta(minutes=idx),
                "feature_a": float(idx % 7),
                "feature_b": float((idx * 3) % 11),
                "option_volume": float(100 + idx),
                "option_type_ce": 1.0 if side_is_ce else 0.0,
                "option_type_pe": 0.0 if side_is_ce else 1.0,
                "moneyness": moneyness,
                "distance_from_atm": abs(moneyness - 1.0) * 100.0,
                "regime_volatile": float(volatile),
                "volatility_regime_classifier": float(volatile),
                "ltp": float(100 + (idx % 17)),
                "bid_ask_spread_pct": 0.02,
                "cost_adjusted_success_5m": int(is_positive),
                "profitable_trade_label": int(is_positive),
                "avoid_trade_label": int(not is_positive),
                "net_forward_return": 0.01 if is_positive else -0.01,
                "future_profit": 1.0,
                "spot_return_1": 0.5,
            }
        )
    return pd.DataFrame(rows)


def test_retrain_all_models_mode_writes_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_root = tmp_path / "artifacts"
    _dataset().to_csv(dataset_path, index=False)
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--dataset",
        str(dataset_path),
        "--retrain-all-models",
        "--live-computable-only",
        "--no-production-adopt",
        "--min-rows",
        "100",
        "--output-dir",
        str(output_root),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=True)
    assert result.returncode == 0
    subdirs = [path for path in output_root.iterdir() if path.is_dir()]
    assert subdirs, "expected timestamped retraining artifact directory"
    artifact_dir = subdirs[0]
    metrics_report = artifact_dir / "metrics_report.json"
    feature_manifest = artifact_dir / "feature_manifest.json"
    training_manifest = artifact_dir / "training_manifest.json"
    threshold_sweep_json = artifact_dir / "threshold_sweep.json"
    threshold_sweep_csv = artifact_dir / "threshold_sweep.csv"
    champion_selection = artifact_dir / "champion_selection_report.json"
    cost_stress_report = artifact_dir / "cost_stress_report.json"
    regime_candidates_json = artifact_dir / "regime_restricted_candidates.json"
    regime_candidates_csv = artifact_dir / "regime_restricted_candidates.csv"
    ranker_candidates_json = artifact_dir / "ranker_candidate_report.json"
    ranker_candidates_csv = artifact_dir / "ranker_candidate_report.csv"
    ce_pe_report = artifact_dir / "ce_pe_champion_report.json"
    small_sample_report = artifact_dir / "small_sample_warning_report.json"
    monthly_stability_report = artifact_dir / "monthly_stability_report.json"
    candidate_cost_stress_report = artifact_dir / "cost_stress_by_candidate.json"
    candidate_cost_stress_alias = artifact_dir / "candidate_cost_stress_report.json"
    fold_stability_report = artifact_dir / "fold_stability_report.json"
    candidate_champion_report = artifact_dir / "candidate_champion_report.json"
    production_blocker_report = artifact_dir / "production_blocker_report.json"
    candidate_summary_report = artifact_dir / "candidate_summary_report.md"
    execution_rescue_report = artifact_dir / "execution_rescue_report.json"
    paper_watchlist_report = artifact_dir / "paper_watchlist_report.json"
    cost_model_audit_report = artifact_dir / "cost_model_audit_report.json"
    horizon_report = artifact_dir / "horizon_comparison_report.json"
    deep_data_audit_report = artifact_dir / "deep_data_audit_report.json"
    final_decision_report = artifact_dir / "final_ml_trading_decision_report.json"
    assert metrics_report.exists()
    assert feature_manifest.exists()
    assert training_manifest.exists()
    assert threshold_sweep_json.exists()
    assert threshold_sweep_csv.exists()
    assert champion_selection.exists()
    assert cost_stress_report.exists()
    assert regime_candidates_json.exists()
    assert regime_candidates_csv.exists()
    assert ranker_candidates_json.exists()
    assert ranker_candidates_csv.exists()
    assert ce_pe_report.exists()
    assert small_sample_report.exists()
    assert monthly_stability_report.exists()
    assert candidate_cost_stress_report.exists()
    assert candidate_cost_stress_alias.exists()
    assert fold_stability_report.exists()
    assert candidate_champion_report.exists()
    assert production_blocker_report.exists()
    assert candidate_summary_report.exists()
    assert execution_rescue_report.exists()
    assert paper_watchlist_report.exists()
    assert cost_model_audit_report.exists()
    assert horizon_report.exists()
    assert deep_data_audit_report.exists()
    assert final_decision_report.exists()
    metrics = json.loads(metrics_report.read_text(encoding="utf-8"))
    manifest = json.loads(feature_manifest.read_text(encoding="utf-8"))
    training = json.loads(training_manifest.read_text(encoding="utf-8"))
    champion = json.loads(champion_selection.read_text(encoding="utf-8"))
    regime_candidates = json.loads(regime_candidates_json.read_text(encoding="utf-8"))
    ranker_candidates = json.loads(ranker_candidates_json.read_text(encoding="utf-8"))
    ce_pe = json.loads(ce_pe_report.read_text(encoding="utf-8"))
    monthly = json.loads(monthly_stability_report.read_text(encoding="utf-8"))
    candidate_cost = json.loads(candidate_cost_stress_report.read_text(encoding="utf-8"))
    fold_stability = json.loads(fold_stability_report.read_text(encoding="utf-8"))
    candidate_champion = json.loads(candidate_champion_report.read_text(encoding="utf-8"))
    execution_rescue = json.loads(execution_rescue_report.read_text(encoding="utf-8"))
    paper_watchlist = json.loads(paper_watchlist_report.read_text(encoding="utf-8"))
    cost_audit = json.loads(cost_model_audit_report.read_text(encoding="utf-8"))
    horizon = json.loads(horizon_report.read_text(encoding="utf-8"))
    deep_audit = json.loads(deep_data_audit_report.read_text(encoding="utf-8"))
    final_decision = json.loads(final_decision_report.read_text(encoding="utf-8"))
    assert metrics["production_adoption_allowed"] is False
    assert training["production_adoption_allowed"] is False
    assert champion["production_adoption_allowed"] is False
    assert champion["champion_rejected"] is True
    assert "future_profit" in manifest["dropped_leakage_columns"]
    assert "net_forward_return" in manifest["evaluation_return_columns"]
    assert "net_forward_return" not in manifest["features"]
    assert "spot_return_1" in manifest["dropped_leakage_columns"]
    assert metrics["models_trained"][0]["selected_threshold"] is not None
    candidate_names = {row["candidate_name"] for row in regime_candidates["candidates"] if row.get("status") == "OK"}
    ranker_names = {row["candidate_name"] for row in ranker_candidates["candidates"] if row.get("status") == "OK"}
    assert "logistic_regression_top_5_per_day" in ranker_names
    assert "random_forest_ce_only" in candidate_names
    assert "random_forest_pe_only" in candidate_names
    assert ce_pe["production_adoption_allowed"] is False
    assert candidate_cost["production_adoption_allowed"] is False
    assert monthly["production_adoption_allowed"] is False
    assert fold_stability["production_adoption_allowed"] is False
    assert candidate_champion["production_adoption_allowed"] is False
    assert paper_watchlist["production_adoption_allowed"] is False
    assert cost_audit["evaluation_return_column"] == "net_forward_return"
    assert "double_counting_warning" in cost_audit
    assert "inferred_embedded_cost" in cost_audit
    assert horizon["production_adoption_allowed"] is False
    assert deep_audit["production_adoption_allowed"] is False
    assert final_decision["production_adoption_allowed"] is False
    assert "cost_1_25x" in candidate_cost["candidates"]["random_forest_ce_only"]
    assert "cost_1_10x" in candidate_cost["candidates"]["random_forest_ce_only"]
    rescue_bases = {row["base_candidate_name"] for row in execution_rescue["rows"]}
    assert rescue_bases <= {"xgboost_pe_itm_atm", "xgboost_pe_volatile"}
    assert execution_rescue["rows"][0]["reject_reasons"] is not None
    assert execution_rescue["rows"][0]["estimated_cost_per_trade"] is not None


def test_train_only_writes_status_files_and_skips_final_reports(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_root = tmp_path / "artifacts"
    _dataset().to_csv(dataset_path, index=False)
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--dataset",
        str(dataset_path),
        "--retrain-all-models",
        "--train-only",
        "--only-target",
        "profitable_trade_label",
        "--only-model",
        "logistic_regression",
        "--min-rows",
        "100",
        "--output-dir",
        str(output_root),
    ]
    subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=True)
    artifact_dir = [path for path in output_root.iterdir() if path.is_dir()][0]
    status_files = list((artifact_dir / "status").glob("*.json"))
    assert status_files
    payload = json.loads(status_files[0].read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["artifact_path"]
    assert payload["metrics_path"]
    assert not (artifact_dir / "final_ml_trading_decision_report.json").exists()


def test_resume_report_only_regenerates_top_level_reports(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_root = tmp_path / "artifacts"
    _dataset().to_csv(dataset_path, index=False)
    train_cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--dataset",
        str(dataset_path),
        "--retrain-all-models",
        "--train-only",
        "--only-target",
        "profitable_trade_label",
        "--only-model",
        "logistic_regression",
        "--min-rows",
        "100",
        "--output-dir",
        str(output_root),
    ]
    subprocess.run(train_cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=True)
    artifact_dir = [path for path in output_root.iterdir() if path.is_dir()][0]
    report_cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--dataset",
        str(dataset_path),
        "--retrain-all-models",
        "--report-only",
        "--resume",
        "--resume-artifact-dir",
        str(artifact_dir),
        "--only-target",
        "profitable_trade_label",
        "--only-model",
        "logistic_regression",
        "--min-rows",
        "100",
        "--output-dir",
        str(output_root),
    ]
    subprocess.run(report_cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=True)
    report_dir = REPO_ROOT / "reports"
    assert list(report_dir.glob("all_models_retraining_report_*.json"))
    assert list(report_dir.glob("paper_readiness_report_*.json"))


def test_only_target_and_only_model_limit_status_pairs(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_root = tmp_path / "artifacts"
    _dataset().to_csv(dataset_path, index=False)
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--dataset",
        str(dataset_path),
        "--retrain-all-models",
        "--train-only",
        "--only-target",
        "avoid_trade_label",
        "--only-model",
        "logistic_regression,random_forest",
        "--min-rows",
        "100",
        "--output-dir",
        str(output_root),
    ]
    subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=True)
    artifact_dir = [path for path in output_root.iterdir() if path.is_dir()][0]
    status_files = sorted((artifact_dir / "status").glob("*.json"))
    names = [p.stem for p in status_files]
    assert any("avoid_trade_label__logistic_regression" in name for name in names)
    assert any("avoid_trade_label__random_forest" in name for name in names)
    assert all("profitable_trade_label" not in name for name in names)


def test_resume_skips_completed_artifact_without_force(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_root = tmp_path / "artifacts"
    _dataset().to_csv(dataset_path, index=False)
    first_cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--dataset",
        str(dataset_path),
        "--retrain-all-models",
        "--train-only",
        "--only-target",
        "profitable_trade_label",
        "--only-model",
        "logistic_regression",
        "--min-rows",
        "100",
        "--output-dir",
        str(output_root),
    ]
    subprocess.run(first_cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=True)
    artifact_dir = [path for path in output_root.iterdir() if path.is_dir()][0]
    model_path = next(artifact_dir.glob("*logistic_regression_profitable_trade_label.pkl"))
    before = model_path.stat().st_mtime
    second_cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--dataset",
        str(dataset_path),
        "--retrain-all-models",
        "--train-only",
        "--resume",
        "--resume-artifact-dir",
        str(artifact_dir),
        "--only-target",
        "profitable_trade_label",
        "--only-model",
        "logistic_regression",
        "--min-rows",
        "100",
        "--output-dir",
        str(output_root),
    ]
    subprocess.run(second_cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=True)
    after = model_path.stat().st_mtime
    assert after == before


def test_resume_completed_random_forest_checkpoint_writes_top_level_reports(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_root = tmp_path / "artifacts"
    _dataset().to_csv(dataset_path, index=False)
    train_cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--dataset",
        str(dataset_path),
        "--retrain-all-models",
        "--train-only",
        "--only-target",
        "profitable_trade_label",
        "--only-model",
        "random_forest",
        "--min-rows",
        "100",
        "--output-dir",
        str(output_root),
    ]
    subprocess.run(train_cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=True)
    artifact_dir = [path for path in output_root.iterdir() if path.is_dir()][0]
    resume_cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--dataset",
        str(dataset_path),
        "--retrain-all-models",
        "--resume",
        "--resume-artifact-dir",
        str(artifact_dir),
        "--only-target",
        "profitable_trade_label",
        "--only-model",
        "random_forest",
        "--min-rows",
        "100",
        "--output-dir",
        str(output_root),
    ]
    subprocess.run(resume_cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=True)
    assert (artifact_dir / "metrics_report.json").exists()
    assert (artifact_dir / "training_manifest.json").exists()
    assert (artifact_dir / "core_retrain_summary.json").exists()
    assert (artifact_dir / "retrain_all_models_summary.json").exists()
    assert (artifact_dir / "final_ml_trading_decision_report.json").exists()


def _edge_filter_frame() -> pd.DataFrame:
    rows = []
    base_ts = pd.Timestamp("2026-06-01 09:15:00+05:30")
    for day in range(4):
        for rank in range(5):
            ts = base_ts + pd.Timedelta(days=day, minutes=rank * 15)
            rows.append(
                {
                    "timestamp": ts,
                    "trade_day": ts.date(),
                    "selected_return": 0.4 - (rank * 0.1) if rank < 3 else -0.2,
                    "prob_edge": 0.90 - (rank * 0.1),
                    "option_type_ce": 1.0 if rank % 2 == 0 else 0.0,
                    "option_type_pe": 0.0 if rank % 2 == 0 else 1.0,
                    "moneyness": [1.0, 1.02, 1.05, 0.96, 0.90][rank],
                    "bid_ask_spread_pct": [0.4, 0.8, 1.2, 2.0, 0.3][rank],
                    "volume": [500, 400, 300, 200, 100][rank],
                    "oi": [900, 800, 700, 600, 500][rank],
                    "ltp": [120, 90, 70, 30, 10][rank],
                    "dte_days": [2, 1, 0, 3, 5][rank],
                    "regime_volatile": [1, 0, 1, 0, 1][rank],
                    "regime_trending": [1, 1, 0, 0, 1][rank],
                    "strike_distance_pct": [0.01, 0.02, 0.04, 0.10, 0.12][rank],
                    "ctx_trend_strength": [1.0, 0.8, -0.2, -0.5, 0.6][rank],
                }
            )
    return pd.DataFrame(rows)


def test_top_n_per_day_selection() -> None:
    frame = _edge_filter_frame()
    selected = retrain._ranked_selection_rows(frame, "prob_edge", top_n=3)
    assert len(selected) == 12
    assert selected.groupby("trade_day").size().eq(3).all()


def test_top_percent_per_day_selection() -> None:
    frame = _edge_filter_frame()
    selected = retrain._ranked_selection_rows(frame, "prob_edge", top_pct=0.40)
    assert len(selected) == 8
    assert pd.to_datetime(selected["timestamp"]).dt.date.value_counts().eq(2).all()


def test_probability_percentile_selection() -> None:
    frame = _edge_filter_frame()
    selected, selection_name, threshold_value = retrain._apply_edge_refinement_selection(
        frame,
        "prob_edge",
        {"kind": "probability_percentile", "value": 90.0},
        fallback_threshold=0.5,
    )
    assert selection_name == "probability_percentile>=90"
    assert threshold_value is not None
    assert 1 <= len(selected) <= len(frame)


def test_target_trade_count_selection() -> None:
    frame = _edge_filter_frame()
    selected, selection_name, threshold_value = retrain._apply_edge_refinement_selection(
        frame,
        "prob_edge",
        {"kind": "target_trade_count", "value": 3},
        fallback_threshold=0.5,
    )
    assert selection_name == "target_3_trades"
    assert threshold_value is not None
    assert len(selected) == 3


def test_spread_filter_skip_when_column_missing() -> None:
    frame = _edge_filter_frame().drop(columns=["bid_ask_spread_pct"])
    mask, skipped = retrain._edge_refinement_filter_mask(frame, {"max_spread_pct": 1.0})
    assert mask.all()
    assert "max_spread_pct" in skipped


def test_spread_filter_applied_when_column_exists() -> None:
    frame = _edge_filter_frame()
    mask, skipped = retrain._edge_refinement_filter_mask(frame, {"max_spread_pct": 1.0})
    assert "max_spread_pct" not in skipped
    assert int(mask.sum()) < len(frame)


def test_ce_pe_and_moneyness_filters() -> None:
    frame = _edge_filter_frame()
    ce_mask, _ = retrain._edge_refinement_filter_mask(frame, {"option_side": "CE"})
    pe_mask, _ = retrain._edge_refinement_filter_mask(frame, {"option_side": "PE"})
    atm_mask, _ = retrain._edge_refinement_filter_mask(frame, {"moneyness_include": ["ATM", "NEAR_ATM"]})
    assert ce_mask.sum() > 0
    assert pe_mask.sum() > 0
    assert atm_mask.sum() > 0
    assert frame.loc[ce_mask, "option_type_ce"].eq(1.0).all()
    assert frame.loc[pe_mask, "option_type_pe"].eq(1.0).all()


def test_edge_refinement_cost_metrics() -> None:
    frame = _edge_filter_frame().copy()
    frame["selected_return"] = pd.Series(([0.8, 0.7, -0.1, 0.5, 0.4] * 4)[: len(frame)], index=frame.index)
    metrics = retrain._edge_refinement_cost_metrics(frame)
    assert metrics["estimated_roundtrip_cost"] >= 0.0
    assert np.isfinite(metrics["return_to_cost_ratio"])
    assert "percentage_of_trades_with_return_greater_than_1_5x_cost" in metrics


def test_cost_aware_score_prefers_cost_stable_candidate() -> None:
    unstable = {
        "cost_stress_1_25x_pf": 0.5,
        "cost_stress_1_50x_pf": 0.1,
        "walk_forward_worst_fold_pf": 1.5,
        "walk_forward_median_pf": 1.6,
        "return_to_cost_ratio": 0.8,
        "sharpe": 5.0,
        "profit_factor": 8.0,
        "number_of_profitable_days": 10,
        "number_of_profitable_months": 2,
        "best_day_profit_share_of_total_profit": 0.40,
        "top_2_days_profit_share": 0.60,
        "best_fold_profit_share_of_total_profit": 0.55,
        "max_drawdown": 50.0,
        "total_return": 20.0,
    }
    stable = {
        "cost_stress_1_25x_pf": 1.2,
        "cost_stress_1_50x_pf": 1.05,
        "walk_forward_worst_fold_pf": 1.1,
        "walk_forward_median_pf": 1.2,
        "return_to_cost_ratio": 1.5,
        "sharpe": 1.5,
        "profit_factor": 1.4,
        "number_of_profitable_days": 50,
        "number_of_profitable_months": 4,
        "best_day_profit_share_of_total_profit": 0.10,
        "top_2_days_profit_share": 0.20,
        "best_fold_profit_share_of_total_profit": 0.20,
        "max_drawdown": 10.0,
        "total_return": 30.0,
    }
    assert retrain._edge_refinement_cost_aware_score(stable) > retrain._edge_refinement_cost_aware_score(unstable)


def test_return_cost_provenance_detects_net_incremental_model() -> None:
    frame = pd.DataFrame(
        {
            "gross_forward_return": [1.0, 0.8, 0.5],
            "net_forward_return": [0.9, 0.7, 0.4],
        }
    )
    provenance = retrain._return_cost_provenance_for_frame(frame, evaluation_return_column="net_forward_return")
    assert provenance["is_evaluation_return_already_net"] is True
    assert provenance["double_counting_prevented"] is True
    assert round(provenance["cost_1_25x_incremental_deduction_used"], 6) == 0.025


def test_uncertain_cost_provenance_blocks_adoption() -> None:
    payload = {
        "trade_count": 180,
        "profit_factor": 1.30,
        "sharpe": 1.10,
        "cost_stress_1_25x_pf": 1.10,
        "cost_stress_1_50x_pf": 1.02,
        "walk_forward_worst_fold_pf": 1.05,
        "walk_forward_median_pf": 1.10,
        "expected_return_after_cost": 0.05,
        "median_return_after_cost": 0.02,
        "return_to_cost_ratio": 1.40,
        "best_day_profit_share_of_total_profit": 0.10,
        "top_2_days_profit_share": 0.20,
        "best_month_profit_share_of_total_profit": 0.30,
        "best_fold_profit_share_of_total_profit": 0.30,
        "number_of_profitable_days": 35,
        "number_of_profitable_months": 3,
        "number_of_profitable_folds": 4,
        "cost_provenance_uncertain": True,
        "tiny_sample_warning": False,
        "leakage_warning": False,
        "daily_stability_passes": True,
        "threshold_stability_passes": True,
    }
    gate = retrain._edge_refinement_gate(payload, full_run_completed=True)
    assert gate["status"] == "BLOCKED_COST_STRESS"
    assert "cost_provenance_uncertain" in gate["all_failed_gates"]


def test_cost_stress_gate_failure() -> None:
    payload = {
        "trade_count": 400,
        "profit_factor": 1.30,
        "sharpe": 1.10,
        "cost_stress_1_25x_pf": 0.95,
        "cost_stress_1_50x_pf": 0.90,
        "walk_forward_worst_fold_pf": 1.00,
        "walk_forward_median_pf": 1.20,
        "expected_return_after_cost": -0.01,
        "median_return_after_cost": -0.01,
        "return_to_cost_ratio": 0.90,
        "best_day_profit_share_of_total_profit": 0.10,
        "top_2_days_profit_share": 0.20,
        "best_month_profit_share_of_total_profit": 0.30,
        "best_fold_profit_share_of_total_profit": 0.30,
        "number_of_profitable_days": 40,
        "number_of_profitable_months": 3,
        "number_of_profitable_folds": 4,
        "tiny_sample_warning": False,
        "leakage_warning": False,
        "daily_stability_passes": True,
        "threshold_stability_passes": True,
    }
    gate = retrain._edge_refinement_gate(payload, full_run_completed=True)
    assert gate["status"] == "BLOCKED_COST_STRESS"
    assert "cost_1_25x_pf_below_watchlist_gate" in gate["failed_gates"]


def test_tiny_sample_rejection() -> None:
    payload = {
        "trade_count": 40,
        "profit_factor": 1.50,
        "sharpe": 1.20,
        "cost_stress_1_25x_pf": 1.20,
        "cost_stress_1_50x_pf": 1.10,
        "walk_forward_worst_fold_pf": 1.10,
        "walk_forward_median_pf": 1.20,
        "expected_return_after_cost": 0.10,
        "median_return_after_cost": 0.08,
        "return_to_cost_ratio": 1.80,
        "best_day_profit_share_of_total_profit": 0.10,
        "top_2_days_profit_share": 0.20,
        "best_month_profit_share_of_total_profit": 0.30,
        "best_fold_profit_share_of_total_profit": 0.30,
        "number_of_profitable_days": 40,
        "number_of_profitable_months": 3,
        "number_of_profitable_folds": 4,
        "tiny_sample_warning": True,
        "leakage_warning": False,
        "daily_stability_passes": True,
        "threshold_stability_passes": True,
    }
    gate = retrain._edge_refinement_gate(payload, full_run_completed=True)
    assert gate["status"] == "BLOCKED_SMALL_SAMPLE"
    assert "tiny_sample_warning_present" in gate["failed_gates"]


def test_watchlist_only_classification() -> None:
    payload = {
        "trade_count": 180,
        "profit_factor": 1.22,
        "sharpe": 1.05,
        "cost_stress_1_25x_pf": 1.06,
        "cost_stress_1_50x_pf": 1.01,
        "walk_forward_worst_fold_pf": 0.95,
        "walk_forward_median_pf": 1.06,
        "expected_return_after_cost": 0.10,
        "median_return_after_cost": 0.05,
        "return_to_cost_ratio": 1.30,
        "best_day_profit_share_of_total_profit": 0.20,
        "top_2_days_profit_share": 0.35,
        "best_month_profit_share_of_total_profit": 0.40,
        "best_fold_profit_share_of_total_profit": 0.45,
        "number_of_profitable_days": 35,
        "number_of_profitable_months": 3,
        "number_of_profitable_folds": 4,
        "tiny_sample_warning": False,
        "leakage_warning": False,
        "daily_stability_passes": True,
        "threshold_stability_passes": True,
    }
    gate = retrain._edge_refinement_gate(payload, full_run_completed=True)
    assert gate["status"] == "PAPER_WATCHLIST"
    assert "trade_count_below_paper_ready_gate" in gate["paper_ready_failed_gates"]


def test_trade_count_above_band_is_rejected() -> None:
    payload = {
        "trade_count": 6001,
        "profit_factor": 1.50,
        "sharpe": 1.20,
        "cost_stress_1_25x_pf": 1.20,
        "cost_stress_1_50x_pf": 1.05,
        "walk_forward_worst_fold_pf": 1.10,
        "walk_forward_median_pf": 1.20,
        "expected_return_after_cost": 0.20,
        "median_return_after_cost": 0.10,
        "return_to_cost_ratio": 1.50,
        "best_day_profit_share_of_total_profit": 0.20,
        "top_2_days_profit_share": 0.35,
        "best_month_profit_share_of_total_profit": 0.40,
        "best_fold_profit_share_of_total_profit": 0.45,
        "number_of_profitable_days": 60,
        "number_of_profitable_months": 4,
        "number_of_profitable_folds": 4,
        "tiny_sample_warning": False,
        "leakage_warning": False,
        "daily_stability_passes": True,
        "threshold_stability_passes": True,
    }
    gate = retrain._edge_refinement_gate(payload, full_run_completed=True)
    assert gate["status"] == "BLOCKED_TOO_MANY_TRADES"
    assert "trade_count_above_watchlist_gate" in gate["failed_gates"]


def test_paper_ready_classification() -> None:
    payload = {
        "trade_count": 300,
        "profit_factor": 1.30,
        "sharpe": 1.30,
        "cost_stress_1_25x_pf": 1.15,
        "cost_stress_1_50x_pf": 1.02,
        "walk_forward_worst_fold_pf": 1.05,
        "walk_forward_median_pf": 1.20,
        "expected_return_after_cost": 0.15,
        "median_return_after_cost": 0.10,
        "return_to_cost_ratio": 1.60,
        "best_day_profit_share_of_total_profit": 0.20,
        "top_2_days_profit_share": 0.30,
        "best_month_profit_share_of_total_profit": 0.40,
        "best_fold_profit_share_of_total_profit": 0.45,
        "number_of_profitable_days": 45,
        "number_of_profitable_months": 4,
        "number_of_profitable_folds": 4,
        "tiny_sample_warning": False,
        "leakage_warning": False,
        "daily_stability_passes": True,
        "threshold_stability_passes": True,
    }
    gate = retrain._edge_refinement_gate(payload, full_run_completed=True)
    assert gate["status"] == "PAPER_READY"


def test_blocked_when_no_candidate_passes() -> None:
    payload = {
        "trade_count": 150,
        "profit_factor": 1.05,
        "sharpe": 0.60,
        "cost_stress_1_25x_pf": 0.90,
        "cost_stress_1_50x_pf": 0.70,
        "walk_forward_worst_fold_pf": 0.80,
        "walk_forward_median_pf": 0.90,
        "expected_return_after_cost": -0.01,
        "median_return_after_cost": -0.02,
        "return_to_cost_ratio": 0.75,
        "best_day_profit_share_of_total_profit": 0.40,
        "top_2_days_profit_share": 0.60,
        "best_month_profit_share_of_total_profit": 0.60,
        "best_fold_profit_share_of_total_profit": 0.60,
        "number_of_profitable_days": 10,
        "number_of_profitable_months": 1,
        "number_of_profitable_folds": 1,
        "tiny_sample_warning": False,
        "leakage_warning": False,
        "daily_stability_passes": False,
        "threshold_stability_passes": False,
    }
    gate = retrain._edge_refinement_gate(payload, full_run_completed=True)
    assert gate["status"] == "BLOCKED_CONCENTRATED_PROFIT"


def test_concentrated_profit_candidate_gets_concentration_status() -> None:
    payload = {
        "trade_count": 180,
        "profit_factor": 3.0,
        "sharpe": 2.0,
        "cost_stress_1_25x_pf": 1.2,
        "cost_stress_1_50x_pf": 1.05,
        "walk_forward_worst_fold_pf": 1.1,
        "walk_forward_median_pf": 1.3,
        "expected_return_after_cost": 0.20,
        "median_return_after_cost": 0.10,
        "return_to_cost_ratio": 1.80,
        "best_day_profit_share_of_total_profit": 0.60,
        "top_2_days_profit_share": 0.75,
        "best_month_profit_share_of_total_profit": 0.70,
        "best_fold_profit_share_of_total_profit": 0.65,
        "number_of_profitable_days": 12,
        "number_of_profitable_months": 1,
        "number_of_profitable_folds": 2,
        "tiny_sample_warning": False,
        "leakage_warning": False,
        "daily_stability_passes": True,
        "threshold_stability_passes": True,
    }
    gate = retrain._edge_refinement_gate(payload, full_run_completed=True)
    assert gate["status"] == "BLOCKED_CONCENTRATED_PROFIT"
    assert gate["primary_block_reason"] == "top_1_day_profit_share_above_35pct"


def test_fast_mode_restricts_candidate_count() -> None:
    completed = [
        {
            "label_name": "profitable_trade_label",
            "model_name": "random_forest",
            "artifact_path": "a.pkl",
            "metrics_path": "a.json",
        },
        {
            "label_name": "profitable_trade_label",
            "model_name": "logistic_regression",
            "artifact_path": "b.pkl",
            "metrics_path": "b.json",
        },
    ]
    plans = retrain._edge_refinement_planned_candidates(
        completed,
        fast=True,
        middle_zone_only=False,
        include_avoid_label_refinement=False,
        only_models=(),
        only_targets=(),
        max_models=0,
    )
    assert plans
    assert all(row["model_name"] == "random_forest" for row in plans)
    names = {row["filter_name"] for row in plans}
    assert "top_1_per_day" in names
    assert "threshold_gte_0_7" in names
    assert "high_volatility_only" not in names


def test_middle_zone_only_restricts_candidate_rules() -> None:
    completed = [
        {
            "label_name": "profitable_trade_label",
            "model_name": "random_forest",
            "artifact_path": "a.pkl",
            "metrics_path": "a.json",
        },
    ]
    plans = retrain._edge_refinement_planned_candidates(
        completed,
        fast=False,
        middle_zone_only=True,
        include_avoid_label_refinement=False,
        only_models=(),
        only_targets=(),
        max_models=0,
    )
    names = {row["filter_name"] for row in plans}
    assert "target_500_trades" in names
    assert "probability_percentile >= 99.5" in names
    assert "high_volatility_only" not in names


def test_completed_refinement_candidate_is_skipped_on_resume(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    candidate_id = "profitable_trade_label__random_forest__foo"
    retrain._write_edge_refinement_status(
        artifact_dir,
        candidate_id,
        {
            "candidate_id": candidate_id,
            "candidate_name": "foo",
            "model_name": "random_forest",
            "label_name": "profitable_trade_label",
            "filter_name": "top_1_per_day",
            "selection_rule": "top_n",
            "status": "completed",
            "metrics": {"status": "BLOCKED", "trade_count": 120},
            "skip_reason": None,
            "error": None,
            "start_time": "2026-06-06T12:00:00",
            "end_time": "2026-06-06T12:00:01",
        },
    )
    rows = retrain._load_edge_refinement_status_rows(artifact_dir)
    assert rows[0]["status"] == "completed"


def test_failed_refinement_candidate_is_recorded(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    candidate_id = "profitable_trade_label__random_forest__bar"
    retrain._write_edge_refinement_status(
        artifact_dir,
        candidate_id,
        {
            "candidate_id": candidate_id,
            "candidate_name": "bar",
            "model_name": "random_forest",
            "label_name": "profitable_trade_label",
            "filter_name": "top_3_per_day",
            "selection_rule": "top_n",
            "status": "failed",
            "metrics": {},
            "skip_reason": None,
            "error": "boom",
            "start_time": "2026-06-06T12:00:00",
            "end_time": "2026-06-06T12:00:01",
        },
    )
    rows = retrain._load_edge_refinement_status_rows(artifact_dir)
    assert rows[0]["error"] == "boom"


def test_report_only_aggregates_checkpoints_and_partial_report_is_valid(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    dataset_path = tmp_path / "dataset.csv"
    _dataset().to_csv(dataset_path, index=False)
    planned = [
        {"candidate_id": "c1", "candidate_name": "c1"},
        {"candidate_id": "c2", "candidate_name": "c2"},
        {"candidate_id": "c3", "candidate_name": "c3"},
    ]
    retrain._write_edge_refinement_status(
        artifact_dir,
        "c1",
        {
            "candidate_id": "c1",
            "candidate_name": "c1",
            "model_name": "random_forest",
            "label_name": "profitable_trade_label",
            "filter_name": "top_1_per_day",
            "selection_rule": "top_n",
            "status": "completed",
            "metrics": {
                "candidate_name": "c1",
                "status": "BLOCKED_COST_STRESS",
                "profit_factor": 1.3,
                "sharpe": 0.9,
                "trade_count": 150,
                "cost_stress_1_25x_pf": 0.9,
                "walk_forward_worst_fold_pf": 0.8,
                "walk_forward_median_pf": 1.0,
                "fold_pf_std": 0.1,
                "max_drawdown": 1.0,
                "best_day_profit_share_of_total_profit": 0.2,
                "top_2_days_profit_share": 0.3,
                "best_month_profit_share_of_total_profit": 0.4,
                "best_fold_profit_share_of_total_profit": 0.45,
                "number_of_profitable_days": 35,
                "number_of_profitable_months": 3,
                "number_of_profitable_folds": 4,
                "stability_score": 1.0,
            },
            "skip_reason": None,
            "error": None,
            "start_time": "2026-06-06T12:00:00",
            "end_time": "2026-06-06T12:00:01",
        },
    )
    retrain._write_edge_refinement_status(
        artifact_dir,
        "c2",
        {
            "candidate_id": "c2",
            "candidate_name": "c2",
            "model_name": "random_forest",
            "label_name": "profitable_trade_label",
            "filter_name": "top_3_per_day",
            "selection_rule": "top_n",
            "status": "failed",
            "metrics": {},
            "skip_reason": None,
            "error": "err",
            "start_time": "2026-06-06T12:00:00",
            "end_time": "2026-06-06T12:00:01",
        },
    )
    payload = retrain._edge_refinement_payload_from_status(
        dataset_path=dataset_path,
        artifact_dir=artifact_dir,
        full_run_completed=False,
        base_model={"model_name": "random_forest", "label_name": "profitable_trade_label", "sharpe": 0.7, "cost_stress_1_25x_pf": 0.3, "worst_fold_pf": 0.4, "daily_gate": "FAIL", "threshold_robust": False},
        status_rows=retrain._load_edge_refinement_status_rows(artifact_dir),
        planned_candidates=planned,
        scan_status="PARTIAL_TIMEOUT",
    )
    assert payload["scan_status"] == "PARTIAL_TIMEOUT"
    assert payload["evaluated_candidate_count"] == 1
    assert payload["failed_candidate_count"] == 1
    assert payload["pending_candidate_count"] == 1
    assert payload["paper_trading_status"] == "BLOCKED"


def test_report_markdown_includes_new_sections() -> None:
    payload = {
        "scan_status": "PARTIAL_TIMEOUT",
        "best_base_model": {"model_name": "random_forest", "label_name": "profitable_trade_label"},
        "base_model_failed_reason": "cost_stress_failed",
        "paper_ready": False,
        "evaluated_candidate_count": 1,
        "skipped_candidate_count": 0,
        "pending_candidate_count": 0,
        "candidates": [{
            "candidate_name": "rf_target_500_trades",
            "status": "BLOCKED_COST_STRESS",
            "stability_score": 1.0,
            "profit_factor": 1.3,
            "sharpe": 1.1,
            "cost_stress_1_25x_pf": 0.9,
            "trade_count": 500,
            "all_failed_gates": ["cost_1_25x_pf_below_watchlist_gate"],
        }],
        "middle_zone_candidates": [{
            "candidate_name": "rf_target_500_trades",
            "trade_count": 500,
            "profit_factor": 1.3,
            "sharpe": 1.1,
            "cost_stress_1_25x_pf": 0.9,
            "cost_stress_1_50x_pf": 0.7,
            "return_to_cost_ratio": 1.1,
            "walk_forward_worst_fold_pf": 1.0,
            "status": "BLOCKED_COST_STRESS",
        }],
        "thin_edge_rejected_candidates": [{
            "candidate_name": "rf_target_500_trades",
            "trade_count": 500,
            "average_return_per_trade": 0.05,
            "expected_return_after_cost": -0.01,
            "return_to_cost_ratio": 0.9,
            "all_failed_gates": ["expected_return_after_cost_non_positive"],
        }],
        "do_not_trust_candidates": [{
            "candidate_name": "xgb_threshold_gte_0_7",
            "profit_factor": 10.0,
            "sharpe": 9.0,
            "trade_count": 17,
            "best_day_profit_share_of_total_profit": 0.5,
            "top_2_days_profit_share": 0.8,
            "best_month_profit_share_of_total_profit": 0.5,
            "best_fold_profit_share_of_total_profit": 0.5,
            "number_of_profitable_days": 5,
            "number_of_profitable_months": 2,
            "walk_forward_worst_fold_pf": 0.0,
            "primary_block_reason": "trade_count_below_watchlist_gate",
        }],
        "best_candidate_by_cost_aware_stability_score": {
            "candidate_name": "rf_target_500_trades",
            "cost_aware_stability_score": 2.0,
            "return_to_cost_ratio": 1.1,
            "cost_stress_1_25x_pf": 0.9,
            "cost_stress_1_50x_pf": 0.7,
        },
        "true_edge_diagnosis": {
            "top_20_by_cost_aware_stability_score": [{
                "candidate_name": "rf_target_500_trades",
                "status": "BLOCKED_THIN_EDGE",
                "return_to_cost_ratio": 1.1,
                "median_return_to_cost_ratio": 0.9,
                "number_of_profitable_days": 20,
                "number_of_profitable_months": 2,
                "worst_fold_pf": 0.95,
                "threshold_robustness_support_count": 1,
                "all_failed_gates": ["return_to_cost_ratio_below_watchlist_gate"],
            }]
        },
        "candidate_blocker_breakdown": {
            "only_return_to_cost_ratio_fail_count": 4,
            "only_daily_stability_fail_count": 1,
            "only_threshold_robustness_fail_count": 0,
            "multiple_gate_fail_count": 8,
            "largest_single_blocker_for_promising_candidates": {"gate": "return_to_cost_ratio_below_watchlist_gate", "count": 5},
        },
        "random_forest_sensitivity_table": [{
            "selection_name": "top_1_per_day",
            "available": True,
            "profit_factor": 2.0,
            "sharpe": 3.0,
            "trade_count": 100,
            "cost_stress_1_25x_pf": 1.5,
            "return_to_cost_ratio": 0.9,
            "profitable_days": 25,
            "top_2_days_profit_share": 0.4,
            "worst_fold_pf": 0.95,
            "threshold_robustness_passes": False,
        }],
        "what_would_need_to_improve": {
            "candidate_targets": [{
                "candidate_name": "rf_target_500_trades",
                "required_min_expected_return_per_trade_for_watchlist": 0.12,
                "incremental_expected_return_per_trade_needed": 0.03,
                "required_cost_reduction_fraction_for_watchlist": 0.15,
                "required_additional_profitable_days_for_watchlist": 10,
                "current_threshold_support_count": 1,
                "required_additional_threshold_support_count": 1,
            }]
        },
    }
    markdown = retrain._edge_refinement_report_markdown(payload)
    assert "## Middle-Zone Candidates" in markdown
    assert "## Thin Edge Rejections" in markdown
    assert "## High Headline PF Candidates To Distrust" in markdown
    assert "## Top Candidate Diagnosis" in markdown
    assert "## Blocker Breakdown" in markdown
    assert "## Random Forest Sensitivity" in markdown
    assert "## What Would Need To Improve" in markdown
    assert "## No-Cheating Recommendation" in markdown


def test_true_edge_diagnosis_payload_sections_present() -> None:
    completed_rows = [
        {
            "candidate_name": "rf_top_1",
            "model_name": "random_forest",
            "label_name": "profitable_trade_label",
            "selection_name": "top_1_per_day",
            "status": "BLOCKED_THIN_EDGE",
            "trade_count": 106,
            "profit_factor": 2.6,
            "sharpe": 5.7,
            "stability_score": 5.0,
            "cost_aware_stability_score": 4.0,
            "return_to_cost_ratio": 0.02,
            "median_return_to_cost_ratio": 0.01,
            "average_estimated_cost_pct": 0.08,
            "expected_return_after_cost": 0.10,
            "median_return_after_cost": 0.02,
            "average_positive_return": 0.5,
            "average_negative_return": -0.2,
            "win_rate": 0.55,
            "best_day_profit_share_of_total_profit": 0.2,
            "top_2_days_profit_share": 0.3,
            "best_fold_profit_share_of_total_profit": 0.25,
            "number_of_profitable_days": 20,
            "number_of_profitable_months": 2,
            "number_of_profitable_folds": 3,
            "walk_forward_worst_fold_pf": 0.95,
            "walk_forward_median_pf": 1.1,
            "threshold_stability": {"supporting_thresholds": [0.65], "rows": [{"threshold": 0.65}]},
            "daily_stability": {"losing_days": 12},
            "monthly_stability": {"months": [{"total_return": 1.0}, {"total_return": -0.5}]},
            "watchlist_failed_gates": ["return_to_cost_ratio_below_watchlist_gate", "daily_stability_failed"],
            "all_failed_gates": ["return_to_cost_ratio_below_watchlist_gate", "daily_stability_failed"],
        }
    ]
    payload = retrain._edge_refinement_payload_from_status(
        dataset_path=Path("data/processed/test.csv"),
        artifact_dir=Path("models/test"),
        full_run_completed=False,
        base_model={"model_name": "random_forest", "label_name": "profitable_trade_label"},
        status_rows=[],
        planned_candidates=[],
        scan_status="COMPLETE",
    )
    payload["candidates"] = completed_rows
    payload["true_edge_diagnosis"] = retrain._edge_refinement_true_edge_diagnosis(completed_rows)
    payload["candidate_blocker_breakdown"] = retrain._edge_refinement_blocker_breakdown(completed_rows)
    payload["random_forest_sensitivity_table"] = retrain._edge_refinement_rf_sensitivity_table(completed_rows)
    payload["what_would_need_to_improve"] = retrain._edge_refinement_improvement_targets(completed_rows)
    assert payload["true_edge_diagnosis"]["top_20_by_stability_score"][0]["candidate_name"] == "rf_top_1"
    assert "multiple_gate_fail_count" in payload["candidate_blocker_breakdown"]
    assert payload["random_forest_sensitivity_table"][0]["selection_name"] == "top_1_per_day"
    assert "candidate_targets" in payload["what_would_need_to_improve"]
    assert payload["paper_trading_status"] == "BLOCKED"


def test_edge_widening_labels_use_net_and_embedded_cost() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="D", tz="Asia/Kolkata"),
            "gross_forward_return": [0.02, 0.02, 0.04],
            "net_forward_return": [0.01, 0.015, 0.03],
        }
    )
    augmented, meta = retrain._augment_edge_widening_labels(df, minimum_absolute_return=0.01)
    assert meta["embedded_cost_median"] > 0.0
    assert augmented["profitable_trade_label_edge_1x_cost"].tolist() == [1.0, 1.0, 1.0]
    assert augmented["profitable_trade_label_edge_3x_cost"].tolist() == [0.0, 0.0, 0.0]
    assert augmented["big_move_profitable_label"].tolist() == [0.0, 1.0, 1.0]


def test_edge_widening_sparse_label_detected() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=100, freq="D", tz="Asia/Kolkata"),
            "sparse_label": [1.0] + [0.0] * 99,
        }
    )
    report = retrain._edge_widening_label_distribution_report(df, "sparse_label")
    assert report["too_sparse"] is True


def test_weekly_selector_works() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=14, freq="D", tz="Asia/Kolkata"),
            "probability": np.linspace(0.1, 0.9, 14),
        }
    )
    selected, selection_name, _ = retrain._apply_edge_refinement_selection(
        df,
        "probability",
        {"kind": "top_n_per_week", "value": 1},
        fallback_threshold=0.5,
    )
    assert selection_name == "top_1_per_week"
    assert len(selected) == 3


def test_edge_widening_report_comparison_table_present() -> None:
    rows = [
        {
            "candidate_name": "rf_old",
            "model_name": "random_forest",
            "label_name": "profitable_trade_label",
            "trade_count": 200,
            "profit_factor": 1.5,
            "sharpe": 1.2,
            "cost_stress_1_25x_pf": 1.1,
            "cost_stress_1_50x_pf": 1.0,
            "return_to_cost_ratio": 0.9,
            "walk_forward_worst_fold_pf": 1.0,
            "threshold_stability": {"supporting_thresholds": [0.65]},
            "cost_aware_stability_score": 2.0,
        },
        {
            "candidate_name": "rf_new",
            "model_name": "random_forest",
            "label_name": "profitable_trade_label_edge_1x_cost",
            "trade_count": 120,
            "profit_factor": 1.7,
            "sharpe": 1.3,
            "cost_stress_1_25x_pf": 1.2,
            "cost_stress_1_50x_pf": 1.1,
            "return_to_cost_ratio": 1.0,
            "walk_forward_worst_fold_pf": 1.1,
            "threshold_stability": {"supporting_thresholds": [0.65, 0.7]},
            "cost_aware_stability_score": 2.5,
        },
    ]
    table = retrain._edge_widening_comparison_table(rows)
    assert any(row["label_name"] == "profitable_trade_label_edge_1x_cost" for row in table)


def test_edge_widening_does_not_promote_without_gates() -> None:
    rows = [
        {
            "candidate_name": "rf_new",
            "model_name": "random_forest",
            "label_name": "profitable_trade_label_edge_1x_cost",
            "trade_count": 120,
            "profit_factor": 1.7,
            "sharpe": 1.3,
            "cost_stress_1_25x_pf": 1.2,
            "cost_stress_1_50x_pf": 1.1,
            "return_to_cost_ratio": 0.9,
            "walk_forward_worst_fold_pf": 0.8,
            "threshold_stability": {"supporting_thresholds": []},
            "status": "BLOCKED_THIN_EDGE",
            "cost_aware_stability_score": 2.5,
        },
    ]
    table = retrain._edge_widening_comparison_table(rows)
    assert table[1]["status"] != "PAPER_WATCHLIST"


def test_edge_widening_noisy_label_detected() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=100, freq="D", tz="Asia/Kolkata"),
            "noisy_label": [1.0] * 50 + [0.0] * 50,
        }
    )
    report = retrain._edge_widening_label_distribution_report(df, "noisy_label")
    assert report["too_noisy"] is True
    assert report["suitable_for_training"] is False
    assert report["negative_count"] == 50


def test_edge_widening_report_contains_why_edge_improved_or_failed() -> None:
    comp_rows = [
        {
            "label_name": "profitable_trade_label",
            "candidate_name": "rf_old",
            "status": "BLOCKED_COST_STRESS",
            "trade_count": 200,
            "profit_factor": 1.5,
            "sharpe": 1.2,
            "cost_stress_1_25x_pf": 1.1,
            "cost_stress_1_50x_pf": 1.0,
            "return_to_cost_ratio": 0.9,
            "worst_fold_pf": 1.0,
            "threshold_support_count": 1,
            "profitable_days": 25,
            "losing_days": 25,
            "top_2_days_profit_share": 0.4,
            "failed_gates": [],
        },
        {
            "label_name": "profitable_trade_label_edge_1x_cost",
            "candidate_name": "rf_new",
            "status": "PAPER_WATCHLIST",
            "trade_count": 120,
            "profit_factor": 1.7,
            "sharpe": 1.3,
            "cost_stress_1_25x_pf": 1.2,
            "cost_stress_1_50x_pf": 1.1,
            "return_to_cost_ratio": 1.1,
            "worst_fold_pf": 1.1,
            "threshold_support_count": 2,
            "profitable_days": 30,
            "losing_days": 20,
            "top_2_days_profit_share": 0.35,
            "failed_gates": [],
        },
    ]
    dist_report = [
        {"label_name": "profitable_trade_label", "too_sparse": False},
        {"label_name": "profitable_trade_label_edge_1x_cost", "too_sparse": True},
    ]
    diag = retrain._edge_widening_failure_diagnosis(comp_rows, dist_report)
    assert len(diag) == 2
    assert diag[1]["did_return_to_cost_improve"] is True
    assert diag[1]["did_cost_stress_pf_remain_stable"] is True
    assert diag[1]["did_daily_stability_improve"] is True
    assert diag[1]["did_profit_concentration_reduce"] is True
    assert diag[1]["is_label_too_sparse"] is True
    assert diag[1]["is_label_better_than_baseline"] is True

    payload = {
        "artifact_dir": "models/test",
        "dataset_path": "data/test.csv",
        "paper_watchlist_reached": False,
        "paper_ready_reached": False,
        "label_distribution_report": [
            {"label_name": "profitable_trade_label_edge_1x_cost", "row_count": 100, "positive_count": 10, "negative_count": 90, "positive_rate": 0.1, "too_sparse": True, "too_noisy": False, "suitable_for_training": False}
        ],
        "comparison_table": comp_rows,
        "edge_improved_or_failed": diag,
    }
    markdown = retrain._edge_widening_report_markdown(payload)
    assert "## Why Edge Improved Or Failed" in markdown
    assert "improved_return_to_cost=" in markdown
    assert "cost_stress_pf_stable=" in markdown
    assert "daily_stability_improved=" in markdown
    assert "profit_concentration_reduced=" in markdown
    assert "label_too_sparse=" in markdown
    assert "better_than_baseline=" in markdown


def test_candidate_dict_contains_all_requested_metrics() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01 09:15:00", periods=20, freq="min", tz="Asia/Kolkata"),
            "trade_day": [pd.Timestamp("2026-06-01").date()] * 20,
            "selected_return": [0.01] * 10 + [-0.01] * 10,
            "prob_label_rf": np.linspace(0.4, 0.9, 20),
            "option_volume": [500.0] * 20,
            "oi": [1000.0] * 20,
            "bid_ask_spread_pct": [0.01] * 20,
            "ltp": [100.0] * 20,
            "option_type_ce": [1.0] * 20,
            "option_type_pe": [0.0] * 20,
            "moneyness": [1.0] * 20,
            "_evaluation_return_column_used": ["net_forward_return"] * 20,
        }
    )
    candidate = retrain._evaluate_edge_refinement_candidate(
        df,
        candidate_name="test_candidate",
        model_name="rf",
        label_name="label",
        filters={},
        selection={"kind": "threshold", "value": 0.5},
        fallback_threshold=0.5,
        full_run_completed=False,
        leakage_warning=False,
        fold_count=2,
    )
    required_metrics = [
        "candidate_name", "model_name", "label_name", "selector",
        "trade_count", "profit_factor", "sharpe", "cost_stress_1_25x_pf",
        "cost_stress_1_50x_pf", "expected_return_after_cost",
        "median_return_after_cost", "return_to_cost_ratio",
        "median_return_to_cost_ratio", "profitable_days", "losing_days",
        "profitable_months", "losing_months", "walk_forward_worst_fold_pf",
        "walk_forward_median_pf", "top_2_days_profit_share",
        "best_fold_profit_share", "daily_stability_passes",
        "threshold_stability_passes", "fold_stability_passes",
        "status", "failed_gates"
    ]
    for m in required_metrics:
        assert m in candidate, f"Metric '{m}' is missing from candidate dictionary"
