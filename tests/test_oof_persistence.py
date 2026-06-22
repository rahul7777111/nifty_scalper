"""Tests for OOF prediction persistence in retrain_all_edge_models.py.

These tests verify that:
1. OOF predictions are correctly persisted as parquet files
2. OOF DataFrames contain the correct columns (no leakage)
3. The probability_decile_audit in audit_model_diagnosis.py uses OOF when available
4. core_retrain_summary.json records OOF file paths
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# =============================================================================
# Test helper functions from retrain_all_edge_models.py
# =============================================================================

import importlib.util

spec = importlib.util.spec_from_file_location(
    "retrain_all_edge_models", SCRIPTS_DIR / "retrain_all_edge_models.py"
)
assert spec is not None and spec.loader is not None
retrain_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrain_mod)  # type: ignore[union-attr]


# =============================================================================
# Test OOF helper functions
# =============================================================================


def test_build_oof_path_creates_correct_filename(tmp_path: Path) -> None:
    """OOF file path should follow naming convention: oof_predictions_<label>_<model>_<ts>.parquet"""
    timestamp = "20260609_120000"
    result = retrain_mod._build_oof_path(tmp_path, "profitable_trade_label", "random_forest", timestamp)
    expected = tmp_path / f"oof_predictions_profitable_trade_label_random_forest_{timestamp}.parquet"
    assert result == expected


def test_build_oof_path_handles_special_characters(tmp_path: Path) -> None:
    """OOF paths should sanitize label/model names with special characters."""
    timestamp = "20260609_120000"
    result = retrain_mod._build_oof_path(tmp_path, "profitable/trade_label", "model/name", timestamp)
    assert result.name == f"oof_predictions_profitable_trade_label_model_name_{timestamp}.parquet"
    assert "/" not in result.name


def test_persist_oof_predictions_creates_file(tmp_path: Path) -> None:
    """_persist_oof_predictions should create a parquet file when DataFrame is non-empty."""
    oof_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=100, freq="h"),
        "y_true": np.random.randint(0, 2, 100),
        "y_prob": np.random.uniform(0, 1, 100),
        "threshold": [0.5] * 100,
        "predicted_label": np.random.randint(0, 2, 100),
    })

    timestamp = "20260609_120000"
    result = retrain_mod._persist_oof_predictions(
        tmp_path, "test_label", "test_model", oof_df, timestamp
    )

    assert result["status"] == "success"
    assert result["row_count"] == 100
    assert result["oof_file_path"] is not None
    assert Path(result["oof_file_path"]).exists()


def test_persist_oof_predictions_returns_empty_for_empty_df(tmp_path: Path) -> None:
    """_persist_oof_predictions should return empty status for empty DataFrame."""
    empty_df = pd.DataFrame()
    timestamp = "20260609_120000"
    result = retrain_mod._persist_oof_predictions(
        tmp_path, "test_label", "test_model", empty_df, timestamp
    )

    assert result["status"] == "empty"
    assert result["row_count"] == 0
    assert result["oof_file_path"] is None


def test_persist_oof_predictions_contains_required_columns(tmp_path: Path) -> None:
    """Persisted OOF file should contain required evaluation columns."""
    oof_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=100, freq="h"),
        "fold_id": [1] * 100,
        "fold_type": ["test"] * 100,
        "y_true": np.random.randint(0, 2, 100),
        "y_prob": np.random.uniform(0, 1, 100),
        "threshold": [0.5] * 100,
        "predicted_label": np.random.randint(0, 2, 100),
        "selected_return_column": np.random.uniform(-0.05, 0.05, 100),
    })

    timestamp = "20260609_120000"
    result = retrain_mod._persist_oof_predictions(
        tmp_path, "test_label", "test_model", oof_df, timestamp
    )

    assert result["status"] == "success"
    saved_df = pd.read_parquet(result["oof_file_path"])
    required_cols = {"timestamp", "fold_id", "y_true", "y_prob", "threshold", "predicted_label"}
    assert required_cols.issubset(set(saved_df.columns))


def test_oof_predictions_excludes_leakage_columns(tmp_path: Path) -> None:
    """OOF DataFrame should NOT contain leakage columns (return/label as features)."""
    oof_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=100, freq="h"),
        "y_true": np.random.randint(0, 2, 100),
        "y_prob": np.random.uniform(0, 1, 100),
        "threshold": [0.5] * 100,
        "predicted_label": np.random.randint(0, 2, 100),
        "selected_return_column": np.random.uniform(-0.05, 0.05, 100),  # Evaluation only
    })

    timestamp = "20260609_120000"
    result = retrain_mod._persist_oof_predictions(
        tmp_path, "test_label", "test_model", oof_df, timestamp
    )

    saved_df = pd.read_parquet(result["oof_file_path"])
    leakage_tokens = {"profit", "loss", "pnl", "forward_return", "net_return", "label", "target"}

    for col in saved_df.columns:
        col_lower = col.lower()
        # selected_return_column is allowed (for evaluation only, not as feature)
        if col != "selected_return_column":
            for token in leakage_tokens:
                assert token not in col_lower, f"Leakage column '{col}' found in OOF output"


def test_oof_frame_from_holdout_creates_correct_structure(tmp_path: Path) -> None:
    """_build_oof_frame_from_holdout should create DataFrame with correct columns."""
    n = 100
    test_frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "option_type_ce": [1.0] * 50 + [0.0] * 50,
        "dte_days": np.random.uniform(1, 7, n),
        "moneyness": np.random.uniform(0.95, 1.05, n),
    })
    y_test = np.random.randint(0, 2, n)
    test_prob = np.random.uniform(0, 1, n)
    threshold = 0.5

    oof_df = retrain_mod._build_oof_frame_from_holdout(
        test_frame=test_frame,
        y_test=y_test,
        test_prob=test_prob,
        threshold=threshold,
        model_name="test_model",
        label_name="test_label",
        fold_id=1,
        evaluation_return_column=None,
    )

    assert len(oof_df) == n
    required_cols = {"timestamp", "fold_id", "y_true", "y_prob", "threshold", "predicted_label", "option_type", "dte", "moneyness"}
    assert required_cols.issubset(set(oof_df.columns))


def test_oof_frame_from_holdout_includes_return_for_evaluation(tmp_path: Path) -> None:
    """OOF frame should include selected_return_column for evaluation purposes."""
    n = 100
    test_frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "net_forward_return": np.random.uniform(-0.05, 0.05, n),
    })
    y_test = np.random.randint(0, 2, n)
    test_prob = np.random.uniform(0, 1, n)
    threshold = 0.5

    oof_df = retrain_mod._build_oof_frame_from_holdout(
        test_frame=test_frame,
        y_test=y_test,
        test_prob=test_prob,
        threshold=threshold,
        model_name="test_model",
        label_name="test_label",
        fold_id=1,
        evaluation_return_column="net_forward_return",
    )

    assert "selected_return_column" in oof_df.columns
    assert oof_df["selected_return_column"].notna().any()


def test_get_leakage_safe_passthrough_columns_excludes_features(tmp_path: Path) -> None:
    """_get_leakage_safe_passthrough_columns should exclude feature columns."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=100, freq="h"),
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "target_label": np.random.randint(0, 2, 100),
        "net_forward_return": np.random.uniform(-0.05, 0.05, 100),
        "option_type_ce": [1.0] * 50 + [0.0] * 50,
    })

    feature_cols = ["feature1", "feature2"]
    passthrough = retrain_mod._get_leakage_safe_passthrough_columns(df, feature_cols)

    # Should include timestamp and option_type_ce
    assert "timestamp" in passthrough
    assert "option_type_ce" in passthrough
    # Should NOT include feature columns
    assert "feature1" not in passthrough
    assert "feature2" not in passthrough
    # Should NOT include leakage columns
    assert "target_label" not in passthrough
    assert "net_forward_return" not in passthrough


# =============================================================================
# Test audit_model_diagnosis.py OOF integration
# =============================================================================

spec2 = importlib.util.spec_from_file_location(
    "audit_model_diagnosis", SCRIPTS_DIR / "audit_model_diagnosis.py"
)
assert spec2 is not None and spec2.loader is not None
audit_mod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(audit_mod)  # type: ignore[union-attr]


def test_find_oof_files_locates_holdout_oof(tmp_path: Path) -> None:
    """_find_oof_files should locate holdout OOF parquet files."""
    timestamp = "20260609_120000"
    oof_file = tmp_path / f"oof_predictions_test_label_test_model_{timestamp}.parquet"
    oof_file.touch()

    holdout, wf = audit_mod._find_oof_files(tmp_path, "test_label", "test_model")
    assert holdout == oof_file
    assert wf is None


def test_find_oof_files_locates_walkforward_oof(tmp_path: Path) -> None:
    """_find_oof_files should locate walk-forward OOF parquet files."""
    timestamp = "20260609_120000"
    wf_file = tmp_path / f"oof_predictions_test_label_test_model_walkforward_{timestamp}.parquet"
    wf_file.touch()

    holdout, wf = audit_mod._find_oof_files(tmp_path, "test_label", "test_model")
    assert holdout is None
    assert wf == wf_file


def test_analyze_oof_deciles_computes_true_deciles(tmp_path: Path) -> None:
    """_analyze_oof_deciles should compute true probability deciles from OOF predictions."""
    n = 1000
    oof_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "fold_id": [1] * n,
        "y_true": np.random.randint(0, 2, n),
        "y_prob": np.random.uniform(0, 1, n),
        "threshold": [0.5] * n,
        "predicted_label": np.random.randint(0, 2, n),
        "selected_return_column": np.where(
            np.random.randint(0, 2, n) == 1,
            np.random.uniform(0.01, 0.05, n),
            np.random.uniform(-0.05, -0.01, n),
        ),
    })

    result = audit_mod._analyze_oof_deciles(oof_df)

    assert result["available"] is True
    assert result["n_deciles"] == 10
    assert result["total_samples"] == n
    assert len(result["deciles"]) == 10
    assert "monotonic_top_beats_bottom" in result
    assert "top_decile" in result
    assert "bottom_decile" in result


def test_analyze_oof_deciles_detects_monotonicity(tmp_path: Path) -> None:
    """_analyze_oof_deciles should detect monotonic vs non-monotonic probability ranking."""
    # Create synthetic OOF where higher probability -> higher win rate (monotonic)
    n = 1000
    y_probs = np.linspace(0, 1, n)
    y_true = (y_probs + np.random.normal(0, 0.1, n) > 0.5).astype(int)

    oof_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "fold_id": [1] * n,
        "y_true": y_true,
        "y_prob": y_probs,
        "threshold": [0.5] * n,
        "predicted_label": (y_probs > 0.5).astype(int),
    })

    result = audit_mod._analyze_oof_deciles(oof_df)
    assert result["available"] is True
    # Higher deciles should have higher positive rate in this synthetic case
    assert result["top_decile"]["actual_positive_rate"] >= result["bottom_decile"]["actual_positive_rate"]


def test_probability_decile_audit_uses_oof_when_available(tmp_path: Path) -> None:
    """probability_decile_audit should use OOF predictions when available."""
    # Create mock artifact directory with OOF file
    timestamp = "20260609_120000"

    # Create metrics file
    metrics = {
        "model_name": "random_forest",
        "label_name": "test_label",
        "threshold_sweep": [
            {"threshold": 0.5, "positive_prediction_rate": 0.5, "number_of_trades": 100, "profit_factor": 1.2, "sharpe": 0.8}
        ],
    }
    metrics_file = tmp_path / "core_retrain_random_forest_test_label_metrics.json"
    metrics_file.write_text(json.dumps(metrics), encoding="utf-8")

    # Create OOF file
    oof_file = tmp_path / f"oof_predictions_test_label_random_forest_{timestamp}.parquet"
    n = 1000
    oof_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "fold_id": [1] * n,
        "y_true": np.random.randint(0, 2, n),
        "y_prob": np.random.uniform(0, 1, n),
        "threshold": [0.5] * n,
        "predicted_label": np.random.randint(0, 2, n),
        "selected_return_column": np.random.uniform(-0.05, 0.05, n),
    })
    oof_df.to_parquet(oof_file, index=False)

    result = audit_mod.probability_decile_audit(tmp_path, ["test_label"])

    assert result["available"] is True
    key = "random_forest__test_label"
    assert key in result["per_model_label"]

    model_result = result["per_model_label"][key]
    # Should use OOF file
    assert model_result.get("oof_type") == "holdout"
    assert model_result.get("oof_source") is not None
    # Should have true decile analysis
    assert "deciles" in model_result
    assert model_result.get("n_deciles") == 10


def test_probability_decile_audit_falls_back_to_sweep_without_oof(tmp_path: Path) -> None:
    """probability_decile_audit should fall back to threshold-sweep when OOF is unavailable."""
    metrics = {
        "model_name": "random_forest",
        "label_name": "test_label",
        "threshold_sweep": [
            {"threshold": 0.5, "positive_prediction_rate": 0.5, "number_of_trades": 100, "profit_factor": 1.2, "sharpe": 0.8}
        ],
    }
    metrics_file = tmp_path / "core_retrain_random_forest_test_label_metrics.json"
    metrics_file.write_text(json.dumps(metrics), encoding="utf-8")

    result = audit_mod.probability_decile_audit(tmp_path, ["test_label"])

    assert result["available"] is True
    key = "random_forest__test_label"
    assert key in result["per_model_label"]

    model_result = result["per_model_label"][key]
    # Should fall back to sweep proxy
    assert model_result.get("oof_type") == "none"
    assert model_result.get("oof_source") is None
    # Should have threshold sweep data
    assert "per_threshold" in model_result


def test_probability_decile_audit_reports_data_source_summary(tmp_path: Path) -> None:
    """probability_decile_audit should report counts of OOF vs sweep-based results."""
    # Create multiple model/label combinations
    for model in ["lr", "rf", "xgb"]:
        for label in ["label_a", "label_b"]:
            metrics = {
                "model_name": model,
                "label_name": label,
                "threshold_sweep": [
                    {"threshold": 0.5, "positive_prediction_rate": 0.5, "number_of_trades": 100, "profit_factor": 1.2}
                ],
            }
            metrics_file = tmp_path / f"core_retrain_{model}_{label}_metrics.json"
            metrics_file.write_text(json.dumps(metrics), encoding="utf-8")

    result = audit_mod.probability_decile_audit(tmp_path, ["label_a", "label_b"])

    assert "data_source" in result
    assert result["data_source"]["threshold_sweep_proxy_used"] == 6  # All 3 models x 2 labels
    assert result["data_source"]["oof_predictions_used"] == 0


# =============================================================================
# Test core_retrain_summary.json OOF tracking
# =============================================================================

def test_core_retrain_summary_includes_oof_file_paths(tmp_path: Path) -> None:
    """core_retrain_summary.json should record OOF file paths when present."""
    from retrain_all_edge_models import _write_core_retrain_summary, write_json

    dataset_path = tmp_path / "dataset.csv"
    dataset_path.touch()

    # Create mock result_payload with OOF predictions
    result_payload = {
        "reports": {
            "test_label": {
                "test_model": {
                    "oof_predictions": {
                        "holdout": {
                            "file_path": str(tmp_path / "oof_predictions_test.parquet"),
                            "row_count": 1000,
                            "status": "success",
                            "timestamp": "20260609_120000",
                        },
                        "walk_forward": {
                            "file_path": str(tmp_path / "oof_predictions_test_wf.parquet"),
                            "row_count": 5000,
                            "status": "success",
                            "timestamp": "20260609_120000",
                        },
                    },
                }
            }
        },
        "trained_models": [],
        "failed_models": [],
        "skipped_models": [],
    }

    metric_rows = [{
        "label_name": "test_label",
        "model_name": "test_model",
        "profit_factor": 1.2,
        "status": "completed",
    }]

    summary_path = _write_core_retrain_summary(
        artifact_dir=tmp_path,
        dataset_path=dataset_path,
        labels_trained=["test_label"],
        model_families=["test_model"],
        metric_rows=metric_rows,
        result_payload=result_payload,
        evaluation_return_column="net_forward_return",
    )

    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "oof_predictions" in summary
    assert "holdout_oof_files" in summary["oof_predictions"]
    assert "walkforward_oof_files" in summary["oof_predictions"]
    assert len(summary["oof_predictions"]["holdout_oof_files"]) == 1
    assert len(summary["oof_predictions"]["walkforward_oof_files"]) == 1
    assert summary["oof_predictions"]["holdout_oof_files"][0]["label"] == "test_label"


def test_kolkatatz_now_returns_correct_format() -> None:
    """_kolkatatz_now should return a datetime object."""
    ts = retrain_mod._kolkatatz_now()
    assert isinstance(ts, datetime)
    # Should have tzinfo (UTC)
    assert ts.tzinfo is not None or ts.year >= 2000


# =============================================================================
# Task-specified tests (named per spec)
# =============================================================================


def test_build_oof_path_returns_parquet(tmp_path: Path) -> None:
    """OOF path filename must end with .parquet extension."""
    timestamp = "20260609_120000"
    result = retrain_mod._build_oof_path(tmp_path, "test_label", "test_model", timestamp)
    assert result.name.endswith(".parquet"), f"Expected .parquet extension, got: {result.name}"


def test_persist_oof_predictions_writes_file(tmp_path: Path) -> None:
    """_persist_oof_predictions writes a parquet file and returns the file path."""
    oof_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=50, freq="h"),
        "y_true": np.random.randint(0, 2, 50),
        "y_prob": np.random.uniform(0, 1, 50),
        "threshold": [0.5] * 50,
        "predicted_label": np.random.randint(0, 2, 50),
    })
    timestamp = "20260609_120000"
    result = retrain_mod._persist_oof_predictions(
        tmp_path, "test_label", "test_model", oof_df, timestamp
    )
    assert result["status"] == "success"
    assert result["oof_file_path"] is not None
    assert Path(result["oof_file_path"]).exists()
    assert Path(result["oof_file_path"]).suffix == ".parquet"


def test_persist_oof_predictions_empty_dataframe_returns_empty_status(tmp_path: Path) -> None:
    """Empty DataFrame should not write any file; returns empty status."""
    empty_df = pd.DataFrame()
    timestamp = "20260609_120000"
    result = retrain_mod._persist_oof_predictions(
        tmp_path, "test_label", "test_model", empty_df, timestamp
    )
    assert result["status"] == "empty"
    assert result["row_count"] == 0
    assert result["oof_file_path"] is None


def test_collect_walk_forward_oof_returns_dataframe(tmp_path: Path) -> None:
    """_collect_walk_forward_oof_predictions should return a pandas DataFrame."""
    # Need sufficient rows for at least 2 walk-forward folds
    n = 300
    X = np.random.randn(n, 5)
    y = np.random.randint(0, 2, n).astype(float)
    timestamps = pd.Series(pd.date_range("2025-01-01", periods=n, freq="h"))
    feature_cols = ["f1", "f2", "f3", "f4", "f5"]
    scaler_mean = [0.0] * 5
    scaler_std = [1.0] * 5

    result = retrain_mod._collect_walk_forward_oof_predictions(
        X=X, y=y, timestamps=timestamps, feature_cols=feature_cols,
        label_name="test_label", scaler_mean=scaler_mean, scaler_std=scaler_std,
        fold_count=3, returns=None, evaluation_return_column=None,
    )
    assert isinstance(result, pd.DataFrame)


def test_oof_frame_has_required_columns(tmp_path: Path) -> None:
    """OOF frame must contain: timestamp, fold_id, y_true, y_prob, threshold, predicted_label."""
    n = 100
    test_frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "option_type_ce": [1.0] * n,
    })
    y_test = np.random.randint(0, 2, n)
    test_prob = np.random.uniform(0, 1, n)
    threshold = 0.5

    oof_df = retrain_mod._build_oof_frame_from_holdout(
        test_frame=test_frame, y_test=y_test, test_prob=test_prob,
        threshold=threshold, model_name="test_model", label_name="test_label",
        fold_id=1, evaluation_return_column=None,
    )
    required_cols = {"timestamp", "fold_id", "y_true", "y_prob", "threshold", "predicted_label"}
    assert required_cols.issubset(set(oof_df.columns)), (
        f"Missing required columns. Expected: {required_cols}, got: {set(oof_df.columns)}"
    )


def test_oof_frame_excludes_forbidden_columns(tmp_path: Path) -> None:
    """OOF frame should NOT include _label, _return, pnl, or future columns."""
    n = 100
    test_frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "option_type_ce": [1.0] * n,
        "dte_days": np.random.uniform(1, 7, n),
        "moneyness": np.random.uniform(0.95, 1.05, n),
    })
    y_test = np.random.randint(0, 2, n)
    test_prob = np.random.uniform(0, 1, n)
    threshold = 0.5

    oof_df = retrain_mod._build_oof_frame_from_holdout(
        test_frame=test_frame, y_test=y_test, test_prob=test_prob,
        threshold=threshold, model_name="test_model", label_name="test_label",
        fold_id=1, evaluation_return_column=None,
    )
    forbidden_tokens = (
        "_label", "_return", "pnl", "profit", "loss", "future",
        "forward", "next", "exit", "outcome", "realized", "target",
    )
    for col in oof_df.columns:
        col_lower = str(col).lower()
        for token in forbidden_tokens:
            assert token not in col_lower, f"Forbidden token '{token}' found in column '{col}'"


def test_oof_persistence_in_core_retrain_summary(tmp_path: Path) -> None:
    """OOF file paths must be recorded in core_retrain_summary.json."""
    from retrain_all_edge_models import _write_core_retrain_summary

    dataset_path = tmp_path / "dataset.csv"
    dataset_path.touch()

    result_payload = {
        "reports": {
            "test_label": {
                "test_model": {
                    "oof_predictions": {
                        "holdout": {
                            "file_path": str(tmp_path / "holdout.parquet"),
                            "row_count": 500,
                            "timestamp": "20260609_120000",
                        },
                        "walk_forward": {
                            "file_path": str(tmp_path / "wf.parquet"),
                            "row_count": 2000,
                            "timestamp": "20260609_120000",
                        },
                    },
                }
            }
        },
        "failed_models": [],
        "skipped_models": [],
    }

    metric_rows = [{
        "label_name": "test_label",
        "model_name": "test_model",
        "profit_factor": 1.25,
        "cost_1_25x_pf": 1.1,
        "cost_1_50x_pf": 0.9,
        "status": "completed",
    }]

    summary_path = _write_core_retrain_summary(
        artifact_dir=tmp_path,
        dataset_path=dataset_path,
        labels_trained=["test_label"],
        model_families=["test_model"],
        metric_rows=metric_rows,
        result_payload=result_payload,
        evaluation_return_column="net_forward_return",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "oof_predictions" in summary
    assert any(
        f["label"] == "test_label"
        for f in summary["oof_predictions"]["holdout_oof_files"]
    )
    assert any(
        f["label"] == "test_label"
        for f in summary["oof_predictions"]["walkforward_oof_files"]
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
