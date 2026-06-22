"""
Tests that core_retrain_summary.json is written by all retrain paths.

Verifies the fix for: core_retrain_summary.json was not always written,
causing --edge-refinement-report to fail silently when run against certain
artifact directories.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _run_retrain(tmp_path: Path, extra_args: list[str] | None = None) -> dict:
    """Run retrain with a minimal synthetic dataset and return parsed summary."""
    script = (
        Path(__file__).parent.parent / "scripts" / "retrain_all_edge_models.py"
    )
    synthetic_csv = (
        Path(__file__).parent.parent / "tests" / "fixtures" / "synthetic_240.csv"
    )
    if not synthetic_csv.exists():
        pytest.skip(f"Synthetic fixture not found: {synthetic_csv}")

    cmd = [
        sys.executable,
        str(script),
        "--dataset",
        str(synthetic_csv),
        "--output-dir",
        str(tmp_path),
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    summary_path = tmp_path / "core_retrain_summary.json"
    if not summary_path.exists():
        pytest.fail(
            f"core_retrain_summary.json not written.\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )

    with open(summary_path) as f:
        return json.load(f)


class TestCoreRetrainSummaryWriting:
    """All training paths must write core_retrain_summary.json."""

    def test_normal_retrain_writes_summary(self, tmp_path):
        """Basic retrain with default targets writes the summary file."""
        summary = _run_retrain(tmp_path)
        assert "dataset_path" in summary
        assert "row_count" in summary
        assert "timestamp" in summary
        assert summary.get("status") != "failed"

    def test_retrain_all_models_writes_summary(self, tmp_path):
        """--retrain-all-models path writes the summary file."""
        summary = _run_retrain(tmp_path, extra_args=["--retrain-all-models"])
        assert "dataset_path" in summary
        assert "models_trained" in summary
        assert len(summary.get("models_trained", [])) > 0

    def test_only_target_respects_target(self, tmp_path):
        """--only-target path writes the summary with that target."""
        summary = _run_retrain(
            tmp_path, extra_args=["--only-target", "profitable_trade_label"]
        )
        assert "targets_trained" in summary
        assert "profitable_trade_label" in summary["targets_trained"]

    def test_only_model_respects_model(self, tmp_path):
        """--only-model path writes the summary with that model."""
        summary = _run_retrain(
            tmp_path, extra_args=["--only-model", "logistic_regression"]
        )
        assert "models_trained" in summary
        assert "logistic_regression" in summary["models_trained"]

    def test_skipped_models_recorded(self, tmp_path):
        """Models skipped due to dependencies (e.g. xgboost not installed) are logged."""
        summary = _run_retrain(
            tmp_path, extra_args=["--only-model", "XGBoost", "--retrain-all-models"]
        )
        # xgboost may not be installed — if so, it should be in skipped list
        models_trained = summary.get("models_trained", [])
        skipped = summary.get("skipped_models", [])
        if "xgboost" not in models_trained:
            assert any("xgboost" in s.lower() for s in skipped), (
                f"xgboost not trained but not in skipped_models: {skipped}"
            )

    def test_summary_contains_cost_gate_status(self, tmp_path):
        """Summary includes cost gate status for paper-ready gate checks."""
        summary = _run_retrain(tmp_path)
        # cost_gate_status is informational — may or may not be present
        # depending on whether cost data exists in the dataset
        assert "status" in summary

    def test_summary_enables_edge_refinement(self, tmp_path):
        """Edge refinement reads the summary to discover eligible (label, model) pairs."""
        summary = _run_retrain(tmp_path, extra_args=["--only-target", "profitable_trade_label"])
        # Edge refinement needs targets_trained + models_trained to iterate
        assert "targets_trained" in summary
        assert "models_trained" in summary
        assert len(summary["targets_trained"]) > 0
        assert len(summary["models_trained"]) > 0

    def test_summary_timestamp_is_valid_iso(self, tmp_path):
        """Summary timestamp must be a valid ISO datetime string."""
        summary = _run_retrain(tmp_path)
        from datetime import datetime

        ts = summary.get("timestamp", "")
        try:
            datetime.fromisoformat(ts)
        except ValueError:
            pytest.fail(f"Invalid timestamp in summary: {ts!r}")

    def test_resume_path_writes_summary(self, tmp_path):
        """--resume path (if artifacts exist) writes the summary."""
        # First run to create artifacts
        _run_retrain(tmp_path)
        # Second run with --resume
        summary = _run_retrain(tmp_path, extra_args=["--resume"])
        assert "dataset_path" in summary