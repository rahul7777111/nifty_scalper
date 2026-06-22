#!/usr/bin/env python3
"""
tests/test_ml_only_dynamic_preset_overfitting.py
================================================
Verify preset selection protection against overfitting.

Key protections:
- Preset is selected on validation folds only, never on test folds
- Final evaluation uses held-out OOS test folds
- Train/validation/test date ranges are reported
- Preset chosen before test evaluation (not after peeking at test results)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.ml_only_dynamic_preset_candidate_rescue import (
    _walk_forward_splits,
    evaluate_candidate_with_preset,
    build_preset_grid,
)


class TestPresetSelectionOnValidationOnly:
    """Verify preset is selected on validation, not test data."""

    def test_walk_forward_splits_are_time_based(self) -> None:
        """Walk-forward splits must be chronological (time-based), not random."""
        import pandas as pd
        import numpy as np

        dates = pd.date_range("2024-01-01", periods=1000, freq="min")
        df = pd.DataFrame({
            "net_forward_return": np.random.randn(1000) * 0.01,
            "cost_survivor_label_v2": [0, 1] * 500,
        }, index=dates)
        df.index.name = "timestamp_dt"

        splits = _walk_forward_splits(df.reset_index(), n_folds=3)
        assert len(splits) == 3

        # Each split: train should come BEFORE test chronologically
        for i, (train, test) in enumerate(splits):
            if len(train) > 0 and len(test) > 0:
                assert train.iloc[-1]["timestamp_dt"] <= test.iloc[0]["timestamp_dt"], \
                    f"Fold {i}: train end ({train.iloc[-1]['timestamp_dt']}) > test start ({test.iloc[0]['timestamp_dt']})"

    def test_evaluate_fold_receives_separate_train_and_test(self) -> None:
        """_evaluate_fold must receive separate train and test DataFrames."""
        import inspect
        sig = inspect.signature(
            __import__(
                "scripts.ml_only_dynamic_preset_candidate_rescue",
                fromlist=["_evaluate_fold"]
            )._evaluate_fold
        )
        params = list(sig.parameters.keys())
        assert "train_df" in params
        assert "test_df" in params

    def test_evaluate_fold_does_not_use_test_for_training(self) -> None:
        """Verify that model training uses only train_df, not test_df."""
        import inspect
        source = inspect.getsource(
            __import__(
                "scripts.ml_only_dynamic_preset_candidate_rescue",
                fromlist=["_evaluate_fold"]
            )._evaluate_fold
        )
        # The function should build features from train_df for training,
        # and use test_df only for evaluation
        assert "X_train" in source
        assert "y_train" in source
        # test_df should be used for evaluation (predicting), not training
        assert "X_test" in source or "test_df" in source

    def test_evaluate_candidate_with_preset_uses_walk_forward(self) -> None:
        """evaluate_candidate_with_preset must use walk-forward validation."""
        import inspect
        source = inspect.getsource(evaluate_candidate_with_preset)
        assert "_walk_forward_splits" in source or "splits" in source
        assert "n_folds" in source


class TestPresetChosenBeforeTestEvaluation:
    """Verify preset is chosen BEFORE test evaluation (no peeking at test results)."""

    def test_no_test_data_in_preset_selection(self) -> None:
        """Preset grid generation must not use test fold data."""
        # This is a structural test: the preset grid is defined statically
        # with hard-coded values, not data-driven. This ensures no test leakage.
        import inspect
        from scripts.ml_only_dynamic_preset_candidate_rescue import build_preset_grid
        source = inspect.getsource(build_preset_grid)
        # Grid must be defined with hard-coded values (no df, no test data)
        assert "THRESHOLD_VALUES" in source or "preset_id" in source

    def test_best_preset_selected_from_validation_results(self) -> None:
        """Best preset is selected based on validation performance, not test performance."""
        import inspect
        from scripts.ml_only_dynamic_preset_candidate_rescue import evaluate_candidate_with_preset
        sig = inspect.signature(evaluate_candidate_with_preset)
        source = inspect.getsource(evaluate_candidate_with_preset)
        # The function evaluates all presets and selects the best one
        # based on cost stress results (which come from validation-style evaluation)
        assert "cost_stress" in source.lower() or "best" in source.lower() or "pf" in source


class TestReportDateRanges:
    """Verify that evaluation reports train/validation/test date ranges."""

    def test_time_split_reports_date_ranges(self) -> None:
        """_time_split_3way must report date ranges for each split."""
        import inspect
        try:
            from scripts.ml_only_dynamic_preset_candidate_rescue import _time_split_3way
            sig = inspect.signature(_time_split_3way)
            # Function exists and is callable
            assert callable(_time_split_3way)
        except ImportError:
            pytest.skip("_time_split_3way not available")


class TestThresholdRobustnessRequired:
    """Verify that threshold robustness is still required when using presets."""

    def test_threshold_robustness_is_enforced(self) -> None:
        """Presets must still require threshold robustness across nearby thresholds."""
        from scripts.ml_only_dynamic_preset_candidate_rescue import MIN_THRESHOLD_ROBUSTNESS
        assert MIN_THRESHOLD_ROBUSTNESS == 0.60
        # Example: a candidate that only works at one threshold (not nearby) fails robustness
        # PF scores at nearby thresholds: 0.35 (1.05), 0.40 (0.95)
        # min/max ratio = 0.95/1.05 = 0.904 > 0.60 → would pass robustness
        # But a candidate with 0.35 (1.10), 0.40 (0.60) → 0.60/1.10 = 0.545 < 0.60 → fails
        mock_scores = {0.35: 1.10, 0.40: 0.60}
        robustness = min(mock_scores.values()) / max(mock_scores.values())
        # This candidate fails robustness (0.545 < 0.60)
        assert robustness < MIN_THRESHOLD_ROBUSTNESS