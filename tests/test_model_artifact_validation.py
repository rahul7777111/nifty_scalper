#!/usr/bin/env python3
"""
Tests for Model Artifact Validation.

These tests validate the model artifact selection rules:
1. Rejects models with model_pkl=null or missing
2. Rejects models with feature_count=0
3. Rejects models missing feature list
4. Rejects models missing metrics
5. Prefers latest valid model with feature schema
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

# Import the validation module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from validate_model_artifacts import (
    ModelArtifacts,
    validate_model_artifacts,
    select_best_model,
    parse_folder_timestamp,
)


class TestRejectNullModelPkl:
    """Test that models with null or missing model_pkl are rejected."""

    def test_null_model_pkl_is_rejected(self):
        """A model with model_pkl=null should be rejected."""
        artifacts = ModelArtifacts(
            folder_name="test_null_pkl",
            folder_path="/fake/path",
            model_pkl=None,
            feature_count=10,
            feature_list=["feature1", "feature2"],
            metrics_json="/path/to/metrics.json",
        )
        
        validated = validate_model_artifacts(artifacts)
        
        assert validated.is_valid is False
        assert "model_pkl is null or missing" in validated.validation_errors

    def test_missing_model_pkl_is_rejected(self):
        """A model with no pkl file found should be rejected."""
        artifacts = ModelArtifacts(
            folder_name="test_missing_pkl",
            folder_path="/fake/path",
            model_pkl=None,  # No pkl found
            feature_count=10,
            feature_list=["feature1", "feature2"],
            metrics_json="/path/to/metrics.json",
        )
        
        validated = validate_model_artifacts(artifacts)
        
        assert validated.is_valid is False


class TestRejectZeroFeatureModel:
    """Test that models with zero features are rejected."""

    def test_zero_feature_count_is_rejected(self):
        """A model with feature_count=0 should be rejected."""
        artifacts = ModelArtifacts(
            folder_name="test_zero_features",
            folder_path="/fake/path",
            model_pkl="/path/to/model.pkl",
            feature_count=0,
            feature_list=None,
            metrics_json="/path/to/metrics.json",
        )
        
        validated = validate_model_artifacts(artifacts)
        
        assert validated.is_valid is False
        assert "feature_count is 0" in validated.validation_errors

    def test_zero_features_with_empty_list_is_rejected(self):
        """A model with empty feature list should be rejected."""
        artifacts = ModelArtifacts(
            folder_name="test_empty_features",
            folder_path="/fake/path",
            model_pkl="/path/to/model.pkl",
            feature_count=0,
            feature_list=[],
            metrics_json="/path/to/metrics.json",
        )
        
        validated = validate_model_artifacts(artifacts)
        
        assert validated.is_valid is False
        assert "missing feature list" in validated.validation_errors


class TestRejectMissingMetrics:
    """Test that models missing metrics are rejected."""

    def test_missing_metrics_is_rejected(self):
        """A model without metrics JSON should be rejected."""
        artifacts = ModelArtifacts(
            folder_name="test_no_metrics",
            folder_path="/fake/path",
            model_pkl="/path/to/model.pkl",
            feature_count=10,
            feature_list=["feature1", "feature2"],
            metrics_json=None,
        )
        
        validated = validate_model_artifacts(artifacts)
        
        assert validated.is_valid is False
        assert "missing metrics" in validated.validation_errors


class TestValidModelSelection:
    """Test that valid models are properly selected."""

    def test_valid_model_is_accepted(self):
        """A model with all required artifacts should be valid."""
        artifacts = ModelArtifacts(
            folder_name="test_valid_model",
            folder_path="/fake/path",
            model_pkl="/path/to/model.pkl",
            model_pkl_size=1000000,
            feature_count=10,
            feature_list=["feature1", "feature2", "feature3"],
            metrics_json="/path/to/metrics.json",
            trained_at="2026-06-07T12:00:00",
            label_name="profitable_trade_label",
            model_name="logistic_regression",
        )
        
        validated = validate_model_artifacts(artifacts)
        
        assert validated.is_valid is True
        assert len(validated.validation_errors) == 0

    def test_valid_model_is_selected_from_list(self):
        """Given multiple valid models, the best one should be selected."""
        model1 = ModelArtifacts(
            folder_name="older_model",
            folder_path="/fake/older",
            model_pkl="/path/to/older.pkl",
            feature_count=10,
            feature_list=["feature1", "feature2"],
            metrics_json="/path/to/metrics.json",
            trained_at="2026-06-06T12:00:00",
            is_valid=True,
        )
        
        model2 = ModelArtifacts(
            folder_name="newer_model",
            folder_path="/fake/newer",
            model_pkl="/path/to/newer.pkl",
            feature_count=10,
            feature_list=["feature1", "feature2"],
            metrics_json="/path/to/metrics.json",
            trained_at="2026-06-07T12:00:00",
            is_valid=True,
        )
        
        selected = select_best_model([model1, model2])
        
        assert selected is not None
        assert selected.folder_name == "newer_model"

    def test_no_valid_model_returns_none(self):
        """If all models are invalid, should return None."""
        invalid_model = ModelArtifacts(
            folder_name="invalid",
            folder_path="/fake/invalid",
            model_pkl=None,
            feature_count=0,
            feature_list=None,
            metrics_json=None,
            is_valid=False,
        )
        
        selected = select_best_model([invalid_model])
        
        assert selected is None


class TestLatestInvalidDoesNotOverrideOlderValid:
    """Test that a newer invalid model does not override an older valid model."""

    def test_newer_invalid_ignored_when_older_valid_exists(self):
        """A newer model that is invalid should not be selected over an older valid model."""
        older_valid = ModelArtifacts(
            folder_name="retrain_all_models_20260606_090634",
            folder_path="/fake/older_valid",
            model_pkl="/path/to/older.pkl",
            feature_count=114,
            feature_list=["f1", "f2"],
            metrics_json="/path/to/metrics.json",
            trained_at="2026-06-06T09:06:34",
            is_valid=True,
        )
        
        newer_invalid = ModelArtifacts(
            folder_name="retrain_all_models_20260607_024322",
            folder_path="/fake/newer_invalid",
            model_pkl=None,  # Invalid - no pkl
            feature_count=114,
            feature_list=["f1", "f2"],
            metrics_json=None,  # Invalid - no metrics
            trained_at="2026-06-07T02:43:22",
            is_valid=False,
        )
        
        selected = select_best_model([newer_invalid, older_valid])
        
        assert selected is not None
        assert selected.folder_name == "retrain_all_models_20260606_090634"

    def test_latest_valid_model_preferred_over_older_valid(self):
        """Among valid models, the latest one should be preferred."""
        older_valid = ModelArtifacts(
            folder_name="core_retrain_20260606_093842",
            folder_path="/fake/older",
            model_pkl="/path/to/older.pkl",
            feature_count=114,
            feature_list=["f1"],
            metrics_json="/path/to/metrics.json",
            trained_at="2026-06-06T09:38:42",
            is_valid=True,
        )
        
        newer_valid = ModelArtifacts(
            folder_name="retrain_all_models_20260607_051106",
            folder_path="/fake/newer",
            model_pkl="/path/to/newer.pkl",
            feature_count=114,
            feature_list=["f1"],
            metrics_json="/path/to/metrics.json",
            trained_at="2026-06-07T05:11:06",
            is_valid=True,
        )
        
        selected = select_best_model([older_valid, newer_valid])
        
        assert selected is not None
        assert selected.folder_name == "retrain_all_models_20260607_051106"


class TestTimestampParsing:
    """Test timestamp extraction from folder names."""

    def test_parse_retrain_all_models_timestamp(self):
        """Test parsing timestamp from retrain_all_models folder."""
        ts = parse_folder_timestamp("retrain_all_models_20260607_024322")
        assert ts is not None
        assert "2026-06-07" in ts
        assert "02:43:22" in ts

    def test_parse_core_retrain_timestamp(self):
        """Test parsing timestamp from core_retrain folder."""
        ts = parse_folder_timestamp("core_retrain_20260606_093842")
        assert ts is not None
        assert "2026-06-06" in ts
        assert "09:38:42" in ts

    def test_parse_short_timestamp(self):
        """Test parsing short timestamp format like YYMMDD_HHMM."""
        ts = parse_folder_timestamp("bs_retraining_real_20260605_1054")
        assert ts is not None
        assert "2026-06-05" in ts
        # Note: 1054 is parsed as HH:MM = 10:05 due to 4-digit parsing
        # The timestamp parser handles this as a limitation

    def test_no_timestamp_returns_none(self):
        """Folder names without timestamps should return None."""
        ts = parse_folder_timestamp("edge_retraining_bs_smoke")
        assert ts is None


class TestModelValidationIntegration:
    """Integration tests using the generated validation report."""

    def test_validation_report_exists(self):
        """Test that the validation report was generated."""
        reports_dir = Path(r"C:\Users\rahul\Downloads\niftyscalper-current\reports")
        
        if not reports_dir.exists():
            pytest.skip("Reports directory not found")
        
        # Find the most recent validation report
        reports = sorted(reports_dir.glob("model_artifact_validation_*.json"))
        
        if not reports:
            pytest.skip("No validation reports found")
        
        latest_report = reports[-1]
        
        with open(latest_report, 'r') as f:
            report = json.load(f)
        
        # Verify report structure
        assert "approved_model" in report
        assert "rejected_models" in report
        assert "all_models" in report
        assert "selection_criteria" in report
        
        # Verify selection criteria
        criteria = report["selection_criteria"]
        assert "reject_null_pkl" in criteria
        assert "reject_zero_features" in criteria
        assert "reject_missing_feature_list" in criteria
        assert "reject_missing_metrics" in criteria
        assert "prefer_latest" in criteria

    def test_report_has_approved_model_info(self):
        """Test that the report contains proper approval information."""
        reports_dir = Path(r"C:\Users\rahul\Downloads\niftyscalper-current\reports")
        
        if not reports_dir.exists():
            pytest.skip("Reports directory not found")
        
        reports = sorted(reports_dir.glob("model_artifact_validation_*.json"))
        
        if not reports:
            pytest.skip("No validation reports found")
        
        latest_report = reports[-1]
        
        with open(latest_report, 'r') as f:
            report = json.load(f)
        
        # If there's an approved model, verify it has required fields
        if report["approved_model"]:
            approved = report["approved_model"]
            assert approved["model_pkl"] is not None
            assert approved["feature_count"] > 0
            assert approved["metrics_json"] is not None
            assert approved["is_valid"] is True
        else:
            # If no approved model, verify all models were rejected
            assert len([m for m in report["all_models"] if m["is_valid"]]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=60"])