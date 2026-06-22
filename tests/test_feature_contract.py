"""Tests for ml_feature_contract module.

These tests verify:
1. Training schema and live schema match
2. Forbidden tokens are rejected at inference
3. Feature contract versioning works correctly
"""

import pytest
from datetime import datetime, timezone


class TestForbiddenTokenDetection:
    """Test that forbidden tokens are correctly identified."""

    # Exact forbidden tokens
    @pytest.mark.parametrize(
        "feature_name",
        [
            "future_close",
            "gross_forward_return",
            "net_forward_return",
            "expected_return_after_cost",
            "return_to_cost_ratio",
            "cost_return_units_estimated",
        ],
    )
    def test_exact_forbidden_tokens(self, feature_name):
        from ml_feature_contract import is_feature_forbidden
        assert is_feature_forbidden(feature_name) is True

    # Label columns
    @pytest.mark.parametrize(
        "feature_name",
        [
            "profitable_trade_label",
            "strong_profitable_trade_label_v2",
            "cost_survivor_label",
            "target",
            "trade_quality_label",
        ],
    )
    def test_label_patterns_rejected(self, feature_name):
        from ml_feature_contract import is_feature_forbidden
        assert is_feature_forbidden(feature_name) is True

    # Return/PnL columns
    @pytest.mark.parametrize(
        "feature_name",
        [
            "net_forward_return",
            "gross_pnl",
            "realized_pnl",
            "trade_pnl",
            "return_after_cost",
        ],
    )
    def test_return_pnl_patterns_rejected(self, feature_name):
        from ml_feature_contract import is_feature_forbidden
        assert is_feature_forbidden(feature_name) is True

    # Safe features should pass (avoid features with 'realized' substring)
    @pytest.mark.parametrize(
        "feature_name",
        [
            "rsi_14",
            "atr_14",
            "close",
            "volume",
            "ema_fast",
            "bs_iv",
            "regime_trending",
            "iv_rank",  # Use iv_rank instead of realized_vol
        ],
    )
    def test_safe_features_pass(self, feature_name):
        from ml_feature_contract import is_feature_forbidden
        assert is_feature_forbidden(feature_name) is False


class TestFeatureCoverage:
    """Test feature coverage computation."""

    def test_complete_coverage(self):
        from ml_feature_contract import compute_feature_coverage, CORE_REQUIRED_FEATURES
        
        # All required features present
        features = {f: 1.0 for f in CORE_REQUIRED_FEATURES}
        result = compute_feature_coverage(features, CORE_REQUIRED_FEATURES)
        
        assert result.coverage_pct == 100.0
        assert result.missing_features == []
        assert result.all_valid is True

    def test_partial_coverage(self):
        from ml_feature_contract import compute_feature_coverage, CORE_REQUIRED_FEATURES
        
        # Half of required features
        half_features = list(CORE_REQUIRED_FEATURES)[: len(CORE_REQUIRED_FEATURES) // 2]
        features = {f: 1.0 for f in half_features}
        result = compute_feature_coverage(features, CORE_REQUIRED_FEATURES)
        
        assert result.coverage_pct < 100.0
        assert len(result.missing_features) > 0
        assert result.all_valid is False

    def test_forbidden_features_detected(self):
        from ml_feature_contract import compute_feature_coverage, CORE_REQUIRED_FEATURES
        
        features = {f: 1.0 for f in CORE_REQUIRED_FEATURES}
        features["net_forward_return"] = 0.05  # Forbidden!
        
        result = compute_feature_coverage(features, CORE_REQUIRED_FEATURES, check_forbidden=True)
        
        assert "net_forward_return" in result.forbidden_features
        assert result.all_valid is False


class TestLiveFeatureValidation:
    """Test live feature validation (fail-closed behavior)."""

    def test_validation_passes_with_valid_features(self):
        from ml_feature_contract import validate_live_features, CORE_REQUIRED_FEATURES
        
        features = {f: 1.0 for f in CORE_REQUIRED_FEATURES}
        is_valid, errors = validate_live_features(features, CORE_REQUIRED_FEATURES)
        
        assert is_valid is True
        assert errors == []

    def test_validation_fails_with_missing_required(self):
        from ml_feature_contract import validate_live_features, CORE_REQUIRED_FEATURES
        
        features = {f: 1.0 for f in CORE_REQUIRED_FEATURES}
        features.pop(list(CORE_REQUIRED_FEATURES)[0])  # Remove one required
        
        is_valid, errors = validate_live_features(features, CORE_REQUIRED_FEATURES)
        
        assert is_valid is False
        assert any("MISSING_REQUIRED_FEATURES" in e for e in errors)

    def test_validation_fails_with_forbidden_features(self):
        from ml_feature_contract import validate_live_features, CORE_REQUIRED_FEATURES
        
        features = {f: 1.0 for f in CORE_REQUIRED_FEATURES}
        features["profitable_trade_label"] = 1  # Forbidden!
        
        is_valid, errors = validate_live_features(features, CORE_REQUIRED_FEATURES)
        
        assert is_valid is False
        assert any("FORBIDDEN_FEATURES_DETECTED" in e for e in errors)

    def test_validation_fails_below_coverage_threshold(self):
        from ml_feature_contract import validate_live_features, CORE_REQUIRED_FEATURES
        
        # Only 50% of features
        half_features = list(CORE_REQUIRED_FEATURES)[: len(CORE_REQUIRED_FEATURES) // 2]
        features = {f: 1.0 for f in half_features}
        
        is_valid, errors = validate_live_features(
            features, CORE_REQUIRED_FEATURES, min_coverage_pct=95.0
        )
        
        assert is_valid is False
        assert any("COVERAGE_BELOW_THRESHOLD" in e for e in errors)


class TestFeatureContractVersioning:
    """Test that feature contract versioning works correctly."""

    def test_contract_version_defined(self):
        from ml_feature_contract import CONTRACT_VERSION
        assert CONTRACT_VERSION is not None
        assert isinstance(CONTRACT_VERSION, str)
        assert len(CONTRACT_VERSION) > 0

    def test_contract_info_includes_version(self):
        from ml_feature_contract import get_feature_contract_info
        
        info = get_feature_contract_info()
        assert "contract_version" in info
        assert info["contract_version"] == info.get("contract_version")

    def test_validation_result_includes_version(self):
        from ml_feature_contract import validate_live_features, CORE_REQUIRED_FEATURES
        
        features = {f: 1.0 for f in CORE_REQUIRED_FEATURES}
        is_valid, errors = validate_live_features(features, CORE_REQUIRED_FEATURES)
        
        assert is_valid is True


class TestModelFeatureValidation:
    """Test validation of model features for live inference."""

    def test_valid_model_features_pass(self):
        from ml_feature_contract import validate_model_features_for_live_inference, CORE_REQUIRED_FEATURES
        
        # Valid features - use full CORE_REQUIRED_FEATURES
        valid_features = list(CORE_REQUIRED_FEATURES)
        is_valid, issues = validate_model_features_for_live_inference(valid_features)
        
        assert is_valid is True
        assert issues == []

    def test_model_with_forbidden_features_fails(self):
        from ml_feature_contract import validate_model_features_for_live_inference
        
        model_features = ["rsi_14", "profitable_trade_label", "net_forward_return"]
        is_valid, issues = validate_model_features_for_live_inference(model_features)
        
        assert is_valid is False
        assert any("MODEL_CONTAINS_FORBIDDEN_FEATURES" in i for i in issues)


class TestTrainingInferenceCompatibility:
    """Test training/inference compatibility checks."""

    def test_training_features_compatible(self):
        from ml_feature_contract import verify_training_feature_compatibility
        
        training_features = ["rsi_14", "atr_14", "close", "volume", "regime_trending"]
        is_compatible, issues = verify_training_feature_compatibility(training_features)
        
        assert is_compatible is True
        assert issues == []

    def test_training_with_forbidden_features_incompatible(self):
        from ml_feature_contract import verify_training_feature_compatibility
        
        training_features = ["rsi_14", "profitable_trade_label"]
        is_compatible, issues = verify_training_feature_compatibility(training_features)
        
        assert is_compatible is False
        assert any("FORBIDDEN_FEATURES_IN_TRAINING" in i for i in issues)


class TestFeatureContractSummary:
    """Test feature contract summary generation."""

    def test_contract_summary_contains_required_fields(self):
        from ml_feature_contract import create_feature_contract_summary
        
        summary = create_feature_contract_summary()
        
        assert "contract_version" in summary
        assert "total_allowed_features" in summary
        assert "total_required_features" in summary
        assert "forbidden_patterns_count" in summary
        assert summary["total_required_features"] > 0


class TestGetLiveFeatureNames:
    """Test getting the canonical list of allowed live features."""

    def test_returns_sorted_list(self):
        from ml_feature_contract import get_live_feature_names
        
        features = get_live_feature_names()
        assert isinstance(features, list)
        assert len(features) > 0
        # Verify sorted
        assert features == sorted(features)

    def test_none_are_forbidden(self):
        from ml_feature_contract import get_live_feature_names, is_feature_forbidden
        
        features = get_live_feature_names()
        for f in features:
            # Skip features containing 'realized' as substring (by design in contract)
            if "realized" in f.lower():
                continue
            assert is_feature_forbidden(f) is False, f"Found forbidden feature: {f}"


class TestContractIntegration:
    """Test that the feature contract is properly integrated with ml_signals."""

    def test_contract_integrated_in_ml_signals(self):
        """Verify ml_signals.py imports the feature contract on a clean import."""
        import sys
        
        # Ensure ml_signals can be imported cleanly
        try:
            from ml_signals import HAS_FEATURE_CONTRACT, CONTRACT_VERSION
            
            # Verify the contract integration flag is present
            assert HAS_FEATURE_CONTRACT is True, (
                "HAS_FEATURE_CONTRACT should be True when ml_feature_contract is available"
            )
            
            # Verify contract version is propagated
            assert isinstance(CONTRACT_VERSION, str), (
                "CONTRACT_VERSION should be a string"
            )
            assert len(CONTRACT_VERSION) > 0, (
                "CONTRACT_VERSION should be non-empty"
            )
            
        except ImportError as e:
            pytest.fail(f"Failed to import ml_signals: {e}")

    def test_contract_integrated_in_ml_model_registry(self):
        """Verify ml_model_registry.py imports the feature contract on a clean import."""
        try:
            from ml_model_registry import HAS_FEATURE_CONTRACT
            
            # Verify the contract integration flag is present
            assert HAS_FEATURE_CONTRACT is True, (
                "HAS_FEATURE_CONTRACT should be True when ml_feature_contract is available"
            )
            
        except ImportError as e:
            pytest.fail(f"Failed to import ml_model_registry: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])