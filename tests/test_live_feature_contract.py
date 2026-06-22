"""tests/test_live_feature_contract.py
====================================
Validation tests for the live feature contract.

These tests verify that:
1. The feature contract correctly identifies required features
2. Live features from 100 candles meet coverage >= 95%
3. No missing candle-derived features that are truly available from OHLCV data
4. The contract is aligned with what build_market_feature_vector() produces

Run with: python -m pytest tests/test_live_feature_contract.py -v
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

# -- paths --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from src.ml_feature_contract import (
    CONTRACT_VERSION,
    CORE_REQUIRED_FEATURES,
    REQUIRED_CANDLE_FEATURES,
    REQUIRED_FEATURE_SCHEMA,
    ALLOWED_LIVE_FEATURES,
    compute_feature_coverage,
    validate_live_features,
    is_feature_forbidden,
    get_feature_contract_info,
)
from src.ml_pipeline import EXCLUDED_MODEL_FEATURES


class TestLiveFeatureContractBasics:
    """Basic validation that the contract is well-formed."""

    def test_contract_version_is_defined(self):
        """Contract version must be defined and non-empty."""
        assert CONTRACT_VERSION is not None
        assert isinstance(CONTRACT_VERSION, str)
        assert len(CONTRACT_VERSION) > 0

    def test_core_required_features_not_empty(self):
        """CORE_REQUIRED_FEATURES must contain features."""
        assert len(CORE_REQUIRED_FEATURES) > 0

    def test_required_candle_features_not_empty(self):
        """REQUIRED_CANDLE_FEATURES must contain features."""
        assert len(REQUIRED_CANDLE_FEATURES) > 0

    def test_allowed_live_features_superset_of_required(self):
        """ALLOWED_LIVE_FEATURES must be a superset of CORE_REQUIRED_FEATURES."""
        for feature in CORE_REQUIRED_FEATURES:
            assert feature in ALLOWED_LIVE_FEATURES, (
                f"Feature {feature} is in CORE_REQUIRED_FEATURES but not in ALLOWED_LIVE_FEATURES"
            )


class TestFeatureContractExclusions:
    """Verify that EXCLUDED_MODEL_FEATURES are not in REQUIRED_CANDLE_FEATURES."""

    def test_excluded_features_not_in_required_candle(self):
        """Features in EXCLUDED_MODEL_FEATURES should not be in REQUIRED_CANDLE_FEATURES.
        
        This ensures the contract is aligned with what build_market_feature_vector() actually produces.
        If a feature is required but excluded by the pipeline, validation would always fail.
        """
        excluded_candle = REQUIRED_CANDLE_FEATURES & EXCLUDED_MODEL_FEATURES
        assert len(excluded_candle) == 0, (
            f"The following features are in both REQUIRED_CANDLE_FEATURES and "
            f"EXCLUDED_MODEL_FEATURES (ml_pipeline.py): {excluded_candle}. "
            f"This causes validation to fail because the contract requires features "
            f"that the pipeline explicitly excludes from output."
        )

    def test_excluded_features_not_in_core_required(self):
        """Features in EXCLUDED_MODEL_FEATURES should not be in CORE_REQUIRED_FEATURES."""
        excluded_core = CORE_REQUIRED_FEATURES & EXCLUDED_MODEL_FEATURES
        assert len(excluded_core) == 0, (
            f"The following features are in both CORE_REQUIRED_FEATURES and "
            f"EXCLUDED_MODEL_FEATURES: {excluded_core}"
        )


class TestLiveFeatureCoverage100Candles:
    """Test coverage with 100 candles worth of live features.
    
    This simulates a live feature dict with all features that would be
    available from 100 candles of OHLCV data.
    """

    @pytest.fixture
    def live_feature_dict_100_candles(self) -> Dict[str, float]:
        """Create a feature dict simulating 100 candles of live data.
        
        This includes all features that build_market_feature_vector() and
        live_feature_builder.py would produce from 100 candles.
        """
        features: Dict[str, float] = {}
        
        # Last candle OHLCV (the prefixed versions)
        features["last_open"] = 24650.0
        features["last_high"] = 24720.0
        features["last_low"] = 24580.0
        features["last_close"] = 24690.0
        features["last_volume"] = 1250000.0
        
        # Candlestick structure features
        features["body_pct"] = 0.0016
        features["range_pct"] = 0.0057
        features["gap_pct"] = 0.0
        features["upper_wick_pct"] = 0.3
        features["lower_wick_pct"] = 0.5
        features["close_location_pct"] = 0.7
        
        # Candlestick patterns
        features["bullish_engulfing"] = 0.0
        features["bearish_engulfing"] = 0.0
        features["doji"] = 0.0
        features["hammer"] = 0.0
        features["shooting_star"] = 0.0
        
        # Return features (ret_1 is EXCLUDED by pipeline, not included)
        features["ret_3"] = 0.0025
        features["ret_5"] = 0.0041
        features["ret_10"] = 0.0082
        features["ret_mean"] = 0.0021
        features["ret_std"] = 0.0035
        features["ret_min"] = -0.005
        features["ret_max"] = 0.012
        
        # Volume stats
        features["vol_mean"] = 1100000.0
        features["vol_std"] = 180000.0
        features["vol_min"] = 850000.0
        features["vol_max"] = 1450000.0
        
        # Technical indicators
        features["ema_fast"] = 24675.0
        features["ema_slow"] = 24620.0
        features["ema_diff_pct"] = 0.0022
        features["rsi_14"] = 55.3
        features["atr_14"] = 125.5
        features["atr_pct"] = 0.0051
        features["adx_14"] = 28.5
        # roc_14 is EXCLUDED by pipeline
        features["choppiness_14"] = 45.2
        features["supertrend_dir"] = 1.0
        # supertrend_gap_pct is EXCLUDED by pipeline
        
        # Pivot levels
        features["pivot_pp_dist_pct"] = 0.001
        features["pivot_r1_dist_pct"] = 0.008
        features["pivot_s1_dist_pct"] = -0.006
        
        # Price action
        features["close_vs_open_pct"] = 0.0016
        features["range_to_atr"] = 1.1
        features["momentum_lookback_pct"] = 0.015
        
        # Regime features
        features["regime_trending"] = 0.0
        features["regime_volatile"] = 0.0
        features["regime_mean_reverting"] = 1.0
        features["regime_quiet"] = 0.0
        
        return features

    def test_coverage_with_100_candles_above_95_percent(
        self, live_feature_dict_100_candles: Dict[str, float]
    ):
        """Coverage must be >= 95% with 100 candles of live features."""
        result = compute_feature_coverage(
            live_feature_dict_100_candles,
            required_features=CORE_REQUIRED_FEATURES,
            check_forbidden=True,
        )
        
        assert result.coverage_pct >= 95.0, (
            f"Feature coverage {result.coverage_pct}% is below 95% threshold. "
            f"Missing features: {result.missing_features[:10]}"
        )

    def test_validation_passes_with_100_candles(
        self, live_feature_dict_100_candles: Dict[str, float]
    ):
        """Live validation should pass with complete 100 candles of features."""
        is_valid, errors = validate_live_features(
            live_feature_dict_100_candles,
            required_features=CORE_REQUIRED_FEATURES,
            min_coverage_pct=95.0,
        )
        
        assert is_valid is True, (
            f"Validation failed with complete 100 candles features: {errors}"
        )

    def test_no_missing_candle_features_from_live_data(
        self, live_feature_dict_100_candles: Dict[str, float]
    ):
        """Verify no missing candle-derived features that should be available.
        
        Features like ret_1, range_pct, roc_14, supertrend_gap_pct are EXCLUDED
        by the pipeline and should NOT be considered missing if they're not in the
        live feature dict (since the pipeline won't produce them).
        """
        result = compute_feature_coverage(
            live_feature_dict_100_candles,
            required_features=CORE_REQUIRED_FEATURES,
        )
        
        # Check that missing features are not candle-derived features that should be available
        # (i.e., not in EXCLUDED_MODEL_FEATURES)
        truly_missing = [
            f for f in result.missing_features 
            if f not in EXCLUDED_MODEL_FEATURES
        ]
        
        assert len(truly_missing) == 0, (
            f"Missing features that should be available from live OHLCV data: "
            f"{truly_missing}. Missing (excluded by pipeline): "
            f"{[f for f in result.missing_features if f in EXCLUDED_MODEL_FEATURES]}"
        )


class TestFeatureContractEdgeCases:
    """Test edge cases in feature validation."""

    def test_validation_fails_when_required_feature_missing(self):
        """Validation must fail if a required feature is truly missing."""
        # Create a dict with only half the required features
        half_features = {f: 1.0 for f in list(CORE_REQUIRED_FEATURES)[:len(CORE_REQUIRED_FEATURES)//2]}
        
        is_valid, errors = validate_live_features(
            half_features,
            required_features=CORE_REQUIRED_FEATURES,
            min_coverage_pct=95.0,
        )
        
        assert is_valid is False
        assert any("MISSING_REQUIRED_FEATURES" in e for e in errors)

    def test_validation_fails_when_coverage_below_threshold(self):
        """Validation must fail when coverage is below threshold."""
        # Create a dict with 80% of required features
        features_80pct = {f: 1.0 for f in list(CORE_REQUIRED_FEATURES)[:int(len(CORE_REQUIRED_FEATURES)*0.8)]}
        
        is_valid, errors = validate_live_features(
            features_80pct,
            required_features=CORE_REQUIRED_FEATURES,
            min_coverage_pct=95.0,
        )
        
        assert is_valid is False
        assert any("COVERAGE_BELOW_THRESHOLD" in e for e in errors)

    def test_validation_fails_when_forbidden_feature_present(self):
        """Validation must fail if a forbidden feature is detected."""
        features = {f: 1.0 for f in CORE_REQUIRED_FEATURES}
        features["net_forward_return"] = 0.05  # Forbidden!
        
        is_valid, errors = validate_live_features(
            features,
            required_features=CORE_REQUIRED_FEATURES,
        )
        
        assert is_valid is False
        assert any("FORBIDDEN_FEATURES_DETECTED" in e for e in errors)

    def test_forbidden_features_are_not_in_allowed(self):
        """Forbidden features must not be in ALLOWED_LIVE_FEATURES."""
        # These are definitely forbidden
        forbidden = ["future_close", "net_forward_return", "trade_pnl_label"]
        
        for feature in forbidden:
            if feature in ALLOWED_LIVE_FEATURES:
                pytest.fail(f"Forbidden feature {feature} is in ALLOWED_LIVE_FEATURES")


class TestContractAlignmentWithPipeline:
    """Verify the contract is aligned with what the pipeline produces."""

    def test_all_required_candle_features_in_allowed(self):
        """All features in REQUIRED_CANDLE_FEATURES must be in ALLOWED_LIVE_FEATURES."""
        for feature in REQUIRED_CANDLE_FEATURES:
            assert feature in ALLOWED_LIVE_FEATURES, (
                f"Feature {feature} is in REQUIRED_CANDLE_FEATURES but not in ALLOWED_LIVE_FEATURES"
            )

    def test_required_feature_schema_matches_candle_features(self):
        """REQUIRED_FEATURE_SCHEMA must include all REQUIRED_CANDLE_FEATURES."""
        for feature in REQUIRED_CANDLE_FEATURES:
            assert feature in REQUIRED_FEATURE_SCHEMA, (
                f"Feature {feature} is in REQUIRED_CANDLE_FEATURES but not in REQUIRED_FEATURE_SCHEMA"
            )

    def test_contract_info_contains_necessary_fields(self):
        """get_feature_contract_info() must return required fields."""
        info = get_feature_contract_info()
        
        required_fields = [
            "contract_version",
            "contract_epoch", 
            "total_allowed_features",
            "total_required_features",
            "core_required_features",
        ]
        
        for field in required_fields:
            assert field in info, f"Missing field {field} in contract info"


class TestNoFalsePositivesOnLiveFeatures:
    """Verify that live-computable features are not falsely flagged as missing."""

    def test_ohlcv_last_prefixed_are_available(self):
        """last_open, last_high, last_low, last_close, last_volume must be in CORE_REQUIRED."""
        required_ohlcv = {"last_open", "last_high", "last_low", "last_close", "last_volume"}
        for feature in required_ohlcv:
            assert feature in CORE_REQUIRED_FEATURES, (
                f"{feature} is a required OHLCV feature but not in CORE_REQUIRED_FEATURES"
            )

    def test_common_technical_indicators_are_available(self):
        """Common technical indicators must be in CORE_REQUIRED."""
        common_indicators = {"rsi_14", "atr_14", "adx_14", "ema_fast", "ema_slow"}
        for feature in common_indicators:
            assert feature in CORE_REQUIRED_FEATURES, (
                f"{feature} is a common technical indicator but not in CORE_REQUIRED_FEATURES"
            )

    def test_regime_features_are_available(self):
        """Regime features must be in CORE_REQUIRED."""
        regime_features = {"regime_trending", "regime_volatile", "regime_mean_reverting", "regime_quiet"}
        for feature in regime_features:
            assert feature in CORE_REQUIRED_FEATURES, (
                f"{feature} is a regime feature but not in CORE_REQUIRED_FEATURES"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])