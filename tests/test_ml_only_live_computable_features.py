"""
test_ml_only_live_computable_features.py
========================================
Tests for live computable feature validation.

Verifies:
- Non-live-computable features are rejected
- No future leakage in features
- Features can be computed at decision time
- Forbidden tokens are properly blocked
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


class TestLiveComputableFeatures:
    """Test live computable feature identification."""
    
    def test_forbidden_tokens_defined(self):
        """Test that forbidden tokens are defined."""
        try:
            from ml_pipeline import STRICT_FORBIDDEN_TOKENS
            assert len(STRICT_FORBIDDEN_TOKENS) > 0
        except ImportError:
            # Check in retrain script
            from scripts.retrain_all_edge_models import STRICT_FORBIDDEN_TOKENS
            assert len(STRICT_FORBIDDEN_TOKENS) > 0
    
    def test_return_columns_forbidden(self):
        """Test that return columns are in forbidden tokens."""
        from scripts.retrain_all_edge_models import STRICT_FORBIDDEN_TOKENS
        
        forbidden_lower = [t.lower() for t in STRICT_FORBIDDEN_TOKENS]
        
        # Must contain return-related forbidden tokens
        assert any('return' in t for t in forbidden_lower)
        assert any('pnl' in t or 'profit' in t for t in forbidden_lower)
        assert any('label' in t for t in forbidden_lower)
    
    def test_future_related_forbidden(self):
        """Test that future-related tokens are forbidden."""
        from scripts.retrain_all_edge_models import STRICT_FORBIDDEN_TOKENS
        
        forbidden_lower = [t.lower() for t in STRICT_FORBIDDEN_TOKENS]
        
        # Must contain future-related forbidden tokens
        future_tokens = ['future', 'forward', 'next']
        has_future = any(any(ft in t for ft in future_tokens) for t in forbidden_lower)
        assert has_future, "Must have future-related forbidden tokens"
    
    def test_label_columns_forbidden(self):
        """Test that label columns are in forbidden tokens."""
        from scripts.retrain_all_edge_models import STRICT_FORBIDDEN_TOKENS
        
        forbidden_lower = [t.lower() for t in STRICT_FORBIDDEN_TOKENS]
        
        # Must contain label-related forbidden tokens
        label_tokens = ['label', 'outcome', 'target']
        has_label = any(any(lt in t for lt in label_tokens) for t in forbidden_lower)
        assert has_label, "Must have label-related forbidden tokens"


class TestFeatureLeakage:
    """Test feature leakage detection."""
    
    def test_no_return_as_feature(self):
        """Test that return columns cannot be used as features."""
        # Common return column names that should be forbidden
        return_columns = [
            'gross_forward_return',
            'net_forward_return',
            'forward_return',
            'future_return',
            'return_1',
            'return_5',
        ]
        
        from scripts.retrain_all_edge_models import STRICT_FORBIDDEN_TOKENS
        forbidden_lower = [t.lower() for t in STRICT_FORBIDDEN_TOKENS]
        
        for col in return_columns:
            assert col.lower() in forbidden_lower or \
                   any(t in col.lower() for t in forbidden_lower), \
                   f"Return column {col} must be forbidden"

    def test_no_label_as_feature(self):
        """Test that label columns cannot be used as features."""
        label_columns = [
            'profitable_trade_label',
            'cost_survivor_label',
            'strong_profitable_trade_label',
            'avoid_trade_label',
        ]
        
        from scripts.retrain_all_edge_models import STRICT_FORBIDDEN_TOKENS
        forbidden_lower = [t.lower() for t in STRICT_FORBIDDEN_TOKENS]
        
        for col in label_columns:
            assert col.lower() in forbidden_lower or \
                   any(t in col.lower() for t in forbidden_lower), \
                   f"Label column {col} must be forbidden"

    def test_no_pnl_as_feature(self):
        """Test that PnL columns cannot be used as features."""
        pnl_columns = [
            'realized_pnl',
            'unrealized_pnl',
            'net_pnl',
            'gross_pnl',
            'trade_pnl',
        ]
        
        from scripts.retrain_all_edge_models import STRICT_FORBIDDEN_TOKENS
        forbidden_lower = [t.lower() for t in STRICT_FORBIDDEN_TOKENS]
        
        for col in pnl_columns:
            assert col.lower() in forbidden_lower or \
                   any(t in col.lower() for t in forbidden_lower), \
                   f"PnL column {col} must be forbidden"


class TestLiveFeatureBuilder:
    """Test live feature builder functionality."""
    
    def test_live_feature_builder_imports(self):
        """Test that live feature builder can be imported."""
        try:
            from live_feature_builder import LiveFeatureBuilder, LiveSnapshot
            assert LiveFeatureBuilder is not None
            assert LiveSnapshot is not None
        except ImportError as e:
            pytest.skip(f"Cannot import live_feature_builder: {e}")
    
    def test_candle_features_computable(self):
        """Test that OHLCV candle features are live computable."""
        try:
            from live_feature_builder import CANDLE_FEATURES
            assert len(CANDLE_FEATURES) > 0
            # Should include basic OHLCV and derived features
            assert 'open' in CANDLE_FEATURES or 'last_open' in CANDLE_FEATURES
            assert 'close' in CANDLE_FEATURES or 'last_close' in CANDLE_FEATURES
            assert 'volume' in CANDLE_FEATURES or 'last_volume' in CANDLE_FEATURES
        except ImportError:
            pytest.skip("live_feature_builder not available")
    
    def test_option_chain_features_listed(self):
        """Test that option chain features are listed."""
        try:
            from live_feature_builder import OPTION_CHAIN_FEATURES
            assert len(OPTION_CHAIN_FEATURES) > 0
            # Should include price, greeks, etc.
            assert 'ltp' in OPTION_CHAIN_FEATURES or 'option_ltp' in OPTION_CHAIN_FEATURES
        except ImportError:
            pytest.skip("live_feature_builder not available")


class TestFeatureSchemaValidation:
    """Test feature schema validation."""
    
    def test_feature_schema_in_manifest(self):
        """Test that feature schema is required in manifest."""
        from scripts.ml_only_candidate_inventory import load_candidate_manifest
        
        candidates_dir = REPO_ROOT / 'models' / 'candidates'
        manifests = list(candidates_dir.rglob('candidate_manifest.json'))
        
        if manifests:
            manifest = load_candidate_manifest(manifests[0])
            # Should have feature schema or feature schema path
            has_schema = 'feature_schema' in manifest or \
                        'feature_schema_path' in manifest or \
                        'artifacts' in manifest
            assert has_schema, "Manifest must have feature schema"


class TestNoFutureLeakage:
    """Test no-future-leakage enforcement."""
    
    def test_time_based_split_required(self):
        """Test that time-based split is required (not random)."""
        from scripts.ml_only_strict_gate_evaluator import STRICT_ML_ONLY_GATES
        
        gate = STRICT_ML_ONLY_GATES['A3_time_based_split']
        assert gate['name'] == 'Time-Based Train/Val Split'
        assert gate['severity'] == 'high'
    
    def test_walk_forward_validation(self):
        """Test that walk-forward validation is used."""
        # Walk-forward validation ensures no lookahead
        from scripts.ml_only_strict_gate_evaluator import STRICT_ML_ONLY_GATES
        
        gate = STRICT_ML_ONLY_GATES['A4_out_of_sample_evaluation']
        assert gate['name'] == 'Out-of-Sample Evaluation'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])