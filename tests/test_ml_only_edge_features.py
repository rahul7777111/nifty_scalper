"""
test_ml_only_edge_features.py
==============================
Tests for ML-only edge feature system (corrected for new tournament API).

Verifies:
- Live-computable features are properly identified
- Forbidden features are rejected
- Features can be computed at decision time
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestLiveComputableFeatures:
    """Test live-computable feature identification."""
    
    def test_tournament_has_get_live_features_method(self):
        """Test that tournament has _get_live_features method."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path
        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )
        assert hasattr(tournament, '_get_live_features')
    
    def test_live_features_returned(self):
        """Test that live computable features are returned."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path
        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )
        features = tournament._get_live_features()
        assert len(features) > 0


class TestForbiddenFeatures:
    """Test that forbidden features are rejected."""
    
    def _get_features(self):
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path
        t = MLEdgeTournament(
            REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            Path('/tmp/fake'), Path('/tmp'), 1
        )
        return t._get_live_features()
    
    def test_return_features_forbidden(self):
        features = self._get_features()
        forbidden = ['_return', 'forward_return', 'future_return', 'gross_forward', 'net_forward']
        for feat in features:
            for pattern in forbidden:
                assert pattern.lower() not in feat.lower()
    
    def test_label_features_forbidden(self):
        features = self._get_features()
        forbidden = ['_label', 'cost_survivor', 'profitable_trade', 'avoid_trade']
        for feat in features:
            for pattern in forbidden:
                assert pattern.lower() not in feat.lower()
    
    def test_pnl_features_forbidden(self):
        features = self._get_features()
        forbidden = ['pnl', 'profit', 'loss', 'realized_pnl', 'unrealized_pnl']
        for feat in features:
            for pattern in forbidden:
                assert pattern.lower() not in feat.lower()
    
    def test_future_features_forbidden(self):
        features = self._get_features()
        forbidden = ['future_', 'next_', 'forward_']
        for feat in features:
            for pattern in forbidden:
                assert pattern.lower() not in feat.lower()


class TestFeatureSets:
    """Test different feature sets (via hardcoded feature selection)."""
    
    def test_candle_features_exist(self):
        """Test that OHLCV candle features exist."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path
        t = MLEdgeTournament(
            REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            Path('/tmp/fake'), Path('/tmp'), 1
        )
        features = t._get_live_features()
        candle_related = [f for f in features if any(c in f.lower() for c in 
                    ['open', 'high', 'low', 'close', 'ltp', 'volume', 'oi'])]
        assert len(candle_related) > 0


class TestFeatureAvailability:
    """Test that features are available in dataset."""
    
    def test_basic_features_available(self):
        """Test that basic features are available in dataset."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path
        t = MLEdgeTournament(
            REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            Path('/tmp/fake'), Path('/tmp'), 1
        )
        features = t._get_live_features()
        expected = ['ltp', 'volume', 'oi', 'strike_price', 'dte_days']
        for feat in expected:
            assert feat in features, f"Expected feature {feat} not found"
    
    def test_option_type_available(self):
        """Test that option_type is available for filtering."""
        import pandas as pd
        df = pd.read_csv(
            REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            nrows=100
        )
        assert 'option_type' in df.columns
    
    def test_timestamp_available(self):
        """Test that timestamp is available for time-based split."""
        import pandas as pd
        df = pd.read_csv(
            REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            nrows=100
        )
        assert 'timestamp' in df.columns


class TestFeatureCoverage:
    """Test that feature sets provide adequate coverage."""
    
    def test_features_include_price_data(self):
        """Test that features include price data."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path
        t = MLEdgeTournament(
            REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            Path('/tmp/fake'), Path('/tmp'), 1
        )
        features = t._get_live_features()
        price_related = [f for f in features if 'ltp' in f.lower() or 'price' in f.lower() or 'close' in f.lower()]
        assert len(price_related) > 0
    
    def test_features_include_volume_data(self):
        """Test that features include volume data."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path
        t = MLEdgeTournament(
            REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            Path('/tmp/fake'), Path('/tmp'), 1
        )
        features = t._get_live_features()
        vol_related = [f for f in features if 'volume' in f.lower()]
        assert len(vol_related) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])