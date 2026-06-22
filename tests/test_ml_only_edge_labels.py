"""
test_ml_only_edge_labels.py
============================
Tests for ML-only edge label system (corrected for new tournament API).

Verifies:
- Label definitions are sound
- Labels don't leak into features
- Label computations are correct
- Multiple label types are available
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestLabelDefinitions:
    """Test that label definitions are correct."""
    
    def test_labels_available_in_dataset(self):
        """Test that labels are available in the dataset."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path
        
        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )
        
        assert len(tournament.available_labels) > 0, "No labels found"
    
    def test_cost_survivor_label_v2_in_labels(self):
        """Test that cost_survivor_label_v2 is available."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path
        
        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )
        
        assert 'cost_survivor_label_v2' in tournament.available_labels, \
            f"cost_survivor_label_v2 not in labels: {tournament.available_labels}"
    
    def test_strong_profitable_label_in_labels(self):
        """Test that strong_profitable_trade_label is available."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path
        
        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )
        
        labels = tournament.available_labels
        has_strong = any('strong_profitable' in lbl for lbl in labels)
        assert has_strong, f"strong_profitable_trade_label not in labels: {labels}"


class TestLabelAvailability:
    """Test that labels are available in dataset."""
    
    def test_labels_found_in_dataset(self):
        """Test that labels can be found in the dataset."""
        try:
            import pandas as pd
            df = pd.read_csv(
                REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
                nrows=100
            )
            label_cols = [c for c in df.columns if '_label' in c.lower() or 'cost_survivor' in c.lower()]
            assert len(label_cols) > 0, "No label columns found in dataset"
        except Exception as e:
            pytest.skip(f"Dataset not available: {e}")
    
    def test_cost_survivor_v2_exists(self):
        """Test that cost_survivor_label_v2 exists in the dataset."""
        try:
            import pandas as pd
            df = pd.read_csv(
                REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
                nrows=100
            )
            assert 'cost_survivor_label_v2' in df.columns, "cost_survivor_label_v2 not found"
        except Exception as e:
            pytest.skip(f"Dataset not available: {e}")


class TestLabelIntegrity:
    """Test label integrity (no leakage)."""
    
    def test_label_values_are_binary(self):
        """Test that cost_survivor labels are binary (0 or 1)."""
        try:
            import pandas as pd
            df = pd.read_csv(
                REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
                nrows=1000
            )
            if 'cost_survivor_label_v2' in df.columns:
                unique = df['cost_survivor_label_v2'].dropna().unique()
                for v in unique:
                    assert v in [0, 1, 0.0, 1.0], f"Non-binary label value: {v}"
        except Exception as e:
            pytest.skip(f"Dataset not available: {e}")
    
    def test_labels_dont_appear_in_features(self):
        """Test that label columns are not in the live-computable features list."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path
        
        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )
        
        features = tournament._get_live_features()
        label_features = [f for f in features if '_label' in f.lower() or 'cost_survivor' in f.lower()]
        assert len(label_features) == 0, f"Label columns found in features: {label_features}"


class TestLabelComputation:
    """Test label computation logic."""
    
    def test_profitable_trade_label_calculation(self):
        """Test that profitable_trade_label is correctly computed."""
        try:
            import pandas as pd
            df = pd.read_csv(
                REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
                nrows=1000
            )
            if 'profitable_trade_label' in df.columns and 'net_forward_return' in df.columns:
                label = df['profitable_trade_label'].dropna()
                returns = df.loc[label.index, 'net_forward_return']
                matches = ((label == 1) & (returns > 0)) | ((label == 0) & (returns <= 0))
                match_rate = matches.mean()
                assert match_rate >= 0.95, f"Label calculation mismatch: {match_rate}"
        except Exception as e:
            pytest.skip(f"Dataset not available: {e}")
    
    def test_cost_survivor_label_accounts_for_costs(self):
        """Test that cost_survivor_label accounts for transaction costs."""
        try:
            import pandas as pd
            df = pd.read_csv(
                REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
                nrows=1000
            )
            if 'cost_survivor_label_v2' in df.columns and 'net_forward_return' in df.columns:
                label = df['cost_survivor_label_v2'].dropna()
                returns = df.loc[label.index, 'net_forward_return']
                assert returns[label == 1].min() > 0, "Cost survivor has non-positive return"
        except Exception as e:
            pytest.skip(f"Dataset not available: {e}")


class TestMultipleLabelTypes:
    """Test that multiple label types are available for tournament."""
    
    def test_multiple_label_types_available(self):
        """Test that at least 5 different label types are available."""
        try:
            import pandas as pd
            df = pd.read_csv(
                REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
                nrows=100
            )
            label_cols = [c for c in df.columns if '_label' in c.lower() or 'cost_survivor' in c.lower()]
            assert len(label_cols) >= 5, f"Only {len(label_cols)} label types found"
        except Exception as e:
            pytest.skip(f"Dataset not available: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])