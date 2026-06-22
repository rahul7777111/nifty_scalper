"""
test_ml_only_tournament_leakage_block.py
=========================================
Tests that verify leakage patterns are blocked from features.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestLeakageBlocked:
    """Test that all leakage patterns are blocked from features."""

    def test_return_columns_blocked(self):
        """Test that return columns are blocked."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path

        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )

        features = tournament._get_live_features()
        forbidden = ['return', 'forward', 'future', 'pnl', 'label']

        for f in features:
            for pat in forbidden:
                assert pat not in f.lower(), f"Feature {f} contains forbidden '{pat}'"

    def test_label_columns_blocked(self):
        """Test that label columns are blocked."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path

        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )

        features = tournament._get_live_features()
        label_patterns = ['_label', 'cost_survivor', 'profitable_trade', 'avoid_trade']

        for f in features:
            for pat in label_patterns:
                assert pat.lower() not in f.lower(), f"Feature {f} contains label pattern '{pat}'"

    def test_pnl_columns_blocked(self):
        """Test that PnL columns are blocked."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path

        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )

        features = tournament._get_live_features()
        pnl_patterns = ['pnl', 'profit', 'loss', 'realized', 'mfe', 'mae']

        for f in features:
            for pat in pnl_patterns:
                assert pat.lower() not in f.lower(), f"Feature {f} contains PnL pattern '{pat}'"

    def test_exit_outcome_columns_blocked(self):
        """Test that exit/outcome columns are blocked."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path

        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )

        features = tournament._get_live_features()
        exit_patterns = ['exit', 'outcome', 'trade_result']

        for f in features:
            for pat in exit_patterns:
                assert pat.lower() not in f.lower(), f"Feature {f} contains exit pattern '{pat}'"

    def test_forbidden_patterns_list_complete(self):
        """Test that FORBIDDEN_FEATURE_PATTERNS is comprehensive."""
        from scripts.ml_only_edge_tournament import FORBIDDEN_FEATURE_PATTERNS
        
        required_patterns = [
            'return', 'forward', 'future', 'pnl', 'label',
            'target', 'outcome', 'exit', 'mfe', 'mae',
            'cost_survivor', 'realized'
        ]
        
        for pat in required_patterns:
            assert pat in FORBIDDEN_FEATURE_PATTERNS, \
                f"Missing forbidden pattern: {pat}"


class TestLeakageExperimentLevel:
    """Test that experiment-level leakage checks work."""

    def test_experiment_rejects_leaky_features(self):
        """Test that an experiment with leaky features is rejected."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path

        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )

        # Check that _get_live_features doesn't include leaky columns
        features = tournament._get_live_features()
        
        # These must NOT be in features
        must_not_include = [
            'net_forward_return', 'gross_forward_return', 'future_close',
            'profitable_trade_label', 'cost_survivor_label', 'avoid_trade_label',
            'realized_pnl', 'unrealized_pnl'
        ]
        
        for col in must_not_include:
            assert col not in features, f"Leaky column '{col}' found in features"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])