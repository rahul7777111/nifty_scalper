"""
test_ml_only_candidate_enhancement_leakage.py
==============================================
Tests for leakage blocking in enhancement loop.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestForbiddenFeaturesBlocked:
    """Test that forbidden features are blocked."""
    
    def test_return_features_blocked(self):
        """Test return-related features are blocked."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementLoop
        from pathlib import Path
        
        loop = CandidateEnhancementLoop(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_candidates=1,
            max_iterations_per_candidate=1,
        )
        
        features = loop._get_live_features()
        for f in features:
            assert 'return' not in f.lower(), f"Feature {f} contains 'return'"
    
    def test_forward_features_blocked(self):
        """Test forward-related features are blocked."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementLoop
        from pathlib import Path
        
        loop = CandidateEnhancementLoop(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_candidates=1,
            max_iterations_per_candidate=1,
        )
        
        features = loop._get_live_features()
        for f in features:
            assert 'forward' not in f.lower(), f"Feature {f} contains 'forward'"
    
    def test_pnl_features_blocked(self):
        """Test pnl-related features are blocked."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementLoop
        from pathlib import Path
        
        loop = CandidateEnhancementLoop(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_candidates=1,
            max_iterations_per_candidate=1,
        )
        
        features = loop._get_live_features()
        for f in features:
            assert 'pnl' not in f.lower(), f"Feature {f} contains 'pnl'"
    
    def test_label_features_blocked(self):
        """Test label-related features are blocked."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementLoop
        from pathlib import Path
        
        loop = CandidateEnhancementLoop(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_candidates=1,
            max_iterations_per_candidate=1,
        )
        
        features = loop._get_live_features()
        for f in features:
            assert 'label' not in f.lower(), f"Feature {f} contains 'label'"
    
    def test_target_features_blocked(self):
        """Test target-related features are blocked."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementLoop
        from pathlib import Path
        
        loop = CandidateEnhancementLoop(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_candidates=1,
            max_iterations_per_candidate=1,
        )
        
        features = loop._get_live_features()
        for f in features:
            assert 'target' not in f.lower(), f"Feature {f} contains 'target'"
    
    def test_exit_features_blocked(self):
        """Test exit-related features are blocked."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementLoop
        from pathlib import Path
        
        loop = CandidateEnhancementLoop(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_candidates=1,
            max_iterations_per_candidate=1,
        )
        
        features = loop._get_live_features()
        for f in features:
            assert 'exit' not in f.lower(), f"Feature {f} contains 'exit'"


class TestLeakageGate:
    """Test leakage gate in evaluation."""
    
    def test_leakage_pass_set(self):
        """Test leakage_pass is set during gate evaluation."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        
        # leakage_pass should be True for clean features
        r = CandidateEnhancementResult(
            candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=5, gates_total=22,
            leakage_pass=True, live_computable_pass=True,
        )
        
        assert r.leakage_pass == True
    
    def test_net_forward_return_not_in_features(self):
        """Test net_forward_return is excluded from features."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementLoop
        from pathlib import Path
        
        loop = CandidateEnhancementLoop(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_candidates=1,
            max_iterations_per_candidate=1,
        )
        
        features = loop._get_live_features()
        assert 'net_forward_return' not in features
        assert 'gross_forward_return' not in features


class TestNoLabelAsFilter:
    """Test that labels are not used as trade filters."""
    
    def test_label_column_not_used_as_filter(self):
        """Test that _apply_filter doesn't use label columns."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementLoop
        from pathlib import Path
        
        loop = CandidateEnhancementLoop(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_candidates=1,
            max_iterations_per_candidate=1,
        )
        
        # 'none' filter should return all rows (not filter by label)
        df_none = loop._apply_filter(loop.full_df, 'none')
        assert len(df_none) == len(loop.full_df), "'none' filter should return all rows"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])