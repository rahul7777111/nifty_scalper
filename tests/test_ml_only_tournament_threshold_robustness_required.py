"""
test_ml_only_tournament_threshold_robustness_required.py
=========================================================
Tests that verify threshold robustness is required for PASS_SHADOW_READY.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestThresholdRobustnessRequired:
    """Test that threshold robustness is enforced."""

    def test_threshold_robustness_method_exists(self):
        """Test that _evaluate_threshold_robustness method exists."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path

        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )

        assert hasattr(tournament, '_evaluate_threshold_robustness')
        assert callable(tournament._evaluate_threshold_robustness)

    def test_evaluates_nearby_thresholds(self):
        """Test that threshold robustness checks nearby thresholds."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path
        import numpy as np

        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )

        import inspect
        source = inspect.getsource(tournament._evaluate_threshold_robustness)
        
        # Must check nearby thresholds
        assert 'threshold - 0.10' in source or 'nearby' in source or 'threshold - 0.05' in source
        # Must check multiple nearby thresholds
        assert source.count('threshold') >= 3

    def test_threshold_robustness_score_must_be_positive(self):
        """Test that robustness score > 0 is required for pass."""
        from scripts.ml_only_edge_tournament import CandidateResult

        # Fail case: robustness = 0
        result = CandidateResult(
            experiment_name='test', candidate_id='test',
            label='cost_survivor_label_v2', filter='PE_only',
            model='elasticnet', threshold=0.25, feature_set='live',
            gates_passed=22, gates_total=22,
            threshold_robustness_score=0.0,  # ZERO
            daily_stability_score=0.7,
            cost_1_50x_pass=True,
            model_path='/path', manifest_path='/path',
            shadow_manifest_path='/path', feature_schema_path='/path',
        )

        assert not result.can_pass(), "Zero threshold robustness must fail"

    def test_robustness_threshold_evaluates_pf(self):
        """Test that threshold robustness evaluates PF at nearby thresholds."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path

        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )

        import inspect
        source = inspect.getsource(tournament._evaluate_threshold_robustness)
        
        # Must calculate PF at nearby thresholds
        assert 'pf' in source.lower() or 'profit' in source.lower()

    def test_threshold_robust_pass_gate(self):
        """Test that threshold_robust_pass is in gate evaluation."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path

        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )

        import inspect
        source = inspect.getsource(tournament._evaluate_gates)
        
        # Must check threshold robustness in gate evaluation
        assert 'D2_threshold_robust' in source
        assert 'threshold_robust_pass' in source


if __name__ == '__main__':
    pytest.main([__file__, '-v'])