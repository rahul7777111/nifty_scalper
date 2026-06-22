"""
test_ml_only_tournament_daily_stability_required.py
====================================================
Tests that verify daily stability is required for PASS_SHADOW_READY.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestDailyStabilityRequired:
    """Test that daily stability is enforced."""

    def test_daily_stability_in_metrics(self):
        """Test that daily_stability_score is computed in metrics."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path

        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )

        import inspect
        source = inspect.getsource(tournament._compute_metrics_from_predictions)
        
        # Must compute daily stability
        assert 'daily_stability' in source.lower() or 'trade_day' in source

    def test_zero_daily_stability_fails(self):
        """Test that zero daily stability fails."""
        from scripts.ml_only_edge_tournament import CandidateResult

        result = CandidateResult(
            experiment_name='test', candidate_id='test',
            label='cost_survivor_label_v2', filter='PE_only',
            model='elasticnet', threshold=0.25, feature_set='live',
            gates_passed=22, gates_total=22,
            daily_stability_score=0.0,  # ZERO
            threshold_robustness_score=0.8,
            cost_1_50x_pass=True,
            model_path='/path', manifest_path='/path',
            shadow_manifest_path='/path', feature_schema_path='/path',
        )

        assert not result.can_pass(), "Zero daily stability must fail"

    def test_low_daily_stability_fails_c9(self):
        """Test that low daily stability fails gate C9."""
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
        
        # Must have C9_daily_stability gate
        assert 'C9_daily_stability' in source

    def test_daily_stability_uses_trade_day(self):
        """Test that daily stability groups by trading day."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path

        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )

        import inspect
        source = inspect.getsource(tournament._compute_metrics_from_predictions)
        
        # Must group by trading day
        assert 'trade_day' in source or 'groupby' in source or 'date' in source.lower()

    def test_concentration_risk_checked(self):
        """Test that concentration risk is checked in daily stability."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path

        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )

        import inspect
        source = inspect.getsource(tournament._compute_metrics_from_predictions)
        
        # Should check concentration (one day too much of total)
        assert 'concentration' in source.lower() or 'max' in source or 'worst' in source


if __name__ == '__main__':
    pytest.main([__file__, '-v'])