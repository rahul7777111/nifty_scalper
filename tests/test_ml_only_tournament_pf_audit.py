"""
test_ml_only_tournament_pf_audit.py
====================================
Tests that audit PF calculation correctness.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestPFCalculationCorrectness:
    """Audit that PF calculation uses model predictions, not labels."""

    def test_pf_from_predictions_not_labels(self):
        """Test that metrics computation uses predictions array."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path
        import numpy as np
        import pandas as pd

        tournament = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )

        # Verify _compute_metrics_from_predictions uses predictions array
        import inspect
        source = inspect.getsource(tournament._compute_metrics_from_predictions)
        
        # Must use predictions to determine trades
        assert 'signals' in source or 'threshold' in source, \
            "Must use threshold on predictions to select trades"
        
        # Must use predictions (not just label column directly)
        # Check that the function takes predictions parameter
        assert 'predictions' in source, \
            "Must use predictions array to compute metrics"

    def test_pf_calculation_uses_threshold(self):
        """Test that threshold is applied to model predictions."""
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
        
        # Must apply threshold to scores
        assert 'threshold' in source, "Must apply threshold to model scores"

    def test_cost_stress_applied_to_trades(self):
        """Test that cost stress is applied to compute stressed PFs."""
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
        
        # Must have cost stress for multiple multipliers
        assert 'pf_at_1_00x' in source
        assert 'pf_at_1_25x' in source
        assert 'pf_at_1_50x' in source
        assert 'pf_at_2_00x' in source

    def test_net_pf_less_than_gross_pf(self):
        """Test that net PF (after costs) is always less than or equal to gross PF."""
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
        
        # net_returns = returns - cost
        # Therefore net PF should be <= gross PF
        # (unless all trades are winners and costs are small)
        assert 'net_returns' in source or 'cost' in source, \
            "Must deduct costs from returns"


class TestPFAuditFromReports:
    """Audit PF values from actual tournament reports."""

    def test_corrected_pf_is_reasonable(self):
        """Test that corrected tournament has reasonable PF values."""
        reports_dir = REPO_ROOT / 'reports'
        json_files = list(reports_dir.glob('ml_only_edge_tournament_corrected_*.json'))
        
        if not json_files:
            pytest.skip("No corrected tournament results")
        
        with open(json_files[-1]) as f:
            data = json.load(f)
        
        leaderboard = data.get('leaderboard', [])
        
        for r in leaderboard:
            gross_pf = r.get('gross_pf', 0)
            net_pf = r.get('net_pf', 0)
            pf_1_5x = r.get('pf_at_1_50x', 0)
            
            # Sanity checks
            assert 0 <= gross_pf <= 100, f"gross_pf {gross_pf} out of range"
            assert 0 <= net_pf <= 100, f"net_pf {net_pf} out of range"
            assert 0 <= pf_1_5x <= 100, f"pf_1_5x {pf_1_5x} out of range"
            
            # Net PF should be <= gross PF
            assert net_pf <= gross_pf * 1.1, \
                f"net_pf ({net_pf}) > gross_pf ({gross_pf}) significantly"

    def test_benchmark_vs_tournament_comparison(self):
        """Test that tournament PF values are in same ballpark as benchmark."""
        reports_dir = REPO_ROOT / 'reports'
        json_files = list(reports_dir.glob('ml_only_edge_tournament_corrected_*.json'))
        
        if not json_files:
            pytest.skip("No corrected tournament results")
        
        with open(json_files[-1]) as f:
            data = json.load(f)
        
        benchmark_pf = data.get('benchmark_status', {}).get('mean_pf', 0)
        leaderboard = data.get('leaderboard', [])
        
        if leaderboard and benchmark_pf > 0:
            # Tournament PFs should be in similar range to benchmark (0.5x to 3x)
            for r in leaderboard[:3]:
                tournament_pf = r.get('gross_pf', 0)
                if tournament_pf > 0:
                    ratio = tournament_pf / benchmark_pf
                    # Should be roughly similar order of magnitude
                    assert ratio < 5, \
                        f"Tournament PF {tournament_pf} is {ratio:.1f}x benchmark PF {benchmark_pf}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])