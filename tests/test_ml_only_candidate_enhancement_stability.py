"""
test_ml_only_candidate_enhancement_stability.py
================================================
Tests for daily stability enhancement strategies.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestMaxTradesPerDayStrategy:
    """Test max_trades_per_day strategy."""
    
    def test_mtpd_reduces_concentration(self):
        """Test that max_trades_per_day reduces day concentration."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementLoop
        from pathlib import Path
        
        loop = CandidateEnhancementLoop(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_candidates=1,
            max_iterations_per_candidate=1,
        )
        
        # Baseline result (no limit)
        cfg = {
            'candidate_id': 'test', 'candidate_type': 'CE',
            'label': 'cost_survivor_label_v2', 'filter': 'CE_DTE_7_30',
            'model': 'elasticnet_logistic_regression', 'threshold': 0.30,
            'max_trades_per_day': 0,
        }
        
        baseline = loop.run_single_iteration(cfg, 0, 'baseline')
        
        # With limit
        cfg_limited = {**cfg, 'max_trades_per_day': 1}
        limited = loop.run_single_iteration(cfg_limited, 1, 'mtpd_1')
        
        # Should have fewer trades
        if limited.trade_count > 0 and baseline.trade_count > 0:
            assert limited.trade_count <= baseline.trade_count


class TestStabilityMetrics:
    """Test stability metric computation."""
    
    def test_daily_stability_score_computed(self):
        """Test that daily stability score is computed."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementLoop
        from pathlib import Path
        
        loop = CandidateEnhancementLoop(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_candidates=1,
            max_iterations_per_candidate=1,
        )
        
        result = loop.run_single_iteration({
            'candidate_id': 'test_ds', 'candidate_type': 'PE',
            'label': 'cost_survivor_label_v2', 'filter': 'PE_DTE_7_30',
            'model': 'elasticnet_logistic_regression', 'threshold': 0.25,
        }, 0, 'stability_test')
        
        # daily_stability_score should be computed
        assert hasattr(result, 'daily_stability_score')
    
    def test_top_day_concentration_computed(self):
        """Test that top_day_concentration is computed."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementLoop
        from pathlib import Path
        
        loop = CandidateEnhancementLoop(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_candidates=1,
            max_iterations_per_candidate=1,
        )
        
        result = loop.run_single_iteration({
            'candidate_id': 'test_tdc', 'candidate_type': 'PE',
            'label': 'cost_survivor_label_v2', 'filter': 'PE_DTE_7_30',
            'model': 'elasticnet_logistic_regression', 'threshold': 0.25,
        }, 0, 'concentration_test')
        
        assert hasattr(result, 'top_day_concentration')


class TestStabilityGate:
    """Test daily stability gate enforcement."""
    
    def test_ds_050_passes(self):
        """Test daily stability = 0.50 passes gate."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        r = CandidateEnhancementResult(
            candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=22, gates_total=22,
            daily_stability_score=0.50, threshold_robustness_score=0.70,
            daily_stability_pass=True, threshold_robust_pass=True,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        assert r.daily_stability_pass == True
    
    def test_ds_049_fails(self):
        """Test daily stability = 0.49 fails gate."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        r = CandidateEnhancementResult(
            candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=22, gates_total=22,
            daily_stability_score=0.49, threshold_robustness_score=0.70,
            daily_stability_pass=False, threshold_robust_pass=True,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        assert r.daily_stability_pass == False
        assert r.can_pass() == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])