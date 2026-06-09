"""
test_ml_only_candidate_enhancement_loop.py
===========================================
Tests for the candidate enhancement loop.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestEnhancementLoopImport:
    """Test that enhancement loop imports."""
    
    def test_import(self):
        """Test CandidateEnhancementLoop imports."""
        from scripts.ml_only_candidate_enhancement_loop import (
            CandidateEnhancementLoop, CandidateEnhancementResult,
            TOTAL_GATES, MIN_DAILY_STABILITY, PF_AT_1_50X, COST_PER_TRADE_PCT
        )
        assert TOTAL_GATES == 22
        assert MIN_DAILY_STABILITY == 0.50
        assert PF_AT_1_50X == 1.00


class TestEnhancementConstants:
    """Test gate thresholds are not weakened."""
    
    def test_22_gates(self):
        """Test TOTAL_GATES is 22."""
        from scripts.ml_only_candidate_enhancement_loop import TOTAL_GATES
        assert TOTAL_GATES == 22
    
    def test_1_50x_gate_is_break_even(self):
        """Test 1.50x cost gate is break-even (1.0)."""
        from scripts.ml_only_candidate_enhancement_loop import PF_AT_1_50X
        assert PF_AT_1_50X == 1.00
    
    def test_daily_stability_threshold(self):
        """Test daily stability threshold is 0.50."""
        from scripts.ml_only_candidate_enhancement_loop import MIN_DAILY_STABILITY
        assert MIN_DAILY_STABILITY == 0.50
    
    def test_threshold_robustness_threshold(self):
        """Test threshold robustness threshold is 0.60."""
        from scripts.ml_only_candidate_enhancement_loop import MIN_THRESHOLD_ROBUSTNESS
        assert MIN_THRESHOLD_ROBUSTNESS == 0.60


class TestEnhancementResult:
    """Test CandidateEnhancementResult dataclass."""
    
    def test_result_creation(self):
        """Test result dataclass."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        r = CandidateEnhancementResult(
            candidate_id='test', candidate_type='PE',
            label='cost_survivor_label_v2', filter='PE_DTE_7_30',
            model='elasticnet_logistic_regression', threshold=0.25,
        )
        assert r.candidate_id == 'test'
        assert r.gates_total == 22
        assert r.gates_passed == 0
        d = r.to_dict()
        assert d['candidate_type'] == 'PE'
    
    def test_can_pass_requires_22_gates(self):
        """Test can_pass requires exactly 22/22."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        r = CandidateEnhancementResult(
            candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=22, gates_total=22,
            daily_stability_score=0.55, threshold_robustness_score=0.70,
            daily_stability_pass=True, threshold_robust_pass=True,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        assert r.can_pass() == True
    
    def test_21_gates_fails(self):
        """Test 21/22 gates does NOT pass."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        r = CandidateEnhancementResult(
            candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=21, gates_total=22,
            daily_stability_score=0.55, threshold_robustness_score=0.70,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        assert r.can_pass() == False
    
    def test_low_daily_stability_fails(self):
        """Test daily_stability < 0.50 fails."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        r = CandidateEnhancementResult(
            candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=22, gates_total=22,
            daily_stability_score=0.35,  # Below 0.50
            threshold_robustness_score=0.70,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        assert r.can_pass() == False


class TestEnhancementStrategies:
    """Test enhancement strategies."""
    
    def test_mtpd_strategy(self):
        """Test max_trades_per_day strategy works."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementLoop
        from pathlib import Path
        
        loop = CandidateEnhancementLoop(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_candidates=1,
            max_iterations_per_candidate=1,
        )
        
        cfg = {
            'name': 'test', 'candidate_id': 'test', 'candidate_type': 'PE',
            'label': 'cost_survivor_label_v2', 'filter': 'PE_DTE_7_30',
            'model': 'elasticnet_logistic_regression', 'threshold': 0.30,
            'max_trades_per_day': 3,
            'primary_blockers': ['C9_daily_stability'],
        }
        
        strategies = loop._get_enhancement_strategies(cfg, 0, None)
        
        # Should have baseline and max_trades strategies
        strat_names = [s['strategy'] for s in strategies]
        assert 'baseline' in strat_names
        assert any('max_trades' in s or 'mtpd' in s for s in strat_names)
    
    def test_iteration_1_uses_previous_result(self):
        """Test iteration 1 uses previous result for guidance."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementLoop, CandidateEnhancementResult
        from pathlib import Path
        
        loop = CandidateEnhancementLoop(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_candidates=1,
            max_iterations_per_candidate=1,
        )
        
        cfg = {'name': 'test', 'candidate_id': 'test', 'candidate_type': 'PE',
               'label': 'cost_survivor_label_v2', 'filter': 'PE_DTE_7_30',
               'model': 'elasticnet_logistic_regression', 'threshold': 0.30,
               'max_trades_per_day': 3, 'primary_blockers': ['C9_daily_stability']}
        
        prev = CandidateEnhancementResult(
            candidate_id='test', candidate_type='PE',
            label='cost_survivor_label_v2', filter='PE_DTE_7_30',
            model='elasticnet_logistic_regression', threshold=0.30,
            daily_stability_score=0.35, pf_at_1_50x=1.5,
            top_day_concentration=0.50, max_trades_per_day=3,
        )
        
        strategies = loop._get_enhancement_strategies(cfg, 1, prev)
        
        # Should contain mtpd strategies since C9 fails
        strat_names = [s['strategy'] for s in strategies]
        assert any('mtpd' in s for s in strat_names)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])