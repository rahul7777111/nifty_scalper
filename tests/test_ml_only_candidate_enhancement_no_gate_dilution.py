"""
test_ml_only_candidate_enhancement_no_gate_dilution.py
======================================================
Tests that verify NO gate dilution in enhancement loop.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestGateThresholdsNotDiluted:
    """Test that gate thresholds are not weakened."""
    
    def test_pf_1_00x_not_below_1_15(self):
        """Test PF_AT_1_00X is 1.15, not lower."""
        from scripts.ml_only_candidate_enhancement_loop import PF_AT_1_00X
        assert PF_AT_1_00X == 1.15
    
    def test_pf_1_25x_not_below_1_05(self):
        """Test PF_AT_1_25X is 1.05, not lower."""
        from scripts.ml_only_candidate_enhancement_loop import PF_AT_1_25X
        assert PF_AT_1_25X == 1.05
    
    def test_pf_1_50x_is_break_even(self):
        """Test PF_AT_1_50X is exactly 1.00 (break-even)."""
        from scripts.ml_only_candidate_enhancement_loop import PF_AT_1_50X
        assert PF_AT_1_50X == 1.00  # Cannot be > 1.00
    
    def test_pf_2_00x_is_0_80(self):
        """Test PF_AT_2_00X is 0.80."""
        from scripts.ml_only_candidate_enhancement_loop import PF_AT_2_00X
        assert PF_AT_2_00X == 0.80


class TestDailyStabilityThreshold:
    """Test daily stability threshold is not weakened."""
    
    def test_min_daily_stability_050(self):
        """Test MIN_DAILY_STABILITY is 0.50, not lower."""
        from scripts.ml_only_candidate_enhancement_loop import MIN_DAILY_STABILITY
        assert MIN_DAILY_STABILITY == 0.50
    
    def test_cannot_pass_with_040_stability(self):
        """Test that 0.40 daily stability cannot pass."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        r = CandidateEnhancementResult(
            candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=22, gates_total=22,
            daily_stability_score=0.40,  # Below 0.50
            threshold_robustness_score=0.70,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        assert r.can_pass() == False


class TestThresholdRobustness:
    """Test threshold robustness threshold is not weakened."""
    
    def test_min_threshold_robustness_060(self):
        """Test MIN_THRESHOLD_ROBUSTNESS is 0.60."""
        from scripts.ml_only_candidate_enhancement_loop import MIN_THRESHOLD_ROBUSTNESS
        assert MIN_THRESHOLD_ROBUSTNESS == 0.60
    
    def test_cannot_pass_with_050_robustness(self):
        """Test that 0.50 robustness cannot pass."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        r = CandidateEnhancementResult(
            candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=22, gates_total=22,
            daily_stability_score=0.55,
            threshold_robustness_score=0.50,  # Below 0.60
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        assert r.can_pass() == False


class TestNo21of22Pass:
    """Test that 21/22 is never marked PASS_SHADOW_READY."""
    
    def test_21_gates_ce_fails(self):
        """Test CE 21/22 gates fails."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        r = CandidateEnhancementResult(
            candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=21, gates_total=22,
            daily_stability_score=0.70, threshold_robustness_score=0.80,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        assert r.can_pass() == False
        assert r.status != 'PASS_SHADOW_READY'
    
    def test_21_gates_pe_fails(self):
        """Test PE 21/22 gates fails."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        r = CandidateEnhancementResult(
            candidate_id='test', candidate_type='PE',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=21, gates_total=22,
            daily_stability_score=0.70, threshold_robustness_score=0.80,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        assert r.can_pass() == False
    
    def test_21_gates_combined_fails(self):
        """Test combined 21/22 gates fails."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        r = CandidateEnhancementResult(
            candidate_id='test', candidate_type='COMBINED',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=21, gates_total=22,
            daily_stability_score=0.70, threshold_robustness_score=0.80,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        assert r.can_pass() == False


class TestMinTradesNotLowered:
    """Test minimum trades threshold not lowered."""
    
    def test_min_trades_500(self):
        """Test MIN_TRADES is 500."""
        from scripts.ml_only_candidate_enhancement_loop import MIN_TRADES
        assert MIN_TRADES == 500


class TestCostStressNotReduced:
    """Test transaction cost assumption not reduced."""
    
    def test_cost_per_trade_025(self):
        """Test COST_PER_TRADE_PCT is 0.25%."""
        from scripts.ml_only_candidate_enhancement_loop import COST_PER_TRADE_PCT
        assert COST_PER_TRADE_PCT == 0.0025


if __name__ == '__main__':
    pytest.main([__file__, '-v'])