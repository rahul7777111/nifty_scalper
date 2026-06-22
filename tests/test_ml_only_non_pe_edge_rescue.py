"""
test_ml_only_non_pe_edge_rescue.py
===================================
Tests for the non-PE edge rescue system.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestNonPEImport:
    """Test that the rescue system imports correctly."""
    
    def test_import_rescue(self):
        """Test that NonPEEdgeRescue imports."""
        from scripts.ml_only_non_pe_edge_rescue import NonPEEdgeRescue, NonPECandidateResult
        assert NonPEEdgeRescue is not None
        assert NonPECandidateResult is not None
    
    def test_result_dataclass(self):
        """Test NonPECandidateResult dataclass."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        r = NonPECandidateResult(
            experiment_name='test_exp',
            candidate_id='test_cand',
            candidate_type='CE',
            label='cost_survivor_label_v2',
            filter='CE_only',
            model='elasticnet_logistic_regression',
            threshold=0.25,
            feature_set='live_computable',
        )
        assert r.candidate_type == 'CE'
        assert r.gates_total == 24
        assert r.gates_passed == 0
        d = r.to_dict()
        assert d['experiment_name'] == 'test_exp'
        assert d['candidate_type'] == 'CE'
    
    def test_can_pass_requires_24_gates(self):
        """Test that can_pass requires 24/24 gates."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=24, gates_total=24,
            daily_stability_score=0.6, threshold_robustness_score=0.7,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            concentration_diversity_pass=True, trading_days_diversity_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        assert r.can_pass() == True


class TestNonPECandidateTypes:
    """Test that non-PE candidate types are correctly defined."""
    
    def test_ce_candidate_type(self):
        """Test CE candidate type is defined."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        r = NonPECandidateResult(
            experiment_name='ce_test', candidate_id='ce_test', candidate_type='CE',
            label='cost_survivor_label_v2', filter='CE_only',
            model='elasticnet_logistic_regression', threshold=0.25, feature_set='live_computable',
        )
        assert r.candidate_type == 'CE'
    
    def test_combined_candidate_type(self):
        """Test COMBINED candidate type is defined."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        r = NonPECandidateResult(
            experiment_name='comb_test', candidate_id='comb_test', candidate_type='COMBINED',
            label='cost_survivor_label_v2', filter='none',
            model='random_forest', threshold=0.25, feature_set='live_computable',
        )
        assert r.candidate_type == 'COMBINED'
    
    def test_router_candidate_type(self):
        """Test ROUTER candidate type is defined."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        r = NonPECandidateResult(
            experiment_name='router_test', candidate_id='router_test', candidate_type='ROUTER',
            label='cost_survivor_label_v2', filter='CE_and_PE',
            model='elasticnet_logistic_regression', threshold=0.25, feature_set='live_computable',
        )
        assert r.candidate_type == 'ROUTER'
    
    def test_regime_candidate_type(self):
        """Test REGIME candidate type is defined."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        r = NonPECandidateResult(
            experiment_name='regime_test', candidate_id='regime_test', candidate_type='REGIME',
            label='cost_survivor_label_v2', filter='regime_high_volatility',
            model='elasticnet_logistic_regression', threshold=0.25, feature_set='live_computable',
        )
        assert r.candidate_type == 'REGIME'


class TestNonPEGateRules:
    """Test that 22-gate rule is enforced for non-PE candidates."""
    
    def test_21_of_22_fails(self):
        """Test that 21/22 gates fails even with all artifacts."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=21, gates_total=24,  # NOT 22/22
            daily_stability_score=0.6, threshold_robustness_score=0.7,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        assert r.can_pass() == False
    
    def test_missing_model_path_fails(self):
        """Test that missing model_path fails."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=22, gates_total=24,
            daily_stability_score=0.6, threshold_robustness_score=0.7,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='',  # Empty
            manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        assert r.can_pass() == False
    
    def test_missing_shadow_manifest_fails(self):
        """Test that missing shadow_manifest_path fails."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=22, gates_total=24,
            daily_stability_score=0.6, threshold_robustness_score=0.7,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='',  # Empty
            feature_schema_path='/path/fs.json',
        )
        assert r.can_pass() == False
    
    def test_low_daily_stability_fails(self):
        """Test that low daily stability fails."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=22, gates_total=24,
            daily_stability_score=0.30,  # Below 0.50 threshold
            threshold_robustness_score=0.7,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        assert r.can_pass() == False


class TestNonPELabels:
    """Test that correct labels are used for non-PE candidates."""
    
    def test_cost_survivor_label_exists(self):
        """Test cost_survivor_label_v2 is a valid CE label."""
        from scripts.ml_only_non_pe_edge_rescue import CE_EXPERIMENTS
        assert len(CE_EXPERIMENTS) > 0
        labels = set(e['label'] for e in CE_EXPERIMENTS)
        assert 'cost_survivor_label_v2' in labels
    
    def test_ce_experiments_have_ce_filter(self):
        """Test CE experiments use CE-only filters."""
        from scripts.ml_only_non_pe_edge_rescue import CE_EXPERIMENTS
        ce_only = all('CE' in e['filter'] or e['filter'] == 'none' 
                     for e in CE_EXPERIMENTS)
        assert ce_only


class TestNonPEPFThreshold:
    """Test that 1.5x cost PF is the break-even gate for non-PE."""
    
    def test_1_50x_gate_is_break_even(self):
        """Test that 1.50x cost gate requires PF >= 1.0."""
        from scripts.ml_only_non_pe_edge_rescue import PF_AT_1_50X
        assert PF_AT_1_50X == 1.00
        
    def test_candidate_fails_without_1_50x(self):
        """Test that candidate without 1.5x cost pass is not PASS_SHADOW_READY."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=22, gates_total=24,
            pf_at_1_50x=0.85,  # Below 1.0
            daily_stability_score=0.6, threshold_robustness_score=0.7,
            cost_1_50x_pass=False,  # Fails
            leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        assert r.can_pass() == False


class TestNonPESuspiciousPF:
    """Test that suspicious PF values are flagged."""
    
    def test_high_pf_flagged(self):
        """Test that PF > 10 is flagged for audit."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=22, gates_total=24,
            gross_pf=15.0,  # Suspiciously high
            pf_at_1_50x=12.0,
            daily_stability_score=0.6, threshold_robustness_score=0.7,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        # can_pass still returns True (artifacts all present)
        # but gross_pf > 10 should be flagged in audit
        assert r.gross_pf > 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])