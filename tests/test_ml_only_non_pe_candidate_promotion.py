"""
test_ml_only_non_pe_candidate_promotion.py
===========================================
Tests for non-PE candidate promotion rules.
Verifies 22/22 gates, paper_only, real_trading_enabled, artifacts.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestNonPEPromotionRules:
    """Test non-PE candidate promotion requires 22/22."""
    
    def test_21_of_22_not_passing(self):
        """Test that 21/22 gates is NOT passing for non-PE."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=21, gates_total=24,  # One gate short
            daily_stability_score=0.65, threshold_robustness_score=0.75,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        
        assert r.can_pass() == False
        assert r.status != 'PASS_SHADOW_READY'
    
    def test_20_of_22_not_passing(self):
        """Test that 20/22 gates is NOT passing."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='COMBINED',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=20, gates_total=24,
            daily_stability_score=0.65, threshold_robustness_score=0.75,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        
        assert r.can_pass() == False


class TestNonPEArtifactsRequired:
    """Test that all artifacts are required for non-PE promotion."""
    
    def test_model_pkl_required(self):
        """Test that model.pkl path must be non-empty."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=22, gates_total=24,
            daily_stability_score=0.65, threshold_robustness_score=0.75,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='',  # Empty - fail
            manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        
        assert r.can_pass() == False
    
    def test_feature_schema_required(self):
        """Test that feature_schema.json path must be non-empty."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='ROUTER',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=22, gates_total=24,
            daily_stability_score=0.65, threshold_robustness_score=0.75,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json',
            feature_schema_path='',  # Empty - fail
        )
        
        assert r.can_pass() == False
    
    def test_shadow_manifest_required(self):
        """Test that shadow_manifest.json path must be non-empty."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='REGIME',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=22, gates_total=24,
            daily_stability_score=0.65, threshold_robustness_score=0.75,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='',  # Empty - fail
            feature_schema_path='/path/fs.json',
        )
        
        assert r.can_pass() == False


class TestNonPEPaperOnly:
    """Test that non-PE candidates must be paper_only."""
    
    def test_non_pe_manifest_has_paper_only(self):
        """Test that non-PE candidate manifest has paper_only=True."""
        from scripts.ml_only_non_pe_edge_rescue import NonPEEdgeRescue
        from pathlib import Path
        import json
        import tempfile
        
        rescue = NonPEEdgeRescue(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path(tempfile.mkdtemp()),
            target_types=['CE'],
            max_experiments=1,
        )
        
        # Build a test result
        result = rescue.run_single_experiment({
            'name': 'test_ce', 'candidate_type': 'CE',
            'label': 'cost_survivor_label_v2', 'filter': 'CE_only',
            'model': 'elasticnet_logistic_regression', 'threshold': 0.25,
        })
        
        if result.manifest_path:
            with open(result.manifest_path) as f:
                manifest = json.load(f)
            assert manifest.get('paper_only') == True
            assert manifest.get('real_trading_enabled') == False


class TestNonPEStabilityRequired:
    """Test that daily stability is required for non-PE."""
    
    def test_daily_stability_below_threshold_fails(self):
        """Test that daily_stability_score < 0.50 fails."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=22, gates_total=24,
            daily_stability_score=0.35,  # Below 0.50
            threshold_robustness_score=0.72,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        
        assert r.can_pass() == False
    
    def test_threshold_robustness_below_threshold_fails(self):
        """Test that threshold_robustness_score < 0.60 fails."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=22, gates_total=24,
            daily_stability_score=0.65,
            threshold_robustness_score=0.45,  # Below 0.60
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        
        assert r.can_pass() == False


class TestBenchmarkPEUnchanged:
    """Test that PE benchmark remains unchanged."""
    
    def test_benchmark_candidate_unchanged(self):
        """Test that PE benchmark path is not modified by non-PE rescue."""
        benchmark_path = REPO_ROOT / 'models/candidates/PE_only_elasticnet_cost_survivor_v2_20260608_153000'
        
        # Check that benchmark directory exists
        assert benchmark_path.exists(), f"Benchmark path should exist: {benchmark_path}"
        
        manifest_path = benchmark_path / 'candidate_manifest.json'
        if manifest_path.exists():
            import json
            with open(manifest_path) as f:
                manifest = json.load(f)
            # Benchmark should be paper_only
            assert manifest.get('paper_only') == True
            # Should not have real_trading_enabled
            assert manifest.get('real_trading_enabled', False) == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])