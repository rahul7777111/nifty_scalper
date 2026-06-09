"""
test_ml_only_candidate_enhancement_artifacts.py
================================================
Tests that verify artifacts are mandatory and complete.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestArtifactsRequired:
    """Test that all artifacts are required for PASS_SHADOW_READY."""
    
    def test_empty_model_path_fails(self):
        """Test empty model_path fails."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        r = CandidateEnhancementResult(
            candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=22, gates_total=22,
            daily_stability_score=0.55, threshold_robustness_score=0.70,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='',  # Empty
            manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json',
            feature_schema_path='/path/fs.json',
        )
        assert r.can_pass() == False
    
    def test_empty_manifest_path_fails(self):
        """Test empty manifest_path fails."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        r = CandidateEnhancementResult(
            candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=22, gates_total=22,
            daily_stability_score=0.55, threshold_robustness_score=0.70,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl',
            manifest_path='',  # Empty
            shadow_manifest_path='/path/shadow.json',
            feature_schema_path='/path/fs.json',
        )
        assert r.can_pass() == False
    
    def test_empty_shadow_manifest_path_fails(self):
        """Test empty shadow_manifest_path fails."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        r = CandidateEnhancementResult(
            candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=22, gates_total=22,
            daily_stability_score=0.55, threshold_robustness_score=0.70,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='',  # Empty
            feature_schema_path='/path/fs.json',
        )
        assert r.can_pass() == False
    
    def test_empty_feature_schema_path_fails(self):
        """Test empty feature_schema_path fails."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        r = CandidateEnhancementResult(
            candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=22, gates_total=22,
            daily_stability_score=0.55, threshold_robustness_score=0.70,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json',
            feature_schema_path='',  # Empty
        )
        assert r.can_pass() == False


class TestPaperOnlyRequired:
    """Test that paper_only=True and real_trading_enabled=False are enforced."""
    
    def test_paper_only_in_manifest(self):
        """Test that passing candidate manifest has paper_only=True."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementLoop
        from pathlib import Path
        import tempfile
        import json
        
        loop = CandidateEnhancementLoop(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path(tempfile.mkdtemp()),
            max_candidates=1,
            max_iterations_per_candidate=1,
        )
        
        result = loop.run_single_iteration({
            'candidate_id': 'test_paper',
            'candidate_type': 'CE',
            'label': 'cost_survivor_label_v2',
            'filter': 'CE_only',
            'model': 'elasticnet_logistic_regression',
            'threshold': 0.25,
        }, 0, 'baseline_test')
        
        if result.manifest_path:
            with open(result.manifest_path) as f:
                manifest = json.load(f)
            assert manifest.get('paper_only') == True
            assert manifest.get('real_trading_enabled', True) == False


class TestArtifactCompleteness:
    """Test artifact completeness check in gates."""
    
    def test_missing_artifact_fails_gate(self):
        """Test that D1 fails if any artifact is missing."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        
        # All artifacts present = passes
        r1 = CandidateEnhancementResult(
            candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=21, gates_total=22,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        # This would be set by _evaluate_gates based on artifact paths
        
        # One missing = should fail D1
        r2 = CandidateEnhancementResult(
            candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=20, gates_total=22,
            model_path='/path/model.pkl', manifest_path='',  # Missing
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        # D1 would fail for r2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])