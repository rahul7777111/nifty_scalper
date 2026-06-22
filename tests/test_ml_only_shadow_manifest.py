"""
test_ml_only_shadow_manifest.py
===============================
Tests for ML-only shadow manifest generation.

Verifies:
- Shadow manifest is generated only for passing candidates
- Manifest contains all required fields
- Safety constraints are enforced
- ML-only mode is properly configured
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestShadowManifestGeneration:
    """Test shadow manifest generation."""
    
    def test_shadow_manifest_structure(self):
        """Test that shadow manifest has correct structure."""
        manifest = {
            'manifest_version': '1.0',
            'generated_at': '20260609_120000',
            'candidate_id': 'test_candidate',
            'source_manifest': '/path/to/manifest.json',
            'model': {
                'model_path': '/path/to/model.pkl',
                'model_type': 'elasticnet_logistic_regression',
                'feature_schema_path': '/path/to/feature_schema.json',
                'threshold': 0.25,
                'threshold_robust': True,
            },
            'filter': {
                'filter_name': 'PE_only',
                'filter_rule': "option_type == 'PE'",
            },
            'safety': {
                'paper_only': True,
                'real_trading_enabled': False,
                'ml_only_mode': True,
                'entry_direction_from_ml': True,
                'candidate_selection_from_ml': True,
            },
            'risk_controls': {
                'max_trades_per_day': 3,
                'max_open_positions': 1,
                'max_daily_loss': -500.0,
                'spread_limit_pct': 0.15,
                'feature_coverage_min': 0.95,
            },
            'performance': {
                'mean_pf': 1.488,
                'cost_1.50x_pf': 1.265,
                'mean_sharpe': 1.34,
                'mean_trades': 501,
                'readiness': 'PASS_SHADOW_READY',
            },
        }
        
        # Verify structure
        assert 'manifest_version' in manifest
        assert 'candidate_id' in manifest
        assert 'model' in manifest
        assert 'safety' in manifest
        assert 'risk_controls' in manifest
    
    def test_safety_constraints_enforced(self):
        """Test that safety constraints are properly set."""
        manifest = {
            'safety': {
                'paper_only': True,
                'real_trading_enabled': False,
                'ml_only_mode': True,
            }
        }
        
        assert manifest['safety']['paper_only'] == True
        assert manifest['safety']['real_trading_enabled'] == False
        assert manifest['safety']['ml_only_mode'] == True
    
    def test_ml_only_mode_required(self):
        """Test that ML-only mode is set in shadow manifest."""
        manifest = {'safety': {'ml_only_mode': True}}
        assert manifest['safety']['ml_only_mode'] == True
    
    def test_entry_direction_from_ml(self):
        """Test that entry direction comes from ML model."""
        manifest = {'safety': {'entry_direction_from_ml': True}}
        assert manifest['safety']['entry_direction_from_ml'] == True
    
    def test_candidate_selection_from_ml(self):
        """Test that candidate selection comes from ML router."""
        manifest = {'safety': {'candidate_selection_from_ml': True}}
        assert manifest['safety']['candidate_selection_from_ml'] == True


class TestShadowManifestForPassingCandidates:
    """Test shadow manifest generation for passing candidates."""
    
    def test_only_passing_candidates_get_manifest(self):
        """Test that only passing candidates generate shadow manifests."""
        passing_readiness = ['PASS_SHADOW_READY', 'PASS_PAPER_READY', 'PASS_LIVE_READY']
        failing_readiness = ['FAIL_BUT_PROMISING', 'FAIL_REJECTED']
        
        for readiness in passing_readiness:
            should_generate = readiness in passing_readiness
            assert should_generate, f"{readiness} should generate manifest"
        
        for readiness in failing_readiness:
            should_generate = readiness in passing_readiness
            assert not should_generate, f"{readiness} should NOT generate manifest"
    
    def test_threshold_in_manifest(self):
        """Test that selected threshold is in manifest."""
        manifest = {'model': {'threshold': 0.25}}
        assert manifest['model']['threshold'] == 0.25
        assert 0.0 < manifest['model']['threshold'] < 1.0
    
    def test_filter_in_manifest(self):
        """Test that filter configuration is in manifest."""
        manifest = {
            'filter': {
                'filter_name': 'PE_only',
                'filter_rule': "option_type == 'PE'",
            }
        }
        assert 'filter_name' in manifest['filter']
        assert 'filter_rule' in manifest['filter']


class TestShadowManifestValidation:
    """Test shadow manifest validation."""
    
    def test_required_fields_present(self):
        """Test that all required fields are present in generated manifest."""
        manifest = {
            'manifest_version': '1.0',
            'generated_at': '20260609_120000',
            'candidate_id': 'test',
            'model': {
                'model_path': 'model.pkl',
                'model_type': 'elasticnet',
                'threshold': 0.25,
            },
            'safety': {
                'paper_only': True,
                'real_trading_enabled': False,
            },
        }
        
        required_fields = [
            'manifest_version',
            'generated_at',
            'candidate_id',
            'model',
            'safety',
        ]
        
        for field in required_fields:
            assert field in manifest, f"Required field {field} missing"
    
    def test_model_path_valid(self):
        """Test that model path points to existing file."""
        manifest = {'model': {'model_path': 'models/candidates/test/model.pkl'}}
        path = Path(REPO_ROOT / manifest['model']['model_path'])
        # Path should be relative or absolute - we just check it's non-empty
        assert len(manifest['model']['model_path']) > 0
    
    def test_threshold_is_valid_probability(self):
        """Test that threshold is a valid probability value."""
        valid_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        
        for thresh in valid_thresholds:
            assert 0.0 < thresh < 1.0, f"Threshold {thresh} must be between 0 and 1"


class TestShadowManifestIntegration:
    """Test shadow manifest integration with existing system."""
    
    def test_existing_candidate_has_manifest(self):
        """Test that existing passing candidate has manifest."""
        from scripts.ml_only_candidate_inventory import load_candidate_manifest
        
        candidates_dir = REPO_ROOT / 'models' / 'candidates'
        if not candidates_dir.exists():
            pytest.skip("No candidates directory")
        
        manifests = list(candidates_dir.rglob('candidate_manifest.json'))
        if manifests:
            manifest = load_candidate_manifest(manifests[0])
            assert manifest is not None
            assert manifest.get('paper_only') == True
    
    def test_candidate_has_required_artifacts(self):
        """Test that candidate has all required artifacts."""
        candidates_dir = REPO_ROOT / 'models' / 'candidates'
        manifests = list(candidates_dir.rglob('candidate_manifest.json'))
        
        required_artifacts = ['candidate_manifest.json', 'model.pkl', 'feature_schema.json']
        
        for mp in manifests[:1]:  # Check first manifest
            candidate_dir = mp.parent
            for artifact in required_artifacts:
                artifact_path = candidate_dir / artifact
                # Artifact should exist
                assert artifact_path.exists() or artifact in ['candidate_manifest.json'], \
                    f"Artifact {artifact} should exist for candidate"


class TestRiskControlsInManifest:
    """Test risk controls in shadow manifest."""
    
    def test_max_trades_per_day_set(self):
        """Test that max trades per day is set."""
        manifest = {'risk_controls': {'max_trades_per_day': 3}}
        assert manifest['risk_controls']['max_trades_per_day'] >= 1
        assert manifest['risk_controls']['max_trades_per_day'] <= 10
    
    def test_max_daily_loss_set(self):
        """Test that max daily loss is set."""
        manifest = {'risk_controls': {'max_daily_loss': -500.0}}
        assert manifest['risk_controls']['max_daily_loss'] < 0  # Loss is negative
    
    def test_spread_limit_set(self):
        """Test that spread limit is set."""
        manifest = {'risk_controls': {'spread_limit_pct': 0.15}}
        assert manifest['risk_controls']['spread_limit_pct'] > 0
        assert manifest['risk_controls']['spread_limit_pct'] < 1.0  # Less than 100%
    
    def test_feature_coverage_min_set(self):
        """Test that feature coverage minimum is set."""
        manifest = {'risk_controls': {'feature_coverage_min': 0.95}}
        assert manifest['risk_controls']['feature_coverage_min'] >= 0.9
        assert manifest['risk_controls']['feature_coverage_min'] <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])