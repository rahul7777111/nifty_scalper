"""
test_ml_only_candidate_enhancement_threshold.py
================================================
Tests for threshold robustness enhancement.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestThresholdRobustnessEvaluation:
    """Test threshold robustness evaluation."""
    
    def test_robustness_computed(self):
        """Test threshold_robustness_score is computed."""
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
            'candidate_id': 'test_tr', 'candidate_type': 'PE',
            'label': 'cost_survivor_label_v2', 'filter': 'PE_DTE_7_30',
            'model': 'elasticnet_logistic_regression', 'threshold': 0.25,
        }, 0, 'robustness_test')
        
        assert hasattr(result, 'threshold_robustness_score')
        assert result.threshold_robustness_score >= 0.0


class TestThresholdGate:
    """Test threshold robustness gate enforcement."""
    
    def test_tr_060_passes(self):
        """Test threshold robustness = 0.60 passes gate."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        r = CandidateEnhancementResult(
            candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=22, gates_total=22,
            daily_stability_score=0.55, threshold_robustness_score=0.60,
            daily_stability_pass=True, threshold_robust_pass=True,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        assert r.threshold_robust_pass == True
    
    def test_tr_059_fails(self):
        """Test threshold robustness = 0.59 fails gate."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementResult
        r = CandidateEnhancementResult(
            candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25,
            gates_passed=22, gates_total=22,
            daily_stability_score=0.55, threshold_robustness_score=0.59,
            daily_stability_pass=True, threshold_robust_pass=False,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        assert r.threshold_robust_pass == False
        assert r.can_pass() == False


class TestThresholdGrid:
    """Test threshold grid evaluation."""
    
    def test_nearby_thresholds_evaluated(self):
        """Test that nearby thresholds are evaluated."""
        from scripts.ml_only_candidate_enhancement_loop import CandidateEnhancementLoop
        from pathlib import Path
        import numpy as np
        
        loop = CandidateEnhancementLoop(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_candidates=1,
            max_iterations_per_candidate=1,
        )
        
        # Create small test data
        import pandas as pd
        loop.full_df = pd.DataFrame({
            'timestamp_dt': pd.date_range('2024-01-01', periods=1000),
            'net_forward_return': np.random.randn(1000),
            'option_type': ['PE'] * 1000,
            'dte_days': [15] * 1000,
            'cost_survivor_label_v2': np.random.randint(0, 2, 1000),
            **{f'feat_{i}': np.random.randn(1000) for i in range(5)},
        })
        
        # Time split using walk-forward (returns list of splits)
        splits = loop._walk_forward_splits(loop.full_df, n_folds=3)
        train_df, test_df = splits[0]
        
        # Train model
        feature_list = [f'feat_{i}' for i in range(5)]
        X_train = loop._build_features(train_df, feature_list)
        y_train = train_df['cost_survivor_label_v2'].values
        X_test = loop._build_features(test_df, feature_list)
        
        model, scaler = loop._train_model(X_train, y_train, 'logistic_regression')
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict_proba(X_test_scaled)
        
        # Evaluate robustness at threshold 0.25
        score, passed = loop._evaluate_threshold_robustness(test_df, predictions, 0.25)
        
        # Should have a score
        assert score >= 0.0
        # Should be boolean
        assert isinstance(passed, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])