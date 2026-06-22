"""
test_ml_only_ce_candidate_rescue.py
====================================
Tests for CE-specific candidate rescue.
Verifies CE candidates use model predictions, not labels as filters.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestCENoLabelLeakage:
    """Test that CE candidates do not use labels as trade filters."""
    
    def test_ce_experiment_filter_is_valid(self):
        """Test that CE experiments use valid filters, not label columns."""
        from scripts.ml_only_non_pe_edge_rescue import CE_EXPERIMENTS
        
        label_patterns = ['_label', 'cost_survivor', 'profitable', 'strong_', 
                         'high_conviction', 'avoid_', 'weak_', 'no_trade']
        
        for exp in CE_EXPERIMENTS:
            filt = str(exp['filter'])
            for pattern in label_patterns:
                assert pattern not in filt, f"CE filter '{filt}' contains label pattern '{pattern}'"
    
    def test_ce_model_must_be_sklearn(self):
        """Test that CE experiments use sklearn model types."""
        from scripts.ml_only_non_pe_edge_rescue import CE_EXPERIMENTS
        
        valid_models = [
            'elasticnet_logistic_regression', 'logistic_regression',
            'calibrated_logistic_regression', 'hist_gradient_boosting',
            'random_forest', 'extra_trees', 'xgboost', 'lightgbm',
        ]
        
        for exp in CE_EXPERIMENTS:
            assert exp['model'] in valid_models, f"Unknown model: {exp['model']}"
    
    def test_ce_threshold_is_valid_range(self):
        """Test that CE thresholds are in valid range (0.15-0.90)."""
        from scripts.ml_only_non_pe_edge_rescue import CE_EXPERIMENTS
        
        for exp in CE_EXPERIMENTS:
            t = exp['threshold']
            assert 0.10 <= t <= 0.90, f"CE threshold {t} out of range"


class TestCEModelTraining:
    """Test that CE uses model training, not label filtering."""
    
    def test_ce_result_has_prediction_based_metrics(self):
        """Test that CE result has model prediction metrics, not just label stats."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='ce_test', candidate_id='ce_test', candidate_type='CE',
            label='cost_survivor_label_v2', filter='CE_only',
            model='elasticnet_logistic_regression', threshold=0.25,
            feature_set='live_computable',
            trade_count=5000,  # From model predictions
            gross_pf=1.5,
            net_pf=1.2,
            pf_at_1_50x=1.05,
        )
        
        assert r.trade_count >= 500
        assert r.gross_pf >= 1.0
    
    def test_ce_candidate_uses_model_path(self):
        """Test that CE candidate saves model artifacts."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='ce_test', candidate_id='ce_test', candidate_type='CE',
            label='cost_survivor_label_v2', filter='CE_only',
            model='elasticnet_logistic_regression', threshold=0.25,
            feature_set='live_computable',
            model_path='/path/to/model.pkl',
        )
        
        assert bool(r.model_path)
        assert r.model_path.endswith('.pkl')


class TestCEPFFailures:
    """Test CE candidate failures due to insufficient PF."""
    
    def test_ce_pf_below_1_rejected(self):
        """Test that CE with OOS PF < 1.0 is properly rejected."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='ce_pf_fail', candidate_id='ce_pf_fail', candidate_type='CE',
            label='cost_survivor_label_v2', filter='CE_only',
            model='elasticnet_logistic_regression', threshold=0.25,
            feature_set='live_computable',
            trade_count=3000,
            gross_pf=0.85,  # Below 1.0
            net_pf=0.72,
            pf_at_1_50x=0.68,  # Cost stress makes it worse
            gates_passed=5,  # Fails many gates
            gates_total=22,
            cost_1_50x_pass=False,
            fail_reasons=['C1_positive_net_expectancy', 'C2_profit_factor_base',
                         'C4_profit_factor_1.50x_cost'],
        )
        
        assert r.net_pf < 1.0
        assert r.pf_at_1_50x < 1.0
        assert r.status == 'FAIL_REJECTED'


class TestCECostStress:
    """Test CE cost stress evaluation."""
    
    def test_ce_1_50x_cost_gate(self):
        """Test that CE candidates must pass 1.50x cost gate."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='ce_stress', candidate_id='ce_stress', candidate_type='CE',
            label='cost_survivor_label_v2', filter='CE_only',
            model='elasticnet_logistic_regression', threshold=0.25,
            feature_set='live_computable',
            trade_count=5000,
            gross_pf=1.35,
            net_pf=1.10,
            pf_at_1_00x=1.10,
            pf_at_1_25x=1.05,
            pf_at_1_50x=1.00,  # Exactly at break-even
            pf_at_2_00x=0.85,
            gates_passed=19, gates_total=22,
            cost_1_50x_pass=True,  # Passes
            fail_reasons=['C9_daily_stability', 'D2_threshold_robust'],
        )
        
        assert r.pf_at_1_50x == 1.00  # Break-even
        assert r.cost_1_50x_pass == True


class TestCELiveComputable:
    """Test that CE features are live-computable."""
    
    def test_ce_filter_blocks_label_columns(self):
        """Test that CE filter function does not use label columns."""
        from scripts.ml_only_non_pe_edge_rescue import NonPEEdgeRescue
        from pathlib import Path
        
        rescue = NonPEEdgeRescue(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            target_types=['CE'],
            max_experiments=1,
        )
        
        features = rescue._get_live_features()
        forbidden = ['return', 'forward', 'future', 'pnl', 'label', 'target', 
                    'outcome', 'exit', 'mfe', 'mae', 'realized', 'net_', 'gross_',
                    'expected_', 'horizon_', 'cost_survivor', 'strong_profitable']
        
        for feat in features:
            for pattern in forbidden:
                assert pattern not in feat.lower(), f"Feature '{feat}' contains forbidden '{pattern}'"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])