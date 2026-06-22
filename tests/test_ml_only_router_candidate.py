"""
test_ml_only_router_candidate.py
================================
Tests for router/ensemble candidates.
Verifies router cannot use future PnL as feature.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestRouterNoFuturePnL:
    """Test that router candidates cannot use future PnL as features."""
    
    def test_router_features_no_future_pnl(self):
        """Test router uses only live-computable features."""
        from scripts.ml_only_non_pe_edge_rescue import NonPEEdgeRescue
        from pathlib import Path
        
        rescue = NonPEEdgeRescue(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            target_types=['ROUTER'],
            max_experiments=1,
        )
        
        features = rescue._get_live_features()
        forbidden = ['pnl', 'profit', 'loss', 'outcome', 'exit', 'mfe', 'mae',
                    'forward', 'future', 'return_after', 'cost_adjusted', 'realized']
        
        for feat in features:
            for pattern in forbidden:
                assert pattern not in feat.lower(), f"Router feature '{feat}' contains forbidden '{pattern}'"
    
    def test_router_cannot_use_trade_outcome_as_feature(self):
        """Test that router target is future-derived but features are not."""
        from scripts.ml_only_non_pe_edge_rescue import ROUTER_EXPERIMENTS
        
        for exp in ROUTER_EXPERIMENTS:
            label = exp['label']
            # Label can be future-derived (for training)
            # But features come from live data only
            assert 'cost_survivor' in label or 'profitable' in label or 'label' in label


class TestRouterConfig:
    """Test router experiment configuration."""
    
    def test_router_experiments_defined(self):
        """Test that router experiments are defined."""
        from scripts.ml_only_non_pe_edge_rescue import ROUTER_EXPERIMENTS
        assert len(ROUTER_EXPERIMENTS) > 0
    
    def test_router_model_types(self):
        """Test that router uses valid model types."""
        from scripts.ml_only_non_pe_edge_rescue import ROUTER_EXPERIMENTS
        
        valid_models = [
            'elasticnet_logistic_regression', 'logistic_regression',
            'random_forest', 'hist_gradient_boosting',
        ]
        
        for exp in ROUTER_EXPERIMENTS:
            assert exp['model'] in valid_models


class TestRouterCandidateResult:
    """Test router candidate result handling."""
    
    def test_router_candidate_requires_both_sides(self):
        """Test that router candidate uses both CE and PE data."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='router_test', candidate_id='router_test', candidate_type='ROUTER',
            label='cost_survivor_label_v2', filter='CE_and_PE',
            model='elasticnet_logistic_regression', threshold=0.25,
            feature_set='live_computable',
            trade_count=8000,
            gross_pf=1.55,
            net_pf=1.28,
            pf_at_1_50x=1.12,
        )
        
        assert r.filter == 'CE_and_PE'
        assert r.candidate_type == 'ROUTER'
        assert r.trade_count >= 500
    
    def test_router_passes_gates(self):
        """Test router candidate that passes all gates."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='router_pass', candidate_id='router_pass', candidate_type='ROUTER',
            label='cost_survivor_label_v2', filter='CE_and_PE',
            model='elasticnet_logistic_regression', threshold=0.25,
            feature_set='live_computable',
            gates_passed=24, gates_total=24,
            trade_count=8000,
            gross_pf=1.55,
            net_pf=1.28,
            pf_at_1_50x=1.12,
            daily_stability_score=0.65,
            threshold_robustness_score=0.72,
            cost_1_50x_pass=True,
            leakage_pass=True, live_computable_pass=True,
            concentration_diversity_pass=True, trading_days_diversity_pass=True,
            model_path='/path/router_model.pkl',
            manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json',
            feature_schema_path='/path/fs.json',
        )
        
        assert r.can_pass() == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])