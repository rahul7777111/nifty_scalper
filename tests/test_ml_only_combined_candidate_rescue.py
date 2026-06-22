"""
test_ml_only_combined_candidate_rescue.py
==========================================
Tests for combined CE+PE candidates.
Verifies no option_side target leakage.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestCombinedNoSideLeakage:
    """Test that combined candidates don't leak option side info."""
    
    def test_combined_filter_none_or_valid(self):
        """Test that combined experiments use 'none' or valid regime filters."""
        from scripts.ml_only_non_pe_edge_rescue import COMBINED_EXPERIMENTS
        
        valid_filters = ['none', 'ATM_near', 'high_volume', 'DTE_7_30']
        
        for exp in COMBINED_EXPERIMENTS:
            assert exp['filter'] in valid_filters, f"Invalid combined filter: {exp['filter']}"
    
    def test_combined_no_option_type_in_target(self):
        """Test that combined label doesn't encode option_type."""
        from scripts.ml_only_non_pe_edge_rescue import COMBINED_EXPERIMENTS
        
        for exp in COMBINED_EXPERIMENTS:
            label = exp['label']
            # Label should be about profitability, not option side
            assert 'option_type' not in label.lower()
            assert '_ce_' not in label.lower()
            assert '_pe_' not in label.lower()
    
    def test_combined_uses_both_sides(self):
        """Test that combined filter 'none' includes both CE and PE."""
        from scripts.ml_only_non_pe_edge_rescue import NonPEEdgeRescue
        from pathlib import Path
        import pandas as pd
        
        rescue = NonPEEdgeRescue(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            target_types=['COMBINED'],
            max_experiments=1,
        )
        
        # 'none' filter should return all data
        df = rescue._apply_filter(rescue.full_df, 'none')
        if 'option_type' in df.columns:
            has_ce = (df['option_type'].astype(str).str.upper() == 'CE').any()
            has_pe = (df['option_type'].astype(str).str.upper() == 'PE').any()
            assert has_ce and has_pe, "Combined 'none' filter should include both CE and PE"


class TestCombinedMetrics:
    """Test combined candidate metrics."""
    
    def test_combined_result_has_both_sides(self):
        """Test that combined result represents both CE and PE."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='combo_test', candidate_id='combo_test', candidate_type='COMBINED',
            label='cost_survivor_label_v2', filter='none',
            model='elasticnet_logistic_regression', threshold=0.25,
            feature_set='live_computable',
            trade_count=10000,
            gross_pf=1.4,
            net_pf=1.15,
            pf_at_1_50x=1.02,
        )
        
        assert r.candidate_type == 'COMBINED'
        assert r.filter == 'none'
        assert r.trade_count >= 500


class TestCombinedRouterEnsemble:
    """Test combined/router ensemble candidates."""
    
    def test_router_candidates_exist(self):
        """Test that router experiments are defined."""
        from scripts.ml_only_non_pe_edge_rescue import ROUTER_EXPERIMENTS
        assert len(ROUTER_EXPERIMENTS) > 0
    
    def test_router_filter_is_ce_and_pe(self):
        """Test that router uses both CE and PE data."""
        from scripts.ml_only_non_pe_edge_rescue import ROUTER_EXPERIMENTS
        
        for exp in ROUTER_EXPERIMENTS:
            assert exp['filter'] == 'CE_and_PE', f"Router filter should be 'CE_and_PE', got '{exp['filter']}'"
    
    def test_router_candidate_type(self):
        """Test router candidate type."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='router_test', candidate_id='router_test', candidate_type='ROUTER',
            label='cost_survivor_label_v2', filter='CE_and_PE',
            model='elasticnet_logistic_regression', threshold=0.25,
            feature_set='live_computable',
        )
        
        assert r.candidate_type == 'ROUTER'


class TestCombinedNoFuturePnL:
    """Test that combined candidates don't use future PnL features."""
    
    def test_combined_features_no_future_pnl(self):
        """Test that combined candidate uses only live features."""
        from scripts.ml_only_non_pe_edge_rescue import NonPEEdgeRescue
        from pathlib import Path
        
        rescue = NonPEEdgeRescue(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            target_types=['COMBINED'],
            max_experiments=1,
        )
        
        features = rescue._get_live_features()
        forbidden = ['pnl', 'profit', 'loss', 'outcome', 'exit', 'mfe', 'mae',
                    'forward', 'future', 'return_after', 'cost_adjusted']
        
        for feat in features:
            for pattern in forbidden:
                assert pattern not in feat.lower(), f"Feature '{feat}' contains forbidden '{pattern}'"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])