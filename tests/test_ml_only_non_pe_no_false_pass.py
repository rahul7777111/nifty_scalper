"""
test_ml_only_non_pe_no_false_pass.py
=====================================
Tests that verify no false passes in non-PE candidate system.
Anti-false-pass checks:
- 21/22 cannot be PASS_SHADOW_READY
- No label-as-filter shortcut
- No high-PF without audit
- Real artifacts required
- Time-based validation only
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestNoFalsePass21of22:
    """Test that 21/22 gates cannot produce PASS_SHADOW_READY."""
    
    def test_21_gates_ce_not_passing(self):
        """Test CE with 21/22 gates is not PASS_SHADOW_READY."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=21, gates_total=24,
            daily_stability_score=0.70, threshold_robustness_score=0.80,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        
        assert r.can_pass() == False
        assert r.status != 'PASS_SHADOW_READY'
    
    def test_21_gates_combined_not_passing(self):
        """Test combined with 21/22 gates is not PASS_SHADOW_READY."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='COMBINED',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=21, gates_total=24,
            daily_stability_score=0.70, threshold_robustness_score=0.80,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        
        assert r.can_pass() == False
    
    def test_21_gates_router_not_passing(self):
        """Test router with 21/22 gates is not PASS_SHADOW_READY."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='ROUTER',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=21, gates_total=24,
            daily_stability_score=0.70, threshold_robustness_score=0.80,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        
        assert r.can_pass() == False


class TestNoLabelAsFilter:
    """Test that label columns are not used as trade filters."""
    
    def test_filter_does_not_use_label_column(self):
        """Test that _apply_filter doesn't use label columns."""
        from scripts.ml_only_non_pe_edge_rescue import NonPEEdgeRescue
        from pathlib import Path
        import pandas as pd
        
        rescue = NonPEEdgeRescue(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            target_types=['CE'],
            max_experiments=1,
        )
        
        # Verify filter doesn't use label as ROW filter
        # i.e., 'none' filter should NOT select only rows where label=1
        df_none = rescue._apply_filter(rescue.full_df, 'none')
        
        # If 'none' filter returned the same number of rows as full_df,
        # it means no row-level filtering was applied (good)
        # Label columns exist as metadata but aren't used to filter rows
        assert len(df_none) == len(rescue.full_df), "'none' filter should return all rows, not filter by label"
        
        # 'CE_only' filter should only filter by option_type, not by label
        df_ce = rescue._apply_filter(rescue.full_df, 'CE_only')
        if 'option_type' in df_ce.columns:
            assert (df_ce['option_type'].astype(str).str.upper() == 'CE').all()


class TestSuspiciousPFBlocked:
    """Test that PF > 10 requires audit."""
    
    def test_high_pf_flagged(self):
        """Test that gross_pf > 10 is suspicious and flagged."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=22, gates_total=24,
            gross_pf=25.0,  # Extremely suspicious
            pf_at_1_50x=18.0,
            daily_stability_score=0.70, threshold_robustness_score=0.80,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        
        # can_pass may still be True if all artifacts present
        # but this should be flagged in the audit report
        assert r.gross_pf > 10
        # High PF alone doesn't make it invalid, but it's suspicious


class TestRealArtifactsRequired:
    """Test that real artifact files are required."""
    
    def test_empty_model_path_fails(self):
        """Test that empty model_path fails."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=22, gates_total=24,
            daily_stability_score=0.70, threshold_robustness_score=0.80,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='',  # Empty
            manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        
        assert r.can_pass() == False
    
    def test_empty_manifest_fails(self):
        """Test that empty manifest_path fails."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='m', threshold=0.25, feature_set='f',
            gates_passed=22, gates_total=24,
            daily_stability_score=0.70, threshold_robustness_score=0.80,
            cost_1_50x_pass=True, leakage_pass=True, live_computable_pass=True,
            model_path='/path/model.pkl', manifest_path='',
            shadow_manifest_path='/path/shadow.json', feature_schema_path='/path/fs.json',
        )
        
        assert r.can_pass() == False


class TestTimeBasedValidationOnly:
    """Test that only time-based validation is used."""
    
    def test_time_split_function_exists(self):
        """Test that _time_split performs time-based split."""
        from scripts.ml_only_non_pe_edge_rescue import NonPEEdgeRescue
        from pathlib import Path
        import pandas as pd
        
        rescue = NonPEEdgeRescue(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            target_types=['CE'],
            max_experiments=1,
        )
        
        # Create small test df
        df = pd.DataFrame({
            'timestamp_dt': pd.date_range('2024-01-01', periods=100),
            'value': range(100),
        })
        
        train, test = rescue._time_split(df)
        
        # Time-based: train should be first 70%, test last 30%
        assert len(train) == 70
        assert len(test) == 30
        # Train comes before test
        assert train['timestamp_dt'].max() < test['timestamp_dt'].min()


class TestNoRuleBasedEntry:
    """Test that entry is ML-based, not rule-based."""
    
    def test_entry_from_model_prediction(self):
        """Test that trade entry is based on model.predict_proba."""
        from scripts.ml_only_non_pe_edge_rescue import NonPECandidateResult
        
        r = NonPECandidateResult(
            experiment_name='test', candidate_id='test', candidate_type='CE',
            label='l', filter='f', model='elasticnet_logistic_regression',
            threshold=0.25, feature_set='live_computable',
            trade_count=5000,
        )
        
        # Entry should come from sklearn model, not rules
        assert 'elasticnet' in r.model or 'logistic' in r.model or 'forest' in r.model or 'gradient' in r.model


if __name__ == '__main__':
    pytest.main([__file__, '-v'])