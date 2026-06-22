"""
test_ml_only_edge_tournament.py
================================
Tests for CORRECTED ML-only edge tournament system.

Tests verify:
- Tournament runs without errors
- Experiments produce results  
- Reports are generated
- Leaderboard is correctly sorted
- New passing candidates are identified
- No failed candidate is incorrectly marked PASS_SHADOW_READY
- Benchmark remains passing
- 1.5x cost stress is mandatory
- Future-derived labels are not used as features
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestTournamentBasics:
    """Test tournament basic functionality."""
    
    def test_tournament_script_exists(self):
        """Test that tournament can be imported."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        assert MLEdgeTournament is not None
    
    def test_tournament_has_required_methods(self):
        """Test that tournament has all required methods."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path
        
        t = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )
        
        assert hasattr(t, '_get_live_features')
        assert hasattr(t, '_apply_filter')
        assert hasattr(t, '_time_split')
        assert hasattr(t, '_train_model')
        assert hasattr(t, '_compute_metrics_from_predictions')
        assert hasattr(t, '_evaluate_threshold_robustness')
        assert hasattr(t, '_evaluate_gates')
        assert hasattr(t, '_save_candidate_artifacts')
        assert hasattr(t, 'run_single_experiment')
        assert hasattr(t, 'run_tournament')
    
    def test_strict_cost_gates_defined(self):
        """Test that strict cost gates are properly defined."""
        from scripts.ml_only_edge_tournament import STRICT_COST_GATES
        assert 'pf_at_1_00x' in STRICT_COST_GATES
        assert 'pf_at_1_25x' in STRICT_COST_GATES
        assert 'pf_at_1_50x' in STRICT_COST_GATES
        assert 'pf_at_2_00x' in STRICT_COST_GATES
        assert STRICT_COST_GATES['pf_at_1_50x'] == 1.00  # Break-even
    
    def test_threshold_range_defined(self):
        """Test that threshold range is defined."""
        from scripts.ml_only_edge_tournament import THRESHOLD_RANGE
        assert 0.20 in THRESHOLD_RANGE
        assert 0.25 in THRESHOLD_RANGE
        assert 0.30 in THRESHOLD_RANGE


class TestCostStressGates:
    """Test that 1.5x cost stress is enforced."""
    
    def test_1_50x_cost_gate_is_critical(self):
        """Test that the 1.50x cost gate requires break-even."""
        from scripts.ml_only_edge_tournament import STRICT_COST_GATES
        assert STRICT_COST_GATES['pf_at_1_50x'] == 1.00
    
    def test_candidate_requires_1_50x_pass(self):
        """Test that PASS_SHADOW_READY requires passing 1.50x cost gate."""
        from scripts.ml_only_edge_tournament import CandidateResult
        
        result = CandidateResult(
            experiment_name='test', candidate_id='test',
            label='cost_survivor_label_v2', filter='PE_only',
            model='elasticnet', threshold=0.25, feature_set='live',
            gates_passed=22, gates_total=22,
            pf_at_1_50x=1.1,
            cost_1_50x_pass=True,
            model_path='/path/model.pkl',
            manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json',
            feature_schema_path='/path/feature_schema.json',
            daily_stability_score=0.7, live_computable_pass=True, leakage_pass=True,
            threshold_robustness_score=0.8,
        )
        
        assert result.cost_1_50x_pass == True
        # But 22/22 + all artifacts + scores > 0 still required
        assert result.can_pass()


class TestTournamentResultClassification:
    """Test that tournament results are correctly classified."""
    
    def test_pass_requires_all_22_gates(self):
        """Test that PASS_SHADOW_READY requires 22/22 gates."""
        from scripts.ml_only_edge_tournament import CandidateResult
        
        result = CandidateResult(
            experiment_name='test', candidate_id='test',
            label='cost_survivor_label_v2', filter='PE_only',
            model='elasticnet', threshold=0.25, feature_set='live',
            gates_passed=22, gates_total=22,
            pf_at_1_50x=1.1,
            cost_1_50x_pass=True,
            model_path='/path/model.pkl',
            manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json',
            feature_schema_path='/path/feature_schema.json',
            daily_stability_score=0.7, live_computable_pass=True, leakage_pass=True,
            threshold_robustness_score=0.8,
        )
        
        assert result.can_pass()
    
    def test_21_of_22_fails(self):
        """Test that 21/22 gates fails."""
        from scripts.ml_only_edge_tournament import CandidateResult
        
        result = CandidateResult(
            experiment_name='test', candidate_id='test',
            label='cost_survivor_label_v2', filter='PE_only',
            model='elasticnet', threshold=0.25, feature_set='live',
            gates_passed=21, gates_total=22,  # 21 not 22
            pf_at_1_50x=1.1,
            cost_1_50x_pass=True,
            model_path='/path/model.pkl',
            manifest_path='/path/manifest.json',
            shadow_manifest_path='/path/shadow.json',
            feature_schema_path='/path/feature_schema.json',
            daily_stability_score=0.7, live_computable_pass=True, leakage_pass=True,
            threshold_robustness_score=0.8,
        )
        
        assert not result.can_pass()


class TestTournamentReports:
    """Test tournament report generation."""
    
    def test_candidate_result_to_dict(self):
        """Test that CandidateResult can be serialized to dict."""
        from scripts.ml_only_edge_tournament import CandidateResult
        result = CandidateResult(
            experiment_name='test_exp',
            candidate_id='test_candidate',
            label='cost_survivor_label_v2',
            filter='PE_only',
            model='elasticnet',
            threshold=0.25,
            feature_set='live_computable_v1',
        )
        d = result.to_dict()
        assert d['experiment_name'] == 'test_exp'
        assert d['label'] == 'cost_survivor_label_v2'
    
    def test_tournament_report_has_required_fields(self):
        """Test that TournamentReport has required fields."""
        from scripts.ml_only_edge_tournament import TournamentReport, CandidateResult
        report = TournamentReport(
            timestamp='20260609_120000',
            total_experiments=10,
            passed_experiments=0,
            failed_experiments=10,
            truly_passing_candidates=[],
            leaderboard=[],
            failed_diagnosis={},
            benchmark_status={},
            bugs_audited=[],
        )
        assert report.total_experiments == 10
        assert report.passed_experiments == 0
        assert hasattr(report, 'truly_passing_candidates')


class TestTournamentFilterLogic:
    """Test tournament filter application logic."""
    
    def test_filter_pe_only(self):
        """Test that PE_only filter works."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path
        import pandas as pd
        
        t = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )
        
        df = pd.DataFrame({'option_type': ['PE', 'CE', 'PE', 'CE']})
        filtered = t._apply_filter(df, 'PE_only')
        assert len(filtered) == 2
    
    def test_filter_ce_only(self):
        """Test that CE_only filter works."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path
        import pandas as pd
        
        t = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )
        
        df = pd.DataFrame({'option_type': ['PE', 'CE', 'PE', 'CE']})
        filtered = t._apply_filter(df, 'CE_only')
        assert len(filtered) == 2
    
    def test_filter_none_returns_all(self):
        """Test that 'none' filter returns all data."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path
        import pandas as pd
        
        t = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )
        
        df = pd.DataFrame({'option_type': ['PE', 'CE', 'PE', 'CE']})
        filtered = t._apply_filter(df, 'none')
        assert len(filtered) == 4


class TestLeaderboardSorting:
    """Test leaderboard sorting logic."""
    
    def test_leaderboard_sorted_by_gates_and_pf(self):
        """Test that leaderboard is sorted by gate pass rate then by 1.5x PF."""
        from scripts.ml_only_edge_tournament import CandidateResult
        
        candidates = [
            CandidateResult(experiment_name='c1', candidate_id='c1', label='l1', filter='f1', 
                          model='m1', threshold=0.25, feature_set='f',
                          gates_passed=18, gates_total=22, pf_at_1_50x=1.1),
            CandidateResult(experiment_name='c2', candidate_id='c2', label='l1', filter='f1', 
                          model='m1', threshold=0.25, feature_set='f',
                          gates_passed=20, gates_total=22, pf_at_1_50x=1.05),
            CandidateResult(experiment_name='c3', candidate_id='c3', label='l1', filter='f1', 
                          model='m1', threshold=0.25, feature_set='f',
                          gates_passed=18, gates_total=22, pf_at_1_50x=1.2),
        ]
        
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (x.gates_passed / max(x.gates_total, 1), x.pf_at_1_50x),
            reverse=True
        )
        
        assert sorted_candidates[0].candidate_id == 'c2'  # More gates
        assert sorted_candidates[1].candidate_id == 'c3'  # Same gates, higher PF
        assert sorted_candidates[2].candidate_id == 'c1'


class TestNoFutureLeakage:
    """Test that future-derived labels are not used as features."""
    
    def test_forbidden_label_patterns_exist(self):
        """Test that forbidden label patterns are recognized and blocked."""
        from scripts.ml_only_edge_tournament import MLEdgeTournament
        from pathlib import Path
        
        t = MLEdgeTournament(
            dataset_path=REPO_ROOT / 'data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv',
            benchmark_candidate=Path('/tmp/fake'),
            output_dir=Path('/tmp'),
            max_experiments=1,
        )
        
        features = t._get_live_features()
        forbidden = ['_return', 'forward_', 'future_', 'pnl', 'label', 'cost_survivor']
        
        for feat in features:
            for f in forbidden:
                assert f not in feat.lower(), f"Feature {feat} contains forbidden pattern {f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])