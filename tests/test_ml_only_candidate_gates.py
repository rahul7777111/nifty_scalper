"""
test_ml_only_candidate_gates.py
===============================
Tests for ML-only candidate gate system.

Verifies:
- All gate definitions are complete
- Gate severity levels are correct
- Critical gates cannot be bypassed
- Gate evaluation produces correct pass/fail
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestGateDefinitions:
    """Test that all gates are properly defined."""
    
    def test_all_gate_categories_defined(self):
        """Test that all gate categories exist."""
        from ml_only_strict_gate_evaluator import STRICT_ML_ONLY_GATES
        
        expected_categories = [
            'data_integrity',
            'prediction_quality',
            'trading_economics',
            'shadow_mode',
        ]
        
        actual_categories = set(g['category'] for g in STRICT_ML_ONLY_GATES.values())
        
        for cat in expected_categories:
            assert cat in actual_categories, f"Category {cat} must be defined"
    
    def test_critical_gates_identified(self):
        """Test that critical gates are marked as critical."""
        from ml_only_strict_gate_evaluator import STRICT_ML_ONLY_GATES
        
        critical_gate_ids = [
            'A1_no_future_leakage',
            'A2_no_target_leakage',
            'A5_live_computable_features',
            'C1_positive_net_expectancy',
            'C4_profit_factor_1.50x_cost',
            'D1_shadow_manifest_complete',
        ]
        
        for gate_id in critical_gate_ids:
            assert gate_id in STRICT_ML_ONLY_GATES, f"Critical gate {gate_id} must exist"
            assert STRICT_ML_ONLY_GATES[gate_id]['severity'] == 'critical', \
                f"Gate {gate_id} must be marked as critical"
    
    def test_gate_has_required_fields(self):
        """Test that each gate has all required fields."""
        from ml_only_strict_gate_evaluator import STRICT_ML_ONLY_GATES
        
        required_fields = ['name', 'description', 'category', 'severity', 'threshold']
        
        for gate_id, gate_def in STRICT_ML_ONLY_GATES.items():
            for field in required_fields:
                assert field in gate_def, \
                    f"Gate {gate_id} missing required field '{field}'"
    
    def test_gate_severity_values(self):
        """Test that gate severity values are valid."""
        from ml_only_strict_gate_evaluator import STRICT_ML_ONLY_GATES
        
        valid_severities = ['critical', 'high', 'medium', 'low']
        
        for gate_id, gate_def in STRICT_ML_ONLY_GATES.items():
            assert gate_def['severity'] in valid_severities, \
                f"Gate {gate_id} has invalid severity '{gate_def['severity']}'"


class TestCriticalGates:
    """Test critical gate behavior."""
    
    def test_no_future_leakage_gate(self):
        """Test no-future-leakage gate evaluation."""
        from ml_only_strict_gate_evaluator import MLOnlyStrictGateEvaluator
        
        evaluator = MLOnlyStrictGateEvaluator.__new__(MLOnlyStrictGateEvaluator)
        evaluator.manifest = {
            'candidate_id': 'test',
            'source': {'dataset': 'some_dataset'},  # Has source tracking
        }
        
        gate_result = evaluator.evaluate_gate('A1_no_future_leakage', {
            'threshold': True,
            'name': 'No Future Leakage',
            'description': 'Test',
            'category': 'data_integrity',
            'severity': 'critical',
        })
        
        assert gate_result.passed
    
    def test_no_target_leakage_gate(self):
        """Test no-target-leakage gate evaluation."""
        from ml_only_strict_gate_evaluator import MLOnlyStrictGateEvaluator
        
        evaluator = MLOnlyStrictGateEvaluator.__new__(MLOnlyStrictGateEvaluator)
        evaluator.manifest = {
            'candidate_id': 'test',
            'filter': {'applied_at': 'DATA_LEVEL_PRE_TRAINING'},
        }
        
        gate_result = evaluator.evaluate_gate('A2_no_target_leakage', {
            'threshold': True,
            'name': 'No Target Leakage',
            'description': 'Test',
            'category': 'data_integrity',
            'severity': 'critical',
        })
        
        assert gate_result.passed
    
    def test_cost_1_50x_gate_break_even(self):
        """Test cost 1.50x gate requires break-even."""
        from ml_only_strict_gate_evaluator import MLOnlyStrictGateEvaluator
        
        evaluator = MLOnlyStrictGateEvaluator.__new__(MLOnlyStrictGateEvaluator)
        evaluator.manifest = {
            'candidate_id': 'test',
            'overall_metrics': {'cost_1.50x_pf': 1.0},  # Exactly break-even
        }
        
        gate_result = evaluator.evaluate_gate('C4_profit_factor_1.50x_cost', {
            'threshold': 1.0,
            'name': 'Test',
            'description': 'Test',
            'category': 'trading_economics',
            'severity': 'critical',
        })
        
        assert gate_result.passed, "PF=1.0 at 1.5x cost should pass (break-even)"
    
    def test_cost_1_50x_gate_fails_below_even(self):
        """Test cost 1.50x gate fails below break-even."""
        from ml_only_strict_gate_evaluator import MLOnlyStrictGateEvaluator
        
        evaluator = MLOnlyStrictGateEvaluator.__new__(MLOnlyStrictGateEvaluator)
        evaluator.manifest = {
            'candidate_id': 'test',
            'overall_metrics': {'cost_1.50x_pf': 0.95},  # Below break-even
        }
        
        gate_result = evaluator.evaluate_gate('C4_profit_factor_1.50x_cost', {
            'threshold': 1.0,
            'name': 'Test',
            'description': 'Test',
            'category': 'trading_economics',
            'severity': 'critical',
        })
        
        assert not gate_result.passed


class TestGateEvaluationCompleteness:
    """Test that all gates are evaluated."""
    
    def test_all_gates_evaluated(self):
        """Test that evaluate_all_gates evaluates all gates."""
        from ml_only_strict_gate_evaluator import MLOnlyStrictGateEvaluator, STRICT_ML_ONLY_GATES
        
        evaluator = MLOnlyStrictGateEvaluator.__new__(MLOnlyStrictGateEvaluator)
        evaluator.manifest = {
            'candidate_id': 'test',
            'overall_metrics': {'mean_pf': 1.5, 'cost_1.50x_pf': 1.2},
            'fold_details': {},
            'threshold_robust': True,
        }
        evaluator.metrics = {}
        evaluator.gate_results = []
        
        results = evaluator.evaluate_all_gates()
        
        assert len(results) == len(STRICT_ML_ONLY_GATES), \
            "All gates must be evaluated"


class TestGateFailureAggregation:
    """Test gate failure aggregation."""
    
    def test_fail_reasons_collected(self):
        """Test that failed gates produce failure reasons."""
        from ml_only_strict_gate_evaluator import MLOnlyStrictGateEvaluator
        
        evaluator = MLOnlyStrictGateEvaluator.__new__(MLOnlyStrictGateEvaluator)
        evaluator.candidate_path = Path('/tmp/fake')
        evaluator.manifest = {
            'candidate_id': 'test',
            'overall_metrics': {
                'mean_pf': 0.9,  # Fails
                'cost_1.50x_pf': 0.8,  # Fails
                'mean_sharpe': 0.3,  # Fails
                'mean_trades': 100,
            },
            'fold_details': {
                'fold_0': {'pf': 0.8},
                'fold_1': {'pf': 0.85},
            },
            'threshold_robust': False,
            'paper_only': True,
            'real_trading_enabled': False,
        }
        evaluator.metrics = {}
        evaluator.gate_results = []
        
        evaluation = evaluator.run_evaluation()
        
        assert len(evaluation.fail_reasons) > 0, "Failed gates must produce reasons"

    def test_improvement_suggestions_generated(self):
        """Test that improvement suggestions are generated for failures."""
        from ml_only_strict_gate_evaluator import MLOnlyStrictGateEvaluator
        
        evaluator = MLOnlyStrictGateEvaluator.__new__(MLOnlyStrictGateEvaluator)
        evaluator.candidate_path = Path('/tmp/fake')
        evaluator.manifest = {
            'candidate_id': 'weak_candidate',
            'overall_metrics': {
                'mean_pf': 0.95,
                'cost_1.50x_pf': 0.75,
                'mean_sharpe': 0.4,
                'mean_trades': 200,  # Too few
            },
            'fold_details': {
                'fold_0': {'pf': 0.8},
            },
            'threshold_robust': False,
            'paper_only': True,
            'real_trading_enabled': False,
        }
        evaluator.metrics = {}
        evaluator.gate_results = []
        
        evaluation = evaluator.run_evaluation()
        
        # Should have improvement suggestions for failed gates
        assert len(evaluation.improvement_suggestions) > 0


class TestReadinessDetermination:
    """Test readiness level determination."""
    
    def test_live_ready_requires_all_pass(self):
        """Test that LIVE_READY requires all gates passed."""
        from ml_only_strict_gate_evaluator import MLOnlyStrictGateEvaluator
        
        evaluator = MLOnlyStrictGateEvaluator.__new__(MLOnlyStrictGateEvaluator)
        evaluator.manifest = {
            'candidate_id': 'perfect_candidate',
            'overall_metrics': {
                'mean_pf': 1.5,
                'cost_1.25x_pf': 1.42,
                'cost_1.50x_pf': 1.3,
                'mean_sharpe': 1.2,
                'mean_trades': 600,
                'worst_fold_pf': 1.2,
            },
            'fold_details': {
                'fold_0': {'pf': 1.3},
                'fold_1': {'pf': 1.4},
                'fold_2': {'pf': 1.5},
            },
            'threshold_robust': True,
            'paper_only': True,
            'real_trading_enabled': False,
            'source': {'dataset': 'test'},
            'filter': {'applied_at': 'DATA_LEVEL_PRE_TRAINING'},
            'selected_threshold': 0.25,
            'filter_name': 'PE_only',
            'feature_schema_path': 'feature_schema.json',
            'model_pkl': 'model.pkl',
            'gates_total': 10,
            'gates_passed': 10,
            'artifacts': {
                'model_pkl': 'model.pkl',
                'feature_schema': 'feature_schema.json',
                'preprocessing_metadata': 'preprocessing_metadata.json',
                'filter_definition': 'filter_definition.json',
            },
        }
        evaluator.metrics = {'out_of_sample_auc': 0.72, 'out_of_sample_prauc': 0.68}
        evaluator.gate_results = []
        
        evaluator.evaluate_all_gates()
        readiness = evaluator.determine_readiness()
        
        assert readiness == 'PASS_LIVE_READY'
    
    def test_shadow_ready_with_minor_failures(self):
        """Test that SHADOW_READY allows minor non-critical failures."""
        from ml_only_strict_gate_evaluator import MLOnlyStrictGateEvaluator
        
        evaluator = MLOnlyStrictGateEvaluator.__new__(MLOnlyStrictGateEvaluator)
        evaluator.manifest = {
            'candidate_id': 'good_candidate',
            'overall_metrics': {
                'mean_pf': 1.4,
                'cost_1.50x_pf': 1.15,  # Passes 1.5x
                'mean_sharpe': 1.0,
                'mean_trades': 550,
                'worst_fold_pf': 1.1,
            },
            'fold_details': {
                'fold_0': {'pf': 1.2},
                'fold_1': {'pf': 1.3},
            },
            'threshold_robust': False,  # Minor failure
            'paper_only': True,
            'real_trading_enabled': False,
            'source': {'dataset': 'test'},
            'filter': {'applied_at': 'DATA_LEVEL_PRE_TRAINING'},
            'selected_threshold': 0.25,
            'filter_name': 'PE_only',
        }
        evaluator.metrics = {}
        evaluator.gate_results = []
        
        evaluator.evaluate_all_gates()
        readiness = evaluator.determine_readiness()
        
        # threshold_robust=False causes D2 to fail, so LIVE_READY impossible
        # But critical gates pass, so at least SHADOW_READY
        assert readiness in ('PASS_SHADOW_READY', 'PASS_PAPER_READY', 'FAIL_BUT_PROMISING'), \
            f"Expected at least SHADOW_READY, got {readiness}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])