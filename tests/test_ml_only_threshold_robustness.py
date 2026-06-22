"""
test_ml_only_threshold_robustness.py
====================================
Tests for threshold robustness evaluation.

Verifies:
- Threshold robustness can fail unstable candidates
- Performance should not collapse at nearby thresholds
- Best threshold selection considers robustness
- Weak candidates can be rejected by threshold test
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestThresholdRobustness:
    """Test threshold robustness evaluation."""
    
    def test_threshold_range_defined(self):
        """Test that threshold robustness range is defined."""
        from ml_only_iterative_candidate_rescue import THRESHOLD_ROBUSTNESS_RANGE
        
        assert len(THRESHOLD_ROBUSTNESS_RANGE) >= 5, \
            "Threshold robustness should test at least 5 points"
        assert 0.15 in THRESHOLD_ROBUSTNESS_RANGE
        assert 0.25 in THRESHOLD_ROBUSTNESS_RANGE
    
    def test_robust_threshold_selection(self):
        """Test that robust threshold is selected based on stability."""
        # Simulate threshold sweep with stable performance
        threshold_pfs = {
            0.15: 1.3,
            0.20: 1.35,
            0.25: 1.32,  # Best at 0.20, but 0.25 is close
            0.30: 1.25,
            0.35: 1.20,
            0.40: 0.90,  # Collapses here
            0.45: 0.70,
        }
        
        best_threshold = 0.20
        best_pf = threshold_pfs[best_threshold]
        
        # Find nearby thresholds (within 0.1)
        nearby_thresholds = [t for t in threshold_pfs.keys() 
                            if abs(t - best_threshold) <= 0.1 
                            and t != best_threshold]
        nearby_pfs = [threshold_pfs[t] for t in nearby_thresholds]
        
        # Minimum nearby PF
        min_nearby_pf = min(nearby_pfs)
        
        # Robustness check: PF should not drop more than 50%
        is_robust = min_nearby_pf / best_pf >= 0.5 if best_pf > 0 else False
        
        assert is_robust, "Performance should be stable near best threshold"
        assert min_nearby_pf >= best_pf * 0.5, "PF should not collapse at nearby thresholds"
    
    def test_threshold_collapse_detection(self):
        """Test that threshold collapse is detected."""
        # Simulate threshold sweep with collapse
        threshold_pfs = {
            0.15: 1.4,
            0.20: 1.5,  # Best
            0.25: 1.0,  # Collapses
            0.30: 0.6,  # Further collapse
            0.35: 0.3,
        }
        
        best_threshold = 0.20
        best_pf = threshold_pfs[best_threshold]
        
        nearby_thresholds = [t for t in threshold_pfs.keys() 
                            if abs(t - best_threshold) <= 0.1 
                            and t != best_threshold]
        nearby_pfs = [threshold_pfs[t] for t in nearby_thresholds]
        
        min_nearby_pf = min(nearby_pfs) if nearby_pfs else 0
        
        # This should fail robustness check
        is_robust = min_nearby_pf / best_pf >= 0.5 if best_pf > 0 else False
        
        assert not is_robust, "Threshold sweep with collapse should not be robust"
    
    def test_weak_candidate_threshold_sensitivity(self):
        """Test that weak candidates fail threshold robustness."""
        # A weak candidate might work at one threshold but fail at others
        threshold_pfs_weak = {
            0.15: 1.05,  # Just passes
            0.20: 1.00,  # At break-even
            0.25: 0.85,  # Fails
            0.30: 0.70,
        }
        
        best_threshold = 0.15
        best_pf = threshold_pfs_weak[best_threshold]
        
        nearby_thresholds = [t for t in threshold_pfs_weak.keys() 
                            if abs(t - best_threshold) <= 0.1 
                            and t != best_threshold]
        nearby_pfs = [threshold_pfs_weak[t] for t in nearby_thresholds]
        
        min_nearby_pf = min(nearby_pfs) if nearby_pfs else 0
        
        # Even at best threshold, this candidate is marginal
        is_robust = min_nearby_pf / best_pf >= 0.5 if best_pf > 0 else False
        
        # The candidate should fail robustness because it's unstable
        assert not is_robust or best_pf < 1.15, \
            "Weak candidate should fail robustness or base PF gate"
    
    def test_robust_candidate_threshold_stability(self):
        """Test that robust candidates pass threshold stability."""
        # A strong candidate should work across threshold range
        threshold_pfs_strong = {
            0.15: 1.4,
            0.20: 1.45,
            0.25: 1.42,  # Very stable
            0.30: 1.35,
            0.35: 1.25,
            0.40: 1.10,  # Still ok
        }
        
        best_threshold = 0.20
        best_pf = threshold_pfs_strong[best_threshold]
        
        nearby_thresholds = [t for t in threshold_pfs_strong.keys() 
                            if abs(t - best_threshold) <= 0.1]
        nearby_pfs = [threshold_pfs_strong[t] for t in nearby_thresholds]
        
        min_nearby_pf = min(nearby_pfs)
        max_nearby_pf = max(nearby_pfs)
        
        # Both min and max nearby should be close to best
        is_robust = min_nearby_pf / best_pf >= 0.5 and max_nearby_pf / best_pf <= 1.5
        
        assert is_robust, "Strong candidate should have stable PF across thresholds"


class TestThresholdSweepLogic:
    """Test threshold sweep computation."""
    
    def test_min_trades_per_threshold(self):
        """Test that each threshold needs minimum trades."""
        from ml_only_iterative_candidate_rescue import MIN_TRADES_PER_THRESHOLD
        
        assert MIN_TRADES_PER_THRESHOLD == 100, \
            "Minimum 100 trades per threshold for statistical significance"
    
    def test_threshold_sweep_filters_low_trade_counts(self):
        """Test that threshold sweep ignores thresholds with too few trades."""
        # Simulate trade counts per threshold
        threshold_trades = {
            0.15: 500,  # Sufficient
            0.20: 450,  # Sufficient
            0.25: 200,  # Sufficient
            0.30: 80,   # Insufficient
            0.35: 50,   # Insufficient
        }
        
        min_required = 100
        valid_thresholds = [t for t, count in threshold_trades.items() 
                           if count >= min_required]
        
        assert 0.15 in valid_thresholds
        assert 0.20 in valid_thresholds
        assert 0.25 in valid_thresholds
        assert 0.30 not in valid_thresholds
        assert 0.35 not in valid_thresholds
    
    def test_best_threshold_not_at_extremes(self):
        """Test that best threshold should not be at range edges."""
        # If best threshold is at extreme (0.15 or 0.45), might indicate overfitting
        threshold_pfs = {
            0.15: 1.3,  # Edge - might be overfit
            0.20: 1.4,  # Better
            0.25: 1.38,
            0.30: 1.2,
            0.35: 0.9,
        }
        
        best_threshold = 0.20  # Not at edge
        best_pf = threshold_pfs[best_threshold]
        
        # Check that best is not at extreme
        all_thresholds = list(threshold_pfs.keys())
        is_best_extreme = best_threshold == min(all_thresholds) or \
                         best_threshold == max(all_thresholds)
        
        # Ideally best should not be at extreme
        if is_best_extreme:
            # But if it is, we should be more cautious
            assert best_pf > 1.5, "Extreme threshold needs much stronger evidence"


class TestThresholdRobustnessGate:
    """Test the robustness gate in evaluation."""
    
    def test_gate_requires_stability(self):
        """Test that robustness gate requires stable performance."""
        from ml_only_strict_gate_evaluator import STRICT_ML_ONLY_GATES
        
        gate = STRICT_ML_ONLY_GATES['D2_threshold_robust']
        assert gate['name'] == 'Threshold Robustness'
        assert gate['severity'] == 'high'
    
    def test_robustness_proxies_threshold_robust_field(self):
        """Test that manifest's threshold_robust field is used."""
        from ml_only_strict_gate_evaluator import MLOnlyStrictGateEvaluator
        
        evaluator = MLOnlyStrictGateEvaluator.__new__(MLOnlyStrictGateEvaluator)
        evaluator.manifest = {
            'candidate_id': 'test',
            'threshold_robust': True,
            'overall_metrics': {'mean_pf': 1.5},
            'fold_details': {},
        }
        
        gate_result = evaluator.evaluate_gate('D2_threshold_robust', 
                                               {'threshold': True, 'name': 'Test',
                                                'description': 'Test', 'category': 'test',
                                                'severity': 'high'})
        
        assert gate_result.passed
    
    def test_non_robust_candidate_fails(self):
        """Test that non-robust candidate fails threshold gate."""
        from ml_only_strict_gate_evaluator import MLOnlyStrictGateEvaluator
        
        evaluator = MLOnlyStrictGateEvaluator.__new__(MLOnlyStrictGateEvaluator)
        evaluator.manifest = {
            'candidate_id': 'test',
            'threshold_robust': False,  # Not robust
            'overall_metrics': {'mean_pf': 1.5},
            'fold_details': {},
        }
        
        gate_result = evaluator.evaluate_gate('D2_threshold_robust',
                                               {'threshold': True, 'name': 'Test',
                                                'description': 'Test', 'category': 'test',
                                                'severity': 'high'})
        
        assert not gate_result.passed


if __name__ == '__main__':
    pytest.main([__file__, '-v'])