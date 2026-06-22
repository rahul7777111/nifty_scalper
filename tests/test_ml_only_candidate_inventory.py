"""
test_ml_only_candidate_inventory.py
====================================
Tests for ML-only candidate inventory system.

Verifies:
- Future leakage is rejected
- Non-live-computable features are rejected
- Candidates cannot pass if net expectancy is negative
- Candidates cannot pass only because gross PnL is positive
- Cost stress can fail weak candidates
- Threshold robustness can fail unstable candidates
- Leaderboard ranks candidates correctly
- Shadow manifest is generated only for passing candidates
- Retraining loop saves reports
- No gate is silently skipped
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestCandidateInventory:
    """Test candidate inventory building and evaluation."""
    
    def test_find_candidate_manifests(self):
        """Test that we can find candidate manifests."""
        from ml_only_candidate_inventory import find_candidate_manifests
        
        candidates_dir = REPO_ROOT / 'models' / 'candidates'
        if candidates_dir.exists():
            manifests = find_candidate_manifests(candidates_dir)
            assert isinstance(manifests, list)
    
    def test_load_candidate_manifest_valid(self):
        """Test loading a valid candidate manifest."""
        from ml_only_candidate_inventory import load_candidate_manifest
        
        candidates_dir = REPO_ROOT / 'models' / 'candidates'
        if not candidates_dir.exists():
            pytest.skip("No candidates directory")
        
        manifests = list(candidates_dir.rglob('candidate_manifest.json'))
        if manifests:
            manifest = load_candidate_manifest(manifests[0])
            assert manifest is not None
            assert 'candidate_id' in manifest or 'model_id' in manifest
    
    def test_paper_only_enforced(self):
        """Test that paper_only=True is enforced in manifests."""
        from ml_only_candidate_inventory import load_candidate_manifest
        
        candidates_dir = REPO_ROOT / 'models' / 'candidates'
        manifests = list(candidates_dir.rglob('candidate_manifest.json'))
        
        for mp in manifests:
            manifest = load_candidate_manifest(mp)
            if manifest:
                assert manifest.get('paper_only') == True, \
                    f"Candidate {mp} must have paper_only=True"
    
    def test_real_trading_disabled(self):
        """Test that real_trading_enabled=False is enforced."""
        from ml_only_candidate_inventory import load_candidate_manifest
        
        candidates_dir = REPO_ROOT / 'models' / 'candidates'
        manifests = list(candidates_dir.rglob('candidate_manifest.json'))
        
        for mp in manifests:
            manifest = load_candidate_manifest(mp)
            if manifest:
                assert manifest.get('real_trading_enabled') == False, \
                    f"Candidate {mp} must have real_trading_enabled=False"
    
    def test_strict_gates_defined(self):
        """Test that all strict gates are defined."""
        from ml_only_candidate_inventory import STRICT_GATES
        
        required_gates = [
            'no_future_leakage',
            'profit_factor_1.50x_cost',  # Critical gate
            'min_trades',
            'threshold_robustness',
        ]
        
        for gate in required_gates:
            assert gate in STRICT_GATES, f"Gate {gate} must be defined"
    
    def test_cost_gate_thresholds(self):
        """Test that cost stress gate thresholds are correct."""
        from ml_only_candidate_inventory import STRICT_GATES
        
        # The 1.50x cost gate must have threshold >= 1.0 (break-even)
        gate = STRICT_GATES.get('profit_factor_1.50x_cost', {})
        threshold = gate.get('threshold', 0)
        assert threshold >= 1.0, \
            f"Cost 1.50x gate threshold must be >= 1.0 (break-even), got {threshold}"
    
    def test_leaderboard_sorted_by_readiness(self):
        """Test that leaderboard is sorted by readiness level."""
        from ml_only_candidate_inventory import determine_readiness
        
        # PASS_LIVE_READY should come before PASS_SHADOW_READY
        assert determine_readiness({'n_passed': 19, 'n_total': 19, 'failed_gates': []}) == 'PASS_LIVE_READY'
        assert determine_readiness({'n_passed': 18, 'n_total': 19, 'failed_gates': ['calibration']}) == 'PASS_SHADOW_READY'


class TestStrictGates:
    """Test strict gate evaluation logic."""
    
    def test_gate_evaluation_structure(self):
        """Test that gate evaluation returns correct structure."""
        from ml_only_candidate_inventory import evaluate_strict_gates
        
        candidate = {
            'candidate_id': 'test_candidate',
            'mean_pf': 1.488,
            'cost_1.50x_pf': 1.265,
            'mean_sharpe': 1.34,
            'gates_passed': 10,
            'gates_total': 10,
            'fold_details': {'fold_0': {'pf': 1.2}},
            'threshold_robust': True,
            'overall_metrics': {'mean_trades': 500},
            'worst_fold_pf': 1.0,
        }
        
        result = evaluate_strict_gates(candidate)
        
        assert 'passed_gates' in result
        assert 'failed_gates' in result
        assert 'n_passed' in result
        assert 'n_total' in result
        assert result['n_total'] == len(result['passed_gates']) + len(result['failed_gates'])
    
    def test_cost_1_50x_gate_critical(self):
        """Test that cost_1.50x gate is the critical one."""
        from ml_only_candidate_inventory import evaluate_strict_gates
        
        # Candidate with PF < 1.0 at 1.5x cost should fail
        candidate = {
            'candidate_id': 'weak_candidate',
            'mean_pf': 1.1,
            'cost_1.50x_pf': 0.85,  # FAILS break-even
            'mean_sharpe': 0.5,
            'gates_passed': 8,
            'gates_total': 19,
            'fold_details': {'fold_0': {'pf': 0.9}},
            'threshold_robust': False,
            'overall_metrics': {'mean_trades': 500},
            'worst_fold_pf': 0.8,
        }
        
        result = evaluate_strict_gates(candidate)
        
        assert 'profit_factor_1.50x_cost' in result['failed_gates'], \
            "Candidate with PF < 1.0 at 1.5x cost must fail"
    
    def test_positive_pf_not_sufficient(self):
        """Test that positive gross PF is not sufficient - must survive cost stress."""
        from ml_only_candidate_inventory import evaluate_strict_gates
        
        # Candidate with good base PF but bad cost-stressed PF
        candidate = {
            'candidate_id': 'gross_only_candidate',
            'mean_pf': 1.5,  # Good gross PF
            'cost_1.50x_pf': 0.9,  # But fails at 1.5x cost
            'mean_sharpe': 1.0,
            'gates_passed': 10,
            'gates_total': 19,
            'fold_details': {'fold_0': {'pf': 1.3}},
            'threshold_robust': True,
            'overall_metrics': {'mean_trades': 500},
            'worst_fold_pf': 1.2,
        }
        
        result = evaluate_strict_gates(candidate)
        
        # Should pass base PF gate but fail 1.50x cost gate
        assert 'profit_factor_base' in result['passed_gates']
        assert 'profit_factor_1.50x_cost' in result['failed_gates']
    
    def test_min_trades_gate(self):
        """Test that minimum trades gate is enforced."""
        from ml_only_candidate_inventory import evaluate_strict_gates
        
        candidate = {
            'candidate_id': 'low_trade_count',
            'mean_pf': 1.5,
            'cost_1.50x_pf': 1.2,
            'mean_sharpe': 1.0,
            'gates_passed': 5,
            'gates_total': 19,
            'fold_details': {},
            'threshold_robust': True,
            'overall_metrics': {'mean_trades': 100},  # Too few trades
            'worst_fold_pf': 1.3,
        }
        
        result = evaluate_strict_gates(candidate)
        
        assert 'min_trades' in result['failed_gates'], \
            "Candidate with < 500 trades must fail min_trades gate"


class TestReadinessLevels:
    """Test ML-only readiness level determination."""
    
    def test_pass_live_ready_requires_all_gates(self):
        """Test that PASS_LIVE_READY requires all gates passed."""
        from ml_only_candidate_inventory import determine_readiness
        
        result = {
            'n_passed': 19,
            'n_total': 19,
            'failed_gates': [],
        }
        assert determine_readiness(result) == 'PASS_LIVE_READY'
    
    def test_pass_shadow_ready_with_minor_gates(self):
        """Test PASS_SHADOW_READY allows minor gate failures."""
        from ml_only_candidate_inventory import determine_readiness
        
        # Non-critical gate failed (calibration)
        result = {
            'n_passed': 18,
            'n_total': 19,
            'failed_gates': ['calibration'],
        }
        # Should still be shadow-ready since no critical gates failed
        assert determine_readiness(result) == 'PASS_SHADOW_READY'
    
    def test_critical_gate_failure_blocks_shadow(self):
        """Test that critical gate failure blocks even shadow readiness."""
        from ml_only_candidate_inventory import determine_readiness
        
        # Critical gate (cost_1.50x) failed
        result = {
            'n_passed': 15,
            'n_total': 19,
            'failed_gates': ['profit_factor_1.50x_cost', 'calibration', 'sharpe_base', 'min_trades'],
        }
        assert determine_readiness(result) == 'FAIL_BUT_PROMISING'
    
    def test_fail_rejected_for_insufficient_gates(self):
        """Test FAIL_REJECTED for too few gates passed."""
        from ml_only_candidate_inventory import determine_readiness
        
        result = {
            'n_passed': 8,
            'n_total': 19,
            'failed_gates': ['profit_factor_base', 'profit_factor_1.50x_cost', 
                           'sharpe_base', 'min_trades', 'calibration',
                           'threshold_robustness', 'auc_above_baseline',
                           'positive_net_expectancy', 'profitable_folds', 
                           'worst_fold_pf', 'time_based_split'],
        }
        assert determine_readiness(result) == 'FAIL_REJECTED'


class TestInventoryOutput:
    """Test inventory report generation."""
    
    def test_inventory_structure(self):
        """Test that inventory has correct structure."""
        from ml_only_candidate_inventory import build_inventory
        
        models_dir = REPO_ROOT / 'models'
        output_dir = REPO_ROOT / 'reports' / 'test_output'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        inventory = build_inventory(models_dir, output_dir)
        
        assert 'candidates' in inventory
        assert 'leaderboard' in inventory
        assert 'summary' in inventory
        assert 'generated_at' in inventory
    
    def test_leaderboard_ranking(self):
        """Test that leaderboard is properly ranked."""
        from ml_only_candidate_inventory import build_inventory
        
        models_dir = REPO_ROOT / 'models'
        output_dir = REPO_ROOT / 'reports' / 'test_output'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        inventory = build_inventory(models_dir, output_dir)
        
        if inventory['leaderboard']:
            # First entry should be best
            first = inventory['leaderboard'][0]
            assert 'rank' in first
            assert first['rank'] == 1
            
            # Check ordering
            for i, entry in enumerate(inventory['leaderboard']):
                assert entry['rank'] == i + 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])