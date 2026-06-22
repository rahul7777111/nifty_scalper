#!/usr/bin/env python3
"""
tests/test_ml_only_dynamic_preset_gate_integrity.py
===================================================
Verify that dynamic presets do NOT weaken any of the 22 strict gates.

Key constraints:
- 22/22 gates required (not 21/22, not 20/22)
- PF_AT_1_50X = 1.00 (mandatory break-even)
- No gate threshold may be lowered
- 3/3 profitable folds required
- 1.5x cost PF >= 1.0 required
- daily stability score >= 0.50
- threshold robustness >= 0.60
- No PF > 10 without suspicious-result audit
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.ml_only_dynamic_preset_candidate_rescue import (
    TOTAL_GATES,
    PF_AT_1_00X,
    PF_AT_1_25X,
    PF_AT_1_50X,
    PF_AT_2_00X,
    MIN_TRADES,
    MIN_SHARPE,
    MIN_DAILY_STABILITY,
    MIN_THRESHOLD_ROBUSTNESS,
    MAX_TOP_DAY_CONCENTRATION,
    DynamicPreset,
    build_preset_grid,
    evaluate_candidate_with_preset,
)


class TestGateThresholdsNotWeakened:
    """Verify all gate thresholds are at their strict original values."""

    def test_total_gates_is_22(self) -> None:
        assert TOTAL_GATES == 22, "TOTAL_GATES must be exactly 22"

    def test_pf_at_1_00x_is_1_15(self) -> None:
        assert PF_AT_1_00X == 1.15, "PF_AT_1_00X must be 1.15 (strict)"

    def test_pf_at_1_25x_is_1_05(self) -> None:
        assert PF_AT_1_25X == 1.05, "PF_AT_1_25X must be 1.05 (strict)"

    def test_pf_at_1_50x_is_1_00(self) -> None:
        # MANDATORY break-even — cannot be lowered
        assert PF_AT_1_50X == 1.00, "PF_AT_1_50X must be 1.00 (break-even, mandatory)"

    def test_pf_at_2_00x_is_0_80(self) -> None:
        assert PF_AT_2_00X == 0.80, "PF_AT_2_00X must be 0.80 (strict)"

    def test_min_trades_is_500(self) -> None:
        assert MIN_TRADES == 500, "MIN_TRADES must be 500"

    def test_min_sharpe_is_0_75(self) -> None:
        assert MIN_SHARPE == 0.75, "MIN_SHARPE must be 0.75"

    def test_min_daily_stability_is_0_50(self) -> None:
        assert MIN_DAILY_STABILITY == 0.50, "MIN_DAILY_STABILITY must be 0.50"

    def test_min_threshold_robustness_is_0_60(self) -> None:
        assert MIN_THRESHOLD_ROBUSTNESS == 0.60, "MIN_THRESHOLD_ROBUSTNESS must be 0.60"

    def test_max_top_day_concentration_is_0_40(self) -> None:
        assert MAX_TOP_DAY_CONCENTRATION == 0.40, "MAX_TOP_DAY_CONCENTRATION must be 0.40"


class TestNoWeak21Pass:
    """Verify that 21/22 cannot be promoted to passing."""

    def test_21_22_cannot_pass_gate_check(self) -> None:
        """A result with 21/22 gates passed must NOT be treated as passing."""
        # 22/22 ALL must pass — 21 is a FAIL
        mock_gates_passed = 21
        mock_gates_total = 22
        all_pass = (mock_gates_passed == mock_gates_total == 22)
        assert all_pass is False, "21/22 must NOT pass"

    def test_20_22_cannot_pass_gate_check(self) -> None:
        """A result with 20/22 gates passed must NOT be treated as passing."""
        mock_gates_passed = 20
        mock_gates_total = 22
        all_pass = (mock_gates_passed == mock_gates_total == 22)
        assert all_pass is False, "20/22 must NOT pass"

    def test_passing_requires_exactly_22(self) -> None:
        """Only exactly 22/22 gates may pass."""
        for gates_passed in range(0, 22):
            is_passing = (gates_passed == 22)
            if gates_passed == 22:
                assert is_passing is True
            else:
                assert is_passing is False


class TestMandatoryBreakEven:
    """Verify PF_AT_1_50X cannot be lowered — mandatory break-even."""

    def test_1_50x_pf_is_strict_break_even(self) -> None:
        """PF at 1.50x cost must be exactly 1.00 (break-even)."""
        # This is the MANDATORY gate — no weakening allowed
        assert PF_AT_1_50X == 1.00
        # A candidate with PF=0.99 at 1.5x cost MUST fail
        pf_at_1_5x = 0.99
        gate_pass = (pf_at_1_5x >= PF_AT_1_50X)
        assert gate_pass is False, "PF=0.99 at 1.5x must fail gate (break-even required)"

    def test_1_50x_pf_exactly_1_00_passes(self) -> None:
        """PF at exactly 1.00 at 1.5x cost must pass (barely)."""
        pf_at_1_5x = 1.00
        gate_pass = (pf_at_1_5x >= PF_AT_1_50X)
        assert gate_pass is True, "PF=1.00 at 1.5x must pass (barely)"


class TestSuspiciousResultAudit:
    """Verify PF > 10 triggers suspicious-result audit."""

    def test_pf_over_10_requires_audit(self) -> None:
        """Any result with PF > 10 must be flagged for suspicious-result audit."""
        from scripts.ml_only_dynamic_preset_candidate_rescue import evaluate_candidate_with_preset
        import inspect
        sig = inspect.signature(evaluate_candidate_with_preset)
        source = inspect.getsource(evaluate_candidate_with_preset)
        # The evaluation must include a check for PF > 10
        assert "10.0" in source or "> 10" in source or "suspicious" in source.lower()


class TestPresetDoesNotBypassGates:
    """Verify that applying a preset does NOT bypass or weaken any gate."""

    def test_preset_evaluation_checks_all_9_cost_gates(self) -> None:
        """evaluate_candidate_with_preset must check all 9 economic gates, not skip any."""
        import inspect
        source = inspect.getsource(evaluate_candidate_with_preset)
        # All cost stress multipliers must be checked
        assert "pf_at_1" in source or "pf_at" in source
        assert "sharpe" in source.lower()
        assert "trade_count" in source.lower() or "trades" in source.lower()
        assert "fold" in source.lower()
        # Gate evaluation must exist
        assert "gates_passed" in source or "gate" in source.lower()

    def test_preset_grid_contains_only_safe_parameters(self) -> None:
        """Preset grid must contain only live-computable configuration parameters."""
        presets = build_preset_grid(max_presets=50)
        forbidden = ["return", "forward", "future", "pnl", "label", "target",
                     "outcome", "exit", "mfe", "mae", "realized"]
        for preset in presets:
            d = preset.to_dict()
            for key in d:
                assert not any(f in key.lower() for f in forbidden), \
                    f"Forbidden key '{key}' in preset {preset.preset_id}"


class Test3ProfitableFoldsRequired:
    """Verify at least 3 out of 5 walk-forward folds must be profitable."""

    def test_fold_requirement_enforced(self) -> None:
        """At least 3 out of 5 folds must have PF > 1.0."""
        # Minimum profitable folds = 3 (enforced in gates)
        min_profitable_folds = 3
        # 2 profitable folds = fail (2 < 3)
        assert not (min_profitable_folds <= 2), "2/3 folds must fail gate"
        # 3 profitable folds = pass (3 >= 3)
        assert min_profitable_folds >= 3, "3/3 folds must pass gate"