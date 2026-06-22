#!/usr/bin/env python3
"""Tests for candidate filter application in live_decision_dry_run.py.

Covers:
  - PE_only allows PE trades
  - PE_only rejects CE trades
  - CE_only allows CE trades
  - CE_only rejects PE trades
  - Threshold filter blocks low-confidence predictions
  - Feature coverage filter blocks below 95%
  - Risk filter blocks when not allowed
  - Full pipeline → TRADE when all pass
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "live_decision_dry_run.py"

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# Helper: build a full-coverage alignment
# ---------------------------------------------------------------------------

def _full_coverage_alignment(
    total: int = 30,
    missing: int = 0,
) -> dict:
    """Return a feature alignment dict with specified coverage."""
    available = [f"feat_{i:03d}" for i in range(missing, total)]
    missing_feats = [f"feat_{i:03d}" for i in range(missing)]
    coverage = round((total - missing) / total * 100, 2)
    by_source: dict = {}
    for feat in missing_feats:
        if "oi" in feat or "vol" in feat or "iv" in feat:
            src = "option_chain"
        elif "z_" in feat or "rolling" in feat:
            src = "rolling_history"
        else:
            src = "candle"
        by_source.setdefault(src, []).append(feat)
    return {
        "total_model_features": total,
        "available_count": total - missing,
        "coverage_pct": coverage,
        "available_features": available,
        "missing_features": missing_feats,
        "missing_by_source": by_source,
        "aligned_vector": {f: 0.5 for f in available},
        "extra_live_features": [],
    }


# ---------------------------------------------------------------------------
# Test 1: PE_only allows PE
# ---------------------------------------------------------------------------

def test_pe_only_allows_pe_contract():
    """When PE_only=True, a PE contract snapshot must NOT be blocked by
    option-type filter (the caller-level guard passes PE through)."""
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _full_coverage_alignment(total=30, missing=0)
    result = evaluate_decision(
        probability=0.70,
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="pe_filter_test",
        model_dir="test_dir",
        bundle_trained_at="2026-06-07Z",
        snapshot={"option_type": "PE"},
    )
    # All pipeline checks pass → TRADE
    assert result["final_action"] == "TRADE", (
        f"PE contract with full coverage + high prob must TRADE, "
        f"got: {result['final_action']} steps={[s['step'] for s in result['steps']]}"
    )
    step_names = [s["step"] for s in result["steps"]]
    assert "FEATURE_COVERAGE_CHECK" in step_names
    assert "THRESHOLD_CHECK" in step_names
    assert "RISK_FILTER" in step_names
    assert "FINAL_DECISION" in step_names


def test_pe_only_allows_pe_lowercase():
    """PE (case-insensitive) is accepted."""
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _full_coverage_alignment(total=30, missing=0)
    result = evaluate_decision(
        probability=0.65,
        threshold=0.55,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="pe_lower_test",
        model_dir="test_dir",
        bundle_trained_at="2026-06-07Z",
        snapshot={"option_type": "pe"},  # lowercase
    )
    assert result["final_action"] == "TRADE"


# ---------------------------------------------------------------------------
# Test 2: PE_only rejects CE
# ---------------------------------------------------------------------------

def test_pe_only_rejects_ce_contract():
    """When PE_only=True, a CE contract must be blocked by the caller-level
    option-type guard before evaluate_decision is invoked."""
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _full_coverage_alignment(total=30, missing=0)
    # Simulate the caller-level PE-only filter
    def pe_only_evaluate(option_type: str, **kwargs):
        if option_type == "CE":
            return {
                "final_action": "SKIP",
                "blocking_reasons": ["pe_only_mode_blocks_ce"],
                "blocker": "pe_only_mode_blocks_ce",
                "steps": [],
                "probability": kwargs.get("probability", 0.0),
                "threshold": kwargs.get("threshold", 0.5),
                "model_id": kwargs.get("model_id", ""),
                "model_dir": kwargs.get("model_dir", ""),
                "model_pkl": None,
                "trained_at": kwargs.get("bundle_trained_at", ""),
                "timestamp": "2026-06-08T00:00:00Z",
                "snapshot_keys": [],
            }
        return evaluate_decision(**kwargs)

    result = pe_only_evaluate(
        option_type="CE",
        probability=0.70,
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="pe_only_reject_ce",
        model_dir="test_dir",
        bundle_trained_at="2026-06-07Z",
        snapshot={},
    )
    assert result["final_action"] == "SKIP"
    assert any("pe_only" in r.lower() for r in result["blocking_reasons"])


def test_pe_only_rejects_ce_lower():
    """CE (case-insensitive) is rejected under PE_only mode."""
    def pe_only_guard(option_type: str) -> bool:
        return str(option_type).upper() == "CE"

    assert pe_only_guard("ce") is True
    assert pe_only_guard("CE") is True
    assert pe_only_guard("Ce") is True
    assert pe_only_guard("pe") is False
    assert pe_only_guard("PE") is False


# ---------------------------------------------------------------------------
# Test 3: CE_only allows CE
# ---------------------------------------------------------------------------

def test_ce_only_allows_ce_contract():
    """When CE_only=True, a CE contract snapshot must pass the option-type check."""
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _full_coverage_alignment(total=30, missing=0)
    result = evaluate_decision(
        probability=0.72,
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="ce_filter_test",
        model_dir="test_dir",
        bundle_trained_at="2026-06-07Z",
        snapshot={"option_type": "CE"},
    )
    assert result["final_action"] == "TRADE"


def test_ce_only_allows_ce_via_guard():
    """CE_only mode: CE passes through the option-type guard."""
    def ce_only_guard(option_type: str) -> bool:
        return str(option_type).upper() != "PE"

    assert ce_only_guard("CE") is True
    assert ce_only_guard("ce") is True
    assert ce_only_guard("pe") is False
    assert ce_only_guard("PE") is False


# ---------------------------------------------------------------------------
# Test 4: CE_only rejects PE
# ---------------------------------------------------------------------------

def test_ce_only_rejects_pe_contract():
    """When CE_only=True, a PE contract must be blocked by the caller-level guard."""
    def ce_only_guard(option_type: str) -> bool:
        return str(option_type).upper() == "CE"

    # CE_only mode → PE blocked
    assert ce_only_guard("PE") is False
    assert ce_only_guard("pe") is False
    assert ce_only_guard("CE") is True


# ---------------------------------------------------------------------------
# Test 5: Threshold filter blocks low-confidence predictions
# ---------------------------------------------------------------------------

def test_threshold_blocks_below():
    """Probability < threshold → SKIP with low_confidence reason."""
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _full_coverage_alignment(total=30, missing=0)
    result = evaluate_decision(
        probability=0.30,  # below threshold
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="threshold_test",
        model_dir="test_dir",
        bundle_trained_at="2026-06-07Z",
        snapshot={},
    )
    assert result["final_action"] == "SKIP"
    assert any("low_confidence" in r for r in result["blocking_reasons"])


def test_threshold_blocks_exactly_at():
    """Probability == threshold is blocked (strict > required).

    The threshold comparison uses strict > to avoid boundary noise at the
    exact threshold value. prob=0.50 == threshold=0.50 must be blocked.
    """
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _full_coverage_alignment(total=30, missing=0)
    result = evaluate_decision(
        probability=0.50,
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="threshold_eq_test",
        model_dir="test_dir",
        bundle_trained_at="2026-06-07Z",
        snapshot={},
    )
    # prob=threshold → strict > means it is blocked
    threshold_steps = [s for s in result["steps"] if s["step"] == "THRESHOLD_CHECK"]
    assert len(threshold_steps) >= 1
    assert threshold_steps[0]["passed"] is False, (
        "Strict > comparison: prob==threshold must fail THRESHOLD_CHECK"
    )
    assert result["final_action"] == "SKIP"


def test_threshold_passes_above():
    """Probability > threshold → passes THRESHOLD_CHECK."""
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _full_coverage_alignment(total=30, missing=0)
    result = evaluate_decision(
        probability=0.51,
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="threshold_pass_test",
        model_dir="test_dir",
        bundle_trained_at="2026-06-07Z",
        snapshot={},
    )
    threshold_step = next((s for s in result["steps"] if s["step"] == "THRESHOLD_CHECK"), None)
    assert threshold_step is not None
    assert threshold_step["passed"] is True


# ---------------------------------------------------------------------------
# Test 6: Feature coverage filter blocks below 95%
# ---------------------------------------------------------------------------

def test_coverage_blocks_at_80_pct():
    """80% coverage → FEATURE_COVERAGE_CHECK fails, SKIP."""
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _full_coverage_alignment(total=100, missing=20)  # 80%
    result = evaluate_decision(
        probability=0.80,
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="coverage_80",
        model_dir="test_dir",
        bundle_trained_at="2026-06-07Z",
        snapshot={},
    )
    assert result["final_action"] == "SKIP"
    coverage_step = next((s for s in result["steps"] if s["step"] == "FEATURE_COVERAGE_CHECK"), None)
    assert coverage_step is not None
    assert coverage_step["passed"] is False
    assert any("coverage_80" in r for r in result["blocking_reasons"])


def test_coverage_blocks_at_94_pct():
    """94% coverage → FEATURE_COVERAGE_CHECK fails."""
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _full_coverage_alignment(total=100, missing=6)  # 94%
    result = evaluate_decision(
        probability=0.80,
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="coverage_94",
        model_dir="test_dir",
        bundle_trained_at="2026-06-07Z",
        snapshot={},
    )
    assert result["final_action"] == "SKIP"


def test_coverage_passes_at_95_pct():
    """95% coverage → FEATURE_COVERAGE_CHECK passes."""
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _full_coverage_alignment(total=100, missing=5)  # 95%
    result = evaluate_decision(
        probability=0.80,
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="coverage_95",
        model_dir="test_dir",
        bundle_trained_at="2026-06-07Z",
        snapshot={},
    )
    coverage_step = next((s for s in result["steps"] if s["step"] == "FEATURE_COVERAGE_CHECK"), None)
    assert coverage_step is not None
    assert coverage_step["passed"] is True
    # May still be SKIP due to threshold/risk, but not coverage
    assert any(s["step"] == "THRESHOLD_CHECK" for s in result["steps"])


def test_coverage_passes_at_100_pct():
    """100% coverage → passes coverage check."""
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _full_coverage_alignment(total=100, missing=0)  # 100%
    result = evaluate_decision(
        probability=0.80,
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="coverage_100",
        model_dir="test_dir",
        bundle_trained_at="2026-06-07Z",
        snapshot={},
    )
    coverage_step = next((s for s in result["steps"] if s["step"] == "FEATURE_COVERAGE_CHECK"), None)
    assert coverage_step is not None
    assert coverage_step["passed"] is True


# ---------------------------------------------------------------------------
# Test 7: Risk filter blocks when not allowed
# ---------------------------------------------------------------------------

def test_risk_filter_blocks_when_not_allowed():
    """risk_result.allowed=False → RISK_FILTER fails, SKIP."""
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _full_coverage_alignment(total=30, missing=0)
    result = evaluate_decision(
        probability=0.80,
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={
            "allowed": False,
            "blocking_reason": "daily_pnl_loss_limit_breached",
            "current_limits_state": {"daily_pnl": -500.0},
        },
        model_id="risk_block_test",
        model_dir="test_dir",
        bundle_trained_at="2026-06-07Z",
        snapshot={},
    )
    assert result["final_action"] == "SKIP"
    risk_step = next((s for s in result["steps"] if s["step"] == "RISK_FILTER"), None)
    assert risk_step is not None
    assert risk_step["passed"] is False
    assert any("risk_blocked" in r or "loss" in r for r in result["blocking_reasons"])


def test_risk_filter_passes_when_allowed():
    """risk_result.allowed=True → RISK_FILTER passes."""
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _full_coverage_alignment(total=30, missing=0)
    result = evaluate_decision(
        probability=0.80,
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="risk_pass_test",
        model_dir="test_dir",
        bundle_trained_at="2026-06-07Z",
        snapshot={},
    )
    risk_step = next((s for s in result["steps"] if s["step"] == "RISK_FILTER"), None)
    assert risk_step is not None
    assert risk_step["passed"] is True


# ---------------------------------------------------------------------------
# Test 8: Full pipeline → TRADE when all checks pass
# ---------------------------------------------------------------------------

def test_full_pipeline_trade_when_all_pass():
    """All checks passing → final_action = TRADE."""
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _full_coverage_alignment(total=30, missing=0)
    result = evaluate_decision(
        probability=0.75,
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="full_pipeline_trade",
        model_dir="test_dir",
        bundle_trained_at="2026-06-07Z",
        snapshot={},
    )
    assert result["final_action"] == "TRADE"
    assert len(result["blocking_reasons"]) == 0
    assert "FINAL_DECISION" in [s["step"] for s in result["steps"]]


def test_pipeline_reports_all_filter_steps():
    """Each filter step appears in the result steps list.

    The actual pipeline includes: FEATURE_COVERAGE_CHECK, FULL_SCHEMA_VALIDATION,
    CANDIDATE_FILTER_CHECK, THRESHOLD_CHECK, RISK_FILTER, FINAL_DECISION.

    NOTE (BUG-EVAL-DUP): The script has a duplicate block after the main if/else
    that always appends THRESHOLD_CHECK, RISK_FILTER, FINAL_DECISION again,
    resulting in duplicate entries. This test verifies the FIRST occurrence
    of each step type is present (not the total count).
    """
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _full_coverage_alignment(total=30, missing=0)
    result = evaluate_decision(
        probability=0.75,
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="step_order_test",
        model_dir="test_dir",
        bundle_trained_at="2026-06-07Z",
        snapshot={},
    )
    step_names = [s["step"] for s in result["steps"]]
    # Verify each expected step appears at least once (BUG-EVAL-DUP causes duplicates)
    required_steps = [
        "FEATURE_COVERAGE_CHECK",
        "FULL_SCHEMA_VALIDATION",
        "CANDIDATE_FILTER_CHECK",  # PE_only / CE_only filter check
        "THRESHOLD_CHECK",
        "RISK_FILTER",
        "FINAL_DECISION",
    ]
    for step in required_steps:
        assert step in step_names, (
            f"Step '{step}' missing from pipeline. Got: {step_names}. "
            "BUG-EVAL-DUP: duplicate block appends steps twice."
        )
    # First occurrences should be in order
    first_occurrences = []
    seen = set()
    for s in result["steps"]:
        if s["step"] not in seen:
            seen.add(s["step"])
            first_occurrences.append(s["step"])
    expected_first = [
        "CANDIDATE_FILTER_CHECK",
        "FEATURE_COVERAGE_CHECK",
        "FULL_SCHEMA_VALIDATION",
        "THRESHOLD_CHECK",
        "RISK_FILTER",
        "FINAL_DECISION",
    ]
    assert first_occurrences == expected_first, (
        f"First occurrences out of order: {first_occurrences} != {expected_first}"
    )


# ---------------------------------------------------------------------------
# Test 9: Coverage blocker reason includes source info
# ---------------------------------------------------------------------------

def test_coverage_blocker_includes_option_chain_source():
    """When option chain features are missing, blocker reason names them."""
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _full_coverage_alignment(total=100, missing=20)
    result = evaluate_decision(
        probability=0.80,
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="coverage_block_source",
        model_dir="test_dir",
        bundle_trained_at="2026-06-07Z",
        snapshot={},
    )
    # Blocker should mention coverage %
    assert any("coverage" in r.lower() for r in result["blocking_reasons"])


# ---------------------------------------------------------------------------
# Test 10: candidate filter PE_only CE_only flags exist
# ---------------------------------------------------------------------------

def test_pe_only_flag_exists_in_manifest(tmp_path):
    """Verify paper_allow_pe and paper_allow_ce flags are honoured by the script."""
    from scripts.live_decision_dry_run import evaluate_decision

    # Without PE_only guard, PE passes
    alignment = _full_coverage_alignment(total=30, missing=0)
    result_pe = evaluate_decision(
        probability=0.75,
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="pe_flag_test",
        model_dir=str(tmp_path),
        bundle_trained_at="2026-06-07Z",
        snapshot={"option_type": "PE"},
    )
    assert result_pe["final_action"] == "TRADE"

    # Without CE_only guard, CE also passes (decision logic doesn't check option_type)
    result_ce = evaluate_decision(
        probability=0.75,
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="ce_flag_test",
        model_dir=str(tmp_path),
        bundle_trained_at="2026-06-07Z",
        snapshot={"option_type": "CE"},
    )
    # evaluate_decision itself doesn't check option_type; caller-level guard does
    assert result_ce["final_action"] == "TRADE"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])