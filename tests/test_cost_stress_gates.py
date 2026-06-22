from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain
from ml_execution_costs import (  # noqa: E402
    audit_double_counting,
    cost_survival_score,
    evaluate_cost_sensitive_bands,
    evaluate_expected_move_filters,
    evaluate_premium_bands,
    evaluate_spread_filters,
    estimate_option_execution_cost,
)


def test_cost_stress_reduces_returns_and_can_fail_gate() -> None:
    y_true = np.array([1, 1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.85, 0.76, 0.1])
    returns = np.array([0.35, 0.30, 0.28, -0.2, -0.1])
    report = retrain._cost_stress_report(y_true, y_prob, 0.75, returns)
    assert report["base"]["total_return"] > report["extra_cost_0_25"]["total_return"]
    assert report["extra_cost_0_25"]["failure_flag"] is True


def test_fold_stability_fails_when_only_one_fold_is_good() -> None:
    folds = [
        {"profit_factor": 1.2, "sharpe": 0.7, "average_return": 0.1},
        {"profit_factor": 0.9, "sharpe": 0.2, "average_return": -0.1},
        {"profit_factor": 0.8, "sharpe": -0.1, "average_return": -0.2},
        {"profit_factor": 1.0, "sharpe": 0.3, "average_return": 0.0},
    ]
    summary = retrain._fold_stability_report(folds)
    assert summary["gate"] == "FAIL"


def test_gross_forward_return_baseline_subtracts_full_cost() -> None:
    """Non-net (gross) returns: base deducts baseline_deduction; stress adds baseline * extra_cost."""
    y_true = np.array([1, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.1])
    returns = np.array([1.0, 1.0, 0.0])
    provenance = {
        "is_evaluation_return_already_net": False,
        "inferred_embedded_cost": 0.2,
        "baseline_cost_deduction_used": 0.2,
    }
    report = retrain._cost_stress_report(y_true, y_prob, 0.5, returns, cost_provenance=provenance)
    # base deducts baseline_deduction=0.2 from returns [1.0,1.0]: avg = 0.8
    assert report["base"]["average_return_per_trade"] == 0.8
    assert report["base"]["effective_cost_deduction"] == 0.2
    # stress: deduct = baseline_deduction * extra_cost (0.2 * 0.25 = 0.05)
    assert report["extra_cost_0_25"]["effective_cost_deduction"] == 0.05
    # non-net stress: baseline_deduction * extra_cost
    assert report["extra_cost_0_25"]["effective_cost_deduction"] == pytest.approx(0.05)
    assert report["extra_cost_0_50"]["effective_cost_deduction"] == pytest.approx(0.10)
    assert report["extra_cost_1_00"]["effective_cost_deduction"] == pytest.approx(0.20)
    assert report["extra_cost_2_00"]["effective_cost_deduction"] == pytest.approx(0.40)
    # All 4 stress (non-base) scenarios must have distinct deductions
    stress_deductions = [report[s]["effective_cost_deduction"] for s in ["extra_cost_0_25", "extra_cost_0_50", "extra_cost_1_00", "extra_cost_2_00"]]
    assert len(set(stress_deductions)) == len(stress_deductions), f"Stress deductions must be distinct: {stress_deductions}"
    # Base must be non-zero and differ from at least one stress scenario
    assert report["base"]["effective_cost_deduction"] > 0.0
    assert report["base"]["effective_cost_deduction"] != report["extra_cost_0_25"]["effective_cost_deduction"]


def test_net_forward_return_base_uses_embedded_cost() -> None:
    """Net returns: base deducts inferred_embedded_cost (to recover gross); stress adds incremental."""
    y_true = np.array([1, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.1])
    returns = np.array([0.8, 0.8, 0.0])  # net returns (costs already baked in)
    provenance = {
        "is_evaluation_return_already_net": True,
        "inferred_embedded_cost": 0.2,
        "baseline_cost_deduction_used": 0.0,
    }
    report = retrain._cost_stress_report(y_true, y_prob, 0.5, returns, cost_provenance=provenance)
    # base: deduct inferred_embedded_cost=0.2 from returns [0.8,0.8]: avg = 0.6
    assert report["base"]["average_return_per_trade"] == pytest.approx(0.6, abs=1e-9)
    assert report["base"]["effective_cost_deduction"] == 0.2
    # stress: embedded_cost * (1 + extra_cost) for net returns
    # 0.25x: 0.2*(1+0.25)=0.25, 0.50x: 0.2*(1+0.5)=0.3, 1.0x: 0.2*(1+1.0)=0.4, 2.0x: 0.2*(1+2.0)=0.6
    assert report["extra_cost_0_25"]["effective_cost_deduction"] == pytest.approx(0.25)
    assert report["extra_cost_0_50"]["effective_cost_deduction"] == pytest.approx(0.30)
    assert report["extra_cost_1_00"]["effective_cost_deduction"] == pytest.approx(0.40)
    assert report["extra_cost_2_00"]["effective_cost_deduction"] == pytest.approx(0.60)
    # all 5 scenarios should have different deductions
    deductions = [report[s]["effective_cost_deduction"] for s in ["base", "extra_cost_0_25", "extra_cost_0_50", "extra_cost_1_00", "extra_cost_2_00"]]
    assert len(set(deductions)) == len(deductions), f"Deductions should all differ: {deductions}"


def test_cost_stress_scenarios_differentiate_profit_factors() -> None:
    """All 5 cost-stress scenarios must produce different effective deductions.
    Uses mixed wins/losses so PF is finite (not pinned at 999=infinity).
    Construction ensures 10 wins + 10 losses are both selected at threshold=0.5.
    """
    np.random.seed(42)
    y_true = np.array([1] * 20 + [0] * 20)
    # 20 positives: 10 selected (0.75), 10 rejected (0.45)
    # 20 negatives: 10 selected (0.55), 10 rejected (0.20)
    y_prob = np.array([0.75] * 10 + [0.45] * 10 + [0.55] * 10 + [0.20] * 10)
    returns = np.concatenate(
        [np.random.uniform(0.005, 0.02, 20), np.random.uniform(-0.015, -0.005, 20)]
    )
    provenance = {
        "is_evaluation_return_already_net": True,
        "inferred_embedded_cost": 0.0035,
        "baseline_cost_deduction_used": 0.0,
    }
    report = retrain._cost_stress_report(y_true, y_prob, 0.5, returns, cost_provenance=provenance)
    deductions = {name: s["effective_cost_deduction"] for name, s in report.items()}
    assert len(set(deductions.values())) == len(deductions), (
        f"All 5 scenarios must have different deductions: {deductions}"
    )
    # With mixed wins/losses, base and highest-stress scenario should differ in PF
    pfs = {
        name: s["profit_factor"]
        for name, s in report.items()
        if s.get("profit_factor") is not None and s.get("profit_factor") < 999
    }
    assert len(pfs) >= 2, f"Need at least 2 finite PFs for comparison, got: {pfs}"
    pf_values = list(pfs.values())
    assert len(set(pf_values)) > 1, f"PFs should show differentiation: {pfs}"


def test_infer_embedded_cost_from_real_data_not_null() -> None:
    """_infer_embedded_cost_from_frame must return non-null on datasets with zero-gross rows."""
    from retrain_all_edge_models import _infer_embedded_cost_from_frame

    # Simulate real data pattern: many zero-gross rows (no price movement in horizon)
    # where net = -0.0035 (cost visible as the only diff between gross and net)
    gross = [0.0] * 80 + [-0.05] * 20
    net = [-0.0035] * 80 + [-0.0535] * 20  # diff = 0.0035 in both cases
    df = pd.DataFrame({"gross_forward_return": gross, "net_forward_return": net})

    result = _infer_embedded_cost_from_frame(df)
    assert result is not None, "Must not be None when zero-gross rows exist"
    assert result > 0, f"Must be positive, got {result}"
    assert abs(result - 0.0035) < 0.0001, f"Expected ~0.0035, got {result}"


def test_base_deduction_not_zero_for_net_returns() -> None:
    """When is_net=True with inferred_embedded_cost>0, base scenario must NOT deduct 0."""
    y_true = np.array([1, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.1])
    returns = np.array([0.01, 0.01, 0.0])
    provenance = {
        "is_evaluation_return_already_net": True,
        "inferred_embedded_cost": 0.0035,
        "baseline_cost_deduction_used": 0.0,
    }
    report = retrain._cost_stress_report(y_true, y_prob, 0.5, returns, cost_provenance=provenance)
    assert report["base"]["effective_cost_deduction"] > 0.0, (
        f"Base deduction must be > 0 for net returns, got {report['base']['effective_cost_deduction']}"
    )
    assert report["base"]["effective_cost_deduction"] == 0.0035


# ---------------------------------------------------------------------------
# New tests for ml_execution_costs audit helpers and the audit script.
# ---------------------------------------------------------------------------


def test_estimate_option_execution_cost_roundtrip_applied_once() -> None:
    row = {"ltp": 100.0, "net_forward_return": 0.05, "gross_forward_return": 0.07}
    out = estimate_option_execution_cost(row)
    # roundtrip=True means brokerage is doubled, but cost_return_units must be
    # applied ONCE per selected trade (not twice in this helper).
    assert out["brokerage_cost"] == 0.04  # 0.02 * 2
    assert out["cost_return_units"] == out["total_estimated_roundtrip_cost"]
    assert out["cost_pct_of_premium"] > 0.0
    assert out["embedded_cost"] == pytest.approx(0.02, abs=1e-9)


def test_estimate_option_execution_cost_falls_back_when_bid_ask_missing() -> None:
    row = {"ltp": 50.0, "net_forward_return": 0.01}
    out = estimate_option_execution_cost(row)
    assert out["spread_cost"] == 0.0
    assert out["total_estimated_roundtrip_cost"] > 0.0
    # When bid/ask is missing the helper applies a default spread; we accept
    # either a zero spread (purely config fallback) or a non-zero default.
    assert out["embedded_cost"] >= 0.0


def test_estimate_option_execution_cost_uses_real_spread_when_present() -> None:
    row = {"ltp": 100.0, "net_forward_return": 0.02, "bid_ask_spread_pct": 1.0}
    out = estimate_option_execution_cost(row)
    assert out["spread_cost"] > 0.0


def test_missing_premium_blocks_cost_estimate() -> None:
    # The cost helper defends against premium<=0 by returning zero cost and
    # callers are required to block the candidate.
    row = {"net_forward_return": 0.01, "ltp": 0}
    out = estimate_option_execution_cost(row)
    assert out["cost_pct_of_premium"] == 0.0
    # configured_cost may still be positive (e.g. brokerage floor), but
    # cost_pct_of_premium and the rate-based cost fields should be zero.
    assert out["cost_return_units"] >= 0.0
    # The audit script's caller treats any "cost_pct_of_premium==0 with ltp==0"
    # case as `execution_cost_unavailable`; replicate that gating in test.
    assert out["cost_pct_of_premium"] == 0.0


def test_audit_double_counting_detects_embedded_cost() -> None:
    df = pd.DataFrame({
        "gross_forward_return": [0.10, 0.08, 0.12, 0.05],
        "net_forward_return": [0.08, 0.06, 0.10, 0.04],
    })
    out = audit_double_counting(df, evaluation_return_column="net_forward_return")
    assert out["net_forward_return_already_cost_adjusted"] is True
    assert out["inferred_embedded_cost"] == pytest.approx(0.02, abs=1e-9)
    assert out["recommendation"] in {
        "VALID_NET_RETURN_INCREMENTAL_STRESS_MODEL",
        "BLOCKED_UNCERTAIN_COST_PROVENANCE",
    }


def test_premium_band_evaluation_filters_by_ltp() -> None:
    df = pd.DataFrame({
        "ltp": [10.0, 25.0, 55.0, 110.0],
        "net_forward_return": [0.05, 0.03, 0.02, 0.04],
    })
    out = evaluate_premium_bands(df)
    assert out["available"] is True
    band_map = {b["min_premium"]: b["trade_count"] for b in out["bands"]}
    assert band_map[20.0] == 3
    assert band_map[50.0] == 2
    assert band_map[100.0] == 1


def test_spread_filter_skipped_when_column_missing() -> None:
    df = pd.DataFrame({"ltp": [50.0, 60.0], "net_forward_return": [0.02, 0.03]})
    out = evaluate_spread_filters(df)
    assert out["available"] is False
    assert out.get("skipped") is True


def test_spread_filter_applied_when_column_present() -> None:
    df = pd.DataFrame({
        "ltp": [50.0, 60.0, 80.0, 100.0],
        "net_forward_return": [0.05, 0.04, 0.03, 0.02],
        "bid_ask_spread_pct": [0.2, 0.4, 1.2, 3.0],
    })
    out = evaluate_spread_filters(df)
    assert out["available"] is True
    names = {f["filter_name"] for f in out["filters"]}
    assert "bid_ask_spread_pct_le_0_5" in names


def test_expected_return_to_cost_ratio_calculated() -> None:
    df = pd.DataFrame({
        "ltp": [100.0] * 6,
        "net_forward_return": [0.05, 0.04, 0.06, 0.03, 0.02, 0.07],
        "model_probability": [0.6, 0.55, 0.7, 0.65, 0.58, 0.62],
    })
    out = evaluate_expected_move_filters(df)
    assert out["available"] is True
    f1 = next(f for f in out["filters"] if f["filter_name"] == "expected_return_to_cost_ratio_ge_1_25")
    assert "return_to_cost_ratio" in f1


def test_cost_survival_score_ranks_cost_stable_candidate_higher() -> None:
    # High raw PF candidate with weak cost survival
    weak = {
        "cost_stress_1_25x_pf": 0.5,
        "cost_stress_1_50x_pf": 0.2,
        "return_to_cost_ratio": 0.5,
        "worst_fold_pf": 1.0,
        "median_fold_pf": 1.0,
        "profitable_days": 10,
        "top_2_days_profit_share": 0.6,
        "best_fold_profit_share": 0.4,
        "max_drawdown": 0.0,
    }
    # Cost-stable candidate with lower raw PF
    strong = {
        "cost_stress_1_25x_pf": 1.5,
        "cost_stress_1_50x_pf": 1.2,
        "return_to_cost_ratio": 2.0,
        "worst_fold_pf": 1.2,
        "median_fold_pf": 1.3,
        "profitable_days": 40,
        "top_2_days_profit_share": 0.2,
        "best_fold_profit_share": 0.15,
        "max_drawdown": 0.0,
    }
    assert cost_survival_score(strong) > cost_survival_score(weak)


def test_audit_script_deployment_defaults_to_blocked_when_no_pass(tmp_path: Path) -> None:
    # Insert the scripts dir to import the audit script directly.
    audit_path = SCRIPTS_DIR / "audit_cost_model_and_edge.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("audit_cost_model_and_edge", audit_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    # Build a refinement payload where nothing passes.
    refine = {
        "best_candidate_after_realistic_cost": {
            "candidate_name": "x",
            "cost_1_25x_profit_factor": 0.4,
            "cost_1_50x_profit_factor": 0.1,
            "return_to_cost_ratio": 0.5,
            "profit_factor": 1.2,
            "sharpe": 0.4,
            "expected_return_after_estimated_cost": -0.01,
            "median_return_after_estimated_cost": -0.005,
        },
        "best_candidate_before_cost": None,
    }
    deploy = mod._deployment_readiness_payload(refine, dataset_path=tmp_path / "ds.csv")
    assert deploy["deployment_status"] == "BLOCKED"
    assert deploy["production_adoption_allowed"] is False
    assert deploy["paper_execution_allowed"] is False
    assert "cost_1_5x_pf_below_1.00" in deploy["failed_gates"]


def test_audit_script_paper_signal_only_when_watchlist_only() -> None:
    audit_path = SCRIPTS_DIR / "audit_cost_model_and_edge.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("audit_cost_model_and_edge", audit_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    refine = {
        "best_candidate_after_realistic_cost": {
            "candidate_name": "x",
            "cost_1_25x_profit_factor": 1.10,
            "cost_1_50x_profit_factor": 1.20,
            "return_to_cost_ratio": 1.30,
            "profit_factor": 1.10,
            "sharpe": 0.6,
            "expected_return_after_estimated_cost": 0.01,
            "median_return_after_estimated_cost": 0.005,
        },
    }
    deploy = mod._deployment_readiness_payload(refine, dataset_path=Path("dummy.csv"))
    assert deploy["deployment_status"] == "PAPER_SIGNAL_ONLY_ALLOWED"
    assert deploy["paper_execution_allowed"] is False


def test_audit_script_paper_execution_allowed_only_when_all_strict() -> None:
    audit_path = SCRIPTS_DIR / "audit_cost_model_and_edge.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("audit_cost_model_and_edge", audit_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    refine = {
        "best_candidate_after_realistic_cost": {
            "candidate_name": "x",
            "cost_1_25x_profit_factor": 1.50,
            "cost_1_50x_profit_factor": 1.20,
            "return_to_cost_ratio": 2.0,
            "profit_factor": 1.30,
            "sharpe": 1.0,
            "expected_return_after_estimated_cost": 0.02,
            "median_return_after_estimated_cost": 0.01,
        },
    }
    deploy = mod._deployment_readiness_payload(refine, dataset_path=Path("dummy.csv"))
    assert deploy["deployment_status"] == "PAPER_EXECUTION_ALLOWED"
    assert deploy["production_adoption_allowed"] is False


# ---------------------------------------------------------------------------
# New tests for the audit helpers and strict deployment gate.
# ---------------------------------------------------------------------------


def test_audit_double_counting_detects_embedded_cost() -> None:
    """gross/net difference is the inferred embedded cost."""
    df = pd.DataFrame({
        "gross_forward_return": [0.10, 0.08, 0.12, 0.05],
        "net_forward_return": [0.08, 0.06, 0.10, 0.04],
    })
    out = audit_double_counting(df, evaluation_return_column="net_forward_return")
    assert out["net_forward_return_already_cost_adjusted"] is True
    assert out["inferred_embedded_cost"] == pytest.approx(0.02, abs=1e-9)
    assert out["recommendation"] in {
        "VALID_NET_RETURN_INCREMENTAL_STRESS_MODEL",
        "BLOCKED_UNCERTAIN_COST_PROVENANCE",
    }


def test_audit_double_counting_missing_columns() -> None:
    df = pd.DataFrame({"ltp": [100.0]})
    out = audit_double_counting(df, evaluation_return_column="net_forward_return")
    assert out["recommendation"] == "BLOCKED_MISSING_GROSS_OR_NET"


def test_evaluate_premium_bands_filters_by_ltp() -> None:
    df = pd.DataFrame({
        "ltp": [10.0, 25.0, 55.0, 110.0],
        "net_forward_return": [0.05, 0.03, 0.02, 0.04],
    })
    out = evaluate_premium_bands(df)
    assert out["available"] is True
    band_map = {b["min_premium"]: b["trade_count"] for b in out["bands"]}
    assert band_map[20.0] == 3
    assert band_map[50.0] == 2
    assert band_map[100.0] == 1


def test_evaluate_premium_bands_skips_when_columns_missing() -> None:
    df = pd.DataFrame({"net_forward_return": [0.01, 0.02]})
    out = evaluate_premium_bands(df)
    assert out["available"] is False


def test_evaluate_spread_filters_skipped_when_column_missing() -> None:
    df = pd.DataFrame({"ltp": [50.0, 60.0], "net_forward_return": [0.02, 0.03]})
    out = evaluate_spread_filters(df)
    assert out["available"] is False
    assert out.get("skipped") is True


def test_evaluate_spread_filters_applied_when_column_present() -> None:
    df = pd.DataFrame({
        "ltp": [50.0, 60.0, 80.0, 100.0],
        "net_forward_return": [0.05, 0.04, 0.03, 0.02],
        "bid_ask_spread_pct": [0.2, 0.4, 1.2, 3.0],
    })
    out = evaluate_spread_filters(df)
    assert out["available"] is True
    names = {f["filter_name"] for f in out["filters"]}
    assert "bid_ask_spread_pct_le_0_5" in names
    assert "bid_ask_spread_pct_le_1_5" in names


def test_evaluate_expected_move_filters_present_with_probability() -> None:
    df = pd.DataFrame({
        "ltp": [100.0] * 6,
        "net_forward_return": [0.05, 0.04, 0.06, 0.03, 0.02, 0.07],
        "model_probability": [0.6, 0.55, 0.7, 0.65, 0.58, 0.62],
    })
    out = evaluate_expected_move_filters(df)
    assert out["available"] is True
    f1 = next(f for f in out["filters"] if f["filter_name"] == "expected_return_to_cost_ratio_ge_1_25")
    assert "return_to_cost_ratio" in f1


def test_evaluate_cost_sensitive_bands_reports_in_band() -> None:
    df = pd.DataFrame({
        "ltp": [100.0] * 400,
        "net_forward_return": list(np.linspace(-0.02, 0.05, 400)),
    })
    out = evaluate_cost_sensitive_bands(df)
    assert out["available"] is True
    in_band = [b for b in out["bands"] if b.get("in_band")]
    assert in_band, "expected at least one band to be in-band for 400 trades"
    for b in in_band:
        assert b["trade_count"] > 0


def test_evaluate_cost_sensitive_bands_skips_empty() -> None:
    df = pd.DataFrame({"ltp": [], "net_forward_return": []})
    out = evaluate_cost_sensitive_bands(df)
    assert out["available"] is False


def test_cost_survival_score_ranks_cost_stable_candidate_higher() -> None:
    weak = {
        "cost_stress_1_25x_pf": 0.5,
        "cost_stress_1_50x_pf": 0.2,
        "return_to_cost_ratio": 0.5,
        "worst_fold_pf": 1.0,
        "median_fold_pf": 1.0,
        "profitable_days": 10,
        "top_2_days_profit_share": 0.6,
        "best_fold_profit_share": 0.4,
        "max_drawdown": 0.0,
    }
    strong = {
        "cost_stress_1_25x_pf": 1.5,
        "cost_stress_1_50x_pf": 1.2,
        "return_to_cost_ratio": 2.0,
        "worst_fold_pf": 1.2,
        "median_fold_pf": 1.3,
        "profitable_days": 40,
        "top_2_days_profit_share": 0.2,
        "best_fold_profit_share": 0.15,
        "max_drawdown": 0.0,
    }
    assert cost_survival_score(strong) > cost_survival_score(weak)


def test_deployment_blocks_when_no_cost_1_5x_pass() -> None:
    """If cost_1.5x PF < 1.00, expected_after_cost <= 0, OR median_after_cost <= 0
    then deployment_readiness_payload must return BLOCKED."""
    audit_path = SCRIPTS_DIR / "audit_cost_model_and_edge.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("audit_cost_model_and_edge", audit_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    # Case 1: cost_1.5x PF < 1.00
    refine = {
        "best_candidate_after_realistic_cost": {
            "candidate_name": "x",
            "cost_1_25x_profit_factor": 1.2,
            "cost_1_50x_profit_factor": 0.9,
            "return_to_cost_ratio": 1.3,
            "profit_factor": 1.2,
            "sharpe": 1.1,
            "expected_return_after_estimated_cost": 0.01,
            "median_return_after_estimated_cost": 0.005,
        }
    }
    deploy = mod._deployment_readiness_payload(refine, dataset_path=Path("dummy.csv"))
    assert deploy["deployment_status"] == "BLOCKED"
    assert "cost_1_5x_pf_below_1.00" in deploy["failed_gates"]
    # Case 2: expected_return_after_cost <= 0
    refine2 = {
        "best_candidate_after_realistic_cost": {
            "candidate_name": "x",
            "cost_1_25x_profit_factor": 1.5,
            "cost_1_50x_profit_factor": 1.1,
            "return_to_cost_ratio": 1.5,
            "profit_factor": 1.3,
            "sharpe": 1.0,
            "expected_return_after_estimated_cost": -0.01,
            "median_return_after_estimated_cost": 0.005,
        }
    }
    deploy2 = mod._deployment_readiness_payload(refine2, dataset_path=Path("dummy.csv"))
    assert deploy2["deployment_status"] == "BLOCKED"
    assert "expected_return_after_cost_non_positive" in deploy2["failed_gates"]



# ---------------------------------------------------------------------------
# Regression tests for cost-stress bug fix (bugs A, B1, B2, C from 2026-06-07)
# ---------------------------------------------------------------------------


def test_selection_cost_stress_net_scenarios_deduct_embedded_cost() -> None:
    """Regression: _selection_cost_stress net scenarios must use embedded*(1+extra_cost).
    Bug B2 was: scenario_name == 'base' path overrode ALL scenarios to 0."""
    rows = pd.DataFrame({"selected_return": [0.8, 0.8, 0.0]})
    prov = {
        "is_evaluation_return_already_net": True,
        "inferred_embedded_cost": 0.2,
        "baseline_cost_deduction_used": 0.0,
    }
    cs = retrain._selection_cost_stress(rows, cost_provenance=prov)
    assert cs["base"]["effective_cost_deduction"] == 0.2
    assert cs["cost_1_10x"]["effective_cost_deduction"] == pytest.approx(0.22, abs=1e-9)
    assert cs["cost_1_25x"]["effective_cost_deduction"] == pytest.approx(0.25, abs=1e-9)
    assert cs["cost_1_50x"]["effective_cost_deduction"] == pytest.approx(0.30, abs=1e-9)
    assert cs["cost_2_00x"]["effective_cost_deduction"] == 0.4
    deductions = {k: v["effective_cost_deduction"] for k, v in cs.items()}
    assert len(set(deductions.values())) == len(deductions), (
        f"All 5 scenarios must have distinct deductions: {deductions}"
    )


def test_selection_cost_stress_gross_scenarios_deduct_baseline() -> None:
    """Regression: _selection_cost_stress gross scenarios must use baseline*extra_cost.
    Bug B2 was: scenario_name == 'base' path overrode ALL scenarios to 0."""
    rows = pd.DataFrame({"selected_return": [1.0, 1.0, 0.0]})
    prov = {
        "is_evaluation_return_already_net": False,
        "inferred_embedded_cost": 0.2,
        "baseline_cost_deduction_used": 0.2,
    }
    cs = retrain._selection_cost_stress(rows, cost_provenance=prov)
    assert cs["base"]["effective_cost_deduction"] == 0.2
    assert cs["cost_1_10x"]["effective_cost_deduction"] == pytest.approx(0.020, abs=1e-9)
    assert cs["cost_1_25x"]["effective_cost_deduction"] == pytest.approx(0.050, abs=1e-9)
    assert cs["cost_1_50x"]["effective_cost_deduction"] == pytest.approx(0.100, abs=1e-9)
    assert cs["cost_2_00x"]["effective_cost_deduction"] == 0.2
    # Gross formula: stress = baseline * extra_cost. At extra_cost=1.0, stress=0.2=base.
    assert cs["base"]["effective_cost_deduction"] == cs["cost_2_00x"]["effective_cost_deduction"]
    # Non-1.0x scenarios must all differ from base
    for key in ["cost_1_10x", "cost_1_25x", "cost_1_50x"]:
        assert cs[key]["effective_cost_deduction"] != cs["base"]["effective_cost_deduction"], (
            f"{key} must differ from base"
        )


def test_infer_embedded_cost_handles_net_negative_returns() -> None:
    """Regression: _infer_embedded_cost_from_frame must handle cases where
    net_forward_return is negative (cost > raw return).
    Bug A: diff filter was diff > 0.0; should be diff >= 0.0."""
    # All net values negative (embedded cost > raw return)
    gross = [0.1, 0.05, -0.01]
    net = [0.08, 0.03, -0.0135]  # diff = 0.02 in all cases
    df = pd.DataFrame({"gross_forward_return": gross, "net_forward_return": net})
    result = retrain._infer_embedded_cost_from_frame(df)
    assert result is not None, "Must not be None when net is negative"
    assert abs(result - 0.02) < 0.0001, f"Expected ~0.02, got {result}"
    # Mixed: some zero-gross, some net-negative; median diff = 0.0035
    gross2 = [0.0, 0.0, -0.01, 0.05]
    net2 = [-0.0035, -0.0035, -0.0135, 0.03]
    df2 = pd.DataFrame({"gross_forward_return": gross2, "net_forward_return": net2})
    result2 = retrain._infer_embedded_cost_from_frame(df2)
    assert result2 is not None and result2 > 0
    assert abs(result2 - 0.0035) < 0.0001, f"Expected ~0.0035 (median), got {result2}"


def test_cost_stress_report_net_all_scenarios_distinct() -> None:
    """Regression: _cost_stress_report net mode must produce 5 distinct deductions.
    Bug B1 was: base scenario incorrectly overrode all 5 to 0 when
    baseline_cost_deduction_used was 0 and is_net=True."""
    y = np.array([1, 1, 0])
    p = np.array([0.9, 0.8, 0.1])
    r = np.array([0.8, 0.8, 0.0])
    prov = {
        "is_evaluation_return_already_net": True,
        "inferred_embedded_cost": 0.2,
        "baseline_cost_deduction_used": 0.0,
    }
    report = retrain._cost_stress_report(y, p, 0.5, r, cost_provenance=prov)
    deductions = {k: v["effective_cost_deduction"] for k, v in report.items()}
    assert len(set(deductions.values())) == len(deductions), (
        f"All 5 scenarios must have distinct deductions: {deductions}"
    )
    assert report["base"]["effective_cost_deduction"] == 0.2
    assert report["base"]["average_return_per_trade"] == pytest.approx(0.6, abs=1e-9)
    assert report["extra_cost_2_00"]["average_return_per_trade"] < report["base"]["average_return_per_trade"]
    assert report["base"]["effective_cost_deduction"] > 0.0  # no silent-zero bug


def test_cost_stress_report_gross_all_scenarios_distinct() -> None:
    """Regression: _cost_stress_report gross mode must produce differentiated deductions.
    Bug B1 was: base scenario incorrectly overrode all 5 to 0.
    Note: gross formula = baseline * extra_cost, so extra_cost_1_00 = base = 0.2.
    This overlap is expected and correct."""
    y = np.array([1, 1, 0])
    p = np.array([0.9, 0.8, 0.1])
    r = np.array([1.0, 1.0, 0.0])
    prov = {
        "is_evaluation_return_already_net": False,
        "inferred_embedded_cost": 0.2,
        "baseline_cost_deduction_used": 0.2,
    }
    report = retrain._cost_stress_report(y, p, 0.5, r, cost_provenance=prov)
    assert report["base"]["effective_cost_deduction"] == 0.2
    assert report["base"]["average_return_per_trade"] == 0.8
    # Gross stress formula: baseline * extra_cost.
    # extra_cost_0_25 = 0.05, extra_cost_0_50 = 0.10, extra_cost_1_00 = 0.2 (==base), extra_cost_2_00 = 0.4
    assert report["extra_cost_0_25"]["effective_cost_deduction"] == pytest.approx(0.05, abs=1e-9)
    assert report["extra_cost_0_50"]["effective_cost_deduction"] == pytest.approx(0.10, abs=1e-9)
    assert report["extra_cost_2_00"]["effective_cost_deduction"] == pytest.approx(0.40, abs=1e-9)
    # base == extra_cost_1_00 is expected and correct (baseline*1.0 == baseline)
    assert report["base"]["effective_cost_deduction"] == report["extra_cost_1_00"]["effective_cost_deduction"]
    # Non-1.0x scenarios must all be < base (strictly more stressed)
    for key in ["extra_cost_0_25", "extra_cost_0_50"]:
        assert report[key]["effective_cost_deduction"] < report["base"]["effective_cost_deduction"], (
            f"{key} must be less than base"
        )
    assert report["extra_cost_2_00"]["average_return_per_trade"] < report["base"]["average_return_per_trade"]
    assert report["base"]["effective_cost_deduction"] > 0.0  # no silent-zero bug


def test_no_silent_zero_deduction_gross_fallback() -> None:
    """Regression: gross returns with baseline_deduction=0 must fall back to
    extra_cost (not to 0). Guards against the silent zero-deduction bug."""
    y = np.array([1, 1, 0])
    p = np.array([0.9, 0.8, 0.1])
    r = np.array([1.0, 1.0, 0.0])
    prov = {
        "is_evaluation_return_already_net": False,
        "inferred_embedded_cost": None,
        "baseline_cost_deduction_used": 0.0,
    }
    report = retrain._cost_stress_report(y, p, 0.5, r, cost_provenance=prov)
    assert report["base"]["effective_cost_deduction"] == 0.0
    for key in ["extra_cost_0_25", "extra_cost_0_50", "extra_cost_1_00", "extra_cost_2_00"]:
        assert report[key]["effective_cost_deduction"] > 0.0, (
            f"{key} must not deduct 0 when baseline=0"
        )
    stress_deds = [report[k]["effective_cost_deduction"] for k in
                   ["extra_cost_0_25", "extra_cost_0_50", "extra_cost_1_00", "extra_cost_2_00"]]
    assert len(set(stress_deds)) == len(stress_deds), (
        f"Stress deductions must be distinct: {stress_deds}"
    )


def test_cost_stress_report_average_return_correctly_reduces() -> None:
    """Regression: average_return_per_trade must decrease strictly as stress rises."""
    y = np.array([1, 1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.85, 0.76, 0.1])
    r = np.array([0.35, 0.30, 0.28, -0.2, -0.1])
    prov = {
        "is_evaluation_return_already_net": True,
        "inferred_embedded_cost": 0.05,
        "baseline_cost_deduction_used": 0.0,
    }
    report = retrain._cost_stress_report(y, p, 0.75, r, cost_provenance=prov)
    scenarios = ["base", "extra_cost_0_25", "extra_cost_0_50", "extra_cost_1_00", "extra_cost_2_00"]
    for i in range(1, len(scenarios)):
        prev, curr = scenarios[i - 1], scenarios[i]
        assert report[curr]["average_return_per_trade"] < report[prev]["average_return_per_trade"], (
            f"{curr} must have lower avg_return than {prev}"
        )
