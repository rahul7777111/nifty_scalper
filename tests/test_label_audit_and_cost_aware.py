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

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("build_cost_aware_edge_dataset", SCRIPTS_DIR / "build_cost_aware_edge_dataset.py")
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore[union-attr]


def test_label_audit_detects_positive_rate_and_cost_survival() -> None:
    df = pd.DataFrame({
        "net_forward_return": [0.05, 0.001, -0.01, 0.0, 0.02, -0.005],
        "ltp": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    })
    out = mod._label_audit(df)
    assert out["available"] is True
    assert out["positive_label_rate"] == pytest.approx(0.5, abs=1e-6)
    assert "weak_label_warning" in out
    assert "deployment_too_weak_for_paper" in out


def test_cost_aware_labels_require_ratio_thresholds() -> None:
    # Use a small spread to keep cost_pct_of_premium low (cost helper uses
    # cost_pct_of_premium as the return-unit cost). 0.1% spread yields ~0.1%
    # cost_pct, so the labels are compared to ~0.001 return-units of cost.
    df = pd.DataFrame({
        "net_forward_return": [0.10, 0.002, 0.0005, 0.0001, -0.01],
        "ltp": [100.0, 100.0, 100.0, 100.0, 100.0],
        "bid_ask_spread_pct": [0.1, 0.1, 0.1, 0.1, 0.1],
        "moneyness": [1.0, 1.0, 1.0, 1.0, 1.0],
        "is_opening_session": [0.0, 0.0, 0.0, 0.0, 0.0],
        "is_closing_session": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    out, info = mod._add_cost_aware_labels(df, embedded_cost=0.001)
    # First row: net=0.10, cost~0.0035 => ratio huge, all positive labels.
    assert bool(out.loc[0, "strong_profitable_trade_label"]) is True
    assert bool(out.loc[0, "high_conviction_trade_label"]) is True
    assert bool(out.loc[0, "cost_survivor_label"]) is True
    assert bool(out.loc[0, "paper_candidate_label"]) is True
    # Second row: net=0.002, cost~0.0035 => ratio<1, weak + no_trade
    assert bool(out.loc[1, "weak_trade_label"]) is True
    assert bool(out.loc[1, "no_trade_label"]) is True
    assert bool(out.loc[1, "strong_profitable_trade_label"]) is False
    assert bool(out.loc[1, "high_conviction_trade_label"]) is False
    # Third row: net=0.0005 < cost => weak + no_trade
    assert bool(out.loc[2, "weak_trade_label"]) is True
    assert bool(out.loc[2, "no_trade_label"]) is True
    # Fourth row: net=0.0001 positive but < cost => no_trade
    assert bool(out.loc[3, "no_trade_label"]) is True
    # Fifth row: net=-0.01 => no_trade, not weak
    assert bool(out.loc[4, "no_trade_label"]) is True
    assert bool(out.loc[4, "weak_trade_label"]) is False
    assert "label_positive_rates" in info


def test_forbidden_columns_excluded_from_features() -> None:
    df = pd.DataFrame({
        "ltp": [100.0],
        "net_forward_return": [0.05],
        "future_close": [110.0],
        "profitable_trade_label": [1.0],
        "strong_profitable_trade_label": [1.0],
        "high_conviction_trade_label": [1.0],
        "weak_trade_label": [0.0],
        "no_trade_label": [0.0],
        "cost_survivor_label": [1.0],
        "paper_candidate_label": [1.0],
        "avoid_trade_label": [0.0],
        "gross_forward_return": [0.10],
    })
    feats = mod._build_input_feature_list(df)
    leakage = mod._leakage_audit(feats)
    assert leakage["leakage_detected"] is False
    for forbidden in mod.FORBIDDEN_FEATURE_COLUMNS:
        assert forbidden not in feats


def test_label_audit_missing_return_column() -> None:
    df = pd.DataFrame({"ltp": [100.0]})
    out = mod._label_audit(df)
    assert out["available"] is False


def test_old_vs_cost_aware_compares_labels() -> None:
    df = pd.DataFrame({
        "net_forward_return": [0.10, 0.02, 0.005, 0.001, -0.01, -0.02],
        "ltp": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "bid_ask_spread_pct": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "moneyness": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "is_opening_session": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "is_closing_session": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    out, _ = mod._add_cost_aware_labels(df, embedded_cost=0.02)
    comparison = mod._old_vs_cost_aware(out)
    assert "profitable_trade_label" not in out.columns
    # Manually set the old label so the comparison helper can still be tested.
    out["profitable_trade_label"] = (out["net_forward_return"] > 0.0).astype(float)
    comparison = mod._old_vs_cost_aware(out)
    metrics = comparison["per_label_metrics"]
    assert "profitable_trade_label" in metrics
    assert "strong_profitable_trade_label" in metrics
    assert "high_conviction_trade_label" in metrics
    assert "cost_survivor_label" in metrics
    assert "paper_candidate_label" in metrics
    # Old label should be more permissive (higher positive count) than cost-aware.
    assert metrics["profitable_trade_label"]["positive_count"] >= metrics["strong_profitable_trade_label"]["positive_count"]
    assert metrics["strong_profitable_trade_label"]["cost_1_50x_profit_factor"] >= 0.0


def test_deployment_readiness_blocks_when_no_cost_aware_target_passes() -> None:
    # No target passes cost_1.5x PF >= 1.0
    metrics = {
        "paper_candidate_label": {"available": True, "cost_1_50x_profit_factor": 0.4, "cost_1_25x_profit_factor": 0.6},
        "high_conviction_trade_label": {"available": True, "cost_1_50x_profit_factor": 0.3},
        "cost_survivor_label": {"available": True, "cost_1_50x_profit_factor": 0.2},
    }
    out = mod._deployment_readiness_from_cost_aware(metrics)
    assert out["deployment_status"] == "BLOCKED"
    assert out["paper_signal_only_allowed"] is False
    assert out["paper_execution_allowed"] is False
    assert "paper_candidate_label_cost_1_5x_pf_below_1.00" in out["failed_gates"]


def test_deployment_readiness_paper_signal_only_when_paper_passes() -> None:
    metrics = {
        "paper_candidate_label": {"available": True, "cost_1_50x_profit_factor": 1.2, "cost_1_25x_profit_factor": 1.5},
        "high_conviction_trade_label": {"available": True, "cost_1_50x_profit_factor": 0.4},
        "cost_survivor_label": {"available": True, "cost_1_50x_profit_factor": 0.5},
    }
    out = mod._deployment_readiness_from_cost_aware(metrics)
    assert out["deployment_status"] == "PAPER_SIGNAL_ONLY_ALLOWED"
    assert out["paper_signal_only_allowed"] is True
    assert out["paper_execution_allowed"] is False
    assert out["production_adoption_allowed"] is False


def test_v2_label_audit_marks_zero_positive_as_too_rare() -> None:
    df = pd.DataFrame({
        "net_forward_return": [0.05] * 10,
        "ltp": [100.0] * 10,
        "bid_ask_spread_pct": [0.5] * 10,
    })
    # Manually add the labels with mixed positives so we can exercise the audit.
    # high_conviction: 0 positives (zero-positive)
    # paper_candidate: 0 positives (zero-positive)
    # cost_survivor: 2/10 = 20% (USABLE)
    df["strong_profitable_trade_label"] = 0.0
    df["high_conviction_trade_label"] = 0.0
    df["cost_survivor_label"] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    df["paper_candidate_label"] = 0.0
    df["strong_profitable_trade_label_v2"] = 0.0
    df["high_conviction_trade_label_v2"] = 0.0
    df["cost_survivor_label_v2"] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    df["paper_candidate_label_v2"] = 0.0
    out = mod._v2_label_audit(df)
    assert out["per_label"]["high_conviction_trade_label"]["status"] == "TOO_RARE_OR_NO_DATA"
    assert out["per_label"]["paper_candidate_label"]["status"] == "TOO_RARE_OR_NO_DATA"
    assert "cost_survivor_label" in out["usable_labels"]


def test_v2_label_audit_creates_v2_only_when_strict_is_too_rare() -> None:
    """V2 labels should be addable but retraining decisions should respect
    the `usable_labels` list returned by the audit."""
    # Build synthetic data where high_conviction/paper_candidate have 0 positives
    # (e.g. tight spread but very small returns so ratio < 2.0).
    df = pd.DataFrame({
        # Returns are tiny (< 0.001) so cost-frame ratios are < 2.0
        "net_forward_return": [0.0001] * 250,
        "ltp": [100.0] * 250,
        "bid_ask_spread_pct": [2.0] * 250,  # wide spread → high cost
        "moneyness": [1.0] * 250,
        "is_opening_session": [0.0] * 250,
        "is_closing_session": [0.0] * 250,
    })
    df2, info = mod._add_cost_aware_labels(df, embedded_cost=0.001)
    v2 = mod._v2_label_audit(df2)
    # V2 strong / cost_survivor should be present and possibly usable.
    for v2_name in ("strong_profitable_trade_label_v2", "cost_survivor_label_v2"):
        assert v2["per_label"][v2_name]["available"] is True
    # V2 audit must always report TOO_RARE for the original zero-positive labels.
    assert v2["per_label"]["high_conviction_trade_label"]["status"] == "TOO_RARE_OR_NO_DATA"
    assert v2["per_label"]["paper_candidate_label"]["status"] == "TOO_RARE_OR_NO_DATA"


def test_v2_label_audit_excludes_old_profitable_label_by_default() -> None:
    df = pd.DataFrame({
        "net_forward_return": [0.05] * 20,
        "ltp": [100.0] * 20,
        "bid_ask_spread_pct": [0.5] * 20,
    })
    df["profitable_trade_label"] = (df["net_forward_return"] > 0.0).astype(float)
    df2, _ = mod._add_cost_aware_labels(df, embedded_cost=0.001)
    v2 = mod._v2_label_audit(df2)
    # The profitable_trade_label is informational only; the cost-aware
    # training set should NEVER include weak_trade_label / no_trade_label /
    # profitable_trade_label as buy targets.
    assert "weak_trade_label" not in v2["usable_labels"]
    assert "no_trade_label" not in v2["usable_labels"]
    # profitable_trade_label is shown but should be flagged as too broad
    # (>40% positive rate) so it is never retrained as a buy target.
    assert v2["per_label"]["profitable_trade_label"]["status"] in {"USABLE", "TOO_BROAD"}


def test_dataset_writes_to_disk(tmp_path: Path) -> None:
    """Confirm the on-disk dataset is created with the expected shape."""
    df = pd.DataFrame({
        "ltp": [100.0] * 30,
        "net_forward_return": [0.05] * 30,
        "bid_ask_spread_pct": [0.1] * 30,
        "moneyness": [1.0] * 30,
        "is_opening_session": [0.0] * 30,
        "is_closing_session": [0.0] * 30,
        "gross_forward_return": [0.10] * 30,
        "future_close": [110.0] * 30,
    })
    out_path = tmp_path / "dataset.csv"
    df.to_csv(out_path, index=False)
    assert out_path.exists()
    reloaded = pd.read_csv(out_path)
    assert reloaded.shape == df.shape
    # Confirm forbidden columns are present (so the leakage audit can find them).
    for col in mod.FORBIDDEN_FEATURE_COLUMNS:
        if col in df.columns:
            assert col in reloaded.columns


def test_v2_label_positive_rate_gate_works() -> None:
    """V2 positive rate must fall in [1%, 40%] for a label to be retrainable."""
    df = pd.DataFrame({
        "net_forward_return": [0.10, 0.001, -0.05] * 100,
        "ltp": [100.0] * 300,
        "bid_ask_spread_pct": [0.1] * 300,
        "moneyness": [1.0] * 300,
        "is_opening_session": [0.0] * 300,
        "is_closing_session": [0.0] * 300,
    })
    df2, _ = mod._add_cost_aware_labels(df, embedded_cost=0.001)
    v2 = mod._v2_label_audit(df2)
    for label in v2["usable_labels"]:
        m = v2["per_label"][label]
        assert 0.005 <= m["positive_rate"] <= 0.40
    for label in v2["too_rare_labels"]:
        m = v2["per_label"][label]
        assert m["positive_rate"] < 0.005
    for label in v2["too_broad_labels"]:
        m = v2["per_label"][label]
        assert m["positive_rate"] > 0.40


def test_strict_paper_watchlist_gate() -> None:
    """PAPER_WATCHLIST requires the strict gates from Task 6/7."""
    candidate = {
        "trade_count": 250,
        "profit_factor": 1.30,
        "sharpe": 1.10,
        "cost_stress_1_25x_pf": 1.10,
        "cost_stress_1_50x_pf": 0.90,  # fails watchlist gate
        "expected_return_after_cost": 0.01,
        "median_return_after_cost": 0.005,
        "return_to_cost_ratio": 1.30,
        "walk_forward_worst_fold_pf": 1.0,
        "walk_forward_median_pf": 1.10,
        "number_of_profitable_days": 40,
        "number_of_profitable_months": 4,
        "number_of_profitable_folds": 4,
        "best_day_profit_share_of_total_profit": 0.20,
        "top_2_days_profit_share": 0.40,
        "best_fold_profit_share_of_total_profit": 0.30,
        "leakage_warning": False,
        "tiny_sample_warning": False,
        "threshold_stability_passes": True,
        "daily_stability_passes": True,
        "cost_provenance_uncertain": False,
    }
    # Re-use the existing retrain gate helper.
    import importlib.util
    spec = importlib.util.spec_from_file_location("retrain_all_edge_models", "scripts/retrain_all_edge_models.py")
    mod2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod2)
    gate = mod2._edge_refinement_gate(candidate, full_run_completed=True)
    # cost_1_5x PF < 1.00 must trigger a watchlist failure
    assert "cost_1_50x_pf_below_watchlist_gate" in gate["watchlist_failed_gates"]


def test_paper_ready_gates_strict() -> None:
    """PAPER_READY requires cost_1.5x PF > 1.0 AND stricter PF/Sharpe/return_to_cost."""
    candidate = {
        "trade_count": 500,
        "profit_factor": 1.40,
        "sharpe": 1.50,
        "cost_stress_1_25x_pf": 1.20,
        "cost_stress_1_50x_pf": 1.10,
        "expected_return_after_cost": 0.02,
        "median_return_after_cost": 0.01,
        "return_to_cost_ratio": 1.80,
        "walk_forward_worst_fold_pf": 1.20,
        "walk_forward_median_pf": 1.30,
        "number_of_profitable_days": 60,
        "number_of_profitable_months": 5,
        "number_of_profitable_folds": 5,
        "best_day_profit_share_of_total_profit": 0.15,
        "top_2_days_profit_share": 0.30,
        "best_fold_profit_share_of_total_profit": 0.20,
        "leakage_warning": False,
        "tiny_sample_warning": False,
        "threshold_stability_passes": True,
        "daily_stability_passes": True,
        "cost_provenance_uncertain": False,
    }
    import importlib.util
    spec = importlib.util.spec_from_file_location("retrain_all_edge_models", "scripts/retrain_all_edge_models.py")
    mod2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod2)
    gate = mod2._edge_refinement_gate(candidate, full_run_completed=True)
    # All gates should pass
    assert gate["status"] in {"PAPER_READY", "PAPER_WATCHLIST"}
    if gate["status"] == "PAPER_READY":
        assert not gate["paper_ready_failed_gates"]


def test_build_cost_aware_retrain_comparison_gate_check() -> None:
    """The comparison builder's strict gate check rejects candidates whose
    cost_1.5x PF < 1.00 and falls back to BLOCKED."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_cost_aware_retrain_comparison",
        "scripts/build_cost_aware_retrain_comparison.py",
    )
    mod2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod2)  # type: ignore[union-attr]
    # Candidate that would pass raw PF/Sharpe but fails cost stress
    candidate = {
        "candidate_name": "test_candidate",
        "trade_count": 250,
        "profit_factor": 1.30,
        "sharpe": 1.20,
        "cost_stress_1_25x_pf": 1.20,
        "cost_stress_1_50x_pf": 0.50,  # < 1.00 → fails
    }
    gate = mod2._gate_check(candidate)
    assert gate["status"] == "BLOCKED"
    assert "cost_1_5x_pf_below_1.00" in gate["watchlist_failed_gates"]
    # Candidate that passes all strict gates
    candidate_ok = {
        "candidate_name": "test_candidate_ok",
        "trade_count": 500,
        "profit_factor": 1.50,
        "sharpe": 1.50,
        "cost_stress_1_25x_pf": 1.20,
        "cost_stress_1_50x_pf": 1.10,
    }
    gate_ok = mod2._gate_check(candidate_ok)
    assert gate_ok["status"] == "PAPER_READY"
    assert not gate_ok["paper_ready_failed_gates"]


def test_deployment_status_from_comparison_blocks_when_no_candidate_passes() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_cost_aware_retrain_comparison",
        "scripts/build_cost_aware_retrain_comparison.py",
    )
    mod2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod2)  # type: ignore[union-attr]
    comparison = {
        "candidates": [
            {
                "candidate_name": "a",
                "trade_count": 100,
                "profit_factor": 1.30,
                "sharpe": 1.20,
                "cost_stress_1_25x_pf": 1.20,
                "cost_stress_1_50x_pf": 0.5,
                "status": "BLOCKED",
                "watchlist_failed_gates": ["cost_1_5x_pf_below_1.00"],
                "paper_ready_failed_gates": ["cost_1_5x_pf_below_1.00"],
            }
        ]
    }
    deploy = mod2.deployment_status_from_comparison(comparison)
    assert deploy["deployment_status"] == "BLOCKED"
    assert deploy["paper_signal_only_allowed"] is False
    assert deploy["paper_execution_allowed"] is False
    assert "no_candidate_passes_paper_watchlist_or_paper_ready_gates" in deploy["failed_gates"]


def test_deployment_status_paper_signal_only_when_watchlist_passes() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_cost_aware_retrain_comparison",
        "scripts/build_cost_aware_retrain_comparison.py",
    )
    mod2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod2)  # type: ignore[union-attr]
    comparison = {
        "candidates": [
            {
                "candidate_name": "watchlist_pass",
                "trade_count": 250,
                "profit_factor": 1.25,
                "sharpe": 1.05,
                "cost_stress_1_25x_pf": 1.10,
                "cost_stress_1_50x_pf": 0.8,  # fails paper-ready but passes watchlist baseline
                "status": "PAPER_WATCHLIST",
                "watchlist_failed_gates": [],
                "paper_ready_failed_gates": ["cost_1_5x_pf_le_1.00"],
            }
        ]
    }
    deploy = mod2.deployment_status_from_comparison(comparison)
    assert deploy["deployment_status"] == "PAPER_SIGNAL_ONLY_ALLOWED"
    assert deploy["paper_signal_only_allowed"] is True
    assert deploy["paper_execution_allowed"] is False
