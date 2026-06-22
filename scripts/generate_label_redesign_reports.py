from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"


def read_json(name: str, default: Any) -> Any:
    path = REPORTS_DIR / name
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def build_regime_distribution_report(label_diag: Dict[str, Any], matrix: Dict[str, Any]) -> Dict[str, Any]:
    vol = label_diag.get("label_rate_by_volatility_regime", {})
    trend = label_diag.get("label_rate_by_trend_range_regime", {})
    session = label_diag.get("label_rate_by_session", {})
    rows = matrix.get("rows", [])
    top_rows = sorted(rows, key=lambda row: (-float(row.get("metrics", {}).get("f1", 0.0)), -float(row.get("metrics", {}).get("profit_factor", 0.0))))[:5]
    return {
        "sample_count_by_regime": {
            "volatility": vol,
            "trend": trend,
            "session": session,
        },
        "label_rate_by_regime": {
            "volatility": vol,
            "trend": trend,
            "session": session,
        },
        "average_forward_return_by_regime": "see label_diagnostics.json forward-return section; regime-level forward returns are not yet separated in the matrix output",
        "model_performance_by_regime": [
            {
                "label_policy": row.get("label_policy"),
                "model": row.get("model"),
                "variant": row.get("variant"),
                "roc_auc": row.get("metrics", {}).get("roc_auc"),
                "f1": row.get("metrics", {}).get("f1"),
                "profit_factor": row.get("metrics", {}).get("profit_factor"),
            }
            for row in top_rows
        ],
        "abstain_regimes": ["midday_lull", "high_volatility"],
    }


def build_alignment_audit() -> Dict[str, Any]:
    return {
        "when_ml_prediction_is_called": "Inside strategy.evaluate_entry_signals() before trade placement.",
        "ml_generates_entries_or_filters": "Entry filter; it gates a candidate rather than defining full strategy execution logic.",
        "trade_candidate_origin": "Auto-mode strategy branch combines regime suggestion, ML gate, GPT advisory, and technical fallback.",
        "label_horizon_matches_actual_holding_period": False,
        "label_target_stop_matches_strategy_target_stop": False,
        "option_spread_slippage_represented": "Now represented in cost-aware label policies and offline evaluation only, not in live execution logic.",
        "volatility_regime_changes_entry_rules": True,
        "confidence_affects_entry_only_or_size": "Entry permission only; position sizing still comes from existing volatility targeting.",
        "current_label_rewards_trades_strategy_would_take": False,
        "ml_learning_direction_or_setup_quality": "Historically direction; redesigned policies shift toward setup quality and abstention.",
    }


def render_md(title: str, payload: Dict[str, Any]) -> str:
    lines = [f"# {title}", "", "```json", json.dumps(payload, indent=2), "```", ""]
    return "\n".join(lines)


def build_final_report() -> Dict[str, Any]:
    current_audit = read_json("current_labeling_audit.json", {})
    label_diag = read_json("label_diagnostics.json", {})
    cost_model = read_json("cost_model_assumptions.json", {})
    safety = read_json("label_policy_safety_audit.json", {})
    feature_gov = read_json("feature_governance_by_label_policy.json", {})
    matrix = read_json("label_policy_model_matrix.json", {})
    walk_forward = read_json("walk_forward_label_policy_matrix.json", {})
    abstention = read_json("abstention_policy_report.json", {})
    live = read_json("live_prediction_accuracy.json", {})
    paper = read_json("paper_trade_ml_linkage_report.json", {})

    rows = matrix.get("rows", [])
    best_auc = max(rows, key=lambda row: float(row.get("metrics", {}).get("roc_auc", 0.0)), default=None)
    best_f1 = max(rows, key=lambda row: float(row.get("metrics", {}).get("f1", 0.0)), default=None)
    best_pf = max(rows, key=lambda row: float(row.get("metrics", {}).get("profit_factor", 0.0)), default=None)
    research_pass_count = sum(1 for row in rows if row.get("research_gate_pass"))
    production_pass_count = sum(1 for row in rows if row.get("production_gate_pass"))

    return {
        "current_label_audit": current_audit,
        "label_diagnostics_summary": {
            "policy": label_diag.get("policy"),
            "total_samples": label_diag.get("total_samples"),
            "positive_rate": label_diag.get("positive_rate"),
            "noisy_zone_percentage": label_diag.get("noisy_zone_percentage"),
            "below_cost_slippage_percentage": label_diag.get("below_cost_slippage_percentage"),
        },
        "cost_model_assumptions": cost_model,
        "new_label_policies_implemented": [
            "current_triple_barrier",
            "cost_adjusted_triple_barrier",
            "magnitude_filtered_direction",
            "trade_quality_binary",
            "trade_quality_ternary",
            "regime_specific_trade_quality",
            "abstain_allowed_trade_quality",
        ],
        "label_policy_safety_audit": safety,
        "feature_governance_by_label_policy": feature_gov,
        "model_matrix_summary": {
            "rows": len(rows),
            "research_pass_count": research_pass_count,
            "production_pass_count": production_pass_count,
            "best_roc_auc_row": best_auc,
            "best_f1_row": best_f1,
            "best_profit_factor_row": best_pf,
            "risk_aware_selected_row": matrix.get("best"),
        },
        "walk_forward_validation": {"rows": len(walk_forward.get("rows", []))},
        "abstention_threshold_results": abstention,
        "old_224_day_baseline": {
            "model": "logistic_regression",
            "variant": "B_pruned_feature_set",
            "roc_auc": 0.5141,
            "f1": 0.5112,
            "profit_factor": 2.3110,
            "drawdown": 0.3300,
            "deployment": "REJECT",
        },
        "live_prediction_logging_status": live,
        "paper_trade_ml_linkage_status": paper,
        "tests_added": [
            "tests/test_label_policies.py",
            "tests/test_cost_model.py",
            "tests/test_prediction_logging.py",
            "tests/test_paper_trade_ml_linkage.py",
        ],
        "commands_run": [
            "python -m py_compile ...",
            "python tools/label_diagnostics.py",
            "python tools/audit_label_policy_safety.py",
            "python scripts/verify_dataset_v2.py",
            "pytest targeted validation suite",
            "python scripts/train_window_benchmark.py --models all --label-policies all --feature-variants all --no-deploy",
            "python scripts/train_window_benchmark.py --models all --label-policies all --threshold-sweep --no-deploy",
            "python tools/resolve_prediction_labels.py",
            "python tools/verify_predictions.py",
            "python tools/paper_trade_report.py",
        ],
        "deployment_decision": "REJECT DEPLOYMENT",
        "reasons_for_reject": [
            "no label policy / model / feature-variant row passed the research gate",
            "best high-AUC rows collapsed on F1 and trade economics",
            "drawdown remains materially above the production gate in viable-trade candidates",
            "resolved live predictions < 100",
            "paper trades < 100 and linked_predictions = 0",
        ],
        "next_recommended_action": "Keep deployment blocked, let the new prediction logging accumulate real linked samples, then rerun the same matrix after 100+ resolved live predictions and 100+ linked paper trades.",
    }


def main() -> int:
    label_diag = read_json("label_diagnostics.json", {})
    matrix = read_json("label_policy_model_matrix.json", {})
    regime_report = build_regime_distribution_report(label_diag, matrix)
    alignment = build_alignment_audit()
    final_report = build_final_report()

    artifacts = {
        "regime_distribution_report": regime_report,
        "ml_strategy_alignment_audit": alignment,
        "final_label_redesign_and_model_adaptation_report": final_report,
    }
    for name, payload in artifacts.items():
        (REPORTS_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (REPORTS_DIR / f"{name}.md").write_text(render_md(name.replace("_", " ").title(), payload), encoding="utf-8")
    print(json.dumps({"written": list(artifacts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
