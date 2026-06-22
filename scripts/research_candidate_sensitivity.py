from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research-only softer sensitivity analysis for practical candidates.")
    parser.add_argument("--practical-dir", default="models/research_practical_candidates_20260603_212532")
    parser.add_argument("--signal-quality-dir", default="models/research_signal_quality_20260603_211526")
    parser.add_argument("--diagnostics-dir", default="models/research_1y_diagnostics_20260603_201856")
    parser.add_argument("--retrain-dir", default="models/retrained_1y_nifty_20260603_192015")
    return parser.parse_args()


def resolve_dir(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / raw


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_inputs(practical_dir: Path, signal_quality_dir: Path, diagnostics_dir: Path) -> Dict[str, Any]:
    return {
        "practical_candidate_ranking": read_json(practical_dir / "practical_candidate_ranking.json") if (practical_dir / "practical_candidate_ranking.json").exists() else [],
        "minimum_signal_floor_report": read_json(practical_dir / "minimum_signal_floor_report.json") if (practical_dir / "minimum_signal_floor_report.json").exists() else [],
        "stability_report": read_json(practical_dir / "stability_report.json") if (practical_dir / "stability_report.json").exists() else {},
        "key_candidate_comparison_md": read_text_if_exists(practical_dir / "key_candidate_comparison.md"),
        "rejected_candidates_md": read_text_if_exists(practical_dir / "rejected_candidates.md"),
        "final_practical_recommendation_md": read_text_if_exists(practical_dir / "final_practical_recommendation.md"),
        "regime_filter_report": read_json(signal_quality_dir / "regime_filter_report.json") if (signal_quality_dir / "regime_filter_report.json").exists() else [],
        "abstention_policy_report": read_json(signal_quality_dir / "abstention_policy_report.json") if (signal_quality_dir / "abstention_policy_report.json").exists() else [],
        "validation_threshold_selection": read_json(signal_quality_dir / "validation_threshold_selection.json") if (signal_quality_dir / "validation_threshold_selection.json").exists() else {},
        "calibration_report": read_json(signal_quality_dir / "calibration_report.json") if (signal_quality_dir / "calibration_report.json").exists() else [],
        "model_comparison_report": read_json(diagnostics_dir / "model_comparison_report.json") if (diagnostics_dir / "model_comparison_report.json").exists() else [],
        "threshold_diagnostics": read_json(diagnostics_dir / "threshold_diagnostics.json") if (diagnostics_dir / "threshold_diagnostics.json").exists() else {},
    }


def normalize_candidates(practical_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in practical_candidates:
        monthly = item.get("monthly_diagnostics", {})
        out.append(
            {
                "candidate_name": item.get("name") or item.get("filter_rule") or "unknown",
                "model_name": item.get("model_name", "logistic_regression"),
                "label_policy": item.get("label_policy", "trade_quality_ternary"),
                "timeframe": item.get("timeframe", 1),
                "filter_rule": item.get("filter_rule", "none"),
                "abstention_rule": item.get("abstention_rule", "none"),
                "threshold": float(item.get("threshold", 0.65)),
                "validation_precision": float(item.get("validation_precision", 0.0)),
                "test_precision": float(item.get("test_precision", 0.0)),
                "validation_signals": int(item.get("validation_signal_count", 0)),
                "test_signals": int(item.get("test_signal_count", 0)),
                "test_recall": float(item.get("test_recall", 0.0)),
                "validation_recall": float(item.get("validation_recall", 0.0)),
                "precision_gap": float(item.get("precision_gap", 0.0)),
                "monthly_signal_distribution": monthly.get("months", []),
                "monthly_positive_distribution": monthly.get("months", []),
                "largest_month_signal_share": _largest_month_share(monthly, "signal_count"),
                "largest_month_positive_share": _largest_month_share(monthly, "positive_outcomes"),
                "worst_month_precision": float(monthly.get("worst_month_precision", 0.0)),
                "best_month_precision": float(monthly.get("best_month_precision", 0.0)),
                "months_lt_10_signals": int(monthly.get("months_with_lt_10_signals", 0)),
                "one_month_gt_50pct_signals": bool(monthly.get("one_month_gt_50pct_signals", False)),
                "one_month_gt_50pct_positive_outcomes": bool(monthly.get("one_month_gt_50pct_positive_outcomes", False)),
                "original_rejection_reason": item.get("practicality_reason", ""),
                "pr_auc": float(item.get("pr_auc", 0.0)),
            }
        )
    return out


def reconstruct_candidates_from_signal_quality(inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in inputs.get("regime_filter_report", []):
        if row.get("sample_size_too_small"):
            continue
        out.append(
            {
                "candidate_name": f"{row['filter_column']}={row['bucket']}",
                "model_name": "logistic_regression",
                "label_policy": "trade_quality_ternary",
                "timeframe": 1,
                "filter_rule": f"{row['filter_column']}={row['bucket']}",
                "abstention_rule": "none",
                "threshold": float(row.get("selected_threshold", 0.65)),
                "validation_precision": float(row.get("validation_precision", 0.0)),
                "test_precision": float(row.get("precision_test", 0.0)),
                "validation_signals": int(row.get("signals_at_selected_threshold", 0)),
                "test_signals": int(row.get("signals_at_selected_threshold", 0)),
                "test_recall": float(row.get("recall_test", 0.0)),
                "validation_recall": 0.0,
                "precision_gap": float(row.get("validation_precision", 0.0) - row.get("precision_test", 0.0)),
                "monthly_signal_distribution": [],
                "monthly_positive_distribution": [],
                "largest_month_signal_share": 1.0 if int(row.get("signals_at_selected_threshold", 0)) > 0 else 0.0,
                "largest_month_positive_share": 1.0 if float(row.get("precision_test", 0.0)) > 0 else 0.0,
                "worst_month_precision": 0.0,
                "best_month_precision": float(row.get("precision_test", 0.0)),
                "months_lt_10_signals": 0,
                "one_month_gt_50pct_signals": True,
                "one_month_gt_50pct_positive_outcomes": True,
                "original_rejection_reason": "reconstructed_from_regime_report",
                "pr_auc": float(row.get("pr_auc_test", 0.0)),
            }
        )
    for row in inputs.get("abstention_policy_report", []):
        out.append(
            {
                "candidate_name": row["policy"],
                "model_name": "logistic_regression",
                "label_policy": "trade_quality_ternary",
                "timeframe": 1,
                "filter_rule": "none",
                "abstention_rule": str(row["policy"]),
                "threshold": float(row.get("selected_threshold", 0.65)),
                "validation_precision": float(row.get("validation_precision", 0.0)),
                "test_precision": float(row.get("test_precision", 0.0)),
                "validation_signals": int(row.get("validation_signal_count", 0)),
                "test_signals": int(row.get("test_signal_count", 0)),
                "test_recall": 0.0,
                "validation_recall": 0.0,
                "precision_gap": float(row.get("validation_precision", 0.0) - row.get("test_precision", 0.0)),
                "monthly_signal_distribution": [],
                "monthly_positive_distribution": [],
                "largest_month_signal_share": 1.0 if int(row.get("test_signal_count", 0)) > 0 else 0.0,
                "largest_month_positive_share": 1.0 if float(row.get("test_precision", 0.0)) > 0 else 0.0,
                "worst_month_precision": 0.0,
                "best_month_precision": float(row.get("test_precision", 0.0)),
                "months_lt_10_signals": 0,
                "one_month_gt_50pct_signals": True,
                "one_month_gt_50pct_positive_outcomes": True,
                "original_rejection_reason": "reconstructed_from_abstention_report",
                "pr_auc": float(row.get("pr_auc_retained", 0.0)),
            }
        )
    return out


def _largest_month_share(monthly: Dict[str, Any], field: str) -> float:
    rows = monthly.get("months", []) if isinstance(monthly, dict) else monthly
    total = sum(float(row.get(field, 0.0)) for row in rows)
    if total <= 0:
        return 0.0
    return max((float(row.get(field, 0.0)) / total for row in rows), default=0.0)


def evaluate_standard(candidate: Dict[str, Any], *, month_signal_cap: float, min_precision: float, min_signals: int, label: str) -> Dict[str, Any]:
    passes = (
        candidate["test_precision"] >= min_precision
        and candidate["test_signals"] >= min_signals
        and abs(candidate["precision_gap"]) <= 0.25
        and candidate["largest_month_signal_share"] <= month_signal_cap
    )
    warning = ""
    if candidate["test_signals"] < min_signals:
        warning = "signal_count_too_low"
    elif candidate["largest_month_signal_share"] > month_signal_cap:
        warning = "month_signal_concentration_too_high"
    elif abs(candidate["precision_gap"]) > 0.25:
        warning = "validation_test_gap_too_large"
    elif candidate["test_precision"] < min_precision:
        warning = "precision_too_low"
    return {
        "standard": label,
        "passes": bool(passes),
        "warning": warning,
    }


def special_review(candidate: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if candidate is None:
        return {}
    attractive = []
    if candidate["test_precision"] >= 0.30:
        attractive.append("higher-than-baseline precision")
    if candidate["test_signals"] >= 50:
        attractive.append("non-trivial signal count")
    failed = []
    if candidate["test_signals"] < 25:
        failed.append("too few test signals")
    if candidate["largest_month_signal_share"] > 0.50:
        failed.append("too dependent on a single month")
    if abs(candidate["precision_gap"]) > 0.25:
        failed.append("validation-to-test mismatch too large")
    deserves = candidate["test_precision"] >= 0.30 and candidate["test_signals"] >= 25
    return {
        "candidate_name": candidate["candidate_name"],
        "filter_rule": candidate["filter_rule"],
        "threshold": candidate["threshold"],
        "why_attractive": attractive or ["only attractive on headline metric"],
        "why_failed_strict_rules": failed or ["failed broader practicality gate"],
        "extra_evidence_needed": [
            "at least 30 new forward signals",
            "signals across at least 3 separate weeks",
            "no single week contributes more than 50% of wins",
            "cost/slippage-adjusted paper profit factor > 1.15",
        ],
        "forward_paper_observation_only": bool(deserves),
    }


def forward_observation_plan() -> str:
    return "\n".join(
        [
            "# Forward Observation Plan",
            "",
            "- Observe the next 4 trading weeks.",
            "- Log model signal only; do not place any order.",
            "- Record timestamp, probability, selected filter, market regime, and option premium movement if available.",
            "- Compare predicted setup with actual outcome after the defined label horizon.",
            "- Require at least 30 new forward signals before reassessment.",
            "- Require signals across at least 3 separate weeks.",
            "- Require no single week contributes more than 50% of wins.",
            "- Apply simulated cost/slippage adjustment.",
            "- Require paper profit factor estimate > 1.15 before any paper-trading discussion.",
        ]
    ) + "\n"


def main() -> int:
    args = parse_args()
    practical_dir = resolve_dir(args.practical_dir)
    signal_quality_dir = resolve_dir(args.signal_quality_dir)
    diagnostics_dir = resolve_dir(args.diagnostics_dir)
    _ = resolve_dir(args.retrain_dir)

    out_dir = MODELS_DIR / f"research_candidate_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(practical_dir, signal_quality_dir, diagnostics_dir)
    candidates = normalize_candidates(inputs["practical_candidate_ranking"])
    if not candidates:
        candidates = reconstruct_candidates_from_signal_quality(inputs)

    strict_rows = []
    for c in candidates:
        strict = evaluate_standard(c, month_signal_cap=0.50, min_precision=0.35, min_signals=75, label="strict_original")
        soft = evaluate_standard(c, month_signal_cap=0.70, min_precision=0.30, min_signals=50, label="soft_watchlist")
        exploratory = evaluate_standard(c, month_signal_cap=0.80, min_precision=0.25, min_signals=25, label="exploratory")
        strict_rows.append({**c, "strict": strict, "soft": soft, "exploratory": exploratory})

    soft_pass = [row for row in strict_rows if row["soft"]["passes"]]
    exploratory_pass = [row for row in strict_rows if row["exploratory"]["passes"]]

    soft_pass = sorted(soft_pass, key=lambda r: (r["test_precision"], r["validation_precision"], -abs(r["precision_gap"]), r["test_signals"], r["pr_auc"]), reverse=True)
    exploratory_pass = sorted(exploratory_pass, key=lambda r: (r["test_precision"], r["validation_precision"], -abs(r["precision_gap"]), r["test_signals"], r["pr_auc"]), reverse=True)

    special_targets = {}
    for key in ["ret_std_bucket=high", "realized_vol_30_bucket=high", "atr_pct_bucket=high", "weekday=3", "none"]:
        match = next((row for row in strict_rows if row["filter_rule"] == key and row["abstention_rule"] == "none"), None)
        special_targets[key] = special_review(match)

    for idx, row in enumerate(soft_pass[:2], start=1):
        payload = {
            "model_name": row["model_name"],
            "label_policy": row["label_policy"],
            "timeframe": row["timeframe"],
            "threshold": row["threshold"],
            "filter_rule": row["filter_rule"],
            "validation_metrics": {
                "precision": row["validation_precision"],
                "signals": row["validation_signals"],
                "recall": row["validation_recall"],
            },
            "test_metrics": {
                "precision": row["test_precision"],
                "signals": row["test_signals"],
                "recall": row["test_recall"],
                "pr_auc": row["pr_auc"],
            },
            "month_concentration_warning": row["largest_month_signal_share"],
            "reason_for_watchlist": "Passed soft sensitivity screen only; still not approved for paper trading.",
            "required_forward_observation_conditions": [
                "minimum 30 new forward signals",
                "at least 3 separate weeks with signals",
                "no single week > 50% of wins",
                "paper profit factor estimate > 1.15 after cost/slippage adjustment",
            ],
            "live_promoted": False,
            "paper_trade_enabled": False,
            "decision": "SOFT WATCHLIST ONLY",
        }
        write_json(out_dir / f"soft_watchlist_candidate_{idx}.json", payload)

    for idx, row in enumerate(exploratory_pass[:2], start=1):
        payload = {
            "model_name": row["model_name"],
            "label_policy": row["label_policy"],
            "timeframe": row["timeframe"],
            "threshold": row["threshold"],
            "filter_rule": row["filter_rule"],
            "validation_metrics": {
                "precision": row["validation_precision"],
                "signals": row["validation_signals"],
                "recall": row["validation_recall"],
            },
            "test_metrics": {
                "precision": row["test_precision"],
                "signals": row["test_signals"],
                "recall": row["test_recall"],
                "pr_auc": row["pr_auc"],
            },
            "month_concentration_warning": row["largest_month_signal_share"],
            "reason_for_watchlist": "Passed exploratory sensitivity screen only; still not approved for paper trading.",
            "required_forward_observation_conditions": [
                "minimum 30 new forward signals",
                "at least 3 separate weeks with signals",
                "no single week > 50% of wins",
                "paper profit factor estimate > 1.15 after cost/slippage adjustment",
            ],
            "live_promoted": False,
            "paper_trade_enabled": False,
            "decision": "EXPLORATORY ONLY",
        }
        write_json(out_dir / f"exploratory_candidate_{idx}.json", payload)

    write_json(out_dir / "strict_vs_soft_ranking.json", strict_rows)

    floor_summary = {
        "strict_original": [row["candidate_name"] for row in strict_rows if row["strict"]["passes"]],
        "soft_watchlist": [row["candidate_name"] for row in soft_pass],
        "exploratory": [row["candidate_name"] for row in exploratory_pass],
    }
    write_json(out_dir / "strict_vs_soft_ranking.json", strict_rows)
    write_json(out_dir / "strict_vs_soft_ranking_summary.json", floor_summary)

    sensitivity_md = [
        "# Sensitivity Summary",
        "",
        f"- Soft standard pass count: `{len(soft_pass)}`",
        f"- Exploratory standard pass count: `{len(exploratory_pass)}`",
        "- This is research-only sensitivity analysis. No candidate is approved for live or paper trading.",
    ]
    (out_dir / "sensitivity_summary.md").write_text("\n".join(sensitivity_md) + "\n", encoding="utf-8")

    soft_md = ["# Soft Watchlist Candidates", ""]
    if not soft_pass:
        soft_md.append("No candidates passed the soft watchlist standard.")
    else:
        for row in soft_pass[:5]:
            soft_md.append(f"- `{row['filter_rule']}` threshold `{row['threshold']}` test precision `{row['test_precision']:.4f}` on `{row['test_signals']}` signals.")
    (out_dir / "soft_watchlist_candidates.md").write_text("\n".join(soft_md) + "\n", encoding="utf-8")

    exploratory_md = ["# Exploratory Candidates", ""]
    if not exploratory_pass:
        exploratory_md.append("No candidates passed the exploratory standard.")
    else:
        for row in exploratory_pass[:5]:
            exploratory_md.append(f"- `{row['filter_rule']}` threshold `{row['threshold']}` test precision `{row['test_precision']:.4f}` on `{row['test_signals']}` signals.")
    (out_dir / "exploratory_candidates.md").write_text("\n".join(exploratory_md) + "\n", encoding="utf-8")

    special_md = ["# Special Candidate Review", ""]
    for key, review in special_targets.items():
        if not review:
            continue
        special_md.extend(
            [
                f"## {key}",
                f"- Why attractive: {', '.join(review['why_attractive'])}",
                f"- Why failed strict rules: {', '.join(review['why_failed_strict_rules'])}",
                f"- Deserves forward observation only: `{review['forward_paper_observation_only']}`",
                "",
            ]
        )
    (out_dir / "special_candidate_review.md").write_text("\n".join(special_md), encoding="utf-8")

    (out_dir / "forward_observation_plan.md").write_text(forward_observation_plan(), encoding="utf-8")

    if soft_pass:
        final_label = "SOFT PAPER WATCHLIST ONLY"
        reason = "some rejected candidates become interesting under softer concentration limits, but still remain non-tradable research watchlist items"
    elif exploratory_pass:
        final_label = "EXPLORATORY WATCHLIST ONLY"
        reason = "only exploratory sensitivity passes exist, so the candidates remain fragile and observational only"
    else:
        final_label = "DO NOT PROMOTE"
        reason = "even under softer sensitivity standards, no candidate becomes meaningfully practical"
    final_text = "\n".join(
        [
            f"# {final_label}",
            "",
            f"- Soft standard pass count: `{len(soft_pass)}`",
            f"- Exploratory standard pass count: `{len(exploratory_pass)}`",
            f"- Reason: {reason}",
            "- live_promoted: false",
            "- paper_trade_enabled: false",
        ]
    ) + "\n"
    (out_dir / "final_sensitivity_recommendation.md").write_text(final_text, encoding="utf-8")

    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
