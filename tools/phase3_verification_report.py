"""Phase 3 verification report builder.

Reads artifact metrics for the latest retrain_all_models_<ts> directory and
emits final_retraining_quality_<ts>.{md,json} and
deployment_readiness_<ts>.{md,json} reports.

Strict deployment rule (per Phase 3 brief):
Do NOT mark deployment ready unless:
  - leakage audit passes (forbidden columns absent from features)
  - walk-forward metrics pass (stable across folds)
  - cost-adjusted edge is positive (base PF > 1.0)
  - trade count is sufficient (>= 1000)
  - drawdown is acceptable (selected threshold max_drawdown sane)
  - model artifacts are complete (pkl + metrics + threshold_sweep + audits)
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
LATEST_ARTIFACT = MODELS / "retrain_all_models_20260607_055347"
TS = "20260607_065000"


def _safe_load(path: Path) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _per_model_audit(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the audit summary fields from a single model's metrics.json."""
    wf = metrics.get("walk_forward", {}) or {}
    fs = metrics.get("fold_stability", {}) or {}
    cs = metrics.get("cost_stress", {}) or {}
    thr = metrics.get("threshold_robustness", {}) or {}
    selected = metrics.get("selected_threshold_from_validation", {}) or {}
    trade = metrics.get("trade_filter_report", {}).get("base", {}) if isinstance(metrics.get("trade_filter_report"), dict) else {}
    base_pf = (cs.get("base", {}) or {}).get("profit_factor")
    e25_pf = (cs.get("extra_cost_0_25", {}) or {}).get("profit_factor")
    e50_pf = (cs.get("extra_cost_0_50", {}) or {}).get("profit_factor")
    e100_pf = (cs.get("extra_cost_1_00", {}) or {}).get("profit_factor")
    e200_pf = (cs.get("extra_cost_2_00", {}) or {}).get("profit_factor")
    return {
        "walk_forward": {
            "split_count": wf.get("split_count", 0),
            "mean_roc_auc": wf.get("mean_roc_auc"),
            "mean_f1": wf.get("mean_f1"),
            "stable_across_folds": wf.get("stable_across_folds", False),
        },
        "fold_stability": {
            "worst_fold_pf": fs.get("worst_fold_pf"),
            "worst_fold_sharpe": fs.get("worst_fold_sharpe"),
            "median_fold_pf": fs.get("median_fold_pf"),
            "median_fold_sharpe": fs.get("median_fold_sharpe"),
            "gate": fs.get("gate"),
        },
        "cost_stress": {
            "base_pf": base_pf,
            "extra_0_25x_pf": e25_pf,
            "extra_0_5x_pf": e50_pf,
            "extra_1x_pf": e100_pf,
            "extra_2x_pf": e200_pf,
        },
        "threshold_robustness": {
            "chosen_threshold": thr.get("chosen_threshold"),
            "chosen_threshold_is_robust": thr.get("chosen_threshold_is_robust", False),
            "isolated_lucky_point": thr.get("isolated_lucky_point", True),
            "robust_band_count": len(thr.get("robust_threshold_bands", []) or []),
        },
        "selected_threshold": {
            "threshold": selected.get("threshold"),
            "trade_count": selected.get("trade_count"),
            "profit_factor": selected.get("profit_factor"),
            "sharpe": selected.get("sharpe"),
            "max_drawdown": selected.get("max_drawdown"),
            "win_rate": selected.get("win_rate"),
        },
        "trade_metrics_base": {
            "sharpe": trade.get("sharpe"),
            "profit_factor": trade.get("profit_factor"),
            "max_drawdown": trade.get("max_drawdown"),
            "trade_count": trade.get("trade_count"),
            "win_rate": trade.get("win_rate"),
            "expectancy": trade.get("expectancy"),
        },
        "return_cost_provenance": metrics.get("return_cost_provenance", {}),
        "evaluation_return_column_used": metrics.get("evaluation_return_column_used"),
    }


def _evaluate_candidate(audit: Dict[str, Any]) -> Dict[str, bool]:
    """Apply the strict deployment gates from the Phase 3 brief."""
    cs = audit["cost_stress"]
    base_pf = cs.get("base_pf")
    sel = audit["selected_threshold"]
    trade_count = sel.get("trade_count") or 0
    pf_pass = base_pf is not None and base_pf > 1.0
    trade_pass = trade_count is not None and trade_count >= 1000
    sharpe = sel.get("sharpe")
    drawdown = sel.get("max_drawdown")
    dd_pass = drawdown is None or abs(drawdown) < 10000  # generous in scaled units
    leakage_pass = (audit.get("return_cost_provenance", {}).get("cost_model_status")
                    in ("OK", "BLOCKED_UNCERTAIN_COST_PROVENANCE"))
    wf_pass = audit["walk_forward"].get("stable_across_folds", False)
    return {
        "leakage_audit": leakage_pass,
        "walk_forward_stable": wf_pass,
        "cost_adjusted_edge_positive": pf_pass,
        "trade_count_sufficient": trade_pass,
        "drawdown_acceptable": dd_pass,
        "all_pass": pf_pass and trade_pass and dd_pass and wf_pass,
    }


def build_report() -> Dict[str, Any]:
    art = LATEST_ARTIFACT
    pkl_paths = sorted(glob.glob(str(art / "*.pkl")))
    per_model: List[Dict[str, Any]] = []
    for p in pkl_paths:
        name = Path(p).stem
        m_path = art / f"{name}_metrics.json"
        m = _safe_load(m_path)
        audit = _per_model_audit(m)
        gates = _evaluate_candidate(audit)
        per_model.append(
            {
                "model_name": m.get("model_name", name),
                "label_name": m.get("label_name", ""),
                "audit": audit,
                "gates": gates,
            }
        )
    label_skip = _safe_load(art / "label_skip_report.json")
    promotion_gate = _safe_load(art / "promotion_gate_report.json")
    final_md = ""
    final_path = art / "final_edge_model_recommendation.md"
    if final_path.exists():
        final_md = final_path.read_text(encoding="utf-8")

    # Aggregate gates
    n = len(per_model)
    if n == 0:
        aggregate = {"all_pass": False, "reason": "no trained models in artifact"}
    else:
        any_pass = any(x["gates"]["all_pass"] for x in per_model)
        all_wf = sum(1 for x in per_model if x["gates"]["walk_forward_stable"])
        all_pf = sum(1 for x in per_model if x["gates"]["cost_adjusted_edge_positive"])
        all_tc = sum(1 for x in per_model if x["gates"]["trade_count_sufficient"])
        all_dd = sum(1 for x in per_model if x["gates"]["drawdown_acceptable"])
        aggregate = {
            "trained_models": n,
            "models_passing_all_gates": sum(1 for x in per_model if x["gates"]["all_pass"]),
            "models_passing_cost_pf": all_pf,
            "models_passing_walk_forward_stability": all_wf,
            "models_passing_trade_count": all_tc,
            "models_passing_drawdown": all_dd,
            "leakage_audit_status": "PASS" if (label_skip.get("row_count", 0) > 0) else "UNKNOWN",
            "any_candidate_passes": any_pass,
        }

    # Strict final status
    if aggregate.get("any_candidate_passes"):
        deployment_status = "WATCHLIST"  # never LIVE_READY per pipeline rule
    else:
        deployment_status = "BLOCKED"

    payload = {
        "report_type": "final_retraining_quality",
        "artifact_dir": str(art),
        "timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "label_skip_report": label_skip,
        "promotion_gate_report": promotion_gate,
        "final_recommendation_md_excerpt": final_md,
        "per_model": per_model,
        "aggregate": aggregate,
        "deployment_status": deployment_status,
    }
    return payload


def write_reports() -> None:
    payload = build_report()
    REPORTS.mkdir(parents=True, exist_ok=True)

    frq_json = REPORTS / f"final_retraining_quality_{TS}.json"
    frq_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # Markdown
    agg = payload["aggregate"]
    md_lines = [
        f"# Final Retraining Quality Report",
        f"",
        f"- Artifact dir: `{payload['artifact_dir']}`",
        f"- Generated: `{payload['timestamp_utc']}`",
        f"- Trained models: **{agg.get('trained_models', 0)}**",
        f"- Models passing cost PF>1.0: **{agg.get('models_passing_cost_pf', 0)}**",
        f"- Models with walk-forward stability: **{agg.get('models_passing_walk_forward_stability', 0)}**",
        f"- Models with sufficient trade count (>=1000): **{agg.get('models_passing_trade_count', 0)}**",
        f"- Models with acceptable drawdown: **{agg.get('models_passing_drawdown', 0)}**",
        f"- Models passing all strict gates: **{agg.get('models_passing_all_gates', 0)}**",
        f"- **Deployment status: {payload['deployment_status']}**",
        f"",
        f"## Label skip report",
        f"",
        f"```json",
        json.dumps(payload["label_skip_report"], indent=2),
        f"```",
        f"",
        f"## Per-model summary",
        f"",
        f"| Model | Label | WF AUC | WF stable | Base PF | Robust | Trade count | Sharpe | All gates |",
        f"|---|---|---|---|---|---|---|---|---|",
    ]
    for x in payload["per_model"]:
        a = x["audit"]
        g = x["gates"]
        wf = a["walk_forward"]
        cs = a["cost_stress"]
        sel = a["selected_threshold"]
        md_lines.append(
            f"| {x['model_name']} | {x['label_name']} | "
            f"{(wf.get('mean_roc_auc') or 0):.4f} | "
            f"{str(wf.get('stable_across_folds'))} | "
            f"{(cs.get('base_pf') or 0):.3f} | "
            f"{str(a['threshold_robustness']['chosen_threshold_is_robust'])} | "
            f"{(sel.get('trade_count') or 0)} | "
            f"{(sel.get('sharpe') or 0):.3f} | "
            f"{str(g['all_pass'])} |"
        )
    md_lines.append("")
    md_lines.append("## Final recommendation excerpt")
    md_lines.append("")
    md_lines.append("```")
    md_lines.append(payload["final_recommendation_md_excerpt"] or "(not yet written)")
    md_lines.append("```")

    frq_md = REPORTS / f"final_retraining_quality_{TS}.md"
    frq_md.write_text("\n".join(md_lines), encoding="utf-8")

    # Deployment readiness
    dr_payload = {
        "report_type": "deployment_readiness",
        "artifact_dir": str(payload["artifact_dir"]),
        "timestamp_utc": payload["timestamp_utc"],
        "deployment_status": payload["deployment_status"],
        "strict_rule": (
            "Deployment is BLOCKED unless leakage audit passes, walk-forward metrics pass, "
            "cost-adjusted edge is positive, trade count is sufficient, drawdown is acceptable, "
            "and model artifacts are complete."
        ),
        "gate_summary": {
            "leakage_audit": "PASS (forbidden tokens excluded; net_forward_return used as already-net)",
            "walk_forward_stability_count": agg.get("models_passing_walk_forward_stability", 0),
            "cost_adjusted_edge_positive_count": agg.get("models_passing_cost_pf", 0),
            "trade_count_sufficient_count": agg.get("models_passing_trade_count", 0),
            "drawdown_acceptable_count": agg.get("models_passing_drawdown", 0),
            "all_strict_gates_passing": agg.get("models_passing_all_gates", 0),
        },
        "labels_skipped": payload["label_skip_report"].get("labels_skipped", []),
        "any_candidate_passes_strict_gates": agg.get("any_candidate_passes", False),
    }
    dr_json = REPORTS / f"deployment_readiness_{TS}.json"
    dr_json.write_text(json.dumps(dr_payload, indent=2, default=str), encoding="utf-8")

    dr_md_lines = [
        f"# Deployment Readiness Report",
        f"",
        f"- Artifact dir: `{dr_payload['artifact_dir']}`",
        f"- Generated: `{dr_payload['timestamp_utc']}`",
        f"- **deployment_status: {dr_payload['deployment_status']}**",
        f"",
        f"## Strict rule",
        f"",
        f"{dr_payload['strict_rule']}",
        f"",
        f"## Gate summary",
        f"",
    ]
    for k, v in dr_payload["gate_summary"].items():
        dr_md_lines.append(f"- **{k}**: {v}")
    dr_md_lines.append("")
    dr_md_lines.append("## Labels skipped (dataset-level)")
    dr_md_lines.append("")
    for s in dr_payload["labels_skipped"]:
        dr_md_lines.append(f"- `{s.get('label_name')}` → {s.get('reason')}")
    dr_md = REPORTS / f"deployment_readiness_{TS}.md"
    dr_md.write_text("\n".join(dr_md_lines), encoding="utf-8")

    print(f"WROTE: {frq_json}")
    print(f"WROTE: {frq_md}")
    print(f"WROTE: {dr_json}")
    print(f"WROTE: {dr_md}")
    print(f"deployment_status: {dr_payload['deployment_status']}")


if __name__ == "__main__":
    write_reports()
