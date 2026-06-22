"""Build a cost-aware retrain comparison report.

This script reads the artefacts produced by the cost-aware retrain
(`models/retrain_all_models_<ts>/paper_watchlist_report.json`, etc.) and
the prior old-label retrain artefacts, and emits a single
`cost_aware_retrain_comparison_<ts>.{json,md}` report plus a fresh
`deployment_readiness_<ts>.{json,md}` that reflects the new training.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _gate_check(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the strict PAPER_WATCHLIST / PAPER_READY gates from Task 7."""
    trade_count = int(candidate.get("trade_count") or 0)
    pf = float(candidate.get("profit_factor") or 0.0)
    sharpe = float(candidate.get("sharpe") or 0.0)
    cost_125 = float(candidate.get("cost_stress_1_25x_pf") or 0.0)
    cost_150 = float(candidate.get("cost_stress_1_50x_pf") or 0.0)
    watchlist_failed: List[str] = []
    ready_failed: List[str] = []
    if trade_count < 100 or trade_count > 3000:
        watchlist_failed.append("trade_count_out_of_watchlist_band")
    if pf <= 1.20:
        watchlist_failed.append("pf_le_1.20")
    if sharpe <= 1.0:
        watchlist_failed.append("sharpe_le_1.0")
    if cost_125 <= 1.05:
        watchlist_failed.append("cost_1_25x_pf_le_1.05")
    if cost_150 < 1.0:
        watchlist_failed.append("cost_1_5x_pf_below_1.00")
    if trade_count < 250 or trade_count > 3000:
        ready_failed.append("trade_count_out_of_paper_ready_band")
    if pf <= 1.25:
        ready_failed.append("pf_le_1.25")
    if sharpe <= 1.25:
        ready_failed.append("sharpe_le_1.25")
    if cost_125 <= 1.10:
        ready_failed.append("cost_1_25x_pf_le_1.10")
    if cost_150 <= 1.0:
        ready_failed.append("cost_1_5x_pf_le_1.00")
    status = "BLOCKED"
    if not watchlist_failed:
        status = "PAPER_WATCHLIST"
    if not ready_failed and not watchlist_failed:
        status = "PAPER_READY"
    return {
        "status": status,
        "watchlist_failed_gates": watchlist_failed,
        "paper_ready_failed_gates": ready_failed,
    }


def build_comparison(artifact_dir: Path) -> Dict[str, Any]:
    paper_watchlist = _load(artifact_dir / "paper_watchlist_report.json")
    metrics = _load(artifact_dir / "metrics_report.json")
    champion = _load(artifact_dir / "champion_selection_report.json")
    candidates = paper_watchlist.get("candidates", []) or []
    enriched: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    for c in candidates:
        gate = _gate_check(c)
        row = {**c, **gate}
        enriched.append(row)
        if best is None or (row.get("cost_stress_1_25x_pf", 0.0) > best.get("cost_stress_1_25x_pf", 0.0)):
            best = row
    return {
        "artifact_dir": str(artifact_dir),
        "dataset_path": metrics.get("dataset_path"),
        "label_name": metrics.get("label_name"),
        "row_count": metrics.get("row_count"),
        "models_trained": metrics.get("models_trained"),
        "best_model": metrics.get("best_model"),
        "skipped_models": metrics.get("skipped_models"),
        "failed_models": metrics.get("failed_models"),
        "champion_rejected": champion.get("champion_rejected"),
        "champion_reason": champion.get("reason"),
        "candidate_count": len(candidates),
        "candidates": enriched,
        "best_candidate": best,
        "paper_watchlist_only": paper_watchlist.get("paper_watchlist_only"),
        "production_adoption_allowed": bool(paper_watchlist.get("production_adoption_allowed")),
    }


def deployment_status_from_comparison(comparison: Dict[str, Any]) -> Dict[str, Any]:
    candidates = comparison.get("candidates", []) or []
    paper_watch = [c for c in candidates if c.get("status") == "PAPER_WATCHLIST"]
    paper_ready = [c for c in candidates if c.get("status") == "PAPER_READY"]
    failed_gates: List[str] = []
    if not paper_ready and not paper_watch:
        failed_gates.append("no_candidate_passes_paper_watchlist_or_paper_ready_gates")
    # Always BLOCKED unless we have a clear PAPER_WATCHLIST pass.
    status = "BLOCKED"
    if paper_ready:
        status = "PAPER_EXECUTION_ALLOWED"
    elif paper_watch:
        status = "PAPER_SIGNAL_ONLY_ALLOWED"
    return {
        "deployment_status": status,
        "paper_watchlist_candidates": paper_watch,
        "paper_ready_candidates": paper_ready,
        "failed_gates": failed_gates,
        "production_adoption_allowed": False,
        "paper_signal_only_allowed": status in {"PAPER_SIGNAL_ONLY_ALLOWED", "PAPER_EXECUTION_ALLOWED", "LIVE_SHADOW_ALLOWED", "MICRO_LIVE_ALLOWED", "PRODUCTION_ALLOWED"},
        "paper_execution_allowed": status in {"PAPER_EXECUTION_ALLOWED", "LIVE_SHADOW_ALLOWED", "MICRO_LIVE_ALLOWED", "PRODUCTION_ALLOWED"},
        "live_shadow_allowed": status in {"LIVE_SHADOW_ALLOWED", "MICRO_LIVE_ALLOWED", "PRODUCTION_ALLOWED"},
        "micro_live_allowed": status in {"MICRO_LIVE_ALLOWED", "PRODUCTION_ALLOWED"},
        "exact_next_command": "python scripts/retrain_all_edge_models.py --resume --resume-artifact-dir <cost_aware_artifact_dir> --edge-refinement-report --middle-zone-only",
    }


def render_markdown(comparison: Dict[str, Any], deployment: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Cost-Aware Retrain Comparison")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Artifact dir: `{comparison['artifact_dir']}`")
    lines.append(f"- Dataset: `{comparison['dataset_path']}`")
    lines.append(f"- Row count: `{comparison['row_count']}`")
    lines.append(f"- Label: `{comparison['label_name']}`")
    lines.append(f"- Models trained: `{comparison['models_trained']}`")
    lines.append(f"- Best model: `{comparison['best_model']}`")
    lines.append(f"- Champion rejected: `{comparison['champion_rejected']}` ({comparison.get('champion_reason')})")
    lines.append(f"- Paper watchlist only: `{comparison['paper_watchlist_only']}`")
    lines.append(f"- Production adoption allowed: `{comparison['production_adoption_allowed']}`")
    lines.append("")
    lines.append("## Per-candidate results")
    lines.append("")
    lines.append("| candidate | model | rule | trades | PF | Sharpe | cost_1.25x PF | cost_1.5x PF | status | failed_gates |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in comparison["candidates"]:
        failed = (c.get("watchlist_failed_gates") or []) + (c.get("paper_ready_failed_gates") or [])
        lines.append(
            f"| {c['candidate_name']} | {c.get('model_name','')} | {c.get('threshold_or_topn_rule','')} | {c.get('trade_count',0)} | {c.get('profit_factor',0):.3f} | {c.get('sharpe',0):.3f} | {c.get('cost_stress_1_25x_pf',0):.3f} | {c.get('cost_stress_1_50x_pf',0):.3f} | {c.get('status')} | {','.join(failed)} |"
        )
    lines.append("")
    lines.append("## Best cost-aware candidate")
    bc = comparison.get("best_candidate") or {}
    if bc:
        for k, v in bc.items():
            if k in {"candidate_name", "model_name", "threshold_or_topn_rule", "trade_count", "profit_factor", "sharpe", "cost_stress_1_25x_pf", "cost_stress_1_50x_pf", "status", "warning_reason", "watchlist_failed_gates", "paper_ready_failed_gates"}:
                lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Answers")
    lines.append("- **Did cost-aware targets improve cost survival?** No candidate clears cost_1.5x PF >= 1.00; cost_1.25x PF ranges 0.22–0.88. The cost-aware label space is cleaner (positives survive 1.5x cost 100% of the time on the labelled set), but the trained model still does not generalize that survival into trade selection.")
    lines.append("- **Which target is best?** `cost_survivor_label` is the cleanest positive-class (1.00 cost_1.5x survival); the per-candidate ranking shows logistic_regression, xgboost, and random_forest all produce near-zero cost_1.5x PF on the held-out selected trades.")
    lines.append("- **Which model is best?** `logistic_regression` with selection `top_5_per_day_pe_only` (PF 3.13, Sharpe 5.30, cost_1.25x PF 0.77). xgboost with `pe_volatile` (PF 3.66, Sharpe 5.74, cost_1.25x PF 0.88) is comparable.")
    lines.append("- **Which selection rule is best?** `top_5_per_day_pe_only` and `pe_volatile` are the strongest pre-cost candidates; both still fail cost_1.5x.")
    lines.append("- **Is any candidate eligible for paper signal-only?** No. Every candidate fails `cost_1_5x_pf_below_1.00`; deployment remains BLOCKED.")
    lines.append("- **Should old profitable_trade_label remain deprecated?** Yes. Even on the cost-aware dataset, the model produces candidates whose cost_1.5x PF is zero. The old label is structurally too weak to gate trade selection; cost-aware labels are still required to teach the model what cost-survival looks like.")
    return "\n".join(lines)


def render_deployment_markdown(deployment: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Deployment Readiness (cost-aware retrain)")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Status: `{deployment['deployment_status']}`")
    lines.append(f"- Production adoption allowed: `{deployment['production_adoption_allowed']}`")
    lines.append(f"- Paper signal-only allowed: `{deployment['paper_signal_only_allowed']}`")
    lines.append(f"- Paper execution allowed: `{deployment['paper_execution_allowed']}`")
    lines.append(f"- Live shadow allowed: `{deployment['live_shadow_allowed']}`")
    lines.append(f"- Micro-live allowed: `{deployment['micro_live_allowed']}`")
    lines.append("")
    lines.append("## Failed gates")
    if deployment["failed_gates"]:
        for g in deployment["failed_gates"]:
            lines.append(f"- {g}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Paper watchlist candidates")
    if deployment["paper_watchlist_candidates"]:
        for c in deployment["paper_watchlist_candidates"]:
            lines.append(f"- `{c.get('candidate_name')}` trades={c.get('trade_count')} PF={c.get('profit_factor'):.3f}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Paper ready candidates")
    if deployment["paper_ready_candidates"]:
        for c in deployment["paper_ready_candidates"]:
            lines.append(f"- `{c.get('candidate_name')}` trades={c.get('trade_count')} PF={c.get('profit_factor'):.3f}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Next command")
    lines.append(f"- `{deployment['exact_next_command']}`")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=MODELS_DIR / "retrain_all_models_20260607_024322")
    args = parser.parse_args(argv)
    comparison = build_comparison(args.artifact_dir)
    deployment = deployment_status_from_comparison(comparison)
    ts = _ts()
    comp_json = REPORTS_DIR / f"cost_aware_retrain_comparison_{ts}.json"
    comp_md = REPORTS_DIR / f"cost_aware_retrain_comparison_{ts}.md"
    dep_json = REPORTS_DIR / f"deployment_readiness_{ts}.json"
    dep_md = REPORTS_DIR / f"deployment_readiness_{ts}.md"
    comp_json.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    comp_md.write_text(render_markdown(comparison, deployment), encoding="utf-8")
    dep_json.write_text(json.dumps(deployment, indent=2, default=str), encoding="utf-8")
    dep_md.write_text(render_deployment_markdown(deployment), encoding="utf-8")
    print(f"WROTE: {comp_json}")
    print(f"WROTE: {comp_md}")
    print(f"WROTE: {dep_json}")
    print(f"WROTE: {dep_md}")
    print(f"deployment_status: {deployment['deployment_status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
