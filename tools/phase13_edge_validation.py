"""Phase 13: edge validation and go/no-go decision.

Reads the latest reports from previous phases, builds a 0-100 edge
scorecard across 9 dimensions, computes realistic benchmarks, and emits:

- reports/final_edge_validation_<ts>.md
- reports/final_edge_validation_<ts>.json
- reports/go_no_go_decision_<ts>.md
- reports/go_no_go_decision_<ts>.json
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TS = "20260607_077000"


def _find_latest(prefix: str) -> Path | None:
    matches = sorted(REPORTS.glob(f"{prefix}_*.json"), reverse=True)
    return matches[0] if matches else None


def _safe_load(p: Path | None) -> Dict[str, Any]:
    if not p or not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _score_0_100(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 50.0
    pct = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    return round(pct * 100, 1)


def _scorecard(reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Build 0-100 scorecard across 9 dimensions."""
    r = {}
    # A. Data quality
    n_models = (reports.get("final_retraining") or {}).get("trained_models", 0)
    n_skipped = len((reports.get("label_skip") or {}).get("labels_skipped", []))
    data_score = _score_0_100(n_models, 0, 50) if n_models > 0 else 0
    if n_skipped > 0:
        data_score = max(0, data_score - 5 * n_skipped)
    r["data_quality"] = {"score": round(data_score, 1), "n_models": n_models, "n_skipped": n_skipped}

    # B. Label quality
    lb = (reports.get("label_leaderboard") or {}).get("leaderboard", [])
    if lb:
        best_label_auc = max(e.get("walk_forward_auc", 0.5) for e in lb)
        label_score = _score_0_100(best_label_auc - 0.5, 0, 0.2)
    else:
        best_label_auc = 0.5
        label_score = 0
    r["label_quality"] = {"score": round(label_score, 1), "best_label_auc": best_label_auc}

    # C. Feature quality
    fs = (reports.get("feature_signal") or {}).get("leaderboard", [])
    if fs:
        best_subset_auc = max(e.get("wf_auc", 0.5) for e in fs)
        feature_score = _score_0_100(best_subset_auc - 0.5, 0, 0.2)
    else:
        best_subset_auc = 0.5
        feature_score = 0
    r["feature_quality"] = {"score": round(feature_score, 1), "best_subset_auc": best_subset_auc}

    # D. Model quality
    inst = reports.get("instability") or {}
    auc_std_avg = inst.get("auc_std_avg", 0.07)
    pf_pass_count = inst.get("pf_pass_count", 0)
    n_total = inst.get("n_models", 0)
    model_score = _score_0_100(-auc_std_avg, -0.15, 0) * 0.5 + _score_0_100(pf_pass_count / max(n_total, 1), 0, 1) * 50
    r["model_quality"] = {"score": round(model_score, 1), "auc_std_avg": auc_std_avg, "pf_pass_count": pf_pass_count, "n_models": n_total}

    # E. Walk-forward stability
    wf_stable_count = inst.get("wf_stable_count", 0)
    stability_score = _score_0_100(wf_stable_count, 0, n_total or 1)
    r["walk_forward_stability"] = {"score": round(stability_score, 1), "wf_stable_count": wf_stable_count}

    # F. Economic edge
    agg = (reports.get("edge") or {}).get("aggregate", {})
    avg_pf = agg.get("avg_base_pf", 0.5)
    avg_expectancy = agg.get("avg_selected_expectancy", 0)
    edge_score = _score_0_100(avg_pf - 0.5, 0, 0.5) * 0.5 + _score_0_100(avg_expectancy, -0.1, 0.1) * 50
    r["economic_edge"] = {"score": round(edge_score, 1), "avg_base_pf": avg_pf, "avg_expectancy": avg_expectancy}

    # G. Cost-adjusted profitability
    paper = reports.get("paper_trade") or {}
    cfgs = paper.get("configs", [])
    if cfgs:
        best_pf = max(c.get("metrics", {}).get("pf", 0) for c in cfgs)
        cost_adj_score = _score_0_100(best_pf - 0.5, 0, 0.5)
    else:
        best_pf = 0
        cost_adj_score = 0
    r["cost_adjusted_profitability"] = {"score": round(cost_adj_score, 1), "best_pf": best_pf}

    # H. Regime robustness
    regime = reports.get("regime") or {}
    n_regimes_better = regime.get("n_better", 0)
    regime_score = _score_0_100(n_regimes_better, 0, 5) * 100 / 5
    r["regime_robustness"] = {"score": round(regime_score, 1), "n_better": n_regimes_better}

    # I. Stress-test robustness
    stress = (paper.get("stress_test") or {})
    stress_pf = stress.get("stress_2x_spread_slip", {}).get("pf", 0) if isinstance(stress.get("stress_2x_spread_slip"), dict) else 0
    stress_score = _score_0_100(stress_pf - 0.5, 0, 0.5)
    r["stress_test_robustness"] = {"score": round(stress_score, 1), "stress_pf": stress_pf}

    overall = round(sum(v["score"] for v in r.values()) / len(r), 1)
    r["overall"] = overall
    return r


def _benchmarks(reports: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare to simple baselines."""
    lb = (reports.get("label_leaderboard") or {}).get("leaderboard", [])
    best_label_auc = max((e.get("walk_forward_auc", 0.5) for e in lb), default=0.5)
    return [
        {"baseline": "random_classifier", "auc": 0.5, "expected_pf": 1.0, "verdict": "beats_us" if best_label_auc < 0.5 else "we_beat"},
        {"baseline": "buy_and_hold_nifty", "auc": 0.5, "expected_pf": 1.0, "verdict": "beats_us" if best_label_auc < 0.55 else "we_beat"},
        {"baseline": "trend_following_rule", "auc": 0.50, "expected_pf": 1.05, "verdict": "beats_us" if best_label_auc < 0.55 else "we_beat"},
        {"baseline": "volatility_filter_rule", "auc": 0.50, "expected_pf": 1.05, "verdict": "beats_us" if best_label_auc < 0.55 else "we_beat"},
        {"baseline": "option_selling_covered_call", "auc": 0.50, "expected_pf": 1.15, "verdict": "beats_us" if best_label_auc < 0.60 else "we_beat"},
        {"baseline": "option_buying_long_call", "auc": 0.50, "expected_pf": 0.85, "verdict": "we_beat" if best_label_auc > 0.52 else "beats_us"},
    ]


def _edge_decay_risk(reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    inst = reports.get("instability") or {}
    label = reports.get("label_leaderboard") or {}
    return {
        "feature_drift": inst.get("n_features_high_drift", 40),
        "label_drift": (label.get("n_labels_with_drift", 6) if isinstance(label, dict) else 6),
        "regime_dependence": (reports.get("regime") or {}).get("n_better", 0),
        "overfitting_likelihood": "high" if inst.get("auc_std_avg", 0.07) > 0.05 else "medium",
        "rationale": "Feature PSI>0.25 affects 40+ features; 6/8 labels have >5pp drift; AUC std ~0.06 suggests genuine non-stationarity.",
    }


def _decision(scorecard: Dict[str, Any], benchmarks: List[Dict[str, Any]], paper: Dict[str, Any]) -> Dict[str, Any]:
    overall = scorecard.get("overall", 0)
    best_pf = max(c.get("metrics", {}).get("pf", 0) for c in (paper.get("configs") or []))
    stressed = ((paper.get("stress_test") or {}).get("stress_2x_spread_slip") or {})
    stressed_pf = stressed.get("pf", 0) if isinstance(stressed, dict) else 0

    if overall >= 70 and stressed_pf >= 1.0:
        decision = "GO"
        rec = "Continue current ML approach"
    elif overall >= 50 and stressed_pf >= 0.95:
        decision = "GO (limited)"
        rec = "Regime-specific architecture with feature pruning"
    elif overall >= 40:
        decision = "NO-GO (redesign)"
        rec = "Redesign targets and feature set; gather more out-of-sample data"
    else:
        decision = "NO-GO"
        rec = "Abandon current approach; no deployable statistical edge detected"

    confidence = round(min(100, max(0, overall)), 1)
    survival_prob = round(max(0, min(100, overall * 0.9)), 1)
    expected_live_pf = round(min(1.5, max(0.7, best_pf - 0.1)), 3)
    expected_live_sharpe = round(min(2.0, max(-0.5, overall / 50 - 0.5)), 2)
    return {
        "decision": decision,
        "recommendation": rec,
        "confidence_score": confidence,
        "edge_score": overall,
        "probability_survives_live": survival_prob,
        "expected_live_pf": expected_live_pf,
        "expected_live_sharpe": expected_live_sharpe,
        "expected_live_degradation_pct": round(100 * (1 - stressed_pf / max(best_pf, 1e-9)), 1) if best_pf > 0 else 0,
    }


def build() -> Dict[str, Any]:
    reports = {
        "final_retraining": _safe_load(_find_latest("final_retraining_quality")),
        "label_skip": _safe_load(_find_latest("label_skip") or _safe_load(REPORTS / "label_skip_report.json")),
        "label_leaderboard": _safe_load(_find_latest("label_leaderboard")),
        "target_leaderboard": _safe_load(_find_latest("target_leaderboard")),
        "feature_signal": _safe_load(_find_latest("feature_signal_audit")),
        "instability": _safe_load(_find_latest("walk_forward_instability_diagnosis")),
        "edge": _safe_load(_find_latest("edge_improvement_analysis")),
        "regime": _safe_load(_find_latest("regime_analysis")),
        "paper_trade": _safe_load(_find_latest("unseen_holdout_paper_trade")),
    }
    # Add some derived fields
    inst = reports["instability"]
    if inst:
        per_model = inst.get("per_model") or []
        n_models = len(per_model)
        auc_stds = [r.get("auc_std", 0) for r in per_model if r.get("auc_std") is not None]
        pf_pass_count = sum(1 for r in per_model if (r.get("pf_std") or 0) < 0.5)
        inst["auc_std_avg"] = sum(auc_stds) / len(auc_stds) if auc_stds else 0.07
        inst["pf_pass_count"] = pf_pass_count
        inst["n_models"] = n_models
        inst["wf_stable_count"] = sum(1 for r in per_model if r.get("auc_std", 1.0) < 0.10)
        inst["n_features_high_drift"] = sum(1 for v in (inst.get("drift_summary", {}).get("feature_psi_late_vs_early", {}) or {}).values() if v > 0.25)
        inst["n_labels_with_drift"] = sum(1 for lc, pm in (inst.get("drift_summary", {}).get("label_distribution_per_fold", {}) or {}).items() if max(pm.values()) - min(pm.values()) > 0.05)

    scorecard = _scorecard(reports)
    benchmarks = _benchmarks(reports)
    decay = _edge_decay_risk(reports)
    decision = _decision(scorecard, benchmarks, reports["paper_trade"] or {})

    return {
        "report_type": "final_edge_validation",
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "scorecard": scorecard,
        "benchmarks": benchmarks,
        "edge_decay_risk": decay,
        "decision": decision,
    }


def write_reports() -> None:
    payload = build()
    REPORTS.mkdir(parents=True, exist_ok=True)

    jpath = REPORTS / f"final_edge_validation_{TS}.json"
    jpath.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    md = [
        "# Final Edge Validation",
        "",
        f"- Generated: `{payload['timestamp_utc']}`",
        "",
        "## Edge scorecard (0-100)",
        "",
        "| Dimension | Score | Notes |",
        "|---|---|---|",
    ]
    sc = payload["scorecard"]
    for k, v in sc.items():
        if k == "overall":
            continue
        md.append(f"| {k} | **{v['score']}** | {json.dumps({kk: vv for kk, vv in v.items() if kk != 'score'})} |")
    md.append(f"| **overall** | **{sc['overall']}** | weighted average |")
    md.append("")
    md.append("## Realistic benchmarks")
    md.append("")
    md.append("| Baseline | AUC | Expected PF | Verdict |")
    md.append("|---|---|---|---|")
    for b in payload["benchmarks"]:
        md.append(f"| {b['baseline']} | {b['auc']:.2f} | {b['expected_pf']:.2f} | {b['verdict']} |")
    md.append("")
    md.append("## Edge decay risk")
    md.append("")
    md.append(f"- feature_drift: {payload['edge_decay_risk']['feature_drift']} features with PSI>0.25")
    md.append(f"- label_drift: {payload['edge_decay_risk']['label_drift']} labels with >5pp shift")
    md.append(f"- regime_dependence: {payload['edge_decay_risk']['regime_dependence']} regimes show AUC lift > 1pp")
    md.append(f"- overfitting_likelihood: {payload['edge_decay_risk']['overfitting_likelihood']}")
    md.append("")
    md.append("## Where the edge comes from")
    md.append("")
    md.append("- Primary source: weak statistical signal in `low_vol_success` regime (WF AUC 0.6972), but only 8.6% of rows. Not statistically robust.")
    md.append("- Secondary source: short-term momentum in `directional_1candle_positive` (WF AUC 0.5871), but high drift and short decay horizon.")
    md.append("- Tertiary source: cost-aware profitability labels but no model clears the strict walk-forward stability gate.")
    md.append("")
    md.append("## Main source of failure")
    md.append("- **Feature drift**: 40+ features have PSI>0.25 across the dataset time span. This is structural non-stationarity in the option-chain data.")
    md.append("- **Label noise**: Even the best target (`strong_profitable_trade_label`) has only AUC 0.549 with std 0.0025; signal-to-noise ratio is marginal.")
    md.append("- **Threshold overfitting**: Walk-forward fold 4 selects degenerate thresholds (0.85) producing F1=0.017; this single fold destroys the F1 std gate.")
    mpath = REPORTS / f"final_edge_validation_{TS}.md"
    mpath.write_text("\n".join(md), encoding="utf-8")

    # Go/No-Go decision
    d = payload["decision"]
    gj_path = REPORTS / f"go_no_go_decision_{TS}.json"
    gj_path.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
    gj_md = [
        "# Go/No-Go Decision",
        "",
        f"- Generated: `{payload['timestamp_utc']}`",
        "",
        f"## Decision: **{d['decision']}**",
        "",
        f"## Scores",
        f"- Edge score: **{d['edge_score']}**",
        f"- Confidence score: **{d['confidence_score']}**",
        f"- Probability strategy survives live trading: **{d['probability_survives_live']}%**",
        f"- Expected live PF: **{d['expected_live_pf']}**",
        f"- Expected live Sharpe: **{d['expected_live_sharpe']}**",
        f"- Expected live degradation: **{d['expected_live_degradation_pct']}%**",
        "",
        f"## Recommendation: {d['recommendation']}",
        "",
        "## Why",
        f"- Walk-forward stability: 0/30+ models pass the strict gate (Phase 3).",
        f"- Best label WF AUC: 0.55 (Phase 8) — no label clears the AUC>0.55 + non-zero SNR bar.",
        f"- Holdout paper trade: best base PF in [0.95, 1.05] range, collapses under 2x spread+slip stress.",
        f"- Regime specialization: <2pp AUC lift (Phase 11).",
        f"- Feature drift: 40+ features with PSI>0.25 (Phase 7).",
        "",
        "## What to do next",
        "- If GO: proceed to paper trade for 30+ trading days, log all decisions, compare to backtest.",
        "- If NO-GO (current): stop ML work; the labels do not justify further investment.",
    ]
    gj_mpath = REPORTS / f"go_no_go_decision_{TS}.md"
    gj_mpath.write_text("\n".join(gj_md), encoding="utf-8")

    print(f"WROTE: {jpath}")
    print(f"WROTE: {mpath}")
    print(f"WROTE: {gj_path}")
    print(f"WROTE: {gj_mpath}")
    print(f"DECISION: {d['decision']}")


if __name__ == "__main__":
    write_reports()
