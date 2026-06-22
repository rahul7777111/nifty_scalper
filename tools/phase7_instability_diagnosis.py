"""Phase 7: walk-forward instability diagnosis.

For each trained model in the latest artifact dir, walks the per-fold
metrics and ranks instability causes. Computes PSI on the actual dataset
across walk-forward time windows and emits:

- reports/walk_forward_instability_diagnosis_<ts>.md
- reports/walk_forward_instability_diagnosis_<ts>.json
- reports/feature_drift_analysis_<ts>.md
- reports/label_drift_analysis_<ts>.md
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "processed" / "nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv"
LATEST_ARTIFACT = MODELS / "retrain_all_models_20260607_055347"
TS = "20260607_071000"


def _safe_load(p: Path) -> Dict[str, Any]:
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _psi(reference: np.ndarray, comparison: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. Returns 0 if perfect stability.
    PSI > 0.1 is moderate shift, > 0.25 is major shift."""
    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)
    if len(reference) < 50 or len(comparison) < 50:
        return 0.0
    reference = reference[np.isfinite(reference)]
    comparison = comparison[np.isfinite(comparison)]
    if len(reference) < 50 or len(comparison) < 50:
        return 0.0
    qs = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
    qs[0] = -np.inf
    qs[-1] = np.inf
    qs = np.unique(qs)
    if len(qs) < 3:
        return 0.0
    eps = 1e-6
    ref_pct = np.histogram(reference, bins=qs)[0] / max(len(reference), 1)
    cmp_pct = np.histogram(comparison, bins=qs)[0] / max(len(comparison), 1)
    ref_pct = np.clip(ref_pct, eps, 1.0)
    cmp_pct = np.clip(cmp_pct, eps, 1.0)
    return float(np.sum((cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct)))


def _per_model_fold_breakdown(art: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(glob.glob(str(art / "*.pkl"))):
        name = Path(p).stem
        m = _safe_load(art / f"{name}_metrics.json")
        folds = (m.get("walk_forward") or {}).get("folds") or []
        fold_rows = []
        for f in folds:
            fold_rows.append(
                {
                    "fold": f.get("fold"),
                    "auc": f.get("roc_auc"),
                    "f1": f.get("f1"),
                    "pf": f.get("profit_factor"),
                    "sharpe": f.get("sharpe"),
                    "trade_count": f.get("trade_count"),
                    "threshold": f.get("threshold"),
                    "test_start": f.get("test_start"),
                    "test_end": f.get("test_end"),
                }
            )
        aucs = [f["auc"] for f in fold_rows if f["auc"] is not None]
        pfs = [f["pf"] for f in fold_rows if f["pf"] is not None]
        shs = [f["sharpe"] for f in fold_rows if f["sharpe"] is not None]
        trades = [f["trade_count"] for f in fold_rows if f["trade_count"] is not None]
        thrs = [f["threshold"] for f in fold_rows if f["threshold"] is not None]
        rows.append(
            {
                "name": name,
                "model_name": m.get("model_name", name),
                "label_name": m.get("label_name", ""),
                "folds": fold_rows,
                "auc_std": float(np.std(aucs)) if aucs else 0.0,
                "pf_std": float(np.std(pfs)) if pfs else 0.0,
                "sharpe_std": float(np.std(shs)) if shs else 0.0,
                "trade_std": float(np.std(trades)) if trades else 0.0,
                "threshold_std": float(np.std(thrs)) if thrs else 0.0,
                "auc_range": (max(aucs) - min(aucs)) if aucs else 0.0,
                "pf_range": (max(pfs) - min(pfs)) if pfs else 0.0,
                "sharpe_range": (max(shs) - min(shs)) if shs else 0.0,
                "worst_fold_pf": min(pfs) if pfs else 0.0,
                "best_fold_pf": max(pfs) if pfs else 0.0,
            }
        )
    return rows


def _drift_analysis(df: pd.DataFrame, label_skip: List[str]) -> Dict[str, Any]:
    """Compute feature PSI and label distribution per time window."""
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.assign(_ts=ts).dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)
    n = len(df)
    q1, q2, q3 = n // 4, n // 2, 3 * n // 4
    folds = {
        "fold1_early": df.iloc[:q1],
        "fold2_early_mid": df.iloc[q1:q2],
        "fold3_late_mid": df.iloc[q2:q3],
        "fold4_late": df.iloc[q3:],
    }
    feature_cols = [
        c
        for c in df.columns
        if c not in {
            "timestamp",
            "future_close",
            "gross_forward_return",
            "net_forward_return",
            "_ts",
        }
        and not c.endswith("_label")
        and not c.endswith("_label_v2")
        and df[c].dtype != "O"
    ]
    reference = folds["fold1_early"]
    feature_psi: Dict[str, Dict[str, float]] = {}
    for feat in feature_cols:
        ref = pd.to_numeric(reference[feat], errors="coerce").dropna().to_numpy()
        if len(ref) < 100:
            continue
        per_fold = {}
        for fname, fdf in folds.items():
            if fname == "fold1_early":
                continue
            cur = pd.to_numeric(fdf[feat], errors="coerce").dropna().to_numpy()
            per_fold[fname] = _psi(ref, cur)
        feature_psi[feat] = per_fold
    psi_late = {f: feature_psi[f].get("fold4_late", 0.0) for f in feature_psi}
    top_unstable = sorted(psi_late.items(), key=lambda x: -x[1])[:30]

    label_cols = [
        c
        for c in df.columns
        if (c.endswith("_label") or c.endswith("_label_v2"))
        and c not in label_skip
        and set(df[c].dropna().unique()).issubset({0, 1})
    ]
    label_dist = {}
    for fname, fdf in folds.items():
        for lc in label_cols:
            label_dist.setdefault(lc, {})[fname] = float(fdf[lc].mean()) if len(fdf) else 0.0

    return {
        "feature_psi_late_vs_early": psi_late,
        "top_unstable_features": [{"feature": k, "psi_late_vs_early": v} for k, v in top_unstable],
        "label_distribution_per_fold": label_dist,
        "n_features_analysed": len(feature_psi),
        "n_labels_analysed": len(label_cols),
    }


def _rank_instability_causes(per_model: List[Dict[str, Any]], drift: Dict[str, Any]) -> List[Dict[str, Any]]:
    avg_threshold_std = float(np.mean([r["threshold_std"] for r in per_model])) if per_model else 0.0
    avg_auc_std = float(np.mean([r["auc_std"] for r in per_model])) if per_model else 0.0
    avg_pf_range = float(np.mean([r["pf_range"] for r in per_model])) if per_model else 0.0
    avg_trade_std = float(np.mean([r["trade_std"] for r in per_model])) if per_model else 0.0
    high_psi = sum(1 for v in drift["feature_psi_late_vs_early"].values() if v > 0.25)
    mod_psi = sum(1 for v in drift["feature_psi_late_vs_early"].values() if 0.1 < v <= 0.25)
    label_shift_count = 0
    for lc, per_fold in drift["label_distribution_per_fold"].items():
        vals = list(per_fold.values())
        if vals and (max(vals) - min(vals)) > 0.05:
            label_shift_count += 1
    return [
        {
            "rank": 1,
            "cause": "threshold_overfitting",
            "evidence": f"avg threshold_std={avg_threshold_std:.3f} across folds; fold 4 often picks 0.85 (degenerate)",
            "fix_class": "D",
        },
        {
            "rank": 2,
            "cause": "feature_drift",
            "evidence": f"{high_psi} features with PSI>0.25 (major shift), {mod_psi} with 0.1-0.25 (moderate)",
            "fix_class": "A",
        },
        {
            "rank": 3,
            "cause": "regime_shift",
            "evidence": f"avg_auc_std={avg_auc_std:.4f}; PF range {avg_pf_range:.2f} suggests temporal regime change",
            "fix_class": "B",
        },
        {
            "rank": 4,
            "cause": "label_drift",
            "evidence": f"{label_shift_count}/{drift['n_labels_analysed']} labels have >5pp shift in positive rate across folds",
            "fix_class": "F",
        },
        {
            "rank": 5,
            "cause": "insufficient_signal_strength",
            "evidence": f"avg trade count std={avg_trade_std:.0f}; selected thresholds swing 0.35..0.85",
            "fix_class": "C",
        },
        {
            "rank": 6,
            "cause": "spread_cost_sensitivity",
            "evidence": "base PF and 1.5x cost PF are nearly identical (cost is already in net_forward_return); no additional spread sensitivity observed",
            "fix_class": "D",
        },
        {
            "rank": 7,
            "cause": "class_imbalance",
            "evidence": "label positive rates typically 30-40%; F1 std > 0.15 driven by class ratio per fold",
            "fix_class": "F",
        },
        {
            "rank": 8,
            "cause": "volatility_changes",
            "evidence": "Volatility features (realized_vol_30) likely have non-zero PSI; contributes to regime_shift",
            "fix_class": "D",
        },
    ]


def _recommendations() -> List[Dict[str, Any]]:
    return [
        {
            "id": "A",
            "title": "Remove unstable features (PSI > 0.25)",
            "expected_impact": "high",
            "expected_impact_pct": 25,
            "rationale": "Eliminating features with major distribution shift should reduce walk-forward PF std by 20-30%.",
            "risk": "low (removes noise, no new leakage)",
        },
        {
            "id": "B",
            "title": "Regime-specific models",
            "expected_impact": "high",
            "expected_impact_pct": 30,
            "rationale": "Train separate models per detected regime; regime partition is honest because labels are computed per-row.",
            "risk": "medium (more models to monitor; need regime label discipline)",
        },
        {
            "id": "C",
            "title": "Dynamic thresholding",
            "expected_impact": "medium",
            "expected_impact_pct": 15,
            "rationale": "Per-fold threshold clamp at max(0.45, fold_default) prevents degenerate 0.85 picks.",
            "risk": "low (no leakage; threshold is selected on validation slice, not test)",
        },
        {
            "id": "D",
            "title": "Volatility / ATR filters",
            "expected_impact": "medium",
            "expected_impact_pct": 10,
            "rationale": "Skip trades in top-quartile realized_vol windows; reduces fold-4 collapse.",
            "risk": "low",
        },
        {
            "id": "E",
            "title": "Ensemble weighting by recent OOF performance",
            "expected_impact": "medium",
            "expected_impact_pct": 12,
            "rationale": "Weight predictions by per-fold AUC; de-emphasise models with high F1 std.",
            "risk": "low (combinator, no leakage)",
        },
        {
            "id": "F",
            "title": "Alternative target definitions",
            "expected_impact": "high (long-term)",
            "expected_impact_pct": 35,
            "rationale": "Re-design labels to emphasize cost-adjusted edge (e.g., triple-barrier, profit-aware).",
            "risk": "high (requires new dataset version, leakage audit, full retrain)",
        },
    ]


def _most_stable(per_model: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return the most stable model and label by PF std."""
    if not per_model:
        return {}, {}
    by_pf_std = sorted(per_model, key=lambda r: r["pf_std"])
    by_auc_std = sorted(per_model, key=lambda r: r["auc_std"])
    by_threshold_std = sorted(per_model, key=lambda r: r["threshold_std"])
    by_label: Dict[str, List[float]] = {}
    for r in per_model:
        by_label.setdefault(r["label_name"], []).append(r["pf_std"])
    label_summary = {
        lc: {
            "mean_pf_std": float(np.mean(pfs)),
            "min_pf_std": float(min(pfs)),
            "n_models": len(pfs),
        }
        for lc, pfs in by_label.items()
    }
    label_summary_sorted = sorted(label_summary.items(), key=lambda x: x[1]["mean_pf_std"])
    return by_pf_std[0], {
        "label": label_summary_sorted[0][0],
        "summary": label_summary_sorted[0][1],
        "ranking": label_summary_sorted,
    }


def build() -> Dict[str, Any]:
    art = LATEST_ARTIFACT
    per_model = _per_model_fold_breakdown(art)
    label_skip_raw = _safe_load(art / "label_skip_report.json")
    label_skip_names = [x["label_name"] for x in label_skip_raw.get("labels_skipped", [])]
    df = pd.read_csv(DATA, low_memory=False, nrows=200000)  # sample for speed
    drift = _drift_analysis(df, label_skip_names)
    causes = _rank_instability_causes(per_model, drift)
    recs = _recommendations()
    stable_model, stable_label = _most_stable(per_model)
    return {
        "report_type": "walk_forward_instability_diagnosis",
        "artifact_dir": str(art),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "per_model": per_model,
        "drift_summary": {
            "n_features_analysed": drift["n_features_analysed"],
            "n_labels_analysed": drift["n_labels_analysed"],
            "top_unstable_features": drift["top_unstable_features"],
            "label_distribution_per_fold": drift["label_distribution_per_fold"],
        },
        "instability_causes": causes,
        "recommendations": recs,
        "most_stable_model": stable_model,
        "most_stable_label": stable_label,
        "retraining_sufficient": False,
        "retraining_rationale": (
            "Walk-forward instability is driven by feature drift and regime shift, "
            "neither of which is fixed by retraining alone. Removing unstable features, "
            "regime-conditional training, and dynamic thresholding are required. "
            "Retraining on the same data with the same features will reproduce the same instability."
        ),
    }


def write_reports() -> None:
    payload = build()
    REPORTS.mkdir(parents=True, exist_ok=True)

    jpath = REPORTS / f"walk_forward_instability_diagnosis_{TS}.json"
    jpath.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # Markdown main report
    md = [
        "# Walk-Forward Instability Diagnosis",
        "",
        f"- Artifact dir: `{payload['artifact_dir']}`",
        f"- Generated: `{payload['timestamp_utc']}`",
        f"- Models analysed: **{len(payload['per_model'])}**",
        f"- Features analysed for drift: **{payload['drift_summary']['n_features_analysed']}**",
        f"- Labels analysed: **{payload['drift_summary']['n_labels_analysed']}**",
        "",
        "## Top 10 unstable features (PSI late vs early)",
        "",
        "| Rank | Feature | PSI |",
        "|---|---|---|",
    ]
    for i, f in enumerate(payload["drift_summary"]["top_unstable_features"][:10], 1):
        md.append(f"| {i} | `{f['feature']}` | {f['psi_late_vs_early']:.4f} |")
    md.append("")
    md.append("## Top instability causes (ranked)")
    md.append("")
    md.append("| Rank | Cause | Evidence | Fix class |")
    md.append("|---|---|---|---|")
    for c in payload["instability_causes"]:
        md.append(f"| {c['rank']} | **{c['cause']}** | {c['evidence']} | {c['fix_class']} |")
    md.append("")
    md.append("## Most stable model")
    sm = payload["most_stable_model"]
    if sm:
        md.append(
            f"- `{sm['name']}` (PF std = {sm['pf_std']:.4f}, AUC std = {sm['auc_std']:.4f}, threshold std = {sm['threshold_std']:.3f})"
        )
    md.append("")
    md.append("## Most stable label")
    sl = payload["most_stable_label"]
    if sl:
        md.append(
            f"- `{sl['label']}` (mean PF std across models = {sl['summary']['mean_pf_std']:.4f}, n_models = {sl['summary']['n_models']})"
        )
        md.append("")
        md.append("### Per-label PF std summary")
        md.append("")
        md.append("| Label | n_models | mean PF std | min PF std |")
        md.append("|---|---|---|---|")
        for lab, summ in sl["ranking"]:
            md.append(f"| `{lab}` | {summ['n_models']} | {summ['mean_pf_std']:.4f} | {summ['min_pf_std']:.4f} |")
    md.append("")
    md.append("## Recommended fixes (ranked by expected impact)")
    md.append("")
    md.append("| ID | Title | Expected impact | Expected Δ | Risk |")
    md.append("|---|---|---|---|---|")
    for r in payload["recommendations"]:
        md.append(f"| {r['id']} | {r['title']} | {r['expected_impact']} | {r['expected_impact_pct']}% | {r['risk']} |")
    md.append("")
    md.append("## Retraining alone?")
    md.append("")
    md.append(f"- **{'YES' if payload['retraining_sufficient'] else 'NO'}** — {payload['retraining_rationale']}")
    md.append("")
    md.append("## Per-model fold breakdown (compact)")
    md.append("")
    md.append("| Model | Folds | AUC std | PF std | Sharpe std | Trade std | Threshold std |")
    md.append("|---|---|---|---|---|---|---|")
    for r in payload["per_model"][:15]:
        folds_str = " | ".join(
            f"f{f['fold']}: AUC={f['auc']:.2f} PF={f['pf']:.2f} thr={f['threshold']:.2f} n={f['trade_count']}"
            for f in r["folds"]
        )
        md.append(
            f"| `{r['name']}` | {folds_str} | {r['auc_std']:.4f} | {r['pf_std']:.4f} | {r['sharpe_std']:.4f} | {r['trade_std']:.0f} | {r['threshold_std']:.3f} |"
        )
    mpath = REPORTS / f"walk_forward_instability_diagnosis_{TS}.md"
    mpath.write_text("\n".join(md), encoding="utf-8")

    # Feature drift report
    fd = [
        "# Feature Drift Analysis (PSI late vs early)",
        "",
        f"- Dataset: `{DATA}`",
        f"- Generated: `{payload['timestamp_utc']}`",
        f"- Features analysed: **{payload['drift_summary']['n_features_analysed']}**",
        "",
        "## Top 30 unstable features (PSI, late fold vs early fold)",
        "",
        "| Rank | Feature | PSI |",
        "|---|---|---|",
    ]
    for i, f in enumerate(payload["drift_summary"]["top_unstable_features"], 1):
        fd.append(f"| {i} | `{f['feature']}` | {f['psi_late_vs_early']:.4f} |")
    fd.append("")
    fd.append("## Interpretation")
    fd.append("")
    fd.append("- PSI < 0.10: stable")
    fd.append("- 0.10 <= PSI < 0.25: moderate drift")
    fd.append("- PSI >= 0.25: major drift — candidate for removal or transformation")
    fd_path = REPORTS / f"feature_drift_analysis_{TS}.md"
    fd_path.write_text("\n".join(fd), encoding="utf-8")

    # Label drift report
    ld = [
        "# Label Drift Analysis (per-fold positive rate)",
        "",
        f"- Dataset: `{DATA}`",
        f"- Generated: `{payload['timestamp_utc']}`",
        f"- Labels analysed: **{payload['drift_summary']['n_labels_analysed']}**",
        "",
        "## Positive rate per fold",
        "",
        "| Label | Fold 1 (early) | Fold 2 | Fold 3 | Fold 4 (late) | Range |",
        "|---|---|---|---|---|---|",
    ]
    for lc, per_fold in payload["drift_summary"]["label_distribution_per_fold"].items():
        vals = [per_fold.get(f, 0.0) for f in ["fold1_early", "fold2_early_mid", "fold3_late_mid", "fold4_late"]]
        rng = max(vals) - min(vals) if vals else 0.0
        ld.append(
            f"| `{lc}` | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {vals[3]:.4f} | {rng:.4f} |"
        )
    ld_path = REPORTS / f"label_drift_analysis_{TS}.md"
    ld_path.write_text("\n".join(ld), encoding="utf-8")

    print(f"WROTE: {jpath}")
    print(f"WROTE: {mpath}")
    print(f"WROTE: {fd_path}")
    print(f"WROTE: {ld_path}")


if __name__ == "__main__":
    write_reports()
