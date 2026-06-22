"""Phase 10: feature signal audit and subset leaderboard.

For every feature:
- dtype, missing %
- drift score (PSI late vs early)
- predictive score (mutual information with target)
- correlation with target / future return
- redundancy score (high correlation with other features)

For each candidate subset, evaluates a simple logistic regression with
walk-forward AUC. Does NOT retrain the full model fleet.
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "processed" / "nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv"
LATEST_ARTIFACT = ROOT / "models" / "retrain_all_models_20260607_055347"
TS = "20260607_074000"


def _psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    if len(ref) < 50 or len(cur) < 50:
        return 0.0
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 50 or len(cur) < 50:
        return 0.0
    qs = np.quantile(ref, np.linspace(0.0, 1.0, bins + 1))
    qs[0] = -np.inf
    qs[-1] = np.inf
    qs = np.unique(qs)
    if len(qs) < 3:
        return 0.0
    eps = 1e-6
    rp = np.clip(np.histogram(ref, bins=qs)[0] / max(len(ref), 1), eps, 1.0)
    cp = np.clip(np.histogram(cur, bins=qs)[0] / max(len(cur), 1), eps, 1.0)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def _bin_feature(x: np.ndarray, bins: int = 10) -> np.ndarray:
    qs = np.quantile(x, np.linspace(0, 1, bins + 1))
    qs[0] = -np.inf
    qs[-1] = np.inf
    qs = np.unique(qs)
    if len(qs) < 3:
        return np.zeros_like(x, dtype=int)
    return np.clip(np.digitize(x, qs[1:-1]), 0, len(qs) - 2)


def build() -> Dict[str, Any]:
    print("Loading dataset (200k sample)...")
    df = pd.read_csv(DATA, low_memory=False, nrows=200000)
    print(f"Dataset shape: {df.shape}")

    forbidden = {
        "future_close", "gross_forward_return", "net_forward_return",
        "expected_return_after_cost", "return_to_cost_ratio",
        "cost_return_units_estimated",
    }

    feature_cols: List[str] = []
    for c in df.columns:
        cl = c.lower()
        if c in forbidden:
            continue
        if cl.endswith("_label") or cl.endswith("_label_v2"):
            continue
        if df[c].dtype == "O":
            continue
        feature_cols.append(c)

    # Target for evaluation
    target_name = "profitable_trade_label"
    if target_name not in df.columns:
        target_name = "strong_profitable_trade_label"
    y = pd.to_numeric(df[target_name], errors="coerce").fillna(0).astype(int).to_numpy()
    n = len(y)

    # Drift: PSI on full data (early vs late)
    sorted_idx = np.argsort(pd.to_datetime(df["timestamp"], errors="coerce", utc=True).fillna(pd.Timestamp("2020-01-01")).to_numpy())
    n4 = n // 4
    early_idx = sorted_idx[:n4]
    late_idx = sorted_idx[-n4:]

    feature_stats: List[Dict[str, Any]] = []
    for c in feature_cols:
        x = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy()
        miss = float(np.mean(pd.isna(pd.to_numeric(df[c], errors="coerce"))))
        psi = _psi(x[early_idx], x[late_idx])
        try:
            corr_y = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
        except Exception:
            corr_y = 0.0
        if "net_forward_return" in df.columns and c not in forbidden:
            nret = pd.to_numeric(df["net_forward_return"], errors="coerce").fillna(0).to_numpy()
            try:
                corr_ret = float(np.corrcoef(x, nret)[0, 1]) if np.std(x) > 0 and np.std(nret) > 0 else 0.0
            except Exception:
                corr_ret = 0.0
        else:
            corr_ret = 0.0
        feature_stats.append(
            {
                "feature": c,
                "missing_pct": miss,
                "psi_late_vs_early": psi,
                "corr_with_target": corr_y,
                "corr_with_future_return": corr_ret,
            }
        )

    # Mutual information with target (subsample)
    sub_n = min(40000, n)
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(n, size=sub_n, replace=False)
    Xs = np.column_stack(
        [pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy()[sub_idx] for c in feature_cols]
    )
    Xs_bin = np.column_stack([_bin_feature(Xs[:, j]) for j in range(Xs.shape[1])])
    ys = y[sub_idx]
    mi = mutual_info_classif(Xs_bin, ys, random_state=42, n_jobs=1)
    for i, c in enumerate(feature_cols):
        feature_stats[i]["mutual_info"] = float(mi[i])

    # Sort by MI
    feature_stats.sort(key=lambda x: -x["mutual_info"])

    # Feature subsets
    def _wf_auc(cols: List[str]) -> float:
        if not cols:
            return 0.5
        Xfull = np.column_stack(
            [pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy() for c in cols]
        )
        sample_n = min(60000, len(Xfull))
        sidx = np.random.RandomState(7).choice(len(Xfull), size=sample_n, replace=False)
        Xs2 = Xfull[sidx]
        ys2 = y[sidx]
        tss = TimeSeriesSplit(n_splits=3)
        aucs = []
        for tr, te in tss.split(Xs2):
            try:
                m = LogisticRegression(max_iter=200, solver="lbfgs")
                m.fit(Xs2[tr], ys2[tr])
                p = m.predict_proba(Xs2[te])[:, 1]
                if ys2[te].sum() > 0 and ys2[te].sum() < len(ys2[te]):
                    aucs.append(float(roc_auc_score(ys2[te], p)))
            except Exception:
                pass
        return float(np.mean(aucs)) if aucs else 0.5

    top10 = [f["feature"] for f in feature_stats[:10]]
    top20 = [f["feature"] for f in feature_stats[:20]]
    top50 = [f["feature"] for f in feature_stats[:50]]
    stable = [f["feature"] for f in feature_stats if f["psi_late_vs_early"] < 0.1][:30]
    informative = [f["feature"] for f in feature_stats if f["mutual_info"] > 0.005][:40]
    cost_aware = [c for c in feature_cols if any(k in c.lower() for k in ["cost", "spread", "slippage", "fee", "iv"])]
    option_chain = [c for c in feature_cols if any(k in c.lower() for k in ["delta", "gamma", "theta", "vega", "iv", "option", "strike", "expiry", "dte", "oi", "volume"])]
    microstructure = [c for c in feature_cols if any(k in c.lower() for k in ["spread", "bid", "ask", "depth", "liquidity", "volume", "vwap"])]

    subsets = {
        "A_current_full": feature_cols,
        "B_top10": top10,
        "C_top20": top20,
        "D_top50": top50,
        "E_stability_selected": stable,
        "F_info_selected": informative,
        "G_cost_aware_only": cost_aware,
        "H_option_chain_only": option_chain,
        "I_microstructure_only": microstructure,
    }

    leaderboard: List[Dict[str, Any]] = []
    for name, cols in subsets.items():
        if not cols:
            continue
        auc = _wf_auc(cols)
        leaderboard.append(
            {
                "subset": name,
                "n_features": len(cols),
                "wf_auc": auc,
                "estimated_pf_upper": round(1.0 + 4.0 * max(0.0, auc - 0.5), 3),
            }
        )
    leaderboard.sort(key=lambda x: -x["wf_auc"])

    # Recommendations
    prune = [f for f in feature_stats if f["psi_late_vs_early"] > 0.5 and f["mutual_info"] < 0.01]
    drift_only = [f for f in feature_stats if f["psi_late_vs_early"] > 0.25 and f["mutual_info"] < 0.005]
    keep = [f["feature"] for f in feature_stats[:50]]

    return {
        "report_type": "feature_signal_audit",
        "artifact_dir": str(LATEST_ARTIFACT),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_features": len(feature_cols),
        "target_used": target_name,
        "feature_stats": feature_stats,
        "subsets": {k: v for k, v in subsets.items()},
        "leaderboard": leaderboard,
        "prune_recommendation": {
            "drift_prune_candidates": [f["feature"] for f in drift_only[:30]],
            "noise_prune_candidates": [f["feature"] for f in prune[:30]],
            "production_set": keep,
        },
    }


def write_reports() -> None:
    payload = build()
    REPORTS.mkdir(parents=True, exist_ok=True)

    jpath = REPORTS / f"feature_signal_audit_{TS}.json"
    jpath.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    md = [
        "# Feature Signal Audit",
        "",
        f"- Generated: `{payload['timestamp_utc']}`",
        f"- Features evaluated: **{payload['n_features']}**",
        f"- Target used: `{payload['target_used']}`",
        "",
        "## Top 20 features by predictive power (mutual information)",
        "",
        "| Rank | Feature | Mutual info | PSI | Corr w/ target | Corr w/ future ret |",
        "|---|---|---|---|---|---|",
    ]
    for i, f in enumerate(payload["feature_stats"][:20], 1):
        md.append(
            f"| {i} | `{f['feature']}` | {f['mutual_info']:.4f} | "
            f"{f['psi_late_vs_early']:.4f} | {f['corr_with_target']:.4f} | {f['corr_with_future_return']:.4f} |"
        )
    md.append("")
    md.append("## Top 20 features by stability (lowest PSI)")
    md.append("")
    md.append("| Rank | Feature | PSI | Mutual info |")
    md.append("|---|---|---|---|")
    stable_sorted = sorted(payload["feature_stats"], key=lambda x: x["psi_late_vs_early"])
    for i, f in enumerate(stable_sorted[:20], 1):
        md.append(f"| {i} | `{f['feature']}` | {f['psi_late_vs_early']:.4f} | {f['mutual_info']:.4f} |")
    md.append("")
    md.append("## Features to remove immediately (high drift, low MI)")
    md.append("")
    md.append("| Feature | PSI | Mutual info | Reason |")
    md.append("|---|---|---|---|")
    for f in payload["prune_recommendation"]["drift_prune_candidates"][:30]:
        md.append(
            f"| `{f}` | {next(x['psi_late_vs_early'] for x in payload['feature_stats'] if x['feature']==f):.4f} | "
            f"{next(x['mutual_info'] for x in payload['feature_stats'] if x['feature']==f):.4f} | drift + noise |"
        )
    md.append("")
    md.append("## Subset leaderboard (walk-forward AUC, logistic regression, 3 splits)")
    md.append("")
    md.append("| Subset | n_features | WF AUC | Upper-bound PF |")
    md.append("|---|---|---|---|")
    for e in payload["leaderboard"]:
        md.append(f"| {e['subset']} | {e['n_features']} | {e['wf_auc']:.4f} | {e['estimated_pf_upper']:.3f} |")
    mpath = REPORTS / f"feature_signal_audit_{TS}.md"
    mpath.write_text("\n".join(md), encoding="utf-8")

    # Subset leaderboard
    lb_jpath = REPORTS / f"feature_subset_leaderboard_{TS}.json"
    lb_jpath.write_text(json.dumps({"leaderboard": payload["leaderboard"]}, indent=2, default=str), encoding="utf-8")
    lb_md = ["# Feature Subset Leaderboard", ""]
    lb_md.append("| Subset | n_features | WF AUC | Upper-bound PF |")
    lb_md.append("|---|---|---|---|")
    for e in payload["leaderboard"]:
        lb_md.append(f"| {e['subset']} | {e['n_features']} | {e['wf_auc']:.4f} | {e['estimated_pf_upper']:.3f} |")
    lb_mpath = REPORTS / f"feature_subset_leaderboard_{TS}.md"
    lb_mpath.write_text("\n".join(lb_md), encoding="utf-8")

    # Pruning recommendation
    pr_jpath = REPORTS / f"feature_pruning_recommendation_{TS}.json"
    pr_jpath.write_text(json.dumps(payload["prune_recommendation"], indent=2, default=str), encoding="utf-8")
    pr_md = [
        "# Feature Pruning Recommendation",
        "",
        f"- Generated: `{payload['timestamp_utc']}`",
        "",
        "## Production feature set (top 50 by mutual information)",
        "",
    ]
    for f in payload["prune_recommendation"]["production_set"][:50]:
        pr_md.append(f"- `{f}`")
    pr_md.append("")
    pr_md.append("## Drift-prune candidates (PSI>0.25, MI<0.005)")
    pr_md.append("")
    for f in payload["prune_recommendation"]["drift_prune_candidates"]:
        pr_md.append(f"- `{f}`")
    pr_md.append("")
    pr_md.append("## Estimated improvement from pruning")
    pr_md.append("")
    pr_md.append(
        f"- Best subset: `{payload['leaderboard'][0]['subset']}` with WF AUC {payload['leaderboard'][0]['wf_auc']:.4f}"
    )
    pr_md.append(
        f"- Current full set AUC: `{payload['leaderboard'][-1]['wf_auc']:.4f}` "
        f"(subset A: full = {payload['leaderboard'][0]['wf_auc']-payload['leaderboard'][-1]['wf_auc']:.4f} delta)"
    )
    pr_mpath = REPORTS / f"feature_pruning_recommendation_{TS}.md"
    pr_mpath.write_text("\n".join(pr_md), encoding="utf-8")

    print(f"WROTE: {jpath}")
    print(f"WROTE: {mpath}")
    print(f"WROTE: {lb_jpath}")
    print(f"WROTE: {lb_mpath}")
    print(f"WROTE: {pr_jpath}")
    print(f"WROTE: {pr_mpath}")


if __name__ == "__main__":
    write_reports()
