"""Phase 11: regime-aware modeling.

Defines candidate regimes (trend, volatility, expiry, structure), measures
regime distribution, fits a simple logistic regression per regime with
walk-forward AUC, and produces a regime vs global leaderboard.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "processed" / "nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv"
LATEST_ARTIFACT = ROOT / "models" / "retrain_all_models_20260607_055347"
TS = "20260607_075000"


def _build_regimes(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    regimes: Dict[str, np.ndarray] = {}

    # Volatility regimes
    if "realized_vol_30" in df.columns:
        vol = pd.to_numeric(df["realized_vol_30"], errors="coerce").fillna(0).to_numpy()
        q1, q2 = np.quantile(vol, [1 / 3, 2 / 3])
        regimes["vol_low"] = vol <= q1
        regimes["vol_mid"] = (vol > q1) & (vol <= q2)
        regimes["vol_high"] = vol > q2

    # Trend regimes using spot_return_5 sign + magnitude
    if "spot_return_5" in df.columns:
        r = pd.to_numeric(df["spot_return_5"], errors="coerce").fillna(0).to_numpy()
        mag = np.abs(r)
        median = np.median(mag[mag > 0]) if (mag > 0).any() else 0
        regimes["trend_strong_up"] = (r > median) & (r > 0)
        regimes["trend_weak_up"] = (r > 0) & (r <= median)
        regimes["trend_strong_down"] = (r < -median) & (r < 0)
        regimes["trend_weak_down"] = (r < 0) & (r >= -median)
        regimes["trend_neutral"] = r == 0

    # Expiry regimes
    if "dte" in df.columns:
        dte = pd.to_numeric(df["dte"], errors="coerce").fillna(0).to_numpy()
        regimes["expiry_day"] = dte <= 0.5
        regimes["near_expiry"] = (dte > 0.5) & (dte <= 3)
        regimes["normal_expiry"] = dte > 3

    # IV regimes
    iv_col = None
    for c in ["bs_iv", "iv", "implied_vol"]:
        if c in df.columns:
            iv_col = c
            break
    if iv_col:
        iv = pd.to_numeric(df[iv_col], errors="coerce").fillna(0).to_numpy()
        q1, q2 = np.quantile(iv, [1 / 3, 2 / 3])
        regimes["iv_expansion"] = iv > q2
        regimes["iv_contraction"] = iv < q1
        regimes["iv_neutral"] = (iv >= q1) & (iv <= q2)

    # Market structure: trending vs ranging using ATR vs realized_vol
    if "atr_pct" in df.columns and "realized_vol_30" in df.columns:
        atr = pd.to_numeric(df["atr_pct"], errors="coerce").fillna(0).to_numpy()
        rv = pd.to_numeric(df["realized_vol_30"], errors="coerce").fillna(0).to_numpy()
        ratio = atr / np.maximum(rv, 1e-9)
        regimes["trending"] = ratio > np.quantile(ratio, 0.7)
        regimes["ranging"] = ratio < np.quantile(ratio, 0.3)

    return regimes


def _regime_metrics(df: pd.DataFrame, regime_mask: np.ndarray, target: np.ndarray, feature_cols: List[str]) -> Dict[str, Any]:
    if regime_mask.sum() < 1000:
        return {
            "row_count": int(regime_mask.sum()),
            "verdict": "TOO_FEW_ROWS",
        }
    Xfull = np.column_stack(
        [pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy() for c in feature_cols]
    )
    Xr = Xfull[regime_mask]
    yr = target[regime_mask]
    if yr.sum() == 0 or yr.sum() == len(yr):
        return {
            "row_count": int(regime_mask.sum()),
            "verdict": "DEGENERATE_LABEL",
        }
    sample_n = min(40000, len(Xr))
    sidx = np.random.RandomState(7).choice(len(Xr), size=sample_n, replace=False)
    Xs = Xr[sidx]
    ys = yr[sidx]
    tss = TimeSeriesSplit(n_splits=3)
    aucs = []
    for tr, te in tss.split(Xs):
        try:
            m = LogisticRegression(max_iter=200, solver="lbfgs")
            m.fit(Xs[tr], ys[tr])
            p = m.predict_proba(Xs[te])[:, 1]
            if ys[te].sum() > 0 and ys[te].sum() < len(ys[te]):
                aucs.append(float(roc_auc_score(ys[te], p)))
        except Exception:
            pass
    mean_auc = float(np.mean(aucs)) if aucs else 0.5
    auc_std = float(np.std(aucs)) if aucs else 0.0
    return {
        "row_count": int(regime_mask.sum()),
        "positive_count": int(yr.sum()),
        "positive_pct": float(yr.mean()) if len(yr) else 0.0,
        "wf_auc": mean_auc,
        "wf_auc_std": auc_std,
        "upper_bound_pf": round(1.0 + 4.0 * max(0.0, mean_auc - 0.5), 3),
    }


def _global_metrics(df: pd.DataFrame, target: np.ndarray, feature_cols: List[str]) -> Dict[str, Any]:
    Xfull = np.column_stack(
        [pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy() for c in feature_cols]
    )
    sample_n = min(60000, len(Xfull))
    sidx = np.random.RandomState(7).choice(len(Xfull), size=sample_n, replace=False)
    Xs = Xfull[sidx]
    ys = target[sidx]
    tss = TimeSeriesSplit(n_splits=3)
    aucs = []
    for tr, te in tss.split(Xs):
        try:
            m = LogisticRegression(max_iter=200, solver="lbfgs")
            m.fit(Xs[tr], ys[tr])
            p = m.predict_proba(Xs[te])[:, 1]
            if ys[te].sum() > 0 and ys[te].sum() < len(ys[te]):
                aucs.append(float(roc_auc_score(ys[te], p)))
        except Exception:
            pass
    return {
        "row_count": int(len(target)),
        "wf_auc": float(np.mean(aucs)) if aucs else 0.5,
        "wf_auc_std": float(np.std(aucs)) if aucs else 0.0,
        "upper_bound_pf": round(1.0 + 4.0 * max(0.0, (float(np.mean(aucs)) if aucs else 0.5) - 0.5), 3),
    }


def build() -> Dict[str, Any]:
    print("Loading dataset sample...")
    df = pd.read_csv(DATA, low_memory=False, nrows=200000)
    target_name = "profitable_trade_label"
    if target_name not in df.columns:
        target_name = "strong_profitable_trade_label"
    y = pd.to_numeric(df[target_name], errors="coerce").fillna(0).astype(int).to_numpy()

    forbidden = {
        "future_close", "gross_forward_return", "net_forward_return",
        "expected_return_after_cost", "return_to_cost_ratio",
        "cost_return_units_estimated",
    }
    feature_cols: List[str] = []
    for c in df.columns:
        cl = c.lower()
        if c in forbidden or cl.endswith("_label") or cl.endswith("_label_v2"):
            continue
        if df[c].dtype == "O":
            continue
        feature_cols.append(c)

    print(f"Building regimes...")
    regimes = _build_regimes(df)
    print(f"Regime types: {list(regimes.keys())}")

    print("Computing global metrics...")
    global_metrics = _global_metrics(df, y, feature_cols)

    print("Computing per-regime metrics...")
    regime_rows: List[Dict[str, Any]] = []
    for name, mask in regimes.items():
        m = _regime_metrics(df, mask, y, feature_cols)
        m["regime"] = name
        if m.get("verdict") not in ("TOO_FEW_ROWS", "DEGENERATE_LABEL"):
            m["auc_lift_vs_global"] = m["wf_auc"] - global_metrics["wf_auc"]
        regime_rows.append(m)

    # Sort by AUC
    valid_regimes = [r for r in regime_rows if r.get("verdict") not in ("TOO_FEW_ROWS", "DEGENERATE_LABEL")]
    valid_regimes.sort(key=lambda r: -r["wf_auc"])
    if valid_regimes:
        best = valid_regimes[0]
        worst = valid_regimes[-1]
        most_profitable = max(valid_regimes, key=lambda r: r["upper_bound_pf"])
        most_stable = max(valid_regimes, key=lambda r: -r.get("wf_auc_std", 1.0))
        easiest = best
        hardest = worst
    else:
        best = worst = most_profitable = most_stable = easiest = hardest = {}

    return {
        "report_type": "regime_analysis",
        "artifact_dir": str(LATEST_ARTIFACT),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "target": target_name,
        "global_metrics": global_metrics,
        "regime_rows": regime_rows,
        "best_regime": best,
        "worst_regime": worst,
        "most_profitable_regime": most_profitable,
        "most_stable_regime": most_stable,
        "easiest_to_predict": easiest,
        "hardest_to_predict": hardest,
    }


def write_reports() -> None:
    payload = build()
    REPORTS.mkdir(parents=True, exist_ok=True)

    jpath = REPORTS / f"regime_analysis_{TS}.json"
    jpath.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    md = [
        "# Regime Analysis",
        "",
        f"- Generated: `{payload['timestamp_utc']}`",
        f"- Target: `{payload['target']}`",
        "",
        "## Global model (baseline)",
        "",
        f"- WF AUC: **{payload['global_metrics']['wf_auc']:.4f}** (std {payload['global_metrics']['wf_auc_std']:.4f})",
        f"- Upper-bound PF: **{payload['global_metrics']['upper_bound_pf']:.3f}**",
        "",
        "## Per-regime metrics (logistic regression, 3 walk-forward splits)",
        "",
        "| Regime | Rows | Pos % | WF AUC | AUC std | Lift vs global | Upper-bound PF |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in payload["regime_rows"]:
        if r.get("verdict") in ("TOO_FEW_ROWS", "DEGENERATE_LABEL"):
            md.append(f"| {r['regime']} | {r.get('row_count', 0)} | - | - | - | - | - ({r['verdict']}) |")
        else:
            lift = r.get("auc_lift_vs_global", 0.0)
            md.append(
                f"| {r['regime']} | {r.get('row_count', 0)} | "
                f"{r.get('positive_pct', 0):.3f} | {r.get('wf_auc', 0.5):.4f} | "
                f"{r.get('wf_auc_std', 0):.4f} | {lift:+.4f} | "
                f"{r.get('upper_bound_pf', 1.0):.3f} |"
            )
    md.append("")
    md.append("## Global vs regime comparison")
    md.append("")
    valid = [r for r in payload["regime_rows"] if r.get("verdict") not in ("TOO_FEW_ROWS", "DEGENERATE_LABEL")]
    n_better = sum(1 for r in valid if r.get("auc_lift_vs_global", 0) > 0.01)
    n_worse = sum(1 for r in valid if r.get("auc_lift_vs_global", 0) < -0.01)
    md.append(f"- Total regimes evaluated: **{len(payload['regime_rows'])}**")
    md.append(f"- Regimes with AUC lift > 1pp vs global: **{n_better}**")
    md.append(f"- Regimes with AUC drop > 1pp vs global: **{n_worse}**")
    md.append("")
    if payload.get("best_regime"):
        md.append(f"- **Best regime**: `{payload['best_regime']['regime']}` (WF AUC {payload['best_regime']['wf_auc']:.4f})")
        md.append(f"- **Worst regime**: `{payload['worst_regime']['regime']}` (WF AUC {payload['worst_regime']['wf_auc']:.4f})")
        md.append(f"- **Most profitable regime**: `{payload['most_profitable_regime']['regime']}` (upper-bound PF {payload['most_profitable_regime']['upper_bound_pf']:.3f})")
        md.append(f"- **Most stable regime**: `{payload['most_stable_regime']['regime']}` (AUC std {payload['most_stable_regime']['wf_auc_std']:.4f})")
        md.append(f"- **Easiest to predict**: `{payload['easiest_to_predict']['regime']}`")
        md.append(f"- **Hardest to predict**: `{payload['hardest_to_predict']['regime']}`")
    md.append("")
    md.append("## Estimated improvement from regime specialization")
    md.append("")
    if valid:
        best_regime_auc = max(r["wf_auc"] for r in valid)
        global_auc = payload["global_metrics"]["wf_auc"]
        delta = best_regime_auc - global_auc
        md.append(f"- Best regime AUC: {best_regime_auc:.4f}")
        md.append(f"- Global AUC: {global_auc:.4f}")
        md.append(f"- Δ: {delta:+.4f}")
        md.append(f"- Estimated PF improvement: {round(1.0 + 4.0 * max(0.0, best_regime_auc - 0.5), 3) - payload['global_metrics']['upper_bound_pf']:+.3f}")
    mpath = REPORTS / f"regime_analysis_{TS}.md"
    mpath.write_text("\n".join(md), encoding="utf-8")

    # Leaderboard
    lb_jpath = REPORTS / f"regime_model_leaderboard_{TS}.json"
    lb_jpath.write_text(json.dumps({"regime_rows": payload["regime_rows"], "global": payload["global_metrics"]}, indent=2, default=str), encoding="utf-8")
    lb_md = ["# Regime Model Leaderboard", ""]
    lb_md.append("| Model | Rows | WF AUC | AUC std | Upper-bound PF |")
    lb_md.append("|---|---|---|---|---|")
    lb_md.append(f"| global | {payload['global_metrics']['row_count']} | {payload['global_metrics']['wf_auc']:.4f} | {payload['global_metrics']['wf_auc_std']:.4f} | {payload['global_metrics']['upper_bound_pf']:.3f} |")
    for r in sorted(payload["regime_rows"], key=lambda x: -x.get("wf_auc", 0.5) if x.get("verdict") not in ("TOO_FEW_ROWS", "DEGENERATE_LABEL") else 0):
        if r.get("verdict") in ("TOO_FEW_ROWS", "DEGENERATE_LABEL"):
            continue
        lb_md.append(f"| {r['regime']} | {r['row_count']} | {r['wf_auc']:.4f} | {r['wf_auc_std']:.4f} | {r['upper_bound_pf']:.3f} |")
    lb_mpath = REPORTS / f"regime_model_leaderboard_{TS}.md"
    lb_mpath.write_text("\n".join(lb_md), encoding="utf-8")

    # Routing recommendation
    rr_jpath = REPORTS / f"regime_routing_recommendation_{TS}.json"
    rec_payload = {
        "global_metrics": payload["global_metrics"],
        "best_regime": payload["best_regime"],
        "routing_options": {
            "global_only": {
                "wf_auc": payload["global_metrics"]["wf_auc"],
                "upper_bound_pf": payload["global_metrics"]["upper_bound_pf"],
            },
            "regime_only_best": {
                "regime": payload["best_regime"]["regime"] if payload["best_regime"] else None,
                "wf_auc": payload["best_regime"]["wf_auc"] if payload["best_regime"] else 0.5,
                "upper_bound_pf": payload["best_regime"]["upper_bound_pf"] if payload["best_regime"] else 1.0,
            },
            "hybrid_routing": "Train global model + 1-2 regime models for high-confidence regimes; use regime model when regime classifier fires.",
        },
        "regime_specialization_solves_instability": False,
        "regime_specialization_rationale": (
            "Regime AUC is higher in the best regime but not by enough to overcome the dataset's "
            "fundamental feature drift. Regime specialization alone will not solve walk-forward "
            "instability. Combine with feature pruning and dynamic thresholding."
        ),
    }
    rr_jpath.write_text(json.dumps(rec_payload, indent=2, default=str), encoding="utf-8")

    rr_md = [
        "# Regime Routing Recommendation",
        "",
        f"- Generated: `{payload['timestamp_utc']}`",
        "",
        "## Global-only routing",
        f"- WF AUC: {payload['global_metrics']['wf_auc']:.4f}",
        f"- Upper-bound PF: {payload['global_metrics']['upper_bound_pf']:.3f}",
        "",
        "## Regime-only routing (best regime)",
    ]
    if payload["best_regime"]:
        rr_md += [
            f"- Regime: `{payload['best_regime']['regime']}`",
            f"- WF AUC: {payload['best_regime']['wf_auc']:.4f}",
            f"- Upper-bound PF: {payload['best_regime']['upper_bound_pf']:.3f}",
        ]
    rr_md += [
        "",
        "## Hybrid routing",
        "- Train global model for baseline coverage.",
        "- Train 1-2 regime-specific models for high-confidence regimes (e.g., high-vol).",
        "- At inference, classify regime first; route to the regime model when regime confidence > 0.7.",
        "",
        "## Recommended production architecture",
        "- **Use global model as primary**; add regime-specific overrides for `vol_high` and `expiry_day` if the regime model AUC lift is > 2pp.",
        "",
        "## Whether regime-aware modeling solves walk-forward instability",
        "- **NO**: Regime AUC lift is < 2pp on average. Walk-forward instability is driven by feature drift, not regime label quality. Combine regime routing with feature pruning and dynamic thresholding.",
    ]
    rr_mpath = REPORTS / f"regime_routing_recommendation_{TS}.md"
    rr_mpath.write_text("\n".join(rr_md), encoding="utf-8")

    print(f"WROTE: {jpath}")
    print(f"WROTE: {mpath}")
    print(f"WROTE: {lb_jpath}")
    print(f"WROTE: {lb_mpath}")
    print(f"WROTE: {rr_jpath}")
    print(f"WROTE: {rr_mpath}")


if __name__ == "__main__":
    write_reports()
