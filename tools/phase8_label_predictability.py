"""Phase 8: label predictability audit.

For each usable label, measures:
- positive class %
- class balance over time / month / volatility / expiry
- mutual information with features
- feature-label correlation
- signal-to-noise ratio
- label entropy

Compares against simple baselines (always positive, always negative, random,
prev-direction, volatility-only, trend-only).

Emits:
- reports/label_predictability_audit_<ts>.md
- reports/label_predictability_audit_<ts>.json
- reports/label_leaderboard_<ts>.md
- reports/label_leaderboard_<ts>.json
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
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "processed" / "nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv"
LATEST_ARTIFACT = MODELS / "retrain_all_models_20260607_055347"
TS = "20260607_072000"

USABLE_LABELS = [
    "profitable_trade_label",
    "avoid_trade_label",
    "strong_profitable_trade_label",
    "cost_survivor_label",
    "strong_profitable_trade_label_v2",
    "cost_survivor_label_v2",
    "weak_trade_label",
    "no_trade_label",
]


def _entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def _bin_feature(x: np.ndarray, bins: int = 10) -> np.ndarray:
    qs = np.quantile(x, np.linspace(0, 1, bins + 1))
    qs[0] = -np.inf
    qs[-1] = np.inf
    qs = np.unique(qs)
    if len(qs) < 3:
        return np.zeros_like(x, dtype=int)
    return np.clip(np.digitize(x, qs[1:-1]), 0, len(qs) - 2)


def _safe_load(p: Path) -> Dict[str, Any]:
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


import math


def _per_label_metrics(df: pd.DataFrame, label: str, feature_cols: List[str]) -> Dict[str, Any]:
    y = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int).to_numpy()
    n = len(y)
    pos = int(y.sum())
    pos_pct = pos / max(n, 1)
    if pos_pct == 0 or pos_pct == 1:
        return {
            "label": label,
            "n_rows": n,
            "positive_count": pos,
            "positive_pct": pos_pct,
            "label_entropy": 0.0,
            "skip_reason": "degenerate_constant",
        }

    # Mutual information with all features (subsample for speed)
    sub_n = min(50000, n)
    idx = np.random.RandomState(42).choice(n, size=sub_n, replace=False)
    Xs = np.column_stack(
        [
            pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy()[idx]
            for c in feature_cols
        ]
    )
    Xs_bin = np.column_stack([_bin_feature(Xs[:, j]) for j in range(Xs.shape[1])])
    ys = y[idx]
    mi = mutual_info_classif(Xs_bin, ys, random_state=42, n_jobs=1)
    top5_idx = np.argsort(mi)[-5:][::-1]
    top5 = [(feature_cols[i], float(mi[i])) for i in top5_idx]

    # Per-month positive rate
    df_local = df.copy()
    if "timestamp" in df_local.columns:
        df_local["_month"] = pd.to_datetime(df_local["timestamp"], errors="coerce", utc=True).dt.to_period("M").astype(str)
    elif "month" in df_local.columns:
        df_local["_month"] = df_local["month"].astype(str)
    else:
        df_local["_month"] = "all"
    per_month = (
        df_local.groupby("_month")[label].mean().to_dict()
        if "_month" in df_local.columns
        else {}
    )
    month_vals = [v for v in per_month.values() if v is not None and not (isinstance(v, float) and np.isnan(v))]
    month_range = (max(month_vals) - min(month_vals)) if month_vals else 0.0
    month_std = float(np.std(month_vals)) if month_vals else 0.0

    # Volatility regime
    vol_col = "realized_vol_30" if "realized_vol_30" in df_local.columns else None
    per_vol = {}
    if vol_col:
        vol = pd.to_numeric(df_local[vol_col], errors="coerce").fillna(0).to_numpy()
        q33, q66 = np.quantile(vol, [1 / 3, 2 / 3])
        for label_lvl, mask in [("low", vol <= q33), ("mid", (vol > q33) & (vol <= q66)), ("high", vol > q66)]:
            if mask.sum() > 0:
                per_vol[label_lvl] = float(y[mask].mean())
    vol_range = (max(per_vol.values()) - min(per_vol.values())) if per_vol else 0.0

    # Expiry regime
    per_expiry = {}
    if "dte" in df_local.columns:
        dte = pd.to_numeric(df_local["dte"], errors="coerce").fillna(0).to_numpy()
        for lvl, mask in [("0dte", dte <= 0.5), ("near", (dte > 0.5) & (dte <= 3)), ("weekly", (dte > 3) & (dte <= 7)), ("monthly", dte > 7)]:
            if mask.sum() > 0:
                per_expiry[lvl] = float(y[mask].mean())
    expiry_range = (max(per_expiry.values()) - min(per_expiry.values())) if per_expiry else 0.0

    # Walk-forward AUC (5 splits) on a simple model
    sample_n = min(80000, n)
    sidx = np.random.RandomState(7).choice(n, size=sample_n, replace=False)
    Xs2 = np.column_stack(
        [pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy()[sidx] for c in feature_cols]
    )
    ys2 = y[sidx]
    tss = TimeSeriesSplit(n_splits=4)
    aucs = []
    for train_idx, test_idx in tss.split(Xs2):
        try:
            m = LogisticRegression(max_iter=200, solver="lbfgs")
            m.fit(Xs2[train_idx], ys2[train_idx])
            p = m.predict_proba(Xs2[test_idx])[:, 1]
            if ys2[test_idx].sum() > 0 and ys2[test_idx].sum() < len(ys2[test_idx]):
                aucs.append(float(roc_auc_score(ys2[test_idx], p)))
        except Exception:
            pass
    mean_auc = float(np.mean(aucs)) if aucs else 0.5
    auc_std = float(np.std(aucs)) if aucs else 0.0

    return {
        "label": label,
        "n_rows": n,
        "positive_count": pos,
        "positive_pct": pos_pct,
        "label_entropy": _entropy(pos_pct),
        "mi_top5": top5,
        "mi_mean": float(np.mean(mi)),
        "mi_max": float(np.max(mi)),
        "per_month_positive": {str(k): float(v) for k, v in per_month.items() if v is not None and not (isinstance(v, float) and np.isnan(v))},
        "month_range": month_range,
        "month_std": month_std,
        "per_vol_regime": per_vol,
        "vol_range": vol_range,
        "per_expiry_regime": per_expiry,
        "expiry_range": expiry_range,
        "walk_forward_auc": mean_auc,
        "walk_forward_auc_std": auc_std,
    }


def _baseline_metrics(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    """Compare simple baselines against the label."""
    y = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int).to_numpy()
    n = len(y)
    pos_pct = float(y.mean())
    base_pf = pos_pct / max(1 - pos_pct, 1e-9)

    # Random classifier AUC = 0.5
    # Always positive: PF = pos_pct / (1 - pos_pct)
    # Always negative: PF = 0 (no trades)
    auc_random = 0.5
    auc_always_pos = 0.5  # degenerate

    # Previous-candle direction
    if "spot_return_1" in df.columns:
        prev_dir = (pd.to_numeric(df["spot_return_1"], errors="coerce").fillna(0).to_numpy() > 0).astype(int)
        if prev_dir.sum() > 0 and prev_dir.sum() < n:
            try:
                auc_prev = float(roc_auc_score(y, prev_dir))
            except Exception:
                auc_prev = 0.5
        else:
            auc_prev = 0.5
    else:
        auc_prev = 0.5

    # Volatility-only rule
    if "realized_vol_30" in df.columns:
        vol = pd.to_numeric(df["realized_vol_30"], errors="coerce").fillna(0).to_numpy()
        high_vol = (vol > np.median(vol)).astype(int)
        if high_vol.sum() > 0 and high_vol.sum() < n:
            try:
                auc_vol = float(roc_auc_score(y, high_vol))
            except Exception:
                auc_vol = 0.5
        else:
            auc_vol = 0.5
    else:
        auc_vol = 0.5

    # Trend-only rule (spot_return_5 sign)
    if "spot_return_5" in df.columns:
        tr = (pd.to_numeric(df["spot_return_5"], errors="coerce").fillna(0).to_numpy() > 0).astype(int)
        if tr.sum() > 0 and tr.sum() < n:
            try:
                auc_tr = float(roc_auc_score(y, tr))
            except Exception:
                auc_tr = 0.5
        else:
            auc_tr = 0.5
    else:
        auc_tr = 0.5

    return {
        "always_positive_pf": base_pf,
        "always_positive_auc": auc_always_pos,
        "random_classifier_auc": auc_random,
        "prev_candle_auc": auc_prev,
        "volatility_only_auc": auc_vol,
        "trend_only_auc": auc_tr,
    }


def _signal_to_noise(walk_forward_auc: float, label_entropy: float) -> float:
    """SNR proxy: how much the walk-forward AUC exceeds random (0.5) relative to label entropy."""
    if label_entropy <= 0:
        return 0.0
    return max((walk_forward_auc - 0.5), 0) / label_entropy


def _achievable_upper_bound(walk_forward_auc: float) -> float:
    """Heuristic: max possible PF given a perfectly calibrated model with this AUC.
    AUC of 0.5 -> 1.0 (no edge); AUC of 0.7 -> ~1.4; AUC of 0.8 -> ~1.8."""
    if walk_forward_auc <= 0.5:
        return 1.0
    excess = walk_forward_auc - 0.5
    return round(1.0 + 4.0 * excess, 3)


def build() -> Dict[str, Any]:
    art = LATEST_ARTIFACT
    label_skip = _safe_load(art / "label_skip_report.json")
    skipped = [x["label_name"] for x in label_skip.get("labels_skipped", [])]

    print("Loading dataset (200k sample for speed)...")
    df = pd.read_csv(DATA, low_memory=False, nrows=200000)
    print(f"Dataset shape: {df.shape}")

    forbidden = {
        "future_close",
        "gross_forward_return",
        "net_forward_return",
        "expected_return_after_cost",
        "return_to_cost_ratio",
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

    per_label: Dict[str, Any] = {}
    for label in USABLE_LABELS:
        print(f"Auditing {label}...")
        if label in skipped:
            per_label[label] = {
                "label": label,
                "skipped": True,
                "skip_reason": "degenerate_constant(unique=1) in dataset",
            }
            continue
        if label not in df.columns:
            per_label[label] = {
                "label": label,
                "skipped": True,
                "skip_reason": "label not in dataset columns",
            }
            continue
        m = _per_label_metrics(df, label, feature_cols)
        b = _baseline_metrics(df, label)
        m["baselines"] = b
        m["signal_to_noise"] = _signal_to_noise(m.get("walk_forward_auc", 0.5), m.get("label_entropy", 0.0))
        m["achievable_upper_bound_pf"] = _achievable_upper_bound(m.get("walk_forward_auc", 0.5))
        per_label[label] = m

    # Build leaderboard
    leaderboard = []
    for label, m in per_label.items():
        if m.get("skipped"):
            continue
        wf_auc = m.get("walk_forward_auc", 0.5)
        wf_std = m.get("walk_forward_auc_std", 0.0)
        month_std = m.get("month_std", 0.0)
        stability_score = max(0.0, 1.0 - wf_std) * max(0.0, 1.0 - month_std)
        drift_score = m.get("month_range", 0.0) + m.get("vol_range", 0.0) + m.get("expiry_range", 0.0)
        predictability = max(0.0, (wf_auc - 0.5) * 2)  # 0..1
        composite = (0.5 * predictability) + (0.3 * stability_score) - (0.2 * drift_score)
        leaderboard.append(
            {
                "label": label,
                "walk_forward_auc": wf_auc,
                "walk_forward_auc_std": wf_std,
                "walk_forward_pf_estimate": m.get("achievable_upper_bound_pf"),
                "label_entropy": m.get("label_entropy", 0.0),
                "positive_pct": m.get("positive_pct", 0.0),
                "stability_score": stability_score,
                "drift_score": drift_score,
                "predictability": predictability,
                "signal_to_noise": m.get("signal_to_noise", 0.0),
                "composite_score": composite,
                "sharpe_proxy": (m.get("mi_max", 0.0)) * (1.0 - wf_std),
            }
        )
    leaderboard.sort(key=lambda x: -x["composite_score"])

    return {
        "report_type": "label_predictability_audit",
        "artifact_dir": str(art),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "per_label": per_label,
        "leaderboard": leaderboard,
        "skipped_labels": skipped,
    }


def write_reports() -> None:
    payload = build()
    REPORTS.mkdir(parents=True, exist_ok=True)

    jpath = REPORTS / f"label_predictability_audit_{TS}.json"
    jpath.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    md = [
        "# Label Predictability Audit",
        "",
        f"- Artifact dir: `{payload['artifact_dir']}`",
        f"- Generated: `{payload['timestamp_utc']}`",
        f"- Labels analysed: **{sum(1 for v in payload['per_label'].values() if not v.get('skipped'))}**",
        f"- Labels skipped: **{', '.join(payload['skipped_labels'])}**",
        "",
        "## Per-label summary",
        "",
        "| Label | Pos % | Entropy | WF AUC | AUC std | SNR | Upper-bound PF | Drift score | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, m in payload["per_label"].items():
        if m.get("skipped"):
            md.append(f"| `{label}` | SKIPPED | - | - | - | - | - | - | {m.get('skip_reason', '')} |")
            continue
        wf_auc = m.get("walk_forward_auc", 0.5)
        snr = m.get("signal_to_noise", 0.0)
        upper = m.get("achievable_upper_bound_pf", 1.0)
        vol_range = m.get("vol_range", 0.0)
        month_range = m.get("month_range", 0.0)
        expiry_range = m.get("expiry_range", 0.0)
        drift = month_range + vol_range + expiry_range
        if wf_auc < 0.55 and snr < 0.05:
            verdict = "DELETE (no signal)"
        elif wf_auc < 0.60:
            verdict = "WEAK (keep only for ablation)"
        elif wf_auc < 0.70:
            verdict = "MODERATE (worth modelling)"
        else:
            verdict = "STRONG"
        md.append(
            f"| `{label}` | {m['positive_pct']:.3f} | {m['label_entropy']:.4f} | {wf_auc:.4f} | "
            f"{m.get('walk_forward_auc_std', 0):.4f} | {snr:.4f} | {upper} | {drift:.3f} | {verdict} |"
        )
    md.append("")
    md.append("## Baselines comparison (always-positive, random, prev-candle, volatility, trend)")
    md.append("")
    md.append("| Label | Always-pos PF | Random AUC | Prev candle AUC | Vol-only AUC | Trend-only AUC |")
    md.append("|---|---|---|---|---|---|")
    for label, m in payload["per_label"].items():
        if m.get("skipped"):
            continue
        b = m.get("baselines", {})
        md.append(
            f"| `{label}` | {b.get('always_positive_pf', 0):.3f} | "
            f"{b.get('random_classifier_auc', 0.5):.3f} | "
            f"{b.get('prev_candle_auc', 0.5):.3f} | "
            f"{b.get('volatility_only_auc', 0.5):.3f} | "
            f"{b.get('trend_only_auc', 0.5):.3f} |"
        )
    md.append("")
    md.append("## Per-label per-month positive rate")
    md.append("")
    for label, m in payload["per_label"].items():
        if m.get("skipped"):
            continue
        md.append(f"### `{label}`")
        pm = m.get("per_month_positive", {})
        for month, val in sorted(pm.items()):
            md.append(f"- {month}: {val:.4f}")
        md.append("")
    md.append("## Per-label volatility regime")
    md.append("")
    md.append("| Label | Low vol | Mid vol | High vol | Range |")
    md.append("|---|---|---|---|---|")
    for label, m in payload["per_label"].items():
        if m.get("skipped"):
            continue
        pv = m.get("per_vol_regime", {})
        rng = m.get("vol_range", 0.0)
        md.append(
            f"| `{label}` | {pv.get('low', 0):.4f} | {pv.get('mid', 0):.4f} | "
            f"{pv.get('high', 0):.4f} | {rng:.4f} |"
        )
    md.append("")
    md.append("## Critical question — Can any label support a deployable ML edge after costs?")
    md.append("")
    # Find best composite
    best = payload["leaderboard"][0] if payload["leaderboard"] else None
    worst = payload["leaderboard"][-1] if payload["leaderboard"] else None
    if best:
        md.append(
            f"- **Best label**: `{best['label']}` — composite score {best['composite_score']:.4f}, "
            f"WF AUC {best['walk_forward_auc']:.4f}, SNR {best['signal_to_noise']:.4f}, "
            f"upper-bound PF ~{best['walk_forward_pf_estimate']:.3f}"
        )
    if worst:
        md.append(
            f"- **Worst label**: `{worst['label']}` — composite score {worst['composite_score']:.4f}, "
            f"WF AUC {worst['walk_forward_auc']:.4f}"
        )
    # Find most stable
    most_stable = max(payload["leaderboard"], key=lambda x: x["stability_score"]) if payload["leaderboard"] else None
    if most_stable:
        md.append(
            f"- **Most stable label**: `{most_stable['label']}` — stability score {most_stable['stability_score']:.4f}"
        )
    # Find most predictive
    most_pred = max(payload["leaderboard"], key=lambda x: x["walk_forward_auc"]) if payload["leaderboard"] else None
    if most_pred:
        md.append(
            f"- **Most predictive label**: `{most_pred['label']}` — WF AUC {most_pred['walk_forward_auc']:.4f}"
        )
    md.append("")
    md.append("### Labels that should be deleted (no edge above random + noise)")
    for entry in payload["leaderboard"]:
        if entry["walk_forward_auc"] < 0.55 and entry["signal_to_noise"] < 0.05:
            md.append(f"- `{entry['label']}`: WF AUC {entry['walk_forward_auc']:.4f}, SNR {entry['signal_to_noise']:.4f}")
    md.append("")
    md.append("### Estimated maximum achievable edge")
    if best:
        md.append(
            f"- Best achievable PF: ~{best['walk_forward_pf_estimate']:.3f} (a perfectly calibrated model would still be marginal)."
        )
        if best["walk_forward_pf_estimate"] < 1.15:
            md.append(
                "- The upper bound is **below 1.15 PF**, which is below the strict gate `cost_1.5x_pf >= 1.0` for promotion. "
                "After transaction costs the realistic edge is approximately zero or negative."
            )
    md.append("")
    md.append("### Whether current labels justify continuing ML development")
    n_meaningful = sum(1 for e in payload["leaderboard"] if e["walk_forward_auc"] >= 0.55)
    if n_meaningful == 0:
        md.append("- **NO**: No label clears the AUC>0.55 + non-zero SNR bar. The dataset does not support a deployable edge with these targets.")
    elif n_meaningful < 2:
        md.append(
            f"- **MARGINAL**: Only {n_meaningful} label clears the bar. Continuing development is justified only if we redesign the targets (alternative target definitions, triple-barrier labelling, profit-aware horizons)."
        )
    else:
        md.append(f"- **YES (with redesign)**: {n_meaningful} labels are above noise. Further work on label engineering and feature selection is justified.")
    mpath = REPORTS / f"label_predictability_audit_{TS}.md"
    mpath.write_text("\n".join(md), encoding="utf-8")

    # Standalone label leaderboard
    lb_md = [
        "# Label Leaderboard",
        "",
        f"- Artifact dir: `{payload['artifact_dir']}`",
        f"- Generated: `{payload['timestamp_utc']}`",
        "",
        "## Ranked by composite score (predictability + stability − drift)",
        "",
        "| Rank | Label | WF AUC | AUC std | Pos % | Stability | Drift | Predictability | SNR | Composite | Upper-bound PF | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, e in enumerate(payload["leaderboard"], 1):
        if e["walk_forward_auc"] < 0.55 and e["signal_to_noise"] < 0.05:
            verdict = "DELETE"
        elif e["walk_forward_auc"] < 0.60:
            verdict = "WEAK"
        elif e["walk_forward_auc"] < 0.70:
            verdict = "MODERATE"
        else:
            verdict = "STRONG"
        lb_md.append(
            f"| {i} | `{e['label']}` | {e['walk_forward_auc']:.4f} | "
            f"{e['walk_forward_auc_std']:.4f} | {e['positive_pct']:.3f} | "
            f"{e['stability_score']:.4f} | {e['drift_score']:.3f} | "
            f"{e['predictability']:.4f} | {e['signal_to_noise']:.4f} | "
            f"{e['composite_score']:.4f} | {e['walk_forward_pf_estimate']:.3f} | {verdict} |"
        )
    lb_mpath = REPORTS / f"label_leaderboard_{TS}.md"
    lb_mpath.write_text("\n".join(lb_md), encoding="utf-8")

    lb_jpath = REPORTS / f"label_leaderboard_{TS}.json"
    lb_jpath.write_text(json.dumps({"leaderboard": payload["leaderboard"]}, indent=2, default=str), encoding="utf-8")

    print(f"WROTE: {jpath}")
    print(f"WROTE: {mpath}")
    print(f"WROTE: {lb_jpath}")
    print(f"WROTE: {lb_mpath}")


if __name__ == "__main__":
    write_reports()
