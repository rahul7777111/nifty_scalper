"""Phase 9: target engineering study.

Audits existing labels and evaluates alternative target candidates.

For each candidate target:
- class balance
- entropy
- drift score (per-month positive rate std)
- mutual information with top features
- walk-forward AUC (5 splits, logistic regression)
- expected trade frequency
- estimated upper-bound PF
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
TS = "20260607_073000"


def _safe_load(p: Path) -> Dict[str, Any]:
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


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


def _per_target_eval(df: pd.DataFrame, target: pd.Series, target_name: str, feature_cols: List[str]) -> Dict[str, Any]:
    y = target.fillna(0).astype(int).to_numpy()
    n = len(y)
    pos = int(y.sum())
    pos_pct = pos / max(n, 1)

    if pos_pct == 0 or pos_pct == 1:
        return {
            "target": target_name,
            "n_rows": n,
            "positive_pct": pos_pct,
            "verdict": "DEGENERATE",
        }

    sub_n = min(50000, n)
    rng = np.random.RandomState(42)
    idx = rng.choice(n, size=sub_n, replace=False)
    Xs = np.column_stack(
        [pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy()[idx] for c in feature_cols]
    )
    Xs_bin = np.column_stack([_bin_feature(Xs[:, j]) for j in range(Xs.shape[1])])
    ys = y[idx]
    mi = mutual_info_classif(Xs_bin, ys, random_state=42, n_jobs=1)
    mi_mean = float(np.mean(mi))
    mi_max = float(np.max(mi))

    # per-month positive rate
    df_local = df.copy()
    if "timestamp" in df_local.columns:
        df_local["_month"] = pd.to_datetime(df_local["timestamp"], errors="coerce", utc=True).dt.to_period("M").astype(str)
    else:
        df_local["_month"] = "all"
    per_month = df_local.assign(_y=target.fillna(0).astype(int)).groupby("_month")["_y"].mean().to_dict()
    month_vals = [v for v in per_month.values() if v is not None and not (isinstance(v, float) and np.isnan(v))]
    month_std = float(np.std(month_vals)) if month_vals else 0.0
    month_range = (max(month_vals) - min(month_vals)) if month_vals else 0.0

    # walk-forward AUC
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

    upper_pf = round(1.0 + 4.0 * max(0.0, mean_auc - 0.5), 3) if mean_auc > 0.5 else 1.0

    return {
        "target": target_name,
        "n_rows": n,
        "positive_pct": pos_pct,
        "label_entropy": _entropy(pos_pct),
        "mi_mean": mi_mean,
        "mi_max": mi_max,
        "month_std": month_std,
        "month_range": month_range,
        "walk_forward_auc": mean_auc,
        "walk_forward_auc_std": auc_std,
        "expected_trade_count_200k": int(pos_pct * 200000),
        "estimated_upper_bound_pf": upper_pf,
    }


def _existing_target_audit(df: pd.DataFrame, feature_cols: List[str]) -> List[Dict[str, Any]]:
    existing = [
        "profitable_trade_label",
        "avoid_trade_label",
        "strong_profitable_trade_label",
        "cost_survivor_label",
        "strong_profitable_trade_label_v2",
        "cost_survivor_label_v2",
        "weak_trade_label",
        "no_trade_label",
    ]
    out = []
    for label in existing:
        if label not in df.columns:
            continue
        y = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int)
        m = _per_target_eval(df, y, label, feature_cols)
        m["kind"] = "existing"
        m["formula"] = f"see scripts/option_chain_pipeline_lib.py for {label}"
        out.append(m)
    return out


def _directional_targets(df: pd.DataFrame, feature_cols: List[str]) -> List[Dict[str, Any]]:
    out = []
    for n_bars, col in [(1, "spot_return_1"), (3, "spot_return_3"), (5, "spot_return_5")]:
        if col not in df.columns:
            continue
        r = pd.to_numeric(df[col], errors="coerce").fillna(0)
        target = (r > 0).astype(int)
        m = _per_target_eval(df, target, f"directional_{n_bars}candle_positive", feature_cols)
        m["kind"] = "directional"
        m["formula"] = f"({col} > 0) -> 1"
        out.append(m)
    return out


def _cost_aware_targets(df: pd.DataFrame, feature_cols: List[str]) -> List[Dict[str, Any]]:
    out = []
    if "net_forward_return" not in df.columns:
        return out
    nret = pd.to_numeric(df["net_forward_return"], errors="coerce").fillna(0)
    # Cost hurdle: 0.5% (typical NIFTY options round-trip)
    for hurdle, name in [(0.005, "net_gt_0_5pct"), (0.01, "net_gt_1pct"), (0.015, "net_gt_1_5pct")]:
        target = (nret > hurdle).astype(int)
        m = _per_target_eval(df, target, f"cost_aware_{name}", feature_cols)
        m["kind"] = "cost_aware"
        m["formula"] = f"net_forward_return > {hurdle:.3f}"
        out.append(m)
    return out


def _ranking_targets(df: pd.DataFrame, feature_cols: List[str]) -> List[Dict[str, Any]]:
    out = []
    if "net_forward_return" not in df.columns:
        return out
    nret = pd.to_numeric(df["net_forward_return"], errors="coerce").fillna(0)
    p10 = nret.quantile(0.10)
    p20 = nret.quantile(0.20)
    p80 = nret.quantile(0.80)
    p90 = nret.quantile(0.90)
    for name, tgt in [
        ("top_10pct_returns", nret > p90),
        ("top_20pct_returns", nret > p80),
        ("bottom_10pct_returns", nret < p10),
        ("bottom_20pct_returns", nret < p20),
    ]:
        m = _per_target_eval(df, tgt.astype(int), name, feature_cols)
        m["kind"] = "ranking"
        m["formula"] = f"net_forward_return threshold = {p10 if '10pct' in name and 'bottom' in name else (p20 if 'bottom' in name else (p80 if '20pct' in name else p90))}"
        out.append(m)
    return out


def _regime_targets(df: pd.DataFrame, feature_cols: List[str]) -> List[Dict[str, Any]]:
    out = []
    if "net_forward_return" not in df.columns:
        return out
    if "realized_vol_30" in df.columns:
        vol = pd.to_numeric(df["realized_vol_30"], errors="coerce").fillna(0)
        nret = pd.to_numeric(df["net_forward_return"], errors="coerce").fillna(0)
        for name, mask in [
            ("high_vol_success", vol > vol.quantile(0.75)),
            ("low_vol_success", vol < vol.quantile(0.25)),
        ]:
            tgt = ((nret > 0) & mask).astype(int)
            m = _per_target_eval(df, tgt, name, feature_cols)
            m["kind"] = "regime"
            m["formula"] = f"net_forward_return > 0 AND {name.split('_')[0]}_vol"
            out.append(m)
    return out


def _option_targets(df: pd.DataFrame, feature_cols: List[str]) -> List[Dict[str, Any]]:
    out = []
    # IV expansion: requires 'iv' column or proxy
    iv_proxy = None
    for c in ["iv", "implied_vol", "bs_iv", "bs_iv_atm"]:
        if c in df.columns:
            iv_proxy = c
            break
    if iv_proxy and "net_forward_return" in df.columns:
        iv = pd.to_numeric(df[iv_proxy], errors="coerce").fillna(0)
        nret = pd.to_numeric(df["net_forward_return"], errors="coerce").fillna(0)
        iv_p75 = iv.quantile(0.75)
        tgt = ((nret > 0) & (iv > iv_p75)).astype(int)
        m = _per_target_eval(df, tgt, "iv_expansion_success", feature_cols)
        m["kind"] = "option_specific"
        m["formula"] = f"net_forward_return > 0 AND {iv_proxy} > p75"
        out.append(m)
    return out


def _leaderboard(targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for t in targets:
        if t.get("verdict") == "DEGENERATE":
            continue
        wf_auc = t.get("walk_forward_auc", 0.5)
        wf_std = t.get("walk_forward_auc_std", 0.0)
        month_std = t.get("month_std", 0.0)
        predictability = max(0.0, (wf_auc - 0.5) * 2)
        stability = max(0.0, 1.0 - wf_std) * max(0.0, 1.0 - month_std)
        drift = t.get("month_range", 0.0)
        composite = (0.5 * predictability) + (0.3 * stability) - (0.2 * drift)
        rows.append(
            {
                "target": t["target"],
                "kind": t.get("kind", ""),
                "walk_forward_auc": wf_auc,
                "walk_forward_auc_std": wf_std,
                "positive_pct": t.get("positive_pct", 0.0),
                "stability_score": stability,
                "drift_score": drift,
                "predictability": predictability,
                "composite_score": composite,
                "estimated_upper_bound_pf": t.get("estimated_upper_bound_pf", 1.0),
            }
        )
    rows.sort(key=lambda x: -x["composite_score"])
    return rows


def build() -> Dict[str, Any]:
    print("Loading dataset sample (200k rows)...")
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

    print(f"Feature cols: {len(feature_cols)}")

    existing = _existing_target_audit(df, feature_cols)
    directional = _directional_targets(df, feature_cols)
    cost_aware = _cost_aware_targets(df, feature_cols)
    ranking = _ranking_targets(df, feature_cols)
    regime = _regime_targets(df, feature_cols)
    option_spec = _option_targets(df, feature_cols)

    all_targets = existing + directional + cost_aware + ranking + regime + option_spec
    leaderboard = _leaderboard(all_targets)

    return {
        "report_type": "target_engineering_study",
        "artifact_dir": str(LATEST_ARTIFACT),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_existing": len(existing),
        "n_alternative": len(directional) + len(cost_aware) + len(ranking) + len(regime) + len(option_spec),
        "existing": existing,
        "directional": directional,
        "cost_aware": cost_aware,
        "ranking": ranking,
        "regime": regime,
        "option_specific": option_spec,
        "leaderboard": leaderboard,
    }


def write_reports() -> None:
    payload = build()
    REPORTS.mkdir(parents=True, exist_ok=True)

    jpath = REPORTS / f"target_engineering_study_{TS}.json"
    jpath.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    md = [
        "# Target Engineering Study",
        "",
        f"- Generated: `{payload['timestamp_utc']}`",
        f"- Existing labels audited: **{payload['n_existing']}**",
        f"- Alternative candidates evaluated: **{payload['n_alternative']}**",
        "",
        "## Leaderboard (ranked by composite score)",
        "",
        "| Rank | Target | Kind | WF AUC | Pos % | Stability | Drift | Upper-bound PF | Composite |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, e in enumerate(payload["leaderboard"], 1):
        md.append(
            f"| {i} | `{e['target']}` | {e['kind']} | {e['walk_forward_auc']:.4f} | "
            f"{e['positive_pct']:.3f} | {e['stability_score']:.4f} | "
            f"{e['drift_score']:.3f} | {e['estimated_upper_bound_pf']:.3f} | "
            f"{e['composite_score']:.4f} |"
        )
    md.append("")
    md.append("## Existing labels (formula: see scripts/option_chain_pipeline_lib.py)")
    md.append("")
    md.append("| Target | Pos % | Entropy | WF AUC | AUC std | Drift (month std) | Upper-bound PF |")
    md.append("|---|---|---|---|---|---|---|")
    for t in payload["existing"]:
        md.append(
            f"| `{t['target']}` | {t.get('positive_pct', 0):.3f} | "
            f"{t.get('label_entropy', 0):.4f} | {t.get('walk_forward_auc', 0.5):.4f} | "
            f"{t.get('walk_forward_auc_std', 0):.4f} | {t.get('month_std', 0):.4f} | "
            f"{t.get('estimated_upper_bound_pf', 1.0):.3f} |"
        )
    md.append("")
    md.append("## Directional targets")
    md.append("")
    for t in payload["directional"]:
        md.append(
            f"- `{t['target']}`: pos%={t.get('positive_pct', 0):.3f}, WF AUC={t.get('walk_forward_auc', 0.5):.4f}, formula={t.get('formula', '')}"
        )
    md.append("")
    md.append("## Cost-aware targets")
    md.append("")
    for t in payload["cost_aware"]:
        md.append(
            f"- `{t['target']}`: pos%={t.get('positive_pct', 0):.3f}, WF AUC={t.get('walk_forward_auc', 0.5):.4f}, formula={t.get('formula', '')}"
        )
    md.append("")
    md.append("## Ranking targets")
    md.append("")
    for t in payload["ranking"]:
        md.append(
            f"- `{t['target']}`: pos%={t.get('positive_pct', 0):.3f}, WF AUC={t.get('walk_forward_auc', 0.5):.4f}, formula={t.get('formula', '')}"
        )
    md.append("")
    md.append("## Regime targets")
    md.append("")
    for t in payload["regime"]:
        md.append(
            f"- `{t['target']}`: pos%={t.get('positive_pct', 0):.3f}, WF AUC={t.get('walk_forward_auc', 0.5):.4f}, formula={t.get('formula', '')}"
        )
    md.append("")
    md.append("## Option-specific targets")
    md.append("")
    for t in payload["option_specific"]:
        md.append(
            f"- `{t['target']}`: pos%={t.get('positive_pct', 0):.3f}, WF AUC={t.get('walk_forward_auc', 0.5):.4f}, formula={t.get('formula', '')}"
        )
    md.append("")
    md.append("## Recommendations")
    md.append("")
    lb = payload["leaderboard"]
    if lb:
        best = lb[0]
        worst = lb[-1]
        md.append(f"- **Best target**: `{best['target']}` (composite {best['composite_score']:.4f}, WF AUC {best['walk_forward_auc']:.4f})")
        md.append(f"- **Worst target**: `{worst['target']}` (composite {worst['composite_score']:.4f})")
        # Most stable = highest stability
        most_stable = max(lb, key=lambda x: x["stability_score"])
        md.append(f"- **Most stable target**: `{most_stable['target']}` (stability {most_stable['stability_score']:.4f})")
        # Highest edge = highest upper-bound PF
        highest_edge = max(lb, key=lambda x: x["estimated_upper_bound_pf"])
        md.append(
            f"- **Highest edge target**: `{highest_edge['target']}` (upper-bound PF {highest_edge['estimated_upper_bound_pf']:.3f})"
        )
    # Existing labels
    existing_composite = []
    for e in lb:
        if e["kind"] == "existing":
            existing_composite.append(e)
    if existing_composite:
        best_existing = existing_composite[0]
        md.append("")
        md.append("### Whether current labels should be retired")
        alt_max = max((e for e in lb if e["kind"] != "existing"), key=lambda x: x["composite_score"], default=None)
        if alt_max and alt_max["composite_score"] > best_existing["composite_score"] + 0.05:
            md.append(
                f"- **REPLACE**: Best alternative (`{alt_max['target']}`, composite {alt_max['composite_score']:.4f}) "
                f"dominates best existing (`{best_existing['target']}`, composite {best_existing['composite_score']:.4f}) by >0.05."
            )
        elif alt_max and alt_max["composite_score"] > best_existing["composite_score"]:
            md.append(
                f"- **MODIFY**: Alternative is slightly better; consider blending `{best_existing['target']}` with `{alt_max['target']}`."
            )
        else:
            md.append(
                f"- **KEEP existing**: `{best_existing['target']}` is competitive with alternatives; no replacement needed at this time."
            )
        # Estimated achievable PF
        max_pf = max(e["estimated_upper_bound_pf"] for e in lb)
        md.append("")
        md.append(f"### Estimated maximum achievable PF (any target): {max_pf:.3f}")
        # Production recommendation
        prod_target = lb[0]
        md.append(
            f"### Recommended production target: `{prod_target['target']}` (composite {prod_target['composite_score']:.4f}, "
            f"WF AUC {prod_target['walk_forward_auc']:.4f}, upper-bound PF {prod_target['estimated_upper_bound_pf']:.3f})"
        )
    mpath = REPORTS / f"target_engineering_study_{TS}.md"
    mpath.write_text("\n".join(md), encoding="utf-8")

    # Standalone leaderboard
    lb_jpath = REPORTS / f"target_leaderboard_{TS}.json"
    lb_jpath.write_text(json.dumps({"leaderboard": payload["leaderboard"]}, indent=2, default=str), encoding="utf-8")

    lb_md = [
        "# Target Leaderboard",
        "",
        f"- Generated: `{payload['timestamp_utc']}`",
        "",
        "| Rank | Target | Kind | WF AUC | Pos % | Stability | Drift | Upper-bound PF | Composite |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, e in enumerate(payload["leaderboard"], 1):
        lb_md.append(
            f"| {i} | `{e['target']}` | {e['kind']} | {e['walk_forward_auc']:.4f} | "
            f"{e['positive_pct']:.3f} | {e['stability_score']:.4f} | {e['drift_score']:.3f} | "
            f"{e['estimated_upper_bound_pf']:.3f} | {e['composite_score']:.4f} |"
        )
    lb_mpath = REPORTS / f"target_leaderboard_{TS}.md"
    lb_mpath.write_text("\n".join(lb_md), encoding="utf-8")

    print(f"WROTE: {jpath}")
    print(f"WROTE: {mpath}")
    print(f"WROTE: {lb_jpath}")
    print(f"WROTE: {lb_mpath}")


if __name__ == "__main__":
    write_reports()
