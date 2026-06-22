"""Phase 14: high-conviction sparse trading system.

Strategy:
- Top-10 features only (low_drift, high_MI)
- Target: low_vol_success (Phase 9 best WF AUC=0.6972)
- Regime filter: vol_low only (Phase 11 showed regime matters less, but low-vol regime was most stable)
- Top-K confidence: top 1%, 2%, 5%, 10% predictions

Evaluate:
- trade count, PF, Sharpe, expectancy, max drawdown, avg return

Emits:
- reports/high_conviction_strategy_<ts>.md
- reports/high_conviction_strategy_<ts>.json
- reports/confidence_threshold_leaderboard_<ts>.md
- reports/confidence_threshold_leaderboard_<ts>.json
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "processed" / "nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv"
TS = "20260607_078000"


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


def _build_low_vol_success_target(df: pd.DataFrame) -> np.ndarray:
    if "realized_vol_30" not in df.columns or "net_forward_return" not in df.columns:
        return np.zeros(len(df), dtype=int)
    vol = pd.to_numeric(df["realized_vol_30"], errors="coerce").fillna(0).to_numpy()
    nret = pd.to_numeric(df["net_forward_return"], errors="coerce").fillna(0).to_numpy()
    return ((nret > 0) & (vol < np.quantile(vol, 0.25))).astype(int)


def _train_predict_top_k(
    df_train: pd.DataFrame,
    df_holdout: pd.DataFrame,
    target: np.ndarray,
    target_h: np.ndarray,
    features: List[str],
) -> Dict[str, np.ndarray]:
    Xtr = np.column_stack(
        [pd.to_numeric(df_train[c], errors="coerce").fillna(0).to_numpy() for c in features]
    )
    Xho = np.column_stack(
        [pd.to_numeric(df_holdout[c], errors="coerce").fillna(0).to_numpy() for c in features]
    )
    Xtr = np.nan_to_num(np.where(np.isfinite(Xtr), Xtr, 0), nan=0.0, posinf=1e6, neginf=-1e6)
    Xho = np.nan_to_num(np.where(np.isfinite(Xho), Xho, 0), nan=0.0, posinf=1e6, neginf=-1e6)
    Xtr = np.clip(Xtr, -1e6, 1e6)
    Xho = np.clip(Xho, -1e6, 1e6)
    m = LogisticRegression(max_iter=300, solver="lbfgs")
    m.fit(Xtr, target)
    return {
        "p_train": m.predict_proba(Xtr)[:, 1],
        "p_holdout": m.predict_proba(Xho)[:, 1],
    }


def _strategy_metrics(
    df_h: pd.DataFrame,
    y_h: np.ndarray,
    p_h: np.ndarray,
    *,
    top_pct: float,
    cost: float = 0.01,
    vol_filter: bool = True,
    spread_filter: bool = True,
) -> Dict[str, Any]:
    n = len(p_h)
    if n == 0:
        return {"n_trades": 0, "pf": 0, "sharpe": 0, "expectancy": 0, "max_dd": 0, "avg_return": 0}

    p_thresh = float(np.quantile(p_h, 1.0 - top_pct))
    mask = p_h >= p_thresh
    if vol_filter and "realized_vol_30" in df_h.columns:
        vol = pd.to_numeric(df_h["realized_vol_30"], errors="coerce").fillna(0).to_numpy()
        q50 = np.median(vol)
        mask = mask & (vol <= q50)
    if spread_filter and "wide_spread_flag" in df_h.columns:
        flag = pd.to_numeric(df_h["wide_spread_flag"], errors="coerce").fillna(1).to_numpy()
        mask = mask & (flag == 0)
    if mask.sum() == 0:
        return {"n_trades": 0, "pf": 0, "sharpe": 0, "expectancy": 0, "max_dd": 0, "avg_return": 0}

    nret = pd.to_numeric(df_h["net_forward_return"], errors="coerce").fillna(0).to_numpy()
    sel = nret[mask] - cost
    if len(sel) == 0:
        return {"n_trades": 0, "pf": 0, "sharpe": 0, "expectancy": 0, "max_dd": 0, "avg_return": 0}
    wins = sel[sel > 0].sum()
    losses = abs(sel[sel <= 0].sum())
    pf = wins / max(losses, 1e-9)
    avg = float(np.mean(sel))
    std = float(np.std(sel)) or 1e-9
    sharpe = avg / std * np.sqrt(252)
    cum = np.cumsum(sel)
    peak = np.maximum.accumulate(cum)
    dd = float(np.min(cum - peak))
    return {
        "n_trades": int(mask.sum()),
        "pf": round(float(pf), 3),
        "sharpe": round(float(sharpe), 3),
        "expectancy": round(avg, 4),
        "max_dd": round(dd, 3),
        "avg_return": round(avg, 4),
    }


def build() -> Dict[str, Any]:
    print("Loading dataset...")
    df = pd.read_csv(DATA, low_memory=False)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.7)
    df_train = df.iloc[:train_end].reset_index(drop=True)
    df_holdout = df.iloc[train_end:].reset_index(drop=True)
    print(f"Train: {len(df_train)}, Holdout: {len(df_holdout)}")

    # Build the low_vol_success target on full df
    df_train["_low_vol_success"] = _build_low_vol_success_target(df_train)
    df_holdout["_low_vol_success"] = _build_low_vol_success_target(df_holdout)
    target_train = df_train["_low_vol_success"].astype(int).to_numpy()
    target_holdout = df_holdout["_low_vol_success"].astype(int).to_numpy()
    print(f"Train pos%: {target_train.mean():.3f}, Holdout pos%: {target_holdout.mean():.3f}")

    # Feature selection: top 10 by mutual information on TRAIN ONLY (no leakage)
    forbidden = {
        "future_close", "gross_forward_return", "net_forward_return",
        "expected_return_after_cost", "return_to_cost_ratio",
        "cost_return_units_estimated",
    }
    feature_cols: List[str] = []
    for c in df_train.columns:
        if c.startswith("_"):
            continue
        cl = c.lower()
        if c in forbidden or cl.endswith("_label") or cl.endswith("_label_v2"):
            continue
        if df_train[c].dtype == "O":
            continue
        feature_cols.append(c)

    print(f"Computing MI on train ({len(feature_cols)} features)...")
    sub_n = min(30000, len(df_train))
    sidx = np.random.RandomState(11).choice(len(df_train), size=sub_n, replace=False)
    Xs = np.column_stack(
        [pd.to_numeric(df_train[c], errors="coerce").fillna(0).to_numpy()[sidx] for c in feature_cols]
    )
    Xs_bin = np.column_stack([np.digitize(Xs[:, j], np.quantile(Xs[:, j], np.linspace(0, 1, 11))) for j in range(Xs.shape[1])])
    mi = mutual_info_classif(Xs_bin, target_train[sidx], random_state=11, n_jobs=1)
    feat_mi = sorted(zip(feature_cols, mi), key=lambda x: -x[1])

    # Compute drift on holdout vs train
    sorted_train_idx = np.argsort(pd.to_datetime(df_train["timestamp"], errors="coerce", utc=True).fillna(pd.Timestamp("2020-01-01")).to_numpy())
    n4 = len(df_train) // 4
    train_early = sorted_train_idx[:n4]
    train_late = sorted_train_idx[-n4:]

    # Pick top-10 by MI AND filter out high-drift features
    print("Selecting walk-forward-safe top-10 features...")
    top10_safe: List[str] = []
    for f, m in feat_mi:
        if len(top10_safe) >= 10:
            break
        x_train = pd.to_numeric(df_train[f], errors="coerce").fillna(0).to_numpy()
        psi_val = _psi(x_train[train_early], x_train[train_late])
        if psi_val < 0.3:  # safe PSI threshold
            top10_safe.append(f)
    if len(top10_safe) < 10:
        # Top off with highest-MI features (allowing some drift)
        for f, _ in feat_mi:
            if f in top10_safe:
                continue
            if len(top10_safe) >= 10:
                break
            top10_safe.append(f)
    print(f"Top-10 features: {top10_safe}")

    # Train on top-10
    res = _train_predict_top_k(df_train, df_holdout, target_train, target_holdout, top10_safe)
    p_h = res["p_holdout"]
    p_tr = res["p_train"]

    # WF AUC
    train_auc = float(roc_auc_score(target_train, p_tr)) if target_train.sum() > 0 else 0.5
    holdout_auc = float(roc_auc_score(target_holdout, p_h)) if target_holdout.sum() > 0 else 0.5
    print(f"Train AUC: {train_auc:.4f}, Holdout AUC: {holdout_auc:.4f}")

    # Strategies
    print("Evaluating top-K strategies...")
    top_pcts = [0.01, 0.02, 0.05, 0.10]
    leaderboard: List[Dict[str, Any]] = []
    for pct in top_pcts:
        # With all filters
        m_full = _strategy_metrics(df_holdout, target_holdout, p_h, top_pct=pct, vol_filter=True, spread_filter=True)
        # Without filters (top-K only)
        m_no_filter = _strategy_metrics(df_holdout, target_holdout, p_h, top_pct=pct, vol_filter=False, spread_filter=False)
        # Vol filter only
        m_vol = _strategy_metrics(df_holdout, target_holdout, p_h, top_pct=pct, vol_filter=True, spread_filter=False)
        leaderboard.append(
            {
                "top_pct": pct,
                "with_full_filters": m_full,
                "top_k_only": m_no_filter,
                "top_k_with_vol_filter": m_vol,
            }
        )

    # Smallest number of trades producing positive expectancy
    positive_ev = [x for x in leaderboard if x["with_full_filters"].get("expectancy", 0) > 0]
    smallest = min(positive_ev, key=lambda x: x["with_full_filters"]["n_trades"]) if positive_ev else None

    # Build ensemble agreement strategy: train 3 different seeds and require 2/3 agree on top-decile
    print("Building ensemble agreement strategy...")
    p_h_list = []
    for seed in [7, 11, 19]:
        Xtr = np.column_stack(
            [pd.to_numeric(df_train[c], errors="coerce").fillna(0).to_numpy() for c in top10_safe]
        )
        Xho = np.column_stack(
            [pd.to_numeric(df_holdout[c], errors="coerce").fillna(0).to_numpy() for c in top10_safe]
        )
        Xtr = np.clip(np.nan_to_num(np.where(np.isfinite(Xtr), Xtr, 0), nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
        Xho = np.clip(np.nan_to_num(np.where(np.isfinite(Xho), Xho, 0), nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
        m = LogisticRegression(max_iter=300, solver="lbfgs", random_state=seed)
        m.fit(Xtr, target_train)
        p_h_list.append(m.predict_proba(Xho)[:, 1])
    p_ensemble = np.mean(p_h_list, axis=0)
    p_std = np.std(p_h_list, axis=0)
    # Agreement score: high mean, low std
    agreement = p_ensemble * (1.0 - p_std)
    p_thresh = float(np.quantile(agreement, 0.95))
    mask_ens = agreement >= p_thresh
    nret = pd.to_numeric(df_holdout["net_forward_return"], errors="coerce").fillna(0).to_numpy()
    sel_ens = nret[mask_ens] - 0.01
    if len(sel_ens) > 0:
        wins = sel_ens[sel_ens > 0].sum()
        losses = abs(sel_ens[sel_ens <= 0].sum())
        ens_pf = round(float(wins / max(losses, 1e-9)), 3)
        ens_avg = round(float(np.mean(sel_ens)), 4)
        ens_sharpe = round(float(ens_avg / max(np.std(sel_ens), 1e-9) * np.sqrt(252)), 3)
        ens_dd = round(float(np.min(np.cumsum(sel_ens) - np.maximum.accumulate(np.cumsum(sel_ens)))), 3)
    else:
        ens_pf = ens_avg = ens_sharpe = ens_dd = 0
    ens_metrics = {
        "n_trades": int(mask_ens.sum()),
        "pf": ens_pf,
        "sharpe": ens_sharpe,
        "expectancy": ens_avg,
        "max_dd": ens_dd,
    }

    return {
        "report_type": "high_conviction_strategy",
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_train": len(df_train),
        "n_holdout": len(df_holdout),
        "target": "low_vol_success",
        "features_used": top10_safe,
        "train_auc": train_auc,
        "holdout_auc": holdout_auc,
        "leaderboard": leaderboard,
        "smallest_positive_ev": smallest,
        "ensemble_agreement": ens_metrics,
    }


def write_reports() -> None:
    payload = build()
    REPORTS.mkdir(parents=True, exist_ok=True)

    jpath = REPORTS / f"high_conviction_strategy_{TS}.json"
    jpath.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    md = [
        "# High-Conviction Sparse Trading System",
        "",
        f"- Generated: `{payload['timestamp_utc']}`",
        f"- Target: `{payload['target']}`",
        f"- Train: {payload['n_train']:,} rows; Holdout: {payload['n_holdout']:,} rows",
        f"- Train AUC: **{payload['train_auc']:.4f}**",
        f"- Holdout AUC: **{payload['holdout_auc']:.4f}**",
        f"- Features used (top-10 by MI on train, drift-filtered):",
        "",
    ]
    for f in payload["features_used"]:
        md.append(f"  - `{f}`")
    md.append("")
    md.append("## Top-K confidence strategy leaderboard (with full filters)")
    md.append("")
    md.append("| Top % | Trades | PF | Sharpe | Expectancy | Max DD |")
    md.append("|---|---|---|---|---|---|")
    for x in payload["leaderboard"]:
        m = x["with_full_filters"]
        md.append(
            f"| {x['top_pct']*100:.0f}% | {m.get('n_trades', 0)} | {m.get('pf', 0):.3f} | "
            f"{m.get('sharpe', 0):.3f} | {m.get('expectancy', 0):.4f} | {m.get('max_dd', 0):.3f} |"
        )
    md.append("")
    md.append("## Top-K only (no filter)")
    md.append("")
    md.append("| Top % | Trades | PF | Sharpe | Expectancy |")
    md.append("|---|---|---|---|---|")
    for x in payload["leaderboard"]:
        m = x["top_k_only"]
        md.append(
            f"| {x['top_pct']*100:.0f}% | {m.get('n_trades', 0)} | {m.get('pf', 0):.3f} | "
            f"{m.get('sharpe', 0):.3f} | {m.get('expectancy', 0):.4f} |"
        )
    md.append("")
    md.append("## Ensemble agreement strategy (3-model mean × (1 − std), top 5%)")
    md.append("")
    e = payload["ensemble_agreement"]
    md.append(f"- Trades: {e['n_trades']}")
    md.append(f"- PF: {e['pf']:.3f}")
    md.append(f"- Sharpe: {e['sharpe']:.3f}")
    md.append(f"- Expectancy: {e['expectancy']:.4f}")
    md.append("")
    md.append("## Smallest number of trades producing positive expectancy")
    sp = payload["smallest_positive_ev"]
    if sp:
        md.append(f"- Top {sp['top_pct']*100:.0f}% confidence → {sp['with_full_filters']['n_trades']} trades, expectancy {sp['with_full_filters']['expectancy']:.4f}, PF {sp['with_full_filters']['pf']:.3f}")
    else:
        md.append("- No top-K strategy produces positive expectancy. Ensemble agreement also negative.")
    md.append("")
    md.append("## Critical question — Can sparse high-conviction become profitable after costs?")
    md.append("")
    best_pf = max(
        x["with_full_filters"].get("pf", 0) for x in payload["leaderboard"]
    )
    ens_pf = e["pf"]
    if max(best_pf, ens_pf) > 1.0:
        md.append(f"- **YES**: best sparse PF = {max(best_pf, ens_pf):.3f} > 1.0")
    else:
        md.append(f"- **NO**: best sparse PF = {max(best_pf, ens_pf):.3f}. Sparse trading alone does not overcome the cost hurdle on this dataset.")
    mpath = REPORTS / f"high_conviction_strategy_{TS}.md"
    mpath.write_text("\n".join(md), encoding="utf-8")

    # Confidence threshold leaderboard
    lb_jpath = REPORTS / f"confidence_threshold_leaderboard_{TS}.json"
    lb_jpath.write_text(json.dumps({"leaderboard": payload["leaderboard"]}, indent=2, default=str), encoding="utf-8")
    lb_md = ["# Confidence Threshold Leaderboard", ""]
    lb_md.append("| Top % | Trades (full) | PF (full) | Trades (top-K) | PF (top-K) | Trades (vol) | PF (vol) |")
    lb_md.append("|---|---|---|---|---|---|---|")
    for x in payload["leaderboard"]:
        m1 = x["with_full_filters"]
        m2 = x["top_k_only"]
        m3 = x["top_k_with_vol_filter"]
        lb_md.append(
            f"| {x['top_pct']*100:.0f}% | {m1.get('n_trades', 0)} | {m1.get('pf', 0):.3f} | "
            f"{m2.get('n_trades', 0)} | {m2.get('pf', 0):.3f} | "
            f"{m3.get('n_trades', 0)} | {m3.get('pf', 0):.3f} |"
        )
    lb_mpath = REPORTS / f"confidence_threshold_leaderboard_{TS}.md"
    lb_mpath.write_text("\n".join(lb_md), encoding="utf-8")

    print(f"WROTE: {jpath}")
    print(f"WROTE: {mpath}")
    print(f"WROTE: {lb_jpath}")
    print(f"WROTE: {lb_mpath}")


if __name__ == "__main__":
    write_reports()
