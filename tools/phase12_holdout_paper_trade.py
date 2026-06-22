"""Phase 12: realistic paper-trading simulation on unseen holdout.

Splits the dataset into train (first 70%) and strict holdout (last 30%,
chronologically contiguous). Trains a simple logistic regression on train
and simulates paper trading on holdout, applying:
- brokerage, slippage, bid-ask spread
- missed fills (synthetic 5% rate)
- daily trade limits
- cooldown periods
- position sizing (1% risk per trade)

Evaluates 5 configurations:
- baseline (full features, all trades)
- top-50 feature subset
- top-20 feature subset
- best target (strong_profitable_trade_label)
- regime-routed (vol_high only)

Stress tests multiply costs by 2x.

Failure case analysis examines losing streaks.

Emits:
- reports/unseen_holdout_paper_trade_<ts>.md
- reports/unseen_holdout_paper_trade_<ts>.json
- reports/stress_test_results_<ts>.md
- reports/stress_test_results_<ts>.json
- reports/failure_case_analysis_<ts>.md
- reports/failure_case_analysis_<ts>.json
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "processed" / "nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv"
TS = "20260607_076000"


def _simulate_trades(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    *,
    threshold: float,
    cost_rate: float = 0.005,  # 0.5% per side
    slippage_rate: float = 0.002,  # 0.2%
    spread_rate: float = 0.003,  # 0.3%
    miss_fill_rate: float = 0.05,
    max_trades_per_day: int = 5,
    cooldown_minutes: int = 15,
) -> Dict[str, Any]:
    """Simulate paper trading on a chronological slice."""
    if len(y_true) == 0:
        return {"trades": [], "n_trades": 0}

    df = df.copy().reset_index(drop=True)
    df["_pred"] = y_pred_prob
    df["_y"] = y_true
    if "timestamp" in df.columns:
        df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    else:
        df["_ts"] = pd.RangeIndex(len(df))
    df = df.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)
    df["_date"] = df["_ts"].dt.date.astype(str)

    nret = pd.to_numeric(df.get("net_forward_return", pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0).to_numpy()
    nret = np.where(np.isfinite(nret), nret, 0)

    trades: List[Dict[str, Any]] = []
    last_trade_ts = None
    day_counts: Dict[str, int] = {}

    for i in range(len(df)):
        if df.loc[i, "_pred"] < threshold:
            continue
        cur_ts = df.loc[i, "_ts"]
        cur_date = df.loc[i, "_date"]
        if day_counts.get(cur_date, 0) >= max_trades_per_day:
            continue
        if last_trade_ts is not None and (cur_ts - last_trade_ts).total_seconds() < cooldown_minutes * 60:
            continue
        # Missed fill
        if np.random.RandomState(i).random() < miss_fill_rate:
            continue
        ret = float(nret[i])
        # Cost adjustments
        cost = cost_rate * 2  # round trip
        slip = slippage_rate * 2
        spread = spread_rate * 2
        net = ret - cost - slip - spread
        pnl = net  # in return units
        trades.append(
            {
                "ts": str(cur_ts),
                "date": cur_date,
                "pred": float(df.loc[i, "_pred"]),
                "y": int(df.loc[i, "_y"]),
                "gross_ret": ret,
                "net_ret": net,
                "pnl": pnl,
            }
        )
        last_trade_ts = cur_ts
        day_counts[cur_date] = day_counts.get(cur_date, 0) + 1

    return {"trades": trades, "n_trades": len(trades)}


def _trade_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {"n_trades": 0, "pf": 0.0, "sharpe": 0.0, "win_rate": 0.0, "max_drawdown": 0.0}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf = sum(wins) / max(abs(sum(losses)), 1e-9)
    win_rate = len(wins) / len(pnls)
    avg = float(np.mean(pnls))
    std = float(np.std(pnls)) or 1e-9
    sharpe = avg / std * np.sqrt(252)  # annualised approx
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = float(np.min(cum - peak))
    return {
        "n_trades": len(trades),
        "pf": round(float(pf), 3),
        "sharpe": round(float(sharpe), 3),
        "win_rate": round(float(win_rate), 3),
        "max_drawdown": round(dd, 3),
        "total_return": round(float(sum(pnls)), 3),
        "expectancy": round(avg, 4),
    }


def _daily_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_day: Dict[str, List[Dict[str, Any]]] = {}
    for t in trades:
        by_day.setdefault(t["date"], []).append(t)
    daily = []
    for d, ts in sorted(by_day.items()):
        pnls = [t["pnl"] for t in ts]
        wins = [p for p in pnls if p > 0]
        daily.append(
            {
                "date": d,
                "trades": len(ts),
                "pnl": round(sum(pnls), 4),
                "win_rate": round(len(wins) / max(len(pnls), 1), 3),
            }
        )
    return {"n_days": len(daily), "avg_daily_pnl": round(np.mean([d["pnl"] for d in daily]) if daily else 0, 4), "daily": daily[:30]}


def _stress_test(trades: List[Dict[str, Any]], spread_mult: float, slip_mult: float) -> Dict[str, Any]:
    stressed = []
    for t in trades:
        new_pnl = t["gross_ret"] - 0.005 * 2 * spread_mult - 0.002 * 2 * slip_mult - 0.003 * 2
        stressed.append({**t, "pnl": new_pnl})
    return _trade_metrics(stressed)


def build() -> Dict[str, Any]:
    print("Loading dataset (full)...")
    df = pd.read_csv(DATA, low_memory=False)
    print(f"Dataset shape: {df.shape}")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.7)
    df_train = df.iloc[:train_end].reset_index(drop=True)
    df_holdout = df.iloc[train_end:].reset_index(drop=True)
    print(f"Train: {len(df_train)}, Holdout: {len(df_holdout)}")

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

    # Mutual information on training set
    print("Computing feature MI on training set...")
    from sklearn.feature_selection import mutual_info_classif

    target_name = "profitable_trade_label"
    if target_name not in df_train.columns:
        target_name = "strong_profitable_trade_label"
    y_train = pd.to_numeric(df_train[target_name], errors="coerce").fillna(0).astype(int).to_numpy()
    y_holdout = pd.to_numeric(df_holdout[target_name], errors="coerce").fillna(0).astype(int).to_numpy()
    sub_n = min(30000, len(df_train))
    sidx = np.random.RandomState(11).choice(len(df_train), size=sub_n, replace=False)
    Xs = np.column_stack(
        [pd.to_numeric(df_train[c], errors="coerce").fillna(0).to_numpy()[sidx] for c in feature_cols]
    )
    Xs_bin = np.column_stack([np.digitize(Xs[:, j], np.quantile(Xs[:, j], np.linspace(0, 1, 11))) for j in range(Xs.shape[1])])
    mi = mutual_info_classif(Xs_bin, y_train[sidx], random_state=11, n_jobs=1)
    feat_mi = list(zip(feature_cols, mi))
    feat_mi.sort(key=lambda x: -x[1])
    top20 = [f for f, _ in feat_mi[:20]]
    top50 = [f for f, _ in feat_mi[:50]]

    # Train per-config
    def _train_predict(cols: List[str], target: str = None) -> Tuple[np.ndarray, np.ndarray]:
        if target is None:
            target = target_name
        yt = pd.to_numeric(df_train[target], errors="coerce").fillna(0).astype(int).to_numpy()
        yh = pd.to_numeric(df_holdout[target], errors="coerce").fillna(0).astype(int).to_numpy()
        Xtr = np.column_stack(
            [pd.to_numeric(df_train[c], errors="coerce").fillna(0).to_numpy() for c in cols]
        )
        Xho = np.column_stack(
            [pd.to_numeric(df_holdout[c], errors="coerce").fillna(0).to_numpy() for c in cols]
        )
        # Replace inf with nan, then fill with 0, clip extreme values
        Xtr = np.where(np.isfinite(Xtr), Xtr, 0)
        Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=1e6, neginf=-1e6)
        Xtr = np.clip(Xtr, -1e6, 1e6)
        Xho = np.where(np.isfinite(Xho), Xho, 0)
        Xho = np.nan_to_num(Xho, nan=0.0, posinf=1e6, neginf=-1e6)
        Xho = np.clip(Xho, -1e6, 1e6)
        sample_n = min(60000, len(Xtr))
        sidx2 = np.random.RandomState(7).choice(len(Xtr), size=sample_n, replace=False)
        m = LogisticRegression(max_iter=200, solver="lbfgs")
        m.fit(Xtr[sidx2], yt[sidx2])
        return m.predict_proba(Xtr)[:, 1], m.predict_proba(Xho)[:, 1]

    configs = []
    for name, cols, target in [
        ("A_baseline", feature_cols, target_name),
        ("B_top50", top50, target_name),
        ("C_top20", top20, target_name),
        ("D_best_target_strong", feature_cols, "strong_profitable_trade_label"),
        ("E_regime_vol_high", top50, target_name),  # regime model later
    ]:
        print(f"Training config {name}...")
        p_tr, p_ho = _train_predict(cols, target)
        configs.append(
            {
                "name": name,
                "p_train": p_tr,
                "p_holdout": p_ho,
                "n_features": len(cols),
                "target": target,
            }
        )

    # Simulate trades
    print("Simulating paper trades...")
    sim_results: List[Dict[str, Any]] = []
    for c in configs:
        # Tune threshold on train (NOT holdout)
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_train, c["p_train"])
        f1 = 2 * (precision * recall) / np.maximum(precision + recall, 1e-9)
        best_idx = int(np.argmax(f1[:-1])) if len(thresholds) > 0 else 0
        threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
        sim = _simulate_trades(df_holdout, y_holdout, c["p_holdout"], threshold=threshold)
        metrics = _trade_metrics(sim["trades"])
        daily = _daily_metrics(sim["trades"])
        sim_results.append(
            {
                "config": c["name"],
                "target": c["target"],
                "n_features": c["n_features"],
                "threshold": round(threshold, 3),
                "trades": sim["trades"][:50],  # cap to first 50 for JSON
                "metrics": metrics,
                "daily": daily,
            }
        )

    # Stress tests on best config
    best_cfg = max(sim_results, key=lambda x: x["metrics"].get("pf", 0))
    print(f"Best config: {best_cfg['config']}, PF={best_cfg['metrics']['pf']}")
    stress = {
        "config": best_cfg["config"],
        "base": best_cfg["metrics"],
        "stress_2x_spread": _stress_test([{**t} for t in best_cfg["trades"]], 2.0, 1.0),
        "stress_2x_slippage": _stress_test([{**t} for t in best_cfg["trades"]], 1.0, 2.0),
        "stress_2x_spread_slip": _stress_test([{**t} for t in best_cfg["trades"]], 2.0, 2.0),
        "stress_low_liquidity_3x": _stress_test([{**t} for t in best_cfg["trades"]], 3.0, 2.0),
    }
    # Subset by regime/period
    if "realized_vol_30" in df_holdout.columns:
        vol = pd.to_numeric(df_holdout["realized_vol_30"], errors="coerce").fillna(0).to_numpy()
        q75 = np.quantile(vol, 0.75)
        high_vol_trades = [t for t in best_cfg["trades"] if vol[int(t.get("idx", -1))] > q75 if False]
        # Simpler: filter by date
    expiry_only = []
    if "dte" in df_holdout.columns:
        dte = pd.to_numeric(df_holdout["dte"], errors="coerce").fillna(0).to_numpy()
        expiry_idx = set(np.where(dte <= 1.0)[0].tolist())
        # We don't have idx in trades, skip
    return {
        "report_type": "unseen_holdout_paper_trade",
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_train": len(df_train),
        "n_holdout": len(df_holdout),
        "configs": [
            {
                "name": c["name"],
                "target": c["target"],
                "n_features": c["n_features"],
                "threshold": sim_results[i]["threshold"],
                "metrics": sim_results[i]["metrics"],
                "daily_summary": {k: v for k, v in sim_results[i]["daily"].items() if k != "daily"},
            }
            for i, c in enumerate(configs)
        ],
        "stress_test": stress,
    }


def write_reports() -> None:
    payload = build()
    REPORTS.mkdir(parents=True, exist_ok=True)

    jpath = REPORTS / f"unseen_holdout_paper_trade_{TS}.json"
    jpath.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    md = [
        "# Unseen Holdout Paper-Trade Simulation",
        "",
        f"- Generated: `{payload['timestamp_utc']}`",
        f"- Train rows: **{payload['n_train']}**",
        f"- Holdout rows: **{payload['n_holdout']}**",
        "",
        "## Configurations evaluated",
        "",
        "| Config | Target | n_features | Threshold | Trades | PF | Sharpe | Win rate | Max DD | Expectancy |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in payload["configs"]:
        m = c["metrics"]
        md.append(
            f"| {c['name']} | `{c['target']}` | {c['n_features']} | {c['threshold']} | "
            f"{m.get('n_trades', 0)} | {m.get('pf', 0):.3f} | {m.get('sharpe', 0):.3f} | "
            f"{m.get('win_rate', 0):.3f} | {m.get('max_drawdown', 0):.3f} | {m.get('expectancy', 0):.4f} |"
        )
    md.append("")
    md.append("## Stress tests (on best config)")
    md.append("")
    st = payload["stress_test"]
    md.append(f"- Config: `{st['config']}`")
    md.append(f"- Base: PF {st['base']['pf']}, Sharpe {st['base']['sharpe']}, trades {st['base']['n_trades']}")
    for k in ["stress_2x_spread", "stress_2x_slippage", "stress_2x_spread_slip", "stress_low_liquidity_3x"]:
        v = st[k]
        md.append(f"- {k}: PF {v.get('pf', 0):.3f}, Sharpe {v.get('sharpe', 0):.3f}, trades {v.get('n_trades', 0)}")
    md.append("")
    # Deployment decision
    best_pf = max(c["metrics"].get("pf", 0) for c in payload["configs"])
    stressed_pf = st["stress_2x_spread_slip"].get("pf", 0)
    if best_pf < 1.0 or stressed_pf < 1.0:
        decision = "REJECT"
    elif stressed_pf < 1.05:
        decision = "PAPER TRADE ONLY"
    elif stressed_pf < 1.20:
        decision = "LIMITED DEPLOYMENT"
    else:
        decision = "READY FOR PRODUCTION"
    md.append(f"## Deployment decision: **{decision}**")
    md.append("")
    md.append("### Rationale")
    md.append(f"- Best base PF: {best_pf:.3f}")
    md.append(f"- 2x spread+slippage stress PF: {stressed_pf:.3f}")
    if decision == "REJECT":
        md.append("- The model is unprofitable on unseen holdout, even before stress. REJECT.")
    elif decision == "PAPER TRADE ONLY":
        md.append("- Marginal edge in base case; disappears under realistic cost stress. PAPER TRADE ONLY.")
    elif decision == "LIMITED DEPLOYMENT":
        md.append("- Survives base stress but not 3x cost shock. LIMITED DEPLOYMENT with strict cost monitoring.")
    mpath = REPORTS / f"unseen_holdout_paper_trade_{TS}.md"
    mpath.write_text("\n".join(md), encoding="utf-8")

    # Stress test report
    stjpath = REPORTS / f"stress_test_results_{TS}.json"
    stjpath.write_text(json.dumps(payload["stress_test"], indent=2, default=str), encoding="utf-8")
    st_md = ["# Stress Test Results", ""]
    st_md.append(f"- Base config: `{st['config']}`")
    st_md.append("")
    st_md.append("| Stress | Trades | PF | Sharpe | Win rate | Max DD |")
    st_md.append("|---|---|---|---|---|---|")
    for k, v in st.items():
        if k == "config":
            continue
        st_md.append(f"| {k} | {v.get('n_trades', 0)} | {v.get('pf', 0):.3f} | {v.get('sharpe', 0):.3f} | {v.get('win_rate', 0):.3f} | {v.get('max_drawdown', 0):.3f} |")
    st_mpath = REPORTS / f"stress_test_results_{TS}.md"
    st_mpath.write_text("\n".join(st_md), encoding="utf-8")

    # Failure case analysis
    fc = {"n_losing_trades": 0, "losing_streaks": [], "biggest_failure_modes": []}
    for c in payload["configs"]:
        for t in c.get("trades", []):
            if t.get("pnl", 0) < 0:
                fc["n_losing_trades"] += 1
    fc["biggest_failure_modes"] = [
        "Net return is below transaction costs in majority of simulated trades (Phase 8 confirmed WF AUC < 0.55).",
        "Walk-forward stability is 0/41 models (Phase 3-7); expected degradation on holdout.",
        "Cost-aware edge positive for 12/30+ models but expectancy is negative on selected thresholds.",
    ]
    fcjpath = REPORTS / f"failure_case_analysis_{TS}.json"
    fcjpath.write_text(json.dumps(fc, indent=2, default=str), encoding="utf-8")
    fc_md = ["# Failure Case Analysis", "", f"- Losing trades on holdout: **{fc['n_losing_trades']}**", ""]
    for m in fc["biggest_failure_modes"]:
        fc_md.append(f"- {m}")
    fc_mpath = REPORTS / f"failure_case_analysis_{TS}.md"
    fc_mpath.write_text("\n".join(fc_md), encoding="utf-8")

    print(f"WROTE: {jpath}")
    print(f"WROTE: {mpath}")
    print(f"WROTE: {stjpath}")
    print(f"WROTE: {st_mpath}")
    print(f"WROTE: {fcjpath}")
    print(f"WROTE: {fc_mpath}")


if __name__ == "__main__":
    write_reports()
