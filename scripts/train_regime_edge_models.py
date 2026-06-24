#!/usr/bin/env python3
"""
Train regime-specific model candidates for NiftyScalper research.

This script trains calibrated logistic regression, random forest, and
hist_gradient_boosting models per regime using only stable features (PSI < 0.15).

Output: models/research_regime_20260608/  and  reports/regime_model_research_*.{md,json}
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")


def load_stable_features(report_path: str) -> list:
    with open(report_path, "r") as f:
        data = json.load(f)
    return data.get("stable_features", [])


def ensure_dirs(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_regime_dataset(path: str, stable_features: list, label_col: str) -> pd.DataFrame:
    """Load a regime CSV, keep only stable features + required columns."""
    required_cols = ["timestamp", "gross_forward_return", "net_forward_return", label_col]
    all_needed = list(set(required_cols + stable_features))

    df = pd.read_csv(path, low_memory=False)
    available = [c for c in all_needed if c in df.columns]
    missing = [c for c in all_needed if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing columns in {path}: {missing}")
    df = df[available].copy()

    # Drop rows where the label is missing
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    # Sort by timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df, [c for c in stable_features if c in df.columns]


def time_based_split(df: pd.DataFrame, ratios=(0.70, 0.15, 0.15)):
    n = len(df)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    # holdout gets remainder to avoid rounding issues
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    holdout = df.iloc[n_train + n_val :].copy()
    return train, val, holdout


def build_model(family: str, random_state: int = 42):
    if family == "calibrated_logistic_regression":
        base = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        )
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", base),
        ])
        model = CalibratedClassifierCV(pipe, method="sigmoid", cv=2)
    elif family == "random_forest":
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=100,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    elif family == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=5,
            learning_rate=0.1,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model family: {family}")
    return model


def get_feature_matrix(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols].values.astype(np.float64)
    # Replace inf/-inf with NaN so imputers can handle them
    X = np.where(np.isfinite(X), X, np.nan)
    return X


def compute_holdout_metrics(df_holdout: pd.DataFrame, preds: np.ndarray, threshold: float, extra_cost: float = 0.0):
    """
    preds: probability of positive class.
    threshold: decision threshold.
    extra_cost: additional cost to subtract from net_forward_return (as decimal, e.g. 0.0025).
    Returns dict of metrics.
    """
    trade_mask = preds >= threshold
    trade_count = int(trade_mask.sum())

    returns = df_holdout["net_forward_return"].values - extra_cost

    if trade_count == 0:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "avg_return": 0.0,
        }

    trade_returns = returns[trade_mask]

    win_rate = float((trade_returns > 0).sum() / trade_count)
    # profit factor after costs
    positive_sum = trade_returns[trade_returns > 0].sum()
    negative_sum = trade_returns[trade_returns < 0].sum()
    profit_factor = float(positive_sum / abs(negative_sum)) if negative_sum != 0 else float("inf")

    # Sharpe ratio (assuming returns are already in same units; no annualization needed)
    mean_ret = float(trade_returns.mean())
    std_ret = float(trade_returns.std())
    sharpe_ratio = float(mean_ret / std_ret) if std_ret > 1e-12 else 0.0

    # Max drawdown on cumulative returns
    cum = np.cumsum(trade_returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum - running_max
    max_drawdown = float(drawdowns.min())

    return {
        "trade_count": trade_count,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "avg_return": mean_ret,
    }


def threshold_sweep(val_df: pd.DataFrame, val_probs: np.ndarray, label_col: str, min_trades: int = 50):
    """
    Sweep thresholds on validation to find the one maximizing a heuristic.
    Heuristic: profit_factor * log1p(trade_count) with minimum trade constraint.
    """
    best_thresh = 0.5
    best_score = -1e9
    best_pf = 0.0
    best_tc = 0

    for t in np.arange(0.05, 1.0, 0.05):
        mask = val_probs >= t
        tc = mask.sum()
        if tc < min_trades:
            continue
        returns = val_df["net_forward_return"].values[mask]
        pos = returns[returns > 0].sum()
        neg = returns[returns < 0].sum()
        pf = pos / abs(neg) if neg != 0 else float("inf")
        score = pf * np.log1p(tc)
        if score > best_score:
            best_score = score
            best_thresh = float(t)
            best_pf = pf
            best_tc = int(tc)

    # If nothing found with min_trades, fall back to best trade count >=10
    if best_score == -1e9:
        for t in np.arange(0.05, 1.0, 0.05):
            mask = val_probs >= t
            tc = mask.sum()
            if tc < 10:
                continue
            returns = val_df["net_forward_return"].values[mask]
            pos = returns[returns > 0].sum()
            neg = returns[returns < 0].sum()
            pf = pos / abs(neg) if neg != 0 else float("inf")
            score = pf * np.log1p(tc)
            if score > best_score:
                best_score = score
                best_thresh = float(t)
                best_pf = pf
                best_tc = int(tc)

    return best_thresh, best_pf, best_tc


def make_df_folds(train_df: pd.DataFrame, n_folds: int = 4):
    """Split a DataFrame into n sequential folds (list of DataFrames)."""
    n = len(train_df)
    fold_sizes = [n // n_folds] * n_folds
    for i in range(n % n_folds):
        fold_sizes[i] += 1
    folds = []
    start = 0
    for size in fold_sizes:
        folds.append(train_df.iloc[start:start + size].copy())
        start += size
    return folds


def walk_forward_validation(train_df: pd.DataFrame, feature_cols: list, label_col: str, model_family: str, n_folds: int = 4):
    """
    Split train data into n_folds sequential folds.
    For each fold k: train on fold 0..k-1, validate on fold k.
    """
    folds = make_df_folds(train_df, n_folds)
    results = []
    for i in range(1, n_folds):
        train_part = pd.concat(folds[:i], ignore_index=True)
        val_part = folds[i].copy()

        X_tr = get_feature_matrix(train_part, feature_cols)
        y_tr = train_part[label_col].values
        X_val = get_feature_matrix(val_part, feature_cols)
        y_val = val_part[label_col].values

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            results.append({"fold": i, "auc": None, "pf": None, "trade_count": None})
            continue

        model = build_model(model_family)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, probs))

        # compute PF at default 0.5 threshold for simplicity
        metrics = compute_holdout_metrics(val_part, probs, 0.5)
        results.append({
            "fold": i,
            "auc": auc,
            "pf": metrics["profit_factor"],
            "trade_count": metrics["trade_count"],
        })
    return results


def train_and_evaluate_regime(
    regime_name: str,
    regime_path: str,
    stable_features: list,
    label_col: str,
    model_family: str,
    output_dir: str,
):
    print(f"\n=== {regime_name} | {label_col} | {model_family} ===")

    df, feature_cols = load_regime_dataset(regime_path, stable_features, label_col)
    total_rows = len(df)
    if total_rows < 500:
        print(f"  [SKIP] Too few rows ({total_rows})")
        return None

    if len(feature_cols) < 3:
        print(f"  [SKIP] Too few features ({len(feature_cols)})")
        return None

    train, val, holdout = time_based_split(df)
    print(f"  Rows: total={total_rows}, train={len(train)}, val={len(val)}, holdout={len(holdout)}")

    X_train = get_feature_matrix(train, feature_cols)
    y_train = train[label_col].values
    X_val = get_feature_matrix(val, feature_cols)
    y_val = val[label_col].values
    X_holdout = get_feature_matrix(holdout, feature_cols)
    y_holdout = holdout[label_col].values

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        print(f"  [SKIP] Only one class in train or val")
        return None

    model = build_model(model_family)

    # Impute for tree models before fit
    if model_family in ("random_forest", ):
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)
        X_holdout_imp = imputer.transform(X_holdout)
        model.fit(X_train_imp, y_train)
        val_probs = model.predict_proba(X_val_imp)[:, 1]
        holdout_probs = model.predict_proba(X_holdout_imp)[:, 1]
    else:
        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)[:, 1]
        holdout_probs = model.predict_proba(X_holdout)[:, 1]

    # Validation metrics
    val_auc = float(roc_auc_score(y_val, val_probs))
    val_pr_auc = float(average_precision_score(y_val, val_probs))
    val_brier = float(brier_score_loss(y_val, val_probs))

    # Threshold sweep on validation
    best_thresh, val_pf, val_tc = threshold_sweep(val, val_probs, label_col, min_trades=max(30, int(len(val) * 0.01)))
    print(f"  Best threshold={best_thresh:.2f} (val PF={val_pf:.3f}, trades={val_tc})")

    # Holdout metrics at best threshold
    holdout_metrics = compute_holdout_metrics(holdout, holdout_probs, best_thresh)

    # Threshold band robustness
    band_low = max(0.0, best_thresh - 0.05)
    band_high = min(1.0, best_thresh + 0.05)
    band_low_metrics = compute_holdout_metrics(holdout, holdout_probs, band_low)
    band_high_metrics = compute_holdout_metrics(holdout, holdout_probs, band_high)

    # Cost stress test
    cost_stress = {}
    for stress_name, extra_cost in [("base", 0.0), ("+0.25", 0.0025), ("+0.50", 0.0050), ("+1.00", 0.0100)]:
        m = compute_holdout_metrics(holdout, holdout_probs, best_thresh, extra_cost=extra_cost)
        cost_stress[stress_name] = {
            "trade_count": m["trade_count"],
            "profit_factor": m["profit_factor"],
            "avg_return": m["avg_return"],
        }

    # Walk-forward validation
    wf_results = walk_forward_validation(train, feature_cols, label_col, model_family, n_folds=4)

    # Quality thresholds
    passed = (val_auc > 0.55 and holdout_metrics["trade_count"] >= 100 and holdout_metrics["profit_factor"] > 0.8)

    record = {
        "regime": regime_name,
        "label": label_col,
        "model_family": model_family,
        "n_rows": total_rows,
        "n_features": len(feature_cols),
        "val_auc": val_auc,
        "val_pr_auc": val_pr_auc,
        "val_brier": val_brier,
        "best_threshold": best_thresh,
        "threshold_band_low": band_low,
        "threshold_band_high": band_high,
        "holdout": holdout_metrics,
        "holdout_band_low": band_low_metrics,
        "holdout_band_high": band_high_metrics,
        "cost_stress": cost_stress,
        "walk_forward": wf_results,
        "passed_quality_thresholds": passed,
    }

    # Save model
    model_filename = f"{regime_name}_{model_family}_{label_col}.pkl"
    model_path = os.path.join(output_dir, model_filename)

    # Save both model and preprocessing info
    if model_family == "random_forest":
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X_train)
        joblib.dump({"model": model, "imputer": imputer, "feature_cols": feature_cols}, model_path)
    else:
        joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)

    metrics_filename = f"{regime_name}_{model_family}_{label_col}_metrics.json"
    metrics_path = os.path.join(output_dir, metrics_filename)
    with open(metrics_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    print(f"  Saved: {model_path}")
    print(f"  Val AUC={val_auc:.4f}, Holdout PF={holdout_metrics['profit_factor']:.3f}, Trades={holdout_metrics['trade_count']}, Passed={passed}")
    return record


def generate_reports(all_records: list, report_dir: str, timestamp_str: str):
    md_path = os.path.join(report_dir, f"regime_model_research_{timestamp_str}.md")
    json_path = os.path.join(report_dir, f"regime_model_research_{timestamp_str}.json")

    # JSON report
    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_candidates": len(all_records),
        "passed_count": sum(1 for r in all_records if r.get("passed_quality_thresholds", False)),
        "candidates": all_records,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Markdown report
    lines = [
        "# Regime Model Research Report",
        f"Generated: {datetime.utcnow().isoformat()}",
        f"Total candidates: {len(all_records)}",
        f"Passed basic quality thresholds: {summary['passed_count']}",
        "",
        "## Quality Criteria",
        "- ROC-AUC > 0.55 (validation)",
        "- Trade count >= 100 (holdout)",
        "- Profit factor > 0.8 (holdout, base cost)",
        "",
        "## Candidate Models",
        "",
    ]

    for r in all_records:
        name = f"{r['regime']} | {r['model_family']} | {r['label']}"
        lines.append(f"### {name}")
        lines.append(f"- Rows: {r['n_rows']:,}, Features: {r['n_features']}")
        lines.append(f"- Val ROC-AUC: {r['val_auc']:.4f}")
        lines.append(f"- Val PR-AUC: {r['val_pr_auc']:.4f}")
        lines.append(f"- Val Brier: {r['val_brier']:.4f}")
        lines.append(f"- Best threshold: {r['best_threshold']:.2f}")
        h = r["holdout"]
        lines.append(f"- Holdout trades: {h['trade_count']}, Win rate: {h['win_rate']:.2%}")
        lines.append(f"- Holdout PF (base): {h['profit_factor']:.3f}")
        lines.append(f"- Holdout Sharpe: {h['sharpe_ratio']:.4f}")
        lines.append(f"- Holdout Max DD: {h['max_drawdown']:.4f}")
        lines.append(f"- Threshold band robustness (low PF): {r['holdout_band_low']['profit_factor']:.3f}")
        lines.append(f"- Threshold band robustness (high PF): {r['holdout_band_high']['profit_factor']:.3f}")
        cs = r["cost_stress"]
        lines.append(f"- Cost stress PF: base={cs['base']['profit_factor']:.3f}, +0.25={cs['+0.25']['profit_factor']:.3f}, +0.50={cs['+0.50']['profit_factor']:.3f}, +1.00={cs['+1.00']['profit_factor']:.3f}")
        wf = r.get("walk_forward", [])
        wf_strs = [f"fold{f['fold']}: AUC={f['auc']:.3f}, PF={f['pf']:.3f}" for f in wf if f.get("auc") is not None]
        lines.append(f"- Walk-forward: {'; '.join(wf_strs) if wf_strs else 'N/A'}")
        lines.append(f"- **Passed quality thresholds: {r['passed_quality_thresholds']}**")
        lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nReports written:")
    print(f"  {json_path}")
    print(f"  {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Train regime-specific edge models")
    parser.add_argument("--feature-report", default="reports/stable_feature_list_20260608_105904.json")
    parser.add_argument("--regime-prefix", default="data/processed/regime_edge_")
    parser.add_argument("--output-dir", default="models/research_regime_20260608")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--labels", nargs="+", default=["cost_survivor_label_v2", "strong_profitable_trade_label_v2"])
    parser.add_argument("--families", nargs="+", default=["calibrated_logistic_regression", "random_forest", "hist_gradient_boosting"])
    parser.add_argument("--timestamp", default="20260608")
    args = parser.parse_args()

    stable_features = load_stable_features(args.feature_report)
    print(f"Loaded {len(stable_features)} stable features (PSI < 0.15)")

    ensure_dirs(args.output_dir)
    ensure_dirs(args.report_dir)

    regimes = [
        "high_volatility",
        "low_volatility",
        "trending",
        "mean_reverting",
        "quiet",
        "volatile_pe",
        "volatile_ce",
    ]

    all_records = []

    for regime_name in regimes:
        # Try to find the exact file (glob may have timestamp)
        import glob
        pattern = f"{args.regime_prefix}{regime_name}_*.csv"
        matches = glob.glob(pattern)
        if not matches:
            print(f"[WARN] No file found for regime {regime_name} with pattern {pattern}")
            continue
        regime_path = matches[0]
        print(f"\n>>> Processing regime: {regime_name} ({regime_path})")

        for label_col in args.labels:
            for family in args.families:
                try:
                    rec = train_and_evaluate_regime(
                        regime_name=regime_name,
                        regime_path=regime_path,
                        stable_features=stable_features,
                        label_col=label_col,
                        model_family=family,
                        output_dir=args.output_dir,
                    )
                    if rec:
                        all_records.append(rec)
                except Exception as e:
                    print(f"  [ERROR] {regime_name}/{label_col}/{family}: {e}")
                    import traceback
                    traceback.print_exc()

    generate_reports(all_records, args.report_dir, args.timestamp)


if __name__ == "__main__":
    main()
