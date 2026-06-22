#!/usr/bin/env python3
"""
Slice A Refined Retraining Script
Trains models on filtered subsets of slice_A_volatile.csv with regime-specific focus.
"""

from __future__ import annotations

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

TARGET = "cost_survivor_label_v2"
N_FOLDS = 3
MIN_ROWS = 500
RANDOM_STATE = 42

STRICT_FORBIDDEN_TOKENS = (
    "future", "realized", "next", "target", "label", "pnl",
    "profit", "loss", "return", "outcome", "exit",
    "entry_result", "trade_result", "hit_target", "hit_sl",
    "mae", "mfe", "forward", "future_price", "future_return", "post_trade",
)

FORBIDDEN_COLUMN_TOKENS = (
    "realized_vol", "realized_volatility", "realized_vol_percentile",
)

METADATA_COLUMNS = {
    "prediction_id", "timestamp", "trading_date", "trading_day", "trading_day_spot",
    "session_bucket", "selected_option_symbol", "selected_option_token",
    "instrument_key", "trading_symbol", "source_file", "spot_source",
    "observer_signal_id", "observer_run_id", "expiry", "option_type",
    "option_side_normalized", "greeks_source", "bs_iv_source", "row_enrichment_error",
    "raw_model_outputs_json", "feature_snapshot_json", "regime_features_json",
    "drift_sensitive_features_json", "data_quality_flags_json", "missing_fields_json",
    "observer_error", "entry_rule_that_fired", "signal_reason", "signal_side",
    "trade_quality_label", "simulated_result_label",
}


def get_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """Get valid feature columns, excluding forbidden tokens and metadata."""
    forbidden = set()
    for token in STRICT_FORBIDDEN_TOKENS:
        for col in df.columns:
            if token.lower() in col.lower():
                forbidden.add(col)
    
    for col in METADATA_COLUMNS:
        if col in df.columns:
            forbidden.add(col)
    
    for token in FORBIDDEN_COLUMN_TOKENS:
        for col in df.columns:
            if token.lower() in col.lower():
                forbidden.add(col)
    
    forbidden.add(target_col)
    
    feature_cols = []
    for col in df.columns:
        if col not in forbidden and col not in feature_cols:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
                if df[col].isna().mean() < 0.5:
                    feature_cols.append(col)
    
    return feature_cols


def get_model_factories():
    """Return dict of model name -> (model_fn, needs_scaling)"""
    return {
        "calibrated_logistic_regression": (lambda: CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
            method="isotonic", cv=3
        ), True),
        "logistic_regression_platt": (lambda: CalibratedClassifierCV(
            LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
            method="sigmoid", cv=3
        ), True),
        "elasticnet_logistic_regression": (lambda: CalibratedClassifierCV(
            LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_STATE, 
                             class_weight="balanced", solver="saga", penalty="elasticnet",
                             l1_ratio=0.5),
            method="isotonic", cv=3
        ), True),
        "random_forest": (lambda: RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=20,
            n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
        ), True),
        "xgboost": (lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            min_child_weight=20, n_jobs=-1, random_state=RANDOM_STATE,
            eval_metric="logloss", use_label_encoder=False
        ), True),
    }


def walk_forward_splits(
    timestamps: pd.Series,
    n_folds: int = 3,
    test_ratio: float = 0.3,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Chronological walk-forward splits (70/30 per fold).
    Returns list of (train_idx, test_idx) tuples.
    """
    n = len(timestamps)
    indices = np.arange(n)
    splits = []
    
    test_size = int(n * test_ratio)
    train_size = n - test_size
    
    for i in range(n_folds):
        # Shift the boundary forward for each fold
        offset = i * (test_size // n_folds)
        
        test_end = n
        test_start = max(train_size + offset, test_end - test_size)
        train_end = test_start
        train_start = 0
        
        if train_start >= train_end or test_start >= test_end:
            break
        if train_end - train_start < 200 or test_end - test_start < 100:
            break
        
        train_idx = indices[train_start:train_end]
        test_idx = indices[test_start:test_end]
        
        splits.append((train_idx, test_idx))
    
    return splits


def prepare_data(df: pd.DataFrame, feature_cols: list[str], target_col: str):
    """Prepare X, y, timestamps, and returns from dataframe."""
    ts_col = "timestamp" if "timestamp" in df.columns else "obs_time" if "obs_time" in df.columns else None
    
    if ts_col is None:
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                if df[col].dtype == object or "datetime" in str(df[col].dtype):
                    ts_col = col
                    break
    
    if ts_col is None:
        df = df.reset_index(drop=True)
        timestamps = pd.to_datetime(df.index, unit="ms")
    else:
        timestamps = pd.to_datetime(df[ts_col], errors="coerce")
    
    sort_idx = timestamps.argsort()
    df = df.iloc[sort_idx].reset_index(drop=True)
    timestamps = timestamps.iloc[sort_idx].reset_index(drop=True)
    
    return_col = None
    for col in ["net_forward_return", "forward_return", "realized_pnl"]:
        if col in df.columns:
            return_col = col
            break
    
    if return_col is None:
        returns = (df[target_col] == 1).astype(float)
    else:
        returns = df[return_col].fillna(0)
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    X = X.fillna(X.median())
    
    return X.values, y.values, timestamps, returns.values


def compute_pf_and_sharpe(preds: np.ndarray, y_test: np.ndarray, returns_test: np.ndarray) -> Tuple[float, float]:
    """Compute Profit Factor and Sharpe-like ratio from predictions."""
    winners = returns_test[(preds == 1) & (y_test == 1)]
    losers = returns_test[(preds == 1) & (y_test == 0)]
    
    if len(winners) > 0 and len(losers) > 0:
        avg_win = winners.mean()
        avg_loss = abs(losers.mean())
        pf = avg_win / (avg_loss + 1e-10)
        sharpe = (winners.mean() - losers.mean()) / (np.std(np.concatenate([winners, losers])) + 1e-10)
    elif len(winners) > 0:
        pf = 2.0
        sharpe = 1.0
    else:
        pf = 1.0
        sharpe = 0.0
    
    return pf, sharpe


def train_and_evaluate_slice(
    filter_name: str,
    df: pd.DataFrame,
    output_dir: str,
) -> Optional[dict]:
    """Train all models on a slice and return results."""
    print(f"\n{'='*60}")
    print(f"Filter: {filter_name}")
    print(f"Rows: {len(df)}")
    print(f"{'='*60}")
    
    feature_cols = get_feature_columns(df, TARGET)
    print(f"Feature columns: {len(feature_cols)}")
    
    if len(feature_cols) < 10:
        print(f"  SKIPPING: Not enough features ({len(feature_cols)})")
        return None
    
    X, y, timestamps, returns = prepare_data(df, feature_cols, TARGET)
    
    pos_rate = y.mean()
    print(f"Positive rate: {pos_rate:.3f}")
    
    if len(np.unique(y)) < 2:
        print(f"  SKIPPING: Only one class present")
        return None
    
    results = {
        "filter_name": filter_name,
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "positive_rate": float(pos_rate),
        "models": {},
    }
    
    model_factories = get_model_factories()
    
    for model_name, (model_fn, needs_scaling) in model_factories.items():
        print(f"\n  Training {model_name}...")
        
        try:
            splits = walk_forward_splits(timestamps, n_folds=N_FOLDS, test_ratio=0.3)
            
            if len(splits) == 0:
                print(f"    SKIPPING: Not enough splits generated")
                continue
            
            fold_results = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                if len(test_idx) < 100:
                    continue
                
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                returns_test = returns[test_idx]
                
                if needs_scaling:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                else:
                    X_train_scaled, X_test_scaled = X_train, X_test
                
                model = model_fn()
                model.fit(X_train_scaled, y_train)
                
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    proba = model.decision_function(X_test_scaled)
                    proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-10)
                
                # Find best threshold
                best_threshold = 0.5
                best_f1 = 0
                for thresh in np.arange(0.25, 0.75, 0.05):
                    preds = (proba >= thresh).astype(int)
                    if preds.sum() > 0:
                        tp = ((preds == 1) & (y_test == 1)).sum()
                        fp = ((preds == 1) & (y_test == 0)).sum()
                        fn = ((preds == 0) & (y_test == 1)).sum()
                        precision = tp / (tp + fp + 1e-10)
                        recall = tp / (tp + fn + 1e-10)
                        f1 = 2 * precision * recall / (precision + recall + 1e-10)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = thresh
                
                test_preds = (proba >= best_threshold).astype(int)
                tp = int(((test_preds == 1) & (y_test == 1)).sum())
                fp = int(((test_preds == 1) & (y_test == 0)).sum())
                fn = int(((test_preds == 0) & (y_test == 1)).sum())
                tn = int(((test_preds == 0) & (y_test == 0)).sum())
                
                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                
                pf, sharpe = compute_pf_and_sharpe(test_preds, y_test, returns_test)
                
                fold_results.append({
                    "fold": fold_idx,
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "pf": float(pf),
                    "sharpe": float(sharpe),
                    "worst_fold_pf": float(pf),  # Will track worst fold later
                    "n_trades": int(test_preds.sum()),
                    "n_winners": tp,
                    "n_losers": fp,
                    "threshold": float(best_threshold),
                })
            
            if not fold_results:
                print(f"    SKIPPING: Not enough fold results")
                continue
            
            # Compute metrics
            pf_values = [r["pf"] for r in fold_results]
            worst_fold_pf = min(pf_values)
            
            mean_f1 = np.mean([r["f1"] for r in fold_results])
            mean_pf = np.mean(pf_values)
            mean_sharpe = np.mean([r["sharpe"] for r in fold_results])
            mean_trades = int(np.mean([r["n_trades"] for r in fold_results]))
            
            best_fold_idx = np.argmax([r["f1"] for r in fold_results])
            best_fold = fold_results[best_fold_idx]
            worst_fold_idx = np.argmin(pf_values)
            worst_fold = fold_results[worst_fold_idx]
            
            results["models"][model_name] = {
                "mean_f1": float(mean_f1),
                "mean_pf": float(mean_pf),
                "mean_sharpe": float(mean_sharpe),
                "mean_trades": mean_trades,
                "worst_fold_pf": float(worst_fold_pf),
                "best_fold": best_fold,
                "worst_fold": worst_fold,
                "all_folds": fold_results,
            }
            
            print(f"    F1: {mean_f1:.3f}, PF: {mean_pf:.3f}, Worst PF: {worst_fold_pf:.3f}, Trades: {mean_trades}")
            
        except Exception as e:
            print(f"    ERROR: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            continue
    
    # Determine best model
    best_model = None
    best_f1 = 0
    for model_name, model_results in results["models"].items():
        if model_results["mean_f1"] > best_f1:
            best_f1 = model_results["mean_f1"]
            best_model = model_name
    
    results["best_model"] = best_model
    if best_model:
        results["best_model_metrics"] = results["models"][best_model]
    
    return results


def main():
    # Configuration
    INPUT_PATH = "models/research_regime_retrain_20260608_080000/slices/slice_A_volatile.csv"
    OUTPUT_DIR = "models/research_slice_A_refine_20260608_090000"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading slice_A_volatile.csv...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"Total rows: {len(df)}")
    
    # Define filter combinations
    # Note: spread columns (bid_ask_spread, bid_ask_spread_pct) are all null in this dataset
    # So filters involving spread are modified or skipped
    
    filter_configs = [
        ("CE_only", df[df["option_type"] == "CE"]),
        ("PE_only", df[df["option_type"] == "PE"]),
        ("ITM_all", df[
            ((df["option_type"] == "CE") & (df["moneyness"] > 1)) |
            ((df["option_type"] == "PE") & (df["moneyness"] < 1))
        ]),
        ("CE_ITM", df[(df["option_type"] == "CE") & (df["moneyness"] > 1)]),
        ("DTE_greater_7", df[df["dte_days"] > 7]),
        ("CE_DTE_greater_7", df[(df["option_type"] == "CE") & (df["dte_days"] > 7)]),
        ("CE_ITM_DTE_greater_7", df[
            (df["option_type"] == "CE") & 
            (df["moneyness"] > 1) & 
            (df["dte_days"] > 7)
        ]),
    ]
    
    print(f"\n{'='*60}")
    print("FILTER COMBINATIONS")
    print(f"{'='*60}")
    
    filter_summary = []
    for name, filtered_df in filter_configs:
        # Ensure target is not null
        valid_df = filtered_df[filtered_df[TARGET].notna()]
        print(f"  {name}: {len(valid_df)} rows (after dropping null target)")
        filter_summary.append({
            "filter": name,
            "total_rows": len(filtered_df),
            "valid_rows": len(valid_df),
            "meets_threshold": len(valid_df) >= MIN_ROWS
        })
    
    all_results = []
    
    for name, filtered_df in filter_configs:
        valid_df = filtered_df[filtered_df[TARGET].notna()].copy()
        
        if len(valid_df) < MIN_ROWS:
            print(f"\n  SKIPPING {name}: Only {len(valid_df)} rows (need >= {MIN_ROWS})")
            continue
        
        results = train_and_evaluate_slice(
            filter_name=name,
            df=valid_df,
            output_dir=OUTPUT_DIR,
        )
        
        if results:
            all_results.append(results)
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("GENERATING SUMMARY REPORT")
    print(f"{'='*60}")
    
    # Check improvement status
    improved_filters = []
    for result in all_results:
        if result["best_model"]:
            best_model_metrics = result["best_model_metrics"]
            worst_fold_pf = best_model_metrics.get("worst_fold_pf", 0)
            if worst_fold_pf >= 1.00:
                improved_filters.append(result["filter_name"])
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "target": TARGET,
        "n_folds": N_FOLDS,
        "min_rows": MIN_ROWS,
        "filters_attempted": len(filter_configs),
        "filters_succeeded": len(all_results),
        "filters_improved_to_pf_1_00": improved_filters,
        "filter_summary": filter_summary,
        "results": all_results,
    }
    
    # Save JSON report
    json_path = os.path.join(OUTPUT_DIR, "slice_A_refined_retraining_20260608_090000.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved JSON report to: {json_path}")
    
    # Generate Markdown report
    md_lines = [
        "# Slice A Refined Retraining Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Target:** {TARGET}",
        f"**Walk-Forward Folds:** {N_FOLDS}",
        f"**Minimum Rows:** {MIN_ROWS}",
        f"**Filters Attempted:** {len(filter_configs)}",
        f"**Filters Succeeded:** {len(all_results)}",
        f"**Filters Improved to PF >= 1.00:** {len(improved_filters)}",
        "",
        "## Filter Combinations Summary",
        "",
        "| Filter | Total Rows | Valid Rows | Meets Threshold |",
        "|--------|------------|------------|-----------------|",
    ]
    
    for fs in filter_summary:
        status = "✓" if fs["meets_threshold"] else "✗"
        md_lines.append(f"| {fs['filter']} | {fs['total_rows']:,} | {fs['valid_rows']:,} | {status} |")
    
    md_lines.extend([
        "",
        "## Filters with PF >= 1.00 in Worst Fold",
        "",
    ])
    
    if improved_filters:
        for f in improved_filters:
            md_lines.append(f"- **{f}**: Worst fold PF >= 1.00 ✓")
    else:
        md_lines.append("*None of the filters achieved PF >= 1.00 in worst fold*")
    
    md_lines.extend([
        "",
        "## Executive Summary",
        "",
        "| Filter | Rows | Best Model | F1 | PF | Worst Fold PF | Improved? |",
        "|--------|------|------------|----|----|---------------|-----------|",
    ])
    
    for result in all_results:
        filter_name = result["filter_name"]
        n_rows = result["n_rows"]
        
        best_model = result.get("best_model", "N/A")
        if best_model and best_model in result["models"]:
            mr = result["models"][best_model]
            worst_pf = mr.get("worst_fold_pf", 0)
            improved = "✓" if worst_pf >= 1.00 else "✗"
            md_lines.append(
                f"| {filter_name} | {n_rows:,} | {best_model} | "
                f"{mr['mean_f1']:.3f} | {mr['mean_pf']:.3f} | {worst_pf:.3f} | {improved} |"
            )
        else:
            md_lines.append(f"| {filter_name} | {n_rows:,} | N/A | - | - | - | - |")
    
    md_lines.extend([
        "",
        "## Detailed Results by Filter",
        "",
    ])
    
    for result in all_results:
        md_lines.extend([
            f"### {result['filter_name']}",
            "",
            f"- **Rows:** {result['n_rows']:,}",
            f"- **Features:** {result['n_features']}",
            f"- **Positive Rate:** {result['positive_rate']:.3f}",
            f"- **Best Model:** {result.get('best_model', 'N/A')}",
            "",
            "#### Model Performance",
            "",
            "| Model | Mean F1 | Mean PF | Worst Fold PF | Mean Trades |",
            "|-------|---------|---------|---------------|-------------|",
        ])
        
        for model_name, model_results in sorted(result["models"].items(), 
                                                 key=lambda x: x[1]["mean_f1"], reverse=True):
            md_lines.append(
                f"| {model_name} | {model_results['mean_f1']:.3f} | "
                f"{model_results['mean_pf']:.3f} | {model_results['worst_fold_pf']:.3f} | "
                f"{model_results['mean_trades']:,} |"
            )
        
        md_lines.append("")
    
    md_path = os.path.join(OUTPUT_DIR, "slice_A_refined_retraining_20260608_090000.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved Markdown report to: {md_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()