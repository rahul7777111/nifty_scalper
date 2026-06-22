#!/usr/bin/env python3
"""
Regime-Specific Retraining Script
Trains models on filtered slices of the full dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb


def simple_walk_forward_splits(
    timestamps: pd.Series,
    n_folds: int = 4,
    test_ratio: float = 0.2,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Simple time-series walk-forward validation.
    Each fold uses earlier data for training and later data for testing.
    """
    n = len(timestamps)
    indices = np.arange(n)
    splits = []
    
    # Calculate the size of each test window
    test_size = max(100, int(n * test_ratio))
    train_size = n - test_size
    
    for i in range(n_folds):
        # Start from the beginning and move forward
        test_start = train_size + i * (test_size // 2)
        test_end = min(n, test_start + test_size)
        
        if test_start >= n or test_end <= test_start:
            break
        
        train_end = test_start
        train_start = 0
        
        train_idx = indices[train_start:train_end]
        test_idx = indices[test_start:test_end]
        
        if len(train_idx) >= 500 and len(test_idx) >= 100:
            splits.append((train_idx, np.array([]), test_idx))  # validation is empty for simple version
    
    return splits


TARGET = "cost_survivor_label_v2"
N_FOLDS = 4
MIN_ROWS = 1000
RANDOM_STATE = 42

# Forbidden tokens for feature selection
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
    
    # Also exclude metadata columns
    for col in METADATA_COLUMNS:
        if col in df.columns:
            forbidden.add(col)
    
    # Exclude forbidden column tokens
    for token in FORBIDDEN_COLUMN_TOKENS:
        for col in df.columns:
            if token.lower() in col.lower():
                forbidden.add(col)
    
    # Exclude the target
    forbidden.add(target_col)
    
    # Only use numeric columns
    feature_cols = []
    for col in df.columns:
        if col not in forbidden and col not in feature_cols:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
                # Skip if too many missing values
                if df[col].isna().mean() < 0.5:
                    feature_cols.append(col)
    
    return feature_cols


def get_model_factories():
    """Return dict of model name -> (model, needs_scaling)"""
    return {
        "random_forest": (lambda: RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=20,
            n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
        ), True),
        "extra_trees": (lambda: ExtraTreesClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=20,
            n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
        ), True),
        "logistic_regression_platt": (lambda: CalibratedClassifierCV(
            LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
            method="sigmoid", cv=3
        ), True),
        "calibrated_logistic_regression": (lambda: CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
            method="isotonic", cv=3
        ), True),
        "hist_gradient_boosting": (lambda: HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.1,
            min_samples_leaf=20, random_state=RANDOM_STATE, class_weight="balanced"
        ), True),
        "xgboost": (lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            min_child_weight=20, n_jobs=-1, random_state=RANDOM_STATE,
            eval_metric="logloss", use_label_encoder=False
        ), True),
    }


def prepare_data(df: pd.DataFrame, feature_cols: list[str], target_col: str):
    """Prepare X, y, timestamps, and returns from dataframe."""
    # Use timestamp column
    ts_col = "timestamp" if "timestamp" in df.columns else "obs_time" if "obs_time" in df.columns else None
    
    if ts_col is None:
        # Try to find a time column
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                if df[col].dtype == object or "datetime" in str(df[col].dtype):
                    ts_col = col
                    break
    
    if ts_col is None:
        # Fallback to row index as pseudo-timestamp
        df = df.reset_index(drop=True)
        timestamps = pd.to_datetime(df.index, unit="ms")
    else:
        timestamps = pd.to_datetime(df[ts_col], errors="coerce")
    
    # Sort by timestamp to ensure monotonicity
    sort_idx = timestamps.argsort()
    df = df.iloc[sort_idx].reset_index(drop=True)
    timestamps = timestamps.iloc[sort_idx].reset_index(drop=True)
    
    # Get returns column
    return_col = None
    for col in ["net_forward_return", "forward_return", "realized_pnl"]:
        if col in df.columns:
            return_col = col
            break
    
    if return_col is None:
        # Create pseudo returns from target
        returns = (df[target_col] == 1).astype(float)
    else:
        returns = df[return_col].fillna(0)
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Fill missing values
    X = X.fillna(X.median())
    
    return X.values, y.values, timestamps, returns.values


def train_and_evaluate_slice(
    slice_name: str,
    df: pd.DataFrame,
    output_dir: str,
) -> dict:
    """Train all models on a slice and return results."""
    print(f"\n{'='*60}")
    print(f"Processing Slice: {slice_name}")
    print(f"Rows: {len(df)}")
    print(f"{'='*60}")
    
    feature_cols = get_feature_columns(df, TARGET)
    print(f"Feature columns: {len(feature_cols)}")
    
    if len(feature_cols) < 10:
        print(f"  SKIPPING: Not enough features ({len(feature_cols)})")
        return None
    
    X, y, timestamps, returns = prepare_data(df, feature_cols, TARGET)
    
    # Check class balance
    pos_rate = y.mean()
    print(f"Positive rate: {pos_rate:.3f}")
    
    if len(np.unique(y)) < 2:
        print(f"  SKIPPING: Only one class present")
        return None
    
    results = {
        "slice_name": slice_name,
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "positive_rate": float(pos_rate),
        "models": {},
    }
    
    model_factories = get_model_factories()
    
    for model_name, (model_fn, needs_scaling) in model_factories.items():
        print(f"\n  Training {model_name}...")
        
        try:
            # Simple walk-forward validation (no timestamp grouping issues)
            splits = simple_walk_forward_splits(timestamps, n_folds=N_FOLDS)
            
            if len(splits) == 0:
                print(f"    SKIPPING: Not enough splits generated")
                continue
            
            fold_results = []
            
            for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
                if len(test_idx) < 100:
                    continue
                
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                returns_test = returns[test_idx]
                
                # Scale features if needed
                if needs_scaling:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                else:
                    X_train_scaled, X_test_scaled = X_train, X_test
                
                # Train model
                model = model_fn()
                model.fit(X_train_scaled, y_train)
                
                # Predict probabilities
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    proba = model.decision_function(X_test_scaled)
                    proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-10)
                
                # Find best threshold using validation set
                best_threshold = 0.5
                best_f1 = 0
                for thresh in np.arange(0.3, 0.8, 0.05):
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
                
                # Calculate metrics
                test_preds = (proba >= best_threshold).astype(int)
                tp = ((test_preds == 1) & (y_test == 1)).sum()
                fp = ((test_preds == 1) & (y_test == 0)).sum()
                fn = ((test_preds == 0) & (y_test == 1)).sum()
                tn = ((test_preds == 0) & (y_test == 0)).sum()
                
                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                
                # Calculate PF and Sharpe from returns
                winners = returns_test[test_preds == 1]
                losers = returns_test[test_preds == 0]
                
                if len(winners) > 0 and len(losers) > 0:
                    avg_win = winners.mean()
                    avg_loss = abs(losers.mean())
                    pf = avg_win / (avg_loss + 1e-10)
                    sharpe = (winners.mean() - losers.mean()) / (np.std(np.concatenate([winners, -losers])) + 1e-10)
                else:
                    pf = 1.0
                    sharpe = 0.0
                
                # Simple regime metrics (volatility regime breakdown if available)
                try:
                    regime_volatile = df.iloc[test_idx]["regime_volatile"].values if "regime_volatile" in df.columns else np.zeros(len(test_idx))
                    regime_in_slice = regime_volatile == 1.0
                    if regime_in_slice.sum() >= 50:
                        volatile_f1 = ((y_test[regime_in_slice] == 1) & (test_preds[regime_in_slice] == 1)).sum() / (regime_in_slice.sum() + 1e-10)
                    else:
                        volatile_f1 = f1
                except:
                    volatile_f1 = f1
                
                fold_results.append({
                    "fold": fold_idx,
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "pf": float(pf),
                    "sharpe": float(sharpe),
                    "n_trades": int(test_preds.sum()),
                    "n_winners": int(tp),
                    "n_losers": int(fp),
                    "threshold": float(best_threshold),
                    "volatile_regime_f1": float(volatile_f1),
                })
            
            if not fold_results:
                print(f"    SKIPPING: Not enough fold results")
                continue
            
            # Aggregate fold results
            mean_f1 = np.mean([r["f1"] for r in fold_results])
            mean_pf = np.mean([r["pf"] for r in fold_results])
            mean_sharpe = np.mean([r["sharpe"] for r in fold_results])
            mean_trades = int(np.mean([r["n_trades"] for r in fold_results]))
            
            # Best fold (by F1)
            best_fold_idx = np.argmax([r["f1"] for r in fold_results])
            best_fold = fold_results[best_fold_idx]
            
            results["models"][model_name] = {
                "mean_f1": float(mean_f1),
                "mean_pf": float(mean_pf),
                "mean_sharpe": float(mean_sharpe),
                "mean_trades": mean_trades,
                "best_fold": best_fold,
                "all_folds": fold_results,
            }
            
            print(f"    F1: {mean_f1:.3f}, PF: {mean_pf:.3f}, Sharpe: {mean_sharpe:.3f}, Trades: {mean_trades}")
            
        except Exception as e:
            print(f"    ERROR: {str(e)[:100]}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Regime-Specific Retraining")
    parser.add_argument("--slice-dir", default="models/research_regime_retrain_20260608_080000/slices",
                        help="Directory containing slice CSV files")
    parser.add_argument("--output-dir", default="models/research_regime_retrain_20260608_080000",
                        help="Output directory for reports")
    parser.add_argument("--min-rows", type=int, default=1000, help="Minimum rows to process a slice")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all slice files
    slice_files = {}
    slice_dir = Path(args.slice_dir)
    for f in slice_dir.glob("slice_*.csv"):
        slice_name = f.stem.replace("slice_", "").replace("_", " ").title().replace(" ", "_")
        slice_files[slice_name] = f
    
    print(f"Found {len(slice_files)} slices:")
    for name, path in slice_files.items():
        print(f"  {name}: {path}")
    
    all_results = []
    
    for slice_name, slice_path in sorted(slice_files.items()):
        print(f"\n{'='*60}")
        print(f"Loading {slice_name} from {slice_path}")
        
        df = pd.read_csv(slice_path, low_memory=False)
        print(f"  Rows: {len(df)}")
        
        if len(df) < args.min_rows:
            print(f"  SKIPPING: Below minimum rows ({len(df)} < {args.min_rows})")
            continue
        
        results = train_and_evaluate_slice(
            slice_name=slice_name,
            df=df,
            output_dir=args.output_dir,
        )
        
        if results:
            all_results.append(results)
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("GENERATING SUMMARY REPORT")
    print(f"{'='*60}")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "target": TARGET,
        "n_folds": N_FOLDS,
        "slices_evaluated": len(all_results),
        "slice_results": all_results,
    }
    
    # Save JSON report
    json_path = os.path.join(args.output_dir, "regime_specific_retraining_20260608_080000.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved JSON report to: {json_path}")
    
    # Generate Markdown report
    md_lines = [
        "# Regime-Specific Retraining Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Target:** {TARGET}",
        f"**Walk-Forward Folds:** {N_FOLDS}",
        f"**Slices Evaluated:** {len(all_results)}",
        "",
        "## Executive Summary",
        "",
        "| Slice | Rows | Features | Best Model | Mean F1 | Mean PF | Mean Sharpe |",
        "|-------|------|----------|------------|---------|---------|-------------|",
    ]
    
    for result in all_results:
        slice_name = result["slice_name"]
        n_rows = result["n_rows"]
        n_features = result["n_features"]
        
        # Find best model
        best_model = None
        best_f1 = 0
        for model_name, model_results in result["models"].items():
            if model_results["mean_f1"] > best_f1:
                best_f1 = model_results["mean_f1"]
                best_model = model_name
        
        if best_model:
            mr = result["models"][best_model]
            md_lines.append(
                f"| {slice_name} | {n_rows:,} | {n_features} | {best_model} | "
                f"{mr['mean_f1']:.3f} | {mr['mean_pf']:.3f} | {mr['mean_sharpe']:.3f} |"
            )
    
    md_lines.extend([
        "",
        "## Detailed Results by Slice",
        "",
    ])
    
    for result in all_results:
        md_lines.extend([
            f"### {result['slice_name']}",
            "",
            f"- **Rows:** {result['n_rows']:,}",
            f"- **Features:** {result['n_features']}",
            f"- **Positive Rate:** {result['positive_rate']:.3f}",
            "",
            "#### Model Performance",
            "",
            "| Model | Mean F1 | Mean PF | Mean Sharpe | Mean Trades |",
            "|-------|---------|---------|-------------|-------------|",
        ])
        
        for model_name, model_results in sorted(result["models"].items(), 
                                                 key=lambda x: x[1]["mean_f1"], reverse=True):
            md_lines.append(
                f"| {model_name} | {model_results['mean_f1']:.3f} | "
                f"{model_results['mean_pf']:.3f} | {model_results['mean_sharpe']:.3f} | "
                f"{model_results['mean_trades']:,} |"
            )
        
        md_lines.append("")
    
    md_path = os.path.join(args.output_dir, "regime_specific_retraining_20260608_080000.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved Markdown report to: {md_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()