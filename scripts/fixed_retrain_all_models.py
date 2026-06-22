#!/usr/bin/env python3
"""
Fixed Retraining Script for NiftyScalper ML Models
===================================================

This script fixes the retraining pipeline issues identified by Subagent 2:
1. Output directory naming: Uses 'research_retrain_<timestamp>/' prefix
2. Ensemble resilience: Allows ensemble with minimum 2 models
3. Ensures all eligible model families are trained
4. Properly handles cost-aware labels

Usage:
    python scripts/fixed_retrain_all_models.py --dataset <path> [--model-families <families>]

Requirements Met:
- Selected dataset loads correctly
- Feature/target separation is explicit
- Forbidden columns are excluded (DO NOT use as features)
- Time-based split is used (not random K-fold)
- Walk-forward validation is supported
- All eligible model families are attempted:
  - logistic_regression (calibrated)
  - random_forest
  - extra_trees
  - calibrated_logistic_regression
  - ensemble
- XGBoost only if explicitly installed
- Skipped models explain exact reason
- Failed models are recorded, not silently ignored
- Every successful model saves artifacts

Cost-Aware Target Support:
- cost_survivor_label_v2
- strong_profitable_trade_label_v2
- profitable_trade_label (base)

Output Location:
- All outputs saved under: models/research_retrain_<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add src and scripts to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Import the main retraining function from the fixed script
from retrain_all_edge_models import (
    load_dataset,
    choose_labels,
    detect_column_groups,
    build_feature_sets,
    train_variant,
    _timestamped_retrain_output_dir,
    MODELS_DIR,
    RETRAIN_THRESHOLD_SWEEP,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for fixed retraining."""
    parser = argparse.ArgumentParser(
        description="Fixed retraining script for NiftyScalper ML models"
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv",
        help="Path to the cost-aware edge dataset CSV/Parquet file"
    )
    parser.add_argument(
        "--model-families",
        default="default",
        help="Comma-separated list of model families or 'all'. Default includes: logistic_regression, random_forest, extra_trees, calibrated_logistic_regression, ensemble, and others if available."
    )
    parser.add_argument(
        "--walk-forward-folds",
        type=int,
        default=5,
        help="Number of walk-forward validation folds"
    )
    parser.add_argument(
        "--min-threshold-trades",
        type=int,
        default=500,
        help="Minimum trades required for a threshold to count as valid"
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retrain even if artifacts exist"
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Override output directory"
    )
    return parser.parse_args()


def main() -> None:
    """Run the fixed retraining pipeline."""
    args = parse_args()
    
    # Resolve dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        # Try relative to repo root
        dataset_path = REPO_ROOT / args.dataset
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {args.dataset}")
        sys.exit(1)
    
    print(f"[fixed_retrain] Dataset: {dataset_path}")
    print(f"[fixed_retrain] Model families: {args.model_families}")
    print(f"[fixed_retrain] Walk-forward folds: {args.walk_forward_folds}")
    
    # Load dataset
    print(f"[fixed_retrain] Loading dataset...")
    df = load_dataset(dataset_path)
    print(f"[fixed_retrain] Loaded {len(df)} rows")
    
    # Choose labels (includes cost-aware labels)
    labels, label_skip_report = choose_labels(df)
    print(f"[fixed_retrain] Labels available: {labels}")
    print(f"[fixed_retrain] Labels skipped: {label_skip_report}")
    
    if not labels:
        print("ERROR: No usable labels found")
        sys.exit(1)
    
    # Detect column groups
    column_groups = detect_column_groups(df)
    print(f"[fixed_retrain] Features: {len(column_groups['input_features'])}")
    print(f"[fixed_retrain] Evaluation return column: {column_groups['evaluation_return_column_used']}")
    
    # Build feature sets
    feature_sets = build_feature_sets(df, use_black_scholes_features=False)
    baseline_features = feature_sets["baseline_features"]
    print(f"[fixed_retrain] Baseline features: {len(baseline_features)}")
    
    # Create output directory with fixed naming
    output_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR
    artifact_dir = _timestamped_retrain_output_dir(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fixed_retrain] Output directory: {artifact_dir}")
    
    # Parse model families
    from retrain_all_edge_models import _parse_model_families_arg
    families = _parse_model_families_arg(args.model_families)
    print(f"[fixed_retrain] Models to train: {families}")
    
    # Run training
    print("[fixed_retrain] Starting training...")
    result = train_variant(
        df,
        feature_cols=baseline_features,
        labels=labels,
        artifact_dir=artifact_dir,
        variant_name="research_retrain",
        evaluation_return_column=column_groups["evaluation_return_column_used"],
        walk_forward_folds=args.walk_forward_folds,
        threshold_grid_start=0.50,
        threshold_grid_end=0.90,
        threshold_grid_step=0.025,
        min_threshold_trades=args.min_threshold_trades,
        model_families=families,
        force_retrain=args.force_retrain,
    )
    
    # Generate summary
    print("\n[fixed_retrain] Training complete!")
    print(f"[fixed_retrain] Models trained: {len(result.get('trained_models', []))}")
    print(f"[fixed_retrain] Failed models: {len(result.get('failed_models', []))}")
    print(f"[fixed_retrain] Skipped models: {len(result.get('skipped_models', []))}")
    
    # Write summary report
    summary = {
        "dataset_path": str(dataset_path),
        "artifact_dir": str(artifact_dir),
        "row_count": len(df),
        "labels_trained": labels,
        "model_families": families,
        "models_trained": result.get('trained_models', []),
        "failed_models": result.get('failed_models', []),
        "skipped_models": result.get('skipped_models', []),
        "timestamp": datetime.now().isoformat(),
        "production_adoption_allowed": False,
        "research_only": True,
    }
    
    summary_path = artifact_dir / "fixed_retrain_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n[fixed_retrain] Summary written to: {summary_path}")
    print(f"[fixed_retrain] Reports available in: {artifact_dir}")


if __name__ == "__main__":
    main()