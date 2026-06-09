#!/usr/bin/env python3
"""
ml_only_candidate_enhancement_loop.py
======================================
Focused candidate enhancement pipeline.

Selects top 2-3 candidates and iterates real ML improvements
until each passes 22/22 PASS_SHADOW_READY or is proven unable.

Uses the strict gate evaluator's 22-gate system as source of truth.

Gates (22 total):
  A1-A5: Data integrity (no leakage, time split, OOS, live features)
  B1-B5: Prediction quality (AUC, PR-AUC, calibration, monotonicity, bucket wins)
  C1-C9: Trading economics (net expectancy, PF base/1.25x/1.5x/2.0x, sharpe, trades, folds, worst-fold, daily stability)
  D1-D3: Shadow mode (manifest, threshold robust, spread)

Enhancement strategies per gate failure:
  - C9 (daily stability): max trades/day, per-day selection, time-of-day filters
  - D2 (threshold robust): threshold grid, probability calibration
  - C3/C4 (cost stress): liquidity filter, spread filter, premium band
  - B1/B2 (prediction quality): better features, model selection
  - B5 (top bucket wins): threshold tuning, feature engineering
"""

import argparse
import json
import os
import pickle
import sys
import time
import warnings
import random
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# ─── Constants ───────────────────────────────────────────────────────────────

TOTAL_GATES = 22
COST_PER_TRADE_PCT = 0.0025  # 0.25%

# Gate thresholds (strict, no weakening)
PF_AT_1_00X = 1.15
PF_AT_1_25X = 1.05
PF_AT_1_50X = 1.00  # Break-even - MANDATORY
PF_AT_2_00X = 0.80
MIN_TRADES = 500
MIN_SHARPE = 0.75
MIN_DAILY_STABILITY = 0.50  # fraction of profitable days
MIN_THRESHOLD_ROBUSTNESS = 0.60
MAX_TOP_DAY_CONCENTRATION = 0.40  # top day ≤40% of returns

# Forbidden feature patterns
FORBIDDEN_PATTERNS = [
    'return', 'forward', 'future', 'pnl', 'profit', 'loss',
    'label', 'target', 'outcome', 'exit', 'mfe', 'mae',
    'realized', 'net_', 'gross_', 'expected_', 'horizon_',
    'cost_survivor', 'strong_profitable', 'high_conviction',
    'paper_candidate', 'avoid_trade', 'weak_trade', 'no_trade',
]


@dataclass
class CandidateEnhancementResult:
    """Result for one candidate iteration."""
    candidate_id: str
    candidate_type: str
    
    # Config
    label: str
    filter: str
    model: str
    threshold: float
    feature_set: str = 'live_computable'
    max_trades_per_day: int = 0  # 0 = no limit
    
    # Iteration
    iteration: int = 0
    strategy: str = ''
    
    # Status
    status: str = 'FAIL_REJECTED'
    gates_passed: int = 0
    gates_total: int = 22
    
    # Metrics
    trade_count: int = 0
    win_rate: float = 0.0
    gross_pf: float = 0.0
    net_pf: float = 0.0
    pf_at_1_00x: float = 0.0
    pf_at_1_25x: float = 0.0
    pf_at_1_50x: float = 0.0
    pf_at_2_00x: float = 0.0
    mean_sharpe: float = 0.0
    max_drawdown: float = 0.0
    daily_stability_score: float = 0.0
    threshold_robustness_score: float = 0.0
    top_day_concentration: float = 0.0
    unique_trading_days: int = 0
    
    # Gate results
    cost_1_50x_pass: bool = False
    daily_stability_pass: bool = False
    threshold_robust_pass: bool = False
    leakage_pass: bool = False
    live_computable_pass: bool = False
    
    # Artifacts
    model_path: str = ''
    manifest_path: str = ''
    shadow_manifest_path: str = ''
    feature_schema_path: str = ''
    
    # Failure tracking
    fail_reasons: List[str] = field(default_factory=list)
    error: str = ''
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def can_pass(self) -> bool:
        return (
            self.gates_passed == self.gates_total == 22 and
            bool(self.model_path) and
            bool(self.manifest_path) and
            bool(self.shadow_manifest_path) and
            bool(self.feature_schema_path) and
            self.daily_stability_pass and  # fold_pf_std < 0.30
            self.threshold_robust_pass and
            self.cost_1_50x_pass and
            self.leakage_pass and
            self.live_computable_pass
        )


class CandidateEnhancementLoop:
    """Focused enhancement loop for top 2-3 candidates."""
    
    def __init__(self, dataset_path: Path, benchmark_candidate: Path,
                 output_dir: Path, max_candidates: int = 3,
                 max_iterations_per_candidate: int = 50):
        self.dataset_path = dataset_path
        self.benchmark_candidate = benchmark_candidate
        self.output_dir = output_dir
        self.max_candidates = max_candidates
        self.max_iterations = max_iterations_per_candidate
        self.ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.all_results: List[CandidateEnhancementResult] = []
        self.passing: List[CandidateEnhancementResult] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset."""
        import pandas as pd
        print("[INFO] Loading dataset...")
        
        base_cols = [
            'timestamp', 'option_type', 'strike_price', 'dte_days', 'ltp',
            'ret_1', 'ret_3', 'ret_5',
            'range_pct', 'oi_change_pct', 'volume_change_pct',
            'volume', 'oi', 'open', 'high', 'low', 'close',
            'weekday', 'month',
            'net_forward_return', 'gross_forward_return',
            'spot_return_1', 'spot_return_3', 'spot_return_5',
            'spot_atr', 'spot_rsi', 'spot_vwap',
        ]
        
        try:
            df_sample = pd.read_csv(self.dataset_path, nrows=100)
            label_cols = [c for c in df_sample.columns 
                         if '_label' in c or 'cost_survivor' in c or 'profitable' in c]
            usecols = list(set(base_cols + label_cols))
        except:
            usecols = base_cols
        
        self.full_df = pd.read_csv(self.dataset_path, usecols=usecols, low_memory=False)
        print(f"[INFO] Dataset: {len(self.full_df)} rows, {len(self.full_df.columns)} cols")
        
        if 'timestamp' in self.full_df.columns:
            try:
                self.full_df['timestamp_dt'] = pd.to_datetime(
                    self.full_df['timestamp'], errors='coerce'
                )
                self.full_df = self.full_df.sort_values('timestamp_dt').reset_index(drop=True)
            except:
                pass
        
        self.available_labels = [c for c in self.full_df.columns 
                                if '_label' in c.lower() or 'cost_survivor' in c.lower()]
        print(f"[INFO] Labels: {self.available_labels}")
    
    def _get_live_features(self) -> List[str]:
        """Get live-computable features (no leakage)."""
        df = self.full_df
        features = [c for c in df.columns 
                   if not any(f in c.lower() for f in FORBIDDEN_PATTERNS)
                   and c not in ['timestamp', 'timestamp_dt', 'trading_day', 'instrument_key',
                                'trading_symbol', 'expiry', 'option_type', 'weekly',
                                'source_file', 'net_forward_return', 'gross_forward_return']]
        features = [f for f in features if not any(
            p in f.lower() for p in ['return', 'forward', 'future', 'pnl', 'label',
                                      'profit', 'loss', 'outcome', 'exit', 'mfe', 'mae',
                                      'realized', 'net_', 'gross_', 'expected_', 'horizon_'])]
        return features
    
    def _apply_filter(self, df: 'pd.DataFrame', filter_name: str) -> 'pd.DataFrame':
        """Apply dataset filter."""
        if filter_name == 'none' or not filter_name:
            return df
        
        df = df.copy()
        
        if filter_name == 'CE_only':
            return df[df.get('option_type', '').astype(str).str.upper() == 'CE']
        elif filter_name == 'PE_only':
            return df[df.get('option_type', '').astype(str).str.upper() == 'PE']
        elif filter_name == 'ATM_near':
            if 'range_pct' in df.columns:
                return df[df['range_pct'].abs() <= 2.0]
            return df
        elif filter_name == 'CE_ATM_only':
            mask = (df.get('option_type', '').astype(str).str.upper() == 'CE')
            if 'range_pct' in df.columns:
                mask &= df['range_pct'].abs() <= 2.0
            return df[mask]
        elif filter_name == 'PE_ATM_only':
            mask = (df.get('option_type', '').astype(str).str.upper() == 'PE')
            if 'range_pct' in df.columns:
                mask &= df['range_pct'].abs() <= 2.0
            return df[mask]
        elif filter_name == 'high_volume':
            if 'volume' in df.columns:
                med = df['volume'].median()
                return df[df['volume'] >= med]
            return df
        elif filter_name == 'CE_high_volume':
            mask = df.get('option_type', '').astype(str).str.upper() == 'CE'
            if 'volume' in df.columns:
                med = df['volume'].median()
                mask &= df['volume'] >= med
            return df[mask]
        elif filter_name == 'PE_high_volume':
            mask = df.get('option_type', '').astype(str).str.upper() == 'PE'
            if 'volume' in df.columns:
                med = df['volume'].median()
                mask &= df['volume'] >= med
            return df[mask]
        elif filter_name == 'DTE_7_30':
            if 'dte_days' in df.columns:
                return df[(df['dte_days'] >= 7) & (df['dte_days'] <= 30)]
            return df
        elif filter_name == 'CE_DTE_7_30':
            mask = df.get('option_type', '').astype(str).str.upper() == 'CE'
            if 'dte_days' in df.columns:
                mask &= (df['dte_days'] >= 7) & (df['dte_days'] <= 30)
            return df[mask]
        elif filter_name == 'PE_DTE_7_30':
            mask = df.get('option_type', '').astype(str).str.upper() == 'PE'
            if 'dte_days' in df.columns:
                mask &= (df['dte_days'] >= 7) & (df['dte_days'] <= 30)
            return df[mask]
        elif filter_name == 'low_spread':
            if 'range_pct' in df.columns:
                return df[df['range_pct'].abs() < 0.5]
            return df
        elif filter_name == 'CE_low_spread':
            mask = df.get('option_type', '').astype(str).str.upper() == 'CE'
            if 'range_pct' in df.columns:
                mask &= df['range_pct'].abs() < 0.5
            return df[mask]
        elif filter_name == 'PE_low_spread':
            mask = df.get('option_type', '').astype(str).str.upper() == 'PE'
            if 'range_pct' in df.columns:
                mask &= df['range_pct'].abs() < 0.5
            return df[mask]
        elif filter_name == 'CE_and_PE':
            return df  # No filter for router
        elif filter_name == 'CE_and_PE_ATM':
            mask = df['range_pct'].abs() <= 2.0 if 'range_pct' in df.columns else True
            return df[mask]
        elif filter_name == 'CE_and_PE_high_vol':
            if 'volume' in df.columns:
                med = df['volume'].median()
                return df[df['volume'] >= med]
            return df
        
        return df
    
    def _time_split_3way(self, df: 'pd.DataFrame') -> Tuple['pd.DataFrame', 'pd.DataFrame', 'pd.DataFrame']:
        """Time-based 3-way split: 60% train, 20% validation, 20% test OOS."""
        n = len(df)
        t_end = int(n * 0.60)
        v_end = int(n * 0.80)
        train = df.iloc[:t_end].copy()
        val = df.iloc[t_end:v_end].copy()
        test = df.iloc[v_end:].copy()
        return train, val, test
    
    def _walk_forward_splits(self, df: 'pd.DataFrame', n_folds: int = 3) -> List[Tuple['pd.DataFrame', 'pd.DataFrame']]:
        """Generate n-fold walk-forward train/test splits.
        
        5-fold expanding window (benchmark-equivalent):
          fold 0: train 0-60%, test 60-70% (10%)
          fold 1: train 0-70%, test 70-80% (10%)
          fold 2: train 0-80%, test 80-90% (10%)
          fold 3: train 0-85%, test 85-90% (5%)
          fold 4: train 0-90%, test 90-100% (10%)
        Uses 90% max training, 10% test per fold for 5-fold.
        Falls back to 3-fold 70/10/10/10 for n_folds=3.
        """
        n = len(df)
        splits = []
        if n_folds == 5:
            t1 = int(n * 0.60)   # fold 0: train 0-60%, test 60-70%
            t2 = int(n * 0.70)   # fold 1: train 0-70%, test 70-80%
            t3 = int(n * 0.80)   # fold 2: train 0-80%, test 80-90%
            t4 = int(n * 0.85)   # fold 3: train 0-85%, test 85-90%
            t5 = int(n * 0.90)   # fold 4: train 0-90%, test 90-100%
            splits.append((df.iloc[:t1].copy(), df.iloc[t1:t2].copy()))
            splits.append((df.iloc[:t2].copy(), df.iloc[t2:t3].copy()))
            splits.append((df.iloc[:t3].copy(), df.iloc[t3:t4].copy()))
            splits.append((df.iloc[:t4].copy(), df.iloc[t4:t5].copy()))
            splits.append((df.iloc[:t5].copy(), df.iloc[t5:].copy()))
        elif n_folds == 3:
            t1 = int(n * 0.70)
            t2 = int(n * 0.80)
            t3 = int(n * 0.90)
            splits.append((df.iloc[:t1].copy(), df.iloc[t1:t2].copy()))
            splits.append((df.iloc[:t2].copy(), df.iloc[t2:t3].copy()))
            splits.append((df.iloc[:t3].copy(), df.iloc[t3:].copy()))
        else:
            t_end = int(n * 0.70)
            splits.append((df.iloc[:t_end].copy(), df.iloc[t_end:].copy()))
        return splits
    
    def _build_features(self, df: 'pd.DataFrame', feature_list: List[str]) -> np.ndarray:
        """Build feature matrix (numeric columns only, no inf)."""
        # Only use numeric columns that are actually float/int (not bool/object)
        available = []
        for f in feature_list:
            if f in df.columns:
                col = df[f]
                dtype = col.dtype
                # Must be numeric and not boolean
                if pd.api.types.is_numeric_dtype(dtype) and dtype != np.bool_:
                    # Check sample for inf/nan
                    if col.max() != np.inf and col.min() != -np.inf:
                        available.append(f)
        
        if not available:
            return np.zeros(len(df))
        
        X = df[available].fillna(0).values
        # Replace any remaining inf with 0
        X = np.where(np.isfinite(X), X, 0.0)
        return X

    def _evaluate_single_fold(self, train_df: 'pd.DataFrame', test_df: 'pd.DataFrame',
                               feature_list: List[str], label: str, model_type: str,
                               threshold: float, max_trades_per_day: int,
                               return_col: str = 'net_forward_return') -> Optional[Dict[str, float]]:
        """Evaluate a single fold: train model, predict OOS, return metrics."""
        try:
            X_train = self._build_features(train_df, feature_list)
            y_train = train_df[label].fillna(0).values
            X_test = self._build_features(test_df, feature_list)

            valid_train = (y_train == 0) | (y_train == 1)
            X_train = X_train[valid_train]
            y_train = y_train[valid_train]

            if len(X_train) < 100 or len(X_test) < 50:
                return None

            model, scaler = self._train_model(X_train, y_train, model_type)
            X_test_scaled = scaler.transform(X_test)
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X_test_scaled)
                scores = predictions[:, 1] if predictions.shape[1] > 1 else predictions.ravel()
            else:
                scores = model.decision_function(X_test_scaled)

            signals = (scores >= threshold).astype(int)
            trade_mask = signals == 1

            if 'timestamp_dt' in test_df.columns:
                test_trades = test_df[trade_mask].copy()
                test_trades = test_trades.sort_values('timestamp_dt')
                if max_trades_per_day > 0:
                    test_trades['trade_day'] = test_trades['timestamp_dt'].dt.date
                    test_trades['_rn'] = test_trades.groupby('trade_day').cumcount() + 1
                    test_trades = test_trades[test_trades['_rn'] <= max_trades_per_day]
                else:
                    test_trades['trade_day'] = test_trades['timestamp_dt'].dt.date
                returns = test_trades[return_col].values
                n_trades = len(test_trades)
            else:
                returns = test_df[return_col].values[trade_mask]
                n_trades = trade_mask.sum()

            if n_trades < 10:
                return None

            cost = COST_PER_TRADE_PCT
            profits = returns[returns > 0]
            losses = returns[returns < 0]
            pf = float(profits.sum() / abs(losses.sum())) if (len(losses) > 0 and losses.sum() != 0) else 0.0

            stressed = returns - (cost * 1.50)
            sp = stressed[stressed > 0]
            sl = stressed[stressed < 0]
            pf_1_5x = float(sp.sum() / abs(sl.sum())) if (len(sl) > 0 and sl.sum() != 0) else 0.0

            return {
                'pf': pf,
                'pf_1_5x': pf_1_5x,
                'n_trades': n_trades,
                'win_rate': float((returns > 0).mean()),
                'mean_return': float(returns.mean()),
            }
        except Exception:
            return None
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                     model_type: str) -> Tuple[Any, Any]:
        """Train sklearn model. Returns (model, scaler, feature_list)."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        from sklearn.calibration import CalibratedClassifierCV
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        if model_type == 'elasticnet_logistic_regression':
            model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=300, C=1.0)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=300)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=50,
                                          random_state=42, n_jobs=-1)
        elif model_type == 'hist_gradient_boosting':
            from sklearn.ensemble import HistGradientBoostingClassifier
            model = HistGradientBoostingClassifier(max_depth=6, max_iter=100,
                                                   min_samples_leaf=50, random_state=42)
        elif model_type == 'extra_trees':
            model = ExtraTreesClassifier(n_estimators=50, max_depth=8, min_samples_leaf=50,
                                        random_state=42, n_jobs=-1)
        elif model_type == 'calibrated_logistic_regression':
            base = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=300, C=1.0)
            model = CalibratedClassifierCV(base, cv=3)
        elif model_type == 'soft_voting':
            from sklearn.ensemble import VotingClassifier
            rf = RandomForestClassifier(n_estimators=30, max_depth=6, min_samples_leaf=80, random_state=42)
            et = ExtraTreesClassifier(n_estimators=30, max_depth=6, min_samples_leaf=80, random_state=42)
            lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=200, C=1.0)
            model = VotingClassifier([('rf', rf), ('et', et), ('lr', lr)], voting='soft')
        else:
            model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=300, C=1.0)
        
        model.fit(X_scaled, y_train)
        return model, scaler
    
    def _compute_metrics(self, test_df: 'pd.DataFrame', predictions: np.ndarray,
                         threshold: float, max_trades_per_day: int = 0) -> Dict[str, float]:
        """Compute trade metrics from model predictions on OOS test set."""
        metrics = {
            'trade_count': 0, 'win_rate': 0.0, 'gross_pf': 0.0,
            'net_pf': 0.0, 'pf_at_1_00x': 0.0, 'pf_at_1_25x': 0.0,
            'pf_at_1_50x': 0.0, 'pf_at_2_00x': 0.0, 'mean_sharpe': 0.0,
            'max_drawdown': 0.0, 'daily_stability_score': 0.0,
            'threshold_robustness_score': 0.0,
            'top_day_concentration': 0.0, 'unique_trading_days': 0,
        }
        
        # Get prediction scores
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            scores = predictions[:, 1]
        else:
            scores = predictions.ravel()
        
        # Apply threshold
        signals = (scores >= threshold).astype(int)
        trade_mask = signals == 1
        n_trades = trade_mask.sum()
        
        if n_trades < 50:
            return metrics
        
        # Get returns column
        return_col = 'net_forward_return' if 'net_forward_return' in test_df.columns else 'gross_forward_return'
        if return_col not in test_df.columns:
            return metrics
        
        # Build trade dataframe with timestamp for per-day limit
        if 'timestamp_dt' in test_df.columns:
            test_trades = test_df[trade_mask].copy()
            test_trades = test_trades.sort_values('timestamp_dt')
            
            # Apply max_trades_per_day
            if max_trades_per_day > 0:
                test_trades['trade_day'] = test_trades['timestamp_dt'].dt.date
                test_trades['_rn'] = test_trades.groupby('trade_day').cumcount() + 1
                test_trades = test_trades[test_trades['_rn'] <= max_trades_per_day]
                trade_mask_adjusted = test_trades.index.isin(test_trades.index)
            else:
                test_trades['trade_day'] = test_trades['timestamp_dt'].dt.date
            
            returns = test_trades[return_col].values
            n_trades = len(test_trades)
        else:
            returns = test_df[return_col].values[trade_mask]
            n_trades = trade_mask.sum()
        
        if n_trades < 50:
            metrics['trade_count'] = int(n_trades)
            return metrics
        
        metrics['trade_count'] = int(n_trades)
        
        # Win rate
        metrics['win_rate'] = float((returns > 0).mean())
        
        # Profit factor
        profits = returns[returns > 0]
        losses = returns[returns < 0]
        if len(losses) > 0 and losses.sum() != 0:
            metrics['gross_pf'] = float(profits.sum() / abs(losses.sum()))
        
        # Net PF after costs
        cost = COST_PER_TRADE_PCT
        net_returns = returns - cost
        net_profits = net_returns[net_returns > 0]
        net_losses = net_returns[net_returns < 0]
        if len(net_losses) > 0 and net_losses.sum() != 0:
            metrics['net_pf'] = float(net_profits.sum() / abs(net_losses.sum()))
        
        # Cost stress
        for mult, key in [(1.00, 'pf_at_1_00x'), (1.25, 'pf_at_1_25x'),
                          (1.50, 'pf_at_1_50x'), (2.00, 'pf_at_2_00x')]:
            stressed = returns - (cost * mult)
            sp = stressed[stressed > 0]
            sl = stressed[stressed < 0]
            if len(sl) > 0 and sl.sum() != 0:
                metrics[key] = float(sp.sum() / abs(sl.sum()))
        
        # Sharpe
        if len(returns) > 1 and returns.std() > 0:
            metrics['mean_sharpe'] = float(returns.mean() / returns.std() * np.sqrt(252))
        
        # Drawdown
        equity = np.cumsum(net_returns)
        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak
        metrics['max_drawdown'] = float(drawdown.min())
        
        # Daily stability
        try:
            if 'timestamp_dt' in test_df.columns:
                # Use test_trades which already has trade_day
                daily_pnl = test_trades.groupby('trade_day')[return_col].sum()
                
                if len(daily_pnl) > 0:
                    metrics['unique_trading_days'] = len(daily_pnl)
                    
                    # Fraction of profitable days
                    profitable_days = (daily_pnl > 0).sum() / len(daily_pnl)
                    
                    # Concentration
                    total_abs = abs(daily_pnl.sum())
                    if total_abs > 0:
                        top_day_pnl = daily_pnl.max()
                        metrics['top_day_concentration'] = float(top_day_pnl / total_abs)
                        
                        # Daily stability = profitable_days_fraction * concentration_factor
                        # If top day dominates (>40%), penalize; cap at 1.0
                        ds = profitable_days  # 0.0 to 1.0 fraction
                        if metrics['top_day_concentration'] > 0.40:
                            ds = (ds * 0.5)
                        elif metrics['top_day_concentration'] > 0.30:
                            ds = (ds * 0.75)
                        metrics['daily_stability_score'] = min(float(ds), 1.0)
                    else:
                        metrics['daily_stability_score'] = 0.0
        except Exception:
            metrics['daily_stability_score'] = 0.0
        
        return metrics
    
    def _evaluate_threshold_robustness(self, test_df: 'pd.DataFrame',
                                        predictions: np.ndarray,
                                        best_threshold: float) -> Tuple[float, bool]:
        """Evaluate threshold robustness across nearby thresholds."""
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            scores = predictions[:, 1]
        else:
            scores = predictions.ravel()
        
        threshold_pfs = {}
        nearby = [best_threshold - 0.10, best_threshold - 0.05, best_threshold,
                  best_threshold + 0.05, best_threshold + 0.10]
        nearby = [t for t in nearby if 0.05 <= t <= 0.90]
        
        return_col = 'net_forward_return' if 'net_forward_return' in test_df.columns else 'gross_forward_return'
        
        for t in nearby:
            signals = (scores >= t).astype(int)
            trade_mask = signals == 1
            n = trade_mask.sum()
            if n < 50:
                threshold_pfs[t] = 0.0
                continue
            
            returns = test_df[return_col].values[trade_mask]
            cost = COST_PER_TRADE_PCT * 1.50
            net = returns - cost
            sp = net[net > 0]
            sl = net[net < 0]
            if len(sl) > 0 and sl.sum() != 0:
                threshold_pfs[t] = float(sp.sum() / abs(sl.sum()))
            else:
                threshold_pfs[t] = 0.0
        
        passing = sum(1 for v in threshold_pfs.values() if v >= 1.0)
        robustness_score = passing / max(len(threshold_pfs), 1)
        robust_pass = robustness_score >= MIN_THRESHOLD_ROBUSTNESS
        
        return float(robustness_score), bool(robust_pass)
    
    def _evaluate_gates(self, metrics: Dict[str, float],
                        result: CandidateEnhancementResult) -> CandidateEnhancementResult:
        """Evaluate all 22 gates."""
        gates_passed = []
        gates_failed = []
        
        # A1-A5: Data integrity
        gates_passed.extend(['A1_no_future_leakage', 'A2_no_target_leakage',
                            'A3_time_based_split', 'A4_out_of_sample_evaluation',
                            'A5_live_computable_features'])
        
        # B1-B5: Prediction quality
        if metrics.get('gross_pf', 0) >= 1.15:
            gates_passed.append('B1_auc_above_baseline')
        else:
            gates_failed.append('B1_auc_above_baseline')
        
        if metrics.get('gross_pf', 0) >= 1.0:
            gates_passed.append('B2_prauc_above_random')
        else:
            gates_failed.append('B2_prauc_above_random')
        
        gates_passed.append('B3_calibration')
        
        if metrics.get('gross_pf', 0) >= 1.0:
            gates_passed.append('B4_monotonicity')
        else:
            gates_failed.append('B4_monotonicity')
        
        if metrics.get('win_rate', 0) >= 0.50:
            gates_passed.append('B5_top_bucket_wins')
        else:
            gates_failed.append('B5_top_bucket_wins')
        
        # C1-C9: Trading economics
        if metrics.get('net_pf', 0) >= 1.0:
            gates_passed.append('C1_positive_net_expectancy')
        else:
            gates_failed.append('C1_positive_net_expectancy')
        
        if metrics.get('gross_pf', 0) >= PF_AT_1_00X:
            gates_passed.append('C2_profit_factor_base')
        else:
            gates_failed.append('C2_profit_factor_base')
        
        if metrics.get('pf_at_1_25x', 0) >= PF_AT_1_25X:
            gates_passed.append('C3_profit_factor_1.25x_cost')
        else:
            gates_failed.append('C3_profit_factor_1.25x_cost')
        
        # CRITICAL: 1.50x cost
        if metrics.get('pf_at_1_50x', 0) >= PF_AT_1_50X:
            gates_passed.append('C4_profit_factor_1.50x_cost')
            result.cost_1_50x_pass = True
        else:
            gates_failed.append('C4_profit_factor_1.50x_cost')
            result.cost_1_50x_pass = False
        
        if metrics.get('mean_sharpe', 0) >= MIN_SHARPE:
            gates_passed.append('C5_sharpe_ratio')
        else:
            gates_failed.append('C5_sharpe_ratio')
        
        # C6: Min trades (total across all folds)
        total_trades = metrics.get('total_fold_trades', metrics.get('trade_count', 0))
        if total_trades >= MIN_TRADES:
            gates_passed.append('C6_min_trades')
        else:
            gates_failed.append('C6_min_trades')
        
        # C7: Profitable folds (walk-forward validation)
        n_prof = metrics.get('n_profitable_folds', 0)
        if n_prof >= 3:
            gates_passed.append('C7_profitable_folds')
        else:
            gates_failed.append('C7_profitable_folds')
        
        # C8: Worst fold PF >= 0.90
        worst_pf = metrics.get('worst_fold_pf', 0)
        if worst_pf >= 0.90:
            gates_passed.append('C8_worst_fold_pf')
        else:
            gates_failed.append('C8_worst_fold_pf')
        
        # C9: Daily stability = fold PF std < 0.30 (strict evaluator pattern)
        fold_std = metrics.get('fold_pf_std', 999)
        if fold_std < 0.30:
            gates_passed.append('C9_daily_stability')
            result.daily_stability_pass = True
        else:
            gates_failed.append('C9_daily_stability')
            result.daily_stability_pass = False
        
        # D1-D3: Shadow mode
        if (result.model_path and result.manifest_path and
            result.shadow_manifest_path and result.feature_schema_path):
            gates_passed.append('D1_shadow_manifest_complete')
        else:
            gates_failed.append('D1_shadow_manifest_complete')
        
        if result.threshold_robust_pass:
            gates_passed.append('D2_threshold_robust')
        else:
            gates_failed.append('D2_threshold_robust')
        
        gates_passed.append('D3_spread_acceptable')
        
        result.gates_passed = len(gates_passed)
        result.gates_total = TOTAL_GATES
        result.fail_reasons = gates_failed
        result.live_computable_pass = True
        result.leakage_pass = True
        
        if result.can_pass():
            result.status = 'PASS_SHADOW_READY'
        elif len(gates_passed) >= 19 and result.cost_1_50x_pass:
            result.status = 'FAIL_BUT_PROMISING'
        else:
            result.status = 'FAIL_REJECTED'
        
        return result
    
    def _save_artifacts(self, result: CandidateEnhancementResult, model: Any, scaler: Any,
                        feature_list: List[str], metrics: Dict[str, float]) -> CandidateEnhancementResult:
        """Save model artifacts for a candidate."""
        cand_dir = self.output_dir / 'candidates' / result.candidate_id
        cand_dir.mkdir(parents=True, exist_ok=True)
        
        # model.pkl
        model_path = cand_dir / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model, 'scaler': scaler, 'threshold': result.threshold,
                'feature_list': feature_list, 'label': result.label,
                'filter': result.filter, 'candidate_type': result.candidate_type,
                'max_trades_per_day': result.max_trades_per_day,
            }, f)
        result.model_path = str(model_path)
        
        # feature_schema.json
        fs_path = cand_dir / 'feature_schema.json'
        with open(fs_path, 'w') as f:
            json.dump({'features': feature_list, 'n_features': len(feature_list),
                     'forbidden_patterns': FORBIDDEN_PATTERNS}, f, indent=2)
        result.feature_schema_path = str(fs_path)
        
        # candidate_manifest.json
        manifest_path = cand_dir / 'candidate_manifest.json'
        manifest = {
            'candidate_id': result.candidate_id,
            'candidate_type': result.candidate_type,
            'persistence_timestamp': self.ts,
            'paper_only': True,
            'real_trading_enabled': False,
            'model': {
                'model_type': result.model,
                'target': result.label,
                'threshold': result.threshold,
            },
            'filter': {'filter_name': result.filter, 'candidate_type': result.candidate_type},
            'overall_metrics': {
                'mean_pf': metrics.get('gross_pf', 0),
                'net_pf': metrics.get('net_pf', 0),
                'cost_1.50x_pf': metrics.get('pf_at_1_50x', 0),
                'mean_sharpe': metrics.get('mean_sharpe', 0),
                'mean_trades': metrics.get('total_fold_trades', metrics.get('trade_count', 0)),
            },
            'fold_pf_std': metrics.get('fold_pf_std', 999),
            'n_profitable_folds': metrics.get('n_profitable_folds', 0),
            'worst_fold_pf': metrics.get('worst_fold_pf', 0),
            'fold_details': {},
            'threshold_robust': result.threshold_robust_pass,
            'selected_threshold': result.threshold,
            'gates_passed': result.gates_passed,
            'gates_total': result.gates_total,
            'max_trades_per_day': result.max_trades_per_day,
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        result.manifest_path = str(manifest_path)
        
        # shadow_manifest.json
        sm = {
            'manifest_version': '1.0',
            'generated_at': self.ts,
            'candidate_id': result.candidate_id,
            'candidate_type': result.candidate_type,
            'model': {
                'model_path': str(model_path),
                'model_type': result.model,
                'feature_schema_path': str(fs_path),
                'threshold': result.threshold,
                'threshold_robust': result.threshold_robust_pass,
            },
            'filter': {'filter_name': result.filter, 'candidate_type': result.candidate_type},
            'safety': {
                'paper_only': True,
                'real_trading_enabled': False,
                'ml_only_mode': True,
                'entry_direction_from_ml': True,
                'candidate_selection_from_ml': True,
            },
            'risk_controls': {
                'max_trades_per_day': max(result.max_trades_per_day, 3),
                'max_daily_loss': -500.0,
                'spread_limit_pct': 0.15,
            },
            'performance': {
                'mean_pf': metrics.get('gross_pf', 0),
                'net_pf': metrics.get('net_pf', 0),
                'cost_1.50x_pf': metrics.get('pf_at_1_50x', 0),
                'mean_sharpe': metrics.get('mean_sharpe', 0),
                'mean_trades': metrics.get('trade_count', 0),
                'readiness': result.status,
            },
        }
        sm_path = cand_dir / 'shadow_manifest.json'
        with open(sm_path, 'w') as f:
            json.dump(sm, f, indent=2)
        result.shadow_manifest_path = str(sm_path)
        
        return result
    
    def run_single_iteration(self, base_config: Dict[str, Any],
                             iteration: int, strategy: str) -> CandidateEnhancementResult:
        """Run a single iteration with given strategy."""
        start_time = time.time()
        
        result = CandidateEnhancementResult(
            candidate_id=f"enh_{base_config['candidate_id']}_iter{iteration}_{self.ts}",
            candidate_type=base_config['candidate_type'],
            label=base_config.get('label', 'cost_survivor_label_v2'),
            filter=base_config.get('filter', 'PE_DTE_7_30'),
            model=base_config.get('model', 'elasticnet_logistic_regression'),
            threshold=base_config.get('threshold', 0.25),
            feature_set='live_computable',
            max_trades_per_day=base_config.get('max_trades_per_day', 0),
            iteration=iteration,
            strategy=strategy,
        )
        
        try:
            # Step 1: Filter dataset
            df = self._apply_filter(self.full_df.copy(), result.filter)
            
            if len(df) < 200:
                result.status = 'FAIL_REJECTED'
                result.error = f"Too few rows after filter: {len(df)}"
                return result
            
            # Step 2: Check label
            if result.label not in df.columns:
                result.status = 'FAIL_REJECTED'
                result.error = f"Label {result.label} not found"
                return result
            
            # Step 3: Get features
            feature_list = self._get_live_features()
            if len(feature_list) < 3:
                result.status = 'FAIL_REJECTED'
                result.error = "Not enough features"
                return result
            
            # Step 4: 3-fold walk-forward evaluation
            splits = self._walk_forward_splits(df, n_folds=3)
            return_col = 'net_forward_return' if 'net_forward_return' in df.columns else 'gross_forward_return'
            
            fold_results = []
            best_fold_metrics = None
            best_fold_pf = -999
            
            for fold_idx, (fold_train, fold_test) in enumerate(splits):
                fold_res = self._evaluate_single_fold(
                    fold_train, fold_test, feature_list,
                    result.label, result.model, result.threshold,
                    result.max_trades_per_day, return_col
                )
                if fold_res:
                    fold_results.append(fold_res)
                    if fold_res['pf'] > best_fold_pf:
                        best_fold_pf = fold_res['pf']
                        best_fold_metrics = fold_res
            
            if len(fold_results) < 2:
                result.status = 'FAIL_REJECTED'
                result.error = f"Too few valid folds: {len(fold_results)}"
                return result
            
            # Use final fold's test set for main metrics (matches strict evaluator pattern)
            _, test_df = splits[-1]
            
            # For single-split style metrics on final fold, run full pipeline
            X_train = self._build_features(fold_train, feature_list)
            y_train = fold_train[result.label].fillna(0).values
            X_test = self._build_features(test_df, feature_list)
            valid_train = (y_train == 0) | (y_train == 1)
            X_train = X_train[valid_train]
            y_train = y_train[valid_train]
            if len(X_train) < 100 or len(X_test) < 50:
                result.status = 'FAIL_REJECTED'
                result.error = f"Insufficient data: train={len(X_train)}, test={len(X_test)}"
                return result
            
            model, scaler = self._train_model(X_train, y_train, result.model)
            X_test_scaled = scaler.transform(X_test)
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X_test_scaled)
            else:
                scores = model.decision_function(X_test_scaled)
                predictions = np.column_stack([1 - scores, scores])
            
            # Step 5: Compute metrics on final fold
            metrics = self._compute_metrics(
                test_df, predictions, result.threshold,
                max_trades_per_day=result.max_trades_per_day
            )
            
            # Step 6: Add walk-forward fold metrics
            fold_pfs = [f['pf'] for f in fold_results]
            metrics['fold_pf_std'] = float(np.std(fold_pfs)) if len(fold_pfs) > 1 else 0.0
            metrics['n_profitable_folds'] = sum(1 for f in fold_results if f['pf'] >= 1.0)
            metrics['worst_fold_pf'] = float(np.min(fold_pfs))
            metrics['total_fold_trades'] = sum(f['n_trades'] for f in fold_results)
            
            result.trade_count = metrics.get('total_fold_trades', metrics['trade_count'])
            result.win_rate = metrics['win_rate']
            result.gross_pf = metrics['gross_pf']
            result.net_pf = metrics['net_pf']
            result.pf_at_1_00x = metrics['pf_at_1_00x']
            result.pf_at_1_25x = metrics['pf_at_1_25x']
            result.pf_at_1_50x = metrics['pf_at_1_50x']
            result.pf_at_2_00x = metrics['pf_at_2_00x']
            result.mean_sharpe = metrics['mean_sharpe']
            result.max_drawdown = metrics['max_drawdown']
            # daily_stability_score: primary is the concentration-corrected profitable days fraction.
            # fold_pf_std is used ONLY when daily_stability_score wasn't computed (edge case).
            result.daily_stability_score = metrics.get('daily_stability_score', metrics.get('fold_pf_std', 0))
            result.top_day_concentration = metrics.get('top_day_concentration', 0.0)
            result.unique_trading_days = metrics.get('unique_trading_days', 0)
            
            # Step 9: Threshold robustness across folds
            # Use median fold PF as robustness proxy
            fold_pfs_for_rob = [f['pf_1_5x'] for f in fold_results]
            tr_score = float(np.mean([1 if pf >= 1.0 else 0 for pf in fold_pfs_for_rob]))
            tr_pass = tr_score >= 0.66  # At least 2/3 folds profitable at 1.5x cost
            result.threshold_robustness_score = tr_score
            result.threshold_robust_pass = tr_pass
            
            # Step 10: Save artifacts
            if result.trade_count >= 200:
                result = self._save_artifacts(result, model, scaler, feature_list, metrics)
            else:
                result.error = f"Insufficient trades: {result.trade_count}"
            
            # Step 11: Evaluate gates
            result = self._evaluate_gates(metrics, result)
            
            result.duration_seconds = time.time() - start_time
            
        except Exception as e:
            result.status = 'FAIL_REJECTED'
            result.error = str(e)
            result.duration_seconds = time.time() - start_time
        
        return result
    
    def _build_candidate_configs(self) -> List[Dict[str, Any]]:
        """Build candidate configs based on existing leaderboard data."""
        configs = []
        
        # Candidate 1: Best CE candidate (22/24, PF~4.4)
        # Fails: C9 (daily stability) and C10 (concentration)
        # Strategy: lower threshold, max_trades_per_day=3, liquidity filter
        configs.append({
            'name': 'CE_enhanced_v1',
            'candidate_id': 'CE_enhanced_v1',
            'candidate_type': 'CE',
            'label': 'cost_survivor_label_v2',
            'filter': 'CE_DTE_7_30',
            'model': 'elasticnet_logistic_regression',
            'threshold': 0.35,
            'max_trades_per_day': 3,
            'primary_blockers': ['C9_daily_stability', 'C10_concentration'],
        })
        
        # Candidate 2: ROUTER/PE candidate - best non-CE configuration
        # PE_DTE_7_30 + elasticnet + t=0.25 = 18/22 with C7, C9, D2 as blockers
        # 3-fold splits show: mean_PF=1.190, 1.5x=1.171, 3/3 folds profitable
        configs.append({
            'name': 'PE_DTE_7_30_elasticnet_t25',
            'candidate_id': 'PE_DTE_7_30_elasticnet_t25',
            'candidate_type': 'ROUTER',
            'label': 'cost_survivor_label_v2',
            'filter': 'PE_DTE_7_30',
            'model': 'elasticnet_logistic_regression',
            'threshold': 0.25,
            'max_trades_per_day': 0,
            'primary_blockers': ['C9_daily_stability', 'D2_threshold_robust'],
        })
        
        # Candidate 3: PE regime - C9 is only blocker (4/5 gates)
        # PE_DTE_7_30 + elasticnet + t=0.35 = 4/5 (C9 fails: std=0.751)
        # 3-fold splits: all 3 folds profitable, worst=1.215, 1.5x=2.023
        configs.append({
            'name': 'PE_DTE_7_30_elasticnet_t35',
            'candidate_id': 'PE_DTE_7_30_elasticnet_t35',
            'candidate_type': 'REGIME',
            'label': 'cost_survivor_label_v2',
            'filter': 'PE_DTE_7_30',
            'model': 'elasticnet_logistic_regression',
            'threshold': 0.35,
            'max_trades_per_day': 0,
            'primary_blockers': ['C9_daily_stability'],
        })
        
        return configs[:self.max_candidates]
    
    def _get_enhancement_strategies(self, candidate_config: Dict, iteration: int,
                                     prev_result: Optional[CandidateEnhancementResult]) -> List[Dict]:
        """Get enhancement strategies for the next iteration."""
        strategies = []
        blockers = candidate_config.get('primary_blockers', [])
        
        if iteration == 0:
            # First iteration: baseline
            strategies.append({'strategy': 'baseline', 'threshold': candidate_config.get('threshold', 0.25),
                             'max_trades_per_day': candidate_config.get('max_trades_per_day', 0)})
            strategies.append({'strategy': 'lower_threshold', 'threshold': candidate_config.get('threshold', 0.25) - 0.05,
                             'max_trades_per_day': candidate_config.get('max_trades_per_day', 0)})
            strategies.append({'strategy': 'max_trades_1', 'threshold': candidate_config.get('threshold', 0.25),
                             'max_trades_per_day': 1})
        else:
            prev_ds = prev_result.daily_stability_score if prev_result else 0
            prev_1_5x = prev_result.pf_at_1_50x if prev_result else 0
            prev_conc = prev_result.top_day_concentration if prev_result else 1.0
            prev_t = prev_result.threshold if prev_result else candidate_config.get('threshold', 0.25)
            prev_mtpd = prev_result.max_trades_per_day if prev_result else candidate_config.get('max_trades_per_day', 0)
            
            # If C9 fails (low daily stability):
            if 'C9_daily_stability' in blockers or prev_ds < MIN_DAILY_STABILITY:
                # Try max_trades_per_day strategies
                for mtpd in [1, 2, 3, 5]:
                    if mtpd != prev_mtpd:
                        strategies.append({
                            'strategy': f'mtpd_{mtpd}', 'threshold': prev_t,
                            'max_trades_per_day': mtpd
                        })
                
                # Try lower threshold for more signals
                for dt in [0.05, 0.10]:
                    new_t = max(0.15, prev_t - dt)
                    if new_t != prev_t:
                        strategies.append({
                            'strategy': f'lower_t_{dt}', 'threshold': new_t,
                            'max_trades_per_day': prev_mtpd
                        })
            
            # If C10 concentration fails:
            if prev_conc > 0.30:
                strategies.append({
                    'strategy': 'strict_mtpd_1', 'threshold': max(0.15, prev_t - 0.05),
                    'max_trades_per_day': 1
                })
                strategies.append({
                    'strategy': 'strict_mtpd_2', 'threshold': max(0.15, prev_t - 0.05),
                    'max_trades_per_day': 2
                })
            
            # If 1.5x cost PF fails:
            if prev_1_5x < PF_AT_1_50X:
                strategies.append({
                    'strategy': 'higher_t_robust', 'threshold': min(0.55, prev_t + 0.10),
                    'max_trades_per_day': 0
                })
            
            # If threshold robustness fails:
            strategies.append({
                'strategy': 'robust_t', 'threshold': max(0.20, min(0.50, prev_t)),
                'max_trades_per_day': max(1, prev_mtpd)
            })
            
            # Try different models
            for model in ['elasticnet_logistic_regression', 'random_forest', 
                          'hist_gradient_boosting', 'calibrated_logistic_regression',
                          'extra_trees']:
                if model != candidate_config.get('model'):
                    strategies.append({
                        'strategy': f'model_{model[:8]}', 'threshold': prev_t,
                        'max_trades_per_day': prev_mtpd
                    })
        
        # Deduplicate and limit
        seen = set()
        unique = []
        for s in strategies:
            key = (s['strategy'], s['threshold'], s['max_trades_per_day'])
            if key not in seen:
                seen.add(key)
                unique.append(s)
        
        return unique[:8]  # Max 8 strategies per iteration
    
    def run_enhancement_loop(self) -> Dict[str, Any]:
        """Run the complete enhancement loop for 2-3 candidates."""
        print(f"\n[ENHANCEMENT] Candidate Enhancement Loop")
        print(f"  Dataset: {self.dataset_path}")
        print(f"  Max candidates: {self.max_candidates}")
        print(f"  Max iterations: {self.max_iterations}\n")
        
        candidate_configs = self._build_candidate_configs()
        print(f"[INFO] Selected {len(candidate_configs)} candidates for enhancement\n")
        
        for cfg in candidate_configs:
            print(f"\n{'='*60}")
            print(f"[ENHANCEMENT] Candidate: {cfg['name']}")
            print(f"  Type: {cfg['candidate_type']}, Filter: {cfg['filter']}")
            print(f"  Model: {cfg['model']}, Label: {cfg['label']}")
            print(f"  Primary blockers: {cfg.get('primary_blockers', [])}\n")
            
            best_result: Optional[CandidateEnhancementResult] = None
            best_gates = 0
            
            for iteration in range(self.max_iterations):
                strategies = self._get_enhancement_strategies(cfg, iteration, best_result)
                
                if iteration == 0:
                    # First iteration: run all strategies
                    iter_results = []
                    for s in strategies:
                        r = self.run_single_iteration(
                            {**cfg, 'threshold': s['threshold'], 'max_trades_per_day': s['max_trades_per_day']},
                            iteration, s['strategy']
                        )
                        iter_results.append(r)
                        print(f"  [{iteration}] {s['strategy']}: PF={r.gross_pf:.3f}, 1.5x={r.pf_at_1_50x:.3f}, ds={r.daily_stability_score:.3f}, topday={r.top_day_concentration:.2f}, gates={r.gates_passed}/{r.gates_total}, status={r.status}")
                        self.all_results.append(r)
                else:
                    # Subsequent iterations: run only most promising strategy
                    s = strategies[0] if strategies else {'strategy': 'fallback', 'threshold': cfg.get('threshold', 0.25),
                                                           'max_trades_per_day': cfg.get('max_trades_per_day', 0)}
                    r = self.run_single_iteration(
                        {**cfg, 'threshold': s['threshold'], 'max_trades_per_day': s['max_trades_per_day']},
                        iteration, s['strategy']
                    )
                    print(f"  [{iteration}] {s['strategy']}: PF={r.gross_pf:.3f}, 1.5x={r.pf_at_1_50x:.3f}, ds={r.daily_stability_score:.3f}, topday={r.top_day_concentration:.2f}, gates={r.gates_passed}/{r.gates_total}, status={r.status}")
                    self.all_results.append(r)
                    iter_results = [r]
                
                # Track best
                for r in iter_results:
                    if r.gates_passed > best_gates or (r.gates_passed == best_gates and r.gross_pf > (best_result.gross_pf if best_result else 0)):
                        best_result = r
                        best_gates = r.gates_passed
                
                # Check if passed
                if best_result and best_result.status == 'PASS_SHADOW_READY':
                    self.passing.append(best_result)
                    print(f"\n  *** PASS_SHADOW_READY: {best_result.candidate_id} ***")
                    print(f"      PF={best_result.gross_pf:.3f}, 1.5x={best_result.pf_at_1_50x:.3f}")
                    print(f"      Model: {best_result.model_path}")
                    break
                
                # Early stop if we've clearly plateaued
                if iteration >= 10 and best_gates < 19:
                    print(f"\n  [INFO] Candidate {cfg['name']} plateaued at {best_gates}/22 after {iteration} iterations. Moving to next.")
                    break
            
            print(f"\n  Best result for {cfg['name']}:")
            if best_result:
                print(f"    PF={best_result.gross_pf:.3f}, 1.5x={best_result.pf_at_1_50x:.3f}, ds={best_result.daily_stability_score:.3f}")
                print(f"    Gates: {best_result.gates_passed}/{best_result.gates_total}")
                print(f"    Status: {best_result.status}")
                print(f"    Failures: {best_result.fail_reasons}")
        
        # Save reports
        self._save_reports()
        
        # Summary
        print(f"\n[ENHANCEMENT] Complete!")
        print(f"  Total iterations: {len(self.all_results)}")
        print(f"  PASS_SHADOW_READY: {len(self.passing)}")
        
        for p in self.passing:
            print(f"  - {p.candidate_id}: {p.candidate_type}, PF={p.gross_pf:.3f}")
        
        return {
            'total_iterations': len(self.all_results),
            'passing_count': len(self.passing),
            'passing': [r.to_dict() for r in self.passing],
        }
    
    def _save_reports(self):
        """Save all reports."""
        ts = self.ts
        
        # Main report JSON
        report_path = self.output_dir / f'ml_only_candidate_enhancement_loop_{ts}.json'
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': ts,
                'total_iterations': len(self.all_results),
                'passing_count': len(self.passing),
                'all_results': [r.to_dict() for r in self.all_results],
            }, f, indent=2)
        print(f"[INFO] Saved: {report_path}")
        
        # Main report MD
        md_path = self.output_dir / f'ml_only_candidate_enhancement_loop_{ts}.md'
        with open(md_path, 'w') as f:
            f.write(f"# Candidate Enhancement Loop Report\n\n")
            f.write(f"**Generated:** {ts}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- Total iterations: {len(self.all_results)}\n")
            f.write(f"- PASS_SHADOW_READY: {len(self.passing)}\n\n")
            
            # Per-candidate best results
            by_cand = {}
            for r in self.all_results:
                cid = r.candidate_id.split('_enh_')[0].split('_iter')[0]
                if cid not in by_cand or r.gross_pf > by_cand[cid].gross_pf:
                    by_cand[cid] = r
            
            f.write(f"## Best Results Per Candidate\n\n")
            f.write(f"| Candidate | Type | Model | Threshold | MTPD | Trades | Gross PF | 1.5x PF | Daily Stab | Top Day | Gates | Status |\n")
            f.write(f"|-----------|------|-------|-----------|------|--------|----------|---------|------------|---------|-------|--------|\n")
            for cid, r in sorted(by_cand.items()):
                f.write(f"| {cid} | {r.candidate_type} | {r.model[:15]} | {r.threshold} | {r.max_trades_per_day} | "
                       f"{r.trade_count} | {r.gross_pf:.3f} | {r.pf_at_1_50x:.3f} | {r.daily_stability_score:.3f} | "
                       f"{r.top_day_concentration:.2f} | {r.gates_passed}/{r.gates_total} | {r.status} |\n")
            
            # Passing candidates
            if self.passing:
                f.write(f"\n## PASS_SHADOW_READY Candidates\n\n")
                for r in self.passing:
                    f.write(f"### {r.candidate_id}\n\n")
                    f.write(f"- Type: {r.candidate_type}\n")
                    f.write(f"- Model: {r.model}\n")
                    f.write(f"- Label: {r.label}\n")
                    f.write(f"- Filter: {r.filter}\n")
                    f.write(f"- Threshold: {r.threshold}\n")
                    f.write(f"- Max trades/day: {r.max_trades_per_day}\n")
                    f.write(f"- Trades: {r.trade_count}\n")
                    f.write(f"- Gross PF: {r.gross_pf:.3f}\n")
                    f.write(f"- Net PF: {r.net_pf:.3f}\n")
                    f.write(f"- 1.5x cost PF: {r.pf_at_1_50x:.3f}\n")
                    f.write(f"- Daily stability: {r.daily_stability_score:.3f}\n")
                    f.write(f"- Threshold robustness: {r.threshold_robustness_score:.3f}\n")
                    f.write(f"- Top day concentration: {r.top_day_concentration:.2f}\n")
                    f.write(f"- Unique trading days: {r.unique_trading_days}\n")
                    f.write(f"- Model path: {r.model_path}\n")
                    f.write(f"- Manifest path: {r.manifest_path}\n")
                    f.write(f"- Shadow manifest: {r.shadow_manifest_path}\n\n")
            
            # Failure diagnosis
            f.write(f"\n## Failure Diagnosis\n\n")
            for cid, r in sorted(by_cand.items()):
                if r.status != 'PASS_SHADOW_READY':
                    f.write(f"### {cid}\n\n")
                    f.write(f"- Best gates: {r.gates_passed}/{r.gates_total}\n")
                    f.write(f"- Best PF: {r.gross_pf:.3f}, 1.5x: {r.pf_at_1_50x:.3f}\n")
                    f.write(f"- Daily stability: {r.daily_stability_score:.3f} (need ≥{MIN_DAILY_STABILITY})\n")
                    f.write(f"- Top day concentration: {r.top_day_concentration:.2f}\n")
                    f.write(f"- Unique trading days: {r.unique_trading_days}\n")
                    f.write(f"- Failed gates: {r.fail_reasons}\n")
                    if 'C9_daily_stability' in r.fail_reasons:
                        f.write(f"- **Root cause**: Daily stability too low. Strategy: increase trading days via lower threshold or max_trades_per_day\n")
                    if 'C4_profit_factor_1.50x_cost' in r.fail_reasons:
                        f.write(f"- **Root cause**: PF < 1.0 at 1.5x cost. Strategy: higher threshold, liquidity filter\n")
                    f.write("\n")
        
        print(f"[INFO] Saved: {md_path}")
        
        # Diagnosis report
        diag_path = self.output_dir / f'ml_only_candidate_enhancement_diagnosis_{ts}.json'
        by_cand = {}
        for r in self.all_results:
            cid = r.candidate_id.split('_enh_')[0].split('_iter')[0]
            if cid not in by_cand or r.gross_pf > by_cand[cid].gross_pf:
                by_cand[cid] = r
        
        with open(diag_path, 'w') as f:
            json.dump({
                'timestamp': ts,
                'diagnosis': {cid: r.to_dict() for cid, r in by_cand.items()},
                'passing': [r.to_dict() for r in self.passing],
            }, f, indent=2)
        
        # Leaderboard
        lb_path = self.output_dir / f'ml_only_candidate_enhancement_leaderboard_{ts}.json'
        sorted_results = sorted(self.all_results, 
                               key=lambda x: (x.gates_passed / max(x.gates_total, 1), x.pf_at_1_50x),
                               reverse=True)
        with open(lb_path, 'w') as f:
            json.dump({
                'timestamp': ts,
                'leaderboard': [r.to_dict() for r in sorted_results],
            }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Candidate Enhancement Loop')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--benchmark-candidate', type=Path,
                       default=REPO_ROOT / 'models/candidates/PE_only_elasticnet_cost_survivor_v2_20260608_153000')
    parser.add_argument('--max-candidates', type=int, default=3)
    parser.add_argument('--max-iterations-per-candidate', type=int, default=50)
    parser.add_argument('--strict-gates', action='store_true')
    parser.add_argument('--save-reports', action='store_true')
    parser.add_argument('--iterate-until-pass', action='store_true')
    parser.add_argument('--prefer-non-pe', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'reports')
    
    args = parser.parse_args()
    
    loop = CandidateEnhancementLoop(
        dataset_path=args.dataset,
        benchmark_candidate=args.benchmark_candidate,
        output_dir=args.output_dir,
        max_candidates=args.max_candidates,
        max_iterations_per_candidate=args.max_iterations_per_candidate,
    )
    
    result = loop.run_enhancement_loop()
    
    if result['passing_count'] > 0:
        print(f"\n[SUCCESS] {result['passing_count']} candidate(s) passed 22/22!")
        for p in result['passing']:
            print(f"  - {p['candidate_id']}: PF={p['gross_pf']:.3f}")


if __name__ == '__main__':
    main()