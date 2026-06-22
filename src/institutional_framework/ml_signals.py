"""Consensus Ensemble Signal Generator with Platt Scaling Calibration.

Combines Logistic Regression, RandomForest, and XGBoost with Platt Scaling
probability calibration and dynamic weight routing based on classified market regimes.
"""

from __future__ import annotations

import os
import pickle
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class EnsembleSignal:
    consensus_prob: float
    raw_probs: Dict[str, float]
    regime: str
    is_vetoed: bool
    signal_direction: int  # 1 for buy call, -1 for buy put, 0 for neutral

class DynamicConsensusEnsemble:
    """Heterogeneous ensemble with Platt Scaling and dynamic weights."""
    
    def __init__(self, **kwargs):
        self.lr_params = {
            'penalty': 'elasticnet',
            'solver': 'saga',
            'l1_ratio': 0.40,
            'C': 0.05,
            'max_iter': 500,
            'random_state': 42
        }
        self.rf_params = {
            'n_estimators': 50,
            'max_depth': 4,
            'min_samples_leaf': 20,
            'max_features': 'sqrt',
            'bootstrap': True,
            'n_jobs': 2,
            'random_state': 42
        }
        self.xgb_params = {
            'n_estimators': 60,
            'max_depth': 3,
            'learning_rate': 0.04,
            'subsample': 0.7,
            'colsample_bytree': 0.6,
            'tree_method': 'hist',
            'max_bin': 64,
            'random_state': 42,
            'n_jobs': 2
        }
        
        self.lr_model: Any = None
        self.rf_model: Any = None
        self.xgb_model: Any = None
        self.use_calibration = True

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Fits base estimators wrapped inside Sigmoid CalibratedClassifierCV."""
        n_samples = len(X)
        unique_classes, counts = np.unique(y, return_counts=True)
        
        # Platt Scaling (CalibratedClassifierCV) requires a sufficient sample footprint
        self.use_calibration = bool(n_samples >= 30 and len(unique_classes) >= 2 and min(counts) >= 5)
        
        # 1. Fit regularized Logistic Regression
        try:
            from sklearn.linear_model import LogisticRegression
            base_lr = LogisticRegression(**self.lr_params)
            if self.use_calibration:
                from sklearn.calibration import CalibratedClassifierCV
                self.lr_model = CalibratedClassifierCV(estimator=base_lr, method='sigmoid', cv=3)
            else:
                self.lr_model = base_lr
            self.lr_model.fit(X, y, sample_weight=sample_weight)
        except Exception:
            self.lr_model = None

        # 2. Fit regularized Random Forest
        try:
            from sklearn.ensemble import RandomForestClassifier
            base_rf = RandomForestClassifier(**self.rf_params)
            if self.use_calibration:
                from sklearn.calibration import CalibratedClassifierCV
                self.rf_model = CalibratedClassifierCV(estimator=base_rf, method='sigmoid', cv=3)
            else:
                self.rf_model = base_rf
            self.rf_model.fit(X, y, sample_weight=sample_weight)
        except Exception:
            self.rf_model = None

        # 3. Fit hist-binned XGBoost (with sklearn fallback when xgboost is unavailable)
        try:
            import xgboost as xgb
            base_xgb = xgb.XGBClassifier(**self.xgb_params)
        except Exception:
            try:
                from sklearn.ensemble import GradientBoostingClassifier

                base_xgb = GradientBoostingClassifier(random_state=42)
            except Exception:
                base_xgb = None
        try:
            if base_xgb is None:
                raise RuntimeError("No XGBoost-compatible estimator available")
            if self.use_calibration:
                from sklearn.calibration import CalibratedClassifierCV
                self.xgb_model = CalibratedClassifierCV(estimator=base_xgb, method='sigmoid', cv=3)
            else:
                self.xgb_model = base_xgb
            self.xgb_model.fit(X, y, sample_weight=sample_weight)
        except Exception:
            self.xgb_model = None
            
        return self

    def predict_consensus(
        self,
        X: np.ndarray,
        w_lr: float,
        w_rf: float,
        w_xgb: float,
        tau_upper: float = 0.62,
        tau_lower: float = 0.38,
        regime_name: str = "CHOPPY"
    ) -> EnsembleSignal:
        """Evaluates dynamic consensus probabilities and gates signals with corridors."""
        p_lr = 0.5
        p_rf = 0.5
        p_xgb = 0.5
        
        # Predict individual class 1 probabilities
        if self.lr_model is not None:
            try:
                p_lr = float(self.lr_model.predict_proba(X)[0][1])
            except Exception:
                pass
                
        if self.rf_model is not None:
            try:
                p_rf = float(self.rf_model.predict_proba(X)[0][1])
            except Exception:
                pass
                
        if self.xgb_model is not None:
            try:
                p_xgb = float(self.xgb_model.predict_proba(X)[0][1])
            except Exception:
                pass
                
        # 1. Dynamic weighted average probability
        total_w = w_lr + w_rf + w_xgb
        if total_w > 0:
            consensus_prob = (p_lr * w_lr + p_rf * w_rf + p_xgb * w_xgb) / total_w
        else:
            consensus_prob = 0.5
            
        # 2. Consensus Corridor Gating (Veto Gating)
        is_vetoed = False
        signal_direction = 0
        
        # If weights are disabled (e.g. NEWS_EVENT), immediately veto the trade
        if total_w <= 1e-6:
            is_vetoed = True
        elif consensus_prob >= tau_upper:
            signal_direction = 1
        elif consensus_prob <= tau_lower:
            signal_direction = -1
        else:
            is_vetoed = True  # Signal lies in the uncertain neutral corridor
            
        return EnsembleSignal(
            consensus_prob=round(consensus_prob, 4),
            raw_probs={
                "lr": round(p_lr, 4),
                "rf": round(p_rf, 4),
                "xgb": round(p_xgb, 4)
            },
            regime=regime_name,
            is_vetoed=is_vetoed,
            signal_direction=signal_direction
        )
