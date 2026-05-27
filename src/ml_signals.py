"""Ensemble ML signal scaffolding with feature pipeline and walk-forward metrics."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import os
import logging

from ml_pipeline import MLFeatureContext, build_market_feature_vector, build_supervised_dataset, walk_forward_backtest

try:
    from sklearn.ensemble import RandomForestClassifier
except Exception:
    try:
        from ml_pipeline import FallbackClassifier
        RandomForestClassifier = FallbackClassifier
    except Exception:
        RandomForestClassifier = None


MODEL_PATH = "ml_signal_model.pkl"


@dataclass
class MLModelBundle:
    model: Any
    feature_names: List[str]
    metrics: Dict[str, Any]
    trained_at: str
    scaler_mean: Optional[List[float]] = None
    scaler_std: Optional[List[float]] = None


def _coerce_matrix(X: Any) -> List[List[float]]:
    try:
        if hasattr(X, "to_numpy"):
            return [[float(v) for v in row] for row in X.to_numpy().tolist()]
        return [[float(v) for v in row] for row in list(X)]
    except Exception:
        return []


def _wrap_model(
    model: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    scaler_mean: Optional[List[float]] = None,
    scaler_std: Optional[List[float]] = None,
) -> MLModelBundle:
    return MLModelBundle(
        model=model,
        feature_names=list(feature_names or []),
        metrics=dict(metrics or {}),
        trained_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )


def train_ensemble(
    X: Any,
    y: Sequence[int],
    save_path: Optional[str] = None,
    *,
    feature_names: Optional[Sequence[str]] = None,
    walk_forward_splits: int = 5,
    walk_forward_threshold: float = 0.5,
    use_time_decay: bool = True,
    decay_lambda: float = 0.005,
) -> Optional[Any]:
    if RandomForestClassifier is None:
        return None
    try:
        X_mat = _coerce_matrix(X)
        y_list = [int(v) for v in list(y)]
        if not X_mat or len(X_mat) != len(y_list):
            return None
            
        n_samples = len(X_mat)
        n_features = len(X_mat[0]) if n_samples > 0 else 0
        
        # Calculate standard scaling metrics
        scaler_mean = []
        scaler_std = []
        import math
        if n_samples > 0 and n_features > 0:
            for j in range(n_features):
                col = [X_mat[i][j] for i in range(n_samples)]
                m = sum(col) / n_samples
                scaler_mean.append(m)
                variance = sum((val - m) ** 2 for val in col) / n_samples
                s = math.sqrt(variance)
                scaler_std.append(s if s > 1e-9 else 1e-9)
                
            # Transform dataset
            X_scaled = []
            for i in range(n_samples):
                row = []
                for j in range(n_features):
                    row.append((X_mat[i][j] - scaler_mean[j]) / scaler_std[j])
                X_scaled.append(row)
        else:
            X_scaled = X_mat
            scaler_mean = None
            scaler_std = None

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Calculate Exponential Time-Decay weights if enabled
        sample_weights = None
        if use_time_decay and n_samples > 0:
            sample_weights = []
            for i in range(n_samples):
                w = math.exp(-decay_lambda * (n_samples - 1 - i))
                sample_weights.append(w)
                
        # Train model with optional sample weighting
        if sample_weights is not None:
            import inspect
            sig = inspect.signature(clf.fit)
            if "sample_weight" in sig.parameters:
                clf.fit(X_scaled, y_list, sample_weight=sample_weights)
            else:
                clf.fit(X_scaled, y_list)
        else:
            clf.fit(X_scaled, y_list)
            
        metrics: Dict[str, Any] = {}
        try:
            # Use scaled features in walk-forward evaluation
            wf = walk_forward_backtest(X_scaled, y_list, n_splits=walk_forward_splits, threshold=walk_forward_threshold)
            metrics = dict(wf.get("aggregate") or {})
            metrics["walk_forward_folds"] = wf.get("folds") or []
        except Exception:
            metrics = {}
            
        path = save_path or MODEL_PATH
        bundle = _wrap_model(
            clf,
            feature_names=feature_names,
            metrics=metrics,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
        )
        try:
            import joblib
            joblib.dump(bundle, path)
        except Exception:
            pass
        return bundle
    except Exception:
        return None


def load_model(path: Optional[str] = None) -> Optional[Any]:
    path = path or MODEL_PATH
    try:
        import joblib

        loaded = joblib.load(path)
        if isinstance(loaded, MLModelBundle):
            return loaded
        if isinstance(loaded, dict) and "model" in loaded:
            return MLModelBundle(
                model=loaded.get("model"),
                feature_names=list(loaded.get("feature_names") or []),
                metrics=dict(loaded.get("metrics") or {}),
                trained_at=str(loaded.get("trained_at") or ""),
                scaler_mean=loaded.get("scaler_mean"),
                scaler_std=loaded.get("scaler_std"),
            )
        return MLModelBundle(model=loaded, feature_names=[], metrics={}, trained_at="legacy")
    except Exception:
        return None


def predict(model: Any, X: List[List[float]]) -> List[float]:
    bundle = model
    scaler_mean = None
    scaler_std = None
    if isinstance(bundle, MLModelBundle):
        model = bundle.model
        scaler_mean = bundle.scaler_mean
        scaler_std = bundle.scaler_std
    elif isinstance(bundle, dict) and "model" in bundle:
        model = bundle.get("model")
        scaler_mean = bundle.get("scaler_mean")
        scaler_std = bundle.get("scaler_std")
    if model is None:
        return [0.0 for _ in X]
        
    try:
        # Scale X if scaler is present
        X_scaled = []
        if scaler_mean and scaler_std and len(X) > 0 and len(X[0]) == len(scaler_mean):
            for row in X:
                scaled_row = [(row[j] - scaler_mean[j]) / scaler_std[j] for j in range(len(row))]
                X_scaled.append(scaled_row)
        else:
            X_scaled = X

        probs = model.predict_proba(X_scaled)
        # assume positive class at index 1
        return [float(p[1]) for p in probs]
    except Exception:
        try:
            X_scaled = []
            if scaler_mean and scaler_std and len(X) > 0 and len(X[0]) == len(scaler_mean):
                for row in X:
                    scaled_row = [(row[j] - scaler_mean[j]) / scaler_std[j] for j in range(len(row))]
                    X_scaled.append(scaled_row)
            else:
                X_scaled = X
            preds = model.predict(X_scaled)
            return [float(x) for x in preds]
        except Exception:
            return [0.0 for _ in X]


def train_from_candles(
    candles: Sequence[Any],
    *,
    labels: Optional[Sequence[int]] = None,
    contexts: Optional[Sequence[MLFeatureContext | Dict[str, Any]]] = None,
    save_path: Optional[str] = None,
    lookback: int = 20,
    horizon: int = 1,
) -> Optional[Any]:
    X, y, feature_names = build_supervised_dataset(candles, labels=labels, contexts=contexts, lookback=lookback, horizon=horizon)
    if not X or not y:
        return None
    return train_ensemble(X, y, save_path=save_path, feature_names=feature_names)


def feature_vector_from_candles(candles: Sequence[Any], *, context: Optional[MLFeatureContext | Dict[str, Any]] = None, lookback: int = 20) -> Tuple[List[float], List[str]]:
    return build_market_feature_vector(candles, context=context, lookback=lookback)


def evaluate_walk_forward(X: Any, y: Sequence[int], *, n_splits: int = 5, threshold: float = 0.5) -> Dict[str, Any]:
    return walk_forward_backtest(_coerce_matrix(X), list(y), n_splits=n_splits, threshold=threshold)


class OnlineMLSignal:
    """Online learning wrapper for ML signals with concept drift detection."""
    
    def __init__(self, model_path: Optional[str] = None, lookback: int = 20):
        self.model_bundle = None
        self.lookback = lookback
        self.prediction_history = []  # (features, prediction, actual)
        self.performance_window = 50  # Window for performance tracking
        self.retrain_threshold = 0.6  # Retrain if accuracy drops below this
        self.min_samples_for_retrain = 100
        
        # Retraining cooldown properties
        self.cooldown_samples = 50  # Minimum new samples required between retrains
        self.last_retrain_count = 0  # History size at last successful retrain
        
        if model_path:
            self.load(model_path)
    
    def load(self, path: str) -> bool:
        """Load a trained model from disk."""
        self.model_bundle = load_model(path)
        return self.model_bundle is not None
    
    def save(self, path: Optional[str] = None) -> bool:
        """Save the current model to disk."""
        if self.model_bundle is None:
            return False
        try:
            import joblib
            path = path or MODEL_PATH
            joblib.dump(self.model_bundle, path)
            return True
        except Exception:
            return False
    
    def predict_signal(self, candles: Sequence[Any], 
                       context: Optional[MLFeatureContext | Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate trading signal from candles.
        
        Returns:
            Dictionary with 'signal' (0=short, 1=long), 'probability', 'confidence'
        """
        if self.model_bundle is None:
            return {'signal': 0, 'probability': 0.5, 'confidence': 0.0, 'reason': 'no_model'}
        
        try:
            # Build feature vector
            features, feature_names = feature_vector_from_candles(
                candles, context=context, lookback=self.lookback
            )
            
            if not features:
                return {'signal': 0, 'probability': 0.5, 'confidence': 0.0, 'reason': 'no_features'}
            
            # Predict
            probs = predict(self.model_bundle, [features])
            if not probs:
                return {'signal': 0, 'probability': 0.5, 'confidence': 0.0, 'reason': 'prediction_failed'}
            
            prob = probs[0]
            signal = 1 if prob >= 0.5 else 0
            
            # Calculate confidence based on probability distance from 0.5
            confidence = abs(prob - 0.5) * 2.0
            
            return {
                'signal': signal,
                'probability': prob,
                'confidence': confidence,
                'features': features,
                'feature_names': feature_names,
            }
        except Exception as e:
            return {'signal': 0, 'probability': 0.5, 'confidence': 0.0, 'reason': str(e)}
    
    def update_with_actual(self, features: List[float], prediction: float, actual: int) -> None:
        """Update model with actual outcome for online learning."""
        self.prediction_history.append({
            'features': features,
            'prediction': prediction,
            'actual': actual,
            'timestamp': datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        })
        
        # Trim history
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-500:]
        
        # Check if retraining is needed
        if len(self.prediction_history) >= self.min_samples_for_retrain:
            self._check_and_retrain()
    
    def _check_and_retrain(self) -> None:
        """Check model performance and retrain if necessary."""
        if len(self.prediction_history) < self.performance_window:
            return
            
        # Cooldown check to prevent CPU and disk-write churn
        total_samples = len(self.prediction_history)
        if total_samples - self.last_retrain_count < self.cooldown_samples:
            return
        
        # Calculate recent accuracy
        recent = self.prediction_history[-self.performance_window:]
        correct = sum(1 for r in recent if (r['prediction'] >= 0.5) == (r['actual'] == 1))
        accuracy = correct / len(recent)
        
        if accuracy < self.retrain_threshold:
            self._retrain()
    
    def _retrain(self) -> None:
        """Retrain model with accumulated data."""
        try:
            if len(self.prediction_history) < 50:
                return
            
            X = [r['features'] for r in self.prediction_history]
            y = [r['actual'] for r in self.prediction_history]
            
            # Keep existing feature names if available
            feature_names = None
            if self.model_bundle and hasattr(self.model_bundle, 'feature_names'):
                feature_names = self.model_bundle.feature_names
            
            # Train new model
            new_bundle = train_ensemble(X, y, feature_names=feature_names)
            if new_bundle:
                self.model_bundle = new_bundle
                self.last_retrain_count = len(self.prediction_history)  # Reset cooldown anchor
                logger = logging.getLogger(__name__)
                logger.info(f"Model retrained with {len(X)} samples")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Retraining failed: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current model performance metrics."""
        if not self.prediction_history:
            return {'samples': 0}
        
        total = len(self.prediction_history)
        recent_window = min(self.performance_window, total)
        recent = self.prediction_history[-recent_window:]
        
        correct = sum(1 for r in recent if (r['prediction'] >= 0.5) == (r['actual'] == 1))
        accuracy = correct / len(recent) if recent else 0.0
        
        return {
            'total_samples': total,
            'recent_accuracy': accuracy,
            'recent_samples': len(recent),
            'model_trained_at': self.model_bundle.trained_at if self.model_bundle else None,
            'model_metrics': self.model_bundle.metrics if self.model_bundle else {},
        }
