"""Ensemble ML signal scaffolding with feature pipeline and walk-forward metrics."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import os
import logging
import math
import uuid
import statistics

from ml_pipeline import MLFeatureContext, build_market_feature_vector, build_supervised_dataset_v2, walk_forward_backtest

# Import feature contract for forbidden token enforcement at live inference
try:
    from ml_feature_contract import (
        is_feature_forbidden,
        validate_live_features,
        CORE_REQUIRED_FEATURES,
        CONTRACT_VERSION,
        compute_feature_coverage,
    )
    HAS_FEATURE_CONTRACT = True
except ImportError:
    HAS_FEATURE_CONTRACT = False
    LOGGER = logging.getLogger(__name__)
    LOGGER.warning("ml_feature_contract not available - skipping forbidden token enforcement")

try:
    from sklearn.ensemble import RandomForestClassifier
except Exception:
    try:
        from ml_pipeline import FallbackClassifier
        RandomForestClassifier = FallbackClassifier
    except Exception:
        RandomForestClassifier = None

try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

try:
    from sklearn.calibration import CalibratedClassifierCV
except Exception:
    CalibratedClassifierCV = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


MODEL_PATH = "ml_signal_model.pkl"
LOGGER = logging.getLogger(__name__)


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


def evaluate_ml_gating_before_execution(
    market_data_row: Any,
    model: Any,
    selected_features: Sequence[str],
) -> Tuple[str, float]:
    """Score a live row and emit a deterministic telemetry token.
    
    FAILS CLOSED if forbidden features detected or required features missing.
    """

    prediction_id = f"PRED_{uuid.uuid4().hex[:8].upper()}"
    if model is None:
        return prediction_id, 0.0

    if hasattr(market_data_row, "to_dict"):
        row_dict = market_data_row.to_dict()
    elif isinstance(market_data_row, dict):
        row_dict = dict(market_data_row)
    else:
        raise TypeError("market_data_row must be dict-like or a pandas Series")

    # FAIL CLOSED: Check for forbidden features before building feature vector
    if HAS_FEATURE_CONTRACT:
        forbidden = [fname for fname in row_dict.keys() if is_feature_forbidden(fname)]
        if forbidden:
            LOGGER.error(
                "Live ML gating BLOCKED for prediction_id=%s: FORBIDDEN_FEATURES=%s",
                prediction_id, forbidden
            )
            return prediction_id, 0.0
        
        # Also validate required features
        is_valid, errors = validate_live_features(row_dict, CORE_REQUIRED_FEATURES)
        if not is_valid:
            LOGGER.error(
                "Live ML gating BLOCKED for prediction_id=%s: VALIDATION_FAILED=%s",
                prediction_id, errors
            )
            return prediction_id, 0.0

    feature_vector = [[float(row_dict.get(feature, 0.0) or 0.0) for feature in selected_features]]
    try:
        probabilities = predict(model, feature_vector)
        probability = float(probabilities[0]) if probabilities else 0.0
    except Exception as exc:
        LOGGER.warning("Live ML gating failed for prediction_id=%s: %s", prediction_id, exc)
        probability = 0.0
    return prediction_id, max(0.0, min(1.0, probability))


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
    walk_forward_gap: int = 10,
    walk_forward_threshold: float = 0.5,
    holdout_pct: float = 0.10,
    use_time_decay: bool = True,
    decay_lambda: float = 0.005,
    model_type: str = "ensemble",
    **kwargs,
) -> Optional[Any]:
    try:
        X_mat = _coerce_matrix(X)
        y_list = [int(v) for v in list(y)]
        if not X_mat or len(X_mat) != len(y_list):
            return None
            
        n_samples = len(X_mat)
        n_features = len(X_mat[0]) if n_samples > 0 else 0
        holdout_size = max(1, int(n_samples * float(holdout_pct))) if n_samples >= 20 else 0
        if holdout_size >= n_samples:
            holdout_size = max(0, n_samples - 1)
        train_cutoff = n_samples - holdout_size if holdout_size > 0 else n_samples
        X_train_full = X_mat[:train_cutoff]
        y_train_full = y_list[:train_cutoff]
        X_holdout = X_mat[train_cutoff:] if holdout_size > 0 else []
        y_holdout = y_list[train_cutoff:] if holdout_size > 0 else []

        # Calculate standard scaling metrics
        scaler_mean = []
        scaler_std = []
        if X_train_full and n_features > 0:
            for j in range(n_features):
                col = [X_train_full[i][j] for i in range(len(X_train_full))]
                m = sum(col) / len(X_train_full)
                scaler_mean.append(m)
                variance = sum((val - m) ** 2 for val in col) / len(X_train_full)
                s = math.sqrt(variance)
                scaler_std.append(s if s > 1e-9 else 1e-9)

            def _scale_rows(rows: Sequence[Sequence[float]]) -> List[List[float]]:
                scaled_rows: List[List[float]] = []
                for row_in in rows:
                    scaled_rows.append(
                        [
                            (float(row_in[j]) - scaler_mean[j]) / scaler_std[j]
                            for j in range(n_features)
                        ]
                    )
                return scaled_rows

            X_scaled = _scale_rows(X_mat)
            X_train_scaled = _scale_rows(X_train_full)
            X_holdout_scaled = _scale_rows(X_holdout) if X_holdout else []
        else:
            X_scaled = X_mat
            X_train_scaled = X_train_full
            X_holdout_scaled = X_holdout
            scaler_mean = None
            scaler_std = None

        clf = build_model(model_type=model_type)
        if clf is None:
            return None
        
        # Calculate Exponential Time-Decay weights if enabled
        sample_weights = None
        if use_time_decay and len(X_train_scaled) > 0:
            sample_weights = []
            for i in range(len(X_train_scaled)):
                w = math.exp(-decay_lambda * (len(X_train_scaled) - 1 - i))
                sample_weights.append(w)
                
        # Train model with optional sample weighting
        if sample_weights is not None:
            import inspect
            sig = inspect.signature(clf.fit)
            if "sample_weight" in sig.parameters:
                clf.fit(X_train_scaled, y_train_full, sample_weight=sample_weights)
            else:
                clf.fit(X_train_scaled, y_train_full)
        else:
            clf.fit(X_train_scaled, y_train_full)
            
        metrics: Dict[str, Any] = {}
        try:
            # Use scaled features in walk-forward evaluation
            wf = walk_forward_backtest(
                X_train_scaled,
                y_train_full,
                n_splits=walk_forward_splits,
                gap=walk_forward_gap,
                threshold=walk_forward_threshold,
                model_factory=lambda: build_model(model_type=model_type),
            )
            metrics = dict(wf.get("aggregate") or {})
            metrics["walk_forward_folds"] = wf.get("folds") or []
            metrics["model_type"] = str(model_type or "ensemble")
            metrics["train_samples"] = int(len(X_train_scaled))
            metrics["holdout_samples"] = int(len(X_holdout_scaled))
            metrics["holdout_pct"] = float(holdout_pct)
            if X_holdout_scaled:
                holdout_probs = predict(_wrap_model(clf, feature_names=feature_names, metrics=metrics, scaler_mean=scaler_mean, scaler_std=scaler_std), X_holdout)
                holdout_pred = [1 if float(prob) >= float(walk_forward_threshold) else 0 for prob in holdout_probs]
                holdout_accuracy = sum(1 for yt, yp in zip(y_holdout, holdout_pred) if int(yt) == int(yp)) / len(y_holdout)
                tp = sum(1 for yt, yp in zip(y_holdout, holdout_pred) if int(yt) == 1 and int(yp) == 1)
                fp = sum(1 for yt, yp in zip(y_holdout, holdout_pred) if int(yt) == 0 and int(yp) == 1)
                fn = sum(1 for yt, yp in zip(y_holdout, holdout_pred) if int(yt) == 1 and int(yp) == 0)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                holdout_metrics = {
                    "accuracy": float(holdout_accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }
                if roc_auc_score is not None and len(set(y_holdout)) > 1:
                    try:
                        holdout_metrics["roc_auc"] = float(roc_auc_score(y_holdout, holdout_probs))
                    except Exception:
                        pass
                holdout_metrics.update(
                    _compute_holdout_trade_metrics(
                        y_holdout,
                        holdout_probs,
                        threshold=walk_forward_threshold,
                    )
                )
                metrics["holdout"] = holdout_metrics
            else:
                metrics["holdout"] = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "profit_factor": 0.0, "sharpe": 0.0, "trades_count": 0.0}
        except Exception as exc:
            LOGGER.warning("Walk-forward evaluation failed during training: %s", exc)
            metrics = {}
            
        path = save_path or MODEL_PATH
        bundle = _wrap_model(
            clf,
            feature_names=feature_names,
            metrics=metrics,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
        )
        import joblib
        joblib.dump(bundle, path)
        return bundle
    except Exception as exc:
        LOGGER.exception("Model training failed: %s", exc)
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
    X, y, feature_names = build_supervised_dataset_v2(
        candles,
        labels=labels,
        contexts=contexts,
        lookback=lookback,
        horizon=horizon,
    )


def _compute_holdout_trade_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    threshold: float,
    profit_target: float = 0.01,
    stop_loss: float = 0.005,
) -> Dict[str, float]:
    trade_pnls: List[float] = []
    for truth, prob in zip(y_true, y_prob):
        if float(prob) >= float(threshold):
            trade_pnls.append(float(profit_target) if int(truth) == 1 else -float(stop_loss))
    if not trade_pnls:
        return {"profit_factor": 0.0, "sharpe": 0.0, "trades_count": 0.0}

    wins = sum(p for p in trade_pnls if p > 0.0)
    losses = sum(abs(p) for p in trade_pnls if p < 0.0)
    profit_factor = wins / losses if losses > 0.0 else (99.0 if wins > 0.0 else 0.0)
    if len(trade_pnls) > 1:
        mean_pnl = statistics.fmean(trade_pnls)
        std_pnl = statistics.stdev(trade_pnls)
        sharpe = (mean_pnl / std_pnl) * math.sqrt(252.0) if std_pnl > 0.0 else 0.0
    else:
        sharpe = 0.0
    return {
        "profit_factor": float(profit_factor),
        "sharpe": float(sharpe),
        "trades_count": float(len(trade_pnls)),
    }
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
        
        FAILS CLOSED if forbidden features detected or required features missing.
        
        Returns:
            Dictionary with 'signal' (0=short, 1=long), 'probability', 'confidence',
            and optionally 'contract_version' if feature contract is enabled.
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
            
            # FAIL CLOSED: Validate features against contract before prediction
            if HAS_FEATURE_CONTRACT:
                # Create a dict from the feature vector for validation
                feature_dict = dict(zip(feature_names, features))
                
                # Check for forbidden features
                forbidden = [fname for fname in feature_dict.keys() if is_feature_forbidden(fname)]
                if forbidden:
                    return {
                        'signal': 0, 'probability': 0.0, 'confidence': 0.0,
                        'reason': 'forbidden_features_detected',
                        'forbidden_features': forbidden,
                        'contract_version': CONTRACT_VERSION,
                    }
                
                # Validate required features
                is_valid, errors = validate_live_features(feature_dict, CORE_REQUIRED_FEATURES)
                if not is_valid:
                    return {
                        'signal': 0, 'probability': 0.0, 'confidence': 0.0,
                        'reason': 'feature_validation_failed',
                        'validation_errors': errors,
                        'contract_version': CONTRACT_VERSION,
                    }
            
            # Predict
            probs = predict(self.model_bundle, [features])
            if not probs:
                return {'signal': 0, 'probability': 0.5, 'confidence': 0.0, 'reason': 'prediction_failed'}
            
            prob = probs[0]
            signal = 1 if prob >= 0.5 else 0
            
            # Calculate confidence based on probability distance from 0.5
            confidence = abs(prob - 0.5) * 2.0
            
            result = {
                'signal': signal,
                'probability': prob,
                'confidence': confidence,
                'features': features,
                'feature_names': feature_names,
            }
            
            if HAS_FEATURE_CONTRACT:
                result['contract_version'] = CONTRACT_VERSION
            
            return result
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


def available_model_names() -> List[str]:
    names: List[str] = []
    if LogisticRegression is not None:
        names.append("logistic_regression")
    if RandomForestClassifier is not None:
        names.append("random_forest")
    if XGBClassifier is not None:
        names.append("xgboost")
    if CatBoostClassifier is not None:
        names.append("catboost")
    names.append("ensemble")
    return names


def build_model(model_type: str = "ensemble", *, random_state: int = 42) -> Optional[Any]:
    model_key = str(model_type or "ensemble").strip().lower()
    if model_key == "ensemble":
        return WeightedEnsembleClassifier(random_state=random_state)
    if model_key == "logistic_regression" and LogisticRegression is not None:
        return LogisticRegression(
            max_iter=3000,
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
        )
    if model_key == "random_forest" and RandomForestClassifier is not None:
        base_model = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        if CalibratedClassifierCV is not None:
            try:
                return CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
            except Exception:
                return base_model
        return base_model
    if model_key == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )
    if model_key == "catboost" and CatBoostClassifier is not None:
        return CatBoostClassifier(
            iterations=200,
            depth=4,
            learning_rate=0.05,
            random_seed=random_state,
            verbose=False,
            loss_function="Logloss",
        )
    if model_key == "random_forest" and RandomForestClassifier is None:
        return None
    if model_key == "logistic_regression" and LogisticRegression is None:
        return None
    if model_key == "xgboost" and XGBClassifier is None:
        return None
    if model_key == "catboost" and CatBoostClassifier is None:
        return None
    return WeightedEnsembleClassifier(random_state=random_state)


class WeightedEnsembleClassifier:
    """Simple time-order-safe weighted ensemble over the installed classical models."""

    def __init__(
        self,
        *,
        enabled_models: Optional[Sequence[str]] = None,
        random_state: int = 42,
        n_estimators: int = 200,
        **kwargs,
    ):
        self.enabled_models = [str(m) for m in (enabled_models or []) if str(m or "").strip()]
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.kwargs = dict(kwargs or {})
        self.models: List[Tuple[str, Any]] = []
        self.weights: List[float] = []

    def _candidate_model_names(self) -> List[str]:
        if self.enabled_models:
            return [m for m in self.enabled_models if m != "ensemble"]
        return [name for name in available_model_names() if name != "ensemble"]

    def _predict_positive_probs(self, model: Any, X: Sequence[Sequence[float]]) -> List[float]:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            return [float(row[1]) if len(row) > 1 else float(row[0]) for row in probs]
        preds = model.predict(X)
        return [float(v) for v in preds]

    def fit(self, X, y, sample_weight=None):
        import inspect

        X_mat = _coerce_matrix(X)
        y_list = [int(v) for v in list(y)]
        if not X_mat or len(X_mat) != len(y_list):
            return self

        candidate_names = self._candidate_model_names()
        train_cutoff = max(10, int(len(X_mat) * 0.8))
        X_train = X_mat[:train_cutoff]
        y_train = y_list[:train_cutoff]
        X_val = X_mat[train_cutoff:]
        y_val = y_list[train_cutoff:]

        fitted_models: List[Tuple[str, Any]] = []
        raw_weights: List[float] = []

        for model_name in candidate_names:
            model = build_model(model_name, random_state=self.random_state)
            if model is None:
                continue
            try:
                sig = inspect.signature(model.fit)
                if "sample_weight" in sig.parameters and sample_weight is not None:
                    fit_weights = list(sample_weight[: len(X_train)]) if len(sample_weight) >= len(X_train) else None
                    if fit_weights is not None:
                        model.fit(X_train, y_train, sample_weight=fit_weights)
                    else:
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                if X_val and len(set(y_val)) > 1:
                    val_probs = self._predict_positive_probs(model, X_val)
                    if roc_auc_score is not None:
                        try:
                            weight = max(0.01, float(roc_auc_score(y_val, val_probs)) - 0.5)
                        except Exception:
                            weight = 1.0
                    else:
                        val_preds = [1 if p >= 0.5 else 0 for p in val_probs]
                        accuracy = sum(1 for yt, yp in zip(y_val, val_preds) if yt == yp) / len(y_val)
                        weight = max(0.01, accuracy)
                else:
                    weight = 1.0
                model_full = build_model(model_name, random_state=self.random_state)
                if model_full is None:
                    continue
                sig_full = inspect.signature(model_full.fit)
                if "sample_weight" in sig_full.parameters and sample_weight is not None:
                    model_full.fit(X_mat, y_list, sample_weight=sample_weight)
                else:
                    model_full.fit(X_mat, y_list)
                fitted_models.append((model_name, model_full))
                raw_weights.append(float(weight))
            except Exception:
                continue

        if not fitted_models:
            fallback = None
            if RandomForestClassifier is not None:
                try:
                    fallback = RandomForestClassifier(
                        n_estimators=max(50, int(self.n_estimators)),
                        random_state=self.random_state,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                    )
                except Exception:
                    fallback = None
            if fallback is None:
                from ml_pipeline import FallbackClassifier
                fallback = FallbackClassifier()
            fallback.fit(X_mat, y_list)
            fitted_models = [("fallback", fallback)]
            raw_weights = [1.0]

        weight_sum = sum(raw_weights) or 1.0
        self.models = fitted_models
        self.weights = [float(w / weight_sum) for w in raw_weights]
        return self

    def predict_proba(self, X):
        X_mat = _coerce_matrix(X)
        if not self.models:
            return [[0.5, 0.5] for _ in X_mat]

        accum = [0.0] * len(X_mat)
        for weight, (_, model) in zip(self.weights, self.models):
            probs = self._predict_positive_probs(model, X_mat)
            for idx, prob in enumerate(probs):
                accum[idx] += weight * float(prob)
        return [[1.0 - max(0.0, min(1.0, p)), max(0.0, min(1.0, p))] for p in accum]

    def predict(self, X):
        probs = self.predict_proba(X)
        return [1 if row[1] >= 0.5 else 0 for row in probs]

