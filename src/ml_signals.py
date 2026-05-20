"""Ensemble ML signal scaffolding.

Provides a lightweight training/prediction interface using scikit-learn
ensembles when available, with a simple on-disk model cache.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import os

try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
except Exception:
    np = None
    RandomForestClassifier = None
    train_test_split = None


MODEL_PATH = "ml_signal_model.pkl"


def train_ensemble(X: List[List[float]], y: List[int], save_path: Optional[str] = None) -> Optional[Any]:
    if RandomForestClassifier is None:
        return None
    try:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        path = save_path or MODEL_PATH
        try:
            import joblib

            joblib.dump(clf, path)
        except Exception:
            pass
        return clf
    except Exception:
        return None


def load_model(path: Optional[str] = None) -> Optional[Any]:
    path = path or MODEL_PATH
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        return None


def predict(model: Any, X: List[List[float]]) -> List[float]:
    if model is None:
        return [0.0 for _ in X]
    try:
        probs = model.predict_proba(X)
        # assume positive class at index 1
        return [float(p[1]) for p in probs]
    except Exception:
        try:
            preds = model.predict(X)
            return [float(x) for x in preds]
        except Exception:
            return [0.0 for _ in X]
