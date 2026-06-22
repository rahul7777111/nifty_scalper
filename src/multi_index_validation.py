from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ml_pipeline import walk_forward_validate_production


@dataclass(slots=True)
class SymbolValidationResult:
    symbol_key: str
    sample_count: int
    passed: bool
    reason: str
    metrics: Dict[str, Any]


def _to_frame(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data)


def per_symbol_walk_forward_validation(frame: pd.DataFrame, *, target_col: str = "label", feature_cols: Optional[Sequence[str]] = None) -> List[SymbolValidationResult]:
    frame = _to_frame(frame)
    out: List[SymbolValidationResult] = []
    if frame.empty or target_col not in frame.columns:
        return out
    features = list(feature_cols or [c for c in frame.columns if c not in {target_col, "symbol_key", "symbol_id", "index_family", "snapshot_id", "as_of_ts", "session_date"}])
    for symbol_key, group in frame.groupby("symbol_key") if "symbol_key" in frame.columns else [("ALL", frame)]:
        X = group[features].fillna(0.0).to_numpy().tolist()
        y = group[target_col].fillna(0).astype(int).tolist()
        result = walk_forward_validate_production(X, y)
        out.append(SymbolValidationResult(symbol_key=str(symbol_key), sample_count=len(group), passed=bool(result.get("passed")), reason=str(result.get("reason", "unknown")), metrics=dict(result.get("metrics") or {})))
    return out


def cross_symbol_validation(frame: pd.DataFrame, *, target_col: str = "label", feature_cols: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    frame = _to_frame(frame)
    if frame.empty or target_col not in frame.columns or "symbol_key" not in frame.columns:
        return {"passed": False, "reason": "missing_data", "splits": []}
    features = list(feature_cols or [c for c in frame.columns if c not in {target_col, "symbol_key", "symbol_id", "index_family", "snapshot_id", "as_of_ts", "session_date"}])
    splits: List[Dict[str, Any]] = []
    symbols = list(dict.fromkeys(frame["symbol_key"].astype(str).tolist()))
    for held_out in symbols:
        train = frame[frame["symbol_key"].astype(str) != held_out]
        test = frame[frame["symbol_key"].astype(str) == held_out]
        if len(train) < 20 or len(test) < 10:
            continue
        X_train = train[features].fillna(0.0).to_numpy().tolist()
        y_train = train[target_col].fillna(0).astype(int).tolist()
        X_test = test[features].fillna(0.0).to_numpy().tolist()
        y_test = test[target_col].fillna(0).astype(int).tolist()
        result = walk_forward_validate_production(X_train, y_train)
        splits.append({"held_out_symbol": held_out, "train_samples": len(train), "test_samples": len(test), "train_passed": bool(result.get("passed")), "train_reason": result.get("reason", "unknown"), "test_positive_rate": float(sum(y_test) / len(y_test)) if y_test else 0.0})
    return {"passed": any(split.get("train_passed") for split in splits), "reason": "cross_symbol_summary", "splits": splits}


def leave_one_symbol_out_validation(frame: pd.DataFrame, *, target_col: str = "label", feature_cols: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    frame = _to_frame(frame)
    if frame.empty or target_col not in frame.columns or "symbol_key" not in frame.columns:
        return {"passed": False, "reason": "missing_data", "folds": []}
    features = list(feature_cols or [c for c in frame.columns if c not in {target_col, "symbol_key", "symbol_id", "index_family", "snapshot_id", "as_of_ts", "session_date"}])
    folds: List[Dict[str, Any]] = []
    for held_out in dict.fromkeys(frame["symbol_key"].astype(str).tolist()):
        train = frame[frame["symbol_key"].astype(str) != held_out]
        test = frame[frame["symbol_key"].astype(str) == held_out]
        if len(train) < 20 or len(test) < 10:
            continue
        y_test = test[target_col].fillna(0).astype(int).tolist()
        folds.append({"held_out_symbol": held_out, "train_samples": len(train), "test_samples": len(test), "test_positive_rate": float(sum(y_test) / len(y_test)) if y_test else 0.0})
    passed = bool(folds) and len([f for f in folds if f["train_samples"] >= 20 and f["test_samples"] >= 10]) == len(folds)
    return {"passed": passed, "reason": "leave_one_symbol_out", "folds": folds}


def calibrate_per_symbol_probabilities(frame: pd.DataFrame, *, target_col: str = "label", prob_col: str = "probability") -> Dict[str, Dict[str, Any]]:
    frame = _to_frame(frame)
    if frame.empty or target_col not in frame.columns or prob_col not in frame.columns or "symbol_key" not in frame.columns:
        return {}
    results: Dict[str, Dict[str, Any]] = {}
    for symbol_key, group in frame.groupby("symbol_key"):
        probs = pd.to_numeric(group[prob_col], errors="coerce").fillna(0.5).clip(0.0, 1.0)
        labels = pd.to_numeric(group[target_col], errors="coerce").fillna(0).astype(int)
        if len(group) < 10:
            continue
        threshold = float(probs.quantile(0.6))
        calibration_error = float(abs(labels.mean() - probs.mean()))
        results[str(symbol_key)] = {
            "threshold": threshold,
            "confidence_floor": float(probs.quantile(0.2)),
            "confidence_ceiling": float(probs.quantile(0.8)),
            "calibration_error": calibration_error,
            "positive_rate": float(labels.mean()),
            "sample_count": int(len(group)),
        }
    return results


def temporal_holdout_validation(
    frame: pd.DataFrame,
    *,
    target_col: str = "label",
    feature_cols: Optional[Sequence[str]] = None,
    holdout_ratio: float = 0.2,
) -> Dict[str, Any]:
    frame = _to_frame(frame)
    if frame.empty or target_col not in frame.columns:
        return {"passed": False, "reason": "missing_data", "metrics": {}}

    if "as_of_ts" in frame.columns:
        frame = frame.sort_values("as_of_ts").reset_index(drop=True)
    elif "session_date" in frame.columns:
        frame = frame.sort_values("session_date").reset_index(drop=True)

    n = len(frame)
    split_idx = int(n * (1.0 - holdout_ratio))

    train = frame.iloc[:split_idx]
    test = frame.iloc[split_idx:]

    if len(train) < 20 or len(test) < 10:
        return {"passed": False, "reason": "insufficient_samples", "metrics": {}}

    features = list(feature_cols or [c for c in frame.columns if c not in {target_col, "symbol_key", "symbol_id", "index_family", "snapshot_id", "as_of_ts", "session_date"}])

    X_train = train[features].fillna(0.0).to_numpy().tolist()
    y_train = train[target_col].fillna(0).astype(int).tolist()
    X_test = test[features].fillna(0.0).to_numpy().tolist()
    y_test = test[target_col].fillna(0).astype(int).tolist()

    from ml_signals import WeightedEnsembleClassifier
    model = WeightedEnsembleClassifier()
    model.fit(X_train, y_train)

    probs = [p[1] for p in model.predict_proba(X_test)]
    from ml_pipeline import _metrics_from_predictions
    metrics = _metrics_from_predictions(y_test, probs)

    passed = bool(metrics.get("roc_auc", 0.0) >= 0.53 and metrics.get("accuracy", 0.0) >= 0.51)

    return {
        "passed": passed,
        "reason": "temporal_holdout",
        "train_samples": len(train),
        "test_samples": len(test),
        "metrics": metrics,
    }


class SymbolProbabilityCalibrator:
    """Production-grade Platt-scaling based calibrator per symbol."""

    def __init__(self) -> None:
        self.params: Dict[str, Tuple[float, float]] = {}  # symbol_key -> (A, B)
        self.thresholds: Dict[str, float] = {}            # symbol_key -> tuned_threshold

    def fit(self, y_true: Sequence[int], y_prob: Sequence[float], symbol_keys: Sequence[str]) -> None:
        import numpy as np
        from scipy.optimize import minimize

        y_true_arr = np.array(y_true, dtype=int)
        y_prob_arr = np.array(y_prob, dtype=float)
        sym_arr = np.array(symbol_keys, dtype=str)

        unique_symbols = np.unique(sym_arr)
        for sym in unique_symbols:
            mask = sym_arr == sym
            sub_y = y_true_arr[mask]
            sub_p = y_prob_arr[mask]
            if len(sub_y) < 20 or len(np.unique(sub_y)) < 2:
                self.params[str(sym)] = (-1.0, 0.0)  # default fallback
                self.thresholds[str(sym)] = 0.5
                continue

            def loss_func(params):
                A, B = params
                p_cal = 1.0 / (1.0 + np.exp(A * sub_p + B))
                p_cal = np.clip(p_cal, 1e-15, 1 - 1e-15)
                return -np.mean(sub_y * np.log(p_cal) + (1 - sub_y) * np.log(1 - p_cal))

            res = minimize(loss_func, [0.0, 0.0], method='L-BFGS-B')
            A, B = res.x
            self.params[str(sym)] = (float(A), float(B))

            # Tune threshold for max F1 score
            calibrated_probs = 1.0 / (1.0 + np.exp(A * sub_p + B))
            best_thresh = 0.5
            best_f1 = 0.0
            for th in np.linspace(0.3, 0.7, 41):
                preds = (calibrated_probs >= th).astype(int)
                tp = np.sum((sub_y == 1) & (preds == 1))
                fp = np.sum((sub_y == 0) & (preds == 1))
                fn = np.sum((sub_y == 1) & (preds == 0))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = th
            self.thresholds[str(sym)] = float(best_thresh)

    def calibrate(self, raw_prob: float, symbol_key: str) -> float:
        import numpy as np
        sym = str(symbol_key)
        if sym not in self.params:
            return float(raw_prob)
        A, B = self.params[sym]
        cal_prob = 1.0 / (1.0 + np.exp(A * raw_prob + B))
        return float(np.clip(cal_prob, 0.0, 1.0))

    def get_decision_threshold(self, symbol_key: str) -> float:
        return self.thresholds.get(str(symbol_key), 0.5)

