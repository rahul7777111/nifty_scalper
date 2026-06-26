"""Production-safe XGBoost + RandomForest weighted ensemble classifier."""

from __future__ import annotations

import math
import pickle
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .feature_safety import (
    apply_train_median_fill,
    build_safe_numeric_bool_frame,
    compute_train_medians,
    safe_feature_columns,
)

try:
    from sklearn.base import BaseEstimator, ClassifierMixin, clone
except Exception:  # pragma: no cover - fallback for minimal environments
    class BaseEstimator:  # type: ignore[override]
        pass

    class ClassifierMixin:  # type: ignore[override]
        pass

    def clone(estimator: Any) -> Any:
        return deepcopy(estimator)

try:
    from sklearn.ensemble import RandomForestClassifier
except Exception:  # pragma: no cover - optional dependency
    RandomForestClassifier = None

try:
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover - optional dependency
    LogisticRegression = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


@dataclass
class EnsembleGateConfig:
    min_models_for_decision: int = 1
    max_model_disagreement: Optional[float] = 0.35
    min_probability_edge: float = 0.0
    decision_threshold: float = 0.5
    blocked_class: int = 0


@dataclass
class EnsembleModelConfig:
    version: int = 1
    random_state: int = 42
    rf_weight: float = 0.5
    xgb_weight: float = 0.5
    rf_params: Dict[str, Any] = field(default_factory=dict)
    xgb_params: Dict[str, Any] = field(default_factory=dict)
    xgb_gpu_preference: str = "auto"
    gate: EnsembleGateConfig = field(default_factory=EnsembleGateConfig)


class XGBRFEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Weighted binary ensemble with conservative runtime gating."""

    DEFAULT_RF_PARAMS: Dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 14,
        "min_samples_leaf": 15,
        "min_samples_split": 30,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "bootstrap": True,
    }

    DEFAULT_XGB_CPU_PARAMS: Dict[str, Any] = {
        "n_estimators": 1000,
        "max_depth": 5,
        "learning_rate": 0.025,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 15,
        "reg_lambda": 5.0,
        "reg_alpha": 0.2,
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "n_jobs": -1,
        "tree_method": "hist",
    }

    DEFAULT_XGB_GPU_PARAMS: Tuple[Dict[str, Any], ...] = (
        {"device": "cuda", "tree_method": "hist"},
        {"tree_method": "gpu_hist"},
    )

    def __init__(
        self,
        *,
        random_state: int = 42,
        rf_weight: float = 0.5,
        xgb_weight: float = 0.5,
        rf_params: Optional[Mapping[str, Any]] = None,
        xgb_params: Optional[Mapping[str, Any]] = None,
        xgb_gpu_preference: str = "auto",
        min_models_for_decision: int = 1,
        max_model_disagreement: Optional[float] = 0.35,
        min_probability_edge: float = 0.0,
        decision_threshold: float = 0.5,
        xgb_min_prob: float = 0.5,
        rf_min_prob: float = 0.5,
        block_on_disagreement: bool = True,
        blocked_class: int = 0,
        feature_columns: Optional[Sequence[str]] = None,
        safe_frame: bool = True,
    ) -> None:
        self.random_state = int(random_state)
        self.rf_weight = float(rf_weight)
        self.xgb_weight = float(xgb_weight)
        self.rf_params = dict(rf_params or {})
        self.xgb_params = dict(xgb_params or {})
        self.xgb_gpu_preference = str(xgb_gpu_preference or "auto").strip().lower()
        self.min_models_for_decision = int(min_models_for_decision)
        self.max_model_disagreement = None if max_model_disagreement is None else float(max_model_disagreement)
        self.min_probability_edge = float(min_probability_edge)
        self.decision_threshold = float(decision_threshold)
        self.xgb_min_prob = float(xgb_min_prob)
        self.rf_min_prob = float(rf_min_prob)
        self.block_on_disagreement = bool(block_on_disagreement)
        self.blocked_class = int(blocked_class)
        self.feature_columns = list(feature_columns) if feature_columns is not None else None
        self.safe_frame = bool(safe_frame)

        self.rf_model_: Any = None
        self.xgb_model_: Any = None
        self.rf_calibrator_: Any = None
        self.xgb_calibrator_: Any = None
        self.classes_: np.ndarray = np.asarray([0, 1], dtype=int)
        self.feature_names_in_: List[str] = []
        self.train_medians_: Dict[str, float] = {}
        self.fit_summary_: Dict[str, Any] = {}
        self.xgb_runtime_backend_: str = "uninitialized"

    @classmethod
    def xgb_backend_candidates(
        cls,
        preference: str = "auto",
        extra_params: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        params = dict(extra_params or {})
        preference_key = str(preference or "auto").strip().lower()
        candidates: List[Dict[str, Any]] = []
        if preference_key in {"gpu", "auto"}:
            for gpu_params in cls.DEFAULT_XGB_GPU_PARAMS:
                merged = dict(params)
                merged.update(gpu_params)
                candidates.append(merged)
        merged_cpu = dict(params)
        candidates.append(merged_cpu)
        return candidates

    def get_config(self) -> Dict[str, Any]:
        cfg = EnsembleModelConfig(
            random_state=self.random_state,
            rf_weight=self.rf_weight,
            xgb_weight=self.xgb_weight,
            rf_params=self._resolved_rf_params(),
            xgb_params=self._resolved_xgb_cpu_params(),
            xgb_gpu_preference=self.xgb_gpu_preference,
            gate=EnsembleGateConfig(
                min_models_for_decision=self.min_models_for_decision,
                max_model_disagreement=self.max_model_disagreement,
                min_probability_edge=self.min_probability_edge,
                decision_threshold=self.decision_threshold,
                blocked_class=self.blocked_class,
            ),
        )
        payload = asdict(cfg)
        payload["feature_columns"] = list(self.feature_columns or [])
        payload["safe_frame"] = self.safe_frame
        payload["xgb_min_prob"] = self.xgb_min_prob
        payload["rf_min_prob"] = self.rf_min_prob
        payload["block_on_disagreement"] = self.block_on_disagreement
        return payload

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "XGBRFEnsembleClassifier":
        gate = dict(config.get("gate") or {})
        return cls(
            random_state=int(config.get("random_state", 42)),
            rf_weight=float(config.get("rf_weight", 0.5)),
            xgb_weight=float(config.get("xgb_weight", 0.5)),
            rf_params=config.get("rf_params"),
            xgb_params=config.get("xgb_params"),
            xgb_gpu_preference=str(config.get("xgb_gpu_preference", "auto")),
            min_models_for_decision=int(gate.get("min_models_for_decision", config.get("min_models_for_decision", 1))),
            max_model_disagreement=gate.get("max_model_disagreement", config.get("max_model_disagreement", 0.35)),
            min_probability_edge=float(gate.get("min_probability_edge", config.get("min_probability_edge", 0.0))),
            decision_threshold=float(gate.get("decision_threshold", config.get("decision_threshold", 0.5))),
            xgb_min_prob=float(config.get("xgb_min_prob", 0.5)),
            rf_min_prob=float(config.get("rf_min_prob", 0.5)),
            block_on_disagreement=bool(config.get("block_on_disagreement", True)),
            blocked_class=int(gate.get("blocked_class", config.get("blocked_class", 0))),
            feature_columns=config.get("feature_columns"),
            safe_frame=bool(config.get("safe_frame", True)),
        )

    def _resolved_rf_params(self) -> Dict[str, Any]:
        params = dict(self.DEFAULT_RF_PARAMS)
        params.update(self.rf_params)
        params["random_state"] = self.random_state
        params["n_jobs"] = max(1, int(params.get("n_jobs", 1) or 1))
        return params

    def _resolved_xgb_cpu_params(self) -> Dict[str, Any]:
        params = dict(self.DEFAULT_XGB_CPU_PARAMS)
        params.update(self.xgb_params)
        params["random_state"] = self.random_state
        params["n_jobs"] = max(1, int(params.get("n_jobs", 1) or 1))
        return params

    def _coerce_input_frame(self, X: Any, *, fit: bool) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            base = X.copy()
        elif hasattr(X, "shape") and hasattr(X, "__array__"):
            arr = np.asarray(X)
            columns = list(self.feature_names_in_) if self.feature_names_in_ else [f"f{i}" for i in range(arr.shape[1])]
            base = pd.DataFrame(arr, columns=columns)
        else:
            base = pd.DataFrame(X)

        selected_columns = self.feature_columns
        if fit and selected_columns is None:
            selected_columns = safe_feature_columns(base.columns)
        frame, diagnostics = build_safe_numeric_bool_frame(
            base,
            feature_columns=selected_columns or self.feature_names_in_ or None,
        ) if self.safe_frame else (base.copy(), {"selected_columns": list(base.columns), "kept_columns": list(base.columns)})

        if fit:
            self.feature_names_in_ = list(frame.columns)
            self.train_medians_ = compute_train_medians(frame)
            self.fit_summary_["feature_prep"] = diagnostics
        else:
            if self.feature_names_in_:
                for column in self.feature_names_in_:
                    if column not in frame.columns:
                        frame[column] = np.nan
                frame = frame.loc[:, self.feature_names_in_]
            frame = apply_train_median_fill(frame, self.train_medians_)
        return frame

    def _prepare_y(self, y: Any) -> np.ndarray:
        series = pd.Series(y)
        numeric = pd.to_numeric(series, errors="raise").astype(int)
        values = np.asarray(numeric)
        unique = np.unique(values)
        if len(unique) != 2:
            raise ValueError("XGBRFEnsembleClassifier requires exactly two target classes")
        self.classes_ = unique
        return values

    def _fit_rf(self, X: pd.DataFrame, y: np.ndarray, sample_weight: Optional[Sequence[float]]) -> Tuple[Any, Optional[str]]:
        if RandomForestClassifier is None:
            return None, "random_forest_unavailable"
        estimator = RandomForestClassifier(**self._resolved_rf_params())
        estimator.fit(X, y, sample_weight=sample_weight)
        return estimator, None

    def _fit_binary_sigmoid_calibrator(self, probs: np.ndarray, y: np.ndarray) -> Any:
        if LogisticRegression is None:
            return None
        if probs.size == 0 or len(np.unique(y)) < 2:
            return None
        calibrator = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=self.random_state)
        calibrator.fit(np.asarray(probs, dtype=float).reshape(-1, 1), np.asarray(y, dtype=int))
        return calibrator

    def _fit_xgb(self, X: pd.DataFrame, y: np.ndarray, sample_weight: Optional[Sequence[float]]) -> Tuple[Any, Optional[str]]:
        if XGBClassifier is None:
            self.xgb_runtime_backend_ = "unavailable"
            return None, "xgboost_unavailable"
        base_params = self._resolved_xgb_cpu_params()
        errors: List[str] = []
        for idx, candidate_overrides in enumerate(self.xgb_backend_candidates(self.xgb_gpu_preference, base_params)):
            estimator = XGBClassifier(**candidate_overrides)
            try:
                estimator.fit(X, y, sample_weight=sample_weight)
                if idx == len(self.xgb_backend_candidates(self.xgb_gpu_preference, base_params)) - 1:
                    self.xgb_runtime_backend_ = "cpu"
                else:
                    self.xgb_runtime_backend_ = "gpu"
                return estimator, None
            except Exception as exc:
                errors.append(str(exc))
        self.xgb_runtime_backend_ = "failed"
        return None, "; ".join(errors[-2:]) if errors else "xgboost_fit_failed"

    def fit(
        self,
        X: Any,
        y: Any,
        X_val: Any = None,
        y_val: Any = None,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "XGBRFEnsembleClassifier":
        frame = self._coerce_input_frame(X, fit=True)
        frame = apply_train_median_fill(frame, self.train_medians_)
        target = self._prepare_y(y)

        rf_error = None
        xgb_error = None
        self.rf_model_, rf_error = self._fit_rf(frame, target, sample_weight)
        self.xgb_model_, xgb_error = self._fit_xgb(frame, target, sample_weight)
        if X_val is not None and y_val is not None:
            val_frame = self._coerce_input_frame(X_val, fit=False)
            val_target = pd.to_numeric(pd.Series(y_val), errors="raise").astype(int).to_numpy()
            if self.rf_model_ is not None:
                self.rf_calibrator_ = self._fit_binary_sigmoid_calibrator(
                    self._predict_positive_proba_from_model(self.rf_model_, val_frame),
                    val_target,
                )
            if self.xgb_model_ is not None:
                self.xgb_calibrator_ = self._fit_binary_sigmoid_calibrator(
                    self._predict_positive_proba_from_model(self.xgb_model_, val_frame),
                    val_target,
                )
        self.fit_summary_.update(
            {
                "rows": int(len(frame)),
                "columns": int(frame.shape[1]),
                "rf_available": self.rf_model_ is not None,
                "xgb_available": self.xgb_model_ is not None,
                "rf_calibrated": self.rf_calibrator_ is not None,
                "xgb_calibrated": self.xgb_calibrator_ is not None,
                "rf_error": rf_error,
                "xgb_error": xgb_error,
                "xgb_runtime_backend": self.xgb_runtime_backend_,
                "train_medians": dict(self.train_medians_),
                "config": self.get_config(),
            }
        )
        if self.rf_model_ is None and self.xgb_model_ is None:
            raise RuntimeError("no base model trained successfully")
        return self

    def _predict_positive_proba_from_model(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        probs = model.predict_proba(X)
        arr = np.asarray(probs, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("predict_proba must return two columns for binary classification")
        return arr[:, 1]

    def _apply_calibrator(self, calibrator: Any, probs: np.ndarray) -> np.ndarray:
        if calibrator is None:
            return probs
        calibrated = np.asarray(calibrator.predict_proba(np.asarray(probs, dtype=float).reshape(-1, 1))[:, 1], dtype=float)
        return np.clip(calibrated, 0.0, 1.0)

    def _model_probabilities(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        if self.rf_model_ is not None:
            out["rf"] = self._apply_calibrator(
                self.rf_calibrator_,
                self._predict_positive_proba_from_model(self.rf_model_, X),
            )
        if self.xgb_model_ is not None:
            out["xgb"] = self._apply_calibrator(
                self.xgb_calibrator_,
                self._predict_positive_proba_from_model(self.xgb_model_, X),
            )
        return out

    def _available_weights(self, model_probs: Mapping[str, np.ndarray]) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        if "rf" in model_probs and self.rf_weight > 0:
            weights["rf"] = float(self.rf_weight)
        if "xgb" in model_probs and self.xgb_weight > 0:
            weights["xgb"] = float(self.xgb_weight)
        if not weights and model_probs:
            equal_weight = 1.0 / float(len(model_probs))
            return {name: equal_weight for name in model_probs}
        total = sum(weights.values()) or 1.0
        return {name: value / total for name, value in weights.items()}

    def _decision_details_from_probs(self, model_probs: Mapping[str, np.ndarray]) -> List[Dict[str, Any]]:
        if not model_probs:
            raise RuntimeError("ensemble has no available models for inference")

        weights = self._available_weights(model_probs)
        row_count = len(next(iter(model_probs.values())))
        details: List[Dict[str, Any]] = []

        for idx in range(row_count):
            per_model = {name: float(values[idx]) for name, values in model_probs.items()}
            weighted_prob = sum(per_model[name] * weights.get(name, 0.0) for name in per_model)
            disagreement = (max(per_model.values()) - min(per_model.values())) if len(per_model) > 1 else 0.0
            edge = abs(weighted_prob - self.decision_threshold)
            passed = True
            reasons: List[str] = []

            if len(per_model) < self.min_models_for_decision:
                passed = False
                reasons.append("insufficient_models")
            if self.block_on_disagreement and self.max_model_disagreement is not None and disagreement > self.max_model_disagreement:
                passed = False
                reasons.append("model_disagreement")
            if edge < self.min_probability_edge:
                passed = False
                reasons.append("low_probability_edge")

            predicted_class = int(self.classes_[1] if weighted_prob >= self.decision_threshold else self.classes_[0])
            gated_class = predicted_class if passed else int(self.blocked_class)
            details.append(
                {
                    "combined_probability": weighted_prob,
                    "predicted_class": predicted_class,
                    "gated_class": gated_class,
                    "passed_gate": passed,
                    "gate_reasons": reasons,
                    "probability_edge": edge,
                    "model_disagreement": disagreement,
                    "model_probabilities": per_model,
                    "model_weights": dict(weights),
                    "models_used": list(per_model.keys()),
                    "decision_threshold": self.decision_threshold,
                }
            )
        return details

    def predict_proba(self, X: Any) -> np.ndarray:
        frame = self._coerce_input_frame(X, fit=False)
        model_probs = self._model_probabilities(frame)
        details = self._decision_details_from_probs(model_probs)
        positive = np.asarray([row["combined_probability"] for row in details], dtype=float)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])

    def predict(self, X: Any) -> np.ndarray:
        details = self.decision_details(X)
        positive_class = int(self.classes_[1]) if len(self.classes_) > 1 else 1
        blocked_class = int(self.blocked_class)
        return np.asarray([positive_class if bool(row.get("final_allowed")) else blocked_class for row in details], dtype=int)

    def predict_details(self, X: Any) -> List[Dict[str, Any]]:
        frame = self._coerce_input_frame(X, fit=False)
        return self._decision_details_from_probs(self._model_probabilities(frame))

    def decision_details(self, X: Any) -> List[Dict[str, Any]]:
        frame = self._coerce_input_frame(X, fit=False)
        model_probs = self._model_probabilities(frame)
        rows = self._decision_details_from_probs(model_probs)
        details: List[Dict[str, Any]] = []
        for row in rows:
            per_model = dict(row.get("model_probabilities") or {})
            xgb_prob = float(per_model.get("xgb")) if "xgb" in per_model else None
            rf_prob = float(per_model.get("rf")) if "rf" in per_model else None
            ensemble_prob = float(row.get("combined_probability") or 0.0)
            disagreement = float(row.get("model_disagreement") or 0.0)
            pass_ensemble = ensemble_prob >= self.decision_threshold
            pass_xgb = xgb_prob is None or xgb_prob >= self.xgb_min_prob
            pass_rf = rf_prob is None or rf_prob >= self.rf_min_prob
            pass_disagreement = (not self.block_on_disagreement) or self.max_model_disagreement is None or disagreement <= float(self.max_model_disagreement)
            final_allowed = bool(row.get("passed_gate")) and pass_ensemble and pass_xgb and pass_rf and pass_disagreement
            block_reason = None
            if not pass_ensemble:
                block_reason = "LOW_ENSEMBLE_PROB"
            elif not pass_xgb:
                block_reason = "LOW_XGB_PROB"
            elif not pass_rf:
                block_reason = "LOW_RF_PROB"
            elif not pass_disagreement:
                block_reason = "MODEL_DISAGREEMENT"
            details.append(
                {
                    "xgb_prob": xgb_prob,
                    "rf_prob": rf_prob,
                    "ensemble_prob": ensemble_prob,
                    "model_disagreement": disagreement,
                    "pass_ensemble_threshold": pass_ensemble,
                    "pass_xgb_gate": pass_xgb,
                    "pass_rf_gate": pass_rf,
                    "pass_disagreement_gate": pass_disagreement,
                    "final_allowed": final_allowed,
                    "block_reason": block_reason,
                }
            )
        return details

    def decision_function(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return probs - self.decision_threshold

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as handle:
            pickle.dump(self, handle)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "XGBRFEnsembleClassifier":
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, cls):
            return obj
        raise TypeError(f"artifact is not a {cls.__name__}")
