"""Ensemble artifact loader for RF+XGB combined model directories.

Supports the artifact layout produced by the retraining pipeline:
    random_forest_model.pkl
    xgboost_model_cuda.pkl  (or xgboost_model.pkl)
    ensemble_config.json
    ensemble_metrics.json
    model_comparison_report.json

The loader detects ``ensemble_config.json`` or ``artifact_manifest.json``,
builds live feature vectors using persisted ``feature_columns`` and
``fill_values``, runs both base models, computes a weighted ensemble
probability, and applies explicit gating rules.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


STATUS_OK = "OK"
STATUS_ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"
STATUS_MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
STATUS_PREDICT_FAILED = "PREDICT_FAILED"
STATUS_FEATURES_MISSING = "FEATURES_MISSING"
STATUS_FEATURE_VECTOR_INVALID = "FEATURE_VECTOR_INVALID"

DEFAULT_GATE_CONFIG: Dict[str, Any] = {
    "xgb_weight": 0.70,
    "rf_weight": 0.30,
    "ensemble_threshold": 0.60,
    "xgb_min_prob": 0.58,
    "rf_min_prob": 0.52,
    "max_model_disagreement": 0.25,
    "block_on_disagreement": True,
}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _finite_float(value: object) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def _is_ensemble_artifact_dir(path: Path) -> bool:
    """Return True if directory contains ensemble-specific artifacts.

    Requires ``ensemble_config.json`` (or ``artifact_manifest.json``) plus
    at least one RF and one XGB model pickle to avoid false positives on
    unrelated artifacts that happen to contain a manifest file.
    """
    if not path.is_dir():
        return False
    has_manifest = False
    for name in ("ensemble_config.json", "artifact_manifest.json"):
        if (path / name).exists():
            has_manifest = True
            break
    if not has_manifest:
        return False
    has_rf = _find_rf_model(path) is not None
    has_xgb = _find_xgb_model(path) is not None
    return has_rf and has_xgb


def _find_xgb_model(path: Path) -> Optional[Path]:
    """Find the XGBoost model pickle inside *path*."""
    candidates = [
        path / "xgboost_model_cuda.pkl",
        path / "xgboost_model.pkl",
        path / "xgb_model.pkl",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    # Fallback: any pkl whose name suggests xgboost
    for pkl in sorted(path.glob("*.pkl")):
        lower = pkl.name.lower()
        if "xgb" in lower or "xgboost" in lower:
            return pkl
    return None


def _find_rf_model(path: Path) -> Optional[Path]:
    """Find the RandomForest model pickle inside *path*."""
    candidates = [
        path / "random_forest_model.pkl",
        path / "rf_model.pkl",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    for pkl in sorted(path.glob("*.pkl")):
        lower = pkl.name.lower()
        if "random_forest" in lower or "rf_model" in lower:
            if "xgb" not in lower and "xgboost" not in lower:
                return pkl
    return None


def _load_pickle(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".joblib":
        try:
            import joblib  # type: ignore[import-untyped]
            return joblib.load(path)
        except Exception:
            pass
    with path.open("rb") as fh:
        import pickle
        return pickle.load(fh)


def _find_positive_class_index(model: Any) -> Optional[int]:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None
    try:
        arr = np.asarray(classes)
        if arr.ndim != 1:
            return None
        if arr.size == 2:
            matches = np.where(arr == 1)[0]
            if matches.size == 1:
                return int(matches[0])
            if set(int(c) for c in arr) <= {0, 1}:
                return int(np.argmax(arr))
            return None
        if arr.size == 1:
            return 0
        return None
    except Exception:
        return None


def _predict_prob_positive(model: Any, X: np.ndarray, feature_names: Sequence[str]) -> float:
    """Return the positive-class probability for a *single* row."""
    est = model
    if isinstance(model, dict):
        est = model.get("model") or model.get("estimator") or model

    x_in = X.astype(np.float64)
    if feature_names:
        try:
            import pandas as pd  # type: ignore[import-untyped]
            x_in = pd.DataFrame(x_in, columns=list(feature_names))
        except Exception:
            pass

    if hasattr(est, "predict_proba"):
        proba = np.asarray(est.predict_proba(x_in), dtype=float)
        pos_idx = _find_positive_class_index(est)
        if pos_idx is not None:
            return float(proba[0][pos_idx])
        if proba.ndim > 1 and proba.shape[1] > 1:
            return float(proba[0][1])
        return float(proba[0][0])

    if hasattr(est, "predict"):
        raw = float(np.asarray(est.predict(x_in), dtype=float).ravel()[0])
        return raw if 0.0 <= raw <= 1.0 else _sigmoid(raw)

    raise RuntimeError("no_supported_predict_method")


def _xgb_booster(model: Any) -> bool:
    try:
        import xgboost as xgb_mod  # type: ignore[import-untyped]
        return isinstance(model, xgb_mod.Booster)
    except Exception:
        return type(model).__name__ == "Booster" and str(getattr(model, "__module__", "")).startswith("xgboost")


def _infer_xgb_prob(model: Any, X: np.ndarray, feature_names: Sequence[str]) -> float:
    if _xgb_booster(model):
        try:
            import xgboost as xgb  # type: ignore[import-untyped]

            x_in = X.astype(np.float64)
            if feature_names:
                import pandas as pd  # type: ignore[import-untyped]
                x_in = pd.DataFrame(x_in, columns=list(feature_names))
                dmat = xgb.DMatrix(x_in, feature_names=list(feature_names))
            else:
                dmat = xgb.DMatrix(x_in)
            raw = float(model.predict(dmat)[0])
            return raw if 0.0 <= raw <= 1.0 else _sigmoid(raw)
        except Exception:
            pass
    return _predict_prob_positive(model, X, feature_names)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class EnsembleArtifactLoader:
    """Load and predict from a directory-based RF+XGB ensemble artifact.

    The loader is intentionally defensive: missing config keys fall back to
    sensible defaults, missing models are reported as load failures, and
    prediction gracefully degrades with ``STATUS_*`` reason codes.
    """

    def __init__(self, artifact_dir: Path | str) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.status = ""
        self.error = ""
        self.config: Dict[str, Any] = {}
        self.rf_model: Any = None
        self.xgb_model: Any = None
        self.rf_model_path: Optional[Path] = None
        self.xgb_model_path: Optional[Path] = None
        self.feature_columns: List[str] = []
        self.fill_values: Dict[str, float] = {}
        self.gate_config: Dict[str, Any] = dict(DEFAULT_GATE_CONFIG)

    def load(self) -> "EnsembleArtifactLoader":
        """Attempt to discover and load all ensemble components.

        Returns *self* so callers can chain::

            loader = EnsembleArtifactLoader(path).load()
        """
        if not self.artifact_dir.exists():
            self.status = STATUS_ARTIFACT_NOT_FOUND
            self.error = f"artifact_dir_not_found: {self.artifact_dir}"
            return self

        self.config = _read_json(self.artifact_dir / "ensemble_config.json")
        if not self.config:
            self.config = _read_json(self.artifact_dir / "artifact_manifest.json")

        self.feature_columns = list(self.config.get("feature_columns") or [])
        fill_raw = self.config.get("fill_values") or self.config.get("train_medians") or {}
        self.fill_values = {str(k): _finite_float(v) for k, v in fill_raw.items() if _finite_float(v) is not None}

        # Gate overrides from config
        for key in DEFAULT_GATE_CONFIG:
            if key in self.config:
                self.gate_config[key] = self.config[key]

        # Load models
        self.rf_model_path = _find_rf_model(self.artifact_dir)
        self.xgb_model_path = _find_xgb_model(self.artifact_dir)

        load_errors: List[str] = []
        if self.rf_model_path:
            try:
                self.rf_model = _load_pickle(self.rf_model_path)
            except Exception as exc:
                load_errors.append(f"rf_load_failed:{exc}")
                self.rf_model = None
        else:
            load_errors.append("rf_model_not_found")

        if self.xgb_model_path:
            try:
                self.xgb_model = _load_pickle(self.xgb_model_path)
            except Exception as exc:
                load_errors.append(f"xgb_load_failed:{exc}")
                self.xgb_model = None
        else:
            load_errors.append("xgb_model_not_found")

        if self.rf_model is None and self.xgb_model is None:
            self.status = STATUS_MODEL_LOAD_FAILED
            self.error = "; ".join(load_errors) if load_errors else "no_models_loaded"
            return self

        self.status = STATUS_OK
        self.error = ""
        return self

    def _build_feature_vector(self, snapshot: Mapping[str, Any]) -> Tuple[np.ndarray, int, int]:
        """Return (X_row, missing_count, invalid_count).

        Missing values are imputed with ``self.fill_values``.
        Values that cannot be coerced to float become ``0.0`` and count as
        invalid.
        """
        feature_missing_count = 0
        feature_invalid_count = 0
        row: List[float] = []

        for col in self.feature_columns:
            raw = snapshot.get(col)
            if raw in (None, ""):
                # try aliases
                if col in ("ltp", "last_price"):
                    raw = snapshot.get("close") or snapshot.get("option_price")
                elif col in ("close",):
                    raw = snapshot.get("ltp") or snapshot.get("last_price")
                elif col in ("volume",):
                    raw = snapshot.get("volume") or snapshot.get("total_volume")
                elif col in ("oi", "open_interest"):
                    raw = snapshot.get("oi") or snapshot.get("openInterest") or snapshot.get("open_interest")

            val = _finite_float(raw)
            if val is None:
                fill = self.fill_values.get(col)
                if fill is not None:
                    val = fill
                else:
                    val = 0.0
                    feature_missing_count += 1
            row.append(val)

        arr = np.asarray(row, dtype=float).reshape(1, -1)
        return arr, feature_missing_count, feature_invalid_count

    def predict(self, snapshot: Mapping[str, Any]) -> Dict[str, Any]:
        """Run ensemble inference on a single snapshot row.

        Returns a dict with::

            {
                "status": str,
                "error": str,
                "rf_prob": float | None,
                "xgb_prob": float | None,
                "ensemble_prob": float | None,
                "model_disagreement": float | None,
                "allowed": bool,
                "block_reason": str | None,
                "feature_missing_count": int,
                "feature_invalid_count": int,
                "model_type": str,
            }
        """
        if self.status == STATUS_MODEL_LOAD_FAILED:
            return {
                "status": STATUS_MODEL_LOAD_FAILED,
                "error": self.error or "model_load_failed",
                "rf_prob": None,
                "xgb_prob": None,
                "ensemble_prob": None,
                "model_disagreement": None,
                "allowed": False,
                "block_reason": "MODEL_LOAD_FAILED",
                "feature_missing_count": 0,
                "feature_invalid_count": 0,
                "model_type": "ensemble",
            }

        if not self.feature_columns:
            return {
                "status": STATUS_FEATURES_MISSING,
                "error": "feature_columns_empty_in_config",
                "rf_prob": None,
                "xgb_prob": None,
                "ensemble_prob": None,
                "model_disagreement": None,
                "allowed": False,
                "block_reason": "FEATURES_MISSING",
                "feature_missing_count": 0,
                "feature_invalid_count": 0,
                "model_type": "ensemble",
            }

        X, feature_missing_count, feature_invalid_count = self._build_feature_vector(snapshot)

        if feature_missing_count > 0 and not self.fill_values:
            return {
                "status": STATUS_FEATURES_MISSING,
                "error": f"missing_{feature_missing_count}_features_no_fill_values",
                "rf_prob": None,
                "xgb_prob": None,
                "ensemble_prob": None,
                "model_disagreement": None,
                "allowed": False,
                "block_reason": "FEATURES_MISSING",
                "feature_missing_count": feature_missing_count,
                "feature_invalid_count": feature_invalid_count,
                "model_type": "ensemble",
            }

        rf_prob: Optional[float] = None
        xgb_prob: Optional[float] = None
        predict_error = ""

        if self.rf_model is not None:
            try:
                rf_prob = _predict_prob_positive(self.rf_model, X, self.feature_columns)
            except Exception as exc:
                predict_error = f"rf_predict:{exc}"
                rf_prob = None

        if self.xgb_model is not None:
            try:
                xgb_prob = _infer_xgb_prob(self.xgb_model, X, self.feature_columns)
            except Exception as exc:
                predict_error = f"xgb_predict:{exc}"
                xgb_prob = None

        available: Dict[str, float] = {}
        if rf_prob is not None:
            available["rf"] = float(rf_prob)
        if xgb_prob is not None:
            available["xgb"] = float(xgb_prob)

        if not available:
            return {
                "status": STATUS_PREDICT_FAILED,
                "error": predict_error or "both_models_predict_failed",
                "rf_prob": None,
                "xgb_prob": None,
                "ensemble_prob": None,
                "model_disagreement": None,
                "allowed": False,
                "block_reason": "PREDICT_FAILED",
                "feature_missing_count": feature_missing_count,
                "feature_invalid_count": feature_invalid_count,
                "model_type": "ensemble",
            }

        # Weighted ensemble
        xgb_w = float(self.gate_config.get("xgb_weight", 0.70))
        rf_w = float(self.gate_config.get("rf_weight", 0.30))
        total_w = sum(
            w for name, w in (("xgb", xgb_w), ("rf", rf_w)) if name in available
        ) or 1.0

        ensemble_prob = 0.0
        if "xgb" in available:
            ensemble_prob += available["xgb"] * (xgb_w / total_w)
        if "rf" in available:
            ensemble_prob += available["rf"] * (rf_w / total_w)

        # Disagreement
        if len(available) > 1:
            disagreement = max(available.values()) - min(available.values())
        else:
            disagreement = 0.0

        # Gates
        threshold = float(self.gate_config.get("ensemble_threshold", 0.60))
        xgb_min = float(self.gate_config.get("xgb_min_prob", 0.58))
        rf_min = float(self.gate_config.get("rf_min_prob", 0.52))
        max_disagreement = float(self.gate_config.get("max_model_disagreement", 0.25))
        block_on_disagreement = bool(self.gate_config.get("block_on_disagreement", True))

        pass_ensemble = ensemble_prob >= threshold
        pass_xgb = xgb_prob is None or xgb_prob >= xgb_min
        pass_rf = rf_prob is None or rf_prob >= rf_min
        pass_disagreement = (
            not block_on_disagreement
            or max_disagreement is None
            or disagreement <= max_disagreement
        )

        allowed = pass_ensemble and pass_xgb and pass_rf and pass_disagreement
        block_reason: Optional[str] = None
        if not pass_ensemble:
            block_reason = "LOW_ENSEMBLE_PROB"
        elif not pass_xgb:
            block_reason = "LOW_XGB_PROB"
        elif not pass_rf:
            block_reason = "LOW_RF_PROB"
        elif not pass_disagreement:
            block_reason = "MODEL_DISAGREEMENT"

        return {
            "status": STATUS_OK,
            "error": "",
            "rf_prob": rf_prob,
            "xgb_prob": xgb_prob,
            "ensemble_prob": ensemble_prob,
            "model_disagreement": disagreement,
            "allowed": allowed,
            "block_reason": block_reason,
            "feature_missing_count": feature_missing_count,
            "feature_invalid_count": feature_invalid_count,
            "model_type": "ensemble",
        }


def try_load_ensemble_dir(artifact_dir: Path | str) -> Optional[EnsembleArtifactLoader]:
    """If *artifact_dir* looks like an ensemble artifact, load and return it.

    Returns ``None`` if the directory does not contain ensemble artifacts
    (i.e. no ``ensemble_config.json`` / ``artifact_manifest.json``).
    """
    p = Path(artifact_dir)
    if not _is_ensemble_artifact_dir(p):
        return None
    loader = EnsembleArtifactLoader(p).load()
    if loader.status in (STATUS_MODEL_LOAD_FAILED, STATUS_ARTIFACT_NOT_FOUND):
        return loader
    return loader


def predict_from_ensemble_dir(
    artifact_dir: Path | str,
    snapshot: Mapping[str, Any],
) -> Dict[str, Any]:
    """Convenience one-shot: detect, load, predict.

    Returns a fully-populated dict even when the directory is **not** an
    ensemble artifact (in which case the dict has ``status='not_ensemble'``).
    """
    p = Path(artifact_dir)
    if not _is_ensemble_artifact_dir(p):
        return {
            "status": "not_ensemble",
            "error": "",
            "rf_prob": None,
            "xgb_prob": None,
            "ensemble_prob": None,
            "model_disagreement": None,
            "allowed": None,
            "block_reason": None,
            "feature_missing_count": None,
            "feature_invalid_count": None,
            "model_type": "",
        }
    loader = EnsembleArtifactLoader(p).load()
    return loader.predict(snapshot)
