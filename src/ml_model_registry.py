from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import joblib
import numpy as np

# Import feature contract for forbidden token enforcement at live inference
try:
    from ml_feature_contract import (
        is_feature_forbidden,
        validate_live_features,
        CORE_REQUIRED_FEATURES,
        CONTRACT_VERSION,
    )
    HAS_FEATURE_CONTRACT = True
except ImportError:
    HAS_FEATURE_CONTRACT = False


ALLOWED_PAPER_VERDICTS = {
    "PAPER_TRADE_CANDIDATE_LOW_CONFIDENCE",
    "PAPER_TRADE_CANDIDATE_MEDIUM_CONFIDENCE",
    "PAPER_TRADE_CANDIDATE_HIGH_CONFIDENCE",
    "PRODUCTION_BLOCKED_PAPER_ONLY",
}
SHADOW_ONLY_VERDICTS = {
    "REJECT_NO_EDGE",
    "RESEARCH_ONLY_WEAK_EDGE",
}


@dataclass
class DeploymentManifest:
    path: Path
    payload: Dict[str, Any]

    @property
    def model_id(self) -> str:
        return str(self.payload.get("model_id") or "")

    @property
    def selected_feature_list(self) -> list[str]:
        values = self.payload.get("selected_feature_list")
        if isinstance(values, list):
            return [str(v) for v in values]
        return []


def load_deployment_manifest(path: str | Path) -> DeploymentManifest:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"deployment manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = [
        "model_id",
        "model_family",
        "feature_set_name",
        "label_name",
        "selected_threshold",
        "model_path",
        "paper_readiness_verdict",
        "production_adoption_allowed",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"deployment manifest missing fields: {missing}")
    if bool(payload.get("production_adoption_allowed")):
        raise RuntimeError("production_adoption_allowed must be false for shadow/paper deployment")
    verdict = str(payload.get("paper_readiness_verdict") or "")
    if verdict not in ALLOWED_PAPER_VERDICTS | SHADOW_ONLY_VERDICTS:
        raise RuntimeError(f"invalid paper_readiness_verdict: {verdict}")
    model_path = manifest_path.parent / str(payload.get("model_path"))
    if not model_path.exists():
        raise FileNotFoundError(f"model artifact missing: {model_path}")
    feature_list = payload.get("selected_feature_list")
    feature_list_path = payload.get("selected_feature_list_path")
    if not feature_list and feature_list_path:
        resolved = manifest_path.parent / str(feature_list_path)
        if not resolved.exists():
            raise FileNotFoundError(f"selected_feature_list_path missing: {resolved}")
        feature_list = json.loads(resolved.read_text(encoding="utf-8"))
        payload["selected_feature_list"] = feature_list
    if not isinstance(payload.get("selected_feature_list"), list) or not payload["selected_feature_list"]:
        raise RuntimeError("selected_feature_list must exist and be non-empty")
    return DeploymentManifest(path=manifest_path, payload=payload)


class MLModelRegistry:
    def __init__(self, manifest_path: str | Path | None) -> None:
        self._manifest_path = Path(manifest_path) if manifest_path else None
        self._manifest: DeploymentManifest | None = None
        self._bundle: Any = None
        self._health: Dict[str, Any] = {"ok": False, "reason": "not_loaded"}
        if self._manifest_path:
            self.reload()

    def reload(self) -> None:
        if self._manifest_path is None:
            self._health = {"ok": False, "reason": "manifest_path_missing"}
            return
        try:
            self._manifest = load_deployment_manifest(self._manifest_path)
            resolved_model = self._manifest.path.parent / str(self._manifest.payload["model_path"])
            self._bundle = joblib.load(resolved_model)
            self._health = {"ok": True, "reason": "loaded", "model_id": self._manifest.model_id}
        except Exception as exc:
            self._manifest = None
            self._bundle = None
            self._health = {"ok": False, "reason": str(exc)}

    def get_active_model(self) -> Any:
        return self._bundle

    def get_model_features(self) -> list[str]:
        if self._manifest is None:
            return []
        return list(self._manifest.selected_feature_list)

    def validate_runtime_features(self, snapshot_features: Mapping[str, Any]) -> tuple[bool, list[str], list[str]]:
        """
        Validate runtime features for live inference (fail-closed).
        
        This method checks:
        1. All expected model features are present
        2. All values are finite (no NaN/Inf)
        3. NO forbidden features (labels, returns, PnL, future data) are present
          
        Returns:
            Tuple of (is_valid, missing_features, invalid_features, forbidden_features).
            For backwards compatibility, the first 3 elements match the old signature,
            but a 4th element (forbidden_features) is appended.
        """
        expected = self.get_model_features()
        
        # Check for forbidden features FIRST (fail-closed)
        forbidden: list[str] = []
        if HAS_FEATURE_CONTRACT:
            for name in snapshot_features.keys():
                if is_feature_forbidden(name):
                    forbidden.append(name)
        
        missing = [name for name in expected if name not in snapshot_features]
        invalid: list[str] = []
        for name in expected:
            if name not in snapshot_features:
                continue
            try:
                value = float(snapshot_features[name])
            except Exception:
                invalid.append(name)
                continue
            if not np.isfinite(value):
                invalid.append(name)
        
        is_valid = not missing and not invalid and not forbidden
        return (is_valid, missing, invalid, forbidden)

    def validate_runtime_features_legacy(self, snapshot_features: Mapping[str, Any]) -> tuple[bool, list[str], list[str]]:
        """
        Legacy validation without forbidden token checking.
        Kept for backwards compatibility.
        """
        expected = self.get_model_features()
        missing = [name for name in expected if name not in snapshot_features]
        invalid: list[str] = []
        for name in expected:
            if name not in snapshot_features:
                continue
            try:
                value = float(snapshot_features[name])
            except Exception:
                invalid.append(name)
                continue
            if not np.isfinite(value):
                invalid.append(name)
        return (not missing and not invalid, missing, invalid)

    def explain_missing_features(self, snapshot_features: Mapping[str, Any]) -> Dict[str, Any]:
        ok, missing, invalid, forbidden = self.validate_runtime_features(snapshot_features)
        result = {
            "ok": ok,
            "missing_features": missing,
            "invalid_features": invalid,
            "forbidden_features": forbidden,
            "feature_contract_version": CONTRACT_VERSION if HAS_FEATURE_CONTRACT else None,
        }
        if forbidden:
            result["error"] = (
                f"FORBIDDEN_FEATURES_DETECTED: {forbidden}. "
                "Forbidden features (labels, returns, PnL, future data) "
                "must never be used at live inference."
            )
        return result

    def predict_proba(self, snapshot_features: Mapping[str, Any]) -> float:
        if not self._health.get("ok"):
            raise RuntimeError("model registry not healthy")
        ok, missing, invalid, forbidden = self.validate_runtime_features(snapshot_features)
        if forbidden:
            raise RuntimeError(
                f"FORBIDDEN_FEATURES_DETECTED: {forbidden}. "
                "Live inference FAILS CLOSED: forbidden features must not be present."
            )
        if not ok:
            raise RuntimeError(f"runtime feature validation failed missing={missing} invalid={invalid}")
        vector = np.asarray([[float(snapshot_features[name]) for name in self.get_model_features()]], dtype=float)
        if hasattr(self._bundle, "scaler_mean") and hasattr(self._bundle, "scaler_std"):
            mean = np.asarray(getattr(self._bundle, "scaler_mean") or [], dtype=float)
            std = np.asarray(getattr(self._bundle, "scaler_std") or [], dtype=float)
            if mean.size == vector.shape[1] and std.size == vector.shape[1]:
                safe_std = np.where(std == 0.0, 1.0, std)
                vector = (vector - mean) / safe_std
        model = getattr(self._bundle, "model", self._bundle)
        if hasattr(model, "predict_proba"):
            probs = np.asarray(model.predict_proba(vector), dtype=float)
            positive_idx = self._find_positive_class_index(model)
            if positive_idx is not None:
                return float(probs[0][positive_idx])
            if probs.ndim > 1 and probs.shape[1] > 2:
                raise RuntimeError(
                    "predict_proba returned multi-class output (>2 classes) but classes_ "
                    "attribute is unavailable; cannot determine positive class index. "
                    "For binary classification ensure the model exposes a classes_ attribute."
                )
            return float(probs[0][1] if probs.shape[1] > 1 else probs[0][0])
        if hasattr(model, "predict"):
            preds = model.predict(vector)
            raw = float(preds[0])
            return max(0.0, min(1.0, raw))
        raise RuntimeError("model has neither predict_proba nor predict")

    def _find_positive_class_index(self, model: Any) -> int | None:
        """Safely locate the positive class (label=1) index in model.classes_.

        Returns None if classes_ is not available or on single-class models,
        and raises RuntimeError on multi-class (>2 labels) without explicit handling.
        """
        classes = getattr(model, "classes_", None)
        if classes is None:
            return None
        try:
            classes_arr = np.asarray(classes)
            if classes_arr.ndim != 1:
                return None
            n_classes = classes_arr.size
            if n_classes == 2:
                matches = np.where(classes_arr == 1)[0]
                if matches.size == 1:
                    return int(matches[0])
                if set(int(c) for c in classes_arr) <= {0, 1}:
                    return int(np.argmax(classes_arr))
                return None
            elif n_classes == 1:
                return None
            else:
                raise RuntimeError(
                    f"multi-class model detected ({n_classes} classes) but "
                    "binary positive probability requested; positive class is ambiguous."
                )
        except RuntimeError:
            raise
        except Exception:
            return None

    def model_health_status(self) -> Dict[str, Any]:
        payload = dict(self._health)
        if self._manifest is not None:
            payload["paper_readiness_verdict"] = self._manifest.payload.get("paper_readiness_verdict")
            payload["production_adoption_allowed"] = False
        return payload

    def manifest(self) -> DeploymentManifest | None:
        return self._manifest


def feature_vector_hash(feature_names: Sequence[str], snapshot_features: Mapping[str, Any]) -> str:
    joined = "|".join(f"{name}={snapshot_features.get(name)}" for name in feature_names)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()
