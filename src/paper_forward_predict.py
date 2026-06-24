"""Paper Forward ML prediction: feature alignment, quality gates, probability extraction."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from paper_forward_readiness import (
        XGBOOST_INSTALL_HINT,
        artifact_expects_xgboost,
        classify_load_error,
        is_xgboost_missing_error,
        xgboost_available,
    )
except ImportError:
    try:
        from .paper_forward_readiness import (
            XGBOOST_INSTALL_HINT,
            artifact_expects_xgboost,
            classify_load_error,
            is_xgboost_missing_error,
            xgboost_available,
        )
    except ImportError:
        XGBOOST_INSTALL_HINT = r".\.venv\Scripts\pip.exe install xgboost"

        def xgboost_available() -> bool:
            return False

        def is_xgboost_missing_error(message: str) -> bool:
            return "xgboost" in str(message or "").lower()

        def artifact_expects_xgboost(cand_meta=None, *, model_type="", artifact_dir=None) -> bool:
            return "xgb" in str((cand_meta or {}).get("model_name", "")).lower()

        def classify_load_error(exc, *, cand_meta=None, model_type="", artifact_dir=None) -> str:
            return "XGBOOST_NOT_INSTALLED" if is_xgboost_missing_error(str(exc)) else f"pickle_load_failed:{exc}"

try:
    from pf_logging import log_feature_x, pf_log
except ImportError:
    def pf_log(level, msg, **kwargs):  # type: ignore[misc]
        print(msg, flush=True)

    def log_feature_x(**kwargs):  # type: ignore[misc]
        pass


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except Exception:
        return default


MAX_NAN_RATIO = _env_float("PAPER_FORWARD_MAX_FEATURE_NAN_RATIO", 0.02)
MAX_ZERO_RATIO = _env_float("PAPER_FORWARD_MAX_FEATURE_ZERO_RATIO", 0.90)
MAX_CONSTANT_FEATURES = int(_env_float("PAPER_FORWARD_MAX_CONSTANT_FEATURES", 120))

_ENSEMBLE_COMPONENT_KEYS = {
    "xgb": ("xgb_model", "xgb_model_", "xgboost_model", "xgboost_model_"),
    "rf": ("rf_model", "rf_model_", "random_forest_model", "random_forest_model_"),
}


def _looks_like_model_pickle(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() != ".pkl":
        return False
    blocked = ("metric", "threshold", "ensemble_weight", "oof", "report", "summary")
    return not any(tok in name for tok in blocked)


def _find_positive_class_index(model: Any) -> Optional[int]:
    """Locate positive class (label=1) index in model.classes_."""
    import numpy as np

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
        if n_classes == 1:
            return 0
        return None
    except Exception:
        return None


def _unwrap_estimator(model: Any) -> Tuple[Any, Optional[Any], Optional[Any], str]:
    """Return (estimator, scaler, calibrator, model_class_name). Supports Pipeline wrappers."""
    scaler = None
    calibrator = None
    name = type(model).__name__
    try:
        from sklearn.pipeline import Pipeline  # type: ignore

        if isinstance(model, Pipeline):
            steps = list(model.named_steps.items())
            est = model
            for step_name, step_obj in reversed(steps):
                lname = step_name.lower()
                if scaler is None and hasattr(step_obj, "transform") and "scaler" in lname:
                    scaler = step_obj
                elif calibrator is None and (
                    "calib" in lname or type(step_obj).__name__.startswith("Calibrated")
                ):
                    calibrator = step_obj
                elif hasattr(step_obj, "predict_proba") or hasattr(step_obj, "predict"):
                    est = step_obj
                    name = type(step_obj).__name__
            return est, scaler, calibrator, name
    except Exception:
        pass
    try:
        if type(model).__name__.startswith("Calibrated"):
            calibrator = model
            base = getattr(model, "estimator", None) or getattr(model, "base_estimator", None)
            if base is not None:
                return base, scaler, calibrator, type(base).__name__
    except Exception:
        pass
    return model, scaler, calibrator, name


def _sigmoid(x: float) -> float:
    import math as _m

    if x >= 0:
        z = _m.exp(-x)
        return 1.0 / (1.0 + z)
    z = _m.exp(x)
    return z / (1.0 + z)


def _extract_probability(model: Any, X: Any, *, method_hint: str = "") -> Tuple[float, float, str]:
    """Return (raw_output, probability, method)."""
    import numpy as np

    est, _, calibrator, cls_name = _unwrap_estimator(model)
    if calibrator is not None and hasattr(calibrator, "predict_proba"):
        proba = np.asarray(calibrator.predict_proba(X), dtype=float)
        pos_idx = _find_positive_class_index(calibrator)
        if pos_idx is None:
            pos_idx = _find_positive_class_index(est)
        if pos_idx is not None:
            prob = float(proba[0][pos_idx])
        elif proba.ndim > 1 and proba.shape[1] > 1:
            prob = float(proba[0][1])
        else:
            prob = float(proba[0][0])
        return prob, prob, "calibrated_predict_proba"

    if hasattr(est, "predict_proba"):
        proba = np.asarray(est.predict_proba(X), dtype=float)
        pos_idx = _find_positive_class_index(est)
        if pos_idx is not None:
            prob = float(proba[0][pos_idx])
        elif proba.ndim > 1 and proba.shape[1] > 1:
            prob = float(proba[0][1])
        else:
            prob = float(proba[0][0])
        return prob, prob, "predict_proba"

    if hasattr(est, "decision_function"):
        score = float(np.asarray(est.decision_function(X), dtype=float).ravel()[0])
        prob = _sigmoid(score)
        return score, prob, "decision_function_sigmoid"

    if hasattr(est, "predict"):
        p = est.predict(X)
        raw = float(np.asarray(p, dtype=float).ravel()[0])
        if 0.0 <= raw <= 1.0:
            return raw, raw, "predict"
        prob = _sigmoid(raw)
        return raw, prob, "predict_sigmoid"

    if callable(est):
        raw = float(est(X))
        prob = raw if 0.0 <= raw <= 1.0 else _sigmoid(raw)
        return raw, prob, method_hint or "callable_model"

    raise RuntimeError("no_supported_predict_method")


def _is_xgb_booster(model: Any) -> bool:
    try:
        import xgboost as xgb_mod  # type: ignore

        return isinstance(model, xgb_mod.Booster)
    except Exception:
        return type(model).__name__ == "Booster" and str(getattr(model, "__module__", "")).startswith("xgboost")


def _xgb_input_frame(arr: Any, columns: List[str] | None) -> Any:
    import numpy as np

    arr64 = arr.astype(np.float64)
    if columns:
        try:
            import pandas as pd  # type: ignore

            return pd.DataFrame(arr64, columns=list(columns))
        except Exception:
            pass
    return arr64


def _infer_xgb_probability(model: Any, X: Any, feature_names: List[str] | None) -> Tuple[float, float, str]:
    import numpy as np

    if not xgboost_available():
        raise RuntimeError("No module named 'xgboost'")
    try:
        import xgboost as xgb  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"No module named 'xgboost': {exc}") from exc

    if _is_xgb_booster(model):
        x_in = _xgb_input_frame(X, feature_names)
        dmat = xgb.DMatrix(x_in, feature_names=feature_names if feature_names else None)
        raw = float(model.predict(dmat)[0])
        prob = raw if 0.0 <= raw <= 1.0 else _sigmoid(raw)
        return raw, prob, "xgb_booster_predict"

    est, _, _, _ = _unwrap_estimator(model)
    if hasattr(est, "predict_proba"):
        x_in = _xgb_input_frame(X, feature_names)
        proba = np.asarray(est.predict_proba(x_in), dtype=float)
        pos_idx = _find_positive_class_index(est)
        if pos_idx is not None:
            prob = float(proba[0][pos_idx])
        elif proba.ndim > 1 and proba.shape[1] > 1:
            prob = float(proba[0][1])
        else:
            prob = float(proba[0][0])
        return prob, prob, "xgb_predict_proba"
    if hasattr(est, "predict"):
        x_in = _xgb_input_frame(X, feature_names)
        raw = float(np.asarray(est.predict(x_in), dtype=float).ravel()[0])
        prob = raw if 0.0 <= raw <= 1.0 else _sigmoid(raw)
        return raw, prob, "xgb_predict"
    raise RuntimeError("no_supported_predict_method")


def _extract_named_component(container: Any, keys: tuple[str, ...]) -> Any:
    if container is None:
        return None
    if isinstance(container, dict):
        for key in keys:
            if container.get(key) is not None:
                return container.get(key)
    for key in keys:
        if hasattr(container, key):
            try:
                value = getattr(container, key)
            except Exception:
                continue
            if value is not None:
                return value
    return None


def _extract_component_weight(container: Any, family: str) -> float | None:
    family_key = "rf" if family == "rf" else "xgb"
    direct_keys = (
        f"{family_key}_weight",
        f"{family_key}_model_weight",
        f"{family_key}_probability_weight",
    )
    weights_obj = None
    if isinstance(container, dict):
        weights_obj = container.get("weights") or container.get("model_weights")
        for key in direct_keys:
            if container.get(key) not in (None, ""):
                try:
                    return float(container.get(key))
                except Exception:
                    return None
    else:
        for attr in ("weights", "model_weights"):
            if hasattr(container, attr):
                try:
                    weights_obj = getattr(container, attr)
                    break
                except Exception:
                    pass
        for key in direct_keys:
            if hasattr(container, key):
                try:
                    return float(getattr(container, key))
                except Exception:
                    return None
    if isinstance(weights_obj, dict):
        aliases = [family_key, family, "random_forest" if family_key == "rf" else "xgboost"]
        for key in aliases:
            if weights_obj.get(key) not in (None, ""):
                try:
                    return float(weights_obj.get(key))
                except Exception:
                    return None
    return None


def _extract_xgb_rf_components(container: Any) -> tuple[Any, Any, float, float] | None:
    xgb_model = _extract_named_component(container, _ENSEMBLE_COMPONENT_KEYS["xgb"])
    rf_model = _extract_named_component(container, _ENSEMBLE_COMPONENT_KEYS["rf"])
    if xgb_model is None or rf_model is None:
        return None
    xgb_weight = _extract_component_weight(container, "xgb")
    rf_weight = _extract_component_weight(container, "rf")
    if xgb_weight is None and rf_weight is None:
        xgb_weight = 0.5
        rf_weight = 0.5
    elif xgb_weight is None:
        rf_weight = float(rf_weight)
        xgb_weight = max(0.0, 1.0 - rf_weight)
    elif rf_weight is None:
        xgb_weight = float(xgb_weight)
        rf_weight = max(0.0, 1.0 - xgb_weight)
    return xgb_model, rf_model, float(xgb_weight), float(rf_weight)


def _infer_xgb_rf_ensemble_probability(
    container: Any,
    X: Any,
    feature_names: List[str] | None,
) -> Tuple[float, float, str]:
    components = _extract_xgb_rf_components(container)
    if components is None:
        raise RuntimeError("xgb_rf_ensemble_components_missing")
    xgb_model, rf_model, xgb_weight, rf_weight = components
    total_weight = xgb_weight + rf_weight
    if total_weight <= 0:
        xgb_weight = 0.5
        rf_weight = 0.5
        total_weight = 1.0
    _, xgb_prob, _ = _infer_xgb_probability(xgb_model, X, feature_names)
    _, rf_prob, _ = _extract_probability(rf_model, X)
    prob = ((xgb_prob * xgb_weight) + (rf_prob * rf_weight)) / total_weight
    prob = float(max(0.0, min(1.0, prob)))
    return prob, prob, "xgb_rf_ensemble_predict_proba"


def _infer_xgb_rf_ensemble_details(
    container: Any,
    X: Any,
    feature_names: List[str] | None,
) -> Dict[str, Any]:
    details_fn = getattr(container, "decision_details", None)
    if callable(details_fn):
        try:
            rows = details_fn(X)
            row0 = rows[0] if rows else {}
            if isinstance(row0, dict):
                ensemble_prob = row0.get("ensemble_prob", row0.get("combined_probability"))
                xgb_prob = row0.get("xgb_prob")
                rf_prob = row0.get("rf_prob")
                return {
                    "raw": float(ensemble_prob) if ensemble_prob is not None else None,
                    "prob": float(ensemble_prob) if ensemble_prob is not None else None,
                    "method": "xgb_rf_ensemble_decision_details",
                    "ensemble_prob": None if ensemble_prob is None else float(ensemble_prob),
                    "xgb_prob": None if xgb_prob is None else float(xgb_prob),
                    "rf_prob": None if rf_prob is None else float(rf_prob),
                    "model_disagreement": None if row0.get("model_disagreement") is None else float(row0.get("model_disagreement")),
                    "allowed": bool(row0.get("final_allowed")),
                    "block_reason": row0.get("block_reason"),
                }
        except Exception:
            pass

    raw, prob, method = _infer_xgb_rf_ensemble_probability(container, X, feature_names)
    components = _extract_xgb_rf_components(container)
    xgb_prob = None
    rf_prob = None
    disagreement = None
    if components is not None:
        xgb_model, rf_model, _, _ = components
        try:
            _, xgb_prob, _ = _infer_xgb_probability(xgb_model, X, feature_names)
        except Exception:
            xgb_prob = None
        try:
            _, rf_prob, _ = _extract_probability(rf_model, X)
        except Exception:
            rf_prob = None
        if xgb_prob is not None and rf_prob is not None:
            disagreement = abs(float(xgb_prob) - float(rf_prob))
    return {
        "raw": raw,
        "prob": prob,
        "method": method,
        "ensemble_prob": prob,
        "xgb_prob": xgb_prob,
        "rf_prob": rf_prob,
        "model_disagreement": disagreement,
        "allowed": None,
        "block_reason": None,
    }


def _merge_predict_out(base: Dict[str, Any], debug: Dict[str, Any]) -> Dict[str, Any]:
    err = base.get("error")
    preserve = {
        "raw", "prob", "confidence", "predict_method", "decision", "error",
        "missing_features", "classes_", "artifact_path", "model_present", "X_shape",
        "ensemble_prob", "xgb_prob", "rf_prob", "model_disagreement", "allowed", "block_reason",
        "feature_missing_count", "feature_invalid_count", "model_type",
    }
    merged = dict(debug)
    merged.update(base)
    for k in preserve:
        if k in base:
            merged[k] = base[k]
    if err not in (None, ""):
        merged["error"] = err
    elif "error" in merged:
        merged["error"] = None
    return merged


def _blocked_result(
    error: str,
    *,
    debug: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = {
        "raw": None,
        "prob": None,
        "confidence": None,
        "error": error,
        "predict_method": "none",
        "decision": "BLOCKED",
        **debug,
    }
    if extra:
        out.update(extra)
    return out


def validate_feature_vector_quality(
    feature_order: List[str],
    feat_debug: Dict[str, Any],
    *,
    missing_features: Optional[List[str]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Return (ok, reason_code, details)."""
    import numpy as np

    n = max(len(feature_order), 1)
    missing = list(missing_features or feat_debug.get("required_missing_after_optional_impute") or [])
    nan_count = int(feat_debug.get("nan_count", 0) or 0)
    zero_count = int(feat_debug.get("zero_count", 0) or 0)
    constant_count = int(feat_debug.get("constant_feature_count", 0) or 0)

    if missing:
        return False, "FEATURE_VECTOR_INVALID", {
            "missing_features": missing,
            "nan_count": nan_count,
            "zero_count": zero_count,
        }

    nan_ratio = nan_count / n
    zero_ratio = zero_count / n
    if nan_ratio > MAX_NAN_RATIO:
        bad = feat_debug.get("nan_feature_names") or []
        return False, "FEATURE_VECTOR_INVALID", {
            "reason_detail": f"nan_ratio_{nan_ratio:.3f}_gt_{MAX_NAN_RATIO}",
            "bad_features": bad[:20],
            "nan_count": nan_count,
        }
    if zero_ratio > MAX_ZERO_RATIO:
        return False, "FEATURE_VECTOR_INVALID", {
            "reason_detail": f"zero_ratio_{zero_ratio:.3f}_gt_{MAX_ZERO_RATIO}",
            "zero_count": zero_count,
        }
    if constant_count > MAX_CONSTANT_FEATURES:
        return False, "FEATURE_VECTOR_INVALID", {
            "reason_detail": f"constant_features_{constant_count}_gt_{MAX_CONSTANT_FEATURES}",
            "constant_feature_count": constant_count,
        }
    return True, "", {}


def validate_artifact_identity_at_predict(
    candidate_meta: Optional[Dict[str, Any]],
    artifact_dir: Optional[str],
    feature_order: List[str],
    *,
    threshold: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    if not candidate_meta or not artifact_dir:
        return True, []
    try:
        from candidate_artifact_resolver import validate_artifact_identity
    except ImportError:
        try:
            from .candidate_artifact_resolver import validate_artifact_identity
        except ImportError:
            return True, []

    folder = Path(artifact_dir)
    if not folder.exists():
        return False, ["artifact_dir"]
    ok, mismatches = validate_artifact_identity(candidate_meta, folder)
    details = list(mismatches)
    cid = str(candidate_meta.get("candidate_id") or "")
    if cid and folder.name != cid:
        if "candidate_id" not in details:
            details.append("candidate_id")
    expected_n = int(candidate_meta.get("required_features") or candidate_meta.get("n_features") or 0)
    if expected_n > 0 and len(feature_order) != expected_n:
        details.append("feature_order_length")
    cfg_thr = candidate_meta.get("threshold")
    if threshold is not None and cfg_thr not in (None, "") and abs(float(cfg_thr) - float(threshold)) > 1e-6:
        if candidate_meta.get("shared_artifact_ok") is not True:
            details.append("threshold")
    return len(details) == 0, details


def log_pf_predict_diag(
    *,
    candidate_id: str,
    artifact_path: str = "",
    model_type: str = "",
    required_features_count: int = 0,
    feature_vector_count: int = 0,
    missing_features: Optional[List[str]] = None,
    nan_count: int = 0,
    zero_count: int = 0,
    constant_feature_count: int = 0,
    scaler_present: bool = False,
    model_present: bool = False,
    raw_output: Any = None,
    probability: Any = None,
    confidence: Any = None,
    threshold: Any = None,
    decision: str = "",
    classes_: Any = None,
    predict_method: str = "",
    error: str = "",
    install_hint: str = "",
) -> None:
    miss = missing_features or []
    miss_txt = ",".join(miss[:8]) if miss else "-"
    raw_txt = "-" if raw_output is None else f"{raw_output}"
    prob_txt = "-" if probability is None else f"{probability}"
    conf_txt = "-" if confidence is None else f"{confidence}"
    thr_txt = "-" if threshold is None else f"{threshold}"
    cls_txt = "-" if classes_ is None else str(list(classes_) if hasattr(classes_, "__iter__") else classes_)
    pf_log(
        "INFO",
        f"[PF-PREDICT-DIAG] candidate_id={candidate_id} artifact_path={artifact_path} "
        f"model_type={model_type} required_features_count={required_features_count} "
        f"feature_vector_count={feature_vector_count} missing_features={miss_txt} "
        f"nan_count={nan_count} zero_count={zero_count} constant_feature_count={constant_feature_count} "
        f"scaler_present={scaler_present} model_present={model_present} raw_output={raw_txt} "
        f"probability={prob_txt} confidence={conf_txt} threshold={thr_txt} decision={decision} "
        f"classes_={cls_txt} predict_method={predict_method} error={error or '-'} "
        f"install_hint={install_hint or '-'}",
        rate_key=f"pf_predict_diag:{candidate_id}",
        rate_interval=5.0,
    )


def build_aligned_feature_row(
    feature_order: List[str],
    snapshot: Dict[str, Any],
    debug: Optional[Dict[str, Any]] = None,
    optional_features: Optional[List[str]] = None,
) -> Any:
    """Build exactly one model input row in artifact feature order."""
    import numpy as np

    if not feature_order:
        feature_order = [
            k
            for k in snapshot.keys()
            if not str(k).startswith(("future_", "gross_", "net_", "label", "_"))
        ]
    opts = set(optional_features or [])
    vals: List[float] = []
    nan_count = 0
    inf_count = 0
    zero_count = 0
    required_missing: List[str] = []
    nan_names: List[str] = []
    for f in feature_order:
        v = snapshot.get(f)
        try:
            fv = float(v) if v is not None else np.nan
        except Exception:
            fv = np.nan
        if np.isinf(fv):
            inf_count += 1
            fv = np.nan
        if np.isnan(fv):
            nan_count += 1
            nan_names.append(f)
            if f in opts:
                fv = 0.0
            else:
                required_missing.append(f)
        if fv == 0.0:
            zero_count += 1
        vals.append(fv)
    X = np.array([vals], dtype=np.float32)
    constant_feature_count = 0
    if X.shape[1] > 0:
        col = X[0]
        finite = col[np.isfinite(col)]
        if finite.size:
            constant_feature_count = int(np.sum(finite == finite[0]))
    if debug is not None:
        try:
            debug.update(
                {
                    "X_shape": X.shape,
                    "nan_count": int(nan_count),
                    "inf_count": int(inf_count),
                    "zero_count": int(zero_count),
                    "constant_feature_count": constant_feature_count,
                    "nan_feature_names": nan_names,
                    "min": float(np.nanmin(X)) if X.size else 0.0,
                    "max": float(np.nanmax(X)) if X.size else 0.0,
                    "first10": [float(x) for x in X[0, : min(10, X.shape[1])]],
                    "required_missing_after_optional_impute": required_missing,
                }
            )
        except Exception:
            pass
    return X


def predict_confidence_from_artifact(
    model_pkl_path: str | None,
    snapshot: Dict[str, Any],
    feature_order: List[str] | None = None,
    artifact_dir: str | None = None,
    cand_meta: Dict[str, Any] | None = None,
    *,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Run artifact model inference with quality gates and robust probability extraction."""
    import pickle
    import traceback as tb_mod
    from pathlib import Path as P

    import numpy as np

    cidlog = (cand_meta or {}).get("candidate_id", "unknown")
    debug: Dict[str, Any] = {
        "model_class": "unknown",
        "predict_method": "none",
        "scaler_present": False,
        "scaler_applied": False,
        "calibrator_present": False,
        "calibrator_applied": False,
        "feature_count": 0,
    }

    def _diag(decision: str, out: Dict[str, Any], fo: List[str], feat_dbg: Dict[str, Any]) -> None:
        log_pf_predict_diag(
            candidate_id=cidlog,
            artifact_path=str(out.get("artifact_path") or artifact_dir or model_pkl_path or ""),
            model_type=str(out.get("model_class") or "unknown"),
            required_features_count=len(fo),
            feature_vector_count=int(out.get("feature_count") or len(fo)),
            missing_features=out.get("missing_features") or feat_dbg.get("required_missing_after_optional_impute"),
            nan_count=int(feat_dbg.get("nan_count", 0) or 0),
            zero_count=int(feat_dbg.get("zero_count", 0) or 0),
            constant_feature_count=int(feat_dbg.get("constant_feature_count", 0) or 0),
            scaler_present=bool(out.get("scaler_present")),
            model_present=bool(out.get("model_present")),
            raw_output=out.get("raw"),
            probability=out.get("prob"),
            confidence=out.get("confidence"),
            threshold=threshold,
            decision=decision,
            classes_=out.get("classes_"),
            predict_method=str(out.get("predict_method") or "none"),
            error=str(out.get("error") or ""),
            install_hint=str(out.get("install_hint") or ""),
        )

    pkl_path: Optional[P] = None
    if model_pkl_path:
        pkl_path = P(model_pkl_path)
    elif artifact_dir:
        ad = P(artifact_dir)
        for candp in [ad, ad / (cand_meta or {}).get("candidate_id", ""), ad]:
            mp = candp / "model.pkl"
            if mp.exists():
                pkl_path = mp
                break
            for f in candp.glob("*.pkl"):
                if _looks_like_model_pickle(f):
                    pkl_path = f
                    break
            if pkl_path:
                break

    fo = [str(x) for x in (feature_order or []) if x]
    if not fo and cand_meta:
        fo = [str(x) for x in (cand_meta.get("_feature_order") or cand_meta.get("feature_order") or []) if x]
    if not fo and artifact_dir:
        fsj = P(artifact_dir) / "feature_schema.json"
        if fsj.exists():
            try:
                fsd = json.loads(fsj.read_text(encoding="utf-8"))
                if isinstance(fsd, list):
                    fo = [str(x) for x in fsd if x]
                elif isinstance(fsd, dict):
                    fo = [str(x) for x in (fsd.get("features") or fsd.get("feature_order") or []) if x]
            except Exception:
                pass

    id_ok, id_mismatches = validate_artifact_identity_at_predict(cand_meta, artifact_dir, fo, threshold=threshold)
    if not id_ok:
        out = _blocked_result(
            "ARTIFACT_IDENTITY_MISMATCH",
            debug=debug,
            extra={
                "artifact_path": str(artifact_dir or ""),
                "identity_mismatches": id_mismatches,
                "missing_features": [],
                "model_present": pkl_path is not None and pkl_path.exists(),
            },
        )
        _diag("ARTIFACT_IDENTITY_MISMATCH", out, fo, {})
        return out

    if artifact_expects_xgboost(cand_meta, artifact_dir=artifact_dir) and not xgboost_available():
        out = _blocked_result(
            "XGBOOST_NOT_INSTALLED",
            debug=debug,
            extra={
                "artifact_path": str(artifact_dir or pkl_path or ""),
                "model_present": bool(pkl_path and pkl_path.exists()),
                "install_hint": XGBOOST_INSTALL_HINT,
            },
        )
        _diag("XGBOOST_NOT_INSTALLED", out, fo, {})
        return out

    if not pkl_path or not pkl_path.exists():
        out = _blocked_result(
            "model_pkl_not_found",
            debug=debug,
            extra={"artifact_path": str(artifact_dir or ""), "model_present": False},
        )
        _diag("MODEL_OUTPUT_INVALID", out, fo, {})
        return out

    try:
        with pkl_path.open("rb") as fh:
            bundle = pickle.load(fh)
    except Exception as exc:
        err_code = classify_load_error(
            exc,
            cand_meta=cand_meta,
            artifact_dir=artifact_dir,
        )
        extra: Dict[str, Any] = {"artifact_path": str(pkl_path), "model_present": False}
        if err_code == "XGBOOST_NOT_INSTALLED":
            extra["install_hint"] = XGBOOST_INSTALL_HINT
        out = _blocked_result(err_code, debug=debug, extra=extra)
        _diag(err_code, out, fo, {})
        return out

    inner_model = bundle
    scaler = None
    calibrator = None
    if isinstance(bundle, dict):
        inner_model = bundle.get("model", bundle.get("estimator", bundle))
        scaler = bundle.get("scaler")
        calibrator = bundle.get("calibrator") or bundle.get("calibration_model")
        debug["model_class"] = type(inner_model).__name__
    else:
        debug["model_class"] = type(bundle).__name__

    est, pipe_scaler, pipe_cal, est_name = _unwrap_estimator(inner_model)
    if scaler is None:
        scaler = pipe_scaler
    if calibrator is None:
        calibrator = pipe_cal
    debug["model_class"] = est_name
    if scaler is not None:
        debug["scaler_present"] = True
    if calibrator is not None:
        debug["calibrator_present"] = True

    if not fo:
        fo = [k for k in snapshot.keys() if not str(k).startswith(("future_", "gross_", "net_", "label", "_"))][:200]

    debug["feature_count"] = len(fo)
    optional_feats: List[str] = []
    if artifact_dir:
        for sc in ("preprocessing_metadata.json", "training_metadata.json", "metadata.json", "model_card.json"):
            sp = P(artifact_dir) / sc
            if not sp.exists():
                continue
            try:
                sd = json.loads(sp.read_text(encoding="utf-8"))
                if isinstance(sd, dict):
                    op = sd.get("optional_features") or sd.get("imputable_features") or []
                    if op:
                        optional_feats = [str(x) for x in op if x]
                        break
            except Exception:
                pass
    if cand_meta:
        opm = cand_meta.get("optional_features") or cand_meta.get("imputable_features") or []
        if opm:
            optional_feats = [str(x) for x in opm if x] or optional_feats

    feat_debug: Dict[str, Any] = {}
    X = build_aligned_feature_row(fo, snapshot, feat_debug, optional_features=optional_feats or None)

    ok, reason, qdetail = validate_feature_vector_quality(fo, feat_debug)
    if not ok:
        out = _blocked_result(
            reason,
            debug=debug,
            extra={
                "artifact_path": str(pkl_path),
                "model_present": True,
                "missing_features": qdetail.get("missing_features") or feat_debug.get("required_missing_after_optional_impute"),
                "feature_quality": qdetail,
            },
        )
        _diag(reason, out, fo, feat_debug)
        return out

    try:
        log_feature_x(
            candidate_id=cidlog,
            x_shape=feat_debug.get("X_shape"),
            nan_count=int(feat_debug.get("nan_count", 0) or 0),
            inf_count=int(feat_debug.get("inf_count", 0) or 0),
            zero_count=int(feat_debug.get("zero_count", 0) or 0),
            min_val=feat_debug.get("min"),
            max_val=feat_debug.get("max"),
            first10=feat_debug.get("first10"),
        )
    except Exception:
        pass

    if scaler is not None and hasattr(scaler, "transform"):
        debug["scaler_present"] = True
        try:
            X = scaler.transform(X)
            debug["scaler_applied"] = True
        except Exception as se:
            out = _blocked_result(
                f"scaler_transform_failed:{se}",
                debug=debug,
                extra={"artifact_path": str(pkl_path), "model_present": True, "X_shape": feat_debug.get("X_shape")},
            )
            _diag("MODEL_OUTPUT_INVALID", out, fo, feat_debug)
            return out
    elif artifact_dir and (P(artifact_dir) / "scaler.pkl").exists():
        try:
            with (P(artifact_dir) / "scaler.pkl").open("rb") as sfh:
                scaler = pickle.load(sfh)
            debug["scaler_present"] = True
            X = scaler.transform(X)
            debug["scaler_applied"] = True
        except Exception:
            pass

    try:
        ensemble_components = _extract_xgb_rf_components(inner_model) or _extract_xgb_rf_components(bundle)
        is_xgb = "xgb" in str(type(est)).lower() or str(getattr(est, "__module__", "")).startswith("xgboost")
        if ensemble_components is not None:
            debug["model_class"] = type(inner_model).__name__ if not isinstance(inner_model, dict) else "xgb_rf_ensemble"
            ensemble_details = _infer_xgb_rf_ensemble_details(
                inner_model if ensemble_components is not None and _extract_xgb_rf_components(inner_model) is not None else bundle,
                X,
                fo if fo else None,
            )
            raw = ensemble_details.get("raw")
            prob = ensemble_details.get("prob")
            method = str(ensemble_details.get("method") or "xgb_rf_ensemble_predict_proba")
            debug.update(
                {
                    "ensemble_prob": ensemble_details.get("ensemble_prob"),
                    "xgb_prob": ensemble_details.get("xgb_prob"),
                    "rf_prob": ensemble_details.get("rf_prob"),
                    "model_disagreement": ensemble_details.get("model_disagreement"),
                    "allowed": ensemble_details.get("allowed"),
                    "block_reason": ensemble_details.get("block_reason"),
                    "model_type": "xgb_rf_ensemble",
                }
            )
        elif is_xgb:
            raw, prob, method = _infer_xgb_probability(est, X, fo if fo else None)
        else:
            raw, prob, method = _extract_probability(est if calibrator is None else inner_model, X)
        classes_ = getattr(est, "classes_", None)
        if prob is None or (isinstance(prob, float) and (math.isnan(prob) or math.isinf(prob))):
            out = _blocked_result(
                "MODEL_OUTPUT_INVALID",
                debug=debug,
                extra={
                    "artifact_path": str(pkl_path),
                    "model_present": True,
                    "raw": raw,
                    "X_shape": feat_debug.get("X_shape"),
                },
            )
            _diag("MODEL_OUTPUT_INVALID", out, fo, feat_debug)
            return out

        prob_f = float(max(0.0, min(1.0, prob)))
        out = _merge_predict_out(
            {
                "raw": float(raw) if raw is not None else prob_f,
                "prob": prob_f,
                "confidence": prob_f,
                "error": None,
                "predict_method": method,
                "decision": "OK",
                "artifact_path": str(pkl_path),
                "model_present": True,
                "classes_": list(classes_) if classes_ is not None else None,
                "X_shape": feat_debug.get("X_shape"),
                "missing_features": [],
                "feature_missing_count": 0,
                "feature_invalid_count": 0,
                "ensemble_prob": debug.get("ensemble_prob"),
                "xgb_prob": debug.get("xgb_prob"),
                "rf_prob": debug.get("rf_prob"),
                "model_disagreement": debug.get("model_disagreement"),
                "allowed": debug.get("allowed"),
                "block_reason": debug.get("block_reason"),
                "model_type": debug.get("model_type", debug.get("model_class")),
            },
            debug,
        )
        if debug.get("calibrator_present"):
            out["calibrator_applied"] = method.startswith("calibrated")
        _diag("OK" if prob_f > 0 else "LOW_PROB", out, fo, feat_debug)
        return out
    except Exception as ie:
        ie_msg = str(ie)
        if is_xgboost_missing_error(ie_msg) or (
            artifact_expects_xgboost(cand_meta, model_type=debug.get("model_class", ""), artifact_dir=artifact_dir)
            and not xgboost_available()
        ):
            out = _blocked_result(
                "XGBOOST_NOT_INSTALLED",
                debug=debug,
                extra={
                    "artifact_path": str(pkl_path),
                    "model_present": True,
                    "install_hint": XGBOOST_INSTALL_HINT,
                    "X_shape": feat_debug.get("X_shape"),
                },
            )
            _diag("XGBOOST_NOT_INSTALLED", out, fo, feat_debug)
            return out
        pf_log(
            "ERROR",
            f"[PAPER-FWD-PREDICT-ERROR] cid={cidlog} handled=true exception={type(ie).__name__}:{ie} "
            f"traceback={tb_mod.format_exc()[-200:]}",
            rate_key=f"predict_error:{cidlog}",
            rate_interval=30.0,
        )
        out = _blocked_result(
            "PREDICT_EXCEPTION",
            debug=debug,
            extra={
                "artifact_path": str(pkl_path),
                "model_present": True,
                "route_error": True,
                "route_error_like": True,
                "X_shape": feat_debug.get("X_shape"),
            },
        )
        _diag("PREDICT_EXCEPTION", out, fo, feat_debug)
        return out
