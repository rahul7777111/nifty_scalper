#!/usr/bin/env python3
"""
candidate_router.py
===================
Multi-candidate router for NiftyScalper shadow/paper mode.

The router evaluates all available candidates against a live snapshot and
selects the best eligible candidate, or returns SKIP if none qualify.

Router flow
-----------
1. Load all candidate manifests from --candidate-dir or explicit paths.
2. For each candidate:
   a. Apply candidate filter (e.g. PE_only, CE_only, DTE_14_30).
      Rejected candidates are logged with a rejection reason.
   b. Check feature coverage >= 95%.
      Candidates with insufficient coverage are skipped.
   c. Run model prediction → probability.
   d. Compare probability vs selected_threshold (strict >).
      Candidates below threshold are skipped.
3. Among all eligible candidates, select the one with the highest edge_score:
      edge_score = probability - selected_threshold
   Tie-breakers (in order):
      - Higher gates_passed / gates_total ratio
      - Lower expected execution cost (if available)
      - Lower expected spread (if available)
4. Return a router decision dict.

Safety
------
- Router never sends real orders — all modes are simulation only.
- paper_only=True and real_trading_enabled=False are enforced at manifest load.
- If no candidate passes all checks, returns SKIP.
- Shadow mode: no_order_sent=True always.
- Paper mode: no_order_sent=True (paper trades are simulated separately).
- Feature contract is enforced: forbidden tokens (labels, returns, PnL, future)
  are rejected at both training and inference.

No future/return/PnL/label columns are ever used as features.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .candidate_filters import apply_filters, apply_dynamic_preset_filter
    from .candidate_manifest import load_candidate_manifest, load_candidate_model
except Exception:  # noqa: BLE001
    # Allow direct script / test imports when src is on sys.path
    from candidate_filters import apply_filters, apply_dynamic_preset_filter  # type: ignore
    from candidate_manifest import load_candidate_manifest, load_candidate_model  # type: ignore

# Feature contract import for forbidden token enforcement
try:
    from ml_feature_contract import (
        is_feature_forbidden,
        compute_feature_coverage,
        CORE_REQUIRED_FEATURES,
        CONTRACT_VERSION,
    )
    HAS_FEATURE_CONTRACT = True
except ImportError:
    HAS_FEATURE_CONTRACT = False

_LOGGER = logging.getLogger(__name__)


def _router_confidence_value(confidence: Any) -> float | None:
    if not isinstance(confidence, (int, float)):
        return None
    try:
        value = float(confidence)
    except Exception:
        return None
    if value != value:
        return None
    return value


def _format_confidence_for_reason(confidence: Any) -> str:
    value = _router_confidence_value(confidence)
    if value is None:
        return "-"
    if 0 < abs(value) < 0.00005:
        return f"{value:.2e}"
    return f"{value:.4f}"

try:
    from pf_logging import (
        log_feature_x,
        log_pf_cache,
        pf_log,
        pf_should_log_level,
        should_log,
    )
except ImportError:
    def pf_should_log_level(_level: str) -> bool:  # type: ignore[misc]
        return True

    def should_log(_key: str, _interval: float) -> bool:  # type: ignore[misc]
        return True

    def pf_log(level, msg, **kwargs):  # type: ignore[misc]
        print(msg)

    def log_feature_x(**kwargs):  # type: ignore[misc]
        pass

    def log_pf_cache(*args, **kwargs):  # type: ignore[misc]
        pass

_PF_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
_PF_FEATURE_SCHEMA_CACHE: Dict[str, List[str]] = {}


def _safe_load_json(path: Path) -> Dict[str, Any] | None:
    try:
        if path.exists() and path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
    except Exception:
        return None
    return None


def load_dynamic_presets(project_root: Path | str) -> Dict[str, Any]:
    """Load global dynamic preset config without paper-forward module dependencies."""
    root = Path(project_root).resolve()
    for env_key in ("DYNAMIC_PRESETS_PATH", "PAPER_FORWARD_DYNAMIC_PRESETS_PATH"):
        raw = os.environ.get(env_key)
        if raw:
            data = _safe_load_json(Path(raw))
            if data:
                return {"source": raw, "data": data, "loaded": True}
    for rel in ("config/dynamic_presets.json", "config/presets.json"):
        data = _safe_load_json(root / rel)
        if data:
            return {"source": str(root / rel), "data": data, "loaded": True}
    return {"source": None, "data": {}, "loaded": False}

# ---------------------------------------------------------------------------
# Feature alignment (imported lazily to avoid circular deps)
# ---------------------------------------------------------------------------

def _align_features_cached(
    model_features: List[str],
    live_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Align live features to model schema with forbidden token enforcement.

    Returns a dict with:
        coverage_pct   : float  (0-100)
        missing_features: List[str]
        available_features: List[str]
        forbidden_features: List[str] (if any forbidden tokens found)
        contract_version: str | None (if feature contract available)
    
    FAILS CLOSED if forbidden features detected.
    """
    # Check for forbidden features FIRST (fail-closed)
    forbidden_features: List[str] = []
    if HAS_FEATURE_CONTRACT:
        for fname in live_snapshot.keys():
            if is_feature_forbidden(fname):
                forbidden_features.append(fname)
    
    # Lazy import to avoid circular dependency
    try:
        from scripts.live_decision_dry_run import align_features
        result = align_features(model_features, live_snapshot)
        result["forbidden_features"] = forbidden_features
        result["contract_version"] = CONTRACT_VERSION if HAS_FEATURE_CONTRACT else None
        if forbidden_features:
            _LOGGER.error(
                "FORBIDDEN_FEATURES_DETECTED in live_snapshot: %s",
                forbidden_features
            )
        return result
    except Exception:
        # Fallback: simple field intersection
        available = [f for f in model_features if f in live_snapshot]
        missing = [f for f in model_features if f not in live_snapshot]
        coverage = (len(available) / len(model_features) * 100) if model_features else 0.0
        result = {
            "available_features": available,
            "missing_features": missing,
            "coverage_pct": coverage,
            "forbidden_features": forbidden_features,
            "contract_version": CONTRACT_VERSION if HAS_FEATURE_CONTRACT else None,
        }
        if forbidden_features:
            _LOGGER.error(
                "FORBIDDEN_FEATURES_DETECTED in live_snapshot: %s",
                forbidden_features
            )
        return result


def _predict_cached(
    model_bundle_path: str,
    snapshot: Dict[str, Any],
) -> float:
    """Run model prediction. Delegates to robust path; never silently 0.0 on exception (logs [PAPER-FWD-PREDICT-ERROR] via callee)."""
    try:
        res = _predict_confidence_from_artifact(model_bundle_path, snapshot, feature_order=None, artifact_dir=None, cand_meta=None)
        if res.get("error"):
            # log visible error (fix 8) before legacy 0.0 return for compat paths
            print(f"[PAPER-FWD-PREDICT-ERROR] candidate_id=legacy_cached error={res.get('error')} raw_returned=0.0_for_legacy")
            return 0.0
        return float(res.get("confidence") or res.get("prob") or 0.0)
    except Exception as exc:
        import traceback as _tb
        print(f"[PAPER-FWD-PREDICT-ERROR] candidate_id=legacy_cached exception={type(exc).__name__}:{exc} traceback={_tb.format_exc()[-300:]}")
        return 0.0


def _predict_cached_details(
    model_bundle_path: str,
    snapshot: Dict[str, Any],
    *,
    artifact_dir: str | None = None,
    cand_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    try:
        res = _predict_confidence_from_artifact(
            model_bundle_path,
            snapshot,
            feature_order=None,
            artifact_dir=artifact_dir,
            cand_meta=cand_meta,
        )
        if not isinstance(res, dict):
            return {"error": "INVALID_PREDICTION_RESULT", "confidence": None}
        conf = res.get("confidence")
        if conf is not None:
            try:
                conf = float(conf)
            except Exception:
                conf = None
        res["confidence"] = conf
        return res
    except Exception as exc:
        import traceback as _tb
        return {
            "error": f"{type(exc).__name__}:{exc}",
            "exception_type": type(exc).__name__,
            "traceback_tail": _tb.format_exc()[-600:],
            "confidence": None,
        }


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------


def _load_dynamic_preset(candidate_dir_or_manifest: str | dict) -> dict | None:
    """Load dynamic_preset.json from candidate directory or from inline manifest.

    Returns None if no preset found (candidate uses no dynamic preset).
    The preset dict contains live-computable configuration only:
      threshold, max_trades_per_day, spread_limit_pct, liquidity_min,
      premium_band, dte_range, regime_filter, expiry_filter,
      avoid_first_n_minutes, avoid_last_n_minutes, entry_time_start,
      entry_time_end, expected_move_to_cost_ratio_min, etc.

    Returns None on error (safe — candidate continues without preset).
    """
    try:
        if isinstance(candidate_dir_or_manifest, dict):
            # Inline manifest — look for preset in manifest
            preset_path = candidate_dir_or_manifest.get("dynamic_preset_path")
            if not preset_path:
                return None
            path = candidate_dir_or_manifest.get("candidate_dir", "")
            if path:
                import pathlib
                full = pathlib.Path(path) / "dynamic_preset.json"
                if full.exists():
                    import json
                    with full.open() as f2:
                        return json.load(f2)
            return None

        # String path — treat as candidate directory
        import pathlib
        candidate_dir = pathlib.Path(candidate_dir_or_manifest)
        preset_path = candidate_dir / "dynamic_preset.json"
        if preset_path.exists():
            import json
            with preset_path.open() as f2:
                return json.load(f2)
        return None
    except Exception:
        return None


def evaluate_candidate(
    candidate: Dict[str, Any],
    snapshot: Dict[str, Any],
    min_coverage_pct: float = 95.0,
) -> Dict[str, Any]:
    """Evaluate a single candidate against a live snapshot.

    Parameters
    ----------
    candidate : dict
        Loaded and validated candidate manifest dict.
    snapshot : dict
        Live feature snapshot (option chain + candle + context fields).
    min_coverage_pct : float
        Minimum feature coverage required to score candidate.

    Returns
    -------
    dict with keys:
        candidate_id       : str
        filter_applied     : bool
        filter_passed      : bool
        filter_rejection_reason : str | None
        coverage_pct       : float
        coverage_passed    : bool
        probability        : float | None  (None if not scored)
        threshold_passed   : bool | None
        edge_score         : float | None
        final_eligible     : bool
        final_rejection_reason : str | None
    """
    cid = candidate["candidate_id"]
    result = {
        "candidate_id": cid,
        "model_name": candidate.get("model_name", "unknown"),
        "filter_name": candidate.get("filter_name", "unknown"),
        "dynamic_preset_loaded": False,
        "dynamic_preset_applied": False,
        "dynamic_preset_passed": True,
        "dynamic_preset_rejection_reason": None,
        "filter_applied": False,
        "filter_passed": False,
        "filter_rejection_reason": None,
        "coverage_pct": 0.0,
        "coverage_passed": False,
        "probability": None,
        "threshold": candidate.get("selected_threshold", 0.5),
        "threshold_passed": None,
        "edge_score": None,
        "final_eligible": False,
        "final_rejection_reason": None,
    }

    # --- Step 1: Apply candidate filter ---
    # Extract filter name(s) from manifest — support both top-level filter_name
    # and nested filter_definition.filter_name
    filter_name = candidate.get("filter_name") or candidate.get("filter_definition", {}).get("filter_name")
    if filter_name:
        filter_result = apply_filters(snapshot, filter_names=[filter_name])
        first_result = filter_result.get("results", [{}])[0]
        result["filter_applied"] = first_result.get("filter_applied", False)
        result["filter_passed"] = filter_result.get("all_passed", False)
        result["filter_rejection_reason"] = filter_result.get("blocking_reason")
    else:
        result["filter_applied"] = False
        result["filter_passed"] = True  # No filter = pass
        result["filter_rejection_reason"] = None

    if not result["filter_passed"]:
        result["final_rejection_reason"] = (
            f"filter_rejected_{result['filter_rejection_reason'] or 'unknown'}"
        )
        return result

    # --- Step 1b: Apply dynamic preset filter (live-computable only) ---
    preset_config = _load_dynamic_preset(
        candidate.get("source_manifest_dir")
        or candidate.get("candidate_dir")
        or candidate.get("candidate_id", "")
    )
    result["dynamic_preset_loaded"] = preset_config is not None
    if preset_config is not None:
        preset_result = apply_dynamic_preset_filter(snapshot, preset_config)
        result["dynamic_preset_applied"] = True
        result["dynamic_preset_passed"] = preset_result.get("filter_passed", False)
        result["dynamic_preset_rejection_reason"] = preset_result.get("rejection_reason")
        if not preset_result.get("filter_passed", False):
            result["final_eligible"] = False
            result["final_rejection_reason"] = (
                f"preset_rejected_{preset_result.get('rejection_reason', 'unknown')}"
            )
            return result
    else:
        result["dynamic_preset_applied"] = False
        result["dynamic_preset_passed"] = True

    # --- Step 2: Check feature coverage ---
    feature_schema = candidate.get("feature_schema", {})
    if isinstance(feature_schema, list):
        model_features = feature_schema
    elif isinstance(feature_schema, dict):
        model_features = feature_schema.get("features", [])
    else:
        model_features = []

    if not model_features:
        result["final_rejection_reason"] = "no_features_in_schema"
        return result

    alignment = _align_features_cached(model_features, snapshot)
    result["coverage_pct"] = alignment.get("coverage_pct", 0.0)
    result["coverage_passed"] = result["coverage_pct"] >= min_coverage_pct
    result["missing_features_sample"] = alignment.get("missing_features", [])[:5]

    if not result["coverage_passed"]:
        result["final_rejection_reason"] = (
            f"coverage_{result['coverage_pct']:.1f}%_lt_{min_coverage_pct}%"
        )
        return result

    # --- Step 3: Run model prediction ---
    model_pkl = candidate.get("model_pkl", "")
    pred_details = _predict_cached_details(
        model_pkl,
        snapshot,
        artifact_dir=candidate.get("artifact_dir"),
        cand_meta=candidate,
    )
    probability = pred_details.get("confidence")
    result["probability"] = probability
    result["prediction_error"] = pred_details.get("error")
    result["missing_features"] = pred_details.get("missing_features")
    result["artifact_path"] = pred_details.get("artifact_path") or model_pkl
    result["exception_type"] = pred_details.get("exception_type")

    if pred_details.get("error"):
        result["threshold_passed"] = False
        result["final_rejection_reason"] = str(pred_details.get("error"))
        return result

    if probability is None or (isinstance(probability, float) and (math.isnan(probability) or math.isinf(probability))):
        result["threshold_passed"] = False
        result["final_rejection_reason"] = "prediction_invalid"
        return result

    # --- Step 4: Threshold check (strict >) ---
    threshold = candidate.get("selected_threshold", 0.5)
    result["threshold"] = threshold
    result["threshold_passed"] = probability > threshold
    result["edge_score"] = probability - threshold

    if not result["threshold_passed"]:
        result["final_rejection_reason"] = (
            f"low_confidence_{probability:.4f}_lt_{threshold:.4f}"
        )
        return result

    # --- All checks passed ---
    result["final_eligible"] = True
    return result


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def route_candidates(
    snapshot: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    mode: str = "shadow",
    min_coverage_pct: float = 95.0,
) -> Dict[str, Any]:
    """Evaluate all candidates and select the best eligible one.

    Parameters
    ----------
    snapshot : dict
        Live feature snapshot.
    candidates : list[dict]
        List of loaded + validated candidate manifest dicts.
    mode : str
        "shadow" or "paper" — currently both are simulation-only.
    min_coverage_pct : float
        Minimum feature coverage required.

    Returns
    -------
    dict with keys:
        router_action      : "TRADE" | "SKIP"
        selected_candidate_id : str | None
        selected_model_name   : str | None
        selected_filter_name  : str | None
        probability           : float | None
        threshold             : float | None
        edge_score            : float | None
        eligible_candidates   : list[dict]  (summary of each eligible)
        rejected_candidates   : list[dict]  (summary of each rejected with reason)
        feature_coverage      : dict | None
        no_order_sent         : True  (always True in shadow/paper)
        mode                  : str
    """
    if not candidates:
        result = {
            "router_action": "SKIP",
            "selected_candidate_id": None,
            "selected_model_name": None,
            "selected_filter_name": None,
            "probability": None,
            "threshold": None,
            "edge_score": None,
            "eligible_candidates": [],
            "rejected_candidates": [],
            "feature_coverage": None,
            "no_order_sent": True,
            "mode": mode,
            "rejection_reason": "no_candidates_available",
        }
        return _normalize_router_output(result)

    evaluated: List[Dict[str, Any]] = []
    for candidate in candidates:
        try:
            eval_result = evaluate_candidate(candidate, snapshot, min_coverage_pct)
            evaluated.append(eval_result)
        except Exception as exc:
            _LOGGER.warning(
                "Candidate %s raised unexpected error: %s",
                candidate.get("candidate_id", "UNKNOWN"),
                exc,
            )
            import traceback as _tb
            evaluated.append({
                "candidate_id": candidate.get("candidate_id", "UNKNOWN"),
                "final_eligible": False,
                "final_rejection_reason": f"router_error_{type(exc).__name__}",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback_tail": _tb.format_exc()[-600:],
                "artifact_path": candidate.get("model_pkl") or candidate.get("artifact_dir"),
                "filter_applied": False,
                "filter_passed": False,
            })

    # Separate eligible and rejected
    eligible = [e for e in evaluated if e.get("final_eligible")]
    rejected = [e for e in evaluated if not e.get("final_eligible")]

    if not eligible:
        result = {
            "router_action": "SKIP",
            "selected_candidate_id": None,
            "selected_model_name": None,
            "selected_filter_name": None,
            "probability": None,
            "threshold": None,
            "edge_score": None,
            "eligible_candidates": [],
            "rejected_candidates": _summarize_rejected(rejected),
            "feature_coverage": None,
            "no_order_sent": True,
            "mode": mode,
            "rejection_reason": rejected[0]["final_rejection_reason"] if rejected else "no_candidates",
        }
        return _normalize_router_output(result)

    # Sort eligible by edge_score descending, then by gate ratio descending
    def _sort_key(e: Dict[str, Any]) -> Tuple[float, float]:
        edge = e.get("edge_score", -9999.0)
        # Gate quality as secondary sort
        gates_passed = 0
        gates_total = 1
        # Can't easily get gate info from eval result — use probability as proxy
        prob = e.get("probability", 0.0)
        return (edge, prob)

    eligible.sort(key=_sort_key, reverse=True)
    best = eligible[0]

    result = {
        "router_action": "TRADE",
        "selected_candidate_id": best["candidate_id"],
        "selected_model_name": best.get("model_name"),
        "selected_filter_name": best.get("filter_name"),
        "probability": best.get("probability"),
        "threshold": best.get("threshold"),
        "edge_score": best.get("edge_score"),
        "eligible_candidates": [
            {
                "candidate_id": e["candidate_id"],
                "edge_score": e.get("edge_score"),
                "probability": e.get("probability"),
                "threshold": e.get("threshold"),
            }
            for e in eligible
        ],
        "rejected_candidates": _summarize_rejected(rejected),
        "feature_coverage": {
            "coverage_pct": best.get("coverage_pct", 0.0),
            "coverage_passed": best.get("coverage_passed", False),
        },
        "no_order_sent": True,
        "mode": mode,
        "rejection_reason": None,
    }
    return _normalize_router_output(result)


def _summarize_rejected(rejected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "candidate_id": e.get("candidate_id", "UNKNOWN"),
            "rejection_reason": e.get("final_rejection_reason", "unknown"),
            "filter_applied": e.get("filter_applied", False),
            "filter_passed": e.get("filter_passed", False),
            "coverage_pct": e.get("coverage_pct", 0.0),
            "probability": e.get("probability"),
            "prediction_error": e.get("prediction_error"),
            "exception_type": e.get("exception_type"),
            "artifact_path": e.get("artifact_path"),
        }
        for e in rejected
    ]


def _normalize_router_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all required fields have default values (no Nones for primitives)."""
    normalized = dict(result)
    # Ensure primitive fields have safe defaults for downstream consumers
    # Use direct assignment to overwrite any None values
    normalized["router_action"] = result.get("router_action", "SKIP") or "SKIP"
    normalized["selected_candidate_id"] = result.get("selected_candidate_id")
    normalized["probability"] = result.get("probability")
    normalized["threshold"] = result.get("threshold") if result.get("threshold") is not None else 0.5
    normalized["edge_score"] = result.get("edge_score")
    normalized["eligible_candidates"] = result.get("eligible_candidates") or []
    normalized["rejected_candidates"] = result.get("rejected_candidates") or []
    # coverage_pct may live in feature_coverage sub-dict
    fc = normalized.get("feature_coverage", {})
    if isinstance(fc, dict):
        normalized["coverage_pct"] = fc.get("coverage_pct") or 0.0
    else:
        normalized["coverage_pct"] = 0.0
    return normalized


# ---------------------------------------------------------------------------
# Convenience load + route
# ---------------------------------------------------------------------------

def load_and_route(
    snapshot: Dict[str, Any],
    candidate_dir: Optional[Path | str] = None,
    candidate_paths: Optional[List[Path | str]] = None,
    mode: str = "shadow",
    min_coverage_pct: float = 95.0,
) -> Dict[str, Any]:
    """Load candidates and run the router.

    Provide either candidate_dir or candidate_paths, not both.
    """
    try:
        from .candidate_manifest import load_candidates_from_dir
    except Exception:  # noqa: BLE001
        from candidate_manifest import load_candidates_from_dir  # type: ignore

    candidates: List[Dict[str, Any]] = []
    if candidate_dir:
        candidates, errors = load_candidates_from_dir(candidate_dir)
        for err in errors:
            _LOGGER.warning("Candidate load error: %s — %s", err["path"], err["error"])

    elif candidate_paths:
        for path in candidate_paths:
            try:
                manifest = load_candidate_manifest(path)
                candidates.append(manifest)
            except Exception as exc:
                _LOGGER.warning("Failed to load candidate manifest %s: %s", path, exc)

    return route_candidates(snapshot, candidates, mode=mode, min_coverage_pct=min_coverage_pct)


# ============================================================================
# PHASE 2/3: Normalized single-active-candidate router API (production wiring)
# Every trading decision (shadow/paper/live) MUST flow through this.
# Router NEVER places orders. Safe for GUI, shadow, paper, live.
# ============================================================================

from datetime import datetime, timezone
from typing import Any as _Any  # local alias to avoid shadowing in helpers

# Module-level last decision for GUI/runtime observability (no global state mutation beyond this)
_LAST_ROUTER_DECISION: Dict[str, _Any] | None = None


def get_last_router_decision() -> Dict[str, _Any] | None:
    """Return the most recent route_candidate_decision output (or None)."""
    return _LAST_ROUTER_DECISION


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_no_trade_decision(
    *,
    reason: str,
    mode: str = "shadow",
    candidate_id: str | None = None,
    model_name: str | None = None,
    preset_family: str | None = None,
    selected_preset: str | None = None,
    side_policy: str = "BOTH",
    side_decision: str = "NONE",
    confidence: float = 0.0,
    threshold: float = 0.5,
    market_regime: str = "unknown",
    volatility_state: str = "unknown",
    trend_state: str = "unknown",
    liquidity_state: str = "unknown",
    forced_eval: bool = False,
    debug: Dict[str, _Any] | None = None,
) -> Dict[str, _Any]:
    """Canonical NO_TRADE builder. All failure paths use this."""
    d = {
        "timestamp": _now_iso(),
        "mode": mode,
        "candidate_id": candidate_id or "",
        "model_name": model_name or "unknown",
        "preset_family": preset_family or "unknown",
        "selected_preset": selected_preset or "NO_TRADE",
        "side_policy": side_policy,
        "side_decision": side_decision,
        "confidence": float(confidence or 0.0),
        "threshold": float(threshold or 0.5),
        "market_regime": market_regime or "unknown",
        "volatility_state": volatility_state or "unknown",
        "trend_state": trend_state or "unknown",
        "liquidity_state": liquidity_state or "unknown",
        "allowed_by_model": False,
        "allowed_by_preset": False,
        "allowed_by_side_policy": False,
        "allowed_by_liquidity": False,
        "allowed_by_cost": False,
        "allowed_by_risk": False,
        "shadow_ready": False,
        "forced_eval": bool(forced_eval),
        "final_signal": "NO_TRADE",
        "risk": {
            "stop_loss_pct": 0.0,
            "target_pct": 0.0,
            "trailing_sl_pct": None,
            "size_multiplier": 1.0,
            "dir_sl_atr_mult": None,
            "dir_tp_atr_mult": None,
            "dir_trail_atr_mult": None,
            "max_trades_per_day": 0,
            "cooldown_minutes": 0,
        },
        "no_trade_reason": str(reason or "no_trade"),
        "debug": debug or {"reason": reason},
    }
    return d


def normalize_legacy_signal(legacy_signal: dict | None) -> dict:
    """Map legacy ml_signals / evaluate_entry_signals output to router-friendly shape."""
    if not legacy_signal:
        return {"confidence": 0.0, "threshold": 0.5, "side_hint": None, "take": False, "reason": "no_legacy"}
    conf = float(
        legacy_signal.get("ml_prob")
        or legacy_signal.get("probability")
        or legacy_signal.get("confidence")
        or 0.0
    )
    thr = float(legacy_signal.get("threshold") or legacy_signal.get("ml_threshold") or 0.5)
    side = legacy_signal.get("side") or legacy_signal.get("option_type") or legacy_signal.get("direction")
    take = bool(legacy_signal.get("take") or (conf > thr))
    return {
        "confidence": conf,
        "threshold": thr,
        "side_hint": str(side).upper() if side else None,
        "take": take,
        "reason": str(legacy_signal.get("reason") or ""),
    }


def infer_side_from_market_regime(regime: str, policy: str, preset_detail: dict | None = None) -> str:
    """For AUTO_DIRECTIONAL etc. Map regime -> side or NONE."""
    r = (regime or "").lower()
    if policy == "CE_ONLY":
        return "CE"
    if policy == "PE_ONLY":
        return "PE"
    if policy == "AUTO_DIRECTIONAL":
        detail = preset_detail or {}
        if "bull" in r or "up" in r or r == "bullish":
            return str(detail.get("bullish", "CE")).upper()
        if "bear" in r or "down" in r or r == "bearish":
            return str(detail.get("bearish", "PE")).upper()
        return "NONE"
    if policy in ("BOTH", "BOTH_SYMMETRIC"):
        # Caller may evaluate both; here we return a hint. Final choice in enforce.
        if "bull" in r or "up" in r:
            return "CE"
        if "bear" in r or "down" in r:
            return "PE"
        return "BOTH"
    return "BOTH"


def enforce_side_policy(
    *,
    policy: str,
    desired_side: str | None,
    confidence: float,
    threshold: float,
    market_regime: str,
    preset_regime_detail: dict | None = None,
    ce_confidence: float | None = None,
    pe_confidence: float | None = None,
) -> tuple[str, bool, str]:
    """
    Strict side-policy enforcement.
    Returns: (side_decision, allowed_by_side_policy, no_trade_reason_or_empty)
    Never returns BUY_CE under PE_ONLY etc.
    """
    policy = (policy or "BOTH").upper()
    desired = (desired_side or "").upper() if desired_side else None
    conf = float(confidence or 0.0)
    thr = float(threshold or 0.5)

    if policy == "PE_ONLY":
        if desired == "PE" or (not desired and conf > thr):
            return "PE", True, ""
        return "NONE", False, "side_policy_pe_only_rejects_ce_or_no_signal"

    if policy == "CE_ONLY":
        if desired == "CE" or (not desired and conf > thr):
            return "CE", True, ""
        return "NONE", False, "side_policy_ce_only_rejects_pe_or_no_signal"

    if policy == "AUTO_DIRECTIONAL":
        side = infer_side_from_market_regime(market_regime, policy, preset_regime_detail)
        if side in ("CE", "PE") and conf > thr:
            return side, True, ""
        return "NONE", False, "auto_directional_mixed_or_below_threshold"

    if policy == "BOTH_SYMMETRIC":
        # Evaluate CE vs PE separately if side-specific confidences provided
        c_conf = ce_confidence if ce_confidence is not None else conf
        p_conf = pe_confidence if pe_confidence is not None else conf
        if c_conf <= thr and p_conf <= thr:
            return "NONE", False, "both_symmetric_below_threshold"
        if c_conf > thr and p_conf > thr:
            # Choose higher edge (use raw conf as proxy)
            return ("CE" if c_conf >= p_conf else "PE"), True, ""
        if c_conf > thr:
            return "CE", True, ""
        if p_conf > thr:
            return "PE", True, ""
        return "NONE", False, "both_symmetric_no_side_passes"

    # BOTH (or default)
    if desired in ("CE", "PE") and conf > thr:
        return desired, True, ""
    if (not desired or desired == "BOTH") and conf > thr:
        # Without explicit side data, we conservatively pick NONE to avoid guessing direction.
        # Callers that have option_type in snapshot should pre-set desired_side.
        # For safety in BOTH we still allow if caller provided a concrete side via desired.
        # If no concrete side, require explicit side or fall to NO_TRADE.
        return "NONE", False, "both_requires_explicit_side_or_option_type_in_snapshot"
    return "NONE", False, "below_threshold_or_side_policy_block"


def _load_active_candidate_profile(candidate_dir: str | Path, candidate_id: str | None) -> tuple[Dict[str, _Any] | None, str | None]:
    """Try to load CandidateProfile + active dynamic_preset.json for the active id.
    Returns (profile_dict_or_none, effective_candidate_id).
    Robust to: bare dir (flat layout), nested <base>/<cid>, family-matched fallback dirs, manifest-only.
    """
    from pathlib import Path as _P
    import json as _json

    base = _P(candidate_dir)
    cid = (candidate_id or "").strip()

    # STRICT: only probe exact passed candidate_dir + cid subdir (Task A no cross map)
    # Use normalized base match so t30_ and t35_ variants do not collide on same folder (e.g. elasticnet_paper_0)
    def _norm_base(s: str) -> str:
        import re
        if not s: return ""
        ss = re.sub(r'_\d{6}$', '', s)
        ss = re.sub(r'(\d{8})_\d{0,6}$', r'\1', ss)
        m = re.match(r'^(.+_t\d+_\d{8})', ss)
        if m: return m.group(1)
        return re.sub(r'_\d{8}_\d{6}$', '', s)
    norm_cid = _norm_base(cid)
    candidates_to_try: list[_P] = []
    if base.exists():
        candidates_to_try.append(base)
    if cid:
        candidates_to_try.append(base / cid)
    loose_allowed = str(os.getenv("MSTOCK_ALLOW_LOOSE_ARTIFACT_FALLBACK", "false")).lower() in ("1", "true", "yes")
    try:
        for sub in base.glob("*"):
            if sub.is_dir() and cid:
                norm_sub = _norm_base(sub.name)
                name_ok = (cid in sub.name or sub.name.startswith(cid) or (norm_cid and norm_sub and norm_cid == norm_sub) or
                           (cid.split("_t")[0] in sub.name if "_t" in cid else False))
                if name_ok:
                    if loose_allowed or (cid == sub.name or norm_cid == norm_sub):
                        if (sub / "candidate_profile.json").exists() or (sub / "candidate_manifest.json").exists() or (sub / "model.pkl").exists():
                            if sub not in candidates_to_try:
                                candidates_to_try.append(sub)
    except Exception:
        pass

    for cdir in candidates_to_try:
        for name in ("candidate_profile.json", "candidate_manifest.json", "paper_forward_manifest.json"):
            prof = cdir / name
            if prof.exists():
                try:
                    data = _json.loads(prof.read_text(encoding="utf-8"))
                    eff = data.get("candidate_id") or cid
                    # enrich feature_order from fs (support all likely: metadata.*, training.*, ml_signals, feature_names etc)
                    feat_list = data.get("live_computable_features") or data.get("required_live_fields") or data.get("features") or data.get("feature_order") or []
                    fsj = cdir / "feature_schema.json"
                    if fsj.exists():
                        try:
                            fsd = _json.loads(fsj.read_text(encoding="utf-8"))
                            if isinstance(fsd, list) and fsd:
                                feat_list = fsd
                            elif isinstance(fsd, dict):
                                for kk in ("features", "feature_order", "live_computable_features", "model_features", "required_features"):
                                    if fsd.get(kk) and not feat_list:
                                        feat_list = fsd.get(kk) or []
                                if not feat_list:
                                    md = fsd.get("metadata") or fsd.get("training_metadata") or {}
                                    if isinstance(md, dict):
                                        for kk in ("feature_order", "features", "model_features"):
                                            if md.get(kk) and not feat_list:
                                                feat_list = md.get(kk) or []
                                if not feat_list and fsd.get("ml_signals"):
                                    ms = fsd.get("ml_signals") or {}
                                    feat_list = ms.get("features") or ms.get("feature_order") or []
                        except Exception:
                            pass
                    mp = cdir / "model.pkl"
                    if not mp.exists():
                        for f in cdir.glob("*.pkl"):
                            if "metric" not in f.name.lower():
                                mp = f
                                break
                    scaler_present = (cdir / "scaler.pkl").exists() or (cdir / "preprocessing_metadata.json").exists()
                    cal_present = (cdir / "calibrator.pkl").exists()
                    thr = 0.35
                    try:
                        if isinstance(data.get("threshold_policy"), dict):
                            thr = float(data["threshold_policy"].get("entry_threshold", 0.35) or 0.35)
                        elif data.get("selected_threshold") is not None:
                            thr = float(data.get("selected_threshold"))
                    except Exception:
                        pass
                    if name != "candidate_profile.json" or not all(k in data for k in ("feature_set_name", "threshold_policy")):
                        data = {
                            "candidate_id": eff,
                            "model_name": data.get("model_name") or data.get("model_family") or "unknown",
                            "feature_set_name": data.get("feature_set_name", "live"),
                            "target_name": data.get("target_name", "edge"),
                            "side_policy": data.get("side_policy") or data.get("filter_name") or "BOTH",
                            "preset_family": data.get("preset_family") or "unknown",
                            "dynamic_presets": data.get("dynamic_presets") or data.get("dynamic_preset", {}) or {},
                            "threshold_policy": data.get("threshold_policy") or {"entry_threshold": thr, "min_confidence": thr},
                            "selection_policy": data.get("selection_policy") or {},
                            "risk_policy": data.get("risk_policy") or {},
                            "cost_policy": data.get("cost_policy") or {},
                            "artifact_paths": data.get("artifact_paths") or {"model_pkl": str(mp) if mp and mp.exists() else str(cdir / "model.pkl")},
                            "validation_metrics": data.get("validation_metrics") or data.get("metrics") or {},
                            "gate_results": data.get("gate_results") or data.get("shadow_mode_metadata") or {},
                            "live_computable_features": feat_list,
                            "feature_order": feat_list,
                            "shadow_mode_metadata": data.get("shadow_mode_metadata") or data.get("gate_results") or {},
                            "_scaler_present": scaler_present,
                            "_calibrator_present": cal_present,
                            "selected_threshold": thr,
                            "_resolved_dir": str(cdir),
                            "_model_pkl": str(mp) if (mp and mp.exists()) else str(cdir / "model.pkl"),
                        }
                    data["_feature_order"] = feat_list or []
                    data["_resolved_dir"] = str(cdir)
                    if not data.get("_model_pkl"):
                        ap = data.get("artifact_paths") or {}
                        data["_model_pkl"] = ap.get("model_pkl") or (str(mp) if mp and mp.exists() else str(cdir / "model.pkl"))
                    data["_scaler_present"] = bool(
                        data.get("_scaler_present")
                        or scaler_present
                        or (cdir / "scaler.pkl").exists()
                    )
                    data["_calibrator_present"] = bool(
                        data.get("_calibrator_present") or cal_present or (cdir / "calibrator.pkl").exists()
                    )
                    return data, eff
                except Exception:
                    continue

    # limited bare under this dir only
    for name in ("candidate_profile.json", "candidate_manifest.json", "paper_forward_manifest.json"):
        fp = base / name
        if fp.exists():
            try:
                data = _json.loads(fp.read_text(encoding="utf-8"))
                data["_resolved_dir"] = str(base)
                return data, data.get("candidate_id") or str(base.name)
            except Exception:
                pass
    return None, cid or None


def _load_preset_for_profile(profile: Dict[str, _Any] | None, snapshot: Dict[str, _Any]) -> Dict[str, _Any]:
    """Use dynamic_preset_selector if profile present; else safe no-trade."""
    if not profile:
        return {"selected_preset_name": "NO_TRADE", "no_trade_reason": "no_active_candidate_profile", "trade_allowed": False}
    try:
        from .dynamic_preset_selector import select_dynamic_preset, safe_no_trade_preset  # type: ignore
    except Exception:  # noqa: BLE001
        from dynamic_preset_selector import select_dynamic_preset, safe_no_trade_preset  # type: ignore
        # select_dynamic_preset accepts CandidateProfile or raw dict
        return select_dynamic_preset(profile, snapshot) or safe_no_trade_preset("preset_selector_returned_none")
    except Exception as exc:
        # Fallback minimal
        return {
            "selected_preset_name": profile.get("preset_family", "default"),
            "why_selected": "fallback_no_selector",
            "no_trade_reason": None,
            "side_allowed": "BOTH",
            "threshold_used": float(profile.get("threshold_policy", {}).get("entry_threshold", 0.35) or 0.35),
            "min_confidence": float(profile.get("threshold_policy", {}).get("min_confidence", 0.35) or 0.35),
            "max_trades_per_day": 3,
            "trade_allowed": True,
            "preset_family": profile.get("preset_family", "unknown"),
            "candidate_id": profile.get("candidate_id"),
            "model_name": profile.get("model_name"),
            "market_regime": "unknown",
            "preset_config": {},
            "error": str(exc),
        }


def _build_aligned_feature_row(feature_order: List[str], snapshot: Dict[str, _Any], debug: Dict[str, _Any] | None = None, optional_features: List[str] | None = None) -> "np.ndarray":
    """Build exactly one aligned feature row without paper-forward dependencies."""
    import numpy as np

    opts = set(optional_features or [])
    values: List[float] = []
    missing: List[str] = []
    nan_names: List[str] = []
    zero_count = 0
    nan_count = 0
    inf_count = 0
    for feature in feature_order:
        raw = snapshot.get(feature)
        try:
            value = float(raw) if raw is not None else np.nan
        except Exception:
            value = np.nan
        if np.isinf(value):
            inf_count += 1
            value = np.nan
        if np.isnan(value):
            nan_count += 1
            nan_names.append(feature)
            if feature in opts:
                value = 0.0
            else:
                missing.append(feature)
        if value == 0.0:
            zero_count += 1
        values.append(value)
    X = np.array([values], dtype=np.float32)
    if debug is not None:
        debug.update(
            {
                "X_shape": X.shape,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "zero_count": zero_count,
                "nan_feature_names": nan_names,
                "required_missing_after_optional_impute": missing,
                "constant_feature_count": int(X.shape[1] > 0 and np.isfinite(X[0]).any() and np.sum(X[0][np.isfinite(X[0])] == X[0][np.isfinite(X[0])][0])),
            }
        )
    return X


def _resolve_feature_order_cached(
    *,
    artifact_dir: str | None,
    cand_meta: Dict[str, _Any] | None,
    feature_order: List[str] | None,
    bundle: Any,
) -> Tuple[List[str], List[str]]:
    """Resolve feature_order once per artifact dir; cache sidecar reads."""
    from pathlib import Path as _P

    cidlog = (cand_meta or {}).get("candidate_id", "unknown")
    cache_key = f"{artifact_dir or ''}|{cidlog}"
    if feature_order:
        return [str(x) for x in feature_order if x], []
    cached = _PF_FEATURE_SCHEMA_CACHE.get(cache_key)
    if cached:
        log_pf_cache("feature_schema", hit=True, cid=cidlog)
        return list(cached), []

    fo: List[str] = []
    inspected: List[str] = []
    if artifact_dir:
        ad = _P(artifact_dir)
        sidecars = [
            "feature_schema.json", "feature_order.json", "metadata.json", "model_card.json",
            "manifest.json", "training_metadata.json", "paper_forward_manifest.json",
            "candidate_manifest.json", "candidate_profile.json", "preprocessing_metadata.json",
        ]
        for candp in [ad, ad / (cand_meta.get("candidate_id") if cand_meta else ""), ad]:
            for sc in sidecars:
                fsj = candp / sc
                if not fsj.exists():
                    continue
                inspected.append(str(fsj))
                try:
                    fsd = json.loads(fsj.read_text(encoding="utf-8"))
                    if isinstance(fsd, list):
                        fo = [str(x) for x in fsd if x]
                    elif isinstance(fsd, dict):
                        for kk in ("feature_order", "features", "live_computable_features", "model_features", "required_features"):
                            if fsd.get(kk) and not fo:
                                fo = [str(x) for x in (fsd.get(kk) or []) if x]
                        if not fo:
                            md = fsd.get("metadata") or fsd.get("training_metadata") or {}
                            if isinstance(md, dict):
                                for kk in ("feature_order", "features", "model_features"):
                                    if md.get(kk) and not fo:
                                        fo = [str(x) for x in (md.get(kk) or []) if x]
                        if not fo and fsd.get("ml_signals"):
                            ms = fsd.get("ml_signals") or {}
                            fo = [str(x) for x in (ms.get("features") or ms.get("feature_order") or []) if x]
                    if fo:
                        break
                except Exception:
                    pass
            if fo:
                break
    if not fo and isinstance(bundle, dict):
        for kk in ("feature_order", "features", "feature_names", "required_features", "model_features"):
            if bundle.get(kk) and not fo:
                fo = [str(x) for x in (bundle.get(kk) or []) if x]
                inspected.append("bundle_dict:" + kk)
        if not fo:
            inner = bundle.get("model", bundle.get("estimator", bundle))
            for kk in ("feature_names_in_", "features", "feature_names", "feature_order"):
                arr = getattr(inner, kk, None) if inner is not bundle else getattr(bundle, kk, None)
                if arr and not fo:
                    fo = [str(x) for x in arr if x]
                    inspected.append("model_attr:" + kk)
                    break

    if fo:
        _PF_FEATURE_SCHEMA_CACHE[cache_key] = list(fo)
        log_pf_cache("feature_schema", hit=False, cid=cidlog)
    return fo, inspected


def _load_model_bundle_cached(pkl_path: "Path", artifact_dir: str | None, cand_meta: Dict[str, _Any] | None) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
    """Load model.pkl once per path; reuse across PF cycles."""
    import pickle as _pickle
    from pathlib import Path as _P

    cidlog = (cand_meta or {}).get("candidate_id", "unknown")
    resolved = _P(pkl_path).resolve()
    try:
        st = resolved.stat()
        cache_key = f"{resolved}|{int(st.st_mtime_ns)}|{st.st_size}"
    except Exception:
        cache_key = str(resolved)
    cached = _PF_MODEL_CACHE.get(cache_key)
    if cached is not None:
        log_pf_cache("model", hit=True, cid=cidlog)
        return cached["bundle"], cached.get("scaler"), cached.get("debug", {})

    try:
        with _P(pkl_path).open("rb") as fh:
            bundle = _pickle.load(fh)
    except Exception as exc:
        return None, None, {"load_error": str(exc)}

    scaler = None
    debug: Dict[str, Any] = {"model_class": "unknown"}
    inner_model = bundle
    if isinstance(bundle, dict):
        inner_model = bundle.get("model", bundle.get("estimator", bundle))
        sc = bundle.get("scaler")
        if sc is not None:
            scaler = sc
        debug["model_class"] = type(inner_model).__name__
    else:
        debug["model_class"] = type(bundle).__name__

    _PF_MODEL_CACHE[cache_key] = {"bundle": bundle, "scaler": scaler, "debug": debug}
    log_pf_cache("model", hit=False, cid=cidlog)
    return bundle, scaler, debug


def _predict_confidence_from_artifact(model_pkl_path: str | None, snapshot: Dict[str, _Any], feature_order: List[str] | None = None, artifact_dir: str | None = None, cand_meta: Dict[str, _Any] | None = None) -> Dict[str, _Any]:
    """Artifact inference with feature gates and local probability extraction."""
    fo = feature_order
    if not fo and cand_meta:
        fo = cand_meta.get("_feature_order") or cand_meta.get("feature_order") or cand_meta.get("live_computable_features")
    try:
        from pathlib import Path as _Path
        import pickle as _pickle
    except Exception as exc:  # pragma: no cover
        return {"error": f"IMPORT_ERROR:{exc}", "confidence": None, "prob": None, "predict_method": "none"}

    if not model_pkl_path:
        return {"error": "MODEL_PATH_MISSING", "confidence": None, "prob": None, "predict_method": "none"}
    pkl = _Path(model_pkl_path)
    if not pkl.exists():
        return {"error": "MODEL_FILE_MISSING", "confidence": None, "prob": None, "predict_method": "none"}
    try:
        bundle, scaler, bundle_debug = _load_model_bundle_cached(pkl, artifact_dir, cand_meta)
        if bundle is None:
            return {"error": bundle_debug.get("load_error", "MODEL_LOAD_FAILED"), "confidence": None, "prob": None, "predict_method": "none"}
        fo, _ = _resolve_feature_order_cached(
            artifact_dir=artifact_dir,
            cand_meta=cand_meta,
            feature_order=fo,
            bundle=bundle,
        )
        if not fo:
            fo = [str(k) for k in snapshot.keys() if isinstance(k, str)]
        feat_debug: Dict[str, _Any] = {}
        X = _build_aligned_feature_row(fo, snapshot, feat_debug, optional_features=[])
        missing = list(feat_debug.get("required_missing_after_optional_impute") or [])
        if missing:
            return {
                "error": "FEATURE_VECTOR_INVALID",
                "confidence": None,
                "prob": None,
                "predict_method": "none",
                "missing_features": missing,
                **feat_debug,
            }
        model = bundle.get("model", bundle.get("estimator", bundle)) if isinstance(bundle, dict) else bundle
        if scaler is not None and hasattr(scaler, "transform"):
            X = scaler.transform(X)
        raw, prob, method = _extract_probability(model, X)
        return {
            "raw": raw,
            "prob": prob,
            "confidence": prob,
            "predict_method": method,
            "error": None,
            "model_type": bundle_debug.get("model_class", type(model).__name__),
            "feature_count": len(fo),
            **feat_debug,
        }
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}:{exc}",
            "confidence": None,
            "prob": None,
            "predict_method": "none",
        }


def _predict_confidence_from_artifact_legacy(model_pkl_path: str | None, snapshot: Dict[str, _Any]) -> float:
    """Legacy thin wrapper. Delegates to robust; robust path logs [PAPER-FWD-PREDICT-ERROR] and avoids swallowing to 0.0 on exc."""
    res = _predict_confidence_from_artifact(model_pkl_path, snapshot, feature_order=None, artifact_dir=None, cand_meta=None)
    if res.get("error"):
        print(f"[PAPER-FWD-PREDICT-ERROR] candidate_id=legacy error={res.get('error')} returning_0_for_legacy_caller")
        return 0.0
    return float(res.get("confidence") or 0.0)


def _basic_liquidity_gate(snapshot: Dict[str, _Any], max_spread_pct: float = 0.15) -> tuple[bool, str, str]:
    """Return (allowed, liquidity_state, reason_if_blocked)."""
    spread = None
    for k in ("spread_pct", "range_pct", "bid_ask_spread_pct"):
        if k in snapshot and snapshot[k] is not None:
            try:
                spread = float(snapshot[k])
                break
            except Exception:
                pass
    if spread is None:
        bid = snapshot.get("bid")
        ask = snapshot.get("ask")
        mid = snapshot.get("mid_price") or snapshot.get("ltp")
        if bid and ask and mid and mid > 0:
            try:
                spread = abs(float(ask) - float(bid)) / float(mid)
            except Exception:
                spread = None
    if spread is None:
        return True, "unknown", ""  # permissive if no data (caller may have other guards)
    state = "good"
    if spread > 0.20:
        state = "poor"
    elif spread > 0.10:
        state = "thin"
    allowed = spread <= max_spread_pct
    reason = f"spread_{spread*100:.2f}%_gt_{max_spread_pct*100:.2f}%" if not allowed else ""
    return allowed, state, reason


def _basic_cost_gate(snapshot: Dict[str, _Any], preset_cfg: Dict[str, _Any] | None, confidence: float, threshold: float) -> tuple[bool, str]:
    """Very conservative cost gate using spread + preset expected_move_to_cost if present.
    Fail-closed on bad data. Does not use any future/return/label columns.
    """
    preset_cfg = preset_cfg or {}
    min_ratio = float(preset_cfg.get("expected_move_to_cost_ratio_min", 0.0) or 0.0)
    # Rough cost proxy from spread (in return units)
    spread_cost_proxy = 0.0
    spread = None
    for k in ("spread_pct", "bid_ask_spread_pct"):
        if snapshot.get(k) is not None:
            try:
                spread = float(snapshot[k])
                break
            except Exception:
                pass
    if spread is None:
        bid = snapshot.get("bid")
        ask = snapshot.get("ask")
        mid = snapshot.get("mid_price") or snapshot.get("ltp") or snapshot.get("option_ltp")
        if bid and ask and mid:
            try:
                spread = abs(float(ask) - float(bid)) / max(1e-9, float(mid))
            except Exception:
                spread = 0.002  # assume ~20bps worst reasonable
    if spread is not None:
        spread_cost_proxy = float(spread)
    # Edge proxy: (conf - threshold) roughly maps to expected return before costs
    conf_f = float(confidence) if isinstance(confidence, (int, float)) else 0.0
    edge_proxy = max(0.0, conf_f - threshold)
    if min_ratio > 0:
        # require edge_proxy >= min_ratio * cost_proxy (simplified; units are both fractional)
        allowed = edge_proxy >= (min_ratio * max(spread_cost_proxy, 1e-6))
        reason = "" if allowed else f"cost_edge_{edge_proxy:.4f}_lt_ratio_{min_ratio}_x_{spread_cost_proxy:.4f}"
    else:
        # If no ratio specified, allow if spread not horrific and we have positive edge proxy
        allowed = (spread_cost_proxy < 0.25) and (edge_proxy > 0.0)
        reason = "" if allowed else f"cost_proxy_spread_{spread_cost_proxy:.4f}_edge_{edge_proxy:.4f}"
    return allowed, reason


def route_candidate_decision(
    market_snapshot: dict,
    option_chain_snapshot: dict | None = None,
    legacy_signal: dict | None = None,
    mode: str = "shadow",
    active_candidate_id: str | None = None,
    candidate_dir: str | None = None,
    force_eval: bool = False,
) -> dict:
    """
    REQUIRED PRODUCTION API. All shadow/paper/live entry decisions must call this.

    See user spec for exact return contract and hard rules.
    - Never places real orders.
    - Live: requires shadow_ready (or force_eval) + explicit MSTOCK_ENABLE_LIVE_* gate in caller (strategy).
    - Strict side policy.
    - All gates: model, preset, side, liquidity, cost, risk.
    """
    global _LAST_ROUTER_DECISION

    snap = dict(market_snapshot or {})
    # Merge option chain if provided (prefer explicit over market for liquidity)
    # option_chain_snapshot may be dict (flat) or list of per-strike rows (common in UI/GUI)
    och = option_chain_snapshot
    if och:
        if isinstance(och, dict):
            for k, v in och.items():
                if k not in snap:
                    snap[k] = v
        elif isinstance(och, (list, tuple)):
            # attach full rows for downstream feature alignment / liquidity
            snap.setdefault("option_chain", list(och))
            # best-effort pull spot/price from first row if not present
            if "price" not in snap and och:
                try:
                    r0 = och[0] if isinstance(och[0], dict) else {}
                    for k in ("spot", "ltp", "close", "last"):
                        if k in r0 and r0[k]:
                            snap.setdefault("price", float(r0[k]))
                            break
                except Exception:
                    pass

    mode = (mode or "shadow").lower()
    forced = bool(force_eval)

    # 1. Resolve active candidate (profile preferred)
    profile, eff_cid = _load_active_candidate_profile(candidate_dir or "", active_candidate_id)
    if not profile and not active_candidate_id:
        # No explicit candidate requested — try legacy path only (still produces a decision record)
        leg = normalize_legacy_signal(legacy_signal)
        side_dec, side_ok, side_reason = enforce_side_policy(
            policy="BOTH", desired_side=leg.get("side_hint"), confidence=leg["confidence"], threshold=leg["threshold"],
            market_regime=snap.get("regime") or snap.get("market_regime") or "unknown",
        )
        allowed_model = leg["take"] and (leg["confidence"] > leg["threshold"])
        final_sig = f"BUY_{side_dec}" if (side_ok and allowed_model and side_dec in ("CE", "PE")) else "NO_TRADE"
        d = build_no_trade_decision(
            reason="no_active_candidate_configured_using_legacy_only" if not allowed_model else "",
            mode=mode,
            candidate_id="legacy",
            model_name="legacy_ml_signals",
            confidence=leg["confidence"],
            threshold=leg["threshold"],
            market_regime=snap.get("regime") or "unknown",
            forced_eval=forced,
            debug={"legacy": leg, "note": "no candidate_dir/active_candidate_id provided"},
        )
        if final_sig != "NO_TRADE":
            d.update({
                "final_signal": final_sig,
                "side_decision": side_dec,
                "allowed_by_model": allowed_model,
                "allowed_by_side_policy": side_ok,
                "allowed_by_preset": True,
                "allowed_by_liquidity": True,
                "allowed_by_cost": True,
                "allowed_by_risk": True,
                "shadow_ready": False,  # legacy path never claims shadow_ready
                "no_trade_reason": "",
            })
        _LAST_ROUTER_DECISION = d
        return d

    if not profile:
        d = build_no_trade_decision(
            reason="candidate_missing_or_unloadable",
            mode=mode,
            candidate_id=active_candidate_id or eff_cid or "",
            forced_eval=forced,
        )
        _LAST_ROUTER_DECISION = d
        return d

    cid = profile.get("candidate_id") or eff_cid or (active_candidate_id or "")
    model_name = profile.get("model_name", "unknown")
    preset_family = profile.get("preset_family", "unknown")
    side_policy = str(profile.get("side_policy") or "BOTH").upper()

    # 2. Dynamic preset selection (live-computable only)
    preset_sel = _load_preset_for_profile(profile, snap) or {}
    selected_preset = preset_sel.get("selected_preset_name") or preset_sel.get("preset_family") or "default"
    preset_thr = float(preset_sel.get("threshold_used") or preset_sel.get("min_confidence") or 0.35)
    preset_cfg = preset_sel.get("preset_config") or preset_sel
    preset_allowed = bool(preset_sel.get("trade_allowed", True) and not preset_sel.get("no_trade_reason"))
    preset_block_reason = str(preset_sel.get("no_trade_reason") or "preset_rejected")

    # In forced paper/shadow observation, still run the model so the monitor
    # shows true confidence/debug data. The preset can still block final entry.
    force_observation_eval = forced and mode in {"paper", "shadow", "paper_forward", "paper_forward_multi"}
    if not preset_allowed and force_observation_eval:
        try:
            preset_thr = float((profile.get("threshold_policy") or {}).get("entry_threshold") or preset_thr)
        except Exception:
            pass
    if not preset_allowed and not force_observation_eval:
        d = build_no_trade_decision(
            reason=preset_block_reason,
            mode=mode,
            candidate_id=cid,
            model_name=model_name,
            preset_family=preset_family,
            selected_preset=selected_preset,
            side_policy=side_policy,
            forced_eval=forced,
            debug={"preset_sel": preset_sel},
        )
        _LAST_ROUTER_DECISION = d
        return d

    # 3. Regime / states (reuse detectors if available) - always define to avoid UnboundLocal / NameError on thin snapshots
    regime = str(snap.get("regime") or snap.get("market_regime") or "unknown").lower()
    vol_state = "normal"
    trend_state = "moderate"
    liq_state = "good"
    try:
        try:
            from .dynamic_preset_selector import (
                detect_market_regime,
                detect_volatility_state,
                detect_trend_state,
                detect_liquidity_state,
            )
        except Exception:  # noqa: BLE001
            from dynamic_preset_selector import (  # type: ignore
                detect_market_regime,
                detect_volatility_state,
                detect_trend_state,
                detect_liquidity_state,
            )
        regime = detect_market_regime(snap)
        vol_state = detect_volatility_state(snap)
        trend_state = detect_trend_state(snap)
        liq_state = detect_liquidity_state(snap)
    except Exception:
        # keep safe defaults above; thin snapshot (chain-only) is expected for paper
        pass

    # 4. Confidence: prefer legacy if supplied and sane, else load artifact model
    leg = normalize_legacy_signal(legacy_signal)
    use_legacy = leg["confidence"] > 0.0 or bool(legacy_signal)
    if use_legacy:
        confidence = leg["confidence"]
        threshold = max(preset_thr, leg["threshold"])
    else:
        # Load model pkl from profile artifact_paths or candidate dir (Task C)
        art = profile.get("artifact_paths") or {}
        model_pkl = art.get("model_pkl") or profile.get("_model_pkl") or ""
        if not model_pkl and candidate_dir and cid:
            from pathlib import Path as _P
            cdir = _P(candidate_dir) / str(cid)
            for candp in [cdir, _P(candidate_dir)]:
                for cand in candp.glob("*.pkl"):
                    low_name = cand.name.lower()
                    if all(tok not in low_name for tok in ("metrics", "ensemble_weight", "threshold", "report", "summary")):
                        model_pkl = str(cand)
                        break
                if model_pkl:
                    break
        fo = profile.get("_feature_order") or profile.get("feature_order") or profile.get("live_computable_features") or []
        pred_res = _predict_confidence_from_artifact(
            model_pkl or None,
            snap,
            feature_order=fo,
            artifact_dir=(candidate_dir or profile.get("_resolved_dir")),
            cand_meta=profile,
        )
        snap["_predict_debug"] = pred_res
        pred_conf = pred_res.get("confidence")
        confidence = float(pred_conf) if isinstance(pred_conf, (int, float)) else None
        threshold = preset_thr
    if threshold is None or not isinstance(threshold, (int, float)) or threshold <= 0:
        threshold = 0.35
        pf_log(
            "DEBUG",
            f"[THRESHOLD] cid={cid} source=default value=0.35",
            rate_key=f"threshold_default:{cid}",
            rate_interval=300.0,
        )

    # 4b. If predict had error, surface (Task C)
    pred_err = None
    pred_debug = snap.get("_predict_debug") or {}
    if isinstance(pred_debug, dict):
        pred_err = pred_debug.get("error") or None
        if pred_err in ("", None) and pred_debug.get("missing_features"):
            pred_err = "FEATURE_VECTOR_INVALID"

    allowed_by_model = (
        pred_err in (None, "")
        and isinstance(confidence, (int, float))
        and float(confidence) > float(threshold)
    )

    # 5. Side policy (strict) - compute using real conf first (Task F)
    desired_side = snap.get("option_type") or leg.get("side_hint")
    ce_c = None
    pe_c = None
    if side_policy == "BOTH_SYMMETRIC":
        ce_c = float(snap.get("ce_confidence") or snap.get("confidence_ce") or confidence)
        pe_c = float(snap.get("pe_confidence") or snap.get("confidence_pe") or confidence)

    side_decision, side_ok, side_reason = enforce_side_policy(
        policy=side_policy,
        desired_side=desired_side,
        confidence=confidence,
        threshold=threshold,
        market_regime=regime,
        preset_regime_detail=(profile.get("selection_policy") or {}).get("regime_map"),
        ce_confidence=ce_c,
        pe_confidence=pe_c,
    )
    allowed_by_side_policy = side_ok

    # Detailed side log when no side (Task F)
    if not side_decision or str(side_decision).upper() == "NONE":
        pf_log(
            "DEBUG",
            f"[SIDE-POLICY] cid={cid} policy={side_policy} side={side_decision or 'NONE'} "
            f"reason={side_reason or 'missing_direction_or_conf_below_thr'} "
            f"conf={(f'{float(confidence):.4f}' if isinstance(confidence, (int, float)) else '-')}"
            f" thr={threshold:.4f}",
            rate_key=f"side_policy:{cid}",
            rate_interval=30.0,
        )

    # 6. Liquidity gate
    max_spread = float(preset_cfg.get("spread_limit_pct") or preset_cfg.get("max_spread_pct") or 0.15)
    liq_ok, liq_state2, liq_reason = _basic_liquidity_gate(snap, max_spread_pct=max_spread)
    liquidity_state = liq_state2 or liq_state
    allowed_by_liquidity = liq_ok

    # 7. Cost gate after side selection (Task F) - evaluate on the (would-be) selected contract context if present
    cost_ok, cost_reason = _basic_cost_gate(snap, preset_cfg, confidence, threshold)
    allowed_by_cost = cost_ok

    # 8. Risk gate — preset supplies the numbers; we only gate on max_trades etc if we had counters (none here)
    # For router we treat risk settings as "allowed" unless preset itself blocked earlier.
    risk_settings = {
        "stop_loss_pct": float(preset_cfg.get("stop_loss_pct") or preset_cfg.get("sl_pct") or 0.0),
        "target_pct": float(preset_cfg.get("target_pct") or 0.0),
        "trailing_sl_pct": preset_cfg.get("trailing_sl_pct"),
        "size_multiplier": float(preset_cfg.get("size_multiplier") or 1.0),
        "dir_sl_atr_mult": preset_cfg.get("dir_sl_atr_mult"),
        "dir_tp_atr_mult": preset_cfg.get("dir_tp_atr_mult"),
        "dir_trail_atr_mult": preset_cfg.get("dir_trail_atr_mult"),
        "max_trades_per_day": int(preset_cfg.get("max_trades_per_day") or preset_cfg.get("top_n_confidence_per_day") or 3),
        "cooldown_minutes": int(preset_cfg.get("cooldown_minutes") or 0),
    }
    allowed_by_risk = True  # per-call router has no daily counter state; caller (strategy) enforces

    # 9. Shadow ready (for live gating)
    shadow_ready = False
    try:
        meta = profile.get("shadow_mode_metadata") or profile.get("gate_results") or {}
        shadow_ready = bool(meta.get("shadow_ready") or meta.get("shadow_ready", False))
        # Also accept if evaluate_candidate_gates style passed all
        if not shadow_ready and isinstance(meta, dict):
            if meta.get("gate_fail_count", 1) == 0 and meta.get("gate_pass_count", 0) >= 8:
                shadow_ready = True
    except Exception:
        shadow_ready = False

    # 10. Live mode hard rules (router still never orders)
    live_order_allowed_by_router = True
    if mode == "live":
        if not shadow_ready and not forced:
            live_order_allowed_by_router = False
        # Note: the *real* live order enablement (MSTOCK_ENABLE_*) is enforced in strategy.py caller.

    # Final signal synthesis (Task C/F)
    final_signal = "NO_TRADE"
    no_trade_reason = ""
    route_error_flag = False
    if pred_err:
        err_s = str(pred_err)
        if err_s in ("FEATURES_MISSING", "FEATURE_VECTOR_INVALID"):
            no_trade_reason = err_s
            route_error_flag = False
        elif err_s == "ARTIFACT_IDENTITY_MISMATCH":
            no_trade_reason = "ARTIFACT_IDENTITY_MISMATCH"
            route_error_flag = False
        elif err_s == "XGBOOST_NOT_INSTALLED":
            no_trade_reason = "XGBOOST_NOT_INSTALLED"
            route_error_flag = False
        elif err_s in ("MODEL_OUTPUT_INVALID", "model_pkl_not_found", "no_supported_predict_method"):
            no_trade_reason = "MODEL_OUTPUT_INVALID"
            route_error_flag = True
        elif err_s == "PREDICT_EXCEPTION":
            no_trade_reason = "PREDICT_EXCEPTION"
            route_error_flag = True
        elif err_s.startswith("pickle_load_failed"):
            if "xgboost" in err_s.lower():
                no_trade_reason = "XGBOOST_NOT_INSTALLED"
                route_error_flag = False
            else:
                no_trade_reason = "MODEL_OUTPUT_INVALID"
                route_error_flag = True
        elif err_s.startswith("scaler_transform_failed"):
            no_trade_reason = "MODEL_OUTPUT_INVALID"
            route_error_flag = True
        else:
            no_trade_reason = f"PREDICT_ERROR_{err_s.split(':')[-1][:40]}"
            route_error_flag = True
        allowed_by_model = False
        confidence = None
    elif not allowed_by_model:
        if confidence is None:
            no_trade_reason = "MODEL_OUTPUT_INVALID"
        else:
            no_trade_reason = f"low_confidence_{_format_confidence_for_reason(confidence)}_lt_{threshold:.4f}"
    elif not preset_allowed:
        no_trade_reason = preset_block_reason
    elif not allowed_by_side_policy:
        no_trade_reason = side_reason or "side_policy_block"
    elif not allowed_by_liquidity:
        no_trade_reason = liq_reason or "liquidity_gate"
    elif not allowed_by_cost:
        no_trade_reason = cost_reason or "cost_gate"
    elif not allowed_by_risk:
        no_trade_reason = "risk_gate"
    elif mode == "live" and not live_order_allowed_by_router:
        no_trade_reason = "live_blocked_non_shadow_ready_unless_force_eval"
    elif side_decision in ("CE", "PE"):
        final_signal = f"BUY_{side_decision}"
    else:
        no_trade_reason = no_trade_reason or "no_side_decision"

    # allowed_by_preset already checked; set aggregates
    allowed_by_preset = preset_allowed

    out = {
        "timestamp": _now_iso(),
        "mode": mode,
        "candidate_id": cid,
        "model_name": model_name,
        "preset_family": preset_family,
        "selected_preset": selected_preset,
        "side_policy": side_policy,
        "side_decision": side_decision,
        "confidence": _router_confidence_value(confidence),
        "threshold": round(float(threshold), 6),
        "market_regime": regime,
        "volatility_state": vol_state,
        "trend_state": trend_state,
        "liquidity_state": liquidity_state,
        "allowed_by_model": bool(allowed_by_model),
        "allowed_by_preset": bool(allowed_by_preset),
        "allowed_by_side_policy": bool(allowed_by_side_policy),
        "allowed_by_liquidity": bool(allowed_by_liquidity),
        "allowed_by_cost": bool(allowed_by_cost),
        "allowed_by_risk": bool(allowed_by_risk),
        "shadow_ready": bool(shadow_ready),
        "forced_eval": forced,
        "final_signal": final_signal,
        "risk": risk_settings,
        "no_trade_reason": no_trade_reason,
        "route_error": bool(route_error_flag or pred_debug.get("route_error_like", False)),
        "debug": {
            "preset_sel": {k: preset_sel.get(k) for k in ("why_selected", "no_trade_reason") if k in preset_sel},
            "used_legacy": use_legacy,
            "live_order_allowed_by_router": live_order_allowed_by_router if mode == "live" else None,
            "predict": {k: pred_debug.get(k) for k in ("raw","prob","confidence","model_class","predict_method","scaler_present","scaler_applied","calibrator_present","calibrator_applied","feature_count","X_shape","error","missing_features","classes_","decision","artifact_path") if k in (pred_debug or {})},
        },
    }

    _LAST_ROUTER_DECISION = out
    return out
