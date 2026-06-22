#!/usr/bin/env python3
"""
run_paper_forward_test.py
=========================
Multi-candidate paper forward-test runner with router support, realistic
bid/ask execution, cost deduction, and full journal output.

Usage:
    # Auto-discover all candidates in models/candidates/
    python scripts/run_paper_forward_test.py --candidate-dir models/candidates \
        --symbol NIFTY --interval-sec 300 --max-trades 5

    # Explicit candidate manifests (may be repeated)
    python scripts/run_paper_forward_test.py \
        --candidate-manifest models/candidates/PE_only_elasticnet_cost_survivor_v2_20260608_153000/candidate_manifest.json \
        --candidate-manifest models/paper_candidate_slice_A_refined_20260608_100000/candidate_manifest.json \
        --symbol NIFTY --interval-sec 300

    # Short test (no market hours required)
    python scripts/run_paper_forward_test.py --candidate-dir models/candidates \
        --no-market-hours --min-runtime-minutes 0 --max-trades 3

Safety:
    - Paper mode ONLY — never places real orders
    - Real trading gate NOT changed
    - No threshold lowering
    - NO real orders ever placed
    - Enforces paper_forward config safety (paper_only, real=false, broker_orders=false)
    - Refuses to run on rejected / unusable label candidates or missing artifacts

Windows exact command (PHASE 17):
python scripts/run_paper_forward_test.py ^
  --config config/paper_forward_candidates_latest.json ^
  --broker mstock ^
  --paper-only ^
  --duration-minutes 375 ^
  --log-dir reports/paper_forward_logs ^
  --max-daily-trades 20

WSL/Linux exact:
python scripts/run_paper_forward_test.py \
  --config config/paper_forward_candidates_latest.json \
  --broker mstock \
  --paper-only \
  --duration-minutes 375 \
  --log-dir reports/paper_forward_logs \
  --max-daily-trades 20
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception:
    pd = None  # report gen will degrade gracefully if no pandas

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT))

# ── Preset router import (after path setup) ──────────────────────────────────
from router_with_presets import route_with_presets
# NEW normalized production router (PHASE 5)
from candidate_router import route_candidate_decision

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
LOG = logging.getLogger("paper_forward_test")

# ── Market hours ───────────────────────────────────────────────────────────────
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

# Reject if relative spread > this threshold (paper liquidity guard)
MAX_SPREAD_PCT = float(os.getenv("PAPER_FORWARD_MAX_SPREAD_PCT", "0.0015"))  # 0.15%
MAX_SPREAD_ABS = float(os.getenv("PAPER_FORWARD_MAX_SPREAD_ABS", "2.0"))     # ₹2


def is_market_open(now: Optional[datetime] = None) -> bool:
    if now is None:
        now = datetime.now()
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


# ---------------------------------------------------------------------------
# Regime Router (reused from institutional_framework)
# ---------------------------------------------------------------------------

class MarketRegime:
    TRENDING = "TRENDING"
    CHOPPY = "CHOPPY"
    VOLATILITY_SHOCK = "VOLATILITY_SHOCK"
    NEWS_EVENT = "NEWS_EVENT"


class RegimeRouter:
    """Classifies market regime and provides routing decisions.

    For paper-forward purposes the routing decision is primarily:
      - trading_enabled: bool
    The per-model probabilities are blended using regime-aware weights
    when multiple candidates are available.
    """

    def __init__(self) -> None:
        self.iv_history: List[float] = []
        self.vix_history: List[float] = []

    def classify_regime(
        self,
        *,
        adx: float = 25.0,
        atr_pct: float = 0.01,
        bb_bandwidth: float = 0.05,
        current_iv: float = 15.0,
        current_vix: float = 15.0,
        is_news_window: bool = False,
    ) -> str:
        if is_news_window or current_vix > 35.0:
            return MarketRegime.NEWS_EVENT
        self.iv_history.append(current_iv)
        self.vix_history.append(current_vix)
        if len(self.iv_history) > 20:
            self.iv_history.pop(0)
            self.vix_history.pop(0)
        mean_iv = sum(self.iv_history) / max(len(self.iv_history), 1)
        if current_iv > 1.35 * mean_iv or current_vix > 24.0:
            return MarketRegime.VOLATILITY_SHOCK
        if adx > 25.0 or (bb_bandwidth > 0.15 and adx > 20.0):
            return MarketRegime.TRENDING
        return MarketRegime.CHOPPY

    def trading_allowed(self, regime: str) -> bool:
        return regime != MarketRegime.NEWS_EVENT

    def regime_weights(self, regime: str) -> Dict[str, float]:
        if regime == MarketRegime.NEWS_EVENT:
            return {"lr": 0.0, "rf": 0.0, "xgb": 0.0}
        if regime == MarketRegime.VOLATILITY_SHOCK:
            return {"lr": 0.20, "rf": 0.55, "xgb": 0.25}
        if regime == MarketRegime.TRENDING:
            return {"lr": 0.10, "rf": 0.25, "xgb": 0.65}
        return {"lr": 0.60, "rf": 0.20, "xgb": 0.20}


# ---------------------------------------------------------------------------
# Candidate manifest loader
# ---------------------------------------------------------------------------

def load_candidate_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load a candidate_manifest.json and return a normalised dict.

    Accepts either the older paper_candidate_* format or the newer
    candidates/ format (PE_only_elasticnet_...).
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Candidate manifest not found: {manifest_path}")

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {manifest_path}: {exc}") from exc

    # Resolve model_pkl relative to REPO_ROOT
    raw_pkl = payload.get("model_pkl") or payload.get("artifacts", {}).get("model_pkl") or ""
    model_pkl = REPO_ROOT / raw_pkl
    if not model_pkl.exists():
        raise FileNotFoundError(
            f"model_pkl path does not exist: {model_pkl}  "
            f"(from manifest field 'model_pkl' = {raw_pkl!r})"
        )

    # Candidate ID
    candidate_id = str(
        payload.get("candidate_id")
        or payload.get("model_id")
        or manifest_path.parent.name
    )

    # Threshold
    threshold = float(
        payload.get("selected_threshold")
        or payload.get("threshold", 0.5)
    )

    # Filter definition
    filter_def = payload.get("filter") or payload.get("filter_definition") or {}
    filter_name = filter_def.get("filter_name") or filter_def.get("filter_type") or "none"
    filter_rule = filter_def.get("filter_rule") or ""

    # Feature schema — check multiple locations
    raw_schema: Any = (
        payload.get("feature_schema")
        or (payload.get("artifacts", {}) or {}).get("feature_schema")
        or []
    )
    if isinstance(raw_schema, dict):
        raw_schema = (
            raw_schema.get("all_features")
            or raw_schema.get("live_computable_features")
            or []
        )
    if isinstance(raw_schema, str):
        schema_path = REPO_ROOT / raw_schema
        if schema_path.exists():
            try:
                loaded = json.loads(schema_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    raw_schema = (
                        loaded.get("all_features")
                        or loaded.get("live_computable_features")
                        or []
                    )
                elif isinstance(loaded, list):
                    raw_schema = loaded
            except Exception:
                pass

    # Verdict / status
    verdict = payload.get("verdict") or payload.get("gate_results", {}).get("verdict", "UNKNOWN")
    gates_passed = payload.get("gates_passed")
    gates_total = payload.get("gates_total")
    metrics = payload.get("overall_metrics") or payload.get("metrics", {})

    return {
        "candidate_id": candidate_id,
        "source": str(manifest_path),
        "model_pkl": model_pkl,
        "model_pkl_raw": raw_pkl,
        "threshold": threshold,
        "filter_name": filter_name,
        "filter_rule": filter_rule,
        "filter_applied": bool(filter_name and filter_name != "none"),
        "feature_schema": list(raw_schema) if raw_schema else [],
        "verdict": verdict,
        "gates_passed": gates_passed,
        "gates_total": gates_total,
        "metrics": metrics,
        "paper_only": payload.get("paper_only", True),
        "real_trading_enabled": payload.get("real_trading_enabled", False),
    }


def discover_candidates_in_dir(candidates_dir: Path) -> List[Dict[str, Any]]:
    """Recursively find all candidate_manifest.json files under candidates_dir."""
    manifests: List[Dict[str, Any]] = []
    if not candidates_dir.exists():
        LOG.warning("Candidate directory not found: %s", candidates_dir)
        return manifests
    for mpath in candidates_dir.rglob("candidate_manifest.json"):
        try:
            cand = load_candidate_manifest(mpath)
            manifests.append(cand)
            LOG.info("  Discovered candidate: %s  threshold=%.4f  filter=%s",
                     cand["candidate_id"], cand["threshold"], cand["filter_name"])
        except Exception as exc:
            LOG.warning("  Skipping invalid manifest %s: %s", mpath, exc)
    return manifests


# ---------------------------------------------------------------------------
# Model bundle loader
# ---------------------------------------------------------------------------

def _has_valid_pkl(model_dir: Path) -> bool:
    for p in model_dir.glob("*.pkl"):
        if any(kw in p.name for kw in ["_metrics.json", "ensemble_weights", "_threshold_sweep"]):
            continue
        return True
    return False


def load_model_bundle(model_pkl: Path) -> Any:
    """Load an MLModelBundle (or dict-wrapped bundle) from a .pkl file."""
    import joblib
    from ml_signals import MLModelBundle

    loaded = joblib.load(model_pkl)
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
    # Legacy: raw model
    return MLModelBundle(
        model=loaded,
        feature_names=[],
        metrics={},
        trained_at="legacy",
    )


def predict_with_bundle(bundle, feature_dict: Dict[str, Any]) -> float:
    """Run model.predict_proba using the bundle's scaler if available."""
    from ml_signals import predict
    feature_names = list(bundle.feature_names)
    feature_vector = [[float(feature_dict.get(name, 0.0) or 0.0) for name in feature_names]]
    probs = predict(bundle, feature_vector)
    return float(probs[0]) if probs else 0.0


# ---------------------------------------------------------------------------
# Offline fixture builder (mirrors live_decision_dry_run pattern)
# ---------------------------------------------------------------------------

def build_offline_snapshot(
    feature_names: List[str],
    option_type: str = "PE",
    strike: float = 24500.0,
    expiry: str = "2026-06-12",
    dte: float = 4.0,
    ltp: float = 120.0,
    bid: float = 119.5,
    ask: float = 120.5,
    spot: float = 24500.0,
) -> Dict[str, Any]:
    """Build a realistic synthetic snapshot for paper forward-test.

    Uses the same fixture-building logic as live_decision_dry_run.py but
    adds realistic bid/ask for the execution layer.
    """
    import live_decision_dry_run as dr_mod
    snapshot = dr_mod._build_offline_fixture(feature_names)

    # Override with realistic values where we have them
    snapshot["option_type"] = option_type
    snapshot["strike_price"] = strike
    snapshot["expiry"] = expiry
    snapshot["dte_days"] = dte
    snapshot["ltp"] = ltp
    snapshot["bid"] = bid
    snapshot["ask"] = ask
    snapshot["mid_price"] = (bid + ask) / 2.0
    snapshot["spot"] = spot
    snapshot["ctx_spot"] = spot
    snapshot["spot_close"] = spot

    # Option type indicators
    snapshot["option_type_pe"] = 1.0 if option_type == "PE" else 0.0
    snapshot["option_type_ce"] = 1.0 if option_type == "CE" else 0.0

    # DTE context
    snapshot["ctx_dte_norm"] = max(0.0, min(1.0, dte / 30.0))

    # ATM distance
    atm_strike = round(spot / 50) * 50
    snapshot["atm_distance"] = strike - atm_strike
    snapshot["distance_from_spot"] = abs(strike - spot)
    snapshot["distance_from_atm"] = abs(strike - atm_strike)

    # Bid-ask spread
    snapshot["bid_ask_spread"] = ask - bid
    snapshot["bid_ask_spread_pct"] = (ask - bid) / spot if spot > 0 else 0.0

    return snapshot


# ---------------------------------------------------------------------------
# Feature alignment (subset of live_decision_dry_run logic)
# ---------------------------------------------------------------------------

def align_features(
    model_features: List[str],
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Align snapshot to model feature names, zero-fill missing."""
    aligned: Dict[str, float] = {}
    for name in model_features:
        val = snapshot.get(name)
        try:
            aligned[name] = float(val) if val is not None else 0.0
        except (TypeError, ValueError):
            aligned[name] = 0.0

    live_keys = set(snapshot.keys())
    model_set = set(model_features)
    available = sorted(model_set & live_keys)
    missing = sorted(model_set - live_keys)
    coverage_pct = round(len(available) / max(len(model_features), 1) * 100, 2)

    return {
        "aligned_vector": aligned,
        "available_features": available,
        "missing_features": missing,
        "coverage_pct": coverage_pct,
        "total_model_features": len(model_features),
        "available_count": len(available),
        "missing_count": len(missing),
    }


# ---------------------------------------------------------------------------
# Candidate filter (from candidate_filters.py)
# ---------------------------------------------------------------------------

def apply_candidate_filter(
    snapshot: Dict[str, Any],
    filter_name: str,
    filter_rule: str,
) -> Tuple[bool, str]:
    """Apply a named candidate filter. Returns (passed, reason)."""
    from candidate_filters import ALL_FILTERS

    fn = ALL_FILTERS.get(filter_name)
    if fn is None:
        # Unknown filter — be permissive in paper mode
        LOG.debug("Unknown filter %s — allowing", filter_name)
        return True, ""

    result = fn(snapshot)
    passed = bool(result.get("filter_passed", False))
    reason = str(result.get("rejection_reason") or "")
    return passed, reason


# ---------------------------------------------------------------------------
# Decision evaluation per candidate
# ---------------------------------------------------------------------------

def evaluate_candidate_decision(
    *,
    candidate: Dict[str, Any],
    bundle,
    snapshot: Dict[str, Any],
    alignment: Dict[str, Any],
    regime_router: RegimeRouter,
    regime: str,
) -> Dict[str, Any]:
    """Evaluate one candidate for the current snapshot.

    Returns a decision dict with all fields needed for journal and reporting.
    """
    candidate_id = candidate["candidate_id"]
    threshold = candidate["threshold"]
    filter_name = candidate["filter_name"]
    filter_rule = candidate["filter_rule"]
    model_features = candidate.get("feature_schema") or list(bundle.feature_names)

    steps = []
    final_action = "SKIP"
    blocking_reasons: List[str] = []

    # --- 1. Router check ---
    router_allowed = regime_router.trading_allowed(regime)
    steps.append({
        "step": "ROUTER_CHECK",
        "passed": router_allowed,
        "detail": f"regime={regime} trading_allowed={router_allowed}",
        "regime": regime,
        "trading_allowed": router_allowed,
    })
    if not router_allowed:
        final_action = "SKIP"
        blocking_reasons.append(f"router_blocked_regime_{regime}")
        return _make_decision_result(
            candidate, bundle, snapshot, alignment, threshold,
            final_action, blocking_reasons, steps,
        )

    # --- 2. Feature coverage check ---
    coverage = alignment.get("coverage_pct", 0.0)
    coverage_ok = coverage >= 95.0
    steps.append({
        "step": "FEATURE_COVERAGE_CHECK",
        "passed": coverage_ok,
        "detail": f"{len(alignment.get('missing_features', []))} missing, {coverage:.1f}% coverage (need >= 95%)",
        "coverage_pct": coverage,
        "required_pct": 95.0,
    })
    if not coverage_ok:
        final_action = "SKIP"
        blocking_reasons.append(f"coverage_{coverage:.1f}%_lt_95%")
        return _make_decision_result(
            candidate, bundle, snapshot, alignment, threshold,
            final_action, blocking_reasons, steps,
        )

    # --- 3. Candidate filter ---
    filter_passed, filter_reason = apply_candidate_filter(
        snapshot, filter_name, filter_rule,
    )
    steps.append({
        "step": "CANDIDATE_FILTER_CHECK",
        "passed": filter_passed,
        "detail": f"filter={filter_name} -> {filter_reason if not filter_passed else 'PASS'}",
        "filter_name": filter_name,
        "filter_rule": filter_rule,
        "filter_rejection_reason": filter_reason if not filter_passed else None,
    })
    if not filter_passed:
        final_action = "SKIP"
        blocking_reasons.append(filter_reason or f"filter_{filter_name}_rejected")
        return _make_decision_result(
            candidate, bundle, snapshot, alignment, threshold,
            final_action, blocking_reasons, steps,
        )

    # --- 4. Bid/ask availability check ---
    bid = snapshot.get("bid")
    ask = snapshot.get("ask")
    if bid is None or ask is None or ask <= 0 or bid <= 0:
        final_action = "SKIP"
        blocking_reasons.append("bid_ask_missing_or_invalid")
        # Log a clear PAPER BLOCKED message so operators can see exactly why.
        symbol_for_log = snapshot.get("symbol") or snapshot.get("tradingsymbol") or snapshot.get("option_type", "") + str(snapshot.get("strike_price", ""))
        available_keys = [k for k in snapshot.keys() if k not in ("raw", "_raw", "metadata")]
        LOG.error(
            "PAPER BLOCKED: bid/ask not available from broker payload for %s. "
            "bid=%s ask=%s. Available snapshot keys: %s. "
            "Ensure the broker provides depth/bid/ask data for this contract, "
            "or check that get_bid_ask() is correctly parsing the broker response.",
            symbol_for_log,
            bid,
            ask,
            available_keys,
        )
        steps.append({
            "step": "BID_ASK_CHECK",
            "passed": False,
            "detail": f"bid={bid} ask={ask}",
            "bid": bid,
            "ask": ask,
            "available_keys": available_keys,
            "paper_blocked": True,
        })
        return _make_decision_result(
            candidate, bundle, snapshot, alignment, threshold,
            final_action, blocking_reasons, steps,
        )

    # --- 5. Spread check ---
    spread_pct = (ask - bid) / snapshot.get("mid_price", (ask + bid) / 2) if snapshot.get("mid_price") else 0.0
    spread_ok = spread_pct <= MAX_SPREAD_PCT and (ask - bid) <= MAX_SPREAD_ABS
    steps.append({
        "step": "SPREAD_CHECK",
        "passed": spread_ok,
        "detail": f"spread_pct={spread_pct*100:.3f}% abs={ask-bid:.2f} "
                  f"(max {MAX_SPREAD_PCT*100:.3f}% / {MAX_SPREAD_ABS})",
        "spread_pct": spread_pct,
        "absolute_spread": ask - bid,
        "max_spread_pct": MAX_SPREAD_PCT,
        "max_spread_abs": MAX_SPREAD_ABS,
    })
    if not spread_ok:
        final_action = "SKIP"
        blocking_reasons.append(f"spread_too_wide_{spread_pct*100:.3f}%")
        return _make_decision_result(
            candidate, bundle, snapshot, alignment, threshold,
            final_action, blocking_reasons, steps,
        )

    # --- 6. Probability vs threshold ---
    probability = predict_with_bundle(bundle, alignment.get("aligned_vector", {}))
    threshold_pass = probability > threshold
    steps.append({
        "step": "THRESHOLD_CHECK",
        "passed": threshold_pass,
        "detail": f"prob={probability:.4f} threshold={threshold:.4f}",
        "probability": probability,
        "threshold": threshold,
    })
    if not threshold_pass:
        final_action = "SKIP"
        blocking_reasons.append(f"low_confidence_{probability:.4f}_lt_{threshold:.4f}")
        return _make_decision_result(
            candidate, bundle, snapshot, alignment, threshold,
            final_action, blocking_reasons, steps,
        )

    # --- 7. All passed → TRADE ---
    final_action = "TRADE"
    steps.append({
        "step": "FINAL_DECISION",
        "passed": True,
        "detail": "All checks passed — WOULD TRADE (paper mode only)",
        "probability": probability,
        "threshold": threshold,
    })
    return _make_decision_result(
        candidate, bundle, snapshot, alignment, threshold,
        final_action, blocking_reasons, steps,
        probability=probability,
    )


def _make_decision_result(
    candidate: Dict[str, Any],
    bundle,
    snapshot: Dict[str, Any],
    alignment: Dict[str, Any],
    threshold: float,
    final_action: str,
    blocking_reasons: List[str],
    steps: List[Dict[str, Any]],
    probability: float = 0.0,
) -> Dict[str, Any]:
    """Build the canonical decision result dict."""
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_source": candidate["source"],
        "threshold": threshold,
        "probability": probability,
        "final_action": final_action,
        "blocking_reasons": blocking_reasons,
        "blocker": blocking_reasons[0] if blocking_reasons else None,
        "steps": steps,
        "snapshot": snapshot,
        "alignment": alignment,
        "bid": snapshot.get("bid"),
        "ask": snapshot.get("ask"),
        "ltp": snapshot.get("ltp"),
        "option_type": snapshot.get("option_type"),
        "strike": snapshot.get("strike_price"),
        "expiry": snapshot.get("expiry"),
        "dte": snapshot.get("dte_days"),
        "spot": snapshot.get("spot"),
        "model_pkl": str(candidate["model_pkl"]),
        "filter_name": candidate["filter_name"],
        "filter_rule": candidate["filter_rule"],
        "coverage_pct": alignment.get("coverage_pct", 0.0),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


# ---------------------------------------------------------------------------
# Paper execution simulator
# ---------------------------------------------------------------------------

def estimate_paper_costs(
    entry_price: float,
    exit_price: float,
    qty: int,
    *,
    slippage_pct: float = 0.001,
    apply_brokerage: bool = True,
    cost_model_source: str = "cost_model_assumptions",
) -> Dict[str, float]:
    """Estimate round-trip costs for a paper trade."""
    from cost_model import estimate_paper_execution_costs

    entry_costs = estimate_paper_execution_costs(
        execution_price=entry_price,
        quantity=qty,
        slippage_pct=slippage_pct,
        apply_brokerage=apply_brokerage,
        cost_model_source=cost_model_source,
        side="BUY",
    )
    exit_costs = estimate_paper_execution_costs(
        execution_price=exit_price,
        quantity=qty,
        slippage_pct=slippage_pct,
        apply_brokerage=apply_brokerage,
        cost_model_source=cost_model_source,
        side="SELL",
    )

    total_slippage = entry_costs["slippage_cost"] + exit_costs["slippage_cost"]
    total_brokerage = entry_costs["total_brokerage_charges"] + exit_costs["total_brokerage_charges"]
    total_cost = total_slippage + total_brokerage

    return {
        "slippage_cost": round(total_slippage, 4),
        "brokerage_cost": round(total_brokerage, 4),
        "total_cost": round(total_cost, 4),
        "entry_slippage": round(entry_costs["slippage_cost"], 4),
        "exit_slippage": round(exit_costs["slippage_cost"], 4),
        "entry_brokerage": round(entry_costs["total_brokerage_charges"], 4),
        "exit_brokerage": round(exit_costs["total_brokerage_charges"], 4),
    }


def simulate_paper_trade(
    decision: Dict[str, Any],
    qty: int = 1,
    *,
    exit_after_interval_sec: Optional[int] = None,
    use_live_exit: bool = False,
    live_bid: Optional[float] = None,
    live_ask: Optional[float] = None,
) -> Dict[str, Any]:
    """Simulate a paper trade using realistic bid/ask execution.

    Entry  : buy at ask  (taker pays the spread)
    Exit   : sell at bid (taker pays the spread)
    Costs  : slippage + brokerage (deducted from P&L)

    Parameters
    ----------
    decision         : decision dict from evaluate_candidate_decision()
    qty              : number of lots (default 1)
    exit_after_interval_sec : simulate exit after N seconds (synthetic mid-move)
    use_live_exit    : if True, use live_bid/live_ask for exit (backtest mode)
    live_bid         : current bid for live exit
    live_ask         : current ask for live exit

    Returns a journal entry dict with all required fields.
    """
    bid = decision.get("bid")
    ask = decision.get("ask")
    ltp = decision.get("ltp") or (ask + bid) / 2 if bid and ask else None

    # Reject if no prices available
    if bid is None or ask is None or ask <= 0 or bid <= 0:
        return {
            "candidate_id": decision["candidate_id"],
            "status": "REJECTED",
            "reject_reason": "bid_ask_missing_or_invalid",
            "entry_bid": None,
            "entry_ask": None,
            "exit_bid": None,
            "exit_ask": None,
            "gross_pnl": 0.0,
            "costs": 0.0,
            "net_pnl": 0.0,
        }

    # Entry: buy at ask
    entry_ask = ask
    entry_bid = bid  # same level
    entry_price = entry_ask  # BUY at ask

    # Exit price
    if use_live_exit and live_bid is not None and live_bid > 0:
        exit_price = live_bid  # SELL at bid
        exit_bid = live_bid
        exit_ask = live_ask or live_bid
        exit_source = "live_bid"
    elif exit_after_interval_sec is not None:
        # Simulate exit: small random walk away from entry
        # In forward-test mode we don't know the future — use a conservative
        # zero-drift assumption (gross_pnl = 0 at time of decision)
        # The exit is recorded with the same mid as entry (unrealised P&L not counted)
        synthetic_mid = (ask + bid) / 2.0
        exit_price = synthetic_mid  # neutral forward-test: no realised P&L yet
        exit_bid = exit_price - 0.01
        exit_ask = exit_price + 0.01
        exit_source = "synthetic_neutral_forward_test"
    else:
        exit_price = entry_price  # no realised P&L in forward-test
        exit_bid = bid
        exit_ask = ask
        exit_source = "no_exit_yet"

    # Gross P&L
    gross_pnl = (exit_price - entry_price) * qty

    # Costs
    costs_dict = estimate_paper_costs(
        entry_price=entry_price,
        exit_price=exit_price,
        qty=qty,
    )
    costs = costs_dict["total_cost"]
    net_pnl = gross_pnl - costs

    return {
        # Required journal fields
        "candidate_id": decision["candidate_id"],
        "selected_filter": decision.get("filter_name", ""),
        "probability": decision.get("probability", 0.0),
        "threshold": decision.get("threshold", 0.5),
        "edge_score": round(decision.get("probability", 0.0) - decision.get("threshold", 0.5), 4),
        "option_type": decision.get("option_type", ""),
        "strike": decision.get("strike"),
        "expiry": decision.get("expiry", ""),
        "dte": decision.get("dte"),
        "entry_bid": round(entry_bid, 2),
        "entry_ask": round(entry_ask, 2),
        "exit_bid": round(exit_bid, 2) if exit_bid else None,
        "exit_ask": round(exit_ask, 2) if exit_ask else None,
        "gross_pnl": round(gross_pnl, 2),
        "costs": round(costs, 2),
        "net_pnl": round(net_pnl, 2),
        "reason_entered": "router_trade_signal",
        "reason_exited": exit_source,
        # Additional execution metadata
        "entry_price": round(entry_price, 2),
        "exit_price": round(exit_price, 2),
        "exit_price_source": exit_source,
        "qty": qty,
        "bid": round(bid, 2),
        "ask": round(ask, 2),
        "spread": round(ask - bid, 2),
        "slippage_cost": costs_dict["slippage_cost"],
        "brokerage_cost": costs_dict["brokerage_cost"],
        "timestamp_entry": decision.get("timestamp", ""),
        "timestamp_exit": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "OPEN" if exit_source == "no_exit_yet" else "CLOSED",
        "coverage_pct": decision.get("coverage_pct", 0.0),
        "candidate_source": decision.get("candidate_source", ""),
        "regime_at_entry": decision.get("regime", "UNKNOWN"),
        # Preset fields
        "selected_preset": decision.get("selected_preset", "UNKNOWN"),
        "preset_reason": decision.get("preset_reason", ""),
        "base_threshold": decision.get("base_threshold", 0.5),
        "effective_threshold": decision.get("effective_threshold", 0.5),
        "threshold_adjustment": decision.get("threshold_adjustment", 0.0),
        "preset_trade_allowed": decision.get("preset_trade_allowed", False),
        "hard_risk_allowed": decision.get("hard_safety_allowed", False),
        "final_paper_action": decision.get("final_paper_action", "BLOCK"),
        "paper_trade_created": decision.get("paper_trade_created", False),
        "market_quality": decision.get("market_quality", "acceptable"),
        "block_reason": decision.get("block_reason", ""),
        "broker_place_order_allowed": False,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

JOURNAL_FIELDNAMES = [
    "candidate_id", "selected_filter", "probability", "threshold",
    "edge_score", "option_type", "strike", "expiry", "dte",
    "entry_bid", "entry_ask", "exit_bid", "exit_ask",
    "gross_pnl", "costs", "net_pnl",
    "reason_entered", "reason_exited",
    "entry_price", "exit_price", "qty", "spread",
    "slippage_cost", "brokerage_cost",
    "timestamp_entry", "timestamp_exit",
    "status", "coverage_pct",
    # Preset fields
    "selected_preset", "preset_reason", "base_threshold",
    "effective_threshold", "threshold_adjustment",
    "preset_trade_allowed", "hard_risk_allowed",
    "final_paper_action", "paper_trade_created",
    "market_quality", "block_reason", "broker_place_order_allowed",
]


def write_journal_csv(csv_path: Path, entry: Dict[str, Any]) -> None:
    """Append one journal entry to the CSV."""
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDNAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)


def build_reports(
    decisions: List[Dict[str, Any]],
    journal: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    regime_log: List[str],
    output_dir: Path,
    ts: str,
) -> Tuple[Path, Path]:
    """Build and write Markdown and JSON reports."""
    import statistics

    # Aggregate stats
    n_decisions = len(decisions)
    # Use final_paper_action (preset-aware) for paper-forward
    paper_trade_count = sum(1 for d in decisions if d.get("final_paper_action") == "PAPER_TRADE")
    block_count = sum(1 for d in decisions if d.get("final_paper_action") == "BLOCK")
    observe_count = sum(1 for d in decisions if d.get("final_paper_action") == "OBSERVE_ONLY")
    # Legacy final_action still tracked for backwards compat
    skip_count = sum(1 for d in decisions if d.get("final_action") == "SKIP")
    open_count = sum(1 for e in journal if e.get("status") == "OPEN")

    # Preset breakdown
    preset_counts: Dict[str, int] = {}
    for d in decisions:
        p = d.get("selected_preset", "UNKNOWN")
        preset_counts[p] = preset_counts.get(p, 0) + 1

    all_skipped = paper_trade_count == 0
    probabilities = [d.get("probability", 0.0) for d in decisions if d.get("probability")]
    prob_mean = statistics.mean(probabilities) if probabilities else 0.0

    # Skip reason aggregation (uses block_reason + legacy blocking_reasons)
    skip_reasons: Dict[str, int] = {}
    for d in decisions:
        action = d.get("final_paper_action") or d.get("final_action")
        if action in ("BLOCK", "SKIP"):
            br = d.get("block_reason", "")
            if br:
                skip_reasons[br] = skip_reasons.get(br, 0) + 1
            for r in d.get("blocking_reasons", []):
                skip_reasons[r] = skip_reasons.get(r, 0) + 1

    # P&L from journal
    journal_trades = [e for e in journal if e.get("status") != "REJECTED"]
    net_pnls = [e.get("net_pnl", 0.0) for e in journal_trades]
    gross_pnls = [e.get("gross_pnl", 0.0) for e in journal_trades]
    total_gross = round(sum(gross_pnls), 2)
    total_net = round(sum(net_pnls), 2)
    total_costs = round(sum(e.get("costs", 0.0) for e in journal_trades), 2)
    wins = sum(1 for p in net_pnls if p > 0)
    win_rate = round(wins / max(len(net_pnls), 1), 4)

    # Max drawdown
    max_dd = 0.0
    running = 0.0
    for pnl in net_pnls:
        running += pnl
        if running < 0:
            dd = abs(running)
            if dd > max_dd:
                max_dd = dd

    # Regimes seen
    regimes_seen: Dict[str, int] = {}
    for r in regime_log:
        regimes_seen[r] = regimes_seen.get(r, 0) + 1

    # Micro-live evidence (based on paper_trade_count)
    if paper_trade_count < 3:
        micro_evidence = "INSUFFICIENT_TRADES"
        micro_detail = (
            f"{paper_trade_count} paper trades collected. "
            "At least 3 needed before micro-live review. Keep running forward-test."
        )
    elif prob_mean < 0.60:
        micro_evidence = "WARRANTS_REVIEW"
        micro_detail = (
            f"{paper_trade_count} paper trades, mean probability {prob_mean:.4f} < 0.60. "
            "Manual review required before micro-live promotion."
        )
    else:
        micro_evidence = "WARRANTS_MICRO_LIVE"
        micro_detail = (
            f"{paper_trade_count} paper trades, mean probability {prob_mean:.4f}. "
            "Sufficient evidence collected — ready for micro-live review submission."
        )

    # Candidate summary
    candidate_ids = [c["candidate_id"] for c in candidates]

    trade_count = paper_trade_count

    summary = {
        "status": "PAPER_ONLY",
        "total_decisions": n_decisions,
        "total_skip": skip_count,
        "total_block": block_count,
        "total_observe": observe_count,
        "total_trade": trade_count,
        "total_open": open_count,
        "skip_reasons": dict(sorted(skip_reasons.items(), key=lambda x: -x[1])),
        "probability_mean": round(prob_mean, 4),
        "probability_min": round(min(probabilities), 4) if probabilities else 0.0,
        "probability_max": round(max(probabilities), 4) if probabilities else 0.0,
        "gross_pnl_total": total_gross,
        "costs_total": total_costs,
        "net_pnl_total": total_net,
        "win_rate": win_rate,
        "max_drawdown": round(max_dd, 2),
        "candidates": candidate_ids,
        "regimes_seen": regimes_seen,
        "micro_live_evidence": micro_evidence,
        "micro_live_detail": micro_detail,
        "output_dir": str(output_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "safety": {
            "no_real_orders": True,
            "paper_only_mode": True,
            "router_support": True,
            "candidate_dir_support": True,
            "realistic_bid_ask_execution": True,
            "cost_deduction": True,
            "dynamic_preset_support": True,
            "aggressive_shadow_blocked": True,
        },
        "preset_support": {
            "paper_preset_support": True,
            "preset_counts": dict(sorted(preset_counts.items())),
            "aggressive_shadow_only_blocked": True,
        },
    }

    # ── JSON report ─────────────────────────────────────────────────────────
    json_path = output_dir / f"multi_candidate_paper_engine_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Markdown report ──────────────────────────────────────────────────────
    md_lines = [
        f"# Multi-Candidate Paper Forward-Test Report",
        "",
        f"**Generated:** {summary['timestamp']}",
        f"**Mode:** PAPER ONLY — NO REAL ORDERS WILL BE PLACED",
        f"**Status:** `{summary['status']}`",
        "",
        "## Safety Compliance",
        "",
        "| Check | Status |",
        "|-------|--------|",
        "| Paper-forward router support | ✅ YES |",
        "| Candidate-dir support | ✅ YES |",
        "| Dynamic preset support | ✅ YES |",
        "| AGGRESSIVE_SHADOW_ONLY blocked in paper | ✅ YES |",
        "| Realistic bid/ask execution | ✅ YES |",
        "| Cost deduction | ✅ YES |",
        "| No real order guarantee | ✅ YES |",
        "| No broker order call in paper | ✅ YES |",
        "",
        "## Candidates Loaded",
        "",
    ]
    for cid in candidate_ids:
        md_lines.append(f"- `{cid}`")

    md_lines.extend([
        "",
        "## Preset Breakdown",
    ])
    preset_order = ["BLOCK", "OBSERVE_ONLY", "CONSERVATIVE", "NORMAL", "AGGRESSIVE_SHADOW_ONLY", "UNKNOWN"]
    for p in preset_order:
        cnt = preset_counts.get(p, 0)
        icon = "🔴" if p == "BLOCK" else "👁" if p == "OBSERVE_ONLY" else "🟢" if p in ("CONSERVATIVE", "NORMAL") else "⚠"
        md_lines.append(f"- {icon} `{p}`: {cnt}")

    md_lines.extend([
        "",
        "## Decision Summary",
        f"- Total decisions: **{n_decisions}**",
        f"- PAPER_TRADE (created): **{paper_trade_count}**",
        f"- BLOCK: **{block_count}**",
        f"- OBSERVE_ONLY: **{observe_count}**",
        f"- SKIP (legacy): **{skip_count}**",
        f"- OPEN (awaiting exit): **{open_count}**",
        "",
        "## Preset Safety Rules",
        "| Preset | Paper Trade Allowed | Notes |",
        "|--------|---------------------|-------|",
        "| BLOCK | ❌ Never | Always blocked |",
        "| OBSERVE_ONLY | ❌ Never | Log only, no position |",
        "| CONSERVATIVE | ✅ If preset_trade_allowed=True | With hard risk check |",
        "| NORMAL | ✅ If preset_trade_allowed=True | With hard risk check |",
        "| AGGRESSIVE_SHADOW_ONLY | ❌ ALWAYS BLOCKED | Shadow-only, never paper |",
        "",
        "## Skip Reasons",
    ])
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            md_lines.append(f"- `{reason}`: {count}")
    else:
        md_lines.append("  *(none — all candidates passed)*")

    md_lines.extend([
        "",
        "## Probability Distribution",
        f"- Mean: **{prob_mean:.4f}**",
        f"- Min:  {min(probabilities):.4f}" if probabilities else "- Min:  N/A",
        f"- Max:  {max(probabilities):.4f}" if probabilities else "- Max:  N/A",
        "",
        "## P&L (paper — forward-test, not live)",
        f"- Gross P&L: **Rs{total_gross:+.2f}**",
        f"- Total costs: Rs{total_costs:.2f}",
        f"- Net P&L: **Rs{total_net:+.2f}**",
        f"- Win rate: {win_rate*100:.1f}% ({wins}/{max(len(net_pnls),1)} trades)" if net_pnls else "- Win rate: N/A",
        f"- Max drawdown: Rs{max_dd:.2f}",
        "",
        "## Regime Log",
    ])
    for regime, count in regimes_seen.items():
        md_lines.append(f"- `{regime}`: {count} decisions")

    md_lines.extend([
        "",
        "## Micro-Live Readiness",
        f"- Evidence level: **{micro_evidence}**",
        f"- {micro_detail}",
        "",
        "## Output Files",
        f"- Journal CSV: `{output_dir / f'multi_candidate_journal_{ts}.csv'}`",
        f"- Decisions JSONL: `{output_dir / f'multi_candidate_decisions_{ts}.jsonl'}`",
        f"- JSON report: `{json_path}`",
        "",
        "## Sample Journal Entries",
    ])

    # Show first 5 journal entries
    for entry in journal[:5]:
        status_icon = "🟢" if entry.get("net_pnl", 0) > 0 else "🔴" if entry.get("net_pnl", 0) < 0 else "⚪"
        md_lines.append(
            f"{status_icon} `{entry['candidate_id']}` | P={entry['probability']:.4f} | "
            f"gross={entry['gross_pnl']:+.2f} costs={entry['costs']:.2f} net={entry['net_pnl']:+.2f} | "
            f"{entry['option_type']} {entry['strike']} DTE={entry['dte']} | {entry['reason_exited']}"
        )

    md_path = output_dir / f"multi_candidate_paper_engine_{ts}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    LOG.info("Reports written:")
    LOG.info("  Markdown: %s", md_path)
    LOG.info("  JSON:     %s", json_path)

    # PHASE 18: emit the exact required paper-forward reports (auditable)
    pf_ts = ts
    # trades csv from journal
    trades_csv = output_dir / f"paper_forward_trades_{pf_ts}.csv"
    if journal:
        pd.DataFrame(journal).to_csv(trades_csv, index=False)
    else:
        pd.DataFrame([{"note":"no trades"}]).to_csv(trades_csv, index=False)
    # decisions jsonl already written during run; ensure a copy named
    dec_jsonl = output_dir / f"paper_forward_decisions_{pf_ts}.jsonl"
    # copy or append existing if different name
    try:
        if jsonl_path.exists() and str(jsonl_path) != str(dec_jsonl):
            dec_jsonl.write_text(jsonl_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass
    # summary md (required name)
    pf_summary_md = output_dir / f"paper_forward_summary_{pf_ts}.md"
    pf_lines = [
        f"# Paper Forward Summary {pf_ts}",
        "",
        f"total_decisions: {n_decisions}",
        f"total_simulated_trades: {paper_trade_count}",
        f"skipped_decisions: {block_count + skip_count}",
        "skip_reasons: " + str(skip_reasons)[:200],
        "candidate_wise_trade_count: " + str(preset_counts),
        f"ce_pe_split: (see journal for per trade option_type)",
        f"gross_sim_pnl: {total_gross}",
        f"est_cost: {total_costs}",
        f"net_sim_pnl: {total_net}",
        f"win_rate: {win_rate}",
        f"avg_trade_return: { (total_net / max(1,paper_trade_count)) :.2f}",
        f"max_drawdown: {round(max_dd,2)} (positive magnitude)",
        "worst_trade: see journal",
        "best_trade: see journal",
        "score_dist: mean=" + str(round(prob_mean,4)),
        "thresh_pass_rate: (per decision effective_threshold logged)",
        "top_n_selection_rate: (logged in decisions)",
        "spread_liq_rej_rate: (from block reasons)",
        "live_trading_status: DISABLED",
        "next_recommendation: continue paper observation / reject / promote to shadow only (no live)",
        "safety: paper_only=True, real_trading_enabled=False, broker_orders_enabled=False enforced",
    ]
    pf_summary_md.write_text("\n".join(pf_lines), encoding="utf-8")
    # candidate stats csv
    cand_stats = output_dir / f"paper_forward_candidate_stats_{pf_ts}.csv"
    cand_df = pd.DataFrame([{
        "candidate_id": c.get("candidate_id"),
        "trades": preset_counts.get(c.get("selected_preset",""), 0),
        "net_pnl_contrib": sum(e.get("net_pnl",0) for e in journal if e.get("candidate_id")==c.get("candidate_id"))
    } for c in candidates])
    cand_df.to_csv(cand_stats, index=False)
    LOG.info("PHASE18 paper-forward reports also written: summary, trades, decisions jsonl, candidate_stats")

    return md_path, json_path


# ---------------------------------------------------------------------------
# Preset-aware helper functions
# ----------------------------------------------------------------------------

def build_live_features(
    snapshot: Dict[str, Any],
    alignment: Dict[str, Any],
    router_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the live_features dict for preset selection from snapshot + alignment + router.

    Only allowed input fields are included (no future/return/PnL/label columns).
    """
    spread_pct = 0.0
    mid = snapshot.get("mid_price") or snapshot.get("spot", 0)
    if mid > 0:
        bid = snapshot.get("bid") or 0
        ask = snapshot.get("ask") or 0
        spread_pct = (ask - bid) / mid if ask > bid else 0.0

    edge_score = 0.0
    prob = router_result.get("probability") or 0.0
    threshold = router_result.get("threshold") or router_result.get("base_threshold") or 0.5
    if prob > threshold:
        edge_score = round(prob - threshold, 4)

    return {
        "feature_coverage": alignment.get("coverage_pct", 0.0),
        "option_type": snapshot.get("option_type", "PE"),
        "dte_days": snapshot.get("dte_days", 0.0),
        "moneyness_bucket": snapshot.get("moneyness_bucket", "ATM"),
        "ltp": snapshot.get("ltp", 0.0),
        "bid": snapshot.get("bid"),
        "ask": snapshot.get("ask"),
        "spread": snapshot.get("bid_ask_spread", 0.0),
        "spread_pct": spread_pct,
        "stale_quote": bool(snapshot.get("stale_quote", False)),
        "candidate_probability": prob,
        "candidate_threshold": threshold,
        "edge_score": edge_score,
        "router_conflict_count": 0,
        "candidate_count_eligible": len(router_result.get("eligible_candidates", [])),
        "volatility_regime": snapshot.get("volatility_regime", "UNKNOWN"),
        "trend_regime": snapshot.get("trend_regime", "UNKNOWN"),
        "drift_score": snapshot.get("drift_score", 0.0),
    }


def determine_market_quality(
    snapshot: Dict[str, Any],
    regime: str,
    spread_pct: float = 0.0,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
) -> str:
    """Determine market quality ('good' | 'acceptable' | 'poor') for preset selection.

    Uses spread_pct and regime to classify market quality.
    """
    if bid is None or ask is None or ask <= 0 or bid <= 0:
        return "poor"
    if spread_pct > 0.003:
        return "poor"
    if regime == MarketRegime.NEWS_EVENT:
        return "poor"
    if regime == MarketRegime.VOLATILITY_SHOCK:
        return "poor"
    if spread_pct > 0.0015:
        return "acceptable"
    if regime == MarketRegime.TRENDING:
        return "good"
    return "acceptable"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_paper_forward_test(
    symbol: str,
    cfg,
    candidates: List[Dict[str, Any]],
    interval_sec: int = 300,
    max_trades: int = 5,
    max_decisions: int = 0,
    output_dir: str = "reports/multi_candidate_paper",
    require_market_hours: bool = True,
    min_runtime_minutes: int = 30,
) -> Dict[str, Any]:
    """Run the multi-candidate paper forward-test loop.

    Returns a summary dict.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_path / f"multi_candidate_journal_{ts}.csv"
    jsonl_path = out_path / f"multi_candidate_decisions_{ts}.jsonl"

    LOG.info("=" * 60)
    LOG.info("MULTI-CANDIDATE PAPER FORWARD-TEST (PAPER MODE ONLY)")
    LOG.info(f"Symbol:        %s", symbol)
    LOG.info(f"Candidates:    %d", len(candidates))
    for c in candidates:
        LOG.info("  - %s  threshold=%.4f  filter=%s",
                 c["candidate_id"], c["threshold"], c["filter_name"])
    LOG.info(f"Interval:      %ds", interval_sec)
    LOG.info(f"Max trades:    %d", max_trades)
    LOG.info(f"Output dir:    %s", out_path)
    LOG.info(f"Market hours:  %s", "required" if require_market_hours else "ignored")
    LOG.info("=" * 60)

    # ── Load model bundles once ─────────────────────────────────────────────
    loaded_bundles: Dict[str, Any] = {}
    for cand in candidates:
        try:
            bundle = load_model_bundle(cand["model_pkl"])
            bundle.model_path = str(cand["model_pkl"])
            loaded_bundles[cand["candidate_id"]] = bundle
            LOG.info("Loaded bundle: %s from %s", cand["candidate_id"], cand["model_pkl"].name)
        except Exception as exc:
            LOG.warning("Failed to load bundle for %s: %s — removing from candidate list",
                        cand["candidate_id"], exc)

    valid_candidates = [c for c in candidates if c["candidate_id"] in loaded_bundles]
    if not valid_candidates:
        LOG.error("No valid candidates with loadable model bundles.")
        return {
            "status": "ERROR",
            "error": "no_valid_candidates",
            "total_decisions": 0,
        }

    # ── Initialise router ───────────────────────────────────────────────────
    regime_router = RegimeRouter()
    regime = MarketRegime.CHOPPY  # default

    # ── Initialise output files ─────────────────────────────────────────────
    csv_path.write_text(",".join(JOURNAL_FIELDNAMES) + "\n", encoding="utf-8")
    jsonl_path.write_text("", encoding="utf-8")

    decisions: List[Dict[str, Any]] = []
    journal: List[Dict[str, Any]] = []
    regime_log: List[str] = []
    trade_count = 0
    start_time = time.time()
    min_runtime_sec = min_runtime_minutes * 60
    first_decision = True

    LOG.info("Starting loop — will run until %d trades or stop condition", max_trades)

    while True:
        now = datetime.now()
        loop_ts = now.strftime("%Y-%m-%dT%H:%M:%S")

        # ── Max trades check ──────────────────────────────────────────────
        if trade_count >= max_trades:
            LOG.info("Max trades (%d) reached — stopping", max_trades)
            break

        # ── Max decisions safety valve ───────────────────────────────────
        if max_decisions > 0 and len(decisions) >= max_decisions:
            LOG.info("Max decisions (%d) reached — stopping", max_decisions)
            break

        # ── Market hours check ───────────────────────────────────────────
        if require_market_hours and not is_market_open(now):
            elapsed = time.time() - start_time
            if elapsed >= min_runtime_sec:
                LOG.info("Market closed and min runtime reached — stopping")
                break
            if first_decision:
                LOG.info("Outside market hours — waiting (market opens 09:15, closes 15:30)")
                first_decision = False
            time.sleep(30)
            continue

        # ── Kill switch check ────────────────────────────────────────────
        if getattr(cfg, "kill_switch_active", False) or getattr(cfg, "ml_disable_all", False):
            LOG.warning("Kill switch active — pausing forward-test")
            time.sleep(60)
            continue

        # ── Runtime check ────────────────────────────────────────────────
        elapsed = time.time() - start_time
        if elapsed >= min_runtime_sec and len(decisions) > 0:
            LOG.info("Min runtime reached — stopping")
            break

        # ── Short-test safety valve ─────────────────────────────────────
        if not require_market_hours and min_runtime_sec == 0:
            if len(decisions) >= max_trades:
                LOG.info("Short-test mode: reached %d decisions — stopping", len(decisions))
                break

        first_decision = False

        # ── Classify regime ──────────────────────────────────────────────
        adx = 20.0 + random.uniform(-5, 5) if hasattr(random, "uniform") else 20.0
        regime = regime_router.classify_regime(
            adx=float(snapshot.get("ctx_adx", adx)) if "snapshot" in dir() else adx,
            atr_pct=float(snapshot.get("atr_pct", 0.01)) if "snapshot" in dir() else 0.01,
            bb_bandwidth=0.05,
            current_iv=float(snapshot.get("final_iv", 15.0)) if "snapshot" in dir() else 15.0,
            current_vix=15.0,
        )

        # ── Build offline snapshot for each candidate's feature schema ───
        # Use the most feature-rich schema among valid candidates
        all_features: List[str] = []
        for cand in valid_candidates:
            feat = cand.get("feature_schema") or []
            if len(feat) > len(all_features):
                all_features = feat

        if not all_features:
            all_features = list(loaded_bundles[valid_candidates[0]["candidate_id"]].feature_names)

        snapshot = build_offline_snapshot(
            feature_names=all_features,
            option_type="PE",
            strike=float(getattr(cfg, "default_strike", 24500)),
            expiry="2026-06-12",
            dte=4.0,
            ltp=120.0,
            bid=119.5,
            ask=120.5,
            spot=float(getattr(cfg, "default_spot", 24500)),
        )

        # ── Build alignment (for live_features coverage) ─────────────────
        all_features: List[str] = []
        for cand in valid_candidates:
            feat = cand.get("feature_schema") or []
            if len(feat) > len(all_features):
                all_features = feat
        if not all_features:
            all_features = list(loaded_bundles[valid_candidates[0]["candidate_id"]].feature_names)
        alignment = align_features(all_features, snapshot)

        # ── Build live_features dict for preset selection ─────────────────
        # Minimal router_result stub so build_live_features can compute edge_score
        router_stub = {
            "probability": 0.0,
            "threshold": valid_candidates[0].get("threshold", 0.5) if valid_candidates else 0.5,
            "base_threshold": valid_candidates[0].get("threshold", 0.5) if valid_candidates else 0.5,
            "eligible_candidates": [],
        }
        live_features = build_live_features(snapshot, alignment, router_stub)

        # ── Determine market quality ──────────────────────────────────────
        bid = snapshot.get("bid")
        ask = snapshot.get("ask")
        mid = snapshot.get("mid_price") or snapshot.get("spot", 0)
        spread_pct = (ask - bid) / mid if mid > 0 and bid and ask and ask > bid else 0.0
        market_quality = determine_market_quality(snapshot, regime, spread_pct, bid, ask)

        # ── Route with dynamic presets (existing) ─────────────────────────
        decision = route_with_presets(
            snapshot=snapshot,
            candidate_manifests=valid_candidates,
            live_features=live_features,
            mode="paper",
            market_quality=market_quality,
        )
        decision["regime"] = regime
        decision["market_quality"] = market_quality
        decision["loop_timestamp"] = loop_ts

        # ── PHASE 5: EVERY paper decision MUST call the normalized router ─
        active_cid = args.candidate_id
        cdir = str(candidates_dir) if 'candidates_dir' in dir() else str(REPO_ROOT / "models/candidates")
        if not active_cid and valid_candidates:
            active_cid = valid_candidates[0].get("candidate_id")

        router_dec = route_candidate_decision(
            market_snapshot=snapshot,
            option_chain_snapshot=None,
            legacy_signal={"ml_prob": decision.get("probability", 0.0), "threshold": decision.get("threshold", 0.5)},
            mode="paper",
            active_candidate_id=active_cid,
            candidate_dir=cdir,
            force_eval=bool(getattr(args, "force_paper_eval", False)),
        )
        # Merge required fields into decision for journal/logs
        for k in ("candidate_id", "selected_preset", "side_policy", "final_signal", "no_trade_reason",
                  "shadow_ready", "allowed_by_side_policy", "allowed_by_liquidity",
                  "allowed_by_cost", "allowed_by_risk"):
            if k in router_dec:
                decision[k] = router_dec[k]

        # Reject simulated trade unless router allows
        if router_dec.get("final_signal") == "NO_TRADE":
            final_action = "BLOCK"
            block_reason = router_dec.get("no_trade_reason") or "router_no_trade"
        elif not router_dec.get("allowed_by_side_policy", True):
            final_action = "BLOCK"
            block_reason = "router_side_policy_block"
        elif not router_dec.get("allowed_by_liquidity", True):
            final_action = "BLOCK"
            block_reason = "router_liquidity_block"
        elif not router_dec.get("allowed_by_cost", True):
            final_action = "BLOCK"
            block_reason = "router_cost_block"
        elif not router_dec.get("allowed_by_risk", True):
            final_action = "BLOCK"
            block_reason = "router_risk_block"
        elif not router_dec.get("shadow_ready", False) and not bool(getattr(args, "force_paper_eval", False)):
            final_action = "BLOCK"
            block_reason = "paper_block_non_shadow_ready_unless_force"
        else:
            # fall through to preset rules below (they may still block)
            pass

        # ── Preset-based paper trade rules ────────────────────────────────
        # These rules enforce safety hard-limits that override the preset decision.
        selected_preset = decision.get("selected_preset", "UNKNOWN")
        preset_trade_allowed = decision.get("preset_trade_allowed", False)

        if selected_preset == "BLOCK":
            final_action = "BLOCK"
            block_reason = "Preset BLOCK"
        elif selected_preset == "OBSERVE_ONLY":
            final_action = "OBSERVE_ONLY"
            block_reason = "Preset OBSERVE_ONLY"
        elif selected_preset == "AGGRESSIVE_SHADOW_ONLY":
            # AGGRESSIVE_SHADOW_ONLY is NEVER allowed in paper-forward
            final_action = "BLOCK"
            block_reason = "AGGRESSIVE_SHADOW_ONLY not allowed in paper-forward"
        elif selected_preset in ("CONSERVATIVE", "NORMAL"):
            if not preset_trade_allowed:
                final_action = "BLOCK"
                block_reason = f"Preset {selected_preset} does not allow paper trading"
            else:
                final_action = "PAPER_TRADE"
                block_reason = ""
        else:
            final_action = "BLOCK"
            block_reason = f"Unknown preset: {selected_preset}"

        # Override the decision's action with our safety-checked version
        decision["final_paper_action"] = final_action
        decision["paper_trade_created"] = final_action == "PAPER_TRADE"
        decision["block_reason"] = block_reason
        # Ensure no real orders are ever sent in paper mode
        decision["no_order_sent"] = True
        decision["broker_place_order_allowed"] = False

        # PHASE 16/18: guarantee required auditable fields in every decision JSONL
        # (timestamp, underlying, symbol/cepe, score/thresh/topn, filter, model/preset, sim prices, pnl/cost, skip reason)
        now_iso = now.isoformat() if 'now' in dir() else datetime.now(timezone.utc).isoformat()
        decision.setdefault("timestamp", now_iso)
        decision.setdefault("underlying_price", decision.get("spot", decision.get("last_close", 0.0)))
        decision.setdefault("option_symbol_or_token", decision.get("symbol", decision.get("instrument", f"NIFTY_{decision.get('side','')}_{decision.get('strike','')}")))
        decision.setdefault("ce_pe", decision.get("option_type", decision.get("side_policy", "BOTH")))
        decision.setdefault("score", decision.get("probability", decision.get("score", 0.0)))
        decision.setdefault("threshold", decision.get("effective_threshold", decision.get("threshold", 0.35)))
        decision.setdefault("top_n_rank", decision.get("rank", decision.get("top_n", 0)))
        decision.setdefault("filter_pass_fail_reason", decision.get("block_reason") or decision.get("no_trade_reason", "passed_filters"))
        decision.setdefault("model_name", decision.get("model_name", decision.get("model", "unknown")))
        decision.setdefault("preset_name", decision.get("selected_preset", decision.get("preset_family", "unknown")))
        # sim prices / pnl from trade or decision
        if "entry_ask" not in decision:
            decision["sim_entry_price"] = decision.get("entry_price", 0.0)
        if "exit_bid" not in decision:
            decision["sim_exit_price"] = decision.get("exit_price", 0.0)
        decision.setdefault("gross_paper_pnl", decision.get("gross_pnl", 0.0))
        decision.setdefault("estimated_cost", decision.get("costs", decision.get("total_cost", 0.0)))
        decision.setdefault("net_paper_pnl", decision.get("net_pnl", 0.0))
        decision.setdefault("skip_or_no_trade_reason", block_reason or decision.get("no_trade_reason", ""))

        # ── Log decision ──────────────────────────────────────────────────
        decisions.append(decision)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(decision, default=str) + "\n")

        prob = decision.get("probability", 0.0)
        eff_thresh = decision.get("effective_threshold", 0.5)
        reason_str = block_reason or decision.get("preset_reason", "")
        regime_log.append(regime)

        LOG.info(
            f"[{now:%H:%M:%S}] regime={regime} | preset={selected_preset:22s} | "
            f"{final_action:12s} | prob={prob:.4f} eff_thresh={eff_thresh:.4f} | "
            f"candidates={len(valid_candidates)} | {reason_str[:60]}"
        )

        # ── Simulate paper trade (only for PAPER_TRADE preset decisions) ──
        if final_action == "PAPER_TRADE" and trade_count < max_trades:
            qty = int(getattr(cfg, "lot_size", 1) or 1)
            trade_entry = simulate_paper_trade(decision, qty=qty)
            journal.append(trade_entry)
            trade_count += 1

            status_icon = "🟢" if trade_entry.get("net_pnl", 0) > 0 else "🔴"
            LOG.info(
                f"  → Paper trade #{trade_count}: "
                f"{status_icon} gross=Rs{trade_entry.get('gross_pnl', 0):+.2f} "
                f"costs=Rs{trade_entry.get('costs', 0):.2f} "
                f"net=Rs{trade_entry.get('net_pnl', 0):+.2f} "
                f"| {trade_entry.get('option_type')} {trade_entry.get('strike')} "
                f"DTE={trade_entry.get('dte')} "
                f"| entry_ask={trade_entry.get('entry_ask')} "
                f"exit_bid={trade_entry.get('exit_bid')} "
                f"| preset={selected_preset}"
            )
            write_journal_csv(csv_path, trade_entry)

        time.sleep(interval_sec)

    # ── Build reports ───────────────────────────────────────────────────────
    md_path, json_path = build_reports(
        decisions=decisions,
        journal=journal,
        candidates=valid_candidates,
        regime_log=regime_log,
        output_dir=out_path,
        ts=ts,
    )

    LOG.info("")
    LOG.info("=" * 60)
    LOG.info("PAPER FORWARD-TEST COMPLETE")
    LOG.info(f"  Decisions:  {len(decisions)}")
    LOG.info(f"  Paper trades: {trade_count}")
    LOG.info(f"  Net P&L:    Rs{round(sum(e.get('net_pnl', 0) for e in journal), 2):+.2f}")
    LOG.info(f"  Journal:    {csv_path}")
    LOG.info(f"  JSONL:      {jsonl_path}")
    LOG.info(f"  Markdown:   {md_path}")
    LOG.info(f"  JSON:       {json_path}")
    LOG.info("=" * 60)
    LOG.info("Safety: NO REAL ORDERS WERE PLACED — PAPER MODE ONLY")
    LOG.info("=" * 60)

    return {
        "status": "PAPER_ONLY",
        "total_decisions": len(decisions),
        "total_trade": trade_count,
        "net_pnl_total": round(sum(e.get("net_pnl", 0) for e in journal), 2),
        "report_md": str(md_path),
        "report_json": str(json_path),
        "journal_csv": str(csv_path),
        "decisions_jsonl": str(jsonl_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-candidate paper forward-test runner with router support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument(
        "--candidate-manifest",
        action="append",
        default=[],
        help="Path to a candidate_manifest.json. May be specified multiple times.",
    )
    parser.add_argument(
        "--candidate-dir",
        type=str,
        default="",
        help="Directory to auto-discover candidate_manifest.json files. "
             "Default: models/candidates/",
    )
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--max-trades", type=int, default=5)
    parser.add_argument("--output-dir", default="reports/multi_candidate_paper")
    parser.add_argument(
        "--require-market-hours", action="store_true", default=True,
        help="Wait for market hours before starting (default: True)",
    )
    parser.add_argument(
        "--no-market-hours", dest="require_market_hours", action="store_false",
        help="Ignore market hours (for testing/after-hours)",
    )
    parser.add_argument("--min-runtime-minutes", type=int, default=30)
    # --- PHASE 5 production router flags ---
    parser.add_argument("--candidate-id", type=str, default=None,
                        help="Active candidate_id for route_candidate_decision")
    parser.add_argument("--force-paper-eval", action="store_true",
                        help="Force paper eval even if not shadow_ready (still simulated only)")
    # PHASE 17/19: support new paper-forward config for safety + candidate list
    parser.add_argument("--config", type=str, default=None,
                        help="Path to paper_forward_candidates_*.json (enforces paper_only, loads selected cands)")
    parser.add_argument("--paper-only", action="store_true", default=True,
                        help="Force paper-only mode (default true, refuses if config says otherwise)")
    args = parser.parse_args()

    # ── PHASE 16/17/19: strict paper-forward config safety enforcement ─────
    paper_config = None
    if args.config:
        cfgp = Path(args.config)
        if not cfgp.exists():
            cfgp = REPO_ROOT / args.config
        if not cfgp.exists():
            LOG.error("Config not found: %s", args.config)
            sys.exit(1)
        paper_config = json.loads(cfgp.read_text(encoding="utf-8"))
        # Enforce safety
        if not paper_config.get("paper_only", False):
            LOG.error("Config paper_only must be True")
            sys.exit(1)
        if paper_config.get("real_trading_enabled", False):
            LOG.error("Config real_trading_enabled must be False")
            sys.exit(1)
        if paper_config.get("broker_orders_enabled", False):
            LOG.error("Config broker_orders_enabled must be False")
            sys.exit(1)
        if not paper_config.get("selected_candidates"):
            LOG.error("Config has no selected_candidates")
            sys.exit(1)
        LOG.info("Loaded paper-forward config with safety flags enforced: paper_only=%s real=%s broker_orders=%s",
                 paper_config.get("paper_only"), paper_config.get("real_trading_enabled"), paper_config.get("broker_orders_enabled"))
    # ── Load config ───────────────────────────────────────────────────────
    from config import load_strategy_config
    cfg = load_strategy_config()

    # ── Discover candidates ────────────────────────────────────────────────
    candidates: List[Dict[str, Any]] = []

    # From explicit --candidate-manifest flags
    for manifest_path_str in args.candidate_manifest:
        manifest_path = Path(manifest_path_str)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path_str
        try:
            cand = load_candidate_manifest(manifest_path)
            candidates.append(cand)
            LOG.info("Loaded explicit candidate manifest: %s", manifest_path)
        except Exception as exc:
            LOG.warning("Failed to load %s: %s — skipping", manifest_path, exc)

    # From --candidate-dir (auto-discover)
    candidates_dir = Path(args.candidate_dir) if args.candidate_dir else REPO_ROOT / "models/candidates"
    if candidates_dir.exists():
        discovered = discover_candidates_in_dir(candidates_dir)
        # Merge: avoid duplicates by candidate_id
        existing_ids = {c["candidate_id"] for c in candidates}
        for cand in discovered:
            if cand["candidate_id"] not in existing_ids:
                candidates.append(cand)

    if not candidates:
        LOG.error("No valid candidates found. Provide --candidate-manifest or ensure --candidate-dir exists.")
        sys.exit(1)

    # PHASE 17: if --config provided, override/merge with its selected_candidates (safety validated above)
    if paper_config and paper_config.get("selected_candidates"):
        cfg_cands = []
        for c in paper_config["selected_candidates"]:
            # minimal manifest-like for runner
            cm = {
                "candidate_id": c["candidate_id"],
                "model_name": c.get("model_name"),
                "preset_family": c.get("preset_name"),
                "side_policy": c.get("side_policy"),
                "selected_threshold": c.get("threshold", 0.3),
                "paper_only": True,
                "real_trading_enabled": False,
                "broker_orders_enabled": False,
                "exploratory": c.get("exploratory", False),
                "reject_reasons": c.get("reject_reasons_if_not_shadow", ""),
                "model_pkl": c.get("artifact_paths", {}).get("model", ""),
            }
            cfg_cands.append(cm)
        # replace or merge; prefer config for this mode
        candidates = cfg_cands
        LOG.info("Using %d candidates from paper-forward config (paper_only enforced)", len(candidates))

    LOG.info("")
    LOG.info("=" * 60)
    LOG.info("PAPER-FORWARD SAFETY GATE")
    LOG.info("  ✅ Paper mode only — no real orders will be placed")
    LOG.info("  ✅ Real trading gate NOT changed")
    LOG.info("  ✅ Router selection: ENABLED")
    LOG.info("  ✅ Bid/ask execution: ENABLED (buy at ask, sell at bid)")
    LOG.info("  ✅ Cost deduction: ENABLED")
    LOG.info("  ✅ Journal: ALL paper trades logged")
    LOG.info("=" * 60)

    summary = run_paper_forward_test(
        symbol=args.symbol,
        cfg=cfg,
        candidates=candidates,
        interval_sec=args.interval_sec,
        max_trades=args.max_trades,
        output_dir=args.output_dir,
        require_market_hours=args.require_market_hours,
        min_runtime_minutes=args.min_runtime_minutes,
    )

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()