#!/usr/bin/env python3
"""
src/candidate_lifecycle.py
==========================
Explicit candidate lifecycle management for safe promotion:
DISABLED -> BACKTEST_PASSED -> PAPER_FORWARD -> PAPER_PASSED -> SHADOW ->
SHADOW_PASSED -> LIVE_DRY_RUN -> LIVE_1_LOT -> LIVE_SCALED

This module provides:
- Status constants
- Candidate dataclass with all required fields
- Load / save helpers for config/shadow_candidates.json and config/live_candidate_whitelist.json
- Promotion gate evaluators (pure; no side effects except file I/O in scripts)
- Helpers to compute "would promote" decisions from paper/shadow log summaries

Safety:
- Never enables live trading.
- All defaults are conservative (enabled=False, live_whitelisted=False, max_qty=0).
- Status transitions are append-only in the JSON records (last_promoted_at updated on change).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# =============================================================================
# Status constants (exact strings required by the task)
# =============================================================================

STATUS_DISABLED = "DISABLED"
STATUS_BACKTEST_PASSED = "BACKTEST_PASSED"
STATUS_PAPER_FORWARD = "PAPER_FORWARD"
STATUS_PAPER_PASSED = "PAPER_PASSED"
STATUS_SHADOW = "SHADOW"
STATUS_SHADOW_PASSED = "SHADOW_PASSED"
STATUS_LIVE_DRY_RUN = "LIVE_DRY_RUN"
STATUS_LIVE_1_LOT = "LIVE_1_LOT"
STATUS_LIVE_SCALED = "LIVE_SCALED"

ALL_STATUSES = [
    STATUS_DISABLED,
    STATUS_BACKTEST_PASSED,
    STATUS_PAPER_FORWARD,
    STATUS_PAPER_PASSED,
    STATUS_SHADOW,
    STATUS_SHADOW_PASSED,
    STATUS_LIVE_DRY_RUN,
    STATUS_LIVE_1_LOT,
    STATUS_LIVE_SCALED,
]

# Allowed mode sets (what a candidate at this status may participate in)
ALLOWED_MODES_BY_STATUS = {
    STATUS_DISABLED: [],
    STATUS_BACKTEST_PASSED: ["backtest"],
    STATUS_PAPER_FORWARD: ["paper_forward"],
    STATUS_PAPER_PASSED: ["paper_forward"],
    STATUS_SHADOW: ["shadow", "paper_forward"],
    STATUS_SHADOW_PASSED: ["shadow", "paper_forward"],
    STATUS_LIVE_DRY_RUN: ["shadow", "live_dry_run"],
    STATUS_LIVE_1_LOT: ["shadow", "live_dry_run", "live_1_lot"],
    STATUS_LIVE_SCALED: ["shadow", "live_dry_run", "live_1_lot", "live_scaled"],
}

# =============================================================================
# Candidate record (matches task spec exactly)
# =============================================================================

@dataclass
class Candidate:
    candidate_id: str
    model_path: str = ""
    artifact_path: str = ""
    status: str = STATUS_DISABLED
    allowed_modes: List[str] = field(default_factory=list)
    enabled: bool = False
    live_whitelisted: bool = False
    max_qty_lots: int = 0
    max_trades_per_day: int = 0
    max_daily_loss: float = 0.0
    allowed_option_types: List[str] = field(default_factory=lambda: ["CE", "PE"])
    promotion_notes: str = ""
    last_promoted_at: Optional[str] = None

    # Extra runtime/audit fields (populated by promotion scripts)
    paper_trade_count: int = 0
    paper_days_tested: int = 0
    paper_cost_adjusted_pnl: float = 0.0
    paper_profit_factor: float = 0.0
    paper_win_rate: float = 0.0
    paper_max_drawdown: float = 0.0
    shadow_trade_count: int = 0
    shadow_days: int = 0
    shadow_cost_adjusted_pnl: float = 0.0
    dryrun_payload_count: int = 0
    dryrun_days: int = 0

    # Blockers recorded at last evaluation
    last_block_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # ensure lists are lists
        if not isinstance(d.get("allowed_modes"), list):
            d["allowed_modes"] = list(d.get("allowed_modes") or [])
        if not isinstance(d.get("allowed_option_types"), list):
            d["allowed_option_types"] = list(d.get("allowed_option_types") or ["CE", "PE"])
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Candidate":
        allowed_modes = d.get("allowed_modes") or []
        if isinstance(allowed_modes, str):
            allowed_modes = [allowed_modes]
        allowed_option_types = d.get("allowed_option_types") or ["CE", "PE"]
        if isinstance(allowed_option_types, str):
            allowed_option_types = [allowed_option_types]
        return Candidate(
            candidate_id=str(d.get("candidate_id", "")),
            model_path=str(d.get("model_path", d.get("model/artifact path", ""))),
            artifact_path=str(d.get("artifact_path", d.get("artifact_dir", ""))),
            status=str(d.get("status", STATUS_DISABLED)),
            allowed_modes=list(allowed_modes),
            enabled=bool(d.get("enabled", False)),
            live_whitelisted=bool(d.get("live_whitelisted", False)),
            max_qty_lots=int(d.get("max_qty_lots", d.get("max_qty_lots", 0) or 0)),
            max_trades_per_day=int(d.get("max_trades_per_day", 0) or 0),
            max_daily_loss=float(d.get("max_daily_loss", 0.0) or 0.0),
            allowed_option_types=list(allowed_option_types),
            promotion_notes=str(d.get("promotion_notes", "")),
            last_promoted_at=d.get("last_promoted_at"),
            paper_trade_count=int(d.get("paper_trade_count", 0) or 0),
            paper_days_tested=int(d.get("paper_days_tested", d.get("days_tested", 0)) or 0),
            paper_cost_adjusted_pnl=float(d.get("paper_cost_adjusted_pnl", 0.0) or 0.0),
            paper_profit_factor=float(d.get("paper_profit_factor", 0.0) or 0.0),
            paper_win_rate=float(d.get("paper_win_rate", 0.0) or 0.0),
            paper_max_drawdown=float(d.get("paper_max_drawdown", 0.0) or 0.0),
            shadow_trade_count=int(d.get("shadow_trade_count", 0) or 0),
            shadow_days=int(d.get("shadow_days", 0) or 0),
            shadow_cost_adjusted_pnl=float(d.get("shadow_cost_adjusted_pnl", 0.0) or 0.0),
            dryrun_payload_count=int(d.get("dryrun_payload_count", 0) or 0),
            dryrun_days=int(d.get("dryrun_days", 0) or 0),
            last_block_reason=str(d.get("last_block_reason", d.get("blocker_reason", ""))),
        )


# =============================================================================
# Config file paths (relative to repo root)
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
LOGS_DIR = REPO_ROOT / "logs"
REPORTS_DIR = REPO_ROOT / "reports"

SHADOW_CANDIDATES_PATH = CONFIG_DIR / "shadow_candidates.json"
LIVE_WHITELIST_PATH = CONFIG_DIR / "live_candidate_whitelist.json"
PAPER_FORWARD_CANDIDATES_PATH = CONFIG_DIR / "paper_forward_candidates.json"


# =============================================================================
# Load / Save
# =============================================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_shadow_candidates() -> List[Candidate]:
    p = SHADOW_CANDIDATES_PATH
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        items = data.get("candidates", data if isinstance(data, list) else [])
        return [Candidate.from_dict(c) for c in items if isinstance(c, dict)]
    except Exception:
        return []


def save_shadow_candidates(cands: List[Candidate], *, meta: Optional[Dict[str, Any]] = None) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "updated_at": _now_iso(),
        "mode": "shadow_promotion",
        "safety_note": "Shadow candidates only. No live orders unless explicitly whitelisted and all gates pass.",
        "candidates": [c.to_dict() for c in cands],
    }
    if meta:
        payload.update(meta)
    SHADOW_CANDIDATES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_live_whitelist() -> List[Candidate]:
    p = LIVE_WHITELIST_PATH
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        items = data.get("candidates", data if isinstance(data, list) else [])
        return [Candidate.from_dict(c) for c in items if isinstance(c, dict)]
    except Exception:
        return []


def save_live_whitelist(cands: List[Candidate], *, meta: Optional[Dict[str, Any]] = None) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "updated_at": _now_iso(),
        "mode": "live_dryrun_or_1lot_whitelist",
        "safety_note": "Only whitelisted candidates with status LIVE_1_LOT or LIVE_SCALED may attempt real orders, and only when ORDER_PLACEMENT_ENABLED=true and LIVE_ORDER_DRY_RUN=false.",
        "candidates": [c.to_dict() for c in cands],
    }
    if meta:
        payload.update(meta)
    LIVE_WHITELIST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_candidate_by_id(cands: List[Candidate], cid: str) -> Optional[Candidate]:
    for c in cands:
        if c.candidate_id == cid:
            return c
    return None


# =============================================================================
# Promotion gate helpers (pure logic)
# =============================================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _coerce_optional_bool(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return value


def evaluate_paper_to_shadow(
    paper_stats: Dict[str, Any],
    *,
    min_trades: int = 20,
    min_days: int = 3,
    max_drawdown_limit: float = 5000.0,
    require_cost_adjusted_pnl_ge: float = -1000.0,  # allow small loss but not catastrophic
) -> Tuple[bool, str]:
    """
    Strict promotion rule from PAPER_FORWARD / PAPER_PASSED -> SHADOW.
    Returns (promote: bool, reason: str)
    """
    trade_count = _safe_int(paper_stats.get("trade_count") or paper_stats.get("total_trades"))
    days_tested = _safe_int(paper_stats.get("days_tested") or paper_stats.get("days"))
    stale = _safe_int(paper_stats.get("stale_data_trade_count"))
    dup = _safe_int(paper_stats.get("duplicate_trade_count"))
    bad_expiry = _safe_int(paper_stats.get("invalid_expiry_trade_count"))
    missing_sl = _safe_int(paper_stats.get("missing_sl_count"))
    missing_exit = _safe_int(paper_stats.get("missing_exit_count"))
    after_cutoff = _safe_int(paper_stats.get("after_cutoff_trade_count"))
    cost_pnl = _safe_float(paper_stats.get("cost_adjusted_pnl"))
    max_dd = _safe_float(paper_stats.get("max_drawdown"))

    if trade_count < min_trades:
        return False, f"trade_count {trade_count} < min {min_trades}"
    if days_tested < min_days:
        return False, f"days_tested {days_tested} < min {min_days}"
    if stale > 0:
        return False, f"stale_data_trade_count={stale} (must be 0)"
    if dup > 0:
        return False, f"duplicate_trade_count={dup} (must be 0)"
    if bad_expiry > 0:
        return False, f"invalid_expiry_trade_count={bad_expiry} (must be 0)"
    if missing_sl > 0:
        return False, f"missing_sl_count={missing_sl} (must be 0)"
    if missing_exit > 0:
        return False, f"missing_exit_count={missing_exit} (must be 0)"
    if after_cutoff > 0:
        return False, f"after_cutoff_trade_count={after_cutoff} (must be 0)"
    if cost_pnl < require_cost_adjusted_pnl_ge:
        return False, f"cost_adjusted_pnl {cost_pnl:.2f} < acceptable {require_cost_adjusted_pnl_ge:.2f}"
    if max_dd > max_drawdown_limit:
        return False, f"max_drawdown {max_dd:.2f} > limit {max_drawdown_limit:.2f}"

    return True, "clean paper-forward metrics meet shadow promotion gates"


def evaluate_shadow_to_live_dryrun(
    shadow_stats: Dict[str, Any],
    *,
    min_shadow_days: int = 2,
    min_shadow_trades: int = 10,
) -> Tuple[bool, str]:
    """
    Strict rule: promote at most ONE safest candidate to LIVE_DRY_RUN.
    """
    days = _safe_int(shadow_stats.get("shadow_days"))
    trades = _safe_int(shadow_stats.get("shadow_trade_count") or shadow_stats.get("trade_count"))
    stale = _safe_int(shadow_stats.get("stale_data_trade_count"))
    bad_expiry = _safe_int(shadow_stats.get("invalid_expiry_trade_count"))
    dup = _safe_int(shadow_stats.get("duplicate_trade_count"))
    missing_sl = _safe_int(shadow_stats.get("missing_sl_count"))
    missing_exit = _safe_int(shadow_stats.get("missing_exit_count"))
    risk_bypass = _safe_int(shadow_stats.get("risk_bypass_count") or shadow_stats.get("no_risk_bypass"))
    router_silence = _safe_int(shadow_stats.get("router_silence_count"))
    payload_err = _safe_int(shadow_stats.get("order_payload_error_count"))
    emergency = _safe_int(shadow_stats.get("emergency_stop_events"))
    cost_pnl = _safe_float(shadow_stats.get("cost_adjusted_pnl") or shadow_stats.get("shadow_cost_adjusted_pnl"))
    max_dd = _safe_float(shadow_stats.get("max_drawdown"))

    if days < min_shadow_days:
        return False, f"shadow_days {days} < min {min_shadow_days}"
    if trades < min_shadow_trades:
        return False, f"shadow_trade_count {trades} < min {min_shadow_trades}"
    if stale > 0:
        return False, "stale data trades present"
    if bad_expiry > 0:
        return False, "invalid expiry trades present"
    if dup > 0:
        return False, "duplicate entries present"
    if missing_sl > 0:
        return False, "missing SL events"
    if missing_exit > 0:
        return False, "missing exit events"
    if risk_bypass > 0:
        return False, "risk bypass events"
    if router_silence > 0:
        return False, "router silence periods"
    if payload_err > 0:
        return False, "order payload construction errors"
    if emergency > 0:
        return False, "emergency stop events observed"
    if cost_pnl < -500.0:  # conservative
        return False, f"cost-adjusted result unacceptable ({cost_pnl:.2f})"
    if max_dd > 3000.0:
        return False, f"drawdown {max_dd:.2f} exceeds limit"

    return True, "shadow metrics clean; eligible for live dry-run consideration"


def select_best_for_live_dryrun(candidates: List[Candidate]) -> Optional[Candidate]:
    """
    From the set of SHADOW_PASSED candidates, pick the single best/safest.
    Criteria (simple, conservative):
      - highest shadow_cost_adjusted_pnl (or paper if shadow not present)
      - lowest max_drawdown
      - prefer higher trade_count for statistical significance
    Only one may be selected.
    """
    eligible = [c for c in candidates if c.status in (STATUS_SHADOW, STATUS_SHADOW_PASSED) and c.enabled]
    if not eligible:
        return None

    def score(c: Candidate) -> Tuple[float, float, int]:
        # (pnl desc, -dd asc, trades desc)
        pnl = max(c.shadow_cost_adjusted_pnl, c.paper_cost_adjusted_pnl)
        dd = c.paper_max_drawdown if c.shadow_days == 0 else 0.0  # prefer those with shadow data
        return (-pnl, c.paper_max_drawdown, -c.shadow_trade_count or -c.paper_trade_count)

    eligible.sort(key=score)
    return eligible[0]


def promote_candidate(
    cand: Candidate,
    new_status: str,
    *,
    notes: str = "",
    set_live_whitelist: bool = False,
    max_qty: int = 1,
    max_trades: int = 2,
    max_loss: float = 1000.0,
) -> None:
    """Mutate candidate in place for a promotion step."""
    if new_status not in ALL_STATUSES:
        raise ValueError(f"Unknown status: {new_status}")
    cand.status = new_status
    cand.allowed_modes = list(ALLOWED_MODES_BY_STATUS.get(new_status, []))
    cand.enabled = new_status != STATUS_DISABLED
    cand.promotion_notes = notes
    cand.last_promoted_at = _now_iso()
    if set_live_whitelist:
        cand.live_whitelisted = True
        cand.max_qty_lots = max(1, max_qty)
        cand.max_trades_per_day = max(1, max_trades)
        cand.max_daily_loss = max_loss
    else:
        # When demoting or keeping in shadow, keep live_whitelisted false unless explicitly set before
        if new_status not in (STATUS_LIVE_DRY_RUN, STATUS_LIVE_1_LOT, STATUS_LIVE_SCALED):
            cand.live_whitelisted = False


# =============================================================================
# Env helpers for the new gates (LIVE_MODE, LIVE_ORDER_DRY_RUN, ORDER_PLACEMENT_ENABLED)
# =============================================================================

def get_live_mode_env() -> bool:
    return os.getenv("LIVE_MODE", "").strip().lower() in {"1", "true", "yes"}


def get_live_order_dry_run_env() -> bool:
    return os.getenv("LIVE_ORDER_DRY_RUN", "").strip().lower() in {"1", "true", "yes"}


def get_order_placement_enabled_env() -> bool:
    return os.getenv("ORDER_PLACEMENT_ENABLED", "").strip().lower() in {"1", "true", "yes"}


def get_kill_switch_active() -> bool:
    raw = os.getenv("SCALPER_KILL_SWITCH", "").strip().lower()
    return raw in {"1", "true", "yes"}


def live_dry_run_requested() -> bool:
    """True when user wants to exercise the full payload path without sending orders."""
    return get_live_mode_env() and get_live_order_dry_run_env() and not get_order_placement_enabled_env()


def real_live_orders_fully_enabled() -> bool:
    """The only combination that may lead to real 1-lot orders (still requires candidate whitelist + all other gates)."""
    return (
        get_live_mode_env()
        and not get_live_order_dry_run_env()
        and get_order_placement_enabled_env()
        and not get_kill_switch_active()
    )


# =============================================================================
# Strict computed lifecycle evaluator
# =============================================================================

STAGE_DISABLED = "DISABLED"
STAGE_INVALID_CONFIG = "INVALID_CONFIG"
STAGE_ARTIFACT_MISSING = "ARTIFACT_MISSING"
STAGE_PAPER_FORWARD_ELIGIBLE = "PAPER_FORWARD_ELIGIBLE"
STAGE_PAPER_FORWARD_RUNNING = "PAPER_FORWARD_RUNNING"
STAGE_PAPER_FORWARD_PASSED = "PAPER_FORWARD_PASSED"
STAGE_SHADOW_ELIGIBLE = "SHADOW_ELIGIBLE"
STAGE_SHADOW_RUNNING = "SHADOW_RUNNING"
STAGE_SHADOW_PASSED = "SHADOW_PASSED"
STAGE_LIVE_DRY_RUN_ELIGIBLE = "LIVE_DRY_RUN_ELIGIBLE"
STAGE_LIVE_DRY_RUN_RUNNING = "LIVE_DRY_RUN_RUNNING"
STAGE_LIVE_DRY_RUN_PASSED = "LIVE_DRY_RUN_PASSED"
STAGE_LIVE_1_LOT_ELIGIBLE = "LIVE_1_LOT_ELIGIBLE"
STAGE_LIVE_1_LOT_DEPLOYED = "LIVE_1_LOT_DEPLOYED"

LIFECYCLE_STAGES = [
    STAGE_DISABLED,
    STAGE_INVALID_CONFIG,
    STAGE_ARTIFACT_MISSING,
    STAGE_PAPER_FORWARD_ELIGIBLE,
    STAGE_PAPER_FORWARD_RUNNING,
    STAGE_PAPER_FORWARD_PASSED,
    STAGE_SHADOW_ELIGIBLE,
    STAGE_SHADOW_RUNNING,
    STAGE_SHADOW_PASSED,
    STAGE_LIVE_DRY_RUN_ELIGIBLE,
    STAGE_LIVE_DRY_RUN_RUNNING,
    STAGE_LIVE_DRY_RUN_PASSED,
    STAGE_LIVE_1_LOT_ELIGIBLE,
    STAGE_LIVE_1_LOT_DEPLOYED,
]

VISIBLE_BY_DEFAULT_STAGES = {
    STAGE_PAPER_FORWARD_ELIGIBLE,
    STAGE_PAPER_FORWARD_RUNNING,
    STAGE_PAPER_FORWARD_PASSED,
    STAGE_SHADOW_ELIGIBLE,
    STAGE_SHADOW_RUNNING,
    STAGE_SHADOW_PASSED,
    STAGE_LIVE_DRY_RUN_ELIGIBLE,
    STAGE_LIVE_DRY_RUN_RUNNING,
    STAGE_LIVE_DRY_RUN_PASSED,
    STAGE_LIVE_1_LOT_ELIGIBLE,
    STAGE_LIVE_1_LOT_DEPLOYED,
}

LIVE_ONE_LOT_QTY = 65


@dataclass
class CandidateLifecycleStatus:
    candidate_id: str
    family: str = ""
    side: str = ""
    preset_family: str = ""
    current_stage: str = STAGE_INVALID_CONFIG
    next_stage: str = ""
    can_advance: bool = False
    can_be_live_deployed: bool = False
    block_reason: str = ""
    missing_requirements: List[str] = field(default_factory=list)
    passed_requirements: List[str] = field(default_factory=list)
    risk_status: str = "UNKNOWN"
    artifact_status: str = "UNKNOWN"
    data_status: str = "UNKNOWN"
    promotion_actions_available: List[str] = field(default_factory=list)
    last_updated: str = field(default_factory=_now_iso)
    confidence_health: str = "UNKNOWN"
    trades: int = 0
    days: int = 0
    archived: bool = False
    possible_live_path: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _candidate_id(row: Dict[str, Any]) -> str:
    return str(row.get("candidate_id") or row.get("id") or row.get("name") or "").strip()


def _candidate_artifact_path(row: Dict[str, Any]) -> str:
    return str(row.get("artifact_dir") or row.get("artifact_path") or row.get("artifact") or "").strip()


def _candidate_model_path(row: Dict[str, Any], artifact_dir: Path | None = None) -> str:
    raw = str(row.get("model_path") or row.get("model_file") or "").strip()
    if raw:
        return raw
    if artifact_dir is not None:
        return str(artifact_dir / "model.pkl")
    return ""


def _logical_candidate_key(row: Dict[str, Any]) -> str:
    cid = _candidate_id(row)
    threshold = row.get("threshold", row.get("selected_threshold", row.get("entry_threshold", "")))
    return "|".join(
        str(x or "").strip().lower()
        for x in (
            cid,
            row.get("model_name"),
            row.get("preset_family"),
            row.get("side_policy") or row.get("side") or row.get("family"),
            threshold,
        )
    )


def _load_json_file(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _load_artifact_metadata(artifact_dir: Path) -> Tuple[Dict[str, Any], List[str], str]:
    candidates = [
        "candidate_profile.json",
        "metadata.json",
        "model_metadata.json",
        "manifest.json",
        "deployment_manifest.json",
    ]
    meta: Dict[str, Any] = {}
    source = ""
    for name in candidates:
        payload = _load_json_file(artifact_dir / name)
        if payload:
            meta.update(payload)
            source = name
            break
    feature_order = (
        meta.get("feature_order")
        or meta.get("required_features")
        or meta.get("features")
        or meta.get("feature_names")
        or meta.get("live_computable_features")
        or []
    )
    if not feature_order:
        for schema_name in ("feature_schema.json", "features.json", "feature_order.json"):
            schema = _load_json_file(artifact_dir / schema_name)
            feature_order = (
                schema.get("feature_order")
                or schema.get("required_features")
                or schema.get("features")
                or schema.get("feature_names")
                or schema.get("live_computable_features")
                or []
            )
            if feature_order:
                if not source:
                    source = schema_name
                break
    if isinstance(feature_order, dict):
        feature_order = list(feature_order.keys())
    if isinstance(feature_order, str):
        feature_order = [x.strip() for x in feature_order.split(",") if x.strip()]
    return meta, list(feature_order or []), source


def _candidate_identity_matches(row: Dict[str, Any], meta: Dict[str, Any], artifact_dir: Path) -> Tuple[bool, str]:
    cid = _candidate_id(row)
    artifact_id = str(meta.get("candidate_id") or meta.get("artifact_id") or meta.get("id") or "").strip()
    if artifact_id and artifact_id != cid:
        return False, "ARTIFACT_IDENTITY_MISMATCH"
    if cid and artifact_dir.name and artifact_dir.name != cid and not str(row.get("allow_shared_artifact", "")).lower() in {"1", "true", "yes"}:
        # Some older artifacts omit metadata; directory name is still the safest identity check.
        return False, "ARTIFACT_IDENTITY_MISMATCH"
    return True, "identity matched"


def _dummy_prediction_check(model_path: Path, feature_order: List[str]) -> Tuple[bool, str]:
    if not feature_order:
        return False, "FEATURE_ORDER_EMPTY"
    try:
        import joblib  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return True, "prediction smoke skipped: optional deps unavailable"
    try:
        loaded = joblib.load(model_path)
    except Exception:
        return False, "MODEL_LOAD_FAILED"
    scaler = None
    model = loaded
    if isinstance(loaded, dict):
        model = loaded.get("model") or loaded.get("estimator") or loaded.get("classifier")
        scaler = loaded.get("scaler") or loaded.get("preprocessor")
        expected_n = loaded.get("n_features")
        try:
            if expected_n is not None and int(expected_n) != len(feature_order):
                return False, "FEATURE_VECTOR_INVALID"
        except Exception:
            return False, "FEATURE_VECTOR_INVALID"
    if model is None:
        return False, "MODEL_LOAD_FAILED"
    try:
        x = np.zeros((1, len(feature_order)), dtype=float)
        if scaler is not None:
            try:
                x = scaler.transform(x)
            except Exception:
                return False, "SCALER_FAILED"
        if hasattr(model, "predict_proba"):
            pred = model.predict_proba(x)
            try:
                conf = float(pred[0][-1])
            except Exception:
                return False, "PREDICT_PROBA_FAILED"
        elif hasattr(model, "predict"):
            pred = model.predict(x)
            try:
                conf = float(pred[0])
            except Exception:
                return False, "PREDICT_PROBA_FAILED"
        else:
            return False, "PREDICT_PROBA_FAILED"
    except Exception:
        return False, "PREDICT_PROBA_FAILED"
    if conf != conf:
        return False, "FEATURE_VECTOR_INVALID"
    if conf == 0.0:
        return False, "CONFIDENCE_STUCK_ZERO"
    return True, "prediction smoke passed"


def _runtime_value(runtime_stats: Dict[str, Any], candidate_id: str, *keys: str, default: Any = 0) -> Any:
    per_candidate = runtime_stats.get(candidate_id) if isinstance(runtime_stats.get(candidate_id), dict) else {}
    for source in (per_candidate, runtime_stats):
        for key in keys:
            if isinstance(source, dict) and key in source:
                return source.get(key)
    return default


def evaluate_candidate_lifecycle(
    candidate: Dict[str, Any] | Candidate,
    artifacts: Dict[str, Any] | None = None,
    runtime_stats: Dict[str, Any] | None = None,
    config: Dict[str, Any] | None = None,
) -> CandidateLifecycleStatus:
    row = candidate.to_dict() if isinstance(candidate, Candidate) else dict(candidate or {})
    artifacts = artifacts or {}
    runtime_stats = runtime_stats or {}
    config = config or {}
    cid = _candidate_id(row)
    status = CandidateLifecycleStatus(
        candidate_id=cid,
        family=str(row.get("family") or row.get("model_name") or ""),
        side=str(row.get("side") or row.get("side_policy") or ""),
        preset_family=str(row.get("preset_family") or row.get("family") or ""),
    )
    paper_observation_only = (
        bool(row.get("paper_forward_only"))
        or str(row.get("classification") or "").strip().lower() == "paper_forward_only"
        or "paper observation only" in str(row.get("notes") or "").strip().lower()
    )
    explicit_live_path = any(
        bool(row.get(key))
        for key in (
            "allow_promotion",
            "promotion_allowed",
            "paper_forward_passed",
            "shadow_ready",
            "shadow_passed",
            "live_dry_run_passed",
            "dryrun_passed",
            "live_1lot_deployed",
        )
    )
    if paper_observation_only and not explicit_live_path:
        status.possible_live_path = False

    def fail(stage: str, reason: str) -> CandidateLifecycleStatus:
        status.current_stage = stage
        status.block_reason = reason
        if reason not in status.missing_requirements:
            status.missing_requirements.append(reason)
        status.can_advance = False
        status.can_be_live_deployed = False
        return status

    if not cid:
        return fail(STAGE_INVALID_CONFIG, "CANDIDATE_ID_MISSING")
    enabled = row.get("enabled", True)
    if not isinstance(enabled, bool):
        return fail(STAGE_INVALID_CONFIG, "ENABLED_FLAG_INVALID")
    if enabled is False:
        return fail(STAGE_DISABLED, str(row.get("disabled_reason") or "DISABLED"))

    side = str(row.get("side_policy") or row.get("side") or row.get("family") or "").upper()
    if side and not any(token in side for token in ("CE", "PE", "BOTH", "AUTO", "ATM", "DIRECTIONAL", "SYMMETRIC")):
        return fail(STAGE_INVALID_CONFIG, "SIDE_OR_FAMILY_INVALID")
    threshold = row.get("threshold", row.get("selected_threshold", row.get("entry_threshold", None)))
    if threshold not in (None, ""):
        try:
            threshold_f = float(threshold)
            if threshold_f < 0 or threshold_f > 1:
                return fail(STAGE_INVALID_CONFIG, "THRESHOLD_INVALID")
        except Exception:
            return fail(STAGE_INVALID_CONFIG, "THRESHOLD_INVALID")

    duplicates = set(config.get("duplicate_candidate_ids") or [])
    duplicate_keys = set(config.get("duplicate_logical_keys") or [])
    if cid in duplicates or _logical_candidate_key(row) in duplicate_keys or row.get("_duplicate_active"):
        return fail(STAGE_INVALID_CONFIG, "DUPLICATE_CANDIDATE_IDENTITY")

    artifact_dir_raw = _candidate_artifact_path(row)
    if not artifact_dir_raw:
        return fail(STAGE_ARTIFACT_MISSING, "ARTIFACT_PATH_MISSING")
    artifact_dir = Path(artifact_dir_raw)
    if not artifact_dir.is_absolute():
        artifact_dir = REPO_ROOT / artifact_dir
    if not artifact_dir.exists() or not artifact_dir.is_dir():
        return fail(STAGE_ARTIFACT_MISSING, "ARTIFACT_DIR_MISSING")

    model_path = Path(_candidate_model_path(row, artifact_dir))
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    if not model_path.exists():
        return fail(STAGE_ARTIFACT_MISSING, "MODEL_FILE_MISSING")

    shared_artifact_paths = set(config.get("shared_artifact_paths") or [])
    if str(artifact_dir.resolve()).lower() in shared_artifact_paths and not row.get("allow_shared_artifact"):
        return fail(STAGE_INVALID_CONFIG, "ARTIFACT_SHARED_BY_UNRELATED_CANDIDATES")

    meta, feature_order, meta_source = _load_artifact_metadata(artifact_dir)
    if not meta_source:
        return fail(STAGE_ARTIFACT_MISSING, "MODEL_METADATA_MISSING")
    model_name = str(row.get("model_name") or meta.get("model_name") or "").strip()
    preset_family = str(row.get("preset_family") or meta.get("preset_family") or "").strip()
    if not model_name or model_name.lower() == "unknown":
        return fail(STAGE_INVALID_CONFIG, "MODEL_NAME_UNKNOWN")
    if not preset_family or preset_family.lower() == "unknown":
        return fail(STAGE_INVALID_CONFIG, "PRESET_FAMILY_UNKNOWN")
    if not feature_order:
        return fail(STAGE_ARTIFACT_MISSING, "FEATURE_ORDER_EMPTY")
    if int(row.get("required_features") or len(feature_order) or 0) <= 0:
        return fail(STAGE_ARTIFACT_MISSING, "REQUIRED_FEATURES_ZERO")
    ok_identity, identity_reason = _candidate_identity_matches(row, meta, artifact_dir)
    if not ok_identity:
        return fail(STAGE_INVALID_CONFIG, identity_reason)

    smoke_ok, smoke_reason = _dummy_prediction_check(model_path, feature_order)
    if not smoke_ok:
        return fail(STAGE_ARTIFACT_MISSING if smoke_reason == "MODEL_LOAD_FAILED" else STAGE_INVALID_CONFIG, smoke_reason)

    status.artifact_status = "OK"
    status.passed_requirements.extend([
        "config valid",
        "artifact directory exists",
        "model file exists",
        "feature_order non-empty",
        "artifact identity matched",
        smoke_reason,
    ])

    missing_features = _runtime_value(runtime_stats, cid, "missing_features", default=[])
    if missing_features:
        return fail(STAGE_INVALID_CONFIG, "FEATURES_MISSING")
    confidence = _runtime_value(runtime_stats, cid, "confidence", default=row.get("confidence"))
    predict_attempted = bool(_runtime_value(runtime_stats, cid, "predict_attempted", default=row.get("predict_attempted", False)))
    if confidence is None and predict_attempted:
        return fail(STAGE_INVALID_CONFIG, "PREDICT_CONFIDENCE_NONE")
    if _runtime_value(runtime_stats, cid, "route_error", default=False):
        return fail(STAGE_INVALID_CONFIG, "ROUTE_ERROR")

    has_chain = bool(_runtime_value(runtime_stats, cid, "has_chain", default=True))
    has_candles = bool(_runtime_value(runtime_stats, cid, "has_candles", default=True))
    option_chain_only = bool(row.get("option_chain_only") or row.get("supports_option_chain_only"))
    if not has_chain:
        status.data_status = "WAITING_FOR_CHAIN"
        return fail(STAGE_INVALID_CONFIG, "HAS_CHAIN_FALSE")
    if not has_candles and not option_chain_only:
        status.data_status = "WAITING_FOR_CANDLES"
        return fail(STAGE_INVALID_CONFIG, "HAS_CANDLES_FALSE")
    status.data_status = "OK"

    paper_days = int(_runtime_value(runtime_stats, cid, "paper_forward_days", "paper_days_tested", "days_tested", default=row.get("paper_days_tested", 0)) or 0)
    paper_trades = int(_runtime_value(runtime_stats, cid, "paper_forward_trades", "paper_trade_count", "trade_count", default=row.get("paper_trade_count", 0)) or 0)
    shadow_days = int(_runtime_value(runtime_stats, cid, "shadow_days", default=row.get("shadow_days", 0)) or 0)
    shadow_trades = int(_runtime_value(runtime_stats, cid, "shadow_trades", "shadow_trade_count", default=row.get("shadow_trade_count", 0)) or 0)
    dryrun_days = int(_runtime_value(runtime_stats, cid, "dryrun_days", default=row.get("dryrun_days", 0)) or 0)
    dryrun_payloads = int(_runtime_value(runtime_stats, cid, "dryrun_payload_count", "dryrun_payloads", default=row.get("dryrun_payload_count", 0)) or 0)
    status.trades = max(paper_trades, shadow_trades, dryrun_payloads)
    status.days = max(paper_days, shadow_days, dryrun_days)
    status.confidence_health = str(_runtime_value(runtime_stats, cid, "confidence_health", default="OK"))
    if status.confidence_health.upper() in {"STUCK_ZERO", "CONFIDENCE_STUCK_ZERO"}:
        return fail(STAGE_INVALID_CONFIG, "CONFIDENCE_STUCK_ZERO")

    thresholds = {
        "min_paper_forward_days": 3,
        "min_paper_forward_trades": 20,
        "max_prediction_errors": 0,
        "max_route_errors": 0,
        "max_data_errors": 0,
        "max_consecutive_no_snapshot": 3,
        "min_confidence_health_rate": 0.95,
        "min_fillable_signal_count": 1,
        "min_shadow_days": 2,
        "min_shadow_trades": 10,
        "min_dryrun_days": 1,
        "min_dryrun_payloads": 1,
    }
    thresholds.update(config.get("lifecycle_thresholds") or {})
    for key in ("prediction_errors", "route_errors", "data_errors"):
        if int(_runtime_value(runtime_stats, cid, key, default=row.get(key, 0)) or 0) > int(thresholds.get("max_" + key, 0)):
            return fail(STAGE_INVALID_CONFIG, key.upper())
    if int(_runtime_value(runtime_stats, cid, "consecutive_no_snapshot", default=0) or 0) > int(thresholds["max_consecutive_no_snapshot"]):
        return fail(STAGE_INVALID_CONFIG, "STALE_SNAPSHOT_LOOP")
    if bool(_runtime_value(runtime_stats, cid, "broker_auth_loop", default=False)):
        return fail(STAGE_INVALID_CONFIG, "BROKEN_TOKEN_AUTH_LOOP")

    status.current_stage = STAGE_PAPER_FORWARD_ELIGIBLE
    status.next_stage = STAGE_PAPER_FORWARD_RUNNING
    status.can_advance = bool(status.possible_live_path)
    if not status.possible_live_path:
        status.block_reason = "PAPER_OBSERVATION_ONLY_NOT_LIVE_PATH"
    status.promotion_actions_available.append("start_paper_forward")

    if str(row.get("status") or "").upper() in {STAGE_PAPER_FORWARD_RUNNING, "PAPER_FORWARD", STATUS_PAPER_FORWARD}:
        status.current_stage = STAGE_PAPER_FORWARD_RUNNING
        status.can_advance = False
        status.block_reason = "PAPER_FORWARD_RUNNING"

    paper_passed = bool(row.get("paper_forward_passed") or row.get("shadow_ready"))
    if not paper_passed:
        paper_passed = (
            paper_days >= int(thresholds["min_paper_forward_days"])
            and paper_trades >= int(thresholds["min_paper_forward_trades"])
            and int(_runtime_value(runtime_stats, cid, "fillable_signal_count", default=thresholds["min_fillable_signal_count"]) or 0) >= int(thresholds["min_fillable_signal_count"])
        )
    if paper_passed:
        status.possible_live_path = True
        status.current_stage = STAGE_SHADOW_ELIGIBLE
        status.next_stage = STAGE_SHADOW_RUNNING
        status.can_advance = True
        status.block_reason = ""
        status.promotion_actions_available = ["promote_to_shadow"]

    if bool(row.get("shadow_running")) or str(row.get("status") or "").upper() in {STAGE_SHADOW_RUNNING, STATUS_SHADOW}:
        status.current_stage = STAGE_SHADOW_RUNNING
        status.can_advance = False
        status.block_reason = "SHADOW_RUNNING"
    shadow_passed = bool(row.get("shadow_passed")) or (
        shadow_days >= int(thresholds["min_shadow_days"])
        and shadow_trades >= int(thresholds["min_shadow_trades"])
        and int(_runtime_value(runtime_stats, cid, "critical_runtime_errors", default=0) or 0) == 0
        and not bool(_runtime_value(runtime_stats, cid, "feature_drift", default=False))
    )
    if shadow_passed:
        status.current_stage = STAGE_LIVE_DRY_RUN_ELIGIBLE
        status.next_stage = STAGE_LIVE_DRY_RUN_RUNNING
        status.can_advance = live_dry_run_requested()
        status.block_reason = "" if status.can_advance else "LIVE_DRY_RUN_ENV_NOT_SAFE"
        status.promotion_actions_available = ["promote_to_live_dry_run"]

    dryrun_passed = bool(row.get("live_dry_run_passed") or row.get("dryrun_passed")) or (
        dryrun_days >= int(thresholds["min_dryrun_days"])
        and dryrun_payloads >= int(thresholds["min_dryrun_payloads"])
        and int(_runtime_value(runtime_stats, cid, "dryrun_payload_errors", default=0) or 0) == 0
    )
    if dryrun_passed:
        status.current_stage = STAGE_LIVE_1_LOT_ELIGIBLE
        status.next_stage = STAGE_LIVE_1_LOT_DEPLOYED
        whitelist = list(config.get("live_1lot_whitelist") or [])
        selected_count = int(config.get("live_1lot_selected_count", len(whitelist)))
        max_qty = int(config.get("max_qty", LIVE_ONE_LOT_QTY) or LIVE_ONE_LOT_QTY)
        manual = str(os.getenv("REQUIRE_MANUAL_CONFIRM", str(config.get("require_manual_confirm", "true")))).lower() in {"1", "true", "yes"}
        if selected_count != 1 or (whitelist and cid not in whitelist):
            status.can_advance = False
            status.block_reason = "LIVE_1_LOT_REQUIRES_EXACTLY_ONE_WHITELISTED_CANDIDATE"
        elif max_qty > LIVE_ONE_LOT_QTY and not config.get("allow_qty_above_one_lot"):
            status.can_advance = False
            status.block_reason = "MAX_QTY_EXCEEDS_ONE_NIFTY_LOT"
        elif not manual:
            status.can_advance = False
            status.block_reason = "REQUIRE_MANUAL_CONFIRM_FALSE"
        elif get_order_placement_enabled_env():
            status.can_advance = False
            status.block_reason = "ORDER_PLACEMENT_ENABLED_MUST_REMAIN_FALSE_UNTIL_MANUAL_DEPLOY"
        else:
            status.can_advance = True
            status.block_reason = ""
        status.can_be_live_deployed = status.can_advance
        status.promotion_actions_available = ["promote_best_to_live_1lot"]

    if bool(row.get("live_1lot_deployed")) or str(row.get("status") or "").upper() in {STAGE_LIVE_1_LOT_DEPLOYED, STATUS_LIVE_1_LOT}:
        status.current_stage = STAGE_LIVE_1_LOT_DEPLOYED
        status.next_stage = ""
        status.can_advance = False
        status.can_be_live_deployed = False
        status.block_reason = "ALREADY_DEPLOYED"

    status.risk_status = "SAFE_DRY_RUN" if live_dry_run_requested() else "SAFE_NO_REAL_ORDERS"
    return status


def build_lifecycle_context(candidates: Iterable[Dict[str, Any]], *, live_whitelist: Iterable[str] | None = None) -> Dict[str, Any]:
    ids: Dict[str, int] = {}
    logical: Dict[str, int] = {}
    artifact_to_ids: Dict[str, set[str]] = {}
    rows = [dict(c) for c in candidates if isinstance(c, dict)]
    for row in rows:
        cid = _candidate_id(row)
        if cid:
            ids[cid] = ids.get(cid, 0) + 1
        key = _logical_candidate_key(row)
        if key:
            logical[key] = logical.get(key, 0) + 1
        art = _candidate_artifact_path(row)
        if art:
            p = Path(art)
            if not p.is_absolute():
                p = REPO_ROOT / p
            artifact_to_ids.setdefault(str(p.resolve()).lower(), set()).add(cid)
    return {
        "duplicate_candidate_ids": [cid for cid, count in ids.items() if count > 1],
        "duplicate_logical_keys": [key for key, count in logical.items() if count > 1],
        "shared_artifact_paths": [p for p, idset in artifact_to_ids.items() if len({x for x in idset if x}) > 1],
        "live_1lot_whitelist": list(live_whitelist or []),
        "live_1lot_selected_count": len(list(live_whitelist or [])),
        "max_qty": LIVE_ONE_LOT_QTY,
        "require_manual_confirm": True,
    }


def load_paper_forward_candidate_rows(path: str | Path = PAPER_FORWARD_CANDIDATES_PATH) -> List[Dict[str, Any]]:
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload.get("candidates", payload if isinstance(payload, list) else [])
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        for key in ("enabled", "active", "selected"):
            if key in item:
                item[key] = _coerce_optional_bool(item.get(key))
        normalized_rows.append(item)
    return normalized_rows


def evaluate_candidate_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    runtime_stats: Dict[str, Any] | None = None,
    config: Dict[str, Any] | None = None,
) -> List[CandidateLifecycleStatus]:
    rows_list = [dict(r) for r in rows if isinstance(r, dict)]
    context = build_lifecycle_context(rows_list)
    if config:
        context.update(config)
    return [evaluate_candidate_lifecycle(row, {}, runtime_stats or {}, context) for row in rows_list]


def lifecycle_status_visible(status: CandidateLifecycleStatus, view: str = "qualified") -> bool:
    view_key = (view or "qualified").strip().lower()
    stage = status.current_stage
    if view_key in {"all", "all debug", "show all debug"}:
        return True
    if view_key in {"blocked", "blocked / rejected", "show blocked / rejected"}:
        return stage not in VISIBLE_BY_DEFAULT_STAGES or bool(status.block_reason)
    if view_key in {"paper", "paper forward eligible", "show paper forward eligible"}:
        return stage in {STAGE_PAPER_FORWARD_ELIGIBLE, STAGE_PAPER_FORWARD_RUNNING, STAGE_PAPER_FORWARD_PASSED}
    if view_key in {"shadow", "shadow eligible", "show shadow eligible"}:
        return stage in {STAGE_SHADOW_ELIGIBLE, STAGE_SHADOW_RUNNING, STAGE_SHADOW_PASSED}
    if view_key in {"live", "live eligible", "show live eligible"}:
        return stage in {STAGE_LIVE_DRY_RUN_ELIGIBLE, STAGE_LIVE_DRY_RUN_RUNNING, STAGE_LIVE_DRY_RUN_PASSED, STAGE_LIVE_1_LOT_ELIGIBLE, STAGE_LIVE_1_LOT_DEPLOYED}
    return stage in VISIBLE_BY_DEFAULT_STAGES and status.possible_live_path and not status.block_reason


def lifecycle_row_tag(status: CandidateLifecycleStatus) -> str:
    if status.current_stage in {STAGE_DISABLED}:
        return "disabled"
    if status.current_stage in {STAGE_LIVE_DRY_RUN_ELIGIBLE, STAGE_LIVE_DRY_RUN_RUNNING, STAGE_LIVE_DRY_RUN_PASSED, STAGE_LIVE_1_LOT_ELIGIBLE, STAGE_LIVE_1_LOT_DEPLOYED}:
        return "live"
    if status.can_advance:
        return "advance"
    if status.current_stage.endswith("_RUNNING") or "WAITING" in status.data_status:
        return "waiting"
    if status.block_reason:
        return "blocked"
    return "waiting"


def write_lifecycle_report(
    statuses: Iterable[CandidateLifecycleStatus],
    *,
    archived: Iterable[Dict[str, Any]] | None = None,
    reports_dir: str | Path = REPORTS_DIR / "candidate_lifecycle",
) -> Tuple[Path, Path]:
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    statuses_list = list(statuses)
    archived_list = list(archived or [])
    stage_counts: Dict[str, int] = {}
    block_reasons: Dict[str, int] = {}
    for st in statuses_list:
        stage_counts[st.current_stage] = stage_counts.get(st.current_stage, 0) + 1
        if st.block_reason:
            block_reasons[st.block_reason] = block_reasons.get(st.block_reason, 0) + 1
    payload = {
        "generated_at": _now_iso(),
        "all_candidates": [st.to_dict() for st in statuses_list],
        "active_candidates": [st.to_dict() for st in statuses_list if lifecycle_status_visible(st, "qualified")],
        "archived_candidates": archived_list,
        "stage_counts": stage_counts,
        "block_reasons": block_reasons,
        "live_readiness_status": {
            "live_mode": get_live_mode_env(),
            "live_order_dry_run": get_live_order_dry_run_env(),
            "order_placement_enabled": get_order_placement_enabled_env(),
            "kill_switch_active": get_kill_switch_active(),
        },
    }
    json_path = out_dir / "latest_candidate_lifecycle_report.json"
    csv_path = out_dir / "latest_candidate_lifecycle_report.csv"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    import csv
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "candidate_id", "family", "side", "preset_family", "current_stage", "next_stage",
            "can_advance", "can_be_live_deployed", "block_reason", "missing_requirements",
            "artifact_status", "data_status", "confidence_health", "trades", "days", "last_updated",
        ])
        writer.writeheader()
        for st in statuses_list:
            row = st.to_dict()
            row["missing_requirements"] = ";".join(st.missing_requirements)
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames or []})
    return json_path, csv_path
