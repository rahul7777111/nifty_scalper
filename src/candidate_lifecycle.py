#!/usr/bin/env python3
"""
Minimal candidate lifecycle helpers for paper/live safety.

Paper-forward lifecycle analytics, shadow-mode helpers, GUI filtering, and
promotion dashboards were removed. This module now keeps only the runtime-safe
candidate record, live whitelist storage, promotion helpers, and env guards
used by paper/live paths.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STATUS_DISABLED = "DISABLED"
STATUS_BACKTEST_PASSED = "BACKTEST_PASSED"
STATUS_PAPER_FORWARD = "PAPER_FORWARD"
STATUS_PAPER_PASSED = "PAPER_PASSED"
STATUS_LIVE_DRY_RUN = "LIVE_DRY_RUN"
STATUS_LIVE_1_LOT = "LIVE_1_LOT"
STATUS_LIVE_SCALED = "LIVE_SCALED"

ALL_STATUSES = [
    STATUS_DISABLED,
    STATUS_BACKTEST_PASSED,
    STATUS_PAPER_FORWARD,
    STATUS_PAPER_PASSED,
    STATUS_LIVE_DRY_RUN,
    STATUS_LIVE_1_LOT,
    STATUS_LIVE_SCALED,
]

ALLOWED_MODES_BY_STATUS = {
    STATUS_DISABLED: [],
    STATUS_BACKTEST_PASSED: ["backtest"],
    STATUS_PAPER_FORWARD: ["paper_forward"],
    STATUS_PAPER_PASSED: ["paper_forward"],
    STATUS_LIVE_DRY_RUN: ["live_dry_run"],
    STATUS_LIVE_1_LOT: ["live_dry_run", "live_1_lot"],
    STATUS_LIVE_SCALED: ["live_dry_run", "live_1_lot", "live_scaled"],
}


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
    paper_trade_count: int = 0
    paper_days_tested: int = 0
    paper_cost_adjusted_pnl: float = 0.0
    paper_profit_factor: float = 0.0
    paper_win_rate: float = 0.0
    paper_max_drawdown: float = 0.0
    dryrun_payload_count: int = 0
    dryrun_days: int = 0
    last_block_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["allowed_modes"] = list(payload.get("allowed_modes") or [])
        payload["allowed_option_types"] = list(payload.get("allowed_option_types") or ["CE", "PE"])
        return payload

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Candidate":
        allowed_modes = data.get("allowed_modes") or ALLOWED_MODES_BY_STATUS.get(str(data.get("status") or ""), [])
        if isinstance(allowed_modes, str):
            allowed_modes = [allowed_modes]
        allowed_option_types = data.get("allowed_option_types") or ["CE", "PE"]
        if isinstance(allowed_option_types, str):
            allowed_option_types = [allowed_option_types]
        return Candidate(
            candidate_id=str(data.get("candidate_id", "")),
            model_path=str(data.get("model_path", "")),
            artifact_path=str(data.get("artifact_path", data.get("artifact_dir", ""))),
            status=str(data.get("status", STATUS_DISABLED)),
            allowed_modes=list(allowed_modes),
            enabled=bool(data.get("enabled", False)),
            live_whitelisted=bool(data.get("live_whitelisted", False)),
            max_qty_lots=int(data.get("max_qty_lots", 0) or 0),
            max_trades_per_day=int(data.get("max_trades_per_day", 0) or 0),
            max_daily_loss=float(data.get("max_daily_loss", 0.0) or 0.0),
            allowed_option_types=list(allowed_option_types),
            promotion_notes=str(data.get("promotion_notes", "")),
            last_promoted_at=data.get("last_promoted_at"),
            paper_trade_count=int(data.get("paper_trade_count", 0) or 0),
            paper_days_tested=int(data.get("paper_days_tested", data.get("days_tested", 0)) or 0),
            paper_cost_adjusted_pnl=float(data.get("paper_cost_adjusted_pnl", 0.0) or 0.0),
            paper_profit_factor=float(data.get("paper_profit_factor", 0.0) or 0.0),
            paper_win_rate=float(data.get("paper_win_rate", 0.0) or 0.0),
            paper_max_drawdown=float(data.get("paper_max_drawdown", 0.0) or 0.0),
            dryrun_payload_count=int(data.get("dryrun_payload_count", 0) or 0),
            dryrun_days=int(data.get("dryrun_days", 0) or 0),
            last_block_reason=str(data.get("last_block_reason", data.get("blocker_reason", ""))),
        )


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
LIVE_WHITELIST_PATH = CONFIG_DIR / "live_candidate_whitelist.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_candidates(path: Path) -> List[Candidate]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = data.get("candidates", data if isinstance(data, list) else [])
    return [Candidate.from_dict(item) for item in items if isinstance(item, dict)]


def _write_candidates(path: Path, candidates: List[Candidate], *, mode: str, safety_note: str, meta: Optional[Dict[str, Any]] = None) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "updated_at": _now_iso(),
        "mode": mode,
        "safety_note": safety_note,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
    if meta:
        payload.update(meta)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_live_whitelist() -> List[Candidate]:
    return _read_candidates(LIVE_WHITELIST_PATH)


def save_live_whitelist(cands: List[Candidate], *, meta: Optional[Dict[str, Any]] = None) -> None:
    _write_candidates(
        LIVE_WHITELIST_PATH,
        cands,
        mode="live_dryrun_or_1lot_whitelist",
        safety_note="Only LIVE_1_LOT/LIVE_SCALED whitelisted candidates may attempt real orders.",
        meta=meta,
    )


def get_candidate_by_id(cands: List[Candidate], cid: str) -> Optional[Candidate]:
    for candidate in cands:
        if candidate.candidate_id == cid:
            return candidate
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def promote_candidate(candidate: Candidate, new_status: str, *, notes: str = "") -> Candidate:
    if new_status not in ALL_STATUSES:
        raise ValueError(f"unknown candidate status: {new_status}")
    candidate.status = new_status
    candidate.allowed_modes = list(ALLOWED_MODES_BY_STATUS.get(new_status, []))
    candidate.last_promoted_at = _now_iso()
    if notes:
        candidate.promotion_notes = notes
    if new_status not in (STATUS_LIVE_DRY_RUN, STATUS_LIVE_1_LOT, STATUS_LIVE_SCALED):
        candidate.live_whitelisted = False
        if new_status in (STATUS_DISABLED, STATUS_BACKTEST_PASSED, STATUS_PAPER_FORWARD, STATUS_PAPER_PASSED):
            candidate.max_qty_lots = 0
    if new_status == STATUS_LIVE_DRY_RUN:
        candidate.live_whitelisted = True
        candidate.max_qty_lots = max(int(candidate.max_qty_lots or 0), 1)
    if new_status == STATUS_LIVE_1_LOT:
        candidate.live_whitelisted = True
        candidate.max_qty_lots = 1
    return candidate


def get_live_mode_env() -> bool:
    return str(os.getenv("LIVE_MODE", "")).strip().lower() in {"1", "true", "yes", "on"}


def get_live_order_dry_run_env() -> bool:
    return str(os.getenv("LIVE_ORDER_DRY_RUN", "true")).strip().lower() in {"1", "true", "yes", "on"}


def get_order_placement_enabled_env() -> bool:
    return str(os.getenv("ORDER_PLACEMENT_ENABLED", "")).strip().lower() in {"1", "true", "yes", "on"}


def get_kill_switch_active() -> bool:
    return str(os.getenv("SCALPER_KILL_SWITCH", "")).strip().lower() in {"1", "true", "yes", "on"}
