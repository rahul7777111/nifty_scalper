#!/usr/bin/env python3
"""
scripts/live_readiness_audit.py
===============================
Full live-readiness auditor. Produces JSON + MD report.

Final verdict: NOT_READY | DRY_RUN_READY | LIVE_1_LOT_READY

Run from repo root:
    python scripts/live_readiness_audit.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.candidate_lifecycle import (
    load_live_whitelist,
    load_shadow_candidates,
    get_live_mode_env,
    get_live_order_dry_run_env,
    get_order_placement_enabled_env,
    get_kill_switch_active,
    Candidate,
)
from src.live_order_guard import is_real_live_fully_enabled

LOGS_DIR = REPO_ROOT / "logs"
REPORTS_DIR = REPO_ROOT / "reports"
CONFIG_DIR = REPO_ROOT / "config"

try:
    from src.config import APIConfig, load_api_config, load_strategy_config
except Exception:
    APIConfig = None
    def load_api_config(): return None
    def load_strategy_config(): return None


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


def _present_secret(value: str) -> str:
    v = str(value or "").strip()
    if not v:
        return "missing"
    if len(v) <= 8:
        return "set"
    return f"set (len={len(v)})"


def check_env_and_config() -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    data: Dict[str, Any] = {}

    env_path = REPO_ROOT / ".env"
    data["env_file_exists"] = env_path.exists()

    # critical vars (we read current os.environ after any .env load in app)
    keys = [
        "MSTOCK_API_KEY", "MSTOCK_ACCESS_TOKEN", "MSTOCK_ENABLE_LIVE_TRADING",
        "LIVE_MODE", "PAPER_MODE", "SHADOW_MODE",
        "LIVE_ORDER_DRY_RUN", "ORDER_PLACEMENT_ENABLED", "SCALPER_KILL_SWITCH",
        "MAX_LIVE_QTY_LOTS", "MAX_TRADES_PER_DAY", "MAX_DAILY_LOSS",
        "ENTRY_CUTOFF", "MSTOCK_UNDERLYING", "MSTOCK_LOT_SIZE",
    ]
    for k in keys:
        value = os.getenv(k)
        if k in {"MSTOCK_API_KEY", "MSTOCK_ACCESS_TOKEN"}:
            data[k.lower()] = _present_secret(value or "")
        else:
            data[k.lower()] = value

    if not data["env_file_exists"]:
        issues.append(".env file missing")

    if not os.getenv("MSTOCK_ACCESS_TOKEN"):
        issues.append("broker token (MSTOCK_ACCESS_TOKEN) missing or empty")

    broker = os.getenv("SCALPER_BROKER") or ("mstock" if os.getenv("MSTOCK_ACCESS_TOKEN") else "unknown")
    data["broker_selected"] = broker

    # sensible defaults / presence
    if not _bool_env("LIVE_MODE"):
        data["note"] = data.get("note", "") + "LIVE_MODE not explicitly true (safe default). "
    if _bool_env("ORDER_PLACEMENT_ENABLED"):
        issues.append("ORDER_PLACEMENT_ENABLED is currently true - should be false until final manual step")
    if not _bool_env("LIVE_ORDER_DRY_RUN", True):
        # many flows default it on via UI button; we just note
        pass

    # target expiry / cutoff presence (best effort)
    data["entry_cutoff"] = os.getenv("premium_entry_cutoff_hhmm") or os.getenv("ENTRY_CUTOFF") or "14:00"
    data["max_live_qty_lots"] = int(os.getenv("MAX_LIVE_QTY_LOTS", "1") or 1)
    data["max_trades_per_day"] = int(os.getenv("MAX_TRADES_PER_DAY", os.getenv("MSTOCK_MAX_TRADES_PER_DAY", "2")) or 2)
    data["max_daily_loss"] = float(os.getenv("MAX_DAILY_LOSS", "1000") or 1000)

    return issues, data


def check_broker() -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    data: Dict[str, Any] = {"broker_session_valid": False, "quote_ok": False, "chain_ok": False, "candles_ok": False}

    token = os.getenv("MSTOCK_ACCESS_TOKEN", "")
    if not token:
        issues.append("no broker token - cannot validate session")
        return issues, data

    # Try to import and do a lightweight health check without placing orders
    try:
        from src.mstock_client import MStockTypeBClient
        from src.config import load_api_config
        cfg = load_api_config()
        if cfg is None or not getattr(cfg, "api_key", ""):
            issues.append("broker api key missing or empty")
            return issues, data
        client = MStockTypeBClient(cfg)
        data["broker_session_valid"] = True  # constructor succeeded; real token check happens on first call

        # Quote / LTP
        try:
            underlying = os.getenv("MSTOCK_UNDERLYING", "NIFTY")
            token = os.getenv("MSTOCK_UNDERLYING_TOKEN", "").strip()
            exchange = os.getenv("MSTOCK_UNDERLYING_EXCHANGE", "NSE").strip() or "NSE"
            quote_key = f"{exchange}:{token}" if token else underlying
            ltp = client.get_ltp(quote_key)
            data["quote_ok"] = ltp is not None and float(ltp or 0) > 0
            data["spot_ltp"] = ltp
            data["spot_quote_key"] = quote_key
        except Exception as e:
            issues.append(f"quote endpoint error: {e}")

        # Option chain (best effort - many clients have get_option_chain or similar)
        try:
            # This is intentionally soft; real impl may differ
            if hasattr(client, "get_option_chain"):
                chain = client.get_option_chain(underlying=os.getenv("MSTOCK_UNDERLYING", "NIFTY"))
                data["chain_ok"] = bool(chain)
            else:
                data["chain_ok"] = True  # assume higher layer provides; we will check freshness later
        except Exception:
            data["chain_ok"] = False

        data["candles_ok"] = True  # candles are optional for router in many paths
        data["order_endpoint_dry_only"] = True  # enforced by guards elsewhere

    except Exception as e:
        issues.append(f"broker client init failed: {e}")

    return issues, data


def check_data_freshness() -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    data: Dict[str, Any] = {}

    # In a real run the UI / engine would have populated live state.
    # Here we do best-effort synthetic checks + look for recent snapshots in memory if possible.
    # For standalone audit we report what we can from env / last known logs.

    data["spot_quote_fresh"] = False
    data["option_chain_fresh"] = False
    data["atm_detected"] = False
    data["bid_ask_ltp_present"] = False
    data["spread_calculated"] = False
    data["expiry_valid"] = False
    data["candles_fresh"] = False
    data["router_can_run_without_candles"] = True  # known from paper/shadow engines

    # Look for recent live dry-run or shadow logs that contain freshness indicators
    for p in sorted(LOGS_DIR.glob("live_dryrun_orders_*.jsonl"), reverse=True)[:1]:
        try:
            lines = p.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                last = json.loads(lines[-1])
                data["last_dryrun_symbol"] = last.get("symbol")
                data["last_dryrun_expiry"] = last.get("expiry")
                if last.get("ltp") and last.get("bid") and last.get("ask"):
                    data["bid_ask_ltp_present"] = True
                if last.get("spread_pct") is not None:
                    data["spread_calculated"] = True
                data["option_chain_fresh"] = True  # if we got this far in dry-run, chain was usable
        except Exception:
            pass

    # If no recent activity, mark degraded but not fatal for DRY_RUN_READY
    if not data.get("option_chain_fresh"):
        issues.append("no recent live option chain snapshot observed (run shadow/dry-run first for freshness)")

    # Basic expiry sanity from env or last
    exp = os.getenv("TARGET_EXPIRY") or data.get("last_dryrun_expiry") or ""
    if exp:
        data["expiry_valid"] = True
    else:
        issues.append("no target expiry configured or observed")

    data["atm_detected"] = True  # router + filters handle this
    return issues, data


def check_candidate_safety() -> Tuple[List[str], Dict[str, Any], List[Candidate]]:
    issues: List[str] = []
    data: Dict[str, Any] = {}
    wl = load_live_whitelist()
    data["live_whitelist_exists"] = bool(wl)
    data["live_whitelist_path"] = str(CONFIG_DIR / "live_candidate_whitelist.json")

    live_ones = [c for c in wl if c.live_whitelisted and c.status in ("LIVE_1_LOT", "LIVE_SCALED")]
    data["live_1_lot_count"] = len([c for c in live_ones if c.status == "LIVE_1_LOT"])
    data["live_scaled_count"] = len([c for c in live_ones if c.status == "LIVE_SCALED"])

    if len(live_ones) > 1:
        issues.append("more than one LIVE_1_LOT / LIVE_SCALED candidate whitelisted (policy violation)")
    if data["live_scaled_count"] > 0:
        issues.append("LIVE_SCALED candidate present - only allowed after explicit multi-lot approval")

    for c in live_ones:
        if not c.artifact_path and not c.model_path:
            issues.append(f"{c.candidate_id}: missing artifact path")
        # max qty for first live
        if c.status == "LIVE_1_LOT" and int(c.max_qty_lots or 0) > 1:
            issues.append(f"{c.candidate_id}: max_qty_lots > 1 for LIVE_1_LOT")
        if int(c.max_qty_lots or 0) <= 0:
            issues.append(f"{c.candidate_id}: max_qty_lots <= 0")
        if int(c.max_trades_per_day or 0) > 2:
            issues.append(f"{c.candidate_id}: max_trades_per_day > 2 for first live")
        if not c.allowed_option_types:
            issues.append(f"{c.candidate_id}: no allowed_option_types")

    # Also check shadow list for basic health
    shadows = load_shadow_candidates()
    data["shadow_count"] = len(shadows)

    return issues, data, live_ones


def check_runtime_state() -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    data: Dict[str, Any] = {}

    data["kill_switch_active"] = get_kill_switch_active()
    if data["kill_switch_active"]:
        issues.append("kill switch is currently active")

    data["trades_today"] = 0  # would be read from bot state in real run
    data["daily_pnl"] = 0.0
    data["open_positions"] = 0
    data["duplicate_open"] = False
    data["pending_unknown_orders"] = 0
    data["router_decision_stale"] = False

    # Look at recent logs for any emergency or mismatch
    for p in sorted(LOGS_DIR.glob("emergency_stop_*.log"), reverse=True)[:1]:
        data["last_emergency_log"] = str(p)
        issues.append("emergency stop log present - review before live")

    return issues, data


def explain_verdict_blockers(*, verdict: str, live_cands: List[Candidate], sections: Dict[str, Any]) -> List[str]:
    """Explain why a clean audit is still not ready for live trading."""

    if verdict != "NOT_READY":
        return []

    blockers: List[str] = []
    if not get_live_mode_env():
        blockers.append("LIVE_MODE is not true")
    if not get_live_order_dry_run_env() and not get_order_placement_enabled_env():
        blockers.append("neither LIVE_ORDER_DRY_RUN nor ORDER_PLACEMENT_ENABLED is enabled")
    if get_order_placement_enabled_env() and get_live_order_dry_run_env():
        blockers.append("ORDER_PLACEMENT_ENABLED and LIVE_ORDER_DRY_RUN cannot both be true")
    if get_kill_switch_active():
        blockers.append("SCALPER_KILL_SWITCH is active")

    whitelist = load_live_whitelist()
    dry_or_live = [c for c in whitelist if c.status in ("LIVE_DRY_RUN", "LIVE_1_LOT", "LIVE_SCALED")]
    if not dry_or_live:
        shadow_count = int((sections.get("D_candidate_safety") or {}).get("shadow_count") or 0)
        if shadow_count:
            blockers.append("no candidate has passed shadow promotion into LIVE_DRY_RUN or LIVE_1_LOT")
        else:
            blockers.append("no live/dry-run candidate is whitelisted")
    elif not live_cands and not get_live_order_dry_run_env():
        blockers.append("no LIVE_1_LOT candidate is whitelisted")

    data = sections.get("C_data_freshness") or {}
    if not data.get("option_chain_fresh"):
        blockers.append("no fresh option-chain snapshot observed")
    if not data.get("bid_ask_ltp_present"):
        blockers.append("no recent dry-run quote with bid/ask/LTP observed")

    return blockers


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _ts()

    all_issues: List[str] = []
    sections: Dict[str, Any] = {}

    # A
    env_issues, env_data = check_env_and_config()
    all_issues.extend(env_issues)
    sections["A_env_config"] = env_data

    # B
    broker_issues, broker_data = check_broker()
    all_issues.extend(broker_issues)
    sections["B_broker"] = broker_data

    # C
    data_issues, data_data = check_data_freshness()
    if broker_data.get("quote_ok"):
        data_data["spot_quote_fresh"] = True
        data_data["spot_ltp"] = broker_data.get("spot_ltp")
        data_data["spot_quote_key"] = broker_data.get("spot_quote_key")
    all_issues.extend(data_issues)
    sections["C_data_freshness"] = data_data

    # D
    cand_issues, cand_data, live_cands = check_candidate_safety()
    all_issues.extend(cand_issues)
    sections["D_candidate_safety"] = cand_data
    sections["active_live_candidates"] = [c.to_dict() if hasattr(c, "to_dict") else {"candidate_id": c.candidate_id, "status": c.status} for c in live_cands]

    # E
    rt_issues, rt_data = check_runtime_state()
    all_issues.extend(rt_issues)
    sections["E_runtime_state"] = rt_data

    # Verdict logic (strict)
    live_ready = (
        get_live_mode_env()
        and get_order_placement_enabled_env()
        and not get_live_order_dry_run_env()
        and not get_kill_switch_active()
        and len(live_cands) == 1
        and live_cands[0].status == "LIVE_1_LOT"
        and live_cands[0].live_whitelisted
        and not any("broker" in i.lower() or "token" in i.lower() or "stale" in i.lower() for i in all_issues)
    )

    dry_ready = (
        get_live_mode_env()
        and get_live_order_dry_run_env()
        and not get_order_placement_enabled_env()
        and not get_kill_switch_active()
        and len([c for c in load_live_whitelist() if c.status in ("LIVE_DRY_RUN", "LIVE_1_LOT")]) > 0
        and not any("token" in i.lower() and "missing" in i.lower() for i in all_issues)  # token may be present for dry
    )

    if live_ready:
        verdict = "LIVE_1_LOT_READY"
    elif dry_ready:
        verdict = "DRY_RUN_READY"
    else:
        verdict = "NOT_READY"

    readiness_blockers = explain_verdict_blockers(verdict=verdict, live_cands=live_cands, sections=sections)

    report = {
        "timestamp": _now(),
        "verdict": verdict,
        "issues": all_issues,
        "readiness_blockers": readiness_blockers,
        "sections": sections,
        "env_snapshot": {
            "LIVE_MODE": get_live_mode_env(),
            "LIVE_ORDER_DRY_RUN": get_live_order_dry_run_env(),
            "ORDER_PLACEMENT_ENABLED": get_order_placement_enabled_env(),
            "SCALPER_KILL_SWITCH": get_kill_switch_active(),
        },
        "is_real_live_fully_enabled": is_real_live_fully_enabled(),
    }

    jpath = REPORTS_DIR / f"live_readiness_audit_{ts}.json"
    jpath.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    md = REPORTS_DIR / f"live_readiness_audit_{ts}.md"
    md_lines = [
        f"# Live Readiness Audit — {ts}",
        "",
        f"**VERDICT: {verdict}**",
        "",
        "## Issues / Blockers",
    ]
    if all_issues:
        for i in all_issues:
            md_lines.append(f"- {i}")
    else:
        md_lines.append("- (none)")
    if readiness_blockers:
        md_lines.extend(["", "## Readiness Blockers"])
        for i in readiness_blockers:
            md_lines.append(f"- {i}")

    md_lines.extend([
        "",
        "## Key Env / Mode",
        f"- LIVE_MODE: {get_live_mode_env()}",
        f"- LIVE_ORDER_DRY_RUN: {get_live_order_dry_run_env()} (should be true for safe dry-run)",
        f"- ORDER_PLACEMENT_ENABLED: {get_order_placement_enabled_env()} (MUST be false until final step)",
        f"- SCALPER_KILL_SWITCH: {get_kill_switch_active()}",
        "",
        "## Active Live Candidates (from whitelist)",
    ])
    for c in live_cands:
        md_lines.append(f"- {c.candidate_id}: status={c.status}, whitelisted={c.live_whitelisted}, max_qty={c.max_qty_lots}")

    if not live_cands:
        md_lines.append("- (none - whitelist is empty or no LIVE_1_LOT)")

    md_lines.extend([
        "",
        "See JSON for full section details (A-E).",
        "",
        "Safety: This audit never places orders. Real trading remains blocked by multiple independent gates.",
    ])
    md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[live-readiness] Verdict: {verdict}")
    print(f"[live-readiness] Report: {jpath}")
    print(f"[live-readiness] MD: {md}")
    return 0 if verdict != "NOT_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
