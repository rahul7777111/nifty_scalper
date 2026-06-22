#!/usr/bin/env python3
"""build_readiness_state.py — aggregate latest report evidence into readiness state."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS = REPO_ROOT / "reports"


def find_latest(pattern: str) -> Path | None:
    matches = sorted(REPORTS.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def load_json(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_stale(report_path: Path, max_age_hours: int = 48) -> bool:
    if not report_path or not report_path.exists():
        return True
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(report_path.stat().st_mtime, tz=timezone.utc)
    return age.total_seconds() > max_age_hours * 3600


def main() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper_engine_pass": False,
        "broker_safety_pass": False,
        "real_trading_gate_pass": False,
        "risk_control_gap_fix_pass": False,
        "model_signal_quality_verdict": "UNKNOWN",
        "model_pkl_exists": False,
        "dry_run_coverage_pct": 0.0,
        "latest_probability": 0.0,
        "probability_threshold": 0.5,
        "forward_test_decision_count": 0,
        "forward_test_trade_count": 0,
        "forward_test_net_pnl": 0.0,
        "forward_test_enough_evidence": False,
        "real_trading_allowed_now": False,
        "micro_live_allowed_now": False,
        "blockers": [],
        "reports_used": {},
    }

    def _add(name: str, path: Path | None) -> None:
        state["reports_used"][name] = str(path) if path else ""

    # ── Paper engine readiness ──────────────────────────────────────────────
    pe_path = find_latest("final_paper_engine_readiness_*.json")
    pe = load_json(pe_path)
    state["paper_engine_pass"] = pe.get("status") == "PASS"
    if not state["paper_engine_pass"]:
        state["blockers"].append("paper engine not PASS")
    _add("paper_engine", pe_path)

    # ── Broker safety ───────────────────────────────────────────────────────
    bs_path = find_latest("broker_safety_fix_*.json")
    bs = load_json(bs_path)
    state["broker_safety_pass"] = bs.get("status") == "PASS"
    if not state["broker_safety_pass"]:
        state["blockers"].append("broker safety not PASS")
    _add("broker_safety", bs_path)

    # ── Real trading gate audit ─────────────────────────────────────────────
    rtg_path = find_latest("real_trading_gate_audit_*.json")
    rtg = load_json(rtg_path)
    state["real_trading_gate_pass"] = rtg.get("all_gates_pass") is True or rtg.get("status") == "PASS"
    if not state["real_trading_gate_pass"]:
        state["blockers"].append("real trading gate not PASS")
    _add("real_trading_gate", rtg_path)

    # ── Risk control gap fix ────────────────────────────────────────────────
    rc_path = find_latest("risk_control_gap_fix_*.json")
    rc = load_json(rc_path)
    state["risk_control_gap_fix_pass"] = rc.get("status") == "PASS" or rc.get("all_tests_pass") is True
    if rc and not state["risk_control_gap_fix_pass"]:
        state["blockers"].append("risk control gaps not fixed")
    _add("risk_control_gap_fix", rc_path)

    # ── Model signal quality audit ──────────────────────────────────────────
    msq_path = find_latest("model_signal_quality_audit_*.json")
    msq = load_json(msq_path)
    verdict = str(msq.get("verdict", "UNKNOWN")).upper()
    state["model_signal_quality_verdict"] = verdict
    if verdict in {"BLOCKED", "RESEARCH_ONLY", "FAIL"}:
        state["blockers"].append(f"model signal quality verdict: {verdict}")
    _add("model_signal_quality", msq_path)

    # ── Latest dry-run ──────────────────────────────────────────────────────
    dr_path = find_latest("live_decision_dry_run_*.json")
    dr = load_json(dr_path)
    if dr:
        state["dry_run_coverage_pct"] = float(dr.get("coverage_pct", 0.0))
        state["latest_probability"] = float(dr.get("probability", 0.0))
        state["probability_threshold"] = float(dr.get("threshold", 0.5))
        state["model_pkl_exists"] = bool(dr.get("model_pkl"))
        _add("dry_run", dr_path)
    else:
        state["blockers"].append("no dry-run report")

    # ── Forward-test summary ────────────────────────────────────────────────
    ft_path = find_latest("paper_forward_test_summary_*.json")
    ft = load_json(ft_path)
    if ft:
        state["forward_test_decision_count"] = ft.get("total_decisions", 0)
        state["forward_test_trade_count"] = ft.get("total_trade", 0)
        state["forward_test_net_pnl"] = float(ft.get("net_pnl_total", 0.0))
        state["forward_test_enough_evidence"] = state["forward_test_trade_count"] >= 3
        _add("forward_test", ft_path)
    else:
        state["blockers"].append("no forward-test summary yet")

    # ── Stale report check ──────────────────────────────────────────────────
    for report_name, report_path_str in list(state["reports_used"].items()):
        if report_path_str and is_stale(Path(report_path_str)):
            state["blockers"].append(f"{report_name} report is stale (>48h)")

    # ── Real trading allowed ────────────────────────────────────────────────
    allowed = (
        state["paper_engine_pass"]
        and state["broker_safety_pass"]
        and state["real_trading_gate_pass"]
        and state["model_pkl_exists"]
        and state["dry_run_coverage_pct"] >= 95.0
        and state["latest_probability"] >= state["probability_threshold"]
        and state["risk_control_gap_fix_pass"]
        and state["model_signal_quality_verdict"] not in {"BLOCKED", "RESEARCH_ONLY", "FAIL"}
        and not bool(state["blockers"])
    )
    state["real_trading_allowed_now"] = allowed

    # ── Micro-live allowed ──────────────────────────────────────────────────
    micro_allowed = False
    if state["real_trading_allowed_now"]:
        micro_allowed = (
            state["forward_test_trade_count"] >= 3
            and state["forward_test_net_pnl"] > 0.0
            and float(ft.get("max_drawdown", 9999)) < 500.0
            and state["latest_probability"] >= state["probability_threshold"]
            and state["risk_control_gap_fix_pass"]
            and state["model_signal_quality_verdict"] not in {"BLOCKED", "RESEARCH_ONLY", "FAIL"}
        )
    state["micro_live_allowed_now"] = micro_allowed

    # ── Write JSON ──────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS / f"readiness_state_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    # ── Write Markdown ──────────────────────────────────────────────────────
    md_path = REPORTS / f"readiness_state_{ts}.md"
    md_lines = [
        "# Readiness State Report",
        f"**Generated:** {state['timestamp']}",
        "",
        "## Gate Status",
        f"- paper_engine_pass: {'PASS' if state['paper_engine_pass'] else 'FAIL'}",
        f"- broker_safety_pass: {'PASS' if state['broker_safety_pass'] else 'FAIL'}",
        f"- real_trading_gate_pass: {'PASS' if state['real_trading_gate_pass'] else 'FAIL'}",
        f"- risk_control_gap_fix_pass: {'PASS' if state['risk_control_gap_fix_pass'] else 'FAIL'}",
        f"- model_signal_quality_verdict: {state['model_signal_quality_verdict']}",
        "",
        "## Model / Dry-Run",
        f"- model_pkl_exists: {state['model_pkl_exists']}",
        f"- dry_run_coverage_pct: {state['dry_run_coverage_pct']:.1f}%",
        f"- latest_probability: {state['latest_probability']:.4f}",
        f"- probability_threshold: {state['probability_threshold']:.4f}",
        "",
        "## Forward Test",
        f"- forward_test_decision_count: {state['forward_test_decision_count']}",
        f"- forward_test_trade_count: {state['forward_test_trade_count']}",
        f"- forward_test_net_pnl: {state['forward_test_net_pnl']:.2f}",
        f"- forward_test_enough_evidence: {state['forward_test_enough_evidence']}",
        "",
        "## Trading Permissions",
        f"- real_trading_allowed_now: {state['real_trading_allowed_now']}",
        f"- micro_live_allowed_now: {state['micro_live_allowed_now']}",
        "",
        "## Blockers",
    ]
    if state["blockers"]:
        for b in state["blockers"]:
            md_lines.append(f"- {b}")
    else:
        md_lines.append("- none")

    md_lines.extend([
        "",
        "## Reports Used",
    ])
    for name, path in state["reports_used"].items():
        md_lines.append(f"- {name}: `{path}`")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(state, indent=2))
    print(f"\nReadiness state written to: {json_path}")
    print(f"  Markdown: {md_path}")
    print(f"Real trading allowed: {state['real_trading_allowed_now']}")
    print(f"Micro-live allowed: {state['micro_live_allowed_now']}")
    if state["blockers"]:
        print(f"Blockers: {state['blockers']}")

    return state


if __name__ == "__main__":
    main()
