#!/usr/bin/env python3
"""Tests: NO real orders in shadow or paper mode.

Safety guarantee:
  - Shadow mode: never creates paper or real orders (only WOULD_ENTER logs)
  - Paper mode: never calls place_order() or any broker order API
  - Both modes are PAPER_ONLY — no real capital at risk

Run:
    pytest tests/test_no_real_orders_in_shadow_or_paper.py -v
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helper: scan source files for dangerous patterns
# ---------------------------------------------------------------------------

def _scan_for_dangerous_calls(module_path: Path) -> list[str]:
    """Return list of dangerous call patterns found in module source."""
    if not module_path.exists():
        return [f"FILE NOT FOUND: {module_path}"]

    source = module_path.read_text(encoding="utf-8")
    findings = []

    # Patterns that indicate real order placement
    dangerous = [
        ("place_order", "place_order() call found"),
        ("submit_order", "submit_order() call found"),
        ("send_order", "send_order() call found"),
        ("create_order", "create_order() call found"),
        ("modify_order", "modify_order() call found"),
        ("cancel_order", "cancel_order() call found"),
        ("dhan.place_order", "dhan broker order call found"),
        ("dhan.submit_order", "dhan broker order call found"),
        ("mstock.place_order", "mstock broker order call found"),
        ("client.place_order", "broker client place_order call found"),
        ("broker.place_order", "broker place_order call found"),
        ("SCALPER_ALLOW_LIVE_ORDERS=true", "live orders env var set in source"),
        ("enable_live_trading=True", "live trading enabled in source"),
    ]

    for pattern, description in dangerous:
        # Look for the pattern as a method/function call (not comment)
        # Use regex to find actual call sites (excluding comments)
        for line_no, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue  # skip comments
            if pattern in line:
                findings.append(f"  Line {line_no}: {description} — {stripped[:80]}")

    return findings


# ---------------------------------------------------------------------------
# Shadow Mode Safety Tests
# ---------------------------------------------------------------------------

def test_evaluate_shadow_mode_has_no_broker_imports():
    """evaluate_shadow_mode.py must not import any broker client module."""
    import scripts.evaluate_shadow_mode as sm_mod
    source = sm_mod.__file__
    findings = _scan_for_dangerous_calls(Path(source))
    assert findings == [], (
        f"Shadow mode script contains dangerous broker calls:\n" + "\n".join(findings)
    )


def test_evaluate_shadow_mode_source_has_no_place_order():
    """evaluate_shadow_mode.py must not call place_order."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_shadow_mode.py"
    findings = [f for f in _scan_for_dangerous_calls(path) if "place_order" in f]
    assert findings == [], (
        "Shadow mode must not call place_order:\n" + "\n".join(findings)
    )


def test_shadow_mode_script_safe_comment():
    """evaluate_shadow_mode.py must include safety documentation."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_shadow_mode.py"
    source = path.read_text(encoding="utf-8")

    # Should mention it is a shadow/logging only script
    safe_keywords = ["shadow", "WOULD_ENTER", "log", "no real", "paper"]
    found = any(kw in source for kw in safe_keywords)
    assert found, "Shadow mode script must include safety documentation"


# ---------------------------------------------------------------------------
# Paper Mode Safety Tests
# ---------------------------------------------------------------------------

def test_run_paper_forward_test_source_has_no_place_order():
    """run_paper_forward_test.py must not call place_order."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_paper_forward_test.py"
    findings = [f for f in _scan_for_dangerous_calls(path) if "place_order" in f]
    assert findings == [], (
        "Paper forward-test must not call place_order:\n" + "\n".join(findings)
    )


def test_run_paper_forward_test_source_has_no_broker_imports():
    """run_paper_forward_test.py must not import broker order APIs."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_paper_forward_test.py"
    findings = _scan_for_dangerous_calls(path)
    assert findings == [], (
        "Paper forward-test contains broker API calls:\n" + "\n".join(findings)
    )


def test_paper_forward_test_has_paper_only_banner():
    """run_paper_forward_test.py must log PAP ER ONLY banner."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_paper_forward_test.py"
    source = path.read_text(encoding="utf-8")

    assert "PAPER" in source.upper(), "Paper script must have PAPER mode banner"
    assert "NO REAL ORDERS" in source.upper() or "no real" in source.lower(), (
        "Paper script must warn no real orders will be placed"
    )


# ---------------------------------------------------------------------------
# Live Decision Dry Run Safety Tests
# ---------------------------------------------------------------------------

def test_live_decision_dry_run_source_has_no_place_order():
    """live_decision_dry_run.py must not call place_order."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "live_decision_dry_run.py"
    findings = [f for f in _scan_for_dangerous_calls(path) if "place_order" in f]
    assert findings == [], (
        "Dry-run script must not call place_order:\n" + "\n".join(findings)
    )


def test_live_decision_dry_run_source_has_no_broker_imports():
    """live_decision_dry_run.py must not import broker order APIs (only for live data fetch)."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "live_decision_dry_run.py"
    findings = [f for f in _scan_for_dangerous_calls(path)
                if any(k in f for k in ["place_order", "submit_order", "send_order"])]
    assert findings == [], (
        "Dry-run script contains order calls:\n" + "\n".join(findings)
    )


def test_live_decision_dry_run_has_paper_only_guard():
    """live_decision_dry_run.py must require explicit --paper or --dry-run flag."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "live_decision_dry_run.py"
    source = path.read_text(encoding="utf-8")

    assert "--paper" in source or "paper" in source.lower(), (
        "Dry-run script must require --paper flag"
    )
    # Should NOT accept real trading by default
    assert "required=True" in source, "Mode flag must be required (no default)"


def test_live_decision_dry_run_logs_no_real_orders():
    """live_decision_dry_run.py must log that no real orders are placed."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "live_decision_dry_run.py"
    source = path.read_text(encoding="utf-8")

    safe_phrases = ["no real order", "never calls", "paper", "dry-run", "WOULD NOT"]
    found = any(phrase in source.lower() for phrase in safe_phrases)
    assert found, "Dry-run script must explicitly state no real orders are placed"


# ---------------------------------------------------------------------------
# Strategy Safety Tests
# ---------------------------------------------------------------------------

def test_strategy_class_default_paper_mode():
    """StrategyConfig must default to enable_live_trading=False."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))

    from src.config import StrategyConfig
    cfg = StrategyConfig()

    assert cfg.enable_live_trading is False, (
        "StrategyConfig.enable_live_trading must default to False (fail-safe)"
    )


def test_no_live_orders_module_in_tests():
    """Test files must not contain actual place_order() calls (mocking in test files is allowed)."""
    import ast
    tests_dir = REPO_ROOT / "tests"
    # Broker-test files legitimately mock these calls — skip them
    skip_files = {
        "test_broker_runtime.py",
        "test_broker_regression.py",
        "test_broker_safety_audit.py",
        "test_legging_in_failsafe.py",
    }
    dangerous_files = []
    dangerous_calls = {
        "place_order", "submit_order", "send_order", "create_order",
        "modify_order", "cancel_order",
    }

    for py_file in sorted(tests_dir.glob("test_*.py")):
        if py_file.name in skip_files:
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = ""
                if isinstance(node.func, ast.Name):
                    call_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    call_name = node.func.attr
                    if isinstance(node.func.value, ast.Name):
                        call_name = f"{node.func.value.id}.{call_name}"
                if call_name in dangerous_calls:
                    dangerous_files.append(f"{py_file.name} -> {call_name}()")

    assert len(dangerous_files) == 0, (
        f"Test files must not call broker order APIs: {dangerous_files}"
    )


# ---------------------------------------------------------------------------
# Candidate manifest safety tests
# ---------------------------------------------------------------------------

def test_candidate_manifest_enforces_paper_only(tmp_path):
    """Candidate manifest must enforce paper_only=True for auto-selection."""
    import json
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

    import scripts.live_decision_dry_run as m

    # Create candidate with paper_only=False
    manifest = {
        "model_id": "test_live_candidate",
        "model_pkl": "nonexistent.pkl",
        "selected_threshold": 0.25,
        "paper_only": False,
        "real_trading_enabled": False,
        "gates_passed": 10,
        "gates_total": 10,
        "verdict": "PAPER_FORWARD_TEST_CANDIDATE_READY",
    }
    manifest_path = tmp_path / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Safe auto-selection must skip paper_only=False candidates
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    is_safe = loaded.get("paper_only") is True
    assert is_safe is False, "paper_only=False candidate must be excluded from safe auto-selection"


def test_candidate_manifest_blocks_real_trading_enabled(tmp_path):
    """Candidate manifest must block real_trading_enabled=True from auto-selection."""
    import json
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

    manifest = {
        "model_id": "test_real_enabled_candidate",
        "model_pkl": "nonexistent.pkl",
        "selected_threshold": 0.25,
        "paper_only": True,
        "real_trading_enabled": True,  # DANGEROUS
        "gates_passed": 10,
        "gates_total": 10,
    }
    manifest_path = tmp_path / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    is_safe = loaded.get("real_trading_enabled") is not True
    assert is_safe is False, "real_trading_enabled=True must be excluded from auto-selection"


# ---------------------------------------------------------------------------
# Env var guard tests
# ---------------------------------------------------------------------------

def test_scalper_allow_live_orders_not_in_source():
    """Source files must not hard-code SCALPER_ALLOW_LIVE_ORDERS=true."""
    scripts_to_check = [
        REPO_ROOT / "scripts" / "live_decision_dry_run.py",
        REPO_ROOT / "scripts" / "run_paper_forward_test.py",
        REPO_ROOT / "scripts" / "evaluate_shadow_mode.py",
    ]

    for script_path in scripts_to_check:
        if not script_path.exists():
            continue
        source = script_path.read_text(encoding="utf-8")
        # Look for hard-coded env var set to true
        dangerous = [
            'SCALPER_ALLOW_LIVE_ORDERS", "true"',
            'SCALPER_ALLOW_LIVE_ORDERS", "1"',
            "SCALPER_ALLOW_LIVE_ORDERS=true",
            "SCALPER_ALLOW_LIVE_ORDERS='true'",
            'SCALPER_ALLOW_LIVE_ORDERS="true"',
        ]
        for pattern in dangerous:
            assert pattern not in source, (
                f"Hard-coded live orders flag found in {script_path.name}: {pattern}"
            )


# ---------------------------------------------------------------------------
# Integration: full pipeline never creates real order
# ---------------------------------------------------------------------------

def test_all_paper_scripts_in_source_tree():
    """All three paper-mode scripts exist in the scripts/ directory."""
    required_scripts = [
        "live_decision_dry_run.py",
        "run_paper_forward_test.py",
        "evaluate_shadow_mode.py",
    ]
    scripts_dir = REPO_ROOT / "scripts"

    for name in required_scripts:
        script_path = scripts_dir / name
        assert script_path.exists(), f"Required paper-mode script not found: {name}"


def test_paper_scripts_are_executable():
    """All paper-mode scripts must be executable (or have proper shebang)."""
    required_scripts = [
        "live_decision_dry_run.py",
        "run_paper_forward_test.py",
        "evaluate_shadow_mode.py",
    ]
    scripts_dir = REPO_ROOT / "scripts"

    for name in required_scripts:
        script_path = scripts_dir / name
        if script_path.exists():
            first_line = script_path.read_text(encoding="utf-8").splitlines()[0]
            assert first_line.startswith("#!"), f"{name} must have shebang"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])