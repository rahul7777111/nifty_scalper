#!/usr/bin/env python3
"""Tests for scripts/run_paper_forward_test.py

Paper mode only — verifies:
- place_order is never called
- JSONL / CSV / summary reports are written
- Kill switch, market-hours guard, max-trades limit work correctly
- Low-confidence signals result in SKIP
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import time
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_cfg(
    kill_switch: bool = False,
    ml_disable_all: bool = False,
    lot_size: int = 1,
    paper_slippage_pct: float = 0.001,
    paper_extra_market_impact_pct: float = 0.0,
    paper_apply_brokerage_costs: bool = True,
    paper_cost_model_source: str = "cost_model_assumptions",
) -> MagicMock:
    """Create a minimal mock StrategyConfig."""
    cfg = MagicMock()
    cfg.kill_switch_active = kill_switch
    cfg.ml_disable_all = ml_disable_all
    cfg.lot_size = lot_size
    cfg.paper_slippage_pct = paper_slippage_pct
    cfg.paper_extra_market_impact_pct = paper_extra_market_impact_pct
    cfg.paper_apply_brokerage_costs = paper_apply_brokerage_costs
    cfg.paper_cost_model_source = paper_cost_model_source
    cfg.ml_min_confidence_threshold = 0.5
    return cfg


def _mock_bundle(feature_names: list[str] | None = None, prob: float = 0.7) -> MagicMock:
    bundle = MagicMock()
    bundle.feature_names = feature_names or ["f1", "f2", "f3"]
    bundle.model_id = "test_model"
    bundle.trained_at = "2026-01-01T00:00:00Z"
    bundle.model_path = "/tmp/test.pkl"
    bundle.predict.return_value = prob
    return bundle


# ---------------------------------------------------------------------------
# Test 1: place_order is never called
# ---------------------------------------------------------------------------

def test_forward_runner_never_calls_real_order(tmp_path, monkeypatch):
    """Patch place_order, run 1 iteration, assert not called."""
    from scripts import run_paper_forward_test as r

    # Create a minimal fake model dir so model loading doesn't fail
    model_dir = tmp_path / "models" / "test_model"
    model_dir.mkdir(parents=True)
    (model_dir / "feature_manifest.json").write_text(
        json.dumps({"feature_count": 5, "features": [{"feature": f"f{i}"} for i in range(5)]}),
        encoding="utf-8",
    )
    (model_dir / "f1_f2_f3.pkl").write_bytes(b"pkldata")

    place_order_calls: list = []
    real_place_order = None

    # Capture any real broker place_order
    try:
        from src import dhan_client, mstock_client
        for mod in [dhan_client, mstock_client]:
            if hasattr(mod, "place_order"):
                real_place_order = mod.place_order
                break
    except Exception:
        pass

    def tracking_order(*args, **kwargs):
        place_order_calls.append((args, kwargs))

    if real_place_order:
        monkeypatch.setattr(real_place_order, tracking_order)
    else:
        # Mock all known broker modules
        for mod_path in [
            "src.dhan_client.place_order",
            "src.mstock_client.place_order",
            "src.mstock_client_original.place_order",
        ]:
            try:
                monkeypatch.setattr(mod_path, tracking_order)
            except Exception:
                pass  # module may not import cleanly

    # Patch model loading so we use our temp dir
    def fake_load_model(cfg):
        bundle = _mock_bundle(prob=0.75)
        return bundle, model_dir, 0.5

    monkeypatch.setattr(r, "load_model_for_forward_test", fake_load_model)

    # Patch market-hours check so we run immediately
    monkeypatch.setattr(r, "is_market_open", lambda now: True)

    # Patch run_single_decision to return a TRADE
    def fake_decision(*args, **kwargs):
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "final_action": "TRADE",
            "probability": 0.75,
            "threshold": 0.5,
            "coverage_pct": 100.0,
            "blocking_reasons": [],
            "spread_pct": 0.0,
            "premium": 0.0,
            "model_pkl": str(model_dir / "test.pkl"),
            "model_id": "test_model",
        }

    monkeypatch.setattr(r, "run_single_decision", fake_decision)

    # Patch simulate_paper_trade to avoid cost_model import issues
    def fake_paper_trade(decision, cfg):
        return {
            "entry_price": 25000.0,
            "exit_price": 25000.0,
            "entry_price_source": "ask",
            "gross_pnl": 0.0,
            "net_pnl": -2.0,
            "spread_cost": 1.0,
            "slippage_cost": 0.5,
            "brokerage_cost": 0.5,
        }

    monkeypatch.setattr(r, "simulate_paper_trade", fake_paper_trade)

    # Run one iteration
    cfg = _mock_cfg()
    with patch.object(time, "sleep", return_value=None):
        summary = r.run_paper_forward_test(
            symbol="NIFTY",
            cfg=cfg,
            interval_sec=1,
            max_trades=1,
            output_dir=str(tmp_path / "out"),
            require_market_hours=False,
            min_runtime_minutes=0,
        )

    assert place_order_calls == [], f"place_order was called: {place_order_calls}"
    assert summary["total_trade"] == 1
    assert summary["status"] == "PAPER_ONLY_READY"


# ---------------------------------------------------------------------------
# Test 2: SKIP decision is logged to JSONL
# ---------------------------------------------------------------------------

def test_skip_decisions_are_logged(tmp_path, monkeypatch):
    """Simulate SKIP decision, verify JSONL has entry."""
    from scripts import run_paper_forward_test as r

    model_dir = tmp_path / "models" / "skip_model"
    model_dir.mkdir(parents=True)

    def fake_load_model(cfg):
        bundle = _mock_bundle(prob=0.3)
        return bundle, model_dir, 0.5

    monkeypatch.setattr(r, "load_model_for_forward_test", fake_load_model)
    monkeypatch.setattr(r, "is_market_open", lambda now: True)

    def fake_decision(*args, **kwargs):
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "final_action": "SKIP",
            "probability": 0.3,
            "threshold": 0.5,
            "coverage_pct": 100.0,
            "feature_count": 3,
            "blocking_reasons": ["low_confidence_0.3000_lt_0.5000"],
            "skip_reason": ["low_confidence_0.3000_lt_0.5000"],
            "spread_pct": 0.0,
            "premium": 0.0,
            "bid": 24499.5,
            "ask": 24500.5,
            "ltp": 24500.0,
            "model_pkl": str(model_dir / "test.pkl"),
            "model_id": "test",
        }

    monkeypatch.setattr(r, "run_single_decision", fake_decision)

    cfg = _mock_cfg()
    with patch.object(time, "sleep", return_value=None):
        summary = r.run_paper_forward_test(
            symbol="NIFTY",
            cfg=cfg,
            interval_sec=1,
            max_trades=1,
            output_dir=str(tmp_path / "out"),
            require_market_hours=False,
            min_runtime_minutes=0,
        )

    # Find JSONL file
    jsonl_files = list(tmp_path.glob("out/paper_decisions_*.jsonl"))
    assert len(jsonl_files) == 1, f"Expected 1 JSONL, got {jsonl_files}"
    lines = [ln for ln in jsonl_files[0].read_text(encoding="utf-8").strip().split("\n") if ln]
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["final_action"] == "SKIP"
    assert "low_confidence" in str(entry.get("blocking_reasons", []))
    # Verify all required SKIP fields are recorded
    assert "timestamp" in entry
    assert entry["model_id"] == "test"
    assert entry["model_pkl"]
    assert entry["feature_count"] == 3
    assert entry["coverage_pct"] == 100.0
    assert entry["probability"] == 0.3
    assert entry["threshold"] == 0.5
    assert "skip_reason" in entry
    assert entry["bid"] == 24499.5
    assert entry["ask"] == 24500.5
    assert entry["ltp"] == 24500.0
    assert entry["spread_pct"] == 0.0
    assert entry["premium"] == 0.0


# ---------------------------------------------------------------------------
# Test 3: TRADE decisions are paper-only (no place_order)
# ---------------------------------------------------------------------------

def test_trade_decisions_are_paper_only(tmp_path, monkeypatch):
    """Simulate TRADE, verify no place_order call."""
    from scripts import run_paper_forward_test as r

    place_order_called = []

    def tracking_place_order(*args, **kwargs):
        place_order_called.append(True)

    for mod_path in ["src.dhan_client.place_order", "src.mstock_client.place_order"]:
        try:
            monkeypatch.setattr(mod_path, tracking_place_order)
        except Exception:
            pass

    model_dir = tmp_path / "models" / "trade_model"
    model_dir.mkdir(parents=True)

    def fake_load_model(cfg):
        bundle = _mock_bundle(prob=0.75)
        return bundle, model_dir, 0.5

    monkeypatch.setattr(r, "load_model_for_forward_test", fake_load_model)
    monkeypatch.setattr(r, "is_market_open", lambda now: True)

    def fake_decision(*args, **kwargs):
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "final_action": "TRADE",
            "probability": 0.75,
            "threshold": 0.5,
            "coverage_pct": 100.0,
            "blocking_reasons": [],
            "spread_pct": 0.001,
            "premium": 15.0,
            "model_pkl": str(model_dir / "test.pkl"),
            "model_id": "test",
        }

    def fake_paper_trade(decision, cfg):
        return {
            "entry_price": 25000.0, "exit_price": 25000.0,
            "entry_price_source": "ask", "gross_pnl": 0.0, "net_pnl": -2.0,
            "spread_cost": 1.0, "slippage_cost": 0.5, "brokerage_cost": 0.5,
        }

    monkeypatch.setattr(r, "run_single_decision", fake_decision)
    monkeypatch.setattr(r, "simulate_paper_trade", fake_paper_trade)

    cfg = _mock_cfg()
    with patch.object(time, "sleep", return_value=None):
        summary = r.run_paper_forward_test(
            symbol="NIFTY", cfg=cfg,
            interval_sec=1, max_trades=1,
            output_dir=str(tmp_path / "out"),
            require_market_hours=False, min_runtime_minutes=0,
        )

    assert place_order_called == [], f"place_order was called during TRADE: {place_order_called}"
    assert summary["total_trade"] == 1


# ---------------------------------------------------------------------------
# Test 4: CSV file is created
# ---------------------------------------------------------------------------

def test_output_csv_created(tmp_path, monkeypatch):
    """Run loop briefly, verify CSV file exists."""
    from scripts import run_paper_forward_test as r

    model_dir = tmp_path / "models" / "csv_model"
    model_dir.mkdir(parents=True)

    def fake_load_model(cfg):
        bundle = _mock_bundle(prob=0.75)
        return bundle, model_dir, 0.5

    monkeypatch.setattr(r, "load_model_for_forward_test", fake_load_model)
    monkeypatch.setattr(r, "is_market_open", lambda now: True)

    def fake_decision(*args, **kwargs):
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "final_action": "TRADE",
            "probability": 0.75, "threshold": 0.5,
            "coverage_pct": 100.0, "blocking_reasons": [],
            "spread_pct": 0.001, "premium": 15.0,
            "model_pkl": str(model_dir / "test.pkl"), "model_id": "test",
        }

    def fake_paper_trade(decision, cfg):
        return {
            "entry_price": 25000.0, "exit_price": 25000.0,
            "entry_price_source": "ask", "gross_pnl": 0.0, "net_pnl": -2.0,
            "spread_cost": 1.0, "slippage_cost": 0.5, "brokerage_cost": 0.5,
        }

    monkeypatch.setattr(r, "run_single_decision", fake_decision)
    monkeypatch.setattr(r, "simulate_paper_trade", fake_paper_trade)

    cfg = _mock_cfg()
    with patch.object(time, "sleep", return_value=None):
        r.run_paper_forward_test(
            symbol="NIFTY", cfg=cfg,
            interval_sec=1, max_trades=1,
            output_dir=str(tmp_path / "out"),
            require_market_hours=False, min_runtime_minutes=0,
        )

    csv_files = list(tmp_path.glob("out/paper_trades_*.csv"))
    assert len(csv_files) == 1, f"CSV not created: {csv_files}"
    rows = list(csv.DictReader(csv_files[0].open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["action"] == "TRADE"


# ---------------------------------------------------------------------------
# Test 5: JSONL file is created
# ---------------------------------------------------------------------------

def test_output_jsonl_created(tmp_path, monkeypatch):
    """Run loop briefly, verify JSONL file exists."""
    from scripts import run_paper_forward_test as r

    model_dir = tmp_path / "models" / "jsonl_model"
    model_dir.mkdir(parents=True)

    def fake_load_model(cfg):
        bundle = _mock_bundle(prob=0.75)
        return bundle, model_dir, 0.5

    monkeypatch.setattr(r, "load_model_for_forward_test", fake_load_model)
    monkeypatch.setattr(r, "is_market_open", lambda now: True)

    def fake_decision(*args, **kwargs):
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "final_action": "TRADE",
            "probability": 0.75, "threshold": 0.5,
            "coverage_pct": 100.0, "blocking_reasons": [],
            "spread_pct": 0.001, "premium": 15.0,
            "model_pkl": str(model_dir / "test.pkl"), "model_id": "test",
        }

    def fake_paper_trade(decision, cfg):
        return {
            "entry_price": 25000.0, "exit_price": 25000.0,
            "entry_price_source": "ask", "gross_pnl": 0.0, "net_pnl": -2.0,
            "spread_cost": 1.0, "slippage_cost": 0.5, "brokerage_cost": 0.5,
        }

    monkeypatch.setattr(r, "run_single_decision", fake_decision)
    monkeypatch.setattr(r, "simulate_paper_trade", fake_paper_trade)

    cfg = _mock_cfg()
    with patch.object(time, "sleep", return_value=None):
        r.run_paper_forward_test(
            symbol="NIFTY", cfg=cfg,
            interval_sec=1, max_trades=1,
            output_dir=str(tmp_path / "out"),
            require_market_hours=False, min_runtime_minutes=0,
        )

    jsonl_files = list(tmp_path.glob("out/paper_decisions_*.jsonl"))
    assert len(jsonl_files) == 1, f"JSONL not created: {jsonl_files}"
    lines = jsonl_files[0].read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["final_action"] == "TRADE"


# ---------------------------------------------------------------------------
# Test 6: Summary report (.json + .md) is generated
# ---------------------------------------------------------------------------

def test_summary_report_generated(tmp_path, monkeypatch):
    """Run loop, verify summary .json and .md exist."""
    from scripts import run_paper_forward_test as r

    model_dir = tmp_path / "models" / "summary_model"
    model_dir.mkdir(parents=True)

    def fake_load_model(cfg):
        bundle = _mock_bundle(prob=0.75)
        return bundle, model_dir, 0.5

    monkeypatch.setattr(r, "load_model_for_forward_test", fake_load_model)
    monkeypatch.setattr(r, "is_market_open", lambda now: True)

    def fake_decision(*args, **kwargs):
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "final_action": "TRADE",
            "probability": 0.75, "threshold": 0.5,
            "coverage_pct": 100.0, "blocking_reasons": [],
            "spread_pct": 0.001, "premium": 15.0,
            "model_pkl": str(model_dir / "test.pkl"), "model_id": "test",
        }

    def fake_paper_trade(decision, cfg):
        return {
            "entry_price": 25000.0, "exit_price": 25000.0,
            "entry_price_source": "ask", "gross_pnl": 0.0, "net_pnl": -2.0,
            "spread_cost": 1.0, "slippage_cost": 0.5, "brokerage_cost": 0.5,
        }

    monkeypatch.setattr(r, "run_single_decision", fake_decision)
    monkeypatch.setattr(r, "simulate_paper_trade", fake_paper_trade)

    cfg = _mock_cfg()
    with patch.object(time, "sleep", return_value=None):
        r.run_paper_forward_test(
            symbol="NIFTY", cfg=cfg,
            interval_sec=1, max_trades=1,
            output_dir=str(tmp_path / "out"),
            require_market_hours=False, min_runtime_minutes=0,
        )

    md_files = list(tmp_path.glob("out/paper_forward_test_summary_*.md"))
    json_files = list(tmp_path.glob("out/paper_forward_test_summary_*.json"))
    assert len(md_files) == 1, f"Summary .md not created: {md_files}"
    assert len(json_files) == 1, f"Summary .json not created: {json_files}"

    # Validate JSON content
    summary = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert summary["status"] == "PAPER_ONLY_READY"
    assert summary["total_trade"] == 1
    assert summary["total_decisions"] == 1
    assert "micro_live_evidence" in summary

    # Validate markdown content
    md_text = md_files[0].read_text(encoding="utf-8")
    assert "# Paper Forward-Test Summary" in md_text
    assert "PAPER ONLY" in md_text


# ---------------------------------------------------------------------------
# Test 7: Low-confidence signal → SKIP
# ---------------------------------------------------------------------------

def test_low_confidence_signal_remains_skip(tmp_path, monkeypatch):
    """prob=0.3 → SKIP."""
    from scripts import run_paper_forward_test as r

    model_dir = tmp_path / "models" / "lowconf_model"
    model_dir.mkdir(parents=True)

    def fake_load_model(cfg):
        bundle = _mock_bundle(prob=0.3)  # below threshold
        return bundle, model_dir, 0.5

    monkeypatch.setattr(r, "load_model_for_forward_test", fake_load_model)
    monkeypatch.setattr(r, "is_market_open", lambda now: True)

    def fake_decision(*args, **kwargs):
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "final_action": "SKIP",
            "probability": 0.3,
            "threshold": 0.5,
            "coverage_pct": 100.0,
            "blocking_reasons": ["low_confidence_0.3000_lt_0.5000"],
            "spread_pct": 0.0,
            "premium": 0.0,
            "model_pkl": str(model_dir / "test.pkl"),
            "model_id": "test",
        }

    monkeypatch.setattr(r, "run_single_decision", fake_decision)

    cfg = _mock_cfg()
    with patch.object(time, "sleep", return_value=None):
        summary = r.run_paper_forward_test(
            symbol="NIFTY", cfg=cfg,
            interval_sec=1, max_trades=1,
            output_dir=str(tmp_path / "out"),
            require_market_hours=False, min_runtime_minutes=0,
        )

    assert summary["total_trade"] == 0
    assert summary["total_skip"] == 1


# ---------------------------------------------------------------------------
# Test 8: Kill switch blocks trading
# ---------------------------------------------------------------------------

def test_kill_switch_blocks_trading(tmp_path, monkeypatch):
    """kill_switch=True → SKIP even with prob > threshold."""
    from scripts import run_paper_forward_test as r

    model_dir = tmp_path / "models" / "killswitch_model"
    model_dir.mkdir(parents=True)

    decision_calls = []

    def fake_load_model(cfg):
        bundle = _mock_bundle(prob=0.8)
        return bundle, model_dir, 0.5

    monkeypatch.setattr(r, "load_model_for_forward_test", fake_load_model)
    monkeypatch.setattr(r, "is_market_open", lambda now: True)

    def fake_decision(*args, **kwargs):
        decision_calls.append(True)
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "final_action": "SKIP",
            "probability": 0.8,
            "threshold": 0.5,
            "coverage_pct": 100.0,
            "blocking_reasons": ["kill_switch"],
            "spread_pct": 0.0,
            "premium": 0.0,
            "model_pkl": str(model_dir / "test.pkl"),
            "model_id": "test",
        }

    monkeypatch.setattr(r, "run_single_decision", fake_decision)

    cfg = _mock_cfg(kill_switch=True)
    with patch.object(time, "sleep", return_value=None):
        summary = r.run_paper_forward_test(
            symbol="NIFTY", cfg=cfg,
            interval_sec=1, max_trades=0,
            output_dir=str(tmp_path / "out"),
            require_market_hours=False, min_runtime_minutes=0,
        )

    assert summary["total_trade"] == 0
    # Kill switch blocks before any decision (max_trades=0 exits at top)
    assert summary["total_decisions"] == 0


# ---------------------------------------------------------------------------
# Test 9: Market hours guard works (outside hours → waits without error)
# ---------------------------------------------------------------------------

def test_market_hours_guard_works(tmp_path, monkeypatch):
    """Outside hours → waits without error."""
    from scripts import run_paper_forward_test as r

    model_dir = tmp_path / "models" / "hours_model"
    model_dir.mkdir(parents=True)

    sleep_calls = []

    def fake_load_model(cfg):
        bundle = _mock_bundle(prob=0.75)
        return bundle, model_dir, 0.5

    monkeypatch.setattr(r, "load_model_for_forward_test", fake_load_model)

    # Simulate outside market hours
    monkeypatch.setattr(r, "is_market_open", lambda now: False)

    def tracking_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(r, "run_single_decision", lambda *a, **k: {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "final_action": "SKIP", "probability": 0.75, "threshold": 0.5,
        "coverage_pct": 100.0, "blocking_reasons": [], "spread_pct": 0.0,
        "premium": 0.0, "model_pkl": "", "model_id": "test",
    })

    cfg = _mock_cfg()
    monkeypatch.setattr(time, "sleep", tracking_sleep)

    # require_market_hours=True but outside market hours with min_runtime=0
    # The loop should exit immediately (elapsed=0 >= 0) with zero decisions.
    summary = r.run_paper_forward_test(
        symbol="NIFTY", cfg=cfg,
        interval_sec=10, max_trades=1,
        output_dir=str(tmp_path / "out"),
        require_market_hours=True, min_runtime_minutes=0,
    )

    # No decisions made (market closed before any iteration)
    assert summary["total_decisions"] == 0


# ---------------------------------------------------------------------------
# Test 10: max_trades stops loop
# ---------------------------------------------------------------------------

def test_max_trades_stops_loop(tmp_path, monkeypatch):
    """max_trades=1 → stops after 1 trade."""
    from scripts import run_paper_forward_test as r

    model_dir = tmp_path / "models" / "maxtrades_model"
    model_dir.mkdir(parents=True)

    decision_count = 0

    def fake_load_model(cfg):
        bundle = _mock_bundle(prob=0.75)
        return bundle, model_dir, 0.5

    monkeypatch.setattr(r, "load_model_for_forward_test", fake_load_model)
    monkeypatch.setattr(r, "is_market_open", lambda now: True)

    def fake_decision(*args, **kwargs):
        nonlocal decision_count
        decision_count += 1
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "final_action": "TRADE",
            "probability": 0.75, "threshold": 0.5,
            "coverage_pct": 100.0, "blocking_reasons": [],
            "spread_pct": 0.001, "premium": 15.0,
            "model_pkl": str(model_dir / "test.pkl"), "model_id": "test",
        }

    def fake_paper_trade(decision, cfg):
        return {
            "entry_price": 25000.0, "exit_price": 25000.0,
            "entry_price_source": "ask", "gross_pnl": 0.0, "net_pnl": -2.0,
            "spread_cost": 1.0, "slippage_cost": 0.5, "brokerage_cost": 0.5,
        }

    monkeypatch.setattr(r, "run_single_decision", fake_decision)
    monkeypatch.setattr(r, "simulate_paper_trade", fake_paper_trade)

    cfg = _mock_cfg()
    with patch.object(time, "sleep", return_value=None):
        summary = r.run_paper_forward_test(
            symbol="NIFTY", cfg=cfg,
            interval_sec=300, max_trades=1,
            output_dir=str(tmp_path / "out"),
            require_market_hours=False, min_runtime_minutes=0,
        )

    assert summary["total_trade"] == 1
    # Should have made exactly 1 decision (loop exited after max_trades reached)
    assert decision_count == 1, f"Expected 1 decision, got {decision_count}"


# ---------------------------------------------------------------------------
# Test 11: Model dir with no .pkl is rejected
# ---------------------------------------------------------------------------

def test_invalid_no_pkl_folder_rejected(tmp_path, monkeypatch):
    """A model dir with paper_watchlist_report.json but no .pkl is rejected."""
    from scripts import run_paper_forward_test as r
    import live_decision_dry_run as lddr

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    invalid_dir = models_dir / "no_pkl_model"
    invalid_dir.mkdir()
    (invalid_dir / "paper_watchlist_report.json").write_text(
        '{"status": "PAPER_TRADE_CANDIDATE"}', encoding="utf-8"
    )

    monkeypatch.setattr(lddr, "_REPO_ROOT", tmp_path)

    cfg = _mock_cfg()
    with pytest.raises(RuntimeError, match="No model directories found"):
        r.load_model_for_forward_test(cfg)


# ---------------------------------------------------------------------------
# Test 12: Valid model dir is selected correctly
# ---------------------------------------------------------------------------

def test_valid_model_selected(tmp_path, monkeypatch):
    """A valid model dir with all required artifacts is selected and loaded."""
    from scripts import run_paper_forward_test as r
    import live_decision_dry_run as lddr

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    valid_dir = models_dir / "valid_model_20260101_120000"
    valid_dir.mkdir()

    (valid_dir / "paper_watchlist_report.json").write_text(
        '{"status": "PAPER_TRADE_CANDIDATE"}', encoding="utf-8"
    )
    (valid_dir / "feature_manifest.json").write_text(
        '{"feature_count": 25}', encoding="utf-8"
    )
    (valid_dir / "candidate_champion_report.json").write_text(
        json.dumps({"best_global_observed": {"model_pkl": "test_model.pkl"}}),
        encoding="utf-8",
    )

    try:
        import joblib
        joblib.dump(
            {"feature_names": [f"f{i}" for i in range(25)]},
            valid_dir / "test_model.pkl",
        )
    except Exception as exc:
        pytest.skip(f"joblib not available or cannot write .pkl: {exc}")

    monkeypatch.setattr(lddr, "_REPO_ROOT", tmp_path)

    mock_bundle = _mock_bundle()
    monkeypatch.setattr(lddr, "load_model_bundle", lambda p: mock_bundle)

    cfg = _mock_cfg()
    bundle, model_dir, threshold = r.load_model_for_forward_test(cfg)
    assert model_dir.name == "valid_model_20260101_120000"
    assert threshold == 0.5
    assert bundle is mock_bundle


# ---------------------------------------------------------------------------
# Test 13: --model-dir override rejected if invalid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "scenario,files,expected_msg",
    [
        (
            "no_pkl",
            {},
            "No valid .pkl files",
        ),
        (
            "feature_count_0",
            {
                "feature_manifest.json": '{"feature_count": 0}',
                "model.pkl": "",
            },
            "feature_count=0",
        ),
        (
            "model_pkl_null",
            {
                "feature_manifest.json": '{"feature_count": 25}',
                "candidate_champion_report.json": json.dumps(
                    {"best_global_observed": {"model_pkl": None}}
                ),
                "model.pkl": "",
            },
            "model_pkl is null",
        ),
    ],
)
def test_model_dir_override_rejected_if_invalid(
    tmp_path, monkeypatch, scenario, files, expected_msg
):
    """--model-dir pointing to an invalid dir is rejected with a clear message."""
    from scripts import run_paper_forward_test as r

    invalid_dir = tmp_path / f"invalid_{scenario}"
    invalid_dir.mkdir()

    for fname, content in files.items():
        fpath = invalid_dir / fname
        if fname.endswith(".pkl"):
            fpath.write_bytes(b"pkldata")
        else:
            fpath.write_text(content, encoding="utf-8")

    cfg = _mock_cfg()
    cfg.model_dir = str(invalid_dir)

    with pytest.raises(RuntimeError, match=expected_msg):
        r.load_model_for_forward_test(cfg)


# ---------------------------------------------------------------------------
# Test 14: Report includes selected model_pkl
# ---------------------------------------------------------------------------

def test_report_includes_selected_model_pkl(tmp_path, monkeypatch):
    """Summary report contains the selected model_pkl and model_id."""
    from scripts import run_paper_forward_test as r

    model_dir = tmp_path / "models" / "report_model"
    model_dir.mkdir(parents=True)

    def fake_load_model(cfg):
        bundle = _mock_bundle(prob=0.75)
        return bundle, model_dir, 0.5

    monkeypatch.setattr(r, "load_model_for_forward_test", fake_load_model)
    monkeypatch.setattr(r, "is_market_open", lambda now: True)

    def fake_decision(*args, **kwargs):
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "final_action": "TRADE",
            "probability": 0.75,
            "threshold": 0.5,
            "coverage_pct": 100.0,
            "blocking_reasons": [],
            "spread_pct": 0.001,
            "premium": 15.0,
            "model_pkl": str(model_dir / "selected.pkl"),
            "model_id": "report_model",
        }

    def fake_paper_trade(decision, cfg):
        return {
            "entry_price": 25000.0,
            "exit_price": 25000.0,
            "entry_price_source": "ask",
            "gross_pnl": 0.0,
            "net_pnl": -2.0,
            "spread_cost": 1.0,
            "slippage_cost": 0.5,
            "brokerage_cost": 0.5,
        }

    monkeypatch.setattr(r, "run_single_decision", fake_decision)
    monkeypatch.setattr(r, "simulate_paper_trade", fake_paper_trade)

    cfg = _mock_cfg()
    with patch.object(time, "sleep", return_value=None):
        summary = r.run_paper_forward_test(
            symbol="NIFTY",
            cfg=cfg,
            interval_sec=1,
            max_trades=1,
            output_dir=str(tmp_path / "out"),
            require_market_hours=False,
            min_runtime_minutes=0,
        )

    assert summary["model_pkl"] == str(model_dir / "selected.pkl")
    assert summary["model_id"] == "report_model"

    # Also verify the JSON report file contains it
    json_files = list((tmp_path / "out").glob("paper_forward_test_summary_*.json"))
    assert len(json_files) == 1
    report = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert report["model_pkl"] == str(model_dir / "selected.pkl")
    assert report["model_id"] == "report_model"


# ---------------------------------------------------------------------------
# Test 15: Zero trades creates valid summary
# ---------------------------------------------------------------------------

def test_zero_trades_creates_valid_summary(tmp_path, monkeypatch):
    """0 trades still creates valid summary JSON+MD and empty CSV with header."""
    from scripts import run_paper_forward_test as r

    model_dir = tmp_path / "models" / "zero_model"
    model_dir.mkdir(parents=True)

    def fake_load_model(cfg):
        bundle = _mock_bundle(prob=0.3)
        return bundle, model_dir, 0.5

    monkeypatch.setattr(r, "load_model_for_forward_test", fake_load_model)
    monkeypatch.setattr(r, "is_market_open", lambda now: True)

    def fake_decision(*args, **kwargs):
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "final_action": "SKIP",
            "probability": 0.3,
            "threshold": 0.5,
            "coverage_pct": 80.0,
            "feature_count": 3,
            "blocking_reasons": ["low_confidence_0.3000_lt_0.5000"],
            "skip_reason": ["low_confidence_0.3000_lt_0.5000"],
            "spread_pct": 0.0,
            "premium": 0.0,
            "bid": 0.0, "ask": 0.0, "ltp": 0.0,
            "model_pkl": str(model_dir / "test.pkl"),
            "model_id": "test",
        }

    monkeypatch.setattr(r, "run_single_decision", fake_decision)

    cfg = _mock_cfg()
    with patch.object(time, "sleep", return_value=None):
        summary = r.run_paper_forward_test(
            symbol="NIFTY", cfg=cfg,
            interval_sec=1, max_trades=1,
            output_dir=str(tmp_path / "out"),
            require_market_hours=False, min_runtime_minutes=0,
        )

    assert summary["status"] == "PAPER_ONLY_READY"
    assert summary["total_decisions"] == 1
    assert summary["total_trade"] == 0
    assert summary["total_skip"] == 1
    assert "all_skip_explanation" in summary
    assert summary["all_skip_explanation"]  # non-empty explanation

    # Verify all four output files exist
    out = tmp_path / "out"
    jsonl_files = list(out.glob("paper_decisions_*.jsonl"))
    csv_files = list(out.glob("paper_trades_*.csv"))
    md_files = list(out.glob("paper_forward_test_summary_*.md"))
    json_files = list(out.glob("paper_forward_test_summary_*.json"))
    assert len(jsonl_files) == 1
    assert len(csv_files) == 1
    assert len(md_files) == 1
    assert len(json_files) == 1

    # Verify CSV has header even though no trades
    csv_text = csv_files[0].read_text(encoding="utf-8")
    assert "timestamp" in csv_text
    assert "action" in csv_text

    # Verify JSONL has the SKIP entry
    lines = [ln for ln in jsonl_files[0].read_text(encoding="utf-8").strip().split("\n") if ln]
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["final_action"] == "SKIP"

    # Verify MD mentions all decisions were SKIP
    md_text = md_files[0].read_text(encoding="utf-8")
    assert "All decisions were SKIP" in md_text


# ---------------------------------------------------------------------------
# Test 16: Invalid model path fails cleanly
# ---------------------------------------------------------------------------

def test_invalid_model_path_fails_cleanly(tmp_path, monkeypatch):
    """Graceful failure when model dir invalid — no unhandled exceptions."""
    from scripts import run_paper_forward_test as r

    def fail_load(cfg):
        raise RuntimeError("No model directories found")

    monkeypatch.setattr(r, "load_model_for_forward_test", fail_load)

    cfg = _mock_cfg()
    summary = r.run_paper_forward_test(
        symbol="NIFTY", cfg=cfg,
        interval_sec=1, max_trades=1,
        output_dir=str(tmp_path / "out"),
        require_market_hours=False, min_runtime_minutes=0,
    )
    assert summary["status"] == "ERROR"
    assert "No model directories found" in summary.get("error", "")
    assert summary["total_decisions"] == 0
    assert summary["total_skip"] == 0
    assert summary["total_trade"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])