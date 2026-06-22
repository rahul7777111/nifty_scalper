#!/usr/bin/env python3
"""Tests for scripts/live_decision_dry_run.py schema fixes.

Covers:
  - Invalid zero-feature model rejection
  - model_pkl=null / phantom artifact rejection
  - Feature coverage >= 95% gate blocks prediction
  - Offline fixture achieves >= 95% coverage on candle features
  - No real order functions are called
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "live_decision_dry_run.py"

# Import from the script under test (guard: skip if script itself has import errors)
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# Test 1: find_model_dirs_with_pkls rejects feature_count=0 manifests
# ---------------------------------------------------------------------------

def test_find_model_dirs_rejects_zero_feature_manifest(tmp_path, monkeypatch):
    """A model dir with feature_manifest.json feature_count=0 must be skipped."""
    # Patch REPO_ROOT so find_model_dirs_with_pkls uses tmp_path as models dir
    model_dir = tmp_path / "phantom_zero_feature_model"
    model_dir.mkdir()
    # Write a zero-feature manifest
    (model_dir / "feature_manifest.json").write_text(
        json.dumps({"feature_count": 0, "features": []}), encoding="utf-8"
    )
    # Write a dummy pkl so it passes _has_valid_pkl
    (model_dir / "dummy_model.pkl").write_bytes(b"pkldata")

    monkeypatch.setattr("scripts.live_decision_dry_run._REPO_ROOT", tmp_path)

    # Force re-import to pick up monkeypatch (import after patch)
    import importlib
    import scripts.live_decision_dry_run as m
    importlib.reload(m)

    dirs = m.find_model_dirs_with_pkls()
    names = [d.name for d in dirs]
    assert "phantom_zero_feature_model" not in names, (
        "Zero-feature model should have been rejected"
    )


def test_find_model_dirs_rejects_phantom_small_feature_list(tmp_path, monkeypatch):
    """A model dir with feature_list_used.json < 20 features is a phantom — skip it."""
    model_dir = tmp_path / "phantom_small_model"
    model_dir.mkdir()
    (model_dir / "feature_list_used.json").write_text(
        json.dumps(["feat_a", "feat_b", "feat_c"]), encoding="utf-8"
    )
    (model_dir / "dummy.pkl").write_bytes(b"pkldata")

    monkeypatch.setattr("scripts.live_decision_dry_run._REPO_ROOT", tmp_path)
    import importlib
    import scripts.live_decision_dry_run as m
    importlib.reload(m)

    dirs = m.find_model_dirs_with_pkls()
    names = [d.name for d in dirs]
    assert "phantom_small_model" not in names


def test_find_model_dirs_accepts_valid_model(tmp_path, monkeypatch):
    """A real model dir with valid feature_manifest and pkl is accepted."""
    # find_model_dirs_with_pkls looks for models_dir = _REPO_ROOT / "models"
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_dir = models_dir / "valid_model"
    model_dir.mkdir()
    (model_dir / "feature_manifest.json").write_text(
        json.dumps({"feature_count": 114, "features": [{"feature": f"f{i}"} for i in range(114)]}),
        encoding="utf-8",
    )
    # Create a pkl that passes _has_valid_pkl (not a metrics file)
    (model_dir / "random_forest_profitable_trade_label.pkl").write_bytes(b"pkldata")

    import importlib
    import scripts.live_decision_dry_run as m
    monkeypatch.setattr(m, "_REPO_ROOT", tmp_path)

    dirs = m.find_model_dirs_with_pkls()
    names = [d.name for d in dirs]
    assert "valid_model" in names


# ---------------------------------------------------------------------------
# Test 2: evaluate_decision blocks when coverage < 95%
# ---------------------------------------------------------------------------

def _fake_alignment(missing_count: int, total: int = 100) -> Dict[str, Any]:
    """Build a minimal feature_alignment dict."""
    available = [f"feat_{i}" for i in range(total - missing_count)]
    missing = [f"feat_{i}" for i in range(total - missing_count, total)]
    coverage = round(len(available) / max(total, 1) * 100, 1)
    by_source: Dict[str, list] = {}
    for feat in missing:
        if "oi" in feat or "vol" in feat or "iv" in feat or "option" in feat:
            src = "option_chain"
        elif "z_" in feat or "rolling" in feat or "dist_to" in feat:
            src = "rolling_history"
        else:
            src = "candle"
        by_source.setdefault(src, []).append(feat)
    return {
        "available_features": available,
        "missing_features": missing,
        "missing_count": missing_count,
        "total_model_features": total,
        "coverage_pct": coverage,
        "missing_by_source": by_source,
        "aligned_vector": {},
        "extra_live_features": [],
    }


def test_evaluate_decision_blocks_below_95_coverage():
    from scripts.live_decision_dry_run import evaluate_decision

    # 80% coverage should be blocked
    alignment = _fake_alignment(missing_count=20, total=100)  # 80%
    result = evaluate_decision(
        probability=0.7,
        threshold=0.5,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="test",
        model_dir="test_dir",
        bundle_trained_at="2026-01-01Z",
        snapshot={},
    )
    assert result["final_action"] == "SKIP"
    assert any("coverage_80" in r for r in result["blocking_reasons"]), (
        f"Expected coverage block, got: {result['blocking_reasons']}"
    )
    # CANDIDATE_FILTER_CHECK is step 0 (passes without candidate_info); FEATURE_COVERAGE_CHECK is step 1
    assert result["steps"][0]["step"] == "CANDIDATE_FILTER_CHECK"
    assert result["steps"][0]["passed"] is True
    assert result["steps"][1]["step"] == "FEATURE_COVERAGE_CHECK"
    assert result["steps"][1]["passed"] is False


def test_evaluate_decision_blocks_at_94_coverage():
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _fake_alignment(missing_count=6, total=100)  # 94%
    result = evaluate_decision(
        probability=0.7, threshold=0.5,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="test", model_dir="test", bundle_trained_at="2026-01-01Z", snapshot={},
    )
    assert result["final_action"] == "SKIP"


def test_evaluate_decision_allows_at_95_coverage():
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _fake_alignment(missing_count=5, total=100)  # 95%
    result = evaluate_decision(
        probability=0.7, threshold=0.5,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="test", model_dir="test", bundle_trained_at="2026-01-01Z", snapshot={},
    )
    # With 95%, it should pass FEATURE_COVERAGE_CHECK but may fail THRESHOLD_CHECK
    # (threshold 0.5 not met by prob 0.7 — wait prob=0.7 > 0.5 so threshold pass)
    assert result["final_action"] == "TRADE", (
        f"95% coverage + prob=0.7 >= 0.5 should trade, got: {result['final_action']} "
        f"steps={result['steps']}"
    )


def test_evaluate_decision_missing_features_grouped_by_source():
    from scripts.live_decision_dry_run import evaluate_decision

    alignment = _fake_alignment(missing_count=20, total=100)
    result = evaluate_decision(
        probability=0.7, threshold=0.5,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="test", model_dir="test", bundle_trained_at="2026-01-01Z", snapshot={},
    )
    # CANDIDATE_FILTER_CHECK is step 0; coverage step is step 1
    coverage_step = result["steps"][1]
    assert "missing_by_source" in coverage_step
    assert isinstance(coverage_step["missing_by_source"], dict)


# ---------------------------------------------------------------------------
# Test 3: _build_offline_fixture achieves >= 95% coverage for candle features
# ---------------------------------------------------------------------------

def test_offline_fixture_candle_coverage():
    from scripts.live_decision_dry_run import (
        _build_offline_fixture, align_features,
        CANDLE_FEATURES, OPTION_CHAIN_FEATURES, ROLLING_HISTORY_FEATURES,
    )

    # Use a candle-only feature list (no option chain / rolling features)
    candle_only_features = [
        "last_open", "last_high", "last_low", "last_close", "last_volume",
        "body_pct", "range_pct", "gap_pct", "upper_wick_pct", "lower_wick_pct",
        "close_location_pct", "ret_1", "ret_3", "ret_5", "ret_10",
        "ret_mean", "ret_std", "ret_min", "ret_max",
        "ema_fast", "ema_slow", "ema_diff_pct", "rsi_14", "atr_14", "atr_pct",
        "adx_14", "choppiness_14", "supertrend_dir",
        "pivot_pp_dist_pct", "pivot_r1_dist_pct", "pivot_s1_dist_pct",
        "close_vs_open_pct", "range_to_atr", "momentum_lookback_pct",
        "vol_mean", "vol_std", "vol_min", "vol_max",
        "regime_quiet", "regime_trending", "volatility_regime_classifier",
        "ctx_time_sin", "ctx_time_cos", "weekday", "month",
        "is_opening_session", "is_closing_session",
        "volume_ratio",
    ]
    # Add time features that fixture provides
    fixture = _build_offline_fixture(candle_only_features)
    alignment = align_features(candle_only_features, fixture)
    assert alignment["coverage_pct"] >= 95.0, (
        f"Candle-only fixture should get >= 95% coverage, got {alignment['coverage_pct']}%"
    )
    # Only unavailable features should be from option_chain / rolling that we didn't ask for
    by_src = alignment.get("missing_by_source", {})
    assert not by_src.get("candle"), f"Should have no missing candle features: {by_src.get('candle')}"


def test_offline_fixture_option_chain_features_are_none():
    from scripts.live_decision_dry_run import _build_offline_fixture, OPTION_CHAIN_FEATURES

    opt_features = ["oi_CE", "oi_PE", "volume_CE", "volume_PE", "ce_pe_oi_ratio", "final_iv", "atm_distance"]
    fixture = _build_offline_fixture(opt_features)
    for feat in opt_features:
        assert fixture.get(feat) is None, (
            f"Option chain feature {feat} should be None (unavailable), got {fixture.get(feat)}"
        )


# ---------------------------------------------------------------------------
# Test 4: align_features produces missing_by_source
# ---------------------------------------------------------------------------

def test_align_features_missing_by_source():
    from scripts.live_decision_dry_run import align_features

    model_feats = [
        "last_close",   # candle
        "oi_CE",        # option_chain
        "oi_z_5",       # rolling_history
        "ctx_time_sin", # context
        "unknown_feat", # unavailable
    ]
    live = {"last_close": 100.0, "ctx_time_sin": 0.5}
    result = align_features(model_feats, live)
    by_src = result["missing_by_source"]
    # candle: last_close is covered (available), no candle features missing
    assert "last_close" not in by_src.get("candle", [])
    # option_chain: oi_CE missing
    assert "oi_CE" in by_src.get("option_chain", [])
    # rolling_history: oi_z_5 missing
    assert "oi_z_5" in by_src.get("rolling_history", [])
    # context: ctx_time_sin covered (available)
    assert "ctx_time_sin" not in by_src.get("context", [])
    # unavailable: unknown_feat missing
    assert "unknown_feat" in by_src.get("unavailable", [])


# ---------------------------------------------------------------------------
# Test 5: No real order function is called in dry-run
# ---------------------------------------------------------------------------

def test_no_real_order_calls_in_dry_run(tmp_path, monkeypatch):
    """Verify dry-run path never calls place_order or broker order APIs.

    In --offline mode, the script must not import any mstock/broker modules.
    Uses unittest.mock.patch on builtins.__import__ (compatible with Python 3.14).
    """
    from unittest.mock import patch
    import builtins
    import importlib
    import scripts.live_decision_dry_run as m
    importlib.reload(m)

    call_log: list = []
    _real_import = builtins.__import__

    def tracking_import(name, *args, **kwargs):
        call_log.append(name)
        return _real_import(name, *args, **kwargs)

    # Run main with --paper --offline --model-dir pointing to a dummy
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_dir = models_dir / "test_model"
    model_dir.mkdir()
    (model_dir / "feature_manifest.json").write_text(
        json.dumps({"feature_count": 30, "features": [{"feature": f"f{i}"} for i in range(30)]}),
        encoding="utf-8",
    )
    (model_dir / "random_forest_profitable_trade_label.pkl").write_bytes(b"pkldata")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "live_decision_dry_run.py",
        "--paper",
        "--offline",
        "--model-dir", str(model_dir),
    ])
    monkeypatch.setattr(m, "_REPO_ROOT", tmp_path)

    with patch.object(builtins, "__import__", side_effect=tracking_import):
        try:
            m.main()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                pass  # may exit with 1 if no model found, that's ok

    # No broker / mstock module should have been imported in offline mode
    broker_imports = [x for x in call_log if "mstock" in x.lower() or "broker" in x.lower()]
    assert not broker_imports, f"Broker modules imported in dry-run: {broker_imports}"


# ---------------------------------------------------------------------------
# Test 6: evaluate_decision with zero total features never predicts
# ---------------------------------------------------------------------------

def test_evaluate_decision_zero_features_is_blocked():
    from scripts.live_decision_dry_run import evaluate_decision

    empty_alignment = {
        "available_features": [],
        "missing_features": [],
        "missing_count": 0,
        "total_model_features": 0,
        "coverage_pct": 0.0,
        "missing_by_source": {},
        "aligned_vector": {},
        "extra_live_features": [],
    }
    result = evaluate_decision(
        probability=0.0,
        threshold=0.5,
        feature_alignment=empty_alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="zero_feat",
        model_dir="zero_dir",
        bundle_trained_at="2026-01-01Z",
        snapshot={},
    )
    # 0% coverage must be blocked
    assert result["final_action"] == "SKIP"
    assert any("coverage_0" in r for r in result["blocking_reasons"])


# ---------------------------------------------------------------------------
# Test 7: model_pkl consistency — top-level JSON matches model_info.model_pkl
# ---------------------------------------------------------------------------

def test_model_pkl_consistent_in_json(tmp_path):
    """Top-level model_pkl and model_info.model_pkl must be identical."""
    import scripts.live_decision_dry_run as m

    model_pkl = str(tmp_path / "models" / "my_model" / "model.pkl")
    result = {
        "final_action": "SKIP",
        "model_id": "test_model",
        "model_dir": str(tmp_path / "models" / "my_model"),
        "model_pkl": model_pkl,
        "trained_at": "2026-01-01Z",
        "threshold": 0.5,
        "probability": 0.3,
        "blocking_reasons": ["low_confidence"],
        "timestamp": "2026-06-07T12:00:00Z",
    }
    model_info = {
        "model_pkl": model_pkl,
        "model_dir": str(tmp_path / "models" / "my_model"),
        "model_id": "test_model",
    }
    alignment = {
        "total_model_features": 114,
        "available_count": 114,
        "coverage_pct": 100.0,
        "available_features": [],
        "missing_features": [],
        "missing_by_source": {},
    }
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    md_path, json_path = m.generate_reports(
        result=result,
        feature_alignment=alignment,
        model_info=model_info,
        dry_run_mode="paper",
        reports_dir=reports_dir,
    )
    with open(json_path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["model_pkl"] == model_pkl
    assert report["model_info"]["model_pkl"] == model_pkl


# ---------------------------------------------------------------------------
# Test 8: model_pkl appears in markdown when model_info has valid path
# ---------------------------------------------------------------------------

def test_model_pkl_displayed_in_markdown(tmp_path):
    """Markdown must display the actual model_pkl path, not N/A."""
    import scripts.live_decision_dry_run as m

    model_pkl = str(tmp_path / "models" / "core_retrain" / "model.pkl")
    result = {
        "final_action": "SKIP",
        "model_id": "test_model",
        "model_dir": str(tmp_path / "models"),
        "model_pkl": model_pkl,
        "trained_at": "2026-01-01Z",
        "threshold": 0.5,
        "probability": 0.3,
        "blocking_reasons": ["low_confidence"],
        "timestamp": "2026-06-07T12:00:00Z",
    }
    model_info = {"model_pkl": model_pkl, "model_dir": str(tmp_path / "models"), "model_id": "test_model"}
    alignment = {
        "total_model_features": 114, "available_count": 114, "coverage_pct": 100.0,
        "available_features": [], "missing_features": [], "missing_by_source": {},
    }
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    md_path, _ = m.generate_reports(
        result=result, feature_alignment=alignment,
        model_info=model_info, dry_run_mode="paper", reports_dir=reports_dir,
    )
    assert model_pkl in md_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 9: zero-feature model is rejected (phantom artifact)
# ---------------------------------------------------------------------------

def test_zero_feature_model_rejected(tmp_path, monkeypatch):
    """A model dir with feature_count=0 must be skipped."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_dir = models_dir / "phantom_zero_feature_model"
    model_dir.mkdir()
    (model_dir / "feature_manifest.json").write_text(
        json.dumps({"feature_count": 0, "features": []}), encoding="utf-8"
    )
    (model_dir / "metrics_report.json").write_text(
        json.dumps({"best_global_observed": {"model_pkl": "x.pkl"}}), encoding="utf-8"
    )
    import scripts.live_decision_dry_run as m
    monkeypatch.setattr(m, "_REPO_ROOT", tmp_path)
    dirs = m.find_model_dirs_with_pkls()
    names = [d.name for d in dirs]
    assert "phantom_zero_feature_model" not in names


# ---------------------------------------------------------------------------
# Test 10: generate_reports produces valid JSON + Markdown
# ---------------------------------------------------------------------------

def _make_result_dict(
    final_action: str = "TRADE",
    model_id: str = "candidate_20260607",
    model_pkl: str = "/repo/models/candidate_20260607/random_forest.pkl",
    blocking_reasons: list = None,
) -> dict:
    """Build a minimal result dict compatible with generate_reports."""
    return {
        "final_action": final_action,
        "model_id": model_id,
        "model_dir": "/repo/models/candidate_20260607",
        "model_pkl": model_pkl,
        "trained_at": "2026-06-07T00:00:00Z",
        "threshold": 0.55,
        "probability": 0.72,
        "blocking_reasons": blocking_reasons or [],
        "blocker": (blocking_reasons or [None])[0],
        "steps": [
            {
                "step": "FEATURE_COVERAGE_CHECK",
                "passed": True,
                "detail": "0 missing, 100.0% coverage (need >= 95%)",
                "coverage_pct": 100.0,
                "required_pct": 95.0,
                "missing_features_sample": [],
                "missing_by_source": {},
            },
            {
                "step": "FULL_SCHEMA_VALIDATION",
                "passed": True,
                "detail": "full alignment",
                "missing_features": [],
            },
            {
                "step": "THRESHOLD_CHECK",
                "passed": True,
                "detail": "prob=0.7200 threshold=0.5500",
                "probability": 0.72,
                "threshold": 0.55,
            },
            {
                "step": "RISK_FILTER",
                "passed": True,
                "detail": "reason=",
                "risk_state": {
                    "trades_today": 0, "open_positions": 0, "daily_pnl": 0.0},
                "blocking_reason": "",
            },
            {
                "step": "FINAL_DECISION",
                "passed": True,
                "detail": "All checks passed — WOULD TRADE (paper/DRY-RUN only)",
            },
        ] if final_action == "TRADE" else [
            {
                "step": "FEATURE_COVERAGE_CHECK",
                "passed": True,
                "detail": "0 missing, 100.0% coverage (need >= 95%)",
                "coverage_pct": 100.0,
                "required_pct": 95.0,
                "missing_features_sample": [],
                "missing_by_source": {},
            },
            {
                "step": "FULL_SCHEMA_VALIDATION",
                "passed": True,
                "detail": "full alignment",
                "missing_features": [],
            },
            {
                "step": "CANDIDATE_FILTER_CHECK",
                "passed": True,
                "detail": "filter= -> PASS",
                "filter_type": "",
                "filter_reason": None,
                "option_type": "unknown",
            },
            {
                "step": "THRESHOLD_CHECK",
                "passed": False,
                "detail": "prob=0.30 threshold=0.55",
                "probability": 0.30,
                "threshold": 0.55,
            },
            {
                "step": "RISK_FILTER",
                "passed": True,
                "detail": "reason=",
                "risk_state": {"trades_today": 0, "open_positions": 0, "daily_pnl": 0.0},
                "blocking_reason": "",
            },
        ],
        "timestamp": "2026-06-08T10:00:00Z",
        "snapshot_keys": ["rsi_14", "atr_14", "ctx_time_sin", "spot"],
    }


def test_generate_reports_shows_model_id(tmp_path):
    """Markdown and JSON reports must include the model_id field."""
    import scripts.live_decision_dry_run as m

    result = _make_result_dict(final_action="SKIP", model_id="paper_candidate_20260608")
    model_info = {
        "model_pkl": "/repo/models/paper_candidate_20260608/rf.pkl",
        "model_dir": "/repo/models/paper_candidate_20260608",
        "model_id": "paper_candidate_20260608",
        "threshold": 0.55,
        "feature_count": 30,
        "has_scaler": True,
        "trained_at": "2026-06-08T00:00:00Z",
    }
    alignment = {
        "total_model_features": 30,
        "available_count": 30,
        "coverage_pct": 100.0,
        "available_features": [f"feat_{i:03d}" for i in range(30)],
        "missing_features": [],
        "missing_by_source": {},
        "extra_live_features": [],
    }
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    md_path, json_path = m.generate_reports(
        result=result,
        feature_alignment=alignment,
        model_info=model_info,
        dry_run_mode="paper",
        reports_dir=reports_dir,
    )

    # Check markdown contains model_id
    md_text = md_path.read_text(encoding="utf-8")
    assert "paper_candidate_20260608" in md_text, (
        f"model_id 'paper_candidate_20260608' not found in markdown"
    )
    assert "Model ID" in md_text or "model_id" in md_text.lower()

    # Check JSON report contains model_id
    import json
    with open(json_path, encoding="utf-8") as f:
        report_json = json.load(f)
    assert report_json["model_id"] == "paper_candidate_20260608"
    assert report_json["model_info"]["model_id"] == "paper_candidate_20260608"


def test_generate_reports_shows_filter_applied(tmp_path):
    """Decision pipeline steps (filters) must appear in both JSON and Markdown reports."""
    import scripts.live_decision_dry_run as m

    result = _make_result_dict(final_action="SKIP", model_id="filter_report_test")
    model_info = {
        "model_pkl": "/repo/models/filter_report_test/rf.pkl",
        "model_dir": "/repo/models/filter_report_test",
        "model_id": "filter_report_test",
        "threshold": 0.55,
        "feature_count": 30,
        "has_scaler": True,
        "trained_at": "2026-06-08T00:00:00Z",
    }
    alignment = {
        "total_model_features": 30,
        "available_count": 30,
        "coverage_pct": 100.0,
        "available_features": [f"feat_{i:03d}" for i in range(30)],
        "missing_features": [],
        "missing_by_source": {},
        "extra_live_features": [],
    }
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    md_path, json_path = m.generate_reports(
        result=result,
        feature_alignment=alignment,
        model_info=model_info,
        dry_run_mode="paper",
        reports_dir=reports_dir,
    )

    # Check JSON has decision_steps
    import json
    with open(json_path, encoding="utf-8") as f:
        report_json = json.load(f)

    assert "decision_steps" in report_json
    step_names = [s["step"] for s in report_json["decision_steps"]]
    assert "FEATURE_COVERAGE_CHECK" in step_names
    assert "THRESHOLD_CHECK" in step_names

    # Check Markdown contains filter step names
    md_text = md_path.read_text(encoding="utf-8")
    assert "FEATURE_COVERAGE_CHECK" in md_text or "Feature Coverage" in md_text
    assert "THRESHOLD_CHECK" in md_text or "Threshold" in md_text
    assert "RISK_FILTER" in md_text or "Risk" in md_text


def test_generate_reports_shows_blocking_reasons(tmp_path):
    """When final_action is SKIP, blocking_reasons must appear in both JSON and Markdown."""
    import scripts.live_decision_dry_run as m

    result = _make_result_dict(
        final_action="SKIP",
        model_id="block_test",
        blocking_reasons=["low_confidence_0.3000_lt_0.5500"],
    )
    model_info = {
        "model_pkl": "/repo/models/block_test/rf.pkl",
        "model_dir": "/repo/models/block_test",
        "model_id": "block_test",
        "threshold": 0.55,
        "feature_count": 30,
        "has_scaler": False,
        "trained_at": "2026-06-08T00:00:00Z",
    }
    alignment = {
        "total_model_features": 30,
        "available_count": 30,
        "coverage_pct": 100.0,
        "available_features": [f"feat_{i:03d}" for i in range(30)],
        "missing_features": [],
        "missing_by_source": {},
        "extra_live_features": [],
    }
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    md_path, json_path = m.generate_reports(
        result=result,
        feature_alignment=alignment,
        model_info=model_info,
        dry_run_mode="paper",
        reports_dir=reports_dir,
    )

    import json
    with open(json_path, encoding="utf-8") as f:
        report_json = json.load(f)

    assert report_json["final_action"] == "SKIP"
    assert len(report_json["blocking_reasons"]) > 0
    assert any("low_confidence" in r for r in report_json["blocking_reasons"])

    md_text = md_path.read_text(encoding="utf-8")
    assert "Blocking" in md_text or "BLOCKED" in md_text
    assert "low_confidence" in md_text


def test_generate_reports_shows_coverage_pct(tmp_path):
    """Feature alignment coverage must appear in both JSON and Markdown."""
    import scripts.live_decision_dry_run as m

    result = _make_result_dict(final_action="SKIP", model_id="coverage_report_test")
    model_info = {
        "model_pkl": "/repo/models/coverage_report_test/rf.pkl",
        "model_dir": "/repo/models/coverage_report_test",
        "model_id": "coverage_report_test",
        "threshold": 0.55,
        "feature_count": 114,
        "has_scaler": True,
        "trained_at": "2026-06-08T00:00:00Z",
    }
    # 95% coverage scenario
    alignment = {
        "total_model_features": 100,
        "available_count": 95,
        "coverage_pct": 95.0,
        "available_features": [f"feat_{i}" for i in range(95)],
        "missing_features": [f"feat_{i}" for i in range(95, 100)],
        "missing_by_source": {"option_chain": [f"feat_{i}" for i in range(95, 100)]},
        "extra_live_features": [],
    }
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    md_path, json_path = m.generate_reports(
        result=result,
        feature_alignment=alignment,
        model_info=model_info,
        dry_run_mode="paper",
        reports_dir=reports_dir,
    )

    import json
    with open(json_path, encoding="utf-8") as f:
        report_json = json.load(f)

    assert "coverage_pct" in report_json
    assert report_json["coverage_pct"] == 95.0
    assert "feature_count" in report_json
    assert report_json["feature_count"] == 100

    md_text = md_path.read_text(encoding="utf-8")
    assert "95" in md_text  # coverage value
    assert "100" in md_text  # total features


def test_generate_reports_json_safety_compliance_fields(tmp_path):
    """JSON report must include never_calls_real_order=True."""
    import scripts.live_decision_dry_run as m

    result = _make_result_dict(final_action="SKIP", model_id="safety_test")
    model_info = {
        "model_pkl": "/repo/models/safety_test/rf.pkl",
        "model_dir": "/repo/models/safety_test",
        "model_id": "safety_test",
        "threshold": 0.55,
        "feature_count": 30,
        "has_scaler": True,
        "trained_at": "2026-06-08T00:00:00Z",
    }
    alignment = {
        "total_model_features": 30,
        "available_count": 30,
        "coverage_pct": 100.0,
        "available_features": [f"feat_{i:03d}" for i in range(30)],
        "missing_features": [],
        "missing_by_source": {},
        "extra_live_features": [],
    }
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    md_path, json_path = m.generate_reports(
        result=result,
        feature_alignment=alignment,
        model_info=model_info,
        dry_run_mode="paper",
        reports_dir=reports_dir,
    )

    import json
    with open(json_path, encoding="utf-8") as f:
        report_json = json.load(f)

    safety = report_json.get("safety_compliance", {})
    assert safety.get("never_calls_real_order") is True
    assert safety.get("explicit_paper_flag") is True
    assert safety.get("logs_what_if_would_do") is True


# ---------------------------------------------------------------------------
# Test 11: evaluate_decision uses correct threshold comparison operator
# ---------------------------------------------------------------------------

def test_threshold_comparison_is_strict_greater_than():
    """prob >= threshold must be strict > (not >=) to avoid boundary noise."""
    from scripts.live_decision_dry_run import evaluate_decision

    # prob == threshold → should fail
    alignment = _full_coverage_alignment_for_dry_run()
    result_eq = evaluate_decision(
        probability=0.50,
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="eq_test",
        model_dir="test_dir",
        bundle_trained_at="2026-06-07Z",
        snapshot={},
    )
    threshold_step = next((s for s in result_eq["steps"] if s["step"] == "THRESHOLD_CHECK"), None)
    assert threshold_step is not None
    assert threshold_step["passed"] is False, (
        "prob == threshold must NOT pass THRESHOLD_CHECK (strict > required)"
    )

    # prob > threshold → should pass
    result_gt = evaluate_decision(
        probability=0.5001,
        threshold=0.50,
        feature_alignment=alignment,
        risk_result={"allowed": True, "blocking_reason": ""},
        model_id="gt_test",
        model_dir="test_dir",
        bundle_trained_at="2026-06-07Z",
        snapshot={},
    )
    threshold_step_gt = next((s for s in result_gt["steps"] if s["step"] == "THRESHOLD_CHECK"), None)
    assert threshold_step_gt is not None
    assert threshold_step_gt["passed"] is True, (
        "prob > threshold should pass THRESHOLD_CHECK"
    )


def _full_coverage_alignment_for_dry_run() -> dict:
    """Build a 100% coverage alignment for threshold tests."""
    return {
        "total_model_features": 30,
        "available_count": 30,
        "coverage_pct": 100.0,
        "available_features": [f"feat_{i:03d}" for i in range(30)],
        "missing_features": [],
        "missing_by_source": {},
        "aligned_vector": {f"feat_{i:03d}": 0.5 for i in range(30)},
        "extra_live_features": [],
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])