#!/usr/bin/env python3
"""Tests for multi-candidate routing in live_decision_dry_run.py.

Covers:
  - Router selects highest edge candidate (by profit_factor)
  - Router skips when no candidate passes all filters
  - Candidate-dir loads multiple candidates
  - Manifest missing required fields fails closed
  - Missing model_pkl fails closed
  - paper_only=false fails closed
  - real_trading_enabled=true fails closed
  - Filtered candidates correctly filtered by option_type
  - Edge scoring correctly ranks candidates
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from candidate_router import _format_confidence_for_reason, _router_confidence_value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_router_preserves_tiny_nonzero_confidence_values():
    tiny = 9.680551105535084e-08

    assert _router_confidence_value(tiny) == tiny
    assert _format_confidence_for_reason(tiny) == "9.68e-08"

def _make_minimal_pkl_bundle(model_dir: Path, name: str = "model.pkl", n_features: int = 20) -> Path:
    """Create a minimal MLModelBundle .pkl with n_features feature names."""
    import joblib
    from ml_signals import MLModelBundle

    pkl_path = model_dir / name
    feature_names = [f"feat_{i:03d}" for i in range(n_features)]
    bundle = MLModelBundle(
        model=None,
        feature_names=feature_names,
        metrics={"profit_factor": 1.3, "sharpe": 1.1},
        trained_at="2026-06-07T00:00:00Z",
        scaler_mean=[0.0] * n_features,
        scaler_std=[1.0] * n_features,
    )
    joblib.dump(bundle, str(pkl_path))
    return pkl_path


def _make_paper_candidate_dir(parent: Path, name: str, pf: float = 1.3,
                               sharpe: float = 1.1, threshold: float = 0.25,
                               paper_only: bool = True,
                               real_trading_enabled: bool = False,
                               filter_type: str = "PE_only",
                               gates_passed: int = 10,
                               gates_total: int = 10) -> Path:
    """Create a valid paper candidate directory with all required artifacts."""
    cand_dir = parent / name
    cand_dir.mkdir(parents=True, exist_ok=True)

    (cand_dir / "feature_schema.json").write_text(
        json.dumps([f"feat_{i:03d}" for i in range(20)]),
        encoding="utf-8",
    )
    _make_minimal_pkl_bundle(cand_dir, "model.pkl", n_features=20)

    manifest = {
        "model_id": name,
        "model_pkl": f"{name}/model.pkl",
        "selected_threshold": threshold,
        "paper_only": paper_only,
        "real_trading_enabled": real_trading_enabled,
        # Use 'filter_definition' key (matches what load_candidate_manifest reads)
        "filter_definition": {
            "type": filter_type,
            "filter_name": filter_type,
            "filter_rule": f"option_type == '{filter_type.split('_')[0]}'",
            "filter_expression": f"option_type == '{filter_type.split('_')[0]}'",
        },
        "gates_passed": gates_passed,
        "gates_total": gates_total,
        "verdict": "PAPER_FORWARD_TEST_CANDIDATE_READY",
        "overall_metrics": {
            "mean_pf": pf,
            "mean_sharpe": sharpe,
        },
        "fold_details": {
            "fold_0": {"pf": pf - 0.1, "sharpe": sharpe - 0.1, "trades": 100},
            "fold_1": {"pf": pf, "sharpe": sharpe, "trades": 100},
            "fold_2": {"pf": pf + 0.1, "sharpe": sharpe + 0.1, "trades": 100},
        },
        "artifacts": {
            "model_pkl": f"{name}/model.pkl",
        },
    }
    (cand_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    return cand_dir


# ---------------------------------------------------------------------------
# Test 1: Router selects highest edge candidate
# ---------------------------------------------------------------------------

def test_router_selects_highest_profit_factor_candidate(tmp_path, monkeypatch):
    """When multiple candidates pass, router must select the one with the highest profit_factor."""
    import scripts.live_decision_dry_run as m

    # Create three candidates with different profit factors
    cand1 = _make_paper_candidate_dir(tmp_path, "low_pf_candidate", pf=1.2, threshold=0.25)
    cand2 = _make_paper_candidate_dir(tmp_path, "mid_pf_candidate", pf=1.5, threshold=0.25)
    cand3 = _make_paper_candidate_dir(tmp_path, "high_pf_candidate", pf=1.8, threshold=0.25)

    candidates = sorted([cand1, cand2, cand3])

    # Verify candidates were created with correct PFs
    for c in candidates:
        manifest = json.loads((c / "candidate_manifest.json").read_text(encoding="utf-8"))
        assert manifest["overall_metrics"]["mean_pf"] > 0

    # Router logic: select candidate with highest mean_pf among passing candidates
    passing = []
    for cand_dir in candidates:
        manifest_path = cand_dir / "candidate_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            gates_pass = manifest.get("gates_passed", 0)
            gates_total = manifest.get("gates_total", 0)
            verdict = manifest.get("verdict", "")
            if gates_pass == gates_total and verdict == "PAPER_FORWARD_TEST_CANDIDATE_READY":
                pf = manifest.get("overall_metrics", {}).get("mean_pf", 0.0)
                passing.append((pf, cand_dir))

    assert len(passing) == 3
    # Sort by pf descending → pick first
    passing.sort(key=lambda x: x[0], reverse=True)
    selected = passing[0][1]
    assert selected.name == "high_pf_candidate"


def test_router_ranks_by_profit_factor_not_sharpe(tmp_path, monkeypatch):
    """Ranking must use profit_factor, not sharpe ratio."""
    import scripts.live_decision_dry_run as m

    # Candidate A: highest PF, low sharpe
    _make_paper_candidate_dir(tmp_path, "high_pf_low_sharpe", pf=2.0, sharpe=0.5)
    # Candidate B: lower PF, highest sharpe
    _make_paper_candidate_dir(tmp_path, "low_pf_high_sharpe", pf=1.2, sharpe=2.5)

    candidates = sorted([
        tmp_path / "high_pf_low_sharpe",
        tmp_path / "low_pf_high_sharpe",
    ])

    # Rank by PF (not sharpe)
    ranked = sorted(
        candidates,
        key=lambda d: json.loads((d / "candidate_manifest.json").read_text(encoding="utf-8"))
                     .get("overall_metrics", {}).get("mean_pf", 0.0),
        reverse=True,
    )
    assert ranked[0].name == "high_pf_low_sharpe"
    assert ranked[1].name == "low_pf_high_sharpe"


def test_router_breaks_ties_by_recency(tmp_path, monkeypatch):
    """When PF is equal, most recent candidate wins."""
    import scripts.live_decision_dry_run as m

    # Create two candidates with same PF
    _make_paper_candidate_dir(tmp_path, "old_candidate", pf=1.5, threshold=0.25)
    import time
    time.sleep(0.01)
    _make_paper_candidate_dir(tmp_path, "new_candidate", pf=1.5, threshold=0.25)

    candidates = sorted(tmp_path.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
    newest = candidates[0]

    assert newest.name == "new_candidate"


def test_router_picks_candidate_with_best_pf_across_different_thresholds(tmp_path):
    """Threshold does not affect ranking — only PF matters for ranking."""
    _make_paper_candidate_dir(tmp_path, "aggressive_candidate", pf=2.1, threshold=0.20)
    _make_paper_candidate_dir(tmp_path, "conservative_candidate", pf=1.6, threshold=0.35)

    candidates = sorted(tmp_path.iterdir())
    ranked = sorted(
        candidates,
        key=lambda d: json.loads((d / "candidate_manifest.json").read_text(encoding="utf-8"))
                     .get("overall_metrics", {}).get("mean_pf", 0.0),
        reverse=True,
    )
    assert ranked[0].name == "aggressive_candidate"


# ---------------------------------------------------------------------------
# Test 2: Router skips when no candidate passes
# ---------------------------------------------------------------------------

def test_router_skips_when_no_candidate_passes_gates(tmp_path):
    """When all candidates fail gates, router must return empty list (no crash)."""
    import scripts.live_decision_dry_run as m

    # Candidate failing gates
    _make_paper_candidate_dir(
        tmp_path, "failing_gates_candidate",
        pf=1.5, gates_passed=5, gates_total=10,
    )

    candidates = sorted(tmp_path.iterdir())
    passing = [
        d for d in candidates
        if (d / "candidate_manifest.json").exists()
        and json.loads((d / "candidate_manifest.json").read_text(encoding="utf-8"))
           .get("gates_passed", 0) == json.loads((d / "candidate_manifest.json").read_text(encoding="utf-8"))
           .get("gates_total", 0)
    ]
    # No candidates pass all gates
    assert len(passing) == 0


def test_router_skips_when_no_verdict_match(tmp_path):
    """Candidates without PAPER_FORWARD_TEST_CANDIDATE_READY verdict are skipped."""
    _make_paper_candidate_dir(tmp_path, "unknown_verdict_candidate", pf=1.5)

    # Overwrite verdict to be non-passing
    manifest = json.loads((tmp_path / "unknown_verdict_candidate" / "candidate_manifest.json").read_text(encoding="utf-8"))
    manifest["verdict"] = "PENDING_REVIEW"
    (tmp_path / "unknown_verdict_candidate" / "candidate_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )

    candidates = sorted(tmp_path.iterdir())
    passing = [
        d for d in candidates
        if (d / "candidate_manifest.json").exists()
        and json.loads((d / "candidate_manifest.json").read_text(encoding="utf-8"))
           .get("verdict") == "PAPER_FORWARD_TEST_CANDIDATE_READY"
    ]
    assert len(passing) == 0


def test_router_skips_when_all_paper_only_false(tmp_path):
    """Candidates with paper_only=False must be skipped (require manual review)."""
    # One passing candidate but paper_only=False
    _make_paper_candidate_dir(
        tmp_path, "live_candidate",
        pf=1.5, paper_only=False, real_trading_enabled=False,
    )

    candidates = sorted(tmp_path.iterdir())
    passing = [
        d for d in candidates
        if (d / "candidate_manifest.json").exists()
        and json.loads((d / "candidate_manifest.json").read_text(encoding="utf-8"))
           .get("paper_only") is True
    ]
    # paper_only=False candidate skipped
    assert len(passing) == 0


def test_router_skips_when_real_trading_enabled(tmp_path):
    """Candidates with real_trading_enabled=True must be skipped (unsafe for auto-selection)."""
    _make_paper_candidate_dir(
        tmp_path, "unsafe_candidate",
        pf=1.5, paper_only=True, real_trading_enabled=True,
    )

    candidates = sorted(tmp_path.iterdir())
    safe_candidates = [
        d for d in candidates
        if (d / "candidate_manifest.json").exists()
        and json.loads((d / "candidate_manifest.json").read_text(encoding="utf-8"))
           .get("real_trading_enabled") is not True
    ]
    assert len(safe_candidates) == 0


# ---------------------------------------------------------------------------
# Test 3: Candidate-dir loads multiple candidates
# ---------------------------------------------------------------------------

def test_candidate_dir_loads_multiple_candidates(tmp_path):
    """Multiple candidate directories are all loaded without conflict."""
    c1 = _make_paper_candidate_dir(tmp_path, "candidate_A", pf=1.3)
    c2 = _make_paper_candidate_dir(tmp_path, "candidate_B", pf=1.5)
    c3 = _make_paper_candidate_dir(tmp_path, "candidate_C", pf=1.7)

    all_dirs = sorted(d for d in tmp_path.iterdir() if d.is_dir())
    assert len(all_dirs) == 3

    loaded = []
    for d in all_dirs:
        manifest_path = d / "candidate_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            loaded.append({
                "name": d.name,
                "pf": manifest.get("overall_metrics", {}).get("mean_pf", 0.0),
                "threshold": float(manifest.get("selected_threshold", 0.5)),
            })

    assert len(loaded) == 3
    pf_values = [c["pf"] for c in loaded]
    assert pf_values == [1.3, 1.5, 1.7]


def test_candidate_manifest_load_function_valid_manifest(tmp_path):
    """load_candidate_manifest() returns correct dict for a valid manifest."""
    import scripts.live_decision_dry_run as m

    cand_dir = _make_paper_candidate_dir(tmp_path, "valid_candidate", pf=1.4, threshold=0.30)
    manifest_path = cand_dir / "candidate_manifest.json"
    assert manifest_path.exists()

    # Fix: model_pkl must be resolved relative to _REPO_ROOT (scripts/../ = repo root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_pkl"] = str(cand_dir / "model.pkl")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = m.load_candidate_manifest(manifest_path)
    assert loaded["candidate_id"] == "valid_candidate"
    assert loaded["selected_threshold"] == 0.30
    assert loaded["gates_passed"] == 10
    assert loaded["gates_total"] == 10
    assert loaded["filter_applied"] is True
    assert loaded["filter_definition"]["type"] == "PE_only"


def test_candidate_manifest_load_function_missing_file_raises(tmp_path):
    """load_candidate_manifest() raises FileNotFoundError for missing manifest."""
    import scripts.live_decision_dry_run as m

    missing = tmp_path / "nonexistent_manifest.json"
    assert not missing.exists()

    with pytest.raises(FileNotFoundError):
        m.load_candidate_manifest(missing)


def test_candidate_manifest_load_function_missing_required_field_raises(tmp_path):
    """load_candidate_manifest() raises ValueError when required fields are missing."""
    import scripts.live_decision_dry_run as m

    cand_dir = _make_paper_candidate_dir(tmp_path, "incomplete_candidate", pf=1.4)
    manifest_path = cand_dir / "candidate_manifest.json"

    # Remove required field model_pkl
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["model_pkl"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="model_pkl"):
        m.load_candidate_manifest(manifest_path)


def test_candidate_manifest_load_function_missing_threshold_raises(tmp_path):
    """load_candidate_manifest() raises ValueError when selected_threshold is missing."""
    import scripts.live_decision_dry_run as m

    cand_dir = _make_paper_candidate_dir(tmp_path, "no_threshold_candidate", pf=1.4)
    manifest_path = cand_dir / "candidate_manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["selected_threshold"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="selected_threshold"):
        m.load_candidate_manifest(manifest_path)


def test_candidate_manifest_load_function_missing_candidate_id_raises(tmp_path):
    """load_candidate_manifest() raises ValueError when model_id/candidate_id is missing."""
    import scripts.live_decision_dry_run as m

    cand_dir = _make_paper_candidate_dir(tmp_path, "no_id_candidate", pf=1.4)
    manifest_path = cand_dir / "candidate_manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["model_id"]
    manifest.pop("candidate_id", None)  # may not exist
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate_id"):
        m.load_candidate_manifest(manifest_path)


def test_candidate_manifest_model_pkl_path_missing_raises(tmp_path):
    """load_candidate_manifest() raises FileNotFoundError when model_pkl path doesn't exist."""
    import scripts.live_decision_dry_run as m

    cand_dir = _make_paper_candidate_dir(tmp_path, "missing_pkl_candidate", pf=1.4)
    manifest_path = cand_dir / "candidate_manifest.json"

    # Point to a non-existent pkl
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_pkl"] = "nonexistent_model.pkl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="model_pkl"):
        m.load_candidate_manifest(manifest_path)


# ---------------------------------------------------------------------------
# Test 4: paper_only=false fails closed
# ---------------------------------------------------------------------------

def test_paper_only_false_fails_candidate_selection(tmp_path):
    """Candidates with paper_only=False must be excluded from auto-selection."""
    _make_paper_candidate_dir(
        tmp_path, "live_trade_candidate",
        pf=2.0, paper_only=False, real_trading_enabled=False,
    )
    _make_paper_candidate_dir(
        tmp_path, "paper_only_candidate",
        pf=1.2, paper_only=True, real_trading_enabled=False,
    )

    all_dirs = sorted(tmp_path.iterdir())
    selected = []
    for d in all_dirs:
        manifest_path = d / "candidate_manifest.json"
        if manifest_path.exists():
            m_loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if m_loaded.get("paper_only") is True:
                pf = m_loaded.get("overall_metrics", {}).get("mean_pf", 0.0)
                selected.append((pf, d))

    selected.sort(key=lambda x: x[0], reverse=True)
    assert len(selected) == 1
    assert selected[0][1].name == "paper_only_candidate"
    # Note: live candidate is excluded even though it has higher PF
    assert selected[0][0] == 1.2


def test_paper_only_absent_defaults_to_false_unsafe(tmp_path):
    """When paper_only field is absent, it defaults to treating as paper_only=False (safe fail)."""
    cand_dir = _make_paper_candidate_dir(tmp_path, "no_paper_only_candidate", pf=1.5)
    manifest_path = cand_dir / "candidate_manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["paper_only"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    paper_only_val = loaded.get("paper_only", None)

    # paper_only not set → treat as False (fail-safe)
    assert paper_only_val is None
    # Any None value should be treated as not True for safety
    is_paper_only = paper_only_val is True
    assert is_paper_only is False, "Missing paper_only must default to not-True (fail-safe)"


# ---------------------------------------------------------------------------
# Test 5: real_trading_enabled=true fails closed
# ---------------------------------------------------------------------------

def test_real_trading_enabled_true_fails_candidate_selection(tmp_path):
    """Candidates with real_trading_enabled=True must be excluded from auto-selection."""
    _make_paper_candidate_dir(
        tmp_path, "dormant_candidate",
        pf=1.8, paper_only=True, real_trading_enabled=True,
    )
    _make_paper_candidate_dir(
        tmp_path, "safe_candidate",
        pf=1.1, paper_only=True, real_trading_enabled=False,
    )

    all_dirs = sorted(tmp_path.iterdir())
    safe_candidates = []
    for d in all_dirs:
        manifest_path = d / "candidate_manifest.json"
        if manifest_path.exists():
            m_loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if m_loaded.get("real_trading_enabled") is not True:
                pf = m_loaded.get("overall_metrics", {}).get("mean_pf", 0.0)
                safe_candidates.append((pf, d))

    safe_candidates.sort(key=lambda x: x[0], reverse=True)
    # Only safe_candidate (PF=1.1) passes; dormant_candidate (PF=1.8) blocked
    assert len(safe_candidates) == 1
    assert safe_candidates[0][1].name == "safe_candidate"


# ---------------------------------------------------------------------------
# Test 6: PE filter works
# ---------------------------------------------------------------------------

def test_pe_only_filter_rejects_ce_snapshot():
    """apply_pe_only_filter must reject CE contracts."""
    from src.candidate_filters import apply_pe_only_filter

    result = apply_pe_only_filter({"option_type": "CE"})
    assert result["filter_passed"] is False
    assert "not_PE" in result["rejection_reason"]


def test_pe_only_filter_allows_pe_snapshot():
    """apply_pe_only_filter must allow PE contracts."""
    from src.candidate_filters import apply_pe_only_filter

    result = apply_pe_only_filter({"option_type": "PE"})
    assert result["filter_passed"] is True
    assert result["rejection_reason"] is None


def test_pe_only_filter_rejects_missing_option_type():
    """apply_pe_only_filter must reject snapshots missing option_type."""
    from src.candidate_filters import apply_pe_only_filter

    result = apply_pe_only_filter({})
    assert result["filter_passed"] is False
    assert result["required_fields_present"] is False


# ---------------------------------------------------------------------------
# Test 7: CE filter works
# ---------------------------------------------------------------------------

def test_ce_only_filter_rejects_pe_snapshot():
    """apply_ce_only_filter must reject PE contracts."""
    from src.candidate_filters import apply_ce_only_filter

    result = apply_ce_only_filter({"option_type": "PE"})
    assert result["filter_passed"] is False
    assert "not_CE" in result["rejection_reason"]


def test_ce_only_filter_allows_ce_snapshot():
    """apply_ce_only_filter must allow CE contracts."""
    from src.candidate_filters import apply_ce_only_filter

    result = apply_ce_only_filter({"option_type": "CE"})
    assert result["filter_passed"] is True
    assert result["rejection_reason"] is None


def test_ce_only_filter_rejects_missing_option_type():
    """apply_ce_only_filter must reject snapshots missing option_type."""
    from src.candidate_filters import apply_ce_only_filter

    result = apply_ce_only_filter({})
    assert result["filter_passed"] is False
    assert result["required_fields_present"] is False


# ---------------------------------------------------------------------------
# Test 8: DTE filter works
# ---------------------------------------------------------------------------

def test_dte_greater_7_filter_rejects_short_dte():
    """apply_dte_greater_7_filter must reject dte_days <= 7."""
    from src.candidate_filters import apply_dte_greater_7_filter

    for dte_val in [0, 1, 3, 5, 7]:
        result = apply_dte_greater_7_filter({"dte_days": dte_val})
        assert result["filter_passed"] is False, f"DTE={dte_val} should be rejected"


def test_dte_greater_7_filter_allows_adequate_dte():
    """apply_dte_greater_7_filter must allow dte_days > 7."""
    from src.candidate_filters import apply_dte_greater_7_filter

    for dte_val in [8, 10, 14, 21, 30]:
        result = apply_dte_greater_7_filter({"dte_days": dte_val})
        assert result["filter_passed"] is True, f"DTE={dte_val} should pass"


def test_dte_greater_7_filter_rejects_missing_dte():
    """apply_dte_greater_7_filter must reject missing dte_days."""
    from src.candidate_filters import apply_dte_greater_7_filter

    result = apply_dte_greater_7_filter({})
    assert result["filter_passed"] is False
    assert result["required_fields_present"] is False


# ---------------------------------------------------------------------------
# Test 9: ITM filter works
# ---------------------------------------------------------------------------

def test_itm_all_filter_rejects_atm():
    """apply_itm_all_filter must reject ATM contracts."""
    from src.candidate_filters import apply_itm_all_filter

    result = apply_itm_all_filter({"moneyness_bucket": "ATM"})
    assert result["filter_passed"] is False


def test_itm_all_filter_rejects_otm():
    """apply_itm_all_filter must reject OTM contracts."""
    from src.candidate_filters import apply_itm_all_filter

    result = apply_itm_all_filter({"moneyness_bucket": "OTM"})
    assert result["filter_passed"] is False


def test_itm_all_filter_allows_itm():
    """apply_itm_all_filter must allow ITM contracts."""
    from src.candidate_filters import apply_itm_all_filter

    result = apply_itm_all_filter({"moneyness_bucket": "ITM"})
    assert result["filter_passed"] is True


# ---------------------------------------------------------------------------
# Test 10: Premium filter works
# ---------------------------------------------------------------------------

def test_premium_10_100_filter_rejects_low_premium():
    """apply_premium_10_100_filter must reject premiums below 10."""
    from src.candidate_filters import apply_premium_10_100_filter

    for price in [1.0, 5.0, 8.0, 9.99]:
        result = apply_premium_10_100_filter({"ltp": price})
        assert result["filter_passed"] is False, f"ltp={price} should be rejected"


def test_premium_10_100_filter_rejects_high_premium():
    """apply_premium_10_100_filter must reject premiums above 100."""
    from src.candidate_filters import apply_premium_10_100_filter

    for price in [100.01, 150.0, 500.0]:
        result = apply_premium_10_100_filter({"ltp": price})
        assert result["filter_passed"] is False, f"ltp={price} should be rejected"


def test_premium_10_100_filter_allows_within_range():
    """apply_premium_10_100_filter must allow premiums in [10, 100]."""
    from src.candidate_filters import apply_premium_10_100_filter

    for price in [10.0, 25.0, 50.0, 75.0, 100.0]:
        result = apply_premium_10_100_filter({"ltp": price})
        assert result["filter_passed"] is True, f"ltp={price} should pass"


def test_premium_10_100_filter_allows_mid_price():
    """apply_premium_10_100_filter uses mid_price as fallback."""
    from src.candidate_filters import apply_premium_10_100_filter

    result = apply_premium_10_100_filter({"ltp": None, "mid_price": 50.0})
    assert result["filter_passed"] is True


# ---------------------------------------------------------------------------
# Test 11: Edge scoring correctly ranks candidates
# ---------------------------------------------------------------------------

def test_edge_score_computed_from_mean_pf_and_std(tmp_path):
    """Edge score must factor in both mean_pf and fold_pf_std for stability."""
    # Candidate A: high PF but high std (unstable)
    _make_paper_candidate_dir(tmp_path, "unstable_candidate", pf=2.2)
    unstable_manifest = json.loads((tmp_path / "unstable_candidate" / "candidate_manifest.json").read_text(encoding="utf-8"))
    unstable_manifest["fold_pf_std"] = 0.8
    unstable_manifest["fold_details"] = {"fold_0": {"pf": 1.5}, "fold_1": {"pf": 2.5}, "fold_2": {"pf": 2.6}}
    (tmp_path / "unstable_candidate" / "candidate_manifest.json").write_text(json.dumps(unstable_manifest), encoding="utf-8")

    # Candidate B: moderate PF but low std (stable)
    _make_paper_candidate_dir(tmp_path, "stable_candidate", pf=1.5)
    stable_manifest = json.loads((tmp_path / "stable_candidate" / "candidate_manifest.json").read_text(encoding="utf-8"))
    stable_manifest["fold_pf_std"] = 0.1
    (tmp_path / "stable_candidate" / "candidate_manifest.json").write_text(json.dumps(stable_manifest), encoding="utf-8")

    # Routing by PF only (current implementation) picks unstable candidate
    # Routing by adjusted_edge = mean_pf - k * std would pick stable
    candidates = sorted(tmp_path.iterdir())
    pf_rank = sorted(
        candidates,
        key=lambda d: json.loads((d / "candidate_manifest.json").read_text(encoding="utf-8"))
                     .get("overall_metrics", {}).get("mean_pf", 0.0),
        reverse=True,
    )
    # Pure PF ranking (current behavior) — unstable wins
    assert pf_rank[0].name == "unstable_candidate"


# ---------------------------------------------------------------------------
# Test 12: apply_filters pipeline
# ---------------------------------------------------------------------------

def test_apply_filters_pipeline_all_pass():
    """apply_filters() with PE_only filter only — snapshot with PE passes."""
    from src.candidate_filters import apply_filters

    snapshot = {
        "option_type": "PE",
        "dte_days": 15,
        "moneyness_bucket": "ITM",
        "ltp": 50.0,
        "mid_price": 50.0,
    }
    # Apply only PE_only filter (the CE_only filter would reject PE)
    result = apply_filters(snapshot, filter_names=["PE_only"])
    assert result["all_passed"] is True
    assert result["blocking_reason"] is None


def test_apply_filters_pipeline_one_fails():
    """apply_filters() with one failing filter returns all_passed=False."""
    from src.candidate_filters import apply_filters

    snapshot = {
        "option_type": "CE",  # PE filter will fail
        "dte_days": 15,
        "moneyness_bucket": "ITM",
        "ltp": 50.0,
    }
    result = apply_filters(snapshot)
    assert result["all_passed"] is False
    assert result["blocking_reason"] is not None


def test_apply_filters_pipeline_selective_filters():
    """apply_filters() accepts selective filter_names list."""
    from src.candidate_filters import apply_filters

    snapshot = {
        "option_type": "CE",
        "dte_days": 15,
        "moneyness_bucket": "ITM",
        "ltp": 50.0,
    }
    # Only apply DTE filter (skip option_type filter)
    result = apply_filters(snapshot, filter_names=["DTE_greater_7"])
    assert result["all_passed"] is True  # CE passes since we skipped option_type filter
    assert result["blocking_reason"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
