"""Tests covering recent fixes to the retrain-pipeline scripts.

These tests verify:

  1. PRIMARY_LABELS in scripts/retrain_all_edge_models.py includes the new
     cost-aware labels (and still contains the legacy ones).
  2. choose_labels() returns a (usable, skipped) tuple and that constant /
     out-of-range labels are correctly rejected with non-empty reasons.
  3. _dataset_candidate_report() rejects <64KB files and accepts files
     above that threshold.
  4. dataset_validity() in scripts/build_option_edge_dataset.py picks
     a usable cost-aware label when strict option columns are missing,
     and reports failures when no label is available.
  5. detect_column_groups() / select_feature_columns() include legitimate
     features and exclude forbidden / scoring columns.
  6. The retrain_all_models_summary.json shape matches the documented
     schema (fixture-only check).
  7. The retrain script never references live broker execution calls.

The tests are fully hermetic: they build tiny synthetic DataFrames in
memory and use tmp_path for any on-disk artifacts. They do NOT load the
real 405k-row canonical dataset and they do NOT train any real model.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
RETRAIN_PATH = SCRIPTS_DIR / "retrain_all_edge_models.py"
BUILD_PATH = SCRIPTS_DIR / "build_option_edge_dataset.py"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain  # noqa: E402
import build_option_edge_dataset as build_ds  # noqa: E402


# ---------------------------------------------------------------------------
# Test 1: PRIMARY_LABELS includes the new cost-aware labels.
# ---------------------------------------------------------------------------
def test_primary_labels_includes_cost_aware_labels() -> None:
    primary = list(retrain.PRIMARY_LABELS)
    required_new = [
        "strong_profitable_trade_label",
        "cost_survivor_label",
        "strong_profitable_trade_label_v2",
        "cost_survivor_label_v2",
        "weak_trade_label",
        "no_trade_label",
        "high_conviction_trade_label",
        "paper_candidate_label",
        "high_conviction_trade_label_v2",
        "paper_candidate_label_v2",
    ]
    for label in required_new:
        assert label in primary, f"PRIMARY_LABELS missing cost-aware entry: {label}"
    # Regression: the legacy labels must remain.
    assert "profitable_trade_label" in primary
    assert "avoid_trade_label" in primary


# ---------------------------------------------------------------------------
# Test 2: choose_labels() returns a (usable, skipped) tuple and surfaces
# every disqualification with a non-empty reason.
# ---------------------------------------------------------------------------
def test_choose_labels_returns_tuple_with_skipped_reasons() -> None:
    n = 500
    rng = np.random.default_rng(11)
    # profitable: 50/50 binary, passes all gates.
    profitable = np.tile([0.0, 1.0], n // 2)
    # avoid_trade_label: all NaN, fails the "too_few_non_null" gate.
    avoid = np.full(n, np.nan)
    # high_conviction_trade_label: all 0, fails the "unique_non_null" / constant gate.
    high_conv = np.zeros(n)
    # weak_trade_label: < 0.5% positives (1 of 500 = 0.2%), fails the
    # MIN_POSITIVE_SHARE gate. 1/500 = 0.002 < 0.005.
    weak = np.zeros(n)
    weak[0] = 1.0
    # no_trade_label: > 95% positives (499/500 = 99.8%), fails the
    # MAX_POSITIVE_SHARE gate.
    no_trade = np.ones(n)
    no_trade[0] = 0.0
    df = pd.DataFrame({
        "profitable_trade_label": profitable,
        "avoid_trade_label": avoid,
        "high_conviction_trade_label": high_conv,
        "weak_trade_label": weak,
        "no_trade_label": no_trade,
    })

    result = retrain.choose_labels(df)
    assert isinstance(result, tuple) and len(result) == 2, (
        f"choose_labels must return a (usable, skipped) tuple, got: {type(result)!r}"
    )
    usable, skipped = result
    assert isinstance(usable, list)
    assert isinstance(skipped, list)
    assert "profitable_trade_label" in usable, (
        "profitable_trade_label (50/50 split) should be the sole usable label"
    )

    # Build a {name: reason} map for fast lookup.
    skip_reasons = {
        entry.get("label_name"): entry.get("reason", "")
        for entry in skipped
    }
    expected_skipped = {
        "avoid_trade_label",
        "high_conviction_trade_label",
        "weak_trade_label",
        "no_trade_label",
    }
    missing = expected_skipped - set(skip_reasons.keys())
    assert not missing, f"Expected these labels to be in skip report: {missing}"
    for name in expected_skipped:
        reason = skip_reasons[name]
        assert isinstance(reason, str) and reason.strip(), (
            f"Skip reason for {name!r} must be a non-empty string, got: {reason!r}"
        )


# ---------------------------------------------------------------------------
# Test 3: choose_labels() rejects a constant column.
# ---------------------------------------------------------------------------
def test_choose_labels_rejects_constant_column() -> None:
    df = pd.DataFrame({"profitable_trade_label": [0.0] * 100})
    usable, skipped = retrain.choose_labels(df)
    assert "profitable_trade_label" not in usable, (
        "A constant-zero column must NOT be marked usable"
    )
    matched = next(
        (entry for entry in skipped if entry.get("label_name") == "profitable_trade_label"),
        None,
    )
    assert matched is not None, "Constant column should appear in skip report"
    reason_lower = str(matched.get("reason", "")).lower()
    assert any(
        token in reason_lower
        for token in ("constant", "unique", "degenerate", "out_of_range")
    ), f"Reason for constant column must mention one of constant/unique/degenerate/out_of_range, got: {matched.get('reason')!r}"


# ---------------------------------------------------------------------------
# Test 4: _dataset_candidate_report() rejects <64KB files.
# ---------------------------------------------------------------------------
def test_dataset_candidate_report_rejects_small_files(tmp_path: Path) -> None:
    tiny = tmp_path / "tiny.csv"
    # Header + a single data row → well under 64 KB.
    tiny.write_text("timestamp,profitable_trade_label\n2024-01-01,1\n", encoding="utf-8")
    assert tiny.stat().st_size < 64_000
    report = retrain._dataset_candidate_report(tiny)
    assert report["status"] == "REJECTED"
    assert "too_small_to_be_training_data" in report["reasons"], (
        f"Small files must be rejected with 'too_small_to_be_training_data', got reasons: {report['reasons']}"
    )


# ---------------------------------------------------------------------------
# Test 5: _dataset_candidate_report() does not reject >= 64KB files on size.
# ---------------------------------------------------------------------------
def test_dataset_candidate_report_accepts_real_size(tmp_path: Path) -> None:
    csv_path = tmp_path / "nifty_option_chain_enriched.csv"
    rows = 2000
    cols = ["timestamp", "ltp", "bs_iv", "bs_delta", "bs_gamma", "bs_vega", "net_forward_return", "profitable_trade_label"]
    rng = np.random.default_rng(42)
    base = {
        "timestamp": pd.date_range("2024-01-01", periods=rows, freq="min"),
        "ltp": rng.normal(100.0, 5.0, rows),
        "bs_iv": rng.normal(0.2, 0.05, rows),
        "bs_delta": rng.normal(0.5, 0.1, rows),
        "bs_gamma": rng.normal(0.02, 0.005, rows),
        "bs_vega": rng.normal(1.0, 0.2, rows),
        "net_forward_return": rng.normal(0.0, 1.0, rows),
        "profitable_trade_label": rng.integers(0, 2, rows).astype(float),
    }
    pd.DataFrame(base)[cols].to_csv(csv_path, index=False)
    # Pad to ~100KB by appending a numeric pad column full of values.
    with csv_path.open("a", encoding="utf-8") as fh:
        for _ in range(50):
            fh.write("# pad" + ("x" * 1500) + "\n")
    assert csv_path.stat().st_size >= 64_000, (
        f"Synthetic CSV must be at least 64KB to exercise the size gate, got {csv_path.stat().st_size} bytes"
    )
    report = retrain._dataset_candidate_report(csv_path)
    assert "too_small_to_be_training_data" not in report["reasons"], (
        f"100KB file should not be rejected on size, got reasons: {report['reasons']}"
    )


# ---------------------------------------------------------------------------
# Test 6: dataset_validity() picks a cost-aware label when strict columns
# are missing.
# ---------------------------------------------------------------------------
def test_dataset_validity_accepts_cost_aware_label_when_strict_columns_missing() -> None:
    n = 1000
    rng = np.random.default_rng(7)
    # dataset_validity() requires non-null coverage >= MIN_COVERAGE (0.80).
    # Use 90% non-null with a ~30% positive rate, comfortably above
    # MIN_POSITIVE_SHARE (0.03) and well above the 80% coverage floor.
    label = np.where(rng.random(n) < 0.10, np.nan, rng.integers(0, 2, n).astype(float))
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-06-01", periods=n, freq="min"),
        "ltp": rng.normal(100.0, 5.0, n),
        "profitable_trade_label": label,
        # dataset_validity() also enforces selected_option_symbol coverage
        # >= 80% independently of the label; include it so the only
        # gate under test is the cost-aware label fallback.
        "selected_option_symbol": np.array(["NIFTY24JUN18000CE"] * n),
    })
    # Sanity: no strict option columns.
    assert "option_ltp_at_signal" not in df.columns
    assert "cost_adjusted_success_15m" not in df.columns

    result = build_ds.dataset_validity(df)
    assert result["status"] == "DATASET_READY_FOR_RESEARCH_RETRAINING", (
        f"Cost-aware fallback should make the dataset READY, got: {result}"
    )
    allowed = {
        "profitable_trade_label",
        "strong_profitable_trade_label",
        "cost_survivor_label",
        "strong_profitable_trade_label_v2",
        "cost_survivor_label_v2",
    }
    chosen = result.get("chosen_label") or result.get("chosen_binary_label")
    assert chosen in allowed, (
        f"chosen_label should be one of {allowed}, got: {chosen!r}"
    )
    assert chosen == "profitable_trade_label", (
        f"profitable_trade_label is the only label present in this synthetic frame; expected it to be chosen, got: {chosen!r}"
    )
    assert float(result.get("label_coverage", 0.0)) >= build_ds.MIN_COVERAGE


# ---------------------------------------------------------------------------
# Test 7: dataset_validity() fails when no usable label column is present.
# ---------------------------------------------------------------------------
def test_dataset_validity_fails_when_no_usable_label() -> None:
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-06-01", periods=500, freq="min"),
        "ltp": np.random.default_rng(0).normal(100.0, 5.0, 500),
    })
    result = build_ds.dataset_validity(df)
    assert result["status"] != "DATASET_READY_FOR_RESEARCH_RETRAINING", (
        "A frame with no label column cannot be marked ready"
    )
    failures = result.get("failures", [])
    assert isinstance(failures, list) and len(failures) > 0, (
        f"Failures list must be non-empty when dataset is not ready, got: {result}"
    )


# ---------------------------------------------------------------------------
# Test 8: detect_column_groups() excludes forbidden / scoring columns
# from the input_features list.
# ---------------------------------------------------------------------------
def test_feature_selection_excludes_forbidden_columns() -> None:
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-06-01", periods=200, freq="min"),
        "feature_a": np.linspace(0.0, 1.0, 200),
        "feature_b": np.random.default_rng(1).normal(0.0, 1.0, 200),
        "future_close": np.linspace(100.0, 110.0, 200),
        "gross_forward_return": np.random.default_rng(2).normal(0.0, 1.0, 200),
        "net_forward_return": np.random.default_rng(3).normal(0.0, 1.0, 200),
        "profitable_trade_label": np.random.default_rng(4).integers(0, 2, 200).astype(float),
        "return_to_cost_ratio": np.random.default_rng(5).normal(1.0, 0.1, 200),
        "expected_return_after_cost": np.random.default_rng(6).normal(0.5, 0.1, 200),
        "cost_return_units_estimated": np.random.default_rng(7).normal(0.0, 0.5, 200),
    })
    groups = retrain.detect_column_groups(df)
    input_features = list(groups["input_features"])
    forbidden = list(groups["forbidden_feature_columns"])

    # Allowed features must remain in the training input.
    assert "feature_a" in input_features
    assert "feature_b" in input_features

    # Forbidden / scoring columns must NOT be features.
    forbidden_present = {
        "future_close",
        "gross_forward_return",
        "net_forward_return",
        "profitable_trade_label",
        "return_to_cost_ratio",
        "expected_return_after_cost",
        "cost_return_units_estimated",
    }
    for column in forbidden_present:
        assert column not in input_features, (
            f"{column!r} must be excluded from input_features, got: {input_features}"
        )

    # forbidden_feature_columns list should mention the forbidden ones present.
    for column in forbidden_present:
        assert column in forbidden, (
            f"forbidden_feature_columns missing {column!r}; got: {forbidden}"
        )


# ---------------------------------------------------------------------------
# Test 9: detect_column_groups() keeps realized-vol features (the
# realized_* / ret_* prefixes are whitelisted and must not be flagged).
# ---------------------------------------------------------------------------
def test_feature_selection_keeps_realized_vol_features() -> None:
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-06-01", periods=200, freq="min"),
        "realized_vol_30": np.random.default_rng(8).normal(0.2, 0.05, 200),
        "realized_vol_percentile_60": np.random.default_rng(9).uniform(0.0, 1.0, 200),
        "ret_3": np.random.default_rng(10).normal(0.0, 0.01, 200),
        "ret_mean": np.random.default_rng(11).normal(0.0, 0.005, 200),
        "profitable_trade_label": np.random.default_rng(12).integers(0, 2, 200).astype(float),
    })
    groups = retrain.detect_column_groups(df)
    input_features = list(groups["input_features"])
    for name in ("realized_vol_30", "realized_vol_percentile_60", "ret_3", "ret_mean"):
        assert name in input_features, (
            f"Whitelisted feature {name!r} should remain in input_features, got: {input_features}"
        )


# ---------------------------------------------------------------------------
# Test 10: retrain_all_models_summary.json schema check.
# This is a fixture-only test: we do not run a real retrain. We just
# write a synthetic file with the documented shape and assert that the
# pipeline's schema-documentation (the list of required keys) matches it.
# ---------------------------------------------------------------------------
def test_summary_json_schema_matches_spec(tmp_path: Path) -> None:
    required_keys = {
        "dataset_path",
        "artifact_dir",
        "row_count",
        "date_range",
        "labels_considered",
        "labels_skipped",
        "models_planned",
        "models_trained",
        "models_skipped",
        "models_failed",
        "leakage_columns_dropped",
        "validation_strategy",
        "walk_forward_folds",
        "completed_ok",
        "remaining_blockers",
    }
    fixture = {key: None for key in required_keys}
    fixture.update({
        "dataset_path": str(tmp_path / "fixture_dataset.csv"),
        "artifact_dir": str(tmp_path / "fixture_artifact_dir"),
        "row_count": 100,
        "date_range": ["2024-06-01", "2024-06-02"],
        "labels_considered": ["profitable_trade_label"],
        "labels_skipped": [],
        "models_planned": ["logistic_regression", "random_forest"],
        "models_trained": ["logistic_regression"],
        "models_skipped": [],
        "models_failed": [],
        "leakage_columns_dropped": [],
        "validation_strategy": "purged_embargoed_walk_forward",
        "walk_forward_folds": 5,
        "completed_ok": True,
        "remaining_blockers": [],
    })
    summary_path = tmp_path / "retrain_all_models_summary.json"
    summary_path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")

    parsed = json.loads(summary_path.read_text(encoding="utf-8"))
    missing = required_keys - parsed.keys()
    assert not missing, f"Summary JSON missing required keys: {missing}"

    # The production writer in scripts/retrain_all_edge_models.py must
    # also reference the same key names. Verify the writer code mentions
    # each key (case-sensitive, exact match).
    source = RETRAIN_PATH.read_text(encoding="utf-8")
    for key in required_keys:
        assert f'"{key}"' in source, (
            f"Expected retrain_all_edge_models.py to write key {key!r} in summary JSON, but it was not found."
        )


# ---------------------------------------------------------------------------
# Test 11: The retrain script must never directly call live broker
# execution. This enforces the "do not touch broker execution" invariant.
# ---------------------------------------------------------------------------
def test_no_broker_or_live_trading_imports_in_retrain_script() -> None:
    source = RETRAIN_PATH.read_text(encoding="utf-8")
    forbidden_substrings = [
        "place_order",
        "submit_order",
        "dhan_client.place",
        "mstock_client.place",
    ]
    # Strip the docstring of the file (very first triple-quoted string) and
    # any triple-quoted string that begins with "test" to allow tests
    # guarding this invariant.
    cleaned = source
    cleaned = re.sub(r'""".*?"""', '"""DOCSTRING"""', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"'''.*?'''", "'''DOCSTRING'''", cleaned, flags=re.DOTALL)

    for needle in forbidden_substrings:
        assert needle not in cleaned, (
            f"retrain_all_edge_models.py must not contain a live-broker call to {needle!r}."
        )


def test_chronological_split_ordering() -> None:
    """Verify _build_holdout_split_from_timestamps produces non-overlapping ordered train/val/test."""
    import pandas as pd
    from retrain_all_edge_models import _build_holdout_split_from_timestamps

    dates = [f"2026-06-{d:02d} 10:00:00" for d in range(1, 11)] * 50
    ts = pd.Series(sorted(pd.to_datetime(dates)), name="timestamp")

    split = _build_holdout_split_from_timestamps(ts, train_ratio=0.7, validation_ratio=0.15)

    train_ts = ts.iloc[split["train"]].sort_values()
    val_ts = ts.iloc[split["validation"]].sort_values()
    test_ts = ts.iloc[split["test"]].sort_values()

    assert train_ts.max() < val_ts.min(), (
        f"Train max {train_ts.max()} is not strictly before Val min {val_ts.min()}"
    )
    assert val_ts.max() < test_ts.min(), (
        f"Val max {val_ts.max()} is not strictly before Test min {test_ts.min()}"
    )
    all_idx = set(split["train"]) | set(split["validation"]) | set(split["test"])
    assert len(all_idx) == len(ts), f"Split indices do not cover all rows: {len(all_idx)} vs {len(ts)}"
    assert len(set(split["train"]) & set(split["validation"])) == 0, "Train/Val overlap"
    assert len(set(split["validation"]) & set(split["test"])) == 0, "Val/Test overlap"
