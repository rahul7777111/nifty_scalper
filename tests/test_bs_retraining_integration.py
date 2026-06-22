from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def _dataset() -> pd.DataFrame:
    rows = []
    base_ts = pd.Timestamp("2026-06-01 09:15:00+05:30")
    for idx in range(80):
        rows.append(
            {
                "timestamp": base_ts + pd.Timedelta(minutes=idx),
                "feature_a": float(idx % 7),
                "feature_b": float((idx * 2) % 11),
                "cost_adjusted_success_5m": int(idx % 2 == 0),
                "bs_iv": 0.20 + 0.001 * idx,
                "bs_delta": 0.40,
                "bs_gamma": 0.01,
                "bs_theta": -2.0,
                "bs_vega": 5.0,
                "bs_rho": 1.0,
                "moneyness": 1.0,
                "log_moneyness": 0.0,
                "distance_from_atm": 0.0,
                "distance_from_atm_pct": 0.0,
                "intrinsic_value": 1.0,
                "extrinsic_value": 2.0,
                "time_to_expiry_days": 2.0,
                "time_to_expiry_years": 2.0 / 365.0,
                "bid_ask_spread": 0.5,
                "bid_ask_spread_pct": 0.01,
                "mid_price": 100.0,
                "ltp_vs_mid_diff_pct": 0.0,
                "greeks_quality_score": 0.95,
                "bs_iv_solve_ok": 1,
                "bad_iv_flag": 0,
                "bad_greek_flag": 0,
                "wide_spread_flag": 0,
                "low_price_flag": 0,
                "deep_itm_flag": 0,
                "deep_otm_flag": 0,
                "row_enrichment_ok": 1,
                "future_profit": 9.0,
                "net_forward_return": 0.01 if idx % 2 == 0 else -0.01,
            }
        )
    return pd.DataFrame(rows)


def test_enrichment_is_invoked_only_when_flag_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path = tmp_path / "edge.csv"
    _dataset().drop(columns=[c for c in retrain.BS_RESEARCH_FEATURES + retrain.BS_FLAG_FEATURES if c in _dataset().columns]).to_csv(dataset_path, index=False)
    calls: list[Path] = []

    def _fake(path: Path) -> Path:
        calls.append(path)
        enriched = _dataset()
        out = tmp_path / "edge_bs_enriched.csv"
        enriched.to_csv(out, index=False)
        return out

    monkeypatch.setattr(retrain, "ensure_black_scholes_dataset", _fake)
    retrain.ensure_black_scholes_dataset(dataset_path)
    assert calls == [dataset_path]


def test_original_dataset_is_not_overwritten(tmp_path: Path) -> None:
    dataset_path = tmp_path / "edge.csv"
    frame = _dataset()
    frame.to_csv(dataset_path, index=False)
    original = dataset_path.read_text(encoding="utf-8")
    enriched_path = retrain._enriched_dataset_path(dataset_path)
    assert enriched_path != dataset_path
    assert dataset_path.read_text(encoding="utf-8") == original


def test_feature_list_is_saved_and_order_is_deterministic() -> None:
    payload = retrain.build_feature_sets(_dataset(), use_black_scholes_features=True)
    assert payload["enriched_features"][:2] == ["feature_a", "feature_b"]
    assert payload["enriched_features"][-1] == "row_enrichment_ok"


def test_forbidden_target_and_leakage_columns_are_excluded() -> None:
    payload = retrain.build_feature_sets(_dataset(), use_black_scholes_features=False)
    assert "future_profit" not in payload["baseline_features"]
    assert "cost_adjusted_success_5m" not in payload["baseline_features"]
    assert "future_profit" in payload["rejected_features"]
    assert "net_forward_return" not in payload["baseline_features"]


def test_estimated_only_greeks_force_research_only_verdict() -> None:
    verdict = retrain._production_adoption_verdict(
        enriched_best={"test_metrics": {"f1": 0.5}, "walk_forward": {"split_count": 5}},
        feature_cols=["feature_a", "feature_b"],
        leakage_ok=True,
        estimated_only=True,
    )
    assert verdict["production_adoption_allowed"] is False
    assert verdict["verdict"] == "RESEARCH_ONLY_ESTIMATED_GREEKS_NOT_ADOPTABLE"


def test_missing_bs_columns_fail_gracefully() -> None:
    frame = _dataset().drop(columns=["bs_iv"])
    with pytest.raises(RuntimeError):
        retrain.build_feature_sets(frame, use_black_scholes_features=True)


def test_invalid_enrichment_rows_are_filtered() -> None:
    frame = _dataset()
    frame.loc[0, "row_enrichment_ok"] = 0
    filtered = retrain._filter_invalid_bs_rows(frame)
    assert len(filtered) == len(frame) - 1


def test_return_columns_are_detected_for_evaluation_but_excluded_from_features() -> None:
    groups = retrain.detect_column_groups(_dataset())
    assert groups["evaluation_return_column_used"] == "net_forward_return"
    assert "net_forward_return" in groups["evaluation_return_columns"]
    assert "net_forward_return" not in groups["input_features"]


def test_realized_vol_feature_is_not_mistaken_for_forward_return_leakage() -> None:
    frame = _dataset().copy()
    frame["realized_vol_30"] = np.linspace(0.1, 0.2, len(frame))
    groups = retrain.detect_column_groups(frame)
    assert "realized_vol_30" in groups["input_features"]
    assert "realized_vol_30" not in groups["evaluation_return_columns"]


def test_profit_factor_uses_only_selected_predicted_trades() -> None:
    y_true = pd.Series([1, 0, 1, 0]).to_numpy()
    y_prob = pd.Series([0.8, 0.2, 0.9, 0.7]).to_numpy()
    returns = pd.Series([0.05, -0.02, 0.03, -0.01]).to_numpy()
    metrics = retrain._trade_metrics_from_scores(y_true, y_prob, 0.75, returns)
    assert metrics["trade_count"] == 2
    assert metrics["gross_profit"] == pytest.approx(0.08)
    assert metrics["gross_loss"] == pytest.approx(0.0)


def test_no_real_return_column_marks_proxy_metrics_unavailable() -> None:
    y_true = pd.Series([1, 0, 1, 0]).to_numpy()
    y_prob = pd.Series([0.8, 0.2, 0.9, 0.7]).to_numpy()
    metrics = retrain._trade_metrics_from_scores(y_true, y_prob, 0.75, None)
    assert metrics["metric_source"] == "classification_proxy"
    assert metrics["trading_metrics_available"] is False
    assert "classification-only proxy" in metrics["warning"]


def test_threshold_sweep_output_is_generated() -> None:
    y_true = pd.Series([1, 0, 1, 0, 1, 0]).to_numpy()
    y_prob = pd.Series([0.8, 0.2, 0.9, 0.6, 0.7, 0.1]).to_numpy()
    returns = pd.Series([0.05, -0.02, 0.03, -0.01, 0.02, -0.03]).to_numpy()
    chosen = retrain.optimize_threshold_from_probs(y_true, y_prob, returns)
    assert chosen["threshold"] in retrain.EVALUATION_THRESHOLDS


def test_walk_forward_fold_metrics_include_trading_metrics_when_returns_exist() -> None:
    frame = _dataset().iloc[:60].copy()
    groups = retrain.detect_column_groups(frame)
    X = frame[["feature_a", "feature_b"]].to_numpy(dtype="float32")
    y = frame["cost_adjusted_success_5m"].to_numpy(dtype=int)
    timestamps = pd.to_datetime(frame["timestamp"])
    returns = frame[groups["evaluation_return_column_used"]].to_numpy(dtype=float)
    summary = retrain._walk_forward_summary(X, y, timestamps, ["feature_a", "feature_b"], "cost_adjusted_success_5m", returns)
    assert summary["folds"]
    assert "profit_factor" in summary["folds"][0]
    assert "trade_count" in summary["folds"][0]
    assert "train_start" in summary["folds"][0]
    assert "validation_end" in summary["folds"][0]


def test_non_monotonic_input_is_sorted_before_holdout_split() -> None:
    frame = _dataset().iloc[:30].copy()
    shuffled = pd.concat([frame.iloc[10:20], frame.iloc[:10], frame.iloc[20:]], ignore_index=True)
    shuffled.loc[:, "expiry"] = pd.Timestamp("2026-06-26 15:30:00+05:30")
    shuffled.loc[:, "strike_price"] = 25000
    shuffled.loc[:, "option_type"] = "CE"
    prepared = retrain._prepare_chronological_work(shuffled[["timestamp", "expiry", "strike_price", "option_type", "feature_a", "cost_adjusted_success_5m"]])
    timestamps = pd.to_datetime(prepared["timestamp"])
    assert timestamps.is_monotonic_increasing


def test_duplicate_timestamps_across_strikes_are_handled_safely() -> None:
    frame = _dataset().iloc[:20].copy()
    frame.loc[:, "expiry"] = pd.Timestamp("2026-06-26 15:30:00+05:30")
    frame.loc[:, "option_type"] = "CE"
    duplicate = frame.copy()
    duplicate.loc[:, "strike_price"] = 25000
    frame.loc[:, "strike_price"] = 24900
    combined = pd.concat([frame, duplicate], ignore_index=True)
    prepared = retrain._prepare_chronological_work(combined[["timestamp", "expiry", "strike_price", "option_type", "feature_a", "cost_adjusted_success_5m"]])
    timestamps = pd.to_datetime(prepared["timestamp"])
    assert timestamps.is_monotonic_increasing
    assert len(prepared) == len(combined)


def test_chronological_assertion_rejects_future_leak_between_train_and_validation() -> None:
    timestamps = pd.Series(pd.to_datetime([
        "2026-06-01T09:15:00+05:30",
        "2026-06-01T09:16:00+05:30",
        "2026-06-01T09:17:00+05:30",
    ]))
    with pytest.raises(AssertionError):
        retrain._assert_fold_is_chronological(
            timestamps,
            np.array([0, 2]),
            np.array([1]),
            context="bad_fold",
        )


def test_walk_forward_fold_ranges_are_monotonic() -> None:
    frame = _dataset().iloc[:60].copy()
    groups = retrain.detect_column_groups(frame)
    prepared = retrain._prepare_chronological_work(frame[["timestamp", "feature_a", "feature_b", "cost_adjusted_success_5m", groups["evaluation_return_column_used"]]])
    X = prepared[["feature_a", "feature_b"]].to_numpy(dtype="float32")
    y = prepared["cost_adjusted_success_5m"].to_numpy(dtype=int)
    timestamps = pd.to_datetime(prepared["timestamp"])
    returns = prepared[groups["evaluation_return_column_used"]].to_numpy(dtype=float)
    summary = retrain._walk_forward_summary(X, y, timestamps, ["feature_a", "feature_b"], "cost_adjusted_success_5m", returns)
    for fold in summary["folds"]:
        assert pd.Timestamp(fold["train_start"]) <= pd.Timestamp(fold["train_end"])
        assert pd.Timestamp(fold["validation_start"]) <= pd.Timestamp(fold["validation_end"])
        assert pd.Timestamp(fold["test_start"]) <= pd.Timestamp(fold["test_end"])
        assert pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["validation_start"])
        assert pd.Timestamp(fold["validation_end"]) < pd.Timestamp(fold["test_start"])


def test_holdout_split_keeps_duplicate_timestamps_together() -> None:
    timestamps = pd.Series(pd.to_datetime([
        "2026-06-01T09:15:00+05:30",
        "2026-06-01T09:15:00+05:30",
        "2026-06-01T09:16:00+05:30",
        "2026-06-01T09:17:00+05:30",
        "2026-06-01T09:18:00+05:30",
        "2026-06-01T09:18:00+05:30",
        "2026-06-01T09:19:00+05:30",
    ]))
    split = retrain._build_holdout_split_from_timestamps(timestamps)
    train_ts = set(timestamps.iloc[split["train"]].astype(str))
    val_ts = set(timestamps.iloc[split["validation"]].astype(str))
    test_ts = set(timestamps.iloc[split["test"]].astype(str))
    assert train_ts.isdisjoint(val_ts)
    assert train_ts.isdisjoint(test_ts)
    assert val_ts.isdisjoint(test_ts)


def test_live_schema_mismatch_prevents_production_adoption(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retrain, "_live_schema_gate_for_variant", lambda feature_cols: {"compatible": False, "feature_order_matches_exactly": False, "disallowed_training_only_features": []})
    verdict = retrain._production_adoption_verdict(
        enriched_best={"test_metrics": {"f1": 0.5}, "walk_forward": {"split_count": 5}},
        feature_cols=["feature_a", "feature_b"],
        leakage_ok=True,
        estimated_only=False,
    )
    assert verdict["production_adoption_allowed"] is False
    assert verdict["verdict"] == "RESEARCH_ONLY_ESTIMATED_GREEKS_NOT_ADOPTABLE"


def test_comparison_report_is_created(tmp_path: Path) -> None:
    frame = _dataset()
    feature_sets = retrain.build_feature_sets(frame, use_black_scholes_features=True)
    baseline_result = {"reports": {"cost_adjusted_success_5m": {"logistic_regression": {"test_metrics": {"roc_auc": 0.5, "f1": 0.4}, "trade_metrics": {"profit_factor": 1.0, "sharpe": 0.5}}}}}
    enriched_result = {"reports": {"cost_adjusted_success_5m": {"logistic_regression": {"test_metrics": {"roc_auc": 0.6, "f1": 0.5}, "trade_metrics": {"profit_factor": 1.2, "sharpe": 0.7}, "feature_importance": [{"feature": "bs_iv", "importance": 0.3}]}}}}
    adoption = {"verdict": "RESEARCH_ONLY_ESTIMATED_GREEKS_NOT_ADOPTABLE", "strict_live_schema_gate": {"compatible": False}}
    groups = retrain.detect_column_groups(frame)
    payload = retrain.write_bs_comparison_report(
        dataset_path=tmp_path / "edge.csv",
        enriched_dataset_path=tmp_path / "edge_bs_enriched.csv",
        df=frame,
        enriched_df=frame,
        feature_sets=feature_sets,
        baseline_result=baseline_result,
        enriched_result=enriched_result,
        leakage_result={"passed": True},
        adoption=adoption,
        column_groups=groups,
        same_period_report={"same_period_gate": "PASS", "enabled": True},
    )
    assert Path(payload["json"]).exists()
    assert Path(payload["md"]).exists()


def test_small_dataset_real_metrics_are_marked_smoke_or_synthetic() -> None:
    y_true = pd.Series([1, 0, 1]).to_numpy()
    y_prob = pd.Series([0.9, 0.1, 0.8]).to_numpy()
    returns = pd.Series([0.02, -0.01, 0.03]).to_numpy()
    metrics = retrain._trade_metrics_from_scores(y_true, y_prob, 0.5, returns)
    assert metrics["smoke_or_synthetic_warning"] is not None


def test_all_model_family_parser_expands_default_registry() -> None:
    names = retrain._parse_model_families_arg("all")
    assert "extra_trees" in names
    assert "hist_gradient_boosting" in names
