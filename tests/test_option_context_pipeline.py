from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.option_chain_pipeline_lib import (
    check_live_schema_compatibility,
    merge_option_context_into_reconstructed_dataset,
    validate_option_context_dataset,
)
from scripts.collect_live_option_context import normalize_option_chain_snapshot, parse_dhan_option_chain_payload


LIVE_FEATURES = [
    "base_feature",
    "ctx_iv",
    "ctx_iv_change_pct",
    "ctx_iv_percentile",
    "ctx_delta",
    "ctx_gamma",
    "ctx_vega",
    "ctx_theta",
    "delta_abs",
    "greeks_imbalance",
    "bid_ask_spread_pct",
    "theta_to_vega_ratio",
    "gamma_to_theta_ratio",
]


def _stub_live_features(feature_names: list[str]) -> None:
    import scripts.option_chain_pipeline_lib as lib

    lib.live_feature_schema_summary = lambda: {  # type: ignore[assignment]
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "scaler_mean_len": len(feature_names),
        "scaler_std_len": len(feature_names),
        "model_input_shape": [1, len(feature_names)],
    }


def _make_reconstructed_dataset(tmp_path: Path, *, extra_columns: dict | None = None, feature_columns: list[str] | None = None) -> Path:
    df = pd.DataFrame(
        {
            "timestamp": ["2026-06-04T09:15:00+05:30", "2026-06-04T09:16:00+05:30"],
            "expiry": ["2026-06-11", "2026-06-11"],
            "strike_price": [24700, 24700],
            "option_type": ["CE", "CE"],
            "instrument_key": ["OPT1", "OPT1"],
            "base_feature": [1.0, 2.0],
        }
    )
    if extra_columns:
        for key, value in extra_columns.items():
            df[key] = value
    path = tmp_path / "reconstructed.csv"
    df.to_csv(path, index=False)
    schema = {"feature_columns": feature_columns or ["base_feature"]}
    path.with_name(path.stem + "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    return path


def _make_option_context_dataset(tmp_path: Path, **overrides) -> Path:
    base = {
        "timestamp": ["2026-06-04T09:15:00+05:30", "2026-06-04T09:16:00+05:30"],
        "underlying_spot": [24690.0, 24695.0],
        "expiry": ["2026-06-11", "2026-06-11"],
        "strike": [24700, 24700],
        "option_type": ["CE", "CE"],
        "open": [100.0, 101.0],
        "high": [103.0, 104.0],
        "low": [99.0, 100.5],
        "close": [101.8, 102.2],
        "volume": [1000, 1050],
        "open_interest": [20000, 20100],
        "change_in_oi": [100, 100],
        "iv": [0.18, 0.19],
        "delta": [0.48, 0.49],
        "gamma": [0.01, 0.011],
        "theta": [-4.0, -4.1],
        "vega": [6.0, 6.1],
        "bid": [101.5, 101.9],
        "ask": [102.0, 102.4],
        "spread": [0.5, 0.5],
    }
    base.update(overrides)
    df = pd.DataFrame(base)
    path = tmp_path / "option_context.csv"
    df.to_csv(path, index=False)
    return path


def test_valid_option_context_dataset_passes(tmp_path: Path) -> None:
    path = _make_option_context_dataset(tmp_path)
    payload = validate_option_context_dataset(path)
    assert payload["ok"] is True
    assert payload["final_verdict"] == "OPTION_CONTEXT_DATA_VALID"


def test_missing_iv_fails(tmp_path: Path) -> None:
    path = _make_option_context_dataset(tmp_path, iv=[None, 0.19])
    payload = validate_option_context_dataset(path)
    assert payload["ok"] is False
    assert payload["missing_required_value_count"] > 0


def test_missing_greeks_fail(tmp_path: Path) -> None:
    path = _make_option_context_dataset(tmp_path, delta=[None, None], gamma=[None, None], theta=[None, None], vega=[None, None])
    payload = validate_option_context_dataset(path)
    assert payload["ok"] is False


def test_missing_bid_ask_fails(tmp_path: Path) -> None:
    path = _make_option_context_dataset(tmp_path, bid=[None, None], ask=[None, None])
    payload = validate_option_context_dataset(path)
    assert payload["ok"] is False


def test_bid_gt_ask_fails(tmp_path: Path) -> None:
    path = _make_option_context_dataset(tmp_path, bid=[103.0, 101.9])
    payload = validate_option_context_dataset(path)
    assert payload["ok"] is False
    assert payload["bad_bid_ask_count"] > 0


def test_wrong_spread_fails(tmp_path: Path) -> None:
    path = _make_option_context_dataset(tmp_path, spread=[1.2, 0.5])
    payload = validate_option_context_dataset(path)
    assert payload["ok"] is False
    assert payload["bad_spread_count"] > 0


def test_duplicate_timestamp_expiry_strike_type_fails(tmp_path: Path) -> None:
    path = _make_option_context_dataset(
        tmp_path,
        timestamp=["2026-06-04T09:15:00+05:30", "2026-06-04T09:15:00+05:30"],
    )
    payload = validate_option_context_dataset(path)
    assert payload["ok"] is False
    assert payload["duplicate_count"] > 0


def test_invalid_ce_pe_fails(tmp_path: Path) -> None:
    path = _make_option_context_dataset(tmp_path, option_type=["CALL", "CE"])
    payload = validate_option_context_dataset(path)
    assert payload["ok"] is False


def test_forward_asof_future_merge_is_rejected(tmp_path: Path) -> None:
    _stub_live_features(LIVE_FEATURES)
    reconstructed = _make_reconstructed_dataset(tmp_path)
    context = _make_option_context_dataset(tmp_path, timestamp=["2026-06-04T09:16:00+05:30", "2026-06-04T09:17:00+05:30"])
    payload = merge_option_context_into_reconstructed_dataset(reconstructed, context, allow_backward_asof=True, min_context_coverage=0.0)
    assert payload["exact_match_count"] == 1
    assert payload["asof_match_count"] == 0


def test_backward_asof_merge_is_accepted_within_tolerance(tmp_path: Path) -> None:
    _stub_live_features(LIVE_FEATURES)
    reconstructed = _make_reconstructed_dataset(tmp_path)
    context = _make_option_context_dataset(tmp_path, timestamp=["2026-06-04T09:14:58+05:30", "2026-06-04T09:15:58+05:30"])
    payload = merge_option_context_into_reconstructed_dataset(reconstructed, context, allow_backward_asof=True, asof_tolerance="5s")
    assert payload["asof_match_count"] >= 1
    assert payload["context_coverage_ratio"] == pytest.approx(1.0)


def test_missing_context_values_are_not_filled_with_zero(tmp_path: Path) -> None:
    _stub_live_features(LIVE_FEATURES)
    reconstructed = _make_reconstructed_dataset(tmp_path)
    context = _make_option_context_dataset(tmp_path, gamma=[None, 0.011])
    with pytest.raises(RuntimeError):
        merge_option_context_into_reconstructed_dataset(reconstructed, context)


def test_forbidden_future_label_columns_are_rejected(tmp_path: Path) -> None:
    _stub_live_features(["base_feature", "future_close"])
    reconstructed = _make_reconstructed_dataset(
        tmp_path,
        extra_columns={"future_close": [1.1, 1.2]},
        feature_columns=["base_feature", "future_close"],
    )
    payload = check_live_schema_compatibility(reconstructed)
    assert "future_close" in payload["forbidden_columns_present"]
    assert payload["final_verdict"] == "LIVE_SCHEMA_NOT_COMPATIBLE"


def test_exact_schema_compatibility_passes_only_when_names_and_order_match(tmp_path: Path) -> None:
    _stub_live_features(LIVE_FEATURES)
    reconstructed = _make_reconstructed_dataset(tmp_path)
    context = _make_option_context_dataset(tmp_path)
    merge_payload = merge_option_context_into_reconstructed_dataset(reconstructed, context)
    schema_payload = check_live_schema_compatibility(Path(merge_payload["output_dataset"]))
    assert schema_payload["strict_live_schema_gate"]["compatible"] is True
    assert schema_payload["final_verdict"] == "LIVE_SCHEMA_COMPATIBLE_RESEARCH_ONLY"


def test_schema_mismatch_fails(tmp_path: Path) -> None:
    _stub_live_features(LIVE_FEATURES + ["missing_feature"])
    reconstructed = _make_reconstructed_dataset(tmp_path)
    context = _make_option_context_dataset(tmp_path)
    payload = merge_option_context_into_reconstructed_dataset(reconstructed, context)
    assert payload["strict_live_schema_gate"]["compatible"] is False
    assert "missing_feature" in payload["missing_live_features"]


def test_feature_order_mismatch_fails(tmp_path: Path) -> None:
    _stub_live_features(LIVE_FEATURES)
    reconstructed = _make_reconstructed_dataset(tmp_path)
    context = _make_option_context_dataset(tmp_path)
    payload = merge_option_context_into_reconstructed_dataset(reconstructed, context)
    schema_path = Path(payload["schema_path"])
    schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_payload["feature_columns"] = list(reversed(schema_payload["feature_columns"]))
    schema_path.write_text(json.dumps(schema_payload), encoding="utf-8")
    schema_payload = check_live_schema_compatibility(Path(payload["output_dataset"]))
    assert schema_payload["strict_live_schema_gate"]["feature_order_matches_exactly"] is False
    assert schema_payload["final_verdict"] == "LIVE_SCHEMA_NOT_COMPATIBLE"


def test_estimated_greeks_do_not_count_as_real_live_schema(tmp_path: Path) -> None:
    _stub_live_features(LIVE_FEATURES)
    dataset = _make_reconstructed_dataset(
        tmp_path,
        extra_columns={
            "estimated_ctx_iv": [0.18, 0.19],
            "estimated_ctx_delta": [0.48, 0.49],
            "estimated_ctx_gamma": [0.01, 0.011],
            "estimated_ctx_theta": [-4.0, -4.1],
            "estimated_ctx_vega": [6.0, 6.1],
            "greeks_source": ["estimated_black_scholes", "estimated_black_scholes"],
            "production_adoption_allowed": [False, False],
        },
        feature_columns=["base_feature"],
    )
    payload = check_live_schema_compatibility(dataset)
    assert payload["estimated_only_dataset"] is True
    assert payload["final_verdict"] == "RESEARCH_ONLY_ESTIMATED_GREEKS_NOT_ADOPTABLE"


def test_missing_bid_ask_blocks_production_compatibility(tmp_path: Path) -> None:
    _stub_live_features(["base_feature", "bid_ask_spread_pct"])
    dataset = _make_reconstructed_dataset(
        tmp_path,
        extra_columns={"bid_ask_spread_pct": [0.01, 0.02]},
        feature_columns=["base_feature", "bid_ask_spread_pct"],
    )
    payload = check_live_schema_compatibility(dataset)
    assert "bid" in payload["missing_real_bid_ask_columns"]
    assert payload["final_verdict"] == "LIVE_SCHEMA_NOT_COMPATIBLE"


def test_estimated_only_dataset_cannot_trigger_adoption(tmp_path: Path) -> None:
    _stub_live_features(LIVE_FEATURES)
    dataset = _make_reconstructed_dataset(
        tmp_path,
        extra_columns={
            "estimated_ctx_iv": [0.18, 0.19],
            "production_adoption_allowed": [False, False],
        },
        feature_columns=["base_feature"],
    )
    payload = check_live_schema_compatibility(dataset)
    assert payload["production_adoption_allowed_values"] == [False]
    assert payload["final_verdict"] == "RESEARCH_ONLY_ESTIMATED_GREEKS_NOT_ADOPTABLE"


def test_collector_normalization_maps_broker_fields_correctly() -> None:
    rows = [
        {
            "strike": 24700,
            "option_type": "CE",
            "raw": {
                "ltp": 102.2,
                "volume": 1050,
                "oi": 20100,
                "change_oi": 100,
                "iv": 0.19,
                "delta": 0.49,
                "gamma": 0.011,
                "theta": -4.1,
                "vega": 6.1,
                "bidPrice": 101.9,
                "askPrice": 102.4,
                "lotsize": 75,
            },
        }
    ]
    frame = normalize_option_chain_snapshot(rows, timestamp="2026-06-04T09:16:00+05:30", underlying_spot=24695.0)
    row = frame.iloc[0].to_dict()
    assert row["open_interest"] == 20100
    assert row["change_in_oi"] == 100
    assert row["bid"] == 101.9
    assert row["ask"] == 102.4
    assert row["spread"] == pytest.approx(0.5)


def test_duplicate_snapshot_rows_are_removed_by_normalization() -> None:
    rows = [
        {"strike": 24700, "option_type": "CE", "raw": {"ltp": 100.0}},
        {"strike": 24700, "option_type": "CE", "raw": {"ltp": 101.0}},
    ]
    frame = normalize_option_chain_snapshot(rows, timestamp="2026-06-04T09:16:00+05:30", underlying_spot=24695.0)
    assert len(frame) == 1


def test_dhan_option_chain_payload_is_normalized() -> None:
    payload = {
        "data": {
            "oc": [
                {
                    "securityId": 52175,
                    "tradingSymbol": "NIFTY24OCT24700CE",
                    "exchangeSegment": "NSE_FNO",
                    "expiry": "2026-06-11",
                    "strikePrice": 24700,
                    "optionType": "CALL",
                    "lastPrice": 102.2,
                    "volume": 1050,
                    "openInterest": 20100,
                    "changeOi": 100,
                    "impliedVolatility": 0.19,
                    "delta": 0.49,
                    "gamma": 0.011,
                    "theta": -4.1,
                    "vega": 6.1,
                    "bestBidPrice": 101.9,
                    "bestAskPrice": 102.4,
                    "lotSize": 75,
                    "underlyingPrice": 24695.0,
                }
            ]
        }
    }
    chain = parse_dhan_option_chain_payload(payload, symbol="NIFTY", expiry="2026-06-11")
    assert len(chain) == 1
    row = chain[0]
    assert row["option_type"] == "CE"
    assert row["strike"] == 24700
    assert row["raw"]["iv"] == 0.19
    assert row["raw"]["bid"] == 101.9
    assert row["raw"]["ask"] == 102.4
