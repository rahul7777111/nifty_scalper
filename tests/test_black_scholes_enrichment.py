from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from black_scholes_features import enrich_option_frame


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-06-05T09:20:00+05:30", "2026-06-05T09:21:00+05:30", "2026-06-05T09:22:00+05:30"],
            "expiry": ["2026-06-12", "2026-06-05", "2026-06-12"],
            "strike": [24700, 24700, 25200],
            "option_type": ["CE", "PE", "CALL"],
            "underlying_spot_price": [24720.0, 24690.0, 24720.0],
            "option_ltp": [122.5, 95.0, 1.0],
            "bid": [122.0, 94.5, 0.4],
            "ask": [123.0, 95.5, 1.4],
            "volume": [1000, 900, 0],
            "oi": [15000, 14000, 0],
        }
    )


def test_enrichment_generates_expected_columns_and_preserves_source_data() -> None:
    source = _base_frame()
    enriched, metadata = enrich_option_frame(source)
    assert "underlying_spot_price" in enriched.columns
    assert "bs_iv" in enriched.columns
    assert "final_iv" in enriched.columns
    assert "greeks_quality_score" in enriched.columns
    assert enriched.loc[0, "option_ltp"] == 122.5
    assert metadata["total_rows"] == 3


def test_existing_iv_is_preferred_when_available() -> None:
    source = _base_frame()
    source["iv"] = [0.21, None, None]
    enriched, _ = enrich_option_frame(source)
    assert enriched.loc[0, "final_iv"] == pytest.approx(0.21)
    assert enriched.loc[0, "greeks_source"] == "existing_dataset_iv"


def test_invalid_price_below_intrinsic_marks_row_without_crashing() -> None:
    source = _base_frame()
    source.loc[0, "option_ltp"] = 1.0
    enriched, metadata = enrich_option_frame(source)
    assert bool(enriched.loc[0, "bs_iv_solve_ok"]) is False
    assert bool(enriched.loc[0, "bad_iv_flag"]) is True
    assert metadata["total_rows"] == 3


def test_expiry_day_date_only_defaults_to_market_close_and_sets_flags() -> None:
    source = _base_frame().iloc[[1]].copy()
    source.loc[source.index[0], "timestamp"] = "2026-06-05T09:20:00+05:30"
    source.loc[source.index[0], "expiry"] = "2026-06-05"
    enriched, _ = enrich_option_frame(source)
    assert bool(enriched.iloc[0]["is_expiry_day"]) is True
    assert float(enriched.iloc[0]["time_to_expiry_days"]) > 0.0
    assert bool(enriched.iloc[0]["is_near_expiry"]) is True


def test_quote_quality_flags_are_derived_from_bid_ask_and_price() -> None:
    source = _base_frame().iloc[[2]].copy()
    enriched, _ = enrich_option_frame(source)
    assert bool(enriched.iloc[0]["has_valid_bid_ask"]) is True
    assert bool(enriched.iloc[0]["wide_spread_flag"]) is True
    assert bool(enriched.iloc[0]["low_price_flag"]) is True


def test_alternate_column_names_and_call_put_normalization_work() -> None:
    source = pd.DataFrame(
        {
            "datetime": ["2026-06-05T09:20:00+05:30"],
            "expiration": ["2026-06-12"],
            "strike_price": [24700],
            "type": ["put"],
            "spot": [24720.0],
            "close": [140.0],
        }
    )
    enriched, metadata = enrich_option_frame(source)
    assert enriched.loc[0, "option_side_normalized"] == "PE"
    assert metadata["columns_used"]["timestamp"] == "datetime"


def test_script_writes_output_and_reports(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = tmp_path / "sample.csv"
    output_path = tmp_path / "sample_enriched.csv"
    _base_frame().to_csv(input_path, index=False)
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "enrich_option_dataset_black_scholes.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--sample-output",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    payload = json.loads(completed.stdout)
    assert output_path.exists()
    assert Path(payload["report_paths"]["json"]).exists()
    assert Path(payload["report_paths"]["md"]).exists()
    written = pd.read_csv(output_path)
    assert "final_iv" in written.columns
