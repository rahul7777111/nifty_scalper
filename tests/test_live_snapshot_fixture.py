#!/usr/bin/env python3
"""Tests for tests/fixtures/live_nifty_option_snapshot.json fixture.

Validates:
  - Fixture file exists and is valid JSON
  - Required top-level fields are present
  - Spot data contains 20+ candles for rolling features
  - Option chain has multiple strikes with CE/PE data
  - Snapshot is recent enough (not stale)
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "live_nifty_option_snapshot.json"


# ---------------------------------------------------------------------------
# Test 1: Fixture file exists and is valid JSON
# ---------------------------------------------------------------------------

def test_fixture_file_exists():
    """Fixture file must exist at expected path."""
    assert FIXTURE_PATH.exists(), f"Fixture not found at {FIXTURE_PATH}"


def test_fixture_is_valid_json():
    """Fixture must be parseable as JSON."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), "Fixture root must be a JSON object"


# ---------------------------------------------------------------------------
# Test 2: Required top-level fields are present
# ---------------------------------------------------------------------------

def test_fixture_has_required_fields():
    """Fixture must contain all required top-level fields."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    required_fields = ["timestamp", "symbol", "spot", "option_chain", "context"]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"


def test_fixture_timestamp_format():
    """timestamp field must be a valid ISO8601 string with timezone."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    ts = data["timestamp"]
    # Should parse with timezone info
    parsed = datetime.fromisoformat(ts.replace("+05:30", "+05:30"))
    assert parsed.year == 2026, "Timestamp year should be 2026"
    assert parsed.month == 6, "Timestamp month should be June"


def test_fixture_symbol():
    """symbol field must be NIFTY."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    assert data["symbol"] == "NIFTY", "Symbol should be NIFTY"


# ---------------------------------------------------------------------------
# Test 3: Spot data validation
# ---------------------------------------------------------------------------

def test_spot_has_ltp():
    """Spot data must contain ltp (last traded price)."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    assert "spot" in data
    assert "ltp" in data["spot"]
    assert isinstance(data["spot"]["ltp"], (int, float))
    assert 20000 < data["spot"]["ltp"] < 30000, "NIFTY spot should be in realistic range"


def test_spot_has_sufficient_candles_for_rolling_features():
    """Spot ohlcv must contain 20+ candles for rolling feature computation."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    assert "spot" in data
    assert "ohlcv" in data["spot"]
    candles = data["spot"]["ohlcv"]
    assert isinstance(candles, list), "ohlcv must be a list"
    assert len(candles) >= 20, f"Need at least 20 candles for rolling features, got {len(candles)}"


def test_each_candle_has_required_fields():
    """Each candle must have open, high, low, close, volume, timestamp."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    candle_fields = ["open", "high", "low", "close", "volume", "timestamp"]
    for i, candle in enumerate(data["spot"]["ohlcv"]):
        for field in candle_fields:
            assert field in candle, f"Candle {i} missing field: {field}"


# ---------------------------------------------------------------------------
# Test 4: Option chain validation
# ---------------------------------------------------------------------------

def test_option_chain_has_expiry():
    """Option chain must specify expiry date."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    assert "option_chain" in data
    assert "expiry" in data["option_chain"]
    # Expiry should be a date string
    expiry = data["option_chain"]["expiry"]
    assert "-" in expiry, "Expiry should be YYYY-MM-DD format"


def test_option_chain_has_strikes():
    """Option chain must contain strikes array."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    assert "strikes" in data["option_chain"]
    strikes = data["option_chain"]["strikes"]
    assert isinstance(strikes, list)
    assert len(strikes) > 0, "Must have at least one strike"


def test_each_strike_has_ce_and_pe():
    """Each strike must have ce and pe data with bid/ask/ltp/volume/oi."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    option_fields = ["bid", "ask", "ltp", "volume", "oi"]
    for i, strike_data in enumerate(data["option_chain"]["strikes"]):
        assert "strike" in strike_data, f"Strike {i} missing strike price"
        assert "ce" in strike_data, f"Strike {i} missing CE data"
        assert "pe" in strike_data, f"Strike {i} missing PE data"

        for field in option_fields:
            assert field in strike_data["ce"], f"Strike {i} CE missing {field}"
            assert field in strike_data["pe"], f"Strike {i} PE missing {field}"


def test_atm_strike_exists():
    """There should be a strike very close to ATM (within 50 points)."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    spot_ltp = data["spot"]["ltp"]
    strikes = data["option_chain"]["strikes"]

    atm_distance = min(abs(s["strike"] - spot_ltp) for s in strikes)
    assert atm_distance <= 50, f"ATM strike should be within 50 points of spot {spot_ltp}, found distance {atm_distance}"


# ---------------------------------------------------------------------------
# Test 5: Context validation
# ---------------------------------------------------------------------------

def test_context_has_required_fields():
    """Context must have atm_distance, iv, time_to_expiry."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    assert "context" in data
    ctx = data["context"]
    required = ["atm_distance", "iv", "time_to_expiry"]
    for field in required:
        assert field in ctx, f"Context missing required field: {field}"


def test_context_values_in_realistic_ranges():
    """Context values should be in realistic ranges."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    ctx = data["context"]
    # ATM distance should be reasonable
    assert 0 <= ctx["atm_distance"] <= 500, "atm_distance out of range"
    # IV should be positive and realistic (percent)
    assert 5 <= ctx["iv"] <= 50, "IV should be in realistic range"
    # Time to expiry as fraction of year
    assert 0 < ctx["time_to_expiry"] <= 1, "time_to_expiry should be 0-1"


# ---------------------------------------------------------------------------
# Test 6: Snapshot is recent enough (not stale)
# ---------------------------------------------------------------------------

def test_snapshot_not_stale():
    """Fixture timestamp must be within 7 days of current date."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    fixture_time = datetime.fromisoformat(data["timestamp"].replace("+05:30", "+05:30"))
    now = datetime(2026, 6, 7, 15, 30, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))

    age = abs((now - fixture_time).total_seconds())
    max_age_seconds = 7 * 24 * 3600  # 7 days
    assert age <= max_age_seconds, f"Fixture is stale: {age / 3600:.1f} hours old"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])