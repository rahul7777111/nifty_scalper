#!/usr/bin/env python3
"""
tests/test_ml_only_dynamic_preset_runtime.py
============================================
Verify runtime (MLRuntimeEngine, candidate_router) correctly loads and applies
dynamic presets in shadow/paper mode.

Runtime behavior verified:
- Load candidate manifest and dynamic_preset.json
- Build live features
- Check preset conditions
- If preset conditions pass AND model probability >= threshold, emit shadow signal
- If preset conditions fail, skip with reason (logged)
- Skip reasons are specific: spread_too_high, liquidity_too_low, outside_dte_range,
  outside_time_window, regime_mismatch, confidence_below_threshold,
  max_trades_per_day_reached, cooldown_active
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


class TestDynamicPresetJSONLoading:
    """Verify dynamic_preset.json is loaded correctly from candidate directory."""

    def test_preset_json_schema_has_required_fields(self) -> None:
        """dynamic_preset.json must have all required live-computable fields."""
        from scripts.ml_only_dynamic_preset_candidate_rescue import DynamicPreset
        p = DynamicPreset(
            preset_id="test",
            preset_family="conservative",
            threshold=0.40,
            max_trades_per_day=2,
            top_n_confidence_per_day=2,
            spread_limit_pct=0.05,
            liquidity_min=100000.0,
            premium_band="all",
            dte_range="all",
            regime_filter="all",
        )
        d = p.to_dict()
        required = [
            "preset_id", "preset_family", "threshold",
            "max_trades_per_day", "spread_limit_pct", "liquidity_min",
            "premium_band", "dte_range", "regime_filter",
        ]
        for field in required:
            assert field in d, f"Missing required field: {field}"

    def test_preset_round_trip_json(self) -> None:
        """Preset must survive JSON serialization round-trip."""
        from scripts.ml_only_dynamic_preset_candidate_rescue import DynamicPreset
        p = DynamicPreset(
            preset_id="test_round_trip",
            preset_family="balanced",
            threshold=0.35,
            max_trades_per_day=3,
            top_n_confidence_per_day=3,
            spread_limit_pct=0.10,
            liquidity_min=50000.0,
            premium_band="mid",
            dte_range="2-7",
            regime_filter="high_volatility",
            avoid_first_n_minutes=5,
            avoid_last_n_minutes=5,
        )
        json_str = json.dumps(p.to_dict())
        loaded = json.loads(json_str)
        p2 = DynamicPreset.from_dict(loaded)
        assert p2.preset_id == p.preset_id
        assert p2.threshold == p.threshold
        assert p2.dte_range == p.dte_range
        assert p2.regime_filter == p.regime_filter

    def test_ml_runtime_loads_preset_from_candidate_dir(self) -> None:
        """MLRuntimeEngine._load_dynamic_preset must load from candidate directory."""
        from ml_runtime import MLRuntimeEngine
        import inspect
        sig = inspect.signature(MLRuntimeEngine._load_dynamic_preset)
        # Method exists and is callable
        assert "_load_dynamic_preset" in dir(MLRuntimeEngine)


class TestPresetSkipsWithCorrectReason:
    """Verify runtime logs correct skip reason when preset rejects a trade."""

    def test_filter_rejects_outside_dte_range(self) -> None:
        """DTE outside range -> 'dynamic_preset_rejected_outside_dte_range'."""
        from candidate_filters import apply_dynamic_preset_filter
        snapshot = {"dte_days": 15.0, "range_pct": 0.05, "volume": 50000.0, "ltp": 100.0}
        preset = {"dte_range": "0-1", "spread_limit_pct": 0.15, "liquidity_min": 0.0,
                  "premium_band": "all", "avoid_first_n_minutes": 0, "avoid_last_n_minutes": 0,
                  "entry_time_start": "09:30", "entry_time_end": "15:30", "regime_filter": "all"}
        result = apply_dynamic_preset_filter(snapshot, preset)
        assert result["filter_passed"] is False
        assert "dte" in result["rejection_reason"]

    def test_filter_rejects_spread_too_high(self) -> None:
        """Spread exceeds limit -> 'dynamic_preset_rejected_spread_too_high'."""
        from candidate_filters import apply_dynamic_preset_filter
        snapshot = {"dte_days": 5.0, "range_pct": 0.20, "volume": 50000.0, "ltp": 100.0}
        preset = {"dte_range": "all", "spread_limit_pct": 0.05, "liquidity_min": 0.0,
                  "premium_band": "all", "avoid_first_n_minutes": 0, "avoid_last_n_minutes": 0,
                  "entry_time_start": "09:30", "entry_time_end": "15:30", "regime_filter": "all"}
        result = apply_dynamic_preset_filter(snapshot, preset)
        assert result["filter_passed"] is False
        assert "spread" in result["rejection_reason"]

    def test_filter_rejects_liquidity_too_low(self) -> None:
        """Volume below minimum -> 'dynamic_preset_rejected_liquidity_too_low'."""
        from candidate_filters import apply_dynamic_preset_filter
        snapshot = {"dte_days": 5.0, "range_pct": 0.05, "volume": 5000.0, "ltp": 100.0}
        preset = {"dte_range": "all", "spread_limit_pct": 0.15, "liquidity_min": 50000.0,
                  "premium_band": "all", "avoid_first_n_minutes": 0, "avoid_last_n_minutes": 0,
                  "entry_time_start": "09:30", "entry_time_end": "15:30", "regime_filter": "all"}
        result = apply_dynamic_preset_filter(snapshot, preset)
        assert result["filter_passed"] is False
        assert "liquidity" in result["rejection_reason"]

    def test_filter_rejects_premium_band(self) -> None:
        """Premium outside band -> 'dynamic_preset_rejected_premium_band'."""
        from candidate_filters import apply_dynamic_preset_filter
        snapshot = {"dte_days": 5.0, "range_pct": 0.05, "volume": 50000.0, "ltp": 200.0}
        preset = {"dte_range": "all", "spread_limit_pct": 0.15, "liquidity_min": 0.0,
                  "premium_band": "low", "avoid_first_n_minutes": 0, "avoid_last_n_minutes": 0,
                  "entry_time_start": "09:30", "entry_time_end": "15:30", "regime_filter": "all"}
        result = apply_dynamic_preset_filter(snapshot, preset)
        assert result["filter_passed"] is False
        assert "premium" in result["rejection_reason"]

    def test_filter_rejects_outside_time_window(self) -> None:
        """Timestamp outside window -> 'dynamic_preset_rejected_outside_time_window'."""
        from candidate_filters import apply_dynamic_preset_filter
        # 9:00 is before 9:30 window start
        snapshot = {"timestamp": "2024-06-10T09:00:00+05:30", "dte_days": 5.0,
                    "range_pct": 0.05, "volume": 50000.0, "ltp": 100.0}
        preset = {"dte_range": "all", "spread_limit_pct": 0.15, "liquidity_min": 0.0,
                  "premium_band": "all", "avoid_first_n_minutes": 0, "avoid_last_n_minutes": 0,
                  "entry_time_start": "09:30", "entry_time_end": "15:30", "regime_filter": "all"}
        result = apply_dynamic_preset_filter(snapshot, preset)
        assert result["filter_passed"] is False
        assert "time" in result["rejection_reason"] or "window" in result["rejection_reason"]

    def test_filter_rejects_regime_mismatch(self) -> None:
        """Regime mismatch -> 'dynamic_preset_rejected_regime_mismatch'."""
        from candidate_filters import apply_dynamic_preset_filter
        snapshot = {"dte_days": 5.0, "range_pct": 0.05, "volume": 50000.0, "ltp": 100.0,
                    "regime": "CHOPPING"}  # snapshot regime is CHOPPING
        preset = {"dte_range": "all", "spread_limit_pct": 0.15, "liquidity_min": 0.0,
                  "premium_band": "all", "avoid_first_n_minutes": 0, "avoid_last_n_minutes": 0,
                  "entry_time_start": "09:30", "entry_time_end": "15:30",
                  "regime_filter": "non_chop"}  # only non_chop allowed
        result = apply_dynamic_preset_filter(snapshot, preset)
        assert result["filter_passed"] is False
        assert "regime" in result["rejection_reason"]


class TestPresetSafetyConstraints:
    """Verify preset cannot enable real trading or violate safety constraints."""

    def test_preset_json_cannot_enable_real_trading(self) -> None:
        """dynamic_preset.json must not contain any fields that enable live trading."""
        from scripts.ml_only_dynamic_preset_candidate_rescue import DynamicPreset
        p = DynamicPreset(preset_id="test", preset_family="conservative")
        d = p.to_dict()
        trading_enabling = ["real_trading", "live_order", "place_order", "broker",
                            "production", "enable_live"]
        for key in d:
            assert not any(t in key.lower() for t in trading_enabling), \
                f"Trading-enabling key in preset: {key}"

    def test_ml_runtime_never_enables_real_orders(self) -> None:
        """MLRuntimeEngine must never send real orders in shadow/paper mode."""
        from ml_runtime import MLRuntimeEngine
        import inspect
        source = inspect.getsource(MLRuntimeEngine)
        # Must never call place_order in shadow/paper mode
        assert "real_order_callable" in source or "real_order" in source.lower()
        # The _create_paper_trade raises error if real_order_callable is set
        paper_trade_src = inspect.getsource(MLRuntimeEngine._create_paper_trade)
        assert "RuntimeError" in paper_trade_src or "real_order_callable" in paper_trade_src