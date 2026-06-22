#!/usr/bin/env python3
"""
tests/test_ml_only_dynamic_preset_no_leakage.py
==============================================
Verify dynamic presets use NO future, PnL, return, label, or outcome data.

Leakage categories checked:
- Future return columns (net_forward_return, gross_forward_return)
- Realized PnL (realized_pnl, pnl, profit, loss)
- Exit/MFE/MAE (exit_price, mfe, mae)
- Label/target columns (*_label, cost_survivor, *_label_v2)
- Expected return features (expected_return_after_cost, return_to_cost_ratio)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from candidate_filters import (
    _preset_dte_passes,
    _preset_liquidity_passes,
    _preset_premium_passes,
    _preset_spread_passes,
    _preset_time_window_passes,
    apply_dynamic_preset_filter,
    apply_liquidity_filter,
    apply_spread_limit_filter,
)


class TestPresetNoFutureLeakage:
    """Verify no preset filter uses future/return/PnL/leakage fields."""

    def test_dte_filter_uses_only_dte_days(self) -> None:
        """_preset_dte_passes must NOT look at any return, PnL, or label fields."""
        # Safe snapshot: only dte_days provided
        assert _preset_dte_passes({"dte_days": 5.0}, "2-7") is True
        assert _preset_dte_passes({"dte_days": 15.0}, "2-7") is False
        # Snapshot with leakage fields should still only check dte_days
        assert _preset_dte_passes({
            "dte_days": 5.0,
            "net_forward_return": 0.10,  # LEAKAGE
            "realized_pnl": 50.0,         # LEAKAGE
        }, "2-7") is True

    def test_spread_filter_uses_only_range_pct(self) -> None:
        """_preset_spread_passes must NOT look at any return or PnL fields."""
        assert _preset_spread_passes({"range_pct": 0.05}, 0.10) is True
        assert _preset_spread_passes({"range_pct": 0.15}, 0.10) is False
        # With leakage fields - should still only check range_pct
        assert _preset_spread_passes({
            "range_pct": 0.05,
            "net_forward_return": 0.10,  # LEAKAGE
            "profitable_trade_label": 1,  # LEAKAGE
        }, 0.10) is True

    def test_liquidity_filter_uses_only_volume(self) -> None:
        """_preset_liquidity_passes must NOT look at return or PnL fields."""
        assert _preset_liquidity_passes({"volume": 100000.0}, 50000.0) is True
        assert _preset_liquidity_passes({"volume": 10000.0}, 50000.0) is False
        # With leakage - should still only check volume
        assert _preset_liquidity_passes({
            "volume": 100000.0,
            "realized_pnl": 200.0,  # LEAKAGE
            "cost_survivor_label": 1,  # LEAKAGE
        }, 50000.0) is True

    def test_premium_filter_uses_only_ltp(self) -> None:
        """_preset_premium_passes must NOT look at return or PnL fields."""
        assert _preset_premium_passes({"ltp": 100.0}, "mid") is True
        assert _preset_premium_passes({"ltp": 40.0}, "mid") is False
        assert _preset_premium_passes({
            "ltp": 100.0,
            "expected_return_after_cost": 0.05,  # LEAKAGE
            "mfe": 0.10,  # LEAKAGE
        }, "mid") is True

    def test_dynamic_preset_filter_rejects_snapshot_with_only_leakage_fields(self) -> None:
        """A snapshot containing ONLY leakage fields should not cause the preset filter to crash."""
        snapshot = {
            "option_type": "PE",
            # No live fields - only leakage fields
            "net_forward_return": 0.05,
            "gross_forward_return": 0.06,
            "realized_pnl": 100.0,
            "profitable_trade_label": 1,
            "cost_survivor_label": 1,
        }
        preset = {
            "dte_range": "all",
            "spread_limit_pct": 0.0,
            "liquidity_min": 0.0,
            "premium_band": "all",
            "avoid_first_n_minutes": 0,
            "avoid_last_n_minutes": 0,
            "entry_time_start": "09:30",
            "entry_time_end": "15:30",
            "regime_filter": "all",
        }
        # Should not crash; conservative pass on missing live fields
        result = apply_dynamic_preset_filter(snapshot, preset)
        assert result["filter_passed"] is True  # missing live fields -> conservative pass


class TestPresetNoPnLLeakage:
    """Verify no preset uses realized PnL or exit prices."""

    def test_preset_cannot_use_realized_pnl(self) -> None:
        """Preset config must not reference realized_pnl in any parameter."""
        from scripts.ml_only_dynamic_preset_candidate_rescue import DynamicPreset
        p = DynamicPreset()
        d = p.to_dict()
        pnl_tokens = ["pnl", "profit", "loss", "realized", "mfe", "mae", "exit"]
        for key in d:
            for token in pnl_tokens:
                assert token not in key.lower() or key in [
                    # cooldown_after_loss_minutes is a risk-control config, not realized PnL
                    "cooldown_after_loss_minutes",
                    "daily_stop_loss_pct",
                    "daily_profit_lock_pct",
                ], f"Leakage token '{token}' in preset key: {key}"

    def test_preset_cannot_use_exit_price(self) -> None:
        """Preset must not include exit_price or mfe/mae as configuration."""
        from scripts.ml_only_dynamic_preset_candidate_rescue import DynamicPreset
        p = DynamicPreset()
        d = p.to_dict()
        assert "exit_price" not in d
        assert "mfe" not in d
        assert "mae" not in d


class TestPresetNoLabelLeakage:
    """Verify no preset uses label, target, or outcome columns."""

    def test_preset_cannot_use_label_columns(self) -> None:
        """Preset config must not include any *_label or *_label_v2 columns."""
        from scripts.ml_only_dynamic_preset_candidate_rescue import DynamicPreset
        p = DynamicPreset()
        d = p.to_dict()
        label_tokens = ["label", "cost_survivor", "profitable", "strong_",
                        "high_conviction", "paper_candidate", "avoid_trade",
                        "weak_trade", "no_trade"]
        for key in d:
            lower = key.lower()
            assert not any(t in lower for t in label_tokens), \
                f"Label token found in preset key: {key}"

    def test_preset_cannot_use_target_columns(self) -> None:
        """Preset config must not include target columns."""
        from scripts.ml_only_dynamic_preset_candidate_rescue import DynamicPreset
        p = DynamicPreset()
        d = p.to_dict()
        target_tokens = ["target", "outcome", "horizon_", "expected_return"]
        for key in d:
            lower = key.lower()
            assert not any(t in lower for t in target_tokens), \
                f"Target token found in preset key: {key}"


class TestPresetNoReturnLeakage:
    """Verify no preset uses return or forward return columns."""

    def test_preset_cannot_use_forward_return(self) -> None:
        """Preset config must not include net_forward_return or gross_forward_return."""
        from scripts.ml_only_dynamic_preset_candidate_rescue import DynamicPreset
        p = DynamicPreset()
        d = p.to_dict()
        return_tokens = ["net_forward_return", "gross_forward_return",
                         "forward_return", "return_to_cost"]
        for key in d:
            lower = key.lower()
            assert not any(t in lower for t in return_tokens), \
                f"Return token found in preset key: {key}"