from types import SimpleNamespace

from src.dynamic_preset_selector import select_dynamic_preset
from src.strategy import NiftyScalper, TradeState


def test_dynamic_preset_selector_carries_risk_fields():
    profile = {
        "candidate_id": "cand_1",
        "model_name": "rf",
        "dynamic_presets": {
            "normal": {
                "name": "NORMAL",
                "preset_family": "normal",
                "entry_threshold": 0.42,
                "max_trades_per_day": 2,
                "cooldown_minutes": 15,
                "stop_loss_pct": 0.24,
                "target_pct": 0.16,
                "trailing_sl_pct": 0.05,
                "size_multiplier": 0.6,
                "dir_sl_atr_mult": 1.1,
                "dir_tp_atr_mult": 1.9,
                "dir_trail_atr_mult": 0.8,
            }
        },
    }

    result = select_dynamic_preset(profile, {"regime": "bullish", "atr_pct": 0.008, "spread_pct": 0.02})

    assert result["trade_allowed"] is True
    risk = result["preset_config"]
    assert risk["cooldown_minutes"] == 15
    assert risk["stop_loss_pct"] == 0.24
    assert risk["target_pct"] == 0.16
    assert risk["trailing_sl_pct"] == 0.05
    assert risk["size_multiplier"] == 0.6
    assert risk["dir_sl_atr_mult"] == 1.1
    assert risk["dir_tp_atr_mult"] == 1.9
    assert risk["dir_trail_atr_mult"] == 0.8


def test_entry_qty_keeps_risk_sizing_before_scaling():
    app = NiftyScalper.__new__(NiftyScalper)
    app.cfg = SimpleNamespace(
        lot_size=50,
        risk_per_trade_percentage=1.0,
        account_capital=100000.0,
        risk_scale_enabled=True,
        risk_scale_stopout_factor=0.5,
        risk_scale_recovery_wins=10,
        risk_scale_atr_high=0.0,
        risk_scale_atr_high_factor=1.0,
        risk_scale_step_qty=50,
        risk_scale_min_qty=50,
        risk_scale_max_qty=500,
        gpt_position_sizing_enabled=False,
    )
    app.state = TradeState(open_orders=[], open_multi=[], open_directional=[], consecutive_stopouts=1)
    app._entry_iv_percentile_factor = lambda: 1.0
    app._router_size_multiplier = lambda: 1.0

    qty = app._entry_qty(atr_val=5.0)

    assert qty == 100


def test_trade_risk_meta_overrides_do_not_override_max_open_positions():
    app = NiftyScalper.__new__(NiftyScalper)
    app._last_router_decision = {
        "selected_preset": "CONSERVATIVE",
        "risk": {
            "stop_loss_pct": 0.2,
            "target_pct": 0.12,
            "trailing_sl_pct": 0.04,
            "size_multiplier": 0.5,
            "max_trades_per_day": 1,
            "cooldown_minutes": 20,
            "dir_sl_atr_mult": 1.1,
            "dir_tp_atr_mult": 1.8,
            "dir_trail_atr_mult": 0.7,
            "max_open_positions": 99,
        },
    }

    meta = app._trade_risk_meta_overrides()

    assert meta["ml_dynamic_stop_loss_pct"] == 0.2
    assert meta["ml_dynamic_size_multiplier"] == 0.5
    assert meta["ml_dynamic_preset"] == "CONSERVATIVE"
    assert "ml_dynamic_max_open_positions" not in meta
