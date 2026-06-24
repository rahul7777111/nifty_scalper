import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from strategy import NiftyScalper
from config import load_strategy_config
from market_data import Candle
from datetime import datetime
from ml_pipeline import (
    MLFeatureContext,
    TARGET_FEATURES,
    build_market_feature_vector,
    build_supervised_dataset,
    build_supervised_dataset_v2,
    walk_forward_backtest,
)
from ml_signals import train_ensemble, predict
from backtest_harness import simulate_simple
from strategy_allocator import get_regime_tuning, select_strategy_for_regime


def test_diagnostics_and_ml_simulate():
    class MockClient:
        def get_ltp(self, symbol):
            return 10.0
            
    cfg = load_strategy_config()
    sc = NiftyScalper(MockClient(), cfg)
    # run diagnostics
    diag = sc.run_enhancements_diagnostics()
    assert 'volatility_forecast' in diag
    assert 'regime' in diag

    # synth candles and evaluate
    candles = [Candle(datetime.now(), 100, 101, 99, 100 + i) for i in range(60)]
    res = sc.evaluate_entry_signals(candles)
    assert isinstance(res, dict)
    # simulate placement
    tr = sc._simulate_or_place_basic_directional(take=res.get('take', False), size=res.get('size', 0), reason=res.get('reason','test'), candles=candles)
    # if take was True, trade should be appended
    if res.get('take'):
        assert tr is not None
        assert len(sc.state.open_directional) >= 1
    else:
        assert tr is None


def test_ml_feature_pipeline_and_walk_forward():
    candles = [Candle(datetime.now(), 100 + i, 101 + i, 99 + i, 100 + i + (i % 3 - 1), volume=1000 + i * 10) for i in range(80)]
    ctx = MLFeatureContext(regime="trending", iv=0.21, delta=0.35, gamma=0.02, vega=0.1, theta=-0.03, spot=132.0)
    feats, names = build_market_feature_vector(candles, context=ctx)
    assert feats and names and len(feats) == len(names)
    assert any(n.startswith("regime_") for n in names)

    X, y, feature_names = build_supervised_dataset(candles, lookback=20, horizon=1)
    assert X and y and feature_names
    wf = walk_forward_backtest(X, y, n_splits=3, threshold=0.5)
    assert wf["aggregate"].get("samples", 0) > 0

    model = train_ensemble(X, y, feature_names=feature_names)
    assert model is not None
    probs = predict(model, X[:3])
    assert len(probs) == 3


def test_ml_feature_alignment_for_retraining_and_live_vector():
    candles = []
    base = datetime(2026, 5, 29, 9, 15)
    for idx in range(120):
        minute = idx * 5
        candles.append(
            Candle(
                base.replace(hour=9 + ((15 + minute) // 60), minute=(15 + minute) % 60),
                100.0 + idx * 0.2,
                100.5 + idx * 0.2,
                99.5 + idx * 0.2,
                100.1 + idx * 0.2 + ((idx % 4) - 1) * 0.05,
                volume=1500 + idx * 12,
            )
        )

    ctx = MLFeatureContext(
        regime="trending",
        iv=0.19,
        delta=0.32,
        gamma=0.015,
        vega=0.08,
        theta=-0.04,
        spot=123.5,
        option_price=2.4,
    )
    live_values, live_names = build_market_feature_vector(candles, context=ctx, lookback=30)
    X, _, feature_names = build_supervised_dataset_v2(candles, contexts=[ctx] * len(candles), lookback=30, horizon=5)

    assert len(live_values) == len(live_names)
    assert "is_closing_session" in live_names
    assert "realized_vol_30" in live_names
    assert "opening_range_breakout_strength" in live_names
    assert "rolling_range_position_20" in live_names
    assert "atr_pct_regime_10" in live_names
    assert "volatility_percentile_60" in live_names
    assert "volatility_regime_classifier" in live_names
    assert "bid_ask_spread_pct" in live_names
    assert "is_closing_session" in TARGET_FEATURES
    assert "range_pct" not in TARGET_FEATURES
    assert "ret_1" not in TARGET_FEATURES
    assert "roc_14" not in TARGET_FEATURES
    assert "supertrend_gap_pct" not in TARGET_FEATURES
    assert "volatility_percentile_60" not in TARGET_FEATURES
    assert "volatility_regime_classifier" not in TARGET_FEATURES
    assert X and feature_names
    assert all(name in live_names for name in TARGET_FEATURES)
    assert "ctx_time_sin" not in feature_names
    assert "ctx_time_cos" not in feature_names
    assert "price_to_spot_pct" not in feature_names


def test_train_ensemble_exports_holdout_metrics():
    candles = [Candle(datetime.now(), 100 + i, 101 + i, 99 + i, 100 + (i * 0.2) + ((i % 6) - 2) * 0.3, volume=1000 + i * 10) for i in range(180)]
    X, y, feature_names = build_supervised_dataset_v2(candles, lookback=20, horizon=5)
    bundle = train_ensemble(X, y, feature_names=feature_names, walk_forward_splits=3)
    assert bundle is not None
    assert "holdout" in bundle.metrics
    assert bundle.metrics["holdout_samples"] > 0
    assert "profit_factor" in bundle.metrics["holdout"]
    assert "sharpe" in bundle.metrics["holdout"]


def test_execution_realism_and_regime_tuning():
    pnl, trades = simulate_simple(
        [100, 102, 101, 104, 103],
        [1, 1, 0, -1, 0],
        position_size=2,
        slippage_bps=5.0,
        fee_per_order=2.5,
        partial_fill_rate=0.5,
        return_trades=True,
    )
    assert isinstance(pnl, float)
    assert trades and isinstance(trades[0], dict)
    assert "net_pnl" in trades[0]
    assert trades[0]["fill_fraction"] == 0.5

    tuning = get_regime_tuning("trending")
    assert tuning["ml_threshold"] > 0
    assert select_strategy_for_regime("trending")


def test_greeks_enrichment_and_fallbacks():
    class MockClient:
        def get_ltp(self, symbol):
            return 150.0

    cfg = load_strategy_config()
    sc = NiftyScalper(MockClient(), cfg)
    
    # 1. Verify Strategy side enrichment
    candles = [Candle(datetime.now(), 23900.0, 23910.0, 23890.0, 23900.8) for _ in range(60)]
    tr = sc._simulate_or_place_basic_directional(take=True, size=50, reason="test_bullish", candles=candles)
    
    assert tr is not None
    assert "legs" in tr
    leg = tr["legs"][0]
    assert leg.get("strike") == 23900.0
    assert leg.get("option_type") == "CE"
    assert leg.get("expiry") is not None

    # 2. Verify UI fallback parsing
    # Load unmocked ui module fresh from disk to bypass sys.modules mock pollution
    import importlib.machinery
    import importlib.util
    import os
    from unittest.mock import patch
    
    class DummyTk:
        def __init__(self, *args, **kwargs): pass
        def withdraw(self, *args, **kwargs): pass
        def deiconify(self, *args, **kwargs): pass
        def title(self, *args, **kwargs): pass
        def geometry(self, *args, **kwargs): pass
        def protocol(self, *args, **kwargs): pass
        def bind(self, *args, **kwargs): pass
        
    class DummyWidget:
        def __init__(self, *args, **kwargs): pass
        def grid(self, *args, **kwargs): pass
        def pack(self, *args, **kwargs): pass
        def place(self, *args, **kwargs): pass
        def configure(self, *args, **kwargs): pass
        def config(self, *args, **kwargs): pass
        def heading(self, *args, **kwargs): pass
        def column(self, *args, **kwargs): pass
    
    ui_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/ui.py"))
    loader = importlib.machinery.SourceFileLoader("fresh_ui", ui_path)
    spec = importlib.util.spec_from_loader("fresh_ui", loader)
    fresh_ui_module = importlib.util.module_from_spec(spec)
    
    import sys
    sys.modules["fresh_ui"] = fresh_ui_module
    try:
        with patch('tkinter.Tk', DummyTk), \
             patch('tkinter.ttk.Notebook', DummyWidget), \
             patch('tkinter.ttk.Frame', DummyWidget), \
             patch('tkinter.ttk.Label', DummyWidget), \
             patch('tkinter.ttk.Entry', DummyWidget), \
             patch('tkinter.ttk.Button', DummyWidget), \
             patch('tkinter.ttk.Checkbutton', DummyWidget), \
             patch('tkinter.ttk.Combobox', DummyWidget), \
             patch('tkinter.scrolledtext.ScrolledText', DummyWidget), \
             patch('tkinter.StringVar', DummyWidget), \
             patch('tkinter.BooleanVar', DummyWidget), \
             patch('tkinter.IntVar', DummyWidget):
            loader.exec_module(fresh_ui_module)
    finally:
        sys.modules.pop("fresh_ui", None)
        
    FreshScalperUI = fresh_ui_module.ScalperUI

    class DummyUI:
        def __init__(self):
            self._client = MockClient()
            self._trade_state = {}
        def _is_closed_trade_state(self, tid, st):
            return False
        def _parse_expiry(self, expiry_val):
            return FreshScalperUI._parse_expiry(self, expiry_val)

    ui = DummyUI()
    # Construct a legacy-style leg with missing strike, expiry, and option_type
    legacy_trade = {
        "status": "OPEN",
        "legs": [
            {
                "symbol": "NIFTY28DEC2623950CE",
                "quantity": 50,
                "side": "BUY",
                "entry_price": 150.0,
            }
        ]
    }
    ui._trade_state = {"T1": legacy_trade}
    
    rows, summary = FreshScalperUI._compute_open_leg_greeks(ui, 23988.85)
    assert len(rows) == 1
    row = rows[0]
    assert row.get("strike") == 23950.0
    assert row.get("type") == "CE"
    assert row.get("expiry") is not None

