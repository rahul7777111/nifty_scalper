"""Verification Suite for the Upgraded Institutional ML Trading Framework.

Comprehensively tests online feature stabilization, regime classification,
Platt Scaling probability calibration, risk management limits, slippage modeling,
and async database telemetry logging.
"""

from __future__ import annotations

import os
import sys
import time
import pytest
import numpy as np

# Adjust path to import from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from institutional_framework.feature_stabilizer import WelfordStabilizer, OnlineFeaturePipeline
from institutional_framework.regime_router import RegimeRouter, MarketRegime
from institutional_framework.drift_monitor import ConceptDriftMonitor
from institutional_framework.database import AsyncTelemetryDatabase
from institutional_framework.telemetry import TelemetryTracker
from institutional_framework.execution_engine import IndianOptionsExecutionEngine
from institutional_framework.risk_engine import OptionsRiskEngine, RiskLimits
from institutional_framework.stress_testing import OptionsStressTester
from institutional_framework.ml_signals import DynamicConsensusEnsemble
from institutional_framework.ml_pipeline import InstitutionalMLTradingPipeline, TARGET_FEATURES_24

# --- 1. FEATURE STABILIZER TESTS ---

def test_welford_stabilizer():
    stabilizer = WelfordStabilizer(decay=0.99)
    raw_vals = [10.0, 12.0, 11.0, 13.0, 9.0, 11.0, 12.0]
    
    for val in raw_vals:
        z = stabilizer.update(val)
        assert np.isfinite(z)
        
    assert stabilizer.count == len(raw_vals)
    assert abs(stabilizer.mean - 11.0) < 1.0
    assert stabilizer.var > 0.0

def test_online_feature_pipeline():
    pipeline = OnlineFeaturePipeline(feature_names=["f1", "f2"])
    row = {"f1": 100.5, "f2": 0.05}
    
    processed = pipeline.process_row(row)
    assert "f1" in processed
    assert "f2" in processed
    assert abs(processed["f1"]) < 10.0

# --- 2. REGIME ROUTER TESTS ---

def test_regime_router_classification():
    router = RegimeRouter()
    
    # Choppy/Range Regime (Low ADX, low volatility)
    regime = router.classify_regime(adx=10.0, atr=2.0, bb_bandwidth=0.03, current_iv=0.14, current_vix=12.0)
    assert regime == MarketRegime.CHOPPY
    weights = router.get_routing_weights(regime)
    assert weights.w_lr > weights.w_xgb
    assert weights.trading_enabled is True
    
    # Trending Regime (High ADX)
    regime = router.classify_regime(adx=32.0, atr=6.0, bb_bandwidth=0.16, current_iv=0.15, current_vix=14.0)
    assert regime == MarketRegime.TRENDING
    weights = router.get_routing_weights(regime)
    assert weights.w_xgb > weights.w_lr
    
    # Volatility Shock Regime (High IV spike)
    # Feed historical normal IV first, then trigger shock
    for _ in range(10):
        router.classify_regime(adx=15.0, atr=2.0, bb_bandwidth=0.04, current_iv=0.15, current_vix=13.0)
    regime = router.classify_regime(adx=15.0, atr=8.0, bb_bandwidth=0.25, current_iv=0.28, current_vix=25.0)
    assert regime == MarketRegime.VOLATILITY_SHOCK
    weights = router.get_routing_weights(regime)
    assert weights.w_rf > weights.w_lr
    
    # News Event (Safety Veto)
    regime = router.classify_regime(adx=10.0, atr=2.0, bb_bandwidth=0.03, current_iv=0.14, current_vix=12.0, is_news_window=True)
    assert regime == MarketRegime.NEWS_EVENT
    weights = router.get_routing_weights(regime)
    assert weights.trading_enabled is False

# --- 3. CONCEPT DRIFT MONITOR TESTS ---

def test_concept_drift_monitor():
    ref_stats = {"f1": (0.0, 1.0)}
    monitor = ConceptDriftMonitor(reference_features=ref_stats, window_size=50)
    
    # Record predictions & realizations
    for _ in range(10):
        monitor.record_prediction(pred_prob=0.65, actual_label=1)
        monitor.record_prediction(pred_prob=0.35, actual_label=0)
        
    brier = monitor.calculate_brier_score()
    assert brier >= 0.0
    assert brier <= 1.0
    
    # Verify PSI calculation
    expected = np.random.normal(0, 1, 100)
    actual = np.random.normal(0.5, 1.1, 100)
    psi = monitor.calculate_psi(expected, actual)
    assert psi >= 0.0

# --- 4. TELEMETRY & ASYNC DB WRITER TESTS ---

def test_async_database_and_telemetry(tmp_path):
    db_file = str(tmp_path / "telemetry_test.db")
    db = AsyncTelemetryDatabase(db_path=db_file, batch_size=2, flush_interval_sec=0.1)
    db.start()
    
    tracker = TelemetryTracker(db_writer=db)
    t_now = time.time()
    
    # Measure execution latency and shortfall
    record = tracker.measure_execution(
        t_tick_arrival=t_now - 0.20,
        t_signal_generated=t_now - 0.18,
        t_order_sent=t_now - 0.17,
        t_broker_fill=t_now,
        spot=22400.0,
        opt_bid=120.50,
        opt_ask=121.20,
        fill_price=121.30,
        order_type="BUY"
    )
    
    assert record.latency_ms >= 199.0
    assert record.slippage_ticks == 2.0  # (121.30 - 121.20) / 0.05
    assert record.quality in ["EXCELLENT", "FAIR", "POOR"]
    
    # Pause to allow queue background flush
    time.sleep(0.3)
    db.stop()
    
    # Verify records exist in SQLite
    import sqlite3
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM telemetry")
    count = cursor.fetchone()[0]
    conn.close()
    
    assert count > 0

# --- 5. INDIAN OPTIONS EXECUTION SIMULATOR TESTS ---

def test_indian_options_execution_engine():
    engine = IndianOptionsExecutionEngine(base_brokerage=20.0)
    
    # Test Buy transaction costs
    buy_costs = engine.calculate_nse_friction(premium=150.0, qty=75, strike=22400.0, is_sell=False)
    assert buy_costs.brokerage == 20.0
    assert buy_costs.stt == 0.0  # STT only on SELL
    assert buy_costs.stamp_duty > 0.0
    assert buy_costs.total_friction > buy_costs.brokerage
    
    # Test Sell transaction costs
    sell_costs = engine.calculate_nse_friction(premium=150.0, qty=75, strike=22400.0, is_sell=True)
    assert sell_costs.stt > 0.0
    
    # Test execution fill simulation
    report = engine.simulate_fill(
        target_price=120.0,
        qty=75,
        bid=119.5,
        ask=120.5,
        spot_velocity=10.0,
        current_iv=0.18,
        is_buy=True
    )
    
    assert report.fill_price > 0.0
    assert report.filled_qty <= 75
    assert report.latency_injected_sec > 0.0
    assert report.costs.total_friction > 0.0

# --- 6. OPTIONS CAPITAL RISK ENGINE TESTS ---

def test_options_risk_engine():
    limits = RiskLimits(max_open_positions=3)
    risk_engine = OptionsRiskEngine(starting_capital=200000.0, limits=limits)
    
    # Calculate position size using calibrated win prob (65%) under target volatility
    report = risk_engine.calculate_kelly_size(
        prob=0.65,
        risk_reward=1.5,
        spot_volatility=0.12,
        target_volatility=0.15,
        option_premium=100.0,
        lot_size=75
    )
    
    assert report.is_allowed is True
    assert report.recommended_qty > 0
    assert report.recommended_qty % 75 == 0  # Lot size aligned
    
    # Test Soft Drawdown trigger (restrict leverage)
    risk_engine.update_daily_pnl(realized_pnl=-4000.0, unrealized_pnl=0.0)  # -2.0% (Soft breach)
    report_soft = risk_engine.calculate_kelly_size(
        prob=0.65,
        risk_reward=1.5,
        spot_volatility=0.12,
        target_volatility=0.15,
        option_premium=100.0,
        lot_size=75
    )
    assert report_soft.leverage_mult < report.leverage_mult
    
    # Test Hard Drawdown trigger (Kill Switch)
    risk_engine.update_daily_pnl(realized_pnl=-8000.0, unrealized_pnl=0.0)  # -4.0% (Hard breach)
    report_hard = risk_engine.calculate_kelly_size(
        prob=0.65,
        risk_reward=1.5,
        spot_volatility=0.12,
        target_volatility=0.15,
        option_premium=100.0,
        lot_size=75
    )
    assert report_hard.is_allowed is False
    assert "KILL SWITCH ACTIVE" in report_hard.rejection_reason

# --- 7. MONTE CARLO STRESS TESTER TESTS ---

def test_monte_carlo_stress_testing():
    dummy_trades = np.array([200.0, -150.0, 300.0, -100.0, 150.0, -120.0, 400.0, -250.0] * 10)
    tester = OptionsStressTester(historical_trades=dummy_trades, initial_capital=50000.0)
    
    res = tester.run_stress_test(num_paths=50, path_length=30)
    assert res.risk_of_ruin_pct >= 0.0
    assert res.max_simulated_drawdown_pct >= 0.0
    assert len(res.sharpe_ratio_ci) == 2

# --- 8. PLATT SCALED CONSENSUS ENSEMBLE TESTS ---

def test_dynamic_consensus_ensemble():
    ensemble = DynamicConsensusEnsemble()
    
    # Generate dummy training data
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, (100, 24))
    y_train = rng.choice([0, 1], size=100, p=[0.50, 0.50])
    
    ensemble.fit(X_train, y_train)
    assert ensemble.use_calibration is True
    assert ensemble.lr_model is not None
    assert ensemble.xgb_model is not None
    
    # Run consensus prediction
    X_test = rng.normal(0, 1, (1, 24))
    signal = ensemble.predict_consensus(X_test, w_lr=0.40, w_rf=0.25, w_xgb=0.35)
    
    assert signal.consensus_prob >= 0.0
    assert signal.consensus_prob <= 1.0
    assert "lr" in signal.raw_probs
    assert "xgb" in signal.raw_probs

# --- 9. INTEGRATED COORDINATOR PIPELINE TESTS ---

def test_integrated_ml_trading_pipeline(tmp_path):
    db_file = str(tmp_path / "pipeline_prod.db")
    pipeline = InstitutionalMLTradingPipeline(db_path=db_file, initial_capital=100000.0)
    
    # Mock trained ensemble models
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, (60, 24))
    y_train = rng.choice([0, 1], size=60)
    pipeline.ensemble.fit(X_train, y_train)
    
    # Mock real-time feed dictionary
    tick_features = {name: float(rng.uniform(-2, 2)) for name in TARGET_FEATURES_24}
    
    res = pipeline.process_tick(
        raw_features=tick_features,
        spot_price=22450.0,
        opt_bid=130.20,
        opt_ask=131.00,
        spot_velocity=5.0,
        is_news_window=False
    )
    
    assert res is not None
    signal, risk_report, fill_report = res
    assert signal.consensus_prob > 0.0
    
    pipeline.shutdown()

# --- 10. PROVISIONAL SCORING & MISSING DRIFT TESTS ---

def test_provisional_scoring_and_missing_drift(tmp_path):
    db_file = str(tmp_path / "provisional_test.db")
    
    from institutional_framework.validation_analytics import ValidationAnalyticsEngine
    
    # Initialize engine with a temporary empty database (actual predictions = 0)
    # Debug/Demo modes set to False to prevent automatic backfills
    engine = ValidationAnalyticsEngine(db_path=db_file, validation_debug_mode=False, dashboard_demo_mode=False)
    
    # 1. Verify actual and synthetic counts are 0
    assert engine.get_actual_prediction_count() == 0
    assert engine.get_synthetic_prediction_count() == 0
    assert engine.get_actual_outcome_count() == 0
    assert engine.get_synthetic_outcome_count() == 0
    
    # 2. Verify drift metrics return "N/A" placeholders
    drift = engine.fetch_drift_and_decay()
    assert isinstance(drift, dict)
    assert drift["drift_metrics"]["feature_psi_pcr"] == "N/A"
    assert drift["drift_metrics"]["feature_psi_iv"] == "N/A"
    assert drift["drift_metrics"]["feature_psi_rsi"] == "N/A"
    assert drift["drift_metrics"]["label_drift"] == "N/A"
    assert drift["drift_metrics"]["concept_drift"] == "N/A"
    assert drift["drift_metrics"]["roc_auc"] == "N/A"
    assert drift["drift_metrics"]["sharpe"] == "N/A"
    assert drift["drift_metrics"]["profit_factor"] == "N/A"
    
    # 3. Verify final report data returns provisional zeroed scores
    final_report = engine.generate_final_report_data()
    assert isinstance(final_report, dict)
    assert final_report["edge_quality_score"] == 0.0
    assert final_report["deployment_readiness_score"] == 0.0
    assert final_report["capital_readiness_score"] == 0.0
    assert final_report["overall_auc"] == 0.5
    
    # 4. Verify markdown report generation completes successfully without any float formatting exceptions
    report_md = engine.generate_shadow_validation_report()
    assert report_md is not None
    assert "# NiftyScalper High-Fidelity Shadow Mode Verification" in report_md
    assert "PROVISIONAL" in report_md
    assert "PSI = N/A" in report_md
    assert "Expected Portfolio Drawdown" in report_md


def test_validation_summary_counts_match_predictions(tmp_path):
    import sqlite3

    from institutional_framework.validation_analytics import ValidationAnalyticsEngine

    db_file = str(tmp_path / "summary_counts.db")
    engine = ValidationAnalyticsEngine(db_path=db_file, validation_debug_mode=False, dashboard_demo_mode=False)

    with sqlite3.connect(engine.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (ts, symbol, direction, regime, prediction, probability, confidence, feature_snapshot_json, features_json, model_checksum, model_version, source, reason, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1.0, "NIFTY", "LONG", "Trending", 1.0, 0.72, 0.72, "{}", "{}", "abc", "v1", "ACTIVE", "trade-1", "2026-01-01 09:00:00"),
        )
        cursor.execute(
            "INSERT INTO predictions (ts, symbol, direction, regime, prediction, probability, confidence, feature_snapshot_json, features_json, model_checksum, model_version, source, reason, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (2.0, "NIFTY", "LONG", "Trending", 0.0, 0.41, 0.41, "{}", "{}", "abc", "v1", "SHADOW_FILTERED", "filtered-1", "2026-01-01 10:00:00"),
        )
        cursor.execute(
            "INSERT INTO predictions (ts, symbol, direction, regime, prediction, probability, confidence, feature_snapshot_json, features_json, model_checksum, model_version, source, reason, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (3.0, "NIFTY", "BLOCKED", "Trending", 0.0, 0.33, 0.33, "{}", "{}", "abc", "v1", "SHADOW_BLOCKED", "blocked-1", "2026-01-01 11:00:00"),
        )
        cursor.execute(
            "INSERT INTO trade_outcomes (trade_id, entry_time, exit_time, strategy, regime, iv_rank, pcr, prediction, confidence, pnl, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("trade-1", "2026-01-01 09:00:00", "2026-01-01 09:15:00", "Trend", "Trending", 16.5, 1.1, 1.0, 0.72, 1250.0, "2026-01-01 09:15:00"),
        )
        conn.commit()

    summary = engine.fetch_validation_summary()
    with sqlite3.connect(engine.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(1) FROM predictions")
        total_predictions = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(1) FROM predictions WHERE source = 'ACTIVE'")
        active_predictions = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(1) FROM predictions WHERE source = 'SHADOW_FILTERED'")
        filtered_predictions = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(1) FROM predictions WHERE source = 'SHADOW_BLOCKED'")
        blocked_predictions = cursor.fetchone()[0]

    assert summary["total_predictions"] == total_predictions
    assert summary["active_trades"] == active_predictions
    assert summary["filtered_signals"] == filtered_predictions
    assert summary["blocked_signals"] == blocked_predictions
    assert summary["signal_conversion_rate"] > 0
    assert summary["trade_conversion_rate"] > 0

