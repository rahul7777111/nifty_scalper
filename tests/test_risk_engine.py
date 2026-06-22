"""Comprehensive Unit Tests for the Institutional Risk Engine.

Tests advanced financial analytics, resampling modes, dynamic sizing algorithms,
options Greeks Black-Scholes solvers, machine learning classifiers, anomaly detection,
portfolio correlation benefits, walk-forward testing, and report exporters.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risk_engine.analytics import (
    calculate_comprehensive_metrics,
    compute_cvar,
    compute_drawdowns,
    compute_expectancy,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_ulcer_index,
    detect_fat_tails
)
from risk_engine.ml_risk import MLRiskEngine, RLOptimizer
from risk_engine.options_model import OptionsGreeksModel
from risk_engine.portfolio import PortfolioRiskEngine
from risk_engine.reporting import ReportingEngine
from risk_engine.simulator import SimulationEngine
from risk_engine.stress_testing import StressTestingEngine
from risk_engine.walk_forward import WalkForwardEngine


@pytest.fixture
def sample_trades() -> np.ndarray:
    """Fixture returning a standard option strategy trade sequence."""
    return np.array([
        2400.0, -1800.0, 3100.0, -1200.0, 4200.0, -2200.0, 1500.0, -800.0, 5000.0, -3200.0,
        2100.0, -900.0, 1800.0, -1100.0, 3600.0, -2800.0, 1500.0, -900.0, 4000.0, -1000.0
    ], dtype=np.float64)


# --- 1. FINANCIAL ANALYTICS TESTS ---

def test_drawdown_calculations(sample_trades):
    cum_equity = 100000.0 + np.cumsum(sample_trades)
    cum_equity = np.insert(cum_equity, 0, 100000.0)
    
    drawdowns, max_dd, peak_idx, trough_idx = compute_drawdowns(cum_equity)
    
    assert len(drawdowns) == len(cum_equity)
    assert max_dd >= 0.0
    assert peak_idx <= trough_idx
    assert drawdowns[peak_idx] == 0.0


def test_ulcer_index(sample_trades):
    cum_equity = 100000.0 + np.cumsum(sample_trades)
    cum_equity = np.insert(cum_equity, 0, 100000.0)
    
    ui = compute_ulcer_index(cum_equity)
    assert ui >= 0.0


def test_cvar_metric(sample_trades):
    cvar = compute_cvar(sample_trades, alpha=0.95)
    # Expected value must be positive (representing expected worst-case loss)
    assert cvar >= 0.0


def test_ratios_calculation(sample_trades):
    returns_pct = sample_trades / 100000.0
    
    sharpe = compute_sharpe_ratio(returns_pct)
    sortino = compute_sortino_ratio(returns_pct)
    pf = compute_profit_factor(sample_trades)
    exp = compute_expectancy(sample_trades)
    
    assert isinstance(sharpe, float)
    assert isinstance(sortino, float)
    assert pf > 0.0
    assert isinstance(exp, float)


def test_fat_tail_detection(sample_trades):
    fat_tail_info = detect_fat_tails(sample_trades)
    
    assert "excess_kurtosis" in fat_tail_info
    assert "is_fat_tailed" in fat_tail_info
    assert "power_law_alpha" in fat_tail_info


def test_comprehensive_metrics_packer(sample_trades):
    cum_equity = 100000.0 + np.cumsum(sample_trades)
    cum_equity = np.insert(cum_equity, 0, 100000.0)
    
    m = calculate_comprehensive_metrics(sample_trades, cum_equity, 100000.0)
    
    assert m["total_trades"] == len(sample_trades)
    assert m["sharpe_ratio"] > -100.0
    assert "ruin" in m


# --- 2. VECTORIZED SIMULATION ENGINE TESTS ---

def test_simulation_engine_resampling(sample_trades):
    engine = SimulationEngine(sample_trades)
    
    # Test modes
    shuffled = engine.resample_trades(mode="shuffle")
    bootstrapped = engine.resample_trades(mode="bootstrap")
    block_boot = engine.resample_trades(mode="block_bootstrap", block_size=4)
    
    assert len(shuffled) == len(sample_trades)
    assert len(bootstrapped) == len(sample_trades)
    assert len(block_boot) == len(sample_trades)
    
    # Shuffled should have identical elements
    assert sorted(shuffled.tolist()) == sorted(sample_trades.tolist())


def test_black_swan_injection(sample_trades):
    engine = SimulationEngine(sample_trades)
    
    # 100% crash injection to guarantee active triggers
    stressed = engine.inject_black_swan(
        sample_trades,
        crash_prob=1.0,
        vol_spike_factor=2.0,
        overnight_gap_loss=5000.0,
        slippage_spike=200.0
    )
    
    # Stressed should have much larger losses
    assert np.mean(stressed) < np.mean(sample_trades)


def test_dynamic_sizing_algorithms(sample_trades):
    engine = SimulationEngine(sample_trades)
    
    methods = ["fixed", "fixed_fractional", "kelly", "anti_martingale", "volatility_targeting", "drawdown_adaptive"]
    
    for m in methods:
        eq, sized, metrics = engine.apply_dynamic_sizing(
            sample_trades,
            sizing_method=m,
            start_capital=100000.0
        )
        assert len(eq) == len(sample_trades) + 1
        assert len(sized) == len(sample_trades)
        assert metrics["final_equity"] > 0.0


# --- 3. OPTIONS GREEKS AND DECAY TESTS ---

def test_black_scholes_solver():
    spot = 22000.0
    strike = 22000.0
    time = 7.0 / 365.0  # weekly option
    iv = 0.16
    
    price, greeks = OptionsGreeksModel.black_scholes(spot, strike, time, iv, option_type="call")
    
    assert price > 0.0
    assert 0.0 < greeks["delta"] < 1.0
    assert greeks["gamma"] > 0.0
    assert greeks["theta_daily"] < 0.0  # theta decay is negative PnL for longs


def test_options_decay_simulation():
    op_model = OptionsGreeksModel()
    
    sim = op_model.simulate_path_greeks_decay(
        start_spot=22000.0,
        strike=22000.0,
        days_to_expiry=7.0,
        start_iv=0.16,
        option_type="call",
        iv_crush_day=3,
        iv_crush_pct=0.20
    )
    
    assert len(sim["spot"]) == 8  # 7 days + day 0
    assert len(sim["premium"]) == 8
    # IV should crash on day 3
    assert sim["iv"][3] < 0.16


# --- 4. MACHINE LEARNING & REINFORCEMENT LEARNING TESTS ---

def test_ml_rolling_features(sample_trades):
    ml_engine = MLRiskEngine(sample_trades)
    X, y_fail, y_dd = ml_engine.generate_rolling_features()
    
    assert isinstance(X, pd.DataFrame)
    assert len(X) == len(y_fail) == len(y_dd)
    assert "rolling_win_rate" in X.columns


def test_ml_training_predictions(sample_trades):
    ml_engine = MLRiskEngine(sample_trades)
    res = ml_engine.train_predictive_models()
    
    assert 0.0 <= res["failure_probability"] <= 1.0
    assert 0.0 <= res["drawdown_probability"] <= 1.0
    assert "rolling_win_rate" in res["failure_feature_importance"]


def test_anomaly_detection(sample_trades):
    ml_engine = MLRiskEngine(sample_trades)
    anom_mask, scores = ml_engine.detect_anomalies(contamination=0.10)
    
    assert len(anom_mask) == len(sample_trades)
    assert len(scores) == len(sample_trades)
    assert isinstance(anom_mask, np.ndarray)


def test_rl_sizing_optimizer(sample_trades):
    rl_opt = RLOptimizer(sample_trades)
    res = rl_opt.train_q_learning(episodes=5)
    
    assert "training_rewards" in res
    assert len(res["optimized_equity_path"]) == len(sample_trades) + 1


# --- 5. MULTI-STRATEGY PORTFOLIO TESTS ---

def test_portfolio_correlation():
    # Multi-strategy data dictionary
    data = {
        "Scalper": np.array([1200, -800, 1500, -600, 2000, -1000]),
        "Hedge": np.array([-100, -100, -100, 4000, -100, -100]),
        "Futures": np.array([400, 800, -900, -200, 1500, 300])
    }
    
    port = PortfolioRiskEngine(data)
    corr, cov = port.compute_correlation_covariance()
    
    assert corr.shape == (3, 3)
    assert cov.shape == (3, 3)
    # Diagonals must be 1.0
    assert np.allclose(np.diag(corr), 1.0)


def test_portfolio_diversification_var():
    data = {
        "Scalper": np.array([1200, -800, 1500, -600, 2000, -1000]),
        "Hedge": np.array([-100, -100, -100, 4000, -100, -100]),
        "Futures": np.array([400, 800, -900, -200, 1500, 300])
    }
    
    port = PortfolioRiskEngine(data)
    weights = np.array([0.4, 0.2, 0.4])
    
    res = port.compute_portfolio_var(weights)
    
    assert res["diversification_benefit_value"] >= 0.0
    assert res["diversification_benefit_percent"] >= 0.0


def test_portfolio_joint_paths_simulation():
    data = {
        "Scalper": np.array([1200, -800, 1500, -600, 2000, -1000]),
        "Hedge": np.array([-100, -100, -100, 4000, -100, -100]),
        "Futures": np.array([400, 800, -900, -200, 1500, 300])
    }
    
    port = PortfolioRiskEngine(data)
    res = port.simulate_correlated_portfolio_paths(num_simulations=10, num_steps=5)
    
    assert len(res["final_returns"]) == 10
    assert len(res["representative_curves"]) > 0


# --- 6. PARAMETER STRESS TESTING & WALK-FORWARD TESTS ---

def test_stress_parameter_sensitivity(sample_trades):
    stress = StressTestingEngine(sample_trades)
    
    grid = stress.run_parameter_sensitivity_grid(
        slippage_range=[0, 100],
        failure_prob_range=[0.0, 0.2]
    )
    
    assert grid.shape == (2, 2)
    assert isinstance(grid, pd.DataFrame)


def test_overfitting_score(sample_trades):
    stress = StressTestingEngine(sample_trades)
    scores = stress.calculate_robustness_overfitting_score()
    
    assert 0.0 <= scores["strategy_robustness_score"] <= 100.0
    assert "classification" in scores


def test_walk_forward_rolling_validation(sample_trades):
    wf = WalkForwardEngine(sample_trades)
    res = wf.run_rolling_walk_forward(num_folds=2, train_ratio=0.60)
    
    assert res["num_folds"] > 0
    assert "performance_decay_percent" in res
    assert len(res["out_of_sample_equity_curve"]) > 0


# --- 7. EXPORTERS & HIGH-FIDELITY PDF TESTS ---

def test_report_exporters(sample_trades, tmp_path):
    from unittest.mock import patch, MagicMock
    
    cum_equity = 100000.0 + np.cumsum(sample_trades)
    cum_equity = np.insert(cum_equity, 0, 100000.0)
    metrics = calculate_comprehensive_metrics(sample_trades, cum_equity, 100000.0)
    reporter = ReportingEngine(metrics, sample_trades)

    json_file = str(tmp_path / "report.json")
    csv_file = str(tmp_path / "report.csv")
    xlsx_file = str(tmp_path / "report.xlsx")
    pdf_file = str(tmp_path / "report.pdf")

    reporter.export_json(json_file)
    reporter.export_csv(csv_file)
    reporter.export_excel(xlsx_file)

    # Write a pre-rendered tiny 1x1 png to satisfy the PDF generator
    chart_path = os.path.join(os.path.dirname(pdf_file), "report_chart.png")
    tiny_png = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x01'
        b'\x00\x00\x0c\x00\x01\x14\xbc\x0e\xdd\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    with open(chart_path, "wb") as f:
        f.write(tiny_png)

    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), (MagicMock(), MagicMock()))

    with patch("src.risk_engine.reporting.plt", mock_plt):
        reporter.generate_pdf_report(pdf_file)

    assert os.path.exists(json_file)
    assert os.path.exists(csv_file)
    assert os.path.exists(xlsx_file)
    assert os.path.exists(pdf_file)
    assert os.path.getsize(pdf_file) > 0
