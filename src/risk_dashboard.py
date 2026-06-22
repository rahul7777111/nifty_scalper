"""Institutional Quant Risk Dashboard.

Implements a professional dark-themed risk analytics suite in Streamlit
featuring Plotly visualizations, real-time simulation updates, 3D stress surfaces,
portfolio correlations, options Greeks decay, and AI/ML risk intelligence.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st

# Import our modular risk engines
from risk_engine.analytics import (
    calculate_comprehensive_metrics,
    compute_drawdowns,
    detect_fat_tails
)
from risk_engine.ml_risk import MLRiskEngine, RLOptimizer
from risk_engine.options_model import OptionsGreeksModel
from risk_engine.portfolio import PortfolioRiskEngine
from risk_engine.reporting import ReportingEngine
from risk_engine.simulator import SimulationEngine
from risk_engine.stress_testing import StressTestingEngine
from risk_engine.walk_forward import WalkForwardEngine

# --- PAGE INITIALIZATION & PREMIUM THEME CONFIGURATION ---
st.set_page_config(
    page_title="Institutional Risk & Stress Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injecting professional Dark mode, Glassmorphic styling, and Glowing accents
st.markdown(
    """
    <style>
    /* Global theme settings */
    .stApp {
        background-color: #0E1117;
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1A202C;
        border-right: 1px solid #2D3748;
    }
    /* Glassmorphism KPI cards */
    .metric-card {
        background: rgba(26, 32, 44, 0.65);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(66, 153, 225, 0.4);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #A0AEC0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        color: #F7FAFC;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-stress {
        color: #E53E3E; /* Red accent for stress metrics */
    }
    .metric-benefit {
        color: #38A169; /* Green accent for diversification/benefits */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- DATABASE / FILE LOADER (Q1 Integration) ---
def load_db_trades() -> np.ndarray:
    """Fetch completed trades from trades.db if available."""
    db_path = Path(__file__).parent / "trades.db"
    if not db_path.exists():
        db_path = Path(__file__).parent.parent / "trades.db"
    
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT realized_pnl FROM trades WHERE realized_pnl IS NOT NULL")
                rows = cursor.fetchall()
                if rows:
                    return np.array([float(r[0]) for r in rows])
        except Exception:
            pass
            
    # Standard high-fidelity option strategy backtest demo dataset if no database
    return np.array([
        2400.0, -1800.0, 3100.0, -1200.0, 4200.0, -2200.0, 1500.0, -800.0, 5000.0, -3200.0,
        2100.0, -900.0, 1800.0, -1100.0, 3600.0, -2800.0, 1500.0, -1300.0, 4500.0, -1900.0,
        2800.0, -1600.0, 3300.0, -1000.0, 5200.0, -2600.0, 1900.0, -700.0, 4100.0, -2900.0
    ])

# --- APP SETUP ---
st.title("🛡️ Institutional Risk & Stress Analytics Suite")
st.markdown("Hedge-Fund-Grade Monte Carlo Simulator, Regime Switching, Machine Learning Risk & Greek Exposures")

# Sidebar configuration inputs
st.sidebar.header("⚙️ Configuration Console")

trade_source = st.sidebar.selectbox(
    "Select Trade PnL Source:",
    ["Current Session Database (trades.db)", "Manual Comma-Separated Input", "Option Strategy Backtest (Demo)"]
)

if trade_source == "Current Session Database (trades.db)":
    raw_trades = load_db_trades()
    st.sidebar.info(f"Loaded {len(raw_trades)} trades from database ledger.")
elif trade_source == "Manual Comma-Separated Input":
    manual_pnl = st.sidebar.text_area(
        "Enter comma-separated PnL values:",
        "1500, -800, 2400, -1200, 3100, -1500, 4200, -2200, 5000, -3200, 1800, -900, 2700"
    )
    try:
        raw_trades = np.array([float(x.strip()) for x in manual_pnl.split(",") if x.strip()])
    except ValueError:
        st.sidebar.error("Invalid numeric inputs. Falling back to Demo dataset.")
        raw_trades = load_db_trades()
else:
    raw_trades = load_db_trades()
    st.sidebar.info(f"Loaded demo options backtest with {len(raw_trades)} trades.")

# Basic Inputs
st.sidebar.subheader("Simulation Controls")
mc_mode = st.sidebar.selectbox("Simulation Mode:", ["bootstrap", "shuffle", "block_bootstrap"])
block_size = st.sidebar.slider("Block Bootstrap Size (k):", 2, 15, 5, help="Controls correlation persistence")
sims_count = st.sidebar.number_input("Number of Simulations:", 1000, 100000, 5000, step=1000)

# Sizing Inputs
st.sidebar.subheader("Sizing & Risk Parameters")
sizing_method = st.sidebar.selectbox(
    "Position Sizing Method:",
    ["fixed", "fixed_fractional", "kelly", "anti_martingale", "volatility_targeting", "drawdown_adaptive"]
)
start_capital = st.sidebar.number_input("Initial Capital (₹):", 10000.0, 10000000.0, 100000.0, step=10000.0)
lot_size = st.sidebar.number_input("Lot Size / Contract Factor:", 0.1, 50.0, 1.0, step=0.5)
leverage = st.sidebar.number_input("Portfolio Leverage Multiplier:", 0.5, 10.0, 1.0, step=0.5)

max_risk_pct = st.sidebar.slider("Max Risk per Trade (% of Capital):", 0.5, 10.0, 2.0, step=0.5) / 100.0
max_exposure_pct = st.sidebar.slider("Max Portfolio Exposure (% of Capital):", 10, 100, 80) / 100.0

# Injected Stress / Black Swan parameters
st.sidebar.subheader("🔥 Fat-Tail / Black Swan Injector")
enable_black_swan = st.sidebar.checkbox("Activate Black Swan Engine", value=False)
crash_prob = st.sidebar.slider("Crash Probability per Trade (%):", 0.1, 10.0, 1.0, step=0.1) / 100.0
vol_spike = st.sidebar.slider("Volatility Spike Multiplier:", 1.5, 5.0, 2.0, step=0.5)
gap_loss = st.sidebar.number_input("Overnight Gap Loss (₹):", 1000.0, 100000.0, 5000.0, step=1000.0)
slip_spike = st.sidebar.number_input("Slippage Spike (₹):", 50.0, 2000.0, 200.0, step=50.0)

# --- CORE SIMULATOR & METRIC ENGINE LAUNCH ---
engine = SimulationEngine(raw_trades)

bs_config = {
    "enable": enable_black_swan,
    "crash_probability": crash_prob,
    "volatility_spike_factor": vol_spike,
    "overnight_gap_loss": gap_loss,
    "slippage_spike": slip_spike
}

sizing_config = {
    "method": sizing_method,
    "start_capital": start_capital,
    "lot_size_factor": lot_size,
    "max_risk_pct": max_risk_pct,
    "max_exposure_pct": max_exposure_pct,
    "leverage_multiplier": leverage,
    "risk_controls": {
        "daily_loss_limit": start_capital * 0.10,
        "max_dd_shutdown": 0.40,
        "consecutive_loss_pause": 3,
        "pause_duration_trades": 3
    }
}

# --- TABS CREATION ---
tabs = st.tabs([
    "⚡ Executive Simulator",
    "📊 Advanced Analytics",
    "🌐 Portfolio & Regimes",
    "🔥 Stress & Robustness",
    "📈 Options & Greeks",
    "🧠 AI/ML Risk Intelligence",
    "📁 Walk-Forward & Exporter"
])

# --- TAB 1: EXECUTIVE SIMULATOR ---
with tabs[0]:
    st.header("⚡ Hedge-Fund Monte Carlo Simulator")
    
    # Live simulation streaming / incremental runs simulation
    run_btn = st.button("⚡ Execute High-Performance Run")
    
    if run_btn:
        with st.spinner(f"Running {sims_count} multi-scenario simulations..."):
            sim_res = engine.run_monte_carlo(
                num_simulations=sims_count,
                mode=mc_mode,
                block_size=block_size,
                black_swan_config=bs_config,
                sizing_config=sizing_config
            )
            
        final_rets = sim_res["final_returns"]
        max_dds = sim_res["max_drawdowns"]
        curves = sim_res["representative_curves"]
        
        # Calculate base backtest path
        base_curve = [start_capital]
        c = start_capital
        for pnl in raw_trades:
            c += (pnl * lot_size)
            base_curve.append(c)
            
        # Core metric summaries
        median_return = np.median(final_rets)
        sorted_dds = np.sort(max_dds)
        var_95_dd = sorted_dds[int(sims_count * 0.95)] if sims_count > 0 else 0.0
        worst_dd = sorted_dds[-1]
        
        ruin_count = np.sum(max_dds >= 0.40)
        prob_ruin = ruin_count / sims_count

        # Display KPIs
        kpi_cols = st.columns(6)
        
        with kpi_cols[0]:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Original return</div><div class="metric-value">₹{(base_curve[-1] - start_capital):,.2f}</div></div>', unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Median Return</div><div class="metric-value">₹{median_return:,.2f}</div></div>', unsafe_allow_html=True)
        with kpi_cols[2]:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Worst Drawdown</div><div class="metric-value metric-stress">{worst_dd * 100:.2f}%</div></div>', unsafe_allow_html=True)
        with kpi_cols[3]:
            st.markdown(f'<div class="metric-card"><div class="metric-title">95% VaR Drawdown</div><div class="metric-value metric-stress">{var_95_dd * 100:.2f}%</div></div>', unsafe_allow_html=True)
        with kpi_cols[4]:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Ruin Probability</div><div class="metric-value metric-stress">{prob_ruin * 100:.2f}%</div></div>', unsafe_allow_html=True)
        with kpi_cols[5]:
            # Expectancy
            wins = raw_trades[raw_trades > 0]
            losses = raw_trades[raw_trades < 0]
            win_rate = len(wins) / len(raw_trades) if len(raw_trades) > 0 else 0.5
            exp = (win_rate * np.mean(wins) if len(wins)>0 else 0) + ((1-win_rate) * np.mean(losses) if len(losses)>0 else 0)
            st.markdown(f'<div class="metric-card"><div class="metric-title">Expectancy</div><div class="metric-value">₹{exp:,.2f}</div></div>', unsafe_allow_html=True)

        st.write("")
        
        # Plotly multi-scenario paths
        fig = go.Figure()
        
        # Plot Cyan transparent sample paths
        for path in curves[:50]:
            fig.add_trace(go.Scatter(y=path, mode='lines', line=dict(color='rgba(66, 153, 225, 0.12)', width=1), showlegend=False))
            
        # Median path
        # Rank paths by final value and find the median curve
        ranked_curves = sorted(curves, key=lambda c: c[-1])
        median_curve = ranked_curves[len(ranked_curves) // 2]
        worst_5pct_curve = ranked_curves[int(len(ranked_curves) * 0.05)]

        fig.add_trace(go.Scatter(y=base_curve, mode='lines', name='Original Backtest Path', line=dict(color='#48BB78', width=2.5)))
        fig.add_trace(go.Scatter(y=median_curve, mode='lines', name='Median Path', line=dict(color='#ED8936', width=2.5)))
        fig.add_trace(go.Scatter(y=worst_5pct_curve, mode='lines', name='Worst 5% Stress Path', line=dict(color='#E53E3E', width=2, dash='dash')))
        
        fig.update_layout(
            title="Vectorized Monte Carlo Paths & Risk Trajectories",
            plot_bgcolor="#1A202C",
            paper_bgcolor="#0E1117",
            font_color="#E2E8F0",
            xaxis_title="Trades / Timeline",
            yaxis_title="Account Balance (₹)",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(26, 32, 44, 0.8)")
        )
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown Underwater Area plot
        fig_dd = go.Figure()
        peaks = np.maximum.accumulate(base_curve)
        dds = (peaks - base_curve) / peaks
        
        fig_dd.add_trace(go.Scatter(
            y=-dds * 100,
            fill='tozeroy',
            mode='lines',
            name='Original Drawdown',
            line=dict(color='#C53030', width=1.5),
            fillcolor='rgba(197, 48, 48, 0.3)'
        ))
        
        fig_dd.update_layout(
            title="Underwater Drawdown Profile (%)",
            plot_bgcolor="#1A202C",
            paper_bgcolor="#0E1117",
            font_color="#E2E8F0",
            xaxis_title="Trades",
            yaxis_title="Drawdown (%)"
        )
        st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.info("Click '⚡ Execute High-Performance Run' above to launch 5000+ Monte Carlo pathways and stress tests.")

# --- TAB 2: ADVANCED ANALYTICS ---
with tabs[1]:
    st.header("📊 Advanced Risk Metrics & Return Distributions")
    
    # Advanced stats calculations
    base_curve = [start_capital]
    c = start_capital
    for pnl in raw_trades:
        c += (pnl * lot_size)
        base_curve.append(c)
        
    metrics = calculate_comprehensive_metrics(raw_trades * lot_size, np.array(base_curve), start_capital)
    
    # 2x2 grid of ratios and stats
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Sharpe Ratio</div><div class="metric-value">{metrics["sharpe_ratio"]:.2f}</div></div>', unsafe_allow_html=True)
    with stat_cols[1]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Sortino Ratio</div><div class="metric-value">{metrics["sortino_ratio"]:.2f}</div></div>', unsafe_allow_html=True)
    with stat_cols[2]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Calmar Ratio</div><div class="metric-value">{metrics["calmar_ratio"]:.2f}</div></div>', unsafe_allow_html=True)
    with stat_cols[3]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Ulcer Index</div><div class="metric-value metric-stress">{metrics["ulcer_index"]:.2f}</div></div>', unsafe_allow_html=True)

    st.write("")
    
    # Histogram + KDE Distribution Analysis
    st.subheader("Distribution Analysis of Strategy Returns")
    
    fig_dist = px.histogram(
        raw_trades * lot_size,
        nbins=20,
        title="PnL Return Frequency Histogram & KDE Overlay",
        labels={'value': 'Trade PnL (₹)'},
        color_discrete_sequence=['#3182CE'],
        opacity=0.7
    )
    
    # Compute normal distribution overlay
    mean = np.mean(raw_trades * lot_size)
    std = np.std(raw_trades * lot_size)
    x_val = np.linspace(min(raw_trades * lot_size), max(raw_trades * lot_size), 100)
    pdf = stats.norm.pdf(x_val, mean, std)
    
    # Scale PDF to fit histogram
    pdf_scaled = pdf * len(raw_trades) * (max(raw_trades * lot_size) - min(raw_trades * lot_size)) / 20.0
    
    fig_dist.add_trace(go.Scatter(
        x=x_val,
        y=pdf_scaled,
        mode='lines',
        name='Normal Distribution Comparison',
        line=dict(color='#ED8936', width=2)
    ))
    
    fig_dist.update_layout(
        plot_bgcolor="#1A202C",
        paper_bgcolor="#0E1117",
        font_color="#E2E8F0"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Skewness and kurtosis
    fat_tail_info = detect_fat_tails(raw_trades * lot_size)
    
    det_cols = st.columns(3)
    with det_cols[0]:
        st.write(f"**Skewness:** {metrics['skewness']:.4f}")
        st.write("*(Shows return asymmetry. Positive skewness represents more extreme winning trades.)*")
    with det_cols[1]:
        st.write(f"**Excess Kurtosis:** {metrics['kurtosis']:.4f}")
        st.write("*(Standard Normal distribution excess kurtosis = 0.0. Higher values signal heavy tails.)*")
    with det_cols[2]:
        st.write(f"**Fat-Tail Status:** {'Leptokurtic (Fat-Tailed)' if fat_tail_info['is_fat_tailed'] else 'Normal Distribution'}")
        st.write(f"**Estimated Tail Power-Law Alpha:** {fat_tail_info['power_law_alpha']:.4f}")

# --- TAB 3: PORTFOLIO & REGIMES ---
with tabs[2]:
    st.header("🌐 Multi-Strategy Portfolio Correlations & Regime Switching")
    
    st.subheader("Correlation and Covariance Matrix Analysis")
    
    # Load multi-strategy engine
    port_engine = PortfolioRiskEngine(strategies_data={})
    corr, cov = port_engine.compute_correlation_covariance()
    
    # Render Correlation Matrix Heatmap
    fig_heat = px.imshow(
        corr,
        x=port_engine.strategy_names,
        y=port_engine.strategy_names,
        color_continuous_scale='Viridis',
        title="Pearson Correlation Heatmap Between Core Strategies",
        text_auto=".2f"
    )
    fig_heat.update_layout(
        plot_bgcolor="#1A202C",
        paper_bgcolor="#0E1117",
        font_color="#E2E8F0"
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Diversification Benefits
    st.subheader("Portfolio Diversification & VaR Impact")
    
    # Dynamic portfolio weights slider
    w_cols = st.columns(3)
    w_scalp = w_cols[0].slider("Options Scalping Weight:", 0.0, 1.0, 0.4, step=0.1)
    w_hedge = w_cols[1].slider("Tail Hedging Weight:", 0.0, 1.0, 0.2, step=0.1)
    w_swing = w_cols[2].slider("Swing Futures Weight:", 0.0, 1.0, 0.4, step=0.1)
    
    weights = np.array([w_scalp, w_hedge, w_swing])
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
        
    port_var_metrics = port_engine.compute_portfolio_var(weights)
    
    div_cols = st.columns(4)
    with div_cols[0]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Diversified VaR (95%)</div><div class="metric-value">₹{port_var_metrics["diversified_var"]:,.2f}</div></div>', unsafe_allow_html=True)
    with div_cols[1]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Undiversified VaR (95%)</div><div class="metric-value">₹{port_var_metrics["undiversified_var"]:,.2f}</div></div>', unsafe_allow_html=True)
    with div_cols[2]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Diversification Benefit (₹)</div><div class="metric-value metric-benefit">₹{port_var_metrics["diversification_benefit_value"]:,.2f}</div></div>', unsafe_allow_html=True)
    with div_cols[3]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Diversification Benefit (%)</div><div class="metric-value metric-benefit">{port_var_metrics["diversification_benefit_percent"]:.2f}%</div></div>', unsafe_allow_html=True)

    # Simulated joint Monte Carlo correlated paths
    st.subheader("Correlated Multi-Strategy Portfolio Monte Carlo Simulation")
    run_port_btn = st.button("🌐 Run Correlated Joint Simulation")
    
    if run_port_btn:
        with st.spinner("Executing correlated joint simulation via Cholesky decomposition..."):
            port_res = port_engine.simulate_correlated_portfolio_paths(weights=weights)
            
        fig_port = go.Figure()
        for path in port_res["representative_curves"][:20]:
            fig_port.add_trace(go.Scatter(y=path, mode='lines', line=dict(color='rgba(56, 178, 172, 0.15)', width=1), showlegend=False))
            
        # Draw mean portfolio curve
        mean_path = np.mean(port_res["representative_curves"], axis=0)
        fig_port.add_trace(go.Scatter(y=mean_path, mode='lines', name='Expected Portfolio Path', line=dict(color='#319795', width=3.0)))
        
        fig_port.update_layout(
            plot_bgcolor="#1A202C",
            paper_bgcolor="#0E1117",
            font_color="#E2E8F0",
            xaxis_title="Simulation Steps",
            yaxis_title="Account Equity (₹)"
        )
        st.plotly_chart(fig_port, use_container_width=True)

# --- TAB 4: STRESS & ROBUSTNESS ---
with tabs[3]:
    st.header("🔥 Parameter Sensitivity Stress Testing & Robustness Scores")
    
    stress_engine = StressTestingEngine(raw_trades, start_capital)
    
    # 3D Simulation Surface
    st.subheader("3D Sensitivity Surface: Slippage vs. Latency vs. Ruin Probability")
    
    # Compute 3D surface grid data
    slips = np.linspace(0, 500, 10)
    fails = np.linspace(0, 0.3, 10)
    
    z_ruin = np.zeros((10, 10))
    for s_i, slip in enumerate(slips):
        for f_i, fail in enumerate(fails):
            path_returns = []
            # Calculate 100 paths per grid point
            for _ in range(100):
                path = np.random.choice(raw_trades, size=len(raw_trades), replace=True)
                failures = np.random.random(size=len(raw_trades)) < fail
                adjusted = np.where(failures, -50.0, path) - slip
                path_returns.append(np.sum(adjusted))
            
            worst_dd = np.max((start_capital - (start_capital + np.cumsum(path_returns))) / start_capital)
            z_ruin[s_i, f_i] = float(worst_dd)

    fig_3d = go.Figure(data=[go.Surface(z=z_ruin, x=fails, y=slips, colorscale='Reds')])
    fig_3d.update_layout(
        title="Strategy Drawdown Stress Surface",
        scene=dict(
            xaxis_title="Execution Failure Prob",
            yaxis_title="Slippage Penalty (₹)",
            zaxis_title="Max Drawdown Severity"
        ),
        plot_bgcolor="#1A202C",
        paper_bgcolor="#0E1117",
        font_color="#E2E8F0"
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # Strategy Robustness score and Overfitting
    st.subheader("Hedge-Fund Robustness & Strategy Overfitting score")
    
    with st.spinner("Analyzing strategy robustness and training perturbations..."):
        robust_metrics = stress_engine.calculate_robustness_overfitting_score()
        
    score_cols = st.columns(4)
    with score_cols[0]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Robustness Score</div><div class="metric-value">{robust_metrics["strategy_robustness_score"]:.1f}/100</div></div>', unsafe_allow_html=True)
    with score_cols[1]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Overfitting Probability</div><div class="metric-value metric-stress">{robust_metrics["overfitting_probability"] * 100:.2f}%</div></div>', unsafe_allow_html=True)
    with score_cols[2]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Stability Index</div><div class="metric-value">{robust_metrics["stability_index"]:.4f}</div></div>', unsafe_allow_html=True)
    with score_cols[3]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Adversarial Max DD</div><div class="metric-value metric-stress">{robust_metrics["adversarial_max_drawdown"] * 100:.2f}%</div></div>', unsafe_allow_html=True)

    st.write("")
    st.write(f"**Classification:** `{robust_metrics['classification']}`")
    st.write("*(The Robustness Score penalizes strategies whose returns drop severely under randomized trade removal, noise injection, or worst-case trade sequences.)*")

# --- TAB 5: OPTIONS & GREEKS ---
with tabs[4]:
    st.header("📈 Realistic Options Pricing & Greeks Stress Simulator")
    
    op_model = OptionsGreeksModel()
    
    g_cols = st.columns(5)
    spot = g_cols[0].number_input("Spot Price (₹):", 1000.0, 50000.0, 22000.0, step=100.0)
    strike = g_cols[1].number_input("Strike Price (₹):", 1000.0, 50000.0, 22000.0, step=100.0)
    days_rem = g_cols[2].slider("Days to Expiration:", 0.5, 30.0, 7.0, step=0.5)
    iv = g_cols[3].slider("Implied Volatility (IV):", 0.05, 1.0, 0.16, step=0.01)
    opt_type = g_cols[4].selectbox("Option Type:", ["call", "put"])
    
    price, greeks = op_model.black_scholes(spot, strike, days_rem / 365.0, iv, option_type=opt_type)
    
    # KPI Greeks
    greek_cols = st.columns(5)
    with greek_cols[0]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Premium Price</div><div class="metric-value">₹{price:.2f}</div></div>', unsafe_allow_html=True)
    with greek_cols[1]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Delta (Δ)</div><div class="metric-value">{greeks["delta"]:.4f}</div></div>', unsafe_allow_html=True)
    with greek_cols[2]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Gamma (Γ)</div><div class="metric-value">{greeks["gamma"]:.6f}</div></div>', unsafe_allow_html=True)
    with greek_cols[3]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Theta Daily (Θ)</div><div class="metric-value metric-stress">₹{greeks["theta_daily"]:.2f}</div></div>', unsafe_allow_html=True)
    with greek_cols[4]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Vega 1% (𝒱)</div><div class="metric-value">₹{greeks["vega_1pct"]:.2f}</div></div>', unsafe_allow_html=True)

    st.subheader("Days-to-Expiry Decay & Option Greek Acceleration Curves")
    
    # Option Greeks decay path simulation
    sim_gp = op_model.simulate_path_greeks_decay(
        start_spot=spot,
        strike=strike,
        days_to_expiry=days_rem,
        start_iv=iv,
        option_type=opt_type,
        iv_crush_day=3,
        iv_crush_pct=0.25
    )
    
    # Plot non-linear theta decay and gamma explosion
    fig_dec = go.Figure()
    fig_dec.add_trace(go.Scatter(y=sim_gp["premium"], mode='lines+markers', name='Option Premium (₹)', line=dict(color='#ED8936', width=2)))
    fig_dec.add_trace(go.Scatter(y=sim_gp["gamma"] * 100000, mode='lines', name='Gamma (Scaled x100,000)', line=dict(color='#E53E3E', width=2)))
    
    fig_dec.update_layout(
        title="Realistic Options Theta Decay & Gamma Spikes near Expiration",
        plot_bgcolor="#1A202C",
        paper_bgcolor="#0E1117",
        font_color="#E2E8F0",
        xaxis_title="Simulated Days Elapsed",
        yaxis_title="Option Value Metric"
    )
    st.plotly_chart(fig_dec, use_container_width=True)

# --- TAB 6: AI/ML RISK INTELLIGENCE ---
with tabs[5]:
    st.header("🧠 AI/ML Risk Intelligence & Reinforcement Learning")
    
    ml_engine = MLRiskEngine(raw_trades)
    
    with st.spinner("Analyzing risk parameters and training ML classifiers..."):
        ml_res = ml_engine.train_predictive_models()
        
    ml_cols = st.columns(3)
    with ml_cols[0]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Predicted Failure Prob</div><div class="metric-value metric-stress">{ml_res["failure_probability"] * 100:.1f}%</div></div>', unsafe_allow_html=True)
    with ml_cols[1]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Predicted Drawdown Prob</div><div class="metric-value metric-stress">{ml_res["drawdown_probability"] * 100:.1f}%</div></div>', unsafe_allow_html=True)
    with ml_cols[2]:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Latest Win-Rate Feature</div><div class="metric-value">{ml_res["latest_win_rate"] * 100:.1f}%</div></div>', unsafe_allow_html=True)

    # Anomaly Detection Scatterplot
    st.subheader("Isolation Forest Anomaly Detection")
    anomalies_mask, scores = ml_engine.detect_anomalies(contamination=0.08)
    
    fig_anom = go.Figure()
    
    # Normal trades
    normal_trades = raw_trades[~anomalies_mask]
    fig_anom.add_trace(go.Scatter(
        x=np.where(~anomalies_mask)[0],
        y=normal_trades,
        mode='markers',
        name='Normal Trade PnL',
        marker=dict(color='#3182CE', size=8)
    ))
    
    # Anomalous trades
    anom_trades = raw_trades[anomalies_mask]
    fig_anom.add_trace(go.Scatter(
        x=np.where(anomalies_mask)[0],
        y=anom_trades,
        mode='markers',
        name='Anomalous Outlier / Failure',
        marker=dict(color='#E53E3E', size=12, line=dict(color='#F7FAFC', width=1.5))
    ))
    
    fig_anom.update_layout(
        title="Execution Anomaly & Outlier Identification",
        plot_bgcolor="#1A202C",
        paper_bgcolor="#0E1117",
        font_color="#E2E8F0",
        xaxis_title="Trade Index",
        yaxis_title="Trade PnL (₹)"
    )
    st.plotly_chart(fig_anom, use_container_width=True)

    # Reinforcement Learning Optimizer comparison (Q2 Integration)
    st.subheader("🧠 Reinforcement Learning Sizing Optimizer")
    
    run_rl_btn = st.button("🧠 Train RL Position Sizing Agent")
    
    if run_rl_btn:
        rl_opt = RLOptimizer(raw_trades)
        with st.spinner("Training RL Sizing Agent (Q-Learning) over 100 episodes..."):
            rl_res = rl_opt.train_q_learning(episodes=100)
            
        fig_rl = go.Figure()
        fig_rl.add_trace(go.Scatter(y=rl_res["baseline_equity_path"], mode='lines', name='Baseline standard Sizing (₹)', line=dict(color='#E53E3E', width=2)))
        fig_rl.add_trace(go.Scatter(y=rl_res["optimized_equity_path"], mode='lines', name='RL Sizing Optimized (₹)', line=dict(color='#38A169', width=2.5)))
        
        fig_rl.update_layout(
            title="Equity Curve Comparison: Standard Sizing vs. RL-Optimized Position Sizing",
            plot_bgcolor="#1A202C",
            paper_bgcolor="#0E1117",
            font_color="#E2E8F0",
            xaxis_title="Trades",
            yaxis_title="Account Balance (₹)"
        )
        st.plotly_chart(fig_rl, use_container_width=True)
        
        # Display Action frequency
        freq = rl_res["actions_frequency"]
        st.write(f"**RL Agent Actions Decision Table:** Safe (0.5x): `{freq['safe_0.5x']}` &nbsp;|&nbsp; Standard (1.0x): `{freq['standard_1.0x']}` &nbsp;|&nbsp; Aggressive (2.0x): `{freq['aggressive_2.0x']}`")
    else:
        st.info("Click '🧠 Train RL Position Sizing Agent' above to train a Q-learning agent on your options strategy to optimize dynamic stop-losses and contract scaling.")

# --- TAB 7: WALK-FORWARD & EXPORTER ---
with tabs[6]:
    st.header("📁 Walk-Forward Optimization & Report Exporters")
    
    # Walk-Forward rolling optimization
    st.subheader("Rolling Walk-Forward Validation")
    
    wf_engine = WalkForwardEngine(raw_trades, start_capital)
    
    run_wf_btn = st.button("📁 Execute Chronological Walk-Forward Folds")
    
    if run_wf_btn:
        with st.spinner("Splitting folds and executing rolling parameter search..."):
            wf_res = wf_engine.run_rolling_walk_forward(num_folds=3, train_ratio=0.60)
            
        wf_cols = st.columns(3)
        with wf_cols[0]:
            st.markdown(f'<div class="metric-card"><div class="metric-title">In-Sample Sharpe</div><div class="metric-value">{wf_res["average_in_sample_sharpe"]:.2f}</div></div>', unsafe_allow_html=True)
        with wf_cols[1]:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Out-of-Sample Sharpe</div><div class="metric-value">{wf_res["average_out_of_sample_sharpe"]:.2f}</div></div>', unsafe_allow_html=True)
        with wf_cols[2]:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Performance Decay (%)</div><div class="metric-value metric-stress">{wf_res["performance_decay_percent"]:.1f}%</div></div>', unsafe_allow_html=True)

        st.write("")
        
        # Walk forward curves comparison
        fig_wf = go.Figure()
        fig_wf.add_trace(go.Scatter(y=wf_res["baseline_equity_curve"], mode='lines', name='Baseline Equity Curve (₹)', line=dict(color='#E53E3E', width=2)))
        fig_wf.add_trace(go.Scatter(y=wf_res["out_of_sample_equity_curve"], mode='lines', name='Walk-Forward Validated Equity Curve (₹)', line=dict(color='#38A169', width=2.5)))
        
        fig_wf.update_layout(
            title=" chronological Out-of-Sample Walk-Forward Curve vs. Baseline",
            plot_bgcolor="#1A202C",
            paper_bgcolor="#0E1117",
            font_color="#E2E8F0",
            xaxis_title="Steps / Trades",
            yaxis_title="Account Balance (₹)"
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    # Exporters and report generation
    st.subheader("Hedge-Fund Executive Exporters")
    
    # Auto-compile metrics for reporter
    base_curve = [start_capital]
    c = start_capital
    for pnl in raw_trades:
        c += (pnl * lot_size)
        base_curve.append(c)
        
    full_metrics = calculate_comprehensive_metrics(raw_trades * lot_size, np.array(base_curve), start_capital)
    
    # Initialize reporter
    reporter = ReportingEngine(full_metrics, raw_trades * lot_size)
    
    # Save folder definition
    save_dir = Path(__file__).parent.parent / "reports"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = str(save_dir / "institutional_risk_report.pdf")
    xlsx_path = str(save_dir / "institutional_risk_report.xlsx")
    csv_path = str(save_dir / "trade_pnl_log.csv")
    json_path = str(save_dir / "risk_metrics.json")
    
    rep_cols = st.columns(4)
    
    # PDF download
    if rep_cols[0].button("📄 Compile Institutional PDF Report"):
        with st.spinner("Drawing high-fidelity PDF layouts..."):
            reporter.generate_pdf_report(pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download PDF Report",
                data=f.read(),
                file_name="institutional_risk_report.pdf",
                mime="application/pdf"
            )
            
    # Excel download
    if rep_cols[1].button("📊 Compile Formatted Excel Ledger"):
        with st.spinner("Generating styled sheet columns..."):
            reporter.export_excel(xlsx_path)
        with open(xlsx_path, "rb") as f:
            st.download_button(
                label="📥 Download Excel Ledger",
                data=f.read(),
                file_name="institutional_risk_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
    # CSV download
    if rep_cols[2].button("📝 Generate CSV Logs"):
        reporter.export_csv(csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            st.download_button(
                label="📥 Download CSV Log",
                data=f.read(),
                file_name="trade_pnl_log.csv",
                mime="text/csv"
            )
            
    # JSON download
    if rep_cols[3].button("⚙️ Export JSON Settings"):
        reporter.export_json(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            st.download_button(
                label="📥 Download JSON Settings",
                data=f.read(),
                file_name="risk_metrics.json",
                mime="application/json"
            )
