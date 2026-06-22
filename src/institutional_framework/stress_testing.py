"""Quantitative Monte Carlo Stress Testing and Robustness Engine.

Performs block bootstrap path simulations, latency injections, volatility shocks,
and calculates numerical estimates for Risk of Ruin and Maximum Drawdown.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class StressTestResult:
    risk_of_ruin_pct: float
    max_simulated_drawdown_pct: float
    median_ending_equity: float
    sharpe_ratio_ci: Tuple[float, float]
    passed_stress_test: bool

class OptionsStressTester:
    """Simulates extreme market conditions to evaluate strategy survivability."""
    
    def __init__(self, historical_trades: np.ndarray, initial_capital: float = 100000.0):
        self.historical_trades = historical_trades
        self.initial_capital = initial_capital
        
    def generate_bootstrap_paths(
        self,
        num_paths: int = 1000,
        path_length: int = 250,
        block_size: int = 5
    ) -> np.ndarray:
        """Generates synthetic trade paths using overlapping block bootstrap methods."""
        n_trades = len(self.historical_trades)
        if n_trades == 0:
            return np.zeros((num_paths, path_length))
            
        paths = np.zeros((num_paths, path_length))
        rng = np.random.default_rng()
        
        # Determine valid start indices for overlapping blocks
        max_start = n_trades - block_size
        if max_start <= 0:
            # Fallback to standard standard bootstrap if dataset is too small
            return rng.choice(self.historical_trades, size=(num_paths, path_length), replace=True)
            
        for i in range(num_paths):
            path_trades = []
            while len(path_trades) < path_length:
                start_idx = rng.integers(0, max_start + 1)
                block = self.historical_trades[start_idx : start_idx + block_size]
                path_trades.extend(block)
            paths[i, :] = path_trades[:path_length]
            
        return paths

    def run_stress_test(
        self,
        num_paths: int = 500,
        path_length: int = 150,
        volatility_shock_mult: float = 1.5,
        latency_slippage_ticks: float = 2.0,
        ruin_threshold_pct: float = 0.30     # Ruin defined as a 30% drawdown
    ) -> StressTestResult:
        """Simulates path metrics under stress parameters.
        
        Args:
            num_paths: Number of Monte Carlo path loops.
            path_length: Trade horizon length.
            volatility_shock_mult: Standard deviation multiplier for simulated returns.
            latency_slippage_ticks: Flat penalty tick slippage per trade to model latency spikes.
            ruin_threshold_pct: Capital drawdown ratio denoting total failure.
        """
        paths = self.generate_bootstrap_paths(num_paths, path_length)
        
        # Apply stress parameters
        # Slippage penalty: 1 tick = ₹0.05. Scaled to lot size (e.g. 75 lot size = ₹3.75 per tick per contract)
        slippage_penalty = latency_slippage_ticks * 0.05 * 75.0
        
        simulated_paths = (paths * volatility_shock_mult) - slippage_penalty
        
        ending_equities = []
        max_drawdowns = []
        ruin_count = 0
        sharpes = []
        
        for i in range(num_paths):
            trade_path = simulated_paths[i, :]
            equity_curve = self.initial_capital + np.cumsum(trade_path)
            equity_curve = np.insert(equity_curve, 0, self.initial_capital)
            
            ending_equities.append(equity_curve[-1])
            
            # Peak to trough drawdown
            peaks = np.maximum.accumulate(equity_curve)
            drawdowns = (peaks - equity_curve) / peaks
            max_dd = np.max(drawdowns)
            max_drawdowns.append(max_dd)
            
            # Ruin tracking
            if max_dd >= ruin_threshold_pct:
                ruin_count += 1
                
            # Sharpe calculation
            mean_ret = np.mean(trade_path)
            std_ret = np.std(trade_path)
            sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 1e-6 else 0.0
            sharpes.append(sharpe)
            
        risk_of_ruin = (ruin_count / num_paths) * 100.0
        max_sim_dd = float(np.max(max_drawdowns)) * 100.0
        median_ending = float(np.median(ending_equities))
        
        # Calculate 95% Confidence Interval for Sharpe Ratio
        sharpe_ci = (
            float(np.percentile(sharpes, 2.5)),
            float(np.percentile(sharpes, 97.5))
        )
        
        # Strict institutional acceptance boundary
        passed = (risk_of_ruin < 5.0) and (np.median(max_drawdowns) < 0.20)
        
        return StressTestResult(
            risk_of_ruin_pct=risk_of_ruin,
            max_simulated_drawdown_pct=max_sim_dd,
            median_ending_equity=median_ending,
            sharpe_ratio_ci=sharpe_ci,
            passed_stress_test=passed
        )
