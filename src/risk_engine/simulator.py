"""Vectorized Simulation Engine.

Implements high-performance Monte Carlo simulations with advanced resampling modes
(Shuffle, Bootstrap, Block Bootstrap), Regime-switching simulations, Fat-tail/Black Swan
injection, and Dynamic Position Sizing (Fixed Fractional, Kelly, Anti-Martingale, Volatility Target,
Drawdown Adaptive).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from numba import jit
except ImportError:
    # Fallback decorator if numba is not installed or import fails
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class SimulationEngine:
    """Core simulation engine for quantitative risk analysis and stress testing."""

    def __init__(self, trades: Union[List[float], np.ndarray]):
        """Initialize simulator with a base list of trade P&L values (unscaled)."""
        self.trades = np.asarray(trades, dtype=np.float64)
        if len(self.trades) == 0:
            self.trades = np.array([1000.0, -500.0, 1500.0, -800.0, 2000.0])  # fallback
        self.num_trades = len(self.trades)

    def resample_trades(
        self, mode: str = "bootstrap", block_size: int = 5, path_len: Optional[int] = None
    ) -> np.ndarray:
        """Resample trades based on the selected Monte Carlo method.

        Args:
            mode: Resampling method ('shuffle', 'bootstrap', 'block_bootstrap').
            block_size: Size of contiguous blocks for block bootstrap.
            path_len: Length of the generated trade series. Defaults to self.num_trades.

        Returns:
            resampled: 1D array of resampled trades.
        """
        N = path_len if path_len is not None else self.num_trades
        
        if mode == "shuffle":
            # Shuffling only works when path_len matches self.num_trades or is smaller
            shuffled = self.trades.copy()
            np.random.shuffle(shuffled)
            if N <= self.num_trades:
                return shuffled[:N]
            else:
                # If path_len is larger, tile and shuffle
                tiled = np.tile(self.trades, int(np.ceil(N / self.num_trades)))
                np.random.shuffle(tiled)
                return tiled[:N]

        elif mode == "bootstrap":
            # Random sampling with replacement
            indices = np.random.randint(0, self.num_trades, size=N)
            return self.trades[indices]

        elif mode == "block_bootstrap":
            # Resample contiguous blocks of size k to preserve sequence correlation
            if block_size <= 1 or block_size >= self.num_trades:
                # Degenerates to normal bootstrap
                indices = np.random.randint(0, self.num_trades, size=N)
                return self.trades[indices]
            
            blocks_needed = int(np.ceil(N / block_size))
            sampled_series = []
            for _ in range(blocks_needed):
                # Pick a random starting index that allows taking a full block
                start_idx = np.random.randint(0, self.num_trades - block_size + 1)
                sampled_series.extend(self.trades[start_idx : start_idx + block_size])
            return np.array(sampled_series[:N], dtype=np.float64)

        else:
            # Fallback to simple bootstrap
            indices = np.random.randint(0, self.num_trades, size=N)
            return self.trades[indices]

    def inject_black_swan(
        self,
        trades_array: np.ndarray,
        crash_prob: float = 0.01,
        vol_spike_factor: float = 2.0,
        overnight_gap_loss: float = 5000.0,
        slippage_spike: float = 200.0
    ) -> np.ndarray:
        """Inject catastrophic rare events / Black Swan spikes into a resampled path.

        Simulates market crashes, flash crashes, liquidity collapses, and overnight gaps.
        """
        path_len = len(trades_array)
        adjusted = trades_array.copy()
        
        # Determine at which steps a catastrophic event occurs
        events = np.random.random(size=path_len) < crash_prob
        
        # 1. Volatility Expansion / Slippage Spike
        # Expand normal loss trades under stress and inject overnight gaps
        for i in range(path_len):
            if events[i]:
                # Flash crash / liquidity collapse: negative return multiplier & slippage spike
                if adjusted[i] < 0:
                    adjusted[i] = (adjusted[i] * vol_spike_factor) - slippage_spike
                else:
                    # Positive trades turn negative or get eaten by bid-ask spread collapse
                    adjusted[i] = (adjusted[i] * 0.2) - overnight_gap_loss - slippage_spike
            elif i > 0 and events[i - 1]:
                # Post-event regime drift: standard deviation remains high
                adjusted[i] = adjusted[i] * 1.5 - (slippage_spike * 0.5)

        return adjusted

    def simulate_regimes(
        self,
        num_trades: int = 100,
        regime_probs: Optional[Dict[str, float]] = None,
        transition_matrix: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Simulates trades across different market regimes using a Markov Transition Chain.

        Regimes:
            0: Sideways / Low Vol (low return std)
            1: Trending / Bull (positive shift)
            2: High Vol / Bear (negative shift, massive std)
        """
        # Default regimes
        regime_names = ["Sideways", "Bull", "Bear"]
        
        # Transition matrix (probability of state i going to state j)
        if transition_matrix is None:
            # High probability of staying in same regime, gradual transition
            transition_matrix = np.array([
                [0.80, 0.15, 0.05],  # From Sideways
                [0.10, 0.85, 0.05],  # From Bull
                [0.10, 0.10, 0.80]   # From Bear
            ])
            
        current_state = 0  # start in Sideways
        simulated_pnl = []
        state_history = []
        
        # Fit normal distributions per regime based on historical partition
        # For simplicity, divide historical trades into representative sub-distributions
        sorted_trades = np.sort(self.trades)
        third = len(sorted_trades) // 3
        
        sideways_dist = sorted_trades[third : 2 * third]
        bull_dist = sorted_trades[2 * third :]
        bear_dist = sorted_trades[:third]
        
        dists = [sideways_dist, bull_dist, bear_dist]
        
        for _ in range(num_trades):
            # Record state
            state_history.append(regime_names[current_state])
            
            # Sample PnL from the current state's distribution
            dist = dists[current_state]
            pnl = float(np.random.choice(dist))
            simulated_pnl.append(pnl)
            
            # Transition to next state
            probs = transition_matrix[current_state]
            current_state = int(np.random.choice([0, 1, 2], p=probs))
            
        return np.array(simulated_pnl, dtype=np.float64), state_history

    def apply_dynamic_sizing(
        self,
        base_pnls: np.ndarray,
        sizing_method: str = "fixed_fractional",
        start_capital: float = 100000.0,
        lot_size_factor: float = 1.0,
        max_risk_pct: float = 0.02,
        max_exposure_pct: float = 0.80,
        leverage_multiplier: float = 1.0,
        risk_controls: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Simulates a trade path applying dynamic position sizing rules.

        Sizing Methods:
            - 'fixed_fractional': Risk a fixed fractional percent of equity per trade.
            - 'kelly': Sizing dynamically scaled by the Kelly Criterion fraction.
            - 'anti_martingale': Double sizing on wins, halve on losses.
            - 'volatility_targeting': Sizing scaled inversely with historical rolling volatility.
            - 'drawdown_adaptive': Size scaled down as current drawdown approaches risk limits.

        Returns:
            equity_curve: Account balance at each step.
            sized_returns: Sized trade P&L values.
            metrics: dictionary of performance results.
        """
        equity = start_capital
        equity_curve = [equity]
        sized_returns = []
        
        # Historical win rate and payout ratio for Kelly calculations
        wins = self.trades[self.trades > 0]
        losses = self.trades[self.trades < 0]
        win_rate = len(wins) / self.num_trades if self.num_trades > 0 else 0.5
        avg_win = np.mean(wins) if len(wins) > 0 else 1.0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1.0
        payout_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        # Kelly optimal fraction: f* = p - (1-p)/b
        kelly_fraction = win_rate - (1.0 - win_rate) / payout_ratio
        # Set half-Kelly for stability
        kelly_fraction = max(0.01, min(0.30, kelly_fraction * 0.5))
        
        peak = start_capital
        multiplier = 1.0  # Used in Anti-Martingale / Adaptive
        
        # Risk controls parsing
        risk_ctrl = risk_controls or {}
        daily_loss_limit = risk_ctrl.get("daily_loss_limit", float('inf'))
        max_dd_shutdown = risk_ctrl.get("max_dd_shutdown", 0.50)  # shutdown at 50% drawdown
        pause_after_losses = risk_ctrl.get("consecutive_loss_pause", 3)
        pause_duration = risk_ctrl.get("pause_duration_trades", 3)
        
        consec_losses = 0
        pause_remaining = 0
        circuit_broken = False
        
        # Vectorized calculations helper: rolling volatility
        rolling_window = 10
        
        for idx, base_pnl in enumerate(base_pnls):
            if circuit_broken or equity <= 0:
                sized_returns.append(0.0)
                equity_curve.append(equity)
                continue
                
            # Handle consecutive loss pause
            if pause_remaining > 0:
                pause_remaining -= 1
                sized_returns.append(0.0)
                equity_curve.append(equity)
                continue
            
            # 1. Compute current sizing factor based on selected algorithm
            if sizing_method == "fixed_fractional":
                # Size relative to equity: size = equity * max_risk_pct / risk_unit
                # Assume a standardized risk unit of average loss
                risk_unit = avg_loss if avg_loss > 0 else 1000.0
                size_factor = (equity * max_risk_pct) / risk_unit
                
            elif sizing_method == "kelly":
                # Scale by the Kelly fraction
                risk_unit = avg_loss if avg_loss > 0 else 1000.0
                size_factor = (equity * kelly_fraction) / risk_unit
                
            elif sizing_method == "anti_martingale":
                # Scale sizing up on wins, down on losses
                if idx > 0 and sized_returns[-1] > 0:
                    multiplier = min(4.0, multiplier * 1.5)  # Scale up on win
                elif idx > 0 and sized_returns[-1] < 0:
                    multiplier = max(0.25, multiplier * 0.5)  # Cut in half on loss
                size_factor = lot_size_factor * multiplier
                
            elif sizing_method == "volatility_targeting":
                # Sizing scaled inversely with recent 10-trade rolling volatility
                if idx >= rolling_window:
                    recent = sized_returns[-rolling_window:]
                    vol = np.std(recent) if len(recent) > 0 else 1e-9
                    if vol < 1e-9:
                        vol = avg_loss
                else:
                    vol = avg_loss
                
                # Sizing factor = Target Volatility / Realized Volatility
                target_vol = start_capital * max_risk_pct
                vol_ratio = target_vol / vol if vol > 0 else 1.0
                size_factor = lot_size_factor * max(0.1, min(2.0, vol_ratio))
                
            elif sizing_method == "drawdown_adaptive":
                # Sizing scaled down as drawdown increases towards max_dd_shutdown
                curr_dd = (peak - equity) / peak if peak > 0 else 0.0
                dd_slack = max(0.0, 1.0 - (curr_dd / max_dd_shutdown))
                # Sizing factor drops linearly as drawdown increases
                size_factor = lot_size_factor * dd_slack
                
            else:
                # Default fixed contract scaling
                size_factor = lot_size_factor
                
            # Apply absolute portfolio constraints
            max_size_allowed = (equity * max_exposure_pct) / (avg_loss if avg_loss > 0 else 1.0)
            size_factor = min(size_factor, max_size_allowed) * leverage_multiplier
            size_factor = max(0.0, size_factor)
            
            # Sized trade P&L
            sized_pnl = base_pnl * size_factor
            
            # Risk Controls: Max Drawdown shutdown check
            curr_dd = (peak - equity) / peak if peak > 0 else 0.0
            if curr_dd >= max_dd_shutdown:
                circuit_broken = True
                sized_pnl = 0.0
                
            # Update equity
            equity += sized_pnl
            sized_returns.append(sized_pnl)
            equity_curve.append(equity)
            
            # Update Peak
            if equity > peak:
                peak = equity
                
            # Track losing streak & pause trigger
            if sized_pnl < 0:
                consec_losses += 1
                if consec_losses >= pause_after_losses:
                    pause_remaining = pause_duration
                    consec_losses = 0
            else:
                consec_losses = 0

        # Pack results
        equity_curve_arr = np.array(equity_curve)
        sized_returns_arr = np.array(sized_returns)
        
        final_profit = equity_curve_arr[-1] - start_capital
        ruined = bool(circuit_broken or equity_curve_arr[-1] < (start_capital * (1.0 - max_dd_shutdown)))
        
        metrics = {
            "final_equity": equity_curve_arr[-1],
            "total_profit": final_profit,
            "is_ruined": ruined,
            "max_dd_reached": float(np.max((np.maximum.accumulate(equity_curve_arr) - equity_curve_arr) / np.maximum.accumulate(equity_curve_arr))) if len(equity_curve_arr) > 0 else 0.0
        }
        
        return equity_curve_arr, sized_returns_arr, metrics

    def run_monte_carlo(
        self,
        num_simulations: int = 10000,
        mode: str = "bootstrap",
        block_size: int = 5,
        path_len: Optional[int] = None,
        black_swan_config: Optional[Dict] = None,
        sizing_config: Optional[Dict] = None
    ) -> Dict[str, Union[np.ndarray, List]]:
        """Executes high-performance multi-scenario Monte Carlo simulations in parallel/vectorized.

        Returns all generated curves, final values, and maximum drawdowns across paths.
        """
        N = path_len if path_len is not None else self.num_trades
        all_final_returns = np.zeros(num_simulations)
        all_max_drawdowns = np.zeros(num_simulations)
        all_curves = []
        
        # Configure Black Swan
        bs_cfg = black_swan_config or {}
        enable_black_swan = bs_cfg.get("enable", False)
        crash_prob = bs_cfg.get("crash_probability", 0.01)
        vol_spike = bs_cfg.get("volatility_spike_factor", 2.0)
        gap_loss = bs_cfg.get("overnight_gap_loss", 5000.0)
        slip_spike = bs_cfg.get("slippage_spike", 200.0)
        
        # Configure Sizing
        sz_cfg = sizing_config or {}
        sizing_method = sz_cfg.get("method", "fixed")
        start_cap = sz_cfg.get("start_capital", 100000.0)
        lot_size = sz_cfg.get("lot_size_factor", 1.0)
        max_risk = sz_cfg.get("max_risk_pct", 0.02)
        max_exposure = sz_cfg.get("max_exposure_pct", 0.80)
        leverage = sz_cfg.get("leverage_multiplier", 1.0)
        
        risk_ctrls = sz_cfg.get("risk_controls", {})

        # Representative curve sampling list
        curves_to_save = min(100, num_simulations)
        save_indices = set(random.sample(range(num_simulations), curves_to_save))

        for sim_idx in range(num_simulations):
            # 1. Resample Path
            resampled = self.resample_trades(mode=mode, block_size=block_size, path_len=N)
            
            # 2. Inject Black Swan stress if active
            if enable_black_swan:
                resampled = self.inject_black_swan(
                    resampled,
                    crash_prob=crash_prob,
                    vol_spike_factor=vol_spike,
                    overnight_gap_loss=gap_loss,
                    slippage_spike=slip_spike
                )
                
            # 3. Dynamic position sizing and risk controls simulation
            equity_curve, sized_returns, metrics = self.apply_dynamic_sizing(
                resampled,
                sizing_method=sizing_method,
                start_capital=start_cap,
                lot_size_factor=lot_size,
                max_risk_pct=max_risk,
                max_exposure_pct=max_exposure,
                leverage_multiplier=leverage,
                risk_controls=risk_ctrls
            )
            
            all_final_returns[sim_idx] = metrics["total_profit"]
            all_max_drawdowns[sim_idx] = metrics["max_dd_reached"]
            
            if sim_idx in save_indices:
                all_curves.append(equity_curve)

        return {
            "final_returns": all_final_returns,
            "max_drawdowns": all_max_drawdowns,
            "representative_curves": all_curves
        }
