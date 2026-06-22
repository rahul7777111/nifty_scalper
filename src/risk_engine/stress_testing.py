"""Parameter Stress Testing and Robustness Engine.

Varies execution parameters (slippage, latency, spread widening, execution failure) to generate
sensitivity grids. Implements randomized trade removal (Jackknifing), noise injection, adversarial
stress testing, and computes institutional robustness scores and overfitting probabilities.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


class StressTestingEngine:
    """Quantitative stress testing, sequence disruption, and overfitting analyzer."""

    def __init__(self, trades: Union[List[float], np.ndarray], start_capital: float = 100000.0):
        self.trades = np.asarray(trades, dtype=np.float64)
        self.start_capital = start_capital
        self.num_trades = len(self.trades)

    def run_parameter_sensitivity_grid(
        self,
        slippage_range: List[float],
        failure_prob_range: List[float],
        lot_size: float = 1.0
    ) -> pd.DataFrame:
        """Run simulation grid varying slippage and execution failure probability.

        Varying:
            Slippage: ₹ cost per trade (deducted directly from each trade return)
            Failure Probability: probability that a trade is execution-failed (return is 0.0 or a latency fee)

        Returns:
            A pandas DataFrame representing the sensitivity grid of median final returns.
        """
        grid_results = []

        for slip in slippage_range:
            for fail_p in failure_prob_range:
                # Simulates 1000 paths for this specific combination
                path_returns = []
                for _ in range(500):
                    # Copy and bootstrap
                    path = np.random.choice(self.trades, size=self.num_trades, replace=True)
                    
                    # Apply execution failures (returns are zeroed under failure)
                    failures = np.random.random(size=self.num_trades) < fail_p
                    adjusted = np.where(failures, -50.0, path)  # -50 as a latency fee or flat brokerage
                    
                    # Apply slippage
                    adjusted = (adjusted * lot_size) - slip
                    
                    final_ret = np.sum(adjusted)
                    path_returns.append(final_ret)
                
                median_return = float(np.median(path_returns))
                grid_results.append({
                    "Slippage": slip,
                    "Failure Probability": fail_p,
                    "Median Return": median_return
                })

        df = pd.DataFrame(grid_results)
        # Pivot into a 2D matrix format for heatmap plotting
        pivot_df = df.pivot(index="Slippage", columns="Failure Probability", values="Median Return")
        return pivot_df

    def jackknife_trade_removal(self, remove_pct: float = 0.10, iterations: int = 500) -> np.ndarray:
        """Jackknife sub-sampling: Simulates removing a random subset of trades.

        Tests whether the strategy's historical profit is heavily dependent on a few outlier trades.
        """
        num_remove = int(self.num_trades * remove_pct)
        if num_remove <= 0:
            num_remove = 1
            
        final_returns = []
        for _ in range(iterations):
            # Randomly choose indices to keep
            keep_indices = random.sample(range(self.num_trades), self.num_trades - num_remove)
            sub_trades = self.trades[keep_indices]
            final_returns.append(np.sum(sub_trades))
            
        return np.array(final_returns)

    def inject_noise(self, noise_vol_pct: float = 0.15, iterations: int = 500) -> np.ndarray:
        """Injects random Gaussian noise to trade outcomes to test strategy sensitivity.

        Simulates shifting market environments, spread widening, or execution latency anomalies.
        """
        # Noise magnitude scaled relative to historical standard deviation
        std = np.std(self.trades) if np.std(self.trades) > 0 else 1000.0
        noise_std = std * noise_vol_pct
        
        final_returns = []
        for _ in range(iterations):
            noise = np.random.normal(0, noise_std, size=self.num_trades)
            perturbed = self.trades + noise
            final_returns.append(np.sum(perturbed))
            
        return np.array(final_returns)

    def adversarial_stress_reorder(self) -> Tuple[np.ndarray, float]:
        """Adversarial sequencing: Rearranges trade history in the worst possible order.

        Combines all losses first to maximize drawdown and sequence-of-returns stress.
        """
        sorted_trades = np.sort(self.trades)
        equity = self.start_capital
        peak = self.start_capital
        max_dd = 0.0
        
        equity_curve = [equity]
        
        # Losses first, wins last (Adversarial Stress Path)
        for pnl in sorted_trades:
            equity += pnl
            equity_curve.append(equity)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
                
        return np.array(equity_curve), max_dd

    def calculate_robustness_overfitting_score(self) -> Dict[str, Union[float, str]]:
        """Calculates a comprehensive hedge-fund Robustness & Overfitting Score.

        Outputs:
            - robustness_score: (0-100) combining performance drop under noise and trade removal.
            - overfitting_probability: probability that performance was due to overfitting/outliers.
            - stability_index: ratio of standard deviation in noise tests vs. base return.
            - classification: 'High Robustness', 'Moderate Overfitting Risk', or 'Dangerous Overfitting'
        """
        base_profit = float(np.sum(self.trades))
        
        # 1. Performance under 10% trade removal (Jackknife)
        jack_returns = self.jackknife_trade_removal(remove_pct=0.10)
        median_jack = float(np.median(jack_returns))
        jack_loss_pct = (base_profit - median_jack) / abs(base_profit) if abs(base_profit) > 0 else 1.0

        # 2. Performance under noise injection
        noise_returns = self.inject_noise(noise_vol_pct=0.15)
        median_noise = float(np.median(noise_returns))
        noise_loss_pct = (base_profit - median_noise) / abs(base_profit) if abs(base_profit) > 0 else 1.0
        std_noise = np.std(noise_returns)

        # 3. Adversarial Drawdown
        _, adv_max_dd = self.adversarial_stress_reorder()

        # Calculate scores
        # Stability index: lower is better (less dispersion under noise)
        stability_index = float(std_noise / abs(base_profit)) if abs(base_profit) > 0 else 1.0
        
        # Overfitting Probability: Ratio of paths that ended in overall loss during perturbations
        overfit_prob = float(np.sum(jack_returns < 0) / len(jack_returns))
        
        # Robustness score calculation
        # Max 100, penalized by jackknife drop, noise sensitivity, and adversarial drawdown
        score = 100.0
        score -= min(30.0, max(0.0, jack_loss_pct * 100.0))
        score -= min(30.0, max(0.0, noise_loss_pct * 100.0))
        score -= min(40.0, adv_max_dd * 40.0)
        
        robustness_score = float(max(0.0, min(100.0, score)))

        if robustness_score >= 75.0:
            classification = "High Robustness (Institutional Grade)"
        elif robustness_score >= 50.0:
            classification = "Moderate Overfitting Risk (Trade with Caution)"
        else:
            classification = "Dangerous Overfitting (Overfit Strategy/Fragile)"

        return {
            "strategy_robustness_score": robustness_score,
            "overfitting_probability": overfit_prob,
            "stability_index": stability_index,
            "adversarial_max_drawdown": adv_max_dd,
            "classification": classification,
            "base_profit": base_profit,
            "jackknife_median_profit": median_jack,
            "noise_median_profit": median_noise
        }
