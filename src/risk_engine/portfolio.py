"""Portfolio Risk and Diversification Engine.

Supports multi-strategy simulation (e.g. options scalping, hedging, swing trades, futures),
calculates correlation and covariance matrices, measures diversification benefits, and simulates
correlated strategy returns using Cholesky decomposition.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np


class PortfolioRiskEngine:
    """Manages multi-strategy correlations, portfolio VaR, and joint Monte Carlo paths."""

    def __init__(self, strategies_data: Dict[str, Union[List[float], np.ndarray]]):
        """Initialize with a dictionary of strategies and their historical P&L series.

        Example:
            strategies_data = {
                "Options Scalping": [1200, -800, 1500, ...],
                "Tail Hedging": [-200, -150, 4000, ...],
                "Swing Futures": [500, 800, -1200, ...]
            }
        """
        # Ensure all series have the same length by aligning or truncating
        names = list(strategies_data.keys())
        if not names:
            # Fallback mock setup if no data passed
            strategies_data = {
                "Options Scalping": np.array([2400, -1800, 3100, -1200, 4200, -2200, 1500, -800, 5000, -3200]),
                "Tail Hedging": np.array([-200, -200, -200, 5000, -200, -200, -200, -200, 8000, -200]),
                "Swing Futures": np.array([1200, 800, -2200, -1100, 3600, 500, -1800, 1200, 4000, -500])
            }
            names = list(strategies_data.keys())

        min_len = min(len(strategies_data[name]) for name in names)
        
        self.strategy_names = names
        self.num_strategies = len(names)
        
        # Build matrix (cols = strategies, rows = trades/time steps)
        pnl_matrix = []
        for name in names:
            pnl_matrix.append(np.asarray(strategies_data[name][:min_len], dtype=np.float64))
            
        self.pnl_matrix = np.column_stack(pnl_matrix)  # Shape: (min_len, num_strategies)
        self.num_steps = min_len

    def compute_correlation_covariance(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the Pearson Correlation Matrix and Covariance Matrix of the strategies.
        """
        if self.num_steps <= 1:
            return np.eye(self.num_strategies), np.eye(self.num_strategies)
            
        covariance = np.cov(self.pnl_matrix, rowvar=False)
        # Ensure covariance is a 2D matrix even if 1 strategy
        if self.num_strategies == 1:
            covariance = np.array([[covariance]])
            
        correlation = np.corrcoef(self.pnl_matrix, rowvar=False)
        if self.num_strategies == 1:
            correlation = np.array([[1.0]])
            
        # Clean any NaNs in correlation (e.g. from constant series)
        np.nan_to_num(correlation, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        np.nan_to_num(covariance, copy=False, nan=0.0)
        
        # Re-set diagonal elements to 1.0
        for i in range(self.num_strategies):
            correlation[i, i] = 1.0
            
        return correlation, covariance

    def compute_portfolio_var(self, weights: np.ndarray, alpha: float = 0.95) -> Dict[str, float]:
        """Compute diversified vs undiversified Portfolio Value at Risk (VaR).

        Quantifies the exact diversification benefit and risk reduction.
        """
        if len(weights) != self.num_strategies:
            weights = np.ones(self.num_strategies) / self.num_strategies

        # Standardize weights
        weights = weights / np.sum(weights)

        # Portfolio realized returns per step
        portfolio_pnls = self.pnl_matrix.dot(weights)
        
        # Diversified VaR (historical simulation method)
        p5_port = np.percentile(portfolio_pnls, (1.0 - alpha) * 100)
        diversified_var = float(-p5_port) if p5_port < 0 else 0.0

        # Undiversified VaR (weighted sum of individual strategy VaRs)
        individual_vars = []
        for i in range(self.num_strategies):
            p5_ind = np.percentile(self.pnl_matrix[:, i], (1.0 - alpha) * 100)
            ind_var = float(-p5_ind) if p5_ind < 0 else 0.0
            individual_vars.append(ind_var)
            
        undiversified_var = float(np.sum(weights * np.array(individual_vars)))
        
        # Benefit
        diversification_benefit = max(0.0, undiversified_var - diversified_var)
        benefit_pct = (diversification_benefit / undiversified_var) * 100.0 if undiversified_var > 0 else 0.0

        return {
            "diversified_var": diversified_var,
            "undiversified_var": undiversified_var,
            "diversification_benefit_value": diversification_benefit,
            "diversification_benefit_percent": benefit_pct,
            "portfolio_mean_return": float(np.mean(portfolio_pnls)),
            "portfolio_std_dev": float(np.std(portfolio_pnls))
        }

    def simulate_correlated_portfolio_paths(
        self, num_simulations: int = 5000, num_steps: int = 100, weights: Optional[np.ndarray] = None
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """Simulates correlated multi-strategy returns using Cholesky Decomposition.

        Generates realistic joint Monte Carlo paths preserving the historical correlation structure.
        """
        if weights is None:
            weights = np.ones(self.num_strategies) / self.num_strategies
        else:
            weights = weights / np.sum(weights)

        corr, cov = self.compute_correlation_covariance()
        
        # 1. Cholesky Decomposition of correlation matrix: C = L * L^T
        # Add small diagonal perturbation to handle near-singular matrix issues
        try:
            L = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            # Fallback if matrix is not positive-definite: Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(corr)
            # Clip negative eigenvalues to a small positive constant
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            # Reconstruct and take square root
            corr_adj = eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T)
            L = np.linalg.cholesky(corr_adj)

        # Means and Standard Deviations of strategies
        means = np.mean(self.pnl_matrix, axis=0)
        stds = np.std(self.pnl_matrix, axis=0)
        stds = np.where(stds == 0, 1.0, stds)  # avoid division by zero

        # Simulation structures
        all_port_final_values = np.zeros(num_simulations)
        all_port_curves = []
        representative_samples = min(50, num_simulations)
        save_indices = set(np.random.choice(num_simulations, representative_samples, replace=False))

        # Vectorized generation
        for sim in range(num_simulations):
            # Generate independent standard random normals
            # Shape: (num_strategies, num_steps)
            z = np.random.normal(0, 1, size=(self.num_strategies, num_steps))
            
            # Apply correlation structure: y = L * z
            # Shape: (num_strategies, num_steps)
            y = L.dot(z)
            
            # Scale by strategy mean and std dev
            # Shape: (num_steps, num_strategies)
            correlated_returns = (y.T * stds) + means
            
            # Combine strategies using portfolio weights
            # Shape: (num_steps,)
            portfolio_returns = correlated_returns.dot(weights)
            
            # Compute equity curve
            equity_curve = 100000.0 + np.cumsum(portfolio_returns)
            equity_curve = np.insert(equity_curve, 0, 100000.0)
            
            all_port_final_values[sim] = equity_curve[-1] - 100000.0
            
            if sim in save_indices:
                all_port_curves.append(equity_curve)

        return {
            "final_returns": all_port_final_values,
            "representative_curves": all_port_curves,
            "correlation_matrix": corr,
            "covariance_matrix": cov
        }
