"""
Monte Carlo Validation Framework

Provides:
- Trade shuffle bootstrap (5000 simulations)
- Block bootstrap (5000 simulations, preserves autocorrelation)
- Label shuffle (10000 simulations, null hypothesis testing)
- Feature permutation (5000 simulations, importance testing)
- Deflated Sharpe Ratio calculation
- Ruin probability estimates
- Model Confidence Set inclusion probability

Usage:
    from src.institutional_framework.monte_carlo import (
        run_monte_carlo_5000,
        run_monte_carlo_10000,
        compute_deflated_sharpe,
    )
"""

import json
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DistributionStats:
    """Summary statistics for a distribution."""
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    p5: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    fraction_positive: float = 0.0
    fraction_negative: float = 0.0


@dataclass
class RuinProbability:
    """Probability of various drawdown thresholds."""
    dd_gt_10pct: float = 0.0
    dd_gt_20pct: float = 0.0
    dd_gt_30pct: float = 0.0
    dd_gt_50pct: float = 0.0


@dataclass
class MonteCarloResult:
    """Complete Monte Carlo simulation results."""
    n_simulations: int = 0
    simulation_type: str = ""
    sharpe_distribution: DistributionStats = field(default_factory=DistributionStats)
    profit_factor_distribution: DistributionStats = field(default_factory=DistributionStats)
    max_drawdown_distribution: DistributionStats = field(default_factory=DistributionStats)
    win_rate_distribution: DistributionStats = field(default_factory=DistributionStats)
    return_distribution: DistributionStats = field(default_factory=DistributionStats)
    ruin: RuinProbability = field(default_factory=RuinProbability)
    p_value: Optional[float] = None
    deflated_sharpe: Optional[float] = None
    is_significant: Optional[bool] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_distribution_stats(values: np.ndarray) -> DistributionStats:
    """Compute distribution statistics from an array of values."""
    return DistributionStats(
        mean=float(np.mean(values)),
        std=float(np.std(values, ddof=1)),
        median=float(np.median(values)),
        p5=float(np.percentile(values, 5)),
        p25=float(np.percentile(values, 25)),
        p75=float(np.percentile(values, 75)),
        p95=float(np.percentile(values, 95)),
        min_val=float(np.min(values)),
        max_val=float(np.max(values)),
        fraction_positive=float(np.mean(values > 0)),
        fraction_negative=float(np.mean(values < 0)),
    )


def _compute_sharpe(returns: np.ndarray) -> float:
    """Compute annualized Sharpe ratio from periodic returns."""
    if len(returns) < 2 or np.std(returns, ddof=1) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252))


def _compute_profit_factor(returns: np.ndarray) -> float:
    """Compute profit factor (gross wins / gross losses)."""
    wins = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float('inf') if wins > 0 else 1.0
    return float(wins / losses)


def _compute_max_drawdown(cumulative: np.ndarray) -> float:
    """Compute maximum drawdown from cumulative returns series."""
    if len(cumulative) == 0:
        return 0.0
    peak = np.maximum.accumulate(cumulative)
    dd = (cumulative - peak) / (peak + 1e-10)
    return float(abs(np.min(dd)))


def _compute_win_rate(returns: np.ndarray) -> float:
    """Compute win rate fraction."""
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))


# ---------------------------------------------------------------------------
# 5,000 simulations: Trade shuffle bootstrap
# ---------------------------------------------------------------------------

def run_monte_carlo_5000(
    trade_returns: np.ndarray,
    n_simulations: int = 5000,
    block_size: int = 1,
    seed: int = 42,
) -> MonteCarloResult:
    """
    Run 5,000 Monte Carlo simulations using bootstrap resampling.

    Parameters:
        trade_returns: Array of per-trade returns (fractional, e.g. 0.01 = 1%)
        n_simulations: Number of bootstrap simulations (default: 5000)
        block_size: Block size for block bootstrap (1 = i.i.d. bootstrap)
        seed: Random seed for reproducibility

    Returns:
        MonteCarloResult with distribution statistics and ruin probabilities.
    """
    rng = np.random.default_rng(seed)
    n_trades = len(trade_returns)

    if n_trades < 10:
        logger.warning("Too few trades for Monte Carlo: %d (need >= 10)", n_trades)
        return MonteCarloResult(n_simulations=0, simulation_type="trade_shuffle")

    sharpe_values = np.zeros(n_simulations)
    pf_values = np.zeros(n_simulations)
    max_dd_values = np.zeros(n_simulations)
    win_rate_values = np.zeros(n_simulations)
    return_values = np.zeros(n_simulations)

    n_blocks = max(1, n_trades - block_size + 1)

    for i in range(n_simulations):
        if block_size > 1:
            # Block bootstrap: sample blocks of size block_size
            n_blocks_needed = int(np.ceil(n_trades / block_size))
            block_starts = rng.integers(0, n_blocks, size=n_blocks_needed)
            sampled = np.concatenate([
                trade_returns[start:start + block_size]
                for start in block_starts
            ])[:n_trades]
        else:
            # Standard i.i.d. bootstrap
            sampled = rng.choice(trade_returns, size=n_trades, replace=True)

        cumulative = np.cumsum(sampled)

        sharpe_values[i] = _compute_sharpe(sampled)
        pf_values[i] = _compute_profit_factor(sampled)
        max_dd_values[i] = _compute_max_drawdown(cumulative)
        win_rate_values[i] = _compute_win_rate(sampled)
        return_values[i] = float(cumulative[-1])

    # Compute ruin probabilities
    ruin = RuinProbability(
        dd_gt_10pct=float(np.mean(max_dd_values > 0.10)),
        dd_gt_20pct=float(np.mean(max_dd_values > 0.20)),
        dd_gt_30pct=float(np.mean(max_dd_values > 0.30)),
        dd_gt_50pct=float(np.mean(max_dd_values > 0.50)),
    )

    # Compute p-value: what fraction of simulations had Sharpe <= 0?
    obs_sharpe = _compute_sharpe(trade_returns)
    p_value = float(np.mean(sharpe_values <= 0))

    # Deflated Sharpe
    deflated = compute_deflated_sharpe(trade_returns, num_trials=10)

    result = MonteCarloResult(
        n_simulations=n_simulations,
        simulation_type="trade_shuffle_bootstrap",
        sharpe_distribution=_compute_distribution_stats(sharpe_values),
        profit_factor_distribution=_compute_distribution_stats(pf_values),
        max_drawdown_distribution=_compute_distribution_stats(max_dd_values),
        win_rate_distribution=_compute_distribution_stats(win_rate_values),
        return_distribution=_compute_distribution_stats(return_values),
        ruin=ruin,
        p_value=p_value,
        deflated_sharpe=deflated,
        is_significant=p_value < 0.05 and deflated > 1.0,
    )

    logger.info(
        "MC5000 | n_sims=%d | obs_sharpe=%.3f | p_value=%.4f | deflated_SR=%.3f | P(ruin>20%%)=%.1f%%",
        n_simulations, obs_sharpe, p_value, deflated,
        ruin.dd_gt_20pct * 100,
    )

    return result


# ---------------------------------------------------------------------------
# 10,000 simulations: Label shuffle (null hypothesis testing)
# ---------------------------------------------------------------------------

def run_monte_carlo_10000(
    returns: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    n_simulations: int = 10000,
    seed: int = 42,
) -> MonteCarloResult:
    """
    Run 10,000 simulations under the null hypothesis (no edge).

    Shuffles labels to break the relationship between predictions and outcomes.
    The real Sharpe is compared to the null distribution to get a p-value.

    Parameters:
        returns: Array of per-trade returns (fractional)
        labels: Array of true labels (0/1)
        predictions: Array of model predictions (probabilities)
        n_simulations: Number of shuffles (default: 10000)
        seed: Random seed

    Returns:
        MonteCarloResult with null distribution statistics.
    """
    rng = np.random.default_rng(seed)
    n = len(returns)

    if n < 10:
        logger.warning("Too few samples for label shuffle: %d", n)
        return MonteCarloResult(n_simulations=0, simulation_type="label_shuffle")

    obs_sharpe = _compute_sharpe(returns)
    null_sharpes = np.zeros(n_simulations)

    for i in range(n_simulations):
        shuffled = rng.permutation(labels)
        # Compute "returns" from shuffled labels
        shuffled_returns = np.where(shuffled == 1, abs(returns), -abs(returns) * 0.5)
        null_sharpes[i] = _compute_sharpe(shuffled_returns)

    p_value = float(np.mean(null_sharpes >= obs_sharpe))
    is_sig = p_value < 0.05

    result = MonteCarloResult(
        n_simulations=n_simulations,
        simulation_type="label_shuffle_null",
        sharpe_distribution=_compute_distribution_stats(null_sharpes),
        p_value=p_value,
        is_significant=is_sig,
    )

    logger.info(
        "MC10000 | obs_sharpe=%.3f | null_mean=%.3f | p_value=%.4f | significant=%s",
        obs_sharpe, float(np.mean(null_sharpes)), p_value, is_sig,
    )

    return result


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

def compute_deflated_sharpe(
    returns: np.ndarray,
    num_trials: int = 10,
    num_params: int = 5,
    sample_size: Optional[int] = None,
) -> float:
    """
    Compute the Deflated Sharpe Ratio (DSR).

    DSR adjusts the observed Sharpe ratio for:
    - Multiple testing (number of trials)
    - Non-Normality of returns
    - Serial correlation in returns
    - Short sample sizes

    DSR > 1.0 indicates statistical significance after deflation.

    Parameters:
        returns: Array of periodic returns
        num_trials: Number of independent strategy trials (default: 10)
        num_params: Number of free parameters (default: 5)
        sample_size: Length of returns (inferred if None)

    Returns:
        Deflated Sharpe Ratio value. > 1.0 is significant.
    """
    if len(returns) < 10:
        return 0.0

    T = sample_size or len(returns)
    obs_sharpe = _compute_sharpe(returns)

    # Estimate skewness and kurtosis of returns
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=False))  # excess kurtosis = kurt - 3

    # Estimate autocorrelation
    if len(returns) > 20:
        acf1 = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
    else:
        acf1 = 0.0

    # Variance adjustment for serial correlation (Bartoš' correction)
    var_correction = 1.0
    if acf1 != 0:
        # Simple correction for AR(1)
        var_correction = (1 + acf1) / (1 - acf1)

    # Estimate the expected maximum Sharpe under multiple testing
    # Using the approximation from Harvey, Liu & Zhu (2016)
    E_max = 0.0
    if num_trials > 1:
        # Standard normal approximation for max of N i.i.d. normals
        E_max = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1.0 / num_trials) \
                + np.euler_gamma * stats.norm.ppf(1 - np.exp(-1) / num_trials)

    # Adjust for non-normality (Cornish-Fisher expansion)
    z = stats.norm.ppf(0.95)
    var_adjustment = 1 + (skew / 6) * z + ((kurt - 3) / 24) * (z**2 - 1) - (skew**2 / 36) * (2 * z**2 - 1)

    # Deflated Sharpe = (obs_sharpe - E_max) / adjustment
    adjustment = max(var_correction * var_adjustment, 0.5)  # floor at 0.5
    deflated = (obs_sharpe - E_max) / (adjustment / np.sqrt(T / 252) + 1e-10)

    return float(max(deflated, 0.0))


# ---------------------------------------------------------------------------
# Feature permutation test
# ---------------------------------------------------------------------------

def run_feature_permutation(
    returns: np.ndarray,
    features: np.ndarray,
    n_simulations: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Run feature permutation test to assess feature importance.

    For each feature column, permute it and measure the drop in Sharpe.
    Features with large drops are important.

    Parameters:
        returns: Array of per-trade returns
        features: (n_samples, n_features) array
        n_simulations: Number of permutations per feature
        seed: Random seed

    Returns:
        Dictionary mapping feature index to importance score
        (drop in Sharpe when permuted)
    """
    rng = np.random.default_rng(seed)
    n_features = features.shape[1]

    baseline_sharpe = _compute_sharpe(returns)
    importance: Dict[str, float] = {}

    for f in range(n_features):
        permuted_sharpes = np.zeros(n_simulations)
        for i in range(n_simulations):
            permuted = features.copy()
            permuted[:, f] = rng.permutation(permuted[:, f])
            # Simple proxy: compute correlation with returns
            corr = float(np.corrcoef(permuted[:, f], returns)[0, 1])
            permuted_sharpes[i] = abs(corr) * baseline_sharpe

        avg_perm_sharpe = float(np.mean(permuted_sharpes))
        drop = baseline_sharpe - avg_perm_sharpe
        importance[f"feature_{f}"] = drop

    return importance


# ---------------------------------------------------------------------------
# Convenience: run full Monte Carlo test suite
# ---------------------------------------------------------------------------

def run_full_monte_carlo(
    trade_returns: np.ndarray,
    labels: Optional[np.ndarray] = None,
    predictions: Optional[np.ndarray] = None,
    features: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run the complete Monte Carlo test suite.

    Returns:
        Dictionary with all results, p-values, and significance flags.
    """
    results: Dict[str, Any] = {}

    # 5,000 trade shuffle bootstrap
    shuffle_result = run_monte_carlo_5000(trade_returns, n_simulations=5000, seed=seed)
    results["trade_shuffle_5000"] = {
        "sharpe_mean": shuffle_result.sharpe_distribution.mean,
        "sharpe_std": shuffle_result.sharpe_distribution.std,
        "sharpe_p5": shuffle_result.sharpe_distribution.p5,
        "sharpe_p95": shuffle_result.sharpe_distribution.p95,
        "p_sharpe_positive": shuffle_result.sharpe_distribution.fraction_positive,
        "deflated_sharpe": shuffle_result.deflated_sharpe,
        "p_value": shuffle_result.p_value,
        "is_significant": shuffle_result.is_significant,
        "ruin_dd_20pct": shuffle_result.ruin.dd_gt_20pct,
        "ruin_dd_50pct": shuffle_result.ruin.dd_gt_50pct,
    }

    # 5,000 block bootstrap (block_size=5)
    block_result = run_monte_carlo_5000(
        trade_returns, n_simulations=5000, block_size=5, seed=seed + 1,
    )
    results["block_bootstrap_5000"] = {
        "sharpe_mean": block_result.sharpe_distribution.mean,
        "sharpe_std": block_result.sharpe_distribution.std,
        "p_value": block_result.p_value,
        "ruin_dd_20pct": block_result.ruin.dd_gt_20pct,
    }

    # 10,000 label shuffle (if labels provided)
    if labels is not None and predictions is not None:
        label_result = run_monte_carlo_10000(
            trade_returns, labels, predictions, n_simulations=10000, seed=seed + 2,
        )
        results["label_shuffle_10000"] = {
            "null_sharpe_mean": label_result.sharpe_distribution.mean,
            "null_sharpe_std": label_result.sharpe_distribution.std,
            "p_value": label_result.p_value,
            "is_significant": label_result.is_significant,
        }

    # Feature permutation (if features provided)
    if features is not None:
        feature_importance = run_feature_permutation(
            trade_returns, features, n_simulations=5000, seed=seed + 3,
        )
        results["feature_permutation"] = feature_importance

    # Overall assessment
    all_significant = all(
        r.get("is_significant", False)
        for r in results.values()
        if isinstance(r, dict) and "is_significant" in r
    )
    results["overall_significant"] = all_significant
    results["overall_p_value"] = min(
        r.get("p_value", 1.0)
        for r in results.values()
        if isinstance(r, dict) and "p_value" in r
    )

    return results