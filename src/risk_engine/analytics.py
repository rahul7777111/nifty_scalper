"""Advanced financial analytics library.

Implements professional risk ratios, downside risk metrics, drawdown analyses,
statistical distribution properties, and options-specific risk parameters.
"""

from __future__ import annotations

import math
from typing import Dict, List, Union

import numpy as np
import scipy.stats as stats


def compute_drawdowns(equity_curve: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Calculate the rolling drawdown series, max drawdown, and peak/trough indices.

    Args:
        equity_curve: Array of account equity at each trade/step.

    Returns:
        drawdowns: Array of drawdowns (as fractional percentages).
        max_dd: Maximum peak-to-trough drawdown (as a float).
        peak_idx: Index of peak equity.
        trough_idx: Index of trough equity.
    """
    if len(equity_curve) == 0:
        return np.array([]), 0.0, 0, 0

    peaks = np.maximum.accumulate(equity_curve)
    # Prevent division by zero if equity is zero or negative
    safe_peaks = np.where(peaks <= 0, 1.0, peaks)
    drawdowns = (peaks - equity_curve) / safe_peaks

    max_dd_idx = np.argmax(drawdowns)
    max_dd = float(drawdowns[max_dd_idx])

    # Find the peak corresponding to this drawdown
    peak_idx = int(np.argmax(equity_curve[:max_dd_idx + 1]))
    trough_idx = int(max_dd_idx)

    return drawdowns, max_dd, peak_idx, trough_idx


def compute_ulcer_index(equity_curve: np.ndarray) -> float:
    """Compute the Ulcer Index (UI), a measure of depth and duration of drawdowns.

    UI = sqrt(mean(drawdowns^2))
    """
    if len(equity_curve) <= 1:
        return 0.0
    drawdowns, _, _, _ = compute_drawdowns(equity_curve)
    if len(drawdowns) == 0:
        return 0.0
    # Quadradic average of drawdown percentages
    return float(np.sqrt(np.mean(drawdowns ** 2)) * 100)


def compute_cvar(returns: np.ndarray, alpha: float = 0.95) -> float:
    """Compute Conditional Value at Risk (CVaR) or Expected Shortfall.

    Represents the expected loss in the worst (1 - alpha) percentage of cases.
    For P&L series, we define it positive as the average of the worst losses.
    """
    if len(returns) == 0:
        return 0.0
    sorted_returns = np.sort(returns)
    k = max(1, int(len(sorted_returns) * (1.0 - alpha)))
    worst_returns = sorted_returns[:k]
    # Represent as a positive value (expected loss)
    return float(-np.mean(worst_returns))


def compute_tail_ratio(returns: np.ndarray) -> float:
    """Calculate the Tail Ratio.

    Ratio of the 95th percentile return to the absolute value of the 5th percentile return.
    Shows the asymmetry between extreme positive and extreme negative returns.
    """
    if len(returns) == 0:
        return 1.0
    p95 = np.percentile(returns, 95)
    p5 = np.percentile(returns, 5)
    if abs(p5) < 1e-9:
        return 1.0
    return float(p95 / abs(p5))


def compute_sharpe_ratio(returns: np.ndarray, rf_rate: float = 0.0) -> float:
    """Compute the annualized Sharpe Ratio.

    Sharpe = (mean(returns) - rf) / std(returns)
    """
    if len(returns) <= 1:
        return 0.0
    std_dev = np.std(returns, ddof=1)
    if std_dev < 1e-9:
        return 0.0
    return float((np.mean(returns) - rf_rate) / std_dev * np.sqrt(252))


def compute_sortino_ratio(returns: np.ndarray, rf_rate: float = 0.0) -> float:
    """Compute the annualized Sortino Ratio.

    Sortino = (mean(returns) - rf) / std(negative_returns)
    """
    if len(returns) <= 1:
        return 0.0
    excess_returns = returns - rf_rate
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return 0.0
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    if downside_std < 1e-9:
        return 0.0
    return float(np.mean(excess_returns) / downside_std * np.sqrt(252))


def compute_calmar_ratio(annual_return: float, max_dd: float) -> float:
    """Compute the Calmar Ratio.

    Calmar = Annualized Return / Max Drawdown
    """
    if max_dd < 1e-9:
        return 0.0
    return float(annual_return / max_dd)


def compute_recovery_factor(total_profit: float, max_dd_value: float) -> float:
    """Compute the Recovery Factor.

    Recovery Factor = Total Profit / Max Drawdown Value (absolute money)
    """
    if abs(max_dd_value) < 1e-9:
        return 0.0
    return float(total_profit / abs(max_dd_value))


def compute_profit_factor(returns: np.ndarray) -> float:
    """Compute the Profit Factor.

    Profit Factor = Sum of Gross Profits / Sum of Gross Losses
    """
    profits = returns[returns > 0]
    losses = returns[returns < 0]
    if len(losses) == 0:
        return float('inf') if len(profits) > 0 else 1.0
    return float(np.sum(profits) / abs(np.sum(losses)))


def compute_expectancy(returns: np.ndarray) -> float:
    """Compute the expectancy per trade.

    Expectancy = (Win% * AvgWin) + (Loss% * AvgLoss)
    """
    if len(returns) == 0:
        return 0.0
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = len(wins) / len(returns)
    loss_rate = len(losses) / len(returns)

    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0

    return float((win_rate * avg_win) + (loss_rate * avg_loss))


def detect_fat_tails(returns: np.ndarray) -> dict[str, Union[float, bool]]:
    """Determine if a distribution has fat tails.

    Uses Kurtosis (Kurtosis > 3 represents fat tails / leptokurtic distribution)
    and fits a Power-Law alpha using scipy.stats.
    """
    if len(returns) <= 3:
        return {"excess_kurtosis": 0.0, "is_fat_tailed": False, "alpha": 0.0}

    kurt = float(stats.kurtosis(returns, fisher=True))
    is_fat = bool(kurt > 1.0)  # Moderate to heavy fat tails

    # Simple Power-Law alpha estimation (Pareto tail index)
    try:
        abs_losses = np.abs(returns[returns < 0])
        if len(abs_losses) > 10:
            threshold = np.percentile(abs_losses, 80)
            tail_losses = abs_losses[abs_losses > threshold]
            if len(tail_losses) > 0 and threshold > 0:
                alpha = float(len(tail_losses) / np.sum(np.log(tail_losses / threshold)))
            else:
                alpha = 0.0
        else:
            alpha = 0.0
    except Exception:
        alpha = 0.0

    return {
        "excess_kurtosis": kurt,
        "is_fat_tailed": is_fat,
        "power_law_alpha": alpha
    }


def compute_risk_of_ruin_metrics(
    returns: np.ndarray, start_capital: float, ruin_level: float = 0.50
) -> dict[str, float]:
    """Calculate expanded risk of ruin, expected recovery time, and capital adequacy ratio.

    Args:
        returns: Array of historical trade P&Ls.
        start_capital: Initial capital size.
        ruin_level: Fractional loss defining ruin (default 50% drawdown).
    """
    if len(returns) == 0:
        return {
            "ruin_probability": 1.0,
            "expected_recovery_trades": float('inf'),
            "capital_adequacy_ratio": 0.0
        }

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = len(wins) / len(returns)
    loss_rate = len(losses) / len(returns)

    avg_win = np.mean(wins) if len(wins) > 0 else 1.0
    avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1.0

    # Risk of ruin formula (classic formula):
    # R = ((1 - a) / (1 + a)) ^ U
    # where a = win_rate - loss_rate (edge), U = capital units
    edge = win_rate - loss_rate
    ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

    # Under simple equal sizing
    if edge <= 0:
        ruin_prob = 1.0
    else:
        # Approximate using standard risk-of-ruin equations
        try:
            p = win_rate
            q = loss_rate
            z = ruin_level * start_capital / (avg_loss if avg_loss > 0 else 1.0)
            if q > 0:
                # Classic formula: Ruin Prob = ((q/p)^ratio)^z
                val = ((q / p) ** ratio)
                ruin_prob = float(min(1.0, val ** z))
            else:
                ruin_prob = 0.0
        except Exception:
            ruin_prob = 1.0

    # Expected recovery time in number of trades after a worst-case drawdown
    # Recovery trades = Drawdown amount / Expectancy per trade
    exp = compute_expectancy(returns)
    if exp <= 0:
        expected_recovery = float('inf')
    else:
        worst_dd_money = ruin_level * start_capital
        expected_recovery = float(worst_dd_money / exp)

    # Capital Adequacy Ratio = Capital / (Value at Risk (95%))
    p5 = np.percentile(returns, 5)
    var_95 = abs(p5) if p5 < 0 else 0.0
    if var_95 > 0:
        car = float(start_capital / (var_95 * 10))  # Scale of 10 trades VaR
    else:
        car = 100.0

    return {
        "ruin_probability": ruin_prob,
        "expected_recovery_trades": expected_recovery,
        "capital_adequacy_ratio": car
    }


def compute_mae_mfe_metrics(
    realized_pnls: List[float], max_drawdowns: List[float], max_runups: List[float]
) -> dict[str, List[float]]:
    """Organize Max Adverse Excursion (MAE) and Max Favorable Excursion (MFE) relationships.

    MAE is the maximum unrealized loss during a trade.
    MFE is the maximum unrealized profit during a trade.
    """
    return {
        "pnl": realized_pnls,
        "mae": max_drawdowns,  # Max drawdown per trade
        "mfe": max_runups      # Max runup per trade
    }


def calculate_comprehensive_metrics(
    returns: np.ndarray, equity_curve: np.ndarray, start_capital: Union[float, int]
) -> dict[str, Union[float, int, dict]]:
    """Calculates all advanced metrics and packs them into a structured report dictionary.
    """
    # Safeguard inputs
    if not isinstance(returns, np.ndarray):
        returns = np.array(returns)
    if not isinstance(equity_curve, np.ndarray):
        equity_curve = np.array(equity_curve)
    
    total_profit = float(equity_curve[-1] - start_capital) if len(equity_curve) > 0 else 0.0
    returns_pct = returns / start_capital if start_capital > 0 else returns

    drawdowns, max_dd_pct, peak_idx, trough_idx = compute_drawdowns(equity_curve)
    max_dd_val = float(np.max(np.maximum.accumulate(equity_curve) - equity_curve)) if len(equity_curve) > 0 else 0.0

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = float(len(wins) / len(returns)) if len(returns) > 0 else 0.0

    # Ratios
    sharpe = compute_sharpe_ratio(returns_pct)
    sortino = compute_sortino_ratio(returns_pct)
    
    # Annualized return
    total_days = len(returns)  # assume 1 trade/day for normalization
    ann_return = float((equity_curve[-1] / start_capital) ** (252.0 / max(1, total_days)) - 1.0) if len(equity_curve) > 0 and equity_curve[-1] > 0 else 0.0
    calmar = compute_calmar_ratio(ann_return, max_dd_pct)
    
    rec_factor = compute_recovery_factor(total_profit, max_dd_val)
    profit_factor = compute_profit_factor(returns)
    expectancy = compute_expectancy(returns)
    ulcer = compute_ulcer_index(equity_curve)
    cvar_95 = compute_cvar(returns)
    tail_ratio = compute_tail_ratio(returns)
    fat_tail_info = detect_fat_tails(returns)
    ruin_info = compute_risk_of_ruin_metrics(returns, float(start_capital))

    # Moments
    skew = float(stats.skew(returns)) if len(returns) > 2 else 0.0
    kurt = float(stats.kurtosis(returns, fisher=True)) if len(returns) > 3 else 0.0

    return {
        "total_trades": len(returns),
        "win_rate": win_rate,
        "total_profit": total_profit,
        "max_drawdown_percent": max_dd_pct,
        "max_drawdown_value": max_dd_val,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "recovery_factor": rec_factor,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "ulcer_index": ulcer,
        "cvar_95": cvar_95,
        "tail_ratio": tail_ratio,
        "skewness": skew,
        "kurtosis": kurt,
        "fat_tail": fat_tail_info,
        "ruin": ruin_info,
        "annualized_return": ann_return
    }
