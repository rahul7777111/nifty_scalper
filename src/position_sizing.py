"""Position sizing helpers: volatility targeting and constrained Kelly."""
from __future__ import annotations

import math
from typing import List


def volatility_target_size(cash: float, vol_target: float, forecast_vol: float, price: float) -> int:
    """Return number of contracts/lots to target a volatility exposure.

    `vol_target` and `forecast_vol` are in same units (e.g., daily vol).
    """
    try:
        if forecast_vol <= 0 or price <= 0:
            return 0
        dollar_risk = cash * vol_target
        size = dollar_risk / (price * forecast_vol)
        return max(0, int(size))
    except Exception:
        return 0


def calculate_probabilistic_bet_size(
    calibrated_prob: float,
    optimal_threshold: float,
    standard_error: float = 0.10,
) -> float:
    """Return a bounded bet-size multiplier from calibrated ML confidence.

    The multiplier follows the standardized divergence formula and is clipped
    to [0.0, 1.0] so ML can only scale down or match baseline sizing.
    """

    try:
        probability = max(0.0, min(1.0, float(calibrated_prob)))
        threshold = max(0.0, min(1.0, float(optimal_threshold)))
        stderr = max(float(standard_error), 1e-7)
        if probability < threshold:
            return 0.0
        z_score = (probability - threshold) / stderr
        normal_cdf = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
        multiplier = 2.0 * normal_cdf - 1.0
        return max(0.0, min(1.0, float(multiplier)))
    except Exception:
        return 0.0


def calculate_position_size(
    *,
    baseline_lots: int,
    calibrated_prob: float,
    optimal_threshold: float,
    standard_error: float = 0.10,
    current_drawdown_pct: float = 0.0,
    max_allowable_drawdown: float = 0.05,
    enforce_min_lot: bool = False,
) -> dict:
    """Apply probabilistic sizing and drawdown throttling to baseline lots."""

    try:
        baseline = max(0, int(baseline_lots))
    except Exception:
        baseline = 0

    bet_multiplier = calculate_probabilistic_bet_size(
        calibrated_prob=calibrated_prob,
        optimal_threshold=optimal_threshold,
        standard_error=standard_error,
    )
    scaled_lots = int(math.floor(float(baseline) * bet_multiplier)) if baseline > 0 else 0

    try:
        drawdown = max(0.0, float(current_drawdown_pct))
    except Exception:
        drawdown = 0.0
    try:
        max_dd = max(float(max_allowable_drawdown), 1e-7)
    except Exception:
        max_dd = 0.05

    drawdown_modifier = 1.0
    if drawdown > 0.0:
        drawdown_modifier = max(0.0, 1.0 - (drawdown / max_dd))
    final_lots = int(math.floor(float(scaled_lots) * drawdown_modifier)) if scaled_lots > 0 else 0

    if enforce_min_lot and baseline > 0 and final_lots <= 0:
        final_lots = 1

    return {
        "ml_bet_multiplier": float(bet_multiplier),
        "drawdown_modifier": float(max(0.0, min(1.0, drawdown_modifier))),
        "baseline_vol_lots": int(baseline),
        "final_allocated_lots": int(max(0, final_lots)),
    }


def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    try:
        if win_loss_ratio <= 0:
            return 0.0
        b = win_loss_ratio
        p = win_rate
        q = 1 - p
        f = (b * p - q) / b
        return max(0.0, min(1.0, f))
    except Exception:
        return 0.0


def optimize_sizes_cvar(returns_matrix, target_cvar: float = 0.02, budget: float = 1.0):
    """Wrapper to `risk_cvar.optimize_position_sizes` with a safe fallback.

    `returns_matrix` should be a list of lists (scenarios x assets). Returns
    a list of weights (floats) or None on failure.
    """
    try:
        from risk_cvar import optimize_position_sizes

        return optimize_position_sizes(returns_matrix, target_cvar=target_cvar, budget=budget)
    except Exception:
        # Fallback: equal-weight heuristic scaled to budget
        try:
            n = len(returns_matrix[0]) if returns_matrix and returns_matrix[0] else 0
            if n <= 0:
                return None
            w = [float(budget) / float(n) for _ in range(n)]
            return w
        except Exception:
            return None
