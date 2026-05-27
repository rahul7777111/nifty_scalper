"""Position sizing helpers: volatility targeting and constrained Kelly."""
from __future__ import annotations

from typing import List, Optional
import math


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
