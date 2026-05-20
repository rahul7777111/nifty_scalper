"""Portfolio-level CVaR utilities and position sizing helpers.

Provides a simple Conditional Value at Risk estimator and an optimizer
stub that uses `cvxpy` when available, otherwise falls back to a
heuristic scaling approach.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

try:
    import numpy as np
except Exception:
    np = None

try:
    import cvxpy as cp
except Exception:
    cp = None


def compute_cvar(returns: Iterable[float], alpha: float = 0.95) -> float:
    """Compute empirical CVaR (average loss beyond the VaR) at level alpha.

    `returns` are assumed to be P&L series (positive = profit). We compute
    losses as -returns so larger positive values indicate worse losses.
    """
    try:
        vals = list(returns)
        if not vals:
            return 0.0
        import math

        losses = [-v for v in vals]
        losses.sort()
        k = max(1, int(len(losses) * (1 - alpha)))
        tail = losses[:k]
        if not tail:
            return 0.0
        return float(sum(tail) / len(tail))
    except Exception:
        return 0.0


def optimize_position_sizes(
    returns_matrix: List[List[float]], target_cvar: float, budget: float = 1.0
) -> Optional[List[float]]:
    """Given a matrix of scenario returns (rows=scenarios, cols=assets),
    find non-negative weights that meet a target CVaR using CVX when
    available.

    Returns a list of weights summing to <= budget or None on failure.
    """
    if not returns_matrix:
        return None
    try:
        import numpy as _np

        R = _np.asarray(returns_matrix)
        n_assets = R.shape[1]
        if cp is None:
            # simple heuristic: scale equally until estimated CVaR <= target
            w = _np.ones(n_assets) * (budget / float(n_assets))
            port_returns = R.dot(w)
            cvar = compute_cvar(port_returns.tolist())
            if cvar <= target_cvar:
                return [float(x) for x in w]
            factor = target_cvar / max(cvar, 1e-9)
            w = w * factor
            return [float(x) for x in w]

        # cvxpy formulation: minimize negative expected return subject to CVaR constraint
        w = cp.Variable(n_assets, nonneg=True)
        z = cp.Variable(R.shape[0])
        tau = cp.Variable()
        portfolio = R @ w
        alpha = 0.95
        # CVaR constraints (standard linearized formulation)
        VaR = tau
        loss = -portfolio
        constraints = [z >= 0, z >= loss - VaR]
        cvar_expr = VaR + (1.0 / ((1 - alpha) * R.shape[0])) * cp.sum(z)
        constraints += [cp.sum(w) <= budget, cvar_expr <= target_cvar]
        # objective: maximize expected return
        obj = cp.Maximize(cp.sum(cp.multiply(cp.Constant(R.mean(axis=0)), w)))
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        if w.value is None:
            return None
        return [float(max(0.0, x)) for x in w.value.tolist()]
    except Exception:
        return None
