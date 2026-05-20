"""Volatility forecasting utilities (GARCH + SVI scaffolding).

Provides small, dependency-optional helpers used by the strategy for
IV/volatility forecasting and regime-aware sizing.

This module prefers `arch` when available and falls back to a lightweight
EWMA-based variance estimator when `arch` is not installed so the repo
remains importable in minimal environments.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import math

try:
    import numpy as np
    import pandas as pd
except Exception:  # pragma: no cover - best-effort imports
    np = None
    pd = None

try:
    from arch import arch_model  # type: ignore[import-not-found]
except Exception:
    arch_model = None


def _ewma_vol(series: Iterable[float], span: int = 20) -> float:
    """Simple EWMA volatility estimator (annualized-like scale).

    Returns the last EWMA standard deviation computed on the series.
    """
    if not series:
        return 0.0
    try:
        arr = list(series)
        if len(arr) < 2:
            return 0.0
        alpha = 2.0 / (span + 1.0)
        s2 = arr[0] ** 2
        for x in arr[1:]:
            s2 = alpha * (x ** 2) + (1 - alpha) * s2
        return math.sqrt(s2)
    except Exception:
        return 0.0


def fit_garch(returns: List[float], p: int = 1, q: int = 1, disp: bool = False) -> Optional[object]:
    """Fit a GARCH(p,q) model to the return series.

    Returns the fitted model object when `arch` is available, otherwise None.
    """
    if arch_model is None:
        return None
    try:
        import numpy as _np

        r = _np.asarray(returns) * 100.0
        am = arch_model(r, p=p, q=q, mean="zero", vol="Garch", dist="normal")
        res = am.fit(disp=0 if not disp else 1)
        return res
    except Exception:
        return None


def forecast_volatility(returns: List[float], horizon: int = 1, span: int = 20) -> float:
    """Forecast short-term volatility.

    Strategy code should call this to get a volatility estimate (same units
    as input returns). When GARCH is available we use it; otherwise EWMA.
    """
    if arch_model is not None and len(returns) >= 50:
        try:
            res = fit_garch(returns)
            if res is not None:
                f = res.forecast(horizon=horizon, reindex=False)
                var = f.variance.values[-1, -1]
                return float(math.sqrt(max(var, 0.0)))
        except Exception:
            pass
    # fallback
    return _ewma_vol(returns, span=span)


def svi_fit(*_args, **_kwargs):
    """Placeholder for SVI surface fitting.

    Implementing a full SVI routine is beyond the scope of the scaffold;
    this function is a clear integration point for a pluggable SVI
    implementation. Returns None for now.
    """
    return None
