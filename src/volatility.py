"""Volatility forecasting utilities (GARCH + SVI scaffolding).

Provides small, dependency-optional helpers used by the strategy for
IV/volatility forecasting and regime-aware sizing.

This module prefers `arch` when available and falls back to a lightweight
EWMA-based variance estimator when `arch` is not installed so the repo
remains importable in minimal environments.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import logging
import math

try:
    import numpy as np
except Exception:  # pragma: no cover - best-effort imports
    np = None

LOGGER = logging.getLogger(__name__)
_ARCH_MODEL = None
_ARCH_IMPORT_ATTEMPTED = False


def _get_arch_model():
    """Import arch lazily so UI startup does not block on scipy initialization."""
    global _ARCH_MODEL, _ARCH_IMPORT_ATTEMPTED
    if _ARCH_IMPORT_ATTEMPTED:
        return _ARCH_MODEL
    _ARCH_IMPORT_ATTEMPTED = True
    try:
        from arch import arch_model as _arch_model  # type: ignore[import-not-found]
        _ARCH_MODEL = _arch_model
    except Exception as exc:
        LOGGER.debug("arch import unavailable, using EWMA fallback: %s", exc)
        _ARCH_MODEL = None
    return _ARCH_MODEL


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
    arch_model = _get_arch_model()
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
    arch_model = _get_arch_model() if len(returns) >= 50 else None
    if arch_model is not None:
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


def svi_raw(theta: float, param: dict) -> float:
    """Raw SVI formula: w = a + b * (rho * (theta - m) + sqrt((theta - m)^2 + sigma^2))
    
    Args:
        theta: Log-moneyness (log(K/F))
        param: Dictionary with keys 'a', 'b', 'rho', 'm', 'sigma'
    
    Returns:
        Variance (w = sigma^2 * T)
    """
    try:
        a = float(param.get('a', 0.0))
        b = float(param.get('b', 0.1))
        rho = float(param.get('rho', 0.0))
        m = float(param.get('m', 0.0))
        sigma = float(param.get('sigma', 0.1))
        
        # SVI formula
        d = theta - m
        sqrt_term = math.sqrt(d**2 + sigma**2)
        w = a + b * (rho * d + sqrt_term)
        
        # Ensure non-negative variance
        return max(0.0, w)
    except Exception:
        return 0.0


def svi_total_variance(theta: float, param: dict) -> float:
    """Calculate total variance from SVI parameters.
    
    Returns:
        Total variance (w = sigma^2 * T)
    """
    return svi_raw(theta, param)


def svi_implied_vol(theta: float, param: dict, T: float = 1.0) -> float:
    """Calculate implied volatility from SVI parameters.
    
    Args:
        theta: Log-moneyness (log(K/F))
        param: SVI parameters
        T: Time to expiration (in years)
    
    Returns:
        Implied volatility (annualized)
    """
    try:
        w = svi_raw(theta, param)
        if T <= 0:
            T = 0.001
        iv = math.sqrt(w / T)
        return min(max(iv, 0.01), 5.0)  # Cap between 1% and 500%
    except Exception:
        return 0.20  # Default 20% IV


def svi_fit(log_moneyness: list, total_variance: list, initial_guess: dict = None) -> dict:
    """Fit SVI parameters to market data using least squares.
    
    Args:
        log_moneyness: List of log(K/F) values
        total_variance: List of corresponding total variances (sigma^2 * T)
        initial_guess: Optional initial parameters
    
    Returns:
        Dictionary with fitted parameters 'a', 'b', 'rho', 'm', 'sigma'
        and fit quality metrics.
    """
    try:
        import numpy as np
        from scipy.optimize import minimize
        
        theta_arr = np.array(log_moneyness)
        w_arr = np.array(total_variance)
        
        if len(theta_arr) < 5:
            return _svi_fit_fallback(log_moneyness, total_variance)
        
        # Default initial guess
        if initial_guess is None:
            w_mean = np.mean(w_arr)
            theta_range = np.max(theta_arr) - np.min(theta_arr)
            initial_guess = {
                'a': float(w_mean * 0.5),
                'b': 0.1,
                'rho': 0.0,
                'm': float(np.median(theta_arr)),
                'sigma': max(0.1, float(theta_range * 0.1)),
            }
        
        def objective(params):
            a, b, rho, m, sigma = params
            # Penalize invalid parameters
            if b <= 0 or b > 1 or abs(rho) >= 1 or sigma <= 0:
                return 1e10
            
            w_pred = a + b * (rho * (theta_arr - m) + np.sqrt((theta_arr - m)**2 + sigma**2))
            return float(np.sum((w_arr - w_pred)**2))
        
        x0 = [initial_guess['a'], initial_guess['b'], initial_guess['rho'], 
              initial_guess['m'], initial_guess['sigma']]
        
        result = minimize(objective, x0, method='L-BFGS-B',
                         bounds=[(-1, 1), (0.001, 1), (-0.999, 0.999), 
                                 (None, None), (0.001, 5)],
                         options={'maxiter': 1000})
        
        if result.success:
            params = {
                'a': float(result.x[0]),
                'b': float(result.x[1]),
                'rho': float(result.x[2]),
                'm': float(result.x[3]),
                'sigma': float(result.x[4]),
            }
            # Calculate fit quality
            w_pred = svi_raw(theta_arr, params)
            ss_res = float(np.sum((w_arr - w_pred)**2))
            ss_tot = float(np.sum((w_arr - np.mean(w_arr))**2))
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            params['r_squared'] = r_squared
            params['fit_success'] = True
            return params
        else:
            return _svi_fit_fallback(log_moneyness, total_variance)
            
    except ImportError:
        return _svi_fit_fallback(log_moneyness, total_variance)
    except Exception as e:
        LOGGER.warning("SVI fit failed: %s", e)
        return _svi_fit_fallback(log_moneyness, total_variance)


def _svi_fit_fallback(log_moneyness: list, total_variance: list) -> dict:
    """Fallback SVI fit using simple heuristics."""
    try:
        import numpy as np
        theta_arr = np.array(log_moneyness)
        w_arr = np.array(total_variance)
        
        # Simple linear fit as fallback
        # w ≈ a + b * theta (simplified)
        if len(theta_arr) >= 2:
            coeffs = np.polyfit(theta_arr, w_arr, 1)
            return {
                'a': float(coeffs[1]),
                'b': min(0.5, max(0.01, float(coeffs[0]))),
                'rho': 0.0,
                'm': float(np.median(theta_arr)),
                'sigma': 0.1,
                'r_squared': 0.5,
                'fit_success': False,
                'fallback': True,
            }
    except Exception:
        pass
    
    return {
        'a': 0.04,
        'b': 0.1,
        'rho': 0.0,
        'm': 0.0,
        'sigma': 0.1,
        'r_squared': 0.0,
        'fit_success': False,
        'fallback': True,
    }


def build_vol_surface(option_chain: list, spot_price: float, T: float = 1.0) -> dict:
    """Build volatility surface from option chain data.
    
    Args:
        option_chain: List of dicts with keys 'strike', 'iv', 'type' (call/put)
        spot_price: Current spot price
        T: Time to expiration
    
    Returns:
        Dictionary with SVI parameters and surface metrics
    """
    try:
        if not option_chain or spot_price <= 0:
            return {}
        
        # Extract log-moneyness and total variance
        log_money = []
        total_var = []
        
        for opt in option_chain:
            K = float(opt.get('strike', 0))
            iv = float(opt.get('iv', 0))
            if K <= 0 or iv <= 0:
                continue
            
            theta = math.log(K / spot_price)
            w = (iv ** 2) * T
            log_money.append(theta)
            total_var.append(w)
        
        if len(log_money) < 5:
            return {'error': 'insufficient_data', 'points': len(log_money)}
        
        # Fit SVI
        svi_params = svi_fit(log_money, total_var)
        
        # Calculate ATM volatility
        atm_iv = svi_implied_vol(0.0, svi_params, T)
        
        # Calculate wing slopes (25-delta calls and puts)
        # Approximate: 25-delta ~ log(K/F) = -0.5*sigma for calls, +0.5*sigma for puts
        sigma = svi_params.get('sigma', 0.1)
        theta_25c = -0.5 * sigma
        theta_25p = 0.5 * sigma
        
        iv_25c = svi_implied_vol(theta_25c, svi_params, T)
        iv_25p = svi_implied_vol(theta_25p, svi_params, T)
        
        return {
            'svi_params': svi_params,
            'atm_iv': atm_iv,
            'iv_25c': iv_25c,
            'iv_25p': iv_25p,
            'skew': iv_25p - iv_25c,  # Positive = put skew
            'spot': spot_price,
            'T': T,
            'data_points': len(log_money),
        }
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to build vol surface: {e}")
        return {'error': str(e)}
