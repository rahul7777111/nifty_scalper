from __future__ import annotations

import math
from typing import Literal


OptionType = Literal["CE", "PE"]


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def _d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    iv: float,
) -> tuple[float, float]:
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or iv <= 0:
        raise ValueError("Invalid inputs for Black–Scholes greeks")
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike)
        + (rate + 0.5 * iv * iv) * time_to_expiry
    ) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    return d1, d2


def delta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    iv: float,
    option_type: OptionType,
) -> float:
    d1, _ = _d1_d2(spot, strike, time_to_expiry, rate, iv)
    if option_type == "CE":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def gamma(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    iv: float,
) -> float:
    d1, _ = _d1_d2(spot, strike, time_to_expiry, rate, iv)
    return _norm_pdf(d1) / (spot * iv * math.sqrt(time_to_expiry))


def vega(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    iv: float,
) -> float:
    d1, _ = _d1_d2(spot, strike, time_to_expiry, rate, iv)
    return spot * _norm_pdf(d1) * math.sqrt(time_to_expiry)


def theta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    iv: float,
    option_type: OptionType,
) -> float:
    """Black–Scholes theta (per year, same units as option premium)."""

    d1, d2 = _d1_d2(spot, strike, time_to_expiry, rate, iv)
    sqrt_t = math.sqrt(time_to_expiry)
    first_term = -(spot * _norm_pdf(d1) * iv) / (2.0 * sqrt_t)
    df = math.exp(-rate * time_to_expiry)
    if option_type == "CE":
        second_term = -rate * strike * df * _norm_cdf(d2)
    else:
        second_term = rate * strike * df * _norm_cdf(-d2)
    return first_term + second_term


def bs_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    iv: float,
    option_type: OptionType,
) -> float:
    """Black–Scholes price for a European call/put (no dividends)."""

    d1, d2 = _d1_d2(spot, strike, time_to_expiry, rate, iv)
    df = math.exp(-rate * time_to_expiry)
    if option_type == "CE":
        return spot * _norm_cdf(d1) - strike * df * _norm_cdf(d2)
    return strike * df * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: OptionType,
    *,
    initial_iv: float = 0.20,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> float | None:
    """Compute implied volatility from a market option price.

    Returns IV as a decimal (e.g. 0.18 for 18%) or None if it can't be solved.
    """

    try:
        p = float(price)
        s = float(spot)
        k = float(strike)
        t = float(time_to_expiry)
        r = float(rate)
    except Exception:
        return None

    if p <= 0 or s <= 0 or k <= 0 or t <= 0:
        return None

    df = math.exp(-r * t)
    if option_type == "CE":
        lower = max(0.0, s - k * df)
        upper = s
    else:
        lower = max(0.0, k * df - s)
        upper = k * df

    if p < lower - 1e-6:
        return None
    if p > upper + 1e-6:
        return None

    if abs(p - lower) <= 1e-8:
        return 1e-6

    lo = 1e-6
    hi = 5.0
    try:
        iv = float(initial_iv)
    except Exception:
        iv = 0.20
    if iv <= 0:
        iv = 0.20
    iv = min(max(iv, lo), hi)

    for _ in range(int(max_iter)):
        try:
            model_p = bs_price(s, k, t, r, iv, option_type)
        except Exception:
            break
        diff = model_p - p
        if abs(diff) <= float(tol):
            return float(iv)
        try:
            v = vega(s, k, t, r, iv)
        except Exception:
            v = 0.0
        if v <= 1e-12:
            break
        iv = iv - (diff / v)
        if iv < lo:
            iv = lo
        elif iv > hi:
            iv = hi

    try:
        p_lo = bs_price(s, k, t, r, lo, option_type)
        p_hi = bs_price(s, k, t, r, hi, option_type)
    except Exception:
        return None
    if p < p_lo - 1e-9 or p > p_hi + 1e-9:
        return None

    for _ in range(80):
        mid = (lo + hi) / 2.0
        try:
            p_mid = bs_price(s, k, t, r, mid, option_type)
        except Exception:
            return None
        if abs(p_mid - p) <= float(tol):
            return float(mid)
        if p_mid < p:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)
