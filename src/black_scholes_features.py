from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time
from typing import Any

import numpy as np
import pandas as pd


OPTION_TYPE_MAP = {
    "CE": "CE",
    "CALL": "CE",
    "C": "CE",
    "PE": "PE",
    "PUT": "PE",
    "P": "PE",
}

UNDERLYING_COLUMNS = [
    "underlying_spot_price",
    "spot",
    "underlying",
    "underlying_price",
    "close_underlying",
    "spot_ltp",
    "spot_close",
    "ctx_spot",
    "underlying_spot",
]
OPTION_PRICE_COLUMNS = ["option_ltp", "ltp", "close", "premium", "last_price"]
STRIKE_COLUMNS = ["strike", "strike_price"]
OPTION_TYPE_COLUMNS = ["option_type", "right", "cp", "type"]
EXPIRY_COLUMNS = ["expiry", "expiry_date", "expiration", "contract_expiry"]
TIMESTAMP_COLUMNS = ["timestamp", "datetime", "ts", "candle_time"]
IV_COLUMNS = ["iv", "implied_volatility", "bs_iv"]
BID_COLUMNS = ["bid", "bid_price", "best_bid", "bestBidPrice"]
ASK_COLUMNS = ["ask", "ask_price", "best_ask", "bestAskPrice"]
VOLUME_COLUMNS = ["volume", "vol", "traded_volume"]
OI_COLUMNS = ["oi", "open_interest", "openInterest"]

DEFAULT_EXPIRY_TIME = time(hour=15, minute=30)


@dataclass(frozen=True)
class ColumnResolution:
    underlying_col: str
    option_price_col: str
    strike_col: str
    option_type_col: str
    expiry_col: str
    timestamp_col: str
    iv_col: str | None
    bid_col: str | None
    ask_col: str | None
    volume_col: str | None
    oi_col: str | None


def _find_first_column(columns: list[str], candidates: list[str], *, required: bool = True) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if required:
        raise RuntimeError(f"Missing required dataset column from candidates: {candidates}")
    return None


def resolve_dataset_columns(df: pd.DataFrame) -> ColumnResolution:
    columns = list(df.columns)
    return ColumnResolution(
        underlying_col=_find_first_column(columns, UNDERLYING_COLUMNS),
        option_price_col=_find_first_column(columns, OPTION_PRICE_COLUMNS),
        strike_col=_find_first_column(columns, STRIKE_COLUMNS),
        option_type_col=_find_first_column(columns, OPTION_TYPE_COLUMNS),
        expiry_col=_find_first_column(columns, EXPIRY_COLUMNS),
        timestamp_col=_find_first_column(columns, TIMESTAMP_COLUMNS),
        iv_col=_find_first_column(columns, IV_COLUMNS, required=False),
        bid_col=_find_first_column(columns, BID_COLUMNS, required=False),
        ask_col=_find_first_column(columns, ASK_COLUMNS, required=False),
        volume_col=_find_first_column(columns, VOLUME_COLUMNS, required=False),
        oi_col=_find_first_column(columns, OI_COLUMNS, required=False),
    )


def normalize_option_type(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().upper()
    return OPTION_TYPE_MAP.get(text)


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "" or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price(spot: float, strike: float, t: float, rate: float, q: float, sigma: float, option_type: str) -> float:
    if spot <= 0 or strike <= 0 or t <= 0 or sigma <= 0:
        raise ValueError("Invalid Black-Scholes inputs")
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if option_type == "CE":
        return spot * math.exp(-q * t) * _norm_cdf(d1) - strike * math.exp(-rate * t) * _norm_cdf(d2)
    return strike * math.exp(-rate * t) * _norm_cdf(-d2) - spot * math.exp(-q * t) * _norm_cdf(-d1)


def intrinsic_value(spot: float, strike: float, option_type: str) -> float:
    if option_type == "CE":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def solve_implied_volatility(
    price: float,
    spot: float,
    strike: float,
    t: float,
    rate: float,
    q: float,
    option_type: str,
    *,
    low: float = 0.0001,
    high: float = 5.0,
    tolerance: float = 1e-6,
    max_iter: int = 100,
) -> tuple[float | None, bool, str]:
    if any(value is None for value in (price, spot, strike, t)):
        return None, False, "missing_required_value"
    if price <= 0 or spot <= 0 or strike <= 0 or t <= 0:
        return None, False, "non_positive_input"
    intrinsic = intrinsic_value(spot, strike, option_type)
    if price < intrinsic - 1e-9:
        return None, False, "price_below_intrinsic"
    try:
        low_price = bs_price(spot, strike, t, rate, q, low, option_type)
        high_price = bs_price(spot, strike, t, rate, q, high, option_type)
    except Exception:
        return None, False, "pricing_failed"
    if price < low_price - 1e-8 or price > high_price + 1e-8:
        return None, False, "price_outside_model_bounds"
    lo = low
    hi = high
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        guess = bs_price(spot, strike, t, rate, q, mid, option_type)
        if abs(guess - price) <= tolerance:
            return mid, True, "solved_from_price"
        if guess < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0, True, "solved_from_price"


def compute_greeks(spot: float, strike: float, t: float, rate: float, q: float, sigma: float, option_type: str) -> dict[str, float]:
    if spot <= 0 or strike <= 0 or t <= 0 or sigma <= 0:
        raise ValueError("Invalid inputs for greeks")
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pdf = _norm_pdf(d1)
    sign = 1.0 if option_type == "CE" else -1.0
    delta = math.exp(-q * t) * _norm_cdf(d1) if option_type == "CE" else math.exp(-q * t) * (_norm_cdf(d1) - 1.0)
    gamma = math.exp(-q * t) * pdf / (spot * sigma * sqrt_t)
    vega = spot * math.exp(-q * t) * pdf * sqrt_t
    rho = sign * strike * t * math.exp(-rate * t) * _norm_cdf(sign * d2)
    theta_yearly = (
        -(spot * pdf * sigma * math.exp(-q * t)) / (2.0 * sqrt_t)
        - sign * rate * strike * math.exp(-rate * t) * _norm_cdf(sign * d2)
        + sign * q * spot * math.exp(-q * t) * _norm_cdf(sign * d1)
    )
    return {
        "bs_delta": delta,
        "bs_gamma": gamma,
        "bs_theta": theta_yearly / 365.0,
        "bs_vega": vega,
        "bs_rho": rho,
    }


def parse_timestamp(value: Any, timezone_name: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is None:
        return ts.tz_localize(timezone_name)
    return ts.tz_convert(timezone_name)


def parse_expiry(value: Any, timezone_name: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is not None:
        ts = ts.tz_convert(timezone_name)
    else:
        text = str(value).strip()
        if len(text) <= 10:
            ts = pd.Timestamp.combine(ts.date(), DEFAULT_EXPIRY_TIME)
        ts = ts.tz_localize(timezone_name)
    return ts


def enrich_option_frame(
    df: pd.DataFrame,
    *,
    risk_free_rate: float = 0.065,
    dividend_yield: float = 0.0,
    timezone_name: str = "Asia/Kolkata",
    symbol: str = "NIFTY",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolution = resolve_dataset_columns(df)
    enriched = df.copy()
    enriched["_normalized_option_type"] = enriched[resolution.option_type_col].map(normalize_option_type)
    enriched["_parsed_timestamp"] = enriched[resolution.timestamp_col].map(lambda value: parse_timestamp(value, timezone_name))
    enriched["_parsed_expiry"] = enriched[resolution.expiry_col].map(lambda value: parse_expiry(value, timezone_name))
    total_rows = int(len(enriched))
    row_errors = 0
    solve_ok_count = 0
    bid_ask_count = 0
    final_iv_from_existing = 0

    output_rows: list[dict[str, Any]] = []
    for row in enriched.to_dict(orient="records"):
        result = {key: np.nan for key in [
            "bs_iv",
            "bs_iv_solve_ok",
            "bs_iv_source",
            "final_iv",
            "bs_delta",
            "bs_gamma",
            "bs_theta",
            "bs_vega",
            "bs_rho",
            "moneyness",
            "log_moneyness",
            "distance_from_atm",
            "distance_from_atm_pct",
            "intrinsic_value",
            "extrinsic_value",
            "time_to_expiry_days",
            "time_to_expiry_years",
            "is_expiry_day",
            "is_near_expiry",
            "option_side_normalized",
            "bid_ask_spread",
            "bid_ask_spread_pct",
            "mid_price",
            "ltp_vs_mid_diff",
            "ltp_vs_mid_diff_pct",
            "has_valid_bid_ask",
            "low_price_flag",
            "wide_spread_flag",
            "bad_iv_flag",
            "bad_greek_flag",
            "stale_or_invalid_quote_flag",
            "deep_itm_flag",
            "deep_otm_flag",
            "greeks_source",
            "greeks_quality_score",
            "row_enrichment_ok",
            "row_enrichment_error",
        ]}
        try:
            option_type = row["_normalized_option_type"]
            spot = safe_float(row[resolution.underlying_col])
            price = safe_float(row[resolution.option_price_col])
            strike = safe_float(row[resolution.strike_col])
            ts = row["_parsed_timestamp"]
            expiry = row["_parsed_expiry"]
            existing_iv = safe_float(row.get(resolution.iv_col)) if resolution.iv_col else None
            bid = safe_float(row.get(resolution.bid_col)) if resolution.bid_col else None
            ask = safe_float(row.get(resolution.ask_col)) if resolution.ask_col else None
            volume = safe_float(row.get(resolution.volume_col)) if resolution.volume_col else None
            open_interest = safe_float(row.get(resolution.oi_col)) if resolution.oi_col else None
            if option_type is None:
                raise ValueError("invalid_option_type")
            if ts is pd.NaT or expiry is pd.NaT:
                raise ValueError("invalid_timestamp_or_expiry")
            if spot is None or price is None or strike is None:
                raise ValueError("missing_numeric_input")
            t_seconds = (expiry - ts).total_seconds()
            t_years = max(t_seconds / (365.0 * 24.0 * 3600.0), 0.0)
            t_days = max(t_seconds / (24.0 * 3600.0), 0.0)
            intrinsic = intrinsic_value(spot, strike, option_type)
            extrinsic = price - intrinsic
            moneyness = spot / strike if strike else np.nan
            log_moneyness = math.log(moneyness) if moneyness and moneyness > 0 else np.nan
            distance_from_atm = spot - strike
            distance_from_atm_pct = distance_from_atm / strike if strike else np.nan
            is_expiry_day = bool(ts.date() == expiry.date())
            is_near_expiry = bool(t_days <= 2.0)

            bs_iv, bs_iv_solve_ok, bs_iv_source = solve_implied_volatility(
                price,
                spot,
                strike,
                max(t_years, 1e-9),
                risk_free_rate,
                dividend_yield,
                option_type,
            )
            final_iv = bs_iv
            final_iv_source = "black_scholes_solved"
            if existing_iv is not None and existing_iv > 0:
                final_iv = existing_iv
                final_iv_source = "existing_dataset_iv"
                final_iv_from_existing += 1
            elif bs_iv_solve_ok:
                solve_ok_count += 1

            greeks = {
                "bs_delta": np.nan,
                "bs_gamma": np.nan,
                "bs_theta": np.nan,
                "bs_vega": np.nan,
                "bs_rho": np.nan,
            }
            if final_iv is not None and final_iv > 0 and t_years > 0:
                greeks = compute_greeks(spot, strike, t_years, risk_free_rate, dividend_yield, final_iv, option_type)
            bad_iv_flag = not (final_iv is not None and 0.0001 <= final_iv <= 5.0)
            bad_greek_flag = any(pd.isna(value) for value in greeks.values())

            has_valid_bid_ask = bool(
                bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid
            )
            mid_price = ((bid + ask) / 2.0) if has_valid_bid_ask else np.nan
            bid_ask_spread = (ask - bid) if has_valid_bid_ask else np.nan
            bid_ask_spread_pct = (bid_ask_spread / mid_price) if has_valid_bid_ask and mid_price else np.nan
            ltp_vs_mid_diff = (price - mid_price) if has_valid_bid_ask and mid_price == mid_price else np.nan
            ltp_vs_mid_diff_pct = (ltp_vs_mid_diff / mid_price) if has_valid_bid_ask and mid_price else np.nan
            if has_valid_bid_ask:
                bid_ask_count += 1
            low_price_flag = bool(price <= 1.0)
            wide_spread_flag = bool(has_valid_bid_ask and bid_ask_spread_pct > 0.10)
            stale_or_invalid_quote_flag = bool((has_valid_bid_ask and price <= 0) or (has_valid_bid_ask and abs(ltp_vs_mid_diff_pct) > 0.25))
            deep_itm_flag = bool((option_type == "CE" and moneyness >= 1.05) or (option_type == "PE" and moneyness <= 0.95))
            deep_otm_flag = bool((option_type == "CE" and moneyness <= 0.95) or (option_type == "PE" and moneyness >= 1.05))

            quality_score = 1.0
            quality_score -= 0.35 if bad_iv_flag else 0.0
            quality_score -= 0.25 if bad_greek_flag else 0.0
            quality_score -= 0.20 if wide_spread_flag else 0.0
            quality_score -= 0.10 if low_price_flag else 0.0
            quality_score -= 0.10 if stale_or_invalid_quote_flag else 0.0
            quality_score -= 0.05 if volume is not None and volume <= 0 else 0.0
            quality_score -= 0.05 if open_interest is not None and open_interest <= 0 else 0.0

            result.update(
                {
                    "bs_iv": bs_iv,
                    "bs_iv_solve_ok": bool(bs_iv_solve_ok),
                    "bs_iv_source": bs_iv_source,
                    "final_iv": final_iv,
                    **greeks,
                    "moneyness": moneyness,
                    "log_moneyness": log_moneyness,
                    "distance_from_atm": distance_from_atm,
                    "distance_from_atm_pct": distance_from_atm_pct,
                    "intrinsic_value": intrinsic,
                    "extrinsic_value": extrinsic,
                    "time_to_expiry_days": t_days,
                    "time_to_expiry_years": t_years,
                    "is_expiry_day": is_expiry_day,
                    "is_near_expiry": is_near_expiry,
                    "option_side_normalized": option_type,
                    "bid_ask_spread": bid_ask_spread,
                    "bid_ask_spread_pct": bid_ask_spread_pct,
                    "mid_price": mid_price,
                    "ltp_vs_mid_diff": ltp_vs_mid_diff,
                    "ltp_vs_mid_diff_pct": ltp_vs_mid_diff_pct,
                    "has_valid_bid_ask": has_valid_bid_ask,
                    "low_price_flag": low_price_flag,
                    "wide_spread_flag": wide_spread_flag,
                    "bad_iv_flag": bad_iv_flag,
                    "bad_greek_flag": bad_greek_flag,
                    "stale_or_invalid_quote_flag": stale_or_invalid_quote_flag,
                    "deep_itm_flag": deep_itm_flag,
                    "deep_otm_flag": deep_otm_flag,
                    "greeks_source": final_iv_source,
                    "greeks_quality_score": max(0.0, min(1.0, quality_score)),
                    "row_enrichment_ok": not (bad_iv_flag and bad_greek_flag),
                    "row_enrichment_error": "",
                }
            )
        except Exception as exc:
            row_errors += 1
            result.update(
                {
                    "row_enrichment_ok": False,
                    "row_enrichment_error": str(exc),
                    "bad_iv_flag": True,
                    "bad_greek_flag": True,
                }
            )
        output_rows.append(result)

    feature_df = pd.DataFrame(output_rows)
    feature_columns_to_add = [column for column in feature_df.columns if column not in df.columns]
    merged = pd.concat([df.reset_index(drop=True), feature_df[feature_columns_to_add]], axis=1)
    merged = merged.drop(columns=["_normalized_option_type", "_parsed_timestamp", "_parsed_expiry"], errors="ignore")
    metadata = {
        "symbol": symbol,
        "total_rows": total_rows,
        "row_error_count": row_errors,
        "solve_ok_count": solve_ok_count,
        "rows_with_valid_bid_ask": bid_ask_count,
        "final_iv_from_existing_count": final_iv_from_existing,
        "columns_used": {
            "underlying": resolution.underlying_col,
            "option_price": resolution.option_price_col,
            "strike": resolution.strike_col,
            "option_type": resolution.option_type_col,
            "expiry": resolution.expiry_col,
            "timestamp": resolution.timestamp_col,
            "existing_iv": resolution.iv_col,
            "bid": resolution.bid_col,
            "ask": resolution.ask_col,
            "volume": resolution.volume_col,
            "open_interest": resolution.oi_col,
        },
        "notes": [
            "Theta is per day in option premium units.",
            "Vega is per 1.00 volatility change, not per 1 percent point.",
            "Rho is per 1.00 rate change, not per 1 percent point.",
            "No source columns were overwritten.",
            f"Skipped duplicate derived columns already present in source: {[column for column in feature_df.columns if column in df.columns]}",
        ],
    }
    return merged, metadata
