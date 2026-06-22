from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate historical IV/Greeks for research only.")
    parser.add_argument("--option-dataset", required=True)
    parser.add_argument("--spot-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--risk-free-rate", type=float, default=0.065)
    parser.add_argument("--dividend-yield", type=float, default=0.0)
    return parser.parse_args()


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_price(spot: float, strike: float, t: float, rate: float, q: float, sigma: float, option_type: str) -> float:
    if spot <= 0 or strike <= 0 or t <= 0 or sigma <= 0:
        intrinsic = max(spot - strike, 0.0) if option_type == "CE" else max(strike - spot, 0.0)
        return intrinsic
    d1 = (math.log(spot / strike) + (rate - q + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if option_type == "CE":
        return spot * math.exp(-q * t) * _norm_cdf(d1) - strike * math.exp(-rate * t) * _norm_cdf(d2)
    return strike * math.exp(-rate * t) * _norm_cdf(-d2) - spot * math.exp(-q * t) * _norm_cdf(-d1)


def _bs_greeks(spot: float, strike: float, t: float, rate: float, q: float, sigma: float, option_type: str) -> Dict[str, float]:
    if spot <= 0 or strike <= 0 or t <= 0 or sigma <= 0:
        return {"delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan}
    d1 = (math.log(spot / strike) + (rate - q + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    pdf = _norm_pdf(d1)
    sign = 1.0 if option_type == "CE" else -1.0
    delta = math.exp(-q * t) * _norm_cdf(d1) if option_type == "CE" else math.exp(-q * t) * (_norm_cdf(d1) - 1.0)
    gamma = math.exp(-q * t) * pdf / (spot * sigma * math.sqrt(t))
    vega = spot * math.exp(-q * t) * pdf * math.sqrt(t)
    theta = (
        -(spot * pdf * sigma * math.exp(-q * t)) / (2.0 * math.sqrt(t))
        - sign * rate * strike * math.exp(-rate * t) * _norm_cdf(sign * d2)
        + sign * q * spot * math.exp(-q * t) * _norm_cdf(sign * d1)
    ) / 365.0
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def _implied_vol(price: float, spot: float, strike: float, t: float, rate: float, q: float, option_type: str) -> float:
    if any(v is None for v in [price, spot, strike, t]) or price <= 0 or spot <= 0 or strike <= 0 or t <= 0:
        return np.nan
    low = 1e-4
    high = 5.0
    target = float(price)
    for _ in range(80):
        mid = 0.5 * (low + high)
        guess = _bs_price(spot, strike, t, rate, q, mid, option_type)
        if abs(guess - target) < 1e-5:
            return mid
        if guess > target:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def main() -> None:
    args = parse_args()
    option_df = load_dataset(Path(args.option_dataset)).copy()
    spot_df = load_dataset(Path(args.spot_dataset)).copy()
    option_df["timestamp"] = pd.to_datetime(option_df["timestamp"], errors="coerce")
    spot_df["timestamp"] = pd.to_datetime(spot_df["timestamp"], errors="coerce")
    option_df = option_df.sort_values("timestamp", kind="stable")
    spot_df = spot_df.sort_values("timestamp", kind="stable")
    if "close" not in spot_df.columns:
        raise RuntimeError("Spot dataset must contain a close column.")
    merged = pd.merge_asof(
        option_df,
        spot_df[["timestamp", "close"]].rename(columns={"close": "spot_close"}),
        on="timestamp",
        direction="backward",
    )
    merged["expiry"] = pd.to_datetime(merged["expiry"], errors="coerce")
    merged["strike"] = pd.to_numeric(merged.get("strike", merged.get("strike_price")), errors="coerce")
    merged["close"] = pd.to_numeric(merged.get("close", merged.get("ltp")), errors="coerce")
    merged["spot_close"] = pd.to_numeric(merged["spot_close"], errors="coerce")
    merged["option_type"] = merged["option_type"].astype(str).str.upper().replace({"CALL": "CE", "PUT": "PE"})
    merged["time_to_expiry_years"] = ((merged["expiry"] - merged["timestamp"]).dt.total_seconds() / (365.0 * 24.0 * 3600.0)).clip(lower=1.0 / (365.0 * 24.0 * 60.0))
    merged["estimated_ctx_iv"] = merged.apply(
        lambda row: _implied_vol(
            float(row["close"]) if pd.notna(row["close"]) else np.nan,
            float(row["spot_close"]) if pd.notna(row["spot_close"]) else np.nan,
            float(row["strike"]) if pd.notna(row["strike"]) else np.nan,
            float(row["time_to_expiry_years"]) if pd.notna(row["time_to_expiry_years"]) else np.nan,
            float(args.risk_free_rate),
            float(args.dividend_yield),
            str(row["option_type"]),
        ),
        axis=1,
    )
    greek_rows = merged.apply(
        lambda row: _bs_greeks(
            float(row["spot_close"]) if pd.notna(row["spot_close"]) else np.nan,
            float(row["strike"]) if pd.notna(row["strike"]) else np.nan,
            float(row["time_to_expiry_years"]) if pd.notna(row["time_to_expiry_years"]) else np.nan,
            float(args.risk_free_rate),
            float(args.dividend_yield),
            float(row["estimated_ctx_iv"]) if pd.notna(row["estimated_ctx_iv"]) else np.nan,
            str(row["option_type"]),
        ),
        axis=1,
        result_type="expand",
    )
    merged["estimated_ctx_delta"] = greek_rows["delta"]
    merged["estimated_ctx_gamma"] = greek_rows["gamma"]
    merged["estimated_ctx_theta"] = greek_rows["theta"]
    merged["estimated_ctx_vega"] = greek_rows["vega"]
    merged["estimated_delta_abs"] = merged["estimated_ctx_delta"].abs()
    merged["estimated_greeks_imbalance"] = merged["estimated_ctx_gamma"] + merged["estimated_ctx_theta"] + merged["estimated_ctx_vega"]
    merged["estimated_theta_to_vega_ratio"] = merged["estimated_ctx_theta"] / merged["estimated_ctx_vega"].replace(0.0, np.nan)
    merged["estimated_gamma_to_theta_ratio"] = merged["estimated_ctx_gamma"] / merged["estimated_ctx_theta"].replace(0.0, np.nan)
    merged["greeks_source"] = "estimated_black_scholes"
    merged["production_adoption_allowed"] = False
    out_cols = list(option_df.columns) + [
        "spot_close",
        "time_to_expiry_years",
        "estimated_ctx_iv",
        "estimated_ctx_delta",
        "estimated_ctx_gamma",
        "estimated_ctx_theta",
        "estimated_ctx_vega",
        "estimated_delta_abs",
        "estimated_greeks_imbalance",
        "estimated_theta_to_vega_ratio",
        "estimated_gamma_to_theta_ratio",
        "greeks_source",
        "production_adoption_allowed",
    ]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        merged[out_cols].to_parquet(output_path, index=False)
    else:
        merged[out_cols].to_csv(output_path, index=False)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"historical_estimated_greeks_research_only_{stamp}.md"
    report_path.write_text(
        "\n".join(
            [
                "# Historical Estimated Greeks Report",
                "",
                "- Source: `estimated_black_scholes`",
                "- Production adoption allowed: `false`",
                "- Historical bid/ask/spread were not reconstructed.",
                "- Estimates are based on option close/LTP and backward asof spot merge only.",
                "- Estimated fields do not satisfy live-schema production compatibility.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
