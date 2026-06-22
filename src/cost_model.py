from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return float(default)


@dataclass(frozen=True)
class CostModelAssumptions:
    brokerage_per_side_pct: float = 0.00015
    exchange_charges_pct: float = 0.000035
    taxes_pct: float = 0.00012
    stt_ctt_pct: float = 0.0005
    gst_pct: float = 0.000032
    sebi_charges_pct: float = 0.000001
    stamp_duty_pct: float = 0.00003
    spread_proxy_pct: float = 0.0006
    slippage_proxy_pct: float = 0.0004
    minimum_net_edge_pct: float = 0.0015
    notes: str = (
        "Conservative placeholder round-trip friction assumptions for label generation and offline evaluation only. "
        "These values do not alter live order placement."
    )

    @property
    def round_trip_cost_pct(self) -> float:
        return float(
            2.0 * self.brokerage_per_side_pct
            + 2.0 * self.exchange_charges_pct
            + 2.0 * self.taxes_pct
            + self.stt_ctt_pct
            + self.gst_pct
            + self.sebi_charges_pct
            + self.stamp_duty_pct
            + self.spread_proxy_pct
            + self.slippage_proxy_pct
        )


class CostModel:
    def __init__(self, assumptions: CostModelAssumptions | None = None) -> None:
        self.assumptions = assumptions or CostModel.load_assumptions()

    @staticmethod
    def load_assumptions() -> CostModelAssumptions:
        return CostModelAssumptions(
            brokerage_per_side_pct=_env_float("MSTOCK_LABEL_BROKERAGE_PCT", 0.00015),
            exchange_charges_pct=_env_float("MSTOCK_LABEL_EXCHANGE_CHARGES_PCT", 0.000035),
            taxes_pct=_env_float("MSTOCK_LABEL_TAXES_PCT", 0.00012),
            stt_ctt_pct=_env_float("MSTOCK_LABEL_STT_CTT_PCT", 0.0005),
            gst_pct=_env_float("MSTOCK_LABEL_GST_PCT", 0.000032),
            sebi_charges_pct=_env_float("MSTOCK_LABEL_SEBI_CHARGES_PCT", 0.000001),
            stamp_duty_pct=_env_float("MSTOCK_LABEL_STAMP_DUTY_PCT", 0.00003),
            spread_proxy_pct=_env_float("MSTOCK_LABEL_SPREAD_PROXY_PCT", 0.0006),
            slippage_proxy_pct=_env_float("MSTOCK_LABEL_SLIPPAGE_PROXY_PCT", 0.0004),
            minimum_net_edge_pct=_env_float("MSTOCK_LABEL_MIN_NET_EDGE_PCT", 0.0015),
        )

    def estimate_cost_pct(self, *, extra_spread_pct: float = 0.0, extra_slippage_pct: float = 0.0) -> float:
        return float(
            self.assumptions.round_trip_cost_pct
            + max(0.0, float(extra_spread_pct))
            + max(0.0, float(extra_slippage_pct))
        )

    def net_edge_pct(
        self,
        gross_return_pct: float,
        *,
        extra_spread_pct: float = 0.0,
        extra_slippage_pct: float = 0.0,
    ) -> float:
        return float(gross_return_pct) - self.estimate_cost_pct(
            extra_spread_pct=extra_spread_pct,
            extra_slippage_pct=extra_slippage_pct,
        )

    def trade_is_viable(
        self,
        gross_return_pct: float,
        *,
        extra_spread_pct: float = 0.0,
        extra_slippage_pct: float = 0.0,
        minimum_net_edge_pct: float | None = None,
    ) -> bool:
        min_edge = (
            self.assumptions.minimum_net_edge_pct
            if minimum_net_edge_pct is None
            else max(0.0, float(minimum_net_edge_pct))
        )
        return self.net_edge_pct(
            gross_return_pct,
            extra_spread_pct=extra_spread_pct,
            extra_slippage_pct=extra_slippage_pct,
        ) >= min_edge

    def to_report_dict(self) -> Dict[str, Any]:
        payload = asdict(self.assumptions)
        payload["round_trip_cost_pct"] = self.assumptions.round_trip_cost_pct
        payload["assumptions_source"] = "environment_or_conservative_defaults"
        return payload

    def write_report(self) -> Dict[str, Path]:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = self.to_report_dict()
        json_path = REPORTS_DIR / "cost_model_assumptions.json"
        md_path = REPORTS_DIR / "cost_model_assumptions.md"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        md_lines = [
            "# Cost Model Assumptions",
            "",
            "| Component | Value |",
            "|---|---:|",
        ]
        for key, value in payload.items():
            md_lines.append(f"| {key} | {value} |")
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        return {"json": json_path, "md": md_path}


class OptionChargesCalculator:
    """Broker-style option charge calculator for UI and terminal diagnostics."""

    def __init__(
        self,
        brokerage_per_order: float = 5.0,
        gst_rate: float = 0.18,
        sebi_fee_rate: float = 0.000001,
        nse_exchange_txn_rate: float = 0.0003503,
        bse_exchange_txn_rate: float = 0.000325,
        stt_sell_rate: float = 0.0015,
        stamp_duty_buy_rate: float = 0.00003,
    ) -> None:
        self.brokerage_per_order = max(0.0, float(brokerage_per_order))
        self.gst_rate = max(0.0, float(gst_rate))
        self.sebi_fee_rate = max(0.0, float(sebi_fee_rate))
        self.nse_exchange_txn_rate = max(0.0, float(nse_exchange_txn_rate))
        self.bse_exchange_txn_rate = max(0.0, float(bse_exchange_txn_rate))
        self.stt_sell_rate = max(0.0, float(stt_sell_rate))
        self.stamp_duty_buy_rate = max(0.0, float(stamp_duty_buy_rate))

    @classmethod
    def from_env(cls) -> "OptionChargesCalculator":
        return cls(
            brokerage_per_order=_env_float("MSTOCK_OPTION_BROKERAGE_PER_ORDER", 5.0),
            gst_rate=_env_float("MSTOCK_OPTION_GST_RATE", 0.18),
            sebi_fee_rate=_env_float("MSTOCK_OPTION_SEBI_FEE_RATE", 0.000001),
            nse_exchange_txn_rate=_env_float("MSTOCK_OPTION_NSE_EXCHANGE_TXN_RATE", 0.0003503),
            bse_exchange_txn_rate=_env_float("MSTOCK_OPTION_BSE_EXCHANGE_TXN_RATE", 0.000325),
            stt_sell_rate=_env_float("MSTOCK_OPTION_STT_SELL_RATE", 0.0015),
            stamp_duty_buy_rate=_env_float("MSTOCK_OPTION_STAMP_DUTY_BUY_RATE", 0.00003),
        )

    def calculate_charges(
        self,
        quantity: int | float,
        strike_price: int | float | None,
        premium: int | float,
        *,
        side: str = "SELL",
        exchange: str = "NSE",
    ) -> Dict[str, float]:
        """Calculate option trade charges for one executed leg.

        STT applies to sell-side option premium turnover. ``strike_price`` is
        accepted for compatibility and future broker-specific extensions.
        """

        del strike_price
        qty = abs(float(quantity or 0.0))
        premium_value = max(0.0, float(premium or 0.0))
        turnover = qty * premium_value
        side_u = str(side or "").strip().upper()
        exchange_u = str(exchange or "NSE").strip().upper()
        exchange_rate = self.bse_exchange_txn_rate if exchange_u == "BSE" else self.nse_exchange_txn_rate
        brokerage = self.brokerage_per_order if turnover > 0.0 else 0.0
        sebi_fee = turnover * self.sebi_fee_rate
        exchange_txn = turnover * exchange_rate
        stt = turnover * self.stt_sell_rate if side_u == "SELL" else 0.0
        stamp_duty = turnover * self.stamp_duty_buy_rate if side_u == "BUY" else 0.0
        gst = (brokerage + exchange_txn + sebi_fee) * self.gst_rate
        non_brokerage_charges = stt + stamp_duty + exchange_txn + sebi_fee + gst
        total_charges = brokerage + non_brokerage_charges

        return {
            "turnover": round(turnover, 2),
            "premium_turnover": round(turnover, 2),
            "brokerage": round(brokerage, 2),
            "sebi_fee": round(sebi_fee, 4),
            "sebi_fees": round(sebi_fee, 4),
            "exchange_txn": round(exchange_txn, 2),
            "exchange_charges": round(exchange_txn, 2),
            "exchange_rate": float(exchange_rate),
            "gst": round(gst, 2),
            "stt": round(stt, 2),
            "ctt": round(stt, 2),
            "stamp_duty": round(stamp_duty, 2),
            "non_brokerage_charges": round(non_brokerage_charges, 2),
            "total_charges": round(total_charges, 2),
        }


def estimate_paper_execution_costs(
    execution_price: float,
    quantity: int | float,
    *,  
    slippage_pct: float = 0.001,
    extra_market_impact_pct: float = 0.0,
    apply_brokerage: bool = True,
    cost_model_source: str = "cost_model_assumptions",
    side: str = "BUY",
) -> Dict[str, float]:
    """Estimate paper execution costs for a single leg.

    Parameters
    ----------
    execution_price : float
        The execution price (already adjusted for spread — e.g. ask for buys).
        Spread cost is NOT deducted here because it is embedded in the price.
    quantity : int | float
        Number of units.
    slippage_pct : float
        Additional slippage as a fraction of execution_price. Applied per leg.
        Default 0.001 = 0.1% one-way slippage.
    extra_market_impact_pct : float
        Additional market impact as a fraction of execution_price. 0 disables.
    apply_brokerage : bool
        If True, deduct brokerage + taxes per leg using CostModelAssumptions
        (or ml_execution_costs if cost_model_source="ml_execution_costs").
    cost_model_source : str
        "cost_model_assumptions" (default) or "ml_execution_costs".
    side : str
        "BUY" or "SELL" — affects which charges apply (STT on sell, stamp on buy).

    Returns
    -------
    Dict with keys:
        slippage_cost    : execution_price * slippage_pct * quantity
        market_impact_cost : execution_price * extra_market_impact_pct * quantity
        brokerage_cost   : per-side brokerage (only if apply_brokerage=True)
        stt_cost         : STT/CTT on sell side (only if apply_brokerage=True)
        exchange_cost    : exchange transaction charges
        gst_cost         : GST on brokerage + exchange + sebi
        sebi_cost        : SEBI fees
        stamp_cost       : stamp duty on buy side
        total_brokerage_charges : sum of all brokerage-related costs
        total_cost       : grand total of all cost components

    No double-counting:
        - Spread cost is NOT included here because execution_price already uses
          ask (buy) or bid (sell), so spread is embedded.
        - Slippage is separate from spread.
        - Brokerage/taxes are separate from spread/slippage.
    """
    price = max(0.0, float(execution_price))
    qty = max(0.0, float(quantity))
    turnover = price * qty
    side_u = str(side or "BUY").strip().upper()

    slippage_cost = price * max(0.0, float(slippage_pct)) * qty
    market_impact_cost = price * max(0.0, float(extra_market_impact_pct)) * qty

    brokerage_cost = 0.0
    stt_cost = 0.0
    exchange_cost = 0.0
    gst_cost = 0.0
    sebi_cost = 0.0
    stamp_cost = 0.0

    if apply_brokerage:
        if cost_model_source == "ml_execution_costs":
            # Higher per-trade flat fee model (₹0.02/order ≈ ₹5/lot typical).
            # These values mirror DEFAULT_EXECUTION_COST_CONFIG in ml_execution_costs.py.
            brokerage_per_order = 0.02  # ₹0.02 per lot (₹2 per 100 lots)
            brokerage_cost = float(brokerage_per_order) * max(1.0, qty)  # scale with qty
            stt_cost = turnover * 0.0005   # STT 0.05% on sell turnover
            exchange_cost = turnover * 0.000035
            sebi_cost = turnover * 0.000001
            # GST 18% on (brokerage + exchange_txn + sebi_fee)
            gst_cost = (brokerage_cost + exchange_cost + sebi_cost) * 0.18
            # Stamp duty 0.03% on buy turnover
            stamp_cost = turnover * 0.00003 if side_u == "BUY" else 0.0
        else:
            # Default: use CostModelAssumptions (per-side % model).
            # Brokerage 0.015% per side (already small).
            brokerage_cost = turnover * 0.00015
            stt_cost = turnover * 0.0005 if side_u == "SELL" else 0.0
            exchange_cost = turnover * 0.000035
            sebi_cost = turnover * 0.000001
            gst_cost = (brokerage_cost + exchange_cost + sebi_cost) * 0.000032 / 0.000035 * 0.18
            # approximate: use gst_pct on (brokerage + exchange + sebi)
            gst_cost = (brokerage_cost + exchange_cost + sebi_cost) * 0.18
            stamp_cost = turnover * 0.00003 if side_u == "BUY" else 0.0

    total_brokerage_charges = (
        brokerage_cost + stt_cost + exchange_cost + gst_cost + sebi_cost + stamp_cost
    )
    total_cost = slippage_cost + market_impact_cost + total_brokerage_charges

    return {
        "slippage_cost": round(slippage_cost, 4),
        "market_impact_cost": round(market_impact_cost, 4),
        "brokerage_cost": round(brokerage_cost, 4),
        "stt_cost": round(stt_cost, 4),
        "exchange_cost": round(exchange_cost, 4),
        "gst_cost": round(gst_cost, 4),
        "sebi_cost": round(sebi_cost, 4),
        "stamp_cost": round(stamp_cost, 4),
        "total_brokerage_charges": round(total_brokerage_charges, 4),
        "total_cost": round(total_cost, 4),
        "turnover": round(turnover, 2),
    }


def evaluate_live_execution_friction(contract_order_context: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate whether live top-of-book friction is acceptable for entry routing."""

    ctx = dict(contract_order_context or {})
    client = ctx.get("client")
    symbol = str(ctx.get("symbol") or "").strip()
    exchange = str(ctx.get("exchange") or "").strip() or None
    cost_model = ctx.get("cost_model")
    if not isinstance(cost_model, CostModel):
        cost_model = CostModel()
    enforce_cost_boundary = bool(ctx.get("enforce_cost_boundary", True))

    bid: Optional[float] = None
    ask: Optional[float] = None
    ltp: Optional[float] = None
    if client is not None and symbol and hasattr(client, "get_bid_ask"):
        try:
            raw_bid, raw_ask, raw_ltp = client.get_bid_ask(symbol, exchange_hint=exchange)
            bid = float(raw_bid) if raw_bid is not None else None
            ask = float(raw_ask) if raw_ask is not None else None
            ltp = float(raw_ltp) if raw_ltp is not None else None
        except Exception:
            bid = None
            ask = None
            ltp = None

    if bid is None:
        try:
            bid = float(ctx.get("bid")) if ctx.get("bid") is not None else None
        except Exception:
            bid = None
    if ask is None:
        try:
            ask = float(ctx.get("ask")) if ctx.get("ask") is not None else None
        except Exception:
            ask = None
    if ltp is None:
        try:
            ltp = float(ctx.get("ltp")) if ctx.get("ltp") is not None else None
        except Exception:
            ltp = None

    midpoint: Optional[float] = None
    absolute_spread: Optional[float] = None
    relative_spread_pct = 0.0
    status = "OK"
    reason = "spread_within_training_friction"

    if bid is not None and ask is not None and ask >= bid:
        midpoint = (float(bid) + float(ask)) / 2.0
        absolute_spread = float(ask) - float(bid)
        if midpoint > 0.0:
            relative_spread_pct = absolute_spread / midpoint
    elif ltp is not None and ltp > 0.0:
        midpoint = float(ltp)
        absolute_spread = 0.0
        relative_spread_pct = 0.0
        status = "DEGRADED_QUOTE_FALLBACK"
        reason = "bid_ask_unavailable_using_ltp"
    else:
        status = "REJECT_LIQUIDITY_CEILING"
        reason = "bid_ask_and_ltp_unavailable"

    allowed_cost_pct = float(cost_model.assumptions.round_trip_cost_pct)
    if status != "REJECT_LIQUIDITY_CEILING" and relative_spread_pct > allowed_cost_pct:
        if enforce_cost_boundary:
            status = "REJECT_LIQUIDITY_CEILING"
        else:
            status = "WARN_LIQUIDITY_CEILING"
        reason = "relative_spread_exceeds_training_cost_boundary"

    return {
        "status": str(status),
        "reason": str(reason),
        "symbol": symbol,
        "exchange": exchange,
        "bid": bid,
        "ask": ask,
        "ltp": ltp,
        "midpoint": midpoint,
        "absolute_spread": absolute_spread,
        "relative_spread_pct": float(relative_spread_pct),
        "allowed_cost_pct": allowed_cost_pct,
    }
