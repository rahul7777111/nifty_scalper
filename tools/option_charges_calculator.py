from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cost_model import OptionChargesCalculator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate option leg charges.")
    parser.add_argument("--quantity", type=float, required=True, help="Executed quantity.")
    parser.add_argument("--strike-price", type=float, default=0.0, help="Option strike price.")
    parser.add_argument("--premium", type=float, required=True, help="Executed or live option premium.")
    parser.add_argument("--side", choices=["BUY", "SELL"], default="SELL", help="Leg side; CTT applies to SELL.")
    parser.add_argument("--exchange", choices=["NSE", "BSE"], default="NSE", help="Exchange fee schedule.")
    parser.add_argument("--brokerage-per-order", type=float, default=5.0, help="Flat brokerage per executed order.")
    parser.add_argument("--gst-rate", type=float, default=0.18, help="GST rate on brokerage plus exchange transaction charges.")
    parser.add_argument("--sebi-fee-rate", type=float, default=0.000001, help="SEBI fee rate on turnover.")
    parser.add_argument("--nse-exchange-txn-rate", type=float, default=0.0003503, help="NSE option exchange transaction charge rate.")
    parser.add_argument("--bse-exchange-txn-rate", type=float, default=0.000325, help="BSE option exchange transaction charge rate.")
    parser.add_argument("--stt-sell-rate", type=float, default=0.0015, help="STT rate on sell-side option premium turnover.")
    parser.add_argument("--stamp-duty-buy-rate", type=float, default=0.00003, help="Stamp duty rate on buy-side option premium turnover.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calc = OptionChargesCalculator(
        brokerage_per_order=args.brokerage_per_order,
        gst_rate=args.gst_rate,
        sebi_fee_rate=args.sebi_fee_rate,
        nse_exchange_txn_rate=args.nse_exchange_txn_rate,
        bse_exchange_txn_rate=args.bse_exchange_txn_rate,
        stt_sell_rate=args.stt_sell_rate,
        stamp_duty_buy_rate=args.stamp_duty_buy_rate,
    )
    summary = calc.calculate_charges(
        quantity=args.quantity,
        strike_price=args.strike_price,
        premium=args.premium,
        side=args.side,
        exchange=args.exchange,
    )
    print(json.dumps(summary, indent=2))
    print(f"Total Non-Brokerage Charges: Rs {summary['non_brokerage_charges']:.2f}")
    print(f"Total Trade Charges incl brokerage: Rs {summary['total_charges']:.2f}")


if __name__ == "__main__":
    main()
