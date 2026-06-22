from __future__ import annotations

import argparse
import json
from pathlib import Path

from option_chain_pipeline_lib import validate_option_context_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate future option-context data for IV/Greeks/bid-ask live-schema readiness.")
    parser.add_argument("--dataset", required=True, help="Path to option-context CSV or parquet dataset.")
    parser.add_argument("--report-dir", default="")
    parser.add_argument("--max-iv", type=float, default=5.0)
    parser.add_argument("--min-iv", type=float, default=0.0001)
    parser.add_argument("--spread-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-spread-pct", type=float, default=0.25)
    parser.add_argument("--min-near-atm-rows-per-day", type=int, default=1)
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = validate_option_context_dataset(
        Path(args.dataset),
        report_dir=Path(args.report_dir) if args.report_dir else None,
        max_iv=float(args.max_iv),
        min_iv=float(args.min_iv),
        spread_tolerance=float(args.spread_tolerance),
        max_spread_pct=float(args.max_spread_pct),
        min_near_atm_rows_per_day=int(args.min_near_atm_rows_per_day),
        fail_on_warning=bool(args.fail_on_warning),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
