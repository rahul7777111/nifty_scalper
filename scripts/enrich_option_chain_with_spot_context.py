from __future__ import annotations

import argparse
import json
from pathlib import Path

from option_chain_pipeline_lib import enrich_option_chain_with_spot_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich processed option-chain rows with NIFTY spot/index context.")
    parser.add_argument("--option-dataset", required=True)
    parser.add_argument("--spot-dataset", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--report", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = enrich_option_chain_with_spot_context(
        Path(args.option_dataset),
        Path(args.spot_dataset),
        output=Path(args.output) if args.output else None,
        report=Path(args.report) if args.report else None,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
