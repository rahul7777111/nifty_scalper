from __future__ import annotations

import argparse
import json
from pathlib import Path

from option_chain_pipeline_lib import build_processed_option_chain_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a normalized research dataset from external NIFTY option-chain history.")
    parser.add_argument("--dataset-folder", required=True)
    parser.add_argument("--horizon-bars", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_processed_option_chain_dataset(Path(args.dataset_folder), horizon_bars=int(args.horizon_bars))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
