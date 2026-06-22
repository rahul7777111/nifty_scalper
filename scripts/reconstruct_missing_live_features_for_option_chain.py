from __future__ import annotations

import argparse
import json
from pathlib import Path

from option_chain_pipeline_lib import reconstruct_missing_live_features_for_option_chain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct safely available live features on the enriched option-chain dataset.")
    parser.add_argument("--input-dataset", required=True)
    parser.add_argument("--output-dataset", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--schema-gap-report", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = reconstruct_missing_live_features_for_option_chain(
        Path(args.input_dataset),
        output_dataset=Path(args.output_dataset) if args.output_dataset else None,
        report_path=Path(args.report) if args.report else None,
        schema_gap_report_path=Path(args.schema_gap_report) if args.schema_gap_report else None,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
