from __future__ import annotations

import argparse
import json
from pathlib import Path

from option_chain_pipeline_lib import check_live_schema_compatibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check exact live-schema compatibility for a research dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--live-feature-source", default="auto")
    parser.add_argument("--report-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = check_live_schema_compatibility(
        Path(args.dataset),
        live_feature_source=str(args.live_feature_source),
        report_dir=Path(args.report_dir) if args.report_dir else None,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
