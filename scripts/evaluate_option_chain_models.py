from __future__ import annotations

import argparse
import json
from pathlib import Path

from option_chain_pipeline_lib import evaluate_option_chain_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrained option-chain research artifacts.")
    parser.add_argument("--artifact-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_option_chain_artifacts(Path(args.artifact_dir))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
