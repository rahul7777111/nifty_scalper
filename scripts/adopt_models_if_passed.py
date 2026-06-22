from __future__ import annotations

import argparse
import json
from pathlib import Path

from option_chain_pipeline_lib import adopt_option_chain_models_if_passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adopt option-chain research artifacts only if all gates passed.")
    parser.add_argument("--artifact-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = adopt_option_chain_models_if_passed(Path(args.artifact_dir))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
