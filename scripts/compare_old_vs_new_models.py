from __future__ import annotations

import argparse
import json
from pathlib import Path

from option_chain_pipeline_lib import compare_old_vs_new


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the current live model metadata against new option-chain research artifacts.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--old-model-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = compare_old_vs_new(Path(args.artifact_dir), Path(args.old_model_dir) if args.old_model_dir else None)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
