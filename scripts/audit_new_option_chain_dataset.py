from __future__ import annotations

import argparse
import json
from pathlib import Path

from option_chain_pipeline_lib import audit_option_chain_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an external NIFTY option-chain dataset folder.")
    parser.add_argument("--dataset-folder", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, json_path, md_path = audit_option_chain_dataset(Path(args.dataset_folder))
    payload["report_json"] = str(json_path)
    payload["report_md"] = str(md_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
