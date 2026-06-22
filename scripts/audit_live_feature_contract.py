from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from live_feature_builder import ALL_COMPUTABLE_FEATURES
from ml_feature_contract import ALLOWED_LIVE_FEATURES, CONTRACT_VERSION, create_feature_contract_summary

NON_CONTRACT_HELPERS = {"expiry", "option_type"}


def main() -> None:
    summary = create_feature_contract_summary()
    missing_from_contract = sorted(
        str(name)
        for name in ALL_COMPUTABLE_FEATURES
        if str(name) not in ALLOWED_LIVE_FEATURES and str(name) not in NON_CONTRACT_HELPERS
    )
    payload = {
        "contract_version": CONTRACT_VERSION,
        "allowed_live_feature_count": len(ALLOWED_LIVE_FEATURES),
        "builder_computable_feature_count": len(ALL_COMPUTABLE_FEATURES),
        "builder_features_missing_from_contract": missing_from_contract,
        "contract_summary": summary,
    }
    print(json.dumps(payload, indent=2))
    if missing_from_contract:
        raise SystemExit(
            "Live feature contract audit failed: builder-computable features are missing from the contract."
        )


if __name__ == "__main__":
    main()
