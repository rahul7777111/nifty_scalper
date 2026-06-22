from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
for path in (SCRIPTS_DIR, SRC_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import retrain_all_edge_models as retrain
from ml_feature_contract import ALLOWED_LIVE_FEATURES


def test_live_computable_only_filters_to_dataset_contract_intersection() -> None:
    feature_names = [
        "last_open",
        "range_pct",
        "mean_reversion_zscore",
        "bs_iv",
        "totally_unknown_feature",
    ]

    selected, audit = retrain._filter_live_contract_features(
        feature_names,
        dataset_columns=feature_names,
        strict=False,
    )

    assert selected == [name for name in feature_names if name in ALLOWED_LIVE_FEATURES]
    excluded = {row["feature_name"] for row in audit if row["feature_name"] not in selected}
    assert {"bs_iv", "totally_unknown_feature"} <= excluded


def test_strict_live_contract_fails_on_non_contract_candidate_features() -> None:
    with pytest.raises(RuntimeError, match="STRICT_LIVE_CONTRACT_VIOLATION"):
        retrain._filter_live_contract_features(
            ["last_open", "unknown_feature"],
            dataset_columns=["last_open", "unknown_feature"],
            strict=True,
        )
