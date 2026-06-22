from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def test_live_computable_only_excludes_estimated_greeks() -> None:
    variants = retrain.build_feature_variants(["feature_a", "feature_b"], ["feature_a", "feature_b", "bs_iv", "bs_delta"])
    assert "bs_iv" not in variants["live_computable_only"]["feature_names"]
    assert variants["black_scholes_live_computable"]["live_computable_status"] == "BLOCKED_BY_LIVE_FEATURE_GATE"

