from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def test_reduced_robust_features_exclude_forbidden_and_non_live() -> None:
    features = [
        "spot_close",
        "option_volume",
        "distance_from_atm",
        "regime_trending",
        "bs_delta",
        "net_forward_return",
        "trade_pnl",
    ]
    selected = retrain._reduced_robust_features(features)
    assert "spot_close" in selected
    assert "option_volume" in selected
    assert "distance_from_atm" in selected
    assert "regime_trending" in selected
    assert "bs_delta" not in selected
    assert "net_forward_return" not in selected
    assert "trade_pnl" not in selected
