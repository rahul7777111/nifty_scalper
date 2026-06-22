from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from test_ml_deployment_manifest import _manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_model_registry import MLModelRegistry
from ml_paper_risk_manager import MLPaperRiskManager
from ml_runtime import MLRuntimeEngine


def test_shadow_mode_logs_prediction_but_sends_no_order(tmp_path: Path) -> None:
    registry = MLModelRegistry(_manifest(tmp_path))
    engine = MLRuntimeEngine(
        registry=registry,
        risk_manager=MLPaperRiskManager(),
        log_dir=tmp_path / "logs",
        shadow_mode_enabled=True,
        paper_mode_enabled=False,
    )
    row = engine.evaluate_snapshot({"timestamp": datetime(2026, 6, 5, 10, 0), "feature_a": 1.0, "feature_b": 2.0, "option_ltp": 10.0, "option_type": "CE"})
    assert row["production_order_sent"] is False
    assert (tmp_path / "logs").exists()

