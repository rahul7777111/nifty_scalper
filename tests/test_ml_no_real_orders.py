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


def test_paper_mode_never_calls_real_order_and_raises_if_reachable(tmp_path: Path) -> None:
    registry = MLModelRegistry(_manifest(tmp_path))
    engine = MLRuntimeEngine(
        registry=registry,
        risk_manager=MLPaperRiskManager(),
        log_dir=tmp_path / "logs",
        shadow_mode_enabled=True,
        paper_mode_enabled=True,
        real_order_callable=lambda **kwargs: "bad",
    )
    try:
        engine.evaluate_snapshot({"timestamp": datetime(2026, 6, 5, 10, 0), "feature_a": 1.0, "feature_b": 2.0, "option_ltp": 10.0, "option_type": "CE"})
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass

