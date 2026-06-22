from __future__ import annotations

import sys
from pathlib import Path

from test_ml_deployment_manifest import _manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_model_registry import MLModelRegistry
from ml_runtime import paper_mode_allowed


def test_rejected_model_verdict_allows_shadow_only_blocks_paper(tmp_path: Path) -> None:
    registry = MLModelRegistry(_manifest(tmp_path, verdict="RESEARCH_ONLY_WEAK_EDGE"))
    allowed, reason = paper_mode_allowed(manifest_payload=registry.manifest().payload, registry_health=registry.model_health_status(), schema_ok=True, paper_mode_enabled=True, kill_switch=False)
    assert not allowed


def test_paper_candidate_verdict_allows_paper_when_schema_matches(tmp_path: Path) -> None:
    registry = MLModelRegistry(_manifest(tmp_path, verdict="PAPER_TRADE_CANDIDATE_MEDIUM_CONFIDENCE"))
    allowed, reason = paper_mode_allowed(manifest_payload=registry.manifest().payload, registry_health=registry.model_health_status(), schema_ok=True, paper_mode_enabled=True, kill_switch=False)
    assert allowed

