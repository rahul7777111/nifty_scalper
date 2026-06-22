from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import chart
import ui
from scripts import repair_paper_forward_candidate_artifacts as repair


def test_format_money_uses_rs_prefix() -> None:
    assert ui.format_money(0) == "Rs. 0.00"
    assert ui.format_money(1234.5) == "Rs. 1,234.50"


def test_stale_absolute_artifact_root_converts_to_repo_relative(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    artifact = repo / "artifacts" / "candidates" / "candidate-1"
    artifact.mkdir(parents=True)
    repair.configure_repo_root(repo)

    assert repair.stale_absolute_root(r"C:\Users\rahul\Downloads\nifty_scalper\artifacts\candidates\candidate-1")
    assert repair.repo_rel(artifact) == str(Path("artifacts") / "candidates" / "candidate-1")


def test_apply_artifact_keeps_matched_candidate_enabled(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    artifact_dir = repo / "artifacts" / "candidates" / "candidate-1"
    artifact_dir.mkdir(parents=True)
    model = artifact_dir / "model.pkl"
    model.write_bytes(b"\x80" + b"0" * 1024)
    repair.configure_repo_root(repo)
    artifact = repair.ArtifactInfo(
        folder=artifact_dir,
        candidate_id="candidate-1",
        artifact_id="candidate-1",
        model_path=model,
        feature_order_len=5,
        feature_order_source=str(Path("artifacts") / "candidates" / "candidate-1" / "feature_schema.json"),
    )
    candidate = {"candidate_id": "candidate-1", "enabled": False, "disabled_reason": "ARTIFACT_NOT_FOUND"}

    repair.apply_artifact(candidate, artifact)

    assert candidate["enabled"] is True
    assert "disabled_reason" not in candidate
    assert not Path(candidate["artifact_dir"]).is_absolute()


def test_disable_missing_artifact_candidate() -> None:
    candidate = {"candidate_id": "missing", "enabled": True, "model_path": "x"}
    repair.disable_candidate(candidate)
    assert candidate["enabled"] is False
    assert candidate["disabled_reason"] == "ARTIFACT_NOT_FOUND"
    assert "model_path" not in candidate


class _FakeText:
    def __init__(self, x, y, label="bad"):
        self._pos = (x, y)
        self._visible = True
        self._label = label
        self._transform = object()

    def get_position(self):
        return self._pos

    def get_transform(self):
        return self._transform

    def get_text(self):
        return self._label

    def set_visible(self, value):
        self._visible = value


def test_safe_ax_text_skips_invalid_coordinates() -> None:
    plugin = chart.LiveChartPlugin.__new__(chart.LiveChartPlugin)
    called = {}
    ax = SimpleNamespace(text=lambda *a, **k: called.setdefault("called", True))

    result = chart.LiveChartPlugin.safe_ax_text(plugin, ax, 1e30, 2, "bad")

    assert result is None
    assert "called" not in called


def test_chart_render_fallback_catches_type_error() -> None:
    plugin = chart.LiveChartPlugin.__new__(chart.LiveChartPlugin)
    bad_text = _FakeText(1e30, 2)
    plugin.ax_price = SimpleNamespace(texts=[bad_text], transAxes=object())
    plugin.ax_rsi = SimpleNamespace(texts=[], transAxes=object())
    calls = {"draw": 0}

    def draw():
        calls["draw"] += 1
        if calls["draw"] == 1:
            raise TypeError("_set_transform bad delta")

    plugin.canvas = SimpleNamespace(draw=draw)

    chart.LiveChartPlugin._safe_draw_idle(plugin, fallback_basic=True)

    assert calls["draw"] == 2
    assert bad_text._visible is False
