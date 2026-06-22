from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest

import ui  # noqa: E402
from rf_gui_backtest import find_latest_rf_artifact  # noqa: E402


def _build_bt_app(tmp_path: Path) -> ui.ScalperUI:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable in test environment: {exc}")
    root.withdraw()
    app = object.__new__(ui.ScalperUI)
    app._test_tk_root = root
    app._bt_last_rf_config = ""
    app._bt_last_rf_artifact = ""
    app._bt_thread = None
    app._worker_threads = {}
    app._stop_events = {}
    app._closing = False
    app._ui_closing = False
    app._ui_queue = __import__("queue").Queue()
    app._record_last_action = lambda *_a, **_k: None
    app._runtime_log_exception = lambda *_a, **_k: None
    app._is_ui_alive = lambda *_a, **_k: True
    app.ui_call = lambda fn, *args, **kwargs: fn(*args, **kwargs)
    app._bt_append_log = MagicMock()
    app._bt_discover_rf_test_target = MagicMock(
        return_value={
            "config_path": str(REPO_ROOT / "config" / "rf_random_forest_dynamic_candidate.json"),
            "artifact_path": "",
            "candidate_id": "rf_test_candidate",
            "dataset_hint": "",
            "selected_threshold": 0.4,
            "max_trades_per_day": 1,
            "source": "static_config",
        }
    )
    app._bt_apply_rf_test_target = ui.ScalperUI._bt_apply_rf_test_target.__get__(app, ui.ScalperUI)

    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("timestamp,ltp,symbol\n2026-01-01 09:15:00,100,NIFTY\n", encoding="utf-8")

    app.bt_config_path_var = tk.StringVar(value="")
    app.bt_csv_path_var = tk.StringVar(value=str(csv_path))
    app.bt_retrain_dataset_var = tk.StringVar(value="")
    app.bt_threshold_var = tk.StringVar(value="0.60")
    app.bt_maxday_var = tk.StringVar(value="3")
    app.bt_use_candidate_thresholds_var = tk.BooleanVar(value=True)
    app.bt_selected_only_var = tk.BooleanVar(value=False)
    app.bt_selected_candidates_var = tk.StringVar(value="")
    app.bt_max_rows_var = tk.StringVar(value="")
    app.bt_fast_mode_var = tk.BooleanVar(value=False)
    app.bt_log_text = tk.Text(root)
    return app


def test_bt_ensure_rf_test_config_materializes_wrapper_from_artifact(tmp_path: Path) -> None:
    root = REPO_ROOT / "models" / "rf_gui_retrain_fixed"
    artifact_path, artifact_dir = find_latest_rf_artifact(root)
    if artifact_path is None or artifact_dir is None:
        pytest.skip("RF GUI retrain fixtures not present")

    app = _build_bt_app(tmp_path)
    target = {
        "config_path": "",
        "artifact_path": str(artifact_path),
        "candidate_id": "",
        "dataset_hint": "",
        "selected_threshold": 0.4,
        "max_trades_per_day": 1,
        "source": "session_artifact",
    }
    resolved = app._bt_ensure_rf_test_config(target)
    assert resolved["config_path"]
    assert Path(resolved["config_path"]).is_file()
    assert resolved.get("candidate_id")


def test_on_test_random_forest_starts_backtest_without_crashing(tmp_path: Path, monkeypatch) -> None:
    app = _build_bt_app(tmp_path)
    started = {"called": False}

    def _fake_run() -> None:
        started["called"] = True

    monkeypatch.setattr(app, "_on_run_ml_backtest", _fake_run)
    ui.ScalperUI._on_test_random_forest(app)
    assert started["called"] is True
    assert app.bt_max_rows_var.get() == "100000"
    assert app.bt_fast_mode_var.get() is True


def test_on_test_random_forest_prefers_manual_threshold_over_candidate_threshold(tmp_path: Path, monkeypatch) -> None:
    app = _build_bt_app(tmp_path)
    app.bt_threshold_var.set("0.35")
    app.bt_use_candidate_thresholds_var.set(True)
    started = {"called": False}

    def _fake_run() -> None:
        started["called"] = True

    monkeypatch.setattr(app, "_on_run_ml_backtest", _fake_run)
    ui.ScalperUI._on_test_random_forest(app)

    assert started["called"] is True
    assert app.bt_use_candidate_thresholds_var.get() is False


def test_on_test_random_forest_handles_missing_ui_state_without_crashing(monkeypatch) -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable in test environment: {exc}")
    root.withdraw()
    app = object.__new__(ui.ScalperUI)
    app._test_tk_root = root
    app._bt_discover_rf_test_target = MagicMock(return_value={"error": "unused"})
    shown: dict[str, str] = {}

    def _fake_showerror(title: str, msg: str) -> None:
        shown["title"] = title
        shown["msg"] = msg

    monkeypatch.setattr(ui.messagebox, "showerror", _fake_showerror)
    ui.ScalperUI._on_test_random_forest(app)
    assert shown["title"] == "Test Random Forest"
    assert "RF test controls are not ready yet" in shown["msg"]


def test_on_test_random_forest_stops_cleanly_when_wrapper_materialization_fails(tmp_path: Path, monkeypatch) -> None:
    app = _build_bt_app(tmp_path)
    artifact_path, artifact_dir = find_latest_rf_artifact(REPO_ROOT / "models" / "rf_gui_retrain_fixed")
    if artifact_path is None or artifact_dir is None:
        pytest.skip("RF GUI retrain fixtures not present")

    app._bt_discover_rf_test_target = MagicMock(
        return_value={
            "config_path": "",
            "artifact_path": str(artifact_path),
            "candidate_id": "",
            "dataset_hint": "",
            "selected_threshold": 0.4,
            "max_trades_per_day": 1,
            "source": "session_artifact",
        }
    )
    app._bt_materialize_rf_dynamic_candidate = MagicMock(return_value=(None, None))
    started = {"called": False}
    shown: dict[str, str] = {}

    def _fake_run() -> None:
        started["called"] = True

    def _fake_showerror(title: str, msg: str) -> None:
        shown["title"] = title
        shown["msg"] = msg

    monkeypatch.setattr(app, "_on_run_ml_backtest", _fake_run)
    monkeypatch.setattr(ui.messagebox, "showerror", _fake_showerror)
    ui.ScalperUI._on_test_random_forest(app)
    assert started["called"] is False
    assert shown["title"] == "Test Random Forest"
    assert "Failed to build RF dynamic candidate wrapper" in shown["msg"]


def test_bt_apply_rf_test_target_prefers_artifact_dataset_over_stale_csv(tmp_path: Path) -> None:
    app = _build_bt_app(tmp_path)
    stale_csv = Path(app.bt_csv_path_var.get())
    artifact_dataset = tmp_path / "artifact_dataset.csv"
    artifact_dataset.write_text("timestamp,ltp,symbol\n2026-01-01 09:15:00,101,NIFTY\n", encoding="utf-8")

    target = {
        "config_path": str(REPO_ROOT / "config" / "rf_random_forest_dynamic_candidate.json"),
        "artifact_path": "",
        "candidate_id": "rf_test_candidate",
        "dataset_hint": str(artifact_dataset),
        "selected_threshold": 0.4,
        "max_trades_per_day": 1,
        "source": "static_config",
    }

    ui.ScalperUI._bt_apply_rf_test_target(app, target)
    assert stale_csv != artifact_dataset
    assert Path(app.bt_csv_path_var.get()) == artifact_dataset


def test_bt_apply_rf_test_target_preserves_existing_form_values(tmp_path: Path) -> None:
    app = _build_bt_app(tmp_path)
    app.bt_threshold_var.set("0.60")
    app.bt_maxday_var.set("3")
    app.bt_use_candidate_thresholds_var.set(False)
    app.bt_selected_only_var.set(False)
    app.bt_selected_candidates_var.set("custom_candidate")

    target = {
        "config_path": str(REPO_ROOT / "config" / "rf_random_forest_dynamic_candidate.json"),
        "artifact_path": "",
        "candidate_id": "rf_test_candidate",
        "dataset_hint": "",
        "selected_threshold": 0.4,
        "max_trades_per_day": 1,
        "source": "static_config",
    }

    ui.ScalperUI._bt_apply_rf_test_target(app, target)

    assert app.bt_threshold_var.get() == "0.60"
    assert app.bt_maxday_var.get() == "3"
    assert app.bt_use_candidate_thresholds_var.get() is False
    assert app.bt_selected_only_var.get() is False
    assert app.bt_selected_candidates_var.get() == "custom_candidate"
