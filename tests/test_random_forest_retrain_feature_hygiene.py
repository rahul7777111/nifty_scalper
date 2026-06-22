from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
for path in (SCRIPTS_DIR, SRC_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import retrain_all_edge_models as retrain
from live_feature_builder import LiveFeatureBuilder, LiveSnapshot
from ml_feature_contract import ALLOWED_LIVE_FEATURES


def test_hygiene_passes_after_live_contract_filter() -> None:
    candidate_features = [
        "last_open",
        "range_pct",
        "trend_following_strength",
        "mean_reversion_zscore",
        "stat_arb_confidence",
        "bs_iv",
    ]
    selected, _ = retrain._filter_live_contract_features(candidate_features, strict=False)
    result = retrain._check_training_features_hygiene(selected, label_name="profitable_trade_label")

    assert result["hygiene_passed"] is True
    assert "bs_iv" not in selected
    assert all(name in ALLOWED_LIVE_FEATURES for name in selected)


def test_save_model_artifact_persists_feature_order_and_live_contract_metadata() -> None:
    out_dir = REPO_ROOT / "C_tmp_test_artifact"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = retrain.save_model_artifact(
        out_dir,
        "random_forest",
        "profitable_trade_label",
        {"kind": "dummy"},
        ["last_open", "range_pct", "mean_reversion_zscore"],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        {
            "feature_order": ["last_open", "range_pct", "mean_reversion_zscore"],
            "live_computable_only": True,
            "live_contract_version": "test",
            "excluded_non_live_features": ["bs_iv"],
            "target_column": "profitable_trade_label",
            "model_name": "random_forest",
            "dataset_path": "dataset.csv",
            "train_test_date_split": {"train_start": "2024-01-01", "test_end": "2024-06-01"},
            "metrics": {"roc_auc": 0.5},
        },
    )

    metadata_path = Path(model_path).with_name("random_forest_profitable_trade_label_metadata.json")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["feature_order"] == ["last_open", "range_pct", "mean_reversion_zscore"]
    assert payload["live_computable_only"] is True
    assert payload["excluded_non_live_features"] == ["bs_iv"]


def test_live_builder_exposes_direct_strategy_features() -> None:
    class Candle:
        def __init__(self, i: int) -> None:
            self.open = 100.0 + i
            self.high = 101.0 + i
            self.low = 99.0 + i
            self.close = 100.5 + i
            self.volume = 1000.0 + i
            self.time = datetime(2026, 6, 19, 9, 15) + timedelta(minutes=5 * i)

    snapshot = LiveSnapshot(candles=[Candle(i) for i in range(40)], spot=120.0, atm_iv=0.2)
    builder = LiveFeatureBuilder(
        required_features=[
            "trend_following_confidence",
            "mean_reversion_zscore",
            "stat_arb_confidence",
            "range_pct",
            "ret_1",
        ]
    )
    result = builder.build(snapshot)

    assert result.features["trend_following_confidence"] is not None
    assert result.features["mean_reversion_zscore"] is not None
    assert result.features["stat_arb_confidence"] is not None
    assert result.features["range_pct"] is not None
    assert result.features["ret_1"] is not None


def test_random_forest_validation_search_returns_selected_params() -> None:
    X_train = np.array(
        [[0.0], [0.1], [0.2], [0.3], [0.8], [0.9], [1.0], [1.1]],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    X_val = np.array([[0.15], [0.25], [0.85], [0.95]], dtype=np.float32)
    y_val = np.array([0, 0, 1, 1], dtype=int)
    val_returns = np.array([-0.01, -0.01, 0.02, 0.03], dtype=float)

    model, meta = retrain._fit_random_forest_with_validation_search(
        X_train,
        y_train,
        X_val,
        y_val,
        val_returns,
        minimum_trades=1,
    )

    assert hasattr(model, "predict_proba")
    assert meta["mode"] == "validation_search"
    assert meta["fit_on_validation_only"] is False
    assert meta["runtime_config"]["stage1_trees"] >= 50
    assert meta["selected_params"]
    assert meta["selected_threshold_preview"]["threshold"] in retrain.RETRAIN_THRESHOLD_SWEEP
    assert meta["stage1_candidate_count"] == len(retrain._random_forest_search_space())
    assert meta["stage2_candidate_count"] >= 1
    assert len(meta["search_rows"]) == meta["stage1_candidate_count"] + meta["stage2_candidate_count"] + 1
    tuned_keys = {"criterion", "max_samples", "class_weight", "ccp_alpha", "max_leaf_nodes"}
    assert tuned_keys.issubset(set(meta["selected_params"]))


def test_random_forest_search_space_covers_requested_parameters() -> None:
    space = retrain._random_forest_search_space()

    assert any(row["criterion"] == "entropy" for row in space)
    assert any(row["criterion"] == "log_loss" for row in space)
    assert any(row["max_samples"] is None for row in space)
    assert any(row["max_samples"] == 0.70 for row in space)
    assert any(row["class_weight"] == "balanced" for row in space)
    assert any(float(row["ccp_alpha"]) > 0.0 for row in space)
    assert any(int(row["max_leaf_nodes"]) >= 96 for row in space)


def test_random_forest_runtime_config_respects_environment(monkeypatch) -> None:
    monkeypatch.setenv("RF_SEARCH_STAGE1_TREES", "120")
    monkeypatch.setenv("RF_SEARCH_STAGE2_TREES", "240")
    monkeypatch.setenv("RF_FINAL_TREES", "360")
    monkeypatch.setenv("RF_STAGE1_TOP_K", "2")
    monkeypatch.setenv("RF_N_JOBS", "6")
    monkeypatch.setenv("RF_ADAPTIVE_MARGIN_BPS", "30")

    cfg = retrain._rf_runtime_config()

    assert cfg["stage1_trees"] == 120
    assert cfg["stage2_trees"] == 240
    assert cfg["final_trees"] == 360
    assert cfg["stage1_top_k"] == 2
    assert cfg["n_jobs"] == 6
    assert cfg["adaptive_margin_bps"] == 30
