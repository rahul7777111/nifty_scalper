from __future__ import annotations

import pickle
from pathlib import Path

from src.ml.ensemble_auto_router import EnsembleAutoRouter, STATUS_ARTIFACT_NOT_FOUND


class TinyModel:
    classes_ = [0, 1]

    def predict_proba(self, X):
        row = X[0]
        p = 0.8 if float(row[0]) >= 0.5 else 0.2
        return [[1.0 - p, p]]


def _write_model(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(
            {
                "model": TinyModel(),
                "feature_order": ["option_type_ce", "ltp"],
                "label_mapping": {"0": "no_trade", "1": "favorable_trade"},
                "target_name": "unit_test_favorable_trade",
            },
            fh,
        )


def _base_config(tmp_path: Path, artifact_root: Path):
    return {
        "enabled": True,
        "artifact_search_paths": [str(artifact_root)],
        "models": {
            "elasticnet": {"enabled": True, "label": "ElasticNet", "artifact_path": str(tmp_path / "missing_elastic"), "weight": 0.30},
            "logistic_regression": {"enabled": True, "label": "Logistic Regression", "artifact_path": str(tmp_path / "missing_lr"), "weight": 0.20},
            "calibrated_logistic_regression": {"enabled": True, "label": "Calibrated Logistic Regression", "artifact_path": str(tmp_path / "missing_cal"), "weight": 0.25},
            "xgboost": {"enabled": True, "label": "XGBoost", "artifact_path": str(tmp_path / "missing_xgb"), "weight": 0.25},
        },
        "thresholds": {
            "ensemble_min_valid_models": 2,
            "ensemble_min_confidence": 0.55,
            "ensemble_min_direction_edge": 0.20,
            "ensemble_max_model_disagreement": 0.35,
        },
        "liquidity": {"max_spread_pct": 0.2, "min_ltp": 0.01, "min_volume": 0, "min_oi": 0},
    }


def test_recursive_artifact_discovery_loads_four_model_families(tmp_path: Path):
    root = tmp_path / "artifacts"
    _write_model(root / "candidates" / "ANY_elasticnet_model" / "not_exact_name.pkl")
    _write_model(root / "models" / "foo_logistic_regression_bundle.joblib")
    _write_model(root / "ml_signals" / "bar_platt_calibrated_model.pickle")
    _write_model(root / "candidates" / "baz_xgb_candidate" / "random_file.pkl")

    router = EnsembleAutoRouter(_base_config(tmp_path, root))
    statuses = {row.model_key: row for row in router.model_status()}

    assert len(statuses) == 4
    assert all(row.loaded for row in statuses.values())
    assert statuses["elasticnet"].artifact_path.endswith("not_exact_name.pkl")
    assert statuses["logistic_regression"].artifact_path.endswith("foo_logistic_regression_bundle.joblib")
    assert statuses["calibrated_logistic_regression"].artifact_path.endswith("bar_platt_calibrated_model.pickle")
    assert statuses["xgboost"].artifact_path.endswith("random_file.pkl")


def test_missing_artifacts_show_four_artifact_not_found_rows(tmp_path: Path):
    router = EnsembleAutoRouter(_base_config(tmp_path, tmp_path / "empty_artifacts"))
    rows = router.model_status()
    display_rows = router.model_vote_rows()

    assert len(rows) == 4
    assert len(display_rows) == 4
    assert all(row.loaded is False for row in rows)
    assert all(row.status == STATUS_ARTIFACT_NOT_FOUND for row in rows)
    assert all(row.vote == "NO_VOTE" for row in rows)
    assert all("no matching artifact found" in row.error for row in rows)
    assert all(row["confidence"] is None for row in display_rows)
    assert all("searched paths" in row["error"] for row in display_rows)


def test_fewer_than_two_loaded_models_blocks_trade(tmp_path: Path):
    root = tmp_path / "artifacts"
    _write_model(root / "candidates" / "only_elasticnet" / "model.pkl")
    router = EnsembleAutoRouter(_base_config(tmp_path, root))

    result = router.decide({"ltp": 100.0, "bid": 99.0, "ask": 101.0, "option_type_ce": 1.0, "option_type_pe": 0.0})

    assert result.decision == "NO_TRADE"
    assert result.block_reason == "FEWER_THAN_MIN_VALID_MODELS"
