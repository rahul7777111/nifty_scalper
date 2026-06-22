from __future__ import annotations

import json
import pickle
from pathlib import Path

from src.ml.ensemble_auto_router import EnsembleAutoRouter, EnsemblePaperTrader, selected_option_state_row


class GoodModel:
    classes_ = [0, 1]

    def predict_proba(self, X):
        p = 0.9 if float(X[0][0]) >= 0.5 else 0.2
        return [[1.0 - p, p]]


class FailingModel:
    classes_ = [0, 1]

    def predict_proba(self, X):
        raise RuntimeError("predict exploded")


def _write_bundle(path: Path, model) -> str:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "model.pkl").open("wb") as fh:
        pickle.dump(
            {
                "model": model,
                "feature_order": ["option_type_ce", "ltp"],
                "label_mapping": {"0": "no_trade", "1": "favorable_trade"},
                "target_name": "unit_test_favorable_trade",
            },
            fh,
        )
    return str(path)


def _config(models: dict, *, enabled: bool = False, search_path: str | None = None):
    return {
        "enabled": enabled,
        "paper_mode_enabled": True,
        "artifact_search_paths": [search_path] if search_path else [],
        "models": models,
        "thresholds": {
            "ensemble_min_valid_models": 2,
            "ensemble_min_confidence": 0.55,
            "ensemble_min_direction_edge": 0.20,
            "ensemble_max_model_disagreement": 0.35,
        },
        "liquidity": {"max_spread_pct": 0.2, "min_ltp": 0.01, "min_volume": 0, "min_oi": 0},
        "duplicate_trade_rules": {"max_trades_per_day": 3, "one_position_per_side": True},
        "paper_trading": {"lot_size": 65, "brokerage_per_order": 0.0, "slippage_pct": 0.0, "cost_bps": 0.0},
        "exits": {"entry_cutoff_time": "23:59", "force_exit_time": "23:59", "daily_loss_limit": -999999.0},
    }


def _spec(path: str, weight: float):
    return {"enabled": True, "artifact_path": path, "weight": weight}


def test_config_enabled_loads_as_enabled(tmp_path: Path):
    d1 = _write_bundle(tmp_path / "elasticnet", GoodModel())
    d2 = _write_bundle(tmp_path / "logistic", GoodModel())
    cfg = _config({"elasticnet": _spec(d1, 0.5), "logistic_regression": _spec(d2, 0.5)}, enabled=True)

    router = EnsembleAutoRouter(cfg)

    assert router.enabled is True


def test_start_paper_mode_enables_runtime_router(tmp_path: Path):
    d1 = _write_bundle(tmp_path / "elasticnet", GoodModel())
    d2 = _write_bundle(tmp_path / "logistic", GoodModel())
    router = EnsembleAutoRouter(_config({"elasticnet": _spec(d1, 0.5), "logistic_regression": _spec(d2, 0.5)}, enabled=False))
    paper = EnsemblePaperTrader(router)

    assert router.enabled is False
    router.set_runtime_enabled(True)
    paper.start()

    assert router.enabled is True
    assert paper.paper_running is True


def test_paper_mode_scores_models_before_entry_cutoff_block(tmp_path: Path):
    d1 = _write_bundle(tmp_path / "elasticnet", GoodModel())
    d2 = _write_bundle(tmp_path / "logistic", GoodModel())
    cfg = _config(
        {"elasticnet": _spec(d1, 0.5), "logistic_regression": _spec(d2, 0.5)},
        enabled=True,
    )
    cfg["exits"]["entry_cutoff_time"] = "00:00"
    router = EnsembleAutoRouter(cfg)
    paper = EnsemblePaperTrader(router)
    paper.start()

    result = paper.on_snapshot({"ltp": 100.0, "bid": 99.0, "ask": 101.0})

    assert result.decision == "NO_TRADE"
    assert result.block_reason == "ENTRY_CUTOFF_TIME"
    assert result.valid_model_count == 2
    assert all(row.confidence is not None for row in result.model_outputs if row.valid)


def test_model_table_has_rows_even_if_artifacts_missing(tmp_path: Path):
    router = EnsembleAutoRouter(
        _config(
            {
                "elasticnet": _spec(str(tmp_path / "missing1"), 0.3),
                "logistic_regression": _spec(str(tmp_path / "missing2"), 0.2),
                "calibrated_logistic_regression": _spec(str(tmp_path / "missing3"), 0.25),
                "xgboost": _spec(str(tmp_path / "missing4"), 0.25),
            },
            enabled=True,
        )
    )

    rows = router.model_vote_rows()

    assert len(rows) == 4
    assert {row["status"] for row in rows} == {"ARTIFACT_NOT_FOUND"}
    assert {row["vote"] for row in rows} == {"NO_VOTE"}
    assert all(row["confidence"] is None for row in rows)


def test_reload_models_refreshes_status_from_config_path(tmp_path: Path):
    config_path = tmp_path / "ensemble.json"
    missing_cfg = _config({"elasticnet": _spec(str(tmp_path / "missing"), 1.0)}, enabled=True)
    config_path.write_text(json.dumps(missing_cfg), encoding="utf-8")
    router = EnsembleAutoRouter(config_path=config_path)
    assert router.model_status()[0].status == "ARTIFACT_NOT_FOUND"

    good = _write_bundle(tmp_path / "elasticnet", GoodModel())
    loaded_cfg = _config({"elasticnet": _spec(good, 1.0)}, enabled=True)
    config_path.write_text(json.dumps(loaded_cfg), encoding="utf-8")
    router.reload()

    assert router.model_status()[0].status == "OK"
    assert router.model_status()[0].loaded is True
    assert len(router.model_vote_rows()) == 4


def test_failed_prediction_does_not_become_fake_zero_confidence(tmp_path: Path):
    bad = _write_bundle(tmp_path / "elasticnet", FailingModel())
    good = _write_bundle(tmp_path / "logistic", GoodModel())
    router = EnsembleAutoRouter(_config({"elasticnet": _spec(bad, 0.5), "logistic_regression": _spec(good, 0.5)}, enabled=True))

    result = router.decide({"ltp": 100.0, "bid": 99.0, "ask": 101.0})
    bad_row = next(row for row in result.model_outputs if row.model_key == "elasticnet")

    assert bad_row.status == "PREDICT_FAILED"
    assert bad_row.confidence is None
    assert result.block_reason == "FEWER_THAN_MIN_VALID_MODELS"
    display_row = next(row for row in router.model_vote_rows(result) if row["model_key"] == "elasticnet")
    assert display_row["confidence"] is None


def test_selected_option_state_has_placeholder_when_no_position():
    row = selected_option_state_row(realized_pnl=12.5, unrealized_pnl=-3.25)

    assert row == ("No open paper position", "-", "-", "-", "-", "-", "-", "12.50", "-3.25")


def test_no_artifact_state_has_four_rows_on_startup(tmp_path: Path):
    router = EnsembleAutoRouter(
        _config(
            {
                "elasticnet": _spec(str(tmp_path / "missing1"), 0.3),
                "logistic_regression": _spec(str(tmp_path / "missing2"), 0.2),
                "calibrated_logistic_regression": _spec(str(tmp_path / "missing3"), 0.25),
                "xgboost": _spec(str(tmp_path / "missing4"), 0.25),
            },
            enabled=True,
        )
    )

    rows = router.model_vote_rows()

    assert len(rows) == 4
    assert [row["model_key"] for row in rows] == [
        "elasticnet",
        "logistic_regression",
        "calibrated_logistic_regression",
        "xgboost",
    ]
