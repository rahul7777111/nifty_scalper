from __future__ import annotations

import pickle
from pathlib import Path

from src.ml.ensemble_auto_router import (
    DECISION_BUY_CE,
    DECISION_NO_TRADE,
    STATUS_ARTIFACT_NOT_FOUND,
    STATUS_FEATURES_MISSING,
    STATUS_UNKNOWN_LABEL_MAPPING,
    EnsembleAutoRouter,
    EnsemblePaperTrader,
)


class SideSensitiveModel:
    classes_ = [0, 1]

    def __init__(self, ce_prob: float, pe_prob: float) -> None:
        self.ce_prob = float(ce_prob)
        self.pe_prob = float(pe_prob)

    def predict_proba(self, X):
        row = X[0]
        is_ce = float(row[0]) >= 0.5
        p = self.ce_prob if is_ce else self.pe_prob
        return [[1.0 - p, p]]


def _write_artifact(tmp_path: Path, name: str, model: SideSensitiveModel, features=None) -> str:
    d = tmp_path / name
    d.mkdir()
    with (d / "model.pkl").open("wb") as fh:
        pickle.dump(
            {
                "model": model,
                "feature_order": features or ["option_type_ce", "ltp"],
                "label_mapping": {"0": "no_trade", "1": "favorable_trade"},
                "target_name": "unit_test_favorable_trade",
            },
            fh,
        )
    return str(d)


def _config(tmp_path: Path, models: dict, **overrides):
    cfg = {
        "enabled": True,
        "artifact_search_paths": [str(tmp_path / "artifacts")],
        "models": models,
        "thresholds": {
            "ensemble_min_confidence": 0.55,
            "ensemble_min_direction_edge": 0.2,
            "ensemble_max_model_disagreement": 0.35,
            "ensemble_min_valid_models": 2,
        },
        "liquidity": {"max_spread_pct": 0.20, "min_volume": 0, "min_oi": 0, "min_ltp": 0.01},
        "duplicate_trade_rules": {"max_trades_per_day": 3, "one_position_per_side": True},
        "paper_trading": {
            "lot_size": 65,
            "brokerage_per_order": 0.0,
            "slippage_pct": 0.0,
            "cost_bps": 0.0,
            "trade_log_path": str(tmp_path / "trades.csv"),
            "summary_path": str(tmp_path / "summary.json"),
        },
        "exits": {
            "stop_loss_pct": 0.10,
            "target_pct": 0.20,
            "max_hold_bars": 5,
            "force_exit_time": "23:59",
            "entry_cutoff_time": "23:59",
            "daily_loss_limit": -999999.0,
        },
    }
    cfg.update(overrides)
    return cfg


def _model_spec(path: str, weight: float = 0.5):
    return {"enabled": True, "label": Path(path).name, "artifact_path": path, "weight": weight}


def _snapshot(ltp: float = 100.0, *, spread: float = 1.0):
    return {
        "ltp": ltp,
        "bid": ltp - spread,
        "ask": ltp + spread,
        "volume": 1000,
        "oi": 1000,
        "symbol": "NIFTYCE",
    }


def test_missing_model_artifact_does_not_crash(tmp_path: Path):
    good = _write_artifact(tmp_path, "good", SideSensitiveModel(0.9, 0.2))
    missing = str(tmp_path / "missing")
    router = EnsembleAutoRouter(_config(tmp_path, {"good": _model_spec(good), "missing": _model_spec(missing)}))

    statuses = {m.model_key: m.status for m in router.model_status()}
    assert statuses["missing"] == STATUS_ARTIFACT_NOT_FOUND
    result = router.decide(_snapshot())
    assert result.decision == DECISION_NO_TRADE
    assert result.block_reason == "FEWER_THAN_MIN_VALID_MODELS"


def test_missing_feature_does_not_become_zero_confidence_silently(tmp_path: Path):
    a = _write_artifact(tmp_path, "a", SideSensitiveModel(0.9, 0.2), ["option_type_ce", "missing_feature"])
    b = _write_artifact(tmp_path, "b", SideSensitiveModel(0.8, 0.2))
    router = EnsembleAutoRouter(_config(tmp_path, {"a": _model_spec(a), "b": _model_spec(b)}))

    result = router.decide(_snapshot())
    bad = [m for m in result.model_outputs if m.model_key == "a"][0]
    assert bad.status == STATUS_FEATURES_MISSING
    assert bad.confidence is None
    assert "missing_feature" in bad.missing_features


def test_valid_models_renormalize_weights(tmp_path: Path):
    a = _write_artifact(tmp_path, "a", SideSensitiveModel(0.9, 0.2))
    b = _write_artifact(tmp_path, "b", SideSensitiveModel(0.8, 0.2))
    missing = str(tmp_path / "missing")
    router = EnsembleAutoRouter(
        _config(
            tmp_path,
            {
                "a": _model_spec(a, 0.3),
                "b": _model_spec(b, 0.2),
                "missing": _model_spec(missing, 0.5),
            },
        )
    )

    result = router.decide(_snapshot())
    assert result.decision == DECISION_BUY_CE
    weights = {m.model_key: m.normalized_weight for m in result.model_outputs if m.valid}
    assert round(weights["a"], 3) == 0.600
    assert round(weights["b"], 3) == 0.400


def test_binary_scoring_uses_ce_pe_probabilities_and_raw_output(tmp_path: Path):
    a = _write_artifact(tmp_path, "a", SideSensitiveModel(0.72, 0.41))
    b = _write_artifact(tmp_path, "b", SideSensitiveModel(0.70, 0.40))
    cfg = _config(
        tmp_path,
        {"a": _model_spec(a, 0.5), "b": _model_spec(b, 0.5)},
        thresholds={
            "ensemble_min_confidence": 0.55,
            "ensemble_min_direction_edge": 0.20,
            "ensemble_max_model_disagreement": 0.35,
            "ensemble_min_valid_models": 2,
        },
    )
    router = EnsembleAutoRouter(cfg)

    result = router.decide(_snapshot())
    row = next(m for m in result.model_outputs if m.model_key == "a")

    assert row.probability_ce == 0.72
    assert row.probability_pe == 0.41
    assert row.confidence == 0.72
    assert row.direction_score == 0.31
    assert "[[0.28" in row.probability_ce_raw
    assert "favorable_trade" in row.label_mapping


def test_final_direction_and_confidence_are_weighted_averages(tmp_path: Path):
    a = _write_artifact(tmp_path, "a", SideSensitiveModel(0.72, 0.41))
    b = _write_artifact(tmp_path, "b", SideSensitiveModel(0.60, 0.20))
    router = EnsembleAutoRouter(
        _config(
            tmp_path,
            {"a": _model_spec(a, 0.75), "b": _model_spec(b, 0.25)},
            thresholds={
                "ensemble_min_confidence": 0.55,
                "ensemble_min_direction_edge": 0.20,
                "ensemble_max_model_disagreement": 0.35,
                "ensemble_min_valid_models": 2,
            },
        )
    )

    result = router.decide(_snapshot())

    assert result.decision == DECISION_BUY_CE
    assert abs(result.final_direction - ((0.31 * 0.75) + (0.40 * 0.25))) < 1e-9
    assert abs(result.final_confidence - ((0.72 * 0.75) + (0.60 * 0.25))) < 1e-9


def test_missing_label_metadata_is_no_vote_not_fake_zero(tmp_path: Path):
    d = tmp_path / "unmapped"
    d.mkdir()
    with (d / "model.pkl").open("wb") as fh:
        pickle.dump({"model": SideSensitiveModel(0.9, 0.2), "feature_order": ["option_type_ce", "ltp"]}, fh)
    mapped = _write_artifact(tmp_path, "mapped", SideSensitiveModel(0.8, 0.2))
    router = EnsembleAutoRouter(_config(tmp_path, {"unmapped": _model_spec(str(d), 0.5), "mapped": _model_spec(mapped, 0.5)}))

    result = router.decide(_snapshot())
    row = next(m for m in result.model_outputs if m.model_key == "unmapped")

    assert row.status == STATUS_UNKNOWN_LABEL_MAPPING
    assert row.vote == "NO_VOTE"
    assert row.confidence is None
    assert row.probability_ce is None
    assert "UNKNOWN_LABEL_MAPPING" in row.label_mapping


def test_fewer_than_two_models_gives_no_trade(tmp_path: Path):
    a = _write_artifact(tmp_path, "a", SideSensitiveModel(0.9, 0.2))
    router = EnsembleAutoRouter(_config(tmp_path, {"a": _model_spec(a)}))
    result = router.decide(_snapshot())
    assert result.decision == DECISION_NO_TRADE
    assert result.block_reason == "FEWER_THAN_MIN_VALID_MODELS"


def test_disagreement_block_works(tmp_path: Path):
    a = _write_artifact(tmp_path, "a", SideSensitiveModel(0.9, 0.1))
    b = _write_artifact(tmp_path, "b", SideSensitiveModel(0.1, 0.9))
    router = EnsembleAutoRouter(_config(tmp_path, {"a": _model_spec(a), "b": _model_spec(b)}))
    result = router.decide(_snapshot())
    assert result.decision == DECISION_NO_TRADE
    assert result.block_reason == "MODELS_DISAGREE_STRONGLY"


def test_disagreement_just_below_threshold_does_not_block(tmp_path: Path):
    a = _write_artifact(tmp_path, "a", SideSensitiveModel(0.85, 0.30))
    b = _write_artifact(tmp_path, "b", SideSensitiveModel(0.75, 0.546))
    router = EnsembleAutoRouter(_config(tmp_path, {"a": _model_spec(a), "b": _model_spec(b)}))

    result = router.decide(_snapshot())

    assert result.decision == DECISION_BUY_CE
    assert result.block_reason == ""
    assert abs(result.final_direction - 0.377) < 1e-9


def test_duplicate_position_block_works(tmp_path: Path):
    a = _write_artifact(tmp_path, "a", SideSensitiveModel(0.9, 0.2))
    b = _write_artifact(tmp_path, "b", SideSensitiveModel(0.8, 0.2))
    router = EnsembleAutoRouter(_config(tmp_path, {"a": _model_spec(a), "b": _model_spec(b)}))
    result = router.decide(_snapshot(), open_positions=[{"side": "CE", "status": "OPEN"}])
    assert result.decision == DECISION_NO_TRADE
    assert result.block_reason == "DUPLICATE_POSITION_OPEN"


def test_pnl_uses_lot_size_65(tmp_path: Path):
    a = _write_artifact(tmp_path, "a", SideSensitiveModel(0.9, 0.2))
    b = _write_artifact(tmp_path, "b", SideSensitiveModel(0.8, 0.2))
    router = EnsembleAutoRouter(_config(tmp_path, {"a": _model_spec(a), "b": _model_spec(b)}))
    paper = EnsemblePaperTrader(router)
    paper.start()

    paper.on_snapshot(_snapshot(100.0, spread=0.0))
    assert paper.open_position is not None
    paper.on_snapshot(_snapshot(121.0, spread=0.0))

    assert paper.open_position is None
    assert paper.trades[0]["qty"] == 65
    assert paper.trades[0]["net_pnl"] == (121.0 - 100.0) * 65


def test_paper_trade_export_works(tmp_path: Path):
    a = _write_artifact(tmp_path, "a", SideSensitiveModel(0.9, 0.2))
    b = _write_artifact(tmp_path, "b", SideSensitiveModel(0.8, 0.2))
    router = EnsembleAutoRouter(_config(tmp_path, {"a": _model_spec(a), "b": _model_spec(b)}))
    paper = EnsemblePaperTrader(router)
    paper.start()
    paper.on_snapshot(_snapshot(100.0, spread=0.0))
    paper.on_snapshot(_snapshot(121.0, spread=0.0))

    trades_path, summary_path = paper.export()
    assert trades_path.exists()
    assert summary_path.exists()
    assert "NIFTYCE" in trades_path.read_text(encoding="utf-8")
    assert '"total_trades": 1' in summary_path.read_text(encoding="utf-8")
