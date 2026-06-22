from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_ml_models_from_csv import (  # noqa: E402
    LoadedCandidateModel,
    _materialize_offline_model_features,
    _planned_csv_columns,
    _score_loaded_candidate_frame,
    extract_model_feature_order,
    run_historical_ml_backtest,
)
import scripts.backtest_ml_models_from_csv as bt  # noqa: E402


class SimpleProbaModel(ClassifierMixin, BaseEstimator):
    def __init__(self, features: list[str] | None = None) -> None:
        if features is not None:
            self.feature_names_in_ = features
        self.classes_ = [0, 1]

    def fit(self, X, y=None):
        return self

    def get_params(self, deep: bool = True):
        return {}

    def set_params(self, **params):
        return self

    def predict_proba(self, X):
        vals = []
        rows = X.iterrows() if hasattr(X, "iterrows") else enumerate(X)
        for _, row in rows:
            first_value = row.iloc[0] if hasattr(row, "iloc") else row[0]
            score = min(0.95, max(0.05, 0.2 + 0.2 * float(first_value)))
            vals.append([1.0 - score, score])
        return vals


class SimpleScaler:
    def transform(self, X):
        return X


def _write_candidate_config(root: Path, cid: str, candidate_extra: dict | None = None) -> Path:
    cfg = root / "config.json"
    candidate = {
        "candidate_id": cid,
        "enabled": True,
        "model_name": "elasticnet",
        "preset_family": "BOTH_directional_auto",
        "side_policy": "AUTO_DIRECTIONAL",
        "artifact_dir": f"artifacts/candidates/{cid}",
        "model_path": f"artifacts/candidates/{cid}/model.pkl",
    }
    candidate.update(candidate_extra or {})
    cfg.write_text(json.dumps({"candidates": [candidate]}), encoding="utf-8")
    return cfg


def _write_artifact(root: Path, cid: str, payload: object, sidecar: dict | None = None) -> Path:
    folder = root / "artifacts" / "candidates" / cid
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "model.pkl"
    with path.open("wb") as fh:
        pickle.dump(payload, fh)
    manifest = {
        "candidate_id": cid,
        "model_name": "elasticnet",
        "preset_family": "BOTH_directional_auto",
        "side_policy": "AUTO_DIRECTIONAL",
    }
    (folder / "candidate_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if sidecar is not None:
        (folder / "feature_order.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return path


def _write_csv(path: Path) -> None:
    pd.DataFrame(
        [
            {"timestamp": "2026-06-10 09:15:00", "symbol": "NIFTY", "option_type": "PE", "strike": 23000, "expiry": "2026-06-25", "ltp": 100, "f1": 3.0, "f2": 1.0},
            {"timestamp": "2026-06-10 09:16:00", "symbol": "NIFTY", "option_type": "PE", "strike": 23000, "expiry": "2026-06-25", "ltp": 103, "f1": 4.0, "f2": 1.5},
            {"timestamp": "2026-06-10 09:17:00", "symbol": "NIFTY", "option_type": "PE", "strike": 23000, "expiry": "2026-06-25", "ltp": 106, "f1": 5.0, "f2": 2.0},
        ]
    ).to_csv(path, index=False)


def test_dict_artifact_with_feature_order_works(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path, "elasticnet_BOTH_directional_auto_BOTH_dir_auto_t30_20260610_130953", {"model": SimpleProbaModel(), "feature_order": ["f1", "f2"]})
    with path.open("rb") as fh:
        info = extract_model_feature_order(pickle.load(fh), path)
    assert info["feature_order"] == ["f1", "f2"]
    assert info["model"].__class__.__name__ == "SimpleProbaModel"


def test_plain_sklearn_like_model_with_feature_names_in_works(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path, "elasticnet_BOTH_directional_auto_BOTH_dir_auto_t30_20260610_130953", SimpleProbaModel(["f1", "f2"]))
    with path.open("rb") as fh:
        info = extract_model_feature_order(pickle.load(fh), path)
    assert info["feature_order"] == ["f1", "f2"]
    assert info["feature_source"] == "artifact.feature_names_in_"


def test_sidecar_feature_order_json_works(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path, "elasticnet_BOTH_directional_auto_BOTH_dir_auto_t30_20260610_130953", {"model": SimpleProbaModel()}, {"feature_order": ["f1", "f2"]})
    with path.open("rb") as fh:
        info = extract_model_feature_order(pickle.load(fh), path)
    assert info["feature_order"] == ["f1", "f2"]
    assert str(info["feature_source"]).startswith("sidecar:feature_order.json")


def test_direct_backtest_pkl_feature_list_works(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bt, "_project_root", lambda: tmp_path)
    artifact = tmp_path / "ANY_SIDE__COMBINED__rf__cost_survivor_v2__ATM_near__t40.pkl"
    with artifact.open("wb") as fh:
        pickle.dump(
            {
                "model": SimpleProbaModel(),
                "scaler": SimpleScaler(),
                "threshold": 0.4,
                "feature_list": ["f1", "f2"],
                "label": "cost_survivor_label_v2",
                "filter": "ATM_near",
            },
            fh,
        )
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    result = run_historical_ml_backtest(str(csv_path), str(artifact), output_dir=str(tmp_path / "out"), max_trades_per_day=20)
    assert result.summary["loaded_model_count"] == 1
    assert result.summary["model_ready_count"] == 1
    assert result.summary["selected_score_column"] == "artifact_model_scores"


def test_direct_artifact_picks_threshold_from_adjacent_retrain_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bt, "_project_root", lambda: tmp_path)
    artifact = tmp_path / "retrain_all_models_random_forest_profitable_trade_label.pkl"
    with artifact.open("wb") as fh:
        pickle.dump(
            {
                "model": SimpleProbaModel(),
                "scaler": SimpleScaler(),
                "feature_list": ["f1", "f2"],
            },
            fh,
        )
    (tmp_path / "core_retrain_summary.json").write_text(
        json.dumps(
            {
                "result": {
                    "trained_models": [
                        {
                            "label": "profitable_trade_label",
                            "model": "random_forest",
                            "selected_threshold": 0.35,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        [
            {"timestamp": "2026-06-10 09:15:00", "symbol": "NIFTY", "option_type": "PE", "strike": 23000, "expiry": "2026-06-25", "ltp": 100, "f1": 1.0, "f2": 0.0},
            {"timestamp": "2026-06-10 09:16:00", "symbol": "NIFTY", "option_type": "PE", "strike": 23000, "expiry": "2026-06-25", "ltp": 101, "f1": 1.0, "f2": 0.0},
        ]
    ).to_csv(csv_path, index=False)
    result = run_historical_ml_backtest(str(csv_path), str(artifact), output_dir=str(tmp_path / "out"))
    assert result.summary["rows_above_threshold"] == 2
    assert result.summary["rejection_counts"].get("score_threshold", 0) == 0


def test_empty_feature_list_logs_model_features_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bt, "_project_root", lambda: tmp_path)
    cid = "elasticnet_BOTH_directional_auto_BOTH_dir_auto_t30_20260610_130953"
    _write_artifact(tmp_path, cid, {"model": SimpleProbaModel()})
    cfg = _write_candidate_config(tmp_path, cid)
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    logs: list[str] = []
    result = run_historical_ml_backtest(str(csv_path), str(cfg), output_dir=str(tmp_path / "out"), debug=True, log_fn=logs.append)
    assert result.summary["status"] == "FAILED - 0 MODEL-READY CANDIDATES"
    assert any("MODEL_FEATURES_MISSING" in line for line in logs)


def test_csv_usecols_includes_union_of_model_features() -> None:
    cols = ["timestamp", "ltp", "symbol", "f1", "f2", "f3", "unused"]
    m1 = LoadedCandidateModel("c1", Path("m1.pkl"), None, ["f1", "f2"], 0.3)
    m2 = LoadedCandidateModel("c2", Path("m2.pkl"), None, ["f3"], 0.3)
    planned = _planned_csv_columns(cols, [m1, m2], include_score_columns=False)
    assert {"f1", "f2", "f3"}.issubset(set(planned))


def test_no_model_ready_candidates_stops_early(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bt, "_project_root", lambda: tmp_path)
    cid = "elasticnet_BOTH_directional_auto_BOTH_dir_auto_t30_20260610_130953"
    _write_artifact(tmp_path, cid, {"model": SimpleProbaModel()})
    cfg = _write_candidate_config(tmp_path, cid)
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    result = run_historical_ml_backtest(str(csv_path), str(cfg), output_dir=str(tmp_path / "out"))
    assert result.summary["status"] == "FAILED - 0 MODEL-READY CANDIDATES"
    assert result.summary["rows_processed"] == 0


def test_model_ready_candidate_produces_score_column(tmp_path: Path) -> None:
    model = LoadedCandidateModel("candidate one", Path("model.pkl"), SimpleProbaModel(), ["f1", "f2"], 0.3)
    df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [0.0, 0.0]})
    score_col = _score_loaded_candidate_frame(model, df)
    assert score_col in df.columns
    assert df[score_col].notna().all()


def test_config_present_no_score_fallback_does_not_process_rows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bt, "_project_root", lambda: tmp_path)
    cid = "elasticnet_BOTH_directional_auto_BOTH_dir_auto_t30_20260610_130953"
    _write_artifact(tmp_path, cid, {"model": SimpleProbaModel()})
    cfg = _write_candidate_config(tmp_path, cid)
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    result = run_historical_ml_backtest(str(csv_path), str(cfg), output_dir=str(tmp_path / "out"))
    assert result.summary["selected_score_column"] is None
    assert result.summary["rows_processed"] == 0


def test_direct_artifact_uses_candidate_feature_order_source(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bt, "_project_root", lambda: tmp_path)
    artifact_dir = tmp_path / "models" / "rf_gui_retrain_20260620_100000" / "research_retrain_20260620_100001"
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "retrain_all_models_random_forest_profitable_trade_label.pkl"
    with artifact.open("wb") as fh:
        pickle.dump({"model": SimpleProbaModel(), "threshold": 0.4}, fh)

    wrapper_dir = tmp_path / "models" / "candidates" / "rf_wrapper"
    wrapper_dir.mkdir(parents=True)
    (wrapper_dir / "feature_schema.json").write_text(
        json.dumps({"features": ["f1", "f2"]}),
        encoding="utf-8",
    )

    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "rf_wrapper_candidate",
                        "enabled": True,
                        "model_name": "random_forest",
                        "direct_model_artifact": True,
                        "model_path": str(artifact),
                        "artifact_dir": str(wrapper_dir),
                        "feature_order_source": str(wrapper_dir / "feature_schema.json"),
                        "selected_threshold": 0.4,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)

    result = run_historical_ml_backtest(str(csv_path), str(cfg), output_dir=str(tmp_path / "out"), max_trades_per_day=20)

    assert result.summary["loaded_model_count"] == 1
    assert result.summary["model_ready_count"] == 1


def test_offline_materialization_fills_known_missing_model_features() -> None:
    df = pd.DataFrame(
        {
            "timestamp": ["2026-06-10 09:15:00"],
            "date": [pd.Timestamp("2026-06-10").date()],
            "strike": [23000.0],
            "spot": [23100.0],
            "ltp": [100.0],
            "option_type": ["PE"],
            "dte_days": [7],
            "volume": [1000],
            "oi": [5000],
            "volume_CE": [700],
            "volume_PE": [300],
            "oi_CE": [6000],
            "oi_PE": [4000],
        }
    )
    report = _materialize_offline_model_features(
        df,
        ["strike_price", "bs_iv", "final_iv", "distance_from_atm", "estimated_cost_bps", "liq_regime_day", "ce_pe_rel_strength"],
    )
    assert report["added_count"] >= 7
    for col in ["strike_price", "bs_iv", "final_iv", "distance_from_atm", "estimated_cost_bps", "liq_regime_day", "ce_pe_rel_strength"]:
        assert col in df.columns
        assert pd.notna(df.loc[0, col])


def test_offline_materialization_builds_direct_strategy_features() -> None:
    rows = []
    for i in range(40):
        rows.append(
            {
                "timestamp": f"2026-06-10 09:{15 + i:02d}:00",
                "ts": pd.Timestamp(f"2026-06-10 09:{15 + i:02d}:00"),
                "date": pd.Timestamp("2026-06-10").date(),
                "_contract": "NIFTY-PE-23000-2026-06-25",
                "symbol": "NIFTY",
                "ltp": 100.0 + i * 0.5,
                "volume": 1000 + i,
                "oi": 5000 + i * 10,
            }
        )
    df = pd.DataFrame(rows)
    report = _materialize_offline_model_features(
        df,
        [
            "mean_reversion_zscore",
            "mean_reversion_entry_score",
            "mean_reversion_expected_reversion_pct",
            "mean_reversion_half_life_bars",
            "mean_reversion_buy_call",
            "mean_reversion_buy_put",
            "stat_arb_zscore",
            "stat_arb_confidence",
            "stat_arb_spread_pct",
            "stat_arb_hedge_ratio",
            "stat_arb_long_spread",
            "stat_arb_short_spread",
        ],
    )
    assert report["added_count"] >= 12
    for col in [
        "mean_reversion_zscore",
        "mean_reversion_entry_score",
        "mean_reversion_expected_reversion_pct",
        "mean_reversion_half_life_bars",
        "mean_reversion_buy_call",
        "mean_reversion_buy_put",
        "stat_arb_zscore",
        "stat_arb_confidence",
        "stat_arb_spread_pct",
        "stat_arb_hedge_ratio",
        "stat_arb_long_spread",
        "stat_arb_short_spread",
    ]:
        assert col in df.columns
        assert df[col].notna().all()
