from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_ml_models_from_csv import (  # noqa: E402
    _parse_threshold_from_candidate_id,
    _select_score_column,
    run_historical_ml_backtest,
)


class _StaticProbaModel:
    classes_ = [0, 1]

    def predict_proba(self, X):
        return [[0.1, 0.9] for _ in range(len(X))]


class _ConstantProbModel:
    def __init__(self, prob: float) -> None:
        self.prob = float(prob)
        self.classes_ = [0, 1]

    def predict_proba(self, X):
        return [[1.0 - self.prob, self.prob] for _ in range(len(X))]


def _write_synthetic_csv(path: Path, *, score_col: str = "pred_proba") -> None:
    rows = []
    for i in range(8):
        rows.append(
            {
                "timestamp": f"2026-06-10 09:{15 + i:02d}:00",
                "symbol": "NIFTY",
                "option_type": "PE",
                "strike": 23000,
                "expiry": "2026-06-25",
                "ltp": 100 + i * 5,
                score_col: 0.75 if i in {0, 3, 6} else 0.10,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_embedded_score_backtest_generates_trades(tmp_path: Path) -> None:
    csv_path = tmp_path / "synthetic.csv"
    out_dir = tmp_path / "out"
    _write_synthetic_csv(csv_path)

    result = run_historical_ml_backtest(
        csv_path=str(csv_path),
        candidate_config_path=None,
        output_dir=str(out_dir),
        threshold=0.30,
        target_pct=0.01,
        stoploss_pct=0.10,
        max_hold_bars=2,
        max_trades_per_day=3,
        debug=True,
    )

    assert Path(result.output_paths["trades_csv"]).exists()
    assert result.summary["total_trades"] > 0
    assert result.summary["selected_score_column"] == "pred_proba"
    assert result.summary["rows_above_threshold"] > 0
    trades = pd.read_csv(result.output_paths["trades_csv"])
    assert {
        "candidate_id",
        "entry_ts",
        "exit_ts",
        "symbol",
        "option_type",
        "entry_price",
        "exit_price",
        "gross_pnl",
        "cost",
        "net_pnl",
        "exit_reason",
        "model_score",
        "bars_held",
    }.issubset(trades.columns)


def test_threshold_parsed_from_candidate_id() -> None:
    assert _parse_threshold_from_candidate_id("elasticnet_PE_only_conservative_t30_20260610") == 0.30
    assert _parse_threshold_from_candidate_id("elasticnet_PE_only_conservative_t35_20260610") == 0.35
    assert _parse_threshold_from_candidate_id("elasticnet_PE_only_conservative_t60_20260610") == 0.60


def test_embedded_score_detection_priority() -> None:
    df = pd.DataFrame(
        {
            "timestamp": ["2026-06-10"],
            "ltp": [100],
            "model_score": [0.2],
            "probability": [0.3],
            "pred_proba": [0.4],
        }
    )
    assert _select_score_column(df) == "pred_proba"


def test_missing_model_artifacts_do_not_crash(tmp_path: Path) -> None:
    csv_path = tmp_path / "synthetic.csv"
    cfg_path = tmp_path / "candidates.json"
    out_dir = tmp_path / "out"
    _write_synthetic_csv(csv_path)
    cfg_path.write_text(
        json.dumps({"candidates": [{"candidate_id": "missing_model_t30_20260610", "enabled": True, "artifact_path": "does_not_exist"}]}),
        encoding="utf-8",
    )

    logs: list[str] = []
    result = run_historical_ml_backtest(
        csv_path=str(csv_path),
        candidate_config_path=str(cfg_path),
        output_dir=str(out_dir),
        threshold=0.30,
        target_pct=0.01,
        max_hold_bars=2,
        log_fn=logs.append,
        debug=True,
        allow_embedded_score_fallback=True,
    )

    assert result.summary["total_trades"] > 0
    assert result.summary["artifact_loading_failures"]
    assert any("no loadable" in line.lower() or "no pickle/joblib" in line.lower() for line in logs)


def test_bad_pickle_prints_traceback_and_continues_with_csv_score(tmp_path: Path) -> None:
    csv_path = tmp_path / "synthetic.csv"
    bad_pkl = tmp_path / "bad_model.pkl"
    cfg_path = tmp_path / "candidates.json"
    out_dir = tmp_path / "out"
    _write_synthetic_csv(csv_path)
    bad_pkl.write_bytes(b"not a pickle")
    cfg_path.write_text(
        json.dumps({"candidates": [{"candidate_id": "bad_pickle_t30_20260610", "enabled": True, "artifact_path": str(bad_pkl)}]}),
        encoding="utf-8",
    )

    logs: list[str] = []
    result = run_historical_ml_backtest(
        csv_path=str(csv_path),
        candidate_config_path=str(cfg_path),
        output_dir=str(out_dir),
        threshold=0.30,
        target_pct=0.01,
        max_hold_bars=2,
        log_fn=logs.append,
        debug=True,
        allow_embedded_score_fallback=True,
    )

    assert result.summary["total_trades"] > 0
    assert result.summary["artifact_loading_failures"]


def test_models_root_prefers_guarded_ensemble_wrapper(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.backtest_ml_models_from_csv._project_root", lambda: tmp_path)

    csv_path = tmp_path / "synthetic.csv"
    out_dir = tmp_path / "out"
    _write_synthetic_csv(csv_path, score_col="unused_score")

    models_dir = tmp_path / "models"
    raw_ensemble_dir = models_dir / "all_combined_parquet_rf_xgb_ensemble"
    raw_ensemble_dir.mkdir(parents=True)
    with (raw_ensemble_dir / "xgboost_model_cuda.pkl").open("wb") as fh:
        pickle.dump(_StaticProbaModel(), fh)
    with (raw_ensemble_dir / "random_forest_model.pkl").open("wb") as fh:
        pickle.dump(_StaticProbaModel(), fh)

    wrapper_dir = models_dir / "candidates" / "wrapped_ensemble_candidate"
    wrapper_dir.mkdir(parents=True)
    (wrapper_dir / "feature_schema.json").write_text(json.dumps({"features": []}), encoding="utf-8")

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    wrapper_cfg = config_dir / "ensemble_xgb_rf_dynamic_candidate.json"
    wrapper_cfg.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "wrapped_ensemble_candidate",
                        "enabled": True,
                        "direct_model_artifact": True,
                        "model_name": "xgb_rf_ensemble",
                        "model_path": str(raw_ensemble_dir / "xgboost_model_cuda.pkl"),
                        "artifact_dir": str(wrapper_dir),
                        "feature_order_source": str(wrapper_dir / "feature_schema.json"),
                        "selected_threshold": 0.4,
                        "max_trades_per_day": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = run_historical_ml_backtest(
        csv_path=str(csv_path),
        candidate_config_path=str(models_dir),
        output_dir=str(out_dir),
        threshold=0.30,
        target_pct=0.01,
        stoploss_pct=0.10,
        max_hold_bars=2,
        max_trades_per_day=5,
        debug=True,
    )

    assert result.summary["candidate_config_path"] == str(wrapper_cfg.resolve())
    assert result.summary["requested_candidate_config_path"] == str(models_dir.resolve())


def test_xgb_rf_wrapper_prefers_ensemble_artifact_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.backtest_ml_models_from_csv._project_root", lambda: tmp_path)

    ensemble_dir = tmp_path / "models" / "all_combined_parquet_rf_xgb_ensemble"
    ensemble_dir.mkdir(parents=True)
    with (ensemble_dir / "xgboost_model_cuda.pkl").open("wb") as fh:
        pickle.dump(_ConstantProbModel(0.90), fh)
    with (ensemble_dir / "random_forest_model.pkl").open("wb") as fh:
        pickle.dump(_ConstantProbModel(0.10), fh)
    (ensemble_dir / "ensemble_config.json").write_text(
        json.dumps(
            {
                "feature_columns": ["f1", "f2"],
                "fill_values": {"f1": 0.0, "f2": 0.0},
                "xgb_weight": 0.7,
                "rf_weight": 0.3,
                "ensemble_threshold": 0.6,
                "xgb_min_prob": 0.58,
                "rf_min_prob": 0.52,
                "max_model_disagreement": 0.25,
                "block_on_disagreement": True,
            }
        ),
        encoding="utf-8",
    )

    wrapper_dir = tmp_path / "models" / "candidates" / "wrapped_ensemble_candidate"
    wrapper_dir.mkdir(parents=True)
    feature_schema = wrapper_dir / "feature_schema.json"
    feature_schema.write_text(json.dumps({"features": ["f1", "f2"]}), encoding="utf-8")

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "wrapped_ensemble_candidate",
                        "enabled": True,
                        "model_name": "xgb_rf_ensemble",
                        "direct_model_artifact": True,
                        "artifact_dir": str(ensemble_dir),
                        "model_path": str(ensemble_dir / "xgboost_model_cuda.pkl"),
                        "feature_order_source": str(feature_schema),
                        "selected_threshold": 0.6,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    csv_path = tmp_path / "synthetic.csv"
    pd.DataFrame(
        [
            {"timestamp": "2026-06-10 09:15:00", "symbol": "NIFTY", "option_type": "PE", "strike": 23000, "expiry": "2026-06-25", "ltp": 100, "f1": 1.0, "f2": 2.0},
            {"timestamp": "2026-06-10 09:16:00", "symbol": "NIFTY", "option_type": "PE", "strike": 23000, "expiry": "2026-06-25", "ltp": 102, "f1": 1.0, "f2": 2.0},
        ]
    ).to_csv(csv_path, index=False)

    result = run_historical_ml_backtest(
        csv_path=str(csv_path),
        candidate_config_path=str(cfg_path),
        output_dir=str(tmp_path / "out"),
        threshold=0.6,
        target_pct=0.01,
        stoploss_pct=0.10,
        max_hold_bars=2,
        max_trades_per_day=5,
        debug=True,
    )

    assert result.summary["loaded_model_count"] == 1
    assert result.summary["model_ready_count"] == 1
    assert result.summary["feature_alignment"][0]["model_type"] == "xgb_rf_ensemble"
    assert result.summary["rows_above_threshold"] == 0
