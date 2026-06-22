from __future__ import annotations

import json
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
