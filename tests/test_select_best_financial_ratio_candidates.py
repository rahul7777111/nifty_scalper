from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.select_best_financial_ratio_candidates import run_selection


class F1Model:
    def __init__(self) -> None:
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        probs = np.clip(np.asarray(X)[:, 0].astype(float), 0.0, 1.0)
        return np.column_stack([1.0 - probs, probs])


class F2Model:
    def __init__(self) -> None:
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        probs = np.clip(np.asarray(X)[:, 0].astype(float), 0.0, 1.0)
        return np.column_stack([1.0 - probs, probs])


class F3Model:
    def __init__(self) -> None:
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        probs = np.clip(np.asarray(X)[:, 0].astype(float), 0.0, 1.0)
        return np.column_stack([1.0 - probs, probs])


class ZeroModel:
    def __init__(self) -> None:
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        probs = np.zeros(len(X), dtype=float)
        return np.column_stack([1.0 - probs, probs])


def _write_candidate(
    base: Path,
    candidate_id: str,
    model,
    *,
    feature_order: list[str],
    threshold: float | None = None,
) -> Path:
    folder = base / candidate_id
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_order": feature_order}, folder / "model.pkl")
    payload = {
        "candidate_id": candidate_id,
        "model_name": "logistic_regression",
        "preset_family": "BOTH_directional_auto",
        "side_policy": "BOTH",
        "feature_order": feature_order,
    }
    if threshold is not None:
        payload["threshold"] = threshold
    (folder / "candidate_profile.json").write_text(json.dumps(payload), encoding="utf-8")
    return folder


def _build_dataset(path: Path) -> Path:
    rows = []
    start = pd.Timestamp("2026-01-01T09:15:00+05:30")
    for idx in range(20):
        rows.append(
            {
                "timestamp": (start + pd.Timedelta(minutes=5 * idx)).isoformat(),
                "option_type": "CE" if idx % 2 == 0 else "PE",
                "ltp": 100.0 + idx,
                "gross_forward_return": [
                    1.0,
                    0.9,
                    0.8,
                    0.7,
                    -0.4,
                    -0.3,
                    0.6,
                    0.5,
                    0.4,
                    -0.2,
                    -0.1,
                    0.3,
                    0.25,
                    -0.15,
                    0.2,
                    0.18,
                    -0.12,
                    0.15,
                    0.12,
                    -0.05,
                ][idx],
                "net_forward_return": [
                    0.9,
                    0.8,
                    0.7,
                    0.6,
                    -0.5,
                    -0.4,
                    0.5,
                    0.4,
                    0.3,
                    -0.3,
                    -0.2,
                    0.2,
                    0.15,
                    -0.25,
                    0.1,
                    0.08,
                    -0.22,
                    0.05,
                    0.02,
                    -0.15,
                ][idx],
                "f1": 0.95 if idx < 4 else (0.10 if idx < 12 else 0.55),
                "f2": 0.92 if idx in {0, 2, 4, 6, 8, 10, 12, 14, 16, 18} else 0.88,
                "f3": 0.90 if idx < 16 else 0.20,
                "volume": 1000 + idx,
                "oi": 2000 + idx,
                "strike_price": 25000 + idx * 50,
                "ctx_spot": 25010 + idx * 25,
                "dte_days": 4,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_financial_ratio_selector_ranking_and_rejections(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    dataset = _build_dataset(tmp_path / "dataset.csv")
    output_config = tmp_path / "config" / "paper_forward_candidates_financial_best.json"

    _write_candidate(artifacts / "grp1", "low_trade_profitable", F1Model(), feature_order=["f1"])
    _write_candidate(artifacts / "grp1", "high_win_bad_payoff", F2Model(), feature_order=["f2"], threshold=0.85)
    _write_candidate(artifacts / "grp2", "high_pnl_drawdown", F3Model(), feature_order=["f3"])
    _write_candidate(artifacts / "grp2", "constant_zero", ZeroModel(), feature_order=["f1"])
    _write_candidate(artifacts / "grp3", "missing_features", F1Model(), feature_order=["missing_col"])
    broken = artifacts / "grp3" / "broken_model"
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "candidate_profile.json").write_text(json.dumps({"candidate_id": "broken_model", "feature_order": ["f1"]}), encoding="utf-8")

    existing_config = tmp_path / "config" / "paper_forward_candidates.json"
    existing_config.parent.mkdir(parents=True, exist_ok=True)
    existing_config.write_text('{"keep":"original"}', encoding="utf-8")

    summary = run_selection(
        artifacts_dir=artifacts,
        dataset_path=dataset,
        output_config=output_config,
        apply=True,
        dry_run=False,
        top_n=2,
        min_financial_score=0.0,
        side="ALL",
        include_low_activity=True,
        no_min_trades=True,
        allow_small_loss_if_best=True,
        max_drawdown_limit=None,
        sort_by="financial_score",
    )

    assert summary["valid_model_candidates_found"] >= 4
    assert summary["evaluated_candidates_count"] >= 3
    assert summary["technically_rejected_count"] >= 3
    assert summary["selected_candidates_count"] == 2

    ranked = {row["candidate_id"]: row for row in summary["evaluated_rows"]}
    rejected = {row["candidate_id"]: row for row in summary["rejected_rows"]}

    assert ranked["low_trade_profitable"]["trade_count"] > 0
    assert ranked["low_trade_profitable"]["activity_label"] in {"LOW_ACTIVITY", "MEDIUM_ACTIVITY"}
    assert rejected["constant_zero"]["reject_reason"] == "CONSTANT_ZERO_PREDICTION"
    assert rejected["missing_features"]["reject_reason"] == "FEATURES_MISSING"
    assert rejected["broken_model"]["reject_reason"] in {"ARTIFACT_NOT_FOUND", "MODEL_LOAD_FAILED"}

    assert ranked["low_trade_profitable"]["profit_factor"] > 1.0
    assert ranked["low_trade_profitable"]["financial_score"] >= ranked["high_win_bad_payoff"]["financial_score"]
    assert ranked["high_pnl_drawdown"]["max_drawdown"] >= ranked["low_trade_profitable"]["max_drawdown"]
    assert ranked["high_pnl_drawdown"]["financial_score"] <= ranked["low_trade_profitable"]["financial_score"]
    assert ranked["high_win_bad_payoff"]["payoff_ratio"] <= ranked["low_trade_profitable"]["payoff_ratio"]

    payload = json.loads(output_config.read_text(encoding="utf-8"))
    assert "paper_forward_candidates.json" not in output_config.name
    assert json.loads(existing_config.read_text(encoding="utf-8")) == {"keep": "original"}
    selected_ids = [row["candidate_id"] for row in payload["candidates"]]
    assert selected_ids == summary["selected_candidate_ids"]


def test_financial_ratio_selector_fallback_and_threshold_sweep(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    dataset_path = tmp_path / "dataset.csv"
    df = pd.DataFrame(
        [
            {
                "timestamp": f"2026-01-02T09:{15 + i:02d}:00+05:30",
                "option_type": "CE",
                "ltp": 100.0,
                "gross_forward_return": gross,
                "net_forward_return": net,
                "f1": prob,
                "volume": 1000 + i,
                "oi": 2000 + i,
                "strike_price": 25000.0,
                "ctx_spot": 25010.0,
                "dte_days": 3,
            }
            for i, (prob, gross, net) in enumerate(
                [
                    (0.95, 0.2, -0.1),
                    (0.90, -0.4, -0.5),
                    (0.85, -0.2, -0.3),
                    (0.55, -0.2, -0.3),
                    (0.50, -0.3, -0.4),
                    (0.45, -0.1, -0.2),
                ]
            )
        ]
    )
    df.to_csv(dataset_path, index=False)

    _write_candidate(artifacts, "swept_fallback", F1Model(), feature_order=["f1"])

    summary = run_selection(
        artifacts_dir=artifacts,
        dataset_path=dataset_path,
        output_config=tmp_path / "config" / "paper_forward_candidates_financial_best.json",
        apply=False,
        dry_run=True,
        top_n=1,
        min_financial_score=0.0,
        side="ALL",
        include_low_activity=True,
        no_min_trades=True,
        allow_small_loss_if_best=True,
        max_drawdown_limit=None,
        sort_by="financial_score",
    )

    assert summary["selected_candidates_count"] == 1
    assert summary["fallback_candidates_count"] == 1
    selected = summary["selected_rows"][0]
    assert selected["fallback_candidate"] is True
    assert selected["threshold_source"] == "sweep"
    assert selected["threshold_used"] >= 0.45
