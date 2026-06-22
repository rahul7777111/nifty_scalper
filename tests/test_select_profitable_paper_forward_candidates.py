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

from scripts.select_profitable_paper_forward_candidates import run_selection


class F1ProbabilityModel:
    def __init__(self) -> None:
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        probs = np.clip(np.asarray(X)[:, 0].astype(float), 0.0, 1.0)
        return np.column_stack([1.0 - probs, probs])


class F2ProbabilityModel:
    def __init__(self) -> None:
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        probs = np.clip(np.asarray(X)[:, 1].astype(float), 0.0, 1.0)
        return np.column_stack([1.0 - probs, probs])


class ZeroProbabilityModel:
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
    threshold: float = 0.6,
    model_name: str = "logistic_regression",
) -> Path:
    folder = base / candidate_id
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_order": feature_order}, folder / "model.pkl")
    (folder / "candidate_profile.json").write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "model_name": model_name,
                "preset_family": "BOTH_directional_auto",
                "side_policy": "BOTH",
                "threshold": threshold,
                "feature_order": feature_order,
            }
        ),
        encoding="utf-8",
    )
    return folder


def _build_dataset(path: Path) -> Path:
    rows = [
        {
            "timestamp": "2026-06-10T09:20:00+05:30",
            "option_type": "CE",
            "ltp": 100.0,
            "gross_forward_return": 1.10,
            "net_forward_return": 0.90,
            "f1": 0.95,
            "f2": 0.10,
        },
        {
            "timestamp": "2026-06-10T09:25:00+05:30",
            "option_type": "PE",
            "ltp": 102.0,
            "gross_forward_return": 1.00,
            "net_forward_return": 0.80,
            "f1": 0.92,
            "f2": 0.20,
        },
        {
            "timestamp": "2026-06-10T09:30:00+05:30",
            "option_type": "CE",
            "ltp": 98.0,
            "gross_forward_return": -0.20,
            "net_forward_return": -0.40,
            "f1": 0.20,
            "f2": 0.91,
        },
        {
            "timestamp": "2026-06-10T09:35:00+05:30",
            "option_type": "PE",
            "ltp": 99.0,
            "gross_forward_return": -0.10,
            "net_forward_return": -0.50,
            "f1": 0.15,
            "f2": 0.88,
        },
        {
            "timestamp": "2026-06-10T09:40:00+05:30",
            "option_type": "CE",
            "ltp": 101.0,
            "gross_forward_return": -0.25,
            "net_forward_return": -0.35,
            "f1": 0.12,
            "f2": 0.86,
        },
        {
            "timestamp": "2026-06-10T09:45:00+05:30",
            "option_type": "PE",
            "ltp": 103.0,
            "gross_forward_return": -0.30,
            "net_forward_return": -0.45,
            "f1": 0.10,
            "f2": 0.90,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_profitable_low_trade_candidate_can_pass_without_min_trade_gate(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts" / "candidates"
    dataset = _build_dataset(tmp_path / "dataset.csv")
    output_config = tmp_path / "config" / "paper_forward_candidates_eligible.json"

    _write_candidate(artifacts, "low_trade_profitable", F1ProbabilityModel(), feature_order=["f1", "f2"])
    _write_candidate(artifacts, "high_trade_unprofitable", F2ProbabilityModel(), feature_order=["f1", "f2"])
    _write_candidate(artifacts, "constant_zero", ZeroProbabilityModel(), feature_order=["f1", "f2"])
    _write_candidate(artifacts, "missing_features", F1ProbabilityModel(), feature_order=["f_missing"])
    (artifacts / "missing_artifact").mkdir(parents=True, exist_ok=True)
    (artifacts / "missing_artifact" / "candidate_profile.json").write_text(
        json.dumps(
            {
                "candidate_id": "missing_artifact",
                "model_name": "logistic_regression",
                "preset_family": "BOTH_directional_auto",
                "side_policy": "BOTH",
                "threshold": 0.6,
                "feature_order": ["f1", "f2"],
            }
        ),
        encoding="utf-8",
    )

    existing_config = tmp_path / "config" / "paper_forward_candidates.json"
    existing_config.parent.mkdir(parents=True, exist_ok=True)
    existing_config.write_text('{"keep":"original"}', encoding="utf-8")

    summary = run_selection(
        artifacts_dir=artifacts,
        dataset_path=dataset,
        output_config=output_config,
        apply=True,
        dry_run=False,
        no_min_trades=True,
        max_candidates=None,
        side="ALL",
        include_low_activity=True,
        fallback_threshold=0.6,
    )

    assert summary["candidate_folders_scanned"] == 5
    assert summary["eligible_profitable_candidates_selected"] == 1
    assert summary["rejected_candidates_count"] == 4

    audits = {row["candidate_id"]: row for row in summary["all_audits"]}
    assert audits["low_trade_profitable"]["eligible_paper_forward"] is True
    assert audits["low_trade_profitable"]["trade_count"] == 2
    assert audits["low_trade_profitable"]["activity_label"] == "LOW_ACTIVITY"

    assert audits["high_trade_unprofitable"]["eligible_paper_forward"] is False
    assert audits["high_trade_unprofitable"]["reject_reason"] == "NOT_PROFITABLE_NET_PNL"
    assert audits["missing_artifact"]["reject_reason"] == "ARTIFACT_LOAD_FAILED"
    assert audits["missing_features"]["reject_reason"] == "FEATURES_MISSING"
    assert audits["constant_zero"]["reject_reason"] == "CONSTANT_ZERO_PREDICTION"

    written = json.loads(output_config.read_text(encoding="utf-8"))
    assert [row["candidate_id"] for row in written["candidates"]] == ["low_trade_profitable"]
    assert json.loads(existing_config.read_text(encoding="utf-8")) == {"keep": "original"}
