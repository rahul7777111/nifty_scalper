from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from datetime import datetime


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db import DatabaseManager
from config import load_strategy_config
from market_data import Candle
from ml_signals import evaluate_ml_gating_before_execution
from strategy import NiftyScalper, submit_paper_trade_order


class _StubModel:
    def predict_proba(self, X):
        return [[0.25, 0.75] for _ in X]


def test_prediction_logging_writes_complete_record(tmp_path):
    db_path = tmp_path / "preds.db"
    db = DatabaseManager(str(db_path))
    prediction_id = db.insert_prediction(
        {
            "prediction_id": "pred_1",
            "ts": 123.0,
            "symbol": "NIFTY",
            "exchange": "NSE",
            "token": "12345",
            "underlying_price": 22500.0,
            "direction": "bull",
            "regime": "trending_up",
            "model_version": "ensemble_v1",
            "model_artifact_path": "ml_signal_model.pkl",
            "model_checksum": "abc",
            "label_policy_version": "trade_quality_binary:v1",
            "feature_set_version": "target_features_v1",
            "threshold": 0.55,
            "probability": 0.63,
            "prediction": 0.63,
            "predicted_class": 1,
            "confidence": 0.26,
            "confidence_bucket": "0.6-0.7",
            "strategy_context": "{}",
            "strategy_signal": "directional",
            "reason": "ml_gate",
            "trade_candidate": True,
            "feature_snapshot": {"x": 1.0},
            "features": {"x": 1.0},
            "feature_vector_checksum": "chk",
            "horizon_bars": 45,
        }
    )
    assert prediction_id == "pred_1"

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT prediction_id, model_version, label_policy_version, feature_vector_checksum FROM predictions").fetchone()
    conn.close()
    assert row == ("pred_1", "ensemble_v1", "trade_quality_binary:v1", "chk")


def test_prediction_resolution_updates_record(tmp_path):
    db_path = tmp_path / "preds.db"
    db = DatabaseManager(str(db_path))
    db.insert_prediction({"prediction_id": "pred_2", "ts": 1.0, "symbol": "NIFTY", "probability": 0.5, "prediction": 0.5})
    db.resolve_prediction("pred_2", resolved_label=1, realized_forward_return=0.012, future_label_status="resolved", realized_trade_pnl=120.0)

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT resolved_label, realized_forward_return, future_label_status, realized_trade_pnl FROM predictions WHERE prediction_id='pred_2'").fetchone()
    conn.close()
    assert row == (1, 0.012, "resolved", 120.0)


def test_evaluate_ml_gating_before_execution_returns_token_and_probability():
    prediction_id, probability = evaluate_ml_gating_before_execution(
        {"f1": 1.0, "f2": 2.0},
        _StubModel(),
        ["f1", "f2"],
    )
    assert prediction_id.startswith("PRED_")
    assert len(prediction_id) == 13
    assert probability == 0.75


def test_submit_paper_trade_order_populates_metadata():
    ticket = submit_paper_trade_order({"trade_id": "D1", "metadata": {"reason": "ml_gate"}}, "PRED_ABCD1234", 0.67)
    assert ticket["metadata"]["prediction_id"] == "PRED_ABCD1234"
    assert ticket["metadata"]["probability"] == 0.67
    assert ticket["metadata"]["reason"] == "ml_gate"


def test_shadow_mode_predictions_are_tagged_in_prediction_log(tmp_path):
    class _Client:
        def get_ltp(self, symbol):
            return 100.0

    db_path = tmp_path / "shadow_preds.db"
    cfg = load_strategy_config()
    cfg.shadow_mode = True
    sc = NiftyScalper(_Client(), cfg)
    sc.db_manager = DatabaseManager(str(db_path))
    candle = Candle(datetime.now(), 100.0, 101.0, 99.0, 100.5, volume=1000.0)
    prediction_id = sc._record_ml_prediction(
        prediction_id="shadow_pred_1",
        candles=[candle],
        probability=0.42,
        threshold=0.55,
        take=False,
        reason="technical_filtered",
        regime="quiet",
        suggested_strategy="directional",
        feature_names=["f1", "f2"],
        feature_values=[1.0, 2.0],
        ctx={"regime": "quiet"},
    )
    assert prediction_id == "shadow_pred_1"

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT source, no_trade_reason FROM predictions WHERE prediction_id = 'shadow_pred_1'").fetchone()
    conn.close()
    assert row == ("SHADOW_FILTERED", "technical_filtered")
