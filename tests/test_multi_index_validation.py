from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from multi_index_validation import (
    per_symbol_walk_forward_validation,
    cross_symbol_validation,
    leave_one_symbol_out_validation,
    calibrate_per_symbol_probabilities,
    temporal_holdout_validation,
    SymbolProbabilityCalibrator,
)


def test_temporal_holdout_validation() -> None:
    # Build synthetic frame
    data = []
    base_time = datetime(2026, 5, 1, 9, 15)
    # Make enough samples for train (20) and test (10)
    for i in range(50):
        data.append({
            "as_of_ts": base_time + timedelta(minutes=i),
            "session_date": date(2026, 5, 1),
            "symbol_key": "BANKNIFTY",
            "symbol_id": 2,
            "index_family": "BANKNIFTY",
            "feature1": float(i % 5),
            "feature2": float(i % 3),
            "label": 1 if i % 2 == 0 else 0,
        })
    frame = pd.DataFrame(data)
    res = temporal_holdout_validation(frame, target_col="label", feature_cols=["feature1", "feature2"], holdout_ratio=0.3)
    assert "passed" in res
    assert "metrics" in res
    assert res["train_samples"] == 35
    assert res["test_samples"] == 15


def test_symbol_probability_calibrator() -> None:
    # Build calibrator
    cal = SymbolProbabilityCalibrator()
    y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 24 samples
    y_prob = [0.9, 0.1, 0.8, 0.2, 0.75, 0.3, 0.85, 0.15, 0.9, 0.12, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2, 0.75, 0.3, 0.85, 0.15, 0.9, 0.12, 0.8, 0.2]
    symbol_keys = ["BANKNIFTY"] * 24

    cal.fit(y_true, y_prob, symbol_keys)
    assert "BANKNIFTY" in cal.params
    A, B = cal.params["BANKNIFTY"]

    # Verify probability calibration
    p_cal = cal.calibrate(0.9, "BANKNIFTY")
    assert 0.0 <= p_cal <= 1.0

    thresh = cal.get_decision_threshold("BANKNIFTY")
    assert 0.3 <= thresh <= 0.7
