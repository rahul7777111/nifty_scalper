"""ML training script for the signal ensemble.

The script accepts raw OHLCV candles and derives features through the shared
pipeline. If a label column is not supplied, it trains on next-candle direction.
"""
from __future__ import annotations

import argparse
import os
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="CSV with OHLCV candles or engineered features")
    p.add_argument("--out", default="ml_signal_model.pkl", help="Output model path")
    p.add_argument("--label-column", default="", help="Optional binary label column name")
    p.add_argument("--lookback", type=int, default=20, help="Feature lookback window")
    p.add_argument("--horizon", type=int, default=1, help="Prediction horizon in candles")
    p.add_argument("--walk-forward-splits", type=int, default=5, help="Walk-forward folds for metrics")
    args = p.parse_args()

    from src.ml_signals import train_ensemble
    from src.ml_pipeline import MLFeatureContext, build_supervised_dataset
    from datetime import datetime

    df = pd.read_csv(args.data)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    label_col = str(args.label_column or "").strip()
    if {"open", "high", "low", "close"}.issubset({str(c).lower() for c in df.columns}):
        from market_data import Candle

        def _get(row, name, default=0.0):
            for key in (name, name.capitalize(), name.upper()):
                if key in row and pd.notna(row[key]):
                    return row[key]
            return default

        candles = []
        contexts = []
        labels = None
        if label_col and label_col in df.columns:
            labels = [int(v) for v in df[label_col].fillna(0).tolist()]

        for _, row in df.iterrows():
            candles.append(
                Candle(
                    time=pd.to_datetime(row.get("time") or row.get("timestamp") or datetime.utcnow()),
                    open=float(_get(row, "open", 0.0)),
                    high=float(_get(row, "high", 0.0)),
                    low=float(_get(row, "low", 0.0)),
                    close=float(_get(row, "close", 0.0)),
                    volume=float(_get(row, "volume", 0.0)) if pd.notna(_get(row, "volume", None)) else None,
                )
            )
            contexts.append(
                MLFeatureContext(
                    regime=str(row.get("regime") or ""),
                    iv=float(row.get("iv")) if pd.notna(row.get("iv")) else None,
                    iv_change_pct=float(row.get("iv_change_pct")) if pd.notna(row.get("iv_change_pct")) else None,
                    iv_percentile=float(row.get("iv_percentile")) if pd.notna(row.get("iv_percentile")) else None,
                    delta=float(row.get("delta")) if pd.notna(row.get("delta")) else None,
                    gamma=float(row.get("gamma")) if pd.notna(row.get("gamma")) else None,
                    vega=float(row.get("vega")) if pd.notna(row.get("vega")) else None,
                    theta=float(row.get("theta")) if pd.notna(row.get("theta")) else None,
                    spot=float(row.get("spot")) if pd.notna(row.get("spot")) else None,
                    option_price=float(row.get("option_price")) if pd.notna(row.get("option_price")) else None,
                )
            )

        X, y, feature_names = build_supervised_dataset(
            candles,
            labels=labels,
            contexts=contexts,
            lookback=int(args.lookback),
            horizon=int(args.horizon),
        )
        model = train_ensemble(
            X,
            y,
            save_path=args.out,
            feature_names=feature_names,
            walk_forward_splits=int(args.walk_forward_splits),
        )
    else:
        if df.shape[1] < 2:
            raise SystemExit("CSV must have at least one feature and a target column")
        if label_col and label_col in df.columns:
            y = df[label_col]
            X = df.drop(columns=[label_col])
        else:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        model = train_ensemble(
            X,
            y,
            save_path=args.out,
            feature_names=list(getattr(X, "columns", [])),
            walk_forward_splits=int(args.walk_forward_splits),
        )
    # save a small metadata file
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    meta = {"trained_at": datetime.utcnow().isoformat() + "Z", "model_path": args.out}
    try:
        if hasattr(model, "metrics"):
            meta["metrics"] = getattr(model, "metrics")
        if hasattr(model, "feature_names"):
            meta["feature_names"] = getattr(model, "feature_names")
    except Exception:
        pass
    try:
        import json as _json

        with open(meta_path, "w", encoding="utf-8") as mf:
            mf.write(_json.dumps(meta))
    except Exception:
        pass
    print(f"Trained model saved to {args.out}")

if __name__ == "__main__":
    main()
