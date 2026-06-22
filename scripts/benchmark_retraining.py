"""Benchmark script to run extended validation and retraining benchmarks comparing V1 and V2 pipelines."""
import os
import sys
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
sys.path.insert(0, str(SRC_DIR))

import psutil
from market_data import Candle
from ml_pipeline import (
    build_supervised_dataset,
    build_supervised_dataset_v2,
    MLFeatureContext,
    walk_forward_backtest,
    compute_triple_barrier_labels_vectorized
)
import json
from ml_signals import train_ensemble

def dict_to_candle(raw: dict) -> Candle:
    t_str = raw.get("time", "")
    dt = None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(t_str.split(".")[0], fmt)
            break
        except Exception:
            pass
    if dt is None:
        dt = datetime.now()
    return Candle(
        time=dt,
        open=float(raw.get("open", 0.0) or 0.0),
        high=float(raw.get("high", 0.0) or 0.0),
        low=float(raw.get("low", 0.0) or 0.0),
        close=float(raw.get("close", 0.0) or 0.0),
        volume=float(raw.get("volume", 0.0) or 0.0),
    )

def load_candles() -> list[Candle]:
    files = sorted(DATA_DIR.glob("candles_*.json"))
    all_raw = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            all_raw.extend(data.get("candles") or [])
        except Exception as exc:
            print(f"Warning: Failed to load {path.name}: {exc}")
    
    candles = [dict_to_candle(r) for r in all_raw]
    candles = candles[::10]
    return candles

class MemoryTracker:
    def __init__(self):
        self.peak_memory = 0.0
        self.active = False
        self.thread = None
        self.process = psutil.Process(os.getpid())
        
    def _monitor(self):
        while self.active:
            try:
                mem = self.process.memory_info().rss / (1024 * 1024) # MB
                if mem > self.peak_memory:
                    self.peak_memory = mem
            except Exception:
                pass
            time.sleep(0.01)
            
    def __enter__(self):
        self.peak_memory = 0.0
        self.active = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.active = False
        if self.thread:
            self.thread.join()

def replicate_candles(candles: list[Candle], target_count: int) -> list[Candle]:
    replicated = []
    base_time = datetime(2025, 1, 1, 9, 15, 0)
    while len(replicated) < target_count:
        offset = len(replicated)
        for c in candles:
            new_c = Candle(
                time=base_time + timedelta(minutes=5 * offset),
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                volume=c.volume
            )
            replicated.append(new_c)
            offset += 1
            if len(replicated) == target_count:
                break
    return replicated

def generate_contexts(candles: list[Candle]) -> list[MLFeatureContext]:
    test_contexts = []
    for i, c in enumerate(candles):
        time_sin = np.sin(2 * np.pi * (c.time.hour * 60 + c.time.minute) / 1440.0)
        time_cos = np.cos(2 * np.pi * (c.time.hour * 60 + c.time.minute) / 1440.0)
        iv = 0.15 + 0.05 * np.sin(i / 10.0)
        delta = 0.5 * np.cos(i / 10.0)
        ctx = MLFeatureContext(
            regime="trending" if i % 2 == 0 else "mean_reverting",
            iv=iv,
            iv_change_pct=0.01 * np.sin(i),
            iv_percentile=50.0 + 10.0 * np.cos(i),
            delta=delta,
            gamma=0.02 * np.sin(i),
            vega=0.05 * np.cos(i),
            theta=-0.1 * np.sin(i),
            spot=c.close * 1.001,
            option_price=c.close * 0.02,
            time_sin=time_sin,
            time_cos=time_cos,
            dte_norm=2.5,
        )
        test_contexts.append(ctx)
    return test_contexts

def run_extended_validation():
    print("Loading candles for extended validation...")
    base_candles = load_candles()
    
    periods = [
        ("3 Months", 2250),
        ("6 Months", 4500),
        ("1 Year", 9000)
    ]
    
    for name, size in periods:
        print("\n" + "="*60)
        print(f"EXTENDED VALIDATION: {name} (size: {size} candles)")
        print("="*60)
        
        test_candles = replicate_candles(base_candles, size)
        test_contexts = generate_contexts(test_candles)
        
        for use_tb in [False, True]:
            tb_str = "Triple Barrier" if use_tb else "Standard Labels"
            print(f"\n--- Testing with {tb_str} ---")
            
            # Run V1
            t0 = time.time()
            X_old, y_old, names_old = build_supervised_dataset(
                test_candles,
                contexts=test_contexts,
                lookback=20,
                horizon=5,
                use_triple_barrier=use_tb,
                tb_profit_target_pct=0.01,
                tb_stop_loss_pct=0.005
            )
            t_v1 = time.time() - t0
            
            # Run V2
            t0 = time.time()
            X_new, y_new, names_new = build_supervised_dataset_v2(
                test_candles,
                contexts=test_contexts,
                lookback=20,
                horizon=5,
                use_triple_barrier=use_tb,
                tb_profit_target_pct=0.01,
                tb_stop_loss_pct=0.005
            )
            t_v2 = time.time() - t0
            
            # Assertions & Parity Check
            name_match = (names_old == names_new)
            label_matches = np.sum(np.array(y_old) == np.array(y_new))
            label_match_rate = label_matches / len(y_old) if len(y_old) > 0 else 1.0
            
            X_old_arr = np.array(X_old)
            X_new_arr = np.array(X_new)
            abs_errors = np.abs(X_old_arr - X_new_arr)
            max_abs_error = np.max(abs_errors)
            mean_abs_error = np.mean(abs_errors)
            
            print(f"Sample Count:        {len(X_old)}")
            print(f"Feature Count:       {len(names_old)}")
            print(f"Feature Name Match:  {100.0 if name_match else 0.0:.2f}%")
            print(f"Label Match Rate:    {label_match_rate * 100.0:.2f}%")
            print(f"Max Absolute Error:  {max_abs_error:.4e}")
            print(f"Mean Absolute Error: {mean_abs_error:.4e}")
            print(f"Runtime V1:          {t_v1:.4f}s")
            print(f"Runtime V2:          {t_v2:.4f}s")
            print(f"Speedup Factor:      {t_v1 / t_v2:.2f}x" if t_v2 > 0 else "N/A")

def run_retraining_benchmark():
    print("\n" + "#"*60)
    print("RUNNING RETRAINING BENCHMARKS")
    print("#"*60)
    
    base_candles = load_candles()
    
    benchmarks = [
        ("1 Year", 9000),
        ("2 Years", 18000)
    ]
    
    for name, size in benchmarks:
        print("\n" + "="*60)
        print(f"BENCHMARK: {name} Retraining (size: {size} candles)")
        print("="*60)
        
        test_candles = replicate_candles(base_candles, size)
        test_contexts = generate_contexts(test_candles)
        
        # We pre-generate a dummy label array for isolating feature generation
        dummy_labels = [0] * len(test_candles)
        
        # -------------------------------------------------------------------
        # V1 Retraining Pipeline
        # -------------------------------------------------------------------
        print("\nRunning V1 Pipeline...")
        
        with MemoryTracker() as tracker:
            # 1. Feature Generation Time (use pre-generated labels to isolate feature calculation)
            t0 = time.time()
            X_v1, _, names_v1 = build_supervised_dataset(
                test_candles,
                contexts=test_contexts,
                labels=dummy_labels,
                lookback=20,
                horizon=5
            )
            t_feat_v1 = time.time() - t0
            
            # 2. Label Generation Time (Run dataset builder with Triple Barrier)
            t0 = time.time()
            _, y_v1, _ = build_supervised_dataset(
                test_candles,
                contexts=test_contexts,
                lookback=20,
                horizon=5,
                use_triple_barrier=True,
                tb_profit_target_pct=0.01,
                tb_stop_loss_pct=0.005
            )
            t_label_v1 = time.time() - t0 - t_feat_v1 # subtract feature gen time to isolate
            if t_label_v1 < 0:
                t_label_v1 = 0.0
            
            # 3. Walk-Forward Validation Time
            t0 = time.time()
            wf_v1 = walk_forward_backtest(X_v1, y_v1, n_splits=5)
            t_wf_v1 = time.time() - t0
            
            # 4. Total Retraining Time (Dataset build + WF + Final Model Train)
            t0 = time.time()
            X_full_v1, y_full_v1, _ = build_supervised_dataset(
                test_candles,
                contexts=test_contexts,
                lookback=20,
                horizon=5,
                use_triple_barrier=True
            )
            walk_forward_backtest(X_full_v1, y_full_v1, n_splits=5)
            train_ensemble(X_full_v1, y_full_v1, feature_names=names_v1)
            t_total_v1 = time.time() - t0
            
            ram_v1 = tracker.peak_memory
            
        print(f"V1 Feature Gen Time:  {t_feat_v1:.4f}s")
        print(f"V1 Label Gen Time:    {t_label_v1:.4f}s")
        print(f"V1 Walk-Forward Time: {t_wf_v1:.4f}s")
        print(f"V1 Total Retrain:     {t_total_v1:.4f}s")
        print(f"V1 Peak RAM Usage:    {ram_v1:.2f} MB")
        
        # -------------------------------------------------------------------
        # V2 Retraining Pipeline
        # -------------------------------------------------------------------
        print("\nRunning V2 Pipeline...")
        
        with MemoryTracker() as tracker:
            # 1. Feature Generation Time (use pre-generated labels to isolate feature calculation)
            t0 = time.time()
            X_v2, _, names_v2 = build_supervised_dataset_v2(
                test_candles,
                contexts=test_contexts,
                labels=dummy_labels,
                lookback=20,
                horizon=5
            )
            t_feat_v2 = time.time() - t0
            
            # 2. Label Generation Time (Isolate compute_triple_barrier_labels_vectorized)
            closes = [c.close for c in test_candles]
            t0 = time.time()
            y_v2 = compute_triple_barrier_labels_vectorized(
                closes=closes,
                lookback=20,
                horizon=5,
                use_triple_barrier=True,
                tb_profit_target_pct=0.01,
                tb_stop_loss_pct=0.005
            )
            t_label_v2 = time.time() - t0
            
            # 3. Walk-Forward Validation Time
            t0 = time.time()
            wf_v2 = walk_forward_backtest(X_v2, y_v2, n_splits=5)
            t_wf_v2 = time.time() - t0
            
            # 4. Total Retraining Time (Dataset build + WF + Final Model Train)
            t0 = time.time()
            X_full_v2, y_full_v2, _ = build_supervised_dataset_v2(
                test_candles,
                contexts=test_contexts,
                lookback=20,
                horizon=5,
                use_triple_barrier=True
            )
            walk_forward_backtest(X_full_v2, y_full_v2, n_splits=5)
            train_ensemble(X_full_v2, y_full_v2, feature_names=names_v2)
            t_total_v2 = time.time() - t0
            
            ram_v2 = tracker.peak_memory
            
        print(f"V2 Feature Gen Time:  {t_feat_v2:.4f}s")
        print(f"V2 Label Gen Time:    {t_label_v2:.4f}s")
        print(f"V2 Walk-Forward Time: {t_wf_v2:.4f}s")
        print(f"V2 Total Retrain:     {t_total_v2:.4f}s")
        print(f"V2 Peak RAM Usage:    {ram_v2:.2f} MB")
        
        # Speedups
        print("\n--- Pipeline Speedup Factors ---")
        print(f"Feature Gen Speedup:  {t_feat_v1 / t_feat_v2:.2f}x" if t_feat_v2 > 0 else "N/A")
        print(f"Label Gen Speedup:    {t_label_v1 / t_label_v2:.2f}x" if t_label_v2 > 0 else "N/A")
        print(f"Walk-Forward Speedup: {t_wf_v1 / t_wf_v2:.2f}x" if t_wf_v2 > 0 else "N/A")
        print(f"Total Retrain Speed:  {t_total_v1 / t_total_v2:.2f}x" if t_total_v2 > 0 else "N/A")
        print(f"Peak RAM Delta:       {ram_v2 - ram_v1:.2f} MB")

if __name__ == "__main__":
    run_extended_validation()
    run_retraining_benchmark()
