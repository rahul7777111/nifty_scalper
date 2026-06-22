"""
Phase 17 quick stats: total rows, unique (instrument, date) pairs, trading days.
Counts via wc on the raw line count + a streaming pass with a 200k sample.
"""
import os
import json
import pandas as pd

REPO = r"C:\Users\rahul\Downloads\niftyscalper-current"
os.chdir(REPO)

PATH = "data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv"

# Streaming line count (more memory-friendly than full read)
print(f"Counting lines in {PATH} ...")
n_lines = 0
with open(PATH, "rb") as f:
    for _ in f:
        n_lines += 1
n_rows = n_lines - 1  # header
print(f"  total data rows: {n_rows:,}")

# Streaming pass for unique (instrument, date) and trading days
print("Streaming pass for unique counts ...")
seen_ik = set()
seen_dates = set()
seen_pairs = 0
pair_set = set()
n_seen = 0
chunksize = 100_000
for chunk in pd.read_csv(PATH, usecols=["instrument_key", "timestamp"], chunksize=chunksize, low_memory=False):
    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=False)
    chunk = chunk.dropna(subset=["timestamp", "instrument_key"])
    chunk["_date"] = chunk["timestamp"].dt.date
    seen_ik.update(chunk["instrument_key"].unique().tolist())
    seen_dates.update(chunk["_date"].unique().tolist())
    # unique (ik, date) pairs
    pairs = list(zip(chunk["instrument_key"].tolist(), chunk["_date"].tolist()))
    pair_set.update(pairs)
    n_seen += len(chunk)

print(f"  rows seen (after parse-drop): {n_seen:,}")
print(f"  unique instruments: {len(seen_ik)}")
print(f"  unique trading days: {len(seen_dates)}")
print(f"  unique (instrument, date) pairs: {len(pair_set):,}")
print(f"  rows per trading day (mean): {n_seen/len(seen_dates):.1f}")
print(f"  rows per (instrument, date) (mean): {n_seen/len(pair_set):.3f}")

out = {
    "path": PATH,
    "total_rows": int(n_rows),
    "rows_after_parse": int(n_seen),
    "unique_instruments": len(seen_ik),
    "unique_trading_days": len(seen_dates),
    "unique_pairs": len(pair_set),
    "rows_per_day_mean": round(n_seen/len(seen_dates), 1),
    "rows_per_pair_mean": round(n_seen/len(pair_set), 3),
}
with open(r"C:\Users\rahul\Downloads\niftyscalper-current\reports\_phase17_daily_count.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved.")
