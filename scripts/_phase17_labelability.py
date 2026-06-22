"""
Phase 17 labelability: simulate the V2 builder's per-(instrument, date) forward
horizon coverage at 1/2/5/10/20-bar horizons.

Read-only. Runs on a 20k sample of the cost-aware dataset.
"""
import os
import json
import pandas as pd
import numpy as np

REPO = r"C:\Users\rahul\Downloads\niftyscalper-current"
os.chdir(REPO)

PATH = "data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv"
SAMPLE_N = 20000
HORIZONS = [1, 2, 5, 10, 20]

print(f"Loading {SAMPLE_N} rows from {PATH} ...")
df = pd.read_csv(PATH, nrows=SAMPLE_N, low_memory=False)
print(f"  loaded: {df.shape}")

# Coerce timestamp and find date col
ts_col = "timestamp" if "timestamp" in df.columns else None
if ts_col is None:
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower():
            ts_col = c
            break
print(f"  ts_col: {ts_col}")
df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=False)
df = df.dropna(subset=[ts_col])

ik_col = "instrument_key" if "instrument_key" in df.columns else None
print(f"  ik_col: {ik_col}")
if ik_col is None:
    for c in df.columns:
        if "instrument" in c.lower() or "symbol" in c.lower():
            ik_col = c
            break
print(f"  resolved ik_col: {ik_col}")

# Build (instrument_key, date) groups; within each, count rows in time order.
df["_date"] = df[ts_col].dt.date
df = df.sort_values([ik_col, ts_col]).reset_index(drop=True)
print(f"  unique instruments: {df[ik_col].nunique()}")
print(f"  unique dates: {df['_date'].nunique()}")
print(f"  rows after dedup: {len(df)}")

# For each horizon, mark rows as labelable if they have at least N subsequent rows
# in the same (instrument, date) group.
print("\n=== Labelability by horizon (per (instrument, date) intra-day group) ===")
results = []
total_rows = len(df)
for h in HORIZONS:
    # groupby (ik, date) and count the row index within the group
    g = df.groupby([ik_col, "_date"], sort=False)
    pos = g.cumcount()
    size = g[ts_col].transform("size")
    # labelable if position + h < size
    labelable = (pos + h) < size
    pct = 100.0 * labelable.sum() / total_rows
    print(f"  horizon={h:2d} bars: labelable={int(labelable.sum())}/{total_rows} = {pct:.1f}%")
    results.append({"horizon": h, "labelable": int(labelable.sum()), "total": total_rows, "pct": round(pct, 2)})

# Now also compute labelability in the "next trading day" sense: 1 bar = next row
# in the same instrument (across days).
print("\n=== Labelability by horizon (per instrument, across days) ===")
df2 = df.sort_values([ik_col, ts_col]).reset_index(drop=True)
g2 = df2.groupby(ik_col, sort=False)
pos2 = g2.cumcount()
size2 = g2[ts_col].transform("size")
results2 = []
for h in HORIZONS:
    labelable = (pos2 + h) < size2
    pct = 100.0 * labelable.sum() / total_rows
    print(f"  horizon={h:2d} bars: labelable={int(labelable.sum())}/{total_rows} = {pct:.1f}%")
    results2.append({"horizon": h, "labelable": int(labelable.sum()), "total": total_rows, "pct": round(pct, 2)})

out = r"C:\Users\rahul\Downloads\niftyscalper-current\reports\_phase17_labelability.json"
with open(out, "w") as f:
    json.dump({"intra_day": results, "across_days": results2, "sample_n": SAMPLE_N, "path": PATH}, f, indent=2)
print(f"\nSaved: {out}")
