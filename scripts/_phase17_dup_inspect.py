"""
Phase 17: inspect the 1.216 rows per (instrument, date) ratio.
Streaming pass to see what duplicates are: same ik + same date but different rows.
"""
import os
import json
import pandas as pd
from collections import Counter

REPO = r"C:\Users\rahul\Downloads\niftyscalper-current"
os.chdir(REPO)

PATH = "data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv"

print(f"Streaming {PATH} to find duplicate (ik, date) rows ...")
counter = Counter()
n_rows = 0
ex_dup_samples = []
chunksize = 100_000
for chunk in pd.read_csv(PATH, chunksize=chunksize, low_memory=False):
    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=False)
    chunk = chunk.dropna(subset=["timestamp", "instrument_key"])
    chunk["_date"] = chunk["timestamp"].dt.date
    pairs = list(zip(chunk["instrument_key"].tolist(), chunk["_date"].tolist(), chunk["timestamp"].tolist()))
    n_rows += len(chunk)
    for k in pairs:
        counter[k] += 1

# Group by (ik, date) ignoring timestamp; count rows
pair_no_ts = Counter()
for (ik, d, _), c in counter.items():
    pair_no_ts[(ik, d)] += c

# Distribution of rows-per-(ik,date)
counts = list(pair_no_ts.values())
from collections import Counter as C
dist = C(counts)
print(f"  total rows: {n_rows:,}")
print(f"  unique (ik, date) pairs: {len(pair_no_ts):,}")
print(f"  rows-per-(ik, date) distribution:")
for k in sorted(dist.keys())[:5]:
    print(f"     {k} rows: {dist[k]:,} pairs")
# How many pairs have > 1 row?
multi = sum(c for k, c in dist.items() if k > 1)
print(f"  pairs with >1 row: {multi:,} ({100*multi/len(pair_no_ts):.2f}%)")

# Show examples of pairs with > 1 row
print("\nSample (ik, date) with >1 row (first 5):")
shown = 0
for (ik, d), c in pair_no_ts.items():
    if c > 1 and shown < 5:
        # find all (timestamp, count) for this (ik, d)
        rows_here = [(t, ct) for (ik2, d2, t), ct in counter.items() if ik2 == ik and d2 == d]
        print(f"  ik={ik}, date={d}, rows={c}")
        for t, ct in rows_here[:6]:
            print(f"     ts={t}, count={ct}")
        shown += 1

# Save
out = {
    "total_rows": n_rows,
    "unique_pairs": len(pair_no_ts),
    "rows_per_pair_distribution": {str(k): v for k, v in sorted(dist.items())},
    "pct_multi_row_pairs": round(100 * multi / len(pair_no_ts), 3),
}
with open(r"C:\Users\rahul\Downloads\niftyscalper-current\reports\_phase17_dup_inspect.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved.")
