"""
Phase 17: final summary — time-of-day distribution and exact bar count per (ik, date).
"""
import os
import json
import pandas as pd
from collections import Counter

REPO = r"C:\Users\rahul\Downloads\niftyscalper-current"
os.chdir(REPO)

PATH = "data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv"

# Time-of-day distribution: count rows by (hh, mm) on 100k sample
print(f"Sampling {PATH} for time-of-day distribution ...")
sample = pd.read_csv(PATH, nrows=200000, low_memory=False)
sample["timestamp"] = pd.to_datetime(sample["timestamp"], errors="coerce", utc=False)
sample = sample.dropna(subset=["timestamp"])
sample["_tod"] = sample["timestamp"].dt.strftime("%H:%M")
tod = sample["_tod"].value_counts()
print(f"  total sampled: {len(sample):,}")
print(f"  unique time-of-day bins: {len(tod)}")
print(f"  top 10 time-of-day bins:")
for k, v in tod.head(10).items():
    print(f"     {k}: {v}")

# Now exact (ik, date) bar count distribution across full file
print("\nStreaming full file for (ik, date) -> bar count distribution ...")
per_pair = Counter()
n = 0
chunksize = 100_000
for chunk in pd.read_csv(PATH, usecols=["instrument_key", "timestamp"], chunksize=chunksize, low_memory=False):
    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=False)
    chunk = chunk.dropna(subset=["timestamp", "instrument_key"])
    chunk["_date"] = chunk["timestamp"].dt.date
    pairs = list(zip(chunk["instrument_key"].tolist(), chunk["_date"].tolist()))
    for p in pairs:
        per_pair[p] += 1
    n += len(chunk)
print(f"  total rows: {n:,}")
print(f"  unique (ik, date) pairs: {len(per_pair):,}")
dist = Counter(per_pair.values())
print("  rows-per-(ik, date) distribution:")
for k in sorted(dist.keys()):
    print(f"     {k} bar(s): {dist[k]:,} pairs ({100*dist[k]/len(per_pair):.2f}%)")

# What fraction of (ik, date) has both a 00:00 and a 09:15 timestamp?
print("\nChecking for the dual-timestamp pattern (00:00 + 09:15) ...")
dual_count = 0
hh_per_pair = {}
chunksize = 100_000
for chunk in pd.read_csv(PATH, usecols=["instrument_key", "timestamp"], chunksize=chunksize, low_memory=False):
    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=False)
    chunk = chunk.dropna(subset=["timestamp", "instrument_key"])
    chunk["_date"] = chunk["timestamp"].dt.date
    chunk["_hh"] = chunk["timestamp"].dt.strftime("%H:%M")
    for ik, d, hh in zip(chunk["instrument_key"], chunk["_date"], chunk["_hh"]):
        key = (ik, d)
        if key not in hh_per_pair:
            hh_per_pair[key] = set()
        hh_per_pair[key].add(hh)
seen_hhs = Counter()
for k, v in hh_per_pair.items():
    seen_hhs[tuple(sorted(v))] += 1
print(f"  distinct (time-of-day-set) patterns across all (ik, date) pairs:")
for pattern, cnt in seen_hhs.most_common(10):
    print(f"     {pattern}: {cnt} pairs")
