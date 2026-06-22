"""
Phase 17: trading-day calendar coverage.
"""
import os
import json
import pandas as pd
import numpy as np

REPO = r"C:\Users\rahul\Downloads\niftyscalper-current"
os.chdir(REPO)

PATH = "data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv"

print(f"Streaming {PATH} for unique trading dates ...")
dates = set()
chunksize = 100_000
for chunk in pd.read_csv(PATH, usecols=["timestamp"], chunksize=chunksize, low_memory=False):
    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=False)
    chunk = chunk.dropna(subset=["timestamp"])
    dates.update(chunk["timestamp"].dt.date.unique().tolist())
print(f"  unique dates: {len(dates)}")
sorted_dates = sorted(dates)
print(f"  first: {sorted_dates[0]}")
print(f"  last:  {sorted_dates[-1]}")

# Calendar gaps (consecutive missing weekdays)
ds = pd.DatetimeIndex(sorted_dates)
ds_idx = pd.DatetimeIndex(sorted_dates)
expected = pd.bdate_range(ds_idx.min(), ds_idx.max())  # business days
expected_set = set(expected.date)
actual_set = set(ds_idx.date)
missing_bd = sorted(expected_set - actual_set)
print(f"  business days in range: {len(expected_set)}")
print(f"  missing business days: {len(missing_bd)}")
if missing_bd[:10]:
    print(f"  first 10 missing business days: {missing_bd[:10]}")
if missing_bd[-10:]:
    print(f"  last 10 missing business days: {missing_bd[-10:]}")

# Compute interval between trading days
print(f"\nInterval between consecutive trading dates (in business days):")
intervals = []
for i in range(1, len(sorted_dates)):
    diff = (sorted_dates[i] - sorted_dates[i-1]).days
    intervals.append(diff)
arr = np.array(intervals)
print(f"  min={arr.min()}, median={np.median(arr):.0f}, p90={np.percentile(arr,90):.0f}, p99={np.percentile(arr,99):.0f}, max={arr.max()}")
# How many gaps > 5 days?
gaps_gt5 = (arr > 5).sum()
gaps_gt10 = (arr > 10).sum()
gaps_gt20 = (arr > 20).sum()
print(f"  gaps > 5 days: {gaps_gt5}")
print(f"  gaps > 10 days: {gaps_gt10}")
print(f"  gaps > 20 days: {gaps_gt20}")

out = {
    "path": PATH,
    "first_date": str(sorted_dates[0]),
    "last_date": str(sorted_dates[-1]),
    "unique_dates": len(dates),
    "business_days_in_range": len(expected_set),
    "missing_business_days": len(missing_bd),
    "first_10_missing_bd": [str(d) for d in missing_bd[:10]],
    "interval_days_min": int(arr.min()),
    "interval_days_median": int(np.median(arr)),
    "interval_days_p90": int(np.percentile(arr, 90)),
    "interval_days_p99": int(np.percentile(arr, 99)),
    "interval_days_max": int(arr.max()),
    "gaps_gt_5": int(gaps_gt5),
    "gaps_gt_10": int(gaps_gt10),
    "gaps_gt_20": int(gaps_gt20),
}
with open(r"C:\Users\rahul\Downloads\niftyscalper-current\reports\_phase17_calendar.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("Saved.")
