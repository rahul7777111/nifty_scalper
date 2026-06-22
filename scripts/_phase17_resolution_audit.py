"""
Phase 17 dataset-auditor: resolution (bar frequency) analysis.
Read-only. Writes results to stdout (caller captures into the report).
"""
import os
import sys
import json
import pandas as pd
import numpy as np

REPO = r"C:\Users\rahul\Downloads\niftyscalper-current"
os.chdir(REPO)

# Top 5 datasets by size (with desc)
FILES = [
    ("cost_aware_20260606_211845", "data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv"),
    ("cost_aware_20260606_210742", "data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_210742.csv"),
    ("bs_enriched",               "data/processed/nifty_option_chain_live_feature_reconstructed_20260604_100331_bs_enriched.csv"),
    ("live_feature_100331",       "data/processed/nifty_option_chain_live_feature_reconstructed_20260604_100331.csv"),
    ("live_feature_095301",       "data/processed/nifty_option_chain_live_feature_reconstructed_20260604_095301.csv"),
]

# 1. For each dataset, sample 5000 rows, find timestamp column, find instrument_key
TS_CANDIDATES = ["timestamp", "ts", "datetime", "date_time", "time", "date"]
IK_CANDIDATES = ["instrument_key", "instrument", "tradingsymbol", "symbol", "scrip", "option_key", "key"]

def find_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None

def analyze_dataset(name, path, sample_n=5000):
    print(f"\n=== {name} ===")
    print(f"path: {path}")
    if not os.path.exists(path):
        print("  MISSING")
        return None
    sz = os.path.getsize(path) / 1e6
    print(f"  size_MB: {sz:.1f}")
    try:
        df = pd.read_csv(path, nrows=sample_n, low_memory=False)
    except Exception as e:
        print(f"  read_error: {e}")
        return None
    nrows_sampled = len(df)
    ncols = df.shape[1]
    print(f"  rows_sampled: {nrows_sampled}, cols: {ncols}")
    cols = list(df.columns)
    ts_col = find_col(cols, TS_CANDIDATES)
    ik_col = find_col(cols, IK_CANDIDATES)
    print(f"  ts_col: {ts_col}, ik_col: {ik_col}")
    if ts_col is None:
        print("  no timestamp column found; columns sample:")
        print("   ", cols[:30])
        return None
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=False)
    n_valid_ts = df[ts_col].notna().sum()
    print(f"  valid_timestamps: {n_valid_ts}/{nrows_sampled}")
    if n_valid_ts < 100:
        return None
    min_ts = df[ts_col].min()
    max_ts = df[ts_col].max()
    print(f"  coverage: {min_ts} -> {max_ts}")

    # Unique timestamps per day (over the sample)
    df["_date"] = df[ts_col].dt.date
    tpd = df.groupby("_date")[ts_col].nunique()
    print(f"  unique_dates_in_sample: {len(tpd)}")
    if len(tpd):
        print(f"  unique_timestamps_per_day: min={tpd.min()}, median={tpd.median()}, max={tpd.max()}, mean={tpd.mean():.1f}")
        # Top 5 days
        print("  top days by timestamp count:")
        for d, c in tpd.sort_values(ascending=False).head(5).items():
            print(f"     {d}: {c}")

    # Per-instrument median inter-bar delta (minutes)
    if ik_col is None:
        print("  no instrument_key; skipping per-instrument delta")
        med_delta_min = None
    else:
        df = df.sort_values([ik_col, ts_col])
        per_inst = []
        for ik, g in df.groupby(ik_col):
            g = g.sort_values(ts_col).dropna(subset=[ts_col])
            if len(g) < 5:
                continue
            ts = g[ts_col].drop_duplicates().sort_values()
            if len(ts) < 5:
                continue
            deltas = ts.diff().dropna()
            if len(deltas) == 0:
                continue
            per_inst.append(deltas.median().total_seconds() / 60.0)
        if per_inst:
            per_inst_arr = np.array(per_inst)
            print(f"  per-instrument median inter-bar delta (min): n_inst={len(per_inst)}")
            print(f"     min={per_inst_arr.min():.3f}, median={np.median(per_inst_arr):.3f}, p25={np.percentile(per_inst_arr,25):.3f}, p75={np.percentile(per_inst_arr,75):.3f}, max={per_inst_arr.max():.3f}")
            med_delta_min = float(np.median(per_inst_arr))
        else:
            med_delta_min = None
            print("  could not compute per-instrument deltas (insufficient repeats)")

    # Missing-periods detection: gaps > 2x median delta (in minutes) in the global sorted timeline
    if med_delta_min and med_delta_min > 0:
        gap_thr = pd.Timedelta(minutes=2 * med_delta_min)
        ts_sorted = df[ts_col].drop_duplicates().sort_values()
        gaps = ts_sorted.diff()
        big = gaps[gaps > gap_thr]
        print(f"  missing_periods: {len(big)} gaps > 2x median delta ({gap_thr})")
        if len(big):
            for idx, g in big.head(5).items():
                print(f"     gap of {g} ending at {idx}")
    else:
        big = None

    return {
        "name": name,
        "path": path,
        "size_MB": round(sz, 1),
        "ts_col": ts_col,
        "ik_col": ik_col,
        "rows_sampled": nrows_sampled,
        "ncols": ncols,
        "min_ts": str(min_ts),
        "max_ts": str(max_ts),
        "unique_dates_in_sample": int(len(tpd)),
        "unique_timestamps_per_day_median": int(tpd.median()) if len(tpd) else None,
        "unique_timestamps_per_day_min": int(tpd.min()) if len(tpd) else None,
        "unique_timestamps_per_day_max": int(tpd.max()) if len(tpd) else None,
        "median_inter_bar_delta_min": round(med_delta_min, 3) if med_delta_min is not None else None,
        "missing_periods": int(len(big)) if big is not None else None,
        "intraday": (int(tpd.median()) > 1) if len(tpd) else False,
    }


def main():
    results = []
    for name, path in FILES:
        r = analyze_dataset(name, path)
        if r is not None:
            results.append(r)
    out = r"C:\Users\rahul\Downloads\niftyscalper-current\reports\_phase17_resolution_results.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
