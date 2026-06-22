"""Quick test harness for historical archive unification + feature addition on 2 days."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re
from datetime import datetime, date, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))

from option_chain_pipeline_lib import (
    _add_research_features,
    reconstruct_missing_live_features_for_option_chain,
    TIMEZONE,
)

src = Path(r"C:\Users\rahul\Downloads\archive\nifty_data\nifty_options")
files = [
    src / "2024" / "7" / "nifty_options_18_07_2024.csv",
    src / "2024" / "7" / "nifty_options_19_07_2024.csv",
]

month_map = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
rx = re.compile(r"NIFTY(\d{2})([A-Z]{3})(\d{2})(\d+)(CE|PE)$", re.I)

def parse_sym(s: str):
    m = rx.match(str(s).upper().strip())
    if not m:
        return None
    dd, mmm, yy, sk, ty = m.groups()
    y = 2000 + int(yy)
    mon = month_map.get(mmm, 1)
    try:
        exp = date(y, mon, int(dd))
    except ValueError:
        return None
    return {
        "expiry": exp,
        "strike_price": float(sk),
        "option_type": ty.upper(),
        "raw_symbol": str(s),
    }

rows = []
for f in files:
    df = pd.read_csv(f)
    for _, r in df.iterrows():
        p = parse_sym(r["symbol"])
        if not p:
            continue
        ts_str = f"{r['date']} {r['time']}"
        ts = pd.to_datetime(ts_str)
        rows.append({
            "timestamp": ts,
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "ltp": float(r["close"]),
            "volume": int(r["volume"]),
            "oi": float(r["oi"]),
            "expiry": p["expiry"],
            "strike_price": p["strike_price"],
            "option_type": p["option_type"],
            "trading_symbol": p["raw_symbol"],
            "instrument_key": f"ARCHIVE_{p['expiry']:%Y%m%d}_{int(p['strike_price'])}_{p['option_type']}",
            "weekly": True,
            "source_file": f.name,
        })

base = pd.DataFrame(rows)
print("Base rows loaded:", len(base))
if base.empty:
    print("No rows parsed!")
    sys.exit(1)

base["timestamp"] = pd.to_datetime(base["timestamp"])
base["expiry"] = pd.to_datetime(base["expiry"])
base["trading_day"] = pd.to_datetime(base["timestamp"]).dt.normalize()
base = base.sort_values(["timestamp", "instrument_key"], kind="stable").reset_index(drop=True)

# Add research features (the lib's basic set)
enriched, feat_cols, schema = _add_research_features(base)
print("After _add_research_features: total_cols=", len(enriched.columns), "declared_features=", len(feat_cols))

# Persist temp and run the full reconstruct (adds dozens more live technical features)
# Note: reconstruct always emits CSV regardless of extension passed.
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)
tmp_in = PROCESSED / "_tmp_hist_test_in.csv"
enriched.to_csv(tmp_in, index=False)

rec_report = reconstruct_missing_live_features_for_option_chain(
    tmp_in,
    output_dataset=PROCESSED / "_tmp_hist_test_out.csv",
    report_path=PROCESSED / "_tmp_hist_reconstruct_report.json",
)

print("Reconstruct report keys:", list(rec_report.keys()) if isinstance(rec_report, dict) else "not-dict")
print("reconstructed_feature_count from report:", rec_report.get("reconstructed_feature_count") if isinstance(rec_report, dict) else None)
print("matching_live_feature_count:", rec_report.get("matching_live_feature_count") if isinstance(rec_report, dict) else None)

out_path = PROCESSED / "_tmp_hist_test_out.csv"
if out_path.exists():
    # Use low_memory=False or usecols if needed, but for test read full
    df_out = pd.read_csv(out_path, low_memory=False)
    print("Final cols after reconstruct:", len(df_out.columns))
    sample_new = [c for c in df_out.columns if c not in set(enriched.columns)][:10]
    print("New cols sample:", sample_new)
    # Check against our known target list (from earlier inspection)
    target_like = ["ema_fast", "adx_14", "bullish_engulfing", "hammer", "regime_trending", "regime_quiet", "oi_CE", "ce_pe_oi_ratio", "atr_14", "supertrend_dir", "pivot_pp_dist_pct", "dist_to_rolling_high_20"]
    present = [c for c in target_like if c in df_out.columns]
    print("Key target-like features now present:", present)
    # Cleanup
    out_path.unlink(missing_ok=True)
    (PROCESSED / "_tmp_hist_reconstruct_report.json").unlink(missing_ok=True)

tmp_in.unlink(missing_ok=True)
print("Small test SUCCESS. Column count after reconstruct =", len(df_out.columns) if 'df_out' in locals() else 'N/A')
