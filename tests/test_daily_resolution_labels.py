"""
Tests for daily-resolution V2 labels (scripts/build_v2_research_dataset.py).

Covers the dynamic-horizon path where the input frame is sampled at one
observation per trading day (bar_frequency_minutes ~ 1440). Verifies:
  - A 1-year synthetic daily option-chain frame can be built.
  - The inter-bar delta is ~1440 min (i.e. one trading day).
  - ``default_daily_config()`` produces a V2DatasetConfig whose TB configs
    express horizons in *bar counts* (horizon_bars), not just minutes.
  - ``build_v2_labels`` returns >= 90% non-NaN TB labels on a clean daily
    frame.
  - The leakage audit on the labeled frame reports ``ok: True``.
  - For non-NaN labels, hit_time - entry is bounded by
    ``horizon_bars * bar_frequency_minutes`` minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_v2_research_dataset import (  # noqa: E402
    default_daily_config,
    build_v2_labels,
    run_leakage_audit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_daily_frame(
    n_instruments: int = 2,
    n_days: int = 250,
    start: str = "2026-01-02",
    seed: int = 17,
) -> pd.DataFrame:
    """Build a 1-year daily option-chain frame.

    Uses ``np.busday_offset`` to generate ~250 business days anchored to the
    start date. Produces one row per (instrument, day). LTP is a low-vol
    random walk so the triple-barrier machinery has a non-trivial path to
    chew on.
    """
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    # pd.bdate_range emits Mon-Fri business days; that is the same calendar
    # that np.busday_offset / busday_count use. We use bdate_range for the
    # actual timestamp list since it returns tz-friendly date objects.
    bday_dates = pd.bdate_range(start=start_ts, periods=n_days).date
    frames = []
    for inst in range(n_instruments):
        instrument_key = f"NIFTY_DAILY_{inst}"
        ltp = np.empty(n_days)
        ltp[0] = 100.0 + inst
        for i in range(1, n_days):
            ltp[i] = max(ltp[i - 1] * (1.0 + 0.005 * rng.standard_normal()), 1e-6)
        timestamps = [
            pd.Timestamp(d, tz="Asia/Kolkata") + pd.Timedelta(hours=15, minutes=30)
            for d in bday_dates
        ]
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "instrument_key": instrument_key,
                "ltp": ltp,
                "open": ltp,
                "high": ltp * 1.005,
                "low": ltp * 0.995,
                "volume": rng.integers(100, 1000, size=n_days),
                "oi": rng.integers(1000, 10000, size=n_days),
                "option_type": "CE" if inst % 2 == 0 else "PE",
                "primary_signal_proba": rng.uniform(0.3, 0.8, size=n_days),
            }
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def daily_df() -> pd.DataFrame:
    return _make_synthetic_daily_frame()


# ---------------------------------------------------------------------------
# Inter-bar delta
# ---------------------------------------------------------------------------


class TestInterBarDelta:
    def test_inter_bar_delta_is_one_trading_day(self, daily_df: pd.DataFrame) -> None:
        """The median inter-bar delta in minutes should be ~1440 (one day)."""
        df = daily_df.sort_values(["instrument_key", "timestamp"]).copy()
        deltas_min = (
            df.groupby("instrument_key")["timestamp"]
            .diff()
            .dropna()
            .dt.total_seconds()
            .div(60.0)
        )
        median_min = float(deltas_min.median())
        # bdate_range emits 24h gaps; allow a generous window for tz/timestamp
        # edge cases.
        assert 1400.0 <= median_min <= 1500.0, (
            f"expected ~1440 min between daily bars, got {median_min}"
        )

    def test_default_daily_config_has_1440_min_bars(self) -> None:
        """The daily preset must declare a 1440-min bar frequency."""
        cfg = default_daily_config()
        # The config may carry bar_frequency_minutes on the top-level
        # V2DatasetConfig, on each TB config, or both. Look for 1440 in any
        # of those places.
        top_level = getattr(cfg, "bar_frequency_minutes", None)
        tb_freqs = [getattr(tb, "bar_frequency_minutes", None) for tb in cfg.triple_barriers]
        # At least one of these must be 1440.
        values = [v for v in (top_level, *tb_freqs) if v is not None]
        assert 1440.0 in values, (
            f"expected a 1440-min bar frequency on the daily preset, got {values}"
        )


# ---------------------------------------------------------------------------
# build_v2_labels on a daily frame
# ---------------------------------------------------------------------------


class TestDailyBuildV2Labels:
    def test_at_least_90_pct_rows_have_non_nan_label(
        self, daily_df: pd.DataFrame
    ) -> None:
        cfg = default_daily_config()
        out, audit = build_v2_labels(daily_df, cfg)
        assert len(out) == len(daily_df)
        # For at least one TB config, the labeled rate is >= 90%.
        best_rate = 0.0
        for body in audit["label_audits"].values():
            if isinstance(body, dict) and "rows_labeled" in body and "rows_total" in body:
                if body["rows_total"] > 0:
                    best_rate = max(
                        best_rate, body["rows_labeled"] / body["rows_total"]
                    )
        assert best_rate >= 0.90, f"best labeled rate was {best_rate:.3f} < 0.90"

    def test_leakage_audit_passes(self, daily_df: pd.DataFrame) -> None:
        cfg = default_daily_config()
        out, _ = build_v2_labels(daily_df, cfg)
        report = run_leakage_audit(out)
        assert report["ok"], f"leakage audit failed: {report['errors']}"
        assert report["forbidden_present"] == []
        assert report["errors"] == []

    def test_hit_time_in_trading_days_not_minutes(
        self, daily_df: pd.DataFrame
    ) -> None:
        """For non-NaN labels, (hit_time - entry) is bounded by
        horizon_bars * bar_frequency_minutes, in minutes."""
        cfg = default_daily_config()
        out, _ = build_v2_labels(daily_df, cfg)
        # Get bar_frequency_minutes from the config.
        bf_min = getattr(cfg, "bar_frequency_minutes", 1440.0)
        assert bf_min == pytest.approx(1440.0, abs=1e-9)
        # Per-TB horizon budget in minutes.
        budgets = {tb.name: int(tb.horizon_bars) * float(bf_min) for tb in cfg.triple_barriers}
        entry_ns = (
            np.asarray(out["timestamp"].array, dtype="datetime64[ns]")
            .view("i8")
            .copy()
        )
        for col in out.columns:
            if not (col.endswith("_hit_time") and col.startswith("tb_")):
                continue
            name = col[: -len("_hit_time")]
            label_col = f"{name}_label"
            if label_col not in out.columns or name not in budgets:
                continue
            label = pd.to_numeric(out[label_col], errors="coerce")
            hit_ns = (
                np.asarray(out[col].array, dtype="datetime64[ns]")
                .view("i8")
                .copy()
            )
            labeled = label.notna().to_numpy()
            hit_present = ~pd.isna(hit_ns)
            delta_min = (hit_ns - entry_ns) / 1e9 / 60.0
            within = delta_min <= float(budgets[name]) + 1.0
            bad = labeled & hit_present & ~within
            assert int(bad.sum()) == 0, (
                f"{col}: {int(bad.sum())} hit_times exceed "
                f"horizon_bars * bar_frequency_minutes"
            )
