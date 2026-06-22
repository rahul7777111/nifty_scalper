"""
Tests for intraday-resolution V2 labels (scripts/build_v2_research_dataset.py).

Covers the dynamic-horizon path where the input frame is sampled at 1-minute
bars within a single NSE trading session (09:15 - 15:30 = 375-390 minutes).
Verifies:
  - A 1-day intraday frame (2 instruments, ~390 1-min bars) can be built.
  - ``default_intraday_config()`` produces a V2DatasetConfig whose TB configs
    declare ``bar_frequency_minutes=1.0``.
  - ``build_v2_labels`` returns >= 90% non-NaN TB labels on a clean intraday
    frame (allowing for the trailing horizon gap where rows have no forward
    bars in the same session).
  - The leakage audit on the labeled frame reports ``ok: True``.
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
    default_intraday_config,
    build_v2_labels,
    run_leakage_audit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


NSE_SESSION_START = pd.Timestamp("2026-06-07T09:15:00", tz="Asia/Kolkata")
NSE_BARS_PER_DAY = 390  # typical NSE equity-derivatives session length


def _make_synthetic_intraday_frame(
    n_instruments: int = 2,
    n_bars: int = NSE_BARS_PER_DAY,
    seed: int = 21,
) -> pd.DataFrame:
    """Build a 1-day intraday option-context frame (1-min bars)."""
    rng = np.random.default_rng(seed)
    frames = []
    for inst in range(n_instruments):
        instrument_key = f"NIFTY_INTRA_{inst}"
        ltp = np.empty(n_bars)
        ltp[0] = 100.0 + inst
        for i in range(1, n_bars):
            ltp[i] = max(ltp[i - 1] * (1.0 + 0.0005 * rng.standard_normal()), 1e-6)
        timestamps = [NSE_SESSION_START + pd.Timedelta(minutes=i) for i in range(n_bars)]
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "instrument_key": instrument_key,
                "ltp": ltp,
                "open": ltp,
                "high": ltp * 1.005,
                "low": ltp * 0.995,
                "volume": rng.integers(100, 1000, size=n_bars),
                "oi": rng.integers(1000, 10000, size=n_bars),
                "option_type": "CE" if inst % 2 == 0 else "PE",
                "primary_signal_proba": rng.uniform(0.3, 0.8, size=n_bars),
            }
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def intraday_df() -> pd.DataFrame:
    return _make_synthetic_intraday_frame()


# ---------------------------------------------------------------------------
# Intraday bar frequency
# ---------------------------------------------------------------------------


class TestIntradayBarFrequency:
    def test_default_intraday_config_has_1_min_bars(self) -> None:
        """The intraday preset must declare a 1-min bar frequency."""
        cfg = default_intraday_config()
        top_level = getattr(cfg, "bar_frequency_minutes", None)
        tb_freqs = [getattr(tb, "bar_frequency_minutes", None) for tb in cfg.triple_barriers]
        values = [v for v in (top_level, *tb_freqs) if v is not None]
        assert 1.0 in values, (
            f"expected a 1-min bar frequency on the intraday preset, got {values}"
        )

    def test_intraday_frame_is_1_min_spaced(self, intraday_df: pd.DataFrame) -> None:
        df = intraday_df.sort_values(["instrument_key", "timestamp"]).copy()
        deltas_min = (
            df.groupby("instrument_key")["timestamp"]
            .diff()
            .dropna()
            .dt.total_seconds()
            .div(60.0)
        )
        median_min = float(deltas_min.median())
        assert median_min == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# build_v2_labels on an intraday frame
# ---------------------------------------------------------------------------


class TestIntradayBuildV2Labels:
    def test_at_least_90_pct_rows_have_non_nan_label(
        self, intraday_df: pd.DataFrame
    ) -> None:
        cfg = default_intraday_config()
        out, audit = build_v2_labels(intraday_df, cfg)
        assert len(out) == len(intraday_df)
        # Allow some trailing rows to be unlabeled (within the horizon).
        # At least one TB config should still have >= 90% labeled.
        best_rate = 0.0
        for body in audit["label_audits"].values():
            if isinstance(body, dict) and "rows_labeled" in body and "rows_total" in body:
                if body["rows_total"] > 0:
                    best_rate = max(
                        best_rate, body["rows_labeled"] / body["rows_total"]
                    )
        assert best_rate >= 0.90, f"best labeled rate was {best_rate:.3f} < 0.90"

    def test_leakage_audit_passes(self, intraday_df: pd.DataFrame) -> None:
        cfg = default_intraday_config()
        out, _ = build_v2_labels(intraday_df, cfg)
        report = run_leakage_audit(out)
        assert report["ok"], f"leakage audit failed: {report['errors']}"
        assert report["forbidden_present"] == []
        assert report["errors"] == []

    def test_intraday_horizon_within_bar_budget(
        self, intraday_df: pd.DataFrame
    ) -> None:
        """For non-NaN labels, hit_time - entry is bounded by
        horizon_bars * 1.0 minutes (with a 1-minute slack)."""
        cfg = default_intraday_config()
        out, _ = build_v2_labels(intraday_df, cfg)
        bf_min = getattr(cfg, "bar_frequency_minutes", 1.0)
        assert bf_min == pytest.approx(1.0, abs=1e-9)
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
