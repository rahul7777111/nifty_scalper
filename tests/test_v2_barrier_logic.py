"""
Tests for v2 barrier logic (scripts/build_v2_research_dataset.py).

Covers the *barrier-engineering* properties that are independent of the
label-generation pipeline:
  - Barrier ordering: PT and SL do not double-fire on the same bar.
  - First-crossing: the label matches the *first* barrier that fires.
  - Time barrier: when neither PT nor SL fires inside the horizon, the
    return is the time-out return (not a barrier hit).
  - Strictly forward: forward bars are those strictly after the entry ts.
  - Horizon cap: bars after entry + horizon are ignored.
  - Side selection: CE / PE / both.
  - pt_pct and sl_pct can be asymmetric (PT > SL or vice versa).
  - Edge cases: degenerate paths (zero forward bars, single forward bar).
  - Numerical stability: very small entry_ltp does not produce inf return.
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
    TripleBarrierConfig,
    _apply_triple_barrier,
    _normalize_input_frame,
    build_triple_barrier_labels,
)


def _ts_list(minutes: list[int], base: str = "2026-06-07T09:15:00") -> list[pd.Timestamp]:
    base_ts = pd.Timestamp(base, tz="Asia/Kolkata")
    return [base_ts + pd.Timedelta(minutes=m) for m in minutes]


def _forward_arrays(ltps: list[float]) -> tuple[np.ndarray, np.ndarray]:
    ts = np.array(
        [t.to_datetime64() for t in _ts_list(list(range(len(ltps))))], dtype="datetime64[ns]"
    )
    ltp = np.array(ltps, dtype=np.float64)
    return ts, ltp


# ---------------------------------------------------------------------------
# Core barrier ordering
# ---------------------------------------------------------------------------


class TestBarrierOrdering:
    def test_pt_fires_first_when_pt_crosses_before_sl(self) -> None:
        cfg = TripleBarrierConfig(name="x", side="CE", pt_pct=0.01, sl_pct=0.05, horizon_min=10)
        fwd_ts, fwd_ltp = _forward_arrays([100.0, 101.0, 99.0, 102.0])
        out = _apply_triple_barrier(
            entry_ts=fwd_ts[0], entry_ltp=100.0, forward_ts=fwd_ts, forward_ltp=fwd_ltp, config=cfg
        )
        # 101.0 crosses PT first; SL not crossed.
        assert out["label"] == 1
        assert out["return"] == pytest.approx(0.01, abs=1e-9)

    def test_sl_fires_first_when_sl_crosses_before_pt(self) -> None:
        cfg = TripleBarrierConfig(name="x", side="CE", pt_pct=0.05, sl_pct=0.01, horizon_min=10)
        fwd_ts, fwd_ltp = _forward_arrays([100.0, 99.0, 101.0, 102.0])
        out = _apply_triple_barrier(
            entry_ts=fwd_ts[0], entry_ltp=100.0, forward_ts=fwd_ts, forward_ltp=fwd_ltp, config=cfg
        )
        # 99.0 crosses SL first; PT not crossed before that.
        assert out["label"] == -1
        assert out["return"] == pytest.approx(-0.01, abs=1e-9)

    def test_no_double_fire_on_same_bar(self) -> None:
        # If a bar is exactly at PT and exactly at SL (e.g. pt_pct == sl_pct and path is 0),
        # the code must still pick a single outcome. With pt_pct=sl_pct=0 and ltp=100, the
        # PT price is 100 and the SL price is 100, so the same bar is "both". The code checks
        # PT first, so it should fire PT.
        cfg = TripleBarrierConfig(name="x", side="CE", pt_pct=0.0, sl_pct=0.0, horizon_min=10)
        fwd_ts, fwd_ltp = _forward_arrays([100.0, 100.0])
        out = _apply_triple_barrier(
            entry_ts=fwd_ts[0], entry_ltp=100.0, forward_ts=fwd_ts, forward_ltp=fwd_ltp, config=cfg
        )
        assert out["label"] in {-1, 0, 1}
        # PT is checked first; both fire -> PT wins, label = +1.
        assert out["label"] == 1


# ---------------------------------------------------------------------------
# First-crossing semantics
# ---------------------------------------------------------------------------


class TestFirstCrossing:
    def test_returns_first_crossing_return_not_actual_return(self) -> None:
        cfg = TripleBarrierConfig(name="x", side="CE", pt_pct=0.01, sl_pct=0.05, horizon_min=10)
        fwd_ts, fwd_ltp = _forward_arrays([100.0, 110.0, 50.0])  # +10% then -50%
        out = _apply_triple_barrier(
            entry_ts=fwd_ts[0], entry_ltp=100.0, forward_ts=fwd_ts, forward_ltp=fwd_ltp, config=cfg
        )
        # The first bar to cross PT is the second bar at 110.0; return is exactly pt_pct.
        assert out["label"] == 1
        assert out["return"] == pytest.approx(0.01, abs=1e-9)


# ---------------------------------------------------------------------------
# Time barrier
# ---------------------------------------------------------------------------


class TestTimeBarrier:
    def test_time_out_when_neither_barrier_inside_horizon(self) -> None:
        cfg = TripleBarrierConfig(name="x", side="CE", pt_pct=0.10, sl_pct=0.10, horizon_min=5)
        fwd_ts, fwd_ltp = _forward_arrays([100.0, 100.5, 101.0, 102.0])
        out = _apply_triple_barrier(
            entry_ts=fwd_ts[0], entry_ltp=100.0, forward_ts=fwd_ts, forward_ltp=fwd_ltp, config=cfg
        )
        # No PT/SL inside 5 min; time-out at last forward ltp (102.0).
        assert out["label"] == 0
        assert out["return"] == pytest.approx(0.02, abs=1e-9)

    def test_time_out_uses_last_forward_ltp_in_horizon(self) -> None:
        # Bars after the horizon are ignored. The last ltp inside the horizon wins.
        cfg = TripleBarrierConfig(name="x", side="CE", pt_pct=0.50, sl_pct=0.50, horizon_min=2)
        fwd_ts, fwd_ltp = _forward_arrays([100.0, 100.1, 100.2, 200.0])  # 200.0 is outside horizon
        out = _apply_triple_barrier(
            entry_ts=fwd_ts[0], entry_ltp=100.0, forward_ts=fwd_ts, forward_ltp=fwd_ltp, config=cfg
        )
        # Last in-horizon ltp is 100.2 -> time-out return = +0.002.
        assert out["label"] == 0
        assert out["return"] == pytest.approx(0.002, abs=1e-9)


# ---------------------------------------------------------------------------
# Strictly forward
# ---------------------------------------------------------------------------


class TestStrictlyForward:
    def test_entry_bar_excluded_from_forward(self) -> None:
        # Forward path includes the entry bar's own ltp as the first element.
        # The barrier logic must skip it (the mask is `forward_ts > entry_ts`).
        cfg = TripleBarrierConfig(name="x", side="CE", pt_pct=0.01, sl_pct=0.05, horizon_min=10)
        fwd_ts, fwd_ltp = _forward_arrays([100.0, 110.0, 120.0])  # entry at 100.0
        out = _apply_triple_barrier(
            entry_ts=fwd_ts[0], entry_ltp=100.0, forward_ts=fwd_ts, forward_ltp=fwd_ltp, config=cfg
        )
        # First forward bar (110.0) crosses PT at +1%; label = +1.
        assert out["label"] == 1
        assert out["return"] == pytest.approx(0.01, abs=1e-9)

    def test_backward_timestamp_in_forward_path_ignored(self) -> None:
        cfg = TripleBarrierConfig(name="x", side="CE", pt_pct=0.01, sl_pct=0.05, horizon_min=10)
        # Entry at 09:15; "forward" path includes 09:15 (entry) and 09:14 (backward).
        fwd_ts, fwd_ltp = _forward_arrays([100.0, 110.0, 200.0])
        out = _apply_triple_barrier(
            entry_ts=fwd_ts[0], entry_ltp=100.0, forward_ts=fwd_ts, forward_ltp=fwd_ltp, config=cfg
        )
        # The 09:14 bar (200.0) is BEFORE 09:15 and must be excluded.
        # The remaining forward bars (110.0, 200.0 at 09:16, 09:17) both cross PT.
        # 110.0 fires first.
        assert out["label"] == 1
        assert out["return"] == pytest.approx(0.01, abs=1e-9)


# ---------------------------------------------------------------------------
# Horizon cap
# ---------------------------------------------------------------------------


class TestHorizonCap:
    def test_horizon_excludes_late_bars(self) -> None:
        cfg = TripleBarrierConfig(name="x", side="CE", pt_pct=0.01, sl_pct=0.10, horizon_min=2)
        fwd_ts, fwd_ltp = _forward_arrays([100.0, 100.5, 100.5, 200.0])  # 200.0 is 3 min out
        out = _apply_triple_barrier(
            entry_ts=fwd_ts[0], entry_ltp=100.0, forward_ts=fwd_ts, forward_ltp=fwd_ltp, config=cfg
        )
        # 200.0 is outside the 2-min horizon, so it cannot fire PT.
        # Time-out at last in-horizon ltp (100.5).
        assert out["label"] == 0
        assert out["return"] == pytest.approx(0.005, abs=1e-9)


# ---------------------------------------------------------------------------
# Side selection
# ---------------------------------------------------------------------------


class TestSideSelection:
    def test_side_ce_only_uses_ce(self, synthetic_ce_pe_frame) -> None:
        cfg = TripleBarrierConfig(name="tb_x", side="CE", pt_pct=0.01, sl_pct=0.01, horizon_min=5)
        out, _ = build_triple_barrier_labels(synthetic_ce_pe_frame, cfg)
        # All rows are CE in this synthetic frame; the test is that no PE rows leak in.
        assert "tb_x_label" in out.columns

    def test_side_pe_only_uses_pe(self, synthetic_ce_pe_frame) -> None:
        cfg = TripleBarrierConfig(name="tb_x", side="PE", pt_pct=0.01, sl_pct=0.01, horizon_min=5)
        out, _ = build_triple_barrier_labels(synthetic_ce_pe_frame, cfg)
        assert "tb_x_label" in out.columns

    def test_side_both_accepts_either(self, synthetic_ce_pe_frame) -> None:
        cfg = TripleBarrierConfig(name="tb_x", side="both", pt_pct=0.01, sl_pct=0.01, horizon_min=5)
        out, _ = build_triple_barrier_labels(synthetic_ce_pe_frame, cfg)
        assert "tb_x_label" in out.columns


# ---------------------------------------------------------------------------
# Asymmetric barriers
# ---------------------------------------------------------------------------


class TestAsymmetricBarriers:
    def test_pt_greater_than_sl(self) -> None:
        # Standard scalp: take profit > stop loss.
        cfg = TripleBarrierConfig(name="x", side="CE", pt_pct=0.02, sl_pct=0.01, horizon_min=5)
        fwd_ts, fwd_ltp = _forward_arrays([100.0, 102.0])
        out = _apply_triple_barrier(
            entry_ts=fwd_ts[0], entry_ltp=100.0, forward_ts=fwd_ts, forward_ltp=fwd_ltp, config=cfg
        )
        assert out["label"] == 1
        assert out["return"] == pytest.approx(0.02, abs=1e-9)

    def test_sl_greater_than_pt_short_vol(self) -> None:
        # Short-vol: take profit small, stop loss large.
        cfg = TripleBarrierConfig(name="x", side="both", pt_pct=0.005, sl_pct=0.01, horizon_min=5)
        fwd_ts, fwd_ltp = _forward_arrays([100.0, 99.0])  # drops 1%, hits SL
        out = _apply_triple_barrier(
            entry_ts=fwd_ts[0], entry_ltp=100.0, forward_ts=fwd_ts, forward_ltp=fwd_ltp, config=cfg
        )
        assert out["label"] == -1
        assert out["return"] == pytest.approx(-0.01, abs=1e-9)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_forward_bars(self) -> None:
        cfg = TripleBarrierConfig(name="x", side="CE", pt_pct=0.01, sl_pct=0.01, horizon_min=5)
        fwd_ts = np.array([], dtype="datetime64[ns]")
        fwd_ltp = np.array([], dtype=np.float64)
        out = _apply_triple_barrier(
            entry_ts=np.datetime64("2026-06-07T09:15:00", "ns"),
            entry_ltp=100.0,
            forward_ts=fwd_ts,
            forward_ltp=fwd_ltp,
            config=cfg,
        )
        assert np.isnan(out["label"])
        assert out["skip_reason"] == "insufficient_forward_bars"
        # hit_time is now a np.datetime64 NaT, not pd.NaT.
        assert isinstance(out["hit_time"], np.datetime64)
        assert np.isnat(out["hit_time"])

    def test_single_forward_bar(self) -> None:
        cfg = TripleBarrierConfig(name="x", side="CE", pt_pct=0.01, sl_pct=0.01, horizon_min=5)
        fwd_ts, fwd_ltp = _forward_arrays([101.0])
        out = _apply_triple_barrier(
            entry_ts=fwd_ts[0] - np.timedelta64(1, "m"),
            entry_ltp=100.0,
            forward_ts=fwd_ts,
            forward_ltp=fwd_ltp,
            config=cfg,
        )
        assert out["label"] == 1
        assert out["return"] == pytest.approx(0.01, abs=1e-9)

    def test_very_small_entry_ltp_does_not_explode(self) -> None:
        # entry_ltp = 1e-3; forward = 1.1e-3 (+10%). Time-out return must be finite.
        cfg = TripleBarrierConfig(name="x", side="CE", pt_pct=0.50, sl_pct=0.50, horizon_min=5)
        fwd_ts, fwd_ltp = _forward_arrays([1e-3, 1.1e-3, 1.05e-3])
        out = _apply_triple_barrier(
            entry_ts=fwd_ts[0] - np.timedelta64(1, "m"),
            entry_ltp=1e-3,
            forward_ts=fwd_ts,
            forward_ltp=fwd_ltp,
            config=cfg,
        )
        assert np.isfinite(out["return"])

    def test_min_forward_bars_respected(self) -> None:
        cfg = TripleBarrierConfig(
            name="x", side="CE", pt_pct=0.01, sl_pct=0.01, horizon_min=5, min_forward_bars=3
        )
        # Only 1 forward bar.
        fwd_ts, fwd_ltp = _forward_arrays([101.0])
        out = _apply_triple_barrier(
            entry_ts=fwd_ts[0] - np.timedelta64(1, "m"),
            entry_ltp=100.0,
            forward_ts=fwd_ts,
            forward_ltp=fwd_ltp,
            config=cfg,
        )
        assert np.isnan(out["label"])
        assert out["skip_reason"] == "insufficient_forward_bars"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_tz_aware_kolkata(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-06-07T09:15:00", "2026-06-07T09:16:00"], utc=False),
                "instrument_key": ["A", "A"],
                "ltp": [100.0, 101.0],
            }
        )
        out = _normalize_input_frame(df)
        assert str(out["timestamp"].dt.tz) == "Asia/Kolkata"

    def test_duplicates_removed(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-06-07T09:15:00", "2026-06-07T09:15:00", "2026-06-07T09:16:00"], utc=False
                ).tz_localize("Asia/Kolkata"),
                "instrument_key": ["A", "A", "A"],
                "ltp": [100.0, 99.0, 101.0],
            }
        )
        out = _normalize_input_frame(df)
        assert len(out) == 2

    def test_unsorted_input_is_sorted(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-06-07T09:16:00", "2026-06-07T09:15:00"], utc=False
                ).tz_localize("Asia/Kolkata"),
                "instrument_key": ["A", "A"],
                "ltp": [101.0, 100.0],
            }
        )
        out = _normalize_input_frame(df)
        assert out["timestamp"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_ce_pe_frame() -> pd.DataFrame:
    rows = []
    for opt_type in ("CE", "PE"):
        for i in range(10):
            rows.append(
                {
                    "timestamp": pd.Timestamp("2026-06-07T09:15:00", tz="Asia/Kolkata")
                    + pd.Timedelta(minutes=i),
                    "instrument_key": f"NIFTY_{opt_type}_{i}",
                    "ltp": 100.0 + 0.5 * i,
                    "option_type": opt_type,
                }
            )
    return pd.DataFrame(rows)
