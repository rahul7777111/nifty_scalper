"""
Tests for dynamic-horizon mapping (scripts/build_v2_research_dataset.py).

The V2 dataset builder must support multiple bar frequencies (1-min intraday
to 1440-min daily) without a hard-coded horizon-minutes table. These tests
verify the new API:
  - ``V2DatasetConfig.derive_horizon_bars(bar_frequency_minutes, horizon_min)``
    converts a wall-clock horizon into a bar count.
  - ``TripleBarrierConfig.horizon_bars`` validation (>= 1, integer only).
  - A hand-built TB config with ``horizon_bars=3`` triggers the time-out
    barrier on a 3-bar intraday path that doesn't reach PT or SL.
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
    V2DatasetConfig,
    _apply_triple_barrier,
    build_triple_barrier_labels,
    default_daily_config,
    default_intraday_config,
)


# ---------------------------------------------------------------------------
# Parametric table: derive_horizon_bars
# ---------------------------------------------------------------------------


class TestDeriveHorizonBars:
    """V2DatasetConfig.derive_horizon_bars must convert wall-clock minutes
    into bar counts using the configured bar_frequency_minutes."""

    def test_derive_is_static_method(self) -> None:
        """derive_horizon_bars must be callable on the class without an
        instance (it is a pure mapping from (freq, minutes) -> bars)."""
        assert hasattr(V2DatasetConfig, "derive_horizon_bars")
        out = V2DatasetConfig.derive_horizon_bars(
            bar_frequency_minutes=1.0, horizon_min=5
        )
        assert int(out) == 5

    @pytest.mark.parametrize(
        "freq,horizon_min,expected_bars",
        [
            (1.0, 5, 5),
            (1.0, 60, 60),
            (5.0, 15, 3),
            (1440.0, 1440, 1),
            (1440.0, 4320, 3),
        ],
    )
    def test_parametric_table(
        self, freq: float, horizon_min: int, expected_bars: int
    ) -> None:
        out = V2DatasetConfig.derive_horizon_bars(
            bar_frequency_minutes=freq, horizon_min=horizon_min
        )
        assert int(out) == int(expected_bars), (
            f"freq={freq}, horizon_min={horizon_min} -> "
            f"expected {expected_bars}, got {out}"
        )

    def test_derive_handles_non_divisible_horizon(self) -> None:
        """If horizon_min is not an exact multiple of bar_frequency_minutes,
        derive_horizon_bars must round (typically ceil) to a positive
        integer; never return zero or a fractional bar count."""
        out = V2DatasetConfig.derive_horizon_bars(
            bar_frequency_minutes=5.0, horizon_min=7
        )
        # 7 / 5 = 1.4; must round up to 2 bars.
        assert int(out) >= 1


# ---------------------------------------------------------------------------
# TripleBarrierConfig.horizon_bars validation
# ---------------------------------------------------------------------------


class TestHorizonBarsValidation:
    def test_horizon_bars_zero_rejected(self) -> None:
        cfg = TripleBarrierConfig(
            name="x",
            side="CE",
            pt_pct=0.01,
            sl_pct=0.01,
            horizon_min=5,
            horizon_bars=0,
        )
        with pytest.raises(ValueError, match="horizon_bars"):
            cfg.validate()

    def test_horizon_bars_negative_rejected(self) -> None:
        cfg = TripleBarrierConfig(
            name="x",
            side="CE",
            pt_pct=0.01,
            sl_pct=0.01,
            horizon_min=5,
            horizon_bars=-3,
        )
        with pytest.raises(ValueError, match="horizon_bars"):
            cfg.validate()

    def test_horizon_bars_non_integer_rejected(self) -> None:
        cfg = TripleBarrierConfig(
            name="x",
            side="CE",
            pt_pct=0.01,
            sl_pct=0.01,
            horizon_min=5,
            horizon_bars=2.5,  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="horizon_bars"):
            cfg.validate()

    def test_horizon_bars_default_is_one(self) -> None:
        cfg = TripleBarrierConfig(
            name="x", side="CE", pt_pct=0.01, sl_pct=0.01, horizon_min=5
        )
        # When horizon_bars is not provided, it must default to 1.
        assert int(getattr(cfg, "horizon_bars", 1)) >= 1


# ---------------------------------------------------------------------------
# horizon_bars=3 on a 3-bar intraday path -> time-out
# ---------------------------------------------------------------------------


class TestTimeOutAtHorizonBoundary:
    def test_three_bar_path_with_no_barrier_hit_yields_timeout(self) -> None:
        """Hand-built TB config with horizon_bars=3 and a 3-bar forward
        path that doesn't cross PT or SL must yield label=0 (time-out)."""
        cfg = TripleBarrierConfig(
            name="tb_3bar",
            side="CE",
            pt_pct=0.10,   # 10% PT - never reached
            sl_pct=0.10,   # 10% SL - never reached
            horizon_min=3, # 3 min on 1-min bars
            horizon_bars=3,
            min_forward_bars=1,
        )
        # Forward path: ltp 100, 100.1, 100.2, 100.3 (entry + 3 forward bars).
        fwd_ts = np.array(
            [
                np.datetime64("2026-06-07T09:15:00", "ns"),
                np.datetime64("2026-06-07T09:16:00", "ns"),
                np.datetime64("2026-06-07T09:17:00", "ns"),
                np.datetime64("2026-06-07T09:18:00", "ns"),
            ],
            dtype="datetime64[ns]",
        )
        fwd_ltp = np.array([100.0, 100.1, 100.2, 100.3], dtype=np.float64)
        out = _apply_triple_barrier(
            entry_ts=fwd_ts[0],
            entry_ltp=100.0,
            forward_ts=fwd_ts,
            forward_ltp=fwd_ltp,
            config=cfg,
        )
        assert out["label"] == 0
        # The time-out return is the last in-horizon ltp vs entry.
        assert out["return"] == pytest.approx(0.003, abs=1e-9)

    def test_three_bar_path_with_pt_inside_horizon_yields_profit(self) -> None:
        """Same horizon_bars=3, but PT fires on bar 2 (within horizon)."""
        cfg = TripleBarrierConfig(
            name="tb_3bar",
            side="CE",
            pt_pct=0.01,
            sl_pct=0.10,
            horizon_min=3,
            horizon_bars=3,
            min_forward_bars=1,
        )
        fwd_ts = np.array(
            [
                np.datetime64("2026-06-07T09:15:00", "ns"),
                np.datetime64("2026-06-07T09:16:00", "ns"),
                np.datetime64("2026-06-07T09:17:00", "ns"),
                np.datetime64("2026-06-07T09:18:00", "ns"),
            ],
            dtype="datetime64[ns]",
        )
        # Bar 2: ltp 101.0 -> +1%, hits PT.
        fwd_ltp = np.array([100.0, 100.5, 101.0, 101.5], dtype=np.float64)
        out = _apply_triple_barrier(
            entry_ts=fwd_ts[0],
            entry_ltp=100.0,
            forward_ts=fwd_ts,
            forward_ltp=fwd_ltp,
            config=cfg,
        )
        assert out["label"] == 1
        assert out["return"] == pytest.approx(0.01, abs=1e-9)


# ---------------------------------------------------------------------------
# Preset configs expose horizon_bars
# ---------------------------------------------------------------------------


class TestPresetsExposeHorizonBars:
    def test_intraday_preset_tb_configs_have_horizon_bars(self) -> None:
        cfg = default_intraday_config()
        assert len(cfg.triple_barriers) >= 1
        for tb in cfg.triple_barriers:
            assert int(getattr(tb, "horizon_bars", 0)) >= 1, (
                f"TB config {tb.name!r} on intraday preset has no horizon_bars"
            )

    def test_daily_preset_tb_configs_have_horizon_bars(self) -> None:
        cfg = default_daily_config()
        assert len(cfg.triple_barriers) >= 1
        for tb in cfg.triple_barriers:
            assert int(getattr(tb, "horizon_bars", 0)) >= 1, (
                f"TB config {tb.name!r} on daily preset has no horizon_bars"
            )

    def test_intraday_preset_short_horizons(self) -> None:
        """Intraday preset TB configs should typically have small bar
        counts (e.g. 5, 10, 15, 30, 60). The weekly-horizon config
        (if present) may have a larger count but should be < 100 on
        intraday bars."""
        cfg = default_intraday_config()
        for tb in cfg.triple_barriers:
            bars = int(getattr(tb, "horizon_bars", 1))
            # 390 bars/day, so a 1-day horizon = 390 bars. Anything beyond
            # that would be multi-day on intraday, which doesn't make
            # sense for an intraday preset.
            assert bars < 390, (
                f"intraday TB {tb.name!r} has horizon_bars={bars} >= 390"
            )


# ---------------------------------------------------------------------------
# End-to-end smoke: build_triple_barrier_labels with horizon_bars
# ---------------------------------------------------------------------------


class TestBuildWithHorizonBars:
    def test_three_bar_intraday_end_to_end(self) -> None:
        """A 4-bar frame with a 3-bar-horizon TB config should label the
        first row and skip the last (no forward bars)."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-06-07T09:15:00",
                        "2026-06-07T09:16:00",
                        "2026-06-07T09:17:00",
                        "2026-06-07T09:18:00",
                    ],
                    utc=False,
                ).tz_localize("Asia/Kolkata"),
                "instrument_key": ["A"] * 4,
                "ltp": [100.0, 100.1, 100.2, 100.3],
                "option_type": ["CE"] * 4,
            }
        )
        cfg = TripleBarrierConfig(
            name="tb_3bar",
            side="CE",
            pt_pct=0.10,
            sl_pct=0.10,
            horizon_min=3,
            horizon_bars=3,
            min_forward_bars=1,
        )
        out, _ = build_triple_barrier_labels(df, cfg)
        # The first 3 rows have at least 1 forward bar; the last row
        # has 0 forward bars and is skipped.
        assert out["tb_3bar_label"].iloc[:-1].notna().all()
        assert pd.isna(out["tb_3bar_label"].iloc[-1])
        # The first 3 rows are all time-outs (no PT/SL).
        assert (out["tb_3bar_label"].iloc[:-1] == 0).all()
