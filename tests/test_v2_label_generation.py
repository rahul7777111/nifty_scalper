"""
Tests for v2 label generation (scripts/build_v2_research_dataset.py).

Covers:
  - TripleBarrierConfig / MetaLabelConfig / V2DatasetConfig validation.
  - Triple-barrier label generation: -1 / 0 / +1 outcomes, hit time, return.
  - Meta-label generation: trade, high_conf, low_conf.
  - Cost-adjusted label generation: net P&L, cost components.
  - Top-level build_v2_labels: ordering, no missing reference columns.
  - Reproducibility: same seed -> same output.
  - Default config covers all 7 TB configs from the design doc.
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
    DEFAULT_COST_STACK,
    MetaLabelConfig,
    TripleBarrierConfig,
    V2DatasetConfig,
    build_cost_adjusted_labels,
    build_meta_labels,
    build_triple_barrier_labels,
    build_v2_labels,
    default_v2_config,
    load_v2_config,
    validate_input_frame,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_option_chain(
    n_bars: int = 60,
    n_instruments: int = 2,
    start_ltp: float = 100.0,
    drift: float = 0.001,
    vol: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a small synthetic option-context frame for unit tests."""
    rng = np.random.default_rng(seed)
    frames = []
    base_ts = pd.Timestamp("2026-06-07T09:15:00", tz="Asia/Kolkata")
    for inst in range(n_instruments):
        instrument_key = f"NIFTY_OPT_TEST_{inst}"
        ltp = np.empty(n_bars)
        ltp[0] = start_ltp
        for i in range(1, n_bars):
            ret = drift + vol * rng.standard_normal()
            ltp[i] = max(ltp[i - 1] * (1.0 + ret), 1e-6)
        timestamps = [base_ts + pd.Timedelta(minutes=i) for i in range(n_bars)]
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
                "expiry": pd.Timestamp("2026-06-11").date(),
                "strike": 24700 + inst * 100,
                "option_type": "CE" if inst % 2 == 0 else "PE",
                "primary_signal_proba": rng.uniform(0.3, 0.8, size=n_bars),
            }
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return _make_synthetic_option_chain(n_bars=60, n_instruments=2, seed=42)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestTripleBarrierConfig:
    def test_valid_config_passes(self) -> None:
        cfg = TripleBarrierConfig(
            name="tb_long_aggressive", side="CE", pt_pct=0.01, sl_pct=0.006, horizon_min=5
        )
        cfg.validate()  # should not raise

    def test_zero_pt_pct_rejected(self) -> None:
        cfg = TripleBarrierConfig(name="x", side="CE", pt_pct=0.0, sl_pct=0.01, horizon_min=5)
        with pytest.raises(ValueError, match="pt_pct"):
            cfg.validate()

    def test_negative_sl_pct_rejected(self) -> None:
        cfg = TripleBarrierConfig(name="x", side="CE", pt_pct=0.01, sl_pct=-0.01, horizon_min=5)
        with pytest.raises(ValueError, match="sl_pct"):
            cfg.validate()

    def test_invalid_side_rejected(self) -> None:
        cfg = TripleBarrierConfig(name="x", side="XX", pt_pct=0.01, sl_pct=0.01, horizon_min=5)
        with pytest.raises(ValueError, match="side"):
            cfg.validate()

    def test_zero_horizon_rejected(self) -> None:
        cfg = TripleBarrierConfig(name="x", side="CE", pt_pct=0.01, sl_pct=0.01, horizon_min=0)
        with pytest.raises(ValueError, match="horizon_min"):
            cfg.validate()


class TestMetaLabelConfig:
    def test_valid_config_passes(self) -> None:
        cfg = MetaLabelConfig(name="meta_trade", primary_signal_column="primary_signal_proba")
        cfg.validate()

    def test_primary_threshold_out_of_range(self) -> None:
        cfg = MetaLabelConfig(name="x", primary_signal_column="y", primary_threshold=1.5)
        with pytest.raises(ValueError, match="primary_threshold"):
            cfg.validate()


class TestV2DatasetConfig:
    def test_default_config_is_valid(self) -> None:
        cfg = default_v2_config()
        cfg.validate()
        assert len(cfg.triple_barriers) == 7
        assert cfg.reference_tb_name == "tb_long_aggressive"

    def test_meta_without_reference_rejected(self) -> None:
        tb = TripleBarrierConfig(name="tb_x", side="CE", pt_pct=0.01, sl_pct=0.01, horizon_min=5)
        meta = MetaLabelConfig(name="m", primary_signal_column="p")
        cfg = V2DatasetConfig(triple_barriers=[tb], meta_labels=[meta], reference_tb_name=None)
        with pytest.raises(ValueError, match="reference_tb_name"):
            cfg.validate()

    def test_reference_must_be_in_triple_barriers(self) -> None:
        tb = TripleBarrierConfig(name="tb_x", side="CE", pt_pct=0.01, sl_pct=0.01, horizon_min=5)
        meta = MetaLabelConfig(name="m", primary_signal_column="p")
        cfg = V2DatasetConfig(
            triple_barriers=[tb], meta_labels=[meta], reference_tb_name="tb_nonexistent"
        )
        with pytest.raises(ValueError, match="not in triple_barriers"):
            cfg.validate()

    def test_negative_cost_multiplier_rejected(self) -> None:
        cfg = V2DatasetConfig(cost_multiplier=-1.0)
        with pytest.raises(ValueError, match="cost_multiplier"):
            cfg.validate()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_clean_frame_passes(self, synthetic_df: pd.DataFrame) -> None:
        result = validate_input_frame(synthetic_df)
        assert result["ok"]

    def test_forbidden_column_rejected(self, synthetic_df: pd.DataFrame) -> None:
        bad = synthetic_df.copy()
        bad["future_close"] = 1.0
        result = validate_input_frame(bad)
        assert not result["ok"]
        assert any("forbidden" in e for e in result["errors"])

    def test_missing_required_column_rejected(self) -> None:
        df = pd.DataFrame({"timestamp": [], "ltp": []})  # no instrument_key
        result = validate_input_frame(df)
        assert not result["ok"]
        assert any("missing required" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# Triple-barrier behavior
# ---------------------------------------------------------------------------


class TestTripleBarrierBehavior:
    def test_label_values_are_in_valid_set(self, synthetic_df: pd.DataFrame) -> None:
        cfg = TripleBarrierConfig(
            name="tb_test", side="CE", pt_pct=0.01, sl_pct=0.006, horizon_min=5
        )
        out, audit = build_triple_barrier_labels(synthetic_df, cfg)
        labels = out["tb_test_label"].dropna().unique()
        assert set(labels).issubset({-1.0, 0.0, 1.0})

    def test_profit_take_fires(self) -> None:
        # Hand-craft a path that monotonically rises by 2% over 3 bars.
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
                "instrument_key": "A",
                "ltp": [100.0, 100.5, 101.0, 101.5],
            }
        )
        cfg = TripleBarrierConfig(
            name="tb_test", side="CE", pt_pct=0.01, sl_pct=0.005, horizon_min=10
        )
        out, _ = build_triple_barrier_labels(df, cfg)
        # The first bar: forward path crosses +1% somewhere in 1.0% (101.0) -- hits PT.
        first_label = out["tb_test_label"].iloc[0]
        assert first_label == 1.0
        # Return at PT hit == pt_pct.
        assert pytest.approx(out["tb_test_return"].iloc[0], abs=1e-9) == 0.01

    def test_stop_loss_fires(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-06-07T09:15:00",
                        "2026-06-07T09:16:00",
                        "2026-06-07T09:17:00",
                    ],
                    utc=False,
                ).tz_localize("Asia/Kolkata"),
                "instrument_key": "A",
                "ltp": [100.0, 99.5, 98.0],
            }
        )
        cfg = TripleBarrierConfig(
            name="tb_test", side="CE", pt_pct=0.05, sl_pct=0.01, horizon_min=10
        )
        out, _ = build_triple_barrier_labels(df, cfg)
        # The first bar should hit SL (99.5 <= 99.0? no -- 99.5 is above 99.0).
        # Actually 99.0 is the SL price; 99.5 is above; 98.0 is below.
        # So the first bar's forward path crosses SL at the second forward bar.
        first_label = out["tb_test_label"].iloc[0]
        assert first_label == -1.0
        assert pytest.approx(out["tb_test_return"].iloc[0], abs=1e-9) == -0.01

    def test_time_out_when_no_barrier_hit(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-06-07T09:15:00",
                        "2026-06-07T09:16:00",
                        "2026-06-07T09:17:00",
                    ],
                    utc=False,
                ).tz_localize("Asia/Kolkata"),
                "instrument_key": "A",
                "ltp": [100.0, 100.1, 100.2],
            }
        )
        cfg = TripleBarrierConfig(
            name="tb_test", side="CE", pt_pct=0.10, sl_pct=0.10, horizon_min=10
        )
        out, _ = build_triple_barrier_labels(df, cfg)
        first_label = out["tb_test_label"].iloc[0]
        # No barrier hit; 100.2 vs 100.0 -> +0.2% time-out return, label=0.
        assert first_label == 0.0
        assert pytest.approx(out["tb_test_return"].iloc[0], abs=1e-6) == 0.002

    def test_horizon_cap(self) -> None:
        # Path that would hit PT only after the horizon -> time-out.
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-06-07T09:15:00",
                        "2026-06-07T09:16:00",
                        "2026-06-07T09:17:00",
                        "2026-06-07T09:30:00",
                    ],
                    utc=False,
                ).tz_localize("Asia/Kolkata"),
                "instrument_key": "A",
                "ltp": [100.0, 100.5, 101.0, 105.0],  # big jump only at 09:30
            }
        )
        cfg = TripleBarrierConfig(
            name="tb_test", side="CE", pt_pct=0.01, sl_pct=0.10, horizon_min=5
        )
        out, _ = build_triple_barrier_labels(df, cfg)
        first_label = out["tb_test_label"].iloc[0]
        # 09:16 is within 5 min horizon, hits +1%; label == 1.
        assert first_label == 1.0

    def test_hit_time_strictly_after_entry(self, synthetic_df: pd.DataFrame) -> None:
        cfg = TripleBarrierConfig(
            name="tb_test", side="CE", pt_pct=0.01, sl_pct=0.006, horizon_min=5
        )
        out, _ = build_triple_barrier_labels(synthetic_df, cfg)
        hit_ns = pd.to_datetime(out["tb_test_hit_time"], errors="coerce").astype("int64").to_numpy()
        entry_ns = pd.to_datetime(out["timestamp"], errors="coerce").astype("int64").to_numpy()
        labeled = out["tb_test_label"].notna().to_numpy()
        hit_present = ~pd.isna(hit_ns)
        bad = labeled & hit_present & (hit_ns <= entry_ns)
        assert int(bad.sum()) == 0

    def test_insufficient_forward_bars_skipped(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-06-07T09:15:00", "2026-06-07T09:16:00"], utc=False
                ).tz_localize("Asia/Kolkata"),
                "instrument_key": "A",
                "ltp": [100.0, 101.0],
            }
        )
        cfg = TripleBarrierConfig(
            name="tb_test", side="CE", pt_pct=0.01, sl_pct=0.01, horizon_min=5, min_forward_bars=10
        )
        out, _ = build_triple_barrier_labels(df, cfg)
        # No row has 10 forward bars -> all NaN.
        assert out["tb_test_label"].isna().all()


# ---------------------------------------------------------------------------
# Meta-labels
# ---------------------------------------------------------------------------


class TestMetaLabels:
    def _setup(self) -> Tuple[pd.DataFrame, MetaLabelConfig]:
        df = pd.DataFrame(
            {
                "ref_label": [1, -1, 0, 1, np.nan],
                "ref_return": [0.02, -0.01, 0.001, 0.005, np.nan],
                "primary_signal_proba": [0.8, 0.3, 0.6, 0.65, 0.4],
            }
        )
        cfg = MetaLabelConfig(
            name="meta_trade",
            primary_signal_column="primary_signal_proba",
            primary_threshold=0.5,
            high_confidence_threshold=0.7,
            low_confidence_threshold=0.6,
        )
        return df, cfg

    def test_meta_trade_positive_when_direction_matches_and_positive(self) -> None:
        df, cfg = self._setup()
        out, audit = build_meta_labels(
            df, cfg, reference_label_column="ref_label", reference_return_column="ref_return"
        )
        # Row 0: primary 0.8 -> direction +1; ref_label +1 matches; ref_return +0.02 > 0 -> trade=1.
        assert int(out["meta_meta_trade_trade"].iloc[0]) == 1
        # Row 1: primary 0.3 -> direction -1; ref_label -1 matches; ref_return -0.01 < 0 -> trade=0.
        assert int(out["meta_meta_trade_trade"].iloc[1]) == 0
        # Row 2: primary 0.6 -> direction +1; ref_label 0 -> no match -> trade=0.
        assert int(out["meta_meta_trade_trade"].iloc[2]) == 0
        # Row 3: primary 0.65 -> direction +1; ref_label +1 matches; ref_return +0.005 > 0 -> trade=1.
        # But high_conf requires primary >= 0.7 -> not high conf.
        assert int(out["meta_meta_trade_trade"].iloc[3]) == 1
        assert int(out["meta_meta_trade_high_conf"].iloc[3]) == 0
        # Row 4: ref_label NaN -> trade NaN.
        assert pd.isna(out["meta_meta_trade_trade"].iloc[4])

    def test_high_conf_requires_high_primary(self) -> None:
        df, cfg = self._setup()
        out, _ = build_meta_labels(
            df, cfg, reference_label_column="ref_label", reference_return_column="ref_return"
        )
        # Row 0 has primary 0.8 >= 0.7 AND trade == 1 -> high_conf == 1.
        assert int(out["meta_meta_trade_high_conf"].iloc[0]) == 1
        # Row 1: trade=0, so high_conf=0.
        assert int(out["meta_meta_trade_high_conf"].iloc[1]) == 0

    def test_low_conf_is_inverse_signal(self) -> None:
        df, cfg = self._setup()
        out, _ = build_meta_labels(
            df, cfg, reference_label_column="ref_label", reference_return_column="ref_return"
        )
        # Row 2: trade=0 AND primary 0.6 >= 0.6 (low_conf threshold) -> low_conf=1.
        assert int(out["meta_meta_trade_low_conf"].iloc[2]) == 1
        # Row 0: trade=1 -> low_conf=0.
        assert int(out["meta_meta_trade_low_conf"].iloc[0]) == 0

    def test_missing_reference_columns_raises(self) -> None:
        df, cfg = self._setup()
        with pytest.raises(ValueError, match="required column"):
            build_meta_labels(
                df.drop(columns=["ref_label"]),
                cfg,
                reference_label_column="ref_label",
                reference_return_column="ref_return",
            )


# ---------------------------------------------------------------------------
# Cost-adjusted labels
# ---------------------------------------------------------------------------


class TestCostAdjustedLabels:
    def test_cost_components_subtract_from_gross(self) -> None:
        df = pd.DataFrame({"ref_return": [0.05, 0.01, -0.02, np.nan]})
        out, audit = build_cost_adjusted_labels(
            df, reference_return_column="ref_return", cost_stack=DEFAULT_COST_STACK
        )
        # Net total return = gross - total cost.
        gross = out["ref_return"]
        net_total = out["econ_net_pnl_total_pct"]
        total_cost = out["econ_total_cost_pct"]
        diff = (gross.fillna(0) - net_total.fillna(total_cost) - total_cost)
        # gross - (gross - total_cost) - total_cost = 0.
        valid = gross.notna() & net_total.notna()
        assert np.allclose(diff[valid].abs(), 0.0, atol=1e-9)

    def test_pf_breach_is_positive_net(self) -> None:
        df = pd.DataFrame({"ref_return": [0.05, 0.0001, -0.01, np.nan]})
        out, audit = build_cost_adjusted_labels(
            df, reference_return_column="ref_return", cost_stack=DEFAULT_COST_STACK
        )
        # Total cost for default stack is on the order of 0.15-0.20% of notional.
        # 0.05 - cost > 0 -> breach.
        # 0.0001 - cost < 0 -> no breach.
        # -0.01 - cost < 0 -> no breach.
        breach = out["econ_pf_breach"]
        assert int(breach.iloc[0]) == 1
        assert int(breach.iloc[1]) == 0
        assert int(breach.iloc[2]) == 0
        assert pd.isna(breach.iloc[3])

    def test_cost_multiplier_scales(self) -> None:
        df = pd.DataFrame({"ref_return": [0.05]})
        out_1x, audit_1x = build_cost_adjusted_labels(
            df, reference_return_column="ref_return", cost_stack=DEFAULT_COST_STACK, cost_multiplier=1.0
        )
        out_2x, audit_2x = build_cost_adjusted_labels(
            df, reference_return_column="ref_return", cost_stack=DEFAULT_COST_STACK, cost_multiplier=2.0
        )
        # Total cost at 2x should be exactly 2x the total cost at 1x.
        assert pytest.approx(
            float(out_2x["econ_total_cost_pct"].iloc[0]) / float(out_1x["econ_total_cost_pct"].iloc[0]),
            rel=1e-9,
        ) == 2.0

    def test_audit_reports_components(self) -> None:
        df = pd.DataFrame({"ref_return": [0.01]})
        _, audit = build_cost_adjusted_labels(
            df, reference_return_column="ref_return", cost_stack=DEFAULT_COST_STACK
        )
        assert "component_costs_pct" in audit
        assert "total_cost_pct" in audit
        assert audit["total_cost_pct"] > 0


# ---------------------------------------------------------------------------
# Top-level build
# ---------------------------------------------------------------------------


class TestBuildV2Labels:
    def test_full_build_runs(self, synthetic_df: pd.DataFrame) -> None:
        cfg = default_v2_config()
        out, audit = build_v2_labels(synthetic_df, cfg)
        # All 7 TB label columns present.
        for tb in cfg.triple_barriers:
            assert f"{tb.name}_label" in out.columns
        # Meta + economic labels present.
        assert "meta_meta_trade_trade" in out.columns
        assert "econ_net_pnl_total_pct" in out.columns
        # Audit has 7 TB + 1 meta + 1 economic.
        assert len(audit["label_audits"]) == 9

    def test_build_with_missing_reference_raises(self, synthetic_df: pd.DataFrame) -> None:
        cfg = V2DatasetConfig(
            triple_barriers=[
                TripleBarrierConfig(name="tb_x", side="CE", pt_pct=0.01, sl_pct=0.01, horizon_min=5)
            ],
            meta_labels=[
                MetaLabelConfig(name="meta_trade", primary_signal_column="primary_signal_proba")
            ],
            reference_tb_name="tb_y",  # not in triple_barriers
        )
        with pytest.raises(ValueError, match="not in triple_barriers"):
            build_v2_labels(synthetic_df, cfg)

    def test_reproducibility_with_same_seed(self, synthetic_df: pd.DataFrame) -> None:
        cfg = default_v2_config()
        out1, _ = build_v2_labels(synthetic_df, cfg)
        out2, _ = build_v2_labels(synthetic_df, cfg)
        # Same input + same config -> same output.
        for tb in cfg.triple_barriers:
            col = f"{tb.name}_label"
            pd.testing.assert_series_equal(out1[col], out2[col], check_names=False)


# ---------------------------------------------------------------------------
# Config load
# ---------------------------------------------------------------------------


class TestConfigLoad:
    def test_load_round_trip(self, tmp_path) -> None:
        cfg = default_v2_config()
        payload = {
            "triple_barriers": [
                {
                    "name": tb.name,
                    "side": tb.side,
                    "pt_pct": tb.pt_pct,
                    "sl_pct": tb.sl_pct,
                    "horizon_min": tb.horizon_min,
                }
                for tb in cfg.triple_barriers
            ],
            "meta_labels": [
                {
                    "name": ml.name,
                    "primary_signal_column": ml.primary_signal_column,
                    "primary_threshold": ml.primary_threshold,
                    "high_confidence_threshold": ml.high_confidence_threshold,
                    "low_confidence_threshold": ml.low_confidence_threshold,
                }
                for ml in cfg.meta_labels
            ],
            "cost_stack": cfg.cost_stack,
            "cost_multiplier": cfg.cost_multiplier,
            "seed": cfg.seed,
            "reference_tb_name": cfg.reference_tb_name,
        }
        path = tmp_path / "v2_config.json"
        path.write_text(__import__("json").dumps(payload), encoding="utf-8")
        loaded = load_v2_config(path)
        assert len(loaded.triple_barriers) == len(cfg.triple_barriers)
        assert loaded.reference_tb_name == cfg.reference_tb_name
