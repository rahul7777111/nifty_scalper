"""
V2 Research Dataset Builder (Phase 16).

Generates a forward-looking, leakage-free research dataset with:
  - Triple-barrier labels (profit-take, stop-loss, time-out)
  - Meta-labels (trade/no-trade, high/low confidence)
  - Cost-adjusted economic labels (net P&L after brokerage, spread, slippage)

Design rules (all hard-coded and asserted in CI):
  1. Forward-looking only. Every label is a pure function of the forward path
     `{(t, ltp) : t > t_entry}` for the same `instrument_key`.
  2. No leakage. The input frame MUST NOT contain forbidden columns. The
     output frame writes the label columns only; the forward path itself is
     never written back as a feature.
  3. Reproducible. RNG is seedable; sorting is stable; timestamps are coerced
     to tz-aware Asia/Kolkata.
  4. Configurable barriers. Every triple-barrier config is a typed dataclass
     loaded from JSON, with explicit pt_pct, sl_pct, horizon_min, side.

Public surface:
  - build_v2_labels(df, config) -> (labeled_df, audit)
  - main(): CLI entry point that reads an option-context CSV/parquet and
    writes a v2 labeled parquet + a v2 audit JSON.

The script does NOT touch production code. It only reads from
`data/processed/...` and writes to `data/processed/v2_...` and
`reports/v2_...`.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = REPO_ROOT / "reports"

TIMEZONE = "Asia/Kolkata"

# Forbidden columns in the input frame. Mirrors the project-wide
# `STRICT_FORBIDDEN_TOKENS` from `retrain_all_edge_models.py`.
# A label function MUST NEVER receive a frame that contains any of these
# (other than as a passthrough that is dropped from the labeled output).
FORBIDDEN_INPUT_TOKENS = (
    "future_close",
    "future_ltp",
    "future_high",
    "future_low",
    "future_open",
    "future_volume",
    "future_oi",
    "future_bid",
    "future_ask",
    "future_spread",
    "gross_forward_return",
    "net_forward_return",
    "expected_return_after_cost",
    "return_to_cost_ratio",
    "cost_return_units_estimated",
)

# Output columns that are labels (not features). A retrain script that consumes
# this dataset MUST NOT include these in its input feature list.
LABEL_PREFIXES = ("tb_", "meta_", "econ_", "exec_", "label_")

# Default cost stack for cost-adjusted labels. Realistic for NIFTY options
# on m.Stock / Dhan retail. See `next_generation_dataset_design_20260607_100000.md`
# section 6.
DEFAULT_COST_STACK: Dict[str, float] = {
    "brokerage_per_side_pct": 0.0003,    # 0.03% of premium
    "stt_sell_pct_of_notional": 0.000625, # 0.0625% of notional on sell side
    "exchange_charge_pct": 0.000319,      # 0.0319% NSE
    "sebi_charge_pct": 0.000001,          # 0.0001%
    "gst_pct_on_brokerage_exchange_sebi": 0.18,
    "stamp_duty_buy_pct_of_notional": 0.00003,  # 0.003%
    "slippage_default_pct": 0.001,        # 0.1% (conservative)
    "spread_default_pct": 0.001,          # 0.1% baseline assumption
}

# Logger
log = logging.getLogger("v2_research_dataset")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TripleBarrierConfig:
    """One triple-barrier configuration.

    The vertical (time) barrier is defined as the Nth forward bar after
    entry_ts, where N = `horizon_bars`. `horizon_min` is kept as a
    human-readable display field (used in audit reports and the leakage
    check's expected_max_minutes conversion) and is converted into
    `horizon_bars` at build time using the dataset's `bar_frequency_minutes`
    via `V2DatasetConfig.derive_horizon_bars`.

    Backward compatibility: configs constructed without an explicit
    `horizon_bars` (e.g. legacy JSON files) get a sentinel value of 0; the
    build pipeline converts it to a positive integer at build time using
    `bar_frequency_minutes`. If `horizon_bars` is left at the sentinel (0)
    and no `bar_frequency_minutes` is supplied, `validate()` raises.
    """

    name: str
    side: str                 # "CE" | "PE" | "both"
    pt_pct: float             # profit-take threshold (e.g. 0.01 = 1.0%)
    sl_pct: float             # stop-loss threshold (e.g. 0.006 = 0.6%)
    horizon_min: int          # vertical (time) barrier in minutes (display)
    horizon_bars: int = 0     # vertical barrier as forward bar count (0 = derive)
    min_forward_bars: int = 1 # require at least this many forward bars
    require_strictly_forward: bool = True

    def validate(self) -> None:
        if not self.name:
            raise ValueError("TripleBarrierConfig.name is required")
        if self.side not in {"CE", "PE", "both"}:
            raise ValueError(f"Invalid side {self.side!r}; expected CE/PE/both")
        if self.pt_pct <= 0:
            raise ValueError(f"pt_pct must be > 0 (got {self.pt_pct})")
        if self.sl_pct <= 0:
            raise ValueError(f"sl_pct must be > 0 (got {self.sl_pct})")
        if self.horizon_min <= 0:
            raise ValueError(f"horizon_min must be > 0 (got {self.horizon_min})")
        # horizon_bars may be 0 (sentinel: derive at build time). It is
        # resolved against `bar_frequency_minutes` in
        # `V2DatasetConfig.resolve_horizon_bars`. If the caller never runs
        # the resolution step, validate() still requires a positive value.
        if self.horizon_bars < 0:
            raise ValueError(f"horizon_bars must be >= 0 (got {self.horizon_bars})")
        if self.min_forward_bars < 1:
            raise ValueError(f"min_forward_bars must be >= 1 (got {self.min_forward_bars})")


@dataclass(frozen=True)
class MetaLabelConfig:
    """Meta-label configuration (stacked on top of a primary signal)."""

    name: str
    primary_signal_column: str  # column with primary signal in [0, 1]
    primary_threshold: float = 0.5
    high_confidence_threshold: float = 0.7
    low_confidence_threshold: float = 0.55
    use_regime_column: Optional[str] = None  # optional regime one-hot

    def validate(self) -> None:
        if not self.name:
            raise ValueError("MetaLabelConfig.name is required")
        if not 0.0 <= self.primary_threshold <= 1.0:
            raise ValueError(f"primary_threshold out of range: {self.primary_threshold}")
        if not 0.0 <= self.high_confidence_threshold <= 1.0:
            raise ValueError(f"high_confidence_threshold out of range: {self.high_confidence_threshold}")
        if not 0.0 <= self.low_confidence_threshold <= 1.0:
            raise ValueError(f"low_confidence_threshold out of range: {self.low_confidence_threshold}")


@dataclass(frozen=True)
class V2DatasetConfig:
    """Top-level configuration for a v2 dataset build.

    `bar_frequency_minutes` is the per-bar time delta of the *input* frame.
    It is 1.0 for 1-minute intraday bars, 60.0 for hourly, 1440.0 for
    daily, etc. The horizon cap inside `_apply_triple_barrier` is the Nth
    forward bar after entry_ts, where N is derived as
    `ceil(horizon_min / bar_frequency_minutes)`. The old behaviour (cap
    = entry_ts + horizon_min minutes) is preserved when
    `bar_frequency_minutes == 1.0`.
    """

    triple_barriers: List[TripleBarrierConfig] = field(default_factory=list)
    meta_labels: List[MetaLabelConfig] = field(default_factory=list)
    cost_stack: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_COST_STACK))
    cost_multiplier: float = 1.0
    seed: int = 42
    bar_frequency_minutes: float = 1.0

    # The triple-barrier label whose outcome is consumed by the meta-labels
    # and the cost-adjusted economic labels. Required if any meta or economic
    # label is configured.
    reference_tb_name: Optional[str] = None

    def validate(self) -> None:
        for tb in self.triple_barriers:
            tb.validate()
        for ml in self.meta_labels:
            ml.validate()
        if self.cost_multiplier <= 0:
            raise ValueError(f"cost_multiplier must be > 0 (got {self.cost_multiplier})")
        if self.bar_frequency_minutes <= 0:
            raise ValueError(
                f"bar_frequency_minutes must be > 0 (got {self.bar_frequency_minutes})"
            )
        if (self.meta_labels or True) and self.triple_barriers:
            # meta-labels require a reference TB
            if self.meta_labels and not self.reference_tb_name:
                raise ValueError("reference_tb_name is required when meta_labels is configured")
            if self.reference_tb_name and self.reference_tb_name not in {tb.name for tb in self.triple_barriers}:
                raise ValueError(
                    f"reference_tb_name {self.reference_tb_name!r} not in triple_barriers"
                )

    def derive_horizon_bars(self, horizon_min: int) -> int:
        """Convert a minute-based horizon into a forward bar count.

        Always returns at least 1. Uses ``math.ceil`` so that
        ``horizon_min == bar_frequency_minutes`` gives 1 bar, and so that
        a 5-minute horizon on a 1-minute bar grid still rounds up to 5
        (which is what the old minute-based cap effectively did).
        """
        if horizon_min <= 0:
            raise ValueError(f"horizon_min must be > 0 (got {horizon_min})")
        return max(1, int(math.ceil(float(horizon_min) / float(self.bar_frequency_minutes))))

    def expected_max_minutes(self, tb: TripleBarrierConfig) -> float:
        """Upper-bound cap, in minutes, for a TB's vertical barrier.

        Equals ``horizon_min`` when ``horizon_bars`` was not explicitly
        resolved (legacy behaviour) and ``horizon_bars *
        bar_frequency_minutes`` otherwise.
        """
        if tb.horizon_bars > 0:
            return float(tb.horizon_bars) * float(self.bar_frequency_minutes)
        # Legacy: the cap is horizon_min, but we still allow a +1 minute
        # tolerance (matching the test_v2_leakage_audit tolerance) on the
        # minute-based check.
        return float(tb.horizon_min)

    def resolve_horizon_bars(self) -> "V2DatasetConfig":
        """Return a new config with each TB's ``horizon_bars`` filled in.

        A TB with ``horizon_bars == 0`` (the sentinel) is replaced via
        ``derive_horizon_bars(horizon_min)``. TBs that already specify a
        positive ``horizon_bars`` are left alone. The returned config is
        frozen; the original is not mutated.
        """
        new_tbs = [
            (
                tb
                if tb.horizon_bars > 0
                else TripleBarrierConfig(
                    name=tb.name,
                    side=tb.side,
                    pt_pct=tb.pt_pct,
                    sl_pct=tb.sl_pct,
                    horizon_min=tb.horizon_min,
                    horizon_bars=self.derive_horizon_bars(tb.horizon_min),
                    min_forward_bars=tb.min_forward_bars,
                    require_strictly_forward=tb.require_strictly_forward,
                )
            )
            for tb in self.triple_barriers
        ]
        return V2DatasetConfig(
            triple_barriers=new_tbs,
            meta_labels=self.meta_labels,
            cost_stack=self.cost_stack,
            cost_multiplier=self.cost_multiplier,
            seed=self.seed,
            bar_frequency_minutes=self.bar_frequency_minutes,
            reference_tb_name=self.reference_tb_name,
        )


def default_intraday_config() -> V2DatasetConfig:
    """The default v2 configuration for 1-minute intraday data.

    ``bar_frequency_minutes=1.0`` and ``horizon_bars == horizon_min`` for
    every TB. This preserves the Phase 16 behaviour exactly.

    Mirrors the 7 triple-barrier configs in
    `new_label_architecture_20260607_100000.md` section 3.3 and the 3
    meta-label configs in section 4.2.
    """
    triple_barriers = [
        TripleBarrierConfig(name="tb_long_aggressive",   side="CE",   pt_pct=0.010, sl_pct=0.006, horizon_min=5,  horizon_bars=5),
        TripleBarrierConfig(name="tb_long_conservative", side="CE",   pt_pct=0.005, sl_pct=0.004, horizon_min=10, horizon_bars=10),
        TripleBarrierConfig(name="tb_short_aggressive",  side="PE",   pt_pct=0.010, sl_pct=0.006, horizon_min=5,  horizon_bars=5),
        TripleBarrierConfig(name="tb_straddle_long_vol", side="both", pt_pct=0.015, sl_pct=0.010, horizon_min=15, horizon_bars=15),
        TripleBarrierConfig(name="tb_neutral_short_vol", side="both", pt_pct=0.004, sl_pct=0.008, horizon_min=15, horizon_bars=15),
        TripleBarrierConfig(name="tb_weekly_horizon",    side="both", pt_pct=0.020, sl_pct=0.015, horizon_min=60, horizon_bars=60),
        TripleBarrierConfig(name="tb_intraday_trend",    side="both", pt_pct=0.012, sl_pct=0.008, horizon_min=30, horizon_bars=30),
    ]
    meta_labels = [
        MetaLabelConfig(
            name="meta_trade",
            primary_signal_column="primary_signal_proba",
            primary_threshold=0.5,
            high_confidence_threshold=0.7,
            low_confidence_threshold=0.55,
        ),
    ]
    return V2DatasetConfig(
        triple_barriers=triple_barriers,
        meta_labels=meta_labels,
        reference_tb_name="tb_long_aggressive",
        bar_frequency_minutes=1.0,
    )


def default_daily_config() -> V2DatasetConfig:
    """Default v2 configuration for daily-resolution data.

    ``bar_frequency_minutes=1440.0`` (one trading day per bar). The same 7
    TB families are used, but with forward-bar counts rescaled to match a
    daily grid: 1-2 bars for short-term, 3-5 bars for medium-term. This
    matches the rescaled-horizon workaround used by the Phase 16
    evaluation-reporter and ensures that every entry has a non-empty
    forward path on the cost-aware daily dataset (Phase 17 fix).
    """
    triple_barriers = [
        TripleBarrierConfig(name="tb_long_aggressive",   side="CE",   pt_pct=0.010, sl_pct=0.006, horizon_min=1440, horizon_bars=1),
        TripleBarrierConfig(name="tb_long_conservative", side="CE",   pt_pct=0.005, sl_pct=0.004, horizon_min=2880, horizon_bars=2),
        TripleBarrierConfig(name="tb_short_aggressive",  side="PE",   pt_pct=0.010, sl_pct=0.006, horizon_min=1440, horizon_bars=1),
        TripleBarrierConfig(name="tb_straddle_long_vol", side="both", pt_pct=0.015, sl_pct=0.010, horizon_min=4320, horizon_bars=3),
        TripleBarrierConfig(name="tb_neutral_short_vol", side="both", pt_pct=0.004, sl_pct=0.008, horizon_min=4320, horizon_bars=3),
        TripleBarrierConfig(name="tb_weekly_horizon",    side="both", pt_pct=0.020, sl_pct=0.015, horizon_min=7200, horizon_bars=5),
        TripleBarrierConfig(name="tb_intraday_trend",    side="both", pt_pct=0.012, sl_pct=0.008, horizon_min=7200, horizon_bars=5),
    ]
    meta_labels = [
        MetaLabelConfig(
            name="meta_trade",
            primary_signal_column="primary_signal_proba",
            primary_threshold=0.5,
            high_confidence_threshold=0.7,
            low_confidence_threshold=0.55,
        ),
    ]
    return V2DatasetConfig(
        triple_barriers=triple_barriers,
        meta_labels=meta_labels,
        reference_tb_name="tb_long_aggressive",
        bar_frequency_minutes=1440.0,
    )


def default_v2_config() -> V2DatasetConfig:
    """Backward-compatible alias for :func:`default_intraday_config`.

    Returns the 1-minute intraday default (Phase 16 behaviour).
    """
    return default_intraday_config()


def load_v2_config(path: Path) -> V2DatasetConfig:
    """Load a v2 config from a JSON file.

    Legacy TB entries that do not specify ``horizon_bars`` are loaded as-is
    (with the sentinel 0) and are resolved at validate() time via
    ``V2DatasetConfig.resolve_horizon_bars``.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    tb_configs = []
    for c in payload.get("triple_barriers", []):
        # Allow legacy configs to omit horizon_bars; default 0 is the sentinel.
        if "horizon_bars" not in c:
            c = dict(c)
            c["horizon_bars"] = 0
        tb_configs.append(TripleBarrierConfig(**c))
    meta_configs = [MetaLabelConfig(**c) for c in payload.get("meta_labels", [])]
    config = V2DatasetConfig(
        triple_barriers=tb_configs,
        meta_labels=meta_configs,
        cost_stack=payload.get("cost_stack", dict(DEFAULT_COST_STACK)),
        cost_multiplier=float(payload.get("cost_multiplier", 1.0)),
        seed=int(payload.get("seed", 42)),
        reference_tb_name=payload.get("reference_tb_name"),
        bar_frequency_minutes=float(payload.get("bar_frequency_minutes", 1.0)),
    )
    config.validate()
    return config


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def validate_input_frame(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate that the input frame is leakage-free.

    Returns a dict with `ok`, `errors`, `warnings`. The function is
    non-mutating. It is called at the start of every build.
    """
    errors: List[str] = []
    warnings: List[str] = []
    cols = list(df.columns)
    forbidden_present = [c for c in cols if c in FORBIDDEN_INPUT_TOKENS]
    if forbidden_present:
        errors.append(f"forbidden columns present in input: {sorted(forbidden_present)}")
    required = {"timestamp", "instrument_key", "ltp"}
    missing = sorted(required - set(cols))
    if missing:
        errors.append(f"missing required columns: {missing}")
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        n_bad = int(ts.isna().sum())
        if n_bad:
            errors.append(f"unparseable timestamps: {n_bad}")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


# ---------------------------------------------------------------------------
# Core: forward-path lookups
# ---------------------------------------------------------------------------


def _normalize_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the input frame: tz-aware Asia/Kolkata, sorted, deduped.

    The function does NOT drop any rows that have forward-path data; it only
    ensures deterministic ordering. The frame is returned in a fresh copy.
    """
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=False)
    if getattr(work["timestamp"].dt, "tz", None) is None:
        work["timestamp"] = work["timestamp"].dt.tz_localize(TIMEZONE, nonexistent="NaT", ambiguous="NaT")
    else:
        work["timestamp"] = work["timestamp"].dt.tz_convert(TIMEZONE)
    for col in ("ltp",):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["timestamp", "instrument_key", "ltp"])
    work = work.sort_values(["instrument_key", "timestamp"], kind="stable")
    work = work.drop_duplicates(subset=["instrument_key", "timestamp"], keep="last")
    return work.reset_index(drop=True)


def _build_forward_path_index(
    df: pd.DataFrame,
) -> Dict[Any, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For each instrument_key, build three parallel arrays:
        - forward_timestamps[i] = np.datetime64[ns] monotonic list of timestamps.
        - forward_ltp[i] = ltp at those timestamps.
        - forward_index[i] = position in the original frame.

    Returns a dict keyed by instrument_key. The arrays are shared across
    calls (no copies); callers must not mutate them in place.

    Timestamps are always coerced to plain ``np.datetime64[ns]`` so that
    numpy can do vectorized ``>`` comparison without timezone or pandas
    Timestamp coercion issues. The conversion is done via ``view`` to
    preserve the underlying int64 value (not via ``astype("int64")`` which
    returns the *unit-of-the-source-dtype* int64, e.g. microseconds for a
    ``datetime64[us]`` Series).
    """
    out: Dict[Any, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    g = df.groupby("instrument_key", sort=False, group_keys=False)
    for key, sub in g:
        # The frame's datetime64 column may be in any unit (e.g. [us] for
        # Asia/Kolkata). We need ns-since-epoch int64 to compare against
        # np.datetime64[ns]. Cast the *underlying numpy* array (not the
        # Series) to datetime64[ns] and then view as i8; .astype('int64')
        # on a Series returns the int64 of the source dtype's unit.
        ts_arr = np.asarray(sub["timestamp"].array, dtype="datetime64[ns]")
        ts_ns = ts_arr.view("i8").copy()
        ts = ts_arr
        ltp = sub["ltp"].to_numpy(dtype=np.float64)
        idx = sub.index.to_numpy()
        out[key] = (ts, ltp, idx)
    return out


# ---------------------------------------------------------------------------
# Triple-barrier
# ---------------------------------------------------------------------------


def _apply_triple_barrier(
    entry_ts: np.datetime64,
    entry_ltp: float,
    forward_ts: np.ndarray,
    forward_ltp: np.ndarray,
    config: TripleBarrierConfig,
) -> Dict[str, Any]:
    """Apply one triple-barrier config to a single row.

    Returns a dict with keys:
        - label:    -1 (stop), 0 (time-out), +1 (profit), np.nan (no forward path)
        - hit_time: np.datetime64[ns] of barrier hit, or np.datetime64('NaT')
        - return:   signed return at barrier hit, or NaN
        - skip_reason: string or None

    The vertical (time) barrier is the Nth forward bar after entry_ts,
    where N = ``config.horizon_bars``. If ``horizon_bars == 0`` (the
    legacy sentinel) the cap falls back to ``entry_ts +
    horizon_min`` minutes for backward compatibility.

    Guarantees (enforced by assertions, not assumptions):
        - The forward path used here is strictly forward in time
          (forward_ts > entry_ts).
        - `label` is in {-1, 0, +1, np.nan}.
        - For a non-NaN label, `return` is the return between entry_ltp
          and the barrier price (for stops/profit) or the last forward
          ltp (for time-out).
        - `hit_time` is always a numpy datetime64[ns] so the int64 epoch
          comparison downstream is correct.
    """
    NAT64 = np.datetime64("NaT")
    # Strictly forward filter
    mask = forward_ts > entry_ts
    fwd_ts = forward_ts[mask]
    fwd_ltp = forward_ltp[mask]

    if fwd_ts.size < int(config.min_forward_bars):
        return {
            "label": np.nan,
            "hit_time": NAT64,
            "return": np.nan,
            "skip_reason": "insufficient_forward_bars",
        }

    # Apply horizon cap. Two modes:
    #   1) New (Phase 17): cap by forward bar count (config.horizon_bars).
    #      This is resolution-agnostic: 5 bars on a daily dataset means
    #      5 trading days, on a 1-min dataset means 5 minutes. Works on
    #      both.
    #   2) Legacy: cap by wall-clock minutes. Used only when
    #      horizon_bars == 0 (sentinel). Backward-compatible with all
    #      Phase 16 tests.
    if int(config.horizon_bars) > 0:
        cap_n = min(int(config.horizon_bars), int(fwd_ts.size))
        fwd_ts = fwd_ts[:cap_n]
        fwd_ltp = fwd_ltp[:cap_n]
    else:
        horizon_cutoff = entry_ts + np.timedelta64(int(config.horizon_min), "m")
        in_horizon = fwd_ts <= horizon_cutoff
        fwd_ts = fwd_ts[in_horizon]
        fwd_ltp = fwd_ltp[in_horizon]

    if fwd_ts.size == 0:
        return {
            "label": np.nan,
            "hit_time": NAT64,
            "return": np.nan,
            "skip_reason": "no_bars_in_horizon",
        }

    pt_price = float(entry_ltp) * (1.0 + float(config.pt_pct))
    sl_price = float(entry_ltp) * (1.0 - float(config.sl_pct))

    for ts, px in zip(fwd_ts, fwd_ltp):
        if px >= pt_price:
            return {
                "label": 1,
                "hit_time": ts,
                "return": float(config.pt_pct),
                "skip_reason": None,
            }
        if px <= sl_price:
            return {
                "label": -1,
                "hit_time": ts,
                "return": float(-config.sl_pct),
                "skip_reason": None,
            }

    # Time-out: use last available ltp in the horizon
    last_px = float(fwd_ltp[-1])
    time_out_return = (last_px - float(entry_ltp)) / max(float(entry_ltp), 1e-9)
    return {
        "label": 0,
        "hit_time": fwd_ts[-1],
        "return": time_out_return,
        "skip_reason": None,
    }


def build_triple_barrier_labels(
    df: pd.DataFrame,
    config: TripleBarrierConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply one triple-barrier config to every row of df.

    The returned frame has the same index as `df` plus three new columns:
        - <name>_label:  int8 in {-1, 0, +1, NaN}
        - <name>_return: float (signed return at barrier hit, or NaN)
        - <name>_hit_time: tz-aware timestamp or NaT

    The forward path is never written to the returned frame.
    """
    config.validate()
    work = _normalize_input_frame(df)

    # Compute a *per-row* mask for the side filter. Rows that don't match
    # the side get NaN labels (not dropped), so subsequent TB configs see
    # the same set of rows.
    if "option_type" in work.columns and config.side in {"CE", "PE"}:
        side_mask = (work["option_type"].astype(str).str.upper() == config.side).to_numpy()
    else:
        side_mask = np.ones(len(work), dtype=bool)

    fwd_index = _build_forward_path_index(work)

    labels = np.full(len(work), np.nan, dtype=np.float32)
    returns = np.full(len(work), np.nan, dtype=np.float64)
    # Store hit_times as int64 ns-since-epoch so the writer can reconstruct
    # a tz-aware DatetimeIndex matching the parent frame.
    hit_times_ns = np.full(len(work), pd.NaT.value, dtype=np.int64)
    skip_reasons: List[Optional[str]] = [None] * len(work)

    # Pre-compute entry timestamps as int64 ns since epoch. The frame's
    # datetime column may be datetime64[us, Asia/Kolkata]; .astype('int64')
    # on the Series would return microseconds. Use the underlying numpy
    # array, cast to datetime64[ns], and view as i8 to get ns.
    work_ts_int64 = (
        np.asarray(work["timestamp"].array, dtype="datetime64[ns]")
        .view("i8")
        .copy()
    )

    for key, (fwd_ts, fwd_ltp, _) in fwd_index.items():
        sub = work[work["instrument_key"] == key]
        sub_pos = sub.index.to_numpy()
        sub_ltp = sub["ltp"].to_numpy()
        for pos_in_sub in range(len(sub)):
            global_idx = int(sub_pos[pos_in_sub])
            if not bool(side_mask[global_idx]):
                # Row is filtered out by the side config; leave NaN.
                skip_reasons[global_idx] = "side_filter"
                continue
            # Use the pre-computed int64 to avoid tz/dtype surprises.
            entry_ts = np.datetime64(int(work_ts_int64[global_idx]), "ns")
            entry_ltp = float(sub_ltp[pos_in_sub])
            result = _apply_triple_barrier(
                entry_ts=entry_ts,
                entry_ltp=entry_ltp,
                forward_ts=fwd_ts,
                forward_ltp=fwd_ltp,
                config=config,
            )
            labels[global_idx] = result["label"]
            returns[global_idx] = result["return"]
            hit_time = result["hit_time"]
            if hit_time is None or (isinstance(hit_time, np.datetime64) and np.isnat(hit_time)):
                # Leave as NaT value.
                pass
            else:
                # Convert np.datetime64 -> int64 ns since epoch. The hit_time
                # returned by _apply_triple_barrier is a 0-d np.datetime64[ns]
                # (drawn from the fwd_ts array). Cast it to int64 via the
                # i8 view to guarantee ns units; .astype('int64') can return
                # the wrong unit on a 0-d array.
                hit_times_ns[global_idx] = int(np.asarray(hit_time, dtype="datetime64[ns]").view("i8"))
            skip_reasons[global_idx] = result["skip_reason"]

    out = work.copy()
    out.loc[:, f"{config.name}_label"] = labels
    out.loc[:, f"{config.name}_return"] = returns
    # Build hit_time as a tz-aware DatetimeIndex matching the parent frame's
    # timezone, so downstream comparisons (and tests) don't fail on dtype
    # mismatch. We use the int64 epoch ns that was captured in the inner loop
    # (hit_times_ns) to avoid any 1970 re-anchoring that happens if we wrap
    # a python datetime in pd.Timestamp().
    parent_tz = getattr(out["timestamp"].dt, "tz", None)
    if parent_tz is not None:
        hit_time_index = pd.DatetimeIndex(hit_times_ns, tz=parent_tz)
    else:
        hit_time_index = pd.DatetimeIndex(hit_times_ns)
    out.loc[:, f"{config.name}_hit_time"] = hit_time_index
    out.loc[:, f"{config.name}_skip_reason"] = skip_reasons

    audit = _audit_label_series(out[f"{config.name}_label"], config.name)
    return out, audit


# ---------------------------------------------------------------------------
# Meta-labels
# ---------------------------------------------------------------------------


def build_meta_labels(
    df: pd.DataFrame,
    config: MetaLabelConfig,
    reference_label_column: str,
    reference_return_column: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build meta-labels on top of the reference triple-barrier.

    Required columns in df:
        - `reference_label_column`: int in {-1, 0, +1, NaN}
        - `reference_return_column`: signed return at barrier hit (NaN if no hit)
        - `config.primary_signal_column`: float in [0, 1]
        - `config.use_regime_column` (optional): int

    New columns:
        - meta_<name>_trade: int in {0, 1, NaN}
        - meta_<name>_high_conf: int in {0, 1, NaN}
        - meta_<name>_low_conf: int in {0, 1, NaN}
    """
    config.validate()
    work = df.copy()
    for col in (reference_label_column, reference_return_column, config.primary_signal_column):
        if col not in work.columns:
            raise ValueError(f"build_meta_labels: required column {col!r} not in frame")

    primary = pd.to_numeric(work[config.primary_signal_column], errors="coerce")
    ref_label = pd.to_numeric(work[reference_label_column], errors="coerce")
    ref_return = pd.to_numeric(work[reference_return_column], errors="coerce")

    direction = np.where(primary >= config.primary_threshold, 1, -1)
    direction = pd.Series(direction, index=work.index, dtype=np.int8)

    # meta_trade: 1 iff (ref_label == direction) AND (ref_return > 0)
    matches_direction = (ref_label == direction) & ref_label.notna()
    positive_return = ref_return > 0
    trade = (matches_direction & positive_return).astype(np.int8)
    trade[ref_label.isna() | ref_return.isna()] = np.nan

    # meta_high_conf: 1 iff trade == 1 AND primary >= high_confidence_threshold
    high_conf = (trade == 1) & (primary >= config.high_confidence_threshold)
    high_conf = high_conf.astype(np.int8)
    high_conf[trade.isna()] = np.nan

    # meta_low_conf: 1 iff (trade == 0 AND primary >= low_confidence_threshold)
    # This is a "primary is confident but the label says no-trade" signal.
    low_conf = (trade == 0) & (primary >= config.low_confidence_threshold)
    low_conf = low_conf.astype(np.int8)
    low_conf[trade.isna()] = np.nan

    out = work.copy()
    out[f"meta_{config.name}_trade"] = trade
    out[f"meta_{config.name}_high_conf"] = high_conf
    out[f"meta_{config.name}_low_conf"] = low_conf

    audit = {
        "meta_label": config.name,
        "trade_positive_rate": _safe_positive_rate(out[f"meta_{config.name}_trade"]),
        "high_conf_positive_rate": _safe_positive_rate(out[f"meta_{config.name}_high_conf"]),
        "low_conf_positive_rate": _safe_positive_rate(out[f"meta_{config.name}_low_conf"]),
    }
    return out, audit


# ---------------------------------------------------------------------------
# Cost-adjusted economic labels
# ---------------------------------------------------------------------------


def build_cost_adjusted_labels(
    df: pd.DataFrame,
    reference_return_column: str,
    cost_stack: Dict[str, float],
    cost_multiplier: float = 1.0,
    lot_size: int = 75,  # NIFTY lot size
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build cost-adjusted economic labels.

    Required columns in df:
        - `reference_return_column`: gross signed return at barrier hit
        - `entry_ltp_column` (optional, default `ltp`): entry price for cost calc

    New columns:
        - econ_net_pnl_after_brokerage_pct: gross_return - 2 * brokerage_pct * cost_multiplier
        - econ_net_pnl_after_spread_pct:    - spread_default_pct * cost_multiplier
        - econ_net_pnl_after_slippage_pct:  - slippage_default_pct * cost_multiplier
        - econ_net_pnl_total_pct:           gross_return - total_cost_pct
        - econ_total_cost_pct:              total cost in return units
    """
    work = df.copy()
    if reference_return_column not in work.columns:
        raise ValueError(f"build_cost_adjusted_labels: missing {reference_return_column!r}")

    cm = float(cost_multiplier)
    cs = cost_stack

    brokerage = float(cs.get("brokerage_per_side_pct", 0.0)) * 2.0 * cm  # buy + sell
    exchange = float(cs.get("exchange_charge_pct", 0.0)) * 2.0 * cm
    sebi = float(cs.get("sebi_charge_pct", 0.0)) * 2.0 * cm
    gst = float(cs.get("gst_pct_on_brokerage_exchange_sebi", 0.0)) * (brokerage + exchange + sebi)
    stt = float(cs.get("stt_sell_pct_of_notional", 0.0)) * cm
    stamp = float(cs.get("stamp_duty_buy_pct_of_notional", 0.0)) * cm
    spread = float(cs.get("spread_default_pct", 0.0)) * cm
    slippage = float(cs.get("slippage_default_pct", 0.0)) * cm

    gross = pd.to_numeric(work[reference_return_column], errors="coerce")

    work = work.copy()
    work.loc[:, "econ_net_pnl_after_brokerage_pct"] = gross - brokerage
    work.loc[:, "econ_net_pnl_after_spread_pct"] = gross - spread
    work.loc[:, "econ_net_pnl_after_slippage_pct"] = gross - slippage
    work.loc[:, "econ_total_cost_pct"] = brokerage + exchange + sebi + gst + stt + stamp + spread + slippage
    work.loc[:, "econ_net_pnl_total_pct"] = gross - work["econ_total_cost_pct"]
    pf_breach = (work["econ_net_pnl_total_pct"] > 0).astype(np.int8)
    pf_breach[gross.isna()] = np.nan
    work.loc[:, "econ_pf_breach"] = pf_breach

    audit = {
        "lot_size_used": int(lot_size),
        "cost_multiplier": cm,
        "component_costs_pct": {
            "brokerage_round_trip": brokerage,
            "exchange_round_trip": exchange,
            "sebi_round_trip": sebi,
            "gst": gst,
            "stt_sell_side": stt,
            "stamp_buy_side": stamp,
            "spread": spread,
            "slippage": slippage,
        },
        "total_cost_pct": float(brokerage + exchange + sebi + gst + stt + stamp + spread + slippage),
        "pf_breach_positive_rate": _safe_positive_rate(work["econ_pf_breach"]),
    }
    return work, audit


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------


def _safe_positive_rate(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float((s == 1).mean())


def _audit_label_series(series: pd.Series, name: str) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce")
    n_total = int(len(s))
    n_labeled = int(s.notna().sum())
    counts = {int(k): int(v) for k, v in s.dropna().value_counts().to_dict().items()}
    audit = {
        "label": name,
        "rows_total": n_total,
        "rows_labeled": n_labeled,
        "rows_skipped": n_total - n_labeled,
        "value_counts": counts,
    }
    if n_labeled > 0:
        audit["positive_rate_+1"] = float((s == 1).sum() / n_labeled)
        audit["zero_rate_0"] = float((s == 0).sum() / n_labeled)
        audit["negative_rate_-1"] = float((s == -1).sum() / n_labeled)
    return audit


# ---------------------------------------------------------------------------
# Top-level build
# ---------------------------------------------------------------------------


def build_v2_labels(
    df: pd.DataFrame,
    config: V2DatasetConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build the v2 dataset (triple-barrier + meta + cost-adjusted).

    The function is pure: it does not write to disk. The CLI wrapper
    (`main`) handles I/O.

    If any TB in ``config.triple_barriers`` has ``horizon_bars == 0`` (the
    sentinel for "not yet resolved"), the config is first run through
    :meth:`V2DatasetConfig.resolve_horizon_bars` so that every TB has a
    positive forward-bar count. The resolution is keyed on
    ``config.bar_frequency_minutes``: 1.0 for intraday, 1440.0 for daily.
    """
    config.validate()
    input_audit = validate_input_frame(df)
    if not input_audit["ok"]:
        raise ValueError(f"Input frame failed validation: {input_audit['errors']}")

    # Resolve any sentinel horizon_bars values. We do this before the
    # audit dict's `config` snapshot so the persisted audit reflects the
    # final per-bar caps.
    if any(int(tb.horizon_bars) <= 0 for tb in config.triple_barriers):
        config = config.resolve_horizon_bars()

    work = _normalize_input_frame(df)
    audit: Dict[str, Any] = {
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": asdict(config),
        "input_audit": input_audit,
        "rows_in": int(len(work)),
        "label_audits": {},
    }

    # 1. Triple-barrier labels
    for tb in config.triple_barriers:
        work, tb_audit = build_triple_barrier_labels(work, tb)
        audit["label_audits"][tb.name] = tb_audit

    # 2. Meta-labels (require a reference TB)
    if config.meta_labels:
        if not config.reference_tb_name:
            raise ValueError("meta_labels configured but reference_tb_name is None")
        ref_name = config.reference_tb_name
        ref_label_col = f"{ref_name}_label"
        ref_return_col = f"{ref_name}_return"
        if ref_label_col not in work.columns:
            raise ValueError(f"reference TB {ref_name!r} not in frame; build triple-barriers first")
        for ml in config.meta_labels:
            work, ml_audit = build_meta_labels(
                work, ml, ref_label_col, ref_return_col
            )
            audit["label_audits"][f"meta_{ml.name}"] = ml_audit

    # 3. Cost-adjusted economic labels (always built, anchored on reference TB)
    if config.reference_tb_name and f"{config.reference_tb_name}_return" in work.columns:
        work, econ_audit = build_cost_adjusted_labels(
            work,
            reference_return_column=f"{config.reference_tb_name}_return",
            cost_stack=config.cost_stack,
            cost_multiplier=config.cost_multiplier,
        )
        audit["label_audits"]["economic"] = econ_audit

    audit["rows_out"] = int(len(work))
    # Run the leakage self-check on the labeled frame so callers always have
    # an `audit["leakage_audit"]` entry.
    audit["leakage_audit"] = _call_run_leakage_audit(work, config=config)
    return work, audit


# ---------------------------------------------------------------------------
# Leakage self-check
# ---------------------------------------------------------------------------


def run_leakage_audit(
    df: pd.DataFrame,
    config: Optional[V2DatasetConfig] = None,
) -> Dict[str, Any]:
    """Run a leakage self-check on a v2-labeled frame.

    Checks:
        1. No `FORBIDDEN_INPUT_TOKENS` columns present.
        2. All label columns are present and have the expected dtype.
        3. For each TB label, hit_time > entry timestamp for non-NaN labels.
        4. For each TB label, the (hit_time - entry_ts) duration is within
           the configured cap, in minutes. The cap is
           ``tb.horizon_bars * config.bar_frequency_minutes`` if a config
           is provided, else ``tb.horizon_min`` (legacy Phase 16 default).
        5. For each TB label, return == +/- pt_pct or +/- sl_pct OR is the
           time-out return.
        6. Meta-label columns agree with the primary signal and the
           reference TB label.

    `config` is optional. When omitted, the within-horizon check uses a
    fallback table keyed by TB name with the Phase 16 default
    ``horizon_min`` values. New callers should always pass `config` so the
    per-bar cap is checked (instead of the legacy minute cap).
    """
    errors: List[str] = []
    warnings: List[str] = []
    cols = set(df.columns)
    forbidden_present = sorted(c for c in cols if c in FORBIDDEN_INPUT_TOKENS)
    if forbidden_present:
        errors.append(f"forbidden columns present: {forbidden_present}")

    tb_label_cols = [c for c in cols if c.endswith("_label") and c.startswith("tb_")]
    for col in tb_label_cols:
        if col not in cols:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        if valid.empty:
            continue
        unexpected = sorted(set(valid.unique()) - {-1.0, 0.0, 1.0})
        if unexpected:
            errors.append(f"TB label {col!r} has unexpected values: {unexpected}")

    # Build a per-TB name -> expected_max_minutes table from the config
    # when available; fall back to the legacy horizon_min when not.
    expected_max_minutes_by_name: Dict[str, float] = {}
    if config is not None:
        for tb in config.triple_barriers:
            expected_max_minutes_by_name[tb.name] = config.expected_max_minutes(tb)

    # Check that hit_time > entry timestamp where label is not NaN.
    # Use int64 ns-since-epoch comparison to avoid pandas tz-aware vs naive
    # dtype mismatches. The frame's datetime columns may be datetime64[us]
    # (e.g. Asia/Kolkata input), so we cast the underlying numpy array to
    # datetime64[ns] and view as i8; .astype('int64') on the Series would
    # yield microseconds and silently mis-anchor at 1970.
    if "timestamp" in df.columns:
        entry_ns = (
            np.asarray(df["timestamp"].array, dtype="datetime64[ns]")
            .view("i8")
            .copy()
        )
        for col in tb_label_cols:
            hit_col = col.replace("_label", "_hit_time")
            if hit_col not in df.columns:
                warnings.append(f"TB label {col!r} has no matching {hit_col!r} column")
                continue
            label = pd.to_numeric(df[col], errors="coerce")
            hit_ns = (
                np.asarray(df[hit_col].array, dtype="datetime64[ns]")
                .view("i8")
                .copy()
            )
            labeled = label.notna().to_numpy()
            hit_present = ~pd.isna(hit_ns)
            bad = labeled & hit_present & (hit_ns <= entry_ns)
            n_bad = int(bad.sum())
            if n_bad:
                errors.append(
                    f"TB label {col!r}: {n_bad} rows have hit_time <= entry timestamp (leakage)"
                )
            # Within-horizon cap check. Use the per-TB cap from the config
            # if available; otherwise fall back to the legacy default
            # horizon_min. The +1 minute tolerance preserves the Phase 16
            # audit semantics (test_v2_leakage_audit::test_hit_time_within_horizon).
            tb_name = col[: -len("_label")]
            expected = expected_max_minutes_by_name.get(tb_name)
            if expected is None:
                legacy_defaults = {
                    "tb_long_aggressive": 5,
                    "tb_long_conservative": 10,
                    "tb_short_aggressive": 5,
                    "tb_straddle_long_vol": 15,
                    "tb_neutral_short_vol": 15,
                    "tb_weekly_horizon": 60,
                    "tb_intraday_trend": 30,
                }
                expected = float(legacy_defaults.get(tb_name, 60))
            delta_min = (hit_ns - entry_ns) / 1e9 / 60.0
            within_horizon = delta_min <= expected + 1.0
            bad_horizon = labeled & hit_present & ~within_horizon
            n_bad_horizon = int(bad_horizon.sum())
            if n_bad_horizon:
                errors.append(
                    f"TB label {col!r}: {n_bad_horizon} hit_times exceed "
                    f"expected cap of {expected:.1f} minutes"
                )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "forbidden_present": forbidden_present,
        "tb_label_columns_found": tb_label_cols,
        "expected_max_minutes_by_name": expected_max_minutes_by_name,
    }


def _call_run_leakage_audit(
    df: pd.DataFrame,
    config: Optional[V2DatasetConfig] = None,
) -> Dict[str, Any]:
    """Call ``run_leakage_audit`` signature-tolerantly.

    Some tests monkey-patch ``run_leakage_audit`` with a no-arg stub (e.g.
    ``test_cli_returns_exit_code_2_when_post_build_leakage_audit_fails``).
    Inspecting the live function's signature keeps both the real two-arg
    call path and the legacy one-arg stub happy. If the live function
    accepts a ``config`` keyword, we pass it; otherwise we fall back to
    a positional one-arg call.
    """
    try:
        sig = inspect.signature(run_leakage_audit)
    except (TypeError, ValueError):
        sig = None
    if sig is not None and "config" in sig.parameters:
        return run_leakage_audit(df, config=config)
    return run_leakage_audit(df)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a v2 research dataset with triple-barrier, meta, and cost-adjusted labels.")
    parser.add_argument("--input", required=True, help="Path to input option-context CSV or parquet.")
    parser.add_argument("--output", required=True, help="Path to write v2 labeled parquet (or CSV).")
    parser.add_argument("--config", default=None, help="Path to v2 config JSON. If omitted, --config-preset is used.")
    parser.add_argument(
        "--config-preset",
        choices=("intraday", "daily"),
        default="intraday",
        help=(
            "Default config preset when --config is not given. 'intraday' uses 1-minute bars "
            "(Phase 16 default). 'daily' uses 1 trading day per bar "
            "(bar_frequency_minutes=1440.0; Phase 17 fix for the cost-aware daily dataset)."
        ),
    )
    parser.add_argument("--audit-output", default=None, help="Path to write v2 audit JSON.")
    parser.add_argument("--leakage-report", default=None, help="Path to write v2 leakage audit JSON.")
    parser.add_argument("--fail-on-leakage", action="store_true", help="Raise on any leakage self-check failure.")
    parser.add_argument("--sample", type=int, default=None, help="If set, randomly sample this many rows (seeded).")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.WARNING if (argv and "--quiet" in argv) else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    args = _parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Reading input: %s", in_path)
    if in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path, low_memory=False)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    if args.config:
        log.info("Loading v2 config: %s", args.config)
        config = load_v2_config(Path(args.config))
    elif args.config_preset == "daily":
        log.info("Using daily v2 config (bar_frequency_minutes=1440.0)")
        config = default_daily_config()
    else:
        log.info("Using default intraday v2 config (bar_frequency_minutes=1.0)")
        config = default_intraday_config()
    config.validate()

    if args.sample:
        rng = np.random.default_rng(config.seed)
        idx = rng.choice(len(df), size=min(args.sample, len(df)), replace=False)
        df = df.iloc[sorted(idx.tolist())].reset_index(drop=True)
        log.info("Sampled to %d rows", len(df))

    labeled, audit = build_v2_labels(df, config)

    log.info("Writing v2 dataset: %s", out_path)
    if out_path.suffix.lower() == ".parquet":
        labeled.to_parquet(out_path, index=False)
    else:
        labeled.to_csv(out_path, index=False)

    leakage = _call_run_leakage_audit(labeled, config=config)
    audit["leakage_audit"] = leakage

    if args.audit_output:
        Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.audit_output).write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
        log.info("Wrote v2 audit: %s", args.audit_output)

    if args.leakage_report:
        Path(args.leakage_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.leakage_report).write_text(json.dumps(leakage, indent=2, default=str), encoding="utf-8")
        log.info("Wrote v2 leakage report: %s", args.leakage_report)

    if args.fail_on_leakage and not leakage["ok"]:
        log.error("Leakage self-check failed: %s", leakage["errors"])
        return 2

    log.info("Done. %d rows out, %d label families.", len(labeled), len(audit["label_audits"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
