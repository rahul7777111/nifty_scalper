#!/usr/bin/env python3
"""
candidate_profile.py
====================
Unified schema for ML candidate + dynamic preset trading profiles.

Each deployable candidate is:
  candidate = model + feature_set + side_policy + dynamic_presets + gates + artifacts
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Side policies
# ---------------------------------------------------------------------------

SIDE_CE_ONLY = "CE_ONLY"
SIDE_PE_ONLY = "PE_ONLY"
SIDE_BOTH = "BOTH"
SIDE_AUTO_DIRECTIONAL = "AUTO_DIRECTIONAL"
SIDE_BOTH_SYMMETRIC = "BOTH_SYMMETRIC"

SIDE_FILTER_MAP = {
    SIDE_CE_ONLY: "CE_only",
    SIDE_PE_ONLY: "PE_only",
    SIDE_BOTH: "BOTH",
    SIDE_AUTO_DIRECTIONAL: "AUTO_DIRECTIONAL",
    SIDE_BOTH_SYMMETRIC: "BOTH_SYMMETRIC",
}


# ---------------------------------------------------------------------------
# Dynamic preset
# ---------------------------------------------------------------------------

@dataclass
class DynamicPreset:
    name: str
    option_side_policy: str = SIDE_PE_ONLY
    min_confidence: float = 0.35
    entry_threshold: float = 0.35
    exit_threshold: Optional[float] = None
    max_trades_per_day: int = 3
    cooldown_minutes: int = 0
    entry_cutoff: str = "15:25"
    stop_loss_pct: float = 0.0
    target_pct: float = 0.0
    trailing_sl_pct: Optional[float] = None
    min_premium: float = 0.0
    max_premium: float = 99999.0
    max_spread_pct: float = 0.15
    min_volume: Optional[float] = None
    min_oi: Optional[float] = None
    allowed_dte_min: Optional[int] = None
    allowed_dte_max: Optional[int] = None
    regime_filter: Dict[str, Any] = field(default_factory=dict)
    volatility_filter: Dict[str, Any] = field(default_factory=dict)
    trend_filter: Dict[str, Any] = field(default_factory=dict)
    cost_buffer_bps: float = 25.0
    notes: str = ""
    # Retrainer-compatible fields
    preset_id: str = ""
    preset_family: str = ""
    top_n_confidence_per_day: int = 3
    spread_limit_pct: float = 0.15
    liquidity_min: float = 0.0
    premium_band: str = "all"
    dte_range: str = "all"
    regime_filter_str: str = "all"
    expiry_filter: str = "all"
    avoid_first_n_minutes: int = 0
    avoid_last_n_minutes: int = 0
    entry_time_start: str = "09:30"
    entry_time_end: str = "15:00"
    expected_move_to_cost_ratio_min: float = 0.0
    cooldown_after_loss_minutes: int = 0
    daily_stop_loss_pct: float = 0.0
    daily_profit_lock_pct: float = 0.0

    def __post_init__(self) -> None:
        if not self.preset_id:
            self.preset_id = self.name
        if not self.preset_family:
            self.preset_family = self.name
        if self.spread_limit_pct == 0.15 and self.max_spread_pct != 0.15:
            self.spread_limit_pct = self.max_spread_pct
        if self.entry_threshold and not self.min_confidence:
            self.min_confidence = self.entry_threshold

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DynamicPreset":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "name" not in known:
            known["name"] = str(data.get("preset_id") or data.get("preset_family") or "unnamed_preset")
        if "regime_filter_detail" in data and isinstance(data["regime_filter_detail"], dict):
            known["regime_filter"] = data["regime_filter_detail"]
        if "threshold" in data and "entry_threshold" not in known:
            known["entry_threshold"] = float(data["threshold"])
        return cls(**known)

    def to_retrainer_dict(self) -> Dict[str, Any]:
        """Format for candidate_filters.apply_dynamic_preset_filter."""
        return {
            "preset_id": self.preset_id,
            "preset_family": self.preset_family,
            "option_side_policy": self.option_side_policy,
            "threshold": self.entry_threshold,
            "min_confidence": self.min_confidence,
            "max_trades_per_day": self.max_trades_per_day,
            "top_n_confidence_per_day": self.top_n_confidence_per_day,
            "spread_limit_pct": self.spread_limit_pct,
            "liquidity_min": self.liquidity_min,
            "premium_band": self.premium_band,
            "dte_range": self.dte_range,
            "regime_filter": self.regime_filter_str,
            "expiry_filter": self.expiry_filter,
            "avoid_first_n_minutes": self.avoid_first_n_minutes,
            "avoid_last_n_minutes": self.avoid_last_n_minutes,
            "entry_time_start": self.entry_time_start,
            "entry_time_end": self.entry_time_end,
            "expected_move_to_cost_ratio_min": self.expected_move_to_cost_ratio_min,
            "cooldown_after_loss_minutes": self.cooldown_after_loss_minutes,
            "daily_stop_loss_pct": self.daily_stop_loss_pct,
            "daily_profit_lock_pct": self.daily_profit_lock_pct,
            "min_premium": self.min_premium,
            "max_premium": self.max_premium,
            "cost_buffer_bps": self.cost_buffer_bps,
            "regime_filter_detail": self.regime_filter,
            "volatility_filter": self.volatility_filter,
            "trend_filter": self.trend_filter,
        }


# ---------------------------------------------------------------------------
# Candidate profile
# ---------------------------------------------------------------------------

@dataclass
class CandidateProfile:
    candidate_id: str
    model_name: str
    feature_set_name: str
    target_name: str
    side_policy: str
    preset_family: str
    dynamic_presets: Dict[str, Dict[str, Any]]
    threshold_policy: Dict[str, Any]
    selection_policy: Dict[str, Any]
    risk_policy: Dict[str, Any]
    cost_policy: Dict[str, Any]
    artifact_paths: Dict[str, str]
    validation_metrics: Dict[str, Any]
    gate_results: Dict[str, Any]
    live_computable_features: List[str]
    created_at: str = ""
    shadow_mode_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateProfile":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, root: Path | str) -> Path:
        root = Path(root)
        cand_dir = root / self.candidate_id
        cand_dir.mkdir(parents=True, exist_ok=True)
        profile_path = cand_dir / "candidate_profile.json"
        profile_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        presets_path = cand_dir / "dynamic_presets.json"
        presets_path.write_text(json.dumps(self.dynamic_presets, indent=2), encoding="utf-8")
        if self.validation_metrics:
            (cand_dir / "metrics.json").write_text(
                json.dumps(self.validation_metrics, indent=2), encoding="utf-8"
            )
        if self.gate_results:
            (cand_dir / "gates.json").write_text(
                json.dumps(self.gate_results, indent=2), encoding="utf-8"
            )
        active = self._active_preset_dict()
        if active:
            (cand_dir / "dynamic_preset.json").write_text(
                json.dumps(active, indent=2), encoding="utf-8"
            )
        return cand_dir

    def _active_preset_dict(self) -> Dict[str, Any]:
        if not self.dynamic_presets:
            return {}
        first_key = next(iter(self.dynamic_presets))
        preset_data = self.dynamic_presets[first_key]
        if isinstance(preset_data, dict):
            return preset_data
        return {}


def load_candidate_profile(path: Path | str) -> CandidateProfile:
    path = Path(path)
    if path.is_dir():
        path = path / "candidate_profile.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return CandidateProfile.from_dict(data)


# ---------------------------------------------------------------------------
# Preset families (candidate-specific)
# ---------------------------------------------------------------------------

def _base(
    name: str,
    side: str,
    *,
    threshold: float = 0.35,
    top_n: int = 3,
    spread: float = 0.12,
    liquidity: float = 50000.0,
    regime: str = "all",
    avoid_first: int = 5,
    avoid_last: int = 5,
    entry_start: str = "09:30",
    entry_end: str = "15:00",
    dte: str = "all",
    premium_band: str = "all",
    notes: str = "",
    vol_filter: Optional[Dict[str, Any]] = None,
    trend_filter: Optional[Dict[str, Any]] = None,
    regime_detail: Optional[Dict[str, Any]] = None,
) -> DynamicPreset:
    return DynamicPreset(
        name=name,
        preset_id=name,
        preset_family=name,
        option_side_policy=side,
        min_confidence=threshold,
        entry_threshold=threshold,
        max_trades_per_day=top_n,
        top_n_confidence_per_day=top_n,
        spread_limit_pct=spread,
        max_spread_pct=spread,
        liquidity_min=liquidity,
        premium_band=premium_band,
        dte_range=dte,
        regime_filter_str=regime,
        avoid_first_n_minutes=avoid_first,
        avoid_last_n_minutes=avoid_last,
        entry_time_start=entry_start,
        entry_time_end=entry_end,
        regime_filter=regime_detail or {},
        volatility_filter=vol_filter or {},
        trend_filter=trend_filter or {},
        notes=notes,
    )


def build_preset_family(preset_family: str) -> List[DynamicPreset]:
    """Return preset variants for a named family."""
    builders = {
        "PE_only_conservative": lambda: [
            _base("PE_only_conservative_t30", SIDE_PE_ONLY, threshold=0.30, top_n=2, spread=0.10, notes="PE conservative"),
            _base("PE_only_conservative_t35", SIDE_PE_ONLY, threshold=0.35, top_n=2, spread=0.10),
            _base("PE_only_conservative_t40", SIDE_PE_ONLY, threshold=0.40, top_n=3, spread=0.08),
        ],
        "CE_only_conservative": lambda: [
            _base("CE_only_conservative_t30", SIDE_CE_ONLY, threshold=0.30, top_n=2, spread=0.10, notes="CE conservative"),
            _base("CE_only_conservative_t35", SIDE_CE_ONLY, threshold=0.35, top_n=2, spread=0.10),
            _base("CE_only_conservative_t40", SIDE_CE_ONLY, threshold=0.40, top_n=3, spread=0.08),
        ],
        "BOTH_directional_auto": lambda: [
            _base("BOTH_dir_auto_t30", SIDE_AUTO_DIRECTIONAL, threshold=0.30, top_n=3,
                  regime_detail={"bullish": "CE", "bearish": "PE", "mixed": "no_trade"}),
            _base("BOTH_dir_auto_t35", SIDE_AUTO_DIRECTIONAL, threshold=0.35, top_n=3,
                  regime_detail={"bullish": "CE", "bearish": "PE", "mixed": "no_trade"}),
        ],
        "BOTH_symmetric": lambda: [
            _base("BOTH_sym_t30", SIDE_BOTH_SYMMETRIC, threshold=0.30, top_n=3),
            _base("BOTH_sym_t35", SIDE_BOTH_SYMMETRIC, threshold=0.35, top_n=3),
        ],
        "high_confidence_low_frequency": lambda: [
            _base("hi_conf_low_freq_t45", SIDE_BOTH, threshold=0.45, top_n=1, spread=0.08, liquidity=100000.0),
            _base("hi_conf_low_freq_t50", SIDE_BOTH, threshold=0.50, top_n=1, spread=0.08, liquidity=100000.0),
        ],
        "mid_confidence_high_liquidity": lambda: [
            _base("mid_conf_hi_liq_t30", SIDE_BOTH, threshold=0.30, top_n=3, spread=0.08, liquidity=150000.0),
            _base("mid_conf_hi_liq_t35", SIDE_BOTH, threshold=0.35, top_n=3, spread=0.08, liquidity=150000.0),
        ],
        "volatility_breakout": lambda: [
            _base("vol_breakout_t30", SIDE_BOTH, threshold=0.30, top_n=3, regime="high_volatility",
                  vol_filter={"min_atr_pct": 0.008, "vol_expanding": True}),
            _base("vol_breakout_t35", SIDE_BOTH, threshold=0.35, top_n=2, regime="high_volatility",
                  vol_filter={"min_atr_pct": 0.010, "vol_expanding": True}),
        ],
        "mean_reversion_scalp": lambda: [
            _base("mean_rev_t35", SIDE_BOTH, threshold=0.35, top_n=2, regime="non_chop",
                  trend_filter={"overextended": True, "rsi_extreme": True}),
        ],
        "time_window_open": lambda: [
            _base("time_open_t30", SIDE_BOTH, threshold=0.30, top_n=2, entry_start="09:35", entry_end="10:30",
                  avoid_first=5, notes="Post-open liquid window"),
        ],
        "time_window_midday": lambda: [
            _base("time_midday_t40", SIDE_BOTH, threshold=0.40, top_n=1, entry_start="11:30", entry_end="14:00",
                  avoid_last=30, notes="Midday chop filter"),
        ],
        # PHASE 5: exploratory balanced variants (higher sample yield for eval only; gates remain strict)
        "PE_only_balanced": lambda: [
            _base("PE_only_balanced_t30", SIDE_PE_ONLY, threshold=0.30, top_n=6, spread=0.12, liquidity=0.0, premium_band="all", dte="all", notes="EXPLORATORY: PE balanced for sample volume"),
            _base("PE_only_balanced_t35", SIDE_PE_ONLY, threshold=0.35, top_n=5, spread=0.15, liquidity=0.0, premium_band="all", dte="all", notes="EXPLORATORY"),
        ],
        "CE_only_balanced": lambda: [
            _base("CE_only_balanced_t30", SIDE_CE_ONLY, threshold=0.30, top_n=6, spread=0.12, liquidity=0.0, premium_band="all", dte="all", notes="EXPLORATORY: CE balanced for sample volume"),
            _base("CE_only_balanced_t35", SIDE_CE_ONLY, threshold=0.35, top_n=5, spread=0.15, liquidity=0.0, premium_band="all", dte="all", notes="EXPLORATORY"),
        ],
        "BOTH_auto_balanced": lambda: [
            _base("BOTH_auto_bal_t30", SIDE_AUTO_DIRECTIONAL, threshold=0.30, top_n=8, spread=0.12, liquidity=50000.0, notes="EXPLORATORY: BOTH/AUTO balanced"),
            _base("BOTH_auto_bal_t35", SIDE_AUTO_DIRECTIONAL, threshold=0.35, top_n=6, spread=0.15, liquidity=0.0, notes="EXPLORATORY"),
        ],
        "BOTH_symmetric_balanced": lambda: [
            _base("BOTH_sym_bal_t30", SIDE_BOTH_SYMMETRIC, threshold=0.30, top_n=6, spread=0.12, liquidity=0.0, notes="EXPLORATORY"),
            _base("BOTH_sym_bal_t35", SIDE_BOTH_SYMMETRIC, threshold=0.35, top_n=5, spread=0.15, liquidity=0.0, notes="EXPLORATORY"),
        ],
        "liquidity_realistic_balanced": lambda: [
            _base("liq_real_bal_t30", SIDE_BOTH, threshold=0.30, top_n=5, spread=0.12, liquidity=50000.0, notes="EXPLORATORY: realistic liquidity relaxed for coverage"),
        ],
        "cost_survivor_balanced": lambda: [
            _base("cost_surv_bal_t30", SIDE_AUTO_DIRECTIONAL, threshold=0.30, top_n=7, spread=0.13, liquidity=0.0, notes="EXPLORATORY: tuned around cost_survivor_label_v2 coverage"),
        ],
        # PHASE 10: improved entry filter families (strict gates kept; exploratory if high volume risk or unproven)
        "realistic_liquidity_low_spread": lambda: [
            _base("real_liq_low_spread_t30", SIDE_BOTH, threshold=0.30, top_n=4, spread=0.08, liquidity=80000.0, premium_band="all", dte="all", notes="candidate: realistic liq + tight spread for production-like"),
        ],
        "high_volume_low_spread": lambda: [
            _base("hi_vol_low_spread_t35", SIDE_AUTO_DIRECTIONAL, threshold=0.35, top_n=5, spread=0.07, liquidity=150000.0, notes="EXPLORATORY: high vol filter"),
        ],
        "near_atm_cost_survivor": lambda: [
            _base("near_atm_cs_t30", SIDE_BOTH, threshold=0.30, top_n=4, spread=0.10, liquidity=60000.0, premium_band="mid", dte="2-7", notes="near ATM + cost survivor focus"),
        ],
        "expiry_safe_intraday": lambda: [
            _base("exp_safe_t35", SIDE_AUTO_DIRECTIONAL, threshold=0.35, top_n=3, spread=0.09, liquidity=50000.0, dte="0-1", avoid_last=45, notes="EXPLORATORY: expiry day safe window"),
        ],
        "late_morning_momentum": lambda: [
            _base("late_morn_mom_t30", SIDE_BOTH, threshold=0.30, top_n=5, spread=0.11, liquidity=0.0, entry_start="10:30", entry_end="12:00", notes="EXPLORATORY: momentum window"),
        ],
        "afternoon_decay_safe": lambda: [
            _base("afr_decay_safe_t40", SIDE_AUTO_DIRECTIONAL, threshold=0.40, top_n=3, spread=0.10, liquidity=70000.0, entry_start="13:30", entry_end="15:00", avoid_last=20, notes="decay safe afternoon"),
        ],
        "balanced_ce_pe_router": lambda: [
            _base("bal_ce_pe_t30", SIDE_AUTO_DIRECTIONAL, threshold=0.30, top_n=4, spread=0.10, liquidity=40000.0, regime_detail={"bullish": "CE", "bearish": "PE", "mixed": "BOTH"}, notes="balanced side router"),
        ],
        "high_confidence_topn5": lambda: [
            _base("hi_conf_top5_t40", SIDE_BOTH, threshold=0.40, top_n=5, spread=0.08, liquidity=120000.0, notes="EXPLORATORY: higher topn with high conf"),
        ],
        "stable_threshold_router": lambda: [
            _base("stable_thr_t35", SIDE_AUTO_DIRECTIONAL, threshold=0.35, top_n=4, spread=0.09, liquidity=50000.0, notes="stable thresh focus for robustness"),
        ],
    }
    fn = builders.get(preset_family)
    if fn is None:
        return []
    return fn()


def build_all_preset_families() -> Dict[str, List[DynamicPreset]]:
    families = [
        "PE_only_conservative",
        "CE_only_conservative",
        "BOTH_directional_auto",
        "BOTH_symmetric",
        "high_confidence_low_frequency",
        "mid_confidence_high_liquidity",
        "volatility_breakout",
        "mean_reversion_scalp",
        "time_window_open",
        "time_window_midday",
        "PE_only_balanced",
        "CE_only_balanced",
        "BOTH_auto_balanced",
        "BOTH_symmetric_balanced",
        "liquidity_realistic_balanced",
        "cost_survivor_balanced",
        "realistic_liquidity_low_spread",
        "high_volume_low_spread",
        "near_atm_cost_survivor",
        "expiry_safe_intraday",
        "late_morning_momentum",
        "afternoon_decay_safe",
        "balanced_ce_pe_router",
        "high_confidence_topn5",
        "stable_threshold_router",
    ]
    return {f: build_preset_family(f) for f in families}


# ---------------------------------------------------------------------------
# Candidate matrix (model x preset family)
# ---------------------------------------------------------------------------

CANDIDATE_MATRIX: List[Dict[str, str]] = [
    {"model_name": "elasticnet", "side_policy": SIDE_PE_ONLY, "preset_family": "PE_only_conservative"},
    {"model_name": "elasticnet", "side_policy": SIDE_CE_ONLY, "preset_family": "CE_only_conservative"},
    {"model_name": "elasticnet", "side_policy": SIDE_AUTO_DIRECTIONAL, "preset_family": "BOTH_directional_auto"},
    {"model_name": "logistic_regression", "side_policy": SIDE_AUTO_DIRECTIONAL, "preset_family": "BOTH_directional_auto"},
    {"model_name": "calibrated_logistic_regression", "side_policy": SIDE_BOTH_SYMMETRIC, "preset_family": "BOTH_symmetric"},
    {"model_name": "hist_gradient_boosting", "side_policy": SIDE_AUTO_DIRECTIONAL, "preset_family": "BOTH_directional_auto"},
    {"model_name": "extra_trees", "side_policy": SIDE_BOTH, "preset_family": "high_confidence_low_frequency"},
    {"model_name": "random_forest", "side_policy": SIDE_BOTH, "preset_family": "high_confidence_low_frequency"},
    {"model_name": "xgboost", "side_policy": SIDE_BOTH, "preset_family": "high_confidence_low_frequency"},
    {"model_name": "xgboost", "side_policy": SIDE_BOTH, "preset_family": "volatility_breakout"},
    {"model_name": "weighted_ensemble", "side_policy": SIDE_AUTO_DIRECTIONAL, "preset_family": "BOTH_directional_auto"},
    # PHASE 5 exploratory balanced (entry filters loosened for sample volume only; final gates unchanged, candidates marked exploratory)
    {"model_name": "elasticnet", "side_policy": SIDE_PE_ONLY, "preset_family": "PE_only_balanced"},
    {"model_name": "elasticnet", "side_policy": SIDE_CE_ONLY, "preset_family": "CE_only_balanced"},
    {"model_name": "elasticnet", "side_policy": SIDE_AUTO_DIRECTIONAL, "preset_family": "BOTH_auto_balanced"},
    {"model_name": "logistic_regression", "side_policy": SIDE_AUTO_DIRECTIONAL, "preset_family": "BOTH_auto_balanced"},
    {"model_name": "xgboost", "side_policy": SIDE_BOTH, "preset_family": "cost_survivor_balanced"},
    # PHASE 10 new improved families (mix of production-candidate and exploratory)
    {"model_name": "elasticnet", "side_policy": SIDE_BOTH, "preset_family": "realistic_liquidity_low_spread"},
    {"model_name": "logistic_regression", "side_policy": SIDE_AUTO_DIRECTIONAL, "preset_family": "balanced_ce_pe_router"},
    {"model_name": "calibrated_logistic_regression", "side_policy": SIDE_AUTO_DIRECTIONAL, "preset_family": "afternoon_decay_safe"},
    {"model_name": "xgboost", "side_policy": SIDE_BOTH, "preset_family": "near_atm_cost_survivor"},
    {"model_name": "extra_trees", "side_policy": SIDE_BOTH, "preset_family": "high_confidence_topn5"},
]

MIN_TRADES_GATE = 100
MIN_PF_NET = 1.0
MIN_COST_15X_PF = 1.0
MIN_SIDE_TRADES = 25


def evaluate_candidate_gates(metrics: Dict[str, Any], *, side_policy: str) -> Dict[str, Any]:
    """Evaluate minimum deployment gates for a candidate+preset combo."""
    gates: Dict[str, bool] = {}
    pe_trades = int(metrics.get("pe_trade_count", 0) or 0)
    ce_trades = int(metrics.get("ce_trade_count", 0) or 0)
    total = int(metrics.get("total_trade_count", metrics.get("trade_count", 0)) or 0)
    pf_net = float(metrics.get("pf_net", metrics.get("net_pf", 0)) or 0)
    cost_15 = float(metrics.get("cost_1_5x_pf", metrics.get("cost_1.50x_pf", 0)) or 0)
    leakage = bool(metrics.get("leakage_pass", True))
    live_feat = bool(metrics.get("live_feature_pass", True))

    gates["enough_trades"] = total >= MIN_TRADES_GATE
    gates["pf_net_positive"] = pf_net >= MIN_PF_NET
    gates["cost_1_5x_survival"] = cost_15 >= MIN_COST_15X_PF
    gates["leakage_pass"] = leakage
    gates["live_feature_pass"] = live_feat
    gates["not_tiny_trade_count"] = total >= 50

    if side_policy == SIDE_PE_ONLY:
        gates["side_trade_count"] = pe_trades >= MIN_SIDE_TRADES and ce_trades == 0
    elif side_policy == SIDE_CE_ONLY:
        gates["side_trade_count"] = ce_trades >= MIN_SIDE_TRADES and pe_trades == 0
    else:
        gates["side_trade_count"] = (pe_trades >= MIN_SIDE_TRADES or ce_trades >= MIN_SIDE_TRADES)

    passed = sum(1 for v in gates.values() if v)
    failed = sum(1 for v in gates.values() if not v)
    # FIXED: legacy >=10 was impossible (only 7 gates ever set). shadow_ready now = all defined gates pass.
    shadow_ready = failed == 0

    reject_reasons = [k for k, v in gates.items() if not v]
    return {
        "gates": gates,
        "gate_pass_count": passed,
        "gate_fail_count": failed,
        "shadow_ready": shadow_ready,
        "reject_reasons": reject_reasons,
    }


# Authoritative 3-way promotion contract (PHASE 2)
SHADOW_MIN_TRADES = 200   # for robust shadow (paper_forward_only allowed for 100-199 if other gates pass)
PAPER_MIN_TRADES = 50

def evaluate_shadow_readiness(metrics: Dict[str, Any], *, side_policy: str, is_exploratory: bool = False, artifacts_complete: bool = True) -> Dict[str, Any]:
    """One central gate contract. Returns shadow_ready / paper_forward_only / reject with clear reasons.
    Never lowers thresholds. Used by matrix reports, tests, and promotion.
    """
    pe = int(metrics.get("pe_trade_count", 0) or 0)
    ce = int(metrics.get("ce_trade_count", 0) or 0)
    total = int(metrics.get("total_trade_count", metrics.get("trade_count", 0)) or 0)
    pf_net = float(metrics.get("pf_net", metrics.get("net_pf", 0)) or 0)
    c15 = float(metrics.get("cost_1_5x_pf", metrics.get("cost_1.50x_pf", 0)) or 0)
    c2 = float(metrics.get("cost_2x_pf", 0) or 0)
    net_ret = float(metrics.get("net_return", metrics.get("true_net_return", 0)) or 0)
    wr = float(metrics.get("win_rate", 0) or 0)
    avg_ret = float(metrics.get("avg_trade_return", 0) or 0)
    leak = bool(metrics.get("leakage_pass", True))
    livef = bool(metrics.get("live_feature_pass", True))
    liq = bool(metrics.get("liquidity_pass", True))
    thr_stab = float(metrics.get("threshold_stability_score", 0) or 0)

    core_gates = {}
    core_gates["pf_net_positive"] = pf_net >= 1.0
    core_gates["cost_1_5x_survival"] = c15 >= 1.0
    core_gates["cost_2x_survival"] = c2 >= 1.0   # soft for paper
    core_gates["net_return_positive"] = net_ret > 0
    core_gates["win_rate_ok"] = wr > 0.45
    core_gates["avg_trade_return_ok"] = avg_ret > 0
    core_gates["leakage_pass"] = leak
    core_gates["live_feature_pass"] = livef
    core_gates["liquidity_pass"] = liq

    # volume for shadow vs paper
    core_gates["enough_trades_for_paper"] = total >= PAPER_MIN_TRADES
    core_gates["enough_trades_for_shadow"] = total >= SHADOW_MIN_TRADES
    core_gates["side_trade_count"] = (
        (pe >= 25 and ce == 0) if side_policy == SIDE_PE_ONLY else
        (ce >= 25 and pe == 0) if side_policy == SIDE_CE_ONLY else
        (pe >= 25 or ce >= 25)
    )
    core_gates["threshold_stable"] = thr_stab >= 0.5
    core_gates["artifacts_complete"] = artifacts_complete

    failed_core = [k for k, v in core_gates.items() if not v]
    passed_core = [k for k, v in core_gates.items() if v]

    # Decision
    if not all(core_gates[k] for k in ["pf_net_positive", "cost_1_5x_survival", "net_return_positive", "leakage_pass", "live_feature_pass", "liquidity_pass", "enough_trades_for_paper", "side_trade_count", "artifacts_complete"]):
        mode = "reject"
        shadow = False
        paper = False
        reason = "; ".join(failed_core) or "core gates failed"
    elif core_gates["enough_trades_for_shadow"] and not is_exploratory and core_gates["threshold_stable"]:
        mode = "shadow_ready"
        shadow = True
        paper = False
        reason = ""
    else:
        mode = "paper_forward_only"
        shadow = False
        paper = True
        reason = "passes core but " + ("low volume for shadow (<%d)" % SHADOW_MIN_TRADES if not core_gates["enough_trades_for_shadow"] else "exploratory or threshold stability marginal") + "; safe for simulated paper observation only"

    return {
        "shadow_ready": shadow,
        "paper_forward_only": paper,
        "recommended_mode": mode,
        "gate_pass_count": len(passed_core),
        "gate_fail_count": len(failed_core),
        "passed_gates": passed_core,
        "failed_gates": failed_core,
        "reject_reason": reason,
        "artifact_complete": artifacts_complete,
        "explanation": f"total={total} pf_net={pf_net:.2f} c15={c15:.2f} wr={wr:.2f} stab={thr_stab} exploratory={is_exploratory}",
    }


def make_candidate_id(model_name: str, preset_family: str, preset_name: str, ts: Optional[str] = None) -> str:
    ts = ts or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace(" ", "_").lower()
    safe_preset = preset_name.replace(" ", "_")
    return f"{safe_model}_{preset_family}_{safe_preset}_{ts}"


def filter_name_for_side(side_policy: str) -> str:
    if side_policy == SIDE_PE_ONLY:
        return "PE_only"
    if side_policy == SIDE_CE_ONLY:
        return "CE_only"
    if side_policy in {SIDE_BOTH, SIDE_BOTH_SYMMETRIC}:
        return "BOTH"
    if side_policy == SIDE_AUTO_DIRECTIONAL:
        return "AUTO_DIRECTIONAL"
    return "BOTH"