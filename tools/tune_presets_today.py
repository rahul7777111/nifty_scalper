from __future__ import annotations

import argparse
import os
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Allow running as `python tools/tune_presets_today.py` without installing the package.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_TOOLS = _ROOT / "tools"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from config import load_strategy_config
from market_data import Candle

# Reuse backtest engine.
from backtest_today import (
    _apply_backtest_preset,
    _filter_candles_for_date,
    _load_yahoo_candles,
    run_backtest,
)


def _parse_date(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


@dataclass(frozen=True)
class TuneResult:
    score: float
    params: Dict[str, Any]
    per_tf: Dict[str, Any]


def _objective(pnl: float, max_dd: float, *, dd_penalty: float) -> float:
    return float(pnl) - float(dd_penalty) * float(max_dd)


def _sample_from(values: Iterable[Any], rng: random.Random) -> Any:
    values = list(values)
    if not values:
        raise ValueError("empty sample space")
    return values[rng.randrange(0, len(values))]


def _apply_params(cfg, params: Dict[str, Any]) -> None:
    for k, v in params.items():
        setattr(cfg, k, v)


def _eval_candidate(
    base_cfg,
    candles_by_tf: Dict[str, List[Candle]],
    *,
    horizons: List[int],
    dd_penalty: float,
    slippage_pts: float,
    fees_pts: float,
    min_expectancy: Optional[float],
    min_profit_factor: Optional[float],
    min_entries_per_tf: int,
    min_pnl_per_tf: Optional[float],
    params: Dict[str, Any],
) -> TuneResult:
    cfg = deepcopy(base_cfg)
    _apply_params(cfg, params)

    per_tf: Dict[str, Any] = {}
    total_score = 0.0
    for tf, candles in candles_by_tf.items():
        tf_details: Dict[str, Any] = {}
        tf_score_total = 0.0
        for hz in horizons:
            stats = run_backtest(
                candles,
                cfg,
                horizon=int(hz),
                slippage_pts=float(slippage_pts),
                fees_pts=float(fees_pts),
            )
            expectancy = (float(stats.total_pnl_pts) / float(stats.trades)) if int(stats.trades) > 0 else 0.0
            profit_factor = (
                float(stats.total_win_pts) / float(stats.total_loss_pts)
                if float(stats.total_loss_pts) > 0
                else (float("inf") if float(stats.total_win_pts) > 0 else 0.0)
            )
            score = _objective(stats.total_pnl_pts, stats.max_drawdown_pts, dd_penalty=float(dd_penalty))
            if int(min_entries_per_tf) > 0 and int(stats.entries) < int(min_entries_per_tf):
                # Strong penalty to avoid "no-trade" solutions on any timeframe.
                score -= 1e6
            if min_pnl_per_tf is not None and float(stats.total_pnl_pts) < float(min_pnl_per_tf):
                score -= 1e6
            if min_expectancy is not None and float(expectancy) < float(min_expectancy):
                score -= 1e6
            if min_profit_factor is not None and float(profit_factor) < float(min_profit_factor):
                score -= 1e6

            tf_details[str(int(hz))] = {
                "entries": int(stats.entries),
                "closed": int(stats.trades),
                "gross_pnl": float(stats.gross_pnl_pts),
                "costs": float(stats.total_cost_pts),
                "pnl": float(stats.total_pnl_pts),
                "max_dd": float(stats.max_drawdown_pts),
                "expectancy": float(expectancy),
                "profit_factor": float(profit_factor),
                "max_consec_losses": int(stats.max_consecutive_losses),
                "largest_loss": float(stats.largest_loss_pts),
                "score": float(score),
            }
            tf_score_total += float(score)

        per_tf[tf] = {
            "score_total": float(tf_score_total),
            "by_horizon": tf_details,
        }
        total_score += float(tf_score_total)

    return TuneResult(score=float(total_score), params=dict(params), per_tf=per_tf)


def _tune_one_preset(
    preset: str,
    candles_by_tf: Dict[str, List[Candle]],
    *,
    iters: int,
    seed: int,
    horizons: List[int],
    dd_penalty: float,
    slippage_pts: float,
    fees_pts: float,
    min_expectancy: Optional[float],
    min_profit_factor: Optional[float],
    min_entries_per_tf: int,
    min_pnl_per_tf: Optional[float],
) -> TuneResult:
    # Force a clean config and deterministic preset application.
    saved_env = dict(os.environ)
    try:
        # Clear any explicit env overrides for knobs we tune or presets control.
        clear_keys = [
            "MSTOCK_TIMEFRAME",
            "MSTOCK_MAX_TRADES_PER_DAY",
            "MSTOCK_MAX_OPEN_POSITIONS",
            "MSTOCK_COOLDOWN_SEC",
            "MSTOCK_MIN_ATR",
            "MSTOCK_MAX_ATR",
            "MSTOCK_DIR_MIN_CONFIRMATIONS",
            "MSTOCK_DIR_MIN_SCORE_DIFF",
            "MSTOCK_DIR_MOMENTUM_ATR_MULT",
            "MSTOCK_DIR_EMA_SLOPE_ATR_MULT",
            "MSTOCK_AUTO_DIR_RSI_BUY",
            "MSTOCK_AUTO_DIR_RSI_SELL",
            "MSTOCK_AUTO_DIR_RSI_SLOP",
            "MSTOCK_ENABLE_ADX_FILTER",
            "MSTOCK_ADX_MIN_STRENGTH",
            "MSTOCK_ADX_PERIOD",
        ]
        for k in clear_keys:
            os.environ.pop(k, None)

        # Set the preset for config-side default application.
        os.environ["MSTOCK_PRESET"] = str(preset)

        cfg = load_strategy_config()
        _apply_backtest_preset(cfg, preset)

        rng = random.Random(int(seed))

        # Parameter spaces (intentionally small and impactful).
        # NOTE: these affect the scoring logic used by the backtest tool.
        if preset == "conservative":
            space = {
                # Guardrails (keep conservative trade count controlled)
                "max_trades_per_day": [4, 6, 8],
                "max_open_positions": [1],
                "cooldown_sec": [60.0, 90.0, 120.0],
                "dir_min_confirmations": [4, 5, 6],
                "dir_min_score_diff": [1, 2, 3],
                "auto_dir_rsi_buy": [52.0, 53.0, 54.0, 55.0, 56.0],
                "auto_dir_rsi_sell": [44.0, 45.0, 46.0, 47.0, 48.0],
                "auto_dir_rsi_slop": [0.5, 1.0, 1.5, 2.0],
                "dir_momentum_atr_mult": [0.08, 0.10, 0.12, 0.15, 0.18],
                "dir_ema_slope_atr_mult": [0.03, 0.05, 0.08, 0.10, 0.12],
                # Entry gating (now respected by backtest):
                "min_atr": [0.0, 0.5, 1.0, 2.0, 3.0, 4.0],
                "max_atr": [45.0, 55.0, 65.0, 80.0],

                # ADX filter (key for 5m viability)
                "enable_adx_filter": [True, False],
                "adx_min_strength": [10.0, 15.0, 20.0, 25.0],
            }
        else:
            space = {
                # Guardrails (these are key for aggressive drawdown control)
                "max_trades_per_day": [10, 12, 15, 18, 20],
                "max_open_positions": [1, 2],
                "cooldown_sec": [15.0, 30.0, 45.0, 60.0],
                "dir_min_confirmations": [3, 4, 5],
                "dir_min_score_diff": [0, 1, 2],
                "auto_dir_rsi_buy": [50.0, 51.0, 52.0, 53.0, 54.0],
                "auto_dir_rsi_sell": [46.0, 47.0, 48.0, 49.0, 50.0],
                "auto_dir_rsi_slop": [0.5, 1.0, 1.5, 2.0],
                "dir_momentum_atr_mult": [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22],
                "dir_ema_slope_atr_mult": [0.0, 0.02, 0.03, 0.05, 0.08, 0.10],
                "min_atr": [0.0, 0.5, 1.0, 2.0, 3.0, 4.0],
                "max_atr": [70.0, 80.0, 95.0, 110.0, 130.0],

                # ADX filter (key for 5m viability)
                "enable_adx_filter": [True, False],
                "adx_min_strength": [10.0, 15.0, 20.0, 25.0],
            }

        # Baseline = current preset settings.
        baseline = _eval_candidate(
            cfg,
            candles_by_tf,
            horizons=list(horizons),
            dd_penalty=float(dd_penalty),
            slippage_pts=float(slippage_pts),
            fees_pts=float(fees_pts),
            min_expectancy=min_expectancy,
            min_profit_factor=min_profit_factor,
            min_entries_per_tf=int(min_entries_per_tf),
            min_pnl_per_tf=min_pnl_per_tf,
            params={},
        )
        best = baseline

        for _ in range(int(iters)):
            params = {k: _sample_from(v, rng) for k, v in space.items()}

            # Keep RSI bands sensible.
            if float(params["auto_dir_rsi_buy"]) <= float(params["auto_dir_rsi_sell"]):
                continue

            res = _eval_candidate(
                cfg,
                candles_by_tf,
                horizons=list(horizons),
                dd_penalty=float(dd_penalty),
                slippage_pts=float(slippage_pts),
                fees_pts=float(fees_pts),
                min_expectancy=min_expectancy,
                min_profit_factor=min_profit_factor,
                min_entries_per_tf=int(min_entries_per_tf),
                min_pnl_per_tf=min_pnl_per_tf,
                params=params,
            )
            if float(res.score) > float(best.score):
                best = res

        return best
    finally:
        os.environ.clear()
        os.environ.update(saved_env)


def _print_result(label: str, res: TuneResult) -> None:
    print(f"\n=== {label} ===")
    print(f"score_total: {res.score:.2f}")

    if res.params:
        print("params:")
        for k in sorted(res.params.keys()):
            print(f"  {k} = {res.params[k]}")
    else:
        print("params: (baseline preset)")

    print("per_timeframe:")
    for tf in ["1m", "3m", "5m"]:
        if tf not in res.per_tf:
            continue
        s = res.per_tf[tf]
        by_hz = s.get("by_horizon", {})
        for hz in sorted(by_hz.keys(), key=lambda x: int(x)):
            h = by_hz[hz]
            cost_part = ""
            if float(h.get("costs") or 0.0) > 0:
                cost_part = f" gross={h['gross_pnl']:.2f} costs={h['costs']:.2f}"
            pf_raw = float(h.get("profit_factor") or 0.0)
            pf = "inf" if pf_raw == float("inf") else f"{pf_raw:.2f}"
            print(
                f"  {tf} hz={hz}: pnl={h['pnl']:.2f}{cost_part} dd={h['max_dd']:.2f} exp={h['expectancy']:.2f} pf={pf} "
                f"maxL={h['max_consec_losses']} bigL={h['largest_loss']:.2f} score={h['score']:.2f} entries={h['entries']} closed={h['closed']}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune conservative/aggressive preset knobs for today's candles (1m/3m/5m).")
    ap.add_argument("--underlying", default=os.getenv("MSTOCK_UNDERLYING", "NIFTY"))
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--limit", type=int, default=1400, help="Max candles to fetch per timeframe")
    ap.add_argument("--horizon", type=int, default=3, help="Exit horizon in candles (used if no per-preset horizons are set)")
    ap.add_argument(
        "--conservative-horizons",
        default=None,
        help="Comma-separated horizons for conservative (e.g. 1,3). Default: use --horizon.",
    )
    ap.add_argument(
        "--aggressive-horizons",
        default=None,
        help="Comma-separated horizons for aggressive (e.g. 5,10). Default: use --horizon.",
    )
    ap.add_argument("--iters", type=int, default=200, help="Random search iterations per preset")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--dd-penalty", type=float, default=0.5, help="Objective = pnl - dd_penalty*maxDD")
    ap.add_argument(
        "--slippage-pts",
        type=float,
        default=0.0,
        help="Assumed one-way slippage in index points; charged on entry and exit.",
    )
    ap.add_argument(
        "--fees-pts",
        type=float,
        default=0.0,
        help="Assumed round-trip fees/taxes/brokerage converted to index points.",
    )
    ap.add_argument(
        "--min-entries-per-tf",
        type=int,
        default=1,
        help="Require at least this many entries on each of 1m/3m/5m (default: 1).",
    )
    ap.add_argument(
        "--min-pnl-per-tf",
        type=float,
        default=None,
        help="If set, require each timeframe to have pnl >= this value (else heavy penalty).",
    )
    ap.add_argument(
        "--min-expectancy",
        type=float,
        default=None,
        help="If set, require net expectancy per closed trade >= this value on every timeframe/horizon.",
    )
    ap.add_argument(
        "--min-profit-factor",
        type=float,
        default=None,
        help="If set, require net profit factor >= this value on every timeframe/horizon.",
    )
    args = ap.parse_args()

    def _parse_horizons(s: Optional[str], fallback: int) -> List[int]:
        if s is None or str(s).strip() == "":
            return [int(fallback)]
        out: List[int] = []
        for part in str(s).split(","):
            part = part.strip()
            if not part:
                continue
            out.append(int(part))
        if not out:
            out = [int(fallback)]
        return out

    cons_horizons = _parse_horizons(args.conservative_horizons, int(args.horizon))
    agg_horizons = _parse_horizons(args.aggressive_horizons, int(args.horizon))

    d = _parse_date(args.date) if args.date else datetime.now().date()

    candles_by_tf: Dict[str, List[Candle]] = {}
    for tf in ["1m", "3m", "5m"]:
        candles = _load_yahoo_candles(str(args.underlying), tf, limit=int(args.limit))
        candles = _filter_candles_for_date(candles, d)
        if not candles:
            print(f"No candles available for {args.underlying} on {d.isoformat()} timeframe={tf}.")
            return
        candles_by_tf[tf] = candles

    for preset in ["conservative", "aggressive"]:
        horizons = cons_horizons if preset == "conservative" else agg_horizons
        best = _tune_one_preset(
            preset,
            candles_by_tf,
            iters=int(args.iters),
            seed=int(args.seed) + (1 if preset == "aggressive" else 0),
            horizons=list(horizons),
            dd_penalty=float(args.dd_penalty),
            slippage_pts=float(args.slippage_pts),
            fees_pts=float(args.fees_pts),
            min_expectancy=(float(args.min_expectancy) if args.min_expectancy is not None else None),
            min_profit_factor=(float(args.min_profit_factor) if args.min_profit_factor is not None else None),
            min_entries_per_tf=int(args.min_entries_per_tf),
            min_pnl_per_tf=(float(args.min_pnl_per_tf) if args.min_pnl_per_tf is not None else None),
        )
        _print_result(
            (
                f"best_{preset} horizons={','.join(str(x) for x in horizons)} "
                f"slippage={float(args.slippage_pts):.2f} fees={float(args.fees_pts):.2f}"
            ),
            best,
        )


if __name__ == "__main__":
    main()
