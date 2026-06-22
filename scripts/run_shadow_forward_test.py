#!/usr/bin/env python3
"""
run_shadow_forward_test.py
==========================
Shadow-mode multi-candidate forward test.

In shadow mode, the router evaluates all available candidates against
live snapshots and logs what would have happened — but no paper positions
are created and no real orders are placed.

Usage
-----
    python scripts/run_shadow_forward_test.py \
        --candidate-dir models/candidates \
        --symbol NIFTY \
        --interval-sec 300 \
        --max-decisions 100 \
        --output-dir reports

Or with explicit candidate manifests:
    python scripts/run_shadow_forward_test.py \
        --candidate-manifest models/candidates/PE_only_elasticnet_.../candidate_manifest.json \
        --candidate-manifest models/candidates/CE_only_elasticnet_.../candidate_manifest.json \
        --symbol NIFTY

Safety
------
- Real trading is never enabled.
- paper_only=True and real_trading_enabled=False are enforced at manifest load.
- No real broker order calls are made.
- Shadow mode never creates paper positions.
- AGGRESSIVE_SHADOW_ONLY preset flag is allowed ONLY in shadow mode.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add src/ to path for imports
_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))

from src.candidate_manifest import load_candidate_manifest, load_candidates_from_dir
from src.router_with_presets import route_with_presets
# NEW: normalized single-candidate router (production contract)
from src.candidate_router import route_candidate_decision

_LOGGER = logging.getLogger("shadow_forward_test")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# Snapshot fetcher (mock — replace with real broker in production)
# ---------------------------------------------------------------------------

def fetch_live_snapshot(symbol: str) -> dict:
    """Fetch a live feature snapshot for the given symbol.

    This is a MOCK implementation. In production, replace with the actual
    broker API call (m.Stock Type B SDK) or live feature pipeline.

    Returns a dict with option chain + candle + context fields.
    Must NOT contain future, return, PnL, or label columns.

    Also adds market-quality indicators:
      market_quality : "good" | "acceptable" | "poor"
                       ADX >= 25 = good, ADX >= 15 = acceptable, else poor
      adx            : float
      atr_pct        : float  (ATR / spot * 100)
    """
    # Placeholder — replace with real snapshot
    snapshot = {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Live-computable fields only:
        "option_type": "PE",   # CHANGE THIS for testing
        "dte_days": 8,         # days to expiry
        "moneyness_bucket": "ITM",
        "ltp": 150.0,
        "mid_price": 148.5,
        "last_close": 24500.0,
        "volume": 50000,
        "open_interest": 120000,
        "iv": 14.5,
        "delta": -0.35,
        "theta": -8.2,
        "vega": 0.12,
        "gamma": 0.018,
        "bid": 147.5,
        "ask": 149.5,
        "bid_size": 50,
        "ask_size": 50,
        "ctx_time_sin": 0.5,
        "ctx_time_cos": 0.866,
        "ctx_day_of_week_sin": 0.5,
        "ctx_day_of_week_cos": 0.866,
        "vwap": 24480.0,
        "atr_14": 150.0,
        "spot": 24500.0,
        "trend_strength": 0.3,
        "rsi_14": 55.0,
        "volume_ratio": 1.2,
        # NOTE: add more live-computable fields as needed to match model schema
    }

    # ── Market quality estimation ──────────────────────────────────────────
    adx = float(snapshot.get("ctx_adx") or snapshot.get("adx") or 20.0)
    atr = float(snapshot.get("atr_14") or 0.0)
    spot = float(snapshot.get("spot") or snapshot.get("last_close") or 25000.0)
    atr_pct = (atr / spot) * 100.0 if spot > 0 else 0.0

    if adx >= 25.0:
        market_quality = "good"
    elif adx >= 15.0:
        market_quality = "acceptable"
    else:
        market_quality = "poor"

    snapshot["adx"] = round(adx, 2)
    snapshot["atr_pct"] = round(atr_pct, 4)
    snapshot["market_quality"] = market_quality

    return snapshot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Shadow-mode multi-candidate forward test")
    parser.add_argument("--candidate-dir", type=Path, default=None,
                        help="Directory containing candidate subdirectories")
    parser.add_argument("--candidate-manifest", type=Path, action="append", default=[],
                        help="Explicit candidate manifest path (can be repeated)")
    parser.add_argument("--symbol", default="NIFTY", help="Trading symbol")
    parser.add_argument("--interval-sec", type=int, default=300,
                        help="Seconds between decisions (default: 300 = 5 min)")
    parser.add_argument("--max-decisions", type=int, default=100,
                        help="Maximum number of decisions before exiting (default: 100)")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"),
                        help="Directory to write decision logs (default: reports/)")
    parser.add_argument("--duration-minutes", type=int, default=None,
                        help="Maximum run duration in minutes (default: unlimited)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--once", action="store_true",
                        help="Run one decision cycle and exit (for testing)")
    # --- New production router wiring flags (PHASE 4) ---
    parser.add_argument("--candidate-id", type=str, default=None,
                        help="Active candidate_id to route through route_candidate_decision (single active)")
    parser.add_argument("--candidate-dir", type=Path, default=None,
                        help="Directory containing candidate subdirs (for active candidate profile load)")
    parser.add_argument("--all-shadow-ready", action="store_true",
                        help="Only consider shadow-ready candidates (non-ready will be skipped unless --force)")
    parser.add_argument("--force-shadow-eval", action="store_true",
                        help="Force evaluation even if candidate is not shadow_ready (still no real orders)")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.candidate_dir and not args.candidate_manifest:
        _LOGGER.error("Must provide --candidate-dir or --candidate-manifest")
        sys.exit(1)

    # --- Load candidates ---
    candidates = []
    load_errors = []

    if args.candidate_dir:
        valid, errors = load_candidates_from_dir(args.candidate_dir)
        candidates.extend(valid)
        load_errors.extend(errors)

    for manifest_path in args.candidate_manifest:
        try:
            manifest = load_candidate_manifest(manifest_path)
            candidates.append(manifest)
        except Exception as exc:
            load_errors.append({"path": str(manifest_path), "error": str(exc)})

    if not candidates:
        _LOGGER.error("No valid candidates loaded. Errors:")
        for err in load_errors:
            _LOGGER.error("  %s: %s", err["path"], err["error"])
        sys.exit(1)

    _LOGGER.info("Loaded %d candidate(s): %s",
                 len(candidates),
                 [c["candidate_id"] for c in candidates])
    for err in load_errors:
        _LOGGER.warning("Candidate load warning: %s: %s", err["path"], err["error"])

    # --- Setup output ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    decisions_path = output_dir / f"router_shadow_decisions_{timestamp}.jsonl"
    summary_path = output_dir / f"router_shadow_summary_{timestamp}"

    decisions_log: list = []
    start_time = time.time()
    decision_count = 0
    trade_count = 0
    skip_count = 0

    _LOGGER.info("Shadow mode started — no real orders will be placed")
    _LOGGER.info("Output: %s", decisions_path)

    try:
        while True:
            if args.max_decisions and decision_count >= args.max_decisions:
                _LOGGER.info("Max decisions (%d) reached — exiting", args.max_decisions)
                break

            if args.duration_minutes:
                elapsed = (time.time() - start_time) / 60.0
                if elapsed >= args.duration_minutes:
                    _LOGGER.info("Duration limit (%d min) reached — exiting",
                                 args.duration_minutes)
                    break

            snapshot = fetch_live_snapshot(args.symbol)
            decision_count += 1

            # ── Build live_features dict for route_with_presets (legacy path kept for compat) ─
            live_features = {
                "bid": snapshot.get("bid"),
                "ask": snapshot.get("ask"),
                "spread": (snapshot.get("ask", 0) - snapshot.get("bid", 0))
                          if snapshot.get("bid") and snapshot.get("ask") else None,
                "mid_price": snapshot.get("mid_price"),
                "ltp": snapshot.get("ltp"),
                "stale_quote": False,  # MOCK: always False in shadow mode
            }

            # ── Run preset-aware router (existing multi-candidate path) ─────
            decision = route_with_presets(
                snapshot=snapshot,
                candidates=candidates,
                live_features=live_features,
                mode="shadow",
                min_coverage_pct=95.0,
            )

            # ── PHASE 4: EVERY shadow decision MUST also flow through the normalized candidate_router API
            # Determine active candidate for the single-active normalized path
            active_cid = args.candidate_id
            cdir = str(args.candidate_dir) if args.candidate_dir else (str(args.candidate_dir) if False else None)
            # If not explicitly given, try to pick first loaded candidate id (best effort)
            if not active_cid and candidates:
                active_cid = candidates[0].get("candidate_id")
            if not cdir and args.candidate_dir:
                cdir = str(args.candidate_dir)

            router_decision = route_candidate_decision(
                market_snapshot=snapshot,
                option_chain_snapshot=None,
                legacy_signal={"ml_prob": decision.get("probability"), "threshold": decision.get("threshold") or 0.5},
                mode="shadow",
                active_candidate_id=active_cid,
                candidate_dir=cdir,
                force_eval=bool(args.force_shadow_eval),
            )

            # Enrich legacy decision with full router fields (shadow JSONL must include all)
            for k in [
                "candidate_id", "model_name", "preset_family", "selected_preset",
                "side_policy", "side_decision", "confidence", "threshold",
                "market_regime", "volatility_state", "trend_state", "liquidity_state",
                "allowed_by_model", "allowed_by_preset", "allowed_by_side_policy",
                "allowed_by_liquidity", "allowed_by_cost", "allowed_by_risk",
                "shadow_ready", "forced_eval", "final_signal", "no_trade_reason",
            ]:
                if k in router_decision:
                    decision[k] = router_decision[k]

            # Simulated execution fields (shadow never creates real/paper positions)
            decision["simulated_entry"] = (router_decision.get("final_signal", "NO_TRADE") != "NO_TRADE")
            decision["simulated_exit"] = None
            decision["estimated_return"] = None
            decision["actual_return"] = None

            decision["decision_index"] = decision_count
            decision["snapshot_timestamp"] = snapshot.get("timestamp", "")
            decision["snapshot_option_type"] = snapshot.get("option_type", "")
            decision["market_quality"] = snapshot.get("market_quality", "unknown")
            decision["snapshot_adx"] = snapshot.get("adx")
            decision["snapshot_atr_pct"] = snapshot.get("atr_pct")

            # ── Shadow-only safety fields (always enforced — NEVER modifiable)
            decision["no_order_sent"] = True
            decision["paper_trade_created"] = False
            decision["mode"] = "shadow"
            decision["final_shadow_action"] = decision.get("router_action", "SKIP")

            # If router said NO_TRADE, force shadow action to reflect it
            if router_decision.get("final_signal") == "NO_TRADE":
                decision["router_action"] = "SKIP"
                decision["rejection_reason"] = router_decision.get("no_trade_reason") or decision.get("rejection_reason")

            if decision.get("router_action") == "TRADE" or router_decision.get("final_signal", "").startswith("BUY_"):
                trade_count += 1
                _LOGGER.info(
                    "[DECISION %d] TRADE cand=%s preset=%s prob=%.4f side=%s policy=%s thr=%.4f "
                    "shadow_ready=%s forced=%s final=%s",
                    decision_count,
                    router_decision.get("candidate_id"),
                    router_decision.get("selected_preset", "N/A"),
                    router_decision.get("confidence"),
                    router_decision.get("side_decision"),
                    router_decision.get("side_policy"),
                    router_decision.get("threshold"),
                    router_decision.get("shadow_ready"),
                    router_decision.get("forced_eval"),
                    router_decision.get("final_signal"),
                )
            else:
                skip_count += 1
                _LOGGER.info(
                    "[DECISION %d] SKIP reason=%s  cand=%s preset=%s side_policy=%s final=%s",
                    decision_count,
                    router_decision.get("no_trade_reason") or decision.get("rejection_reason"),
                    router_decision.get("candidate_id"),
                    router_decision.get("selected_preset", "N/A"),
                    router_decision.get("side_policy"),
                    router_decision.get("final_signal"),
                )

            # ── Log to JSONL ─────────────────────────────────────────────────
            decisions_log.append(decision)
            with decisions_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(decision, default=str) + "\n")

            if args.once:
                _LOGGER.info("--once specified — exiting after one decision")
                break

            time.sleep(args.interval_sec)

    except KeyboardInterrupt:
        _LOGGER.info("Interrupted by user — exiting")

    # --- Write summary ---
    elapsed_min = (time.time() - start_time) / 60.0
    summary = {
        "run_timestamp": timestamp,
        "symbol": args.symbol,
        "mode": "shadow",
        "candidate_count": len(candidates),
        "candidate_ids": [c["candidate_id"] for c in candidates],
        "total_decisions": decision_count,
        "trade_count": trade_count,
        "skip_count": skip_count,
        "trade_rate": trade_count / decision_count if decision_count else 0.0,
        "elapsed_minutes": round(elapsed_min, 2),
        "decisions_log": str(decisions_path),
        "no_order_sent": True,
        "paper_trade_created": False,
        "shadow_preset_support": True,
        "safety": {
            "real_trading_enabled": False,
            "paper_only": True,
            "no_real_order_calls": True,
            "aggressive_shadow_only_allowed": True,
            "no_order_sent_guaranteed": True,
            "no_paper_position_guaranteed": True,
        },
    }

    summary_json_path = Path(f"{summary_path}.json")
    summary_md_path = Path(f"{summary_path}.md")

    with summary_json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    with summary_md_path.open("w", encoding="utf-8") as fh:
        fh.write("# Shadow Mode Forward Test Summary\n\n")
        fh.write(f"**Run timestamp:** {timestamp}\n")
        fh.write(f"**Symbol:** {args.symbol}\n")
        fh.write("**Mode:** shadow (no real orders, no paper positions)\n\n")
        fh.write("## Preset Integration\n\n")
        fh.write("| Feature | Value |\n")
        fh.write("|---------|-------|\n")
        fh.write("| Dynamic preset selection | ✅ ENABLED |\n")
        fh.write("| Market quality estimation | ✅ ADX/ATR% based |\n")
        fh.write("| Threshold adjustment | ✅ Based on market_quality |\n")
        fh.write("| AGGRESSIVE_SHADOW_ONLY allowed | ✅ Shadow mode only |\n\n")
        fh.write(f"## Candidates Loaded ({len(candidates)})\n\n")
        for c in candidates:
            fh.write(f"- `{c['candidate_id']}` — {c.get('model_name', '?')} / {c.get('filter_name', '?')}\n")
        fh.write("\n## Statistics\n\n")
        fh.write("| Metric | Value |\n")
        fh.write("|--------|-------|\n")
        fh.write(f"| Total decisions | {decision_count} |\n")
        fh.write(f"| Trades | {trade_count} |\n")
        fh.write(f"| Skips | {skip_count} |\n")
        fh.write(f"| Trade rate | {summary['trade_rate']:.1%} |\n")
        fh.write(f"| Elapsed | {summary['elapsed_minutes']:.1f} min |\n\n")
        fh.write("## Safety Guarantees\n\n")
        fh.write("| Guarantee | Status |\n")
        fh.write("|-----------|--------|\n")
        fh.write("| No real orders sent | ✅ ALWAYS TRUE |\n")
        fh.write("| No paper positions created | ✅ ALWAYS TRUE |\n")
        fh.write("| `no_order_sent` = True | ✅ HARDCODED |\n")
        fh.write("| `paper_trade_created` = False | ✅ HARDCODED |\n")
        fh.write("| AGGRESSIVE_SHADOW_ONLY allowed | ✅ Shadow mode only |\n")
        fh.write("| `mode` field = 'shadow' | ✅ HARDCODED |\n")

    _LOGGER.info("Summary written to %s (.json and .md)", summary_path)
    _LOGGER.info("Decisions log: %s (%d entries)", decisions_path, len(decisions_log))
    print(f"\nShadow run complete: {trade_count} trades, {skip_count} skips, {decision_count} total decisions")


if __name__ == "__main__":
    main()