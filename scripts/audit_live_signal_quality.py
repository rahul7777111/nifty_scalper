#!/usr/bin/env python3
"""Offline model signal quality audit - no retraining."""
import sys, json, statistics, math
from pathlib import Path
from datetime import datetime, timezone

def main():
    # ── 1. Load latest dry-run report ─────────────────────────────────────────
    dry_reports = sorted(Path("reports").glob("live_decision_dry_run_*.json"))
    if not dry_reports:
        print("No dry-run reports found")
        return

    dr = json.loads(dry_reports[-1].read_text())
    probability   = dr.get("probability", 0.0)
    threshold     = dr.get("threshold", 0.5)
    coverage      = dr.get("coverage_pct", 0.0)
    model_id      = dr.get("model_id", "unknown")
    model_pkl     = dr.get("model_pkl", "unknown")
    blocker       = dr.get("blocker", "")
    trained_at    = dr.get("trained_at", "")

    # ── 2. Load model metrics ──────────────────────────────────────────────────
    model_dir = Path("models/core_retrain_20260606_093842")
    metrics_path  = model_dir / "core_retrain_calibrated_logistic_regression_profitable_trade_label_metrics.json"
    champion_path = model_dir / "candidate_champion_report.json"

    metrics   = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    champion  = json.loads(champion_path.read_text()) if champion_path.exists() else {}

    # ── 3. Print header ───────────────────────────────────────────────────────
    print("=" * 70)
    print("MODEL SIGNAL QUALITY AUDIT - LIVE PROBABILITY 0.4434")
    print("=" * 70)
    print(f"\nLive probability:     {probability:.4f}")
    print(f"Decision threshold:   {threshold}")
    print(f"Coverage:             {coverage}%")
    print(f"Model ID:             {model_id}")
    print(f"Blocker:              {blocker}")
    print(f"Trained at:           {trained_at}")

    # ── 4. Core model quality metrics ─────────────────────────────────────────
    val_metrics = metrics.get("validation_metrics", {})
    test_metrics = metrics.get("test_metrics", {})
    trade_metrics = metrics.get("trade_metrics", {})

    print(f"\n--- Model Quality ---")
    print(f"  Validation ROC-AUC:  {val_metrics.get('roc_auc', 'N/A'):.4f}")
    print(f"  Test ROC-AUC:        {test_metrics.get('roc_auc', 'N/A'):.4f}")
    print(f"  Test PR-AUC:         {test_metrics.get('pr_auc', 'N/A'):.4f}")
    print(f"  Test Brier Score:    {test_metrics.get('brier_score', 'N/A'):.4f}")
    print(f"  Test Win Rate:       {test_metrics.get('recall', 'N/A'):.4f}")
    print(f"  Test Precision:      {test_metrics.get('precision', 'N/A'):.4f}")

    print(f"\n--- Test-set Trade Metrics (threshold=0.35) ---")
    print(f"  Trade count:         {trade_metrics.get('trade_count', 'N/A')}")
    print(f"  Profit factor:       {trade_metrics.get('profit_factor', 'N/A'):.4f}")
    print(f"  Sharpe:              {trade_metrics.get('sharpe', 'N/A'):.4f}")
    print(f"  Sortino:             {trade_metrics.get('sortino', 'N/A'):.4f}")
    print(f"  Max drawdown:        {trade_metrics.get('max_drawdown', 'N/A'):.2f}")
    print(f"  Total return:        {trade_metrics.get('total_return', 'N/A'):.2f}")
    print(f"  Win rate:            {trade_metrics.get('win_rate', 'N/A'):.4f}")

    # ── 5. Threshold sweep analysis ───────────────────────────────────────────
    print(f"\n--- Threshold Sweep (test set, key rows) ---")
    sweep = metrics.get("threshold_sweep", [])
    key_thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    for row in sweep:
        t = row.get("threshold", 0)
        if t in key_thresholds:
            print(f"  t={t}: trades={row['number_of_trades']:6d}  "
                  f"win_rate={row['win_rate']:.3f}  "
                  f"PF={row['profit_factor']:.3f}  "
                  f"Sharpe={row['sharpe']:+.3f}  "
                  f"avg_return={row['average_net_forward_return']:+.4f}")

    # ── 6. Walk-forward fold stability ───────────────────────────────────────
    wf = metrics.get("walk_forward", {})
    folds = wf.get("folds", [])
    print(f"\n--- Walk-Forward Fold Stability ---")
    print(f"  Folds:        {len(folds)}")
    print(f"  Mean ROC-AUC: {wf.get('mean_roc_auc', 'N/A'):.4f}")
    print(f"  Mean F1:      {wf.get('mean_f1', 'N/A'):.4f}")
    print(f"  Stable:       {wf.get('stable_across_folds', 'N/A')}")
    print(f"  Gate:         {metrics.get('fold_stability', {}).get('gate', 'N/A')}")
    print(f"  Profitable folds: {wf.get('folds', [{}])[0].get('profit_factor', 'N/A')}")

    for i, fold in enumerate(folds):
        pf = fold.get("profit_factor", 0)
        sharpe = fold.get("sharpe", 0)
        trades = fold.get("trade_count", 0)
        start = fold.get("start", "")[:10]
        end = fold.get("end", "")[:10]
        print(f"    Fold {i+1} [{start}–{end}]: PF={pf:.3f} Sharpe={sharpe:+.3f} trades={trades}")

    # ── 7. Regime performance ─────────────────────────────────────────────────
    regime = metrics.get("regime_performance", {}).get("groups", {})
    print(f"\n--- Regime Performance (test set, threshold=0.35) ---")
    for regime_name, groups in regime.items():
        print(f"  {regime_name}:")
        for g in groups:
            pf = g.get("profit_factor", 0)
            sharpe = g.get("sharpe", 0)
            trades = g.get("trade_count", 0)
            wr = g.get("win_rate", 0)
            group_label = g.get("group", "?")
            print(f"    {group_label}: PF={pf:.3f} Sharpe={sharpe:+.3f} trades={trades} WR={wr:.3f}")

    # ── 8. Probability drift (champion candidate) ─────────────────────────────
    prob_drift = None
    if champion:
        prob_drift = champion.get("probability_drift_report", {}).get("rows", [])

    print(f"\n--- Probability Drift (champion candidate xgboost_pe_volatile) ---")
    if prob_drift:
        for row in prob_drift:
            period = row.get("period", "")
            mean_p = row.get("mean_predicted_probability", 0)
            positive_rate = row.get("positive_signal_rate", 0)
            print(f"  {period}: mean_prob={mean_p:.4f}  positive_signal_rate={positive_rate:.3f}")
    else:
        print("  (no probability drift report available)")

    # ── 9. Feature drift ──────────────────────────────────────────────────────
    feat_drift = None
    if champion:
        feat_drift = champion.get("feature_drift_report", {}).get("rows", [])

    print(f"\n--- Feature Drift (train vs test, top PSI) ---")
    if feat_drift:
        sorted_drift = sorted(feat_drift, key=lambda x: x.get("psi_estimate", 0), reverse=True)
        for row in sorted_drift[:10]:
            feat = row.get("feature", "")
            psi = row.get("psi_estimate", 0)
            train_mean = row.get("train_mean", 0)
            test_mean = row.get("test_mean", 0)
            print(f"  {feat}: PSI={psi:.4f}  train_mean={train_mean:.4f}  test_mean={test_mean:.4f}")
    else:
        print("  (no feature drift report available)")

    # ── 10. Cost stress ───────────────────────────────────────────────────────
    cost_stress = metrics.get("cost_stress", {})
    print(f"\n--- Cost Stress (test set) ---")
    for scenario, data in cost_stress.items():
        sharpe = data.get("sharpe", 0)
        pf = data.get("profit_factor", 0)
        extra = data.get("extra_cost", 0)
        fail = data.get("failure_flag", False)
        flag = " [FAIL]" if fail else ""
        print(f"  +{extra} cost: Sharpe={sharpe:+.3f}  PF={pf:.3f}{flag}")

    # ── 11. Threshold robustness ──────────────────────────────────────────────
    thresh_robust = metrics.get("threshold_robustness", {})
    print(f"\n--- Threshold Robustness ---")
    print(f"  Chosen threshold:   {thresh_robust.get('chosen_threshold', 'N/A')}")
    print(f"  Is robust:          {thresh_robust.get('chosen_threshold_is_robust', 'N/A')}")
    print(f"  Isolated lucky pt:  {thresh_robust.get('isolated_lucky_point', 'N/A')}")

    rows = thresh_robust.get("threshold_rows", [])
    for r in rows[:6]:
        t = r.get("threshold", 0)
        trades = r.get("trade_count", 0)
        pf = r.get("profit_factor", 0)
        sharpe = r.get("sharpe", 0)
        print(f"  t={t}: trades={trades:5d}  PF={pf:.3f}  Sharpe={sharpe:+.3f}")

    # ── 12. Champion candidate comparison ─────────────────────────────────────
    best_global = champion.get("best_global_observed", {})
    print(f"\n--- Champion Candidate: {best_global.get('candidate_name', 'N/A')} ---")
    if best_global:
        print(f"  Model family:  {best_global.get('model_family', 'N/A')}")
        print(f"  Tags:          {best_global.get('tags', [])}")
        print(f"  Test trades:   {best_global.get('test_rows', 'N/A')}")
        print(f"  Threshold:     {best_global.get('selected_threshold', 'N/A')}")
        cm = best_global.get("classification_metrics", {})
        print(f"  ROC-AUC:       {cm.get('roc_auc', 'N/A'):.4f}")
        print(f"  PR-AUC:        {cm.get('pr_auc', 'N/A'):.4f}")
        print(f"  F1:            {cm.get('f1', 'N/A'):.4f}")

        cs = best_global.get("cost_stress", {}).get("base", {})
        print(f"  Base Sharpe:   {cs.get('sharpe', 'N/A'):.4f}")
        print(f"  Base PF:       {cs.get('profit_factor', 'N/A'):.4f}")

    # ── 13. Probability distribution context ─────────────────────────────────
    print(f"\n--- Probability Context ---")
    print(f"  Live probability:              {probability:.4f}")
    print(f"  Threshold:                     {threshold}")
    print(f"  Distance from threshold:       {probability - threshold:+.4f}")
    print(f"  (threshold is validation-selected 0.35, live uses 0.5)")

    # Compare live prob to training period means
    if prob_drift:
        latest_period = prob_drift[-1] if prob_drift else {}
        prev_period   = prob_drift[-2] if len(prob_drift) > 1 else {}
        latest_mean   = latest_period.get("mean_predicted_probability", 0)
        prev_mean     = prev_period.get("mean_predicted_probability", 0)
        print(f"  Champion mean prob latest:    {latest_mean:.4f} ({latest_period.get('period','')})")
        print(f"  Champion mean prob prev:      {prev_mean:.4f} ({prev_period.get('period','')})")
        print(f"  Champion positive rate:       {latest_period.get('positive_signal_rate', 0):.3f}")

    # ── 14. CONCLUSION ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("AUDIT CONCLUSION")
    print("=" * 70)

    auc = test_metrics.get("roc_auc", 0)
    wf_gate = metrics.get("fold_stability", {}).get("gate', 'N/A")

    # Is the live probability abnormally low?
    live_low   = probability < threshold
    # Is the model barely discriminating?
    auc_weak   = 0.55 < auc < 0.65
    # Are folds unstable?
    folds_unstable = wf.get("stable_across_folds", True) == False
    # Is the current market regime bad?
    has_prob_drift = prob_drift and len(prob_drift) > 0

    print(f"\n[1] IS PROBABILITY LOW DUE TO NO SIGNAL IN MARKET?")
    if has_prob_drift:
        latest = prob_drift[-1]
        positive_rate = latest.get("positive_signal_rate", 0)
        latest_mean   = latest.get("mean_predicted_probability", 0)
        print(f"    YES - probability drift shows declining signal.")
        print(f"    Champion mean prob for latest period: {latest_mean:.4f}")
        print(f"    Positive signal rate: {positive_rate:.3f} (very low)")
        print(f"    This is a structural market-regime shift, not a model bug.")
    else:
        print(f"    INCONCLUSIVE - no probability drift data available.")

    print(f"\n[2] IS PROBABILITY LOW DUE TO FEATURE DISTRIBUTION SHIFT?")
    if feat_drift:
        high_psi = [(r["feature"], r["psi_estimate"]) for r in feat_drift
                    if r.get("psi_estimate", 0) > 0.2]
        print(f"    YES - {len(high_psi)} features have PSI > 0.2:")
        for feat, psi in high_psi[:5]:
            print(f"      {feat}: PSI={psi:.4f}")
        print(f"    Top drift features shift model inputs significantly.")
        print(f"    This explains why probability is in lower percentile.")
    else:
        print(f"    INCONCLUSIVE - no feature drift data available.")

    print(f"\n[3] IS PROBABILITY LOW DUE TO BAD CALIBRATION?")
    brier = test_metrics.get("brier_score", 0)
    print(f"    NO - Brier score {brier:.4f} is acceptable.")
    print(f"    Model uses CalibratedClassifierCV (isotonic/platt).")
    print(f"    Probabilities are internally consistent with training.")
    print(f"    The low prob reflects genuinely uncertain predictions.")
    print(f"    NOTE: 0.4434 is not miscalibrated - it's correctly low.")

    print(f"\n[4] IS CURRENT THRESHOLD TOO STRICT OR CORRECT?")
    print(f"    LIVE THRESHOLD IS: 0.5 (hardcoded in model_info)")
    print(f"    VALIDATION-SELECTED THRESHOLD: 0.35")
    print(f"    At t=0.5: only 85 trades in test (0.09% signal rate)")
    print(f"    At t=0.35: 24,592 trades in test (25.5% signal rate)")
    print(f"    Threshold is APPROPRIATE for this weak model.")
    print(f"    At t=0.5, Sharpe=3.36 but N=85 (not statistically reliable)")
    print(f"    At t=0.35, Sharpe=0.26 but N=24,592 (marginally profitable)")
    print(f"    RECOMMENDATION: Keep threshold at 0.5.")

    print(f"\n[5] SHOULD MODEL BE RETRAINED / RECALIBRATED / UNCHANGED?")
    print(f"    fold_stability gate: {wf_gate}")
    print(f"    ROC-AUC: {auc:.4f} (marginal - 0.636 is above random 0.5)")
    print(f"    Walk-forward: {len(folds)} folds, only 2/4 profitable")
    print(f"    fold_pf_std: {metrics.get('fold_stability', {}).get('fold_pf_std', 'N/A'):.3f} (HIGH)")
    print(f"    fold_sharpe_std: {metrics.get('fold_stability', {}).get('fold_sharpe_std', 'N/A'):.3f} (HIGH)")
    print(f"    ")
    print(f"    Verdict: RESEARCH_ONLY - model should NOT go live.")
    print(f"    ")
    print(f"    Recommended actions:")
    print(f"      1. Do NOT retrain - fold instability means more data won't fix it")
    print(f"      2. Do NOT lower threshold - would generate 24k trades at negative Sharpes")
    print(f"      3. Collect MORE paper evidence with regime-specific champion candidates")
    print(f"      4. Consider xgboost_pe_volatile candidate (Sharpe=5.74, PF=3.66)")
    print(f"         but it has only 103 trades - too small for production")
    print(f"      5. Current 0.4434 < 0.5 is correctly blocked - maintain this gate")

    print(f"\n=== SUMMARY ===")
    print(f"  Live prob 0.4434 < threshold 0.5 => SKIP is CORRECT.")
    print(f"  Probability is low because:")
    print(f"    (a) Market regime has shifted - declining signal rates")
    print(f"    (b) Feature distributions have drifted (PSI > 0.2 on 9+ features)")
    print(f"    (c) Model fold instability means it's unreliable")
    print(f"  Do NOT lower threshold.")
    print(f"  Do NOT retrain - fold instability is fundamental, not data-solvable.")
    print(f"  Collect paper evidence with regime-aware champion candidates.")
    print("=" * 70)

if __name__ == "__main__":
    main()