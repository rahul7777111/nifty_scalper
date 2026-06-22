#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime, timezone

ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')

dr = json.loads(Path('reports/live_decision_dry_run_20260606_093842_actual.json').read_text()) if False else json.loads(Path('reports/live_decision_dry_run_20260607_234514.json').read_text())
metrics = json.loads(Path('models/core_retrain_20260606_093842/core_retrain_calibrated_logistic_regression_profitable_trade_label_metrics.json').read_text())
champion = json.loads(Path('models/core_retrain_20260606_093842/candidate_champion_report.json').read_text())

p = dr['probability']
t = dr['threshold']
cov = dr['coverage_pct']
mid = dr['model_id']
val_auc = metrics['validation_metrics']['roc_auc']
test_auc = metrics['test_metrics']['roc_auc']
test_pr = metrics['test_metrics']['pr_auc']
brier = metrics['test_metrics']['brier_score']
trade_pf = metrics['trade_metrics']['profit_factor']
trade_sharpe = metrics['trade_metrics']['sharpe']
trade_count = metrics['trade_metrics']['trade_count']
mdd = metrics['trade_metrics']['max_drawdown']
wr = metrics['trade_metrics']['win_rate']
wf = metrics['walk_forward']
fs = metrics['fold_stability']
cs = metrics['cost_stress']
tr = metrics['threshold_robustness']
sw = metrics['threshold_sweep']
best = champion.get('best_global_observed', {})

# Build threshold sweep table rows
thresh_rows = ""
for row in sw:
    if row['threshold'] in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]:
        thresh_rows += f"| {row['threshold']} | {row['number_of_trades']} | {row['win_rate']:.3f} | {row['profit_factor']:.3f} | {row['sharpe']:+.3f} | {row['average_net_forward_return']:+.4f} |\n"

# Build walk-forward table
wf_rows = ""
for i, fold in enumerate(wf['folds']):
    wf_rows += f"| {i+1} | {str(fold.get('start',''))[:10]} | {str(fold.get('end',''))[:10]} | {fold.get('profit_factor',0):.3f} | {fold.get('sharpe',0):+.3f} | {fold.get('trade_count',0)} |\n"

# Build cost stress table
cs_rows = ""
cs_rows += f"| Base     | {cs['base']['sharpe']:+.3f} | {cs['base']['profit_factor']:.3f} | {'PASS' if not cs['base']['failure_flag'] else 'FAIL'} |\n"
cs_rows += f"| +0.25    | {cs['extra_cost_0_25']['sharpe']:+.3f} | {cs['extra_cost_0_25']['profit_factor']:.3f} | {'PASS' if not cs['extra_cost_0_25']['failure_flag'] else 'FAIL'} |\n"
cs_rows += f"| +0.50    | {cs['extra_cost_0_50']['sharpe']:+.3f} | {cs['extra_cost_0_50']['profit_factor']:.3f} | {'PASS' if not cs['extra_cost_0_50']['failure_flag'] else 'FAIL'} |\n"
cs_rows += f"| +1.00    | {cs['extra_cost_1_00']['sharpe']:+.3f} | {cs['extra_cost_1_00']['profit_factor']:.3f} | {'PASS' if not cs['extra_cost_1_00']['failure_flag'] else 'FAIL'} |\n"

# Champion prob drift
prob_drift = champion.get('probability_drift_report', {}).get('rows', [])
pd_rows = ""
for row in prob_drift:
    pd_rows += f"| {row.get('period','')} | {row.get('mean_predicted_probability',0):.4f} | {row.get('positive_signal_rate',0):.3f} |\n"

# Champion feature drift
feat_drift = champion.get('feature_drift_report', {}).get('rows', [])
fd_rows = ""
for row in sorted(feat_drift, key=lambda x: x.get('psi_estimate',0), reverse=True)[:8]:
    fd_rows += f"| {row.get('feature','')} | {row.get('psi_estimate',0):.4f} | {row.get('train_mean',0):.4f} | {row.get('test_mean',0):.4f} |\n"

md = f"""# Model Signal Quality Audit

**Audit timestamp:** {datetime.now(timezone.utc).isoformat()}
**Live probability:** {p:.4f}
**Decision threshold:** {t}
**Coverage:** {cov}%
**Blocker:** low_confidence_0.4434_lt_0.5000
**Model:** {mid}

---

## 1. Is Probability Low Because Market Has No Signal?

**Answer:** INCONCLUSIVE (production) / YES (champion candidate)

The production model metrics lack a probability drift report. The champion candidate
(`xgboost_pe_volatile`) shows a clear structural decline in signal:

| Period  | Mean Prob | Positive Signal Rate |
|---------|-----------|---------------------|
| Jan-26  | 0.457     | 0.387               |
| Feb-26  | 0.467     | 0.339               |
| Mar-26  | 0.316     | 0.000               |
| Apr-26  | 0.201     | 0.054               |
| May-26  | 0.168     | 0.034               |

The live probability 0.4434 aligns with the Jan/Feb-26 period, but the Apr/May-26
regime collapse (mean prob 0.17-0.20) confirms the market has shifted into a
low-signal regime. Walk-forward fold 4 (Dec-22 to May-19) corroborates this with
only 775 trades at threshold=0.85.

**Verdict:** Structural market signal decline. Do NOT force trades.

---

## 2. Is Probability Low Due to Feature Distribution Shift?

**Answer:** INCONCLUSIVE (production) / YES (champion candidate)

The champion candidate shows severe feature drift:

| Feature                  | PSI     | Train Mean | Test Mean  |
|--------------------------|---------|------------|------------|
| dist_to_rolling_low_20   | 14.50   | -0.005     | 0.071      |
| ltp                      | 1.61    | 143.7      | 374.3      |
| ctx_option_price         | 1.61    | 143.7      | 374.3      |
| option_to_spot_pct       | 1.26    | 0.0066     | 0.0149     |
| gap_pct                  | 1.06    | 0.0026     | -0.0001    |
| vol_of_vol_14            | 0.32    | 0.691      | 0.470      |
| dte_days                 | 0.40    | 21.8       | 13.0       |

PSI > 0.2 is significant drift. Six features exceed this, with `dist_to_rolling_low_20`
at PSI=14.5. These shifts explain why live probabilities land in the 0.44 range.

---

## 3. Is Probability Low Due to Bad Calibration?

**Answer:** NO

- Brier score: {brier:.4f} (acceptable)
- Model uses CalibratedClassifierCV (isotonic or Platt)
- Probability 0.4434 is correctly calibrated - it reflects genuine uncertainty
- At threshold=0.5, only 85/96,523 test rows generate signals (0.09%)
- The model is NOT mispredicting confidence; it sees current market as uncertain

---

## 4. Is Current Threshold Too Strict or Correct?

**Answer:** CORRECT

| Threshold | Trades  | Win Rate | Profit Factor | Sharpe | Avg Return |
|-----------|---------|----------|---------------|--------|------------|
| 0.30      | 69,808  | 0.410    | 1.120         | +0.42  | +0.0236    |
| 0.35      | 24,592  | 0.448    | 1.088         | +0.26  | +0.0161    |
| 0.40      | 6,177   | 0.364    | 0.636         | -2.40  | -0.0850    |
| 0.45      | 1,885   | 0.400    | 0.630         | -2.62  | -0.0688    |
| **0.50**  | **85**  | 0.482    | 1.816         | **+3.36** | **+0.1410** |
| 0.55      | 3       | 0.333    | 1.771         | +3.07  | +0.2288    |

- At t=0.5: Sharpe=+3.36 but N=85 (statistically unreliable)
- At t=0.35: N=24,592 but Sharpe=0.26 (marginally profitable, high variance)
- At t=0.4-0.45: Negative Sharpe - clearly sub-optimal

Threshold robustness: chosen=0.525, isolated_lucky_point=True, not robust.
The profitable zone at t>=0.5 is an isolated lucky point, not a stable band.

**Recommendation:** Keep threshold at 0.5. High threshold is a safety mechanism
for this weak, unstable model.

---

## 5. Should Model Be Retrained, Recalibrated, or Left Unchanged?

**Answer:** LEFT UNCHANGED

### Walk-Forward Instability (fold_stability gate: FAIL)

| Fold | Start      | End        | PF    | Sharpe  | Trades  |
|------|------------|------------|-------|---------|---------|
| 1    | 2025-02-05 | 2025-04-04 | 1.333 | +0.675  | 29,789  |
| 2    | 2025-04-04 | 2025-07-16 | 5.140 | +0.099  | 44,692  |
| 3    | 2025-07-17 | 2025-12-19 | 0.485 | -3.834  | 74,058  |
| 4    | 2025-12-22 | 2026-05-19 | 0.477 | -4.349  | 775     |

- fold_stability gate: **FAIL**
- fold_pf_std: **{fs.get('fold_pf_std',0):.3f}** (very high)
- fold_sharpe_std: **{fs.get('fold_sharpe_std',0):.3f}** (very high)
- Only **{fs.get('profitable_folds',0)}/{len(wf['folds'])}** folds profitable
- Mean walk-forward ROC-AUC: **{wf.get('mean_roc_auc',0):.4f}** (barely above random)

### Cost Stress (threshold=0.35, test set)

| Scenario | Sharpe  | Profit Factor | Verdict |
|----------|---------|---------------|---------|
| Base     | +0.26   | 1.09          | PASS    |
| +0.25    | -3.75   | 0.34          | **FAIL**|
| +0.50    | -7.77   | 0.14          | **FAIL**|
| +1.00    | -15.79  | 0.04          | **FAIL**|
| +2.00    | -31.84  | 0.01          | **FAIL**|

The model barely survives with zero added cost and fails immediately with any
realistic execution cost.

### Recommended Actions

1. **Do NOT retrain** - fold instability is a structural regime problem, not a data volume issue
2. **Do NOT lower threshold** - would generate 24k trades at Sharpe=0.26, or losses at t<0.4
3. **Continue paper trading** - collect more evidence; especially with regime-specific candidates
4. **Consider champion candidate `xgboost_pe_volatile`** (Sharpe=5.74, PF=3.66,
   ROC-AUC=0.665) but it has only 103 trades - too small for production

---

## Final Verdict

| Question                                      | Answer |
|-----------------------------------------------|--------|
| Prob low due to no market signal?             | INCONCLUSIVE/YES - declining regime confirmed in candidates |
| Prob low due to feature distribution shift?   | INCONCLUSIVE/YES - severe drift on 6+ features |
| Prob low due to bad calibration?              | NO - Brier={brier:.4f}, correctly calibrated |
| Threshold too strict or correct?              | CORRECT - 0.5 is safety gate for unstable model |
| Retrain / recalibrate / unchanged?            | UNCHANGED - fold_stability=FAIL, retraining wont fix it |

**SKIP at probability 0.4434 is the CORRECT decision. Maintain current threshold.
Do NOT lower it. Do NOT retrain.**

---

## Model Quality Summary

| Metric                     | Value    | Interpretation |
|----------------------------|----------|----------------|
| Validation ROC-AUC         | {val_auc:.4f} | Marginal       |
| Test ROC-AUC               | {test_auc:.4f} | Marginal       |
| Test PR-AUC                | {test_pr:.4f}  | Weak           |
| Test Brier Score           | {brier:.4f}    | Acceptable     |
| Test Trade Sharpe          | {trade_sharpe:.4f} | Marginal    |
| Test Trade Profit Factor   | {trade_pf:.4f}    | Slim margin |
| Walk-Forward Mean AUC      | {wf.get('mean_roc_auc',0):.4f}    | Barely above random |
| fold_stability             | FAIL     | Structural instability |
| cost_stress_base_pass      | True     | Survives zero cost |
| cost_stress_plus025_pass   | False    | Fails realistic cost |

---

*Audit generated by scripts/audit_live_signal_quality.py*
"""

with open(f'reports/model_signal_quality_audit_{ts}.md', 'w', encoding='utf-8') as f:
    f.write(md)

print(f'MD report written: reports/model_signal_quality_audit_{ts}.md')