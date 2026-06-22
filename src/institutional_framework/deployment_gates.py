"""
Deployment Gates - Automated Pre-Deployment Checker

Gate requirements before live trading:
1. Data Quality: min_samples, no gaps
2. Model Quality: ROC-AUC, F1, accuracy, train-test gap
3. Validation Quality: OOS holdout, Monte Carlo p-value, Deflated Sharpe
4. Shadow Trading: min predictions, ROC-AUC, accuracy
5. Paper Trading: min trades, Sharpe, PF, maxDD
6. Infrastructure: watchdog, DB, reconciliation
7. Risk: daily/weekly loss limits, max open positions
"""

import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Callable

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of a single gate check."""
    gate_name: str
    passed: bool
    value: float
    threshold: float
    message: str
    severity: str = "ERROR"  # ERROR, WARNING, INFO

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeploymentCheckResult:
    """Complete deployment gate check result."""
    passed: bool
    gates: Dict[str, GateResult]
    blocked_by: List[str]
    summary: str
    timestamp: str
    force_overridden: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "gates": {k: v.to_dict() for k, v in self.gates.items()},
            "blocked_by": self.blocked_by,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "force_overridden": self.force_overridden,
        }


class DeploymentGates:
    """
    Automated deployment gate checker.

    Usage:
        gates = DeploymentGates()
        result = gates.check_all()
        if result.passed:
            print("ALL GATES PASSED - Ready for deployment")
        else:
            print(f"BLOCKED BY: {result.blocked_by}")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

        # Default thresholds
        self.min_data_samples = self.config.get("min_data_samples", 1500)
        self.max_gap_seconds = self.config.get("max_gap_seconds", 300)  # 5 min

        self.model_min_auc = self.config.get("model_min_auc", 0.58)
        self.model_min_f1 = self.config.get("model_min_f1", 0.52)
        self.model_min_acc = self.config.get("model_min_acc", 0.53)
        self.model_max_train_test_gap = self.config.get("model_max_train_test_gap", 0.10)

        self.mc_p_value_threshold = self.config.get("mc_p_value_threshold", 0.05)
        self.min_deflated_sharpe = self.config.get("min_deflated_sharpe", 1.0)

        self.shadow_min_predictions = self.config.get("shadow_min_predictions", 100)
        self.shadow_min_auc = self.config.get("shadow_min_auc", 0.55)
        self.shadow_min_acc = self.config.get("shadow_min_acc", 0.53)

        self.paper_min_trades = self.config.get("paper_min_trades", 250)
        self.paper_min_sharpe = self.config.get("paper_min_sharpe", 0.80)
        self.paper_min_pf = self.config.get("paper_min_pf", 1.15)
        self.paper_max_dd = self.config.get("paper_max_dd", 0.15)

        self.max_daily_loss = self.config.get("max_daily_loss", 5000)  # ₹
        self.max_weekly_loss = self.config.get("max_weekly_loss", 15000)
        self.max_open_positions = self.config.get("max_open_positions", 5)

        # Gate provider functions (set by caller for dynamic checks)
        self._providers: Dict[str, Callable[[], Dict[str, Any]]] = {}

    def register_provider(self, gate_name: str, provider_fn: Callable[[], Dict[str, Any]]) -> None:
        """Register a dynamic provider for a gate check."""
        self._providers[gate_name] = provider_fn

    # -----------------------------------------------------------------------
    # Individual gate checks
    # -----------------------------------------------------------------------

    def check_data_quality(self) -> GateResult:
        """Gate 1: Data quality check."""
        try:
            if "data_quality" in self._providers:
                data = self._providers["data_quality"]()
                n_samples = data.get("n_samples", 0)
                max_gap = data.get("max_gap_seconds", 9999)
                total_gaps = data.get("total_gaps", 999)

                issues = []
                if n_samples < self.min_data_samples:
                    issues.append(f"samples={n_samples} < min={self.min_data_samples}")
                if max_gap > self.max_gap_seconds:
                    issues.append(f"max_gap={max_gap}s > threshold={self.max_gap_seconds}s")
                if total_gaps > 10:
                    issues.append(f"gaps={total_gaps} > 10")

                if issues:
                    return GateResult(
                        gate_name="data_quality", passed=False,
                        value=float(n_samples), threshold=float(self.min_data_samples),
                        message="; ".join(issues), severity="ERROR",
                    )
                return GateResult(
                    gate_name="data_quality", passed=True,
                    value=float(n_samples), threshold=float(self.min_data_samples),
                    message=f"n_samples={n_samples}, max_gap={max_gap}s",
                )

            # No provider - skip with warning
            return GateResult(
                gate_name="data_quality", passed=True,
                value=0.0, threshold=float(self.min_data_samples),
                message="No data quality provider registered - SKIPPED",
                severity="WARNING",
            )
        except Exception as e:
            logger.error("Data quality gate failed: %s", e)
            return GateResult(
                gate_name="data_quality", passed=False,
                value=0.0, threshold=float(self.min_data_samples),
                message=f"Exception: {e}", severity="ERROR",
            )

    def check_model_quality(self) -> GateResult:
        """Gate 2: Model quality check."""
        try:
            if "model_quality" in self._providers:
                data = self._providers["model_quality"]()
                auc = data.get("roc_auc", 0.0)
                f1 = data.get("f1", 0.0)
                acc = data.get("accuracy", 0.0)
                train_test_gap = data.get("train_test_gap", 1.0)

                issues = []
                if auc < self.model_min_auc:
                    issues.append(f"ROC-AUC={auc:.3f} < threshold={self.model_min_auc}")
                if f1 < self.model_min_f1:
                    issues.append(f"F1={f1:.3f} < threshold={self.model_min_f1}")
                if acc < self.model_min_acc:
                    issues.append(f"Acc={acc:.3f} < threshold={self.model_min_acc}")
                if train_test_gap > self.model_max_train_test_gap:
                    issues.append(f"train-test gap={train_test_gap:.3f} > {self.model_max_train_test_gap}")

                if issues:
                    return GateResult(
                        gate_name="model_quality", passed=False,
                        value=auc, threshold=float(self.model_min_auc),
                        message="; ".join(issues), severity="ERROR",
                    )
                return GateResult(
                    gate_name="model_quality", passed=True,
                    value=auc, threshold=float(self.model_min_auc),
                    message=f"AUC={auc:.3f}, F1={f1:.3f}, Acc={acc:.3f}, gap={train_test_gap:.3f}",
                )

            return GateResult(
                gate_name="model_quality", passed=True,
                value=0.0, threshold=float(self.model_min_auc),
                message="No model quality provider - SKIPPED",
                severity="WARNING",
            )
        except Exception as e:
            logger.error("Model quality gate failed: %s", e)
            return GateResult(
                gate_name="model_quality", passed=False,
                value=0.0, threshold=float(self.model_min_auc),
                message=f"Exception: {e}", severity="ERROR",
            )

    def check_validation_quality(self) -> GateResult:
        """Gate 3: Validation quality check."""
        try:
            if "validation_quality" in self._providers:
                data = self._providers["validation_quality"]()
                oos_passed = data.get("out_of_time_passed", False)
                mc_p_value = data.get("monte_carlo_p_value", 1.0)
                deflated_sr = data.get("deflated_sharpe", 0.0)

                issues = []
                if not oos_passed:
                    issues.append("Out-of-time validation FAILED")
                if mc_p_value > self.mc_p_value_threshold:
                    issues.append(f"MC p-value={mc_p_value:.4f} > {self.mc_p_value_threshold}")
                if deflated_sr < self.min_deflated_sharpe:
                    issues.append(f"Deflated Sharpe={deflated_sr:.3f} < {self.min_deflated_sharpe}")

                if issues:
                    return GateResult(
                        gate_name="validation_quality", passed=False,
                        value=mc_p_value, threshold=float(self.mc_p_value_threshold),
                        message="; ".join(issues), severity="ERROR",
                    )
                return GateResult(
                    gate_name="validation_quality", passed=True,
                    value=mc_p_value, threshold=float(self.mc_p_value_threshold),
                    message=f"OOS=pass, MC p={mc_p_value:.4f}, DSR={deflated_sr:.3f}",
                )

            return GateResult(
                gate_name="validation_quality", passed=True,
                value=0.0, threshold=float(self.mc_p_value_threshold),
                message="No validation provider - SKIPPED",
                severity="WARNING",
            )
        except Exception as e:
            logger.error("Validation gate failed: %s", e)
            return GateResult(
                gate_name="validation_quality", passed=False,
                value=0.0, threshold=float(self.mc_p_value_threshold),
                message=f"Exception: {e}", severity="ERROR",
            )

    def check_shadow_trading(self) -> GateResult:
        """Gate 4: Shadow trading results check."""
        try:
            if "shadow_trading" in self._providers:
                data = self._providers["shadow_trading"]()
                n_preds = data.get("n_predictions", 0)
                auc = data.get("roc_auc", 0.0)
                acc = data.get("accuracy", 0.0)

                issues = []
                if n_preds < self.shadow_min_predictions:
                    issues.append(f"predictions={n_preds} < min={self.shadow_min_predictions}")
                if auc < self.shadow_min_auc and n_preds >= self.shadow_min_predictions:
                    issues.append(f"ROC-AUC={auc:.3f} < {self.shadow_min_auc}")
                if acc < self.shadow_min_acc and n_preds >= self.shadow_min_predictions:
                    issues.append(f"Acc={acc:.3f} < {self.shadow_min_acc}")

                if issues:
                    return GateResult(
                        gate_name="shadow_trading", passed=False,
                        value=float(n_preds), threshold=float(self.shadow_min_predictions),
                        message="; ".join(issues), severity="ERROR",
                    )
                return GateResult(
                    gate_name="shadow_trading", passed=True,
                    value=float(n_preds), threshold=float(self.shadow_min_predictions),
                    message=f"{n_preds} predictions, AUC={auc:.3f}, Acc={acc:.3f}",
                )

            # Shadow trading is optional - warn but don't block
            return GateResult(
                gate_name="shadow_trading", passed=True,
                value=0.0, threshold=float(self.shadow_min_predictions),
                message="No shadow trading data - SKIPPED",
                severity="WARNING",
            )
        except Exception as e:
            logger.error("Shadow trading gate failed: %s", e)
            return GateResult(
                gate_name="shadow_trading", passed=False,
                value=0.0, threshold=float(self.shadow_min_predictions),
                message=f"Exception: {e}", severity="ERROR",
            )

    def check_paper_trading(self) -> GateResult:
        """Gate 5: Paper trading results check."""
        try:
            if "paper_trading" in self._providers:
                data = self._providers["paper_trading"]()
                n_trades = data.get("n_trades", 0)
                sharpe = data.get("sharpe", 0.0)
                pf = data.get("profit_factor", 1.0)
                max_dd = data.get("max_drawdown", 1.0)

                issues = []
                if n_trades < self.paper_min_trades:
                    issues.append(f"trades={n_trades} < min={self.paper_min_trades}")
                if n_trades >= self.paper_min_trades:
                    if sharpe < self.paper_min_sharpe:
                        issues.append(f"Sharpe={sharpe:.3f} < {self.paper_min_sharpe}")
                    if pf < self.paper_min_pf:
                        issues.append(f"PF={pf:.3f} < {self.paper_min_pf}")
                    if max_dd > self.paper_max_dd:
                        issues.append(f"maxDD={max_dd:.3f} > {self.paper_max_dd}")

                if issues:
                    return GateResult(
                        gate_name="paper_trading", passed=False,
                        value=float(n_trades), threshold=float(self.paper_min_trades),
                        message="; ".join(issues), severity="ERROR",
                    )
                return GateResult(
                    gate_name="paper_trading", passed=True,
                    value=float(n_trades), threshold=float(self.paper_min_trades),
                    message=f"{n_trades} trades, SR={sharpe:.3f}, PF={pf:.3f}, DD={max_dd:.3f}",
                )

            return GateResult(
                gate_name="paper_trading", passed=True,
                value=0.0, threshold=float(self.paper_min_trades),
                message="No paper trading data - SKIPPED",
                severity="WARNING",
            )
        except Exception as e:
            logger.error("Paper trading gate failed: %s", e)
            return GateResult(
                gate_name="paper_trading", passed=False,
                value=0.0, threshold=float(self.paper_min_trades),
                message=f"Exception: {e}", severity="ERROR",
            )

    def check_infrastructure(self) -> GateResult:
        """Gate 6: Infrastructure check."""
        try:
            if "infrastructure" in self._providers:
                data = self._providers["infrastructure"]()
                watchdog_running = data.get("watchdog_running", False)
                db_initialized = data.get("db_initialized", False)
                reconciliation_passed = data.get("reconciliation_passed", False)
                health_ok = data.get("health_ok", False)

                issues = []
                if not watchdog_running:
                    issues.append("Watchdog not running")
                if not db_initialized:
                    issues.append("Trade DB not initialized")
                if not reconciliation_passed:
                    issues.append("Position reconciliation FAILED")
                if not health_ok:
                    issues.append("Health endpoint not responding")

                if issues:
                    return GateResult(
                        gate_name="infrastructure", passed=False,
                        value=float(sum([watchdog_running, db_initialized, reconciliation_passed, health_ok])),
                        threshold=4.0,
                        message="; ".join(issues), severity="ERROR",
                    )
                return GateResult(
                    gate_name="infrastructure", passed=True,
                    value=4.0, threshold=4.0,
                    message="All infrastructure healthy",
                )

            return GateResult(
                gate_name="infrastructure", passed=True,
                value=0.0, threshold=4.0,
                message="No infrastructure provider - SKIPPED",
                severity="WARNING",
            )
        except Exception as e:
            logger.error("Infrastructure gate failed: %s", e)
            return GateResult(
                gate_name="infrastructure", passed=False,
                value=0.0, threshold=4.0,
                message=f"Exception: {e}", severity="ERROR",
            )

    def check_risk(self) -> GateResult:
        """Gate 7: Risk limits check."""
        try:
            if "risk" in self._providers:
                data = self._providers["risk"]()
                daily_loss = abs(data.get("daily_loss", 0))
                weekly_loss = abs(data.get("weekly_loss", 0))
                open_positions = data.get("open_positions", 0)

                issues = []
                if daily_loss > self.max_daily_loss:
                    issues.append(f"Daily loss ₹{daily_loss:,.0f} > limit ₹{self.max_daily_loss:,.0f}")
                if weekly_loss > self.max_weekly_loss:
                    issues.append(f"Weekly loss ₹{weekly_loss:,.0f} > limit ₹{self.max_weekly_loss:,.0f}")
                if open_positions > self.max_open_positions:
                    issues.append(f"Open positions {open_positions} > max {self.max_open_positions}")

                if issues:
                    return GateResult(
                        gate_name="risk", passed=False,
                        value=float(daily_loss), threshold=float(self.max_daily_loss),
                        message="; ".join(issues), severity="ERROR",
                    )
                return GateResult(
                    gate_name="risk", passed=True,
                    value=float(daily_loss), threshold=float(self.max_daily_loss),
                    message=f"Daily loss ₹{daily_loss:,.0f}, open={open_positions}",
                )

            return GateResult(
                gate_name="risk", passed=True,
                value=0.0, threshold=float(self.max_daily_loss),
                message="No risk provider - SKIPPED",
                severity="WARNING",
            )
        except Exception as e:
            logger.error("Risk gate failed: %s", e)
            return GateResult(
                gate_name="risk", passed=False,
                value=0.0, threshold=float(self.max_daily_loss),
                message=f"Exception: {e}", severity="ERROR",
            )

    # -----------------------------------------------------------------------
    # Full check
    # -----------------------------------------------------------------------

    def check_all(self, force: bool = False) -> DeploymentCheckResult:
        """
        Run all deployment gates.

        Args:
            force: If True, overrides failures (logs warning).

        Returns:
            DeploymentCheckResult with pass/fail, per-gate results, blocked_by list.
        """
        gates: Dict[str, GateResult] = {
            "data_quality": self.check_data_quality(),
            "model_quality": self.check_model_quality(),
            "validation_quality": self.check_validation_quality(),
            "shadow_trading": self.check_shadow_trading(),
            "paper_trading": self.check_paper_trading(),
            "infrastructure": self.check_infrastructure(),
            "risk": self.check_risk(),
        }

        blocked_by = [
            name for name, g in gates.items() if not g.passed and g.severity == "ERROR"
        ]

        all_passed = len(blocked_by) == 0

        if all_passed:
            summary = "ALL GATES PASSED - System is cleared for deployment"
            logger.info("DEPLOYMENT GATES: %s", summary)
        elif force:
            summary = f"FORCE OVERRIDE: {len(blocked_by)} gate(s) blocked, but --force applied"
            logger.warning("DEPLOYMENT GATES: %s | Blocked: %s", summary, blocked_by)
        else:
            summary = f"BLOCKED by {len(blocked_by)} gate(s): {', '.join(blocked_by)}"
            logger.error("DEPLOYMENT GATES: %s", summary)

        result = DeploymentCheckResult(
            passed=all_passed or force,
            gates=gates,
            blocked_by=blocked_by if not all_passed else [],
            summary=summary,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            force_overridden=force,
        )

        # Save report
        self._save_report(result)

        return result

    def _save_report(self, result: DeploymentCheckResult) -> None:
        """Save gate check report to JSON file."""
        try:
            report_dir = Path("reports")
            report_dir.mkdir(exist_ok=True)
            report_path = report_dir / f"deployment_gates_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info("Deployment gate report saved to %s", report_path)
        except Exception as e:
            logger.warning("Failed to save deployment gate report: %s", e)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def check_deployment_readiness(
    providers: Optional[Dict[str, Callable[[], Dict[str, Any]]]] = None,
    force: bool = False,
) -> DeploymentCheckResult:
    """
    Quick one-shot deployment readiness check.

    Args:
        providers: Dict mapping gate names to provider functions.
        force: Override failures.

    Returns:
        DeploymentCheckResult.
    """
    gates = DeploymentGates()
    if providers:
        for name, fn in providers.items():
            gates.register_provider(name, fn)
    return gates.check_all(force=force)