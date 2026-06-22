from .monte_carlo import run_monte_carlo_5000, run_monte_carlo_10000, compute_deflated_sharpe
# Optional/legacy submodules: import defensively so missing optional files do not break package import
try:
    from .reconciliation import reconcile_positions
except Exception:
    try:
        # Fall back to a top-level reconciliation module if present (legacy layout)
        import reconciliation as _top_recon  # type: ignore
        reconcile_positions = getattr(_top_recon, "reconcile_positions", None)
    except Exception:
        reconcile_positions = None

try:
    from .out_of_time_validator import OutOfTimeValidator
except Exception:
    OutOfTimeValidator = None

try:
    from .shadow_analytics import ShadowAnalytics
except Exception:
    ShadowAnalytics = None

try:
    from .paper_trading_engine import PaperTradingEngine
except Exception:
    PaperTradingEngine = None

try:
    from .deployment_gates import DeploymentGates, GateResult, DeploymentCheckResult
except Exception:
    DeploymentGates = GateResult = DeploymentCheckResult = None

__all__ = [
    "run_monte_carlo_5000",
    "run_monte_carlo_10000",
    "compute_deflated_sharpe",
    "reconcile_positions",
    "OutOfTimeValidator",
    "ShadowAnalytics",
    "PaperTradingEngine",
    "DeploymentGates",
    "GateResult",
    "DeploymentCheckResult",
]
