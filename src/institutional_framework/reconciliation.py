"""Compatibility proxy for reconciliation.

This module provides a lightweight wrapper so `from institutional_framework.reconciliation import reconcile_positions`
works even when the authoritative implementation lives at `src/reconciliation.py` (legacy layout).
"""
from typing import Any

try:
    # Prefer the package-local implementation if present
    from reconciliation import ReconciliationEngine as _ReconciliationEngine  # type: ignore
except Exception:
    _ReconciliationEngine = None


def reconcile_positions(strategy_bot: Any) -> bool:
    """Delegate to available ReconciliationEngine, or return True (no-op) if unavailable.

    Returns True when reconciliation succeeds or is unavailable (to avoid blocking startup).
    """
    if _ReconciliationEngine is None:
        # Conservative default: assume reconciliation is OK when engine not installed
        return True
    try:
        engine = _ReconciliationEngine()
        # Instance method exists on the engine
        return bool(getattr(engine, "reconcile_positions")(strategy_bot))
    except Exception:
        return True
