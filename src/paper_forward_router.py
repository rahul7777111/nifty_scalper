#!/usr/bin/env python3
"""
paper_forward_router.py
=======================
Thin compatibility shim so that `python -m py_compile src/paper_forward_router.py`
succeeds and legacy imports continue to work.

All real implementation lives in candidate_router.py (the production router).
This module re-exports the public surface used by engine/strategy/UI.
"""
from __future__ import annotations

# Re-export everything the rest of the system expects from the real router
from .candidate_router import *  # noqa: F401,F403

# Ensure the key symbols used in engine/strategy are present even under direct import
try:
    from .candidate_router import (  # type: ignore
        route_candidate_decision,
        _predict_confidence_from_artifact,
        _load_active_candidate_profile,
        get_last_router_decision,
        _LAST_ROUTER_DECISION,
    )
except Exception:  # direct run / test path without package
    from candidate_router import (  # type: ignore
        route_candidate_decision,
        _predict_confidence_from_artifact,
        _load_active_candidate_profile,
        get_last_router_decision,
        _LAST_ROUTER_DECISION,
    )

__all__ = [  # explicit for pyright / importers
    "route_candidate_decision",
    "_predict_confidence_from_artifact",
    "_load_active_candidate_profile",
    "get_last_router_decision",
    "_LAST_ROUTER_DECISION",
]
