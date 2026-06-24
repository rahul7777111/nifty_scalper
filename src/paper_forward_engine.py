#!/usr/bin/env python3
"""
src/paper_forward_engine.py

Multi-candidate paper-forward engine for fair live observation of all paper_forward_only candidates.
All trades are simulated. No broker orders ever.

Safety:
- Always paper/sim only.
- Same snapshot delivered to every candidate.
- Per-candidate independent state.
- force_eval=True allowed only for paper_forward_only.
- shadow_ready must stay false.
"""

from __future__ import annotations
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import threading
import queue
from collections import Counter
import math

# Ensure importable when run as script or from src/
try:
    _THIS = Path(__file__).resolve()
    _ROOT = _THIS.parents[1] if _THIS.parent.name == "src" else _THIS.parent
    for _p in (str(_ROOT), str(_ROOT / "src")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
except Exception:
    pass

# Import the required router (production API)
try:
    from candidate_router import route_candidate_decision, get_last_router_decision
except Exception:
    route_candidate_decision = None  # type: ignore
    get_last_router_decision = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent


def _coerce_config_bool(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


try:
    from paper_forward_readiness import (
        compute_cost_quality,
        format_readiness_log,
        last_good_mark_allowed,
        normalize_ltp_source,
        summarize_readiness,
    )
except ImportError:
    from .paper_forward_readiness import (  # type: ignore
        compute_cost_quality,
        format_readiness_log,
        last_good_mark_allowed,
        normalize_ltp_source,
        summarize_readiness,
    )

try:
    from pf_logging import (
        log_feature_x,
        log_paper_fwd_candidate,
        log_paper_fwd_decision,
        log_paper_fwd_features,
        log_paper_fwd_predict,
        log_paper_fwd_summary,
        log_pf_data,
        log_pf_perf,
        pf_log,
        pf_log_perf_summary,
        pf_should_log_level,
        pf_verbose_decision,
        slim_feature_debug,
    )
except ImportError:
    def pf_should_log_level(_level: str) -> bool:  # type: ignore[misc]
        return True

    def pf_verbose_decision() -> bool:  # type: ignore[misc]
        return False

    def slim_feature_debug(debug):  # type: ignore[misc]
        return dict(debug or {})

    def pf_log(level, msg, **kwargs):  # type: ignore[misc]
        print(msg)

    def log_paper_fwd_decision(**kwargs):  # type: ignore[misc]
        pass

    def log_paper_fwd_features(**kwargs):  # type: ignore[misc]
        pass

    def log_paper_fwd_predict(**kwargs):  # type: ignore[misc]
        pass

    def log_feature_x(**kwargs):  # type: ignore[misc]
        pass

    def log_paper_fwd_summary(**kwargs):  # type: ignore[misc]
        pass

    def log_paper_fwd_candidate(**kwargs):  # type: ignore[misc]
        pass

    def log_pf_data(**kwargs):  # type: ignore[misc]
        pass

    def log_pf_perf(**kwargs):  # type: ignore[misc]
        pass

    def pf_log_perf_summary():  # type: ignore[misc]
        pass


def resolve_repo_path(path: str | Path | None, project_root: Path | str = REPO_ROOT) -> Path | None:
    """Resolve artifact/config paths relative to repo root (Windows-safe)."""
    if not path:
        return None
    p = Path(str(path).replace("\\", "/"))
    if p.is_absolute():
        return p
    root = Path(project_root).resolve()
    return (root / p).resolve()


def _candle_count_from_snapshot(market_snapshot: dict) -> int:
    candles = market_snapshot.get("candles") or []
    if isinstance(candles, (list, tuple)):
        return int(market_snapshot.get("candle_count") or len(candles) or 0)
    return int(market_snapshot.get("candle_count") or 0)


def _min_candles_for_prediction() -> int:
    try:
        from synthetic_option_chain import load_bs_config
        return int(load_bs_config().get("min_candles_for_prediction", 20))
    except Exception:
        return 20


_STALE_WAITING_QUALITIES = frozenset({
    "WAITING_FOR_CANDLES",
    "SYNTHETIC_CHAIN_READY_WAITING_FOR_CANDLES",
    "CANDLES_MISSING",
    "CANDLES_MISSING_NONFATAL",
    "",
})

_PAPER_READY_QUALITIES = frozenset({
    "DATA_OK",
    "REAL_CANDLES_REAL_CHAIN",
    "REAL_CANDLES_REAL_CHAIN_BROKER_LTP",
    "REAL_CANDLES_SYNTHETIC_CHAIN",
    "SYNTHETIC_CHAIN_WITH_BROKER_MARK",
    "SYNTHETIC_CHAIN_OK",
    "SYNTHETIC_CHAIN_WITH_SPOT_CANDLE_FALLBACK",
    "SYNTHETIC_CHAIN_WITH_SYNTHETIC_CANDLES",
})


def reconcile_paper_forward_readiness(
    *,
    data_quality: str,
    candle_rows: int,
    option_rows: int,
    spot: Any,
    min_candles: int | None = None,
    missing_features: list | None = None,
    artifact_ok: bool = True,
    candidate_id: str = "",
    old_quality: str = "",
) -> tuple[str, bool]:
    """Upgrade stale WAITING_FOR_CANDLES when counts prove readiness."""
    min_req = int(min_candles if min_candles is not None else _min_candles_for_prediction())
    has_spot = spot not in (None, "", 0, 0.0)
    has_candles = int(candle_rows or 0) >= min_req
    has_chain = int(option_rows or 0) > 0
    missing_n = len(missing_features or [])
    predict_allowed = False
    quality = str(data_quality or "").strip() or "UNKNOWN"
    if has_spot and has_candles and has_chain and artifact_ok and missing_n == 0:
        if quality in _STALE_WAITING_QUALITIES or quality.startswith("WAITING_FOR_CANDLES"):
            if old_quality and old_quality not in ("", quality) and old_quality == "DATA_OK":
                print(
                    f"[PF-READINESS-DOWNGRADE-BLOCKED] source=snapshot_reconcile "
                    f"candidate_id={candidate_id} old=DATA_OK new={quality} reason=partial_snapshot"
                )
                quality = "DATA_OK"
            else:
                print(
                    f"[PF-READINESS] candidate_id={candidate_id} candles_count={candle_rows} "
                    f"min_required={min_req} option_rows={option_rows} missing_features={missing_n} "
                    f"artifact_ok={artifact_ok} data_quality=DATA_OK predict_allowed=true"
                )
                quality = "DATA_OK"
        if quality in _PAPER_READY_QUALITIES:
            predict_allowed = True
    elif quality in _PAPER_READY_QUALITIES and has_candles and has_chain and has_spot and artifact_ok and missing_n == 0:
        predict_allowed = True
    elif quality == "INVALID_CHAIN_SPOT":
        predict_allowed = False
    return quality, predict_allowed


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _snapshot_allows_synthetic_candles(market_snapshot: dict) -> bool:
    if _truthy(market_snapshot.get("require_real_candles")):
        return False
    flag = market_snapshot.get("allow_synthetic_candle_fallback")
    if flag is not None:
        return _truthy(flag)
    return _truthy(os.getenv("PF_ALLOW_SYNTHETIC_CANDLE_FALLBACK", "true"))


def _apply_spot_candle_fallback_if_needed(market_snapshot: dict) -> None:
    """Inject deterministic synthetic spot candles when BS chain is ready but live candles are insufficient."""
    if not market_snapshot:
        return
    try:
        from synthetic_option_chain import build_spot_candle_fallback, infer_paper_forward_data_quality, load_bs_config
    except Exception:
        return
    cfg = load_bs_config()
    min_candles = int(cfg.get("min_candles_for_prediction", 20))
    candle_count = _candle_count_from_snapshot(market_snapshot)
    if candle_count >= min_candles:
        return
    spot = (
        market_snapshot.get("spot")
        or market_snapshot.get("price")
        or market_snapshot.get("close")
        or market_snapshot.get("last")
    )
    try:
        spot_f = float(spot) if spot not in (None, "", 0, 0.0) else 0.0
    except Exception:
        spot_f = 0.0
    if spot_f <= 0:
        return
    chain_src = str(
        market_snapshot.get("chain_source")
        or market_snapshot.get("option_chain_source")
        or ""
    )
    broker = str(market_snapshot.get("broker_name") or "").lower()
    is_synth = (
        chain_src == "BLACK_SCHOLES_SYNTHETIC"
        or bool(market_snapshot.get("synthetic"))
        or broker in ("mstock", "m.stock", "m_stock", "mstocks")
    )
    if not is_synth:
        return
    option_rows = int(market_snapshot.get("option_rows") or market_snapshot.get("option_chain_rows") or 0)
    if option_rows <= 0:
        oc = market_snapshot.get("option_chain") or market_snapshot.get("_option_chain_snapshot")
        if isinstance(oc, (list, tuple)):
            option_rows = len(oc)
    allow_fb = bool(cfg.get("allow_candle_fallback", True)) and _snapshot_allows_synthetic_candles(market_snapshot)
    if not allow_fb:
        quality = infer_paper_forward_data_quality(
            option_rows=option_rows,
            candle_count=candle_count,
            spot=spot_f,
            chain_source=chain_src,
            synthetic=True,
            candle_source=str(market_snapshot.get("candle_source") or ""),
            synthetic_candles=bool(market_snapshot.get("synthetic_candles")),
        )
        if quality == "SYNTHETIC_CHAIN_WITH_SPOT_CANDLE_FALLBACK":
            quality = "WAITING_FOR_CANDLES"
        market_snapshot["data_quality_status"] = quality
        market_snapshot["synthetic_candles"] = False
        if candle_count <= 0:
            market_snapshot["candle_source"] = market_snapshot.get("candle_source") or "none"
            market_snapshot["candle_error"] = market_snapshot.get("candle_error") or "waiting_for_real_candles"
        return
    fb = build_spot_candle_fallback(spot_f)
    if not fb:
        return
    market_snapshot["candles"] = fb
    market_snapshot["candle_count"] = len(fb)
    market_snapshot["candle_source"] = "SYNTHETIC_SPOT_FALLBACK"
    market_snapshot["synthetic_candles"] = True
    market_snapshot["data_quality_status"] = infer_paper_forward_data_quality(
        option_rows=option_rows,
        candle_count=len(fb),
        spot=spot_f,
        chain_source=chain_src or "BLACK_SCHOLES_SYNTHETIC",
        synthetic=True,
        candle_source="SYNTHETIC_SPOT_FALLBACK",
        synthetic_candles=True,
    )


# ---------------------------------------------------------------------------
# TASK 2+4: Robust artifact resolution + registry fallback + precise reasons
# ---------------------------------------------------------------------------

LOAD_OK = "candidate_loaded_ok"
_PAPER_FORWARD_RUNTIME_DISABLE_OVERRIDES = {
    "trade_count_above_watchlist_gate",
    "worst_fold_pf_below_watchlist_gate",
    "median_fold_pf_below_watchlist_gate",
}
REASONS = {
    "missing_artifact_dir": "ARTIFACT_NOT_FOUND",
    "missing_profile_and_manifest": "ARTIFACT_NOT_FOUND",
    "missing_model_file": "candidate_model_file_missing",
    "missing_feature_list": "candidate_feature_file_missing",
    "missing_preset": "candidate_preset_missing",
    "invalid_json": "candidate_load_exception",
    "pickle_load_failed": "candidate_load_exception",
    "unsupported": "unsupported_model_type",
}


class PaperForwardAuthState:
    """TASK 1: Single source of truth for paper-forward broker auth verification state.
    All footer, poller, engine gating, status label, diagnose must derive from this.
    """
    VALID_STATUSES = (
        "TOKEN_MISSING",
        "CREDENTIALS_MISSING",
        "TOKEN_PRESENT_UNCHECKED",
        "VERIFYING_BROKER_TOKEN",
        "AUTH_OK",
        "AUTH_FAILED",
        "SESSION_EXPIRED",
        "BROKER_CLIENT_INIT_FAILED",
    )

    def __init__(self) -> None:
        self.token_hash: str = ""
        self.status: str = "TOKEN_MISSING"
        self.validation_attempted: bool = False
        self.validation_in_progress: bool = False
        self.validation_started_ts: Any = None
        self.validation_finished_ts: Any = None
        self.endpoint_used: str = ""
        self.error: str = ""
        self.success_count: int = 0
        self.failure_count: int = 0
        self.client_created: bool = False
        self.client_source: str = ""

    def reset_for_token(self, token_hash: str) -> None:
        """Only called on token change."""
        if token_hash and token_hash != self.token_hash:
            old = self.token_hash
            self.token_hash = token_hash
            self.status = "TOKEN_PRESENT_UNCHECKED"
            self.validation_attempted = False
            self.validation_in_progress = False
            self.validation_started_ts = None
            self.validation_finished_ts = None
            self.endpoint_used = ""
            self.error = ""
            # counts keep accumulating across token lives for the session
            self.client_created = False
            self.client_source = ""
            # trace will be emitted by caller

    def set_verifying(self, started_ts: str) -> None:
        self.status = "VERIFYING_BROKER_TOKEN"
        self.validation_in_progress = True
        self.validation_started_ts = started_ts
        self.validation_finished_ts = None
        self.error = ""

    def set_terminal(self, status: str, error: str = "", endpoint: str = "", finished_ts: str = None) -> None:
        if status not in self.VALID_STATUSES:
            status = "AUTH_FAILED"
        self.status = status
        self.error = error or ""
        self.endpoint_used = endpoint or ""
        self.validation_in_progress = False
        self.validation_attempted = True
        self.validation_finished_ts = finished_ts or datetime.now(timezone.utc).isoformat()
        if status == "AUTH_OK":
            self.success_count += 1
        else:
            self.failure_count += 1

    def is_terminal(self) -> bool:
        return self.status in ("AUTH_OK", "AUTH_FAILED", "SESSION_EXPIRED", "BROKER_CLIENT_INIT_FAILED", "TOKEN_MISSING", "CREDENTIALS_MISSING")

    def is_pending(self) -> bool:
        return self.status in ("TOKEN_PRESENT_UNCHECKED", "VERIFYING_BROKER_TOKEN")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "broker_auth": self.status,
            "auth_status": self.status,
            "auth_error": self.error,
            "auth_validation_attempted": self.validation_attempted,
            "auth_validation_in_progress": self.validation_in_progress,
            "auth_validation_started_ts": self.validation_started_ts,
            "auth_validation_finished_ts": self.validation_finished_ts,
            "auth_validation_endpoint": self.endpoint_used,
            "auth_success_count": self.success_count,
            "auth_failure_count": self.failure_count,
            "token_hash_prefix": self.token_hash,
            "token_present": bool(self.token_hash),
            "client_created": self.client_created,
            "client_source": self.client_source,
        }

    def __repr__(self) -> str:
        return f"PaperForwardAuthState(status={self.status}, attempted={self.validation_attempted}, hash={self.token_hash})"


class PaperForwardDataStatus:
    """Central runtime data/auth status for the paper-forward footer and diagnostics."""

    def __init__(
        self,
        broker_name: str = "unknown",
        broker_auth: str = "UNKNOWN",
        auth_error: str = "",
        candle_source: str = "none",
        candle_count: int = 0,
        spot: Any = None,
        option_rows: int = 0,
        option_chain_status: str = "",
        option_chain_expiry: Any = None,
        option_chain_error: str = "",
        option_chain_source: str = "",
        option_chain_age_sec: Any = None,
        latest_tick: Any = None,
        last_snapshot_ts: Any = None,
        data_quality_status: str = "",
        min_option_chain_rows: int = 20,
        auth_last_checked_ts: Any = None,
        auth_validation_attempted: bool = False,
        auth_validation_in_progress: bool = False,
        auth_validation_started_ts: Any = None,
        auth_validation_elapsed_sec: Any = None,
        auth_validation_endpoint: str = "",
        auth_success_count: int = 0,
        auth_failure_count: int = 0,
        token_present: bool = False,
        token_hash_prefix: str = "",
        client_created: bool = False,
        client_source: str = "",
        chain_fetch_attempted: bool = False,
        chain_skipped_reason: str = "",
        candle_fetch_attempted: bool = False,
        candle_skipped_reason: str = "",
        candle_error: str = "",
        last_successful_snapshot_ts: Any = None,
        last_data_quality: str = "",
    ) -> None:
        self.broker_name = broker_name or "unknown"
        self.broker_auth = broker_auth or "UNKNOWN"
        self.auth_error = auth_error or ""
        self.candle_source = candle_source or "none"
        self.candle_count = int(candle_count or 0)
        self.spot = spot
        self.option_rows = int(option_rows or 0)
        self.option_chain_status = option_chain_status or ""
        self.option_chain_expiry = option_chain_expiry
        self.option_chain_error = option_chain_error or ""
        self.option_chain_source = option_chain_source or ""
        self.option_chain_age_sec = option_chain_age_sec
        self.latest_tick = latest_tick
        self.last_snapshot_ts = last_snapshot_ts or latest_tick
        self.min_option_chain_rows = int(min_option_chain_rows or 20)
        self.auth_last_checked_ts = auth_last_checked_ts
        self.auth_validation_attempted = bool(auth_validation_attempted)
        self.auth_validation_in_progress = bool(auth_validation_in_progress)
        self.auth_validation_started_ts = auth_validation_started_ts
        self.auth_validation_elapsed_sec = auth_validation_elapsed_sec
        self.auth_validation_endpoint = auth_validation_endpoint or ""
        self.auth_success_count = int(auth_success_count or 0)
        self.auth_failure_count = int(auth_failure_count or 0)
        self.token_present = bool(token_present)
        self.token_hash_prefix = token_hash_prefix or ""
        self.client_created = bool(client_created)
        self.client_source = client_source or ""
        self.chain_fetch_attempted = bool(chain_fetch_attempted)
        self.chain_skipped_reason = chain_skipped_reason or ""
        self.candle_fetch_attempted = bool(candle_fetch_attempted)
        self.candle_skipped_reason = candle_skipped_reason or ""
        self.candle_error = candle_error or ""
        self.last_successful_snapshot_ts = last_successful_snapshot_ts
        self.last_data_quality = last_data_quality or ""
        self.data_quality_status = data_quality_status or self._infer_quality()
        self.reconcile_quality()

    def reconcile_quality(self) -> str:
        """Reconcile stale data_quality_status against live candle/chain counts."""
        old = self.data_quality_status
        quality, _ = reconcile_paper_forward_readiness(
            data_quality=self.data_quality_status,
            candle_rows=self.candle_count,
            option_rows=self.option_rows,
            spot=self.spot,
            artifact_ok=True,
            old_quality=str(self.last_data_quality or ""),
        )
        self.data_quality_status = quality
        if quality == "DATA_OK" and old in _STALE_WAITING_QUALITIES:
            self.last_data_quality = quality
        return self.data_quality_status

    def _infer_quality(self) -> str:
        cfg_missing = str(self.option_chain_error or "").lower()
        if self.broker_auth in ("UNKNOWN", "TOKEN_SET_NOT_VERIFIED", "TOKEN_PRESENT_UNCHECKED", "VERIFYING_BROKER_TOKEN"):
            return "TOKEN_NOT_VERIFIED"
        if str(self.broker_auth).startswith("SESSION_EXPIRED"):
            return "SESSION_EXPIRED"
        if str(self.broker_auth).startswith("AUTH_FAILED") or self.broker_auth == "BROKER_CLIENT_INIT_FAILED" or self.auth_error:
            return "AUTH_FAILED"
        if "broker_config_missing" in cfg_missing or str(self.option_chain_status).lower() == "broker_config_missing":
            return "BROKER_CONFIG_MISSING"
        if any(x in str(self.option_chain_status or "").upper() for x in ("WAITING_FOR_MSTOCK_EXPIRY", "WAITING_FOR_MSTOCK_EXCHANGE", "WAITING_FOR_MSTOCK_CONFIG")):
            return str(self.option_chain_status).upper()
        if "option_chain_fetch_error" in cfg_missing or str(self.option_chain_status).upper() == "OPTION_CHAIN_FETCH_FAILED":
            return "OPTION_CHAIN_FETCH_ERROR"
        if self.spot in (None, "", 0, 0.0):
            return "SPOT_MISSING"
        if self.option_rows <= 0:
            broker = str(self.broker_name or "").lower()
            if broker in ("mstock", "m.stock", "m_stock", "mstocks"):
                try:
                    from synthetic_option_chain import is_synthetic_chain_enabled
                    if is_synthetic_chain_enabled(broker):
                        return "SYNTHETIC_CHAIN_PENDING"
                except Exception:
                    pass
            return "OPTION_CHAIN_EMPTY"
        chain_src = str(self.option_chain_source or "").upper()
        is_synth = chain_src == "BLACK_SCHOLES_SYNTHETIC"
        if is_synth:
            try:
                from synthetic_option_chain import infer_paper_forward_data_quality
                return infer_paper_forward_data_quality(
                    option_rows=self.option_rows,
                    candle_count=self.candle_count,
                    spot=self.spot,
                    chain_source=self.option_chain_source,
                    synthetic=True,
                )
            except Exception:
                if self.candle_count <= 0:
                    return "WAITING_FOR_CANDLES"
                return "SYNTHETIC_CHAIN_OK"
        if self.option_rows < self.min_option_chain_rows:
            return "THIN_OPTION_CHAIN"
        if self.candle_count <= 0:
            return "CANDLES_MISSING_NONFATAL"
        return "DATA_OK"

    @classmethod
    def from_snapshot(cls, market_snapshot: dict | None, option_chain_snapshot: Any = None) -> "PaperForwardDataStatus":
        snap = dict(market_snapshot or {})
        chain_obj = option_chain_snapshot if option_chain_snapshot is not None else snap.get("option_chain")
        rows = _count_option_rows(chain_obj)
        candles = snap.get("candles") or []
        candle_count = int(snap.get("candle_count") or (len(candles) if isinstance(candles, (list, tuple)) else 0) or 0)
        source = snap.get("candle_source") or snap.get("source") or "none"
        if candle_count and str(source).lower() in ("", "none", "empty"):
            source = "historical"
        spot = snap.get("spot") or snap.get("price") or snap.get("close") or snap.get("last")
        if spot in (None, "", 0, 0.0):
            spot = _derive_spot_from_chain(chain_obj)
        broker_auth = snap.get("broker_auth") or snap.get("auth_status") or "UNKNOWN"
        attempted = bool(snap.get("auth_validation_attempted"))
        # TASK2: only init to UNCHECKED from env if no state/attempt yet; never overwrite terminal or attempted status
        if str(broker_auth).upper() == "UNKNOWN" and not attempted and (snap.get("token_set") or os.getenv("MSTOCK_ACCESS_TOKEN") or os.getenv("DHAN_ACCESS_TOKEN")):
            broker_auth = "TOKEN_PRESENT_UNCHECKED"
        min_rows = int(snap.get("min_option_chain_rows") or os.getenv("PAPER_FORWARD_MIN_OPTION_CHAIN_ROWS", "20") or 20)
        option_status = snap.get("option_chain_status") or _option_chain_quality_status(rows, min_rows)
        option_chain_age = snap.get("option_chain_age_sec")
        if option_chain_age is None:
            ts = snap.get("option_chain_ts") or snap.get("last_option_chain_ts")
            try:
                if ts:
                    if not hasattr(ts, "tzinfo"):
                        ts = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    now = datetime.now(ts.tzinfo or timezone.utc)
                    option_chain_age = max(0.0, (now - ts).total_seconds())
            except Exception:
                option_chain_age = None
        status = cls(
            broker_name=str(snap.get("broker_name") or "unknown"),
            broker_auth=str(broker_auth),
            auth_error=str(snap.get("auth_error") or ""),
            candle_source=str(source),
            candle_count=candle_count,
            spot=spot,
            option_rows=rows,
            option_chain_status=option_status,
            option_chain_expiry=snap.get("option_chain_expiry"),
            option_chain_error=str(snap.get("option_chain_error") or ""),
            option_chain_source=str(snap.get("option_chain_source") or ""),
            option_chain_age_sec=option_chain_age,
            latest_tick=snap.get("timestamp") or snap.get("last_tick"),
            last_snapshot_ts=snap.get("last_snapshot_ts") or snap.get("timestamp"),
            data_quality_status=str(snap.get("data_quality_status") or ""),
            min_option_chain_rows=min_rows,
            auth_last_checked_ts=snap.get("auth_last_checked_ts"),
            auth_validation_attempted=bool(snap.get("auth_validation_attempted")),
            auth_validation_in_progress=bool(snap.get("auth_validation_in_progress")),
            auth_validation_started_ts=snap.get("auth_validation_started_ts"),
            auth_validation_elapsed_sec=snap.get("auth_validation_elapsed_sec"),
            auth_validation_endpoint=str(snap.get("auth_validation_endpoint") or ""),
            auth_success_count=int(snap.get("auth_success_count") or 0),
            auth_failure_count=int(snap.get("auth_failure_count") or 0),
            token_present=bool(snap.get("token_present") or snap.get("token_set")),
            token_hash_prefix=str(snap.get("token_hash_prefix") or ""),
            client_created=bool(snap.get("client_created")),
            client_source=str(snap.get("client_source") or ""),
            chain_fetch_attempted=bool(snap.get("chain_fetch_attempted")),
            chain_skipped_reason=str(snap.get("chain_skipped_reason") or ""),
            candle_fetch_attempted=bool(snap.get("candle_fetch_attempted")),
            candle_skipped_reason=str(snap.get("candle_skipped_reason") or ""),
            candle_error=str(snap.get("candle_error") or ""),
            last_successful_snapshot_ts=snap.get("last_successful_snapshot_ts"),
            last_data_quality=str(snap.get("last_data_quality") or ""),
        )
        status.reconcile_quality()
        return status

    def as_dict(self) -> Dict[str, Any]:
        spot_status = "spot_ok" if self.spot not in (None, "", 0, 0.0) else "spot_missing"
        opt_status = self.option_chain_status or _option_chain_quality_status(self.option_rows, self.min_option_chain_rows)
        data_quality = self.data_quality_status or self._infer_quality()
        return {
            "broker_name": self.broker_name,
            "broker_auth": self.broker_auth,
            "auth_status": self.broker_auth,
            "auth_error": self.auth_error,
            "auth_last_checked_ts": self.auth_last_checked_ts,
            "auth_validation_attempted": self.auth_validation_attempted,
            "auth_validation_in_progress": self.auth_validation_in_progress,
            "auth_validation_started_ts": self.auth_validation_started_ts,
            "auth_validation_elapsed_sec": self.auth_validation_elapsed_sec,
            "auth_validation_endpoint": self.auth_validation_endpoint,
            "auth_success_count": self.auth_success_count,
            "auth_failure_count": self.auth_failure_count,
            "token_present": self.token_present,
            "token_hash_prefix": self.token_hash_prefix,
            "client_created": self.client_created,
            "client_source": self.client_source,
            "chain_fetch_attempted": self.chain_fetch_attempted,
            "chain_skipped_reason": self.chain_skipped_reason,
            "candle_fetch_attempted": self.candle_fetch_attempted,
            "candle_skipped_reason": self.candle_skipped_reason,
            "candle_source": self.candle_source,
            "candle_count": self.candle_count,
            "candle_error": self.candle_error,
            "spot": self.spot,
            "spot_status": spot_status,
            "option_rows": self.option_rows,
            "option_chain_rows": self.option_rows,
            "option_chain_status": opt_status,
            "option_chain_expiry": self.option_chain_expiry,
            "option_chain_error": self.option_chain_error,
            "option_chain_source": self.option_chain_source,
            "option_chain_age_sec": self.option_chain_age_sec,
            "latest_tick": self.latest_tick,
            "last_tick_ts": self.latest_tick,
            "last_snapshot_ts": self.last_snapshot_ts,
            "last_successful_snapshot_ts": self.last_successful_snapshot_ts,
            "last_data_quality": self.last_data_quality or data_quality,
            "data_quality_status": data_quality,
        }

    def footer_text(self) -> str:
        d = self.as_dict()
        err = f" error={d['auth_error']}" if d.get("auth_error") else ""
        candle_err = f" reason={d['candle_error']}" if d.get("candle_error") else ""
        if d.get("option_chain_rows", 0) > 0:
            age = d.get("option_chain_age_sec")
            try:
                age_txt = f" age={float(age):.0f}s" if age is not None else ""
            except Exception:
                age_txt = ""
            source_txt = f" source={d.get('option_chain_source')}" if d.get("option_chain_source") else ""
            chain_detail = f"{source_txt}{age_txt}"
        else:
            chain_detail = f" reason={d['option_chain_error']}" if d.get("option_chain_error") else ""
        return (
            f"Broker auth: {d['broker_auth']}{err} | Candle source: {d['candle_source']} "
            f"candles={d['candle_count']}{candle_err} | {d['spot_status']} | "
            f"Option Chain: {('OK' if d.get('option_chain_rows', 0) > 0 else 'EMPTY')} rows={d['option_chain_rows']}{chain_detail} | "
            f"Data: {('PAPER_READY' if d['data_quality_status'] == 'DATA_OK' else d['data_quality_status'])}"
            f"{(' chain_source=' + str(d.get('option_chain_source'))) if d.get('option_chain_source') == 'BLACK_SCHOLES_SYNTHETIC' else ''}"
        )


class PaperForwardRuntime:
    """TASK 1: Single canonical runtime state for Paper Forward Monitor.
    One source for auth, data, last good snapshot, counters.
    All publishers (poller, build, live_chart, footer, diagnose) must use this.
    """
    def __init__(self):
        self.auth = PaperForwardAuthState()
        self.broker_name = "mstock"
        self.spot = None
        self.option_chain = []  # list of dict rows
        self.option_chain_rows = 0
        self.option_chain_expiry = None
        self.candles = []  # list
        self.candle_count = 0
        self.candle_source = "none"
        self.data_quality = "TOKEN_NOT_VERIFIED"
        self.last_good_snapshot = None
        self.last_good_snapshot_ts = None
        self.last_good_snapshot_quality = "TOKEN_NOT_VERIFIED"
        self.last_good_snapshot_source = ""
        self.last_good_snapshot_rows = 0
        self.last_good_snapshot_candles = 0
        self.last_accepted_snapshot_id = None
        self.last_rejected_snapshot_id = None
        self.bad_snapshots_rejected = 0
        self.snapshots_received = 0
        self.last_reason_counts = {}
        self.client = None
        self.client_source = ""
        self.min_option_chain_rows = 20
        self.last_update_ts = None
        self.candidate_states: Dict[str, "PaperForwardCandidateRuntimeState"] = {}
        # For get_* decision exposure
        self._last_decision: Optional[Dict[str, Any]] = None
        self._decision_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Snapshot management (BLOCKER 1 fix + required API)
    # ------------------------------------------------------------------
    def update_from_good_snapshot(self, snap: Any, source: str, chain: Any = None):
        """Store latest good snapshot, update derived fields, emit accept log.
        Compatibility alias + robust to bad types (BLOCKER 1).
        """
        if not isinstance(snap, dict):
            print(f"[PF-RUNTIME-SNAPSHOT] status=REJECTED reason=invalid_snapshot_type type={type(snap)}")
            return
        self.snapshots_received += 1
        self.last_update_ts = snap.get("timestamp")
        if chain is not None:
            self.option_chain = list(chain) if isinstance(chain, (list, tuple)) else []
            self.option_chain_rows = len(self.option_chain)
        else:
            rows = _count_option_rows(snap.get("option_chain"))
            if rows > 0:
                self.option_chain = snap.get("option_chain") or self.option_chain
                self.option_chain_rows = rows
        if snap.get("candles"):
            self.candles = list(snap.get("candles") or [])
            self.candle_count = len(self.candles)
            self.candle_source = snap.get("candle_source") or snap.get("source") or self.candle_source
        spot = snap.get("spot") or snap.get("price")
        if spot not in (None, 0, 0.0):
            self.spot = float(spot)
        auth = snap.get("broker_auth") or snap.get("auth_status")
        if auth:
            self.auth.status = auth  # sync
        self.data_quality = snap.get("data_quality_status") or "DATA_OK"
        self.last_good_snapshot = dict(snap)
        self.last_good_snapshot_ts = self.last_update_ts
        self.last_good_snapshot_quality = self.data_quality
        self.last_good_snapshot_source = source
        self.last_good_snapshot_rows = self.option_chain_rows
        self.last_good_snapshot_candles = self.candle_count
        self.last_accepted_snapshot_id = snap.get("snapshot_id") or f"acc_{int(time.time())}"
        print(
            f"[PF-RUNTIME-SNAPSHOT] source={source} status=OK "
            f"option_rows={self.option_chain_rows} candles={self.candle_count} "
            f"snapshot_id={self.last_accepted_snapshot_id}"
        )

    def is_good(self, snapshot: Any = None) -> bool:
        """Compatibility alias for UI checks. Robust to non-dict arg (BLOCKER 1)."""
        if snapshot is not None and not isinstance(snapshot, dict):
            print(f"[PF-RUNTIME-SNAPSHOT] status=REJECTED reason=invalid_snapshot_type type={type(snapshot)}")
            return False
        if snapshot:
            # quick check on provided instead of self state
            rows = _count_option_rows(snapshot.get("option_chain") or snapshot.get("option_rows"))
            auth = str(snapshot.get("broker_auth") or snapshot.get("auth_status") or "")
            return auth == "AUTH_OK" and rows >= self.min_option_chain_rows
        return (
            self.auth.status == "AUTH_OK"
            and self.option_chain_rows >= self.min_option_chain_rows
            and (self.candle_count > 0 or True)
        )

    def get_canonical_snapshot(self) -> dict:
        snap = {
            "timestamp": self.last_update_ts or datetime.now(timezone.utc).isoformat(),
            "spot": self.spot,
            "price": self.spot,
            "broker_auth": self.auth.status,
            "auth_status": self.auth.status,
            "auth_error": self.auth.error,
            "auth_validation_attempted": self.auth.validation_attempted,
            "option_chain": list(self.option_chain),
            "option_rows": self.option_chain_rows,
            "candles": list(self.candles),
            "candle_count": self.candle_count,
            "candle_source": self.candle_source,
            "source": self.last_good_snapshot_source or "canonical_runtime",
            "data_quality_status": self.data_quality,
            "snapshot_id": self.last_accepted_snapshot_id,
        }
        return snap

    # New safe accessors per spec
    def get_latest_snapshot(self) -> dict:
        return self.last_good_snapshot or self.get_canonical_snapshot()

    def get_latest_chain(self) -> list:
        return list(self.option_chain or [])

    def get_last_decision(self) -> dict:
        return self._last_decision or {}

    def get_decision_rows(self) -> list:
        return list(self._decision_rows or [])

    def is_good_snapshot(self, snapshot: Any = None) -> bool:
        if snapshot is not None and not isinstance(snapshot, dict):
            print(f"[PF-RUNTIME-SNAPSHOT] status=REJECTED reason=invalid_snapshot_type type={type(snapshot)}")
            return False
        if snapshot:
            rows = _count_option_rows(snapshot.get("option_chain"))
            auth = str(snapshot.get("broker_auth") or snapshot.get("auth_status") or "")
            return auth == "AUTH_OK" and rows >= self.min_option_chain_rows
        return self.is_good()

    def reject_snapshot(self, snap: dict, source: str, reason: str):
        """Record rejection + required log."""
        self.bad_snapshots_rejected += 1
        self.last_rejected_snapshot_id = snap.get("snapshot_id") or f"rej_{int(time.time())}"
        print(
            f"[PF-RUNTIME-SNAPSHOT] source={source} status=REJECTED "
            f"reason={reason} snapshot_id={self.last_rejected_snapshot_id}"
        )

    # Back-compat no-op / simple setters used indirectly
    def update_snapshot(self, *a, **k):
        # delegated to update_from_good in callers; keep for safety
        pass


class PaperForwardCandidateRuntimeState:
    """TASK 2: canonical per-candidate runtime state synced from decisions.
    Holds full attributes for table, PnL, paper pos, reasons etc.
    Updated by engine after every accepted decision.
    """
    def __init__(self, candidate_id: str):
        self.candidate_id = candidate_id
        self.enabled = True
        self.model = ""
        self.preset = ""
        self.side = ""
        self.final_signal = "NO_TRADE"
        self.confidence: Optional[float] = None
        self.threshold = 0.0
        self.predict_attempted = False
        self.reason_code = ""
        self.reason_detail = ""
        self.pos = "FLAT"
        self.qty = 0
        self.lot_size = 65
        self.lots = 1
        self.direction = "BUY"
        self.entry_price = 0.0
        self.current_price = 0.0
        self.unreal_pnl = 0.0
        self.realized_pnl = 0.0
        self.trades = 0
        self.entries = 0
        self.exits = 0
        self.open_since: Optional[str] = None
        self.entry_symbol = ""
        self.entry_strike: Optional[float] = None
        self.option_type = ""
        self.pnl_source = ""
        self.wins = 0
        self.losses = 0
        self.win_rate = 0.0
        self.max_drawdown = 0.0
        self.last_snapshot_id = None
        self.last_eval_ts = None
        self.data_quality = ""
        self.missing_features: List[str] = []
        self.last_action = ""
        self.raw_reason = ""
        self.ensemble_prob: Optional[float] = None
        self.xgb_prob: Optional[float] = None
        self.rf_prob: Optional[float] = None
        self.block_reason = ""
        self.allowed: Optional[bool] = None
        self.model_type = ""
        self.feature_missing_count = 0
        self.feature_invalid_count = 0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _option_chain_quality_status(rows: int, min_rows: int = 20) -> str:
    rows = int(rows or 0)
    min_rows = int(min_rows or 20)
    if rows <= 0:
        return "option_chain_empty"
    if rows < min_rows:
        return "THIN_OPTION_CHAIN"
    return "DATA_OK"


def _estimate_paper_cost(premium: float, qty: int = 1) -> float:
    """Rough round-trip cost for synthetic paper marks."""
    prem = max(float(premium or 0), 0.05)
    return prem * 0.0025 * max(int(qty or 1), 1)


def _positive_int(value: Any) -> Optional[int]:
    try:
        out = int(float(value))
        return out if out > 0 else None
    except Exception:
        return None


def _paper_forward_default_lot_size() -> tuple[int, str]:
    lot = _positive_int(os.getenv("NIFTY_LOT_SIZE"))
    if lot:
        return lot, "config"
    lot = _positive_int(os.getenv("PAPER_FORWARD_DEFAULT_QTY"))
    if lot:
        return lot, "config"
    return 65, "default"


def _resolve_paper_position_qty(contract: Optional[dict] = None, lots: Any = 1) -> tuple[int, int, int, str]:
    lots_i = _positive_int(lots) or 1
    lot_size = _positive_int((contract or {}).get("lot_size"))
    source = "contract" if lot_size else ""
    if not lot_size:
        lot_size, source = _paper_forward_default_lot_size()
    qty = lots_i * lot_size
    return qty, lot_size, lots_i, source


def _compute_option_pnl(entry_price: Any, current_price: Any, qty: Any = None, side: str = "BUY") -> float:
    """Rupee P&L for option premium movement. BUY is long option, SELL is short option."""
    entry = float(entry_price or 0.0)
    current = float(current_price or 0.0)
    qty_i = _positive_int(qty) or _paper_forward_default_lot_size()[0]
    direction = str(side or "BUY").upper()
    if "SELL" in direction or direction == "SHORT":
        return (entry - current) * qty_i
    return (current - entry) * qty_i


def _parse_ts_epoch(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _is_paper_synthetic_mode(market_snapshot: dict) -> bool:
    chain_src = str(market_snapshot.get("chain_source") or market_snapshot.get("option_chain_source") or "")
    candle_src = str(market_snapshot.get("candle_source") or "").upper()
    return (
        chain_src == "BLACK_SCHOLES_SYNTHETIC"
        or bool(market_snapshot.get("synthetic"))
        or bool(market_snapshot.get("synthetic_candles"))
        or candle_src == "SYNTHETIC_SPOT_FALLBACK"
    )


def _is_eod_cutoff(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        ist = now.astimezone(ZoneInfo("Asia/Kolkata"))
    except Exception:
        ist = now
    return ist.hour > 15 or (ist.hour == 15 and ist.minute >= 30)


def _option_pnl_pct(side: str, entry_p: float, mark_px: float) -> float:
    entry_p = max(float(entry_p or 0), 0.01)
    return (mark_px - entry_p) / entry_p


def _evaluate_paper_exit(
    *,
    side: str,
    final_sig: str,
    entry_p: float,
    mark_px: float,
    entry_epoch: float,
    now_epoch: float,
    cfg: Dict[str, Any],
    is_synth: bool,
) -> Tuple[bool, str]:
    hold_sec = max(0.0, now_epoch - entry_epoch) if entry_epoch else 0.0
    if is_synth and hold_sec < float(cfg.get("synthetic_min_exit_gap_sec", 0)):
        pct = _option_pnl_pct(side, entry_p, mark_px)
        if pct <= -float(cfg.get("paper_stop_pct", 0.10)):
            pass
        else:
            return False, ""
    pct = _option_pnl_pct(side, entry_p, mark_px)
    if pct >= float(cfg.get("paper_target_pct", 0.20)):
        return True, "PAPER_EXIT_TARGET"
    if pct <= -float(cfg.get("paper_stop_pct", 0.10)):
        return True, "PAPER_EXIT_STOP"
    if hold_sec >= float(cfg.get("paper_max_hold_sec", 900)):
        return True, "PAPER_EXIT_MAX_HOLD"
    if cfg.get("exit_on_opposite_signal", True):
        if side == "CE" and final_sig == "BUY_PE":
            return True, "PAPER_EXIT_OPPOSITE"
        if side == "PE" and final_sig == "BUY_CE":
            return True, "PAPER_EXIT_OPPOSITE"
    if _is_eod_cutoff():
        return True, "PAPER_EXIT_EOD"
    return False, ""


def _entry_cooldown_remaining(st: dict, cfg: Dict[str, Any], now_epoch: float, is_synth: bool) -> float:
    if not is_synth:
        return 0.0
    gap = float(cfg.get("synthetic_min_entry_gap_sec", 300))
    last_ts = max(_parse_ts_epoch(st.get("last_entry_ts")), _parse_ts_epoch(st.get("last_exit_ts")))
    if last_ts <= 0:
        return 0.0
    elapsed = now_epoch - last_ts
    return max(0.0, gap - elapsed)


def _can_open_paper_entry(
    st: dict,
    *,
    final_sig: str,
    confidence: Any,
    threshold: Any,
    is_synth: bool,
    cfg: Dict[str, Any],
    now_epoch: float,
) -> Tuple[bool, str, float]:
    if st.get("open_position"):
        return False, "HOLD_EXISTING_POSITION", 0.0
    max_open = int(cfg.get("max_open_per_candidate", 1))
    if max_open <= 0:
        return False, "HOLD_EXISTING_POSITION", 0.0
    remaining = _entry_cooldown_remaining(st, cfg, now_epoch, is_synth)
    if remaining > 0:
        return False, "ENTRY_COOLDOWN", remaining
    if is_synth and final_sig in ("BUY_CE", "BUY_PE"):
        prev_sig = str(st.get("last_eval_signal") or "")
        prev_conf = float(st.get("last_confidence") or 0)
        thr = float(threshold or 0)
        conf_f = float(confidence or 0)
        crossed = prev_conf < thr <= conf_f
        signal_rising = prev_sig in ("", "NO_TRADE") and final_sig in ("BUY_CE", "BUY_PE")
        if not (signal_rising or crossed):
            return False, "SIGNAL_DEDUPE", 0.0
    return True, "", 0.0


def _synthetic_option_mark(
    market_snapshot: dict,
    side: str,
    *,
    strike: Optional[float] = None,
) -> Tuple[Optional[float], str, str]:
    """Return (mid_price, symbol, pnl_source) from synthetic option chain."""
    chain = market_snapshot.get("option_chain") or market_snapshot.get("_option_chain_snapshot") or []
    if not isinstance(chain, (list, tuple)):
        return None, "", ""
    spot = (
        market_snapshot.get("spot")
        or market_snapshot.get("price")
        or market_snapshot.get("close")
        or market_snapshot.get("last")
    )
    try:
        spot_f = float(spot) if spot not in (None, "", 0, 0.0) else 0.0
    except Exception:
        spot_f = 0.0
    opt_side = "CE" if "CE" in str(side or "").upper() else "PE"
    target_strike = strike
    if target_strike is None and spot_f > 0:
        try:
            from synthetic_option_chain import atm_strike, load_bs_config
            target_strike = atm_strike(spot_f, int(load_bs_config().get("strike_step", 50)))
        except Exception:
            target_strike = round(spot_f / 50) * 50
    best = None
    for row in chain:
        if not isinstance(row, dict):
            continue
        if str(row.get("option_type", "")).upper() != opt_side:
            continue
        if target_strike is not None:
            try:
                if abs(float(row.get("strike_price") or row.get("strike") or 0) - float(target_strike)) > 0.01:
                    continue
            except Exception:
                pass
        best = row
        break
    if best is None:
        for row in chain:
            if isinstance(row, dict) and str(row.get("option_type", "")).upper() == opt_side:
                best = row
                break
    if not best:
        return None, "", ""
    try:
        mid = float(best.get("mid") or best.get("ltp") or best.get("theoretical_price") or 0)
    except Exception:
        mid = 0.0
    sym = str(best.get("trading_symbol") or best.get("symbol") or "")
    return (mid if mid > 0 else None), sym, "SYNTHETIC_MARK"


def _option_price_from_row(row: dict) -> Optional[float]:
    """Best-effort option premium from a chain row (broker LTP/mid preferred)."""
    if not isinstance(row, dict):
        return None
    for key in (
        "ltp", "LTP", "last_price", "lastPrice", "last_price_value",
        "last_traded_price", "lastTradedPrice", "option_ltp", "close",
        "mid", "theoretical_price",
    ):
        try:
            val = float(row.get(key) or 0)
            if val > 0:
                return val
        except Exception:
            pass
    try:
        bid = float(row.get("bid") or row.get("bid_price") or row.get("best_bid") or row.get("best_bid_price") or 0)
        ask = float(row.get("ask") or row.get("ask_price") or row.get("best_ask") or row.get("best_ask_price") or 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        if bid > 0:
            return bid
        if ask > 0:
            return ask
    except Exception:
        pass
    return None


def _option_token_from_row(row: dict) -> str:
    if not isinstance(row, dict):
        return ""
    for key in (
        "token",
        "symbolToken",
        "symboltoken",
        "instrumentToken",
        "instrumenttoken",
        "instrument_token",
        "security_id",
        "securityId",
        "securityid",
        "scrip_token",
        "scripToken",
    ):
        val = row.get(key)
        if val not in (None, ""):
            return str(val).strip()
    return ""


def _enrich_contract_tokens(contract: dict, client: Any = None) -> dict:
    """Fill missing broker token fields via ScripMaster structured lookup."""
    out = dict(contract or {})
    tok = _option_token_from_row(out)
    sym = str(
        out.get("trading_symbol")
        or out.get("symbol")
        or out.get("tradingsymbol")
        or ""
    ).strip()
    if tok and tok.isdigit():
        out["token"] = tok
        out.setdefault("symbolToken", tok)
        out.setdefault("security_id", tok)
        out.setdefault("exchange", str(out.get("exchange") or "NFO").strip().upper() or "NFO")
        return out
    sm = None
    if client is not None and hasattr(client, "_get_scripmaster"):
        try:
            sm = client._get_scripmaster()
        except Exception:
            sm = None
    if sm is None:
        return out
    try:
        from scripmaster import contract_dict_from_scrip_row, parse_option_tradingsymbol
    except Exception:
        try:
            from src.scripmaster import contract_dict_from_scrip_row, parse_option_tradingsymbol  # type: ignore
        except Exception:
            return out
    row = None
    if sym and hasattr(sm, "lookup_option_contract"):
        row = sm.lookup_option_contract(tradingsymbol=sym, exch="NFO")
    if row is None and sym:
        parsed = parse_option_tradingsymbol(sym)
        if parsed and hasattr(sm, "lookup_option_contract"):
            row = sm.lookup_option_contract(
                underlying=str(parsed.get("underlying") or ""),
                expiry=parsed.get("expiry"),
                strike=parsed.get("strike"),
                option_type=str(parsed.get("option_type") or ""),
                exch="NFO",
            )
    if row is None:
        try:
            strike = int(float(out.get("strike_price") or out.get("strike") or 0))
        except Exception:
            strike = None
        exp_raw = out.get("expiry") or out.get("expiry_date")
        exp_date = None
        if exp_raw:
            try:
                from datetime import date as _date
                if hasattr(exp_raw, "isoformat"):
                    exp_date = exp_raw if isinstance(exp_raw, _date) else exp_raw.date()
                else:
                    from scripmaster import _parse_date
                    exp_date = _parse_date(exp_raw)
            except Exception:
                exp_date = None
        if strike and exp_date and hasattr(sm, "lookup_option_contract"):
            row = sm.lookup_option_contract(
                underlying=str(out.get("symbol_root") or "NIFTY"),
                expiry=exp_date,
                strike=strike,
                option_type=str(out.get("option_type") or ""),
                exch="NFO",
            )
    if row is not None:
        enriched = contract_dict_from_scrip_row(row)
        for k, v in enriched.items():
            if v in (None, ""):
                continue
            if k in ("trading_symbol", "symbol", "tradingsymbol", "token", "symbolToken", "security_id", "exchange"):
                out[k] = v
            elif not out.get(k):
                out[k] = v
        for k in ("ltp", "mid", "theoretical_price", "bid", "ask"):
            if out.get(k) not in (None, "", 0, 0.0):
                out[k] = out.get(k)
    out.setdefault("exchange", "NFO")
    return out


def _contract_payload_from_row(row: dict, *, client: Any = None, candidate_id: str = "") -> dict:
    sym = str(row.get("trading_symbol") or row.get("symbol") or row.get("tradingsymbol") or "")
    contract = {
        "trading_symbol": sym,
        "symbol": sym,
        "tradingsymbol": sym,
        "exchange": str(row.get("exchange") or row.get("exch_seg") or "NFO").strip().upper() or "NFO",
        "token": _option_token_from_row(row),
        "symbolToken": _option_token_from_row(row),
        "security_id": _option_token_from_row(row),
        "strike_price": row.get("strike_price") or row.get("strike"),
        "strike": row.get("strike") or row.get("strike_price"),
        "option_type": row.get("option_type") or row.get("type"),
        "expiry": row.get("expiry") or row.get("expiry_date"),
        "expiry_date": row.get("expiry_date") or row.get("expiry"),
        "lot_size": row.get("lot_size"),
        "ltp": row.get("ltp") or row.get("last_price") or row.get("mid") or row.get("theoretical_price"),
        "mid": row.get("mid"),
        "theoretical_price": row.get("theoretical_price"),
        "bid": row.get("bid"),
        "ask": row.get("ask"),
    }
    contract = _enrich_contract_tokens(contract, client)
    tok = _option_token_from_row(contract)
    cid = candidate_id or "unknown"
    row_spot = row.get("spot") or row.get("underlying_price")
    opt_ltp = _option_price_from_row(row) or _option_price_from_row(contract)
    chain_kind = "synthetic_chain" if bool(row.get("synthetic")) or str(row.get("chain_source") or "") == "BLACK_SCHOLES_SYNTHETIC" else "real_chain"
    print(
        f"[PF-CONTRACT-SELECT] candidate_id={cid} symbol={contract.get('symbol')} "
        f"token={tok} security_id={contract.get('security_id')} "
        f"spot={row_spot} option_ltp={opt_ltp} source={chain_kind} "
        f"exchange={contract.get('exchange')} strike={contract.get('strike_price')} "
        f"option_type={contract.get('option_type')}"
    )
    if not tok or not str(tok).isdigit():
        print(f"[PF-CONTRACT-SELECT-ERROR] candidate_id={cid} reason=missing_token_after_selection symbol={sym}")
    return contract


_LTP_FAIL_CACHE: Dict[str, float] = {}
_LTP_SUCCESS_CACHE: Dict[str, str] = {}
_LTP_FAIL_TTL_SEC = float(os.getenv("PF_LTP_FAIL_TTL_SEC", "2.0") or 2.0)
_LTP_FAIL_RETRY_ONCE = str(os.getenv("PF_LTP_FAIL_RETRY_ONCE", "true")).strip().lower() in {"1", "true", "yes"}
_LAST_GOOD_MARKS: Dict[str, Dict[str, Any]] = {}
_LAST_GOOD_MARK_TTL_SEC = float(os.getenv("PF_LAST_GOOD_MARK_TTL_SEC", "30") or 30)
_MARK_SOURCE_PRIORITY = {
    "SYNTHETIC_MARK": 10,
    "STALE_SYNTHETIC": 15,
    "LAST_GOOD_MARK": 40,
    "MID_BID_ASK": 60,
    "REAL_CHAIN_LTP": 70,
    "CHAIN_OPTION_LTP": 70,
    "BROKER_LTP": 100,
    "BROKER_LIVE_LTP": 100,
}


def _ltp_cache_key(symbol: str, token: str, exchange: str) -> str:
    return f"{exchange}:{token or symbol}".upper()


def _ltp_fail_key(method: str, key: str) -> str:
    return f"{method}:{str(key or '').strip().upper()}"


def _ltp_symbol_key(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _ltp_cache_lookup(symbol: str, token: str = "", exchange: str = "NFO") -> Tuple[str, bool]:
    """Return (token, hit) from success cache keyed by symbol."""
    sym_key = _ltp_symbol_key(symbol)
    if sym_key and sym_key in _LTP_SUCCESS_CACHE:
        cached_tok = _LTP_SUCCESS_CACHE[sym_key]
        print(f"[PF-LTP-CACHE] hit symbol={sym_key} token={cached_tok}")
        return cached_tok, True
    if token and sym_key:
        print(f"[PF-LTP-CACHE] miss symbol={sym_key} token={token}")
    return token, False


def _ltp_fail_cached(key: str, *, allow_retry: bool = False, symbol: str = "") -> bool:
    ts = _LTP_FAIL_CACHE.get(key)
    if ts is None:
        return False
    age = time.time() - ts
    if age >= _LTP_FAIL_TTL_SEC:
        _LTP_FAIL_CACHE.pop(key, None)
        return False
    if allow_retry and _LTP_FAIL_RETRY_ONCE and age >= (_LTP_FAIL_TTL_SEC * 0.5):
        _LTP_FAIL_CACHE.pop(key, None)
        return False
    sym_key = _ltp_symbol_key(symbol) if symbol else ""
    if sym_key:
        print(f"[PF-LTP-CACHE] hit symbol={sym_key} status=SKIPPED_CACHED_FAIL age={age:.2f}s")
    return True


def _remember_ltp_fail(key: str) -> None:
    _LTP_FAIL_CACHE[key] = time.time()


def _mark_symbol_key(symbol: str, token: str = "", exchange: str = "NFO") -> str:
    sym = str(symbol or "").strip().upper()
    tok = str(token or "").strip().upper()
    exch = str(exchange or "NFO").strip().upper() or "NFO"
    return sym or f"{exch}:{tok}"


def _record_last_good_mark(
    *,
    symbol: str,
    price: Any,
    source: str,
    token: str = "",
    exchange: str = "NFO",
) -> None:
    try:
        px = float(price)
    except Exception:
        return
    if px <= 0:
        return
    src = str(source or "").upper()
    if src not in {"BROKER_LTP", "BROKER_LIVE_LTP", "REAL_CHAIN_LTP", "CHAIN_OPTION_LTP", "MID_BID_ASK"}:
        return
    key = _mark_symbol_key(symbol, token, exchange)
    if not key:
        return
    _LAST_GOOD_MARKS[key] = {
        "symbol": str(symbol or "").strip(),
        "token": str(token or "").strip(),
        "exchange": str(exchange or "NFO").strip().upper() or "NFO",
        "price": px,
        "source": src,
        "timestamp": time.time(),
    }


def _last_good_mark(symbol: str, token: str = "", exchange: str = "NFO") -> tuple[Optional[float], str, bool, Dict[str, Any]]:
    keys = [_mark_symbol_key(symbol, token, exchange)]
    if symbol:
        keys.append(_mark_symbol_key(symbol, "", exchange))
    for key in keys:
        rec = _LAST_GOOD_MARKS.get(key)
        if not rec:
            continue
        age = time.time() - float(rec.get("timestamp") or 0.0)
        stale = age > _LAST_GOOD_MARK_TTL_SEC
        try:
            px = float(rec.get("price") or 0.0)
        except Exception:
            px = 0.0
        if px > 0:
            return px, str(rec.get("source") or "LAST_GOOD_MARK"), stale, dict(rec)
    return None, "", True, {}


def _log_mark_source(symbol: str, selected: str, previous: str, price: Any, stale: bool, reason: str) -> None:
    try:
        from paper_forward_readiness import normalize_ltp_source
    except ImportError:
        from .paper_forward_readiness import normalize_ltp_source  # type: ignore
    try:
        price_s = f"{float(price):.4f}"
    except Exception:
        price_s = str(price)
    sel = normalize_ltp_source(selected)
    prev = normalize_ltp_source(previous) if previous else "-"
    print(
        f"[PF-MARK-SOURCE] symbol={symbol} selected={sel} previous={prev} "
        f"price={price_s} stale={str(bool(stale)).lower()} reason={reason}"
    )


def _chain_rows_for_marking(market_snapshot: dict) -> List[dict]:
    rows: List[dict] = []
    for key in ("broker_live_option_chain", "broker_option_chain", "option_chain", "_option_chain_snapshot"):
        chain = market_snapshot.get(key)
        if isinstance(chain, (list, tuple)):
            rows.extend([r for r in chain if isinstance(r, dict)])
    return rows


def _row_has_broker_token(row: dict) -> bool:
    tok = _option_token_from_row(row)
    return bool(tok and str(tok).isdigit())


def _row_exchange_ok(row: dict) -> bool:
    exch = str(row.get("exchange") or row.get("exch_seg") or row.get("exch") or "NFO").strip().upper()
    return exch in {"NFO", "NSEFO", ""}


def _match_option_chain_row(
    chain: Sequence[Any],
    *,
    side: str,
    strike: Optional[float],
    spot: float,
    strike_step: int = 50,
    entry_symbol: str = "",
) -> Optional[dict]:
    opt_side = "CE" if "CE" in str(side or "").upper() else "PE"
    desired_symbol = str(entry_symbol or "").strip().upper()
    if desired_symbol:
        token_match = None
        for row in chain:
            if not isinstance(row, dict):
                continue
            row_symbol = str(row.get("trading_symbol") or row.get("symbol") or row.get("tradingsymbol") or "").strip().upper()
            if row_symbol and row_symbol == desired_symbol:
                if _row_has_broker_token(row) and _row_exchange_ok(row):
                    return row
                if token_match is None:
                    token_match = row
        if token_match is not None:
            return token_match
    target_strike = strike
    if target_strike is None and spot > 0:
        try:
            from synthetic_option_chain import atm_strike
            target_strike = atm_strike(spot, strike_step)
        except Exception:
            target_strike = round(spot / max(strike_step, 1)) * strike_step
    best = None
    best_score = float("inf")
    allowed_types = {opt_side, "CALL" if opt_side == "CE" else "PUT", "C" if opt_side == "CE" else "P"}
    for row in chain:
        if not isinstance(row, dict):
            continue
        if str(row.get("option_type", "")).upper() not in allowed_types:
            continue
        if not _row_exchange_ok(row):
            continue
        try:
            row_strike = float(row.get("strike_price") or row.get("strike") or 0)
        except Exception:
            row_strike = 0.0
        if target_strike is not None and row_strike > 0:
            dist = abs(row_strike - float(target_strike))
            token_bonus = -0.001 if _row_has_broker_token(row) else 0.0
            score = dist + token_bonus
            if dist <= 0.01 and _row_has_broker_token(row):
                return row
            if score < best_score:
                best_score = score
                best = row
        elif best is None:
            best = row
    return best


def _fetch_live_broker_option_ltp(
    client: Any,
    *,
    symbol: str,
    token: str = "",
    exchange: str = "NFO",
    contract: Optional[dict] = None,
) -> Tuple[Optional[float], str]:
    """Fetch live option LTP from broker; returns (price, source_tag)."""
    contract = dict(contract or {})
    sym = str(contract.get("trading_symbol") or contract.get("symbol") or symbol or "").strip()
    tok = str(contract.get("token") or contract.get("symbolToken") or contract.get("security_id") or token or "").strip()
    if client is None or (not sym and not tok):
        return None, ""
    exch = str(contract.get("exchange") or exchange or "NFO").strip().upper() or "NFO"

    if ":" in sym:
        try:
            maybe_exch, maybe_sym = sym.split(":", 1)
            maybe_exch = maybe_exch.strip().upper()
            if maybe_exch in {"NSE", "BSE", "NFO", "NSEFO", "NSECM", "BSECM"} and maybe_sym.strip():
                sym = maybe_sym.strip()
                exch = "NFO" if maybe_exch in {"NFO", "NSEFO"} else maybe_exch
        except Exception:
            pass
    if exch in {"NSEFO", "NFO"}:
        exch = "NFO"

    contract = _enrich_contract_tokens(
        {"trading_symbol": sym, "symbol": sym, "token": tok, "exchange": exch, **contract},
        client,
    )
    sym = str(contract.get("trading_symbol") or contract.get("symbol") or sym)
    tok = _option_token_from_row(contract)
    cached_tok, cache_hit = _ltp_cache_lookup(sym, tok, exch)
    if cache_hit and cached_tok:
        tok = cached_tok
    exch = str(contract.get("exchange") or exch).strip().upper() or "NFO"

    # 1) Token-based quote/LTP
    if tok and tok.isdigit():
        cache_key = _ltp_fail_key("token_quote", _ltp_cache_key(sym, tok, exch))
        if _ltp_fail_cached(cache_key, allow_retry=True, symbol=sym):
            print(f"[PF-LTP-RESOLVE] symbol={sym} method=token token={tok} status=SKIPPED_CACHED_FAIL")
        elif hasattr(client, "fetch_option_quote_for_paper"):
            try:
                print(f"[PF-LTP-TRY] method=token symbol={sym} token={tok} exchange={exch}")
                quote = client.fetch_option_quote_for_paper(exch, tok, sym) or {}
                px = _option_price_from_row(quote)
                if px and px > 0:
                    if sym:
                        _LTP_SUCCESS_CACHE[_ltp_symbol_key(sym)] = tok
                    _LTP_FAIL_CACHE.pop(cache_key, None)
                    print(f"[PF-LTP-RESOLVE] symbol={sym} method=token token={tok} status=OK")
                    print(f"[PF-LTP-SELECTED] method=token price={float(px):.4f}")
                    print(f"[PF-LTP] source=broker_ltp price={float(px):.4f}")
                    return float(px), "BROKER_LTP"
                print(f"[PF-LTP-RESOLVE] symbol={sym} method=token token={tok} status=FAILED")
                _remember_ltp_fail(cache_key)
            except Exception as exc:
                print(f"[PF-LTP-RESOLVE] symbol={sym} method=token token={tok} status=FAILED error={str(exc)[:80]}")
                _remember_ltp_fail(cache_key)
        if hasattr(client, "get_ltp"):
            for sym_key in (f"{exch}:{tok}", tok, f"NFO:{tok}"):
                fail_key = _ltp_fail_key("token_get_ltp", sym_key)
                if not sym_key or _ltp_fail_cached(fail_key, allow_retry=True, symbol=sym):
                    continue
                try:
                    print(f"[PF-LTP-TRY] method=token symbol={sym} key={sym_key}")
                    raw_ltp = client.get_ltp(sym_key)
                    if raw_ltp in (None, ""):
                        print(f"[PF-LTP-RESOLVE] symbol={sym} method=token key={sym_key} status=BROKER_LTP_UNAVAILABLE")
                        continue
                    ltp = float(raw_ltp)
                    if ltp > 0:
                        if sym:
                            _LTP_SUCCESS_CACHE[_ltp_symbol_key(sym)] = tok
                        _LTP_FAIL_CACHE.pop(fail_key, None)
                        print(f"[PF-LTP-RESOLVE] symbol={sym} method=token token={tok} status=OK")
                        print(f"[PF-LTP-SELECTED] method=token price={ltp:.4f}")
                        print(f"[PF-LTP] source=broker_ltp price={ltp:.4f}")
                        return ltp, "BROKER_LTP"
                except Exception as exc:
                    print(f"[PF-LTP-RESOLVE] symbol={sym} method=token token={tok} status=FAILED error={str(exc)[:80]}")
                    _remember_ltp_fail(fail_key)

    # 2) Resolve token via broker if still missing
    if sym and (not tok or not tok.isdigit()) and hasattr(client, "resolve_exchange_token"):
        try:
            resolved_exch, resolved_tok = client.resolve_exchange_token(sym, exchange_hint=exch)
            if resolved_tok:
                tok = str(resolved_tok).strip()
            if resolved_exch:
                exch = "NFO" if str(resolved_exch).upper() in {"NSEFO", "NFO"} else str(resolved_exch).upper()
            if tok and tok.isdigit():
                print(f"[PF-LTP-RESOLVE] symbol={sym} method=scripmaster_structured token={tok} status=OK")
                return _fetch_live_broker_option_ltp(
                    client,
                    symbol=sym,
                    token=tok,
                    exchange=exch,
                    contract=contract,
                )
            print(f"[PF-LTP-RESOLVE] symbol={sym} method=scripmaster_structured status=FAILED")
        except Exception as exc:
            print(f"[PF-LTP-RESOLVE] symbol={sym} method=scripmaster_structured status=FAILED error={str(exc)[:80]}")

    # 3) Exchange + trading symbol
    if hasattr(client, "get_ltp"):
        for sym_key in (f"{exch}:{sym}", sym, f"NFO:{sym}"):
            fail_key = _ltp_fail_key("exchange_symbol", sym_key)
            if not sym_key or _ltp_fail_cached(fail_key, allow_retry=True, symbol=sym):
                continue
            try:
                print(f"[PF-LTP-TRY] method=exchange_symbol symbol={sym} key={sym_key}")
                raw_ltp = client.get_ltp(sym_key)
                if raw_ltp in (None, ""):
                    print(f"[PF-LTP-RESOLVE] symbol={sym} method=exchange_symbol key={sym_key} status=BROKER_LTP_UNAVAILABLE")
                    continue
                ltp = float(raw_ltp)
                if ltp > 0:
                    if sym and tok:
                        _LTP_SUCCESS_CACHE[_ltp_symbol_key(sym)] = tok
                    _LTP_FAIL_CACHE.pop(fail_key, None)
                    _LTP_FAIL_CACHE.pop(_ltp_fail_key("symbol", sym), None)
                    print(f"[PF-LTP-RESOLVE] symbol={sym} method=exchange_symbol key={sym_key} status=OK")
                    print(f"[PF-LTP-SELECTED] method=exchange_symbol price={ltp:.4f}")
                    print(f"[PF-LTP] source=broker_ltp price={ltp:.4f}")
                    return ltp, "BROKER_LTP"
            except Exception as exc:
                print(f"[PF-LTP-RESOLVE] symbol={sym} method=exchange_symbol key={sym_key} status=FAILED error={str(exc)[:80]}")
                _remember_ltp_fail(fail_key)

    print(f"[PF-LTP] source=synthetic_fallback reason=broker_ltp_unavailable symbol={sym} token={tok}")
    print(f"[PF-OPTION-LIVE-LTP] status=UNAVAILABLE symbol={sym} token={tok} exchange={exch}")
    return None, ""


def _broker_option_mark(
    market_snapshot: dict,
    side: str,
    *,
    strike: Optional[float] = None,
    entry_symbol: str = "",
    spot: float = 0.0,
    strike_step: int = 50,
    candidate_id: str = "",
) -> Tuple[Optional[float], str, str, Optional[float], dict]:
    """Resolve option mark from live broker chain/LTP; None if unavailable."""
    chain = _chain_rows_for_marking(market_snapshot)
    client = market_snapshot.get("broker_client")
    row = _match_option_chain_row(chain, side=side, strike=strike, spot=spot, strike_step=strike_step, entry_symbol=entry_symbol)
    contract: dict = {}
    if row is not None:
        contract = _contract_payload_from_row(row, client=client, candidate_id=candidate_id)
        is_synth_row = bool(row.get("synthetic")) or str(row.get("chain_source") or "") == "BLACK_SCHOLES_SYNTHETIC"
        px = _option_price_from_row(row) or _option_price_from_row(contract)
        sym = str(contract.get("trading_symbol") or contract.get("symbol") or entry_symbol or "")
        used_strike = strike
        try:
            used_strike = float(contract.get("strike_price") or contract.get("strike") or strike or 0) or strike
        except Exception:
            pass
        live_px, src = _fetch_live_broker_option_ltp(client, symbol=sym, contract=contract)
        if live_px and live_px > 0:
            pnl_src = "BROKER_LTP" if src in {"BROKER_LTP", "BROKER_LIVE_LTP"} else src
            _record_last_good_mark(
                symbol=sym,
                token=_option_token_from_row(contract),
                exchange=str(contract.get("exchange") or "NFO"),
                price=float(live_px),
                source=pnl_src,
            )
            return float(live_px), sym, pnl_src, used_strike, contract
        if px and px > 0 and not is_synth_row:
            src = "MID_BID_ASK" if _option_price_from_row(row) and (row.get("bid") or row.get("ask") or row.get("bid_price") or row.get("ask_price")) else "CHAIN_OPTION_LTP"
            _record_last_good_mark(
                symbol=sym,
                token=_option_token_from_row(contract),
                exchange=str(contract.get("exchange") or "NFO"),
                price=float(px),
                source=src,
            )
            return float(px), sym, src, used_strike, contract

    sym = str(entry_symbol or "").strip()
    if client is not None and sym:
        contract = _enrich_contract_tokens({"trading_symbol": sym, "symbol": sym, "exchange": "NFO"}, client)
        live_px, src = _fetch_live_broker_option_ltp(client, symbol=sym, contract=contract)
        if live_px and live_px > 0:
            pnl_src = "BROKER_LTP" if src in {"BROKER_LTP", "BROKER_LIVE_LTP"} else src
            _record_last_good_mark(
                symbol=str(contract.get("trading_symbol") or sym),
                token=_option_token_from_row(contract),
                exchange=str(contract.get("exchange") or "NFO"),
                price=float(live_px),
                source=pnl_src,
            )
            return float(live_px), str(contract.get("trading_symbol") or sym), pnl_src, strike, contract
    return None, "", "", None, contract


def _count_option_rows(option_chain_snapshot: Any) -> int:
    if isinstance(option_chain_snapshot, (list, tuple)):
        return len([r for r in option_chain_snapshot if isinstance(r, dict)])
    if isinstance(option_chain_snapshot, dict):
        if isinstance(option_chain_snapshot.get("option_chain"), (list, tuple)):
            return len(option_chain_snapshot.get("option_chain") or [])
        if isinstance(option_chain_snapshot.get("rows"), (list, tuple)):
            return len(option_chain_snapshot.get("rows") or [])
        if any(k in option_chain_snapshot for k in ("strike", "strikePrice", "strike_price", "option_type", "CE", "PE", "tradingsymbol", "symbol")):
            return 1
        return int(option_chain_snapshot.get("option_rows") or option_chain_snapshot.get("option_chain_rows") or 0)
    return 0


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except Exception:
        return None


def _to_float_or_zero(value: Any) -> float:
    """Numeric coercion for arithmetic; preserves 0.0 (unlike _to_float missing-sentinel)."""
    try:
        if value is None or value == "":
            return 0.0
        out = float(value)
        if math.isnan(out):
            return 0.0
        return out
    except Exception:
        return 0.0


def format_paper_forward_confidence(
    confidence: Any,
    *,
    predict_attempted: bool = False,
    reason: str = "",
    load_status: str = "",
) -> str:
    """UI/export confidence cell: numeric prob or explicit skip reason (never bare N/A when ready)."""
    if isinstance(confidence, (int, float)) and not (isinstance(confidence, float) and math.isnan(confidence)):
        value = float(confidence)
        if 0 < abs(value) < 0.00005:
            return f"{value:.2e}"
        return f"{value:.4f}"
    ru = str(reason or "").upper()
    ls = str(load_status or "").upper()
    if "ARTIFACT" in ls or "ARTIFACT" in ru:
        return "ARTIFACT_NOT_FOUND"
    if "FEATURE_ORDER_MISSING" in ls or "FEATURE_ORDER_MISSING" in ru:
        return "FEATURES_MISSING"
    if "FEATURES_MISSING" in ru or "MISSING_FEATURES" in ru:
        return "FEATURES_MISSING"
    if "MODEL_NOT_LOADED" in ru or "model_pkl_not_found" in ru.lower():
        return "MODEL_NOT_LOADED"
    if "PREDICT_ERROR" in ru or "model_route_exception" in ru.lower():
        return "PREDICT_EXCEPTION"
    if "INVALID_OUTPUT" in ru:
        return "INVALID_OUTPUT"
    if "NO_VALID_CONTRACT" in ru:
        return "NO_VALID_CONTRACT"
    if "XGBOOST_NOT_INSTALLED" in ru:
        return "XGBOOST_NOT_INSTALLED"
    if ru.startswith("LOW_CONFIDENCE") or "low_confidence" in ru.lower():
        return "LOW_CONFIDENCE"
    if predict_attempted:
        return "0.0000"
    if any(x in ru for x in ("WAITING_", "TOKEN_NOT", "SESSION_EXPIRED", "BROKER_", "CANDLE", "OPTION_CHAIN", "DATA_NOT", "NOT_READY")):
        return "NOT_READY"
    if ru:
        return ru[:40]
    return "NOT_READY"


def _paper_fwd_log(prefix: str, **fields: Any) -> None:
    """Verbose structured log — suppressed at INFO unless explicitly DEBUG."""
    if not pf_should_log_level("DEBUG"):
        return
    parts = [f"{k}={fields[k]}" for k in sorted(fields.keys())]
    cid = str(fields.get("candidate_id") or "")
    pf_log(
        "DEBUG",
        f"[{prefix}] " + " ".join(parts),
        rate_key=f"{prefix}:{cid}",
        rate_interval=30.0,
    )


def _walk_dict_nodes(obj: Any) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        nodes.append(obj)
        for value in obj.values():
            nodes.extend(_walk_dict_nodes(value))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            nodes.extend(_walk_dict_nodes(item))
    return nodes


def _derive_spot_from_chain(obj: Any) -> Optional[float]:
    keys = (
        "underlyingValue",
        "underlying_value",
        "underlyingPrice",
        "underlying_price",
        "underlying_spot_price",
        "underlyingSpotPrice",
        "index_spot",
        "indexSpot",
        "spot",
        "spot_price",
        "spotPrice",
    )
    for node in _walk_dict_nodes(obj):
        for key in keys:
            value = _to_float(node.get(key))
            if value is not None:
                return value
    return None


def _first_option_row(option_chain_snapshot: Any, candidate: Dict[str, Any] | None = None, spot: Any = None) -> Dict[str, Any]:
    if isinstance(option_chain_snapshot, dict):
        if isinstance(option_chain_snapshot.get("option_chain"), (list, tuple)) and option_chain_snapshot["option_chain"]:
            option_chain_snapshot = option_chain_snapshot["option_chain"]
        else:
            return dict(option_chain_snapshot)
    if not isinstance(option_chain_snapshot, (list, tuple)) or not option_chain_snapshot:
        return {}
    rows = [r for r in option_chain_snapshot if isinstance(r, dict)]
    if not rows:
        return {}
    side = str((candidate or {}).get("side_policy") or (candidate or {}).get("side") or "").upper()
    desired_side = ""
    if side in ("CE", "CALL") or ("CE_ONLY" in side):
        desired_side = "CE"
    elif side in ("PE", "PUT") or ("PE_ONLY" in side):
        desired_side = "PE"
    if desired_side:
        side_rows = [
            r for r in rows
            if str(r.get("option_type") or r.get("ce_pe") or r.get("right") or r.get("side") or "").upper()
            in (desired_side, "CALL" if desired_side == "CE" else "PUT", "C" if desired_side == "CE" else "P")
        ]
        if side_rows:
            rows = side_rows
    s = _to_float(spot)
    if s is not None:
        def dist(row: Dict[str, Any]) -> float:
            strike = _to_float(row.get("strike") or row.get("strike_price") or row.get("strikePrice") or row.get("StrikePrice"))
            return abs((strike if strike is not None else s) - s)
        return dict(sorted(rows, key=dist)[0])
    return dict(rows[0])


def _coerce_live_builder_candle(value: Any) -> Any:
    """Return a market_data.Candle-like object for the richer live feature builder."""
    try:
        from market_data import Candle
    except Exception:
        Candle = None  # type: ignore
    if value is None:
        return None
    if all(hasattr(value, k) for k in ("open", "high", "low", "close")):
        return value
    if not isinstance(value, dict):
        return None
    ts = _parse_datetime_any(value.get("time") or value.get("timestamp") or value.get("ts")) or datetime.now(timezone.utc)
    if Candle is not None:
        return Candle(
            time=ts,
            open=float(_to_float(value.get("open")) or 0.0),
            high=float(_to_float(value.get("high")) or _to_float(value.get("open")) or 0.0),
            low=float(_to_float(value.get("low")) or _to_float(value.get("open")) or 0.0),
            close=float(_to_float(value.get("close")) or _to_float(value.get("last")) or _to_float(value.get("ltp")) or 0.0),
            volume=_to_float(value.get("volume") or value.get("candle_volume") or 0.0),
        )
    return type("LiveCandle", (), {
        "time": ts,
        "open": float(_to_float(value.get("open")) or 0.0),
        "high": float(_to_float(value.get("high")) or _to_float(value.get("open")) or 0.0),
        "low": float(_to_float(value.get("low")) or _to_float(value.get("open")) or 0.0),
        "close": float(_to_float(value.get("close")) or _to_float(value.get("last")) or _to_float(value.get("ltp")) or 0.0),
        "volume": _to_float(value.get("volume") or value.get("candle_volume") or 0.0),
    })()


def _normalize_option_row_for_live_builder(row: Dict[str, Any], chain: Any, spot: Any) -> Dict[str, Any]:
    out = dict(row or {})
    out["strike"] = _to_float(out.get("strike") or out.get("strike_price") or out.get("strikePrice") or out.get("StrikePrice")) or 0.0
    opt_type = str(out.get("option_type") or out.get("ce_pe") or out.get("right") or out.get("side") or "").upper()
    if opt_type in ("CALL", "C"):
        opt_type = "CE"
    elif opt_type in ("PUT", "P"):
        opt_type = "PE"
    out["option_type"] = opt_type
    out["ltp"] = _to_float(out.get("ltp") or out.get("last_price") or out.get("lastPrice") or out.get("option_ltp") or out.get("traded_price") or out.get("close")) or 0.0
    out["bid"] = _to_float(out.get("bid") or out.get("bid_price") or out.get("best_bid")) or 0.0
    out["ask"] = _to_float(out.get("ask") or out.get("ask_price") or out.get("best_ask")) or 0.0
    out["iv"] = _to_float(out.get("iv") or out.get("final_iv") or out.get("bs_iv") or out.get("implied_volatility")) or 0.0
    out["delta"] = _to_float(out.get("delta") or out.get("bs_delta")) or 0.0
    out["gamma"] = _to_float(out.get("gamma") or out.get("bs_gamma")) or 0.0
    out["theta"] = _to_float(out.get("theta") or out.get("bs_theta")) or 0.0
    out["vega"] = _to_float(out.get("vega") or out.get("bs_vega")) or 0.0
    out["oi"] = _to_float(out.get("oi") or out.get("open_interest") or out.get("openInterest")) or 0.0
    out["volume"] = _to_float(out.get("volume") or out.get("traded_volume") or out.get("tradedVolume")) or 0.0
    out["change_in_oi"] = _to_float(out.get("change_in_oi") or out.get("oi_change") or out.get("changeOI")) or 0.0
    out["open"] = _to_float(out.get("open") or out.get("open_price")) or out["ltp"]
    out["high"] = _to_float(out.get("high") or out.get("high_price")) or max(out["open"], out["ltp"])
    out["low"] = _to_float(out.get("low") or out.get("low_price")) or min(out["open"], out["ltp"])
    out["spot_at_time"] = _to_float(out.get("spot_at_time") or out.get("spot") or out.get("underlying") or spot) or 0.0

    rows = []
    if isinstance(chain, dict):
        rows = chain.get("option_chain") or chain.get("rows") or []
    elif isinstance(chain, (list, tuple)):
        rows = chain
    strike = out.get("strike")
    ce_oi = pe_oi = ce_vol = pe_vol = 0.0
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        r_strike = _to_float(r.get("strike") or r.get("strike_price") or r.get("strikePrice") or r.get("StrikePrice"))
        if strike and r_strike and abs(float(r_strike) - float(strike)) > 1e-9:
            continue
        r_side = str(r.get("option_type") or r.get("ce_pe") or r.get("right") or r.get("side") or "").upper()
        if r_side in ("CALL", "C"):
            r_side = "CE"
        elif r_side in ("PUT", "P"):
            r_side = "PE"
        r_oi = _to_float(r.get("oi") or r.get("open_interest") or r.get("openInterest")) or 0.0
        r_vol = _to_float(r.get("volume") or r.get("traded_volume") or r.get("tradedVolume")) or 0.0
        if r_side == "CE":
            ce_oi += r_oi
            ce_vol += r_vol
        elif r_side == "PE":
            pe_oi += r_oi
            pe_vol += r_vol
    out["oi_CE"] = _to_float(out.get("oi_CE")) or ce_oi or (out["oi"] if opt_type == "CE" else 0.0)
    out["oi_PE"] = _to_float(out.get("oi_PE")) or pe_oi or (out["oi"] if opt_type == "PE" else 0.0)
    out["volume_CE"] = _to_float(out.get("volume_CE")) or ce_vol or (out["volume"] if opt_type == "CE" else 0.0)
    out["volume_PE"] = _to_float(out.get("volume_PE")) or pe_vol or (out["volume"] if opt_type == "PE" else 0.0)
    return out


def _latest_candle(candles: Any) -> Dict[str, Any]:
    if not isinstance(candles, (list, tuple)) or not candles:
        return {}
    c = candles[-1]
    if isinstance(c, dict):
        return dict(c)
    out = {}
    for k in ("open", "high", "low", "close", "volume", "time", "timestamp"):
        if hasattr(c, k):
            out[k] = getattr(c, k)
    return out


def _load_required_features(candidate: Dict[str, Any]) -> List[str]:
    feats = candidate.get("_feature_list") or candidate.get("features") or candidate.get("feature_schema") or []
    if isinstance(feats, dict):
        feats = feats.get("features") or feats.get("live_computable_features") or []
    return [str(f) for f in feats if f]


def _apply_feature_aliases(features: Dict[str, Any]) -> None:
    """Alias feature names without mixing underlying spot with option premium."""
    try:
        from paper_forward_spot_integrity import is_valid_nifty_underlying_spot
    except Exception:
        def is_valid_nifty_underlying_spot(_v: Any) -> bool:  # type: ignore[misc]
            return True

    authoritative_spot = None
    for key in ("spot", "underlying", "underlying_price", "ctx_spot"):
        val = features.get(key)
        if val not in (None, "") and is_valid_nifty_underlying_spot(val):
            authoritative_spot = float(val)
            break

    alias_groups = [
        ("ce_pe", "option_type", "right", "side"),
        ("close", "ltp", "last_price", "lastPrice", "option_ltp", "ctx_option_price", "traded_price", "last"),
        ("bid_price", "bid", "best_bid"),
        ("ask_price", "ask", "best_ask"),
        ("volume", "vol", "traded_volume", "tradedVolume"),
        ("open_interest", "openInterest", "oi"),
        ("dte", "days_to_expiry", "dte_days", "time_to_expiry"),
        ("strike", "strikePrice", "StrikePrice", "strike_price"),
        ("iv", "implied_volatility"),
    ]
    for group in alias_groups:
        value = next((features.get(k) for k in group if features.get(k) not in (None, "")), None)
        if value is None:
            continue
        for k in group:
            features.setdefault(k, value)

    if authoritative_spot is not None:
        for key in ("spot", "underlying", "underlying_price", "ctx_spot"):
            features[key] = authoritative_spot


def _parse_datetime_any(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        try:
            return datetime(int(value.year), int(value.month), int(value.day), tzinfo=timezone.utc)
        except Exception:
            return None
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d%b%Y", "%d%b%y", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _safe_series(candles: Any, key: str) -> List[float]:
    out: List[float] = []
    if not isinstance(candles, (list, tuple)):
        return out
    for c in candles:
        v = c.get(key) if isinstance(c, dict) else getattr(c, key, None)
        fv = _to_float(v)
        if fv is not None:
            out.append(fv)
    return out


def _rsi(values: List[float], period: int = 14) -> Optional[float]:
    if len(values) <= period:
        return None
    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
    recent = diffs[-period:]
    gains = [d for d in recent if d > 0]
    losses = [-d for d in recent if d < 0]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _std(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def build_paper_forward_feature_frame(candidate: Dict[str, Any], snapshot: Dict[str, Any]) -> Tuple[Any, List[str], Dict[str, Any]]:
    """Build one live feature row and return (DataFrame, missing_required, debug)."""
    import pandas as pd

    snap = dict(snapshot or {})
    chain = snap.get("option_chain") or snap.get("option_chain_snapshot") or snap.get("_option_chain_snapshot")
    spot = snap.get("spot") or snap.get("price") or snap.get("underlying") or snap.get("underlying_price") or snap.get("close")
    row = _first_option_row(chain, candidate, spot)
    candles = snap.get("candles") or []
    candle = _latest_candle(candles)
    required = _load_required_features(candidate)

    features: Dict[str, Any] = {}
    for src in (snap, row):
        for k, v in src.items():
            if k not in ("option_chain", "option_chain_snapshot", "_option_chain_snapshot", "candles") and v is not None:
                features[k] = v

    authoritative_spot = spot
    try:
        from paper_forward_spot_integrity import is_valid_nifty_underlying_spot, option_premium_from_row
        if authoritative_spot not in (None, "") and not is_valid_nifty_underlying_spot(authoritative_spot):
            authoritative_spot = None
        if row and authoritative_spot is None:
            for key in ("spot", "underlying_price", "underlying"):
                candidate = row.get(key)
                if candidate not in (None, "") and is_valid_nifty_underlying_spot(candidate):
                    authoritative_spot = candidate
                    break
        opt_premium = option_premium_from_row(row) if row else None
    except Exception:
        opt_premium = None

    if authoritative_spot not in (None, ""):
        features["spot"] = authoritative_spot
        features["underlying"] = authoritative_spot
        features["underlying_price"] = authoritative_spot
        features["ctx_spot"] = authoritative_spot
    _apply_feature_aliases(features)
    if authoritative_spot not in (None, ""):
        features["spot"] = authoritative_spot
        features["underlying"] = authoritative_spot
        features["underlying_price"] = authoritative_spot
        features["ctx_spot"] = authoritative_spot
    if opt_premium is not None and opt_premium > 0:
        features["ltp"] = opt_premium
        features["option_ltp"] = opt_premium
        features["ctx_option_price"] = opt_premium
        features.setdefault("last_price", opt_premium)
    strike = _to_float(features.get("strike"))
    spot_f = _to_float(features.get("spot"))
    if strike is not None and spot_f is not None:
        features["moneyness"] = (strike - spot_f) / max(abs(spot_f), 1e-9)
        features["abs_moneyness"] = abs(features["moneyness"])
        features["strike_price"] = strike
    bid = _to_float(features.get("bid") or features.get("bid_price"))
    ask = _to_float(features.get("ask") or features.get("ask_price"))
    ltp = _to_float(features.get("ltp") or features.get("last_price") or features.get("close"))
    if ltp is not None:
        features["ltp"] = ltp
        features["last_price"] = ltp
    if bid is not None and ask is not None:
        features["spread"] = ask - bid
        if ltp:
            features["spread_pct"] = (ask - bid) / max(abs(ltp), 1e-9)
    opt_type = str(features.get("option_type") or features.get("ce_pe") or features.get("right") or "").upper()
    if opt_type in ("CALL", "C"):
        opt_type = "CE"
    if opt_type in ("PUT", "P"):
        opt_type = "PE"
    if opt_type:
        features["option_type"] = opt_type
        features["ce_pe"] = opt_type
        features["side"] = opt_type
        features["is_call"] = 1 if opt_type == "CE" else 0
        features["is_put"] = 1 if opt_type == "PE" else 0
        features["option_type_ce"] = 1 if opt_type == "CE" else 0
        features["option_type_pe"] = 1 if opt_type == "PE" else 0
    expiry_dt = _parse_datetime_any(features.get("expiry") or features.get("expiry_date") or features.get("expDate"))
    ts_dt = _parse_datetime_any(snap.get("timestamp") or snap.get("last_tick"))
    now_dt = ts_dt or datetime.now(timezone.utc)
    if expiry_dt is not None:
        if expiry_dt.tzinfo is None:
            expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
        dte = max(0.0, (expiry_dt - now_dt).total_seconds() / 86400.0)
        features["dte"] = dte
        features["days_to_expiry"] = dte
        features["dte_days"] = dte
        features["time_to_expiry"] = dte
        features["expiry_weekday"] = expiry_dt.weekday()
        features["is_weekly"] = 1
    features["timestamp"] = snap.get("timestamp") or now_dt.isoformat()
    features["hour"] = now_dt.hour
    features["minute"] = now_dt.minute
    features["minute_of_day"] = now_dt.hour * 60 + now_dt.minute
    features["day_of_week"] = now_dt.weekday()
    features["weekday"] = now_dt.weekday()
    features["month"] = now_dt.month
    session_minutes = now_dt.hour * 60 + now_dt.minute
    features["ctx_time_sin"] = math.sin(2.0 * math.pi * session_minutes / 1440.0)
    features["ctx_time_cos"] = math.cos(2.0 * math.pi * session_minutes / 1440.0)
    features["is_opening_session"] = 1.0 if 9 * 60 + 15 <= session_minutes <= 10 * 60 + 30 else 0.0
    features["is_closing_session"] = 1.0 if session_minutes >= 15 * 60 else 0.0
    features["is_midday_lull"] = 1.0 if 11 * 60 + 30 <= session_minutes <= 13 * 60 + 30 else 0.0
    if candle:
        for k, v in candle.items():
            if k in ("open", "high", "low", "close"):
                features[k] = v
            elif k == "volume":
                features.setdefault("candle_volume", v)
            else:
                features.setdefault(k, v)
        opens = _safe_series(candles, "open")
        highs = _safe_series(candles, "high")
        lows = _safe_series(candles, "low")
        closes = _safe_series(candles, "close")
        vols = _safe_series(candles, "volume")
        if vols:
            features.setdefault("candle_volume", vols[-1])
        if len(closes) >= 2 and closes[-2]:
            returns = (closes[-1] - closes[-2]) / max(abs(closes[-2]), 1e-9)
            features.setdefault("returns", returns)
            features.setdefault("return_1", returns)
            features.setdefault("ret_1", features["returns"])
        if len(closes) >= 4 and closes[-4]:
            features.setdefault("return_3", (closes[-1] - closes[-4]) / max(abs(closes[-4]), 1e-9))
        if len(closes) >= 6:
            rets = []
            for i in range(max(1, len(closes) - 20), len(closes)):
                if closes[i - 1]:
                    rets.append((closes[i] - closes[i - 1]) / max(abs(closes[i - 1]), 1e-9))
            std = _std(rets)
            if std is not None:
                features.setdefault("ret_std", std)
                features.setdefault("realized_vol", std)
        if len(closes) >= 14:
            features.setdefault("sma_14", sum(closes[-14:]) / 14.0)
            features.setdefault("ema_14", features["sma_14"])
            rsi = _rsi(closes, 14)
            if rsi is not None:
                features.setdefault("rsi", rsi)
                features.setdefault("rsi_14", rsi)
        if highs and lows and closes:
            features.setdefault("candle_range", highs[-1] - lows[-1])
            if opens:
                features.setdefault("body", abs(closes[-1] - opens[-1]))
                features.setdefault("upper_wick", highs[-1] - max(opens[-1], closes[-1]))
                features.setdefault("lower_wick", min(opens[-1], closes[-1]) - lows[-1])
            if len(highs) >= 14 and len(lows) >= 14 and len(closes) >= 14:
                trs = []
                for i in range(max(1, len(closes) - 14), len(closes)):
                    trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
                if trs:
                    features.setdefault("atr", sum(trs) / len(trs))
                    if closes[-1]:
                        features.setdefault("atr_pct", features["atr"] / abs(closes[-1]))
                        features.setdefault("atr_14", features["atr"])
                if len(closes) >= 21:
                    features.setdefault("ema_fast", sum(closes[-9:]) / 9.0)
                    features.setdefault("ema_slow", sum(closes[-21:]) / 21.0)
                    if closes[-1]:
                        features.setdefault("ema_diff_pct", (features["ema_fast"] - features["ema_slow"]) / abs(closes[-1]))
            if closes[-1]:
                features.setdefault("range_pct", (highs[-1] - lows[-1]) / abs(closes[-1]))
                if opens:
                    features.setdefault("body_pct", abs(closes[-1] - opens[-1]) / abs(closes[-1]))
                    features.setdefault("close_vs_open_pct", (closes[-1] - opens[-1]) / abs(opens[-1] or closes[-1]))
                    features.setdefault("upper_wick_pct", (highs[-1] - max(opens[-1], closes[-1])) / abs(closes[-1]))
                    features.setdefault("lower_wick_pct", (min(opens[-1], closes[-1]) - lows[-1]) / abs(closes[-1]))
                    features.setdefault("close_location_pct", (closes[-1] - lows[-1]) / max(highs[-1] - lows[-1], 1e-9))
            features.setdefault("last_open", opens[-1] if opens else None)
            features.setdefault("last_high", highs[-1] if highs else None)
            features.setdefault("last_low", lows[-1] if lows else None)
            features.setdefault("last_close", closes[-1] if closes else None)
            features.setdefault("last_volume", vols[-1] if vols else features.get("candle_volume"))
            features.setdefault("spot_close", closes[-1] if closes else spot)
            features.setdefault("open_spot", opens[-1] if opens else spot)
            features.setdefault("high_spot", highs[-1] if highs else spot)
            features.setdefault("low_spot", lows[-1] if lows else spot)
            features.setdefault("volume_spot", vols[-1] if vols else 0.0)
            features.setdefault("weekday_spot", features["weekday"])
            features.setdefault("spot_range_pct", features.get("range_pct"))
            features.setdefault("spot_atr", features.get("atr_14") or features.get("atr"))
            features.setdefault("spot_rsi", features.get("rsi_14") or features.get("rsi"))
            if vols and sum(vols) > 0 and highs and lows and closes:
                pv = [((h + l + c) / 3.0) * v for h, l, c, v in zip(highs, lows, closes, vols)]
                features.setdefault("spot_vwap", sum(pv) / sum(vols))
    _apply_feature_aliases(features)

    live_feature_error = ""
    try:
        from live_feature_builder import LiveSnapshot, build_live_features

        live_candles = []
        for c in (candles if isinstance(candles, (list, tuple)) else []):
            cc = _coerce_live_builder_candle(c)
            if cc is not None:
                live_candles.append(cc)
        live_option = _normalize_option_row_for_live_builder(row, chain, spot)
        live_spot = _to_float(spot) or _to_float(features.get("spot")) or _to_float(features.get("underlying_price"))
        live_atm_iv = _to_float(features.get("final_iv") or features.get("bs_iv") or features.get("iv"))
        live_result = build_live_features(
            LiveSnapshot(candles=live_candles, spot=live_spot, atm_iv=live_atm_iv, option_chain=live_option, timestamp=now_dt),
            required_features=required or None,
        )
        for k, v in (live_result.features or {}).items():
            if v is not None:
                existing = features.get(k)
                if isinstance(v, (int, float)) and float(v) == 0.0 and existing not in (None, "", 0, 0.0):
                    continue
                features[k] = v
        features["_live_feature_coverage_pct"] = live_result.coverage_pct
    except Exception as exc:
        live_feature_error = str(exc)

    _apply_feature_aliases(features)

    # === BLOCKER 3: live-computable cost/liquidity/percentile features ===
    # These are required by cost-aware edge models but missing from basic live builder in paper-forward context.
    # Compute using full 'chain' (available here) + current row values. Approx for live.
    cid = (candidate or {}).get("candidate_id", "unknown")
    try:
        cost_names = [
            "estimated_cost_bps", "spread_cost_component", "slippage_cost_component",
            "brokerage_or_fee_component", "cost_pct_of_premium",
            "spread_pctile_day", "volume_pctile_day", "oi_pctile_day",
            "premium_pctile_expiry", "recent_vol_proxy"
        ]
        full_rows: List[Dict[str, Any]] = []
        if isinstance(chain, (list, tuple)):
            full_rows = [r for r in chain if isinstance(r, dict)]
        elif isinstance(chain, dict):
            full_rows = [r for r in (chain.get("option_chain") or chain.get("rows") or []) if isinstance(r, dict)]
        ltp_f = _to_float(features.get("ltp") or features.get("last_price") or features.get("close"))
        bid_f = _to_float(features.get("bid") or features.get("bid_price"))
        ask_f = _to_float(features.get("ask") or features.get("ask_price"))
        spot_f = _to_float(features.get("spot") or features.get("underlying_price") or features.get("price"))
        vol_f = _to_float(features.get("volume") or features.get("traded_volume"))
        oi_f = _to_float(features.get("oi") or features.get("open_interest"))
        prem_f = ltp_f
        exp_key = features.get("expiry") or features.get("expDate") or features.get("expiry_date")
        # costs
        default_fee_bps = float(os.getenv("PAPER_FORWARD_DEFAULT_FEE_BPS", "1.5"))
        spread_pct = None
        mid_f = ((bid_f + ask_f) / 2.0) if bid_f is not None and ask_f is not None else ltp_f
        if bid_f is not None and ask_f is not None and mid_f and mid_f > 0:
            spread_pct = max(ask_f - bid_f, 0.0) / max(mid_f, 0.05)
        if spread_pct is not None:
            half = spread_pct / 2.0
            features["spread_cost_component"] = half
            liq_pen = 0.0005 if (vol_f or 0) < 5000 else 0.0
            features["slippage_cost_component"] = half + liq_pen
            fee_p = default_fee_bps / 10000.0
            features["brokerage_or_fee_component"] = fee_p
            est_p = half + fee_p + features["slippage_cost_component"]
            features["estimated_cost_bps"] = est_p * 10000.0
            features["cost_pct_of_premium"] = est_p
            print(f"[COST-FEATURE] using_default_fee_bps={default_fee_bps} candidate_id={cid}")
        default_spread_bps = float(os.getenv("PAPER_FORWARD_DEFAULT_SPREAD_BPS", "10.0"))
        default_slippage_bps = float(os.getenv("PAPER_FORWARD_DEFAULT_SLIPPAGE_BPS", "5.0"))
        if features.get("spread_cost_component") in (None, ""):
            features["spread_cost_component"] = max(default_spread_bps, 0.0) / 10000.0
        if features.get("slippage_cost_component") in (None, ""):
            features["slippage_cost_component"] = max(default_slippage_bps, 0.0) / 10000.0
        if features.get("brokerage_or_fee_component") in (None, ""):
            features["brokerage_or_fee_component"] = default_fee_bps / 10000.0
        est_p = (
            float(features.get("spread_cost_component") or 0.0)
            + float(features.get("slippage_cost_component") or 0.0)
            + float(features.get("brokerage_or_fee_component") or 0.0)
        )
        features["estimated_cost_bps"] = float(features.get("estimated_cost_bps") or (est_p * 10000.0))
        features["cost_pct_of_premium"] = float(features.get("cost_pct_of_premium") or est_p)
        # pctiles
        def _p_rank(lst: List[float], cur: float) -> float:
            lst = [float(x) for x in lst if x is not None]
            if not lst or cur is None:
                return 0.5
            s = sorted(lst)
            k = sum(1 for x in s if x <= float(cur))
            return k / len(s)
        sp_l, v_l, o_l, p_l = [], [], [], []
        for r in full_rows:
            rb = _to_float(r.get("bid") or r.get("bid_price"))
            ra = _to_float(r.get("ask") or r.get("ask_price"))
            rl = _to_float(r.get("ltp") or r.get("last_price") or r.get("close"))
            rv = _to_float(r.get("volume") or r.get("traded_volume"))
            ro = _to_float(r.get("oi") or r.get("open_interest"))
            rp = rl
            re = r.get("expiry") or r.get("expDate") or r.get("expiry_date")
            if rb and ra and rl and rl > 0:
                sp_l.append((ra - rb) / rl)
            if rv is not None: v_l.append(rv)
            if ro is not None: o_l.append(ro)
            if (not exp_key or str(re) == str(exp_key)) and rp is not None:
                p_l.append(rp)
            elif not exp_key and rp is not None:
                p_l.append(rp)
        features["spread_pctile_day"] = _p_rank(sp_l, spread_pct)
        features["volume_pctile_day"] = _p_rank(v_l, vol_f)
        features["oi_pctile_day"] = _p_rank(o_l, oi_f)
        features["premium_pctile_expiry"] = _p_rank(p_l, prem_f)
        # vol proxy
        rv_p = features.get("realized_vol") or features.get("ret_std") or features.get("atr_pct")
        if rv_p is None:
            atrv = features.get("atr") or features.get("atr_14")
            if spot_f and atrv:
                rv_p = atrv / max(spot_f, 1e-6)
            else:
                rv_p = 0.0
            print(f"[COST-FEATURE] recent_vol_proxy fallback used candidate_id={cid}")
        features["recent_vol_proxy"] = rv_p
        comp = [cn for cn in cost_names if cn in features and features.get(cn) is not None]
        miss_a = [cn for cn in (required or []) if cn in cost_names and (cn not in features or features.get(cn) is None)]
        _cq_early, _af_early = compute_cost_quality(
            row=row if isinstance(row, dict) else {},
            features=features,
            used_default_fee=True,
            used_default_spread=spread_pct is None,
            used_chain_pctile=bool(full_rows),
        )
        features["cost_quality"] = _cq_early
        features["cost_approx_flags"] = _af_early
        print(
            f"[LIVE-COST-FEATURES] candidate_id={cid} computed={comp} missing_after={miss_a} "
            f"approx_flags={_af_early} cost_quality={_cq_early}"
        )
    except Exception as _cost_exc:
        print(f"[LIVE-COST-FEATURES] candidate_id={cid} error={_cost_exc}")

    _apply_feature_aliases(features)
    ltp_f = _to_float(features.get("ltp") or features.get("last_price") or features.get("close"))
    spot_f = _to_float(features.get("spot") or features.get("underlying_price") or spot)
    strike_f = _to_float(features.get("strike_price") or features.get("strike"))
    if features.get("spot_vwap") in (None, "") and spot_f:
        features["spot_vwap"] = _to_float(features.get("spot_close") or features.get("close")) or spot_f
    if ltp_f is not None and spot_f:
        features.setdefault("ctx_option_price", ltp_f)
        features.setdefault("option_to_spot_pct", ltp_f / abs(spot_f))
    if strike_f is not None and spot_f:
        dist = (strike_f - spot_f) / abs(spot_f)
        features.setdefault("distance_from_spot", abs(dist))
        features.setdefault("atm_distance", dist)
        features.setdefault("strike_distance_pct", abs(dist))
        opt_type = str(features.get("option_type") or "").upper()
        if opt_type == "CE":
            intrinsic = max(0.0, spot_f - strike_f)
        elif opt_type == "PE":
            intrinsic = max(0.0, strike_f - spot_f)
        else:
            intrinsic = 0.0
        if ltp_f is not None:
            features.setdefault("intrinsic_value", intrinsic)
            features.setdefault("extrinsic_value", max(0.0, ltp_f - intrinsic))
    for src, dest in (("iv", "final_iv"), ("final_iv", "bs_iv"), ("delta", "bs_delta"), ("gamma", "bs_gamma"), ("theta", "bs_theta"), ("vega", "bs_vega")):
        if features.get(dest) in (None, "") and features.get(src) not in (None, ""):
            features[dest] = features[src]
    opt_type_for_greeks = str(features.get("option_type") or "").upper()
    dte_years = _to_float(features.get("time_to_expiry_years"))
    if dte_years is None and features.get("dte_days") is not None:
        dte_years = float(features.get("dte_days") or 0.0) / 365.0
    iv_f = _to_float(features.get("final_iv") or features.get("bs_iv") or features.get("iv"))
    if (iv_f is None or iv_f <= 0) and all(v is not None for v in (ltp_f, spot_f, strike_f, dte_years)) and opt_type_for_greeks in ("CE", "PE"):
        try:
            from greeks import implied_volatility
            iv_f = implied_volatility(float(ltp_f), float(spot_f), float(strike_f), max(float(dte_years), 1e-9), 0.065, opt_type_for_greeks)
        except Exception:
            iv_f = None
    if (iv_f is None or iv_f <= 0) and all(v is not None for v in (spot_f, strike_f, dte_years)) and opt_type_for_greeks in ("CE", "PE"):
        iv_f = 0.18
    if iv_f is not None and iv_f > 0:
        if features.get("final_iv") in (None, "", 0, 0.0):
            features["final_iv"] = iv_f
        if features.get("bs_iv") in (None, "", 0, 0.0):
            features["bs_iv"] = iv_f
    if all(v is not None for v in (spot_f, strike_f, dte_years, iv_f)) and opt_type_for_greeks in ("CE", "PE"):
        try:
            from greeks import delta as bs_delta, gamma as bs_gamma, theta as bs_theta, vega as bs_vega
            t_years = max(float(dte_years), 1e-9)
            if features.get("bs_delta") in (None, ""):
                features["bs_delta"] = bs_delta(float(spot_f), float(strike_f), t_years, 0.065, float(iv_f), opt_type_for_greeks)
            if features.get("bs_gamma") in (None, ""):
                features["bs_gamma"] = bs_gamma(float(spot_f), float(strike_f), t_years, 0.065, float(iv_f))
            if features.get("bs_theta") in (None, ""):
                features["bs_theta"] = bs_theta(float(spot_f), float(strike_f), t_years, 0.065, float(iv_f), opt_type_for_greeks) / 365.0
            if features.get("bs_vega") in (None, ""):
                features["bs_vega"] = bs_vega(float(spot_f), float(strike_f), t_years, 0.065, float(iv_f))
        except Exception:
            pass
    if "bs_rho" not in features:
        features["bs_rho"] = 0.0
    if "greeks_quality_score" not in features:
        greeks_available = any(features.get(k) not in (None, "", 0, 0.0) for k in ("bs_delta", "bs_gamma", "bs_theta", "bs_vega", "final_iv"))
        features["greeks_quality_score"] = 1.0 if greeks_available else 0.0
    if strike_f is not None and spot_f:
        features.setdefault("log_moneyness", math.log(max(strike_f, 1e-9) / max(spot_f, 1e-9)))
        features.setdefault("distance_from_atm", abs(strike_f - spot_f))
        features.setdefault("distance_from_atm_pct", abs(strike_f - spot_f) / abs(spot_f))
    if "time_to_expiry_days" not in features and features.get("dte_days") is not None:
        features["time_to_expiry_days"] = features.get("dte_days")
    if "time_to_expiry_years" not in features and features.get("dte_days") is not None:
        features["time_to_expiry_years"] = float(features.get("dte_days") or 0.0) / 365.0
    if "bid_ask_spread" not in features:
        bid_f = _to_float(features.get("bid") or features.get("bid_price"))
        ask_f = _to_float(features.get("ask") or features.get("ask_price"))
        if bid_f is not None and ask_f is not None:
            features["bid_ask_spread"] = max(0.0, ask_f - bid_f)
    if features.get("bid_ask_spread") in (None, "") and ltp_f is not None:
        features["bid_ask_spread"] = 0.0
    if "bid_ask_spread_pct" not in features:
        spread_f = _to_float(features.get("bid_ask_spread") or features.get("spread"))
        if spread_f is not None and ltp_f:
            features["bid_ask_spread_pct"] = spread_f / abs(ltp_f)
    if features.get("mid_price") in (None, ""):
        bid_f = _to_float(features.get("bid") or features.get("bid_price"))
        ask_f = _to_float(features.get("ask") or features.get("ask_price"))
        if bid_f is not None and ask_f is not None and ask_f >= bid_f:
            features["mid_price"] = (bid_f + ask_f) / 2.0
        elif ltp_f is not None:
            features["mid_price"] = ltp_f
    if features.get("ltp_vs_mid_diff") in (None, "") and ltp_f is not None and features.get("mid_price") is not None:
        features["ltp_vs_mid_diff"] = ltp_f - float(features["mid_price"])
    if features.get("ltp_vs_mid_diff_pct") in (None, "") and features.get("ltp_vs_mid_diff") is not None and features.get("mid_price"):
        features["ltp_vs_mid_diff_pct"] = float(features["ltp_vs_mid_diff"]) / abs(float(features["mid_price"]))
    features.setdefault("ctx_adx", features.get("adx_14", 0.0) or 0.0)
    features.setdefault("ctx_choppiness", features.get("choppiness_14", 0.0) or 0.0)
    features.setdefault("ctx_volume_sma", features.get("vol_mean", features.get("volume", 0.0)) or 0.0)
    features.setdefault("oi_z_5", 0.0)
    features.setdefault("volume_z_5", 0.0)

    # === Complete live cost/liquidity/chain-agg features for valid candidate (BLOCKER 2) ===
    # Compute late to have all base features (bid/ask from live/manual, full_rows)
    cid = (candidate or {}).get("candidate_id", "unknown")
    try:
        full_rows = []
        if isinstance(chain, (list, tuple)):
            full_rows = [r for r in chain if isinstance(r, dict)]
        elif isinstance(chain, dict):
            full_rows = [r for r in (chain.get("option_chain") or chain.get("rows") or []) if isinstance(r, dict)]
        ltp_f = _to_float(features.get("ltp") or features.get("last_price") or features.get("close") or features.get("option_ltp"))
        bid_f = _to_float(features.get("bid") or features.get("bid_price") or features.get("best_bid"))
        ask_f = _to_float(features.get("ask") or features.get("ask_price") or features.get("best_ask"))
        spot_f = _to_float(features.get("spot") or features.get("underlying_price") or features.get("price"))
        vol_f = _to_float(features.get("volume") or features.get("traded_volume"))
        oi_f = _to_float(features.get("oi") or features.get("open_interest"))
        spread_pct = _to_float(features.get("bid_ask_spread_pct") or features.get("option_bid_ask_spread_pct") or features.get("spread_pct"))
        mid_f = ((bid_f + ask_f) / 2.0) if bid_f is not None and ask_f is not None else ltp_f
        if spread_pct is None and bid_f is not None and ask_f is not None and mid_f and mid_f > 0:
            spread_pct = max(ask_f - bid_f, 0.0) / max(mid_f, 0.05)
        default_fee_bps = float(os.getenv("PAPER_FORWARD_DEFAULT_FEE_BPS", "2.0"))
        # A-E cost components
        if spread_pct is not None:
            spread_c = float(spread_pct)
            features["spread_cost_component"] = spread_c
            liq_pen = 0.0005 if (vol_f or 0) < 5000 else 0.0
            slip_c = 0.5 * spread_c + liq_pen
            features["slippage_cost_component"] = slip_c
            fee_c = default_fee_bps / 10000.0
            features["brokerage_or_fee_component"] = fee_c
            est_p = spread_c + slip_c + fee_c
            features["estimated_cost_bps"] = est_p * 10000.0
            features["cost_pct_of_premium"] = est_p
        else:
            # safe defaults to avoid missing
            default_spread_bps = float(os.getenv("PAPER_FORWARD_DEFAULT_SPREAD_BPS", "10.0"))
            default_slippage_bps = float(os.getenv("PAPER_FORWARD_DEFAULT_SLIPPAGE_BPS", "5.0"))
            features.setdefault("spread_cost_component", default_spread_bps / 10000.0)
            features.setdefault("slippage_cost_component", default_slippage_bps / 10000.0)
            features.setdefault("brokerage_or_fee_component", default_fee_bps / 10000.0)
            est_p = float(features["spread_cost_component"]) + float(features["slippage_cost_component"]) + float(features["brokerage_or_fee_component"])
            features.setdefault("estimated_cost_bps", est_p * 10000)
            features.setdefault("cost_pct_of_premium", est_p)
        for cost_key in ("spread_cost_component", "slippage_cost_component", "brokerage_or_fee_component", "estimated_cost_bps", "cost_pct_of_premium"):
            if features.get(cost_key) in (None, ""):
                if cost_key == "estimated_cost_bps":
                    features[cost_key] = est_p * 10000.0
                elif cost_key == "cost_pct_of_premium":
                    features[cost_key] = est_p
                elif cost_key == "brokerage_or_fee_component":
                    features[cost_key] = default_fee_bps / 10000.0
                elif cost_key == "slippage_cost_component":
                    features[cost_key] = float(os.getenv("PAPER_FORWARD_DEFAULT_SLIPPAGE_BPS", "5.0")) / 10000.0
                else:
                    features[cost_key] = float(os.getenv("PAPER_FORWARD_DEFAULT_SPREAD_BPS", "10.0")) / 10000.0
        # F-J chain aggregates / rank / regime (use full_rows)
        ce_vol = 0.0
        pe_vol = 0.0
        dists = []
        curr_dist = None
        for r in full_rows:
            ot = str(r.get("option_type") or r.get("ce_pe") or r.get("right") or "").upper()
            if ot in ("CALL", "C"): ot = "CE"
            elif ot in ("PUT", "P"): ot = "PE"
            v = _to_float(r.get("volume") or r.get("traded_volume") or r.get("volume_CE") or r.get("volume_PE") or 0)
            if ot == "CE": ce_vol += v or 0
            elif ot == "PE": pe_vol += v or 0
            # dist
            stk = _to_float(r.get("strike") or r.get("strike_price"))
            if stk is not None and spot_f:
                d = abs(stk - spot_f)
                dists.append(d)
                if r is row or (abs(stk - _to_float(row.get("strike") if isinstance(row,dict) else 0)) < 1e-6 if row else False):
                    curr_dist = d
        features["ce_vol_day"] = ce_vol
        features["pe_vol_day"] = pe_vol
        tot_v = ce_vol + pe_vol
        features["ce_pe_vol_imbalance"] = (ce_vol - pe_vol) / max(tot_v, 1e-9) if tot_v else 0.0
        # atm dist rank (percentile of dist; low dist low rank)
        if curr_dist is None and spot_f and _to_float(features.get("strike_price") or features.get("strike")):
            curr_dist = abs( _to_float(features.get("strike_price") or features.get("strike")) - spot_f )
        if dists and curr_dist is not None:
            s_d = sorted(dists)
            k = sum(1 for d in s_d if d <= curr_dist)
            features["atm_distance_rank_day"] = k / len(s_d)
        else:
            features.setdefault("atm_distance_rank_day", 0.5)
        # liq regime numeric 0/1/2
        sp_p = spread_pct or features.get("spread_pct", 0.02) or 0.02
        v_tot = sum(_to_float(r.get("volume") or 0) for r in full_rows) or (vol_f or 0)
        o_tot = sum(_to_float(r.get("oi") or 0) for r in full_rows) or (oi_f or 0)
        if sp_p < 0.01 and v_tot > 50000 and o_tot > 200000:
            lreg = 2.0
        elif sp_p < 0.03 and v_tot > 5000:
            lreg = 1.0
        else:
            lreg = 0.0
        features["liq_regime_day"] = lreg

        # BLOCKER 2: add spread_regime_day and ce_pe_rel_strength
        try:
            sp_pct = features.get("spread_pct") or features.get("bid_ask_spread_pct") or features.get("option_bid_ask_spread_pct")
            if sp_pct is None and bid_f is not None and ask_f is not None and ltp_f and ltp_f > 0:
                sp_pct = (ask_f - bid_f) / ltp_f
            if sp_pct is not None:
                if sp_pct <= 0.001:
                    features["spread_regime_day"] = 0.0
                elif sp_pct <= 0.003:
                    features["spread_regime_day"] = 1.0
                else:
                    features["spread_regime_day"] = 2.0
            else:
                features["spread_regime_day"] = 1.0  # neutral fallback

            # ce_pe_rel_strength from vols or oi
            ce_v = _to_float_or_zero(features.get("volume_CE"))
            pe_v = _to_float_or_zero(features.get("volume_PE"))
            ce_o = _to_float_or_zero(features.get("oi_CE"))
            pe_o = _to_float_or_zero(features.get("oi_PE"))
            if ce_v + pe_v > 0:
                features["ce_pe_rel_strength"] = (ce_v - pe_v) / max(ce_v + pe_v, 1e-9)
            elif ce_o + pe_o > 0:
                features["ce_pe_rel_strength"] = (ce_o - pe_o) / max(ce_o + pe_o, 1e-9)
            else:
                features["ce_pe_rel_strength"] = 0.0
        except Exception:
            features.setdefault("spread_regime_day", 1.0)
            features.setdefault("ce_pe_rel_strength", 0.0)

        added_count = sum(
            1 for k in (
                "estimated_cost_bps", "spread_cost_component", "slippage_cost_component",
                "brokerage_or_fee_component", "cost_pct_of_premium", "ce_vol_day", "pe_vol_day",
                "ce_pe_vol_imbalance", "atm_distance_rank_day", "liq_regime_day",
                "spread_regime_day", "ce_pe_rel_strength",
            ) if k in features
        )
        pf_log(
            "DEBUG",
            f"[LIVE-FEATURE-FILL] cid={cid} added_count={added_count}",
            rate_key=f"live_feature_fill:{cid}",
            rate_interval=60.0,
        )
    except Exception as _fexc:
        pf_log(
            "WARN",
            f"[LIVE-FEATURE-FILL] cid={cid} error={_fexc}",
            rate_key=f"live_feature_fill_err:{cid}",
            rate_interval=60.0,
        )

    # BLOCKER 2: ensure spread_regime_day and ce_pe_rel_strength computed before required check and feature_contract
    sp_pct = _to_float(features.get("spread_pct") or features.get("bid_ask_spread_pct") or features.get("option_bid_ask_spread_pct"))
    if sp_pct is None:
        bid = _to_float(features.get("bid") or features.get("bid_price"))
        ask = _to_float(features.get("ask") or features.get("ask_price"))
        ltp = _to_float(features.get("ltp") or features.get("last_price") or features.get("close"))
        if bid and ask and ltp and ltp > 0:
            sp_pct = (ask - bid) / ltp
    if sp_pct is not None:
        if sp_pct <= 0.001:
            features["spread_regime_day"] = 0.0
        elif sp_pct <= 0.003:
            features["spread_regime_day"] = 1.0
        else:
            features["spread_regime_day"] = 2.0
    else:
        features["spread_regime_day"] = 1.0
    ce_v = _to_float_or_zero(features.get("volume_CE"))
    pe_v = _to_float_or_zero(features.get("volume_PE"))
    ce_o = _to_float_or_zero(features.get("oi_CE"))
    pe_o = _to_float_or_zero(features.get("oi_PE"))
    if (ce_v + pe_v) > 0:
        features["ce_pe_rel_strength"] = (ce_v - pe_v) / max(ce_v + pe_v, 1e-9)
    elif (ce_o + pe_o) > 0:
        features["ce_pe_rel_strength"] = (ce_o - pe_o) / max(ce_o + pe_o, 1e-9)
    else:
        features["ce_pe_rel_strength"] = 0.0
    pf_log(
        "DEBUG",
        f"[LIVE-FEATURE-FILL] spread_regime_day={features.get('spread_regime_day')} "
        f"ce_pe_rel_strength={features.get('ce_pe_rel_strength')}",
        rate_key="live_feature_fill",
        rate_interval=60.0,
    )

    spread_from_broker = (
        _to_float(features.get("bid") or features.get("bid_price")) is not None
        and _to_float(features.get("ask") or features.get("ask_price")) is not None
        and not _row_is_synthetic(row if isinstance(row, dict) else None)
    )
    used_default_spread = spread_pct is None if "spread_pct" in locals() else not spread_from_broker
    used_default_fee = True
    _chain_rows = full_rows if "full_rows" in locals() else (
        [r for r in chain if isinstance(r, dict)] if isinstance(chain, (list, tuple)) else []
    )
    used_chain_pctile = bool(_chain_rows) and any(
        k in features for k in ("spread_pctile_day", "volume_pctile_day", "oi_pctile_day", "premium_pctile_expiry")
    )
    cq, approx_flags = compute_cost_quality(
        row=row if isinstance(row, dict) else {},
        features=features,
        used_default_fee=used_default_fee,
        used_default_spread=used_default_spread,
        used_chain_pctile=used_chain_pctile,
    )
    features["cost_quality"] = cq
    features["cost_approx_flags"] = approx_flags

    def _feat_missing(name: str) -> bool:
        val = features.get(name)
        if val is None:
            return True
        try:
            if isinstance(val, float) and math.isnan(val):
                return True
        except Exception:
            pass
        return False

    if features.get("volume_ratio") in (None, "") and features.get("vol_mean") not in (None, "", 0, 0.0):
        last_vol = _to_float(features.get("last_volume") or features.get("volume"))
        vol_mean = _to_float(features.get("vol_mean"))
        if last_vol is not None and vol_mean:
            features["volume_ratio"] = last_vol / max(vol_mean, 1e-9)

    missing = [f for f in required if _feat_missing(f)]
    df = pd.DataFrame([{k: v for k, v in features.items() if not isinstance(v, (list, dict, tuple))}])
    non_null_count = int(df.notna().sum(axis=1).iloc[0]) if not df.empty else 0
    nan_count = int(df.isna().sum(axis=1).iloc[0]) if not df.empty else 0
    zero_count = 0
    if not df.empty:
        for v in df.iloc[0].tolist():
            try:
                if float(v) == 0.0:
                    zero_count += 1
            except Exception:
                pass
    return df, missing, {
        "required_features_count": len(required),
        "available_features_count": len(df.columns),
        "available_features": list(df.columns),
        "missing_features": missing,
        "option_rows": _count_option_rows(chain),
        "candles_count": len(candles) if isinstance(candles, (list, tuple)) else 0,
        "feature_non_null_count": non_null_count,
        "feature_nan_count": nan_count,
        "feature_zero_count": zero_count,
        "feature_order": required,
        "live_feature_error": live_feature_error,
        "coverage_pct": round(((len(required) - len(missing)) / max(len(required), 1)) * 100.0, 2) if required else 100.0,
        "cost_quality": features.get("cost_quality", "APPROX"),
        "cost_approx_flags": features.get("cost_approx_flags", []),
    }


def _row_is_synthetic(row: Optional[Dict[str, Any]]) -> bool:
    if not row:
        return False
    return bool(row.get("synthetic")) or str(row.get("chain_source") or "") == "BLACK_SCHOLES_SYNTHETIC"


def load_candidate_registry(project_root: Path | str = REPO_ROOT) -> Dict[str, Dict[str, Any]]:
    """Legacy + new: also loads the canonical paper_forward_artifact_registry.json if present."""
    root = Path(project_root).resolve()
    reg: Dict[str, Dict[str, Any]] = {}

    # New canonical registry (preferred)
    canon = root / "config" / "paper_forward_artifact_registry.json"
    if canon.exists():
        try:
            data = json.loads(canon.read_text(encoding="utf-8"))
            for cid, entry in (data.get("candidates") or {}).items():
                reg[cid] = {
                    "candidate_id": cid,
                    "artifact_dir": entry.get("resolved_dir") or entry.get("explicit_artifact_dir"),
                    "model_path": entry.get("model_path"),
                    "feature_list_path": entry.get("feature_list_path"),
                    "load_status": entry.get("load_status"),
                    "source": "artifact_registry",
                }
        except Exception:
            pass

    # Existing report + disk scan logic (append/override)
    report_globs = [
        "reports/best_paper_forward_selection*.json",
        "reports/paper_forward_candidate_selection*.json",
        "reports/paper_forward_candidates_full*.json",
        "reports/*persist*paper*.json",
        "reports/*selection*.json",
        "reports/deployment*.json",
        "reports/*candidate*.json",
    ]
    for g in report_globs:
        for rp in root.glob(g):
            try:
                data = json.loads(rp.read_text(encoding="utf-8"))
                items = []
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = data.get("candidates", data.get("selected", data.get("paper_forward_only", []))) or []
                for c in items:
                    if not isinstance(c, dict):
                        continue
                    cid = c.get("candidate_id") or c.get("id")
                    if not cid:
                        continue
                    ad = c.get("artifact_dir") or c.get("candidate_dir") or c.get("path") or c.get("model_dir")
                    entry = reg.setdefault(cid, {"candidate_id": cid, "sources": []})
                    if ad:
                        entry["artifact_dir"] = ad
                    entry["sources"].append(str(rp.name))
                    for k in ("model_name", "preset_family", "side_policy", "paper_forward_only"):
                        if k in c and k not in entry:
                            entry[k] = c[k]
            except Exception:
                continue

    # On-disk scan (existing)
    art_base = root / "artifacts" / "candidates"
    if art_base.exists():
        for d in sorted(art_base.iterdir()):
            if not d.is_dir():
                continue
            prof = d / "candidate_profile.json"
            man = d / "candidate_manifest.json"
            pman = d / "paper_forward_manifest.json"
            has_core = prof.exists() or man.exists() or pman.exists()
            model_p = (d / "model.pkl").exists() or bool(list(d.glob("*.pkl")))
            if has_core or model_p:
                cid = d.name
                for fp in (prof, man, pman):
                    if fp.exists():
                        try:
                            meta = json.loads(fp.read_text(encoding="utf-8"))
                            cid = meta.get("candidate_id") or meta.get("id") or cid
                            break
                        except Exception:
                            pass
                entry = reg.setdefault(cid, {"candidate_id": cid, "sources": ["disk_scan"]})
                if "artifact_dir" not in entry or not Path(entry.get("artifact_dir", "")).exists():
                    entry["artifact_dir"] = str(d.relative_to(root))
                entry["has_profile"] = prof.exists()
                entry["has_manifest"] = man.exists()
                entry["has_model_pkl"] = model_p
    return reg
    """Build cid -> artifact info map from reports selection/persist outputs + on-disk scan.
    Backward compatible fallback when paper_forward_candidates.json has ids but bad/missing paths.
    """
    root = Path(project_root).resolve()
    reg: Dict[str, Dict[str, Any]] = {}
    report_globs = [
        "reports/best_paper_forward_selection*.json",
        "reports/paper_forward_candidate_selection*.json",
        "reports/paper_forward_candidates_full*.json",
        "reports/*persist*paper*.json",
        "reports/*selection*.json",
        "reports/deployment*.json",
        "reports/*candidate*.json",
    ]
    for g in report_globs:
        for rp in root.glob(g):
            try:
                data = json.loads(rp.read_text(encoding="utf-8"))
                items = []
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = data.get("candidates", data.get("selected", data.get("paper_forward_only", []))) or []
                for c in items:
                    if not isinstance(c, dict):
                        continue
                    cid = c.get("candidate_id") or c.get("id")
                    if not cid:
                        continue
                    ad = c.get("artifact_dir") or c.get("candidate_dir") or c.get("path") or c.get("model_dir")
                    entry = reg.setdefault(cid, {"candidate_id": cid, "sources": []})
                    if ad:
                        entry["artifact_dir"] = ad
                    entry["sources"].append(str(rp.name))
                    # carry useful meta
                    for k in ("model_name", "preset_family", "side_policy", "paper_forward_only"):
                        if k in c and k not in entry:
                            entry[k] = c[k]
            except Exception:
                continue
    # On-disk scan of artifacts/candidates for any that expose profile/manifest + model
    art_base = root / "artifacts" / "candidates"
    if art_base.exists():
        for d in sorted(art_base.iterdir()):
            if not d.is_dir():
                continue
            prof = d / "candidate_profile.json"
            man = d / "candidate_manifest.json"
            pman = d / "paper_forward_manifest.json"
            has_core = prof.exists() or man.exists() or pman.exists()
            model_p = (d / "model.pkl").exists() or any((d / n).exists() for n in ("model.pkl",) ) or bool(list(d.glob("*.pkl")))
            if has_core or model_p:
                # read cid from files if possible
                cid = d.name
                for fp in (prof, man, pman):
                    if fp.exists():
                        try:
                            meta = json.loads(fp.read_text(encoding="utf-8"))
                            cid = meta.get("candidate_id") or meta.get("id") or cid
                            break
                        except Exception:
                            pass
                entry = reg.setdefault(cid, {"candidate_id": cid, "sources": ["disk_scan"]})
                if "artifact_dir" not in entry or not Path(entry.get("artifact_dir", "")).exists():
                    entry["artifact_dir"] = str(d.relative_to(root))
                entry["has_profile"] = prof.exists()
                entry["has_manifest"] = man.exists()
                entry["has_model_pkl"] = model_p
    return reg


def _candidate_file_exists(p: Path) -> bool:
    try:
        return p.exists() and p.is_file()
    except Exception:
        return False


def _normalize_cid_base(cid: str) -> str:
    from candidate_artifact_resolver import _normalize_cid_base as _norm_base

    return _norm_base(cid)


def resolve_candidate_artifacts(
    candidate: Dict[str, Any],
    project_root: Path | str = REPO_ROOT,
    *,
    strict: bool = True,
    allow_fallback: bool = False,
) -> Dict[str, Any]:
    """Strict shared resolver wrapper for paper-forward candidate loading."""
    from candidate_artifact_resolver import resolve_candidate_artifact, log_artifact_resolution

    root = Path(project_root).resolve()
    if not strict and str(os.getenv("MSTOCK_STRICT_ARTIFACTS", "true")).lower() in ("0", "false", "no"):
        strict = False
    if not allow_fallback and str(os.getenv("MSTOCK_ALLOW_ARTIFACT_FALLBACK", "false")).lower() in ("1", "true", "yes"):
        allow_fallback = True

    resolution = resolve_candidate_artifact(
        candidate,
        root,
        strict=strict,
        allow_fallback=allow_fallback,
    )
    log_artifact_resolution(resolution, prefix="[ARTIFACT-MAP]")
    result = resolution.to_legacy_dict()
    result.update({
        "model_name": candidate.get("model_name", "unknown"),
        "preset_family": candidate.get("preset_family", "unknown"),
        "side_policy": candidate.get("side_policy", candidate.get("side", "BOTH")),
    })

    if not resolution.loaded or not resolution.resolved_dir:
        return result

    rd = root / Path(resolution.resolved_dir)
    files_to_check = {
        "candidate_profile.json": rd / "candidate_profile.json",
        "candidate_manifest.json": rd / "candidate_manifest.json",
        "paper_forward_manifest.json": rd / "paper_forward_manifest.json",
        "model_pkl": rd / "model.pkl",
        "feature_schema.json": rd / "feature_schema.json",
        "dynamic_preset.json": rd / "dynamic_preset.json",
        "dynamic_presets.json": rd / "dynamic_presets.json",
        "preprocessing_metadata.json": rd / "preprocessing_metadata.json",
        "metrics.json": rd / "metrics.json",
        "gates.json": rd / "gates.json",
    }
    exists_map = {k: _candidate_file_exists(v) if v else False for k, v in files_to_check.items()}
    result["files"] = {k: str(v) if v else None for k, v in files_to_check.items()}
    result["exists"] = exists_map

    status = LOAD_OK
    if not exists_map.get("model_pkl"):
        status = "missing_model_file"
    elif not exists_map.get("feature_schema.json"):
        status = "missing_feature_list"

    result["status"] = status
    result["load_status"] = REASONS.get(status, status if status != LOAD_OK else LOAD_OK)
    if status != LOAD_OK:
        result["suggested_fix"] = f"Ensure model + feature_schema under exact dir {rd}"
    return result

class PaperForwardEngine:
    def __init__(
        self,
        candidate_ids: Optional[List[str]] = None,
        candidate_file: Optional[str] = None,
        artifacts_dir: str = "artifacts/candidates",
        log_dir: str = "logs",
        reports_dir: str = "reports",
        broker_safe_mode: bool = True,
    ):
        # ensure containers exist immediately after construction (BLOCKER fix)
        self._ensure_candidate_state_containers()

        # explicit inits of state containers BEFORE loading candidates (per task)
        self._candidate_states: Dict[str, PaperForwardCandidateRuntimeState] = {}
        self._candidate_status: Dict[str, Any] = {}
        self._candidate_artifacts: Dict[str, Any] = {}
        self._candidate_decisions: Dict[str, Any] = {}
        self._decision_cache: Dict[str, Any] = {}
        self._state_lock = threading.RLock()

        self.artifacts_dir = Path(artifacts_dir)
        self.log_dir = Path(log_dir)
        self.reports_dir = Path(reports_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.broker_safe_mode = broker_safe_mode

        self.candidates: List[Dict[str, Any]] = []
        self._state: Dict[str, Dict[str, Any]] = {}  # per-cid simulated state
        self._decisions: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._snapshot_counter = 0
        # TASK 5 runtime diagnostics counters
        self._snapshots_received = 0
        self._evaluations_count = 0
        self._predictions_attempted = 0
        self._predictions_success = 0
        self._route_errors = 0
        self._skipped_missing_features = 0
        self._skipped_market_data = 0
        self._last_snapshot_ts = None
        self._latest_market_snapshot: Dict[str, Any] = {}
        self._latest_option_chain_snapshot: Any = None
        self._reason_counts: Dict[str, int] = {}
        self._last_data_status = PaperForwardDataStatus()
        self.last_data_quality = ""
        self._route_exception_details: List[Dict[str, Any]] = []
        # TASK6 auth gating counters (do not count evals while auth pending)
        self._auth_wait_cycles = 0
        self._duplicate_entry_blocked = 0
        self._entry_cooldown_blocked = 0
        self._signal_dedupe_blocked = 0
        self._last_auth_pending_reason = ""
        self._export_decision_buffer: List[Dict[str, Any]] = []
        self._pf_model_cache: Dict[str, Any] = {}
        self._pf_feature_schema_cache: Dict[str, Any] = {}
        self._scripmaster_cache: Dict[str, Any] = {}
        self._option_chain_cache: Dict[str, Any] = {"rows": [], "ts": 0.0, "source": ""}
        self._option_chain_cache_ttl_sec = float(os.getenv("PF_OPTION_CHAIN_CACHE_TTL_SEC", "5") or 5)
        self._ltp_token_cache: Dict[str, Any] = {}
        self._ltp_token_cache_ttl_sec = float(os.getenv("PF_LTP_TOKEN_CACHE_TTL_SEC", "30") or 30)
        self._ltp_failure_cache: Dict[str, float] = {}
        self.last_readiness_summary: Dict[str, Any] = {}

        print(f"[PF-ENGINE-INIT] state_containers_ready=true candidate_states={len(self._candidate_states)}")

        self._load_candidates(candidate_ids, candidate_file)

        # TASK2: init canonical per-cand runtime states (population after load)
        # (empty dict already created above)
        for c in self.candidates:
            cid = c["candidate_id"]
            if cid not in self._candidate_states:
                cs = PaperForwardCandidateRuntimeState(cid)
                cs.enabled = c.get("enabled", True)
                cs.model = c.get("model_name", "")
                cs.preset = c.get("preset_family", c.get("selected_preset", ""))
                cs.side = c.get("side_policy", "")
                self._candidate_states[cid] = cs
                if cid not in self._state:
                    self._state[cid] = {
                        "open_position": False, "side": None, "entry_time": None, "entry_price": None,
                        "current_price": None, "realized_pnl": 0.0, "unrealized_pnl": 0.0,
                        "total_entries": 0, "total_exits": 0, "total_trades": 0,
                        "wins": 0, "losses": 0, "max_drawdown": 0.0,
                        "last_signal": None, "last_no_trade_reason": "",
                        "selected_preset": cs.preset, "confidence": 0.0, "threshold": 0.0,
                        "error_count": 0, "last_update": None, "predict_attempted": False,
                    }

        print(f"[PF-ENGINE-LOAD] declared={len(self.candidates)} valid={sum(1 for c in self.candidates if c.get('enabled',True) and c.get('_load_status')==LOAD_OK)} invalid={sum(1 for c in self.candidates if not c.get('enabled',True) or c.get('_load_status') != LOAD_OK)}")

    def _ensure_candidate_state_containers(self):
        """Ensure all candidate state containers exist (for startup robustness and calls before full init)."""
        if not hasattr(self, "_candidate_states") or self._candidate_states is None:
            self._candidate_states = {}
        if not hasattr(self, "_candidate_status") or self._candidate_status is None:
            self._candidate_status = {}
        if not hasattr(self, "_candidate_artifacts") or self._candidate_artifacts is None:
            self._candidate_artifacts = {}
        if not hasattr(self, "_candidate_decisions") or self._candidate_decisions is None:
            self._candidate_decisions = {}
        if not hasattr(self, "_decision_cache") or self._decision_cache is None:
            self._decision_cache = {}
        if not hasattr(self, "_state_lock") or self._state_lock is None:
            self._state_lock = threading.RLock()

    def _get_or_create_candidate_state(self, cid: str) -> "PaperForwardCandidateRuntimeState":
        self._ensure_candidate_state_containers()
        if cid not in self._candidate_states:
            self._candidate_states[cid] = PaperForwardCandidateRuntimeState(cid)
        return self._candidate_states[cid]

    def _apply_paper_forward_decision_to_state(self, decision: Dict[str, Any]) -> None:
        """TASK 3: sync decision into canonical candidate runtime state.
        Called after every accepted decision (from _record and good paths).
        """
        self._ensure_candidate_state_containers()
        cid = decision.get("candidate_id")
        if not cid:
            return
        cs = self._get_or_create_candidate_state(cid)
        # core fields
        cs.final_signal = decision.get("final_signal", decision.get("last_signal", cs.final_signal))
        cs.confidence = decision.get("confidence")
        cs.threshold = decision.get("threshold", cs.threshold)
        cs.predict_attempted = bool(decision.get("predict_attempted", cs.predict_attempted))
        cs.reason_code = decision.get("reason_code") or decision.get("no_trade_reason", cs.reason_code)
        cs.reason_detail = decision.get("no_trade_reason") or decision.get("reason_detail", cs.reason_detail) or cs.reason_code
        cs.last_snapshot_id = decision.get("snapshot_id", cs.last_snapshot_id)
        cs.last_eval_ts = decision.get("timestamp") or decision.get("last_update", cs.last_eval_ts)
        cs.data_quality = decision.get("debug", {}).get("data_quality_status", "") or decision.get("data_quality_status", cs.data_quality)
        cs.missing_features = decision.get("missing_features", cs.missing_features)
        cs.last_action = decision.get("simulated_action", decision.get("paper_action", cs.last_action))
        cs.block_reason = str(
            decision.get("block_reason")
            or decision.get("no_trade_reason")
            or decision.get("reason_code")
            or cs.block_reason
            or ""
        )
        if "allowed" in decision:
            try:
                cs.allowed = bool(decision.get("allowed"))
            except Exception:
                cs.allowed = None
        if "model_type" in decision:
            cs.model_type = str(decision.get("model_type") or cs.model_type or "")
        if "feature_missing_count" in decision:
            try:
                cs.feature_missing_count = int(decision.get("feature_missing_count") or 0)
            except Exception:
                cs.feature_missing_count = 0
        elif decision.get("missing_features") is not None:
            try:
                cs.feature_missing_count = len(decision.get("missing_features") or [])
            except Exception:
                cs.feature_missing_count = 0
        if "feature_invalid_count" in decision:
            try:
                cs.feature_invalid_count = int(decision.get("feature_invalid_count") or 0)
            except Exception:
                cs.feature_invalid_count = 0
        for attr_name in ("ensemble_prob", "xgb_prob", "rf_prob"):
            if attr_name not in decision:
                continue
            raw_val = decision.get(attr_name)
            try:
                setattr(cs, attr_name, None if raw_val in (None, "") else float(raw_val))
            except Exception:
                setattr(cs, attr_name, None)

        # paper pos / pnl from decision or state
        if "position_status" in decision:
            cs.pos = decision["position_status"]
            if str(cs.pos or "").upper() == "FLAT":
                cs.entry_strike = None
                cs.option_type = ""
                cs.entry_symbol = ""
                cs.side = ""
                cs.qty = 0
        if "selected_option_type" in decision or "sim_side" in decision:
            cs.option_type = str(decision.get("selected_option_type") or decision.get("sim_side") or cs.option_type or "")
            cs.side = cs.option_type or cs.side
        if "selected_strike" in decision or "entry_strike" in decision:
            strike_val = decision.get("selected_strike", decision.get("entry_strike"))
            if strike_val in (None, "", 0, 0.0):
                cs.entry_strike = None
            else:
                try:
                    cs.entry_strike = float(strike_val)
                except Exception:
                    cs.entry_strike = None
        if "selected_symbol" in decision:
            cs.entry_symbol = str(decision.get("selected_symbol") or "")
        if "unrealized_pnl" in decision:
            cs.unreal_pnl = float(decision.get("unrealized_pnl", cs.unreal_pnl) or 0)
        if "realized_pnl" in decision:
            cs.realized_pnl = float(decision.get("realized_pnl", cs.realized_pnl) or 0)
        if "total_trades" in decision:
            cs.trades = int(decision.get("total_trades", cs.trades) or 0)
        if "total_entries" in decision:
            cs.entries = int(decision.get("total_entries", cs.entries) or 0)
        if "total_exits" in decision:
            cs.exits = int(decision.get("total_exits", cs.exits) or 0)
        if "open_since" in decision:
            cs.open_since = decision.get("open_since") or cs.open_since
        if "entry_symbol" in decision:
            cs.entry_symbol = str(decision.get("entry_symbol") or cs.entry_symbol or "")
        if "pnl_source" in decision:
            cs.pnl_source = str(decision.get("pnl_source") or cs.pnl_source or "")
        cs.raw_reason = decision.get("no_trade_reason") or decision.get("reason_code") or cs.raw_reason
        if "wins" in decision:
            cs.wins = int(decision.get("wins", cs.wins) or 0)
        if "losses" in decision:
            cs.losses = int(decision.get("losses", cs.losses) or 0)
        if "win_rate" in decision:
            cs.win_rate = float(decision.get("win_rate", cs.win_rate) or 0)
        if "max_drawdown" in decision:
            cs.max_drawdown = float(decision.get("max_drawdown", cs.max_drawdown) or 0)
        entry_decision_price = decision.get("entry_price", decision.get("sim_entry_price"))
        if entry_decision_price is not None:
            cs.entry_price = float(entry_decision_price)
        current_decision_price = decision.get("option_current_price", decision.get("current_price", decision.get("sim_exit_price")))
        if current_decision_price is not None:
            cs.current_price = float(current_decision_price)
        if "position_qty" in decision:
            cs.qty = int(decision.get("position_qty", cs.qty) or 0)
        elif "qty" in decision:
            cs.qty = int(decision.get("qty", cs.qty) or 0)

        # update global reason counts (TASK5)
        reason = cs.reason_code or "ok"
        self._reason_counts[reason] = self._reason_counts.get(reason, 0) + 1

        # also sync to legacy _state for compat
        st = self._state.setdefault(cid, {})
        st["last_no_trade_reason"] = cs.reason_code
        st["last_signal"] = cs.final_signal
        st["confidence"] = cs.confidence
        st["threshold"] = cs.threshold
        st["predict_attempted"] = cs.predict_attempted
        st["simulated_action"] = cs.last_action or st.get("simulated_action", "")
        st["paper_action"] = cs.last_action or st.get("paper_action", "")
        st["last_update"] = cs.last_eval_ts
        st["current_price"] = cs.current_price
        st["option_current_price"] = cs.current_price
        st["open_position"] = cs.pos != "FLAT"
        st["side"] = cs.side
        st["entry_price"] = cs.entry_price
        st["entry_strike"] = cs.entry_strike
        if cs.pos == "FLAT":
            st["selected_strike"] = None
            st["selected_option_type"] = ""
            st["selected_symbol"] = ""
        else:
            st["selected_strike"] = cs.entry_strike
            st["selected_option_type"] = cs.option_type or cs.side or ""
            st["selected_symbol"] = cs.entry_symbol
        st["unrealized_pnl"] = cs.unreal_pnl
        st["realized_pnl"] = cs.realized_pnl
        st["total_trades"] = cs.trades
        st["wins"] = cs.wins
        st["losses"] = cs.losses
        st["win_rate"] = cs.win_rate
        st["max_drawdown"] = cs.max_drawdown
        st["ensemble_prob"] = cs.ensemble_prob
        st["xgb_prob"] = cs.xgb_prob
        st["rf_prob"] = cs.rf_prob
        st["block_reason"] = cs.block_reason
        st["allowed"] = cs.allowed
        st["model_type"] = cs.model_type
        st["feature_missing_count"] = cs.feature_missing_count
        st["feature_invalid_count"] = cs.feature_invalid_count

        pf_log(
            "DEBUG",
            f"[PAPER-FWD-STATE-APPLY] cid={cid} signal={cs.final_signal} conf={cs.confidence} "
            f"reason={cs.reason_code} pos={cs.pos}",
            rate_key=f"state_apply:{cid}",
            rate_interval=30.0,
        )

    def _load_candidates(self, candidate_ids: Optional[List[str]], candidate_file: Optional[str]):
        self._ensure_candidate_state_containers()
        loaded = []
        if candidate_file:
            p = Path(candidate_file)
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception as exc:
                    print(f"[PAPER-FWD-LOAD] candidate_file_read_failed path={p} error={type(exc).__name__}:{exc}")
                    data = {"candidates": []}
                c_list = data.get("candidates", data if isinstance(data, list) else [])
                for c in c_list:
                    try:
                        if c.get("paper_forward_only") or c.get("classification") == "paper_forward_only" or c.get("recommended_mode") == "paper_forward_only":
                            loaded.append(c)
                    except Exception as exc:
                        print(f"[PAPER-FWD-LOAD] candidate_row_invalid error={type(exc).__name__}:{exc}")
        if candidate_ids:
            for cid in candidate_ids:
                if not any(x["candidate_id"] == cid for x in loaded):
                    loaded.append({
                        "candidate_id": cid,
                        "enabled": True,
                        "classification": "paper_forward_only",
                        "artifact_dir": str(self.artifacts_dir / cid),
                        "model_name": "unknown",
                        "preset_family": "unknown",
                        "side_policy": "BOTH",
                        "paper_forward_only": True,
                        "shadow_ready": False,
                        "notes": "paper observation only"
                    })
        # Dedup exact duplicate paper-forward rows only. Candidate IDs can be long
        # and threshold variants must remain distinct unless the full identity
        # matches.
        final = []
        seen = set()
        duplicate_removed = 0
        declared_count = len(loaded)
        for c in loaded:
            cid = str(c.get("candidate_id") or "")
            if not cid:
                continue
            threshold = c.get("threshold")
            if threshold in (None, ""):
                threshold = c.get("selected_threshold") or c.get("entry_threshold") or ""
            key = (
                cid,
                str(c.get("artifact_dir") or ""),
                str(c.get("model_name") or ""),
                str(c.get("preset_family") or ""),
                str(c.get("side_policy") or ""),
                str(threshold or ""),
            )
            if key in seen:
                duplicate_removed += 1
                continue
            seen.add(key)
            if c.get("paper_forward_only") or c.get("classification") == "paper_forward_only":
                c = dict(c)
                c["enabled"] = _coerce_config_bool(c.get("enabled", True), default=True)
                disabled_reason = str(c.get("disabled_reason") or "").strip()
                if (not c["enabled"]) and disabled_reason in _PAPER_FORWARD_RUNTIME_DISABLE_OVERRIDES:
                    c["_orig_disabled_reason"] = disabled_reason
                    c["_orig_enabled_config"] = False
                    c["enabled"] = True
                    c["disabled_reason"] = ""
                    c["_paper_forward_runtime_override"] = disabled_reason
                c["_orig_enabled"] = c["enabled"]
                c["_row_key"] = "|".join(key)
                final.append(c)
        self._candidate_load_summary = {
            "config_candidates": declared_count,
            "loaded_candidates": len(final),
            "duplicate_removed_count": duplicate_removed,
            "enabled_candidates": sum(1 for c in final if c.get("enabled", True)),
            "disabled_candidates": sum(1 for c in final if not c.get("enabled", True)),
        }

        # TASK2/4 + PHASE 2/3: resolve artifacts + presets using canonical registries first
        reg = load_candidate_registry(REPO_ROOT)  # now includes paper_forward_artifact_registry.json
        preset_reg = load_dynamic_presets(REPO_ROOT)

        # Also load the dedicated preset registry (TASK 3)
        preset_registry_path = REPO_ROOT / "config" / "paper_forward_preset_registry.json"
        dedicated_preset_reg = _safe_load_json(preset_registry_path) or {"presets": {}}

        resolved_any = 0
        preset_resolved = 0
        for c in final:
            cid = c["candidate_id"]
            configured_ad = c.get("artifact_dir") or str(self.artifacts_dir / cid)
            res = resolve_candidate_artifacts(c, REPO_ROOT)
            c["_resolve"] = res
            c["_config_artifact_dir"] = c.get("artifact_dir") or configured_ad
            # STRICT: adopt only if resolved + identity ok (per new A-E + validation)
            adopted = False
            idst = res.get("artifact_identity_status", "MISSING")
            if res.get("resolved_dir") and idst in ("ok", "EXACT_MATCH"):
                rd = res.get("resolved_dir")
                c["artifact_dir"] = rd
                c["_artifact_resolved_from"] = "resolve"
                resolved_any += 1
                adopted = True
            elif idst and idst != "ok":
                c["artifact_dir"] = None
                c["_artifact_resolved_from"] = "identity_mismatch"
                c["_load_status"] = res.get("load_status") or "ARTIFACT_IDENTITY_MISMATCH"
                c["disabled_reason"] = res.get("load_status") or f"ARTIFACT_IDENTITY_MISMATCH expected={c['candidate_id']}"
                if c.get("enabled", True):
                    c["enabled"] = False
            if not adopted:
                if c.get("_load_status") == LOAD_OK:
                    c["artifact_dir"] = configured_ad
                else:
                    c["artifact_dir"] = None
                c["_artifact_resolved_from"] = c.get("_artifact_resolved_from", "config_exact")

            # Prefer dedicated paper_forward_preset_registry, then global, then per-cand
            pres = resolve_candidate_preset(c, {"loaded": True, "data": dedicated_preset_reg.get("presets", {})})
            if not pres.get("preset_loaded_ok") or pres.get("preset_fallback_generated"):
                # fall back to previous global logic
                pres2 = resolve_candidate_preset(c, preset_reg)
                if pres2.get("preset_loaded_ok"):
                    pres = pres2

            c["_preset"] = pres
            c["selected_preset"] = pres.get("selected", c.get("preset_family", "PAPER_OBSERVE_ONLY"))
            if pres.get("preset_fallback_generated"):
                c["_preset_fallback"] = True
                print(f"[PAPER-FWD-PRESET] candidate_id={cid} status=FALLBACK selected={c['selected_preset']} source={pres.get('source')}")
            elif pres.get("preset_loaded_ok"):
                preset_resolved += 1
                print(f"[PAPER-FWD-PRESET] candidate_id={cid} status=OK selected={c['selected_preset']} source={pres.get('source')}")

            c["_load_status"] = res.get("load_status") or res.get("status", "ARTIFACT_NOT_FOUND")
            # respect config disabled
            if not c.get("enabled", True):
                c["_load_status"] = c.get("disabled_reason", "ARTIFACT_MISSING")
            if c["_load_status"] != LOAD_OK or c["_load_status"] == "ARTIFACT_MISSING":
                print(f"[PAPER-FWD-ARTIFACT] candidate_id={cid} status=FAILED reason={c['_load_status']} resolved={c.get('artifact_dir')}")
            else:
                print(f"[PAPER-FWD-ARTIFACT] candidate_id={cid} status=OK resolved={c.get('artifact_dir')}")

            # ensure invalid have confidence=None not default 0.0
            if not c.get("enabled", True) or c.get("_load_status") in ("FEATURE_ORDER_MISSING", "ARTIFACT_MISSING", "ARTIFACT_NOT_FOUND", "candidate_missing_artifact_paths"):
                st_missing = self._state.setdefault(cid, {})
                st_missing["confidence"] = None
                if c.get("_load_status") in ("ARTIFACT_MISSING", "ARTIFACT_NOT_FOUND", "candidate_missing_artifact_paths"):
                    st_missing["last_no_trade_reason"] = "ARTIFACT_NOT_FOUND"
                    st_missing["artifact_dir"] = None
                    st_missing["model_path"] = None
                    st_missing["suggestion"] = "run scripts/repair_paper_forward_candidate_artifacts.py --dry-run"
                cs = self._candidate_states.get(cid)
                if cs:
                    cs.confidence = None
                    if c.get("_load_status") in ("ARTIFACT_MISSING", "ARTIFACT_NOT_FOUND", "candidate_missing_artifact_paths"):
                        cs.reason_code = "ARTIFACT_NOT_FOUND"

        self.candidates = final
        self._candidate_load_summary.update({
            "enabled_candidates": sum(1 for c in self.candidates if c.get("enabled", True)),
            "disabled_candidates": sum(1 for c in self.candidates if not c.get("enabled", True)),
        })
        # Load per-candidate feature contract + full metadata (Tasks A/B + required multi-key fo support)
        for c in self.candidates:
            cid = c["candidate_id"]
            art_dir = c.get("artifact_dir") or ""
            if c.get("_load_status") != LOAD_OK:
                c["_feature_list"] = []
                c["_feature_order"] = []
                c["required_features"] = 0
                c["_scaler_present"] = False
                c["_calibrator_present"] = False
                c["_feature_order_src"] = ""
                c["_requires_candles"] = False
                c["_inspected_for_fo"] = []
                print(
                    f"[CAND-VALIDATE] candidate_id={cid} resolved_artifact_dir= "
                    f"model_path= model_class={c.get('model_name', 'unknown')} feature_order_len=0 "
                    f"first_10_features=[] preset_family={c.get('preset_family', 'unknown')} "
                    f"side={c.get('side_policy', 'BOTH')} threshold={c.get('threshold', 0.0)} "
                    f"artifact_identity_status={(c.get('_resolve') or {}).get('artifact_identity_status', 'MISSING')}"
                )
                continue
            flist = []
            feature_order_src = ""
            scaler_present = False
            calibrator_present = False
            model_name = c.get("model_name", "unknown")
            preset_family = c.get("preset_family", "unknown")
            side_policy = c.get("side_policy", "BOTH")
            threshold = 0.35
            ad = Path(art_dir) if art_dir else Path(".")
            probed_dirs = [ad, ad / cid, ad / (ad.name or "")]
            inspected_files: List[str] = []
            checked_keys: List[str] = ["feature_order", "features", "live_computable_features", "required_features", "model_features", "metadata.feature_order", "training_metadata.feature_order", "ml_signals.features"]
            sidecar_names = ["feature_schema.json", "feature_order.json", "metadata.json", "model_card.json", "manifest.json", "training_metadata.json", "paper_forward_manifest.json", "candidate_manifest.json", "candidate_profile.json", "preprocessing_metadata.json"]

            for candp in probed_dirs:
                # 1. feature_schema + other sidecar json with ALL supported keys (fix 2)
                for sc_name in sidecar_names:
                    fs = candp / sc_name
                    if fs.exists():
                        inspected_files.append(str(fs))
                        if not flist:
                            try:
                                data = json.loads(fs.read_text(encoding="utf-8"))
                                if isinstance(data, list):
                                    flist = [str(x) for x in data if x]
                                elif isinstance(data, dict):
                                    # support nested + many aliases per spec (metadata.feature_order, training_metadata.feature_order, ml_signals bundle etc)
                                    for key in ("feature_order", "features", "live_computable_features", "model_features", "required_features"):
                                        if key in data and not flist:
                                            flist = [str(x) for x in (data.get(key) or []) if x]
                                    # direct metadata.feature_order etc
                                    for meta_k in ("metadata", "training_metadata"):
                                        if not flist:
                                            md = data.get(meta_k) or {}
                                            if isinstance(md, dict):
                                                for key in ("feature_order", "features", "model_features", "required_features"):
                                                    if md.get(key) and not flist:
                                                        flist = [str(x) for x in (md.get(key) or []) if x]
                                    if not flist and data.get("ml_signals"):
                                        ms = data.get("ml_signals") or {}
                                        flist = [str(x) for x in (ms.get("features") or ms.get("feature_order") or ms.get("live_computable_features") or []) if x]
                                    # also check top level for bundle style
                                    if not flist:
                                        for key in ("feature_names", "feature_names_in_"):
                                            if data.get(key) and not flist:
                                                flist = [str(x) for x in (data.get(key) or []) if x]
                                if flist:
                                    feature_order_src = str(fs)
                            except Exception:
                                pass
                # manifests for meta (extended)
                for mname in ("candidate_profile.json", "candidate_manifest.json", "paper_forward_manifest.json", "metadata.json", "model_card.json"):
                    mp = candp / mname
                    if mp.exists():
                        inspected_files.append(str(mp))
                        try:
                            m = json.loads(mp.read_text(encoding="utf-8"))
                            model_name = m.get("model_name") or m.get("model", model_name) or model_name
                            preset_family = m.get("preset_family") or m.get("preset", preset_family) or preset_family
                            side_policy = m.get("side_policy") or m.get("filter_name") or side_policy
                            thr = m.get("selected_threshold") or m.get("threshold_policy", {}).get("entry_threshold") or m.get("threshold")
                            if thr:
                                threshold = float(thr)
                        except Exception:
                            pass
                # preprocessing for scaler/cal
                pp = candp / "preprocessing_metadata.json"
                if pp.exists():
                    inspected_files.append(str(pp))
                    try:
                        pd = json.loads(pp.read_text(encoding="utf-8"))
                        if pd.get("scaler"):
                            scaler_present = True
                        if pd.get("calibrator") or pd.get("calibration"):
                            calibrator_present = True
                    except Exception:
                        pass
                # sidecar pkl scaler/cal
                if (candp / "scaler.pkl").exists() or (candp / "preprocessing" / "scaler.pkl").exists():
                    scaler_present = True
                if (candp / "calibrator.pkl").exists():
                    calibrator_present = True

            # 2. Load from the actual trained artifact bundle (pickle) itself (fix 2)
            if not flist:
                model_p = None
                res = c.get("_resolve") or {}
                if res.get("model_file"):
                    model_p = Path(res["model_file"])
                else:
                    for candp in probed_dirs:
                        mp = candp / "model.pkl"
                        if mp.exists():
                            model_p = mp
                            break
                        for f in candp.glob("*.pkl"):
                            if "metric" not in f.name.lower() and "ensemble" not in f.name.lower():
                                model_p = f
                                break
                        if model_p:
                            break
                if model_p and model_p.exists():
                    inspected_files.append(str(model_p))
                    try:
                        import pickle as _pkl
                        with model_p.open("rb") as fh:
                            b = _pkl.load(fh)
                        # support bundle dict or model with attrs
                        if isinstance(b, dict):
                            for k in ("feature_order", "features", "feature_names", "required_features", "model_features", "feature_names_in_"):
                                if b.get(k) and not flist:
                                    flist = [str(x) for x in (b.get(k) or []) if x]
                                    feature_order_src = f"{model_p}:{k}"
                            # nested common + ml bundle
                            inner = b.get("model") or b.get("estimator") or b.get("ml_signals") or b
                            if not flist:
                                for k in ("feature_names_in_", "feature_names", "features", "feature_order", "model_features"):
                                    arr = getattr(inner, k, None) or b.get(k) or (inner.get(k) if isinstance(inner, dict) else None)
                                    if arr and not flist:
                                        flist = [str(x) for x in (arr if isinstance(arr, (list,tuple)) else []) if x]
                                        feature_order_src = f"{model_p}:bundle.{k}"
                            if not flist and isinstance(b.get("ml_signals"), dict):
                                msb = b.get("ml_signals") or {}
                                for k in ("features", "feature_order"):
                                    if msb.get(k) and not flist:
                                        flist = [str(x) for x in (msb.get(k) or []) if x]
                                        feature_order_src = f"{model_p}:ml_signals.{k}"
                        else:
                            for k in ("feature_names_in_", "features", "feature_names", "feature_order"):
                                arr = getattr(b, k, None)
                                if arr and not flist:
                                    flist = [str(x) for x in arr if x]
                                    feature_order_src = f"{model_p}:model.{k}"
                    except Exception as e:
                        inspected_files.append(f"pickle_fo_fail:{e}")

            # also try sidecar for selected_threshold if not found
            if threshold in (0, 0.0, None) or not threshold:
                for mname in ("candidate_manifest.json", "metrics.json", "gates.json", "model_card.json"):
                    for cp in probed_dirs:
                        mp = cp / mname
                        if mp.exists():
                            inspected_files.append(str(mp))
                            try:
                                md = json.loads(mp.read_text(encoding="utf-8"))
                                if "selected_threshold" in md:
                                    threshold = float(md["selected_threshold"])
                                    break
                                tp = md.get("threshold_policy") or {}
                                if isinstance(tp, dict) and tp.get("entry_threshold"):
                                    threshold = float(tp["entry_threshold"])
                                    break
                            except Exception:
                                pass
                    if threshold not in (0, 0.0, None):
                        break
            if threshold in (0, 0.0, None) or not threshold:
                threshold = 0.35
                print(f"[THRESHOLD] candidate_id={cid} source=default value=0.35")

            # attach
            c["model_name"] = model_name
            c["preset_family"] = preset_family
            c["side_policy"] = side_policy
            c["threshold"] = threshold
            c["_feature_list"] = flist or []
            c["_feature_order"] = flist or []
            c["required_features"] = len(flist or [])
            c["_scaler_present"] = scaler_present
            c["_calibrator_present"] = calibrator_present
            c["_feature_order_src"] = feature_order_src or (str(probed_dirs[0] / "feature_schema.json") if (probed_dirs[0] / "feature_schema.json").exists() else "")
            c["_requires_candles"] = any("ret_" in f or "atr" in f or "adx" in f or "candle" in f.lower() for f in (flist or []))
            c["_inspected_for_fo"] = inspected_files

            # 4/5: strong [CAND-VALIDATE] after loading each (print all requested fields)
            res = c.get("_resolve") or {}
            model_p_str = res.get("model_file") or (str(probed_dirs[0] / "model.pkl") if (probed_dirs[0] / "model.pkl").exists() else "")
            first10 = (flist or [])[:10]
            print(
                f"[CAND-VALIDATE] candidate_id={cid} resolved_artifact_dir={c.get('artifact_dir','')} "
                f"model_path={model_p_str} model_class={model_name} feature_order_len={len(flist or [])} "
                f"first_10_features={first10} preset_family={preset_family} side={side_policy} threshold={threshold}"
            )

            # B validation + detailed disable if fo_len==0 for ML (do not silently continue)
            is_ml = model_name not in ("rule", "rules", "heuristic", "none", "")
            if is_ml and c["required_features"] == 0:
                prior = str(c.get("_load_status") or "")
                if "ARTIFACT_NOT_FOUND" in prior or "ARTIFACT_IDENTITY" in prior or "ARTIFACT_MISSING" in prior or "missing_artifact" in prior:
                    # keep the more precise artifact error; do not overwrite with generic FEATURE
                    pass
                else:
                    reason_detail = f"FEATURE_ORDER_MISSING artifact_dir={c.get('artifact_dir')} model_path={model_p_str} checked_keys={checked_keys} inspected={inspected_files[:8]}"
                    print(f"[FEATURE-ORDER-MISSING] {reason_detail}")
                    c["_load_status"] = "FEATURE_ORDER_MISSING"
                    c["disabled_reason"] = reason_detail
                    if c.get("enabled", True):
                        c["enabled"] = False
        en = sum(1 for c in self.candidates if c.get("enabled", True))
        inv = len(self.candidates) - en
        print(f"[PAPER-FWD-LOAD] Loaded {len(self.candidates)} paper_forward_only candidates (enabled_valid={en} invalid={inv} resolved_ok={resolved_any}, real_preset_or_dedicated={preset_resolved})")
        print(
            "[PF-CANDIDATE] "
            f"config_candidates={self._candidate_load_summary.get('config_candidates', 0)} "
            f"loaded={self._candidate_load_summary.get('loaded_candidates', 0)} "
            f"enabled={self._candidate_load_summary.get('enabled_candidates', 0)} "
            f"disabled={self._candidate_load_summary.get('disabled_candidates', 0)} "
            f"duplicate_removed={self._candidate_load_summary.get('duplicate_removed_count', 0)}"
        )

    def _record_decision(self, cid: str, decision: Dict[str, Any], ts: str, market_snapshot: dict) -> None:
        """Store one candidate decision and update runtime counters consistently."""
        self._ensure_candidate_state_containers()
        reason = decision.get("no_trade_reason") or decision.get("reason_code") or "ok"
        decision["reason_code"] = reason
        decision.setdefault("route_error", False)
        decision.setdefault("live_orders_enabled", False)
        decision.setdefault("broker_orders_enabled", False)
        self._decisions.append(decision)
        skip_eval_count = bool(decision.get("_skip_eval_count"))
        if not skip_eval_count:
            self._evaluations_count += 1
            if decision.get("predict_attempted"):
                self._predictions_attempted += 1
                if isinstance(decision.get("confidence"), (int, float)):
                    self._predictions_success += 1
        if reason == "feature_vector_missing_columns":
            self._skipped_missing_features += 1
        if reason in (
            "token_not_verified",
            "session_expired",
            "broker_auth_failed",
            "candles_empty",
            "option_chain_empty",
            "thin_option_chain",
            "spot_missing",
            "warmup_wait",
            "stale_snapshot",
            "data_not_ready",
            "WAITING_FOR_MSTOCK_EXCHANGE",
            "WAITING_FOR_OPTION_CHAIN_FETCH",
            "SCRIPMASTER_NO_MATCHING_EXPIRY",
        ):
            self._skipped_market_data += 1
        # reason count handled in _apply_paper_forward_decision_to_state (TASK5)

        st = self._state.setdefault(cid, {})
        st["last_no_trade_reason"] = reason
        st["last_signal"] = decision.get("final_signal", "NO_TRADE")
        st["confidence"] = decision.get("confidence")
        st["threshold"] = decision.get("threshold", st.get("threshold", 0.0))
        st["predict_attempted"] = bool(decision.get("predict_attempted"))
        st["last_update"] = ts
        st["current_price"] = market_snapshot.get("price") or market_snapshot.get("last") or market_snapshot.get("close") or st.get("current_price")

        # TASK3: canonical apply for state, counts, paper pos sync
        self._apply_paper_forward_decision_to_state(decision)

    def _log_route_exception(
        self,
        candidate: Dict[str, Any],
        market_snapshot: dict,
        option_chain_snapshot: Any,
        required: List[str],
        feature_debug: Dict[str, Any],
        exc: BaseException,
        tb: str,
    ) -> Dict[str, Any]:
        detail = {
            "candidate_id": candidate.get("candidate_id"),
            "model": candidate.get("model_name"),
            "preset": candidate.get("selected_preset") or candidate.get("preset_family"),
            "snapshot_keys": sorted(list((market_snapshot or {}).keys())),
            "option_rows": _count_option_rows(option_chain_snapshot),
            "candles_count": feature_debug.get("candles_count", 0),
            "required_features_count": len(required),
            "available_features_count": feature_debug.get("available_features_count", 0),
            "missing_features_sample": list(feature_debug.get("missing_features", []))[:10],
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": tb,
        }
        self._route_exception_details.append(detail)
        self._route_exception_details = self._route_exception_details[-25:]
        pf_log(
            "ERROR",
            f"[PAPER-FWD-ROUTE-EXCEPTION] cid={detail.get('candidate_id')} "
            f"type={detail.get('exception_type')} msg={str(detail.get('exception_message', ''))[:120]}",
            rate_key=f"route_exception:{detail.get('candidate_id')}",
            rate_interval=30.0,
        )
        return detail

    def on_market_snapshot(self, market_snapshot: dict, option_chain_snapshot: dict | None = None) -> List[dict]:
        """
        Core: deliver identical snapshot to every enabled candidate.
        Returns list of decision dicts (one per candidate).
        """
        cycle_t0 = time.perf_counter()
        feature_ms = 0.0
        predict_ms = 0.0
        signal_count = 0
        no_trade_count = 0
        error_count = 0
        self._ensure_candidate_state_containers()
        self._snapshot_counter += 1
        snap_id = f"snap_{int(time.time())}_{self._snapshot_counter}"
        ts = datetime.now(timezone.utc).isoformat()
        results = []

        market_snapshot = dict(market_snapshot or {})
        if option_chain_snapshot is not None:
            market_snapshot.setdefault("_option_chain_snapshot", option_chain_snapshot)
            if isinstance(option_chain_snapshot, (list, tuple)):
                market_snapshot.setdefault("option_chain", list(option_chain_snapshot))
        _apply_spot_candle_fallback_if_needed(market_snapshot)
        self._latest_market_snapshot = dict(market_snapshot)
        self._latest_option_chain_snapshot = option_chain_snapshot if option_chain_snapshot is not None else market_snapshot.get("option_chain")
        candle_rows = _candle_count_from_snapshot(market_snapshot)
        try:
            from synthetic_option_chain import load_bs_config
            min_candles = int(load_bs_config().get("min_candles_for_prediction", 20))
        except Exception:
            min_candles = 20
        has_candles = candle_rows >= min_candles
        has_chain = _count_option_rows(option_chain_snapshot if option_chain_snapshot is not None else market_snapshot.get("option_chain")) > 0

        snap_log = {
            "ts": ts,
            "spot": market_snapshot.get("price") or market_snapshot.get("close") or market_snapshot.get("last"),
            "option_rows": _count_option_rows(option_chain_snapshot if option_chain_snapshot is not None else market_snapshot.get("option_chain")),
            "candles": bool(has_candles),
            "source": market_snapshot.get("source", "ui_or_engine"),
        }
        pf_log(
            "DEBUG",
            f"[PAPER-FWD-SNAPSHOT] {snap_log}",
            rate_key="paper_fwd_snapshot",
            rate_interval=5.0,
        )
        self._snapshots_received += 1
        self._last_snapshot_ts = ts
        self._last_data_status = PaperForwardDataStatus.from_snapshot(market_snapshot, option_chain_snapshot)
        data_status = self._last_data_status.as_dict()
        _paper_fwd_log(
            "PAPER-FWD-FLOW",
            phase="snapshot_received",
            data_quality=data_status.get("data_quality_status"),
            spot=snap_log.get("spot"),
            option_rows=snap_log.get("option_rows"),
            candle_rows=candle_rows,
            candidates=len(self.candidates),
        )
        if data_status.get("spot") not in (None, "", 0, 0.0) and not (
            market_snapshot.get("spot") or market_snapshot.get("price") or market_snapshot.get("close") or market_snapshot.get("last")
        ):
            market_snapshot["spot"] = data_status.get("spot")
            market_snapshot["price"] = data_status.get("spot")
        data_quality = data_status.get("data_quality_status") or "UNKNOWN"
        if (
            data_quality == "TOKEN_NOT_VERIFIED"
            and not any(k in market_snapshot for k in ("broker_auth", "auth_status", "token_present", "token_set"))
        ):
            data_quality = "DATA_OK"
            data_status["data_quality_status"] = data_quality
        option_rows = int(data_status.get("option_chain_rows") or 0)
        spot_val = (
            market_snapshot.get("spot")
            or market_snapshot.get("price")
            or market_snapshot.get("close")
            or market_snapshot.get("last")
            or data_status.get("spot")
        )
        data_quality, _engine_ready = reconcile_paper_forward_readiness(
            data_quality=data_quality,
            candle_rows=candle_rows,
            option_rows=option_rows,
            spot=spot_val,
            min_candles=min_candles,
            old_quality=str(market_snapshot.get("last_data_quality") or self.last_data_quality or ""),
        )
        if data_quality == "DATA_OK":
            market_snapshot["data_quality_status"] = "DATA_OK"
            data_status["data_quality_status"] = "DATA_OK"

        log_pf_data(
            spot=spot_val,
            option_rows=option_rows,
            candles=candle_rows,
            data_quality=data_quality,
            spot_source=str(market_snapshot.get("spot_source") or market_snapshot.get("source") or ""),
        )

        with self._lock:
            for cand in self.candidates:
                cid = cand["candidate_id"]  # keep configured stable id (BLOCKER 4)
                load_status = cand.get("_load_status", LOAD_OK)
                art_dir = cand.get("artifact_dir") or str(self.artifacts_dir / cid)
                artifact_id = Path(art_dir).name if art_dir else cid
                enabled = bool(cand.get("enabled", True))

                # Always emit a decision row per declared candidate (even if disabled/fo-missing) so monitors/tests see all.
                # But if disabled or bad load, short-circuit with precise reason (no predict).
                art_path = resolve_repo_path(art_dir, REPO_ROOT)
                art_exists = bool(art_path and art_path.exists())
                if not enabled or load_status != LOAD_OK or not art_exists:
                    reason = load_status if load_status != LOAD_OK else (cand.get("disabled_reason") or "candidate_disabled_or_missing_artifact")
                    log_paper_fwd_decision(
                        candidate_id=cid,
                        signal="NO_TRADE",
                        reason=reason,
                        allowed=f"enabled={enabled},artifact={art_exists}",
                    )
                    log_paper_fwd_candidate(
                        candidate_id=cid,
                        signal="NO_TRADE",
                        reason=reason,
                    )
                    err_dec = {
                        "snapshot_id": snap_id,
                        "timestamp": ts,
                        "candidate_id": cid,
                        "paper_forward_only": True,
                        "shadow_ready": False,
                        "forced_eval": True,
                        "final_signal": "NO_TRADE",
                        "no_trade_reason": reason,
                        "reason_code": reason,
                        "confidence": None,
                        "predict_attempted": False,
                        "simulated_action": "NONE",
                        "route_error": False,
                        "missing_features": [],
                        "live_orders_enabled": False,
                        "broker_orders_enabled": False,
                    }
                    if reason:
                        err_dec["error"] = reason
                    results.append(err_dec)
                    self._record_decision(cid, err_dec, ts, market_snapshot)
                    continue

                candle_note = ""
                if not has_candles and has_chain:
                    candle_note = "candle_missing_or_warmup;option_chain_ok"
                elif not has_candles and not has_chain:
                    candle_note = "candles_empty;option_chain_empty"

                required = _load_required_features(cand)
                feat_t0 = time.perf_counter()
                try:
                    feature_frame, missing, feature_debug = build_paper_forward_feature_frame(cand, market_snapshot)
                except Exception as e:
                    feature_frame = None
                    missing = required[:]
                    feature_debug = {
                        "required_features_count": len(required),
                        "available_features_count": 0,
                        "missing_features": missing,
                        "feature_build_error": str(e),
                        "option_rows": _count_option_rows(option_chain_snapshot),
                        "candles_count": 0,
                        "coverage_pct": 0.0,
                    }
                    pf_log("WARN", f"[PAPER-FWD-FEATURES] cid={cid} feature_build_error={e}")
                feature_ms += (time.perf_counter() - feat_t0) * 1000.0

                artifact_ok = enabled and load_status == LOAD_OK and art_exists
                data_quality, predict_allowed = reconcile_paper_forward_readiness(
                    data_quality=data_quality,
                    candle_rows=candle_rows,
                    option_rows=option_rows,
                    spot=spot_val,
                    min_candles=min_candles,
                    missing_features=missing,
                    artifact_ok=artifact_ok,
                    candidate_id=cid,
                    old_quality=str(market_snapshot.get("data_quality_status") or ""),
                )
                reason = ""
                if predict_allowed and data_quality == "DATA_OK" and not missing and artifact_ok:
                    reason = ""
                elif data_quality == "TOKEN_NOT_VERIFIED":
                    reason = "token_not_verified"
                elif data_quality == "SESSION_EXPIRED":
                    reason = "session_expired"
                elif data_quality == "AUTH_FAILED":
                    reason = "broker_auth_failed"
                elif data_quality == "BROKER_CONFIG_MISSING":
                    reason = "broker_config_missing"
                elif data_quality == "OPTION_CHAIN_FETCH_ERROR":
                    reason = "option_chain_fetch_error"
                elif data_quality == "THIN_OPTION_CHAIN":
                    reason = "thin_option_chain"
                elif required and missing:
                    reason = "feature_vector_missing_columns"
                elif data_quality == "OPTION_CHAIN_EMPTY":
                    broker = str(market_snapshot.get("broker_name") or "").lower()
                    if broker in ("mstock", "m.stock", "m_stock", "mstocks"):
                        try:
                            from synthetic_option_chain import is_synthetic_chain_enabled
                            if is_synthetic_chain_enabled(broker):
                                if market_snapshot.get("spot") or market_snapshot.get("price"):
                                    reason = "synthetic_chain_ready" if has_chain else "waiting_for_candles"
                                else:
                                    reason = "WAITING_FOR_SPOT"
                            else:
                                reason = "option_chain_empty"
                        except Exception:
                            reason = "option_chain_empty"
                    else:
                        reason = "option_chain_empty"
                elif data_quality in ("SYNTHETIC_CHAIN_PENDING",):
                    reason = "synthetic_chain_ready"
                elif data_quality in ("WAITING_FOR_CANDLES", "SYNTHETIC_CHAIN_READY_WAITING_FOR_CANDLES") and not has_candles:
                    reason = "" if has_chain and spot_val and not cand.get("_requires_candles") else "waiting_for_candles"
                elif not has_candles:
                    reason = "" if has_chain and spot_val and not cand.get("_requires_candles") else "waiting_for_candles"
                elif data_quality in _PAPER_READY_QUALITIES:
                    reason = "" if (has_candles or (has_chain and spot_val and not cand.get("_requires_candles"))) else "waiting_for_candles"
                elif data_quality == "READY_FOR_PREDICTION":
                    reason = "" if (has_candles or (has_chain and spot_val and not cand.get("_requires_candles"))) else "waiting_for_candles"
                elif data_quality in ("WAITING_FOR_MSTOCK_EXCHANGE", "WAITING_FOR_MSTOCK_EXPIRY", "WAITING_FOR_MSTOCK_CONFIG"):
                    reason = data_quality
                elif data_quality in ("OPTION_CHAIN_EMPTY_RESPONSE", "EMPTY_RESPONSE", "EMPTY"):
                    broker = str(market_snapshot.get("broker_name") or "").lower()
                    chain_src = str(market_snapshot.get("chain_source") or market_snapshot.get("option_chain_source") or "")
                    if broker in ("mstock", "m.stock", "m_stock", "mstocks") or chain_src == "BLACK_SCHOLES_SYNTHETIC":
                        reason = "waiting_for_candles" if has_chain else (
                            "WAITING_FOR_SPOT" if not (market_snapshot.get("spot") or market_snapshot.get("price")) else "synthetic_chain_ready"
                        )
                    else:
                        reason = "WAITING_FOR_OPTION_CHAIN_FETCH"
                elif "WAITING_FOR_OPTION_CHAIN" in str(data_quality or "").upper():
                    broker = str(market_snapshot.get("broker_name") or "").lower()
                    if broker in ("mstock", "m.stock", "m_stock", "mstocks"):
                        reason = "synthetic_chain_ready" if has_chain else "WAITING_FOR_SPOT"
                    else:
                        reason = "WAITING_FOR_OPTION_CHAIN_FETCH"
                elif "SCRIPMASTER" in str(data_quality or "").upper() or "NO_MATCHING_EXPIRY" in str(data_quality or "").upper():
                    reason = "SCRIPMASTER_NO_MATCHING_EXPIRY"
                elif data_quality in ("SPOT_MISSING", "WAITING_FOR_SPOT") or not (market_snapshot.get("spot") or market_snapshot.get("price") or market_snapshot.get("close") or market_snapshot.get("last")):
                    reason = "WAITING_FOR_SPOT"
                elif data_quality in ("CANDLES_MISSING", "CANDLES_MISSING_NONFATAL"):
                    reason = "" if has_chain and (market_snapshot.get("spot") or market_snapshot.get("price")) else "candles_empty"
                elif data_quality and data_quality != "DATA_OK":
                    reason = "data_not_ready"  # last resort; prefer exact above
                elif required and missing:
                    min_cov = float(os.getenv("PAPER_FORWARD_MIN_FEATURE_COVERAGE", "95.0") or 95.0)
                    coverage_pct = float(feature_debug.get("coverage_pct") or 0.0)
                    if coverage_pct < min_cov:
                        reason = "feature_vector_missing_columns"
                elif cand.get("_requires_candles") and not has_candles:
                    reason = "candles_empty"

                chain_source = str(market_snapshot.get("chain_source") or market_snapshot.get("option_chain_source") or "")
                if reason:
                    log_paper_fwd_features(
                        candidate_id=cid,
                        required=len(required),
                        available=int(feature_debug.get("available_features_count") or 0),
                        missing=list(missing),
                        nan=int(feature_debug.get("feature_nan_count") or 0),
                        zero=int(feature_debug.get("feature_zero_count") or 0),
                        coverage=float(feature_debug.get("coverage_pct") or 0),
                    )
                    log_paper_fwd_decision(
                        candidate_id=cid,
                        signal="NO_TRADE",
                        reason=reason,
                        threshold=cand.get("threshold"),
                    )
                    no_trade_count += 1
                    err_dec = {
                        "snapshot_id": snap_id,
                        "timestamp": ts,
                        "candidate_id": cid,
                        "paper_forward_only": True,
                        "shadow_ready": False,
                        "forced_eval": True,
                        "final_signal": "NO_TRADE",
                        "no_trade_reason": reason,
                        "reason_code": reason,
                        "confidence": None,
                        "predict_attempted": False,
                        "simulated_action": "NONE",
                        "route_error": False,
                        "block_reason": reason,
                        "allowed": False,
                        "model_type": str(cand.get("model_name") or "unknown"),
                        "feature_missing_count": len(missing or []),
                        "feature_invalid_count": int(feature_debug.get("feature_invalid_count") or feature_debug.get("feature_nan_count") or 0),
                        "_skip_eval_count": reason in (
                            "token_not_verified",
                            "session_expired",
                            "broker_auth_failed",
                            "broker_config_missing",
                            "option_chain_fetch_error",
                            "candles_empty",
                            "thin_option_chain",
                            "WAITING_FOR_SPOT",
                            "waiting_for_candles",
                            "synthetic_chain_ready",
                            "warmup_wait",
                            "stale_snapshot",
                            "data_not_ready",
                            "WAITING_FOR_MSTOCK_EXCHANGE",
                            "WAITING_FOR_OPTION_CHAIN_FETCH",
                            "SCRIPMASTER_NO_MATCHING_EXPIRY",
                        ),
                        "missing_features": missing[:10],
                        "live_orders_enabled": False,
                        "broker_orders_enabled": False,
                        "debug": {
                            "candle_state": candle_note,
                            "has_chain": has_chain,
                            "has_candles": has_candles,
                            "feature_contract": slim_feature_debug(feature_debug),
                            "data_quality_status": data_quality,
                            "option_rows": option_rows,
                            "predict_allowed": False,
                        },
                    }
                    self._export_decision_buffer.append({"candidate_id": cid, "decision": err_dec, "feature_debug": feature_debug})
                    results.append(err_dec)
                    self._record_decision(cid, err_dec, ts, market_snapshot)
                    continue

                try:
                    if route_candidate_decision is None:
                        raise RuntimeError("route_candidate_decision not available")

                    log_paper_fwd_features(
                        candidate_id=cid,
                        required=len(required),
                        available=int(feature_debug.get("available_features_count") or 0),
                        missing=list(missing),
                        nan=int(feature_debug.get("feature_nan_count") or 0),
                        zero=int(feature_debug.get("feature_zero_count") or 0),
                        coverage=float(feature_debug.get("coverage_pct") or 0),
                    )
                    pred_t0 = time.perf_counter()
                    route_snapshot = dict(market_snapshot)
                    try:
                        if feature_frame is not None and not feature_frame.empty:
                            enriched_features = feature_frame.iloc[0].to_dict()
                            for fk, fv in enriched_features.items():
                                if fv is None:
                                    continue
                                if isinstance(fv, float) and math.isnan(fv):
                                    continue
                                route_snapshot[fk] = fv
                            route_snapshot["feature_coverage_pct"] = feature_debug.get("coverage_pct")
                            route_snapshot["missing_features"] = list(missing)
                    except Exception:
                        route_snapshot = dict(market_snapshot)

                    dec = route_candidate_decision(
                        market_snapshot=route_snapshot,
                        option_chain_snapshot=dict(option_chain_snapshot) if option_chain_snapshot and not isinstance(option_chain_snapshot, list) else option_chain_snapshot,
                        legacy_signal=None,
                        mode="paper",
                        active_candidate_id=cid,
                        candidate_dir=art_dir,
                        force_eval=True,  # allowed for paper_forward_only observation
                    )
                    dec = dict(dec or {})
                    dec["snapshot_id"] = snap_id
                    dec["timestamp"] = ts
                    dec["candidate_id"] = cid  # configured stable (BLOCKER 4)
                    dec["artifact_id"] = artifact_id
                    dec["artifact_dir"] = art_dir
                    dec["paper_forward_only"] = True
                    dec["shadow_ready"] = False
                    dec["forced_eval"] = True
                    dec["live_orders_enabled"] = False
                    dec["broker_orders_enabled"] = False
                    dec["predict_attempted"] = True
                    dec["route_error"] = False
                    if candle_note:
                        dec["debug"] = dec.get("debug", {})
                        dec["debug"]["candle_state"] = candle_note
                    dec.setdefault("debug", {})
                    dec["debug"]["feature_contract"] = slim_feature_debug(feature_debug)
                    dec["debug"]["data_quality_status"] = data_quality
                    dec["cost_quality"] = feature_debug.get("cost_quality", "APPROX")
                    dec["cost_approx_flags"] = feature_debug.get("cost_approx_flags", [])
                    if pf_verbose_decision() and pf_should_log_level("TRACE"):
                        pf_log(
                            "TRACE",
                            f"[PAPER-FWD-DECISION-OBJ] cid={cid} keys={len(dec)}",
                            rate_key=f"decision_obj:{cid}",
                            rate_interval=60.0,
                        )
                    self._export_decision_buffer.append({"candidate_id": cid, "decision": dec, "feature_debug": feature_debug})
                    model_type = "unknown"
                    try:
                        model_path = (cand.get("_resolve") or {}).get("files", {}).get("model_pkl")
                        model_type = Path(str(model_path)).suffix or "artifact"
                    except Exception:
                        pass
                    pdbg = (dec.get("debug") or {}).get("predict") or {}
                    raw_out = pdbg.get("raw", dec.get("raw_output", dec.get("confidence")))
                    scaler_ap = bool(pdbg.get("scaler_applied", dec.get("debug", {}).get("scaler_applied", False)))
                    cal_ap = bool(pdbg.get("calibrator_applied", dec.get("debug", {}).get("calibrator_applied", False)))
                    conf_val = dec.get("confidence")
                    dec.setdefault("block_reason", dec.get("no_trade_reason") or dec.get("reason_code") or "")
                    if "allowed" not in dec:
                        dec["allowed"] = all(
                            dec.get(flag_name) is not False
                            for flag_name in (
                                "allowed_by_model",
                                "allowed_by_side_policy",
                                "allowed_by_liquidity",
                                "allowed_by_cost",
                                "allowed_by_risk",
                            )
                        )
                    dec.setdefault("model_type", str(cand.get("model_name") or model_type or "unknown"))
                    dec.setdefault("feature_missing_count", len(missing or []))
                    dec.setdefault(
                        "feature_invalid_count",
                        int(feature_debug.get("feature_invalid_count") or feature_debug.get("feature_nan_count") or 0),
                    )
                    predict_ms += (time.perf_counter() - pred_t0) * 1000.0
                    log_paper_fwd_predict(
                        candidate_id=cid,
                        model=str(cand.get("model_name", model_type)),
                        raw_output=raw_out,
                        confidence=conf_val,
                        threshold=dec.get("threshold"),
                        data_quality=data_quality,
                        predict_allowed=True,
                        scaler_applied=scaler_ap,
                        calibrator_applied=cal_ap,
                        missing_count=len(missing),
                        feature_debug=feature_debug,
                    )

                    # If router gave generic, upgrade to more precise when we know inputs were thin
                    ntr = dec.get("no_trade_reason") or ""
                    if not has_chain and "candidate" not in ntr:
                        # leave as is; router will have said low_conf or feature etc.
                        pass

                    # Update per-candidate simulated state
                    state_update = self.update_candidate_state(cid, dec, market_snapshot)
                    dec.update(state_update)

                    results.append(dec)
                    self._record_decision(cid, dec, ts, market_snapshot)
                    sig = str(dec.get("final_signal") or "NO_TRADE")
                    if sig.startswith("BUY_"):
                        signal_count += 1
                    else:
                        no_trade_count += 1
                    _allowed_parts = []
                    if dec.get("allowed_by_model") is not False:
                        _allowed_parts.append("model")
                    if dec.get("allowed_by_side_policy") is not False:
                        _allowed_parts.append("side")
                    if dec.get("allowed_by_liquidity") is not False:
                        _allowed_parts.append("liq")
                    if dec.get("allowed_by_cost") is not False:
                        _allowed_parts.append("cost")
                    if dec.get("allowed_by_risk") is not False:
                        _allowed_parts.append("risk")
                    log_paper_fwd_decision(
                        candidate_id=cid,
                        signal=sig,
                        confidence=dec.get("confidence"),
                        threshold=dec.get("threshold"),
                        reason=str(dec.get("no_trade_reason") or dec.get("reason_code") or ""),
                        side=str(state_update.get("selected_option_type") or dec.get("side_decision") or ""),
                        pos=str(state_update.get("position_status") or ""),
                        strike=state_update.get("selected_strike") or dec.get("selected_strike"),
                        opt_type=state_update.get("selected_option_type") or dec.get("selected_option_type"),
                        allowed=",".join(_allowed_parts) or "-",
                    )
                    log_paper_fwd_candidate(
                        candidate_id=cid,
                        signal=sig,
                        confidence=dec.get("confidence"),
                        reason=str(dec.get("no_trade_reason") or dec.get("reason_code") or ""),
                        pos=str(state_update.get("position_status") or ""),
                        strike=state_update.get("selected_strike") or dec.get("selected_strike"),
                        opt_type=state_update.get("selected_option_type") or dec.get("selected_option_type"),
                    )
                    chain_is_synth = _is_paper_synthetic_mode(market_snapshot)
                    if chain_is_synth and sig.startswith("BUY_"):
                        pf_log(
                            "INFO",
                            f"[LIVE-ORDER-GUARD] blocked reason=synthetic_data cid={cid}",
                            rate_key=f"live_order_guard:{cid}",
                            rate_interval=60.0,
                        )
                    dec["live_orders_enabled"] = False
                    dec["broker_orders_enabled"] = False
                    if sig.startswith("BUY_"):
                        pf_log(
                            "DEBUG",
                            f"[ORDER-GUARD] paper_only=true live_orders=false cid={cid}",
                            rate_key=f"order_guard:{cid}",
                            rate_interval=60.0,
                        )
                except Exception as e:
                    tb = traceback.format_exc()
                    detail = self._log_route_exception(cand, market_snapshot, option_chain_snapshot, required, feature_debug, e, tb)
                    log_paper_fwd_decision(
                        candidate_id=cid,
                        signal="NO_TRADE",
                        reason=f"model_route_exception:{e}",
                    )
                    log_paper_fwd_candidate(
                        candidate_id=cid,
                        signal="NO_TRADE",
                        reason="model_route_exception",
                    )
                    # structured safe decision, never let exception escape per-candidate
                    err_dec = {
                        "snapshot_id": snap_id,
                        "timestamp": ts,
                        "candidate_id": cid,
                        "paper_forward_only": True,
                        "shadow_ready": False,
                        "forced_eval": True,
                        "final_signal": "NO_TRADE",
                        "no_trade_reason": "model_route_exception",
                        "reason_code": "model_route_exception",
                        "confidence": None,
                        "predict_attempted": False,
                        "simulated_action": "NONE",
                        "route_error": True,
                        "error": str(e),
                        "exception_type": type(e).__name__,
                        "live_orders_enabled": False,
                        "broker_orders_enabled": False,
                        "debug": {"tb": tb, "route_exception": detail, "candle_state": candle_note},
                    }
                    self._state.setdefault(cid, {})["error_count"] = self._state.get(cid, {}).get("error_count", 0) + 1
                    self._route_errors += 1
                    results.append(err_dec)
                    self._record_decision(cid, err_dec, ts, market_snapshot)
                    error_count += 1
        cost_qualities = [
            str((r.get("cost_quality") or (r.get("debug") or {}).get("feature_contract", {}).get("cost_quality") or "APPROX"))
            for r in results
        ]
        ltp_sources = [
            normalize_ltp_source(str(r.get("mark_source") or r.get("pnl_source") or ""))
            for r in results
            if r.get("mark_source") or r.get("pnl_source")
        ]
        readiness = summarize_readiness(
            candidates_total=len(self.candidates),
            results=results,
            cost_qualities=cost_qualities,
            ltp_sources=ltp_sources,
            data_quality=data_quality,
        )
        self.last_readiness_summary = readiness
        print(format_readiness_log(readiness))

        cycle_ms = (time.perf_counter() - cycle_t0) * 1000.0
        log_paper_fwd_summary(
            candidates=len(self.candidates),
            signals=signal_count,
            no_trade=no_trade_count,
            errors=error_count,
            elapsed_ms=int(cycle_ms),
        )
        log_pf_perf(
            cycle_ms=round(cycle_ms, 1),
            snapshot_ms=0,
            feature_ms=round(feature_ms, 1),
            predict_ms=round(predict_ms, 1),
            ui_ms=0,
            log_ms=0,
            candidates=len(self.candidates),
        )
        self._export_decision_buffer = self._export_decision_buffer[-50:]
        pf_log_perf_summary()
        return results

    def update_candidate_state(self, candidate_id: str, decision: dict, market_snapshot: dict) -> dict:
        self._ensure_candidate_state_containers()
        try:
            from synthetic_option_chain import atm_strike, load_bs_config, load_paper_sim_config
            bs_cfg = load_bs_config()
            sim_cfg = load_paper_sim_config()
        except Exception:
            bs_cfg = {"strike_step": 50}
            sim_cfg = {}

        default_lot_size, default_qty_source = _paper_forward_default_lot_size()
        default_state = {
            "open_position": False, "side": None, "entry_time": None, "entry_price": None,
            "entry_strike": None, "entry_symbol": "", "current_price": None, "spot_price": None,
            "selected_strike": None, "selected_option_type": "", "selected_symbol": "",
            "direction": "BUY", "lots": 1, "lot_size": default_lot_size, "qty": default_lot_size,
            "position_qty": default_lot_size, "qty_source": default_qty_source,
            "realized_pnl": 0.0, "unrealized_pnl": 0.0,
            "total_entries": 0, "total_exits": 0, "total_trades": 0,
            "wins": 0, "losses": 0, "max_drawdown": 0.0,
            "last_signal": None, "last_eval_signal": None, "last_confidence": None,
            "last_entry_signal": None, "last_entry_ts": None, "last_exit_ts": None,
            "last_action_ts": None, "last_no_trade_reason": "",
            "selected_preset": "", "confidence": 0.0, "threshold": 0.0,
            "error_count": 0, "last_update": None, "pnl_source": "",
            "synthetic_mode": False, "simulated_action": "", "paper_action": "",
        }
        st = self._state.setdefault(candidate_id, dict(default_state))
        for key, value in default_state.items():
            st.setdefault(key, value)

        final_sig = str(decision.get("final_signal", decision.get("side_decision", "NO_TRADE")) or "NO_TRADE")
        prev_eval_signal = st.get("last_eval_signal")
        prev_confidence = st.get("last_confidence")
        spot_px = float(
            market_snapshot.get("spot")
            or market_snapshot.get("price")
            or market_snapshot.get("last")
            or market_snapshot.get("close")
            or 0.0
        )
        is_synth = _is_paper_synthetic_mode(market_snapshot)
        st["spot_price"] = spot_px
        st["synthetic_mode"] = is_synth
        now_iso = datetime.now(timezone.utc).isoformat()
        now_epoch = time.time()
        st["last_update"] = now_iso
        st["selected_preset"] = decision.get("selected_preset", st.get("selected_preset", ""))
        st["confidence"] = decision.get("confidence", decision.get("probability", st.get("confidence", 0.0)))
        st["threshold"] = decision.get("threshold", decision.get("effective_threshold", st.get("threshold", 0.0)))
        if decision.get("cost_quality"):
            st["cost_quality"] = decision.get("cost_quality")
        st["last_signal"] = final_sig

        cs = self._get_or_create_candidate_state(candidate_id)
        cs.final_signal = final_sig
        cs.last_eval_ts = now_iso
        cs.confidence = st["confidence"]
        cs.threshold = st["threshold"]
        qty = _positive_int(st.get("position_qty") or st.get("qty")) or _paper_forward_default_lot_size()[0]
        st["position_qty"] = qty
        st["qty"] = qty
        st.setdefault("lots", 1)
        st.setdefault("lot_size", qty)
        st.setdefault("direction", "BUY")

        def _apply_contract_to_state(contract: dict) -> None:
            if not contract:
                return
            sym = str(contract.get("trading_symbol") or contract.get("symbol") or "")
            tok = _option_token_from_row(contract)
            if sym:
                st["entry_symbol"] = sym
                st["selected_symbol"] = sym
                cs.entry_symbol = sym
            if tok:
                st["entry_token"] = tok
                st["symbolToken"] = tok
                st["security_id"] = tok
            if contract.get("strike_price") not in (None, ""):
                st["entry_strike"] = contract.get("strike_price")
                st["selected_strike"] = contract.get("strike_price")
                cs.entry_strike = contract.get("strike_price")
            if contract.get("option_type"):
                st["selected_option_type"] = contract.get("option_type")
                cs.option_type = str(contract.get("option_type"))
            qty_i, lot_size_i, lots_i, qty_source = _resolve_paper_position_qty(contract, st.get("lots") or 1)
            st["qty"] = qty_i
            st["position_qty"] = qty_i
            st["lot_size"] = lot_size_i
            st["lots"] = lots_i
            st["qty_source"] = qty_source
            cs.qty = qty_i
            cs.lot_size = lot_size_i
            cs.lots = lots_i
            cs.direction = "BUY"

        def _resolve_option_mark(side: str, *, strike: Optional[float] = None) -> Tuple[float, str, str, Optional[float], dict]:
            try:
                from synthetic_option_chain import load_paper_mark_config
                mark_cfg = load_paper_mark_config()
            except Exception:
                mark_cfg = {"use_broker_option_prices": True, "broker_price_fallback_synthetic": True}

            if bool(mark_cfg.get("use_broker_option_prices", True)):
                bpx, bsym, bsrc, bstrike, bcontract = _broker_option_mark(
                    market_snapshot,
                    side,
                    strike=strike,
                    entry_symbol=str(st.get("entry_symbol") or ""),
                    spot=spot_px,
                    strike_step=int(bs_cfg.get("strike_step", 50)),
                    candidate_id=candidate_id,
                )
                if bpx is not None and bpx > 0:
                    _apply_contract_to_state(bcontract)
                    prev_px, prev_src, prev_stale, _prev = _last_good_mark(
                        bsym or st.get("entry_symbol", ""),
                        _option_token_from_row(bcontract),
                        str((bcontract or {}).get("exchange") or "NFO"),
                    )
                    _record_last_good_mark(
                        symbol=bsym or st.get("entry_symbol", ""),
                        token=_option_token_from_row(bcontract),
                        exchange=str((bcontract or {}).get("exchange") or "NFO"),
                        price=float(bpx),
                        source=bsrc,
                    )
                    _log_mark_source(
                        bsym or st.get("entry_symbol", ""),
                        bsrc,
                        prev_src,
                        bpx,
                        prev_stale,
                        "broker_or_chain_mark",
                    )
                    return float(bpx), bsym, bsrc, bstrike, bcontract

            lg_px, lg_src, lg_stale, lg = _last_good_mark(str(st.get("entry_symbol") or ""), str(st.get("entry_token") or st.get("symbolToken") or ""), "NFO")
            if lg_px and last_good_mark_allowed(lg, _LAST_GOOD_MARK_TTL_SEC):
                sym_lg = str(lg.get("symbol") or st.get("entry_symbol") or "")
                _log_mark_source(sym_lg, "LAST_GOOD_MARK", lg_src, lg_px, False, "broker_ltp_temporarily_unavailable")
                return float(lg_px), sym_lg, "LAST_GOOD_MARK", strike, {
                    "trading_symbol": sym_lg,
                    "symbol": sym_lg,
                    "token": lg.get("token"),
                    "symbolToken": lg.get("token"),
                    "security_id": lg.get("token"),
                    "exchange": lg.get("exchange") or "NFO",
                    "strike_price": strike or st.get("entry_strike"),
                    "option_type": side,
                }

            if is_synth and bool(mark_cfg.get("broker_price_fallback_synthetic", True)):
                px, sym, src = _synthetic_option_mark(market_snapshot, side, strike=strike)
                if px is not None and px > 0:
                    selected_src = "STALE_SYNTHETIC" if lg_px and lg_stale else "SYNTHETIC_MARK"
                    _log_mark_source(sym or st.get("entry_symbol", ""), selected_src, lg_src, px, bool(lg_stale), "no_live_mark_available")
                    used_strike = strike
                    if used_strike is None:
                        try:
                            used_strike = atm_strike(spot_px, int(bs_cfg.get("strike_step", 50)))
                        except Exception:
                            used_strike = None
                    chain = _chain_rows_for_marking(market_snapshot)
                    row = _match_option_chain_row(
                        chain, side=side, strike=used_strike, spot=spot_px,
                        strike_step=int(bs_cfg.get("strike_step", 50)), entry_symbol=sym,
                    )
                    contract = _contract_payload_from_row(row or {}, client=market_snapshot.get("broker_client"), candidate_id=candidate_id) if row else _enrich_contract_tokens({"trading_symbol": sym, "symbol": sym, "exchange": "NFO"}, market_snapshot.get("broker_client"))
                    _apply_contract_to_state(contract)
                    return float(px), str(contract.get("trading_symbol") or sym), selected_src, used_strike, contract
            elif not is_synth:
                px, sym, src = _synthetic_option_mark(market_snapshot, side, strike=strike)
                if px is not None and px > 0:
                    used_strike = strike
                    return float(px), sym, "CHAIN_MARK", used_strike, {}
            return 0.0, "", "SYNTHETIC_MARK", None, {}

        def _set_paper_reason(reason: str) -> str:
            st["last_no_trade_reason"] = reason
            cs.reason_code = reason
            cs.raw_reason = reason
            return reason

        def _mark_open_position() -> Tuple[float, float]:
            side = st.get("side") or "CE"
            mark_px, sym, pnl_src, _, _ = _resolve_option_mark(side, strike=st.get("entry_strike"))
            entry_p = float(st.get("entry_price") or mark_px or 0)
            if mark_px <= 0:
                mark_px = float(st.get("current_price") or st.get("option_current_price") or entry_p or 0.0)
                pnl_src = st.get("pnl_source") or "LAST_GOOD_MARK"
            qty_i = _positive_int(st.get("position_qty") or st.get("qty") or (cs.qty if cs else None)) or _paper_forward_default_lot_size()[0]
            unreal = _compute_option_pnl(entry_p, mark_px, qty_i, side=st.get("direction") or "BUY")
            price_diff = float(mark_px) - float(entry_p)
            st["current_price"] = mark_px
            st["option_current_price"] = mark_px
            st["selected_strike"] = st.get("entry_strike")
            st["selected_option_type"] = side
            st["selected_symbol"] = st.get("entry_symbol", sym)
            st["unrealized_pnl"] = unreal
            st["price_diff"] = price_diff
            st["pnl_formula"] = "(current_option_price - entry_price) * qty"
            st["qty"] = qty_i
            st["position_qty"] = qty_i
            st["pnl_source"] = pnl_src or st.get("pnl_source") or "SYNTHETIC_MARK"
            st["last_mark_time"] = datetime.now(timezone.utc).isoformat()
            cs.current_price = mark_px
            cs.unreal_pnl = unreal
            cs.qty = qty_i
            cs.pnl_source = st["pnl_source"]
            cs.pos = f"OPEN_{side}"
            cs.side = side
            cs.entry_strike = st.get("entry_strike")
            cs.option_type = side
            cs.entry_symbol = st.get("entry_symbol", sym)
            mark_reason = "broker_token" if cs.pnl_source == "BROKER_LTP" else (
                "real_chain_row" if cs.pnl_source == "REAL_CHAIN_LTP" else "synthetic_fallback"
            )
            print(
                f"[PF-PAPER-MARK] candidate_id={candidate_id} symbol={st.get('entry_symbol', sym)} "
                f"source={cs.pnl_source} reason={mark_reason} "
                f"entry_price={entry_p:.4f} current_option_price={mark_px:.4f} spot={spot_px:.2f} "
                f"unreal={unreal:.4f}"
            )
            print(
                f"[PF-PNL] candidate_id={candidate_id} symbol={st.get('entry_symbol', sym)} "
                f"entry={entry_p:.2f} current={float(mark_px):.2f} qty={qty_i} "
                f"diff={price_diff:.2f} unreal={unreal:.2f}"
            )
            return mark_px, unreal

        # --- Open position: mark / exit / hold ---
        if st.get("open_position"):
            side = st.get("side") or "CE"
            if final_sig in ("BUY_CE", "BUY_PE"):
                sig_side = "CE" if "CE" in final_sig else "PE"
                if sig_side == side:
                    print(
                        f"[PF-DUPLICATE-ENTRY-BLOCKED] candidate_id={candidate_id} "
                        f"existing_side={side} signal={final_sig}"
                    )
                    self._duplicate_entry_blocked += 1
                    mark_px, unreal = _mark_open_position()
                    reason = _set_paper_reason("HOLD_EXISTING_POSITION")
                    st["last_action_ts"] = now_iso
                    st["last_eval_signal"] = final_sig
                    st["last_confidence"] = st["confidence"]
                    cs.entries = st.get("total_entries", cs.entries)
                    return {
                        "simulated_action": "HOLD_EXISTING_PAPER_POSITION",
                        "unrealized_pnl": unreal,
                        "current_price": mark_px,
                        "option_current_price": mark_px,
                        "spot_price": spot_px,
                        "no_trade_reason": reason,
                        "reason_code": reason,
                        "position_status": cs.pos,
                        "selected_strike": st.get("entry_strike"),
                        "selected_option_type": side,
                        "selected_symbol": st.get("entry_symbol", ""),
                        "total_entries": st.get("total_entries", 0),
                    }

            mark_px, unreal = _mark_open_position()
            entry_p = float(st.get("entry_price") or mark_px)
            should_exit, exit_reason = _evaluate_paper_exit(
                side=side,
                final_sig=final_sig,
                entry_p=entry_p,
                mark_px=mark_px,
                entry_epoch=_parse_ts_epoch(st.get("entry_time")),
                now_epoch=now_epoch,
                cfg=sim_cfg,
                is_synth=is_synth,
            )
            if should_exit and exit_reason:
                exit_qty = _positive_int(st.get("position_qty") or st.get("qty") or cs.qty) or _paper_forward_default_lot_size()[0]
                realized_delta = _compute_option_pnl(entry_p, mark_px, exit_qty, side=st.get("direction") or "BUY")
                st["open_position"] = False
                st["total_exits"] += 1
                st["total_trades"] += 1
                st["last_exit_ts"] = now_iso
                st["realized_pnl"] += realized_delta
                if realized_delta > 0:
                    st["wins"] += 1
                else:
                    st["losses"] += 1
                st["win_rate"] = st["wins"] / max(1, st["total_trades"])
                st["max_drawdown"] = min(st["max_drawdown"], st["realized_pnl"])
                st["unrealized_pnl"] = 0.0
                st["open_since"] = None
                st["selected_strike"] = None
                st["selected_option_type"] = ""
                st["selected_symbol"] = ""
                st["last_action_ts"] = now_iso
                reason = _set_paper_reason(exit_reason)
                cs.pos = "FLAT"
                cs.qty = 0
                cs.entry_strike = None
                cs.option_type = ""
                cs.realized_pnl = st["realized_pnl"]
                cs.unreal_pnl = 0.0
                cs.trades = st["total_trades"]
                cs.exits = st["total_exits"]
                cs.entries = st["total_entries"]
                cs.wins = st["wins"]
                cs.losses = st["losses"]
                cs.win_rate = st["win_rate"]
                cs.open_since = None
                cs.last_action = "EXIT"
                exit_tag = str(exit_reason).replace("PAPER_EXIT_", "")
                print(
                    f"[PF-PAPER-EXIT] candidate_id={candidate_id} reason={exit_tag} "
                    f"entry={entry_p:.4f} exit={mark_px:.4f} pnl={realized_delta:.4f}"
                )
                print(
                    f"[PF-EXIT-PNL] candidate_id={candidate_id} symbol={st.get('entry_symbol', '')} "
                    f"entry={entry_p:.2f} exit={float(mark_px):.2f} qty={exit_qty} realized={realized_delta:.2f}"
                )
                st["last_eval_signal"] = final_sig
                st["last_confidence"] = st["confidence"]
                return {
                    "simulated_action": "EXIT",
                    "sim_exit_price": mark_px,
                    "realized_pnl_delta": realized_delta,
                    "total_trades": cs.trades,
                    "total_exits": cs.exits,
                    "total_entries": cs.entries,
                    "no_trade_reason": reason,
                    "reason_code": reason,
                    "position_status": "FLAT",
                    "selected_strike": None,
                    "selected_option_type": "",
                    "selected_symbol": "",
                }

            reason = _set_paper_reason("HOLD_EXISTING_POSITION")
            st["last_action_ts"] = now_iso
            st["last_eval_signal"] = final_sig
            st["last_confidence"] = st["confidence"]
            return {
                "simulated_action": "HOLD",
                "unrealized_pnl": unreal,
                "current_price": mark_px,
                "option_current_price": mark_px,
                "spot_price": spot_px,
                "no_trade_reason": reason,
                "position_status": cs.pos,
                "selected_strike": st.get("entry_strike"),
                "selected_option_type": side,
                "selected_symbol": st.get("entry_symbol", ""),
                "total_entries": st.get("total_entries", 0),
            }

        # --- Flat: attempt new entry ---
        if final_sig in ("BUY_CE", "BUY_PE"):
            allowed, block_reason, remaining = _can_open_paper_entry(
                st,
                final_sig=final_sig,
                confidence=st["confidence"],
                threshold=st["threshold"],
                is_synth=is_synth,
                cfg=sim_cfg,
                now_epoch=now_epoch,
            )
            if not allowed:
                if block_reason == "ENTRY_COOLDOWN":
                    self._entry_cooldown_blocked += 1
                    print(f"[PF-ENTRY-COOLDOWN] candidate_id={candidate_id} remaining_sec={remaining:.0f}")
                elif block_reason == "SIGNAL_DEDUPE":
                    self._signal_dedupe_blocked += 1
                    print(
                        f"[PF-SIGNAL-DEDUPE] candidate_id={candidate_id} signal={final_sig} "
                        f"reason=repeated_signal prev={prev_eval_signal}"
                    )
                elif block_reason == "HOLD_EXISTING_POSITION":
                    self._duplicate_entry_blocked += 1
                reason = _set_paper_reason(block_reason)
                st["last_eval_signal"] = final_sig
                st["last_confidence"] = st["confidence"]
                return {
                    "simulated_action": "NONE",
                    "no_trade_reason": reason,
                    "reason_code": reason,
                    "total_entries": st.get("total_entries", 0),
                }

            side = "CE" if "CE" in final_sig else "PE"
            entry_px, sym, pnl_src, entry_strike, entry_contract = _resolve_option_mark(side)
            if entry_px <= 0:
                st["last_eval_signal"] = final_sig
                st["last_confidence"] = st["confidence"]
                return {"simulated_action": "NONE", "total_entries": st.get("total_entries", 0)}

            qty, lot_size, lots, qty_source = _resolve_paper_position_qty(entry_contract, lots=1)
            st["open_position"] = True
            st["side"] = side
            st["direction"] = "BUY"
            st["entry_time"] = now_iso
            st["open_since"] = now_iso
            st["last_entry_ts"] = now_iso
            st["last_action_ts"] = now_iso
            st["entry_price"] = entry_px
            st["qty"] = qty
            st["position_qty"] = qty
            st["lot_size"] = lot_size
            st["lots"] = lots
            st["qty_source"] = qty_source
            st["entry_symbol"] = str(entry_contract.get("trading_symbol") or sym)
            st["entry_strike"] = entry_strike
            st["selected_strike"] = entry_strike
            st["selected_option_type"] = side
            st["selected_symbol"] = str(entry_contract.get("trading_symbol") or sym)
            if entry_contract:
                tok = _option_token_from_row(entry_contract)
                if tok:
                    st["entry_token"] = tok
                    st["symbolToken"] = tok
                    st["security_id"] = tok
            st["current_price"] = entry_px
            st["option_current_price"] = entry_px
            st["total_entries"] += 1
            st["last_entry_signal"] = final_sig
            st["unrealized_pnl"] = 0.0
            st["pnl_source"] = pnl_src
            reason = _set_paper_reason("TRADE_OPENED_PAPER")
            cs.pos = f"OPEN_{side}"
            cs.side = side
            cs.qty = qty
            cs.lot_size = lot_size
            cs.lots = lots
            cs.direction = "BUY"
            cs.entry_price = entry_px
            cs.entry_symbol = sym
            cs.entry_strike = entry_strike
            cs.option_type = side
            cs.open_since = st["open_since"]
            cs.entries = st["total_entries"]
            cs.current_price = entry_px
            cs.unreal_pnl = 0.0
            cs.pnl_source = pnl_src
            cs.last_action = "ENTER"
            chain_label = "BLACK_SCHOLES_SYNTHETIC" if is_synth else "paper"
            print(
                f"[PF-PAPER-ENTRY] candidate_id={candidate_id} side={side} symbol={sym} "
                f"entry_price={entry_px:.4f} qty={qty} source={chain_label}"
            )
            print(
                f"[PF-POS-QTY] candidate_id={candidate_id} symbol={sym} "
                f"lots={lots} lot_size={lot_size} qty={qty} source={qty_source}"
            )
            st["last_eval_signal"] = final_sig
            st["last_confidence"] = st["confidence"]
            return {
                "simulated_action": "ENTER",
                "sim_entry_price": entry_px,
                "sim_side": side,
                "qty": qty,
                "position_qty": qty,
                "lot_size": lot_size,
                "lots": lots,
                "direction": "BUY",
                "total_entries": st["total_entries"],
                "entry_symbol": sym,
                "entry_strike": entry_strike,
                "selected_strike": entry_strike,
                "selected_option_type": side,
                "selected_symbol": sym,
                "open_since": st["open_since"],
                "pnl_source": pnl_src,
                "current_price": entry_px,
                "option_current_price": entry_px,
                "spot_price": spot_px,
                "no_trade_reason": reason,
                "reason_code": reason,
                "position_status": cs.pos,
            }

        no_trade_r = decision.get("no_trade_reason", decision.get("block_reason", ""))
        if no_trade_r:
            _set_paper_reason(str(no_trade_r))
        elif final_sig == "NO_TRADE":
            _set_paper_reason("NO_TRADE")
        st["last_eval_signal"] = final_sig
        st["last_confidence"] = st["confidence"]
        cs.entries = st.get("total_entries", cs.entries)
        cs.exits = st.get("total_exits", cs.exits)
        return {
            "simulated_action": "NONE",
            "total_entries": cs.entries,
            "total_exits": cs.exits,
            "spot_price": spot_px,
            "no_trade_reason": st.get("last_no_trade_reason", ""),
        }

    # TASK6: explicit paper position helpers (called or mirrored in update/apply)
    def _pf_open_paper_position(self, cid: str, side: str, price: float, ts: str = "") -> None:
        cs = self._get_or_create_candidate_state(cid)
        cs.pos = f"LONG_{side}" if side else "LONG"
        qty, lot_size, lots, _ = _resolve_paper_position_qty(None, lots=1)
        cs.qty = qty
        cs.lot_size = lot_size
        cs.lots = lots
        cs.direction = "BUY"
        cs.entry_price = price
        cs.current_price = price
        cs.unreal_pnl = 0.0
        cs.last_action = "ENTER"
        cs.last_eval_ts = ts or cs.last_eval_ts
        st = self._state.setdefault(cid, {})
        st["open_position"] = True
        st["side"] = side
        st["entry_price"] = price
        st["current_price"] = price
        st["qty"] = qty
        st["position_qty"] = qty
        st["lot_size"] = lot_size
        st["lots"] = lots
        st["direction"] = "BUY"
        st["unrealized_pnl"] = 0.0

    def _pf_update_paper_position_pnl(self, cid: str, price: float) -> None:
        cs = self._get_or_create_candidate_state(cid)
        if cs.pos == "FLAT" or not cs.entry_price:
            return
        entry = cs.entry_price
        qty = _positive_int(cs.qty) or _paper_forward_default_lot_size()[0]
        unreal = _compute_option_pnl(entry, price, qty, side=cs.direction or "BUY")
        cs.current_price = price
        cs.unreal_pnl = unreal
        st = self._state.setdefault(cid, {})
        st["current_price"] = price
        st["unrealized_pnl"] = unreal
        st["price_diff"] = float(price) - float(entry)
        st["pnl_formula"] = "(current_option_price - entry_price) * qty"

    def _pf_close_paper_position(self, cid: str, price: float, ts: str = "") -> None:
        cs = self._get_or_create_candidate_state(cid)
        if cs.pos == "FLAT" or not cs.entry_price:
            return
        entry = cs.entry_price
        qty = _positive_int(cs.qty) or _paper_forward_default_lot_size()[0]
        unreal = _compute_option_pnl(entry, price, qty, side=cs.direction or "BUY")
        cs.realized_pnl += unreal
        cs.trades += 1
        if unreal > 0:
            cs.wins += 1
        else:
            cs.losses += 1
        cs.win_rate = cs.wins / max(1, cs.trades) if cs.trades else 0
        cs.pos = "FLAT"
        cs.qty = 0
        cs.unreal_pnl = 0.0
        cs.last_action = "EXIT"
        cs.last_eval_ts = ts or cs.last_eval_ts
        st = self._state.setdefault(cid, {})
        st["open_position"] = False
        st["realized_pnl"] = cs.realized_pnl
        st["unrealized_pnl"] = 0.0
        st["total_trades"] = cs.trades
        st["wins"] = cs.wins
        st["losses"] = cs.losses
        st["win_rate"] = cs.win_rate
        st["max_drawdown"] = min(st.get("max_drawdown", 0), cs.realized_pnl)

    def _refresh_open_position_marks_from_latest_snapshot(self) -> None:
        """Refresh monitor current prices from the newest snapshot for open paper positions."""
        snapshot = dict(getattr(self, "_latest_market_snapshot", {}) or {})
        option_snapshot = getattr(self, "_latest_option_chain_snapshot", None)
        if option_snapshot is not None:
            snapshot.setdefault("_option_chain_snapshot", option_snapshot)
            if "option_chain" not in snapshot and isinstance(option_snapshot, (list, tuple)):
                snapshot["option_chain"] = list(option_snapshot)
        if not snapshot:
            return

        spot_px = _to_float(snapshot.get("spot") or snapshot.get("price") or snapshot.get("close") or snapshot.get("last")) or 0.0
        for cid, st in list(self._state.items()):
            cs = self._candidate_states.get(cid)
            cs_pos = str(cs.pos if cs else "").upper()
            is_open = bool(st.get("open_position")) or cs_pos.startswith("OPEN") or cs_pos.startswith("LONG")
            if not is_open:
                continue

            side = str(st.get("side") or (cs.side if cs else "") or (cs.option_type if cs else "") or "").upper()
            if side not in ("CE", "PE"):
                pos = str(cs.pos if cs else st.get("position_status", "")).upper()
                if "CE" in pos:
                    side = "CE"
                elif "PE" in pos:
                    side = "PE"
            if side not in ("CE", "PE"):
                continue

            strike = _to_float(st.get("entry_strike") or (cs.entry_strike if cs else None) or st.get("selected_strike"))
            entry_symbol = str(st.get("entry_symbol") or (cs.entry_symbol if cs else "") or st.get("selected_symbol") or "")
            mark_px, mark_sym, mark_src, used_strike, mark_contract = _broker_option_mark(
                snapshot,
                side,
                strike=strike,
                entry_symbol=entry_symbol,
                spot=spot_px,
                candidate_id=cid,
            )
            if not mark_px or mark_px <= 0:
                existing_px = _to_float(
                    st.get("option_current_price")
                    or st.get("current_price")
                    or (cs.current_price if cs else None)
                )
                if existing_px and existing_px > 0:
                    if cs:
                        cs.current_price = float(existing_px)
                    print(
                        f"[PF-PAPER-MARK-REFRESH] candidate_id={cid} status=KEEP_LAST_MARK "
                        f"entry_symbol={entry_symbol} strike={strike} side={side} current={float(existing_px):.2f}"
                    )
                    continue
                lg_px, lg_src, lg_stale, lg = _last_good_mark(
                    entry_symbol,
                    str(st.get("entry_token") or st.get("symbolToken") or ""),
                    "NFO",
                )
                if lg_px and last_good_mark_allowed(lg, _LAST_GOOD_MARK_TTL_SEC):
                    mark_px = lg_px
                    mark_sym = str(lg.get("symbol") or entry_symbol)
                    mark_src = "LAST_GOOD_MARK"
                    used_strike = strike
                    mark_contract = {
                        "trading_symbol": mark_sym,
                        "symbol": mark_sym,
                        "token": lg.get("token"),
                        "exchange": lg.get("exchange") or "NFO",
                    }
                    _log_mark_source(mark_sym, mark_src, lg_src, mark_px, False, "broker_ltp_temporarily_unavailable")
                else:
                    mark_px, mark_sym, mark_src = _synthetic_option_mark(snapshot, side, strike=strike)
                    used_strike = strike
                    mark_src = "STALE_SYNTHETIC" if lg_px and lg_stale else (mark_src or "SYNTHETIC_MARK")
                    _log_mark_source(mark_sym or entry_symbol, mark_src, lg_src, mark_px, bool(lg_stale), "no_live_mark_available")
            if not mark_px or mark_px <= 0:
                print(f"[PF-PAPER-MARK-REFRESH] candidate_id={cid} status=NO_MARK entry_symbol={entry_symbol} strike={strike} side={side}")
                continue

            qty = _positive_int(st.get("position_qty") or st.get("qty") or (cs.qty if cs else None)) or _paper_forward_default_lot_size()[0]
            if not st.get("lot_size") or not st.get("lots"):
                _qty, lot_size, lots, qty_source = _resolve_paper_position_qty(mark_contract or None, lots=st.get("lots") or 1)
                if not st.get("position_qty") and not st.get("qty"):
                    qty = _qty
                st["lot_size"] = lot_size
                st["lots"] = lots
                st["qty_source"] = qty_source
                if cs:
                    cs.lot_size = lot_size
                    cs.lots = lots
            entry_px = _to_float(st.get("entry_price") or (cs.entry_price if cs else None)) or float(mark_px)
            unreal = _compute_option_pnl(entry_px, float(mark_px), qty, side=st.get("direction") or "BUY")
            price_diff = float(mark_px) - float(entry_px)

            st["open_position"] = True
            st["side"] = side
            st["direction"] = st.get("direction") or "BUY"
            st["qty"] = qty
            st["position_qty"] = qty
            st["current_price"] = float(mark_px)
            st["option_current_price"] = float(mark_px)
            st["unrealized_pnl"] = unreal
            st["price_diff"] = price_diff
            st["pnl_formula"] = "(current_option_price - entry_price) * qty"
            st["pnl_source"] = mark_src or st.get("pnl_source") or "LATEST_MARK"
            st["last_mark_time"] = datetime.now(timezone.utc).isoformat()
            st["last_update"] = datetime.now(timezone.utc).isoformat()
            if spot_px > 0:
                st["spot_price"] = spot_px
            if used_strike is not None:
                st["entry_strike"] = used_strike
                st["selected_strike"] = used_strike
            st["selected_option_type"] = side
            if mark_contract:
                sym_use = str(mark_contract.get("trading_symbol") or mark_sym or entry_symbol)
                st["entry_symbol"] = sym_use
                st["selected_symbol"] = sym_use
                tok = _option_token_from_row(mark_contract)
                if tok:
                    st["entry_token"] = tok
                    st["symbolToken"] = tok
                    st["security_id"] = tok
            elif mark_sym or entry_symbol:
                st["entry_symbol"] = entry_symbol or mark_sym
                st["selected_symbol"] = entry_symbol or mark_sym

            if cs:
                cs.pos = f"OPEN_{side}"
                cs.side = side
                cs.option_type = side
                cs.qty = qty
                cs.direction = st["direction"]
                cs.current_price = float(mark_px)
                cs.unreal_pnl = unreal
                cs.pnl_source = st["pnl_source"]
                cs.last_eval_ts = st["last_update"]
                if not cs.entry_price and entry_px:
                    cs.entry_price = entry_px
                if used_strike is not None:
                    cs.entry_strike = used_strike
                if entry_symbol or mark_sym:
                    cs.entry_symbol = entry_symbol or mark_sym

            print(
                f"[PF-PAPER-MARK-REFRESH] candidate_id={cid} entry={entry_px:.2f} "
                f"cur={float(mark_px):.2f} source={st.get('pnl_source')} symbol={st.get('selected_symbol', '')}"
            )
            print(
                f"[PF-PNL] candidate_id={cid} symbol={st.get('selected_symbol', '')} "
                f"entry={entry_px:.2f} current={float(mark_px):.2f} qty={qty} "
                f"diff={price_diff:.2f} unreal={unreal:.2f}"
            )

    def get_status_table(self) -> List[dict]:
        self._ensure_candidate_state_containers()
        self._refresh_open_position_marks_from_latest_snapshot()
        rows = []
        for c in self.candidates:
            cid = c["candidate_id"]
            st = self._state.get(cid, {})
            load_st = c.get("_load_status", "")
            cs = self._candidate_states.get(cid)
            if cs:
                # prefer canonical runtime state (TASK2/3)
                st["last_signal"] = cs.final_signal or st.get("last_signal")
                st["confidence"] = cs.confidence if cs.confidence is not None else st.get("confidence")
                st["threshold"] = cs.threshold or st.get("threshold")
                st["predict_attempted"] = cs.predict_attempted
                st["last_no_trade_reason"] = cs.reason_code or st.get("last_no_trade_reason")
                st["last_update"] = cs.last_eval_ts or st.get("last_update")
                st["current_price"] = cs.current_price or st.get("current_price")
                st["open_position"] = bool(st.get("open_position")) or str(cs.pos or "").startswith("OPEN")
                st["side"] = cs.side or st.get("side")
                if cs.qty:
                    st["qty"] = cs.qty
                    st["position_qty"] = cs.qty
                st["lot_size"] = cs.lot_size or st.get("lot_size") or 65
                st["lots"] = cs.lots or st.get("lots") or 1
                st["direction"] = cs.direction or st.get("direction") or "BUY"
                st["entry_price"] = cs.entry_price or st.get("entry_price")
                st["entry_strike"] = cs.entry_strike if cs.entry_strike is not None else st.get("entry_strike")
                if st.get("open_position"):
                    st["selected_strike"] = st.get("entry_strike")
                    st["selected_option_type"] = cs.option_type or st.get("side") or ""
                    st["selected_symbol"] = cs.entry_symbol or st.get("entry_symbol", "")
                st["unrealized_pnl"] = cs.unreal_pnl
                st["realized_pnl"] = cs.realized_pnl
                st["total_trades"] = cs.trades
                st["total_entries"] = cs.entries
                st["total_exits"] = cs.exits
                st["open_since"] = cs.open_since
                st["entry_symbol"] = cs.entry_symbol
                st["pnl_source"] = cs.pnl_source
                st["wins"] = cs.wins
                st["losses"] = cs.losses
                st["win_rate"] = cs.win_rate
                st["max_drawdown"] = cs.max_drawdown
                st["predict_attempted"] = cs.predict_attempted
                if cs.data_quality:
                    # for debug
                    pass
            conf = st.get("confidence")
            last_reason = st.get("last_no_trade_reason") or ""
            raw_reason = last_reason
            try:
                from synthetic_option_chain import map_paper_forward_reason
                display_reason, raw_reason = map_paper_forward_reason(last_reason)
            except Exception:
                display_reason = last_reason
            is_invalid = bool(
                (load_st and any(x in str(load_st).upper() for x in ("FEATURE_ORDER_MISSING", "ARTIFACT_MISSING", "ARTIFACT_INVALID", "MISSING", "candidate_missing_artifact_paths"))) or
                (last_reason and any(x in last_reason.upper() for x in ("FEATURE_ORDER_MISSING", "ARTIFACT_MISSING", "ARTIFACT_INVALID", "ARTIFACT_MISSING_NO_MODEL_OR_FEATURE_ORDER")))
            )
            if is_invalid or (last_reason and any(x in last_reason.upper() for x in ("FEATURE_ORDER_MISSING", "ARTIFACT_MISSING", "ARTIFACT_INVALID"))):
                conf_out = None
            else:
                conf_out = float(conf) if isinstance(conf, (int, float)) else None
            threshold = st.get("threshold", 0.0)
            threshold_out = round(threshold, 4) if isinstance(threshold, (int, float)) else 0.0
            # Prefer runtime last reason, else show load status (precise) or waiting
            if not last_reason:
                if load_st and load_st != LOAD_OK:
                    last_reason = load_st
                    display_reason = last_reason
                    raw_reason = last_reason
                else:
                    last_reason = "candidate_loaded_ok_waiting_for_snapshot"
                    display_reason = last_reason
                    raw_reason = last_reason
            predict_attempted = bool(st.get("predict_attempted"))
            paper_action = str(st.get("simulated_action") or st.get("paper_action") or (cs.last_action if cs else "") or "").upper()
            if not paper_action:
                if st.get("open_position"):
                    paper_action = "HOLD"
                elif predict_attempted:
                    paper_action = "NONE"
                else:
                    paper_action = "WAIT"
            trade_reason = str(display_reason or last_reason or "")
            rows.append({
                "candidate_id": cid,
                "enabled": c.get("enabled", True),
                "model_name": c.get("model_name", ""),
                "preset_family": c.get("preset_family", ""),
                "side_policy": c.get("side_policy", ""),
                "last_side_decision": st.get("side", ""),
                "final_signal": st.get("last_signal", "") or "NO_TRADE",
                "confidence": conf_out,
                "confidence_display": format_paper_forward_confidence(
                    conf_out,
                    predict_attempted=predict_attempted,
                    reason=str(raw_reason or last_reason or ""),
                    load_status=str(load_st or ""),
                ),
                "paper_action": paper_action,
                "trade_reason": trade_reason,
                "threshold": threshold_out,
                "position_status": "OPEN" if st.get("open_position") else "FLAT",
                "selected_strike": st.get("selected_strike") if st.get("open_position") else None,
                "selected_option_type": st.get("selected_option_type") if st.get("open_position") else "",
                "selected_symbol": st.get("selected_symbol") if st.get("open_position") else "",
                "entry_strike": st.get("entry_strike"),
                "entry_price": st.get("entry_price"),
                "current_price": st.get("option_current_price") or st.get("current_price"),
                "option_current_price": st.get("option_current_price") or st.get("current_price"),
                "spot_price": st.get("spot_price"),
                "qty": st.get("qty") or st.get("position_qty") or 65,
                "position_qty": st.get("position_qty") or st.get("qty") or 65,
                "lot_size": st.get("lot_size") or 65,
                "lots": st.get("lots") or 1,
                "direction": st.get("direction") or "BUY",
                "side": st.get("side") or "",
                "price_diff": round(float(st.get("price_diff") or 0.0), 2),
                "pnl_formula": st.get("pnl_formula") or "(current_option_price - entry_price) * qty",
                "mark_source": normalize_ltp_source(st.get("pnl_source", "")),
                "last_mark_time": st.get("last_mark_time", ""),
                "cost_quality": st.get("cost_quality", "APPROX"),
                "synthetic_mode": bool(st.get("synthetic_mode")),
                "ensemble_prob": st.get("ensemble_prob"),
                "xgb_prob": st.get("xgb_prob"),
                "rf_prob": st.get("rf_prob"),
                "block_reason": st.get("block_reason") or st.get("last_no_trade_reason"),
                "allowed": st.get("allowed"),
                "model_type": st.get("model_type") or c.get("model_name", ""),
                "feature_missing_count": st.get("feature_missing_count", len(st.get("missing_features") or [])),
                "feature_invalid_count": st.get("feature_invalid_count", 0),
                "unrealized_pnl": round(st.get("unrealized_pnl", 0.0), 2),
                "realized_pnl": round(st.get("realized_pnl", 0.0), 2),
                "total_trades": st.get("total_trades", 0),
                "total_entries": st.get("total_entries", 0),
                "total_exits": st.get("total_exits", 0),
                "open_since": st.get("open_since"),
                "entry_symbol": st.get("entry_symbol", ""),
                "pnl_source": st.get("pnl_source", ""),
                "wins": st.get("wins", 0),
                "losses": st.get("losses", 0),
                "win_rate": round(st.get("win_rate", 0.0), 3),
                "max_drawdown": round(st.get("max_drawdown", 0.0), 2),
                "last_no_trade_reason": str(display_reason or last_reason or ""),
                "raw_reason": str(raw_reason or last_reason or ""),
                "last_update": st.get("last_update"),
                "error_count": st.get("error_count", 0),
                "paper_forward_only": True,
                "shadow_ready": False,
                "live_orders_enabled": False,
                "_load_status": load_st,
                "_row_key": c.get("_row_key") or cid,
                "artifact_dir": c.get("artifact_dir"),
                "predict_attempted": predict_attempted,
            })
        return rows

    def write_debug_export(self, extra: Optional[Dict[str, Any]] = None) -> Path:
        """Write reports/paper_forward_debug_<timestamp>.json with full pipeline diagnostics."""
        ds: Dict[str, Any] = {}
        try:
            ds = (self._last_data_status.as_dict() if self._last_data_status else {}) or {}
        except Exception:
            ds = {}
        snap = dict(self._latest_market_snapshot or {})
        chain = self._latest_option_chain_snapshot
        chain_rows = len(chain) if isinstance(chain, (list, tuple)) else (_count_option_rows(chain) if chain else 0)
        candle_rows = _candle_count_from_snapshot(snap)
        per_candidate: List[Dict[str, Any]] = []
        ui_rows = self.get_status_table()
        for c in self.candidates:
            cid = c["candidate_id"]
            st = self._state.get(cid, {})
            res = c.get("_resolve") or {}
            per_candidate.append({
                "candidate_id": cid,
                "row_key": c.get("_row_key") or cid,
                "enabled": c.get("enabled", True),
                "artifact_dir": c.get("artifact_dir"),
                "model_path": res.get("model_file"),
                "metadata_path": c.get("_feature_order_src"),
                "required_features": c.get("required_features", len(c.get("_feature_order") or [])),
                "feature_order_len": len(c.get("_feature_order") or []),
                "load_status": c.get("_load_status"),
                "predict_attempted": bool(st.get("predict_attempted")),
                "confidence": st.get("confidence"),
                "confidence_display": format_paper_forward_confidence(
                    st.get("confidence"),
                    predict_attempted=bool(st.get("predict_attempted")),
                    reason=str(st.get("last_no_trade_reason") or ""),
                    load_status=str(c.get("_load_status") or ""),
                ),
                "final_signal": st.get("last_signal"),
                "reason": st.get("last_no_trade_reason"),
                "ensemble_prob": st.get("ensemble_prob"),
                "xgb_prob": st.get("xgb_prob"),
                "rf_prob": st.get("rf_prob"),
                "block_reason": st.get("block_reason") or st.get("last_no_trade_reason"),
                "allowed": st.get("allowed"),
                "model_type": st.get("model_type") or c.get("model_name"),
                "feature_missing_count": st.get("feature_missing_count", len(st.get("missing_features") or [])),
                "feature_invalid_count": st.get("feature_invalid_count", 0),
                "selected_strike": st.get("selected_strike"),
                "selected_option_type": st.get("selected_option_type"),
                "selected_symbol": st.get("selected_symbol"),
                "symbol": st.get("selected_symbol") or st.get("entry_symbol"),
                "qty": st.get("qty") or st.get("position_qty") or 65,
                "lot_size": st.get("lot_size") or 65,
                "lots": st.get("lots") or 1,
                "entry_price": st.get("entry_price"),
                "current_price": st.get("option_current_price") or st.get("current_price"),
                "price_diff": st.get("price_diff"),
                "unrealized_pnl": st.get("unrealized_pnl", 0.0),
                "realized_pnl": st.get("realized_pnl", 0.0),
                "pnl_formula": st.get("pnl_formula") or "(current_option_price - entry_price) * qty",
            })
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "broker_status": {
                "broker_name": ds.get("broker_name"),
                "broker_auth": ds.get("broker_auth") or ds.get("auth_status"),
                "auth_error": ds.get("auth_error"),
                "data_quality_status": ds.get("data_quality_status"),
            },
            "snapshot_summary": {
                "spot": snap.get("spot") or snap.get("price"),
                "chain_source": snap.get("chain_source") or ds.get("option_chain_source"),
                "candle_source": ds.get("candle_source"),
                "candle_count": candle_rows,
                "option_chain_rows": chain_rows,
                "last_snapshot_ts": self._last_snapshot_ts,
            },
            "candidate_count": len(self.candidates),
            "per_candidate": per_candidate,
            "ui_rows": ui_rows,
            "diagnostics": self.get_diagnostics(),
            "exceptions": list(getattr(self, "_debug_exceptions", []) or []),
        }
        if extra:
            payload.update(extra)
        full_debug: List[Dict[str, Any]] = []
        for item in getattr(self, "_export_decision_buffer", []) or []:
            if not isinstance(item, dict):
                continue
            cid = item.get("candidate_id")
            cand = next((c for c in self.candidates if c.get("candidate_id") == cid), {})
            full_debug.append({
                "candidate_id": cid,
                "decision": item.get("decision"),
                "feature_contract": item.get("feature_debug"),
                "feature_order": cand.get("_feature_order") or [],
                "available_features": (item.get("feature_debug") or {}).get("available_features"),
                "model_path": (cand.get("_resolve") or {}).get("files", {}).get("model_pkl"),
            })
        if full_debug:
            payload["full_debug_export"] = full_debug
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = self.reports_dir / f"paper_forward_debug_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"[PAPER-FWD-UI] debug_export={out_path}")
        return out_path

    def write_summary(self) -> dict:
        ds = {}
        try:
            ds = (self._last_data_status.as_dict() if self._last_data_status else {}) or {}
        except Exception:
            ds = {}
        is_synth = (
            str(ds.get("option_chain_source") or "").upper() == "BLACK_SCHOLES_SYNTHETIC"
            or any(s.get("synthetic_mode") for s in self._state.values())
        )
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_candidates": len(self.candidates),
            "active_sim_positions": sum(1 for s in self._state.values() if s.get("open_position")),
            "total_entries": sum(int(s.get("total_entries") or 0) for s in self._state.values()),
            "completed_trades": sum(int(s.get("total_trades") or 0) for s in self._state.values()),
            "candidates": self.get_status_table(),
            "live_orders_enabled": False,
            "broker_orders_enabled": False,
            "simulation_mode": "SIMULATED USING BLACK_SCHOLES_SYNTHETIC_CHAIN" if is_synth else "PAPER_SIM",
            "chain_source": "BLACK_SCHOLES_SYNTHETIC" if is_synth else ds.get("option_chain_source"),
            "candle_source": ds.get("candle_source"),
            "synthetic_candles": str(ds.get("candle_source") or "").upper() == "SYNTHETIC_SPOT_FALLBACK",
            "not_real_broker_market_data": is_synth,
            "data_quality_status": ds.get("data_quality_status"),
        }
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        with open(self.reports_dir / f"paper_forward_multi_summary_{ts}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        # md
        sim_note = summary.get("simulation_mode", "PAPER_SIM")
        lines = [
            "# Paper Forward Multi Summary " + ts,
            "",
            f"Candidates: {len(self.candidates)}",
            f"Mode: {sim_note}",
            "All simulated paper only. Live orders: FALSE.",
            "",
        ]
        for r in summary["candidates"]:
            lines.append(
                f"- {r['candidate_id']}: pos={r['position_status']} entries={r.get('total_entries', 0)} "
                f"completed_trades={r['total_trades']} real={r['realized_pnl']} un={r['unrealized_pnl']} wr={r['win_rate']}"
            )
        with open(self.reports_dir / f"paper_forward_multi_summary_{ts}.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return summary

    def write_jsonl(self, row: dict):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d")
        with open(self.log_dir / f"paper_forward_multi_{ts}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def get_diagnostics(self) -> Dict[str, Any]:
        """TASK 5: runtime counters for UI / diagnose popup."""
        try:
            from synthetic_option_chain import load_paper_sim_config
            sim_cfg = load_paper_sim_config()
        except Exception:
            sim_cfg = {}
        now_epoch = time.time()
        open_by_cid: Dict[str, Any] = {}
        for cid, s in self._state.items():
            is_synth = bool(s.get("synthetic_mode"))
            open_by_cid[cid] = {
                "open": bool(s.get("open_position")),
                "side": s.get("side"),
                "entry_symbol": s.get("entry_symbol"),
                "last_entry_ts": s.get("last_entry_ts"),
                "last_exit_ts": s.get("last_exit_ts"),
                "cooldown_remaining_sec": round(
                    _entry_cooldown_remaining(s, sim_cfg, now_epoch, is_synth), 1
                ),
                "option_current_price": s.get("option_current_price"),
                "pnl_source": s.get("pnl_source"),
            }
        return {
            "snapshots_received": self._snapshots_received,
            "evaluations_count": self._evaluations_count,
            "predictions_attempted": self._predictions_attempted,
            "predictions_success": self._predictions_success,
            "route_errors": self._route_errors,
            "skipped_missing_features": self._skipped_missing_features,
            "skipped_market_data": self._skipped_market_data,
            "last_snapshot_ts": self._last_snapshot_ts,
            "reason_counts": dict(self._reason_counts),
            "loaded_candidates": len(self.candidates),
            "candidate_load_summary": dict(getattr(self, "_candidate_load_summary", {}) or {}),
            "candidate_rows_updated": sum(1 for s in self._state.values() if s.get("last_update")),
            "data_status": self._last_data_status.as_dict(),
            "auth_wait_cycles": self._auth_wait_cycles,
            "route_exception_count": len(self._route_exception_details),
            "route_exception_type_counts": dict(Counter(d.get("exception_type") for d in self._route_exception_details)),
            "route_exception_candidate_ids": sorted({str(d.get("candidate_id")) for d in self._route_exception_details if d.get("candidate_id")}),
            "route_exception_tracebacks": self._route_exception_details[:5],
            "top_missing_features": dict(Counter(f for d in self._decisions[-200:] for f in (d.get("missing_features") or [])).most_common(20)),
            "duplicate_entry_blocked": self._duplicate_entry_blocked,
            "entry_cooldown_blocked": self._entry_cooldown_blocked,
            "signal_dedupe_blocked": self._signal_dedupe_blocked,
            "open_positions_by_candidate": open_by_cid,
            "live_order_guard": "PASS",
            "readiness_summary": dict(getattr(self, "last_readiness_summary", {}) or {}),
        }


# =============================================================================
# PHASE 2/3 CENTRAL LOADERS (exposed for diagnose, UI, engine, tests)
# =============================================================================

def load_dynamic_presets(project_root: Path | str = REPO_ROOT) -> Dict[str, Any]:
    """Central loader. Priority:
    1. Env DYNAMIC_PRESETS_PATH / PAPER_FORWARD_DYNAMIC_PRESETS_PATH
    2. config/dynamic_presets.json
    3. config/presets.json
    4. config/paper_forward_presets.json
    5. reports/*dynamic*preset*.json + *candidate*selection*.json (embedded)
    6. registries if they embed preset data
    7. per-artifact dynamic_presets.json fallbacks
    Supports dict keyed by name or list-of-presets or nested.
    """
    root = Path(project_root).resolve()
    # Env
    for envk in ("DYNAMIC_PRESETS_PATH", "PAPER_FORWARD_DYNAMIC_PRESETS_PATH"):
        ep = os.environ.get(envk)
        if ep:
            p = Path(ep)
            if p.exists():
                data = _safe_load_json(p)
                if data:
                    return {"source": str(p), "data": data, "loaded": True}

    # Standard configs
    for rel in ("config/dynamic_presets.json", "config/presets.json", "config/paper_forward_presets.json"):
        p = root / rel
        data = _safe_load_json(p)
        if data:
            return {"source": str(p), "data": data, "loaded": True}

    # Reports with embedded
    for pat in ("reports/*dynamic*preset*.json", "reports/*candidate*selection*.json"):
        for rp in root.glob(pat):
            data = _safe_load_json(rp)
            if data and (isinstance(data, dict) and (data.get("presets") or data.get("dynamic_presets") or any("preset" in k.lower() for k in data))):
                return {"source": str(rp), "data": data, "loaded": True, "embedded": True}

    # Registries
    for rel in ("config/candidate_registry.json", "config/model_registry.json"):
        p = root / rel
        data = _safe_load_json(p)
        if data:
            items = data.get("candidates", data if isinstance(data, list) else [])
            for it in (items or []):
                if it and (it.get("preset") or it.get("dynamic_preset") or it.get("preset_family")):
                    return {"source": str(p), "data": data, "loaded": True, "from_registry": True}

    # Artifact embedded
    art_base = root / "artifacts" / "candidates"
    if art_base.exists():
        for d in sorted(art_base.iterdir()):
            for fn in ("dynamic_presets.json", "dynamic_preset.json"):
                pp = (d / d.name / fn) if (d / d.name).exists() else (d / fn)
                if pp.exists():
                    data = _safe_load_json(pp)
                    if data:
                        return {"source": str(pp), "data": data, "loaded": True, "from_artifact": True}

    return {"source": None, "data": {}, "loaded": False}


def _safe_load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        if p.exists() and p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None

# make sure os is imported (already at top in current file)


def resolve_candidate_preset(candidate: Dict[str, Any], preset_registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return structured status. Generate safe PAPER fallback if needed. Never leave as no_dynamic_presets_configured when we can fallback."""
    cid = str(candidate.get("candidate_id") or candidate.get("id") or "?")
    pfam = candidate.get("preset_family") or candidate.get("preset") or ""
    art_dir = candidate.get("artifact_dir") or candidate.get("_resolve", {}).get("resolved_dir") or ""

    per_cand: Dict[str, Any] = {}
    if art_dir:
        ad = Path(art_dir)
        candidates = [ad, ad / cid, ad / (ad.name if ad.name else "")]
        for base in candidates:
            for fn in ("dynamic_presets.json", "dynamic_preset.json", "candidate_profile.json"):
                pp = base / fn
                if pp.exists():
                    dd = _safe_load_json(pp) or {}
                    cand_ps = dd.get("dynamic_presets") or dd.get("presets") or (dd if isinstance(dd, dict) and any("preset" in k.lower() for k in dd) else {})
                    if cand_ps:
                        per_cand = cand_ps if isinstance(cand_ps, dict) else {}
                        break
            if per_cand:
                break

    reg_data = (preset_registry or {}).get("data", {}) if preset_registry and preset_registry.get("loaded") else {}
    g_presets = {}
    if isinstance(reg_data, dict):
        g_presets = reg_data.get("presets", reg_data.get("dynamic_presets", {})) or {}

    if pfam and (pfam in per_cand or pfam in g_presets):
        return {"preset_loaded_ok": True, "preset_missing": False, "preset_fallback_generated": False, "preset_invalid_schema": False, "selected": pfam, "source": "per_cand" if pfam in per_cand else "global"}

    if per_cand:
        sel = next(iter(per_cand.keys()), pfam or "per_cand_embedded")
        return {"preset_loaded_ok": True, "preset_missing": False, "preset_fallback_generated": False, "preset_invalid_schema": False, "selected": sel, "source": "per_cand_artifact"}

    if g_presets:
        sel = pfam if pfam in g_presets else ("CONSERVATIVE" if "CONSERVATIVE" in g_presets else next(iter(g_presets.keys()), "OBSERVE_ONLY"))
        return {"preset_loaded_ok": True, "preset_missing": False, "preset_fallback_generated": True, "preset_invalid_schema": False, "selected": sel, "source": "global_fallback"}

    # PAPER-ONLY safe fallback (never blocks observation)
    return {
        "preset_loaded_ok": False,
        "preset_missing": True,
        "preset_fallback_generated": True,
        "preset_invalid_schema": False,
        "selected": "PAPER_OBSERVE_ONLY",
        "source": "generated_fallback",
        "reason": "no_dynamic_presets_configured_fallback_for_paper_sim",
    }


# resolve_candidate_artifacts authoritative implementation lives in the body above (PHASE 3 robust version with fuzzy + nested support).
