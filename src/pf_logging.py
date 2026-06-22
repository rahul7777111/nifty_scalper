#!/usr/bin/env python3
"""Paper Forward logging verbosity, rate limiting, and performance diagnostics."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

_LEVELS = {"ERROR": 0, "WARN": 1, "INFO": 2, "DEBUG": 3, "TRACE": 4}

_rate_limit_ts: Dict[str, float] = {}
_rate_limit_skipped: Dict[str, int] = {}
_lines_written = 0
_bytes_written = 0
_rate_limited_total = 0


def pf_log_level() -> int:
    return _LEVELS.get(str(os.getenv("PF_LOG_LEVEL", "INFO")).strip().upper(), 2)


def pf_verbose_features() -> bool:
    return str(os.getenv("PF_VERBOSE_FEATURES", "false")).strip().lower() in {"1", "true", "yes"}


def pf_verbose_decision() -> bool:
    return str(os.getenv("PF_VERBOSE_DECISION", "false")).strip().lower() in {"1", "true", "yes"}


def pf_verbose_predict() -> bool:
    return str(os.getenv("PF_VERBOSE_PREDICT", "false")).strip().lower() in {"1", "true", "yes"}


def pf_log_full_feature_list() -> bool:
    return str(os.getenv("PF_LOG_FULL_FEATURE_LIST", "false")).strip().lower() in {"1", "true", "yes"}


def pf_should_log_level(level: str) -> bool:
    return _LEVELS.get(str(level).upper(), 2) <= pf_log_level()


def should_log(key: str, interval_sec: float) -> bool:
    """Return True if this log key may emit (rate-limited)."""
    global _rate_limited_total
    now = time.time()
    last = _rate_limit_ts.get(key, 0.0)
    if now - last < float(interval_sec):
        _rate_limit_skipped[key] = _rate_limit_skipped.get(key, 0) + 1
        _rate_limited_total += 1
        return False
    skipped = _rate_limit_skipped.pop(key, 0)
    if skipped > 0:
        print(f"[LOG-RATE-LIMIT] skipped key={key} count={skipped}")
    _rate_limit_ts[key] = now
    return True


def pf_log(
    level: str,
    msg: str,
    *,
    rate_key: str = "",
    rate_interval: float = 0.0,
) -> None:
    global _lines_written, _bytes_written
    if not pf_should_log_level(level):
        return
    if rate_key and rate_interval > 0 and not should_log(rate_key, rate_interval):
        return
    print(msg, flush=True)
    _lines_written += 1
    _bytes_written += len(msg.encode("utf-8", errors="replace")) + 1


def pf_log_perf_summary() -> None:
    pf_log(
        "DEBUG",
        f"[LOG-PERF] lines_written={_lines_written} bytes_written={_bytes_written} "
        f"rate_limited={_rate_limited_total}",
        rate_key="log_perf_summary",
        rate_interval=30.0,
    )


def slim_feature_debug(debug: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compact feature contract for live logs; full lists only at TRACE+PF_LOG_FULL_FEATURE_LIST."""
    if not isinstance(debug, dict):
        return {}
    if pf_log_full_feature_list() and pf_should_log_level("TRACE"):
        return dict(debug)
    out = dict(debug)
    avail = out.get("available_features")
    if isinstance(avail, list):
        out["available_features"] = f"<{len(avail)} cols>"
    fo = out.get("feature_order")
    if isinstance(fo, list):
        out["feature_order"] = f"<{len(fo)} features>"
    return out


def log_paper_fwd_features(
    *,
    candidate_id: str,
    required: int,
    available: int,
    missing: List[str],
    nan: int = 0,
    zero: int = 0,
    coverage: float = 0.0,
) -> None:
    key = f"pf_features:{candidate_id}"
    if pf_should_log_level("DEBUG"):
        pf_log(
            "DEBUG",
            f"[PAPER-FWD-FEATURES] cid={candidate_id} required={required} available={available} "
            f"missing={len(missing)} nan={nan} zero={zero} coverage={coverage:.1f} "
            f"top_missing={missing[:5]}",
            rate_key=key,
            rate_interval=30.0,
        )
    else:
        pf_log(
            "INFO",
            f"[PAPER-FWD-FEATURES] cid={candidate_id} required={required} available={available} "
            f"missing={len(missing)} nan={nan} zero={zero} coverage={coverage:.1f}",
            rate_key=key,
            rate_interval=30.0,
        )


def log_paper_fwd_decision(
    *,
    candidate_id: str,
    signal: str,
    confidence: Any = None,
    threshold: Any = None,
    reason: str = "",
    side: str = "",
    pos: str = "",
    strike: Any = None,
    opt_type: str = "",
    elapsed_ms: int = 0,
    allowed: str = "",
) -> None:
    conf_txt = "-" if confidence is None else f"{float(confidence):.4f}"
    thr_txt = "-" if threshold is None else f"{float(threshold):.4f}"
    pf_log(
        "INFO",
        f"[PAPER-FWD-DECISION] cid={candidate_id} signal={signal} conf={conf_txt} "
        f"threshold={thr_txt} side={side or '-'} reason={reason[:80] or 'OK'} "
        f"pos={pos or '-'} strike={strike or '-'} opt_type={opt_type or '-'} "
        f"allowed={allowed or '-'} elapsed_ms={elapsed_ms}",
        rate_key=f"pf_decision:{candidate_id}",
        rate_interval=30.0 if signal in ("NO_TRADE", "") else 5.0,
    )


def log_paper_fwd_predict(
    *,
    candidate_id: str,
    model: str = "",
    raw_output: Any = None,
    confidence: Any = None,
    threshold: Any = None,
    data_quality: str = "",
    predict_allowed: bool = True,
    scaler_applied: bool = False,
    calibrator_applied: bool = False,
    missing_count: int = 0,
    feature_debug: Optional[Dict[str, Any]] = None,
) -> None:
    if not pf_verbose_predict() and not pf_should_log_level("DEBUG"):
        return
    conf_txt = "-" if confidence is None else f"{float(confidence):.4f}"
    raw_txt = "-" if raw_output is None else f"{raw_output}"
    fd = slim_feature_debug(feature_debug or {})
    pf_log(
        "DEBUG",
        f"[PAPER-FWD-PREDICT] cid={candidate_id} model={model} raw={raw_txt} conf={conf_txt} "
        f"thr={threshold} quality={data_quality} allowed={predict_allowed} "
        f"scaler={scaler_applied} cal={calibrator_applied} missing={missing_count} "
        f"nan={fd.get('feature_nan_count', 0)} zero={fd.get('feature_zero_count', 0)}",
        rate_key=f"pf_predict:{candidate_id}",
        rate_interval=30.0,
    )


def log_feature_x(
    *,
    candidate_id: str,
    x_shape: Any = None,
    nan_count: int = 0,
    inf_count: int = 0,
    zero_count: int = 0,
    min_val: Any = None,
    max_val: Any = None,
    first10: Any = None,
) -> None:
    if not pf_verbose_features() and not pf_should_log_level("DEBUG"):
        return
    if pf_should_log_level("TRACE") and pf_log_full_feature_list():
        first_txt = first10
    else:
        first_txt = "<redacted>"
    pf_log(
        "DEBUG",
        f"[FEATURE-X] cid={candidate_id} X_shape={x_shape} nan={nan_count} inf={inf_count} "
        f"zero={zero_count} min={min_val} max={max_val} first10={first_txt}",
        rate_key=f"feature_x:{candidate_id}",
        rate_interval=30.0,
    )


def log_paper_fwd_candidate(
    *,
    candidate_id: str,
    signal: str,
    confidence: Any = None,
    reason: str = "",
    pos: str = "",
    strike: Any = None,
    opt_type: str = "",
) -> None:
    conf_txt = "-" if confidence is None else f"{float(confidence):.4f}"
    pf_log(
        "INFO",
        f"[PAPER-FWD-CANDIDATE] cid={candidate_id} signal={signal} conf={conf_txt} "
        f"reason={reason[:60] or 'OK'} pos={pos or '-'} strike={strike or '-'} opt_type={opt_type or '-'}",
        rate_key=f"pf_candidate:{candidate_id}",
        rate_interval=30.0 if signal in ("NO_TRADE", "") else 5.0,
    )


def log_pf_data(
    *,
    spot: Any = None,
    option_rows: int = 0,
    candles: int = 0,
    data_quality: str = "",
    spot_source: str = "",
    extra: str = "",
) -> None:
    parts = [
        f"spot={spot if spot is not None else '-'}",
        f"option_rows={option_rows}",
        f"candles={candles}",
        f"data_quality={data_quality or '-'}",
    ]
    if spot_source:
        parts.append(f"spot_source={spot_source}")
    if extra:
        parts.append(extra)
    pf_log(
        "INFO",
        f"[PF-DATA] {' '.join(parts)}",
        rate_key="pf_data_status",
        rate_interval=5.0,
    )


def log_scripmaster_cache(*, hit: bool, key: str = "", rows: int = 0, elapsed_ms: int = 0) -> None:
    status = "hit" if hit else ("loaded" if rows else "miss")
    pf_log(
        "DEBUG",
        f"[SCRIPMASTER-CACHE] {status} key={key or '-'} rows={rows} elapsed_ms={elapsed_ms}",
        rate_key=f"scripmaster_cache:{status}:{key}",
        rate_interval=30.0,
    )


def log_paper_fwd_summary(
    *,
    candidates: int,
    signals: int,
    no_trade: int,
    errors: int,
    elapsed_ms: int,
) -> None:
    pf_log(
        "INFO",
        f"[PAPER-FWD-SUMMARY] candidates={candidates} signals={signals} no_trade={no_trade} "
        f"errors={errors} elapsed_ms={elapsed_ms}",
        rate_key="paper_fwd_summary",
        rate_interval=5.0,
    )


def log_pf_cache(kind: str, *, hit: bool, **fields: Any) -> None:
    status = "hit" if hit else "miss"
    parts = " ".join(f"{k}={fields[k]}" for k in sorted(fields.keys()))
    pf_log(
        "DEBUG",
        f"[PF-CACHE] {kind} {status} {parts}".strip(),
        rate_key=f"pf_cache:{kind}:{status}:{fields.get('cid', fields.get('symbol', ''))}",
        rate_interval=30.0,
    )


def log_pf_perf(**fields: Any) -> None:
    parts = " ".join(f"{k}={fields[k]}" for k in sorted(fields.keys()))
    pf_log("INFO", f"[PF-PERF] {parts}", rate_key="pf_perf_cycle", rate_interval=5.0)
    cycle_ms = float(fields.get("cycle_ms") or 0)
    if cycle_ms > 1000:
        slow = max(
            (("snapshot", float(fields.get("snapshot_ms") or 0)),
             ("feature", float(fields.get("feature_ms") or 0)),
             ("predict", float(fields.get("predict_ms") or 0)),
             ("ui", float(fields.get("ui_ms") or 0)),
             ("log", float(fields.get("log_ms") or 0))),
            key=lambda x: x[1],
        )
        pf_log("WARN", f"[PF-LAG-WARN] cycle_ms={cycle_ms:.0f} slow_part={slow[0]} slow_ms={slow[1]:.0f}")


def log_gui_perf(**fields: Any) -> None:
    parts = " ".join(f"{k}={fields[k]}" for k in sorted(fields.keys()))
    pf_log("DEBUG", f"[GUI-PERF] {parts}", rate_key="gui_perf", rate_interval=10.0)


@contextmanager
def pf_timer() -> Iterator[Dict[str, float]]:
    t0 = time.perf_counter()
    bucket: Dict[str, float] = {}
    yield bucket
    bucket["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0


# Public alias requested by PF performance spec.
_should_log = should_log