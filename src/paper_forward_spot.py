#!/usr/bin/env python3
"""
src/paper_forward_spot.py

Central spot and candle resolution for Paper Forward Monitor.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SPOT_CACHE_PATHS = (
    REPO_ROOT / "data" / "live_spot.json",
    REPO_ROOT / "logs" / "last_spot.json",
)
DEFAULT_MSTOCK_NIFTY_TOKEN = "26000"
DEFAULT_SPOT_STALE_SEC = 300.0


@dataclass
class SpotResolveResult:
    ok: bool
    spot: float | None
    source: str
    error_code: str | None = None
    stale: bool = False
    tried: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "spot": self.spot,
            "source": self.source,
            "error_code": self.error_code,
            "stale": self.stale,
            "tried": list(self.tried),
        }


@dataclass
class CandleResolveResult:
    ok: bool
    candles: List[Any]
    source: str
    error: str = ""
    tried: List[str] = field(default_factory=list)
    synthetic_candles: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "rows": len(self.candles),
            "source": self.source,
            "error": self.error,
            "tried": list(self.tried),
            "synthetic_candles": self.synthetic_candles,
        }


def _min_candles_for_prediction() -> int:
    try:
        return int(os.getenv("PF_MIN_CANDLES_FOR_PREDICTION", "20"))
    except Exception:
        return 20


def _to_float(value: Any) -> Optional[float]:
    if value in (None, "", "n/a", "N/A", "nan"):
        return None
    try:
        v = float(value)
        if v > 0 and v == v:
            return v
    except Exception:
        pass
    return None


def _mstock_nifty_token() -> str:
    for key in ("MSTOCK_NIFTY_INDEX_TOKEN", "MSTOCK_UNDERLYING_TOKEN", "MSTOCK_OPTION_TOKEN"):
        tok = str(os.getenv(key, "") or "").strip()
        if tok.isdigit():
            return tok
    return DEFAULT_MSTOCK_NIFTY_TOKEN


def _mstock_spot_quote_keys() -> List[str]:
    token = _mstock_nifty_token()
    exch = (
        os.getenv("MSTOCK_UNDERLYING_EXCHANGE")
        or os.getenv("MSTOCK_EXCHANGE")
        or "NSE"
    ).strip().upper() or "NSE"
    underlying = (os.getenv("MSTOCK_UNDERLYING") or os.getenv("MSTOCK_SYMBOL") or "NIFTY").strip()
    keys = [f"{exch}:{token}", token, f"{exch}:{underlying}", underlying]
    out: List[str] = []
    for k in keys:
        if k and k not in out:
            out.append(k)
    return out


def fetch_mstock_nifty_spot_ltp(client: Any) -> Tuple[Optional[float], str, str]:
    """Fetch NIFTY index LTP from m.Stock. Returns (spot, status, detail)."""
    if client is None or not hasattr(client, "get_ltp"):
        return None, "NO_CLIENT", "client_missing_or_no_get_ltp"
    token = _mstock_nifty_token()
    last_err = ""
    for key in _mstock_spot_quote_keys():
        try:
            spot = _to_float(client.get_ltp(key))
            if spot is not None:
                print(f"[MSTOCK-SPOT-FETCH] token={token} key={key} status=OK spot={spot:.2f}")
                return spot, "OK", key
        except Exception as exc:
            last_err = str(exc)
            print(f"[MSTOCK-SPOT-FETCH] token={token} key={key} status=FAILED error={last_err[:120]}")
    print(f"[MSTOCK-SPOT-FETCH] token={token} status=FAILED spot=None")
    return None, "FAILED", last_err or "all_keys_failed"


def _read_spot_cache(max_age_sec: float = DEFAULT_SPOT_STALE_SEC) -> Tuple[Optional[float], bool]:
    now = time.time()
    for path in SPOT_CACHE_PATHS:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            spot = _to_float(payload.get("spot") or payload.get("price") or payload.get("ltp"))
            if spot is None:
                continue
            ts_raw = payload.get("ts") or payload.get("timestamp") or payload.get("updated_at")
            stale = False
            if ts_raw is not None:
                try:
                    if isinstance(ts_raw, (int, float)):
                        age = now - float(ts_raw)
                    else:
                        dt = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                        age = now - dt.timestamp()
                    stale = age > max_age_sec
                except Exception:
                    stale = True
            return spot, stale
        except Exception:
            continue
    return None, False


def write_spot_cache(spot: float, source: str) -> None:
    payload = {
        "spot": float(spot),
        "source": source,
        "ts": time.time(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for path in SPOT_CACHE_PATHS:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[PF-SPOT-CACHE-WRITE] spot={spot:.2f} source={source} path={path.name}")
            break
        except Exception:
            continue


def resolve_live_spot_for_paper_forward(
    *,
    app_state: Optional[Dict[str, Any]] = None,
    client: Any = None,
    broker: str = "mstock",
    current: Any = None,
    chain: Any = None,
    candles: Any = None,
    spot_quote_key_fn: Optional[Callable[[], str]] = None,
    allow_option_chain_spot: bool = False,
    prefer_live_quote: bool = False,
) -> SpotResolveResult:
    """Resolve live spot for Paper Forward with explicit priority order."""
    tried: List[str] = []
    state = app_state or {}
    broker_norm = str(broker or "mstock").strip().lower()

    def _try(label: str, value: Any, source: str, *, stale: bool = False) -> Optional[SpotResolveResult]:
        tried.append(label)
        spot = _to_float(value)
        if spot is not None:
            res = SpotResolveResult(True, spot, source, stale=stale, tried=list(tried))
            print(f"[PF-SPOT-RESOLVE] source={source} ok=True spot={spot:.2f} stale={stale}")
            return res
        return None

    def _try_broker_live_quote() -> Optional[SpotResolveResult]:
        if client is None or not hasattr(client, "get_ltp"):
            return None
        if broker_norm == "dhan":
            sym = os.getenv("DHAN_UNDERLYING", "NIFTY") or "NIFTY"
            tried.append("dhan_ltp")
            try:
                spot = _to_float(client.get_ltp(sym))
                if spot is not None:
                    print(f"[PF-SPOT-RESOLVE] source=dhan_ltp ok=True spot={spot:.2f} stale=False")
                    write_spot_cache(spot, "dhan_ltp")
                    return SpotResolveResult(True, spot, "dhan_ltp", tried=tried)
            except Exception as exc:
                print(f"[PF-DATA] spot_quote_fallback_failed={exc}")
            return None

        keys: List[str] = []
        if spot_quote_key_fn is not None:
            try:
                keys.append(str(spot_quote_key_fn() or ""))
            except Exception:
                pass
        keys.extend(_mstock_spot_quote_keys())
        seen = set()
        for key in keys:
            if not key or key in seen:
                continue
            seen.add(key)
            tried.append(f"mstock_ltp:{key}")
            try:
                spot = _to_float(client.get_ltp(key))
                if spot is not None:
                    print(f"[PF-SPOT-RESOLVE] source=mstock_ltp ok=True spot={spot:.2f} stale=False")
                    write_spot_cache(spot, "mstock_ltp")
                    return SpotResolveResult(True, spot, "mstock_ltp", tried=tried)
            except Exception as exc:
                print(f"[MSTOCK-SPOT-FETCH] key={key} status=FAILED error={exc}")
        spot, _status, _ = fetch_mstock_nifty_spot_ltp(client)
        tried.append("fetch_mstock_nifty_spot_ltp")
        if spot is not None:
            print(f"[PF-SPOT-RESOLVE] source=mstock_ltp ok=True spot={spot:.2f} stale=False")
            write_spot_cache(spot, "mstock_ltp")
            return SpotResolveResult(True, spot, "mstock_ltp", tried=tried)
        return None

    if prefer_live_quote:
        hit = _try_broker_live_quote()
        if hit:
            return hit

    hit = _try("current_arg", current, "runtime_current")
    if hit:
        return hit

    for label, key in (
        ("runtime.spot", ("_pf_runtime", "spot")),
        ("runtime.price", ("_pf_runtime", "price")),
        ("spot_ltp_live", ("_spot_ltp_live",)),
        ("dash_spot_var", ("_dash_spot_var",)),
    ):
        obj = state
        parts = key if isinstance(key, tuple) else (key,)
        for part in parts:
            obj = getattr(obj, part, None) if not isinstance(obj, dict) else obj.get(part)
            if obj is None:
                break
        if label.endswith("var") and obj is not None and hasattr(obj, "get"):
            obj = obj.get()
        hit = _try(label, obj, label.replace(".", "_"))
        if hit:
            return hit

    plugin = state.get("live_chart_plugin") if isinstance(state, dict) else getattr(state, "live_chart_plugin", None)
    if plugin is not None:
        for attr, src in (("_spot", "live_chart_plugin_spot"), ("spot", "live_chart_plugin_spot"), ("_last_spot", "live_chart_plugin_last_spot")):
            hit = _try(f"live_chart.{attr}", getattr(plugin, attr, None), src)
            if hit:
                return hit

    scalper = state.get("_scalper") or state.get("scalper") if isinstance(state, dict) else (
        getattr(state, "_scalper", None) or getattr(state, "scalper", None)
    )
    if scalper is not None:
        for attr, src in (
            ("last_spot", "scalper_last_spot"),
            ("spot", "scalper_spot"),
            ("index_ltp", "scalper_index_ltp"),
            ("_spot_ltp", "scalper_spot_ltp"),
        ):
            hit = _try(f"scalper.{attr}", getattr(scalper, attr, None), src)
            if hit:
                return hit

    if candles:
        last = candles[-1]
        if isinstance(last, dict):
            hit = _try("latest_candle_close", last.get("close") or last.get("last") or last.get("price"), "latest_candle_close")
        else:
            hit = _try("latest_candle_close", getattr(last, "close", None), "latest_candle_close")
        if hit:
            return hit

    hit = _try_broker_live_quote()
    if hit:
        return hit

    fb = str(os.getenv("PF_FALLBACK_SPOT", "") or "").strip()
    if fb:
        tried.append("PF_FALLBACK_SPOT")
        hit = _try("PF_FALLBACK_SPOT", fb, "PF_FALLBACK_SPOT")
        if hit:
            return hit

    cached, stale = _read_spot_cache()
    if cached is not None:
        tried.append("spot_cache_file")
        hit = _try("spot_cache_file", cached, "spot_cache_file", stale=stale)
        if hit:
            return hit

    if allow_option_chain_spot and chain:
        try:
            from paper_forward_spot_integrity import derive_spot_from_option_chain_payload
            spot, src = derive_spot_from_option_chain_payload(chain)
            tried.append("option_chain_derived")
            hit = _try("option_chain_derived", spot, src or "option_chain_derived")
            if hit:
                return hit
        except Exception:
            pass

    print(f"[PF-SPOT-MISSING] tried={tried}")
    return SpotResolveResult(False, None, "", error_code="SPOT_MISSING", tried=tried)


def _normalize_candle_rows(cached: Sequence[Any], *, limit: int = 100) -> List[Any]:
    if not cached:
        return []
    return list(cached)[-limit:]


def resolve_paper_forward_candles(
    *,
    app_state: Optional[Dict[str, Any]] = None,
    client: Any = None,
    broker: str = "mstock",
    spot: Optional[float] = None,
    allow_synthetic_fallback: Optional[bool] = None,
) -> CandleResolveResult:
    """Resolve candles for Paper Forward with real-candle priority."""
    tried: List[str] = []
    state = app_state or {}
    min_rows = _min_candles_for_prediction()
    plugin = state.get("live_chart_plugin") if isinstance(state, dict) else getattr(state, "live_chart_plugin", None)
    scalper = state.get("_scalper") or state.get("scalper") if isinstance(state, dict) else (
        getattr(state, "_scalper", None) or getattr(state, "scalper", None)
    )

    # Priority: live_chart_cache -> broker historical -> scalper caches
    cache_sources = (
        ("live_chart_cache", state.get("_latest_candles") if isinstance(state, dict) else getattr(state, "_latest_candles", None)),
        ("live_chart_raw_cache", getattr(plugin, "_raw_candles", None) if plugin is not None else None),
        ("live_chart_render_cache", getattr(plugin, "_candles", None) if plugin is not None else None),
        ("chart_candles_cache", state.get("_chart_candles") if isinstance(state, dict) else getattr(state, "_chart_candles", None)),
        ("scalper_spot_candles", getattr(scalper, "_spot_candles", None) if scalper is not None else None),
        ("scalper_candles", getattr(scalper, "_candles", None) if scalper is not None else None),
    )
    best_real: Tuple[List[Any], str] = ([], "")
    for source, cached in cache_sources:
        tried.append(source)
        if not cached:
            continue
        rows = _normalize_candle_rows(cached)
        print(f"[PF-CANDLES-RESOLVE] source={source} rows={len(rows)}")
        if len(rows) >= min_rows and len(rows) > len(best_real[0]):
            best_real = (rows, source)

    if best_real[0]:
        print(f"[PF-CANDLES-REAL-PREFERRED] rows={len(best_real[0])} source={best_real[1]}")
        print(f"[PF-CANDLES-RESOLVE] selected_source={best_real[1]} rows={len(best_real[0])}")
        return CandleResolveResult(True, best_real[0], best_real[1], tried=tried)

    if client is not None:
        broker_norm = str(broker or "mstock").strip().lower()
        if broker_norm == "dhan":
            token = os.getenv("DHAN_UNDER_SECURITY_ID", os.getenv("DHAN_UNDERLYING_SECURITY_ID", "13")).strip()
            exchange = os.getenv("DHAN_UNDER_EXCHANGE_SEGMENT", "IDX_I").strip().upper() or "IDX_I"
        else:
            token = os.getenv("MSTOCK_UNDERLYING_TOKEN", _mstock_nifty_token()).strip() or _mstock_nifty_token()
            exchange = os.getenv("MSTOCK_UNDERLYING_EXCHANGE", os.getenv("MSTOCK_EXCHANGE", "NSE")).strip().upper() or "NSE"
        hist_source = f"{broker_norm}_historical"
        tried.append(hist_source)
        try:
            if hasattr(client, "resolve_exchange_token") and not token:
                exchange, token = client.resolve_exchange_token("NIFTY", exchange_hint="NSE")
            if token and hasattr(client, "fetch_index_candles"):
                candles, interval = client.fetch_index_candles(
                    token, exchange=exchange, limit=100, timeframe="ONE_MINUTE", force_historical_only=True
                )
                if candles:
                    rows = _normalize_candle_rows(candles)
                    print(f"[PF-CANDLES-RESOLVE] source={hist_source} rows={len(rows)} interval={interval}")
                    if len(rows) >= min_rows:
                        print(f"[PF-CANDLES-REAL-PREFERRED] rows={len(rows)} source={hist_source}")
                        print(f"[PF-CANDLES-RESOLVE] selected_source={hist_source} rows={len(rows)}")
                        return CandleResolveResult(True, rows, hist_source, tried=tried)
                    if len(rows) > len(best_real[0]):
                        best_real = (rows, hist_source)
        except Exception as exc:
            print(f"[PF-CANDLES-RESOLVE] source={hist_source} rows=0 error={exc}")

    if best_real[0]:
        print(f"[PF-CANDLES-RESOLVE] selected_source={best_real[1]} rows={len(best_real[0])}")
        return CandleResolveResult(True, best_real[0], best_real[1], tried=tried)

    if allow_synthetic_fallback is None:
        allow_synthetic_fallback = str(os.getenv("PF_ALLOW_SYNTHETIC_CANDLE_FALLBACK", "true")).lower() in {"1", "true", "yes"}
    if allow_synthetic_fallback and spot is not None and spot > 0:
        try:
            from synthetic_option_chain import build_spot_candle_fallback
            rows = build_spot_candle_fallback(float(spot))
            tried.append("SYNTHETIC_SPOT_FALLBACK")
            if rows:
                print(f"[PF-CANDLES-RESOLVE] selected_source=SYNTHETIC_SPOT_FALLBACK rows={len(rows)}")
                return CandleResolveResult(
                    True,
                    rows,
                    "SYNTHETIC_SPOT_FALLBACK",
                    tried=tried,
                    synthetic_candles=True,
                )
        except Exception as exc:
            print(f"[PF-CANDLES-RESOLVE] synthetic_spot_fallback error={exc}")

    print(f"[PF-CANDLES-MISSING] tried={tried}")
    return CandleResolveResult(False, [], "none", error="no_candles", tried=tried)
