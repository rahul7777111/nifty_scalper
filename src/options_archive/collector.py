from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import asdict
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from config import APIConfig
from greeks import delta as bs_delta
from greeks import gamma as bs_gamma
from greeks import implied_volatility as solve_iv
from greeks import theta as bs_theta
from greeks import vega as bs_vega
from logger_setup import get_logger
from mstock_client import MStockTypeBClient

from .models import ARCHIVE_SCHEMA_VERSION, ArchiveConfig, ArchiveResult, ArchiveSnapshot, ArchiveState
from .symbols import resolve_symbol_spec
from .qa import DailyQAReporter
from .storage import OptionsArchiveStorage


IST = ZoneInfo("Asia/Kolkata")


def _parse_time(hhmm: str) -> dt_time:
    hour, minute = [int(part) for part in str(hhmm).split(":", 1)]
    return dt_time(hour=hour, minute=minute, tzinfo=IST)


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
        if math.isnan(number):
            return None
        return number
    except Exception:
        return None


def _normalize_option_type(value: Any) -> Optional[str]:
    text = str(value or "").strip().upper()
    if text in {"CE", "CALL", "C"}:
        return "CE"
    if text in {"PE", "PUT", "P"}:
        return "PE"
    if "CALL" in text:
        return "CE"
    if "PUT" in text:
        return "PE"
    return None


def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    for candidate in (
        text,
        text.replace("/", "-"),
        text[:10],
    ):
        try:
            return datetime.fromisoformat(candidate).date()
        except Exception:
            pass
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(text[:11], fmt).date()
        except Exception:
            continue
    return None


def _normalise_iv(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    iv = float(value)
    if iv > 1.5:
        iv /= 100.0
    if iv <= 0:
        return None
    return iv


def _liquidity_bucket(total_volume: float, median_spread_bps: float) -> str:
    if total_volume >= 100000 and median_spread_bps <= 8:
        return "high"
    if total_volume >= 25000 and median_spread_bps <= 20:
        return "medium"
    return "low"


def _volatility_bucket(iv: Optional[float]) -> str:
    if iv is None:
        return "unknown"
    if iv < 0.14:
        return "low"
    if iv < 0.22:
        return "medium"
    return "high"


class OptionsArchiveCollector:
    """Collect point-in-time option-chain snapshots from m.Stock."""

    def __init__(self, api_config: APIConfig, config: Optional[ArchiveConfig] = None, logger: Optional[logging.Logger] = None) -> None:
        self.config = config or ArchiveConfig()
        spec = resolve_symbol_spec(self.config.symbol_key)
        self.config.symbol_id = spec.symbol_id
        self.config.index_family = spec.index_family
        self.config.underlying = spec.underlying
        self.config.spot_symbol = spec.spot_symbol
        self.config.chain_underlying = spec.chain_underlying
        self.config.lot_size = spec.lot_size
        self.config.expiry_cycle_type = spec.expiry_cycle_type
        self.logger = logger or get_logger("options_archive.collector")
        self.client = MStockTypeBClient(api_config)
        self.storage = OptionsArchiveStorage(self.config, logger=self.logger)
        self.qa_reporter = DailyQAReporter()
        self.state = self.storage.load_state()

    def _now(self) -> datetime:
        return datetime.now(tz=IST)

    def _is_market_day(self, now: datetime) -> bool:
        return now.weekday() < 5

    def _session_open(self, now: datetime) -> datetime:
        return datetime.combine(now.date(), dt_time(9, 15, tzinfo=IST), tzinfo=IST)

    def _session_close(self, now: datetime) -> datetime:
        return datetime.combine(now.date(), dt_time(15, 30, tzinfo=IST), tzinfo=IST)

    def _is_expiry_day(self, snapshot_rows: Sequence[Dict[str, Any]], session_date: date) -> bool:
        expiries = {_parse_date(row.get("expiry")) for row in snapshot_rows}
        expiries.discard(None)
        return session_date in expiries

    def _spot_move(self, spot: float) -> Tuple[Optional[float], Optional[float]]:
        last_spot = self.state.last_snapshot_spot
        if last_spot is None or last_spot <= 0:
            return None, None
        points = float(spot - last_spot)
        pct = float((spot / last_spot - 1.0) * 100.0)
        return points, pct

    def _next_base_due(self, now: datetime) -> bool:
        last = self.state.last_base_snapshot_ts
        if last is None:
            return now >= self._session_open(now)
        return (now - last) >= timedelta(minutes=max(1, int(self.config.snapshot_interval_minutes)))

    def _is_open_window(self, now: datetime) -> bool:
        open_ts = self._session_open(now)
        return open_ts <= now <= open_ts + timedelta(minutes=max(1, int(self.config.open_snapshot_window_minutes)))

    def _retry(self, label: str, fn):
        attempts = max(1, int(self.config.max_retries))
        delay = max(0.25, float(self.config.retry_backoff_seconds))
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self.logger.warning("%s attempt %d/%d failed: %s", label, attempt, attempts, exc)
                if attempt < attempts:
                    time.sleep(delay * attempt)
        raise last_exc  # type: ignore[misc]

    def _normalize_chain_row(self, row: Dict[str, Any], spot: float, as_of: datetime) -> Optional[Dict[str, Any]]:
        option_type = _normalize_option_type(
            row.get("option_type") or row.get("opt_type") or row.get("right") or row.get("cp") or row.get("CEPE")
        )
        if option_type is None:
            return None

        strike = _to_float(row.get("strike") or row.get("strikePrice") or row.get("strike_price"))
        if strike is None or strike <= 0:
            return None

        expiry = _parse_date(row.get("expiry") or row.get("expiryDate") or row.get("expiry_date") or row.get("expiry_dt"))
        if expiry is None:
            return None

        ltp = _to_float(row.get("ltp") or row.get("LTP") or row.get("last_price") or row.get("lastPrice"))
        bid = _to_float(row.get("bid") or row.get("best_bid") or row.get("bidPrice"))
        ask = _to_float(row.get("ask") or row.get("best_ask") or row.get("askPrice"))
        volume = _to_float(row.get("volume") or row.get("tradedVolume") or row.get("tradeVolume")) or 0.0
        oi = _to_float(row.get("oi") or row.get("openInterest") or row.get("open_interest")) or 0.0
        mid = None
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
        market_price = mid if mid is not None else ltp

        iv = _normalise_iv(_to_float(row.get("iv") or row.get("impliedVolatility") or row.get("implied_volatility")))
        if iv is None and market_price is not None and market_price > 0:
            try:
                time_to_expiry = max(1e-6, (expiry - as_of.date()).days / 365.0)
                iv = solve_iv(market_price, spot, strike, time_to_expiry, self.config.risk_free_rate, option_type)
            except Exception:
                iv = None

        delta = _to_float(row.get("delta") or row.get("greeks_delta"))
        gamma = _to_float(row.get("gamma") or row.get("greeks_gamma"))
        vega = _to_float(row.get("vega") or row.get("greeks_vega"))
        theta = _to_float(row.get("theta") or row.get("greeks_theta"))

        if iv is not None:
            try:
                time_to_expiry = max(1e-6, (expiry - as_of.date()).days / 365.0)
                delta = delta if delta is not None else bs_delta(spot, strike, time_to_expiry, self.config.risk_free_rate, iv, option_type)
                gamma = gamma if gamma is not None else bs_gamma(spot, strike, time_to_expiry, self.config.risk_free_rate, iv)
                vega = vega if vega is not None else bs_vega(spot, strike, time_to_expiry, self.config.risk_free_rate, iv)
                theta = theta if theta is not None else bs_theta(spot, strike, time_to_expiry, self.config.risk_free_rate, iv, option_type)
            except Exception:
                pass

        normalized = {
            "as_of_ts": as_of.isoformat(),
            "session_date": as_of.date().isoformat(),
            "symbol_key": self.config.symbol_key,
            "symbol_id": self.config.symbol_id,
            "index_family": self.config.index_family,
            "underlying": self.config.underlying,
            "spot": spot,
            "strike": strike,
            "expiry": expiry.isoformat(),
            "option_type": option_type,
            "symbol": str(row.get("symbol") or row.get("tradingsymbol") or row.get("tradingSymbol") or "").strip(),
            "ltp": ltp,
            "bid": bid,
            "ask": ask,
            "volume": volume,
            "oi": oi,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "iv": iv,
            "mid": mid,
            "dte": max(0, (expiry - as_of.date()).days),
            "expiry_cycle_type": self.config.expiry_cycle_type,
            "source_row": row,
            "schema_version": ARCHIVE_SCHEMA_VERSION,
        }
        return normalized

    def collect_snapshot(self, *, capture_reason: str, force: bool = False) -> ArchiveResult:
        as_of = self._now()
        session_date = as_of.date()
        spot_source = "ltp"
        spot = None
        try:
            spot = self._retry("spot_ltp", lambda: self.client.get_ltp(self.config.spot_symbol))
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("LTP fetch failed for %s: %s", self.config.spot_symbol, exc)
            try:
                candles = self._retry("spot_candles", lambda: self.client.get_candles(self.config.spot_symbol, "5m", limit=1))
                if candles:
                    spot = float(candles[-1].close)
                    spot_source = "candles"
            except Exception as candle_exc:  # noqa: BLE001
                self.logger.error("Spot fallback failed: %s", candle_exc)

        if spot is None:
            raise RuntimeError("Unable to fetch spot price from m.Stock")

        raw_chain = self._retry("option_chain", lambda: self.client.get_option_chain(self.config.chain_underlying))
        chain_rows: List[Dict[str, Any]] = []
        for row in raw_chain or []:
            if not isinstance(row, dict):
                continue
            normalized = self._normalize_chain_row(row, float(spot), as_of)
            if normalized is None:
                continue
            chain_rows.append(normalized)

        if self.config.max_chain_rows and len(chain_rows) > int(self.config.max_chain_rows):
            chain_rows = chain_rows[: int(self.config.max_chain_rows)]

        total_volume = float(sum(float(row.get("volume") or 0.0) for row in chain_rows))
        spread_values = [((float(row["ask"]) - float(row["bid"])) / max(1e-9, float(spot)) * 10000.0) for row in chain_rows if row.get("ask") is not None and row.get("bid") is not None]
        median_spread_bps = sorted(spread_values)[len(spread_values) // 2] if spread_values else 0.0
        volatility_bucket = _volatility_bucket(next((row.get("iv") for row in chain_rows if row.get("iv") is not None), None))
        liquidity_bucket = _liquidity_bucket(total_volume, median_spread_bps)

        expiry_day = self._is_expiry_day(chain_rows, session_date)
        move_points, move_pct = self._spot_move(float(spot))
        snapshot = ArchiveSnapshot(
            snapshot_id=uuid.uuid4().hex,
            as_of_ts=as_of,
            session_date=session_date,
            symbol_key=self.config.symbol_key,
            symbol_id=self.config.symbol_id,
            index_family=self.config.index_family,
            underlying=self.config.underlying,
            spot_symbol=self.config.spot_symbol,
            spot=float(spot),
            source="mstock",
            capture_reason=capture_reason,
            spot_source=spot_source,
            is_expiry_day=expiry_day,
            spot_move_points=move_points,
            spot_move_pct=move_pct,
            volatility_bucket=volatility_bucket,
            liquidity_bucket=liquidity_bucket,
            expiry_cycle_type=self.config.expiry_cycle_type,
            raw_payload={
                "capture_reason": capture_reason,
                "spot_source": spot_source,
                "spot": float(spot),
                "option_chain_rows": len(chain_rows),
            },
            chain_rows=chain_rows,
        )

        run_id = snapshot.snapshot_id
        self.storage.write_run_start(run_id, snapshot)
        try:
            result = self.storage.persist_snapshot(snapshot)
            self.state.last_snapshot_ts = as_of
            self.state.last_snapshot_spot = float(spot)
            if capture_reason in {"base_5m", "market_open", "expiry_boost"}:
                self.state.last_base_snapshot_ts = as_of
                self.state.last_base_snapshot_spot = float(spot)
            self.state.last_snapshot_id = snapshot.snapshot_id
            self.storage.save_state(self.state)
            self.storage.write_run_finish(run_id, "success")
            return result
        except Exception as exc:  # noqa: BLE001
            self.storage.write_run_finish(run_id, "failed", str(exc))
            raise

    def should_capture_now(self, now: Optional[datetime] = None) -> Tuple[bool, List[str]]:
        now = now or self._now()
        reasons: List[str] = []
        if not self._is_market_day(now):
            return False, reasons
        if now < self._session_open(now) or now > self._session_close(now):
            return False, reasons

        if self.state.last_snapshot_ts is None:
            reasons.append("market_open")
            return True, reasons

        due_base = self._next_base_due(now)
        if due_base:
            reasons.append("base_5m")

        if self._is_open_window(now) and (self.state.last_base_snapshot_ts is None or self.state.last_base_snapshot_ts.date() != now.date()):
            reasons.append("market_open")

        if self.state.last_snapshot_spot is not None:
            try:
                spot = self._retry("spot_probe", lambda: self.client.get_ltp(self.config.spot_symbol))
                points = abs(float(spot) - float(self.state.last_snapshot_spot))
                pct = abs((float(spot) / float(self.state.last_snapshot_spot) - 1.0) * 100.0)
                if points >= float(self.config.large_move_points) or pct >= float(self.config.large_move_pct):
                    reasons.append("large_spot_move")
            except Exception:
                pass

        if due_base and self.state.last_snapshot_ts.date() == now.date() and now.time() >= _parse_time(self.config.expiry_boost_start_hhmm):
            reasons.append("expiry_boost")

        return bool(reasons), sorted(set(reasons))

    def run_once(self, *, force: bool = False) -> Optional[ArchiveResult]:
        should, reasons = self.should_capture_now()
        if not should and not force:
            return None
        reason = "+".join(reasons) if reasons else "manual"
        self.logger.info("Collecting snapshot reason=%s", reason)
        return self.collect_snapshot(capture_reason=reason, force=force)

    def run_forever(self) -> None:
        self.logger.info("Starting options archive collector for %s", self.config.underlying)
        while True:
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Collector iteration failed: %s", exc)
            time.sleep(max(5, int(self.config.scheduler_poll_seconds)))

    def run_daily_qa(self, session_date: Optional[date] = None):
        session_date = session_date or self._now().date()
        clean_paths = self.storage.list_clean_paths(session_date=session_date)
        report = self.qa_reporter.build_report(session_date, clean_paths)
        summary = {
            "run_id": self.state.last_snapshot_id or "",
            "symbol_key": self.config.symbol_key,
            "symbol_id": self.config.symbol_id,
            "index_family": self.config.index_family,
            "session_date": report.session_date.isoformat(),
            "snapshot_count": report.snapshot_count,
            "contract_count": report.contract_count,
            "raw_completeness_pct": report.raw_completeness_pct,
            "clean_completeness_pct": report.clean_completeness_pct,
            "missing_iv_rows": report.missing_iv_rows,
            "missing_greeks_rows": report.missing_greeks_rows,
            "duplicate_contract_rows": report.duplicate_contract_rows,
            "status": report.status,
            "details": report.details,
        }
        return self.storage.write_daily_qa_report(session_date, summary)
