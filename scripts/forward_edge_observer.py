from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATA_DIR = REPO_ROOT / "data" / "forward_edge_observer"
DB_PATH = DATA_DIR / "forward_edge_observer.db"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(REPO_ROOT / ".scalper.env")
except Exception:
    pass

from config import load_api_config, load_strategy_config
from cost_model import CostModel
from db import DatabaseManager
from greeks import OptionType, delta, gamma, implied_volatility, theta as bs_theta, vega
from market_data import Candle
from ml_signals import evaluate_ml_gating_before_execution, feature_vector_from_candles, load_model
from mstock_client import MStockTypeBClient
from retrain_nifty_1year import clean_frame, prepare_minute_frame
from strategy import NiftyScalper


OBSERVER_ONLY = True
WARNING_BANNER = "FORWARD EDGE OBSERVER ONLY - NO ORDERS WILL BE PLACED"
HORIZON_MINUTES = [1, 3, 5, 10, 15]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research-only forward edge observer. Read-only. No orders.")
    parser.add_argument("--once", action="store_true", help="Capture one observer snapshot and exit.")
    parser.add_argument("--interval-sec", type=int, default=0, help="Continuous observation interval in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Use local-only data path and avoid broker/API reads where possible.")
    parser.add_argument("--export-jsonl", action="store_true", help="Export forward_signals rows to a JSONL file.")
    return parser.parse_args()


def now_ist() -> datetime:
    return datetime.now()


def session_bucket(ts: datetime) -> str:
    hhmm = ts.hour * 60 + ts.minute
    if hhmm < 10 * 60:
        return "OPEN"
    if hhmm >= 14 * 60 + 30:
        return "CLOSE"
    return "MID"


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, default=str, ensure_ascii=False)


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if np.isnan(out) or np.isinf(out):
            return None
        return float(out)
    except Exception:
        return None


def ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def install_execution_guards(client: Optional[MStockTypeBClient]) -> List[str]:
    guarded: List[str] = []
    if client is None:
        return guarded

    def _blocked(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("Observer safety guard: execution API call blocked in observer mode.")

    for name in ("place_order", "modify_order", "cancel_order", "exit_position", "squareoff"):
        if hasattr(client, name):
            try:
                setattr(client, name, _blocked)
                guarded.append(name)
            except Exception:
                pass
    return guarded


class ObserverDB:
    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_dir()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS observer_runs (
                    observer_run_id TEXT PRIMARY KEY,
                    started_at TEXT,
                    mode TEXT,
                    interval_sec INTEGER,
                    observer_only INTEGER,
                    warning_banner TEXT,
                    safety_report_path TEXT,
                    notes_json TEXT
                );

                CREATE TABLE IF NOT EXISTS forward_signals (
                    observer_signal_id TEXT PRIMARY KEY,
                    observer_run_id TEXT,
                    prediction_id TEXT,
                    timestamp TEXT,
                    trading_date TEXT,
                    session_bucket TEXT,
                    spot_symbol TEXT,
                    spot_token TEXT,
                    spot_price REAL,
                    spot_open REAL,
                    spot_high REAL,
                    spot_low REAL,
                    spot_close REAL,
                    spot_volume REAL,
                    selected_option_symbol TEXT,
                    selected_option_token TEXT,
                    CE_or_PE TEXT,
                    strike REAL,
                    expiry TEXT,
                    moneyness REAL,
                    distance_from_spot REAL,
                    option_ltp_at_signal REAL,
                    bid REAL,
                    ask REAL,
                    spread REAL,
                    spread_pct REAL,
                    option_volume REAL,
                    option_oi REAL,
                    option_change_oi REAL,
                    option_iv REAL,
                    option_delta REAL,
                    option_gamma REAL,
                    option_theta REAL,
                    option_vega REAL,
                    model_name TEXT,
                    model_version_path TEXT,
                    label_policy TEXT,
                    predicted_class INTEGER,
                    predicted_probability REAL,
                    confidence_bucket TEXT,
                    threshold_used REAL,
                    raw_model_outputs_json TEXT,
                    feature_snapshot_json TEXT,
                    regime_features_json TEXT,
                    drift_sensitive_features_json TEXT,
                    entry_rule_that_fired TEXT,
                    signal_side TEXT,
                    signal_reason TEXT,
                    risk_gate_would_pass INTEGER,
                    risk_gate_block_reason TEXT,
                    simulated_entry_price REAL,
                    assumed_slippage REAL,
                    assumed_brokerage REAL,
                    assumed_spread_cost REAL,
                    simulated_stop_loss REAL,
                    simulated_target REAL,
                    fixed_horizon_minutes INTEGER,
                    exit_price_after_horizon REAL,
                    max_favorable_excursion REAL,
                    max_adverse_excursion REAL,
                    simulated_gross_pnl REAL,
                    simulated_net_pnl_after_costs REAL,
                    simulated_result_label TEXT,
                    trade_quality_label TEXT,
                    data_quality_flags_json TEXT,
                    quote_staleness_seconds REAL,
                    missing_fields_json TEXT,
                    observer_error TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS forward_signal_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observer_signal_id TEXT,
                    horizon_minutes INTEGER,
                    due_at TEXT,
                    observed_at TEXT,
                    option_ltp_at_horizon REAL,
                    gross_pnl REAL,
                    net_pnl_after_costs REAL,
                    max_favorable_excursion_until_horizon REAL,
                    max_adverse_excursion_until_horizon REAL,
                    hit_target_before_horizon INTEGER,
                    hit_stop_before_horizon INTEGER,
                    result_label TEXT,
                    status TEXT,
                    source_note TEXT,
                    UNIQUE(observer_signal_id, horizon_minutes)
                );

                CREATE TABLE IF NOT EXISTS observer_errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observer_run_id TEXT,
                    timestamp TEXT,
                    stage TEXT,
                    error_text TEXT,
                    payload_json TEXT
                );

                CREATE TABLE IF NOT EXISTS observer_config (
                    config_key TEXT PRIMARY KEY,
                    config_value TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_forward_signals_timestamp ON forward_signals(timestamp);
                CREATE INDEX IF NOT EXISTS idx_forward_signals_trading_date ON forward_signals(trading_date);
                CREATE INDEX IF NOT EXISTS idx_forward_signals_option_symbol ON forward_signals(selected_option_symbol);
                CREATE INDEX IF NOT EXISTS idx_forward_signals_cepe ON forward_signals(CE_or_PE);
                CREATE INDEX IF NOT EXISTS idx_forward_signals_model_name ON forward_signals(model_name);
                CREATE INDEX IF NOT EXISTS idx_forward_signals_predprob ON forward_signals(predicted_probability);
                CREATE INDEX IF NOT EXISTS idx_forward_signals_signal_side ON forward_signals(signal_side);
                CREATE INDEX IF NOT EXISTS idx_forward_signals_result ON forward_signals(simulated_result_label);
                CREATE INDEX IF NOT EXISTS idx_outcomes_signal_horizon ON forward_signal_outcomes(observer_signal_id, horizon_minutes);
                """
            )
            conn.commit()

    def insert_run(self, observer_run_id: str, mode: str, interval_sec: int, safety_report_path: Path, notes: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO observer_runs (
                    observer_run_id, started_at, mode, interval_sec, observer_only, warning_banner, safety_report_path, notes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observer_run_id,
                    now_ist().isoformat(),
                    mode,
                    int(interval_sec),
                    1,
                    WARNING_BANNER,
                    str(safety_report_path),
                    json_dumps(notes),
                ),
            )
            conn.commit()

    def insert_signal(self, record: Dict[str, Any]) -> None:
        keys = list(record.keys())
        values = [record[k] for k in keys]
        placeholders = ",".join(["?"] * len(keys))
        with self._connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO forward_signals ({','.join(keys)}) VALUES ({placeholders})",
                values,
            )
            conn.commit()

    def insert_outcome_stub(self, observer_signal_id: str, horizon_minutes: int, due_at: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO forward_signal_outcomes (
                    observer_signal_id, horizon_minutes, due_at, result_label, status, source_note
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (observer_signal_id, horizon_minutes, due_at, "NO_DATA", "pending", "awaiting_future_quote"),
            )
            conn.commit()

    def update_outcome(self, observer_signal_id: str, horizon_minutes: int, payload: Dict[str, Any]) -> None:
        assignments = ", ".join([f"{k}=?" for k in payload.keys()])
        with self._connect() as conn:
            conn.execute(
                f"UPDATE forward_signal_outcomes SET {assignments} WHERE observer_signal_id=? AND horizon_minutes=?",
                list(payload.values()) + [observer_signal_id, horizon_minutes],
            )
            conn.commit()

    def pending_outcomes(self, as_of: datetime) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT o.*, s.selected_option_symbol, s.selected_option_token, s.option_ltp_at_signal,
                       s.assumed_slippage, s.assumed_brokerage, s.assumed_spread_cost, s.simulated_stop_loss,
                       s.simulated_target, s.signal_side, s.timestamp
                FROM forward_signal_outcomes o
                JOIN forward_signals s ON s.observer_signal_id = o.observer_signal_id
                WHERE o.status='pending' AND o.due_at <= ?
                ORDER BY o.due_at ASC
                """,
                (as_of.isoformat(),),
            ).fetchall()
            return [dict(r) for r in rows]

    def export_jsonl(self, out_path: Path) -> int:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM forward_signals ORDER BY timestamp ASC").fetchall()
        with out_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json_dumps(dict(row)) + "\n")
        return len(rows)

    def signal_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(1) AS c FROM forward_signals").fetchone()
            return int(row["c"]) if row else 0

    def log_error(self, observer_run_id: str, stage: str, error_text: str, payload: Optional[Dict[str, Any]] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO observer_errors (observer_run_id, timestamp, stage, error_text, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (observer_run_id, now_ist().isoformat(), stage, error_text, json_dumps(payload or {})),
            )
            conn.commit()


class ForwardEdgeObserver:
    def __init__(self, *, dry_run: bool = False) -> None:
        self.dry_run = bool(dry_run)
        self.observer_run_id = f"FERUN_{uuid.uuid4().hex[:12].upper()}"
        self.db = ObserverDB(DB_PATH)
        self.cost_model = CostModel()
        self.cfg = load_strategy_config()
        self.api_cfg = load_api_config()
        self.client: Optional[MStockTypeBClient] = None
        self.scalper: Optional[NiftyScalper] = None
        self.ml_model = load_model()
        self.guard_names: List[str] = []
        self.model_path = str((REPO_ROOT / "ml_signal_model.pkl").resolve())
        self.safety_report_path = DATA_DIR / "observer_safety_report.md"
        self._latest_local_frame: Optional[pd.DataFrame] = None
        self._setup_runtime()

    def _setup_runtime(self) -> None:
        if not self.dry_run:
            try:
                self.client = MStockTypeBClient(self.api_cfg)
            except Exception:
                self.client = None
        self.guard_names = install_execution_guards(self.client)
        if self.client is not None:
            try:
                observer_cfg = self.cfg
                observer_cfg.enable_live_trading = False
                observer_cfg.shadow_mode = False
                self.scalper = NiftyScalper(self.client, observer_cfg)
            except Exception:
                self.scalper = None
        self._write_safety_report()

    def _write_safety_report(self) -> None:
        lines = [
            "# Observer Safety Report",
            "",
            f"- Timestamp: `{now_ist().isoformat()}`",
            f"- Observer only: `{OBSERVER_ONLY}`",
            f"- Warning: `{WARNING_BANNER}`",
            f"- No order APIs were called: `True`",
            f"- No broker execution calls were called: `True`",
            f"- No production model was overwritten: `True`",
            f"- Observer writes only to: `{DB_PATH}` and `data/forward_edge_observer/forward_edge_signals_YYYYMMDD.jsonl`",
            f"- Read-only mode: `True`",
            f"- Guarded execution methods: `{', '.join(self.guard_names) if self.guard_names else 'none_installed_or_client_unavailable'}`",
        ]
        ensure_dir()
        self.safety_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def register_run(self, mode: str, interval_sec: int) -> None:
        notes = {
            "dry_run": self.dry_run,
            "client_available": self.client is not None,
            "model_loaded": self.ml_model is not None,
        }
        self.db.insert_run(self.observer_run_id, mode, interval_sec, self.safety_report_path, notes)

    def _load_local_frame(self) -> pd.DataFrame:
        if self._latest_local_frame is None:
            frame, _ = prepare_minute_frame()
            frame, _ = clean_frame(frame)
            self._latest_local_frame = frame
        return self._latest_local_frame.copy()

    def _recent_candles(self, limit: int = 80) -> List[Candle]:
        frame = self._load_local_frame()
        rows = frame.tail(limit)
        candles: List[Candle] = []
        for row in rows.itertuples(index=False):
            candles.append(
                Candle(
                    time=pd.Timestamp(getattr(row, "timestamp")).to_pydatetime(),
                    open=float(getattr(row, "open")),
                    high=float(getattr(row, "high")),
                    low=float(getattr(row, "low")),
                    close=float(getattr(row, "close")),
                    volume=safe_float(getattr(row, "volume", None)),
                )
            )
        return candles

    def _latest_spot_snapshot(self) -> Dict[str, Any]:
        candles = self._recent_candles(limit=5)
        latest = candles[-1]
        snapshot = {
            "spot_symbol": str(getattr(self.cfg, "underlying", "NIFTY") or "NIFTY"),
            "spot_token": str(getattr(self.cfg, "underlying_token", "") or os.getenv("COLLECT_SYMBOL_TOKEN", "26000")),
            "spot_price": float(latest.close),
            "spot_open": float(latest.open),
            "spot_high": float(latest.high),
            "spot_low": float(latest.low),
            "spot_close": float(latest.close),
            "spot_volume": safe_float(latest.volume),
            "candles": candles,
        }
        if self.client is not None and not self.dry_run:
            try:
                symbol = str(getattr(self.cfg, "underlying", "NIFTY") or "NIFTY")
                exchange = str(getattr(self.cfg, "underlying_exchange", "NSE") or "NSE")
                live_spot = float(self.client.get_ltp(f"{exchange}:{symbol}"))
                snapshot["spot_price"] = live_spot
                snapshot["spot_close"] = live_spot
            except Exception:
                pass
        return snapshot

    def _select_option_contract(self, spot_price: float, signal_side: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        missing_fields: List[str] = []
        if self.client is None or self.dry_run:
            missing_fields.extend(
                [
                    "selected_option_symbol",
                    "selected_option_token",
                    "strike",
                    "expiry",
                    "option_ltp_at_signal",
                    "bid",
                    "ask",
                    "option_oi",
                    "option_volume",
                ]
            )
            return None, missing_fields
        try:
            chain = self.client.get_option_chain(str(getattr(self.cfg, "underlying", "NIFTY") or "NIFTY"))
        except Exception:
            chain = []
        if not chain:
            missing_fields.extend(["selected_option_symbol", "selected_option_token", "strike", "expiry"])
            return None, missing_fields
        desired = "CE" if signal_side.upper() == "BUY_CALL" else "PE"
        same_side = [row for row in chain if str(row.get("option_type") or "").upper() == desired]
        candidates = same_side or chain
        best = min(candidates, key=lambda row: abs(float(row.get("strike") or 0.0) - float(spot_price)))
        return dict(best), missing_fields

    def _quote_option(self, contract: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
        fields = {
            "selected_option_symbol": None,
            "selected_option_token": None,
            "CE_or_PE": None,
            "strike": None,
            "expiry": None,
            "moneyness": None,
            "distance_from_spot": None,
            "option_ltp_at_signal": None,
            "bid": None,
            "ask": None,
            "spread": None,
            "spread_pct": None,
            "option_volume": None,
            "option_oi": None,
            "option_change_oi": None,
            "option_iv": None,
            "option_delta": None,
            "option_gamma": None,
            "option_theta": None,
            "option_vega": None,
        }
        missing: List[str] = []
        if not contract:
            missing.extend(fields.keys())
            return fields, missing
        fields["selected_option_symbol"] = str(contract.get("symbol") or "")
        fields["selected_option_token"] = str(contract.get("token") or "")
        fields["CE_or_PE"] = str(contract.get("option_type") or "")
        fields["strike"] = safe_float(contract.get("strike"))
        fields["expiry"] = str(contract.get("expiry") or "")
        fields["option_volume"] = safe_float(contract.get("volume") or contract.get("vol"))
        fields["option_oi"] = safe_float(contract.get("oi") or contract.get("open_interest"))
        fields["option_change_oi"] = safe_float(contract.get("change_oi") or contract.get("oi_change"))
        fields["option_iv"] = safe_float(contract.get("iv") or contract.get("implied_volatility"))
        bid = safe_float(contract.get("bid"))
        ask = safe_float(contract.get("ask"))
        fields["bid"] = bid
        fields["ask"] = ask
        if bid is not None and ask is not None:
            fields["spread"] = ask - bid
            midpoint = (ask + bid) / 2.0 if (ask + bid) > 0 else None
            fields["spread_pct"] = ((ask - bid) / midpoint) if midpoint else None
        if self.client is not None:
            try:
                symbol = str(contract.get("symbol") or "")
                token = str(contract.get("token") or "")
                exchange = str(contract.get("exchange") or "NFO")
                quote_symbol = f"{exchange}:{token}" if token else f"{exchange}:{symbol}"
                fields["option_ltp_at_signal"] = float(self.client.get_ltp(quote_symbol))
            except Exception:
                fields["option_ltp_at_signal"] = safe_float(contract.get("ltp"))
        else:
            fields["option_ltp_at_signal"] = safe_float(contract.get("ltp"))

        for key in ("selected_option_symbol", "selected_option_token", "strike", "expiry", "option_ltp_at_signal"):
            if fields.get(key) in (None, ""):
                missing.append(key)
        return fields, missing

    def _compute_greeks(self, spot_price: float, option_fields: Dict[str, Any]) -> None:
        opt_price = safe_float(option_fields.get("option_ltp_at_signal"))
        strike = safe_float(option_fields.get("strike"))
        iv = safe_float(option_fields.get("option_iv"))
        side = str(option_fields.get("CE_or_PE") or "").upper()
        if opt_price is None or strike is None or spot_price <= 0 or side not in {"CE", "PE"}:
            return
        opt_type = OptionType.CALL if side == "CE" else OptionType.PUT
        tte = 1.0 / 365.0
        rate = 0.06
        try:
            if iv is None:
                iv = float(implied_volatility(opt_price, spot_price, strike, tte, rate, opt_type))
                option_fields["option_iv"] = iv
        except Exception:
            iv = None
        if iv is None:
            return
        try:
            option_fields["option_delta"] = float(delta(spot_price, strike, tte, rate, iv, opt_type))
        except Exception:
            pass
        try:
            option_fields["option_gamma"] = float(gamma(spot_price, strike, tte, rate, iv))
        except Exception:
            pass
        try:
            option_fields["option_theta"] = float(bs_theta(spot_price, strike, tte, rate, iv, opt_type))
        except Exception:
            pass
        try:
            option_fields["option_vega"] = float(vega(spot_price, strike, tte, rate, iv))
        except Exception:
            pass

    def _confidence_bucket(self, probability: float) -> str:
        lower = max(0.0, min(0.9, np.floor(float(probability) * 10.0) / 10.0))
        upper = min(1.0, lower + 0.1)
        return f"{lower:.1f}-{upper:.1f}"

    def _label_policy(self) -> str:
        try:
            return str((getattr(self.ml_model, "metrics", {}) or {}).get("label_policy") or "unknown")
        except Exception:
            return "unknown"

    def _model_name(self) -> str:
        try:
            return str((getattr(self.ml_model, "metrics", {}) or {}).get("model_type") or "ml_signal_model")
        except Exception:
            return "ml_signal_model"

    def _build_feature_payload(self, candles: Sequence[Candle], spot_price: float, option_fields: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], float, str]:
        context = {
            "spot": spot_price,
            "option_price": option_fields.get("option_ltp_at_signal"),
            "iv": option_fields.get("option_iv"),
            "delta": option_fields.get("option_delta"),
            "gamma": option_fields.get("option_gamma"),
            "theta": option_fields.get("option_theta"),
            "vega": option_fields.get("option_vega"),
            "volume_sma": option_fields.get("option_volume"),
            "dte_norm": 1.0 / 30.0,
        }
        features, names = feature_vector_from_candles(candles, context=context, lookback=20)
        feature_snapshot = {str(name): float(features[idx]) for idx, name in enumerate(names) if idx < len(features)}
        regime_features = {
            key: feature_snapshot.get(key)
            for key in ("regime_trending", "regime_volatile", "regime_mean_reverting", "regime_quiet", "is_opening_session", "is_closing_session", "is_midday_lull")
        }
        drift_sensitive = {key: feature_snapshot.get(key) for key in ("ret_std", "realized_vol_30", "atr_14", "atr_pct")}
        prediction_id, probability = evaluate_ml_gating_before_execution(feature_snapshot, self.ml_model, list(names))
        return feature_snapshot, regime_features, drift_sensitive, float(probability), str(prediction_id)

    def _signal_from_probability(self, probability: float) -> Tuple[int, str, str]:
        predicted_class = 1 if probability >= 0.5 else 0
        if predicted_class == 1:
            return 1, "BUY_CALL", "ml_probability_above_threshold"
        return 0, "BUY_PUT", "ml_probability_below_threshold"

    def _simulate_trade(self, signal_side: str, option_fields: Dict[str, Any]) -> Dict[str, Any]:
        option_ltp = safe_float(option_fields.get("option_ltp_at_signal"))
        spread = safe_float(option_fields.get("spread")) or 0.0
        assumed_slippage = float((option_ltp or 0.0) * 0.001) if option_ltp else 0.0
        assumed_brokerage = float(getattr(self.cost_model.assumptions, "brokerage_per_side_pct", 0.0) or 0.0)
        assumed_spread_cost = float(spread / 2.0) if spread else 0.0
        if option_ltp is None:
            return {
                "simulated_entry_price": None,
                "assumed_slippage": assumed_slippage,
                "assumed_brokerage": assumed_brokerage,
                "assumed_spread_cost": assumed_spread_cost,
                "simulated_stop_loss": None,
                "simulated_target": None,
                "fixed_horizon_minutes": 15,
                "exit_price_after_horizon": None,
                "max_favorable_excursion": None,
                "max_adverse_excursion": None,
                "simulated_gross_pnl": None,
                "simulated_net_pnl_after_costs": None,
                "simulated_result_label": "NO_DATA",
                "trade_quality_label": "NO_DATA",
            }
        entry = option_ltp + assumed_slippage + assumed_spread_cost
        stop = entry * 0.92
        target = entry * 1.10
        return {
            "simulated_entry_price": entry,
            "assumed_slippage": assumed_slippage,
            "assumed_brokerage": assumed_brokerage,
            "assumed_spread_cost": assumed_spread_cost,
            "simulated_stop_loss": stop,
            "simulated_target": target,
            "fixed_horizon_minutes": 15,
            "exit_price_after_horizon": None,
            "max_favorable_excursion": None,
            "max_adverse_excursion": None,
            "simulated_gross_pnl": None,
            "simulated_net_pnl_after_costs": None,
            "simulated_result_label": "NO_DATA",
            "trade_quality_label": "NO_DATA",
        }

    def capture_signal(self) -> Optional[str]:
        ts = now_ist()
        data_quality_flags: List[str] = []
        missing_fields: List[str] = []
        observer_error = None
        try:
            spot = self._latest_spot_snapshot()
            candles = spot.pop("candles")
            spot_price = float(spot["spot_price"])
            placeholder_option_fields = {
                "option_ltp_at_signal": None,
                "option_iv": None,
                "option_delta": None,
                "option_gamma": None,
                "option_theta": None,
                "option_vega": None,
                "option_volume": None,
            }
            feature_snapshot, regime_features, drift_sensitive, probability, prediction_id = self._build_feature_payload(
                candles, spot_price, placeholder_option_fields
            )
            predicted_class, signal_side, signal_reason = self._signal_from_probability(probability)
            contract, contract_missing = self._select_option_contract(spot_price, signal_side)
            missing_fields.extend(contract_missing)
            option_fields, quote_missing = self._quote_option(contract)
            missing_fields.extend(quote_missing)
            option_fields["distance_from_spot"] = (safe_float(option_fields.get("strike")) or 0.0) - spot_price if option_fields.get("strike") is not None else None
            option_fields["moneyness"] = abs(float(option_fields["distance_from_spot"])) if option_fields.get("distance_from_spot") is not None else None
            self._compute_greeks(spot_price, option_fields)
            feature_snapshot, regime_features, drift_sensitive, probability, prediction_id = self._build_feature_payload(
                candles, spot_price, option_fields
            )
            predicted_class, signal_side, signal_reason = self._signal_from_probability(probability)
            threshold = 0.5
            risk_gate_pass = option_fields.get("option_ltp_at_signal") is not None
            risk_gate_block_reason = "" if risk_gate_pass else "missing_option_quote"
            simulation = self._simulate_trade(signal_side, option_fields)
            candle_ts = candles[-1].time
            if getattr(candle_ts, "tzinfo", None) is not None and getattr(ts, "tzinfo", None) is None:
                ts_for_diff = datetime.now(candle_ts.tzinfo)
            else:
                ts_for_diff = ts
            quote_staleness_seconds = max(0.0, (ts_for_diff - candle_ts).total_seconds())
            if quote_staleness_seconds > 120:
                data_quality_flags.append("stale_underlying_candles")
            if not risk_gate_pass:
                data_quality_flags.append("missing_option_quote")

            record = {
                "observer_signal_id": f"FESIG_{uuid.uuid4().hex[:12].upper()}",
                "observer_run_id": self.observer_run_id,
                "prediction_id": prediction_id,
                "timestamp": ts.isoformat(),
                "trading_date": ts.date().isoformat(),
                "session_bucket": session_bucket(ts),
                **spot,
                **option_fields,
                "model_name": self._model_name(),
                "model_version_path": self.model_path,
                "label_policy": self._label_policy(),
                "predicted_class": predicted_class,
                "predicted_probability": probability,
                "confidence_bucket": self._confidence_bucket(probability),
                "threshold_used": threshold,
                "raw_model_outputs_json": json_dumps({"probability": probability}),
                "feature_snapshot_json": json_dumps(feature_snapshot),
                "regime_features_json": json_dumps(regime_features),
                "drift_sensitive_features_json": json_dumps(drift_sensitive),
                "entry_rule_that_fired": "observer_ml_snapshot",
                "signal_side": signal_side,
                "signal_reason": signal_reason,
                "risk_gate_would_pass": 1 if risk_gate_pass else 0,
                "risk_gate_block_reason": risk_gate_block_reason,
                **simulation,
                "data_quality_flags_json": json_dumps(sorted(set(data_quality_flags))),
                "quote_staleness_seconds": quote_staleness_seconds,
                "missing_fields_json": json_dumps(sorted(set(missing_fields))),
                "observer_error": observer_error,
            }
            self.db.insert_signal(record)
            for horizon in HORIZON_MINUTES:
                due_at = (ts + timedelta(minutes=horizon)).isoformat()
                self.db.insert_outcome_stub(record["observer_signal_id"], horizon, due_at)
            return str(record["observer_signal_id"])
        except Exception as exc:
            observer_error = str(exc)
            self.db.log_error(self.observer_run_id, "capture_signal", observer_error)
            return None

    def _current_option_ltp(self, selected_option_symbol: str, selected_option_token: str) -> Optional[float]:
        if self.client is None:
            return None
        try:
            if selected_option_token:
                return float(self.client.get_ltp(f"NFO:{selected_option_token}"))
            if selected_option_symbol:
                return float(self.client.get_ltp(f"NFO:{selected_option_symbol}"))
        except Exception:
            return None
        return None

    def refresh_pending_outcomes(self) -> int:
        updated = 0
        pending = self.db.pending_outcomes(now_ist())
        for row in pending:
            current_ltp = None if self.dry_run else self._current_option_ltp(str(row.get("selected_option_symbol") or ""), str(row.get("selected_option_token") or ""))
            entry = safe_float(row.get("option_ltp_at_signal"))
            slippage = safe_float(row.get("assumed_slippage")) or 0.0
            brokerage = safe_float(row.get("assumed_brokerage")) or 0.0
            spread_cost = safe_float(row.get("assumed_spread_cost")) or 0.0
            stop = safe_float(row.get("simulated_stop_loss"))
            target = safe_float(row.get("simulated_target"))
            if current_ltp is None or entry is None:
                payload = {
                    "observed_at": now_ist().isoformat(),
                    "status": "no_data",
                    "result_label": "NO_DATA",
                    "source_note": "quote_unavailable",
                }
            else:
                gross = current_ltp - entry
                net = gross - slippage - brokerage - spread_cost
                result = "WIN" if net > 0 else "LOSS" if net < 0 else "BREAKEVEN"
                payload = {
                    "observed_at": now_ist().isoformat(),
                    "option_ltp_at_horizon": current_ltp,
                    "gross_pnl": gross,
                    "net_pnl_after_costs": net,
                    "max_favorable_excursion_until_horizon": max(0.0, gross),
                    "max_adverse_excursion_until_horizon": min(0.0, gross),
                    "hit_target_before_horizon": 1 if (target is not None and current_ltp >= target) else 0,
                    "hit_stop_before_horizon": 1 if (stop is not None and current_ltp <= stop) else 0,
                    "result_label": result,
                    "status": "complete",
                    "source_note": "live_quote_snapshot",
                }
            self.db.update_outcome(str(row["observer_signal_id"]), int(row["horizon_minutes"]), payload)
            updated += 1
        return updated


def export_jsonl(db: ObserverDB) -> Path:
    out_path = DATA_DIR / f"forward_edge_signals_{now_ist().strftime('%Y%m%d')}.jsonl"
    db.export_jsonl(out_path)
    return out_path


def main() -> None:
    args = parse_args()
    mode = "dry-run" if args.dry_run else "once" if args.once else "continuous" if args.interval_sec > 0 else "once"
    observer = ForwardEdgeObserver(dry_run=args.dry_run)
    observer.register_run(mode, int(args.interval_sec or 0))

    print(WARNING_BANNER)
    if args.export_jsonl and not args.once and args.interval_sec <= 0 and not args.dry_run:
        path = export_jsonl(observer.db)
        print(json.dumps({"exported_jsonl": str(path), "signal_count": observer.db.signal_count()}, indent=2))
        return

    if args.once or args.dry_run or args.interval_sec <= 0:
        signal_id = observer.capture_signal()
        updated = observer.refresh_pending_outcomes()
        jsonl_path = export_jsonl(observer.db)
        print(
            json.dumps(
                {
                    "observer_run_id": observer.observer_run_id,
                    "signal_logged": bool(signal_id),
                    "observer_signal_id": signal_id,
                    "signals_total": observer.db.signal_count(),
                    "pending_outcomes_updated": updated,
                    "db_path": str(DB_PATH),
                    "jsonl_path": str(jsonl_path),
                    "safety_report_path": str(observer.safety_report_path),
                },
                indent=2,
            )
        )
        return

    while True:
        signal_id = observer.capture_signal()
        updated = observer.refresh_pending_outcomes()
        print(json.dumps({"timestamp": now_ist().isoformat(), "signal_id": signal_id, "outcomes_updated": updated}, indent=2))
        time.sleep(max(5, int(args.interval_sec)))


if __name__ == "__main__":
    main()
