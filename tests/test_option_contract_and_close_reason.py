from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import strategy
from market_data import Candle
from datetime import datetime, timedelta, time as dt_time


class _Client:
    def __init__(self, *, token: str = "12345") -> None:
        self.token = token
        self.ltp_calls = 0

    def resolve_exchange_token_symbol(self, symbol, exchange_hint=None):
        return "NFO", self.token, symbol

    def get_ltp(self, key):
        self.ltp_calls += 1
        return 100.0


def test_contract_selection_returns_tokenized_option_contract():
    app = object.__new__(strategy.NiftyScalper)
    app.client = _Client(token="456")
    contract = strategy.NiftyScalper._option_contract_from_row(
        app,
        {"symbol": "NIFTY09JUN23300PE", "exchange": "NFO", "strike": 23300.0, "option_type": "PE", "expiry": "09JUN"},
    )
    assert contract is not None
    assert contract.instrument_token == "456"
    assert contract.tradingsymbol == "NIFTY09JUN23300PE"


def test_missing_token_blocks_entry_safely():
    app = object.__new__(strategy.NiftyScalper)
    app.client = _Client(token="")
    app.cfg = type("Cfg", (), {"enable_live_trading": False})()
    app._can_open_trade_type = lambda **kwargs: True
    app._entry_qty = lambda atr: 1
    app._dynamic_pyramid_max_level = lambda: 0
    app.state = type("State", (), {"open_directional": []})()
    app._entry_block_for_missing_contract_token = lambda **kwargs: setattr(app, "_blocked", kwargs["context"])
    res = strategy.NiftyScalper._open_directional_from_option(
        app,
        {"symbol": "NIFTY09JUN23300PE", "exchange": "NFO", "strike": 23300.0, "option_type": "PE", "expiry": "09JUN"},
        name="short_call",
        spot=23300.0,
        atr_val=100.0,
        append_state=False,
    )
    assert res is None
    assert getattr(app, "_blocked", "") == "short_call"
    assert app.client.ltp_calls == 0


def test_short_call_positive_pnl_not_labeled_stop_loss():
    app = object.__new__(strategy.NiftyScalper)
    reason = strategy.NiftyScalper._normalize_close_reason(app, "stop_loss", 50.0, True)
    assert reason != "STOP_LOSS"


def test_adx_block_happens_before_ml_option_ltp_path():
    now = datetime.now()
    candles = [
        Candle(time=now + timedelta(minutes=i), open=100 + i, high=101 + i, low=99 + i, close=100 + i, volume=1000)
        for i in range(40)
    ]

    class _Cfg:
        max_trades_per_day = 10
        cooldown_sec = 0
        premium_entry_cutoff_hhmm = ""
        polling_interval_sec = 1.0
        timeframe = "1m"
        atr_period = 14
        enable_mtf_confirmation = False
        enable_opening_filter = False
        enable_adx_filter = True
        adx_period = 14
        adx_min_strength = 25.0
        max_atr = 1e9
        min_atr = 0.0
        ema_fast = 9
        ema_slow = 21
        enable_chop_filter = False
        enable_roc_filter = False
        enable_vwap_filter = False
        nifty_weekly_only = False
        bypass_weekly_filter = True
        strategy_name = "directional"
        enable_volume_filter = False
        enable_supertrend_filter = False
        debug_log_no_signal = False
        max_open_positions = 10
        symbol = "NIFTY"
        underlying = "NIFTY"

    class _Client2:
        def get_candles(self, symbol, timeframe, limit):
            return candles

        def get_ltp(self, key):
            return 23300.0

        def get_option_chain(self, symbol):
            return [
                {"symbol": "NIFTY09JUN23300CE", "exchange": "NFO", "token": "1", "strike": 23300.0, "option_type": "CE", "expiry": "09JUN"},
                {"symbol": "NIFTY09JUN23300PE", "exchange": "NFO", "token": "2", "strike": 23300.0, "option_type": "PE", "expiry": "09JUN"},
            ]

    app = object.__new__(strategy.NiftyScalper)
    app.cfg = _Cfg()
    app.client = _Client2()
    app.state = type(
        "State",
        (),
        {
            "open_directional": [],
            "open_multi": [],
            "trades_today": 0,
            "last_entry_ts": 0.0,
            "last_exit_ts": 0.0,
            "last_stopout_ts": 0.0,
        },
    )()
    app._entries_paused = False
    app._on_tick = None
    app._last_router_snapshot = {}
    app._current_entry_recommended_by_gpt = False
    app._iv_pct_hist = []
    app._intraday_only_enabled = lambda: False
    app._risk_status = lambda: (True, "")
    app._now_ist_time = lambda: dt_time(10, 0)
    app._session_overrides = lambda now_t: (0.0, 1e9, 40.0, 60.0, "mid")
    app._log_throttled = lambda *args, **kwargs: None
    app._note_entry_blocked = lambda reason, extras=None: setattr(app, "_blocked_reason", reason)
    app._candle_age_sec = lambda candle: 0.0
    app._vwap = lambda candles: None
    app._try_get_ltp = lambda symbol, exchange=None: 23300.0
    app._portfolio_caps_allow_legs = lambda *args, **kwargs: True
    app._clear_auto_gpt_strike_context = lambda: None
    app._normalize_strategy_name = lambda name: "directional"
    app._note_strategy_considered = lambda *args, **kwargs: None
    app._note_strategy_decision = lambda *args, **kwargs: None
    app._trend_strength = lambda *args, **kwargs: 0.0
    app.evaluate_entry_signals = lambda candles: (_ for _ in ()).throw(AssertionError("ML path should not run before ADX block"))

    original_adx = strategy.adx
    try:
        strategy.adx = lambda highs, lows, closes, period=14: 10.0
        strategy.NiftyScalper._decide_entries(app)
    finally:
        strategy.adx = original_adx

    assert "ADX" in getattr(app, "_blocked_reason", "")
