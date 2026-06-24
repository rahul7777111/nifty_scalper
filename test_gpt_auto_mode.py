from __future__ import annotations

import sys
import types
from datetime import datetime
import importlib

from src.config import StrategyConfig
from src.market_data import Candle


if "gpt_advisor" not in sys.modules:
    sys.modules["gpt_advisor"] = importlib.import_module("src.gpt_advisor")

from src import strategy as strategy_mod
from src.strategy import NiftyScalper, TradeState


class _FakeAnalysis:
    preset_request = ""
    preset_reason = ""
    confidence = 0.91
    recommended_strategy = "directional"
    strategy_parameters = {}
    directional_strike_offset_steps = None
    reason = "use directional"


class _FakeClient:
    def __init__(self, candles: list[Candle], chain: list[dict[str, object]]) -> None:
        self._candles = candles
        self._chain = chain

    def get_candles(self, *args, **kwargs):
        return list(self._candles)

    def get_option_chain(self, *args, **kwargs):
        return list(self._chain)


def _make_bot(cfg: StrategyConfig, client: object) -> NiftyScalper:
    bot = NiftyScalper.__new__(NiftyScalper)
    bot.cfg = cfg
    bot.client = client
    bot.state = TradeState(open_orders=[], open_multi=[], open_directional=[])
    bot._trade_seq = 0
    bot._iv_pct_hist = []
    bot._entry_spread_watch = {}
    bot._strategy_winrate = {}
    bot._disabled_strategies = {}
    bot._last_router_snapshot = {}
    bot._last_auto_fallback_note = ""
    bot._auto_gpt_strategy_parameters = {}
    bot._auto_gpt_directional_steps = None
    bot._auto_gpt_strike_context = {}
    bot._on_tick = None
    # initialize counters used by entry-blocking helpers
    bot._entry_block_counts = {}
    bot._last_entry_block = {"code": "", "reason": "", "ts": 0.0}
    bot._strategy_considered_counts = {}
    bot._strategy_decision_counts = {}
    bot._last_trade_type_key = None
    bot._last_trade_type_streak = 0
    bot._strategy_selected_counts = {}
    bot._event_sink = None
    bot._ltp_watch = {}
    bot._auto_delta_hedge_symbol_cache = {}
    return bot


def test_gpt_auto_select_returns_directional(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    import sys as _sys
    _sys.modules.setdefault("gpt_advisor", types.SimpleNamespace())
    _sys.modules.setdefault("src.gpt_advisor", _sys.modules["gpt_advisor"])
    monkeypatch.setattr(_sys.modules["gpt_advisor"], "analyze_market", lambda **kwargs: _FakeAnalysis(), raising=False)

    cfg = StrategyConfig(strategy_name="auto", gpt_enable=True, gpt_auto_select=True)
    bot = _make_bot(
        cfg,
        _FakeClient(
            candles=[],
            chain=[
                {"symbol": "NIFTY100CE", "strike": 100, "option_type": "CE", "ltp": 10, "exchange": "NFO"},
                {"symbol": "NIFTY100PE", "strike": 100, "option_type": "PE", "ltp": 10, "exchange": "NFO"},
            ],
        ),
    )

    bot._gpt_open_positions_mtm_snapshot = lambda max_trades=6: []
    bot._estimate_option_theta_from_chain_row = lambda row, spot: None
    bot._row_expiry_date = lambda row: None
    bot._normalize_strategy_name = lambda name: str(name or "").strip().lower()
    bot._gpt_allowed_strategies_for_auto = lambda: ["directional", "long_call", "short_put"]

    rec = bot._gpt_auto_select_strategy(
        chain=[
            {"symbol": "NIFTY100CE", "strike": 100, "option_type": "CE", "ltp": 10, "exchange": "NFO"},
            {"symbol": "NIFTY100PE", "strike": 100, "option_type": "PE", "ltp": 10, "exchange": "NFO"},
        ],
        spot=100.0,
        atr_val=10.0,
        rsi_val=55.0,
        trend_strength=0.01,
        vwap_val=100.0,
        call_atm={"symbol": "NIFTY100CE", "strike": 100, "ltp": 10, "exchange": "NFO"},
        put_atm={"symbol": "NIFTY100PE", "strike": 100, "ltp": 10, "exchange": "NFO"},
    )

    assert rec == "directional"


def test_auto_gpt_directional_opens_trade(monkeypatch) -> None:
    candles = [
        Candle(
            time=datetime(2026, 5, 20, 9, 15 + i),
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.0 + i,
            volume=1000.0,
        )
        for i in range(30)
    ]
    chain = [
        {"symbol": "NIFTY100CE", "strike": 100, "option_type": "CE", "ltp": 10.0, "exchange": "NFO", "token": "1"},
        {"symbol": "NIFTY100PE", "strike": 100, "option_type": "PE", "ltp": 10.0, "exchange": "NFO", "token": "2"},
    ]
    cfg = StrategyConfig(
        strategy_name="auto",
        gpt_enable=True,
        gpt_auto_select=True,
        gpt_apply_paper=True,
        enable_live_trading=False,
        symbol="NIFTY",
        underlying="NIFTY",
        timeframe="1m",
        max_trades_per_day=10,
        max_open_positions=6,
        cooldown_sec=0.0,
        max_stale_ltp_sec=0.0,
        entry_candle_max_age_sec=0.0,
        enable_opening_filter=False,
        enable_vwap_filter=False,
        enable_adx_filter=False,
        enable_volume_filter=False,
        enable_supertrend_filter=False,
        enable_roc_filter=False,
        enable_choppiness_filter=False,
        enable_mtf_confirmation=False,
        enable_rsi_confluence=False,
        enable_premium_rsi_filter=False,
        dir_min_confirmations=1,
        dir_min_score_diff=0,
        dir_allow_tie_break_entries=True,
        auto_strategy_lock_minutes=0,
        lot_size=50,
    )
    bot = _make_bot(cfg, _FakeClient(candles=candles, chain=chain))

    monkeypatch.setattr(strategy_mod, "ema", lambda closes, period: 101.0 if int(period) == 9 else 100.0)
    monkeypatch.setattr(strategy_mod, "rsi", lambda closes, period: 60.0)
    monkeypatch.setattr(strategy_mod, "atr", lambda highs, lows, closes, period: 10.0)
    monkeypatch.setattr(strategy_mod, "adx", lambda highs, lows, closes, period: 30.0)
    monkeypatch.setattr(strategy_mod, "roc", lambda closes, period: 1.0)
    monkeypatch.setattr(strategy_mod, "choppiness_index", lambda highs, lows, closes, period: 45.0)
    monkeypatch.setattr(strategy_mod, "supertrend", lambda highs, lows, closes, period, multiplier: 95.0)
    monkeypatch.setattr(strategy_mod, "is_bullish_engulfing", lambda candles: False)
    monkeypatch.setattr(strategy_mod, "is_bearish_engulfing", lambda candles: False)
    monkeypatch.setattr(strategy_mod, "is_doji", lambda candles: False)
    monkeypatch.setattr(strategy_mod, "is_hammer", lambda candles: False)
    monkeypatch.setattr(strategy_mod, "is_shooting_star", lambda candles: False)
    monkeypatch.setattr(bot, "_now_ist_time", lambda: datetime(2026, 5, 20, 10, 0).time())
    monkeypatch.setattr(bot, "_session_overrides", lambda now_t: (0.0, 999999.0, 30.0, 70.0, "mid"))
    monkeypatch.setattr(bot, "_risk_status", lambda: (True, ""))
    monkeypatch.setattr(bot, "_strategy_router", lambda **kwargs: {"candidates": ["directional"], "selected": "directional"})
    monkeypatch.setattr(bot, "_gpt_auto_select_strategy", lambda **kwargs: "directional")
    bot._last_gpt_bias = "CE"
    monkeypatch.setattr(bot, "_clear_auto_gpt_strike_context", lambda: None)
    monkeypatch.setattr(bot, "_set_auto_gpt_strike_context", lambda **kwargs: None)
    monkeypatch.setattr(bot, "_filter_weekly_only", lambda chain: list(chain))
    monkeypatch.setattr(bot, "_select_atm_option", lambda chain, spot, kind: chain[0] if kind == "CE" else chain[1])
    monkeypatch.setattr(bot, "_try_get_ltp", lambda symbol, exchange=None: 100.0 if symbol == "NIFTY" else 10.0)
    monkeypatch.setattr(bot, "_entry_price_bounds_ok", lambda price: (True, ""))
    monkeypatch.setattr(bot, "_check_entry_liquidity", lambda legs: (True, ""))
    monkeypatch.setattr(bot, "_gpt_entry_gate", lambda **kwargs: True)
    monkeypatch.setattr(bot, "_resolve_directional_trade_style", lambda **kwargs: "long")
    monkeypatch.setattr(bot, "_get_directional_strike_offset", lambda atr_val: 0.0)
    monkeypatch.setattr(bot, "_pick_nearest_strike", lambda chain, option_type, target_strike: chain[0] if option_type == "CE" else chain[1])
    monkeypatch.setattr(bot, "_can_open_trade_type", lambda **kwargs: True)

    bot._decide_entries()

    assert len(bot.state.open_directional) == 1
    assert bot.state.open_directional[0]["name"] == "long_call"
    assert bot.state.open_directional[0]["side"] == "BUY"


def test_gpt_entry_gate_handles_dict_advice(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    import sys as _sys
    _sys.modules.setdefault("gpt_advisor", types.SimpleNamespace())
    _sys.modules.setdefault("src.gpt_advisor", _sys.modules["gpt_advisor"])
    monkeypatch.setattr(
        _sys.modules["gpt_advisor"],
        "advise_trade",
        lambda **kwargs: {"decision": "TAKE", "reason": "valid entry"},
        raising=False,
    )

    cfg = StrategyConfig(strategy_name="auto", gpt_enable=True, gpt_mode="gate", gpt_apply_paper=True)
    bot = _make_bot(cfg, _FakeClient(candles=[], chain=[]))
    bot._gpt_should_apply_now = lambda is_paper: True

    allowed = bot._gpt_entry_gate(proposal={"trade_name": "long_call"}, is_paper=True)

    assert allowed is True
    assert getattr(bot, "_last_gpt_gate_decision", "") == "TAKE"


def test_gpt_auto_select_alias_normalization(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    import sys as _sys
    _sys.modules.setdefault("gpt_advisor", types.SimpleNamespace())
    _sys.modules.setdefault("src.gpt_advisor", _sys.modules["gpt_advisor"])

    class AnalysisMock:
        preset_request = ""
        preset_reason = ""
        confidence = 0.95
        recommended_strategy = "bull call"
        strategy_parameters = {}
        directional_strike_offset_steps = None
        reason = "good setup"
        ce_pe_bias = "CE"

    monkeypatch.setattr(_sys.modules["gpt_advisor"], "analyze_market", lambda **kwargs: AnalysisMock(), raising=False)

    cfg = StrategyConfig(strategy_name="auto", gpt_enable=True, gpt_auto_select=True)
    bot = _make_bot(
        cfg,
        _FakeClient(
            candles=[],
            chain=[
                {"symbol": "NIFTY100CE", "strike": 100, "option_type": "CE", "ltp": 10, "exchange": "NFO"},
                {"symbol": "NIFTY100PE", "strike": 100, "option_type": "PE", "ltp": 10, "exchange": "NFO"},
            ],
        ),
    )

    bot._gpt_open_positions_mtm_snapshot = lambda max_trades=6: []
    bot._estimate_option_theta_from_chain_row = lambda row, spot: None
    bot._row_expiry_date = lambda row: None

    rec = bot._gpt_auto_select_strategy(
        chain=[
            {"symbol": "NIFTY100CE", "strike": 100, "option_type": "CE", "ltp": 10, "exchange": "NFO"},
        ],
        spot=100.0,
        atr_val=10.0,
        rsi_val=55.0,
        trend_strength=0.01,
        vwap_val=100.0,
        call_atm={"symbol": "NIFTY100CE", "strike": 100, "ltp": 10, "exchange": "NFO"},
        put_atm={"symbol": "NIFTY100PE", "strike": 100, "ltp": 10, "exchange": "NFO"},
    )

    # "bull call" resolves/normalizes to "bull_call_spread"
    assert rec == "bull_call_spread"
    assert bot._last_gpt_bias == "CE"


def test_gpt_multileg_bypass(monkeypatch) -> None:
    candles = [
        Candle(time=datetime(2026, 5, 20, 9, 15 + i), open=100.0, high=100.0, low=100.0, close=100.0, volume=1000.0)
        for i in range(30)
    ]
    chain = [
        {"symbol": "NIFTY100CE", "strike": 100, "option_type": "CE", "ltp": 10.0, "exchange": "NFO", "token": "1"},
        {"symbol": "NIFTY100PE", "strike": 100, "option_type": "PE", "ltp": 10.0, "exchange": "NFO", "token": "2"},
    ]
    cfg = StrategyConfig(
        strategy_name="auto",
        gpt_enable=True,
        gpt_auto_select=True,
        gpt_apply_paper=True,
        enable_live_trading=False,
        symbol="NIFTY",
        underlying="NIFTY",
        timeframe="1m",
        enable_trend_filter=True,
        enable_premium_rsi_filter=True,
        enable_vwap_filter=True,
        max_trend_strength=0.01,
        vwap_max_dev_pct=0.01,
        lot_size=50,
    )
    bot = _make_bot(cfg, _FakeClient(candles=candles, chain=chain))

    monkeypatch.setattr(strategy_mod, "ema", lambda closes, period: 150.0 if int(period) == 9 else 100.0)
    monkeypatch.setattr(strategy_mod, "rsi", lambda closes, period: 95.0)
    monkeypatch.setattr(strategy_mod, "atr", lambda highs, lows, closes, period: 10.0)
    monkeypatch.setattr(bot, "_now_ist_time", lambda: datetime(2026, 5, 20, 10, 0).time())
    monkeypatch.setattr(bot, "_session_overrides", lambda now_t: (0.0, 999999.0, 30.0, 70.0, "mid"))
    monkeypatch.setattr(bot, "_risk_status", lambda: (True, ""))
    monkeypatch.setattr(bot, "_gpt_auto_select_strategy", lambda **kwargs: "short_straddle")

    enter_called = False
    def mock_enter(chain_arg, spot_arg, atr_val):
        nonlocal enter_called
        enter_called = True

    monkeypatch.setattr(bot, "_enter_short_straddle", mock_enter)
    monkeypatch.setattr(bot, "_filter_weekly_only", lambda chain: list(chain))
    monkeypatch.setattr(bot, "_select_atm_option", lambda chain, spot, kind: chain[0] if kind == "CE" else chain[1])
    monkeypatch.setattr(bot, "_try_get_ltp", lambda symbol, exchange=None: 100.0 if symbol == "NIFTY" else 10.0)
    monkeypatch.setattr(bot, "_entry_price_bounds_ok", lambda price: (True, ""))
    monkeypatch.setattr(bot, "_check_entry_liquidity", lambda legs: (True, ""))
    monkeypatch.setattr(bot, "_gpt_entry_gate", lambda **kwargs: True)

    bot._decide_entries()
    assert enter_called is True


def test_gpt_directional_bypass(monkeypatch) -> None:
    candles = [
        Candle(time=datetime(2026, 5, 20, 9, 15 + i), open=100.0, high=100.0, low=100.0, close=100.0, volume=1000.0)
        for i in range(30)
    ]
    chain = [
        {"symbol": "NIFTY100CE", "strike": 100, "option_type": "CE", "ltp": 10.0, "exchange": "NFO", "token": "1"},
        {"symbol": "NIFTY100PE", "strike": 100, "option_type": "PE", "ltp": 10.0, "exchange": "NFO", "token": "2"},
    ]
    cfg = StrategyConfig(
        strategy_name="auto",
        gpt_enable=True,
        gpt_auto_select=True,
        gpt_apply_paper=True,
        enable_live_trading=False,
        symbol="NIFTY",
        underlying="NIFTY",
        timeframe="1m",
        enable_rsi_confluence=True,
        enable_adx_filter=True,
        enable_volume_filter=True,
        lot_size=50,
    )
    bot = _make_bot(cfg, _FakeClient(candles=candles, chain=chain))

    monkeypatch.setattr(strategy_mod, "ema", lambda closes, period: 101.0 if int(period) == 9 else 100.0)
    monkeypatch.setattr(strategy_mod, "rsi", lambda closes, period: 80.0)
    monkeypatch.setattr(strategy_mod, "atr", lambda highs, lows, closes, period: 10.0)
    monkeypatch.setattr(strategy_mod, "adx", lambda highs, lows, closes, period: 10.0)
    monkeypatch.setattr(bot, "_now_ist_time", lambda: datetime(2026, 5, 20, 10, 0).time())
    monkeypatch.setattr(bot, "_session_overrides", lambda now_t: (0.0, 999999.0, 30.0, 70.0, "mid"))
    monkeypatch.setattr(bot, "_risk_status", lambda: (True, ""))
    
    monkeypatch.setattr(bot, "_gpt_auto_select_strategy", lambda **kwargs: "long_put")
    bot._last_gpt_bias = "PE"

    monkeypatch.setattr(bot, "_filter_weekly_only", lambda chain: list(chain))
    monkeypatch.setattr(bot, "_select_atm_option", lambda chain, spot, kind: chain[0] if kind == "CE" else chain[1])
    monkeypatch.setattr(bot, "_try_get_ltp", lambda symbol, exchange=None: 100.0 if symbol == "NIFTY" else 10.0)
    monkeypatch.setattr(bot, "_entry_price_bounds_ok", lambda price: (True, ""))
    monkeypatch.setattr(bot, "_check_entry_liquidity", lambda legs: (True, ""))
    monkeypatch.setattr(bot, "_gpt_entry_gate", lambda **kwargs: True)
    monkeypatch.setattr(bot, "_resolve_directional_trade_style", lambda **kwargs: "long")
    monkeypatch.setattr(bot, "_get_directional_strike_offset", lambda atr_val: 0.0)
    monkeypatch.setattr(bot, "_pick_nearest_strike", lambda chain, option_type, target_strike: chain[0] if option_type == "CE" else chain[1])
    monkeypatch.setattr(bot, "_can_open_trade_type", lambda **kwargs: True)

    bot._decide_entries()

    assert len(bot.state.open_directional) == 1
    assert bot.state.open_directional[0]["name"] == "long_put"
    assert bot.state.open_directional[0]["side"] == "BUY"


def test_gpt_require_recommendation_fallback(monkeypatch) -> None:
    candles = [
        Candle(
            time=datetime(2026, 5, 20, 9, 15 + i),
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.0 + i,
            volume=1000.0,
        )
        for i in range(30)
    ]
    chain = [
        {"symbol": "NIFTY100CE", "strike": 100, "option_type": "CE", "ltp": 10.0, "exchange": "NFO", "token": "1"},
        {"symbol": "NIFTY100PE", "strike": 100, "option_type": "PE", "ltp": 10.0, "exchange": "NFO", "token": "2"},
    ]
    # 1. Require recommendation is True -> should skip entry because GPT returned None
    monkeypatch.setenv("MSTOCK_GPT_API_KEY", "dummy")
    cfg = StrategyConfig(
        strategy_name="auto",
        gpt_enable=True,
        gpt_auto_select=True,
        gpt_require_recommendation=True,
        gpt_apply_paper=True,
        enable_live_trading=False,
        symbol="NIFTY",
        underlying="NIFTY",
        timeframe="1m",
        max_trades_per_day=10,
        max_open_positions=6,
        cooldown_sec=0.0,
        max_stale_ltp_sec=0.0,
        entry_candle_max_age_sec=0.0,
        enable_opening_filter=False,
        enable_vwap_filter=False,
        enable_adx_filter=False,
        enable_volume_filter=False,
        enable_supertrend_filter=False,
        enable_roc_filter=False,
        enable_choppiness_filter=False,
        enable_mtf_confirmation=False,
        enable_rsi_confluence=False,
        enable_premium_rsi_filter=False,
        dir_min_confirmations=1,
        dir_min_score_diff=0,
        dir_allow_tie_break_entries=True,
        auto_strategy_lock_minutes=0,
        lot_size=50,
    )
    bot = _make_bot(cfg, _FakeClient(candles=candles, chain=chain))

    monkeypatch.setattr(strategy_mod, "ema", lambda closes, period: 101.0 if int(period) == 9 else 100.0)
    monkeypatch.setattr(strategy_mod, "rsi", lambda closes, period: 60.0)
    monkeypatch.setattr(strategy_mod, "atr", lambda highs, lows, closes, period: 10.0)
    monkeypatch.setattr(strategy_mod, "adx", lambda highs, lows, closes, period: 30.0)
    monkeypatch.setattr(strategy_mod, "roc", lambda closes, period: 1.0)
    monkeypatch.setattr(strategy_mod, "choppiness_index", lambda highs, lows, closes, period: 45.0)
    monkeypatch.setattr(strategy_mod, "supertrend", lambda highs, lows, closes, period, multiplier: 95.0)
    monkeypatch.setattr(strategy_mod, "is_bullish_engulfing", lambda candles: False)
    monkeypatch.setattr(strategy_mod, "is_bearish_engulfing", lambda candles: False)
    monkeypatch.setattr(strategy_mod, "is_doji", lambda candles: False)
    monkeypatch.setattr(strategy_mod, "is_hammer", lambda candles: False)
    monkeypatch.setattr(strategy_mod, "is_shooting_star", lambda candles: False)
    monkeypatch.setattr(bot, "_now_ist_time", lambda: datetime(2026, 5, 20, 10, 0).time())
    monkeypatch.setattr(bot, "_session_overrides", lambda now_t: (0.0, 999999.0, 30.0, 70.0, "mid"))
    monkeypatch.setattr(bot, "_risk_status", lambda: (True, ""))
    monkeypatch.setattr(bot, "_strategy_router", lambda **kwargs: {"candidates": ["directional"], "selected": "directional"})
    # GPT returns no recommendation
    monkeypatch.setattr(bot, "_gpt_auto_select_strategy", lambda **kwargs: None)
    bot._last_gpt_bias = None
    monkeypatch.setattr(bot, "_clear_auto_gpt_strike_context", lambda: None)
    monkeypatch.setattr(bot, "_set_auto_gpt_strike_context", lambda **kwargs: None)
    monkeypatch.setattr(bot, "_filter_weekly_only", lambda chain: list(chain))
    monkeypatch.setattr(bot, "_select_atm_option", lambda chain, spot, kind: chain[0] if kind == "CE" else chain[1])
    monkeypatch.setattr(bot, "_try_get_ltp", lambda symbol, exchange=None: 100.0 if symbol == "NIFTY" else 10.0)
    monkeypatch.setattr(bot, "_entry_price_bounds_ok", lambda price: (True, ""))
    monkeypatch.setattr(bot, "_check_entry_liquidity", lambda legs: (True, ""))
    monkeypatch.setattr(bot, "_gpt_entry_gate", lambda **kwargs: True)
    monkeypatch.setattr(bot, "_resolve_directional_trade_style", lambda **kwargs: "long")
    monkeypatch.setattr(bot, "_get_directional_strike_offset", lambda atr_val: 0.0)
    monkeypatch.setattr(bot, "_pick_nearest_strike", lambda chain, option_type, target_strike: chain[0] if option_type == "CE" else chain[1])
    monkeypatch.setattr(bot, "_can_open_trade_type", lambda **kwargs: True)

    bot._decide_entries()
    # Should not open any trades
    assert len(bot.state.open_directional) == 0

    # 2. Require recommendation is False -> should fall back to router heuristic (directional)
    bot.cfg.gpt_require_recommendation = False
    bot._decide_entries()
    assert len(bot.state.open_directional) == 1
    assert bot.state.open_directional[0]["name"] == "long_call"
