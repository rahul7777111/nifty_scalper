"""
tests/test_synthetic_option_chain.py

Unit tests for Black-Scholes synthetic option chain (Paper Forward only).
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from synthetic_option_chain import (
    CHAIN_SOURCE,
    PAPER_FORWARD_REQUIRED_COLUMNS,
    atm_strike,
    bs_greeks,
    bs_price,
    build_spot_candle_fallback,
    estimate_iv_from_candles,
    generate_synthetic_option_chain,
    infer_paper_forward_data_quality,
    is_synthetic_chain_enabled,
    map_paper_forward_reason,
    verify_paper_forward_columns,
)


def test_bs_call_price_positive():
    px = bs_price(24500.0, 24500.0, 7 / 365.0, 0.06, 0.0, 0.18, "CE")
    assert px > 0


def test_bs_put_price_positive():
    px = bs_price(24500.0, 24500.0, 7 / 365.0, 0.06, 0.0, 0.18, "PE")
    assert px > 0


def test_ce_delta_positive_pe_delta_negative():
    ce = bs_greeks(24500.0, 24500.0, 7 / 365.0, 0.06, 0.0, 0.18, "CE")
    pe = bs_greeks(24500.0, 24500.0, 7 / 365.0, 0.06, 0.0, 0.18, "PE")
    assert ce["delta"] > 0
    assert pe["delta"] < 0


def test_generated_chain_has_ce_and_pe_rows():
    rows, meta = generate_synthetic_option_chain(
        spot=24500.0,
        expiry="16-06-2026",
        timestamp=datetime(2026, 6, 15, 10, 0, tzinfo=timezone.utc),
        config={"strikes_each_side": 3, "strike_step": 50},
    )
    assert meta["ce_rows"] > 0
    assert meta["pe_rows"] > 0
    assert meta["ce_rows"] == meta["pe_rows"]


def test_atm_strike_included():
    spot = 24523.0
    rows, meta = generate_synthetic_option_chain(
        spot=spot,
        expiry=date(2026, 6, 16),
        config={"strikes_each_side": 2, "strike_step": 50},
    )
    atm = atm_strike(spot, 50)
    strikes = {r["strike_price"] for r in rows}
    assert atm in strikes


def test_bid_mid_ask_ordering():
    rows, _ = generate_synthetic_option_chain(spot=24500.0, expiry="16-06-2026", config={"strikes_each_side": 2})
    for row in rows:
        assert row["bid"] <= row["mid"] <= row["ask"]
        assert row["bid"] >= 0.05


def test_required_paper_forward_columns_exist():
    rows, _ = generate_synthetic_option_chain(spot=24500.0, expiry="16-06-2026", config={"strikes_each_side": 1})
    ok, missing = verify_paper_forward_columns(rows)
    assert ok, f"missing={missing}"
    for col in PAPER_FORWARD_REQUIRED_COLUMNS:
        assert col in rows[0]


def test_iv_from_candles_clamped(monkeypatch):
    candles = [{"close": 100 + i * 0.5} for i in range(80)]
    iv, source, _ = estimate_iv_from_candles(candles, default_iv=0.18)
    assert source == "candles"
    assert 0.08 <= iv <= 0.45


def test_iv_default_when_no_candles():
    iv, source, reason = estimate_iv_from_candles([], default_iv=0.18)
    assert source == "default"
    assert reason == "no_candles"
    assert iv == 0.18


def test_infer_synthetic_chain_ok_with_candles():
    q = infer_paper_forward_data_quality(
        option_rows=42,
        candle_count=100,
        spot=24500.0,
        chain_source=CHAIN_SOURCE,
        synthetic=True,
    )
    assert q == "SYNTHETIC_CHAIN_OK"


def test_infer_waiting_for_candles_without_fallback(monkeypatch):
    monkeypatch.setenv("PF_ALLOW_SYNTHETIC_CANDLE_FALLBACK", "false")
    q = infer_paper_forward_data_quality(
        option_rows=42,
        candle_count=0,
        spot=24500.0,
        chain_source=CHAIN_SOURCE,
        synthetic=True,
    )
    assert q == "WAITING_FOR_CANDLES"


def test_infer_spot_candle_fallback_by_default(monkeypatch):
    monkeypatch.delenv("PF_ALLOW_SYNTHETIC_CANDLE_FALLBACK", raising=False)
    q = infer_paper_forward_data_quality(
        option_rows=42,
        candle_count=0,
        spot=24500.0,
        chain_source=CHAIN_SOURCE,
        synthetic=True,
    )
    assert q == "SYNTHETIC_CHAIN_WITH_SPOT_CANDLE_FALLBACK"


def test_mstock_empty_response_falls_back_when_enabled(monkeypatch):
    monkeypatch.setenv("PF_USE_BLACK_SCHOLES_CHAIN", "true")
    assert is_synthetic_chain_enabled("mstock") is True


def test_synthetic_chain_source_constant():
    rows, meta = generate_synthetic_option_chain(spot=24500.0, expiry="16-06-2026", config={"strikes_each_side": 1})
    assert meta["chain_source"] == CHAIN_SOURCE
    assert rows[0]["chain_source"] == CHAIN_SOURCE
    assert rows[0]["synthetic"] is True
    assert rows[0]["quality"] == "SYNTHETIC_CHAIN"


def test_synthetic_chain_writes_canonical_pf_cache_contract():
    import ui as ui_mod
    txt = Path(ui_mod.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "_pf_try_synthetic_option_chain" in txt
    assert "BS_CHAIN_SOURCE" in txt or "BLACK_SCHOLES_SYNTHETIC" in txt
    assert "PF-CHAIN-CACHE-WRITE" in txt


def test_synthetic_chain_blocks_live_orders():
    import mstock_client as mc
    txt = Path(mc.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "LIVE-ORDER-GUARD" in txt
    assert "synthetic_data" in txt


def test_gui_shows_bs_synth_label():
    import ui as ui_mod
    txt = Path(ui_mod.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "BS SYNTH" in txt


def test_mstock_place_order_guard_runtime(monkeypatch):
    from mstock_client import MStockTypeBClient

    client = MStockTypeBClient.__new__(MStockTypeBClient)
    client._paper_forward_chain_source = CHAIN_SOURCE
    with pytest.raises(RuntimeError, match="LIVE-ORDER-GUARD"):
        client.place_order("NIFTY", "BUY", 1)


def test_spot_resolver_uses_cached_spot(tmp_path, monkeypatch):
    from paper_forward_spot import resolve_live_spot_for_paper_forward, write_spot_cache

    monkeypatch.setenv("PF_FALLBACK_SPOT", "")
    write_spot_cache(23912.5, "test_cache")
    res = resolve_live_spot_for_paper_forward(app_state={}, client=None, broker="mstock")
    assert res.ok or res.spot is not None or "spot_cache_file" in res.tried


def test_spot_resolver_uses_mstock_ltp_token(monkeypatch):
    from paper_forward_spot import resolve_live_spot_for_paper_forward

    class _Client:
        def get_ltp(self, key):
            assert "26000" in str(key) or key == "NSE:26000"
            return 23950.0

    monkeypatch.setenv("MSTOCK_NIFTY_INDEX_TOKEN", "26000")
    monkeypatch.setenv("PF_FALLBACK_SPOT", "")
    res = resolve_live_spot_for_paper_forward(app_state={}, client=_Client(), broker="mstock")
    assert res.ok
    assert res.spot == 23950.0
    assert res.source == "mstock_ltp"


def test_spot_resolver_prefer_live_quote_over_current(monkeypatch):
    from paper_forward_spot import resolve_live_spot_for_paper_forward

    class _Client:
        def __init__(self):
            self.calls = 0

        def get_ltp(self, key):
            self.calls += 1
            return 23950.0 + self.calls

    client = _Client()
    monkeypatch.setenv("MSTOCK_NIFTY_INDEX_TOKEN", "26000")
    monkeypatch.setenv("PF_FALLBACK_SPOT", "")
    first = resolve_live_spot_for_paper_forward(
        app_state={"_spot_ltp_live": 111.0},
        client=client,
        broker="mstock",
        current=24000.0,
        prefer_live_quote=True,
    )
    second = resolve_live_spot_for_paper_forward(
        app_state={"_spot_ltp_live": first.spot},
        client=client,
        broker="mstock",
        current=first.spot,
        prefer_live_quote=True,
    )

    assert first.ok
    assert second.ok
    assert first.source == "mstock_ltp"
    assert second.source == "mstock_ltp"
    assert first.spot == 23951.0
    assert second.spot == 23952.0
    assert client.calls == 2


def test_spot_resolver_no_option_chain_required():
    from paper_forward_spot import resolve_live_spot_for_paper_forward

    res = resolve_live_spot_for_paper_forward(
        app_state={"_spot_ltp_live": 24000.0},
        client=None,
        broker="mstock",
        chain=[],
        allow_option_chain_spot=False,
    )
    assert res.ok
    assert res.spot == 24000.0


def test_bs_chain_after_spot_even_if_real_chain_empty(monkeypatch):
    monkeypatch.setenv("PF_USE_BLACK_SCHOLES_CHAIN", "true")
    rows, meta = generate_synthetic_option_chain(
        spot=23900.0,
        expiry="16-06-2026",
        config={"strikes_each_side": 2},
    )
    assert len(rows) > 0
    assert meta["chain_source"] == CHAIN_SOURCE


def test_candles_zero_fallback_false_quality(monkeypatch):
    monkeypatch.setenv("PF_ALLOW_SYNTHETIC_CANDLE_FALLBACK", "false")
    q = infer_paper_forward_data_quality(
        option_rows=82,
        candle_count=0,
        spot=23900.0,
        chain_source=CHAIN_SOURCE,
        synthetic=True,
    )
    assert q == "WAITING_FOR_CANDLES"


def test_candles_zero_fallback_true_allows_prediction(monkeypatch):
    monkeypatch.setenv("PF_ALLOW_SYNTHETIC_CANDLE_FALLBACK", "true")
    q = infer_paper_forward_data_quality(
        option_rows=82,
        candle_count=0,
        spot=23900.0,
        chain_source=CHAIN_SOURCE,
        synthetic=True,
    )
    assert q == "SYNTHETIC_CHAIN_WITH_SPOT_CANDLE_FALLBACK"


def test_mstock_empty_chain_does_not_require_broker_chain():
    import ui as ui_mod
    txt = Path(ui_mod.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "_pf_ensure_mstock_synthetic_chain_if_needed" in txt
    assert "PF-MSTOCK-SYNTH-FALLBACK" in txt


def test_engine_mstock_skips_waiting_for_option_chain():
    import paper_forward_engine as pfe
    txt = Path(pfe.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "synthetic_chain_ready" in txt
    assert "SYNTHETIC_CHAIN_PENDING" in txt


def test_ui_has_central_spot_resolver():
    import ui as ui_mod
    txt = Path(ui_mod.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "resolve_live_spot_for_paper_forward" in txt
    assert "PF-SPOT-RESOLVE" in Path(REPO_ROOT / "src" / "paper_forward_spot.py").read_text(encoding="utf-8")


def test_synthetic_candle_fallback_generates_100_rows():
    rows = build_spot_candle_fallback(23910.70)
    assert len(rows) == 100
    sample = rows[0]
    for key in ("timestamp", "open", "high", "low", "close", "volume"):
        assert key in sample
    assert rows[0]["synthetic_candles"] is True
    rows2 = build_spot_candle_fallback(23910.70)
    assert rows[0]["close"] == rows2[0]["close"]


def test_real_candles_preferred_over_synthetic(monkeypatch):
    from paper_forward_spot import resolve_paper_forward_candles

    real = [{"close": 23900 + i * 0.1, "open": 23900, "high": 23901, "low": 23899, "volume": 100} for i in range(25)]
    monkeypatch.setenv("PF_ALLOW_SYNTHETIC_CANDLE_FALLBACK", "true")
    res = resolve_paper_forward_candles(app_state={"_latest_candles": real}, spot=23910.0)
    assert res.ok
    assert len(res.candles) >= 20
    assert res.source == "live_chart_cache"
    assert res.synthetic_candles is False


def test_pf_ui_candle_fetch_requires_real_candles_by_default(monkeypatch):
    from ui import ScalperUI

    monkeypatch.delenv("PF_ALLOW_SYNTHETIC_CANDLE_FALLBACK", raising=False)
    ui = object.__new__(ScalperUI)

    candles, source, error = ui._pf_fetch_candles_readonly(None, spot=23910.0)

    assert candles == []
    assert source == "none"
    assert error in {"no_candles", "waiting_for_real_candles"}
    assert ui._pf_synthetic_candles_active is False


def test_engine_real_candle_snapshot_blocks_synthetic_injection(monkeypatch):
    from paper_forward_engine import _apply_spot_candle_fallback_if_needed

    monkeypatch.setenv("PF_ALLOW_SYNTHETIC_CANDLE_FALLBACK", "true")
    snap = {
        "spot": 23910.0,
        "chain_source": CHAIN_SOURCE,
        "synthetic": True,
        "option_rows": 82,
        "candles": [],
        "candle_count": 0,
        "require_real_candles": True,
        "allow_synthetic_candle_fallback": False,
    }

    _apply_spot_candle_fallback_if_needed(snap)

    assert snap["candles"] == []
    assert snap["candle_count"] == 0
    assert snap["synthetic_candles"] is False
    assert snap["data_quality_status"] == "WAITING_FOR_CANDLES"


def test_insufficient_candles_blocks_when_fallback_disabled(monkeypatch):
    monkeypatch.setenv("PF_ALLOW_SYNTHETIC_CANDLE_FALLBACK", "false")
    monkeypatch.setenv("PF_MIN_CANDLES_FOR_PREDICTION", "20")
    q = infer_paper_forward_data_quality(
        option_rows=82,
        candle_count=1,
        spot=23900.0,
        chain_source=CHAIN_SOURCE,
        synthetic=True,
    )
    assert q == "WAITING_FOR_CANDLES"


def test_synthetic_chain_with_synthetic_candles_allows_prediction(monkeypatch):
    monkeypatch.setenv("PF_ALLOW_SYNTHETIC_CANDLE_FALLBACK", "true")
    rows = build_spot_candle_fallback(23900.0)
    q = infer_paper_forward_data_quality(
        option_rows=82,
        candle_count=len(rows),
        spot=23900.0,
        chain_source=CHAIN_SOURCE,
        synthetic=True,
        candle_source="SYNTHETIC_SPOT_FALLBACK",
        synthetic_candles=True,
    )
    assert q == "SYNTHETIC_CHAIN_WITH_SYNTHETIC_CANDLES"


def test_reason_mapper_converts_raw_reasons():
    assert map_paper_forward_reason("ok")[0] == "TRADE_OPENED_PAPER"
    assert map_paper_forward_reason("low_confidence_0.0000_lt_0.3000")[0] == "low_confidence_0.0000_lt_0.3000"
    assert map_paper_forward_reason("auto_directional_mixed_regime_no_trade")[0] == "MIXED_REGIME_NO_TRADE"
    assert map_paper_forward_reason("synthetic_chain_ready")[0] == "SYNTHETIC_CHAIN_READY"
    assert map_paper_forward_reason("waiting_for_candles")[0] == "WAITING_FOR_CANDLES"
    assert map_paper_forward_reason("ARTIFACT_NOT_FOUND")[0] == "ARTIFACT_NOT_FOUND"


def _mock_paper_engine():
    from paper_forward_engine import PaperForwardEngine

    eng = PaperForwardEngine.__new__(PaperForwardEngine)
    eng._state = {}
    eng._candidate_states = {}
    eng._duplicate_entry_blocked = 0
    eng._entry_cooldown_blocked = 0
    eng._signal_dedupe_blocked = 0
    eng._ensure_candidate_state_containers = PaperForwardEngine._ensure_candidate_state_containers.__get__(eng, PaperForwardEngine)
    eng._get_or_create_candidate_state = PaperForwardEngine._get_or_create_candidate_state.__get__(eng, PaperForwardEngine)
    eng.update_candidate_state = PaperForwardEngine.update_candidate_state.__get__(eng, PaperForwardEngine)
    return eng


def _synth_snap(spot: float = 23900.0):
    rows, _ = generate_synthetic_option_chain(spot=spot, expiry="16-06-2026", config={"strikes_each_side": 2})
    return {
        "spot": spot,
        "price": spot,
        "chain_source": CHAIN_SOURCE,
        "synthetic": True,
        "candle_source": "SYNTHETIC_SPOT_FALLBACK",
        "synthetic_candles": True,
        "option_chain": rows,
    }


def test_synthetic_paper_entry_increments_entry_count():
    eng = _mock_paper_engine()

    snap = _synth_snap()
    upd = eng.update_candidate_state(
        "test_cand",
        {"final_signal": "BUY_CE", "confidence": 0.55, "threshold": 0.3},
        snap,
    )
    st = eng._state["test_cand"]
    assert st["total_entries"] == 1
    assert upd.get("total_entries") == 1
    assert upd.get("simulated_action") == "ENTER"


def test_synthetic_pnl_mark_updates_unrealized_pnl():
    from paper_forward_engine import PaperForwardEngine, _synthetic_option_mark

    rows, _ = generate_synthetic_option_chain(spot=23950.0, expiry="16-06-2026", config={"strikes_each_side": 2})
    snap = {
        "spot": 23950.0,
        "price": 23950.0,
        "chain_source": CHAIN_SOURCE,
        "synthetic": True,
        "option_chain": rows,
    }
    atm = atm_strike(23950.0, 50)
    mark_px, _, pnl_src = _synthetic_option_mark(snap, "CE", strike=atm)
    assert mark_px is not None
    assert pnl_src == "SYNTHETIC_MARK"

    eng = _mock_paper_engine()
    eng._state = {
        "test_cand": {
            "open_position": True,
            "side": "CE",
            "entry_price": mark_px,
            "entry_strike": atm,
            "entry_symbol": "TESTCE",
            "total_entries": 1,
            "total_exits": 0,
            "total_trades": 0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "wins": 0,
            "losses": 0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "qty": 1,
        }
    }

    upd = eng.update_candidate_state(
        "test_cand",
        {"final_signal": "BUY_CE", "confidence": 0.55},
        snap,
    )
    assert upd.get("simulated_action") == "HOLD_EXISTING_PAPER_POSITION"
    assert upd.get("no_trade_reason") == "HOLD_EXISTING_POSITION"
    assert eng._state["test_cand"]["pnl_source"] == "SYNTHETIC_MARK"
    assert eng._state["test_cand"]["current_price"] < 500
    assert eng._state["test_cand"]["current_price"] != snap["spot"]


def test_open_candidate_blocks_duplicate_entry():
    eng = _mock_paper_engine()
    snap = _synth_snap()
    eng.update_candidate_state("c1", {"final_signal": "BUY_CE", "confidence": 0.9, "threshold": 0.3}, snap)
    upd2 = eng.update_candidate_state("c1", {"final_signal": "BUY_CE", "confidence": 0.9, "threshold": 0.3}, snap)
    assert eng._state["c1"]["total_entries"] == 1
    assert upd2.get("simulated_action") == "HOLD_EXISTING_PAPER_POSITION"
    assert upd2.get("no_trade_reason") == "HOLD_EXISTING_POSITION"
    assert eng._duplicate_entry_blocked >= 1


def test_synthetic_cooldown_blocks_repeated_entries(monkeypatch):
    monkeypatch.setenv("PF_SYNTHETIC_MIN_ENTRY_GAP_SECONDS", "300")
    eng = _mock_paper_engine()
    snap = _synth_snap()
    eng.update_candidate_state("c1", {"final_signal": "BUY_CE", "confidence": 0.9, "threshold": 0.3}, snap)
    eng._state["c1"]["open_position"] = False
    eng._state["c1"]["last_exit_ts"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    eng._state["c1"]["last_eval_signal"] = "NO_TRADE"
    upd = eng.update_candidate_state("c1", {"final_signal": "BUY_CE", "confidence": 0.95, "threshold": 0.3}, snap)
    assert upd.get("no_trade_reason") == "ENTRY_COOLDOWN"
    assert eng._state["c1"]["total_entries"] == 1


def test_repeated_same_signal_does_not_open_new_trade():
    eng = _mock_paper_engine()
    snap = _synth_snap()
    eng.update_candidate_state("c1", {"final_signal": "BUY_CE", "confidence": 0.9, "threshold": 0.3}, snap)
    eng._state["c1"]["open_position"] = False
    eng._state["c1"]["last_exit_ts"] = "2000-01-01T00:00:00+00:00"
    eng._state["c1"]["last_entry_ts"] = "2000-01-01T00:00:00+00:00"
    eng._state["c1"]["last_eval_signal"] = "BUY_CE"
    eng._state["c1"]["last_confidence"] = 0.9
    upd = eng.update_candidate_state("c1", {"final_signal": "BUY_CE", "confidence": 0.91, "threshold": 0.3}, snap)
    assert upd.get("no_trade_reason") == "SIGNAL_DEDUPE"
    assert eng._state["c1"]["total_entries"] == 1


def test_opposite_signal_exits_when_enabled(monkeypatch):
    monkeypatch.setenv("PF_SYNTHETIC_MIN_EXIT_GAP_SECONDS", "0")
    monkeypatch.setenv("PF_PAPER_MAX_HOLD_SECONDS", "99999")
    monkeypatch.setenv("PF_EXIT_ON_OPPOSITE_SIGNAL", "true")
    from datetime import datetime, timezone

    eng = _mock_paper_engine()
    snap = _synth_snap()
    eng.update_candidate_state("c1", {"final_signal": "BUY_CE", "confidence": 0.9, "threshold": 0.3}, snap)
    eng._state["c1"]["entry_time"] = datetime.now(timezone.utc).isoformat()
    upd = eng.update_candidate_state("c1", {"final_signal": "BUY_PE", "confidence": 0.9, "threshold": 0.3}, snap)
    assert upd.get("simulated_action") == "EXIT"
    assert "OPPOSITE" in str(upd.get("no_trade_reason", ""))


def test_target_exit_works(monkeypatch):
    monkeypatch.setenv("PF_PAPER_TARGET_PCT", "0.0001")
    monkeypatch.setenv("PF_SYNTHETIC_MIN_EXIT_GAP_SECONDS", "0")
    from paper_forward_engine import _evaluate_paper_exit
    from synthetic_option_chain import load_paper_sim_config

    cfg = load_paper_sim_config()
    ok, reason = _evaluate_paper_exit(
        side="CE",
        final_sig="BUY_CE",
        entry_p=100.0,
        mark_px=100.05,
        entry_epoch=0,
        now_epoch=1000,
        cfg=cfg,
        is_synth=True,
    )
    assert ok is True
    assert reason == "PAPER_EXIT_TARGET"


def test_stoploss_exit_works(monkeypatch):
    monkeypatch.setenv("PF_PAPER_STOPLOSS_PCT", "0.05")
    monkeypatch.setenv("PF_SYNTHETIC_MIN_EXIT_GAP_SECONDS", "0")
    from paper_forward_engine import _evaluate_paper_exit
    from synthetic_option_chain import load_paper_sim_config

    cfg = load_paper_sim_config()
    ok, reason = _evaluate_paper_exit(
        side="CE",
        final_sig="BUY_CE",
        entry_p=100.0,
        mark_px=90.0,
        entry_epoch=0,
        now_epoch=1000,
        cfg=cfg,
        is_synth=True,
    )
    assert ok is True
    assert reason == "PAPER_EXIT_STOP"


def test_current_price_uses_option_mid_not_spot():
    eng = _mock_paper_engine()
    snap = _synth_snap(23900.0)
    upd = eng.update_candidate_state("c1", {"final_signal": "BUY_CE", "confidence": 0.9, "threshold": 0.3}, snap)
    assert upd.get("option_current_price", upd.get("current_price", 0)) < 1000
    assert upd.get("spot_price", snap["spot"]) > 20000


def test_synthetic_mode_export_marks_synthetic_data():
    from paper_forward_engine import PaperForwardEngine
    from pathlib import Path
    import tempfile

    eng = PaperForwardEngine.__new__(PaperForwardEngine)
    eng.candidates = []
    eng._state = {"c1": {"synthetic_mode": True, "total_entries": 1}}
    eng._last_data_status = type("DS", (), {"as_dict": lambda self: {"option_chain_source": "BLACK_SCHOLES_SYNTHETIC", "candle_source": "SYNTHETIC_SPOT_FALLBACK"}})()
    eng.get_status_table = lambda: []
    with tempfile.TemporaryDirectory() as td:
        eng.reports_dir = Path(td)
        eng.log_dir = Path(td)
        summary = PaperForwardEngine.write_summary(eng)
    assert summary["chain_source"] == "BLACK_SCHOLES_SYNTHETIC"
    assert summary["candle_source"] == "SYNTHETIC_SPOT_FALLBACK"
    assert summary["not_real_broker_market_data"] is True


def test_live_order_guard_blocks_synthetic_data_orders():
    from mstock_client import MStockTypeBClient

    client = MStockTypeBClient.__new__(MStockTypeBClient)
    client._paper_forward_chain_source = CHAIN_SOURCE
    with pytest.raises(RuntimeError, match="LIVE-ORDER-GUARD"):
        client.place_order("NIFTY", "BUY", 1)


def test_diagnostics_report_ready_synthetic_chain_with_synthetic_candles(monkeypatch):
    monkeypatch.setenv("PF_ALLOW_SYNTHETIC_CANDLE_FALLBACK", "true")
    rows = build_spot_candle_fallback(23900.0)
    q = infer_paper_forward_data_quality(
        option_rows=82,
        candle_count=len(rows),
        spot=23900.0,
        chain_source=CHAIN_SOURCE,
        synthetic=True,
        candle_source="SYNTHETIC_SPOT_FALLBACK",
        synthetic_candles=True,
    )
    assert q == "SYNTHETIC_CHAIN_WITH_SYNTHETIC_CANDLES"


def test_broker_option_mark_prefers_live_chain_ltp():
    from paper_forward_engine import _broker_option_mark

    snap = {
        "spot": 23900.0,
        "broker_live_option_chain": [
            {
                "strike_price": 23900.0,
                "option_type": "CE",
                "ltp": 187.5,
                "trading_symbol": "NIFTY16JUN2623900CE",
                "exchange": "NFO",
                "token": "12345",
            }
        ],
    }
    px, sym, src, strike = _broker_option_mark(snap, "CE", strike=23900.0, spot=23900.0)
    assert px == 187.5
    assert sym == "NIFTY16JUN2623900CE"
    assert src == "BROKER_CHAIN_LTP"
    assert strike == 23900.0


def test_broker_option_mark_prefers_selected_symbol_live_quote_over_stale_chain_ltp():
    from paper_forward_engine import _broker_option_mark

    class _FakeClient:
        def __init__(self):
            self.calls = []

        def fetch_option_quote_for_paper(self, exchange, token, symbol):
            self.calls.append((exchange, token, symbol))
            return {"ltp": 193.75}

    client = _FakeClient()
    snap = {
        "spot": 23900.0,
        "broker_live_option_chain": [
            {
                "strike_price": 23900.0,
                "option_type": "CE",
                "ltp": 187.5,
                "trading_symbol": "NIFTY16JUN2623900CE",
                "exchange": "NFO",
                "token": "12345",
            }
        ],
        "broker_client": client,
    }

    px, sym, src, strike = _broker_option_mark(
        snap,
        "CE",
        strike=23900.0,
        spot=23900.0,
        entry_symbol="NIFTY16JUN2623900CE",
    )

    assert px == 193.75
    assert sym == "NIFTY16JUN2623900CE"
    assert src == "BROKER_LIVE_LTP"
    assert strike == 23900.0
    assert client.calls == [("NFO", "12345", "NIFTY16JUN2623900CE")]


def test_broker_option_mark_matches_entry_symbol_before_nearest_strike():
    from paper_forward_engine import _broker_option_mark

    snap = {
        "spot": 23900.0,
        "option_chain": [
            {
                "strike_price": 23900.0,
                "option_type": "CE",
                "ltp": 100.0,
                "tradingsymbol": "STALE_OLD_SYMBOL",
            },
            {
                "strike_price": 23900.0,
                "option_type": "CE",
                "ltp": 126.5,
                "tradingsymbol": "NIFTY16JUN2623900CE",
            },
        ],
    }

    px, sym, src, strike = _broker_option_mark(
        snap,
        "CE",
        strike=23900.0,
        spot=23900.0,
        entry_symbol="NIFTY16JUN2623900CE",
    )

    assert px == 126.5
    assert sym == "NIFTY16JUN2623900CE"
    assert src == "BROKER_CHAIN_LTP"
    assert strike == 23900.0


def test_broker_option_mark_fetches_live_ltp_for_synthetic_row():
    from paper_forward_engine import _broker_option_mark

    class _FakeClient:
        def fetch_option_quote_for_paper(self, exchange, token, symbol):
            return {"ltp": 201.25, "bid_price": 200.0, "ask_price": 202.5}

    snap = {
        "spot": 23900.0,
        "option_chain": [
            {
                "strike_price": 23900.0,
                "option_type": "CE",
                "mid": 50.0,
                "synthetic": True,
                "chain_source": CHAIN_SOURCE,
                "trading_symbol": "NIFTY16JUN2623900CE",
                "exchange": "NFO",
                "token": "99999",
            }
        ],
        "broker_client": _FakeClient(),
    }
    px, sym, src, _ = _broker_option_mark(snap, "CE", strike=23900.0, spot=23900.0)
    assert px == 201.25
    assert src == "BROKER_LIVE_LTP"
    assert "NIFTY" in sym


def test_broker_option_mark_uses_security_id_alias_for_live_quote():
    from paper_forward_engine import _broker_option_mark

    class _FakeClient:
        def __init__(self):
            self.calls = []

        def fetch_option_quote_for_paper(self, exchange, token, symbol):
            self.calls.append((exchange, token, symbol))
            return {"bid_price": 120.0, "ask_price": 122.0}

    client = _FakeClient()
    snap = {
        "spot": 23900.0,
        "option_chain": [
            {
                "strike_price": 23900.0,
                "option_type": "PE",
                "ltp": 99.0,
                "tradingsymbol": "NIFTY16JUN2623900PE",
                "security_id": "55555",
                "exchange": "NFO",
            }
        ],
        "broker_client": client,
    }

    px, sym, src, _ = _broker_option_mark(
        snap,
        "PE",
        strike=23900.0,
        spot=23900.0,
        entry_symbol="NIFTY16JUN2623900PE",
    )

    assert px == 121.0
    assert src == "BROKER_LIVE_LTP"
    assert sym == "NIFTY16JUN2623900PE"
    assert client.calls == [("NFO", "55555", "NIFTY16JUN2623900PE")]


def test_broker_option_mark_tries_token_ltp_when_full_quote_empty():
    from paper_forward_engine import _broker_option_mark

    class _FakeClient:
        def __init__(self):
            self.ltp_keys = []

        def fetch_option_quote_for_paper(self, exchange, token, symbol):
            return {}

        def get_ltp(self, key):
            self.ltp_keys.append(key)
            if key == "NFO:77777":
                return 144.5
            raise RuntimeError("wrong key")

    client = _FakeClient()
    snap = {
        "spot": 23900.0,
        "option_chain": [
            {
                "strike_price": 23900.0,
                "option_type": "CE",
                "ltp": 100.0,
                "tradingsymbol": "NIFTY16JUN2623900CE",
                "instrumentToken": "77777",
                "exchange": "NFO",
            }
        ],
        "broker_client": client,
    }

    px, _sym, src, _ = _broker_option_mark(
        snap,
        "CE",
        strike=23900.0,
        spot=23900.0,
        entry_symbol="NIFTY16JUN2623900CE",
    )

    assert px == 144.5
    assert src == "BROKER_LIVE_LTP"
    assert client.ltp_keys[0] == "NFO:77777"


def test_broker_option_mark_resolves_symbol_token_before_using_stale_chain_ltp():
    from paper_forward_engine import _broker_option_mark

    class _FakeClient:
        def __init__(self):
            self.resolve_calls = []
            self.quote_calls = []
            self.ltp_keys = []

        def resolve_exchange_token(self, symbol, exchange_hint=None):
            self.resolve_calls.append((symbol, exchange_hint))
            return "NFO", "77777"

        def fetch_option_quote_for_paper(self, exchange, token, symbol):
            self.quote_calls.append((exchange, token, symbol))
            return {}

        def get_ltp(self, key):
            self.ltp_keys.append(key)
            if key == "NFO:77777":
                return 144.5
            raise RuntimeError("wrong key")

    client = _FakeClient()
    snap = {
        "spot": 23900.0,
        "option_chain": [
            {
                "strike_price": 23900.0,
                "option_type": "CE",
                "ltp": 100.0,
                "tradingsymbol": "NIFTY16JUN2623900CE",
                "exchange": "NFO",
            }
        ],
        "broker_client": client,
    }

    px, _sym, src, _ = _broker_option_mark(
        snap,
        "CE",
        strike=23900.0,
        spot=23900.0,
        entry_symbol="NIFTY16JUN2623900CE",
    )

    assert px == 144.5
    assert src == "BROKER_LIVE_LTP"
    assert client.resolve_calls == [("NIFTY16JUN2623900CE", "NFO")]
    assert client.quote_calls == [("NFO", "77777", "NIFTY16JUN2623900CE")]
    assert client.ltp_keys[0] == "NFO:77777"


def test_status_table_refresh_updates_open_position_from_live_ltp():
    from paper_forward_engine import PaperForwardEngine

    class _FakeClient:
        def __init__(self):
            self.calls = 0

        def fetch_option_quote_for_paper(self, exchange, token, symbol):
            self.calls += 1
            return {"ltp": 210.0 + self.calls}

    client = _FakeClient()
    eng = PaperForwardEngine.__new__(PaperForwardEngine)
    eng._ensure_candidate_state_containers = PaperForwardEngine._ensure_candidate_state_containers.__get__(eng, PaperForwardEngine)
    eng._get_or_create_candidate_state = PaperForwardEngine._get_or_create_candidate_state.__get__(eng, PaperForwardEngine)
    eng._refresh_open_position_marks_from_latest_snapshot = PaperForwardEngine._refresh_open_position_marks_from_latest_snapshot.__get__(eng, PaperForwardEngine)
    eng.get_status_table = PaperForwardEngine.get_status_table.__get__(eng, PaperForwardEngine)
    eng.candidates = [{"candidate_id": "c1", "enabled": True, "_load_status": "candidate_loaded_ok"}]
    eng._candidate_states = {}
    eng._state = {
        "c1": {
            "open_position": True,
            "side": "CE",
            "entry_price": 200.0,
            "entry_strike": 23900.0,
            "entry_symbol": "NIFTY16JUN2623900CE",
            "qty": 1,
            "total_entries": 1,
        }
    }
    eng._latest_market_snapshot = {
        "spot": 23900.0,
        "broker_client": client,
        "broker_live_option_chain": [
            {
                "strike_price": 23900.0,
                "option_type": "CE",
                "ltp": 187.5,
                "trading_symbol": "NIFTY16JUN2623900CE",
                "exchange": "NFO",
                "token": "12345",
            }
        ],
    }
    eng._latest_option_chain_snapshot = eng._latest_market_snapshot["broker_live_option_chain"]

    first = eng.get_status_table()[0]
    second = eng.get_status_table()[0]

    assert first["option_current_price"] == 211.0
    assert second["option_current_price"] == 212.0
    assert second["pnl_source"] == "BROKER_LIVE_LTP"


def test_pf_decision_cache_does_not_overwrite_fresh_current_option_price():
    import ui as ui_mod

    app = ui_mod.ScalperUI.__new__(ui_mod.ScalperUI)
    app._pf_latest_decisions_by_candidate = {
        "c1": {
            "candidate_id": "c1",
            "last_update": "2026-06-16T10:00:10+00:00",
            "position_status": "OPEN",
            "current_price": 187.5,
            "option_current_price": 187.5,
            "unrealized_pnl": -12.5,
            "selected_strike": 23900.0,
            "selected_option_type": "CE",
            "selected_symbol": "NIFTY16JUN2623900CE",
        }
    }
    rows = [
        {
            "candidate_id": "c1",
            "last_update": "2026-06-16T10:00:00+00:00",
            "position_status": "OPEN",
            "current_price": 212.0,
            "option_current_price": 212.0,
            "unrealized_pnl": 12.0,
            "selected_strike": 23900.0,
            "selected_option_type": "CE",
            "selected_symbol": "NIFTY16JUN2623900CE",
        }
    ]

    merged = ui_mod.ScalperUI._pf_merge_cached_decisions_into_rows(app, rows)

    assert merged[0]["option_current_price"] == 212.0
    assert merged[0]["current_price"] == 212.0
    assert merged[0]["unrealized_pnl"] == 12.0


def test_pf_ui_row_refresh_overrides_equal_entry_current_with_live_ltp():
    import ui as ui_mod

    class _FakeClient:
        def __init__(self):
            self.resolve_calls = []
            self.ltp_keys = []

        def resolve_exchange_token(self, symbol, exchange_hint=None):
            self.resolve_calls.append((symbol, exchange_hint))
            return "NFO", "77777"

        def get_ltp(self, key):
            self.ltp_keys.append(key)
            if key == "NFO:77777":
                return 48.25
            raise RuntimeError("wrong key")

    app = ui_mod.ScalperUI.__new__(ui_mod.ScalperUI)
    app._client = _FakeClient()
    app.pf_engine = None
    app._get_cached_option_chain_rows = lambda: (
        [
            {
                "strike_price": 23900.0,
                "option_type": "CE",
                "ltp": 42.82,
                "tradingsymbol": "NIFTY16JUN2623900CE",
                "exchange": "NFO",
            }
        ],
        "test",
    )

    rows = [
        {
            "candidate_id": "c1",
            "position_status": "OPEN",
            "entry_price": 42.82,
            "current_price": 42.82,
            "option_current_price": 42.82,
            "selected_strike": 23900.0,
            "selected_option_type": "CE",
            "selected_symbol": "NIFTY16JUN2623900CE",
        }
    ]

    refreshed = ui_mod.ScalperUI._pf_apply_live_option_marks_to_rows(app, rows)

    assert refreshed[0]["entry_price"] == 42.82
    assert refreshed[0]["option_current_price"] == 48.25
    assert refreshed[0]["current_price"] == 48.25
    assert refreshed[0]["pnl_source"] == "UI_BROKER_LIVE_LTP"
    assert app._client.resolve_calls == [("NIFTY16JUN2623900CE", "NFO")]
    assert app._client.ltp_keys[0] == "NFO:77777"


def test_paper_mark_config_defaults_broker_prices_on():
    from synthetic_option_chain import load_paper_mark_config

    cfg = load_paper_mark_config()
    assert cfg["use_broker_option_prices"] is True
    assert cfg["broker_price_fallback_synthetic"] is True
