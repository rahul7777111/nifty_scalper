from __future__ import annotations

import threading


def _ui_helper():
    from ui import ScalperUI
    from paper_forward_engine import PaperForwardRuntime

    obj = object.__new__(ScalperUI)
    obj._market_data_lock = threading.RLock()
    obj._pf_runtime = PaperForwardRuntime()
    obj._pf_auth_state = obj._pf_runtime.auth
    obj._pf_auth_state.status = "AUTH_OK"
    obj._pf_auth_state.validation_attempted = True
    obj._latest_candles = []
    obj._last_option_chain_status = "EMPTY"
    obj._last_option_chain_reason = ""
    obj._last_option_chain_error = None
    obj._last_option_chain_source = ""
    obj.after = lambda _delay, fn=None: fn() if fn else None
    obj._option_chain_status_var = type("Var", (), {"set": lambda self, value: None})()
    obj._dash_candles_var = type("Var", (), {"get": lambda self: "0", "set": lambda self, value: None})()
    obj._dash_spot_var = type("Var", (), {"get": lambda self: "n/a", "set": lambda self, value: None})()
    obj._dash_last_tick_var = type("Var", (), {"set": lambda self, value: None})()
    obj.pf_broker_status_var = type("Var", (), {"set": lambda self, value: None})()
    obj.pf_status_var = type("Var", (), {"set": lambda self, value: None})()
    obj.pf_live_var = type("Var", (), {"set": lambda self, value: None})()
    obj._pf_ensure_auth_terminal = lambda force=False: ("AUTH_OK", "", None)
    obj._pf_auth_status_fields = lambda: {"broker_auth": "AUTH_OK", "auth_validation_attempted": True}
    obj._pf_auth_trace = lambda *args, **kwargs: None
    return obj


def test_normalizer_accepts_list_of_dicts():
    ui = _ui_helper()
    rows = ui._normalize_option_chain_rows([{"tradingsymbol": "NIFTY25000CE", "strikePrice": 25000, "optionType": "CALL"}])
    assert len(rows) == 1
    assert rows[0]["symbol"] == "NIFTY25000CE"
    assert rows[0]["strike"] == 25000
    assert rows[0]["option_type"] == "CE"


def test_normalizer_accepts_dataframe():
    pd = __import__("pandas")
    ui = _ui_helper()
    rows = ui._normalize_option_chain_rows(pd.DataFrame([{"tsym": "NIFTY25000PE", "strike_price": 25000, "type": "PE"}]))
    assert len(rows) == 1
    assert rows[0]["symbol"] == "NIFTY25000PE"
    assert rows[0]["option_type"] == "PE"


def test_normalizer_accepts_dict_wrapper():
    ui = _ui_helper()
    rows = ui._normalize_option_chain_rows({"data": [{"symbol": "NIFTY25000CE", "strike": 25000, "option_type": "CE"}]})
    assert len(rows) == 1
    assert rows[0]["strike"] == 25000


def test_cache_helper_writes_ui_and_scalper_attrs():
    ui = _ui_helper()
    scalper = type("Scalper", (), {})()
    ui._scalper = scalper
    rows = ui._set_latest_option_chain_cache([{"symbol": "NIFTY25000CE", "strike": 25000, "option_type": "CE"}], source="test", reason="ok")
    assert len(rows) == 1
    assert len(ui._option_chain_data) == 1
    assert len(ui.option_chain_data) == 1
    assert len(ui.latest_option_chain) == 1
    assert len(ui._latest_option_chain_rows) == 1
    assert len(ui._paper_forward_option_chain_rows) == 1
    assert len(scalper._option_chain_data) == 1
    assert len(scalper.option_chain_data) == 1
    assert len(scalper.latest_option_chain) == 1


def test_snapshot_builder_includes_option_chain_rows():
    ui = _ui_helper()
    ui._set_latest_option_chain_cache([{"symbol": "NIFTY25000CE", "strike": 25000, "option_type": "CE", "spot": 25010}], source="test", reason="ok")
    snap = ui._build_paper_snapshot_for_engine()
    assert snap["option_chain_rows"] == 1
    assert snap["option_chain"]
    assert snap["option_chain_source"] == "test"


def test_empty_option_chain_decision_does_not_increment_eval_count():
    from collections import deque
    from paper_forward_engine import PaperForwardEngine

    eng = object.__new__(PaperForwardEngine)
    eng._decisions = deque(maxlen=10)
    eng._evaluations_count = 0
    eng._predictions_attempted = 0
    eng._predictions_success = 0
    eng._skipped_missing_features = 0
    eng._skipped_market_data = 0
    eng._state = {}
    eng._runtime_states = {}
    eng._reason_counts = {}
    eng._apply_paper_forward_decision_to_state = lambda decision: None
    decision = {
        "candidate_id": "c1",
        "no_trade_reason": "option_chain_empty",
        "predict_attempted": False,
        "_skip_eval_count": True,
    }
    eng._record_decision("c1", decision, "ts", {"price": 25000})
    assert eng._evaluations_count == 0
    assert eng._skipped_market_data == 1


def test_paper_forward_decision_attributes_are_not_reset_after_apply():
    from collections import deque
    from paper_forward_engine import PaperForwardEngine, PaperForwardCandidateRuntimeState

    eng = object.__new__(PaperForwardEngine)
    eng.candidates = [{
        "candidate_id": "c1",
        "enabled": True,
        "model_name": "m",
        "preset_family": "p",
        "side_policy": "BOTH",
        "_load_status": "LOAD_OK",
    }]
    eng._candidate_states = {"c1": PaperForwardCandidateRuntimeState("c1")}
    eng._state = {"c1": {
        "open_position": False,
        "last_no_trade_reason": "",
        "confidence": None,
        "predict_attempted": False,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "total_trades": 0,
    }}
    eng._decisions = deque(maxlen=10)
    eng._evaluations_count = 0
    eng._predictions_attempted = 0
    eng._predictions_success = 0
    eng._skipped_missing_features = 0
    eng._skipped_market_data = 0
    eng._reason_counts = {}
    decision = {
        "candidate_id": "c1",
        "timestamp": "2026-06-12T09:15:00+05:30",
        "final_signal": "NO_TRADE",
        "no_trade_reason": "score_below_threshold",
        "confidence": 0.42,
        "threshold": 0.7,
        "predict_attempted": True,
        "unrealized_pnl": 12.5,
        "realized_pnl": 3.0,
        "total_trades": 2,
    }
    eng._record_decision("c1", decision, decision["timestamp"], {"price": 25000})
    row = eng.get_status_table()[0]
    assert row["last_no_trade_reason"] == "score_below_threshold"
    assert row["confidence"] == 0.42
    assert row["unrealized_pnl"] == 12.5
    assert row["realized_pnl"] == 3.0
    assert row["total_trades"] == 2
    assert row["last_update"] == decision["timestamp"]


def test_ui_merges_latest_decision_cache_into_status_rows():
    ui = _ui_helper()
    decision = {
        "candidate_id": "c1",
        "timestamp": "2026-06-12T09:15:00+05:30",
        "final_signal": "NO_TRADE",
        "no_trade_reason": "low_confidence_0.0000_lt_0.3500",
        "confidence": 0.0,
        "threshold": 0.35,
        "predict_attempted": True,
    }
    ui._pf_remember_decisions([decision])
    rows = ui._pf_merge_cached_decisions_into_rows([{
        "candidate_id": "c1",
        "final_signal": "NO_TRADE",
        "confidence": None,
        "last_no_trade_reason": "candidate_loaded_ok_waiting_for_snapshot",
        "last_update": None,
    }])
    assert rows[0]["last_no_trade_reason"] == "low_confidence_0.0000_lt_0.3500"
    assert rows[0]["confidence"] == 0.0
    assert rows[0]["threshold"] == 0.35
    assert rows[0]["last_update"] == decision["timestamp"]
