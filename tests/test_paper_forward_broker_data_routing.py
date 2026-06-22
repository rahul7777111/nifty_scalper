import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ui as ui_mod
from paper_forward_engine import PaperForwardDataStatus, PaperForwardEngine


def test_dhan_option_chain_underlying_value_produces_spot_ok(monkeypatch):
    monkeypatch.setenv("SCALPER_BROKER", "dhan")
    monkeypatch.setenv("DHAN_CLIENT_ID", "cid")
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "tok")
    monkeypatch.delenv("MSTOCK_OPTION_EXPIRY", raising=False)
    monkeypatch.delenv("MSTOCK_OPTION_EXCHANGE_ID", raising=False)

    chain = [
        {"strike": 24000 + i * 50, "option_type": "PE", "ltp": 100, "raw": {"underlyingValue": 23948.7}}
        for i in range(24)
    ]
    status = PaperForwardDataStatus.from_snapshot({"broker_name": "dhan", "broker_auth": "AUTH_OK"}, chain)
    d = status.as_dict()
    assert d["spot_status"] == "spot_ok"
    assert d["option_chain_status"] == "DATA_OK"


def test_mstock_missing_expiry_is_broker_config_missing(monkeypatch):
    monkeypatch.setenv("SCALPER_BROKER", "mstock")
    monkeypatch.setenv("MSTOCK_OPTION_EXCHANGE_ID", "5")
    monkeypatch.setenv("MSTOCK_OPTION_TOKEN", "13")
    monkeypatch.delenv("MSTOCK_OPTION_EXPIRY", raising=False)

    cfg = ui_mod.validate_option_chain_config_for_active_broker("mstock")
    assert cfg["missing_keys"] == ["MSTOCK_OPTION_EXPIRY"]
    status = PaperForwardDataStatus.from_snapshot(
        {
            "broker_name": "mstock",
            "broker_auth": "AUTH_OK",
            "option_chain_status": "broker_config_missing",
            "option_chain_error": "broker_config_missing: missing_keys=MSTOCK_OPTION_EXPIRY",
        },
        [],
    )
    assert status.as_dict()["data_quality_status"] == "BROKER_CONFIG_MISSING"


def test_dhan_does_not_require_mstock_option_chain_config(monkeypatch):
    monkeypatch.setenv("SCALPER_BROKER", "dhan")
    monkeypatch.setenv("DHAN_CLIENT_ID", "cid")
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("DHAN_UNDERLYING_SECURITY_ID", "13")
    monkeypatch.delenv("MSTOCK_OPTION_EXPIRY", raising=False)
    monkeypatch.delenv("MSTOCK_OPTION_EXCHANGE_ID", raising=False)
    monkeypatch.delenv("MSTOCK_OPTION_TOKEN", raising=False)

    cfg = ui_mod.validate_option_chain_config_for_active_broker("dhan")
    assert cfg["ok"] is True
    assert cfg["missing_keys"] == []
    assert "MSTOCK_OPTION_EXPIRY" not in cfg["missing_keys"]


def test_dhan_requires_canonical_underlying_security_id(monkeypatch):
    monkeypatch.setenv("SCALPER_BROKER", "dhan")
    monkeypatch.setenv("DHAN_CLIENT_ID", "cid")
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "tok")
    monkeypatch.delenv("DHAN_UNDERLYING_SECURITY_ID", raising=False)
    monkeypatch.delenv("DHAN_UNDER_SECURITY_ID", raising=False)
    monkeypatch.delenv("DHAN_NIFTY_SECURITY_ID", raising=False)

    cfg = ui_mod.validate_option_chain_config_for_active_broker("dhan")
    assert cfg["ok"] is False
    assert "DHAN_UNDERLYING_SECURITY_ID_OR_DHAN_NIFTY_SECURITY_ID" in cfg["missing_keys"]


def test_dhan_accepts_nifty_security_id_alias(monkeypatch):
    monkeypatch.setenv("SCALPER_BROKER", "dhan")
    monkeypatch.setenv("DHAN_CLIENT_ID", "cid")
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "tok")
    monkeypatch.delenv("DHAN_UNDERLYING_SECURITY_ID", raising=False)
    monkeypatch.setenv("DHAN_NIFTY_SECURITY_ID", "13")
    monkeypatch.delenv("MSTOCK_OPTION_EXPIRY", raising=False)
    monkeypatch.delenv("MSTOCK_OPTION_EXCHANGE_ID", raising=False)
    monkeypatch.delenv("MSTOCK_OPTION_TOKEN", raising=False)

    cfg = ui_mod.validate_option_chain_config_for_active_broker("dhan")

    assert cfg["ok"] is True
    assert cfg["missing_keys"] == []


def test_snapshot_cache_prefers_paper_forward_canonical_rows():
    obj = ui_mod.ScalperUI.__new__(ui_mod.ScalperUI)
    obj._market_data_lock = None
    obj._paper_forward_option_chain_rows = [{"strike_price": 24000, "option_type": "PE", "ltp": 100}]
    obj._latest_option_chain_rows = [{"strike_price": 1, "option_type": "CE", "ltp": 1}]
    obj._option_chain_data = []
    obj.option_chain_data = []
    obj.latest_option_chain = []
    obj._latest_option_chain_cache = []
    obj._last_option_chain_rows = []
    obj._selected_broker = lambda: "dhan"

    rows, source = ui_mod.ScalperUI._get_cached_option_chain_rows(obj)

    assert source == "ui._paper_forward_option_chain_rows"
    assert rows[0]["strike_price"] == 24000


def test_empty_option_chain_status_is_broker_specific(monkeypatch):
    monkeypatch.setenv("SCALPER_BROKER", "dhan")
    monkeypatch.setenv("DHAN_CLIENT_ID", "cid")
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("DHAN_NIFTY_SECURITY_ID", "13")
    cfg = ui_mod.validate_option_chain_config_for_active_broker("dhan")
    assert cfg["ok"] is True
    assert cfg["empty_reason"] == ""


def test_candidate_config_duplicate_exact_row_dedupes(tmp_path):
    art = tmp_path / "c1"
    art.mkdir()
    cfg = {
        "candidates": [
            {"candidate_id": "c1", "paper_forward_only": True, "enabled": False, "artifact_dir": str(art), "model_name": "m", "preset_family": "p", "side_policy": "PE_ONLY", "threshold": 0.35},
            {"candidate_id": "c1", "paper_forward_only": True, "enabled": False, "artifact_dir": str(art), "model_name": "m", "preset_family": "p", "side_policy": "PE_ONLY", "threshold": 0.35},
        ]
    }
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    eng = PaperForwardEngine(candidate_file=str(path), artifacts_dir=str(tmp_path), broker_safe_mode=True)
    summary = eng.get_diagnostics()["candidate_load_summary"]
    assert summary["config_candidates"] == 2
    assert summary["loaded_candidates"] == 1
    assert summary["duplicate_removed_count"] == 1


def test_candidate_config_16_rows_not_reduced_without_duplicates(tmp_path):
    candidates = [
        {
            "candidate_id": f"c{i}",
            "paper_forward_only": True,
            "enabled": False,
            "artifact_dir": str(tmp_path / f"missing{i}"),
            "model_name": "m",
            "preset_family": "p",
            "side_policy": "PE_ONLY",
            "threshold": i / 100,
        }
        for i in range(16)
    ]
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps({"candidates": candidates}), encoding="utf-8")
    eng = PaperForwardEngine(candidate_file=str(path), artifacts_dir=str(tmp_path), broker_safe_mode=True)
    assert len(eng.candidates) == 16
    assert eng.get_diagnostics()["candidate_load_summary"]["duplicate_removed_count"] == 0


def test_stale_loop_generation_ui_update_is_ignored():
    obj = ui_mod.ScalperUI.__new__(ui_mod.ScalperUI)
    obj._pf_loop_generation = 2
    obj.pf_engine = object()
    obj._pf_refresh_monitor_ui([], source="test", generation=1)
    assert obj._pf_loop_generation == 2


def test_start_multi_starts_data_poller_when_main_bot_idle(monkeypatch):
    class Var:
        def __init__(self):
            self.value = ""
        def set(self, value):
            self.value = value

    class FakeEngine:
        def __init__(self, *args, **kwargs):
            self.candidates = []
        def get_status_table(self):
            return []
        def write_summary(self):
            return {}

    obj = ui_mod.ScalperUI.__new__(ui_mod.ScalperUI)
    obj.pf_status_var = Var()
    obj.pf_live_var = Var()
    obj._pf_engine_thread = None
    obj._pf_stop_event = None
    obj._pf_loop_generation = 0
    obj._scalper = None
    obj._pf_runtime = ui_mod.PaperForwardRuntime()
    obj._pf_auth_state = obj._pf_runtime.auth
    obj._pf_refresh_monitor_ui = lambda *a, **k: None
    obj._pf_schedule_update = lambda *a, **k: None
    obj._pf_run_loop = lambda generation=0: None
    obj._pf_ensure_auth_terminal = lambda force=False: ("TOKEN_MISSING", "", None)
    obj._build_paper_snapshot_for_engine = lambda: {}
    obj._get_cached_option_chain_rows = lambda: ([], "")
    obj._pf_set_table_auth_block_reason = lambda status: None
    obj._pf_auth_trace = lambda *a, **k: None
    started = {}
    obj._start_paper_forward_data_poller = lambda generation=None: started.setdefault("generation", generation)
    monkeypatch.setattr(ui_mod, "PaperForwardEngine", FakeEngine)
    monkeypatch.delenv("MSTOCK_ENABLE_LIVE_ORDERS", raising=False)

    obj._pf_start_multi()
    assert started["generation"] == 1
