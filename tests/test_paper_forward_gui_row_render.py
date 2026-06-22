import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI

def test_render_row_from_state():
    ui = object.__new__(ScalperUI)
    # dummy tree not needed for render
    r = {"enabled": True, "candidate_id": "c1", "model_name": "xgb", "preset_family": "p", "side_policy": "CE", "final_signal": "NO_TRADE", "confidence": None, "predict_attempted": False, "position_status": "FLAT", "unrealized_pnl": 0, "realized_pnl": 0, "total_trades": 0, "win_rate": 0, "max_drawdown": 0, "last_no_trade_reason": "FEATURES_MISSING", "raw_reason": "FEATURES_MISSING", "last_update": "t1"}
    vals = ui._render_paper_forward_row(r) if hasattr(ui, "_render_paper_forward_row") else (None,)
    # basic check
    assert vals[0] == "Y"  # enabled
    assert vals[6] == "FEATURES_MISSING"  # conf shows explicit skip reason, not N/A
    assert vals[7] == "-"
    assert vals[8] == "FEATURES_MISSING"
    assert vals[10] == "-"
    assert vals[11] == "-"
    assert vals[12] == "-"
    assert "feature" in vals[23].lower() or vals[23]  # reason
    print("[TEST] gui row render ok")


def test_render_row_exposes_paper_action_and_trade_reason():
    class Tree:
        def __getitem__(self, key):
            if key == "columns":
                return (
                    "enabled", "candidate_id", "final_signal", "conf", "paper_action", "trade_reason", "pos",
                )
            raise KeyError(key)

    ui = object.__new__(ScalperUI)
    ui.pf_tree = Tree()
    r = {
        "enabled": True,
        "candidate_id": "c1",
        "final_signal": "NO_TRADE",
        "confidence": 9.68e-8,
        "predict_attempted": True,
        "paper_action": "NONE",
        "trade_reason": "low_confidence_9.68e-08_lt_0.3500",
        "position_status": "FLAT",
    }

    vals = ui._render_paper_forward_row(r)

    assert vals == (
        "Y",
        "c1",
        "NO_TRADE",
        "9.68e-08",
        "NONE",
        "low_confidence_9.68e-08_lt_0.3500",
        "FLAT",
    )


def test_cached_decision_does_not_override_newer_status_row():
    ui = object.__new__(ScalperUI)
    ui._pf_latest_decisions_by_row_key = {
        "c1|artifacts/c1|0.3": {
            "candidate_id": "c1",
            "_row_key": "c1|artifacts/c1|0.3",
            "final_signal": "NO_TRADE",
            "confidence": 0.1,
            "last_no_trade_reason": "old_reason",
            "last_update": "2026-06-12T04:00:00+00:00",
        }
    }
    rows = [{
        "candidate_id": "c1",
        "_row_key": "c1|artifacts/c1|0.3",
        "artifact_dir": "artifacts/c1",
        "threshold": 0.3,
        "final_signal": "BUY_CE",
        "confidence": 0.8,
        "last_no_trade_reason": "fresh_reason",
        "last_update": "2026-06-12T04:00:03+00:00",
        "position_status": "OPEN",
    }]

    merged = ui._pf_merge_cached_decisions_into_rows(rows)

    assert merged[0]["final_signal"] == "BUY_CE"
    assert merged[0]["confidence"] == 0.8
    assert merged[0]["last_no_trade_reason"] == "fresh_reason"


class _FakeTree:
    def __init__(self, columns):
        self._columns = tuple(columns)
        self.rows = {}

    def __getitem__(self, key):
        if key == "columns":
            return self._columns
        raise KeyError(key)

    def exists(self, iid):
        return iid in self.rows

    def item(self, iid, values=None):
        if values is not None:
            self.rows[iid] = list(values)
        return {"values": self.rows.get(iid, [])}

    def insert(self, _parent, _index, iid=None, values=(), tags=()):
        self.rows[iid] = list(values)

    def set(self, iid, column, value):
        idx = self._columns.index(column)
        row = self.rows.setdefault(iid, [""] * len(self._columns))
        row[idx] = value

    def get_children(self):
        return list(self.rows)

    def delete(self, iid):
        self.rows.pop(iid, None)

    def update_idletasks(self):
        pass

    def selection(self):
        return ()

    def yview(self):
        return (0.0, 1.0)

    def yview_moveto(self, _value):
        return None


def test_refresh_table_updates_all_existing_row_columns():
    ui = object.__new__(ScalperUI)
    ui.pf_tree = _FakeTree((
        "enabled", "candidate_id", "model", "preset", "side", "final_signal",
        "conf", "pos", "sel_strike", "sel_type", "sel_symbol", "entries",
        "exits", "entry_px", "cur_opt_px", "spot", "unreal_pnl", "real_pnl",
        "trades", "wr", "max_dd", "last_reason", "updated",
    ))

    ui._pf_refresh_table([{
        "enabled": True,
        "candidate_id": "c1",
        "model_name": "old_model",
        "preset_family": "old_preset",
        "side_policy": "CE",
        "final_signal": "NO_TRADE",
        "confidence": 0.1,
        "predict_attempted": True,
        "position_status": "FLAT",
        "selected_strike": None,
        "selected_option_type": "",
        "selected_symbol": "",
        "unrealized_pnl": 0,
        "realized_pnl": 0,
        "total_trades": 0,
        "win_rate": 0,
        "max_drawdown": 0,
        "last_no_trade_reason": "old_reason",
        "last_update": "t1",
    }])
    ui._pf_refresh_table([{
        "enabled": True,
        "candidate_id": "c1",
        "model_name": "new_model",
        "preset_family": "new_preset",
        "side_policy": "PE",
        "final_signal": "BUY_PE",
        "confidence": 0.876,
        "predict_attempted": True,
        "position_status": "OPEN",
        "selected_strike": 24000,
        "selected_option_type": "PE",
        "selected_symbol": "NIFTY24000PE",
        "unrealized_pnl": 12.3,
        "realized_pnl": 4.5,
        "total_trades": 7,
        "win_rate": 0.25,
        "max_drawdown": -1.2,
        "last_no_trade_reason": "fresh_reason",
        "last_update": "t2",
    }])

    assert ui.pf_tree.rows["c1"] == [
        "Y", "c1", "new_model", "new_preset", "PE", "BUY_PE",
        "0.8760", "OPEN", "24000", "PE", "NIFTY24000PE", 0,
        0, "-", "-", "-", "12.30", "4.50", 7, "25.00%",
        "-1.2", "fresh_reason", "t2",
    ]


def test_empty_refresh_preserves_loaded_paper_forward_candidates():
    ui = object.__new__(ScalperUI)
    ui.pf_tree = _FakeTree(("enabled", "candidate_id", "model"))
    ui.pf_candidates = [
        {"enabled": True, "candidate_id": "c1", "model_name": "m1", "allowed_modes": ["paper_forward"]},
        {"enabled": True, "candidate_id": "c2", "model_name": "m2", "allowed_modes": ["paper_forward"]},
    ]
    ui._paper_forward_state = {"candidates": list(ui.pf_candidates), "rows": list(ui.pf_candidates)}
    ui._pf_last_gui_rows_by_candidate = {}
    ui._pf_lifecycle_by_candidate = {}
    ui._pf_lifecycle_by_row_key = {}
    ui._pf_lifecycle_statuses_by_candidate = {}
    ui._pf_tree_iids = {}
    ui._pf_last_display_values = {}

    ui._pf_refresh_table(ui.pf_candidates, allow_full_clear=True)
    assert set(ui.pf_tree.rows) == {"c1", "c2"}


def test_empty_engine_candidate_list_does_not_clear_config_baseline_rows():
    ui = object.__new__(ScalperUI)
    ui.pf_tree = _FakeTree(("enabled", "candidate_id", "model"))
    baseline = [
        {"enabled": True, "candidate_id": "c1", "model_name": "m1", "allowed_modes": ["paper_forward"]},
        {"enabled": True, "candidate_id": "c2", "model_name": "m2", "allowed_modes": ["paper_forward"]},
    ]
    ui._pf_config_candidate_rows = list(baseline)
    ui.pf_candidates = []
    ui._paper_forward_state = {"candidates": [], "rows": []}
    ui._pf_last_gui_rows_by_candidate = {}
    ui._pf_lifecycle_by_candidate = {}
    ui._pf_lifecycle_by_row_key = {}
    ui._pf_lifecycle_statuses_by_candidate = {}
    ui._pf_tree_iids = {}
    ui._pf_last_display_values = {}

    ui._pf_refresh_table([])

    assert set(ui.pf_tree.rows) == {"c1", "c2"}

    ui._pf_refresh_table([])

    assert set(ui.pf_tree.rows) == {"c1", "c2"}
