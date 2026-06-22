from __future__ import annotations

from types import SimpleNamespace

from src.candidate_lifecycle import (
    CandidateLifecycleStatus,
    STAGE_ARTIFACT_MISSING,
    STAGE_DISABLED,
    STAGE_PAPER_FORWARD_ELIGIBLE,
    lifecycle_status_visible,
)


def test_invalid_candidate_hidden_by_default():
    status = CandidateLifecycleStatus(candidate_id="bad", current_stage=STAGE_ARTIFACT_MISSING, block_reason="MODEL_FILE_MISSING")
    assert lifecycle_status_visible(status, "Show Qualified Only") is False
    assert lifecycle_status_visible(status, "Show Blocked / Rejected") is True


def test_valid_candidate_visible_in_qualified_view():
    status = CandidateLifecycleStatus(candidate_id="good", current_stage=STAGE_PAPER_FORWARD_ELIGIBLE)
    assert lifecycle_status_visible(status, "Show Qualified Only") is True


def test_paper_observation_only_hidden_from_default_but_visible_in_paper_view():
    status = CandidateLifecycleStatus(
        candidate_id="paper_only",
        current_stage=STAGE_PAPER_FORWARD_ELIGIBLE,
        block_reason="PAPER_OBSERVATION_ONLY_NOT_LIVE_PATH",
        possible_live_path=False,
    )
    assert lifecycle_status_visible(status, "Show Qualified Only") is False
    assert lifecycle_status_visible(status, "Show Paper Forward Eligible") is True


class FakeTree:
    def __init__(self):
        self.rows = {}

    def selection(self):
        return ()

    def yview(self):
        return (0.0, 1.0)

    def yview_moveto(self, _pos):
        return None

    def exists(self, iid):
        return iid in self.rows

    def item(self, iid, **kwargs):
        if kwargs:
            self.rows.setdefault(iid, {"values": (), "tags": ()}).update(kwargs)
            return None
        return self.rows.get(iid, {"values": (), "tags": ()})

    def insert(self, _parent, _index, iid, values, tags=()):
        self.rows[iid] = {"values": tuple(values), "tags": tuple(tags)}

    def get_children(self):
        return tuple(self.rows)

    def delete(self, iid):
        self.rows.pop(iid, None)

    def __getitem__(self, key):
        if key == "columns":
            return ("candidate_id",)
        raise KeyError(key)


def test_gui_table_updates_existing_rows_instead_of_inserting_duplicates():
    from src.ui import ScalperUI

    app = SimpleNamespace(_tab_last_update_ts={}, _is_ui_alive=lambda widget=None: True)
    tree = FakeTree()
    upsert = ScalperUI._upsert_tree_rows
    mapping = upsert(app, tree, [{"iid": "candidate-1", "values": ("candidate-1", "A")}], tab="test")
    mapping = upsert(app, tree, [{"iid": "candidate-1", "values": ("candidate-1", "B")}], iid_map=mapping, tab="test")

    assert len(tree.rows) == 1
    assert tree.rows["candidate-1"]["values"] == ("candidate-1", "B")


def test_paper_forward_filter_to_empty_removes_stale_rows():
    from src.ui import ScalperUI

    tree = FakeTree()
    tree.rows["paper_only"] = {"values": ("paper_only",), "tags": ()}
    blocked = CandidateLifecycleStatus(
        candidate_id="paper_only",
        current_stage=STAGE_PAPER_FORWARD_ELIGIBLE,
        block_reason="PAPER_OBSERVATION_ONLY_NOT_LIVE_PATH",
        possible_live_path=False,
    )
    app = SimpleNamespace()
    app.pf_tree = tree
    app._gui_is_closing = lambda: False
    app._gui_widget_alive = lambda widget=None: True
    app._pf_sort_rows_stable = lambda rows: rows
    app._pf_lifecycle_by_candidate = {"paper_only": blocked}
    app._pf_lifecycle_by_row_key = {}
    app._pf_tree_iids = {"paper_only": "paper_only"}
    app._pf_last_display_values = {"paper_only": ("paper_only",)}
    app._pf_row_cache_key = lambda row: str(row.get("candidate_id") or "")
    app._pf_runtime_id = lambda row: str(row.get("candidate_id") or "")
    app._pf_stabilize_row_data = lambda runtime_id, row: row
    app._render_paper_forward_row = lambda row: (row.get("candidate_id"),)
    app._pf_coalesce_display_values = lambda runtime_id, old_vals, vals: vals
    app._pf_lifecycle_filter_rows = lambda rows: ScalperUI._pf_lifecycle_filter_rows(app, rows)
    app._pf_dict_get = lambda key, default=None: getattr(app, key, default)

    ScalperUI._pf_refresh_table(app, [{"candidate_id": "paper_only"}])

    assert tree.rows == {}


def test_paper_forward_filter_hides_rows_with_enabled_n():
    from src.ui import ScalperUI

    app = SimpleNamespace()
    app.pf_lifecycle_filter_var = SimpleNamespace(get=lambda: "Show Paper Forward Eligible")
    app._pf_lifecycle_by_candidate = {}
    app._pf_lifecycle_by_row_key = {}
    app._pf_row_cache_key = lambda row: str(row.get("candidate_id") or "")

    rows = [{"candidate_id": "paper_only_disabled", "enabled": "N"}]

    assert ScalperUI._pf_lifecycle_filter_rows(app, rows) == []


def test_paper_forward_filter_hides_rows_missing_enabled_in_eligible_view():
    from src.ui import ScalperUI

    status = CandidateLifecycleStatus(candidate_id="paper_missing_enabled", current_stage=STAGE_PAPER_FORWARD_ELIGIBLE)
    app = SimpleNamespace()
    app.pf_lifecycle_filter_var = SimpleNamespace(get=lambda: "Show Paper Forward Eligible")
    app._pf_lifecycle_by_candidate = {"paper_missing_enabled": status}
    app._pf_lifecycle_statuses_by_candidate = {"paper_missing_enabled": [status]}
    app._pf_lifecycle_by_row_key = {}
    app._pf_row_cache_key = lambda row: str(row.get("candidate_id") or "")

    rows = [{"candidate_id": "paper_missing_enabled"}]

    assert ScalperUI._pf_lifecycle_filter_rows(app, rows) == []


def test_paper_forward_filter_hides_lifecycle_disabled_even_if_row_stale_enabled():
    from src.ui import ScalperUI

    status = CandidateLifecycleStatus(candidate_id="paper_disabled", current_stage=STAGE_DISABLED, block_reason="DISABLED")
    app = SimpleNamespace()
    app.pf_lifecycle_filter_var = SimpleNamespace(get=lambda: "Show Paper Forward Eligible")
    app._pf_lifecycle_by_candidate = {"paper_disabled": status}
    app._pf_lifecycle_statuses_by_candidate = {"paper_disabled": [status]}
    app._pf_lifecycle_by_row_key = {"paper_disabled": status}
    app._pf_row_cache_key = lambda row: str(row.get("candidate_id") or "")

    rows = [{"candidate_id": "paper_disabled", "enabled": True}]

    assert ScalperUI._pf_lifecycle_filter_rows(app, rows) == []


def test_paper_forward_filter_handles_duplicate_ids_without_showing_disabled_variant():
    from src.ui import ScalperUI

    enabled_status = CandidateLifecycleStatus(candidate_id="dup", current_stage=STAGE_PAPER_FORWARD_ELIGIBLE)
    disabled_status = CandidateLifecycleStatus(candidate_id="dup", current_stage=STAGE_DISABLED, block_reason="DISABLED")
    app = SimpleNamespace()
    app.pf_lifecycle_filter_var = SimpleNamespace(get=lambda: "Show Paper Forward Eligible")
    app._pf_lifecycle_by_candidate = {"dup": disabled_status}
    app._pf_lifecycle_statuses_by_candidate = {"dup": [enabled_status, disabled_status]}
    app._pf_lifecycle_by_row_key = {}
    app._pf_row_cache_key = lambda row: str(row.get("candidate_id") or "")

    rows = [
        {"candidate_id": "dup", "enabled": True},
        {"candidate_id": "dup", "enabled": False},
    ]

    filtered = ScalperUI._pf_lifecycle_filter_rows(app, rows)

    assert filtered == [{"candidate_id": "dup", "enabled": True, "lifecycle_stage": STAGE_PAPER_FORWARD_ELIGIBLE, "lifecycle_block_reason": "", "lifecycle_can_advance": False}]


def test_paper_forward_filter_shows_shadow_rows_allowed_for_paper_forward():
    from src.ui import ScalperUI

    status = CandidateLifecycleStatus(
        candidate_id="shadow_pf",
        current_stage="SHADOW_RUNNING",
        block_reason="SHADOW_RUNNING",
    )
    app = SimpleNamespace()
    app.pf_lifecycle_filter_var = SimpleNamespace(get=lambda: "Show Paper Forward Eligible")
    app._pf_lifecycle_by_candidate = {"shadow_pf": status}
    app._pf_lifecycle_statuses_by_candidate = {"shadow_pf": [status]}
    app._pf_lifecycle_by_row_key = {}
    app._pf_row_cache_key = lambda row: str(row.get("candidate_id") or "")

    rows = [{"candidate_id": "shadow_pf", "enabled": True, "status": "SHADOW", "allowed_modes": ["shadow", "paper_forward"]}]

    filtered = ScalperUI._pf_lifecycle_filter_rows(app, rows)

    assert len(filtered) == 1
    assert filtered[0]["candidate_id"] == "shadow_pf"
    assert filtered[0]["lifecycle_stage"] == "SHADOW_RUNNING"
