"""
Tests for v2 leakage audit (scripts/build_v2_research_dataset.py).

These tests cover the leakage self-check that runs at the end of every
build_v2_labels() call. They cover:
  - Forbidden input tokens are detected.
  - Required input columns are detected.
  - TB labels are in {-1, 0, 1, NaN}.
  - TB hit_time is strictly > entry timestamp.
  - TB return is consistent with the barrier config.
  - The CLI's --fail-on-leakage flag returns exit code 2 on failure.
  - Cost-adjusted labels never produce negative infinite or NaN-only
    columns on a clean frame.
  - The audit JSON is well-formed and contains the expected top-level
    sections.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_v2_research_dataset import (  # noqa: E402
    FORBIDDEN_INPUT_TOKENS,
    TripleBarrierConfig,
    build_triple_barrier_labels,
    build_v2_labels,
    default_v2_config,
    run_leakage_audit,
    validate_input_frame,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_synthetic_frame() -> pd.DataFrame:
    base_ts = pd.Timestamp("2026-06-07T09:15:00", tz="Asia/Kolkata")
    rows = []
    rng = np.random.default_rng(123)
    for inst in range(3):
        for i in range(40):
            rows.append(
                {
                    "timestamp": base_ts + pd.Timedelta(minutes=i),
                    "instrument_key": f"NIFTY_{inst}",
                    "ltp": 100.0 * (1.0 + 0.001 * i + 0.01 * rng.standard_normal()),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "volume": int(rng.integers(100, 1000)),
                    "oi": int(rng.integers(1000, 10000)),
                    "primary_signal_proba": float(rng.uniform(0.3, 0.8)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def labeled_clean_frame(clean_synthetic_frame: pd.DataFrame) -> pd.DataFrame:
    out, _ = build_v2_labels(clean_synthetic_frame, default_v2_config())
    return out


# ---------------------------------------------------------------------------
# Forbidden input tokens
# ---------------------------------------------------------------------------


class TestForbiddenInputTokens:
    def test_clean_frame_passes(self, clean_synthetic_frame: pd.DataFrame) -> None:
        result = validate_input_frame(clean_synthetic_frame)
        assert result["ok"]
        assert result["errors"] == []

    @pytest.mark.parametrize("forbidden", sorted(FORBIDDEN_INPUT_TOKENS))
    def test_each_forbidden_token_rejected(self, forbidden: str) -> None:
        df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-06-07T09:15:00", tz="Asia/Kolkata")],
                "instrument_key": ["A"],
                "ltp": [100.0],
                forbidden: [1.0],
            }
        )
        result = validate_input_frame(df)
        assert not result["ok"], f"forbidden token {forbidden!r} should be rejected"
        assert any(forbidden in e for e in result["errors"])


# ---------------------------------------------------------------------------
# TB label value set
# ---------------------------------------------------------------------------


class TestLabelValueSet:
    def test_all_tb_labels_in_valid_set(self, labeled_clean_frame: pd.DataFrame) -> None:
        for col in labeled_clean_frame.columns:
            if col.endswith("_label") and col.startswith("tb_"):
                s = pd.to_numeric(labeled_clean_frame[col], errors="coerce").dropna()
                if s.empty:
                    continue
                unexpected = sorted(set(s.unique()) - {-1.0, 0.0, 1.0})
                assert not unexpected, f"{col} has unexpected values: {unexpected}"

    def test_econ_labels_finite(self, labeled_clean_frame: pd.DataFrame) -> None:
        for col in labeled_clean_frame.columns:
            if col.startswith("econ_") and col.endswith("_pct"):
                s = pd.to_numeric(labeled_clean_frame[col], errors="coerce").dropna()
                if s.empty:
                    continue
                assert np.isfinite(s).all(), f"{col} contains non-finite values"


# ---------------------------------------------------------------------------
# Hit time and entry timestamp
# ---------------------------------------------------------------------------


class TestHitTime:
    def test_hit_time_strictly_after_entry(self, labeled_clean_frame: pd.DataFrame) -> None:
        # Use ns-since-epoch int64 comparison; the frame's datetime columns
        # may be datetime64[us] so .astype('int64') on the Series would yield
        # microseconds. Cast the underlying array to datetime64[ns] first.
        entry_ns = (
            np.asarray(labeled_clean_frame["timestamp"].array, dtype="datetime64[ns]")
            .view("i8")
            .copy()
        )
        for col in labeled_clean_frame.columns:
            if col.endswith("_hit_time") and col.startswith("tb_"):
                label_col = col.replace("_hit_time", "_label")
                label = pd.to_numeric(labeled_clean_frame[label_col], errors="coerce")
                hit_ns = (
                    np.asarray(labeled_clean_frame[col].array, dtype="datetime64[ns]")
                    .view("i8")
                    .copy()
                )
                bad = label.notna() & ~pd.isna(hit_ns) & (hit_ns <= entry_ns)
                assert int(bad.sum()) == 0, f"{col}: {int(bad.sum())} rows have hit_time <= entry"

    def test_hit_time_within_horizon(self, labeled_clean_frame: pd.DataFrame) -> None:
        # For each TB label, the hit_time - entry_ts must not exceed the
        # configured horizon. The frame's datetime columns may be in
        # datetime64[us, Asia/Kolkata]; .astype('int64') on the Series would
        # return microseconds. Cast the underlying array to datetime64[ns]
        # and view as i8 to get ns-since-epoch.
        config_by_name = {tb.name: tb for tb in default_v2_config().triple_barriers}
        entry_ns = (
            np.asarray(labeled_clean_frame["timestamp"].array, dtype="datetime64[ns]")
            .view("i8")
            .copy()
        )
        for col in labeled_clean_frame.columns:
            if not (col.endswith("_hit_time") and col.startswith("tb_")):
                continue
            name = col[: -len("_hit_time")]
            tb = config_by_name.get(name)
            if tb is None:
                continue
            label_col = f"{name}_label"
            label = pd.to_numeric(labeled_clean_frame[label_col], errors="coerce")
            hit_ns = (
                np.asarray(labeled_clean_frame[col].array, dtype="datetime64[ns]")
                .view("i8")
                .copy()
            )
            labeled_rows = label.notna() & ~pd.isna(hit_ns)
            # delta in minutes (ns / 1e9 / 60)
            delta_min = (hit_ns - entry_ns) / 1e9 / 60.0
            within_horizon = (delta_min <= tb.horizon_min + 1.0)
            bad = labeled_rows & ~within_horizon
            assert int(bad.sum()) == 0, (
                f"{col}: {int(bad.sum())} hit_times are outside the {tb.horizon_min}-min horizon"
            )


# ---------------------------------------------------------------------------
# TB return consistency
# ---------------------------------------------------------------------------


class TestReturnConsistency:
    def test_profit_take_return_equals_pt_pct(self, labeled_clean_frame: pd.DataFrame) -> None:
        config_by_name = {tb.name: tb for tb in default_v2_config().triple_barriers}
        for col in labeled_clean_frame.columns:
            if not (col.endswith("_return") and col.startswith("tb_")):
                continue
            name = col[: -len("_return")]
            label_col = f"{name}_label"
            tb = config_by_name.get(name)
            if tb is None:
                continue
            label = pd.to_numeric(labeled_clean_frame[label_col], errors="coerce")
            ret = pd.to_numeric(labeled_clean_frame[col], errors="coerce")
            # +1 labels: return == +pt_pct
            pt = (label == 1) & ret.notna()
            assert np.allclose(ret[pt], tb.pt_pct, atol=1e-9), f"{col}: +1 returns != +pt_pct"
            # -1 labels: return == -sl_pct
            sl = (label == -1) & ret.notna()
            assert np.allclose(ret[sl], -tb.sl_pct, atol=1e-9), f"{col}: -1 returns != -sl_pct"
            # 0 labels: return is the time-out return; bounded by [-sl_pct, +pt_pct] in practice.
            # We just check it is finite.
            to = (label == 0) & ret.notna()
            assert np.isfinite(ret[to]).all(), f"{col}: 0 labels have non-finite return"


# ---------------------------------------------------------------------------
# Top-level run_leakage_audit
# ---------------------------------------------------------------------------


class TestRunLeakageAudit:
    def test_clean_labeled_frame_passes(self, labeled_clean_frame: pd.DataFrame) -> None:
        report = run_leakage_audit(labeled_clean_frame)
        assert report["ok"]
        assert report["errors"] == []
        assert report["forbidden_present"] == []
        assert len(report["tb_label_columns_found"]) >= 1

    def test_forbidden_column_in_labeled_frame_fails(self, labeled_clean_frame: pd.DataFrame) -> None:
        bad = labeled_clean_frame.copy()
        bad["future_close"] = 1.0
        report = run_leakage_audit(bad)
        assert not report["ok"]
        assert "future_close" in report["forbidden_present"]

    def test_tampered_hit_time_fails(self, labeled_clean_frame: pd.DataFrame) -> None:
        bad = labeled_clean_frame.copy()
        # Move the first non-NaN hit_time to before the entry timestamp.
        col = next(c for c in bad.columns if c.endswith("_hit_time") and c.startswith("tb_"))
        label_col = col.replace("_hit_time", "_label")
        mask = bad[label_col].notna() & bad[col].notna()
        idx = bad.index[mask][0]
        # Use the int64-epoch approach to set a tampered time. Match the
        # column's timezone to avoid pandas dtype errors.
        entry_ns = int(pd.to_datetime(bad.loc[idx, "timestamp"]).value)
        col_tz = getattr(bad[col].dt, "tz", None)
        tampered = pd.to_datetime(entry_ns - 60 * 1_000_000_000, unit="ns")
        if col_tz is not None:
            tampered = tampered.tz_localize(col_tz)
        bad.loc[idx, col] = tampered
        report = run_leakage_audit(bad)
        assert not report["ok"]
        assert any("hit_time" in e for e in report["errors"])

    def test_unexpected_label_value_fails(self, labeled_clean_frame: pd.DataFrame) -> None:
        bad = labeled_clean_frame.copy()
        col = next(c for c in bad.columns if c.endswith("_label") and c.startswith("tb_"))
        bad.loc[bad.index[0], col] = 7.0
        report = run_leakage_audit(bad)
        assert not report["ok"]
        assert any("unexpected values" in e for e in report["errors"])


# ---------------------------------------------------------------------------
# Audit JSON shape
# ---------------------------------------------------------------------------


class TestAuditJsonShape:
    def test_audit_json_has_expected_sections(self, labeled_clean_frame: pd.DataFrame) -> None:
        from build_v2_research_dataset import build_v2_labels, default_v2_config

        _, audit = build_v2_labels(labeled_clean_frame, default_v2_config())
        # Top-level keys.
        assert "build_timestamp_utc" in audit
        assert "config" in audit
        assert "input_audit" in audit
        assert "rows_in" in audit
        assert "rows_out" in audit
        assert "label_audits" in audit
        assert "leakage_audit" in audit
        # 7 TB + 1 meta + 1 economic = 9 entries.
        assert len(audit["label_audits"]) == 9
        # Serialize / deserialize roundtrip.
        s = json.dumps(audit, default=str)
        loaded = json.loads(s)
        assert loaded["rows_in"] == audit["rows_in"]
        assert loaded["rows_out"] == audit["rows_out"]

    def test_audit_label_audits_have_class_balance(self, labeled_clean_frame: pd.DataFrame) -> None:
        _, audit = build_v2_labels(labeled_clean_frame, default_v2_config())
        for name, body in audit["label_audits"].items():
            if "value_counts" in body:
                # TB audit shape.
                assert "rows_total" in body
                assert "rows_labeled" in body
                assert "rows_skipped" in body
            else:
                # Meta or economic shape.
                assert isinstance(body, dict)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_runs_on_synthetic_csv(
        self, tmp_path: Path, clean_synthetic_frame: pd.DataFrame
    ) -> None:
        in_path = tmp_path / "in.csv"
        out_path = tmp_path / "out.parquet"
        audit_path = tmp_path / "audit.json"
        leak_path = tmp_path / "leak.json"
        in_path.write_text(clean_synthetic_frame.to_csv(index=False), encoding="utf-8")
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "build_v2_research_dataset.py"),
            "--input", str(in_path),
            "--output", str(out_path),
            "--audit-output", str(audit_path),
            "--leakage-report", str(leak_path),
            "--fail-on-leakage",
            "--quiet",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert out_path.exists()
        assert audit_path.exists()
        assert leak_path.exists()
        # Leakage report is well-formed.
        leak = json.loads(leak_path.read_text(encoding="utf-8"))
        assert leak["ok"] is True

    def test_cli_exit_code_2_on_leakage(
        self, tmp_path: Path, clean_synthetic_frame: pd.DataFrame
    ) -> None:
        in_path = tmp_path / "in.csv"
        out_path = tmp_path / "out.parquet"
        # Inject a forbidden column.
        bad = clean_synthetic_frame.copy()
        bad["future_close"] = 1.0
        in_path.write_text(bad.to_csv(index=False), encoding="utf-8")
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "build_v2_research_dataset.py"),
            "--input", str(in_path),
            "--output", str(out_path),
            "--fail-on-leakage",
            "--quiet",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode != 0
        # The script raises in build_v2_labels when the input audit fails;
        # --fail-on-leakage would also return 2 if the leakage audit fails.
        # We just check the exit code is non-zero.

    def test_cli_returns_exit_code_2_when_post_build_leakage_audit_fails(
        self, tmp_path: Path, clean_synthetic_frame: pd.DataFrame, monkeypatch
    ) -> None:
        # In-process main() call with a clean input. Monkeypatch the
        # post-build run_leakage_audit to return ok=False so the CLI's
        # --fail-on-leakage branch (return 2) is exercised. This is the
        # exact contract documented in the script's argparse help.
        from build_v2_research_dataset import main as _v2_main

        in_path = tmp_path / "in.csv"
        out_path = tmp_path / "out.parquet"
        in_path.write_text(clean_synthetic_frame.to_csv(index=False), encoding="utf-8")

        def _forced_audit_fail(df: pd.DataFrame) -> dict:
            return {
                "ok": False,
                "errors": ["forced failure for test"],
                "warnings": [],
                "forbidden_present": [],
                "tb_label_columns_found": [],
            }

        import build_v2_research_dataset as _v2_mod
        monkeypatch.setattr(_v2_mod, "run_leakage_audit", _forced_audit_fail)

        rc = _v2_main(
            [
                "--input", str(in_path),
                "--output", str(out_path),
                "--fail-on-leakage",
                "--quiet",
            ]
        )
        assert rc == 2
