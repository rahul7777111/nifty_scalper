"""
Test suite for avoid_trade_label trivial-inverse audit.

Context (2026-06-09 audit):
  - avoid_trade_label = (net_forward_return <= 0.0)
  - profitable_trade_label = (net_forward_return > 0.0)
  - Therefore: avoid_trade_label == 1 - profitable_trade_label (trivial inverse)

Goals:
1. Confirm avoid_trade_label is exactly inverse of profitable_trade_label.
2. Verify that the trivial-inverse flag is set in the overlay audit.
3. Verify that choose_labels() correctly skips avoid_trade_label when flagged.
4. Verify that no code path can promote a trivial-inverse avoid model as a
   real second-stage edge without checking the trivial_inverse flag first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATA_DIR = REPO_ROOT / "data" / "processed"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_latest_cost_aware_dataset(limit: int = 100_000) -> pd.DataFrame:
    """Load the latest cost-aware edge dataset, up to `limit` rows."""
    candidates = sorted(
        DATA_DIR.glob("nifty_option_chain_cost_aware_edge_dataset_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No nifty_option_chain_cost_aware_edge_dataset_*.csv found in {DATA_DIR}"
        )
    return pd.read_csv(candidates[0], low_memory=False, nrows=limit)


def is_trivial_inverse(df: pd.DataFrame, profit_col: str, avoid_col: str) -> bool:
    """Return True if avoid_col == 1 - profit_col for all valid rows."""
    profit = pd.to_numeric(df[profit_col], errors="coerce")
    avoid = pd.to_numeric(df[avoid_col], errors="coerce")
    mask = profit.notna() & avoid.notna()
    if not mask.any():
        return False
    return bool(np.allclose(
        avoid[mask].to_numpy(dtype=float),
        1.0 - profit[mask].to_numpy(dtype=float),
        equal_nan=False,
    ))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAvoidLabelTrivialInverse:
    """Tests verifying that trivial-inverse avoid labels cannot be promoted."""

    def test_avoid_is_trivial_inverse_on_current_binary_dataset(self):
        """CONFIRMED: avoid_trade_label == 1 - profitable_trade_label on the dataset."""
        df = load_latest_cost_aware_dataset()
        assert "profitable_trade_label" in df.columns, "profitable_trade_label column missing"
        assert "avoid_trade_label" in df.columns, "avoid_trade_label column missing"

        trivial = is_trivial_inverse(df, "profitable_trade_label", "avoid_trade_label")
        assert trivial, (
            "EXPECTED FAILURE BEFORE FIX: avoid_trade_label should be trivial inverse "
            "of profitable_trade_label. If this fails, the label definition may have changed."
        )

        # Also verify the positive rates sum to ~1.0 (except NaN rows)
        profit_rate = float(pd.to_numeric(df["profitable_trade_label"], errors="coerce").mean())
        avoid_rate = float(pd.to_numeric(df["avoid_trade_label"], errors="coerce").mean())
        assert abs((profit_rate + avoid_rate) - 1.0) < 0.01, (
            f"profitable_rate={profit_rate:.4f} + avoid_rate={avoid_rate:.4f} should ~= 1.0"
        )

    def test_choose_labels_skips_trivial_avoid(self):
        """CHOOSE_LABELS must skip avoid_trade_label when AVOID_LABEL_TRIVIAL_INVERSE is True."""
        # Import the module-level flag and the choose_labels function
        from retrain_all_edge_models import AVOID_LABEL_TRIVIAL_INVERSE

        assert AVOID_LABEL_TRIVIAL_INVERSE is True, (
            "AVOID_LABEL_TRIVIAL_INVERSE must be True to skip the trivial inverse avoid label"
        )

        df = load_latest_cost_aware_dataset()
        from retrain_all_edge_models import choose_labels

        usable, skipped = choose_labels(df)

        # avoid_trade_label must NOT be in usable
        assert "avoid_trade_label" not in usable, (
            "EXPECTED FAILURE BEFORE FIX: avoid_trade_label should NOT be in usable labels "
            "because it is a trivial inverse of profitable_trade_label."
        )

        # avoid_trade_label should be in skipped with the correct reason
        avoid_skipped = [s for s in skipped if s.get("label_name") == "avoid_trade_label"]
        assert len(avoid_skipped) == 1, (
            f"Expected exactly one skip record for avoid_trade_label, got {len(avoid_skipped)}"
        )
        reason = avoid_skipped[0]["reason"]
        assert "trivial_inverse" in reason, (
            f"Expected skip reason to contain 'trivial_inverse', got: {reason}"
        )

    def test_two_stage_overlay_detects_trivial_inverse(self):
        """TWO_STAGE_OVERLAY must report avoid_trade_label_is_trivial_inverse=True."""
        df = load_latest_cost_aware_dataset()
        from audit_model_diagnosis import two_stage_overlay

        result = two_stage_overlay(df)

        assert result.get("available") is True, f"Overlay not available: {result.get('reason')}"
        assert "avoid_trade_label_is_trivial_inverse" in result, (
            "Result must contain avoid_trade_label_is_trivial_inverse key"
        )
        assert result["avoid_trade_label_is_trivial_inverse"] is True, (
            "EXPECTED FAILURE BEFORE FIX: avoid_trade_label_is_trivial_inverse should be True "
            "because avoid is exactly 1 - profit."
        )

    def test_primary_labels_excludes_trivial_avoid(self):
        """PRIMARY_LABELS must NOT include avoid_trade_label (it was removed 2026-06-09)."""
        from retrain_all_edge_models import PRIMARY_LABELS

        assert "avoid_trade_label" not in PRIMARY_LABELS, (
            "EXPECTED FAILURE BEFORE FIX: avoid_trade_label should NOT be in PRIMARY_LABELS "
            "because it is a trivial inverse that wastes compute and creates misleading overlay claims."
        )

    def test_no_avoid_label_in_usable_when_trivial_inverse_flag_is_true(self):
        """Integration: with the flag True, avoid_trade_label must never appear in usable labels."""
        from retrain_all_edge_models import (
            AVOID_LABEL_TRIVIAL_INVERSE,
            choose_labels,
        )

        if not AVOID_LABEL_TRIVIAL_INVERSE:
            # Skip this path if the flag is False (real avoid label exists)
            return

        df = load_latest_cost_aware_dataset()
        usable, _ = choose_labels(df)

        # Hard assertion: avoid_trade_label must never be usable when flag is True
        assert "avoid_trade_label" not in usable, (
            "FATAL: avoid_trade_label is in usable labels even though "
            "AVOID_LABEL_TRIVIAL_INVERSE is True. This means the trivial-inverse "
            "check is not working correctly."
        )


class TestAvoidLabelTrivialInverseGuard:
    """Tests verifying that code consuming avoid_trade_label checks the trivial_inverse flag."""

    def test_two_stage_overlay_output_includes_guard_field(self):
        """The overlay report MUST include avoid_trade_label_is_trivial_inverse for consumers."""
        df = load_latest_cost_aware_dataset(limit=10_000)
        from audit_model_diagnosis import two_stage_overlay

        result = two_stage_overlay(df)

        # Ensure the guard field exists
        assert "avoid_trade_label_is_trivial_inverse" in result, (
            "MISSING GUARD FIELD: avoid_trade_label_is_trivial_inverse must be in overlay result"
        )

        # When True, consumers MUST skip the avoid model
        if result["avoid_trade_label_is_trivial_inverse"]:
            # The render function should document the no-op nature
            from audit_model_diagnosis import render_overlay_md
            md_text = render_overlay_md(result)
            assert "trivial inverse" in md_text.lower(), (
                "Rendered overlay markdown must prominently document the trivial inverse finding"
            )

    def test_avoid_refinement_flag_excludes_trivial_avoid(self):
        """
        The --include-avoid-label-refinement flag controls whether avoid is refined.
        With AVOID_LABEL_TRIVIAL_INVERSE=True, even passing the flag should not
        train a useful avoid model (it is skipped by choose_labels).
        """
        from retrain_all_edge_models import (
            AVOID_LABEL_TRIVIAL_INVERSE,
            choose_labels,
        )

        assert AVOID_LABEL_TRIVIAL_INVERSE is True

        df = load_latest_cost_aware_dataset()
        usable, _ = choose_labels(df)

        # The refinement logic in _edge_refinement_planned_candidates only operates
        # on completed models. Since avoid_trade_label is not in PRIMARY_LABELS and
        # not in usable labels, it cannot be refined regardless of the flag.
        assert "avoid_trade_label" not in usable


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))