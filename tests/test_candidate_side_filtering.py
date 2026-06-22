#!/usr/bin/env python3
"""
tests/test_candidate_side_filtering.py

PHASE 3 audit test: proves side policy filters (PE_ONLY, CE_ONLY, BOTH, AUTO) behave correctly
and do not accidentally drop all rows for the cost-aware dataset.
Uses live-computable option_type column only.
"""

import pandas as pd
import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from candidate_filters import (
    apply_pe_only_filter,
    apply_ce_only_filter,
)
import retrain_dynamic_candidate_matrix as matrix_mod  # for _filter_df_for_side


DATASET_CANDIDATE = REPO_ROOT / "data/processed/nifty_option_chain_cost_aware_edge_dataset_20260610_121730.csv"


def _load_sample(n=20000):
    if not DATASET_CANDIDATE.exists():
        pytest.skip(f"dataset not present: {DATASET_CANDIDATE}")
    df = pd.read_csv(DATASET_CANDIDATE, nrows=n, low_memory=False)
    # ensure option_type normalized
    if "option_type" in df.columns:
        df["option_type"] = df["option_type"].astype(str).str.upper().str.strip()
    return df


def test_dataset_has_balanced_ce_pe():
    df = _load_sample(50000)
    assert "option_type" in df.columns
    vc = df["option_type"].value_counts().to_dict()
    assert "PE" in vc and "CE" in vc
    assert vc["PE"] > 1000 and vc["CE"] > 1000, "dataset must have substantial PE and CE rows"


def test_pe_only_filter_keeps_only_pe_and_nonzero():
    df = _load_sample(30000)
    pe_df = matrix_mod._filter_df_for_side(df, "PE_ONLY")
    assert len(pe_df) > 0, "PE_ONLY on real dataset sample must keep >0 rows"
    assert (pe_df["option_type"] == "PE").all(), "PE_ONLY must contain only PE"
    assert "CE" not in pe_df["option_type"].values, "PE_ONLY must exclude CE"


def test_ce_only_filter_keeps_only_ce_and_nonzero():
    df = _load_sample(30000)
    ce_df = matrix_mod._filter_df_for_side(df, "CE_ONLY")
    assert len(ce_df) > 0, "CE_ONLY on real dataset sample must keep >0 rows"
    assert (ce_df["option_type"] == "CE").all(), "CE_ONLY must contain only CE"
    assert "PE" not in ce_df["option_type"].values, "CE_ONLY must exclude PE"


def test_both_keeps_ce_and_pe():
    df = _load_sample(20000)
    both = matrix_mod._filter_df_for_side(df, "BOTH")
    assert len(both) == len(df) or len(both) > 1000
    has_pe = (both["option_type"] == "PE").any()
    has_ce = (both["option_type"] == "CE").any()
    assert has_pe and has_ce, "BOTH must be able to see both CE and PE rows"


def test_auto_directional_does_not_remove_all_rows():
    df = _load_sample(20000)
    auto = matrix_mod._filter_df_for_side(df, "AUTO_DIRECTIONAL")
    # AUTO does not side-filter the df; it passes full to router which splits internally
    assert len(auto) > 0
    # ensure option_type still has both (router will decide per row)
    vc = auto["option_type"].value_counts().to_dict()
    assert vc.get("PE", 0) > 0 and vc.get("CE", 0) > 0


def test_src_candidate_filters_pe_only_runtime():
    snap_pe = {"option_type": "PE", "ltp": 80.0}
    snap_ce = {"option_type": "CE", "ltp": 80.0}
    r_pe = apply_pe_only_filter(snap_pe)
    r_ce = apply_pe_only_filter(snap_ce)
    assert r_pe.get("filter_passed") is True or r_pe.get("passed") is True
    assert r_ce.get("filter_passed") is False or r_ce.get("passed") is False


def test_src_candidate_filters_ce_only_runtime():
    snap_pe = {"option_type": "PE", "ltp": 80.0}
    snap_ce = {"option_type": "CE", "ltp": 80.0}
    r_pe = apply_ce_only_filter(snap_pe)
    r_ce = apply_ce_only_filter(snap_ce)
    assert r_ce.get("filter_passed") is True or r_ce.get("passed") is True
    assert r_pe.get("filter_passed") is False or r_pe.get("passed") is False


def test_onehot_columns_not_inverted():
    df = _load_sample(10000)
    if "option_type_ce" in df.columns and "option_type_pe" in df.columns:
        # ce flag should be 1 precisely when option_type==CE
        ce_rows = df[df["option_type"] == "CE"]
        pe_rows = df[df["option_type"] == "PE"]
        if len(ce_rows) > 0:
            assert (ce_rows["option_type_ce"] == 1).all()
            assert (ce_rows["option_type_pe"] == 0).all()
        if len(pe_rows) > 0:
            assert (pe_rows["option_type_pe"] == 1).all()
            assert (pe_rows["option_type_ce"] == 0).all()
