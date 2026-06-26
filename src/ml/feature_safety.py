"""Feature-safety helpers for leakage-aware ML training and inference."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

try:
    from src.ml_feature_contract import FORBIDDEN_EXACT_TOKENS, is_feature_forbidden
except Exception:  # pragma: no cover - script import path fallback
    try:
        from ml_feature_contract import FORBIDDEN_EXACT_TOKENS, is_feature_forbidden  # type: ignore
    except Exception:  # pragma: no cover - minimal fallback
        FORBIDDEN_EXACT_TOKENS = tuple()

        def is_feature_forbidden(name: object) -> bool:
            return False

# Exact leakage / target columns known to exist in historical datasets.
LEAKAGE_EXACT_COLUMNS: Tuple[str, ...] = (
    "future_close",
    "gross_forward_return",
    "net_forward_return",
    "expected_return_after_cost",
    "return_to_cost_ratio",
    "cost_return_units_estimated",
    "cost_stress_1_0x_return",
    "cost_stress_1_5x_return",
    "cost_stress_2_0x_return",
    "cost_stress_1_0x_label",
    "cost_stress_1_5x_label",
    "cost_stress_2_0x_label",
    "net_return_after_cost",
)

# Dataset / artifact metadata that should not be treated as model features.
NON_FEATURE_COLUMNS: Tuple[str, ...] = (
    "prediction_id",
    "timestamp",
    "datetime",
    "date",
    "trading_date",
    "trading_day",
    "trading_day_spot",
    "session_bucket",
    "selected_option_symbol",
    "selected_option_token",
    "instrument_key",
    "trading_symbol",
    "source_file",
    "spot_source",
    "observer_signal_id",
    "observer_run_id",
    "expiry",
    "option_type",
    "option_side_normalized",
    "greeks_source",
    "bs_iv_source",
    "row_enrichment_error",
    "raw_model_outputs_json",
    "feature_snapshot_json",
    "regime_features_json",
    "drift_sensitive_features_json",
    "data_quality_flags_json",
    "missing_fields_json",
    "observer_error",
    "entry_rule_that_fired",
    "signal_reason",
    "signal_side",
)

TARGET_NAME_SUFFIXES: Tuple[str, ...] = (
    "_label",
    "_label_v2",
    "_target",
    "_outcome",
)

TARGET_NAME_PARTS: Tuple[str, ...] = (
    "profitable_trade_label",
    "avoid_trade_label",
    "strong_profitable_trade_label",
    "high_conviction_trade_label",
    "weak_trade_label",
    "no_trade_label",
    "cost_survivor_label",
    "paper_candidate_label",
)

_LOWER_NON_FEATURE_COLUMNS = {item.lower() for item in NON_FEATURE_COLUMNS}
_LOWER_LEAKAGE_EXACT_COLUMNS = {item.lower() for item in LEAKAGE_EXACT_COLUMNS}
_LOWER_FORBIDDEN_EXACT_TOKENS = {item.lower() for item in FORBIDDEN_EXACT_TOKENS}
_LOWER_TARGET_NAME_PARTS = {item.lower() for item in TARGET_NAME_PARTS}


def get_default_leakage_columns(target: str) -> List[str]:
    columns = [
        "future_close",
        "gross_forward_return",
        "net_forward_return",
    ]
    if str(target or "").strip().lower() == "profitable_trade_label":
        columns.append("avoid_trade_label")
    return columns


def get_default_non_feature_columns() -> List[str]:
    preferred = [
        "timestamp",
        "expiry",
        "instrument_key",
        "trading_symbol",
        "source_file",
        "trading_day",
        "trading_day_spot",
        "spot_source",
        "bs_iv_source",
        "option_side_normalized",
        "greeks_source",
        "row_enrichment_error",
    ]
    for column in NON_FEATURE_COLUMNS:
        if column not in preferred:
            preferred.append(column)
    return preferred


def _normalize_name(name: object) -> str:
    return str(name or "").strip()


def is_target_column(name: object) -> bool:
    column = _normalize_name(name)
    lowered = column.lower()
    if not lowered:
        return False
    if lowered in _LOWER_TARGET_NAME_PARTS:
        return True
    return lowered.endswith(TARGET_NAME_SUFFIXES)


def is_non_feature_column(name: object) -> bool:
    column = _normalize_name(name)
    lowered = column.lower()
    if not lowered:
        return False
    if lowered in _LOWER_NON_FEATURE_COLUMNS:
        return True
    if lowered in _LOWER_LEAKAGE_EXACT_COLUMNS:
        return True
    if lowered in _LOWER_FORBIDDEN_EXACT_TOKENS:
        return True
    return is_target_column(column) or bool(is_feature_forbidden(column))


def leakage_columns(columns: Iterable[object]) -> List[str]:
    return [str(col) for col in columns if is_non_feature_column(col)]


def safe_feature_columns(
    columns: Iterable[object],
    *,
    target_columns: Optional[Sequence[str]] = None,
    extra_drop: Optional[Sequence[str]] = None,
) -> List[str]:
    target_set = {_normalize_name(col).lower() for col in (target_columns or ()) if _normalize_name(col)}
    extra_drop_set = {_normalize_name(col).lower() for col in (extra_drop or ()) if _normalize_name(col)}
    safe: List[str] = []
    for raw_name in columns:
        name = _normalize_name(raw_name)
        lowered = name.lower()
        if not name:
            continue
        if lowered in target_set or lowered in extra_drop_set:
            continue
        if is_non_feature_column(name):
            continue
        safe.append(name)
    return safe


def bool_to_int_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if is_bool_dtype(out[column]):
            out[column] = out[column].astype("int8")
    return out


def build_safe_numeric_bool_frame(
    df: pd.DataFrame,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    target_columns: Optional[Sequence[str]] = None,
    extra_drop: Optional[Sequence[str]] = None,
    coerce_numeric: bool = True,
    bool_to_int: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    selected = list(feature_columns) if feature_columns is not None else safe_feature_columns(
        df.columns,
        target_columns=target_columns,
        extra_drop=extra_drop,
    )
    frame = df.loc[:, [col for col in selected if col in df.columns]].copy()
    dropped_missing = [col for col in selected if col not in df.columns]

    dropped_non_numeric: List[str] = []
    if bool_to_int:
        frame = bool_to_int_frame(frame)

    keep_cols: List[str] = []
    for column in frame.columns:
        series = frame[column]
        if is_numeric_dtype(series) or is_bool_dtype(series):
            keep_cols.append(column)
            continue
        if coerce_numeric:
            coerced = pd.to_numeric(series, errors="coerce")
            if not coerced.isna().all():
                frame[column] = coerced
                keep_cols.append(column)
                continue
        dropped_non_numeric.append(column)

    frame = frame.loc[:, keep_cols]
    if bool_to_int:
        frame = bool_to_int_frame(frame)

    diagnostics = {
        "selected_columns": list(selected),
        "kept_columns": list(frame.columns),
        "dropped_missing_columns": dropped_missing,
        "dropped_non_numeric_columns": dropped_non_numeric,
        "forbidden_or_non_feature_columns": leakage_columns(df.columns),
    }
    return frame, diagnostics


def build_safe_feature_frame(
    df: pd.DataFrame,
    target: str,
    live_computable_only: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    del live_computable_only
    extra_drop = list(get_default_non_feature_columns()) + list(get_default_leakage_columns(target))
    return build_safe_numeric_bool_frame(
        df,
        target_columns=[target],
        extra_drop=extra_drop,
    )


def drop_target_columns(
    df: pd.DataFrame,
    *,
    target_columns: Optional[Sequence[str]] = None,
    extra_drop: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    safe_cols = safe_feature_columns(df.columns, target_columns=target_columns, extra_drop=extra_drop)
    return df.loc[:, safe_cols].copy()


def filter_target_rows(
    df: pd.DataFrame,
    target_column: str,
    *,
    allowed_values: Optional[Sequence[Any]] = None,
    dropna: bool = True,
) -> pd.DataFrame:
    if target_column not in df.columns:
        raise KeyError(f"target column not found: {target_column}")

    series = df[target_column]
    mask = pd.Series(True, index=df.index)
    if dropna:
        mask &= series.notna()

    numeric = pd.to_numeric(series, errors="coerce")
    mask &= numeric.notna()
    finite_mask = numeric.map(lambda value: bool(value is not None and not math.isnan(value) and math.isfinite(value)))
    mask &= finite_mask.fillna(False)

    if allowed_values is not None:
        mask &= series.isin(list(allowed_values))
    return df.loc[mask].copy()


def extract_binary_target(
    df: pd.DataFrame,
    target_column: str,
    *,
    allowed_values: Sequence[Any] = (0, 1),
) -> pd.Series:
    filtered = filter_target_rows(df[[target_column]], target_column, allowed_values=allowed_values)
    return pd.to_numeric(filtered[target_column], errors="raise").astype(int)


def compute_train_medians(frame: pd.DataFrame) -> Dict[str, float]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    medians = frame.median(axis=0, skipna=True, numeric_only=True)
    out: Dict[str, float] = {}
    for column, value in medians.items():
        try:
            numeric = float(value)
        except Exception:
            continue
        if math.isnan(numeric) or not math.isfinite(numeric):
            continue
        out[str(column)] = numeric
    return out


def apply_train_median_fill(
    frame: pd.DataFrame,
    train_medians: Mapping[str, Any],
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    out = frame if inplace else frame.copy()
    fill_map: Dict[str, float] = {}
    for column in out.columns:
        if column not in train_medians:
            continue
        try:
            fill_value = float(train_medians[column])
        except Exception:
            continue
        if math.isnan(fill_value) or not math.isfinite(fill_value):
            continue
        fill_map[column] = fill_value
    if fill_map:
        out = out.fillna(value=fill_map)
    return out


def prepare_training_matrices(
    df: pd.DataFrame,
    target_column: str,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    extra_drop: Optional[Sequence[str]] = None,
    allowed_target_values: Sequence[Any] = (0, 1),
    apply_median_fill: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    filtered = filter_target_rows(df, target_column, allowed_values=allowed_target_values)
    X, diagnostics = build_safe_numeric_bool_frame(
        filtered,
        feature_columns=feature_columns,
        target_columns=[target_column],
        extra_drop=extra_drop,
    )
    y = pd.to_numeric(filtered[target_column], errors="raise").astype(int)
    medians = compute_train_medians(X)
    if apply_median_fill:
        X = apply_train_median_fill(X, medians)
    diagnostics = dict(diagnostics)
    diagnostics["target_column"] = target_column
    diagnostics["train_medians"] = medians
    diagnostics["row_count"] = int(len(filtered))
    return X, y, diagnostics
