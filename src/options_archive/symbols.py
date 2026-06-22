from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import ArchiveConfig, SymbolSpec


SYMBOL_REGISTRY: Dict[str, SymbolSpec] = {
    "NIFTY": SymbolSpec(
        symbol_key="NIFTY",
        symbol_id=1,
        index_family="NIFTY",
        underlying="NIFTY",
        spot_symbol="NSE:NIFTY",
        chain_underlying="NIFTY",
        lot_size=75,
        expiry_cycle_type="weekly",
    ),
    "BANKNIFTY": SymbolSpec(
        symbol_key="BANKNIFTY",
        symbol_id=2,
        index_family="BANKNIFTY",
        underlying="BANKNIFTY",
        spot_symbol="NSE:BANKNIFTY",
        chain_underlying="BANKNIFTY",
        lot_size=35,
        expiry_cycle_type="weekly",
    ),
    "FINNIFTY": SymbolSpec(
        symbol_key="FINNIFTY",
        symbol_id=3,
        index_family="FINNIFTY",
        underlying="FINNIFTY",
        spot_symbol="NSE:FINNIFTY",
        chain_underlying="FINNIFTY",
        lot_size=65,
        expiry_cycle_type="weekly",
    ),
    "MIDCPNIFTY": SymbolSpec(
        symbol_key="MIDCPNIFTY",
        symbol_id=4,
        index_family="MIDCPNIFTY",
        underlying="MIDCPNIFTY",
        spot_symbol="NSE:MIDCPNIFTY",
        chain_underlying="MIDCPNIFTY",
        lot_size=120,
        expiry_cycle_type="weekly",
    ),
}


def normalize_symbol_key(symbol_key: str) -> str:
    return str(symbol_key or "").strip().upper()


def resolve_symbol_spec(symbol_key: str) -> SymbolSpec:
    key = normalize_symbol_key(symbol_key)
    if not key:
        raise ValueError("symbol_key is required")
    if key not in SYMBOL_REGISTRY:
        raise KeyError(f"Unsupported index symbol: {symbol_key}")
    return SYMBOL_REGISTRY[key]


def archive_config_for_symbol(symbol_key: str, *, archive_root=None, overrides: Optional[Dict[str, object]] = None) -> ArchiveConfig:
    spec = resolve_symbol_spec(symbol_key)
    config = ArchiveConfig(
        symbol_key=spec.symbol_key,
        symbol_id=spec.symbol_id,
        index_family=spec.index_family,
        archive_root=Path(archive_root) if archive_root is not None else Path("archive") / "options",
        underlying=spec.underlying,
        spot_symbol=spec.spot_symbol,
        chain_underlying=spec.chain_underlying,
        lot_size=spec.lot_size,
        expiry_cycle_type=spec.expiry_cycle_type,
    )
    if overrides:
        for name, value in overrides.items():
            setattr(config, name, value)
    return config


def iter_supported_symbols(symbol_keys: Optional[Iterable[str]] = None) -> List[SymbolSpec]:
    if symbol_keys is None:
        return list(SYMBOL_REGISTRY.values())
    out: List[SymbolSpec] = []
    for symbol_key in symbol_keys:
        out.append(resolve_symbol_spec(symbol_key))
    return out