import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from types import SimpleNamespace

from mstock_client import MStockTypeBClient


def _client() -> MStockTypeBClient:
    client = MStockTypeBClient(SimpleNamespace(api_key="test"))
    client._load_instruments = lambda: []
    client._get_scripmaster = lambda: None
    return client


def test_parse_typeb_call_put_lists():
    client = _client()
    payload = {
        "contractModel": {
            "sym": "NIFTY",
            "exid": 2,
            "exp": 1429972200,
        },
        "call": [
            "42313,26200,0",
            "84549,26300,0",
        ],
        "put": [
            "42314,26200,0",
            "84550,26300,0",
        ],
        "spot": "NIFTY,26000,NSEEQ",
    }

    chain = client._parse_option_chain_api_data(payload, underlying="NIFTY")
    assert len(chain) == 4
    assert all(isinstance(r, dict) for r in chain)
    assert all("raw" not in r for r in chain)

    ce = [r for r in chain if r.get("option_type") == "CE"]
    pe = [r for r in chain if r.get("option_type") == "PE"]
    assert len(ce) == 2
    assert len(pe) == 2

    first = ce[0]
    assert first.get("token") == "42313"
    assert float(first.get("strike") or 0) == 26200.0
    assert first.get("symbol_root") == "NIFTY"
    assert first.get("exchange") == "NFO"
    assert first.get("expiry") == date(2015, 4, 25)
    assert str(first.get("symbol") or "").startswith("NIFTY")


def test_parse_flat_option_dict_list():
    client = _client()
    payload = [
        {
            "tradingsymbol": "NIFTY20MAY2626200CE",
            "symboltoken": "12345",
            "strikePrice": 26200,
            "optionType": "CE",
            "iv": 0.18,
            "expiry": "2026-05-20",
        },
        {
            "tradingsymbol": "NIFTY20MAY2626200PE",
            "symboltoken": "12346",
            "strikePrice": 26200,
            "optionType": "PE",
            "iv": 0.19,
            "expiry": "2026-05-20",
        },
    ]

    chain = client._parse_option_chain_api_data(payload, underlying="NIFTY")
    assert len(chain) == 2
    assert chain[0]["option_type"] == "CE"
    assert chain[1]["option_type"] == "PE"
    assert chain[0]["symbol"] == "NIFTY20MAY2626200CE"
    assert float(chain[0]["iv"]) == 0.18


def test_parse_strike_bucket_ce_pe_objects():
    client = _client()
    payload = [
        {
            "strike": 26200,
            "CE": {"symboltoken": "111", "ltp": 120.5, "iv": 0.2},
            "PE": {"symboltoken": "222", "ltp": 118.0, "iv": 0.21},
        }
    ]

    chain = client._parse_option_chain_api_data(payload, underlying="NIFTY")
    assert len(chain) == 2
    tokens = {str(r.get("token")) for r in chain}
    assert tokens == {"111", "222"}
    strikes = {float(r.get("strike") or 0) for r in chain}
    assert strikes == {26200.0}


def test_get_scripmaster_falls_back_to_local_instrument_csv(monkeypatch, tmp_path):
    csv_path = tmp_path / "instrument (2).csv"
    csv_path.write_text(
        "Exch,Token,Underlying,Expiry,Strike,OptionType,TradingSymbol,LotSize\n"
        "NFO,12345,NIFTY,16-06-2026,23500,CE,NIFTY16JUN2623500CE,65\n",
        encoding="utf-8",
    )

    client = MStockTypeBClient(SimpleNamespace(api_key="test"))
    client._load_instruments = lambda: []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MSTOCK_SCRIPMASTER_PATH", str(tmp_path / "missing.csv"))

    sm = client._get_scripmaster()

    assert sm is not None
    assert Path(sm.csv_path) == csv_path
