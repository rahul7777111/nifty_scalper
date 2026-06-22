from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import StrategyConfig
from strategy import NiftyScalper


class DhanClient:
    pass


class OtherClient:
    pass


def test_prefer_synthetic_warmup_for_dhan(monkeypatch) -> None:
    bot = NiftyScalper.__new__(NiftyScalper)
    bot.client = DhanClient()
    bot._bool_env = lambda name, default=False: False  # type: ignore[assignment]
    monkeypatch.delenv("MSTOCK_INTRADAY_ONLY", raising=False)
    assert NiftyScalper._prefer_synthetic_warmup(bot) is True


def test_prefer_synthetic_warmup_respects_intraday_mode_for_others(monkeypatch) -> None:
    bot = NiftyScalper.__new__(NiftyScalper)
    bot.client = OtherClient()
    bot._bool_env = lambda name, default=False: name == "MSTOCK_USE_INTRADAY_CHART"  # type: ignore[assignment]
    monkeypatch.delenv("MSTOCK_INTRADAY_ONLY", raising=False)
    assert NiftyScalper._prefer_synthetic_warmup(bot) is True
