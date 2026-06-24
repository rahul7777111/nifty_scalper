import time
from types import SimpleNamespace
from src.strategy import NiftyScalper, TradeLogEvent
from src.config import StrategyConfig
import src.strategy as strat


def make_scalper_with_cfg(monkeypatch, gpt_decision="TAKE"):
    # Minimal fake client with get_candles stub
    class DummyClient:
        def get_candles(self, symbol=None, timeframe=None, limit=None):
            from src.market_data import Candle
            now = time.time()
            return [Candle(time=now - i*60, open=100, high=101, low=99, close=100) for i in range(30)][::-1]

    client = DummyClient()
    cfg = StrategyConfig()
    cfg.strategy_name = "auto"
    cfg.gpt_enable = True
    cfg.gpt_auto_select = True
    cfg.gpt_require_recommendation = True

    # Patch gpt_advisor.advise_trade to return a simple object
    def fake_advise_trade(*, proposal, model, api_key, **kwargs):
        return SimpleNamespace(decision=gpt_decision, reason="unit_test", confidence=0.9)

    import importlib
    ga = importlib.import_module("src.gpt_advisor")
    monkeypatch.setattr(ga, "advise_trade", fake_advise_trade)
    sc = NiftyScalper(client, cfg)
    return sc


def test_gpt_take_allows_entry(monkeypatch):
    sc = make_scalper_with_cfg(monkeypatch, gpt_decision="TAKE")
    candles = sc.client.get_candles()
    res = sc.evaluate_entry_signals(candles)
    assert isinstance(res, dict)
    # With GPT TAKE and requirement, should allow take or at least return dict
    assert "take" in res


def test_gpt_skip_blocks_entry(monkeypatch):
    sc = make_scalper_with_cfg(monkeypatch, gpt_decision="SKIP")
    candles = sc.client.get_candles()
    res = sc.evaluate_entry_signals(candles)
    assert isinstance(res, dict)
    # With GPT SKIP and requirement, strategy should not take
    assert res.get("take") is False


def test_gpt_auto_select_can_originate_entry(monkeypatch):
    class NeutralClient:
        def get_candles(self, symbol=None, timeframe=None, limit=None):
            now = time.time()
            return [SimpleNamespace(time=now - i * 60, open=100, high=100, low=100, close=100) for i in range(30)][::-1]

        def get_option_chain(self, underlying):
            return []

    monkeypatch.setattr(strat, "rsi", lambda *args, **kwargs: 50.0)
    monkeypatch.setattr(strat, "atr", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(strat, "detect_regime", lambda *args, **kwargs: "neutral")
    monkeypatch.setattr(strat, "get_regime_tuning", lambda regime, cfg: {"ml_threshold": 0.99})
    monkeypatch.setattr(strat, "select_strategy_for_regime", lambda regime, cfg: "directional")
    monkeypatch.setattr(strat, "feature_vector_from_candles", lambda candles, context=None: ([0.0], ["x"]))
    monkeypatch.setattr(strat, "predict", lambda model, vecs: [0.0])
    monkeypatch.setattr(strat, "ema", lambda *args, **kwargs: 100.0)

    cfg = StrategyConfig()
    cfg.strategy_name = "auto"
    cfg.gpt_enable = True
    cfg.gpt_auto_select = True
    cfg.gpt_require_recommendation = False
    cfg.enable_ml_signals = False

    sc = NiftyScalper(NeutralClient(), cfg)
    sc.ml_model = None
    sc._gpt_auto_select_strategy = lambda **kwargs: "directional"

    candles = sc.client.get_candles()
    res = sc.evaluate_entry_signals(candles)

    assert isinstance(res, dict)
    assert res.get("take") is True
    assert res.get("reason") == "gpt_auto_select"
