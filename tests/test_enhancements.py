import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from strategy import NiftyScalper
from config import load_strategy_config
from market_data import Candle
from datetime import datetime


def test_diagnostics_and_ml_simulate():
    cfg = load_strategy_config()
    sc = NiftyScalper(None, cfg)
    # run diagnostics
    diag = sc.run_enhancements_diagnostics()
    assert 'volatility_forecast' in diag
    assert 'regime' in diag

    # synth candles and evaluate
    candles = [Candle(datetime.now(), 100, 101, 99, 100 + i) for i in range(60)]
    res = sc.evaluate_entry_signals(candles)
    assert isinstance(res, dict)
    # simulate placement
    tr = sc._simulate_or_place_basic_directional(take=res.get('take', False), size=res.get('size', 0), reason=res.get('reason','test'), candles=candles)
    # if take was True, trade should be appended
    if res.get('take'):
        assert tr is not None
        assert len(sc.state.open_directional) >= 1
    else:
        assert tr is None
