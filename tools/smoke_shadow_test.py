from __future__ import annotations
import sqlite3, time, os
from datetime import datetime

# Ensure current workspace is importable
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from config import StrategyConfig
from strategy import NiftyScalper
from db import DatabaseManager
from ml_signals import MLModelBundle

# Minimal mock client returning simple candles
class SimpleCandle:
    def __init__(self, o,h,l,c,ts=None):
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.ts = ts or time.time()
        self.volume = 100.0
        self.vwap = c

class DummyClient:
    def get_candles(self, symbol, timeframe, limit=200):
        # generate simple synthetic candles
        now = time.time()
        candles = []
        for i in range(200):
            c = 10000.0 + (i % 10) * 0.1
            candles.append(SimpleCandle(c-0.05, c+0.05, c-0.1, c, now - (200-i)*60))
        return candles

# Prepare DB and clear predictions
db = DatabaseManager()
with sqlite3.connect(db.db_path) as conn:
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions')
    cursor.execute('DELETE FROM trades')
    conn.commit()

# Baseline metrics
before_metrics = db.get_expectancy_metrics()
print('before shadow_predictions=', before_metrics.get('shadow_predictions'))

# No sanity insert for this run; DB is clean

# Create strategy with dummy client and config
cfg = StrategyConfig()
cfg.max_trades_per_day = 0
cfg.underlying = 'NIFTY'
cfg.timeframe = '1m'
cfg.atr_period = 14
cfg.ema_fast = 5
cfg.ema_slow = 13

client = DummyClient()
strategy = NiftyScalper(client, cfg)

# Attach a simple mock ML model that returns high probability
class MockModel:
    def predict_proba(self, X):
        # return [prob0, prob1]
        return [[0.2, 0.9] for _ in X]

mock_bundle = MLModelBundle(model=MockModel(), feature_names=[], metrics={}, trained_at=datetime.utcnow().isoformat()+'Z')
strategy.ml_model = mock_bundle

# Trigger decide entries which should be blocked by max_trades_per_day and record prediction
strategy._decide_entries()

# (Do NOT invoke direct helper; rely on _decide_entries() flow)

# Read last prediction row
with sqlite3.connect(db.db_path) as conn:
    cursor = conn.cursor()
    cursor.execute('SELECT ts, prediction, model_checksum, reason FROM predictions ORDER BY id DESC LIMIT 1')
    row = cursor.fetchone()
    if row:
        ts, prob, checksum, reason = row
        print('prediction row: ts=', datetime.fromtimestamp(ts).isoformat(), 'prob=', prob, 'checksum=', checksum, 'reason=', reason)
    else:
        print('No prediction row found')

# Verify no trades placed
with sqlite3.connect(db.db_path) as conn:
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(1) FROM trades')
    trades_count = cursor.fetchone()[0]
    print('trades_count=', trades_count)

# Verify dashboard metric
after_metrics = db.get_expectancy_metrics()
print('after shadow_predictions=', after_metrics.get('shadow_predictions'))

# Exit status
if row and trades_count == 0 and after_metrics.get('shadow_predictions',0) == before_metrics.get('shadow_predictions',0)+1:
    print('SMOKE TEST PASSED')
    exit(0)
else:
    print('SMOKE TEST FAILED')
    exit(2)
