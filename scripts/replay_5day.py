"""Simple 5-day replay harness to count bars, feature vectors, predictions, filters, and logs.

Usage: run inside the repo virtualenv: `python scripts/replay_5day.py`
"""
import os, sys, json, heapq
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from datetime import datetime
from market_data import Candle
from ml_signals import feature_vector_from_candles, predict

DATA_DIR = os.path.join(os.getcwd(), 'data')

# find latest 5 candle files by filename date
files = []
for fn in os.listdir(DATA_DIR):
    if fn.startswith('candles_') and fn.endswith('.json'):
        files.append(os.path.join(DATA_DIR, fn))
files = sorted(files)
files = files[-5:]

# load model if available
MODEL_PATH = os.path.join(os.getcwd(), 'ml_signal_model.pkl')
model = None
try:
    import joblib
    sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"[REPLAY] joblib.load failed: {e}")
        model = None
except Exception:
    try:
        import pickle
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except Exception:
        import traceback
        print("[REPLAY] pickle.load failed:\n" + traceback.format_exc())
        model = None

print(f"[REPLAY] model_loaded={model is not None} type={type(model).__name__ if model is not None else None}")
try:
    print(f"[REPLAY] model_training_samples={(getattr(model,'metrics',{}) or {}).get('training_samples') if model is not None else None}")
except Exception:
    pass

threshold = float(os.getenv('REPLAY_ML_THRESHOLD', '0.55'))

bars = 0
features = 0
preds = 0
filtered = 0
logged = 0

rolling = []
for f in files:
    with open(f, 'r', encoding='utf-8') as fh:
        doc = json.load(fh)
    c_list = doc.get('candles') or []
    for c in c_list:
        try:
            t = datetime.fromisoformat(c['time'])
        except Exception:
            t = datetime.strptime(c['time'], '%Y-%m-%d %H:%M:%S')
        candle = Candle(time=t, open=float(c['open']), high=float(c['high']), low=float(c['low']), close=float(c['close']), volume=float(c.get('volume') or 0.0))
        rolling.append(candle)
        bars += 1
        # attempt feature vector every bar with lookback 20
        # Provide a minimal context so feature generation doesn't early-return.
        ctx = {
            'iv': 15.0,
            'pcr_oi': 1.0,
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
            'spot': float(candle.close),
            'option_price': float(candle.close) * 0.01,
        }
        feat_vec, feat_names = feature_vector_from_candles(rolling, context=ctx, lookback=20)
        if feat_vec:
            features += 1
            if model is not None:
                prob = predict(model, [feat_vec])[0]
                preds += 1
                if prob < threshold:
                    filtered += 1
                else:
                    # would be logged in shadow mode; simulate logging
                    logged += 1

print(json.dumps({
    'bars_processed': bars,
    'features_generated': features,
    'predictions_generated': preds,
    'predictions_filtered': filtered,
    'predictions_logged': logged,
}, indent=2))
