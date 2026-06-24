import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ui import ScalperUI


def count(w):
    children = w.winfo_children()
    return len(children) + sum(count(c) for c in children)

app = ScalperUI()
frames = [
    ('Live Dashboard', app.dashboard_frame),
    ('Trade Logs', app.trade_frame),
    ('Signals/Greeks', app.signals_frame),
    ('GPT Advisor', app.gpt_frame),
    ('Settings', app.settings_frame),
    ('Trade Builder', app.builder_frame),
    ('Backtest Runner', app.backtest_frame),
    ('Trade Journal', app.journal_frame),
    ('Optimizer', app.optimizer_frame),
    ('Open Option Legs', app.option_legs_frame),
    ('Managed Positions', app.managed_positions_frame),
]
for n,f in frames:
    print(f"{n}: {count(f)} descendants")
app.destroy()
