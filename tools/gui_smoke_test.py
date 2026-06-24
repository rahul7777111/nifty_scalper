import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ui import ScalperUI

results = []
app = None
try:
    app = ScalperUI()
    # 1) allocator preview
    try:
        app._update_allocator_preview()
        results.append(('update_allocator_preview', 'ok'))
    except Exception as e:
        results.append(('update_allocator_preview', f'ERR: {e}'))

    # 2) suggest optimizer exit
    try:
        app._suggest_optimizer_exit()
        results.append(('suggest_optimizer_exit', 'ok'))
    except Exception as e:
        results.append(('suggest_optimizer_exit', f'ERR: {e}'))

    # 3) run backtest comparison
    try:
        app._run_backtest_comparison()
        results.append(('run_backtest_comparison', 'ok'))
    except Exception as e:
        results.append(('run_backtest_comparison', f'ERR: {e}'))

    # 4) run built-in backtest runner
    try:
        app._bt_series_var.set('100,101,102,101,103,104,105')
        app._bt_signals_var.set('0,1,1,0,1,1,1')
        app._bt_slippage_var.set('2.0')
        app._bt_fee_var.set('0.0')
        app._bt_fill_var.set('1.0')
        app._bt_run()
        results.append(('_bt_run', 'ok'))
    except Exception as e:
        results.append(('_bt_run', f'ERR: {e}'))

    # 5) trade builder add/simulate
    try:
        app._tb_strategy_var.set('iron_condor')
        app._tb_underlying_var.set('NIFTY')
        app._tb_leg_type_var.set('CE')
        app._tb_strike_var.set('17850')
        app._tb_side_var.set('SELL')
        app._tb_qty_var.set('1')
        app._tb_price_var.set('1.0')
        app._tb_add_leg()
        results.append(('_tb_add_leg', 'ok'))
    except Exception as e:
        results.append(('_tb_add_leg', f'ERR: {e}'))

    try:
        app._tb_simulate_trade()
        results.append(('_tb_simulate_trade', 'ok'))
    except Exception as e:
        results.append(('_tb_simulate_trade', f'ERR: {e}'))

    # 6) journal refresh
    try:
        app._refresh_trade_journal()
        results.append(('_refresh_trade_journal', 'ok'))
    except Exception as e:
        results.append(('_refresh_trade_journal', f'ERR: {e}'))

    # 7) broker health refresh
    try:
        app._refresh_broker_health()
        results.append(('_refresh_broker_health', 'ok'))
    except Exception as e:
        results.append(('_refresh_broker_health', f'ERR: {e}'))

    # 8) run GPT health check if available
    try:
        if hasattr(app, '_on_gpt_health_check'):
            app._on_gpt_health_check()
            results.append(('_on_gpt_health_check', 'ok'))
        else:
            results.append(('_on_gpt_health_check', 'missing'))
    except Exception as e:
        results.append(('_on_gpt_health_check', f'ERR: {e}'))

finally:
    if app:
        try:
            app.destroy()
        except Exception:
            pass

for k, v in results:
    print(f"{k}: {v}")
