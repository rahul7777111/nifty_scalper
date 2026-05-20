import sys
content = open('src/ui.py', encoding='utf-8').read()
new_save = '''
                if dir_sl_var.get().strip():
                    v = str(float(dir_sl_var.get().strip()))
                    os.environ["MSTOCK_DIR_SL_ATR_MULT"] = v
                    to_persist["MSTOCK_DIR_SL_ATR_MULT"] = v
                if dir_tp_var.get().strip():
                    v = str(float(dir_tp_var.get().strip()))
                    os.environ["MSTOCK_DIR_TP_ATR_MULT"] = v
                    to_persist["MSTOCK_DIR_TP_ATR_MULT"] = v
                if delta_target_var.get().strip():
                    v = str(float(delta_target_var.get().strip()))
                    os.environ["MSTOCK_TARGET_DELTA"] = v
                    to_persist["MSTOCK_TARGET_DELTA"] = v
                to_env_bool(delta_enabled_var, "MSTOCK_ENABLE_DELTA_STRIKE_SELECTION")
                to_env_bool(theta_enabled_var, "MSTOCK_ENABLE_THETA_DECAY_FILTER")
                to_env_bool(chop_hard_var, "MSTOCK_CHOPPINESS_HARD_FILTER")
                to_env_bool(rsi_conf_var, "MSTOCK_ENABLE_RSI_CONFLUENCE")
'''
anchor = 'if dir_be_var.get().strip():'
if anchor in content:
    content = content.replace(anchor, new_save + '\n                ' + anchor)
    open('src/ui.py', 'w', encoding='utf-8').write(content)
    print('UI save added successfully.')
else:
    print('Anchor not found.')
