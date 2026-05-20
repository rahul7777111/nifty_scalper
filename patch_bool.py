import sys
content = open('src/ui.py', encoding='utf-8').read()

old_save = '''                to_env_bool(delta_enabled_var, "MSTOCK_ENABLE_DELTA_STRIKE_SELECTION")
                to_env_bool(theta_enabled_var, "MSTOCK_ENABLE_THETA_DECAY_FILTER")
                to_env_bool(chop_hard_var, "MSTOCK_CHOPPINESS_HARD_FILTER")
                to_env_bool(rsi_conf_var, "MSTOCK_ENABLE_RSI_CONFLUENCE")'''

new_save = '''                os.environ["MSTOCK_ENABLE_DELTA_STRIKE_SELECTION"] = "1" if delta_enabled_var.get() else ""
                to_persist["MSTOCK_ENABLE_DELTA_STRIKE_SELECTION"] = os.environ["MSTOCK_ENABLE_DELTA_STRIKE_SELECTION"]
                os.environ["MSTOCK_ENABLE_THETA_DECAY_FILTER"] = "1" if theta_enabled_var.get() else ""
                to_persist["MSTOCK_ENABLE_THETA_DECAY_FILTER"] = os.environ["MSTOCK_ENABLE_THETA_DECAY_FILTER"]
                os.environ["MSTOCK_CHOPPINESS_HARD_FILTER"] = "1" if chop_hard_var.get() else ""
                to_persist["MSTOCK_CHOPPINESS_HARD_FILTER"] = os.environ["MSTOCK_CHOPPINESS_HARD_FILTER"]
                os.environ["MSTOCK_ENABLE_RSI_CONFLUENCE"] = "1" if rsi_conf_var.get() else ""
                to_persist["MSTOCK_ENABLE_RSI_CONFLUENCE"] = os.environ["MSTOCK_ENABLE_RSI_CONFLUENCE"]'''

if old_save in content:
    content = content.replace(old_save, new_save)
    open('src/ui.py', 'w', encoding='utf-8').write(content)
    print('UI saving replaced successfully.')
else:
    print('Anchor not found.')
