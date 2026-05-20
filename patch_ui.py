import sys
content = open('src/ui.py', encoding='utf-8').read()
new_ui_fields = '''
        row += 1
        tk.Label(content, text='Initial SL / TP (ATR Mult)').grid(row=row, column=0, sticky=\"w\", padx=8)
        sltp_frame = tk.Frame(content)
        sltp_frame.grid(row=row, column=1, sticky=\"w\", padx=8)
        dir_sl_var = tk.StringVar(value=str(getattr(cfg, \"dir_sl_atr_mult\", 1.5)))
        dir_tp_var = tk.StringVar(value=str(getattr(cfg, \"dir_tp_atr_mult\", 3.0)))
        tk.Entry(sltp_frame, textvariable=dir_sl_var, width=10).pack(side=tk.LEFT)
        tk.Entry(sltp_frame, textvariable=dir_tp_var, width=10).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text='Delta Strikes / Theta Filter').grid(row=row, column=0, sticky=\"w\", padx=8)
        delta_frame = tk.Frame(content)
        delta_frame.grid(row=row, column=1, sticky=\"w\", padx=8)
        delta_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, \"enable_delta_strike_selection\", False)))
        delta_target_var = tk.StringVar(value=str(getattr(cfg, \"target_delta\", 0.40)))
        theta_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, \"enable_theta_decay_filter\", False)))
        ttk.Checkbutton(delta_frame, text=\"Delta\", variable=delta_enabled_var).pack(side=tk.LEFT)
        tk.Entry(delta_frame, textvariable=delta_target_var, width=6).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Checkbutton(delta_frame, text=\"Theta\", variable=theta_enabled_var).pack(side=tk.LEFT)

        row += 1
        tk.Label(content, text='Hard CHOP / RSI Confluence').grid(row=row, column=0, sticky=\"w\", padx=8)
        filter_frame = tk.Frame(content)
        filter_frame.grid(row=row, column=1, sticky=\"w\", padx=8)
        chop_hard_var = tk.BooleanVar(value=bool(getattr(cfg, \"choppiness_hard_filter\", False)))
        rsi_conf_var = tk.BooleanVar(value=bool(getattr(cfg, \"enable_rsi_confluence\", False)))
        ttk.Checkbutton(filter_frame, text=\"Hard Chop\", variable=chop_hard_var).pack(side=tk.LEFT)
        ttk.Checkbutton(filter_frame, text=\"RSI Conf\", variable=rsi_conf_var).pack(side=tk.LEFT, padx=(6, 0))
'''
anchor = 'dir_be_var = tk.StringVar(value=str(cfg.dir_breakeven_atr_mult))'
if anchor in content:
    content = content.replace(anchor, new_ui_fields + '\n        ' + anchor)
    open('src/ui.py', 'w', encoding='utf-8').write(content)
    print('UI fields added successfully.')
else:
    print('Anchor not found.')
