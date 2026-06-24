import sys

with open('src/ui.py', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.splitlines()

start_idx = -1
for i, line in enumerate(lines):
    if 'to_persist["MSTOCK_DELTA_HEDGE_VOL_HIGH_TOL_FACTOR"] = v' in line:
        start_idx = i + 3
        break

end_idx = -1
if start_idx != -1:
    for i in range(start_idx, len(lines)):
        if 'os.environ["MSTOCK_TUNE_MIN_PROFIT_FACTOR"]' in lines[i]:
            end_idx = i + 1
            break

if start_idx == -1 or end_idx == -1:
    print('Could not find start or end index', start_idx, end_idx)
    sys.exit(1)

old_block = lines[start_idx:end_idx]

fixed_block = """
                persist_settings_env(to_persist, to_unset)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Error", f"Failed to apply settings: {exc}")
                return

            # If the bot is running, offer an immediate restart so changes apply.
            try:
                running = bool(self._bot_thread and self._bot_thread.is_alive())
            except Exception:
                running = False

            if running:
                do_restart = messagebox.askyesno(
                    "Settings updated",
                    "Settings saved. Restart bot now to apply changes?",
                )
                if do_restart:
                    self._stop_and_restart_bot()
                    return

            messagebox.showinfo(
                "Settings updated",
                "Settings saved. Stop the bot (if running) and click Start Bot again for changes to take effect.",
            )

        row += 1
        tk.Button(content, text="Apply", command=apply_changes).grid(row=row, column=0, sticky="w", padx=8, pady=(10, 8))
        tk.Button(content, text="Close", command=win.destroy).grid(row=row, column=1, sticky="e", padx=8, pady=(10, 8))

        # Ensure initial scroll region is correct.
        _on_content_configure()
        canvas.yview_moveto(0.0)

    def _set_env_from_fields(self) -> None:
        os.environ["MSTOCK_API_KEY"] = getattr(self, "api_key_var", tk.StringVar()).get().strip()
        os.environ["MSTOCK_USERNAME"] = getattr(self, "username_var", tk.StringVar()).get().strip()
        os.environ["MSTOCK_PASSWORD"] = getattr(self, "password_var", tk.StringVar()).get()
        token = getattr(self, "access_token_var", tk.StringVar()).get().strip()
        if token:
            os.environ["MSTOCK_ACCESS_TOKEN"] = token
"""

for line in old_block:
    if line.strip() == '' or "MSTOCK_ACCESS_TOKEN" in line:
        continue
    stripped = line.strip()
    original_indent = len(line) - len(line.lstrip('\t '))
    new_indent = max(8, original_indent - 8)
    if stripped.startswith('#') and original_indent < 8:
        new_indent = 8
    if original_indent == 8:
        new_indent = 8
    
    fixed_block += (" " * new_indent) + stripped + "\n"

new_lines = lines[:start_idx] + fixed_block.split('\n')[1:-1] + lines[end_idx:]

with open('src/ui.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines) + '\n')

print('Fixed ui.py!')
