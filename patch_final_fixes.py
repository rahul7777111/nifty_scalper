"""Apply final fixes to enhancments."""

import re


def fix_config():
    with open("src/config.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Fix 1: Revert entry_require_bid_ask to False
    old = "entry_require_bid_ask: bool = True"
    new = "entry_require_bid_ask: bool = False"
    if old in content:
        content = content.replace(old, new)
        print("FIX 1: entry_require_bid_ask reverted to False")
    else:
        print("FIX 1: entry_require_bid_ask already set correctly or not found")

    with open("src/config.py", "w", encoding="utf-8") as f:
        f.write(content)


def fix_strategy():
    with open("src/strategy.py", "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Fix 2: Integrate _session_exit_adjustments into _manage_open_trades
    # Find the section after `if post_trail > 0:` / `trail_mult_local = post_trail`
    # The pattern is: after the post-partial exit block, insert session adjustment call

    # First, find where the post-partial block ends
    # We expect: the line with "trail_mult_local = post_trail" (already present)
    # Then the session adjustment code to be inserted
    
    insert_idx = None
    for i, line in enumerate(lines):
        # Look for the post-partial trail assignment, after which we insert session adjustments
        if 'trail_mult_local = post_trail' in line:
            insert_idx = i + 1
            break

    if insert_idx is None:
        print("ERROR: Could not find post-partial trail assignment point")
        return

    # Check if session adjustment is already integrated
    already_integrated = any(
        'session_exit_adjustments' in line and 'be_mult_local' in line
        for line in lines[insert_idx:insert_idx+30]
    )
    if already_integrated:
        print("FIX 2: Session exit adjustments already integrated in _manage_open_trades")
    else:
        # Insert the session adjustment code
        session_code = [
            "            # Session-specific exit adjustments (overrides)\n",
            "            if bool(getattr(self.cfg, \"session_exit_enabled\", False)):\n",
            "                try:\n",
            "                    session_adj = self._session_exit_adjustments()\n",
            "                    if isinstance(session_adj, dict):\n",
            "                        sa_sl = float(session_adj.get(\"sl_mult\", 1.0) or 1.0)\n",
            "                        sa_tp = float(session_adj.get(\"tp_mult\", 1.0) or 1.0)\n",
            "                        sa_trail = float(session_adj.get(\"trail_mult\", 1.0) or 1.0)\n",
            "                        be_mult_local *= sa_sl\n",
            "                        trail_mult_local *= sa_trail\n",
            "                except Exception:\n",
            "                    pass\n",
        ]
        lines[insert_idx:insert_idx] = session_code
        print(f"FIX 2: Session exit adjustments integrated at line {insert_idx + 1}")

    with open("src/strategy.py", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("strategy.py saved.")


fix_config()
fix_strategy()
print("\nAll fixes applied.")
