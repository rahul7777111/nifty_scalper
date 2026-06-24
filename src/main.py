from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env file explicitly so environment variables are available
# regardless of VS Code terminal settings
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"[main] Loaded .env from {env_path}")
    else:
        print(f"[main] No .env file found at {env_path}")
except ImportError:
    print("[main] python-dotenv not installed, using existing environment")

_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
_SDK_DIR = _REPO_ROOT / "pytradingapi-typeB-main"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if _SDK_DIR.exists() and str(_SDK_DIR) not in sys.path:
    sys.path.insert(0, str(_SDK_DIR))

from src.config import load_api_config, load_strategy_config
from src.dhan_client import DhanClient
from src.mstock_client import MStockTypeBClient
from src.strategy import NiftyScalper
from src import gpt


def main() -> None:
    api_cfg = load_api_config()
    strat_cfg = load_strategy_config()
    broker = str(os.getenv("SCALPER_BROKER", "mstock") or "mstock").strip().lower()
    client = DhanClient(strat_cfg) if broker == "dhan" else MStockTypeBClient(api_cfg)

    print(f"Logging in to {broker}...")
    try:
        client.login(interactive=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Login failed: {exc}")
        return
    print("Login successful.")

    scalper = NiftyScalper(client, strat_cfg)
    print("Running enhancements diagnostics...")
    try:
        diag = scalper.run_enhancements_diagnostics()
        print("Enhancements diagnostics:", diag)
    except Exception as exc:  # noqa: BLE001
        print("Diagnostics failed:", exc)

    print("Checking GPT health...")
    try:
        gpt.health_check(print_result=True)
    except Exception as exc:  # noqa: BLE001
        print("GPT health check failed:", exc)

    print("Starting Nifty scalper loop... (Ctrl+C to stop)")
    scalper.run_forever()


if __name__ == "__main__":
    main()

