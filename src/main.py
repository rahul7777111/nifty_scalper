from __future__ import annotations
import sys
import os
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

sys.path.append(os.path.abspath("pytradingapi-typeB-main"))

from config import load_api_config, load_strategy_config
from mstock_client import MStockTypeBClient
from strategy import NiftyScalper
import gpt


def main() -> None:
    api_cfg = load_api_config()
    strat_cfg = load_strategy_config()

    client = MStockTypeBClient(api_cfg)

    print("Logging in to m.Stock (Type B API)...")
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

