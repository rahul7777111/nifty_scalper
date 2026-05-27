"""Train the exit RL agent using an environment (if available).

This script delegates to `src.exit_agent.ExitAgent` and will attempt to
train if `stable-baselines3` and an env are present. It's intentionally
small and safe for CI.
"""
from __future__ import annotations

import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=10000)
    args = p.parse_args()

    from src.exit_agent import ExitAgent

    agent = ExitAgent()
    res = agent.train_offline(None, timesteps=args.timesteps, env=None)
    if res is None:
        print("No training performed (stable-baselines3 or env missing)")
    else:
        print("Trained ExitAgent model saved in memory")

if __name__ == "__main__":
    main()
