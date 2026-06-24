"""RL Exit Agent harness stub.

Provides a simple interface for offline training (if stable-baselines3 is
available) and a no-op fallback otherwise.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ExitAgent:
    def __init__(self) -> None:
        self.model = None

    def train_offline(self, dataset, timesteps: int = 10000, env=None) -> Optional[object]:
        """Train an RL agent offline using provided dataset and env.

        If stable-baselines3 is missing, this is a no-op and returns None.
        """
        try:
            from stable_baselines3 import PPO
        except Exception:
            logger.warning("stable-baselines3 not available; skipping RL training")
            return None

        if env is None:
            logger.warning("No env provided for RL training; skipping")
            return None

        try:
            model = PPO('MlpPolicy', env, verbose=0)
            model.learn(total_timesteps=int(timesteps))
            self.model = model
            return model
        except Exception as e:
            logger.exception("RL training failed: %s", e)
            return None

    def suggest_exit(self, observation) -> dict:
        """Given an observation, return an exit suggestion dict.

        If a trained model exists, use it; otherwise return a simple rule-based suggestion.
        """
        if self.model is not None:
            try:
                act, _ = self.model.predict(observation, deterministic=True)
                return {"action": int(act)}
            except Exception:
                pass
        # fallback: hold (0) or exit (1) based on a simple threshold in observation
        try:
            risk = float(observation.get("risk") or 0.0)
            if risk > 0.5:
                return {"action": 1, "reason": "risk_high"}
        except Exception:
            pass
        return {"action": 0, "reason": "hold"}
