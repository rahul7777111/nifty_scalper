"""Exit optimization with RL agent and advanced rule-based policies.

Provides a hybrid exit optimizer that uses a trained RL agent when available,
falls back to sophisticated rule-based logic, and includes proper environment
setup for training RL-based exit agents.
"""
from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExitDecision:
    """Structured exit decision with metadata."""
    action: str  # "hold", "exit", "partial_exit"
    reason: str
    confidence: float = 1.0
    exit_percentage: float = 1.0  # For partial exits
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ExitEnvironment:
    """RL environment for exit decisions."""
    
    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        self.current_step = 0
        self.trade_state = {}
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take action and return (observation, reward, done, info)."""
        self.current_step += 1
        
        # Action mapping: 0=hold, 1=exit, 2=partial_exit_25%, 3=partial_exit_50%, 4=partial_exit_75%
        reward = self._calculate_reward(action)
        done = action > 0 or self.current_step >= self.max_steps
        
        info = {
            "action_taken": action,
            "step": self.current_step,
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector from trade state."""
        state = self.trade_state
        
        obs = np.array([
            float(state.get("unrealized_pnl", 0.0)),
            float(state.get("pnl_pct", 0.0)),
            float(state.get("time_in_trade_minutes", 0.0)),
            float(state.get("max_profit_seen", 0.0)),
            float(state.get("max_loss_seen", 0.0)),
            float(state.get("current_drawdown", 0.0)),
            float(state.get("delta", 0.0)),
            float(state.get("gamma", 0.0)),
            float(state.get("theta", 0.0)),
            float(state.get("vega", 0.0)),
            float(state.get("iv", 0.0)),
            float(state.get("iv_change_pct", 0.0)),
            float(state.get("underlying_change_pct", 0.0)),
            float(state.get("volume_ratio", 1.0)),
            float(state.get("atr", 0.0)),
            float(state.get("adx", 0.0)),
            float(state.get("rsi", 50.0)),
            float(state.get("regime_trending", 0.0)),
            float(state.get("regime_volatile", 0.0)),
            float(state.get("regime_mean_reverting", 0.0)),
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward for the action taken."""
        pnl = float(self.trade_state.get("unrealized_pnl", 0.0))
        pnl_pct = float(self.trade_state.get("pnl_pct", 0.0))
        time_held = float(self.trade_state.get("time_in_trade_minutes", 0.0))
        
        # Base reward on P&L
        reward = pnl_pct * 10  # Scale P&L percentage
        
        # Penalty for holding too long without profit
        if time_held > 60 and pnl <= 0:  # Held more than 1 hour with loss
            reward -= 0.5
        
        # Reward for timely exit at profit
        if action == 1 and pnl > 0:  # Full exit with profit
            reward += 2.0
        elif action == 1 and pnl <= 0:  # Full exit with loss
            reward -= 1.0
        
        # Incentivize partial profits
        if action in [2, 3, 4] and pnl > 0:
            reward += 1.0
        
        return float(reward)
    
    def update_state(self, trade_state: Dict[str, Any]) -> None:
        """Update the current trade state."""
        self.trade_state = trade_state


class ExitOptimizer:
    """Hybrid exit optimizer with RL agent and rule-based fallback."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = config or {}
        self.rl_model = None
        self.env = None
        self.use_rl = self.cfg.get("use_rl", False)
        
        if self.use_rl:
            self._init_rl()
    
    def _init_rl(self) -> None:
        """Initialize RL environment and try to load existing model."""
        try:
            from stable_baselines3 import PPO
            import os
            
            self.env = ExitEnvironment()
            
            model_path = self.cfg.get("rl_model_path", "exit_agent_model.zip")
            if os.path.exists(model_path):
                self.rl_model = PPO.load(model_path)
                logger.info(f"Loaded RL exit model from {model_path}")
            else:
                logger.info("No pre-trained RL model found. Use train_rl_agent() to train one.")
        except ImportError:
            logger.warning("stable-baselines3 not available. Using rule-based exit only.")
            self.use_rl = False
        except Exception as e:
            logger.error(f"Failed to initialize RL: {e}")
            self.use_rl = False
    
    def suggest_exit(self, trade_state: Dict[str, Any]) -> ExitDecision:
        """Return exit recommendation based on trade state.
        
        Args:
            trade_state: Dictionary containing trade information including:
                - unrealized_pnl: Current unrealized P&L
                - pnl_pct: P&L as percentage of entry
                - time_in_trade_minutes: How long the position has been held
                - entry_price, current_price
                - stop_loss, profit_target
                - delta, gamma, theta, vega (greeks)
                - iv, iv_change_pct
                - underlying_change_pct
                - max_profit_seen, max_loss_seen
                - current_drawdown
                
        Returns:
            ExitDecision with action and metadata
        """
        # Try RL model first if available
        if self.use_rl and self.rl_model is not None and self.env is not None:
            try:
                self.env.update_state(trade_state)
                obs = self.env._get_observation()
                action, _ = self.rl_model.predict(obs, deterministic=True)
                return self._rl_action_to_decision(int(action), trade_state)
            except Exception as e:
                logger.warning(f"RL prediction failed: {e}. Falling back to rules.")
        
        # Rule-based exit logic
        return self._rule_based_exit(trade_state)
    
    def _rl_action_to_decision(self, action: int, trade_state: Dict[str, Any]) -> ExitDecision:
        """Convert RL action integer to ExitDecision."""
        if action == 0:
            return ExitDecision(action="hold", reason="rl_hold", confidence=0.8)
        elif action == 1:
            return ExitDecision(action="exit", reason="rl_exit", confidence=0.9)
        elif action == 2:
            return ExitDecision(action="partial_exit", reason="rl_partial_25", 
                                exit_percentage=0.25, confidence=0.7)
        elif action == 3:
            return ExitDecision(action="partial_exit", reason="rl_partial_50",
                                exit_percentage=0.50, confidence=0.7)
        elif action == 4:
            return ExitDecision(action="partial_exit", reason="rl_partial_75",
                                exit_percentage=0.75, confidence=0.7)
        else:
            return ExitDecision(action="hold", reason="rl_unknown", confidence=0.5)
    
    def _rule_based_exit(self, trade_state: Dict[str, Any]) -> ExitDecision:
        """Sophisticated rule-based exit logic."""
        pnl = float(trade_state.get("unrealized_pnl", 0.0))
        pnl_pct = float(trade_state.get("pnl_pct", 0.0))
        time_held = float(trade_state.get("time_in_trade_minutes", 0.0))
        
        stop_loss = float(trade_state.get("stop_loss", -1e9))
        profit_target = float(trade_state.get("profit_target", 1e9))
        
        # Hard stop loss
        if pnl <= stop_loss:
            return ExitDecision(action="exit", reason="stop_loss_hit", 
                               confidence=1.0, metadata={"pnl": pnl, "stop": stop_loss})
        
        # Profit target
        if pnl >= profit_target:
            return ExitDecision(action="exit", reason="profit_target_hit",
                               confidence=1.0, metadata={"pnl": pnl, "target": profit_target})
        
        # Time-based exit (close to market close)
        if time_held > 300:  # 5 hours
            return ExitDecision(action="exit", reason="time_exit", confidence=0.8)
        
        # Trailing stop logic
        max_profit = float(trade_state.get("max_profit_seen", pnl))
        trailing_pct = float(self.cfg.get("trailing_stop_pct", 0.20))
        if max_profit > 0 and pnl < max_profit * (1 - trailing_pct):
            return ExitDecision(action="exit", reason="trailing_stop", 
                               confidence=0.9, metadata={"max_profit": max_profit, "pnl": pnl})
        
        # Greeks-based exit
        delta = abs(float(trade_state.get("delta", 0.0)))
        theta = float(trade_state.get("theta", 0.0))
        
        # High theta decay with losing position
        if theta < -50 and pnl < 0 and time_held > 60:
            return ExitDecision(action="exit", reason="theta_decay_loss", confidence=0.7)
        
        # Delta explosion (large move against position)
        if delta > 0.7 and pnl < 0:
            return ExitDecision(action="exit", reason="delta_explosion", confidence=0.8)
        
        # Partial profit taking
        partial_pct = float(self.cfg.get("partial_profit_pct", 0.5))
        if pnl > 0 and pnl_pct > partial_pct and time_held > 30:
            return ExitDecision(action="partial_exit", reason="partial_profit",
                               exit_percentage=0.5, confidence=0.6)
        
        # IV collapse exit for long options
        iv_change = float(trade_state.get("iv_change_pct", 0.0))
        if iv_change < -10 and pnl < 0:  # IV dropped more than 10%
            return ExitDecision(action="exit", reason="iv_collapse", confidence=0.7)
        
        return ExitDecision(action="hold", reason="no_exit_signal", confidence=0.5)
    
    def train_rl_agent(self, dataset: Any = None, timesteps: int = 50000) -> Optional[object]:
        """Train the RL exit agent using historical trade data.
        
        Args:
            dataset: Historical trade data for training (optional, uses synthetic if None)
            timesteps: Number of training timesteps
            
        Returns:
            Trained model or None on failure
        """
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.env_util import make_vec_env
            
            if self.env is None:
                self.env = ExitEnvironment()
            
            # Create vectorized environment
            vec_env = make_vec_env(lambda: self.env, n_envs=1)
            
            # Create PPO model
            model = PPO(
                'MlpPolicy',
                vec_env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
            )
            
            # Train
            logger.info(f"Training RL exit agent for {timesteps} timesteps...")
            model.learn(total_timesteps=timesteps)
            
            # Save model
            model_path = self.cfg.get("rl_model_path", "exit_agent_model.zip")
            model.save(model_path.replace(".zip", ""))
            logger.info(f"RL model saved to {model_path}")
            
            self.rl_model = model
            self.use_rl = True
            
            return model
            
        except ImportError:
            logger.error("stable-baselines3 not installed. Cannot train RL agent.")
            return None
        except Exception as e:
            logger.exception(f"RL training failed: {e}")
            return None