"""Reinforcement learning agent integration using stable-baselines3."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    from stable_baselines3 import DDPG, PPO
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:  # pragma: no cover
    DDPG = PPO = None  # type: ignore
    ReplayBuffer = DummyVecEnv = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class RLConfig:
    algorithm: str = "ppo"
    max_steps: int = 20000
    train_interval: str = "1d"


class TradingEnvironment:
    """Simple vectorized environment stub for RL training."""

    def __init__(self, history: np.ndarray):
        self.history = history
        self.index = 0
        self.position = 0
        self.balance = 0.0
        self.done = False

    def reset(self):  # type: ignore[override]
        self.index = 0
        self.position = 0
        self.balance = 0.0
        self.done = False
        return self._state()

    def step(self, action):  # type: ignore[override]
        if self.done:
            return self._state(), 0.0, True, {}
        price_change = self.history[self.index]
        reward = self.position * price_change
        if action == 0:  # hold
            pass
        elif action == 1:  # buy
            self.position = 1
        elif action == 2:  # sell
            self.position = -1
        elif action == 3:  # close
            self.position = 0
        drawdown_penalty = -0.01 * abs(self.position) * max(-reward, 0)
        reward += drawdown_penalty
        self.balance += reward
        self.index += 1
        if self.index >= len(self.history):
            self.done = True
        return self._state(), float(reward), self.done, {"balance": self.balance}

    def _state(self):
        return np.array([self.history[self.index], self.position, self.balance], dtype=float)

    def render(self):  # pragma: no cover - optional
        LOGGER.info("Env index=%d position=%d balance=%.2f", self.index, self.position, self.balance)


class RLAgent:
    def __init__(self, config: RLConfig, model_dir: str = "models") -> None:
        self.config = config
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[object] = None
        self.model_path = self.model_dir / f"rl_{self.config.algorithm}.zip"

    def _build_env(self, history: np.ndarray):
        env = TradingEnvironment(history)
        if DummyVecEnv is None:
            return env
        return DummyVecEnv([lambda: env])

    def train(self, history: np.ndarray) -> None:
        env = self._build_env(history)
        if self.config.algorithm == "ppo":
            if PPO is None:
                raise ImportError("stable-baselines3 is required for PPO")
            model = PPO("MlpPolicy", env, verbose=0)
        elif self.config.algorithm == "ddpg":
            if DDPG is None:
                raise ImportError("stable-baselines3 is required for DDPG")
            model = DDPG("MlpPolicy", env, verbose=0)
        else:
            raise ValueError(f"Unsupported RL algorithm {self.config.algorithm}")
        model.learn(total_timesteps=self.config.max_steps)
        model.save(self.model_path)
        self.model = model
        LOGGER.info("RL model trained and saved to %s", self.model_path)

    def load(self) -> None:
        if not self.model_path.exists():
            LOGGER.warning("RL model file %s does not exist", self.model_path)
            return
        if self.config.algorithm == "ppo":
            if PPO is None:
                raise ImportError("stable-baselines3 is required for PPO")
            self.model = PPO.load(self.model_path)
        elif self.config.algorithm == "ddpg":
            if DDPG is None:
                raise ImportError("stable-baselines3 is required for DDPG")
            self.model = DDPG.load(self.model_path)
        LOGGER.info("Loaded RL model from %s", self.model_path)

    def act(self, state: np.ndarray) -> int:
        if self.model is None:
            LOGGER.debug("RL model not loaded; returning hold action")
            return 0
        if hasattr(self.model, "predict"):
            action, _ = self.model.predict(state, deterministic=True)
            return int(action)
        return 0

    def update_from_experience(self, state, action, reward, next_state, done) -> None:
        if self.model is None:
            return
        if isinstance(self.model, PPO):  # type: ignore[arg-type]
            # PPO does not support manual replay buffer updates; rely on retraining schedule.
            return
        if isinstance(self.model, DDPG):  # type: ignore[arg-type]
            buffer: ReplayBuffer = self.model.replay_buffer
            buffer.add(state, next_state, np.array([action]), np.array([reward]), np.array([done]))
