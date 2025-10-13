"""Risk management utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .optimizer import optimize_stop_levels

LOGGER = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    max_drawdown: float = 0.1
    daily_profit_target: float = 0.05
    daily_loss_limit: float = 0.02
    kelly_fraction: float = 0.5
    volatility_lookback: int = 120


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self.start_equity = None
        self.daily_pnl = 0.0
        self.drawdown = 0.0

    def reset_session(self, equity: float) -> None:
        self.start_equity = equity
        self.daily_pnl = 0.0
        self.drawdown = 0.0
        LOGGER.info("Risk manager session reset | equity=%.2f", equity)

    def update_equity(self, equity: float) -> None:
        if self.start_equity is None:
            self.reset_session(equity)
            return
        self.daily_pnl = equity - self.start_equity
        self.drawdown = min(self.drawdown, self.daily_pnl)

    def check_limits(self) -> bool:
        if self.start_equity is None:
            return True
        equity = self.start_equity + self.daily_pnl
        if self.daily_pnl <= -self.config.daily_loss_limit * self.start_equity:
            LOGGER.warning("Daily loss limit reached. Stopping trading.")
            return False
        if self.daily_pnl >= self.config.daily_profit_target * self.start_equity:
            LOGGER.info("Daily profit target reached. Stopping trading.")
            return False
        if equity <= self.start_equity * (1 - self.config.max_drawdown):
            LOGGER.error("Max drawdown exceeded. Stopping trading.")
            return False
        return True

    def kelly_fraction(self, win_rate: float, payoff: float) -> float:
        edge = win_rate * (payoff + 1) - 1
        if payoff <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        frac = edge / payoff
        return max(min(frac, 1.0), 0.0) * self.config.kelly_fraction

    def position_size(self, equity: float, win_rate: float, payoff: float, pip_value: float) -> float:
        fraction = self.kelly_fraction(win_rate, payoff)
        volume = equity * fraction / (pip_value + 1e-9)
        LOGGER.debug("Position size computed | equity=%.2f fraction=%.4f volume=%.4f", equity, fraction, volume)
        return max(volume, 0.0)

    def optimize_sl_tp(self, objective: callable, sl_init: float, tp_init: float) -> Dict[str, float]:
        sl, tp, score = optimize_stop_levels(objective, sl_init, tp_init)
        LOGGER.debug("Optimized SL/TP | sl=%.5f tp=%.5f score=%.5f", sl, tp, score)
        return {"sl": sl, "tp": tp, "score": score}

    def trailing_stop(self, current_price: float, entry_price: float, volatility: float, direction: str) -> float:
        buffer = volatility * 1.5
        if direction == "buy":
            return max(entry_price + buffer, current_price - buffer)
        else:
            return min(entry_price - buffer, current_price + buffer)
