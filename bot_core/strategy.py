"""Trading strategy engines."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .logger import PerformanceTracker
from .utils import atr, exponential_moving_average, realized_volatility, rolling_z_score, scale_series

LOGGER = logging.getLogger(__name__)


@dataclass
class StrategySignal:
    action: str
    confidence: float
    metadata: Dict[str, float]


class BaseStrategy:
    name: str = "base"

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        raise NotImplementedError


class TrendFollowingStrategy(BaseStrategy):
    name = "trend"

    def __init__(self, fast: int = 9, slow: int = 21) -> None:
        self.fast = fast
        self.slow = slow

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        fast_ma = exponential_moving_average(df["close"], self.fast)
        slow_ma = exponential_moving_average(df["close"], self.slow)
        momentum = df["close"].pct_change().rolling(5).sum()
        signal = 0
        if fast_ma.iloc[-1] > slow_ma.iloc[-1] and momentum.iloc[-1] > 0:
            signal = 1
        elif fast_ma.iloc[-1] < slow_ma.iloc[-1] and momentum.iloc[-1] < 0:
            signal = -1
        confidence = float(abs(fast_ma.iloc[-1] - slow_ma.iloc[-1]) / (df["close"].iloc[-1] + 1e-6))
        return StrategySignal(
            action="buy" if signal > 0 else "sell" if signal < 0 else "hold",
            confidence=min(confidence, 1.0),
            metadata={"fast_ma": fast_ma.iloc[-1], "slow_ma": slow_ma.iloc[-1], "momentum": momentum.iloc[-1]},
        )


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"

    def __init__(self, lookback: int = 20, threshold: float = 1.5) -> None:
        self.lookback = lookback
        self.threshold = threshold

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        z = rolling_z_score(df["close"], self.lookback)
        value = z.iloc[-1]
        if value > self.threshold:
            action = "sell"
        elif value < -self.threshold:
            action = "buy"
        else:
            action = "hold"
        confidence = float(min(abs(value) / (self.threshold + 1e-6), 1.0))
        return StrategySignal(action=action, confidence=confidence, metadata={"z_score": value})


class BreakoutStrategy(BaseStrategy):
    name = "breakout"

    def __init__(self, atr_period: int = 14) -> None:
        self.atr_period = atr_period

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        volatility = atr(df, self.atr_period).iloc[-1]
        recent_range = df["high"].rolling(20).max().iloc[-1] - df["low"].rolling(20).min().iloc[-1]
        price = df["close"].iloc[-1]
        upper_break = df["high"].rolling(20).max().iloc[-2]
        lower_break = df["low"].rolling(20).min().iloc[-2]
        if price > upper_break + volatility:
            action = "buy"
        elif price < lower_break - volatility:
            action = "sell"
        else:
            action = "hold"
        confidence = float(scale_series(pd.Series([recent_range, volatility])).iloc[0])
        return StrategySignal(action=action, confidence=confidence, metadata={"atr": volatility, "range": recent_range})


class FusionLayer:
    """Combines strategy signals using adaptive weights."""

    def __init__(self, tracker: PerformanceTracker | None = None) -> None:
        self.tracker = tracker or PerformanceTracker()

    def combine(self, signals: List[StrategySignal]) -> StrategySignal:
        weights = []
        actions = {"buy": 1, "sell": -1, "hold": 0}
        weighted_action = 0.0
        total_weight = 0.0
        metadata: Dict[str, float] = {}
        for signal in signals:
            weight = self.tracker.accuracy(signal.metadata.get("name", signal.action)) + 0.01
            weights.append(weight)
            weighted_action += actions.get(signal.action, 0) * weight * signal.confidence
            total_weight += weight * signal.confidence
            for key, value in signal.metadata.items():
                metadata[f"{signal.action}_{key}"] = value
        metadata["weights_sum"] = total_weight
        score = weighted_action / (total_weight + 1e-9)
        if score > 0.1:
            action = "buy"
        elif score < -0.1:
            action = "sell"
        else:
            action = "hold"
        confidence = float(min(abs(score), 1.0))
        return StrategySignal(action=action, confidence=confidence, metadata=metadata)


def compute_feature_matrix(df: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["rsi"] = ta_rsi(df["close"], window=14)
    df["volatility"] = realized_volatility(df["close"], window=lookback)
    df["momentum"] = df["close"].diff(lookback)
    df["range"] = df["high"] - df["low"]
    df = df.dropna()
    return df


def ta_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=window).mean()
    roll_down = down.rolling(window=window).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))
