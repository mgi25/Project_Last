"""Logging utilities for trading and training."""
from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from .utils import ensure_directories


class TradeLogger:
    """CSV-based trade logger."""

    def __init__(self, trade_log_path: str, session_log_path: Optional[str] = None) -> None:
        self.trade_log_path = Path(trade_log_path)
        ensure_directories(self.trade_log_path.parent)
        self.session_logger = logging.getLogger("session")
        if session_log_path:
            ensure_directories(Path(session_log_path).parent)
            handler = logging.FileHandler(session_log_path)
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.session_logger.addHandler(handler)
        self.session_logger.setLevel(logging.INFO)
        self._initialize_trade_log()

    def _initialize_trade_log(self) -> None:
        if not self.trade_log_path.exists():
            with self.trade_log_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    [
                        "timestamp",
                        "symbol",
                        "signal",
                        "confidence",
                        "volume",
                        "price",
                        "sl",
                        "tp",
                        "pnl",
                        "strategy_details",
                        "ml_prediction",
                        "rl_action",
                    ]
                )

    def log_trade(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        pnl: float,
        strategy_details: Dict[str, float],
        ml_prediction: Dict[str, float],
        rl_action: str,
    ) -> None:
        now = datetime.utcnow().isoformat()
        details = ";".join(f"{k}:{v:.4f}" for k, v in strategy_details.items())
        ml_details = ";".join(f"{k}:{v:.4f}" for k, v in ml_prediction.items())
        row = [now, symbol, signal, confidence, volume, price, sl, tp, pnl, details, ml_details, rl_action]
        with self.trade_log_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(row)
        self.session_logger.info(
            "Trade logged | %s %s vol=%.2f price=%.5f pnl=%.2f", symbol, signal, volume, price, pnl
        )

    def read_trades(self) -> pd.DataFrame:
        return pd.read_csv(self.trade_log_path)

    def log_event(self, message: str) -> None:
        self.session_logger.info(message)


class PerformanceTracker:
    """Maintains accuracy metrics for strategy engines."""

    def __init__(self, lookback: int = 100) -> None:
        self.lookback = lookback
        self.history: Dict[str, Iterable[float]] = {}

    def update(self, key: str, outcome: float) -> None:
        values = list(self.history.get(key, []))[-self.lookback :]
        values.append(outcome)
        self.history[key] = values

    def accuracy(self, key: str) -> float:
        values = list(self.history.get(key, []))
        if not values:
            return 0.5
        return sum(values) / len(values)


def configure_root_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
