"""Utility helpers for the Newton + AI hybrid scalping bot."""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover - optional dependency in CI
    mt5 = None  # type: ignore


LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    """Dataclass wrapper around configuration dictionary."""

    raw: Dict[str, Any]

    @property
    def account(self) -> Dict[str, Any]:
        return self.raw.get("account", {})

    @property
    def symbols(self) -> Iterable[str]:
        return self.raw.get("symbols", [])

    @property
    def timeframe(self) -> str:
        return self.raw.get("timeframe", "M1")

    @property
    def mode(self) -> str:
        return self.raw.get("mode", "backtest")

    @property
    def risk(self) -> Dict[str, Any]:
        return self.raw.get("risk", {})

    @property
    def optimization(self) -> Dict[str, Any]:
        return self.raw.get("optimization", {})

    @property
    def ml(self) -> Dict[str, Any]:
        return self.raw.get("ml", {})

    @property
    def rl(self) -> Dict[str, Any]:
        return self.raw.get("rl", {})

    @property
    def logging(self) -> Dict[str, Any]:
        return self.raw.get("logging", {})


def load_config(path: str | Path) -> Config:
    """Load configuration JSON file."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return Config(data)


def ensure_directories(*paths: str | Path) -> None:
    """Create directories if they do not exist."""

    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def timeframe_to_mt5(timeframe: str) -> Optional[int]:
    """Convert textual timeframe to MT5 constant."""

    if mt5 is None:
        return None
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
    }
    return mapping.get(timeframe.upper())


def exponential_moving_average(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rolling_z_score(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    return (series - mean) / (std + 1e-9)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def threaded(fn):
    """Decorator to run a function in a daemon thread."""

    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def timestamp() -> float:
    return time.time()


def realized_volatility(series: pd.Series, window: int = 120) -> pd.Series:
    returns = np.log(series / series.shift())
    return returns.rolling(window=window, min_periods=window).std().fillna(0)


def scale_series(series: pd.Series) -> pd.Series:
    min_val, max_val = series.min(), series.max()
    if max_val - min_val < 1e-9:
        return pd.Series(np.zeros_like(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def to_numpy(data: Iterable[float]) -> np.ndarray:
    return np.array(list(data), dtype=float)


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_directories(path.parent)
    df.to_csv(path, index=False)


class GracefulKiller:
    """Signal-aware shutdown helper."""

    stop = False

    def __init__(self):
        try:
            import signal

            signal.signal(signal.SIGINT, self.exit_gracefully)
            signal.signal(signal.SIGTERM, self.exit_gracefully)
        except Exception:  # pragma: no cover - signals may not be available
            LOGGER.warning("Signal handling not available in this environment")

    def exit_gracefully(self, *_) -> None:
        LOGGER.info("Shutdown signal received. Stopping trading loop...")
        self.stop = True


def connect_mt5(login: int, password: str, server: str, path: str | None = None) -> bool:
    if mt5 is None:
        LOGGER.error("MetaTrader5 package is not installed.")
        return False
    if not mt5.initialize(path):
        LOGGER.error("MT5 initialization failed: %s", mt5.last_error())
        return False
    authorized = mt5.login(login, password=password, server=server)
    if not authorized:
        LOGGER.error("MT5 login failed: %s", mt5.last_error())
        return False
    LOGGER.info("Connected to MetaTrader5 server %s", server)
    return True


def shutdown_mt5() -> None:
    if mt5 is not None:
        mt5.shutdown()


def download_ohlc(symbol: str, timeframe: str, bars: int = 1000) -> pd.DataFrame:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is required for live data")
    mt_timeframe = timeframe_to_mt5(timeframe)
    if mt_timeframe is None:
        raise ValueError(f"Unsupported timeframe {timeframe}")
    rates = mt5.copy_rates_from_pos(symbol, mt_timeframe, 0, bars)
    if rates is None:
        raise RuntimeError(f"Failed to download rates for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df
