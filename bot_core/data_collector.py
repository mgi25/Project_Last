"""Historical data collection utilities for the trading bot."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


class DataCollector:
    """Download historical OHLC data from MetaTrader5 and store as CSV files."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.symbols: Iterable[str] = config.get("symbols", [])
        self.timeframe: str = str(config.get("timeframe", "M1")).upper()
        self.days: int = int(config.get("collection_days", 30))
        self.data_dir = Path(config.get("data_dir", "data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _timeframe_to_mt5(timeframe: str) -> Optional[int]:
        try:
            import MetaTrader5 as mt5
        except ImportError:  # pragma: no cover - optional dependency during tests
            return None
        mapping = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        return mapping.get(timeframe.upper())

    def _collect_symbol(self, symbol: str, timeframe_const: int, start: datetime, end: datetime) -> None:
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:  # pragma: no cover - optional dependency during tests
            LOGGER.error("MetaTrader5 package missing when collecting data: %s", exc)
            raise

        LOGGER.info("Collecting data for %s", symbol)
        print(f"⬇️  Downloading {symbol} {self.timeframe} candles...")

        if not mt5.symbol_select(symbol, True):
            error = mt5.last_error()
            LOGGER.error("Failed to select symbol %s: %s", symbol, error)
            print(f"❌ Failed to select {symbol}: {error}")
            return

        rates = mt5.copy_rates_range(symbol, timeframe_const, start, end)
        if rates is None:
            error = mt5.last_error()
            LOGGER.error("Failed to download rates for %s: %s", symbol, error)
            print(f"❌ Failed to download {symbol}: {error}")
            return

        df = pd.DataFrame(rates)
        if df.empty:
            LOGGER.warning("No data returned for %s", symbol)
            print(f"⚠️  No data returned for {symbol}")
            return

        df["time"] = pd.to_datetime(df["time"], unit="s")
        file_path = self.data_dir / f"{symbol.lower()}_{self.timeframe.lower()}.csv"
        df.to_csv(file_path, index=False)

        LOGGER.info("Saved %d rows for %s to %s", len(df), symbol, file_path)
        print(f"✅ Saved {len(df)} rows to {file_path}")

    def run(self) -> None:
        """Execute the data collection pipeline."""
        if not self.symbols:
            LOGGER.warning("No symbols configured for data collection")
            print("⚠️  No symbols provided in configuration")
            return

        timeframe_const = self._timeframe_to_mt5(self.timeframe)
        if timeframe_const is None:
            raise ValueError(f"Unsupported timeframe '{self.timeframe}' for MetaTrader5")

        end = datetime.utcnow()
        start = end - timedelta(days=self.days)

        LOGGER.info("Starting data collection for %d symbols", len(list(self.symbols)))
        for symbol in self.symbols:
            try:
                self._collect_symbol(symbol, timeframe_const, start, end)
            except Exception as exc:  # pragma: no cover - log and continue other symbols
                LOGGER.exception("Error collecting data for %s: %s", symbol, exc)
        LOGGER.info("Data collection complete")
        print("📦 Data collection complete")
