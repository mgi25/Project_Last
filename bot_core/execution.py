"""Trade execution layer with MT5 integration and simulation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .utils import connect_mt5, download_ohlc, shutdown_mt5

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover
    mt5 = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    mode: str = "backtest"
    slippage: float = 0.0001


class ExecutionEngine:
    def __init__(self, config: ExecutionConfig, account: Dict[str, str]) -> None:
        self.config = config
        self.account = account
        self.connected = False
        self.simulated_positions: Dict[str, Dict[str, float]] = {}

    def initialize(self) -> None:
        if self.config.mode == "live":
            self.connected = connect_mt5(
                login=int(self.account.get("login", 0)),
                password=self.account.get("password", ""),
                server=self.account.get("server", ""),
                path=self.account.get("path"),
            )
        else:
            self.connected = False
            LOGGER.info("Execution engine in backtest mode.")

    def shutdown(self) -> None:
        if self.connected and mt5 is not None:
            shutdown_mt5()
            self.connected = False
        LOGGER.info("Execution engine shutdown complete.")

    def _symbol_info(self, symbol: str):
        if mt5 is None:
            return None
        return mt5.symbol_info(symbol)

    def _place_order_live(
        self, symbol: str, action: str, volume: float, price: float, sl: float, tp: float
    ) -> Optional[int]:
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package is required for live execution")
        order_type = mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": int(self.config.slippage * 1e5),
            "magic": 20240925,
            "comment": "NewtonAI",
        }
        result = mt5.order_send(request)
        if result is None:
            LOGGER.error("Order send failed: %s", mt5.last_error())
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            LOGGER.error("Order rejected: %s", result)
            return None
        LOGGER.info("Order placed | ticket=%s symbol=%s action=%s volume=%.2f", result.order, symbol, action, volume)
        return result.order

    def _place_order_simulation(
        self, symbol: str, action: str, volume: float, price: float, sl: float, tp: float
    ) -> int:
        ticket = len(self.simulated_positions) + 1
        self.simulated_positions[str(ticket)] = {
            "symbol": symbol,
            "action": action,
            "volume": volume,
            "price": price,
            "sl": sl,
            "tp": tp,
        }
        LOGGER.info("Simulated order %s stored.", ticket)
        return ticket

    def place_order(
        self, symbol: str, action: str, volume: float, price: float, sl: float, tp: float
    ) -> Optional[int]:
        if self.config.mode == "live":
            return self._place_order_live(symbol, action, volume, price, sl, tp)
        return self._place_order_simulation(symbol, action, volume, price, sl, tp)

    def modify_order(self, ticket: int, sl: float, tp: float) -> bool:
        if self.config.mode == "live":
            if mt5 is None:
                raise RuntimeError("MetaTrader5 package is required for live execution")
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": sl,
                "tp": tp,
            }
            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                LOGGER.error("Modify order failed: %s", mt5.last_error())
                return False
            return True
        else:
            if str(ticket) in self.simulated_positions:
                self.simulated_positions[str(ticket)]["sl"] = sl
                self.simulated_positions[str(ticket)]["tp"] = tp
                LOGGER.info("Simulated order %s modified.", ticket)
                return True
        return False

    def close_order(self, ticket: int) -> bool:
        if self.config.mode == "live":
            if mt5 is None:
                raise RuntimeError("MetaTrader5 package is required for live execution")
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False
            pos = position[0]
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(pos.symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(pos.symbol).ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": int(self.config.slippage * 1e5),
                "magic": 20240925,
                "comment": "NewtonAI close",
            }
            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                LOGGER.error("Close order failed: %s", mt5.last_error())
                return False
            LOGGER.info("Order %s closed", ticket)
            return True
        else:
            if str(ticket) in self.simulated_positions:
                del self.simulated_positions[str(ticket)]
                LOGGER.info("Simulated order %s closed.", ticket)
                return True
        return False

    def fetch_history(self, symbol: str, timeframe: str, bars: int = 1000) -> pd.DataFrame:
        if self.config.mode == "live":
            return download_ohlc(symbol, timeframe, bars)
        path = Path("data") / f"{symbol}_{timeframe}.csv"
        if path.exists():
            return pd.read_csv(path)
        raise FileNotFoundError(f"Historical data for {symbol} not found in backtest mode")
