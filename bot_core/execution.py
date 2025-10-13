"""Trade execution layer with MT5 integration and simulation."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .logger import TradeLogger
from .ml_model import MLConfig, MLPredictor
from .risk import RiskConfig, RiskManager
from .strategy import (
    BreakoutStrategy,
    FusionLayer,
    MeanReversionStrategy,
    StrategySignal,
    TrendFollowingStrategy,
)
from .utils import GracefulKiller, atr as atr_indicator, connect_mt5, download_ohlc, shutdown_mt5

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover
    mt5 = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    mode: str = "backtest"
    slippage: float = 0.0001
    refresh_rate_ms: int = 1000
    max_open_trades: int = 1
    trade_cooldown_sec: int = 30
    enable_trailing_stop: bool = False
    trailing_stop_distance: float = 0.0
    history_bars: int = 500
    min_ml_confidence: float = 0.55
    max_cycles: int | None = None


class ExecutionEngine:
    def __init__(
        self,
        raw_config: Dict[str, Any],
        exec_config: ExecutionConfig | None = None,
        account: Optional[Dict[str, str]] = None,
        demo: bool = False,
    ) -> None:
        self.raw_config = raw_config
        self.config = exec_config or ExecutionConfig()
        self.config.mode = self.config.mode or raw_config.get("mode", "backtest")
        self.account = account or raw_config.get("account", {})
        self.demo = demo
        self.connected = False
        self.simulated_positions: Dict[str, Dict[str, float]] = {}
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.symbols: List[str] = list(raw_config.get("symbols", []))
        self.timeframe: str = str(raw_config.get("timeframe", "M1"))
        self.execution_settings: Dict[str, Any] = raw_config.get("execution", {})
        self.logging_settings: Dict[str, Any] = raw_config.get("logging", {})
        self.config.refresh_rate_ms = int(
            self.execution_settings.get("refresh_rate_ms", self.config.refresh_rate_ms)
        )
        self.config.max_open_trades = int(
            self.execution_settings.get("max_open_trades", self.config.max_open_trades)
        )
        self.config.trade_cooldown_sec = int(
            self.execution_settings.get("trade_cooldown_sec", self.config.trade_cooldown_sec)
        )
        self.config.enable_trailing_stop = bool(
            self.execution_settings.get("enable_trailing_stop", self.config.enable_trailing_stop)
        )
        self.config.trailing_stop_distance = float(
            self.execution_settings.get(
                "trailing_stop_distance", self.config.trailing_stop_distance
            )
        )
        self.config.history_bars = int(
            self.execution_settings.get("history_bars", self.config.history_bars)
        )
        self.config.min_ml_confidence = float(
            self.execution_settings.get("min_ml_confidence", self.config.min_ml_confidence)
        )
        max_cycles = self.execution_settings.get("max_cycles")
        if max_cycles is not None:
            try:
                self.config.max_cycles = int(max_cycles)
            except (TypeError, ValueError):
                LOGGER.warning("Invalid max_cycles value %r; ignoring.", max_cycles)
        self.risk_manager = RiskManager(RiskConfig(**raw_config.get("risk", {})))
        self.fusion_layer = FusionLayer()
        self.strategies: List[TrendFollowingStrategy | MeanReversionStrategy | BreakoutStrategy] = [
            TrendFollowingStrategy(),
            MeanReversionStrategy(),
            BreakoutStrategy(),
        ]
        self.trade_logger = TradeLogger(
            self.logging_settings.get("trade_log", "logs/trades.csv"),
            self.logging_settings.get("session_log"),
        )
        ml_cfg = MLConfig(**raw_config.get("ml", {}))
        self.ml_predictor = MLPredictor(ml_cfg)
        self.ml_predictor.load()
        self.min_confidence = float(self.execution_settings.get("min_ml_confidence", self.config.min_ml_confidence))
        self.cooldowns: Dict[str, float] = {}
        self.initial_equity = float(self.account.get("equity", 10000.0))

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

    def start(self) -> None:
        LOGGER.info(
            "Starting execution engine | mode=%s demo=%s symbols=%s",
            self.config.mode,
            self.demo,
            ",".join(self.symbols) or "<none>",
        )
        self.initialize()
        if self.config.mode != "live":
            LOGGER.warning("Execution engine start called with mode=%s; aborting.", self.config.mode)
            return
        if not self.demo and not self.connected:
            LOGGER.error("MT5 connection unavailable; cannot trade live.")
            return
        if self.ml_predictor.model is None:
            LOGGER.error("ML model not loaded; aborting live trading loop.")
            return

        killer = GracefulKiller()
        self.risk_manager.reset_session(self.initial_equity)
        cycle = 0
        try:
            while not killer.stop:
                cycle += 1
                for symbol in self.symbols:
                    if killer.stop:
                        break
                    self._process_symbol(symbol)
                if self.config.max_cycles is not None and cycle >= self.config.max_cycles:
                    LOGGER.info("Max cycles reached (%s). Stopping execution loop.", self.config.max_cycles)
                    break
                time.sleep(max(self.config.refresh_rate_ms, 200) / 1000.0)
        finally:
            self.shutdown()

    def _process_symbol(self, symbol: str) -> None:
        now = time.time()
        last_trade = self.cooldowns.get(symbol, 0.0)
        if now - last_trade < self.config.trade_cooldown_sec:
            return
        if self.config.max_open_trades and len(self.open_positions) >= self.config.max_open_trades:
            LOGGER.debug("Max open trades reached; skipping %s", symbol)
            return
        try:
            history = self.fetch_history(symbol, self.timeframe, bars=self.config.history_bars)
        except Exception as exc:
            LOGGER.error("Failed to fetch history for %s: %s", symbol, exc)
            return
        if history.empty or len(history) < 50:
            LOGGER.debug("Insufficient history for %s", symbol)
            return

        ml_signal = self.ml_predictor.predict(history)
        strategy_signals: List[StrategySignal] = []
        for strategy in self.strategies:
            try:
                signal = strategy.generate(history)
            except Exception as exc:
                LOGGER.error("Strategy %s failed for %s: %s", strategy.name, symbol, exc)
                continue
            signal.metadata["name"] = strategy.name
            strategy_signals.append(signal)
        if not strategy_signals:
            return
        fused = self.fusion_layer.combine(strategy_signals)
        action = self._resolve_action(fused.action, ml_signal)
        confidence = max(ml_signal.get("confidence", 0.0), fused.confidence)
        if action == "hold" or confidence < self.min_confidence:
            return

        price = float(history["close"].iloc[-1])
        atr_series = atr_indicator(history, 14)
        volatility = float(atr_series.iloc[-1]) if len(atr_series.dropna()) else price * 0.001
        sl, tp = self._compute_sl_tp(price, volatility, action)
        payoff = abs(tp - price) / max(abs(price - sl), 1e-9)
        win_rate = ml_signal.get("p_up", 0.5) if action == "buy" else ml_signal.get("p_down", 0.5)
        volume = self.risk_manager.position_size(
            equity=self.initial_equity,
            win_rate=win_rate,
            payoff=payoff,
            pip_value=10.0,
        )
        if volume <= 0:
            LOGGER.debug("Computed volume %.4f not tradeable for %s", volume, symbol)
            return
        ticket: Optional[int]
        if self.demo:
            ticket = self._place_order_simulation(symbol, action, volume, price, sl, tp)
        else:
            ticket = self.place_order(symbol, action, volume, price, sl, tp)
        if ticket is None:
            return
        ticket_key = str(ticket)
        self.open_positions[ticket_key] = {
            "symbol": symbol,
            "action": action,
            "volume": volume,
            "price": price,
            "sl": sl,
            "tp": tp,
        }
        self.cooldowns[symbol] = now
        self.trade_logger.log_trade(
            symbol=symbol,
            signal=action,
            confidence=confidence,
            volume=volume,
            price=price,
            sl=sl,
            tp=tp,
            pnl=0.0,
            strategy_details=fused.metadata,
            ml_prediction=ml_signal,
            rl_action="idle",
        )

    def _resolve_action(self, strategy_action: str, ml_signal: Dict[str, float]) -> str:
        ml_bias = ml_signal.get("bias", "hold")
        if strategy_action == "hold" and ml_signal.get("confidence", 0.0) >= self.min_confidence:
            return "buy" if ml_bias == "buy" else "sell"
        if strategy_action == "hold":
            return "hold"
        if ml_bias in {"buy", "sell"}:
            if ml_bias != strategy_action and ml_signal.get("confidence", 0.0) < self.min_confidence + 0.1:
                return "hold"
            return ml_bias if ml_signal.get("confidence", 0.0) >= self.min_confidence else strategy_action
        return strategy_action

    def _compute_sl_tp(self, price: float, volatility: float, action: str) -> tuple[float, float]:
        if self.config.enable_trailing_stop and self.config.trailing_stop_distance > 0:
            base_distance = self.config.trailing_stop_distance
        else:
            base_distance = volatility * 1.5
        distance = max(base_distance, volatility * 1.2)
        if action == "buy":
            sl = price - distance
            tp = price + distance * 2
        else:
            sl = price + distance
            tp = price - distance * 2
        return float(sl), float(tp)
