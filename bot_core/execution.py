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
from .strategy import BreakoutStrategy, FusionLayer, MeanReversionStrategy, StrategySignal, TrendFollowingStrategy
from .utils import GracefulKiller, atr as ta_atr, connect_mt5, download_ohlc, shutdown_mt5

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
    order_retry_attempts: int = 1
    max_open_trades: int = 3
    probability_threshold: float = 0.6
    max_cycles: int = 1
    enable_trailing_stop: bool = False
    trailing_stop_distance: float = 0.0
    trade_cooldown_sec: int = 0
    min_volume: float = 0.01
    max_volume: float = 5.0
    pip_value: float = 10.0
    stop_loss_atr_multiplier: float = 1.5
    take_profit_atr_multiplier: float = 2.5
    starting_equity: float = 10000.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ExecutionConfig":
        execution_cfg = dict(config.get("execution", {}))
        mode = "live" if config.get("mode") == "live" and not config.get("demo", False) else "backtest"
        return cls(
            mode=mode,
            slippage=float(execution_cfg.get("slippage", 0.0001)),
            refresh_rate_ms=int(execution_cfg.get("refresh_rate_ms", 1000)),
            order_retry_attempts=int(execution_cfg.get("order_retry_attempts", 1)),
            max_open_trades=int(execution_cfg.get("max_open_trades", 3)),
            probability_threshold=float(execution_cfg.get("probability_threshold", 0.6)),
            max_cycles=int(execution_cfg.get("max_cycles", 1)),
            enable_trailing_stop=bool(execution_cfg.get("enable_trailing_stop", False)),
            trailing_stop_distance=float(execution_cfg.get("trailing_stop_distance", 0.0)),
            trade_cooldown_sec=int(execution_cfg.get("trade_cooldown_sec", 0)),
            min_volume=float(execution_cfg.get("min_volume", 0.01)),
            max_volume=float(execution_cfg.get("max_volume", 5.0)),
            pip_value=float(execution_cfg.get("pip_value", 10.0)),
            stop_loss_atr_multiplier=float(execution_cfg.get("stop_loss_atr_multiplier", 1.5)),
            take_profit_atr_multiplier=float(execution_cfg.get("take_profit_atr_multiplier", 2.5)),
            starting_equity=float(execution_cfg.get("starting_equity", 10000.0)),
        )


class ExecutionEngine:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.raw_config = config
        self.config = ExecutionConfig.from_config(config)
        self.account = config.get("account", {})
        self.connected = False
        self.simulated_positions: Dict[str, Dict[str, float]] = {}
        self.symbols: List[str] = list(config.get("symbols", []))
        self.timeframe: str = str(config.get("timeframe", "M1"))
        risk_cfg_raw = config.get("risk", {})
        self.risk_manager = RiskManager(
            RiskConfig(
                max_drawdown=float(risk_cfg_raw.get("max_drawdown", 0.1)),
                daily_profit_target=float(risk_cfg_raw.get("daily_profit_target", 0.05)),
                daily_loss_limit=float(risk_cfg_raw.get("daily_loss_limit", 0.02)),
                kelly_fraction=float(risk_cfg_raw.get("kelly_fraction", 0.5)),
                volatility_lookback=int(risk_cfg_raw.get("volatility_lookback", 120)),
            )
        )
        logging_cfg = config.get("logging", {})
        self.trade_logger = TradeLogger(
            logging_cfg.get("trade_log", "logs/trades.csv"),
            session_log_path=logging_cfg.get("session_log"),
        )
        self.ml_config = MLConfig(**config.get("ml", {}))
        self.ml_model = MLPredictor(self.ml_config)
        self.strategies = [
            TrendFollowingStrategy(),
            MeanReversionStrategy(),
            BreakoutStrategy(),
        ]
        self.fusion_layer = FusionLayer()
        self._last_trade_timestamp: Dict[str, float] = {}
        self._equity_fallback = self.config.starting_equity

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
        if not path.exists():
            path = Path("data") / f"{symbol.lower()}_{timeframe.lower()}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if bars < len(df):
                df = df.tail(bars)
            return df.reset_index(drop=True)
        raise FileNotFoundError(f"Historical data for {symbol} not found in backtest mode")

    def _current_equity(self) -> float:
        if self.connected and mt5 is not None:
            info = mt5.account_info()
            if info is not None:
                return float(getattr(info, "equity", self._equity_fallback))
        return float(self._equity_fallback)

    def _open_trades_count(self) -> int:
        if self.config.mode == "live" and mt5 is not None:
            positions = mt5.positions_get()
            return len(positions) if positions else 0
        return len(self.simulated_positions)

    def _generate_signal(self, df: pd.DataFrame) -> StrategySignal:
        signals: List[StrategySignal] = []
        for strategy in self.strategies:
            try:
                signal = strategy.generate(df)
                signal.metadata.setdefault("name", strategy.name)
                signals.append(signal)
            except Exception as exc:  # pragma: no cover - robust to individual failures
                LOGGER.error("Strategy %s failed: %s", strategy.name, exc)
        if not signals:
            return StrategySignal(action="hold", confidence=0.0, metadata={})
        if len(signals) == 1:
            return signals[0]
        return self.fusion_layer.combine(signals)

    def _decide_action(self, signal: StrategySignal, prediction: Dict[str, float]) -> str:
        suggested = signal.action if signal.action != "hold" else ("buy" if prediction["p_up"] >= prediction["p_down"] else "sell")
        probability = prediction["p_up"] if suggested == "buy" else prediction["p_down"]
        composite_score = probability * max(signal.confidence, 0.1)
        threshold = max(self.config.probability_threshold, self.ml_model.config.decision_threshold)
        if composite_score < threshold:
            return "hold"
        return suggested

    def _calculate_volume(self, equity: float, prediction: Dict[str, float], action: str) -> float:
        payoff = self.config.take_profit_atr_multiplier / max(self.config.stop_loss_atr_multiplier, 1e-6)
        win_rate = prediction["p_up"] if action == "buy" else prediction["p_down"]
        volume = self.risk_manager.position_size(equity, win_rate, payoff, self.config.pip_value)
        return float(min(max(volume, self.config.min_volume), self.config.max_volume))

    def _derive_stops(self, df: pd.DataFrame, price: float, action: str) -> tuple[float, float]:
        atr_values = ta_atr(df, period=14).dropna()
        atr_value = float(atr_values.iloc[-1]) if not atr_values.empty else float(df["close"].pct_change().std() * price)
        atr_value = max(atr_value, price * 0.0005)
        sl_distance = atr_value * self.config.stop_loss_atr_multiplier
        tp_distance = atr_value * self.config.take_profit_atr_multiplier
        if action == "buy":
            return price - sl_distance, price + tp_distance
        return price + sl_distance, price - tp_distance

    def _execute_order(
        self, symbol: str, action: str, volume: float, price: float, sl: float, tp: float
    ) -> Optional[int]:
        attempts = 0
        while attempts < max(self.config.order_retry_attempts, 1):
            ticket = self.place_order(symbol, action, volume, price, sl, tp)
            if ticket is not None:
                return ticket
            attempts += 1
            time.sleep(0.5)
        LOGGER.error("Failed to execute order for %s after %d attempts", symbol, attempts)
        return None

    def start(self) -> None:
        LOGGER.info("Initializing execution engine for %s mode", self.config.mode)
        self.initialize()
        try:
            self.ml_model.load()
            if self.ml_model.model is None:
                raise RuntimeError("ML model not trained. Run in train mode first.")
        except Exception as exc:
            LOGGER.error("Failed to load ML model: %s", exc)
            self.shutdown()
            return

        self.config.probability_threshold = max(
            self.config.probability_threshold, self.ml_model.config.decision_threshold
        )
        killer = GracefulKiller()
        equity = self._current_equity()
        self._equity_fallback = equity
        self.risk_manager.reset_session(equity)

        cycles = 0
        while cycles < max(self.config.max_cycles, 1) and not killer.stop:
            for symbol in self.symbols:
                last_trade = self._last_trade_timestamp.get(symbol, 0.0)
                if (
                    self.config.trade_cooldown_sec > 0
                    and time.time() - last_trade < self.config.trade_cooldown_sec
                ):
                    continue
                try:
                    history = self.fetch_history(
                        symbol,
                        self.timeframe,
                        bars=max(500, self.ml_model.config.features_lookback * 5),
                    )
                except Exception as exc:
                    LOGGER.error("Failed to fetch history for %s: %s", symbol, exc)
                    continue
                if history.empty or len(history) < self.ml_model.config.features_lookback:
                    LOGGER.warning("Not enough history for %s", symbol)
                    continue
                signal = self._generate_signal(history)
                prediction = self.ml_model.predict(history)
                action = self._decide_action(signal, prediction)
                if action == "hold":
                    continue
                if self._open_trades_count() >= self.config.max_open_trades > 0:
                    LOGGER.debug("Max open trades reached; skipping %s", symbol)
                    continue
                equity = self._current_equity()
                self.risk_manager.update_equity(equity)
                if not self.risk_manager.check_limits():
                    LOGGER.warning("Risk limits reached; halting trading loop")
                    killer.stop = True
                    break
                price = float(history["close"].iloc[-1])
                volume = self._calculate_volume(equity, prediction, action)
                if volume <= 0:
                    LOGGER.debug("Volume calculation returned zero for %s", symbol)
                    continue
                sl, tp = self._derive_stops(history, price, action)
                ticket = self._execute_order(symbol, action, volume, price, sl, tp)
                if ticket is None:
                    continue
                self._last_trade_timestamp[symbol] = time.time()
                self.trade_logger.log_trade(
                    symbol=symbol,
                    signal=action,
                    confidence=signal.confidence,
                    volume=volume,
                    price=price,
                    sl=sl,
                    tp=tp,
                    pnl=0.0,
                    strategy_details=signal.metadata,
                    ml_prediction=prediction,
                    rl_action="n/a",
                )
            cycles += 1
            if cycles < max(self.config.max_cycles, 1) and not killer.stop:
                time.sleep(self.config.refresh_rate_ms / 1000)

        LOGGER.info("Execution engine loop completed after %d cycles", cycles)
        self.shutdown()
