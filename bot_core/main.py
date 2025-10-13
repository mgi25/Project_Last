"""Entry point orchestrating the Newton + AI hybrid scalping bot."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .execution import ExecutionConfig, ExecutionEngine
from .logger import TradeLogger, configure_root_logger
from .ml_model import MLConfig, MLPredictor
from .optimizer import BayesianOptimizer, GradientDescentOptimizer, NewtonOptimizer, sharpe_ratio
from .risk import RiskConfig, RiskManager
from .rl_agent import RLAgent, RLConfig
from .strategy import (
    BreakoutStrategy,
    FusionLayer,
    MeanReversionStrategy,
    StrategySignal,
    TrendFollowingStrategy,
    compute_feature_matrix,
)
from .utils import Config, GracefulKiller, ensure_directories, load_config

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Newton + AI Hybrid Scalping Bot")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--mode", choices=["live", "backtest"], default=None, help="Override trading mode")
    parser.add_argument("--backtest-file", default=None, help="Optional CSV file for backtesting")
    return parser.parse_args()


def instantiate_components(config: Config, mode: str) -> Dict[str, object]:
    execution = ExecutionEngine(ExecutionConfig(mode=mode), config.account)
    risk_manager = RiskManager(RiskConfig(**config.risk))
    ml_predictor = MLPredictor(MLConfig(**config.ml))
    rl_agent = RLAgent(RLConfig(**config.rl))
    strategies = [
        TrendFollowingStrategy(),
        MeanReversionStrategy(),
        BreakoutStrategy(),
    ]
    fusion = FusionLayer()
    trade_logger = TradeLogger(config.logging.get("trade_log", "logs/trades.csv"), config.logging.get("session_log"))
    return {
        "execution": execution,
        "risk": risk_manager,
        "ml": ml_predictor,
        "rl": rl_agent,
        "strategies": strategies,
        "fusion": fusion,
        "logger": trade_logger,
    }



def calibrate_strategies(strategies: List, data: pd.DataFrame) -> None:
    returns = data['close'].pct_change().dropna()
    if len(returns) < 100:
        return
    trend = next((s for s in strategies if isinstance(s, TrendFollowingStrategy)), None)
    if trend is not None:
        def objective_trend(x):
            fast = max(2, int(np.exp(x[0])))
            slow = max(fast + 1, int(np.exp(x[1])))
            price = data['close'].rolling(fast).mean() - data['close'].rolling(slow).mean()
            signal = np.sign(price).shift(1).fillna(0)
            pnl = (returns * signal).sum()
            return pnl
        optimizer = NewtonOptimizer(max_iter=10)
        x0 = np.log(np.array([trend.fast, trend.slow]))
        x_opt, _ = optimizer.optimize(objective_trend, x0)
        trend.fast = max(2, int(np.exp(x_opt[0])))
        trend.slow = max(trend.fast + 1, int(np.exp(x_opt[1])))
        LOGGER.info('Trend strategy optimized | fast=%s slow=%s', trend.fast, trend.slow)
    mean_rev = next((s for s in strategies if isinstance(s, MeanReversionStrategy)), None)
    if mean_rev is not None:
        bounds = {'threshold': (0.5, 3.0)}
        def objective(params):
            threshold = params['threshold']
            z = (data['close'] - data['close'].rolling(mean_rev.lookback).mean()) / (data['close'].rolling(mean_rev.lookback).std() + 1e-6)
            signal = np.where(z > threshold, -1, np.where(z < -threshold, 1, 0))
            pnl = (returns * signal).sum()
            return float(pnl)
        if BayesianOptimizer is not None:  # type: ignore
            try:
                optimizer = BayesianOptimizer(n_trials=10)
                best_params, best_value = optimizer.optimize(objective, bounds)
                mean_rev.threshold = float(best_params['threshold'])
                LOGGER.info('Mean reversion optimized | threshold=%.2f score=%.2f', mean_rev.threshold, best_value)
            except Exception as exc:
                LOGGER.warning('Bayesian optimization failed: %s', exc)


def load_historical_data(symbol: str, path: str | None, engine: ExecutionEngine, timeframe: str) -> pd.DataFrame:
    if path:
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        return df
    try:
        return engine.fetch_history(symbol, timeframe)
    except FileNotFoundError:
        raise


def evaluate_strategies(df: pd.DataFrame, strategies: List[TrendFollowingStrategy]) -> List[StrategySignal]:
    signals = []
    for strategy in strategies:
        try:
            signal = strategy.generate(df)
            signal.metadata["name"] = strategy.name
            signals.append(signal)
        except Exception as exc:
            LOGGER.exception("Strategy %s failed: %s", strategy.name, exc)
    return signals


def optimize_weights(returns: np.ndarray) -> np.ndarray:
    optimizer = GradientDescentOptimizer(learning_rate=0.1, max_iter=50)
    weights, _ = optimizer.optimize(lambda w: sharpe_ratio(returns * w.mean()), np.ones(3))
    return np.clip(weights, 0, 1)


def daily_learning_cycle(trade_logger: TradeLogger, ml_predictor: MLPredictor, rl_agent: RLAgent) -> None:
    try:
        trades = trade_logger.read_trades()
    except FileNotFoundError:
        LOGGER.warning("No trade history for learning cycle")
        return
    if trades.empty:
        LOGGER.info("Trade log empty; skipping learning cycle")
        return
    ml_data_path = trade_logger.trade_log_path
    df = pd.read_csv(ml_data_path)
    if len(df) > 100:
        features = compute_feature_matrix(df.rename(columns={"price": "close", "volume": "tick_volume"}))
        try:
            ml_predictor.train(features)
        except Exception as exc:
            LOGGER.exception("ML training failed: %s", exc)
    returns = df["pnl"].fillna(0).values
    if len(returns) > 50:
        rl_history = returns[-500:]
        try:
            rl_agent.train(rl_history.astype(float))
        except Exception as exc:
            LOGGER.exception("RL training failed: %s", exc)


def trading_loop(
    symbol: str,
    data: pd.DataFrame,
    components: Dict[str, object],
    config: Config,
    start_index: int = 200,
) -> None:
    execution: ExecutionEngine = components["execution"]  # type: ignore[assignment]
    risk: RiskManager = components["risk"]  # type: ignore[assignment]
    ml: MLPredictor = components["ml"]  # type: ignore[assignment]
    rl: RLAgent = components["rl"]  # type: ignore[assignment]
    strategies: List = components["strategies"]  # type: ignore[assignment]
    fusion: FusionLayer = components["fusion"]  # type: ignore[assignment]
    logger: TradeLogger = components["logger"]  # type: ignore[assignment]

    killer = GracefulKiller()
    risk.reset_session(equity=100000.0)
    equity = 100000.0

    ml.load()
    rl.load()

    open_position = None
    position_entry_price = 0.0
    position_ticket = None

    for idx in range(start_index, len(data)):
        if killer.stop:
            LOGGER.info("Stopping trading loop for %s", symbol)
            break
        window = data.iloc[: idx + 1]
        signal_list = evaluate_strategies(window, strategies)
        fused_signal = fusion.combine(signal_list)
        ml_prediction = ml.predict(window)
        state = np.array([
            window["close"].iloc[-1],
            window["close"].pct_change().iloc[-1],
            fused_signal.confidence,
            ml_prediction["p_up"],
        ])
        rl_action = rl.act(state)
        action_map = {0: "hold", 1: "buy", 2: "sell", 3: "close"}
        rl_decision = action_map.get(rl_action, "hold")
        final_action = fused_signal.action
        if rl_decision in {"buy", "sell"}:
            final_action = rl_decision
        elif rl_decision == "close" and open_position is not None:
            final_action = "close"

        price = window["close"].iloc[-1]
        volatility = window["close"].pct_change().rolling(30).std().iloc[-1]
        pip_value = 10.0
        win_rate = 0.65
        payoff = 1.8
        volume = risk.position_size(100000.0, win_rate, payoff, pip_value)

        def pnl_objective(sl: float, tp: float) -> float:
            reward = payoff * tp - (1 - win_rate) * sl
            return reward - volatility * 0.1

        sl_tp = risk.optimize_sl_tp(pnl_objective, sl_init=0.001, tp_init=0.002)

        if final_action == "buy":
            position_entry_price = price
            position_ticket = execution.place_order(symbol, "buy", volume, price, price - sl_tp["sl"], price + sl_tp["tp"])
            open_position = "buy"
        elif final_action == "sell":
            position_entry_price = price
            position_ticket = execution.place_order(symbol, "sell", volume, price, price + sl_tp["sl"], price - sl_tp["tp"])
            open_position = "sell"
        elif final_action == "close" and position_ticket is not None:
            execution.close_order(position_ticket)
            open_position = None
            position_ticket = None

        pnl = 0.0
        if open_position == "buy":
            pnl = (price - position_entry_price) * volume
        elif open_position == "sell":
            pnl = (position_entry_price - price) * volume

        logger.log_trade(
            symbol=symbol,
            signal=final_action,
            confidence=fused_signal.confidence,
            volume=volume,
            price=price,
            sl=sl_tp["sl"],
            tp=sl_tp["tp"],
            pnl=pnl,
            strategy_details={meta: value for meta, value in fused_signal.metadata.items()},
            ml_prediction=ml_prediction,
            rl_action=rl_decision,
        )

        equity += pnl
        risk.update_equity(equity)

        if not risk.check_limits():
            LOGGER.warning("Risk limits reached; breaking loop for %s", symbol)
            break

    execution.shutdown()


def main() -> None:
    args = parse_args()
    configure_root_logger()
    config = load_config(args.config)
    mode = args.mode or config.mode
    ensure_directories("data", "logs", "models")
    components = instantiate_components(config, mode)
    execution: ExecutionEngine = components["execution"]  # type: ignore[assignment]
    execution.initialize()

    for symbol in config.symbols:
        try:
            data = load_historical_data(symbol, args.backtest_file, execution, config.timeframe)
        except FileNotFoundError:
            LOGGER.warning("No data for symbol %s; skipping", symbol)
            continue
        calibrate_strategies(components["strategies"], data)
        trading_loop(symbol, data, components, config)

    daily_learning_cycle(components["logger"], components["ml"], components["rl"])


if __name__ == "__main__":
    main()
