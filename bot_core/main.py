"""Command-line entry point for the Newton + AI Hybrid Forex Scalping Bot."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the trading bot."""
    parser = argparse.ArgumentParser(description="Exness MT5 Hybrid Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["collect", "train", "tune", "backtest", "live"],
        required=True,
        help="Operation mode to execute",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode (metadata only, no live orders)",
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    return parser.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a JSON configuration file."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data: Dict[str, Any] = json.load(handle)
    return data


def ensure_directories(*directories: str | Path) -> None:
    """Ensure directories exist before writing files."""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def configure_logging(log_file: Path) -> None:
    """Configure application-wide logging to file and console."""
    ensure_directories(log_file.parent)
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(console)


def initialize_mt5(cfg: Dict[str, Any]) -> bool:
    """Initialise the MetaTrader5 terminal using credentials from the config."""
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:  # pragma: no cover - optional dependency during tests
        print("❌ MetaTrader5 package is not installed.")
        logging.error("MetaTrader5 package missing: %s", exc)
        return False

    # Support both flat and nested account credential structures
    account_cfg: Dict[str, Any] = {}
    if isinstance(cfg.get("account"), dict):
        account_cfg = dict(cfg["account"])

    terminal_path = account_cfg.get("path", cfg.get("path"))
    login = account_cfg.get("login", cfg.get("account"))
    password = account_cfg.get("password", cfg.get("password"))
    server = account_cfg.get("server", cfg.get("server"))

    if login is None:
        print("❌ MT5 initialization failed: missing login credential")
        logging.error("MT5 initialization failed: missing login credential")
        return False

    try:
        login_int = int(login)
    except (TypeError, ValueError):
        print(f"❌ MT5 initialization failed: invalid login value {login!r}")
        logging.error("MT5 initialization failed: invalid login value %r", login)
        return False

    initialized = mt5.initialize(path=terminal_path, login=login_int, password=password, server=server)
    if not initialized:
        error = mt5.last_error()
        print("❌ MT5 initialization failed:", error)
        logging.error("MT5 initialization failed: %s", error)
        return False

    print(f"✅ Connected to MT5 account {login_int} on {server}")
    logging.info("Connected to MT5 account %s on %s", login_int, server)
    return True


def shutdown_mt5() -> None:
    """Shutdown MetaTrader5 if available."""
    try:
        import MetaTrader5 as mt5
    except ImportError:  # pragma: no cover - optional dependency during tests
        return
    mt5.shutdown()


def _default_data_path(config: Dict[str, Any]) -> Path:
    symbols = config.get("symbols") or []
    timeframe = str(config.get("timeframe", "M1")).lower()
    data_dir = Path(config.get("data_dir", "data"))
    if symbols:
        return data_dir / f"{symbols[0].lower()}_{timeframe}.csv"
    return data_dir / f"default_{timeframe}.csv"


def main() -> None:
    """Main entrypoint orchestrating the bot modes."""
    args = parse_args()
    config = load_config(args.config)
    config["demo"] = bool(args.demo)

    ensure_directories("data", "logs", "models")
    configure_logging(Path("logs") / "main.log")

    logging.info("Starting bot in %s mode", args.mode)

    mt5_connected = False
    try:
        if args.mode in {"collect", "live"}:
            mt5_connected = initialize_mt5(config)
            if not mt5_connected:
                raise RuntimeError("Failed to initialize MetaTrader5")

        if args.mode == "collect":
            from bot_core.data_collector import DataCollector

            collector = DataCollector(config)
            collector.run()

        elif args.mode == "train":
            try:
                from bot_core.ml_model import MLModel  # type: ignore[attr-defined]
            except ImportError:
                from bot_core.ml_model import MLConfig, MLPredictor
                import pandas as pd

                ml_cfg = config.get("ml", {})
                trainer = MLPredictor(MLConfig(**ml_cfg))
                data_path = Path(config.get("training_data", _default_data_path(config)))
                if not data_path.exists():
                    raise FileNotFoundError(
                        f"Training data not found at {data_path}. Run in collect mode first."
                    )
                dataset = pd.read_csv(data_path)
                metrics = trainer.train(dataset)
                logging.info("Training complete: %s", metrics)
            else:
                trainer = MLModel(config)
                trainer.train()

        elif args.mode == "tune":
            try:
                from bot_core.optimizer import Optimizer  # type: ignore[attr-defined]
            except ImportError:
                from bot_core.optimizer import BayesianOptimizer, NewtonOptimizer
                import numpy as np

                logging.info("Running legacy tuning pipeline")

                def objective(x: np.ndarray) -> float:
                    return -float(np.sum((x - 1.0) ** 2))

                newton = NewtonOptimizer(max_iter=25)
                x_opt, score = newton.optimize(objective, np.array([0.5, 0.5]))
                logging.info("Newton optimization result: x=%s score=%.4f", x_opt, score)

                if config.get("use_bayesian", True):
                    bounds = {"threshold": (0.1, 2.0)}
                    try:
                        bayes = BayesianOptimizer(n_trials=10)
                        params, best = bayes.optimize(lambda p: -abs(p["threshold"] - 1.0), bounds)
                        logging.info("Bayesian tuning result: params=%s score=%.4f", params, best)
                    except Exception as exc:
                        logging.warning("Bayesian optimization unavailable: %s", exc)
            else:
                optimizer = Optimizer(config)
                optimizer.run()

        elif args.mode == "backtest":
            try:
                from bot_core.strategy import Backtester  # type: ignore[attr-defined]
            except ImportError:
                import pandas as pd
                from bot_core.strategy import TrendFollowingStrategy

                logging.info("Running basic backtest")
                data_path = Path(config.get("backtest_data", _default_data_path(config)))
                if not data_path.exists():
                    raise FileNotFoundError(
                        f"Backtest data not found at {data_path}. Collect data first."
                    )
                df = pd.read_csv(data_path, parse_dates=["time"])  # type: ignore[arg-type]
                strategy = TrendFollowingStrategy()
                df["return"] = df["close"].pct_change().fillna(0)
                signals = []
                for idx in range(len(df)):
                    window = df.iloc[: idx + 1]
                    if len(window) < 50:
                        signals.append(0)
                        continue
                    signal = strategy.generate(window)
                    signals.append(1 if signal.action == "buy" else -1 if signal.action == "sell" else 0)
                df["signal"] = signals
                df["strategy_return"] = df["signal"].shift(1).fillna(0) * df["return"]
                total_return = df["strategy_return"].sum()
                sharpe = (
                    df["strategy_return"].mean() / (df["strategy_return"].std() + 1e-9) * (252 ** 0.5)
                    if df["strategy_return"].std() > 0
                    else 0.0
                )
                win_rate = (
                    (df["strategy_return"] > 0).sum() / max((df["strategy_return"] != 0).sum(), 1)
                )
                logging.info("Backtest complete | return=%.4f sharpe=%.4f win_rate=%.2f", total_return, sharpe, win_rate)
            else:
                backtester = Backtester(config)
                backtester.run()

        elif args.mode == "live":
            try:
                from bot_core.execution import ExecutionEngine  # type: ignore[attr-defined]
            except ImportError:
                logging.error("ExecutionEngine not implemented. Live mode unavailable.")
            else:
                engine = ExecutionEngine(config)
                engine.start()

        logging.info("Completed %s mode successfully", args.mode)
    except Exception as exc:  # pragma: no cover - broad catch to log unexpected errors
        logging.exception("Error during %s: %s", args.mode, exc)
        raise
    finally:
        shutdown_mt5()
        logging.info("MT5 connection closed")


if __name__ == "__main__":
    main()
