# Newton + AI Hybrid Forex Scalping Bot

This repository contains a fully modular trading system for MetaTrader 5 (MT5) focused on high-frequency scalping on the 1-minute timeframe. The bot combines mathematical optimization, classical quantitative strategies, machine learning, and reinforcement learning to trade major forex pairs using the Exness broker.

## Project Structure
```
bot_core/
    main.py               # Entry point and orchestration
    optimizer.py          # Newton, gradient, Bayesian optimizers
    strategy.py           # Trend, mean-reversion, breakout strategies
    ml_model.py           # Machine-learning model management
    rl_agent.py           # Reinforcement learning agent interface
    risk.py               # Risk management utilities
    execution.py          # MT5 execution and simulation layer
    logger.py             # Trade and session logging
    utils.py              # Shared helpers and indicators
config.json               # Runtime configuration
requirements.txt          # Python dependencies
data/                    # Historical data cache
logs/                     # Trading and training logs
models/                   # Persisted ML/RL models
```

## Features
- **Data Handling:** Connects to MetaTrader 5 for tick and OHLC data retrieval or uses locally cached data in backtesting mode.
- **Optimization Stack:** Newton–Raphson optimizer for fast parameter tuning, gradient and Bayesian optimization for global search.
- **Trading Strategies:** Multiple engines (trend, mean reversion, breakout) with a fusion ensemble that weights each engine by recent accuracy.
- **Machine Learning:** Configurable pipeline supporting Random Forest or LSTM models with automated retraining from trading logs.
- **Reinforcement Learning:** PPO/DDPG integration using `stable-baselines3` with offline and online learning support.
- **Risk Controls:** Kelly criterion-based sizing, Newton-optimized SL/TP, trailing stops, daily loss/profit caps, and panic exits.
- **Execution Layer:** Robust MT5 order management with retry logic and a built-in simulation mode.
- **Continuous Learning:** Automated daily retraining and parameter re-optimization with persisted artifacts in `models/`.

## Usage
1. Install dependencies: `pip install -r requirements.txt`.
2. Update `config.json` with your MetaTrader 5 account credentials and desired parameters.
3. Run the bot:
   ```bash
   python -m bot_core.main --config config.json
   ```
4. Use `--mode backtest` for historical simulation or `--mode live` for live trading.

## Disclaimer
Trading foreign exchange on margin carries a high level of risk and may not be suitable for all investors. This project is provided for educational purposes only. Use at your own risk.
