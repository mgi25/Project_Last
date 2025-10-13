# End-to-End Setup & Operation Guide

This document walks you through preparing the environment, connecting to an Exness MetaTrader 5 (MT5) terminal, and running each operating mode of the trading bot. The project ships only a command-line application—there is no bundled website or graphical dashboard.

## 1. Prerequisites

| Requirement | Notes |
| --- | --- |
| **Operating system** | Windows 10/11 (for MT5 terminal) or Linux for backtesting/training without live trading. |
| **Python** | Version 3.9 or later. Verify with `python --version`. |
| **MetaTrader 5** | Install the Exness MT5 terminal and make sure you can log in manually once. |
| **Broker account** | Active Exness demo or live account with login, password, and server name. |
| **Git** | Optional, but recommended for pulling updates. |
| **Virtual environment** | Recommended. |

## 2. Clone and install dependencies

```bash
# Clone the repository
git clone https://github.com/<your-org>/Project_Last.git
cd Project_Last

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Configure MT5 credentials and runtime options

1. Launch the Exness MT5 terminal and confirm the account can connect.
2. Open `config.json` in your editor.
3. Update the `account` section:
   ```json
   "account": {
     "login": 12345678,
     "password": "YOUR_PASSWORD",
     "server": "Exness-MT5Real",
     "path": "C:\\Program Files\\MetaTrader 5 EXNESS\\terminal64.exe"
   }
   ```
   * `path` must point to the MT5 terminal executable on your machine.
4. Adjust symbols, risk settings, and other options to match your strategy.

## 4. Optional: Download historical data

If you plan to run backtests without connecting to MT5, place CSV files inside the `data/` directory. The default pipeline also downloads data automatically when MT5 is available.

## 5. Run the training, tuning, and backtest pipelines

All commands are executed from the project root (`Project_Last/`). The `--mode` flag selects the workflow.

```bash
# Train or retrain the machine-learning models
python -m bot_core.main --mode train

# Run hyperparameter tuning (Newton + Optuna Bayesian search)
python -m bot_core.main --mode tune

# Execute a historical backtest using the latest model and strategy config
python -m bot_core.main --mode backtest
```

Logs are stored under `logs/`, and trained artifacts are saved in `models/`.

## 6. Connect to MT5 and run in live mode

```bash
# Ensure the MT5 terminal is closed before launching; the script will start it
python -m bot_core.main --mode live
```

What happens next:
1. The bot launches the MT5 terminal located at `account.path` and attempts to log in with the provided credentials.
2. Historical candles are pulled for each symbol in `config.json`.
3. The execution engine fuses strategy, ML, and risk layers, then begins monitoring live ticks.
4. Orders are placed via MT5 with automatic retry, cooldown, and position limits.

If you encounter the error `ExecutionEngine.__init__() missing 1 required positional argument: 'account'`, make sure you are on the latest commit and rerun the command; the constructor now consumes the raw configuration dictionary.

## 7. Useful operational tips

- **Switching symbols**: update the `symbols` array in `config.json` and rerun your desired mode.
- **Log monitoring**: tail `logs/session.log` while the bot runs to review activity.
- **Stopping the bot**: press `Ctrl+C`. The engine will attempt to close open positions if `close_all_on_shutdown` is `true`.
- **Scheduled retraining**: automate the `train` and `tune` commands with `Task Scheduler` (Windows) or `cron` (Linux).

## 8. Troubleshooting

| Symptom | Resolution |
| --- | --- |
| MT5 does not start | Verify the `account.path` value and that the path uses escaped backslashes (`\\`). |
| Login fails | Confirm credentials in MT5 directly, then update `config.json`. |
| Missing dependency errors | Re-run `pip install -r requirements.txt` inside your active virtual environment. |
| No trades placed | Check risk limits (`max_open_trades`, `daily_loss_limit`), and confirm the strategy is generating signals in logs. |

## 9. No web interface bundled

This repository does not include a web front end. To visualize performance you can import the CSV logs into tools like Excel, Google Sheets, or build a dashboard separately.

