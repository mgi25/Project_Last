"""Hybrid ML training pipeline for the Newton + AI Forex Scalping Bot."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from xgboost import XGBClassifier

LOGGER = logging.getLogger(__name__)


def ensure_directory(path: Path | str) -> Path:
    """Ensure a directory exists and return its Path object."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _load_config(config_path: str | Path) -> Dict:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    with config_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _detect_tree_method() -> str:
    """Select the most appropriate XGBoost tree method available."""
    tree_method = os.environ.get("XGB_TREE_METHOD")
    if tree_method:
        return tree_method
    try:
        import cupy  # noqa: F401

        return "gpu_hist"
    except Exception:  # pragma: no cover - GPU optional
        return "hist"


def compute_profit_factor(returns: np.ndarray, predictions: np.ndarray) -> float:
    """Compute the profit factor from realised returns and predictions."""
    trade_returns = returns * predictions
    gains = trade_returns[trade_returns > 0].sum()
    losses = -trade_returns[trade_returns < 0].sum()
    if losses == 0:
        return float(gains) if gains > 0 else 0.0
    return float(gains / losses)


def compute_win_rate(returns: np.ndarray, predictions: np.ndarray) -> float:
    """Compute the win rate based on positive realised returns."""
    executed = predictions > 0
    if not np.any(executed):
        return 0.0
    wins = (returns[executed] > 0).sum()
    return float(wins / executed.sum())


@dataclass(slots=True)
class DatasetBundle:
    """Container for split dataset components."""

    X_train_tab: np.ndarray
    X_test_tab: np.ndarray
    X_train_seq: np.ndarray
    X_test_seq: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    returns_train: np.ndarray
    returns_test: np.ndarray
    scaler: StandardScaler
    lookback: int


class FeatureBuilder:
    """Constructs advanced features from raw OHLCV symbol data."""

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.feature_columns: List[str] = []

    def _load_symbol(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [col.strip().lower() for col in df.columns]
        expected = {"time", "open", "high", "low", "close"}
        if not expected.issubset(df.columns):
            raise ValueError(f"CSV {path} missing required columns {expected - set(df.columns)}")
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
        df["symbol"] = path.stem
        return df

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_price = df["open"]
        volume = df.get("tick_volume")
        if volume is None:
            volume = df.get("volume")
        if volume is None:
            volume = pd.Series(np.nan, index=df.index)

        rsi = RSIIndicator(close=close, window=14).rsi()
        macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        atr = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
        bb = BollingerBands(close=close, window=20, window_dev=2)

        price_range = (high - low).replace(0, np.nan)

        feature_frame = pd.DataFrame(
            {
                "time": df["time"],
                "symbol": df["symbol"],
                "close": close,
                "rsi": rsi,
                "macd": macd_ind.macd(),
                "macd_signal": macd_ind.macd_signal(),
                "macd_hist": macd_ind.macd_diff(),
                "atr": atr,
                "bb_width": bb.bollinger_wband(),
                "bb_percent_b": bb.bollinger_pband(),
                "momentum": close.pct_change(periods=4),
                "volume_ratio": volume / (volume.rolling(20).mean() + 1e-9),
                "candle_body_ratio": (close - open_price) / price_range,
                "range_percentile": (close - low) / price_range,
            }
        )

        feature_frame["return_1"] = close.pct_change()
        for lag in range(1, 11):
            feature_frame[f"lag_return_{lag}"] = feature_frame["return_1"].shift(lag)
        feature_frame["rolling_mean_5"] = close.rolling(5).mean()
        feature_frame["rolling_std_5"] = close.rolling(5).std()
        feature_frame["rolling_mean_10"] = close.rolling(10).mean()
        feature_frame["rolling_std_10"] = close.rolling(10).std()

        feature_frame["target"] = (close.shift(-1) > close).astype(int)
        feature_frame["future_return"] = close.shift(-1) / close - 1.0
        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).dropna()

        self.feature_columns = [
            col
            for col in feature_frame.columns
            if col
            not in {
                "time",
                "symbol",
                "target",
                "future_return",
            }
        ]
        return feature_frame

    def build(self) -> pd.DataFrame:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        frames: List[pd.DataFrame] = []
        for path in sorted(self.data_dir.glob("*.csv")):
            try:
                raw = self._load_symbol(path)
                feats = self._compute_features(raw)
            except Exception as exc:  # pragma: no cover - data issues
                LOGGER.warning("Skipping %s due to error: %s", path.name, exc)
                continue
            frames.append(feats)
        if not frames:
            raise RuntimeError("No valid data files found for feature construction")
        dataset = pd.concat(frames, ignore_index=True)
        dataset = dataset.sort_values("time").reset_index(drop=True)
        self.feature_columns = [
            col
            for col in dataset.columns
            if col not in {"time", "symbol", "target", "future_return"}
        ]
        return dataset


class HybridModel:
    """Combine XGBoost and LSTM predictions into a hybrid ensemble."""

    def __init__(self, xgb_model: XGBClassifier, lstm_model: Sequential) -> None:
        self.xgb = xgb_model
        self.lstm = lstm_model

    def predict(self, X_tab: np.ndarray, X_seq: np.ndarray) -> np.ndarray:
        prob_xgb = self.xgb.predict_proba(X_tab)[:, 1]
        prob_lstm = self.lstm.predict(X_seq, verbose=0).flatten()
        return (prob_xgb + prob_lstm) / 2.0


class MLTrainer:
    """Main training orchestrator for the hybrid ML system."""

    def __init__(self, config: str | Path | Dict[str, Any] = "config.json") -> None:
        if isinstance(config, (str, Path)):
            self.config = _load_config(config)
        else:
            self.config = dict(config)
        ml_cfg = self.config.get("ml", {})
        self.train_ratio = ml_cfg.get("train_ratio", 0.8)
        self.optuna_trials = int(ml_cfg.get("optuna_trials", 20))
        self.lookback_default = int(ml_cfg.get("lookback", 50))
        data_dir = self.config.get("data", {}).get("save_path", "data")

        ensure_directory("logs")
        ensure_directory("models")

        self.feature_builder = FeatureBuilder(data_dir=data_dir)
        self.dataset = self.feature_builder.build()
        self.feature_columns = self.feature_builder.feature_columns
        self.study: optuna.Study | None = None
        self.best_params: Dict | None = None
        self.xgb_model: XGBClassifier | None = None
        self.lstm_model: Sequential | None = None
        self.scaler: StandardScaler | None = None

    def _prepare_datasets(self, lookback: int) -> DatasetBundle:
        features = self.dataset[self.feature_columns].values
        targets = self.dataset["target"].to_numpy()
        future_returns = self.dataset["future_return"].to_numpy()

        effective_n = len(targets) - lookback
        if effective_n <= 0:
            raise ValueError("Lookback larger than dataset length")

        split_idx = max(int(effective_n * self.train_ratio), 1)
        scaler = StandardScaler()
        scaler.fit(features[lookback : lookback + split_idx])
        features_scaled = scaler.transform(features)

        sequences = []
        for idx in range(lookback, len(features_scaled)):
            sequences.append(features_scaled[idx - lookback : idx])
        sequences = np.asarray(sequences, dtype=np.float32)

        X_tab = features_scaled[lookback:]
        y = targets[lookback:]
        returns = future_returns[lookback:]

        X_train_tab = X_tab[:split_idx]
        X_test_tab = X_tab[split_idx:]
        X_train_seq = sequences[:split_idx]
        X_test_seq = sequences[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        returns_train = returns[:split_idx]
        returns_test = returns[split_idx:]

        return DatasetBundle(
            X_train_tab=X_train_tab,
            X_test_tab=X_test_tab,
            X_train_seq=X_train_seq,
            X_test_seq=X_test_seq,
            y_train=y_train,
            y_test=y_test,
            returns_train=returns_train,
            returns_test=returns_test,
            scaler=scaler,
            lookback=lookback,
        )

    def _build_lstm_model(
        self,
        input_shape: Tuple[int, int],
        learning_rate: float,
        dropout: float,
    ) -> Sequential:
        model = Sequential(
            [
                LSTM(128, return_sequences=True, input_shape=input_shape),
                Dropout(dropout),
                LSTM(64),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _train_xgb(
        self,
        bundle: DatasetBundle,
        params: Dict[str, float | int | str],
    ) -> XGBClassifier:
        tree_method = _detect_tree_method()
        clf = XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            min_child_weight=float(params["min_child_weight"]),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method=tree_method,
            use_label_encoder=False,
            scale_pos_weight=float(params.get("scale_pos_weight", 1.0)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
        )
        clf.fit(
            bundle.X_train_tab,
            bundle.y_train,
            eval_set=[(bundle.X_test_tab, bundle.y_test)],
            verbose=False,
            early_stopping_rounds=25,
        )
        return clf

    def _train_lstm(
        self,
        bundle: DatasetBundle,
        learning_rate: float,
        dropout: float,
        batch_size: int,
        epochs: int = 20,
    ) -> Sequential:
        model = self._build_lstm_model(
            input_shape=(bundle.X_train_seq.shape[1], bundle.X_train_seq.shape[2]),
            learning_rate=learning_rate,
            dropout=dropout,
        )
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
        model.fit(
            bundle.X_train_seq,
            bundle.y_train,
            validation_data=(bundle.X_test_seq, bundle.y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0,
        )
        return model

    def _evaluate(
        self,
        hybrid: HybridModel,
        bundle: DatasetBundle,
    ) -> Dict[str, float]:
        probs = hybrid.predict(bundle.X_test_tab, bundle.X_test_seq)
        predictions = (probs > 0.5).astype(int)
        accuracy = accuracy_score(bundle.y_test, predictions)
        precision = precision_score(bundle.y_test, predictions, zero_division=0)
        recall = recall_score(bundle.y_test, predictions, zero_division=0)
        f1 = f1_score(bundle.y_test, predictions, zero_division=0)
        profit_factor = compute_profit_factor(bundle.returns_test, predictions)
        win_rate = compute_win_rate(bundle.returns_test, predictions)
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "profit_factor": float(profit_factor),
            "win_rate": float(win_rate),
        }

    def optimize(self) -> None:
        LOGGER.info("Starting Optuna optimization with %d trials", self.optuna_trials)

        def objective(trial: optuna.Trial) -> float:
            lookback = trial.suggest_int("lookback", 40, 70)
            bundle = self._prepare_datasets(lookback)

            pos = max(bundle.y_train.sum(), 1)
            neg = max(len(bundle.y_train) - bundle.y_train.sum(), 1)
            scale_pos_weight = neg / pos

            xgb_params = {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 200, 600),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
                "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
                "min_child_weight": trial.suggest_float("xgb_min_child_weight", 1.0, 10.0),
                "scale_pos_weight": scale_pos_weight,
            }

            lstm_lr = trial.suggest_float("lstm_learning_rate", 1e-4, 5e-3, log=True)
            lstm_dropout = trial.suggest_float("lstm_dropout", 0.2, 0.5)
            lstm_batch = trial.suggest_categorical("lstm_batch_size", [32, 64, 128])

            try:
                xgb_model = self._train_xgb(bundle, xgb_params)
                lstm_model = self._train_lstm(
                    bundle,
                    learning_rate=lstm_lr,
                    dropout=lstm_dropout,
                    batch_size=int(lstm_batch),
                    epochs=20,
                )
            except Exception as exc:  # pragma: no cover - training errors
                LOGGER.warning("Trial failed due to error: %s", exc)
                raise optuna.exceptions.TrialPruned() from exc

            hybrid = HybridModel(xgb_model, lstm_model)
            metrics = self._evaluate(hybrid, bundle)

            score = metrics["f1"] * metrics["profit_factor"] * metrics["win_rate"]
            trial.set_user_attr("metrics", metrics)
            return score

        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False)
        self.best_params = {
            "lookback": self.study.best_params["lookback"],
            "xgb": {
                "n_estimators": self.study.best_params["xgb_n_estimators"],
                "max_depth": self.study.best_params["xgb_max_depth"],
                "learning_rate": self.study.best_params["xgb_learning_rate"],
                "subsample": self.study.best_params["xgb_subsample"],
                "colsample_bytree": self.study.best_params["xgb_colsample"],
                "min_child_weight": self.study.best_params["xgb_min_child_weight"],
            },
            "lstm": {
                "learning_rate": self.study.best_params["lstm_learning_rate"],
                "dropout": self.study.best_params["lstm_dropout"],
                "batch_size": self.study.best_params["lstm_batch_size"],
            },
            "score": self.study.best_value,
            "metrics": self.study.best_trial.user_attrs.get("metrics", {}),
        }
        ensure_directory("logs")
        with Path("logs/optuna_results.json").open("w", encoding="utf-8") as handle:
            json.dump(self.best_params, handle, indent=2)
        LOGGER.info("Optuna optimization complete. Best score: %.4f", self.study.best_value)

    def train_all(self) -> None:
        if self.study is None:
            self.optimize()
        assert self.best_params is not None

        lookback = int(self.best_params.get("lookback", self.lookback_default))
        bundle = self._prepare_datasets(lookback)
        xgb_params = dict(self.best_params["xgb"])
        pos = max(bundle.y_train.sum(), 1)
        neg = max(len(bundle.y_train) - bundle.y_train.sum(), 1)
        xgb_params["scale_pos_weight"] = neg / pos
        xgb_params.setdefault("reg_lambda", 1.0)

        self.scaler = bundle.scaler
        self.xgb_model = self._train_xgb(bundle, xgb_params)
        self.lstm_model = self._train_lstm(
            bundle,
            learning_rate=float(self.best_params["lstm"]["learning_rate"]),
            dropout=float(self.best_params["lstm"]["dropout"]),
            batch_size=int(self.best_params["lstm"]["batch_size"]),
            epochs=20,
        )
        hybrid = HybridModel(self.xgb_model, self.lstm_model)
        metrics = self._evaluate(hybrid, bundle)

        thresholds = {
            "accuracy": 0.70,
            "f1": 0.68,
            "win_rate": 0.65,
            "profit_factor": 1.8,
        }
        for metric, target in thresholds.items():
            value = metrics.get(metric, 0.0)
            if value < target:
                LOGGER.warning(
                    "Metric %s=%.3f below target %.2f", metric, value, target
                )
            else:
                LOGGER.info("Metric %s=%.3f meets target %.2f", metric, value, target)

        ensure_directory("logs")
        metrics_path = Path("logs/ml_training_metrics.json")
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        ensure_directory("models")
        joblib.dump(self.xgb_model, "models/model.pkl")
        joblib.dump(self.scaler, "models/scaler.pkl")
        self.lstm_model.save("models/lstm_model.h5")

        LOGGER.info("✅ Model trained successfully")
        LOGGER.info("Saved models:\n   - models/model.pkl\n   - models/lstm_model.h5")
        print("✅ Model trained successfully")
        print("Saved models:\n   - models/model.pkl\n   - models/lstm_model.h5")


__all__ = ["FeatureBuilder", "HybridModel", "MLTrainer"]
