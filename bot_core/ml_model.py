"""Machine learning model management for directional prediction."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class MLConfig:
    """Configuration container for the ML predictor."""

    model_type: str = "random_forest"
    features_lookback: int = 50
    retrain_interval: str = "1d"
    train_test_split: float = 0.2
    random_state: int | None = 42
    model_save_path: str | None = None
    scaler_save_path: str | None = None
    feature_list: list[str] | None = None
    evaluation_metric: str = "f1"

    def test_size(self) -> float:
        """Return the test split proportion derived from ``train_test_split``."""

        ratio = self.train_test_split
        if not 0.0 < ratio < 1.0:
            return 0.2
        # If the ratio is greater than 0.5 we interpret it as the training share
        # to maintain backwards compatibility with configs using ``train_test_split``
        # to mean training size (e.g. 0.8 for 80% train / 20% test).
        if ratio > 0.5:
            ratio = 1.0 - ratio
        return max(min(ratio, 0.95), 0.05)


class LSTMClassifier(nn.Module):  # type: ignore[misc]
    def __init__(self, input_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):  # type: ignore[override]
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class MLPredictor:
    def __init__(self, config: MLConfig, model_dir: str = "models") -> None:
        self.config = config
        self.model_dir = Path(model_dir)
        self.scaler = StandardScaler()
        self.model: RandomForestClassifier | GradientBoostingClassifier | HistGradientBoostingClassifier | LogisticRegression | LSTMClassifier | None = None
        self.feature_names: List[str] = []
        self.selected_model_type: str = self.config.model_type
        if self.config.model_save_path:
            self.model_path = Path(self.config.model_save_path)
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.model_path = self.model_dir / f"ml_{self.config.model_type}.pkl"
        if self.config.scaler_save_path:
            self.scaler_path = Path(self.config.scaler_save_path)
            self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.scaler_path = None

    def _prepare_features(
        self, df: pd.DataFrame, include_target: bool = True
    ) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        df = df.copy()
        df["return"] = df["close"].pct_change()
        df["high_low"] = df["high"] - df["low"]
        df["candle_body"] = df["close"] - df["open"]
        df["momentum"] = df["close"].pct_change(5)
        df["volatility"] = (
            df["return"].rolling(self.config.features_lookback).std().fillna(0)
        )
        df["rsi"] = df["close"].rolling(14).apply(lambda x: self._rsi(x), raw=False)
        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        rolling_mean = df["close"].rolling(20)
        mid = rolling_mean.mean()
        std = df["close"].rolling(20).std(ddof=0)
        upper = mid + 2 * std
        lower = mid - 2 * std
        df["bollinger_width"] = (upper - lower) / (mid + 1e-9)
        df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
        volume_series = df.get("tick_volume")
        if volume_series is None:
            volume_series = df.get("volume", pd.Series(0, index=df.index))
        df["volume"] = volume_series.fillna(0)
        volume_sma = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / (volume_sma + 1e-9)
        df["price_change"] = df["close"].pct_change()
        df["future_return"] = df["close"].pct_change().shift(-1)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        available_features = {
            "return": df["return"],
            "high_low": df["high_low"],
            "candle_body": df["candle_body"],
            "rsi": df["rsi"],
            "volatility": df["volatility"],
            "volume": df["volume"],
            "momentum": df["momentum"],
            "macd": df["macd"],
            "macd_signal": df["macd_signal"],
            "macd_hist": df["macd_hist"],
            "bollinger_width": df["bollinger_width"],
            "volume_ratio": df["volume_ratio"],
            "price_change": df["price_change"],
            "atr": df["atr"],
        }

        if self.feature_names:
            selected = [name for name in self.feature_names if name in available_features]
        elif self.config.feature_list:
            selected = [name for name in self.config.feature_list if name in available_features]
            missing = set(self.config.feature_list) - set(selected)
            if missing:
                LOGGER.warning("Ignoring unavailable features: %s", ", ".join(sorted(missing)))
        else:
            selected = [
                "return",
                "high_low",
                "candle_body",
                "rsi",
                "volatility",
                "volume",
                "momentum",
                "macd",
                "macd_hist",
                "bollinger_width",
                "volume_ratio",
                "price_change",
                "atr",
            ]
        if not selected:
            raise ValueError("No valid features selected for model training")

        self.feature_names = selected
        feature_frame = pd.DataFrame({name: available_features[name] for name in selected})

        if include_target:
            future_returns = df["future_return"].values
            y = (future_returns > 0).astype(int)
            return feature_frame.values, y, future_returns
        return feature_frame.values, None, None

    @staticmethod
    def _rsi(series: pd.Series) -> float:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.mean()
        roll_down = down.mean()
        rs = roll_up / (roll_down + 1e-9)
        return 100 - (100 / (1 + rs))

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        X, y, future_returns = self._prepare_features(df)
        if y is None or future_returns is None or len(X) < 50:
            raise ValueError("Insufficient data for training")

        if self.config.model_type == "lstm":
            return self._train_lstm(X, y)

        test_size = self.config.test_size()
        split_index = int(len(X) * (1 - test_size))
        if split_index <= 0 or split_index >= len(X):
            raise ValueError("Invalid train/test split resulting in empty partitions")
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        test_future_returns = future_returns[split_index:]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        model, metrics = self._fit_model(X_train_scaled, y_train, X_test_scaled, y_test, test_future_returns)

        # Refit on the complete dataset for deployment
        self.scaler.fit(X)
        X_full_scaled = self.scaler.transform(X)
        self.model = clone(model).fit(X_full_scaled, y)

        self._persist_model()
        return metrics

    def _train_lstm(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if torch is None:
            raise ImportError("PyTorch is required for LSTM model")
        test_size = self.config.test_size()
        split_index = int(len(X) * (1 - test_size))
        if split_index <= 0 or split_index >= len(X):
            raise ValueError("Invalid train/test split resulting in empty partitions")
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        input_size = X_train_scaled.shape[1]
        model = LSTMClassifier(input_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        for _ in range(75):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
            outputs = model(X_test_tensor)
            preds = outputs.argmax(dim=1).numpy()
        accuracy = accuracy_score(y_test, preds)
        self.model = model
        payload = {
            "model_state": model.state_dict(),
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_type": "lstm",
            "input_size": input_size,
        }
        torch.save(payload, self.model_path)
        if self.scaler_path is not None:
            joblib.dump(self.scaler, self.scaler_path)
        LOGGER.info("LSTM trained with accuracy %.3f", accuracy)
        return {"accuracy": float(accuracy), "model_type": "lstm"}

    def _fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_future_returns: np.ndarray,
    ) -> Tuple[
        RandomForestClassifier | GradientBoostingClassifier | HistGradientBoostingClassifier | LogisticRegression,
        Dict[str, float],
    ]:
        if self.config.model_type == "lstm":
            raise ValueError("LSTM training handled separately")

        candidates: Dict[str, RandomForestClassifier | GradientBoostingClassifier | HistGradientBoostingClassifier | LogisticRegression] = {
            "random_forest": RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=5,
                random_state=self.config.random_state,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                learning_rate=0.05, n_estimators=300, max_depth=3, random_state=self.config.random_state
            ),
            "hist_gradient_boosting": HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.05,
                max_iter=400,
                random_state=self.config.random_state,
            ),
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=self.config.random_state,
            ),
        }

        if self.config.model_type == "auto":
            best_name = None
            best_score = -np.inf
            best_model = None
            for name, estimator in candidates.items():
                estimator.fit(X_train, y_train)
                preds = estimator.predict(X_test)
                score = self._metric_score(y_test, preds)
                LOGGER.info("Auto-eval %s -> %s=%.4f", name, self.config.evaluation_metric, score)
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_model = estimator
            if best_model is None or best_name is None:
                raise RuntimeError("Auto model selection failed")
            self.selected_model_type = best_name
            LOGGER.info("Selected model type: %s", best_name)
            base_model = best_model
        else:
            if self.config.model_type not in candidates:
                raise ValueError(f"Unsupported model type {self.config.model_type}")
            base_model = candidates[self.config.model_type]
            base_model.fit(X_train, y_train)
            self.selected_model_type = self.config.model_type

        y_pred = base_model.predict(X_test)
        proba = (
            base_model.predict_proba(X_test)[:, 1]
            if hasattr(base_model, "predict_proba")
            else (y_pred.astype(float))
        )
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        strategy_returns = np.where(y_pred == 1, 1, -1) * test_future_returns
        gross_profit = strategy_returns[strategy_returns > 0].sum()
        gross_loss = -strategy_returns[strategy_returns < 0].sum()
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        win_rate = float(np.mean(strategy_returns > 0)) if len(strategy_returns) else 0.0
        avg_confidence = float(np.mean(np.abs(proba - 0.5) * 2))

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "avg_confidence": avg_confidence,
            "model_type": self.selected_model_type,
        }
        LOGGER.info(
            "Model %s metrics | acc=%.3f prec=%.3f rec=%.3f f1=%.3f pf=%.2f win=%.2f",
            self.selected_model_type,
            accuracy,
            precision,
            recall,
            f1,
            profit_factor,
            win_rate,
        )
        return base_model, metrics

    def _metric_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        metric = self.config.evaluation_metric.lower()
        if metric == "precision":
            return precision_score(y_true, y_pred, zero_division=0)
        if metric == "recall":
            return recall_score(y_true, y_pred, zero_division=0)
        if metric == "accuracy":
            return accuracy_score(y_true, y_pred)
        return f1_score(y_true, y_pred, zero_division=0)

    def _persist_model(self) -> None:
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_type": self.selected_model_type,
        }
        joblib.dump(payload, self.model_path)
        if self.scaler_path is not None:
            joblib.dump(self.scaler, self.scaler_path)

    def load(self) -> None:
        if not self.model_path.exists():
            LOGGER.warning("ML model file %s does not exist", self.model_path)
            return
        data = None
        try:
            data = joblib.load(self.model_path)
        except Exception:
            if torch is None:
                raise
            data = torch.load(self.model_path, map_location="cpu")

        if isinstance(data, dict) and "model_state" in data:
            if torch is None:
                raise ImportError("PyTorch is required for LSTM model")
            input_size = data.get("input_size", len(self.feature_names) or 6)
            model = LSTMClassifier(input_size)
            model.load_state_dict(data["model_state"])
            self.model = model
            if "scaler" in data:
                self.scaler = data["scaler"]
            self.feature_names = data.get("feature_names", self.feature_names)
            self.selected_model_type = data.get("model_type", "lstm")
        elif isinstance(data, dict) and "model" in data:
            self.model = data["model"]
            if "scaler" in data:
                self.scaler = data["scaler"]
            if "feature_names" in data:
                self.feature_names = data["feature_names"]
            self.selected_model_type = data.get("model_type", self.selected_model_type)
        else:
            self.model = data
            if self.scaler_path is not None and self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
            else:
                scaler_fallback = self.model_path.with_name(
                    f"{self.model_path.stem}_scaler.pkl"
                )
                if scaler_fallback.exists():
                    self.scaler = joblib.load(scaler_fallback)
        LOGGER.info("Loaded ML model from %s", self.model_path)

    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        if self.model is None:
            LOGGER.warning("ML model not loaded; returning neutral predictions")
            return {"p_up": 0.5, "p_down": 0.5, "bias": "hold", "confidence": 0.0}
        X, _, _ = self._prepare_features(df, include_target=False)
        if len(X) == 0:
            return {"p_up": 0.5, "p_down": 0.5, "bias": "hold", "confidence": 0.0}
        X_scaled = self.scaler.transform(X)
        latest = X_scaled[-1:]
        if isinstance(self.model, LSTMClassifier):
            if torch is None:
                raise ImportError("PyTorch is required for LSTM model")
            with torch.no_grad():
                tensor = torch.tensor(latest, dtype=torch.float32).unsqueeze(1)
                logits = self.model(tensor)
                prob = torch.softmax(logits, dim=1).numpy()[0]
        elif hasattr(self.model, "predict_proba"):
            prob = self.model.predict_proba(latest)[0]
        else:
            decision = self.model.decision_function(latest)  # type: ignore[call-arg]
            prob_up = 1 / (1 + np.exp(-decision))
            prob = np.array([1 - prob_up, prob_up]).reshape(-1)
        p_down = float(prob[0])
        p_up = float(prob[1])
        confidence = abs(p_up - p_down)
        bias = "buy" if p_up >= p_down else "sell"
        return {
            "p_down": p_down,
            "p_up": p_up,
            "bias": bias,
            "confidence": confidence,
            "model_type": self.selected_model_type,
        }
