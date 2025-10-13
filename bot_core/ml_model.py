"""Machine learning model management for directional prediction."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
        self.model: RandomForestClassifier | LSTMClassifier | None = None
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

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = df.copy()
        df["return"] = df["close"].pct_change()
        df["high_low"] = df["high"] - df["low"]
        df["body"] = df["close"] - df["open"]
        df["rsi"] = df["close"].rolling(14).apply(lambda x: self._rsi(x), raw=False)
        df["volatility"] = df["close"].pct_change().rolling(self.config.features_lookback).std().fillna(0)
        df["volume"] = df.get("tick_volume", 0)
        df = df.dropna()
        y = (df["close"].shift(-1) > df["close"]).astype(int)[:-1]
        X = df.iloc[:-1]
        features = X[["return", "high_low", "body", "rsi", "volatility", "volume"]].values
        return features, y.values

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
        X, y = self._prepare_features(df)
        if len(X) < 50:
            raise ValueError("Insufficient data for training")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size(), shuffle=False
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        if self.config.model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=200, max_depth=5, random_state=self.config.random_state
            )
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            self.model = model
            if self.scaler_path is not None:
                joblib.dump(model, self.model_path)
                joblib.dump(self.scaler, self.scaler_path)
            else:
                joblib.dump({"model": model, "scaler": self.scaler}, self.model_path)
            LOGGER.info("RandomForest trained with accuracy %.3f", accuracy)
            return {"accuracy": float(accuracy)}
        elif self.config.model_type == "lstm":
            if torch is None:
                raise ImportError("PyTorch is required for LSTM model")
            input_size = X_train_scaled.shape[1]
            model = LSTMClassifier(input_size)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            for _ in range(50):
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
            torch.save({"model_state": model.state_dict(), "scaler": self.scaler}, self.model_path)
            self.model = model
            LOGGER.info("LSTM trained with accuracy %.3f", accuracy)
            return {"accuracy": float(accuracy)}
        else:
            raise ValueError(f"Unsupported model type {self.config.model_type}")

    def load(self) -> None:
        if not self.model_path.exists():
            LOGGER.warning("ML model file %s does not exist", self.model_path)
            return
        if self.config.model_type == "random_forest":
            if self.scaler_path is not None and self.scaler_path.exists():
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
            else:
                data = joblib.load(self.model_path)
                if isinstance(data, dict) and "model" in data:
                    self.model = data["model"]
                    self.scaler = data["scaler"]
                else:
                    self.model = data
                    scaler_fallback = self.model_path.with_name(
                        f"{self.model_path.stem}_scaler.pkl"
                    )
                    if scaler_fallback.exists():
                        self.scaler = joblib.load(scaler_fallback)
        elif self.config.model_type == "lstm":
            if torch is None:
                raise ImportError("PyTorch is required for LSTM model")
            data = torch.load(self.model_path, map_location="cpu")
            input_size = data.get("input_size", 6)
            model = LSTMClassifier(input_size)
            model.load_state_dict(data["model_state"])
            self.model = model
            self.scaler = data["scaler"]
        LOGGER.info("Loaded ML model from %s", self.model_path)

    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        if self.model is None:
            LOGGER.warning("ML model not loaded; returning neutral predictions")
            return {"p_up": 0.5, "p_down": 0.5}
        X, _ = self._prepare_features(df)
        if len(X) == 0:
            return {"p_up": 0.5, "p_down": 0.5}
        X_scaled = self.scaler.transform(X)
        latest = X_scaled[-1:]
        if isinstance(self.model, RandomForestClassifier):
            prob = self.model.predict_proba(latest)[0]
        else:
            if torch is None:
                raise ImportError("PyTorch is required for LSTM model")
            with torch.no_grad():
                tensor = torch.tensor(latest, dtype=torch.float32).unsqueeze(1)
                logits = self.model(tensor)
                prob = torch.softmax(logits, dim=1).numpy()[0]
        return {"p_down": float(prob[0]), "p_up": float(prob[1])}
