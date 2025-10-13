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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
    decision_threshold: float = 0.55

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


if torch is not None:

    class LSTMClassifier(nn.Module):  # type: ignore[misc]
        def __init__(self, input_size: int, hidden_size: int = 32) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 2)

        def forward(self, x):  # type: ignore[override]
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out)

else:

    class LSTMClassifier:  # type: ignore[misc]
        def __init__(self, *_args, **_kwargs) -> None:
            raise ImportError("PyTorch is required for LSTM model")


class MLPredictor:
    def __init__(self, config: MLConfig, model_dir: str = "models") -> None:
        self.config = config
        self.model_dir = Path(model_dir)
        self.scaler = StandardScaler()
        self.model: RandomForestClassifier | LSTMClassifier | None = None
        self.feature_columns: List[str] = (
            self.config.feature_list
            if self.config.feature_list
            else [
                "return",
                "high_low",
                "body",
                "rsi",
                "volatility",
                "volume",
                "macd",
                "atr",
                "bollinger_width",
                "volume_ratio",
                "price_change",
                "candle_body",
                "momentum",
            ]
        )
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
        self, df: pd.DataFrame, with_target: bool = True
    ) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        df = df.copy()
        lookback = max(int(self.config.features_lookback), 5)
        df["return"] = df["close"].pct_change()
        df["high_low"] = df["high"] - df["low"]
        df["body"] = df["close"] - df["open"]
        df["price_change"] = df["close"].pct_change().fillna(0)
        df["candle_body"] = df["close"] - df["open"]
        df["momentum"] = df["close"].pct_change(periods=min(lookback, 10)).fillna(0)
        df["rsi"] = df["close"].rolling(14).apply(lambda x: self._rsi(x), raw=False)
        df["volatility"] = df["return"].rolling(lookback).std().fillna(0)
        volume = df.get("tick_volume")
        if volume is None:
            volume = df.get("volume", pd.Series(0, index=df.index))
        df["volume"] = volume.fillna(0)
        df["volume_ratio"] = df["volume"] / (df["volume"].rolling(lookback).mean() + 1e-9)
        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = (ema_fast - ema_slow).fillna(0)
        rolling_mean = df["close"].rolling(20)
        mean_close = rolling_mean.mean()
        std_close = rolling_mean.std()
        df["bollinger_width"] = ((mean_close + 2 * std_close) - (mean_close - 2 * std_close)) / (df["close"] + 1e-9)
        high_low = df[["high", "low", "close"]].copy()
        high_low["prev_close"] = high_low["close"].shift(1)
        tr_components = pd.concat(
            [
                high_low["high"] - high_low["low"],
                (high_low["high"] - high_low["prev_close"]).abs(),
                (high_low["low"] - high_low["prev_close"]).abs(),
            ],
            axis=1,
        )
        df["atr"] = tr_components.max(axis=1).rolling(window=14, min_periods=14).mean()
        df = df.replace([np.inf, -np.inf], np.nan)

        feature_columns = [col for col in self.feature_columns if col in df.columns]
        if with_target:
            df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
            df["future_return"] = df["close"].pct_change().shift(-1)
            df = df.dropna(subset=feature_columns + ["target", "future_return"])
            features = df[feature_columns].values
            return features, df["target"].astype(int).values, df["future_return"].values
        df = df.dropna(subset=feature_columns)
        return df[feature_columns].values, None, None

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
        X, y, future_returns = self._prepare_features(df, with_target=True)
        if len(X) < 50:
            raise ValueError("Insufficient data for training")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size(), shuffle=False
        )
        future_returns_test = future_returns[len(X_train) :]
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        metrics: Dict[str, float] = {}
        if self.config.model_type == "lstm":
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
            metrics.update({"accuracy": float(accuracy), "model_type": "lstm"})
            return metrics
        candidate_models = {
            "random_forest": RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=2,
                random_state=self.config.random_state,
                class_weight="balanced",
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=600,
                max_depth=None,
                random_state=self.config.random_state,
                class_weight="balanced",
            ),
            "hist_gradient_boosting": HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=500,
                max_depth=6,
                random_state=self.config.random_state,
            ),
            "logistic_regression": LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=self.config.random_state,
            ),
        }

        preferred_models = (
            [self.config.model_type]
            if self.config.model_type in candidate_models
            else list(candidate_models.keys())
        )

        best_model_name = ""
        best_profit_factor = -np.inf
        best_accuracy = -np.inf
        best_threshold = self.config.decision_threshold
        evaluation_summary: Dict[str, Dict[str, float]] = {}

        for name in preferred_models:
            model = clone(candidate_models[name])
            model.fit(X_train_scaled, y_train)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test_scaled)
                prob_up = proba[:, 1]
            else:
                decision = model.decision_function(X_test_scaled)
                prob_up = 1 / (1 + np.exp(-decision))
            y_pred = (prob_up >= 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            thresholds = np.linspace(0.5, 0.75, 11)
            model_best_pf = -np.inf
            model_best_threshold = best_threshold
            model_best_win_rate = 0.0
            model_best_avg_return = 0.0
            for threshold in thresholds:
                long_mask = prob_up >= threshold
                short_mask = prob_up <= 1 - threshold
                signals = np.zeros_like(prob_up)
                signals[long_mask] = 1
                signals[short_mask] = -1
                active_returns = future_returns_test[signals != 0] * signals[signals != 0]
                if active_returns.size == 0:
                    continue
                positive = active_returns[active_returns > 0].sum()
                negative = -active_returns[active_returns < 0].sum()
                profit_factor = positive / (negative + 1e-9)
                win_rate = float((active_returns > 0).sum() / len(active_returns))
                avg_return = float(active_returns.mean())
                if profit_factor > model_best_pf or (
                    np.isclose(profit_factor, model_best_pf) and win_rate > model_best_win_rate
                ):
                    model_best_pf = float(profit_factor)
                    model_best_threshold = float(threshold)
                    model_best_win_rate = win_rate
                    model_best_avg_return = avg_return
            evaluation_summary[name] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "profit_factor": float(model_best_pf if model_best_pf != -np.inf else 0.0),
                "win_rate": float(model_best_win_rate),
                "avg_return": float(model_best_avg_return),
                "threshold": float(model_best_threshold),
            }
            if model_best_pf > best_profit_factor or (
                np.isclose(model_best_pf, best_profit_factor) and accuracy > best_accuracy
            ):
                best_profit_factor = model_best_pf
                best_accuracy = accuracy
                best_model_name = name
                best_threshold = model_best_threshold

        if not best_model_name:
            raise ValueError("No suitable model found during training")

        LOGGER.info(
            "Selected ML model %s | accuracy=%.3f profit_factor=%.3f win_rate=%.2f threshold=%.2f",
            best_model_name,
            evaluation_summary[best_model_name]["accuracy"],
            evaluation_summary[best_model_name]["profit_factor"],
            evaluation_summary[best_model_name]["win_rate"],
            best_threshold,
        )

        # Refit on all data using the best model configuration
        self.config.model_type = best_model_name
        self.config.decision_threshold = best_threshold
        self.scaler.fit(X)
        X_scaled_full = self.scaler.transform(X)
        final_model = clone(candidate_models[best_model_name])
        final_model.fit(X_scaled_full, y)
        self.model = final_model

        payload = {
            "model": self.model,
            "model_type": best_model_name,
            "features": self.feature_columns,
            "threshold": self.config.decision_threshold,
        }
        if self.scaler_path is not None:
            joblib.dump(payload, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
        else:
            payload["scaler"] = self.scaler
            joblib.dump(payload, self.model_path)

        metrics = evaluation_summary[best_model_name]
        metrics["model_type"] = best_model_name
        return metrics

    def load(self) -> None:
        if not self.model_path.exists():
            LOGGER.warning("ML model file %s does not exist", self.model_path)
            return
        if self.scaler_path is not None and self.scaler_path.exists():
            payload = joblib.load(self.model_path)
            self.model = payload.get("model")
            self.scaler = joblib.load(self.scaler_path)
        else:
            payload = joblib.load(self.model_path)
            if isinstance(payload, dict):
                self.model = payload.get("model")
                scaler = payload.get("scaler")
                if scaler is not None:
                    self.scaler = scaler
            else:
                self.model = payload
                scaler_fallback = self.model_path.with_name(f"{self.model_path.stem}_scaler.pkl")
                if scaler_fallback.exists():
                    self.scaler = joblib.load(scaler_fallback)
        if isinstance(payload, dict):
            model_type = payload.get("model_type")
            if model_type:
                self.config.model_type = model_type
            stored_features = payload.get("features")
            if stored_features:
                self.feature_columns = list(stored_features)
            threshold = payload.get("threshold")
            if threshold is not None:
                self.config.decision_threshold = float(threshold)
        if self.config.model_type == "lstm":
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
        X, _, _ = self._prepare_features(df, with_target=False)
        if len(X) == 0:
            return {"p_up": 0.5, "p_down": 0.5}
        X_scaled = self.scaler.transform(X)
        latest = X_scaled[-1:]
        if isinstance(self.model, RandomForestClassifier):
            prob = self.model.predict_proba(latest)[0]
        elif isinstance(self.model, (ExtraTreesClassifier, HistGradientBoostingClassifier, LogisticRegression)):
            prob = self.model.predict_proba(latest)[0]
        else:
            if torch is None:
                raise ImportError("PyTorch is required for LSTM model")
            with torch.no_grad():
                tensor = torch.tensor(latest, dtype=torch.float32).unsqueeze(1)
                logits = self.model(tensor)
                prob = torch.softmax(logits, dim=1).numpy()[0]
        return {"p_down": float(prob[0]), "p_up": float(prob[1])}
