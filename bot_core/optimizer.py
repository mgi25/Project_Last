"""Optimization utilities combining Newton, gradient, and Bayesian methods."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

try:
    import optuna
except Exception:  # pragma: no cover - optional dependency
    optuna = None  # type: ignore

LOGGER = logging.getLogger(__name__)


ObjectiveFunction = Callable[[np.ndarray], float]


def numerical_gradient(func: ObjectiveFunction, x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    grad = np.zeros_like(x)
    for i in range(len(x)):
        delta = np.zeros_like(x)
        delta[i] = epsilon
        grad[i] = (func(x + delta) - func(x - delta)) / (2 * epsilon)
    return grad


def numerical_hessian(func: ObjectiveFunction, x: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    n = len(x)
    hessian = np.zeros((n, n))
    fx = func(x)
    for i in range(n):
        for j in range(n):
            ei = np.zeros_like(x)
            ej = np.zeros_like(x)
            ei[i] = epsilon
            ej[j] = epsilon
            fpp = func(x + ei + ej)
            fpm = func(x + ei - ej)
            fmp = func(x - ei + ej)
            fmm = func(x - ei - ej)
            hessian[i, j] = (fpp - fpm - fmp + fmm) / (4 * epsilon ** 2)
    return hessian


@dataclass
class NewtonOptimizer:
    max_iter: int = 20
    tolerance: float = 1e-6

    def optimize(self, func: ObjectiveFunction, x0: np.ndarray) -> Tuple[np.ndarray, float]:
        x = x0.astype(float)
        for iteration in range(self.max_iter):
            grad = numerical_gradient(func, x)
            hessian = numerical_hessian(func, x)
            try:
                step = np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                LOGGER.warning("Hessian not invertible; falling back to gradient step")
                step = grad
            x_new = x - step
            if np.linalg.norm(x_new - x) < self.tolerance:
                LOGGER.debug("Newton optimizer converged in %d iterations", iteration)
                return x_new, func(x_new)
            x = x_new
        LOGGER.debug("Newton optimizer reached max iterations")
        return x, func(x)


@dataclass
class GradientDescentOptimizer:
    learning_rate: float = 0.1
    max_iter: int = 100

    def optimize(self, func: ObjectiveFunction, x0: np.ndarray) -> Tuple[np.ndarray, float]:
        x = x0.astype(float)
        for iteration in range(self.max_iter):
            grad = numerical_gradient(func, x)
            x -= self.learning_rate * grad
            if np.linalg.norm(grad) < 1e-6:
                LOGGER.debug("Gradient descent converged in %d iterations", iteration)
                break
        return x, func(x)


@dataclass
class BayesianOptimizer:
    n_trials: int = 25

    def optimize(self, objective: Callable[[Dict[str, float]], float], bounds: Dict[str, Tuple[float, float]]):
        if optuna is None:
            raise ImportError("optuna is required for Bayesian optimization")

        def optuna_objective(trial: "optuna.trial.Trial") -> float:
            params = {}
            for name, (low, high) in bounds.items():
                params[name] = trial.suggest_float(name, low, high)
            score = objective(params)
            trial.set_user_attr("score", score)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(optuna_objective, n_trials=self.n_trials)
        return study.best_params, study.best_value


def optimize_stop_levels(objective: Callable[[float, float], float], sl_init: float, tp_init: float) -> Tuple[float, float, float]:
    """Optimize stop-loss / take-profit using Newton-Raphson on log-scale."""

    def wrapped(x: np.ndarray) -> float:
        sl, tp = np.exp(x[0]), np.exp(x[1])
        return objective(sl, tp)

    optimizer = NewtonOptimizer(max_iter=15)
    x_opt, score = optimizer.optimize(wrapped, np.log(np.array([sl_init, tp_init])))
    return float(np.exp(x_opt[0])), float(np.exp(x_opt[1])), float(score)


def sharpe_ratio(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    mean = np.mean(returns)
    std = np.std(returns) + 1e-9
    return mean / std * np.sqrt(252 * 24 * 60)  # annualized per minute


def pnl_objective_factory(returns: np.ndarray) -> ObjectiveFunction:
    def objective(params: np.ndarray) -> float:
        weight = params
        return np.dot(weight, returns[: len(weight)])

    return objective
