"""Registro declarativo de candidatos de pronostico."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[int], Any]
    resolution: str
    supports_exogenous: bool
    backend: str
    family: str = "sklearn"


def _sklearn_factory(name: str):
    def create(seed: int):
        if name == "Ridge":
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0)
        if name == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=1)
        if name == "MLP":
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(hidden_layer_sizes=(32, 16), random_state=seed, max_iter=200)
        if name == "XGBoost":
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=100, random_state=seed, tree_method="hist")
        if name == "LightGBM":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=100, random_state=seed, verbosity=-1)
        if name == "CatBoost":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(iterations=100, random_seed=seed, verbose=False)
        raise KeyError(name)
    return create


def _forecast_factory(name: str, seasonal_periods: int = 7):
    def create(seed: int):
        from prodma.forecast_models import create_forecaster
        kwargs = {"seasonal_periods": seasonal_periods}
        if name == "LSTM":
            kwargs.update(device="auto", random_state=seed)
        return create_forecaster(name, **kwargs)
    return create


def model_registry() -> dict[str, ModelSpec]:
    result = {
        name: ModelSpec(name, _sklearn_factory(name), "15min", False, "auto" if name in {"XGBoost", "LightGBM", "CatBoost"} else "cpu")
        for name in ("Ridge", "RandomForest", "XGBoost", "LightGBM", "CatBoost", "MLP")
    }
    result["SeasonalNaive"] = ModelSpec(
        "SeasonalNaive", _forecast_factory("SeasonalNaive", seasonal_periods=672), "15min", False, "cpu", "forecaster"
    )
    for name in ("ARIMA", "SARIMA", "SARIMAX", "HoltWinters", "Prophet", "LSTM"):
        result[name] = ModelSpec(name, _forecast_factory(name), "1h" if name == "LSTM" else "1d",
                                 name == "SARIMAX", "auto" if name == "LSTM" else "cpu", "forecaster")
    return result
