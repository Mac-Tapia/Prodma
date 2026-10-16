"""Modelos de pronóstico requeridos por el plan de Electro Dunas.

Los modelos estadísticos se destinan principalmente a series mensuales. Los
modelos con rezagos y la LSTM también admiten series de mayor frecuencia,
siempre que el llamador suministre una ventana e horizonte razonables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


class MissingOptionalDependency(RuntimeError):
    """Una familia opcional no puede ejecutarse en el entorno actual."""


def _series(y: pd.Series | np.ndarray) -> pd.Series:
    result = y.astype(float).copy() if isinstance(y, pd.Series) else pd.Series(y, dtype=float)
    result = result.replace([np.inf, -np.inf], np.nan)
    if result.isna().any():
        raise ValueError("La serie objetivo contiene valores faltantes o no finitos")
    if len(result) < 8:
        raise ValueError("Se requieren al menos 8 observaciones")
    return result


def _finite(values: Any, horizon: int) -> np.ndarray:
    out = np.asarray(values, dtype=float).reshape(-1)
    if len(out) != horizon:
        raise ValueError(f"El modelo devolvió {len(out)} valores; se esperaban {horizon}")
    if not np.isfinite(out).all():
        raise ValueError("El modelo generó valores no finitos")
    return out


class BaseForecaster:
    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "BaseForecaster":
        raise NotImplementedError

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        raise NotImplementedError


@dataclass
class SeasonalNaiveForecaster(BaseForecaster):
    seasonal_periods: int = 12

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "SeasonalNaiveForecaster":
        self.y_ = _series(y)
        if len(self.y_) < self.seasonal_periods:
            raise ValueError("Historial menor que un ciclo estacional")
        return self

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        season = self.y_.iloc[-self.seasonal_periods :].to_numpy()
        return _finite(np.resize(season, horizon), horizon)


def _checked_exog(columns: list[str] | None, exog: pd.DataFrame | None, horizon: int | None = None) -> pd.DataFrame | None:
    """Valida exogenas futuras contra las columnas de entrenamiento; None si el modelo no las usa."""
    if columns is None:
        if exog is not None:
            raise ValueError("el modelo fue entrenado sin exogenas y no admite exogenas futuras")
        return None
    if exog is None:
        raise ValueError("el modelo fue entrenado con exogenas y requiere exogenas futuras")
    if list(exog.columns) != columns or (horizon is not None and len(exog) != horizon):
        raise ValueError("Las variables exogenas futuras no coinciden con el entrenamiento")
    if exog.isna().any(axis=None):
        raise ValueError("Las variables exogenas futuras contienen valores faltantes")
    return exog.astype(float)


@dataclass
class ARIMAForecaster(BaseForecaster):
    order: tuple[int, int, int] = (1, 1, 0)

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "ARIMAForecaster":
        self.y_ = _series(y)
        if exog is not None and (len(exog) != len(self.y_) or exog.isna().any(axis=None)):
            raise ValueError("Variables exogenas invalidas o desalineadas")
        self.exog_columns_ = None if exog is None else list(exog.columns)
        self.result_ = ARIMA(self.y_, exog=None if exog is None else exog.astype(float), order=self.order).fit()
        return self

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        future = _checked_exog(self.exog_columns_, exog, horizon)
        return _finite(self.result_.forecast(steps=horizon, exog=future), horizon)


@dataclass
class SARIMAForecaster(BaseForecaster):
    seasonal_periods: int = 12
    order: tuple[int, int, int] = (1, 0, 0)
    seasonal_order_base: tuple[int, int, int] = (1, 0, 0)

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "SARIMAForecaster":
        self.y_ = _series(y)
        if len(self.y_) < 2 * self.seasonal_periods:
            raise ValueError("SARIMA requiere al menos dos ciclos estacionales")
        if exog is not None and (len(exog) != len(self.y_) or exog.isna().any(axis=None)):
            raise ValueError("Variables exogenas invalidas o desalineadas")
        self.exog_columns_ = None if exog is None else list(exog.columns)
        seasonal_order = (*self.seasonal_order_base, self.seasonal_periods)
        self.result_ = SARIMAX(
            self.y_,
            exog=None if exog is None else exog.astype(float),
            order=self.order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=200)
        return self

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        future = _checked_exog(self.exog_columns_, exog, horizon)
        return _finite(self.result_.forecast(steps=horizon, exog=future), horizon)


@dataclass
class HoltWintersForecaster(BaseForecaster):
    seasonal_periods: int = 12

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "HoltWintersForecaster":
        self.y_ = _series(y)
        if len(self.y_) < 2 * self.seasonal_periods:
            raise ValueError("Holt-Winters requiere al menos dos ciclos estacionales")
        self.result_ = ExponentialSmoothing(
            self.y_,
            trend="add",
            seasonal="add",
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        ).fit(optimized=True)
        return self

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        return _finite(self.result_.forecast(horizon), horizon)


@dataclass
class SARIMAXForecaster(BaseForecaster):
    seasonal_periods: int = 12
    order: tuple[int, int, int] = (1, 0, 0)
    seasonal_order_base: tuple[int, int, int] = (1, 0, 0)

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "SARIMAXForecaster":
        self.y_ = _series(y)
        if exog is None:
            raise ValueError("SARIMAX requiere variables exógenas de entrenamiento")
        if len(exog) != len(self.y_) or exog.isna().any(axis=None):
            raise ValueError("Variables exógenas inválidas o desalineadas")
        self.exog_columns_ = list(exog.columns)
        seasonal_order = (*self.seasonal_order_base, self.seasonal_periods)
        self.result_ = SARIMAX(
            self.y_,
            exog=exog.astype(float),
            order=self.order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=200)
        return self

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        if exog is None:
            raise ValueError("SARIMAX requiere variables exógenas futuras")
        if list(exog.columns) != self.exog_columns_ or len(exog) != horizon:
            raise ValueError("Las variables exógenas futuras no coinciden con el entrenamiento")
        return _finite(self.result_.forecast(steps=horizon, exog=exog.astype(float)), horizon)


@dataclass
class ProphetForecaster(BaseForecaster):
    seasonal_periods: int = 12

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "ProphetForecaster":
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise MissingOptionalDependency(
                "Prophet requiere instalar el paquete opcional 'prophet'"
            ) from exc
        values = _series(y)
        if not isinstance(values.index, pd.DatetimeIndex):
            raise ValueError("Prophet requiere un DatetimeIndex")
        frame = pd.DataFrame({"ds": values.index.tz_localize(None), "y": values.to_numpy()})
        self.freq_ = pd.infer_freq(values.index) or "MS"
        # Los umbrales de estacionalidad dependen del CALENDARIO cubierto, no de un
        # conteo de filas fijo: ese conteo asumía datos mensuales (24 filas = 2 años).
        # Con datos diarios/horarios hay muchas más filas por año, así que se mide el
        # tramo real cubierto y se habilita la estacionalidad semanal solo cuando la
        # resolución es diaria o más fina (con datos mensuales, Prophet no puede
        # estimar un patrón semanal inexistente en el muestreo).
        dias_cubiertos = (values.index[-1] - values.index[0]).days
        resolucion_diaria_o_mas_fina = bool(self.freq_) and self.freq_[0] in ("D", "h", "H", "T", "m", "s", "S")
        self.model_ = Prophet(
            yearly_seasonality=dias_cubiertos >= 2 * 365,
            weekly_seasonality=resolucion_diaria_o_mas_fina and len(values) >= 14,
            daily_seasonality=False,
        )
        self.model_.fit(frame)
        self.last_date_ = frame.ds.iloc[-1]
        return self

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        future = pd.DataFrame(
            {"ds": pd.date_range(self.last_date_, periods=horizon + 1, freq=self.freq_)[1:]}
        )
        return _finite(self.model_.predict(future)["yhat"], horizon)


def resolve_torch_device(requested: str = "auto") -> str:
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("device debe ser auto, cpu o gpu")
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        if requested == "gpu":
            raise MissingOptionalDependency(
                "LSTM GPU requiere instalar 'torch' con soporte CUDA"
            ) from exc
        return "cpu"
    cuda = bool(torch.cuda.is_available())
    if requested == "gpu" and not cuda:
        raise RuntimeError("Se solicitó LSTM GPU pero torch.cuda.is_available() es False")
    return "cuda:0" if cuda else "cpu"


@dataclass
class LSTMForecaster(BaseForecaster):
    seasonal_periods: int = 12
    device: str = "auto"
    epochs: int = 50
    hidden_size: int = 32
    seed: int = 42

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "LSTMForecaster":
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise MissingOptionalDependency(
                "LSTM requiere instalar el paquete opcional 'torch'"
            ) from exc
        values = _series(y).to_numpy(dtype=np.float32)
        self.device_ = resolve_torch_device(self.device)
        self.window_ = min(max(4, self.seasonal_periods), len(values) // 3)
        self.mean_, self.std_ = float(values.mean()), float(values.std() or 1.0)
        z = (values - self.mean_) / self.std_
        X = np.stack([z[i : i + self.window_] for i in range(len(z) - self.window_)])
        target = z[self.window_ :]
        torch.manual_seed(self.seed)
        if self.device_.startswith("cuda"):
            torch.cuda.manual_seed_all(self.seed)

        class Network(nn.Module):
            def __init__(self, hidden_size: int):
                super().__init__()
                self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
                self.output = nn.Linear(hidden_size, 1)

            def forward(self, batch):
                encoded, _ = self.lstm(batch)
                return self.output(encoded[:, -1, :]).squeeze(-1)

        self.model_ = Network(self.hidden_size).to(self.device_)
        features = torch.tensor(X[:, :, None], device=self.device_)
        labels = torch.tensor(target, device=self.device_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(self.model_(features), labels)
            loss.backward()
            optimizer.step()
        self.tail_ = list(z[-self.window_ :])
        return self

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        import torch

        values = list(self.tail_)
        self.model_.eval()
        with torch.no_grad():
            for _ in range(horizon):
                batch = torch.tensor(
                    np.asarray(values[-self.window_ :], dtype=np.float32)[None, :, None],
                    device=self.device_,
                )
                values.append(float(self.model_(batch).item()))
        prediction = np.asarray(values[-horizon:]) * self.std_ + self.mean_
        return _finite(prediction, horizon)


class LagRegressorForecaster(BaseForecaster):
    def __init__(self, model_name: str, seasonal_periods: int = 12, device: str = "auto", seed: int = 42):
        self.model_name = model_name
        self.window_ = max(4, seasonal_periods)
        self.device = device
        self.seed = seed

    def _estimator(self):
        if self.model_name == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=self.seed)
        if self.model_name == "XGBoost":
            from xgboost import XGBRegressor

            device = "cuda:0" if resolve_torchless_gpu(self.device) else "cpu"
            return XGBRegressor(
                n_estimators=300,
                tree_method="hist",
                device=device,
                random_state=self.seed,
                n_jobs=2,
            )
        raise ValueError(f"Regresor con rezagos no soportado: {self.model_name}")

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "LagRegressorForecaster":
        values = _series(y).to_numpy()
        if len(values) <= self.window_:
            raise ValueError("Historial insuficiente para las ventanas de rezago")
        X = np.stack([values[i : i + self.window_] for i in range(len(values) - self.window_)])
        target = values[self.window_ :]
        self.model_ = self._estimator().fit(X, target)
        self.tail_ = list(values[-self.window_ :])
        return self

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        values = list(self.tail_)
        for _ in range(horizon):
            row = np.asarray(values[-self.window_ :], dtype=float).reshape(1, -1)
            values.append(float(self.model_.predict(row)[0]))
        return _finite(values[-horizon:], horizon)


def resolve_torchless_gpu(requested: str) -> bool:
    """Resuelve CUDA para XGBoost sin hacer depender los árboles de PyTorch."""
    if requested == "cpu":
        return False
    if requested not in {"auto", "gpu"}:
        raise ValueError("device debe ser auto, cpu o gpu")
    try:
        from xgboost import XGBRegressor

        probe = XGBRegressor(n_estimators=1, max_depth=1, tree_method="hist", device="cuda:0")
        probe.fit(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0.0, 1.0, 2.0, 3.0]))
        return True
    except Exception:
        if requested == "gpu":
            raise RuntimeError("Se solicitó XGBoost GPU pero el microentrenamiento CUDA falló")
        return False


MODEL_NAMES = (
    "SeasonalNaive",
    "ARIMA",
    "SARIMA",
    "HoltWinters",
    "SARIMAX",
    "Prophet",
    "RandomForest",
    "XGBoost",
    "LSTM",
)


def available_model_names() -> tuple[str, ...]:
    return MODEL_NAMES


def create_forecaster(name: str, seasonal_periods: int = 12, **kwargs: Any) -> BaseForecaster:
    factories = {
        "SeasonalNaive": lambda: SeasonalNaiveForecaster(seasonal_periods),
        "ARIMA": lambda: ARIMAForecaster(kwargs.get("order", (1, 1, 0))),
        "SARIMA": lambda: SARIMAForecaster(seasonal_periods),
        "HoltWinters": lambda: HoltWintersForecaster(seasonal_periods),
        "SARIMAX": lambda: SARIMAXForecaster(seasonal_periods),
        "Prophet": lambda: ProphetForecaster(seasonal_periods),
        "RandomForest": lambda: LagRegressorForecaster(
            "RandomForest", seasonal_periods, kwargs.get("device", "cpu"), kwargs.get("seed", 42)
        ),
        "XGBoost": lambda: LagRegressorForecaster(
            "XGBoost", seasonal_periods, kwargs.get("device", "auto"), kwargs.get("seed", 42)
        ),
        "LSTM": lambda: LSTMForecaster(
            seasonal_periods=seasonal_periods,
            device=kwargs.get("device", "auto"),
            epochs=kwargs.get("epochs", 50),
            hidden_size=kwargs.get("hidden_size", 32),
            seed=kwargs.get("seed", 42),
        ),
    }
    if name not in factories:
        raise ValueError(f"Modelo desconocido: {name}. Opciones: {', '.join(MODEL_NAMES)}")
    return factories[name]()
