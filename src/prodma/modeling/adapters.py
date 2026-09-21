"""Adaptadores para una interfaz comun de pronostico."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prodma.modeling.registry import ModelSpec


@dataclass(frozen=True)
class TrainingSlice:
    x: pd.DataFrame
    y: pd.Series
    train_positions: np.ndarray
    validation_positions: np.ndarray
    exogenous: pd.DataFrame | None = None


@dataclass(frozen=True)
class ForecastRequest:
    x: pd.DataFrame
    index: pd.DatetimeIndex
    horizon: int
    exogenous: pd.DataFrame | None = None


class ForecastAdapter:
    def __init__(self, spec: ModelSpec, seed: int):
        self.spec = spec
        self.seed = seed
        self.model = spec.factory(seed)
        self.effective_backend = "cpu"

    def fit(self, training: TrainingSlice) -> "ForecastAdapter":
        positions = np.asarray(training.train_positions)
        if self.spec.family == "sklearn":
            self.model.fit(training.x.iloc[positions], training.y.iloc[positions])
        else:
            y = training.y.iloc[positions]
            exog = training.exogenous.iloc[positions] if training.exogenous is not None else None
            if self.spec.resolution != "15min":
                rule = "1h" if self.spec.resolution == "1h" else "1D"
                y = y.resample(rule).mean().dropna()
                exog = exog.resample(rule).mean().reindex(y.index) if exog is not None else None
            self.model.fit(y, exog=exog if self.spec.supports_exogenous else None)
        device = getattr(self.model, "device_", None)
        if device:
            self.effective_backend = str(device)
        return self

    def predict(self, request: ForecastRequest) -> pd.Series:
        if request.horizon != len(request.index):
            raise ValueError("horizonte e indice no coinciden")
        if self.spec.family == "sklearn":
            values = self.model.predict(request.x)
        elif self.spec.resolution == "15min":
            values = self.model.predict(
                request.horizon,
                exog=request.exogenous if self.spec.supports_exogenous else None,
            )
        else:
            rule = "1h" if self.spec.resolution == "1h" else "1D"
            periods = pd.Series(0, index=request.index).resample(rule).size()
            coarse_exog = None
            if request.exogenous is not None:
                coarse_exog = request.exogenous.resample(rule).mean().reindex(periods.index)
            coarse = np.asarray(self.model.predict(
                len(periods), exog=coarse_exog if self.spec.supports_exogenous else None
            ), dtype=float)
            lookup = pd.Series(coarse, index=periods.index)
            keys = request.index.floor("h") if self.spec.resolution == "1h" else request.index.normalize()
            values = lookup.reindex(keys).to_numpy()
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) != request.horizon or not np.isfinite(values).all():
            raise ValueError("prediccion no finita o de longitud incorrecta")
        return pd.Series(values, index=request.index, name="prediccion")
