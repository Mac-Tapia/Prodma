"""Representaciones meteorologicas reproducibles en cada origen de pronostico."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeatherFeatures:
    observed: pd.Series
    climatology: pd.Series
    anomaly: pd.Series
    available_for_forecast: pd.Series
    mode: str


def _map_climatology(index: pd.DatetimeIndex, climatology: pd.Series) -> pd.Series:
    keys = pd.MultiIndex.from_arrays([index.month, index.hour])
    values = pd.Series(keys.map(climatology), index=index, dtype=float, name="temp_climatologica_c")
    if values.isna().any():
        values = values.fillna(float(climatology.median()))
    return values


def build_weather_features(weather: pd.DataFrame, index: pd.DatetimeIndex, origin: pd.Timestamp,
                           mode: str = "operational_climatology",
                           scenario: pd.Series | None = None) -> WeatherFeatures:
    if mode not in {"operational_climatology", "weather_scenario"}:
        raise ValueError(f"modo meteorologico desconocido: {mode}")
    frame = weather[["fecha_hora", "temperatura_c"]].copy()
    frame["fecha_hora"] = pd.to_datetime(frame["fecha_hora"])
    frame = frame.drop_duplicates("fecha_hora").sort_values("fecha_hora")
    origin = pd.Timestamp(origin)
    available = frame.loc[frame["fecha_hora"] <= origin]
    if available.empty:
        raise ValueError("no hay temperatura disponible antes del origen")
    climatology_table = available.groupby([
        available["fecha_hora"].dt.month,
        available["fecha_hora"].dt.hour,
    ])["temperatura_c"].median()
    target_index = pd.DatetimeIndex(index)
    climatology = _map_climatology(target_index, climatology_table)

    observed_source = available.set_index("fecha_hora")["temperatura_c"].astype(float)
    union = observed_source.index.union(target_index[target_index <= origin])
    interpolated = observed_source.reindex(union).interpolate(method="time", limit_area="inside")
    observed = pd.Series(np.nan, index=target_index, dtype=float, name="temperatura_observada_c")
    historical_index = target_index[target_index <= origin]
    observed.loc[historical_index] = interpolated.reindex(historical_index)

    if mode == "weather_scenario":
        if scenario is None or scenario.reindex(target_index).isna().any():
            raise ValueError("escenario meteorologico incompleto")
        climatology = scenario.reindex(target_index).astype(float).rename("temperatura_escenario_c")
    anomaly = (observed - climatology).rename("anomalia_temperatura_c")
    forecast_available = pd.Series(True, index=target_index, name="temperatura_disponible_pronostico")
    return WeatherFeatures(observed, climatology, anomaly, forecast_available, mode)
