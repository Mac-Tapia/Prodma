"""Exogenas futuras desde pronostico operativo (Open-Meteo), con cobertura y calidad verificadas.

Solo se acepta calidad `pronostico_operativo`. Los pasos del horizonte se interpolan linealmente
desde la resolucion horaria del proveedor. Fuera de la cobertura del pronostico, o con valores
faltantes, el calculo se detiene: nunca se extrapola ni se rellena en silencio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prodma.external.connectors.open_meteo import CALIDAD_PRONOSTICO

REQUIRED_COLUMNS = ("fecha_hora", "temperatura_c", "calidad")


def future_temperature_from_forecast(forecast: pd.DataFrame, stamps) -> pd.DataFrame:
    if not set(REQUIRED_COLUMNS).issubset(forecast.columns):
        raise ValueError(f"el pronostico requiere columnas {REQUIRED_COLUMNS}")
    if forecast["temperatura_c"].isna().any():
        raise ValueError("el pronostico tiene valores faltantes")
    if not (forecast["calidad"] == CALIDAD_PRONOSTICO).all():
        raise ValueError("solo se acepta calidad pronostico operativo como exogena futura")
    stamps = pd.DatetimeIndex(stamps)
    series = forecast.set_index(pd.DatetimeIndex(forecast["fecha_hora"]))["temperatura_c"].astype(float).sort_index()
    if stamps.min() < series.index.min() or stamps.max() > series.index.max():
        raise ValueError(
            f"pasos {stamps.min()}..{stamps.max()} fuera de la cobertura del pronostico "
            f"{series.index.min()}..{series.index.max()}"
        )
    union = series.index.union(stamps)
    interpolated = series.reindex(union).interpolate(method="time").reindex(stamps)
    if not np.isfinite(interpolated.to_numpy()).all():
        raise ValueError("interpolacion del pronostico no finita")
    return pd.DataFrame({"temperatura_c": interpolated.to_numpy()}, index=stamps)
