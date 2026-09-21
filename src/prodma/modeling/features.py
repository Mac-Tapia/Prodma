"""Variables de series temporales calculadas solo con pasado."""
from __future__ import annotations

import numpy as np
import pandas as pd


def join_exogenous_features(endogenous: pd.DataFrame, exogenous: pd.DataFrame) -> pd.DataFrame:
    """Une solo por indice identico; nunca expande la ventana de la demanda."""
    if not endogenous.index.equals(exogenous.index):
        raise ValueError("indice exogeno desalineado")
    overlap = set(endogenous).intersection(exogenous)
    if overlap:
        raise ValueError(f"columnas exogenas duplicadas: {sorted(overlap)}")
    return pd.concat([endogenous, exogenous], axis=1)


def make_features(clean: pd.DataFrame) -> pd.DataFrame:
    z = clean["contexto"].astype(float)
    idx = pd.DatetimeIndex(clean.index)
    result = pd.DataFrame(index=idx)
    hour = idx.hour + idx.minute / 60
    for harmonic in (1, 2, 3):
        result[f"hora_sin{harmonic}"] = np.sin(2 * np.pi * harmonic * hour / 24)
        result[f"hora_cos{harmonic}"] = np.cos(2 * np.pi * harmonic * hour / 24)
    result["anio_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    result["anio_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    for day in range(1, 7):
        result[f"dia_{day}"] = (idx.dayofweek == day).astype(int)
    result["fin_semana"] = (idx.dayofweek >= 5).astype(int)
    for lag in (1, 4, 96, 672):
        result[f"lag_{lag}"] = z.shift(lag)
    for window in (4, 24, 96, 672):
        rolling = z.shift(1).rolling(window, min_periods=window)
        result[f"media_{window}"] = rolling.mean()
        result[f"std_{window}"] = rolling.std()
        result[f"mediana_{window}"] = rolling.median()
        result[f"min_{window}"] = rolling.min()
        result[f"max_{window}"] = rolling.max()
        result[f"p95_{window}"] = rolling.quantile(0.95)
    weekly = pd.concat([z.shift(672 * step) for step in (1, 2, 3, 4)], axis=1)
    result["media_mismo_dia_hora_4sem"] = weekly.mean(axis=1)
    result["std_mismo_dia_hora_4sem"] = weekly.std(axis=1)
    imputed = clean.get("es_imputado", pd.Series(False, index=idx)).astype(float)
    result["prop_imputado_96"] = imputed.shift(1).rolling(96, min_periods=1).mean()
    return result
