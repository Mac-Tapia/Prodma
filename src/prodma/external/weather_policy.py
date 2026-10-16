"""Política meteorológica: observado, pronóstico operativo y climatología nunca se mezclan.

Estados por fila:
- `observado`: hora <= origen y medición/reanálisis válido.
- `faltante`: hora <= origen sin valor válido (no se imputa aquí).
- `pronostico_operativo`: hora > origen con pronóstico emitido <= origen.
- `climatologia_planeamiento`: hora > origen sin pronóstico; mediana mes-hora de lo observado <= origen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ESTADO_OBSERVADO = "observado"
ESTADO_FALTANTE = "faltante"
ESTADO_PRONOSTICO = "pronostico_operativo"
ESTADO_CLIMATOLOGIA = "climatologia_planeamiento"


def classify_weather(
    observed: pd.DataFrame,
    origin: pd.Timestamp,
    *,
    grid: pd.DatetimeIndex | None = None,
    forecast: pd.DataFrame | None = None,
    forecast_issue_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    origin = pd.Timestamp(origin)
    if forecast is not None:
        if forecast_issue_time is None:
            raise ValueError("el pronóstico requiere fecha de emisión")
        if pd.Timestamp(forecast_issue_time) > origin:
            raise ValueError("pronóstico emitido posterior al origen")
    grid = pd.DatetimeIndex(grid if grid is not None else observed["fecha_hora"])
    obs = observed.set_index("fecha_hora")["temperatura_c"].astype(float)
    obs = obs[~obs.index.duplicated(keep="last")]
    quality = observed.set_index("fecha_hora")["calidad"]
    quality = quality[~quality.index.duplicated(keep="last")]

    past = grid <= origin
    usable_past = obs.reindex(grid[past])
    valid_past = usable_past.notna() & quality.reindex(grid[past]).ne("faltante").to_numpy()

    past_index = grid[past]
    valid_values = usable_past.where(valid_past)
    month_hour = valid_values.dropna()
    climatology_table = month_hour.groupby([month_hour.index.month, month_hour.index.hour]).median()
    fallback = float(month_hour.median()) if len(month_hour) else float("nan")

    future_index = grid[~past]
    climatology = pd.Series(
        [climatology_table.get((t.month, t.hour), fallback) for t in future_index], index=future_index, dtype=float,
    )

    forecast_values = pd.Series(dtype=float)
    if forecast is not None and len(future_index):
        forecast_values = forecast.set_index("fecha_hora")["temperatura_c"].astype(float).reindex(future_index)

    estado = np.empty(len(grid), dtype=object)
    observed_col = np.full(len(grid), np.nan)
    forecast_col = np.full(len(grid), np.nan)
    climatology_col = np.full(len(grid), np.nan)

    past_positions = np.flatnonzero(past)
    future_positions = np.flatnonzero(~past)
    for pos, instant, valid in zip(past_positions, past_index, valid_past):
        estado[pos] = ESTADO_OBSERVADO if valid else ESTADO_FALTANTE
        if valid:
            observed_col[pos] = float(usable_past.loc[instant])
    for pos, instant in zip(future_positions, future_index):
        has_forecast = len(forecast_values) and pd.notna(forecast_values.get(instant))
        if has_forecast:
            estado[pos] = ESTADO_PRONOSTICO
            forecast_col[pos] = float(forecast_values.loc[instant])
        else:
            estado[pos] = ESTADO_CLIMATOLOGIA
            climatology_col[pos] = float(climatology.loc[instant])

    return pd.DataFrame({
        "fecha_hora": grid,
        "estado": estado,
        "temperatura_observada_c": observed_col,
        "temperatura_pronostico_c": forecast_col,
        "temperatura_climatologia_c": climatology_col,
    })

HISTORICAL_PRIORITY = ("senamhi", "open_meteo_archive", "nasa_power")


def choose_historical_source(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une fuentes históricas por prioridad explícita; la fuente de menor prioridad solo rellena faltantes.

    Cada fila lleva `fuente_seleccionada`, así el origen de cada valor queda trazable.
    """
    unknown = sorted(set(frames) - set(HISTORICAL_PRIORITY))
    if unknown:
        raise ValueError(f"fuente histórica sin prioridad declarada: {unknown}")
    ordered = [name for name in HISTORICAL_PRIORITY if name in frames]
    if not ordered:
        raise ValueError("no hay fuentes históricas")
    instants = pd.DatetimeIndex(sorted(set().union(*[set(pd.to_datetime(frames[n]["fecha_hora"])) for n in ordered])))
    values = pd.Series(np.nan, index=instants, dtype=float)
    chosen = pd.Series(None, index=instants, dtype=object)
    for name in ordered:
        source = frames[name].assign(fecha_hora=lambda f: pd.to_datetime(f["fecha_hora"]))
        source = source.drop_duplicates("fecha_hora").set_index("fecha_hora")["temperatura_c"].astype(float)
        candidates = source.reindex(instants)
        fill = values.isna() & candidates.notna()
        values[fill] = candidates[fill]
        chosen[fill] = name
    return pd.DataFrame({
        "fecha_hora": instants,
        "temperatura_c": values.to_numpy(),
        "fuente_seleccionada": chosen.to_numpy(),
    })
