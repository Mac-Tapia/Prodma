"""Conector NASA POWER (hourly, reanálisis MERRA-2) normalizado a hora de Lima.

Reglas:
- `header.time_standard` es obligatorio: LST o UTC. Lima opera en UTC-5 sin horario de verano,
  así que LST y la hora local coinciden; UTC se convierte restando 5 h.
- `fill_value` (normalmente -999) se convierte en NaN con calidad `faltante`; nunca es temperatura.
- Solo se acepta la unidad Celsius declarada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LIMA_UTC_OFFSET = pd.Timedelta(hours=-5)
CALIDAD_REANALISIS = "reanalisis_no_es_medicion_directa"
CALIDAD_FALTANTE = "faltante"
UNIDADES_CELSIUS = {"C", "°C", "degC"}


def parse_power_hourly(payload: dict) -> pd.DataFrame:
    header = payload["header"]
    units = payload["parameters"]["T2M"]["units"]
    if units not in UNIDADES_CELSIUS:
        raise ValueError(f"unidad no soportada para T2M: {units!r}; se requiere Celsius")
    standard = header["time_standard"]
    values = payload["properties"]["parameter"]["T2M"]
    fill = float(header.get("fill_value", -999.0))

    timestamps = pd.to_datetime(pd.Series(list(values.keys())), format="%Y%m%d%H")
    if standard == "UTC":
        timestamps = timestamps + LIMA_UTC_OFFSET
    elif standard != "LST":
        raise ValueError(f"time_standard no soportado: {standard!r}")

    raw = pd.to_numeric(pd.Series(list(values.values())), errors="coerce").astype(float)
    missing = raw.isna() | np.isclose(raw, fill)
    lon, lat = payload["geometry"]["coordinates"][0], payload["geometry"]["coordinates"][1]
    frame = pd.DataFrame({
        "fecha_hora": timestamps.to_numpy(),
        "temperatura_c": raw.where(~missing).to_numpy(),
        "calidad": np.where(missing, CALIDAD_FALTANTE, CALIDAD_REANALISIS),
        "fuente": "nasa_power",
        "modelo": ",".join(header.get("sources", [])),
        "lat": lat,
        "lon": lon,
    })
    return frame.sort_values("fecha_hora").reset_index(drop=True)
