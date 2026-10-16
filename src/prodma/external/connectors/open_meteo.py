"""Conector Open-Meteo (archivo ERA5 y pronóstico) con zona horaria y unidad verificadas.

- Archivo: reanálisis, no medición directa. Se conserva la etiqueta de calidad.
- Pronóstico: la hora de emisión no viene en la respuesta; la aporta quien descarga
  (fecha_descarga del snapshot) y se guarda en `emitido_en` para validar el origen.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

ARCHIVE_BASE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "America/Lima"
CALIDAD_REANALISIS = "reanalisis_no_es_medicion_directa"
CALIDAD_PRONOSTICO = "pronostico_operativo"
CALIDAD_FALTANTE = "faltante"
_CELSIUS = re.compile(r"^\s*(?:°|º|�)?\s*C\s*$")


def archive_url(lat: float, lon: float, start: str, end: str) -> str:
    return (f"{ARCHIVE_BASE}?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
            f"&hourly=temperature_2m&timezone=America%2FLima")


def forecast_url(lat: float, lon: float, *, past_days: int, forecast_days: int) -> str:
    return (f"{FORECAST_BASE}?latitude={lat}&longitude={lon}&hourly=temperature_2m"
            f"&past_days={past_days}&forecast_days={forecast_days}&timezone=America%2FLima")


def _parse(payload: dict, quality: str, source: str, issue_time: pd.Timestamp | None) -> pd.DataFrame:
    if payload.get("timezone") != TIMEZONE:
        raise ValueError(f"zona horaria no soportada: {payload.get('timezone')!r}; se requiere {TIMEZONE}")
    unit = payload.get("hourly_units", {}).get("temperature_2m", "")
    if not _CELSIUS.match(unit):
        raise ValueError(f"unidad no soportada para temperature_2m: {unit!r}; se requiere Celsius")
    hourly = payload["hourly"]
    times = pd.to_datetime(pd.Series(hourly["time"]), format="%Y-%m-%dT%H:%M")
    values = pd.to_numeric(pd.Series(hourly["temperature_2m"]), errors="coerce").astype(float)
    missing = values.isna()
    frame = pd.DataFrame({
        "fecha_hora": times.to_numpy(),
        "temperatura_c": values.where(~missing).to_numpy(),
        "calidad": np.where(missing, CALIDAD_FALTANTE, quality),
        "fuente": source,
    })
    if issue_time is not None:
        frame["emitido_en"] = pd.Timestamp(issue_time)
    return frame.sort_values("fecha_hora").reset_index(drop=True)


def parse_archive(payload: dict) -> pd.DataFrame:
    return _parse(payload, CALIDAD_REANALISIS, "open_meteo_archive", None)


def parse_forecast(payload: dict, *, issue_time: pd.Timestamp) -> pd.DataFrame:
    return _parse(payload, CALIDAD_PRONOSTICO, "open_meteo_forecast", issue_time)
