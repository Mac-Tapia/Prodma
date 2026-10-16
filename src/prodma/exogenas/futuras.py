"""Exogenas futuras desde pronostico operativo (Open-Meteo), con cobertura y calidad verificadas.

Solo se acepta calidad `pronostico_operativo`. Los pasos del horizonte se interpolan linealmente
desde la resolucion horaria del proveedor. Fuera de la cobertura del pronostico, o con valores
faltantes, el calculo se detiene: nunca se extrapola ni se rellena en silencio.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from prodma.external.connectors.open_meteo import CALIDAD_PRONOSTICO, forecast_url

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


OPEN_METEO_LICENCE = "CC-BY 4.0"
OPEN_METEO_SOURCE = "Open-Meteo forecast API (https://open-meteo.com)"


def fetch_forecast_payload(lat: float, lon: float, *, opener: Callable = urllib.request.urlopen,
                           timeout: float = 20, past_days: int = 1, forecast_days: int = 3):
    """Descarga el pronostico horario. Devuelve (payload, bytes crudos) para el snapshot.

    Uso no comercial: el limite publico es 600 llamadas/min, 5.000/h y 10.000/dia. Un
    uso comercial requiere suscripcion; el codigo no lo verifica por si mismo.
    """
    url = forecast_url(lat, lon, past_days=past_days, forecast_days=forecast_days)
    with opener(url, timeout=timeout) as response:
        raw = response.read()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("error"):
        raise ValueError(f"Open-Meteo rechazo la consulta: {payload.get('reason', 'sin motivo')}")
    return payload, raw


def save_forecast_snapshot(raw: bytes, root: Path | str, issue_time: pd.Timestamp) -> Path:
    """Guarda la respuesta cruda con su hash y la atribucion de licencia."""
    issue = pd.Timestamp(issue_time)
    directory = Path(root) / issue.strftime("%Y-%m-%d")
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"forecast_{issue.strftime('%H%M')}"
    path = directory / f"{stem}.json"
    path.write_bytes(raw)
    meta = {"fuente": OPEN_METEO_SOURCE, "licencia": OPEN_METEO_LICENCE, "uso": "no_comercial",
            "emitido_en": issue.isoformat(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    path.with_name(f"{stem}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def open_meteo_future_exog(section: dict, stamps, *, opener: Callable = urllib.request.urlopen) -> pd.DataFrame:
    """Descarga, guarda snapshot (si hay carpeta) y convierte el pronostico a los pasos pedidos."""
    from prodma.external.connectors.open_meteo import parse_forecast

    issue = pd.Timestamp.now(tz="America/Lima").tz_localize(None).floor("min")
    payload, raw = fetch_forecast_payload(
        float(section["latitud"]), float(section["longitud"]), opener=opener,
        past_days=int(section.get("dias_pasados", 1)), forecast_days=int(section.get("dias_pronostico", 3)),
    )
    if section.get("carpeta_snapshot"):
        save_forecast_snapshot(raw, section["carpeta_snapshot"], issue)
    forecast = parse_forecast(payload, issue_time=issue)
    return future_temperature_from_forecast(forecast, stamps)