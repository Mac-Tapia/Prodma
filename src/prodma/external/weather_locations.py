"""Asignación de medidores a ubicaciones meteorológicas con distancia explícita."""
from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt

import pandas as pd

ASIGNADO = "ASIGNADO"
BLOCKED_MISSING_SOURCE = "BLOCKED_MISSING_SOURCE"
BLOCKED_OUT_OF_RANGE = "BLOCKED_OUT_OF_RANGE"
EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class WeatherLocation:
    id: str
    lat: float
    lon: float
    source: str


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))


def assign_meter_locations(meters: pd.DataFrame, locations: list[WeatherLocation], max_km: float) -> pd.DataFrame:
    rows = []
    for record in meters.itertuples(index=False):
        if pd.isna(record.lat) or pd.isna(record.lon):
            rows.append({"identificador": record.identificador, "ubicacion": None,
                         "distancia_km": float("nan"), "estado": BLOCKED_MISSING_SOURCE})
            continue
        nearest = min(locations, key=lambda loc: haversine_km(record.lat, record.lon, loc.lat, loc.lon))
        distance = haversine_km(record.lat, record.lon, nearest.lat, nearest.lon)
        rows.append({"identificador": record.identificador, "ubicacion": nearest.id,
                     "distancia_km": distance,
                     "estado": ASIGNADO if distance <= max_km else BLOCKED_OUT_OF_RANGE})
    return pd.DataFrame(rows, columns=["identificador", "ubicacion", "distancia_km", "estado"])
