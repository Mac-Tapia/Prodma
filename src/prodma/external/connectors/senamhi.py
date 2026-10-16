"""SENAMHI: exportación verificada de estaciones de Ica, o estado bloqueado explícito.

La web de SENAMHI (datos hidrometeorológicos de Ica) es accesible, pero la descarga histórica
es un formulario interactivo sin API documentada. Hasta que se incorpore una exportación
verificada y su procedencia, la fuente queda en BLOCKED_MISSING_SOURCE.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BLOCKED_MISSING_SOURCE = "BLOCKED_MISSING_SOURCE"
REQUIRED = ("codigo_estacion", "fecha_hora", "temperatura_c", "fuente_verificacion")


def source_status(export_path: Path | str) -> dict:
    if Path(export_path).is_file():
        return {"estado": "CANDIDATO_VERIFICABLE", "motivo": "exportación presente; validar con parse_verified_export"}
    return {
        "estado": BLOCKED_MISSING_SOURCE,
        "motivo": ("sin exportación SENAMHI verificada para Ica: la web de datos hidrometeorológicos "
                   "requiere descarga interactiva sin API documentada"),
    }


def parse_verified_export(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED if column not in frame.columns]
    if missing:
        raise ValueError(f"exportación SENAMHI sin columnas {missing}")
    result = pd.DataFrame({
        "fecha_hora": pd.to_datetime(frame["fecha_hora"]),
        "temperatura_c": pd.to_numeric(frame["temperatura_c"], errors="coerce"),
        "calidad": "medicion_directa",
        "fuente": "senamhi",
        "codigo_estacion": frame["codigo_estacion"].astype(str),
    })
    return result.sort_values("fecha_hora").reset_index(drop=True)
