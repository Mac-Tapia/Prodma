"""Conciliación de matrices exógenas: toda columna tiene fuente declarada y estado terminal.

Una columna sin fuente declarada es un error, nunca se descarta en silencio. Una columna
completamente nula se reporta como BLOCKED_INVALID. Las fuentes deshabilitadas aparecen con
su estado terminal de configuración aunque no aporten columnas.
"""
from __future__ import annotations

import pandas as pd

UNEXPLAINED_COLUMN = "columna exógena sin fuente declarada"

# Columna de la matriz -> fuente del catálogo que la produce.
COLUMN_SOURCE = {
    "feriado": "calendario_pe",
    "fin_semana": "calendario_pe",
    "no_laborable": "calendario_pe",
    "temp_climatologica_c": "temperatura_ica",
    "pbi_last_known": "pbi_bcrp",
    "poblacion_ica": "poblacion_ica",
}


def reconcile_exogenous_matrix(frame: pd.DataFrame, *, enabled_sources: set[str],
                               disabled_states: dict[str, str]) -> pd.DataFrame:
    unexplained = [column for column in frame.columns if column not in COLUMN_SOURCE]
    if unexplained:
        raise ValueError(f"{UNEXPLAINED_COLUMN}: {unexplained}")
    rows = []
    for column in frame.columns:
        source = COLUMN_SOURCE[column]
        coverage = float(frame[column].notna().mean() * 100) if len(frame) else 0.0
        if source not in enabled_sources:
            state = disabled_states.get(source, "SIN_ESTADO")
        elif coverage == 0.0:
            state = "BLOCKED_INVALID"
        else:
            state = "USED"
        rows.append({"columna": column, "fuente": source, "cobertura_pct": coverage, "estado": state})
    for source, state in sorted(disabled_states.items()):
        if source not in {row["fuente"] for row in rows}:
            rows.append({"columna": None, "fuente": source, "cobertura_pct": 0.0, "estado": state})
    return pd.DataFrame(rows, columns=["columna", "fuente", "cobertura_pct", "estado"])
