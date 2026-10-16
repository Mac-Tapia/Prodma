"""Ranking determinista con gates explicitos; no decide por metricas de diagnostico."""
from __future__ import annotations

import pandas as pd

REQUIRED = ("modelo", "estado", "mae", "estabilidad", "cobertura_error", "complejidad", "segundos")


def rank_candidates(candidates: pd.DataFrame, coverage_tolerance: float) -> pd.DataFrame:
    """Ordena por precision, luego estabilidad, simplicidad y tiempo; conserva todos los candidatos."""
    if candidates.empty or not set(REQUIRED).issubset(candidates.columns):
        raise ValueError(f"tabla de candidatos vacia o sin columnas {REQUIRED}")
    table = candidates.loc[:, list(REQUIRED)].copy()
    causes = []
    for row in table.itertuples(index=False):
        if row.estado != "TRAINED":
            causes.append(f"estado={row.estado}")
        elif pd.isna(row.mae) or pd.isna(row.estabilidad) or pd.isna(row.cobertura_error):
            causes.append("metrica_faltante")
        elif abs(row.cobertura_error) > coverage_tolerance:
            causes.append("calibracion_fuera_de_tolerancia")
        else:
            causes.append("")
    table["causa_exclusion"] = causes
    table["admitido"] = table["causa_exclusion"] == ""
    admitted = table.loc[table["admitido"]].sort_values(
        ["mae", "estabilidad", "complejidad", "segundos", "modelo"], kind="mergesort"
    )
    table["posicion"] = pd.NA
    table.loc[admitted.index, "posicion"] = range(1, len(admitted) + 1)
    table["posicion"] = table["posicion"].astype("Int64")
    return table.loc[admitted.index.tolist() + table.index.difference(admitted.index).tolist()].reset_index(drop=True)
