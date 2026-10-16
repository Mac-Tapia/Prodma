"""Ranking determinista con gates explicitos; no decide por metricas de diagnostico."""
from __future__ import annotations

import pandas as pd

REQUIRED = ("modelo", "estado", "mae", "estabilidad", "cobertura_error", "complejidad", "segundos")


def rank_candidates(candidates: pd.DataFrame, coverage_tolerance: float,
                    require_calibration: bool = True) -> pd.DataFrame:
    """Ordena por precision, luego estabilidad, simplicidad y tiempo; conserva todos los candidatos.

    Si `require_calibration` es False, una cobertura no evaluada no excluye al candidato,
    pero queda marcada como `NO_EVALUADA` en la tabla de salida.
    """
    if candidates.empty or not set(REQUIRED).issubset(candidates.columns):
        raise ValueError(f"tabla de candidatos vacia o sin columnas {REQUIRED}")
    table = candidates.loc[:, list(REQUIRED)].copy()
    causes = []
    calibration = []
    for row in table.itertuples(index=False):
        calibration_missing = pd.isna(row.cobertura_error)
        calibration.append("NO_EVALUADA" if calibration_missing else "EVALUADA")
        if row.estado != "TRAINED":
            causes.append(f"estado={row.estado}")
        elif pd.isna(row.mae) or pd.isna(row.estabilidad) or (require_calibration and calibration_missing):
            causes.append("metrica_faltante")
        elif not calibration_missing and abs(row.cobertura_error) > coverage_tolerance:
            causes.append("calibracion_fuera_de_tolerancia")
        else:
            causes.append("")
    table["causa_exclusion"] = causes
    table["calibracion"] = calibration
    table["admitido"] = table["causa_exclusion"] == ""
    admitted = table.loc[table["admitido"]].sort_values(
        ["mae", "estabilidad", "complejidad", "segundos", "modelo"], kind="mergesort"
    )
    table["posicion"] = pd.NA
    table.loc[admitted.index, "posicion"] = range(1, len(admitted) + 1)
    table["posicion"] = table["posicion"].astype("Int64")
    return table.loc[admitted.index.tolist() + table.index.difference(admitted.index).tolist()].reset_index(drop=True)
