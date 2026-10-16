"""Clientes: histórico realizado separado del escenario de planeamiento confirmado."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BLOCKED_MISSING_SOURCE = "BLOCKED_MISSING_SOURCE"
ESTADO_CAPACIDAD_VALIDA = "VALIDA"
ESTADO_MAPEADO = "MAPEADO"


def historical_connections_status(path: Path | str) -> dict:
    if Path(path).is_file():
        return {"estado": "CANDIDATO_VERIFICABLE", "motivo": "archivo presente; validar con realized_connections"}
    return {
        "estado": BLOCKED_MISSING_SOURCE,
        "motivo": ("sin altas realizadas con fecha en el repositorio: el archivo de nuevos clientes "
                   "contiene solo incrementos anuales proyectados, no altas históricas"),
    }


def realized_connections(frame: pd.DataFrame) -> pd.DataFrame:
    required = ("cliente", "medidor", "fecha_alta")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"altas realizadas sin columnas {missing}")
    dates = pd.to_datetime(frame["fecha_alta"], errors="coerce")
    bad = dates.isna()
    if bad.any():
        rows = [int(i) + 2 for i in frame.index[bad]]
        raise ValueError(f"fecha_alta faltante o inválida en filas {rows}")
    result = frame[list(required)].copy()
    result["disponible_desde"] = dates
    return result.reset_index(drop=True)


def customer_scenarios(table: pd.DataFrame) -> pd.DataFrame:
    """Escenario por medidor y año. Solo cuenta lo válido y mapeado; el resto queda sin escenario.

    No modifica la tabla de entrada ni la historia: el escenario se suma después de la predicción base.
    """
    included = table[(table["estado_capacidad"] == ESTADO_CAPACIDAD_VALIDA)
                     & (table["estado_mapeo"] == ESTADO_MAPEADO)].copy()
    included["superior"] = included["demanda_kw"].where(included["incremento_kw"] > 0)
    grouped = included.groupby(["medidor", "anio"], as_index=False).agg(
        central_kw=("incremento_kw", "sum"),
        limite_superior_kw=("superior", "sum"),
        clientes=("cliente", "nunique"),
    )
    grouped["limite_inferior_kw"] = 0.0
    grouped["calidad"] = "scenario_no_observed"
    rejected = table[~table.index.isin(included.index)].groupby(["medidor", "anio"], as_index=False).agg(
        clientes_excluidos=("cliente", "nunique"))
    result = table[["medidor", "anio"]].drop_duplicates().merge(grouped, on=["medidor", "anio"], how="left")
    result = result.merge(rejected, on=["medidor", "anio"], how="left")
    without_scenario = result["central_kw"].isna()
    result.loc[without_scenario, ["limite_superior_kw", "limite_inferior_kw"]] = float("nan")
    result["calidad"] = "scenario_no_observed"
    return result[["medidor", "anio", "limite_inferior_kw", "central_kw", "limite_superior_kw",
                   "clientes", "clientes_excluidos", "calidad"]].sort_values(["medidor", "anio"]).reset_index(drop=True)
