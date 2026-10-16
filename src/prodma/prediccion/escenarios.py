"""Escenarios de clientes sobre el pronostico base: base + incrementos declarados por escenario.

Contrato estricto: un incremento faltante, un paso fuera del horizonte o un total negativo
detienen el calculo. Nunca se rellenan ni se descartan en silencio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCENARIO_COLUMNS = ("escenario", "timestamp", "incremento")


def apply_scenarios(base: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    if not set(("timestamp", "prediccion")).issubset(base.columns):
        raise ValueError("el pronostico base requiere columnas timestamp y prediccion")
    if not np.isfinite(base["prediccion"].to_numpy(dtype=float)).all():
        raise ValueError("el pronostico base contiene valores no finitos")
    if not set(SCENARIO_COLUMNS).issubset(scenarios.columns):
        raise ValueError(f"los escenarios requieren columnas {SCENARIO_COLUMNS}")
    if scenarios["incremento"].isna().any():
        raise ValueError("incremento faltante en escenarios; no se rellena")
    stamps = pd.DatetimeIndex(base["timestamp"])
    scenario_stamps = pd.DatetimeIndex(pd.to_datetime(scenarios["timestamp"]))
    if not scenario_stamps.isin(stamps).all():
        raise ValueError("escenarios con pasos fuera del horizonte del pronostico base")
    increments = scenarios.assign(timestamp=scenario_stamps).groupby(["escenario", "timestamp"], as_index=False)[
        "incremento"
    ].sum()
    frames = []
    for name in ["base", *sorted(increments["escenario"].unique())]:
        frame = pd.DataFrame({"escenario": name, "timestamp": stamps, "prediccion_base": base["prediccion"].to_numpy(dtype=float)})
        if name != "base":
            selected = increments.loc[increments["escenario"] == name].set_index("timestamp")["incremento"]
            frame["incremento"] = frame["timestamp"].map(selected).fillna(0.0).to_numpy(dtype=float)
        else:
            frame["incremento"] = 0.0
        frame["prediccion_total"] = frame["prediccion_base"] + frame["incremento"]
        if (frame["prediccion_total"] < 0).any():
            raise ValueError(f"escenario {name} produce demanda negativa")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


CUSTOMER_SCENARIOS = {"inferior": "limite_inferior_kw", "central": "central_kw", "superior": "limite_superior_kw"}


def customer_increments(customer_table: pd.DataFrame, meter: str, stamps) -> pd.DataFrame:
    """Incremento acumulado por escenario de clientes para un medidor.

    Supuesto declarado: cada columna anual es la potencia nueva que entra el 1 de enero de ese
    año y se acumula. Un medidor sin escenario es un error explicito, no un cero implicito.
    """
    rows = customer_table.loc[customer_table["medidor"].astype(str) == str(meter)]
    if rows.empty:
        raise ValueError(f"el medidor {meter} no tiene escenario de clientes")
    if rows[list(CUSTOMER_SCENARIOS.values())].isna().any(axis=None):
        raise ValueError(f"escenario de {meter} con valores faltantes")
    stamps = pd.DatetimeIndex(stamps)
    records = []
    for name, column in CUSTOMER_SCENARIOS.items():
        per_year = rows.groupby("anio")[column].sum().sort_index()
        cumulative = per_year.cumsum()
        for stamp in stamps:
            applicable = cumulative.loc[cumulative.index <= stamp.year]
            records.append({"escenario": name, "timestamp": stamp,
                            "incremento": float(applicable.iloc[-1]) if len(applicable) else 0.0})
    return pd.DataFrame(records)
