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


def _ramp_fraction(stamps: pd.DatetimeIndex, ramp_months: float) -> np.ndarray:
    """Fraccion de la potencia anual ya conectada: rampa lineal desde el 1 de enero del anio."""
    if ramp_months <= 0:
        return np.ones(len(stamps))
    start = pd.to_datetime(stamps.year.astype(str) + "-01-01")
    elapsed = (stamps - start) / pd.Timedelta(days=1)
    ramp_days = ramp_months * 365.25 / 12
    return np.clip(np.asarray(elapsed, dtype=float) / ramp_days, 0.0, 1.0)


def customer_increments(customer_table: pd.DataFrame, meter: str, stamps, ramp_months: float = 0.0,
                        coincidence: float = 1.0) -> pd.DataFrame:
    """Incremento por escenario de clientes para un medidor.

    Supuestos declarados y configurables (defaults = escalon anual sin coincidencia):
    - cada columna anual es la potencia nueva de ese anio; los anios previos ya estan conectados;
    - dentro del anio en curso la potencia entra en rampa lineal de `ramp_months` desde el 1 de enero;
    - `coincidence` multiplica el incremento (factor de simultaneidad, en (0, 1]).
    Un medidor sin escenario es un error explicito, no un cero implicito.
    """
    if not 0 < coincidence <= 1:
        raise ValueError("el factor de coincidencia debe estar en (0, 1]")
    if ramp_months < 0:
        raise ValueError("la rampa no puede ser negativa")
    rows = customer_table.loc[customer_table["medidor"].astype(str) == str(meter)]
    if rows.empty:
        raise ValueError(f"el medidor {meter} no tiene escenario de clientes")
    if rows[list(CUSTOMER_SCENARIOS.values())].isna().any(axis=None):
        raise ValueError(f"escenario de {meter} con valores faltantes")
    stamps = pd.DatetimeIndex(stamps)
    fraction = _ramp_fraction(stamps, float(ramp_months))
    records = []
    for name, column in CUSTOMER_SCENARIOS.items():
        per_year = rows.groupby("anio")[column].sum().sort_index()
        for position, stamp in enumerate(stamps):
            previous = per_year.loc[per_year.index < stamp.year].sum()
            current = per_year.get(stamp.year, 0.0)
            total = (previous + current * fraction[position]) * coincidence
            records.append({"escenario": name, "timestamp": stamp, "incremento": float(total)})
    return pd.DataFrame(records)