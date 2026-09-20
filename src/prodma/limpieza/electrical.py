"""Controles fisicos configurables para mediciones electricas."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ElectricalLimits:
    voltage_min: float | None = None
    voltage_max: float | None = None
    frequency_min: float = 55.0
    frequency_max: float = 65.0
    relative_tolerance: float = 0.02


def electrical_flags(frame: pd.DataFrame, limits: ElectricalLimits) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    if {"P", "Q", "S_medida"} <= set(frame):
        minimum_s = np.hypot(frame["P"], frame["Q"])
        result["S_incoherente"] = frame["S_medida"] < minimum_s * (1 - limits.relative_tolerance)
        result["factor_potencia_invalido"] = (frame["P"].abs() / frame["S_medida"].replace(0, np.nan)) > 1 + limits.relative_tolerance
    else:
        result["S_incoherente"] = False
        result["factor_potencia_invalido"] = False
    if "Frecuencia" in frame:
        result["frecuencia_fuera_rango"] = ~frame["Frecuencia"].between(limits.frequency_min, limits.frequency_max)
    else:
        result["frecuencia_fuera_rango"] = False
    if "V_prom" in frame and limits.voltage_min is not None and limits.voltage_max is not None:
        result["tension_fuera_rango"] = ~frame["V_prom"].between(limits.voltage_min, limits.voltage_max)
    else:
        result["tension_fuera_rango"] = False
    return result.fillna(False).astype(bool)


def detect_frozen_sensor(series: pd.Series, min_run: int = 16, tolerance: float = 1e-9) -> pd.Series:
    changed = series.isna() | series.diff().abs().gt(tolerance)
    groups = changed.cumsum()
    lengths = groups.map(groups.value_counts())
    return (lengths >= min_run) & series.notna()
