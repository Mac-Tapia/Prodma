"""Uniones as-of que respetan fecha de disponibilidad de la informacion."""
from __future__ import annotations

import pandas as pd


def asof_values(index: pd.DatetimeIndex, vintages: pd.DataFrame, origin: pd.Timestamp) -> pd.Series:
    required = {"periodo", "valor", "fecha_disponible"}
    missing = required - set(vintages)
    if missing:
        raise ValueError(f"faltan columnas de vintage: {sorted(missing)}")
    origin = pd.Timestamp(origin)
    available = vintages.copy()
    available["periodo"] = pd.to_datetime(available["periodo"])
    available["fecha_disponible"] = pd.to_datetime(available["fecha_disponible"])
    available = available.loc[available["fecha_disponible"] <= origin].sort_values(["periodo", "fecha_disponible"])
    available = available.drop_duplicates("periodo", keep="last")
    targets = pd.DataFrame({"timestamp": pd.DatetimeIndex(index)}).sort_values("timestamp")
    if available.empty:
        return pd.Series(float("nan"), index=pd.DatetimeIndex(index), dtype=float)
    joined = pd.merge_asof(
        targets,
        available[["periodo", "valor"]].sort_values("periodo"),
        left_on="timestamp",
        right_on="periodo",
        direction="backward",
    )
    result = pd.Series(joined["valor"].to_numpy(dtype=float), index=pd.DatetimeIndex(joined["timestamp"]), dtype=float)
    return result.reindex(pd.DatetimeIndex(index))


def pbi_feature_frame(index: pd.DatetimeIndex, vintages: pd.DataFrame, origin: pd.Timestamp,
                      scenario: pd.Series | None = None) -> pd.DataFrame:
    result = pd.DataFrame({"pbi_last_known": asof_values(index, vintages, origin)}, index=index)
    if scenario is not None:
        aligned = scenario.reindex(index)
        if aligned.isna().any():
            raise ValueError("escenario PBI incompleto para el horizonte")
        result["pbi_scenario"] = aligned.astype(float)
    return result
