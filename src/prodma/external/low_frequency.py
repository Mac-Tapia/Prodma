"""Features de baja frecuencia con visibilidad por instante (sin relleno hacia atrás)."""
from __future__ import annotations

import pandas as pd


def step_feature(index: pd.DatetimeIndex, points: pd.DataFrame) -> pd.Series:
    """Para cada instante, el último valor cuya `fecha_disponible` es <= ese instante."""
    required = {"periodo", "valor", "fecha_disponible"}
    missing = required - set(points)
    if missing:
        raise ValueError(f"faltan columnas de puntos de baja frecuencia: {sorted(missing)}")
    targets = pd.DataFrame({"instante": pd.DatetimeIndex(index)}).sort_values("instante")
    published = points.copy()
    published["fecha_disponible"] = pd.to_datetime(published["fecha_disponible"])
    published = published.sort_values("fecha_disponible")
    joined = pd.merge_asof(
        targets, published[["fecha_disponible", "valor"]],
        left_on="instante", right_on="fecha_disponible", direction="backward",
    )
    result = pd.Series(joined["valor"].to_numpy(dtype=float), index=pd.DatetimeIndex(joined["instante"]))
    return result.reindex(pd.DatetimeIndex(index)).rename("valor_baja_frecuencia")
