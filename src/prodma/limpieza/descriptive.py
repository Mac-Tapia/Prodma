"""Estadistica descriptiva auditable de variables limpias."""
from __future__ import annotations

import numpy as np
import pandas as pd


def describe_variable(frame: pd.DataFrame) -> dict[str, float | int]:
    observed = frame["observado"].dropna()
    return {
        "observados": int(frame["observado"].notna().sum()),
        "imputados": int((frame["observado"].isna() & frame["contexto"].notna()).sum()),
        "faltantes_contexto": int(frame["contexto"].isna().sum()),
        "media": float(observed.mean()) if len(observed) else np.nan,
        "mediana": float(observed.median()) if len(observed) else np.nan,
        "desviacion": float(observed.std()) if len(observed) > 1 else np.nan,
        "mad": float((observed - observed.median()).abs().median()) if len(observed) else np.nan,
        "minimo": float(observed.min()) if len(observed) else np.nan,
        "maximo": float(observed.max()) if len(observed) else np.nan,
        "ceros": int((observed == 0).sum()),
        "negativos": int((observed < 0).sum()),
        "atipicos": int(frame["es_atipico"].sum()),
        "recortados": int(frame["es_recortado"].sum()),
    }
