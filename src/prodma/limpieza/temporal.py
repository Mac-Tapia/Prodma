"""Imputacion causal que separa contexto y objetivo."""
from __future__ import annotations

import numpy as np
import pandas as pd

from prodma.limpieza.contracts import CLEAN_COLUMNS


def build_clean_variable(raw: pd.Series, forward_limit: int = 4, weekly_lag: int = 672) -> pd.DataFrame:
    observed = raw.replace([np.inf, -np.inf], np.nan).astype(float)
    forward = observed.ffill(limit=forward_limit) if forward_limit > 0 else observed.copy()
    weekly = observed.shift(weekly_lag)
    context = forward.fillna(weekly)
    method = pd.Series("", index=raw.index, dtype="string")
    method.loc[observed.isna() & forward.notna()] = "ffill"
    method.loc[observed.isna() & forward.isna() & weekly.notna()] = "semana_observada"
    is_observed = observed.notna()
    frame = pd.DataFrame(index=raw.index)
    frame["observado"] = observed
    frame["contexto"] = context
    frame["limpio"] = observed
    frame["objetivo"] = observed.where(is_observed)
    frame["es_observado"] = is_observed
    frame["es_imputado"] = ~is_observed & context.notna()
    frame["metodo_imputacion"] = method
    frame["es_atipico"] = False
    frame["es_recortado"] = False
    return frame.loc[:, CLEAN_COLUMNS]
