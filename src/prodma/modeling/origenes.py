"""Pronosticos recursivos y por origenes, compartidos por las etapas 4 y 5."""
from __future__ import annotations

import numpy as np
import pandas as pd

from prodma.modeling.features import make_features

def recursive_forecast(model, columns: list[str], clean: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Pronostica `horizon` pasos; cada valor predicho alimenta los rezagos del paso siguiente."""
    frequency = pd.infer_freq(clean.index) or "15min"
    history = clean[["contexto", "es_imputado"]].astype({"contexto": float}).copy()
    stamps: list[pd.Timestamp] = []
    values: list[float] = []
    for _ in range(horizon):
        next_stamp = history.index[-1] + pd.tseries.frequencies.to_offset(frequency)
        history.loc[next_stamp] = [np.nan, 0.0]
        features = make_features(history)
        row = features.iloc[[-1]][columns]
        value = float(np.asarray(model.predict(row), dtype=float).reshape(-1)[0])
        if not np.isfinite(value):
            raise ValueError("prediccion recursiva no finita")
        history.loc[next_stamp, "contexto"] = value
        stamps.append(next_stamp)
        values.append(value)
    return pd.DataFrame({"timestamp": [str(s) for s in stamps], "paso": range(1, horizon + 1), "prediccion": values})


def origin_forecasts(model, columns: list[str], clean: pd.DataFrame, final_stamps, horizon: int) -> pd.DataFrame:
    """Pronostica `horizon` pasos desde cada origen dentro del test final (sin reentrenar)."""
    stamps = pd.DatetimeIndex(final_stamps)
    step = clean.index[1] - clean.index[0]
    rows: list[pd.DataFrame] = []
    for origin, start in enumerate(range(0, len(stamps) - horizon + 1, horizon)):
        history = clean.loc[: stamps[start] - step]
        forecast = recursive_forecast(model, columns, history, horizon)
        observed = clean["contexto"].reindex(stamps[start:start + horizon]).to_numpy(dtype=float)
        rows.append(pd.DataFrame({"origen": origin, "y": observed, "prediccion": forecast["prediccion"].to_numpy()}))
    return pd.concat(rows, ignore_index=True)

