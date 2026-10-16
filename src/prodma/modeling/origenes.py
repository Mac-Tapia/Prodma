"""Pronosticos recursivos y por origenes, compartidos por las etapas 4 y 5.

Sin `exog`, el modelo solo usa variables endogenas. Con `exog` (serie temporal completa de
exogenas), los valores de cada paso se toman de esa serie: si son los realizados, el resultado
es una cobertura condicionada a exogenas conocidas, nunca un pronostico operativo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prodma.modeling.features import make_features


def _features_with_exog(history: pd.DataFrame, exog: pd.DataFrame | None) -> pd.DataFrame:
    features = make_features(history)
    if exog is None:
        return features
    overlap = set(features.columns).intersection(exog.columns)
    aligned = exog.drop(columns=list(overlap)).reindex(features.index)
    return pd.concat([features, aligned], axis=1)


def recursive_forecast(model, columns: list[str], clean: pd.DataFrame, horizon: int,
                       exog: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pronostica `horizon` pasos; cada valor predicho alimenta los rezagos del paso siguiente."""
    frequency = pd.infer_freq(clean.index) or "15min"
    history = clean[["contexto", "es_imputado"]].astype({"contexto": float}).copy()
    stamps: list[pd.Timestamp] = []
    values: list[float] = []
    for _ in range(horizon):
        next_stamp = history.index[-1] + pd.tseries.frequencies.to_offset(frequency)
        history.loc[next_stamp] = [np.nan, 0.0]
        features = _features_with_exog(history, exog)
        row = features.iloc[[-1]]
        missing = [column for column in columns if column not in row.columns or row[column].isna().any()]
        if missing:
            raise ValueError(f"variables sin valor en el paso {next_stamp}: {missing}")
        value = float(np.asarray(model.predict(row[columns]), dtype=float).reshape(-1)[0])
        if not np.isfinite(value):
            raise ValueError("prediccion recursiva no finita")
        history.loc[next_stamp, "contexto"] = value
        stamps.append(next_stamp)
        values.append(value)
    return pd.DataFrame({"timestamp": [str(s) for s in stamps], "paso": range(1, horizon + 1), "prediccion": values})


def origin_forecasts(model, columns: list[str], clean: pd.DataFrame, final_stamps, horizon: int,
                     exog: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pronostica `horizon` pasos desde cada origen dentro del test final (sin reentrenar)."""
    stamps = pd.DatetimeIndex(final_stamps)
    step = clean.index[1] - clean.index[0]
    rows: list[pd.DataFrame] = []
    for origin, start in enumerate(range(0, len(stamps) - horizon + 1, horizon)):
        history = clean.loc[: stamps[start] - step]
        forecast = recursive_forecast(model, columns, history, horizon, exog=exog)
        observed = clean["contexto"].reindex(stamps[start:start + horizon]).to_numpy(dtype=float)
        rows.append(pd.DataFrame({"origen": origin, "y": observed, "prediccion": forecast["prediccion"].to_numpy()}))
    return pd.concat(rows, ignore_index=True)
