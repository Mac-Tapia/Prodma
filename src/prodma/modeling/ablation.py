"""Ablacion temporal de conjuntos exogenos con transformaciones fold-local."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prodma.modeling.features import join_exogenous_features


@dataclass(frozen=True)
class FeatureSetBundle:
    endogenous: pd.DataFrame
    y: pd.Series


@dataclass(frozen=True)
class AblationResult:
    selected: str
    reason: str
    fold_metrics: pd.DataFrame
    selected_columns: tuple[str, ...]


def _positions(fold) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(fold, tuple):
        return np.asarray(fold[0]), np.asarray(fold[1])
    return np.asarray(fold.train_index), np.asarray(fold.validation_index)


CALENDAR_TOKENS = ("feriado", "fin_semana", "no_laborable", "hora_", "dia_", "anio_")
WEATHER_TOKENS = ("temp_", "temperatura_")
ABLATION_ORDER = ("calendar", "weather", "calendar_weather", "socioeconomic", "all_eligible")


def feature_set_columns(exogenous: pd.DataFrame) -> dict[str, list[str]]:
    """Grupos de exogenas. Cada columna elegible pertenece a un grupo o a `socioeconomic`."""
    calendar = [column for column in exogenous if column.startswith(CALENDAR_TOKENS)]
    weather = [column for column in exogenous if column.startswith(WEATHER_TOKENS)]
    grouped = set(calendar) | set(weather)
    socioeconomic = [column for column in exogenous if column not in grouped]
    return {
        "endogenous_only": [],
        "calendar": calendar,
        "weather": weather,
        "calendar_weather": list(dict.fromkeys(calendar + weather)),
        "socioeconomic": socioeconomic,
        "all_eligible": list(exogenous.columns),
    }


def compare_feature_sets(base: FeatureSetBundle, exogenous: pd.DataFrame, folds,
                         minimum_improvement: float = 0.0) -> AblationResult:
    if not base.endogenous.index.equals(base.y.index) or not exogenous.index.equals(base.y.index):
        raise ValueError("features y objetivo deben compartir indice y orden")
    candidates = feature_set_columns(exogenous)
    rows: list[dict] = []
    errors: dict[str, list[float]] = {name: [] for name in candidates}
    for name, columns in candidates.items():
        x = join_exogenous_features(base.endogenous, exogenous[columns])
        for fold_number, fold in enumerate(folds):
            train, validation = _positions(fold)
            estimator = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
            estimator.fit(x.iloc[train], base.y.iloc[train])
            prediction = estimator.predict(x.iloc[validation])
            rmse = float(np.sqrt(np.mean((base.y.iloc[validation].to_numpy() - prediction) ** 2)))
            errors[name].append(rmse)
            rows.append({"feature_set": name, "fold": fold_number, "RMSE": rmse, "columns": len(x.columns)})
    selected = "endogenous_only"
    for name in ABLATION_ORDER:
        current = np.asarray(errors[selected])
        candidate = np.asarray(errors[name])
        relative = (current.mean() - candidate.mean()) / max(current.mean(), np.finfo(float).eps)
        if relative >= minimum_improvement and (candidate < current).all():
            selected = name
    reason = "mejora_estable_fuera_de_muestra" if selected != "endogenous_only" else "no_mejora_fuera_de_muestra"
    selected_columns = tuple(base.endogenous.columns) + tuple(candidates[selected])
    return AblationResult(selected, reason, pd.DataFrame(rows), selected_columns)
