"""Selección de features por algoritmo genético (GA).

Uso previsto: en vez de entrenar los 6 modelos ML con TODAS las columnas de
`features()` (48 en la versión actual del notebook, entre calendario, armónicos,
climatología, lags y rolling stats), se elige por serie (medidor, variable) el
subconjunto que mejor generaliza en validación cruzada temporal -- con ventanas de
entrenamiento limitadas (sobre todo en modo rápido), un subconjunto más chico puede
generalizar mejor que usar las 48 columnas siempre.

No agrega una dependencia nueva (DEAP, etc.): es un GA simple y auto-contenido
(cromosoma binario = qué columnas incluir, selección por torneo, cruce de un punto,
mutación bit a bit), suficiente para un espacio de búsqueda de decenas de features.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def _fitness_rmse_cv(
    X: pd.DataFrame, y: pd.Series, mascara: np.ndarray, cv: Sequence[tuple[np.ndarray, np.ndarray]]
) -> float:
    """-RMSE promedio en CV temporal de un Ridge simple sobre las columnas de `mascara`.

    Ridge (no uno de los 6 candidatos reales) es deliberado: la selección de features
    debe ser barata de evaluar -- se llama cientos de veces por serie -- y sirve como
    proxy razonable de qué columnas aportan señal real frente a ruido, sin acoplar la
    selección a un solo modelo de los 6 que compiten después.
    """
    if mascara.sum() == 0:
        return -np.inf
    columnas = X.columns[mascara.astype(bool)]
    errores = []
    for idx_tr, idx_va in cv:
        Xtr, Xva = X.iloc[idx_tr][columnas], X.iloc[idx_va][columnas]
        ytr, yva = y.iloc[idx_tr], y.iloc[idx_va]
        modelo = Ridge(alpha=1.0)
        modelo.fit(Xtr, ytr)
        pred = modelo.predict(Xva)
        errores.append(np.sqrt(np.mean((yva.to_numpy() - pred) ** 2)))
    return -float(np.mean(errores))


def seleccionar_features_ga(
    X: pd.DataFrame,
    y: pd.Series,
    cv: Sequence[tuple[np.ndarray, np.ndarray]],
    poblacion: int = 20,
    generaciones: int = 15,
    prob_mutacion: float = 0.05,
    semilla: int = 42,
) -> tuple[list[str], float]:
    """Devuelve (columnas_seleccionadas, fitness) -- nunca una lista vacía.

    `cv` son los mismos folds de `temporal_cv` ya usados en el resto de la celda de
    entrenamiento (no se inventa una validación distinta). Determinista dado `semilla`.
    """
    if X.shape[1] == 0:
        raise ValueError('X no tiene columnas')
    rng = np.random.RandomState(semilla)
    n_features = X.shape[1]

    poblacion_actual = rng.randint(0, 2, size=(poblacion, n_features))
    for fila in poblacion_actual:
        if fila.sum() == 0:
            fila[rng.randint(n_features)] = 1

    mejor_mascara = poblacion_actual[0].copy()
    mejor_fitness = -np.inf

    for _ in range(generaciones):
        fitnesses = np.array([_fitness_rmse_cv(X, y, ind, cv) for ind in poblacion_actual])
        idx_mejor = int(fitnesses.argmax())
        if fitnesses[idx_mejor] > mejor_fitness:
            mejor_fitness = float(fitnesses[idx_mejor])
            mejor_mascara = poblacion_actual[idx_mejor].copy()

        nueva_poblacion = []
        for _ in range(poblacion):
            i, j = rng.randint(0, poblacion, 2)
            padre1 = poblacion_actual[i] if fitnesses[i] > fitnesses[j] else poblacion_actual[j]
            i, j = rng.randint(0, poblacion, 2)
            padre2 = poblacion_actual[i] if fitnesses[i] > fitnesses[j] else poblacion_actual[j]
            punto = rng.randint(1, n_features) if n_features > 1 else 1
            hijo = np.concatenate([padre1[:punto], padre2[punto:]])
            mutar = rng.random(n_features) < prob_mutacion
            hijo = np.where(mutar, 1 - hijo, hijo)
            if hijo.sum() == 0:
                hijo[rng.randint(n_features)] = 1
            nueva_poblacion.append(hijo)
        poblacion_actual = np.array(nueva_poblacion)

    columnas_seleccionadas = list(X.columns[mejor_mascara.astype(bool)])
    return columnas_seleccionadas, mejor_fitness
