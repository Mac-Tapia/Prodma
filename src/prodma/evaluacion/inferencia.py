"""Pruebas de inferencia sobre perdidas pareadas fuera de muestra."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _aligned(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if not a.index.equals(b.index):
        raise ValueError("perdidas no alineadas por bloque")
    if len(a) < 2:
        raise ValueError("se requieren al menos dos pares")
    return a.to_numpy(dtype=float), b.to_numpy(dtype=float)


def paired_wilcoxon(loss_a: pd.Series, loss_b: pd.Series) -> dict:
    a, b = _aligned(loss_a, loss_b)
    difference = a - b
    if np.allclose(difference, 0):
        return {"n": len(a), "mejor_diferencia": 0.0, "p_valor": 1.0}
    statistic = stats.wilcoxon(a, b, alternative="two-sided")
    return {"n": len(a), "mejor_diferencia": float(difference.mean()), "p_valor": float(statistic.pvalue)}


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, horizon: int) -> dict:
    """Diebold-Mariano con varianza HAC (Bartlett) de lags = horizonte - 1."""
    a, b = _aligned(loss_a, loss_b)
    d = a - b
    n = len(d)
    lags = max(int(horizon) - 1, 0)
    centred = d - d.mean()
    variance = float(centred @ centred) / n
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        variance += 2 * weight * float(centred[lag:] @ centred[:-lag]) / n
    if variance <= 0:
        return {"estadistico": 0.0 if d.mean() == 0 else np.inf * np.sign(d.mean()),
                "p_valor": 1.0 if d.mean() == 0 else 0.0, "lags": lags}
    statistic = d.mean() / np.sqrt(variance / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(statistic)))
    return {"estadistico": float(statistic), "p_valor": float(p_value), "lags": lags}


def friedman_holm(table: pd.DataFrame) -> dict:
    """Friedman sobre bloques x modelos y Wilcoxon pareado contra el mejor, con Holm."""
    if table.shape[1] < 3 or table.shape[0] < 3:
        raise ValueError("Friedman requiere al menos tres modelos y tres bloques")
    columns = [table[name].to_numpy(dtype=float) for name in table.columns]
    friedman = stats.friedmanchisquare(*columns)
    best = table.mean().idxmin()
    rivals = [name for name in table.columns if name != best]
    raw = [paired_wilcoxon(table[name], table[best])["p_valor"] for name in rivals]
    order = np.argsort(raw)
    adjusted = np.empty(len(raw))
    running = 0.0
    m = len(raw)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * raw[index]))
        adjusted[index] = running
    pairwise = pd.DataFrame({"modelo": rivals, "p_valor": raw, "p_holm": adjusted})
    return {"p_friedman": float(friedman.pvalue), "mejor_modelo": best, "pairwise": pairwise}
