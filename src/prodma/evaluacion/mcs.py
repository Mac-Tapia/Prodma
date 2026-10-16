"""Model Confidence Set (Hansen, Lunde y Nason, 2011) con estadistico TR_max y bootstrap de bloques moviles."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _block_indices(rng: np.random.Generator, n: int, block: int, replications: int) -> np.ndarray:
    block = max(1, min(int(block), n))
    starts = rng.integers(0, n - block + 1, size=(replications, int(np.ceil(n / block))))
    indices = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(replications, -1)
    return indices[:, :n]


def _pair_statistics(values: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve medias de diferencias, errores estandar bootstrap y estadistico TR por par."""
    means = values.mean(axis=0)
    differences = means[:, None] - means[None, :]
    boot = values[indices].mean(axis=1)  # B x m
    boot_diff = boot[:, :, None] - boot[:, None, :]  # B x m x m
    standard_error = boot_diff.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(standard_error > 0, differences / standard_error,
                     np.where(differences == 0, 0.0, np.sign(differences) * np.inf))
    return differences, standard_error, t


def model_confidence_set(losses: pd.DataFrame, alpha: float, block_length: int = 10,
                         replications: int = 1000, seed: int = 0) -> dict:
    if losses.shape[1] < 2:
        raise ValueError("MCS requiere al menos dos modelos")
    if losses.shape[0] < 30:
        raise ValueError("MCS requiere al menos 30 bloques alineados")
    if not 0 < alpha < 1:
        raise ValueError("alpha debe estar en (0, 1)")
    if losses.isna().any().any():
        raise ValueError("perdidas con valores faltantes; alinear antes del MCS")
    rng = np.random.default_rng(seed)
    indices = _block_indices(rng, losses.shape[0], block_length, replications)
    current = list(losses.columns)
    eliminated: list[str] = []
    order: list[dict] = []
    p_values: dict[str, float] = {}
    while True:
        values = losses[current].to_numpy(dtype=float)
        differences, standard_error, t = _pair_statistics(values, indices)
        observed = np.nanmax(np.abs(t)) if t.size else 0.0
        if not np.isfinite(observed):
            boot_stat = np.full(len(indices), np.inf)
        else:
            boot = values[indices].mean(axis=1)
            boot_diff = boot[:, :, None] - boot[:, None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                boot_t = np.abs((boot_diff - differences[None]) / standard_error[None])
            boot_t = np.where(np.isfinite(boot_t), boot_t, 0.0)
            boot_stat = boot_t.reshape(len(indices), -1).max(axis=1)
        p_value = float(np.mean(boot_stat >= observed))
        if p_value >= alpha or len(current) == 1:
            for name in current:
                p_values.setdefault(name, p_value)
            return {"mcs": current, "eliminados": eliminated, "orden_eliminacion": order, "p_valores": p_values}
        average = np.nanmean(np.where(np.isfinite(t), t, np.nan), axis=1)
        worst = current[int(np.nanargmax(average))]
        p_values[worst] = p_value
        order.append({"modelo": worst, "p_valor_al_eliminar": p_value})
        eliminated.append(worst)
        current.remove(worst)
