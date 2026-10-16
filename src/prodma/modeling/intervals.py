"""Intervalos conformales divididos calibrados con residuos fuera del ajuste."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

MINIMUM_CALIBRATION = 30


class CalibrationInsufficient(ValueError):
    state = "CALIBRACION_INSUFICIENTE"


def split_conformal_radius(calibration_residuals, alpha: float) -> float:
    """Radio simetrico con correccion de muestra finita sobre residuos absolutos."""
    residuals = np.abs(np.asarray(calibration_residuals, dtype=float))
    if not 0 < alpha < 1:
        raise ValueError("alpha debe estar en (0, 1)")
    if residuals.size < MINIMUM_CALIBRATION or not np.isfinite(residuals).all():
        raise CalibrationInsufficient(
            f"{CalibrationInsufficient.state}: se requieren {MINIMUM_CALIBRATION} residuos finitos"
        )
    n = residuals.size
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        raise CalibrationInsufficient(f"{CalibrationInsufficient.state}: alpha demasiado pequeno para n={n}")
    return float(np.sort(residuals)[rank - 1])


def calibrated_interval(prediction: pd.Series, calibration_residuals, alpha: float):
    radius = split_conformal_radius(calibration_residuals, alpha)
    return prediction - radius, prediction + radius


def conditional_coverage(y: pd.Series, lower: pd.Series, upper: pd.Series, horizon: pd.Series) -> pd.DataFrame:
    """Cobertura empirica de cada horizonte; no promedia horizontes entre si."""
    if not (y.index.equals(lower.index) and y.index.equals(upper.index) and y.index.equals(horizon.index)):
        raise ValueError("y, limites y horizonte deben compartir indice")
    covered = (y >= lower) & (y <= upper)
    frame = pd.DataFrame({"horizonte": horizon.to_numpy(), "cubierto": covered.to_numpy()})
    grouped = frame.groupby("horizonte")["cubierto"]
    return pd.DataFrame({
        "horizonte": grouped.size().index.to_numpy(),
        "n": grouped.size().to_numpy(),
        "cobertura": grouped.mean().to_numpy(),
    })
