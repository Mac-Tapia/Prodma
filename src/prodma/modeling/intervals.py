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


def calibrate_final_test(final: pd.DataFrame, horizons, alpha: float) -> pd.DataFrame:
    """Calibra con la primera mitad del test final (intocado) y mide cobertura en la segunda.

    `horizons` agrupa cada paso en el menor horizonte configurado que lo contiene.
    """
    ordered_horizons = sorted(int(h) for h in horizons)
    rows: list[pd.DataFrame] = []
    for name, group in final.sort_values("paso").groupby("modelo", sort=True):
        steps = group["paso"].to_numpy(dtype=int)
        residuals = (group["y"] - group["prediccion"]).to_numpy(dtype=float)
        half = len(group) // 2
        radius = split_conformal_radius(residuals[:half], alpha)
        evaluation = group.iloc[half:]
        prediction = pd.Series(evaluation["prediccion"].to_numpy(), index=evaluation["paso"].to_numpy())
        observed = pd.Series(evaluation["y"].to_numpy(), index=evaluation["paso"].to_numpy())
        bucket = pd.Series(
            [next((h for h in ordered_horizons if step <= h), ordered_horizons[-1]) for step in evaluation["paso"]],
            index=evaluation["paso"].to_numpy(),
        )
        table = conditional_coverage(observed, prediction - radius, prediction + radius, bucket)
        table.insert(0, "modelo", name)
        table["radio"] = radius
        table["alpha"] = alpha
        table["mitad_calibracion"] = int(half)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["modelo", "horizonte", "n", "cobertura", "radio", "alpha", "mitad_calibracion"]
    )


def calibrate_rolling(origin_frame: pd.DataFrame, horizon: int, alpha: float) -> pd.DataFrame:
    """Calibra con la primera mitad de los origenes y mide cobertura en la segunda.

    `origin_frame` tiene columnas origen, y, prediccion; cada origen aporta `horizon` pasos.
    """
    origins = sorted(origin_frame["origen"].unique())
    half = len(origins) // 2
    if half == 0:
        raise CalibrationInsufficient(f"{CalibrationInsufficient.state}: se requieren al menos dos origenes")
    calibration = origin_frame[origin_frame["origen"].isin(origins[:half])]
    evaluation = origin_frame[origin_frame["origen"].isin(origins[half:])]
    radius = split_conformal_radius((calibration["y"] - calibration["prediccion"]).to_numpy(dtype=float), alpha)
    prediction = pd.Series(evaluation["prediccion"].to_numpy(dtype=float), index=range(len(evaluation)))
    observed = pd.Series(evaluation["y"].to_numpy(dtype=float), index=range(len(evaluation)))
    bucket = pd.Series(int(horizon), index=range(len(evaluation)))
    table = conditional_coverage(observed, prediction - radius, prediction + radius, bucket)
    table["radio"] = radius
    table["alpha"] = alpha
    table["origenes_calibracion"] = int(half)
    return table
