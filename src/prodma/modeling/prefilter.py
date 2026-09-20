"""Prefiltro barato y aislado del test final."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from prodma.modeling.registry import ModelSpec


@dataclass(frozen=True)
class PrefilterPolicy:
    max_seconds: float = 300.0


def run_prefilter(specs: dict[str, ModelSpec], x: pd.DataFrame, y: pd.Series,
                  train_positions, validation_positions, policy: PrefilterPolicy, seed: int) -> pd.DataFrame:
    rows = []
    for name, spec in specs.items():
        started = time.perf_counter()
        try:
            model = spec.factory(seed)
            model.fit(x.iloc[train_positions], y.iloc[train_positions])
            prediction = np.asarray(model.predict(x.iloc[validation_positions]), dtype=float)
            if len(prediction) != len(validation_positions) or not np.isfinite(prediction).all():
                raise ValueError("prediccion no finita o de longitud incorrecta")
            elapsed = time.perf_counter() - started
            if elapsed > policy.max_seconds:
                raise TimeoutError(f"excede {policy.max_seconds} segundos")
            rmse = float(np.sqrt(np.mean((y.iloc[validation_positions].to_numpy() - prediction) ** 2)))
            rows.append({"modelo": name, "estado": "APROBADO", "RMSE": rmse, "segundos": elapsed, "error": ""})
        except (ImportError, ModuleNotFoundError) as exc:
            rows.append({"modelo": name, "estado": "NO_DISPONIBLE", "RMSE": np.nan,
                         "segundos": time.perf_counter() - started, "error": str(exc)})
        except Exception as exc:
            rows.append({"modelo": name, "estado": "RECHAZADO", "RMSE": np.nan,
                         "segundos": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)
