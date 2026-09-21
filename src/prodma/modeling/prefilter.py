"""Prefiltro barato y aislado del test final."""
from __future__ import annotations

import importlib.metadata
import multiprocessing as mp
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from prodma.modeling.registry import ModelSpec
from prodma.modeling.adapters import ForecastAdapter, ForecastRequest, TrainingSlice


@dataclass(frozen=True)
class PrefilterPolicy:
    max_seconds: float = 300.0
    isolated_process: bool = True


def _library_version(model) -> str:
    module = type(model).__module__.split(".")[0]
    package = {"sklearn": "scikit-learn"}.get(module, module)
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _fit_predict(spec: ModelSpec, training: TrainingSlice, seed: int) -> dict:
    import psutil

    adapter = ForecastAdapter(spec, seed).fit(training)
    positions = np.asarray(training.validation_positions)
    prediction = adapter.predict(ForecastRequest(
        x=training.x.iloc[positions], index=pd.DatetimeIndex(training.y.index[positions]),
        horizon=len(positions),
        exogenous=training.exogenous.iloc[positions] if training.exogenous is not None else None,
    ))
    actual = training.y.iloc[positions].to_numpy(dtype=float)
    return {
        "RMSE": float(np.sqrt(np.mean((actual - prediction.to_numpy()) ** 2))),
        "backend_efectivo": adapter.effective_backend,
        "memoria_pico_mb": psutil.Process().memory_info().rss / (1024 ** 2),
        "version": _library_version(adapter.model),
    }


def _registered_worker(name: str, training: TrainingSlice, seed: int, queue) -> None:
    from prodma.modeling.registry import model_registry
    try:
        queue.put(("ok", _fit_predict(model_registry()[name], training, seed)))
    except Exception as exc:
        queue.put(("error", (type(exc).__name__, str(exc))))


def _isolated_registered(name: str, training: TrainingSlice, seed: int, timeout: float) -> dict:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_registered_worker, args=(name, training, seed, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        raise TimeoutError(f"excede {timeout} segundos")
    if queue.empty():
        raise RuntimeError(f"proceso termino sin resultado (exitcode={process.exitcode})")
    state, payload = queue.get()
    if state == "error":
        error_type, message = payload
        if error_type == "MissingOptionalDependency":
            raise ModuleNotFoundError(message)
        raise RuntimeError(f"{error_type}: {message}")
    return payload


def run_prefilter(specs: dict[str, ModelSpec], x: pd.DataFrame | TrainingSlice,
                  y: pd.Series | PrefilterPolicy | None = None, train_positions=None,
                  validation_positions=None, policy: PrefilterPolicy | None = None, seed: int = 42) -> pd.DataFrame:
    if isinstance(x, TrainingSlice):
        training = x
        if isinstance(y, PrefilterPolicy):
            policy = y
    else:
        if y is None or isinstance(y, PrefilterPolicy):
            raise ValueError("y es obligatorio para la interfaz historica")
        training = TrainingSlice(x=x, y=y, train_positions=np.asarray(train_positions),
                                 validation_positions=np.asarray(validation_positions))
    policy = policy or PrefilterPolicy()
    registered_names = set(__import__("prodma.modeling.registry", fromlist=["model_registry"]).model_registry())
    rows = []
    for name, spec in specs.items():
        started = time.perf_counter()
        try:
            if policy.isolated_process and name in registered_names:
                result = _isolated_registered(name, training, seed, policy.max_seconds)
            else:
                result = _fit_predict(spec, training, seed)
            elapsed = time.perf_counter() - started
            rows.append({"modelo": name, "estado": "APROBADO", "segundos": elapsed, "error": "",
                         "backend_solicitado": spec.backend, **result})
        except TimeoutError as exc:
            rows.append({"modelo": name, "estado": "TIMEOUT", "RMSE": np.nan,
                         "segundos": time.perf_counter() - started, "error": str(exc),
                         "backend_solicitado": spec.backend, "backend_efectivo": "none",
                         "memoria_pico_mb": np.nan, "version": "unknown"})
        except (ImportError, ModuleNotFoundError) as exc:
            rows.append({"modelo": name, "estado": "NO_DISPONIBLE", "RMSE": np.nan,
                         "segundos": time.perf_counter() - started, "error": str(exc),
                         "backend_solicitado": spec.backend, "backend_efectivo": "none",
                         "memoria_pico_mb": np.nan, "version": "unknown"})
        except Exception as exc:
            rows.append({"modelo": name, "estado": "RECHAZADO", "RMSE": np.nan,
                         "segundos": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}",
                         "backend_solicitado": spec.backend, "backend_efectivo": "none",
                         "memoria_pico_mb": np.nan, "version": "unknown"})
    return pd.DataFrame(rows)
