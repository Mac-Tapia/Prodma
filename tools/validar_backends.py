"""Microentrenamientos reales para registrar el backend efectivo."""
from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "no_instalado"


def _tree_data(seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(128, 6)).astype(np.float32)
    y = (2 * x[:, 0] - x[:, 1] + 0.1 * rng.normal(size=len(x))).astype(np.float32)
    return x[:112], y[:112], x[112:]


def _xgboost(seed: int) -> dict:
    from xgboost import XGBRegressor
    x, y, future = _tree_data(seed)
    error = "ninguno"
    requested = "cuda:0"
    try:
        model = XGBRegressor(n_estimators=8, max_depth=2, tree_method="hist", device="cuda:0",
                             random_state=seed, verbosity=0)
        model.fit(x, y)
        prediction = model.predict(future)
        effective, device = "gpu", "cuda:0"
    except Exception as exc:
        error = f"GPU no disponible: {type(exc).__name__}: {exc}"
        model = XGBRegressor(n_estimators=8, max_depth=2, tree_method="hist", device="cpu",
                             random_state=seed, verbosity=0)
        model.fit(x, y)
        prediction = model.predict(future)
        effective, device = "cpu", "cpu"
    return {"modelo": "XGBoost", "solicitado": requested, "efectivo": effective,
            "fit_ok": True, "predict_ok": bool(np.isfinite(prediction).all()),
            "dispositivo": device, "version": _version("xgboost"), "motivo_fallback": error}


def _catboost(seed: int) -> dict:
    from catboost import CatBoostRegressor
    x, y, future = _tree_data(seed)
    error = "ninguno"
    try:
        model = CatBoostRegressor(iterations=8, depth=2, task_type="GPU", devices="0",
                                  random_seed=seed, verbose=False)
        model.fit(x, y)
        prediction = model.predict(future)
        effective, device = "gpu", "cuda:0"
    except Exception as exc:
        error = f"GPU no disponible: {type(exc).__name__}: {exc}"
        model = CatBoostRegressor(iterations=8, depth=2, task_type="CPU",
                                  random_seed=seed, verbose=False)
        model.fit(x, y)
        prediction = model.predict(future)
        effective, device = "cpu", "cpu"
    return {"modelo": "CatBoost", "solicitado": "cuda:0", "efectivo": effective,
            "fit_ok": True, "predict_ok": bool(np.isfinite(prediction).all()),
            "dispositivo": device, "version": _version("catboost"), "motivo_fallback": error}


def _lightgbm(seed: int) -> dict:
    from lightgbm import LGBMRegressor
    x, y, future = _tree_data(seed)
    error = "ninguno"
    try:
        model = LGBMRegressor(n_estimators=8, max_depth=2, device_type="gpu",
                              random_state=seed, verbosity=-1)
        model.fit(x, y)
        prediction = model.predict(future)
        effective, device = "gpu", "opencl"
    except Exception as exc:
        error = f"GPU no disponible: {type(exc).__name__}: {exc}"
        model = LGBMRegressor(n_estimators=8, max_depth=2, device_type="cpu",
                              random_state=seed, verbosity=-1)
        model.fit(x, y)
        prediction = model.predict(future)
        effective, device = "cpu", "cpu"
    return {"modelo": "LightGBM", "solicitado": "opencl_gpu", "efectivo": effective,
            "fit_ok": True, "predict_ok": bool(np.isfinite(prediction).all()),
            "dispositivo": device, "version": _version("lightgbm"), "motivo_fallback": error}


def _lstm(seed: int) -> dict:
    from prodma.forecast_models import create_forecaster
    index = pd.date_range("2026-01-01", periods=64, freq="h")
    y = pd.Series(20 + np.sin(np.arange(64) * 2 * np.pi / 24), index=index)
    model = create_forecaster("LSTM", seasonal_periods=8, device="auto", epochs=2,
                              hidden_size=8, seed=seed)
    model.fit(y)
    prediction = model.predict(4)
    device = str(model.device_)
    effective = "gpu" if device.startswith("cuda") else "cpu"
    reason = "ninguno" if effective == "gpu" else "torch.cuda.is_available()=False; CPU efectivo"
    return {"modelo": "LSTM", "solicitado": "auto", "efectivo": effective,
            "fit_ok": True, "predict_ok": bool(np.isfinite(prediction).all()),
            "dispositivo": device, "version": _version("torch"), "motivo_fallback": reason}


def run_backend_smoke(output: Path | str, seed: int = 42) -> pd.DataFrame:
    rows = []
    for runner in (_xgboost, _catboost, _lightgbm, _lstm):
        try:
            rows.append(runner(seed))
        except Exception as exc:
            rows.append({"modelo": runner.__name__.lstrip("_").title(), "solicitado": "auto",
                         "efectivo": "none", "fit_ok": False, "predict_ok": False,
                         "dispositivo": "none", "version": "unknown",
                         "motivo_fallback": f"{type(exc).__name__}: {exc}"})
    result = pd.DataFrame(rows)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    result.to_csv(temporary, index=False)
    temporary.replace(destination)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_backend_smoke(args.output, args.seed)
    print(result.to_string(index=False))
    return 0 if result[["fit_ok", "predict_ok"]].all().all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
