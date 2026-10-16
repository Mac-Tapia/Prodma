"""Entrenamiento temporal anidado de candidatos aprobados."""
from __future__ import annotations

import importlib.metadata
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from prodma.modeling.adapters import ForecastAdapter, ForecastRequest, TrainingSlice
from prodma.modeling.checkpoints import CheckpointKey, store_checkpoint
from prodma.modeling.registry import ModelSpec
from prodma.modeling.splits import TrainingPartition


@dataclass(frozen=True)
class CleanTrainingData:
    x: pd.DataFrame
    y: pd.Series
    context: pd.Series
    observed_mask: pd.Series
    imputed_mask: pd.Series


PREDICTION_COLUMNS = ("timestamp", "modelo", "trial", "fold", "horizonte", "y", "prediccion")


@dataclass(frozen=True)
class TrainingResult:
    table: pd.DataFrame
    checkpoints: tuple[Path, ...]
    traces: pd.DataFrame
    timings: pd.DataFrame
    predictions: pd.DataFrame
    final_test: pd.DataFrame


def _prediction_rows(name: str, trial: int, fold: int, horizon: int, data: CleanTrainingData,
                     positions: np.ndarray, prediction: pd.Series) -> list[dict]:
    observed = data.y.iloc[positions]
    return [
        {"timestamp": str(stamp), "modelo": name, "trial": int(trial), "fold": int(fold),
         "horizonte": int(horizon), "y": float(actual), "prediccion": float(value)}
        for stamp, actual, value in zip(observed.index, observed.to_numpy(), prediction.to_numpy())
    ]


def build_training_slice(clean: pd.DataFrame, features: pd.DataFrame) -> CleanTrainingData:
    required = {"limpio", "contexto", "es_observado", "es_imputado"}
    missing = required - set(clean)
    if missing:
        raise ValueError(f"faltan campos de calidad: {sorted(missing)}")
    if not clean.index.equals(features.index):
        raise ValueError("features desalineadas con serie limpia")
    observed = clean["es_observado"].fillna(False).astype(bool)
    imputed = clean["es_imputado"].fillna(False).astype(bool)
    eligible = observed & clean["limpio"].notna()
    return CleanTrainingData(
        x=features.loc[eligible].copy(),
        y=clean.loc[eligible, "limpio"].astype(float).copy(),
        context=clean["contexto"].astype(float).copy(),
        observed_mask=observed,
        imputed_mask=imputed,
    )


def _version(model) -> str:
    module = type(model).__module__.split(".")[0]
    package = {"sklearn": "scikit-learn"}.get(module, module)
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _spec_with_trial(spec: ModelSpec, trial: optuna.Trial) -> tuple[ModelSpec, dict]:
    params: dict = {}
    if spec.name == "Ridge":
        from sklearn.linear_model import Ridge
        params["alpha"] = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
        factory = lambda seed: Ridge(alpha=params["alpha"])
    elif spec.name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        params["n_estimators"] = trial.suggest_int("n_estimators", 30, 120)
        params["max_depth"] = trial.suggest_int("max_depth", 3, 16)
        factory = lambda seed: RandomForestRegressor(**params, random_state=seed, n_jobs=1)
    else:
        factory = spec.factory
    return ModelSpec(spec.name, factory, spec.resolution, spec.supports_exogenous, spec.backend, spec.family), params


def _positions(index: pd.DatetimeIndex, timestamps: pd.DatetimeIndex) -> np.ndarray:
    positions = index.get_indexer(timestamps)
    if (positions < 0).any():
        raise ValueError("fold contiene timestamps sin objetivo observado")
    return positions


def train_approved_models(specs: dict[str, ModelSpec], data: CleanTrainingData,
                          partition: TrainingPartition, checkpoint_dir: Path | str,
                          matrix_sha256: str, exogenous_sha256: str, seed: int,
                          n_trials: int = 5, selected_feature_set: str = "preselected",
                          exogenous_columns: tuple[str, ...] = ()) -> TrainingResult:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    exog_frame = data.x.loc[:, list(exogenous_columns)] if exogenous_columns else None
    table_rows: list[dict] = []
    trace_rows: list[dict] = []
    timing_rows: list[dict] = []
    checkpoints: list[Path] = []
    prediction_rows_by_trial: dict[tuple[str, int], list[dict]] = {}
    final_prediction_rows: list[dict] = []
    final_test_rows: list[dict] = []
    for name, base_spec in specs.items():
        started = time.perf_counter()
        try:
            trial_parameters: dict[int, dict] = {}

            def objective(trial: optuna.Trial) -> float:
                trial_spec, parameters = _spec_with_trial(base_spec, trial)
                trial_parameters[trial.number] = parameters
                fold_errors: list[float] = []
                for fold_number, fold in enumerate(partition.inner_folds):
                    train_positions = _positions(data.y.index, fold.train_index)
                    validation_positions = _positions(data.y.index, fold.validation_index)
                    training = TrainingSlice(data.x, data.y, train_positions, validation_positions, exog_frame)
                    adapter = ForecastAdapter(trial_spec, seed + trial.number).fit(training)
                    prediction = adapter.predict(ForecastRequest(
                        data.x.iloc[validation_positions], pd.DatetimeIndex(data.y.index[validation_positions]),
                        len(validation_positions),
                        None if exog_frame is None else exog_frame.iloc[validation_positions],
                    ))
                    rmse = float(np.sqrt(np.mean((data.y.iloc[validation_positions].to_numpy() - prediction.to_numpy()) ** 2)))
                    fold_errors.append(rmse)
                    prediction_rows_by_trial.setdefault((name, trial.number), []).extend(_prediction_rows(
                        name, trial.number, fold_number, fold.horizon, data, validation_positions, prediction
                    ))
                    trace_rows.append({
                        "modelo": name, "trial": trial.number, "fold": fold_number, "RMSE": rmse,
                        "train_end": str(fold.train_index[-1]), "validation_start": str(fold.validation_index[0]),
                        "validation_end": str(fold.validation_index[-1]), "seed": seed + trial.number,
                        "feature_set": selected_feature_set, "parameters": json.dumps(parameters, sort_keys=True),
                    })
                return float(np.mean(fold_errors))

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
            actual_trials = n_trials if base_spec.name in {"Ridge", "RandomForest"} else 1
            study.optimize(objective, n_trials=actual_trials, show_progress_bar=False)
            best_spec, best_params = _spec_with_trial(base_spec, study.best_trial)
            calibration_positions = _positions(data.y.index, partition.calibration_index)
            final_adapter = ForecastAdapter(best_spec, seed).fit(
                TrainingSlice(data.x, data.y, calibration_positions, np.array([], dtype=int), exog_frame)
            )
            final_prediction_rows.extend(prediction_rows_by_trial.get((name, study.best_trial.number), []))
            final_positions = data.y.index.get_indexer(partition.final_test_index)
            if (final_positions < 0).any():
                raise ValueError("test final sin objetivo observado")
            final_prediction = final_adapter.predict(ForecastRequest(
                data.x.iloc[final_positions], pd.DatetimeIndex(partition.final_test_index), len(final_positions),
                None if exog_frame is None else exog_frame.iloc[final_positions],
            ))
            final_test_rows.extend(
                {"timestamp": str(stamp), "modelo": name, "paso": step + 1,
                 "y": float(actual), "prediccion": float(value)}
                for step, (stamp, actual, value) in enumerate(zip(
                    partition.final_test_index, data.y.iloc[final_positions].to_numpy(), final_prediction.to_numpy()
                ))
            )
            library_version = _version(final_adapter.model)
            key = CheckpointKey(
                matrix_sha256, exogenous_sha256, tuple(data.x.columns), name, library_version,
                final_adapter.effective_backend, seed,
                transformations=("fold_local_imputer_scaler_when_required",),
                hyperparameters=json.dumps(best_params, sort_keys=True),
                train_end=str(partition.calibration_index[-1]), source_version="working-tree",
            )
            checkpoint = store_checkpoint(checkpoint_dir, key, {
                "model": final_adapter.model, "columns": tuple(data.x.columns),
                "feature_set": selected_feature_set, "partition": partition,
            })
            checkpoints.append(checkpoint)
            elapsed = time.perf_counter() - started
            table_rows.append({
                "modelo": name, "estado": "ENTRENADO", "inner_RMSE": float(study.best_value),
                "best_trial": int(study.best_trial.number), "parameters": json.dumps(best_params, sort_keys=True),
                "matrix_sha256": matrix_sha256, "exogenous_sha256": exogenous_sha256,
                "backend_efectivo": final_adapter.effective_backend, "seed": seed,
                "train_end": str(partition.calibration_index[-1]), "test_start": str(partition.final_test_index[0]),
                "error": "",
            })
            timing_rows.append({"modelo": name, "fase": "tuning_y_refit", "segundos": elapsed})
        except Exception as exc:
            elapsed = time.perf_counter() - started
            table_rows.append({
                "modelo": name, "estado": "FALLIDO", "inner_RMSE": np.nan,
                "matrix_sha256": matrix_sha256, "exogenous_sha256": exogenous_sha256,
                "backend_efectivo": "none", "seed": seed, "error": f"{type(exc).__name__}: {exc}",
            })
            timing_rows.append({"modelo": name, "fase": "fallo", "segundos": elapsed})
    return TrainingResult(
        pd.DataFrame(table_rows), tuple(checkpoints), pd.DataFrame(trace_rows), pd.DataFrame(timing_rows),
        pd.DataFrame(final_prediction_rows, columns=list(PREDICTION_COLUMNS)),
        final_test=pd.DataFrame(final_test_rows, columns=["timestamp", "modelo", "paso", "y", "prediccion"]),
    )
