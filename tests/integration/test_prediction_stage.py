from dataclasses import replace
import hashlib

import numpy as np
import pandas as pd

from prodma.artifacts import ManifestStore, RunLayout
from prodma.config import load_config
from prodma.stages.prediction import run_prediction
from prodma.stages.training import run_training
from prodma.stages.evaluation import run_evaluation

ROOT = __import__("pathlib").Path(__file__).parents[2]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_pipeline(tmp_path, name, values, temperature):
    base = load_config(ROOT / "config/default.yaml")
    raw = dict(base.raw)
    raw["training"] = {"folds": 2, "min_train": 300, "n_trials": 1}
    config = replace(base, output_root=tmp_path, horizons=(24, 96), raw=raw)
    layout = RunLayout.create(tmp_path, name)
    index = pd.date_range("2025-01-01", periods=1200, freq="15min", name="Fecha")
    matrix = pd.DataFrame({"M1_P": values}, index=index)
    quality = pd.DataFrame({
        "M1__P__observado": values, "M1__P__limpio": values, "M1__P__contexto": values,
        "M1__P__es_observado": True, "M1__P__es_imputado": False,
    }, index=index)
    exogenous = pd.DataFrame({"feriado": 0.0, "temperatura_c": temperature}, index=index)
    data_dir = layout.stage("02_limpieza", "datos")
    paths = {key: data_dir / filename for key, filename in {
        "matrix": "matriz_limpia.parquet", "quality": "matriz_calidad.parquet",
        "exogenous_historical": "matriz_exogenas_historicas.parquet",
    }.items()}
    matrix.to_parquet(paths["matrix"])
    quality.to_parquet(paths["quality"])
    exogenous.to_parquet(paths["exogenous_historical"])
    ManifestStore(layout).update_stage("02_limpieza", "SUCCESS", artifacts={
        **{key: str(value) for key, value in paths.items()}, "sha256": _sha(paths["matrix"]),
        "exogenous_sha256": _sha(paths["exogenous_historical"]),
    })
    assert run_training(config, layout, model_names=("Ridge",)).status == "SUCCESS"
    assert run_evaluation(config, layout).status == "SUCCESS"
    return config, layout


def test_prediction_stage_forecasts_next_horizon_when_best_model_needs_no_future_exogenous(tmp_path):
    rng = np.random.default_rng(21)
    index_length = 1200
    values = 100 + 8 * np.sin(np.arange(index_length) * 2 * np.pi / 96) + rng.normal(scale=0.3, size=index_length)
    temperature = rng.normal(size=index_length)  # sin senal: la ablacion no la usa
    config, layout = _run_pipeline(tmp_path, "run-prediction-endogenous", values, temperature)

    result = run_prediction(config, layout)

    assert result.status == "SUCCESS"
    forecast = pd.read_csv(result.output / "series" / "M1_P" / "csv" / "pronostico_operativo.csv")
    assert len(forecast) == 96
    assert np.isfinite(forecast["prediccion"]).all()
    first = pd.Timestamp(forecast["timestamp"].iloc[0])
    assert first == pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=15 * index_length)
    summary = pd.read_csv(result.output / "csv" / "resumen_prediccion.csv")
    assert summary.loc[0, "estado"] == "PRONOSTICADA"


def test_prediction_stage_blocks_best_model_that_needs_future_exogenous(tmp_path):
    rng = np.random.default_rng(22)
    index_length = 1200
    temperature = 20 + 6 * np.sin(np.arange(index_length) * 2 * np.pi / 96) + rng.normal(scale=0.2, size=index_length)
    values = 100 + 9 * (temperature - 20) + rng.normal(scale=0.5, size=index_length)
    config, layout = _run_pipeline(tmp_path, "run-prediction-exogenous", values, temperature)

    result = run_prediction(config, layout)

    summary = pd.read_csv(result.output / "csv" / "resumen_prediccion.csv")
    assert summary.loc[0, "estado"] == "REQUIERE_EXOGENAS_FUTURAS"
    assert not (result.output / "series" / "M1_P" / "csv" / "pronostico_operativo.csv").exists()


def test_rolling_origin_coverage_is_published_for_every_configured_horizon(tmp_path):
    rng = np.random.default_rng(23)
    index_length = 1200
    values = 100 + 8 * np.sin(np.arange(index_length) * 2 * np.pi / 96) + rng.normal(scale=0.3, size=index_length)
    temperature = rng.normal(size=index_length)
    config, layout = _run_pipeline(tmp_path, "run-rolling-coverage", values, temperature)

    result = run_prediction(config, layout)

    table = pd.read_csv(result.output / "series" / "M1_P" / "csv" / "calibracion_rolling.csv")
    assert set(table["horizonte"]) == {24, 96}
    assert (table["n"] == 96).all()
    assert table["cobertura"].between(0, 1).all()
    assert (table["radio"] > 0).all()
