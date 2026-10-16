from dataclasses import replace
import hashlib

import joblib
import numpy as np
import pandas as pd

from prodma.artifacts import ManifestStore, RunLayout
from prodma.config import load_config
from prodma.stages.training import run_training

ROOT = __import__("pathlib").Path(__file__).parents[2]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_arima_is_trained_with_exogenous_features_and_its_predictions_are_exported(tmp_path):
    base = load_config(ROOT / "config/default.yaml")
    raw = dict(base.raw)
    raw["training"] = {"folds": 2, "min_train": 400, "n_trials": 1}
    config = replace(base, output_root=tmp_path, horizons=(24, 96), raw=raw)
    layout = RunLayout.create(tmp_path, "run-arima-exog")
    index = pd.date_range("2025-01-01", periods=12000, freq="15min", name="Fecha")
    rng = np.random.default_rng(8)
    temperature = 20 + 6 * np.sin(np.arange(len(index)) * 2 * np.pi / 96) + rng.normal(scale=0.2, size=len(index))
    values = 100 + 9 * (temperature - 20) + rng.normal(scale=0.5, size=len(index))
    matrix = pd.DataFrame({"M1_P": values}, index=index)
    quality = pd.DataFrame({
        "M1__P__observado": values, "M1__P__limpio": values, "M1__P__contexto": values,
        "M1__P__es_observado": True, "M1__P__es_imputado": False,
    }, index=index)
    exogenous = pd.DataFrame({"feriado": 0.0, "temperatura_c": temperature}, index=index)
    data_dir = layout.stage("02_limpieza", "datos")
    paths = {name: data_dir / filename for name, filename in {
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

    result = run_training(config, layout, model_names=("ARIMA",))

    series = result.output / "series" / "M1_P"
    trained = pd.read_csv(series / "csv" / "entrenamientos.csv")
    assert trained.loc[trained["modelo"] == "ARIMA", "estado"].iloc[0] == "ENTRENADO", trained.to_dict("records")
    predictions = pd.read_csv(series / "csv" / "predicciones_validacion.csv")
    assert set(predictions["modelo"]) == {"ARIMA"}
    assert predictions["prediccion"].notna().all()
    final = pd.read_csv(series / "csv" / "predicciones_test_final.csv")
    assert len(final) == 192
    exported = joblib.load(series / "modelos" / "ARIMA.joblib")
    assert exported.exog_columns_ is not None and "temperatura_c" in exported.exog_columns_
    assert set(exported.exog_columns_) <= {"feriado", "temperatura_c"}
