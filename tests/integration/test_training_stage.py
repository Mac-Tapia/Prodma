from dataclasses import replace
import hashlib

import numpy as np
import pandas as pd
import joblib

from prodma.artifacts import ManifestStore, RunLayout
from prodma.config import load_config
from prodma.modeling.features import join_exogenous_features, make_features
from prodma.stages.training import run_training


ROOT = __import__("pathlib").Path(__file__).parents[2]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_training_stage_writes_isolated_recoverable_artifacts(tmp_path):
    base = load_config(ROOT / "config/default.yaml")
    raw = dict(base.raw)
    raw["training"] = {"folds": 2, "min_train": 300, "n_trials": 1, "max_series": 1}
    config = replace(base, output_root=tmp_path, horizons=(24, 96), raw=raw)
    layout = RunLayout.create(tmp_path, "run-training")
    index = pd.date_range("2025-01-01", periods=1200, freq="15min", name="Fecha")
    values = 100 + 8 * np.sin(np.arange(len(index)) * 2 * np.pi / 96)
    matrix = pd.DataFrame({"M1_P": values}, index=index)
    quality = pd.DataFrame({
        "M1__P__observado": values,
        "M1__P__limpio": values,
        "M1__P__contexto": values,
        "M1__P__es_observado": True,
        "M1__P__es_imputado": False,
    }, index=index)
    exogenous = pd.DataFrame({"feriado": 0, "temperatura_c": 24.0, "sin_vintage": np.nan}, index=index)
    data_dir = layout.stage("02_limpieza", "datos")
    matrix_path = data_dir / "matriz_limpia.parquet"
    quality_path = data_dir / "matriz_calidad.parquet"
    exogenous_path = data_dir / "matriz_exogenas_historicas.parquet"
    matrix.to_parquet(matrix_path)
    quality.to_parquet(quality_path)
    exogenous.to_parquet(exogenous_path)
    ManifestStore(layout).update_stage("02_limpieza", "SUCCESS", artifacts={
        "matrix": str(matrix_path), "quality": str(quality_path),
        "exogenous_historical": str(exogenous_path),
        "sha256": _sha(matrix_path), "exogenous_sha256": _sha(exogenous_path),
    })

    result = run_training(config, layout, model_names=("SeasonalNaive", "Ridge"))

    assert result.status == "SUCCESS"
    series = result.output / "series" / "M1_P"
    assert (series / "csv/entrenamientos.csv").is_file()
    assert len(list((series / "modelos").glob("*.joblib"))) == 2
    assert len(list((series / "checkpoints").glob("*.joblib"))) == 2
    envelope = joblib.load(series / "checkpoints" / "Ridge.joblib")
    assert "sin_vintage" not in envelope["key"].columns
    clean = pd.DataFrame({key.split("__")[-1]: value for key, value in quality.items()}, index=index)
    features = join_exogenous_features(
        make_features(clean), exogenous.drop(columns=["feriado"])
    ).dropna(axis=1, how="all")
    features["feriado"] = exogenous["feriado"]
    features = features.loc[:, envelope["key"].columns]
    exported = joblib.load(series / "modelos" / "Ridge.joblib")
    np.testing.assert_allclose(
        envelope["payload"]["model"].predict(features.iloc[-10:]),
        exported.predict(features.iloc[-10:]),
    )
    assert list((series / "times").glob("*.csv"))
    assert list((series / "traces").glob("*.csv"))
    assert ManifestStore(layout).read()["stages"]["03_entrenamiento"]["status"] == "SUCCESS"

    restarted = run_training(config, layout, model_names=("SeasonalNaive", "Ridge"))
    assert restarted.status == "SUCCESS"
    assert restarted.artifacts["reused"] is True


def test_training_stage_rejects_changed_stage_two_artifact(tmp_path):
    config = replace(load_config(ROOT / "config/default.yaml"), output_root=tmp_path)
    layout = RunLayout.create(tmp_path, "run-tampered")
    path = layout.stage("02_limpieza", "datos") / "matriz_limpia.parquet"
    pd.DataFrame({"M1_P": [1.0]}, index=pd.date_range("2025-01-01", periods=1, freq="15min")).to_parquet(path)
    ManifestStore(layout).update_stage("02_limpieza", "SUCCESS", artifacts={"matrix": str(path), "sha256": "0" * 64})

    result = run_training(config, layout, model_names=("Ridge",))

    assert result.status == "FAILED"
    assert "hash" in result.warnings[0].lower()


def test_training_stage_skips_target_without_contiguous_observed_test(tmp_path):
    base = load_config(ROOT / "config/default.yaml")
    raw = dict(base.raw)
    raw["training"] = {"folds": 1, "min_train": 150, "n_trials": 1}
    config = replace(base, output_root=tmp_path, horizons=(48,), raw=raw)
    layout = RunLayout.create(tmp_path, "run-select")
    index = pd.date_range("2025-01-01", periods=400, freq="15min", name="Fecha")
    matrix = pd.DataFrame({"A_P": np.arange(400.0), "B_P": np.arange(400.0)}, index=index)
    columns = {}
    for meter in ("A", "B"):
        observed = pd.Series(True, index=index)
        if meter == "A":
            observed.iloc[-10] = False
        for field, values in {
            "observado": matrix[f"{meter}_P"].where(observed), "limpio": matrix[f"{meter}_P"],
            "contexto": matrix[f"{meter}_P"], "es_observado": observed, "es_imputado": ~observed,
        }.items():
            columns[f"{meter}__P__{field}"] = values
    quality = pd.DataFrame(columns, index=index)
    exogenous = pd.DataFrame({"feriado": 0}, index=index)
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

    result = run_training(config, layout, model_names=("Ridge",))

    assert result.status == "SUCCESS"
    assert result.artifacts["entrenadas"] == ["B_P"]
    assert result.artifacts["no_elegibles"] == ["A_P"]
    assert (result.output / "csv" / "resumen_series.csv").is_file()


def test_training_stage_marks_forecaster_incompatible_when_exogenous_selected(tmp_path):
    base = load_config(ROOT / "config/default.yaml")
    raw = dict(base.raw)
    raw["training"] = {"folds": 2, "min_train": 300, "n_trials": 1}
    config = replace(base, output_root=tmp_path, horizons=(24, 96), raw=raw)
    layout = RunLayout.create(tmp_path, "run-incompatible")
    index = pd.date_range("2025-01-01", periods=1200, freq="15min", name="Fecha")
    rng = np.random.default_rng(5)
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

    result = run_training(config, layout, model_names=("SeasonalNaive", "Ridge"))

    assert result.status == "SUCCESS"
    series = result.output / "series" / "M1_P"
    compatibility = pd.read_csv(series / "csv" / "compatibilidad_exogenas.csv").set_index("modelo")
    assert compatibility.loc["SeasonalNaive", "estado"] == "INCOMPATIBLE_EXOG"
    assert compatibility.loc["Ridge", "estado"] == "COMPATIBLE"
    assert compatibility.loc["Ridge", "exogenas"] > 0
    trained = pd.read_csv(series / "csv" / "entrenamientos.csv")
    assert set(trained["modelo"]) == {"Ridge"}
    assert not (series / "modelos" / "SeasonalNaive.joblib").exists()
