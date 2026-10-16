from dataclasses import replace
import hashlib

import numpy as np
import pandas as pd

from prodma.artifacts import ManifestStore, RunLayout
from prodma.config import load_config
from prodma.stages.evaluation import run_evaluation
from prodma.stages.training import run_training

ROOT = __import__("pathlib").Path(__file__).parents[2]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_two(layout, index, values, temperature):
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


def test_evaluation_stage_ranks_trained_series_from_out_of_sample_predictions(tmp_path):
    base = load_config(ROOT / "config/default.yaml")
    raw = dict(base.raw)
    raw["training"] = {"folds": 2, "min_train": 300, "n_trials": 1}
    config = replace(base, output_root=tmp_path, horizons=(24, 96), raw=raw)
    layout = RunLayout.create(tmp_path, "run-evaluation")
    index = pd.date_range("2025-01-01", periods=1200, freq="15min", name="Fecha")
    rng = np.random.default_rng(5)
    temperature = 20 + 6 * np.sin(np.arange(len(index)) * 2 * np.pi / 96) + rng.normal(scale=0.2, size=len(index))
    values = 100 + 9 * (temperature - 20) + rng.normal(scale=0.5, size=len(index))
    _stage_two(layout, index, values, temperature)

    training = run_training(config, layout, model_names=("SeasonalNaive", "Ridge"))
    assert training.status == "SUCCESS"

    result = run_evaluation(config, layout)

    assert result.status == "SUCCESS"
    ranking = pd.read_csv(result.output / "series" / "M1_P" / "csv" / "ranking.csv").set_index("modelo")
    assert set(ranking.index) == {"Ridge", "SeasonalNaive"}
    assert ranking.loc["Ridge", "estado"] == "TRAINED"
    assert ranking.loc["Ridge", "mae"] < ranking.loc["SeasonalNaive", "mae"]
    assert ranking.loc["Ridge", "posicion"] == 1
    assert set(ranking["calibracion"]) == {"NO_EVALUADA"}
    summary = pd.read_csv(result.output / "csv" / "resumen_evaluacion.csv")
    assert summary.loc[0, "mejor_modelo"] == "Ridge"
    assert summary.loc[0, "bloques_no_alineados"] == 0
    assert ManifestStore(layout).read()["stages"]["04_evaluacion"]["status"] == "SUCCESS"


def test_evaluation_stage_refuses_when_training_did_not_succeed(tmp_path):
    config = replace(load_config(ROOT / "config/default.yaml"), output_root=tmp_path)
    layout = RunLayout.create(tmp_path, "run-no-training")
    ManifestStore(layout).update_stage("03_entrenamiento", "FAILED", error="prueba")

    result = run_evaluation(config, layout)

    assert result.status == "FAILED"
    assert "SUCCESS" in result.warnings[0]


def test_evaluation_publishes_conformal_coverage_from_untouched_final_test(tmp_path):
    base = load_config(ROOT / "config/default.yaml")
    raw = dict(base.raw)
    raw["training"] = {"folds": 2, "min_train": 300, "n_trials": 1}
    config = replace(base, output_root=tmp_path, horizons=(24, 96), raw=raw)
    layout = RunLayout.create(tmp_path, "run-intervals")
    index = pd.date_range("2025-01-01", periods=1200, freq="15min", name="Fecha")
    rng = np.random.default_rng(5)
    temperature = 20 + 6 * np.sin(np.arange(len(index)) * 2 * np.pi / 96) + rng.normal(scale=0.2, size=len(index))
    values = 100 + 9 * (temperature - 20) + rng.normal(scale=0.5, size=len(index))
    _stage_two(layout, index, values, temperature)
    assert run_training(config, layout, model_names=("Ridge",)).status == "SUCCESS"

    result = run_evaluation(config, layout)

    series = result.output / "series" / "M1_P"
    final = pd.read_csv(layout.stage("03_entrenamiento") / "series" / "M1_P" / "csv" / "predicciones_test_final.csv")
    assert set(final["modelo"]) == {"Ridge"}
    assert len(final) == 192  # 2 x horizonte maximo (96)
    coverage = pd.read_csv(series / "csv" / "calibracion_intervalos.csv")
    assert {"modelo", "horizonte", "n", "cobertura", "radio", "alpha"}.issubset(coverage.columns)
    assert coverage.loc[coverage["horizonte"] == 96, "n"].iloc[0] == 96
    assert coverage["cobertura"].between(0, 1).all()
