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

def _with_tolerance(config, layout, tolerance):
    """Rerun evaluation with an explicit coverage tolerance so the admitted path is exercised."""
    adjusted = replace(config, raw={**config.raw, "evaluation": {"coverage_tolerance": tolerance}})
    assert run_evaluation(adjusted, layout).status == "SUCCESS"
    return adjusted


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
    config = _with_tolerance(config, layout, 0.5)

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
    config = _with_tolerance(config, layout, 0.5)

    result = run_prediction(config, layout)

    table = pd.read_csv(result.output / "series" / "M1_P" / "csv" / "calibracion_rolling.csv")
    assert set(table["horizonte"]) == {24, 96}
    assert (table["n"] == 96).all()
    assert table["cobertura"].between(0, 1).all()
    assert (table["radio"] > 0).all()


def test_evaluation_ranking_uses_measured_calibration_for_models_without_future_exogenous(tmp_path):
    rng = np.random.default_rng(24)
    index_length = 1200
    values = 100 + 8 * np.sin(np.arange(index_length) * 2 * np.pi / 96) + rng.normal(scale=0.3, size=index_length)
    temperature = rng.normal(size=index_length)
    config, layout = _run_pipeline(tmp_path, "run-ranking-calibration", values, temperature)

    ranking = pd.read_csv(layout.stage("04_evaluacion") / "series" / "M1_P" / "csv" / "ranking.csv").set_index("modelo")
    assert ranking.loc["Ridge", "calibracion"] == "EVALUADA"
    assert np.isfinite(ranking.loc["Ridge", "cobertura_error"])


def test_default_gate_blocks_a_model_whose_calibration_deviates_and_reports_it(tmp_path):
    rng = np.random.default_rng(21)
    index_length = 1200
    values = 100 + 8 * np.sin(np.arange(index_length) * 2 * np.pi / 96) + rng.normal(scale=0.3, size=index_length)
    temperature = rng.normal(size=index_length)
    config, layout = _run_pipeline(tmp_path, "run-default-gate", values, temperature)

    ranking = pd.read_csv(layout.stage("04_evaluacion") / "series" / "M1_P" / "csv" / "ranking.csv").set_index("modelo")
    assert ranking.loc["Ridge", "calibracion"] == "EVALUADA"
    # Semilla fija: la desviacion de cobertura (~0.11) supera la tolerancia por defecto (0.05).
    assert ranking.loc["Ridge", "cobertura_error"] > 0.05
    assert ranking.loc["Ridge", "causa_exclusion"] == "calibracion_fuera_de_tolerancia"
    assert not bool(ranking.loc["Ridge", "admitido"])
    result = run_prediction(config, layout)
    summary = pd.read_csv(result.output / "csv" / "resumen_prediccion.csv")
    assert summary.loc[0, "estado"] == "SIN_MODELO_ADMITIDO"

def test_prediction_stage_publishes_client_scenarios_on_top_of_operational_forecast(tmp_path):
    rng = np.random.default_rng(25)
    index_length = 1200
    values = 100 + 8 * np.sin(np.arange(index_length) * 2 * np.pi / 96) + rng.normal(scale=0.3, size=index_length)
    temperature = rng.normal(size=index_length)
    config, layout = _run_pipeline(tmp_path, "run-client-scenarios", values, temperature)
    scenario_table = pd.DataFrame([{
        "medidor": "M1", "anio": 2025, "limite_inferior_kw": 0.0, "central_kw": 5.0,
        "limite_superior_kw": 8.0, "clientes": 2, "clientes_excluidos": 0, "calidad": "scenario_no_observed",
    }])
    scenario_path = layout.stage("02_limpieza", "datos") / "escenario_clientes_confirmados.parquet"
    scenario_table.to_parquet(scenario_path, index=False)
    manifest = ManifestStore(layout)
    stage_two = manifest.read()["stages"]["02_limpieza"]
    manifest.update_stage("02_limpieza", "SUCCESS", artifacts={
        **stage_two["artifacts"], "new_customer_scenario": str(scenario_path),
    })
    config = _with_tolerance(config, layout, 0.5)

    result = run_prediction(config, layout)

    table = pd.read_csv(result.output / "series" / "M1_P" / "csv" / "pronostico_escenarios.csv")
    assert set(table["escenario"]) == {"base", "inferior", "central", "superior"}
    assert len(table) == 4 * 96
    base = table.loc[table["escenario"] == "base", "prediccion_total"].to_numpy()
    central = table.loc[table["escenario"] == "central", "prediccion_total"].to_numpy()
    np.testing.assert_allclose(central - base, 5.0)
    summary = pd.read_csv(result.output / "csv" / "resumen_prediccion.csv")
    assert summary.loc[0, "escenarios"] == "ESCENARIOS_APLICADOS"


def test_conditional_coverage_for_exogenous_models_is_labelled_and_never_ranked(tmp_path):
    rng = np.random.default_rng(26)
    index_length = 1200
    temperature = 20 + 6 * np.sin(np.arange(index_length) * 2 * np.pi / 96) + rng.normal(scale=0.2, size=index_length)
    values = 100 + 9 * (temperature - 20) + rng.normal(scale=0.5, size=index_length)
    config, layout = _run_pipeline(tmp_path, "run-conditional-coverage", values, temperature)

    result = run_prediction(config, layout)

    summary = pd.read_csv(result.output / "csv" / "resumen_prediccion.csv")
    assert summary.loc[0, "estado"] == "REQUIERE_EXOGENAS_FUTURAS"
    table = pd.read_csv(result.output / "series" / "M1_P" / "csv" / "calibracion_condicionada_exogenas.csv")
    assert set(table["horizonte"]) == {24, 96}
    assert (table["n"] == 96).all()
    assert set(table["condicion"]) == {"EXOGENAS_REALIZADAS"}
    assert table["cobertura"].between(0, 1).all()
    ranking = pd.read_csv(layout.stage("04_evaluacion") / "series" / "M1_P" / "csv" / "ranking.csv").set_index("modelo")
    assert ranking.loc["Ridge", "calibracion"] == "NO_EVALUADA"


def _future_temperature(stamps, shift):
    phase = np.arange(len(stamps)) + 1200
    return pd.DataFrame({"temperatura_c": 20 + 6 * np.sin(phase * 2 * np.pi / 96) + shift}, index=stamps)


def test_exogenous_model_forecasts_when_future_exogenous_is_provided_and_responds_to_it(tmp_path):
    rng = np.random.default_rng(27)
    index_length = 1200
    temperature = 20 + 6 * np.sin(np.arange(index_length) * 2 * np.pi / 96) + rng.normal(scale=0.2, size=index_length)
    values = 100 + 9 * (temperature - 20) + rng.normal(scale=0.5, size=index_length)
    config, layout = _run_pipeline(tmp_path, "run-future-exog", values, temperature)
    config = _with_tolerance(config, layout, 0.5)
    stamps = pd.date_range(pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=15 * index_length), periods=96, freq="15min")


    warm = run_prediction(config, layout, future_exog=_future_temperature(stamps, 0.0))
    summary = pd.read_csv(warm.output / "csv" / "resumen_prediccion.csv")
    assert summary.loc[0, "estado"] == "PRONOSTICADA"
    assert summary.loc[0, "exogenas_futuras"] == "PROVISTAS"
    low = pd.read_csv(warm.output / "series" / "M1_P" / "csv" / "pronostico_operativo.csv")["prediccion"].to_numpy()
    assert len(low) == 96 and np.isfinite(low).all()

    hot = run_prediction(config, layout, future_exog=_future_temperature(stamps, 5.0))
    high = pd.read_csv(hot.output / "series" / "M1_P" / "csv" / "pronostico_operativo.csv")["prediccion"].to_numpy()
    assert (high - low).mean() > 10.0  # +5 C de temperatura futura -> mas demanda


def test_exogenous_model_reports_missing_future_variables_explicitly(tmp_path):
    rng = np.random.default_rng(28)
    index_length = 1200
    temperature = 20 + 6 * np.sin(np.arange(index_length) * 2 * np.pi / 96) + rng.normal(scale=0.2, size=index_length)
    values = 100 + 9 * (temperature - 20) + rng.normal(scale=0.5, size=index_length)
    config, layout = _run_pipeline(tmp_path, "run-missing-future-exog", values, temperature)
    config = _with_tolerance(config, layout, 0.5)
    stamps = pd.date_range(pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=15 * index_length), periods=96, freq="15min")
    wrong = pd.DataFrame({"otra": np.zeros(96)}, index=stamps)

    result = run_prediction(config, layout, future_exog=wrong)

    summary = pd.read_csv(result.output / "csv" / "resumen_prediccion.csv")
    assert summary.loc[0, "estado"] == "REQUIERE_EXOGENAS_FUTURAS"
    assert "NO_PROVISTAS" in summary.loc[0, "exogenas_futuras"]
    assert not (result.output / "series" / "M1_P" / "csv" / "pronostico_operativo.csv").exists()
