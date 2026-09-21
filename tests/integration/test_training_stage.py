from dataclasses import replace
import hashlib

import numpy as np
import pandas as pd

from prodma.artifacts import ManifestStore, RunLayout
from prodma.config import load_config
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
    exogenous = pd.DataFrame({"feriado": 0, "temperatura_c": 24.0}, index=index)
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
    assert (result.output / "csv/entrenamientos.csv").is_file()
    assert len(list((result.output / "modelos").glob("*.joblib"))) == 2
    assert len(list((result.output / "checkpoints").glob("*.joblib"))) == 2
    assert list((result.output / "times").glob("*.csv"))
    assert list((result.output / "traces").glob("*.csv"))
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
