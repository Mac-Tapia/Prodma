from dataclasses import replace

import numpy as np
import pandas as pd

from prodma.artifacts import ManifestStore, RunLayout
from prodma.config import load_config
from prodma.stages.training import run_training
from pathlib import Path
import hashlib


ROOT = Path(__file__).parents[2]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_eligible_series_is_trained_not_only_the_first(tmp_path):
    """Regresión: el stage original ejecutaba `break` tras la primera serie elegible."""
    base = load_config(ROOT / "config/default.yaml")
    raw = dict(base.raw)
    raw["training"] = {"folds": 2, "min_train": 300, "n_trials": 1}
    config = replace(base, output_root=tmp_path, horizons=(24, 96), raw=raw)
    layout = RunLayout.create(tmp_path, "run-todas")
    index = pd.date_range("2025-01-01", periods=1200, freq="15min", name="Fecha")
    names = ("A_P", "B_P", "C_P")
    matrix = pd.DataFrame({name: np.sin(np.arange(1200.0) / (5 + i)) + i for i, name in enumerate(names)}, index=index)
    columns = {}
    for name in names:
        meter = name.split("_")[0]
        for field, values in {
            "observado": matrix[name], "limpio": matrix[name], "contexto": matrix[name],
            "es_observado": pd.Series(True, index=index), "es_imputado": pd.Series(False, index=index),
        }.items():
            columns[f"{meter}__P__{field}"] = values
    quality = pd.DataFrame(columns, index=index)
    exogenous = pd.DataFrame({"feriado": 0}, index=index)
    data_dir = layout.stage("02_limpieza", "datos")
    paths = {"matrix": data_dir / "matriz_limpia.parquet", "quality": data_dir / "matriz_calidad.parquet",
             "exogenous_historical": data_dir / "matriz_exogenas_historicas.parquet"}
    matrix.to_parquet(paths["matrix"])
    quality.to_parquet(paths["quality"])
    exogenous.to_parquet(paths["exogenous_historical"])
    ManifestStore(layout).update_stage("02_limpieza", "SUCCESS", artifacts={
        **{k: str(v) for k, v in paths.items()}, "sha256": _sha(paths["matrix"]),
        "exogenous_sha256": _sha(paths["exogenous_historical"]),
    })

    result = run_training(config, layout, model_names=("SeasonalNaive",))

    assert result.status == "SUCCESS", (result.warnings, result.artifacts)
    assert sorted(result.artifacts["entrenadas"]) == sorted(names)
    summary = pd.read_csv(result.output / "csv" / "resumen_series.csv")
    assert set(summary["serie"]) == set(names)
    for name in names:
        assert (result.output / "series" / name / "csv" / "entrenamientos.csv").is_file()
        assert (result.output / "series" / name / "csv" / "ablacion.csv").is_file()
