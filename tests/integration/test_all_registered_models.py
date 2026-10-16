from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from prodma.artifacts import ManifestStore, RunLayout
from prodma.config import load_config
from prodma.modeling.registry import model_registry
from prodma.stages.training import run_training


ROOT = Path(__file__).parents[2]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_registered_models_have_a_row_with_terminal_state_per_series(tmp_path):
    base = load_config(ROOT / "config/default.yaml")
    raw = dict(base.raw)
    raw["training"] = {"folds": 2, "min_train": 300, "n_trials": 1, "prefilter_max_seconds": 240}
    config = replace(base, output_root=tmp_path, horizons=(24, 96), raw=raw)
    layout = RunLayout.create(tmp_path, "run-13-modelos")
    index = pd.date_range("2025-01-01", periods=1200, freq="15min", name="Fecha")
    values = 100 + 8 * np.sin(np.arange(len(index)) * 2 * np.pi / 96)
    matrix = pd.DataFrame({"M1_P": values}, index=index)
    quality = pd.DataFrame({
        "M1__P__observado": values, "M1__P__limpio": values, "M1__P__contexto": values,
        "M1__P__es_observado": True, "M1__P__es_imputado": False,
    }, index=index)
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

    result = run_training(config, layout)

    prefilter = pd.read_csv(result.output / "series" / "M1_P" / "csv" / "prefiltro.csv")
    assert set(prefilter["modelo"]) == set(model_registry())
    assert len(prefilter) == len(model_registry())
    assert prefilter["estado"].notna().all()
