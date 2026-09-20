import json

import numpy as np
import pandas as pd

from prodma.artifacts import RunLayout
from prodma.limpieza.temporal import build_clean_variable
from prodma.stages.cleaning import matrix_output_path, run_cleaning


def test_imputed_context_never_becomes_observed_target():
    idx = pd.date_range("2025-01-01", periods=8, freq="15min")
    raw = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, 8.0], index=idx)
    clean = build_clean_variable(raw, forward_limit=1, weekly_lag=672)
    assert clean.loc[idx[1], "contexto"] == 1.0
    assert not clean.loc[idx[1], "es_observado"]
    assert clean.loc[idx[1], "metodo_imputacion"] == "ffill"
    assert np.isnan(clean.loc[idx[1], "objetivo"])


def test_weekly_fill_is_marked_but_not_an_observed_target():
    idx = pd.date_range("2025-01-01", periods=674, freq="15min")
    raw = pd.Series(np.nan, index=idx)
    raw.iloc[0] = 12.0
    clean = build_clean_variable(raw, forward_limit=0, weekly_lag=672)
    assert clean.iloc[672]["contexto"] == 12.0
    assert clean.iloc[672]["metodo_imputacion"] == "semana_observada"
    assert np.isnan(clean.iloc[672]["objetivo"])


def test_matrix_path_belongs_to_cleaning_stage(tmp_path):
    layout = RunLayout.create(tmp_path, "r1")
    assert matrix_output_path(layout).parent == layout.stage("02_limpieza", "datos")


def test_cleaning_stage_publishes_matrix_and_hash(tmp_path):
    layout = RunLayout.create(tmp_path, "r1")
    idx = pd.date_range("2025-01-01", periods=3, freq="15min", name="Fecha")
    source = layout.stage("01_ingesta", "parquet") / "a.parquet"
    pd.DataFrame({"P": [1.0, np.nan, 3.0], "Q": [2.0, 2.0, 2.0]}, index=idx).to_parquet(source)
    catalog = {"sheets": [{"estado": "ok", "medidor": "M1", "path": str(source)}]}
    (layout.stage("01_ingesta") / "catalogo.json").write_text(json.dumps(catalog), encoding="utf-8")

    result = run_cleaning(None, layout)

    matrix = pd.read_parquet(result.artifacts["matrix"])
    assert "M1__P__objetivo" in matrix
    assert np.isnan(matrix.loc[idx[1], "M1__P__objetivo"])
    assert len(result.artifacts["sha256"]) == 64
    assert (layout.stage("02_limpieza", "estadistica_descriptiva") / "resumen_por_medidor.csv").is_file()
    assert (layout.stage("02_limpieza", "figuras") / "M1_P_crudo_limpio.png").stat().st_size > 0
