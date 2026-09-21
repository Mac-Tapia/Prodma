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
    assert list(matrix.columns) == ["M1_P", "M1_Q"]
    assert matrix.loc[idx[1], "M1_P"] == 1.0
    assert len(result.artifacts["sha256"]) == 64
    quality = pd.read_parquet(result.artifacts["quality"])
    assert np.isnan(quality.loc[idx[1], "M1__P__objetivo"])
    assert (layout.stage("02_limpieza", "estadistica_descriptiva") / "resumen_por_medidor.csv").is_file()
    assert (layout.stage("02_limpieza", "figuras") / "M1_P_crudo_limpio.png").stat().st_size > 0


def test_latest_year_defines_canonical_meter_catalog(tmp_path):
    layout = RunLayout.create(tmp_path, "r1")
    stale = layout.stage("02_limpieza", "figuras") / "RETIRADO_P_crudo_limpio.png"
    stale.write_bytes(b"stale")
    idx = pd.date_range("2025-01-01", periods=2, freq="15min", name="Fecha")
    old = layout.stage("01_ingesta", "parquet") / "old.parquet"
    established_old = layout.stage("01_ingesta", "parquet") / "established_old.parquet"
    established_new = layout.stage("01_ingesta", "parquet") / "established_new.parquet"
    current = layout.stage("01_ingesta", "parquet") / "current.parquet"
    pd.DataFrame({"P": [1.0, 2.0], "Q": [3.0, 4.0]}, index=idx).to_parquet(old)
    pd.DataFrame({"P": [1.0, 2.0], "Q": [3.0, 4.0], "S_medida": [4.0, 5.0]}, index=idx).to_parquet(established_old)
    idx_new = pd.date_range("2026-01-01", periods=2, freq="15min", name="Fecha")
    pd.DataFrame({"P": [2.0, 3.0], "Q": [4.0, 5.0], "S_medida": [5.0, 6.0]}, index=idx_new).to_parquet(established_new)
    pd.DataFrame({"P": [5.0, 6.0], "Q": [7.0, 8.0], "S_medida": [9.0, 10.0]}, index=idx).to_parquet(current)
    catalog = {"sheets": [
        {"estado": "ok", "anio": 2025, "medidor": "RETIRADO", "path": str(old)},
        {"estado": "ok", "anio": 2025, "medidor": "ACTUAL", "path": str(established_old)},
        {"estado": "ok", "anio": 2026, "medidor": "ACTUAL", "path": str(established_new)},
        {"estado": "ok", "anio": 2026, "medidor": "NUEVO", "path": str(current)},
    ]}
    (layout.stage("01_ingesta") / "catalogo.json").write_text(json.dumps(catalog), encoding="utf-8")

    result = run_cleaning(None, layout)
    matrix = pd.read_parquet(result.artifacts["matrix"])

    assert list(matrix.columns) == ["ACTUAL_P", "ACTUAL_Q", "ACTUAL_S"]
    assert not any("RETIRADO" in column for column in matrix.columns)
    recent = pd.read_parquet(result.artifacts["recent"])
    assert list(recent.columns) == ["NUEVO_P", "NUEVO_Q", "NUEVO_S"]
    assert recent.index.min() == idx.min()
    assert not stale.exists()
