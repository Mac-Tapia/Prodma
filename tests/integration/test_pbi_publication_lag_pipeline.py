from pathlib import Path

import pandas as pd
import pytest

from prodma.artifacts import RunLayout
from prodma.config import load_config
from prodma.exogenas.catalog import build_exogenous_catalog
from prodma.exogenas.matrix import _pbi_vintages, build_exogenous_matrices, publish_exogenous_matrices


ROOT = Path(__file__).parents[2]
PBI_CSV = ROOT / "variables_externas" / "economia" / "pbi_mensual_bcrp_2023_2026.csv"


def test_pbi_availability_uses_declared_lag_not_download_date():
    early = _pbi_vintages(PBI_CSV, lag_months=3)
    assert (early["fecha_disponible"] - early["periodo"]).dt.days.between(89, 93).all()
    # El último periodo es jun 2026: con rezago de 3 meses queda disponible en sep 2026.`n    assert early["fecha_disponible"].max() == pd.Timestamp("2026-09-01")


def test_pbi_without_declared_lag_fails_closed():
    with pytest.raises(ValueError, match="rezago"):
        _pbi_vintages(PBI_CSV, lag_months=None)


def test_pbi_is_historically_available_in_the_pipeline_matrix():
    """Regresión: con la fecha de descarga 2026 el PBI histórico quedaba 100 % nulo."""
    config = load_config(ROOT / "config/default.yaml")
    index = pd.date_range("2024-01-01 00:00", periods=96, freq="15min", name="Fecha")

    matrices = build_exogenous_matrices(config, build_exogenous_catalog(config), index,
                                        future_end=pd.Timestamp("2024-01-02 00:00"))

    assert matrices.historical["pbi_last_known"].notna().all()


def test_publication_writes_collinearity_report(tmp_path):
    config = load_config(ROOT / "config/default.yaml")
    index = pd.date_range("2024-01-01 00:00", periods=96 * 3, freq="15min", name="Fecha")
    matrices = build_exogenous_matrices(config, build_exogenous_catalog(config), index,
                                        future_end=pd.Timestamp("2024-01-04 00:00"))
    layout = RunLayout.create(tmp_path, "run")

    publish_exogenous_matrices(matrices, layout, demand=None)

    report = pd.read_csv(layout.stage("02_limpieza", "estadistica_descriptiva") / "colinealidad_exogenas.csv")
    assert {"variable_a", "variable_b", "correlacion", "estado"} <= set(report.columns)


def test_every_disabled_source_has_a_terminal_status_in_config():
    config = load_config(ROOT / "config/default.yaml")
    terminal = {"USED", "REJECTED_WITH_EVIDENCE", "SCENARIO_ONLY", "BLOCKED_MISSING_SOURCE",
                "BLOCKED_INVALID", "SUPERSEDED"}
    sources = config.raw["exogenous"]["sources"]
    undecided = [name for name, record in sources.items()
                 if not record.get("enabled", True) and record.get("estado") not in terminal]
    assert undecided == []
