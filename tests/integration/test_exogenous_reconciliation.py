import pandas as pd
import pytest

from prodma.exogenas.matrix import build_exogenous_matrices
from prodma.external.quality import (
    UNEXPLAINED_COLUMN,
    reconcile_exogenous_matrix,
)
from prodma.config import load_config
from prodma.exogenas.catalog import build_exogenous_catalog
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_column_without_declared_source_is_an_explicit_error():
    frame = pd.DataFrame({"feriado": [0, 1], "variable_misteriosa": [1.0, 2.0]}, index=pd.RangeIndex(2))

    with pytest.raises(ValueError, match=UNEXPLAINED_COLUMN):
        reconcile_exogenous_matrix(frame, enabled_sources={"calendario_pe"}, disabled_states={})


def test_all_null_column_is_reported_as_blocked_not_dropped():
    frame = pd.DataFrame({"pbi_last_known": [float("nan")] * 3}, index=pd.RangeIndex(3))

    report = reconcile_exogenous_matrix(frame, enabled_sources={"pbi_bcrp"}, disabled_states={})

    row = report.set_index("columna").loc["pbi_last_known"]
    assert row["estado"] == "BLOCKED_INVALID"
    assert row["cobertura_pct"] == 0.0


def test_real_matrix_reconciles_every_enabled_source_and_terminal_disabled_state():
    config = load_config(ROOT / "config/default.yaml")
    catalog = build_exogenous_catalog(config)
    index = pd.date_range("2024-01-01 00:00", periods=96, freq="15min", name="Fecha")
    matrices = build_exogenous_matrices(config, catalog, index, future_end=pd.Timestamp("2024-01-02"))
    disabled = {name: rec["estado"] for name, rec in config.raw["exogenous"]["sources"].items()
                if not rec.get("enabled", True)}

    report = reconcile_exogenous_matrix(matrices.historical, enabled_sources={s.name for s in catalog.enabled_sources()},
                                        disabled_states=disabled)

    assert set(matrices.historical.columns) <= set(report["columna"])
    assert {"senamhi_ica", "nasa_power", "clientes_confirmados", "produccion_regional"} <= set(report["fuente"])
    disabled_rows = report[report["fuente"].isin(disabled)]
    assert disabled_rows["estado"].notna().all()
