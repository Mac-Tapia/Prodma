from pathlib import Path

import pandas as pd
import pytest

from prodma.external.connectors.inei import (
    BLOCKED_MISSING_SOURCE,
    classify_population,
    production_status,
    validate_official_table_identity,
)


POPULATION = Path(__file__).parents[2] / "variables_externas" / "poblacion" / "poblacion_ica_2023_2026.csv"


def _frame(rows):
    return pd.DataFrame(rows, columns=["fecha", "poblacion", "dato", "fuente"])


def test_only_official_observations_are_observed_covariates():
    frame = _frame([
        ("2023-06-30", 1041198, "interpolado_lineal", "x"),
        ("2024-06-30", 1062346, "observado_inei", "x"),
        ("2025-06-30", 1083494, "extrapolado_lineal", "x"),
    ])

    result = classify_population(frame)

    assert list(result["uso"]) == ["escenario", "covariable_observada", "escenario"]
    assert list(result["estado"]) == ["SUPUESTO_ESCENARIO", "OBSERVADO_OFICIAL", "SUPUESTO_ESCENARIO"]


def test_scenario_bounds_are_flagged_invalid_with_fewer_than_two_official_points():
    frame = _frame([
        ("2023-06-30", 1041198, "interpolado_lineal", "x"),
        ("2024-06-30", 1062346, "observado_inei", "x"),
    ])

    result = classify_population(frame)

    assert result["cota_valida"].eq(False).all()


def test_unknown_population_label_fails_closed():
    frame = _frame([("2024-06-30", 1, "estimado_a_ojo", "x")])

    with pytest.raises(ValueError, match="dato desconocido"):
        classify_population(frame)


@pytest.mark.skipif(not POPULATION.exists(), reason="serie de población no disponible")
def test_current_population_file_reconciles_every_row():
    frame = pd.read_csv(POPULATION)

    result = classify_population(frame)

    assert len(result) == len(frame)
    assert (result["uso"] == "covariable_observada").sum() == (frame["dato"] == "observado_inei").sum()


def _identity(**overrides):
    meta = {"ambito": "Ica", "tabla_id": "T-1", "unidad": "personas", "anio_base": 2017,
            "fecha_publicacion": "2025-01-01", "frecuencia": "anual"}
    meta.update(overrides)
    return meta


def test_official_table_identity_requires_every_field_and_ica_scope():
    validate_official_table_identity(_identity())
    with pytest.raises(ValueError, match="unidad"):
        validate_official_table_identity(_identity(unidad=None))
    with pytest.raises(ValueError, match="ámbito"):
        validate_official_table_identity(_identity(ambito="Lima"))


def test_regional_production_without_source_is_an_explicit_blocked_status(tmp_path):
    status = production_status(tmp_path / "produccion_regional.csv", meta=None)

    assert status["estado"] == BLOCKED_MISSING_SOURCE
    assert "produccion" in status["motivo"]
