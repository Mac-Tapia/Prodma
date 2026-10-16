import pandas as pd
import pytest

from prodma.customers.scenarios import (
    BLOCKED_MISSING_SOURCE,
    customer_scenarios,
    historical_connections_status,
    realized_connections,
)


def _scenario_input() -> pd.DataFrame:
    return pd.DataFrame({
        "cliente": ["A", "A", "B"],
        "medidor": ["M1", "M1", "M2"],
        "anio": [2026, 2027, 2026],
        "incremento_kw": [100.0, 50.0, 30.0],
        "demanda_kw": [200.0, 200.0, 60.0],
        "estado_capacidad": ["VALIDA", "VALIDA", "REVISAR_CAPACIDAD"],
        "estado_mapeo": ["MAPEADO", "MAPEADO", "SIN_MAPEO"],
    })


def test_realized_connections_without_dates_are_rejected_not_guessed():
    frame = pd.DataFrame({"cliente": ["A"], "medidor": ["M1"], "fecha_alta": [None]})

    with pytest.raises(ValueError, match="fecha_alta"):
        realized_connections(frame)


def test_realized_connections_carry_their_availability_date():
    frame = pd.DataFrame({"cliente": ["A"], "medidor": ["M1"], "fecha_alta": ["2025-03-01"]})

    result = realized_connections(frame)

    assert result.loc[0, "disponible_desde"] == pd.Timestamp("2025-03-01")


def test_absent_realized_history_is_an_explicit_blocked_status(tmp_path):
    status = historical_connections_status(tmp_path / "altas_realizadas.csv")

    assert status["estado"] == BLOCKED_MISSING_SOURCE
    assert "altas" in status["motivo"]


def test_scenarios_are_bounded_ordered_and_never_add_to_history():
    source = _scenario_input()
    before = source.copy()

    scenarios = customer_scenarios(source)

    assert (scenarios["limite_inferior_kw"] <= scenarios["central_kw"]).all()
    assert (scenarios["central_kw"] <= scenarios["limite_superior_kw"]).all()
    pd.testing.assert_frame_equal(source, before)


def test_capacity_review_and_unmapped_meters_do_not_enter_central_scenario():
    scenarios = customer_scenarios(_scenario_input()).set_index(["medidor", "anio"])

    assert pd.isna(scenarios.loc[("M2", 2026), "central_kw"])
    assert scenarios.loc[("M1", 2026), "central_kw"] == pytest.approx(100.0)
