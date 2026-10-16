import numpy as np
import pandas as pd
import pytest

from prodma.prediccion.escenarios import apply_scenarios

STAMPS = pd.date_range("2026-03-01 00:00", periods=4, freq="15min")


def _base():
    return pd.DataFrame({"timestamp": STAMPS, "prediccion": [100.0, 110.0, 120.0, 130.0]})


def _scenarios():
    rows = []
    for stamp in STAMPS[:2]:
        rows.append({"escenario": "bajo", "timestamp": stamp, "incremento": 5.0})
        rows.append({"escenario": "alto", "timestamp": stamp, "incremento": 15.0})
    return pd.DataFrame(rows)


def test_each_scenario_is_base_plus_its_increments_and_base_is_kept():
    result = apply_scenarios(_base(), _scenarios()).set_index(["escenario", "timestamp"])
    assert result.loc[("base", STAMPS[0]), "prediccion_total"] == 100.0
    assert result.loc[("bajo", STAMPS[0]), "prediccion_total"] == 105.0
    assert result.loc[("alto", STAMPS[1]), "prediccion_total"] == 125.0
    # Pasos sin incremento conservan la base en todos los escenarios.
    assert result.loc[("alto", STAMPS[3]), "prediccion_total"] == 130.0


def test_scenario_timestamps_outside_the_horizon_are_rejected():
    outside = _scenarios().assign(timestamp=pd.Timestamp("2026-04-01"))
    with pytest.raises(ValueError):
        apply_scenarios(_base(), outside)


def test_missing_increment_is_rejected_not_filled():
    broken = _scenarios()
    broken.loc[0, "incremento"] = np.nan
    with pytest.raises(ValueError):
        apply_scenarios(_base(), broken)


def test_negative_total_is_rejected_because_demand_cannot_be_negative():
    too_low = _scenarios().assign(incremento=-500.0)
    with pytest.raises(ValueError):
        apply_scenarios(_base(), too_low)


def test_base_forecast_must_be_finite():
    base = _base()
    base.loc[1, "prediccion"] = np.inf
    with pytest.raises(ValueError):
        apply_scenarios(base, _scenarios())


from prodma.prediccion.escenarios import customer_increments


def _customer_table():
    return pd.DataFrame({
        "medidor": ["M1", "M1", "M2"],
        "anio": [2026, 2027, 2026],
        "limite_inferior_kw": [0.0, 0.0, 0.0],
        "central_kw": [10.0, 20.0, 5.0],
        "limite_superior_kw": [15.0, 30.0, 8.0],
    })


def test_increments_are_cumulative_from_january_first_of_each_year():
    stamps = pd.DatetimeIndex(["2026-12-31 23:45", "2027-01-01 00:00"])
    result = customer_increments(_customer_table(), "M1", stamps).set_index(["escenario", "timestamp"])
    assert result.loc[("central", stamps[0]), "incremento"] == 10.0
    assert result.loc[("central", stamps[1]), "incremento"] == 30.0
    assert result.loc[("superior", stamps[1]), "incremento"] == 45.0
    assert result.loc[("inferior", stamps[1]), "incremento"] == 0.0


def test_meter_without_scenario_is_rejected_not_zero_filled():
    with pytest.raises(ValueError):
        customer_increments(_customer_table(), "M9", pd.DatetimeIndex(["2026-01-01"]))


def test_increments_work_with_apply_scenarios_end_to_end():
    stamps = pd.date_range("2026-12-31 23:00", periods=4, freq="15min")
    base = pd.DataFrame({"timestamp": stamps, "prediccion": [100.0, 100.0, 100.0, 100.0]})
    result = apply_scenarios(base, customer_increments(_customer_table(), "M1", stamps))
    assert set(result["escenario"]) == {"base", "inferior", "central", "superior"}
    total = result.set_index(["escenario", "timestamp"])["prediccion_total"]
    assert total.loc[("central", stamps[-1])] == 110.0  # todos los pasos son de 2026: solo cuenta el incremento 2026
