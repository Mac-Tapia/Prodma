import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prodma.external.connectors.nasa_power import parse_power_hourly
from prodma.external.weather_policy import classify_weather


FIXTURE = Path(__file__).parents[1] / "fixtures" / "external" / "nasa_power" / "ica_t2m_2023_01_01_02.json"


def _payload(values: dict, time_standard="LST", units="C", fill=-999.0) -> dict:
    return {
        "properties": {"parameter": {"T2M": values}},
        "header": {"time_standard": time_standard, "fill_value": fill, "sources": ["MERRA2"]},
        "parameters": {"T2M": {"units": units}},
        "geometry": {"coordinates": [-75.729, -14.068, 1319.63]},
    }


def test_lst_header_is_local_lima_time_and_fixture_reconciles():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    frame = parse_power_hourly(payload)

    assert len(frame) == 48
    assert frame["fecha_hora"].iloc[0] == pd.Timestamp("2023-01-01 00:00")
    assert frame["fecha_hora"].iloc[-1] == pd.Timestamp("2023-01-02 23:00")
    assert frame["temperatura_c"].iloc[0] == pytest.approx(14.98)
    assert set(frame["calidad"]) == {"reanalisis_no_es_medicion_directa"}
    assert frame["fuente"].iloc[0] == "nasa_power"


def test_utc_header_is_converted_to_lima():
    payload = _payload({"2023010105": 20.0}, time_standard="UTC")

    frame = parse_power_hourly(payload)

    assert frame["fecha_hora"].iloc[0] == pd.Timestamp("2023-01-01 00:00")


def test_fill_value_becomes_missing_and_is_labelled():
    payload = _payload({"2023010100": -999.0, "2023010101": 21.5})

    frame = parse_power_hourly(payload).set_index("fecha_hora")

    assert np.isnan(frame.loc[pd.Timestamp("2023-01-01 00:00"), "temperatura_c"])
    assert frame.loc[pd.Timestamp("2023-01-01 00:00"), "calidad"] == "faltante"
    assert frame.loc[pd.Timestamp("2023-01-01 01:00"), "temperatura_c"] == pytest.approx(21.5)


def test_non_celsius_units_are_rejected():
    with pytest.raises(ValueError, match="unidad"):
        parse_power_hourly(_payload({"2023010100": 300.0}, units="K"))


def _observed() -> pd.DataFrame:
    return parse_power_hourly(json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_observations_after_origin_are_never_used_as_observed():
    origin = pd.Timestamp("2023-01-01 12:00")

    result = classify_weather(_observed(), origin).set_index("fecha_hora")

    after = result.loc[result.index > origin]
    assert set(after["estado"]) == {"climatologia_planeamiento"}
    assert after["temperatura_observada_c"].isna().all()
    assert after["temperatura_climatologia_c"].notna().all()
    before = result.loc[result.index <= origin]
    assert set(before["estado"]) <= {"observado", "faltante"}
    assert before["temperatura_pronostico_c"].isna().all()


def test_forecast_issued_after_origin_is_rejected_as_leakage():
    origin = pd.Timestamp("2023-01-01 12:00")
    forecast = pd.DataFrame({"fecha_hora": [pd.Timestamp("2023-01-01 13:00")], "temperatura_c": [25.0]})

    with pytest.raises(ValueError, match="posterior al origen"):
        classify_weather(_observed(), origin, forecast=forecast,
                         forecast_issue_time=pd.Timestamp("2023-01-01 12:30"))


def test_operational_forecast_fills_only_its_own_column_and_state():
    origin = pd.Timestamp("2023-01-01 12:00")
    forecast = pd.DataFrame({"fecha_hora": [pd.Timestamp("2023-01-01 13:00")], "temperatura_c": [25.0]})

    result = classify_weather(_observed(), origin, forecast=forecast,
                              forecast_issue_time=pd.Timestamp("2023-01-01 11:00")).set_index("fecha_hora")

    row = result.loc[pd.Timestamp("2023-01-01 13:00")]
    assert row["estado"] == "pronostico_operativo"
    assert row["temperatura_pronostico_c"] == pytest.approx(25.0)
    assert np.isnan(row["temperatura_observada_c"])
