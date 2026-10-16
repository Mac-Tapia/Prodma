import numpy as np
import pandas as pd
import pytest

from prodma.external.connectors.open_meteo import CALIDAD_PRONOSTICO, parse_forecast
from prodma.exogenas.futuras import future_temperature_from_forecast


def _payload(hours=48, start="2026-10-16T00:00"):
    times = pd.date_range(start, periods=hours, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    return {
        "timezone": "America/Lima",
        "hourly_units": {"temperature_2m": "°C"},
        "hourly": {"time": times, "temperature_2m": [20.0 + (i % 24) * 0.5 for i in range(hours)]},
    }


def test_hourly_forecast_is_interpolated_to_the_requested_quarter_hours():
    forecast = parse_forecast(_payload(), issue_time=pd.Timestamp("2026-10-15T23:00"))
    stamps = pd.date_range("2026-10-16 00:00", periods=8, freq="15min")
    frame = future_temperature_from_forecast(forecast, stamps)
    assert list(frame.columns) == ["temperatura_c"]
    assert frame.index.equals(stamps)
    # Entre 00:00 (20.0) y 01:00 (20.5): 00:15 debe ser 20.125.
    assert frame.iloc[1, 0] == pytest.approx(20.125)
    assert np.isfinite(frame.to_numpy()).all()


def test_stamps_outside_forecast_coverage_are_rejected_not_extrapolated():
    forecast = parse_forecast(_payload(hours=6), issue_time=pd.Timestamp("2026-10-15T23:00"))
    stamps = pd.date_range("2026-10-16 00:00", periods=40, freq="15min")
    with pytest.raises(ValueError, match="fuera de la cobertura"):
        future_temperature_from_forecast(forecast, stamps)


def test_forecast_with_missing_values_is_rejected():
    payload = _payload()
    payload["hourly"]["temperature_2m"][3] = None
    forecast = parse_forecast(payload, issue_time=pd.Timestamp("2026-10-15T23:00"))
    assert (forecast["calidad"] == "faltante").any()
    with pytest.raises(ValueError, match="faltantes"):
        future_temperature_from_forecast(forecast, pd.date_range("2026-10-16 00:00", periods=8, freq="15min"))


def test_only_forecast_quality_is_accepted_as_future_input():
    forecast = parse_forecast(_payload(), issue_time=pd.Timestamp("2026-10-15T23:00"))
    assert set(forecast["calidad"]) == {CALIDAD_PRONOSTICO}
    reanalysis = forecast.assign(calidad="reanalisis_no_es_medicion_directa")
    with pytest.raises(ValueError, match="pronostico operativo"):
        future_temperature_from_forecast(reanalysis, pd.date_range("2026-10-16 00:00", periods=4, freq="15min"))
