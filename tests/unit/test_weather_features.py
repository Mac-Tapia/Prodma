import numpy as np
import pandas as pd
import pytest

from prodma.exogenas.weather import build_weather_features


ORIGIN = pd.Timestamp("2025-01-08 23:00")
INDEX = pd.date_range("2025-01-01", "2025-01-10 23:00", freq="1h")


def synthetic_weather_with_future_heatwave():
    hour = INDEX.hour.to_numpy()
    return pd.DataFrame({"fecha_hora": INDEX, "temperatura_c": 20 + 5 * np.sin(2 * np.pi * hour / 24)})


def test_climatology_for_validation_is_fit_only_before_origin():
    weather = synthetic_weather_with_future_heatwave()
    first = build_weather_features(weather, INDEX, origin=ORIGIN)
    weather.loc[weather.fecha_hora > ORIGIN, "temperatura_c"] += 100
    second = build_weather_features(weather, INDEX, origin=ORIGIN)
    pd.testing.assert_series_equal(first.climatology, second.climatology)


def test_future_observed_weather_is_not_exposed_in_operational_mode():
    result = build_weather_features(synthetic_weather_with_future_heatwave(), INDEX, origin=ORIGIN)
    assert result.observed.loc[result.observed.index > ORIGIN].isna().all()
    assert result.climatology.notna().all()
    assert result.available_for_forecast.all()


def test_weather_scenario_requires_complete_explicit_series():
    scenario = pd.Series(25.0, index=INDEX[:-1])
    with pytest.raises(ValueError, match="incompleto"):
        build_weather_features(synthetic_weather_with_future_heatwave(), INDEX, ORIGIN,
                               mode="weather_scenario", scenario=scenario)
