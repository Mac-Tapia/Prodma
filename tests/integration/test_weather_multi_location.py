import pandas as pd
import pytest

from prodma.external.weather_locations import (
    BLOCKED_MISSING_SOURCE,
    BLOCKED_OUT_OF_RANGE,
    ASIGNADO,
    WeatherLocation,
    assign_meter_locations,
    haversine_km,
)


ICA = WeatherLocation("ica_centro", -14.0678, -75.7286, "nasa_power")
LIMA = WeatherLocation("lima_centro", -12.0464, -77.0428, "nasa_power")


def test_haversine_matches_known_distance_between_ica_and_lima():
    distance = haversine_km(ICA.lat, ICA.lon, LIMA.lat, LIMA.lon)

    assert 260 < distance < 270


def test_meters_without_coordinates_are_blocked_not_dropped():
    meters = pd.DataFrame({"identificador": ["L-6605", "L-SI112"], "lat": [None, -14.07], "lon": [None, -75.73]})

    result = assign_meter_locations(meters, [ICA, LIMA], max_km=50).set_index("identificador")

    assert result.loc["L-6605", "estado"] == BLOCKED_MISSING_SOURCE
    assert pd.isna(result.loc["L-6605", "ubicacion"])
    assert result.loc["L-SI112", "estado"] == ASIGNADO
    assert result.loc["L-SI112", "ubicacion"] == "ica_centro"


def test_meter_outside_range_of_every_location_is_explicit():
    meters = pd.DataFrame({"identificador": ["L-LIM"], "lat": [-12.05], "lon": [-77.04]})

    result = assign_meter_locations(meters, [ICA], max_km=50)

    assert result.loc[0, "estado"] == BLOCKED_OUT_OF_RANGE
    assert result.loc[0, "distancia_km"] > 200


def test_nearest_location_is_selected_with_its_distance():
    meters = pd.DataFrame({"identificador": ["L-LIM"], "lat": [-12.06], "lon": [-77.05]})

    result = assign_meter_locations(meters, [ICA, LIMA], max_km=50)

    assert result.loc[0, "ubicacion"] == "lima_centro"
    assert result.loc[0, "distancia_km"] < 5
