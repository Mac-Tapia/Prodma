import pandas as pd
import pytest

from prodma.exogenas.validation import validate_calendar, validate_frame


def test_duplicate_exogenous_timestamp_is_rejected():
    frame = pd.DataFrame({"fecha_hora": ["2026-01-01 00:15"] * 2, "temperatura_c": [20, 21]})
    with pytest.raises(ValueError, match="timestamp duplicado"):
        validate_frame(frame, expected_frequency="15min", timezone="America/Lima")


def test_unverified_holiday_is_reported_not_silently_accepted():
    report = validate_calendar(pd.DataFrame({"fecha": ["2026-07-28"], "feriado": [True], "norma": [""]}))
    assert report.warnings == ("feriado_sin_norma",)


def test_irregular_frequency_is_rejected():
    frame = pd.DataFrame({
        "fecha_hora": ["2026-01-01 00:00", "2026-01-01 00:15", "2026-01-01 00:45"],
        "temperatura_c": [20.0, 20.1, 20.3],
    })
    with pytest.raises(ValueError, match="frecuencia"):
        validate_frame(frame, expected_frequency="15min", timezone="America/Lima")


def test_non_finite_value_is_rejected():
    frame = pd.DataFrame({
        "fecha_hora": ["2026-01-01 00:00", "2026-01-01 00:15"],
        "temperatura_c": [20.0, float("inf")],
    })
    with pytest.raises(ValueError, match="no finito"):
        validate_frame(frame, expected_frequency="15min", timezone="America/Lima")
