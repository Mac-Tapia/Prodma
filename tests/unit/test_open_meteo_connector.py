import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prodma.external.connectors.open_meteo import (
    archive_url,
    forecast_url,
    parse_archive,
    parse_forecast,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "external" / "open_meteo" / "archive_ica_2023_01_01_02.json"


def _payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_archive_is_parsed_as_local_lima_reanalysis_and_reconciles_with_fixture():
    frame = parse_archive(_payload())

    assert len(frame) == 48
    assert frame["fecha_hora"].iloc[0] == pd.Timestamp("2023-01-01 00:00")
    assert frame["temperatura_c"].iloc[0] == pytest.approx(21.0)
    assert set(frame["calidad"]) == {"reanalisis_no_es_medicion_directa"}
    assert set(frame["fuente"]) == {"open_meteo_archive"}


def test_timezone_other_than_lima_is_rejected_fail_closed():
    payload = _payload()
    payload["timezone"] = "UTC"

    with pytest.raises(ValueError, match="zona horaria"):
        parse_archive(payload)


def test_non_celsius_units_are_rejected():
    payload = _payload()
    payload["hourly_units"]["temperature_2m"] = "K"

    with pytest.raises(ValueError, match="unidad"):
        parse_archive(payload)


def test_missing_values_are_nan_not_zero():
    payload = _payload()
    payload["hourly"]["temperature_2m"][5] = None

    frame = parse_archive(payload)

    assert np.isnan(frame["temperatura_c"].iloc[5])
    assert frame["calidad"].iloc[5] == "faltante"


def test_forecast_keeps_its_issue_time_and_is_marked_as_operational():
    payload = _payload()
    issued = pd.Timestamp("2023-01-01 06:00")

    frame = parse_forecast(payload, issue_time=issued)

    assert set(frame["calidad"]) == {"pronostico_operativo"}
    assert (frame["emitido_en"] == issued).all()


def test_urls_match_documented_open_meteo_shape():
    assert archive_url(-14.0678, -75.7286, "2023-01-01", "2023-01-02") == (
        "https://archive-api.open-meteo.com/v1/archive?latitude=-14.0678&longitude=-75.7286"
        "&start_date=2023-01-01&end_date=2023-01-02&hourly=temperature_2m&timezone=America%2FLima"
    )
    assert "forecast?" in forecast_url(-14.0678, -75.7286, past_days=1, forecast_days=2)
