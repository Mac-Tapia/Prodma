import json
from pathlib import Path

import pandas as pd
import pytest

from prodma.exogenas.asof import asof_values
from prodma.external.connectors.bcrp import (
    bcrp_series_url,
    parse_bcrp_periods,
    publication_calendar,
)
from prodma.external.contracts import ExternalSourceSpec
from prodma.external.snapshots import replay_snapshot, store_snapshot


FIXTURE = Path(__file__).parents[1] / "fixtures" / "external" / "bcrp" / "pbi_2023_q1.json"
SPEC = ExternalSourceSpec(name="bcrp_pbi", schema_version="1",
                          licence="BCRPData términos de uso", source_url="https://estadisticas.bcrp.gob.pe/")


def _payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_parses_periods_units_and_base_year_from_the_api_payload():
    periods = parse_bcrp_periods(_payload())

    assert list(periods["periodo"]) == [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01"), pd.Timestamp("2023-03-01")]
    assert periods["valor"].iloc[0] == pytest.approx(175.538186117904)
    assert set(periods["base_anual"]) == {2007}


def test_no_data_marker_becomes_missing_value():
    payload = _payload()
    payload["periods"][1]["values"] = ["n.d."]

    periods = parse_bcrp_periods(payload)

    assert pd.isna(periods["valor"].iloc[1])


def test_unknown_month_name_is_rejected():
    payload = _payload()
    payload["periods"][0]["name"] = "Xxx.2023"

    with pytest.raises(ValueError, match="mes"):
        parse_bcrp_periods(payload)


def test_publication_lag_is_declared_and_never_the_download_date():
    periods = parse_bcrp_periods(_payload())

    calendar = publication_calendar(periods, lag_months=3, fecha_descarga="2026-10-15")

    first = calendar.set_index("periodo").loc[pd.Timestamp("2023-01-01")]
    assert first["fecha_disponible"] == pd.Timestamp("2023-04-01")
    assert (calendar["fecha_disponible"] < pd.Timestamp("2026-01-01")).all()
    assert calendar["politica_disponibilidad"].eq("rezago_conservador_3m").all()


def test_publication_requires_an_explicit_positive_lag():
    periods = parse_bcrp_periods(_payload())

    with pytest.raises(ValueError, match="rezago"):
        publication_calendar(periods, lag_months=0, fecha_descarga="2026-10-15")


def test_series_url_matches_the_documented_api_shape():
    assert bcrp_series_url("PN01773AM", "2023-01", "2023-03") == (
        "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/PN01773AM/json/2023-01/2023-03"
    )


def test_raw_response_is_snapshotted_and_replayed_identically(tmp_path):
    raw = FIXTURE.read_bytes()

    def fetcher(url, timeout):
        return {"content-length": str(len(raw))}, iter([raw])

    snap = store_snapshot(SPEC, fetcher, bcrp_series_url("PN01773AM", "2023-01", "2023-03"), tmp_path,
                          fecha_descarga="2026-10-15", timeout=10, attempts_max=1, sleep=lambda s: None)

    replayed = replay_snapshot(snap.path, expected_sha256=snap.sha256)
    assert json.loads(replayed.decode("utf-8")) == _payload()
