import json
from pathlib import Path

import pandas as pd

from prodma.exogenas.asof import asof_values
from prodma.external.connectors.bcrp import parse_bcrp_periods, publication_calendar


FIXTURE = Path(__file__).parents[1] / "fixtures" / "external" / "bcrp" / "pbi_2023_q1.json"


def _vintages(download_date: str) -> pd.DataFrame:
    periods = parse_bcrp_periods(json.loads(FIXTURE.read_text(encoding="utf-8")))
    return publication_calendar(periods, lag_months=3, fecha_descarga=download_date)


def test_value_is_not_visible_before_its_declared_publication_date():
    vintages = _vintages("2026-10-15")
    targets = pd.date_range("2023-02-01", periods=2, freq="MS")

    values = asof_values(pd.DatetimeIndex(targets), vintages, origin=pd.Timestamp("2023-03-31"))

    assert values.isna().all()


def test_historical_pbi_has_nonzero_coverage_at_a_later_origin():
    """Regresión: asignar la fecha de descarga 2026 dejaba 2023 sin valores disponibles."""
    vintages = _vintages("2026-10-15")
    targets = pd.date_range("2023-04-01", "2023-06-01", freq="MS")

    values = asof_values(pd.DatetimeIndex(targets), vintages, origin=pd.Timestamp("2023-06-01"))

    assert values.notna().all()
    assert values.iloc[0] == values.iloc[0]


def test_asof_ignores_download_date_entirely():
    early = _vintages("2026-10-15")
    late = _vintages("2030-01-01")
    targets = pd.DatetimeIndex(pd.date_range("2023-06-01", periods=1, freq="MS"))

    a = asof_values(targets, early, origin=pd.Timestamp("2023-06-01"))
    b = asof_values(targets, late, origin=pd.Timestamp("2023-06-01"))

    pd.testing.assert_series_equal(a, b)
