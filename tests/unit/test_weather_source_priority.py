import pandas as pd

from prodma.external.weather_policy import choose_historical_source


def _frame(source: str, rows: dict[str, float | None]) -> pd.DataFrame:
    return pd.DataFrame({
        "fecha_hora": pd.to_datetime(list(rows)),
        "temperatura_c": list(rows.values()),
        "fuente": source,
        "calidad": ["faltante" if v is None else "x" for v in rows.values()],
    })


def test_measured_station_wins_over_reanalysis_when_present():
    senamhi = _frame("senamhi", {"2023-01-01 00:00": 22.0, "2023-01-01 01:00": None})
    open_meteo = _frame("open_meteo_archive", {"2023-01-01 00:00": 21.0, "2023-01-01 01:00": 20.5})

    merged = choose_historical_source({"senamhi": senamhi, "open_meteo_archive": open_meteo})

    row = merged.set_index("fecha_hora")
    assert row.loc[pd.Timestamp("2023-01-01 00:00"), "temperatura_c"] == 22.0
    assert row.loc[pd.Timestamp("2023-01-01 00:00"), "fuente_seleccionada"] == "senamhi"


def test_lower_priority_source_fills_only_missing_values_and_is_recorded():
    senamhi = _frame("senamhi", {"2023-01-01 01:00": None})
    open_meteo = _frame("open_meteo_archive", {"2023-01-01 01:00": 20.5})

    merged = choose_historical_source({"senamhi": senamhi, "open_meteo_archive": open_meteo})

    row = merged.set_index("fecha_hora").loc[pd.Timestamp("2023-01-01 01:00")]
    assert row["temperatura_c"] == 20.5
    assert row["fuente_seleccionada"] == "open_meteo_archive"
