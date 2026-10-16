from pathlib import Path

import pandas as pd
import pytest

from prodma.external.connectors.calendar_pe import (
    CalendarError,
    UnverifiedCalendarError,
    calendar_features,
    canonical_calendar,
    load_calendar_csv,
    verification_report,
)


HEADER = "fecha,dia_semana,feriado,nombre,fin_semana,no_laborable,ambito,norma"
REAL = Path(__file__).parents[2] / "variables_externas" / "calendario" / "feriados_peru_2023_2036.csv"


def _csv(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "calendario.csv"
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def _days(start: str, end_exclusive: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end_exclusive, freq="D", inclusive="left")]


def _rows_for(start: str, end_exclusive: str, holidays: dict[str, str] | None = None,
              norma: dict[str, str] | None = None, bridges: set[str] = frozenset()) -> list[str]:
    holidays, norma = holidays or {}, norma or {}
    names = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
    rows = []
    for day in _days(start, end_exclusive):
        weekend = pd.Timestamp(day).weekday() >= 5
        feriado = day in holidays
        no_laborable = feriado or weekend or day in bridges
        rows.append(",".join([
            day, names[pd.Timestamp(day).weekday()], str(feriado), holidays.get(day, ""), str(weekend),
            str(no_laborable), "nacional", norma.get(day, ""),
        ]))
    return rows


def test_official_holiday_with_norm_is_verified(tmp_path):
    path = _csv(tmp_path, _rows_for("2023-12-30", "2024-01-03",
                                    holidays={"2024-01-01": "Año Nuevo"},
                                    norma={"2024-01-01": "Ley 1 (ejemplo)"}))

    calendar = canonical_calendar(load_calendar_csv(path))

    row = calendar.set_index("fecha").loc[pd.Timestamp("2024-01-01")]
    assert bool(row["feriado"]) is True
    assert bool(row["verificado"]) is True


def test_holiday_without_norm_is_reported_and_blocks_labels(tmp_path):
    path = _csv(tmp_path, _rows_for("2028-07-27", "2028-07-31", holidays={"2028-07-28": "Fiestas Patrias"}))
    calendar = canonical_calendar(load_calendar_csv(path))

    report = verification_report(calendar, "2028-07-27", "2028-07-31")
    assert report["sin_norma"] == ["2028-07-28"]
    with pytest.raises(UnverifiedCalendarError, match="2028-07-28"):
        calendar_features(calendar, "2028-07-27", "2028-07-31")


def test_bridge_day_is_non_working_only_with_legal_basis(tmp_path):
    rows = _rows_for("2028-06-01", "2028-06-06", bridges={"2028-06-05"})
    path = _csv(tmp_path, rows)
    calendar = canonical_calendar(load_calendar_csv(path))

    assert verification_report(calendar, "2028-06-01", "2028-06-06")["sin_norma"] == ["2028-06-05"]


def test_leap_day_is_required_and_unique(tmp_path):
    rows = _rows_for("2024-02-27", "2024-03-02")
    calendar = canonical_calendar(load_calendar_csv(_csv(tmp_path, rows)))
    assert pd.Timestamp("2024-02-29") in set(calendar["fecha"])

    without_leap = [r for r in rows if not r.startswith("2024-02-29")]
    with pytest.raises(CalendarError, match="cobertura"):
        verification_report(canonical_calendar(load_calendar_csv(_csv(tmp_path, without_leap))),
                            "2024-02-27", "2024-03-02")


def test_duplicate_date_is_rejected(tmp_path):
    rows = _rows_for("2025-01-01", "2025-01-04")
    path = _csv(tmp_path, rows + [rows[0]])

    with pytest.raises(CalendarError, match="duplicad"):
        canonical_calendar(load_calendar_csv(path))


def test_weekday_name_must_match_the_date(tmp_path):
    rows = _rows_for("2025-01-01", "2025-01-03")
    rows[0] = rows[0].replace("miercoles", "lunes")
    path = _csv(tmp_path, rows)

    with pytest.raises(CalendarError, match="dia_semana"):
        canonical_calendar(load_calendar_csv(path))


def test_horizon_coverage_fails_closed_when_days_are_missing(tmp_path):
    path = _csv(tmp_path, _rows_for("2027-12-01", "2027-12-31"))
    calendar = canonical_calendar(load_calendar_csv(path))

    with pytest.raises(CalendarError, match="cobertura"):
        verification_report(calendar, "2027-12-01", "2028-01-02")


def test_features_are_deterministic_and_cover_requested_range(tmp_path):
    path = _csv(tmp_path, _rows_for("2027-12-01", "2028-01-05", holidays={"2028-01-01": "Año Nuevo"},
                                    norma={"2028-01-01": "Norma verificada (ejemplo)"}))
    calendar = canonical_calendar(load_calendar_csv(path))

    features = calendar_features(calendar, "2027-12-01", "2028-01-05")

    assert len(features) == 35
    assert {"dow_sin", "dow_cos", "month_sin", "month_cos", "feriado", "dist_feriado"} <= set(features.columns)
    assert features.loc[pd.Timestamp("2028-01-01"), "feriado"] == 1
    assert features.loc[pd.Timestamp("2027-12-31"), "dist_feriado"] == 1


@pytest.mark.skipif(not REAL.exists(), reason="calendario real no disponible")
def test_real_calendar_covers_planning_horizon_and_reports_unverified_holidays():
    calendar = canonical_calendar(load_calendar_csv(REAL))

    report = verification_report(calendar, "2023-01-01", "2028-01-01")

    assert report["dias"] == 1826
    assert report["feriados"] > 0
    # Estado actual del calendario: ningún feriado tiene norma registrada.
    assert report["verificados"] == 0
    assert len(report["sin_norma"]) == report["feriados"]
