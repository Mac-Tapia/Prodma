"""Calendario peruano: normalización verificable y features derivadas deterministas.

Reglas:
- Un día aparece exactamente una vez y su `dia_semana` coincide con la fecha.
- `fin_semana` es determinista (sábado/domingo) y no requiere norma.
- Un feriado o un puente (no laborable sin fin de semana) requiere `norma`; sin ella
  queda `verificado=False`. Las features que usan esa etiqueta fallan cerrado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("fecha", "dia_semana", "feriado", "fin_semana", "no_laborable", "norma")
WEEKDAYS = ("lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo")
TIPO_FERIADO = "feriado"
TIPO_PUENTE = "puente"
TIPO_FIN_SEMANA = "fin_semana"
TIPO_LABORABLE = "laborable"


class CalendarError(ValueError):
    pass


class UnverifiedCalendarError(CalendarError):
    pass


def load_calendar_csv(path) -> pd.DataFrame:
    """Lee el CSV oficial conservando el texto; prueba UTF-8 y luego Latin-1 (nombres con ñ)."""
    try:
        return pd.read_csv(path, encoding="utf-8", dtype={"norma": "string", "nombre": "string"})
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", dtype={"norma": "string", "nombre": "string"})


def canonical_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise CalendarError(f"calendario sin columnas {missing}")
    table = frame.copy()
    table["fecha"] = pd.to_datetime(table["fecha"], format="%Y-%m-%d", errors="coerce")
    if table["fecha"].isna().any():
        raise CalendarError("fecha inválida en calendario")
    if table["fecha"].duplicated().any():
        duplicated = table.loc[table["fecha"].duplicated(), "fecha"].dt.strftime("%Y-%m-%d").tolist()
        raise CalendarError(f"fecha duplicada en calendario: {duplicated[:5]}")
    table = table.sort_values("fecha").reset_index(drop=True)

    expected_names = table["fecha"].dt.weekday.map(lambda d: WEEKDAYS[d])
    if not (table["dia_semana"].astype(str).str.strip().str.lower() == expected_names).all():
        raise CalendarError("dia_semana no coincide con la fecha")

    weekend = table["fecha"].dt.weekday >= 5
    feriado = table["feriado"].astype(str).str.lower().eq("true")
    fin_semana = table["fin_semana"].astype(str).str.lower().eq("true")
    if not (fin_semana == weekend).all():
        raise CalendarError("fin_semana no coincide con sábado/domingo")
    norma = table["norma"].fillna("").astype(str).str.strip()
    no_laborable = table["no_laborable"].astype(str).str.lower().eq("true")

    tipo = np.where(feriado, TIPO_FERIADO,
                    np.where(fin_semana, TIPO_FIN_SEMANA,
                             np.where(no_laborable, TIPO_PUENTE, TIPO_LABORABLE)))
    verificado = ~np.isin(tipo, [TIPO_FERIADO, TIPO_PUENTE]) | norma.ne("").to_numpy()
    return pd.DataFrame({
        "fecha": table["fecha"],
        "tipo": tipo,
        "feriado": feriado.to_numpy(),
        "no_laborable": no_laborable.to_numpy() | fin_semana.to_numpy() | feriado.to_numpy(),
        "nombre": table["nombre"].fillna("").astype(str).to_numpy() if "nombre" in table else "",
        "norma": norma.to_numpy(),
        "verificado": verificado,
    })


def _range(start: str, end_exclusive: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end_exclusive), freq="D", inclusive="left")


def verification_report(calendar: pd.DataFrame, start: str, end_exclusive: str) -> dict:
    required = _range(start, end_exclusive)
    present = pd.DatetimeIndex(calendar["fecha"])
    missing = required.difference(present)
    if len(missing):
        raise CalendarError(f"cobertura insuficiente: faltan {len(missing)} días desde {missing[0].date()}")
    window = calendar[calendar["fecha"].isin(required)]
    needs_norm = window["tipo"].isin([TIPO_FERIADO, TIPO_PUENTE])
    unverified = window.loc[needs_norm & ~window["verificado"], "fecha"]
    return {
        "inicio": str(required[0].date()),
        "fin_exclusivo": str(pd.Timestamp(end_exclusive).date()),
        "dias": len(required),
        "feriados": int((window["tipo"] == TIPO_FERIADO).sum()),
        "puentes": int((window["tipo"] == TIPO_PUENTE).sum()),
        "verificados": int((needs_norm & window["verificado"]).sum()),
        "sin_norma": [d.strftime("%Y-%m-%d") for d in unverified],
    }


def calendar_features(calendar: pd.DataFrame, start: str, end_exclusive: str,
                      require_verified: bool = True) -> pd.DataFrame:
    report = verification_report(calendar, start, end_exclusive)
    if require_verified and report["sin_norma"]:
        raise UnverifiedCalendarError(
            f"{len(report['sin_norma'])} feriados/puentes sin norma verificada: {report['sin_norma'][:5]}"
        )
    window = calendar[calendar["fecha"].isin(_range(start, end_exclusive))].set_index("fecha").sort_index()
    dow = window.index.weekday
    month = window.index.month
    holidays = window.index[window["feriado"].to_numpy()]
    if len(holidays):
        distance = np.array([np.min(np.abs((holidays - d).days)) for d in window.index])
    else:
        distance = np.full(len(window), 999)
    features = pd.DataFrame({
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "feriado": window["feriado"].astype(int).to_numpy(),
        "no_laborable": window["no_laborable"].astype(int).to_numpy(),
        "dist_feriado": distance,
    }, index=window.index)
    features.index.name = "fecha"
    return features
