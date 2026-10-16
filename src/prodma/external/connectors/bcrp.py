"""Conector BCRP (API de series) con disponibilidad declarada, nunca inferida de la descarga.

Sin calendario de vintages publicado, la disponibilidad se aproxima con un rezago
conservador explícito: `fecha_disponible = periodo + lag_months`. La fecha de descarga
se conserva solo como metadato de procedencia.
"""
from __future__ import annotations

import re

import pandas as pd

BASE_URL = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"
MESES = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "sep": 9, "oct": 10, "nov": 11, "dic": 12,
}
NO_DATA = {"n.d.", "n.d", ""}
_PERIOD = re.compile(r"^([A-Za-z]{3})\.(\d{4})$")
_BASE = re.compile(r"ndice\s+(\d{4})\s*=\s*100")


def bcrp_series_url(serie: str, start: str, end: str) -> str:
    return f"{BASE_URL}/{serie}/json/{start}/{end}"


def _period(name: str) -> pd.Timestamp:
    match = _PERIOD.match(name.strip())
    if not match or match.group(1).lower() not in MESES:
        raise ValueError(f"mes no reconocido en periodo BCRP: {name!r}")
    return pd.Timestamp(year=int(match.group(2)), month=MESES[match.group(1).lower()], day=1)


def parse_bcrp_periods(payload: dict) -> pd.DataFrame:
    base_match = _BASE.search(payload.get("config", {}).get("title", ""))
    base_year = int(base_match.group(1)) if base_match else None
    rows = []
    for item in payload["periods"]:
        raw = (item.get("values") or [""])[0]
        text = str(raw).strip()
        if text.lower() in NO_DATA:
            value = float("nan")
        else:
            value = float(text)
        rows.append({
            "periodo": _period(item["name"]),
            "periodo_nombre": item["name"],
            "valor": value,
            "base_anual": base_year,
        })
    return pd.DataFrame(rows).sort_values("periodo").reset_index(drop=True)


def publication_calendar(periods: pd.DataFrame, *, lag_months: int, fecha_descarga: str) -> pd.DataFrame:
    if not isinstance(lag_months, int) or lag_months < 1:
        raise ValueError("debe declararse un rezago de publicación entero >= 1 mes")
    calendar = periods[["periodo", "valor"]].copy()
    calendar["fecha_disponible"] = calendar["periodo"] + pd.DateOffset(months=lag_months)
    calendar["politica_disponibilidad"] = f"rezago_conservador_{lag_months}m"
    calendar["fecha_descarga"] = fecha_descarga
    return calendar


def parse_period_name(name: str) -> pd.Timestamp:
    return _period(name)
