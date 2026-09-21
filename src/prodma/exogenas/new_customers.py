"""Ingesta conservadora de clientes confirmados como escenarios separados."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from prodma.artifacts import RunLayout


@dataclass(frozen=True)
class NewCustomerResult:
    table: pd.DataFrame
    source_path: Path
    sha256: str
    sheet: str


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_new_customers(path: Path | str, known_meters: set[str] | None = None) -> NewCustomerResult:
    source = Path(path)
    workbook = pd.ExcelFile(source)
    if "demandanueva." not in workbook.sheet_names:
        raise ValueError("falta hoja demandanueva.")
    raw = pd.read_excel(source, sheet_name="demandanueva.")
    normalized = {_normalize(column): column for column in raw.columns}
    aliases = {
        "cliente": "nombre del cliente nuevo",
        "demanda_kw": "demanda kw",
        "factor_potencia": "factor de potencia",
        "alimentador": "aliementador",
        "medidor": "medidor",
    }
    missing = [name for name, alias in aliases.items() if alias not in normalized]
    if missing:
        raise ValueError(f"faltan columnas de nuevos clientes: {missing}")
    base = pd.DataFrame({name: raw[normalized[alias]] for name, alias in aliases.items()})
    year_columns: dict[int, object] = {}
    for column in raw.columns:
        match = re.fullmatch(r"(20\d{2})(?:\.0)?", str(column))
        if match:
            year_columns[int(match.group(1))] = column
    if not year_columns:
        raise ValueError("no hay columnas anuales de incremento")
    for column in ("demanda_kw", "factor_potencia"):
        base[column] = pd.to_numeric(base[column], errors="coerce")
    if base[["demanda_kw", "factor_potencia"]].isna().any(axis=None):
        raise ValueError("demanda o factor de potencia no numerico")
    if (base["demanda_kw"] < 0).any() or (~base["factor_potencia"].between(0, 1, inclusive="right")).any():
        raise ValueError("demanda o factor de potencia fuera de rango")
    parts = []
    for year, source_column in sorted(year_columns.items()):
        part = base.copy()
        part["anio"] = year
        part["incremento_kw"] = pd.to_numeric(raw[source_column], errors="coerce").fillna(0.0)
        parts.append(part)
    table = pd.concat(parts, ignore_index=True)
    if (table["incremento_kw"] < 0).any():
        raise ValueError("incremento anual negativo")
    totals = table.groupby("cliente")["incremento_kw"].sum()
    capacities = table.groupby("cliente")["demanda_kw"].first()
    offenders = set(totals.index[totals > capacities + 1e-9])
    table["estado_capacidad"] = table["cliente"].map(
        lambda customer: "REVISAR_CAPACIDAD" if customer in offenders else "VALIDA"
    )
    table["fecha_entrada"] = pd.NaT
    if known_meters is None:
        table["estado_mapeo"] = "NO_VALIDADO"
    else:
        table["estado_mapeo"] = table["medidor"].astype(str).map(
            lambda meter: "MAPEADO" if meter in known_meters else "SIN_MAPEO"
        )
    return NewCustomerResult(table, source.resolve(), _sha256(source), "demandanueva.")


def write_new_customer_audit(result: NewCustomerResult, layout: RunLayout) -> Path:
    path = layout.stage("01_ingesta", "auditoria") / "nuevos_clientes.json"
    payload = {
        "source": str(result.source_path), "sha256": result.sha256, "sheet": result.sheet,
        "rows": len(result.table), "customers": int(result.table["cliente"].nunique()),
        "years": sorted(int(year) for year in result.table["anio"].unique()),
        "customers_capacity_review": int(result.table.loc[
            result.table["estado_capacidad"] == "REVISAR_CAPACIDAD", "cliente"
        ].nunique()),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def publish_new_customer_scenario(result: NewCustomerResult, layout: RunLayout) -> dict[str, str]:
    audit = write_new_customer_audit(result, layout)
    table = result.table.copy()
    valid = table["estado_capacidad"].eq("VALIDA")
    table["central_component"] = table["incremento_kw"].where(valid)
    table["upper_component"] = table["demanda_kw"].where(valid & table["incremento_kw"].gt(0))
    scenario = table.groupby(["medidor", "anio", "estado_mapeo", "estado_capacidad"], as_index=False).agg(
        limite_inferior_kw=("incremento_kw", lambda values: 0.0),
        central_kw=("central_component", lambda values: values.sum(min_count=1)),
        limite_superior_kw=("upper_component", lambda values: values.sum(min_count=1)),
        clientes=("cliente", "nunique"),
    )
    scenario["calidad"] = "scenario_no_observed"
    scenario["fecha_entrada"] = pd.NaT
    target = layout.stage("02_limpieza", "datos") / "escenario_clientes_confirmados.parquet"
    scenario.to_parquet(target, index=False)
    return {"new_customer_scenario": str(target), "new_customer_audit": str(audit),
            "new_customer_sha256": result.sha256}
