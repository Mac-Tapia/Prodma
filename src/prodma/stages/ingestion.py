"""Etapa 1: inventario por contenido e ingesta estandarizada de Excel."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from prodma.artifacts import RunLayout
from prodma.config import PipelineConfig
from prodma.custody.authorized import build_authorized_catalog, write_authorized_catalog
from prodma.custody.inventory import (
    BLOCKING_STATES,
    MISSING_EXPECTED,
    CustodyBlockedError,
    discover_inputs,
    write_inventory,
)
from prodma.ingesta.excel import ingestar_libro
from prodma.exogenas.new_customers import ingest_new_customers, write_new_customer_audit
from prodma.stages import StageResult


@dataclass(frozen=True)
class InputRecord:
    path: Path
    sha256: str
    size: int
    mtime_ns: int
    kind: str
    year: int | None


def _extract_year(path: Path) -> int | None:
    match = re.fullmatch(r"demanda(\d{4})\.xlsx?", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def fingerprint_input(path: Path | str) -> InputRecord:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    year = _extract_year(path)
    return InputRecord(path, digest.hexdigest(), stat.st_size, stat.st_mtime_ns,
                       "historico" if year is not None else "otro", year)


def discover_historical_files(directory: Path | str, years) -> list[Path]:
    directory = Path(directory)
    found: list[Path] = []
    for year in years:
        candidates = [directory / f"demanda{year}{suffix}" for suffix in (".xls", ".xlsx")]
        existing = [path for path in candidates if path.is_file()]
        if not existing:
            raise FileNotFoundError(f"No existe historico para {year} en {directory}")
        if len(existing) > 1:
            raise ValueError(f"Archivos ambiguos para {year}: {', '.join(path.name for path in existing)}")
        found.append(existing[0])
    return found


def discover_all_historical_files(directory: Path | str) -> list[Path]:
    """Todos los libros anuales presentes, sin lista fija de años (orden cronológico)."""
    directory = Path(directory)
    by_year: dict[int, list[Path]] = {}
    for path in directory.glob("demanda*.xls*"):
        year = _extract_year(path)
        if year is not None:
            by_year.setdefault(year, []).append(path)
    found: list[Path] = []
    for year in sorted(by_year):
        existing = sorted(by_year[year])
        if len(existing) > 1:
            raise ValueError(f"Archivos ambiguos para {year}: {', '.join(p.name for p in existing)}")
        found.append(existing[0])
    if not found:
        raise FileNotFoundError(f"No hay libros demanda<AAAA>.xls[x] en {directory}")
    return found


def _serializable_record(record: InputRecord) -> dict:
    payload = asdict(record)
    payload["path"] = str(record.path)
    return payload


def _publish_custody(config: PipelineConfig, layout: RunLayout, inputs: dict) -> dict:
    """Inventario y catálogo autorizado en 00_custodia; falla cerrada si algo bloquea."""
    custody_dir = layout.root / "00_custodia"
    roots = [
        config.project_root / inputs.get("historical_dir", "datahistorica"),
        config.project_root / inputs.get("new_customers_dir", "nuevosclientes"),
        config.project_root / inputs.get("external_dir", "variables_externas"),
    ]
    catalog_setting = inputs.get("authorized_catalog")
    expected = [config.project_root / catalog_setting] if catalog_setting else []
    inventory = discover_inputs(roots, expected_paths=expected)
    if not inventory.accepted:
        blocked = [
            f"{entry.path}: {entry.estado} ({entry.motivo})"
            for entry in inventory.entries if entry.estado in BLOCKING_STATES
        ] + [f"{missing}: {MISSING_EXPECTED}" for missing in inventory.expected_missing]
        raise CustodyBlockedError("Inventario no aceptado: " + "; ".join(blocked))
    published = write_inventory(inventory, custody_dir)
    result: dict = {"inventario": published["inventario"], "resumen": published["resumen"]}
    if catalog_setting:
        catalog = build_authorized_catalog(config.project_root / catalog_setting)
        authorized = write_authorized_catalog(catalog, custody_dir)
        result.update(catalogo=authorized["parquet"], catalogo_sha256=authorized["sha256"],
                      catalogo_resumen=authorized["summary"])
    return result


def run_ingestion(config: PipelineConfig, layout: RunLayout) -> StageResult:
    inputs = config.raw.get("inputs", {})
    source_dir = config.project_root / inputs.get("historical_dir", "datahistorica")
    custody = _publish_custody(config, layout, inputs)
    if inputs.get("years") is None:
        files = discover_all_historical_files(source_dir)
    else:
        files = discover_historical_files(source_dir, tuple(int(year) for year in inputs["years"]))
    fingerprints = [fingerprint_input(path) for path in files]
    parquet_dir = layout.stage("01_ingesta", "parquet")
    audit_dir = layout.stage("01_ingesta", "auditoria")
    rows: list[dict] = []
    for path in files:
        rows.extend(ingestar_libro(path, parquet_dir, destino_auditoria=audit_dir, imprimir=lambda *_: None))
    successful = [row for row in rows if row["estado"] == "ok"]
    new_customer_records: list[dict] = []
    new_customer_dir = config.project_root / inputs.get("new_customers_dir", "nuevosclientes")
    new_customer_files = sorted(new_customer_dir.glob("*nuevos*clientes*.xls*"))
    if len(new_customer_files) > 1:
        raise ValueError("Multiples archivos de nuevos clientes")
    if new_customer_files:
        customer_result = ingest_new_customers(new_customer_files[0])
        customer_table_path = parquet_dir / "nuevos_clientes.parquet"
        customer_result.table.to_parquet(customer_table_path, index=False)
        customer_audit_path = write_new_customer_audit(customer_result, layout)
        new_customer_records.append({
            "path": str(customer_result.source_path), "sha256": customer_result.sha256,
            "table": str(customer_table_path), "audit": str(customer_audit_path),
        })
    catalog = {
        "inputs": [_serializable_record(record) for record in fingerprints],
        "sheets": rows,
        "new_customers": new_customer_records,
    }
    catalog_path = layout.stage("01_ingesta") / "catalogo.json"
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    status = "SUCCESS" if successful and len(successful) == len(rows) else ("PARTIAL" if successful else "FAILED")
    return StageResult(
        stage="01_ingesta",
        status=status,
        output=layout.stage("01_ingesta"),
        artifacts={
            "catalogo": str(catalog_path),
            "parquet": [row["path"] for row in successful],
            "auditoria": [str(audit_dir / f"{Path(row['path']).stem}.json") for row in successful],
            "new_customers": new_customer_records,
            "custodia": custody,
        },
    )
