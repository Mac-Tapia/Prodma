"""Etapa 1: inventario por contenido e ingesta estandarizada de Excel."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from prodma.artifacts import RunLayout
from prodma.config import PipelineConfig
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


def _serializable_record(record: InputRecord) -> dict:
    payload = asdict(record)
    payload["path"] = str(record.path)
    return payload


def run_ingestion(config: PipelineConfig, layout: RunLayout) -> StageResult:
    inputs = config.raw.get("inputs", {})
    source_dir = config.project_root / inputs.get("historical_dir", "datahistorica")
    years = tuple(int(year) for year in inputs.get("years", (2023, 2024, 2025, 2026)))
    files = discover_historical_files(source_dir, years)
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
        },
    )
