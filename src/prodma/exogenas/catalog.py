"""Construccion del catalogo exogeno desde configuracion versionada."""
from __future__ import annotations

import hashlib
from pathlib import Path

from prodma.config import PipelineConfig
from prodma.exogenas.contracts import ExogenousCatalog, ExogenousSource


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_exogenous_catalog(config: PipelineConfig) -> ExogenousCatalog:
    records = config.raw.get("exogenous", {}).get("sources", {})
    sources: list[ExogenousSource] = []
    for name, record in records.items():
        enabled = bool(record.get("enabled", True))
        path = (config.project_root / record["path"]).resolve()
        if enabled and not path.is_file():
            raise FileNotFoundError(f"Falta fuente exogena habilitada {name}: {path}")
        sources.append(ExogenousSource(
            name=str(name),
            path=path,
            frequency=str(record["frequency"]),
            value_columns=tuple(str(column) for column in record.get("value_columns", ())),
            availability=str(record["availability"]),
            future_policy=str(record["future_policy"]),
            source_url=str(record.get("source_url", "")),
            sha256=_sha256(path) if enabled else "",
            enabled=enabled,
            publication_lag_months=record.get("publication_lag_months"),
        ))
    return ExogenousCatalog(tuple(sources))
