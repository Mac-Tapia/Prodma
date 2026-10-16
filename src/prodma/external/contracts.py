"""Contratos de fuentes externas: identidad, licencia y esquema obligatorios."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExternalSourceSpec:
    name: str
    schema_version: str
    licence: str
    source_url: str

    def __post_init__(self) -> None:
        for field_name in ("name", "schema_version", "licence", "source_url"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"la fuente requiere {field_name} no vacío")


@dataclass(frozen=True)
class RawSnapshot:
    path: Path
    sha256: str
    size: int
    metadata: dict = field(default_factory=dict)
