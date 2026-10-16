"""Tipos inmutables para fuentes exogenas auditables."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExogenousSource:
    name: str
    path: Path
    frequency: str
    value_columns: tuple[str, ...]
    availability: str
    future_policy: str
    source_url: str
    sha256: str
    enabled: bool = True
    publication_lag_months: int | None = None


@dataclass(frozen=True)
class ExogenousCatalog:
    sources: tuple[ExogenousSource, ...]

    def enabled_sources(self) -> tuple[ExogenousSource, ...]:
        return tuple(source for source in self.sources if source.enabled)
