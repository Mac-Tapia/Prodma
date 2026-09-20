"""Contratos compartidos por las etapas del pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str
    output: Path
    artifacts: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
