"""Configuracion tipada del pipeline modular."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    output_root: Path
    frequency: str
    timezone: str
    seed: int
    horizons: tuple[int, ...]
    raw: dict[str, Any]


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def load_config(path: Path | str) -> PipelineConfig:
    """Carga YAML y resuelve las rutas con respecto al archivo de configuracion."""
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    project_root = _resolve(base, payload.get("project_root", "."))
    output_root = _resolve(project_root, payload.get("output_root", "resultados"))
    horizons = tuple(int(value) for value in payload.get("horizons", (96, 672)))
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons debe contener enteros positivos")
    return PipelineConfig(
        project_root=project_root,
        output_root=output_root,
        frequency=str(payload.get("frequency", "15min")),
        timezone=str(payload.get("timezone", "America/Lima")),
        seed=int(payload.get("seed", 42)),
        horizons=horizons,
        raw=payload,
    )
