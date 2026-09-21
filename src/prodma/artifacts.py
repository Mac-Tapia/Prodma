"""Rutas, manifiestos atomicos y bloqueo de ejecuciones."""
from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE_CHILDREN: dict[str, tuple[str, ...]] = {
    "01_ingesta": ("parquet", "auditoria", "logs"),
    "02_limpieza": ("datos", "csv", "figuras", "estadistica_descriptiva", "auditoria", "logs"),
    "03_entrenamiento": ("modelos", "checkpoints", "figuras", "csv", "times", "traces", "logs"),
    "04_evaluacion": (
        "metricas", "inferencia_parametrica", "inferencia_no_parametrica",
        "diagnosticos", "ranking", "figuras", "logs",
    ),
    "05_prediccion": ("pronosticos", "intervalos", "escenarios", "figuras", "entregables", "logs"),
}
STAGES = tuple(STAGE_CHILDREN)


def sha256_file(path: Path | str) -> str:
    """Identidad binaria reproducible de un artefacto publicado."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @classmethod
    def create(cls, output_root: Path | str, run_id: str) -> "RunLayout":
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("run_id debe ser un nombre simple no vacio")
        root = Path(output_root) / run_id
        root.mkdir(parents=True, exist_ok=True)
        layout = cls(root)
        for stage, children in STAGE_CHILDREN.items():
            for child in children:
                layout.stage(stage, child).mkdir(parents=True, exist_ok=True)
        return layout

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def stage(self, name: str, child: str | None = None) -> Path:
        if name not in STAGE_CHILDREN:
            raise ValueError(f"Etapa desconocida: {name}")
        if child is not None and child not in STAGE_CHILDREN[name]:
            raise ValueError(f"Subcarpeta desconocida para {name}: {child}")
        return self.root / name / child if child else self.root / name


class ManifestStore:
    def __init__(self, layout: RunLayout):
        self.layout = layout

    def read(self) -> dict[str, Any]:
        if not self.layout.manifest_path.exists():
            return {"run_id": self.layout.root.name, "stages": {}}
        return json.loads(self.layout.manifest_path.read_text(encoding="utf-8"))

    def update_stage(self, stage: str, status: str, **data: Any) -> None:
        if stage not in STAGES:
            raise ValueError(f"Etapa desconocida: {stage}")
        payload = self.read()
        payload.setdefault("stages", {})[stage] = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        temporary = self.layout.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(self.layout.manifest_path)


@dataclass
class RunLock:
    path: Path
    _stream: Any

    @classmethod
    def acquire(cls, run_root: Path | str) -> "RunLock":
        root = Path(run_root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / ".run.lock"
        try:
            stream = path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise RuntimeError(f"ejecucion activa para {root.name}") from exc
        stream.write(json.dumps({"pid": os.getpid(), "created_at": datetime.now(timezone.utc).isoformat()}))
        stream.flush()
        return cls(path=path, _stream=stream)

    def release(self) -> None:
        if not self._stream.closed:
            self._stream.close()
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> "RunLock":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
