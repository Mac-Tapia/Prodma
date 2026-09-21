"""Checkpoints atomicos con identidad completa del entrenamiento."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields
from pathlib import Path

import joblib


class CheckpointMismatch(ValueError):
    pass


@dataclass(frozen=True)
class CheckpointKey:
    matrix_sha256: str
    exogenous_sha256: str
    columns: tuple[str, ...]
    model_name: str
    library_version: str
    backend: str
    seed: int
    transformations: tuple[str, ...] = ()
    hyperparameters: str = "{}"
    train_end: str = ""
    source_version: str = "unknown"


def _path(directory: Path | str, key: CheckpointKey) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", key.model_name)
    return Path(directory) / f"{safe_name}.joblib"


def store_checkpoint(directory: Path | str, key: CheckpointKey, payload) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = _path(directory, key)
    temporary = target.with_suffix(target.suffix + ".tmp")
    joblib.dump({"key": key, "payload": payload}, temporary)
    with temporary.open("r+b") as stream:
        os.fsync(stream.fileno())
    temporary.replace(target)
    return target


def load_checkpoint(directory: Path | str, requested: CheckpointKey):
    target = _path(directory, requested)
    if not target.is_file():
        raise FileNotFoundError(f"checkpoint inexistente: {target}")
    envelope = joblib.load(target)
    stored = envelope.get("key")
    if not isinstance(stored, CheckpointKey):
        raise CheckpointMismatch("formato de checkpoint incompatible")
    differences = [field.name for field in fields(CheckpointKey)
                   if getattr(stored, field.name) != getattr(requested, field.name)]
    if differences:
        raise CheckpointMismatch(f"checkpoint incompatible: {', '.join(differences)}")
    return envelope["payload"]
