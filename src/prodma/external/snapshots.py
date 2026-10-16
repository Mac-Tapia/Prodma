"""Snapshots crudos: escritura atómica, hash verificado y reproducción offline."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Callable

from prodma.external.contracts import ExternalSourceSpec, RawSnapshot
from prodma.external.http import Fetcher, fetch_with_retry

PAYLOAD_NAME = "respuesta.raw"
METADATA_NAME = "respuesta.raw.json"


class SnapshotIntegrityError(ValueError):
    pass


def store_snapshot(
    spec: ExternalSourceSpec,
    fetcher: Fetcher,
    url: str,
    root: Path | str,
    *,
    fecha_descarga: str,
    timeout: float,
    attempts_max: int,
    backoff_seconds: float = 1.0,
    sleep: Callable[[float], None] = lambda seconds: None,
) -> RawSnapshot:
    """Guarda `<root>/<fuente>/<fecha_descarga>/respuesta.raw` solo si el cuerpo está completo."""
    headers, chunks = fetch_with_retry(
        fetcher, url, timeout=timeout, attempts_max=attempts_max,
        backoff_seconds=backoff_seconds, sleep=sleep,
    )
    directory = Path(root) / spec.name / fecha_descarga
    directory.mkdir(parents=True, exist_ok=True)
    partial = directory / f".partial-{uuid.uuid4().hex}"
    digest = hashlib.sha256()
    size = 0
    try:
        with partial.open("wb") as stream:
            for chunk in chunks:
                stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        expected = headers.get("content-length")
        if expected is not None and int(expected) != size:
            raise SnapshotIntegrityError(
                f"tamaño recibido {size} no coincide con content-length {expected}"
            )
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    target = directory / PAYLOAD_NAME
    os.replace(partial, target)
    metadata = {
        "fuente": spec.name,
        "url": url,
        "fecha_descarga": fecha_descarga,
        "sha256": digest.hexdigest(),
        "tamano": size,
        "licencia": spec.licence,
        "schema_version": spec.schema_version,
        "fuente_url": spec.source_url,
        "cabeceras": dict(headers),
    }
    (directory / METADATA_NAME).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return RawSnapshot(target, digest.hexdigest(), size, metadata)


def replay_snapshot(path: Path | str, *, expected_sha256: str) -> bytes:
    """Lee un snapshot ya guardado y verifica su hash antes de devolver el contenido."""
    payload = Path(path).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise SnapshotIntegrityError(f"sha256 no coincide: esperado {expected_sha256}, obtenido {actual}")
    return payload
