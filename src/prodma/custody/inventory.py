"""Inventario inmutable de entradas: descubrimiento recursivo, hash y estados terminales.

Estados por archivo: SUPPORTED, QUARANTINED, DUPLICATE, FAILED, DOCUMENTATION y
MISSING_EXPECTED (declarada en configuración y ausente). DOCUMENTATION es una decisión
documentada: los Markdown informativos se registran con hash pero no bloquean la
aceptación ni alimentan el pipeline. Los marcadores `.gitkeep` se listan aparte.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

SUPPORTED = "SUPPORTED"
QUARANTINED = "QUARANTINED"
DUPLICATE = "DUPLICATE"
FAILED = "FAILED"
DOCUMENTATION = "DOCUMENTATION"
MISSING_EXPECTED = "MISSING_EXPECTED"

BLOCKING_STATES = frozenset({QUARANTINED, FAILED, MISSING_EXPECTED})
IGNORED_NAMES = frozenset({".gitkeep"})
DOCUMENTATION_SUFFIXES = frozenset({".md"})
DATA_SUFFIXES = frozenset({".xls", ".xlsx", ".csv"})
_XLSX_MAGIC = b"PK\x03\x04"
_XLS_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


class CustodyBlockedError(RuntimeError):
    """El inventario tiene estados que impiden iniciar la ingesta (fallo cerrado)."""


@dataclass(frozen=True)
class InventoryEntry:
    path: Path
    raiz: str
    size: int
    mtime_ns: int
    sha256: str
    estado: str
    motivo: str


@dataclass
class Inventory:
    entries: list[InventoryEntry]
    expected_missing: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    missing_roots: list[str] = field(default_factory=list)

    def by_state(self) -> Counter:
        counts = Counter(entry.estado for entry in self.entries)
        counts.update(MISSING_EXPECTED for _ in self.expected_missing)
        return counts

    @property
    def accepted(self) -> bool:
        return not any(entry.estado in BLOCKING_STATES for entry in self.entries) and not self.expected_missing


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _signature_problem(path: Path) -> str | None:
    """Motivo si la firma no coincide con la extensión; None si es coherente."""
    suffix = path.suffix.lower()
    with path.open("rb") as stream:
        head = stream.read(4096)
    if suffix == ".xlsx":
        return None if head.startswith(_XLSX_MAGIC) else "firma no coincide con .xlsx"
    if suffix == ".xls":
        return None if head.startswith(_XLS_MAGIC) else "firma no coincide con .xls"
    if suffix == ".csv":
        return "firma no coincide con texto CSV" if b"\x00" in head else None
    return None


def _classify(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix in DOCUMENTATION_SUFFIXES:
        return DOCUMENTATION, "documentación informativa; no alimenta el pipeline"
    if suffix not in DATA_SUFFIXES:
        return QUARANTINED, f"extensión no soportada: {suffix or '(sin extensión)'}"
    problem = _signature_problem(path)
    if problem:
        return QUARANTINED, problem
    return SUPPORTED, "formato reconocido"


def discover_inputs(roots: Iterable[Path | str], expected_paths: Iterable[Path | str] = ()) -> Inventory:
    entries: list[InventoryEntry] = []
    ignored: list[str] = []
    missing_roots: list[str] = []
    seen_hashes: dict[str, Path] = {}
    files: list[tuple[Path, Path]] = []
    for root in (Path(r) for r in roots):
        if not root.is_dir():
            missing_roots.append(str(root))
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path.name in IGNORED_NAMES:
                ignored.append(str(path))
                continue
            files.append((root, path))

    for root, path in sorted(files, key=lambda item: str(item[1])):
        try:
            stat = path.stat()
            digest, size = _sha256_and_size(path)
            estado, motivo = _classify(path)
        except OSError as exc:
            entries.append(InventoryEntry(path, str(root), 0, 0, "", FAILED, f"{type(exc).__name__}: {exc}"))
            continue
        if estado == SUPPORTED and digest in seen_hashes:
            estado, motivo = DUPLICATE, f"contenido idéntico a {seen_hashes[digest].name}"
        elif estado == SUPPORTED:
            seen_hashes[digest] = path
        entries.append(InventoryEntry(path, str(root), size, stat.st_mtime_ns, digest, estado, motivo))

    expected_missing = [str(Path(p)) for p in expected_paths if not Path(p).is_file()]
    return Inventory(entries, expected_missing, ignored, missing_roots)


def _summary(inventory: Inventory) -> dict:
    counts = {state: n for state, n in inventory.by_state().items() if n}
    return {
        "total_descubiertos": len(inventory.entries),
        "esperados_ausentes": len(inventory.expected_missing),
        "por_estado": counts,
        "aceptado": inventory.accepted,
        "ignorados": inventory.ignored,
        "raices_ausentes": inventory.missing_roots,
    }


def write_inventory(inventory: Inventory, out_dir: Path | str) -> dict:
    """Publica inventario y resumen, y los reabre para comprobar su contenido."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "inventario.jsonl"
    summary_path = out_dir / "input_summary.json"
    with jsonl.open("w", encoding="utf-8") as stream:
        for entry in inventory.entries:
            record = {
                "ruta": str(entry.path), "raiz": entry.raiz, "tamano": entry.size,
                "mtime_ns": entry.mtime_ns, "sha256": entry.sha256,
                "estado": entry.estado, "motivo": entry.motivo,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        for missing in inventory.expected_missing:
            stream.write(json.dumps({"ruta": missing, "estado": MISSING_EXPECTED}, ensure_ascii=False) + "\n")
    summary = _summary(inventory)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    reopened = json.loads(summary_path.read_text(encoding="utf-8"))
    lines = jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(inventory.entries) + len(inventory.expected_missing)
    return {
        "inventario": str(jsonl),
        "resumen": str(summary_path),
        "sha256_inventario": hashlib.sha256(jsonl.read_bytes()).hexdigest(),
        "summary": reopened,
    }
