"""Resolución determinista de columnas en hojas Excel históricas."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Mapping, Sequence


def normalize_header(value: object) -> str:
    """Normaliza un encabezado conservando el criterio usado por el pipeline."""
    ascii_value = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
    return re.sub(r"[^a-z0-9]+", " ", ascii_value).strip()


def find_first_columns(
    headers: Sequence[object],
    aliases: Mapping[str, Collection[str]],
    *,
    required: Collection[str] = (),
) -> dict[str, int]:
    """Devuelve la primera columna coincidente para cada campo.

    Si un encabezado o alias aparece varias veces, se conserva siempre su primera
    aparición de izquierda a derecha. Esta regla es uniforme para todos los libros.
    """
    normalized = [normalize_header(header) for header in headers]
    columns: dict[str, int] = {}
    for target, accepted in aliases.items():
        accepted_normalized = {normalize_header(alias) for alias in accepted}
        match = next(
            (index for index, header in enumerate(normalized) if header in accepted_normalized),
            None,
        )
        if match is not None:
            columns[target] = match

    missing = sorted(set(required) - columns.keys())
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(missing)}")
    return columns
