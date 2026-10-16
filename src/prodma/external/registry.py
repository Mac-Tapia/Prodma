"""Registro explícito de plugins por nombre de fuente (sin ramas por nombre)."""
from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class DuplicateSourceError(ValueError):
    pass


class UnsupportedSourceError(ValueError):
    pass


class SourceRegistry(Generic[T]):
    def __init__(self) -> None:
        self._entries: dict[str, T] = {}

    def register(self, name: str, entry: T) -> None:
        if name in self._entries:
            raise DuplicateSourceError(f"fuente ya registrada: {name}")
        self._entries[name] = entry

    def resolve(self, name: str) -> T:
        try:
            return self._entries[name]
        except KeyError:
            raise UnsupportedSourceError(f"fuente habilitada sin validador: {name}") from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))
