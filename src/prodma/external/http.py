"""Descarga con reintentos, backoff exponencial y timeout explícito."""
from __future__ import annotations

from typing import Callable, Iterable, Mapping, Tuple

Fetcher = Callable[[str, float], Tuple[Mapping[str, str], Iterable[bytes]]]


def fetch_with_retry(
    fetcher: Fetcher,
    url: str,
    *,
    timeout: float,
    attempts_max: int,
    backoff_seconds: float,
    sleep: Callable[[float], None],
) -> tuple[Mapping[str, str], Iterable[bytes]]:
    """Reintenta solo la apertura de la respuesta; un corte a mitad de cuerpo se propaga."""
    if attempts_max < 1:
        raise ValueError("attempts_max debe ser >= 1")
    last_error: Exception | None = None
    for attempt in range(attempts_max):
        try:
            return fetcher(url, timeout)
        except Exception as exc:  # error de red u HTTP: se reintenta hasta agotar intentos
            last_error = exc
            if attempt + 1 < attempts_max:
                sleep(backoff_seconds * (2 ** attempt))
    assert last_error is not None
    raise last_error
