"""Bloqueo de archivo con caducidad: un worker vivo nunca pierde su lock ante otro."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable


class WorkerLockHeld(RuntimeError):
    pass


class WorkerLock:
    def __init__(self, root: Path | str, name: str, ttl: float = 300.0,
                 clock: Callable[[], float] = time.time):
        self.path = Path(root) / "locks" / f"{name}.lock"
        self.ttl = float(ttl)
        self.clock = clock
        self._held = False

    def acquire(self) -> "WorkerLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if attempt == 0 and self._is_stale():
                    self.path.unlink(missing_ok=True)
                    continue
                raise WorkerLockHeld(f"lock ocupado: {self.path.name}")
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"pid": os.getpid(), "heartbeat": self.clock()}, handle)
            self._held = True
            return self
        raise WorkerLockHeld(f"lock ocupado: {self.path.name}")

    def refresh(self) -> None:
        if self._held:
            self.path.write_text(json.dumps({"pid": os.getpid(), "heartbeat": self.clock()}), encoding="utf-8")

    def release(self) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False

    def _is_stale(self) -> bool:
        try:
            heartbeat = float(json.loads(self.path.read_text(encoding="utf-8"))["heartbeat"])
        except (OSError, ValueError, KeyError):
            return True
        return self.clock() - heartbeat > self.ttl

    def __enter__(self) -> "WorkerLock":
        return self.acquire()

    def __exit__(self, *exc_info) -> None:
        self.release()
