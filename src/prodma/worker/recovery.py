"""Recuperacion tras caida: trabajos RUNNING sin lock vigente pasan a INTERRUPTED."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from prodma.worker.locks import WorkerLock
from prodma.worker.runner import JobStore


def recover_state(root: Path | str, ttl: float = 300.0, clock: Callable[[], float] = time.time) -> list[str]:
    store = JobStore(root)
    changed: list[str] = []
    for job in store.all():
        if job["status"] != "RUNNING":
            continue
        probe = WorkerLock(root, job["job_id"], ttl=ttl, clock=clock)
        if probe.path.exists() and not probe._is_stale():
            continue
        probe.path.unlink(missing_ok=True)
        store.write(job["job_id"], "INTERRUPTED", error="lock vencido tras caida del worker")
        changed.append(job["job_id"])
    return changed
