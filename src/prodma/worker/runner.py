"""Ejecucion de trabajos con estado persistente y lock por trabajo."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

from prodma.worker.locks import WorkerLock


class JobStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.directory = self.root / "jobs"

    def write(self, job_id: str, status: str, error: str = "") -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"{job_id}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"job_id": job_id, "status": status, "error": error}), encoding="utf-8")
        os.replace(temporary, target)

    def read(self, job_id: str) -> dict:
        return json.loads((self.directory / f"{job_id}.json").read_text(encoding="utf-8"))

    def all(self) -> list[dict]:
        if not self.directory.is_dir():
            return []
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.directory.glob("*.json"))]


class WorkerRunner:
    def __init__(self, root: Path | str, ttl: float = 300.0, clock: Callable[[], float] = time.time):
        self.root = Path(root)
        self.ttl = ttl
        self.clock = clock
        self.jobs = JobStore(self.root)

    def run(self, job_id: str, work: Callable[[], object]):
        lock = WorkerLock(self.root, job_id, ttl=self.ttl, clock=self.clock).acquire()
        try:
            self.jobs.write(job_id, "RUNNING")
            result = work()
        except Exception as exc:
            self.jobs.write(job_id, "FAILED", error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            self.jobs.write(job_id, "SUCCESS")
            return result
        finally:
            lock.release()
