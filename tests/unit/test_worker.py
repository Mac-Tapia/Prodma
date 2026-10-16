import json

import pytest

from prodma.worker import WorkerLock, WorkerRunner, recover_state
from prodma.worker.locks import WorkerLockHeld
from prodma.worker.runner import JobStore


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def test_lock_is_exclusive_while_fresh(tmp_path):
    clock = FakeClock()
    with WorkerLock(tmp_path, "run-a", ttl=60, clock=clock):
        with pytest.raises(WorkerLockHeld):
            WorkerLock(tmp_path, "run-a", ttl=60, clock=clock).acquire()


def test_stale_lock_can_be_taken_over(tmp_path):
    clock = FakeClock()
    WorkerLock(tmp_path, "run-b", ttl=60, clock=clock).acquire()
    clock.now += 61
    lock = WorkerLock(tmp_path, "run-b", ttl=60, clock=clock)
    lock.acquire()
    lock.release()


def test_release_removes_lock_file(tmp_path):
    lock = WorkerLock(tmp_path, "run-c", ttl=60, clock=FakeClock())
    lock.acquire()
    lock.release()
    assert not (tmp_path / "locks" / "run-c.lock").exists()


def test_runner_records_success_and_releases_lock(tmp_path):
    runner = WorkerRunner(tmp_path, clock=FakeClock())
    result = runner.run("job-1", lambda: {"n": 3})
    assert result == {"n": 3}
    job = JobStore(tmp_path).read("job-1")
    assert job["status"] == "SUCCESS"
    assert not (tmp_path / "locks" / "job-1.lock").exists()


def test_runner_records_failure_with_error_and_releases_lock(tmp_path):
    runner = WorkerRunner(tmp_path, clock=FakeClock())

    def boom():
        raise RuntimeError("fallo controlado")

    with pytest.raises(RuntimeError):
        runner.run("job-2", boom)
    job = JobStore(tmp_path).read("job-2")
    assert job["status"] == "FAILED"
    assert "fallo controlado" in job["error"]
    assert not (tmp_path / "locks" / "job-2.lock").exists()


def test_runner_refuses_a_job_already_held_by_another_worker(tmp_path):
    clock = FakeClock()
    WorkerLock(tmp_path, "job-3", ttl=60, clock=clock).acquire()
    runner = WorkerRunner(tmp_path, clock=clock)
    with pytest.raises(WorkerLockHeld):
        runner.run("job-3", lambda: None)


def test_recovery_marks_running_jobs_with_stale_lock_as_interrupted(tmp_path):
    clock = FakeClock()
    store = JobStore(tmp_path)
    store.write("stale", "RUNNING")
    store.write("fresh", "RUNNING")
    store.write("done", "SUCCESS")
    WorkerLock(tmp_path, "stale", ttl=60, clock=clock).acquire()
    clock.now += 61
    WorkerLock(tmp_path, "fresh", ttl=60, clock=clock).acquire()

    changed = recover_state(tmp_path, ttl=60, clock=clock)

    assert changed == ["stale"]
    assert store.read("stale")["status"] == "INTERRUPTED"
    assert store.read("fresh")["status"] == "RUNNING"
    assert store.read("done")["status"] == "SUCCESS"


def test_job_store_write_is_atomic_json(tmp_path):
    JobStore(tmp_path).write("job-9", "SUCCESS")
    assert json.loads((tmp_path / "jobs" / "job-9.json").read_text())["status"] == "SUCCESS"
