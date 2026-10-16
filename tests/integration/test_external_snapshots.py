import hashlib

import pytest

from prodma.external.contracts import ExternalSourceSpec
from prodma.external.http import fetch_with_retry
from prodma.external.snapshots import (
    SnapshotIntegrityError,
    replay_snapshot,
    store_snapshot,
)


SPEC = ExternalSourceSpec(
    name="bcrp_pbi",
    schema_version="1",
    licence="BCRP términos de uso",
    source_url="https://estadisticas.bcrp.gob.pe/",
)


def _chunks(*parts):
    return iter(parts)


def test_retry_recovers_after_transient_failures_with_backoff(tmp_path):
    attempts, sleeps = [], []

    def fetcher(url, timeout):
        attempts.append((url, timeout))
        if len(attempts) < 3:
            raise ConnectionError("corte temporal")
        return {"content-length": "11"}, _chunks(b"hello", b" world")

    headers, chunks = fetch_with_retry(
        fetcher, "https://example/serie", timeout=7, attempts_max=3,
        backoff_seconds=0.5, sleep=sleeps.append,
    )

    assert len(attempts) == 3
    assert all(timeout == 7 for _, timeout in attempts)
    assert sleeps == [0.5, 1.0]
    assert b"".join(chunks) == b"hello world"


def test_partial_download_is_never_promoted(tmp_path):
    def fetcher(url, timeout):
        def broken():
            yield b"parcial"
            raise ConnectionError("cortado a mitad")
        return {}, broken()

    with pytest.raises(ConnectionError):
        store_snapshot(SPEC, fetcher, "https://example/serie", tmp_path,
                       fecha_descarga="2026-10-15", timeout=5, attempts_max=1, sleep=lambda s: None)

    target = tmp_path / "bcrp_pbi" / "2026-10-15"
    assert not target.exists() or list(target.iterdir()) == []


def test_content_length_mismatch_is_rejected(tmp_path):
    def fetcher(url, timeout):
        return {"content-length": "100"}, iter([b"solo diez."])

    with pytest.raises(SnapshotIntegrityError, match="tamaño"):
        store_snapshot(SPEC, fetcher, "https://example/serie", tmp_path,
                       fecha_descarga="2026-10-15", timeout=5, attempts_max=1, sleep=lambda s: None)


def test_snapshot_is_stored_with_metadata_and_replays_offline(tmp_path):
    payload = b'{"periodo":"ene.2023","valor":1.5}'

    def fetcher(url, timeout):
        return {"content-length": str(len(payload))}, iter([payload])

    snap = store_snapshot(SPEC, fetcher, "https://example/serie", tmp_path,
                          fecha_descarga="2026-10-15", timeout=5, attempts_max=1, sleep=lambda s: None)

    assert snap.sha256 == hashlib.sha256(payload).hexdigest()
    assert snap.path.parent == tmp_path / "bcrp_pbi" / "2026-10-15"
    assert snap.metadata["url"] == "https://example/serie"
    assert snap.metadata["licencia"] == SPEC.licence
    assert snap.metadata["schema_version"] == "1"
    replayed = replay_snapshot(snap.path, expected_sha256=snap.sha256)
    assert replayed == payload


def test_tampered_snapshot_fails_replay(tmp_path):
    def fetcher(url, timeout):
        return {}, iter([b"original"])

    snap = store_snapshot(SPEC, fetcher, "https://example/serie", tmp_path,
                          fecha_descarga="2026-10-15", timeout=5, attempts_max=1, sleep=lambda s: None)
    snap.path.write_bytes(b"alterado")

    with pytest.raises(SnapshotIntegrityError, match="sha256"):
        replay_snapshot(snap.path, expected_sha256=snap.sha256)


def test_spec_requires_schema_and_licence_metadata():
    with pytest.raises(ValueError, match="licence"):
        ExternalSourceSpec(name="x", schema_version="1", licence="", source_url="https://a")
