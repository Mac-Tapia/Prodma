import dataclasses

import pytest

from prodma.modeling.checkpoints import CheckpointKey, CheckpointMismatch, load_checkpoint, store_checkpoint


def _key():
    return CheckpointKey(
        "matrix-a", "exog-a", ("lag_1",), "Ridge", "1.0", "cpu", 42,
        transformations=("median_imputer", "standard_scaler"),
        hyperparameters='{"alpha":1.0}', train_end="2026-01-01T00:00:00", source_version="abc123",
    )


def test_checkpoint_rejects_changed_exogenous_hash(tmp_path):
    saved = _key()
    store_checkpoint(tmp_path, saved, {"model": "sentinel"})
    with pytest.raises(CheckpointMismatch, match="exogenous_sha256"):
        load_checkpoint(tmp_path, dataclasses.replace(saved, exogenous_sha256="exog-b"))


def test_checkpoint_rejects_changed_ordered_columns(tmp_path):
    saved = _key()
    store_checkpoint(tmp_path, saved, {"model": "sentinel"})
    with pytest.raises(CheckpointMismatch, match="columns"):
        load_checkpoint(tmp_path, dataclasses.replace(saved, columns=("lag_96", "lag_1")))


def test_checkpoint_roundtrip_is_atomic(tmp_path):
    payload = {"model": "sentinel", "score": 1.5}
    path = store_checkpoint(tmp_path, _key(), payload)
    assert load_checkpoint(tmp_path, _key()) == payload
    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp"))
