import json
from pathlib import Path

import pytest

from prodma.artifacts import ManifestStore, RunLayout, RunLock
from prodma.config import load_config


def test_load_config_resolves_paths_and_horizons(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "project_root: .\n"
        "output_root: resultados\n"
        "frequency: 15min\n"
        "timezone: America/Lima\n"
        "seed: 42\n"
        "horizons: [96, 672]\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.project_root == tmp_path.resolve()
    assert config.output_root == (tmp_path / "resultados").resolve()
    assert config.frequency == "15min"
    assert config.timezone == "America/Lima"
    assert config.horizons == (96, 672)


def test_layout_creates_exclusive_stage_directories(tmp_path):
    layout = RunLayout.create(tmp_path, "run-001")

    assert layout.stage("03_entrenamiento", "checkpoints") == (
        tmp_path / "run-001" / "03_entrenamiento" / "checkpoints"
    )
    assert layout.stage("04_evaluacion", "ranking").is_dir()


def test_unknown_stage_is_rejected(tmp_path):
    layout = RunLayout.create(tmp_path, "run-001")

    with pytest.raises(ValueError, match="Etapa desconocida"):
        layout.stage("06_otro")


def test_manifest_update_preserves_other_stages_and_leaves_no_temp_file(tmp_path):
    layout = RunLayout.create(tmp_path, "run-001")
    store = ManifestStore(layout)
    store.update_stage("01_ingesta", "SUCCESS", artifacts={"catalogo": "catalogo.json"})
    store.update_stage("02_limpieza", "RUNNING")

    payload = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    assert payload["stages"]["01_ingesta"]["status"] == "SUCCESS"
    assert payload["stages"]["02_limpieza"]["status"] == "RUNNING"
    assert not layout.manifest_path.with_suffix(".json.tmp").exists()


def test_second_lock_for_same_run_fails(tmp_path):
    first = RunLock.acquire(tmp_path / "run-001")
    try:
        with pytest.raises(RuntimeError, match="ejecucion activa"):
            RunLock.acquire(tmp_path / "run-001")
    finally:
        first.release()

    assert not (tmp_path / "run-001" / ".run.lock").exists()
