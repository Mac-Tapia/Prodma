from pathlib import Path

import yaml

from prodma.evidencia import validate_evidence_matrix


EVIDENCE = Path(__file__).parents[2] / "docs" / "evidencia_cientifica.yaml"


def test_validated_modules_have_multilingual_scientific_traceability():
    assert validate_evidence_matrix(EVIDENCE) == []


def test_pipeline_modules_cannot_disappear_from_evidence_matrix():
    payload = yaml.safe_load(EVIDENCE.read_text(encoding="utf-8"))
    assert set(payload["modulos"]) == {
        "01_ingesta", "02_limpieza", "03_entrenamiento", "04_evaluacion", "05_prediccion"
    }
