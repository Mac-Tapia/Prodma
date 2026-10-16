import re
from pathlib import Path


ROOT = Path(__file__).parents[2]
SPECS = ROOT / "specs"
PLAN_CIERRE = ROOT / "docs" / "superpowers" / "plans" / "2026-10-12-cierre-integral-exogenas-modelos-prodma.md"
PLAN_REESTRUCTURACION = ROOT / "docs" / "superpowers" / "plans" / "2026-10-10-reestructuracion-integral-prodma.md"
SPEC_011 = SPECS / "011-exogenas-clientes-planeamiento" / "spec.md"
TASKS_011 = SPECS / "011-exogenas-clientes-planeamiento" / "tasks.md"

REQ_PATTERN = re.compile(r"\*\*(S\d{3}-FR-\d{3}):\*\*")
ROW_HEADER = "Hallazgo verificado"


def _requirements_by_spec():
    found = {}
    for spec in sorted(SPECS.glob("*/spec.md")):
        found[spec.parent.name] = REQ_PATTERN.findall(spec.read_text(encoding="utf-8"))
    return found


def _all_requirement_ids():
    return [rid for ids in _requirements_by_spec().values() for rid in ids]


def _diagnostic_rows():
    """Filas de la tabla diagnóstica del plan de cierre, sin cabecera ni separador."""
    lines = PLAN_CIERRE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("## Diagnóstico que este plan debe cerrar"))
    rows = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        if line.startswith("|") and not line.startswith("|---") and ROW_HEADER not in line:
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def _trace_rows():
    """Filas de la tabla de trazabilidad diagnóstica de la Spec 011."""
    lines = SPEC_011.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("## Trazabilidad diagnóstica"))
    rows = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        if line.startswith("|") and not line.startswith("|---") and "Hallazgo" not in line:
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def test_requirement_ids_are_unique_across_all_specs():
    ids = _all_requirement_ids()
    duplicates = sorted({rid for rid in ids if ids.count(rid) > 1})
    assert duplicates == []


def test_spec_011_declares_requirements():
    assert SPEC_011.exists(), "Falta specs/011-exogenas-clientes-planeamiento/spec.md"
    assert _requirements_by_spec()["011-exogenas-clientes-planeamiento"]


def test_every_spec_011_requirement_is_planned_in_tasks():
    ids = _requirements_by_spec()["011-exogenas-clientes-planeamiento"]
    assert TASKS_011.exists(), "Falta specs/011-exogenas-clientes-planeamiento/tasks.md"
    tasks_text = TASKS_011.read_text(encoding="utf-8")
    missing = [rid for rid in ids if rid not in tasks_text]
    assert missing == []


def test_diagnostic_findings_match_between_plan_and_spec_011():
    plan_findings = [row[0] for row in _diagnostic_rows()]
    spec_findings = [row[0] for row in _trace_rows()]
    assert plan_findings, "El plan de cierre no contiene tabla diagnóstica"
    assert sorted(spec_findings) == sorted(plan_findings)


def test_every_diagnostic_row_maps_to_requirement_task_test_and_evidence():
    known = set(_all_requirement_ids())
    rows = _trace_rows()
    assert rows
    for finding, requirements, task, test, evidence in (row[:5] for row in rows):
        assert requirements, f"Sin requisito: {finding}"
        referenced = re.findall(r"S\d{3}-FR-\d{3}", requirements)
        assert referenced, f"Requisito no identificador: {finding}"
        assert set(referenced) <= known, f"Requisito inexistente en {finding}: {referenced}"
        assert re.fullmatch(r"Tarea \d{1,2}( (y|a) \d{1,2})?", task), f"Tarea inválida: {finding}"
        assert test.startswith("tests/"), f"Prueba inválida: {finding}"
        assert evidence, f"Sin evidencia: {finding}"


def test_every_requirement_of_spec_011_is_traced_to_a_diagnostic_row():
    ids = set(_requirements_by_spec()["011-exogenas-clientes-planeamiento"])
    traced = set()
    for row in _trace_rows():
        traced.update(re.findall(r"S\d{3}-FR-\d{3}", row[1]))
    assert sorted(ids - traced) == []


def test_convergence_documents_do_not_hardcode_requirement_count():
    targets = list(SPECS.glob("**/*.md")) + [PLAN_REESTRUCTURACION]
    offenders = [
        str(path.relative_to(ROOT))
        for path in targets
        if re.search(r"\b82\b", path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
