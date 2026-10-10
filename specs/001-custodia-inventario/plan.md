# Custodia e inventario Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inventariar y sellar todas las entradas, incluidas expectativas ausentes, y publicar el catálogo autorizado de medidores.

**Architecture:** Modelos inmutables representan archivos y estados; descubrimiento y catálogo no interpretan mediciones. La etapa publica JSONL/JSON/Parquet atómicos y revalida hashes antes de finalizar.

**Tech Stack:** Python 3.11, pathlib, hashlib, openpyxl en lectura, PyArrow, pytest.

**Spec:** `specs/001-custodia-inventario/spec.md`

## Global Constraints

- Originales de solo lectura; rutas almacenadas relativas a la raíz.
- Normalizar espacios/case en `Medidor`, pero conservar guiones y caracteres identificadores.
- Contabilizar archivos descubiertos y rutas obligatorias ausentes.

## Review Focus

- Extensión desconocida se cataloga como `QUARANTINED`.
- Ruta configurada ausente se cataloga como `MISSING_EXPECTED`.
- Duplicado de medidor idéntico conserva todas sus filas; conflicto bloquea.
- Archivo mutado entre descubrimiento y cierre falla.
- Ruta fuera de raíces autorizadas no ingresa al inventario.

---

### Task 1: Modelos e inventario recursivo

**Files:**
- Create: `src/prodma/custody/models.py`
- Create: `src/prodma/custody/inventory.py`
- Create: `src/prodma/custody/__init__.py`
- Test: `tests/unit/test_input_inventory.py`

**Interfaces:**
- Produces: `InventoryEntry`, `InventoryState`, `discover_inputs(project_root: Path, roots: tuple[Path, ...], expected: tuple[Path, ...]) -> tuple[InventoryEntry, ...]`, `verify_unchanged(entries) -> None`.

- [ ] **Step 1: Write failing tests** `test_discovers_nested_future_files`, `test_unknown_extension_is_quarantined`, `test_missing_expected_is_recorded`, `test_outside_root_is_rejected`, and `test_changed_bytes_fail_verification`.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/unit/test_input_inventory.py -q` and expect failures for missing module.
- [ ] **Step 3: Implement** the interfaces above with deterministic relative-path ordering and streaming SHA-256.
- [ ] **Step 4: Run** the same command and expect all tests to pass.
- [ ] **Step 5: Commit** `git add src/prodma/custody tests/unit/test_input_inventory.py && git commit -m "feat: inventory all input files"`.

### Task 2: Catálogo autorizado de medidores

**Files:**
- Create: `src/prodma/catalog/authorized_meters.py`
- Create: `src/prodma/catalog/__init__.py`
- Test: `tests/unit/test_authorized_meter_catalog.py`

**Interfaces:**
- Consumes: `InventoryEntry` from Task 1.
- Produces: `AuthorizedMeter`, `AuthorizedMeterCatalog`, `read_authorized_meters(path: Path, sheet: str = "alimentador_medidor", header: str = "Medidor") -> AuthorizedMeterCatalog`.

- [ ] **Step 1: Write failing tests** for row-2 header detection, 112-row/110-unique current workbook characterization, exact duplicate provenance, conflicting duplicates, missing sheet/header and preservation of `L-6605` punctuation.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/unit/test_authorized_meter_catalog.py -q` and expect missing API failures.
- [ ] **Step 3: Implement** read-only parsing and exact normalized identity; no fuzzy aliasing.
- [ ] **Step 4: Run** the focused tests and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma/catalog tests/unit/test_authorized_meter_catalog.py && git commit -m "feat: enforce authorized meter catalog"`.

### Task 3: Publicación y reconciliación de custodia

**Files:**
- Create: `src/prodma/custody/stage.py`
- Modify: `src/prodma/artifacts.py`
- Modify: `src/prodma/config.py`
- Test: `tests/integration/test_custody_stage.py`

**Interfaces:**
- Consumes: `discover_inputs`, `verify_unchanged`, `read_authorized_meters`, `RunLayout`.
- Produces: `run_custody(config: PipelineConfig, layout: RunLayout) -> StageResult`; artefactos `input_inventory.jsonl`, `input_summary.json`, `authorized_meters.parquet`.

- [ ] **Step 1: Write failing integration tests** asserting atomic publication, state totals, hashes, expected-missing behavior and no writes to source mtimes/hashes.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/integration/test_custody_stage.py -q` and expect missing stage failure.
- [ ] **Step 3: Implement** the stage and add a `00_custodia` layout without changing existing stage names 01–05.
- [ ] **Step 4: Run** focused tests plus `tests/unit/test_config_artifacts.py` and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma tests/integration/test_custody_stage.py && git commit -m "feat: publish custody evidence"`.
