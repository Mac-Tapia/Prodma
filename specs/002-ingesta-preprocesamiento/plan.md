# Ingesta y preprocesamiento Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convertir cada hoja soportada del inventario en datos estandarizados preservando valores y procedencia.

**Architecture:** Lectores por formato producen registros canónicos; el resolver de esquema es determinista y separado del publicador. Una reconciliación final cubre archivo, hoja, filas, columnas y series.

**Tech Stack:** Python 3.11, openpyxl/xlrd según dependencias existentes, pandas, PyArrow, pytest.

**Spec:** `specs/002-ingesta-preprocesamiento/spec.md`

## Global Constraints

- Consumir únicamente `InputInventory` aceptado.
- No imputar, recortar ni eliminar atípicos.
- Mantener `America/Lima`, frecuencia canónica `15min` y valor original.

## Review Focus

- Libro con hoja vacía o decorativa conserva un diagnóstico.
- Encabezados duplicados usan primera coincidencia y auditan alternativas.
- Duplicados temporales contradictorios no eligen valor silenciosamente.
- Fechas inválidas o fuera de tolerancia quedan rechazadas con origen.
- Cambio de hash desde Spec 001 cancela publicación.

---

### Task 1: Contratos y resolución de esquema

**Files:**
- Create: `src/prodma/ingestion/contracts.py`
- Create: `src/prodma/ingestion/schema.py`
- Test: `tests/unit/test_ingestion_schema.py`

**Interfaces:**
- Consumes: `InventoryEntry`.
- Produces: `SourceCell`, `CanonicalRecord`, `SheetAudit`, `resolve_schema(headers: Sequence[object], aliases: Mapping[str, Sequence[str]]) -> ResolvedSchema`.

- [ ] **Step 1: Write failing tests** for first-left match, missing required field, duplicate alternatives and stable source coordinates.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/unit/test_ingestion_schema.py -q` and expect missing module failure.
- [ ] **Step 3: Implement** contracts and reuse behavior from `excel_schema.find_first_columns` behind the new interface.
- [ ] **Step 4: Run** focused tests plus `tests/unit/test_excel_schema.py` and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma/ingestion tests/unit/test_ingestion_schema.py && git commit -m "feat: define canonical ingestion schema"`.

### Task 2: Lectura exhaustiva de libros

**Files:**
- Create: `src/prodma/ingestion/readers.py`
- Modify: `src/prodma/ingesta/excel.py`
- Test: `tests/integration/test_ingestion_all_sheets.py`

**Interfaces:**
- Consumes: `ResolvedSchema`, `InventoryEntry`.
- Produces: `read_workbook(entry: InventoryEntry, config: IngestionConfig) -> tuple[CanonicalRecord, ...]`, `tuple[SheetAudit, ...]`.

- [ ] **Step 1: Write failing tests** using synthetic multi-sheet books for accepted, empty, malformed, duplicate and contradictory rows.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/integration/test_ingestion_all_sheets.py -q` and expect missing reader failure.
- [ ] **Step 3: Implement** enumeration of every sheet and source-cell provenance; keep the old function as a temporary adapter.
- [ ] **Step 4: Run** focused tests plus `tests/integration/test_ingestion_xlsx.py` and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma/ingestion src/prodma/ingesta tests/integration/test_ingestion_all_sheets.py && git commit -m "feat: ingest every workbook sheet"`.

### Task 3: Publicación Parquet y cobertura

**Files:**
- Create: `src/prodma/ingestion/stage.py`
- Modify: `src/prodma/stages/ingestion.py`
- Test: `tests/integration/test_ingestion_reconciliation.py`

**Interfaces:**
- Consumes: `InputInventory`, `read_workbook`, `RunLayout`.
- Produces: `run_ingestion_v2(config, layout, inventory) -> StageResult`, `IngestionManifest` and Parquet by source/sheet/meter.

- [ ] **Step 1: Write failing tests** asserting atomic outputs, exact value preservation, complete sheet-state totals and hash revalidation.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/integration/test_ingestion_reconciliation.py -q` and expect failure.
- [ ] **Step 3: Implement** publication to `01_ingesta/` and make the existing stage delegate to `run_ingestion_v2`.
- [ ] **Step 4: Run** all ingestion unit/integration tests and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma tests/integration/test_ingestion_reconciliation.py && git commit -m "feat: reconcile ingestion coverage"`.
