# Limpieza masiva Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Limpiar causalmente todas las series ingeridas y publicar una matriz única con procedencia y cobertura reconciliadas.

**Architecture:** Un contrato largo por variable conserva original, observado, contexto y limpio; reglas temporales, estadísticas, eléctricas y exógenas son funciones separadas. El stage compone reglas y publica solo después de validar esquema, finitud y cobertura.

**Tech Stack:** Python 3.11, pandas, NumPy, SciPy, PyArrow, matplotlib, pytest.

**Spec:** `specs/003-limpieza-masiva/spec.md`

## Global Constraints

- Procesar todas las series de Spec 002 y contabilizar cada destino.
- Prohibir ventanas centradas y objetivos imputados.
- No inventar límites eléctricos sin metadatos.

## Review Focus

- Serie totalmente nula queda visible y no rompe series independientes.
- Hueco inicial no usa backfill desde el futuro.
- Cambio de nivel sostenido no se recorta como pico aislado.
- Exógena publicada tarde no aparece antes de su fecha de disponibilidad.
- Columna no finita sin bandera impide publicar matriz.

---

### Task 1: Contrato canónico de variable limpia

**Files:**
- Create: `src/prodma/cleaning/contracts.py`
- Create: `src/prodma/cleaning/temporal.py`
- Test: `tests/unit/test_clean_variable_contract_v2.py`

**Interfaces:**
- Produces: `CleanVariableFrame`, `CleaningOutcome`, `build_temporal_context(observed: pd.Series, policy: CleaningPolicy) -> CleanVariableFrame`.

- [ ] **Step 1: Write failing tests** for grid completion, limited forward fill, weekly lag, initial gap, observed/imputed exclusivity and future perturbation invariance.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/unit/test_clean_variable_contract_v2.py -q` and expect failure.
- [ ] **Step 3: Implement** the long-form contract and adapt `limpieza.temporal.build_clean_variable` without duplicating rules.
- [ ] **Step 4: Run** focused tests plus existing clean-contract tests and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma/cleaning src/prodma/limpieza tests/unit/test_clean_variable_contract_v2.py && git commit -m "feat: define causal clean variable contract"`.

### Task 2: Reglas de calidad separadas

**Files:**
- Create: `src/prodma/cleaning/outliers.py`
- Create: `src/prodma/cleaning/electrical.py`
- Create: `src/prodma/cleaning/quality.py`
- Test: `tests/unit/test_cleaning_quality_rules_v2.py`

**Interfaces:**
- Consumes: `CleanVariableFrame`.
- Produces: `apply_quality_rules(frame, metadata, policy) -> CleanVariableFrame`, `QualityEvent`.

- [ ] **Step 1: Write failing tests** for isolated spike, level shift, frozen sensor, zero sequence, P/Q/S coherence, absent electrical limits and non-finite flags.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/unit/test_cleaning_quality_rules_v2.py -q` and expect failure.
- [ ] **Step 3: Implement** wrappers over migrated `limpieza.outliers` and `limpieza.electrical`, removing production duplication after parity.
- [ ] **Step 4: Run** focused and existing cleaning-rule tests and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma/cleaning src/prodma/limpieza tests/unit/test_cleaning_quality_rules_v2.py && git commit -m "feat: apply auditable cleaning rules"`.

### Task 3: Exógenas as-of y descriptivos

**Files:**
- Create: `src/prodma/cleaning/exogenous.py`
- Create: `src/prodma/cleaning/descriptive.py`
- Test: `tests/integration/test_cleaning_asof_and_descriptives.py`

**Interfaces:**
- Consumes: `ExogenousCatalog`, `CleanVariableFrame`.
- Produces: `join_exogenous_asof(...) -> pd.DataFrame`, `describe_clean_series(...) -> dict[str, object]`.

- [ ] **Step 1: Write failing tests** for publication lag, future policies, all-null exógenas, stable date range and descriptive counts matching masks.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/integration/test_cleaning_asof_and_descriptives.py -q` and expect failure.
- [ ] **Step 3: Implement** by consolidating existing `exogenas/` and `limpieza/descriptive.py` behind the new interfaces.
- [ ] **Step 4: Run** all exogenous and descriptive tests and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma/cleaning src/prodma/exogenas tests/integration/test_cleaning_asof_and_descriptives.py && git commit -m "feat: join causal exogenous features"`.

### Task 4: Stage masivo y matriz única

**Files:**
- Create: `src/prodma/cleaning/stage.py`
- Modify: `src/prodma/stages/cleaning.py`
- Test: `tests/integration/test_mass_cleaning_stage.py`

**Interfaces:**
- Consumes: `IngestionManifest`, reglas Tasks 1–3, `RunLayout`.
- Produces: `run_cleaning_v2(config, layout, ingestion) -> StageResult`, `CleanMatrixContract(path: Path, sha256: str, rows: int, series: tuple[str, ...])`.

- [ ] **Step 1: Write failing tests** for full-series reconciliation, isolated failure, non-finite blocker, atomic publish, reopen/hash/schema validation and no imputed targets.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/integration/test_mass_cleaning_stage.py -q` and expect failure.
- [ ] **Step 3: Implement** streaming/batched series processing and make the legacy stage delegate to it.
- [ ] **Step 4: Run** all cleaning tests and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma tests/integration/test_mass_cleaning_stage.py && git commit -m "feat: publish reconciled clean matrix"`.
