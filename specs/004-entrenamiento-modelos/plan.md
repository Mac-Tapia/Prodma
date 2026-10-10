# Entrenamiento de modelos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entrenar candidatos reproducibles desde la matriz limpia, con particiones causales, backend efectivo y checkpoints incompatibles por diseño.

**Architecture:** El registro declara capacidades; un selector construye objetivos observados y particiones; adaptadores uniforman fit/predict; el stage coordina prefiltro, tuning y persistencia sin tocar el test final.

**Tech Stack:** Python 3.11, pandas, scikit-learn, Optuna y backends existentes, joblib, pytest.

**Spec:** `specs/004-entrenamiento-modelos/spec.md`

## Global Constraints

- Consumir el `CleanMatrixContract` y su hash exacto.
- Usar solo objetivos observados y test final contiguo.
- Registrar backend efectivo después de fit/predict real.

## Review Focus

- Medidor autorizado sin test observado suficiente queda excluido con causa.
- Exógena toda nula no llega al estimador.
- Timeout de un modelo no detiene candidatos independientes.
- Cambio de orden de columnas invalida checkpoint.
- Solicitud GPU con backend CPU no se etiqueta como GPU.

---

### Task 1: Selección de series y registro de candidatos

**Files:**
- Create: `src/prodma/training/contracts.py`
- Create: `src/prodma/training/selection.py`
- Modify: `src/prodma/modeling/registry.py`
- Test: `tests/unit/test_training_selection_v2.py`

**Interfaces:**
- Consumes: `CleanMatrixContract`, `AuthorizedMeterCatalog`.
- Produces: `TrainingTarget`, `ModelCapability`, `select_training_targets(matrix, catalog, horizons) -> tuple[TrainingTarget, ...]`.

- [ ] **Step 1: Write failing tests** for catalog intersection, observed-only targets, contiguous test, insufficient history and all-null exogenous removal.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/unit/test_training_selection_v2.py -q` and expect failure.
- [ ] **Step 3: Implement** contracts and adapt the existing registry to return declared capabilities.
- [ ] **Step 4: Run** focused plus registry/split tests and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma/training src/prodma/modeling tests/unit/test_training_selection_v2.py && git commit -m "feat: select auditable training targets"`.

### Task 2: Prefiltro y validación anidada

**Files:**
- Create: `src/prodma/training/prefilter.py`
- Create: `src/prodma/training/tuning.py`
- Test: `tests/unit/test_training_prefilter_v2.py`

**Interfaces:**
- Consumes: `TrainingTarget`, `ModelCapability`.
- Produces: `PrefilterResult`, `TuningResult`, `run_prefilter_v2(...)`, `tune_candidate(...)`.

- [ ] **Step 1: Write failing tests** for reference baseline, timeout, unavailable dependency, non-finite prediction and proof that final-test indices never enter tuning.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/unit/test_training_prefilter_v2.py -q` and expect failure.
- [ ] **Step 3: Implement** using adapters and temporal splits already present, with stable status vocabulary.
- [ ] **Step 4: Run** focused plus existing prefilter/nested-split tests and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma/training tests/unit/test_training_prefilter_v2.py && git commit -m "feat: isolate model prefilter and tuning"`.

### Task 3: Persistencia y backend efectivo

**Files:**
- Create: `src/prodma/training/persistence.py`
- Modify: `src/prodma/modeling/checkpoints.py`
- Modify: `src/prodma/modeling/adapters.py`
- Test: `tests/integration/test_training_persistence_v2.py`

**Interfaces:**
- Produces: `TrainingArtifact`, `EffectiveBackend`, `save_training_artifact(...)`, `load_and_verify_training_artifact(...)`.

- [ ] **Step 1: Write failing tests** for matrix/column/version/backend mismatch, atomic model write, prediction replay and requested-versus-effective backend.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/integration/test_training_persistence_v2.py -q` and expect failure.
- [ ] **Step 3: Implement** immutable checkpoint keys and post-fit backend evidence.
- [ ] **Step 4: Run** focused plus checkpoint/effective-backend tests and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma/training src/prodma/modeling tests/integration/test_training_persistence_v2.py && git commit -m "feat: verify training persistence and backend"`.

### Task 4: Stage de entrenamiento

**Files:**
- Create: `src/prodma/training/stage.py`
- Modify: `src/prodma/stages/training.py`
- Test: `tests/integration/test_training_stage_v2.py`

**Interfaces:**
- Produces: `run_training_v2(config, layout, clean_contract, catalog) -> StageResult`, `TrainingRun`.

- [ ] **Step 1: Write failing tests** for artifact isolation, per-model failure, resume, source hash mismatch and complete manifest.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/integration/test_training_stage_v2.py -q` and expect failure.
- [ ] **Step 3: Implement** stage orchestration and delegate legacy entry point.
- [ ] **Step 4: Run** all training/modeling tests and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma tests/integration/test_training_stage_v2.py && git commit -m "feat: orchestrate recoverable training"`.
