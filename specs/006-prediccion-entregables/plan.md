# Predicción y entregables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publicar pronósticos y entregables únicamente para el subconjunto autorizado, con horizonte configurable y semántica explícita de operación o planeamiento.

**Architecture:** Un validador construye `PredictionRequest`; motores separados producen operación y planeamiento; delivery serializa y reabre artefactos. El backend, no React, es autoridad sobre medidores y horizonte.

**Tech Stack:** Python 3.11, pandas, NumPy, PyArrow, openpyxl para exportación nueva, matplotlib, pytest.

**Spec:** `specs/006-prediccion-entregables/spec.md`

## Global Constraints

- Elegibles = autorizados ∩ limpios ∩ historia suficiente ∩ ganador válido.
- Fin inicial exclusivo `2028-01-01 00:00`, zona `America/Lima`.
- No llamar intervalo predictivo a bandas no calibradas.

## Review Focus

- Lista manipulada con medidor no autorizado se rechaza completa.
- Fecha final anterior al origen o sin alineación 15 minutos se rechaza.
- Serie sin ganador nunca aparece como cero.
- Exógena futura incompleta bloquea el modelo que la requiere.
- Entregable reabierto no contiene medidores adicionales.

---

### Task 1: Elegibilidad y contrato de solicitud

**Files:**
- Create: `src/prodma/prediction/contracts.py`
- Create: `src/prodma/prediction/eligibility.py`
- Test: `tests/unit/test_prediction_eligibility.py`

**Interfaces:**
- Consumes: `AuthorizedMeterCatalog`, `CleanMatrixContract`, `WinnerSelection`.
- Produces: `EligibleMeter`, `PredictionRequest`, `build_eligibility(...)`, `validate_prediction_request(...)`.

- [ ] **Step 1: Write failing tests** for four-way intersection, default-all selection, external injection, empty selection and horizon boundaries.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/unit/test_prediction_eligibility.py -q` and expect failure.
- [ ] **Step 3: Implement** exact set validation and reason-coded exclusions.
- [ ] **Step 4: Run** focused tests and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma/prediction tests/unit/test_prediction_eligibility.py && git commit -m "feat: validate prediction eligibility"`.

### Task 2: Motores operacional y de planeamiento

**Files:**
- Create: `src/prodma/prediction/operational.py`
- Create: `src/prodma/prediction/planning.py`
- Test: `tests/unit/test_prediction_engines.py`

**Interfaces:**
- Produces: `ForecastSeries`, `forecast_operational(...)`, `forecast_planning(...)`.

- [ ] **Step 1: Write failing tests** for exact start, continuous index, validated-horizon label, scenario separation, exogenous coverage and finite outputs.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/unit/test_prediction_engines.py -q` and expect failure.
- [ ] **Step 3: Implement** by migrating approved behavior from `recursivo.py`, `sanidad.py` and `reanudar.py` without duplicating production paths.
- [ ] **Step 4: Run** focused plus recursive/sanity tests and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma/prediction src/prodma/recursivo.py src/prodma/sanidad.py tests/unit/test_prediction_engines.py && git commit -m "feat: separate operational and planning forecasts"`.

### Task 3: Delivery y verificación de artefactos

**Files:**
- Create: `src/prodma/delivery/exports.py`
- Create: `src/prodma/delivery/verification.py`
- Test: `tests/integration/test_prediction_delivery.py`

**Interfaces:**
- Consumes: `PredictionRequest`, `ForecastSeries`.
- Produces: `DeliveryManifest`, `publish_deliverables(...) -> DeliveryManifest`, `verify_deliverables(...) -> None`.

- [ ] **Step 1: Write failing tests** for Parquet/CSV/Excel reopen, selected-meter equality, units, metadata, finite values and no unexpected sheets/rows.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/integration/test_prediction_delivery.py -q` and expect failure.
- [ ] **Step 3: Implement** atomic exports and PowerFactory tables in `05_prediccion/entregables`.
- [ ] **Step 4: Run** focused tests and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma/delivery tests/integration/test_prediction_delivery.py && git commit -m "feat: verify prediction deliverables"`.

### Task 4: Stage de predicción

**Files:**
- Create: `src/prodma/prediction/stage.py`
- Test: `tests/integration/test_prediction_stage.py`

**Interfaces:**
- Produces: `run_prediction(config, layout, request, winners) -> StageResult`.

- [ ] **Step 1: Write failing tests** for no-winner block, partial independent failure, selected subset, custom horizon and manifest completeness.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/integration/test_prediction_stage.py -q` and expect failure.
- [ ] **Step 3: Implement** stage orchestration and artifact verification before `SUCCESS`.
- [ ] **Step 4: Run** all prediction/delivery tests and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma/prediction tests/integration/test_prediction_stage.py && git commit -m "feat: orchestrate authorized predictions"`.
