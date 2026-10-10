# Plan de implementación — Migración y convergencia integral

> **For agentic workers:** REQUIRED: Follow this plan task-by-task with TDD. Preserve existing verified behavior until parity is demonstrated; do not delete legacy paths merely because the new path compiles.

**Goal:** Migrar el proyecto actual a la arquitectura modular, retirar duplicaciones solo con paridad demostrada y cerrar cada requisito con evidencia reproducible.

**Architecture:** Una fachada CLI/orquestador conducirá las cinco etapas. Los notebooks quedarán como clientes exploratorios. La convergencia se medirá mediante pruebas de caracterización, matrices requisito-evidencia y un smoke real sobre copias controladas de los Excel.

**Tech Stack:** Python, pytest, Parquet, YAML, FastAPI/React ya implementados, scripts PowerShell.

---

## Global constraints

- No borrar el flujo heredado antes de una comparación explícita.
- Preservar entradas del usuario; generar artefactos en directorios de corrida.
- Toda desviación aceptada debe quedar explicada y aprobada.
- La aceptación real requiere lectura de Excel, etapas completas y verificación de artefactos; no basta mocks.
- `medidoralimentador.xlsx` permanece como input no versionado salvo decisión explícita del usuario.

## Task 1: Migrar CLI y configuración al orquestador

**Files:**
- Modify: `src/prodma/cli.py`
- Modify: `src/prodma/pipeline.py`
- Modify: `config/default.yaml`
- Create: `src/prodma/config/migration.py`
- Test: `tests/integration/migration/test_cli_config_migration.py`

**Step 1 — Write failing tests:** Cubrir configuración antigua, nueva, precedencia, horizonte configurable y errores incompatibles.

**Step 2 — Run to verify RED:** `pytest tests/integration/migration/test_cli_config_migration.py -q`

**Step 3 — Implement minimally:** Adaptar configuración antigua al contrato nuevo y enrutar CLI por el orquestador.

**Step 4 — Run to verify GREEN:** `pytest tests/integration/migration/test_cli_config_migration.py -q`

**Step 5 — Commit:** `git add src/prodma config/default.yaml tests/integration/migration && git commit -m "refactor: route legacy CLI through modular pipeline"`

## Task 2: Convertir notebooks en clientes reproducibles

**Files:**
- Modify: `notebooks/01_ingesta.ipynb`
- Modify: `notebooks/02_limpieza.ipynb`
- Modify: `notebooks/03_entrenamiento.ipynb`
- Create: `notebooks/04_evaluacion.ipynb`
- Create: `notebooks/05_prediccion.ipynb`
- Create: `tests/integration/migration/test_notebook_clients.py`

**Step 1 — Write failing tests:** Inspeccionar notebooks para impedir lógica productiva duplicada y exigir llamadas a API pública.

**Step 2 — Run to verify RED:** `pytest tests/integration/migration/test_notebook_clients.py -q`

**Step 3 — Implement minimally:** Reducir celdas a configuración, invocación, lectura de artefactos y visualización.

**Step 4 — Run to verify GREEN:** `pytest tests/integration/migration/test_notebook_clients.py -q`

**Step 5 — Commit:** `git add notebooks tests/integration/migration && git commit -m "refactor: make notebooks modular pipeline clients"`

## Task 3: Caracterización, paridad y retiro controlado

**Files:**
- Create: `tests/characterization/test_legacy_pipeline.py`
- Create: `tests/integration/migration/test_pipeline_parity.py`
- Modify: `src/prodma/pipeline.py`
- Modify: `src/prodma/__init__.py`

**Step 1 — Write characterization tests:** Congelar comportamientos heredados válidos y casos que se corrigen deliberadamente.

**Step 2 — Run baseline:** `pytest tests/characterization tests/integration/migration/test_pipeline_parity.py -q`

**Step 3 — Implement migration:** Eliminar duplicación únicamente donde los artefactos y contratos sean equivalentes; conservar adaptadores de compatibilidad deprecados.

**Step 4 — Run to verify GREEN:** `pytest tests/characterization tests/integration/migration/test_pipeline_parity.py -q`

**Step 5 — Commit:** `git add src/prodma tests/characterization tests/integration/migration && git commit -m "refactor: converge legacy and modular pipelines"`

## Task 4: Matriz requisito-evidencia automatizada

**Files:**
- Create: `scripts/build_compliance_matrix.py`
- Create: `docs/verification/compliance-matrix.md`
- Test: `tests/verification/test_compliance_matrix.py`

**Step 1 — Write failing tests:** Exigir que cada requisito de specs 001–010 apunte a una prueba y evidencia, sin duplicados ni estados ambiguos.

**Step 2 — Run to verify RED:** `pytest tests/verification/test_compliance_matrix.py -q`

**Step 3 — Implement minimally:** Analizar specs/metadatos de pruebas y generar matriz determinista con estado `pending|verified|blocked`.

**Step 4 — Run to verify GREEN:** `pytest tests/verification/test_compliance_matrix.py -q`

**Step 5 — Commit:** `git add scripts/build_compliance_matrix.py docs/verification tests/verification && git commit -m "test: trace requirements to verification evidence"`

## Task 5: Convergencia completa y smoke real

**Files:**
- Create: `scripts/verify.ps1`
- Create: `docs/verification/acceptance-runbook.md`
- Create: `tests/e2e/test_real_pipeline_smoke.py`
- Modify: `README.md`

**Step 1 — Write failing gate:** El script debe fallar si falta Excel requerido, GPU solicitada no se usa, un medidor no está autorizado, una etapa no publica manifiesto o React no completa el recorrido.

**Step 2 — Run focused suites:** `pytest tests/unit tests/integration tests/api tests/agent tests/verification -q`

**Step 3 — Execute full gates:** `pwsh -File scripts/verify.ps1` sobre copias controladas de los Excel y directorio de salida nuevo.

**Step 4 — Inspect evidence:** Verificar hashes, cobertura, backend efectivo GPU/CPU, ganadores, horizonte, entregables y recorrido Playwright; documentar cualquier `blocked` sin degradarlo a éxito.

**Step 5 — Commit:** `git add scripts/verify.ps1 docs/verification tests/e2e README.md && git commit -m "test: add end-to-end Prodma acceptance gate"`

## Review focus

- Regresiones ocultas por retiro prematuro del código heredado.
- Diferencia entre prueba sintética y corrida real con Excel/GPU.
- Requisitos sin prueba, evidencia sin hash o estado sobrevendido.
- Compatibilidad de notebook, CLI, API y React sobre el mismo orquestador.
