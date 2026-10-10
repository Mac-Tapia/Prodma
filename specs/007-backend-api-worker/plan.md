# Plan de implementación — Backend API y worker local

> **For agentic workers:** REQUIRED: Follow this plan task-by-task with TDD. Do not claim operational success from unit tests alone; preserve manifests, hashes, logs, and stage evidence.

**Goal:** Exponer el pipeline modular mediante una API FastAPI local y un worker persistente, recuperable y observable, sin acoplar la ejecución a React.

**Architecture:** FastAPI será la capa HTTP de loopback; SQLite conservará el estado de control; un worker local ejecutará una etapa por vez mediante el orquestador de aplicación. Los artefactos científicos seguirán perteneciendo a cada corrida y SQLite no sustituirá sus manifiestos.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, SQLAlchemy/SQLite, Server-Sent Events, pytest, httpx.

---

## Global constraints

- Escuchar en loopback por defecto y rechazar orígenes no configurados.
- Una corrida activa por workspace; cancelación cooperativa entre checkpoints seguros.
- No ejecutar lógica científica dentro de handlers HTTP.
- Toda transición de estado debe ser transaccional y auditable.
- Un reinicio debe recuperar corridas interrumpidas sin duplicar artefactos aprobados.

## Task 1: Modelo de control y repositorio SQLite

**Files:**
- Create: `src/prodma/control/models.py`
- Create: `src/prodma/control/repository.py`
- Create: `src/prodma/control/database.py`
- Test: `tests/unit/control/test_repository.py`

**Step 1 — Write failing tests:** Cubrir creación de corrida, transición válida, rechazo de transición inválida, checkpoints, eventos y artefactos con hash.

**Step 2 — Run to verify RED:** `pytest tests/unit/control/test_repository.py -q`

**Step 3 — Implement minimally:** Definir `Run`, `StageExecution`, `RunEvent`, `ArtifactRef` y `ControlRepository` con transacciones explícitas.

**Step 4 — Run to verify GREEN:** `pytest tests/unit/control/test_repository.py -q`

**Step 5 — Commit:** `git add src/prodma/control tests/unit/control && git commit -m "feat: add persistent pipeline control repository"`

## Task 2: Orquestador de aplicación desacoplado

**Files:**
- Create: `src/prodma/application/orchestrator.py`
- Create: `src/prodma/application/stage_registry.py`
- Test: `tests/integration/application/test_orchestrator.py`

**Step 1 — Write failing tests:** Probar orden de etapas, ejecución parcial, prerequisitos, reanudación desde checkpoint y propagación fail-closed.

**Step 2 — Run to verify RED:** `pytest tests/integration/application/test_orchestrator.py -q`

**Step 3 — Implement minimally:** Crear `PipelineOrchestrator.start(request)`, `resume(run_id)` y `cancel(run_id)` usando puertos de etapa y repositorio.

**Step 4 — Run to verify GREEN:** `pytest tests/integration/application/test_orchestrator.py -q`

**Step 5 — Commit:** `git add src/prodma/application tests/integration/application && git commit -m "feat: orchestrate modular pipeline runs"`

## Task 3: Worker, bloqueo y recuperación

**Files:**
- Create: `src/prodma/worker/runner.py`
- Create: `src/prodma/worker/locks.py`
- Create: `src/prodma/worker/recovery.py`
- Test: `tests/integration/worker/test_worker_lifecycle.py`

**Step 1 — Write failing tests:** Cubrir exclusión mutua, cancelación cooperativa, crash simulado, recuperación y no duplicación de publicación.

**Step 2 — Run to verify RED:** `pytest tests/integration/worker/test_worker_lifecycle.py -q`

**Step 3 — Implement minimally:** Implementar lock de workspace, checkpoints por etapa y recuperación determinista de estados `running` abandonados.

**Step 4 — Run to verify GREEN:** `pytest tests/integration/worker/test_worker_lifecycle.py -q`

**Step 5 — Commit:** `git add src/prodma/worker tests/integration/worker && git commit -m "feat: add recoverable local pipeline worker"`

## Task 4: API local, SSE y descarga segura

**Files:**
- Create: `src/prodma/api/app.py`
- Create: `src/prodma/api/schemas.py`
- Create: `src/prodma/api/routes/runs.py`
- Create: `src/prodma/api/routes/catalog.py`
- Create: `src/prodma/api/routes/artifacts.py`
- Create: `src/prodma/api/routes/events.py`
- Test: `tests/api/test_runs_api.py`
- Test: `tests/api/test_artifacts_api.py`

**Step 1 — Write failing tests:** Probar crear/listar/consultar/cancelar corridas, catálogo autorizado, eventos SSE, validación de horizonte y protección contra path traversal.

**Step 2 — Run to verify RED:** `pytest tests/api -q`

**Step 3 — Implement minimally:** Exponer contratos tipados; resolver artefactos solamente desde referencias registradas y verificadas.

**Step 4 — Run to verify GREEN:** `pytest tests/api -q`

**Step 5 — Commit:** `git add src/prodma/api tests/api && git commit -m "feat: expose local pipeline control API"`

## Task 5: Arranque y configuración operativa

**Files:**
- Modify: `src/prodma/cli.py`
- Modify: `config/default.yaml`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Test: `tests/integration/api/test_local_startup.py`

**Step 1 — Write failing test:** Validar que `prodma serve` inicializa DB, worker y API en loopback con directorios configurables.

**Step 2 — Run to verify RED:** `pytest tests/integration/api/test_local_startup.py -q`

**Step 3 — Implement minimally:** Agregar comando `serve`, lifecycle de recursos y cierre limpio.

**Step 4 — Verify:** `pytest tests/unit/control tests/integration/application tests/integration/worker tests/api tests/integration/api -q`

**Step 5 — Commit:** `git add src/prodma config/default.yaml pyproject.toml requirements.txt tests && git commit -m "feat: add local pipeline service entrypoint"`

## Review focus

- Estados imposibles o carreras entre API y worker.
- Cancelación durante escritura y publicación atómica.
- Lectura arbitraria de archivos mediante endpoints.
- Recuperación después de matar el proceso.
