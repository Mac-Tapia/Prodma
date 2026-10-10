# Plan de implementación — Frontend web React

> **For agentic workers:** REQUIRED: Follow this plan task-by-task with TDD. The UI reports backend evidence; it must never fabricate or reinterpret a successful stage.

**Goal:** Proveer una consola React local para controlar, observar y descargar resultados del pipeline, incluyendo selección restringida de medidores y horizonte configurable.

**Architecture:** SPA React/TypeScript consumiendo exclusivamente la API tipada. TanStack Query administra estado remoto; el progreso llega por SSE; formularios validan localmente pero el backend conserva autoridad.

**Tech Stack:** React, TypeScript, Vite, TanStack Query, React Hook Form, Zod, Vitest, Testing Library, Playwright.

---

## Global constraints

- No duplicar reglas científicas ni catálogos dentro del navegador.
- Selección inicial: todos los medidores elegibles; el usuario solo puede deseleccionar.
- Horizonte por defecto exclusivo: `2028-01-01T00:00:00`.
- Estados, errores y artefactos deben provenir de respuestas reales del backend.
- Accesibilidad de teclado, etiquetas explícitas y contraste verificable.

## Task 1: Base React tipada y contrato API

**Files:**
- Create: `web/package.json`
- Create: `web/vite.config.ts`
- Create: `web/tsconfig.json`
- Create: `web/src/main.tsx`
- Create: `web/src/app/App.tsx`
- Create: `web/src/api/client.ts`
- Create: `web/src/api/types.ts`
- Test: `web/src/app/App.test.tsx`

**Step 1 — Write failing test:** Render inicial, proveedor de queries y manejo visible de API no disponible.

**Step 2 — Run to verify RED:** `npm --prefix web test -- --run`

**Step 3 — Implement minimally:** Crear shell, cliente tipado y configuración de pruebas.

**Step 4 — Run to verify GREEN:** `npm --prefix web test -- --run`

**Step 5 — Commit:** `git add web && git commit -m "feat: scaffold typed React pipeline console"`

## Task 2: Estado de corridas y eventos en vivo

**Files:**
- Create: `web/src/features/runs/api.ts`
- Create: `web/src/features/runs/useRunEvents.ts`
- Create: `web/src/features/runs/RunList.tsx`
- Test: `web/src/features/runs/RunList.test.tsx`
- Test: `web/src/features/runs/useRunEvents.test.ts`

**Step 1 — Write failing tests:** Listado, reconexión SSE, orden de eventos, cancelación y estado de recuperación.

**Step 2 — Run to verify RED:** `npm --prefix web test -- --run src/features/runs`

**Step 3 — Implement minimally:** Queries, mutaciones y stream con reconexión acotada.

**Step 4 — Run to verify GREEN:** `npm --prefix web test -- --run src/features/runs`

**Step 5 — Commit:** `git add web/src/features/runs && git commit -m "feat: show live pipeline runs"`

## Task 3: Fuentes, cobertura y calidad

**Files:**
- Create: `web/src/features/data/SourcesView.tsx`
- Create: `web/src/features/data/CoverageView.tsx`
- Create: `web/src/features/data/QualityView.tsx`
- Test: `web/src/features/data/DataViews.test.tsx`

**Step 1 — Write failing tests:** Mostrar inventario, archivos omitidos con motivo, cobertura por medidor y reglas de calidad fallidas.

**Step 2 — Run to verify RED:** `npm --prefix web test -- --run src/features/data`

**Step 3 — Implement minimally:** Tablas paginadas/filtrables y enlaces a evidencia del backend.

**Step 4 — Run to verify GREEN:** `npm --prefix web test -- --run src/features/data`

**Step 5 — Commit:** `git add web/src/features/data && git commit -m "feat: visualize source coverage and data quality"`

## Task 4: Selección autorizada y horizonte

**Files:**
- Create: `web/src/features/prediction/PredictionForm.tsx`
- Create: `web/src/features/prediction/MeterSelector.tsx`
- Create: `web/src/features/prediction/HorizonField.tsx`
- Test: `web/src/features/prediction/PredictionForm.test.tsx`

**Step 1 — Write failing tests:** Todos elegibles seleccionados, solo deselección, medidor inelegible no seleccionable, default 2027 completo y validación de fin exclusivo.

**Step 2 — Run to verify RED:** `npm --prefix web test -- --run src/features/prediction`

**Step 3 — Implement minimally:** Formulario accesible, resumen de exclusiones y payload tipado.

**Step 4 — Run to verify GREEN:** `npm --prefix web test -- --run src/features/prediction`

**Step 5 — Commit:** `git add web/src/features/prediction && git commit -m "feat: control authorized forecast requests"`

## Task 5: Resultados, artefactos y recorrido E2E

**Files:**
- Create: `web/src/features/results/ResultsView.tsx`
- Create: `web/src/features/results/ArtifactList.tsx`
- Create: `web/e2e/pipeline.spec.ts`
- Create: `web/playwright.config.ts`
- Test: `web/src/features/results/ResultsView.test.tsx`

**Step 1 — Write failing tests:** Resultados parciales/finales, intervalos, diagnósticos, hashes, descarga y error bloqueante.

**Step 2 — Run to verify RED:** `npm --prefix web test -- --run src/features/results`

**Step 3 — Implement minimally:** Vistas de resultados y descarga desde referencias API.

**Step 4 — Verify:** `npm --prefix web test -- --run && npm --prefix web run build && npm --prefix web run e2e`

**Step 5 — Commit:** `git add web && git commit -m "feat: complete React pipeline cockpit"`

## Review focus

- Diferencia visual entre pendiente, ejecutando, fallido, cancelado y aprobado.
- Protección frente a catálogos obsoletos y doble envío.
- Accesibilidad y rendimiento con más de cien medidores y muchos eventos.
- Ausencia de valores simulados en una corrida real.
