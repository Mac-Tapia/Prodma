# Reestructuración integral de Prodma Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entregar Prodma como aplicación local modular, auditable y completa desde la custodia de entradas hasta React y los entregables de proyección.

**Architecture:** Monolito modular con núcleo Python y contratos de artefactos, FastAPI en loopback, worker local, SQLite para control y React/TypeScript. Las diez sub-specs se implementan en orden; ninguna fase consume datos crudos cuando existe un artefacto contractual.

**Tech Stack:** Python 3.11, pandas, PyArrow, scikit-learn y modelos existentes, FastAPI, SQLite, React, TypeScript, Vite, Vitest, Testing Library y Playwright.

**Spec:** `specs/000-reestructuracion-integral/roadmap.md`

## Global Constraints

- Cumplir `.specify/memory/constitution.md` y todos los requisitos `Snnn-FR-nnn` descubiertos en `specs/`.
- Tratar `datahistorica/`, `nuevosclientes/`, `variables_externas/` y archivos raíz declarados como solo lectura.
- Mantener Python 3.11, frecuencia `15min` y zona `America/Lima`.
- Descubrir todos los archivos actuales y futuros; no usar una lista fija de años.
- Entrenar, evaluar y predecir únicamente desde `matriz_limpia.parquet` verificada.
- Restringir proyección a `medidoralimentador.xlsx` y registrar sus hashes.
- Conservar `2028-01-01 00:00` como fin exclusivo inicial, editable desde React.
- No borrar resultados históricos ni cambios del usuario.
- Ejecutar RED-GREEN-REFACTOR, pruebas focalizadas, suite completa y smoke real proporcional.

## Review Focus

- Archivo nuevo o extensión desconocida: debe aparecer en inventario y cuarentena, nunca desaparecer.
- Excel que cambia mientras se procesa: debe invalidar la etapa sin publicar datos mezclados.
- Medidor autorizado sin datos limpios y medidor limpio no autorizado: ambos deben excluirse con causas distintas.
- Reinicio durante entrenamiento GPU: debe dejar estado interrumpido recuperable, nunca `SUCCESS`.
- Horizonte o medidor manipulado desde el cliente: el backend debe rechazarlo aunque React lo hubiera validado.

---

## File and dependency map

| Slice | Plan | Interfaces producidas |
|---|---|---|
| 001 | `specs/001-custodia-inventario/plan.md` | `InputInventory`, `AuthorizedMeterCatalog` |
| 002 | `specs/002-ingesta-preprocesamiento/plan.md` | `IngestionManifest`, Parquet estandarizado |
| 003 | `specs/003-limpieza-masiva/plan.md` | `CleanMatrixContract` |
| 004 | `specs/004-entrenamiento-modelos/plan.md` | `TrainingRun`, modelos/checkpoints |
| 005 | `specs/005-evaluacion-seleccion/plan.md` | `EvaluationReport`, `WinnerSelection` |
| 006 | `specs/006-prediccion-entregables/plan.md` | `PredictionRequest`, entregables |
| 007 | `specs/007-backend-api-worker/plan.md` | OpenAPI, eventos y repositorio de jobs |
| 008 | `specs/008-frontend-react/plan.md` | Aplicación web local |
| 009 | `specs/009-agentes-contexto-memoria/plan.md` | `AGENTS.md` y skills locales |
| 010 | `specs/010-migracion-convergencia/plan.md` | CLI final, notebook ligero y matriz de convergencia |

## Requirement-to-task coverage

| Spec requirements | Implementation tasks |
|---|---|
| S001-FR-001..004 | 001-T01, 001-T03 |
| S001-FR-005..006 | 001-T02 |
| S001-FR-007 | 001-T03 |
| S002-FR-001..005 | 002-T01, 002-T02 |
| S002-FR-006..007 | 002-T03 |
| S003-FR-001..006 | 003-T01, 003-T02 |
| S003-FR-007 | 003-T03 |
| S003-FR-008..009 | 003-T04 |
| S004-FR-001..003 | 004-T01 |
| S004-FR-004..005 | 004-T02 |
| S004-FR-006..008 | 004-T03, 004-T04 |
| S005-FR-001..002 | 005-T01 |
| S005-FR-003..004 | 005-T02 |
| S005-FR-005..007 | 005-T03 |
| S006-FR-001..003 | 006-T01 |
| S006-FR-004..007 | 006-T02, 006-T03 |
| S006-FR-008..009 | 006-T04 |
| S007-FR-001..002 | 007-T04, 007-T05 |
| S007-FR-003..006 | 007-T01, 007-T02, 007-T03 |
| S007-FR-007..009 | 007-T03, 007-T04 |
| S008-FR-001 | 008-T01 |
| S008-FR-002..004 | 008-T02, 008-T03, 008-T04 |
| S008-FR-005 | 008-T04 |
| S008-FR-006..009 | 008-T02, 008-T05 |
| S009-FR-001, S009-FR-007..008 | 009-T01 |
| S009-FR-002 | 009-T02 |
| S009-FR-003..006 | 009-T03 |
| S009-FR-009 | 009-T04 |
| S010-FR-001..005 | 010-T01, 010-T02, 010-T03 |
| S010-FR-006 | 010-T04 |
| S010-FR-007..008 | 010-T05 |

La validación de 010-T04 debe descubrir los identificadores `Snnn-FR-nnn` desde `specs/` y fallar si cualquiera queda sin prueba o evidencia.

## Execution gates

- [ ] **Gate G1:** 001–003 publican inventario, ingesta y matriz limpia reconciliados.
- [ ] **Gate G2:** 004–006 producen modelo, evaluación y proyección autorizada reproducibles.
- [ ] **Gate G3:** 007–008 controlan el mismo núcleo desde API y React sin duplicar ciencia.
- [ ] **Gate G4:** 009–010 documentan contexto, retiran duplicación y demuestran convergencia.
- [ ] **Gate G5:** suite completa, E2E y corrida real reabierta con hashes válidos.

Cada gate exige review antes de avanzar. Los commits de tareas no deben mezclar slices.
