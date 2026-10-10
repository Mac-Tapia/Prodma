# Evaluación y selección Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluar candidatos sobre bloques temporales comparables y elegir un ganador trazable por medidor, variable y horizonte.

**Architecture:** Backtesting produce observaciones pareadas; métricas y diagnósticos no deciden; un ranking versionado aplica gates y desempate por simplicidad. La etapa conserva todos los candidatos y exclusiones.

**Tech Stack:** Python 3.11, pandas, NumPy, SciPy, statsmodels, pytest.

**Spec:** `specs/005-evaluacion-seleccion/spec.md`

## Global Constraints

- Evaluar horizontes declarados con bloques idénticos entre candidatos.
- Comparar siempre con referencias simples.
- No extrapolar validez a horizontes no evaluados.

## Review Focus

- Ceros o escalas pequeñas no producen MAPE engañoso publicado.
- Modelos con bloques distintos no entran en prueba pareada.
- Muestra insuficiente impide inferencia, no borra métricas descriptivas.
- Predicción no finita excluye candidato con causa.
- Empate reproduce siempre el mismo ganador simple.

---

### Task 1: Backtesting y métricas

**Files:**
- Create: `src/prodma/evaluation/contracts.py`
- Create: `src/prodma/evaluation/backtesting.py`
- Create: `src/prodma/evaluation/metrics.py`
- Test: `tests/unit/test_evaluation_metrics.py`

**Interfaces:**
- Consumes: `TrainingRun`.
- Produces: `ForecastBlock`, `MetricSet`, `evaluate_blocks(actual, predicted, baseline) -> MetricSet`.

- [ ] **Step 1: Write failing tests** for exact metrics, zero denominator, peak timing, energy error, identical block IDs and unvalidated horizon.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/unit/test_evaluation_metrics.py -q` and expect failure.
- [ ] **Step 3: Implement** contracts and migrate `evaluacion/backtesting.py` behind the new namespace.
- [ ] **Step 4: Run** focused plus existing backtesting tests and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma/evaluation src/prodma/evaluacion tests/unit/test_evaluation_metrics.py && git commit -m "feat: evaluate temporal forecast blocks"`.

### Task 2: Diagnóstico e inferencia

**Files:**
- Create: `src/prodma/evaluation/diagnostics.py`
- Create: `src/prodma/evaluation/inference.py`
- Test: `tests/unit/test_evaluation_inference.py`

**Interfaces:**
- Produces: `DiagnosticReport`, `ComparisonResult`, `compare_candidates(block_errors, alpha) -> tuple[ComparisonResult, ...]`.

- [ ] **Step 1: Write failing tests** for paired alignment, Wilcoxon/Friedman routing, multiplicity correction, effect size and insufficient-sample status.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/unit/test_evaluation_inference.py -q` and expect failure.
- [ ] **Step 3: Implement** diagnostics as evidence fields, never as isolated vetoes.
- [ ] **Step 4: Run** focused tests and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma/evaluation tests/unit/test_evaluation_inference.py && git commit -m "feat: add paired forecast inference"`.

### Task 3: Ranking y stage

**Files:**
- Create: `src/prodma/evaluation/ranking.py`
- Create: `src/prodma/evaluation/stage.py`
- Test: `tests/integration/test_evaluation_stage.py`

**Interfaces:**
- Produces: `WinnerSelection`, `rank_candidates(metrics, diagnostics, policy) -> tuple[WinnerSelection, ...]`, `run_evaluation(...) -> StageResult`.

- [ ] **Step 1: Write failing tests** for gates, exclusions retained, deterministic simplicity tie-break, reproducible ranking and no-winner series.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/integration/test_evaluation_stage.py -q` and expect failure.
- [ ] **Step 3: Implement** ranking and isolated `04_evaluacion/` publication.
- [ ] **Step 4: Run** all evaluation tests and expect pass.
- [ ] **Step 5: Commit** `git add src/prodma/evaluation tests/integration/test_evaluation_stage.py && git commit -m "feat: publish traceable model ranking"`.
