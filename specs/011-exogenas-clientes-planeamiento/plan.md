# Plan: Spec 011 exógenas, clientes y planeamiento

**Spec:** `specs/011-exogenas-clientes-planeamiento/spec.md`
**Tareas:** `specs/011-exogenas-clientes-planeamiento/tasks.md`

## Alcance de esta entrega

Cubre el contrato, la trazabilidad y la prueba de cobertura. La implementación de
conectores, matrices y modelos corresponde a las tareas T02 a T21 y no se declara
completa hasta sus propias pruebas y artefactos reabiertos.

## Módulos objetivo

- `src/prodma/external/` (registro, snapshots, conectores, calidad)
- `src/prodma/customers/` (históricos y escenarios)
- `src/prodma/exogenas/` (matrices, as-of, meteorología, clientes)
- `src/prodma/modeling/` (compatibilidad, ablación, búsqueda por familia)
- `src/prodma/evaluation/`, `src/prodma/prediction/`, `src/prodma/delivery/`
- `src/prodma/api/`, `src/prodma/worker/`, `src/prodma/control/`, `web/`

## Verificación de la entrega

- `tests/unit/test_spec_coverage.py`: unicidad de identificadores, tareas que citan
  cada requisito, paridad de hallazgos entre plan y spec, trazabilidad completa y
  ausencia de cifras fijas en convergencia.
- `tests/unit/test_scientific_evidence.py`: evidencia científica existente.

## Riesgos

- Algunos conectores dependen de disponibilidad de red; el fallo debe quedar como
  estado `BLOCKED_*`, nunca como omisión.
- La cobertura de vintages BCRP puede no existir; en ese caso se aplica un rezago
  conservador documentado.
