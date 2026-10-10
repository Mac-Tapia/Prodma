# Spec 004: entrenamiento reproducible y recuperable

**Entrada:** hash aceptado de `matriz_limpia.parquet`.
**Resultado:** candidatos entrenados, checkpoints y procedencia completa.

## Requisitos funcionales

- **S004-FR-001:** formar objetivos solo con valores limpios observados; el contexto
  imputado puede usarse únicamente en rezagos/features aprobados.
- **S004-FR-002:** filtrar el universo por catálogo autorizado y suficiencia de
  historia, registrando cada exclusión.
- **S004-FR-003:** mantener registro versionado de referencias estacionales y
  modelos estadísticos/ML con resolución, exógenas y backend declarados.
- **S004-FR-004:** ejecutar prefiltro sin consultar el test final y clasificar cada
  candidato como `APPROVED`, `REJECTED`, `UNAVAILABLE`, `TIMEOUT` o `FAILED`.
- **S004-FR-005:** ajustar hiperparámetros/features solo dentro de validación
  temporal anidada o equivalente causal.
- **S004-FR-006:** registrar backend solicitado y efectivo después de un fit/predict
  real; disponibilidad de paquete o hardware no prueba uso GPU.
- **S004-FR-007:** invalidar checkpoints si cambia matriz, columnas ordenadas,
  particiones, configuración, semilla, versión o backend.
- **S004-FR-008:** separar modelos, checkpoints, figuras, CSV, tiempos, trazas y
  logs dentro de `03_entrenamiento/`.

## Aceptación

- Una prueba demuestra que el entrenamiento consumió el hash exacto de Spec 003.
- El test final es contiguo, observado y nunca alimenta selección interna.
- Recargar modelo/checkpoint reproduce predicciones dentro de tolerancia declarada.
- Un fallo de modelo no oculta ni invalida resultados independientes, pero queda
  visible en el ranking posterior.

## Trazabilidad prevista

Módulos objetivo: `prodma/training/`, migrando `modeling/`,
`forecast_models.py` y `stages/training.py`. Evidencia en `03_entrenamiento/`.
