# Spec 005: evaluación estadística y selección

**Entrada:** candidatos y particiones selladas de Spec 004.
**Resultado:** ranking versionado por medidor, variable y horizonte.

## Requisitos funcionales

- **S005-FR-001:** evaluar con origen rodante y horizontes equivalentes al uso
  declarado, preservando los mismos bloques entre modelos comparados.
- **S005-FR-002:** calcular RMSE, MAE, WAPE, MASE/RMSSE, sesgo, error de energía,
  pico y hora de pico; MAPE/R² solo cuando sean interpretables.
- **S005-FR-003:** publicar residuales y diagnósticos de autocorrelación,
  heterocedasticidad y estabilidad sin usarlos como veto automático aislado.
- **S005-FR-004:** aplicar comparaciones pareadas paramétricas o no paramétricas
  según supuestos, con corrección múltiple, tamaños de efecto e intervalos.
- **S005-FR-005:** exigir integridad, finitud, reproducibilidad y comparación contra
  referencia antes de puntuar costo o complejidad.
- **S005-FR-006:** conservar todos los candidatos y causas de exclusión; los empates
  favorecen la alternativa más simple según regla versionada.
- **S005-FR-007:** no declarar validez fuera de los horizontes efectivamente
  evaluados.

## Aceptación

- Repetir la evaluación con los mismos artefactos reproduce métricas y ranking.
- Cambiar orden de entrada no cambia el resultado.
- Cada ganador enlaza modelo, test, métricas, referencia y regla de selección.
- Una serie sin ganador queda bloqueada para publicación en Spec 006.

## Trazabilidad prevista

Módulos objetivo: `prodma/evaluation/` con backtesting, métricas, inferencia,
diagnósticos y ranking. Evidencia en `04_evaluacion/`.
